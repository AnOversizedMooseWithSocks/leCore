"""PROBE (measure, do not ship): does sampling soft shadows in the FULL multidimensional space -- image x light --
and sharing those samples across pixels beat plain per-pixel Monte-Carlo, at an equal ray budget? And how does the
win depend on occluder frequency (one big blocker vs many small ones)? This is Hachisuka et al. 2008
(multidimensional adaptive sampling) tested on the engine's own `ndfield.sparse_reconstruct` pattern -- probe
sparse, interpolate with a kernel, refine where uncertain -- which was written for mazes/fields, never rendering.

The soft shadow at a floor point p is the average over the area light of the VISIBILITY v(p, L): the fraction of
the light that p can see. The integrand v(p, L) over the 4D space (p is 2D on the floor, L is 2D on the light) is
NOISE-FREE per sample (one deterministic occlusion test), which is the whole point -- we sample the clean integrand
and only collapse to the image at the very end, instead of averaging noisy per-pixel results.

Three methods, matched on total visibility tests (the cost):
  UNIFORM   : N random light points per pixel, average           (standard stratified soft-shadow MC -- the baseline)
  SHARED    : B random samples in the whole 4D box, kernel-reconstruct, integrate out the light dims
              (tests pure sample-SHARING: a sample informs nearby pixels too)
  ADAPTIVE  : seed B/2, then add B/2 where NEARBY samples DISAGREE most (an oracle-free uncertainty signal, the same
              detector the landmark probe validated), then reconstruct + integrate  (tests refine-where-tricky)

Reported per scene: the error of each method vs a dense ground truth, at equal budget. The claim under test is that
SHARED/ADAPTIVE beat UNIFORM when the shadow structure is low-frequency (one big blocker) and lose their edge when
it is high-frequency (many small blockers) -- the honest, scene-dependent breakeven.
"""
import time
import numpy as np

# --- geometry: a floor, a square area light above it, and one or more flat square occluders between them ----------
FLOOR_EXTENT = 1.5           # floor points span [-1.5, 1.5] in x and z
LIGHT_CENTER = np.array([0.0, 3.0, 0.0])
LIGHT_HALF = 0.8             # the area light is 1.6 x 1.6
OCC_Y = 1.2                  # occluder height (between floor at y=0 and light at y=3)


def make_occluders(kind):
    # both scenes cover roughly the same footprint, so the AVERAGE shadow is similar -- what differs is the spatial
    # FREQUENCY of the shadow (its bandwidth), which is exactly what should decide whether the adaptive method wins.
    if kind == "one_big":
        return [(0.0, 0.0, 0.7)]                                   # (cx, cz, half) -- a single big blocker
    if kind == "many_small":
        g = np.linspace(-0.6, 0.6, 5)
        return [(cx, cz, 0.11) for cx in g for cz in g]           # 25 little blockers over the same area
    raise ValueError(kind)


def visibility(coords, occ):
    """The oracle: for each row (px, pz, u, v) return v in {0,1} -- can the floor point (px,0,pz) see the light point
    (u,v) on the panel, or does an occluder block the straight segment between them? One occlusion test per row,
    fully vectorised. This is the noise-free integrand we sample."""
    coords = np.atleast_2d(np.asarray(coords, float))
    px, pz, u, v = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    lx = LIGHT_CENTER[0] + u; lz = LIGHT_CENTER[2] + v            # world light point
    t = OCC_Y / LIGHT_CENTER[1]                                    # where the segment crosses the occluder plane
    cx = px + t * (lx - px)                                        # crossing point in the occluder plane
    cz = pz + t * (lz - pz)
    vis = np.ones(len(coords))
    for (ox, oz, oh) in occ:                                       # blocked if the crossing lands on any occluder card
        vis[(np.abs(cx - ox) < oh) & (np.abs(cz - oz) < oh)] = 0.0
    return vis


def ground_truth(occ, nimg, nlight):
    # dense reference soft shadow: every pixel, averaged over a fine grid of light points
    g = (np.arange(nimg := nimg if False else nimg) + 0.5) / nimg * (2 * FLOOR_EXTENT) - FLOOR_EXTENT
    PX, PZ = np.meshgrid(g, g, indexing="ij")
    lg = (np.arange(nlight) + 0.5) / nlight * (2 * LIGHT_HALF) - LIGHT_HALF
    U, V = np.meshgrid(lg, lg, indexing="ij")
    shadow = np.zeros((nimg, nimg))
    for i in range(nimg):
        for j in range(nimg):
            coords = np.stack([np.full(U.size, PX[i, j]), np.full(U.size, PZ[i, j]), U.ravel(), V.ravel()], axis=1)
            shadow[i, j] = visibility(coords, occ).mean()
    return shadow


# --- the kernel reconstruction (Nadaraya-Watson), mirroring holographic_ndfield's interpolator, blocked for memory -
def nw_reconstruct(query, pts, vals, bw):
    out = np.empty(len(query))
    for s in range(0, len(query), 1024):
        q = query[s:s + 1024]
        d2 = ((q[:, None, :] - pts[None, :, :]) ** 2).sum(2)       # (b, K) squared distance in the 4D box
        w = np.exp(-d2 / (2 * bw * bw))
        wsum = w.sum(1)
        out[s:s + 1024] = np.where(wsum > 1e-12, (w * vals[None, :]).sum(1) / np.maximum(wsum, 1e-12), 0.0)
    return out


def integrate_shadow(pts, vals, occ, nimg, nlight_int, bw):
    # reconstruct the 4D visibility field, then integrate out the light dims to get the per-pixel soft shadow
    g = (np.arange(nimg) + 0.5) / nimg * (2 * FLOOR_EXTENT) - FLOOR_EXTENT
    lg = (np.arange(nlight_int) + 0.5) / nlight_int * (2 * LIGHT_HALF) - LIGHT_HALF
    PX, PZ = np.meshgrid(g, g, indexing="ij")
    U, V = np.meshgrid(lg, lg, indexing="ij")
    shadow = np.zeros((nimg, nimg))
    # build all query points (pixel x light-int-grid) and reconstruct in one blocked pass
    for i in range(nimg):
        q = np.stack([np.repeat(PX[i], U.size), np.repeat(PZ[i], U.size),
                      np.tile(U.ravel(), nimg), np.tile(V.ravel(), nimg)], axis=1)
        rec = nw_reconstruct(q, pts, vals, bw).reshape(nimg, U.size)
        shadow[i] = rec.mean(1)                                    # integrate (average) over the light points
    return shadow


def box_sample(n, rng):
    # a sample in the 4D box: (px, pz) over the floor, (u, v) over the light
    x = rng.uniform(-FLOOR_EXTENT, FLOOR_EXTENT, (n, 2))
    l = rng.uniform(-LIGHT_HALF, LIGHT_HALF, (n, 2))
    return np.concatenate([x, l], axis=1)


def run_scene(kind, budget, nimg=24, nlight_int=6, seed=0):
    occ = make_occluders(kind)
    rng = np.random.default_rng(seed)
    gt = ground_truth(occ, nimg, nlight=16)
    bw = 0.16 * (2 * FLOOR_EXTENT)                                 # kernel bandwidth in box units (shared by both)

    def err(shadow):
        return float(np.abs(shadow - gt).mean())

    # ---- UNIFORM: N random light points per pixel; cost = pixels * N tests ----
    N = max(1, budget // (nimg * nimg))
    g = (np.arange(nimg) + 0.5) / nimg * (2 * FLOOR_EXTENT) - FLOOR_EXTENT
    PX, PZ = np.meshgrid(g, g, indexing="ij")
    uni = np.zeros((nimg, nimg)); calls_u = 0
    for i in range(nimg):
        for j in range(nimg):
            l = rng.uniform(-LIGHT_HALF, LIGHT_HALF, (N, 2))
            coords = np.stack([np.full(N, PX[i, j]), np.full(N, PZ[i, j]), l[:, 0], l[:, 1]], axis=1)
            uni[i, j] = visibility(coords, occ).mean(); calls_u += N

    # ---- SHARED: B random samples in the whole 4D box, reconstruct, integrate ----
    B = calls_u                                                    # match the uniform ray budget exactly
    pts = box_sample(B, rng); vals = visibility(pts, occ)
    sh_shared = integrate_shadow(pts, vals, occ, nimg, nlight_int, bw)

    # ---- ADAPTIVE: seed B/2, then add B/2 where NEARBY samples disagree (oracle-free), reconstruct, integrate ----
    half = B // 2
    pts_a = box_sample(half, rng); vals_a = visibility(pts_a, occ)
    # score a dense candidate set by how much its nearest samples DISAGREE (variance) -- high == likely a shadow edge.
    cand = box_sample(4 * half, rng)
    d2 = ((cand[:, None, :] - pts_a[None, :, :]) ** 2).sum(2)
    w = np.exp(-d2 / (2 * bw * bw))
    wsum = np.maximum(w.sum(1), 1e-12)
    mean = (w * vals_a[None, :]).sum(1) / wsum
    var = (w * (vals_a[None, :] - mean[:, None]) ** 2).sum(1) / wsum   # weighted local disagreement, NO oracle
    take = np.argsort(-var)[: B - half]                           # spend the rest of the budget on the tricky spots
    add = cand[take]; vadd = visibility(add, occ)                 # (these ARE part of the budget: real tests)
    pts_a = np.vstack([pts_a, add]); vals_a = np.concatenate([vals_a, vadd])
    sh_adapt = integrate_shadow(pts_a, vals_a, occ, nimg, nlight_int, bw)

    return {"uniform": (calls_u, err(uni)), "shared": (B, err(sh_shared)), "adaptive": (len(vals_a), err(sh_adapt))}


def main():
    budget = 24 * 24 * 12                                          # ~6900 visibility tests, same for every method
    print(f"budget: {budget} visibility tests per method   (image 24x24, light integrated on 6x6)\n")
    print(f"{'scene':>12} | {'method':>9} {'tests':>6} {'meanErr':>9}")
    for kind in ("one_big", "many_small"):
        t = time.time()
        res = run_scene(kind, budget)
        for method in ("uniform", "shared", "adaptive"):
            calls, e = res[method]
            tag = "  <- baseline" if method == "uniform" else ""
            print(f"{kind:>12} | {method:>9} {calls:>6} {e:>9.4f}{tag}")
        print(f"{'':>12}   ({time.time() - t:.0f}s)\n")
    print("read: meanErr is the average per-pixel soft-shadow error vs a dense ground truth, at EQUAL ray budget.")
    print("shared/adaptive beating uniform means sharing samples across the 4D space pays; the gap should shrink or")
    print("reverse on many_small, where the shadow is high-frequency and there is little coherence to exploit.")


if __name__ == "__main__":
    main()

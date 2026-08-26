"""PROBE 2 (measure, do not ship): the last experiment showed sample-sharing is a VARIANCE tool -- it beats per-pixel
MC where the soft shadow is noisy-but-smooth and loses where it is sharp, and an isotropic kernel smears the very
edges the adaptive samples were spent on. Two fixes, both from parts already in the engine, plus Moose's mode-toggle:

  ANISO  : reconstruct with a STEERED kernel -- narrow ACROSS shadow edges, wide ALONG them -- using the structure
           tensor (the same gradient-outer-product `holographic_vision.harris` builds). This is Hachisuka 2008's
           anisotropic reconstruction. The claim: it should rescue the ONE-BIG (sharp) case the isotropic kernel lost.
  ROUTER : precompute WHICH mode each pixel needs. A cheap shared pre-pass gives a coarse shadow; its edge magnitude
           map flags the SHARP pixels; we then spend the rest of the ray budget on per-pixel MC THERE (unbiased at the
           edge) and keep the cheap shared estimate on the SMOOTH pixels. This is "precompute which thing we need,"
           one ray budget, routed by region.

Tested on three scenes -- one_big (all sharp), many_small (all smooth), and mixed (a big blocker AND small ones, so
a single global mode is wrong and routing should matter). All methods share one visibility-test budget. We keep
whatever the numbers say, win or lose.
"""
import time
import numpy as np

# reuse the scene oracle + constants from the first probe instead of re-deriving them
from tools.multidim_shadow_probe import visibility, ground_truth, box_sample, FLOOR_EXTENT, LIGHT_HALF
from holographic.misc.holographic_vision import gradient as vision_gradient          # edge magnitude map for the router  reuse

NIMG = 24


def make_occluders(kind):
    if kind == "one_big":
        return [(0.0, 0.0, 0.7)]
    if kind == "many_small":
        g = np.linspace(-0.6, 0.6, 5)
        return [(cx, cz, 0.11) for cx in g for cz in g]
    if kind == "mixed":                                             # a big blocker on one side, small ones on the other
        occ = [(-0.6, 0.0, 0.55)]
        g = np.linspace(0.2, 1.0, 3)
        return occ + [(cx, cz, 0.10) for cx in g for cz in np.linspace(-0.7, 0.7, 4)]
    raise ValueError(kind)


def pixel_centers(nimg):
    g = (np.arange(nimg) + 0.5) / nimg * (2 * FLOOR_EXTENT) - FLOOR_EXTENT
    PX, PZ = np.meshgrid(g, g, indexing="ij")
    return PX, PZ, g


# --- soft shadow from scattered 4D samples: a spatial (px,pz) kernel-weighted average of sample visibilities. The
#     light dims (u,v) are the integration variable, so we DON'T weight by them -- gathering samples with varied
#     (u,v) near a floor point IS the light integral. isotropic or steered by a per-pixel 2x2 precision metric. -----
def reconstruct_isotropic(samples, vis, nimg, sigma):
    PX, PZ, _ = pixel_centers(nimg)
    sx, sz = samples[:, 0], samples[:, 1]
    shadow = np.zeros((nimg, nimg))
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    for i in range(nimg):                                           # row at a time to bound memory
        dx = PX[i][:, None] - sx[None, :]                          # (nimg, B)
        dz = PZ[i][:, None] - sz[None, :]
        w = np.exp(-(dx * dx + dz * dz) * inv2s2)
        wsum = np.maximum(w.sum(1), 1e-12)
        shadow[i] = (w * vis[None, :]).sum(1) / wsum
    return shadow


def structure_tensor(S0):
    """Per-pixel edge orientation + anisotropy from the gradient outer product (locally averaged) -- the same
    structure tensor holographic_vision.harris uses for corners. Returns across-edge unit vectors (e1) and an
    anisotropy in [0,1] (1 == a clean straight edge, 0 == flat/isotropic)."""
    gpx, gpz = np.gradient(S0)                                      # gradients along the px and pz axes (unambiguous)
    def box(a):                                                     # local averaging (the tensor's window)
        p = np.pad(a, 1, mode="edge")
        return sum(p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)) / 9.0
    Jxx, Jzz, Jxz = box(gpx * gpx), box(gpz * gpz), box(gpx * gpz)
    tmp = np.sqrt(np.maximum(((Jxx - Jzz) * 0.5) ** 2 + Jxz * Jxz, 0.0))
    l1 = (Jxx + Jzz) * 0.5 + tmp                                    # larger eigenvalue (across the edge)
    l2 = (Jxx + Jzz) * 0.5 - tmp
    theta = 0.5 * np.arctan2(2.0 * Jxz, Jxx - Jzz)                 # orientation of the across-edge eigenvector
    e1 = np.stack([np.cos(theta), np.sin(theta)], axis=-1)         # (nimg, nimg, 2) across-edge direction
    aniso = (l1 - l2) / (l1 + l2 + 1e-9)
    return e1, aniso


def reconstruct_anisotropic(samples, vis, nimg, sigma):
    # pre-pass: a cheap isotropic reconstruction, just to find WHERE the edges are and which way they point
    S0 = reconstruct_isotropic(samples, vis, nimg, sigma)
    e1, aniso = structure_tensor(S0)
    PX, PZ, _ = pixel_centers(nimg)
    sx, sz = samples[:, 0], samples[:, 1]
    shadow = np.zeros((nimg, nimg))
    kA, kB, min_frac = 0.75, 0.8, 0.35                            # narrow across (down to 35% sigma), widen along
    for i in range(nimg):
        for j in range(nimg):
            a = float(aniso[i, j])
            c, s = float(e1[i, j, 0]), float(e1[i, j, 1])          # across-edge unit vector
            s_across = sigma * max(min_frac, 1.0 - kA * a)         # tight across a strong edge -> preserves the penumbra
            s_along = sigma * (1.0 + kB * a)                       # loose along it -> pools samples, kills noise
            # 2x2 precision metric M = e1 e1^T / s_across^2 + e2 e2^T / s_along^2  (e2 perpendicular to e1)
            m11 = (c * c) / s_across ** 2 + (s * s) / s_along ** 2
            m22 = (s * s) / s_across ** 2 + (c * c) / s_along ** 2
            m12 = (c * s) * (1.0 / s_across ** 2 - 1.0 / s_along ** 2)
            dx = PX[i, j] - sx; dz = PZ[i, j] - sz                 # (B,) offsets to every sample
            q = m11 * dx * dx + 2.0 * m12 * dx * dz + m22 * dz * dz  # Mahalanobis distance^2
            w = np.exp(-0.5 * q)
            shadow[i, j] = (w * vis).sum() / max(w.sum(), 1e-12)
    return shadow


def route_and_render(occ, nimg, budget, sigma, seed=0):
    """The mode-router: a cheap shared pre-pass, then spend the rest of the budget on per-pixel MC only at the SHARP
    pixels the edge map flags, keeping the cheap estimate on the smooth ones. Precomputed routing, one budget."""
    rng = np.random.default_rng(seed)
    B_pre = int(budget * 0.4)                                      # cheap coarse pass to locate the edges
    pre = box_sample(B_pre, rng); pre_v = visibility(pre, occ)
    S0 = reconstruct_isotropic(pre, pre_v, nimg, sigma)
    mag, _ = vision_gradient(S0)                                   # edge strength per pixel (reuse the engine's Sobel)
    thresh = np.percentile(mag, 82.0)                              # the top ~18% strongest-edge pixels are "sharp"
    sharp = mag >= thresh
    n_sharp = int(sharp.sum())
    B_remain = budget - B_pre
    N = max(1, B_remain // max(n_sharp, 1))                        # per-sharp-pixel MC samples
    PX, PZ, _ = pixel_centers(nimg)
    out = S0.copy(); tests = B_pre
    idx = np.argwhere(sharp)
    for (i, j) in idx:                                             # unbiased per-pixel MC exactly where it's needed
        l = rng.uniform(-LIGHT_HALF, LIGHT_HALF, (N, 2))
        coords = np.stack([np.full(N, PX[i, j]), np.full(N, PZ[i, j]), l[:, 0], l[:, 1]], axis=1)
        out[i, j] = visibility(coords, occ).mean(); tests += N
    return out, tests, n_sharp


def uniform(occ, nimg, budget, seed=0):
    rng = np.random.default_rng(seed)
    N = max(1, budget // (nimg * nimg))
    PX, PZ, _ = pixel_centers(nimg)
    out = np.zeros((nimg, nimg))
    for i in range(nimg):
        for j in range(nimg):
            l = rng.uniform(-LIGHT_HALF, LIGHT_HALF, (N, 2))
            coords = np.stack([np.full(N, PX[i, j]), np.full(N, PZ[i, j]), l[:, 0], l[:, 1]], axis=1)
            out[i, j] = visibility(coords, occ).mean()
    return out, nimg * nimg * N


def main():
    budget = NIMG * NIMG * 12
    sigma = 0.18
    print(f"budget: {budget} visibility tests per method   (image {NIMG}x{NIMG}, kernel sigma {sigma})\n")
    print(f"{'scene':>12} | {'method':>9} {'tests':>6} {'meanErr':>9}   notes")
    for kind in ("one_big", "many_small", "mixed"):
        occ = make_occluders(kind)
        gt = ground_truth(occ, NIMG, nlight=16)
        def e(s):
            return float(np.abs(s - gt).mean())
        t = time.time()
        u, cu = uniform(occ, NIMG, budget)
        pts = box_sample(budget, np.random.default_rng(1)); vis = visibility(pts, occ)
        sh = reconstruct_isotropic(pts, vis, NIMG, sigma)
        an = reconstruct_anisotropic(pts, vis, NIMG, sigma)
        ro, tr, nsharp = route_and_render(occ, NIMG, budget, sigma)
        best = min([("uniform", e(u)), ("shared", e(sh)), ("aniso", e(an)), ("router", e(ro))], key=lambda x: x[1])[0]
        for name, s, tests in (("uniform", u, cu), ("shared", sh, budget), ("aniso", an, budget), ("router", ro, tr)):
            star = "  <- best" if name == best else ""
            extra = f"  ({nsharp} sharp px routed to MC)" if name == "router" else ""
            print(f"{kind:>12} | {name:>9} {tests:>6} {e(s):>9.4f}{star}{extra}")
        print(f"{'':>12}   ({time.time() - t:.0f}s)\n")
    print("read: meanErr vs dense ground truth at equal budget. aniso should rescue the sharp (one_big) case the")
    print("isotropic kernel lost; router should win or tie on 'mixed', where one global mode is wrong.")


if __name__ == "__main__":
    main()

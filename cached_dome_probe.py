"""BUILD + MEASURE: the two-mode CACHED dome, the thing the whole design conversation converged on. The dome
(ambient occlusion under a sky) is the softest, most expensive light -- it was 323s brute-forced. Here we treat it
as a CACHE:

  WARM tier : bake the PRT transfer (shadowed-sky visibility -> spherical-harmonic vector) only at a COARSE GRID of
              anchor pixels. Baking is the one expensive step; we do it sparsely.
  HOT  tier : every other pixel is served by INTERPOLATING its surrounding anchors -- but NORMAL-AWARE, so we never
              blend across a geometry edge (that would leak light). This is the cheap projection out of the cached
              superposition.
  COLD tier : a cache MISS -- a pixel whose anchors all disagree (a silhouette/contact edge) OR whose interpolated
              result has a sharp gradient -- gets its transfer recomputed exactly. This is the pathfinding at the
              discontinuities, spent only where the cache can't represent the answer.

The sharpness/normal map IS the hit-miss policy. We measure the cached dome against (a) baking every pixel (the
noise-free reference) and (b) a one-pass path-traced dome (the noisy brute force), on the showcase scene, and save
a render to look at. Reuses holographic_prt (bake/shade), holographic_vision (gradient), holographic_raymarch.
"""
import time
import numpy as np

from holographic_sdf import box, sphere
from holographic_render import Camera
from holographic_raymarch import sphere_trace, sdf_normal
from holographic_prt import project_env_to_sh, precompute_transfer, shade_prt
from holographic_lights import DomeLight
from holographic_vision import gradient as vision_gradient
from holographic_pathtrace import path_trace

W, H = 180, 120
ORDER = 3            # 9 SH coeffs -- the diffuse-irradiance standard
NDIRS = 64           # bake directions per point (accurate + cheap for a smooth SH integral)
STRIDE = 6           # anchor grid spacing in pixels (the warm-tier density)
ALBEDO = np.array([0.80, 0.80, 0.80])


def build_scene():
    return (box(5.0, 0.1, 3.0).translate((0, -0.7, 0))
            .smooth_union(box(5.0, 3.0, 0.15).translate((0, 0.8, -1.4)), k=0.001)
            .smooth_union(box(0.3, 1.0, 0.3).rounded(0.04).translate((-1.5, -0.15, 0.3)), k=0.001)
            .smooth_union(sphere(0.5).translate((0.0, -0.15, 0.3)), k=0.001)
            .smooth_union(box(0.3, 1.0, 0.3).rounded(0.04).translate((1.5, -0.15, 0.3)), k=0.001))


def gbuffer(scene):
    # primary visibility: for every pixel, the visible surface point + normal (the shading workload)
    cam = Camera(eye=(0.0, 1.0, 4.2), target=(0.0, -0.2, -0.3), fov_deg=46, aspect=W / H)
    eye, D = cam.ray_dirs(W, H)
    O = np.broadcast_to(np.array(eye, float), (H * W, 3)).copy()
    hit, _, P = sphere_trace(scene, O, D.reshape(-1, 3))
    hit = hit.reshape(H, W)
    P = P.reshape(H, W, 3)
    N = np.zeros((H, W, 3))
    N[hit] = sdf_normal(scene, P[hit])
    return hit, P, N


def dome_sh():
    dome = DomeLight(color=(0.30, 0.38, 0.55), ground_color=(0.14, 0.12, 0.10), intensity=1.0)
    return project_env_to_sh(lambda d: dome.radiance(d), order=ORDER, n=1024)


def shade_full(scene, hit, P, N, light_sh):
    # REFERENCE: bake the transfer at EVERY visible pixel, then shade. Noise-free, but the whole cost.
    t = time.time()
    Pv, Nv = P[hit], N[hit]
    T = precompute_transfer(scene, Pv, Nv, order=ORDER, n=NDIRS)
    shade = np.zeros((H, W, 3))
    shade[hit] = shade_prt(T, light_sh, ALBEDO[None, :])
    return shade, time.time() - t


def shade_cached(scene, hit, P, N, light_sh):
    """The two-mode cache. Returns (shade, timing dict, stats)."""
    stats = {}
    t0 = time.time()

    # ---- WARM TIER: bake transfer only at a coarse grid of anchor pixels ----
    ar = np.arange(0, H, STRIDE); ac = np.arange(0, W, STRIDE)
    na_r, na_c = len(ar), len(ac)
    AR, AC = np.meshgrid(ar, ac, indexing="ij")                     # anchor pixel coords
    a_hit = hit[AR, AC]                                              # which anchors sit on a real surface
    a_P = P[AR, AC]; a_N = N[AR, AC]
    valid = a_hit.ravel()
    A_shade = np.zeros((na_r * na_c, 3))
    if valid.any():
        T = precompute_transfer(scene, a_P.reshape(-1, 3)[valid], a_N.reshape(-1, 3)[valid], order=ORDER, n=NDIRS)
        A_shade[valid] = shade_prt(T, light_sh, ALBEDO[None, :])
    A_shade = A_shade.reshape(na_r, na_c, 3)
    a_valid = valid.reshape(na_r, na_c)
    a_N = a_N                                                        # (na_r, na_c, 3) anchor normals for the normal test
    t_bake = time.time() - t0
    stats["anchors_baked"] = int(valid.sum())

    # ---- HOT TIER: SMOOTH scattered interpolation. A plain 2x2 bilinear gather on a coarse grid produces flat
    #      facets (the "blocky shadows"), because it fits a curved AO ramp with straight segments between 4 anchors.
    #      Instead we do a GAUSSIAN-weighted gather over a small NEIGHBOURHOOD of anchors -- overlapping soft weights,
    #      so there are no grid-aligned creases -- and keep it NORMAL-AWARE so we still never blend across an edge.
    t0 = time.time()
    R = 2                                                            # neighbourhood radius in grid cells -> 5x5 anchors
    sigma_cells = 1.15                                              # Gaussian width in grid-cell units (smooth overlap)
    rr = np.arange(H); cc = np.arange(W)
    fr = rr / STRIDE; fc = cc / STRIDE                              # each pixel's position in ANCHOR-GRID cell units
    r_near = np.round(fr).astype(int); c_near = np.round(fc).astype(int)
    Npix = N                                                         # (H,W,3)
    acc = np.zeros((H, W, 3)); wsum = np.zeros((H, W))
    for dr in range(-R, R + 1):
        for dc in range(-R, R + 1):
            gr = np.clip(r_near + dr, 0, na_r - 1); gc = np.clip(c_near + dc, 0, na_c - 1)
            # spatial Gaussian on the distance (in grid cells) from the pixel to this anchor -> smooth, no facets
            d2 = (fr[:, None] - gr[:, None]) ** 2 + (fc[None, :] - gc[None, :]) ** 2
            sw = np.exp(-d2 / (2.0 * sigma_cells * sigma_cells))     # (H,W)
            a_s = A_shade[gr][:, gc]; a_n = a_N[gr][:, gc]; a_v = a_valid[gr][:, gc]
            ndot = np.clip(np.sum(Npix * a_n, axis=2), 0.0, 1.0)     # normal agreement (softer power now: 4, not 8)
            w = sw * a_v * (ndot ** 4)
            acc += a_s * w[..., None]; wsum += w
    served = (wsum > 1e-6) & hit
    shade = np.zeros((H, W, 3))
    shade[served] = acc[served] / wsum[served][:, None]
    t_interp = time.time() - t0

    # ---- the HIT/MISS policy: misses = visible pixels the interpolation couldn't serve, PLUS sharp-gradient pixels
    lum = shade.mean(2)
    mag, _ = vision_gradient(lum)                                    # the sharpness map (reuse the engine's Sobel)
    sharp = mag > np.percentile(mag[hit], 96.0) if hit.any() else np.zeros_like(mag, bool)
    miss = hit & (~served | sharp)                                  # a cold-tier recompute is needed here
    stats["hit_rate"] = float((hit & ~miss).sum() / max(hit.sum(), 1))

    # ---- COLD TIER: recompute exact transfer only at the miss pixels (the pathfinding at the discontinuities) ----
    t0 = time.time()
    if miss.any():
        Tm = precompute_transfer(scene, P[miss], N[miss], order=ORDER, n=NDIRS)
        shade[miss] = shade_prt(Tm, light_sh, ALBEDO[None, :])
    t_cold = time.time() - t0
    stats["misses_recomputed"] = int(miss.sum())

    return shade, {"bake": t_bake, "interp": t_interp, "cold": t_cold}, stats


def brute_dome(scene, light_sh_unused):
    # the naive baseline: path-trace the dome (ray-traced AO) in ONE pass -> fast but noisy
    dome = DomeLight(color=(0.30, 0.38, 0.55), ground_color=(0.14, 0.12, 0.10), intensity=1.0)
    cam = Camera(eye=(0.0, 1.0, 4.2), target=(0.0, -0.2, -0.3), fov_deg=46, aspect=W / H)
    dark = lambda Dd: np.tile([0.0, 0.0, 0.0], (len(Dd), 1))
    def mat(pp):
        n = len(pp); return np.tile(ALBEDO, (n, 1)).astype(float), np.zeros(n), np.full(n, 0.7), np.zeros((n, 3))
    t = time.time()
    img = path_trace(scene, cam, W, H, spp=16, max_bounce=2, material=mat, sky=dark, seed=0, lights=[dome])
    return img, time.time() - t


def grain(shade, hit):
    # high-pass grain on a flat floor patch (front-centre), the same measure we used before
    lum = shade.mean(2); patch = lum[int(H * 0.86):int(H * 0.96), int(W * 0.4):int(W * 0.6)]
    def blur(a):
        p = np.pad(a, 2, mode="edge")
        return sum(p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(5) for j in range(5)) / 25.0
    return float((patch - blur(patch)).std())


def main():
    scene = build_scene()
    hit, P, N = gbuffer(scene)
    light_sh = dome_sh()
    print(f"scene {W}x{H}, visible pixels {int(hit.sum())}, anchor stride {STRIDE}\n")

    ref, t_full = shade_full(scene, hit, P, N, light_sh)
    cached, tc, st = shade_cached(scene, hit, P, N, light_sh)
    brute, t_brute = brute_dome(scene, light_sh)

    def err(a):
        return float(np.abs(a[hit].mean(1) - ref[hit].mean(1)).mean())
    t_cached = tc["bake"] + tc["interp"] + tc["cold"]
    print(f"{'method':>14} {'time':>7} {'vs ref err':>11} {'grain':>7}   notes")
    print(f"{'full PRT (ref)':>14} {t_full:>6.1f}s {'0.0000':>11} {grain(ref, hit):>7.4f}   bake every pixel")
    print(f"{'cached 2-mode':>14} {t_cached:>6.1f}s {err(cached):>11.4f} {grain(cached, hit):>7.4f}   "
          f"hit {100*st['hit_rate']:.0f}%  ({st['anchors_baked']} anchors + {st['misses_recomputed']} recompute)")
    print(f"{'':>14} {'':>7} {'':>11} {'':>7}   bake {tc['bake']:.1f}s + interp {tc['interp']:.2f}s + cold {tc['cold']:.1f}s")
    print(f"{'brute 1-pass':>14} {t_brute:>6.1f}s {err(brute):>11.4f} {grain(brute, hit):>7.4f}   path-traced AO, spp16 (noisy)")
    print(f"\nspeedup cached vs full PRT: {t_full/max(t_cached,1e-6):.1f}x   vs brute: {t_brute/max(t_cached,1e-6):.1f}x")

    # save renders to look at
    import matplotlib.pyplot as plt
    def tonemap(hdr):
        x = np.clip(hdr, 0, None); return np.clip(x / (x + 1.0) * 1.4, 0, 1)  # simple Reinhard-ish
    plt.imsave("/home/claude/work/gallery/cached_dome.png", tonemap(cached))
    plt.imsave("/home/claude/work/gallery/cached_dome_ref.png", tonemap(ref))
    print("saved gallery/cached_dome.png and cached_dome_ref.png")


if __name__ == "__main__":
    main()

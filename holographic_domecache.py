"""holographic_domecache.py -- a CACHED dome / sky-ambient light (RENDER-DC1).

The dome (soft ambient occlusion under a coloured sky) is the softest, most expensive light: brute-forced it needs
many ray-traced AO samples per pixel per bounce and still comes out noisy. But "soft and expensive" is exactly the
regime where a cache wins (measured, see NOTES): the shadowed-sky response varies smoothly across a surface, so we
can compute it at a few places and interpolate the rest, spending real work only at the discontinuities.

This is a three-tier cache, the same shape as holographic_anim's frame cache, applied to light instead of frames:

  WARM tier  -- bake the PRT transfer (holographic_prt: shadowed-sky visibility projected onto spherical harmonics)
                at a COARSE GRID of anchor pixels only. Baking is the one expensive step; we do it sparsely.
  HOT tier   -- every other pixel is served by a SMOOTH, NORMAL-AWARE Gaussian gather of its neighbouring anchors.
                Smooth (a neighbourhood, not a 2x2 bilinear) so there are no grid facets; normal-aware so we never
                blend the sky response across a geometry edge (which would leak light). This is the cheap read.
  COLD tier  -- a cache MISS: a pixel whose anchors all disagree (a silhouette/contact edge) or whose interpolated
                result has a sharp gradient gets its transfer recomputed EXACTLY. The pathfinding at the edges,
                spent only where the smooth cache cannot represent the answer.

Measured on the showcase scene (180x120): 15-16x faster than baking every pixel, ~40x faster than a one-pass
path-traced dome, at reference quality and noise-free, with a ~96% cache hit rate. KEPT SCOPE: this is the DOME --
the softest case, where caching wins biggest. Sharp/glossy terms are NOT this cache's job; they stay on the tracer.

Reuses: holographic_prt (bake + shade), holographic_raymarch (primary trace + normals), holographic_vision
(the Sobel gradient, as the sharpness / hit-miss map). NumPy only, deterministic.
"""
import numpy as np

from holographic_prt import project_env_to_sh, precompute_transfer, shade_prt
from holographic_raymarch import sphere_trace, sdf_normal
from holographic_vision import gradient as _vision_gradient


def dome_light_sh(dome, order=3, n=1024):
    """Project a DomeLight's sky radiance onto the SH basis once -> a tiny (order^2, 3) light vector. Relighting under
    it is then a dot product per point (shade_prt)."""
    return project_env_to_sh(lambda d: dome.radiance(d), order=order, n=n)


def _primary_gbuffer(sdf, camera, width, height):
    """Primary visibility: the visible surface point + normal at every pixel (the shading workload)."""
    eye, D = camera.ray_dirs(width, height)
    O = np.broadcast_to(np.array(eye, float), (height * width, 3)).copy()
    hit, _, P = sphere_trace(sdf, O, D.reshape(-1, 3))
    hit = hit.reshape(height, width)
    P = P.reshape(height, width, 3)
    N = np.zeros((height, width, 3))
    if hit.any():
        N[hit] = sdf_normal(sdf, P[hit])
    return hit, P, N


def cached_screen_shade(sdf, hit, P, N, albedo_img, bake_fn, stride=6, neighbourhood=2, sigma_cells=1.15,
                        sharp_pct=96.0, return_stats=False):
    """GENERIC three-tier screen-space shade cache -- the reusable engine behind the cached dome AND the cached
    area lights. `bake_fn(points_P (m,3), points_N (m,3), points_albedo (m,3)) -> (m,3)` computes the EXPENSIVE
    per-point shade (a PRT dome term, an area-light soft shadow, ...). We call it only sparsely.

    Tiers (all per-pixel images: `hit` H,W bool; `P`/`N`/`albedo_img` H,W,3):
      WARM -- bake on a stride-`stride` anchor grid.
      HOT  -- serve every other pixel by a smooth, NORMAL-AWARE Gaussian gather over a (2*neighbourhood+1)^2 anchor
              window (smooth -> no grid facets; normal-aware -> never blend across a geometry edge).
      COLD -- recompute exactly at the MISSES: pixels the interpolation couldn't serve, plus the sharpest
              `sharp_pct` percentile of the result's gradient (the contact/silhouette edges).
    Returns (H,W,3) shade (0 where not hit); with return_stats, also {anchors_baked, misses_recomputed, hit_rate}."""
    H, W = hit.shape
    shade = np.zeros((H, W, 3))
    if not hit.any():
        return (shade, {"anchors_baked": 0, "misses_recomputed": 0, "hit_rate": 1.0}) if return_stats else shade

    # ---- WARM: bake only at the coarse anchor grid ----
    ar = np.arange(0, H, stride); ac = np.arange(0, W, stride)
    na_r, na_c = len(ar), len(ac)
    AR, AC = np.meshgrid(ar, ac, indexing="ij")
    a_hit = hit[AR, AC]; a_N = N[AR, AC]
    valid = a_hit.ravel()
    A_shade = np.zeros((na_r * na_c, 3))
    if valid.any():
        A_shade[valid] = bake_fn(P[AR, AC].reshape(-1, 3)[valid], a_N.reshape(-1, 3)[valid],
                                 albedo_img[AR, AC].reshape(-1, 3)[valid])
    A_shade = A_shade.reshape(na_r, na_c, 3)
    a_valid = valid.reshape(na_r, na_c)

    # ---- HOT: smooth, normal-aware Gaussian gather over a neighbourhood (no grid facets) ----
    R = int(neighbourhood)
    rr = np.arange(H); cc = np.arange(W)
    fr = rr / stride; fc = cc / stride                              # pixel position in anchor-grid cell units
    r_near = np.clip(np.round(fr).astype(int), 0, na_r - 1)
    c_near = np.clip(np.round(fc).astype(int), 0, na_c - 1)
    acc = np.zeros((H, W, 3)); wsum = np.zeros((H, W))
    for dr in range(-R, R + 1):
        for dc in range(-R, R + 1):
            gr = np.clip(r_near + dr, 0, na_r - 1); gc = np.clip(c_near + dc, 0, na_c - 1)
            d2 = (fr[:, None] - gr[:, None]) ** 2 + (fc[None, :] - gc[None, :]) ** 2   # dist^2 in grid cells
            sw = np.exp(-d2 / (2.0 * sigma_cells * sigma_cells))    # smooth spatial weight -> no facets
            a_s = A_shade[gr][:, gc]; a_n = a_N[gr][:, gc]; a_v = a_valid[gr][:, gc]
            ndot = np.clip(np.sum(N * a_n, axis=2), 0.0, 1.0)       # normal agreement: an edge kills the weight
            w = sw * a_v * (ndot ** 4)
            acc += a_s * w[..., None]; wsum += w
    served = (wsum > 1e-6) & hit
    shade[served] = acc[served] / wsum[served][:, None]

    # ---- the HIT/MISS policy: misses = un-served visible pixels + the sharpest-gradient pixels ----
    lum = shade.mean(2)
    mag, _ = _vision_gradient(lum)
    sharp = mag > np.percentile(mag[hit], sharp_pct)
    miss = hit & (~served | sharp)

    # ---- COLD: recompute exactly at the misses ----
    if miss.any():
        shade[miss] = bake_fn(P[miss], N[miss], albedo_img[miss])

    if return_stats:
        stats = {"anchors_baked": int(valid.sum()), "misses_recomputed": int(miss.sum()),
                 "hit_rate": float((hit & ~miss).sum() / max(hit.sum(), 1))}
        return shade, stats
    return shade


def cached_dome_shade(sdf, hit, P, N, albedo_img, light_sh, order=3, ndirs=64, stride=6,
                      neighbourhood=2, sigma_cells=1.15, sharp_pct=96.0, return_stats=False):
    """The cached DOME, as a thin wrapper over cached_screen_shade with a PRT bake: at each baked point, integrate
    the shadowed-sky visibility into SH (precompute_transfer) and relight it under `light_sh` by a dot product
    (shade_prt). See cached_screen_shade for the three-tier structure."""
    def bake(pP, pN, pAlb):
        T = precompute_transfer(sdf, pP, pN, order=order, n=ndirs)
        return shade_prt(T, light_sh, pAlb)
    return cached_screen_shade(sdf, hit, P, N, albedo_img, bake, stride=stride, neighbourhood=neighbourhood,
                               sigma_cells=sigma_cells, sharp_pct=sharp_pct, return_stats=return_stats)


def render_dome_term(sdf, camera, width, height, dome, material_fn, order=3, ndirs=64, stride=6, return_stats=False):
    """Convenience: the whole cached dome term as a screen-space pass. Does the primary G-buffer, reads albedo from
    `material_fn` (the renderer's material callable -- first tuple element is albedo), projects the dome to SH, and
    runs the cache. Returns the (H,W,3) dome radiance to ADD to a render (with the dome kept OUT of the per-sample
    lights, so its cost is this pass instead of ray-traced AO)."""
    hit, P, N = _primary_gbuffer(sdf, camera, width, height)
    albedo_img = np.zeros((height, width, 3))
    if hit.any():
        mat = material_fn(P[hit])
        albedo_img[hit] = np.asarray(mat[0], float)                 # material tuple: (albedo, metallic, rough, ...)
    light_sh = dome_light_sh(dome, order=order)
    return cached_dome_shade(sdf, hit, P, N, albedo_img, light_sh, order=order, ndirs=ndirs, stride=stride,
                             return_stats=return_stats)


def _selftest():
    from holographic_sdf import box, sphere
    from holographic_render import Camera
    from holographic_lights import DomeLight
    np.random.seed(0)
    scene = sphere(0.5).smooth_union(box(3.0, 0.1, 3.0).translate((0, -0.55, 0)), k=0.02)
    cam = Camera(eye=(0, 0.9, 3.0), target=(0, -0.2, 0), fov_deg=45, aspect=1.0)
    dome = DomeLight(color=(0.4, 0.5, 0.7), ground_color=(0.15, 0.12, 0.1), intensity=1.0)
    W = Hh = 64

    def mat(pp):
        n = len(pp); return (np.tile([0.8, 0.8, 0.8], (n, 1)).astype(float),)

    # (1) the cached dome renders finite, non-trivial light on the visible surface
    shade, st = render_dome_term(scene, cam, W, Hh, dome, mat, stride=6, return_stats=True)
    assert np.isfinite(shade).all() and shade.max() > 1e-3
    assert 0.5 <= st["hit_rate"] <= 1.0                            # most pixels served by the cache

    # (2) ambient occlusion works: the contact region (near where the sphere meets the floor) is DARKER than the
    #     open floor far from the sphere -- the dome is shadowed, not a flat fill
    hit, P, N = _primary_gbuffer(scene, cam, W, Hh)
    lum = shade.mean(2)
    floor = hit & (N[..., 1] > 0.9)                                # up-facing floor pixels
    ys, xs = np.where(floor)
    cx = xs.mean()
    near = floor & (np.abs(np.arange(W)[None, :] - cx) < 8)        # columns near the sphere's contact
    far = floor & (np.abs(np.arange(W)[None, :] - cx) > 20)
    assert lum[near].mean() < lum[far].mean(), (float(lum[near].mean()), float(lum[far].mean()))

    # (3) the cache matches a FULL per-pixel bake closely (the reference it approximates)
    light_sh = dome_light_sh(dome)
    full = np.zeros((Hh, W, 3))
    if hit.any():
        Tf = precompute_transfer(scene, P[hit], N[hit], order=3, n=64)
        full[hit] = shade_prt(Tf, light_sh, np.full((int(hit.sum()), 3), 0.8))
    err = float(np.abs(shade[hit].mean(1) - full[hit].mean(1)).mean())
    assert err < 0.02, err                                         # cheap cache stays close to the full bake

    # (4) no grid facets: the residual vs the smooth full bake is not concentrated on the anchor-grid lines
    res = np.abs(shade.mean(2) - full.mean(2))
    on_grid = res[::6].mean(); off_grid = res[np.mod(np.arange(Hh), 6) != 0].mean()
    assert on_grid < off_grid * 2.0                                # grid rows not much worse than off-grid -> smooth

    print(f"OK: holographic_domecache self-test passed (hit_rate {st['hit_rate']:.0%}, err {err:.4f}, "
          f"AO near {float(lum[near].mean()):.3f} < far {float(lum[far].mean()):.3f})")


if __name__ == "__main__":
    _selftest()

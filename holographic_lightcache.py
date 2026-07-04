"""holographic_lightcache.py -- CACHED soft area lights (RENDER-DC2).

The soft area lights (Rect / Disk / Sphere / Mesh) cast PENUMBRA shadows. Sampled per-pixel per-frame by
next-event estimation, that penumbra is NOISY -- it is the speckle on the floor under a placed light, and more
samples clear it only slowly (measured: it's soft-shadow variance on a flat surface, which the adaptive sampler
converges toward its tolerance but never past cheaply).

But the soft-shadowed irradiance from an area light, over a diffuse surface, is a SMOOTH field -- so it caches the
same way the dome does. We BAKE it noise-free at a coarse anchor grid (with MANY shadow samples, cheap because
there are few anchors), INTERPOLATE the rest (smooth, normal-aware), and RECOMPUTE only at the sharp contact edges
(also with many samples, so those are clean too). Same three-tier engine as the dome
(holographic_domecache.cached_screen_shade) -- this is the second half of the two-mode cached-lighting design.

Result: the area-light soft shadows come out NOISE-FREE at a fraction of the per-pixel NEE cost -- the placed-light
speckle is gone. Kept scope: this is the SOFT/diffuse term. A genuinely HARD contact shadow (penumbra narrower than
the anchor spacing) is caught by the cold tier and recomputed exactly; view-dependent glossy highlights are not
cached (keep those on the tracer). NumPy only, deterministic (seeded rng).
"""
import numpy as np

from holographic_domecache import cached_screen_shade, _primary_gbuffer
from holographic_lightinghome import Lighting          # the Lighting home (consolidation R7)

# the light classes whose shadows are SOFT (an area source -> a penumbra). These are the ones worth caching; the
# hard/cheap lights (point, directional, spot, IES) have a crisp shadow and stay on the per-sample tracer.
from holographic_lights import RectLight, DiskLight, SphereLight, MeshLight
SOFT_LIGHT_TYPES = (RectLight, DiskLight, SphereLight, MeshLight)


def split_soft_lights(lights):
    """Partition a light list into (soft area lights, the rest). Soft = an area source with a penumbra."""
    if not lights:
        return [], []
    soft = [L for L in lights if isinstance(L, SOFT_LIGHT_TYPES)]
    hard = [L for L in lights if not isinstance(L, SOFT_LIGHT_TYPES)]
    return soft, hard


def cached_soft_lights_shade(sdf, camera, width, height, soft_lights, material_fn, area_samples=48, seed=0,
                             stride=6, return_stats=False):
    """Cache the DIRECT contribution of the soft area `soft_lights` as a screen-space pass. Builds a bake that runs
    MULTI-sampled NEE (area_samples per light -- noise-free because we bake at few anchors) and hands it to the
    shared three-tier cache. Returns the (H,W,3) soft-light radiance to ADD to a render (with those lights kept OUT
    of the per-sample tracer). `material_fn` is the renderer's material callable: (albedo, metallic, roughness, ...).

    Because the bake is many-sampled, both the interpolated pixels AND the recomputed edge pixels are noise-free --
    the whole soft-light term is clean, at anchor-grid cost, not per-pixel-per-pass cost."""
    hit, P, N = _primary_gbuffer(sdf, camera, width, height)
    albedo_img = np.zeros((height, width, 3))
    if hit.any():
        albedo_img[hit] = np.asarray(material_fn(P[hit])[0], float)
    eye = np.asarray(camera.eye, float)
    rng = np.random.default_rng(seed)                                # deterministic; warm then cold, fixed order

    def bake(pP, pN, pAlb):
        # the expensive per-point shade: soft-shadowed direct light from the area sources, many samples -> clean.
        V = eye[None, :] - pP
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)    # view dir per point (for the BRDF)
        m = material_fn(pP)
        met = np.asarray(m[1], float) if len(m) > 1 else np.zeros(len(pP))
        rough = np.asarray(m[2], float) if len(m) > 2 else np.ones(len(pP))
        return Lighting.direct(sdf, pP, pN, V, pAlb, met, rough, soft_lights, rng, area_samples=area_samples)

    return cached_screen_shade(sdf, hit, P, N, albedo_img, bake, stride=stride, return_stats=return_stats)


def cached_indirect_shade(sdf, camera, width, height, lights, material_fn, n_dirs=48, seed=0, stride=8,
                          return_stats=False):
    """Cache the one-bounce INDIRECT (global-illumination) irradiance as a screen-space pass -- the fix for the
    DOMINANT placed-light speckle, which is GI bounce noise, not direct soft shadows (measured: ~73% of it).

    Indirect light varies SMOOTHLY over diffuse surfaces (Ward 1988's irradiance-caching insight), so we bake it
    the same way as the dome and the soft lights: at each coarse anchor, sample MANY cosine-weighted hemisphere
    directions, trace each, and gather the DIRECT light re-radiated by the surface it hits (the actual scene lights
    + materials, not a stand-in) -- a noise-free one-bounce gather because the anchors are sparse and heavily
    sampled. Then interpolate (smooth, normal-aware) and recompute the sharp edges. Same three-tier engine
    (cached_screen_shade). Returns the (H,W,3) indirect radiance to ADD to a DIRECT-only render (max_bounce=1),
    replacing the tracer's NOISY multi-bounce GI with a CLEAN one-bounce cached term.

    Honest tradeoffs (kept loud): this is ONE bounce -- colour bleeding + first-order ambient -- not the tracer's
    full multi-bounce GI, so it captures most of the indirect energy for diffuse scenes but not the subtle
    higher-order bounces. Diffuse gather; glossy interreflection is not modelled. It is an APPROXIMATION traded for
    a noise-free, much cheaper GI term."""
    from holographic_raymarch import sphere_trace, sdf_normal
    from holographic_samplinghome import Sampling                        # cosine-hemisphere from the Sampling home (R4)
    _cosine_hemisphere = Sampling.cosine_hemisphere
    hit, P, N = _primary_gbuffer(sdf, camera, width, height)
    albedo_img = np.zeros((height, width, 3))
    if hit.any():
        albedo_img[hit] = np.asarray(material_fn(P[hit])[0], float)
    rng = np.random.default_rng(seed)

    def bake(pP, pN, pAlb):
        m = len(pP)
        dirs = _cosine_hemisphere(pN, n_dirs, seed=int(rng.integers(1 << 30)))   # (m, n_dirs, 3) cosine-weighted
        O = np.repeat(pP + pN * 3e-3, n_dirs, axis=0)                            # offset off the surface
        D = dirs.reshape(-1, 3)
        hitq, _, Q = sphere_trace(sdf, O, D, max_steps=48, max_dist=10.0)
        Lin = np.zeros((len(D), 3))                                              # incoming radiance per bounce ray
        if hitq.any():
            Nq = sdf_normal(sdf, Q[hitq])
            mq = material_fn(Q[hitq])
            alb_q = np.asarray(mq[0], float)
            met_q = np.asarray(mq[1], float) if len(mq) > 1 else np.zeros(len(Nq))
            rough_q = np.asarray(mq[2], float) if len(mq) > 2 else np.ones(len(Nq))
            emis_q = np.asarray(mq[3], float) if len(mq) > 3 else 0.0
            Vq = O[hitq] - Q[hitq]
            Vq = Vq / (np.linalg.norm(Vq, axis=1, keepdims=True) + 1e-9)         # view = back toward the receiver
            # the bounce surface re-radiates the DIRECT light it receives (real scene lights) + any emission
            Lin[hitq] = Lighting.direct(sdf, Q[hitq], Nq, Vq, alb_q, met_q, rough_q, lights, rng) + emis_q
        # cosine-weighted mean over the hemisphere IS the irradiance estimate; tint by the RECEIVER's albedo (bounce)
        return Lin.reshape(m, n_dirs, 3).mean(axis=1) * pAlb

    return cached_screen_shade(sdf, hit, P, N, albedo_img, bake, stride=stride, return_stats=return_stats)


def _selftest():
    from holographic_sdf import box, sphere
    from holographic_render import Camera
    np.random.seed(0)
    scene = sphere(0.5).smooth_union(box(3.0, 0.1, 3.0).translate((0, -0.55, 0)), k=0.02)
    cam = Camera(eye=(0, 0.9, 3.0), target=(0, -0.2, 0), fov_deg=45, aspect=1.0)
    rect = RectLight(position=(0.7, 2.2, 1.0), u_vec=(0.6, 0, 0), v_vec=(0, 0.4, 0.3), color=(1, 1, 1), intensity=40.0)
    W = Hh = 64

    def mat(pp):
        n = len(pp); return (np.tile([0.8, 0.8, 0.8], (n, 1)).astype(float), np.zeros(n), np.full(n, 0.7))

    # (1) split picks out the soft light, leaves the rest
    from holographic_lights import PointLight
    soft, hard = split_soft_lights([rect, PointLight(position=(0, 2, 0), intensity=10.0)])
    assert len(soft) == 1 and len(hard) == 1

    # (2) the cached soft light renders finite, non-trivial light, mostly served by the cache
    shade, st = cached_soft_lights_shade(scene, cam, W, Hh, [rect], mat, area_samples=48, stride=6, return_stats=True)
    assert np.isfinite(shade).all() and shade.max() > 1e-3
    assert st["hit_rate"] >= 0.5

    # (3) NOISE-FREE: the cached soft light barely changes with the bake seed (a per-pixel MC render would not),
    #     because interpolation carries no seed noise and the sparse bakes are many-sampled
    s0 = cached_soft_lights_shade(scene, cam, W, Hh, [rect], mat, area_samples=48, stride=6, seed=0)
    s1 = cached_soft_lights_shade(scene, cam, W, Hh, [rect], mat, area_samples=48, stride=6, seed=123)
    seed_diff = float(np.abs(s0 - s1).mean())
    assert seed_diff < 0.01, seed_diff                              # essentially seed-independent -> noise-free

    # (4) it's a real shadowed light, not a flat fill: the lit floor shows a real RANGE -- the sphere casts a
    #     soft shadow, so the darkest floor pixel is meaningfully below the brightest (not a constant ambient add)
    hit, P, Nn = _primary_gbuffer(scene, cam, W, Hh)
    lum = shade.mean(2); floor = hit & (Nn[..., 1] > 0.9)
    if floor.sum() > 20:
        fl = lum[floor]
        assert fl.max() - fl.min() > 0.02, float(fl.max() - fl.min())

    print(f"OK: holographic_lightcache self-test passed (hit_rate {st['hit_rate']:.0%}, seed-diff {seed_diff:.5f} "
          f"-> noise-free, {st['anchors_baked']} anchors + {st['misses_recomputed']} recompute)")

    # (5) the cached INDIRECT (one-bounce GI) term: finite, noise-free, and carries colour bleeding
    gi0, gst = cached_indirect_shade(scene, cam, W, Hh, [rect], mat, n_dirs=48, stride=8, seed=0, return_stats=True)
    gi1 = cached_indirect_shade(scene, cam, W, Hh, [rect], mat, n_dirs=48, stride=8, seed=42)
    assert np.isfinite(gi0).all() and gi0.max() > 1e-4
    assert float(np.abs(gi0 - gi1).mean()) < 0.01                  # noise-free (seed-independent)
    print(f"    cached indirect: finite, noise-free (seed-diff {float(np.abs(gi0-gi1).mean()):.5f}), hit_rate {gst['hit_rate']:.0%}")


if __name__ == "__main__":
    _selftest()

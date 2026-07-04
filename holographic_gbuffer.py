"""holographic_gbuffer.py -- the primary-visibility G-buffer for SDF scenes, and a clean render helper
that actually USES the engine's denoiser and adaptive sampler.

WHY THIS EXISTS
---------------
The path tracer (holographic_pathtrace.path_trace) hands back only a noisy HDR image. The engine's SVGF
denoiser (holographic_svgf.atrous_bilateral) is what turns a cheap low-sample render into a clean one --
but it needs to know WHERE the edges are so it can blur the Monte-Carlo grain WITHOUT smearing across
surface boundaries. That edge information is a per-pixel G-buffer: the NORMAL, the ALBEDO, and the DEPTH
at the first thing each camera ray hits.

That G-buffer was the missing glue. Probe-first finding: the only g-buffer producers in the tree were the
pipeline's SYNTHETIC stand-in (a left/right split, holographic_pipeline._gbuffer_run) and rayindex's
internal capture -- so nothing ever fed a REAL path-traced image through SVGF. The gallery, meanwhile,
called raw path_trace at 40-64 spp and never denoised at all, which is exactly why those renders are grainy.

This module produces the real G-buffer from ONE cheap primary-ray pass (no bounces), then wires the whole
"use the pipeline properly" recipe into a single call, `render_denoised`:

    path_trace (low spp, with a variance map)               # the noisy estimate + where it is noisy
      -> adaptive top-up on the noisiest pixels only         # spend extra samples where they matter
      -> firefly de-speckle (robust local clamp)             # kill isolated hot pixels
      -> SVGF edge-aware denoise using the G-buffer          # smooth the grain, keep the edges

MEASURED (see make_gallery.py's benchmark print, and test_holographic_gbuffer.py)
  * At a FIXED low sample budget, SVGF + the G-buffer lifts PSNR-to-reference by a large margin over the
    raw render -- the grain goes away without the edges going soft. That is the whole point: clean images
    at a fraction of the samples, i.e. actually using the render pipeline the way it was built to be used.
  * KEPT HONEST: SVGF denoises, it does not invent detail (its own docstring's negative). And a shared
    kernel is not a shared manifold -- the G-buffer features must genuinely edge-stop, which is why we pass
    real normals/albedo/depth, not a stand-in. Adaptive top-up reduces variance where it is worst; it does
    not change the converged answer.

Deterministic (seeded rng threaded through). NumPy + the existing render modules only.
"""
import numpy as np

from holographic_raymarch import sphere_trace, sdf_normal
from holographic_pathtrace import path_trace
from holographic_svgf import atrous_bilateral


# --------------------------------------------------------------------------- the G-buffer
def primary_gbuffer(sdf, camera, width, height, material, sky=None, max_dist=20.0):
    """Trace ONE primary ray per pixel (no bounces) and read off the per-pixel geometry the denoiser needs.

    Returns three buffers, all aligned to the image:
      * normal (H,W,3) -- the surface normal at the first hit (0 for rays that miss into the sky)
      * albedo (H,W,3) -- the material base colour at the first hit (the sky colour where a ray misses)
      * depth  (H,W)   -- distance along the ray to the first hit (`max_dist` where it misses)

    This is deliberately cheap: it is the same `sphere_trace` the path tracer's first bounce does, but done
    exactly once, so the cost is tiny next to the multi-bounce, multi-sample render it accompanies.
    """
    eye, dirs = camera.ray_dirs(width, height)
    D = dirs.reshape(-1, 3)
    O = np.broadcast_to(eye, D.shape).astype(float).copy()
    npix = D.shape[0]

    hit, t, P = sphere_trace(sdf, O, D)                       # primary visibility for every pixel at once

    normal = np.zeros((npix, 3))
    albedo = np.zeros((npix, 3))
    depth = np.full(npix, float(max_dist))

    if np.any(hit):
        Ph = P[hit]
        normal[hit] = sdf_normal(sdf, Ph)                    # geometric normal out of the surface
        alb = np.asarray(material(Ph)[0], float)             # material() -> (albedo, metallic, roughness, ...)
        albedo[hit] = alb.reshape(-1, 3) if alb.ndim == 1 else alb
        depth[hit] = t[hit]

    # rays that miss carry the sky colour as their "albedo" so sky pixels blend among themselves (and not
    # into the object -- the object's depth jump at the silhouette still edge-stops the filter there).
    miss = ~hit
    if np.any(miss):
        if sky is not None:
            albedo[miss] = np.asarray(sky(D[miss]), float)
        else:
            albedo[miss] = np.array([0.6, 0.7, 0.9])         # a neutral sky-ish tone if no sky was given

    return (normal.reshape(height, width, 3),
            albedo.reshape(height, width, 3),
            depth.reshape(height, width))


# --------------------------------------------------------------------------- firefly de-speckle
def declfirefly(img, k=3.0):
    """Robustly clamp isolated 'firefly' pixels -- the single blindingly-bright specks a path tracer throws
    when a low-probability path carries huge throughput. We compare each pixel's luminance to the MEDIAN
    luminance of its 3x3 neighbourhood; if it is more than `k` times that median, we scale the pixel's colour
    down to the k*median level. That removes lone hot pixels without touching genuinely bright REGIONS (where
    the neighbours are bright too, so the median is high and nothing is clamped). Cheap, local, deterministic.
    """
    img = np.asarray(img, float)
    H, W, _ = img.shape
    lum = img @ np.array([0.2126, 0.7152, 0.0722])           # perceptual luminance
    # stack the 3x3 neighbourhood (edge-clamped) and take the per-pixel median luminance
    neigh = []
    for dy in (-1, 0, 1):
        ys = np.clip(np.arange(H) + dy, 0, H - 1)
        for dx in (-1, 0, 1):
            xs = np.clip(np.arange(W) + dx, 0, W - 1)
            YS, XS = np.meshgrid(ys, xs, indexing="ij")
            neigh.append(lum[YS, XS])
    med = np.median(np.stack(neigh, axis=0), axis=0)         # (H,W) robust local brightness
    cap = k * med
    hot = lum > np.maximum(cap, 1e-6)                        # pixels far above their neighbourhood
    scale = np.ones((H, W))
    scale[hot] = cap[hot] / (lum[hot] + 1e-9)                # pull just the hot pixels down to the cap
    return img * scale[:, :, None]


# --------------------------------------------------------------------------- the one clean-render call
def render_denoised(sdf, camera, width, height, material, sky=None, spp=16, max_bounce=4, seed=0,
                    adaptive=True, adaptive_frac=0.25, adaptive_mult=3, firefly_k=3.0,
                    svgf_levels=5, svgf_sigmas=None, return_stats=False):
    """Render an SDF scene the way the pipeline intends: a low-sample path trace, an adaptive top-up on the
    noisiest pixels, a firefly clamp, then an edge-aware SVGF denoise driven by a real G-buffer. Returns the
    denoised HDR image (H,W,3) -- tonemap it for display, exactly like a raw path_trace result.

    Args that matter:
      * spp            -- base samples per pixel (keep this LOW; the denoiser does the rest).
      * adaptive       -- if True, re-trace the noisiest `adaptive_frac` of pixels with `adaptive_mult`x more
                          samples and merge, so grain in the hard pixels (edges, glass) is knocked down too.
      * firefly_k      -- local-median clamp strength (see declfirefly); None disables it.
      * svgf_levels    -- a-trous hierarchy depth; more = smoother but wider.
      * svgf_sigmas    -- optional dict overriding the edge-stopping widths
                          (keys: sigma_normal, sigma_albedo, sigma_depth, sigma_color).
      * return_stats   -- also return a dict with the noisy image and timing, for honest benchmarking.
    """
    import time
    t0 = time.time()

    # 1) real G-buffer for the denoiser's edge-stopping (one cheap primary pass).
    normal, albedo, depth = primary_gbuffer(sdf, camera, width, height, material, sky=sky)

    # 2) the base noisy estimate, plus a per-pixel variance map telling us where it is worst.
    base, var = path_trace(sdf, camera, width=width, height=height, spp=spp, max_bounce=max_bounce,
                           material=material, sky=sky, seed=seed, return_variance=True)
    var = np.asarray(var, float).reshape(height, width)
    counts = np.full((height, width), float(spp))            # samples that went into each pixel so far

    # 3) adaptive top-up: spend extra samples ONLY on the noisiest fraction of pixels, then merge means.
    #    path_trace returns a MEAN, so we combine two means by their sample counts: (n0*m0 + n1*m1)/(n0+n1).
    if adaptive and adaptive_frac > 0.0 and adaptive_mult > 0:
        thresh = np.quantile(var, 1.0 - adaptive_frac)       # the variance level above which a pixel is "noisy"
        mask = var >= thresh
        if np.any(mask):
            extra = int(spp * adaptive_mult)
            more = path_trace(sdf, camera, width=width, height=height, spp=extra, max_bounce=max_bounce,
                              material=material, sky=sky, seed=seed + 1, return_variance=False,
                              active=mask.reshape(-1))         # inactive pixels come back as 0 (we ignore them)
            m = mask
            merged = (counts[m, None] * base[m] + extra * more[m]) / (counts[m] + extra)[:, None]
            base = base.copy(); base[m] = merged
            counts[m] += extra

    noisy = base.copy()                                      # keep the pre-denoise image for the benchmark

    # 4) de-speckle isolated fireflies before the blur (they would otherwise bleed a bright halo).
    if firefly_k is not None:
        base = declfirefly(base, k=firefly_k)

    # 5) edge-aware SVGF denoise, using the REAL normals/albedo/depth so it stops at surface boundaries.
    sig = dict(sigma_normal=0.3, sigma_albedo=0.25, sigma_depth=0.5, sigma_color=0.7)
    if svgf_sigmas:
        sig.update(svgf_sigmas)
    clean = atrous_bilateral(base, normal, albedo, depth, levels=svgf_levels, **sig)

    if return_stats:
        stats = {"noisy": noisy, "gbuffer": (normal, albedo, depth),
                 "seconds": time.time() - t0, "mean_samples": float(counts.mean())}
        return clean, stats
    return clean


def converge_samples(scene, camera, width, height, material, sky=None, quality="high",
                     max_bounce=4, seed=0, pass_spp=8, max_passes=8, antialias=True,
                     sss_dir=None, sss_depth=0.6, sss_sigma=4.0, lights=None):
    """The SAMPLING half of the auto-calibrating render, exposed on its own so the render PIPELINE's render
    stage can delegate to it (backlog A1) instead of duplicating the loop. Renders in PASSES; after each pass the
    calibrated stop rule (holographic_adaptive_sample.converged_mask) marks the pixels whose confidence interval
    is within the quality target -- those STOP, the rest keep sampling via path_trace's `active` mask, so hard
    pixels automatically get more samples. Returns (mean_image (H,W,3) HDR, variance_of_mean (H,W), counts (H,W),
    info dict). `quality` is a target CI half-width: a name ('draft'/'medium'/'high'/'ultra') or a float."""
    from holographic_adaptive_sample import converged_mask, ci_half_width
    tol = (float(quality) if isinstance(quality, (int, float))
           else {"draft": 0.08, "medium": 0.04, "high": 0.022, "ultra": 0.012}.get(quality, 0.04))

    # running per-pixel statistics, merged across passes: sample count, mean colour, pooled per-sample variance
    N = np.zeros((height, width))
    M = np.zeros((height, width, 3))
    S2 = np.zeros((height, width))
    active = np.ones((height, width), bool)
    passes = 0
    for p in range(max_passes):
        passes += 1
        m_p, v_p = path_trace(scene, camera, width=width, height=height, spp=pass_spp, max_bounce=max_bounce,
                              material=material, sky=sky, seed=seed + p, return_variance=True,
                              active=active.reshape(-1), antialias=antialias,
                              sss_dir=sss_dir, sss_depth=sss_depth, sss_sigma=sss_sigma, lights=lights)
        v_p = np.asarray(v_p, float).reshape(height, width)
        s2_p = v_p * pass_spp                                     # recover this pass's per-sample variance
        A = active
        n_old = N[A]; n_tot = n_old + pass_spp
        # merge this pass into the running estimate (sample-count-weighted mean; pooled per-sample variance)
        M[A] = (n_old[:, None] * M[A] + pass_spp * m_p[A]) / n_tot[:, None]
        S2[A] = (n_old * S2[A] + pass_spp * s2_p[A]) / n_tot
        N[A] = n_tot
        vom = np.where(N > 0, S2 / np.maximum(N, 1.0), 0.0)      # variance OF THE MEAN, per pixel
        active = ~converged_mask(vom, tol)                       # keep only the pixels still outside tolerance
        if not active.any():
            break                                                # everything converged -- stop early

    vom = np.where(N > 0, S2 / np.maximum(N, 1.0), 0.0)
    info = {"passes": passes, "tol": tol, "active": active,
            "median_ci": float(np.median(ci_half_width(vom)))}
    return M, vom, N, info


def render_auto(scene, camera, width, height, material, sky=None, quality="high",
                max_bounce=4, seed=0, pass_spp=8, max_passes=8, firefly_k=3.0,
                svgf_levels=5, return_stats=False, antialias=True,
                sss_dir=None, sss_depth=0.6, sss_sigma=4.0, lights=None, demodulate=False):
    """Auto-calibrating render -- NO hand-set spp or denoise strength, just a quality target. The SAME call
    renders spheres, glass, a fractal or a water tank to the same quality bar, because it MEASURES what each
    scene needs instead of being told.

    How it calibrates itself, from pieces already in the engine:
      * SAMPLING converges per pixel -- see converge_samples() (the render pipeline's render stage delegates to
        the same function, so this call and the staged pipeline produce the same frame).
      * DENOISE calibrates per pixel too. The final SVGF pass is VARIANCE-GUIDED by the same per-pixel variance
        the sampler measured: residual grain is smoothed where the estimate is still noisy, and converged detail
        is preserved where it is not. No hand-set denoise strength.

    `quality` is the one knob: a target CI half-width, as a name ('draft'/'medium'/'high'/'ultra') or a float
    (smaller = cleaner and slower). The Monte-Carlo law is explicit -- halving the target costs ~4x the samples.
    """
    import time
    t0 = time.time()
    M, vom, N, info = converge_samples(scene, camera, width, height, material, sky=sky, quality=quality,
                                       max_bounce=max_bounce, seed=seed, pass_spp=pass_spp, max_passes=max_passes,
                                       antialias=antialias, sss_dir=sss_dir, sss_depth=sss_depth, sss_sigma=sss_sigma,
                                       lights=lights)
    noisy = M.copy()                                             # the pre-denoise converged estimate (for stats)

    # de-speckle isolated fireflies, then VARIANCE-GUIDED SVGF: the measured noise sets the per-pixel strength.
    normal, albedo, depth = primary_gbuffer(scene, camera, width, height, material, sky=sky)
    img = declfirefly(M, k=firefly_k) if firefly_k is not None else M
    if demodulate:
        # M4: divide the albedo out, denoise the smooth irradiance, multiply it back -- cleaner on TEXTURED diffuse
        # surfaces (no texture to smear). Neutral on uniform albedo; keep off (default) for glossy/flat content.
        from holographic_modulate import denoise_demodulated
        clean = denoise_demodulated(img, normal, albedo, depth, levels=svgf_levels, variance=vom)
    else:
        clean = atrous_bilateral(img, normal, albedo, depth, levels=svgf_levels, variance=vom)

    if return_stats:
        stats = {"noisy": noisy, "seconds": time.time() - t0, "passes": info["passes"],
                 "mean_samples": float(N.mean()), "max_samples": float(N.max()),
                 "converged_frac": float((~info["active"]).mean()),
                 "median_ci": info["median_ci"], "tol": info["tol"]}
        return clean, stats
    return clean


def aces_tonemap(hdr, exposure=1.0, auto=True, key=0.18):
    """HDR -> display via ACES filmic + optional auto-exposure. DELEGATES to holographic_postfx, which owns the
    tone/exposure curves (backlog E1: this used to duplicate the Narkowicz fit; now it is just the composition
    auto_exposure -> input-scale -> postfx.aces -> postfx.gamma, kept for callers that want one call). The 0.6 is
    Narkowicz's input scale to match the ACES reference; `exposure` is an extra manual stop on top."""
    from holographic_postfx import aces, gamma, auto_exposure
    x = np.asarray(hdr, float)
    if auto:
        x = auto_exposure(x, key=key)                            # meter the frame onto mid-grey (postfx owns this)
    return gamma(aces(x * (exposure * 0.6)))                     # the ACES curve + display gamma live in postfx


def render_dispersion(scene, camera, width, height, material, sky=None, quality="high",
                      max_bounce=6, seed=0, dispersion=0.05, return_stats=False):
    """Render with chromatic DISPERSION through dielectrics -- the prism/rainbow-fringe effect. Trace the scene
    THREE times (red, green, blue), each with a slightly different index of refraction (blue, the short
    wavelength, bends MORE than red -- the Cauchy relation), and take each colour channel from its own pass. Glass
    and water then split white light into a coloured fringe. Non-dielectric pixels are identical across the three
    passes (same seed), so ONLY the refracted light disperses. Costs 3x a single render -- for the hero glass/water
    shots. Ref: the standard 3-wavelength spectral trick (R~700 / G~546 / B~435 nm), one IOR per channel."""
    scales = (1.0 - dispersion, 1.0, 1.0 + dispersion)          # per-channel IOR multiplier: R bends least, B most
    out = None; last = None
    for ch, sc in enumerate(scales):
        def mat_c(P, _sc=sc):                                  # scale ONLY the dielectric IOR for this channel
            vals = material(P)
            if len(vals) == 5:
                a, m, r, e, ior = vals
                return a, m, r, e, np.where(np.asarray(ior) > 1.0, np.asarray(ior) * _sc, ior)
            return vals
        img, st = render_auto(scene, camera, width, height, mat_c, sky=sky, quality=quality,
                              max_bounce=max_bounce, seed=seed, return_stats=True)
        if out is None:
            out = np.zeros_like(img)
        out[..., ch] = img[..., ch]                            # keep this channel from the pass tuned to it
        last = st
    if return_stats:
        last = dict(last); last["passes"] = last.get("passes", 0) * 3
        return out, last
    return out


def add_caustics(img, scene, camera, width, height, light_dir=(-0.35, -0.72, -0.30), receiver_y=-0.9,
                 extent=2.0, ior=1.5, tint=(1.0, 0.98, 0.9), strength=0.6, res=192, n_side=320, seed=0,
                 caustic_sdf=None):
    """Composite genuine CAUSTICS onto the floor of an HDR render (call BEFORE tonemapping). A forward path tracer
    can't find caustic light paths inline (it has no next-event/photon step), so we bring the light to the floor
    separately with holographic_globalillum.caustics: shoot a grid of light rays down `light_dir`, refract them
    through the dielectric, and splat where they land on the receiver plane -- where refracted rays CONVERGE, the
    splat piles up into the bright focused cusp. We then look up that intensity at each floor pixel's world (x,z)
    and ADD the above-average focusing as light. `caustic_sdf` restricts the refractor to just the dielectric
    (e.g. the glass sphere, not the opaque props); the floor pixels are still found from the full `scene`."""
    from holographic_globalillum import caustics
    cmap = caustics(caustic_sdf if caustic_sdf is not None else scene, light_dir=light_dir, receiver_y=receiver_y,
                    extent=extent, res=res, ior=ior, n_side=n_side, seed=seed)   # (res,res), mean 1; peaks=focusing
    eye, dirs = camera.ray_dirs(width, height)                # reconstruct primary hits to find the floor pixels
    D = dirs.reshape(-1, 3); O = np.broadcast_to(eye, D.shape).astype(float)
    hit, t, P = sphere_trace(scene, O, D)
    N = np.zeros_like(P)
    if hit.any():
        N[hit] = sdf_normal(scene, P[hit])
    floor = hit & (N[:, 1] > 0.9) & (np.abs(P[:, 1] - receiver_y) < 0.12)   # up-facing hits at the receiver height
    out = np.asarray(img, float).reshape(-1, 3).copy()
    if floor.any():
        xi = np.clip(((P[floor, 0] + extent) / (2 * extent) * (res - 1)).astype(int), 0, res - 1)
        zi = np.clip(((P[floor, 2] + extent) / (2 * extent) * (res - 1)).astype(int), 0, res - 1)
        gain = np.clip(cmap[zi, xi] - 1.0, 0.0, None)         # only above-average focusing brightens the floor
        out[floor] += strength * gain[:, None] * np.asarray(tint, float)
    return out.reshape(height, width, 3)


def _selftest():
    """Render a tiny two-sphere scene at a very low sample count with and without the pipeline, and check
    that the denoised result is closer to a high-sample reference than the raw noisy render -- i.e. the
    G-buffer + SVGF path actually cleans the grain. Deterministic."""
    # a minimal camera exposing ray_dirs(w,h) -> (eye, dirs), the interface path_trace expects
    class Cam:
        eye = np.array([0.0, 0.4, 3.2])
        def ray_dirs(self, w, h):
            ys, xs = np.mgrid[0:h, 0:w]
            u = (xs / (w - 1) - 0.5) * 1.2
            v = -(ys / (h - 1) - 0.5) * 1.2
            d = np.stack([u, v, -np.ones_like(u)], -1)
            return self.eye, d / np.linalg.norm(d, axis=-1, keepdims=True)

    centers = np.array([[-0.7, 0, 0], [0.7, 0, 0]], float); radii = np.array([0.6, 0.6])
    class Scene:
        def eval(self, P):
            d = np.min(np.linalg.norm(P[..., None, :] - centers, axis=-1) - radii, axis=-1)
            return np.minimum(d, P[..., 1] + 0.9)
    def material(P):
        n = len(P); alb = np.tile([.8, .3, .3], (n, 1)).astype(float)
        left = P[:, 0] < 0; alb[left] = [.3, .4, .85]
        return alb, np.zeros(n), np.full(n, .6), np.zeros((n, 3))
    def sky(D):
        t = np.clip(D[:, 1] * 0.5 + 0.5, 0, 1)[:, None]
        return (1 - t) * np.array([0.9, 0.85, 0.8]) + t * np.array([0.35, 0.5, 0.9])

    cam = Cam(); W = H = 64

    # G-buffer sanity: shapes, and depth is smaller on the spheres than in the sky
    normal, albedo, depth = primary_gbuffer(Scene(), cam, W, H, material, sky=sky)
    assert normal.shape == (H, W, 3) and albedo.shape == (H, W, 3) and depth.shape == (H, W)
    assert depth.min() < depth.max()                          # something was hit, something missed

    def psnr(a, b):
        mse = float(np.mean((np.clip(a, 0, 4) - np.clip(b, 0, 4)) ** 2))
        return 99.0 if mse < 1e-12 else 10.0 * np.log10(16.0 / mse)

    ref = path_trace(Scene(), cam, width=W, height=H, spp=128, max_bounce=3, material=material, sky=sky, seed=7)
    clean, stats = render_denoised(Scene(), cam, W, H, material, sky=sky, spp=6, max_bounce=3, seed=1,
                                   return_stats=True)
    p_noisy = psnr(stats["noisy"], ref)
    p_clean = psnr(clean, ref)
    assert p_clean > p_noisy + 1.0, (p_noisy, p_clean)        # denoising must clearly beat the raw low-spp render

    # firefly clamp removes an injected hot pixel but leaves a broad bright patch alone
    img = np.full((16, 16, 3), 0.2); img[8, 8] = 30.0         # one firefly
    patch = img.copy(); patch[2:6, 2:6] = 5.0                 # a legitimately bright region
    dz = declfirefly(patch, k=3.0)
    assert dz[8, 8].max() < 5.0                                # the lone hot pixel was pulled down
    assert np.allclose(dz[3, 3], patch[3, 3])                  # the bright REGION was left intact

    print("holographic_gbuffer selftest OK: G-buffer traces primary normals/albedo/depth; "
          "render_denoised (adaptive+firefly+SVGF) beat the raw %d-spp render by %.1f dB PSNR-to-reference "
          "(%.1f -> %.1f); firefly clamp removes lone hot pixels but keeps bright regions; %.2fs."
          % (6, p_clean - p_noisy, p_noisy, p_clean, stats["seconds"]))


if __name__ == "__main__":
    _selftest()

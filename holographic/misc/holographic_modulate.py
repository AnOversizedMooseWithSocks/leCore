"""holographic_modulate.py -- the modulate / demodulate primitive (M1) = bind / unbind, spent as bake-and-query.

For a diffuse surface a rendered pixel is a PRODUCT: radiance = albedo * irradiance -- a crisp, structured carrier
(the texture / base colour) times a smooth, expensive residual (the lighting). Splitting them is an UNBIND
(demodulate); putting them back is a BIND (remodulate). The engine already spends this exact move five times as
bake-and-query -- matcompile (fold constant channels), matbake (bake position-dependent), viewlut (bake the
view-dependent part), prt (bake the light transport), radiance (render = query) -- each baking the SMOOTH factor
and multiplying the CRISP one at query. This names the primitive once, and adds the one place it was missing:
DENOISING (M4).

Why it matters for denoising (the measured problem): a denoiser that filters COLOUR directly is stuck choosing
between smoothing noise and smearing texture, so it leaves a noise floor. DEMODULATE first -- divide the albedo
out -- and what's left is the smooth IRRADIANCE, which has no texture to smear, so it denoises cleanly; then
REMODULATE (multiply the crisp albedo back) restores detail undamaged. Standard in real-time path-tracing
denoisers (radiance demodulation); it is the missing quality step here.

Real basis: Ramamoorthi & Hanrahan (SIGGRAPH 2001, rendering as a convolution -> bind); Sloan/Kautz/Snyder
(SIGGRAPH 2002, PRT = bake-and-query); radiance-demodulation denoise/upscale (e.g. Zhuang et al. 2023). NumPy only.
"""
import numpy as np


def demodulate(signal, carrier, eps=1e-3):
    """Split a signal into its SMOOTH residual by dividing out a known CRISP carrier: residual = signal/(carrier+eps).
    This is an UNBIND -- in the FFT/HRR domain the same elementwise division IS unbind. The residual denoises /
    upscales / compresses cleanly because the carrier's high-frequency structure has been removed. Requires a
    KNOWN, non-zero carrier (eps guards the divide); where the carrier is unknown -- a photo, where albedo isn't
    given -- you must RECOVER it first (intrinsic decomposition), which is ill-posed. Kept negative, stated plainly."""
    return np.asarray(signal, float) / (np.asarray(carrier, float) + eps)


def remodulate(residual, carrier):
    """Put the crisp carrier back: signal = residual * carrier. This is a BIND. The round-trip is EXACT where the
    carrier is known: remodulate(demodulate(x, c), c) == x (to the eps guard)."""
    return np.asarray(residual, float) * np.asarray(carrier, float)


def denoise_demodulated(image, normal, albedo, depth, variance=None, levels=5, eps=1e-3, carrier_floor=0.05,
                        **atrous_kw):
    """M4 -- denoise by DEMODULATION (the diffuse-clean path). Divide the albedo out to get the smooth irradiance,
    denoise THAT with the shipped a-trous bilateral, then multiply the albedo back. Because irradiance carries no
    texture, the filter can smooth the noise hard without smearing surface detail -- which is exactly what the
    guide-only filter (albedo as an edge-stop only) cannot do.

    We keep the GEOMETRY edge-stops (normal + depth) so shadow / silhouette edges in the irradiance survive, and
    pass a UNIFORM albedo guide (material edges are already gone after demodulation). `variance` is a colour-space
    guide; demodulation rescales it by ~1/albedo^2, but it is used only as a RELATIVE smoothing strength, so a
    constant rescale doesn't change the result -- pass it through.

    Kept negatives (both measured, stated plainly):
      * Demodulation only pays when the albedo VARIES (texture). On a uniform-albedo surface there is nothing to
        separate, so it is a no-op at best -- use it for textured/diffuse content, not flat matte.
      * The carrier must be non-zero. Where albedo is near-black (the sky / background), dividing it out explodes
        and bleeds into silhouettes, so we DON'T demodulate there: `carrier_floor` marks those pixels and leaves
        the carrier at 1 (a plain guide denoise) so the background stays put.
      * Diffuse only -- radiance = albedo * irradiance is a plain product for diffuse surfaces; for glossy /
        view-dependent surfaces keep the guide-only path (call atrous_bilateral directly)."""
    from holographic.rendering.holographic_svgf import atrous_bilateral
    albedo = np.asarray(albedo, float)
    real = albedo.mean(2) > carrier_floor                              # real surfaces vs near-black background
    carrier = np.where(real[..., None], albedo, 1.0)                   # background carrier=1 -> demod is a no-op there
    irr = demodulate(image, carrier, eps)                             # image / albedo -> smooth irradiance
    irr_clean = atrous_bilateral(irr, normal, np.ones_like(albedo), depth,
                                 levels=levels, variance=variance, **atrous_kw)
    return remodulate(irr_clean, carrier)                            # * albedo -> texture restored undamaged


def _carrier_masked(albedo, carrier_floor):
    """Replace near-black carrier pixels (sky / background, albedo ~ 0) with 1 so dividing them out doesn't explode
    and bleed. Returns the safe carrier."""
    albedo = np.asarray(albedo, float)
    real = albedo.mean(albedo.ndim - 1) > carrier_floor
    return np.where(real[..., None], albedo, 1.0)


def _downsample_to(a, lh, lw):
    """Area-average `a` (H,W,C) down to (lh,lw,C). An anti-aliased shrink -- averaging each output cell's block --
    so the result matches what a low-res render INTEGRATED (not a point sample, which would alias)."""
    a = np.asarray(a, float); H, W = a.shape[:2]
    if H % lh == 0 and W % lw == 0:                                   # clean integer ratio -> exact block average
        fh, fw = H // lh, W // lw
        return a.reshape(lh, fh, lw, fw, -1).mean(axis=(1, 3))
    ys = (np.arange(lh) * H / lh).astype(int); xs = (np.arange(lw) * W / lw).astype(int)
    return a[np.ix_(ys, xs)]                                          # non-integer fallback (nearest block)


def superres_demodulated(low_image, high_albedo, high_normal=None, high_depth=None, eps=1e-3, carrier_floor=0.05):
    """M5 -- UPSCALE by demodulation. The lighting is the expensive thing to render, and it is SMOOTH; the texture
    is the crisp thing, and it is CHEAP (a material lookup, no light transport). So: render the lighting at LOW
    resolution, DEMODULATE its albedo out to get the low-res smooth irradiance, upscale THAT (smooth -> upscales
    cleanly, nothing to alias), then REMODULATE with the CRISP HIGH-RES albedo. You pay low-res lighting cost and
    get high-res detail, because the detail lives in the albedo.

    `low_image` (h,w,3) is the low-res render; `high_albedo` (H,W,3) is the cheap high-res albedo (defines the
    target size and IS the carrier). We demodulate by the high albedo DOWNSAMPLED to the low resolution -- an
    anti-aliased carrier that matches what the low render integrated; demodulating by a point-sampled low albedo
    instead lets high-frequency texture alias the carrier and carries that aliasing through (measured). `high_normal`
    /`high_depth` are accepted for callers that pass the full G-buffer (unused by the plain upscale).

    Real basis: radiance-demodulation super-resolution (Zhuang et al. 2023). Kept negatives: helps where albedo
    VARIES (texture); neutral on flat; near-black background masked; diffuse only."""
    from holographic.rendering.holographic_fsr import easu_upscale
    high_albedo = np.asarray(high_albedo, float)
    Hh, Wh = high_albedo.shape[:2]
    lh, lw = low_image.shape[:2]
    low_carrier = _carrier_masked(_downsample_to(high_albedo, lh, lw), carrier_floor)   # anti-aliased low carrier
    irr_low = demodulate(low_image, low_carrier, eps)                 # low-res smooth irradiance (no texture)
    irr_high = easu_upscale(irr_low, Hh / lh)                         # upscale the SMOOTH irradiance (no clip)
    if irr_high.shape[:2] != (Hh, Wh):                               # guard odd rounding: pad/crop to target
        fixed = np.zeros((Hh, Wh, 3)); hh = min(Hh, irr_high.shape[0]); ww = min(Wh, irr_high.shape[1])
        fixed[:hh, :ww] = irr_high[:hh, :ww]; irr_high = fixed
    return remodulate(irr_high, _carrier_masked(high_albedo, carrier_floor))   # * crisp high-res albedo


def render_demodulated_upscale(sdf, camera, low_wh, high_wh, material_fn, sky=None, quality="medium",
                               max_bounce=3, seed=0, lights=None):
    """M5 convenience: render a high-resolution frame at ~LOW-resolution lighting cost. Render the expensive
    lighting at `low_wh` = (w, h), read the CHEAP high-res G-buffer at `high_wh` (albedo/normal/depth -- a single
    primary ray each, no light transport), and combine them by demodulation (superres_demodulated). The high-res
    detail comes from the albedo, so a textured surface looks sharp at a fraction of the render cost.

    Returns the (H, W, 3) high-res image. Kept negative: the WIN needs the albedo to VARY (texture); on a uniform
    matte surface this is no better than a plain upscale (there's no albedo detail to restore)."""
    from holographic.rendering.holographic_gbuffer import render_auto, primary_gbuffer
    lw, lh = low_wh; hw, hh = high_wh
    low_img = render_auto(sdf, camera, lw, lh, material_fn, sky=sky, quality=quality, max_bounce=max_bounce,
                          seed=seed, lights=lights)                    # LOW-res render: the expensive lighting
    high_n, high_alb, high_z = primary_gbuffer(sdf, camera, hw, hh, material_fn, sky=sky)  # CHEAP high-res G-buffer
    return superres_demodulated(low_img, high_alb, high_n, high_z)     # demod-upscale (carrier from the high albedo)


def _selftest():
    from holographic.rendering.holographic_svgf import atrous_bilateral
    rng = np.random.default_rng(0)

    # (1) the primitive is a clean round-trip where the carrier is known
    x = rng.random((8, 8, 3)); c = 0.2 + rng.random((8, 8, 3))
    assert np.allclose(remodulate(demodulate(x, c, eps=0.0), c), x, atol=1e-9)

    # (2) M4 beats guide-only denoising on a TEXTURED + smoothly-lit surface under noise -- it removes more noise
    #     AND preserves the texture edges better, because it never filters the texture at all.
    H = W = 96
    # crisp carrier: a checkerboard albedo (lots of hard material edges)
    yy, xx = np.mgrid[0:H, 0:W]
    checker = (((xx // 8) + (yy // 8)) % 2).astype(float) * 0.6 + 0.3       # 0.3 / 0.9 tiles
    albedo = np.stack([checker, checker, checker], axis=2)
    # smooth irradiance: a gentle gradient (the "lighting"), no texture
    irr = (0.4 + 0.5 * (xx / W))[..., None] * np.ones(3)
    clean = albedo * irr                                                    # the true radiance (product)
    noisy = clean + rng.normal(0, 0.06, clean.shape)                       # Monte-Carlo-like noise
    normal = np.zeros((H, W, 3)); normal[..., 2] = 1.0                     # flat facing surface
    depth = np.ones((H, W))

    guide = atrous_bilateral(noisy, normal, albedo, depth, levels=5)       # current: guide-only
    demod = denoise_demodulated(noisy, normal, albedo, depth, levels=5)    # M4: demodulated

    # noise removed = closeness to the true clean image (lower error is better)
    err_guide = float(np.abs(guide - clean).mean())
    err_demod = float(np.abs(demod - clean).mean())
    assert err_demod < err_guide, (err_demod, err_guide)

    # texture preserved = the checker's edge contrast survives comparably (demod restores the exact crisp albedo,
    # so its edges stay within a whisker of the guide filter's -- the win is the noise, not an edge penalty)
    def edge_contrast(im):
        row = im[H // 2].mean(1)
        return float(np.abs(np.diff(row)).max())
    assert edge_contrast(demod) >= edge_contrast(guide) * 0.95            # edges comparable (not smeared)

    # (3) M5 super-res demodulation: render lighting LOW-res, upscale the smooth irradiance, remodulate the CRISP
    #     high-res albedo -> recovers texture a plain colour upscale would blur.
    from holographic.rendering.holographic_fsr import easu_upscale
    low_img = albedo[::2, ::2] * irr[::2, ::2]                        # a 2x-smaller "low-res" render
    m5 = superres_demodulated(low_img, albedo)                        # upscale via demodulation -> high-res
    plain = easu_upscale(low_img, 2.0)[:H, :W]                        # naive: upscale the COLOUR directly
    err_m5 = float(np.abs(m5 - clean).mean())
    err_plain = float(np.abs(plain[:H, :W] - clean).mean())
    assert err_m5 < err_plain, (err_m5, err_plain)                    # demod upscale beats plain colour upscale

    print(f"OK: holographic_modulate self-test passed (round-trip exact; M4 err {err_demod:.4f} < guide {err_guide:.4f}; "
          f"M5 err {err_m5:.4f} < plain {err_plain:.4f})")


if __name__ == "__main__":
    _selftest()

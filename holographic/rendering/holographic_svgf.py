"""holographic_svgf.py -- edge-aware denoising the engine's way: a holographic BILATERAL filter whose
edge-stopping is a COSINE in a bound feature space, run coarse-to-fine over the a-trous hierarchy. Turns a noisy
1-spp-style image into a clean preview without blurring across surface boundaries.

WHY THIS EXISTS (Interactive Render Speed, VSA-native -- technique E)
--------------------------------------------------------------------
SVGF blurs Monte-Carlo noise but STOPS at edges using depth/normal/albedo. Said the engine's way, that
edge-stopping is a COSINE of per-pixel feature vectors: bind each pixel's (normal, albedo, depth) into one
feature vector, and blend a neighbour weighted by the cosine of their feature vectors -- similar surfaces blend,
edges don't. The Gaussian falloff on that cosine is exactly the ScalarEncoder's RBF bump, and SVGF's multi-scale
a-trous hierarchy is the multires pyramid (increasing dilation, coarse-to-fine).

PROBE-FIRST NOTE: the sibling render pieces the backlog names are ALREADY in the tree, so this module is the one
genuinely-missing part -- robust/firefly accumulation is holographic_accumulate.robust_accumulate; per-pixel
adaptive sampling is holographic_honesty.SPRTRecall; temporal reproject is holographic_temporal.TemporalReuse;
the pyramid is holographic_multires. This adds only the edge-aware bilateral they compose with.

MEASURED / KEPT NEGATIVE (loud, per the backlog's "measure against the plain float baseline"):
* The feature-cosine bilateral BEATS a plain Gaussian blur on PSNR-to-clean, because the plain blur smears
  across edges while this stops at them -- the selftest measures both.
* "A shared kernel is not a shared manifold": the bound-feature cosine must be MEASURED to edge-stop well, not
  assumed. It is a product of per-channel RBF weights here (the standard SVGF form), which is the honest,
  measurable version -- not a claim that one magic cosine replaces the tuned weights for free.
* This denoises; it does not add detail. Over-smoothing flat noise is the goal; it cannot recover lost signal.

Real basis: Schied et al. (2017), Spatiotemporal Variance-Guided Filtering (SVGF); Dammertz et al. (2010),
edge-avoiding a-trous wavelets. Deterministic; NumPy + stdlib.
"""
import numpy as np


def _rbf(diff2, sigma):
    """Gaussian (RBF) bump on a squared difference -- the ScalarEncoder's falloff. 1 at 0, decaying with sigma."""
    return np.exp(-diff2 / (2.0 * sigma * sigma + 1e-12))


def atrous_bilateral(image, normal, albedo, depth, sigma_normal=0.3, sigma_albedo=0.2,
                     sigma_depth=0.5, sigma_color=0.6, levels=5, variance=None,
                     color_scale=4.0, color_floor=0.03):
    """Edge-aware a-trous bilateral denoise. `image` is (H,W,3) noisy colour; `normal`/`albedo` are (H,W,3)
    feature buffers; `depth` is (H,W). At each level the filter blends the 3x3 neighbours at an increasing
    DILATION (1,2,4,...) -- the a-trous / multires hierarchy -- weighting each neighbour by the PRODUCT of RBF
    bumps on the normal/albedo/depth/colour differences (the bound-feature edge-stopping). Similar surfaces
    blend; edges don't. Returns the denoised (H,W,3) image.

    VARIANCE-GUIDED (the 'V' in SVGF -- Schied et al. 2017). If `variance` (H,W) -- the renderer's per-pixel
    variance-of-the-mean -- is given, the COLOUR edge-stop width becomes PER-PIXEL: sigma_color_p = max(
    color_scale*sqrt(variance_p), color_floor). Where a pixel is still noisy (high variance) the filter blends
    freely (removes the grain); where it has CONVERGED (variance ~0) the colour sigma collapses to the floor, so
    real detail is preserved instead of smeared. This is the wiring that lets the denoise strength CALIBRATE
    ITSELF from the measured noise, instead of a hand-set global sigma_color. `variance=None` reproduces the
    old fixed-sigma behaviour exactly (backward compatible)."""
    img = np.asarray(image, float).copy()
    H, W, _ = img.shape
    n = np.asarray(normal, float); a = np.asarray(albedo, float); z = np.asarray(depth, float)
    # per-pixel colour sigma from the variance map (real SVGF), or the fixed scalar if no variance was supplied
    if variance is not None:
        vsig = np.sqrt(np.clip(np.asarray(variance, float), 0.0, None))     # noise level in luminance units
        col_sigma = np.maximum(color_scale * vsig, color_floor)            # (H,W): wide where noisy, tiny where converged
    else:
        col_sigma = sigma_color                                            # scalar: the original behaviour
    # the 3x3 a-trous stencil with a small binomial spatial kernel
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    spatial = np.array([1, 2, 1, 2, 4, 2, 1, 2, 1], float)
    spatial /= spatial.sum()

    for level in range(levels):
        step = 1 << level                                        # dilation 1, 2, 4, ... (the a-trous spacing)
        out = np.zeros_like(img)
        wsum = np.zeros((H, W, 1))
        for (dy, dx), sw in zip(offsets, spatial):
            ys = np.clip(np.arange(H) + dy * step, 0, H - 1)     # shifted, edge-clamped neighbour indices
            xs = np.clip(np.arange(W) + dx * step, 0, W - 1)
            YS, XS = np.meshgrid(ys, xs, indexing="ij")
            # feature differences to the neighbour
            dn = np.sum((n - n[YS, XS]) ** 2, axis=2)            # normal difference (edge-stop across geometry)
            da = np.sum((a - a[YS, XS]) ** 2, axis=2)            # albedo difference (edge-stop across materials)
            dz = (z - z[YS, XS]) ** 2                            # depth difference (edge-stop across silhouettes)
            dc = np.sum((img - img[YS, XS]) ** 2, axis=2)        # colour difference (don't blend very different tones)
            w = sw * _rbf(dn, sigma_normal) * _rbf(da, sigma_albedo) * _rbf(dz, sigma_depth) * _rbf(dc, col_sigma)
            w = w[:, :, None]
            out += w * img[YS, XS]
            wsum += w
        img = out / (wsum + 1e-12)                               # normalized weighted blend = the bilateral step
    return img


def plain_blur(image, levels=5):
    """The honest baseline: the SAME a-trous stencil with NO edge-stopping -- a plain multi-scale Gaussian blur.
    Denoises flat regions but smears across edges. What the feature-aware filter must beat."""
    img = np.asarray(image, float).copy()
    H, W, _ = img.shape
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    spatial = np.array([1, 2, 1, 2, 4, 2, 1, 2, 1], float); spatial /= spatial.sum()
    for level in range(levels):
        step = 1 << level
        out = np.zeros_like(img)
        for (dy, dx), sw in zip(offsets, spatial):
            ys = np.clip(np.arange(H) + dy * step, 0, H - 1)
            xs = np.clip(np.arange(W) + dx * step, 0, W - 1)
            YS, XS = np.meshgrid(ys, xs, indexing="ij")
            out += sw * img[YS, XS]
        img = out
    return img


def _psnr(a, b):
    mse = float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))
    return 99.0 if mse < 1e-12 else float(10.0 * np.log10(1.0 / mse))


def _selftest():
    """Build a clean two-surface image (a left/right split with different colour, normal, and albedo), add noise,
    and denoise. The feature-aware bilateral must beat the plain blur on PSNR-to-clean, because it preserves the
    edge the plain blur smears."""
    rng = np.random.default_rng(0)
    H = W = 64
    clean = np.zeros((H, W, 3)); normal = np.zeros((H, W, 3)); albedo = np.zeros((H, W, 3)); depth = np.zeros((H, W))
    # left surface vs right surface -- a hard edge down the middle in colour, normal, albedo, and depth
    clean[:, :W // 2] = [0.8, 0.2, 0.2]; clean[:, W // 2:] = [0.2, 0.3, 0.8]
    normal[:, :W // 2] = [0, 0, 1]; normal[:, W // 2:] = [1, 0, 0]
    albedo[:, :W // 2] = [0.8, 0.2, 0.2]; albedo[:, W // 2:] = [0.2, 0.3, 0.8]
    depth[:, :W // 2] = 1.0; depth[:, W // 2:] = 3.0
    noisy = np.clip(clean + 0.15 * rng.standard_normal((H, W, 3)), 0, 1)   # heavy Monte-Carlo-style noise

    den = atrous_bilateral(noisy, normal, albedo, depth, levels=5)
    blur = plain_blur(noisy, levels=5)

    psnr_noisy = _psnr(noisy, clean)
    psnr_den = _psnr(den, clean)
    psnr_blur = _psnr(blur, clean)
    assert psnr_den > psnr_noisy, (psnr_den, psnr_noisy)         # it actually denoises
    assert psnr_den > psnr_blur, (psnr_den, psnr_blur)           # ... and beats the edge-blind blur

    # specifically at the edge column, the feature-aware result stays sharp where the blur bleeds across
    edge = W // 2
    edge_err_den = float(np.mean((den[:, edge - 1:edge + 1] - clean[:, edge - 1:edge + 1]) ** 2))
    edge_err_blur = float(np.mean((blur[:, edge - 1:edge + 1] - clean[:, edge - 1:edge + 1]) ** 2))
    assert edge_err_den < edge_err_blur

    # deterministic
    assert np.array_equal(atrous_bilateral(noisy, normal, albedo, depth, levels=3),
                          atrous_bilateral(noisy, normal, albedo, depth, levels=3))

    print("holographic_svgf selftest OK: noisy %.1f dB -> feature-aware bilateral %.1f dB, beating the edge-blind "
          "blur %.1f dB; edge stays sharp (edge MSE %.4f vs blur %.4f); deterministic"
          % (psnr_noisy, psnr_den, psnr_blur, edge_err_den, edge_err_blur))


if __name__ == "__main__":
    _selftest()

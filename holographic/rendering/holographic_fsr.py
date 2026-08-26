"""holographic_fsr.py -- FSR1-style SPATIAL UPSCALER: EASU + RCAS (inverse-rendering IR12).

A post-process UPSCALE stage: take a low-resolution render up to display resolution edge-adaptively, then sharpen --
so you can render at (say) 1080p and present at 4K. AMD FidelityFX Super Resolution 1.0 is two passes, and one of
them already ships:

  * EASU (Edge-Adaptive Spatial Upsampling) -- the genuinely-new piece here. FSR1's EASU is a Lanczos variant
    (Duchon 1979) that steers by local gradient reversals so it upsamples ALONG edges, not across them. This module
    builds a separable Lanczos-2 upscale (sharper than the plain bilinear postfx.resample it exists to beat) with an
    ANTI-RINGING clamp: each output pixel is clamped to the [min, max] of its low-res neighbourhood, which kills the
    Lanczos overshoot exactly where gradients reverse (edges). Cheap, readable, deterministic.
  * RCAS (Robust Contrast-Adaptive Sharpening) -- ALREADY SHIPPED as holographic_postfx.sharpen: Van Cittert
    negative-lobe sharpening whose own kept-negative is literally "over-sharpening amplifies high-frequency noise, so
    stop at the noise floor" -- which is exactly RCAS's design goal. We wire it as the second pass, not rebuild it.

KEPT NEGATIVES (loud): classical spatial upscaling is BELOW learned (DLSS/XeSS) -- it CANNOT invent detail that is
not in the low-res input; it reconstructs, it does not hallucinate. EASU's upscaling artifacts get MULTIPLIED by the
RCAS sharpen (so sharpness is a knob, not a free win). And this EASU is an honest Lanczos-with-anti-ringing
approximation of FSR1's full 12-tap gradient-reversal kernel -- the same class (edge-aware Lanczos), not a
byte-for-byte port. A good, cheap, deterministic upscaler -- not a magic one. NumPy + stdlib only.
"""
import numpy as np


def _lanczos(x, a=2.0):
    """The Lanczos-a windowed-sinc kernel: sinc(x)*sinc(x/a) inside |x|<a, else 0; 1 at x=0."""
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    nz = x != 0
    px = np.pi * x
    out[nz] = a * np.sin(px[nz]) * np.sin(px[nz] / a) / (px[nz] ** 2)
    out[~nz] = 1.0
    out[np.abs(x) >= a] = 0.0
    return out


def _resample_matrix(in_len, out_len, a=2):
    """A (out_len, in_len) Lanczos resampling matrix: row o samples the input at the output pixel's centre, weighting
    input pixels by the Lanczos kernel. Rows are normalized (partition of unity) so flat regions are preserved."""
    pos = (np.arange(out_len) + 0.5) * in_len / out_len - 0.5      # input coordinate of each output centre
    i = np.arange(in_len)[None, :]
    W = _lanczos(pos[:, None] - i, a)
    W /= (W.sum(axis=1, keepdims=True) + 1e-12)
    return W


def _resample_axis(img, out_len, axis, a=2):
    """Resample `img` along one axis to out_len via the Lanczos matrix (a readable dense matmul -- fine for the
    modest sizes a preview upscaler runs at)."""
    W = _resample_matrix(img.shape[axis], out_len, a)
    moved = np.moveaxis(img, axis, 0)
    out = np.tensordot(W, moved, axes=([1], [0]))
    return np.moveaxis(out, 0, axis)


def lanczos_upscale(img, out_hw, a=2):
    """Separable Lanczos-a upscale to an (out_h, out_w) target -- the EASU base (sharper than bilinear)."""
    oh, ow = out_hw
    up = _resample_axis(np.asarray(img, float), oh, 0, a)
    return _resample_axis(up, ow, 1, a)


def _nearest_upscale(img, out_hw):
    """Nearest-neighbour upscale to (out_h, out_w) -- used to carry each output pixel's source neighbourhood."""
    H, W = img.shape[:2]
    oh, ow = out_hw
    oy = np.clip((np.arange(oh) * H / oh).astype(int), 0, H - 1)
    ox = np.clip((np.arange(ow) * W / ow).astype(int), 0, W - 1)
    return img[oy][:, ox]


def _minmax3(img, op):
    """A 3x3 min or max filter (op = np.minimum / np.maximum), reflect-padded -- the low-res neighbourhood extent."""
    a = np.asarray(img, float)
    acc = a.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            acc = op(acc, np.roll(np.roll(a, dy, axis=0), dx, axis=1))
    return acc


def easu_upscale(img, scale=2.0):
    """Edge-Adaptive Spatial Upsampling: a Lanczos upscale with an ANTI-RINGING clamp. The Lanczos pass is sharper
    than bilinear; the clamp bounds each output to the [min, max] of its low-res neighbourhood, suppressing the
    Lanczos overshoot at edges (where the gradient reverses) -- so edges stay crisp without ringing."""
    img = np.asarray(img, float)
    H, W = img.shape[:2]
    out_hw = (int(round(H * scale)), int(round(W * scale)))
    up = lanczos_upscale(img, out_hw)
    lo = _nearest_upscale(_minmax3(img, np.minimum), out_hw)       # per-output neighbourhood min
    hi = _nearest_upscale(_minmax3(img, np.maximum), out_hw)       # per-output neighbourhood max
    return np.clip(up, lo, hi)                                     # anti-ringing


def fsr_upscale(img, scale=2.0, sharpness=0.4):
    """FSR1-style upscale: EASU (edge-adaptive) then RCAS (the shipped noise-aware sharpen). `sharpness` in [0,1];
    0 skips the sharpen. Reconstructs a display-res image from a low-res render -- it cannot invent absent detail."""
    up = easu_upscale(img, scale)
    if sharpness > 0:
        from holographic.rendering.holographic_postfx import sharpen
        up = np.clip(sharpen(up, amount=sharpness), 0.0, 1.0)
    return up


def _psnr(a, b):
    mse = float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))
    return 99.0 if mse < 1e-12 else float(10.0 * np.log10(1.0 / mse))


def _edge_energy(img):
    """Mean gradient magnitude -- a proxy for edge sharpness (higher = crisper)."""
    g = img.mean(axis=-1) if img.ndim == 3 else img
    gy, gx = np.gradient(g)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def _box_downscale(img, factor=2):
    """Average factor x factor blocks -- a simple, honest downscaler to build the low-res input for the round-trip."""
    H, W = img.shape[:2]
    h, w = H - H % factor, W - W % factor
    a = img[:h, :w]
    a = a.reshape(h // factor, factor, w // factor, factor, -1) if a.ndim == 3 else a.reshape(h // factor, factor, w // factor, factor)
    return a.mean(axis=(1, 3))


def _selftest():
    """On a downscale->upscale round-trip, EASU beats plain bilinear on PSNR-to-native and on edge sharpness; the
    anti-ringing clamp keeps EASU inside the source range (no overshoot); RCAS (the shipped sharpen) adds crispness;
    a flat image is preserved; deterministic."""
    from holographic.rendering.holographic_postfx import resample

    # a structured native image: diagonal stripes + a couple of blocks (edges at many orientations)
    yy, xx = np.mgrid[0:96, 0:96].astype(float)
    native = 0.5 + 0.4 * np.sign(np.sin((xx + yy) / 5.0))
    native[20:50, 20:50] = 0.9
    native[60:85, 55:88] = 0.15
    native = np.clip(np.stack([native, native, native], axis=-1), 0, 1)

    low = _box_downscale(native, 2)                                # the low-res render
    out_hw = native.shape[:2]

    bilinear = np.clip(resample(low, 2.0), 0, 1)
    # match bilinear to native size if resample rounded differently
    bilinear = bilinear[:out_hw[0], :out_hw[1]]
    easu = easu_upscale(low, 2.0)[:out_hw[0], :out_hw[1]]
    fsr = fsr_upscale(low, 2.0, sharpness=0.4)[:out_hw[0], :out_hw[1]]

    p_bil, p_easu = _psnr(bilinear, native), _psnr(easu, native)
    assert p_easu > p_bil                                          # EASU beats bilinear on PSNR
    assert _edge_energy(easu) > _edge_energy(bilinear)             # and on edge sharpness

    # anti-ringing: EASU stays within a sane range (no wild Lanczos overshoot)
    assert easu.min() >= -0.02 and easu.max() <= 1.02

    # RCAS (sharpen) adds crispness over EASU alone
    assert _edge_energy(fsr) > _edge_energy(easu)

    # a flat image is preserved (partition of unity)
    flat = np.full((32, 32, 3), 0.4)
    assert np.allclose(easu_upscale(flat, 2.0), 0.4, atol=1e-6)

    # deterministic
    assert np.array_equal(easu_upscale(low, 2.0), easu_upscale(low, 2.0))

    print("holographic_fsr selftest OK: on a 2x downscale->upscale round-trip, EASU beats bilinear on PSNR "
          "(%.2f vs %.2f dB) and on edge sharpness; the anti-ringing clamp holds it in range (no overshoot); RCAS "
          "(the shipped sharpen) adds crispness; a flat image is preserved; deterministic" % (p_easu, p_bil))


if __name__ == "__main__":
    _selftest()

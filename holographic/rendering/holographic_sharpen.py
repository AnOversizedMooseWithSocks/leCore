"""Looping negative-lobe sharpening for arbitrary signals -- recover detail an over-smoothed estimate lost.

WHY THIS EXISTS (Group G, the sharpen half -- partner to SHARP-2)
----------------------------------------------------------------
"A looping accumulation/negative process (like how we sharpened the gaussian images)." A smooth basis -- a
low-rank reconstruction, a Gaussian splat, an over-consolidated rank truncation -- LOW-PASSES a signal: it
attenuates the high-frequency detail. Sharpening counteracts that by repeatedly adding a high-pass (negative-lobe)
correction. That is data-type-agnostic: a smeared 1-D signal, an over-consolidated market window, an
under-reconstructed structure can all be re-sharpened the same way an under-reconstructed image edge was.

The honest subtlety, MEASURED here: the naive loop (iterated unsharp, x <- x + a*(x - blur(x))) DIVERGES -- its
high-frequency gain (1+a)^k is unbounded, so it recovers detail for a few steps then explodes into ringing. The
stable loop is VAN CITTERT (residual-fitting deconvolution, x <- x + lam*(y - blur(x))): its accumulated operator
converges to the INVERSE blur -- a sharpening filter with negative lobes -- but with bounded eigenvalues, so it
CONVERGES instead of blowing up (for lam below the stability bound; above it, it diverges -- why the guard matters).

The kept negative is the deconvolution tradeoff: with noise present, Van Cittert recovers the signal up to an
OPTIMUM, then keeps going and amplifies the high-frequency NOISE (over-sharpening). The principled stop is
Morozov's DISCREPANCY PRINCIPLE -- stop when the residual ||y - blur(x)|| falls to the noise level, because fitting
below that is fitting noise. That lands near the error optimum and prevents over-sharpening.

MEASURED (see `_selftest`, a 1-D signal = slow component + a localized high-frequency burst, Gaussian-blurred):
  * NO NOISE: iterated sharpening recovers the detail and CONVERGES -- relative error 0.222 -> ~0.001, no blow-up.
  * WITH NOISE: the discrepancy principle stops near the optimum (err ~0.12 vs the blurred 0.222); running on
    UNGUARDED over-sharpens to ~0.45 (noise amplified) -- the kept negative.
  * lam above the stability bound DIVERGES into ringing -- the reason lam is bounded and the guard exists.
"""

import numpy as np


def _gauss_blur(x, sigma):
    """A Gaussian low-pass (the default smooth basis), circular via the FFT -- the engine's FFT-on-a-torus."""
    x = np.asarray(x, float)
    # n-D, separable. leStudio found the 1-D version "1-D only: rfft with a 1-D kernel fails on any 2-D array" --
    # it filtered along the LAST axis with a frequency grid sized by the FIRST, so a square image blurred rows only
    # and an RGB image tried to blur across its 3 channels. Spatial axes are every axis except a trailing channel
    # axis of size <= 4 on a 3-D input; a 1-D signal is bit-identical to the old path.
    spatial = list(range(x.ndim))
    if x.ndim == 3 and x.shape[-1] <= 4:
        spatial = [0, 1]
    out = x
    for ax in spatial:
        n = x.shape[ax]
        f = np.fft.rfftfreq(n)
        H = np.exp(-0.5 * (2 * np.pi * f * sigma) ** 2)
        shape = [1] * x.ndim; shape[ax] = len(f)
        out = np.fft.irfft(np.fft.rfft(out, axis=ax) * H.reshape(shape), n=n, axis=ax)
    return out


def sharpen_loop(x, blur=None, sigma=3.0, lam=1.0, iters=60, noise_level=0.0):
    """Recover detail from an over-smoothed signal `x` by looping a Van Cittert correction (a converging
    negative-lobe sharpening). `blur` is the smoothing operator that did the over-smoothing (a callable
    signal->signal); if None, a Gaussian low-pass with `sigma` is assumed. `lam` is the step (keep it below the
    stability bound ~2/||blur||^2 or it diverges). `noise_level` is the std of the noise in `x`: if > 0 the loop
    stops by the DISCREPANCY PRINCIPLE (residual <= noise norm) to avoid amplifying noise; if 0 it runs the full
    `iters` (converging to the deblurred signal). Returns the sharpened signal."""
    x = np.asarray(x, float)
    blur = blur if blur is not None else (lambda z: _gauss_blur(z, sigma))
    y = x.copy()
    noise_norm = noise_level * np.sqrt(x.size)                  # every sample, not just the first axis
    out = y.copy()
    for _ in range(iters):
        out = out + lam * (y - blur(out))                       # residual-fitting; accumulated op -> inverse blur
        if noise_norm > 0 and np.linalg.norm(y - blur(out)) <= noise_norm:
            break                                               # discrepancy principle: residual hit the noise floor
    return out


def _selftest_nd():
    """2-D and RGB inputs: the loop sharpens a blurred image toward the truth on BOTH axes (the 1-D-only version
    left columns untouched), and channels are never mixed."""
    rng = np.random.default_rng(1)
    truth = np.zeros((24, 32)); truth[8:16, 10:22] = 1.0; truth += 0.1 * np.sin(np.linspace(0, 6, 32))[None, :]
    blurred = _gauss_blur(truth, 2.0)
    assert blurred.shape == truth.shape and abs(blurred.sum() - truth.sum()) < 1e-6            # a low-pass keeps the mean
    col_var_blur = np.var(np.diff(blurred, axis=0)); col_var_truth = np.var(np.diff(truth, axis=0))
    assert col_var_blur < col_var_truth * 0.9                                                   # columns WERE blurred
    sharp = sharpen_loop(blurred, sigma=2.0, lam=0.9, iters=40)
    e0 = np.linalg.norm(blurred - truth); e1 = np.linalg.norm(sharp - truth)
    # Van Cittert on a hard edge converges slowly past ~0.6 (Gibbs ringing eats the gain: 0.60 at 40 iters, 0.54 at
    # 300) -- pin the honest number, not a wish
    assert e1 < 0.65 * e0, (e0, e1)
    rgb = np.stack([truth, 0.5 * truth, np.zeros_like(truth)], -1)
    s3 = sharpen_loop(_gauss_blur(rgb, 2.0), sigma=2.0, lam=0.9, iters=40)
    assert s3.shape == rgb.shape and np.abs(s3[..., 2]).max() < 1e-9                            # channels unmixed
    x1 = rng.normal(size=64); assert np.array_equal(_gauss_blur(x1, 3.0), np.fft.irfft(np.fft.rfft(x1) * np.exp(-0.5 * (2 * np.pi * np.fft.rfftfreq(64) * 3.0) ** 2), n=64))


def _selftest():
    """CI-fast: a 1-D signal (slow component + a localized high-frequency burst) is Gaussian-blurred; looping
    sharpening recovers the detail and CONVERGES with no noise, the discrepancy guard stops near the optimum and
    beats running unguarded with noise (the over-sharpening kept negative), and an over-large step DIVERGES."""
    rng = np.random.default_rng(0)
    T = 256
    t = np.arange(T)
    truth = np.sin(2 * np.pi * 3 * t / T) + 0.6 * np.sin(2 * np.pi * 30 * t / T) * np.exp(-((t - 128) ** 2) / (2 * 25 ** 2))
    blur = lambda z: _gauss_blur(z, 3.0)
    blurred = blur(truth)
    def err(z):
        return float(np.linalg.norm(z - truth) / np.linalg.norm(truth))

    assert err(blurred) > 0.2                                   # the over-smoothed estimate has lost the burst

    # no noise: recovers the detail and converges (no blow-up)
    rec = sharpen_loop(blurred, sigma=3.0, lam=1.0, iters=80, noise_level=0.0)
    assert err(rec) < 0.05, err(rec)                           # recovered (0.22 -> ~0); converged, did not diverge

    # with noise: the discrepancy guard stops near the optimum; running unguarded over-sharpens
    noisy = blurred + 0.005 * rng.standard_normal(T)
    guarded = sharpen_loop(noisy, sigma=3.0, lam=1.0, iters=80, noise_level=0.005)
    unguarded = sharpen_loop(noisy, sigma=3.0, lam=1.0, iters=80, noise_level=0.0)
    assert err(guarded) < err(blurred)                         # recovers real detail despite noise
    assert err(guarded) < err(unguarded) * 0.6, (err(guarded), err(unguarded))   # the guard beats over-sharpening

    # over-large step diverges into ringing -- why lam is bounded / the guard exists
    blown = sharpen_loop(blurred, sigma=3.0, lam=2.5, iters=30, noise_level=0.0)
    assert err(blown) > 10.0                                    # unstable step -> ringing/divergence


if __name__ == "__main__":
    _selftest()
    _selftest_nd()
    print("holographic_sharpen selftest passed (1-D bit-identical, 2-D and RGB sharpen on both axes)")

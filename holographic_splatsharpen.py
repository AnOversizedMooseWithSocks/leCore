"""C4 probe (cross-cutting: XDATA-3 negative-lobe sharpening -> splat/archive reconstruction). KEPT NEGATIVE.

THE PROPOSAL (Milanfar's seat, RED / Van Cittert): a splat render is a sum of smooth Gaussians and is therefore
over-smoothed (splat_aniso's own kept negative says a few Gaussians cannot hold high frequency). So -- the
reasoning went -- post-process the render with the looping negative-lobe sharpener (XDATA-3) to recover the edge
detail, guarded by the discrepancy principle. High upside on paper.

THE MEASURED ANSWER: it does NOT work, and the reason is structural, not a tuning failure. Van Cittert
deconvolution assumes the smooth signal is blur(truth) -- a CONVOLUTION of the thing you want. A splat render is
not that. It is a sparse sum of Gaussians, which is approximately blur(the splat CENTRES) -- a handful of spikes,
not the original image. So deconvolving a splat render drives it toward those centres (spikes/ringing), NOT
toward the edges the splat basis discarded. Sharpening the render at every sigma/iteration setting tested makes
it WORSE, not better (relative error rises ~5-8%).

THE DECISIVE CONTROL (why this is a real finding, not a broken sharpener): the SAME 2-D Van Cittert sharpener,
applied to a GENUINE Gaussian blur of the truth, RECOVERS ~42% of the error (0.37 -> 0.22). The machinery works.
The negative is specifically that a splat render is sum-of-Gaussians(centres), not blur(truth).

THE LESSON (and why it belongs on record): this is the image-domain twin of the ACCUM-1 jitter negative and the
generate_vector bare-codebook negative -- you cannot manufacture detail that was never stored. A lossy smooth
basis (splats) THREW AWAY the high frequency; no negative-lobe loop recovers information that is not in the
render. Sharpening helps a signal that was genuinely low-passed; it cannot un-throw-away a lossy approximation.

No faculty, no tour line -- the finding is the negative. The 2-D Van Cittert here is the vehicle for the control.
"""

import numpy as np


def gauss_blur2(x, sigma):
    """A 2-D Gaussian low-pass, circular via the FFT on both axes (the engine's FFT-on-a-torus, in 2-D)."""
    x = np.asarray(x, float)
    H, W = x.shape
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    Hf = np.exp(-0.5 * (2 * np.pi * sigma) ** 2 * (fy ** 2 + fx ** 2))
    return np.fft.irfft2(np.fft.rfft2(x) * Hf, s=(H, W))


def vc_sharpen2(render, sigma, lam=1.0, iters=60, noise_norm=0.0):
    """The 2-D extension of sharpen_loop: recover detail from an over-smoothed image by looping a Van Cittert
    correction (out <- out + lam*(render - blur(out))), a converging negative-lobe sharpening. With noise_norm>0
    it stops by the discrepancy principle. Works on a GENUINE blur; see the module note for why it does NOT
    recover a splat render's lost detail."""
    y = np.asarray(render, float)
    out = y.copy()
    for _ in range(iters):
        out = out + lam * (y - gauss_blur2(out, sigma))
        if noise_norm > 0 and np.linalg.norm(y - gauss_blur2(out, sigma)) <= noise_norm:
            break
    return out


def _selftest():
    """CI-fast: records the C4 negative airtight. (1) CONTROL -- the 2-D Van Cittert sharpener recovers detail
    from a GENUINE Gaussian blur of the truth (the machinery works). (2) NEGATIVE -- sharpening a SPLAT RENDER of
    the same truth does NOT improve it at any setting (the render is sum-of-Gaussians(centres), not blur(truth),
    so deconvolution recovers spikes, not the discarded edges -- you cannot recover what the lossy basis threw
    away)."""
    from holographic_splat import splat_fit, splat_render

    T = np.zeros((48, 48))
    T[8:20, 8:24] = 1.0
    T[28:40, 26:42] = 0.7
    T[24:30, 10:16] = 0.5                                            # sharp-edged boxes: high-frequency content

    def err(z):
        return float(np.linalg.norm(z - T) / np.linalg.norm(T))

    # (1) CONTROL: the sharpener recovers detail from a GENUINE convolutional blur
    sig = 2.0
    blurred = gauss_blur2(T, sig)
    recovered = vc_sharpen2(blurred, sigma=sig, lam=1.0, iters=60)
    assert err(recovered) < err(blurred) * 0.7, (err(recovered), err(blurred))   # recovers >=30% -> machinery works

    # (2) NEGATIVE: sharpening a SPLAT RENDER does not help at ANY setting tested
    R = splat_render(splat_fit(T, 25, refit=True), T.shape)
    e_raw = err(R)
    best = min(err(vc_sharpen2(R, sigma=sg, lam=1.0, iters=it))
               for sg in (0.8, 1.0, 1.5, 2.0, 3.0) for it in (10, 25, 50))
    assert best >= e_raw * 0.99, (best, e_raw)                      # nothing beats the raw render -> detail not recoverable


if __name__ == "__main__":
    _selftest()
    print("holographic_splatsharpen C4 negative selftest passed (sharpening cannot un-throw-away discarded detail)")

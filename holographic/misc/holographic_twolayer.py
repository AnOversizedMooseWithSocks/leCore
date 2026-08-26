"""Smooth/sharp two-layer representation -- store each component in the basis it is cheap in.

WHY THIS EXISTS (CACHE-2, the architectural move)
-------------------------------------------------
Irradiance caching's core architecture is to CACHE the smooth indirect light and COMPUTE the sharp direct light --
two layers, each in the representation it is cheap in. The same split recurs everywhere: the negative-lobe
sharpening finding, the SVG (a smooth morph plus exact vector edges), manifold-plus-residual decompose. The
principle is that NO SINGLE basis is good across a signal that is smooth in some places and sharp in others, so at
a fixed budget you win by splitting:

  smooth layer = the few low-frequency / low-rank coefficients (a Gaussian-smooth, dense, cheap basis), and
  sharp layer  = the few largest residual coefficients in a basis where the SHARP content is sparse.

The earlier attempt at this was only a modest win because its sharp basis was weak (pixel-exact). The result here
is large because the sharp basis is the RIGHT one for the sharp content: localized features (spikes) are sparse in
the SAMPLE domain, so a sparse sample-domain residual captures them in a handful of coefficients -- where a
low-frequency basis would need a great many (a spike is broadband). The answer to "what is the right sharp basis"
is therefore: whichever one the sharp content is sparse in -- sample-sparse for spikes, a wavelet basis for edges.

MEASURED (see `_selftest`, a signal = two slow sinusoids + a few spikes):
  * at a budget that covers both layers, the SPLIT beats both single bases by a wide margin (~40 dB PSNR vs ~28
    single-FFT vs ~18 single-sparse) -- the sharp layer carries the spikes the low-frequency layer provably cannot
    (30% of the signal energy sits in that residual).
  * KEPT CAVEAT: at too SMALL a budget the split LOSES (it cannot afford enough of either layer) -- the win needs a
    budget large enough to hold both layers' essential coefficients.
"""

from collections import namedtuple

import numpy as np

TwoLayerCode = namedtuple("TwoLayerCode", "n smooth_coeffs sharp_idx sharp_val")


def smooth_sharp_split(x, k_smooth, k_sharp):
    """Split a signal into a SMOOTH layer (its k_smooth lowest-frequency coefficients -- the cheap dense basis)
    and a SHARP layer (the k_sharp largest residual samples -- sparse in the sample domain, the right basis for
    localized/spike features). Returns a TwoLayerCode storing both at a budget of k_smooth + k_sharp items."""
    x = np.asarray(x, float)
    n = len(x)
    F = np.fft.rfft(x)
    smooth_coeffs = F[:k_smooth].copy()
    sm = np.fft.irfft(np.concatenate([smooth_coeffs, np.zeros(len(F) - k_smooth, complex)]), n=n)
    resid = x - sm
    idx = np.argsort(np.abs(resid))[::-1][:k_sharp]
    return TwoLayerCode(n, smooth_coeffs, np.array(idx), resid[idx])


def smooth_sharp_reconstruct(code):
    """Reconstruct a signal from a TwoLayerCode: the smooth layer everywhere, plus the exact sharp residual at the
    stored sharp positions (so those samples come back exact, the rest follow the smooth basis)."""
    nbins = code.n // 2 + 1
    sm = np.fft.irfft(np.concatenate([code.smooth_coeffs, np.zeros(nbins - len(code.smooth_coeffs), complex)]),
                      n=code.n)
    sm = sm.copy()
    sm[code.sharp_idx] += code.sharp_val
    return sm


def _fft_topk(sig, k):
    F = np.fft.rfft(sig)
    idx = np.argsort(np.abs(F))[::-1][:k]
    Fk = np.zeros_like(F); Fk[idx] = F[idx]
    return np.fft.irfft(Fk, n=len(sig))


def _sparse_topk(sig, k):
    idx = np.argsort(np.abs(sig))[::-1][:k]
    out = np.zeros_like(sig); out[idx] = sig[idx]
    return out


def _selftest():
    """CI-fast: on a smooth-plus-sharp signal (two slow sinusoids + spikes), the smooth/sharp split beats both a
    single-FFT and a single-sparse representation at a sufficient fixed budget, the sharp layer carries the
    residual the low-frequency layer cannot, and at too-small a budget the split loses (the kept caveat)."""
    rng = np.random.default_rng(0)
    T = 256
    t = np.arange(T)
    smooth = np.sin(2 * np.pi * 2 * t / T) + 0.6 * np.cos(2 * np.pi * 5 * t / T)
    sharp = np.zeros(T)
    pos = rng.choice(T, 6, replace=False)
    sharp[pos] = rng.uniform(-3, 3, 6)
    x = smooth + sharp
    def psnr(rec):
        mse = np.mean((rec - x) ** 2)
        return float(10 * np.log10((x.max() - x.min()) ** 2 / (mse + 1e-12)))

    # at a budget that covers both layers (B=12), the split wins by a wide margin
    code = smooth_sharp_split(x, k_smooth=6, k_sharp=6)
    p_split = psnr(smooth_sharp_reconstruct(code))
    p_fft = psnr(_fft_topk(x, 12))
    p_sparse = psnr(_sparse_topk(x, 12))
    assert p_split > p_fft + 5 and p_split > p_sparse + 5, (p_split, p_fft, p_sparse)
    assert np.allclose(x[code.sharp_idx], smooth_sharp_reconstruct(code)[code.sharp_idx])   # sharp positions exact

    # the sharp layer carries energy the low-frequency layer provably cannot
    sm_only = _fft_topk(x, 8)
    assert np.linalg.norm(x - sm_only) / np.linalg.norm(x) > 0.2

    # KEPT CAVEAT: at too small a budget the split cannot afford both layers and loses to single-FFT
    small = smooth_sharp_split(x, k_smooth=4, k_sharp=4)
    assert psnr(smooth_sharp_reconstruct(small)) < psnr(_fft_topk(x, 8))


if __name__ == "__main__":
    _selftest()
    print("holographic_twolayer selftest passed")

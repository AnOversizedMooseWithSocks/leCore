"""Multi-resolution pyramid -- an anti-aliased mipmap of a signal, for coarse-to-fine recall (SCALE-1).

WHY THIS EXISTS (SCALE-1)
-------------------------
Mipmaps, flow pyramids and 3DGS densification all make coarse-to-fine an EXPLICIT strategy: keep the signal at
several resolutions and read the one the query needs, refining toward fine only where it matters. The engine already
leans this way implicitly (recursive/fractal structure, HoloForest's coarse descent, consolidation's low-rank-first
pass), and this makes the multi-resolution ARCHIVE explicit.

The decisive property is anti-aliasing on a COARSE query. If you want a low-resolution view of a signal, you cannot
just subsample the full-resolution store -- any content above the coarse Nyquist FOLDS into the low band and corrupts
it (aliasing). A mipmap level was LOW-PASS FILTERED before it was downsampled, so its coarse view is clean. Each
level is also smaller, so a coarse read is cheap; and the levels together are a progressive code -- the coarsest is a
usable approximation and finer levels add detail back, exact at the top.

THE RELATION TO THE TWO-LAYER CODEC (kept honest). CACHE-2's smooth/sharp split is a fixed TWO-level decomposition
tuned to a storage budget; this is the multi-LEVEL spatial hierarchy with the distinct anti-aliased-LOD property --
each coarse level is a smaller, alias-free array you can read on its own. Same family, different job.

MEASURED (see `_selftest`, a low-frequency signal plus a high frequency above the coarse Nyquist):
  * a 1/8 coarse query from the pyramid matches the true low-frequency band an order of magnitude better than naive
    subsampling, which aliases the high frequency into the low band.
  * each level is half the size of the one below (a coarse read is cheap), and the full level reconstructs exactly.
"""

import numpy as np


def _lowpass(s, keep):
    F = np.fft.rfft(s)
    F[keep:] = 0
    return np.fft.irfft(F, n=len(s))


def _decimate2(s):
    """Halve the resolution WITHOUT aliasing: low-pass to the new Nyquist, then drop every second sample."""
    return _lowpass(s, max(1, len(s) // 4))[::2]


def build_pyramid(signal, n_levels=5):
    """Build an anti-aliased mipmap: [full, half, quarter, ...] where each level is low-pass filtered before being
    downsampled by two, so every coarse level is a clean (alias-free), smaller view of the signal. Returns the list
    of levels, coarsest last."""
    levels = [np.asarray(signal, float)]
    for _ in range(n_levels - 1):
        if len(levels[-1]) < 4:
            break
        levels.append(_decimate2(levels[-1]))
    return levels


def upsample_to(level, n):
    """Resample a pyramid level back to length `n` (linear interpolation) -- the LOD read, so a coarse level can be
    compared or used at full length."""
    level = np.asarray(level, float)
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(level)), level)


def naive_subsample(signal, factor):
    """The cautionary baseline: subsample the full-resolution store with NO pre-filter -- aliases."""
    return np.asarray(signal, float)[::factor]


def _selftest():
    """CI-fast: a 1/8 coarse query from the anti-aliased pyramid matches the true low-frequency band far better than
    a naive subsample (which aliases the high frequency into the low band); levels halve in size; the full level is
    exact."""
    N = 1024
    x = np.arange(N) / N
    sig = np.sin(2 * np.pi * 2 * x) + 0.6 * np.sin(2 * np.pi * 150 * x)   # low + (above-coarse-Nyquist) high freq
    true_low = np.sin(2 * np.pi * 2 * x)                                  # the legitimate coarse content
    pyr = build_pyramid(sig, 5)

    # levels halve in size (a coarse read is cheap), and the full level is exact
    assert [len(l) for l in pyr] == [1024, 512, 256, 128, 64], [len(l) for l in pyr]
    assert np.allclose(pyr[0], sig)

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mip = upsample_to(pyr[3], N)                                          # 1/8 level, anti-aliased
    naive = upsample_to(naive_subsample(sig, 8), N)                       # 1/8 subsample, aliased
    assert rmse(mip, true_low) < rmse(naive, true_low) * 0.5, (rmse(mip, true_low), rmse(naive, true_low))

    # the aliased high frequency leaves a spurious spike under naive subsampling but not in the mipmap
    def bin_energy(s, lo, hi):
        F = np.fft.rfft(s); return float(np.sum(np.abs(F[lo:hi]) ** 2))
    assert bin_energy(naive, 20, 25) > 100 * (bin_energy(mip, 20, 25) + 1e-9)


if __name__ == "__main__":
    _selftest()
    print("holographic_multires selftest passed")

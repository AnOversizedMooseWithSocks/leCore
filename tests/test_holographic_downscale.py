"""CI wrapper for denoise-by-downscale (XDATA-1, the Group G entry). The module ships its asserts in `_selftest`:
a rank-3 subspace invisible in any single noisy vector is recovered by pooling many samples, and slow sinusoids
buried under 2x noise are recovered by keeping the strongest spectral components -- both flagged 'found' against a
permutation null, while pure noise of either type reports nothing (fail-safe). This collects that check into the
suite."""
from holographic.misc.holographic_downscale import _selftest


def test_holographic_downscale_selftest():
    _selftest()

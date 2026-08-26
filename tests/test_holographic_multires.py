"""CI wrapper for the multi-resolution pyramid / mipmap (SCALE-1). The module ships its asserts in `_selftest`: a
coarse query from the anti-aliased pyramid matches the true low-frequency band far better than a naive subsample
(which aliases high frequency into the low band), levels halve in size, and the full level is exact. This collects
that check into the suite."""
from holographic.misc.holographic_multires import _selftest


def test_holographic_multires_selftest():
    _selftest()

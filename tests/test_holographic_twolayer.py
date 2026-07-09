"""CI wrapper for the smooth/sharp two-layer codec (CACHE-2). The module ships its asserts in `_selftest`: on a
smooth-plus-sharp signal the split (low-frequency smooth layer + sparse-sample sharp layer) beats both a single-FFT
and a single-sparse representation at a sufficient fixed budget, the sharp layer carries the residual the
low-frequency layer cannot, and at too-small a budget the split loses (the kept caveat). This collects that check
into the suite."""
from holographic.misc.holographic_twolayer import _selftest


def test_holographic_twolayer_selftest():
    _selftest()

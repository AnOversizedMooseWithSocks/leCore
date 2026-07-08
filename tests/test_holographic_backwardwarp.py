"""CI wrapper for the backward-warp hole-free validation (PHASE-2). The module ships its asserts in `_selftest`:
under a non-uniform warp a forward scatter leaves holes and overlaps, while a backward gather (the form the engine's
unbind already is) leaves none and reconstructs exactly. This collects that check into the suite."""
from holographic.misc.holographic_backwardwarp import _selftest


def test_holographic_backwardwarp_selftest():
    _selftest()

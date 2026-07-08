"""CI wrapper for the gradient-free reservoir (Family 2 of the substrate-native learning program).
The module ships its own asserts in _selftest -- a FIXED reservoir (holostuff's permute recurrence) with
a single ridge-regression readout learns one-step prediction and is bit-deterministic. This collects it."""
from holographic.rendering.holographic_reservoir import _selftest


def test_holographic_reservoir_selftest():
    _selftest()

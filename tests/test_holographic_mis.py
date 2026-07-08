"""CI wrapper for Multiple Importance Sampling (Veach's balance heuristic for combining the engine's
estimators). The module ships its asserts in `_selftest`: on a coarse sharp-kernel manifold with a mix of
on-grid + off-grid cues, naive averaging of hard 1-NN and soft Hopfield is WORSE than the better single (the
warning), while the balance-heuristic combination beats naive averaging and both singles. This collects that
check into the suite."""
from holographic.rendering.holographic_mis import _selftest


def test_holographic_mis_selftest():
    _selftest()

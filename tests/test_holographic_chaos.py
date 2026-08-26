"""CI wrapper for the nonlinear-dynamics companion (the reservoir aimed at chaotic flow). The module
ships its own asserts in _selftest: the reservoir learns the chaotic Lorenz one-step operator far
better (>10x) than the best linear map (full DMD) and persistence -- where a linear transfer is pinned
at the chaos floor -- deterministically, while NOT overclaiming the closed-loop horizon (kept as a loud
negative). This collects that check into the suite."""
from holographic.misc.holographic_chaos import _selftest


def test_holographic_chaos_selftest():
    _selftest()

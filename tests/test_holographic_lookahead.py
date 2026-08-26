"""CI wrapper for the D4 negative (cross-cutting RAY-1 re-anchoring -> model-based planning). The module ships its
asserts in `_selftest`: a per-action bind-displacement forward model learned from the creature's experience
COLLAPSES the actions (the four predicted leaves are nearly identical), so re-anchored lookahead ranks the actions
the same as the plain reactive value and can never decide differently -- redundant. The re-anchoring mechanism
itself keeps the rollout on-manifold (RAY-1 confirmed); there is just nothing for it to improve, because a single
linear/bind operator per action is too coarse for the egocentric sense-space. A kept negative. This collects it."""
from holographic.misc.holographic_lookahead import _selftest


def test_holographic_lookahead_negative_selftest():
    _selftest()

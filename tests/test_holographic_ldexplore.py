"""CI wrapper for the D1 negative (cross-cutting SAMPLE-1 -> creature exploration). The module ships its asserts
in `_selftest`: low-discrepancy action selection covers FAR fewer distinct cells than epsilon-random on an open
grid -- because a walk accumulates displacement and a balanced (low-discrepancy) action sequence cancels it,
while random's imbalance is the diffusive drift that explores. The transfer that pays for direct sampling is
harmful for sequential exploration. A kept negative. This collects that check."""
from holographic.misc.holographic_ldexplore import _selftest


def test_holographic_ldexplore_negative_selftest():
    _selftest()

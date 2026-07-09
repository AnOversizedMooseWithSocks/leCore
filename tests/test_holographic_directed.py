"""CI wrapper for directed structure (a permutation direction role for sequences/graphs). The module ships its
asserts in `_selftest`: the direction role suppresses the predecessor leak that makes the undirected baseline
ambiguous, it generalises to a branching graph node (all successors recovered from one unbind), and it composes
with the throughput-gated walk. This collects that check into the suite."""
from holographic.misc.holographic_directed import _selftest


def test_holographic_directed_selftest():
    _selftest()

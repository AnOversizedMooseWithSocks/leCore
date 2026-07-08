"""CI wrapper for throughput-gated traversal (Russian roulette for holographic paths). The module ships its
asserts in `_selftest`: the gate stops at the throughput floor and recovers the good prefix (logic), and on
a real directed linked list stored in superposition it recovers every valid hop then abstains the instant
the chain is exhausted, at a fraction of a fixed depth's steps. This collects that check into the suite."""
from holographic.misc.holographic_traverse import _selftest


def test_holographic_traverse_selftest():
    _selftest()

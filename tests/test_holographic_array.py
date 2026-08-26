"""CI wrapper for the federated RAID array module (Path D). The module ships its own asserts in
_selftest -- RAID-5 reconstructs a lost shard exactly, RAID-6 survives two losses, and one parity
cannot recover two (the information floor). This collects that check into the suite."""
from holographic.misc.holographic_array import _selftest


def test_holographic_array_selftest():
    _selftest()

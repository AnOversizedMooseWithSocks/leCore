"""CI wrapper for the compute-in-superposition module (Path D, the WIDTH faculty). Its _selftest
asserts the core contract -- a single keyed item recovers exactly with a unitary key -- and prints
the honest capacity decay across width. This collects that contract into the suite."""
from holographic.misc.holographic_superposed import _selftest


def test_holographic_superposed_selftest():
    _selftest()

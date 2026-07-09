"""CI wrapper for the exact RNS-phasor arithmetic module (Path D's arithmetic lever). Its _selftest asserts
the core contract -- modular accumulation via phasor binding is exact for thousands of terms, and integer
matmul is exact where a lossy bundle degrades -- and shows the range federating over moduli channels."""
from holographic.misc.holographic_rns import _selftest


def test_holographic_rns_selftest():
    _selftest()

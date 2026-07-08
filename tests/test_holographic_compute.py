"""CI wrapper for the federated / cleanup-gated forward-pass module (Path D's compute win). Its _selftest
asserts the core results -- federating the weight rows moves the class-capacity wall (K=8 tracks the exact
classifier where K=1 collapses), and a between-layers cleanup helps a deep pass -- on a NumPy blob task."""
from holographic.misc.holographic_compute import _selftest


def test_holographic_compute_selftest():
    _selftest()

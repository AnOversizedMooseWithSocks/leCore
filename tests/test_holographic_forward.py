"""CI wrapper for the Forward-Forward algorithm (Family 4 of the learning program -- the local-gradient
DEPTH corner). The module ships its own asserts in _selftest: a stack trained ONLY by local goodness
objectives (no backprop, no settling) classifies a separable multi-class task and makes positive goodness
exceed negative, deterministically. (Its measured negative -- it trails a linear model at this scale -- is
documented in the module docstring; the selftest checks the MECHANISM, not a competitive accuracy.)"""
from holographic.misc.holographic_forward import _selftest


def test_holographic_forward_selftest():
    _selftest()

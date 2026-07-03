"""Pattern fields: deterministic, [0,1], socket-compatible."""
import numpy as np
from holographic_pattern import PATTERNS, make_pattern, value_noise, fbm, field_lerp


def test_all_patterns_in_range_and_deterministic():
    P = np.random.default_rng(0).uniform(-2, 2, (1500, 3))
    for name in PATTERNS:
        f = make_pattern(name)
        v = np.asarray(f(P), float)
        assert v.min() >= -1e-9 and v.max() <= 1.0 + 1e-9, name
        assert np.array_equal(np.asarray(f(P)), np.asarray(f(P)))    # bit-identical re-evaluation


def test_noise_is_hashseed_independent_and_seedable():
    P = np.random.default_rng(1).uniform(-3, 3, (500, 3))
    a = value_noise(3.0, seed=7)(P); b = value_noise(3.0, seed=7)(P)
    assert np.array_equal(a, b)                                      # integer-lattice hash, not salted hash()
    c = value_noise(3.0, seed=8)(P)
    assert not np.allclose(a, c)                                     # seed actually perturbs


def test_field_lerp_bounds_and_colour():
    P = np.random.default_rng(2).uniform(-2, 2, (800, 3))
    g = field_lerp(make_pattern("checker", scale=2.0), 0.1, 0.9)
    gv = g(P); assert gv.min() >= 0.1 - 1e-9 and gv.max() <= 0.9 + 1e-9
    col = field_lerp(fbm(scale=2.0, octaves=3), (1, 0, 0), (0, 0, 1))(P)
    assert col.shape == (800, 3)                                     # colour channels work


def test_pattern_resolves_through_param_socket():
    """The reason the module exists: a pattern IS a Param field, resolved per point by resolve_param."""
    from holographic_param import Param, resolve_param
    P = np.random.default_rng(3).uniform(-2, 2, (300, 3))
    p = Param(field=make_pattern("stripes", scale=2.0))
    v = resolve_param(p, points=P)
    assert v.shape == (300,) and v.min() >= -1e-9 and v.max() <= 1.0 + 1e-9

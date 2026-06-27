"""Tests for the determinism contract (ISA-1): the shared sign/tie-break utility and the bit-exact de-silo of
spectral.sign_fix and chart._fix_signs onto it."""

import numpy as np

from holographic_determinism import fix_eigvec_signs, argmax_tiebreak, _selftest


def test_module_selftest():
    _selftest()


def test_sign_rule_is_deterministic_and_sign_invariant():
    rng = np.random.default_rng(3)
    V = rng.standard_normal((40, 7))
    a = fix_eigvec_signs(V)
    b = fix_eigvec_signs(V)
    assert np.array_equal(a, b)                            # deterministic
    assert np.allclose(fix_eigvec_signs(V), fix_eigvec_signs(-V))  # V and -V -> same fixed basis
    for j in range(a.shape[1]):                            # the rule: largest-|entry| is non-negative
        i = int(np.argmax(np.abs(a[:, j])))
        assert a[i, j] >= 0


def test_sign_rule_is_idempotent():
    rng = np.random.default_rng(4)
    V = rng.standard_normal((25, 5))
    once = fix_eigvec_signs(V)
    assert np.array_equal(fix_eigvec_signs(once), once)


def test_copy_flag_preserves_each_call_sites_behavior():
    rng = np.random.default_rng(5)
    V = rng.standard_normal((15, 4))
    Vc = V.copy()
    out = fix_eigvec_signs(Vc, copy=True)                  # chart's contract: input untouched
    assert np.array_equal(Vc, V)
    out_ip = fix_eigvec_signs(Vc, copy=False)              # spectral's contract: in place, same array
    assert out_ip is Vc
    assert np.array_equal(out, out_ip)                     # both produce the same values


def test_argmax_tiebreak_picks_lowest_index():
    assert argmax_tiebreak(np.array([2.0, 5.0, 5.0, 1.0])) == 1
    assert argmax_tiebreak(np.array([9.0, 9.0, 9.0])) == 0


def test_spectral_sign_fix_is_now_the_shared_rule_bit_exact():
    # The de-silo must not change spectral's output: sign_fix == fix_eigvec_signs, bit-for-bit.
    from holographic_spectral import sign_fix
    rng = np.random.default_rng(6)
    V = rng.standard_normal((30, 6))
    assert np.array_equal(sign_fix(V.copy()), fix_eigvec_signs(V))


def test_chart_sign_fix_is_now_the_shared_rule_bit_exact():
    from holographic_chart import _fix_signs
    rng = np.random.default_rng(7)
    Y = rng.standard_normal((22, 3))
    assert np.array_equal(_fix_signs(Y), fix_eigvec_signs(Y))

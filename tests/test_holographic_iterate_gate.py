"""Tests for the closed-form iterate RE-ENABLE (holographic_iterate.iterate_gated) -- exact in regime, safe fallback."""
import numpy as np
from holographic.misc.holographic_iterate import iterate_gated, is_bind_operator, bind_kernel_of
from holographic.agents_and_reasoning.holographic_ai import bind


def _contractive_op(D=256, seed=0):
    rng = np.random.default_rng(seed)
    kernel = rng.standard_normal(D)
    kernel /= np.max(np.abs(np.fft.rfft(kernel))) * 1.001      # |transfer| <= 1 so values stay bounded
    return (lambda x: bind(kernel, x)), kernel, rng


def test_closed_form_exact_for_bind_operator():
    op, kernel, rng = _contractive_op()
    state = rng.standard_normal(256)
    res, info = iterate_gated(op, state, k=300)
    slow = state.copy()
    for _ in range(300):
        slow = bind(kernel, slow)
    assert info["used"] == "superior" and np.allclose(res, slow, atol=1e-8)


def test_nonlinear_operator_falls_back():
    op, kernel, rng = _contractive_op()
    op_nl = lambda x: np.tanh(bind(kernel, x))
    res, info = iterate_gated(op_nl, rng.standard_normal(256), k=300)
    assert info["used"] == "fallback" and info["score"] == 0.0


def test_small_k_skips_the_detector():
    op, _, rng = _contractive_op()
    res, info = iterate_gated(op, rng.standard_normal(256), k=3, min_k=8)
    assert info["used"] == "fallback" and info["reason"] == "k<min_k"


def test_detector_recognizes_convolution():
    op, kernel, rng = _contractive_op()
    assert is_bind_operator(op, 256) == 1.0
    assert is_bind_operator(lambda x: np.tanh(op(x)), 256) == 0.0
    assert np.allclose(bind_kernel_of(op, 256), kernel, atol=1e-8)  # impulse response == kernel


def test_gate_never_worse_than_stepping():
    # exactness guarantee: for any bind operator the closed form == stepping, so the gate can't do worse
    op, kernel, rng = _contractive_op(seed=3)
    state = rng.standard_normal(256)
    for k in (16, 128, 1000):
        res, _ = iterate_gated(op, state, k=k)
        slow = state.copy()
        for _ in range(k):
            slow = bind(kernel, slow)
        assert np.allclose(res, slow, atol=1e-8)

"""Tests for spectral iteration of a bind operator (RT-I1): the k-step jump in one eval matches the k-bind
rollout, the limit is closed-form, the eigendecomposition is the free FFT spectrum, and convergence/stall is read
off the spectrum before running."""

import numpy as np
import pytest

from holographic_iterate import transfer, step_k, limit, dominant_eigenvector, spectral_profile
from holographic_dynamics import Propagator
from holographic_ai import bind, cosine

N = 256


def _make_U(H):
    return np.fft.irfft(H, n=N)


def test_eigendecomposition_is_the_free_fft_spectrum():
    rng = np.random.default_rng(0)
    U = rng.standard_normal(N)
    assert np.array_equal(transfer(U), np.fft.rfft(U))      # the spectrum IS the rfft -- no dense O(n^3) work


def test_k_step_jump_matches_k_bind_rollout():
    # THE BAR: one eval (transfer to the k-th power) reproduces k sequential binds, to FFT tolerance.
    rng = np.random.default_rng(0)
    U = _make_U(0.9 * np.exp(1j * rng.uniform(0, 2 * np.pi, N // 2 + 1)))
    P = Propagator(U, U)
    state = rng.standard_normal(N)
    for k in (1, 3, 8, 20, 50):
        assert np.max(np.abs(P.rollout(state, k)[-1] - step_k(state, U, k))) < 1e-9


def test_contractive_limit_is_closed_form_zero():
    rng = np.random.default_rng(1)
    U = _make_U(0.85 * np.exp(1j * rng.uniform(0, 2 * np.pi, N // 2 + 1)))
    state = rng.standard_normal(N)
    assert np.linalg.norm(limit(state, U)) < 1e-9           # closed-form limit, no iteration
    assert np.linalg.norm(step_k(state, U, 80)) < 0.05 * np.linalg.norm(state)   # actual rollout agrees


def test_divergent_operator_has_no_finite_limit():
    rng = np.random.default_rng(2)
    U = _make_U(1.1 * np.exp(1j * rng.uniform(0, 2 * np.pi, N // 2 + 1)))
    state = rng.standard_normal(N)
    with pytest.raises(ValueError):
        limit(state, U)
    assert spectral_profile(U)["regime"] == "divergent"


def test_regime_is_read_off_the_spectrum_before_running():
    rng = np.random.default_rng(3)
    state = rng.standard_normal(N)
    contr = _make_U(0.8 * np.exp(1j * rng.uniform(0, 2 * np.pi, N // 2 + 1)))
    div = _make_U(1.05 * np.exp(1j * rng.uniform(0, 2 * np.pi, N // 2 + 1)))
    assert spectral_profile(contr)["regime"] == "contractive"
    assert spectral_profile(div)["regime"] == "divergent"
    # the prediction matches the actual behaviour
    assert np.linalg.norm(step_k(state, contr, 50)) < np.linalg.norm(state)       # decays
    assert np.linalg.norm(step_k(state, div, 30)) > np.linalg.norm(state)         # grows


def test_spectral_gap_predicts_power_iteration_speed():
    # the linear cousin of a resonator stall: a small spectral gap (near-degenerate top eigenvalues) -> slow.
    rng = np.random.default_rng(4)

    def steps(U, tol=1e-4, maxit=300):
        x = rng.standard_normal(N); x /= np.linalg.norm(x); prev = None
        for it in range(maxit):
            x = bind(U, x); x /= np.linalg.norm(x)
            if prev is not None and 1 - abs(cosine(x, prev)) < tol:
                return it
            prev = x.copy()
        return maxit

    big = np.full(N // 2 + 1, 0.2, dtype=complex); big[3] = 1.0; big[7] = 0.4
    small = np.full(N // 2 + 1, 0.2, dtype=complex); small[3] = 1.0; small[7] = 0.95
    Ub, Us = _make_U(big), _make_U(small)
    assert spectral_profile(Ub)["spectral_gap"] < spectral_profile(Us)["spectral_gap"]
    assert steps(Ub) < steps(Us)                            # larger gap converges faster


def test_dominant_eigenvector_is_deterministic_and_unit():
    rng = np.random.default_rng(5)
    H = np.full(N // 2 + 1, 0.2, dtype=complex); H[5] = 1.0
    U = _make_U(H)
    v1 = dominant_eigenvector(U)
    v2 = dominant_eigenvector(U)
    assert np.array_equal(v1, v2)                            # deterministic (sign-pinned)
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-9             # unit
    # it is the fixed direction of power iteration: binding it leaves its direction unchanged
    w = bind(U, v1); w /= np.linalg.norm(w)
    assert abs(cosine(w, v1)) > 0.99

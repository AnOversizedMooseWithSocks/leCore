"""Precomputed Radiance Transfer: an orthonormal SH basis, transfer that captures self-shadowing, relight as a dot product."""
import numpy as np
from holographic_prt import sh_eval, _sphere_dirs, project_env_to_sh, precompute_transfer, shade_prt


def test_sh_basis_orthonormal():
    w = _sphere_dirs(20000)
    Y = sh_eval(w, order=4)
    G = (4.0 * np.pi / len(w)) * (Y.T @ Y)
    assert np.abs(G - np.eye(Y.shape[1])).max() < 0.03           # orthonormal to Monte-Carlo tolerance


def test_dc_recovers_ambient():
    # a fully-open upward-facing point under a unit-white sky returns ~1 (the DC/irradiance recovery)
    class Empty:
        def eval(s, P): return np.full(len(P), 10.0)             # nothing anywhere -> never occluded
    T = precompute_transfer(Empty(), np.array([[0.0, 0, 0]]), np.array([[0.0, 1.0, 0.0]]), order=3, n=800)
    L = project_env_to_sh(lambda d: np.ones((len(d), 3)), order=3, n=4000)
    assert 0.85 < float(shade_prt(T, L)[0, 0]) < 1.15


def test_transfer_captures_self_shadow():
    class TwoSpheres:
        def eval(s, P):
            a = np.linalg.norm(P - np.array([-1.0, 0, 0]), axis=1) - 0.9
            b = np.linalg.norm(P - np.array([1.0, 0, 0]), axis=1) - 0.9
            return np.minimum(a, b)
    sdf = TwoSpheres()
    T_open = precompute_transfer(sdf, np.array([[-1.0, 0.9, 0.0]]), np.array([[0.0, 1.0, 0.0]]), order=3, n=800)
    T_blocked = precompute_transfer(sdf, np.array([[-0.15, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]), order=3, n=800)
    L = project_env_to_sh(lambda d: np.ones((len(d), 3)), order=3, n=3000)
    assert float(shade_prt(T_open, L)[0, 0]) > float(shade_prt(T_blocked, L)[0, 0]) * 1.3   # blocked -> darker


def test_relight_is_a_dot_product():
    # two different lights on one transfer produce different shading, each a pure matrix-vector product
    T = np.random.default_rng(0).random((50, 9))
    L1 = np.random.default_rng(1).random((9, 3)); L2 = np.random.default_rng(2).random((9, 3))
    r1 = shade_prt(T, L1); r2 = shade_prt(T, L2)
    assert r1.shape == (50, 3) and not np.allclose(r1, r2)
    assert np.allclose(shade_prt(T, L1), np.clip(T @ L1, 0, None))

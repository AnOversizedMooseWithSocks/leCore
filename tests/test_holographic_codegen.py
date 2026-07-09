"""Tests for SymPy design-time codegen -> exact gradients/SDF normals (CODEGEN-1).
Skips cleanly when sympy is absent (a design-time dep)."""
import numpy as np
from holographic.misc.holographic_codegen import HAS_SYMPY, sdf_normal_fn, compile_field, gradient_fn


def test_sphere_exact_normal_beats_finite_difference():
    if not HAS_SYMPY:
        return
    R = 1.3
    val, nrm = sdf_normal_fn(f"sqrt(x**2 + y**2 + z**2) - {R}")
    P = np.random.default_rng(0).standard_normal((100, 3)) * 1.5
    analytic = P / np.linalg.norm(P, axis=1, keepdims=True)
    assert np.max(np.abs(nrm(P) - analytic)) < 1e-12         # exact to machine precision

    def fd(P, h):
        g = np.zeros_like(P)
        for i in range(3):
            e = np.zeros(3); e[i] = h
            g[:, i] = (val(P + e) - val(P - e)) / (2 * h)
        return g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
    assert np.max(np.abs(fd(P, 1e-2) - analytic)) > 1e-12     # finite differences carry step error


def test_torus_normal_matches_numeric():
    if not HAS_SYMPY:
        return
    tval, tnrm = sdf_normal_fn("sqrt((sqrt(x**2+y**2)-1.0)**2 + z**2) - 0.4")
    P = np.random.default_rng(1).standard_normal((40, 3))
    ng = np.zeros_like(P)
    for i in range(3):
        e = np.zeros(3); e[i] = 1e-6
        ng[:, i] = (tval(P + e) - tval(P - e)) / 2e-6
    ng = ng / (np.linalg.norm(ng, axis=1, keepdims=True) + 1e-12)
    assert np.max(np.abs(tnrm(P) - ng)) < 1e-4


def test_force_is_negative_gradient():
    if not HAS_SYMPY:
        return
    g = gradient_fn("0.5*(x**2 + y**2 + z**2)", ("x", "y", "z"))
    P = np.random.default_rng(2).standard_normal((10, 3))
    assert np.allclose(-g(P), -P)                             # quadratic well -> force = -p


def test_compile_field_value_and_grad_shapes():
    if not HAS_SYMPY:
        return
    c = compile_field("x**2 + y**2", ("x", "y"))
    P = np.random.default_rng(3).standard_normal((7, 2))
    assert c["value"](P).shape == (7,) and c["gradient"](P).shape == (7, 2)

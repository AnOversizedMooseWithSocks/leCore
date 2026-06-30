"""Tests for the Riemannian geometry layer extracted from leOS (SPHERE-1)."""
import numpy as np
from holographic_sphere import frechet_mean, geodesic_variance, parallel_transport
from holographic_ai import geodesic, exp_map


def _norm(v):
    return v / np.linalg.norm(v)


def _spread_set(k, spread, seed, D=64):
    r = np.random.default_rng(seed)
    base = _norm(r.standard_normal(D))
    out = []
    for _ in range(k):
        t = spread * r.standard_normal(D)
        t = t - np.dot(t, base) * base                      # into base's tangent plane
        out.append(exp_map(base, t))
    return out


def test_frechet_mean_minimizes_geodesic_variance():
    pts = _spread_set(40, 0.6, seed=1)
    fm = frechet_mean(pts)
    euclid = _norm(sum(pts))                                 # what bundle does
    assert geodesic_variance(pts, fm) <= geodesic_variance(pts, euclid) + 1e-9


def test_frechet_mean_is_unit_and_single_element_identity():
    pts = _spread_set(10, 0.4, seed=2)
    fm = frechet_mean(pts)
    assert abs(np.linalg.norm(fm) - 1.0) < 1e-6
    one = _norm(np.random.default_rng(3).standard_normal(64))
    assert np.allclose(frechet_mean([one]), one)


def test_geometry_matters_for_spread_not_tight():
    spread = _spread_set(40, 0.6, seed=4)
    tight = _spread_set(40, 0.05, seed=5)
    gap_spread = geodesic(frechet_mean(spread), _norm(sum(spread)))
    gap_tight = geodesic(frechet_mean(tight), _norm(sum(tight)))
    assert gap_spread > gap_tight                            # diverges from bundle only when spread


def test_parallel_transport_preserves_length_and_tangency():
    rng = np.random.default_rng(0)
    p = _norm(rng.standard_normal(64)); q = _norm(rng.standard_normal(64))
    v = rng.standard_normal(64); v = v - np.dot(v, p) * p   # tangent at p
    tq = parallel_transport(v, p, q)
    assert abs(np.linalg.norm(tq) - np.linalg.norm(v)) < 1e-9    # length preserved
    assert abs(float(np.dot(tq, q))) < 1e-9                      # lands in q's tangent plane


def test_parallel_transport_identity_at_same_point():
    rng = np.random.default_rng(1)
    p = _norm(rng.standard_normal(64))
    v = rng.standard_normal(64); v = v - np.dot(v, p) * p
    assert np.allclose(parallel_transport(v, p, p), v)

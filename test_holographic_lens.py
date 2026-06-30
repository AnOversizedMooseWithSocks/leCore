"""Tests for gradient-field navigation + caustic detection extracted from leOS gravitational_lens (LENS-1)."""
import numpy as np
from holographic_lens import (field_force, deflect, detect_caustic, navigate, _normalize)
from holographic_ai import geodesic


def _two_attractors(seed=0, D=32):
    rng = np.random.default_rng(seed)
    a1 = _normalize(rng.standard_normal(D)); a2 = _normalize(rng.standard_normal(D))
    while abs(np.dot(a1, a2)) > 0.3:
        a2 = _normalize(rng.standard_normal(D))
    return np.stack([a1, a2]), rng


def test_deflect_moves_toward_nearest_attractor():
    A, rng = _two_attractors(1)
    q = _normalize(A[0] + 0.5 * rng.standard_normal(A.shape[1]))
    before = geodesic(q, A[0])
    lensed, dmag, _ = deflect(q, A, sigma=0.8, strength=0.5)
    assert geodesic(lensed, A[0]) < before and dmag > 0


def test_navigate_approaches_attractor():
    A, rng = _two_attractors(2)
    q = _normalize(A[0] + 0.5 * rng.standard_normal(A.shape[1]))
    nav = navigate(q, A, sigma=0.8, strength=0.6)
    assert geodesic(nav["final"], A[0]) < 0.2


def test_caustic_high_at_midpoint_low_near_attractor():
    A, rng = _two_attractors(3)
    mid = _normalize(A[0] + A[1])                            # equidistant -> ambiguous
    near = _normalize(A[0] + 0.05 * rng.standard_normal(A.shape[1]))
    c_mid = detect_caustic(mid, A, sigma=0.8)[0]
    c_near = detect_caustic(near, A, sigma=0.8)[0]
    assert c_mid > 0.5 and c_near < 0.2 and c_mid > c_near


def test_single_attractor_no_caustic():
    A, _ = _two_attractors(4)
    score, n = detect_caustic(A[0], A[:1], sigma=0.8)        # only one attractor -> no ambiguity
    assert score == 0.0


def test_field_force_points_toward_mass():
    A, rng = _two_attractors(5)
    q = _normalize(A[0] + 0.4 * rng.standard_normal(A.shape[1]))
    f = field_force(q, A[:1], sigma=0.8)                     # force from a1 alone should reduce angle to a1
    from holographic_ai import exp_map
    moved = exp_map(q, 0.3 * f)
    assert geodesic(moved, A[0]) < geodesic(q, A[0])

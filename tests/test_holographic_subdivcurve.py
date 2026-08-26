"""Tests for subdivision curves on hypervector sequences (ARCH-5): Chaikin corner-cutting, the 1-D inward mirror of
FWD-8's mesh subdivision. Counts double (refine), a straight line stays straight (affine reproduction), the curve
converges, a zig-zag's roughness shrinks (low-pass), and Chaikin is approximating (control points not interpolated)."""

import numpy as np

from holographic.mesh_and_geometry.holographic_subdivcurve import chaikin_subdivide, subdivide_sequence, _roughness


def _seq(n=6, dim=64, seed=0):
    return np.random.default_rng(seed).standard_normal((n, dim))


# ---- refine: count doubling -----------------------------------------------------------------------
def test_open_counts_double_each_level():
    P = _seq(6)
    assert [len(subdivide_sequence(P, l)) for l in range(4)] == [6, 10, 18, 34]   # 2(n-1)


def test_closed_counts_double_each_level():
    P = _seq(6)
    assert [len(subdivide_sequence(P, l, closed=True)) for l in range(4)] == [6, 12, 24, 48]   # 2n


def test_single_level_open_count():
    P = _seq(7)
    assert len(chaikin_subdivide(P)) == 2 * (7 - 1)


# ---- affine reproduction (FWD-8's "flat stays flat") ----------------------------------------------
def test_straight_line_of_vectors_stays_straight():
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal(48), rng.standard_normal(48)
    ramp = np.array([a + (b - a) * t for t in np.linspace(0, 1, 6)])
    sub = subdivide_sequence(ramp, 3)
    dn = (b - a) / np.linalg.norm(b - a)
    max_resid = max(np.linalg.norm((p - a) - np.dot(p - a, dn) * dn) for p in sub)
    assert max_resid < 1e-12


# ---- convergence ----------------------------------------------------------------------------------
def test_curve_length_converges():
    Q = _seq(8, seed=2)
    lengths = [float(np.sum(np.linalg.norm(np.diff(subdivide_sequence(Q, l), axis=0), axis=1))) for l in range(6)]
    deltas = [abs(lengths[i + 1] - lengths[i]) for i in range(len(lengths) - 1)]
    assert all(deltas[i + 1] < deltas[i] for i in range(len(deltas) - 1))


# ---- low-pass smoothing ---------------------------------------------------------------------------
def test_zigzag_roughness_shrinks_each_level():
    zig = np.zeros((10, 64)); zig[::2, 0] = 1.0; zig[1::2, 0] = -1.0
    rough = [_roughness(subdivide_sequence(zig, l)) for l in range(5)]
    assert all(rough[i + 1] < rough[i] for i in range(len(rough) - 1))


# ---- kept negative: approximating, not interpolating ----------------------------------------------
def test_chaikin_is_approximating_not_interpolating():
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal(48), rng.standard_normal(48)
    ramp = np.array([a + (b - a) * t for t in np.linspace(0, 1, 6)])
    interior = ramp[2]
    nearest = min(np.linalg.norm(interior - p) for p in subdivide_sequence(ramp, 2))
    assert nearest > 1e-6                                  # control points are cut, not preserved


# ---- edge cases / determinism ---------------------------------------------------------------------
def test_short_sequence_returned_unchanged():
    P = _seq(1)
    assert np.array_equal(subdivide_sequence(P, 3), P)


def test_subdivision_is_deterministic():
    P = _seq(6)
    assert np.array_equal(subdivide_sequence(P, 2), subdivide_sequence(P, 2))

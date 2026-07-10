"""G2 -- the generic scatter/gather, and G4's histogram, which turned out to be the same function.

THE AUDIT'S CORRECTION, for the fourth time. The backlog said: *"`scatter_to_field` is typed to a 2-D grid `(H,W)`
with bilinear spread; promote the general case."* Reading the code: `scatter_to_field` is a thin **door** onto
`holographic_transfer.scatter`, which is already rank-agnostic, kernel-parameterised, vector-valued, and has an
exact permutation-invariant sibling. **The generic existed; only the door was graphics-typed.**

What was genuinely missing was a `nearest` kernel -- one node, weight 1. That is the GPU's scatter (an atomic add
at an index), and scattering ones at integer coordinates IS `np.bincount`. So G4's "deterministic histogram" is not
a new module: it is `scatter(kernel="nearest")`, and `scatter_exact(kernel="nearest")` is a histogram that is
bit-identical however the points are ordered.

THE DATA. A scatter's order-dependence only shows when cells COLLIDE and the weights have DYNAMIC RANGE. Uniform
weights on distinct cells reorder to the same bits and prove nothing, so every exactness test here uses 4,000
points onto a 16x16 grid with weights spanning 16 orders of magnitude.
"""

import numpy as np
import pytest

from holographic.misc.holographic_fields import scatter_to_field, scatter_to_field_3d
from holographic.misc.holographic_transfer import _KERNELS, gather, scatter, scatter_exact


def _colliding(n=4000, shape=(16, 16), seed=0):
    """Points that COLLIDE (many per cell) with weights of wide dynamic range -- the only data that exposes the
    order-dependence of `np.add.at`."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, shape[0] - 1, size=(n, len(shape)))
    vals = rng.normal(size=n) * 10.0 ** rng.integers(-8, 8, size=n)
    return pts, vals


# ---------------------------------------------------------------------------------------------------------
# the generic was already general
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("D", [1, 2, 3, 4])
def test_scatter_is_rank_agnostic_and_mass_preserving(D):
    rng = np.random.default_rng(D)
    shape = (6,) * D
    pts = rng.uniform(0, 5, size=(50, D))
    vals = rng.normal(size=50)
    g = scatter(pts, vals, shape, kernel="bilinear", periodic=True)
    assert g.shape == shape
    assert abs(float(g.sum()) - float(vals.sum())) < 1e-12   # partition of unity: nothing created or lost


def test_scatter_carries_vector_values():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 5, size=(30, 2))
    vals = rng.normal(size=(30, 3))
    g = scatter(pts, vals, (6, 6), periodic=True)
    assert g.shape == (6, 6, 3)
    assert np.allclose(g.sum(axis=(0, 1)), vals.sum(axis=0), atol=1e-12)


def test_clamped_edges_keep_the_mass_inside():
    edge = np.array([[-2.0, -2.0], [7.0, 7.0]])
    g = scatter(edge, np.ones(2), (6, 6), periodic=False)
    assert abs(float(g.sum()) - 2.0) < 1e-12


def test_scatter_and_gather_use_the_same_kernel_and_are_adjoint():
    # <scatter(p, v), f> == <v, gather(f, p)> for any field f -- the definition of an adjoint pair.
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 5, size=(40, 2))
    v = rng.normal(size=40)
    f = rng.normal(size=(6, 6))
    lhs = float((scatter(pts, v, (6, 6), periodic=True) * f).sum())
    rhs = float(v @ gather(f, pts, periodic=True))
    assert abs(lhs - rhs) < 1e-9


def test_the_graphics_doors_delegate_to_the_generic():
    rng = np.random.default_rng(2)
    pts = rng.uniform(0, 5, size=(20, 2))
    v = rng.normal(size=20)
    # scatter_to_field swaps (x,y) -> (row,col); the generic, fed the swapped points, must agree bit-for-bit
    assert np.array_equal(scatter_to_field((6, 6), pts, v),
                          scatter(pts[:, ::-1], v, (6, 6), kernel="bilinear", periodic=True))

    p3 = rng.uniform(0, 5, size=(20, 3))
    assert scatter_to_field_3d((6, 6, 6), p3, v).shape == (6, 6, 6)


# ---------------------------------------------------------------------------------------------------------
# the nearest kernel: the GPU's scatter, and a histogram
# ---------------------------------------------------------------------------------------------------------

def test_nearest_is_a_registered_kernel():
    assert "nearest" in _KERNELS
    wfn, nnode, shift = _KERNELS["nearest"]
    assert nnode == 1 and shift == -0.5


def test_a_nearest_scatter_of_ones_is_exactly_bincount():
    idx = np.random.default_rng(0).integers(0, 8, size=500)
    g = scatter(idx[:, None].astype(float), np.ones(500), (8,), kernel="nearest")
    assert np.array_equal(g, np.bincount(idx, minlength=8).astype(float))


def test_the_nearest_tie_convention_is_stated_and_deterministic():
    # 2.5 is exactly between cells 2 and 3. It rounds UP, every time, because base = floor(p + 0.5).
    for _ in range(5):
        g = scatter(np.array([[2.5]]), np.array([1.0]), (6,), kernel="nearest")
        assert int(np.argmax(g)) == 3
    assert int(np.argmax(scatter(np.array([[2.4999]]), np.array([1.0]), (6,), kernel="nearest"))) == 2


def test_nearest_preserves_mass_in_any_rank():
    rng = np.random.default_rng(3)
    for D in (1, 2, 3):
        pts = rng.uniform(0, 5, size=(60, D))
        v = rng.normal(size=60)
        g = scatter(pts, v, (6,) * D, kernel="nearest")
        assert abs(float(g.sum()) - float(v.sum())) < 1e-12   # one node, weight 1: nothing to lose


# ---------------------------------------------------------------------------------------------------------
# exactness: a scatter is a reduce PER CELL
# ---------------------------------------------------------------------------------------------------------

def test_the_premise_a_float_scatter_depends_on_point_order():
    pts, vals = _colliding()
    perm = np.random.default_rng(9).permutation(len(pts))
    a = scatter(pts, vals, (16, 16))
    b = scatter(pts[perm], vals[perm], (16, 16))
    assert not np.array_equal(a, b)
    assert np.abs(a - b).max() > 1e-12                        # measured 1.12e-08


def test_the_data_actually_collides_and_has_dynamic_range():
    # Guard: uniform weights on distinct cells reorder to the same bits and would prove nothing.
    pts, vals = _colliding()
    cells = np.floor(pts).astype(int)
    _uniq = len(set(map(tuple, cells)))
    assert len(pts) > 10 * _uniq                              # many points per cell
    assert np.abs(vals).max() / max(np.abs(vals).min(), 1e-300) > 1e10


@pytest.mark.parametrize("kernel", ["nearest", "bilinear", "bspline"])
def test_scatter_exact_is_permutation_invariant_for_every_kernel(kernel):
    pts, vals = _colliding()
    perm = np.random.default_rng(9).permutation(len(pts))
    a = scatter_exact(pts, vals, (16, 16), kernel=kernel)
    b = scatter_exact(pts[perm], vals[perm], (16, 16), kernel=kernel)
    assert np.array_equal(a, b)


def test_an_exact_nearest_scatter_is_a_permutation_invariant_histogram():
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 8, size=500)
    w = rng.normal(size=500) * 10.0 ** rng.integers(-8, 8, size=500)
    pts = idx[:, None].astype(float)
    perm = rng.permutation(500)

    assert not np.array_equal(scatter(pts, w, (8,), kernel="nearest"),
                              scatter(pts[perm], w[perm], (8,), kernel="nearest"))       # float: 9.31e-09
    assert np.array_equal(scatter_exact(pts, w, (8,), kernel="nearest"),
                          scatter_exact(pts[perm], w[perm], (8,), kernel="nearest"))     # exact: identical


def test_kept_negative_the_exact_scatter_trades_precision_for_reproducibility():
    # Same trade as `scan_exact`: quantization at `bits`. It is not more ACCURATE than the float scatter, and
    # asserting otherwise would be claiming a win it does not have.
    pts, vals = _colliding(n=200, shape=(8, 8))
    a = scatter(pts, vals, (8, 8))
    e = scatter_exact(pts, vals, (8, 8))
    amp = float(np.abs(a).max())
    assert np.abs(a - e).max() > 0.0                          # they differ, by the quantization
    assert np.abs(a - e).max() < 1e-6 * amp                   # ... and only by that


# ---------------------------------------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------------------------------------

def test_wired_to_the_mind_and_discoverable_from_gpu_vocabulary():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    idx = np.random.default_rng(0).integers(0, 8, size=200)
    hist = m.scatter(idx[:, None].astype(float), np.ones(200), (8,), kernel="nearest")
    assert np.array_equal(hist, np.bincount(idx, minlength=8).astype(float))

    pts, vals = _colliding(n=500, shape=(8, 8))
    perm = np.random.default_rng(1).permutation(len(pts))
    assert np.array_equal(m.scatter_exact(pts, vals, (8, 8)),
                          m.scatter_exact(pts[perm], vals[perm], (8, 8)))

    g = m.scatter(np.array([[1.5, 1.5]]), np.array([1.0]), (4, 4))
    assert abs(float(m.gather(g, np.array([[1.5, 1.5]]))[0]) - 0.25) < 1e-12

    for phrase in ("histogram of values", "order independent scatter", "splat values to a grid"):
        assert "Scatter / gather" in str(m.find_capability(phrase)[:3]), phrase

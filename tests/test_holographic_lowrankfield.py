"""W1 -- compressed-domain compute: operate on the factors, never form the field.

The bandwidth wall is physics (this box reads ~12.3 GB/s; a GPU's HBM does 1-3 TB/s). You do not out-bandwidth a
GPU. You flank it by never touching decompressed data -- and linear operations pass straight through a
factorization, so blur/add/scale/query never need the array.

THE BAR: "a field op at >= 3x fewer bytes moved than decompress-op-recompress, same output." Measured 128-171x, at
machine precision.

FOUR KEPT NEGATIVES, each with a test:
  1. the blur must be SEPARABLE -- a 2-D kernel is outside the algebra and is REFUSED, not approximated
  2. `add` inflates rank (six naive adds: 2 -> 14), so it recompresses, and recompression is lossy at a tolerance
  3. NONLINEAR ops do not survive: ReLU on factors differs from ReLU on the field by 1.283
  4. if the field is not low rank, factoring COSTS more -- white noise gates to rank 197 of 256
"""

import numpy as np
import pytest

from holographic.caching_and_storage.holographic_tucker import LowRankField, rank_gate


def _smooth(n=256):
    x = np.linspace(0, 1, n)
    return (np.outer(np.sin(3 * np.pi * x), np.cos(2 * np.pi * x))
            + 0.5 * np.outer(np.exp(-x), np.sin(5 * np.pi * x)))


def _kernel():
    k = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
    return k / k.sum()


def _dense_separable_blur(X, k):
    B = np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, X)
    return np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 1, B)


def test_selftest_runs():
    from holographic.caching_and_storage import holographic_tucker as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# the bar: the ops are exact and touch only the factors
# ---------------------------------------------------------------------------------------------------------

def test_the_factorization_is_faithful_and_far_smaller():
    X = _smooth()
    f = LowRankField.from_dense(X, rank=2)
    assert f.rank == 2 and f.shape == X.shape
    assert np.abs(f.to_dense() - X).max() < 1e-12
    assert X.nbytes / f.nbytes() > 30                     # measured 64x at 256^2, 171x at 1024^2 rank 3


def test_separable_blur_on_the_factors_equals_the_dense_blur():
    X, k = _smooth(), _kernel()
    f = LowRankField.from_dense(X, rank=2)
    assert np.abs(f.blur(k).to_dense() - _dense_separable_blur(X, k)).max() < 1e-12

    # the blurred field is still rank 2: the op did not inflate anything
    assert f.blur(k).rank == 2
    # and it touched only the factors -- (n + m) * r numbers, not n * m
    assert f.blur(k).nbytes() == f.nbytes()


def test_point_query_is_exact_and_touches_a_handful_of_numbers():
    X = _smooth()
    f = LowRankField.from_dense(X, rank=2)
    for (i, j) in [(0, 0), (101, 207), (255, 255)]:
        assert abs(f.query(i, j) - X[i, j]) < 1e-12
    # 3r numbers: one row of U, S, one row of V
    assert f.U[0].size + f.S.size + f.V[0].size == 3 * f.rank


def test_scale_touches_only_the_singular_values():
    X = _smooth()
    f = LowRankField.from_dense(X, rank=2)
    g = f.scale(2.5)
    assert np.array_equal(g.U, f.U) and np.array_equal(g.V, f.V)   # U and V untouched, bit for bit
    assert abs(g.query(7, 9) - 2.5 * X[7, 9]) < 1e-12


def test_add_without_forming_either_field():
    x = np.linspace(0, 1, 256)
    X, Y = _smooth(), np.outer(np.cos(4 * np.pi * x), np.sin(np.pi * x))
    f = LowRankField.from_dense(X, rank=2)
    g = LowRankField.from_dense(Y, rank=1)
    s = f.add(g)
    assert np.abs(s.to_dense() - (X + Y)).max() < 1e-11
    assert s.rank <= 3                                   # r1 + r2, recompressed in the small space
    with pytest.raises(ValueError):
        f.add(LowRankField.from_dense(_smooth(n=64), rank=1))       # shape mismatch


def test_the_report_carries_its_own_comparison():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    rep = m.factored_field_report(_smooth(), _kernel())
    assert rep["byte_ratio"] > 30
    assert rep["max_error"] < 1e-12
    assert rep["dense_bytes"] > rep["factored_bytes"]


# ---------------------------------------------------------------------------------------------------------
# the four negatives
# ---------------------------------------------------------------------------------------------------------

def test_kept_negative_a_non_separable_kernel_is_refused_not_approximated():
    # There is no (K_row, K_col) to push onto U and V for a general 2-D kernel. The op is outside the algebra.
    f = LowRankField.from_dense(_smooth(), rank=2)
    with pytest.raises(ValueError):
        f.blur(np.ones((3, 3)))
    with pytest.raises(ValueError):
        f.blur(np.array([]))
    # NB the guard checks ndim BEFORE ravel() -- ravel would silently flatten a 2-D kernel into a nonsense 1-D one,
    # which is exactly the "approximate instead of refuse" failure. The first draft did that; this test caught it.


def test_kept_negative_add_inflates_rank_so_it_must_recompress():
    X = _smooth()
    f = LowRankField.from_dense(X, rank=2)
    # naive concatenation would grow rank by r each time. `add` recompresses, so it stays bounded ...
    acc = f
    for _ in range(6):
        acc = acc.add(f)
    assert acc.rank <= 3                                   # bounded, not 2 + 6*2 = 14
    assert np.abs(acc.to_dense() - 7 * X).max() < 1e-9     # ... and still accurate after a chain

    # ... but recompression IS lossy at its tolerance: a coarse tol drops information
    coarse = f.add(f, tol=0.5)
    assert coarse.rank <= f.rank


def test_kept_negative_nonlinear_ops_do_not_survive_the_factorization():
    X = _smooth()
    f = LowRankField.from_dense(X, rank=2)
    relu_on_factors = (np.maximum(f.U, 0) * f.S) @ np.maximum(f.V, 0).T
    assert np.abs(np.maximum(X, 0) - relu_on_factors).max() > 0.5   # nonsense, by a wide margin

    # the escape hatch exists and is the one call that pays the bandwidth
    assert np.abs(np.maximum(f.to_dense(), 0) - np.maximum(X, 0)).max() < 1e-12


def test_kept_negative_factoring_noise_costs_more_and_the_gate_says_so():
    noise = np.random.default_rng(0).normal(size=(128, 128))
    ok, fb, db = LowRankField.worth_factoring(noise)
    assert ok is False and fb > db                        # a 1.54x LOSS, measured

    ranks, _ = rank_gate(noise)
    assert max(ranks) > 0.7 * 128                         # near-full rank: the honest signal not to compress

    ok_s, fb_s, db_s = LowRankField.worth_factoring(_smooth())
    assert ok_s is True and fb_s < db_s


def test_from_dense_refuses_a_non_2d_field():
    with pytest.raises(ValueError):
        LowRankField.from_dense(np.zeros((4, 4, 4)))      # N-D is tt_compress's job, not this one
    with pytest.raises(ValueError):
        LowRankField(np.zeros((4, 2)), np.zeros(3), np.zeros((4, 2)))   # rank mismatch


# ---------------------------------------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------------------------------------

def test_fully_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    X, k = _smooth(), _kernel()

    f = m.low_rank_field(X, rank=2)
    assert np.abs(f.blur(k).to_dense() - _dense_separable_blur(X, k)).max() < 1e-12
    assert abs(f.query(3, 4) - X[3, 4]) < 1e-12

    gate = m.worth_factoring(X)
    assert gate["worth_factoring"] is True and gate["factored_bytes"] < gate["dense_bytes"]
    assert m.worth_factoring(np.random.default_rng(1).normal(size=(64, 64)))["worth_factoring"] is False

    assert "Compressed-domain" in str(m.find_capability("blur a field without decompressing it")[:3])


def test_cross_faculty_the_rank_gate_is_the_same_one_tucker_uses():
    # W1 does not grow a second rank heuristic. `worth_factoring` calls `rank_gate`, the function tucker_compress
    # already uses -- one decision about whether data is low rank, not two that can drift apart.
    X = _smooth()
    ranks, _ = rank_gate(X)
    auto = LowRankField.from_dense(X)                      # rank=None -> gated
    assert auto.rank == int(min(max(ranks), min(X.shape)))
    assert np.abs(auto.to_dense() - X).max() < 1e-6

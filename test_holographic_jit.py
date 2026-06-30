"""Tests for the optional Numba accelerator + fast-sweeping eikonal SDF (JIT-1).
Pass with OR without numba installed: the bit-exactness test is skipped when numba is absent."""
import numpy as np
from holographic_jit import (distance_transform, signed_distance_2d,
                            _fast_sweep_2d, _fast_sweep_2d_impl, _BIG, HAS_NUMBA)


def _disk(N=128, R=40.0):
    yy, xx = np.mgrid[0:N, 0:N]
    c = (N - 1) / 2.0
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    return r, (r <= R), R


def test_disk_sdf_matches_analytic():
    r, inside, R = _disk()
    sdf = signed_distance_2d(inside, h=1.0, n_rounds=3)
    analytic = r - R
    band = np.abs(analytic) < 25
    assert np.max(np.abs(sdf[band] - analytic[band])) < 1.5    # within ~1 cell of true distance


def test_inside_negative_outside_positive():
    _, inside, _ = _disk()
    sdf = signed_distance_2d(inside, h=1.0, n_rounds=3)
    assert (sdf[inside] <= 0).all() and (sdf[~inside] >= 0).all()


def test_distance_transform_single_seed():
    mask = np.zeros((21, 21), bool); mask[10, 10] = True
    d = distance_transform(mask, h=1.0, n_rounds=3)
    assert d[10, 10] == 0.0
    assert abs(d[10, 13] - 3.0) < 1e-6                         # 3 cells straight out
    assert d[0, 0] > d[5, 5] > 0                               # grows with distance


def test_jit_equals_pure_when_available():
    if not HAS_NUMBA:
        return                                                 # no numba -> nothing to compare; pure path is used
    _, inside, _ = _disk()
    d_pure = _fast_sweep_2d_impl(np.where(inside, 0.0, _BIG), 1.0, 2)
    d_jit = _fast_sweep_2d(np.where(inside, 0.0, _BIG), 1.0, 2)
    assert np.allclose(d_pure, d_jit, atol=1e-9)               # JIT is bit-faithful (no parallel/fastmath)


def test_pure_impl_callable_directly():
    # the pure source is always available as a fallback regardless of numba
    d = _fast_sweep_2d_impl(np.where(np.eye(10, dtype=bool), 0.0, _BIG), 1.0, 2)
    assert d.shape == (10, 10) and d.min() == 0.0

"""Tests for the 3-D fast-sweeping eikonal SDF (3-D twin of signed_distance_2d)."""
import numpy as np
from holographic_jit import signed_distance_3d, distance_transform_3d, _fast_sweep_3d, _fast_sweep_3d_impl, _BIG, HAS_NUMBA


def test_ball_sdf_within_one_cell():
    N, R = 40, 12.0
    zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]; c = (N - 1) / 2.0
    r = np.sqrt((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2)
    sdf = signed_distance_3d(r <= R, h=1.0, n_rounds=3)
    band = np.abs(r - R) < 6
    assert np.max(np.abs(sdf[band] - (r - R)[band])) < 1.5          # within ~1 cell of analytic r-R


def test_inside_is_negative_outside_positive():
    N, R = 24, 7.0
    zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]; c = (N - 1) / 2.0
    r = np.sqrt((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2)
    sdf = signed_distance_3d(r <= R, h=1.0, n_rounds=2)
    assert sdf[int(c), int(c), int(c)] < 0                         # centre is inside
    assert sdf[0, 0, 0] > 0                                        # corner is outside


def test_jit_equals_pure_bit_exact():
    if not HAS_NUMBA:
        return
    N = 32
    zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]; c = (N - 1) / 2.0
    inside = np.sqrt((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2) <= 9
    seed = np.where(~inside, 0.0, _BIG)
    assert np.allclose(_fast_sweep_3d(seed.copy(), 1.0, 1), _fast_sweep_3d_impl(seed.copy(), 1.0, 1), atol=1e-9)


def test_distance_transform_seed_is_zero():
    N = 16
    seed = np.zeros((N, N, N), bool); seed[8, 8, 8] = True
    d = distance_transform_3d(seed, h=1.0, n_rounds=2)
    assert d[8, 8, 8] == 0.0
    assert abs(d[8, 8, 11] - 3.0) < 1e-6                           # 3 cells along one axis

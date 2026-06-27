"""Tests for anisotropic / steering kernels (RT-IV1): the FPE per-axis bandwidth, the steering-fit, and the
measured win on dense directional data -- plus the kept negatives (isotropic data: no help; flat axis: no nan)."""

import numpy as np

from holographic_fpe import VectorFunctionEncoder
from holographic_steering import steer_bandwidths, kernel_regress, _best_rmse

BOUNDS = [(0, 10), (0, 10)]


def test_fpe_per_axis_bandwidth_is_anisotropic():
    # a per-axis bandwidth makes the kernel fall off differently per axis (smooth where bw is small)
    ani = VectorFunctionEncoder(2, dim=2048, bounds=BOUNDS, bandwidth=[0.8, 4.0], seed=1)
    assert ani.kernel_at([1.0, 0.0]) > ani.kernel_at([0.0, 1.0])     # smooth in x (small bw), sharp in y
    # a scalar bandwidth is isotropic: equal falloff on both axes
    iso = VectorFunctionEncoder(2, dim=2048, bounds=BOUNDS, bandwidth=2.0, seed=1)
    assert abs(iso.kernel_at([1.0, 0.0]) - iso.kernel_at([0.0, 1.0])) < 0.02


def test_fpe_scalar_bandwidth_backward_compatible():
    # passing a scalar still works and broadcasts to all axes (the original behaviour)
    enc = VectorFunctionEncoder(2, dim=1024, bounds=BOUNDS, bandwidth=3.0, seed=1)
    assert enc.bandwidth == [3.0, 3.0]


def test_per_axis_bandwidth_length_checked():
    try:
        VectorFunctionEncoder(2, dim=512, bounds=BOUNDS, bandwidth=[1.0, 2.0, 3.0], seed=1)
        assert False, "should reject a per-axis bandwidth of the wrong length"
    except ValueError:
        pass


def _ridge_data(grid_n=14):
    def f(p):
        return np.tanh(3.0 * (p[1] - 5.0))                           # constant in x, sharp edge in y
    g = np.linspace(0.5, 9.5, grid_n)
    Xtr = np.array([[x, y] for x in g for y in g])
    ytr = np.array([f(p) for p in Xtr])
    rng = np.random.default_rng(0)
    Xte = rng.uniform(1, 9, (120, 2))
    yte = np.array([f(p) for p in Xte])
    return Xtr, ytr, Xte, yte


def test_anisotropic_beats_isotropic_on_dense_directional_data():
    # THE BAR: on a dense, strongly-directional ridge, the best anisotropic kernel beats the best isotropic one.
    Xtr, ytr, Xte, yte = _ridge_data()
    grid = [0.2, 0.5, 1.0, 2.0, 4.0, 7.0]
    iso = _best_rmse(BOUNDS, Xtr, ytr, Xte, yte, [b for b in grid])
    ani = _best_rmse(BOUNDS, Xtr, ytr, Xte, yte, [[bx, by] for bx in grid for by in grid])
    assert ani < iso * 0.97                                          # a clear margin (~8% measured)


def test_steering_recovers_the_right_direction_on_dense_data():
    # the flat axis (x) should get a smaller bandwidth (smoother) than the sharp axis (y)
    Xtr, ytr, _, _ = _ridge_data()
    bw = steer_bandwidths(Xtr, ytr, base=2.0)
    assert bw[0] < bw[1]


def test_steering_handles_a_perfectly_flat_axis():
    # a perfectly flat axis gives gradient 0 -- must not produce nan/inf (the divide-by-zero guard)
    Xtr, ytr, _, _ = _ridge_data()
    bw = steer_bandwidths(Xtr, ytr, base=2.0)
    assert all(np.isfinite(b) and b > 0 for b in bw)


def test_isotropic_data_gives_no_anisotropic_advantage_kept_negative():
    # THE KEPT NEGATIVE: when the data varies equally on both axes, anisotropy must NOT help meaningfully.
    def f(p):
        return np.sin(2.0 * p[0]) + np.sin(2.0 * p[1])
    g = np.linspace(0.5, 9.5, 14)
    Xtr = np.array([[x, y] for x in g for y in g])
    ytr = np.array([f(p) for p in Xtr])
    rng = np.random.default_rng(1)
    Xte = rng.uniform(1, 9, (120, 2))
    yte = np.array([f(p) for p in Xte])
    grid = [0.5, 1.0, 2.0, 4.0]
    iso = _best_rmse(BOUNDS, Xtr, ytr, Xte, yte, [b for b in grid])
    ani = _best_rmse(BOUNDS, Xtr, ytr, Xte, yte, [[bx, by] for bx in grid for by in grid])
    assert ani > iso * 0.95                                          # anisotropy barely helps on isotropic data

"""Tests for auto-bandwidth KDE via the encoder (holographic_kde): the disciplined form of the band-limited-encoding
faculty, landed on the encoder's documented RBF-as-KDE use. LCV bandwidth selection matches the kernel to the data
and beats the fixed default several-fold."""

import numpy as np

from holographic_kde import kde_bandwidth, density_estimate, _silverman_bandwidth


def _gauss(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def _sample(density_fn, n, ceil, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        c = rng.uniform(0, 1)
        if rng.uniform(0, ceil) < density_fn(c):
            out.append(c)
    return np.array(out)


def _shape_rmse(est, truth):
    a = np.sum(est * truth) / np.sum(est * est) if np.sum(est * est) > 0 else 0.0
    return np.sqrt(np.mean((a * est - truth) ** 2))


_QX = np.linspace(0.02, 0.98, 200)
_BIMODAL = lambda x: 0.5 * _gauss(x, 0.3, 0.05) + 0.5 * _gauss(x, 0.7, 0.07)
_UNIMODAL = lambda x: _gauss(x, 0.5, 0.12)


def test_lcv_beats_default_on_bimodal():
    xs = _sample(_BIMODAL, 400, 6.0, 0)
    truth = _BIMODAL(_QX)
    e_def = _shape_rmse(density_estimate(xs, 0, 1, _QX, bandwidth=1.8)[0], truth)
    e_lcv = _shape_rmse(density_estimate(xs, 0, 1, _QX, method="lcv")[0], truth)
    assert e_lcv < e_def / 3


def test_lcv_lands_near_optimum_bimodal():
    xs = _sample(_BIMODAL, 400, 6.0, 0)
    truth = _BIMODAL(_QX)
    grid = np.linspace(2, 80, 60)
    opt = grid[int(np.argmin([_shape_rmse(density_estimate(xs, 0, 1, _QX, bandwidth=b)[0], truth) for b in grid]))]
    assert abs(kde_bandwidth(xs, 0, 1, "lcv") - opt) < 12


def test_lcv_lands_near_optimum_unimodal():
    us = _sample(_UNIMODAL, 300, 4.0, 7)
    truth = _UNIMODAL(_QX)
    grid = np.linspace(2, 80, 60)
    opt = grid[int(np.argmin([_shape_rmse(density_estimate(us, 0, 1, _QX, bandwidth=b)[0], truth) for b in grid]))]
    assert abs(kde_bandwidth(us, 0, 1, "lcv") - opt) < 12


def test_estimate_correlates_with_truth():
    xs = _sample(_BIMODAL, 400, 6.0, 0)
    est, _ = density_estimate(xs, 0, 1, _QX, method="lcv")
    assert np.corrcoef(est, _BIMODAL(_QX))[0, 1] > 0.95


def test_silverman_beats_default_but_worse_than_lcv():
    xs = _sample(_BIMODAL, 400, 6.0, 0)
    truth = _BIMODAL(_QX)
    e_def = _shape_rmse(density_estimate(xs, 0, 1, _QX, bandwidth=1.8)[0], truth)
    e_sil = _shape_rmse(density_estimate(xs, 0, 1, _QX, method="silverman")[0], truth)
    e_lcv = _shape_rmse(density_estimate(xs, 0, 1, _QX, method="lcv")[0], truth)
    assert e_sil < e_def and e_sil > e_lcv               # principled fallback, but over-smooths multimodal


def test_density_integrates_to_about_one():
    xs = _sample(_BIMODAL, 400, 6.0, 0)
    grid = np.linspace(0, 1, 400)
    d, _ = density_estimate(xs, 0, 1, grid, method="lcv")
    area = np.sum((d[:-1] + d[1:]) * 0.5 * np.diff(grid))
    assert 0.9 < area < 1.1


def test_capacity_negative_small_dim_is_worse():
    xs = _sample(_BIMODAL, 400, 6.0, 0)
    bw = kde_bandwidth(xs, 0, 1, "lcv")
    big = np.corrcoef(density_estimate(xs, 0, 1, _QX, dim=1024, bandwidth=bw)[0], _BIMODAL(_QX))[0, 1]
    small = np.corrcoef(density_estimate(xs, 0, 1, _QX, dim=16, bandwidth=bw)[0], _BIMODAL(_QX))[0, 1]
    assert big > small                                   # bandwidth can't rescue too small a dimension


def test_silverman_bandwidth_is_a_number():
    xs = _sample(_BIMODAL, 400, 6.0, 0)
    assert _silverman_bandwidth(xs, 0, 1) > 0


def test_deterministic():
    xs = _sample(_BIMODAL, 400, 6.0, 0)
    assert kde_bandwidth(xs, 0, 1, "lcv") == kde_bandwidth(xs, 0, 1, "lcv")
    assert np.array_equal(density_estimate(xs, 0, 1, _QX)[0], density_estimate(xs, 0, 1, _QX)[0])

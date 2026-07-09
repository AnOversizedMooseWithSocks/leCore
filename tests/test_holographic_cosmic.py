"""Tests for local structure classification extracted from leOS cosmic_web (COSMIC-1)."""
import numpy as np
from holographic.misc.holographic_cosmic import local_structure, classify_cloud, participation_ratio, _local_pca, VOID, FILAMENT, WALL, NODE


def _embed(low, D=32, noise=0.0005, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((D, D)))[0][:, :low.shape[1]]
    return low @ Q.T + noise * rng.standard_normal((low.shape[0], D))


def _mean_pr(cloud, k, m=30):
    idx = np.linspace(20, len(cloud) - 20, m).astype(int)
    return float(np.mean([participation_ratio(_local_pca(cloud[i], cloud, k)[0]) for i in idx]))


def test_intrinsic_dim_tracks_true_dimension():
    rng = np.random.default_rng(1)
    fil = _embed(np.linspace(0, 4, 300)[:, None], seed=1)
    sheet = _embed(rng.uniform(-1, 1, (400, 2)), seed=2)
    blob = _embed(rng.uniform(-1, 1, (500, 3)), seed=3)
    df, ds, db = _mean_pr(fil, 12), _mean_pr(sheet, 14), _mean_pr(blob, 18)
    assert df < ds < db and df < 1.4 and db > 2.2            # 1-D < 2-D < 3-D, monotonic (finite-sample under-est)


def test_one_d_cloud_is_mostly_filament():
    fil = _embed(np.linspace(0, 4, 300)[:, None], seed=4)
    _, _, summary = classify_cloud(fil, k=12)
    assert summary[FILAMENT] > 0.6


def test_two_d_cloud_is_not_filament():
    rng = np.random.default_rng(5)
    sheet = _embed(rng.uniform(-1, 1, (400, 2)), seed=5)
    _, _, summary = classify_cloud(sheet, k=14)
    assert summary[WALL] + summary[NODE] > summary[FILAMENT]


def test_local_structure_returns_fields():
    rng = np.random.default_rng(6)
    cloud = _embed(rng.uniform(-1, 1, (200, 2)), seed=6)
    info = local_structure(cloud[100], cloud, k=12)
    assert info["type"] in (VOID, FILAMENT, WALL, NODE)
    assert info["intrinsic_dim"] >= 0.0 and info["n_neighbors"] >= 3


def test_noise_inflates_dimension_kept_negative():
    fil = _embed(np.linspace(0, 4, 300)[:, None], noise=0.0005, seed=7)
    fil_noisy = _embed(np.linspace(0, 4, 300)[:, None], noise=0.01, seed=7)
    assert _mean_pr(fil_noisy, 12) > _mean_pr(fil, 12)      # documents the noise-inflation limitation

"""Tests for holographic_samplinghome -- the Sampling home (R4: patterns/directions/MIS/accumulation, one home)."""
import numpy as np
from holographic_samplinghome import Sampling, sampling_backends


def test_low_discrepancy_routes_exactly():
    from holographic_lowdiscrepancy import low_discrepancy
    assert np.array_equal(Sampling.low_discrepancy(24, d=2, seed=5), low_discrepancy(24, d=2, seed=5))


def test_cosine_hemisphere_unit_and_upper():
    N = np.array([[0., 1., 0.], [0.6, 0.8, 0.], [0., 0., 1.]])
    dirs = Sampling.cosine_hemisphere(N, 64, seed=0)
    assert dirs.shape == (3, 64, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=2), 1.0, atol=1e-6)
    assert (np.einsum("mnk,mk->mn", dirs, N) >= -1e-6).all()      # never below the surface


def test_cosine_hemisphere_deterministic():
    N = np.array([[0., 1., 0.]])
    assert np.array_equal(Sampling.cosine_hemisphere(N, 32, seed=1), Sampling.cosine_hemisphere(N, 32, seed=1))


def test_cosine_hemisphere_matches_globalillum_and_lightcache_source():
    # the three former copies now share ONE implementation, bit-identical
    from holographic_globalillum import _cosine_hemisphere
    N = np.array([[0., 1., 0.], [0.3, 0.7, 0.65]])
    assert np.array_equal(_cosine_hemisphere(N, 40, seed=9), Sampling.cosine_hemisphere(N, 40, seed=9))


def test_accumulate_clamps_firefly():
    clean = np.ones((4, 4, 3)) * 0.5
    acc = Sampling.accumulate([clean, clean, clean + 5.0, clean], schedule="mean", clamp_k=2.5)
    assert abs(float(acc.mean()) - 0.5) < 0.3                     # not dragged toward the naive ~1.6


def test_poisson_disk_routes():
    from holographic_sampling import poisson_disk_sample
    a = Sampling.poisson_disk(0.1, ((0, 0), (1, 1)), seed=0)
    b = poisson_disk_sample(0.1, ((0, 0), (1, 1)), k=30, seed=0)
    assert np.array_equal(a, b)


def test_backends_listed():
    assert set(sampling_backends()) == {"low_discrepancy", "poisson_disk", "cosine_hemisphere", "mis_weight", "accumulate"}

"""Sweep 3 item 2: the shared spatial index matches brute force byte-for-byte."""
import numpy as np
from holographic_spatial import SpatialGrid, brute_radius, brute_knn


def _pts(rng, n, d): return rng.uniform(0, 10, size=(n, d))


def test_radius_matches_brute():
    rng = np.random.default_rng(0)
    for D in (2, 3):
        pts = _pts(rng, 300, D); g = SpatialGrid(pts, 1.0)
        for _ in range(20):
            q = rng.uniform(0, 10, D); r = float(rng.uniform(0.3, 2.5))
            assert g.radius(q, r) == brute_radius(pts, q, r)


def test_knn_and_closest_match_brute():
    rng = np.random.default_rng(1)
    for D in (2, 3):
        pts = _pts(rng, 300, D); g = SpatialGrid(pts, 1.0)
        for _ in range(20):
            q = rng.uniform(0, 10, D); k = int(rng.integers(1, 10))
            assert g.knn(q, k) == brute_knn(pts, q, k)
            assert g.closest(q)[0] == brute_knn(pts, q, 1)[0]


def test_clustered_and_empty():
    rng = np.random.default_rng(2)
    clustered = np.concatenate([rng.normal(2, 0.2, (150, 2)), rng.normal(8, 0.2, (150, 2))])
    g = SpatialGrid(clustered, 0.5)
    assert g.knn([2.0, 2.0], 5) == brute_knn(clustered, [2.0, 2.0], 5)
    empty = SpatialGrid(np.zeros((0, 3)), 1.0)
    assert empty.closest([0, 0, 0]) == (-1, float("inf")) and empty.knn([0, 0, 0], 3) == []


def test_deterministic():
    rng = np.random.default_rng(3); pts = _pts(rng, 200, 3)
    a = SpatialGrid(pts, 1.0); b = SpatialGrid(pts, 1.0)
    assert a.radius([5, 5, 5], 1.5) == b.radius([5, 5, 5], 1.5)

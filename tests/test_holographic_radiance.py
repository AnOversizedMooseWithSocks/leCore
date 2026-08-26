"""Tests for holographic_radiance: scene radiance as a queryable field, tiling that breaks the capacity wall, deltas."""
import numpy as np
from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
from holographic.rendering.holographic_radiance import HolographicRadianceField, TiledRadianceField, reconstruct_view


def _smooth_samples(n=400, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1.5, 1.5, (n, 3))
    cols = np.clip(0.5 + 0.4 * np.sin(pts * 1.5), 0.0, 1.0)
    return pts, cols


def test_bake_query_recovers_radiance():
    """A baked radiance field returns the baked colours when queried at the sample points (sharp kernel)."""
    pts, cols = _smooth_samples()
    enc = VectorFunctionEncoder(3, dim=2048, bounds=[(-2, 2)] * 3, kernel="rbf", bandwidth=16.0, seed=0)
    rad = HolographicRadianceField(enc, pts, cols)
    rgb, cov = rad.query(pts)
    assert np.abs(rgb - cols).mean() < 0.12


def test_empty_space_reads_low_coverage():
    """A point far from any sample (inside bounds) has much lower coverage than an occupied point."""
    pts, cols = _smooth_samples()
    enc = VectorFunctionEncoder(3, dim=1024, bounds=[(-3, 3)] * 3, kernel="rbf", bandwidth=16.0, seed=0)
    rad = HolographicRadianceField(enc, pts, cols)
    _, cov_occ = rad.query(pts[:50])
    _, cov_empty = rad.query(np.array([[2.7, 2.7, 2.7]]))   # corner, far from the centred samples, still in bounds
    assert abs(cov_empty[0]) < np.abs(cov_occ).mean() * 0.5


def test_tiling_beats_single_vector_capacity():
    """Tiling the radiance field into bricks reconstructs better than one vector at the same total budget -- the
    capacity wall is per-vector, and tiling moves it (refine the grid -> higher fidelity)."""
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1.5, 1.5, (6000, 3))                 # many samples -> over a single vector's capacity
    cols = np.clip(0.5 + 0.4 * np.sin(pts * 2.0), 0.0, 1.0)
    bounds = [(-2, 2)] * 3
    single = HolographicRadianceField(VectorFunctionEncoder(3, dim=1024, bounds=bounds, bandwidth=16.0, seed=0), pts, cols)
    rgb_s, _ = single.query(pts)
    err_single = np.abs(rgb_s - cols).mean()
    tiled = TiledRadianceField(bounds, grid=8, dim=512, bandwidth=18.0, halo=1, seed=0).bake(pts, cols)
    rgb_t, _ = tiled.query(pts)
    err_tiled = np.abs(rgb_t - cols).mean()
    assert err_tiled < err_single                            # tiling (smaller per-brick dim!) beats one big vector
    assert tiled.n_bricks() > 1


def test_delta_rebuild_is_local():
    """Rebuilding one brick changes only that brick's queries -- an O(change) update, not a global rebuild."""
    rng = np.random.default_rng(2)
    pts = rng.uniform(-1.5, 1.5, (2000, 3)); cols = np.clip(0.5 + 0.4 * np.sin(pts), 0, 1)
    bounds = [(-2, 2)] * 3
    tiled = TiledRadianceField(bounds, grid=6, dim=512, bandwidth=14.0, halo=1, seed=0).bake(pts, cols)
    ci = tiled._cell_of(pts)
    touch = tuple(ci[0])
    before = tiled.query(pts)[0].copy()
    cols2 = cols.copy(); cols2[np.all(ci == np.array(touch), axis=1)] = np.array([1.0, 0.0, 0.0])
    tiled.rebuild_cells(pts, cols2, [touch])
    after = tiled.query(pts)[0]
    changed = np.abs(after - before).max(1) > 1e-6
    assert changed.any() and not changed.all()              # some queries changed, but not all -- local update


def test_reconstruct_view_runs():
    """reconstruct_view returns a frame from field queries at hit points."""
    pts, cols = _smooth_samples()
    enc = VectorFunctionEncoder(3, dim=1024, bounds=[(-2, 2)] * 3, bandwidth=16.0, seed=0)
    rad = HolographicRadianceField(enc, pts, cols)
    hits = np.tile(pts[:64], (4, 1))[:256]
    img = reconstruct_view(rad, hits, np.ones(256, bool), 16, 16)
    assert img.shape == (16, 16, 3)

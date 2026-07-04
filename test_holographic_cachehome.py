"""Tests for holographic_cachehome -- the Cache home (H2: bake-and-query, one shared grid core)."""
import numpy as np
from holographic_cachehome import Cache, BakedGrid, cache_backends


def test_grid_points_matches_inline_meshgrid():
    lo = np.array([-1., -0.5, -1.2]); hi = np.array([1., 0.7, 1.3]); res = 10
    pts, r = Cache.grid_points(lo, hi, res)
    axes = [np.linspace(lo[k], hi[k], res) for k in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    ref = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    assert np.array_equal(pts, ref) and tuple(r) == (res, res, res)


def test_per_axis_resolution():
    lo = np.zeros(3); hi = np.ones(3)
    pts, r = Cache.grid_points(lo, hi, (4, 6, 8))
    assert tuple(r) == (4, 6, 8) and len(pts) == 4 * 6 * 8


def test_bake_grid_scalar_and_channels():
    lo = -np.ones(3); hi = np.ones(3)
    g_scalar, _ = Cache.bake_grid(lambda P: P[:, 0], lo, hi, 8)
    g_col, _ = Cache.bake_grid(lambda P: P, lo, hi, 8)
    assert g_scalar.shape == (8, 8, 8) and g_col.shape == (8, 8, 8, 3)


def test_trilinear_exact_at_nodes():
    lo = -np.ones(3); hi = np.ones(3); res = 8
    fn = lambda P: P[:, 0] ** 2 + P[:, 1]
    bg = Cache.bake(fn, vary="position", lo=lo, hi=hi, res=res)
    pts, _ = Cache.grid_points(lo, hi, res)
    node = pts[123][None, :]
    assert abs(float(bg.sample(node)[0]) - float(fn(node)[0])) < 1e-9


def test_constant_strategy():
    assert Cache.bake(lambda: 3.14, vary="constant") == 3.14


def test_matbake_and_sdfbake_delegate_bit_identical():
    # the H2 done-when: two bakes route through Cache.grid_points with bit-identical output
    from holographic_matbake import bake_field
    from holographic_sdfbake import bake_sdf_grid
    lo = np.array([-1.2, -0.8, -1.0]); hi = np.array([1.1, 0.9, 1.3]); res = 16
    fn = lambda P: P[:, 0] * 0.7 - np.sin(P[:, 1] * 3.0)
    bf = bake_field(fn, "roughness", lo, hi, res=res)
    axes = [np.linspace(lo[k], hi[k], res) for k in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    ref = fn(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)).reshape(res, res, res)
    assert np.array_equal(bf.grid, ref)
    sdf = lambda P: np.linalg.norm(P, axis=1) - 0.6
    dist, _ = bake_sdf_grid(sdf, lo, hi, res)
    dref = np.asarray(sdf(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)), float).reshape(res, res, res)
    assert np.array_equal(dist, dref)


def test_backends_listed():
    assert set(cache_backends()) == {"constant", "position", "view", "time"}

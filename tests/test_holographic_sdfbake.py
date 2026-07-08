"""Baked SDF grid: O(1) sampling that reproduces the analytic field, independent of scene complexity."""
import numpy as np
from holographic.mesh_and_geometry.holographic_sdfbake import GridSDF, bake_sdf_grid


def _u(fn):
    return type("U", (), {"eval": staticmethod(fn)})()


def test_baked_grid_matches_analytic():
    sphere = lambda P: np.linalg.norm(P, axis=1) - 1.0
    lo = np.full(3, -2.0); hi = np.full(3, 2.0)
    gs = GridSDF.bake(_u(sphere), lo, hi, 64)
    pts = np.random.default_rng(0).uniform(-1.8, 1.8, size=(4000, 3))
    assert np.abs(gs.eval(pts) - sphere(pts)).max() < 3 * ((hi - lo).max() / 63)   # accurate to ~a cell


def test_baked_sampling_is_O1_in_primitive_count():
    import time
    class Union:
        def __init__(s, n): s.cs = np.random.default_rng(1).uniform(-1, 1, (n, 3))
        def eval(s, P): return np.min(np.stack([np.linalg.norm(P - c, axis=1) - 0.4 for c in s.cs]), axis=0)
    lo = np.full(3, -2.0); hi = np.full(3, 2.0)
    big = Union(60); gs = GridSDF.bake(big, lo, hi, 48)
    q = np.random.default_rng(2).uniform(-1.8, 1.8, (20000, 3))
    t = time.time(); [big.eval(q) for _ in range(5)]; t_analytic = time.time() - t
    t = time.time(); [gs.eval(q) for _ in range(5)]; t_baked = time.time() - t
    assert t_baked < t_analytic                                    # one sample beats evaluating 60 primitives


def test_outside_box_is_conservative():
    sphere = lambda P: np.linalg.norm(P, axis=1) - 1.0
    lo = np.full(3, -2.0); hi = np.full(3, 2.0)
    gs = GridSDF.bake(_u(sphere), lo, hi, 32)
    far = np.array([[10.0, 0.0, 0.0]])                             # well outside the baked box
    assert gs.eval(far)[0] > 0                                     # positive (never overshoots into the volume)

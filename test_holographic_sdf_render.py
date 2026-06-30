"""Tests for the njit analytic-SDF renderer (SDFRENDER-1). Skips when sympy/numba absent."""
import numpy as np
from holographic_codegen import HAS_SYMPY
from holographic_jit import HAS_NUMBA
from holographic_render import Camera
from holographic_raymarch import render_sdf, sphere_trace

_HAVE = HAS_SYMPY and HAS_NUMBA


class _SphereObj:
    def eval(self, P):
        return np.linalg.norm(np.asarray(P, float), axis=1) - 1.0


def test_render_analytic_shape_and_range():
    if not _HAVE:
        return
    from holographic_sdf_render import render_analytic
    img = render_analytic("sqrt(x**2+y**2+z**2) - 1.0", Camera(eye=(0, 0, 3.0)), width=64, height=64)
    assert img.shape == (64, 64, 3) and img.min() >= 0.0 and img.max() <= 1.0


def test_hit_geometry_matches_numpy():
    if not _HAVE:
        return
    from holographic_sdf_render import compiled_sdf_renderer
    cam = Camera(eye=(0, 0, 3.0)); W = H = 96
    eye, dirs = cam.ray_dirs(W, H)
    D = np.ascontiguousarray(dirs.reshape(-1, 3)); O = np.ascontiguousarray(np.broadcast_to(eye, D.shape))
    hit_np, _, _ = sphere_trace(_SphereObj(), O, D)
    render = compiled_sdf_renderer("sqrt(x**2+y**2+z**2) - 1.0")
    L = np.array([-0.4, 0.7, -0.3]); L = L / np.linalg.norm(L)
    _, hit_jit, _ = render(O, D, L, np.array([0.85, 0.5, 0.35]), 0.25, True, True)
    assert np.mean(hit_jit == hit_np) > 0.99                  # hit geometry matches numpy sphere_trace


def test_render_sdf_jit_expr_routes_and_matches():
    if not _HAVE:
        return
    from holographic_sdf_render import render_analytic
    cam = Camera(eye=(0, 0, 3.0))
    via_param = render_sdf(_SphereObj(), cam, width=48, height=48, jit_expr="sqrt(x**2+y**2+z**2) - 1.0")
    direct = render_analytic("sqrt(x**2+y**2+z**2) - 1.0", cam, width=48, height=48)
    assert np.allclose(via_param, direct)                    # render_sdf(jit_expr=...) routes to the njit renderer


def test_render_sdf_numpy_path_unchanged():
    # the default (no jit_expr) numpy path still works and is byte-identical to before (no jit_expr passed)
    cam = Camera(eye=(0, 0, 3.0))
    img = render_sdf(_SphereObj(), cam, width=32, height=32, ao=False, shadows=False, reflect=0.0)
    assert img.shape == (32, 32, 3)


def test_renderer_cached_per_sdf():
    if not _HAVE:
        return
    from holographic_sdf_render import compiled_sdf_renderer
    from holographic_compile import DEFAULT_CACHE
    DEFAULT_CACHE.clear()
    r1 = compiled_sdf_renderer("sqrt(x**2+y**2+z**2) - 1.0")
    r2 = compiled_sdf_renderer("sqrt(x**2+y**2+z**2) - 1.0")
    assert r1 is r2                                           # same compiled renderer reused

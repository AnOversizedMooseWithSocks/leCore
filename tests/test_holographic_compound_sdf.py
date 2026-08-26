"""Tests for compound SDF combinators + the gradient-cache exact-Jacobian win (SWEEP-1)."""
import numpy as np
from holographic.misc.holographic_codegen import HAS_SYMPY, sphere, box, op_union, op_intersect, op_subtract, op_smooth_union, sdf_numba_fn
from holographic.misc.holographic_jit import HAS_NUMBA
from holographic.caching_and_storage.holographic_cache import gradient_cache_fd, gradient_cache_symbolic

_HAVE = HAS_SYMPY and HAS_NUMBA


def test_union_value_is_min_of_parts():
    if not _HAVE:
        return
    s1 = sphere((-0.6, 0, 0), 0.8); s2 = sphere((0.6, 0, 0), 0.8)
    d = sdf_numba_fn(op_union(s1, s2))
    P = np.array([[-0.6, 0, 0], [0.6, 0, 0], [3.0, 0, 0]])
    v = d["grid_value"](P)
    assert abs(v[0] + 0.8) < 1e-9 and abs(v[1] + 0.8) < 1e-9   # inside each sphere by 0.8
    assert v[2] > 0                                            # outside both


def test_compound_with_box_uses_fd_normal_fallback():
    if not _HAVE:
        return
    scene = op_subtract(op_union(sphere((-0.6, 0, 0), 0.8), box((0.7, 0, 0), (0.5, 0.5, 0.5))),
                        sphere((0.7, 0.3, 0.4), 0.4))
    d = sdf_numba_fn(scene)
    assert d["exact_normal"] is False                         # box's nested Min/Max -> FD normal fallback
    P = np.array([[-1.4, 0, 0]])                              # ON the sphere-1 surface (not its centre)
    n = d["grid_normal"](P)
    assert abs(np.linalg.norm(n[0]) - 1.0) < 1e-5             # still a unit normal


def test_single_primitive_keeps_exact_normal():
    if not _HAVE:
        return
    d = sdf_numba_fn(sphere((0, 0, 0), 1.0))
    assert d["exact_normal"] is True                          # a plain sphere keeps the exact symbolic normal
    P = np.array([[2.0, 0, 0]])
    assert np.allclose(d["grid_normal"](P), [[1.0, 0, 0]], atol=1e-10)


def test_compound_renders_and_matches_numpy():
    if not _HAVE:
        return
    from holographic.rendering.holographic_render import Camera
    from holographic.rendering.holographic_raymarch import sphere_trace
    from holographic.rendering.holographic_sdf_render import compiled_sdf_renderer
    scene = op_union(sphere((-0.5, 0, 0), 0.7), sphere((0.5, 0, 0), 0.7))

    def py(P):
        P = np.asarray(P, float)
        return np.minimum(np.linalg.norm(P - [-0.5, 0, 0], axis=1) - 0.7,
                          np.linalg.norm(P - [0.5, 0, 0], axis=1) - 0.7)

    class O:
        def eval(self, P): return py(P)
    cam = Camera(eye=(0, 0, 3.0)); W = H = 80
    eye, dirs = cam.ray_dirs(W, H); D = np.ascontiguousarray(dirs.reshape(-1, 3)); Oo = np.ascontiguousarray(np.broadcast_to(eye, D.shape))
    hn, _, _ = sphere_trace(O(), Oo, D)
    L = np.array([-0.4, 0.7, -0.3]); L = L / np.linalg.norm(L)
    _, hj, _ = compiled_sdf_renderer(scene)(Oo, D, L, np.array([0.85, 0.5, 0.35]), 0.25, True, True)
    assert np.mean(hj == hn) > 0.99                           # compound hit geometry matches numpy


def test_gradient_cache_symbolic_exact_vs_fd():
    if not HAS_SYMPY:
        return
    anchors = np.random.default_rng(0).uniform(-1, 1, (15, 2))

    def fld(a): return np.sin(a[0]) * np.cos(a[1])
    c_fd = gradient_cache_fd(fld, anchors, eps=1e-3)
    c_sym = gradient_cache_symbolic("sin(x)*cos(y)", anchors, variables=("x", "y"))
    ax, ay = anchors[:, 0], anchors[:, 1]
    analytic = np.stack([np.cos(ax) * np.cos(ay), -np.sin(ax) * np.sin(ay)], axis=1)
    assert np.max(np.abs(c_sym.jacobians - analytic)) < 1e-12     # symbolic is exact
    assert np.max(np.abs(c_fd.jacobians - analytic)) > 1e-9       # FD carries truncation error

"""Tests for holographic_raycoherence: a bounce is a transform of its parent; coherent reflection traces fewer rays."""
import numpy as np
from holographic.rendering.holographic_raycoherence import reflect_transform, trace_reflection_color, coherent_reflection
from holographic.simulation_and_physics.holographic_semantic import _scene_setup, parse_description, realize_scene
from holographic.rendering.holographic_render import Camera
from holographic.rendering.holographic_raymarch import sphere_trace, sdf_normal


def test_reflect_transform_flips_about_normal_and_increments_bounce():
    O2, D2, b2 = reflect_transform(np.zeros((1, 3)), np.array([[0, -1.0, 0]]),
                                   np.array([[0, 0, 0.0]]), np.array([[0, 1.0, 0]]), bounce=np.array([0]))
    assert D2[0, 1] > 0.99                                      # a downward ray off an up-normal reflects upward
    assert b2[0] == 1                                          # the bounce counter incremented


def _gbuffer(desc, eye, W, H):
    objs = parse_description(desc)["objects"]; rs = realize_scene(objs)
    ctx = _scene_setup(None, True, "clear", "bright", (0.75, 0.9, 0.85), rs=rs)
    cam = Camera(eye=eye, target=(0, 0.1, 0), fov_deg=48)
    e, dirs = cam.ray_dirs(W, H); O = np.broadcast_to(e, (W * H, 3)).astype(float); D = dirs.reshape(-1, 3)
    union = ctx["union"]; hit, t, Pp = sphere_trace(union, O, D)
    P = np.zeros((W * H, 3)); N = np.zeros((W * H, 3)); ids = -np.ones(W * H, int)
    P[hit] = Pp[hit]; N[hit] = sdf_normal(union, Pp[hit]); ids[hit] = union.ids(Pp[hit])
    mirror = np.zeros(W * H, bool); mirror[hit] = ctx["refl"][ids[hit]] > 0.05
    return ctx, P, N, D, ids, mirror


def test_coherent_reflection_traces_fewer_rays_on_smooth_mirror():
    W = H = 96
    ctx, P, N, D, ids, mirror = _gbuffer("a huge mirror ball", (0, 0.4, 3.4), W, H)
    full = np.zeros((W * H, 3))
    O2, D2 = reflect_transform(None, D[mirror], P[mirror], N[mirror])
    full[mirror] = trace_reflection_color(ctx, O2, D2)
    approx, n_traced, n_mirror = coherent_reflection(ctx, P, N, D, ids, mirror, W, H, stride=4, var_tol=0.03)
    assert n_mirror > 500                                       # there is a real mirror surface to reconstruct
    assert n_traced < 0.6 * n_mirror                           # traced far fewer rays than per-pixel
    assert np.mean((full[mirror] - approx[mirror]) ** 2) < 5e-3   # close to the exact reflection (smooth content)

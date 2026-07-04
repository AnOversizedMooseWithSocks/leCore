"""Tests for holographic_shadowhome -- the Shadow home (R8: visibility strategies, one home)."""
import numpy as np
from holographic_shadowhome import Shadow, shadow_strategies


def _scene():
    from holographic_sdf import sphere, box
    return sphere(0.4).translate((0, 0.6, 0)).smooth_union(box(3.0, 0.1, 3.0).translate((0, -0.1, 0)), k=0.02)


def test_soft_shadow_darker_under_occluder_and_routes():
    from holographic_raymarch import soft_shadow
    scene = _scene(); up = np.array([0., 1., 0.]); eps = 3e-3
    under = np.array([[0.0, eps, 0.0]]); away = np.array([[1.4, eps, 0.0]])
    s_under = Shadow.soft(scene, under, up); s_away = Shadow.soft(scene, away, up)
    assert s_under[0] < s_away[0]                                 # under the ball is more shadowed
    assert np.array_equal(s_under, soft_shadow(scene, under, up)) # bit-identical routing


def test_ambient_occlusion_in_range_and_routes():
    from holographic_raymarch import ambient_occlusion
    scene = _scene()
    P = np.array([[1.4, 3e-3, 0.0]]); N = np.array([[0.0, 1.0, 0.0]])
    ao = Shadow.ambient_occlusion(scene, P, N)
    assert (0.0 <= ao).all() and (ao <= 1.0).all()
    assert np.array_equal(ao, ambient_occlusion(scene, P, N))


def test_hard_shadow_ray_blocks_and_clears():
    scene = _scene()
    N = np.array([[0.0, 1.0, 0.0]]); Ldir = np.array([[0.0, 1.0, 0.0]]); dist = np.array([5.0])
    under = np.array([[0.0, 3e-3, 0.0]]); away = np.array([[1.4, 3e-3, 0.0]])
    assert Shadow.hard(scene, under, N, Ldir, dist)[0] == 0.0     # blocked by the ball
    assert Shadow.hard(scene, away, N, Ldir, dist)[0] == 1.0      # clear path to the light


def test_strategies_listed():
    assert set(shadow_strategies()) == {"soft", "ambient_occlusion", "hard", "prt(baked)"}


def test_prt_note_is_a_string():
    assert isinstance(Shadow.prt_visibility_note(), str) and "precompute_transfer" in Shadow.prt_visibility_note()

"""Tests for holographic_lightcache -- the cached soft area lights (bake sparse many-sampled NEE + interpolate + recompute)."""
import numpy as np
from holographic_lightcache import cached_soft_lights_shade, split_soft_lights, SOFT_LIGHT_TYPES
from holographic_domecache import _primary_gbuffer
from holographic_sdf import box, sphere
from holographic_render import Camera
from holographic_lights import RectLight, DiskLight, SphereLight, MeshLight, PointLight, SpotLight, IESLight


def _scene_cam():
    scene = sphere(0.5).smooth_union(box(3.0, 0.1, 3.0).translate((0, -0.55, 0)), k=0.02)
    cam = Camera(eye=(0, 0.9, 3.0), target=(0, -0.2, 0), fov_deg=45, aspect=1.0)
    return scene, cam


def _mat(pp):
    n = len(pp); return (np.tile([0.8, 0.8, 0.8], (n, 1)).astype(float), np.zeros(n), np.full(n, 0.7))


def test_split_soft_lights_picks_area_sources():
    soft, hard = split_soft_lights([RectLight(), PointLight(), SpotLight(), DiskLight(),
                                    IESLight(profile=np.ones(8)), SphereLight()])
    assert len(soft) == 3 and len(hard) == 3                          # rect/disk/sphere soft; point/spot/ies hard


def test_area_lights_flagged_soft():
    for L in (RectLight(), DiskLight(), SphereLight()):
        assert getattr(L, "soft", False) is True
    for L in (PointLight(), SpotLight(), IESLight(profile=np.ones(8))):
        assert getattr(L, "soft", False) is False


def test_cached_soft_light_lights_and_high_hit_rate():
    scene, cam = _scene_cam()
    rect = RectLight(position=(0.7, 2.2, 1.0), u_vec=(0.6, 0, 0), v_vec=(0, 0.4, 0.3), intensity=40.0)
    shade, st = cached_soft_lights_shade(scene, cam, 64, 64, [rect], _mat, area_samples=48, return_stats=True)
    assert np.isfinite(shade).all() and shade.max() > 1e-3
    assert st["hit_rate"] >= 0.5


def test_cached_soft_light_is_noise_free():
    # the whole point: the cached soft light barely changes with the bake seed (a per-pixel MC render would), because
    # interpolation carries no seed noise and the sparse bakes are many-sampled
    scene, cam = _scene_cam()
    rect = RectLight(position=(0.7, 2.2, 1.0), u_vec=(0.6, 0, 0), v_vec=(0, 0.4, 0.3), intensity=40.0)
    s0 = cached_soft_lights_shade(scene, cam, 64, 64, [rect], _mat, area_samples=48, seed=0)
    s1 = cached_soft_lights_shade(scene, cam, 64, 64, [rect], _mat, area_samples=48, seed=99)
    assert float(np.abs(s0 - s1).mean()) < 0.01                      # essentially seed-independent -> noise-free


def test_more_bake_samples_cleaner():
    # more samples at the (sparse) anchors -> a cleaner bake -> smaller run-to-run difference
    scene, cam = _scene_cam()
    rect = RectLight(position=(0.7, 2.2, 1.0), u_vec=(0.6, 0, 0), v_vec=(0, 0.4, 0.3), intensity=40.0)
    def sd(ns):
        a = cached_soft_lights_shade(scene, cam, 48, 48, [rect], _mat, area_samples=ns, seed=0)
        b = cached_soft_lights_shade(scene, cam, 48, 48, [rect], _mat, area_samples=ns, seed=1)
        return float(np.abs(a - b).mean())
    assert sd(64) <= sd(8) + 1e-4


def test_empty_soft_lights_is_zero():
    scene, cam = _scene_cam()
    shade = cached_soft_lights_shade(scene, cam, 32, 32, [], _mat, area_samples=16)
    assert np.allclose(shade, 0.0)


def test_cached_indirect_lights_and_noise_free():
    from holographic_lightcache import cached_indirect_shade
    scene, cam = _scene_cam()
    rect = RectLight(position=(0.7, 2.2, 1.0), u_vec=(0.6, 0, 0), v_vec=(0, 0.4, 0.3), intensity=40.0)
    gi0, st = cached_indirect_shade(scene, cam, 64, 64, [rect], _mat, n_dirs=48, stride=8, seed=0, return_stats=True)
    gi1 = cached_indirect_shade(scene, cam, 64, 64, [rect], _mat, n_dirs=48, stride=8, seed=7)
    assert np.isfinite(gi0).all() and gi0.max() > 1e-4
    assert float(np.abs(gi0 - gi1).mean()) < 0.01                    # baked at sparse anchors, many rays -> noise-free
    assert st["hit_rate"] >= 0.5


def test_cached_indirect_colour_bleeding():
    # a red wall next to a white floor: the one-bounce indirect on the floor should be tinted red (more R than B)
    from holographic_lightcache import cached_indirect_shade
    from holographic_sdf import box, sphere
    from holographic_render import Camera
    scene = (box(3.0, 0.1, 3.0).translate((0, -0.6, 0))
             .smooth_union(box(0.1, 1.5, 3.0).translate((-1.0, 0.3, 0)), k=0.02)
             .smooth_union(sphere(0.4).translate((0.3, -0.2, 0)), k=0.02))
    cam = Camera(eye=(1.2, 0.9, 3.0), target=(-0.2, -0.2, 0), fov_deg=48, aspect=1.0)

    def matfn(P):
        n = len(P); alb = np.tile([0.85, 0.85, 0.85], (n, 1)).astype(float)
        alb[P[:, 0] < -0.85] = [0.85, 0.12, 0.12]                    # the left wall is red
        return alb, np.zeros(n), np.full(n, 0.7), np.zeros((n, 3))
    L = [RectLight(position=(0.5, 2.2, 1.0), u_vec=(0.5, 0, 0), v_vec=(0, 0.4, 0.3), intensity=40.0)]
    gi = cached_indirect_shade(scene, cam, 72, 72, L, matfn, n_dirs=48, stride=8, seed=0)
    lit = gi.sum(2) > 0.005
    assert (gi[..., 0] - gi[..., 2])[lit].mean() > 0.01              # red bleed onto the scene

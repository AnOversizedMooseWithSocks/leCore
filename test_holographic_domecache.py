"""Tests for holographic_domecache -- the cached dome / sky-ambient light (warm bake + hot interpolate + cold recompute)."""
import numpy as np
from holographic_domecache import (dome_light_sh, cached_dome_shade, render_dome_term, _primary_gbuffer)
from holographic_prt import precompute_transfer, shade_prt
from holographic_sdf import box, sphere
from holographic_render import Camera
from holographic_lights import DomeLight


def _scene_cam():
    scene = sphere(0.5).smooth_union(box(3.0, 0.1, 3.0).translate((0, -0.55, 0)), k=0.02)
    cam = Camera(eye=(0, 0.9, 3.0), target=(0, -0.2, 0), fov_deg=45, aspect=1.0)
    return scene, cam


def _mat(pp):
    n = len(pp); return (np.tile([0.8, 0.8, 0.8], (n, 1)).astype(float),)


def test_cached_dome_lights_and_high_hit_rate():
    scene, cam = _scene_cam()
    dome = DomeLight(color=(0.4, 0.5, 0.7), ground_color=(0.15, 0.12, 0.1), intensity=1.0)
    shade, st = render_dome_term(scene, cam, 64, 64, dome, _mat, stride=6, return_stats=True)
    assert np.isfinite(shade).all() and shade.max() > 1e-3
    assert st["hit_rate"] >= 0.5                                   # most pixels served by the cache, not recomputed
    assert st["anchors_baked"] > 0 and st["misses_recomputed"] >= 0


def test_dome_is_shadowed_not_flat_fill():
    # ambient occlusion: the sphere's contact region is darker than the open floor (a real dome, not a constant add)
    scene, cam = _scene_cam()
    dome = DomeLight(color=(0.4, 0.5, 0.7), intensity=1.0)
    shade = render_dome_term(scene, cam, 64, 64, dome, _mat, stride=6)
    hit, P, N = _primary_gbuffer(scene, cam, 64, 64)
    lum = shade.mean(2)
    floor = hit & (N[..., 1] > 0.9)
    xs = np.where(floor)[1]; cx = xs.mean()
    cols = np.arange(64)[None, :]
    near = floor & (np.abs(cols - cx) < 8)
    far = floor & (np.abs(cols - cx) > 20)
    assert lum[near].mean() < lum[far].mean()


def test_cache_matches_full_bake():
    scene, cam = _scene_cam()
    dome = DomeLight(color=(0.4, 0.5, 0.7), ground_color=(0.15, 0.12, 0.1), intensity=1.0)
    shade = render_dome_term(scene, cam, 64, 64, dome, _mat, stride=6)
    hit, P, N = _primary_gbuffer(scene, cam, 64, 64)
    light_sh = dome_light_sh(dome)
    full = np.zeros((64, 64, 3))
    T = precompute_transfer(scene, P[hit], N[hit], order=3, n=64)
    full[hit] = shade_prt(T, light_sh, np.full((int(hit.sum()), 3), 0.8))
    err = float(np.abs(shade[hit].mean(1) - full[hit].mean(1)).mean())
    assert err < 0.02                                             # the sparse cache stays close to baking every pixel


def test_no_grid_facets_in_residual():
    # the "blocky shadows" bug: a bilinear grid gather leaves facets on the anchor-grid lines. The smooth Gaussian
    # gather must not -- the residual vs the smooth full bake is not concentrated on the stride-6 grid rows.
    scene, cam = _scene_cam()
    dome = DomeLight(color=(0.4, 0.5, 0.7), intensity=1.0)
    shade = render_dome_term(scene, cam, 64, 64, dome, _mat, stride=6)
    hit, P, N = _primary_gbuffer(scene, cam, 64, 64)
    light_sh = dome_light_sh(dome)
    full = np.zeros((64, 64, 3))
    T = precompute_transfer(scene, P[hit], N[hit], order=3, n=64)
    full[hit] = shade_prt(T, light_sh, np.full((int(hit.sum()), 3), 0.8))
    res = np.abs(shade.mean(2) - full.mean(2))
    on_grid = res[::6].mean(); off_grid = res[np.mod(np.arange(64), 6) != 0].mean()
    assert on_grid < off_grid * 2.0                              # grid rows not a facet spike


def test_empty_gbuffer_is_safe():
    # a camera looking at nothing -> no visible surface -> zero shade, no crash
    scene = sphere(0.3).translate((0, 0, -50))
    cam = Camera(eye=(0, 0, 3.0), target=(0, 0, 4.0), fov_deg=30, aspect=1.0)
    dome = DomeLight(intensity=1.0)
    shade, st = render_dome_term(scene, cam, 32, 32, dome, _mat, return_stats=True)
    assert np.all(shade == 0.0) and st["anchors_baked"] == 0

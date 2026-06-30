"""Tests for GI (irradiance cache) and caustics (forward light splat) (LIGHT-2)."""

import numpy as np
from holographic_sdf import sphere, plane
from holographic_globalillum import gather_indirect, irradiance_cache, read_cache, caustics


def _floor_points(n=10):
    P = np.array([[x, -0.85, z] for x in np.linspace(-1, 1, n) for z in np.linspace(-1, 1, n)])
    N = np.broadcast_to(np.array([0., 1, 0]), P.shape).copy()
    return P, N


def test_irradiance_cache_reconstructs_dense_gi():
    scene = sphere(0.7).union(plane(-0.85))
    P, N = _floor_points(10)
    dense = gather_indirect(scene, P, N, (-0.4, 0.7, -0.3), n_dirs=12, seed=1)
    cache = irradiance_cache(scene, P, N, (-0.4, 0.7, -0.3), n_cache=20, n_dirs=12, seed=1)
    approx = read_cache(cache, P)
    assert np.abs(approx - dense).mean() < 0.15              # sparse cache ~ dense GI (indirect light is smooth)
    # the cache really is sparse
    assert len(cache[0]) <= 20 < len(P)


def test_gi_gathers_positive_indirect_from_a_lit_surface():
    # a point whose hemisphere faces a large lit floor catches positive one-bounce indirect light
    scene = plane(-0.85)
    P = np.array([[0.0, 0.3, 0.0]]); N = np.array([[0.0, -1.0, 0.0]])   # facing DOWN at the lit floor
    indirect = gather_indirect(scene, P, N, (-0.2, 0.9, -0.1), n_dirs=32, seed=2)
    assert indirect.sum() > 0.0                              # the lit floor bounces light up to the point
    assert (indirect >= 0).all()                             # irradiance is non-negative everywhere


def test_caustics_focus_concentrates_light():
    scene = sphere(0.7).union(plane(-1.2))
    c = caustics(scene, ior=1.5, n_side=160, res=128, receiver_y=-1.2)
    assert c.max() > 5.0                                     # refraction focuses: a peak well above uniform
    assert c.mean() > 0                                      # normalised to mean ~1


def test_caustics_without_refraction_is_flatter():
    # ior=1.0 -> no bending -> rays pass straight -> no focusing, so a much lower peak than ior=1.5
    scene = sphere(0.7).union(plane(-1.2))
    flat = caustics(scene, ior=1.0, n_side=160, res=128, receiver_y=-1.2)
    focused = caustics(scene, ior=1.5, n_side=160, res=128, receiver_y=-1.2)
    assert focused.max() > flat.max()

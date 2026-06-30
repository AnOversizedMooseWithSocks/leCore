"""Tests for the field-native SDF lighting: sphere trace, AO, soft shadow, sky dome, refraction, SSS (LIGHT-1)."""

import numpy as np
from holographic_sdf import sphere, plane, torus
from holographic_render import Camera
from holographic_raymarch import (sphere_trace, sdf_normal, ambient_occlusion, soft_shadow,
                                  sky_dome, refract_dir, subsurface, render_sdf)


def test_sphere_trace_hits_a_sphere_and_normal_points_out():
    s = sphere(1.0)
    O = np.array([[0.0, 0.0, 3.0]]); D = np.array([[0.0, 0.0, -1.0]])
    hit, t, P = sphere_trace(s, O, D)
    assert hit[0] and abs(t[0] - 2.0) < 0.05                 # surface at z=+1, ray from z=3
    N = sdf_normal(s, P)
    assert np.allclose(N[0], [0, 0, 1], atol=0.05)          # outward normal at the front pole


def test_ambient_occlusion_darkens_a_crease():
    scene = sphere(0.7).union(plane(-0.7))
    open_floor = ambient_occlusion(scene, np.array([[3.0, -0.7, 0.0]]), np.array([[0., 1, 0]]))[0]
    P = np.array([[0.0, -0.68, 0.7]])
    crease = ambient_occlusion(scene, P, sdf_normal(scene, P))[0]
    assert crease < open_floor and open_floor > 0.9


def test_soft_shadow_blocks_under_an_occluder():
    scene = sphere(0.7).union(plane(-0.8))
    lit = soft_shadow(scene, np.array([[3.0, -0.79, 0.0]]), np.array([0., 1, 0]))[0]
    shadowed = soft_shadow(scene, np.array([[0.0, -0.79, 0.0]]), np.array([0., 1, 0]))[0]
    assert shadowed < 0.3 and lit > 0.9


def test_sky_dome_sun_is_brighter_than_away():
    sun = (-0.4, 0.7, -0.3)
    toward = sky_dome(np.array([sun]) / np.linalg.norm(sun), sun_dir=sun)[0]
    away = sky_dome(np.array([[0.4, -0.7, 0.3]]), sun_dir=sun)[0]
    assert toward.sum() > away.sum()                         # the sun direction is the brightest


def test_sky_dome_samples_a_provided_hdri():
    env = np.zeros((4, 8, 3)); env[0, 0] = [9.0, 0, 0]       # a bright red patch at one lon/lat
    out = sky_dome(np.array([[0., 1, 0], [0, -1, 0]]), env=env)
    assert out.shape == (2, 3)                               # equirectangular sampling returns per-direction colour


def test_refraction_bends_and_total_internal_reflects():
    # straight-on ray through a flat interface is (nearly) undeviated
    D = np.array([[0.0, 0.0, -1.0]]); N = np.array([[0.0, 0.0, 1.0]])
    assert np.allclose(refract_dir(D, N, 1.5)[0], [0, 0, -1], atol=1e-6)
    # a grazing ray exiting a dense medium total-internal-reflects (returns a reflected dir, not NaN)
    Dg = np.array([[0.9, 0.0, -0.436]]); Dg = Dg / np.linalg.norm(Dg)
    r = refract_dir(Dg, np.array([[0.0, 0.0, -1.0]]), 1.5)
    assert np.all(np.isfinite(r))


def test_subsurface_thin_transmits_more_than_thick():
    thin = torus(0.7, 0.12); thick = sphere(0.7)
    Pt = np.array([[0.0, 0.0, 0.82]]); Nt = sdf_normal(thin, Pt)
    Ps = np.array([[0.0, 0.0, 0.7]]); Ns = sdf_normal(thick, Ps)
    L = np.array([0.0, 0.0, -1.0])
    assert subsurface(thin, Pt, Nt, L)[0] > subsurface(thick, Ps, Ns, L)[0]


def test_render_sdf_produces_a_valid_image():
    scene = sphere(0.8).union(plane(-0.8))
    cam = Camera(eye=(1.6, 1.0, 2.4), target=(0, 0, 0), fov_deg=45)
    img = render_sdf(scene, cam, 48, 48, reflect=0.2, refract=0.3, sss=0.2)
    assert img.shape == (48, 48, 3) and 0.0 <= img.min() and img.max() <= 1.0

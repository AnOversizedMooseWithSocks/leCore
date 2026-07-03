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


def test_active_only_trace_matches_all_rays():
    """The active-only sphere_trace is bit-identical to evaluating every ray every step (a speed-only change)."""
    import numpy as np
    from holographic_raymarch import sphere_trace
    from holographic_render import Camera
    sphere = lambda P: np.linalg.norm(P - np.array([0, 0, 0]), axis=1) - 1.0
    un = type("U", (), {"eval": staticmethod(sphere)})()
    cam = Camera(eye=(0, 0, 4), target=(0, 0, 0), fov_deg=45)
    e, d = cam.ray_dirs(64, 64); O = np.broadcast_to(e, (64 * 64, 3)).astype(float); D = d.reshape(-1, 3)

    def old(sdf, O, D, max_steps=96, max_dist=20.0, surf_eps=1e-3):
        O = np.asarray(O, float); D = np.asarray(D, float); M = len(D)
        t = np.zeros(M); hit = np.zeros(M, bool); active = np.ones(M, bool)
        for _ in range(max_steps):
            P = O + t[:, None] * D; dd = sdf.eval(P); nh = active & (dd < surf_eps); hit |= nh
            active &= ~nh; active &= (t < max_dist)
            if not active.any():
                break
            t = t + np.where(active, np.clip(dd, 0.0, None), 0.0)
        return hit, t, O + t[:, None] * D
    h0, t0, _ = old(un, O, D); h1, t1, _ = sphere_trace(un, O, D)
    assert np.array_equal(h0, h1) and np.abs(t0 - t1).max() == 0.0


def test_over_relaxation_opt_in_is_safe_and_default_exact():
    """relax=1.0 is the exact marcher; relax>1 (opt-in) keeps hits nearly identical (bounded disagreement)."""
    import numpy as np
    from holographic_raymarch import sphere_trace
    from holographic_render import Camera
    rng = np.random.default_rng(0)
    class U:
        def __init__(s, n): s.cs = rng.uniform(-2, 2, (n, 3))
        def eval(s, P): return np.min(np.stack([np.linalg.norm(P - c, axis=1) - 0.5 for c in s.cs]), axis=0)
        def ids(s, P): return np.argmin(np.stack([np.linalg.norm(P - c, axis=1) for c in s.cs]), axis=0)
    un = U(10)
    cam = Camera(eye=(0, 0, 7), target=(0, 0, 0), fov_deg=50)
    e, d = cam.ray_dirs(96, 96); O = np.broadcast_to(e, (96 * 96, 3)).astype(float); D = d.reshape(-1, 3)
    h0, t0, _ = sphere_trace(un, O, D, relax=1.0)
    h1, t1, _ = sphere_trace(un, O, D, relax=1.4)
    # over-relaxation never misses geometry catastrophically here (open scene): near-identical hits
    assert (h0 == h1).mean() > 0.99
    both = h0 & h1
    assert float(np.abs(t0[both] - t1[both]).max()) < 0.05        # depths agree closely where both hit

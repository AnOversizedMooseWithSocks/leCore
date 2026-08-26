"""Hair groom (H1/H2/H3/H7): roots on a surface, PBD strand dynamics, guide interpolation, curl-noise wind."""
import numpy as np
from holographic.mesh_and_geometry.holographic_groom import Strand, groom, build_strand_body, simulate_strands, interpolate_strands, CurlWind, _normalize
from holographic.mesh_and_geometry.holographic_sdf import sphere

S = sphere(1.0)
BOUNDS = ([-1.6, -1.6, -1.6], [1.6, 1.6, 1.6])


def test_groom_roots_on_surface_along_normal():
    strands = groom(S.eval, 40, BOUNDS, length=0.8, n_pts=8, seed=0)
    roots = np.array([s.root for s in strands])
    assert np.abs(np.linalg.norm(roots, axis=1) - 1.0).max() < 0.05
    st = strands[0]
    base = _normalize(st.points[1] - st.points[0])
    assert np.dot(base, st.root_normal) > 0.9 and abs(st.length() - 0.8) < 0.05


def test_curl_makes_nonstraight():
    curly = groom(S.eval, 5, BOUNDS, length=0.8, n_pts=12, curl=2.0, seed=0)
    straightness = np.linalg.norm(curly[0].points[-1] - curly[0].points[0]) / curly[0].length()
    assert straightness < 0.95


def test_pbd_strand_falls_without_stretching():
    one = groom(S.eval, 1, BOUNDS, length=0.8, n_pts=10, seed=3)
    tip0 = one[0].points[-1].copy(); L0 = one[0].length()
    moved = simulate_strands(one, steps=80, gravity=(0.0, -9.8, 0.0), body_sdf=S.eval)
    assert moved[0].points[-1][1] < tip0[1] - 0.05
    assert abs(moved[0].length() - L0) < 0.05 * L0
    assert np.allclose(moved[0].root, one[0].root)


def test_guide_interpolation_and_clumping():
    guides = groom(S.eval, 60, BOUNDS, length=0.8, n_pts=8, curl=1.0, seed=1)
    render_roots = np.array([g.root for g in guides]) * 1.001
    loose = interpolate_strands(guides, render_roots, k=3, clump=0.0)
    tight = interpolate_strands(guides, render_roots, k=3, clump=1.0)
    assert len(loose) == len(render_roots)
    gt = np.array([g.points[-1] for g in guides])
    spread_loose = np.linalg.norm(np.array([s.points[-1] for s in loose]) - gt, axis=1).mean()
    spread_tight = np.linalg.norm(np.array([s.points[-1] for s in tight]) - gt, axis=1).mean()
    assert spread_tight <= spread_loose + 1e-9


def test_curl_noise_wind_moves_without_ballooning():
    wind = CurlWind(strength=3.0, seed=2)
    calm = groom(S.eval, 1, BOUNDS, length=0.8, n_pts=10, seed=5)
    blown = simulate_strands([calm[0]], steps=60, gravity=(0.0, -1.0, 0.0), wind=wind.force)
    assert np.linalg.norm(blown[0].points[-1] - calm[0].points[-1]) > 0.02
    assert abs(blown[0].length() - calm[0].length()) < 0.06 * calm[0].length()


def test_deterministic():
    a = groom(S.eval, 10, BOUNDS, seed=7); b = groom(S.eval, 10, BOUNDS, seed=7)
    assert np.array_equal(a[0].points, b[0].points)

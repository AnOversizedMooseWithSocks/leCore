"""Tests for holographic_raydiff: a perpendicular ray frame transported through a bounce reconstructs the bundle."""
import numpy as np
from holographic.rendering.holographic_raydiff import transport_pencil, find_focus, pencil_radius_at, reflect_off_sphere, perpendicular_basis, lobe_sigma, dispersion_spread


def test_frame_predicts_dense_bundle_focus():
    C = np.array([0, 0, 0.0]); R = 2.0; D = np.array([0, 0, -1.0]); eps = 0.03
    O = np.array([0.0, 0, 1.9])                                # inside -> concave far wall -> focus
    P, D2, hit = transport_pencil(O, D, C, R, eps)
    s_focus, _ = find_focus(P, D2, s_max=4.0)
    assert abs(s_focus - R / 2) < 0.15 * (R / 2)               # matches analytic f = R/2
    u, v = perpendicular_basis(D); ang = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    off = eps * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v)
    Pb, Nb, D2b, hb = reflect_off_sphere(O + off, np.broadcast_to(D, (100, 3)), C, R)
    ss = np.linspace(1e-3, 4.0, 400)
    s_bundle = ss[int(np.argmin([np.sqrt(((Pb + s * D2b)[:, :2].var(0)).sum()) for s in ss]))]
    assert abs(s_focus - s_bundle) < 0.1                       # 5-ray frame agrees with the 100-ray bundle


def test_pencil_collapses_at_focus_caustic():
    C = np.array([0, 0, 0.0]); R = 2.0
    P, D2, hit = transport_pencil(np.array([0.0, 0, 1.9]), np.array([0, 0, -1.0]), C, R, 0.03)
    r_near = pencil_radius_at(P, D2, 0.05); _, r_focus = find_focus(P, D2, 4.0)
    assert r_focus < 0.1 * r_near                              # area -> 0 at the focus (the caustic)


def test_lobe_combines_geometry_roughness_light():
    C = np.array([0, 0, 0.0]); R = 2.0
    P, D2, hit = transport_pencil(np.array([0.0, 0, 1.9]), np.array([0, 0, -1.0]), C, R, 0.03)
    base = lobe_sigma(P, D2, 0.6)
    rough = lobe_sigma(P, D2, 0.6, roughness=0.05)
    both = lobe_sigma(P, D2, 0.6, roughness=0.05, light_half_angle=0.03)
    assert rough > base and both > rough                       # roughness and soft light widen the one lobe


def test_dispersion_fans_by_wavelength():
    D = np.array([0.7, 0, -0.7]); N = np.array([0, 0, 1.0])
    spread = dispersion_spread(D, N, [1 / 1.513, 1 / 1.532])   # red vs blue eta into crown glass
    assert spread > 1e-3                                       # a real chromatic fan

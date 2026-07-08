"""Tests for holographic_thinfilm -- thin-film interference iridescence (soap bubble / oil slick)."""
import numpy as np
from holographic.rendering.holographic_thinfilm import interference_reflectance, spectrum_to_rgb, thin_film_tint, iridescent_socket


def test_reflectance_spectrum_in_range():
    spec = interference_reflectance(320.0, 1.0)
    assert spec.min() >= 0.0 and spec.max() <= 1.0               # reflectance is a valid [0,1] spectrum


def test_tint_cycles_with_thickness():
    # sweeping the film thickness marches the tint through the spectrum (the defining property)
    tints = np.array([thin_film_tint(t, 1.0) for t in np.linspace(100, 800, 40)])
    assert tints.min() >= 0.0 and tints.max() <= 1.0
    assert tints.std(axis=0).mean() > 0.05                       # colour genuinely varies with thickness


def test_tint_shifts_with_view_angle():
    # the same film reads a different colour head-on vs grazing -> iridescence is view-dependent
    head_on = thin_film_tint(320.0, 1.0)
    grazing = thin_film_tint(320.0, 0.2)
    assert np.linalg.norm(head_on - grazing) > 0.05


def test_per_point_thickness_and_angle():
    # the core is vectorised over per-point thickness AND per-point angle together
    d = np.array([200.0, 350.0, 500.0]); cv = np.array([1.0, 0.6, 0.3])
    spec = interference_reflectance(d, cv)
    assert spec.shape == (3, 60)
    rgb = spectrum_to_rgb(spec)
    assert rgb.shape == (3, 3)
    assert rgb.std(axis=0).mean() > 0.02                         # three different films -> three different colours


def test_thicker_film_denser_fringes():
    # a thicker film packs more interference fringes across the visible band -> the spectrum oscillates more
    thin = interference_reflectance(150.0, 1.0)
    thick = interference_reflectance(900.0, 1.0)
    # count sign changes of the derivative (oscillations)
    def wiggles(s):
        d = np.diff(s); return int(np.sum(np.diff(np.sign(d)) != 0))
    assert wiggles(thick) > wiggles(thin)


def test_socket_is_view_dependent():
    base = np.array([0.2, 0.2, 0.25])
    sock = iridescent_socket(base, thickness_nm=300.0, strength=1.0)
    P = np.zeros((5, 3)); N = np.tile([0.0, 0.0, 1.0], (5, 1))
    V = np.array([[0, 0, 1.0], [0.3, 0, 0.95], [0.6, 0, 0.8], [0.8, 0, 0.6], [0.95, 0, 0.31]])
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    cols = sock(P, N, V)
    assert cols.shape == (5, 3) and cols.std(axis=0).mean() > 0.02   # different angles -> different colours


def test_deterministic():
    assert np.array_equal(thin_film_tint(275.0, 0.7), thin_film_tint(275.0, 0.7))


def test_phase_flip_changes_colour():
    # including / omitting the half-wave flip changes the colour (soap-on-air vs a non-flipping stack)
    a = thin_film_tint(300.0, 1.0, phase_flip=True)
    b = thin_film_tint(300.0, 1.0, phase_flip=False)
    assert np.linalg.norm(a - b) > 0.02

"""Tests for the vectorized deformers and blendshapes (ANIM-1)."""

import numpy as np
from holographic_deform import taper, twist, bend, lattice_deform, blendshapes


def _bar(n=9):
    return np.array([[x, 0.0, 0.0] for x in np.linspace(-1, 1, n)])


def test_taper_scales_cross_section_along_axis():
    P = np.array([[1.0, 1.0, z] for z in np.linspace(0, 1, 5)])
    out = taper(P, 1.0, axis=2)                               # widen toward high z
    # the cross-section (x,y) grows with z
    assert out[-1, 0] > out[0, 0] and out[-1, 1] > out[0, 1]
    assert np.allclose(out[:, 2], P[:, 2])                    # the axis coordinate is untouched


def test_twist_rotates_cross_section_and_is_invertible():
    P = _bar()
    P = np.column_stack([P[:, 0], np.ones(len(P)), np.zeros(len(P))])   # offset in y so twist is visible
    out = twist(P, np.pi, axis=0)
    assert not np.allclose(out, P)
    back = twist(out, -np.pi, axis=0)                         # twisting back recovers the original
    assert np.allclose(back, P, atol=1e-9)


def test_bend_curves_a_straight_bar_into_an_arc():
    bar = _bar()
    out = bend(bar, np.pi / 2, axis=0, up=2)
    assert out[0, 2] > 0.1 and out[-1, 2] > 0.1              # both ends rise out of the plane
    assert abs(out[0, 2] - out[-1, 2]) < 1e-9               # symmetric
    assert (out[:, 0].max() - out[:, 0].min()) < 2.0         # the arc is shorter in x than the straight bar


def test_lattice_deform_identity_and_translation():
    P = np.random.default_rng(0).uniform(-1, 1, (40, 3))
    bounds = (np.array([-1., -1, -1]), np.array([1., 1, 1]))
    off = np.zeros((3, 3, 3, 3))
    assert np.allclose(lattice_deform(P, bounds, off), P)     # zero offsets -> identity
    off[:] = np.array([0.2, -0.1, 0.0])
    assert np.allclose(lattice_deform(P, bounds, off) - P, np.array([0.2, -0.1, 0.0]), atol=1e-6)


def test_blendshapes_is_a_weighted_bundle():
    base = np.random.default_rng(1).uniform(-1, 1, (30, 3))
    t1 = base + np.array([1.0, 0, 0]); t2 = base + np.array([0, 1.0, 0])
    assert np.allclose(blendshapes(base, [t1, t2], [0.0, 0.0]), base)
    assert np.allclose(blendshapes(base, [t1, t2], [1.0, 0.0]), t1)
    # superposition: half of each target = base + 0.5*dx + 0.5*dy
    mix = blendshapes(base, [t1, t2], [0.5, 0.5])
    assert np.allclose(mix, base + np.array([0.5, 0.5, 0.0]))

"""Spectral upsampling: RGB -> a physical reflectance, with no table and no autodiff.

Contracts only. The headline is a machine-precision round trip over the sRGB gamut; the stated
bounds (the asymptotic corners, the bake-once cost) are pinned so they cannot rot into folklore.
"""
import numpy as np
import pytest

import lecore
from holographic.rendering.holographic_observer import human_cie
from holographic.rendering.holographic_spectralup import (
    _basis, fit_palette, fit_rgb, reflectance_to_rgb, rgb_to_spectrum, sigmoid, sigmoid_spectrum)


@pytest.fixture(scope="module")
def obs():
    return human_cie(90)


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=64, seed=0)


def test_flat_reflectance_is_exactly_white(obs):
    """The illuminant kept negative, pinned. Reflectance becomes colour only under a light; without the
    von Kries row scaling a flat unit reflectance came back [1.198, 0.950, 0.907] and white had a 0.198
    error no solver could remove, because the target was unreachable."""
    A, lam, _, _ = _basis(obs)
    assert np.allclose(A @ np.ones_like(lam), 1.0, atol=1e-12)


@pytest.mark.parametrize("c", [(50.0, -50.0, 10.0), (-80.0, 40.0, -3.0), (0.0, 0.0, 0.0),
                               (500.0, 500.0, 500.0)])
def test_reflectance_is_bounded_by_construction(c, obs):
    """The sigmoid is why a fitted reflectance cannot invent energy. No clamp is applied anywhere; if
    this fails the function space has been changed and energy conservation went with it."""
    s = sigmoid_spectrum(np.array(c), obs["wavelengths_nm"])
    assert s.min() >= 0.0 and s.max() <= 1.0


def test_sigmoid_matches_the_published_algebraic_form():
    """S(x) = 1/2 + x/(2*sqrt(1+x^2)) -- algebraic, not logistic. S(0) is exactly 1/2 and the tails
    approach 0 and 1 without reaching them, which is what makes the corners asymptotic."""
    assert sigmoid(0.0) == 0.5
    assert 0.0 < float(sigmoid(-1e6)) < 1e-5 and 1.0 - 1e-5 < float(sigmoid(1e6)) < 1.0


@pytest.mark.parametrize("rgb", [(0.8, 0.2, 0.2), (0.2, 0.7, 0.3), (0.5, 0.5, 0.5), (0.1, 0.2, 0.9),
                                 (0.717, 0.981, 0.575), (0.0, 0.483, 0.608), (0.02, 0.02, 0.02),
                                 (0.999, 0.999, 0.999)])
def test_round_trip_is_machine_precision(rgb, obs):
    """THE HEADLINE. Jakob & Hanika claim zero error on the full sRGB gamut; this reproduces it with a
    solve instead of their 9 MiB table. The saturated entries here are the ones that broke pure
    Gauss-Newton -- a 1e-6 tolerance would pass the broken solver, so the gate is 1e-9."""
    assert np.abs(reflectance_to_rgb(rgb_to_spectrum(rgb, obs), obs) - np.array(rgb)).max() < 1e-9


@pytest.mark.parametrize("corner", [(1.0, 1.0, 1.0), (0.0, 0.0, 0.0)])
def test_the_corners_are_asymptotic_and_bounded(corner, obs):
    """The STATED BOUND, not a failure. Exactly white needs f == 1 at every wavelength, which S reaches
    only as x -> infinity. Pinned so nobody 'fixes' it by loosening the contract above."""
    err = np.abs(reflectance_to_rgb(rgb_to_spectrum(corner, obs), obs) - np.array(corner)).max()
    assert err < 2e-6


def test_the_gamut_sweep_holds_not_just_the_probes(obs):
    """A parametrized handful can be lucky. This is the population claim the module docstring makes."""
    rng = np.random.default_rng(0)
    errs = [np.abs(reflectance_to_rgb(rgb_to_spectrum(g, obs), obs) - g).max()
            for g in rng.random((60, 3))]
    assert max(errs) < 1e-9, max(errs)


def test_fit_is_deterministic(obs):
    """Same input, same coefficients, bit for bit -- multi-start with any randomness would break it."""
    assert np.array_equal(fit_rgb((0.3, 0.6, 0.9), obs), fit_rgb((0.3, 0.6, 0.9), obs))


def test_palette_deduplicates_and_agrees_with_the_single_path(obs):
    """De-duplication is the whole point of the palette path: the fit is the expensive part and real
    textures repeat colours heavily."""
    cols = np.array([[0.8, 0.2, 0.2], [0.5, 0.5, 0.5], [0.8, 0.2, 0.2]])
    pal = fit_palette(cols, obs)
    assert np.array_equal(pal[0], pal[2])
    assert np.allclose(pal[0], fit_rgb(cols[0], obs), atol=0.0)


def test_emission_path_would_have_returned_black(mind, obs):
    """WHY reflectance_to_rgb exists as its own function. spectrum_to_rgb is built for EMISSION -- its
    'none' mode divides XYZ by a constant calibrated for blackbody radiance -- so handing it a
    reflectance in [0,1] clips to black with nothing to say why. This pins the trap so the two doors
    are never quietly merged."""
    sp = mind.rgb_to_spectrum((0.8, 0.2, 0.2))
    # mode='none' fails LOUDLY -- black.
    assert np.allclose(np.asarray(mind.spectrum_to_rgb(sp, mode="none"), float), 0.0, atol=1e-6)
    # The DEFAULT mode='hue' fails QUIETLY, which is the more dangerous half: it discards luminance
    # and renormalises, so it returns a plausible-looking colour that is simply not the input.
    quiet = np.asarray(mind.spectrum_to_rgb(sp), float)
    assert np.abs(quiet - np.array([0.8, 0.2, 0.2])).max() > 0.1, quiet
    assert quiet.max() > 0.5, "the quiet failure looks like a real colour; that is the point"
    # And the door that actually closes.
    assert np.abs(np.asarray(mind.reflectance_to_rgb(sp), float)
                  - np.array([0.8, 0.2, 0.2])).max() < 1e-9


def test_the_cheap_path_matches_the_full_one(mind):
    """Coefficients -> reflectance must agree with the fitted curve, or 'bake once' is a lie."""
    lam = np.asarray(human_cie(90)["wavelengths_nm"], float)
    c = mind.rgb_to_spectrum_coeffs((0.3, 0.6, 0.9))
    assert np.allclose(np.asarray(mind.spectrum_from_coeffs(c, lam), float),
                       np.asarray(mind.rgb_to_spectrum((0.3, 0.6, 0.9)), float), atol=1e-12)

"""Blackbody (T3): colour marches red->white->blue with temperature; Wien peak correct; deterministic."""
import numpy as np
from holographic_blackbody import blackbody_rgb, peak_wavelength_nm, planck_radiance


def test_colour_marches_red_to_blue_with_temperature():
    cool = blackbody_rgb(1000.0); day = blackbody_rgb(6500.0); hot = blackbody_rgb(12000.0)
    assert cool[0] > cool[1] > cool[2]                              # ember is red-dominant
    assert (day.max() - day.min()) < 0.35                           # daylight ~neutral
    assert hot[2] > cool[2]                                         # blue rises with temperature
    assert (cool[0] / (cool[2] + 1e-6)) > (hot[0] / (hot[2] + 1e-6))   # red/blue ratio falls as it heats


def test_wien_peak_and_planck_shape():
    assert 480 < peak_wavelength_nm(5772) < 520                     # the Sun peaks ~green
    assert peak_wavelength_nm(1000) > 780                           # an ember peaks in the IR
    lam = np.linspace(380e-9, 780e-9, 50)
    hot = planck_radiance(lam, 6000.0); cool = planck_radiance(lam, 3000.0)
    assert (hot > cool).all()                                       # hotter emits more at every visible wavelength


def test_deterministic_and_in_range():
    for t in (900.0, 2800.0, 6500.0):
        c = blackbody_rgb(t)
        assert c.shape == (3,) and c.min() >= 0 and c.max() <= 1
        assert np.array_equal(blackbody_rgb(t), blackbody_rgb(t))

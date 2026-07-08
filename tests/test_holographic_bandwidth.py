"""Tests for holographic_bandwidth: the genuinely-new parts of the fractal/bandwidth probe (dimension itself is
already shipped via box-counting / R-S). Spectral bandwidth drives the encoder knob; the slope-vs-increment
cross-check flags singularities the shipped single-estimator dimension cannot catch."""

import numpy as np

from holographic.misc.holographic_bandwidth import spectral_bandwidth, spectral_dimension, _increment_dimension, fractal_confidence


def _fbm(n, H, seed=1):
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(n)
    amp = np.zeros(len(f))
    amp[1:] = f[1:] ** (-(2 * H + 1) / 2)
    return np.fft.irfft(amp * np.exp(1j * rng.uniform(0, 2 * np.pi, len(f))), n)


def test_bandwidth_separates_smooth_and_broadband():
    n = 8192
    smooth = np.sin(2 * np.pi * 3 * np.arange(n) / n)
    white = np.random.default_rng(2).standard_normal(n)
    assert spectral_bandwidth(smooth) < 0.05 and spectral_bandwidth(white) > 0.5


def test_bandwidth_is_a_fraction_of_nyquist():
    bw = spectral_bandwidth(_fbm(8192, 0.5))
    assert 0.0 <= bw <= 1.0


def test_rougher_fbm_occupies_more_bandwidth():
    assert spectral_bandwidth(_fbm(8192, 0.2)) > spectral_bandwidth(_fbm(8192, 0.9))


def test_spectral_dimension_recovers_fbm():
    for H in (0.3, 0.5, 0.8):
        assert abs(spectral_dimension(_fbm(8192, H)) - (2 - H)) < 0.1


def test_increment_dimension_recovers_fbm():
    for H in (0.3, 0.5, 0.8):
        assert abs(_increment_dimension(_fbm(8192, H)) - (2 - H)) < 0.1


def test_cross_check_agrees_on_clean_fbm():
    for H in (0.3, 0.5, 0.8):
        d_spec, d_inc, agree = fractal_confidence(_fbm(8192, H))
        assert agree and min(d_spec, d_inc) - 0.15 < (2 - H) < max(d_spec, d_inc) + 0.15


def test_cross_check_flags_a_step_singularity():
    n = 8192
    step = np.zeros(n); step[n // 2:] = 1.0
    assert not fractal_confidence(step)[2]


def test_cross_check_flags_a_pure_tone():
    n = 8192
    tone = np.sin(2 * np.pi * 3 * np.arange(n) / n)
    assert not fractal_confidence(tone)[2]


def test_deterministic():
    x = _fbm(8192, 0.5)
    assert spectral_bandwidth(x) == spectral_bandwidth(x)
    assert fractal_confidence(x) == fractal_confidence(x)

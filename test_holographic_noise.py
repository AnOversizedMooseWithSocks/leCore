"""Tests for G1 holographic procedural noise (holographic_noise): a single band is one hypervector (an FPE
bundle of random-weighted RBF kernels), band-limited by the kernel; fBm is a weighted superposition of
per-octave band fields -- the octave bundle. Roughness tracks persistence; deep fBm is FFT-bound."""

import numpy as np

from holographic_fpe import VectorFunctionEncoder
from holographic_noise import noise_field, sample, FractalNoise, _selftest


def _enc():
    return VectorFunctionEncoder(2, dim=512, bounds=[(0, 8), (0, 8)], kernel="rbf", bandwidth=3.0, seed=1)


def test_single_band_deterministic():
    enc = _enc()
    assert np.allclose(noise_field(enc, seed=5), noise_field(enc, seed=5))


def test_single_band_is_band_limited():
    enc = _enc()
    f = noise_field(enc, seed=5)
    line = np.array([sample(enc, f, [x, 4.0]) for x in np.linspace(0.5, 7.5, 120)])
    line -= line.mean()
    ac1 = np.corrcoef(line[:-1], line[1:])[0, 1]
    assert ac1 > 0.9                      # smooth: strong lag-1 autocorrelation


def test_fbm_roughness_tracks_persistence():
    def rough(gain):
        fb = FractalNoise(2, dim=512, bounds=[(0, 8), (0, 8)], octaves=4, lacunarity=2.0,
                          gain=gain, base_bandwidth=3.0, seed=3)
        prof = np.array([fb.query([x, 4.0]) for x in np.linspace(0.3, 7.7, 200)])
        return np.std(np.diff(prof)) / (np.std(prof) + 1e-9)
    assert rough(0.90) > rough(0.25)


def test_fbm_is_the_octave_bundle():
    fb = FractalNoise(2, dim=512, bounds=[(0, 8), (0, 8)], octaves=3, gain=0.5, base_bandwidth=2.0, seed=4)
    p = [3.3, 5.1]
    manual = sum(a * e.query(f, p) for a, e, f in zip(fb.amplitudes, fb.encoders, fb.fields)) / fb._norm
    assert abs(manual - fb.query(p)) < 1e-12


def test_parallel_fbm_query_many_matches_serial():
    fb = FractalNoise(2, dim=512, bounds=[(0, 8), (0, 8)], octaves=4, gain=0.5, base_bandwidth=2.0, seed=8)
    rng = np.random.default_rng(8)
    pts = rng.uniform(0, 8, (128, 2))
    serial = fb.query_many(pts, workers=1)
    parallel = fb.query_many(pts, workers=4)
    assert np.allclose(parallel, serial, atol=1e-12)


def test_selftest_runs():
    _selftest()

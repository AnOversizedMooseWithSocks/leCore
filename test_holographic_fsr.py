"""Inverse-rendering IR12: FSR1-style upscaler -- EASU (edge-adaptive Lanczos) + RCAS (shipped sharpen)."""
import numpy as np
from holographic_postfx import resample
from holographic_fsr import (lanczos_upscale, easu_upscale, fsr_upscale, _box_downscale, _psnr, _edge_energy)


def _native():
    yy, xx = np.mgrid[0:96, 0:96].astype(float)
    n = 0.5 + 0.4 * np.sign(np.sin((xx + yy) / 5.0)); n[20:50, 20:50] = 0.9; n[60:85, 55:88] = 0.15
    return np.clip(np.stack([n, n, n], axis=-1), 0, 1)


def test_easu_beats_bilinear_psnr():
    native = _native(); low = _box_downscale(native, 2); hw = native.shape[:2]
    bil = np.clip(resample(low, 2.0), 0, 1)[:hw[0], :hw[1]]
    easu = easu_upscale(low, 2.0)[:hw[0], :hw[1]]
    assert _psnr(easu, native) > _psnr(bil, native)


def test_easu_sharper_than_bilinear():
    native = _native(); low = _box_downscale(native, 2); hw = native.shape[:2]
    bil = np.clip(resample(low, 2.0), 0, 1)[:hw[0], :hw[1]]
    easu = easu_upscale(low, 2.0)[:hw[0], :hw[1]]
    assert _edge_energy(easu) > _edge_energy(bil)


def test_anti_ringing_in_range():
    low = _box_downscale(_native(), 2)
    easu = easu_upscale(low, 2.0)
    assert easu.min() >= -0.02 and easu.max() <= 1.02          # no wild Lanczos overshoot


def test_rcas_adds_crispness():
    low = _box_downscale(_native(), 2)
    easu = easu_upscale(low, 2.0)
    fsr = fsr_upscale(low, 2.0, sharpness=0.4)
    assert _edge_energy(fsr) > _edge_energy(easu)


def test_flat_preserved():
    assert np.allclose(easu_upscale(np.full((32, 32, 3), 0.4), 2.0), 0.4, atol=1e-6)


def test_upscale_shape():
    up = easu_upscale(np.zeros((30, 40, 3)), 2.0)
    assert up.shape == (60, 80, 3)


def test_deterministic():
    low = _box_downscale(_native(), 2)
    assert np.array_equal(easu_upscale(low, 2.0), easu_upscale(low, 2.0))

"""Tests for the optional FFT backend (FFT-1). Default numpy must stay byte-identical; pyfftw is opt-in/off."""
import numpy as np
import holographic_fft as F
from holographic_ai import bind, bind_batch


def test_default_backend_is_numpy_byte_identical():
    assert F.fft_backend() == "numpy"
    a = np.random.default_rng(0).standard_normal(1024)
    assert np.array_equal(F.rfft(a), np.fft.rfft(a))
    assert np.array_equal(F.irfft(F.rfft(a), n=1024), np.fft.irfft(np.fft.rfft(a), n=1024))


def test_bind_unchanged_by_wiring():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(1024); b = rng.standard_normal(1024)
    assert np.array_equal(bind(a, b), np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=1024))
    A = rng.standard_normal((6, 512)); B = rng.standard_normal((6, 512))
    assert np.array_equal(bind_batch(A, B),
                          np.fft.irfft(np.fft.rfft(A, axis=1) * np.fft.rfft(B, axis=1), n=512, axis=1))


def test_pyfftw_optin_and_restore():
    if not F.HAS_PYFFTW:
        assert F.fft_backend() == "numpy"
        return
    a = np.random.default_rng(2).standard_normal(1024)
    F.use_pyfftw(True)
    try:
        assert F.fft_backend() == "pyfftw"
        assert np.allclose(F.rfft(a), np.fft.rfft(a), atol=1e-10)   # tolerance, not bit-exact
    finally:
        F.use_pyfftw(False)
    assert F.fft_backend() == "numpy"                         # restored to deterministic default

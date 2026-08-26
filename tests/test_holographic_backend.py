"""Tests for the optional GPU backend -- the NumPy fallback path (what runs with no CUDA device) (BACKEND-1)."""
import numpy as np
from holographic.misc.holographic_backend import array_module, get_array_module, to_device, asnumpy, on_gpu_island, gpu_available, gpu_enabled, enable_gpu, device_report


def test_cpu_module_is_numpy():
    assert array_module("cpu") is np
    assert get_array_module(np.zeros(3)) is np


def test_host_device_roundtrip_is_identity_on_cpu():
    x = np.arange(12.0).reshape(3, 4)
    assert np.array_equal(asnumpy(to_device(x)), x)


def test_gpu_island_returns_numpy():
    @on_gpu_island
    def k(a, b):
        xp = get_array_module(a, b)
        return xp.fft.irfft(xp.fft.rfft(a, axis=-1) * xp.fft.rfft(b, axis=-1), n=a.shape[-1], axis=-1)
    a = np.random.default_rng(0).standard_normal((3, 16)); b = np.random.default_rng(1).standard_normal((3, 16))
    out = k(a, b)
    assert isinstance(out, np.ndarray) and out.shape == (3, 16)


def test_enable_gpu_does_not_force_when_unavailable():
    # toggling the request must never claim GPU is active when no device is present
    state = enable_gpu(True)
    assert state == (gpu_available() and True)                     # active only if truly available
    enable_gpu(False)
    assert not gpu_enabled()


def test_device_report_is_string():
    assert isinstance(device_report(), str)

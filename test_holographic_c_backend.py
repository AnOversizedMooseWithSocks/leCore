import os
import subprocess
import sys

import numpy as np
import pytest

import holographic_c
from holographic_ai import cosine, random_vector, unitary_vector


pytestmark = pytest.mark.skipif(
    not holographic_c.available(),
    reason="C holographic shared library is not built",
)


def _numpy_bind(a, b):
    return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))


def _numpy_bind_fixed(role, rows):
    return np.fft.irfft(
        np.fft.rfft(role)[None, :] * np.fft.rfft(rows, axis=1),
        n=rows.shape[1],
        axis=1,
    )


def _numpy_weighted_sum(rows, weights=None):
    if weights is None:
        return np.sum(rows, axis=0)
    return np.sum(rows * np.asarray(weights)[:, None], axis=0)


def test_c_bind_matches_numpy_fft():
    rng = np.random.default_rng(7)
    a = random_vector(256, rng)
    b = random_vector(256, rng)
    assert np.allclose(holographic_c.bind(a, b), _numpy_bind(a, b), atol=1e-10)


def test_c_bind_fixed_matches_numpy_rows():
    rng = np.random.default_rng(17)
    role = random_vector(256, rng)
    rows = np.stack([random_vector(256, rng) for _ in range(6)])
    got = holographic_c.bind_fixed(role, rows)
    want = _numpy_bind_fixed(role, rows)
    assert got.shape == rows.shape
    assert np.allclose(got, want, atol=1e-10)
    for i in range(rows.shape[0]):
        assert np.allclose(got[i], holographic_c.bind(role, rows[i]), atol=1e-10)


def test_c_weighted_sum_matches_numpy_rows_without_normalizing():
    rng = np.random.default_rng(23)
    rows = np.stack([random_vector(256, rng) for _ in range(7)])
    weights = rng.normal(size=rows.shape[0])
    got = holographic_c.weighted_sum(rows, weights)
    want = _numpy_weighted_sum(rows, weights)
    assert np.allclose(got, want, atol=1e-12)
    assert np.allclose(holographic_c.weighted_sum(rows), _numpy_weighted_sum(rows), atol=1e-12)
    assert not np.isclose(np.linalg.norm(got), 1.0)


def test_c_memory_recalls_unitary_key_value_pair():
    rng = np.random.default_rng(8)
    key = unitary_vector(512, rng)
    value = random_vector(512, rng)
    mem = holographic_c.HolographicMemory(512)
    mem.learn(key, value)
    assert cosine(mem.recall(key), value) > 0.999999
    assert np.linalg.norm(mem.trace) > 0.0


def test_holographic_ai_can_install_c_backend_by_env():
    env = os.environ.copy()
    env["HOLOSTUFF_USE_C"] = "1"
    env["HOLOSTUFF_C_STRICT"] = "1"
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            (
                "import holographic_ai, holographic_c; "
                "assert holographic_ai.HolographicMemory is holographic_c.HolographicMemory; "
                "assert holographic_ai.bind is holographic_c.bind; "
                "assert holographic_ai.bind_fixed is holographic_c.bind_fixed; "
                "assert holographic_ai.weighted_sum is holographic_c.weighted_sum"
            ),
        ],
        env=env,
    )


def test_c_backend_explicit_install_updates_symbol_table():
    symbols = {}
    assert holographic_c.install(symbols, strict=True)
    assert symbols["bind"] is holographic_c.bind
    assert symbols["bind_fixed"] is holographic_c.bind_fixed
    assert symbols["weighted_sum"] is holographic_c.weighted_sum
    assert symbols["unbind"] is holographic_c.unbind
    assert symbols["HolographicMemory"] is holographic_c.HolographicMemory

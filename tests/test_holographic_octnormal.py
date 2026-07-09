"""Tests for octahedral normal encoding (holographic_octnormal): the S^2 -> 2-number manifold-quantization map. The
continuous map is an exact bijection; quantizing the 2 intrinsic DOF beats quantizing 3 ambient x/y/z components at
equal storage."""

import numpy as np

from holographic.mesh_and_geometry.holographic_octnormal import oct_encode, oct_decode, oct_quantize, oct_dequantize


def _unit(n):
    rng = np.random.default_rng(n)
    v = rng.standard_normal((5000, 3))
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _ang(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1, 1)))


def test_continuous_roundtrip_is_exact():
    N = _unit(0)
    assert _ang(N, oct_decode(oct_encode(N))).max() < 1e-3


def test_axis_aligned_and_poles_roundtrip():
    axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1.0], [-1, 0, 0], [0, -1, 0]])
    assert _ang(axes, oct_decode(oct_encode(axes))).max() < 1e-6   # incl. the z<0 pole (the fold edge case)


def test_quantized_8bit_error_is_small_and_bounded():
    N = _unit(1)
    err = _ang(N, oct_dequantize(oct_quantize(N, 8), 8))
    assert err.max() < 1.0


def test_codes_are_in_valid_range():
    codes = oct_quantize(_unit(2), 8)
    assert codes.min() >= 0 and codes.max() < (1 << 8) and codes.shape[-1] == 2


def test_decode_outputs_unit_vectors():
    out = oct_dequantize(oct_quantize(_unit(3), 10), 10)
    assert np.allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-9)


def test_more_bits_lower_error():
    N = _unit(4)
    e8 = _ang(N, oct_dequantize(oct_quantize(N, 8), 8)).mean()
    e12 = _ang(N, oct_dequantize(oct_quantize(N, 12), 12)).mean()
    assert e12 < e8


def test_octahedral_beats_naive_xyz_at_equal_budget():
    N = _unit(5)
    oct16 = oct_dequantize(oct_quantize(N, 8), 8)                  # 8+8 = 16 bits

    def qn(a, bits):
        levels = (1 << bits) - 1
        return np.round((a + 1) * 0.5 * levels) / levels * 2 - 1

    nb = np.stack([qn(N[:, 0], 5), qn(N[:, 1], 5), qn(N[:, 2], 6)], axis=-1)   # 5+5+6 = 16 bits
    nb = nb / np.linalg.norm(nb, axis=-1, keepdims=True)
    assert _ang(N, oct16).mean() < _ang(N, nb).mean()


def test_deterministic():
    N = _unit(6)
    assert np.array_equal(oct_quantize(N, 8), oct_quantize(N, 8))
    assert np.array_equal(oct_dequantize(oct_quantize(N, 8), 8), oct_dequantize(oct_quantize(N, 8), 8))

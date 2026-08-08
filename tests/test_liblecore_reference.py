"""Definitional utility checks used by the native liblecore conformance harness."""

import numpy as np

from holographic.misc.holographic_reference import (
    ref_cleanup,
    ref_cosine_many,
    ref_cosine_many_f64_f32,
    ref_dot,
    ref_dot_many,
    ref_normalize,
)


def test_reference_normalize_and_ordered_dot_edges():
    zero = np.zeros(4, dtype=np.float64)
    assert np.array_equal(ref_normalize(zero), zero)
    assert np.allclose(ref_normalize([3.0, 4.0]), [0.6, 0.8])

    # This cancellation pins the documented ascending component order.
    assert ref_dot([1e16, 1.0, -1e16], [1.0, 1.0, 1.0]) == 0.0
    assert np.array_equal(ref_dot_many([1.0, 2.0], [[3.0, 4.0], [5.0, 6.0]]), [11.0, 17.0])


def test_reference_cleanup_pins_ties_nan_and_zero_precedence():
    codebook = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert ref_cleanup([1.0, 0.0], codebook) == (0, 1.0)

    nan_index, nan_score = ref_cleanup([np.nan, 1.0], codebook)
    assert nan_index == 0
    assert np.isnan(nan_score)

    # ref_cosine checks the zero norm before evaluating the dot, even when the other row contains NaN.
    scores = ref_cosine_many([0.0, 0.0], [[np.nan, 1.0]])
    assert np.array_equal(scores, [0.0])


def test_reference_mixed_f64_f32_cosine_is_explicit():
    query = np.asarray([1.0, 2.0], dtype=np.float64)
    corpus = np.asarray([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]], dtype=np.float32)
    scores = ref_cosine_many_f64_f32(query, corpus)
    assert scores.dtype == np.float64
    assert np.allclose(scores, [1.0 / np.sqrt(5.0), 2.0 / np.sqrt(5.0), 0.0])

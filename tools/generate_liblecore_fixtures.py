#!/usr/bin/env python3
"""Generate the committed liblecore ISA-1 differential fixture corpus.

The corpus uses no random-number generator.  Every finite input comes from the
integer/dyadic ``index-formula-v1`` below, and the JSON records the resulting
IEEE-754 bits explicitly.  This keeps fixture meaning stable across language
runtimes and avoids making a particular RNG implementation part of the ISA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from holographic.misc.holographic_reference import (  # noqa: E402
    ref_bind,
    ref_bundle,
    ref_cleanup,
    ref_cosine,
    ref_cosine_many,
    ref_cosine_many_f64_f32,
    ref_dot,
    ref_dot_many,
    ref_involution,
    ref_normalize,
    ref_permute,
    ref_unbind,
)


DEFAULT_OUTPUT = REPOSITORY_ROOT / "tests/native/fixtures/liblecore_isa_v1.json"
FULL_DIMENSIONS = (1, 2, 3, 7, 8, 64)
REDUCTION_DIMENSIONS = (384, 1024, 4096)
EDGE_DIMENSIONS = (3, 8)


def _sequence(dimension: int, lane: int, dtype: np.dtype) -> np.ndarray:
    """Bounded, nonzero dyadic data defined only by integer arithmetic."""
    values = []
    for index in range(dimension):
        numerator = (
            ((index + 1) * (17 * lane + 11) + 13 * lane * lane + 5 * index * index)
            % 193
        ) - 96
        if numerator == 0:
            numerator = lane + 1
        sign = -1 if (index + lane) % 5 == 0 else 1
        values.append(sign * numerator / 64.0)
    return np.asarray(values, dtype=dtype)


def _encode(values: object, dtype: np.dtype) -> dict:
    dtype = np.dtype(dtype)
    array = np.asarray(values, dtype=dtype)
    if dtype == np.dtype(np.float64):
        unsigned = array.reshape(-1).view(np.uint64)
        width = 16
    elif dtype == np.dtype(np.float32):
        unsigned = array.reshape(-1).view(np.uint32)
        width = 8
    else:
        raise TypeError(f"unsupported fixture dtype {dtype}")
    return {
        "dtype": dtype.name,
        "shape": list(array.shape),
        "bits": [format(int(value), f"0{width}x") for value in unsigned],
    }


def _ordered_dot(a: np.ndarray, b: np.ndarray, dtype: np.dtype):
    scalar = np.dtype(dtype).type
    result = scalar(0.0)
    for index in range(len(a)):
        result = scalar(result + scalar(a[index] * b[index]))
    return result


def _normalize(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    scalar = np.dtype(dtype).type
    norm_squared = _ordered_dot(values, values, dtype)
    norm = scalar(np.sqrt(norm_squared))
    if norm > scalar(0.0):
        return np.asarray([scalar(value / norm) for value in values], dtype=dtype)
    return values.copy()


def _bind(a: np.ndarray, b: np.ndarray, dtype: np.dtype) -> np.ndarray:
    scalar = np.dtype(dtype).type
    dimension = len(a)
    output = np.empty(dimension, dtype=dtype)
    for output_index in range(dimension):
        total = scalar(0.0)
        for left_index in range(dimension):
            right_index = (output_index - left_index) % dimension
            total = scalar(total + scalar(a[left_index] * b[right_index]))
        output[output_index] = total
    return output


def _involution(values: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return values.copy()
    return np.concatenate((values[:1], values[:0:-1]))


def _unbind(composite: np.ndarray, key: np.ndarray, dtype: np.dtype) -> np.ndarray:
    return _bind(composite, _involution(key), dtype)


def _cosine(a: np.ndarray, b: np.ndarray, dtype: np.dtype):
    scalar = np.dtype(dtype).type
    norm_a_squared = _ordered_dot(a, a, dtype)
    norm_b_squared = _ordered_dot(b, b, dtype)
    if norm_a_squared == scalar(0.0) or norm_b_squared == scalar(0.0):
        return scalar(0.0)
    norm_a = scalar(np.sqrt(norm_a_squared))
    norm_b = scalar(np.sqrt(norm_b_squared))
    return scalar(_ordered_dot(a, b, dtype) / scalar(norm_a * norm_b))


def _bundle(rows: np.ndarray, dtype: np.dtype) -> np.ndarray:
    scalar = np.dtype(dtype).type
    total = np.zeros(rows.shape[1], dtype=dtype)
    for row in rows:
        for index in range(rows.shape[1]):
            total[index] = scalar(total[index] + row[index])
    return _normalize(total, dtype)


def _f64_references(
    a: np.ndarray,
    b: np.ndarray,
    rows: np.ndarray,
    shift: int,
) -> dict:
    cleanup_index, cleanup_score = ref_cleanup(a, rows)
    return {
        "normalize": _encode(ref_normalize(a), np.float64),
        "dot": _encode(ref_dot(a, b), np.float64),
        "dot_many": _encode(ref_dot_many(a, rows), np.float64),
        "cosine": _encode(ref_cosine(a, b), np.float64),
        "cosine_many": _encode(ref_cosine_many(a, rows), np.float64),
        "bind": _encode(ref_bind(a, b), np.float64),
        "unbind": _encode(ref_unbind(a, b), np.float64),
        "involution": _encode(ref_involution(a), np.float64),
        "permute": _encode(ref_permute(a, shift), np.float64),
        "bundle": _encode(ref_bundle(rows), np.float64),
        "cleanup": {
            "index": cleanup_index,
            "score": _encode(cleanup_score, np.float64),
        },
    }


def _f32_references(
    a: np.ndarray,
    b: np.ndarray,
    rows: np.ndarray,
    shift: int,
) -> dict:
    scores = np.asarray([_cosine(a, row, np.float32) for row in rows], dtype=np.float32)
    cleanup_index = int(np.argmax(scores))
    return {
        "normalize": _encode(_normalize(a, np.float32), np.float32),
        "dot": _encode(_ordered_dot(a, b, np.float32), np.float32),
        "dot_many": _encode(
            [_ordered_dot(a, row, np.float32) for row in rows], np.float32
        ),
        "cosine": _encode(_cosine(a, b, np.float32), np.float32),
        "cosine_many": _encode(scores, np.float32),
        "bind": _encode(_bind(a, b, np.float32), np.float32),
        "unbind": _encode(_unbind(a, b, np.float32), np.float32),
        "involution": _encode(_involution(a), np.float32),
        "permute": _encode(np.roll(a, shift), np.float32),
        "bundle": _encode(_bundle(rows, np.float32), np.float32),
        "cleanup": {
            "index": cleanup_index,
            "score": _encode(scores[cleanup_index], np.float32),
        },
    }


def _finite_case(dimension: int, dtype: np.dtype) -> dict:
    dtype = np.dtype(dtype)
    a = _sequence(dimension, 1, dtype)
    b = _sequence(dimension, 2, dtype)
    rows = np.stack(
        (b, _sequence(dimension, 3, dtype), _sequence(dimension, 4, dtype))
    )
    paired_left = np.stack((a, b, rows[1]))
    paired_right = np.stack((b, rows[1], rows[2]))
    shift = -(dimension + 2)
    if dtype == np.dtype(np.float64):
        expected = _f64_references(a, b, rows, shift)
        bind = ref_bind
        unbind = ref_unbind
    else:
        expected = _f32_references(a, b, rows, shift)
        bind = lambda left, right: _bind(left, right, np.float32)
        unbind = lambda trace, key: _unbind(trace, key, np.float32)

    expected["bind_batch"] = _encode(
        [bind(left, right) for left, right in zip(paired_left, paired_right)], dtype
    )
    expected["bind_fixed"] = _encode([bind(a, row) for row in rows], dtype)
    expected["unbind_all"] = _encode([unbind(a, key) for key in rows], dtype)

    result = {
        "name": f"formula-d{dimension}",
        "dimension": dimension,
        "shift": shift,
        "inputs": {
            "a": _encode(a, dtype),
            "b": _encode(b, dtype),
            "rows": _encode(rows, dtype),
            "paired_left": _encode(paired_left, dtype),
            "paired_right": _encode(paired_right, dtype),
        },
        "expected": expected,
    }
    if dtype == np.dtype(np.float32):
        mixed_query = np.asarray(a, dtype=np.float64) + np.float64(1.0 / 128.0)
        result["inputs"]["mixed_query"] = _encode(mixed_query, np.float64)
        result["expected"]["cosine_many_f64_f32"] = _encode(
            ref_cosine_many_f64_f32(mixed_query, rows), np.float64
        )
    return result


def _reduction_case(dimension: int, dtype: np.dtype) -> dict:
    dtype = np.dtype(dtype)
    query = _sequence(dimension, 5, dtype)
    rows = np.stack((_sequence(dimension, 6, dtype), _sequence(dimension, 7, dtype)))
    if dtype == np.dtype(np.float64):
        dots = ref_dot_many(query, rows)
        cosines = ref_cosine_many(query, rows)
    else:
        dots = np.asarray(
            [_ordered_dot(query, row, np.float32) for row in rows], dtype=np.float32
        )
        cosines = np.asarray(
            [_cosine(query, row, np.float32) for row in rows], dtype=np.float32
        )
    return {
        "name": f"ordered-reduction-d{dimension}",
        "dimension": dimension,
        "inputs": {"query": _encode(query, dtype), "rows": _encode(rows, dtype)},
        "expected": {
            "dot_many": _encode(dots, dtype),
            "cosine_many": _encode(cosines, dtype),
        },
    }


def _edge_case(dimension: int, dtype: np.dtype) -> dict:
    dtype = np.dtype(dtype)
    finite = _sequence(dimension, 9, dtype)
    zero = np.zeros(dimension, dtype=dtype)
    nan_vector = finite.copy()
    nan_vector[0] = np.nan
    inf_vector = finite.copy()
    inf_vector[0] = np.inf
    impulse = np.zeros(dimension, dtype=dtype)
    impulse[0] = 1.0
    shifted_impulse = np.zeros(dimension, dtype=dtype)
    shifted_impulse[1] = 1.0
    ties = np.stack((finite, finite, -finite))
    first_nan_candidates = np.stack((finite, nan_vector, finite))
    reindex_special = finite.copy()
    if dtype == np.dtype(np.float64):
        reindex_bits = reindex_special.view(np.uint64)
        reindex_bits[0] = np.uint64(0x7FF8000000000042)
        if dimension > 1:
            reindex_bits[1] = np.uint64(0x8000000000000000)
        if dimension > 2:
            reindex_bits[2] = np.uint64(0x7FF0000000000000)
        if dimension > 3:
            reindex_bits[3] = np.uint64(0xFFF0000000000000)
    else:
        reindex_bits = reindex_special.view(np.uint32)
        reindex_bits[0] = np.uint32(0x7FC00042)
        if dimension > 1:
            reindex_bits[1] = np.uint32(0x80000000)
        if dimension > 2:
            reindex_bits[2] = np.uint32(0x7F800000)
        if dimension > 3:
            reindex_bits[3] = np.uint32(0xFF800000)
    return {
        "dimension": dimension,
        "inputs": {
            "finite": _encode(finite, dtype),
            "zero": _encode(zero, dtype),
            "nan_vector": _encode(nan_vector, dtype),
            "inf_vector": _encode(inf_vector, dtype),
            "impulse": _encode(impulse, dtype),
            "shifted_impulse": _encode(shifted_impulse, dtype),
            "ties": _encode(ties, dtype),
            "first_nan_candidates": _encode(first_nan_candidates, dtype),
            "zero_sum_rows": _encode(np.stack((finite, -finite)), dtype),
            "reindex_special": _encode(reindex_special, dtype),
        },
        "expected": {
            "impulse_bind": _encode(finite, dtype),
            "shifted_impulse_bind": _encode(np.roll(finite, 1), dtype),
            "zero_normalize": _encode(zero, dtype),
            "zero_sum_bundle": _encode(zero, dtype),
            "zero_nan_cosine": _encode(np.asarray(0.0, dtype=dtype), dtype),
            "tie_cleanup_index": 0,
            "first_nan_cleanup_index": 1,
            "first_nan_cleanup_score_class": "nan",
            "raw_nan_result_class": "nan",
            "finite_validation_status": "LECORE_ENONFINITE",
            "special_involution": _encode(_involution(reindex_special), dtype),
            "special_permute_minus_two": _encode(
                np.roll(reindex_special, -2), dtype
            ),
        },
    }


def build_fixture() -> dict:
    profiles = {}
    profile_specs = (
        (
            "HRR_F64_V1",
            np.dtype(np.float64),
            "0x00010001",
            {"atol": 1e-9, "rtol": 0.0, "status": "frozen"},
        ),
        (
            "HRR_F32_V1",
            np.dtype(np.float32),
            "0x00010002",
            {"atol": 1e-5, "rtol": 0.0, "status": "preview-hypothesis"},
        ),
    )
    for name, dtype, profile_id, tolerance in profile_specs:
        profiles[name] = {
            "profile_id": profile_id,
            "dtype": dtype.name,
            "tolerance": tolerance,
            "input_domain": {
                "finite_formula_range": [-1.5, 1.5],
                "rounding": "round-to-nearest",
                "subnormal_inputs": "excluded",
                "nonzero_subnormal_outputs": "excluded",
            },
            "finite_cases": [_finite_case(dimension, dtype) for dimension in FULL_DIMENSIONS],
            "reduction_cases": [
                _reduction_case(dimension, dtype) for dimension in REDUCTION_DIMENSIONS
            ],
            "edge_cases": [_edge_case(dimension, dtype) for dimension in EDGE_DIMENSIONS],
        }

    return {
        "schema": "org.lecore.conformance-fixture.v1",
        "abi_preview": 0,
        "isa_version": 1,
        "generator": {
            "owner": "leCore",
            "path": "tools/generate_liblecore_fixtures.py",
            "algorithm": "index-formula-v1/no-rng",
            "formula": "sign(i,lane)*((((i+1)*(17*lane+11)+13*lane^2+5*i^2) mod 193)-96)/64",
            "sign": "-1 when (i+lane) mod 5 is 0; +1 otherwise",
            "zero_numerator_rule": "replace an exact zero numerator with lane+1",
        },
        "encoding": {
            "float_bits": "fixed-width lowercase IEEE-754 hexadecimal in logical scalar order",
            "shape": "row-major C order",
            "byte_order": "bit-pattern integers; independent of host byte order",
        },
        "dimensions": {
            "full_operation_cases": list(FULL_DIMENSIONS),
            "ordered_reduction_cases": list(REDUCTION_DIMENSIONS),
            "edge_cases": list(EDGE_DIMENSIONS),
        },
        "decision_rules": {
            "cleanup": "first NaN; otherwise highest score; exact ties choose lowest index",
            "reindex": "bit exact",
        },
        "profiles": profiles,
    }


def render_fixture() -> str:
    return json.dumps(build_fixture(), indent=2, sort_keys=True, allow_nan=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when the committed fixture differs",
    )
    arguments = parser.parse_args()
    rendered = render_fixture()
    if arguments.check:
        try:
            existing = arguments.output.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise SystemExit(f"fixture is missing: {arguments.output}")
        if existing != rendered:
            raise SystemExit(
                f"fixture drift: run {Path(__file__).relative_to(REPOSITORY_ROOT)}"
            )
        print(f"fixture is current: {arguments.output}")
        return
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()

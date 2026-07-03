#!/usr/bin/env python3
"""Compare NumPy bind_fixed with the C fixed-vector batch binding path."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

os.environ.pop("HOLOSTUFF_USE_C", None)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from holographic_ai import bind_fixed as numpy_bind_fixed, random_vector  # noqa: E402
import holographic_c  # noqa: E402


def _timed_calls(fn, role, rows, loops: int) -> float:
    t0 = time.perf_counter()
    for _ in range(loops):
        fn(role, rows)
    return time.perf_counter() - t0


def bench_one(dim: int, row_count: int, loops: int, repeats: int) -> dict[str, float | int | str]:
    rng = np.random.default_rng(9000 + dim + row_count)
    role = random_vector(dim, rng)
    rows = (
        np.stack([random_vector(dim, rng) for _ in range(row_count)])
        if row_count
        else np.empty((0, dim), dtype=np.float64)
    )

    want = numpy_bind_fixed(role, rows)
    got = holographic_c.bind_fixed(role, rows)
    max_abs = float(np.max(np.abs(got - want))) if row_count else 0.0
    if max_abs > 1e-9:
        raise AssertionError(f"C bind_fixed mismatch dim={dim} rows={row_count}: {max_abs}")

    numpy_bind_fixed(role, rows)
    holographic_c.bind_fixed(role, rows)

    py_seconds = [_timed_calls(numpy_bind_fixed, role, rows, loops) for _ in range(repeats)]
    c_seconds = [_timed_calls(holographic_c.bind_fixed, role, rows, loops) for _ in range(repeats)]
    py_median = statistics.median(py_seconds)
    c_median = statistics.median(c_seconds)
    return {
        "runtime": "bind_fixed_summary",
        "dim": dim,
        "rows": row_count,
        "loops": loops,
        "repeats": repeats,
        "max_abs": max_abs,
        "c_path_expected": holographic_c._bind_fixed_uses_c(row_count, dim),
        "python_calls_per_second_median": loops / py_median if py_median else 0.0,
        "c_calls_per_second_median": loops / c_median if c_median else 0.0,
        "speedup_c_over_python": py_median / c_median if c_median else 0.0,
        "backend": holographic_c.backend_path() or "none",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", default="128,256,512,1024")
    parser.add_argument("--rows", default="1,8,32")
    parser.add_argument("--loops", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if not holographic_c.available():
        raise RuntimeError("C holographic shared library is not built")

    dims = [int(x) for x in args.dims.split(",") if x.strip()]
    row_counts = [int(x) for x in args.rows.split(",") if x.strip()]
    for dim in dims:
        for row_count in row_counts:
            row = bench_one(dim, row_count, args.loops, args.repeats)
            print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

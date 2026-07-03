#!/usr/bin/env python3
"""Compare NumPy HolographicMemory with the C trace kernel.

The timed section excludes vector/key setup. It measures the architectural hot
path: store bind(state, action) into one trace, then recall/cleanup actions from
query states.
"""

from __future__ import annotations

import argparse
import statistics
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from holographic_ai import HolographicMemory, random_vector, unitary_vector  # noqa: E402


def python_run(dim: int, pairs: int, actions_n: int, queries: int) -> dict[str, float | int | str]:
    rng = np.random.default_rng(1234)
    states = np.stack([unitary_vector(dim, rng) for _ in range(pairs)])
    actions = np.stack([random_vector(dim, rng) for _ in range(actions_n)])
    mem = HolographicMemory(dim)

    t0 = time.perf_counter()
    for i in range(pairs):
        mem.learn(states[i], actions[i % actions_n])
    store_seconds = time.perf_counter() - t0

    correct = 0
    t0 = time.perf_counter()
    for i in range(queries):
        j = i % pairs
        est = mem.recall(states[j])
        pred = int(np.argmax(actions @ est))
        correct += pred == (j % actions_n)
    query_seconds = time.perf_counter() - t0

    return {
        "runtime": "python_numpy",
        "dim": dim,
        "pairs": pairs,
        "actions": actions_n,
        "queries": queries,
        "store_seconds": store_seconds,
        "query_seconds": query_seconds,
        "stores_per_second": pairs / store_seconds if store_seconds else 0.0,
        "queries_per_second": queries / query_seconds if query_seconds else 0.0,
        "accuracy": correct / queries if queries else 0.0,
    }


def c_run(binary: Path, dim: int, pairs: int, actions_n: int, queries: int) -> dict:
    out = subprocess.check_output(
        [str(binary), str(dim), str(pairs), str(actions_n), str(queries)],
        text=True,
    )
    return json.loads(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", default="128,256,512,1024")
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--actions", type=int, default=8)
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--binary", type=Path)
    args = parser.parse_args()

    c_dir = Path(__file__).resolve().parents[1]
    if args.binary:
        binary = args.binary
    else:
        binary = c_dir / "build" / "scalar" / "bench_trace"
        subprocess.check_call(["make", "-C", str(c_dir), str(binary.relative_to(c_dir))])

    dims = [int(x) for x in args.dims.split(",") if x.strip()]
    rows = []
    for dim in dims:
        for _ in range(args.repeats):
            py = python_run(dim, args.pairs, args.actions, args.queries)
            c = c_run(binary, dim, args.pairs, args.actions, args.queries)
            rows.extend([py, c])
            if not args.summary:
                print(json.dumps(py, sort_keys=True))
                print(json.dumps(c, sort_keys=True))
    if args.summary:
        by_dim: dict[int, dict[str, list[dict]]] = {}
        for row in rows:
            by_dim.setdefault(int(row["dim"]), {}).setdefault(str(row["runtime"]), []).append(row)
        for dim in dims:
            py_rows = by_dim[dim]["python_numpy"]
            c_runtime = next(runtime for runtime in by_dim[dim] if runtime != "python_numpy")
            c_rows = by_dim[dim][c_runtime]
            py_store = statistics.median(float(r["stores_per_second"]) for r in py_rows)
            c_store = statistics.median(float(r["stores_per_second"]) for r in c_rows)
            py_query = statistics.median(float(r["queries_per_second"]) for r in py_rows)
            c_query = statistics.median(float(r["queries_per_second"]) for r in c_rows)
            summary = {
                "runtime": "summary",
                "c_runtime": c_runtime,
                "dim": dim,
                "pairs": args.pairs,
                "actions": args.actions,
                "queries": args.queries,
                "repeats": args.repeats,
                "python_store_per_second_median": py_store,
                "c_store_per_second_median": c_store,
                "store_speedup_c_over_python": c_store / py_store if py_store else 0.0,
                "python_query_per_second_median": py_query,
                "c_query_per_second_median": c_query,
                "query_speedup_c_over_python": c_query / py_query if py_query else 0.0,
                "python_accuracy_median": statistics.median(float(r["accuracy"]) for r in py_rows),
                "c_accuracy_median": statistics.median(float(r["accuracy"]) for r in c_rows),
            }
            print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

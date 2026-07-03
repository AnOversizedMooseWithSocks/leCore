#!/usr/bin/env python3
"""Compare Python HoloMachine execution with the C core program runner."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.pop("HOLOSTUFF_USE_C", None)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from holographic_ai import cosine  # noqa: E402
from holographic_machine import HoloMachine  # noqa: E402


def _split_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def make_program(length: int, names: list[str]) -> list[tuple[str, str]]:
    if length < 2:
        raise ValueError("program length must include at least LOAD and HALT")
    ops = ["BIND", "BUNDLE"]
    program = [("LOAD", names[0])]
    for i in range(1, length - 1):
        program.append((ops[(i - 1) % len(ops)], names[i % len(names)]))
    program.append(("HALT", ""))
    return program


def bench_one(dim: int, length: int, loops: int, repeats: int) -> dict[str, float | int | str]:
    vm = HoloMachine(dim=dim, seed=7)
    program = make_program(length, vm.data_names)
    program_vec = vm.assemble(program)
    py_acc, py_trace = vm.run(program_vec, max_steps=length)
    c_acc, c_trace = vm.run_c_basic(program_vec, max_steps=length)
    expected_trace = program[:-1]
    if py_trace != expected_trace or c_trace != expected_trace:
        raise RuntimeError(f"trace mismatch: python={py_trace!r} c={c_trace!r}")
    parity = cosine(py_acc, c_acc)
    if parity < 0.999999:
        raise RuntimeError(f"C program runner drifted from Python VM: cosine={parity}")

    py_rates = []
    c_rates = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(loops):
            vm.run(program_vec, max_steps=length)
        py_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(loops):
            vm.run_c_basic(program_vec, max_steps=length)
        c_seconds = time.perf_counter() - t0
        py_rates.append(loops / py_seconds if py_seconds else 0.0)
        c_rates.append(loops / c_seconds if c_seconds else 0.0)

    py_rate = statistics.median(py_rates)
    c_rate = statistics.median(c_rates)
    return {
        "runtime": "program_summary",
        "dim": dim,
        "instructions": length - 1,
        "encoded_slots": length,
        "loops": loops,
        "repeats": repeats,
        "python_runs_per_second_median": py_rate,
        "c_runs_per_second_median": c_rate,
        "speedup_c_over_python": c_rate / py_rate if py_rate else 0.0,
        "acc_cosine_c_vs_python": parity,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", default="1024,2048,4096")
    parser.add_argument("--lengths", default="4,8,12,16")
    parser.add_argument("--loops", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    rows = []
    for dim in _split_ints(args.dims):
        for length in _split_ints(args.lengths):
            row = bench_one(dim, length, args.loops, args.repeats)
            rows.append(row)
            print(json.dumps(row, sort_keys=True))

    if not args.summary:
        return 0
    speedups = [float(row["speedup_c_over_python"]) for row in rows]
    print(
        json.dumps(
            {
                "runtime": "program_geomean",
                "cases": len(rows),
                "speedup_c_over_python_geomean": statistics.geometric_mean(speedups),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure liblecore's direct/radix regimes without changing AUTO policy.

The benchmark includes Python/ctypes call overhead but reuses output buffers, so it is a realistic lower-level
adapter measurement rather than a claim about one isolated FFT instruction. Results are descriptive evidence;
this tool never rewrites dispatch thresholds.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Callable

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from bindings.python.lecore_native import Library  # noqa: E402
from holographic.agents_and_reasoning.holographic_ai import bind as numpy_bind  # noqa: E402


MINIMUM_OPTIMIZED_ITERATIONS = 1000


def timed_ns(operation: Callable[[], None], iterations: int, repeats: int) -> float:
    samples = []
    operation()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            started = time.perf_counter_ns()
            for _ in range(iterations):
                operation()
            samples.append((time.perf_counter_ns() - started) / iterations)
    finally:
        if was_enabled:
            gc.enable()
    return float(statistics.median(samples))


def setup_ns(library: Library, dimension: int, profile: str, backend: str, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        context = library.context(dimension, profile=profile, backend=backend)
        context.close()
        samples.append(time.perf_counter_ns() - started)
    return float(statistics.median(samples))


def benchmark_dimension(
    library: Library,
    dimension: int,
    profile: str,
    repeats: int,
    target_work: int,
) -> dict[str, object]:
    dtype = np.float64 if profile == "f64" else np.float32
    rng = np.random.default_rng(0x1EC0_0000 + dimension + (0 if profile == "f64" else 1))
    left = rng.standard_normal(dimension).astype(dtype)
    right = rng.standard_normal(dimension).astype(dtype)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)

    direct_iterations = max(1, min(2000, target_work // (dimension * dimension)))
    optimized_iterations = max(MINIMUM_OPTIMIZED_ITERATIONS, direct_iterations)
    outputs: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    scratch: dict[str, int] = {}

    for backend, iterations in (("direct", direct_iterations), ("radix2", optimized_iterations)):
        with library.context(dimension, profile=profile, backend=backend) as context:
            output = np.empty(dimension, dtype=dtype)

            def call_native() -> None:
                context.bind(left, right, out=output)

            timings[backend] = timed_ns(call_native, iterations, repeats)
            outputs[backend] = output.copy()
            scratch[backend] = context.scratch_bytes

    with library.context(dimension, profile=profile, backend="auto") as context:
        auto_backend = context.backend

    entry: dict[str, object] = {
        "dimension": dimension,
        "profile": profile,
        "auto_backend": auto_backend,
        "direct_iterations": direct_iterations,
        "optimized_iterations": optimized_iterations,
        "direct_ns": timings["direct"],
        "radix2_ns": timings["radix2"],
        "radix2_speedup": timings["direct"] / timings["radix2"],
        "direct_scratch_bytes": scratch["direct"],
        "radix2_scratch_bytes": scratch["radix2"],
        "radix2_max_abs_vs_direct": float(
            np.max(np.abs(outputs["radix2"].astype(np.float64) - outputs["direct"].astype(np.float64)))
        ),
        "direct_setup_ns": setup_ns(library, dimension, profile, "direct", repeats),
        "radix2_setup_ns": setup_ns(library, dimension, profile, "radix2", repeats),
    }

    if profile == "f64":
        numpy_output = np.empty(dimension, dtype=np.float64)

        def call_numpy() -> None:
            nonlocal numpy_output
            numpy_output = numpy_bind(left, right)

        numpy_time = timed_ns(call_numpy, optimized_iterations, repeats)
        entry.update(
            {
                "numpy_fft_ns": numpy_time,
                "radix2_speedup_vs_numpy": numpy_time / timings["radix2"],
                "numpy_max_abs_vs_direct": float(
                    np.max(np.abs(numpy_output - outputs["direct"]))
                ),
            }
        )
    return entry


def parse_dimensions(value: str) -> list[int]:
    dimensions = [int(item) for item in value.split(",") if item.strip()]
    if not dimensions or any(dimension <= 0 or dimension & (dimension - 1) for dimension in dimensions):
        raise argparse.ArgumentTypeError("dimensions must be a comma-separated list of positive powers of two")
    return dimensions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument(
        "--dimensions",
        type=parse_dimensions,
        default=parse_dimensions("8,16,32,64,128,256,512,1024"),
    )
    parser.add_argument("--profiles", choices=("f64", "f32", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--target-work", type=int, default=4_000_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()
    if arguments.repeats <= 0 or arguments.target_work <= 0:
        parser.error("repeats and target-work must be positive")

    library = Library(arguments.library)
    profiles = ("f64", "f32") if arguments.profiles == "both" else (arguments.profiles,)
    results = [
        benchmark_dimension(
            library,
            dimension,
            profile,
            arguments.repeats,
            arguments.target_work,
        )
        for profile in profiles
        for dimension in arguments.dimensions
    ]
    report = {
        "schema_version": 1,
        "policy_effect": "none; AUTO remains unchanged",
        "benchmark": {
            "repeats": arguments.repeats,
            "target_work": arguments.target_work,
            "minimum_optimized_iterations": MINIMUM_OPTIMIZED_ITERATIONS,
        },
        "library": {
            "path": library.path,
            "bytes": Path(library.path).stat().st_size,
            "version": library.version,
            "abi": library.abi_version,
            "isa": library.isa_version,
            "capabilities": library.capabilities,
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "results": results,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(encoded, encoding="utf-8")
    if arguments.json:
        print(encoded, end="")
    else:
        print("profile  dim   direct us  radix us  speedup  max |delta|")
        for result in results:
            print(
                f"{result['profile']:>7} {result['dimension']:>5} "
                f"{result['direct_ns'] / 1000:>10.2f} "
                f"{result['radix2_ns'] / 1000:>9.2f} "
                f"{result['radix2_speedup']:>8.2f} "
                f"{result['radix2_max_abs_vs_direct']:.3e}"
            )
        print("AUTO policy unchanged; adopter-backed promotion evidence is still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


MINIMUM_OPTIMIZED_ITERATIONS = 4000


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


def paired_timed_ns(
    candidate_operation: Callable[[], None],
    baseline_operation: Callable[[], None],
    iterations: int,
    repeats: int,
) -> tuple[float, float]:
    """Time two operations in alternating order to reduce ordering bias."""

    samples: tuple[list[float], list[float]] = ([], [])
    operations = (candidate_operation, baseline_operation)
    candidate_operation()
    baseline_operation()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for repeat in range(repeats):
            order = (0, 1) if repeat % 2 == 0 else (1, 0)
            for index in order:
                started = time.perf_counter_ns()
                for _ in range(iterations):
                    operations[index]()
                samples[index].append((time.perf_counter_ns() - started) / iterations)
    finally:
        if was_enabled:
            gc.enable()
    return tuple(float(statistics.median(values)) for values in samples)


def setup_ns(library: Library, dimension: int, profile: str, backend: str, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        context = library.context(dimension, profile=profile, backend=backend)
        context.close()
        samples.append(time.perf_counter_ns() - started)
    return float(statistics.median(samples))


def paired_setup_ns(
    candidate_library: Library,
    baseline_library: Library,
    dimension: int,
    profile: str,
    backend: str,
    repeats: int,
) -> tuple[float, float]:
    samples: tuple[list[float], list[float]] = ([], [])
    libraries = (candidate_library, baseline_library)
    for repeat in range(repeats):
        order = (0, 1) if repeat % 2 == 0 else (1, 0)
        for index in order:
            started = time.perf_counter_ns()
            context = libraries[index].context(dimension, profile=profile, backend=backend)
            context.close()
            samples[index].append(time.perf_counter_ns() - started)
    return tuple(float(statistics.median(values)) for values in samples)


def dimension_inputs(dimension: int, profile: str) -> tuple[np.ndarray, np.ndarray]:
    dtype = np.float64 if profile == "f64" else np.float32
    rng = np.random.default_rng(0x1EC0_0000 + dimension + (0 if profile == "f64" else 1))
    left = rng.standard_normal(dimension).astype(dtype)
    right = rng.standard_normal(dimension).astype(dtype)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    return left, right


def result_entry(
    dimension: int,
    profile: str,
    direct_iterations: int,
    optimized_iterations: int,
    timings: dict[str, float],
    outputs: dict[str, np.ndarray],
    scratch: dict[str, int],
    setup_timings: dict[str, float],
    auto_backend: int,
) -> dict[str, object]:
    return {
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
            np.max(
                np.abs(
                    outputs["radix2"].astype(np.float64)
                    - outputs["direct"].astype(np.float64)
                )
            )
        ),
        "direct_setup_ns": setup_timings["direct"],
        "radix2_setup_ns": setup_timings["radix2"],
    }


def benchmark_dimension(
    library: Library,
    dimension: int,
    profile: str,
    repeats: int,
    target_work: int,
) -> dict[str, object]:
    dtype = np.float64 if profile == "f64" else np.float32
    left, right = dimension_inputs(dimension, profile)

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

    setup_timings = {
        backend: setup_ns(library, dimension, profile, backend, repeats)
        for backend in ("direct", "radix2")
    }
    entry = result_entry(
        dimension,
        profile,
        direct_iterations,
        optimized_iterations,
        timings,
        outputs,
        scratch,
        setup_timings,
        auto_backend,
    )

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


def benchmark_dimension_pair(
    candidate_library: Library,
    baseline_library: Library,
    dimension: int,
    profile: str,
    repeats: int,
    target_work: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Benchmark candidate and baseline with identical work and alternating order."""

    dtype = np.float64 if profile == "f64" else np.float32
    left, right = dimension_inputs(dimension, profile)
    direct_iterations = max(1, min(2000, target_work // (dimension * dimension)))
    optimized_iterations = max(MINIMUM_OPTIMIZED_ITERATIONS, direct_iterations)
    libraries = (candidate_library, baseline_library)
    timings: tuple[dict[str, float], dict[str, float]] = ({}, {})
    outputs: tuple[dict[str, np.ndarray], dict[str, np.ndarray]] = ({}, {})
    scratch: tuple[dict[str, int], dict[str, int]] = ({}, {})
    setup_timings: tuple[dict[str, float], dict[str, float]] = ({}, {})

    for backend, iterations in (("direct", direct_iterations), ("radix2", optimized_iterations)):
        with candidate_library.context(
            dimension, profile=profile, backend=backend
        ) as candidate_context:
            with baseline_library.context(
                dimension, profile=profile, backend=backend
            ) as baseline_context:
                candidate_output = np.empty(dimension, dtype=dtype)
                baseline_output = np.empty(dimension, dtype=dtype)

                def call_candidate() -> None:
                    candidate_context.bind(left, right, out=candidate_output)

                def call_baseline() -> None:
                    baseline_context.bind(left, right, out=baseline_output)

                candidate_ns, baseline_ns = paired_timed_ns(
                    call_candidate,
                    call_baseline,
                    iterations,
                    repeats,
                )
                timings[0][backend] = candidate_ns
                timings[1][backend] = baseline_ns
                outputs[0][backend] = candidate_output.copy()
                outputs[1][backend] = baseline_output.copy()
                scratch[0][backend] = candidate_context.scratch_bytes
                scratch[1][backend] = baseline_context.scratch_bytes

        candidate_setup_ns, baseline_setup_ns = paired_setup_ns(
            candidate_library,
            baseline_library,
            dimension,
            profile,
            backend,
            repeats,
        )
        setup_timings[0][backend] = candidate_setup_ns
        setup_timings[1][backend] = baseline_setup_ns

    auto_backends = []
    for library in libraries:
        with library.context(dimension, profile=profile, backend="auto") as context:
            auto_backends.append(context.backend)

    entries = tuple(
        result_entry(
            dimension,
            profile,
            direct_iterations,
            optimized_iterations,
            timings[index],
            outputs[index],
            scratch[index],
            setup_timings[index],
            auto_backends[index],
        )
        for index in range(2)
    )

    if profile == "f64":
        numpy_output = np.empty(dimension, dtype=np.float64)

        def call_numpy() -> None:
            nonlocal numpy_output
            numpy_output = numpy_bind(left, right)

        numpy_time = timed_ns(call_numpy, optimized_iterations, repeats)
        for index, entry in enumerate(entries):
            entry.update(
                {
                    "numpy_fft_ns": numpy_time,
                    "radix2_speedup_vs_numpy": numpy_time / timings[index]["radix2"],
                    "numpy_max_abs_vs_direct": float(
                        np.max(np.abs(numpy_output - outputs[index]["direct"]))
                    ),
                }
            )
    return entries


def parse_dimensions(value: str) -> list[int]:
    dimensions = [int(item) for item in value.split(",") if item.strip()]
    if not dimensions or any(dimension <= 0 or dimension & (dimension - 1) for dimension in dimensions):
        raise argparse.ArgumentTypeError("dimensions must be a comma-separated list of positive powers of two")
    return dimensions


def build_report(
    library: Library,
    results: list[dict[str, object]],
    repeats: int,
    target_work: int,
    comparison_mode: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "policy_effect": "none; AUTO remains unchanged",
        "benchmark": {
            "repeats": repeats,
            "target_work": target_work,
            "minimum_optimized_iterations": MINIMUM_OPTIMIZED_ITERATIONS,
            "comparison_mode": comparison_mode,
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


def print_results(results: list[dict[str, object]]) -> None:
    print("profile  dim   direct us  radix us  speedup  max |delta|")
    for result in results:
        print(
            f"{result['profile']:>7} {result['dimension']:>5} "
            f"{result['direct_ns'] / 1000:>10.2f} "
            f"{result['radix2_ns'] / 1000:>9.2f} "
            f"{result['radix2_speedup']:>8.2f} "
            f"{result['radix2_max_abs_vs_direct']:.3e}"
        )


def print_paired_results(
    candidate_results: list[dict[str, object]],
    baseline_results: list[dict[str, object]],
) -> None:
    print("profile  dim   candidate direct/radix us   baseline direct/radix us   cand/base")
    for candidate, baseline in zip(candidate_results, baseline_results):
        print(
            f"{candidate['profile']:>7} {candidate['dimension']:>5} "
            f"{candidate['direct_ns'] / 1000:>10.2f}/{candidate['radix2_ns'] / 1000:<9.2f} "
            f"{baseline['direct_ns'] / 1000:>10.2f}/{baseline['radix2_ns'] / 1000:<9.2f} "
            f"{candidate['direct_ns'] / baseline['direct_ns']:>5.2f}x/"
            f"{candidate['radix2_ns'] / baseline['radix2_ns']:.2f}x"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--baseline-library", type=Path)
    parser.add_argument("--baseline-output", type=Path)
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
    if (arguments.baseline_library is None) != (arguments.baseline_output is None):
        parser.error("baseline-library and baseline-output must be supplied together")

    library = Library(arguments.library)
    profiles = ("f64", "f32") if arguments.profiles == "both" else (arguments.profiles,)
    baseline_results = None
    baseline_report = None
    if arguments.baseline_library is None:
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
        comparison_mode = "single"
    else:
        baseline_library = Library(arguments.baseline_library)
        results = []
        baseline_results = []
        for profile in profiles:
            for dimension in arguments.dimensions:
                candidate_result, baseline_result = benchmark_dimension_pair(
                    library,
                    baseline_library,
                    dimension,
                    profile,
                    arguments.repeats,
                    arguments.target_work,
                )
                results.append(candidate_result)
                baseline_results.append(baseline_result)
        comparison_mode = "interleaved"
        baseline_report = build_report(
            baseline_library,
            baseline_results,
            arguments.repeats,
            arguments.target_work,
            comparison_mode,
        )

    report = build_report(
        library,
        results,
        arguments.repeats,
        arguments.target_work,
        comparison_mode,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(encoded, encoding="utf-8")
    if baseline_report is not None:
        arguments.baseline_output.write_text(
            json.dumps(baseline_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if arguments.json:
        print(encoded, end="")
    elif baseline_results is not None:
        print_paired_results(results, baseline_results)
        print("Candidate and baseline repeats interleaved; AUTO policy unchanged.")
    else:
        print_results(results)
        print("AUTO policy unchanged; adopter-backed promotion evidence is still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

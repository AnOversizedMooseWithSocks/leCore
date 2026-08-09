#!/usr/bin/env python3
"""Measure liblecore's direct/radix regimes without changing AUTO policy.

The timed operation is a pre-resolved ctypes call to the public C ABI. Context
setup, NumPy validation, and Python adapter checks stay outside the timed region.
Results are descriptive evidence; this tool never rewrites dispatch thresholds.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from functools import partial
import gc
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Callable, Optional, Tuple
import uuid

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from bindings.python.lecore_native import Library  # noqa: E402


MINIMUM_OPTIMIZED_ITERATIONS = 4000
MAXIMUM_CALIBRATED_ITERATIONS = 1_000_000
CALIBRATION_HEADROOM = 1.25
CALIBRATION_PILOTS = 3
MAXIMUM_MEASUREMENT_ATTEMPTS = 3
MEASUREMENT_LAYER = "public-c-abi"
MEASUREMENT_MODE = "pre-resolved-bind"


@dataclass(frozen=True)
class TimingSamples:
    iterations: int
    elapsed_ns: list[int]

    @property
    def median_ns(self) -> float:
        return float(
            statistics.median(elapsed / self.iterations for elapsed in self.elapsed_ns)
        )


def timed_samples(
    operation: Callable[[], object], iterations: int, repeats: int
) -> TimingSamples:
    elapsed_samples = []
    operation()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            started = time.perf_counter_ns()
            for _ in range(iterations):
                operation()
            elapsed_samples.append(time.perf_counter_ns() - started)
    finally:
        if was_enabled:
            gc.enable()
    return TimingSamples(iterations=iterations, elapsed_ns=elapsed_samples)


def paired_timed_samples(
    candidate_operation: Callable[[], object],
    baseline_operation: Callable[[], object],
    iterations: int,
    repeats: int,
) -> Tuple[TimingSamples, TimingSamples]:
    """Capture aligned candidate/base samples while alternating first position."""

    samples: Tuple[list[int], list[int]] = ([], [])
    operations = (candidate_operation, baseline_operation)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for repeat in range(repeats):
            order = (0, 1) if repeat % 2 == 0 else (1, 0)
            for index in order:
                started = time.perf_counter_ns()
                for _ in range(iterations):
                    operations[index]()
                samples[index].append(time.perf_counter_ns() - started)
    finally:
        if was_enabled:
            gc.enable()
    return (
        TimingSamples(iterations=iterations, elapsed_ns=samples[0]),
        TimingSamples(iterations=iterations, elapsed_ns=samples[1]),
    )


def raw_bind_operation(
    library: Library,
    context: object,
    profile: str,
    left: np.ndarray,
    right: np.ndarray,
    output: np.ndarray,
) -> Tuple[Callable[[], int], str]:
    """Resolve the exported ABI function, context handle, and pointers once."""

    scalar = ctypes.c_double if profile == "f64" else ctypes.c_float
    pointer = ctypes.POINTER(scalar)
    name = f"lecore_hrr_bind_{profile}"
    function = getattr(library._dll, name)
    handle = context._pointer_locked()
    operation = partial(
        function,
        handle,
        left.ctypes.data_as(pointer),
        right.ctypes.data_as(pointer),
        output.ctypes.data_as(pointer),
    )
    return operation, name


def verify_operation(library: Library, operation: Callable[[], int], name: str) -> None:
    library._check(operation(), name)


def _measure_elapsed_ns(operation: Callable[[], object], iterations: int) -> int:
    started = time.perf_counter_ns()
    for _ in range(iterations):
        operation()
    return max(1, time.perf_counter_ns() - started)


def calibrated_iterations(
    operations: Tuple[Callable[[], object], ...],
    initial_iterations: int,
    sample_target_ns: int,
    measure_elapsed: Optional[Callable[[Callable[[], object], int], int]] = None,
) -> int:
    """Size from the fastest of several equal-count candidate/base pilots."""

    if not operations:
        raise ValueError("at least one operation is required for calibration")
    measure = _measure_elapsed_ns if measure_elapsed is None else measure_elapsed
    fastest_elapsed = None
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for pilot in range(CALIBRATION_PILOTS):
            order = range(len(operations))
            if pilot % 2:
                order = reversed(range(len(operations)))
            for index in order:
                elapsed = max(1, measure(operations[index], initial_iterations))
                if fastest_elapsed is None or elapsed < fastest_elapsed:
                    fastest_elapsed = elapsed
    finally:
        if was_enabled:
            gc.enable()
    if fastest_elapsed is None:
        raise RuntimeError("calibration did not produce a timing")
    scaled = math.ceil(
        initial_iterations
        * sample_target_ns
        * CALIBRATION_HEADROOM
        / fastest_elapsed
    )
    iterations = max(initial_iterations, scaled)
    return min(iterations, MAXIMUM_CALIBRATED_ITERATIONS)


def measure_with_escalation(
    measure: Callable[[int], Tuple[TimingSamples, ...]],
    initial_iterations: int,
    sample_target_ns: int,
) -> Tuple[TimingSamples, ...]:
    """Discard short batches and symmetrically increase work a bounded number of times."""

    iterations = initial_iterations
    shortest_elapsed = 0
    for _ in range(MAXIMUM_MEASUREMENT_ATTEMPTS):
        timing_groups = measure(iterations)
        if not timing_groups or any(
            timing.iterations != iterations or not timing.elapsed_ns
            for timing in timing_groups
        ):
            raise RuntimeError("measurement returned invalid timing groups")
        shortest_elapsed = min(
            elapsed
            for timing in timing_groups
            for elapsed in timing.elapsed_ns
        )
        if shortest_elapsed >= sample_target_ns:
            return timing_groups
        scaled = math.ceil(
            iterations
            * sample_target_ns
            * CALIBRATION_HEADROOM
            / max(1, shortest_elapsed)
        )
        next_iterations = min(
            max(iterations + 1, scaled), MAXIMUM_CALIBRATED_ITERATIONS
        )
        if next_iterations <= iterations:
            break
        iterations = next_iterations
    raise RuntimeError(
        "could not reach the sample-duration target after "
        f"{MAXIMUM_MEASUREMENT_ATTEMPTS} bounded measurement attempts; "
        f"shortest sample was {shortest_elapsed} ns"
    )


def descriptive_numpy_bind():
    """Load the application-level reference only for descriptive reports."""

    from holographic.agents_and_reasoning.holographic_ai import bind

    return bind


def setup_ns(
    library: Library, dimension: int, profile: str, backend: str, repeats: int
) -> float:
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
    timings: dict[str, TimingSamples],
    outputs: dict[str, np.ndarray],
    scratch: dict[str, int],
    setup_timings: Optional[dict[str, float]],
    auto_backend: int,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "dimension": dimension,
        "profile": profile,
        "auto_backend": auto_backend,
        "direct_iterations": timings["direct"].iterations,
        "optimized_iterations": timings["radix2"].iterations,
        "direct_elapsed_ns": timings["direct"].elapsed_ns,
        "radix2_elapsed_ns": timings["radix2"].elapsed_ns,
        "direct_ns": timings["direct"].median_ns,
        "radix2_ns": timings["radix2"].median_ns,
        "radix2_speedup": timings["direct"].median_ns
        / timings["radix2"].median_ns,
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
    }
    if setup_timings is not None:
        entry.update(
            {
                "direct_setup_ns": setup_timings["direct"],
                "radix2_setup_ns": setup_timings["radix2"],
            }
        )
    return entry


def benchmark_dimension(
    library: Library,
    dimension: int,
    profile: str,
    repeats: int,
    target_work: int,
    sample_target_ns: int,
    descriptive_metrics: bool,
) -> dict[str, object]:
    dtype = np.float64 if profile == "f64" else np.float32
    left, right = dimension_inputs(dimension, profile)

    direct_iterations = max(1, min(2000, target_work // (dimension * dimension)))
    optimized_iterations = max(MINIMUM_OPTIMIZED_ITERATIONS, direct_iterations)
    outputs: dict[str, np.ndarray] = {}
    timings: dict[str, TimingSamples] = {}
    scratch: dict[str, int] = {}

    for backend, iterations in (
        ("direct", direct_iterations),
        ("radix2", optimized_iterations),
    ):
        with library.context(dimension, profile=profile, backend=backend) as context:
            output = np.empty(dimension, dtype=dtype)

            operation, operation_name = raw_bind_operation(
                library, context, profile, left, right, output
            )
            verify_operation(library, operation, operation_name)
            iterations = calibrated_iterations(
                (operation,), iterations, sample_target_ns
            )
            timing_groups = measure_with_escalation(
                lambda count: (timed_samples(operation, count, repeats),),
                iterations,
                sample_target_ns,
            )
            timings[backend] = timing_groups[0]
            verify_operation(library, operation, operation_name)
            outputs[backend] = output.copy()
            scratch[backend] = context.scratch_bytes

    with library.context(dimension, profile=profile, backend="auto") as context:
        auto_backend = context.backend

    setup_timings = None
    if descriptive_metrics:
        setup_timings = {
            backend: setup_ns(library, dimension, profile, backend, repeats)
            for backend in ("direct", "radix2")
        }
    entry = result_entry(
        dimension,
        profile,
        timings,
        outputs,
        scratch,
        setup_timings,
        auto_backend,
    )

    if descriptive_metrics and profile == "f64":
        numpy_bind = descriptive_numpy_bind()
        numpy_output = np.empty(dimension, dtype=np.float64)

        def call_numpy() -> None:
            nonlocal numpy_output
            numpy_output = numpy_bind(left, right)

        numpy_time = timed_samples(
            call_numpy, timings["radix2"].iterations, repeats
        ).median_ns
        entry.update(
            {
                "numpy_fft_ns": numpy_time,
                "radix2_speedup_vs_numpy": numpy_time
                / timings["radix2"].median_ns,
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
    sample_target_ns: int,
    descriptive_metrics: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    """Benchmark candidate and baseline with identical work and alternating order."""

    dtype = np.float64 if profile == "f64" else np.float32
    left, right = dimension_inputs(dimension, profile)
    direct_iterations = max(1, min(2000, target_work // (dimension * dimension)))
    optimized_iterations = max(MINIMUM_OPTIMIZED_ITERATIONS, direct_iterations)
    libraries = (candidate_library, baseline_library)
    timings: tuple[dict[str, TimingSamples], dict[str, TimingSamples]] = ({}, {})
    outputs: tuple[dict[str, np.ndarray], dict[str, np.ndarray]] = ({}, {})
    scratch: tuple[dict[str, int], dict[str, int]] = ({}, {})
    setup_timings: tuple[dict[str, float], dict[str, float]] = ({}, {})

    for backend, iterations in (
        ("direct", direct_iterations),
        ("radix2", optimized_iterations),
    ):
        with candidate_library.context(
            dimension, profile=profile, backend=backend
        ) as candidate_context:
            with baseline_library.context(
                dimension, profile=profile, backend=backend
            ) as baseline_context:
                candidate_output = np.empty(dimension, dtype=dtype)
                baseline_output = np.empty(dimension, dtype=dtype)

                candidate_operation, candidate_name = raw_bind_operation(
                    candidate_library,
                    candidate_context,
                    profile,
                    left,
                    right,
                    candidate_output,
                )
                baseline_operation, baseline_name = raw_bind_operation(
                    baseline_library,
                    baseline_context,
                    profile,
                    left,
                    right,
                    baseline_output,
                )
                verify_operation(candidate_library, candidate_operation, candidate_name)
                verify_operation(baseline_library, baseline_operation, baseline_name)
                iterations = calibrated_iterations(
                    (candidate_operation, baseline_operation),
                    iterations,
                    sample_target_ns,
                )
                candidate_samples, baseline_samples = measure_with_escalation(
                    lambda count: paired_timed_samples(
                        candidate_operation,
                        baseline_operation,
                        count,
                        repeats,
                    ),
                    iterations,
                    sample_target_ns,
                )
                verify_operation(candidate_library, candidate_operation, candidate_name)
                verify_operation(baseline_library, baseline_operation, baseline_name)
                timings[0][backend] = candidate_samples
                timings[1][backend] = baseline_samples
                outputs[0][backend] = candidate_output.copy()
                outputs[1][backend] = baseline_output.copy()
                scratch[0][backend] = candidate_context.scratch_bytes
                scratch[1][backend] = baseline_context.scratch_bytes

        if descriptive_metrics:
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
            timings[index],
            outputs[index],
            scratch[index],
            setup_timings[index] if descriptive_metrics else None,
            auto_backends[index],
        )
        for index in range(2)
    )

    if descriptive_metrics and profile == "f64":
        numpy_bind = descriptive_numpy_bind()
        numpy_output = np.empty(dimension, dtype=np.float64)

        def call_numpy() -> None:
            nonlocal numpy_output
            numpy_output = numpy_bind(left, right)

        numpy_time = timed_samples(
            call_numpy, timings[0]["radix2"].iterations, repeats
        ).median_ns
        for index, entry in enumerate(entries):
            entry.update(
                {
                    "numpy_fft_ns": numpy_time,
                    "radix2_speedup_vs_numpy": numpy_time
                    / timings[index]["radix2"].median_ns,
                    "numpy_max_abs_vs_direct": float(
                        np.max(np.abs(numpy_output - outputs[index]["direct"]))
                    ),
                }
            )
    return entries


def parse_dimensions(value: str) -> list[int]:
    dimensions = [int(item) for item in value.split(",") if item.strip()]
    if not dimensions or any(
        dimension <= 0 or dimension & (dimension - 1) for dimension in dimensions
    ):
        raise argparse.ArgumentTypeError(
            "dimensions must be a comma-separated list of positive powers of two"
        )
    return dimensions


def build_report(
    library: Library,
    results: list[dict[str, object]],
    repeats: int,
    target_work: int,
    sample_target_ns: int,
    comparison_mode: str,
    metrics_scope: str,
    pair_id: Optional[str] = None,
) -> dict[str, object]:
    benchmark: dict[str, object] = {
        "repeats": repeats,
        "target_work": target_work,
        "sample_target_ns": sample_target_ns,
        "calibration_pilots": CALIBRATION_PILOTS,
        "maximum_calibrated_iterations": MAXIMUM_CALIBRATED_ITERATIONS,
        "maximum_measurement_attempts": MAXIMUM_MEASUREMENT_ATTEMPTS,
        "minimum_optimized_iterations": MINIMUM_OPTIMIZED_ITERATIONS,
        "comparison_mode": comparison_mode,
        "measurement_layer": MEASUREMENT_LAYER,
        "measurement_mode": MEASUREMENT_MODE,
        "metrics_scope": metrics_scope,
    }
    if pair_id is not None:
        benchmark["pair_id"] = pair_id
    return {
        "schema_version": 2,
        "policy_effect": "none; AUTO remains unchanged",
        "benchmark": benchmark,
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
        direct_ratios = [
            candidate_elapsed / baseline_elapsed
            for candidate_elapsed, baseline_elapsed in zip(
                candidate["direct_elapsed_ns"], baseline["direct_elapsed_ns"]
            )
        ]
        radix2_ratios = [
            candidate_elapsed / baseline_elapsed
            for candidate_elapsed, baseline_elapsed in zip(
                candidate["radix2_elapsed_ns"], baseline["radix2_elapsed_ns"]
            )
        ]
        print(
            f"{candidate['profile']:>7} {candidate['dimension']:>5} "
            f"{candidate['direct_ns'] / 1000:>10.2f}/{candidate['radix2_ns'] / 1000:<9.2f} "
            f"{baseline['direct_ns'] / 1000:>10.2f}/{baseline['radix2_ns'] / 1000:<9.2f} "
            f"{statistics.median(direct_ratios):>5.2f}x/"
            f"{statistics.median(radix2_ratios):.2f}x"
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
    parser.add_argument(
        "--sample-target-ms",
        type=float,
        default=30.0,
        help="calibrate each backend to at least this approximate sample duration",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="omit descriptive setup and NumPy metrics from a CI-gate report",
    )
    arguments = parser.parse_args()
    if (
        arguments.repeats <= 0
        or arguments.target_work <= 0
        or not math.isfinite(arguments.sample_target_ms)
        or arguments.sample_target_ms <= 0
    ):
        parser.error("repeats, target-work, and sample-target-ms must be positive")
    if (arguments.baseline_library is None) != (arguments.baseline_output is None):
        parser.error("baseline-library and baseline-output must be supplied together")

    library = Library(arguments.library)
    profiles = ("f64", "f32") if arguments.profiles == "both" else (arguments.profiles,)
    baseline_results = None
    baseline_report = None
    descriptive_metrics = not arguments.gate
    metrics_scope = "descriptive" if descriptive_metrics else "gate"
    sample_target_ns = int(arguments.sample_target_ms * 1_000_000)
    pair_id = None
    if arguments.baseline_library is None:
        results = [
            benchmark_dimension(
                library,
                dimension,
                profile,
                arguments.repeats,
                arguments.target_work,
                sample_target_ns,
                descriptive_metrics,
            )
            for profile in profiles
            for dimension in arguments.dimensions
        ]
        comparison_mode = "single"
    else:
        baseline_library = Library(arguments.baseline_library)
        pair_id = uuid.uuid4().hex
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
                    sample_target_ns,
                    descriptive_metrics,
                )
                results.append(candidate_result)
                baseline_results.append(baseline_result)
        comparison_mode = "paired-interleaved"
        baseline_report = build_report(
            baseline_library,
            baseline_results,
            arguments.repeats,
            arguments.target_work,
            sample_target_ns,
            comparison_mode,
            metrics_scope,
            pair_id,
        )

    report = build_report(
        library,
        results,
        arguments.repeats,
        arguments.target_work,
        sample_target_ns,
        comparison_mode,
        metrics_scope,
        pair_id,
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

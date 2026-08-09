#!/usr/bin/env python3
"""Enforce liblecore's portable performance-regression CI policy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class GateInputError(ValueError):
    """The report or policy is malformed and cannot be evaluated safely."""


@dataclass(frozen=True)
class Measurement:
    profile: str
    dimension: int
    direct_ns: float
    radix2_ns: float
    speedup: float
    minimum_speedup: Optional[float]
    max_abs_delta: float
    delta_limit: float
    max_candidate_slowdown: float
    baseline_required: bool = False
    baseline_direct_ns: Optional[float] = None
    baseline_radix2_ns: Optional[float] = None
    direct_slowdown: Optional[float] = None
    radix2_slowdown: Optional[float] = None

    @property
    def passed(self) -> bool:
        return (
            (self.minimum_speedup is None or self.speedup >= self.minimum_speedup)
            and self.max_abs_delta <= self.delta_limit
            and (
                not self.baseline_required
                or (
                    self.baseline_direct_ns is not None
                    and self.baseline_radix2_ns is not None
                )
            )
            and (
                self.direct_slowdown is None
                or self.direct_slowdown <= self.max_candidate_slowdown
            )
            and (
                self.radix2_slowdown is None
                or self.radix2_slowdown <= self.max_candidate_slowdown
            )
        )



@dataclass(frozen=True)
class ResultTimings:
    direct_iterations: int
    radix2_iterations: int
    direct_elapsed_ns: Sequence[int]
    radix2_elapsed_ns: Sequence[int]
    direct_ns: float
    radix2_ns: float
    speedup: float
    max_abs_delta: float


@dataclass(frozen=True)
class Evaluation:
    measurements: Sequence[Measurement]
    failures: Sequence[str]
    markdown: str
    baseline_compared: bool

    @property
    def passed(self) -> bool:
        return not self.failures


def _load_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateInputError(f"could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise GateInputError(f"{label} root must be a JSON object")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise GateInputError(f"{label} must be an object")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GateInputError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateInputError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive and " if positive else ""
        raise GateInputError(f"{label} must be {qualifier}finite")
    return number


def _policy_regimes(
    policy: Mapping[str, Any],
) -> Tuple[
    Mapping[str, Any],
    float,
    List[Tuple[str, int, Optional[float], float]],
]:
    if _integer(policy.get("schema_version"), "policy.schema_version") != 2:
        raise GateInputError("unsupported performance policy schema_version")
    required_library = _mapping(policy.get("required_library"), "policy.required_library")
    max_candidate_slowdown = _finite(
        policy.get("max_candidate_slowdown"),
        "policy.max_candidate_slowdown",
        positive=True,
    )
    if max_candidate_slowdown < 1.0:
        raise GateInputError("policy.max_candidate_slowdown must be >= 1")
    raw_dimensions = policy.get("dimensions")
    if not isinstance(raw_dimensions, list) or not raw_dimensions:
        raise GateInputError("policy.dimensions must be a non-empty array")
    dimensions = []
    for index, raw_dimension in enumerate(raw_dimensions):
        dimension = _integer(raw_dimension, f"policy.dimensions[{index}]", minimum=1)
        if dimension & (dimension - 1):
            raise GateInputError(f"policy dimension {dimension} must be a power of two")
        if dimension in dimensions:
            raise GateInputError(f"duplicate policy dimension {dimension}")
        dimensions.append(dimension)

    profiles = _mapping(policy.get("profiles"), "policy.profiles")
    regimes: List[Tuple[str, int, Optional[float], float]] = []
    if not profiles:
        raise GateInputError("policy.profiles must not be empty")
    for profile, raw_profile_policy in profiles.items():
        if not isinstance(profile, str) or not profile:
            raise GateInputError("policy profile names must be non-empty strings")
        profile_policy = _mapping(raw_profile_policy, f"policy.profiles.{profile}")
        delta_limit = _finite(
            profile_policy.get("max_abs_delta"),
            f"policy.profiles.{profile}.max_abs_delta",
        )
        if delta_limit < 0.0:
            raise GateInputError(f"policy.profiles.{profile}.max_abs_delta must be >= 0")
        raw_minimums = profile_policy.get("minimum_speedup", {})
        minimums = _mapping(
            raw_minimums, f"policy.profiles.{profile}.minimum_speedup"
        )
        unknown_dimensions = set(minimums) - {str(item) for item in dimensions}
        if unknown_dimensions:
            unknown = sorted(unknown_dimensions)[0]
            raise GateInputError(
                f"policy.profiles.{profile}.minimum_speedup has unguarded dimension {unknown}"
            )
        for dimension in dimensions:
            raw_minimum = minimums.get(str(dimension))
            minimum = None
            if raw_minimum is not None:
                minimum = _finite(
                    raw_minimum,
                    f"policy.profiles.{profile}.minimum_speedup.{dimension}",
                    positive=True,
                )
            regimes.append((profile, dimension, minimum, delta_limit))
    regimes.sort(key=lambda item: (item[0], item[1]))
    return required_library, max_candidate_slowdown, regimes


def _validate_metadata(
    report: Mapping[str, Any],
    policy: Mapping[str, Any],
    required_library: Mapping[str, Any],
    label: str,
) -> int:
    expected_report_schema = _integer(
        policy.get("report_schema_version"), "policy.report_schema_version"
    )
    if (
        _integer(report.get("schema_version"), f"{label}.schema_version")
        != expected_report_schema
    ):
        raise GateInputError(
            f"{label}.schema_version must equal policy report schema {expected_report_schema}"
        )
    expected_effect = policy.get("required_policy_effect")
    if not isinstance(expected_effect, str) or not expected_effect:
        raise GateInputError("policy.required_policy_effect must be a non-empty string")
    if report.get("policy_effect") != expected_effect:
        raise GateInputError(
            f"{label}.policy_effect must be {expected_effect!r}; AUTO policy may have changed"
        )
    library = _mapping(report.get("library"), f"{label}.library")
    for field in ("abi", "isa"):
        expected = _integer(required_library.get(field), f"policy.required_library.{field}")
        actual = _integer(library.get(field), f"{label}.library.{field}")
        if actual != expected:
            raise GateInputError(f"{label}.library.{field} is {actual}, expected {expected}")
    mask = _integer(
        required_library.get("capabilities_mask"),
        "policy.required_library.capabilities_mask",
    )
    capabilities = _integer(library.get("capabilities"), f"{label}.library.capabilities")
    if capabilities & mask != mask:
        raise GateInputError(
            f"{label}.library.capabilities 0x{capabilities:x} lacks required mask 0x{mask:x}"
        )

    benchmark = _mapping(report.get("benchmark"), f"{label}.benchmark")
    for policy_field, report_field in (
        ("required_measurement_layer", "measurement_layer"),
        ("required_measurement_mode", "measurement_mode"),
        ("required_metrics_scope", "metrics_scope"),
    ):
        expected = policy.get(policy_field)
        if not isinstance(expected, str) or not expected:
            raise GateInputError(f"policy.{policy_field} must be a non-empty string")
        actual = benchmark.get(report_field)
        if actual != expected:
            raise GateInputError(
                f"{label}.benchmark.{report_field} is {actual!r}, expected {expected!r}"
            )
    minimum_repeats = _integer(
        policy.get("minimum_repeats"), "policy.minimum_repeats", minimum=1
    )
    repeats = _integer(benchmark.get("repeats"), f"{label}.benchmark.repeats", minimum=1)
    if repeats < minimum_repeats:
        raise GateInputError(
            f"{label}.benchmark.repeats is {repeats}, below required {minimum_repeats}"
        )
    minimum_sample_elapsed_ns = _integer(
        policy.get("minimum_sample_elapsed_ns"),
        "policy.minimum_sample_elapsed_ns",
        minimum=1,
    )
    sample_target_ns = _integer(
        benchmark.get("sample_target_ns"),
        f"{label}.benchmark.sample_target_ns",
        minimum=1,
    )
    if sample_target_ns < minimum_sample_elapsed_ns:
        raise GateInputError(
            f"{label}.benchmark.sample_target_ns is {sample_target_ns}, below "
            f"the policy sample floor {minimum_sample_elapsed_ns}"
        )
    minimum_calibration_pilots = _integer(
        policy.get("minimum_calibration_pilots"),
        "policy.minimum_calibration_pilots",
        minimum=1,
    )
    calibration_pilots = _integer(
        benchmark.get("calibration_pilots"),
        f"{label}.benchmark.calibration_pilots",
        minimum=1,
    )
    if calibration_pilots < minimum_calibration_pilots:
        raise GateInputError(
            f"{label}.benchmark.calibration_pilots is {calibration_pilots}, below "
            f"required {minimum_calibration_pilots}"
        )
    return repeats


def _index_results(
    report: Mapping[str, Any], label: str
) -> Dict[Tuple[str, int], Mapping[str, Any]]:
    raw_results = report.get("results")
    if not isinstance(raw_results, list):
        raise GateInputError(f"{label}.results must be an array")

    indexed: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for index, raw_result in enumerate(raw_results):
        result = _mapping(raw_result, f"{label}.results[{index}]")
        profile = result.get("profile")
        if not isinstance(profile, str) or not profile:
            raise GateInputError(
                f"{label}.results[{index}].profile must be a non-empty string"
            )
        dimension = _integer(
            result.get("dimension"), f"{label}.results[{index}].dimension", minimum=1
        )
        key = (profile, dimension)
        if key in indexed:
            raise GateInputError(
                f"duplicate {label} result for {profile} dimension {dimension}"
            )
        indexed[key] = result
    return indexed


def _comparison_mode(report: Mapping[str, Any], label: str) -> str:
    benchmark = _mapping(report.get("benchmark"), f"{label}.benchmark")
    mode = benchmark.get("comparison_mode")
    if not isinstance(mode, str) or not mode:
        raise GateInputError(f"{label}.benchmark.comparison_mode must be a non-empty string")
    return mode


def _pair_id(report: Mapping[str, Any], label: str) -> str:
    benchmark = _mapping(report.get("benchmark"), f"{label}.benchmark")
    pair_id = benchmark.get("pair_id")
    if not isinstance(pair_id, str) or not pair_id:
        raise GateInputError(f"{label}.benchmark.pair_id must be a non-empty string")
    return pair_id


def _elapsed_samples(
    value: Any,
    label: str,
    repeats: int,
    minimum_sample_elapsed_ns: int,
) -> List[int]:
    if not isinstance(value, list):
        raise GateInputError(f"{label} must be an array")
    if len(value) != repeats:
        raise GateInputError(
            f"{label} has {len(value)} samples, expected benchmark.repeats={repeats}"
        )
    samples = []
    for index, raw_elapsed in enumerate(value):
        elapsed = _integer(raw_elapsed, f"{label}[{index}]", minimum=1)
        if elapsed < minimum_sample_elapsed_ns:
            raise GateInputError(
                f"{label}[{index}] is {elapsed} ns, below required sample duration "
                f"{minimum_sample_elapsed_ns} ns"
            )
        samples.append(elapsed)
    return samples


def _read_result(
    result: Mapping[str, Any],
    label: str,
    required_auto_backend: int,
    minimum_optimized_iterations: int,
    repeats: int,
    minimum_sample_elapsed_ns: int,
) -> ResultTimings:
    actual_auto_backend = _integer(result.get("auto_backend"), f"{label}.auto_backend")
    if actual_auto_backend != required_auto_backend:
        raise GateInputError(
            f"{label}.auto_backend is {actual_auto_backend}, expected "
            f"{required_auto_backend}; AUTO dispatch changed"
        )
    direct_iterations = _integer(
        result.get("direct_iterations"), f"{label}.direct_iterations", minimum=1
    )
    optimized_iterations = _integer(
        result.get("optimized_iterations"), f"{label}.optimized_iterations", minimum=1
    )
    if optimized_iterations < minimum_optimized_iterations:
        raise GateInputError(
            f"{label}.optimized_iterations is {optimized_iterations}, below required "
            f"{minimum_optimized_iterations}"
        )

    direct_elapsed_ns = _elapsed_samples(
        result.get("direct_elapsed_ns"),
        f"{label}.direct_elapsed_ns",
        repeats,
        minimum_sample_elapsed_ns,
    )
    radix2_elapsed_ns = _elapsed_samples(
        result.get("radix2_elapsed_ns"),
        f"{label}.radix2_elapsed_ns",
        repeats,
        minimum_sample_elapsed_ns,
    )
    direct_ns = float(
        statistics.median(elapsed / direct_iterations for elapsed in direct_elapsed_ns)
    )
    radix2_ns = float(
        statistics.median(elapsed / optimized_iterations for elapsed in radix2_elapsed_ns)
    )
    reported_direct_ns = _finite(
        result.get("direct_ns"), f"{label}.direct_ns", positive=True
    )
    reported_radix2_ns = _finite(
        result.get("radix2_ns"), f"{label}.radix2_ns", positive=True
    )
    for field, reported, calculated in (
        ("direct_ns", reported_direct_ns, direct_ns),
        ("radix2_ns", reported_radix2_ns, radix2_ns),
    ):
        if not math.isclose(reported, calculated, rel_tol=1e-9, abs_tol=1e-6):
            raise GateInputError(
                f"{label}.{field} is inconsistent with its elapsed samples"
            )
    reported_speedup = _finite(
        result.get("radix2_speedup"), f"{label}.radix2_speedup", positive=True
    )
    speedup = direct_ns / radix2_ns
    if not math.isclose(reported_speedup, speedup, rel_tol=1e-9, abs_tol=1e-12):
        raise GateInputError(
            f"{label}.radix2_speedup is inconsistent with direct_ns / radix2_ns"
        )
    delta = _finite(
        result.get("radix2_max_abs_vs_direct"),
        f"{label}.radix2_max_abs_vs_direct",
    )
    if delta < 0.0:
        raise GateInputError(f"{label}.radix2_max_abs_vs_direct must be >= 0")
    return ResultTimings(
        direct_iterations=direct_iterations,
        radix2_iterations=optimized_iterations,
        direct_elapsed_ns=direct_elapsed_ns,
        radix2_elapsed_ns=radix2_elapsed_ns,
        direct_ns=direct_ns,
        radix2_ns=radix2_ns,
        speedup=speedup,
        max_abs_delta=delta,
    )


def evaluate(
    report: Mapping[str, Any],
    policy: Mapping[str, Any],
    baseline: Optional[Mapping[str, Any]] = None,
) -> Evaluation:
    required_library, max_candidate_slowdown, regimes = _policy_regimes(policy)
    repeats = _validate_metadata(report, policy, required_library, "report")
    indexed = _index_results(report, "report")

    baseline_indexed: Optional[Dict[Tuple[str, int], Mapping[str, Any]]] = None
    if baseline is not None:
        baseline_repeats = _validate_metadata(
            baseline, policy, required_library, "baseline"
        )
        if baseline_repeats != repeats:
            raise GateInputError(
                "report and baseline benchmark.repeats must match for paired samples"
            )
        baseline_indexed = _index_results(baseline, "baseline")
        expected_mode = policy.get("required_baseline_comparison_mode")
        if not isinstance(expected_mode, str) or not expected_mode:
            raise GateInputError(
                "policy.required_baseline_comparison_mode must be a non-empty string"
            )
        for compared_report, label in ((report, "report"), (baseline, "baseline")):
            actual_mode = _comparison_mode(compared_report, label)
            if actual_mode != expected_mode:
                raise GateInputError(
                    f"{label}.benchmark.comparison_mode is {actual_mode!r}, "
                    f"expected {expected_mode!r}"
                )
        report_pair_id = _pair_id(report, "report")
        baseline_pair_id = _pair_id(baseline, "baseline")
        if report_pair_id != baseline_pair_id:
            raise GateInputError(
                "report and baseline benchmark.pair_id must identify the same paired run"
            )
    else:
        expected_mode = policy.get("required_bootstrap_comparison_mode")
        if not isinstance(expected_mode, str) or not expected_mode:
            raise GateInputError(
                "policy.required_bootstrap_comparison_mode must be a non-empty string"
            )
        actual_mode = _comparison_mode(report, "report")
        if actual_mode != expected_mode:
            raise GateInputError(
                f"report.benchmark.comparison_mode is {actual_mode!r}, "
                f"expected {expected_mode!r}"
            )

    required_auto_backend = _integer(
        required_library.get("auto_backend"), "policy.required_library.auto_backend"
    )
    minimum_optimized_iterations = _integer(
        policy.get("minimum_optimized_iterations"),
        "policy.minimum_optimized_iterations",
        minimum=1,
    )
    minimum_sample_elapsed_ns = _integer(
        policy.get("minimum_sample_elapsed_ns"),
        "policy.minimum_sample_elapsed_ns",
        minimum=1,
    )

    measurements: List[Measurement] = []
    failures: List[str] = []
    for profile, dimension, minimum, delta_limit in regimes:
        key = (profile, dimension)
        if key not in indexed:
            failures.append(f"missing result for {profile} dimension {dimension}")
            continue
        result = indexed[key]
        label = f"report result {profile}/{dimension}"
        candidate_timing = _read_result(
            result,
            label,
            required_auto_backend,
            minimum_optimized_iterations,
            repeats,
            minimum_sample_elapsed_ns,
        )

        baseline_direct_ns: Optional[float] = None
        baseline_radix2_ns: Optional[float] = None
        direct_slowdown: Optional[float] = None
        radix2_slowdown: Optional[float] = None
        if baseline_indexed is not None:
            if key not in baseline_indexed:
                failures.append(
                    f"missing baseline result for {profile} dimension {dimension}"
                )
            else:
                baseline_timing = _read_result(
                    baseline_indexed[key],
                    f"baseline result {profile}/{dimension}",
                    required_auto_backend,
                    minimum_optimized_iterations,
                    repeats,
                    minimum_sample_elapsed_ns,
                )
                if (
                    candidate_timing.direct_iterations
                    != baseline_timing.direct_iterations
                    or candidate_timing.radix2_iterations
                    != baseline_timing.radix2_iterations
                ):
                    raise GateInputError(
                        f"{profile}/{dimension} candidate and baseline iteration "
                        "counts must match for paired samples"
                    )
                baseline_direct_ns = baseline_timing.direct_ns
                baseline_radix2_ns = baseline_timing.radix2_ns
                direct_slowdown = float(
                    statistics.median(
                        candidate_elapsed / baseline_elapsed
                        for candidate_elapsed, baseline_elapsed in zip(
                            candidate_timing.direct_elapsed_ns,
                            baseline_timing.direct_elapsed_ns,
                        )
                    )
                )
                radix2_slowdown = float(
                    statistics.median(
                        candidate_elapsed / baseline_elapsed
                        for candidate_elapsed, baseline_elapsed in zip(
                            candidate_timing.radix2_elapsed_ns,
                            baseline_timing.radix2_elapsed_ns,
                        )
                    )
                )

        measurement = Measurement(
            profile=profile,
            dimension=dimension,
            direct_ns=candidate_timing.direct_ns,
            radix2_ns=candidate_timing.radix2_ns,
            speedup=candidate_timing.speedup,
            minimum_speedup=minimum,
            max_abs_delta=candidate_timing.max_abs_delta,
            delta_limit=delta_limit,
            max_candidate_slowdown=max_candidate_slowdown,
            baseline_required=baseline is not None,
            baseline_direct_ns=baseline_direct_ns,
            baseline_radix2_ns=baseline_radix2_ns,
            direct_slowdown=direct_slowdown,
            radix2_slowdown=radix2_slowdown,
        )
        measurements.append(measurement)
        if minimum is not None and candidate_timing.speedup < minimum:
            failures.append(
                f"{profile}/{dimension} radix-2 speedup "
                f"{candidate_timing.speedup:.2f}x is below {minimum:.2f}x"
            )
        if candidate_timing.max_abs_delta > delta_limit:
            failures.append(
                f"{profile}/{dimension} max delta "
                f"{candidate_timing.max_abs_delta:.3e} exceeds {delta_limit:.3e}"
            )
        if (
            measurement.direct_slowdown is not None
            and measurement.direct_slowdown > max_candidate_slowdown
        ):
            failures.append(
                f"{profile}/{dimension} direct candidate/base slowdown "
                f"{measurement.direct_slowdown:.2f}x exceeds {max_candidate_slowdown:.2f}x"
            )
        if (
            measurement.radix2_slowdown is not None
            and measurement.radix2_slowdown > max_candidate_slowdown
        ):
            failures.append(
                f"{profile}/{dimension} radix-2 candidate/base slowdown "
                f"{measurement.radix2_slowdown:.2f}x exceeds {max_candidate_slowdown:.2f}x"
            )

    markdown = render_markdown(
        report,
        measurements,
        failures,
        baseline_compared=baseline is not None,
        max_candidate_slowdown=max_candidate_slowdown,
    )
    return Evaluation(
        measurements=measurements,
        failures=failures,
        markdown=markdown,
        baseline_compared=baseline is not None,
    )


def render_markdown(
    report: Mapping[str, Any],
    measurements: Iterable[Measurement],
    failures: Sequence[str],
    *,
    baseline_compared: bool,
    max_candidate_slowdown: float,
) -> str:
    host = report.get("host") if isinstance(report.get("host"), dict) else {}
    library = report.get("library") if isinstance(report.get("library"), dict) else {}
    status = "PASS" if not failures else "FAIL"
    comparison = (
        f"Candidate and base samples are paired and interleaved on this runner; "
        f"the median paired slowdown may be at most {max_candidate_slowdown:.2f}x "
        "per backend."
        if baseline_compared
        else "Bootstrap mode: the base commit has no benchmarkable liblecore, so "
        "only same-build invariants apply."
    )
    lines = [
        "## liblecore performance regression",
        "",
        f"**Gate: {status}.** {comparison}",
        "",
        f"Host: `{host.get('platform', 'unknown')}` / `{host.get('machine', 'unknown')}`; "
        f"Python `{host.get('python', 'unknown')}`; NumPy `{host.get('numpy', 'unknown')}`; "
        f"liblecore `{library.get('version', 'unknown')}` (ABI `{library.get('abi', 'unknown')}`, "
        f"ISA `{library.get('isa', 'unknown')}`).",
        "",
        "Layer: public C ABI through a fixed pre-resolved ctypes transition, with the "
        "bind function, context, and array pointers resolved before timing. "
        "Every latency cell contains raw per-repeat elapsed times that satisfy the "
        "policy duration floor.",
        "",
        "| Profile | Dimension | Direct (us) | Radix-2 (us) | Speedup | Floor | Median paired direct/base | Median paired radix/base | Max delta | Limit | Result |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for item in measurements:
        direct_base = "--" if item.direct_slowdown is None else f"{item.direct_slowdown:.2f}x"
        radix_base = "--" if item.radix2_slowdown is None else f"{item.radix2_slowdown:.2f}x"
        speedup_floor = (
            "--"
            if item.minimum_speedup is None
            else f"{item.minimum_speedup:.2f}x"
        )
        lines.append(
            f"| {item.profile} | {item.dimension} | {item.direct_ns / 1000.0:.2f} | "
            f"{item.radix2_ns / 1000.0:.2f} | {item.speedup:.2f}x | "
            f"{speedup_floor} | {direct_base} | {radix_base} | "
            f"{item.max_abs_delta:.3e} | "
            f"{item.delta_limit:.3e} | {'PASS' if item.passed else 'FAIL'} |"
        )
    if failures:
        lines.extend(["", "### Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
    lines.extend(
        [
            "",
            "`LECORE_BACKEND_AUTO` remains direct; this gate does not change dispatch policy.",
            "",
        ]
    )
    return "\n".join(lines)


def _input_failure_markdown(error: Exception) -> str:
    return "\n".join(
        [
            "## liblecore performance regression",
            "",
            "**Gate: FAIL.** The benchmark report or policy was not valid.",
            "",
            f"- {error}",
            "",
        ]
    )


def _write_summary(path: Path, markdown: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as destination:
        destination.write(markdown)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    arguments = parser.parse_args(argv)

    try:
        report = _load_object(arguments.report, "report")
        baseline = (
            _load_object(arguments.baseline, "baseline")
            if arguments.baseline is not None
            else None
        )
        policy = _load_object(arguments.policy, "policy")
        evaluation = evaluate(report, policy, baseline)
        markdown = evaluation.markdown
        failures = list(evaluation.failures)
    except GateInputError as error:
        markdown = _input_failure_markdown(error)
        failures = [str(error)]

    print(markdown, end="")
    if arguments.summary is not None:
        _write_summary(arguments.summary, markdown)
    if failures:
        for failure in failures:
            print(f"performance gate: {failure}", file=sys.stderr)
        return 1
    print("performance gate: all guarded regimes passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Regression tests for the liblecore performance-policy checker."""

from __future__ import annotations

import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.bench_liblecore import (  # noqa: E402
    MAXIMUM_CALIBRATED_ITERATIONS,
    MAXIMUM_MEASUREMENT_ATTEMPTS,
    TimingSamples,
    calibrated_iterations,
    measure_with_escalation,
)
from benchmarks.check_liblecore_performance import GateInputError, evaluate, main  # noqa: E402


def policy():
    return {
        "schema_version": 2,
        "report_schema_version": 2,
        "required_policy_effect": "none; AUTO remains unchanged",
        "required_measurement_layer": "public-c-abi",
        "required_measurement_mode": "pre-resolved-bind",
        "required_metrics_scope": "gate",
        "required_baseline_comparison_mode": "paired-interleaved",
        "required_bootstrap_comparison_mode": "single",
        "minimum_repeats": 10,
        "minimum_calibration_pilots": 3,
        "minimum_optimized_iterations": 4000,
        "minimum_sample_elapsed_ns": 10_000_000,
        "max_candidate_slowdown": 1.35,
        "dimensions": [256],
        "required_library": {
            "abi": 0,
            "isa": 1,
            "capabilities_mask": 15,
            "auto_backend": 1,
        },
        "profiles": {
            "f64": {"max_abs_delta": 1e-12, "minimum_speedup": {"256": 2.0}},
            "f32": {"max_abs_delta": 1e-4, "minimum_speedup": {"256": 2.0}},
        },
    }


def _set_samples(entry, backend, per_call_ns):
    iterations_field = (
        "direct_iterations" if backend == "direct" else "optimized_iterations"
    )
    elapsed_field = f"{backend}_elapsed_ns"
    latency_field = f"{backend}_ns"
    iterations = entry[iterations_field]
    elapsed = [int(round(value * iterations)) for value in per_call_ns]
    entry[elapsed_field] = elapsed
    entry[latency_field] = float(
        statistics.median(value / iterations for value in elapsed)
    )
    entry["radix2_speedup"] = entry["direct_ns"] / entry["radix2_ns"]


def _set_latency(entry, backend, latency_ns):
    _set_samples(entry, backend, [latency_ns] * 10)


def report(comparison_mode="single"):
    results = []
    for profile, delta in (("f64", 1e-15), ("f32", 1e-7)):
        results.append(
            {
                "profile": profile,
                "dimension": 256,
                "auto_backend": 1,
                "direct_iterations": 500,
                "optimized_iterations": 4000,
                "direct_elapsed_ns": [20_000_000] * 10,
                "radix2_elapsed_ns": [32_000_000] * 10,
                "direct_ns": 40_000.0,
                "radix2_ns": 8_000.0,
                "radix2_speedup": 5.0,
                "radix2_max_abs_vs_direct": delta,
            }
        )
    benchmark = {
        "comparison_mode": comparison_mode,
        "measurement_layer": "public-c-abi",
        "measurement_mode": "pre-resolved-bind",
        "metrics_scope": "gate",
        "repeats": 10,
        "target_work": 64_000_000,
        "sample_target_ns": 30_000_000,
        "calibration_pilots": 3,
        "maximum_calibrated_iterations": 1_000_000,
        "maximum_measurement_attempts": 3,
        "minimum_optimized_iterations": 4000,
    }
    if comparison_mode == "paired-interleaved":
        benchmark["pair_id"] = "test-pair"
    return {
        "schema_version": 2,
        "policy_effect": "none; AUTO remains unchanged",
        "benchmark": benchmark,
        "library": {"version": "0.1.0", "abi": 0, "isa": 1, "capabilities": 255},
        "host": {
            "platform": "test",
            "machine": "test",
            "python": "3.9",
            "numpy": "test",
        },
        "results": results,
    }


class PerformanceGateTests(unittest.TestCase):
    def test_calibration_uses_fastest_equal_count_pilot_across_pair(self):
        candidate = lambda: None
        baseline = lambda: None
        pilot_elapsed = {
            candidate: iter((100_000_000, 1_000_000, 2_000_000)),
            baseline: iter((80_000_000, 3_000_000, 4_000_000)),
        }
        observed_counts = []

        def fake_measure(operation, iterations):
            observed_counts.append(iterations)
            return next(pilot_elapsed[operation])

        iterations = calibrated_iterations(
            (candidate, baseline),
            100,
            30_000_000,
            measure_elapsed=fake_measure,
        )
        self.assertEqual(iterations, 3750)
        self.assertEqual(observed_counts, [100] * 6)

    def test_fast_final_batch_is_discarded_and_pair_is_escalated_together(self):
        operation = lambda: None
        iterations = calibrated_iterations(
            (operation,),
            100,
            30_000_000,
            measure_elapsed=lambda _operation, _iterations: 40_000_000,
        )
        self.assertEqual(iterations, 100)
        measured_counts = []

        def fake_paired_measure(count):
            measured_counts.append(count)
            if len(measured_counts) == 1:
                return (
                    TimingSamples(count, [5_000_000] * 10),
                    TimingSamples(count, [50_000_000] * 10),
                )
            return (
                TimingSamples(count, [37_500_000] * 10),
                TimingSamples(count, [375_000_000] * 10),
            )

        candidate, baseline = measure_with_escalation(
            fake_paired_measure, iterations, 30_000_000
        )
        self.assertEqual(measured_counts, [100, 750])
        self.assertEqual(candidate.iterations, baseline.iterations)
        self.assertEqual(candidate.iterations, 750)
        self.assertGreaterEqual(min(candidate.elapsed_ns), 30_000_000)

    def test_measurement_escalation_is_attempt_and_iteration_bounded(self):
        measured_counts = []

        def always_short(count):
            measured_counts.append(count)
            return (TimingSamples(count, [1_000_000] * 10),)

        with self.assertRaisesRegex(RuntimeError, "bounded measurement attempts"):
            measure_with_escalation(always_short, 100, 30_000_000)
        self.assertEqual(len(measured_counts), MAXIMUM_MEASUREMENT_ATTEMPTS)
        self.assertLessEqual(max(measured_counts), MAXIMUM_CALIBRATED_ITERATIONS)
        self.assertEqual(measured_counts, sorted(measured_counts))

    def test_valid_bootstrap_report_passes_and_renders_scores(self):
        result = evaluate(report(), policy())
        self.assertTrue(result.passed)
        self.assertIn("Gate: PASS", result.markdown)
        self.assertIn("public C ABI", result.markdown)
        self.assertIn("5.00x", result.markdown)
        self.assertEqual(len(result.measurements), 2)
        self.assertFalse(result.baseline_compared)

    def test_same_runner_paired_baseline_passes_and_is_rendered(self):
        candidate = report("paired-interleaved")
        baseline = report("paired-interleaved")
        _set_latency(candidate["results"][0], "direct", 42_000.0)
        _set_latency(candidate["results"][0], "radix2", 8_400.0)
        result = evaluate(candidate, policy(), baseline)
        self.assertTrue(result.passed)
        self.assertTrue(result.baseline_compared)
        self.assertIn("median paired slowdown", result.markdown)
        self.assertIn("1.05x", result.markdown)

    def test_proportional_candidate_slowdown_fails_against_base(self):
        candidate = report("paired-interleaved")
        baseline = report("paired-interleaved")
        for entry in candidate["results"]:
            _set_latency(entry, "direct", 60_000.0)
            _set_latency(entry, "radix2", 12_000.0)
        result = evaluate(candidate, policy(), baseline)
        self.assertFalse(result.passed)
        joined = "\n".join(result.failures)
        self.assertIn("direct candidate/base slowdown 1.50x", joined)
        self.assertIn("radix-2 candidate/base slowdown 1.50x", joined)

    def test_gate_uses_median_of_paired_ratios(self):
        candidate = report("paired-interleaved")
        baseline = report("paired-interleaved")
        base_samples = [40_000.0] * 5 + [400_000.0] * 5
        candidate_samples = [80_000.0] * 5 + [400_000.0] * 5
        _set_samples(baseline["results"][0], "direct", base_samples)
        _set_samples(candidate["results"][0], "direct", candidate_samples)

        ratio_of_medians = (
            candidate["results"][0]["direct_ns"]
            / baseline["results"][0]["direct_ns"]
        )
        self.assertLess(ratio_of_medians, 1.35)
        result = evaluate(candidate, policy(), baseline)
        self.assertFalse(result.passed)
        self.assertIn(
            "direct candidate/base slowdown 1.50x", "\n".join(result.failures)
        )

    def test_every_backend_sample_must_meet_duration_floor(self):
        for backend in ("direct", "radix2"):
            with self.subTest(backend=backend):
                candidate = report()
                candidate["results"][0][f"{backend}_elapsed_ns"][0] = 9_999_999
                with self.assertRaisesRegex(GateInputError, "sample duration"):
                    evaluate(candidate, policy())

    def test_pair_identity_and_sample_count_are_required(self):
        candidate = report("paired-interleaved")
        baseline = report("paired-interleaved")
        baseline["benchmark"]["pair_id"] = "another-run"
        with self.assertRaisesRegex(GateInputError, "same paired run"):
            evaluate(candidate, policy(), baseline)

        candidate = report()
        candidate["results"][0]["direct_elapsed_ns"].pop()
        with self.assertRaisesRegex(GateInputError, "expected benchmark.repeats"):
            evaluate(candidate, policy())

    def test_speed_regression_fails_when_policy_declares_a_floor(self):
        candidate = report()
        _set_latency(candidate["results"][0], "radix2", 30_000.0)
        result = evaluate(candidate, policy())
        self.assertFalse(result.passed)
        self.assertIn("below 2.00x", "\n".join(result.failures))

    def test_speed_floor_is_optional_for_backend_optimization(self):
        candidate = report()
        candidate_policy = policy()
        candidate_policy["profiles"]["f64"].pop("minimum_speedup")
        _set_latency(candidate["results"][0], "radix2", 30_000.0)
        result = evaluate(candidate, candidate_policy)
        self.assertTrue(result.passed)

    def test_excess_numeric_delta_fails(self):
        candidate = report()
        candidate["results"][1]["radix2_max_abs_vs_direct"] = 2e-4
        result = evaluate(candidate, policy())
        self.assertFalse(result.passed)
        self.assertIn("exceeds", "\n".join(result.failures))

    def test_missing_regime_fails(self):
        candidate = report()
        candidate["results"].pop()
        result = evaluate(candidate, policy())
        self.assertFalse(result.passed)
        self.assertIn("missing result for f32 dimension 256", result.failures)

    def test_missing_baseline_regime_fails_the_rendered_row(self):
        candidate = report("paired-interleaved")
        baseline = report("paired-interleaved")
        baseline["results"].pop(0)
        result = evaluate(candidate, policy(), baseline)
        self.assertFalse(result.passed)
        self.assertIn("missing baseline result for f64 dimension 256", result.failures)
        f64 = next(item for item in result.measurements if item.profile == "f64")
        self.assertFalse(f64.passed)

    def test_sequential_reports_cannot_claim_a_baseline_comparison(self):
        with self.assertRaisesRegex(GateInputError, "comparison_mode"):
            evaluate(report(), policy(), report())

    def test_paired_report_cannot_claim_bootstrap_mode(self):
        with self.assertRaisesRegex(GateInputError, "comparison_mode"):
            evaluate(report("paired-interleaved"), policy())

    def test_duplicate_regime_is_rejected(self):
        candidate = report()
        candidate["results"].append(copy.deepcopy(candidate["results"][0]))
        with self.assertRaisesRegex(GateInputError, "duplicate"):
            evaluate(candidate, policy())

    def test_nonfinite_and_inconsistent_values_are_rejected(self):
        for field, value, message in (
            ("direct_ns", math.inf, "finite"),
            ("radix2_ns", 0.0, "positive"),
            ("radix2_speedup", math.nan, "finite"),
        ):
            with self.subTest(field=field):
                candidate = report()
                candidate["results"][0][field] = value
                with self.assertRaisesRegex(GateInputError, message):
                    evaluate(candidate, policy())
        candidate = report()
        candidate["results"][0]["direct_ns"] = 99.0
        with self.assertRaisesRegex(GateInputError, "elapsed samples"):
            evaluate(candidate, policy())

    def test_metadata_change_is_rejected(self):
        candidate = report()
        candidate["policy_effect"] = "AUTO changed"
        with self.assertRaisesRegex(GateInputError, "AUTO policy"):
            evaluate(candidate, policy())

    def test_measurement_layer_mode_and_scope_are_required(self):
        for field in ("measurement_layer", "measurement_mode", "metrics_scope"):
            with self.subTest(field=field):
                candidate = report()
                candidate["benchmark"][field] = "adapter-wrapper"
                with self.assertRaisesRegex(GateInputError, field):
                    evaluate(candidate, policy())

    def test_actual_auto_backend_change_is_rejected(self):
        candidate = report()
        candidate["results"][0]["auto_backend"] = 2
        with self.assertRaisesRegex(GateInputError, "AUTO dispatch changed"):
            evaluate(candidate, policy())

    def test_cli_failure_is_nonzero_and_publishes_summary(self):
        candidate = report()
        _set_latency(candidate["results"][0], "radix2", 30_000.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "report.json"
            policy_path = root / "policy.json"
            summary_path = root / "summary.md"
            report_path.write_text(json.dumps(candidate), encoding="utf-8")
            policy_path.write_text(json.dumps(policy()), encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = main(
                    [
                        "--report",
                        str(report_path),
                        "--policy",
                        str(policy_path),
                        "--summary",
                        str(summary_path),
                    ]
                )
            self.assertEqual(status, 1)
            self.assertIn("Gate: FAIL", summary_path.read_text(encoding="utf-8"))
            self.assertIn("below 2.00x", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

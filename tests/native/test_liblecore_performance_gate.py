#!/usr/bin/env python3
"""Regression tests for the liblecore performance-policy checker."""

from __future__ import annotations

import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.check_liblecore_performance import GateInputError, evaluate, main  # noqa: E402


def policy():
    return {
        "schema_version": 1,
        "report_schema_version": 1,
        "required_policy_effect": "none; AUTO remains unchanged",
        "required_baseline_comparison_mode": "interleaved",
        "required_bootstrap_comparison_mode": "single",
        "minimum_repeats": 10,
        "minimum_optimized_iterations": 4000,
        "max_candidate_slowdown": 1.35,
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


def report(comparison_mode="single"):
    results = []
    for profile, delta in (("f64", 1e-15), ("f32", 1e-7)):
        results.append(
            {
                "profile": profile,
                "dimension": 256,
                "auto_backend": 1,
                "direct_iterations": 244,
                "optimized_iterations": 4000,
                "direct_ns": 40_000.0,
                "radix2_ns": 8_000.0,
                "radix2_speedup": 5.0,
                "radix2_max_abs_vs_direct": delta,
            }
        )
    return {
        "schema_version": 1,
        "policy_effect": "none; AUTO remains unchanged",
        "benchmark": {
            "comparison_mode": comparison_mode,
            "repeats": 10,
            "target_work": 64_000_000,
            "minimum_optimized_iterations": 4000,
        },
        "library": {"version": "0.1.0", "abi": 0, "isa": 1, "capabilities": 255},
        "host": {"platform": "test", "machine": "test", "python": "3.9", "numpy": "test"},
        "results": results,
    }


class PerformanceGateTests(unittest.TestCase):
    def test_valid_report_passes_and_renders_scores(self):
        result = evaluate(report(), policy())
        self.assertTrue(result.passed)
        self.assertIn("Gate: PASS", result.markdown)
        self.assertIn("5.00x", result.markdown)
        self.assertEqual(len(result.measurements), 2)
        self.assertFalse(result.baseline_compared)

    def test_same_runner_baseline_passes_and_is_rendered(self):
        candidate = report("interleaved")
        baseline = report("interleaved")
        candidate["results"][0]["direct_ns"] = 42_000.0
        candidate["results"][0]["radix2_ns"] = 8_400.0
        result = evaluate(candidate, policy(), baseline)
        self.assertTrue(result.passed)
        self.assertTrue(result.baseline_compared)
        self.assertIn("1.05x", result.markdown)

    def test_proportional_candidate_slowdown_fails_against_base(self):
        candidate = report("interleaved")
        baseline = report("interleaved")
        for entry in candidate["results"]:
            entry["direct_ns"] *= 1.5
            entry["radix2_ns"] *= 1.5
        result = evaluate(candidate, policy(), baseline)
        self.assertFalse(result.passed)
        joined = "\n".join(result.failures)
        self.assertIn("direct candidate/base slowdown 1.50x", joined)
        self.assertIn("radix-2 candidate/base slowdown 1.50x", joined)

    def test_speed_regression_fails(self):
        candidate = report()
        candidate["results"][0]["radix2_ns"] = 30_000.0
        candidate["results"][0]["radix2_speedup"] = 4.0 / 3.0
        result = evaluate(candidate, policy())
        self.assertFalse(result.passed)
        self.assertIn("below 2.00x", "\n".join(result.failures))

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
        candidate = report("interleaved")
        baseline = report("interleaved")
        baseline["results"].pop(0)
        result = evaluate(candidate, policy(), baseline)
        self.assertFalse(result.passed)
        self.assertIn("missing baseline result for f64 dimension 256", result.failures)
        f64 = next(item for item in result.measurements if item.profile == "f64")
        self.assertFalse(f64.passed)

    def test_sequential_reports_cannot_claim_a_baseline_comparison(self):
        with self.assertRaisesRegex(GateInputError, "comparison_mode"):
            evaluate(report(), policy(), report())

    def test_interleaved_report_cannot_claim_bootstrap_mode(self):
        with self.assertRaisesRegex(GateInputError, "comparison_mode"):
            evaluate(report("interleaved"), policy())

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
        candidate["results"][0]["radix2_speedup"] = 99.0
        with self.assertRaisesRegex(GateInputError, "inconsistent"):
            evaluate(candidate, policy())

    def test_metadata_change_is_rejected(self):
        candidate = report()
        candidate["policy_effect"] = "AUTO changed"
        with self.assertRaisesRegex(GateInputError, "AUTO policy"):
            evaluate(candidate, policy())

    def test_actual_auto_backend_change_is_rejected(self):
        candidate = report()
        candidate["results"][0]["auto_backend"] = 2
        with self.assertRaisesRegex(GateInputError, "AUTO dispatch changed"):
            evaluate(candidate, policy())

    def test_cli_failure_is_nonzero_and_publishes_summary(self):
        candidate = report()
        candidate["results"][0]["radix2_ns"] = 30_000.0
        candidate["results"][0]["radix2_speedup"] = 4.0 / 3.0
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

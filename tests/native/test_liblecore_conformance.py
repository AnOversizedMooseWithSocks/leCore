"""Fixture-driven differential conformance checks for the ABI-0 C kernel.

Run directly so the shared artifact is always explicit::

    python3 tests/native/test_liblecore_conformance.py --library /path/to/liblecore.so

The module is safe to discover in the repository's ordinary Python suite: it
skips when no explicit artifact was supplied and never searches the machine for
a library or dispatches away from the NumPy implementation.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import platform
import queue
import re
import sys
import threading
from typing import Any, Dict, Optional
import unittest
import weakref

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from bindings.python.lecore_native import (  # noqa: E402
    LECORE_BACKEND_DIRECT,
    LECORE_CAP_RADIX2,
    LECORE_PROFILE_HRR_F32_V1,
    LECORE_PROFILE_HRR_F64_V1,
    Library,
    LeCoreClosedError,
    LeCoreCompatibilityError,
    LeCoreNonFiniteError,
    LeCoreThreadError,
)
from tools.generate_liblecore_fixtures import render_fixture  # noqa: E402


DEFAULT_FIXTURE = REPOSITORY_ROOT / "tests/native/fixtures/liblecore_isa_v1.json"
_LIBRARY_PATH: Optional[str] = None
_FIXTURE_PATH: Path = DEFAULT_FIXTURE
_BACKEND = "direct"
_REPORT: Dict[str, Any] = {}
_BACKEND_NAMES = {0: "auto", 1: "direct", 2: "radix2"}


def _decode(specification: dict) -> np.ndarray:
    dtype = np.dtype(specification["dtype"])
    if dtype == np.dtype(np.float64):
        unsigned_dtype = np.uint64
    elif dtype == np.dtype(np.float32):
        unsigned_dtype = np.uint32
    else:
        raise AssertionError(f"unsupported fixture dtype {dtype}")
    bits = np.asarray(
        [int(value, 16) for value in specification["bits"]],
        dtype=unsigned_dtype,
    )
    return bits.view(dtype).reshape(tuple(specification["shape"])).copy(order="C")


def _scalar(specification: dict) -> float:
    return float(_decode(specification).reshape(()))


def _profile_key(profile: str) -> str:
    return "HRR_F64_V1" if profile == "f64" else "HRR_F32_V1"


def _compiler_metadata(artifact: str) -> dict:
    """Read CMake's adjacent, immutable compiler record when available."""
    build_directory = Path(artifact).resolve().parent
    metadata = {"status": "not exposed by ABI-0 artifact introspection"}
    cache = build_directory / "CMakeCache.txt"
    if cache.is_file():
        match = re.search(
            r"^CMAKE_C_COMPILER:FILEPATH=(.+)$",
            cache.read_text(encoding="utf-8", errors="replace"),
            re.MULTILINE,
        )
        if match:
            metadata["path"] = match.group(1)
    records = sorted(build_directory.glob("CMakeFiles/*/CMakeCCompiler.cmake"))
    if records:
        contents = records[-1].read_text(encoding="utf-8", errors="replace")
        for field, key in (
            ("CMAKE_C_COMPILER_ID", "id"),
            ("CMAKE_C_COMPILER_VERSION", "version"),
        ):
            match = re.search(rf'set\({field} "([^"]*)"\)', contents)
            if match:
                metadata[key] = match.group(1)
        if "id" in metadata:
            metadata["status"] = "reported by adjacent CMake build metadata"
    return metadata


def _record_value(profile: str, operation: str, difference: float) -> None:
    values = _REPORT.setdefault("profiles", {}).setdefault(profile, {}).setdefault(
        "max_abs_error", {}
    )
    values[operation] = max(float(values.get(operation, 0.0)), float(difference))


def _record_decision(profile: str, operation: str, passed: bool) -> None:
    decisions = _REPORT.setdefault("profiles", {}).setdefault(profile, {}).setdefault(
        "decisions", {}
    )
    result = decisions.setdefault(operation, {"checked": 0, "agreed": 0})
    result["checked"] += 1
    result["agreed"] += int(bool(passed))


def _require_backend_capability(library, backend: str) -> None:
    if backend == "radix2" and not library.capabilities & LECORE_CAP_RADIX2:
        raise RuntimeError(
            "forced radix2 conformance requires LECORE_CAP_RADIX2; "
            "the selected artifact does not advertise it"
        )


class LiblecoreConformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if _LIBRARY_PATH is None:
            raise unittest.SkipTest("no explicit --library artifact supplied")
        cls.library = Library(_LIBRARY_PATH)
        cls.fixture_text = _FIXTURE_PATH.read_text(encoding="utf-8")
        cls.fixture = json.loads(cls.fixture_text)
        _REPORT.update(
            {
                "schema": "org.lecore.conformance-report.v1",
                "artifact": {
                    "path": cls.library.path,
                    "version": cls.library.version,
                    "abi": cls.library.abi_version,
                    "isa": cls.library.isa_version,
                    "capabilities": f"0x{cls.library.capabilities:016x}",
                },
                "fixture": {
                    "path": str(_FIXTURE_PATH.resolve()),
                    "schema": cls.fixture["schema"],
                    "generator": cls.fixture["generator"],
                },
                "target": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "machine": platform.machine(),
                    "python": platform.python_version(),
                    "compiler": _compiler_metadata(cls.library.path),
                },
                "backend_requested": _BACKEND,
                "backends_observed": [],
                "errors": {},
                "profiles": {
                    "f64": {
                        "profile_id": "0x00010001",
                        "tolerance": cls.fixture["profiles"]["HRR_F64_V1"]["tolerance"],
                    },
                    "f32": {
                        "profile_id": "0x00010002",
                        "tolerance": cls.fixture["profiles"]["HRR_F32_V1"]["tolerance"],
                    },
                },
            }
        )
        _require_backend_capability(cls.library, _BACKEND)

    def _supports_dimension(self, dimension: int) -> bool:
        return _BACKEND != "radix2" or (
            dimension > 0 and dimension & (dimension - 1) == 0
        )

    def _context(self, dimension: int, profile: str, *, finite: bool = False):
        context = self.library.context(
            dimension,
            profile=profile,
            backend=_BACKEND,
            finite=finite,
        )
        observed = _REPORT["backends_observed"]
        backend_record = {
            "id": context.backend,
            "name": _BACKEND_NAMES.get(context.backend, "unknown"),
        }
        if backend_record not in observed:
            observed.append(backend_record)
        return context

    def _assert_values(
        self,
        actual: np.ndarray,
        expected: np.ndarray,
        *,
        profile: str,
        operation: str,
        atol: float,
        exact: bool = False,
    ) -> None:
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(actual.flags.c_contiguous)
        if exact:
            unsigned = np.uint64 if actual.dtype.itemsize == 8 else np.uint32
            agreement = np.array_equal(
                actual.reshape(-1).view(unsigned),
                expected.reshape(-1).view(unsigned),
            )
            _record_decision(profile, operation, agreement)
            self.assertTrue(agreement, f"{profile}/{operation} is not bit exact")
            return
        self.assertTrue(np.array_equal(np.isnan(actual), np.isnan(expected)))
        self.assertTrue(np.array_equal(np.isposinf(actual), np.isposinf(expected)))
        self.assertTrue(np.array_equal(np.isneginf(actual), np.isneginf(expected)))
        self.assertTrue(np.array_equal(np.isfinite(actual), np.isfinite(expected)))
        finite = np.isfinite(expected)
        difference = (
            float(np.max(np.abs(actual[finite] - expected[finite])))
            if np.any(finite)
            else 0.0
        )
        _record_value(profile, operation, difference)
        self.assertLessEqual(
            difference,
            atol,
            f"{profile}/{operation} max abs error {difference} exceeds {atol}",
        )

    def _assert_scalar(
        self,
        actual: float,
        expected: float,
        *,
        profile: str,
        operation: str,
        atol: float,
    ) -> None:
        if np.isnan(expected):
            self.assertTrue(np.isnan(actual))
            difference = 0.0
        elif np.isposinf(expected):
            self.assertTrue(np.isposinf(actual))
            difference = 0.0
        elif np.isneginf(expected):
            self.assertTrue(np.isneginf(actual))
            difference = 0.0
        else:
            self.assertTrue(np.isfinite(actual))
            difference = abs(float(actual) - float(expected))
            self.assertLessEqual(difference, atol)
        _record_value(profile, operation, difference)

    def test_00_fixture_metadata_and_regeneration(self) -> None:
        self.assertEqual(self.fixture["schema"], "org.lecore.conformance-fixture.v1")
        self.assertEqual(self.fixture["abi_preview"], 0)
        self.assertEqual(self.fixture["isa_version"], 1)
        self.assertEqual(self.fixture["generator"]["algorithm"], "index-formula-v1/no-rng")
        self.assertEqual(self.fixture_text, render_fixture())
        missing_radix = type("MissingRadix", (), {"capabilities": 0})()
        _require_backend_capability(missing_radix, "direct")
        with self.assertRaises(RuntimeError):
            _require_backend_capability(missing_radix, "radix2")

    def test_01_explicit_artifact_and_context_metadata(self) -> None:
        with self.assertRaises(TypeError):
            Library(None)  # type: ignore[arg-type]
        with self.assertRaises(FileNotFoundError):
            Library(_FIXTURE_PATH.parent / "not-a-library")
        with self.assertRaises(LeCoreCompatibilityError):
            Library(self.library.path, expected_abi=1)
        with self.assertRaises(LeCoreCompatibilityError):
            Library(self.library.path, expected_isa=2)
        with self.assertRaises(LeCoreCompatibilityError):
            Library(self.library.path, expected_abi=False)
        with self.assertRaises(LeCoreCompatibilityError):
            Library(self.library.path, expected_isa=1.0)  # type: ignore[arg-type]
        # Unsupported adapter contracts are rejected before artifact resolution,
        # and therefore before ABI-0 ctypes declarations can be installed.
        with self.assertRaises(LeCoreCompatibilityError):
            Library(
                _FIXTURE_PATH.parent / "not-a-library",
                expected_abi=1,
            )

        expected_profiles = {
            "f64": LECORE_PROFILE_HRR_F64_V1,
            "f32": LECORE_PROFILE_HRR_F32_V1,
        }
        for profile, profile_id in expected_profiles.items():
            with self._context(8, profile) as context:
                self.assertEqual(context.dimension, 8)
                self.assertEqual(context.profile, profile_id)
                self.assertGreater(context.scratch_bytes, 0)
                if _BACKEND == "direct":
                    self.assertEqual(context.backend, LECORE_BACKEND_DIRECT)
            self.assertTrue(context.closed)
            with self.assertRaises(LeCoreClosedError):
                context.normalize(np.zeros(8, dtype=context.dtype))

    def test_02_strict_arrays_aliases_and_lifetime(self) -> None:
        dimension = 8 if _BACKEND == "radix2" else 3
        case = next(
            item
            for item in self.fixture["profiles"]["HRR_F64_V1"]["finite_cases"]
            if int(item["dimension"]) == dimension
        )
        a = _decode(case["inputs"]["a"])
        b = _decode(case["inputs"]["b"])
        with self._context(dimension, "f64") as context:
            with self.assertRaises(TypeError):
                context.normalize([1.0, 2.0, 3.0])  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                context.normalize(a.astype(np.float32))
            with self.assertRaises(ValueError):
                context.normalize(np.zeros(dimension + 1, dtype=np.float64))
            with self.assertRaises(ValueError):
                context.normalize(np.zeros(dimension * 2, dtype=np.float64)[::2])
            read_only = np.empty(dimension, dtype=np.float64)
            read_only.flags.writeable = False
            with self.assertRaises(ValueError):
                context.normalize(a, out=read_only)

            overlapping = np.arange(dimension + 1, dtype=np.float64)
            with self.assertRaises(ValueError):
                context.normalize(
                    overlapping[:dimension], out=overlapping[1 : dimension + 1]
                )

            expected = context.normalize(a)
            aliased = a.copy()
            returned = context.normalize(aliased, out=aliased)
            self.assertIs(returned, aliased)
            np.testing.assert_allclose(aliased, expected, atol=1e-9, rtol=0)

            expected = context.involution(a)
            aliased = a.copy()
            context.involution(aliased, out=aliased)
            np.testing.assert_array_equal(aliased, expected)

            expected = context.permute(a, -5)
            aliased = a.copy()
            context.permute(aliased, -5, out=aliased)
            np.testing.assert_array_equal(aliased, expected)

            expected = context.bind(a, b)
            aliased_a = a.copy()
            context.bind(aliased_a, b, out=aliased_a)
            np.testing.assert_allclose(aliased_a, expected, atol=1e-9, rtol=0)
            aliased_b = b.copy()
            context.bind(a, aliased_b, out=aliased_b)
            np.testing.assert_allclose(aliased_b, expected, atol=1e-9, rtol=0)

            expected = context.unbind(a, b)
            aliased = a.copy()
            context.unbind(aliased, b, out=aliased)
            np.testing.assert_allclose(aliased, expected, atol=1e-9, rtol=0)

            rows = np.stack((a, b))
            with self.assertRaises(ValueError):
                context.bundle(rows, out=rows[0])
            with self.assertRaises(ValueError):
                context.bind_batch(rows, rows, out=rows)

    def test_02b_unique_thread_ownership_and_nonfinite_assertions(self) -> None:
        dimension = 8
        case = next(
            item
            for item in self.fixture["profiles"]["HRR_F64_V1"]["finite_cases"]
            if int(item["dimension"]) == dimension
        )
        a = _decode(case["inputs"]["a"])
        b = _decode(case["inputs"]["b"])
        with self._context(dimension, "f64") as context:
            with self.assertRaises(TypeError):
                copy.copy(context)
            with self.assertRaises(TypeError):
                copy.deepcopy(context)

            barrier = threading.Barrier(5)
            results = queue.Queue()  # type: queue.Queue

            def attempt(label, operation) -> None:
                barrier.wait(timeout=5.0)
                try:
                    operation()
                except BaseException as error:
                    results.put((label, error))
                else:
                    results.put((label, None))

            operations = {
                "call": lambda: context.dot(a, b),
                "close": context.close,
                "scratch_property": lambda: context.scratch_bytes,
                "closed_property": lambda: context.closed,
            }
            threads = [
                threading.Thread(target=attempt, args=(label, operation))
                for label, operation in operations.items()
            ]
            for thread in threads:
                thread.start()
            barrier.wait(timeout=5.0)
            for thread in threads:
                thread.join(timeout=5.0)
                self.assertFalse(thread.is_alive())

            observed = {}
            for _ in threads:
                label, error = results.get_nowait()
                observed[label] = error
            self.assertEqual(set(observed), set(operations))
            for error in observed.values():
                self.assertIsInstance(error, LeCoreThreadError)

            # Rejected foreign-thread access cannot close or corrupt the owner’s
            # context, and the native handle remains usable on its creator.
            self.assertFalse(context.closed)
            self._assert_scalar(
                context.dot(a, b),
                _scalar(case["expected"]["dot"]),
                profile="f64",
                operation="dot_after_thread_rejection",
                atol=1e-9,
            )

        expected_nonfinite = np.asarray([np.inf, -np.inf, np.nan], dtype=np.float64)
        for wrong_nonfinite in (
            np.asarray([-np.inf, -np.inf, np.nan], dtype=np.float64),
            np.asarray([1.0, -np.inf, np.nan], dtype=np.float64),
        ):
            with self.assertRaises(AssertionError):
                self._assert_values(
                    wrong_nonfinite,
                    expected_nonfinite,
                    profile="harness",
                    operation="nonfinite_classification_regression",
                    atol=0.0,
                )

    def test_02c_foreign_thread_finalization_is_exactly_once(self) -> None:
        tracking_library = Library(self.library.path)
        original_destroy = tracking_library._dll.lecore_context_destroy
        destroy_threads = []
        destroy_lock = threading.Lock()

        def tracked_destroy(pointer) -> None:
            with destroy_lock:
                destroy_threads.append(threading.get_ident())
            original_destroy(pointer)

        tracking_library._dll.lecore_context_destroy = tracked_destroy

        def dispose_last_reference_on_foreign_thread(holder, reference):
            release_threads = []

            def release() -> None:
                release_threads.append(threading.get_ident())
                value = holder.pop()
                del value
                gc.collect()

            thread = threading.Thread(target=release)
            thread.start()
            thread.join(timeout=5.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(holder, [])
            self.assertIsNone(reference())
            self.assertEqual(len(release_threads), 1)
            return release_threads[0]

        try:
            unclosed = tracking_library.context(8, profile="f64", backend="direct")
            unclosed_reference = weakref.ref(unclosed)
            unclosed_holder = [unclosed]
            del unclosed
            release_thread = dispose_last_reference_on_foreign_thread(
                unclosed_holder,
                unclosed_reference,
            )
            self.assertEqual(len(destroy_threads), 1)
            self.assertEqual(destroy_threads[0], release_thread)
            self.assertNotEqual(destroy_threads[0], threading.get_ident())
            gc.collect()
            self.assertEqual(len(destroy_threads), 1)

            owner_exit_holder = []

            def create_then_exit() -> None:
                owner_exit_holder.append(
                    tracking_library.context(8, profile="f64", backend="direct")
                )

            creator_thread = threading.Thread(target=create_then_exit)
            creator_thread.start()
            creator_thread.join(timeout=5.0)
            self.assertFalse(creator_thread.is_alive())
            self.assertEqual(len(owner_exit_holder), 1)
            after_exit_reference = weakref.ref(owner_exit_holder[0])
            after_exit_release_thread = dispose_last_reference_on_foreign_thread(
                owner_exit_holder,
                after_exit_reference,
            )
            self.assertEqual(len(destroy_threads), 2)
            self.assertEqual(destroy_threads[1], after_exit_release_thread)

            explicitly_closed = tracking_library.context(
                8, profile="f64", backend="direct"
            )
            explicitly_closed.close()
            self.assertEqual(len(destroy_threads), 3)
            self.assertEqual(destroy_threads[2], threading.get_ident())
            closed_reference = weakref.ref(explicitly_closed)
            closed_holder = [explicitly_closed]
            del explicitly_closed
            dispose_last_reference_on_foreign_thread(
                closed_holder,
                closed_reference,
            )
            gc.collect()
            self.assertEqual(
                len(destroy_threads),
                3,
                "finalization destroyed an explicitly closed context twice",
            )
        finally:
            tracking_library._dll.lecore_context_destroy = original_destroy

    def _exercise_finite_profile(self, profile: str) -> None:
        specification = self.fixture["profiles"][_profile_key(profile)]
        atol = float(specification["tolerance"]["atol"])
        for case in specification["finite_cases"]:
            dimension = int(case["dimension"])
            if not self._supports_dimension(dimension):
                continue
            inputs = case["inputs"]
            expected = case["expected"]
            a = _decode(inputs["a"])
            b = _decode(inputs["b"])
            rows = _decode(inputs["rows"])
            paired_left = _decode(inputs["paired_left"])
            paired_right = _decode(inputs["paired_right"])
            with self._context(dimension, profile) as context:
                self._assert_values(
                    context.normalize(a),
                    _decode(expected["normalize"]),
                    profile=profile,
                    operation="normalize",
                    atol=atol,
                )
                self._assert_scalar(
                    context.dot(a, b),
                    _scalar(expected["dot"]),
                    profile=profile,
                    operation="dot",
                    atol=atol,
                )
                self._assert_values(
                    context.dot_many(a, rows),
                    _decode(expected["dot_many"]),
                    profile=profile,
                    operation="dot_many",
                    atol=atol,
                )
                self._assert_scalar(
                    context.cosine(a, b),
                    _scalar(expected["cosine"]),
                    profile=profile,
                    operation="cosine",
                    atol=atol,
                )
                native_scores = context.cosine_many(a, rows)
                expected_scores = _decode(expected["cosine_many"])
                self._assert_values(
                    native_scores,
                    expected_scores,
                    profile=profile,
                    operation="cosine_many",
                    atol=atol,
                )
                decision_agrees = int(np.argmax(native_scores)) == int(
                    np.argmax(expected_scores)
                )
                _record_decision(profile, "cosine_many_argmax", decision_agrees)
                self.assertTrue(decision_agrees)
                self._assert_values(
                    context.bind(a, b),
                    _decode(expected["bind"]),
                    profile=profile,
                    operation="bind",
                    atol=atol,
                )
                self._assert_values(
                    context.bind(b, a),
                    _decode(expected["bind"]),
                    profile=profile,
                    operation="bind_commutativity",
                    atol=atol,
                )
                self._assert_values(
                    context.unbind(a, b),
                    _decode(expected["unbind"]),
                    profile=profile,
                    operation="unbind",
                    atol=atol,
                )
                self._assert_values(
                    context.involution(a),
                    _decode(expected["involution"]),
                    profile=profile,
                    operation="involution",
                    atol=0.0,
                    exact=True,
                )
                self._assert_values(
                    context.permute(a, int(case["shift"])),
                    _decode(expected["permute"]),
                    profile=profile,
                    operation="permute",
                    atol=0.0,
                    exact=True,
                )
                self._assert_values(
                    context.bundle(rows),
                    _decode(expected["bundle"]),
                    profile=profile,
                    operation="bundle",
                    atol=atol,
                )
                index, score = context.cleanup(a, rows)
                index_agrees = index == int(expected["cleanup"]["index"])
                _record_decision(profile, "cleanup", index_agrees)
                self.assertTrue(index_agrees)
                self._assert_scalar(
                    score,
                    _scalar(expected["cleanup"]["score"]),
                    profile=profile,
                    operation="cleanup_score",
                    atol=atol,
                )
                self._assert_values(
                    context.bind_batch(paired_left, paired_right),
                    _decode(expected["bind_batch"]),
                    profile=profile,
                    operation="bind_batch",
                    atol=atol,
                )
                self._assert_values(
                    context.bind_fixed(a, rows),
                    _decode(expected["bind_fixed"]),
                    profile=profile,
                    operation="bind_fixed",
                    atol=atol,
                )
                self._assert_values(
                    context.unbind_all(a, rows),
                    _decode(expected["unbind_all"]),
                    profile=profile,
                    operation="unbind_all",
                    atol=atol,
                )

    def test_03_f64_fixture_operations_and_batches(self) -> None:
        self._exercise_finite_profile("f64")

    def test_04_f32_fixture_operations_and_batches(self) -> None:
        self._exercise_finite_profile("f32")

    def test_05_ordered_reduction_dimensions(self) -> None:
        for profile in ("f64", "f32"):
            specification = self.fixture["profiles"][_profile_key(profile)]
            atol = float(specification["tolerance"]["atol"])
            for case in specification["reduction_cases"]:
                dimension = int(case["dimension"])
                if not self._supports_dimension(dimension):
                    continue
                query = _decode(case["inputs"]["query"])
                rows = _decode(case["inputs"]["rows"])
                with self._context(dimension, profile) as context:
                    self._assert_values(
                        context.dot_many(query, rows),
                        _decode(case["expected"]["dot_many"]),
                        profile=profile,
                        operation=f"dot_many_d{dimension}",
                        atol=atol,
                    )
                    self._assert_values(
                        context.cosine_many(query, rows),
                        _decode(case["expected"]["cosine_many"]),
                        profile=profile,
                        operation=f"cosine_many_d{dimension}",
                        atol=atol,
                    )

    def test_06_mixed_f64_f32_scorer(self) -> None:
        profile = self.fixture["profiles"]["HRR_F32_V1"]
        for case in profile["finite_cases"]:
            dimension = int(case["dimension"])
            if not self._supports_dimension(dimension):
                continue
            query = _decode(case["inputs"]["mixed_query"])
            rows = _decode(case["inputs"]["rows"])
            expected = _decode(case["expected"]["cosine_many_f64_f32"])
            with self._context(dimension, "f32") as context:
                self._assert_values(
                    context.cosine_many_f64_f32(query, rows),
                    expected,
                    profile="mixed_f64_f32",
                    operation="cosine_many",
                    atol=1e-9,
                )
                with self.assertRaises(TypeError):
                    context.cosine_many_f64_f32(query.astype(np.float32), rows)
            with self._context(dimension, "f64") as wrong_context:
                with self.assertRaises(LeCoreCompatibilityError):
                    wrong_context.cosine_many_f64_f32(query, rows)

    def test_07_zero_nonfinite_and_cleanup_decisions(self) -> None:
        for profile in ("f64", "f32"):
            specification = self.fixture["profiles"][_profile_key(profile)]
            atol = float(specification["tolerance"]["atol"])
            for edge in specification["edge_cases"]:
                dimension = int(edge["dimension"])
                if not self._supports_dimension(dimension):
                    continue
                inputs = edge["inputs"]
                finite = _decode(inputs["finite"])
                zero = _decode(inputs["zero"])
                nan_vector = _decode(inputs["nan_vector"])
                inf_vector = _decode(inputs["inf_vector"])
                impulse = _decode(inputs["impulse"])
                shifted_impulse = _decode(inputs["shifted_impulse"])
                ties = _decode(inputs["ties"])
                first_nan = _decode(inputs["first_nan_candidates"])
                zero_sum_rows = _decode(inputs["zero_sum_rows"])
                reindex_special = _decode(inputs["reindex_special"])
                expected = edge["expected"]
                with self._context(dimension, profile) as context:
                    self._assert_values(
                        context.involution(reindex_special),
                        _decode(expected["special_involution"]),
                        profile=profile,
                        operation="special_involution_bits",
                        atol=0.0,
                        exact=True,
                    )
                    self._assert_values(
                        context.permute(reindex_special, -2),
                        _decode(expected["special_permute_minus_two"]),
                        profile=profile,
                        operation="special_permute_bits",
                        atol=0.0,
                        exact=True,
                    )
                    self._assert_values(
                        context.bind(impulse, finite),
                        _decode(expected["impulse_bind"]),
                        profile=profile,
                        operation="impulse_bind",
                        atol=atol,
                    )
                    self._assert_values(
                        context.bind(shifted_impulse, finite),
                        _decode(expected["shifted_impulse_bind"]),
                        profile=profile,
                        operation="shifted_impulse_bind",
                        atol=atol,
                    )
                    np.testing.assert_array_equal(
                        context.normalize(zero), _decode(expected["zero_normalize"])
                    )
                    np.testing.assert_array_equal(
                        context.bundle(zero_sum_rows), _decode(expected["zero_sum_bundle"])
                    )
                    zero_nan = context.cosine(zero, nan_vector)
                    self.assertEqual(zero_nan, 0.0)
                    self.assertFalse(np.signbit(zero_nan))
                    tied_index, _ = context.cleanup(finite, ties)
                    tied_agrees = tied_index == int(expected["tie_cleanup_index"])
                    _record_decision(profile, "lowest_index_tie", tied_agrees)
                    self.assertTrue(tied_agrees)
                    nan_index, nan_score = context.cleanup(finite, first_nan)
                    nan_agrees = nan_index == int(expected["first_nan_cleanup_index"])
                    _record_decision(profile, "first_nan", nan_agrees)
                    self.assertTrue(nan_agrees)
                    self.assertTrue(np.isnan(nan_score))
                    self.assertTrue(np.all(np.isnan(context.bind(finite, nan_vector))))
                    self.assertTrue(np.any(~np.isfinite(context.bind(finite, inf_vector))))

                with self._context(dimension, profile, finite=True) as checked:
                    for bad in (nan_vector, inf_vector):
                        output = np.full(dimension, 17.0, dtype=finite.dtype)
                        with self.assertRaises(LeCoreNonFiniteError):
                            checked.normalize(bad, out=output)
                        np.testing.assert_array_equal(
                            output, np.full(dimension, 17.0, dtype=finite.dtype)
                        )
                        _REPORT["errors"][f"{profile}_finite_rejection"] = (
                            _REPORT["errors"].get(f"{profile}_finite_rejection", 0) + 1
                        )
                for bad in (nan_vector, inf_vector):
                    with self.assertRaises(LeCoreNonFiniteError):
                        self.library.validate(bad)

    def test_08_mixed_zero_nan_precedence(self) -> None:
        edge = next(
            item
            for item in self.fixture["profiles"]["HRR_F32_V1"]["edge_cases"]
            if self._supports_dimension(int(item["dimension"]))
        )
        dimension = int(edge["dimension"])
        query = _decode(edge["inputs"]["zero"]).astype(np.float64)
        rows = np.stack(
            (
                _decode(edge["inputs"]["nan_vector"]),
                _decode(edge["inputs"]["finite"]),
            )
        )
        with self._context(dimension, "f32") as context:
            scores = context.cosine_many_f64_f32(query, rows)
        np.testing.assert_array_equal(scores, np.zeros(2, dtype=np.float64))
        self.assertFalse(np.any(np.signbit(scores)))


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, help="explicit shared-library artifact")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--backend", choices=("direct", "radix2", "auto"), default="direct")
    parser.add_argument("--report", type=Path, help="also write the JSON report to this path")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    global _LIBRARY_PATH, _FIXTURE_PATH, _BACKEND
    arguments = _parse_arguments()
    _LIBRARY_PATH = arguments.library
    _FIXTURE_PATH = arguments.fixture
    _BACKEND = arguments.backend
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LiblecoreConformance)
    result = unittest.TextTestRunner(verbosity=1 if arguments.quiet else 2).run(suite)
    _REPORT["result"] = {
        "passed": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
    }
    rendered = json.dumps(_REPORT, indent=2, sort_keys=True) + "\n"
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Parity and fallback gates for the optional native Qwen GDN recurrence."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from holographic.io_and_interop.holographic_gdnaccel import (
    GDNRecurrence, _C_SOURCE, gdn_recurrence_numpy)


def _inputs(tokens=37, key_dim=16, value_dim=16, seed=7):
    """Qwen3.5 mini-fixture head counts/dimensions (the real model divided by 8)."""
    rng = np.random.default_rng(seed)
    heads = 16
    q = rng.standard_normal((tokens, heads, key_dim), dtype=np.float32) * 0.1
    k = rng.standard_normal(q.shape, dtype=np.float32) * 0.1
    v = rng.standard_normal((tokens, heads, value_dim), dtype=np.float32) * 0.1
    beta = 1.0 / (1.0 + np.exp(-rng.standard_normal((tokens, heads), dtype=np.float32)))
    g = -np.abs(rng.standard_normal((tokens, heads), dtype=np.float32)) * 0.03
    return q, k, v, beta.astype(np.float32), g.astype(np.float32)


def test_numpy_backend_is_the_unchanged_default():
    arrays = _inputs()
    expected = gdn_recurrence_numpy(*arrays)
    dispatch = GDNRecurrence()
    got = dispatch(*arrays)
    assert np.array_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])
    report = dispatch.report()
    assert report["active"] == "numpy"
    assert report["scope"] == "full_sequence_gdn_recurrence"
    assert report["numpy_calls"] == 1
    assert report["direct_numpy_calls"] == 1
    assert report["fallback_calls"] == 0


def test_auto_is_rejected_instead_of_silently_aliasing_c():
    with pytest.raises(ValueError, match="numpy or c"):
        GDNRecurrence("auto")


def test_native_scan_parity_including_resumed_state():
    from holographic.io_and_interop import holographic_ccrun as ccrun

    arrays = _inputs(tokens=53)
    first_ref = gdn_recurrence_numpy(*arrays)
    dispatch = GDNRecurrence("c")
    first = dispatch(*arrays)
    if dispatch.report()["refused"]:
        # A compiler-less installation supports the exact NumPy fallback.  If
        # a compiler exists, refusal is a failed native contract, not a pass.
        assert ccrun.cc_available() is None, dispatch.report()["refused"]
        assert np.array_equal(first[0], first_ref[0])
        assert np.array_equal(first[1], first_ref[1])
        return
    assert np.allclose(first[0], first_ref[0], rtol=5e-5, atol=5e-6)
    assert np.allclose(first[1], first_ref[1], rtol=5e-5, atol=5e-6)

    tail = _inputs(tokens=19, seed=11)
    resumed_ref = gdn_recurrence_numpy(*tail, initial=first_ref[1])
    resumed = dispatch(*tail, initial=first[1])
    assert np.allclose(resumed[0], resumed_ref[0], rtol=5e-5, atol=5e-6)
    assert np.allclose(resumed[1], resumed_ref[1], rtol=5e-5, atol=5e-6)
    report = dispatch.report()
    assert report["active"] == "c"
    assert {row["resumed"] for row in report["validated_regimes"]} == {False, True}
    assert report["native_calls"] == 2
    assert report["native_tokens"] == 72
    assert report["native_attempts"] == 2
    assert report["fallback_calls"] == 0
    assert report["fresh_and_resumed_parity_complete"] is True
    assert report["native_libraries"]
    provenance = report["native_libraries"][0]
    assert provenance["flags"] == ["-O2", "-ffp-contract=off"]
    assert len(provenance["library_sha256"]) == 64
    assert len(provenance["cache_key_sha256"]) == 64
    assert provenance["compiler"]["host_machine"]
    assert provenance["compiler"]["target"]
    assert provenance["compiler"]["pointer_bits"] in (32, 64)
    assert provenance["compiler"]["execution_environment"]["PATH"]


def test_native_compile_survives_ilxyr_cleared_environment(tmp_path):
    """Mirror ilxyr's env_clear boundary in a fresh Python interpreter."""
    from holographic.io_and_interop import holographic_ccrun as ccrun

    if ccrun.cc_available() is None:
        pytest.skip("cleared-environment test needs a C compiler")
    cache = tmp_path / "cleared-env-cache"
    program = """
import json
import sys
from holographic.io_and_interop import holographic_ccrun as ccrun
ccrun.CACHE_DIR = sys.argv[1]
details = ccrun.compile_cached_details('void lecore_env_probe(void) {}\\n', opt='safe')
print(json.dumps(details, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(cache)],
        cwd=Path(__file__).resolve().parents[1],
        env={"ILXYR_EXPERIMENT_ID": "ci-env-clear",
             "ILXYR_RUN_ID": "ci-env-clear"},
        capture_output=True, text=True, check=False, timeout=120)
    assert completed.returncode == 0, completed.stderr
    details = json.loads(completed.stdout)
    assert Path(details["path"]).is_file()
    assert len(details["library_sha256"]) == 64
    compile_env = details["compiler"]["execution_environment"]
    assert compile_env["LC_ALL"] == "C"
    assert os.path.dirname(details["compiler"]["path"]) in \
        compile_env["PATH"].split(os.pathsep)


def test_compiler_failure_preserves_stderr_in_refusal(tmp_path, monkeypatch):
    from holographic.io_and_interop import holographic_ccrun as ccrun
    from holographic.io_and_interop.holographic_emit import EmitError

    class Completed:
        def __init__(self, returncode=0, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(command, **_kwargs):
        if "--version" in command:
            return Completed(stdout="synthetic cc 1.0")
        if "-dumpmachine" in command:
            return Completed(stdout="synthetic-target")
        return Completed(returncode=23, stderr="synthetic linker failure")

    monkeypatch.setattr(ccrun, "cc_available", lambda: sys.executable)
    monkeypatch.setattr(ccrun.subprocess, "run", fake_run)
    monkeypatch.setattr(ccrun, "CACHE_DIR", str(tmp_path / "failure-cache"))
    with pytest.raises(EmitError) as caught:
        ccrun.compile_cached_details("void failure_probe(void) {}\n", opt="safe")
    message = str(caught.value)
    assert "exit code 23" in message
    assert "synthetic linker failure" in message


def test_bad_native_result_is_refused_and_falls_back(monkeypatch):
    arrays = _inputs(tokens=5)
    dispatch = GDNRecurrence("c")

    class Wrong:
        def __call__(self, q, k, v, beta, g, initial):
            return (np.full((q.shape[0], q.shape[1], v.shape[2]), 99, q.dtype),
                    np.full(initial.shape, 99, q.dtype))

    monkeypatch.setattr(dispatch, "_native", lambda dtype: Wrong())
    expected = gdn_recurrence_numpy(*arrays)
    got = dispatch(*arrays)
    assert np.array_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])
    report = dispatch.report()
    assert "parity gate failed" in report["refused"]
    assert report["native_attempts"] == 1
    assert report["native_calls"] == 0
    assert report["fallback_calls"] == 1
    assert report["fallback_tokens"] == len(arrays[0])

    # Refusal is permanent: subsequent calls do not attempt native code again,
    # and the report distinguishes them from an explicitly selected NumPy run.
    dispatch(*arrays)
    report = dispatch.report()
    assert report["native_attempts"] == 1
    assert report["fallback_calls"] == 2
    assert report["direct_numpy_calls"] == 0


def test_native_abi_uses_caller_owned_scratch_not_a_vla():
    assert "T kv[nv]" not in _C_SOURCE
    assert "T *scratch" in _C_SOURCE


def test_cold_compile_cache_is_race_safe_and_provenanced(tmp_path, monkeypatch):
    from holographic.io_and_interop import holographic_ccrun as ccrun

    if ccrun.cc_available() is None:
        pytest.skip("cold-cache race test needs a C compiler")
    monkeypatch.setattr(ccrun, "CACHE_DIR", str(tmp_path / "cc-cache"))
    source = "void lecore_cache_probe(void) {}\n"
    with ThreadPoolExecutor(max_workers=4) as pool:
        details = list(pool.map(
            lambda _index: ccrun.compile_cached_details(source, opt="safe"),
            range(4)))
    assert len({row["path"] for row in details}) == 1
    assert len({row["cache_key_sha256"] for row in details}) == 1
    assert len({row["library_sha256"] for row in details}) == 1
    assert all(len(row["cache_key_sha256"]) == 64 for row in details)
    assert all(row["compiler"]["version"] for row in details)
    assert all(row["compiler"]["target"] for row in details)
    assert not list((tmp_path / "cc-cache").glob("*.tmp"))
    assert not list((tmp_path / "cc-cache").glob("*.c"))
    warm = ccrun.compile_cached_details(source, opt="safe")
    assert warm["cache_hit"] is True
    assert warm["library_sha256"] == details[0]["library_sha256"]


def test_mini_qwen_runtime_and_resumed_logits_match(tmp_path):
    """Exercise the substitution inside the structure-faithful Qwen fixture."""
    from holographic.io_and_interop.holographic_gdnruntime import (
        GDNRuntime, load_runtime)
    from tools.build_mini_qwen import build

    model_dir = tmp_path / "mini-qwen"
    build(model_dir, layers=4, vocab=512)
    reference, config = load_runtime(model_dir)
    native_config = dict(config, gdn_recurrence_backend="c")
    native = GDNRuntime(reference.w, native_config)

    prefix = list(range(32, 52))
    ref_logits, ref_state = reference.forward(prefix, collect_state=True)
    got_logits, got_state = native.forward(prefix, collect_state=True)
    assert np.allclose(got_logits, ref_logits, rtol=1e-4, atol=1e-5)

    tail = [52, 53, 54, 55, 56, 57]
    ref_tail, ref_resumed = reference.forward(
        tail, collect_state=True, resume=ref_state)
    got_tail, got_resumed = native.forward(
        tail, collect_state=True, resume=got_state)
    assert np.allclose(got_tail, ref_tail, rtol=1e-4, atol=1e-5)
    for layer in ref_resumed.gdn:
        assert np.allclose(got_resumed.gdn[layer]["S"],
                           ref_resumed.gdn[layer]["S"], rtol=5e-5, atol=5e-6)
        assert np.allclose(got_resumed.gdn[layer]["conv"],
                           ref_resumed.gdn[layer]["conv"], rtol=1e-4, atol=1e-5)
    report = native.acceleration_report()["full_sequence_gdn_recurrence"]
    step_report = native.acceleration_report()["cached_step_gdn_recurrence"]
    assert step_report == {"scope": "single_token_cached_step",
                           "active": "numpy", "native_available": False,
                           "calls": 0}
    if report["refused"] is None:
        assert report["active"] == "c"
        assert {row["resumed"] for row in report["validated_regimes"]} == {False, True}
        assert report["fresh_and_resumed_parity_complete"] is True
    else:
        assert report["active"] == "numpy"

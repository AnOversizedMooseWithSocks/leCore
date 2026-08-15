#!/usr/bin/env python3
"""Run the full Qwen3.5 installation acceptance contract for ilxyr."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
import gc
import hashlib
import json
import math
import multiprocessing
import os
import platform
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (str(HERE), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

from contract import (METRIC_NAMES, OFFICIAL_DEPENDENCY_VERSIONS,  # noqa: E402
                      model_manifest)


EXECUTOR_OUTPUT_KEYS = frozenset(("metrics", "source"))
SOURCE_SNAPSHOT_KEYS = frozenset(("repository", "commit", "artifacts"))
EXTERNAL_ARTIFACT_KEYS = frozenset(("path", "sha256"))
RUNTIME_PROVENANCE_SCHEMA = "lecore.qwen35-runtime-provenance.v1"


def log(message):
    print(message, file=sys.stderr, flush=True)


@contextmanager
def selected_gdn_backend(backend):
    """Temporarily select one runtime backend without leaking into treatment."""
    name = "LECORE_GDN_BACKEND"
    previous = os.environ.get(name)
    os.environ[name] = str(backend)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


class ProgressRecorder:
    """Durable, monotonic progress evidence that is also visible in live logs."""

    def __init__(self, path, upload_uri=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.upload_uri = str(upload_uri).strip() if upload_uri else None
        self.upload_attempts = 0
        self.upload_failures = 0
        self.upload_requests = 0
        self.started = time.monotonic()
        self.timings = {}
        self._file_lock = threading.Lock()
        self._upload_condition = threading.Condition()
        self._upload_requested_generation = 0
        self._upload_completed_generation = 0
        self._uploader_stopping = False
        self._uploader = None
        if self.upload_uri:
            self._uploader = threading.Thread(
                target=self._upload_loop, name="qwen-progress-uploader",
                daemon=True)
            self._uploader.start()

    def _upload_loop(self):
        """Coalesce progress snapshots without blocking evaluator queue reads."""
        while True:
            with self._upload_condition:
                self._upload_condition.wait_for(
                    lambda: (self._upload_requested_generation >
                             self._upload_completed_generation) or
                            self._uploader_stopping)
                if (self._uploader_stopping and
                        self._upload_requested_generation <=
                        self._upload_completed_generation):
                    return
                generation = self._upload_requested_generation
                self.upload_attempts += 1
            snapshot = self.path.with_suffix(self.path.suffix + ".upload")
            try:
                with self._file_lock:
                    snapshot.write_bytes(self.path.read_bytes())
                completed = subprocess.run(
                    ["aws", "s3", "cp", str(snapshot), self.upload_uri,
                     "--only-show-errors"], capture_output=True, text=True,
                    check=False, timeout=30)
                if completed.returncode != 0:
                    with self._upload_condition:
                        self.upload_failures += 1
                    log("progress upload warning: %s" %
                        (completed.stderr.strip()[-1000:] or
                         "aws s3 cp exited %d" % completed.returncode))
            except (OSError, subprocess.SubprocessError) as exc:
                with self._upload_condition:
                    self.upload_failures += 1
                log("progress upload warning: %s: %s" %
                    (type(exc).__name__, exc))
            finally:
                try:
                    snapshot.unlink()
                except OSError:
                    pass
                with self._upload_condition:
                    self._upload_completed_generation = max(
                        self._upload_completed_generation, generation)
                    self._upload_condition.notify_all()

    def flush_upload(self, timeout=70):
        """Wait for the newest coalesced snapshot and stop the uploader."""
        if self._uploader is None:
            return True
        deadline = time.monotonic() + float(timeout)
        with self._upload_condition:
            while (self._upload_completed_generation <
                   self._upload_requested_generation):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.upload_failures += 1
                    log("progress upload warning: final flush timed out")
                    return False
                self._upload_condition.wait(timeout=remaining)
            self._uploader_stopping = True
            self._upload_condition.notify_all()
        self._uploader.join(timeout=1)
        return not self._uploader.is_alive()

    def emit(self, kind, **fields):
        record = {"kind": str(kind),
                  "monotonic_seconds": time.monotonic() - self.started,
                  **fields}
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        # One append/open per record makes each chunk externally tail-able even
        # if the runner is killed before its final metrics artifact is written.
        with self._file_lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        log("PROGRESS " + line)
        if self.upload_uri and kind in ("evaluation_chunk", "stage_complete",
                                        "stage_error", "run_complete"):
            with self._upload_condition:
                self.upload_requests += 1
                self._upload_requested_generation += 1
                self._upload_condition.notify_all()
        return record

    @contextmanager
    def stage(self, name):
        start = time.monotonic()
        self.emit("stage_start", stage=name)
        try:
            yield
        except BaseException as exc:
            elapsed = time.monotonic() - start
            self.timings[name] = elapsed
            self.emit("stage_error", stage=name, elapsed_seconds=elapsed,
                      error_type=type(exc).__name__)
            raise
        else:
            elapsed = time.monotonic() - start
            self.timings[name] = elapsed
            self.emit("stage_complete", stage=name, elapsed_seconds=elapsed)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_keys(label, payload, expected):
    if not isinstance(payload, dict):
        raise ValueError("%s must be a JSON object" % label)
    actual = frozenset(payload)
    if actual != expected:
        raise ValueError(
            "%s keys drifted: expected %s, got %s" %
            (label, sorted(expected), sorted(actual)))


def build_executor_output(metrics, source):
    """Build the exact closed JSON envelope accepted by frozen ilxyr.

    This validation intentionally duplicates only the tiny public boundary,
    while CI submits the resulting object to the frozen ilxyr parser itself.
    Extended run provenance belongs in a hashed external artifact, never as an
    undeclared field under ``source``.
    """
    _require_exact_keys("executor source", source, SOURCE_SNAPSHOT_KEYS)
    if not isinstance(source["repository"], str) or not source["repository"]:
        raise ValueError("executor source repository must be a non-empty string")
    if not isinstance(source["commit"], str) or not source["commit"]:
        raise ValueError("executor source commit must be a non-empty string")
    if (len(source["commit"]) != 40 or
            any(char not in "0123456789abcdef" for char in source["commit"])):
        raise ValueError("executor source commit must be lowercase 40-character hex")
    if not isinstance(source["artifacts"], list):
        raise ValueError("executor source artifacts must be a list")
    artifacts = []
    for index, artifact in enumerate(source["artifacts"]):
        _require_exact_keys(
            "executor source artifact %d" % index, artifact,
            EXTERNAL_ARTIFACT_KEYS)
        path = artifact["path"]
        digest = artifact["sha256"]
        if not isinstance(path, str) or not path:
            raise ValueError("executor source artifact path must be non-empty")
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError(
                "executor source artifact sha256 must be lowercase hex")
        artifacts.append({"path": path, "sha256": digest})

    if not isinstance(metrics, dict) or frozenset(metrics) != frozenset(METRIC_NAMES):
        raise ValueError("executor metrics do not match the frozen metric contract")
    normalized_metrics = {}
    for name in METRIC_NAMES:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)):
            raise ValueError("executor metric %s must be numeric" % name)
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("executor metric %s must be finite" % name)
        normalized_metrics[name] = value

    envelope = {
        "metrics": normalized_metrics,
        "source": {
            "repository": source["repository"],
            "commit": source["commit"],
            "artifacts": artifacts,
        },
    }
    _require_exact_keys("executor output", envelope, EXECUTOR_OUTPUT_KEYS)
    return envelope


def acceptance_gate_pass(required_gates, native_gdn_active,
                         require_native_gdn=False):
    """Resolve GO only from the frozen mandatory gates for this run policy."""
    gates = list(required_gates)
    if require_native_gdn:
        gates.append(native_gdn_active)
    return float(all(float(value) >= 1.0 for value in gates))


def executor_contract_fixture(min_tokens=4096, source_commit=None):
    """A zero-compute representative envelope for the frozen-ilxyr CI run."""
    metrics = {name: 0.0 for name in METRIC_NAMES}
    for name in (
            "acceptance_pass", "source_clean", "experimental_installer_used",
            "tokenizer_parity_pass", "reference_logit_parity_pass",
            "full_evaluation_pass", "statistical_gate_pass", "reload_pass",
            "official_reload_pass", "text_generation_pass",
            "vision_smoke_pass"):
        metrics[name] = 1.0
    metrics["eval_tokens"] = float(min_tokens)
    metrics["sequential_looks_completed"] = 1.0
    metrics["original_perplexity"] = 1.0
    metrics["installed_perplexity"] = 1.0
    metrics["paired_block_length"] = 1.0
    metrics["paired_effective_tokens"] = float(min_tokens)
    source = {
        "repository": "https://github.com/atimics/holostuff.git",
        "commit": source_commit or "0" * 40,
        "artifacts": [{
            "path": "experiments/qwen35_acceptance/run.py",
            "sha256": sha256_file(Path(__file__)),
        }],
    }
    return build_executor_output(metrics, source)


def peak_rss_mb(who):
    try:
        import resource
        if isinstance(who, str):
            who = getattr(resource, who)
        raw = float(resource.getrusage(who).ru_maxrss)
        return raw / (1024.0 * 1024.0) if platform.system() == "Darwin" else raw / 1024.0
    except Exception:
        return 0.0


def available_memory_mb():
    """Best-effort current physical-memory headroom, without extra packages."""
    try:
        if Path("/proc/meminfo").is_file():
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size) / 1e6
    except Exception:
        return 0.0


def memory_chunk_plan(runtime, requested=128, max_chunk=512,
                      budget_fraction=0.20, workers=1, available_mb=None):
    """Choose a larger bounded chunk from measured host/model memory facts.

    The output/log-softmax pair dominates bounded evaluation memory: both hold
    roughly one float32 value per vocabulary item and token.  Hidden and MLP
    intermediates are included conservatively.  The cap keeps hybrid attention
    quadratic work bounded even on very large hosts.
    """
    requested = max(1, int(requested))
    max_chunk = max(requested, int(max_chunk))
    workers = max(1, int(workers))
    available = (available_memory_mb() if available_mb is None
                 else float(available_mb))
    vocab = int(runtime.lm_head.shape[0])
    hidden = int(runtime.cfg.get("hidden", runtime.lm_head.shape[1]))
    intermediate = int(runtime.cfg.get("intermediate", 4 * hidden))
    # logits + shifted exponentials, plus conservative layer intermediates.
    bytes_per_token = 4 * (2 * vocab + 12 * hidden + 4 * intermediate)
    budget_bytes = max(0.0, available) * 1e6 * float(budget_fraction) / workers
    safe = int(budget_bytes // max(bytes_per_token, 1))
    # ``requested`` is the explicitly frozen, already-known-safe baseline.
    # Host telemetry is used only to admit a larger shape, never to silently
    # rewrite that baseline downward on platforms with weak memory reporting.
    target = min(max_chunk, max(requested, safe))
    # Power-of-two multiples produce stable, replayable GEMM shapes.
    selected = requested
    while selected * 2 <= target:
        selected *= 2
    return {
        "requested_chunk_size": requested,
        "selected_chunk_size": selected,
        "max_chunk_size": max_chunk,
        "available_memory_mb": float(available),
        "budget_fraction": float(budget_fraction),
        "workers": workers,
        "estimated_bytes_per_token": int(bytes_per_token),
        "estimated_chunk_working_mb": float(selected * bytes_per_token / 1e6),
    }


def parallel_evaluation_feasible(model_mb, installed_mb, available_mb,
                                 reserve_fraction=0.30):
    """Conservative auto-mode admission for two read-only evaluator workers."""
    available_mb = float(available_mb)
    # Room for both checkpoints, lazy decode/BLAS scratch, and two 512-token
    # vocabulary workspaces.  The fixed reserve covers Python/Torch and the OS.
    needed_mb = 2.5 * (float(model_mb) + float(installed_mb)) + 6144.0
    return available_mb > 0 and needed_mb <= available_mb * (1.0 - reserve_fraction)


def token_losses(logits, targets):
    # Keep the tokens x vocabulary slab in its source dtype.  Converting the
    # entire Qwen slab to float64 doubled peak memory; only the row reductions
    # need float64 accumulation for stable evidence.
    logits = np.asarray(logits)
    targets = np.asarray(targets, np.int64)
    maximum = logits.max(axis=-1, keepdims=True)
    shifted = np.exp(logits - maximum)
    lse = np.log(shifted.sum(axis=-1, dtype=np.float64)) + maximum.ravel()
    return np.asarray(lse - logits[np.arange(len(targets)), targets], np.float64)


def streamed_losses(runtime, token_ids, chunk_size=128, max_chunk_size=None,
                    memory_budget_fraction=0.20, workers=1, phase="evaluation",
                    on_chunk=None, stop_event=None, frozen_chunk_plan=None):
    """Return exact per-token losses while exposing every completed chunk."""
    ids = list(map(int, token_ids))
    if len(ids) < 2:
        raise ValueError("streamed measurement needs at least two token ids")
    plan = (dict(frozen_chunk_plan) if frozen_chunk_plan is not None else
            memory_chunk_plan(
                runtime, requested=chunk_size,
                max_chunk=(chunk_size if max_chunk_size is None else max_chunk_size),
                budget_fraction=memory_budget_fraction, workers=workers))
    selected = int(plan["selected_chunk_size"])
    state = None
    losses = []
    evaluated = 0
    started = time.monotonic()
    # The first call consumes one context token plus N target tokens.  Every
    # resumed call consumes N more targets, so progress lands on exact looks.
    cursor = 0
    total = len(ids) - 1
    while evaluated < total:
        if stop_event is not None and stop_event.is_set():
            break
        # Start at the known-safe baseline; subsequent calls use the common,
        # parent-admitted shape frozen for both paired evaluators.
        planned = int(plan["requested_chunk_size"]) if evaluated == 0 else selected
        target_count = min(planned, total - evaluated)
        if state is None:
            chunk = ids[:target_count + 1]
            logits, state = runtime.forward(chunk, collect_state=True)
            chunk_losses = token_losses(logits[:-1], chunk[1:])
            cursor = target_count + 1
        else:
            chunk = ids[cursor:cursor + target_count]
            boundary_logits = np.asarray(state.logits)
            logits, state = runtime.forward(chunk, collect_state=True, resume=state)
            prediction = np.concatenate([boundary_logits[None, :], logits[:-1]], axis=0)
            chunk_losses = token_losses(prediction, chunk)
            cursor += target_count
        start = evaluated
        evaluated += len(chunk_losses)
        losses.append(chunk_losses)
        elapsed = time.monotonic() - started
        record = {
            "phase": phase,
            "chunk_index": len(losses) - 1,
            "eval_start": start,
            "eval_stop": evaluated,
            "eval_tokens_complete": evaluated,
            "eval_tokens_total": total,
            "chunk_size": len(chunk_losses),
            "selected_chunk_size": selected,
            "elapsed_seconds": elapsed,
            "tokens_per_second": evaluated / max(elapsed, 1e-12),
            "peak_rss_mb": peak_rss_mb("RUSAGE_SELF"),
        }
        if on_chunk is not None:
            on_chunk(record, np.asarray(chunk_losses, np.float64))
        else:
            log("  %s measured %d/%d evaluation tokens" %
                (phase, evaluated, total))
    if not losses:
        raise RuntimeError("evaluation stopped before the first chunk completed")
    return np.concatenate(losses), plan


def streamed_measure(runtime, token_ids, chunk_size=128, resamples=800,
                     max_chunk_size=None, memory_budget_fraction=0.20,
                     recorder=None, phase="evaluation"):
    """Measure thousands of tokens without materializing all vocabulary logits."""
    from holographic.io_and_interop.holographic_measure import summarize_nll

    def report(record, _losses):
        if recorder is not None:
            recorder.emit("evaluation_chunk", **record)

    losses, _plan = streamed_losses(
        runtime, token_ids, chunk_size=chunk_size,
        max_chunk_size=max_chunk_size,
        memory_budget_fraction=memory_budget_fraction, phase=phase,
        on_chunk=report if recorder is not None else None)
    return summarize_nll(losses, resamples=resamples, seed=0)


def sequential_rejection_test(installed_nll, original_nll, regression_limit,
                              look_index, total_looks, family_alpha=0.05,
                              resamples=10000):
    """Multiplicity-corrected paired interim test; it may only emit NO-GO.

    Each frozen look spends alpha/K.  Passing is deliberately impossible at an
    interim look: every GO still requires the complete preregistered token set
    and the unchanged final 95% paired block interval.
    """
    from holographic.io_and_interop.holographic_measure import (
        _block_shape, better_than)

    total_looks = max(1, int(total_looks))
    alpha_each = float(family_alpha) / total_looks
    installed_nll = np.asarray(installed_nll)
    original_nll = np.asarray(original_nll)
    if len(installed_nll) != len(original_nll):
        raise ValueError("sequential look requires paired losses")
    _tau, proposed_block = _block_shape(installed_nll - original_nll)
    if proposed_block > len(installed_nll) // 2:
        # With fewer than two legal dependence blocks, a moving-block
        # bootstrap cannot estimate sampling uncertainty. Never substitute an
        # IID interval or a one-block degenerate interval at an early boundary.
        return {
            "look_index": int(look_index),
            "eval_tokens": int(len(installed_nll)),
            "alpha_spent": 0.0,
            "lower_bound_nats": None,
            "regression_limit_nats": float(regression_limit),
            "reject": False,
            "block": int(proposed_block),
            "effective_n": 1,
            "skipped": True,
            "reason": "fewer than two dependence blocks",
        }
    # better_than returns a two-sided interval, so alpha=2*alpha_each places
    # its lower endpoint at the one-sided Bonferroni boundary alpha_each.
    interim = better_than(
        {"nll": np.asarray(installed_nll),
         "perplexity": float(np.exp(np.mean(installed_nll)))},
        {"nll": np.asarray(original_nll),
         "perplexity": float(np.exp(np.mean(original_nll)))},
        alpha=min(0.999, 2.0 * alpha_each), seed=1000 + int(look_index),
        resamples=max(10000, int(resamples)))
    return {
        "look_index": int(look_index),
        "eval_tokens": int(len(installed_nll)),
        "alpha_spent": alpha_each,
        "lower_bound_nats": float(interim["ci_lo_nats"]),
        "regression_limit_nats": float(regression_limit),
        "reject": bool(interim["ci_lo_nats"] > float(regression_limit)),
        "block": int(interim["block"]),
        "effective_n": int(interim["effective_n"]),
        "skipped": False,
    }


def parse_sequential_looks(value, total_tokens):
    looks = {int(item.strip()) for item in str(value).split(",") if item.strip()}
    if any(item < 1 for item in looks):
        raise ValueError("sequential looks must be positive token counts")
    if any(item < 1000 and item < int(total_tokens) for item in looks):
        raise ValueError("interim sequential looks must include at least 1000 paired tokens")
    return sorted({item for item in looks if item <= int(total_tokens)} |
                  {int(total_tokens)})


def _measurement_worker(model_dir, token_ids, phase, chunk_size, max_chunk_size,
                        memory_budget_fraction, workers, gdn_backend,
                        frozen_chunk_plan, out_queue, stop_event):
    try:
        os.environ["LECORE_GDN_BACKEND"] = str(gdn_backend)
        from holographic.io_and_interop.holographic_gdnruntime import load_runtime
        runtime, _ = load_runtime(model_dir)

        def send_chunk(record, losses):
            out_queue.put({"kind": "chunk", "phase": phase,
                           "record": record, "losses": losses})

        losses, plan = streamed_losses(
            runtime, token_ids, chunk_size=chunk_size,
            max_chunk_size=max_chunk_size,
            memory_budget_fraction=memory_budget_fraction, workers=workers,
            phase=phase, on_chunk=send_chunk, stop_event=stop_event,
            frozen_chunk_plan=frozen_chunk_plan)
        acceleration = (runtime.acceleration_report()
                        if hasattr(runtime, "acceleration_report") else None)
        out_queue.put({"kind": "done", "phase": phase, "plan": plan,
                       "n_tokens": len(losses),
                       "peak_rss_mb": peak_rss_mb("RUSAGE_SELF"),
                       "acceleration": acceleration})
    except BaseException as exc:
        out_queue.put({"kind": "error", "phase": phase,
                       "error": "%s: %s" % (type(exc).__name__, exc)})


def paired_evaluation(model_dir, installed_dir, token_ids, chunk_size,
                      max_chunk_size, memory_budget_fraction, recorder,
                      sequential_looks, allow_early_rejection,
                      regression_limit, gdn_backend="numpy", parallel=True):
    """Evaluate one frozen token stream, concurrently when memory admits it."""
    from holographic.io_and_interop.holographic_gdnruntime import load_runtime
    from holographic.io_and_interop.holographic_measure import summarize_nll

    phases = (("original", model_dir), ("installed", installed_dir))
    buffers = {name: [] for name, _path in phases}
    plans = {}
    worker_peaks = {}
    acceleration_reports = {}
    looks = []
    early_at = None

    # Choose one schedule before launching workers.  Independent MemAvailable
    # samples could otherwise select different resume boundaries, introducing
    # floating-point chunking noise into a supposedly paired loss difference.
    with selected_gdn_backend("numpy"):
        planning_runtime, _ = load_runtime(model_dir)
    common_chunk_plan = memory_chunk_plan(
        planning_runtime, requested=chunk_size, max_chunk=max_chunk_size,
        budget_fraction=memory_budget_fraction, workers=(2 if parallel else 1))
    del planning_runtime
    gc.collect()
    recorder.emit("evaluation_chunk_plan", **common_chunk_plan)

    def maybe_look():
        nonlocal early_at
        if early_at is not None:
            return True
        common = min(sum(len(item) for item in buffers["original"]),
                     sum(len(item) for item in buffers["installed"]))
        completed_looks = {item["eval_tokens"] for item in looks}
        for index, target in enumerate(sequential_looks):
            if target > common or target in completed_looks:
                continue
            original = np.concatenate(buffers["original"])[:target]
            installed = np.concatenate(buffers["installed"])[:target]
            verdict = sequential_rejection_test(
                installed, original, regression_limit, index,
                len(sequential_looks))
            looks.append(verdict)
            recorder.emit("sequential_look", **verdict)
            if (verdict["reject"] and allow_early_rejection and
                    target < sequential_looks[-1]):
                early_at = target
                return True
        return False

    if parallel:
        context = multiprocessing.get_context("spawn")
        out_queue = context.Queue()
        stop_event = context.Event()
        processes = []
        for phase, path in phases:
            process = context.Process(
                target=_measurement_worker,
                args=(str(path), token_ids, phase, chunk_size, max_chunk_size,
                      memory_budget_fraction, 2, gdn_backend,
                      common_chunk_plan, out_queue, stop_event))
            process.start()
            processes.append((phase, process))
        done = set()
        try:
            while len(done) < len(phases):
                try:
                    message = out_queue.get(timeout=1.0)
                except queue.Empty:
                    missing = [phase for phase, process in processes
                               if not process.is_alive() and phase not in done]
                    if missing:
                        raise RuntimeError(
                            "evaluation worker exited without complete evidence: %s" %
                            ", ".join(missing))
                    continue
                if message["kind"] == "error":
                    raise RuntimeError("%s evaluation failed: %s" %
                                       (message["phase"], message["error"]))
                if message["kind"] == "chunk":
                    phase = message["phase"]
                    buffers[phase].append(np.asarray(message["losses"], np.float64))
                    recorder.emit("evaluation_chunk", **message["record"])
                    if maybe_look():
                        stop_event.set()
                elif message["kind"] == "done":
                    phase = message["phase"]
                    received = sum(len(item) for item in buffers[phase])
                    if received != int(message["n_tokens"]):
                        raise RuntimeError(
                            "%s evaluation evidence count mismatch: %d != %d" %
                            (phase, received, int(message["n_tokens"])))
                    if message["plan"] != common_chunk_plan:
                        raise RuntimeError(
                            "%s evaluator departed from the frozen chunk plan" %
                            phase)
                    plans[phase] = message["plan"]
                    worker_peaks[phase] = float(message["peak_rss_mb"])
                    acceleration_reports[phase] = message.get("acceleration")
                    done.add(phase)
        finally:
            stop_event.set()
            for _phase, process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
    else:
        for phase, path in phases:
            with selected_gdn_backend(gdn_backend):
                runtime, _ = load_runtime(path)
            stop_event = threading.Event()

            def collect(record, losses, phase=phase):
                buffers[phase].append(np.asarray(losses, np.float64))
                recorder.emit("evaluation_chunk", **record)
                if maybe_look():
                    stop_event.set()

            losses, plans[phase] = streamed_losses(
                runtime, token_ids, chunk_size=chunk_size,
                max_chunk_size=max_chunk_size,
                memory_budget_fraction=memory_budget_fraction, workers=1,
                phase=phase, on_chunk=collect, stop_event=stop_event,
                frozen_chunk_plan=common_chunk_plan)
            worker_peaks[phase] = peak_rss_mb("RUSAGE_SELF")
            acceleration_reports[phase] = (
                runtime.acceleration_report()
                if hasattr(runtime, "acceleration_report") else None)
            # The original must be complete before paired serial looks exist.
            # The installed phase can honor an early rejection between chunks.
            del runtime
            gc.collect()
            if early_at is not None:
                break

    original_losses = np.concatenate(buffers["original"])
    installed_losses = np.concatenate(buffers["installed"])
    common = min(len(original_losses), len(installed_losses))
    cutoff = min(common, early_at) if early_at is not None else common
    original_losses = original_losses[:cutoff]
    installed_losses = installed_losses[:cutoff]
    return {
        "before": summarize_nll(original_losses, resamples=800, seed=0),
        "after": summarize_nll(installed_losses, resamples=800, seed=0),
        "parallel": bool(parallel),
        "common_chunk_plan": common_chunk_plan,
        "plans": plans,
        "worker_peak_rss_mb": worker_peaks,
        "acceleration_reports": acceleration_reports,
        "sequential_looks": looks,
        "early_rejection_at": early_at,
    }


def load_reference(model_dir):
    import torch
    try:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float32, trust_remote_code=True).eval()
    except Exception as first:
        try:
            from transformers import Qwen3_5ForConditionalGeneration
            return Qwen3_5ForConditionalGeneration.from_pretrained(
                model_dir, torch_dtype=torch.float32).eval()
        except Exception:
            raise RuntimeError("official Transformers text model did not load") from first


def reference_parity(model_dir, token_ids, tolerance):
    import torch
    from transformers import AutoTokenizer
    from holographic.io_and_interop.holographic_bpe import BPE
    from holographic.io_and_interop.holographic_gdnruntime import load_runtime

    prompt = "The holographic engine binds and bundles hypervectors."
    official = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    ref_ids = list(official.encode(prompt, add_special_tokens=False))
    own_ids = BPE.from_dir(model_dir).encode(prompt)
    tokenizer_pass = ref_ids == own_ids and len(ref_ids) >= 2
    probe = ref_ids[: min(24, len(ref_ids))] if tokenizer_pass else list(token_ids[:24])
    runtime, _ = load_runtime(model_dir)
    ours = np.asarray(runtime.forward(probe), np.float64)
    model = load_reference(model_dir)
    with torch.no_grad():
        reference = model(input_ids=torch.tensor([probe], dtype=torch.long)).logits[0]
    reference = reference.detach().float().cpu().numpy()
    rel = float(np.max(np.abs(ours - reference)) /
                max(float(np.max(np.abs(reference))), 1e-12))
    del model, runtime, ours, reference
    gc.collect()
    return float(tokenizer_pass), rel, float(rel <= float(tolerance))


def run_installer(model_dir, installed_dir, installation_corpus, transcript):
    command = [
        sys.executable,
        str(REPO / "assimilation" / "install.py"),
        "--experimental",
        str(model_dir),
        str(installed_dir),
        "--doc",
        str(installation_corpus),
        "--device",
        "cpu",
    ]
    log("running experimental installer; transcript: %s" % transcript)
    transcript.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    # Installation remains the preregistered NumPy treatment.  The optional C
    # recurrence is an evaluation accelerator, not an unrecorded model change.
    environment["LECORE_GDN_BACKEND"] = "numpy"
    with transcript.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=REPO, stdout=handle,
                                   stderr=subprocess.STDOUT, check=False,
                                   env=environment)
    if completed.returncode != 0:
        tail = transcript.read_text(encoding="utf-8", errors="replace")[-4000:]
        log(tail)
        raise RuntimeError("experimental installer exited %d" % completed.returncode)


def official_dependency_preflight():
    """Fail before model work when the frozen official smoke stack is absent."""
    from importlib import metadata

    found = {}
    failures = []
    for distribution, expected in OFFICIAL_DEPENDENCY_VERSIONS.items():
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            failures.append("%s is not installed" % distribution)
            continue
        found[distribution] = version
        if version.split("+", 1)[0] != expected:
            failures.append("%s==%s, expected %s" %
                            (distribution, version, expected))
    try:
        import torchvision  # noqa: F401
    except Exception as exc:
        failures.append("torchvision import failed: %s: %s" %
                        (type(exc).__name__, str(exc)[:300]))
    if failures:
        raise RuntimeError(
            "official dependency preflight failed: %s; install %s" %
            ("; ".join(failures),
             HERE / "requirements-cpu.txt"))
    return found


def official_output_smokes(installed_dir):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoTokenizer
    diagnostics = []
    model = None
    processor = None
    tokenizer = None
    text_inputs = text_out = vision_inputs = vision_out = None
    official_reload = 0
    text_pass = 0
    vision_pass = 0

    def checked_load(model_class, **kwargs):
        loaded = model_class.from_pretrained(
            installed_dir, output_loading_info=True, **kwargs)
        model_value, info = loaded
        incompatible = {
            key: info.get(key) or []
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys",
                        "error_msgs")
            if info.get(key)
        }
        if incompatible:
            del model_value
            gc.collect()
            raise RuntimeError("official state-dict incompatibility: %s" %
                               json.dumps(incompatible, default=str)[:4000])
        return model_value.eval()

    try:
        from transformers import AutoModelForImageTextToText
        model = checked_load(AutoModelForImageTextToText,
                             dtype=torch.float32,
                             trust_remote_code=True)
        official_reload = 1
    except Exception as auto_exc:
        diagnostics.append("AutoModelForImageTextToText: %s: %s" %
                           (type(auto_exc).__name__, str(auto_exc)[:1000]))
        try:
            from transformers import Qwen3_5ForConditionalGeneration
            model = checked_load(Qwen3_5ForConditionalGeneration,
                                 dtype=torch.float32)
            official_reload = 1
        except Exception as qwen_exc:
            diagnostics.append("Qwen3_5ForConditionalGeneration: %s: %s" %
                               (type(qwen_exc).__name__, str(qwen_exc)[:1000]))

    if model is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                installed_dir, trust_remote_code=True)
        except Exception as exc:
            diagnostics.append("AutoTokenizer: %s: %s" %
                               (type(exc).__name__, str(exc)[:1000]))
        try:
            processor = AutoProcessor.from_pretrained(
                installed_dir, trust_remote_code=True)
        except Exception as exc:
            diagnostics.append("AutoProcessor: %s: %s" %
                               (type(exc).__name__, str(exc)[:1000]))

    # Text generation is an independent official gate. V3 coupled it to
    # AutoProcessor, so a missing vision-only dependency incorrectly prevented
    # the text smoke from running at all.
    if model is not None and tokenizer is not None:
        try:
            text_inputs = tokenizer(
                "Explain what a checksum protects.", return_tensors="pt")
            with torch.no_grad():
                text_out = model.generate(**text_inputs, max_new_tokens=4)
            text_pass = int(
                text_out.shape[-1] > text_inputs["input_ids"].shape[-1])
        except Exception as exc:
            diagnostics.append("text generation: %s: %s" %
                               (type(exc).__name__, str(exc)[:1000]))

    if model is not None and processor is not None:
        try:
            image = Image.new("RGB", (32, 32), color=(32, 96, 160))
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Name the dominant color."},
            ]}]
            vision_inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt")
            with torch.no_grad():
                vision_out = model.generate(**vision_inputs, max_new_tokens=1)
            vision_pass = int(
                vision_out.shape[-1] > vision_inputs["input_ids"].shape[-1])
        except Exception as exc:
            diagnostics.append("vision generation: %s: %s" %
                               (type(exc).__name__, str(exc)[:1000]))

    peak_gpu = (float(torch.cuda.max_memory_allocated()) / 1e6
                if torch.cuda.is_available() else 0.0)
    del model, processor, tokenizer, text_inputs, text_out, vision_inputs, vision_out
    gc.collect()
    return (float(official_reload), float(text_pass), float(vision_pass),
            peak_gpu, diagnostics)


def source_snapshot():
    repository = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], cwd=REPO, text=True).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO, text=True).strip())
    paths = [
        HERE / "run.py",
        HERE / "contract.py",
        HERE / "generate.py",
        REPO / "assimilation" / "install.py",
        REPO / "holographic" / "io_and_interop" / "holographic_install_lecore.py",
        REPO / "holographic" / "io_and_interop" / "holographic_hrnngrow.py",
        REPO / "holographic" / "agents_and_reasoning" / "holographic_memsearch.py",
        REPO / "holographic" / "io_and_interop" / "holographic_measure.py",
        REPO / "holographic" / "io_and_interop" / "holographic_gdnruntime.py",
        REPO / "holographic" / "io_and_interop" / "holographic_gdnaccel.py",
        REPO / "holographic" / "io_and_interop" / "holographic_ccrun.py",
        REPO / "holographic" / "io_and_interop" / "holographic_emit.py",
        HERE / "requirements-cpu.txt",
    ]
    return float(not dirty), {
        "repository": repository,
        "commit": commit,
        "artifacts": [{"path": str(path.relative_to(REPO)),
                       "sha256": sha256_file(path)} for path in paths],
    }


def frozen_input_snapshot(model_dir, installation_corpus, evaluation_corpus):
    model_digest, model_files = model_manifest(model_dir)
    return {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "model": {
            "path": str(model_dir),
            "manifest_sha256": model_digest,
            "files": model_files,
        },
        "installation_corpus": {
            "path": str(installation_corpus),
            "sha256": sha256_file(installation_corpus),
            "bytes": installation_corpus.stat().st_size,
        },
        "evaluation_corpus": {
            "path": str(evaluation_corpus),
            "sha256": sha256_file(evaluation_corpus),
            "bytes": evaluation_corpus.stat().st_size,
        },
    }


def require_frozen_input_identity(snapshot, expected):
    observed = {
        "source_commit": snapshot["source_commit"],
        "model_digest": snapshot["model"]["manifest_sha256"],
        "installation_sha256": snapshot["installation_corpus"]["sha256"],
        "evaluation_sha256": snapshot["evaluation_corpus"]["sha256"],
    }
    drift = {name: {"expected": expected[name], "observed": observed[name]}
             for name in expected if observed[name] != expected[name]}
    if drift:
        raise ValueError("frozen experiment inputs drifted: %s" %
                         json.dumps(drift, sort_keys=True))
    return observed


def runtime_provenance_snapshot(frozen_inputs, installed_dir):
    """Record extended run inputs and checkpoint bytes outside ilxyr source."""
    return {
        "schema": RUNTIME_PROVENANCE_SCHEMA,
        "inputs": frozen_inputs,
        "emitted_checkpoint": emitted_checkpoint_snapshot(installed_dir),
    }


def emitted_checkpoint_snapshot(installed_dir):
    """Bind every emitted byte without embedding model weights in evidence."""
    root = Path(installed_dir)
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append({"path": str(path.relative_to(root)),
                      "sha256": sha256_file(path),
                      "bytes": path.stat().st_size,
                      "model_weights": path.suffix == ".safetensors"})
    return {"path": str(root), "files": files,
            "total_bytes": sum(item["bytes"] for item in files)}


def argument_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("installed_dir", type=Path)
    ap.add_argument("installation_corpus", type=Path,
                    help="text used only to ground the experimental installation")
    ap.add_argument("evaluation_corpus", type=Path,
                    help="separate held-out text used only for paired evaluation")
    ap.add_argument("--expected-source-commit", required=True)
    ap.add_argument("--expected-model-digest", required=True)
    ap.add_argument("--expected-installation-sha256", required=True)
    ap.add_argument("--expected-evaluation-sha256", required=True)
    ap.add_argument("--min-tokens", type=int, default=4096)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--max-chunk-size", type=int, default=512)
    ap.add_argument("--memory-budget-fraction", type=float, default=0.20)
    ap.add_argument("--evaluation-mode", choices=("auto", "serial", "parallel"),
                    default="auto")
    ap.add_argument("--gdn-backend", choices=("numpy", "c"),
                    help="default: LECORE_GDN_BACKEND, otherwise numpy")
    ap.add_argument("--require-native-gdn", action="store_true",
                    help="make complete parity-gated native GDN evidence a mandatory acceptance gate")
    ap.add_argument("--progress-upload-uri",
                    help="optional s3:// URI refreshed after every evaluation chunk")
    ap.add_argument("--sequential-looks", default="1024,2048,3072,4096")
    ap.add_argument("--allow-early-rejection", action="store_true",
                    help="honor frozen multiplicity-corrected interim NO-GO looks")
    ap.add_argument("--max-regression", type=float, default=0.01)
    ap.add_argument("--logit-tolerance", type=float, default=1e-3)
    return ap


def _main(argv=None, metric_stdout=None):
    if metric_stdout is None:
        metric_stdout = sys.stdout
    args = argument_parser().parse_args(argv)

    model_dir = args.model_dir.resolve()
    installed_dir = args.installed_dir.resolve()
    installation_corpus = args.installation_corpus.resolve()
    evaluation_corpus = args.evaluation_corpus.resolve()
    if not model_dir.is_dir():
        raise SystemExit("model_dir does not exist: %s" % model_dir)
    if not installation_corpus.is_file() or not evaluation_corpus.is_file():
        raise SystemExit("both installation and evaluation corpora must exist")
    installation_digest = sha256_file(installation_corpus)
    evaluation_digest = sha256_file(evaluation_corpus)
    if installation_digest == evaluation_digest:
        raise SystemExit("installation and evaluation corpora must have distinct contents")
    if installed_dir.exists() and any(installed_dir.iterdir()):
        raise SystemExit("installed_dir must be absent or empty: %s" % installed_dir)
    if int(args.min_tokens) < 1000:
        raise SystemExit("--min-tokens must be at least 1000 for an acceptance run")
    if args.chunk_size < 1 or args.max_chunk_size < args.chunk_size:
        raise SystemExit("chunk sizes must be positive and max must cover initial")
    if not 0 < args.memory_budget_fraction <= 0.5:
        raise SystemExit("--memory-budget-fraction must be in (0, 0.5]")
    try:
        sequential_looks = parse_sequential_looks(
            args.sequential_looks, args.min_tokens)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.progress_upload_uri and not args.progress_upload_uri.startswith("s3://"):
        raise SystemExit("--progress-upload-uri must be an s3:// URI")
    gdn_backend = args.gdn_backend or os.environ.get("LECORE_GDN_BACKEND", "numpy")
    if gdn_backend not in ("numpy", "c"):
        raise SystemExit("LECORE_GDN_BACKEND must be numpy or c")
    if args.require_native_gdn and gdn_backend != "c":
        raise SystemExit("--require-native-gdn requires --gdn-backend c")
    artifact_dir = installed_dir.parent / (installed_dir.name + ".acceptance")
    progress_upload_uri = (args.progress_upload_uri or
                           os.environ.get("LECORE_PROGRESS_UPLOAD_URI"))
    if progress_upload_uri and not progress_upload_uri.startswith("s3://"):
        raise SystemExit("LECORE_PROGRESS_UPLOAD_URI must be an s3:// URI")
    recorder = ProgressRecorder(
        artifact_dir / "progress.jsonl", upload_uri=progress_upload_uri)
    recorder.emit(
        "run_policy", min_tokens=int(args.min_tokens),
        initial_chunk_size=int(args.chunk_size),
        max_chunk_size=int(args.max_chunk_size),
        memory_budget_fraction=float(args.memory_budget_fraction),
        evaluation_mode=args.evaluation_mode,
        gdn_backend=gdn_backend,
        require_native_gdn=bool(args.require_native_gdn),
        reference_backend="numpy",
        installation_backend="numpy",
        progress_upload_uri=progress_upload_uri,
        sequential_looks=sequential_looks,
        sequential_family_alpha=0.05,
        allow_early_rejection=bool(args.allow_early_rejection),
        allow_early_acceptance=False)

    with recorder.stage("official_dependency_preflight"):
        dependency_versions = official_dependency_preflight()
    recorder.emit("official_dependencies", versions=dependency_versions)
    with recorder.stage("frozen_input_identity"):
        frozen_inputs = frozen_input_snapshot(
            model_dir, installation_corpus, evaluation_corpus)
        try:
            require_frozen_input_identity(frozen_inputs, {
                "source_commit": args.expected_source_commit,
                "model_digest": args.expected_model_digest,
                "installation_sha256": args.expected_installation_sha256,
                "evaluation_sha256": args.expected_evaluation_sha256,
            })
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    from holographic.io_and_interop.holographic_bpe import BPE
    from holographic.io_and_interop.holographic_gdnruntime import load_runtime
    from holographic.io_and_interop.holographic_measure import better_than

    text = evaluation_corpus.read_text(encoding="utf-8", errors="replace")
    token_ids = BPE.from_dir(model_dir).encode(text)
    needed = int(args.min_tokens) + 1
    if len(token_ids) < needed:
        raise SystemExit("evaluation corpus produced %d tokens; acceptance requires at least %d"
                         % (len(token_ids), needed))
    token_ids = token_ids[:needed]

    with recorder.stage("source_snapshot"):
        source_clean, source = source_snapshot()
    with recorder.stage("reference_parity"):
        with selected_gdn_backend("numpy"):
            tok_pass, ref_error, ref_pass = reference_parity(
                model_dir, token_ids, args.logit_tolerance)
    if not source_clean:
        raise RuntimeError("refusing installation from a dirty source checkout")
    if not tok_pass:
        raise RuntimeError("refusing installation after tokenizer parity failed")
    if not ref_pass:
        raise RuntimeError(
            "refusing installation after reference-logit parity failed "
            "(relative error %.9g > %.9g)" %
            (ref_error, float(args.logit_tolerance)))

    with recorder.stage("experimental_installation"):
        run_installer(model_dir, installed_dir, installation_corpus,
                      artifact_dir / "install.log")
    child_peak = peak_rss_mb("RUSAGE_CHILDREN")

    with recorder.stage("lecore_reload"):
        with selected_gdn_backend(gdn_backend):
            installed, _ = load_runtime(installed_dir)
        reload_logits = np.asarray(installed.forward(token_ids[:16]))
        reload_pass = float(np.isfinite(reload_logits).all())
        reload_acceleration = (installed.acceleration_report()
                               if hasattr(installed, "acceleration_report") else None)
        del installed, reload_logits
        gc.collect()

    model_mb = sum(path.stat().st_size for path in model_dir.glob("*.safetensors")) / 1e6
    checkpoint_mb = sum(path.stat().st_size for path in
                        installed_dir.glob("*.safetensors")) / 1e6
    memory_available = available_memory_mb()
    auto_parallel = parallel_evaluation_feasible(
        model_mb, checkpoint_mb, memory_available)
    use_parallel = (args.evaluation_mode == "parallel" or
                    (args.evaluation_mode == "auto" and auto_parallel))
    recorder.emit(
        "evaluation_admission", requested_mode=args.evaluation_mode,
        parallel=use_parallel, auto_parallel_feasible=auto_parallel,
        available_memory_mb=memory_available, model_checkpoint_mb=model_mb,
        installed_checkpoint_mb=checkpoint_mb)
    regression_limit = math.log1p(float(args.max_regression))
    with recorder.stage("paired_evaluation"):
        evaluation = paired_evaluation(
            model_dir, installed_dir, token_ids,
            chunk_size=args.chunk_size, max_chunk_size=args.max_chunk_size,
            memory_budget_fraction=args.memory_budget_fraction,
            recorder=recorder, sequential_looks=sequential_looks,
            allow_early_rejection=args.allow_early_rejection,
            regression_limit=regression_limit, gdn_backend=gdn_backend,
            parallel=use_parallel)
    parent_evaluation_peak = peak_rss_mb("RUSAGE_SELF")
    before, after = evaluation["before"], evaluation["after"]

    comparison = better_than(after, before, resamples=1200, seed=0)
    full_evaluation_pass = float(
        comparison["n_tokens"] == int(args.min_tokens))
    sequential_early_rejection = float(
        evaluation["early_rejection_at"] is not None)
    def complete_native_evidence(report):
        recurrence = (report or {}).get("full_sequence_gdn_recurrence", {})
        checks = recurrence.get("validated_regimes") or []
        return bool(
            recurrence.get("requested") == "c" and
            recurrence.get("active") == "c" and
            recurrence.get("refused") is None and
            int(recurrence.get("native_calls", 0)) > 0 and
            int(recurrence.get("native_tokens", 0)) > 0 and
            {bool(row.get("resumed")) for row in checks if row.get("passed")} ==
                {False, True} and
            all(row.get("passed") for row in checks))

    native_gdn_active = float(
        gdn_backend == "c" and
        len(evaluation["acceleration_reports"]) == 2 and
        all(complete_native_evidence(report)
            for report in evaluation["acceleration_reports"].values()))
    # Interim looks can establish NO-GO only.  GO remains tied to the unchanged
    # final 4096-position upper interval (or the explicitly frozen min_tokens).
    statistical_pass = float(
        full_evaluation_pass >= 1.0 and
        comparison["ci_hi_nats"] <= regression_limit)
    with recorder.stage("official_output_smokes"):
        official_reload, text_pass, vision_pass, peak_gpu, smoke_diagnostics = \
            official_output_smokes(installed_dir)
    runtime_provenance = runtime_provenance_snapshot(
        frozen_inputs, installed_dir)
    provenance_path = artifact_dir / "runtime-provenance.json"
    provenance_path.write_text(
        json.dumps(runtime_provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    source["artifacts"].append({
        "path": str(provenance_path),
        "sha256": sha256_file(provenance_path),
    })

    worker_peaks = list(evaluation["worker_peak_rss_mb"].values())
    evaluation_aggregate_peak_upper = (
        parent_evaluation_peak + sum(worker_peaks)
        if evaluation["parallel"] else parent_evaluation_peak)
    peak_rss = max(peak_rss_mb("RUSAGE_SELF"), child_peak,
                   evaluation_aggregate_peak_upper)
    required = [source_clean, tok_pass, ref_pass, full_evaluation_pass,
                statistical_pass,
                reload_pass, official_reload, text_pass, vision_pass]
    metrics = {
        "acceptance_pass": acceptance_gate_pass(
            required, native_gdn_active,
            require_native_gdn=bool(args.require_native_gdn)),
        "source_clean": source_clean,
        "spectral_filtering_enabled": 0.0,
        "experimental_installer_used": 1.0,
        "tokenizer_parity_pass": tok_pass,
        "reference_logit_parity_pass": ref_pass,
        "reference_logit_relative_error": ref_error,
        "eval_tokens": float(comparison["n_tokens"]),
        "full_evaluation_pass": full_evaluation_pass,
        "parallel_evaluation_used": float(evaluation["parallel"]),
        "sequential_early_rejection": sequential_early_rejection,
        "sequential_looks_completed": float(len(evaluation["sequential_looks"])),
        "evaluation_wall_seconds": float(recorder.timings["paired_evaluation"]),
        "native_gdn_acceleration_active": native_gdn_active,
        "original_perplexity": float(before["perplexity"]),
        "installed_perplexity": float(after["perplexity"]),
        "perplexity_delta_pct": float(comparison["delta_pct"]),
        "paired_ci_lo_nats": float(comparison["ci_lo_nats"]),
        "paired_ci_hi_nats": float(comparison["ci_hi_nats"]),
        "statistical_gate_pass": statistical_pass,
        "paired_block_length": float(comparison["block"]),
        "paired_effective_tokens": float(comparison["effective_n"]),
        "peak_rss_mb": float(peak_rss),
        "peak_gpu_mb": float(peak_gpu),
        "emitted_checkpoint_mb": float(checkpoint_mb),
        "reload_pass": reload_pass,
        "official_reload_pass": official_reload,
        "text_generation_pass": text_pass,
        "vision_smoke_pass": vision_pass,
    }
    if tuple(metrics) != METRIC_NAMES:
        raise RuntimeError("runner metric order/contract drift: %r != %r"
                           % (tuple(metrics), METRIC_NAMES))
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("non-finite acceptance metric")
    executor_output = build_executor_output(metrics, source)
    recorder.emit("run_complete", acceptance_pass=metrics["acceptance_pass"],
                  eval_tokens=metrics["eval_tokens"])
    recorder.flush_upload()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "source": source,
                    "runtime_provenance": runtime_provenance,
                    "diagnostics": {
                        "official_smokes": smoke_diagnostics,
                        "official_dependencies": dependency_versions,
                        "stage_timings_seconds": recorder.timings,
                        "progress_publication": {
                            "upload_uri": recorder.upload_uri,
                            "requests": recorder.upload_requests,
                            "attempts": recorder.upload_attempts,
                            "failures": recorder.upload_failures,
                            "local_path": str(recorder.path),
                            "sha256": sha256_file(recorder.path),
                        },
                        "evaluation": {
                            "common_chunk_plan": evaluation["common_chunk_plan"],
                            "plans": evaluation["plans"],
                            "worker_peak_rss_mb": evaluation["worker_peak_rss_mb"],
                            "parent_evaluation_peak_rss_mb": parent_evaluation_peak,
                            "aggregate_evaluation_peak_rss_upper_mb":
                                evaluation_aggregate_peak_upper,
                            "acceleration_reports": evaluation["acceleration_reports"],
                            "reload_acceleration": reload_acceleration,
                            "sequential_looks": evaluation["sequential_looks"],
                            "early_rejection_at": evaluation["early_rejection_at"],
                            "frozen_full_token_requirement": int(args.min_tokens),
                            "early_acceptance_allowed": False,
                        },
                    }},
                   indent=2, sort_keys=True))
    print(json.dumps(executor_output, sort_keys=True),
          file=metric_stdout, flush=True)
    return 0


def main(argv=None):
    # ilxyr accepts exactly one JSON document on stdout. Treat every diagnostic
    # from leCore, NumPy, Torch, Transformers, or a future dependency as stderr
    # by construction, then write only the final envelope to the saved stream.
    metric_stdout = sys.stdout
    with redirect_stdout(sys.stderr):
        return _main(argv, metric_stdout=metric_stdout)


if __name__ == "__main__":
    raise SystemExit(main())

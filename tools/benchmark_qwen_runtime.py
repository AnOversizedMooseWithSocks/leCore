#!/usr/bin/env python3
"""Benchmark the Qwen runtime's chunk-size, memory, and cost envelope.

The controller launches one fresh child process per chunk size so peak RSS is not
contaminated by an earlier trial.  Each child executes the same resumed-forward
and full-vocabulary NLL path used by the acceptance experiment.  The resulting
JSON is intended to be archived alongside an ilxyr run and consumed when
choosing the next AWS instance; it is not itself an acceptance result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


GIB = 1024 ** 3
MIB = 1024 ** 2
SCHEMA = "lecore.qwen_runtime_benchmark.v1"
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_positive_ints(value, option="value"):
    try:
        values = sorted(set(int(part.strip()) for part in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("%s must be comma-separated integers" % option) from exc
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("%s must contain positive integers" % option)
    return values


def parse_rates(value):
    """Parse a user-supplied RAM-class to hourly-USD map (for example 32=0.50)."""
    if not value:
        return {}
    rates = {}
    try:
        for item in value.split(","):
            ram, rate = item.split("=", 1)
            ram, rate = int(ram.strip()), float(rate.strip())
            if ram <= 0 or rate < 0 or not math.isfinite(rate):
                raise ValueError
            rates[ram] = rate
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "rates must look like 32=0.50,64=1.00,128=2.00") from exc
    return rates


def peak_rss_bytes():
    import resource

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and the AWS runner report KiB.
    return value if platform.system() == "Darwin" else value * 1024


def total_memory_bytes():
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _token_losses(logits, targets):
    import numpy as np

    # Match the acceptance runner: retain the tokens x vocabulary slab in its
    # source dtype and promote only the row reductions.  An eager float64 copy
    # would make the benchmark overstate the memory needed by the real path.
    values = np.asarray(logits)
    targets = np.asarray(targets, np.int64)
    maximum = values.max(axis=-1, keepdims=True)
    shifted = np.exp(values - maximum)
    lse = np.log(shifted.sum(axis=-1, dtype=np.float64)) + maximum.ravel()
    return np.asarray(
        lse - values[np.arange(len(targets)), targets], dtype=np.float64)


def _worker(model_dir, corpus, chunk_size, tokens, warmup_tokens):
    """Run one isolated measurement and return JSON-serializable evidence."""
    import numpy as np

    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from holographic.io_and_interop.holographic_bpe import BPE
    from holographic.io_and_interop.holographic_gdnruntime import load_runtime

    started = time.perf_counter()
    text = Path(corpus).read_text(encoding="utf-8", errors="replace")
    token_ids = list(map(int, BPE.from_dir(model_dir).encode(text)))
    required = int(tokens) + 1
    if len(token_ids) < required:
        raise ValueError("corpus produced %d tokens; benchmark needs %d" %
                         (len(token_ids), required))
    token_ids = token_ids[:required]

    load_started = time.perf_counter()
    runtime, _ = load_runtime(model_dir)
    load_seconds = time.perf_counter() - load_started
    loaded_peak = peak_rss_bytes()
    warmup = token_ids[:max(2, min(int(warmup_tokens), len(token_ids)))]
    runtime.forward(warmup)

    state = None
    losses = []
    eval_started = time.perf_counter()
    for start in range(0, len(token_ids), int(chunk_size)):
        chunk = token_ids[start:start + int(chunk_size)]
        if state is None:
            if len(chunk) < 2:
                continue
            logits, state = runtime.forward(chunk, collect_state=True)
            losses.append(_token_losses(logits[:-1], chunk[1:]))
        else:
            boundary_logits = np.asarray(state.logits)
            logits, state = runtime.forward(chunk, collect_state=True, resume=state)
            prediction = np.concatenate([boundary_logits[None, :], logits[:-1]], axis=0)
            losses.append(_token_losses(prediction, chunk))
    evaluation_seconds = time.perf_counter() - eval_started
    joined = np.ascontiguousarray(np.concatenate(losses), dtype=np.float64)
    if len(joined) != int(tokens):
        raise RuntimeError("measured %d losses instead of %d" % (len(joined), tokens))
    acceleration = (runtime.acceleration_report()
                    if hasattr(runtime, "acceleration_report") else None)
    return {
        "status": "ok",
        "chunk_size": int(chunk_size),
        "prediction_tokens": int(len(joined)),
        "load_seconds": float(load_seconds),
        "evaluation_seconds": float(evaluation_seconds),
        "tokens_per_second": float(len(joined) / max(evaluation_seconds, 1e-12)),
        "loaded_peak_rss_bytes": int(loaded_peak),
        "peak_rss_bytes": int(peak_rss_bytes()),
        "loss_mean_nats": float(joined.mean()),
        "losses_sha256_float64": hashlib.sha256(joined.tobytes()).hexdigest(),
        "acceleration": acceleration,
        # The controller consumes this to calculate true cross-chunk parity, then
        # removes it from the archived report.
        "_losses": joined.tolist(),
        "wall_seconds": float(time.perf_counter() - started),
    }


def _launch_worker(script, model_dir, corpus, chunk_size, tokens, warmup_tokens,
                   threads, gdn_backend):
    command = [
        sys.executable, str(script), "--worker", str(model_dir), str(corpus),
        "--chunk-size", str(chunk_size), "--tokens", str(tokens),
        "--warmup-tokens", str(warmup_tokens),
        "--gdn-backend", str(gdn_backend),
    ]
    env = os.environ.copy()
    for name in THREAD_ENV:
        env[name] = str(threads)
    env["LECORE_GDN_BACKEND"] = str(gdn_backend)
    completed = subprocess.run(command, cwd=script.parents[1], env=env,
                               capture_output=True, text=True, check=False)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0:
        return {
            "status": "error",
            "chunk_size": int(chunk_size),
            "returncode": int(completed.returncode),
            "stderr_tail": completed.stderr[-4000:],
            "stdout_tail": completed.stdout[-1000:],
        }
    try:
        return json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return {
            "status": "error",
            "chunk_size": int(chunk_size),
            "returncode": int(completed.returncode),
            "stderr_tail": completed.stderr[-4000:],
            "stdout_tail": completed.stdout[-1000:],
            "parse_error": str(exc),
        }


def memory_recommendation(results, ram_classes, concurrent_runtimes=2,
                          headroom_factor=1.25, system_reserve_gib=2.0,
                          full_run_peak_mib=0.0, selected_chunk_size=None):
    successful = [row for row in results if row.get("status") == "ok"]
    if not successful:
        return {"status": "unavailable", "reason": "no successful chunk-size trial"}
    max_trial_peak = max(int(row["peak_rss_bytes"]) for row in successful)
    selected = next((row for row in successful
                     if int(row["chunk_size"]) == int(selected_chunk_size)), None) \
        if selected_chunk_size is not None else None
    evaluation_peak = int(selected["peak_rss_bytes"]) if selected else max_trial_peak
    parallel_peak = evaluation_peak * int(concurrent_runtimes)
    observed_full_peak = int(float(full_run_peak_mib) * MIB)
    workload_peak = max(parallel_peak, observed_full_peak)
    required = int(math.ceil(workload_peak * float(headroom_factor) +
                             float(system_reserve_gib) * GIB))
    decisions = []
    recommended = None
    for ram in sorted(set(map(int, ram_classes))):
        capacity = ram * GIB
        fits = capacity >= required
        if fits and recommended is None:
            recommended = ram
        decisions.append({
            "ram_gib": ram,
            "fits": fits,
            "margin_gib": float((capacity - required) / GIB),
        })
    return {
        "status": "ok" if recommended is not None else "larger_class_required",
        "scope": "evaluation RSS plus optional observed full-run peak",
        "selected_chunk_size": (int(selected["chunk_size"]) if selected else None),
        "measured_single_runtime_peak_bytes": evaluation_peak,
        "max_observed_trial_peak_bytes": max_trial_peak,
        "projected_parallel_evaluation_peak_bytes": parallel_peak,
        "observed_full_run_peak_bytes": observed_full_peak,
        "planning_peak_bytes": workload_peak,
        "headroom_factor": float(headroom_factor),
        "system_reserve_gib": float(system_reserve_gib),
        "required_capacity_bytes": required,
        "concurrent_evaluation_runtimes": int(concurrent_runtimes),
        "recommended_ram_gib": recommended,
        "class_decisions": decisions,
    }


def annotate_parity(results, atol):
    successful = [row for row in results if row.get("status") == "ok"]
    if not successful:
        return None
    import numpy as np

    reference = np.asarray(successful[0].pop("_losses"), dtype=np.float64)
    successful[0]["max_abs_loss_delta_vs_smallest_chunk"] = 0.0
    successful[0]["numerical_parity"] = True
    for row in successful[1:]:
        values = np.asarray(row.pop("_losses"), dtype=np.float64)
        delta = float(np.max(np.abs(values - reference)))
        row["max_abs_loss_delta_vs_smallest_chunk"] = delta
        row["numerical_parity"] = bool(delta <= float(atol))
    return int(successful[0]["chunk_size"])


def speed_recommendation(results, parity_atol, eval_tokens_per_runtime,
                         eval_passes, concurrent_runtimes, fixed_overhead_minutes,
                         rates):
    candidates = [row for row in results
                  if row.get("status") == "ok" and row.get("numerical_parity")]
    if not candidates:
        return {"status": "unavailable", "reason": "no parity-preserving trial"}
    fastest = max(candidates, key=lambda row: float(row["tokens_per_second"]))
    waves = int(math.ceil(float(eval_passes) / max(int(concurrent_runtimes), 1)))
    evaluation_seconds = (waves * int(eval_tokens_per_runtime) /
                          float(fastest["tokens_per_second"]))
    duration = evaluation_seconds + float(fixed_overhead_minutes) * 60.0
    costs = [{
        "ram_gib": int(ram),
        "hourly_usd": float(rate),
        "estimated_compute_usd": float(rate * duration / 3600.0),
    } for ram, rate in sorted(rates.items())]
    return {
        "status": "ok",
        "recommended_chunk_size": int(fastest["chunk_size"]),
        "measured_tokens_per_second": float(fastest["tokens_per_second"]),
        "parity_atol": float(parity_atol),
        "eval_tokens_per_runtime": int(eval_tokens_per_runtime),
        "evaluation_passes": int(eval_passes),
        "concurrent_evaluation_runtimes": int(concurrent_runtimes),
        "execution_waves": waves,
        "fixed_overhead_minutes": float(fixed_overhead_minutes),
        "estimated_total_seconds": float(duration),
        "cost_estimates": costs,
        "cost_note": "Rates are caller-supplied and exclude storage, transfer, and taxes.",
    }


def annotate_cost_fit(speed, memory):
    """Join caller prices to the independently derived memory constraints."""
    decisions = {int(row["ram_gib"]): bool(row["fits"])
                 for row in memory.get("class_decisions", [])}
    eligible = []
    for row in speed.get("cost_estimates", []):
        row["fits_memory_projection"] = decisions.get(int(row["ram_gib"]), False)
        if row["fits_memory_projection"]:
            eligible.append(row)
    speed["least_estimated_cost_fitting_class_gib"] = (
        min(eligible, key=lambda row: row["estimated_compute_usd"])["ram_gib"]
        if eligible else None)


def model_manifest(model_dir, include_weight_hashes=True):
    names = {
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "model.safetensors.index.json",
    }
    paths = [path for path in Path(model_dir).iterdir()
             if path.is_file() and (path.name in names or
                                    (include_weight_hashes and path.suffix == ".safetensors"))]
    return [{"name": path.name, "bytes": path.stat().st_size,
             "sha256": sha256_file(path)} for path in sorted(paths)]


def git_snapshot(repo, source_files):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "commit": commit,
        "tracked_checkout_dirty": dirty,
        "source_files": [{"path": str(path.relative_to(repo)),
                          "sha256": sha256_file(path)} for path in source_files],
    }


def controller(args):
    script = Path(__file__).resolve()
    repo = script.parents[1]
    model_dir = args.model_dir.resolve()
    corpus = args.corpus.resolve()
    if not model_dir.is_dir():
        raise SystemExit("model directory does not exist: %s" % model_dir)
    if not corpus.is_file():
        raise SystemExit("corpus does not exist: %s" % corpus)
    if args.tokens < max(args.chunk_sizes):
        raise SystemExit("--tokens must be at least the largest chunk size")

    results = []
    for chunk_size in args.chunk_sizes:
        print("benchmarking chunk size %d" % chunk_size, file=sys.stderr, flush=True)
        results.append(_launch_worker(script, model_dir, corpus, chunk_size,
                                      args.tokens, args.warmup_tokens, args.threads,
                                      args.gdn_backend))
    parity_reference = annotate_parity(results, args.parity_atol)
    speed = speed_recommendation(
        results, args.parity_atol, args.eval_tokens_per_runtime,
        args.eval_passes, args.concurrent_runtimes, args.fixed_overhead_minutes,
        args.ram_hourly_usd)
    memory = memory_recommendation(
        results, args.ram_classes, args.concurrent_runtimes,
        args.headroom_factor, args.system_reserve_gib, args.full_run_peak_mib,
        speed.get("recommended_chunk_size"))
    annotate_cost_fit(speed, memory)
    runtime_sources = [
        repo / "holographic" / "io_and_interop" / "holographic_gdnruntime.py",
        repo / "holographic" / "io_and_interop" / "holographic_gdnaccel.py",
        repo / "holographic" / "io_and_interop" / "holographic_ccrun.py",
        repo / "holographic" / "io_and_interop" / "holographic_emit.py",
    ]
    report = {
        "schema": SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "Qwen leCore evaluation runtime; not an acceptance decision",
        "inputs": {
            "model_dir": str(model_dir),
            "weight_hashes_included": not args.skip_weight_hashes,
            "model_files": model_manifest(model_dir, not args.skip_weight_hashes),
            "corpus": {"path": str(corpus), "bytes": corpus.stat().st_size,
                       "sha256": sha256_file(corpus)},
        },
        "settings": {
            "chunk_sizes": args.chunk_sizes,
            "prediction_tokens_per_trial": int(args.tokens),
            "warmup_tokens": int(args.warmup_tokens),
            "threads_per_runtime": int(args.threads),
            "gdn_recurrence_backend_requested": args.gdn_backend,
            "parity_atol": float(args.parity_atol),
            "loss_reference_chunk_size": parity_reference,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": __import__("numpy").__version__,
            "cpu_model": cpu_model(),
            "logical_cpus": os.cpu_count(),
            "total_memory_bytes": total_memory_bytes(),
            "thread_environment_in_workers": {name: str(args.threads)
                                               for name in THREAD_ENV},
        },
        "source": git_snapshot(repo, [script, *runtime_sources]),
        "results": results,
        "memory_recommendation": memory,
        "speed_and_cost_projection": speed,
        "limitations": [
            "Peak RSS is process RSS, conservatively multiplied for concurrent runtimes; shared file-backed pages are not discounted.",
            "Use the larger of this projection and a full acceptance run's observed peak before downsizing.",
            "Duration and cost are extrapolations from a short run; validate the selected configuration with a miniature end-to-end run.",
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, args.output)
    print(rendered, end="")
    return 0 if any(row.get("status") == "ok" for row in results) else 2


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", nargs="?", type=Path)
    parser.add_argument("corpus", nargs="?", type=Path)
    parser.add_argument("--chunk-sizes", type=lambda value: parse_positive_ints(value, "chunk sizes"),
                        default=[64, 128, 256])
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--gdn-backend", choices=("numpy", "c"),
                        default="numpy")
    # Full-vocabulary logits originate in float32.  1e-6 nats is stricter than
    # the acceptance logit tolerance while allowing harmless reduction-order
    # differences across chunk boundaries and the parity-gated C recurrence.
    parser.add_argument("--parity-atol", type=float, default=1e-6)
    parser.add_argument("--ram-classes", type=lambda value: parse_positive_ints(value, "RAM classes"),
                        default=[32, 64, 128])
    parser.add_argument("--concurrent-runtimes", type=int, default=2)
    parser.add_argument("--headroom-factor", type=float, default=1.25)
    parser.add_argument("--system-reserve-gib", type=float, default=2.0)
    parser.add_argument("--full-run-peak-mib", type=float, default=0.0)
    parser.add_argument("--eval-tokens-per-runtime", type=int, default=4096)
    parser.add_argument("--eval-passes", type=int, default=2)
    parser.add_argument("--fixed-overhead-minutes", type=float, default=0.0)
    parser.add_argument("--ram-hourly-usd", type=parse_rates, default={})
    parser.add_argument("--skip-weight-hashes", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--chunk-size", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    numeric_positive = ("tokens", "warmup_tokens", "threads", "concurrent_runtimes",
                        "eval_tokens_per_runtime", "eval_passes")
    if any(getattr(args, name) <= 0 for name in numeric_positive):
        raise SystemExit("token, thread, runtime, and pass counts must be positive")
    finite_nonnegative = (args.system_reserve_gib, args.full_run_peak_mib,
                          args.fixed_overhead_minutes)
    if (not math.isfinite(args.headroom_factor) or args.headroom_factor < 1 or
            any(not math.isfinite(value) or value < 0
                for value in finite_nonnegative)):
        raise SystemExit("headroom must be >= 1 and memory reserves/peaks must be non-negative")
    if not math.isfinite(args.parity_atol) or args.parity_atol < 0:
        raise SystemExit("--parity-atol must be finite and non-negative")
    if args.model_dir is None or args.corpus is None:
        raise SystemExit("model_dir and corpus are required")
    if args.worker:
        if args.chunk_size is None or args.chunk_size <= 0:
            raise SystemExit("worker needs a positive --chunk-size")
        os.environ["LECORE_GDN_BACKEND"] = args.gdn_backend
        print(json.dumps(_worker(args.model_dir.resolve(), args.corpus.resolve(),
                                 args.chunk_size, args.tokens, args.warmup_tokens),
                         sort_keys=True))
        return 0
    return controller(args)


if __name__ == "__main__":
    raise SystemExit(main())

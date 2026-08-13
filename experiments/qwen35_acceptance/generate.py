#!/usr/bin/env python3
"""Generate a complete ilxyr project for the Qwen3.5 acceptance experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from contract import (METRIC_NAMES, METRIC_SPECS,  # noqa: E402
                      OFFICIAL_DEPENDENCY_VERSIONS, RUNNER_POLICY_SCHEMA)


MODEL_REF = "model://openai/codex/gpt-5/2026-08-12/qwen-acceptance-design"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest(model_dir):
    # Bind every top-level file in the materialized snapshot.  The processor,
    # chat template, and index are part of the official vision/text smoke just
    # as surely as config.json and the weights are.  Hidden local cache state
    # is deliberately excluded.
    paths = sorted(path for path in model_dir.iterdir()
                   if path.is_file() and not path.name.startswith("."))
    if not any(path.suffix == ".safetensors" for path in paths):
        raise ValueError("no safetensors checkpoint in %s" % model_dir)
    records = [{"path": path.name, "sha256": sha256_file(path),
                "bytes": path.stat().st_size} for path in paths]
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), records


def actor(role):
    return {"id": "model://codex/lecore-qwen/%s" % role,
            "kind": "model", "model_ref": MODEL_REF + "/" + role}


def contribution(identifier, stage, role, title, body, inputs, claims, confidence):
    return {
        "schema": "ilxyr.contribution.v1",
        "id": identifier,
        "stage": stage,
        "actor": actor(role),
        "title": title,
        "body": body,
        "input_refs": inputs,
        "claims": claims,
        "confidence": confidence,
    }


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def frozen_sequential_looks(min_tokens):
    """Four deterministic looks; interim boundaries may reject, never accept."""
    total = int(min_tokens)
    return sorted({look for look in
                   (int(round(total * fraction / 4.0))
                    for fraction in (1, 2, 3, 4))
                   if look >= 1000 or look == total} | {total})


def canonical_policy_digest(policy):
    """Content identity for the complete v4 runner/treatment policy."""
    if policy.get("schema") != RUNNER_POLICY_SCHEMA:
        raise ValueError("runner policy must use %s" % RUNNER_POLICY_SCHEMA)
    canonical = json.dumps(
        policy, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("installation_corpus", type=Path,
                    help="text used only to ground the installation")
    ap.add_argument("evaluation_corpus", type=Path,
                    help="separate held-out text used only for evaluation")
    ap.add_argument("out_dir", type=Path,
                    help="directory that will receive the ilxyr project JSON")
    ap.add_argument("--installed-dir", type=Path,
                    help="empty destination for the experiment output checkpoint")
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--min-tokens", type=int, default=4096)
    ap.add_argument("--timeout-seconds", type=int, default=21600)
    ap.add_argument("--compute-credits", type=int, default=100)
    ap.add_argument("--experiment-version", type=int, default=4,
                    help="monotonic formal-attempt identity (minimum/default: 4)")
    ap.add_argument("--initial-chunk-size", type=int, default=128)
    ap.add_argument("--max-chunk-size", type=int,
                    help="requires --benchmark-report when above initial chunk")
    ap.add_argument("--benchmark-report", type=Path,
                    help="checksummed output from tools/benchmark_qwen_runtime.py")
    ap.add_argument("--memory-budget-fraction", type=float, default=0.20)
    ap.add_argument("--evaluation-mode", choices=("auto", "serial", "parallel"),
                    default="auto")
    ap.add_argument("--gdn-backend", choices=("numpy", "c"), default="c")
    ap.add_argument("--progress-upload-uri",
                    help="optional s3:// URI refreshed after every evaluation chunk")
    ap.add_argument("--ilxyr-cli", type=Path,
                    default=Path.home() / "develop" / "ilxyr" / "target" / "debug" / "ilxyr")
    args = ap.parse_args(argv)

    model_dir = args.model_dir.expanduser().resolve()
    installation_corpus = args.installation_corpus.expanduser().resolve()
    evaluation_corpus = args.evaluation_corpus.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    installed_dir = ((args.installed_dir.expanduser().resolve())
                     if args.installed_dir else out_dir / "installed-checkpoint")
    # Preserve the selected executable path instead of dereferencing its final
    # symlink.  A venv's ``python`` normally points at the system interpreter;
    # resolving it records /usr/bin/python and silently drops the venv's site
    # packages when ilxyr executes the absolute program.
    python = args.python.expanduser().absolute()
    if (not model_dir.is_dir() or not installation_corpus.is_file()
            or not evaluation_corpus.is_file() or not python.is_file()):
        ap.error("model_dir, both corpora, and --python must exist")
    if int(args.min_tokens) < 1000:
        ap.error("--min-tokens must be at least 1000")
    if int(args.experiment_version) < 4:
        ap.error("this repaired policy is v4; --experiment-version must be at least 4")
    if args.initial_chunk_size < 1:
        ap.error("--initial-chunk-size must be positive")
    if not 0 < args.memory_budget_fraction <= 0.5:
        ap.error("--memory-budget-fraction must be in (0, 0.5]")
    if args.progress_upload_uri and not args.progress_upload_uri.startswith("s3://"):
        ap.error("--progress-upload-uri must be an s3:// URI")
    sequential_looks = frozen_sequential_looks(args.min_tokens)

    model_digest, model_files = model_manifest(model_dir)
    installation_digest = sha256_file(installation_corpus)
    evaluation_digest = sha256_file(evaluation_corpus)
    if installation_digest == evaluation_digest:
        ap.error("installation and evaluation corpora must have distinct contents")
    benchmark = None
    benchmark_digest = None
    benchmark_recommended_chunk = None
    if args.benchmark_report:
        benchmark_path = args.benchmark_report.expanduser().resolve()
        if not benchmark_path.is_file():
            ap.error("--benchmark-report does not exist")
        try:
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            ap.error("invalid --benchmark-report: %s" % exc)
        if benchmark.get("schema") != "lecore.qwen_runtime_benchmark.v1":
            ap.error("--benchmark-report has the wrong schema")
        inputs = benchmark.get("inputs") or {}
        if (inputs.get("corpus") or {}).get("sha256") != evaluation_digest:
            ap.error("--benchmark-report does not bind the evaluation corpus")
        if not inputs.get("weight_hashes_included"):
            ap.error("--benchmark-report must include model weight hashes")
        current_hashes = {row["path"]: row["sha256"] for row in model_files}
        for row in inputs.get("model_files") or []:
            if current_hashes.get(row.get("name")) != row.get("sha256"):
                ap.error("--benchmark-report model hash mismatch: %s" %
                         row.get("name"))
        benchmark_recommended_chunk = int(
            (benchmark.get("speed_and_cost_projection") or {}).get(
                "recommended_chunk_size", 0))
        if benchmark_recommended_chunk < args.initial_chunk_size:
            ap.error("--benchmark-report has no usable recommended chunk")
        benchmark_digest = sha256_file(benchmark_path)
    if args.max_chunk_size is None:
        args.max_chunk_size = (benchmark_recommended_chunk or
                               args.initial_chunk_size)
    if args.max_chunk_size < args.initial_chunk_size:
        ap.error("--max-chunk-size must cover the initial chunk")
    if (args.max_chunk_size > args.initial_chunk_size and
            benchmark_recommended_chunk is None):
        ap.error("increasing chunk size requires --benchmark-report")
    if (benchmark_recommended_chunk is not None and
            args.max_chunk_size > benchmark_recommended_chunk):
        ap.error("--max-chunk-size exceeds the benchmark recommendation")
    requirements_cpu = HERE / "requirements-cpu.txt"
    if not requirements_cpu.is_file():
        ap.error("missing frozen official dependency lock: %s" % requirements_cpu)
    runner_policy = {
        "schema": RUNNER_POLICY_SCHEMA,
        # Stable summary keys retained for simple report consumers.  They are
        # inside the canonical object (and therefore cannot drift from the
        # detailed sections without changing identity).
        "accepted_requires_full_tokens": int(args.min_tokens),
        "initial_chunk_size": int(args.initial_chunk_size),
        "max_chunk_size": int(args.max_chunk_size),
        "gdn_backend": args.gdn_backend,
        "sequential_looks": sequential_looks,
        "early_acceptance_allowed": False,
        "early_rejection_allowed": True,
        "benchmark_report_sha256": benchmark_digest,
        "evaluation": {
            "accepted_requires_full_tokens": int(args.min_tokens),
            "initial_chunk_size": int(args.initial_chunk_size),
            "max_chunk_size": int(args.max_chunk_size),
            "memory_budget_fraction": float(args.memory_budget_fraction),
            "mode": args.evaluation_mode,
            "worker_isolation": "one_process_per_checkpoint",
            "paired_position_order": "frozen_corpus_prefix",
            "common_chunk_schedule_required": True,
            "backend": args.gdn_backend,
            "reference_backend": "numpy",
            "benchmark_report_sha256": benchmark_digest,
        },
        "statistical_decision": {
            "maximum_perplexity_regression_ratio": 0.01,
            "final_interval": "paired_moving_block_bootstrap_two_sided_95pct",
            "final_resamples": 1200,
            "final_seed": 0,
            "summary_resamples_per_model": 800,
            "sequential_looks": sequential_looks,
            "sequential_family_alpha": 0.05,
            "sequential_resamples": 10000,
            "sequential_boundary": "one_sided_bonferroni_lower_bound",
            "minimum_interim_paired_tokens": 1000,
            "minimum_dependence_blocks_per_look": 2,
            "early_acceptance_allowed": False,
            "early_rejection_allowed": True,
        },
        "reference_parity": {
            "backend": "numpy",
            "maximum_relative_logit_error": 0.001,
            "tokenizer_ids_must_match": True,
            "must_pass_before_installation": True,
        },
        "treatment": {
            "entrypoint": "assimilation/install.py",
            "experimental_acknowledgement": True,
            "device": "cpu",
            "gdn_backend": "numpy",
            "spectral_filtering_enabled": False,
            "prepend_layers": "installer_default_proportional_8pct",
            "registers": "installer_default_width_eighth",
            "passages": "installer_default_hidden_width_1mb_sidecar_budget",
            "weight_resident_metadata": False,
        },
        "sidecars": {
            "metadata_location": "lecore.json",
            "weight_resident_metadata_forbidden": True,
            "index_policy": "always_sidecar_in_default_treatment",
            "sidecar_index_in_paired_model_evaluation": False,
            "sidecars_in_evidence_bundle": True,
        },
        "official_compatibility": {
            "dependency_versions": dict(OFFICIAL_DEPENDENCY_VERSIONS),
            "environment_lock": "experiments/qwen35_acceptance/requirements-cpu.txt",
            "environment_lock_sha256": sha256_file(requirements_cpu),
            "dependency_mismatch_outcome": "execution_failure_before_model_work",
            "reload_requires_no_state_dict_incompatibilities": True,
            "text_generation_required": True,
            "vision_input_generation_required": True,
        },
        "telemetry": {
            "progress_upload_uri": args.progress_upload_uri,
            "local_progress_fsync_per_record": True,
            "upload_failure_changes_scientific_outcome": False,
        },
        "execution_limits": {
            "timeout_seconds": int(args.timeout_seconds),
            "maximum_compute_credits": int(args.compute_credits),
        },
    }
    policy_digest = canonical_policy_digest(runner_policy)
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    stem = "lecore.qwen35.install.%s.%s.%s.%s.%s.v%d" % (
        model_digest[:12], installation_digest[:12], evaluation_digest[:12],
        source_commit[:12], policy_digest[:12], int(args.experiment_version))
    ids = {
        "hypothesis": stem + ".hypothesis",
        "foundation": stem + ".foundation",
        "engineering": stem + ".engineering-review",
        "design": stem + ".experiment-design",
        "experiment": stem + ".acceptance",
    }

    hypothesis = contribution(
        ids["hypothesis"], "hypothesis", "research-director",
        "Qwen3.5 can carry a reloadable leCore layer-prepending installation",
        "The experimental layer-prepending installer can emit an ordinary Qwen3.5 checkpoint while preserving reference behavior within a preregistered paired confidence bound and retaining official text and vision-language execution.",
        [],
        ["Pre-install leCore logits match the official reference implementation.",
         "The installed checkpoint reloads and remains within the paired regression budget.",
         "Official Transformers text generation and vision input both remain operational."],
        0.35)
    foundation = contribution(
        ids["foundation"], "mathematical_foundation", "statistical-reviewer",
        "Paired moving-block inference is required for token-loss comparisons",
        "Per-token losses are serially correlated. The acceptance decision therefore uses paired installed-minus-original token NLLs, estimates an autocorrelation-aware moving-block length, and requires the upper 95 percent confidence bound to stay within a one-percent perplexity regression budget. Up to four frozen looks, each after at least one thousand paired positions, divide a family-wise five-percent error budget by Bonferroni correction and may only stop for clear rejection; every accepted result still measures the complete preregistered token count. Point estimates and independent-token bootstrap intervals are not acceptance evidence.",
        [ids["hypothesis"]],
        ["Every accepted outcome measures the full preregistered paired token count; an early rejected outcome contains at least one thousand paired positions and its multiplicity-corrected boundary.",
         "The statistical gate is decided by the paired block interval rather than the point perplexity delta."],
        0.95)
    engineering = contribution(
        ids["engineering"], "engineering_review", "engineering-reviewer",
        "Qwen acceptance runner and provenance boundary",
        "The shell-free runner uses absolute paths, records the exact leCore commit and checker hashes, keeps spectral filtering disabled, invokes the layer-prepending path only with its experimental acknowledgement, records peak memory, reloads the emitted artifact through both leCore and official Transformers, and exercises the official text and image-text interfaces. Future runs request the narrow native Gated-DeltaNet recurrence, whose first real fresh and resumed states are parity-gated against NumPy and whose refusal falls back safely with diagnostics.",
        [ids["hypothesis"], ids["foundation"]],
        ["The emitted stdout is exactly the ilxyr metrics/source envelope.",
         "Installer logs, durable per-chunk progress, monotonic stage timings, memory-selected chunk sizes, and a human-readable metrics artifact are retained beside the output checkpoint."],
        0.85)
    design = contribution(
        ids["design"], "experiment_design", "experiment-designer",
        "One-shot Qwen3.5 installation acceptance run",
        "Execute once against the content-bound public checkpoint, installation corpus, and separate held-out evaluation corpus. Do not tune thresholds, sequential looks, chunk limits, or replace either corpus after admission. Memory facts select a chunk no larger than %d and two isolated workers evaluate the same frozen token positions concurrently when a conservative host-memory admission passes. Sequential looks at %s paired tokens divide alpha equally across the frozen looks and may only stop for NO-GO; GO always requires all %d positions and the unchanged final 95 percent paired block interval. Resolve accepted only when source cleanliness, tokenizer parity, reference-logit parity, the paired statistical gate, leCore disk reload, official Transformers reload, official text generation, and official vision smoke all pass. An official model capability failure emits zero-valued gates and is preserved as rejected evidence; dependency or runner failure remains execution_failure." % (int(args.max_chunk_size), ", ".join(map(str, sequential_looks)), int(args.min_tokens)),
        [ids["hypothesis"], ids["foundation"], ids["engineering"]],
        ["Accepted and rejected are exhaustive for a valid metrics envelope.",
         "Runtime or dependency failure resolves separately as execution_failure."],
        0.95)

    experiment = {
        "schema": "ilxyr.experiment.v1",
        "id": ids["experiment"],
        "title": "Qwen3.5 leCore layer-prepending acceptance",
        "hypothesis": hypothesis["body"],
        "rationale": "This converts the unresolved Qwen integration claims into one frozen, replayable, statistically gated run. Spectral filtering is a separate research control and is forbidden in this contract.",
        "proposer": actor("research-director"),
        "lineage": {
            "hypothesis": ids["hypothesis"],
            "mathematical_foundation": ids["foundation"],
            "engineering_review": ids["engineering"],
            "experiment_design": ids["design"],
        },
        "baseline": "baseline://Qwen/Qwen3.5-0.8B/%s" % model_digest,
        "datasets": [
            "dataset://lecore/qwen-installation/%s" % installation_digest,
            "dataset://lecore/qwen-evaluation/%s" % evaluation_digest,
        ],
        "models": ["weight://Qwen/Qwen3.5-0.8B/%s" % model_digest],
        "metrics": METRIC_SPECS,
        "seeds": [0],
        "runner_policy": runner_policy,
        "runner_policy_digest": policy_digest,
        "outcome_contract": {
            "primary_metric": "acceptance_pass",
            "success_outcome": "accepted",
            "outcomes": [
                {"id": "accepted", "description": "Every frozen Qwen acceptance gate passed.",
                 "predicate": {"kind": "metric", "metric": "acceptance_pass", "operator": "gte", "threshold": 1}},
                {"id": "rejected", "description": "The run completed but at least one acceptance gate failed.",
                 "predicate": {"kind": "metric", "metric": "acceptance_pass", "operator": "lt", "threshold": 1}},
                {"id": "execution_failure", "description": "The runner failed or did not emit its exact metric contract.",
                 "predicate": {"kind": "execution_failure"}},
            ],
        },
        "execution": {
            "executor": "local-command",
            "program": str(python),
            "args": [str(HERE / "run.py"), str(model_dir), str(installed_dir),
                     str(installation_corpus), str(evaluation_corpus),
                     "--min-tokens", str(int(args.min_tokens)),
                     "--chunk-size", str(int(args.initial_chunk_size)),
                     "--max-chunk-size", str(int(args.max_chunk_size)),
                     "--memory-budget-fraction", str(float(args.memory_budget_fraction)),
                     "--evaluation-mode", args.evaluation_mode,
                     "--gdn-backend", args.gdn_backend,
                     "--sequential-looks", ",".join(map(str, sequential_looks)),
                     "--allow-early-rejection"] +
                    (["--progress-upload-uri", args.progress_upload_uri]
                     if args.progress_upload_uri else []),
            "timeout_seconds": int(args.timeout_seconds),
            "max_cost_credits": int(args.compute_credits),
            "network": "open",
        },
        "funding": {"required_compute_credits": int(args.compute_credits),
                    "minimum_forecasters": 2, "minimum_total_stake": 10},
        "security": {"weight_class": "public", "code_policy": "arbitrary",
                     "export_policy": "artifacts"},
        "evidence_authority": {
            "level": "corpus_proxy",
            "scope": {"seeds": [0],
                      "eval_set": "dataset://lecore/qwen-evaluation/%s" % evaluation_digest,
                      "coverage": 1.0},
            # ilxyr artifact refs name objects already present in its local
            # content-addressed store; raw file SHA-256 values are not object
            # refs. Start empty on a fresh workspace. compile adds the four
            # contribution artifacts, while the model/corpus digests stay
            # frozen in their handles and the runner attests checker files.
            "provenance": {"artifact_hashes": [],
                           "model_lineage": "model://Qwen/Qwen3.5-0.8B/%s" % model_digest,
                           "checker": "checker://lecore/qwen35-acceptance/%s/v%d"
                           % (policy_digest, int(args.experiment_version))},
        },
        "expected_outputs": (["metrics.%s" % name for name in METRIC_NAMES]
                             + ["resolved_outcome", "forecast_settlements"]),
    }

    forecasts = [
        {"schema": "ilxyr.forecast.v1", "id": stem + ".forecast.empirical",
         "experiment_id": ids["experiment"], "forecaster": actor("forecaster-empirical"),
         "probabilities": {"accepted": 0.20, "rejected": 0.65, "execution_failure": 0.15},
         "stake": 5,
         "rationale": "Structural rehearsal passes, but no complete post-memory-fix real-Qwen install and powered acceptance run has succeeded yet."},
        {"schema": "ilxyr.forecast.v1", "id": stem + ".forecast-mechanistic",
         "experiment_id": ids["experiment"], "forecaster": actor("forecaster-mechanistic"),
         "probabilities": {"accepted": 0.15, "rejected": 0.55, "execution_failure": 0.30},
         "stake": 5,
         "rationale": "The blank prepend has a strong identity construction, while full-checkpoint memory pressure and official multimodal reload remain material execution risks."},
    ]
    funding = {
        "schema": "ilxyr.funding.v1", "id": stem + ".funding",
        "experiment_id": ids["experiment"],
        "funder": {"id": "service://lecore/qwen-experiment-generator", "kind": "service"},
        "compute_credits": int(args.compute_credits),
        "rationale": "Reserve one complete public-weight Qwen acceptance run under the frozen contract.",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "hypothesis.json": hypothesis,
        "foundation.json": foundation,
        "engineering-review.json": engineering,
        "experiment-design.json": design,
        "experiment.json": experiment,
        "forecast-empirical.json": forecasts[0],
        "forecast-mechanistic.json": forecasts[1],
        "funding.json": funding,
    }
    for name, payload in files.items():
        write_json(out_dir / name, payload)

    cli = str(args.ilxyr_cli.expanduser().resolve())
    order = ["hypothesis.json", "foundation.json", "engineering-review.json",
             "experiment-design.json"]
    commands = ([[cli, "contribute", "<workspace>", str(out_dir / name)] for name in order]
                + [[cli, "compile", "<workspace>", str(out_dir / "experiment.json")],
                   [cli, "forecast", "<workspace>", str(out_dir / "forecast-empirical.json")],
                   [cli, "forecast", "<workspace>", str(out_dir / "forecast-mechanistic.json")],
                   [cli, "fund", "<workspace>", str(out_dir / "funding.json")],
                   [cli, "admit", "<workspace>", ids["experiment"]],
                   [cli, "run", "<workspace>", ids["experiment"], "--execute"],
                   [cli, "status", "<workspace>", ids["experiment"]],
                   [cli, "verify", "<workspace>"]])
    write_json(out_dir / "project.json", {
        "schema": "lecore.ilxyr-project.v1", "experiment_id": ids["experiment"],
        "experiment_version": int(args.experiment_version),
        "source_commit": source_commit, "model_digest": model_digest,
        "runner_policy_digest": policy_digest,
        "model_files": model_files,
        "corpora": {
            "installation": {"path": str(installation_corpus),
                             "sha256": installation_digest,
                             "bytes": installation_corpus.stat().st_size},
            "evaluation": {"path": str(evaluation_corpus),
                           "sha256": evaluation_digest,
                           "bytes": evaluation_corpus.stat().st_size},
        },
        "installed_dir": str(installed_dir), "commands": commands,
        "runner_policy": runner_policy,
    })
    print(json.dumps({"project_dir": str(out_dir),
                      "experiment_id": ids["experiment"],
                      "project": str(out_dir / "project.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

from contract import METRIC_NAMES, METRIC_SPECS  # noqa: E402


MODEL_REF = "model://openai/codex/gpt-5/2026-08-12/qwen-acceptance-design"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest(model_dir):
    names = ["config.json", "tokenizer.json", "vocab.json", "merges.txt"]
    paths = [model_dir / name for name in names if (model_dir / name).is_file()]
    paths += sorted(model_dir.glob("*.safetensors"))
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
    ap.add_argument("--ilxyr-cli", type=Path,
                    default=Path.home() / "develop" / "ilxyr" / "target" / "debug" / "ilxyr")
    args = ap.parse_args(argv)

    model_dir = args.model_dir.expanduser().resolve()
    installation_corpus = args.installation_corpus.expanduser().resolve()
    evaluation_corpus = args.evaluation_corpus.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    installed_dir = ((args.installed_dir.expanduser().resolve())
                     if args.installed_dir else out_dir / "installed-checkpoint")
    python = args.python.expanduser().resolve()
    if (not model_dir.is_dir() or not installation_corpus.is_file()
            or not evaluation_corpus.is_file() or not python.is_file()):
        ap.error("model_dir, both corpora, and --python must exist")
    if int(args.min_tokens) < 1000:
        ap.error("--min-tokens must be at least 1000")

    model_digest, model_files = model_manifest(model_dir)
    installation_digest = sha256_file(installation_corpus)
    evaluation_digest = sha256_file(evaluation_corpus)
    if installation_digest == evaluation_digest:
        ap.error("installation and evaluation corpora must have distinct contents")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    stem = "lecore.qwen35.install.%s.%s.%s.%s.v1" % (
        model_digest[:12], installation_digest[:12], evaluation_digest[:12],
        source_commit[:12])
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
        "Per-token losses are serially correlated. The acceptance decision therefore uses paired installed-minus-original token NLLs, estimates an autocorrelation-aware moving-block length, and requires the upper 95 percent confidence bound to stay within a one-percent perplexity regression budget. Point estimates and independent-token bootstrap intervals are not acceptance evidence.",
        [ids["hypothesis"]],
        ["At least the preregistered minimum number of paired token positions is measured.",
         "The statistical gate is decided by the paired block interval rather than the point perplexity delta."],
        0.95)
    engineering = contribution(
        ids["engineering"], "engineering_review", "engineering-reviewer",
        "Qwen acceptance runner and provenance boundary",
        "The shell-free runner uses absolute paths, records the exact leCore commit and checker hashes, keeps spectral filtering disabled, invokes the layer-prepending path only with its experimental acknowledgement, records peak memory, reloads the emitted artifact, and exercises the official Transformers text and image-text interfaces.",
        [ids["hypothesis"], ids["foundation"]],
        ["The emitted stdout is exactly the ilxyr metrics/source envelope.",
         "Installer logs and a human-readable metrics artifact are retained beside the output checkpoint."],
        0.85)
    design = contribution(
        ids["design"], "experiment_design", "experiment-designer",
        "One-shot Qwen3.5 installation acceptance run",
        "Execute once against the content-bound public checkpoint, installation corpus, and separate held-out evaluation corpus. Do not tune thresholds or replace either corpus after admission. Resolve accepted only when source cleanliness, tokenizer parity, reference-logit parity, the paired statistical gate, disk reload, official text generation, and official vision smoke all pass. A cleanly executed no-go is preserved as rejected evidence.",
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
                     "--min-tokens", str(int(args.min_tokens))],
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
                           "checker": "checker://lecore/qwen35-acceptance/v1"},
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
        "source_commit": source_commit, "model_digest": model_digest,
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
    })
    print(json.dumps({"project_dir": str(out_dir),
                      "experiment_id": ids["experiment"],
                      "project": str(out_dir / "project.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

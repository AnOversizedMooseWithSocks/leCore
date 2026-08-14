#!/usr/bin/env python3
"""Exercise the generated Qwen project through ilxyr admission, without running it."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EXPERIMENT = REPO / "experiments" / "qwen35_acceptance"
EXPECTED_REVISION = (EXPERIMENT / "ilxyr-revision.txt").read_text(
    encoding="utf-8").strip()


def run(command, *, cwd=REPO):
    completed = subprocess.run(
        [str(part) for part in command], cwd=cwd, capture_output=True,
        text=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            "command failed (%d): %s\nstdout:\n%s\nstderr:\n%s" % (
                completed.returncode, " ".join(map(str, command)),
                completed.stdout, completed.stderr))
    return completed.stdout


def parse_json(output, label):
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError("%s did not emit JSON: %s" % (label, output)) from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ilxyr-cli", type=Path, required=True)
    parser.add_argument("--ilxyr-source", type=Path, required=True)
    args = parser.parse_args(argv)
    cli = args.ilxyr_cli.expanduser().resolve()
    ilxyr_source = args.ilxyr_source.expanduser().resolve()
    if not cli.is_file():
        parser.error("--ilxyr-cli does not exist: %s" % cli)
    revision = run(
        ["git", "rev-parse", "HEAD"], cwd=ilxyr_source).strip()
    if revision != EXPECTED_REVISION:
        parser.error("ilxyr source revision drifted: expected %s, got %s" % (
            EXPECTED_REVISION, revision))

    with tempfile.TemporaryDirectory(prefix="lecore-qwen-ilxyr-ci-") as raw:
        root = Path(raw)
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}\n", encoding="utf-8")
        (model / "model.safetensors").write_bytes(
            b"ci-only content-bound placeholder")
        installation = root / "installation.txt"
        evaluation = root / "evaluation.txt"
        installation.write_text("installation-only corpus\n", encoding="utf-8")
        evaluation.write_text("held-out evaluation-only corpus\n", encoding="utf-8")
        project_dir = root / "project"
        workspace = root / "workspace"

        generated = parse_json(run([
            sys.executable, EXPERIMENT / "generate.py", model, installation,
            evaluation, project_dir, "--experiment-version", "5",
            "--min-tokens", "1000", "--ilxyr-cli", cli,
        ]), "generator")
        project = json.loads((project_dir / "project.json").read_text(
            encoding="utf-8"))
        experiment = json.loads((project_dir / "experiment.json").read_text(
            encoding="utf-8"))
        if generated["experiment_id"] != project["experiment_id"]:
            raise RuntimeError("generator and project experiment IDs disagree")
        if "runner_policy" in experiment or "runner_policy_digest" in experiment:
            raise RuntimeError("ilxyr experiment contains non-schema policy fields")

        run([cli, "init", workspace])
        for name in ("hypothesis.json", "foundation.json",
                     "engineering-review.json", "experiment-design.json"):
            run([cli, "contribute", workspace, project_dir / name])
        run([cli, "compile", workspace, project_dir / "experiment.json"])
        run([cli, "forecast", workspace,
             project_dir / "forecast-empirical.json"])
        run([cli, "forecast", workspace,
             project_dir / "forecast-mechanistic.json"])
        run([cli, "fund", workspace, project_dir / "funding.json"])
        admission = parse_json(
            run([cli, "admit", workspace, project["experiment_id"]]),
            "admission")
        status = parse_json(
            run([cli, "status", workspace, project["experiment_id"]]),
            "status")
        verification = parse_json(run([cli, "verify", workspace]), "verify")

        checks = admission.get("checks") or []
        if not admission.get("accepted") or not checks or not all(
                check.get("passed") for check in checks):
            raise RuntimeError("miniature project did not pass admission: %r" % admission)
        if not status.get("compiled_ref"):
            raise RuntimeError("miniature project has no compiled artifact")
        if status.get("execution_started"):
            raise RuntimeError("CI preflight unexpectedly executed the experiment")
        if not verification.get("valid"):
            raise RuntimeError("ilxyr workspace verification failed: %r" % verification)
        print(json.dumps({
            "admission_checks": len(checks),
            "compiled_ref": status["compiled_ref"],
            "experiment_id": project["experiment_id"],
            "execution_started": False,
            "ilxyr_revision": EXPECTED_REVISION,
            "valid": True,
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

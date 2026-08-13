#!/usr/bin/env python3
"""Build a deterministic publication manifest for one Qwen result directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any


EXCLUDED_NAMES = {"publication-manifest.json", "publication-receipt.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evidence_ref(result_dir: Path) -> str:
    events_path = result_dir / "ledger" / "events.jsonl"
    with events_path.open("r", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    recorded = [
        event["artifact_ref"]
        for event in events
        if event.get("event_type") == "EvidenceRecorded"
    ]
    if len(recorded) != 1:
        raise ValueError(f"expected exactly one EvidenceRecorded event, got {len(recorded)}")
    return recorded[0]


def build_manifest(result_dir: Path) -> dict[str, Any]:
    project = read_json(result_dir / "project" / "project.json")
    status = read_json(result_dir / "result" / "ilxyr-status.json")
    verification = read_json(result_dir / "result" / "ilxyr-verify.json")
    latest_evidence = status["latest_evidence"]

    files = []
    for path in sorted(item for item in result_dir.rglob("*") if item.is_file()):
        if path.name in EXCLUDED_NAMES:
            continue
        relative = path.relative_to(result_dir).as_posix()
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "media_type": media_type,
            }
        )

    return {
        "schema": "lecore.qwen35-publication-manifest.v1",
        "experiment_id": project["experiment_id"],
        "resolved_outcome": latest_evidence["resolved_outcome"],
        "run_ref": latest_evidence["run_ref"],
        "evidence_ref": evidence_ref(result_dir),
        "ledger_verification": verification,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.expanduser().resolve()
    manifest = build_manifest(result_dir)
    output = result_dir / "publication-manifest.json"
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{sha256(output)}  {output}")


if __name__ == "__main__":
    main()

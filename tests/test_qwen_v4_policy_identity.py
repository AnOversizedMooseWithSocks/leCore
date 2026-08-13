"""Focused identity gates for the content-bound Qwen v4 runner policy."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_generator():
    path = ROOT / "experiments" / "qwen35_acceptance" / "generate.py"
    spec = importlib.util.spec_from_file_location("qwen_v4_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def complete_policy(generator):
    return {
        "schema": generator.RUNNER_POLICY_SCHEMA,
        "evaluation": {
            "accepted_requires_full_tokens": 4096,
            "initial_chunk_size": 128,
            "max_chunk_size": 256,
            "memory_budget_fraction": 0.2,
            "mode": "parallel",
            "backend": "c",
        },
        "statistical_decision": {
            "sequential_looks": [1024, 2048, 3072, 4096],
            "sequential_resamples": 10000,
        },
        "reference_parity": {"backend": "numpy"},
        "treatment": {
            "gdn_backend": "numpy",
            "spectral_filtering_enabled": False,
        },
        "sidecars": {
            "weight_resident_metadata_forbidden": True,
            "index_policy": "always_sidecar_in_default_treatment",
        },
        "official_compatibility": {
            "dependency_versions": dict(generator.OFFICIAL_DEPENDENCY_VERSIONS),
        },
        "telemetry": {"progress_upload_uri": None},
        "execution_limits": {"timeout_seconds": 21600,
                             "maximum_compute_credits": 100},
    }


def test_policy_digest_is_canonical_and_every_policy_section_is_identity_bound():
    generator = load_generator()
    policy = complete_policy(generator)
    digest = generator.canonical_policy_digest(policy)
    reordered = json.loads(json.dumps(policy, sort_keys=True))

    assert generator.canonical_policy_digest(reordered) == digest
    for section in ("evaluation", "statistical_decision", "reference_parity",
                    "treatment", "sidecars", "official_compatibility",
                    "telemetry", "execution_limits"):
        changed = copy.deepcopy(policy)
        changed[section]["identity_probe"] = section
        assert generator.canonical_policy_digest(changed) != digest


def test_policy_digest_rejects_an_unversioned_policy():
    generator = load_generator()
    policy = complete_policy(generator)
    policy.pop("schema")

    try:
        generator.canonical_policy_digest(policy)
    except ValueError as exc:
        assert generator.RUNNER_POLICY_SCHEMA in str(exc)
    else:
        raise AssertionError("unversioned runner policy was accepted")


def generate_v4(tmp_path, name, *extra):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"content-bound-test-weights")
    installation = tmp_path / "installation.txt"
    evaluation = tmp_path / "evaluation.txt"
    installation.write_text("installation corpus\n", encoding="utf-8")
    evaluation.write_text("held out evaluation corpus\n", encoding="utf-8")
    project = tmp_path / name
    command = [
        sys.executable,
        str(ROOT / "experiments" / "qwen35_acceptance" / "generate.py"),
        str(model), str(installation), str(evaluation), str(project),
        "--min-tokens", "1000", "--experiment-version", "4", *extra,
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True,
                               text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    return (json.loads((project / "project.json").read_text()),
            json.loads((project / "experiment.json").read_text()))


def test_v4_experiment_id_and_checker_bind_complete_policy(tmp_path):
    generator = load_generator()
    project, experiment = generate_v4(tmp_path, "default")
    changed, _ = generate_v4(
        tmp_path, "serial", "--evaluation-mode", "serial",
        "--timeout-seconds", "18000")
    policy = project["runner_policy"]
    digest = generator.canonical_policy_digest(policy)

    assert project["runner_policy_digest"] == digest
    assert experiment["runner_policy_digest"] == digest
    assert experiment["runner_policy"] == policy
    assert digest[:12] in experiment["id"]
    assert experiment["id"].endswith(".v4.acceptance")
    assert experiment["evidence_authority"]["provenance"]["checker"] == \
        "checker://lecore/qwen35-acceptance/%s/v4" % digest
    assert changed["experiment_id"] != project["experiment_id"]
    assert policy["official_compatibility"]["dependency_versions"] == \
        generator.OFFICIAL_DEPENDENCY_VERSIONS
    lock = ROOT / policy["official_compatibility"]["environment_lock"]
    assert policy["official_compatibility"]["environment_lock_sha256"] == \
        generator.sha256_file(lock)
    assert policy["treatment"]["gdn_backend"] == "numpy"
    assert policy["treatment"]["spectral_filtering_enabled"] is False
    assert policy["sidecars"]["weight_resident_metadata_forbidden"] is True

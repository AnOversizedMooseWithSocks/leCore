"""Portable gates for the Qwen3.5/ilxyr experiment contract."""

import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_gate_uses_blocks_for_correlated_losses():
    from holographic.io_and_interop.holographic_measure import better_than

    # Ten long correlated regions look like 500 independent observations to an
    # IID bootstrap. They are only about ten independent pieces of evidence.
    delta = np.repeat([-0.12] * 4 + [0.06] * 6, 50)
    baseline = np.full(len(delta), 3.0)
    a = {"nll": baseline + delta,
         "perplexity": float(np.exp(np.mean(baseline + delta)))}
    b = {"nll": baseline, "perplexity": float(np.exp(np.mean(baseline)))}
    verdict = better_than(a, b, resamples=2000, seed=7)

    assert verdict["block"] > 1
    assert verdict["effective_n"] < len(delta) // 4
    assert verdict["ci_lo_nats"] <= 0 <= verdict["ci_hi_nats"]
    assert verdict["verdict"] == "INDISTINGUISHABLE"


def test_long_block_bootstrap_never_falls_back_to_iid_positions():
    from holographic.io_and_interop.holographic_measure import _bootstrap_means

    values = np.arange(12, dtype=np.float64)
    # One full-length moving block has one legal start, so every draw must keep
    # the serial sequence's mean.  The old IID fallback varied these means and
    # manufactured precision precisely when correlation was strongest.
    means = _bootstrap_means(values, np.random.default_rng(0), 200, len(values))
    assert np.array_equal(means, np.full(200, values.mean()))


def test_randomized_svd_preserves_float32():
    from holographic.io_and_interop.holographic_unicron import rsvd

    matrix = np.random.default_rng(0).standard_normal((64, 48)).astype(np.float32)
    u, singular, vt = rsvd(matrix, 8, seed=0, power=1)
    assert u.dtype == singular.dtype == vt.dtype == np.float32
    assert np.isfinite(u).all() and np.isfinite(singular).all() and np.isfinite(vt).all()


def test_file_backed_safetensors_decodes_only_on_access(tmp_path):
    from holographic.io_and_interop.holographic_unicron import (
        SafetensorWeights, save_safetensors)

    path = tmp_path / "weights.safetensors"
    tensors = {
        "f32": np.arange(24, dtype=np.float32).reshape(6, 4),
        "bf16": np.linspace(-1, 1, 20, dtype=np.float32).reshape(5, 4),
    }
    save_safetensors(path, tensors, dtypes={"f32": "F32", "bf16": "BF16"})
    store = SafetensorWeights(path, max_cached=1)

    assert store.stats == {"hits": 0, "misses": 0, "decoded_bytes": 0}
    assert isinstance(store["f32"], np.memmap)
    assert store.stats["decoded_bytes"] == 0
    decoded = store["bf16"]
    assert decoded.dtype == np.float32
    assert store.stats["decoded_bytes"] == decoded.nbytes


def build_fixture(tmp_path):
    output = tmp_path / "mini-qwen"
    command = [sys.executable, str(ROOT / "tools" / "build_mini_qwen.py"),
               str(output), "--layers", "4", "--vocab", "512"]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True,
                               text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    return output, json.loads(completed.stdout)


def test_portable_qwen_fixture_loads_from_arbitrary_cwd(tmp_path):
    from holographic.io_and_interop.holographic_bpe import BPE
    from holographic.io_and_interop.holographic_gdnruntime import load_runtime
    from holographic.io_and_interop.holographic_unicron import SafetensorWeights

    output, report = build_fixture(tmp_path)
    tokenizer = BPE.from_dir(output)
    ids = tokenizer.encode("portable Qwen fixture")
    runtime, config = load_runtime(output)

    assert report["layers"] == config["n_layers"] == 4
    assert ids and tokenizer.decode(ids) == "portable Qwen fixture"
    assert isinstance(runtime.w, SafetensorWeights)
    assert runtime.embed.dtype == np.float32
    logits = runtime.forward(ids[:12])
    assert logits.shape[-1] == 512 and np.isfinite(logits).all()


def test_prepend_matches_official_qwen_tensor_schema(tmp_path):
    from holographic.io_and_interop.holographic_gdnruntime import (
        load_runtime, load_weights_dir)
    from holographic.io_and_interop.holographic_prepend import prepend_layers

    output, _ = build_fixture(tmp_path)
    _runtime, config = load_runtime(output)
    weights = load_weights_dir(output)
    installed, installed_config = prepend_layers(weights, config, n=2)
    prefix = "model.language_model.layers.0."

    assert prefix + "linear_attn.in_proj_qkv.weight" in installed
    assert prefix + "linear_attn.in_proj_z.weight" in installed
    assert prefix + "linear_attn.in_proj_a.weight" in installed
    assert prefix + "linear_attn.in_proj_b.weight" in installed
    assert prefix + "linear_attn.in_proj_qkvz.weight" not in installed
    assert prefix + "linear_attn.in_proj_ba.weight" not in installed
    assert prefix + "linear_attn.conv1d.bias" not in installed
    assert installed[prefix + "mlp.gate_proj.weight"].shape[0] == \
        config["intermediate"]
    assert np.count_nonzero(installed[prefix + "input_layernorm.weight"]) == 0
    assert installed_config["n_layers"] == config["n_layers"] + 2


def test_installed_mini_qwen_reloads_and_generates_with_transformers(tmp_path):
    torch = __import__("pytest").importorskip("torch")
    safetensors = __import__("pytest").importorskip("safetensors.torch")
    transformers = __import__("pytest").importorskip("transformers")

    output, _ = build_fixture(tmp_path)
    corpus = tmp_path / "installation.txt"
    corpus.write_text("A portable checkpoint must reload through its public API. " * 900)
    installed = tmp_path / "installed-mini-qwen"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "assimilation" / "install.py"),
         "--experimental", str(output), str(installed), "--doc", str(corpus),
         "--device", "cpu"],
        cwd=ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr

    config_json = json.loads((installed / "config.json").read_text())
    text_config = transformers.Qwen3_5TextConfig(**config_json["text_config"])
    model = transformers.Qwen3_5ForCausalLM(text_config)
    checkpoint = safetensors.load_file(installed / "model.safetensors")
    text_checkpoint = {
        key.replace("model.language_model.", "model.", 1): value
        for key, value in checkpoint.items()
        if key.startswith("model.language_model.")
    }
    incompatible = model.load_state_dict(text_checkpoint, strict=False)
    model.tie_weights()

    assert not incompatible.unexpected_keys
    assert set(incompatible.missing_keys) <= {"lm_head.weight"}
    input_ids = torch.tensor([[65, 66, 67, 68]], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
        generated = model.generate(input_ids=input_ids, max_new_tokens=2,
                                   do_sample=False)
    assert torch.isfinite(logits).all()
    assert generated.shape[-1] == input_ids.shape[-1] + 2


def test_streamed_measure_matches_whole_forward(tmp_path):
    from holographic.io_and_interop.holographic_bpe import BPE
    from holographic.io_and_interop.holographic_gdnruntime import load_runtime
    from holographic.io_and_interop.holographic_measure import measure

    output, _ = build_fixture(tmp_path)
    ids = BPE.from_dir(output).encode("streamed evaluation stays paired " * 6)
    whole_runtime, _ = load_runtime(output)
    whole = measure(whole_runtime, ids, resamples=50)
    runner = load_script("qwen_acceptance_runner",
                         ROOT / "experiments" / "qwen35_acceptance" / "run.py")
    streamed_runtime, _ = load_runtime(output)
    streamed = runner.streamed_measure(streamed_runtime, ids, chunk_size=11,
                                       resamples=50)

    assert len(whole["nll"]) == len(streamed["nll"])
    assert np.allclose(whole["nll"], streamed["nll"], rtol=2e-5, atol=2e-5)


def test_runner_records_monotonic_stage_and_chunk_progress(tmp_path):
    runner = load_script("qwen_acceptance_progress_runner",
                         ROOT / "experiments" / "qwen35_acceptance" / "run.py")

    class State:
        logits = None

    class Runtime:
        lm_head = np.zeros((64, 8), np.float32)
        cfg = {"hidden": 8, "intermediate": 24}

        def forward(self, ids, collect_state=False, resume=None):
            logits = np.zeros((len(ids), 64), np.float32)
            state = State()
            state.logits = logits[-1]
            return (logits, state) if collect_state else logits

    recorder = runner.ProgressRecorder(tmp_path / "progress.jsonl")
    with recorder.stage("fixture_evaluation"):
        result = runner.streamed_measure(
            Runtime(), list(range(21)), chunk_size=4, max_chunk_size=8,
            recorder=recorder, phase="fixture", resamples=20)

    records = [json.loads(line) for line in
               (tmp_path / "progress.jsonl").read_text().splitlines()]
    stamps = [item["monotonic_seconds"] for item in records]
    chunks = [item for item in records if item["kind"] == "evaluation_chunk"]
    assert stamps == sorted(stamps)
    assert chunks[-1]["eval_tokens_complete"] == result["n_tokens"] == 20
    assert chunks[-1]["eval_tokens_total"] == 20
    assert recorder.timings["fixture_evaluation"] >= 0


def test_memory_plan_scales_chunks_and_parallel_admission_is_conservative():
    runner = load_script("qwen_acceptance_memory_runner",
                         ROOT / "experiments" / "qwen35_acceptance" / "run.py")

    class Weight:
        shape = (248320, 1024)

    class Runtime:
        lm_head = Weight()
        cfg = {"hidden": 1024, "intermediate": 3584}

    plan = runner.memory_chunk_plan(
        Runtime(), requested=128, max_chunk=512, budget_fraction=0.20,
        workers=2, available_mb=128_000)
    assert plan["selected_chunk_size"] == 512
    assert plan["estimated_chunk_working_mb"] > 0
    assert runner.parallel_evaluation_feasible(2200, 2300, 128_000)
    assert not runner.parallel_evaluation_feasible(2200, 2300, 12_000)


def test_sequential_rule_can_only_reject_after_a_thousand_paired_tokens():
    runner = load_script("qwen_acceptance_sequential_runner",
                         ROOT / "experiments" / "qwen35_acceptance" / "run.py")
    original = np.full(1024, 3.0)
    clearly_worse = original + 0.2
    equivalent = original.copy()
    limit = np.log1p(0.01)

    rejected = runner.sequential_rejection_test(
        clearly_worse, original, limit, look_index=0, total_looks=4,
        resamples=100)
    not_rejected = runner.sequential_rejection_test(
        equivalent, original, limit, look_index=0, total_looks=4,
        resamples=100)
    assert rejected["reject"] is True
    assert not_rejected["reject"] is False
    assert rejected["alpha_spent"] == 0.0125
    assert runner.parse_sequential_looks("1024,2048,3072,4096", 4096) == \
        [1024, 2048, 3072, 4096]
    with __import__("pytest").raises(ValueError, match="at least 1000"):
        runner.parse_sequential_looks("512,4096", 4096)


def test_parallel_paired_evaluation_preserves_order(tmp_path, monkeypatch):
    import importlib

    runner = importlib.import_module("experiments.qwen35_acceptance.run")
    from holographic.io_and_interop.holographic_bpe import BPE
    from holographic.io_and_interop.holographic_ccrun import cc_available

    if cc_available() is None:
        __import__("pytest").skip("parallel native-cache test needs a C compiler")
    monkeypatch.setenv("LECORE_GDN_BACKEND", "c")
    monkeypatch.setenv("LECORE_CC_CACHE", str(tmp_path / "cold-cc-cache"))

    output, _ = build_fixture(tmp_path)
    ids = BPE.from_dir(output).encode("paired worker ordering " * 4)[:33]
    recorder = runner.ProgressRecorder(tmp_path / "paired-progress.jsonl")
    result = runner.paired_evaluation(
        output, output, ids, chunk_size=4, max_chunk_size=8,
        memory_budget_fraction=0.20, recorder=recorder,
        sequential_looks=[len(ids) - 1], allow_early_rejection=False,
        regression_limit=np.log1p(0.01), gdn_backend="c", parallel=True)

    assert result["parallel"] is True
    assert result["before"]["n_tokens"] == len(ids) - 1
    assert np.array_equal(result["before"]["nll"], result["after"]["nll"])
    assert result["early_rejection_at"] is None
    assert result["plans"]["original"] == result["plans"]["installed"] == \
        result["common_chunk_plan"]
    assert set(result["acceleration_reports"]) == {"original", "installed"}
    assert all(report["full_sequence_gdn_recurrence"]["active"] == "c"
               for report in result["acceleration_reports"].values())


def test_progress_recorder_refreshes_s3_after_each_chunk(tmp_path, monkeypatch):
    runner = load_script("qwen_acceptance_upload_runner",
                         ROOT / "experiments" / "qwen35_acceptance" / "run.py")
    calls = []

    class Completed:
        returncode = 0
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Completed()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    recorder = runner.ProgressRecorder(
        tmp_path / "progress.jsonl",
        upload_uri="s3://evidence/future/progress.jsonl")
    recorder.emit("evaluation_chunk", eval_tokens_complete=128)
    assert recorder.flush_upload()
    assert calls[0][0][:3] == ["aws", "s3", "cp"]
    assert calls[0][0][-2:] == ["s3://evidence/future/progress.jsonl",
                                "--only-show-errors"]
    assert recorder.upload_attempts == 1
    assert recorder.upload_failures == 0
    assert recorder.upload_requests == 1


def test_generator_emits_complete_ilxyr_contract(tmp_path):
    output, _ = build_fixture(tmp_path)
    installation_corpus = tmp_path / "installation.txt"
    evaluation_corpus = tmp_path / "evaluation.txt"
    installation_corpus.write_text(
        "Installation grounding is separate from evaluation. " * 400)
    evaluation_corpus.write_text(
        "Evidence must be paired, replayable, and recorded. " * 400)
    project = tmp_path / "project"
    command = [sys.executable,
               str(ROOT / "experiments" / "qwen35_acceptance" / "generate.py"),
               str(output), str(installation_corpus), str(evaluation_corpus),
               str(project), "--min-tokens", "1000"]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True,
                               text=True, check=False)
    assert completed.returncode == 0, completed.stderr

    contract = load_script("qwen_acceptance_contract",
                           ROOT / "experiments" / "qwen35_acceptance" / "contract.py")
    experiment = json.loads((project / "experiment.json").read_text())
    manifest = json.loads((project / "project.json").read_text())
    assert [metric["name"] for metric in experiment["metrics"]] == list(contract.METRIC_NAMES)
    assert experiment["execution"]["program"] == str(Path(sys.executable).absolute())
    assert "--research-spectral" not in experiment["execution"]["args"]
    assert experiment["execution"]["args"][3:5] == [
        str(installation_corpus.resolve()), str(evaluation_corpus.resolve())]
    assert len(experiment["datasets"]) == 2
    assert manifest["corpora"]["installation"]["sha256"] != \
        manifest["corpora"]["evaluation"]["sha256"]
    assert "corpus_sha256" not in manifest
    assert {item["path"] for item in manifest["model_files"]} >= {
        "config.json", "tokenizer.json", "model.safetensors",
    }
    assert len(manifest["commands"]) == 12
    assert manifest["runner_policy"]["early_acceptance_allowed"] is False
    assert manifest["runner_policy"]["accepted_requires_full_tokens"] == 1000
    assert manifest["runner_policy"]["sequential_looks"] == [1000]
    assert manifest["runner_policy"]["gdn_backend"] == "c"
    assert manifest["runner_policy"]["max_chunk_size"] == 128
    assert len(manifest["runner_policy_digest"]) == 64
    assert manifest["runner_policy_digest"][:12] in manifest["experiment_id"]
    gdn_index = experiment["execution"]["args"].index("--gdn-backend")
    assert experiment["execution"]["args"][gdn_index + 1] == "c"
    assert "--allow-early-rejection" in experiment["execution"]["args"]
    for name in ("hypothesis.json", "foundation.json", "engineering-review.json",
                 "experiment-design.json", "forecast-empirical.json",
                 "forecast-mechanistic.json", "funding.json"):
        assert (project / name).is_file()


def test_generator_binds_benchmark_before_increasing_chunk(tmp_path):
    output, _ = build_fixture(tmp_path)
    installation = tmp_path / "installation.txt"
    evaluation = tmp_path / "evaluation.txt"
    installation.write_text("Install grounding. " * 400)
    evaluation.write_text("Held-out benchmark corpus. " * 400)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "schema": "lecore.qwen_runtime_benchmark.v1",
        "inputs": {
            "weight_hashes_included": True,
            "corpus": {"sha256": digest(evaluation)},
            "model_files": [
                {"name": path.name, "sha256": digest(path)}
                for path in sorted(output.iterdir()) if path.is_file()
            ],
        },
        "speed_and_cost_projection": {"recommended_chunk_size": 256},
    }
    report_path = tmp_path / "benchmark.json"
    report_path.write_text(json.dumps(report))
    project = tmp_path / "project"
    completed = subprocess.run(
        [sys.executable,
         str(ROOT / "experiments" / "qwen35_acceptance" / "generate.py"),
         str(output), str(installation), str(evaluation), str(project),
         "--min-tokens", "1000", "--benchmark-report", str(report_path)],
        cwd=tmp_path, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((project / "project.json").read_text())
    assert manifest["runner_policy"]["max_chunk_size"] == 256
    assert manifest["runner_policy"]["benchmark_report_sha256"] == digest(report_path)

    refused = subprocess.run(
        [sys.executable,
         str(ROOT / "experiments" / "qwen35_acceptance" / "generate.py"),
         str(output), str(installation), str(evaluation), str(tmp_path / "bad"),
         "--min-tokens", "1000", "--max-chunk-size", "256"],
        cwd=tmp_path, capture_output=True, text=True, check=False)
    assert refused.returncode != 0
    assert "requires --benchmark-report" in refused.stderr


def test_generator_preserves_virtual_environment_python_path(tmp_path):
    output, _ = build_fixture(tmp_path)
    installation_corpus = tmp_path / "installation.txt"
    evaluation_corpus = tmp_path / "evaluation.txt"
    installation_corpus.write_text("Installation material. " * 400)
    evaluation_corpus.write_text("Held-out evaluation material. " * 400)
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable).resolve())
    project = tmp_path / "project"

    completed = subprocess.run(
        [str(venv_python),
         str(ROOT / "experiments" / "qwen35_acceptance" / "generate.py"),
         str(output), str(installation_corpus), str(evaluation_corpus),
         str(project), "--python", str(venv_python), "--min-tokens", "1000"],
        cwd=tmp_path, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr

    experiment = json.loads((project / "experiment.json").read_text())
    assert experiment["execution"]["program"] == str(venv_python.absolute())
    assert experiment["execution"]["program"] != str(venv_python.resolve())


def test_generator_can_freeze_a_third_formal_attempt(tmp_path):
    output, _ = build_fixture(tmp_path)
    installation_corpus = tmp_path / "installation.txt"
    evaluation_corpus = tmp_path / "evaluation.txt"
    installation_corpus.write_text("Installation material. " * 400)
    evaluation_corpus.write_text("Held-out evaluation material. " * 400)
    project = tmp_path / "project-v3"

    completed = subprocess.run(
        [sys.executable,
         str(ROOT / "experiments" / "qwen35_acceptance" / "generate.py"),
         str(output), str(installation_corpus), str(evaluation_corpus),
         str(project), "--min-tokens", "1000", "--experiment-version", "3"],
        cwd=tmp_path, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr

    experiment = json.loads((project / "experiment.json").read_text())
    manifest = json.loads((project / "project.json").read_text())
    assert experiment["id"].endswith(".v3.acceptance")
    assert experiment["evidence_authority"]["provenance"]["checker"].endswith("/v3")
    assert manifest["experiment_version"] == 3


def test_generator_rejects_reused_installation_and_evaluation_corpus(tmp_path):
    output, _ = build_fixture(tmp_path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("The exact same bytes may not serve both experiment roles. " * 200)
    command = [sys.executable,
               str(ROOT / "experiments" / "qwen35_acceptance" / "generate.py"),
               str(output), str(corpus), str(corpus), str(tmp_path / "project"),
               "--min-tokens", "1000"]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True,
                               text=True, check=False)
    assert completed.returncode == 2
    assert "distinct contents" in completed.stderr


def test_model_manifest_binds_processor_and_template_files(tmp_path):
    output, _ = build_fixture(tmp_path)
    (output / "preprocessor_config.json").write_text('{"size": 32}\n')
    (output / "chat_template.jinja").write_text("{{ messages }}\n")
    (output / ".cache-marker").write_text("local cache state\n")
    generator = load_script(
        "qwen_acceptance_generator",
        ROOT / "experiments" / "qwen35_acceptance" / "generate.py")

    _digest, files = generator.model_manifest(output)
    names = {item["path"] for item in files}
    assert "preprocessor_config.json" in names
    assert "chat_template.jinja" in names
    assert ".cache-marker" not in names


def test_risky_entrypoints_are_explicitly_opt_in(tmp_path, capsys):
    installer = subprocess.run(
        [sys.executable, str(ROOT / "assimilation" / "install.py")],
        cwd=tmp_path, capture_output=True, text=True, check=False)
    assert installer.returncode == 2
    assert "--experimental" in installer.stderr

    driver = load_script("qwen_assimilation_driver", ROOT / "assimilation" / "run.py")
    driver.download = lambda _model, _work: (str(tmp_path), ["fixture.safetensors"])
    driver.assimilate = lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("spectral path ran without opt-in"))
    assert driver.main(["--workdir", str(tmp_path)]) == 0
    assert "spectral filtering is disabled by default" in capsys.readouterr().out
    assert "[ ! -d" not in (ROOT / "assimilation" / "install.sh").read_text()

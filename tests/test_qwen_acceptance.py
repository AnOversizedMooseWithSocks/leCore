"""Portable gates for the Qwen3.5/ilxyr experiment contract."""

import importlib.util
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
    assert experiment["execution"]["program"] == str(Path(sys.executable).resolve())
    assert "--research-spectral" not in experiment["execution"]["args"]
    assert experiment["execution"]["args"][3:5] == [
        str(installation_corpus.resolve()), str(evaluation_corpus.resolve())]
    assert len(experiment["datasets"]) == 2
    assert manifest["corpora"]["installation"]["sha256"] != \
        manifest["corpora"]["evaluation"]["sha256"]
    assert "corpus_sha256" not in manifest
    assert len(manifest["commands"]) == 12
    for name in ("hypothesis.json", "foundation.json", "engineering-review.json",
                 "experiment-design.json", "forecast-empirical.json",
                 "forecast-mechanistic.json", "funding.json"):
        assert (project / name).is_file()


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

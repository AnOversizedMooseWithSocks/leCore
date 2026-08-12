#!/usr/bin/env python3
"""Run the full Qwen3.5 installation acceptance contract for ilxyr."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (str(HERE), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

from contract import METRIC_NAMES  # noqa: E402


def log(message):
    print(message, file=sys.stderr, flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_rss_mb(who):
    try:
        import resource
        if isinstance(who, str):
            who = getattr(resource, who)
        raw = float(resource.getrusage(who).ru_maxrss)
        return raw / (1024.0 * 1024.0) if platform.system() == "Darwin" else raw / 1024.0
    except Exception:
        return 0.0


def token_losses(logits, targets):
    logits = np.asarray(logits, np.float64)
    targets = np.asarray(targets, np.int64)
    maximum = logits.max(axis=-1, keepdims=True)
    lse = np.log(np.exp(logits - maximum).sum(axis=-1)) + maximum.ravel()
    return lse - logits[np.arange(len(targets)), targets]


def streamed_measure(runtime, token_ids, chunk_size=128, resamples=800):
    """Measure thousands of tokens without materializing all vocabulary logits."""
    from holographic.io_and_interop.holographic_measure import summarize_nll

    ids = list(map(int, token_ids))
    if len(ids) < 2:
        raise ValueError("streamed measurement needs at least two token ids")
    state = None
    losses = []
    for start in range(0, len(ids), int(chunk_size)):
        chunk = ids[start:start + int(chunk_size)]
        if state is None:
            if len(chunk) < 2:
                continue
            logits, state = runtime.forward(chunk, collect_state=True)
            losses.append(token_losses(logits[:-1], chunk[1:]))
        else:
            boundary_logits = np.asarray(state.logits)
            logits, state = runtime.forward(chunk, collect_state=True, resume=state)
            prediction = np.concatenate([boundary_logits[None, :], logits[:-1]], axis=0)
            losses.append(token_losses(prediction, chunk))
        log("  measured %d/%d input tokens" % (min(start + len(chunk), len(ids)), len(ids)))
    return summarize_nll(np.concatenate(losses), resamples=resamples, seed=0)


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


def run_installer(model_dir, installed_dir, corpus, transcript):
    command = [
        sys.executable,
        str(REPO / "assimilation" / "install.py"),
        "--experimental",
        str(model_dir),
        str(installed_dir),
        "--doc",
        str(corpus),
        "--device",
        "cpu",
    ]
    log("running experimental installer; transcript: %s" % transcript)
    transcript.parent.mkdir(parents=True, exist_ok=True)
    with transcript.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=REPO, stdout=handle,
                                   stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        tail = transcript.read_text(encoding="utf-8", errors="replace")[-4000:]
        log(tail)
        raise RuntimeError("experimental installer exited %d" % completed.returncode)


def official_output_smokes(installed_dir):
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            installed_dir, torch_dtype=torch.float32, trust_remote_code=True).eval()
    except Exception:
        from transformers import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            installed_dir, torch_dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained(installed_dir, trust_remote_code=True)

    text_inputs = processor(text="Explain what a checksum protects.", return_tensors="pt")
    with torch.no_grad():
        text_out = model.generate(**text_inputs, max_new_tokens=4)
    text_pass = int(text_out.shape[-1] > text_inputs["input_ids"].shape[-1])

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
    vision_pass = int(vision_out.shape[-1] > vision_inputs["input_ids"].shape[-1])
    peak_gpu = (float(torch.cuda.max_memory_allocated()) / 1e6
                if torch.cuda.is_available() else 0.0)
    del model, processor, text_inputs, text_out, vision_inputs, vision_out
    gc.collect()
    return float(text_pass), float(vision_pass), peak_gpu


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
        REPO / "assimilation" / "install.py",
        REPO / "holographic" / "io_and_interop" / "holographic_measure.py",
        REPO / "holographic" / "io_and_interop" / "holographic_gdnruntime.py",
    ]
    return float(not dirty), {
        "repository": repository,
        "commit": commit,
        "artifacts": [{"path": str(path.relative_to(REPO)),
                       "sha256": sha256_file(path)} for path in paths],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("installed_dir", type=Path)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--min-tokens", type=int, default=4096)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--max-regression", type=float, default=0.01)
    ap.add_argument("--logit-tolerance", type=float, default=1e-3)
    args = ap.parse_args(argv)

    model_dir = args.model_dir.resolve()
    installed_dir = args.installed_dir.resolve()
    corpus = args.corpus.resolve()
    if installed_dir.exists() and any(installed_dir.iterdir()):
        raise SystemExit("installed_dir must be absent or empty: %s" % installed_dir)
    if int(args.min_tokens) < 1000:
        raise SystemExit("--min-tokens must be at least 1000 for an acceptance run")

    from holographic.io_and_interop.holographic_bpe import BPE
    from holographic.io_and_interop.holographic_gdnruntime import load_runtime
    from holographic.io_and_interop.holographic_measure import better_than

    text = corpus.read_text(encoding="utf-8", errors="replace")
    token_ids = BPE.from_dir(model_dir).encode(text)
    needed = int(args.min_tokens) + 1
    if len(token_ids) < needed:
        raise SystemExit("corpus produced %d tokens; acceptance requires at least %d"
                         % (len(token_ids), needed))
    token_ids = token_ids[:needed]

    source_clean, source = source_snapshot()
    tok_pass, ref_error, ref_pass = reference_parity(
        model_dir, token_ids, args.logit_tolerance)

    original, _ = load_runtime(model_dir)
    before = streamed_measure(original, token_ids, chunk_size=args.chunk_size)
    del original
    gc.collect()

    artifact_dir = installed_dir.parent / (installed_dir.name + ".acceptance")
    run_installer(model_dir, installed_dir, corpus, artifact_dir / "install.log")
    child_peak = peak_rss_mb("RUSAGE_CHILDREN")

    installed, _ = load_runtime(installed_dir)
    reload_logits = np.asarray(installed.forward(token_ids[:16]))
    reload_pass = float(np.isfinite(reload_logits).all())
    after = streamed_measure(installed, token_ids, chunk_size=args.chunk_size)
    del installed, reload_logits
    gc.collect()

    comparison = better_than(after, before, resamples=1200, seed=0)
    regression_limit = math.log1p(float(args.max_regression))
    statistical_pass = float(comparison["ci_hi_nats"] <= regression_limit)
    text_pass, vision_pass, peak_gpu = official_output_smokes(installed_dir)

    checkpoint_mb = sum(path.stat().st_size for path in
                        installed_dir.glob("*.safetensors")) / 1e6
    peak_rss = max(peak_rss_mb("RUSAGE_SELF"), child_peak)
    required = [source_clean, tok_pass, ref_pass, statistical_pass,
                reload_pass, text_pass, vision_pass]
    metrics = {
        "acceptance_pass": float(all(v >= 1.0 for v in required)),
        "source_clean": source_clean,
        "spectral_filtering_enabled": 0.0,
        "experimental_installer_used": 1.0,
        "tokenizer_parity_pass": tok_pass,
        "reference_logit_parity_pass": ref_pass,
        "reference_logit_relative_error": ref_error,
        "eval_tokens": float(comparison["n_tokens"]),
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
        "text_generation_pass": text_pass,
        "vision_smoke_pass": vision_pass,
    }
    if tuple(metrics) != METRIC_NAMES:
        raise RuntimeError("runner metric order/contract drift: %r != %r"
                           % (tuple(metrics), METRIC_NAMES))
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("non-finite acceptance metric")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "source": source}, indent=2, sort_keys=True))
    print(json.dumps({"metrics": metrics, "source": source}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

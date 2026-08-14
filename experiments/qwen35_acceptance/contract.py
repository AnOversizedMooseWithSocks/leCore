"""Frozen metric and dependency contract for Qwen acceptance experiments."""

import hashlib
import json
from pathlib import Path


# v4 binds these exact public-interface dependencies into the admitted
# experiment identity.  The runner independently verifies the same versions
# before model work; keeping the declared contract here lets the generator hash
# the dependency boundary without importing the heavyweight runtime module.
OFFICIAL_DEPENDENCY_VERSIONS = {
    "Pillow": "12.3.0",
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "transformers": "5.14.0",
}

RUNNER_POLICY_SCHEMA = "lecore.qwen35.runner-policy.v4"


def model_manifest(model_dir):
    """Content-bind every public top-level model/processor file."""
    root = Path(model_dir)
    paths = sorted(path for path in root.iterdir()
                   if path.is_file() and not path.name.startswith("."))
    if not any(path.suffix == ".safetensors" for path in paths):
        raise ValueError("no safetensors checkpoint in %s" % root)

    def digest(path):
        value = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()

    records = [{"path": path.name, "sha256": digest(path),
                "bytes": path.stat().st_size} for path in paths]
    canonical = json.dumps(
        records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), records


METRIC_SPECS = [
    {"name": "acceptance_pass", "unit": "boolean", "description": "All mandatory Qwen installation acceptance gates passed, encoded as 0 or 1."},
    {"name": "source_clean", "unit": "boolean", "description": "The tracked leCore checkout was clean when the run began, encoded as 0 or 1."},
    {"name": "spectral_filtering_enabled", "unit": "boolean", "description": "Whether research-only spectral filtering was used; the acceptance path requires 0."},
    {"name": "experimental_installer_used", "unit": "boolean", "description": "The explicitly acknowledged layer-prepending installer ran, encoded as 0 or 1."},
    {"name": "tokenizer_parity_pass", "unit": "boolean", "description": "leCore and the official Transformers tokenizer produced identical reference token IDs."},
    {"name": "reference_logit_parity_pass", "unit": "boolean", "description": "Pre-install leCore logits matched the official Transformers text model within the frozen tolerance."},
    {"name": "reference_logit_relative_error", "unit": "ratio", "description": "Maximum absolute pre-install logit error divided by the maximum absolute reference logit."},
    {"name": "eval_tokens", "unit": "tokens", "description": "Paired token positions included in the streamed evaluation."},
    {"name": "full_evaluation_pass", "unit": "boolean", "description": "The complete preregistered paired token count was measured; every accepted outcome requires 1."},
    {"name": "parallel_evaluation_used", "unit": "boolean", "description": "Original and installed checkpoints were evaluated concurrently after the memory admission check."},
    {"name": "sequential_early_rejection", "unit": "boolean", "description": "A frozen multiplicity-corrected interim look proved NO-GO before the final look; this can never produce GO."},
    {"name": "sequential_looks_completed", "unit": "looks", "description": "Number of preregistered paired sequential looks completed."},
    {"name": "evaluation_wall_seconds", "unit": "seconds", "description": "Monotonic wall time for the paired original/installed evaluation stage."},
    {"name": "native_gdn_acceleration_active", "unit": "boolean", "description": "Both paired evaluators used the requested parity-gated native Gated-DeltaNet recurrence rather than its safe NumPy fallback."},
    {"name": "original_perplexity", "unit": "perplexity", "description": "Original checkpoint perplexity on the frozen corpus and chunking procedure."},
    {"name": "installed_perplexity", "unit": "perplexity", "description": "Installed checkpoint perplexity on the same token positions."},
    {"name": "perplexity_delta_pct", "unit": "percent", "description": "Installed minus original perplexity as a percentage of original."},
    {"name": "paired_ci_lo_nats", "unit": "nats_per_token", "description": "Lower 95 percent paired moving-block-bootstrap bound for installed minus original NLL."},
    {"name": "paired_ci_hi_nats", "unit": "nats_per_token", "description": "Upper 95 percent paired moving-block-bootstrap bound for installed minus original NLL."},
    {"name": "statistical_gate_pass", "unit": "boolean", "description": "The paired upper confidence bound stayed within the preregistered maximum regression."},
    {"name": "paired_block_length", "unit": "tokens", "description": "Moving-block length inferred from autocorrelation in paired token loss differences."},
    {"name": "paired_effective_tokens", "unit": "tokens", "description": "Effective paired sample size after serial-correlation adjustment."},
    {"name": "peak_rss_mb", "unit": "megabytes", "description": "Conservative peak resident-memory bound: the largest single-process phase or the summed evaluator-process peaks when evaluation ran concurrently."},
    {"name": "peak_gpu_mb", "unit": "megabytes", "description": "Peak accelerator allocation reported by PyTorch, or 0 when no accelerator was used."},
    {"name": "emitted_checkpoint_mb", "unit": "megabytes", "description": "Total safetensors size of the emitted installed checkpoint."},
    {"name": "reload_pass", "unit": "boolean", "description": "The emitted checkpoint reloaded from disk and produced finite logits."},
    {"name": "official_reload_pass", "unit": "boolean", "description": "The emitted checkpoint reloaded through the official Transformers Qwen model without missing, unexpected, or mismatched layer tensors."},
    {"name": "text_generation_pass", "unit": "boolean", "description": "The emitted checkpoint generated text through the official Transformers model."},
    {"name": "vision_smoke_pass", "unit": "boolean", "description": "The emitted checkpoint accepted a synthetic image through the official Qwen vision-language processor and generated a token."},
]

METRIC_NAMES = tuple(item["name"] for item in METRIC_SPECS)

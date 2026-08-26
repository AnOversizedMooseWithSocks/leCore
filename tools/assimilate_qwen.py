#!/usr/bin/env python3
"""Assimilate a real checkpoint (e.g. Qwen3.5-0.8B) with leCore's Unicron pass.

Run on a machine that has the weights (this repo's sandbox cannot reach HF):

    # 1. get the weights (either form works)
    #    hf download Qwen/Qwen3.5-0.8B --include "*.safetensors"     (HF)
    #    ...or an F16 / Q8_0 .gguf                                   (llama.cpp)
    # 2. assimilate
    python3 tools/assimilate_qwen.py model.safetensors model_unicron.safetensors
    # 3. MEASURE -- the step that makes it real. The output is UNVERIFIED until
    #    perplexity / task eval runs on BOTH files on your runtime, e.g.:
    #    llama-perplexity -m before.gguf -f wiki.test.raw
    #    llama-perplexity -m after.gguf  -f wiki.test.raw
    #    Ship only if the delta is acceptable. Believe the measurement.

What the pass does (see holographic_unicron.assimilate_model for the full WHY):
  * name-policy skip: embeddings / lm_head / conv / norms untouched, no SVD spent
  * per projection: Marchenko-Pastur filter -- keep the learned spectral outliers,
    drop the still-random bulk; "shrink" debiases spikes by the noise floor
  * guard: layers whose outliers carry <1% of energy pass through (random != useless)
  * huge matrices go through seeded randomized SVD (Halko et al. 2011)
  * output: DENSE tensors under ORIGINAL names/shapes -- loads wherever input loaded

Multi-shard HF checkpoints: pass each shard; names are disjoint across shards so
per-shard processing is exact, not an approximation.
"""
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from holographic.io_and_interop.holographic_unicron import assimilate_model


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    pin, pout = sys.argv[1], sys.argv[2]
    t0 = time.time()
    _, rep = assimilate_model(pin, out_path=pout)
    ranks = sorted(li["rank"] for li in rep["layers"].values()) or [0]
    print("assimilated in %.1fs" % (time.time() - t0))
    print("  filtered: %d matrices | policy-skipped: %d | guarded: %d"
          % (rep["filtered"], len(rep["skipped"]), len(rep["guarded"])))
    print("  effective ranks kept (min/median/max): %d / %d / %d"
          % (ranks[0], ranks[len(ranks) // 2], ranks[-1]))
    print("  wrote: %s" % rep["out_path"])
    print("  " + rep["verify"])


if __name__ == "__main__":
    main()

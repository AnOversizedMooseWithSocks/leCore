# Installing leCore into a model with Unicron

One command turns a stock Llama-family checkpoint into a model that carries leCore's
primitives natively: **no leCore imports at inference, no prompt asking for it** — ordinary
weights that run in any harness.

    python3 tools/unicron_install.py <model_dir_or_file> <out_dir> \
        --from-partition ~/.lecore --budget auto --spread 3 \
        --bake-algebra --bake-swarm 4 --gain 0.0 --cartridge lecore_cart

Inputs accepted: a single `.safetensors`, a **sharded** directory (reads the index; names
are disjoint across shards so per-shard processing is exact), or a `.gguf`. The on-disk
dtype is **preserved** — writing a bf16 checkpoint back as float32 doubles the file, which
Unicron learned the hard way on a 1.75 GB Qwen.

## What gets installed

1. **Facts** — batched Kohonen/MEMIT solve on `mlp.down_proj`, from **taught-only**
   partition facts (a model-cached answer is provisional and is never written into weights
   as if established). Keys are **unit-normalized** SwiGLU gated activations; values are
   tied-embedding targets raised just past the current argmax.
   `--spread N` shares each fact across N consecutive layers (`r^l = residual/(L-l+1)`), so
   no single matrix absorbs the whole edit — measured 0.084 on one layer becoming
   0.028 / 0.043 / 0.092 across three.
2. **Algebra** — `bind` is a **circulant matrix**, verified identical to leCore's `bind` to
   **9.4e-17**. `bundle` is the residual stream, already free. Unbinding by involution is
   HRR-*approximate* and is reported as a cosine, never as an error.
3. **Swarm** — a routed bank of leCore circuits with a content-keyed gate, so the model runs
   a swarm internally, per token. It **arrives off** (`--gain 0.0` is a bit-exact no-op) and
   the byte budget is reported: +196,608 params / 768 KB for 4 experts.

## The five-point health gate — pass or refuse

| # | Check | Default |
|---|---|---|
| 1 | seed-identical (byte-identical rerun) | required |
| 2 | key is unit-normalized | structural |
| 3 | `|delta|/|W|` bounded | < 0.10 |
| 4 | stable rank preserved | drop < 30% |
| 5 | locality drift on 50 unrelated keys | < 0.10 |

**Locality is the budget.** Efficacy is easy — every fact hits at every N — but drift and
norm climb with N. `--budget auto` **bisects** for the largest fact count that still passes
instead of guessing a capacity number. Use `--plan` for a dry run that writes nothing.

## Cartridges

`--cartridge` stores the **deltas** (which compose, scale and stack) *and* the **originals**
of the edited tensors. Delta-subtraction alone cannot be exact once the model is written at
its source dtype — preserving bf16 rounds the weights and left a 2.4e-04 residual — so the
originals make revert exact at any dtype:

    python3 tools/unicron_install.py <installed> <out> --revert lecore_cart.npz [--revert-swarm]

`--revert-swarm` also trims baked swarm neurons back to the pre-bake shapes recorded in the
cartridge (baking widens `down_proj`, e.g. 384 → 896).

## After the install

`out_dir/lecore.json` records what was installed, the full gate table, sample facts, and
the Ouroboros config. Attach the result with the same governance as the hosted service:

    from tools.local_rung import LocalRung
    rung = LocalRung.gdn("out_dir", partition="~/.lecore")   # Ouroboros in the forward pass
    m.boot(partition="~/.lecore", doctrine=True, llm=rung); m.zoo_attach(rung)

## The honest boundary — printed on every run, written into the manifest

Structure, keying, spectral health, locality and determinism are **proven** by the install.
**Efficacy generalization (paraphrases) and perplexity retention are not** — they require
real trained weights on a real runtime. On the random-init test model the budget bisects to
N=2; that is the random-weight ceiling, not the real one. Run the bisect on real weights and
read the number rather than trusting an extrapolation.

Recommended first real target: **SmolLM2-135M** (plain Llama, tied embeddings make the value
target exact, iterates on CPU). Then **Qwen3.5-0.8B** with `--root model.language_model.`.

## The qwen3.5:0.8b live-test runbook (cp65)

Run `python tools/unicron_preflight.py <model_dir>` first — every check corresponds to a
defect that actually happened in this project. A green preflight prints the exact
sequence: unicron_install with `--budget auto --spread 3 --bake-algebra --cartridge`
(AlphaEdit null-space protection is on in the FACTS solve — `preservation_after` in the
report must read ~0; measured 0.1014 → 2.69e-09 on the reference install), then
`assimilation/galvatron.py OUT --install` for the 13 residents (Ouroboros at layer n//2,
dk 64, decay 0.98; expect AUDIT 4/4), then measure on the real runtime the two things a
sandbox cannot prove: efficacy generalization and perplexity retention. The cartridge
reverts exactly (~4e-09).

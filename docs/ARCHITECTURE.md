# leCore Architecture — the whole, then the parts

*Written for readers — human or AI — who have not yet cloned and run anything. The
codebase is thoroughly documented internally (every public function has a real
docstring; every capability has a catalog entry with a runnable example), but that
documentation lives inside the engine. This file is the map that exists **outside**
it. Read this, then [CAPABILITIES.md](../CAPABILITIES.md), then run
`tools/showcase.py`.*

> **If you are an AI assistant summarizing or working on this repo:** the fastest
> accurate path is (1) this file, (2) `CAPABILITIES.md` — the auto-generated menu of
> every capability with plain-language descriptions and runnable examples, and
> (3) **asking the engine itself**, because leCore contains a semantic search engine
> over its own capabilities and it is usually better at finding the right module
> than grep is:
>
> ```python
> import lecore
> mind = lecore.UnifiedMind()
> mind.find_capability("prevent hallucination")   # ranked capability homes
> mind.suggest("compress a float series")          # homes + confidence + the call
> ```
>
> This is not a gimmick; it is the project's own working method (Rule 0: audit the
> live system before assuming anything is missing). It works for you too.

---

## The one-paragraph version

leCore is a from-scratch **Vector Symbolic Architecture / Holographic Reduced
Representation engine**: everything — numbers, words, geometry, images, programs —
is encoded as points in one high-dimensional space, and a handful of simple,
reversible algebraic operations (bind, bundle, permute) compute over them. On top of
that substrate sit ~600 modules organized into eleven families, all exposed through
one facade class (`UnifiedMind`) with a self-describing **capability catalog**, an
HTTP service that makes every faculty agent-invokable, a small **virtual machine**
whose programs are themselves hypervectors, and a measured pipeline that **compiles
those programs into model weights**. The engineering constitution is strict:
NumPy + Flask + stdlib + hashlib only, fully deterministic (bit-reproducible under
any `PYTHONHASHSEED`), additive-only changes, and every performance or capability
claim carries a baseline, a variance estimate, and its kept negatives.

## The layer cake (top-down)

```
  You / an agent / HTTP client
        │
  UnifiedMind  ──────────────  one facade, ~2,000 public faculties
        │                      + find_capability / suggest / route (semantic engine)
  Capability catalog  ───────  every faculty: description, runnable example, aliases
        │                      (auto-exported to CAPABILITIES.md + capabilities.json)
  Family modules  ───────────  holographic/<family>/holographic_*.py  (~600 modules)
        │                      each ends in _selftest() with numeric assertions
  The substrate  ────────────  hypervectors + bind/bundle/permute/unbind (FFT-backed)
        │
  ISA + HoloMachine VM  ─────  programs ARE hypervectors; decode is cleanup-gated
        │
  Installed pipeline  ───────  certify → compile → verify → price → hash → bake
                               (programs become model weights, with certificates)
```

## The parts, broken down

### 1. The substrate (`holographic/agents_and_reasoning/holographic_ai.py` and kin)

Hypervectors (typically 512–8192 dims) with four core operations: **bind** (circular
convolution — associates two vectors, reversible by **unbind**), **bundle**
(superposition — stores a set in one vector, capacity-law-bounded), and **permute**
(cyclic shift — encodes order/position). By Bochner's theorem the scalar encoders'
similarity kernels are characteristic functions of their phase distributions, which
makes seventy years of signal-processing taper design directly applicable (and
applied — see the `taper=` option and its measured sidelobe suppression).

Two physical laws govern everything above: the **capacity law** (how many items fit
in a bundle before recall degrades — closed form, measured, and used *predictively*
by the allocator) and **conservation** (a trace's fixed energy is partitioned, never
created — `trace_partition` reads the ledger: signal / crosstalk / damage).

### 2. The families (`holographic/<family>/`)

Eleven directories: `agents_and_reasoning`, `caching_and_storage`, `io_and_interop`,
`materials_and_texture`, `mesh_and_geometry`, `misc`, `rendering`,
`sampling_and_signal`, `scene_and_pipeline`, `semantic_router`,
`simulation_and_physics` — plus `unified` (the facade parts). Every module ends with
a `_selftest()` that asserts **numeric contracts** (exactness to a stated tolerance,
planted truths with dedicated RNGs, and *kept negatives* — refuted approaches pinned
so they cannot be reinvented). Comments explain **why** (the trade-off, the paper,
the negative avoided), never what the syntax does.

### 3. UnifiedMind (`holographic/unified/`, facade in `holographic/misc/holographic_unified.py`)

One class, thousands of thin **delegating** methods. A faculty never reimplements —
it names, documents, and forwards. This is load-bearing: the HTTP service
introspects every public method into `GET /tools`, so anything wired here is
instantly callable by an agent via `POST /invoke`. The governing rule of the whole
project: **a capability that `find_capability` can't surface and `/invoke` can't
call does not exist.**

### 4. The capability catalog & semantic engine (`holographic/caching_and_storage/holographic_catalog*.py`)

Every capability registers with a plain-language description, a **runnable**
example, and generous aliases — including *outsider vocabulary* ("prevent
hallucination", "reproducible AI"), because discoverability includes the words of
people who don't speak ours. `find_capability` is semantic search over this
catalog; `suggest` adds confidence and the concrete call; `route` decides
act-vs-choose. `capdoc.py` exports the whole catalog to `CAPABILITIES.md` (human)
and `capabilities.json` (machine) — CI blocks merges if they drift from the code.

**This is why "use leCore to learn leCore" is real advice**: the catalog is the
documentation, kept honest by lint (`skill_lint` runs every example; `catalog_gaps`
and `reachability_audit` hold at 0/0/0 — nothing import-only, nothing undocumented).

### 5. The retrieval organ, with honesty built in (`caching_and_storage`)

Exact tiled search (bit-identical to dense, memory bounded by the tile, streams off
disk), an approximate forest, and nested-descent "screens" — and **none of the
approximate routes may serve without a measured label**: `recall_budget=` measures
recall on *the caller's own vectors* (Wilson CI) and demotes to exact, number
attached, when the data defeats the structure. **Calibrated abstention** sits on
top: a noise null (hash-seeded, saturation-guarded) lets the index refuse queries
whose best match is noise-level, at a promised false-alarm rate. Measured on real
Wikipedia vectors; nothing in the 2026 ANN literature ships either feature.

### 6. The ISA and the VM (`docs/ISA*.md`, `holographic/agents_and_reasoning/holographic_machine.py`)

A small instruction set (LOAD/BIND/BUNDLE/PERMUTE/CALL/REPEAT/STORE/RECALL/…) whose
**programs are themselves hypervectors**: instructions are role-bound atoms bundled
at position codes, and the VM decodes them holographically (cleanup-gated) at run
time. Ops carry EXACT/TOL conformance tags. The decode has a real capacity wall
(program length × SNR vs dim) — found by fuzzing, now instrumented rather than
folklore.

### 7. The installed pipeline (`holographic_projector.py`, `holographic_compileinstall.py`, `holographic_nativemodel.py`)

The bridge from code to weights, every arrow measured:

- **certify** — `probe_project` measures any callable with basis vectors and either
  certifies it (permutation → D ints; circulant → D floats; dense → D²; detection
  most-specific-first) or **refuses** (nonlinear). The refusal is the core/shell
  boundary, discovered by measurement, and the refusal object is unusable by design.
- **compile** — `compile_installed` turns a symbolic VM program into a chain of
  certified matvecs + register slots; REPEAT of a linear body collapses to **one
  operator power** (spectral for circulants — exact).
- **verify** — `verify_conformance` runs three referees (VM, installed chain,
  symbolic interpreter) and checks **instrument validity first**: a decode-limited
  VM run is flagged, not miscounted.
- **price** — every payload carries residual, conditioning (spectrum max/min + a
  per-step chain amplification bound; deep non-unitary chains warn and name the
  fix), and fp16/bf16 quantization error.
- **hash** — sha256 per payload, so installation can verify what landed.
- **bake** — `NativeHoloModel`: a model whose file is the **rule** (~250 bytes of
  `{dim, seed, program}`) and whose `load()` re-bakes bit-identical weights;
  `unitary=True` for deep programs (depth-256 error 7.8e82 → 6e-15, measured);
  `to_dense()` exports any layer as the literal host matrix.

Live models built against this pipeline: <https://huggingface.co/staccs>.

### 8. The honesty layer (everywhere)

Not a module — a set of enforced habits: measured baselines with variance;
**kept negatives** logged in docstrings and `docs/NOTES_concepts.md` (the lab
notebook — wins *and* refutations); perfect scores treated as instrument
hypotheses; reference implementations shipped verbatim beside every fast path (the
oracle, the admissibility evidence, and the merge arbiter); and the audit battery
(`tools/reachability_audit.py`, `tools/catalog_gaps.py`, `tools/skill_lint.py`)
holding at 0/0/0 in CI.

### 9. Delivery & reproducibility

`PYTHONHASHSEED=0` for canonical runs; `hashlib` never `hash()`; seeded
`default_rng` everywhere; one stated tie rule (`topk_det`) delegated to by every
ranking path; 6,300+ tests; docs regenerated from the live catalog with a CI drift
gate; and every release zip clean-extract-verified under a **randomized** hash seed
— determinism proven where it is hardest, not where it is convenient.

## Reading order (for the thorough)

1. This file — the map.
2. [`CAPABILITIES.md`](../CAPABILITIES.md) — the menu (or skip it and ask
   `find_capability` directly; that is what it is for).
3. `tools/showcase.py` — the six flagship claims as live assertions (~2 s).
4. [`docs/SHOWCASE.md`](SHOWCASE.md) — what summaries miss, and what leCore is not.
5. `docs/ISA.md` + `docs/CONVENTIONS.md` — the contracts.
6. `docs/NOTES_concepts.md` — the honest lab notebook, newest entries last.
7. `REFERENCE.md` — the full generated module reference, when you need depth.

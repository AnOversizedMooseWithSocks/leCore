# leCore

**The core of [leOS](https://github.com/AnOversizedMooseWithSocks/leOS), on its own. A from-scratch engine that represents *everything* — memory, meaning, 3-D geometry, physics, images — as points in one very large space, and computes with a handful of simple, reversible operations. Pure NumPy. No neural networks, no pretrained models, no GPU required.**

[![tests](https://github.com/AnOversizedMooseWithSocks/leCore/actions/workflows/ci.yml/badge.svg)](https://github.com/AnOversizedMooseWithSocks/leCore/actions/workflows/ci.yml)

> **New here (human or AI)?** Don't summarize this project from the file tree — the
> capabilities are documented in the engine's own catalog and stay invisible to a skim.
> 1. **Read [CAPABILITIES.md](CAPABILITIES.md)** — the auto-generated menu of everything
>    the engine does, with runnable examples and search aliases.
> 2. **Run the proof**: `PYTHONHASHSEED=0 python3 tools/showcase.py` — six flagship
>    claims as live assertions, ~2 s, laptop CPU, no GPU/torch.
> 3. **Ask the engine itself** — it is often *easier to use leCore to learn leCore*,
>    because it carries a semantic search engine over its own capabilities:
>    ```python
>    import lecore
>    mind = lecore.UnifiedMind()
>    mind.find_capability("prevent hallucination")   # ranked capability homes
>    mind.suggest("compress a float series")          # homes + confidence + the call
>    ```
> 4. **The map**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (the whole, then the
>    parts) and [docs/SHOWCASE.md](docs/SHOWCASE.md) (what summaries miss, and what
>    this project is *not*). AI assistants: see also [llms.txt](llms.txt).

---

## What is this?

Most software uses a *different* tool for every job: a database for memory, a mesh for 3-D, a solver for physics, a neural net for perception. They don't share much, so gluing them together is most of the work.

leCore takes the opposite bet. It represents everything the same way — as a **hypervector**, which is just a very long list of numbers (a point in a high-dimensional space) — and combines those points with a tiny algebra:

- **bind** — glue two things together into one (a role and its value, a shape and where it sits).
- **bundle** — overlay many things into one (a set, an average, a memory of several examples).
- **cleanup** — take a noisy result and snap it back to the nearest thing you actually know.

That's most of it. The surprising part — and the reason the project exists — is that this *same* small toolkit turns out to describe a memory, a 3-D shape, a force field, and a step of a simulation. The project's motto is **"as above, so below"**: the same patterns keep showing up at every scale. So instead of a pile of unrelated subsystems, you get **one substrate** where memory, geometry, and physics are all the same kind of thing and can talk to each other for free.

It's written to be **read**: plain NumPy, commented, deterministic. If you can read Python and picture a list of numbers, you can follow how it works.

If the module count still looks like unrelated sprawl, read **[docs/THE_THESIS.md](docs/THE_THESIS.md)** — the junk test, with receipts: five modules that look unrelated collapsing into one operator, the measured numbers behind each claim, and a ten-minute tour. Or ask the system itself: `mind.find_capability("is this junk")` answers, by design.

## Why does it exist?

Two reasons.

1. **The name says it: leCore is the core of leOS.** It's the vector engine at the heart of **[leOS — the Latent Embedding Operating System](https://github.com/AnOversizedMooseWithSocks/leOS)** — also mine — **extracted and improved** so it could be developed, tested, and hardened on its own, and then **folded back into leOS** later. leOS is the larger vision (a local, CPU-only "subconscious substrate" that sits beneath a language model); leCore is the from-scratch engine underneath it. Several of the best ideas here (coarse-to-fine resolution, fractal structure, fountain codes) came straight from leOS.

2. **To show one substrate can carry the whole load — honestly.** The engine is built on a short list of non-negotiables: pure NumPy/stdlib, **no learned weights and no black boxes**, **deterministic** (same input, same output, every run), and **honestly measured** — every claim has a baseline, and kept negatives (the things that *didn't* work) stay in the record on purpose. It's an old-school engineering project: readable, dependency-light, and skeptical of its own results.

## What can it do?

leCore has grown large, so here's the **generalized** view — the families of capability, not a feature list:

- **Remember and recall by content.** A robust, self-organizing associative memory: give it something *like* what you stored and it finds the match — and tells you *how confident* it is, or abstains when it doesn't know. (No exact keys required; it degrades gracefully instead of failing hard.)

- **Represent and reason over structured knowledge.** Build concepts up from parts and take them back apart — records with named fields, relationships, analogies ("the capital of France is to Paris as the capital of Japan is to…"), all with the bind/bundle/cleanup algebra.

- **Build and render 3-D scenes.** Shapes as math (signed-distance fields), point clouds (Gaussian splats), meshes, and a from-scratch path tracer with materials and lighting — all in NumPy. Preview fast, refine to photoreal.

- **Simulate the physical world.** Fluids, cloth and soft bodies, particles, and shaped force fields (attractors, wind, "sticky" volumes) — the same substrate the geometry lives in.

- **Run programs that are themselves data.** A small virtual machine where a "program" is a hypervector the engine can *inspect, price, and execute* — so the system can reason about its own actions, not just perform them.

- **Stay honest and reversible.** Compression that decompresses exactly, error-correction that survives lost data, and a measurement layer that reports uncertainty rather than bluffing.

You don't have to use all of it. Each capability works on its own; the point is that they *share one space*, so they compose.

**You don't have to memorize any of it, either.** The engine keeps a searchable catalog of what it can do, so a plain-English description of your problem finds the right tool — `mind.find_capability("search a big pile of vectors")`, or `mind.suggest("edit an image")` for ranked options with the call to make, or `mind.route("render a scene")` which either hands you the call (when it's sure) or a short list of choices (when it isn't). The full plain-language menu — every capability, what it does, and the one line that gets you started — lives in **[`CAPABILITIES.md`](CAPABILITIES.md)**, and it's generated from that same catalog by CI so it never goes stale.

## What does its output look like?

Everything below was rendered or measured by the engine itself — no external renderer, no
plotting library in core. The full set (dozens more, each with the test that produced it)
is in **[`GALLERY.md`](GALLERY.md)**.

| | |
|---|---|
| ![A groomed furry critter, strand-level fur](gallery/render_fur.png) | ![Crystal grains and ore inclusions](gallery/render_crystal.png) |
| *Strand-level groomed fur, path-traced* | *Procedural crystal grains with ore inclusions* |
| ![Metal bars glowing by temperature](gallery/render_hot_metal.png) | ![Thin-film iridescence on a bubble and an oil-slick sphere](gallery/render_iridescence.png) |
| *Hot metal: emission from physical temperature* | *Thin-film iridescence: soap bubble, oil slick* |

And the measurement culture, in pictures:

| | |
|---|---|
| ![Graceful degradation under damage](gallery/graceful_degradation.png) | ![Capacity curve](gallery/capacity_curve.png) |
| *Recall vs. storage destroyed — the fragment principle as a graph* | *The measured capacity law the scale advisor consults* |

## How do you use it?

It's a plain Python library. The core needs **only NumPy** — nothing else is ever required. Everything beyond that (the web UI, image I/O, tests and plots, and the `numba`/CuPy/SymPy/Zig accelerators) is **opt-in**, and you pull in exactly what you want with pip "extras."

**The quick way — install from PyPI.** The package is published as **`leos-core`** (the core of the larger **leOS** project); the import name stays `lecore`:

```bash
pip install leos-core              # installs the engine (+ NumPy)
python -c "import lecore; print(lecore.UnifiedMind)"
```

**Or work from a clone** (what you want if you're hacking on the engine or running the tour/UI):

```bash
git clone https://github.com/AnOversizedMooseWithSocks/leCore
cd leCore
pip install numpy                  # the ONLY hard requirement -- the core runs on NumPy alone
python app.py                      # then open the browser UI and click "Run full system tour"
```

**Adding the optional bits.** Each optional group has a name. If you installed from PyPI, name the extras on the package (`pip install "leos-core[name]"`); if you're working from a clone, use the dot, which means "this folder" (`pip install .[name]`). Combine names with commas. (The quotes around `leos-core[ui]` just keep some shells from trying to interpret the brackets — the dot form rarely needs them.)

```bash
# from PyPI (no clone):                     # from a clone (the dot = this folder):
pip install "leos-core[ui]"                 # pip install .[ui]        the browser UI + image I/O   (Flask, Pillow)
pip install "leos-core[jit]"                # pip install .[jit]       numba-compiled fast paths     (numba)
pip install "leos-core[symbolic]"           # pip install .[symbolic]  design-time gradients         (SymPy)
pip install "leos-core[zig]"                # pip install .[zig]       native batch kernels, 2-5x     (ziglang -- whole
                                            #                          toolchain in one wheel, bit-identical in safe mode)
pip install "leos-core[images]"             # pip install .[images]    jpg/webp/... image I/O         (Pillow, no Flask)
pip install "leos-core[dev]"                # pip install .[dev]       run the tests and make plots  (pytest, matplotlib)
pip install "leos-core[all]"                # pip install .[all]       everything portable, one shot
pip install "leos-core[ui,jit]"             # pip install .[ui,jit]    ...or combine whichever you want

# GPU support is separate, because CuPy is tied to your CUDA version:
pip install "leos-core[gpu]"                # pip install .[gpu]       tries plain `cupy`; if that fails, install
                                            #                          the matching wheel by hand, e.g. cupy-cuda12x
```

If you'd rather not install leCore as a package at all, you can of course just `pip install` those same libraries directly (`pip install flask pillow`, etc.) and run from the clone — the extras are simply a convenient, named way to do it. The engine notices what's present and lights up the matching fast paths on its own; nothing optional is ever required.

The **fastest way to get it** is the tour: it runs the whole engine end to end in about half a minute, and finishes by having the unified mind assemble its own concepts from a bare pile of examples.

In code, the heart of it is one class, **`UnifiedMind`**, which carries every general capability on one shared space:

```python
from holographic.misc.holographic_unified import UnifiedMind

mind = UnifiedMind(dim=4096, seed=0)     # one high-dimensional space; seed -> fully deterministic

# teach it a few things by description, then recall by CONTENT (not by an exact key):
mind.learn("a small red round fruit",  "apple")
mind.learn("a long soft yellow fruit", "banana")
(label, description), score = mind.recall("something red you can eat")   # -> ('apple', ...) with a score

# the raw algebra everything is built on (bind / unbind / cleanup):
from holographic.agents_and_reasoning.holographic_ai import Vocabulary, bind, unbind

vocab = Vocabulary(dim=4096, seed=0)
role, filler = vocab.get("role"), vocab.get("filler")   # two named random vectors
bound   = bind(role, filler)             # glue two vectors into one
noisy   = unbind(bound, role)            # recover the filler -- approximately (bind is reversible, but lossy)
name, _ = vocab.cleanup(noisy)           # snap the noisy result to the nearest known vector -> "filler"
```

*(Every line above actually runs — the README's Python examples are checked in CI. The modules keep their `holographic_` prefix from the project's origins.)* From there, the same `mind` object is where you reach the geometry, rendering, simulation, and program-execution capabilities.

**Describe a scene and shape it in words.** You can hand the engine a description and it builds a scene of *named* objects you can then adjust by talking to it — and when it doesn't understand a word, it says so and suggests alternatives instead of failing silently:

```python
scene = mind.build_scene("a big red metal sphere and a small blue glass box on a sunny day")
scene.adjust("make the sphere bigger")        # reference a named object, change it in plain words
scene.adjust("change the box to metal")
scene.adjust("make the pyramid golden")       # no pyramid -> changes nothing, and scene.feedback explains why
image  = scene.render()                       # best-effort 3-D render (default camera, the scene's sun/sky)
frames = scene.simulate(steps=40)             # a simple gravity drop of the objects
scene.options()                               # what you *can* say: the objects, and the words for colour/material/size
```

It's a controlled vocabulary, on purpose — deterministic and honest about its limits, not a black-box language model.

**Run it as a standalone HTTP service.** leCore ships a small, dependency-free server (`holographic_service.py`, standard-library `http.server`) so you can drive it over HTTP — a SQL/GraphQL data store, long-running jobs you can pause and resume, and an agent-facing skills API (`GET /skills`, `POST /skills/suggest`, `POST /skills/route`) that lets a program discover and call capabilities the same way the `mind` methods above do. See **[`SERVICE.md`](SERVICE.md)** for the endpoints and `curl` examples.

## What "holographic" actually buys you

Every item is spread across *every* number, so destroying storage destroys a little of everything rather
than all of a few things. Measured — 16 key/value pairs in 1024 floats, random slots zeroed, against a
plain contiguous store holding the same 16 items in the *same* 1024 floats (64 each):

| slots destroyed | holographic recall@1 | contiguous items intact |
|---|---|---|
| 10% | **100.0%** | 0.5% |
| 40% | **100.0%** | 0.0% |
| 80% | **97.0%** | 0.0% |
| 90% | 75.6% | 0.0% |

Read the right-hand column first: a contiguous store is *already gone* at 10% damage, because every item
needs all 64 of its own floats and the chance all 64 survive is about one in a thousand. The holographic
store still answers every query correctly at **40% loss**, and most of them at 80%.

![The same experiment as a curve](gallery/graceful_degradation.png)

Reproduce it with `python3 -m pytest tests/test_degradation_table.py -q`. Harness: `bind`/`unbind` with
cleanup by nearest value in the codebook, 40 trials per row, seeded. These are **means over 40 trials** —
a single draw of 16 items reports in steps of 6.25% and will look tidier and better than the truth.

## Installing it into language models: Unicron and Ouroboros

The engine doesn't just sit next to language models — it installs *into* them. **Unicron**
writes leCore capabilities directly into pretrained LLM weights (tested against a real
production model: Qwen3.5-0.8B — BF16, hybrid attention, vision tower), with streaming
weight loading so small machines can do it. The installs are constructed and deterministic,
not trained: a *recipe* records how to reconstruct one at about 3× under the weights it
produces (store the rule, not the bytes), composing two separately-trained models is literal
vector addition (`compose == add`, exact), and **removal is exact** — install a donor's
behavior into a host, measure the transfer, then ablate it and the host returns to its
original behavior exactly.

**Ouroboros** is the closed memory loop that follows: the linear-attention state matrix
inside such models *is* a holographic memory (a theorem about its algebra, not a metaphor),
so leCore can **read from and write into a running model's memory with no forward pass at
all** — measured externally at read cosine 0.935 / write 0.951 on the production algebra,
with measured deletion and a *predictive* capacity law (0.932 predicted vs 0.905 measured;
1.000 exact at reference scale). Durable memory lives in a per-tenant store that survives
restarts; consolidation is transcript-only *by API shape*, because the obvious alternative
was measured and refuted (rehearsing a state's own reads back into it degrades it,
0.767 → 0.730 — kept as a negative). The memory has three edit verbs with different prices:
**write** adds content and pays crosstalk; **pose** reshapes stored values as an isometry
(recall exactly preserved, inverse exact to 1e-17, zero capacity cost); **key-pose**
relocates addresses without touching content, exactly.

## Determinism is a proof system

Because the engine is bit-deterministic (`PYTHONHASHSEED=0`, `hashlib` everywhere, seeded
RNG, stable sorts), a replay is a *proof*: run the same call twice and the input/output
hashes match bit-for-bit. Every call through the MCP/HTTP surface returns a receipt —

```json
"lecore.receipt": {
  "input_sha256":  "46b57dd6…",
  "output_sha256": "50dba5a3…",
  "deterministic": true
}
```

— so verifying a result degenerates to comparing a sha256, at zero marginal cost, covering
*every* operation (memory writes, retrieval, exploration), not just inference. Compare:
zero-knowledge proof systems for LLM inference cost hundreds of seconds to days per
generation; here the whole engine being deterministic makes the proof free.

## The six levers: how walls fall here

When blocked, the codebase walks six levers **in order** before declaring anything
impossible — each carries measured kills:

1. **Bake once, sample O(1).** A compiled gather rule answers in one dot product: 182,010×
   at N=2048 *when reused* (and honestly 0.03× when not — its own docstring says so).
2. **Partition into a commutative monoid.** Work that distributes merges and un-merges for
   free — the retrieval index merges two corpora and ablates one *without rebuild*, laws
   pinned by tests.
3. **Determinism instead of storage.** Regenerate from seeds; recipes instead of bytes;
   receipts instead of trust.
4. **Lift to where the problem is linear** — and not only along the dimension axis: when
   that's dead, lift along precision, roles, or phase (see the benchmark below).
5. **Tile under an orchestrator.** The wave scheduler colours 2,000 contending transactions
   over 300 shared keys into 24 conflict-free waves — 83× lock-free, deterministic.
6. **A measured limit is a composability boundary.** Every capacity law's number is not a
   wall but a **tile size**: groups of K under a coordinator, which has a *different* shape
   with a *different* measured limit — split and coordinate again when you hit it.
   `hierarchical_pack` ships this (more items than the flat law allows, by cleaning up
   *between* levels); recursion plus determinism means every level can be compressed,
   cached, or replaced by a generator. **Limits become the quantization grain of the
   hierarchy.** This is also why the codebase nests: the VM installs inside model weights,
   the swarm coordinator installs inside weights, ladders serve ladders — "as above, so
   below" is the sixth lever's operating manual.

## The machine inside the machine

`mind.machine_map()` returns a spec sheet of NumPy-native units occupying the same *roles*
as GPU silicon, each with a measured cost model: numpy itself as the SIMD lanes (`@` is
BLAS at 116 GFLOP/s), batched operator power as the tensor core (4.3×, exact to 1.9e-12),
`sphere_trace` as the RT core, superposition packing as SIMT width (with its 1/√K capacity
law stated), a counter-based per-thread RNG, kernel fusion (a 2,000-step loop matched to
6.7e-16 at 80×), sleeping islands as occupancy, and a wave scheduler. The memory side is a
five-tier ladder — L0 compiled operators (121 ns) through L4 compressed-RAM low-rank fields
(171× fewer bytes; and it *refuses* white noise, which would cost 1.54× more) — and
`memory_mountain()` measures the host's real cache tiers so cost models predict from
measured floors, not datasheets. A stored-program holographic VM (`HoloMachine`) runs
vector programs on top, with a content-addressed compile cache and a decoded-instruction
cache — and Unicron installs VM units into model weights.

## Benchmarked against FAISS, on adversarial data

An independent-researcher-style dispute harness: real ABTT-whitened embeddings plus
on-manifold near-duplicate cliques, exact float64 ground truth, a hardness gate that
refuses friendly random vectors. Results (recall@10, median ms/query):

| rung | leCore auto | FAISS Flat (exact) | FAISS IVF | FAISS HNSW |
|---|---|---|---|---|
| 100k×768 | **1.000 @ 9.7 ms** | 1.000 @ 27.1 ms | 0.875 @ 3.3 ms | 0.853 @ 0.51 ms |
| 1M×128 | **1.000 @ 34.8 ms** | — | 0.940 @ 5.5 ms | 0.600 @ 0.17 ms |

The mechanism is the doctrine executing: structure levers first (certified sphere tracing —
24× where data has cluster mass, and honestly 100%-touched on whitened dust, kept both
ways), then lever 4 on the *precision* axis — quantization error is spectrum-immune, so a
row-scaled int8 scan under a provable worst-case error bound yields a candidate set that
*provably* contains the exact top-k including ties, rescored in f64. An adaptive ladder
measures every route on *your* data at *your* k and serves the fastest whose certified
bound meets budget. Exact answers at quantized-scan speed; the only 1.000 in the 1M table.
Dispute the numbers by re-running, not by re-describing (`tools/benchmarks_faiss.py`).

Is this table exercising the holographic core? **No — by measurement, and that is the point**:
[the honest answer](docs/ANSWER_benchmark_and_vsa.md) documents where VSA auditioned for the
hot path and lost (centroid 0.797 vs HRR bundle 0.789, a kept negative), and where it is
measured as decisively load-bearing (noisy-key recall 0.889 where an exact dict scores 0.000).

## Why it doesn't hallucinate about stored facts

The defense is structural, not a prompt: (1) facts are stored verbatim and hash-addressed,
and memory fidelity is *measured*; (2) every readout is snapped to a real stored item or
refused — destroy half a trace (raw cosine 0.144) and cleanup still identifies 24/24;
(3) abstention is calibrated — `Index.nearest(query, abstain=α)` returns *empty* when the
best hit's false-alarm probability, judged against a null built from the corpus's own
vocabulary, exceeds α; (4) retrieval answers carry receipts, so a claimed source is
verified, not trusted; (5) drift is caught mechanically — generated docs and code are
hash-diffed against their deterministic source of truth in CI, and session state restores
exactly. Known facts cannot drift; unknown facts cannot be invented; every answer is
auditable.

## What's actually new (with prior art named)

The math is old and the repo cites it with dates — that's the method, not a weakness;
novelty in engineering is the *arrangement*, and the test of an arrangement is measurement.
After searching the literature through mid-2026, each claim below names its closest prior
art so it stays falsifiable: **external zero-pass memory read/write with exact inverse**
(vs ROME/MEMIT weight editing, fast-weight programmers and TTT/Titans — all in-pass, or new
architectures); **behavior transfer with exact free rejection** (vs approximate task
arithmetic, and SISA's exact-by-retraining); **memory edits as solvable constrained group
actions** (vs fixed permutations and fractional power encoding); **null-gated exploration**
(vs novelty search without a significance gate); **computation billable by hash** (vs zkML
at seconds-to-days per proof); **deterministic worst-case certified quantized retrieval as
a measured system** (vs RaBitQ's probabilistic bounds); and **the salamander theorems** —
Lashley's and Pietsch's lesion arguments run as pinned computational theorems for the first
time. The full argument, with the measured receipts behind every claim, is in
**[`docs/THE_THESIS.md`](docs/THE_THESIS.md)**.

## The rules it plays by

If you contribute or build on it, these are the load-bearing rules — they're what keep it trustworthy:

- **NumPy / stdlib only** in the core. No PyTorch, no scikit-learn, no pretrained models. (`numba`, CuPy, and SymPy are opt-in extras, never required.)
- **No learned weights, no black boxes.** Everything is an explicit, inspectable computation.
- **Deterministic.** Fixed seeds, stable sorts, reproducible bit-for-bit.
- **Additive and backward-compatible.** New capability is added; existing behavior isn't broken.
- **Honestly measured.** Every improvement beats a real baseline; failures are kept in the record, not hidden.
- **Readable.** Commented code that explains *why*, minimal machinery.

## Where it comes from, and how it's funded

leCore is the extracted, hardened core of **[leOS](https://github.com/AnOversizedMooseWithSocks/leOS)** — my larger project — and is meant to be folded back into it once it's proven out here. It also powers **[leStudio](https://github.com/AnOversizedMooseWithSocks/leOS-Studio/tree/main/2d/lestudio)**, a 2D image editor built on the same engine (the image toolkit in the sections above is what's under its hood). You can read about the whole vision at **[discoverleos.com](https://discoverleos.com/)** (a dedicated **leCore** section is being added).

Like leOS, leCore is **free and open source**, and the work that keeps it free is paid for by liquidity-pool fees from the **$leOS token on Solana**. The funding model is deliberately simple: fees come from *trading volume, not price*, so the most direct way to support the project is to trade the token — buying, selling, or rotating between pairs all generate fees that fund development, regardless of which way the price moves. Full details, the three-pool setup, and the verifiable contract are on the [leOS site](https://discoverleos.com/) (token contract `5xgsnby6P9zqGK71J7H4yJLxzqPvNbC7rDZxNzjHmj7e`, verifiable on [Solscan](https://solscan.io/token/5xgsnby6P9zqGK71J7H4yJLxzqPvNbC7rDZxNzjHmj7e)).

## Learning more

- **[`FEATURE_GUIDE.md`](FEATURE_GUIDE.md)** — a **hands-on how-to** for the most recently added features: composable
  materials/textures, the describe-a-scene authoring flow (naming, texturing, external files), external-asset
  relocation and the queryable file map, the message-bus + optional-agent harness, and the opt-in language layer. Every
  example is a short, commented, runnable snippet. The best place to start if you want to *use* the new capabilities.
- **[`RENDERING_GUIDE.md`](RENDERING_GUIDE.md)** — a **practical guide to 3D, rendering & simulation**: the
  surface-vs-volume mental model, three ways to make a cloud (describe it, one call, or by hand), building scenes
  from words, cameras and lights, volumetric smoke/fog/fire, and the adaptive path tracer. Every snippet is
  verified-runnable, with real cost numbers and a troubleshooting table. Start here if you want to *make a picture*.
- **[`docs/SDF_COOKBOOK.md`](docs/SDF_COOKBOOK.md)** — a **practical guide to signed distance fields**: the
  constructors-are-functions / combinators-are-methods model, the runnable build→evaluate→mesh pipeline, the
  data-driven NodeGraph route (with its `'a'`/`'b'`/`'out'` socket names), and the collected gotchas
  (`box` takes three scalars, `rotate` takes an axis vector). Every snippet is verified-runnable.
- **[`DEVELOPMENT_STRATEGY.md`](DEVELOPMENT_STRATEGY.md)** — the **standard process for changing leCore**: audit with
  `find_capability` first, wire every capability to a mind faculty (so it is `/invoke`-able), register it in the
  catalog so it is discoverable, and run the reachability/gap audits — the discipline that keeps the codebase from
  growing gaps or isolating code in tests. Read this before making code changes.
- **[`CAPABILITIES.md`](CAPABILITIES.md)** — the **front-door menu**: a plain-language, grouped list of what leCore can
  do and the one call that starts each job. The friendliest place to begin if you're deciding whether the engine
  already does the thing you need. Generated from the live capability catalog by `capdoc.py` and kept in sync by CI.
- **[`capabilities.json`](capabilities.json)** — the **machine-readable sibling** of `CAPABILITIES.md`, for tools and apps that ingest
  the capability list as data rather than parsing the prose. Generated in the same `capdoc.py` run from the same
  catalog (so the two never disagree), and CI-gated so a consumer never reads a stale copy. It is a versioned
  contract: a top-level `schema_version` plus a flat `capabilities` array, each entry `{name, does, example,
  aliases, native, theme}`. Consumers should check `schema_version` and refuse a major version they don't know.
- **[`REFERENCE.md`](REFERENCE.md)** — the **code reference**: a file/module map and a plain-language breakdown
  of every module (its "why this exists" note plus its public functions and classes). Start here to find your
  way around. It's generated from the code by `docgen.py` and kept in sync automatically by CI, so it never
  drifts from what's actually there.
- **[`API_QUICKREF.md`](API_QUICKREF.md)** — the **app-builder's quick reference**: one scannable line per public
  class/function for the modules you actually touch when building on leCore (scene, mesh, camera, render, ship).
- **[`SERVICE.md`](SERVICE.md)** — the **standalone HTTP service**: every endpoint (data store, jobs, and the
  agent-facing skills API) with `curl` examples, for driving leCore as an app rather than a library.
- **[`GALLERY.md`](GALLERY.md)** — a **visual showcase**: renders, procedural patterns, memory/reconstruction demos, and performance charts, straight from the engine's tests (the visual companion to the code reference).
- **[`docs/GLSL_GUIDE.md`](docs/GLSL_GUIDE.md)** — **building with leCoreGLSL**: an interactive field demo, the ten verified
  shader kernels, the three shader shapes, what the GPU is measurably good and bad at here, and
  what you can build (games and simulation: yes; retrieval: no; VR: untested and labelled so).
- **[`writing_vsa_programs.md`](writing_vsa_programs.md)** — the **VSA program writing guide**: how to express
  your own logic as a holographic program on `HoloMachine`, the small stored-program machine, without baking it
  into the core. Read this when you want to run custom logic over the vector algebra.
- **[`THEORY.md`](docs/THEORY.md)** — the load-bearing claims and what backs each one (the honest middle ground, not a paper).
- **[`NOTES_concepts.md`](docs/NOTES_concepts.md)** — the running design log: what was tried, what worked, what didn't.
- **[`ISA.md`](docs/ISA.md)** — the small instruction set the engine's programs are built from.
- The module docstrings — every `holographic_*.py` file opens with a plain-language "why this exists" (and
  those are exactly what `REFERENCE.md` gathers up for you).

**How the docs stay honest.** Three of the files above are *generated* from the code and *gated* in CI, so they can't
quietly fall out of date: `REFERENCE.md` (from module docstrings), `API_QUICKREF.md` and `CAPABILITIES.md` (from the
catalog). `SERVICE.md` is mostly hand-written, but its endpoint table is checked against the service's real route
registry, so a new or renamed endpoint can't ship undocumented. On top of the test suite, CI also runs two small checks
that keep the engine usable rather than just correct — a **discoverability gate** (`tools/catalog_gaps.py`: every
capability a user would ask for has a findable home) and an **invocation gate** (`tools/skill_lint.py`: every faculty
carries a docstring an agent can act on, and every "how to call it" example actually resolves). If you add a capability
and forget to document or register it, CI tells you which one.

## Status

Active research engine, and a large one — hundreds of modules and thousands of tests, all green in CI. It's real and it runs, but it's a research project under steady development, not a finished product. Expect sharp edges, expect it to keep growing, and expect every surprising result to come with the measurement that earned it.

## License

See [`LICENSE`](LICENSE).

---

*Built from scratch, in the open, one vector at a time.*

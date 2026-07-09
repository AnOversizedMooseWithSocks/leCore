# leCore

**The core of [leOS](https://github.com/AnOversizedMooseWithSocks/leOS), on its own. A from-scratch engine that represents *everything* — memory, meaning, 3-D geometry, physics, images — as points in one very large space, and computes with a handful of simple, reversible operations. Pure NumPy. No neural networks, no pretrained models, no GPU required.**

[![tests](https://github.com/AnOversizedMooseWithSocks/leCore/actions/workflows/ci.yml/badge.svg)](https://github.com/AnOversizedMooseWithSocks/leCore/actions/workflows/ci.yml)

---

## What is this?

Most software uses a *different* tool for every job: a database for memory, a mesh for 3-D, a solver for physics, a neural net for perception. They don't share much, so gluing them together is most of the work.

leCore takes the opposite bet. It represents everything the same way — as a **hypervector**, which is just a very long list of numbers (a point in a high-dimensional space) — and combines those points with a tiny algebra:

- **bind** — glue two things together into one (a role and its value, a shape and where it sits).
- **bundle** — overlay many things into one (a set, an average, a memory of several examples).
- **cleanup** — take a noisy result and snap it back to the nearest thing you actually know.

That's most of it. The surprising part — and the reason the project exists — is that this *same* small toolkit turns out to describe a memory, a 3-D shape, a force field, and a step of a simulation. The project's motto is **"as above, so below"**: the same patterns keep showing up at every scale. So instead of a pile of unrelated subsystems, you get **one substrate** where memory, geometry, and physics are all the same kind of thing and can talk to each other for free.

It's written to be **read**: plain NumPy, commented, deterministic. If you can read Python and picture a list of numbers, you can follow how it works.

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

## How do you use it?

It's a plain Python library. The core needs **only NumPy** — nothing else is ever required. Everything beyond that (the web UI, image I/O, tests and plots, and the `numba`/CuPy/SymPy accelerators) is **opt-in**, and you pull in exactly what you want with pip "extras."

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

## The rules it plays by

If you contribute or build on it, these are the load-bearing rules — they're what keep it trustworthy:

- **NumPy / stdlib only** in the core. No PyTorch, no scikit-learn, no pretrained models. (`numba`, CuPy, and SymPy are opt-in extras, never required.)
- **No learned weights, no black boxes.** Everything is an explicit, inspectable computation.
- **Deterministic.** Fixed seeds, stable sorts, reproducible bit-for-bit.
- **Additive and backward-compatible.** New capability is added; existing behavior isn't broken.
- **Honestly measured.** Every improvement beats a real baseline; failures are kept in the record, not hidden.
- **Readable.** Commented code that explains *why*, minimal machinery.

## Where it comes from, and how it's funded

leCore is the extracted, hardened core of **[leOS](https://github.com/AnOversizedMooseWithSocks/leOS)** — my larger project — and is meant to be folded back into it once it's proven out here. You can read about the whole vision at **[discoverleos.com](https://discoverleos.com/)** (a dedicated **leCore** section is being added).

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
- **[`DEVELOPMENT_STRATEGY.md`](DEVELOPMENT_STRATEGY.md)** — the **standard process for changing leCore**: audit with
  `find_capability` first, wire every capability to a mind faculty (so it is `/invoke`-able), register it in the
  catalog so it is discoverable, and run the reachability/gap audits — the discipline that keeps the codebase from
  growing gaps or isolating code in tests. Read this before making code changes.
- **[`CAPABILITIES.md`](CAPABILITIES.md)** — the **front-door menu**: a plain-language, grouped list of what leCore can
  do and the one call that starts each job. The friendliest place to begin if you're deciding whether the engine
  already does the thing you need. Generated from the live capability catalog by `capdoc.py` and kept in sync by CI.
- **`capabilities.json`** — the **machine-readable sibling** of `CAPABILITIES.md`, for tools and apps that ingest
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
- **[`writing_vsa_programs.md`](writing_vsa_programs.md)** — the **VSA program writing guide**: how to express
  your own logic as a holographic program on `HoloMachine`, the small stored-program machine, without baking it
  into the core. Read this when you want to run custom logic over the vector algebra.
- **`THEORY.md`** — the load-bearing claims and what backs each one (the honest middle ground, not a paper).
- **`NOTES_concepts.md`** — the running design log: what was tried, what worked, what didn't.
- **`ISA.md`** — the small instruction set the engine's programs are built from.
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

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

## How do you use it?

It's a plain Python library. The core needs **only NumPy** — nothing else is ever required. Everything beyond that (the web UI, image I/O, tests and plots, and the `numba`/CuPy/SymPy accelerators) is **opt-in**, and you pull in exactly what you want with pip "extras."

```bash
git clone https://github.com/AnOversizedMooseWithSocks/leCore
cd leCore
pip install numpy                  # the ONLY hard requirement -- the core runs on NumPy alone
python app.py                      # then open the browser UI and click "Run full system tour"
```

**Adding the optional bits.** Each optional group has a name; install it from the cloned folder with `pip install .[name]` (note the dot — it means "this folder"). Combine names with commas.

```bash
pip install .[ui]         # the browser UI + image I/O          (Flask, Pillow)
pip install .[jit]        # numba-compiled fast paths           (numba)
pip install .[symbolic]   # design-time symbolic gradients      (SymPy)
pip install .[dev]        # run the tests and make plots        (pytest, matplotlib)
pip install .[all]        # everything portable, in one shot
pip install .[ui,jit]     # ...or combine whichever you want

# GPU support is separate, because CuPy is tied to your CUDA version:
pip install .[gpu]        # tries plain `cupy`; if that fails, install the matching wheel by
                          # hand instead, e.g.  pip install cupy-cuda12x
```

If you'd rather not install leCore as a package at all, you can of course just `pip install` those same libraries directly (`pip install flask pillow`, etc.) and run from the clone — the extras are simply a convenient, named way to do it. The engine notices what's present and lights up the matching fast paths on its own; nothing optional is ever required.

The **fastest way to get it** is the tour: it runs the whole engine end to end in about half a minute, and finishes by having the unified mind assemble its own concepts from a bare pile of examples.

In code, the heart of it is one class, **`UnifiedMind`**, which carries every general capability on one shared space. The flavor (illustrative):

```python
from holographic_unified import UnifiedMind

mind = UnifiedMind(dim=4096, seed=0)     # one high-dimensional space; seed -> fully deterministic

# teach it a few things, then recall by content (not by exact key):
mind.remember("apple",  "a red fruit")
mind.remember("banana", "a yellow fruit")
hit = mind.recall("something red you can eat")   # -> the closest stored item, WITH a confidence

# the raw algebra everything is built on:
a, b = mind.encode("role"), mind.encode("filler")
bound   = mind.bind(a, b)                 # glue two vectors into one
back    = mind.unbind(bound, a)           # recover the filler (bind is reversible)
guess   = mind.cleanup(back)              # snap the noisy result to the nearest known vector
```

*(Method names above are illustrative of the shape — see the docs for exact signatures. The modules keep their `holographic_` prefix from the project's origins.)* From there, the same `mind` object is where you reach the geometry, rendering, simulation, and program-execution capabilities.

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

- **[`REFERENCE.md`](REFERENCE.md)** — the **code reference**: a file/module map and a plain-language breakdown
  of every module (its "why this exists" note plus its public functions and classes). Start here to find your
  way around. It's generated from the code by `docgen.py` and kept in sync automatically by CI, so it never
  drifts from what's actually there.
- **[`GALLERY.md`](GALLERY.md)** — a **visual showcase**: renders, procedural patterns, memory/reconstruction demos, and performance charts, straight from the engine's tests (the visual companion to the code reference).
- **[`writing_vsa_programs.md`](writing_vsa_programs.md)** — the **VSA program writing guide**: how to express
  your own logic as a holographic program on `HoloMachine`, the small stored-program machine, without baking it
  into the core. Read this when you want to run custom logic over the vector algebra.
- **`THEORY.md`** — the load-bearing claims and what backs each one (the honest middle ground, not a paper).
- **`NOTES_concepts.md`** — the running design log: what was tried, what worked, what didn't.
- **`ISA.md`** — the small instruction set the engine's programs are built from.
- **`holographic_metrics.py`** — the central JSON/Markdown evidence rollup; run `make metrics` for fast
  evidence, `make metrics-c-tests` for NumPy/C mode parity, or `make metrics-path-d` to refresh the core
  Path D caches first.
- The module docstrings — every `holographic_*.py` file opens with a plain-language "why this exists" (and
  those are exactly what `REFERENCE.md` gathers up for you).

## Status

Active research engine, and a large one — hundreds of modules and thousands of tests, all green in CI. It's real and it runs, but it's a research project under steady development, not a finished product. Expect sharp edges, expect it to keep growing, and expect every surprising result to come with the measurement that earned it.

## License

See [`LICENSE`](LICENSE).

---

*Built from scratch, in the open, one vector at a time.*

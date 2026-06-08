# Holographic VSA

A from-scratch, **numpy-only** holographic / vector-symbolic engine - and a small
web UI on top of it.

One idea runs through the whole thing: represent *everything* - a number, a word,
a record, a fact, a creature's situation, an image - as a point in a very high
dimensional space, and combine those points with a few reversible algebraic
operations (`bind`, `bundle`, `permute`). Out of that one substrate you get
associative memory, learned word meaning, structured records, symbolic reasoning,
a little reinforcement-learning creature, and a damage-tolerant image archive.
No neural-network framework, no pretrained models, no GPU - just numpy. (The UI,
image I/O, tests, and plots use Flask / Pillow / pytest / matplotlib.)

If you only read one thing: run it (below), click **Run full system tour**, and
watch all eight subsystems work in ~20 seconds.

---

## Quick start

### Windows
Double-click **`run.bat`** - it installs dependencies on first run, starts the
server, and opens `http://127.0.0.1:5000`.

### Any platform
    pip install -r requirements.txt
    python app.py            # then open http://127.0.0.1:5000

If `pip install` fails with **"Access is denied" / WinError 5** (a system-wide
Python you can't write to), install into a local virtual environment instead -
which is exactly what `run.bat` now does automatically:

    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
    .venv\Scripts\python app.py

(or add `--user` to the `pip install` to install just for your account.)

The web UI has these panels: **System tour** (runs every subsystem once),
**Compression & speed** (single- and many-file size, encode/decode timing, and corruption resilience vs JPEG/PNG),
**Batch operations** (superposition capacity, 1-bit memory, cleanup throughput), **Creature** (trains the forager
with poison present, then animates its random-vs-trained forage step-by-step and shows the seek-vs-avoid reflex),
**Test suite** (runs the full pytest suite), **Query & recall** (the interactive image demo - degrade an
image, optionally destroy part of the plate, watch it get recalled), **Recall by description**
(cross-modal recall - describe an image in words and get the matching one back from the tag address space),
and **Set packer** (delta-code a set of related images against one reference - shared structure stored once).

### From the command line
    python tour.py                    # guided tour of all subsystems (~20s)
    python holographic_creature.py    # any module runs its own demo
    python holographic_encoders.py    # numbers / text / records demos
    pytest -q                         # the whole test suite (60 tests)

---

## What it can do (with results from `tour.py`)

Everything below lives on the *same* vector substrate and the same memory.

**Numbers.** Encode a real value as a unit vector so nearby numbers get nearby
vectors, then read the number back out - even from a noisy vector.
`decode(encode(7.2)) = 7.19`; a noisy vector of `4.0` still decodes to `3.92`.

**Text.** Learn word vectors from raw co-occurrence (no gradients, no labels).
After a few passes over a tiny corpus: `cat ~ dog = 0.77` but `cat ~ car = 0.36`;
nearest word to `truck` is `car (0.88)`. The geometry carries meaning.

**Mixed records.** Pack a number, a category, and a free-text note into one
2048-d vector and read each field back individually: `price -> 140.7` (stored
142.5), `trend -> up`. Record-to-record similarity reflects all fields at once.

**Key -> value memory.** Store many `key -> value` pairs superposed in a *single*
vector and recall them by content, with cleanup snapping noisy results to the
nearest known symbol. A handful of `country -> capital` facts in one 1024-d
vector, all recalled correctly; it degrades gracefully when overloaded.

**Reasoning.** A resonator network factors a composite vector
(`subject (x) relation (x) object`, three things bound together) back into its
parts knowing only the vocabularies - recovering 6/6 facts where a single unbind
cannot. Also included: split-conformal error bars, an epistemic "how well do I
know this" map, and a semantic compass.

**A learning creature.** A grid-world forager with a purely holographic mind
(perceive -> decide -> learn by remembering experiences; no neural net). From a
random baseline of `-0.27` reward / `0.2` food, after 120 episodes it reaches
`+7.30` reward / `7.8` food - it taught itself to find food. The full demo
in the UI also shows poison avoidance -- it learns to seek food AND route around
hazards (over 12 worlds it reaches food without touching poison in ~11). The
module demo (`python holographic_creature.py`) adds a working-memory
scene with limited vision.

**A damage-tolerant image archive.** Store a gallery of images superposed into a
few plates; recall the clean original from a noisy / blurred / occluded query,
even after destroying a large fraction of the plate. The bundled demo recalls
6/6 images from each corruption, and still 6/6 (reconstructing at ~52 dB) with
40% of every plate destroyed. Two further capabilities sit on the same store:
**cross-modal recall** (`recall_by_tags`) addresses an image by word/number tags
bundled into a hypervector, so "radial pink" alone returns the right image (6/6
on the demo gallery, no picture needed); and **quantized plates** (`quantize(4)`)
shrink the store ~8x (844 KB -> 107 KB) with content recall still 6/6 and
recovery degrading gracefully (79 -> 30 dB). Both work together - the tag
addresses live outside the plates, so cross-modal recall is unaffected by
quantization.

**A delta set-packer for related images** (`holographic_pack.py`). Single-file
codecs compress each image alone, so a family of images that shares structure
pays for that shared content in every file. The packer stores a set as one
reference plus per-image deltas (residual mod 256, zlib'd), bit-exact and in 8-bit
integers throughout. On a six-logo suite that shares a background and ring it packs
to ~39% of per-file PNG and beats gzip-ing the whole set; honestly, on images that
are already compressible on their own (gradients, photos) per-file PNG/JPEG win,
and the built-in benchmark shows both so the choice is clear. A lossy
Walsh-Hadamard tier was prototyped and dropped because it never beat JPEG.

**More engine pieces** (each with a runnable demo): residue (exact integer)
arithmetic on vectors, signed-distance regions of the sphere, a predictive filter
that stays quiet on the expected, a unified scalar "field" abstraction,
two-timescale diffusion, Kuramoto-style emergent grouping, a tool orchestrator
with circuit-breakers and reusable skeletons, online unsupervised concept
formation, and a hypervector reaction-diffusion cellular automaton.

---

## How it works

**The core.** Atoms are random high-dimensional vectors, nearly orthogonal by the
blessing of dimensionality. `bind` (a reversible element-wise combine) ties two
together into something dissimilar to both; `unbind` recovers a partner.
`bundle` (normalised sum) overlays things into a set you can still query. A
`Vocabulary` mints clean atoms and `cleanup` snaps a noisy vector back to the
nearest known one. That is the whole toolkit; every subsystem is a different way
of arranging those operations.

**The image archive** adds three image-specific steps:

1. *DCT, keep the big coefficients.* Each colour channel goes through a pure-numpy
   orthonormal 2-D DCT; only the largest `K` coefficients are kept (their
   positions stored as a small bitmask, counted honestly in the size).
2. *Spread with structured keys.* The kept coefficients are scattered and run
   through a **Walsh-Hadamard transform** with a fixed random sign pattern, so
   each one is smeared across *all* `D` plate values - no plate value is special.
   The key operator is matrix-free and an exact isometry, so an undamaged plate
   decodes exactly with one adjoint pass.
3. *Recover.* Undamaged - one multiply. Damaged - a mask marks survivors and a
   small conjugate-gradient solve recovers from what is left, graceful until the
   survivors drop below the stored coefficient count.

Multiple images share one plate via *disjoint* key-slot pools (keeping the
combined keys orthonormal), and content-addressable recall keeps a tiny thumbnail
fingerprint *outside* the plate so recognition survives even when the plate is
wrecked.

---

## Benchmarks vs. existing tech (image archive)

Measured by `bench_vs_jpeg.py` on a 240x240 colour image. Reproduce: `python bench_vs_jpeg.py`.

**Plain compression - JPEG/PNG win, and that's fine.** The hologram is not a
compressor: PNG 1.5 KB, JPEG q85 5.3 KB (29.8 dB), hologram 4-bit 42.2 KB
(27.9 dB). Use a real codec if you want small files.

**Corruption resilience - the hologram wins enormously.** Corrupt the same
*fraction* of a JPEG file vs. plate cells (mean PSNR over 8 trials; 0 dB = no
longer decodes):

| corrupted | JPEG q85 | Hologram |
|-----------|----------|----------|
| 0.1%      | 9.2 dB   | 27.9 dB  |
| 1%        | 4.0 dB   | 27.9 dB  |
| 10%       | 0.0 dB   | 27.8 dB  |
| 40%       | 0.0 dB   | 27.0 dB  |

A JPEG dies at a tenth of a percent (its headers, DC terms, and entropy-coder
state are single points of failure); the hologram is essentially untouched at 40%,
because corruption is just uniform noise with no privileged bytes to destroy. See
`figures/bench_corruption.png`.

**Also measured:** the Walsh-Hadamard keys use ~3,200x less memory than a dense
random-projection matrix and run ~57x faster, with identical fidelity; conjugate-
gradient decoding gives ~8x the usable capacity of a matched filter; 10 images
multiplex into one plate with no crosstalk.

**Batch retrieval (1-bit vs float).** Finding the right stored item from a noisy
query, over a 10,000-item database (`bench_batch.py`): 1-bit sign hypervectors
with Hamming similarity match float32 cosine on accuracy (**100% vs 100% recall@1**
on a 20%-corrupted query) while using **32x less memory** (10 MB vs 328 MB). In
pure numpy the float matmul is faster (BLAS is hard to beat without a dedicated
popcount kernel); the 1-bit win is the footprint, which fits in cache at scale.

---

## Project layout (flat on purpose - everything imports cleanly)

The engine (pure numpy):

    holographic_ai.py         bind/bundle/cleanup, key->value memory, learner, reflex, drift
    holographic_encoders.py   numbers (scalar/fractional-power), text, mixed records
    holographic_reasoning.py  resonator, conformal intervals, epistemic map, compass
    holographic_creature.py   grid-world + a holographic RL mind (the forager)
    holographic_extras.py     residue arithmetic, SDF regions, predictive filter
    holographic_field.py      scalar field abstraction (one field, many roles)
    holographic_diffusion.py  two-timescale double diffusion
    holographic_sync.py       Kuramoto-style emergent grouping
    holographic_orchestrator.py  tool planner with circuit-breakers + skeletons
    holographic_emergence.py  online unsupervised concept formation
    holographic_automaton.py  hypervector reaction-diffusion CA (demoscene)
    holographic_image.py      WHT keys, DCT codec, quantised plate, damage decode
    holographic_archive.py    content-addressable multi-image memory
    holographic_pack.py       lossless delta set-packer for related images

The app and tour:

    app.py        Flask UI (system tour + test runner + image recall)
    tour.py       command-line tour of every subsystem
    run.bat       Windows launcher

Tests (60 total):

    test_holographic.py           core engine
    test_holographic_image.py     image store / WHT / quantisation
    test_holographic_archive.py   archive recall + damage
    test_holographic_pack.py      delta set-packer round-trip + size

Research / provenance (run from this folder, e.g. `python exp_wht.py`):

    exp_*.py, bench_vs_jpeg.py, bench_fig.py, benchmark_holographic.py,
    stress_holographic.py, make_test_image.py
    figures/   rendered results

---

## Honest limitations

- **Not a competitive image compressor** - use JPEG/WebP/AVIF for small files.
- **Hard capacity / damage cliff** - image recovery is graceful only until the
  surviving plate cells drop below the stored coefficient count (the demo archive
  sits at load 0.37, so its cliff is ~63%; the UI caps damage at 70%).
- **Small-scale by design** - this is a clear, readable, from-scratch engine for
  learning and experimenting, not a tuned production system. The text corpus is
  tiny, the creature world is small, and the vectors are modest. It is built to be
  understood and extended.

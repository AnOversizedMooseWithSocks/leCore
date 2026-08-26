# holostuff Benchmark Program — One Real-Data Test Per Seat

*The panel was asked a sharper question than usual: not "what could we build" but "prove holostuff
is **effective and more useful in your field than what you already use** — on a real dataset, against
your field's real existing method." Each of the nineteen seats gets exactly one test. As always,
every test is attributed to a SEAT and that field's REAL published method, never a fabricated
personal opinion. The faculties named below were grounded against the LIVE `UnifiedMind` (110+ public
methods), not memory — every entry point cited (`recognize`, `recall_calibrated`, `stream_recognize`,
`denoise`, `splat_archive`, `solve_maze`, `assemble`, `learn_dynamics`, `decompose_structure`,
`high_capacity_memory`, `consolidate`, `generate_vector`, `compress_signal`, …) exists today. And in
the engine's own spirit, every test ships with the NEGATIVE designed in — the regime where holostuff
will **not** win and is not claimed to. A benchmark that only reports wins would betray the project.*

---

## 0. How the data gets in — the GitHub-only fetch constraint

The app's network egress is restricted to `github.com`, `raw.githubusercontent.com`, and
`codeload.github.com` (plus `files.pythonhosted.org` for `pip`). That single fact drove most of the
dataset choices below, so it is worth stating the mechanics before the tests.

**The new layer.** Today the engine has three data on-ramps: **books** via NLTK's Gutenberg corpus
(`unified_app.load_gutenberg`), **photographs** via a local image folder (`holographic_photos.load_photo_folder`,
which already learned the honest lesson that *JPEG beats our DCT coder on bytes but our plate is
robust where JPEG is brittle*), and **market data** via a committed Binance `.npz`/`.json` in `data/`.
None of them fetch from GitHub. This program adds a fourth on-ramp — a small, deterministic GitHub
fetcher — and leaves the other three untouched.

**The git-lfs trap (read this once).** `raw.githubusercontent.com` serves a file's *git object*. If a
repo stores a binary via **git-lfs**, the object at that path is a tiny *pointer*, not the data — the
real bytes live on `media.githubusercontent.com`, which is **not** on the allowlist. A GitHub *archive
zip* (`codeload`) has the same problem: it ships LFS pointers, not LFS content. So the reliable
datasets are the ones a repo **commits directly** as ordinary git objects. Below, each dataset is
tagged for sourcing confidence:

- **`[direct]`** — committed as a normal git object, fetches cleanly over the allowlist.
- **`[care]`** — may be LFS-backed or oversized; has a committed-fallback or a `pip` reader noted.
- **`[gen]`** — generated locally from a committed solver; nothing is downloaded (one seat only).

**The fetcher (sketch).** Minimal frameworks, stdlib only, pinned to commit SHAs so the data can
never shift under a test (the determinism rule applies to inputs, not just code):

```python
# holographic_datasets.py
# Deterministic GitHub-only dataset fetcher. WHY stdlib-only: the engine's rule is
# minimal dependencies, and urllib + hashlib + zipfile cover everything we need.
import hashlib, os, urllib.request, zipfile

DATA_DIR = "data/external"            # everything lands here; never the working dir that snapshots wipe
MAX_BYTES = 100 * 1024 * 1024         # hard 100 MB guard per file -- refuse anything larger

# Each entry pins a COMMIT SHA (not a branch) so the bytes are frozen. sha256 catches
# silent corruption or an LFS-pointer-instead-of-data surprise (the hash simply won't match).
SOURCES = {
    "htru2": dict(
        url="https://raw.githubusercontent.com/<owner>/<repo>/<sha>/HTRU_2.csv",
        sha256="<fill-at-build-time>", license="UCI/CC-BY", seat="Siemion"),
    # ... one entry per dataset below ...
}

def _download(url, dest):
    # WHY a manual loop with a size guard: a mis-pinned URL or an LFS miss can hand us a
    # giant or a 130-byte pointer; we want to fail LOUDLY in both directions, not half-write.
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r:
        total = 0
        with open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES:
                    f.close(); os.remove(dest)
                    raise RuntimeError(f"{url} exceeds {MAX_BYTES} bytes -- refusing (size guard)")
                f.write(chunk)
    return total

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()

def ensure(name):
    """Idempotent: fetch once, verify checksum, reuse forever. Re-running is free and safe."""
    spec = SOURCES[name]
    dest = os.path.join(DATA_DIR, name + os.path.splitext(spec["url"])[1])
    if os.path.exists(dest) and _sha256(dest) == spec["sha256"]:
        return dest                                   # already have the exact bytes -- done
    got = _download(spec["url"], dest)
    have = _sha256(dest)
    if spec["sha256"] and have != spec["sha256"]:     # empty sha during bring-up; pin before CI
        raise RuntimeError(f"{name}: sha256 mismatch (got {have}); "
                           f"likely an LFS pointer or a moved file -- re-pin the SHA")
    return dest
```

Whole-repo grabs (VGLC, BCCD, FSDD) use `codeload.github.com/<owner>/<repo>/zip/<sha>` through the
same guard, then `zipfile.extractall` into `data/external/`. Each `ensure()` call is logged with the
seat, the size, and the license, so the suite's provenance is auditable. **The one exception is the
Stam seat**, marked `[gen]`: real fluid fields under 100 MB live on Zenodo/Drive (off the allowlist),
so that field is *generated* from a committed FFT-on-a-torus solver — which is literally Stam's own
method — rather than downloaded.

---

## Roster at a glance

| Seat | Member | Dataset (GitHub, ~size) | Existing method it must beat / match | holostuff faculty | The honest win it's testing |
|---|---|---|---|---|---|
| Radio astronomy | **Tarter** | Voyager-1 BL/GBT filterbank `[care]`, ~tens MB | Matched filter + fixed-window detect | `stream_recognize`, `recall_calibrated` | Decide in fewer samples; abstain on noise |
| SETI | **Siemion** | HTRU2 pulsar candidates `[direct]`, ~2 MB | Lyon 2016 GH-VFDT classifier | `recognize_batch` (FDR) + `SelfOrganizingMind` | Label-free, FDR-controlled, deterministic triage |
| Protein folding | **Baker** | A few small PDB structures `[care]`, <5 MB | Rosetta fragment assembly / greedy search | `assemble` (Tero flow) + `consolidate` | Lower placement energy at equal search cost |
| Neuroscience | **Olshausen** | MNIST glyphs `[care]`, ~11 MB | Dense resonator (Kymn 2024) | `decompose_structure` (SBC) | More factors × alphabet before the cliff |
| Particle physics | **Cranmer** | MAGIC gamma telescope `[direct]`, ~1.5 MB | Boosted-tree signal/background | `recognize` + `calibration_report` + FDR | Calibrated false-alarm control + look-elsewhere |
| 3D / raytracing | **Pharr** | GloVe-50d subset `[care]` / Fashion-MNIST `[direct]` | Exact NN + kd-tree (recall@k vs QPS) | `HoloForest.recall` + `recall_calibrated` | Sublinear recall that ABSTAINS honestly |
| Fungus / mold | **Adamatzky** | City coords (Tero Tokyo / TSPLIB) `[direct]`, <1 MB | Tero 2010 Physarum; MST / Steiner | `solve_maze` → multi-terminal flow | MST-cost network with higher fault tolerance |
| Demoscene | **Quílez** | Brodatz / periodic textures `[care]`, ~few MB | JPEG/PNG at matched bytes | `generate_vector`, `fractal_dimension` | Higher PSNR/byte on self-similar content |
| Video games | **Togelius** | VGLC Mario levels `[direct]`, ~1 MB | Small DQN; published PCG playability | `decide` + `decide_confidence` | Reactive parity + calibrated, explainable action |
| Soft/hard body | **Macklin** | Short mocap clip `[care]`, <10 MB | H.264 motion comp; Kalman / least-squares | iterate-a-projection + bind-as-shift | VSA-native motion comp; bit-exact determinism |
| Smoke/water | **Stam** | 2-D Navier–Stokes vorticity `[gen]` | Persistence, mean; FNO reference | `learn_dynamics` (Koopman/DMD) | Beat persistence AND mean; exact round-trip |
| Medical imaging | **Ozcan** | BCCD blood-cell microscopy `[direct]`, ~7 MB | Single denoise + TV inpainting | `splat_archive` + `denoise(pnp)` (RED) | Reconstruct degraded plate better than one-shot |
| Audio | **Puckette** | FSDD spoken digits `[direct]`, ~few MB | MFCC + cosine kNN | `high_capacity_memory` (FHRR) + audio modality | Content-addressable sound recall; audio dynamics |
| File compression | **Duda** | Fashion-MNIST / GloVe store `[direct]` | int8 + gzip/zstd (zstd = his ANS) | `save(quant='rd')` (KLT→water-fill→rANS) | Fewer bits/vector at fixed cosine fidelity |
| VSA / HRR | **Plate** | UMLS knowledge graph `[direct]`, <2 MB | HolE (Nickel 2016 = HRR for KGs); TransE | native bind/cleanup + capacity diagnostic | HolE-ballpark MRR + a live capacity readout |
| Quantum / tensor nets | **Stoudenmire** | Frey faces `[care]`, ~1 MB | MPS/DMRG truncation; PCA | `consolidate` (SVD/KLT) + MPS-bind (stretch) | Truncation == tensor-net truncation on real data |
| Denoising | **Milanfar** | Set12 / BSD68 `[care]`, ~few MB | NLM (Buades 2005), BM3D (Dabov 2007) | `denoise` (manifold/NLM/PnP/codebook) | Match classical NLM; win in the high-noise regime |
| Gaussian splatting | **Drettakis** | Kodak image set `[care]`, ~few MB | JPEG at matched bytes; WHT-plate archive | `splat_field` / `splat_archive` | Queryable, refinable, erasure-robust scene code |
| Generative art | **Eno** | JSB Chorales `[direct]`, <1 MB | First-order Markov; bare-codebook sampler | `generate_vector` over a composed subspace | Valid-AND-novel structures, not stored atoms |

---

## The tests, seat by seat

### Jill Tarter — radio astronomy
- **Dataset.** The Breakthrough Listen Voyager-1 observation taken with the Green Bank Telescope X-band
  receiver (30 Dec 2015), distributed in the **blimpy** repo as the single-coarse-channel filterbank
  `tests/Voyager_data/Voyager1.single_coarse.fine_res.fil`. Real instrument data containing a genuine
  narrowband technosignature — Voyager's carrier plus telemetry sidebands — in real receiver noise and
  bandpass. Reader: `blimpy` (`pip` from PyPI, allowlisted). **`[care]`:** if the `.fil` is LFS-backed,
  fall back to `pip install blimpy` (which carries its own sample) or place the file manually — real BL
  data is genuinely hard to get under a github-only constraint, and this is the one seat where that may
  bite.
- **Existing method it's measured against.** Matched filtering against a known waveform, with a fixed
  integration window, plus the field's shuffled-null discipline (re-run the detector on scrambled data
  and demand it collapse).
- **holostuff faculty.** `stream_recognize` (Wald SPRT over the spectrogram channel as a cue stream),
  `recognize` (calibrated false-alarm p per cue), `recall_calibrated`.
- **The bar.** Detect Voyager's carrier+sidebands versus a frequency-shuffled null in **fewer expected
  samples** than a fixed-window detector at matched (α, β), and **abstain** (no detection) on the
  off-source / pure-noise channels. The win is "decide as fast as the evidence allows, and prove it
  isn't a pipeline artifact" — exactly her field's move, now callable per channel.
- **The negative we keep.** holostuff is *not* a Doppler-drift search engine. On the stationary carrier
  it should win; on an *uncompensated drifting* narrowband cue it needs a matched-filter bank over
  candidate drift rates (the field's own fallback, A1) before SPRT applies. Stated, not hidden.

### Andrew Siemion — SETI
- **Dataset.** **HTRU2** — 17,898 pulsar candidates from the High Time Resolution Universe survey, 8
  engineered features each, 1,639 real pulsars vs 16,259 RFI/noise. Tiny CSV, mirrored directly on
  GitHub. `[direct]`.
- **Existing method.** Lyon et al. 2016 (*Fifty Years of Pulsar Candidate Selection*) — the GH-VFDT
  tree classifier and the standard imbalanced-learning pipeline the field uses.
- **holostuff faculty.** `SelfOrganizingMind` (unsupervised class discovery, no labels) +
  `recognize_batch` (Benjamini–Hochberg FDR across the 17,898 candidates) + the per-candidate
  calibrated p as a novelty score.
- **The bar.** At a **fixed false-discovery rate** across the whole candidate set, recover real pulsars
  with recall competitive with the published classifier — while being **label-free, deterministic, and
  air-gappable**, and with the calibrated p tracking the noise floor. The "flag anything that isn't
  noise" ambition is the calibrated-novelty p-value made operational.
- **The negative we keep.** Eight hand-engineered features already linearly separate much of HTRU2; a
  tuned supervised tree will match or beat raw *accuracy*. holostuff's edge is FDR-controlled,
  interpretable, unsupervised triage at scan scale — not topping the accuracy table.

### David Baker — protein folding
- **Dataset.** A handful of small protein structures from a directly-committed source (e.g. Biopython's
  `Tests/PDB` fixtures, or a few RCSB entries mirrored on GitHub). `[care]` — pin the exact files;
  keep them small.
- **Existing method.** Rosetta fragment assembly — build a global backbone from a library of local
  motifs under an energy — benchmarked against greedy / exhaustive placement at equal cost.
- **holostuff faculty.** `assemble` (fragment assembly via deterministic Tero flow) + `consolidate`
  (superpose two assembled structures, read their overlap).
- **The bar.** Reconstruct a backbone from its own local-fragment library at **lower placement energy
  than greedy at equal search cost**, and demonstrate structure-compare via `consolidate`. This tests
  the **search machinery** on the canonical hard-landscape field.
- **The negative we keep.** `assemble`'s energy is a placement-mismatch **stand-in**, not a force field
  (this is the genuinely-unbuilt A3: pluggable, cleanup-based energy). So the test measures search, not
  folding accuracy — it will not approach Rosetta/AlphaFold structure quality and does not claim to.

### Bruno Olshausen — neuroscience
- **Dataset.** **MNIST** glyphs, used as the parts of *composed visual scenes* (object × position ×
  scale × colour bound into one product), the setup of the resonator-network scene-parsing literature.
  `[care]` (MNIST is sometimes LFS/external; Fashion-MNIST `[direct]` is the committed fallback).
- **Existing method.** Kymn, Olshausen et al. 2024, *Compositional Factorization of Visual Scenes* —
  the dense resonator network, with its known low operational-capacity ceiling and limit-cycle stalls.
- **holostuff faculty.** `decompose_structure` / the SBC resonator (B2 sparse block codes, block-local
  convolution), already wired.
- **The bar.** Factor the composed product into its factors at **strictly more factors × alphabet at
  fixed dimension** than the dense resonator, with the convergence signal correctly flagging the
  non-converged cases. A working model of cortical scene parsing on a deterministic substrate.
- **The negative we keep.** SBC is a *parallel mode* beside the dense kernel, not a replacement; and
  the exact-reconstruction `validated` certificate is brittle on approximate inputs — the calibrated-soft
  confidence (A2) is the real remaining work.

### Kyle Cranmer — particle physics
- **Dataset.** **MAGIC Gamma Telescope** — 19,020 Monte-Carlo events, 10 features, gamma signal vs
  hadron background. Tiny CSV, directly on GitHub. Simulated detector data is exactly his
  simulation-based-inference home. `[direct]`.
- **Existing method.** A boosted-tree / random-forest signal-background classifier — and, crucially,
  whether its score is *calibrated*.
- **holostuff faculty.** `recognize` (calibrated false-alarm p) + `calibration_report` (coverage on
  held-out noise) + `recognize_batch` (FDR = look-elsewhere control).
- **The bar.** Thresholding the recognition p at α **holds the false-alarm rate at α** on held-out
  hadron background (coverage), AND FDR controls false discoveries across many candidate events — and
  this holds **as the store grows** (the open growing-store coverage question, A6). Calibration a
  black-box classifier doesn't give for free.
- **The negative we keep.** A tuned BDT will win on raw AUC. holostuff's claim is a detector whose
  stated false-alarm rate you can *trust*, with look-elsewhere control — not peak discrimination.

### Matt Pharr — 3D modeling & raytracing
- **Dataset.** A **GloVe** word-vector subset (e.g. 50-d) as a real embedding store — or, because
  glove.6B is large/LFS, **Fashion-MNIST** flattened images as the committed-`[direct]` fallback store.
  This same store is shared with the Duda seat (one store, two faculties — the project's "one dataset,
  many lenses" thesis made literal).
- **Existing method.** Exact brute-force nearest-neighbour and a kd-tree/BVH, on the ANN-benchmark axes
  (recall@k vs distance-computations / queries-per-second).
- **holostuff faculty.** `HoloForest.recall` (sublinear random-projection forest) + `recall_calibrated`
  (now routed *through* the forest, A-tier fix) + cross-tree agreement.
- **The bar.** Match exact NN **recall@10 at a fraction of the distance computations**, AND correctly
  **abstain** on out-of-vocabulary / off-manifold queries where a kd-tree silently returns a wrong
  neighbour — "did the traversal find something, or am I guessing?" answered with a calibrated p *and*
  the structural agreement signal.
- **The negative we keep.** Mature ANN libraries (FAISS, HNSW) beat HoloForest on the raw
  recall-speed Pareto. The differentiator is the **calibrated abstention** plus the content-addressable
  archive, not topping the ANN leaderboard.

### Andrew Adamatzky — fungus & mold
- **Dataset.** Real geographic terminal coordinates — the Tero 2010 36-point Tokyo-area city layout, or
  a small directly-committed city-coordinate CSV / TSPLIB instance (e.g. `berlin52`). `[direct]`, <1 MB.
- **Existing method.** Tero et al. 2010 (*Rules for Biologically Inspired Adaptive Network Design*, the
  Physarum/Tokyo-rail result), compared to a minimum spanning tree and a Steiner-tree approximation.
- **holostuff faculty.** `solve_maze` (deterministic Tero flow) extended from single-source/single-sink
  to **multi-terminal network design** (A9), returning the network as a typed structure.
- **The bar.** Build a network connecting the terminals whose **total length is within a few % of the
  MST** while achieving **higher fault tolerance** — Tero's actual finding (comparable cost, superior
  resilience). Beat a naive MST on the cost/fault-tolerance trade-off, the problem his slime moulds are
  famous for.
- **The negative we keep.** The solver is currently single-source/single-sink — multi-terminal is
  genuinely unbuilt (A9); and a dedicated Steiner solver beats it on pure length. The claim is the
  cost-*resilience* frontier, not minimal length.

### Iñigo Quílez — demoscene
- **Dataset.** The **Brodatz** texture album (or a periodic-tiling set) — real images with strong
  self-similar/periodic structure, mirrored on GitHub. `[care]` (pin the exact mirror).
- **Existing method.** A standard codec (JPEG/PNG) at a matched byte budget — the demoscene "maximal
  richness from a tiny deterministic kernel" claim made measurable.
- **holostuff faculty.** `generate_vector`, `compose_nested` / `nested_scene_structure`, `fractal_dimension`.
- **The bar.** Regenerate a self-similar/periodic texture from a **small holographic seed + repetition
  rule at higher reconstruction PSNR per byte** than the codec for the self-similar class, with
  `fractal_dimension` matching the texture's measured box-counting dimension.
- **The negative we keep.** The project's own recurring lesson: **only self-similar/periodic content
  benefits.** On a non-self-similar natural photo the seed loses to JPEG, exactly as "compression is
  structure-dependent" predicts. Both regimes get reported.

### Julian Togelius — video games
- **Dataset.** **VGLC** (the Video Game Level Corpus) Super Mario tilemaps — real published game levels,
  tiny, directly on GitHub. `[direct]`.
- **Existing method.** A small DQN / black-box RL agent (which acts confidently out-of-distribution),
  and the field's published PCG playability metrics.
- **holostuff faculty.** `decide` / `reinforce` / `actions` (the deterministic, self-explaining creature
  brain) + `decide_confidence` (calibrated abstention/exploration, A-tier shipped).
- **The bar.** On a **bounded reactive** sub-task in a real level, match the DQN's reward while (a)
  explaining each decision in human sense-terms, (b) **abstaining or exploring on novel tiles** where
  the DQN acts confidently wrong, and learning online without catastrophic forgetting. NPCs you can
  ship and trust.
- **The negative we keep.** The creature brain is **reactive — it does not plan.** On tasks needing
  lookahead, the maze/flow solver plans, not the brain. So the bar is reactive-task parity + calibrated
  honesty + explainability, never beating a planning agent.

### Miles Macklin — soft/hard body dynamics
- **Dataset.** A short motion-capture clip (a small directly-committed BVH, or a CMU-mocap mirror) and a
  noisy sensor trajectory. `[care]` — pin the clip; possible LFS.
- **Existing method.** H.264-style motion compensation (acknowledged unbeatable) for the rigid-shift
  demo; Kalman / least-squares for the projection-denoise demo.
- **holostuff faculty.** The iterate-a-projection faculty (A4: resonator + denoise + dynamics unified as
  "project onto constraints") + bind-as-rigid-shift (motion-compensated residual) + the
  **determinism/tie-break audit** of the new calibrated/null code paths.
- **The bar.** (a) A rigid inter-frame motion is captured by a **single bind that zeroes the residual**
  (VSA-native motion compensation); (b) the projection faculty denoises a noisy trajectory competitively
  with a standard filter; (c) the calibrated/null paths are **bit-identical run-to-run** with no
  knife-edge p-value ties — his `bind_batch` lesson (a bit-exact change still flipped a trajectory)
  applied to the honesty layer.
- **The negative we keep.** H.264 wins on compression. The claim is the *primitive* (a shift is one
  bind) and the *determinism discipline*, not a video codec.

### Jos Stam — smoke/water simulation
- **Dataset.** A 2-D incompressible Navier–Stokes vorticity field on a torus, **generated** by a small
  committed pseudospectral solver — i.e. Stam's own *"A Simple Fluid Solver based on the FFT."* `[gen]`,
  nothing downloaded. The honest exception: real fluid fields under 100 MB live off the allowlist, and
  a deterministic spectral sim is the field-standard way to make fluid data anyway.
- **Existing method.** Persistence and mean baselines (the B4 bar), with a Fourier Neural Operator as
  the modern reference — itself FFT-based, the rhyme with holostuff's binding.
- **holostuff faculty.** `learn_dynamics` (a learned bind operator = Koopman/DMD in Fourier
  coordinates): prediction is one bind, the trajectory is content-addressable.
- **The bar.** On a field that **genuinely has dynamics**, multi-step prediction beats **both
  persistence AND mean** (closing the loop the market-data negative left open, A8), and the trajectory
  **round-trips exactly** (recover the field k steps ago via the inverse operator).
- **The negative we keep.** A trained FNO is more accurate. holostuff's win is a **training-free,
  deterministic, O(1)-advance, content-addressable** representation — and on the order book it was a tie
  (kept on record).

### Aydogan Ozcan — medical imaging
- **Dataset.** **BCCD** — 364 real light-microscope peripheral-blood-smear JPEGs (640×480), MIT-licensed,
  committed directly on GitHub. `[direct]`, ~7 MB.
- **Existing method.** A single-shot denoise plus total-variation inpainting — classical
  reconstruction-under-degradation.
- **holostuff faculty.** `splat_archive` / the WHT plate archive (store) + `denoise(method='pnp')` run
  as a Plug-and-Play / RED **loop**, not a one-shot (A5).
- **The bar.** Store the cell images, **erase/undersample a region** (simulate sensor degradation), and
  reconstruct via PnP/RED using holostuff's `denoise` as the prior — beating a single denoise and TV
  inpaint on PSNR/SSIM of the recovered region; validate the σ estimate.
- **The negative we keep.** Not competing with trained deep reconstruction nets; and manifold-projection
  denoise **over-smooths at low noise** (the kept −1.4 dB at σ=0.3 negative) — which is exactly why the
  noise-adaptive path exists.

### Miller Puckette — audio engineering
- **Dataset.** **FSDD** (Free Spoken Digit Dataset) — real recordings of spoken digits, committed
  directly on GitHub, a few MB. `[direct]`.
- **Existing method.** MFCC + cosine kNN — the standard content-based audio-retrieval baseline.
- **holostuff faculty.** `high_capacity_memory` (the FHRR phasor memory — his representation exactly) +
  a new spectral/audio modality (encode a magnitude/phase spectrum as an FHRR hypervector) +
  `learn_dynamics` on the audio frame sequence (A7, the dynamics proving ground B4 flagged).
- **The bar.** Encode digit spectra as phasor hypervectors, bundle, and **recall the right digit/speaker
  by content** at accuracy competitive with MFCC+kNN — a content-addressable sound memory — and have
  `learn_dynamics` on the frame sequence **beat persistence and mean** (the audio dynamics B4 promised
  but the market data couldn't show).
- **The negative we keep.** Deep audio embeddings win on accuracy. The claim is a content-addressable
  FHRR sound store and that audio, unlike returns, has learnable linear-in-Fourier dynamics.

### Jarek Duda — file compression
- **Dataset.** The **same embedding store** the Pharr seat uses (Fashion-MNIST `[direct]`, or the GloVe
  subset) — one store, indexed sublinearly by Pharr and compressed here.
- **Existing method.** int8 quantization plus gzip/zstd on float32 — and zstd *is* Duda's ANS, the
  pointed nod.
- **holostuff faculty.** `save(quant='rd')` — the rate-distortion code already in the kernel:
  `consolidate` (KLT/SVD) → per-component water-filling bit allocation → a bit-exact rANS coder (B5).
- **The bar.** **Fewer bits per vector at a fixed cosine fidelity** than int8 *and* than zstd-on-float32,
  on the real store, with the recovered cosines pinned to the originals within a stated tolerance.
- **The negative we keep.** The recurring lesson: **high-entropy/full-rank vectors don't compress** —
  the KLT win needs real low-rank structure; on random vectors there's no gain. And binary quantization
  corrupts the similarity geometry (the documented negative). Both reported.

### Tony Plate — VSA & holographic computing
- **Dataset.** **UMLS** — 135 biomedical concepts, 49 relations, ~6.5k triples, committed directly in a
  GitHub KGE repo. Tiny. Kinship/Nations are alternatives (but carry high inverse-relation fractions,
  so UMLS is the cleaner pick). `[direct]`.
- **Existing method.** **HolE** (Nickel, Rosasco, Poggio 2016, *Holographic Embeddings of Knowledge
  Graphs*) — which is literally circular-correlation HRR applied to KGs, Plate's own foundation in
  another costume — plus TransE; scored by MRR / Hits@k.
- **holostuff faculty.** Native bind/cleanup link prediction (encode subject ⊛ relation, bundle, recall
  the tail) + the capacity / SNR diagnostic (A6).
- **The bar.** Link-prediction MRR/Hits@k on UMLS in the **ballpark of HolE's published numbers** using
  holostuff's native HRR binding, AND the capacity diagnostic correctly **predicts where binding
  saturates** as triples accumulate (operating point vs the cliff) and whether the false-alarm rate
  holds at α as the store grows.
- **The negative we keep.** Trained KGE models (ComplEx, RotatE) beat HolE/HRR on these benchmarks.
  holostuff's value is that **HolE *is* its own algebra** — a custodial validation that the foundation
  works on a real graph, plus a *live* capacity readout, not SOTA link prediction.

### Miles Stoudenmire — quantum / tensor networks
- **Dataset.** **Frey faces** — a low-rank face-video dataset, ~1 MB, on GitHub mirrors. `[care]`
  (MNIST per his own MPS paper is the alternative). Genuinely low-rank, so the truncation comparison is
  clean.
- **Existing method.** MPS/DMRG low-rank truncation (Stoudenmire & Schwab 2016) and standard PCA.
- **holostuff faculty.** `consolidate` (SVD/KLT — a tensor-network truncation by another name) + the
  optional tensor-train bind mode (A14, flagged speculative).
- **The bar.** `consolidate` recovers the effective low-rank subspace, **matching an MPS truncation's
  reconstruction-vs-rank curve** on the real low-rank data; and (stretch) a tensor-train bind mode
  raises binding capacity vs HRR convolution at fixed dimension.
- **The negative we keep.** The MPS-bind is **the heaviest, most speculative item** — research, not a
  quick wire; and `consolidate` *is* PCA, so the claim is the tensor-network **equivalence on real
  data**, not a new result.

### Peyman Milanfar — denoising & inverse problems
- **Dataset.** **Set12 / BSD68** — the classical image-denoising test sets, small, committed in standard
  GitHub denoising repos (e.g. KAIR/DnCNN). `[care]` (pin the mirror).
- **Existing method.** Classical **NLM** (Buades, Coll, Morel 2005) and **BM3D** (Dabov et al. 2007) —
  the real published self-similarity denoisers.
- **holostuff faculty.** `denoise` — manifold / adaptive / NLM / PnP / codebook (B7–B9).
- **The bar.** On real images + additive Gaussian noise, holostuff's **NLM/self-similar denoise is
  competitive with classical NLM** (its own algorithm running natively on the engine's recall), and the
  manifold/PnP modes **help in the high-noise / low-rank regime**, with the adaptive mode estimating σ
  to avoid the low-noise over-smoothing negative.
- **The negative we keep.** Will not beat BM3D or deep denoisers; **manifold projection HURTS at low
  noise** and **DESTROYS no-manifold signal** (both kept negatives, already measured). The contribution
  is the *unification* and knowing exactly where each map helps.

### George Drettakis — Gaussian splatting
- **Dataset.** The **Kodak** image set — 24 real photographs, the classic codec benchmark, on GitHub
  mirrors. `[care]`.
- **Existing method.** JPEG at a matched byte budget, plus the internal WHT-plate archive as a second
  baseline.
- **holostuff faculty.** `splat_field` / `splat_archive` (B8) — a scene as a content-addressable bundle
  of role-bound Gaussian primitives.
- **The bar.** Fit a real Kodak image as a splat bundle that **matches/beats the WHT-plate archive's
  PSNR at a fixed byte budget**, while ADDING **region-query** ("what's near here?") and **progressive
  refinement** (densify where the residual is large — 3DGS's own move); and show that fitting few splats
  to a noisy image *denoises* it (the splat↔denoise meeting point).
- **The negative we keep.** **Isotropic 2-D only** — anisotropic covariances and the 3-D primitive (the
  actual 3DGS) are deliberately deferred (A13); JPEG wins on raw bytes-per-dB. The win is the
  **queryable, refinable, erasure-robust** representation, not codec efficiency.

### Brian Eno — philosophy / generative art
- **Dataset.** **JSB Chorales** — the full corpus of 382 four-part Bach chorales, JSON, tiny, committed
  directly on GitHub. `[direct]`.
- **Existing method.** A first-order Markov chord model, and the kept-negative **bare-codebook sampler**
  (which can only return stored atoms).
- **holostuff faculty.** `generate_vector` (B10 diffusion — run the denoiser backwards from noise) over
  a **composed subspace** (A11), not the bare codebook.
- **The bar.** Build a composed subspace from the chorales and generate novel four-part sequences by
  running the denoiser backwards from noise — generated chords are **valid** (high calibrated recall
  against the chorale manifold / obey basic voice-leading) yet **novel** (not verbatim a training
  chorale) — beating the bare-codebook sampler that can only echo stored atoms.
- **The negative we keep.** Not competing with deep music models; **bare-codebook generation is
  degenerate** (the kept negative). The claimed, interesting regime is generation over the *composed*
  manifold. *What counts as noise is a choice of which manifold to keep* — and the chorale manifold is
  the choice that makes "valid" precise.

---

## How to sequence the build

Ordered by value over effort, with each one's honest status. The split is sharp: **twelve of these
ride entirely on faculties that already ship** — the only new code is the data harness — while a
handful are gated on a genuinely-unbuilt faculty.

**Wave 1 — pure data-harness over shipped faculties (cheapest; do first).**
Tarter, Siemion, Cranmer, Pharr, Duda, Plate, Ozcan, Milanfar, Drettakis, Eno, Olshausen, Togelius.
Each needs only: a `SOURCES` entry, a small loader, and a measurement script against the named
baseline. `recognize`/`recall_calibrated`/`recognize_batch`/`denoise`/`splat_archive`/`decompose_structure`/
`save(quant='rd')`/`generate_vector`/`decide_confidence` are all live. Land the **fetcher itself** (§0)
as the very first commit — it unblocks all twelve.

**Wave 2 — needs one small new piece.**
- **Stam** — the committed FFT-on-a-torus solver (a ~50-line pseudospectral generator) before
  `learn_dynamics` has a field with real dynamics to chew on. `[gen]`.
- **Puckette** — the spectral/audio modality (encode magnitude/phase as an FHRR hypervector). A7.

**Wave 3 — gated on genuinely-unbuilt faculties (research-heavier).**
- **Adamatzky** — multi-terminal flow (A9): extend `solve_maze` from single-source/sink to a network.
- **Baker** — pluggable, cleanup-based energy + structure-compare for `assemble` (A3).
- **Stoudenmire** — the tensor-train (MPS) bind mode (A14): the heaviest, most speculative.
- **Macklin** — the unified iterate-a-projection faculty (A4) plus the determinism audit.

Each test, when it lands, keeps the close-out ritual the modules ship under: a `tour.py` line, a
`NOTES_concepts.md` entry recording what the data said (wins *and* the designed-in negative), the
README counts, an integration test that runs the pipeline end to end (not just an import check —
remember the lesson that naive cross-module chaining *regressed*), and a clean zip rebuild verified
from extraction. Backward-compatible defaults; negatives on the record.

---

## The honest bottom line

This is a benchmark *suite*, not a victory lap. Read down it and the shape is consistent: against each
field's own mature method, holostuff's win is rarely "higher raw accuracy" — a tuned BDT, FAISS, BM3D,
H.264, a trained FNO or KGE model will each beat it on its home metric, and the suite says so up front.
What holostuff offers instead, and what these tests are built to *measure*, is the property the mature
tools mostly lack: **calibrated honesty** (Tarter, Siemion, Cranmer, Pharr), **graceful degradation
under erasure** (Ozcan, Drettakis), **determinism and explainability** (Togelius, Macklin),
**content-addressability** (Puckette, Pharr, Plate), and **structure-dependent compression** that is
honest about when it fails (Quílez, Duda). A few are places it can genuinely lead — calibrated streaming
detection, reconstruction-under-degradation as a solved inverse problem, generation over a composed
manifold, a live capacity readout grounded in the very theory (HolE) that a seat-holder invented.

The suite's real value is that it turns "more useful in their field" from a claim into a *measurement*,
each with its failure regime on the record before the first run. That is the engine's own method —
ground every idea in a field's real published baseline, prototype it on real data, ship what clears its
bar, and write down plainly where the data said no — scaled from a single module to a nineteen-seat
benchmark.

---

### Datasets and methods this program leans on
- **Tarter:** UCBerkeleySETI/blimpy Voyager-1 GBT filterbank; matched filtering + Wald, *Sequential Analysis* (SPRT).
- **Siemion:** HTRU2 (HTRU survey); Lyon, Stappers, Cooper, Brooke, Knowles (2016), *Fifty Years of Pulsar Candidate Selection*, MNRAS.
- **Baker:** Rosetta fragment-assembly methodology; small PDB structures (Biopython fixtures / RCSB mirror).
- **Olshausen:** MNIST; Kymn, Mazelet, Ng, Kleyko, Olshausen (2024), *Compositional Factorization of Visual Scenes*; Frady et al. (2020), *Resonator Networks*.
- **Cranmer:** MAGIC Gamma Telescope (UCI); simulation-based inference + look-elsewhere / trials-factor methodology.
- **Pharr / Duda:** GloVe (Pennington, Socher, Manning 2014) / Fashion-MNIST (Xiao, Rasul, Vollgraf 2017); ANN-benchmark methodology; Duda (2009/2013), *Asymmetric Numeral Systems*; KLT water-filling, rate-distortion theory.
- **Adamatzky:** Tero, Takagi, Saigusa, Ito, Bebber, Fricker, Yumiki, Kobayashi, Nakagaki (2010), *Rules for Biologically Inspired Adaptive Network Design*, Science; MST / Steiner baselines; TSPLIB.
- **Quílez:** Brodatz texture album; box-counting fractal dimension; JPEG baseline.
- **Togelius:** Summerville et al., *The VGLC: The Video Game Level Corpus*; DQN; standard PCG playability metrics.
- **Macklin:** CMU MoCap / BVH; Macklin et al., position-based / XPBD; H.264 motion compensation; Kalman filtering.
- **Stam:** Stam (2001), *A Simple Fluid Solver based on the FFT*; Koopman / Dynamic Mode Decomposition; Li et al. (2020), *Fourier Neural Operator* (reference).
- **Ozcan:** BCCD (Shenggan/BCCD_Dataset); Venkatakrishnan et al. (2013), *Plug-and-Play Priors*; Romano, Elad, Milanfar (2017), *RED*; total-variation inpainting.
- **Puckette:** Free Spoken Digit Dataset (Jakobovski); MFCC + kNN; the phase vocoder.
- **Plate:** UMLS / Kinship / Nations (in TimDettmers/ConvE); Nickel, Rosasco, Poggio (2016), *Holographic Embeddings of Knowledge Graphs*; TransE.
- **Stoudenmire:** Frey faces / MNIST; Stoudenmire & Schwab (2016), *Supervised Learning with Tensor Networks*; MPS/DMRG truncation.
- **Milanfar:** Set12 / BSD68 (cszn/KAIR); Buades, Coll, Morel (2005), *Non-Local Means*; Dabov et al. (2007), *BM3D*.
- **Drettakis:** Kodak image set; Kerbl, Kopanas, Leimkühler, Drettakis (2023), *3D Gaussian Splatting*; JPEG baseline.
- **Eno:** JSB Chorales (czhuang/JSB-Chorales-dataset, from Boulanger-Lewandowski 2012); first-order Markov; Ramsauer et al. (2020), *Hopfield Networks is All You Need* (the generative-denoising engine).

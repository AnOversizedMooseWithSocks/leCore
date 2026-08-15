# Benchmarks — real data, SOTA context, negatives loud

Run it yourself: `PYTHONHASHSEED=0 python3 tools/benchmarks_flagship.py` (~1 min, CPU).
All numbers below are from that script on real data (35,934×768 Wikipedia embeddings,
S&P 500 CSV). Category honesty first: **leCore is a KB-to-1M-scale engine by design.**
We do not claim to beat HNSW/ScaNN raw QPS at SIFT1M/DEEP1B scale — that comparison is
a category error in both directions. We benchmark what the 2026 field's own literature
says it is missing.

## 1. Calibrated abstention — no SOTA system ships this

Promised false-alarm rate vs realized, on **shuffled-real** noise (the adversarial null;
iid gaussian is the easy case), 400 queries per cell:

| promised α | realized FA | power |
|---|---|---|
| 0.01 | 0.013 | 1.000 |
| 0.05 | 0.055 | 1.000 |

Both realized rates sit inside the binomial 95% CI of the promise (n=400). The engine
refuses noise at the rate it promised and keeps every true signal. Nothing on the
ann-benchmarks leaderboard makes — or could check — this promise.

## 2. Self-measured approximate search — the field is asking for this

Context from the literature: DARTH (2025) turns recall into a service-level objective
via adaptive early termination — the closest prior art to our recall budget. A May-2026
production post-mortem documents HNSW recall **silently degrading past ~200k vectors**,
concluding "instrument before your users find it." That instruction is this feature.

Ours, measured on the caller's own vectors (Wilson 95% CI): screens recall@1 **0.97
[0.94, 0.99]** at 35% of the corpus scanned, order-independent — and when data defeats
the structure, the index **demotes to exact with the number attached**. No silent low
recall, structurally.

**Negative found, then CONVERTED (the sequence matters):** the first bench run measured
screens at 21.1 ms/q — *slower* than exact BLAS (10.6). Diagnosis: the fused matmul was
already there, but fancy-index gathering copied ~77MB per query and a per-query Python
dict mapped 12k scores. Lever 1 (bake once, scan views): block members now lay contiguous
at build, candidates are slices, the dict is positional takes. **Re-measured: 5.1 ms/q —
1.9× faster than exact at the same 0.97 recall, same tie rule.** The loss is kept on
record above because the conversion is only credible with the loss beside it.

Then the second lever: `Index(fast=True)`, a two-stage f32 engine under every route —
f32 scan at half the memory traffic, f64 rescore of an over-fetched shortlist, and a
margin **arbiter** that falls back to full f64 whenever f32 rounding could flip the
boundary (fallbacks counted; a planted boundary-overflow tie pins that it fires).
Results are identical to f64 — indices bit-equal, scores within 1e-10 — by
construction, not sampling. **Measured: exact 10.4 → 5.1 ms/q; screens 5.6 → 1.9 ms/q.
Net: 5.5× faster than exact f64 at 0.97 self-measured recall, exactness arbitrated,
zero fallbacks observed on the real corpus.**

## 3. Rule-sized models — the Tracr-lane comparison

Tracr (DeepMind 2023) compiles programs into transformer weights and stores **the
weights**. Our lane stores **the rule**: a 175-byte model file re-bakes 2,048 certified
weight parameters bit-identically (measured in the script; the arbitrary-precision claim
is the sha256 in the manifest). Same task family, ~4 orders of magnitude smaller
artifact, exactness by construction rather than by training.

## 4. Lossless codecs — the honest bar, then the bar raised

Byte-plane packing (`float_pack_bytes`): transpose the byte planes so like bytes sit
together, then lzma. **1.19× on the same embedding bytes where every general codec gets
1.08×**, byte-exact round trip. Kept negative: row-delta before planing adds nothing —
embedding rows are not sequentially correlated; measured, recorded, not shipped.

### The baselines

General-purpose baselines on our real data: sp500.csv → gzip 2.79×, bz2 3.19×, lzma
3.75×; float32 embeddings → **~1.08× for all three** (embeddings are near-incompressible
to general codecs). Any leCore codec claim must beat these numbers on the same bytes or
say why it measures something else. This table exists so future claims have a bar.

## 5. Bit-reproducibility as a contract

One tie rule (lowest index) delegated to by every ranking path; `PYTHONHASHSEED` pinned
for canonical runs and **randomized** for release clean-extract verification; the entire
benchmark script repeats bit-identically. The SOTA leaderboards do not measure this
because their entrants cannot promise it.

## What would raise eyebrows, precisely

Not a QPS bar chart. The eyebrow is the *contract stack*: a promised false-alarm rate
realized within CI on adversarial noise + recall self-measured on your data with honest
demotion + models four orders of magnitude smaller than the weights they re-bake +
bit-identical runs — all in NumPy + stdlib, all re-runnable in one minute. The field's
own 2026 papers and post-mortems describe the missing instrumentation; this repo ships
it, measured.

## 6. The installability verdict rate — the metric that keeps us honest

Signature-level census: 72.1% of 1,944 faculties are shaped like certification candidates.
**Probe-sample verdict (n=80, deterministic, SIGALRM-guarded): 8.8% actually certify today**
— 87.5% of nominal candidates take non-vector arguments the signature could not exclude;
3.8% are callable but genuinely nonlinear (refused, correctly). The candidate number
flattered us by 8×; the verdict number is the one on the wall. Every projector-vocabulary
extension and every reshaping adapter must move **this** rate, re-measured by
`python3 tools/installability_census.py --probe`.

### 6b. Typed probe of the not-probe-callable — and a lever that measured zero

What the 87.5% actually take (n=70): 18.6% **text** (token-space work — the host's native
job, counted out of the projector's ledger honestly), 14.3% dict, 7.1% int, 7.1% float
list, 52.9% none of the battery. **Kept negative: the reshape-adapter lever measured a
0.0-point delta on this sample** — no 2D-array takers; the hypothesis that flattening
adapters would move the verdict rate did not survive measurement. The corrected roadmap:
the mind facade's signatures are the wrong sampling frame for "what math installs" —
facades are parameterized entry points; the certifiable cores are the inner module
functions the FAC compiler already consumes directly. The next census frame is the module
level, not the facade level.

## 7. The memory hierarchy — measured, and it predicts the rest of this document

`mind.memory_mountain()` sweeps streaming bandwidth against working-set size on the box
you run it on. This box: **~90 GB/s peak at 0.5–1 MB** (L2-resident), a knee through
1–4 MB, and a **~26 GB/s floor from 4 MB out** — L3 and RAM indistinguishable on this
virtualized host, reported as one floor because inventing a boundary the data doesn't
show would be fiction. The payoff is prediction: bytes-touched ÷ floor reproduces the
fast-arbiter table to ~15% — exact f64 **9.1 predicted / 10.4 measured** ms, f32
**4.5 / 5.1**, screens-f32 **1.6 / 1.9**. Section 2's speedups are not cleverness;
they are the mountain wearing three different working sets, and now the engine can
tell you that *before* you benchmark.

Two instrument honesties, pinned: the small-size flank measures Python/BLAS dispatch,
not L1 — a Python-level probe cannot see L1 and the tier detector excludes that flank
by design; and the storage stack's claims were spot-checked the same session
(cold-store round trips byte-exact; the WGSL virtual-GPU lane reports
`{available: False, why: wgpu not installed}` in this container — an environment
refusal with its reason, not a silent skip).

### 6c. The module frame — a hypothesis refuted by its own census

The typed-probe session predicted the facade was the wrong sampling frame and the module
level would reveal the compilable math. Measured (688 modules, 2,913 public functions,
1,105 single-required-arg, n=140 probed): **8.6% certify — statistically the same as the
facade's 8.8%.** The frame-correction hypothesis is dead, on our own instrument, and the
composition is informative: ~80% of single-arg functions take structured arguments at
*both* levels, while **refusals triple at module level (11.4% vs 3.8%)** — callable,
genuinely nonlinear inner math. The corrected roadmap: installed coverage grows through
the host vocabulary (turning refusals into gated/rmsnorm-style certifications) and
through FAC's adapter lambdas — which the single-arg census systematically undercounts,
since every installed customer so far was a closure the census would have skipped.

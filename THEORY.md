# holostuff — Theory and Guarantees

*The load-bearing claims the engine rests on, gathered in one place and each tagged by what backs it.
This is not a paper and makes no novel proofs — it is the honest middle ground the project actually
occupies: results **cited** from the literature, results **measured** here against a named test, and the
**kept negatives** that bound every claim. The rule for this document is the project's rule: no statement
appears without either a citation or a test.*

## How to read the tags

- **[CITED: ref]** — established in the literature; holostuff implements or relies on it. The engine does
  not re-prove it.
- **[MEASURED: pointer]** — measured in this repository. The pointer is the test (or the
  `NOTES_concepts.md` section) that backs it, runnable from a clean checkout.
- **[KEPT NEGATIVE: pointer]** — a measured limitation, kept loud on the record so the claim above it is
  not oversold.

Everything below is dense NumPy on one bind/bundle/cleanup kernel; the standing scope limit (**moderate N**,
because the only linear algebra used is dense `eigh`/`svd`, no sparse solver and so no second dependency) is
stated once here and applies throughout.

---

## 1. The algebra: bind, bundle, cleanup

- **bind(a, b) is circular convolution, computed with the real FFT.** [CITED: Plate, *Holographic Reduced
  Representations*, 1995 / CSLI 2003] · [MEASURED: `holographic_ai.bind`; the cleanup it feeds matches the
  reference cleanup and round-trips exactly — `test_core_persistence.py::test_core_cleanup_matches_vocabulary_cleanup`,
  `::test_vocabulary_round_trips_exactly`].
- **unbind is circular correlation (the involution), the *approximate* inverse.** `unbind(p, bind(p, x)) ≈ x`;
  it is exact only for unitary (unit-magnitude-spectrum) keys, and for a generic Gaussian key
  `bind(p, unbind(p, x))` is near-noise. [CITED: Plate] · [MEASURED: the round-trip degrades to cosine ~0.067
  for a Gaussian key — this is the mechanism behind the BLD-1 collision negative in §6,
  `test_holographic_verify.py::test_linear_collision_is_constructible_kept_negative`].
- **bundle is superposition (vector sum); cleanup is nearest-codebook (argmax of the dot).**
  [CITED: Plate] · [MEASURED: `test_core_persistence.py::test_core_cleanup_matches_vocabulary_cleanup`].
- **The one identity that unifies the whole engine: a binding is a circular convolution is a per-frequency
  phase rotation is a compressed projection of the tensor product.** HRR's convolution is a
  random projection of Smolensky's tensor-product binding; the FHRR variant *is* the per-bin phase rotation.
  [CITED: Smolensky, *Tensor product variable binding*, 1990; Plate 2003] · [MEASURED: the tensor mode in
  `holographic_tensor.py`; the FHRR phasor bind/unbind is exact —
  `test_holographic_fhrr.py::test_fhrr_bind_unbind_is_exact`].

---

## 2. Capacity (how much fits before noise wins)

- **A fixed-D trace has finite capacity; past a load that scales ~D/log it, noise wins — the "capacity
  cliff."** [CITED: Plate's HRR capacity analysis] · [MEASURED: shown, not hidden — a 2048-d memory recalls
  100% of 64 pairs but ~0% of 2048; surfaced live by `UnifiedMind.capacity_report` and
  `holographic_tree.capacity_curve`, and recorded in `NOTES_concepts.md` "Capacity / SNR vs the cliff"].
- **Dense-associative (modern Hopfield) cleanup has exponential capacity and single-step retrieval, and at
  β→∞ reproduces the engine's exact hard-nearest-neighbour decision — so it is a strict superset of the
  classical cleanup.** [CITED: Krotov & Hopfield 2016; Demircigil et al. 2017; Ramsauer et al. 2020,
  *Hopfield Networks is All You Need*] · [MEASURED: `test_holographic_hopfield.py` (the update is the
  Ramsauer rule bit-for-bit; high-β pins it to argmax; the decisive win is continuous-vector denoising)].
- **FHRR (complex unit-phasor) atoms hold substantially more pairs than the real-valued HRR core at high
  load.** [CITED: the cross-VSA capacity comparisons that recommend FHRR] · [MEASURED: at 40 pairs / 256-d,
  ~0.90 vs ~0.61 readback; at 60, ~0.74 vs ~0.40 — `test_holographic_fhrr.py::test_fhrr_holds_more_pairs_than_real_hrr_at_high_load`].
- **[KEPT NEGATIVE]** FHRR has *no* advantage at low load (where the engine normally runs both are already
  perfect), and the real-domain `unitary` atoms do not capture FHRR's edge on their own. [MEASURED:
  `test_holographic_fhrr.py::test_fhrr_has_no_advantage_at_low_load`].

---

## 3. Geometry and compression

- **Real structured states collapse to a low-rank subspace; an SVD/KLT (consolidation) finds it, shrinking
  storage ~21× with the decision geometry intact.** [MEASURED: the consolidated brain round-trips through its
  own basis — `test_core_persistence.py::test_consolidated_brain_round_trips_with_its_basis`].
- **That KLT *is* the decorrelating transform rate-distortion theory asks for; chaining KLT → water-filling
  bit-allocation → a bit-exact rANS coder yields a geometry-preserving code.** [CITED: rate-distortion theory;
  Duda, *Asymmetric Numeral Systems*, 2009/2013] · [MEASURED: rANS is bit-exact and codes within ~0.3% of
  entropy; the geometry code preserves cosines — `test_holographic_ratedistortion.py`].
- **Against the off-the-shelf tool, that code wins exactly where the data is low-rank and loses where it is
  not.** [MEASURED: `benchmarks/bench_compression.py`, asserted in `test_benchmarks.py` — rd beats zlib/lzma
  ~34× on rank-8 data at N=2000, and **loses** to int8+zlib on full-rank random data (no structure for the
  KLT to exploit; the shared basis costs more than it saves)].
- **int8 / `auto` quantization is decision-safe — chosen only where the data's own separation proves 8-bit
  leaves the argmax intact.** [MEASURED: `test_core_persistence.py` (the dynamic mix engages; fidelity holds)].
- **[KEPT NEGATIVE]** *binary* quantization is not decision-safe: it distorts the pairwise-similarity
  geometry enough to corrupt fine readback, so `auto` never selects it. [MEASURED / NOTES: "Dynamic
  quantization"].

---

## 4. Search (sublinear recall, deterministic flow)

- **A random-projection-tree forest reaches near-exact recall@1 at a small, shrinking fraction of the
  comparisons.** [CITED: Dasgupta & Freund, random-projection trees] · [MEASURED: `benchmarks/bench_recall.py`,
  asserted in `test_benchmarks.py` — 3% of the comparisons at N=20k with recall@1 100%→97%].
- **[KEPT NEGATIVE]** the forest buys sublinear *work*, not wall-clock: against a single BLAS matrix-vector
  scan it only wins on wall-time past ~20k items. [MEASURED: same benchmark].
- **The Tero flow-conductance model finds min-cost paths deterministically, ~100–340× faster than the
  stochastic elitist-ant solver it replaced.** [CITED: Tero et al. 2007, the Physarum flow model] ·
  [MEASURED: optimal on braided mazes and deterministic — `test_holographic_flow.py`].

---

## 5. Detection and honesty (calibrated, not asserted)

- **A recall can report a calibrated false-alarm probability** — the null is the distribution of match
  scores on *random* queries, so thresholding at p≤α holds the false-alarm rate at α. [MEASURED: `RecallNull`;
  the coverage check confirms ≈0.05 at α=0.05 on held-out noise — `test_holographic_honesty.py` and the
  calibration tests recorded in `NOTES_concepts.md` "Calibration coverage"].
- **Streaming detection reaches a target error pair (α, β) in fewer expected samples than any fixed-N rule.**
  [CITED: Wald, *Sequential Analysis* — the SPRT optimality] · [MEASURED: the savings appear in the
  overlap regime (well-separated → ~1 sample, heavy overlap → ~4, each ≈half a fixed window) — the SPRT tests
  in `NOTES_concepts.md` "The scan faculty" / "Honesty reaches action"].
- **Across many candidates, the look-elsewhere / trials factor is controlled by false-discovery-rate
  control.** [CITED: Benjamini-Hochberg; Benjamini-Yekutieli for dependent tests] · [MEASURED: `bh_fdr`, wired
  through `recognize_batch` and the `scan` faculty].

---

## 6. Self-verifying storage (BLD-1)

- **A segment tree whose combine is bundle over position-bound items detects and localises a single tampered
  item in ≤ log₂(n) composite comparisons, deterministically.** [MEASURED: 40/40 localised in 7 checks at
  n=64 with 0 false positives; a slot swap is caught because position is bound into each leaf —
  `test_holographic_verify.py::test_detect_and_localize_single_tamper_in_log_checks`,
  `::test_position_binding_catches_reordering`].
- **[KEPT NEGATIVE — the load-bearing one]** the root is a *linear* combination, so the map items→root is
  many-to-one for n>1 and collisions are *constructible*: a key-aware adversary cancels a change to one item
  with a deconvolution-chosen change to another, leaving the root bit-for-bit unchanged. This is evidence of
  **accidental corruption / uncoordinated tampering, not cryptographic tamper-proofing** (a hash resists
  collisions; cancelling a linear sum is a division). [MEASURED:
  `test_holographic_verify.py::test_linear_collision_is_constructible_kept_negative` — an invisible canceling
  pair at cosine 1.0]. The plan's separate guess — that quantising the checksums would blind it to small
  tampers — was **disproven** (detection holds to 2-bit at n=1024, because high dimension always pushes some
  component across a quantiser boundary).

---

## 7. Determinism (an engineering guarantee, not a theorem)

- **Every faculty is seeded and reproducible run-to-run; a saved mind round-trips identically — both
  `classify` and `decide` give the same answers after save/load.** [MEASURED:
  `test_core_persistence.py` round-trip suite; `test_integration.py` (the mind save/load round-trips classify
  AND decide identically, including under `quant='rd'`)].
- **The bit-exact tie-break discipline.** A change identical to 1e-12 can still flip an argmax on a knife-edge
  tie (the `bind_batch` lesson), so vectorizations are kept bit-identical *and* out of tie-sensitive paths.
  [MEASURED / NOTES: the `bind_batch` negative; the INV-5 pass deliberately left the creature's tie-sensitive
  `decide` un-vectorized while shipping the FHRR-cleanup vectorization, which is a value-cleanup path with
  sims matching to ~1e-16 — `test_holographic_fhrr.py::test_phasor_cleanup_vectorized_matches_bruteforce`].

---

## 8. The standing kept-negatives (each measured, each travels)

The negatives are first-class results; where the data said no is on the record.

| Negative | Where it bites | Backed by |
|---|---|---|
| Linear bundle is not collision-resistant | self-verifying storage vs a key-aware adversary | `test_holographic_verify.py` (§6) |
| Discrete atoms favour exact 1-NN | no learned-energy approximation beats hard nearest-neighbour on discrete pointers | `NOTES_concepts.md` learned-energy / `test_holographic_hopfield.py` |
| Binary quantization distorts geometry | fine readback corrupts; `auto` won't pick it | `NOTES_concepts.md` "Dynamic quantization" |
| rd loses on full-rank data | no low-rank for the KLT to exploit | `test_benchmarks.py` (§3) |
| Forest loses wall-time at small N | a BLAS scan is faster until ~20k items | `test_benchmarks.py` (§4) |
| Fixed-rank denoising over-smooths at low noise | needs the noise estimate / adaptive rank | `test_holographic_denoise_adaptive.py` |
| n-gram confidence is unreliable on sparse context | a once-seen context returns 1.00; held-out accuracy is the honest metric | `NOTES_concepts.md` generation thread |
| de-Doppler bank vectorization regresses at scale | `np.roll`'s C path beats a fancy-index gather on large arrays | `NOTES_concepts.md` INV-1 |
| FHRR `B.conj() @ q` is no faster than the loop | the conjugate copy costs as much as the matvec; the real-matvec is the win | `NOTES_concepts.md` INV-5 |
| PnP/manifold denoising hurts where there is no manifold | projecting random data destroys signal | `NOTES_concepts.md` denoising cluster |

---

## What this engine is, and is not

It **is**: one set of primitives (bind, bundle, cleanup, on the real FFT) shown — by citation where the math
is known and by measurement where it is not — to be load-bearing across capacity, factorization, detection,
search, dynamics, and storage; with the honest cliff and the kept negatives surfaced rather than hidden.

It **is not**: a source of novel proofs, a cryptographic system (§6), or a large-graph spectral engine
(dense `eigh` bounds it to moderate N). Those boundaries are the point — they are stated so the claims above
stay exactly as strong as their evidence, and no stronger.

# Panel Review — Cross-Cutting Backlog

*The 19-seat panel convened on the cross-cutting backlog (the above/below transfer sweep + the text-gen,
image-gen, and creature-brain reviews). As always, each seat speaks only through that field's REAL published
methods — no opinion is invented beyond what the method itself argues. The job: pressure-test the sweep's
framing, challenge the priority, and return a debated ordering with dissents on the record.*

---

## The question put to the panel

Sixteen candidate transfers of the rendering-lessons faculties, in four groups: A (below the substrate), B
(text generation), C (image generation), D (the creature brain). The backlog's framing claim is that the
powers are mostly ABOVE-stack usage patterns and the kernel is already near-optimal, so the below-stack items
are honest probes most likely to return kept negatives. Is that framing right, and what order maximises
learned-result-per-unit-effort?

---

## Seat-by-seat (grouped by the methods that bear on the items)

**Pharr — PBRT, multiple-importance sampling (Veach & Guibas), Russian roulette, low-discrepancy & blue-noise
sampling.** His methods are load-bearing for the most items (A1, B1, B2, B4, D1, D3), so his read carries
weight — and it cuts *against* over-optimism on his own turf. On A1: low-discrepancy and blue-noise win where
samples are sparse relative to the space; in a 1024-dimensional space, atoms are already so sparse and
near-orthogonal that the coverage advantage is asymptotically gone. The MIS items (B1, D3) are where his
method genuinely transfers — the balance heuristic is *designed* for combining two estimators of the same
quantity, which is exactly predictor-vs-verifier (B1) — but he flags the precondition the MIS literature is
explicit about: the two estimators must be on a common density scale, or the balance weights are meaningless.
Russian roulette (B2) is sound but he agrees it largely duplicates `generate_structured`'s existing
coherence defence.

**Plate — Holographic Reduced Representations, the capacity mathematics, the Welch bound.** On A1 he is the
decisive voice: the expected coherence of random unit atoms approaches the Welch lower bound as dimension
grows, so a repulsion-spread codebook can only recover the gap to Welch, which is vanishing at d=1024. His
capacity analysis says the recall cliff is set by the bundle's signal-to-crosstalk ratio, which improves with
dimension far faster than with codebook spreading. Verdict: A1 is a real effect at small dims and a near-no-op
at the operating point — keep it as a probe, expect the negative.

**Olshausen — sparse coding, efficient codes, the resonator/substrate.** Endorses the framing's below-stack
conclusion from the coding side: an efficient code over a high-dimensional space is already near-optimal under
concentration, so A1/A2 have little to improve. But he is the strongest advocate for **A3**: adaptive kernel
placement is the *efficient-coding* principle itself — allocate representational resource in proportion to the
density of what is represented — and an encoder on a uniform grid is provably wasteful on a non-uniform value
distribution. A3 is the one below-stack item his method predicts will pay.

**Milanfar — Regularisation by Denoising (RED), Plug-and-Play priors, deblurring, kernel regression.** Leads
the image work and wants **C4** at the front of group C, reframed in his terms: a splat render is a known
forward operator (a sum of Gaussians) applied to a latent sharp image; recovering the latent is *deconvolution*,
and the looping negative-lobe sharpener with the discrepancy stop is a Van-Cittert/RED iteration with a
data-consistency guard — exactly the regime where his methods provably converge while naive unsharp diverges.
He also backs **B3/C3**: RED's convergence theory says you iterate to a fixed point, and stopping when the
residual stops moving is the correct, not merely cheaper, termination. On **A2** he is cool: cleanup against an
explicit discrete codebook is detection, not deconvolution, and detection is already solved by the matched
filter (argmax) — the negative-lobe trick has nothing to sharpen there.

**Drettakis — 3D Gaussian Splatting, adaptive densification.** Wants **C1 pulled to the front of the whole
list**, not fifth. His position from the method: densification — splitting and growing Gaussians where the
gradient says the reconstruction is under-resourced — is *the* mechanism that makes Gaussian splatting work,
and it is precisely what addresses `splat_aniso`'s kept negative (non-convex, warm-start-dependent, more
splats not monotone). A coarse-to-fine residual-pyramid warm start is the from-scratch analogue of his
densification schedule. He concedes C1 is more effort than the adaptive-stops, which is why the panel does not
seat it first, but he registers that it is the highest-ceiling image result on the board.

**Macklin — Position-Based / XPBD constraint solvers, deterministic iterative methods, bit-exact tie-breaks.**
Owns the determinism rule for every adaptive item (B3, C3, D2): an early-stop or a robust accumulation that is
not bit-identical to the old path when the option is off, and not deterministic in seed when on, is a
regression regardless of its average-case win. His solver experience also backs C3 directly: a fixed iteration
count is the wrong default for an iterative solver; convergence-gated iteration with a floor and ceiling is the
standard, and it is risk-free when the gate only fires after the residual flattens.

**Eno — generative systems, the reframe (what counts as "noise", what counts as "done").** The conscience on
the adaptive-stops. His caution: "done" for a generative process is a *choice*, not a fact, and a stop
criterion tuned to first-convergence will reliably amputate the late, interesting part of a generative
trajectory. He signs off on B3/C3 *only* with the stop predicated on stability-and-validity (the structure has
settled and re-encodes correctly), never on the first step that merely looks converged — and he wants novelty
measured before/after, not assumed unchanged.

**Togelius — procedural content generation, game AI, reinforcement learning.** Carries the creature group and
will not let D1 oversell. From the exploration literature: quasi-random/low-discrepancy exploration helps when
the thing being covered is a *space* — a continuous state space, a parameter space — and degenerates to
round-robin over a handful of discrete actions, where it buys almost nothing. So D1's bar must be written on
STATE coverage and reward-vs-episodes, not action diversity, and its value is contingent on the state
representation exposing something to cover. He strongly backs **D2** (robust value estimation is standard
defensive RL) and is lukewarm on **D3** (a safety veto is a constraint, and his methods treat constraints as
hard masks, not as estimators to blend — MIS is the wrong tool for the veto, though it fits a value+novelty
blend).

**Baker & Adamatzky — fragment-assembly / energy-landscape search; Physarum flow search.** Both read **D4**
(re-anchored lookahead) as a search problem in disguise: a rollout that is not periodically snapped back to a
known-good configuration is a random walk that leaves the feasible manifold — Baker's fragment assembly
re-anchors to library fragments for exactly this reason, and Adamatzky's flow networks re-anchor to the
nutrient sources. Both endorse the RAY-2 discipline for imagined rollouts and both flag that D4 requires a
forward model the creature does not have — it is the research item, highest ceiling, largest effort.

**Cranmer — sequential analysis, SPRT-style stopping, calibration.** Backs the adaptive-stops (B3/C3) from the
sequential-testing side: the principled stop is when the evidence that further iteration will change the
outcome falls below threshold, which is exactly a convergence/stability test, and it has a calibrated false-stop
rate. Notes this is the same machinery as the throughput gate (RAY-1) and the resonator early-stop already
shipped.

**Puckette — phase vocoder, FFT analysis/resynthesis.** Owns **C2**: phase-domain interpolation is the phase
vocoder's core, and image morphing in the coefficient-phase domain is its 2-D cousin — uniform feature motion
because phase encodes position. He also states the failure mode plainly (it is the phase vocoder's classic
artifact): once a coefficient's phase advance between frames exceeds π the unwrapping is ambiguous and the
morph tears — which is PHASE-1's recorded wrapping negative, so C2 is an honest-negative-friendly probe, not a
sure win.

**Ozcan — computational holographic reconstruction under degradation.** Seconds Milanfar on C4 from the optics
side: recovering a sharp field from a smoothed measurement is the reconstruction problem his methods solve, and
a data-consistency-guarded iteration is the right form.

**Quílez — procedural/distance-field rendering, multi-resolution.** Supports C1's multi-resolution framing
(coarse-to-fine is the natural order for building structure) and is otherwise quiet here.

**Stam, Duda, Stoudenmire, Tarter, Siemion — fluid solvers on the torus; ANS / rate-distortion; tensor-network
low-rank; SETI detection statistics.** No method with decisive bearing on these sixteen items; they defer.
(Tarter/Siemion note in passing that the throughput-gate/stop family is the same detection-threshold logic as
their work, already reflected in the shipped RAY-1 and the adaptive-stops here.)

---

## The debates, on the record

**Debate 1 — is the below-stack group worth doing at all?** Pharr and Plate argue A1/A2 are near-no-ops at the
operating dimension (concentration of measure; random codebooks already near the Welch bound; argmax already
the matched filter). Olshausen agrees for A1/A2 but carves out **A3** as the genuine below-stack win (efficient
coding — adaptive resource allocation on a non-uniform distribution). *Resolution:* the framing stands — A3 is
promoted as the one real below-stack item; A1/A2 drop to honest probes with the negative expected. This is
recorded as the sweep's main finding: the powers are above-stack usage patterns, the substrate is already
near-optimal.

**Debate 2 — where does the headline image item sit?** Drettakis wants C1 (densification) first; Milanfar wants
C4 (deconvolution) leading the image work. *Resolution:* both are right about ceiling, but the panel orders by
result-per-effort, and C4 is a smaller, self-contained post-process of an existing render while C1 is a new
multi-scale fitter — so C4 seats ahead of C1, with Drettakis's dissent (C1 is the higher result) recorded.

**Debate 3 — do the adaptive-stops risk the generative output?** Eno warns first-convergence stops amputate
novelty; Macklin warns any non-deterministic stop is a regression; Cranmer supplies the principled criterion.
*Resolution:* B3/C3 proceed with the stop predicated on stability-AND-validity (not first-convergence),
bit-exact-deterministic, with novelty measured before/after — risk-free under those conditions, which is why
they seat first.

**Debate 4 — how much to expect from creature exploration?** Togelius insists D1's value lives in STATE-space
coverage, not action round-robin, so its bar is written on coverage + learning speed and its payoff is
contingent on the state representation. *Resolution:* D1 kept mid-list with the caveat in its bar; not
oversold.

**Debate 5 — is the lookahead a transfer or a new capability?** Baker and Adamatzky confirm the RAY-2
re-anchoring discipline is exactly right for rollouts, but D4 needs a forward model that does not exist.
*Resolution:* D4 is the research item — highest ceiling, largest effort — seated last among the real builds,
ahead of only the no-op probes.

---

## The panel's final ordering (the list)

Ordered by learned-result-per-unit-effort, risk-free-and-certain first, research and probes last. Dissents
noted inline.

1. **C3 — adaptive-stop for the anisotropic splat fit.** Risk-free, trivial, certain small win. *(Macklin,
   Cranmer.)*
2. **B3 — adaptive-stop diffusion for `generate_structure`.** Risk-free, easy, certain small win — stop on
   stability+validity, deterministic, novelty measured. *(ADAPT-2; Milanfar/Cranmer; Eno's condition.)*
3. **C4 — negative-lobe sharpening of the splat / archive reconstruction.** Proven transfer (XDATA-3), high
   upside on inherently over-smoothed renders; a guarded RED/Van-Cittert deconvolution. *(Milanfar leads,
   Ozcan seconds.)*
4. **D2 — robust reward / value accumulation.** Proven transfer (ACCUM-3) straight into the creature's value
   path; winsorise + harmonic-weight, deterministic. *(Togelius; robust statistics; Macklin's determinism
   rule.)*
5. **C1 — coarse-to-fine splat densification.** The headline image item; also fixes `splat_aniso`'s local-
   optimum kept negative via a residual-pyramid warm start. *(Drettakis — **dissent: pull to the front**, this
   is the highest-ceiling image result; seated 5th only on effort.)*
6. **A3 — adaptive encoder kernel placement.** The one promising below-stack transfer (CACHE-3 → encoder) —
   efficient coding on a non-uniform value distribution. *(Olshausen, Milanfar.)*
7. **D1 — low-discrepancy exploration.** Plausible creature win; bar written on STATE-space coverage + learning
   speed, not action round-robin. *(Togelius — caveat on the record.)*
8. **B1 — MIS-weighted steered generation.** Combine predictor + verifier by the balance heuristic; medium
   effort, uncertain until the two scores are put on a common density scale. *(Pharr — precondition noted.)*
9. **C2 — phase-domain scene morph.** Honest-negative-friendly probe; the phase-vocoder wrapping artifact
   (PHASE-1's recorded negative) bounds it. *(Puckette.)*
10. **D4 — re-anchored, throughput-gated lookahead.** Research / new capability — needs a forward model the
    creature lacks; highest ceiling, largest effort; the RAY-2 discipline applied to imagined rollouts.
    *(Baker, Adamatzky.)*
11. **The likely-no-op probes — run last, expect kept negatives:** **A1** (LD codebook — concentration/Welch),
    **A2** (negative-lobe cleanup — argmax is the matched filter), **B2** (gated generation — redundant with
    `generate_structured`), **B4** (LD sampling — a categorical draw), **D3** (MIS decision — the veto is a
    constraint, not an estimator), **D5** (observation denoise — high-dim averaging already does it).
    Confirming the engine is already right here is the result. *(Pharr, Plate, Olshausen, Togelius.)*

---

*Panel discipline held: positions are the seats' real published methods applied to these items, dissents are
recorded not smoothed, and the negatives are kept loud — the below-stack group is expected to confirm the
substrate is already near-optimal, which is itself the sweep's finding.*

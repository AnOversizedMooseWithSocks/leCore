# holostuff Cross-Cutting Backlog — do the new "super-powers" transfer elsewhere in the stack?

*An above/below sweep. The rendering-lessons campaign shipped ~19 faculties and a clutch of measured
negatives: low-discrepancy sampling (SAMPLE-1), throughput-gated re-anchored traversal (RAY-1/RAY-2),
adaptive splat count and adaptive iteration count (ADAPT-1/ADAPT-2), negative-lobe sharpening (XDATA-3),
multiple-importance combination (MIS-1), the gradient/irradiance cache and adaptive cache placement
(CACHE-1/CACHE-3), the smooth/sharp two-layer codec (CACHE-2), robust accumulation (ACCUM-2/3),
denoise-by-downscale and looping diffusion (XDATA-1/2), the FHRR phase-domain morph (PHASE-1), the
multi-resolution mipmap (SCALE-1), and the hole-free backward map (PHASE-2). The question this backlog
asks: which of these are general operators that pay off ABOVE (in the higher faculties — text gen, image
gen, the creature brain) or BELOW (in the kernel/substrate)? Grounded in an audit of the live code; every
item carries a measurement bar and, where the prior is that it won't pay, an honest "likely-no-op" flag.*

---

## The sweep's framing conclusion (state it before the list)

Reading the powers against the stack, they split cleanly into two kinds, and the split is the main result:

- **Most are USAGE patterns, and they live ABOVE.** Re-anchoring, adaptive stopping, importance combination,
  low-discrepancy sampling, negative-lobe sharpening, coarse-to-fine — these are *where you apply an
  existing primitive in a multi-step process*, not new primitives. So the richest transfers are in the
  faculties that run iterative or multi-step processes: generation (diffusion, beam search), the splat
  fitter (gradient descent), and the creature's learning/exploration loop.

- **The kernel is already near-optimal, so BELOW is mostly negative — which is worth confirming.** In high
  dimension i.i.d. random atoms are already near-orthogonal (concentration of measure puts random codebook
  coherence close to the Welch bound), so a "better-spread codebook" buys little at the operating dims; and
  hard-argmax cleanup is Bayes-optimal for "which atom" against an explicit codebook (the B1 finding —
  hard NN already wins on discrete atoms), so a "sharper cleanup" has nothing to fix there. The one
  genuinely promising below-stack transfer is to the *encoder*, whose kernels sit on a uniform grid and
  could be placed adaptively (CACHE-3) for non-uniform value distributions.

So the backlog is weighted toward the faculties, with the below-stack items kept as honest probes whose
most likely outcome is a kept negative confirming the substrate is already right.

---

## Group A — Below the stack (the substrate)

### A1 — Low-discrepancy / blue-noise codebook construction   [SAMPLE-1 → kernel]   *likely no-op at operating dims*
**What.** Build codebook atoms by a repulsion / low-discrepancy process on the unit hypersphere (maximise the
minimum pairwise separation, blue-noise on S^(d-1)) instead of i.i.d. Gaussian, so atoms are maximally
separated and cross-talk is minimised.
**Why.** SAMPLE-1 measured ~28% tighter coverage and ~13× lower integration error from low-discrepancy
points; atoms are points on the sphere and lower mutual coherence means cleaner cleanup and more capacity
before the noise-wins cliff.
**How.** A few steps of Riesz-energy / repulsion relaxation (or a spherical R-sequence) vs i.i.d. atoms;
opt-in codebook constructor, default unchanged. Measure max off-diagonal coherence AND pairs/atoms recovered
at fixed dim.
**Bar.** Lower coherence AND measurably higher recall capacity than i.i.d. atoms, deterministic.
**Risk (the honest prior).** Concentration of measure: at d=1024 random atoms are already near the Welch
coherence bound, so the gain is likely tiny here and only real at small dims. Expect a kept negative at the
operating point. *Seats: Pharr (low-discrepancy/blue-noise sampling), Plate (HRR capacity, the Welch bound).*

### A2 — Negative-lobe cleanup sharpening   [XDATA-3 → kernel]   *likely no-op for discrete*
**What.** When cleanup is near a tie, apply a negative-lobe (deconvolution) correction — sharpen the winner
by subtracting a fraction of the runner-up — the way negative reconstruction lobes sharpen edges.
**Why.** XDATA-3 showed negative lobes recover detail a smooth estimate lost; an ambiguous cleanup is a
blurred decision peak.
**How.** Replace argmax(V·q) with a sharpened score (deconvolve the similarity profile) only when the top
two are close; measure cleanup accuracy near collisions.
**Bar.** Better cleanup near near-collisions, no cost on well-separated cases.
**Risk (the honest prior).** Hard NN is already Bayes-optimal for "which atom" against an explicit codebook
(the B1 result); the only possible win is on CONTINUOUS / superposition cleanup, not discrete. Likely a kept
negative for discrete. *Seats: Milanfar (deblurring/RED), the negative-lobe finding.*

### A3 — Adaptive encoder kernel placement   [CACHE-3 → encoder]   *the promising below-stack item*
**What.** The ScalarEncoder/FPE places its kernels on a UNIFORM grid over [lo, hi]. Place them adaptively —
denser where the encoded value distribution has mass/detail — exactly CACHE-3's curvature-equidistribution,
applied to the encoder's basis instead of a cache.
**Why.** CACHE-3 just measured ~7× fewer anchors for equal quality on a non-uniform field; a uniform encoder
wastes resolution on empty value ranges and under-resolves clusters. For a known, non-uniform value
distribution this should give finer decode resolution where it matters at equal dimension.
**How.** An opt-in "fitted" encoder: given a sample of the value distribution, place kernels at equal-mass
quantiles of (density)^p (CACHE-3's rule); default stays the uniform grid. Measure decode resolution / error
across the value range vs the uniform encoder at equal kernel count.
**Bar.** Lower decode error in the dense regions at equal kernel count on a non-uniform value distribution,
ties on a uniform one (the CACHE-3 control).
**Risk.** The general encoder is distribution-agnostic; this needs the distribution (a fit step), so it is an
opt-in specialisation, not a default. *Seats: Milanfar (sampling density), Olshausen (efficient codes), the
CACHE-3 result.*

---

## Group B — Text generation

### B1 — MIS-weighted steered generation   [MIS-1 → text]
**What.** `generate_structured`'s beam keeps the candidate that best preserves running structure — a single
criterion. Combine the predictor's likelihood and the structure-verifier's coherence by the BALANCE
HEURISTIC (MIS), each weighted by its reliability, and beam/pick on the combined score.
**Why.** MIS-1: the balance heuristic provably combines two estimators with lower variance than choosing one.
The predictor (fluency) and the verifier (coherence) are two estimators of "good next token"; the current
code throws away the predictor's signal at the selection step.
**How.** Score candidates by the balance-heuristic blend of (predictor prob, verifier coherence); measure
held-out coherence (loop/incoherence rate) AND perplexity vs the verifier-only beam.
**Bar.** Better coherence AND not-worse perplexity than the current single-criterion beam.
**Risk.** MIS assumes comparable densities; the two scores may need calibrating before they combine, and the
verifier-pick may already be near-best. *Seats: Pharr (MIS / Veach–Guibas balance heuristic).*

### B2 — Throughput-gated generation (Russian-roulette stop / beam prune)   [RAY-1 → text]   *may be redundant*
**What.** Stop generation (or prune a beam) the instant the running coherence falls below a floor — the
traversal's throughput gate, abstaining on an incoherent tail instead of emitting it.
**Why.** RAY-1's gate recovers the valid prefix and abstains exactly when the signal is gone, at lower cost.
**Risk.** `generate_structured` already defends coherence and (measured) escapes the loops greedy decoding
falls into, so this may be largely redundant; the new part is explicit abstention/pruning. *Seats: Pharr
(Russian roulette).*

### B3 — Adaptive-stop diffusion for `generate_structure`   [ADAPT-2 → text]   *risk-free*
**What.** The B10 diffusion runs a FIXED annealing schedule. Stop the moment the composed structure has
converged (every slot decodes to a stable, valid vocabulary atom and is unchanged for a step), like the
resonator early-stop.
**Why.** ADAPT-2: stopping on convergence matched fixed-count quality at far lower average cost and never hurt
accuracy. An easy structure reaches the manifold well before the last annealing step.
**How.** After each step, check slot stability + validity; stop, with min/max bounds and the schedule
preserved up to the stop. Measure validity/novelty at average steps vs the fixed schedule.
**Bar.** Matches fixed-step validity AND novelty at lower average steps; bit-identical when the option is off.
**Risk.** Annealing may need its tail for NOVELTY (stopping on first validity could collapse diversity — the
Eno caveat); the stop must be on stability, not mere first-convergence. *Seats: the resonator early-stop
(ADAPT-2), Milanfar/Cranmer (RED & sequential-stopping convergence), Eno (don't amputate novelty).*

### B4 — Low-discrepancy / stratified sampling in generation   [SAMPLE-1 → text]   *likely no-op*
**What.** Use stratified/low-discrepancy draws where generation samples (nucleus top-p, the diffusion's
injected noise) for more even coverage of the candidate distribution.
**Risk.** A categorical token draw does not obviously benefit from low-discrepancy (it is a single
multinomial pick); the only plausible target is the continuous diffusion noise. Likely small / no-op.
*Seats: Pharr (stratified sampling).*

---

## Group C — Image generation

### C1 — Coarse-to-fine splat densification   [SCALE-1 + ADAPT-1 → image]   *the headline image item*
**What.** Fit splats coarse-to-fine: a few WIDE splats for global structure, then progressively add NARROW
splats only where the residual is large — 3DGS's densification, made explicit through the multi-resolution
pyramid, and a principled warm start for `splat_aniso`.
**Why.** SCALE-1 (multi-resolution) + ADAPT-1 (adaptive count) + the 3DGS densify move. Coarse-to-fine
reaches a target at fewer splats on multi-scale content, and — directly addressing `splat_aniso`'s kept
negative (non-convex, depends on the isotropic warm start, more splats don't help monotonically) — a
residual-pyramid warm start should escape the local optimum the current single-scale warm start falls into.
**How.** Build a residual pyramid (SCALE-1); place wide splats at the coarse level, refine with narrow splats
at per-level residual peaks (ADAPT-1 placement); hand the result to `aniso_fit` as the warm start. Measure
reconstruction at equal splat budget vs the current matching pursuit, and final aniso MSE vs the isotropic
warm start.
**Bar.** Better reconstruction at equal splat budget on multi-scale images, and a better (lower-MSE, less
warm-start-dependent) anisotropic fit than today's.
**Risk.** May overlap with `splat_fit`'s existing scale selection on single-scale content; the win is on
genuinely multi-scale fields. *Seats: Drettakis (3D Gaussian Splatting densification) — would pull this to
the front; Quílez/Pharr (multi-resolution).*

### C2 — Phase-domain scene morph   [PHASE-1 → image]   *honest-negative-friendly*
**What.** `morph_scene` slerps in the DCT-coefficient domain. Apply PHASE-1: interpolate the PHASE of the
DCT/FFT coefficients (shortest arc, magnitudes kept) for uniform feature motion where amplitude/slerp blending
eases or ghosts.
**Why.** PHASE-1 measured uniform constant-velocity motion and energy preservation from phase-domain
interpolation; image morphing is the canonical phase-interpolation application (coefficient phase encodes
where features sit).
**Bar.** Smoother, fewer-ghost morphs than the DCT slerp under large change — or an honest negative (the slerp
may already be adequate, and PHASE-1's wrapping negative applies once coefficient phase shifts exceed π).
*Seats: Puckette (phase vocoder), the phase-based frame-interpolation basis (PHASE-1).*

### C3 — Adaptive-stop for the anisotropic splat fit   [ADAPT-2 → image]   *risk-free*
**What.** `splat_aniso` runs a FIXED ~200 Adam steps. Stop when the reconstruction MSE has converged
(relative improvement below a threshold), with min/max bounds — the resonator early-stop, for a gradient fit.
**Why.** ADAPT-2: fixed iteration counts over-compute easy problems; an easy field converges in far fewer
steps at identical final MSE.
**Bar.** Matches the fixed-200-step MSE at lower average steps; bit-identical when off (Macklin's determinism
rule). **Risk.** Modest where 200 steps are genuinely needed. *Seats: Macklin (iterative-solver convergence,
deterministic tie-breaks), the resonator early-stop.*

### C4 — Negative-lobe sharpening of the splat / archive reconstruction   [XDATA-3 → image]   *high upside*
**What.** A splat or archive reconstruction is a sum of smooth Gaussians and is inherently over-smoothed
(`splat_aniso`'s own kept negative says a few Gaussians can't hold high frequency). Post-process the render
with the looping negative-lobe sharpener (XDATA-3, Van Cittert), guarded by the discrepancy principle.
**Why.** XDATA-3 measured that looping negative-lobe sharpening recovers detail an over-smoothed signal lost
and converges where naive unsharp diverges; the splat render is exactly an over-smoothed estimate with the
detail provably absent.
**How.** Run `sharpen_loop` on the render with the discrepancy-principle stop; measure edge sharpness / PSNR
to the original vs the raw render.
**Bar.** Sharper reconstruction (better PSNR / edge metric) than the raw splat render, the discrepancy guard
preventing over-sharpening and splat-artifact ringing.
**Risk.** Sharpening can amplify splat ringing; the guard must hold. *Seats: Milanfar (RED / deblurring) —
would lead the image work here; Ozcan (reconstruction-under-degradation).*

---

## Group D — Creature brain

### D1 — Low-discrepancy exploration   [SAMPLE-1 → creature]
**What.** Replace epsilon-greedy RANDOM exploration with low-discrepancy / quasi-random exploration that
covers the action space (and, where the state is continuous, the state space) evenly, so the agent explores
systematically rather than clumping and revisiting.
**Why.** SAMPLE-1's tighter coverage → more distinct states/actions visited per exploration budget → faster
learning; random exploration clumps and leaves gaps.
**How.** Drive the explore choice from a low-discrepancy sequence over the action set (and state-coverage
targets where applicable); measure coverage-per-step AND reward-vs-episodes on the maze gauntlet vs
epsilon-greedy.
**Bar.** Faster state/action coverage AND faster learning than epsilon-greedy random.
**Risk (the honest caveat).** Over a TINY discrete action set, low-discrepancy ≈ round-robin (small gain); the
real win is STATE-space coverage, which is harder to drive and depends on the state representation. *Seats:
Togelius (game-AI exploration, procedural content) — with this caveat front and centre; Pharr (low-discrepancy
sampling).*

### D2 — Robust reward / value accumulation   [ACCUM-3 → creature]   *proven transfer*
**What.** The brain's `value()` averages over remembered (state→reward) prototypes. Apply ACCUM-3's outlier
clamping (winsorise a reward to k robust-scales from the median before it enters the estimate) and ACCUM-2's
harmonic weighting (for a stationary target), so one anomalous reward can't swing a value.
**Why.** ACCUM-3 measured ~100× lower error under outliers with no loss on clean data; rewards are noisy and a
single outlier shouldn't dominate a value estimate.
**How.** Winsorise + harmonic-weight the per-(state,action) reward aggregation; measure value-estimate error
under injected reward outliers AND policy quality on clean rewards vs the plain average.
**Bar.** More robust value estimates under outlier/noisy rewards, at-least-as-good policy on clean rewards.
**Risk.** Cosine-to-prototype is already bounded; the win is specifically under reward outliers. *Seats:
robust statistics (Huber), Togelius (RL value estimation), the ACCUM-3 result.*

### D3 — MIS-combined decision signals   [MIS-1 → creature]   *partial fit*
**What.** `decide()` combines a value estimate, a safety reflex, and optionally novelty. Combine them by a
balance heuristic weighted by reliability instead of veto + greedy.
**Risk.** The safety reflex is a HARD CONSTRAINT (a veto), not an estimator to blend — MIS fits the
value/novelty blend, not the constraint. Likely only a partial application. *Seats: Pharr (MIS).*

### D4 — Re-anchored, throughput-gated lookahead   [RAY-1/RAY-2 → creature]   *research / new capability*
**What.** `decide()` is reactive (one step). For lookahead, roll out a holographic forward model a few
imagined steps, RE-ANCHORING (cleanup to the experienced-state manifold) each imagined step and
THROUGHPUT-GATING the rollout (stop when the imagined trajectory goes dark) — RAY-1/RAY-2 applied to imagined
rollouts.
**Why.** RAY-2 measured that a chain of transitions without re-anchoring drifts off-manifold and collapses
(12/12 hops with, 1/12 without); a holographic forward model rolled out raw would diverge after a step or two,
and re-anchoring is exactly what keeps an imagined trajectory valid.
**How.** Learn a forward model (next-state as a bind/transition operator), roll it out with per-step cleanup
and the throughput gate; measure imagined-vs-actual rollout fidelity with vs without re-anchoring, and
planning quality.
**Bar.** Re-anchored rollouts stay accurate to a useful horizon where raw rollouts collapse (the RAY-2
contrast in the creature's state space), enabling planning the reactive brain can't do.
**Risk.** The creature has NO forward model today — this is new capability, the largest effort here, and a
genuine research item, not a thin transfer. *Seats: Baker (energy-landscape / fragment search), Adamatzky
(flow search), the RAY-2 result; model-based RL.*

### D5 — Observation denoising   [XDATA-1/2 → creature]   *likely no-op*
**What.** Denoise the creature's noisy observations (downscale / looping denoise) before perception, so the
value estimate is built on a cleaned state.
**Risk.** The high-dimensional perceptual encoder already averages out independent noise (concentration again);
likely a small/no-op gain. *Seats: Milanfar (denoising), Olshausen (sparse coding).*

---

## An initial priority (value over effort) — to be debated by the panel

1. **C3** adaptive-stop aniso fit — risk-free, trivial, certain small win.
2. **B3** adaptive-stop diffusion — risk-free, easy, certain small win.
3. **C4** negative-lobe image sharpening — proven transfer, high upside on over-smooth renders.
4. **D2** robust reward accumulation — proven transfer (ACCUM-3), straight into the value path.
5. **C1** coarse-to-fine densification — the headline image item; also fixes the aniso local-optimum negative.
6. **A3** adaptive encoder kernel placement — the one promising below-stack transfer (CACHE-3).
7. **D1** low-discrepancy exploration — plausible creature win, with the tiny-action-set caveat.
8. **B1** MIS-weighted steered generation — uncertain (calibration), medium effort.
9. **C2** phase-domain scene morph — honest-negative-friendly probe.
10. **D4** re-anchored lookahead / forward model — research, new capability, highest upside.
11. **The likely-no-op probes** — **A1** (LD codebook), **A2** (negative-lobe cleanup), **B2** (gated
    generation), **B4** (LD sampling), **D3** (MIS decision), **D5** (observation denoise): run as honest
    probes, several expected to land as kept negatives confirming the substrate/faculty is already right.

---

*Assembled from the live code. Each item names a SEAT and that field's REAL published methods; no opinion is
attributed beyond what the method itself argues. Negatives are flagged where the prior says the transfer
won't pay — confirming the engine is already right is a result the project keeps.*

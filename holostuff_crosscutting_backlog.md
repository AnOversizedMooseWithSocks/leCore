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

---

# Part II — the DCC reverse-transfer (3-D modeling → the stack)

*A second reverse-transfer sweep, this time from the DCC (3-D modeling) backlog rather than the rendering one.
The brief (Moose): "our entire holographic system is geometry, so if we are adding 3D-modeling things, some of
it should apply elsewhere." The founding thesis run in reverse. The organising realisation: holostuff was never
non-geometric — a hypervector is a point, `bind` a rigid motion, `bundle` a centroid, `cleanup` a nearest-point
projection, `consolidation` the low-D subspace, the capacity cliff concentration-of-measure crowding. The DCC
items add EXPLICIT 3-D geometry on top of that IMPLICIT high-D geometry, so the question is: which explicit-3-D
operation is a special case of a general geometric operation the rest of the stack LACKS (◆ genuinely un-built)
or UNDER-USES (○ already general, an enrichment)? ▽ = an analogy to hold lightly. Labels set by a live code
audit, not enthusiasm; every item is a HYPOTHESIS with a bar and a stated failure mode, nothing prototyped.*

## Group I — Diagonalise the iterated operator: the spectral-limit family

### RT-I1 — `operator_limit` / spectral-iteration   ◆ GENUINELY UN-BUILT   ✓ SHIPPED
*Shipped: holographic_iterate.py. THE KEY INSIGHT: a bind is circular convolution -> DIAGONAL in the Fourier
basis, so the eigenvalues of the operator U are just its rfft spectrum (the eigendecomposition is FREE, no dense
O(n^3) SVD -- "live in the Fourier form"). step_k jumps k binds in ONE eval (raise the transfer to the k-th
power), matching the k-bind rollout to ~1e-15 (MEASURED: 20-step jump == 20 binds to 9e-16); limit is closed-form
(contractive -> 0, no iteration); spectral_profile reads the regime (contractive/marginal/divergent) and the
spectral gap (slow if near-degenerate) BEFORE running. Mind faculties propagator_jump + propagator_spectrum. Kept
negative: only LINEAR operators diagonalise -- the true (nonlinear) resonator needs delay-embedding, so the
"predict a resonator stall" bar is met in its linear cousin (power-iteration convergence from the gap) with the
nonlinearity caveat. Determinism: eigenvector sign pinned (ISA-1 fence). test_holographic_iterate.py (7) + 1
integration. 1156 -> 1164.

**THE DCC REVERSE-TRANSFER THREAD IS COMPLETE (RT-III1, RT-II1, RT-IV1, RT-I1).** Graph-Laplacian denoise ->
nonlinear manifold chart -> steering kernels -> spectral iteration. Four reverse-transfers from the 3D/DCC domain
into the engine, each measured with its kept negative.
- **Cross-connection.** Stam's exact subdivision eval (DCC-B3) is an eigendecomposition of the refinement matrix
  — diagonalise once, evaluate any level / the limit in closed form. The SAME math three faculties need and none
  does: the dynamics propagator (`learn_dynamics`, k-step rollout = k binds; U is a per-frequency Fourier transfer
  → diagonalise → k-step jump is ONE eval, limit is the dominant eigenvector); the diffusion sampler
  (`hopfield.generate`, steady state = leading eigenstructure); the resonator (stalls at wrong fixed points —
  exactly what the eigen-spectrum predicts).
- **Grounded:** `dynamics` uses the Fourier transfer but never a closed-form limit/eigendecomposition; resonator
  has "fixed point" language but no spectral eval. Genuinely un-built.
- **Faculty / seats:** eigendecompose an iterated linear operator, expose k-step + limit closed-form, read
  convergence/stall off the spectrum. Stam (subdivision eval) + Stoudenmire (spectral/low-rank) + Koopman/DMD.
- **Above/below:** BELOW. **Bar:** jump k dynamics steps in one eval matching the k-bind rollout to tolerance;
  predict a resonator non-convergence from the spectrum BEFORE running. **Fail:** only linear operators
  diagonalise; nonlinear needs delay-embedding (dynamics' own negative). Dense eigendecomp is O(n³) (the
  `topology` module already timed out on this) — live in the Fourier/structured form where the spectrum is free,
  not a dense SVD at D=4096. Eigenvector sign/order ties → the determinism fence (fix the sign convention; the
  `spectral` module already does).

## Group II — Flatten the manifold: charts & embeddings

### RT-II1 — nonlinear manifold chart (conformal / ARAP / Tutte)   ◆ GENUINELY UN-BUILT
- **Cross-connection — and the irony.** The least-holostuff item on the backlog, UV unwrapping (DCC-D1:
  LSCM/ARAP/Tutte), is secretly the most general: distortion-minimizing flattening of a curved 2-manifold to a
  low-D chart — the embedding problem the whole stack faces and only solves LINEARLY.
- **Grounded:** the only manifold-to-low-D map is `consolidation` (SVD = LINEAR). A nonlinear, distortion-aware
  chart is not there. (consolidation is the ○ linear member; this is the nonlinear extension.)
- **Faculty / seats:** a faithful low-D chart of ANY holostuff manifold (concept space, creature state/value
  space, codebook similarity graph). Payoffs: (1) interpretability — finally SEE the concept manifold / brain
  state space; Tutte draws the scene/concept graph or the HoloForest directly. (2) a tighter storage coordinate
  than raw SVD where the manifold is curved. Olshausen (representation geometry) + consolidation + Tutte + Lévy/Liu.
- **Above/below:** BOTH. **Bar:** a conformal/ARAP chart of a KNOWN curved manifold (the FPE ring/torus; the
  `energy` module's torus) beats linear SVD on 2-D reconstruction fidelity and visibly separates classes the SVD
  blurs. **Fail:** parameterization assumes a disk-topology chart — a closed manifold needs seams/cuts first (the
  `topology` module finds the genus → where to cut); high curvature makes some distortion unavoidable (LSCM's own
  limit) — honest, not a bug.

## Group III — Signals on graphs and fields

### RT-III1 — graph-Laplacian / spectral-graph filtering   ◆ GENUINELY UN-BUILT   ← panel #1
- **Cross-connection.** Mesh smoothing as graph-signal denoising (DCC-C2) filters a signal (vertex positions) on a
  graph. holostuff is full of graphs — the codebook similarity graph, the HoloForest, the store adjacency, the
  scene/sequence chains.
- **Grounded:** `graph_memory` does cosine k-means clustering, NOT a Laplacian or spectral filtering. Genuinely
  un-built.
- **Faculty / seats:** Taubin (λ|μ, no-shrink) / bilateral / spectral-graph filters that denoise/regularize the
  CODEBOOK over its own k-NN similarity graph (non-local means on the concept graph), low-pass a value function
  over a state graph, smooth an embedding. Milanfar ("A Tour of Modern Image Filtering" links denoisers to graph
  Laplacians) + Taubin.
- **Above/below:** BELOW. **Bar:** denoising a noisy codebook over its k-NN graph beats per-vector denoising on
  recall; Taubin avoids the volume-shrink a naive Laplacian causes. **Fail:** building the graph is O(n²) UNLESS
  you reuse the HoloForest's sublinear k-NN (reuse the index you already have).

### RT-III2 — level-set extraction (the boundary of any field)   ○ ALREADY GENERAL
- **Cross-connection.** Marching cubes (DCC-B4/B6) extracts the surface where an SDF crosses zero. The `field`
  module is already general ("the SDF, the value function, the density/void map, the compass: every one is a
  field"), so this generalises to "extract the threshold-crossing surface of any field": a classifier decision
  boundary, the honesty layer's p=α confidence region, an attractor basin boundary, the memory-void edge. Quílez
  (fields) + the field module.
- **Above/below:** ABOVE. **Bar:** extract the p=α surface of the honesty layer and show it bounds the abstain
  region. **Fail:** marching needs a sampled grid — curse of dimensionality at D=4096; practical only on 2-3-D
  PROJECTIONS of the field (ties to RT-II1: chart first, then march).

### RT-III3 — smooth-min as a general soft-combine   ▽ HOLD LIGHTLY
- Quílez's smooth-minimum (the SDF blend, DCC-B4) is a cousin of the softmax / modern-Hopfield cleanup already in
  the stack. The softmax is already there, so this is a small operator, not a capability — a one-liner, not a sprint.

## Group IV — Kernels, metrics, and noise: the designed-spectrum family

### RT-IV1 — anisotropic / steering kernels (a direction-dependent metric)   ◆ GENUINELY UN-BUILT   ✓ SHIPPED
*Shipped: holographic_fpe.py's VectorFunctionEncoder now takes a PER-AXIS bandwidth (scalar still works,
backward-compatible) -- a diagonal steering kernel (small bw = smooth axis, large bw = sharp). holographic_
steering.py adds steer_bandwidths (fit per-axis bandwidth from data) + kernel_regress; mind faculty
steering_regress. Bar met IN THE RIGHT REGIME: on DENSE directional data (a sharp ridge) the steered kernel beats
the best isotropic RBF ~8% (pools along the flat axis, sharp across the edge). Kept negatives (loud): on SPARSE
data the win is ~1-3% (isotropic stays baseline); on ISOTROPIC data ~0%; "low frequency" is not "low variation";
and the steering ESTIMATE is unreliable on scattered data (gradient polluted by other axes -- needs dense/grid),
a full per-point covariance worse still. test_holographic_steering.py (7) + 1 integration. 1148 -> 1156.
- **Cross-connection.** Anisotropic Gaussian splats (DCC-B5, covariance per splat) ARE steering kernels —
  Milanfar's own steering-kernel regression (Takeda, Farsiu, Milanfar 2007): a local kernel whose shape adapts to
  the data's direction.
- **Grounded:** no anisotropic similarity / steering kernel in the encoders or denoiser. Genuinely un-built.
- **Faculty / seats:** an anisotropic FPE kernel (similarity stretched along informative directions), a local-
  COVARIANCE manifold representation (covariance = local tangent space), steering-kernel denoising. Milanfar
  (steering kernels) + Drettakis (aniso splats).
- **Above/below:** BELOW. **Bar:** an anisotropic encoder kernel beats the isotropic RBF on data with directional
  structure (smooth along one axis, sharp along another). **Fail:** per-point covariances are expensive and
  overfit with few samples (the splat module's own aniso kept negative) — keep isotropic as the honest baseline.

### RT-IV2 — procedural noise = the FPE kernel = the diffusion noise schedule   ○ ALREADY GENERAL
- **Cross-connection.** Procedural noise (Perlin/Worley, DCC-D4) is designed-spectrum randomness, and holostuff
  already connects spectrum↔kernel via Bochner (the FPE kernel IS its phase distribution's characteristic
  function). So procedural noise, the FPE kernel, and the diffusion sampler's noise schedule are ONE object: a
  single "designed-spectrum noise" knob. Quílez (noise) + Puckette (spectra) + the FPE/Bochner thread.
- **Above/below:** BELOW. **Bar:** coloured-noise injection in the B10 diffusion (instead of white) yields
  better-structured samples. **Fail:** may simply tie white noise — keep it on the record if so.

## Group V — Factoring, frames, and constraints

### RT-V1 — instancing = factored (template × parameter) storage   ○ → ◆ for the index
- **Cross-connection.** Instancing (DCC-A3: instance = `bind(geometry, transform)`, scene = `bundle`) is
  weight-sharing / factored storage, which the stack already does twice (the FPE function rep `f = Σ wᵢ encode(pᵢ)`;
  the `RecordEncoder` schema instanced with field values). The ◆ piece is THE REGION QUERY: generalise
  `recall_region` into "index ANY bundle by WHERE its members live" — time, frequency, feature, or position.
  Plate (binding) + Drettakis (a splat scene is a bundle).
- **Above/below:** MIDDLE. **Bar:** region-query a bundle of time-stamped events by time-window with calibrated
  precision (reuse `splat_region`). **Fail:** the capacity cliff — a bundle drowns its members past K; past it,
  fall back to the explicit list (the constructed-vs-stored rule again).

### RT-V2 — skinning = partition-of-unity soft-assignment = a MoE on frames   ○ ALREADY GENERAL
- **Cross-connection.** Skinning (DCC-E2) blends bones' transforms by per-vertex weights — a partition-of-unity
  soft-assignment to local frames, exactly the `moe` module's `GatedMixture` (present). The ◆ enrichment: skinning
  blends ROTATIONS, and rotor / dual-quaternion blending (no candy-wrapper collapse) beats naive averaging — so
  anywhere the MoE's local models are transforms, use rotor blending. The `moe` module + Macklin (clean transform
  blends).
- **Above/below:** MIDDLE. **Bar:** a rotor-blended MoE of local frames beats a linear-blended one on a
  rotation-structured task. **Fail:** only matters where the local models are rotations — niche.

### RT-V3 — constrained projection: generalise IK's constraints upward   ◆ (extends the A4 unification)
- **Cross-connection.** IK (DCC-E3) is already the same "iterate-a-projection" faculty as resonator + denoise +
  dynamics + cloth (A4). What IK ADDS is a CONSTRAINT VOCABULARY (joint limits, reach targets, pole vectors).
  Push it up: constrained GENERATION (projection-onto-the-constraint each diffusion step — "make a scene where X
  holds"), constrained RECALL (nearest stored item satisfying a predicate), constrained PLANNING. Macklin
  (constraint projection) + the generation/recall threads.
- **Above/below:** ABOVE. **Bar:** constrained generation (B10 diffusion + a projection step) yields samples that
  satisfy a hard predicate while staying valid, and beats rejection sampling on cost. **Fail:** non-convex
  constraints aren't projections and can stall (the resonator's wrong-fixed-point problem returns) — keep it as a
  kept negative where it does.

## Group VI — The order question (deepest, held lightly)   ▽

### RT-VI1 — non-commutative binding for order in general
- **Cross-connection.** HRR's `bind` is COMMUTATIVE (order-blind) — which is why sequences/time/causality lean on
  the `permute` workaround. The DCC need for exact non-commutative rotation composition is the same need: the
  `clifford` result is precisely that a non-commutative product captures order where commutative convolution
  provably cannot (its measured 0.66 order-gap). The provocation: is a non-commutative binding the right substrate
  for order IN GENERAL, and is `permute` a patch over a missing primitive? Plate (HRR + its commutativity limit) +
  the Clifford result + the sequence/recurrent thread.
- **Above/below:** BELOW (most speculative). **Held lightly:** Clifford's 2^d blow-up rules it out as a general
  high-D substrate (its own kept negative); the takeaway is CONCEPTUAL ("use a non-commutative op where order is
  load-bearing"), not "replace `bind`". **Bar:** a principled non-commutative bind encodes order with less
  crosstalk than `permute` on a sequence task. **Fail:** any non-commutative substrate likely costs more than one
  FFT — measure the tradeoff before believing the win.

## Part II priority (panel, value over effort, ◆ only)

1. **RT-III1 graph-Laplacian** — cleanest, clearest bar, REUSES the HoloForest for the k-NN graph. Denoise-the-
   codebook is an immediately testable win. **✓ SHIPPED** (holographic_graphsignal.py; Taubin no-shrink + the
   high-noise win over per-vector, low-noise kept negative).
2. **RT-II1 manifold chart** — highest interpretability payoff (see the concept space at last), `topology` guides
   the cuts, upgrades the storage coordinate. Most visible result for the least new theory. **✓ SHIPPED**
   (holographic_chart.py; Isomap beats linear SVD on a curved manifold 5/5 seeds, Laplacian-Eigenmaps secondary).
3. **RT-IV1 steering / anisotropic kernel** — solid, bounded, Milanfar's own method; carries a known overfit
   negative so the bar is honest from the start. **← next.**
4. **RT-I1 operator-limit / spectral-iteration** — the most beautiful unification (subdivision = dynamics =
   diffusion = resonator) but the most research-heavy and O(n³)-haunted; do it in the Fourier/structured form, and
   only after the cheaper three prove the reverse-transfer pays.

The ○ enrichments (RT-III2 level-sets, RT-IV2 designed-spectrum noise, RT-V1 region-query, RT-V2 skinning-as-MoE,
RT-III3 smooth-min) ride along with their DCC items at no extra research cost. RT-VI1 (order) stays a written
provocation until something cheaper forces the issue.

---

# Part III — the VSA ISA backlog: maturing the assembly layer

*A THIRD backlog merged in for unified sequencing (Moose's request). Different genre from Parts I-II (those
are reverse-transfer sweeps; this is an ISA-maturation DEPENDENCY SPINE -- the order IS the deliverable). The
audit's finding: holostuff has ALREADY built a VSA instruction-set architecture -- the kernel (bind/unbind/
bundle/permute/cosine/involution/atom-gen) is the instruction set; HoloMachine is an accumulator-machine
assembler+interpreter (LOAD/BIND/BUNDLE/PERMUTE/CALL/APPLY/IFMATCH/ITERATE/REPEAT/HALT, assemble(), run(), a
function library, program-as-data); StructureRecipe is the bytecode/IR; the resonator is the disassembler.
This is what assembly/ISA history says to do NEXT, in dependency order. Topical successor to the (complete)
holostuff_vm_backlog.md (VM-1..3, PIPE-1, SYN-1, REC-1, GEN-1 all done). Each item: SEAT + real method, a bar,
an anticipated kept negative.*

## The spine (why this order)
Everything an ISA does -- extensions, safe optimization, registers, calling conventions, higher languages -- is
defined RELATIVE TO a written contract of the base instructions' exact semantics. So the contract comes first.
GROUNDED COST OF ITS ABSENCE (verified in live code, June 2026): the determinism/tie-break behaviour is
specified FOUR different ways across modules -- `cleanup` leans on numpy's implicit argmax (ties -> lowest
index, written nowhere; holographic_ai.py `int(sims.argmax())`), `spectral` invented "largest-magnitude
component positive" explicitly citing "the same bit-exact-tie class as the bind_batch bug", `flow` carries its
own `_weighted_laplacian`, and `holographic_chart.py` (RT-II1, just shipped) reinvented the SAME sign rule as a
private `_fix_signs` rather than sharing it. Same bug class, re-litigated four times, with code duplication as
the price. The contract (Tier 0) ends that.

## Tier 0 — The contract (do first; everything is defined against it)

### ISA-1 — write the ISA contract: exact base-instruction semantics, determinism included   [BUILD]   ✓ SHIPPED
*Shipped: ISA.md (per-instruction observable semantics, EXACT/TOL tags, the arch/microarch boundary) +
holographic_determinism.py (`fix_eigvec_signs`, `argmax_tiebreak`) as the one home for the determinism rule;
spectral.sign_fix and chart._fix_signs de-siloed onto it BIT-EXACT (the fourth scattered copy removed, 19
spectral/chart tests unchanged). 1091->1098.*
- **Seat/basis.** Cranmer (reproducible-analysis discipline, RECAST -- a frozen re-runnable contract) + Macklin
  (the bit-exact tie-break lesson is his territory; this also answers his STANDING determinism-audit request).
- **Order.** FIRST. Unblocks the §7 vectorization work SAFELY, prerequisite for the extension discipline
  (ISA-3) and every new instruction (ISA-4/5), and captures the bind_batch lesson permanently.
- **What.** A written spec of each base instruction's exact, frozen, observable semantics -- inputs, outputs,
  normalization, edge cases (NaN, zero vector, ties), and ONE determinism/tie-break rule superseding the four
  scattered ones (argmax ties -> lowest index; eigvec sign -> largest-magnitude component positive; reductions
  in a fixed documented order). The architecture; FFT/BLAS/forest impls are microarchitecture BELOW it. Plus
  the de-silo: ONE shared sign-convention utility that `spectral` and `chart` CITE instead of each reinventing.
- **Why.** x86's durability is that the ISA is a frozen contract while implementations vary underneath. The
  bind_batch bug IS a microarch change (batched BLAS, bit-exact to 1e-12) leaking through an under-specified
  contract. Write the contract and the leak has a name and a test.
- **How.** A THEORY.md-adjacent `ISA.md`: one section per instruction; the determinism rule stated once; the
  microarch/arch boundary drawn explicitly. A small shared determinism utility module the scattered sites adopt.
- **Bar.** Every base op has a written spec with edge cases enumerated; the four scattered tie-break conventions
  are reconciled into one documented rule and the modules updated to CITE it, not reinvent it.
- **Anticipated negative.** Over-specifying freezes incidental float quirks as contract. Spec the OBSERVABLE
  semantics callers depend on (the argmax decision, the unbind exactness), not every last reduction bit no
  caller can observe -- and say which is which.

### ISA-2 — the conformance suite + reference implementations (the contract's teeth)   [BUILD]   ✓ SHIPPED
*Shipped: holographic_reference.py (definitional reference impls -- `ref_bind` a direct O(D^2) convolution etc.,
verified vs the kernel to machine epsilon) + the TOL/EXACT conformance checks (`value_conformant`/
`exact_conformant`/`decision_conformant`) + `run_conformance` exposed as the mind faculty `conformance_report()`.
test_isa_conformance.py: all base ops conform, the convolution-identity golden vectors, and the bind_batch-class
regression -- a value-conformant change that flips a decision is caught BY CONSTRUCTION. 1098->1107.*
- **Seat/basis.** Cranmer (conformance discipline; golden tests as the spec made executable).
- **Order.** SECOND, immediately after ISA-1 -- a contract with no enforcement is just prose.
- **What.** Per instruction, a definitional reference implementation (simplest, slowest, obviously-correct) plus
  golden vectors. ANY implementation (FFT `bind`, `bind_batch`, a future BLAS `bundle`) must match the reference
  to a stated tolerance AND match the tie-break rule EXACTLY (zero tolerance where the tie is observable).
- **Why.** This is what lets §7 go fast without fear: a vectorized op is "conformant" iff it passes the suite.
  §7's "pin the vectorized op to the scalar result" tests become INSTANCES of this suite.
- **How.** `test_isa_conformance.py`: reference impls + golden vectors + the tie-break regression -- including a
  regression test for the bind_batch bug ITSELF (a summation-reordered bind that flips a trajectory must FAIL).
- **Bar.** The suite catches the bind_batch-class bug by construction; all current kernel impls pass; flow's
  duplicated Laplacian can now be de-duplicated BECAUSE the shared version is conformance-pinned.
- **Anticipated negative.** Tolerance is a judgment call. Resolution (from ISA-1): numeric tolerance on the
  CONTINUOUS outputs, EXACT match on the OBSERVABLE decision (the argmax, the tie-break).

## Tier 1 — The extension discipline

### ISA-3 — formalize the ISA-extension framing for the parallel bind modes   [BUILD]   ✓ SHIPPED
*Shipped: ISA_EXTENSIONS.md -- the base instruction set frozen and listed (boundary principle: base = the
kernel almost every faculty uses, so `permute` is base; extension = regime-specific opt-in module), one page per
extension (regime / measured win / cost / conformance), and the earning-its-place proposal template. Regime wins
measured fresh: Clifford exact 3-D rotation (err 1.1e-16), FPE designed kernel (1.0->0.04 vs flat random atoms),
tensor capacity (recall 0.87 vs HRR 0.28 at overload). test_isa_extensions.py pins all three + base-unchanged.
1107->1111. Tiers 0-1 of the ISA spine complete.*
- **Seat/basis.** Stoudenmire (tensor/capacity extension) + Plate (what stays in the minimal base ISA).
- **Order.** THIRD -- an extension is defined RELATIVE TO the base contract (ISA-1).
- **What.** Document Clifford-bind (rotations), tensor-bind (capacity), FPE (spatial/continuous) as named,
  opt-in ISA EXTENSIONS -- the VSA analog of x86 + SSE/AVX/AES-NI. Each gets a one-page spec: its REGIME, the
  MEASURED win over base `bind`, its OWN conformance tests, and the standing rule that the base kernel stays
  minimal. Plus a proposal template ("a new bind mode must earn its place by a measured regime win").
- **Why.** Real ISAs grow as base + extensions, not by bloating the base -- holostuff already does this by
  instinct (the Clifford docstring states the rule). Naming it makes it policy.
- **How.** `ISA_EXTENSIONS.md`, one page per extension; enumerate and freeze the base instruction set.
- **Bar.** Each existing extension has its spec page; a new-extension template with the earning-its-place bar;
  the base kernel explicitly listed and minimal.
- **Anticipated negative.** The base/extension boundary is debatable (is `permute` base or extension?). Pick a
  principle -- base = what (almost) every faculty uses; extension = regime-specific -- and apply it consistently.

## Tier 2 — The machine model (improve HoloMachine, governed by the contract)

### ISA-4 — accumulator -> a small register file   [BUILD]   ✓ SHIPPED
*Shipped: HoloMachine grown from one ACC to named slots R0..R7 with two additive opcodes, STORE r / RECALL r
(backward-compatible -- all 14 prior machine tests pass). Slots held SEPARATELY -> reads are EXACT (cosine 1.000,
bit-for-bit). The bar: a k-instruction intermediate needed again costs a full re-derivation without registers but
one RECALL with them. Kept negative measured: a BUNDLED register file (the disk pattern) has a literal capacity
cliff -- perfect to ~16 slots at dim 1024, then degrades (64 -> ~0.92); holds 64 at dim 4096. Register count is a
capacity question for the bundled rep, which is why the slots are separate. test_isa_registers.py (5) + 1
integration. 1111 -> 1117.*
- **Seat/basis.** Plate (composition / role-addressing) + the machine thread. **Order.** FOURTH -- new
  instructions (LOAD/STORE to a slot) that must conform to the contract; cheapest concrete HoloMachine win.
- **Grounded.** HoloMachine has "one register: the accumulator (ACC)" -- the most primitive design.
- **What.** A handful of named vector slots (registers) beyond ACC, with load/store opcodes, so a program holds
  intermediates without re-deriving them (slots = named vectors with unitary roles; reads exact, cost low).
- **Bar.** A program that re-threads a value through ACC repeatedly is shorter / uses fewer binds with
  registers; register reads are exact.
- **Anticipated negative (lovely + VSA-native).** "Register pressure" is LITERAL: if the register file is a
  bundle, slots share the crosstalk budget, so too many registers degrade readback. Measure how many slots fit
  before recall drops -- register count is a capacity question, not a free choice.

### ISA-5 — a documented calling convention + a permute-stack for recursion   [BUILD]   ✓ SHIPPED
*Shipped: (1) the ABI in ISA.md -- CALL f is an ACC->ACC transform (ACC = arg/return), and registers + the
permute-stack are FRAME-LOCAL so a callee cannot corrupt the caller (every register callee-saved by
construction; measured cosine 1.000). (2) The permute-stack (PUSH/POP opcodes + stack_push/stack_pop) -- push =
permute+bundle, pop = cleanup+inverse-permute; reverse-via-stack runs correctly through the machine. Kept
negative measured: the holographic stack's depth is crosstalk-bounded (safe ~4-8 at dim 1024, ~0.48 by depth 16)
-- same shape as the B8 cliff; exact deep work uses the registers. Backward-compatible (prior machine tests
pass). test_isa_callstack.py (5) + 1 integration. 1117 -> 1123. Tiers 0-2 of the spine complete.*
- **Seat/basis.** Plate (the ABI) + the machine thread. **Order.** FIFTH -- builds on ISA-4; needs the contract
  for the new stack instructions.
- **Grounded.** `CALL` "extracts the named function and runs it on the current ACC" (the implicit convention);
  `permute` exists.
- **What.** Formalize the ACC-threading CALLING CONVENTION (what CALL passes/preserves) and add a STACK using
  `permute` as push (unbind as pop) so HoloMachine programs can recurse and nest calls.
- **Bar.** A small recursive program (e.g. a recursive structure builder) runs correctly via the stack; the
  convention is documented and the function library obeys it.
- **Anticipated negative.** Stack depth is bounded by crosstalk too -- each permute-push rides a bundle, so deep
  recursion hits the capacity cliff (same shape as the B8 iterated-decode cliff). Measure the safe depth.

## Tier 3 — The layers above assembly

### ISA-6 — a macro layer: parameterized recipe/procedure templates   [BUILD]   ✓ SHIPPED
*Shipped: holographic_template.py -- a RecipeTemplate is a StructureRecipe with named HOLES filled at
instantiation; different args give DISTINCT, BIT-EXACT structures (the recipe's exactness carries). Starter
library (pair / record / ordered_pair); mind faculties instantiate_template + template_names. Kept negative --
macro HYGIENE: atoms derive from names, so an un-namespaced internal atom would collide with a same-named caller
atom (capture). Fix: internal atoms namespaced under a reserved "@tmpl:<name>:" prefix; witness cosine
internal-vs-caller ~0 (-0.04) with the discipline, 1.0 without. test_holographic_template.py (8) + 1 integration.
1123 -> 1132.*
- **Seat/basis.** Puckette (Pd/Max -- he built a composition language over real-time primitives) + the recipe/
  procedure thread. **Order.** SIXTH -- assembly before macros; builds on the recipe IR + procedure scaffolding.
- **Grounded.** `procedure_to_recipe`, `learn_recipe_grammar`, `generate_procedure`, `recall_procedure` exist --
  a "procedure" abstraction already halfway to a named subroutine/macro.
- **What.** Parameterized templates -- a recipe/procedure with HOLES filled at instantiation -- plus a small
  named-template library.
- **Bar.** A parameterized template instantiated with different arguments produces the correct distinct
  structures BIT-EXACT (the recipe's exactness carries); a starter library exists.
- **Anticipated negative.** Macro HYGIENE -- a template that captures/rebinds atoms collides with the caller's
  atoms. Needs a fresh-atom discipline for template holes.

### ISA-7 — a higher-level language compiling to the recipe IR   [RESEARCH]   ✓ SHIPPED
*Shipped: holographic_lang.py -- a small declarative STRUCTURE-DESCRIPTION language (S-expressions) that lowers
to StructureRecipe. Surface: a symbol is an atom; (bind a b)/(bundle ...)/(permute a n) lower to recipe ops; the
ISA-6 templates appear as forms ((record name moose)). parse/unparse round-trip; compile_spec -> recipe;
realize_spec -> vector. Mind faculties compile_structure + realize_structure. Bar met: compiles correct + realizes
BIT-EXACT ((bind a b) == bind(atom a, atom b); a (record ...) form == the ISA-6 template directly); surface
round-trips. Kept scope boundary ENFORCED: one domain only -- no variables/control/functions; an unknown form is
a ValueError, not a no-op (test_scope_boundary). test_holographic_lang.py (8) + 1 integration. 1132 -> 1141.
The assembly tower (kernel -> macros -> language) is complete; only ISA-8 (the reversible/quantum frontier)
remains.*
- **Seat/basis.** Puckette (a declarative DSP language over primitives) + Eno (language-as-generative-system) +
  Plate. **Order.** SEVENTH -- top of the tower; depends on ISA-6 + the IR. Research-heavy, so late.
- **Grounded.** The DCC material-node-graph-as-StructureRecipe and the typed-structure unification (program =
  tree = scene = one recipe) are ALREADY early forms of "a higher-level description that lowers to the IR."
- **What.** A small high-level language with a compiler lowering to StructureRecipe. Frame the typed unification
  as the IR this language targets.
- **Bar.** A declarative spec (start with ONE domain -- material graph or scene spec) compiles to a correct
  recipe and realizes bit-exact; round-trips.
- **Anticipated negative.** A general-purpose language is large and easy to over-scope. SCOPE TO ONE DOMAIN
  FIRST; do not build a general language up front. Most speculative -- last for a reason.

## Tier 4 — The reversible/quantum discipline (the frontier)

### ISA-8 — adopt the reversible-computing / error-correction model (cleanup = error correction)   [RESEARCH]   ✓ SHIPPED
*Shipped: holographic_reversible.py + ISA_REVERSIBLE.md. (a) REVERSIBILITY AUDIT (verified): bind/unbind/permute/
involution reversible, bundle/superpose/cleanup lossy -- cleanup IS error correction. (b) AUTO-CLEANUP SCHEDULER
(the measured core): an oracle-free health signal (cosine to nearest atom) triggers cleanup before the cliff;
generalizes the coherence-gate from store-maintenance to program-execution. MEASURED (bursty damage): adaptive
holds fidelity (frac-below-0.9 = 0.000) at 5 cleanups vs the matching fixed cadence's 16 (~1/3). (c) FHRR-as-
diagonal-unitary framing. LOUD NEGATIVE on record: this is an ANALOGY, not physics -- VSA is NOT a quantum
computer; borrow the discipline, not the physics. Mind faculties reversibility_audit + run_with_auto_cleanup.
test_holographic_reversible.py (6) + 1 integration. 1141 -> 1148.*

**THE VSA ISA SPINE IS COMPLETE (ISA-1 .. ISA-8).** Determinism contract -> conformance suite -> extension
discipline -> register file -> calling convention + permute-stack -> macros -> structure language -> reversible/
error-correction model. Eight items, each with its kept negative; the recurring lesson -- superposition buys
composability and pays in a crosstalk cliff -- recurred as the bundled disk, the bundled register file, the
permute-stack depth cliff, and the coherence budget the auto-cleanup scheduler manages.
- **Seat/basis.** Stoudenmire (quantum-inspired/tensor networks; flagged FHRR as "a stop on the road from
  quantum amplitudes to classical hypervectors") + the FHRR/honesty/coherence threads. **Order.** LAST --
  longest-horizon, most conceptual. Its one practical sub-part (the auto-cleanup scheduler) could pull earlier.
- **What.** Make the engine's true nature explicit -- VSA assembly is not x86; it is noisy, bounded, REVERSIBLE,
  structurally a reversible/quantum ISA where `cleanup` = error correction, `capacity` = coherence budget,
  re-anchoring = an error-correction round. Then import: (a) a REVERSIBILITY AUDIT -- classify each instruction
  as exactly invertible (bind/unbind, permute) vs information-destroying (bundle, cleanup); (b) an ERROR-BUDGET
  TRACKER + AUTO-CLEANUP SCHEDULER -- estimate accumulated crosstalk along a program (reusing the capacity
  diagnostic) and insert a `cleanup` BEFORE the cliff (generalizing RAY-1 re-anchoring + the shipped
  coherence-gate from store-maintenance to program-execution); (c) connect FHRR's diagonal-UNITARY structure
  (`bind` = phase rotation) to the quantum-gate framing for principled capacity bounds.
- **Bar.** The auto-cleanup scheduler keeps a long program's output above a fidelity threshold at FEWER cleanups
  than a fixed schedule (echoing the coherence-gate's matched accuracy at ~1/3 the passes); the reversibility
  audit classifies each instruction correctly.
- **Anticipated negative (loud).** The quantum analogy is INSPIRATIONAL, not literal: VSA is not a quantum
  computer (no exponential superposition, no physical entanglement). Borrow the discipline (budget + error-
  correct + reversibility bookkeeping); do NOT overclaim the physics. The practical do-able core is (b) the
  scheduler; (a) and (c) are framing.

## Part III order, at a glance
ISA-1 (contract) -> ISA-2 (conformance suite) -> ISA-3 (extension discipline) -> ISA-4 (register file) ->
ISA-5 (calling convention + permute-stack) -> ISA-6 (macros) -> ISA-7 (HLL, scope to one domain) [research] ->
ISA-8 (reversible/quantum; practical core = auto-cleanup scheduler) [research, frontier]. Tiers 0-1 (ISA-1..3)
are the do-now block; Tiers 2-3 (ISA-4..7) mature the machine and grow the language tower; Tier 4 (ISA-8) is the
conceptual frontier with one practical scheduler inside it.

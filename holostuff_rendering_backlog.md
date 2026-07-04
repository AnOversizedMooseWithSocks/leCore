# holostuff rendering-engine lessons — research / build / validate backlog

*A focused backlog distilled from the recent cross-domain study: 3D rendering engines, image samplers,
reconstruction filters, frame interpolators, upscalers, and production ray tracers (V-Ray's adaptive
sampler, MIS, irradiance caching, Russian roulette, firefly clamping) — plus the "ray is an accumulating
vector" probe and the smooth/sharp architectural move. The organising insight: a renderer is a machine for
spending a fixed budget UNEVENLY (more effort where the signal is hard, less where it's easy) and for
COMBINING cheap estimates, and the engine faces the same problems and already half-implements several of the
answers. The deepest through-line (Group G): denoise / generate / sharpen are NOT image-specific — they are
manifold operations that apply to any data type.*

*Same discipline as every other arc: each item delegates to existing machinery where possible, carries an
honest scope + a measurement bar, keeps its negatives, and follows the close-out ritual + clean zip when it
ships. Items marked **Status: measured** already have a prototype run on the real substrate (the numbers are
quoted) — for those the remaining work is wiring + tests, not discovery.*

---

## Group A — Adaptive compute (spend effort where the signal is hard)

### ADAPT-1 — adaptive-threshold splat count (noise threshold drives the count)   [BUILD]
**What.** Give `splat_field` / `splat_fit` a noise-threshold mode: place a minimum number of splats, then keep
adding at the residual peak until residual-RMS / dynamic-range drops below a threshold, bounded by k_min/k_max.
The user specifies QUALITY; the count adapts to the content — V-Ray's adaptive sampler, on the engine's renderer.
**Why.** A fixed K either wastes splats on simple fields or starves complex ones. The count should be set by the
image, not guessed.
**How.** Wrap the existing greedy MP loop with the residual-RMS stop; thread `noise_thresh` (and k_min/k_max)
through `splat_field`. Keep fixed-K as the default (backward compatible).
**Status: measured.** On smooth fields, one threshold gave 15 splats for a simple field vs 29 for a complex one
at matched quality; fixed K=20 gave 34 dB (simple) vs 27 dB (complex) — same budget over/under-serving. Honest
caveat: the threshold gates the *greedy* residual, so quality is only approximately equalised.
**Bar.** At a fixed noise threshold the count tracks content while PSNR stays within a stated band; on a fixed K
the two fields' PSNRs diverge. Ships with the refit (ADAPT-1 chooses count, refit fixes amplitudes — orthogonal).

### ADAPT-2 — adaptive iteration count for the iterative faculties   [BUILD]
**What.** The same variance-gate, applied to iteration COUNT rather than splat count: `generate_structure`'s
annealing steps, cleanup passes, the resonator's projection iterations. Run a convergence/variance check each
step and stop per-item when converged; spend the saved budget where it isn't.
**Why.** Fixed iteration counts over-compute easy items and under-compute hard ones. The engine already proved
this works once — the SPRT in the recall path IS an adaptive sampler (accumulate evidence, stop at a confidence
bound). The lesson is to recognise that pattern as general.
**How.** A small per-faculty convergence predicate (e.g. step-to-step cosine delta, or the readout margin/entropy)
with min/max bounds; default to the current fixed count for back-compat.
**Scope/risk.** Each faculty needs its own cheap convergence signal; the win is modest where the fixed count is
already small (e.g. 16 anneal steps). Measure per faculty before wiring.
**Bar.** Adaptive count matches fixed-high-count quality at lower AVERAGE cost, per faculty, surviving the variance
harness. (Note: throughput-gating for traversal specifically is RAY-1.)

---

## Group B — Combining & caching (reuse cheap estimates instead of recomputing)

### MIS-1 — balance-heuristic combination of the engine's recall / abstention signals   [BUILD + RESEARCH]
**What.** Multiple Importance Sampling for the engine's several estimators of the same quantity: exact 1-NN
(great on discrete atoms), manifold interpolation / learned cleanup (great on smooth manifolds), HoloForest
(sublinear, approximate), and the multiple abstention signals (forest cross-tree agreement + the calibrated-null
p-value). Combine them by Veach's balance heuristic, weighting each by its per-query reliability (variance-aware
MIS = inverse-variance weights).
**Why.** Right now these are picked or blended ad hoc, and the MIS result is a warning about that: NAIVELY
AVERAGING estimators usually INCREASES variance; balance-heuristic weighting reduces it, each strategy covering
the others' weak regime. This is genuinely new machinery, not a refinement of something we have.
**How.** A `combine_estimators([(estimate, pdf_or_confidence), ...])` helper applying the balance heuristic;
first target = a combined recall confidence (forest agreement + calibrated p) feeding the existing abstain path.
**Scope/risk.** Needs a per-estimator "pdf"/reliability; for the abstention signals that's their calibrated
confidence, which exists. The cleanest demo is engine-grounded combination, not a generic integral.
**Bar.** The combined recall/abstention signal beats every single strategy AND beats naive averaging on
discrimination (AUC / calibrated false-alarm at fixed recall), on a held-out mix of discrete + smooth queries.

### CACHE-1 — gradient-cached manifold decode (irradiance gradients)   [BUILD]
**What.** Store a value AND its local Jacobian at sparse codebook/manifold anchors and interpolate first-order
(Ward's irradiance gradients), with bounded-support (validity-radius) weights so a gradient only extrapolates
where the linear approximation holds. Turns an expensive smooth decode into sparse-evaluate-plus-interpolate.
**Why.** The engine's manifold/decode faculties evaluate a smooth map; first-order caching is a real speedup.
**Status: measured.** Value+gradient at 25 anchors matched value-only at 49 — gradients roughly HALVE the anchor
count. KEPT NEGATIVE: a naive global-weight version actively HURT (distant anchors dump bad long-range
extrapolations into every query) — rediscovering exactly why irradiance caching needs a validity radius +
neighbor clamping. The borrowable unit is the whole package: sparse anchors + stored gradients + locality guard.
**How.** `gradient_cache(anchors, values, jacobians)` + `interp_first_order(q, validity_radius)`; first target a
ScalarEncoder/manifold decode whose map is smooth.
**Bar.** Match a dense decode's accuracy at a stated fraction of the evaluations, with the validity-radius guard
preventing the global-weight failure (both regimes tested).

### CACHE-2 — smooth/sharp two-layer representation (the architectural move)   [RESEARCH → BUILD]
**What.** Split a signal into a smooth layer (cheap basis: broad splats / low-rank / interpolated cache) and a
sharp layer (computed directly: sparse exact residual, or a sharp basis), budgeting each separately — irradiance
caching's core architecture (cache the smooth indirect light, compute the sharp direct light).
**Why.** No single basis is good across a smooth-plus-sharp signal; the right basis per component wins at fixed
budget. This is the deepest borrowable idea — the same smooth+sharp split recurs in the negative-lobe finding,
the SVG (smooth morph + exact edges), and manifold-plus-residual decompose.
**Status: measured (modest).** At a fixed budget the split hit 15.7 dB vs 13.7 (smooth-only) and 4.7 (sparse-only)
— a real but small win WITH SPLATS, because the Gaussian smooth basis and the pixel-exact sharp layer are both
weak. The win grows with a better sharp basis. RESEARCH question: what is the right SHARP basis in the hypervector
domain (the SVG vector edges? a wavelet-like sparse code?) — that is where the split becomes worth shipping.
**How.** An archive/codec mode that stores a smooth code (CACHE-1 / consolidation) + a sparse sharp residual;
report PSNR-at-fixed-bytes vs the WHT-plate and uniform-splat archives.
**Bar.** Beat a single-representation archive at a fixed byte budget on real images, with the sharp layer carrying
the high-frequency detail the smooth layer provably cannot.

### CACHE-3 — adaptive (gradient-driven) cache / codebook placement   [RESEARCH]
**What.** Place cache/codebook anchors DENSER where the cached quantity changes fast (high gradient/curvature),
sparser where it's flat — irradiance caching's adaptive record density, instead of uniform placement.
**Why.** Uniform placement wastes anchors on flat regions and under-resolves the bends. Adaptive caching used ~7x
fewer records for the same quality in the GI literature.
**How.** A placement pass that seeds anchors by a rate-of-change metric (the residual gradient), with neighbor
clamping for discontinuities; ties to ADAPT-1's residual-peak placement (already gradient-ish) and CACHE-1.
**Bar.** Match uniform-placement decode/reconstruction quality at materially fewer anchors on a field with
non-uniform smoothness.

---

## Group C — Reconstruction & sharpening (negative lobes)

### SHARP-1 — Mitchell-Netravali kernel option for the ScalarEncoder   [BUILD]
**What.** Add a Mitchell-Netravali (B=C=1/3) reconstruction kernel to `ScalarEncoder` beside the existing
"sinc" and "rbf" — the encoder IS a reconstruction filter in similarity space, and Mitchell is the graphics
world's settled sharpness-vs-ringing sweet spot (negative lobes sharpen; too-large lobes ring).
**Why.** "sinc" rings and "rbf" (Gaussian, all-positive) blurs; Mitchell's bounded negative lobes are the
balance. Directly testable against decode accuracy.
**How.** One more kernel branch building the similarity profile; default unchanged.
**Status: linked.** The splat joint-refit was MEASURED to synthesise negative lobes (51% of amplitudes negative,
clustered at edges, mean edge-distance 0.007 vs 0.015) — concrete evidence that negative lobes are the sharpening
mechanism a bare Gaussian lacks. SHARP-1 brings that into the encoder kernel directly.
**Bar.** Mitchell beats "rbf" on decode precision near sharp transitions and beats "sinc" on ringing/false-match
robustness, on a continuous-value decode benchmark.

### SHARP-2 — negative-lobe sharpening generalised to non-image data   [RESEARCH]  (see Group G)
**What.** The accumulation-with-negatives that sharpened the Gaussian splats, applied to ANY recalled/decoded
signal: a hypervector, a market trajectory, a structure — iterate a high-pass / negative-lobe correction to
concentrate a smeared estimate, rather than only for images.
**Why.** Sharpening is a manifold operation (counteract the low-pass blur of a smooth basis), not an image trick.
A recalled hypervector that came back smeared (low-throughput) could be sharpened the same way.
**How.** A `sharpen(x, basis)` that applies the negative-lobe correction in the chosen domain; measure on a
non-image signal (a denoised market window, a smeared recall) whether it recovers detail without amplifying noise.
**Bar.** Recover real high-frequency structure on a non-image signal beyond what the smooth basis alone gives,
with an honest noise-amplification check (the ringing limit).

---

## Group D — Accumulation & sampling

### ACCUM-1 — jittered sub-pixel splat accumulation (the right supersampling)   [RESEARCH → BUILD]
**What.** TAA/DLSS's lesson done correctly: instead of supersampling a FIXED splat set (a measured no-op — a
Gaussian sum is band-limited, nothing to anti-alias), JITTER the fit at sub-pixel offsets across passes (Halton /
golden-ratio offsets) and ACCUMULATE, letting splats land at sub-pixel edge positions.
**Why.** It's the honest, correct version of the original supersampling instinct, and the TAA literature says it
works (jitter + accumulate → supersampling quality). Open question: does it sharpen PAST the joint refit?
**How.** Fit on k sub-pixel-shifted grids, accumulate the splat sets (or the renders) with adaptive weights;
compare to the refit-only result at equal splat budget.
**Scope/risk.** May not beat the refit (the refit already uses the pixel-aligned splats optimally); honest negative
is a fine outcome. Sub-pixel placement is the thing to measure.
**Bar.** Sharper edges than refit-only at equal budget, or a kept negative explaining why pixel-aligned + refit is
already enough.

### ACCUM-2 — adaptive harmonic-weight accumulation   [BUILD]
**What.** Where the engine accumulates/averages (consolidation, HoloForest vote-averaging, any iterate-and-average),
use adaptive harmonic-series weights (1/n) rather than a fixed blend — TAA's finding that a fixed alpha caps quality.
**Why.** Harmonic weights give even weight to all samples seen so far → faster, more stable convergence than a fixed
exponential blend. Cheap, general.
**How.** Swap fixed-alpha blends for a running 1/n (or Robbins-Monro) schedule where applicable.
**Bar.** Faster/lower-variance convergence than the fixed blend on a real accumulation (e.g. consolidation over a
growing store), with no regression.

### ACCUM-3 — firefly / outlier clamping for robust accumulation   [BUILD]
**What.** V-Ray's adaptivity clamp / TAA history rectification, as robust accumulation: when consolidation or the
forest averages votes, clamp outlier contributions (by neighbourhood statistics) so one bad estimate can't dominate.
**Why.** A single firefly (outlier recall/vote) skews an average; clamping is the standard robust fix.
**How.** A clamp on per-sample contribution relative to the running local mean/variance, in the averaging paths.
**Bar.** Improved robustness (lower error under injected outliers) on a recall/consolidation average, with no loss
on clean data.

### SAMPLE-1 — low-discrepancy (golden-ratio R2 / Halton) sampler utility   [BUILD]
**What.** A tiny deterministic low-discrepancy sampler (Roberts' R2 / golden-ratio, ~one line of NumPy) wired in
wherever the engine samples for COVERAGE rather than for a specific value: generation seeds, codebook/anchor
placement, the jitter offsets in ACCUM-1.
**Why.** Random sampling clumps (noise), grids alias; low-discrepancy is the compromise.
**Status: measured.** R2 gave 28% tighter coverage than random (dispersion 0.1645 vs 0.2295) for 64 points.
**How.** `low_discrepancy(n, d, seed)` returning an R2 (or Halton) point set; offer it as the default sampler for
coverage-oriented call sites, keeping `default_rng` where independence is actually wanted.
**Bar.** Measurably better coverage (lower dispersion / discrepancy) at the call sites that want coverage, with a
downstream win where it feeds something (e.g. generation-seed diversity, anchor placement for CACHE-1/3).

---

## Group E — Traversal as ray accumulation (the V-Ray path, in holographic space)

### RAY-1 — throughput-gated traversal / Russian roulette   [BUILD]
**What.** Carry a cheap running "throughput" (cleanup confidence / recoverable coherence) alongside any holographic
traversal — recursive composition, the resonator, multi-hop associative recall — and terminate when it drops below
threshold. Unbiased early-out that also tells you when a recall chain has degraded into noise.
**Why.** In the FFT/phasor domain a bind is a multiplicative throughput step, so a bind-chain is a RAY whose
recoverable signal attenuates; Russian roulette transfers directly.
**Status: measured.** Traversing a linked list in superposition, throughput decayed multiplicatively (0.376 →
0.158 → 0.071 → 0.026 …) and the cheap cleanup confidence tracked the true throughput almost exactly while signal
remained — so gating on it stops right when the ray goes dark, WITHOUT ground truth.
**How.** A `traverse(..., throughput_floor=…)` helper (or a mixin) that monitors confidence per step and returns
(result, stopped_at, reason); applies to the recursive/resonator/recall paths.
**Bar.** On a traversal of varying difficulty, throughput-gating matches full-depth useful output at lower average
cost and abstains exactly when recall has decayed to noise (validated against ground truth on a benchmark).

### RAY-2 — re-anchoring discipline between bounces   [VALIDATE → BUILD]
**What.** Re-project intermediate states onto the manifold (cleanup) at each step of a deep composition/traversal —
the path-traced form of "a shared kernel is not a shared manifold." Audit the engine's deep-composition faculties to
ensure they re-anchor between steps; add it where they don't.
**Why.** Without re-anchoring the accumulated state drifts off-manifold and the recoverable signal collapses.
**Status: measured.** Re-anchoring each hop held throughput flat and got 6/6 hops correct; the raw chain died in 2.
This is rendering's next-event estimation (connect to a known anchor each bounce instead of hoping the path stays
valid).
**How.** Audit `compose_nested` / fractal / resonator / multi-hop paths; insert a cleanup-per-step where missing,
behind a measured bar (cleanup is not free, so only where drift is real).
**Bar.** Deep traversals stay accurate to a stated depth WITH re-anchoring and demonstrably fail without it, at a
justified per-step cleanup cost.

### RAY-3 — directed structure (direction role) for sequences / graphs in superposition   [BUILD]
**What.** Encode sequence/graph links with a DIRECTION role (a fixed permutation) so unbinding by a node yields only
its successor, not both neighbours — `bind(x_i, perm(x_{i+1}))`, traversed by `perm⁻¹ ∘ unbind`.
**Why.** Discovered the hard way: a bundle of `bind(x_i, x_{i+1})` is UNDIRECTED — unbinding returns predecessor and
successor at equal strength, so traversal is ambiguous. A permutation breaks the symmetry (the backward term becomes
noise).
**Status: measured.** Directed chain traversed cleanly (6/6) where the undirected one was ambiguous.
**How.** A directed-link encode/decode in the sequence/graph faculties (a fixed permutation as the "next" role);
back-compatible additive option.
**Bar.** Reliable forward traversal of a stored sequence/graph (vs the undirected baseline's ambiguity), at the
expected capacity.

---

## Group F — Interpolation & multi-resolution

### PHASE-1 — FHRR phase-domain morph (graceful interpolation)   [RESEARCH]
**What.** Morph/interpolate in the FHRR PHASE domain (phase shift = motion), not the amplitude/spatial domain —
phase-based frame interpolation's trick, which "fails gently" where flow/spatial methods distort. FHRR is already
the engine's phase domain.
**Why.** Phase-domain interpolation degrades gracefully under large changes; the current morph interpolates whole
hypervectors (amplitude-domain). Worth testing whether phase-domain morphs are smoother/more robust.
**How.** Interpolate the FHRR phases between two states and resynthesise; compare to the current hypervector morph
on a scene/sequence with large change.
**Bar.** Smoother, fewer-artifact interpolations than the amplitude-domain morph under large change, on a measured
example (or an honest negative).

### PHASE-2 — backward/invertible warping note (unbind avoids holes)   [VALIDATE / conceptual]
**What.** Record + validate that the engine's unbind is an invertible BACKWARD map, so it sidesteps the holes/overlaps
of forward warping (splatting) by construction — each target finds its source.
**Why.** Frame interpolation moved from forward warping (holes) to backward warping; the engine gets the backward
form for free. Worth stating where it matters (e.g. preferring unbind-based recovery over splat-forward where both
are options).
**Bar.** A documented note + a small demo confirming unbind-based recovery has no hole/overlap artifact where a
forward-splat analogue would. (Low effort; mostly conceptual.)

### SCALE-1 — explicit coarse-to-fine strategy / mipmap-style multi-resolution   [RESEARCH]
**What.** Make coarse-to-fine an explicit STRATEGY: recall/decode/denoise coarse first, refine only where the
residual is large (flow pyramids / mipmaps / 3DGS densification). A multi-resolution archive (store at several
resolutions, pick per query) is the mipmap analogue.
**Why.** The engine already leans this way (recursive/fractal, HoloForest's coarse descent, consolidation's
low-rank-first), but implicitly. Making it explicit is what 3DGS densification and flow pyramids do deliberately.
**How.** A coarse→fine wrapper that refines on the residual; a multi-resolution archive mode.
**Bar.** Equal quality at lower cost (or higher quality at equal cost) vs single-resolution, on recall/reconstruction,
by refining only where needed.

---

## Group G — Cross-data-type generalisation (the through-line)

*The point that ties the whole arc together: denoising, generation, and sharpening are MANIFOLD operations, not
image operations. If we can denoise/generate/sharpen an image more accurately, the SAME mechanisms apply to any data
type the engine holds — hypervectors, market windows, scalar fields, sequences, structures. Three concrete mechanisms,
each already proven in one domain, to be generalised and validated across data types.*

### XDATA-1 — denoise-by-downscale as a general pattern-finder   [RESEARCH]
**What.** "Patterns can be found by downscaling to eliminate noise": project ANY data to a coarse / low-rank /
low-frequency representation, where independent noise averages out and the structure survives — then read the pattern
there. The image version is downsampling; the engine version is consolidation (low-rank SVD) and the trajectory
denoiser (SSA/Cadzow) — generalise the *principle* and apply it deliberately across data types.
**Why.** Downscaling = low-pass = noise removal = pattern reveal, and it is data-type-agnostic. The engine already has
the pieces (consolidation, `_denoise_signal`/trajectory_denoise); the move is to treat "downscale to find the pattern"
as a first-class, general faculty and point it at non-image data.
**How.** A `find_pattern_by_downscale(data, level)` that picks the coarse representation by data topology (low-rank for
correlated vectors, low-frequency for signals, coarse-grid for fields) and reports the structure found at that scale;
validate on a noisy non-image dataset where the pattern is invisible at full resolution.
**Bar.** Recover a known pattern from a noisy NON-image dataset that is undetectable at full resolution, by downscaling
— and show the recovered pattern is the signal, not an artefact (honest control on pure noise: nothing found).

### XDATA-2 — looping denoise (diffusion) for arbitrary data manifolds   [RESEARCH → BUILD]
**What.** "A looping denoising process": iterate a denoiser/cleanup and it walks onto the manifold (denoising) or,
from noise, generates a sample (generation) — the cleanup-attractor / diffusion the engine already runs for codebooks
(`generate_structure`, `hopfield.generate`). Generalise it to denoise/generate over data manifolds that are NOT the
discrete codebook (market windows, learned subspaces, composed structures).
**Why.** Generation and denoising are the same operation in different regimes (the addendum's B10 result), and it is
manifold-agnostic. The open frontier (already flagged as VG-2 / Eno's request): looping over a COMPOSED or LEARNED
manifold, not the bare codebook.
**How.** Point the looping-denoise at a non-codebook manifold (a consolidation subspace fit to real data, or a composed
structure space); measure denoise quality and generation novelty-vs-validity there. Couples to VG-2 (the multi-noise
denoiser needed for genuinely non-convex learned manifolds) and XDATA-1 (downscale defines the manifold to loop on).
**Bar.** Iterated denoise improves a noisy NON-image signal toward its manifold (idempotent settling), and from-noise
generation produces novel-but-valid samples on a composed/learned manifold — beating both the bare codebook and simple
interpolation where interpolation provably leaves the manifold.

### XDATA-3 — looping accumulation/negative sharpening for arbitrary signals   [RESEARCH]
**What.** "A looping accumulation/negative process (like how we sharpened gaussian images)": iterate the
accumulation-with-negative-lobes that sharpened the splats, applied to any smeared estimate — concentrate a
low-throughput recall, a blurred market signal, an over-smoothed structure — by repeatedly adding a negative-lobe /
high-pass correction. The general partner to SHARP-2.
**Why.** Sharpening counteracts the low-pass blur of a smooth basis, and that is data-type-agnostic. A recalled
hypervector that came back smeared, or an over-consolidated (rank-truncated) signal, can be re-sharpened the same way an
under-reconstructed image edge was.
**How.** A `sharpen_loop(x, basis, iters)` applying the negative-lobe/high-pass correction repeatedly with a stability
guard; measure recovered detail vs the ringing/noise-amplification limit on a non-image signal.
**Bar.** Iterated sharpening recovers real detail on a non-image signal beyond the smooth basis, converging (not
diverging into ringing), with an explicit kept negative on where it over-sharpens.

---

## Suggested sequencing (value over effort; "measured" = prototype already in hand)

1. **SAMPLE-1** — trivial, broad, already measured (28% tighter coverage). Cheapest win; also unblocks ACCUM-1's jitter.
2. **RAY-1** — measured; general and on-mission (the throughput gate touches recursion, resonator, and recall).
3. **ADAPT-1** — measured; ships alongside the existing splat refit (orthogonal: count vs amplitudes).
4. **SHARP-1** — concrete; the encoder is literally a reconstruction filter, and the negative-lobe mechanism is
   already evidenced.
5. **RAY-3** — measured; a clean, real requirement for sequence/graph encoding.
6. **MIS-1** — the highest-value NEW capability (principled estimator combination), a bit more design.
7. **CACHE-1** — measured, but carries the validity-radius machinery as baggage; worth it for decode speedups.
8. **ACCUM-2 / ACCUM-3** — cheap robustness/convergence improvements to the averaging paths.
9. **Group G (XDATA-1/2/3)** — the through-line; research-heavy but the conceptual payoff (denoise/generate/sharpen on
   any data) is the largest. XDATA-1 is the natural entry (defines the manifold the others loop on).
10. **The rest** — ADAPT-2, CACHE-2/3, ACCUM-1, RAY-2, PHASE-1/2, SCALE-1 — research/validate as the above land.

11. **Group H — light-transport lessons (GI / AO / render channels / SSS / caustics)** — a new horizon added to
    investigate AFTER the current list lands. Hypotheses to probe, not yet measured (below).

---

## Group H — Light-transport phenomena to mine for lessons (INVESTIGATE — added after the current list)

*Each is a rendering phenomenon with a genuine VSA/HRR analog worth probing, framed as a HYPOTHESIS to
prototype-and-measure, not a measured result. The discipline is the same as the rest of the backlog: ground in
the real rendering method, find the honest primitive it maps to, prototype the load-bearing claim, keep the
negatives. Several interlock — GI is the umbrella, AO/SSS/caustics are facets of it, AOVs are how you read them
apart.*

### GI-1 — Global illumination (multi-bounce transport as a fixed-point accumulation)   [INVESTIGATE]
**What (rendering).** The rendering equation L = Le + ∫ fr·L·cos — radiance is emitted light plus the integral of
incoming (bounced) light, solved to many bounces (path tracing, photon mapping, radiosity, final gather, instant
radiosity / virtual point lights, light-propagation volumes). The property every GI method exploits: the
INDIRECT (diffuse) field is low-frequency and smooth, which is exactly why sparse caching + interpolation works.
**The holostuff hook (hypothesis).** GI is a FIXED-POINT of a recursive accumulation — the engine already has the
pieces: RAY-1's throughput-gated traversal IS bounce accumulation with re-anchoring (the Russian-roulette stop is
the path-tracer's), and the resonator's alternating projection is a fixed-point solver. The indirect field's
smoothness = a LOW-RANK manifold (consolidation / CACHE-1). Conjecture: a holographic "indirect memory" that
accumulates multi-hop contributions and reads them back smoothly, with the indirect part living in a low-rank
subspace (cacheable, à la irradiance caching) and the direct part kept sharp (the smooth/sharp two-layer split,
CACHE-2). **To probe.** Does a multi-bounce accumulation over a holographic structure converge to a fixed point
whose indirect component is low-rank (consolidation-compressible) and sparsely cacheable + interpolable?

### AO-1 — Ambient occlusion (local crowding / interference as occlusion)   [INVESTIGATE]
**What (rendering).** How much ambient light reaches a point given only LOCAL geometry — how enclosed it is.
SSAO estimates it from local depth/normals; it is the cheap crevice-darkening / contact-shadow term. The lesson:
a purely local measure of how much a point's neighbourhood is blocked.
**The holostuff hook (hypothesis).** AO is a local OCCUPANCY/INTERFERENCE measure. The VSA analog is CROSSTALK as
occlusion: how much a stored item's recall is masked by its neighbours — a heavily-crosstalked atom is
"occluded". This rhymes with RecallNull's noise floor (local occupancy) and the forest's cross-tree agreement
(local density). Conjecture: a per-item "occlusion" diagnostic = local interference/crowding, flagging which
atoms sit in dense neighbourhoods (poorly recallable) vs isolated (clean), a capacity/placement signal that could
drive adaptive codebook spacing (CACHE-3). **To probe.** Does a local-crowding measure predict an atom's recall
failure, and does re-spacing or down-weighting crowded regions improve recall — occlusion as a placement guide?

### AOV-1 — Render channels / arbitrary output variables (separable channel decode + edit)   [INVESTIGATE]
**What (rendering).** A renderer outputs not one image but many named channels — diffuse, specular, direct,
indirect, albedo, normal, depth, object-ID — that composite to the beauty pass and are editable independently
(denoise the albedo, regrade the specular, key on object-ID). The lesson: decompose output into named,
linearly-separable components, and edit in channel space.
**The holostuff hook (hypothesis).** This is role-filler binding / factorization EXACTLY: a scene as a bundle of
role-bound channels (diffuse_role ⊛ diffuse + specular_role ⊛ specular + …), each read back by unbinding its
role — which the engine already does (typed structures, decompose_structure). The mining target is the AOV
EDITING move: modify one channel and re-composite (compositing = bundling, a linear op) without disturbing the
others, and characterise which decompositions are cleanly linearly-separable in the bundle vs which interfere.
**To probe.** Store a multi-channel structure; verify each channel unbinds cleanly; edit one channel, re-bundle,
and measure cross-channel leakage (does regrading specular disturb diffuse?). The most directly-mappable of the
five — the engine's bind *is* AOVs.

### SSS-1 — Subsurface scattering (diffusion transport with a profile kernel)   [INVESTIGATE]
**What (rendering).** Light enters a translucent medium, scatters internally, exits elsewhere (skin, marble,
wax). Modelled by a diffusion approximation — the dipole / BSSRDF (Jensen et al.), a multi-exponential diffusion
PROFILE; light spreads by a diffusion Green's function.
**The holostuff hook (hypothesis).** SSS is DIFFUSION with a specific spatial kernel — exactly the smoothing /
heat-kernel / denoising operations on the manifold (the Milanfar denoising thread). The diffusion profile is a
kernel applied by iteration/convolution. Conjecture: model information "bleeding" through a representation as
diffusion — iterated bundling/smoothing with a tuned multi-exponential profile, or a learned diffusion kernel on
the manifold — and connect it to consolidation (the low-rank part IS the diffused/smooth component). The lesson
worth mining is the dual of CACHE's sharp decode: when do you WANT information to diffuse (graceful
generalisation) vs stay sharp (exact recall)? **To probe.** Apply an iterated multi-exponential diffusion to a
stored field; does a diffusion-profile smoothing generalise/denoise better than a single-scale Gaussian blur on
the engine's manifolds, and at what point does it over-spread (kept negative)?

### CAUS-1 — Caustics (focusing / measure-zero exact-transport search)   [INVESTIGATE]
**What (rendering).** Light focused through refractive/reflective surfaces into bright concentrated patterns
(light through water, a lens). Hard because the contributing paths are SPECULAR — a measure-zero manifold that
random sampling misses. Methods: photon mapping (the canonical caustics solver) and Manifold Exploration (Jakob &
Marschner) — a manifold-walk that finds valid specular path chains.
**The holostuff hook (hypothesis).** Caustics are CONCENTRATION/FOCUSING — the dual of SSS's diffusion: a broad
source focused to a sharp pattern by a precise deterministic transform. In VSA, focusing is a sharp bind/unbind
chain or the resonator finding an EXACT factorisation (concentrating a superposition onto a few sharp peaks). The
sampling difficulty (measure-zero specular manifold) rhymes with the resonator's combinatorial search for the
exact factor combination, and Jakob's Manifold Exploration (walking a constraint manifold to a valid path) rhymes
with the engine's manifold/resonator walks and the directed-graph traversal (RAY-3). Conjecture: caustic focusing
~ the sharpening/resonator operation that concentrates diffuse evidence onto exact atoms; manifold-exploration ~
a guided walk on a constraint manifold for an exact-transport solution. **To probe.** Can a resonator/sharpening
"focus" a diffuse superposition onto exact factors (the caustic), and can a manifold-walk find exact-transport
paths a naive search misses?

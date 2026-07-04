# holostuff Panel Review — the rendering-engine lessons backlog

*The panel was shown `holostuff_rendering_backlog.md` (21 items, 7 groups) and asked two things: what does
your field GAIN from these additions, and in what ORDER should they be built. As always, every interest below
is attributed to a SEAT and that field's REAL published methods — no opinion is put in anyone's mouth beyond
what their own work would call for. The backlog is graphics-dense, so the rendering seats carry the most here;
that is correct weighting for this list, not a thumb on the scale. Where seats genuinely PULL in different
directions on order, that tension is stated rather than smoothed over.*

---

## Seat by seat — what each field gains (tied to its methods)

**Matt Pharr — 3D / raytracing (PBRT: reconstruction filters, stratified & adaptive sampling, MIS, irradiance
caching, Russian roulette, mipmapping).** Touches nearly the whole list — SAMPLE-1, SHARP-1, ADAPT-1, MIS-1,
CACHE-1/3, RAY-1, ACCUM-1/3, SCALE-1. The PBRT chapters these lessons came from, now running on a
content-addressable hypervector substrate. The specific gain he'd flag: the engine's BVH (HoloForest) finally
gets the adaptive-sampling and caching layer a renderer expects to sit on top of an acceleration structure.

**Peyman Milanfar — denoising / inverse problems (RED, Plug-and-Play, NLM/BM3D; "a denoiser is a map of the
signal manifold").** The seat behind the through-line. Group G (XDATA-1/2/3) IS his thesis made general:
denoise/generate/sharpen are one manifold operation, data-type-agnostic. Also CACHE-1/2 (manifold caching and
the smooth/sharp split), SHARP-2, and RAY-2 (re-projection = a denoise step). Gain: his central claim realized
as a toolkit that solves inverse problems on any data, not just images.

**Kyle Cranmer — particle physics (likelihood-free inference, look-elsewhere, calibrated detectors).** The
engine's epistemics are his. MIS-1 (principled estimator combination — averaging raises variance, balance
heuristic lowers it), RAY-1 (unbiased early termination), ADAPT-2 (variance-gated stopping), ACCUM-3 (robust
estimation under outliers). Gain: an unbiased, calibrated way to combine and stop — his discipline, wired in.

**George Drettakis — Gaussian splatting (3DGS, differentiable fit, adaptive density control).** ADAPT-1 is
*literally* 3DGS densification (count set by image complexity); ACCUM-1 (jittered sub-pixel refinement);
CACHE-2 (splats as the smooth basis). Gain: the splat archive gets the adaptive-density and sub-pixel
machinery that makes real 3DGS work.

**Tony Plate — HRR / VSA (the foundation; capacity mathematics).** RAY-3 is foundational HRR — directed
sequences via permutation, fixing the ambiguity a bare `bind(x_i,x_{i+1})` bundle has. Also RAY-1 (throughput
= his capacity cliff, made into a live readout) and RAY-2 (cleanup between steps). Gain: directed structure
done correctly, and capacity as something the engine watches in flight.

**Bruno Olshausen — neuroscience (sparse coding, resonator networks, modern Hopfield).** RAY-1 (when has the
resonator's iterative peeling converged — a throughput stop), RAY-2 (cleanup re-anchoring between resonator
steps), XDATA-2 (modern-Hopfield attractor dynamics ARE the looping denoise). Gain: a resonator that knows
when it's done and never drifts off-manifold.

**Miles Macklin — PBD/XPBD (iterative constraint projection, bit-exact tie-breaks).** RAY-2 is his method by
another name — re-anchoring each bounce is projecting onto constraints each iteration. Also ACCUM-3 (outlier
clamping), ADAPT-2 (solver iteration count), ACCUM-1 (the sub-pixel precision and determinism he lives on).
Gain: the projection-each-step discipline formalized, with the tie-break/clamp robustness he'd insist on.

**Miller Puckette — audio (the phase vocoder, Pd, FFT analysis/resynthesis).** PHASE-1 is the phase vocoder:
interpolate in the FHRR phase domain, which fails gently where amplitude-domain morphs distort. Also SHARP-1/2
(filter design, high-pass). Gain: phase-domain morphing on his home turf, and reconstruction-kernel choice.

**Jos Stam — smoke/water (Stable Fluids, the FFT-on-a-torus solver).** PHASE-1 (phase-domain interpolation —
his FFT-on-a-torus IS the engine's bind), PHASE-2 (advection-as-unbind = invertible backward warping, no
holes), RAY-1 (transport with attenuation). Gain: a graceful phase-domain morph and the invertible-warp result.

**Miles Stoudenmire — tensor networks (MPS/DMRG, low-rank truncation, renormalization).** XDATA-1 is
coarse-graining: downscale / low-rank truncation = the decorrelating step that reveals structure. Also SCALE-1
(multi-resolution = renormalization group) and CACHE-2 (low-rank smooth + residual). Gain: RG-style
coarse-graining as a general pattern-finder in the engine.

**Jarek Duda — file compression (ANS, rate-distortion, KLT water-filling).** CACHE-2 (the right basis per
component is a rate-distortion question), XDATA-1 (downscale/low-rank = the KLT decorrelating transform),
SAMPLE-1 (low-discrepancy ties to coverage/coding). Gain: a smooth/sharp codec framed as a rate-distortion
frontier.

**Aydogan Ozcan — medical imaging (lensless holographic microscopy, reconstruction under degradation).**
CACHE-1 (reconstruct a smooth field from sparse samples — his inverse problem), CACHE-2 (smooth/sharp
decomposition for reconstruction). Gain: reconstruction-from-sparse with gradients, on the archive.

**Iñigo Quílez — demoscene (SDF raymarching, procedural generation, domain repetition).** SAMPLE-1
(deterministic low-discrepancy seeds for procedural generation), SCALE-1 (LOD / multi-resolution — the SDF
art), PHASE-1 (morphing). Gain: better-distributed deterministic seeds and explicit level-of-detail.

**Julian Togelius — game AI (PCG, explainable agents).** ADAPT-2 (an agent that thinks more when uncertain),
RAY-1 (the agent abstains when its state-recall has gone to noise). Gain: agents that spend compute by
difficulty and know when they don't know.

**Jill Tarter — radio astronomy (matched filtering, RFI rejection, the shuffled-null).** ADAPT-2 and RAY-1
(decide as fast as the evidence allows — sequential detection with an honest stop), XDATA-1 (integrate noise
down to reveal a faint signal). Gain: a detector that stops at the right moment and a downscale-to-find-faint
primitive.

**Andrew Siemion — SETI (ML petabyte search, FDR, "flag anything that isn't noise").** MIS-1 (combine triage
signals principally), ADAPT-2 (allocate scan effort), XDATA-1 (find non-Gaussian structure by coarse-graining).
Gain: combined, calibrated triage with effort spent where it's hard.

**David Baker — protein folding (Rosetta fragment assembly, rugged energy landscapes).** CACHE-3 (place
samples where the landscape changes fast — adaptive density on a rugged surface), XDATA-2 (looping descent on
an energy landscape = the diffusion loop). Gain: landscape search that concentrates effort at the bends, and a
generative descent.

**Andrew Adamatzky — Physarum / unconventional computing (flow-conductance, network design).** RAY-1 (his
own dynamics — tubes thicken with flux, thin ones die — is throughput-gated path pruning), RAY-2
(re-anchoring). Gain: a principled prune for dead paths. (His heavier items live in the flow/network backlog.)

**Brian Eno — generative art (generative systems, Oblique Strategies, the reframe).** XDATA-2 (generation is a
process you seed and let run — diffusion from noise), XDATA-3 ("what counts as noise is a choice of which
manifold to keep" — sharpening as a semantic act), SHARP-2. Gain: the generative-denoising loop generalized —
and the reminder that his "honour thy error as a hidden intention" is exactly the kept-negatives discipline the
whole backlog runs on.

---

## Where the panel's methods converge — the build order

The order falls out of three things the seats agree on: do the cheap, already-measured wins first; put the
items the *most fields* depend on early; and respect the dependencies. It lands close to the backlog's own
value/effort ranking, now with the seats behind each placement.

1. **SAMPLE-1 — low-discrepancy sampler.** *Pharr, Quílez, Cranmer, Tarter/Siemion, Duda.* Foundational
   coverage everything downstream that samples sits on (Pharr's sampling chapter, Quílez's deterministic
   procedural seeds, the detection seats' quasi-Monte-Carlo); cheap; already measured (28% tighter); unblocks
   ACCUM-1's jitter. The natural first to re-establish the close-out rhythm.

2. **RAY-1 — throughput-gated traversal / Russian roulette.** *Pharr, Cranmer, Olshausen, Plate, Stam,
   Tarter/Siemion, Togelius, Adamatzky.* The broadest convergence on the whole list: Russian roulette (Pharr) =
   unbiased stopping (Cranmer) = resonator convergence (Olshausen) = the capacity cliff as a live readout
   (Plate) = the detection decision (Tarter/Siemion) = Physarum's dying tubes (Adamatzky). Measured, general,
   touches the most faculties.

3. **ADAPT-1 — adaptive splat count.** *Drettakis, Pharr, Cranmer.* The renderer's adaptive sampler is
   3DGS densification (Drettakis) and PBRT adaptive sampling (Pharr). Measured; ships alongside the existing
   refit (count vs amplitudes — orthogonal).

4. **SHARP-1 — Mitchell-Netravali kernel for the ScalarEncoder.** *Pharr, Puckette, Milanfar.* Completes the
   reconstruction-filter family with the one graphics settled on; the encoder is literally a filter, and the
   negative-lobe mechanism is already evidenced by the refit.

5. **RAY-3 — directed structure (direction role).** *Plate, Olshausen, Stoudenmire.* Foundational HRR; fixes a
   real correctness gap (sequences in superposition are currently ambiguous). Measured, clean, and it unblocks
   any sequence/graph work.

6. **MIS-1 — balance-heuristic estimator combination.** *Cranmer, Pharr, Milanfar, Siemion.* The highest-value
   NEW capability; a bit more design than the measured items, so it follows the quick wins.

7. **CACHE-1 — gradient-cached manifold decode.** *Pharr, Milanfar, Ozcan, Stoudenmire.* Measured (gradients
   halve the anchors), but it carries the validity-radius machinery as baggage, so it slots after the lighter
   pieces.

8. **ACCUM-3 then ACCUM-2 — robust + harmonic accumulation.** *Pharr, Milanfar, Cranmer, Macklin.* Cheap
   robustness and convergence upgrades to the averaging paths (consolidation, forest votes).

9. **Group G — XDATA-1 → XDATA-2 → XDATA-3.** *Milanfar, Eno, Stoudenmire, Olshausen, Baker, Duda,
   Tarter/Siemion.* The through-line and the largest conceptual payoff; research-heavy. XDATA-1 leads because
   coarse-graining (Stoudenmire's RG, Milanfar's manifold map) DEFINES the manifold the looping denoise (XDATA-2)
   and sharpening (XDATA-3) then operate on.

10. **The remainder — ADAPT-2, CACHE-2/3, ACCUM-1, RAY-2, PHASE-1/2, SCALE-1** — as the above land.

### The two honest tensions (stated, not smoothed)

- **Plate would pull RAY-3 earlier.** From the HRR-foundation seat, an ambiguity in how the substrate stores
  sequences is a *correctness* issue, not a feature, and correctness-of-substrate arguably precedes the
  performance items. The order keeps it at #5 only because the measured performance wins (1–3) are cheaper and
  lower-risk — but if substrate-correctness is the priority, RAY-3 moves to #2.

- **Milanfar (with Eno) would pull Group G earlier.** From the denoising seat, the cross-data generalization is
  not the dessert — it's the main result (a denoiser is a manifold map for *any* data), and the rest are
  refinements of machinery that already works. The order defers it only because it's research-heavy and the
  quick wins re-establish rhythm first. If the goal is the biggest idea over the fastest ship, Group G leads.

---

## The honest bottom line

The panel's methods converge cleanly on the front of the list: SAMPLE-1 then RAY-1 are the cheapest, most
broadly-backed, already-measured wins, and both re-establish the close-out rhythm at low risk. The genuinely new
capability the most rigorous seats want is MIS-1 (Cranmer's principled combination), and the deepest prize the
denoising seat wants is Group G (Milanfar's manifold operations, generalized to any data — the user's own
through-line). Two seats dissent on *order* for principled reasons (Plate: substrate-correctness first → RAY-3
up; Milanfar/Eno: biggest idea first → Group G up), and those are real forks worth a decision rather than a
default. Absent that call, start at SAMPLE-1 and work down.

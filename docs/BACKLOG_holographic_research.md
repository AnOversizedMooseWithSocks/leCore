# leCore backlog — holographic sweep + prior sweep + GPU/GLSL research (next session)

Built by combining three inputs: (1) THIS session's **holographic-ness sweep** (are we using the fast VSA/FFT
programs instead of direct Python?), (2) the **prior wiring/generalization sweep**, and (3) the **GPU/GLSL research
report** (July 2026). Every item states: what, why, where (module + faculty), acceptance, the kept-negative to watch,
and its type — `[HOLO]` use a holographic/FFT program, `[GEN]` generalize/consolidate, `[RESEARCH]` from the report,
`[PROCESS]` ritual change.

## Standing constraints (apply to EVERY item — do not relax)
- **NumPy/Flask/stdlib/hashlib core only.** numba/CuPy/Zig/WGSL are opt-in accelerators. **Every GPU/accelerator path
  MUST have a pure-NumPy fallback and must run + pass tests with the accelerator absent.**
- **Additive, backward-compatible, deterministic.** New capability = new module or default-off flag. Existing emitted
  bytes / numeric outputs must not flip. `PYTHONHASHSEED=0`, `hashlib` not `hash()`, seeded `default_rng`.
- **Measurement over narrative.** Every "faster" claim ships with a baseline, a variance estimate across seeds, and
  its kept negatives loud in docstrings + NOTES. A win without a proper baseline is not a result.
- **Rule 0 first.** `find_capability` with 5+ stranger phrasings before building; reuse/extend before adding a sibling.

---

## A. Holographic-ness sweep — use the fast VSA/FFT programs, not direct Python

### A1 `[HOLO]` Spectral-accelerated harmonic inpaint (replace/augment the Jacobi loop)
- **Where:** `holographic/sampling_and_signal/holographic_inpaint.py` (`harmonic_fill`); reuse
  `holographic_laplacian.diffusion_transfer` / `diffusion_operator` and `holographic_shader.Pipeline`.
- **Why:** `harmonic_fill` is iterative **Jacobi** — it propagates information one pixel per sweep, so a large hole
  needs many sweeps (item 6's kept negative N10: no factorization to amortize). The engine already owns the **FFT
  shader algebra**: `diffuse_spectral` / `diffusion_operator` propagate globally in ONE FFT pair. Harmonic inpaint is a
  *Dirichlet* problem (known pixels fixed) on a *non-periodic* domain, so we cannot call `solve_poisson_spectral`
  directly — its docstring warns the closed form is "simply wrong" off a periodic domain. BUT a **diffuse-and-reproject**
  scheme (spectral heat smooth → re-impose known pixels → repeat) converges in far fewer passes than Jacobi because
  each spectral step is global. This is the research report's "fast smoother" idea meeting our own FFT algebra.
- **Acceptance:** on a batch of masked images, spectral-reproject reaches the SAME steady state as Jacobi within a
  documented tolerance (e.g. `< 1e-3` max-abs) in measurably fewer passes / less wall-time; default path unchanged
  (new `method="spectral"` flag, `method="jacobi"` default until measured to win). Report mean/spread/CI across seeds.
- **Kept-negative to watch:** periodic FFT wrap can leak across the frame border (Dirichlet ≠ periodic) — mirror-pad
  or window before the FFT smooth, and MEASURE the border error. If spectral-reproject does not beat Jacobi on real
  holes, file it as a kept negative (like N10) and keep Jacobi. **Do not ship a quality regression to look fast.**

### A2 `[GEN]` Bound `map_frames`' cache with `ColdStore` (stop unbounded RAM growth over a long clip)
- **Where:** `holographic/io_and_interop/holographic_framesource.py` (`map_frames`); reuse
  `holographic_coldstore.ColdStore`.
- **Why:** `map_frames`' `cache` is a plain `{seq: out}` dict — correct for invalidation, but **unbounded**: a long
  clip grows RAM without bound. `ColdStore(keep_warm=N)` is the engine's drop-in bounded LRU keyed store. Let `cache`
  optionally be a ColdStore (or add `max_frames=`), so a long grade holds only the recent window warm.
- **Acceptance:** memoisation still exact (same seq → cached, byte-identical), but resident set is bounded to
  `keep_warm`; plain-dict path unchanged (default). One test pins the bound over > `keep_warm` distinct frames.
- **Kept-negative to watch:** ColdStore's spill/warm has a cost — measure it's cheaper than the recompute it saves for
  a realistic per-frame `fn`; if `fn` is trivial, an unbounded dict may still win (document the crossover).

### A3 `[HOLO]` (research bet, lower priority) Holographic segmentation vs numpy region-growing
- **Where:** `holographic/misc/holographic_vision.py` (`segment_image`); probe `packet_demux` /
  `decompose_piecewise` / "Segment a photo into object regions (demux)" / superposition-cleanup.
- **Why:** `segment_image` is numpy region-growing (the 73× `max_dim` downsample already shipped). The engine has
  demux / superposition-cleanup segmentation primitives; a holographic route MIGHT match quality faster.
- **Acceptance:** ONLY ship if the holographic route matches numpy segmentation quality (region IoU within tolerance)
  at equal-or-better speed on a probe set. **This is a maybe** — likely a measured negative; treat as research, not a
  committed change. Keep the numpy path as the default and the baseline.

### Validated by the sweep (NOT gaps — record so no one "fixes" them into regressions)
- **`map_frames` keys on `seq`, not a frame fingerprint** — CORRECT. `memoize_pure`'s own kept negative ("a cheap
  function of a large array LOSES 21× to byte-fingerprinting") proves seq-keying is right; do **not** reroute
  `map_frames` through `memoize_pure`.
- **`frame_key` / container hashing use `hashlib`** — CORRECT and constitutionally required (never `hash()`).
- **`shader_pipeline` (item 7) is already the FFT shader algebra** — already holographic; nothing to change.
- **The container is a file format, distinct from `ColdStore` (RAM store) and `holographic_archive`** — no reinvention;
  keep separate. (A2 is the one place they should meet.)

---

## B. Prior-sweep residue + generalization catches

### B1 `[GEN]` Shared GLSL shader-assembly helper (the emitters hand-roll their wrappers)
- **Where:** promote into `holographic/io_and_interop/holographic_emit.py` (already the code-emission family — WGSL/C/
  JS/Zig live there, and `glsl_float`/`glsl_vec3` were consolidated there last sweep). Callers: `holographic_sdf.
  _emit_shader`, `holographic_postfx.chain_to_glsl`, `holographic_pattern.pattern_to_glsl`, `holographic_domain.
  cosine_palette_to_glsl`.
- **Why:** `void mainImage` and the `#version 300 es` / `precision` / `out` wrapping are **hand-rolled separately** in
  sdf and postfx; pattern/palette emit bare functions with no assembly. There is no shared "assemble a full WebGL2
  shader from these functions + this entry" helper — the capstone (pattern→palette→postfx→SDF in ONE shader) is manual
  string concat. Add `assemble_glsl(functions, entry, target="webgl2"|"shadertoy")` that composes function bodies + a
  `mainImage`/`main` entry + optional WebGL2 version/precision/out wrapping, deterministically. Each emitter delegates
  its wrapper to it; a new `mind.compose_shader([...])` faculty makes the capstone first-class.
- **Acceptance:** every existing emitter's output stays **byte-identical** (delegation only), the composed capstone
  shader parses as valid GLSL ES 3.00 (shaderfrog), and one faculty assembles SDF+postfx+pattern+palette into a single
  shader. Pin with a test.
- **Kept-negative to watch:** naming collisions when composing (two functions named `palette`) — the assembler must
  detect/namespace duplicates or document the caller's responsibility.

### B2 `[PROCESS]` Make `tools/regen_docs.py` the close-out doc step (not `capdoc.py`+`docgen.py` by name)
- **Why:** last sweep found the per-item ritual regenerated only `capdoc`/`docgen`, leaving `DOC_MAP.md` (stale at 518)
  and `FACULTY_MAP.md` behind. `regen_docs.py` runs all 7 generators / 9 outputs.
- **Action:** update the session close-out ritual (step "SYNC generated docs") to run `python3 tools/regen_docs.py`;
  no code change, a checklist change. (Already fixed the stale docs; this prevents recurrence.)

---

## C. Research-derived items (July 2026 GPU/GLSL report)

> NOTE: reconcile each item's exact constants/formulas against the delivered research report at build time; the
> techniques below are the named, established ones the research targeted, and are correct to ground the work in.
> **Every GPU path here has the required NumPy fallback: tonemappers and the hash run on CPU too; the WGSL emitter is
> pure code-gen; the IIR blur is pure NumPy.**

### C1 `[RESEARCH]` Modern tonemappers as new POINTWISE postfx stages: AgX, Khronos PBR Neutral, (opt.) Tony McMapface
- **Where:** `holographic/rendering/holographic_postfx.py` — new `EFFECTS` entries `agx`, `pbr_neutral` (and maybe
  `tony`); they are **pointwise**, so they slot into item 9's `_GLSL_POINTWISE` and emit to GLSL exactly, AND run in
  NumPy as the CPU path.
- **Why:** AgX (Sobotka) and Khronos **PBR Neutral** are the mid-2020s successors to ACES — better hue handling and
  highlight desaturation; AgX is now the Blender default view transform. They're a straight quality upgrade to our
  colour pipeline and fit our pointwise-emittable architecture perfectly. Tony McMapface is a LUT-based tonemapper —
  emittable only if we bake its LUT (heavier; treat as optional).
- **Acceptance:** each new stage runs in NumPy AND emits GLSL matching the NumPy transcription per-point to float
  precision (same probe-grid test as items 9/10); parses as GLSL ES 3.00; default chains unchanged (ACES stays until
  a preset opts in). AgX's fitted polynomial / matrix form (public) is the CPU + GLSL reference; Tony needs its LUT
  baked to a texture-or-poly fit (defer if the fit's error is loose).
- **Kept-negative to watch:** AgX has a matrix + log-encode + polynomial + inverse-matrix; get the exact fitted
  constants right or the look drifts. Tony's LUT is not a closed form — a poly fit has documented error; keep the
  negative if the fit is too loose to ship.

### C2 `[RESEARCH]` GPU-reproducible integer hash → **unblock the refused `noise`/`fbm` GLSL emit** (item 10)
- **Where:** a shared `hash32` in `holographic/io_and_interop/holographic_emit.py` (or beside `hash_unit`), used by
  both a NumPy `uint32` value-noise/fbm reference AND the GLSL emitter in `holographic_pattern.pattern_to_glsl`.
- **Why:** item 10 REFUSED `noise`/`fbm` because their int64-wrap lattice hash can't be reproduced in GLSL ES 3.00's
  32-bit ints. The research's fix: adopt a **32-bit integer hash** (PCG-hash; Jarzynski & Olano, "Hash Functions for
  GPU Rendering", JCGT 2020; Wolfe/"demofox") and use the **same 32-bit arithmetic on both sides** — NumPy `uint32`
  with explicit `& 0xFFFFFFFF` wrap on the CPU, `uint` in GLSL. Then value-noise/fbm become **per-point reproducible
  and emittable**, closing item 10's kept negative.
- **Acceptance:** a new `noise`/`fbm` variant built on `hash32` matches per-point between the NumPy `uint32` reference
  and an independent transcription of the emitted GLSL on a probe grid (to float precision after the identical
  integer hash); `pattern_to_glsl("noise")` now emits instead of raising; the OLD int64 noise stays the default so no
  existing output flips (new is opt-in, e.g. `hash="pcg32"`).
- **Kept-negative to watch:** NumPy int promotion — every multiply/shift must be masked to 32 bits or the CPU and GLSL
  diverge. Pin the exact constants. Do NOT change the existing int64 noise's output.

### C3 `[RESEARCH]` Faster CPU blur: recursive/IIR Gaussian + summed-area tables as O(n) alternatives to FFT convolution
- **Where:** `holographic/rendering/holographic_postfx.py` `_fft_blur`, and any FFT-blur caller (bloom/glare/dof/
  denoise/sharpen). Add `_iir_gauss` (Young–van Vliet / Deriche recursion) and/or `_sat_box` (summed-area / integral
  image, repeated for approximate Gaussian) as **opt-in** fast paths.
- **Why:** FFT blur is O(n log n) with a full transform pair; recursive IIR Gaussian is **O(n) per axis, independent
  of sigma**, and summed-area box blur is O(n) — both pure NumPy. For large sigma (bloom/dof) these are materially
  faster while staying within a tight, documented tolerance of the true Gaussian.
- **Acceptance:** the fast path matches `_fft_blur` within a documented max-abs tolerance on a probe set AND is
  measurably faster at the sigmas bloom/dof actually use (report the crossover sigma); FFT stays the default (exact
  baseline). Deterministic, pure NumPy.
- **Kept-negative to watch:** IIR Gaussian has edge/boundary transients and sigma-accuracy limits at small sigma; SAT
  box needs 3 passes to approximate a Gaussian and still isn't exact. Measure the tolerance; keep FFT as the exact
  reference and default.

### C4 `[RESEARCH]` Bloom / glare / DoF the honest way: an OPT-IN multi-pass emitter (host runs the passes)
- **Where:** extend the item-9 emitter (`holographic_postfx`) with a multi-pass mode that emits a small ordered set of
  fragment passes + the ping-pong/render-target + (for DoF) depth-texture contract the host wires; reuses B1's
  assembler.
- **Why:** item 9 correctly refused bloom/glare/dof in a single pass (no intermediate texture). The research confirms
  faithful bloom is genuinely multi-pass (threshold → down-sample → separable blur → up-sample → composite); modern
  DoF is gather/tile-based. Rather than fake a single-pass approximation, emit the **pass DAG** (each pass a pointwise-
  or separable-fragment we CAN emit) plus a documented host wiring recipe, so the host runs real bloom on the GPU.
- **Acceptance:** `to_glsl(..., multipass=True)` returns an ordered list of passes (each valid GLSL ES 3.00) + a
  machine-readable wiring spec (inputs/outputs/targets); a NumPy reference executing the same pass DAG matches
  `PostChain.apply`'s bloom within a documented tolerance. Single-pass pointwise path unchanged; multipass is opt-in.
- **Kept-negative to watch:** separable-blur-as-two-passes ≠ true 2-D Gaussian at the corners — document the error;
  this is an approximation of the FFT bloom, stated loud, not sold as exact.

### C5 `[RESEARCH]` WGSL emitter path alongside GLSL (WebGPU is broadly shipping mid-2026)
- **Where:** `holographic/io_and_interop/holographic_emit` already emits WGSL for kernels; extend the shader emitters
  (postfx/pattern/palette/sdf) to target `dialect="wgsl"` via B1's assembler.
- **Why:** by mid-2026 WebGPU/WGSL is broadly available; a WGSL target future-proofs the "leCore generates your GPU
  code" story and unlocks compute shaders (real multi-pass bloom, prefix-sum SAT on GPU) the WebGL2 fragment path
  can't express. Purely additive code-gen — no runtime GPU dependency in core.
- **Acceptance:** the postfx/pattern/palette/sdf emitters accept `dialect="wgsl"` and produce WGSL that a WGSL parser
  validates; the WGSL and GLSL emitters share B1's assembler and the same NumPy reference (per-point match); GLSL stays
  the default. **Reconcile scope with the report** (which effects are worth WGSL first).
- **Kept-negative to watch:** WGSL is not GLSL with renamed types (binding model, entry points, `textureSample` vs
  `texture`, no implicit conversions). Do not machine-translate GLSL→WGSL; emit WGSL from the same source-of-truth
  functions. Keep it additive and default-off.

---

## Suggested execution order (dependencies noted)
1. **B1** (shared GLSL assembler) — unblocks/cleans C1, C4, C5; pure consolidation, byte-identical.
2. **C1** (AgX / PBR Neutral tonemappers) — high value, fits the pointwise-emittable path, low risk.
3. **C2** (32-bit hash → noise/fbm emit) — closes a named kept negative; self-contained.
4. **A1** (spectral harmonic inpaint) — the marquee `[HOLO]` win; research bet, measure hard.
5. **C3** (IIR / SAT fast CPU blur) — speeds bloom/dof/denoise; measure tolerance.
6. **A2** (ColdStore-bounded map_frames cache) — small, safe.
7. **C4** (multi-pass bloom/DoF emitter) — larger; depends on B1.
8. **C5** (WGSL emitter) — larger, additive; depends on B1.
9. **A3** (holographic segmentation) — research/maybe; likely a measured negative, lowest priority.
10. **B2** (ritual: `regen_docs.py`) — process, fold in immediately.

## Global "don't miss it" checklist for the build session
- Every item: Rule-0 audit (5+ phrasings) → build in the right family module → `file_python_check` after each edit →
  loud `_selftest` with numeric asserts + kept negatives → wire a delegating default-off faculty → catalog entry
  (does ≤600 chars, 5/5 discoverability) → three audits 0/0/0 → integration test → **`tools/regen_docs.py`** →
  NOTES append → zip rebuild + clean-extract verify → `present_files`.
- Every GPU/accelerator path: prove the **pure-NumPy fallback** runs and passes with the accelerator absent.
- Every "faster": baseline + variance + kept negative, in the docstring and NOTES. No un-baselined wins.
- Nothing left import-only (reachability import-only set must stay the pre-existing 8).

# holostuff Inverse-Rendering & Auto-Material Backlog — analysis-by-synthesis, image→scene, auto-bump

*Grounded in a probe of the LIVE code (not memory), per the guide's probe-first rule (§6). Two related
capabilities the brief asks for: (A) **auto-bump** — derive a height/normal map from a supplied image so
a material without a bump map can just "auto bump"; and (B) **analysis-by-synthesis** — look at a photo,
build the closest scene we can from known concepts, and tune camera/lighting/composition to match, with
a confidence gate. The paradigm is **vision-as-inference** (Olshausen's seat) and structurally it is
"iterate a projection" pointed at the renderer — already-shipped machinery, wired into a loop. Every item
is SEAT + REAL published method, value-over-effort ranked, kept negatives loud. Nothing is a result until
it beats a proper baseline.*

*Rev 2 adds §3b (SIGGRAPH-grounded methods that push the boundaries — FFT normal integration, photometric
stereo, intrinsic decomposition, and an SVGF-style 1-spp denoiser) and §3c (style transfer & post-processing).
The discipline throughout: the modern SOTA here is mostly **learned** and therefore non-borrowable, so we take
the **classical ancestor** those methods replaced — and, pleasingly, several are FFT/gradient-domain, the
engine's own operator. A probe confirmed there is **no GAN** in the tree (the generative modules are
denoising-based, not adversarial) — and none is needed: classical style transfer is patch/statistics-based.*

---

## 0. Live-codebase audit (rev 3 — probe-first, against the latest zip)

*The latest codebase (282 engine modules) shows the team has been building straight from these backlogs —
so several items are **already done and better-integrated than this doc assumed**, and this audit's job is
to stop us re-requesting them and to point the remaining items at the shipped parts they should reuse (§5.1).
`svgf.py` even carries its own probe-first note listing the sibling render pieces already in the tree.
**This section supersedes the item statuses below;** §1 onward is the original assessment, kept for its
reasoning.*

### A. Already built — do NOT re-request

| Backlog item | Live module | Status |
|---|---|---|
| **IR10** SVGF 1-spp denoiser | `holographic_svgf.py` | **DONE** — a holographic bilateral whose edge-stop is a *cosine of bound (normal, albedo, depth) feature vectors*, run à-trous over `multires`; measured to beat a plain Gaussian on PSNR-to-clean. Its named siblings are all built too: `accumulate.robust_accumulate` (firefly), `temporal.TemporalReuse` (reproject), `adaptive_sample` + `honesty.SPRTRecall` (adaptive sampling), `multires` (pyramid). |
| Renderer adaptive-sampling stop rule (sweep §5.2) | `holographic_adaptive_sample.py` | **DONE** — a calibrated per-pixel stop from the renderer's variance-of-the-mean. |
| **IR6** depth→abstaining-splat lift | `holographic_photo3d.py` | **DONE** — single depth+colour → per-pixel front-surface Gaussians with a geometric support score, **abstaining** on invalid depth, occlusion edges, grazing angles, and the unobserved back. (Depth *estimation* stays the bounded input — matches IR6's posture.) |
| Trusted-horizon "predict-if-confident" (F6) | `holographic_horizon.py` | **DONE** — multi-horizon conformal, interval widens with depth, reports `trusted_horizon`. |
| The confidence layer IR1/IR4/IR10 lean on | `holographic_conformal.py` (+ `forecast`, `analog`, `temporal`) | **DONE** — split-conformal `(point, interval, abstain)`, the `forecast(data)` router, analog recall, temporal reuse. |
| Render/sim orchestration (what IR4 should run *on*) | `holographic_pipeline.py` | **DONE** — `Stage` declares what it `needs`/`produces` so the builder catches a missing G-buffer *before* running; deterministic `FrameState`. IR4's loop should be Stages on this, not a bespoke loop. |

### B. Two corrections the code makes to my earlier framing (kept loud)

1. **Per-pixel adaptive sampling is a CLT/variance interval, NOT conformal.** `adaptive_sample.py` explicitly
   corrects the sweep's "conformal everywhere": a per-pixel Monte-Carlo estimate is a *sample mean* with no
   per-pixel calibration set, so its confidence interval is Gaussian (half-width `z·√(var-of-mean)`), and
   halving it costs 4× the samples (the standard MC law). That is the right tool there; conformal is for
   estimates that *have* a held-out residual set. I carry this into IR10's variance-guided bandwidth too — it
   is a variance/CLT gate, not a conformal one.
2. **`postfx.reinhard` is a TONEMAP, not Reinhard colour transfer.** Naming collision to avoid: `postfx`
   already has `reinhard()` (HDR→LDR, `x/(1+x)`) and `color_grade()` (manual contrast/saturation/temperature
   curves). ST1's *reference-based statistical colour transfer* (match a target image's mean/covariance) is a
   different function and is **not** built — a small add, but don't confuse the two.

### C. Already have the parts — REUSE, don't rebuild (the "full advantage" half)

- **IR1 auto-bump** — the height→normal core **exists**: `displace.bump_normals(mesh, scalar_fn, amount)`
  perturbs shading normals from a scalar field's tangential slope (bump mapping, no vertices move), and
  `displace_mesh` does real displacement. So IR1's *only* genuinely-new code is the **image→height estimator**
  (grayscale + high-pass); the normal-from-height, the `Material` channel, and `displace` are all done.
- **IR7 surface-from-gradient** — the **FFT-Poisson solver exists**: `fluid.py` does pressure projection *in
  the Fourier domain on a periodic (toroidal) domain*, which is exactly the solve Frankot–Chellappa needs — and
  its periodic boundary is exactly what makes the result **seamless-for-tiling** (the feature IR7 wanted). So
  IR7 is "reuse/extract the fluid projection for gradient integration," not "write a Poisson solver."
- **ST3 guided upsampling** — `svgf.py` is *already* a feature-cosine bilateral filter, so a joint-bilateral
  (guided) upsampler is that filter guided by the full-res G-buffer; `postfx.resample` and
  `multires.upsample_to` give the basic up/downscale. ST3 is mostly "reuse `svgf` as the guided upsampler."
- **IR4 the loop** — build it as Stages on `holographic_pipeline.py` (above), reusing `photo3d`,
  `adaptive_sample`, `horizon`, and the confidence layer.

### D. Genuine gaps that remain (correctly requested)

- **IR1** image→height *estimator* (the front-end) — small; feeds the existing `bump_normals`.
- **IR4** the **render-vs-target compare metric** — this is the real IR4 gap: `vision.py` has **no**
  image-compare/SSIM (only histograms/edges — the *ingredients*). Plus the gradient-free optimise loop.
- **ST1** the `color_transfer(img, reference)` function — small; the grade slot exists.
- **IR7** the thin wiring of the fluid FFT-Poisson solve to a `surface_from_gradient` call — small (solver exists).
- **ST2** example-based texture/style (Image Analogies / Quilting) — the `creature_mind` "exemplar" hit was
  incidental; genuine gap.
- **IR9** intrinsic/Retinex albedo–shading split — the `prt` hit was incidental (PRT reflectance ≠ intrinsic
  decomposition); genuine gap.
- **IR2** shape-from-shading and **IR8** photometric stereo — genuine, but both remain **opt-in / lower
  priority** (ill-posed / needs multi-light input), as the doc already scoped them.

### E. Net effect on priorities

The list got *shorter and sharper* — the method working. IR10, the IR6 lift, the horizon gate, the confidence
layer, and the render pipeline are **done**; IR1, IR7, and ST3 are mostly **reuse** of shipped parts. The
genuinely-new remaining code is small: an **image→height estimator** (IR1), a **render-vs-target metric + a
gradient-free loop on `pipeline.py`** (IR4), and a **`color_transfer` function** (ST1) — with IR2/IR8/IR9/ST2
left honestly optional. Revised order: ship **IR1's estimator → existing `bump_normals`** and **ST1** first
(both tiny); **reuse `fluid`'s solver for IR7** and **`svgf` for ST3**; build **IR4's metric + loop on
`pipeline.py`**; leave IR2/IR8/IR9/ST2 as opt-in.

**Rev 3 also adds IR11 (§3d), the 3D plate** — and it is the same lesson once more: a "new" capability that is
~90% shipped parts (`splat` + `photo3d` + `image`'s masked-recovery + `analog` recall), because it is the
engine's oldest move — recover the whole from a corrupted part — pointed one dimension up. It is the honest,
retrieval-based answer to the back-of-the-object boundary IR6 can only abstain on.

---

## 1. The audit — most of the parts are already in the box

The probe shortens the job, as usual. Both halves of an inverse-rendering loop already exist; what is
missing is the loop that joins them, the image→hypothesis bridge, and the height/depth estimator.

**The "look at an image" half — built.** `holographic_vision.py` (582 ln, "seeing with arithmetic," no
learned weights, readable numpy): `to_gray`, `sobel`, `gradient`, `edges` (the exact primitives auto-bump
needs), plus `dominant_colours`, `hue_histogram`, `orientation_histogram`, `hough_lines`, `harris`/
`corners`, `shape_stats`. This is the perception front-end.

**The "compose a scene from concepts → render" half — built.** `holographic_semantic.py` (1149 ln) parses
a scene description into VSA object records and `realize`s a 3-D SDF scene; `holographic_render.py` renders
it through a real `Camera` (pinhole, tunable `eye`/`target`/`fov_deg`) and `Light` (a directional **sun**
with a tunable `direction`, plus point/ambient). The landscape vocabulary is there: `holographic_terrain.py`
(fBm heightfield → mesh or heightfield SDF) for mountains, `holographic_noise.py`/`curlnoise.py` for clouds.

**The material + displacement stack — built, and this is what auto-bump plugs into.** `holographic_material.py`
holds a `Material` as role-bound channels — albedo, roughness, metallic, **normal**, **height/displacement**,
emission, ao, opacity — each a constant or a texture field, bound and bundled, recoverable per channel. So a
derived bump is just *populating a channel that already exists*. `holographic_displace.py` (G3, "Displacement &
bump — push a surface along its normal by a scalar field") is the consumer: two paths, SDF-native (subtract the
height from the distance) and mesh (offset vertices). `holographic_octnormal.py` (octahedral map, Cigolle et
al. JCGT 2014) stores the derived normals compactly on their manifold.

**And the supporting cast — built.** The image archive + `HoloForest` analog recall (warm-start a scene from
the nearest stored one), `project_onto_constraints` (the iterate-a-projection engine the loop *is*), and the
conformal confidence layer from the forecasting backlog (the accept/abstain gate).

**The gaps (the actual backlog):** (1) the render→compare→adjust **loop**; (2) an **image→scene-hypothesis
bridge**; (3) a **height/normal estimator** from an image (auto-bump); and (4) true **monocular scene depth**,
which is the one honestly-bounded item.

---

## 2. Two capabilities, and why one is far easier than the other

The brief lumps these ("depth/displacement maps from images... for automatic bump maps"), but the honest
technical split matters, so state it plainly:

- **Auto-bump / material height (A)** is **tractable and mostly pure arithmetic.** A bump map does not need
  metric depth — it needs a *plausible relative height field* over a roughly flat, fronto-parallel material
  patch, and that falls out of luminance + gradients (+ optionally a light-aware pass). No learned weights,
  squarely inside the constitution. This is the concrete "auto bump" button, and it is the highest-value,
  lowest-effort item here.
- **Analysis-by-synthesis scene matching (B)** is **harder and partly bounded.** The render-compare-adjust
  loop is buildable from shipped parts, but the *geometry* side hits the one non-borrowable wall: true
  monocular metric depth of a scene needs a learned prior the engine bans. So B **matches the visible frame**
  — palette, horizon, light direction, layer composition — and **abstains on the geometry it cannot recover**.

Keeping (A) and (B) separate is the whole point: auto-bump ships early and clean; scene depth stays the
honestly-abstaining research item, and neither is blocked on the other.

---

## 3. The backlog items (ranked)

Each: **what**, the **VSA-native how**, the **real basis + seat**, the **bar**, the **kept-negative risk**,
an **honest effort** label. All follow the §8 close-out ritual.

### IR1. Auto-bump — image → height → normal → material channel *(highest value; the concrete ask; mostly wiring)*
**What.** An "auto bump" option: when a material is given an albedo texture but no bump/normal map, derive a
plausible height field and tangent-space normal map from that image and populate the material's `normal`/
`height` channel automatically. The artist ticks *auto bump*; the engine fills it in.

**VSA-native how (pure arithmetic, reuses `vision` + `displace` + `material`).** Three small steps:
1. **Image → height.** Grayscale via `vision.to_gray`, then a **high-pass** to separate fine surface detail
   from large-scale lighting/albedo gradients (so a slow brightness ramp across the photo does not become a
   giant fake slope). The high-passed luminance is the height field: brighter = raised, darker = recessed.
2. **Height → normal.** The surface is `(x, y, h(x,y))`; its normal is the (negated) gradient of the height,
   normalized — the standard grayscale-to-normal every DCC tool ships, and `vision.sobel`/`gradient` already
   compute the derivatives:

   ```python
   # height h: (H,W) float in [0,1]; strength scales how "deep" the relief reads.
   def normal_from_height(h, strength=2.0):
       gx, gy = sobel_x(h), sobel_y(h)          # dh/dx, dh/dy  (reuse vision.sobel)
       # tangent-space normal = normalize(-strength*dh/dx, -strength*dh/dy, 1).
       # z = 1 keeps it a valid unit normal; strength tilts it toward the gradient.
       nx, ny, nz = -strength * gx, -strength * gy, np.ones_like(h)
       inv = 1.0 / np.sqrt(nx*nx + ny*ny + nz*nz)
       return np.stack([nx*inv, ny*inv, nz*inv], axis=-1)   # (H,W,3), unit
   # pack to an RGB normal map with n*0.5+0.5; store compactly via octnormal.
   ```
3. **Feed the material.** Write the normal (and, if the user wants real relief, the height) into the existing
   `Material` `normal`/`height` channel. `displace` consumes the height for genuine geometry (IR5); `octnormal`
   quantizes the normals on their manifold.

**Real basis + seat.** Classical grayscale-to-normal (the game-tool standard); frequency-based detail/base
separation (a high-pass is the simplest layer of it). **Seat: Milanfar** — separating fine detail from
large-scale shading is his high-pass / denoise-as-manifold-map turf — plus the material/`displace` thread.

**Bar.** On a set of real material photos (brick, bark, fabric, stucco), the auto-derived normal map, **relit
under a moving light, produces plausible surface relief** a human rates clearly better than flat; where a
ground-truth normal map exists, report mean **angular error** vs it. On a *flat printed* control (a photo of a
poster), it **correctly abstains** — low bump-confidence, fall back to flat — rather than inventing relief.

**Kept-negative risk (loud).** Luminance-as-height is a **heuristic, not a measurement**: (1) *baked
directional lighting* becomes fake grooves (a real shadow reads as a crevice) — the high-pass reduces the
slow component but not a hard cast shadow; (2) *albedo that isn't height* becomes fake relief (a painted
stripe reads as a ridge) — this is a fundamental ambiguity no arithmetic resolves without more information
(hence IR2's light-aware pass, still bounded). It is a **plausible perceptual bump, not a depth map**, and
the bump/displacement **strength** is a user knob, not an inferred physical scale.

**Effort: low–medium.** Mostly wiring `vision` → `material`; one estimation function + a high-pass. Purely
additive, and it lands the button the brief asked for.

### IR2. Light-aware height (shape-from-shading, opt-in when the light direction is known) *(medium; a better bump)*
**What.** When a light direction is available — supplied by the artist, or estimated by the IR4 loop — do a
light-aware **albedo/shading separation** and recover height by partially inverting the Lambertian shading,
for a cleaner bump than luminance alone.

**VSA-native how.** Given light direction `L`, Lambertian shading is `n·L`; solving for the normal that best
explains the observed luminance (under a constant-albedo assumption) yields a height gradient to integrate.
Small numpy; reuses `vision` gradients and the IR1 integration.

**Real basis + seat.** Shape-from-shading (Horn 1970/1989). **Seat: Milanfar** (image formation / restoration).

**Kept-negative risk (loud).** Single-image shape-from-shading is **ill-posed** — it assumes roughly constant
albedo, a known single light, and a Lambertian surface, and it fails when any of those break. So it is
strictly **opt-in**, and **IR1's luminance heuristic stays the honest default fallback**; IR2 is used only when
its assumptions hold and it measurably beats IR1.

**Effort: medium.** Builds on IR1 + a light direction.

### IR3. The perception → scene-hypothesis bridge *(medium–high; the front-end for B)*
**What.** Turn `vision.py`'s features into a **starting scene hypothesis** and seed parameters: the palette
(`dominant_colours`) sets sky/terrain colours; the horizon (`hough_lines`) sets the camera pitch and the
sky/ground split; the top-to-bottom luminance gradient estimates the **sun direction** (low + warm = sunset);
the horizontal layer bands map to sky / mountains / foreground. Then **analog-recall** (`HoloForest`) the
nearest stored scene as a warm start, and compose the hypothesis as a VSA scene via `semantic`.

**Real basis + seat.** Vision-as-inference / scene-gist estimation. **Seat: Olshausen** (inference) + **Pharr**
(sublinear analog recall — the `HoloForest`).

**Kept-negative risk (loud).** This is **archetype-level detection, not semantic segmentation** — it works
when the photo matches a known scene vocabulary (a landscape-at-sunset is close to ideal) and it should
**abstain, not hallucinate**, when the scene is outside that vocabulary. Sun-direction-from-gradient is a
coarse estimate, refined by the loop (IR4), not a measurement.

**Effort: medium.** Wiring `vision` features → `semantic` seeds + a recall warm-start.

### IR4. The analysis-by-synthesis loop — render → compare → adjust *(high value; the headline; the auto-calibration cycle)*
**What.** The loop the brief describes: analyze the target, build a hypothesis (IR3), render it, compare to the
target, adjust camera/lighting/composition to reduce the difference, repeat — and **gate on confidence**,
accepting when the match is confidently good and abstaining on what cannot be matched. The loop *is* the
"auto-calibration."

**VSA-native how.** `analyze (vision)` → `hypothesize (IR3)` → `render` (`render`/`sdf_render`, tuning
`Camera.fov_deg`/`eye`/`target` for composition, `Light.direction` for the sun, `terrain`/`noise` params for
mountains/clouds) → `compare` with a **perceptual metric** built from `vision` (colour-histogram + edge/
orientation agreement + multi-scale structure — *not* raw pixel MSE, which a one-pixel shift wrecks) →
`adjust` **gradient-free** (coordinate descent, or the resonator's alternating projection, or a small sampled
population — the **width/superposition lever renders a whole candidate batch cheaply**) → `gate` with conformal
(accept / abstain). It is `project_onto_constraints` with the target image as the constraint.

**Real basis + seat.** Analysis-by-synthesis / inverse graphics; render-and-compare pose/scene estimation.
**Seat: Olshausen** (vision as inference) + **Stam/Macklin** (render-and-compare, predict-and-verify) +
**Cranmer** (the calibrated accept/abstain gate). *Prior art, named honestly as the thing we CANNOT borrow:*
**differentiable rendering** (Mitsuba 3, nvdiffrast) backprops the pixel error into the scene — and autodiff is
banned by the constitution, which is exactly why the loop is gradient-free.

**Bar.** First, a **self-recovery test** (the honest way to validate inverse rendering without a real photo):
render a *known* scene, hand the rendered pixels back to the pipeline, and recover the `Camera` and sun
`Light.direction` **within tolerance** — you know the ground truth going in, so the error is exact. Then, on
real photos, match palette + horizon + light-direction + composition to a target confidence, **abstaining** on
the geometry it cannot recover. Measured against a fixed hand-set scene (the strawman).

**Kept-negative risk (loud).** (1) **Gradient-free** → coarser and slower than differentiable inverse
rendering; it leans on determinism, analog-recall seeding, and rendering many candidates at once. (2) The
**perceptual-metric ceiling** is roughly SSIM-style structural comparison — no LPIPS-quality perceptual loss,
because that is learned. (3) It **matches the visible frame and abstains on occluded geometry and metric
depth** — see IR6.

**Effort: high.** The research piece; built on IR1–IR3 + the conformal gate. Ship the self-recovery test
first; it is the measurable milestone.

### IR5. Displacement / geometry from a confident height map *(medium; opt-in true relief)*
**What.** Promote a **high-confidence** height map (from IR1/IR2) from a bump to actual geometry via
`holographic_displace` (SDF: subtract the height from the distance; mesh: offset vertices along the normal) —
for hero surfaces where a normal-only bump is not enough at grazing angles or silhouettes.

**VSA-native how.** Pure reuse of `displace` (already built) with the IR1 height as the scalar field; gate on
the IR1 bump-confidence so only trustworthy relief becomes geometry.

**Kept-negative risk (loud).** Only as good as the height estimate (inherits IR1's ambiguities); adds real
geometry cost; **gated on confidence** so a shaky height does not deform a mesh.

**Effort: medium.** Wiring the height into the shipped displacement paths.

### IR6. Monocular scene depth — the bounded, abstaining item *(low priority; the honest boundary)*
**What.** True metric monocular depth of an arbitrary scene (MiDaS / Depth-Anything style) — named here so no
future session tries to fake it.

**Honest posture (this is the kept negative *as the item*).** Robust single-image metric depth needs a
**learned prior the constitution bans** (no torch, no learned weights). So the engine does **not** claim it.
For scene-scale depth the honest options are: (a) **require a supplied depth/height map** from the artist, or
(b) use only the **relative cues available in the frame** — layer/occlusion ordering, horizon, relative size —
with **heavy abstention** and no metric claim. This is the genuinely non-borrowable part of photo-to-3D, and
it is exactly *why* auto-bump (IR1) is scoped to materials, where relative height is enough and metric depth is
not needed.

**Effort: n/a (a boundary, not a build).** The deliverable is the honest scope, kept loud.

---

## 3b. SIGGRAPH additions (rev 2 — papers that push the boundaries)

*A pass through the graphics literature for methods that overcome the boundaries above, under one
discipline: the modern SOTA is mostly **learned** (a trained network — non-borrowable), so what we take
is the **classical ancestor** those learned methods replaced. The happy pattern: several of them are
FFT / gradient-domain, which is the engine's native operator. Each is cited by paper.*

### IR7. FFT normal↔height integration — consistent and seamless (Frankot–Chellappa) *(low; upgrades IR1; pure FFT)*
**What.** IR1 goes height→normal by a gradient; the inverse — recover a *consistent* height field from a
normal/gradient field — is the classic **surface-from-gradient** problem, and its canonical solver is
**pure FFT**. Frankot & Chellappa (1988) project the (generally non-integrable) gradient field onto
integrable Fourier basis functions: with `FT(∂x) = jξx`, the height is
`Z = irfft2( (−jξx·P − jξy·Q) / (ξx² + ξy²) )` — one forward transform of the gradients, a per-frequency
divide, one inverse. That is the engine's own operator (`bind` *is* FFT convolution); a handful of lines
on `rfft2`/`irfft2`, reusing `postfx`'s FFT machinery.
**Why it matters for auto-bump.** It makes the height↔normal round-trip **integrable and drift-free**, and
— the useful accident — its **periodic boundary makes the result seamlessly tileable**, which is exactly
what a material texture wants. It also unifies with `postfx`'s FFT-convolution family and with Poisson-style
gradient-domain editing (the screened-Poisson solver is the same object plus a data term).
**Real basis + seat.** Frankot & Chellappa (1988); Simchony, Chellappa & Shao (1990, DCT/DST variants);
Agrawal, Raskar & Chellappa (2006, the range of gradient-field reconstructions). **Seat: Stam/Puckette**
(FFT-on-a-domain is the bind operator) + the material thread.
**Kept-negative risk (loud).** The **periodic boundary is a systematic bias on non-periodic surfaces** (the
textbook distortion — a face's cheek and nose forced equal at the border). For *tileable materials*
periodicity is a feature; for a bounded scene surface, prefer the DCT/DST variant (Simchony et al.) or a
Poisson solve with the real boundary. State which regime you are in.
**Effort: low.** A few lines of `rfft2`/`irfft2` over shipped machinery. Do it *with* IR1.

### IR8. Real normals from multiple lights — photometric stereo *(medium; the honest "do it right" upgrade)*
**What.** When the artist can supply **three or more shots of the same surface under different, known light
directions**, recover the **exact per-pixel normal** — no luminance heuristic, no ambiguity. The
physically-grounded auto-bump for anyone who can take a few photos.
**VSA-native how.** Lambertian shading is `I = ρ (n·L)`; stack ≥3 observations and solve the small per-pixel
linear system for `ρ·n`, normalize to get `n` (and `ρ`, the albedo, for free). Vectorized numpy over the
pixel grid; feed `n` through IR7 for a consistent height, and into the `Material` normal channel.
**Real basis + seat.** Woodham (1980, photometric stereo). **Seat: Ozcan/Milanfar** (recovering structure
from multiple measurements — Ozcan's computational-imaging turf).
**Kept-negative risk (loud).** Needs **multiple registered images under known lights** — most users have only
one (fall back to IR1/IR2). Assumes **Lambertian** surfaces and known lights; specular/shadowed pixels
violate it and want robust (median/RANSAC) variants. But where the inputs exist, this is **measured** relief,
not a heuristic — the honest ceiling.
**Effort: medium.** A per-pixel linear solve + IR7 integration.

### IR9. Separate albedo from shading — intrinsic decomposition (Retinex) *(medium; attacks IR1's core ambiguity)*
**What.** IR1's deepest failure is that *albedo that isn't height* becomes fake relief (a painted stripe reads
as a ridge). **Intrinsic image decomposition** splits an image into a **reflectance (albedo)** layer and a
**shading (illumination)** layer — and it is the *shading* layer that carries the real relief cue, cleanly
separated from the paint.
**VSA-native how.** The classical route is **Retinex** (Land & McCann): assume large gradients are reflectance
edges and small smooth gradients are shading, threshold the gradient field accordingly, and **reintegrate with
the same FFT/Poisson solver as IR7** — so intrinsic decomposition, surface-from-gradient, and gradient-domain
editing are **one gradient-domain engine** (Agrawal et al. show Poisson and Frankot–Chellappa are special cases
of the general gradient-field reconstruction). Feed the albedo to the material's albedo channel and the shading
to IR1's height estimate for a much cleaner bump.
**Real basis + seat.** Land & McCann (Retinex); Horn (1974); Agrawal, Raskar & Chellappa (2006). **Seat:
Milanfar** (intrinsic images / gradient-domain restoration).
**Kept-negative risk (loud).** Single-image intrinsic decomposition is **ill-posed** — the split is a
heuristic (a gradient threshold) and it fails on soft reflectance edges and hard cast shadows. It *improves*
IR1's separation; it does not *solve* it, and the learned SOTA is better and non-borrowable — say so.
**Effort: medium.** A gradient threshold + the IR7 reintegration.

### IR10. Render at 1 spp, denoise to converged — SVGF-style variance-guided à-trous *(high value; the render speedup, and NOT a neural net)*
**What.** The big "impossible" speedup, fully borrowable because it is classical signal processing, not a
network. **SVGF** (Schied et al. 2017) reconstructs a clean image from a **1-sample-per-pixel** path trace:
temporally accumulate reprojected history to raise the effective sample count, estimate per-pixel luminance
**variance**, and drive an edge-avoiding **à-trous wavelet** filter whose bandwidth **widens where variance
(noise) is high and tightens where it is low** — so noise is smoothed while edges and detail survive. Render
cheap, denoise to look expensive.
**VSA-native how, and why it fits us three ways.** (1) `pathtrace` **already returns per-pixel variance** — the
exact signal SVGF is driven by. (2) The à-trous wavelet is a **dilated convolution**, and `postfx` already does
FFT-convolution while `multires` already builds the pyramid — "one operator, many costumes." (3) The
variance→bandwidth rule **is the same confidence gate** as the forecasting/conformal work: denoise (and spend
samples) *more* where the estimate is uncertain, *less* where it is already confident. The edge-stopping
functions use the G-buffer (normal/depth/albedo) the renderer already has; temporal accumulation reuses the
reprojection ideas in `backwardwarp`/`raycoherence`.
**Real basis + seat.** Dammertz et al. (2010, edge-avoiding à-trous wavelet, EGSR); Schied et al. (2017, SVGF,
HPG). **Seat: Pharr** (the reconstruction filter that makes path tracing tractable) + **Milanfar** (edge-aware
filtering as a manifold map) + **Cranmer** (the variance-driven gate).
**Bar.** A 1-spp path trace + SVGF reaches a target SSIM to a high-spp reference at a fraction of the samples;
the variance-guided bandwidth measurably beats a fixed-width blur at matched detail. Measured against the
un-denoised 1-spp (noise) and a naive Gaussian (over-blur) — the honest baselines.
**Kept-negative risk (loud).** Denoising **trades variance for bias** — over-blur is the failure, worst on
glossy reflections and thin detail (the papers say so); temporal accumulation can **ghost** under fast motion
(needs reprojection rejection). It is a *reconstruction*, not free convergence. The learned denoisers
(OIDN/OptiX) are sharper and non-borrowable; SVGF is the borrowable classical one, and it is genuinely good.
**Effort: medium–high.** A wire of variance→à-trous→temporal over shipped parts; the temporal reprojection is
the fiddly bit.

---

## 3c. Style transfer & post-processing (the second question — non-GAN, classical)

**First, honest scope.** holostuff has **no GAN.** The generative modules (`hopfield`, `diffuse`/`diffusion`,
`generate`) are *denoising / attractor-based* — generation by running a denoiser backwards from noise — not
adversarial, and a GAN would need gradient training and learned weights the constitution bans. **But you do
not need one for this.** The pre-deep-learning style-transfer literature is classical, patch/statistics-based,
and borrowable, and it plugs straight into `postfx` + the archive + `HoloForest` recall. There is a direct
panel line, too: **Elad & Milanfar, "Style Transfer via Texture Synthesis" (IEEE TIP 2017)** is a *non-neural*
style-transfer method — Milanfar's own published work, so it attributes to his seat cleanly.

### ST1. Colour transfer — instant grading / mood match *(low; the easy, powerful win)*
**What.** Match a target image's colour **statistics** (mean and covariance in a decorrelated colour space)
onto a render — Reinhard et al. (2001). Pure statistics, a few lines, and it is exactly the "match the
sunset's mood" knob IR4 wants: grade a render toward a reference photo. Slots into `postfx`'s colour-grade
stage.
**Kept negative.** Global statistics only — it moves *colour*, not *content*, and can wash out when the two
palettes are very different (local variants fix this at more cost). **Effort: low.**

### ST2. Example-based texture / style — Image Analogies & Image Quilting *(medium; artistic + material)*
**What.** "A is to A′ as B is to B′": learn a filter from an example pair and apply it to a new image, all by
**patch matching over a Gaussian pyramid** — no learned weights (Hertzmann et al., SIGGRAPH 2001; Efros &
Freeman, *Image Quilting*, SIGGRAPH 2001; Elad & Milanfar 2017). Uses: **automatic artistic filtering**
(painterly renders), **texture-by-numbers** (paint a label map → synthesise a material), **texture synthesis**
for seamless tileable materials (feeds IR1 auto-bump), and **super-resolution** (a low-res→high-res analogy).
The patch search is native — it is `HoloForest` recall over patch descriptors.
**Kept negatives (loud).** Classical example-based transfer needs the example pair to **approximately align**,
and its quality is **below neural** for arbitrary artistic styles (the literature is explicit — Image Analogies
"gave poor synthesis results" next to Gatys for hard cases); it is best for **texture, colour, and material**,
not free-form painterly restyle. It is patch-copying, so it can repeat or seam (Image Quilting's min-cut
boundary mitigates that). **Effort: medium.**

### ST3. Example-based super-resolution as a render speedup *(medium; the speedup angle)*
**What.** The quality-and-speed payoff of ST2: **render small, upscale by example** — an Image-Analogies
low→high analogy, or classical **joint bilateral / guided upsampling** (He et al. 2013, guided filter) steered
by the full-res G-buffer the renderer already has. Combined with IR10 (denoise the cheap render), this is a
fully-classical, no-learned-weights analogue of render-cheap-then-enhance.
**Kept negative.** Classical upsampling **invents plausible, not true, detail** and tops out below learned
super-resolution; guided upsampling needs a clean full-res guide (the G-buffer supplies it). **Effort: medium.**

**What you gain, summarised.** *Speedups:* IR10 (1-spp → clean) and ST3 (render small → upscale) — both
classical, both driven by the variance / G-buffer you already compute. *Quality:* IR10's edge-aware
reconstruction, ST1's reference grading, ST2's material detail and seamless tiling. *New post-fx:* reference
colour grading (ST1), painterly/artistic filters and texture-by-numbers (ST2) — added to `postfx`'s existing
bloom / glare / DoF / tonemap / grade chain. All non-GAN, all inside the constitution, all reusing
`postfx` + `multires` + `HoloForest` + the confidence gate.

---

## 3d. The 3D plate — recall the whole from a partial view (rev 3)

### IR11. A 3D splat-bundle object archive — the plate, one dimension up *(medium; the honest answer to the back-of-the-object boundary)*

**What.** Store a library of *complete* 3D objects/scenes as content-addressable, damage-tolerant splat
bundles — the 2-D plate's mechanism (disjoint-slot Walsh–Hadamard keys + a view fingerprint for recall +
joint masked recovery) applied to 3-D splat scenes. Its headline use for this pipeline: when a photo gives
`photo3d` only the visible front and it abstains on the rest, **match that front to the nearest stored
*complete* object and recover the whole — including the unobserved back — by treating the missing geometry
as "damage."** Retrieval, not hallucination, and it **abstains when nothing in the library matches**.

**Think holographically — this is one operation, not a new one.** "Recover the clean whole from a corrupted
part" is the engine's oldest move. It is `cleanup` (snap a noisy vector to a stored one). It is
`consolidation` (project onto the subspace real states occupy). It is the resonator (factor a bundle back
into its parts). It is `holographic_analog`'s forecasting — "find the stored thing whose visible part looks
like this, return the whole." The 2-D plate is that move for **images**; IR11 is the *same move for
geometry*, because a scene is a bundle of Gaussians exactly as a memory is a bundle of role-bound vectors —
so a 3-D plate is a bundle like any other bundle. **Nothing new is invented; a shipped primitive is pointed
at a new field.** That is the whole reason it is cheap.

Run the §5.2 up / down / sideways check and it holds at every scale:

- **Down** (the components): the same recovery works on ONE object recovered from its front, on ONE splat
  recovered from the bundle, on a region recalled by a box query (`splat_archive` already does region query).
- **Up** (the whole): it works when the object is itself part of a larger scene — recover a whole scene (a
  bundle of object-plates) from a partial view. A plate of plates is still a plate.
- **Sideways** (the costumes): *field* (recover a splat / SDF / radiance field), *structure* (recover a scene
  factorization), *sequence* (this **is** `analog`'s "recall what followed" — time is the missing part
  instead of the back), *program* (recall a recipe from a partial spec). One engine, five costumes.

**VSA-native how — reuse, don't rebuild (per the §0 audit, every piece already ships):**
- A scene is already a splat bundle (`holographic_splat`), and `photo3d.photo_to_gaussians` already produces
  the per-pixel 3-D splats to store and to query with.
- Storage reuses the 2-D plate: disjoint WHT key slots so stored objects stay orthonormal with no crosstalk
  (`holographic_image._fwht`, the `capacity*keep <= dim` budget), and a small **view fingerprint** per object
  kept *outside* the plate for recall — exactly the archive's design.
- Completion reuses the plate's **joint masked recovery** — the CG solver already in `holographic_image`
  (`_cg`): the very same "solve for the unknown entries given the known ones" that reconstructs an occluded
  image, now with the unobserved back as the mask.
- Recall reuses `holographic_analog` / `HoloForest`: match the photo's front fingerprint to the nearest
  stored complete object.
- The confidence gate is the shared one: fingerprint similarity below a floor → **abstain** and fall back to
  `photo3d`'s honest front-only reconstruction. Same abstention discipline as everywhere else.

**The shape of it, readable:**
```python
# STORE a complete object: give it a plate slot, keep a view fingerprint for recall.
def archive_object(plate, splats, view_fingerprint, key_slots):
    # Each object gets a DISJOINT pool of WHT key slots -> the keys stay orthonormal,
    # so objects don't bleed into each other (exactly how the 2-D plate multiplexes images).
    # The splats ARE the payload; the fingerprint lives OUTSIDE the plate so recall
    # keeps working even when much of the plate is damaged.
    plate.add(bind(wht_keys(key_slots), pack(splats)))   # superpose into the plate
    plate.fingerprints.append(view_fingerprint)

# RECALL + COMPLETE from a partial front view: return the whole object, or abstain.
def complete_from_view(plate, front_splats, front_fingerprint, match_floor=0.6):
    i, similarity = plate.nearest(front_fingerprint)     # analog recall by fingerprint
    if similarity < match_floor:
        return None            # not in the library -> ABSTAIN, keep photo3d's front-only (honest)
    whole = plate.recover(i)   # joint masked recovery: the CG solve fills the unobserved "damage"
    return align(whole, front_splats)   # coarse pose/scale onto the observed front
```

**Real basis + seat.** The 2-D plate's own validated lineage (WHT structured keys, disjoint-slot
multiplexing, joint masked recovery — all in `archive`/`image`); 3-D Gaussian Splatting (Kerbl, Kopanas,
Leimkühler & Drettakis 2023) for scene-as-Gaussian-sum; the Splatter Image / Flash3D single-view lift
`photo3d` already cites. **Seat: Ozcan** (recover a clean signal from a degraded/partial holographic
measurement — the archive's literal job) + **Drettakis** (the splat scene) + **Pharr/Olshausen**
(content-addressable recall).

**Bar.** On a small library of complete objects rendered to single-view front splats, recall-and-complete
recovers the held-out back geometry **closer to ground truth than `photo3d`'s abstain-only baseline** for
objects the library contains, and **correctly abstains** (falls back to front-only) for an object it does
not. Measured against `photo3d` alone (the honest baseline) and a "mirror the front to fake a back" strawman.

**Kept-negative risk (loud).**
- **Coverage-limited: it completes only objects it has stored.** Outside the library it abstains, so the win
  scales with library coverage, and the honest output on an unseen shape is "front only" — never an invented
  back. This is the point: retrieval, not hallucination.
- **Capacity is a dial, not a wall** (per the compute-architecture reconsideration): a plate holds ~0.1–0.2×D
  objects before crosstalk; the escapes are the disjoint-slot budget already in the plate, plus tiling
  (`octree`) and the splat archive's importance-ordering — bucket-size and spill, don't cap.
- **Lossy and isotropic** — inherits `splat_archive`'s kept negatives (exact-vs-low-byte trade; isotropic
  splats).
- **Registration** — matching a photo's front to a stored object needs coarse pose/scale alignment first (the
  fingerprint gives rough, the IR4 loop refines); a wrong alignment must surface as a low match and abstain,
  never a confident-wrong completion.

**Effort: medium — mostly assembly.** The splat bundle, the single-view splats, the WHT plate storage, the CG
masked-recovery solver, and the analog recall all already ship; IR11 wires them into "recall a complete object
and recover its whole from a partial view," plus a coarse aligner. It is the §0 audit's discipline in action —
a "new" capability that is ~90% shipped parts, which is exactly what thinking holographically buys.

---

## 3e. Upscaling & render acceleration (rev 4 — post-process upscale + "larger res without taking forever")

*The modern answers here — DLSS, XeSS, FSR2/3's ML path, Intel's OIDN — are all **learned** and therefore
non-borrowable. So this pass takes the **classical ancestors** those replaced, and — thinking
holographically — every one turns out to be a move the engine already owns. Upscaling and render
acceleration are not new capabilities; they are three shipped moves wearing a rendering costume:*

1. **Recover a full signal from a partial/coarse observation** — the plate / masked recovery / `cleanup`.
   *This is both **checkerboard rendering** (shade half the pixels, recover the rest) and **upscaling**
   (recover full-res from low-res).* IR11's move, one more costume.
2. **Coarse-to-fine + recover the lost detail** — the `multires` pyramid + `sharpen` (high-pass detail
   recovery) + `svgf` (edge-aware). *This is **FSR1** (edge-adaptive upsample + contrast-adaptive sharpen).*
3. **Spend compute where the signal is uncertain, amortized over frames** — `adaptive_sample` (variance),
   `sampling`/`lowdiscrepancy` (blue-noise, less noise per sample), `temporal` (accumulate across frames).

So the "new" code is small; the rest is wiring shipped parts — the §0 audit lesson again.

### IR12. FSR1-style spatial upscaler (EASU + RCAS) — the post-process upscale option *(medium; half of it already ships)*
**What.** A post-process **upscale** stage: take a low-resolution render → display resolution, edge-adaptively,
then sharpen — so you can render at (say) 1080p and present at 4K. The upscale toggle the brief asks for.
**VSA-native how — reuse, don't rebuild.** FSR1 is two passes, and one of them already exists:
- **EASU (Edge-Adaptive Spatial Upsampling)** — a modified 2-tap **Lanczos** whose weights are steered by
  *gradient reversals* (how neighbouring gradients differ), so it upsamples along edges instead of across
  them. The gradient analysis is `vision.sobel`/`gradient` (shipped); the directional/elliptical weighting by
  local gradient reversals is the **one genuinely-new piece** — today's `postfx.resample` is plain bilinear,
  which FSR1 exists precisely to beat. Small and readable: a per-output-pixel weighted Lanczos tap set whose
  anisotropy comes from the local gradient.
- **RCAS (Robust Contrast-Adaptive Sharpening)** — a post-upscale sharpen that *avoids amplifying noise*.
  This is **`holographic_sharpen` already** — Van Cittert negative-lobe sharpening whose own kept-negative is
  literally "over-sharpening amplifies high-frequency noise, so stop at the noise floor," which is exactly
  RCAS's design goal. The sharpen half is done; wire it as the second pass.
- Both slot into `postfx` as an ordered stage, beside the existing bloom/grade/`resample` chain.
**Real basis + seat.** AMD FidelityFX Super Resolution 1.0 (EASU + RCAS) — open source, no ML, with a
**SIGGRAPH 2021 presentation** (Unity + AMD, *advances.realtimerendering.com/s2021*); NVIDIA Image Scaling
(NIS) is the same class. The EASU kernel is a Lanczos variant (Duchon 1979). **Seat: Milanfar** (edge-aware
resampling / detail as a manifold map is his turf) + the postfx thread.
**Bar.** On a downscale→upscale round-trip, EASU beats plain bilinear (`postfx.resample`) on PSNR and edge
sharpness to the native-res reference; RCAS (via `sharpen`) recovers detail without pushing past its principled
noise-floor stop. Measured against bilinear (the strawman) and native-res (ground truth).
**Kept-negative risk (loud).** Classical spatial upscaling is **below learned** (DLSS/XeSS): it cannot invent
detail that is not in the low-res input, and EASU's upscaling artifacts get **multiplied by the RCAS pass**
(measured and reported by reviewers). A good, cheap, deterministic upscaler — not a magic one.
**Effort: medium.** The EASU kernel is new (small); RCAS is `sharpen` (done); wiring is a `postfx` stage.

### IR13. Checkerboard / sparse rendering — shade half the pixels, recover the rest *(medium; the sampling trick; the reconstruction engine already ships)*
**What.** Shade only ~50% of the pixels (a 2×2 checkerboard pattern, alternated between frames) and
**reconstruct** the unshaded pixels — roughly halving the shading cost for a near-full-resolution result. The
"larger resolution without it taking forever" trick, done as a sampling pattern rather than a naive
lower-resolution render.
**VSA-native how — the holographic gem.** The reconstruction *is* the plate's masked recovery (IR11's move,
one costume over): the unshaded pixels are **"damage,"** recovered by the `_cg` masked-recovery solve already
in `holographic_image` (`reconstruct(mask=...)`, `damage_mask`). Edge-aware reconstruction reuses `svgf`'s
feature-cosine bilateral; the temporal variant (alternate the checkerboard, reproject the previous frame by
motion/depth) reuses `temporal.TemporalReuse` + `backwardwarp`; the sub-pixel jitter reuses `lowdiscrepancy`.
The genuinely-new work is a thin "shade a masked subset" render mode plus wiring the existing reconstruction:
```python
# CHECKERBOARD RENDER: shade only half the pixels, recover the rest.
def render_checkerboard(scene, camera, H, W, frame_parity=0):
    mask   = checkerboard_mask(H, W, frame_parity)   # 2x2 pattern; flips each frame so gaps fill over time
    shaded = render_pixels(scene, camera, mask)       # shade ONLY where mask is True  -> ~half the work
    # the UNSHADED pixels are "damage" -> recover them exactly as a damaged plate is recovered:
    full   = image_reconstruct(shaded, mask)          # the shipped _cg masked-recovery solve
    return full                                       # (temporal: blend with the reprojected previous frame)
```
**Real basis + seat.** Checkerboard rendering (PS4 Pro; EA Frostbite — *Battlefield 1*, *Mass Effect:
Andromeda*; Ubisoft *Rainbow Six Siege*, mid-2010s) and **Intel's Checkerboard Rendering (CBR) for real-time
upscaling** (Intel white paper, 2018, with an open-source DX12 sample) — the "Intel has some too." **Seat:
Ozcan** (recover a clean signal from a *partial/masked* measurement — the archive's literal job) + **Pharr**
(the reconstruction filter) + **Stam/Macklin** (shade fewer, reconstruct).
**Bar.** Checkerboard-shade + reconstruct matches full-resolution quality (PSNR) at ~half the shaded pixels,
and the temporal variant reduces shimmer versus single-frame reconstruction. Measured against full-res (ground
truth) *and* against shading at half-resolution-then-upscaling — checkerboard should beat the latter at matched
cost (the documented CBR advantage: better accuracy per unit of render cost than plain upscaling).
**Kept-negative risk (loud).** Reconstruction **costs more than simply rendering at a lower resolution** — it
is a *trade* (better accuracy per cost), not free. Under motion it can **shimmer** in fine detail unless the
reprojection rejects disoccluded pixels by depth/motion (else ghosting), and colour/visibility exist only at
the render-target resolution, so it is a reconstruction, not true supersampling.
**Effort: medium.** Mostly wiring the masked-shade mode + the shipped reconstruction; the temporal reprojection
is the fiddly bit (the same one SVGF already handles).

### Already shipped — compose these, don't re-request (the "full advantage" half)

The rest of the render-acceleration toolbox is already in the tree and just needs composing (via
`pipeline.py`): **adaptive sampling** (`adaptive_sample` — spend samples where variance is high),
**blue-noise / low-discrepancy sampling** (`sampling`, `lowdiscrepancy` — less noise per sample = an effective
speedup), **temporal accumulation** (`temporal.TemporalReuse`, `backwardwarp` — amortise samples across
frames), **MIS** (`mis` — the Veach balance heuristic), **firefly clamping** (`accumulate.robust_accumulate`),
and **irradiance / radiance caching + interpolation** (`cache`, `raycoherence`, `prt` — compute GI sparsely and
interpolate, which is exactly V-Ray's *light cache* / *irradiance map* move). None of these needs re-building —
they are the shipped levers IR12/IR13 sit alongside.

---

## 3f. Render channels (AOVs / passes) — one bundle, many views (rev 5)

### IR14. Render channels — choose what to separate; default to beauty *(medium; mostly exposing an `unbind` that already works, plus one renderer step)*

**What.** Let the renderer output selectable, *separate* channels — per object, per material, per material
property, per simulation property, per data pass — each with its own alpha, for compositing, science, and
debugging. The user chooses which channels to separate for which things (a small selection spec); with no
spec, they just get the **beauty** pass (the fully-composed render) exactly as today.

**Think holographically — a render channel is an `unbind`, and the scene is a bundle at every level.** This is
compose ⇄ decompose, the move the whole engine is built on, so it is not a new capability so much as an
exposure of an existing one. The scene is a bundle of object records; each object is a bundle of (geometry,
appearance); each material is a bundle of channel roles; each sim state is a bundle of field roles. "Separate
this out" is exactly the engine's *decompose* — `unbind` the role you want, `cleanup`/resonate to recover it —
and the material system **already does it**: `material.channel(name)` is literally
`unbind(record, role(name))`, pulling albedo / roughness / normal / emission / AO / opacity back out of the
bundled record at cosine ~1. The channel selection a compositor wants is the engine's core operation, one
costume over (`decompose_structure` / the resonator, pointed at render output).

The levels, and the pass each `unbind` gives you:

| Level | The bundle | Unbind by | The pass you get | Status |
|---|---|---|---|---|
| **Scene** | object records | object ID | **per-object matte** (Cryptomatte) | substrate ships (`scene` resonate-to-recover-parts) |
| **Object** | (geometry, appearance) | appearance role | the object's **material** | ships (`unbind` either side) |
| **Material** | channel roles | property role | **albedo / roughness / metallic / normal / height / emission / AO / opacity** | **ships** (`material.channel()`) |
| **Simulation** | field roles | field role | **velocity / density / temperature / pressure** | ships (`fluid.vel`/`.density`/`.temperature` are arrays) |
| **G-buffer** | per-pixel geometry | — | **normal / depth / position / motion** | ships (already computed for SVGF) |

**VSA-native how — reuse, don't rebuild.** Material-property passes read out `material.channel(name)` (shipped,
one `unbind` per selected channel). Per-object mattes are the resonator/`unbind` by object ID over the scene
bundle (the anti-aliased *coverage* version — Cryptomatte's fractional edge weight — is the per-pixel object-ID
weight). Sim-property passes read the fields directly (already separate arrays). Data passes expose the
G-buffer the renderer already computes. Each pass carries its **alpha** — the material's `opacity` channel and
the per-object coverage — so everything composites. And the **selection** is a small spec (a list of
`(level, role)` pairs) the renderer honours; **default = beauty**, unchanged. It lands as a `pipeline.Stage`
that declares the requested channels in what it `produces`.

**A small readable sketch of the shape:**
```python
# A channel is an unbind. Choosing channels = choosing which roles to unbind, at which level.
def render_channels(scene, camera, want=None):
    beauty = render(scene, camera)                    # the default composed image (unchanged)
    if not want:                                      # no spec -> just the beauty pass, exactly as today
        return {"beauty": beauty}
    out = {"beauty": beauty}
    for level, role in want:                          # e.g. ("material","albedo"), ("object", obj_id)
        if level == "material":
            out[role] = per_pixel(lambda m: m.channel(role))    # unbind the material channel (SHIPPED)
        elif level == "object":
            out[f"object:{role}"] = object_matte(scene, role)   # resonate/unbind by object id, + coverage(alpha)
        elif level == "sim":
            out[role] = read_field(scene.sim, role)             # velocity / density / ... already an array
        elif level == "data":
            out[role] = gbuffer(role)                           # normal / depth / position / motion (computed)
    return out                                        # each entry is its own buffer, with alpha
```

**The one genuine bit of new renderer work (kept honest): lighting/transport passes.** The data / ID / material
/ sim passes above are cheap — an `unbind` or an array read. But the *lighting* passes a compositor also wants
— **direct, indirect, diffuse, specular, GI, shadow, AO** — are different: `holographic_pathtrace` integrates
over *all* paths and averages, so to split them it must **accumulate a separate labelled buffer per
contribution during the trace** (tag each sample by bounce type — direct vs indirect, diffuse vs specular — and
add it to the right buffer). That is real renderer work, not an `unbind`, and it is the part to scope as its own
step. V-Ray/Arnold do exactly this; it is standard, just not free.

**Real basis + seat.** Arbitrary Output Variables / render elements (V-Ray, Arnold, RenderMan); Cryptomatte
(Friedman & Jones, 2015) for coverage-weighted ID mattes; OpenEXR multi-channel as the container. **Seat:
Pharr** (render passes / light-transport decomposition is his rendering turf) + the material/scene-decompose
thread (**Plate** — one bundle, many views).

**Bar.** The selected channels **composite back to the beauty pass** — a per-object matte set covers the frame
with no gaps or double-counting, and the diffuse+specular+… light passes sum to beauty within tolerance (the
compositor's "the passes must add up" invariant), measured. Default (no spec) is **bit-identical** to today's
render.

**Kept-negative risk (loud).**
- **Lighting passes need trace-time accumulation** (above) — real work, and making them *sum exactly to beauty*
  takes care at the boundaries (Russian roulette, MIS weights), or the passes won't add up.
- **Crosstalk on unbound channels**: `material.channel()` recovers a channel *plus capacity crosstalk* (its own
  kept negative) — fine for a matte / debug / visualisation pass, but a channel meant for exact re-lighting
  wants the clean field, not the crosstalk-carrying recovery. Mark which passes are exact vs approximate.
- **Deep compositing is a further step**: per-pass alpha is easy; *deep* data (many depth samples per pixel for
  true volumetric/transparency compositing) is a heavier format, out of scope for v1.
- **Memory**: N channels = N buffers — make it **opt-in per channel** via the spec, never all-on by default.

**Effort: medium — mostly exposure, plus the lighting-pass step.** The data / ID / material / sim passes are
wiring the shipped `material.channel` + scene-decompose + sim fields + G-buffer into a selectable multi-buffer
output on `pipeline.py`; the lighting passes are the one genuinely-new renderer piece. **The §0 audit lesson
once more**: the scene was already a bundle at every level, so "separate the channels" is mostly exposing the
`unbind` the engine already runs.

---

## 4. The honest boundaries (kept loud, per §6)

- **Bump ≠ depth.** IR1 makes a plausible *relative* height for **materials**; it is not a depth sensor and
  does not need to be. Do not let "auto bump" imply "we estimate real depth."
- **Metric monocular scene depth is the learned-prior part the constitution bans** (IR6) → abstain or require
  a supplied map; never fake it.
- **Gradient-free** (IR4): no differentiable rendering; the loop is guess-render-compare-refine, which is
  coarser and slower but native and deterministic.
- **Archetype-level detection, not semantic segmentation** (IR3): works inside a known scene vocabulary,
  abstains outside it.
- **Perceptual-metric ceiling** (IR4): SSIM-style structural comparison, not learned perceptual loss.
- **Luminance-as-height is a heuristic** (IR1): baked lighting → fake grooves, albedo-that-isn't-height →
  fake ridges; the high-pass, intrinsic split (IR9), and light-aware pass reduce but do not remove the ambiguity.
- **FFT integration assumes periodicity** (IR7): seamless for tileable *materials* (a feature), but a
  systematic bias on non-periodic *scene* surfaces — use the DCT/Poisson variant there.
- **Photometric stereo needs multiple known lights** (IR8) and a Lambertian surface — a "do it right" path for
  those who can supply the shots, not a single-image method.
- **Denoising trades variance for bias** (IR10): over-blur on glossy/thin detail, ghosting under motion; it is
  a reconstruction, not free convergence.
- **Classical style transfer is below neural** (ST1–ST3): strong for colour, texture, and material, weaker for
  free-form artistic restyle; upsampling invents plausible-not-true detail. **No GAN, and none needed.**
- **The 3D plate completes by retrieval, not invention** (IR11): it recovers the unobserved back only from
  *complete objects it has stored*, and abstains on anything the library does not cover — so the win scales
  with library coverage, and it never hallucinates a back it has not seen. That is the honest, VSA-native
  answer to the one boundary IR6 could not cross.
- **Classical acceleration is honest, not magic** (IR12/IR13): the FSR1-style upscaler cannot invent detail
  absent from the low-res input (learned upscalers can, and are non-borrowable), and checkerboard reconstruction
  is a *trade* — it costs more than a plain low-res render and can shimmer under motion. Both are cheap,
  deterministic, and real; neither is DLSS.
- **Render channels split cheaply — except the lighting passes** (IR14): data / ID / material / sim passes are a
  free `unbind` or array read, but diffuse / specular / GI / shadow passes need the path tracer to accumulate
  labelled buffers during the trace, built so they sum back to beauty. And an `unbind`-recovered channel carries
  capacity crosstalk — exact enough for a matte or a debug view, approximate for exact re-lighting.
- **The abstention is part of the product.** Auto-bump emits a bump-confidence and falls back to flat where the
  image does not support it; the scene loop matches what it can and abstains on the rest. A version that never
  abstains is the one to distrust — the same discipline as the forecasting backlog's conformal gate, which is
  the shared confidence engine underneath both.

---

## 5. Sequencing (value over effort)

**Auto-material track (ships early, mostly classical arithmetic):**
1. **IR1 — auto-bump** *(low–med; the concrete ask; ship first).*
2. **IR7 — FFT normal↔height integration** *(low; native FFT, makes IR1 consistent + seamless-tileable; do with IR1).*
3. **IR5 — displacement from a confident height** *(med; small reuse of `displace`, rides on IR1).*
4. **IR9 — intrinsic (Retinex) albedo/shading split** *(med; a cleaner IR1 bump, same FFT solver as IR7).*
5. **IR8 — photometric stereo** *(med; the "do it right" real-normal path, when multi-light shots exist).*
6. **IR2 — light-aware height** *(med; better bump once a light direction exists — from IR4 or the artist).*

**Render speed & post-fx track (classical, driven by the variance/G-buffer you already compute):**
7. **ST1 — colour transfer** *(low; instant reference grading; slots into `postfx`; also feeds IR4's mood match).*
8. **IR12 — FSR1-style upscaler (EASU + RCAS)** *(medium; the post-process upscale option — new EASU kernel + the shipped `sharpen` as RCAS; a `postfx` stage).*
9. **IR10 — SVGF-style 1-spp denoiser** *(high value; the render speedup; med–high effort but the biggest win here).*
10. **IR13 — checkerboard / sparse rendering** *(medium; shade half, recover the rest via the shipped masked-recovery solver — "larger res without taking forever").*
11. **ST3 — example-based / guided super-resolution** *(med; render small → upscale; pairs with IR10/IR12).*
12. **ST2 — example-based texture / artistic style** *(med; material synthesis + painterly post-fx).*

**Scene-matching track (the research headline):**
11. **IR3 — the perception→hypothesis bridge** → **IR4 — the analysis-by-synthesis loop** *(high; ship the self-recovery test first; ST1 grading and IR8/IR9 material estimates feed it).*
12. **IR11 — the 3D splat-bundle object archive** *(medium; mostly assembly of shipped parts — `splat` + `photo3d` + `image`'s masked-recovery + `analog` recall; it is the honest back-completion IR6 abstains on, and it warm-starts IR3/IR4 with the nearest stored scene).*
13. **IR6 — monocular scene depth** *(a boundary, not a build; keep the honest posture on record).*

**Compositing / output track (independent — useful for compositing, science, debugging):**
14. **IR14 — render channels (AOVs)** *(medium; the data / ID / material / sim passes are a near-free exposure of the shipped `unbind` — ship those early; the lighting passes (diffuse/specular/GI) are the one genuinely-new renderer step, scoped separately).*

The shared confidence engine (the forecasting backlog's conformal layer) underlies IR1's abstain-to-flat, IR4's
accept/abstain, **and IR10's variance-guided bandwidth** — build it once, and the auto-material, the denoiser,
and the scene-matching sides all delegate their "am I sure enough?" question to it. That is the through-line: the
variance the renderer already computes is the same signal that drives the confidence gate, the adaptive sampler,
and now the denoiser.

---

## 6. The recurring lesson

The probe found the usual shape: the parts are already in the box — a numpy vision front-end, a semantic
scene compositor that realizes to SDF and renders, a material stack with `normal`/`height` channels already
first-class, a displacement operator, analog recall, and iterate-a-projection — and the work is *wiring them
into a loop*, not building new machinery. The honest cut is between **auto-bump**, which needs only a plausible
relative height and ships early and clean, and **scene depth**, which needs the one learned prior the engine
refuses and so stays a bounded, abstaining item. "Auto bump when none is supplied" becomes: read the image with
arithmetic, high-pass the detail, turn the gradient into normals, drop them into the channel that was already
waiting — and flag it honestly where the picture doesn't support it. As above, so below: an image is a field, a
material is a field, a scene is a bundle of fields, and the same five operations compose and decompose all of
them.

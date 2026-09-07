
## integrations/ folder: openzoo harness integrations + audit sweep (session)
Built integrations/ (10 harnesses: OpenWebUI pipe function, LibreChat yaml, Continue yaml,
aider env, SillyTavern/AnythingLLM/Cursor/Cline README-only, Hermes yaml, GrokCLI json).
Registered pointer capability "Harness integrations for openzoo" in catalog p06 (import-only
declared negative: integrations never import lecore; HTTP to localhost:8402/v1 only). Battery 5/5.
BUGS FOUND & FIXED in sweep:
- openzoo model ids are provider-prefixed AND contain dots (nvidia/nemotron-3.5-lightning);
  original placeholder deepseek-v4-flash would 404 everywhere. Fixed in 10 files.
- OpenWebUI pipe _strip_owui_prefix split on first "." -- mangled bare dotted ids to
  "5-lightning". Fixed with dot-before-slash + digitless-head guards; pinned in selftest
  as a kept negative.
- Pipe forwarded OpenWebUI bookkeeping keys (chat_id/metadata/...) upstream; now allowlisted
  to OpenAI chat fields. Round-trip tested against a mock server (models fetch, non-stream +
  receipt, SSE stream, 402 guidance) -- all pass.
- Literal "{a,b}" junk dirs from failed brace expansion caught BY RUNNING the catalog example.
KEPT NEGATIVE: no client-side corpus spill in any integration -- spill is server-side at the
zoo so all harnesses benefit and nothing double-bills. NOT LIVE-VERIFIED: a settled paid call
(no funded wallet in this environment); receipt-in-usage shape unconfirmed -- pipe degrades
gracefully either way.


## Sweep 138 -- caustics + dispersion: the rainbow the renderer could not make

**Brief.** "Check that our rendering pipeline can handle caustics and dispersion. If it's missing,
add it. Use SOTA methods." Then, mid-sweep: "we are a holographic framework and rendering engine, so
we may be able to do things better than SOTA. Also make sure it works within the VSA application
pipeline."

**Audit first.** Caustics: PRESENT and real -- `mind.caustics` forward-traces light, refracts through
the SDF and splats with `np.add.at`. Dispersion: PARTIAL. `dispersion_spread(D, N, iors)` in
holographic_raydiff could already split a bundle across several indices, but its own test passed two
hand-picked numbers `[1/1.513, 1/1.532]` because NOTHING IN THE ENGINE COULD TURN A WAVELENGTH INTO AN
INDEX. `find_capability` returned only fallbacks for sellmeier / ior / fresnel / spectral_render /
hero_wavelength. That was the licence to build.

**Built.** `holographic/rendering/holographic_dispersion.py` -- Sellmeier n(lambda) for BK7 / SF11 /
fused silica pinned to catalogue constants, Abbe number over the Fraunhofer d/F/C lines, Hero
Wavelength Spectral Sampling (Wilkie/Nawaz/Droske/Weidlich/Hanika, EGSR 2014) as the SOTA
stratification, and `anchor_layers` as the shortcut past it. Wired as three delegating verbs
(`sellmeier_ior`, `abbe_number`, `spectral_caustics`), catalogued in p08, landed as
`applications/demoscene/prism.py`, tests in `tests/test_dispersion.py`.

**The holographic win, measured against the strongest honest baseline.** Hero sampling must trace
every stratum because it is a Monte Carlo estimator -- it cannot reuse a sample it never drew. A
deterministic tracer can. The per-wavelength layer family is extremely low rank: 99.59% of SVD energy
in ONE singular value, 99.94% in three. So k anchors reconstruct all C layers. BK7, C=16, 96x96,
against the C-trace ground truth (not against one trace, which would be a strawman):

| k | speedup | rel RGB err | chromatic saturation retained |
|---|---------|-------------|-------------------------------|
| 2 | 6.1x    | 0.0116      | 99.6%                         |
| 3 | 4.9x    | 0.0072      | 100.1%                        |
| 4 | 3.8x    | 0.0085      | 99.8%                         |
| 8 | 1.9x    | 0.0030      | 98.8%                         |

`anchors=None` is the default and is byte-identical to the un-accelerated path.

**The zero baseline that makes it a result.** A monochrome caustic through the same observer has
saturation EXACTLY 0.0 -- not approximately; one wavelength cannot be two colours. The spectral render
scores 0.4032 on BK7. There is no baseline argument to have.

### KEPT NEGATIVES

**1. THREE METRICS IN A ROW FAILED THE SAME WAY, and each one nearly killed a true result.** Centroid
separation returned 0.0000 px (a centred sphere under axial light is radially symmetric, so every
wavelength's centroid sits at the centre). Radial peak radius returned 61.0 px for every wavelength
(binning radial SUMS lets the outermost annulus win on area alone -- an area-weighting bug in my own
instrument). Spot RMS radius returned 52.00 px for every index (correct, but measuring a focal SPOT as
though it were a ring). All three said "dispersion is invisible" while the raw images differed by 43%
per-pixel. The fix was never a cleverer scalar: it was to MEASURE IN THE OUTPUT SPACE. Grey is grey.

**2. A SMALL REL-ERR DOES NOT CERTIFY A RECONSTRUCTION -- and this one was written up as a feature
before the code contradicted it.** The first interpolator bracketed targets between anchors picked in
HERO ORDER. Hero wavelengths arrive ROTATED, not sorted, so `searchsorted` ran on a non-monotone axis
and interpolated between the wrong pair, laying ghost energy where no wavelength focused. It inflated
saturation to 108.5% of ground truth while rel-err stayed a respectable 0.0236 -- it read as a mild
accuracy cost and was a WRONG PICTURE. It was documented in three places as a real property of k=2
("two anchors overshoot") before `prism._selftest` assertion 5, written to REQUIRE the overshoot,
failed against the corrected sorted-axis code. All three sites were retracted. Two lessons kept: the
pedestal dominates the norm, so rel-err cannot see the interesting part; and **a negative found with a
throwaway probe is a claim about the probe until the shipped code reproduces it.**

**3. A GLASS DOES NOT WANT TO BE A HYPERVECTOR.** The obvious VSA move is to bake n(lambda) into an FPE
function hypervector so a glass composes with bind/bundle like any other object. Measured, it does not
pay. Baking n(lambda) directly gives correlation **-0.25** -- worse than useless: the curve is 1.49%
variation on a 1.52 mean, so the constant dominates the superposition and the dispersion sits at
crosstalk level. Subtracting the mean first fixes the SHAPE (corr +0.97, and dim 8192 is no better than
2048 -- the limit is the KDE kernel, not capacity), but the FPE query is a SIMILARITY readout, not a
calibrated value: the recovered slope is 15x too steep, and a two-point affine refit against the F and
C lines still leaves 38-42% error across the curve's own range. 4096 floats and a calibration step to
approximate three multiply-adds. Sellmeier stays a closed form. **General lesson: before baking any
field into an FPE vector, check its variation against its mean -- a near-constant function has nearly
nothing for the encoder to hold.**

**4. `reorganize_repo.py` DOES NOT EXIST in this checkout**, so build-loop step 5a as written in the
session instructions cannot run. Substituted `compileall` over holographic/ + applications/ + tests/,
which is the honest equivalent. The instruction should be updated or the tool restored.

### Deltas
- +1 module (holographic_dispersion), +3 mind verbs (2373 total), +2 catalog cards, +1 application
  (prism -- demoscene, 7 in the library), +1 test file (17 tests).
- Audits: reachability 0 duplicates, catalog_gaps 0, skill_lint 0 (one 713-char card regression caught
  and trimmed to budget).
- HTTP round-trip proven: /tools exposes all three verbs of 2348; /invoke returns 200 for
  sellmeier_ior, abbe_number and app_run('prism').

### Three red tests found in the full run, none caused by this sweep, all three fixed

The full suite (6,910 collected) surfaced four failures. One was mine and trivial: `applications`
pins its own registry COUNT so a silently dropped application fails loudly, and adding `prism` made
it 7. The count was bumped. The other three were pre-existing and are worth recording because two of
them are repeat offenders of patterns already on file.

**`test_no_unreviewed_public_name_collisions` -- `signature_of` in codestructure vs shapeprobe.**
Bodies read, as the audit demands. codestructure.signature_of takes an AST NODE and derives a call
shape from SOURCE without importing (that is the point -- the merge census must read a file it cannot
safely execute); shapeprobe.signature_of takes a LIVE CALLABLE and uses inspect.signature. Neither can
delegate to the other: one has no object, the other has no source. Accepted into KNOWN_COLLISIONS with
that reason rather than forced into a fake unification.

**`test_ambiguous_queries_collapse_the_margin` -- THE SIXTH INSTANCE of "a test that failed because
the work succeeded", and its own neighbour already carried the fix.** It asserted
`r_amb["flip_rate"] > r_norm["flip_rate"]`, a STRICT inequality on a quantity that is 0.0 for both
samples whenever 8 bits is comfortably enough for the shipped index. At 794 rows the mechanism it
exists to check passes handsomely -- ambiguous margin 0.064 against normal 0.567, an 8.9x collapse
against a 5x gate -- and the test still went red, because 0.0 > 0.0 is false. It was failing BECAUSE
THE QUANTIZER IS DOING WELL. The margin collapse is the mechanism; flip rate is a downstream
consequence only observable once the index is dense enough to have any flips at all. Now asserted
non-decreasing and reported. **The rule stands and has now cost six tests: pin the contract, never
the size of the mess.**

**`test_the_gather_unit_has_constant_marginal_cost_in_n` -- load-flaky; passed 3/3 in isolation,
failed under `pytest -n 4`.** Its gate was already a RATIO rather than a wall-clock ceiling, which is
the hard-won half. What it lacked was PAIRING: one unpaired `marginal(256)` call means a single
contention spike decides the verdict. Now the median of three paired ratios -- exactly the remedy
infinite_zoom paid two flaky tests to learn, applied here instead of rediscovered a third time.

**Suite after: 6,910 collected, 4 fixed, 0 failing.** The three pre-existing failures were not in this
sweep's blast radius, but shipping a zip with a red suite would hand the next session a false baseline.

### One instruction-vs-reality gap worth fixing upstream
Build-loop step 5a calls `reorganize_repo.analyze_repo` -- **`reorganize_repo.py` is not in this
checkout**, so that step cannot run as written. Substituted `compileall` over holographic/,
applications/ and tests/ (all clean). Either restore the tool or update the instruction.


## Sweep 139 -- the inverse door: RGB -> spectrum, and the 9 MiB table that did not need to exist

**Brief.** Same standing ask, re-run: sweep caustics + dispersion, add what is missing, use SOTA, look
for the holographic angle, make it work in the VSA application pipeline.

**Audit first, and this time most of it came back PRESENT.** `find_capability` on twelve user-phrasings
found: fresnel -> `fresnel_schlick` in holographic_brdf (real, tested); total internal reflection ->
`refract` (real); thin-film iridescence -> `holographic_thinfilm` with `interference_reflectance` and
`thin_film_tint` (real, tested); polarisation -> Stokes state (real). Caustics and dispersion, landed
last sweep. **One phrasing returned only unrelated fallbacks: "convert an RGB colour to a spectrum".**
The engine had spectrum -> colour and wavelength -> index, and no way back. That is the licence.

**Why the gap mattered more than it looked.** Every material in every scene is authored in RGB. Without
RGB -> spectrum the whole spectral stack only worked on things whose spectrum was already known by
name -- you could render a rainbow through BK7, and nothing an artist had actually picked.

**Built.** `holographic/rendering/holographic_spectralup.py`, four verbs (`rgb_to_spectrum`,
`rgb_to_spectrum_coeffs`, `spectrum_from_coeffs`, `reflectance_to_rgb`), catalogued (8/8 discoverable),
`tests/test_spectralup.py`, and `applications/demoscene/spectral_glass.py` -- the first program that
needs both this sweep and the last one.

**SOTA adopted, verbatim and credited.** Jakob & Hanika, *A Low-Dimensional Function Space for Efficient
Spectral Upsampling*, CGF 38(2) / Eurographics 2019: `f(lambda) = S(c0*t^2 + c1*t + c2)` with the
algebraic sigmoid `S(x) = 1/2 + x/(2*sqrt(1+x^2))`. Three coefficients; bounded in [0,1] by
construction, so a fitted reflectance cannot invent energy.

**THE HOLOGRAPHIC ANGLE, and it is lever 3 verbatim -- determinism instead of storage.** Their fit needs
CERES (Levenberg-Marquardt + forward-mode autodiff) and is far too slow inline, so the method ships a
PRECOMPUTED TABLE: three 64^3 cubes, ~9 MiB. leCore may not depend on CERES or autodiff and would rather
not ship 9 MiB of binary. It does not have to. **A quadratic through three points is exactly
determined** -- inverting the sigmoid at three anchor wavelengths turns the fit into a 3x3 Vandermonde
solve, and LM over three unknowns with a numerical Jacobian finishes it. Same "iterate a projection"
family as IK, PBD, PnP and the resonator; no new machinery.

Measured, 400 random sRGB colours against the engine's own observer:

| | Jakob & Hanika 2019 | this |
|---|---|---|
| round-trip error | zero on sRGB gamut | max **9.4e-13**, median 1.2e-14, 400/400 < 1e-6 |
| table | 9 MiB (3x64^3) | **0 bytes** |
| dependencies | CERES + autodiff | NumPy |
| per-colour cost | table lookup (~ns) | **0.64 ms** |

### KEPT NEGATIVES

**1. 0.64 ms IS NOT A TABLE LOOKUP, and that is the entire trade.** The 9 MiB exists precisely to make
this free at render time. Fitting per pixel would be catastrophic -- 1080p is 2M fits, over twenty
minutes. This is a BAKE-ONCE capability and nothing else: fit each MATERIAL once (a scene has tens),
keep the three coefficients, and evaluation is then the same ~6 flops the paper quotes. `fit_palette`
de-duplicates and `spectrum_of` is the cheap path, kept as a separate function so the expensive one is
not reached for by habit. **We did not beat SOTA on speed; we beat it on storage and dependencies, at
equal accuracy, and the loss is stated in the same breath.**

**2. PURE GAUSS-NEWTON DIVERGES EXACTLY WHERE THE PAPER SAYS IT WILL.** Without adaptive damping, 59 of
400 colours failed (worst round-trip error 0.98), and the failing set was measurably the SATURATED one:
mean channel spread 0.67 against 0.52 for the whole sample. The paper's own reasoning explains it --
gamut-rim spectra must approach MacAdam box shapes, and a smooth quadratic seed starts far from a box.
LM damping plus three deterministic multi-start anchor triples fixed all 59 **and was FASTER** (0.64 ms
vs 1.0 ms), because it converges in fewer steps. A diverging solver is not a slow solver.

**3. THE ILLUMINANT IS NOT OPTIONAL, and omitting it silently breaks white.** Reflectance is not colour;
it becomes colour only under a light. Integrating against the bare observer and normalising by Y sent a
flat unit reflectance to RGB [1.198, 0.950, 0.907] -- because this observer is near equal-energy (flat
XYZ [0.998, 1.000, 0.999]) while the sRGB matrix expects D65. White then carried a 0.198 error **that no
solver tuning could remove, because the target was unreachable**. Fixed with a von Kries row scaling
pinning flat reflectance to exactly [1,1,1]. Pinned by a test.

**4. THE CORNERS ARE ASYMPTOTIC -- a bound, not a bug.** Exactly white needs f == 1 at every wavelength,
which S only approaches, so (1,1,1) and (0,0,0) land at ~1e-6 with saturated coefficients while 0.999
grey fits to 8e-16. Property of the function space, the paper's too. Pinned separately so nobody
"fixes" it by loosening the 1e-9 contract for every other colour.

**5. THE ENGINE'S OWN FORWARD PATH RETURNS BLACK FOR A REFLECTANCE.** `spectrum_to_rgb` is built for
EMISSION: mode='none' divides XYZ by a hardcoded **1.5e13**, calibrated for blackbody radiance. A
reflectance in [0,1] integrates to XYZ ~24, so 24/1.5e13 clips to zero -- black, with nothing to say
why. Neither function is wrong; radiance and reflectance are different physical quantities. But a user
calling `rgb_to_spectrum` then `spectrum_to_rgb` hits it immediately, so `reflectance_to_rgb` was added
as the explicit forward partner (round-trips to 1e-16) and a test pins the trap so the two doors are
never quietly merged.

**6. BLENDING COEFFICIENTS IS NOT BLENDING SPECTRA.** Lerping two 3-coefficient records gives a
physical spectrum reading as the right intermediate colour (red -> purple -> blue) but differing from
the linear spectral mix by up to **0.36** at t=0.75. Note carefully: validity is NOT the advantage --
a convex combination of two valid reflectances is always valid too. The advantage is **30x compression
(3 floats vs 90 samples)** and that is the whole of it. If you need a pigment mix, mix the spectra.
Pinned by an assertion that the divergence still EXISTS, so it cannot silently become free.

### ABOVE / BELOW SWEEP -- all four directions, measured not asserted
- **DOWN** (components of its own input): an (8,8,3) RGB texture -> (8,8,3) coefficients, 64 unique
  fits de-duplicated, per-pixel round trip 4.3e-10.
- **UP** (its input as a component of something larger): tint -> dispersive caustic through
  `spectral_glass`; the tint reaches the image (red share 0.334 -> 0.616, blue 0.316 -> 0.146) and the
  dispersion survives it (saturation 0.109 -> 0.763).
- **SIDEWAYS** (costumes): field ((8,8,3) -> (8,8,90) hyperspectral), structure (a material as 3
  floats), sequence (a 5-step material morph, all samples in [0,1]), program (all four verbs 200 OK
  over /invoke).

### Deltas
- +1 module (796 total), +4 mind verbs (2377), +1 catalog card, +1 application (spectral_glass --
  8 in the library, demoscene now four), +1 test file (11 tests).
- Audits: reachability 0/0, catalog_gaps 0, skill_lint 0. HTTP: /tools 2352, all four verbs plus
  app_run('spectral_glass') return 200.

### Two red tests in the full run (6,759 passed) -- one mine and correct, one the third repeat

**`test_unmentioned_verbs_do_not_grow` -- MINE, and the gate did its job.** Four new verbs, none
mentioned in any human doc; the gate offers "document the new ones or --rebase WITH A REASON" and the
right answer was to document. `RENDERING_GUIDE.md` gained section 10 (spectral rendering: wavelength ->
index, caustic with its colour, RGB -> spectrum, and the three composed), which took unmentioned verbs
1988 -> 1980, under the 1984 budget -- it documented last sweep's three as well, which had also been
undocumented. Every snippet in the new section was RUN, not just written; that is how the
`spectrum_to_rgb` correction below was caught.

**A correction the guide-running caught.** The write-up said `spectrum_to_rgb` "returns black" for a
reflectance. That is only its `mode='none'` path. Its DEFAULT `mode='hue'` returns
**[1.0, 0.483, 0.472]** for a reflectance fitted to (0.8, 0.2, 0.2) -- a plausible-looking, wrong
colour, which is the more dangerous failure of the two. Corrected in the guide, the module docstring
and the test, which now pins BOTH modes: the loud failure and the quiet one.

**`test_kept_negative_a_cheap_function_of_a_large_array_loses` -- load-flaky, pre-existing, and the
THIRD occurrence of one pattern.** Three wall-clock measurements compared against each other, taken
once each: passed 3/3 alone, failed under `-n 4`. Same remedy as before -- sample all three together,
three times, and gate on the median of each RATIO. The ordering claim was real; the single sample was
not. **Running tally: infinite_zoom (twice), the gather unit, and now this. An unpaired wall-clock
comparison is a coin flip wearing a lab coat, and this repo has now paid for that lesson four times.**

Suite after: **6,780 passed, 0 failing.**


## Sweep 140 -- renders: dispersion + caustics in the material-preview environment

**Brief.** "I would like to see renders that demonstrate dispersion and caustics. Make sure the
rendering environment is our default preview render environment that we use for material previews."

**Audit first.** `find_capability("preview a material")` -> holographic_preview.material_ball: the
material ball, orthographic camera down -z onto a unit sphere at the origin, one directional light at
(0.6, 0.7, 0.5), Cook-Torrance, background gradient, Reinhard tone-map. That is the environment. Every
render below uses that sphere and that presentation.

**Built (all additive, all defaults byte-identical):**
- `material_ball(linear=True)` -- radiance before the tone-map. Needed because a spectral render must
  combine per-wavelength results in LINEAR radiance and tone-map ONCE: x/(1+x) is not additive, so
  tone-mapping each wavelength and summing after is a different and wrong image.
- `spectral_material_ball` (+ mind verb, catalogued, 4/4 discoverable) -- the preview environment
  shaded once per hero wavelength. Measured max 0.151 difference against the RGB ball.
- `caustics(center=(x,z), window=)` (+ threaded through `spectral_caustics`) -- see below.

### THE CAPABILITY GAP THE RENDERS EXPOSED
`extent` was doing two unrelated jobs: it sized the emitter ray grid AND the receiver window, both
centred on the origin. So a caustic thrown off-centre by an oblique light could only be brought into
view by WIDENING -- which widened the light source to match, spreading the same rays over more area and
washing the caustic out. Chromatic separation is a fixed world-space distance (measured 0.049 world
units between 400nm and 700nm through SF11), so it is only visible at magnification, and zooming was
exactly the thing that could not be done. `window=` sets the receiver half-width alone; `center=` sets
where it looks. Defaults reproduce the original exactly, pinned by an assertion.

### THE RESULT -- dispersion, measured rather than admired
Spot radius of the caustic core at a FIXED receiver, SF11, one number changed:

| wavelength | n (SF11) | spot radius |
|---|---|---|
| 420 nm | 1.8333 | 25.28 px |
| 470 nm | 1.8110 | 19.65 px |
| 520 nm | 1.7966 | 17.98 px |
| 580 nm | 1.7865 | 15.39 px |
| 680 nm | 1.7736 | 16.25 px |

Monotonic across the visible band and a **36% change in spot size**, driven entirely by the Sellmeier
index. Blue focuses shorter, so at a fixed plane it is past focus and broader. Chromatic saturation:
BK7 0.398, SF11 0.269, **monochrome exactly 0.0000**.

### KEPT NEGATIVES

**1. THERE IS NO `glass=` ON THE MATERIAL BALL, AND THE FIRST CUT SHIPPED ONE.** It modulated each
wavelength by the glass's Sellmeier index, to show dispersion on the ball. Measured difference against
no glass at all: BK7 **0.00028**, SF11 **0.00101** -- invisible on a [0,1] image. The reason is
physical and total: an opaque Cook-Torrance BRDF has NO TRANSMITTED PATH, so there is no refraction for
an index to bend and the modulation collapses to a near-constant gain. The parameter was REMOVED rather
than shipped. **Dispersion lives in the transmitted path**; the ball's honest spectral job is
reflectance.

**2. THE PREVIEW'S OWN LIGHT IS A POOR CAUSTIC LIGHT, and that is a fact about product photography.**
Its studio lighting is oblique (62-66% downward). Through a sphere that gives a smeared caustic --
**peak 4.0x mean, thrown ~6 world units off-centre** -- against **20.4x** for a near-axial light. Both
were rendered. This is why glass is not lit like metal, and it is reported rather than quietly swapped.

**3. TWO CONVENTIONS THAT COST REAL TIME.** The preview's `light_dir` is a SHADING vector (surface ->
light); `mind.caustics` wants the direction light TRAVELS. Passed through unnegated, every ray heads
away from the receiver and the render's max intensity is **exactly 0.0** -- a blank image with no error.
And locating the caustic by THRESHOLD CENTROID twice reported a core in empty sky, because scattered
unrefracted rays drag the centroid; locating by PEAK at successively finer windows is what worked.

**4. THE HUE-LIFT INVENTS COLOUR AT THE EDGES OF VISION.** The first filmstrip ran 380-713nm and
rendered 380 as blue and 713 as yellow. Where the CIE curves nearly vanish, `spectrum_to_rgb`'s
mode='hue' normalisation divides by almost nothing and produces a saturated, wrong hue. Restricted to
420-680nm, where the hues are honest (violet -> blue -> green -> yellow -> red).

### Deltas
- +2 mind verbs (spectral_material_ball, and caustics/spectral_caustics gained center=/window=),
  +1 catalog card, +1 preview mode (linear=). Audits 0/0/0. 9 renders under renders/.


## Sweep 141 -- dispersion in the VIEW path: Blender parity, honestly scored

**Brief.** "This is the dispersion I was looking for. We need parity with Blender, and they recently
added dispersion, and it's fast. Their caustics also looks great." (with a Cycles dispersion-glass
reference image)

**The reference is a DIFFERENT LIGHT PATH from what sweeps 138-140 built.** Those did the LIGHT path:
light -> glass -> receiver, the caustic. Blender's dispersion-glass shot is the CAMERA path: eye ->
glass -> world, where every internally refracted edge splits into rainbow fringes. `find_capability`
on eight phrasings ("dispersive glass shader", "see through a glass object", "dielectric bsdf",
"render glass with rainbow edges") returned only fallbacks. Real gap, and the visible half.

**What existed to build on:** the path tracer already carries a smooth-dielectric BSDF with a
PER-POINT IOR in its material tuple. Nothing needed rewriting -- the wavelength's index is swapped in
wherever the caller's own material said "this is glass", so an existing scene becomes dispersive
unchanged.

**Built:** `cauchy_from_abbe` / `cauchy_n` (+ `mind.cauchy_ior`) and `spectral_render` (+
`mind.dispersive_render`), catalogued (6/6 discoverable), 6 new tests.

### PARITY, SCORED HONESTLY

**Blender's dispersion is RGB-BASED, not spectral** -- three channels with three indices, per 80.lv's
report of the Cycles PR ("an RGB-based implementation rather than a true spectral one"). So the
parity table is not the one-sided loss it looks like:

| | Blender Cycles | leCore |
|---|---|---|
| spectral model | 3 channels (RGB) | N hero wavelengths + Sellmeier + CIE observer |
| artist parameterisation | IOR + dispersion slider | IOR + Abbe (`cauchy_ior`) -- **same numbers transfer** |
| hardware | GPU (OptiX/CUDA) | NumPy, CPU |
| cost | seconds | **~107 s per wavelength** at 288x288 spp40 |

Rendered both at 288x288 spp40: C=3 (Blender's channel count) took 329 s and scores chromatic
saturation **0.4648**; C=14 (true spectral) took 934 s and scores **0.3462**. The three-channel
version is HARSHER, not better -- garish RGB-ish fringes against the spectral version's smooth
gradation, mean |difference| 0.033 with a 0.50 peak. **We are spectrally more correct and an order of
magnitude slower, and the slowness is architecture (NumPy CPU vs a production GPU renderer), not
something a sweep closes.** Adopting Blender's Abbe parameterisation means a material authored there
transfers by its numbers rather than by eye.

### KEPT NEGATIVES

**1. THE ANCHOR TRICK DOES NOT TRANSFER TO THE VIEW PATH, and the reason is instructive.** Caustic
layers are a deterministic rank-3 family (99.94% of SVD energy in three singular values) which is what
bought 4.9x in sweep 138. Path-traced layers carry MONTE CARLO NOISE, and **noise is full-rank by
construction**: measured over 8 wavelengths the spectrum runs 0.9938 / 0.9950 / 0.9960 / 0.9969 -- a
long flat tail that is noise, not signal. Interpolating between two noisy renders reproduces neither
the signal nor the noise. `anchors=` is passed through for deterministic tracers but is NOT the
default here and must not be assumed free. **The lever was a property of determinism, not of
dispersion.**

**2. DISPERSION IS INVISIBLE WITHOUT EDGES TO SPLIT, and this nearly read as a broken feature.** The
first view-path render came out grey -- chromatic saturation 0.25, visually nothing -- under a smooth
gradient sky. The view path splits colour only where the REFRACTED IMAGE HAS EDGES; a featureless
environment gives every wavelength the same picture. Against a studio sky with HDR ceiling panels the
same scene measures **0.46**. The preview environment's own sky docstring already recorded half this
lesson ("a flat sky renders [glass] as flat discs"); dispersion needs it twice over. **Light the scene
with structure, or there is nothing to disperse.**

**3. MORE WAVELENGTHS LOWERS THE SATURATION NUMBER, which is the opposite of "more is more".** C=3
scores 0.4648 and C=14 scores 0.3462 -- and C=14 is the better image. Saturation measures how far
apart the channels are, and three coarse strata put them further apart than fourteen fine ones do.
A metric that rewards the cruder render is a metric to read carefully, not a leaderboard.

### Deltas
- +2 mind verbs (cauchy_ior, dispersive_render), +1 catalog card (18 in p08), +6 tests (25 in
  test_dispersion). Audits 0/0/0. Full suite **6687 passed, 0 failed**.


## Sweep 142 -- lighting the scene so BOTH effects are visible, through the adaptive pipeline

**Brief.** "The lighting is too dim, and there's no caustics visible. I want to see dispersion and
caustics. Fix the lighting of the environment to make it more appropriate. Make sure we use the
adaptive render pipeline including the post effects and upscaling."

**Audit first.** `find_capability` found the pipeline already complete and already composed:
`path_trace_adaptive` (CI-driven, stop when the pixel is proven), `render_specimen` (trace -> G-buffer
-> SVGF denoise -> firefly clamp -> searched exposure, ONE call), `post_process` (a named PostChain),
`upscale` (FSR1-style EASU+RCAS), plus `svgf_denoise`, `clamp_fireflies`, `aces_tonemap`, `sky_model`.
Nothing needed building for the pipeline itself -- what was missing was knowing how to DRIVE it, and
three of the four things learned were traps in tools that were working correctly.

### THE ONE THING THAT ACTUALLY NEEDED BUILDING
**Brute-force path tracing is the wrong algorithm for a caustic**, and no amount of lighting fixes it.
The tracer's own kept negative says why: no next-event estimation, so light is gathered only when a
bounce happens to hit an emitter -- and a caustic needs a small bright source, which is exactly what
random hemisphere sampling almost never hits. Measured while trying: **pushing the key panel from 95
to 220 to strengthen the caustic made the raw grain WORSE, 0.76 -> 1.19.** Every extra stop
concentrates the same light into fewer, hotter, rarer paths. That is the estimator, not a setting.

So: each path gets the algorithm built for it, and they are composited -- what production renderers do
with a photon-mapped caustic pass. New `holographic/rendering/holographic_causticpass.py` (+ verbs
`caustic_pass`, `composite_caustic`, catalogued 5/5, 11 tests). The projection is EXACT: a plane is
analytic, so every pixel's floor hit is a closed-form ray/plane intersection -- no G-buffer, no
reprojection error.

Also added, both additive and default-inert: `caustics(refracted_only=)` and `material_ball(linear=)`.

### KEPT NEGATIVES -- four traps, three of them in tools behaving correctly

**1. `aces_tonemap` HAS AUTO-EXPOSURE AND IS THEREFORE SCALE-INVARIANT.** Every hand-computed white
point and exposure multiplier fed to it was silently ignored -- measured, scaling its input 0.25x and
4x changed the output by under 0.001. Its actual control is `key`, the middle-grey target. The default
0.18 is the standard and is WRONG for a studio plate that is 90% dark background: it lifts black to
0.38 grey, which is exactly the "washed out" look. Lower the key; do not raise the exposure.

**2. THE POST CHAIN OWNS THE TONEMAP.** `post_process`'s default chain is
`exposure -> bloom -> aces -> chromatic_aberration -> vignette -> film_grain -> gamma`. It takes LINEAR
HDR. Tonemapping first and then running it applies ACES TWICE and lifted p50 from 0.108 to 0.415. The
chain is now explicit: `film_grain` dropped (it puts back the grain SVGF just removed) and `bloom`
turned up, because a caustic is focused light and bloom is what makes focused light read as BRIGHT
rather than merely white.

**3. THE EMITTER APERTURE'S OWN SHADOW, and it read as a rendering bug for three renders.** The
caustic emitter is a SQUARE grid of parallel rays. Rays that MISS the object carry straight on and
splat as a hard-edged bright QUADRILATERAL across the floor -- correct for a standalone caustic image
(that quad is the surrounding lit floor) and wrong for a composite, where it both double-counts direct
light the beauty pass already has and paints a rectangle. **Baseline subtraction did not remove it,
because a shape is not an offset** -- that was the wrong hypothesis, tried and discarded before the
emitter grid was suspected. Fixed with `refracted_only=True`: lit fraction 0.997 -> 0.150, rectangle
gone, and it is documented as NOT optional for compositing.

**4. THE SPECKLE IS UNDER-SAMPLING AND FILTERING IS NOT THE CURE.** Spectral rendering divides the
sample budget by C, so each wavelength gets 1/C the rays and each layer's surviving fireflies get
TINTED differently by the observer -- the sum carries coloured speckle no single layer showed.
Denoising the COMBINED frame against the shared G-buffer removes 34% while keeping 84% of highlights
(levels=3; levels=5 and 7 keep only 64% and 60% -- edge-aware is not the same as free). Clamping
harder was tried and REJECTED: pct 97 cut speckle 2.6x but clipped p99 from 0.920 to 0.377, i.e. it
removed the highlights the render is of. The fix for noise is samples.

**5. A PERCENTILE NEEDS ENOUGH SAMPLES, found by a test rather than by a render.** `composite`
normalises by the pattern's 99th percentile (its peak is a few colliding splat cells -- an outlier).
Injecting one 500x spike moves the normalised body by 99.8% at 65 lit pixels and by exactly 0.0% at
200 or more. Real caustics light thousands, so this bounds degenerate inputs -- but a 16x16 test
pattern IS degenerate, and finding that out from a test beat finding it out from a render.

### LIGHTING, the actual answer
Both effects want the SAME thing -- high contrast -- and a bright uniform sky kills both: at sky mean
0.68 the floor sits at 0.57, so a caustic (a RATIO above its surroundings) has nothing to stand out
from and the glass has no dark background to fringe against. The rig is now a dark surround with small
very bright panels: **2.5% of the sphere emitting, 180:1 contrast**, key/fill/rim. Beauty and caustic
get their own rigs on purpose -- one is lighting an object, the other is casting a pattern.
Also measured: swapping the solid gem for a torus dropped the beauty pass p99 from 2.59 to 0.14 on
identical lighting. A ring intercepts far less environment; the shape and the rig are not independent.

### Deltas
- +1 module (797), +4 mind verbs (caustic_pass, composite_caustic, + refracted_only/linear params),
  +1 catalog card (19 in p08), +11 tests. Audits 0/0/0. Full suite **6618 passed, 0 failed**.
- RENDERING_GUIDE gained 10f (caustic pass) and 10g (lighting a glass render); snippets run.


## Sweep 143 -- a render that outlives its process: progressive buckets over the existing job machinery

**Brief.** "Renders should be able to be paused and resumed, so we can render higher quality and higher
resolution images even with a limited environment. We have distributed rendering that can run in a
separate process also."

**Audit first, and almost everything already existed.** `find_capability` found `job_pause` /
`job_resume` / `job_cancel` (checkpointed, "survives an app restart"), `local_pool` (real worker
PROCESSES, own interpreter, own GIL), `farm` (cross-machine), and holographic_jobs' whole model:
buckets + a MONOID reducer + a JSON checkpoint. Nothing about that needed building.

**The gap was stated by the previous work, in its own docstring.** `job_submit` says of itself:
"ATOMIC. One bucket, so progress is 0 then 1, and job_pause/job_resume cannot split the call ... do not
expect a partial render." Submitting a render as a job made it ONE opaque unit. **A render was not
expressed as buckets.**

**The fix changes the shape of the render, not the machinery.** Monte Carlo samples are iid, so an
image is a MEAN of batches: order does not matter, batches can be computed anywhere at any time, and
they combine by addition. That is exactly (buckets, sum, checkpoint). Split the SAMPLES, not the image,
and pause/resume/restart-survival/farm all come free. New
`holographic/rendering/holographic_progressive.py` + verbs `render_progressive`,
`progressive_preview`, `progressive_render_for`, `progressive_checkpoint`, `progressive_restore`.
Catalogued 7/7, 16 tests.

### THE CONTRACT IS EQUALITY, NOT SIMILARITY
- Paused at 5/10 buckets and resumed: **byte-identical** (max|A-B| = 0.00e+00) to an uninterrupted run.
- Paused in one process, checkpointed, finished in a **completely different process**: byte-identical.
  `persisted: True`.
- `workers=2` on separate processes: **byte-identical** to serial (atol=0), 10.4s -> 5.9s = **1.76x on
  a 2-core box, 88% parallel efficiency**. (4 workers gave 1.91x on those same 2 cores -- that is
  oversubscription, not scaling, and is reported as such rather than quoted as 1.91x.)
- Monoid check against a 256-spp reference: 8 progressive buckets track single-shot at every step
  (0.0368/0.0263/0.0220/0.0197/0.0180/0.0167/0.0156/0.0150 vs 0.0372/0.0272/0.0228/0.0203/0.0185/
  0.0173/0.0163/0.0157). Progressive is ~4% ahead throughout -- not claimed as a win, only as evidence
  that splitting costs nothing.

**Demonstrated end to end:** a 340x265 render at **576 spp assembled across FOUR separate process
invocations** of 80-100s each -- a quality this container cannot reach in one sitting. Grain 0.020,
against 0.55 raw / 0.26 denoised at ~50 spp in sweep 142.

### KEPT NEGATIVES

**1. EVERY BUCKET NEEDS ITS OWN SEED, and getting it wrong fails SILENTLY.** The tracer is
deterministic in its seed -- verified, same seed gives a byte-identical image, and that is what the
engine's whole reproducibility rests on. It also means N buckets at one seed return N copies of the
SAME image, whose mean is that image: an hour of rendering with the noise of one bucket, and a progress
bar saying it worked. `sample_buckets` exists so this cannot be done by hand.

**2. A MATERIAL CALLBACK CANNOT BE CHECKPOINTED -- the real boundary of restart survival.** The scene
survives (SDF -> DSL string -> SDF, verified bit-exact on a twisted/rotated/translated torus, not just
a sphere), the camera survives (its numbers), and a sky survives when given as parameters. A material
is an arbitrary Python function of surface position and there is no honest way to serialise one.
Inventing a mini-language for it would be a worse version of scene documents, which are JSON by
construction. So it passes through untouched, the render RUNS, and the job reports persisted=False --
the manager's existing honest degradation rather than a new failure. **The demo render came back as
the default grey scene precisely because of this, which is how the limit was found.**

**3. EQUAL BUCKETS, ON PURPOSE.** Equal spp keeps the readout a plain mean and the reducer the stock
"sum". Unequal buckets need a weighted mean -- a weight carried through the checkpoint, a new reducer,
a wider format, and a new way to be subtly wrong. The cost is that spp granularity IS the bucket size.

**4. A JOB ON ITS OWN BACKEND WAS INVISIBLE TO THE JOB VERBS, and my own change caused it.** A render
started with workers>1 runs on a pool-backed manager, but `job_pause`/`job_resume`/`job_status`/
`job_result`/`job_list` all reached into `self._job_manager` directly -- so pausing a distributed
render raised KeyError. Caught by testing the feature rather than the unit. All six now resolve which
manager owns the job; `job_list` unions both.

**5. `render_preview` WAS ALREADY TAKEN.** The first cut of this work shadowed an existing faculty (a
fast rough scene look) -- a duplicate definition, which the reachability audit calls a HARD ERROR. The
runtime caught it first, as a TypeError about a missing `camera` argument. Renamed to
`progressive_preview`; audit clean.

### Deltas
- +1 module (798), +5 mind verbs, +1 catalog card (20 in p08), +16 tests, RENDERING_GUIDE 10h
  (snippets run). Audits 0/0/0.

### Two red tests in the full run (6869 passed) -- both fixed, and the second is the FIFTH of its kind

**`combine` collided between holographic_progressive and holographic_shader -- mine, and renamed
rather than registered.** shader.combine blends shader pipelines into one transfer; mine averaged
render buckets. Genuinely different, so a KNOWN_COLLISIONS entry would have been defensible -- but the
name is generic and mine was the newcomer, so it became `combine_buckets`, which is also what it
actually does. **A collision you can rename away is not a collision worth budgeting.**

**`test_place_unit_runs_on_measured_numbers` -- load-flaky, and the FIFTH unpaired wall-clock
comparison this repo has paid for.** `spec_sheet` MEASURES itself, which is the point of it (it refuses
to quote a comment) -- but that puts every verdict on wall-clock timings taken on a shared box, and
under `pytest -n 2` they shift enough to flip a threshold. Passed every time in isolation. Now a
majority vote over three sheets, the same remedy as infinite_zoom (twice), the gather unit, and the
memoize cost model. **Running tally: five. The claims are real; a single sample of them is not.**

Suite after: **6871 passed, 0 failed.**


## Sweep 144 -- why the dispersion was invisible: four causes, three of them mine

**Brief.** "This doesn't show dispersion, the quality is low, there is weird stuff going on, also not
really seeing caustics. Search online for examples from Blender."

All four complaints were correct, and they had four different causes.

### 1. THERE WAS NO DISPERSION IN THAT IMAGE AT ALL
Sweep 143's glass render traced a **single ior (1.62)** and accumulated buckets of the SAME wavelength.
Progressive accumulation got built and the spectral path was dropped on the floor in the same sweep
that built it. Nothing subtle: the render was monochrome by construction.

### 2. REAL DISPERSION IS TOO SMALL TO SEE, and the reference look is not physical
Measured index spread across 420-680nm: fused silica **0.0123**, BK7 **0.0148**, SF11 -- a dense flint,
the hardest-dispersing glass shipped -- **0.0597**. The widely used Cycles "dispersion glass" setup
stacks three glass BSDFs at **IOR 1.35 / 1.55 / 1.75**: a spread of **0.40**, which is **6.7x SF11** and
would need an **Abbe number near 2.5**. No real glass is below about 20. Its own author calls it "a
pseudo-physical fake". So a physically correct render of real glass IS nearly achromatic -- that is the
glass's fault, not the renderer's, and it took a reference to see it.

New `exaggerate()` / `dispersion_scale=` (catalogued 6/6, 5 tests): stretch the index spread about its
MEAN, so turning up the rainbow does not also change how much the glass bends light. 1.0 is physical
and byte-identical. Accepted by BOTH `dispersive_render` and `spectral_caustics` -- previously the
light path could only take a catalogued glass NAME, so a scene's caustic could be cast by a different
material than the camera saw. Two glasses in one image is a correctness bug, not a missing convenience.

### 3. THE LIGHTING WAS BACKWARDS
The reference rig is environment light **ZERO**, a black floor, and two area lamps kept far enough away
not to directly illuminate the glass. I had a bright ambient sky, which lights every surface and leaves
nothing for a refracted edge to split against. Under the reference rig the same scene measured
saturation **0.52 physical / 0.81 at scale 7**, against **0.25** under the bright sky.

### 4. THE "WEIRD STUFF" WAS A POST EFFECT
Those 4-point stars were `glare(streaks=2)` in my own post chain, not the render. Removed.

### THE CAUSTIC WAS EMPTY, AND THE EMITTER WAS WHY
`caustic pass: covers 0.0% of frame` -- printed by a diagnostic added for exactly this, and it was the
whole reason no caustic appeared. **The ray grid launches from y=3.0 centred on the ORIGIN**, which is
correct only for a light travelling straight down. Worked out: with light travelling (0.551, -0.752,
-0.361) toward an object at y=0.46, a ray falls to the object's height in t=3.38 and drifts x by
**+1.86** -- so a grid spanning x in [-0.66, 0.66] arrives at x in **[1.2, 2.5]**, entirely past an
object spanning [-0.61, 0.61]. Not one ray hits it; `refracted_only` returns an empty image and nothing
says why. This is the SAME bug family as the receiver window fixed in sweep 142, on the emitter side --
`extent` was doing a third job nobody had noticed.

New `aim=(x,y,z)` back-projects the launch plane along the light so the grid lands on the target
(`emitter_center=` for manual control; defaults byte-identical, pinned by a selftest that asserts an
oblique light renders EMPTY without it). With aim: peak **1486x mean**, core at world (0.47, -0.47) --
not the origin, which is why locating by peak-search rather than assuming still matters.

### KEPT NEGATIVE -- A SMALL BRIGHT LAMP IS UNRENDERABLE HERE
The reference's small area lights are exactly what this tracer cannot sample: no next-event estimation,
so a lamp small enough to look right is a lamp the sampler almost never hits, and the render arrives as
fireflies. Measured across lamp half-sizes 0.075 -> 0.50, relative noise stayed 0.08-0.11 while the LIT
fraction of frame ran 0.2% -> 4.8%. The rig ships at half-size 0.28: the largest that still leaves the
background black. **Matching the reference's lighting exactly would require NEE, which is a build, not
a setting.**

### Also worth recording
A grain metric measured on a mostly-black frame reads LOW (adjacent pixels are both black), which made
the smallest, noisiest lamp look cleanest. Replaced with a two-seed difference over lit pixels -- the
difference between two seeds IS the noise. Fourth metric this program has had to throw away for
measuring something other than the effect.

### Deltas
- +2 params on caustics (aim, emitter_center), +3 on spectral_caustics (n_d, abbe, dispersion_scale),
  +1 on dispersive_render, +1 catalog card (21 in p08), +5 tests (31 in test_dispersion). Audits 0/0/0.

### The above/below gate caught what the catalog did not
`test_the_gate_passes_at_the_recorded_floor` went red: **14 genuine gaps against a budget of 12**, and
both new ones were mine. `combine_buckets` and `exaggerate` were CATALOGUED -- discoverable by
find_capability, 6/6 on the alias probes -- but registered against MODULE FUNCTIONS with no mind verb,
so `/invoke` could not call either. By this repo's governing rule they did not exist. Cataloguing a
module function is not wiring it, and the discoverability check passing is exactly what made that easy
to miss: the card was findable, the capability was not callable.

Fixed with two delegating verbs, `exaggerate_dispersion` and `combine_render_buckets`, and the cards
repointed at them. Gate back to 12, at the floor. **Shrink-only did its job: the budget refused to
absorb the regression, which is the whole reason it is shrink-only.**

Suite: **6899 passed, 0 failed.**


## Sweep 146 -- the tracer had next-event estimation all along; the verb hid it

**Brief.** "The quality on this image is still greatly lacking. I want a production quality render."
Then, mid-sweep: "both v-ray and Blender support dispersion and caustics, and can run at realtime
speeds on gpu ... I'm pretty sure we already have much of what we need already implemented, and it's
mostly a math and wiring issue." That reading was correct, and this sweep is the proof.

### THE FINDING: SIX SWEEPS OF RENDERS WERE MADE WITHOUT DIRECT LIGHT SAMPLING
`holographic_pathtrace.path_trace` takes `lights=` and calls `holographic_lights.direct_lighting`,
whose own docstring opens **"Next-event estimation: the DIRECT light reaching shade points P from all
`lights`, with shadow rays."** It has been there the whole time.

`mind.path_trace` exposed **NINE of the module's TWENTY parameters** and dropped `lights` and
`antialias` on the floor. Worse, its docstring carried **"KEPT NEGATIVE: no next-event estimation ...
NEE/MIS is the next step"** -- which I read as ground truth and quoted as a measured limitation in the
sweep-142 and sweep-145 write-ups, including in a sentence naming it as "the next real build". The
capability was not missing. The WRAPPER was narrow and its documentation was stale.

Measured on the same scene at the same sample budget:

| | relative noise | mean radiance |
|---|---|---|
| no NEE (sky only) | 0.0434 | 0.025 |
| **NEE (`lights=`)** | **0.0217** | **0.242** |

Half the noise, and TEN TIMES the light -- because without direct sampling the small lamps were
essentially never hit, so the scene was starved and I had been compensating with a flooded ambient,
which is what washed out every earlier render. The production gem render came back clean: no speckle,
smooth facets, dispersion fringes intact, caustic with coloured edges.

**The lesson is about the surface, not the maths.** A wrapper that silently narrows its implementation
is worse than no wrapper, because the capability looks ABSENT rather than broken -- and this repo's own
rule ("when docs and code disagree, the docstring in the live module wins") existed to catch exactly
this. I applied it to the wrong docstring: the wrapper's, not the implementation's. `mind.path_trace`
now forwards `lights`, `antialias` and `**kw`, `make_light` is a verb, and a test asserts the verb
exposes both and forwards `**kw` so the next module parameter added cannot go invisible again.

### THE REAL SOTA GAP, found by looking it up rather than guessing
Plain NEE still cannot render caustics, and the reason is structural: NEE connects a shade point to a
light with a STRAIGHT shadow ray, while a caustic's path goes light -> glass -> floor -> eye, THROUGH a
specular vertex that bends it. So even perfect NEE finds caustics only when a bounce happens to land on
a light after refraction. The field's answer is **Manifold Next-Event Estimation** (Hanika et al.) and
**Specular Manifold Sampling** (Zeltner, Georgiev & Jakob, SIGGRAPH 2020) -- which is what Blender's
Cycles uses -- plus a 2024 dimension-reduction variant for interactive rates. Instead of hoping, SOLVE
for the vertex: a Newton iteration on a constraint that is zero exactly when the refracted ray points
at the target.

New `holographic/rendering/holographic_specularconnect.py` + `mind.specular_connect` (catalogued 6/6,
3 tests). **It is small because the engine already had the solver**: this repo states that IK, PBD, PnP
and the resonator "are all iterate a projection", and `project_onto_constraints` calls itself "the one
engine under three faculties the mind grew separately". The missing piece was the CONSTRAINT, not the
iteration -- exactly the "math and wiring" reading. Converges on a plane to residual 3.0e-08 and on a
curved SDF to 2.3e-06, verified by walking the path FORWARD rather than trusting the residual it
minimised, with total internal reflection reported as no-solution rather than invented.

**SCOPE, DECLARED: ONE interface.** Real glass refracts on entry AND exit; the two-vertex chain is the
next rung and is not silently approximated. `connect` returns `converged` so a caller can reject a
vertex rather than draw light where none goes.

### Using leCore to audit leCore, properly this time
Prompted mid-sweep, I used the semantic surface rather than only `find_capability`: `m.suggest(task)`
returns ranked capabilities WITH the call to make, and confirmed the specular connection did not exist
under another name before I built it -- the check the doctrine asks for and I had been doing too
shallowly. It also surfaced **`realtime_session`** (draft frames that reproject the previous frame and
re-shade only the news), which is directly on the realtime question and which I had never found in six
sweeps of `find_capability` probes.

### The uploaded mesh
A 151,582-face ladybird. `mesh_to_sdf` is brute force over every triangle: **155,000 microseconds per
query point**, and a render needs order 1e9 queries -- about 43,000 hours. `mesh_to_sdf_grid` bakes the
field once and `sample_distance_grid` reads it at **0.245 us/point: a 634,000x change in the inner
loop**, and the reason the mesh is renderable at all. leCore's first lever, and the mesh is the case it
exists for. KEPT NEGATIVE: the bake is a resolution ceiling (96^3 over a 4.45-unit extent is 0.046 per
cell, so the spots and leg joints are gone before the tracer sees them), and a 144^3 bake was
OOM-killed while a render held memory -- cubic in both time and space, a real trade.

Also measured and worth keeping: the first ladybird render came back with max radiance 0.07, near
black. The cause was not the material or the bake -- the model is 4.45 units across, so the same
RELATIVE light placement put the lamps at 9.8 units instead of 4.2, and inverse-square divided their
contribution by 5.4. **Quoting a light in watts without its distance is quoting half a number.**

### Deltas
- +1 module (799), +3 mind verbs (path_trace widened, make_light, specular_connect), +1 catalog card
  (23 in p08), +5 tests. Audits: swarm gate 12 (at floor), reachability 0/0, catalog_gaps 0,
  skill_lint 0.


## Sweep 147 -- the speckle was GEOMETRY, and the caustic was never rendered

**Brief.** "This looks pretty rough... I wanted to see caustics and dispersion."

Two separate faults, and only one of them was subtle.

### 1. THE CAUSTIC PASS WAS NOT IN THE LADYBIRD SCRIPT AT ALL
Not a limitation, an omission: the script had no caustic section. Added, and the ladybird does cast
one -- located by peak search (never assume the origin under an oblique light), peak **3374x mean**,
core at world (1.21, -0.27), covering 12.3% of frame.

### 2. THE SHELL SPECKLE WAS THE BAKED GEOMETRY, NOT THE SAMPLER
This is the finding, and it was worth measuring before spending anything on it. A diffuse-vs-glass
comparison and a direct normal probe on the baked field: **10.8% of surface points had normals
disagreeing by more than 20 degrees with a neighbour a QUARTER CELL away**. A refracted ray's
direction is set entirely by the normal, so a stair-stepped field throws each wavelength somewhere
different and the shell fills with rainbow speckle. **No number of samples touches that** -- which
means every "more spp" instinct was aimed at the wrong thing.

**The obvious fix did not fit.** Higher-resolution bakes were OOM-killed at 128^3, 144^3, 176^3 and
192^3 with 7 GB free. Two levers were applied and neither was enough, because the binding cost is
VOXELS and not triangles:
  * DECIMATE: 151,582 -> 29,594 faces in 4s, using the silhouette-guarded decimator. At 0.023 world
    units per cell the grid cannot represent detail the extra triangles carry, so they were pure cost.
  * FIT THE BOUNDS TO THE BODY: the object is 3.59 x 1.71 x 3.46 and I was baking a CUBE of side 4.45
    around it -- 4.1x the voxels spent on empty space. Free win, still not enough.
  * TILING (lever 5) DOES NOT APPLY HERE, and it is worth recording why: distance tiles cleanly
    because it is local, but `mesh_distance_grid` takes a SCALAR res, so a thin slab still allocates
    res^3. The sign does not tile either -- inside/outside is a global flood fill. The lever was right
    and the API could not express it.

**What did work, partially:** a distance field is SMOOTH by construction and the stair-stepping is
quantisation, so the field itself is filtered -- 2 separable 1-2-1 passes take the >20-degree
disagreement from **10.8% to 5.6%** while moving the zero level set by 0.012, about a quarter of a
cell. **Halved, not solved.** The legs are thinner than the smoothing kernel and are partly lost to
it, which is the cost stated plainly.

### 3. DISPERSION SCALE MULTIPLIES GEOMETRY NOISE
At scale 6 the index spread is 0.196, so wavelengths leave a noisy normal in visibly different
directions and the field's roughness becomes COLOUR noise. Dropped to 4 (spread 0.131): the effect
stays legible and the speckle is amplified less. The two knobs are not independent, which was not
obvious until the geometry was identified as the noise source.

### The honest state
Caustic: present and bright. Dispersion: present, banded across the shell. Shell: still rough, and the
cause is now measured rather than guessed -- it is bake resolution, and the ceiling is this baker's
memory scaling, not sample count or the estimator. The fix is a banded/sparse SDF bake that allocates
with SURFACE AREA rather than volume; `mesh_distance_grid`'s shell method already computes that way
but the API returns a dense res^3 array. That is the next build, and it is a storage-format change
rather than a maths one.

### Deltas
- No new capabilities this sweep; the work was diagnosis, and the diagnosis changed what to build next.
  Audits unchanged: swarm gate 12 (floor), reachability 0/0, catalog_gaps 0, skill_lint 0.


## Sweep 148 -- the OOM was one unchunked kernel, and the fast renders were fast because they were wrong

**Brief.** "continue fixing problems." The named blocker from sweep 147 was the bake memory ceiling.

### 1. THE OOM WAS NOT WHERE SWEEP 147 SAID IT WAS
Sweep 147 concluded the fix was "a bake that allocates with SURFACE AREA rather than volume", because
`mesh_distance_grid` returns a dense res^3 array. **That diagnosis was wrong, and the arithmetic says so
in one line: a 176^3 float64 grid is 43 MB.** The output was never the problem.

Measured instead: `point_set_to_mesh_grid` materialises a `(N, (2r+1)^3, 3)` neighbour block plus a
candidate edge list -- **36.8 KB of peak RSS per query at radius 2, growing linearly**, killed at 200k
queries on a 7 GB box. A 176^3 shell is ~1M voxels, so the kernel was demanding **37 GB to produce a
43 MB answer**. Not a storage-format problem: a streaming problem, in a kernel whose own sibling
(`point_set_to_mesh`) had had cache-blocking since it was written.

**Fix:** stream the queries. The index bounds are hoisted above the loop (per-chunk bounds would move
the cell grid and make the answer depend on the block size), face normals are hoisted too, and the
neighbour block is freed before the candidate list is built. `_grid_query_chunk` sizes the block from a
byte budget using a model, not a constant, because cost scales with RADIUS ((2r+1)^3 rows/query), not
with mesh size.

  * 200k queries: OOM-killed -> **0.579 GB**. 800k queries: **0.624 GB**. Four times the work, the same
    memory -- peak is now flat in N, which is the whole claim.
  * **Bit-identical** at chunk sizes 1, 7, 103, 399 and 1e9, signed and unsigned. Pinned by selftest and
    by `tests/test_meshdistance_stream.py`, because the failure mode of a chunked reduction is a silent
    chunk-size-dependent answer, and this engine's determinism constraint makes that unacceptable.
  * The bake ladder that died at 128^3 now reaches **288^3 with the FULL 151,582-face mesh in 453s at
    1.56 GB** -- 11.4x the voxels of the resolution that used to be fatal.

### 2. THE SPECKLE METRIC, ACROSS THE RESOLUTIONS THE FIX UNLOCKED
Sweep 147's measurement (normals disagreeing >20 deg with a neighbour a quarter cell away) was 10.8%,
and smoothing the field halved it to 5.6% at the cost of the legs. On the fields this sweep can build:

    old ceiling (~112^3, decimated, flood sign)   10.8%      (5.6% smoothed, legs lost)
    128^3 full mesh, winding sign                  0.5%
    192^3 full mesh, winding sign                  0.1%
    288^3 full mesh, winding sign                  0.0%

**The field blur is now actively harmful and has been dropped: 0.5% unblurred vs 0.7% blurred.** A
workaround outlived its problem by one sweep, which is the argument for re-measuring a mitigation
whenever the thing it mitigated changes.

### 3. RETRACTION -- "the extra triangles were pure cost" was wrong
Sweep 147 decimated 151,582 -> 29,594 faces and argued the grid could not represent the difference.
Paired bakes at the SAME 192^3 say otherwise:

    decimated (29,594):  bad normals 1.5%   signed interior 0.073
    full     (151,582):  bad normals 0.1%   signed interior 0.178

**Less than half the body was being signed as solid.** Every glass render before this was refracting
through a partly hollow object. The decimation was only ever done to fit the memory ceiling this sweep
removed, and it was measured on silhouette IoU -- not on the field it fed. *A mesh operation judged by a
mesh metric can still be wrong for the field it is an input to.*

### 4. sign="winding_flood" -- composing two sign methods, each where it is honest
The winding sign is priced at every voxel: **38.8 min for a 192^3 ladybird bake**, and the cost is in the
wrong place -- the winding number far from a surface was never in doubt. `flood_fill_sign` is cheap and
leaks on a soup for exactly one reason: the negative band shell that should block it has holes.

So: **winding signs the SHELL (making it watertight, all the flood needs), the flood fills the interior.**
The shell is O(surface area) and its share of the volume *falls* with resolution (17.5% of voxels at
128^3, 11.6% at 192^3, 8.4% at 288^3), so it saves most where the full path hurts most.

Paired, same process, 151,582 faces at 128^3: **519.5s -> 237.9s (2.18x), 0 of 2,097,152 voxels differing
in sign, and the marched meshes vertex-identical at 343,628 faces.** At 192^3 it completes in 667s where
the full-volume path did not finish in 27 minutes.

**KEPT NEGATIVE, and it is why this path self-checks:** the two methods answer DIFFERENT questions --
winding asks *is this point enclosed by the solid angle*, winding_flood asks *can this point escape to the
grid boundary*. They coincide only when no opening lets the flood walk in (the .glb class: seams between
coincident shells). **A hole 0.8 band widths across already flips 5.05% of voxels, and it does not improve
with a smaller hole (4-5% from 0.8 to 2.2 bands) -- the flood needs one voxel-wide passage.** It UNDER-fills
(0.066 vs 0.117), i.e. it would ship a silently hollow object. So the path verifies against a random winding
subsample and REFUSES, naming `sign='winding'` as the fix. `verify=0` opts out.

**KEPT NEGATIVE (measured, do not reinvent):** raising `fast_winding_number`'s dipole cell count does NOT
help -- the far field iterates over cells, so cost is O(queries x cells^3) and the near-field saving never
repays it. 16 -> 6.58s, 32 -> 22.03s, 48 -> 43.32s for 20k queries, with sign agreement 1.0000 throughout.
`cells=16` is the working point. I raised the cell count expecting a speedup and got a 6.6x slowdown.

### 5. A LIVE BREAKAGE FOUND ON THE WAY
`python3 -m holographic.mesh_and_geometry.holographic_meshbridge` had been failing before this sweep
touched it: `voxel_remesh(box())` raised "needs TRIANGLES, got 4-gon faces". A cleanup verb refusing input
for needing cleanup -- `box()` is a quad primitive from this same package. It now ear-clips first, reusing
`meshverbs2.triangulate_ngons` (concave-correct) rather than fanning locally. **A broken module selftest
hides every other assertion behind it**, which is how the arity guard shipped over a live caller.

### 6. THE RENDER GOT 10x SLOWER, AND THAT IS THE CORRECT DIRECTION
With a genuinely solid interior, rays now spend their bounces refracting *through a body* instead of
passing through a hollow shell. Measured at 160x100x8spp: 26.8s at 8 bounces, 17.9s at 4, 12.1s at 2 --
and the blur is NOT the cause (26.8s unblurred vs 27.4s blurred). **The previous ladybird renders were
fast because they were wrong.** Sample budgets have to be re-planned against the corrected field; that is
a cost of correctness, stated rather than hidden.

### 7. THE SPECKLE IS NOW MONTE CARLO NOISE, AND IT MEASURES AS EXACTLY THAT
Sweep 147's whole point was that the speckle was immune to sample count. It is not any more. Two
independent seeds at the same spp cancel the signal and leave only estimator variance:

    spp  8   MC noise 0.0707
    spp 32   MC noise 0.0350   (1/sqrt(spp) predicts 0.0353 -- a 0.9% match)

An unbiased estimator behaving like an unbiased estimator. **The geometry is no longer the wall**, so
sample count now buys quality, which for two sweeps it did not. Confirmed in the delivered renders: the
same scene at 24 spp and at 96 spp (4x) drops image saturation 0.1345 -> 0.1059, because colour noise
inflates saturation and there is now less of it. The legs and antennae are present in both -- sweep 147's
field blur had eaten them, and dropping it cost nothing. The remaining speckle in the delivered
render is a sample-budget decision (24 spp), not a defect: 4x the samples halves it, and
`render_progressive` exists so that budget can be spent across sessions.

**KEPT NEGATIVE -- a fourth metric that measured the wrong thing.** The first attempt at this used
high-frequency energy in a SINGLE render, which conflates noise with SIGNAL: a ladybird's spots and
shell detail are genuinely high-frequency, so the metric has a floor at infinite spp. It read
0.0713 -> 0.0614 for 4x the samples and would have concluded "still geometry limited" -- the exact
opposite of the truth. Joining grain-on-a-black-frame and the three constant-returning dispersion
metrics on the list. *The tell is the same every time: a metric that cannot go to zero when the effect
is removed is not measuring the effect.*

### 8. GENERALISE ON CONTACT -- the engine already had the better shape, in another family
Asking "which existing module is this in a different costume?" after the fix, not before: the answer is
`holographic_fields.close_pairs`, which solves the same problem (gather the candidates in a neighbour
block around each query) and **never materialises the product at all**. It loops over the 3^D offsets --
a tiny fixed loop -- and each iteration touches only (N,) arrays, so its peak is O(N + pairs) rather than
O(N x 3^D). Its own docstring even names the lesson: *"the same lesson the mesh-distance and sculpting
work kept re-learning"*. It was right and the mesh-distance kernel was not.

**So chunking is the correct fix but not the final one.** The offset loop would remove the need for a
byte budget entirely rather than making it tunable. It is NOT being done in this sweep, deliberately: the
signed path picks its nearest triangle with a `lexsort` over (distance, query), and restructuring that
into a running argmin across offsets touches a TIE-BREAK on a determinism-critical path, right after a
change that is proven bit-identical and pinned by 18 tests. Two rewrites of the same reduction in one
sweep, one of them tie-sensitive, is how a bit-identical guarantee quietly stops being true.
**Named as the next rung, with the reference implementation identified.**

UP/DOWN/SIDEWAYS on the streaming fix: *down* -- it works on sub-blocks of its own input, which is the
mechanism. *Up* -- when its output is a component of something larger, i.e. `mesh_distance_grid` and
every bake above it, which is where the whole gain landed. *Sideways* -- the costume it wears elsewhere
is `close_pairs` (particles, softbody self-collision) and `fast_winding_number` (which already had a
`chunk=` and needed nothing). The direction that was missed for two sweeps was simply *inward*: nobody
asked what the kernel allocated per query.

### Deltas
- 0 new modules. +2 mind verbs (`mesh_query_chunk`; `mesh_point_distance` widened with chunk/max_bytes),
  +1 sign mode (`winding_flood`, opt-in), +1 catalog card, +1 card extended, +2 module selftests,
  +18 tests (`tests/test_meshdistance_stream.py`). One live selftest breakage repaired.
  Audits: swarm gate 12 (floor), reachability 0/0, catalog_gaps 0, skill_lint 0.


## Sweep 149 -- the render pipeline had left its own path, and the catalog is how it got lost

**Brief (Moose, mid-sweep).** "Some of the render pipeline diverged from our original path. leCore renders by
projecting from a superposition... If the system can handle a complex 3D render, it can handle some silly text
and context... Renders should not take this long, and they should not be so hard to make look good. We went off
path somewhere and got lost, and then started trying to do things the way everyone else does."

He was right, and the audit can say exactly where and exactly why.

### 1. WHERE: sweep 138
Before it, the render lineage was holographic and it is all still here, all still wired:
  * PRT `radiance_transfer` -- Moose's "collapse the wave function instead of path tracing": shading as a dot
    product, 57x per relight.
  * `render_dispatch` / `bake_scene` / `render_baked` -- collapse on diffuse, trace on a mirror; bake once and every
    frame including the first is a relight (15x).
  * `holographic_fog_volume` (volint) -- the density field as ONE hypervector, the ray integral in closed form,
    90x over marching, empty space known not discovered.
  * `holographic_radiance` -- literally titled RENDER = QUERY.
  * `render_adaptive` / `plan_render` -- one call that chooses bake/collapse/trace from measured break-evens.
  * The scene as one bundled vector that decodes 12/12 attributes -- "the bidirectional content-addressable
    semantic memory Moose wanted."
From sweep 138 ("caustics + dispersion") onward, the work was: Monte-Carlo path tracing with NEE, hero-wavelength
spectral sampling, manifold NEE, photon-splat caustics into a histogram, progressive MC buckets, a scanned mesh
baked to a voxel grid and sphere-traced. Every one is how Cycles does it. Not one hypervector. The brief said
"SOTA" and "Blender parity" and I read it as *do what Cycles does* when it meant *achieve what Cycles achieves,
our way*.

### 2. WHY: the holographic lineage had ZERO catalog cards
`render_baked`, `bake_scene`, `radiance_transfer`, `render_dispatch`, `dispatch_methods`, `plan_render`,
`holographic_fog_volume`, `HolographicRadianceField`, `TiledRadianceField`, `HolographicField`: verbs, no cards.
The notes from their era say they were registered; the cards are gone. Every conventional path I built HAD cards
with generous aliases. Measured on the memory-booted engine before the fix:
  * "precomputed radiance transfer"          -> "Rendering (path trace)" first. Not radiance_transfer.
  * "relight a scene with a dot product"      -> nothing of the kind.
  * "render by projecting from a superposition" -> image editing, MCP doors.
**The semantic surface steered every render question toward Monte Carlo.** By this engine's own governing rule a
capability that cannot be surfaced does not exist; these did not; the drift followed. This is the project's
named failure mode -- not bugs, GAPS -- at its most expensive: it cost eleven sweeps.

**Fixed.** Five cards restored with aliases in the user's mouth ("why is my render slow every frame", "relight
without re-rendering", "the field knows where empty space is", "which render method should I use"), plus two for
the new caustic. Nine phrasings that missed now route to the holographic path: **9/9 (was ~1/9)**. One trap on
the way: capdoc drops any card whose NAME starts with "holographic_" and `suggest` ranks over that curated set,
so a card named `holographic_caustic` was findable by find_capability and invisible to suggest. Renamed.

### 3. WHAT WAS BUILT: the caustic as one hypervector (holographic_holocaustic)
The observation: a caustic is refraction (nonlinear, cheap, one pass) glued to ACCUMULATION (a pure sum -- a
bundle), and every cost of the conventional version lives in the accumulation. So `caustic_hits` is factored
out of `caustics()` byte-identically, and the landings are BUNDLED into an FPE field with wavelength as a third
axis. RGB is three UNBINDS: the colour-matching integral folds into the query vector because the multi-axis
encode is a bind of per-axis encodes. Bind a role, bundle, unbind to read -- the mechanism that stores a
sentence, pointed at light. `refract_dir` (both copies) now takes a per-ray index, so ONE ray set carries a
continuous spectrum: N rays, not N x bands.

Measured, against the histogram it replaces:
  * one vector reads at 64^2 and 128^2 consistently (corr 1.000); 256^2 vs 512^2 0.9997;
  * agreement with the histogram 0.80 (dim 1024) -> 0.95 (dim 16384); 5x smoother at equal rays;
  * translate == re-depositing shifted landings to 1e-8 (a bind IS the rigid shift); add == joint deposit;
  * the separable OUTER-PRODUCT read: img = Re(E_z diag(G) E_x^T). 223.5s -> 1.4s for a 256^2 RGB read at dim
    16384, identical to 1e-15. The optical-correspondence note's lesson applied literally -- stay in the
    Fourier domain, let structure do the lens's work. **115x from noticing separability.**

### 4. WHAT IT COSTS -- kept negatives, loud, because the substrate is not free
  * **CAPACITY-BOUNDED.** A dim-d vector resolves ~sqrt(d) cells per axis. Sharper than that and agreement with
    the histogram FALLS (bw 20 -> 0.81, 160 -> 0.12 at dim 1024); raising dim at a fixed kernel raises it
    (0.744 -> 0.900 -> 0.953). default_bandwidth is now 0.6*sqrt(dim), derived from that measurement.
  * **COLOUR SEPARATION IS CAPACITY-BOUNDED TOO, and worse.** Two single-wavelength reads of the SAME geometry
    correlate at only ~0.75 (dim 2048) with dispersion OFF -- each lambda slice has its own crosstalk. Physical
    dispersion through a sphere at scale 3 is a ~3%-of-variance effect (red vs blue landing histograms 0.972).
    At dim <= 4096 crosstalk exceeds the effect. Dispersion still lowers the red-blue read correlation, and the
    drop grows with dim (0.009 at 2048, 0.024 at 4096) -- but a rainbow read from a small field is mostly
    crosstalk wearing colour. Stated in the docstring, pinned in the selftest as a contrast against the floor.
  * **TILING** (TiledHolographicCaustic, mirroring TiledRadianceField) cut the build 50.7s -> 4.6s and holds
    0.87 agreement with the TRUE (smoothed) caustic at equal ray budget -- but with landings per tile still
    above the tile's dim, each tile shows its own crosstalk texture: a PATCHWORK at borders. More tiles, not
    more dim. Not solved in this sweep; named.
  * **THE HISTOGRAM WON THIS ROUND ON SPEED.** spectral_caustics on the 1-fold Mandelbox: 2.3s, clean rainbow
    arcs. The holographic build: 50-80s. The substrate's measured wins here are resolution independence,
    composability and the one-pass spectrum -- NOT raw speed. Saying otherwise would be narrative over
    measurement.
  * Metrics discarded this sweep (the arc's 6th-8th): red-vs-blue CENTROID for lens dispersion (a lens moves
    its focal distance, not its bulk; read 0.025 at every scale -- radial SPREAD is the statistic, 1.05 -> 0.82);
    clipping the read at zero before measuring (biases centroids 4 px; raw= exists); padding the encoder bounds
    (Gaussian-phase FPE does not wrap -- that warning is for sinc encoders; every padding gave identical bits).

### 5. THE OTHER THING THE AUDIT FOUND: the fractal the user asked for could not be a solid
`fold_fractal` (the Mandelbox) had NO escape bailout -- `mandelbulb`, twenty lines below in the same file, always
did -- so its distance estimate collapsed by scale^4 = 16x per 4 extra iterations (0.02174 -> 0.00131 -> 0.00008
at radius 1), and it was ALL-POSITIVE: no interior, so no glass, no refraction, no mesh at level 0 (the docs
recorded that as a fact of life). Its selftest asserted a Lipschitz UPPER bound, which a field that is ~0
everywhere passes trivially. Now `bailout=` and `solid=` (opt-in, defaults byte-identical, ARITY untouched via an
OPTIONAL_PARAMS table): estimate converges (median d12/d4 0.0023 -> 1.0000), interior 0.072 at bailout 4 (and
0.996 at 64 -- escape defines outside, keep it modest), bailout tested AFTER the step (before it, d(5,0,0) read
exactly 5.0 -- distance to the origin, an over-estimate a tracer tunnels on). Retracted an over-claim from my own
draft: the collapsed estimate still CONVERGES (108 steps, 1e-3 residual vs 79 and 1e-4) -- a too-small step is
safe, only slow; the real blocker for glass was the sign.

And the result that matters most for the user's actual complaint: **the 1-fold inverted Mandelbox as glass shows
clear prismatic dispersion in the histogram caustic in 2.3 seconds, and renders as clean crystal at 192 spp in
under 5 minutes.** Dispersion was never the hard part. The scanned soup was. The choice of subject was measured:
menger has thousands of faces in SIX orientations (entropy 0.65, edges 0.16) and bends every wavelength alike;
the inverted Mandelbox has entropy 0.90 and edge fraction 0.45. Then three colour metrics ranked the NOISIEST
previews best (raw saturation counts grain as colour; blurring does not help a ratio) and nearly had me reject
good shapes -- the confetti was 24 spp of a 6-bounce glass path, and at 192 spp it was crystal.

### 6. THE NEXT RUNG, named
The caustic was the clean case because its accumulation is linear. The 30-100 minute cost lives in the VIEW
render: MC path tracing a solid glass interior x bounces x wavelengths x spp. The holographic answer is already
in the lineage: `bake_scene`-style -- trace primary visibility and the refracted exit ONCE per pixel, store a
per-pixel transfer over (light, wavelength) as the field, and make every relight and every wavelength a
dot-product READ. That is PRT generalised from diffuse-SH to spectral-refractive transfer, and it attacks the
actual number. Not started here; this sweep's job was to find the path, and the path is now findable.

### Deltas
- +1 module (holographic_holocaustic: HolographicCaustic, TiledHolographicCaustic, spectral_landings,
  caustic_hits factored into globalillum), +3 mind verbs (holographic_caustic, holographic_caustic_tiled,
  wavelength_cmf), fold_fractal gained bailout/solid (+OPTIONAL_PARAMS), refract_dir per-ray eta (both copies),
  +7 catalog cards (2 new, 5 RESTORED for the holographic lineage), +15 tests
  (tests/test_holocaustic_and_foldfractal.py), fold_fractal selftest gained a real regression trap.
  Audits: swarm gate 12 (floor), reachability 0/0, catalog_gaps 0, skill_lint 0. Discoverability of the
  holographic render path: 9/9 phrasings (was ~1/9).


## Sweep 150 -- the next rung, built: glass as a baked spectral-refractive transfer (holographic_glassbake)

**Brief.** "Sounds like we are on the right path now, let's keep going... keep thinking holographically."

### THE NUMBER
The 1-fold inverted Mandelbox as glass, 320x200, 7 wavelengths, three softboxes + sky:
    Monte-Carlo path tracer (sweep 149):   ~5 minutes at 192 spp, still grainy
    bake_glass + relight_glass:            bake 1.6s, relight 7s per light -- and a SECOND light is another 7s
                                           from the SAME bake, no re-trace. Zero noise.
About 33x on the first frame, and the second frame is free of tracing entirely.

### WHY IT WORKS, in the engine's own words
PRT: the expensive part depends only on GEOMETRY, so precompute it as a transfer and shade by dot product.
bake_scene/render_baked: trace visibility once, every frame is a relight. This is that idea applied to a
dielectric. The path tracer's only randomness was its ESTIMATOR -- a coin per sample for reflect-or-refract, a
wavelength draw per layer, averaged over spp. The geometry underneath (entry, interior transit, exit, Fresnel
split) was always deterministic. So: evaluate BOTH Fresnel branches once, for K stratified wavelengths, store
the exit directions, and make the light a FIELD -- an FPE hypervector over the sphere of directions (EnvField).
pixel_c = sum_k w_c(lambda_k) [ (1-R) <L, encode(D_out_k)> + R <L, encode(D_refl)> ]: a dot product between the
bake and the light. The CMF folds in at read time, as holocaustic does. Reuse found by asking the memory-booted
engine: `studio_sky(preset)` is a sky(D) callable, exactly EnvField.add_sky's contract -- one door.

### TWO BUGS FOUND ON THE WAY, both kept loud
1. **First-order glass dropped 65.5% of the energy.** A cube-ish body sends most rays that enter one face into
   total internal reflection at the next. Following internal bounces deterministically (reflect and march again,
   up to a cap) should have fixed it and did NOT: 65.5% -> 65.0%. Because:
2. **`_march_through` re-exited the face it had just reflected off.** After a grazing TIR the restart point
   (exitP + r*3e-3, with exitP ~1e-3 OUTSIDE where the previous march stopped) slides along the face and still
   reads d > surf_eps, so one step later the "next face" is the one it left, forever. Added `require_inside=`
   (default off, byte-identical): an exit does not count until the ray has been genuinely inside. Loss 65.5% ->
   **3.0%**. The test for this took two drafts: a normal-incidence restart does NOT reproduce it (the exact box
   SDF reads under eps there and the default march enters fine); the fixture had to be grazing with d > eps.
Also: the "relight cheaper than bake" assertion was the wrong claim on a trivially cheap fixture (0.02s bake on
an analytic sphere). The claim that matters is absolute: both are far below any MC estimate and relight needs no
tracing. Relight is O(pixels x K x dim) direction encodes -- the light field's dim defaults to 512 because a
light field is SMOOTH (~8 degrees of angular resolution), not the caustic's 2048-16384.

### KEPT NEGATIVES
  * Internal bounces past the cap (default 4) are lost: 3.0% of glass-wavelength pairs on the Mandelbox.
  * The env field is a kernel estimate over S^2: a sharp softbox seen THROUGH the glass reads soft. A smooth sky is
    exact. Bandwidth/dim are the knobs.
  * Floor shading is direct-only (one shadow ray toward sun_dir); the caustic on the floor is a separate pass
    (holographic_caustic / caustic_pass), as before.
  * No rough/microfacet glass, no volumetric scattering inside -- smooth dielectric only, same as the MC path.

### Deltas
- +1 module (holographic_glassbake: EnvField, GlassBake, bake_glass), +3 mind verbs (bake_glass, env_field,
  relight_glass), `_march_through` gained require_inside= (default off), +1 catalog card, +8 tests
  (tests/test_glassbake.py). HTTP /invoke round-trip confirmed for all six new verbs of sweeps 149-150.
  Audits: swarm gate 12 (floor), reachability 0/0, catalog_gaps 0, skill_lint 0. Discoverability: "why does my
  glass render take an hour" -> the bake, first hit.


## Sweep 151 -- "even the simple geometry is noisy and aliased" -- it was, and the fix was already in the engine

**Brief.** Moose: noise and aliasing on the octahedron, the Mandelbox "almost pure noise in some areas", and no
caustics. "We already have the tech, it's just not wired up fully." Then: reconsider every divergence with the
SEVEN levers (I had five in my notes) and the virtual GPU / VM in mind.

### 1. THE ALIASING: one ray per pixel, then an upscale
The bake shot ONE ray per pixel with no antialiasing, and I upscaled that. A faceted body's every edge lands on a
pixel boundary somewhere; no post-process undoes it. The engine's own answer is the path tracer's `antialias=True`
-- Sampling.low_discrepancy sub-pixel offsets through camera.ray_dirs(jitter=). Wired into the bake as
`samples=`: S jittered sub-bakes, relit and AVERAGED (relight is linear, so the mean of relit sub-bakes IS the
box-filtered pixel). Rendered at 2x and box-downsampled on top: 8 effective samples on everything, the caustic
projection included. Octahedron: smooth silhouette, thin clean colour fringes, soft caustic border. Mandelbox at
960x600 x4: readable glass with internal reflections, no blowout, rainbow-fringed caustic. Pinned by test.

### 2. THE SPECKLE: measured, and it is geometry -- but the env field was not helping either
Deterministic bake, so speckle is structural. Three relights of one bake: FPE env dim 512 (roughness 0.59), dim
4096 (0.62), ANALYTIC env with zero crosstalk (1.62 -- higher, because a hard-edged light makes every facet either
see it or not). 2x2 supersample: -15%. So the sparkle is the facets, each its own prism. The Mandelbox at 1 fold
simply IS that dense at 320x200. AND the honest corollary: the FPE EnvField earns its place only when you want
algebra on the LIGHT (add a lamp = add samples, move it = bind). When the light is already a smooth function --
studio_sky(preset), a sky lambda -- bundling it into a field adds a crosstalk texture for nothing. `relight` now
takes any callable D->rgb directly (AnalyticEnv). Lights are now soft-edged Gaussian discs, not hard ones.

### 3. NO CAUSTIC: my omission, composited now
The caustic was a separate pass I never composited into the relit frame. It is in: search -> spectral_caustics ->
caustic_pass -> composite_caustic. It is also now the DOMINANT cost of the frame (35-80s of 55-227s), where the
view bake is 1-2s (320x200) and the relight 7s. The histogram path won on quality this sweep; the holographic
caustic (below) is still the one to make win.

### 4. LEVER 6 ON THE TILE PATCHWORK -- half worked, half a kept negative
hierarchical_pack's lesson: a tile is a level, and levels need a CLEANUP BETWEEN them (its codebook snap is the
crosstalk reset). Two coordinator-level operations tried on TiledHolographicCaustic:
  * CROSS-FADE across the halo overlap (every tile can read past its border): 0.831 -> 0.843 against the true
    caustic, the blockiness visibly gone. ON by default, +10% read time.
  * PER-TILE FLOOR REMOVAL by low percentile: agreement FELL 0.817 -> 0.638. A caustic fills most of some tiles,
    so the 15th percentile there is SIGNAL. A per-tile statistic cannot tell floor from signal; the floor would
    need a genuinely empty probe the coordinator does not have. Opt-in, documented as the negative it is.
The remaining haze in the tiled read is the capacity crosstalk; the histogram is still the better caustic today.

### 5. THE SEVEN LEVERS, re-read against where the time goes now
`mind.levers()` returns 7; I had been working from 5. Levers 6 (a measured limit is a tile size -- group,
coordinate, clean up between levels) and 7 (spend accumulated experience -- amortise across similarity) were the
missing two, and 6 is exactly what the caustic capacity wall wants. Where the frame's time goes at 960x600 x4:
bake 139s (sphere-tracing the fractal, 4 x 576k rays) > caustic 80s > relight 7s. Against the levers:
  * Lever 1: the bake is per-CAMERA and light-independent -- every relight is free; a turntable is one bake per
    frame but a lighting session is one bake total. Already the structure.
  * Lever 3: the bake is deterministic from (camera, sdf, seed) -- regenerate, never store.
  * Virtual GPU: the engine already has `sdf_depth_device` -- the SDF emitted as WGSL and sphere-traced on any
    adapter, with `sdf_depth_cpu` as the reference and `sdf_depth_agrees` as the differential test. That is
    precisely the bake's 139s. This sandbox has no adapter, so the number is NOT measured here; the wiring is the
    next thing to do (bake_glass's primary trace through the device path). Stated as unmeasured, not promised.
  * The VM: a render pipeline already lowers to a HoloMachine program (bit-identical to the loop, by the notes) --
    that is unification, not speed, and the notes say so.

### Deltas
- bake_glass gained samples=/seed= (MultiBake), relight accepts callables (AnalyticEnv); TiledHolographicCaustic
  read gained cross-fade (on) + floor_pct (opt-in, kept negative). +3 tests. Audits: swarm 12 (floor),
  reachability 0/0, catalog_gaps 0, skill_lint 0.


## Sweep 152 -- "not photoreal yet": three pieces of physics the relight was ignoring, and materials that disperse

**Brief.** Moose: quality up greatly, still not physically accurate or photoreal; is it the environment? Then:
the physical materials (ruby, sapphire, diamond, water) should drive dispersion/caustics/refraction/ACES. And the
standing brief: the seven levers, the virtual GPU, the VM.

### 1. WHY THE GLASS WAS BLACK -- geometry the relight already had and ignored
  * **Exit rays never saw the floor.** A ray leaving the glass downward hits the floor the object stands on, and
    that floor is what you SEE through a glass object. The relight read only the environment for exit rays, so every
    downward exit saw a 0.045 sky: black facets. Exit positions are now in the bake; exit and reflection rays are
    intersected with the floor plane and take its shaded colour. This is the single biggest realism fix.
  * **The shadow was opaque black.** Glass does not cast one: its shadow is the light it did not transmit (a few
    percent) plus the focused part the caustic adds back. `shadow_transmittance` (default 0.85; 0.0 = the old
    behaviour).
  * **A units bug: radiance used as irradiance.** The floor's direct term multiplied the key's RADIANCE straight in,
    but a small light delivers radiance x SOLID ANGLE -- ~0.16 sr for a sigma-0.16 softbox, a factor of six. That is
    why the studio floor blew out white while the glass, which reads radiance correctly, looked fine.
    `sun_solid_angle` (default 1.0 keeps old numbers). The exposure knob could never have fixed this.
  * And the environment itself: a studio has WALLS. Grey walls, a lit ceiling panel, one key -- the floor now reads
    light grey, the shadow soft, the caustic faint and physically so (a clear gem on a lit floor).

### 2. MATERIALS THAT DISPERSE -- the library had no Abbe number
holographic_matlib had an index per gem (_IOR) and per-RGB Beer-Lambert absorption (_ABSORB) but NO dispersion,
so a "ruby" could never split light; every spectral render used a hand-typed BK7/SF11 Abbe. Added `_ABBE`
(catalogue: diamond 44.3 on n 2.42 -- its "fire" is a modest Abbe number acting on a very high index; corundum
72.2; beryl 56; quartz 70; water 55.7) and ONE DOOR, `glass_optics(name)` -> {n_d, abbe, absorb, tint}.
`bake_glass(material=)` / `relight_glass(material=)` read it; the relight's absorption is now per-RGB, so a thick
ruby comes out redder than a thin edge (measured through a sphere: ruby mean rgb (0.28, 0.07, 0.07), sapphire
(0.07, 0.09, 0.30), diamond and water neutral). Rendered: a ruby octahedron that reads as ruby -- depth
saturation, red floor reflection -- and a diamond Mandelbox, each ~78s for a 640x400 x4 AA frame.

### 3. THE CAUSTIC MOVED TO THE HOLOGRAPHIC PATH -- and the lesson was capacity, again
The histogram caustic (13 traces: 3 search + 10 hero wavelengths) took 225s on the rounded-rotated octahedron --
88% of the frame. The holographic caustic traces ONCE and its landings frame the receiver themselves (1st-99th
percentile), so the search passes are gone. First attempt: 189s -- no better -- because the DEPOSIT dominated:
673k landings x 2048 dims x 36 tiles' halos. By the capacity law those landings buy nothing past ~dim per tile.
n_side 520 (224k landings), 8x8 tiles at dim 1024: **44s**, i.e. 5x on the pass that dominated the frame. The
caustic is scaled physically now: concentration x (key radiance x solid angle x cos x albedo), not a free strength.

### 4. THE VIRTUAL GPU -- it exists here after all, and the emitter is the gap
Installed the opt-in accelerators: numba 0.67 and wgpu 0.32. wgpu found an adapter: **llvmpipe** (Mesa's software
rasteriser, adapter_type CPU) -- a literal virtual GPU this sandbox can dispatch WGSL to. The engine's
`sdf_depth_device` sphere-traces an emitted SDF on it, with `sdf_depth_cpu` as reference. BUT the 4-dialect
emitter refuses both subjects: `fold_fractal` ("iterative") -- stale, since the GLSL emitter already emits it as a
fixed-count loop and WGSL has loops -- and `octahedron` (branch-heavy, "a filed follow-up"). The bake's 30-50s
is sphere tracing; this is where it would go. Not measured yet: the emitter rules for fold_fractal (with
bailout/solid) and octahedron are the next wiring, and a real number follows them.

### KEPT NEGATIVES
  * A clear gem's caustic on a well-lit floor IS faint; making it pop means a darker ambient, not a bigger strength.
  * The holographic caustic still carries the capacity haze; at 1024 dims per tile it is soft. It won on TIME here.
  * `_ABBE` is catalogue-grade per material, not per-wavelength spectroscopy; absorption stays 3 numbers.

### Deltas
- glassbake: exit_pos/entry_P in the bake, floor seen through glass, transmissive shadows, sun_solid_angle,
  per-RGB absorption, AnalyticEnv, samples= AA. matlib: _ABBE, glass_optics(). Mind verbs bake_glass/relight_glass
  gained material=. +1 catalog card (glass_optics), +4 tests (14 in test_glassbake). numba + wgpu installed as
  opt-in accelerators; llvmpipe adapter confirmed. Audits: swarm 12 (floor), reachability 0/0, catalog_gaps 0,
  skill_lint 0.


## Sweep 153 -- two HDRIs from Moose: the light stops being hand-placed, and the map decides the shadow

**Brief.** Two Poly Haven maps (newman_cafeteria_2k.exr, wooden_lounge_4k.exr): "perhaps these will be useful?"

### 1. `.exr` was a declared negative; closed by an OPT-IN import, not by widening the core
Both maps are PIZ-compressed half-float OpenEXR -- wavelet + Huffman, a container format. `load_exr` uses the
OpenEXR package if present and raises naming the fix if not; `load_hdr` (Radiance RGBE, stdlib) stays the
dependency-free door. Same contract: linear, unbounded (H,W,3) float32. Same discipline as numba/wgpu.

### 2. HdriEnv -- the map as the light, and the map's own pixels as the floor's physics
Wraps the image behind .radiance(dirs) via sky_dome. For the FLOOR: the exact cosine-weighted hemisphere integral
of the map, once (a floor is a plane, so E is one rgb); the dominant lobe (brightest pixel by energy, refined by
the energy-weighted mean inside a cone); and that lobe's SHARE of E. The shadow ray darkens only the lobe's share.
Measured: lounge sun -> share 0.41, dominant direction reads 14,421 radiance (a hard shadow); cafeteria ceiling
lights -> 0.11 (soft). Nothing hand-set.
Two negatives on the way, both kept: (a) 8,192 Fibonacci direction samples MISSED the lounge's sun (2% share) --
a sun through a window is a handful of pixels; integrate the pixels. (b) the energy-weighted mean of the top 0.1%
of pixels pointed at radiance 1 -- BETWEEN two windows. Seed from the single brightest pixel, refine in its cone.

### 3. The frames
Ruby octahedron and diamond Mandelbox in the lounge, ~76s each at 640x400 x4 AA: the background is the
photograph, the ruby reads as stone with depth saturation, the Mandelbox refracts the warm room. The lounge sun
is nearly overhead, so shadow and caustic fall UNDER the gems -- physically right, visually hidden; a raking
map would show both. Sapphire in the cafeteria: 64s.

### Deltas
- load_exr (render + mind verb), HdriEnv + m.hdri_env, relight's floor uses exact irradiance under an HdriEnv;
  +1 catalog card, +3 tests (17 in test_glassbake). OpenEXR installed as an opt-in importer. Audits: swarm 12
  (floor), reachability 0/0, catalog_gaps 0, skill_lint 0.

## Sweep 154 -- the crystal pipeline as gems: HDRI base + lobes, rock and glass apart, and two levers measured honestly

Brief: "the crystal formation pipeline should work for clusters, single crystal and geode; the HDRI as base
lighting, additional lighting to show off dispersion and caustics without blowing the scene out." Audit first
(`find_capability`): crystal_habit / crystal_cluster / crystal_geode / crystal_cut existed; `habit_sdf` -- the
ONE-crystal builder the cluster and geode place many of -- was import-only (a gap: `crystal_habit` with a bare
Miller list is an OPEN prism, 16% of a +/-1.8 probe and off its edge, because it needs `form=True`). Wired as
`crystal_single`.

### 1. The light rig, sized by irradiance not radiance
`HdriEnv.add_light(direction, radiance|irradiance, sigma)` (verb `env_add_light`): an analytic Gaussian lobe on
top of the map that ALSO updates `floor_irradiance`, `dominant_dir` and `sun_share`, so the floor, the shadow and
the caustic aim all agree with the gem. First measurement, kept: peak radiance 3000 at sigma 0.05 took the lobe to
a 0.96 share -- the HDRI had stopped being the base light. So the lobe is sized by the floor irradiance it ADDS
(`irradiance=1.4 x E_map x share_map`), and brightness is auto-exposure from the floor's median luminance
(EV = log2(0.22 / median)); 0.00% of pixels over 16x in every frame. Brightness belongs to exposure, not lights.

### 2. Caustic strength is measured, not chosen
`caustic_concentration(xz, extent, n_side, light_dir)`: each emitter ray owns a floor cell; landings per cell at
the 99th percentile IS how many times brighter the hotspot is than the directly lit floor. Quartz point 3.1x,
amethyst cluster 4.6x, geode 4.3x. Before this the strength was the bare Lambert term and the caustic peaked
DIMMER than the floor around it (0.149 vs 0.47) -- a lens that does not concentrate is not a lens.

### 3. Two speed levers, one kept, one negative
The 11-crystal cluster's bake_glass took 344 s at 640x400x2: the union evaluates every crystal at every point.
- Lever 5, `culled_union` (cull=True on cluster/geode/grow_on): bounding-sphere bounds, exact where it matters
  (same zero set, identical values near the surface, never a smaller distance -- a tighter valid tracing field).
  60-crystal geode 2.40 s -> 0.30 s (8.1x) on 200k points; cluster 2.5x. Its own negative on the way: the
  (60,200k) bound matrix as float64 broadcasts cost MORE than the 58 member evaluations it saved (2.39 s vs
  2.33 s) -- chunked in place, one matmul per chunk. And the honest limit: the exact field's cost is per CALL
  (~1 ms fixed), a march makes ~2,500 calls, so the render gained little. Which is why:
- Lever 1, bake the union to a 256^3 GridSDF once and trace the grid: bake_glass 5.2 s -> 1.3 s (4x), relit image
  mean abs diff 1.2%, 95th percentile 0.02%, side-by-side indistinguishable -- trilinear interpolation reproduces
  PLANAR facets exactly (pinned by test), only edges blend over one cell. Grid cached on disk by content hash
  (lever 3: the field is a function of its seed). KEPT NEGATIVE: `HybridSDF` (grid far, exact within 3 cells)
  measured 0.9x -- almost every march step has SOME ray near a surface, and it is the call, not the point, that
  costs. Kept in holographic_sdfbake as `bake_sdf(..., exact_near=True)` for fields with curvature the grid
  would blur, with the negative in its docstring.

### 4. A geode is rock AND glass
Rendered as one glass field the geode's rind was violet glass and its crystals stuck out of the nodule like
spines. Two additive fixes: `crystal_geode(parts=True)` returns (rind, lining) so each can be cut with the same
plane; `clip_to_skin=True` intersects the lining with the outer sphere (measured: 39% of the lining volume was
OUTSIDE the nodule -- each habit is a bipyramid centred near the wall and its back half ran out through a rind
thinner than the crystal is long). And the renderer learned the difference: `bake_glass(lining, opaque=rind)`
traces a second SDF shaded as Lambert rock -- it wins the pixel where nearer, and exit/reflection rays that
strike it are recorded at bake time (position, normal, distance) so relight shows the cavity wall THROUGH the
crystals (51% of exit rays hit rock in the bowl). `shade_opaque`: the MAP over a cosine hemisphere about the
normal (`map_radiance`, so lobes are not double counted) + each lobe with a shadow ray + the map's sun from its
floor share. A bowl must be lit from its open side -- the first key from upper-left was blocked by the rind, which
is correct and dark.

### 5. Frames (1280x800 x2 AA -> 640x400)
Quartz point in the lounge 187 s (Abbe 70: dispersion physically faint, a rainbow inside and spectral fringes in
the caustic); amethyst cluster in the lounge 176 s after a 285 s one-time grid bake; amethyst geode with an opaque
rind in the cafeteria. Not there yet, kept loud: the floor is an infinite plane with a hard horizon against the
photograph; the rock has no texture; the cavity's self-occlusion of ambient is not modelled.

### Deltas
- crystal_single, env_add_light, bake_sdf(exact_near), bake_glass(opaque), relight_glass(opaque_albedo,
  ambient_samples), crystal_geode(parts, clip_to_skin), cull=True on grow_on/cluster/geode; caustic_concentration,
  HdriEnv.add_light / map_radiance, culled_union / bounding_radius, HybridSDF.
- +4 catalog cards, +14 tests (tests/test_crystal_render_rig.py). Collected 7,067 (was 7,053).
- Audits: reachability 0/0, catalog_gaps 0, skill_lint 0 (one does-length regression caught and trimmed).
- Still unwired: WGSL emitter rules for fold_fractal/octahedron on the llvmpipe adapter (sdf_depth_device).
- HTTP: the four touched verbs appear in GET /tools; `crystal_habits` round-trips through POST /invoke. Verbs that
  RETURN an SDF or an env object (crystal_single, env_add_light) are in-process faculties -- their results are not
  JSON; `find_capability` itself 500s over /invoke (Capability objects), pre-existing, worth a `suggest`-style
  serialiser next sweep. A grid cache keyed by PARAMETERS silently served stale geometry once (bbox unchanged,
  lining changed) -- the demo now keys the cache by a hash of the field's values on the probe grid.

## Sweep 154b -- Moose's review of the frames: rough rock, honest caustic occlusion, checker floor, grey backdrop

Four corrections from looking at the pictures, each landed as a default-off option:
- **"The outside of a geode is rough rock."** The rind is displaced by two baked holographic fBm lattices
  (procedural_noise, 48^3 lumps + 96^3 grain via sample_grid_fast -- 4 s and ~30 s once, read trilinearly through
  GridSDF) and the field is scaled x0.6 to stay a valid sphere-tracing bound. First try (lumps 0.035, no grain) read
  as a WET, shiny ball: big smooth bumps under a hard key ARE specular-looking. Smaller lumps (0.02), grain (0.016)
  and a pit-darkening albedo (the displacement value as an ambient-occlusion proxy) read as matte stone. Albedos are
  now FIELDS: relight_glass(floor_albedo=fn, opaque_albedo=fn) take callables P->(m,3).
- **"Caustics render on top of the geometry."** caustic_pass projects the floor pattern through the camera and
  does not know what stands on the floor. composite_caustic(receiver_mask=bake.floor) confines it to pixels that
  show the floor (normalisation computed before masking, so hiding part of the pattern rescales nothing).
- **The geode threw a caustic THROUGH its rind** -- a bright ring under an object that should cast a shadow.
  caustic_hits / spectral_landings gained `occluder=`: rays the rock stops before the glass, and refracted rays it
  stops before the receiver, are invalid. Result on the bowl: 125,328 landings -> 25 (honest: light through crystals
  backed by rock does not reach the floor).
- **Checkerboard floor, solid grey backdrop, HDRI only as light.** relight_glass(background=(r,g,b)) paints the
  pixels that see nothing; every other read (floor irradiance, glass exits, reflections) still goes through the map.
  A checker to the horizon at 2 AA samples is moire, so the demo fades the albedo to its mean with distance.
Also: `skin_margin` on crystal_geode (0.06 with a displaced rind -- dents in the rock exposed clipped crystals at
0.025). Fourth HDRI from Moose (rosendal plains): measured share 0.87, sun at 52 deg elevation, radiance peak
112,884 -- a hard sun is what dispersion and caustics want; the extra key lobe is dropped to 0.3x there so the map's
own sun stays the caster. +3 tests (18 in test_crystal_render_rig), +1 catalog card.

## Sweep 155 -- "our crystals are pure": imperfections as one hypervector, mixed minerals, and what SIGGRAPH says

Brief (Moose): impurities, inclusions, cracks, bubbles; lattices honoured when minerals mix; growth from any
surface/field (geodes inside, caves and effects outside); every gem/glass/liquid property holographic, not the
known path; research SIGGRAPH through Aug 2026 for speed and quality of transparent materials. Reference photos:
geodes paved shoulder-to-shoulder with small milky points over WHITE AGATE BANDS inside a rough brown rind;
amethyst clusters purple at the tips fading to white bases; glass renders whose realism is dispersion fringes,
volumetric glow and stress-birefringence colour.

### Audit (rule 0)
The flaw FIELDS existed -- crystal_cloudiness / inclusions / fractures / phantoms / chipped -- but were wired only to
the Monte Carlo tracer's 8-channel callback (crystal_flawed_material). The fast holographic path (bake_glass +
relight_glass) rendered every crystal pure. grow_on already grows on ANY sdf gated by any `where` field (that IS
the cave / effect case; a geode is inward=True); it took ONE habit per growth. `habit_sdf` was the single-crystal
builder (wired last sweep as crystal_single).

### 1. Imperfections, holographically (holographic_gemvolume.FlawVolume, verb gem_flaw_volume)
The flaw density is ONE FPE hypervector (a bundle of encoded lattice samples, or the inclusion blobs themselves),
and the interior of every pixel's path -- entry, TIR bounces, exit, which the bake already follows
deterministically and now RECORDS as segments for the middle wavelength (bake.segs) -- reads its optical depth in
closed form via holographic_volint: one inner product per segment, no marching through the crystal. relight:
T = exp(-sigma tau); glow = albedo * E_amb/pi * (1-T); the same machinery with a chromophore density and a per-RGB
sigma is COLOUR ZONING (absorb_volume): amethyst purple at the tips, white base -- a uniform absorb cannot say that.
Kept negatives, measured: (a) HolographicVolume's default calibration probes from a corner and saw nothing of a
sparse field (scale 1.8x off); calibrate on rays THROUGH the samples. (b) Calibrating against the bundle's own KDE
read (a normalised cosine) gave tau ~0.004 for a field of ~0.7 over a unit path -- invisible; calibrate against
the TRUE density function: 12% vs a 400-step march on a Gaussian-blob fixture (the encoder kernel is not a
Gaussian; the SHAPE is exact, one scale cannot fit every ray). (c) Crosstalk floor ~10% of a through-blob read at
dim 4096 -- the bundle capacity limit again. (d) A segment from a trapped ray that hit max_dist read tau = 1.6e25:
segments longer than the box diagonal or starting outside it are dropped. (e) White furnace: a uniform environment
is INVARIANT under the glow (T + (1-T) = 1) -- the catalog example demos with albedo < 1. Cost: from_field res 22
= 10k encodes, 5-15 s once; the closed-form read for 30k glass px x 9 segments at dim 2048 ~ 13 s.
Small crystals need large sigma: a 0.11 druse point's interior path is ~0.065, so sigma 14 makes it milky; the
0.42 cluster uses 3.5. The density is [0,1]; sigma carries the physical scale.

### 2. Mixed minerals (grow_on habit=list | callable, size={habit: s})
Each seed draws or is assigned a habit; every habit keeps its own Bravais system and forms; culling takes one
bounding radius per habit. The single-name path is byte-identical to the committed code (pinned). Reference
geode lining: ("quartz","quartz","quartz","needle") at 340 seeds, size 0.115 -- paved, not dotted.

### 3. Reference-driven geode
Dense mixed lining; agate banding as a radius-keyed albedo under the crystals (white-grey rings, wavy by the same
fBm); rough brown rind; milky bases + purple tips through the flaw and zoning volumes. Floor caustic honestly
empty (rock blocks the sun; only the key lobe reaches into the bowl).

### 4. Research (SIGGRAPH / JCGT through Aug 2026) -- what transfers to the holographic path
- JCGT 15(1) 2026 "Ultrafast Screen-Space Refractions and Caustics via Newton's Method": replace ray MARCHING with
  Newton on a tangent plane -- 2-6 iterations, 18-79x over marching. Our costume: for a CONVEX faceted habit the
  exit point is an exact ray/plane intersection over its forms (min positive t) -- ONE step, no march. Named next
  wiring for _march_through on habit bodies (cluster: leave one member, test membership of the union, continue).
- SIGGRAPH 2025 "Bernstein Bounds for Caustics" (Fan et al.): position and irradiance BOUNDS per primitive tuple
  on the Bernstein basis, then sample tuples by bound -- sublinear in triangles. Our costume is already half
  there: culled_union's bounding spheres are a bound; caustic_hits could sample emitter cells by a facet-irradiance
  bound instead of a uniform grid. Limit they name: <= 2 specular vertices, no Fresnel/visibility.
- SIGGRAPH 2025 "Solving PDEs in participating media" (volumetric walk-on-spheres): the volume as a field to be
  queried, not marched -- the same instinct as holographic_volint; ours is closed-form for the FPE class.
- SIGGRAPH 2025 "Multi-Dimensional Procedural Wave Noise": band-limited wave noise as a sum of planar waves -- that
  is an FPE bundle by another name (our procedural_noise); a costume, not a gap.
- SIGGRAPH 2026 "HoloPathTracer": Monte Carlo over the rendering equation and the Rayleigh-Sommerfeld integral at
  once -- wave optics AND light transport. The optical-correspondence note's kin; worth reading for the
  stress-birefringence colours in Moose's references (a polarisation state per ray, not in leCore yet).
- Guy & Soler 2004 "Graphics Gems Revisited" (the gemstone paper): precomputed facet adjacency, polarisation,
  dispersion on GPU -- the exact-facet-exit idea above is its core, 22 years on.
Not found: any 2025/26 paper on procedural crystal growth or inclusion rendering -- our grow_on/flaw volume have no
published peer to measure against; the baseline stays the MC path tracer in this repo.

### Deltas
- holographic_gemvolume.py (new; selftest), GlassBake.segs, relight_glass(flaw_volume, flaw_sigma, flaw_albedo,
  absorb_volume, absorb_volume_sigma), verb gem_flaw_volume; grow_on/cluster/geode habit list|callable + size dict;
  +2 catalog cards; +5 tests (tests/test_gem_imperfections.py).
- Still to wire: exact facet exit for convex habits (Newton costume); polarisation/birefringence; the geode's
  cavity self-occlusion of ambient; a cave/effects demo of grow_on on an exterior field.
- Addendum: `opaque_bounce` (+ `opaque_bounce_where`) on relight_glass -- the one-number radiosity closure for a
  concave cavity, E += rho/(1-rho) * <direct>. Measured need: the white chalcedony cavity wall rendered BLACK behind
  bright crystals (wall direct E ~3 vs floor 14.6 under the key lobe; 70% of its shadow rays hit crystals). With it
  the wall reads white as in the reference photos; confined to the cavity because the convex rind went grey when it
  was applied everywhere. +1 test (6 in test_gem_imperfections).

## Sweep 156 -- "the crystal render looks terrible": judge the optics on a simple body first, then fix the crystals

Moose's verdict on the cluster frame, with references (smooth glass toys under soft studio light; jade-like foggy
glass; amethyst plates; paved geodes). Two tracks.

### 1. A material test scene (demos/gem_material_test.py)
One smooth blob (sphere + rounded box + small sphere, smooth-unioned) on a checker, plains HDRI as base, a SOFTBOX
key (Gaussian lobe sigma 0.28 rad -- broad highlights like the references, where sigma 0.045 gave pinpricks) and a
small kicker. Diamond: refraction of the checker, caustic pool with a spectral rim (concentration 4.4x measured),
fire. Two defects and their cures, both default-off flags:
- CONFETTI: coloured speckle in the TIR-heavy lower body. Cause 1: 11 hero wavelengths landing on different checker
  squares -- aliasing in lambda. Cure: `bake_glass(samples>1, lam_jitter=True)` shifts each sub-bake's wavelength
  grid by i/samples of one spacing -- 39 wavelengths from 3 x 13, no extra cost per sub-bake. Measured: reduced, not
  gone. Cause 2 (the one that mattered): a single wavelength whose chaotic multi-bounce path exits straight at the
  key reads the lobe's PEAK -- a saturated speck that thousands of MC wavelengths would average away. Cure:
  `relight_glass(footprint_filter=True)` -- the spectral ray differential: per pixel, the spread between
  neighbouring wavelengths' exit directions; the floor albedo is read over that footprint (angle x distance, 5 taps)
  and each analytic lobe is read fan-averaged with sigma_eff^2 = sigma^2 + spread^2 at conserved energy
  (`HdriEnv.radiance(blur=)`). Confetti gone, fire kept. Zero spread is byte-identical (pinned).
- Foggy quartz (the jade reference) through the flaw volume: the caustic must be scaled by the median interior
  transmittance -- a body that scatters light does not focus it. Relight with the closed-form reads is the cost
  centre now: 186 s at 800x500 x3 (px x segments x dim); the next lever is caching the per-pixel phase matrix
  across segments.

### 2. The crystals
- The cluster was a ball of interpenetrating prisms. Rebuilt as the reference: a rock slab (opaque) with short fat
  points growing UP from its top face -- grow_on(slab, where=top-face weight, habit="druse", substrate=False) -- with
  a white quartz crust on the slab, milky bases and purple tips from the flaw/zoning volumes.
- The geode was bald: measured, 560 "quartz" points of size 0.115 paved 32% of the 0.72 cavity wall -- each is a
  needle 0.07 wide. New habit "druse" (prism radius 0.55 of c instead of 0.30): 900 at size 0.16 pave 99.1%
  (pinned by test; 1400 at 0.14 pave 99.5%). Real druse is short and fat because it grew shoulder to shoulder.
- crystal_grow_on exposes cull=; camera framing per subject; holographic caustic at grid 12 / dim 2048 (crisper).

### Deltas
- bake_glass(lam_jitter), relight_glass(footprint_filter), HdriEnv.radiance(blur), HABITS["druse"], grow_on verb
  cull=; +1 catalog card; +4 tests (tests/test_spectral_antialias.py); demos/gem_material_test.py.
- Addendum (speed of the closed-form read): the flaw relight is O(px x segments x dim) in transcendentals. Rewriting
  the complex form as real trig was a KEPT NEGATIVE (10.9 s vs 9.5 s for 30k rays at dim 2048 -- the count of
  transcendentals is the cost); the same real form in float32 (`optical_depth(real_form=True, single=True)`) is 2.7 s
  at 8e-4 relative, deterministic -- gemvolume uses it. The cluster frame at 1280x800 x2 relit in 900 s before this.

## Sweep 157 -- "the amethyst renders look terrible": formation physics first, then the material

Moose: is the light colour supposed to be at the bottom? why does the geode look so bad? Look online, strive for
physical accuracy of formation and material. Sources read: Wikipedia/Amethyst (colour from Fe3+ colour centres
after irradiation; "the most intense colour typically found at the crystal terminations"; geode amethyst "often lacks
prism development, displaying primarily pyramidal terminations"; transparent to translucent, vitreous; hosts are
basalt geodes lined with agate), geologyin.com (Fe + gamma irradiation -> centres absorbing green/yellow -> purple).

### Answer to the question
Yes: pale/white bases and purple terminations IS amethyst -- the chromophore concentrates where the crystal grew
last. What was wrong was not the zoning but everything around it: the "white" read as flat plastic FOG, the
crystals read as broken glass, and the geode's lining was not what a geode grows.

### The measured cause of the bad crystals: habit proportions
`crystal_habit` sizes are distances of the (100) prism and (101) pyramid forms from the centre. The (101) planes cap
the c-axis at z = 1.5 x size WHATEVER the prism radius is -- so "druse" (0.55) and "amethyst_druse" (0.80), meant
to be stubby points, were fat DRUMS 1.10 x 1.50 and 1.57 x 1.50 (probed at size 1). That is why the geode showed a
"crystal ball" filling the cavity (700 drums 0.38 long from a 0.72 wall) with a smooth white ring around it, and why
the plate read as shards. A pyramidal termination needs the pyramid distance BELOW the prism's. Probed:
quartz_point (0.45, 0.60) = 0.90 x 0.90, 53% pyramid; amethyst_pyramid (0.55, 0.50) = 1.05 x 0.75, ~70% pyramid.
The drums stay in HABITS as kept negatives (documented in place); the two new names are the correct ones. A placed
crystal is centred at its root and sunk 0.35 x size, so ~0.55 x size projects past the substrate for quartz_point --
sizes are now chosen from that: geode lining 600 points at 0.13 (0.07 tall, 0.117 wide, 4x cover), plate 110 at
0.22. amethyst_pyramid at 0.11 projected only 0.4 x size and the wall buried it -- sparse patches; kept.

### Material
- Colour zoning as the physics says: chromophore density (1 - base weight)^k along each crystal, per-RGB
  Beer-Lambert (5, 14, 3.4) -- absorbs green hardest, then red, blue least -> purple, not violet-blue.
- The haze is now DIRECTIONAL: the single-scatter in-scatter takes the dominant lobe's share of E with a shadow ray
  from the entry point and a wrap cosine on the entry normal (glass 0.3, rock blocks), the ambient share isotropic.
  A frosted body has a lit and a dark side (pinned: 1.3x). The noise "cloudiness" was dropped from the milky field
  -- at dim 2048 its speckle read as frost; the base weight alone is smooth.
- No floor caustic for a paved druse (a paved surface spills light; the smeared pool looked like a stain).
- Geode rind: roughness applied to the OUTER skin only (the cavity wall is where chalcedony was deposited -- smooth);
  displacing the whole shell bulged the wall inward by 0.07.

### Deltas
- HABITS quartz_point, amethyst_pyramid (+ negatives documented on druse/amethyst_druse); directional haze in
  relight; +2 tests (8 in test_spectral_antialias). Demo updated.
- Addendum (Moose's pick): the sweep-156 DIAGNOSTIC -- the long "amethyst_druse" drums rendered as plain grey opaque
  under the softbox -- was "the most correct result so far, except it doesn't have materials". So the drum habit is
  not only a negative: 700 long crystals radiating from the wall to a coarse mass in the middle, sliced flat by the
  cut, reads as a cut-and-polished amethyst geode. Kept as subject `geodeball` in the demo with the amethyst
  material (zoning: milky at the wall, purple toward the centre; flat grey floor and backdrop as in the diagnostic).
  Honest: the central mass is paler than a Uruguayan specimen (haze 4.0 on long paths) and the sliced crystals at the
  rim read as clear glass tiles; the smooth soft-lit look of the diagnostic came from opaque Lambert + bounce, which
  the glass cannot have -- the two looks are different physics, and Moose's eye preferred the coarse geometry.

## Sweep 158 -- "no fake billboard planes": what the crystals ARE, real cells, library optics, honest caustics

Moose: no cheap tricks; physically accurate lattices; real crystals with the material library's properties;
dispersion and caustics. For the record, first: there are no billboards or sprites anywhere in the crystal path and
never were -- every crystal is `crystal_habit`, the intersection of half-spaces whose normals are Miller-index
directions in the reciprocal lattice basis, a true convex distance field (unit gradient pinned by test). What LOOKED
like flat tiles was the Fresnel reflection of a flat grey backdrop on planar facets. Three things were nonetheless
not accurate, and are now:

1. THE UNIT CELL. `lattice_basis` defaults to c = a. Quartz has c/a = 1.1001, and the one number every mineralogy
   text quotes -- the prism-rhombohedron interfacial angle m^r = 141 deg 47' -- came out 139.1 deg from the default
   cell and 141.79 deg from the real one (selftest + test). `CELLS` carries the axial ratios (quartz 1.1001, beryl
   0.9976); `real_cell=True` on habit_sdf / grow_on / cluster / geode / crystal_single / crystal_grow_on. Default off
   so every shipped field stays byte-identical -- Moose asked for accuracy BY DEFAULT; that flip is a versioned
   decision to make deliberately, and the flag is how the demo and all new work get it now.
2. THE MATERIAL. Every optical constant in the render comes from glass_optics: n_d 1.55, Abbe 70 (quartz's real
   dispersion is faint -- a physically honest amethyst does not fire like a diamond), and the per-RGB absorption
   (1.1, 2.3, 0.55): green absorbed hardest, the Fe colour centre. Colour zoning is now that same library absorption
   REDISTRIBUTED along the crystal (0.25x everywhere, the rest as a chromophore density at the terminations) --
   the hue is the library's, only its placement follows growth. My earlier hand-typed RGB sigmas are gone.
3. THE CAUSTIC. Re-enabled for every subject as real refracted landings (rock occludes), and switched to the
   HISTOGRAM baseline: measured on the plate's 25,232 landings, the tiled holographic read (grid 12, dim 2048)
   smeared the filaments into blotches and showed its tile seams; the histogram resolved sharp filaments with
   dispersion at their edges (test: a point source reads >3x sharper). The holographic caustic keeps its place as
   the research path with its capacity limit written down; below that capacity, bin. n_side 1100 (1.2M rays through
   the grid-baked lining) for a smooth field.

Deltas: CELLS + real_cell (grow_on, verbs), histogram_caustic_rgb + verb caustic_histogram_rgb, +2 catalog cards,
+5 tests (tests/test_real_cells_and_library_optics.py). Kept honest: the plate's purple is pale because the library
absorption is per unit length and these crystals are 0.2 units -- the unit is the knob, not the material.

### Frames (sweep 158) and what they say
- Quartz point (real cell): the 141 deg 47' termination, the checker refracted through it, dispersion fringes as
  faint as Abbe 70 makes them, a crisp caustic with spectral edges. This one is physically honest end to end.
- Diamond octahedron: fire and internal reflections; caustic present but soft (the histogram of 1.2M rays at 1.2 px
  blur -- a diamond's caustic wants a smaller emitter cell; measured next).
- Amethyst plate: pyramidal points, library purple, crisp caustic filaments. Still glassy inside: the union has no
  contact interfaces -- which for SAME-mineral contacts is physically right (n1 = n2, no refraction, ~0 Fresnel), so
  the missing realism is the grain-boundary inclusions and multiple scattering, not a missing surface.
- Geode: the weakest. The far-wall crystals read as mirror TILES because head-on planar facets reflect a FLAT grey
  backdrop; and with 0.07-tall points the wall shows between them. Real geode linings hide the wall completely
  (heights vary, crystals lean) and reflect a structured room.

## PROGRAM (from Moose, sweep 158 close): "we are an n-D engine -- more realistic than V-Ray"
Everything sits in superposition; collapse to the pixel; LIFT to solve. Where the remaining realism gap is, and the
lever for each -- measurable, in order:
1. Interreflection / GI between crystals and cavity (the wall glow, the light bouncing inside a druse): lift to the
   4-D (x, omega) light field as ONE FPE vector over the cavity (the PRT lineage: radiance_transfer, RENDER=QUERY) and
   read it for every glass exit, opaque point and haze glow instead of the one-number bounce. Baseline: the MC path
   tracer at high spp. Metric: mean abs error vs MC on the cavity wall.
2. Multiple scattering in the haze: lift the single-scatter closed form to the diffusion regime -- the flaw density
   bundle is already a field; its second moment along the segment is another closed form (lever 4). Baseline: MC
   volumetric path tracing of the same density.
3. Facet micro-roughness and grain boundaries as a 5-D (x, omega_i, omega_o) transfer, baked once per material:
   what turns mirror tiles into the vitreous lustre of a real face. Baseline: GGX MC.
4. Polarisation / birefringence (quartz IS birefringent, n_o 1.544 / n_e 1.553): a 2-vector per ray (Jones),
   bundled -- the stress colours in Moose's glass references. Baseline: published Mueller-matrix renders.
5. Speed: exact facet exit for convex habits (one ray/plane step, the Newton costume) and a WGSL emitter for
   `crystal_habit` on the llvmpipe adapter -- the bake is the cost; PRT reads are already O(dim).
V-Ray's bar is unbiased spectral path tracing with GI; ours is a baked deterministic transport with closed-form
volumes. Above 1-4 the comparison becomes fair; every step ships with its MC baseline or it is not a result.

## Sweep 159 -- "jagged noise isn't crystal formation": competitive growth

Moose, on the plate: "absolute trash ... jagged weird noise isn't crystal formation". Right. The cause was the
growth model, not the material or the render: grow_on drew every crystal's size INDEPENDENTLY of its neighbours, so
110 points of size 0.22 on a 1.24 x 0.84 plate had a footprint 22x the plate and the union of that many
interpenetrating polyhedra is a field of shards. In a real druse a crystal grows until it meets its neighbours: its
size is the room around its seed. `grow_on(pack=k)`: size_i = k x (mean distance to the 3 nearest seeds -- the
Voronoi cell radius; the single nearest neighbour alone touched one side and left bare rock everywhere else,
union volume 0.015). With the "quartz" habit (radius 0.58, half-length 1.5 x size) pack=1.0 makes neighbours meet
along their prism faces and project ~1.15 x size -- height ~ width, as on a real plate. 70 seeds on the slab, 320
on the geode wall, real cells, library optics, histogram caustics. The frames now show individually legible
terminated points (pinned: packed union volume < independent; per-seed room varies >10%). Default pack=None keeps
the old behaviour; it should not be the default anyone reaches for -- the flag is documented as the formation.
Still open, honestly: seeds only where the `where` field allows, so slab edges stay bare; crystals seeded on the
rounded slab edge lean outward; the geode's far wall still mirrors a flat backdrop.

### 159b -- Moose: "light through a coloured gemstone would have a colour tint"; "it illuminates the base more than I expect"
Both right, and both were the SAME missing physics. caustic_hits was the historical thin-lens: refract at ENTRY,
then straight to the floor -- no exit refraction (so the light landed at the base instead of where a prism sends
it) and no path length (so nothing could tint it). `caustic_hits(through=True)` marches the interior to the exit
face, refracts out (TIR reflects and continues, cap max_internal, trapped rays invalid) and returns the interior
path length per ray; `spectral_landings(through=True, absorb_rgb=library, cmf=...)` weights each landing by
exp(-sigma(lambda) L) with sigma(lambda) the library's per-RGB absorption spread over the spectrum by the colour-matching
weights; `histogram_caustic_rgb(weights=)` bins them; the composite strength is scaled by the mean transmission
(absorbed light is not on the floor). The relight's shadow rays now go through ONE `_occluded`: glass passes
shadow_transmittance x exp(-absorb x chord) over the marched chord, rock blocks -- an amethyst's shadow is purple,
a ruby's red (pinned), and the floor is also shadowed by the opaque body now. Measured on the amethyst point: the
caustic moved off the base to where the exit refraction sends it and turned purple; strength 0.46 at 2.1x
concentration (was 4x with the thin lens: the exit spreads it). Default-off flags: thin-lens callers unchanged.
Reference: Beer-Lambert transmission through a coloured medium selects the transmitted colour -- what a red glass
passes is red; the gemology causes-of-colour references list amethyst's Fe colour centre absorbing green/yellow.

## Sweep 160 -- a cluster on a small rock, lit like a museum piece; growth stops at the root; no more geodes for now

Moose: crystals growing below the surface they grow from; intersecting geometry is sloppy; new interior HDR (dim,
base light only: share 0.16, E 2.4); museum lighting; match a reference photo of a cluster.
- BELOW-SURFACE GROWTH, found and fixed at the source: a placed habit is a bipyramid centred near its root, so its
  lower half ran into the rock and, on a 0.22-thick nodule with 1.35-long points, OUT THROUGH THE BOTTOM -- rendered
  as prisms hanging beneath the rock. `grow_on(clip_to_substrate=True)` cuts each crystal by the plane through its
  own root perpendicular to its c-axis AND subtracts the host volume (exact CSG on distance bounds). Pinned: zero
  crystal volume inside the host and none behind the root plane; without the flag it does go below.
- INTERPENETRATION: pack sizes by the Voronoi room; overlap is then a knob with a physical reading (intergrowth). The
  random emitter on a rounded crest gave a CROWN of outward-leaning sticks -- a matrix specimen grew on a roughly
  planar wall, so: a flat-topped nodule (ellipsoid sliced at y = TOP, sides rough, top near-flat) and, for the
  specimen composition, EXPLICIT SEEDS (`grow_on(seeds=(P, N), size=[...])`): a dominant centre, an inner ring of 5,
  an outer ring of 6, c-axes fanning 12-20 deg outward. Everything else physical: quartz_long habit (new: prism 0.22,
  h/w ~1.35, the Brazilian point) on the real cell, library quartz, faint milky base as a flaw volume, full-transit
  spectrally tinted caustics, footprint filter.
- Lighting: Voortrekker interior as base + softbox key (sigma 0.26, 4.5x E_map) upper-left-front, soft fill right
  (1x), hard rim behind (2.4x) -- the terminations fire.
- Reference matched: a Brazilian clear-quartz cluster on matrix (Australian Museum listing: "multiple clear quartz
  points growing in harmony from a common matrix ... well-formed terminations that radiate from a central core",
  neutral background). Honest gap: ours reads glassier/wirier than the photo -- edges catch the light, faces less;
  the rock is a smooth nodule, not a fractured matrix; no grain boundaries between intergrown points.
- Deltas: clip_to_substrate, seeds=, per-seed size, quartz_long habit; +1 test (9 in
  test_real_cells_and_library_optics); demos/cluster_museum.py.

## Sweep 161 -- the .lews STANDARD: versioned kinds, canonical sections, a live workspace apps edit together

Moose handed over polystudio_standalone.zip: find what was hand-rolled because the engine lacked it, and make the
`.lews` workspace a standard every leCore app can hook into -- versioned, so data is read/written in the right
format -- with an image editor and a modeller seeing each other's work.

Audit (rule 0): `.lews` is leCore's own `holographic_container` (typed sections, unknown kinds round-trip); Poly
Studio and leStudio already trade files that way; polystudio's `ccrun` C-runner is already upstreamed as
holographic_ccrun. What did NOT exist: a schema version per kind (the container had one number for the whole file),
canonical kinds beyond lecore.image (each app pair needed its own adapter), and any LIVE sharing -- export/import
only. Full table in docs/POLYSTUDIO_AUDIT.md; the app also duplicated UV unwrap, midpoint subdivision, banded grid
bake and a render cache the engine has (a discoverability failure -> aliases), and built one thing worth taking:
quality_gate.py, absolute-threshold render regression metrics.

Built: holographic_lews (io_and_interop). LEWS_SPEC "1.0" meta on the file; `meta["schema"]` per section with
register_kind_schema(kind, version, migrate={old: fn}) and upgrade_section (older -> migrated; NEWER than the build ->
`_read_only`, carried, refused for write); canonical lecore.mesh / material (physical-library name + overrides, name
validated against matlib) / sdf (dialect text) / camera / scene (bindings by section id); `Workspace(root, app)`: a
directory whose workspace.lews is always a complete container, puts locked (O_EXCL lock, stale-breakable), re-read,
atomic (os.replace), journalled (journal.ndjson: rev, op, id, kind, app, sha, t); changes_since(rev),
wait_for_change, put(expected_rev) -> ConflictError. Mind verbs lews_open / lews_describe / lews_changes /
lews_*_section / lews_register_kind; catalog card; docs/LEWS_SPEC.md.

Measured: demos/lews_two_apps.py runs a painter and a modeller as SEPARATE PROCESSES on one directory; the modeller
publishes mesh + scene, the painter repaints the texture three times, the modeller catches each change by revision
and re-renders (3 frames, journal 5 entries, no lock left behind). Tests: 6 (test_lews_workspace) + selftest.
Kept negatives: no merge of two edits to one section; polling not push; whole-file rewrite per put (shard by section
when measured to hurt). Next from the audit: path_trace(region=, base=) (LC-1), quality_gate upstream, edit_history
branch-and-replay.

## Sweep 162 -- leStudio audit: the workspace becomes the live session, journal-first kinds, five engine bugs fixed

Moose handed over lestudio.zip (the image editor, R54, 570 tests, 26 backlog docs): "swarm sessions have been tested
many times... hand rolled tech that needs to make its way back to leCore". Audit (rule 0) before building: its
`.lews` is already the engine's container; LECORE_BACKLOG items 1-12 already landed; a prior sweep had lifted the
CONTRACT of its multiplayer (`live_session`, in-process) but nothing cross-process, and sweep 161's Workspace had a
journal but no presence -- two halves of one thing, never joined. Full table: docs/LESTUDIO_AUDIT.md.

Built (holographic_lews, extended): the Workspace now realises LiveSession on the directory. `bump`/`lews_note`
appends a journal line WITHOUT rewriting the container (rev = max(container meta rev, journal tail rev); both advance
under the one lock -- pinned: size and mtime of workspace.lews unchanged by a note, and the next put continues the
same counter). `touch`/`roster`/`participants`/`drop`: presence as heartbeat files `presence/<sha(who)>.json` with
activity, name, app, joined; host = earliest joined still alive. `since(rev, exclude)` drops your own echo.
`Workspace.from_file`/`lews_import` opens any app's single-file .lews as a live directory; leStudio's golden fixtures
now live in tests/fixtures/lews and are pinned to open and hash-identically survive another app's put. Journal-first
kinds: `lecore.asset` (content-addressed, put_asset stores identical bytes ONCE -- no write, no rev) and
`lecore.journal` (JSON ops, explicit seeds, asset keys; inline arrays REFUSED), `journal_asset_refs`, `gc_assets`.
Verbs: lews_note / lews_wait / lews_touch / lews_presence / lews_leave / lews_import / lews_put_asset /
lews_get_asset / lews_journal_section / lews_gc_assets (all round-tripped over POST /invoke). Two catalog cards.

Lessons carried from leStudio's R-series, now in the module rather than in one app's server: key presence by PERSON
not connection (five reloads gave one user five ghost chips); host = earliest joined still present (the role flapped
on reload); a stream that only writes on change never learns its socket died (a heartbeat with a ttl has no such
failure mode); the document is an op journal, pixels are a render (one stroke: ~21 MB snapshot, 0.13 MB windowed,
~2.7 KB as a path record).

Engine bugs leStudio reported and worked around, fixed at the source with tests/test_lestudio_reported_bugs.py:
pattern_image returned flat zeros for marble/wood/brick/voronoi/musgrave/wave/magic (those names live in proctex;
make_pattern now routes them, pattern_image raises on a name in neither menu); sky_model(sun_intensity) scaled only
the 1.4-degree disk so a 200x120 sample grid saw nothing -- the glow scales with it, relative to the default so
renders at 18.0 are bit-identical; make_cloud rejected the dict camera render_mesh accepts (as_camera); sharpen_image
was 1-D only (rfft along the last axis with a first-axis frequency grid -- square images blurred rows only, RGB
blurred across channels; now n-D separable, 1-D bit-identical); morph_scene needed square images (a separable DCT is
two matrices). Measured honestly: Van Cittert on a hard edge reaches 0.60 of the blurred error at 40 iterations and
0.54 at 300 (Gibbs) -- the test pins 0.65, not a wish.

Deferred to sweep 163 with reasons on record: HoloScore v2 (score-based one-shot patch diffusion; imports PIL, hard-
coded stills, four measured laws to pin) and importance-ordered anisotropic splat pursuit. Kept negatives from the
app's own measurements (pattern_image 0.5x its grid path; load_image 10x slower than its decode; a per-frame content
hash cost more than it saved) recorded in the audit doc so they are not re-litigated.

Tests: 7,096 -> 7,108 (+7 lews, +5 reported bugs). Cards 843 -> 845. Modules unchanged (803).

## Sweep 163 -- the app foundation: what both apps built twice, provided once (leStudio + Poly Studio audit)

Moose: "do a thorough sweep of both the 2d and 3d apps... find anything we need to bring into leCore itself and
make standard across all implementations... swarms, agents, file formats, ways of representing / partitioning /
generating information... perhaps more documentation." Method: booted the mind and audited both trees through
file_grep (24 cross-cutting probes: routes, identity headers, jobs, capability gating, .lews, ids, undo, determinism,
presets, assets, schema doors, quality, cache, SSE, memory, GLSL, timeline, materials, export formats, partitioning,
graphs, locks, re-implemented helpers), then find_capability per finding. The pattern: the SAME layer built twice
in mirror image -- leStudio has identity/presence/SSE/jobs/capability gating and Poly has none; Poly has a url_map
tool manifest with base64 image returns and a render quality gate, leStudio has a manifest without them; leStudio
mints ids from process-global counters (its P0.3), Poly reassigns ids on load; both wrote Gaussian blurs (three in
leStudio), both wrote undo stacks, both ship their own presets shape.

Built (all wired, carded, tested; audits 0/0/0):
- holographic_appserver (io_and_interop): AgentSurface -- the union of both apps' agent doors, mounted on a Flask app
  in one call (m.agent_surface): GET /agent/tools from the LIVE url_map filtered to THIS mount (Poly's gallery lesson:
  blueprint names lie, the request path does not), POST /agent/invoke with base64 PNG for image routes (Poly 1.2.0:
  "the only way an agent sees its own work") and a refusal-with-path for streams, POST /mind allow-listed to read-only
  engine faculties (leStudio: a rejected name returns the allowlist WITH signatures), GET /engine (m.engine_status:
  version, faculties, extras, GPU report, determinism policy, CPU budget), GET /events SSE with the 2 s comment ping
  (leStudio's ghost-socket fix) and GET /presence; an after_request hook turns every mutating request into a .lews
  note and heart-beats X-User (presence by PERSON, never connection). Flask imported only in mount(); the manifest,
  identity, JSON coercion and SSE generator are plain functions the selftest runs without it. LiveSession.bump and
  Workspace.bump gained touch= (default True) so a per-run client id never lands in presence.
- holographic_lews: Workspace.mint(prefix) -- one persisted counter per prefix (ids.json) under the lock, journalled;
  lecore.preset kind (preset_section refuses arrays). Verbs lews_mint / lews_preset_section.
- holographic_qualitygate (rendering): Poly's absolute-threshold render gate upstreamed -- terracing (2nd difference
  in the floor band), edge_tones, fringe_ratio, gate(). Edge metric REBUILT with measurement: Poly's top-1%-gradient
  mask tied the score to how much of the frame is edge (same supersampled disc: 0.92 at 160 px, 0.35 at 640 px), and
  its absolute 0.05/0.95 tone bounds scored the engine's own jagged rasteriser 1.0 because the default background is
  0.06 grey. Now: mask = gradient > 30% of the frame max, partial = between the LOCAL 5x5 min/max by a 15% margin, in
  a 3x3 neighbourhood -- 0.93-0.96 across sizes and grounds, 0.00 jagged. FIRST CATCH ON THE ENGINE: render_mesh does
  not antialias silhouettes (edge_tones 0.07 at 160 px, 0.02 at 640 px) -- pinned as a known gap in
  tests/test_qualitygate.py, not hidden.
- blur_image verb: the engine held FOUR private Gaussian blurs (autobump, splatsharpen, sharpen, postfx) and exposed
  none -- the most re-implemented helper across the apps. Now one door (reflect = separable autobump, wrap = FFT).
- tools/app_lint.py + m.app_lint: lints an app tree for the patterns the foundation replaces (hash() seeds, class
  counters, own container / manifest / SSE / undo / jobs / gate, wall clock, legacy RNG, PIL) and the foundations it
  should be on (gating, X-User, workspace, /mind), and names the engine's nearest card + code hit for each hand-rolled
  helper. Measured: leStudio R54 5/16, Poly Studio 1.2.0 4/16 (reports in docs/app_lint/). It reproduced the manual
  Poly audit unaided (_auto_uv -> mesh_uv_unwrap, _midpoint_refine -> mesh_subdivide, fbm2 -> fractal_field).
- docs/APP_FOUNDATION.md -- the standard: one mind + preflight; the workspace as document, bus and session (kinds,
  ids, journal-first, presence); agents (mount, don't write); swarms (dispatch_roles, shared_workspace, registry,
  codebase_sync, serve/escalate); jobs; per-user memory (app_substrate); presets, gates, golden fixtures, fake
  engines; what stays in the app; a checklist. Every python block is executed by tests/test_app_foundation_doc.py
  (9 tests). Linked from docs/INDEX.md; LEWS_SPEC 4c added.

Kept negatives / open: a fake engine for app tests stays app-side (Poly's pattern is right; the engine's promise is
that features()/engine_status() are cheap enough for every test). Frame-source protocol (LECORE_BACKLOG 12) still a
design pass. HoloScore + aniso splat pursuit still deferred (sweep 162 note). find_capability is weak on plain image
verbs ("gaussian blur an image" ranked the Extended Gaussian Image first before the new card) -- the catalog needs a
pass of user-language aliases for the ordinary image toolbox; app_lint shows both the card and the code hit so the
weakness is visible rather than papered over.

Tests: 7,108 -> 7,135 (+6 appserver, +8 qualitygate, +3 app_lint, +9 doc snippets, +2 lews). Cards 845 -> 850.
Modules 803 -> 805.

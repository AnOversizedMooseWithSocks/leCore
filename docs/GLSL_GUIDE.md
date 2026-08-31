# Building with leCoreGLSL

**Prerequisite:** this assumes you already know what leCore is — hypervectors, bind/bundle/unbind,
cleanup, the catalog. If you don't, read [`README.md`](../README.md) and
[`docs/THEORY.md`](THEORY.md) first, then come back. Nothing below re-explains the algebra.

This document covers one question: **what does the GLSL layer actually give you, and what can you
build with it?**

---

## The one-paragraph version

leCoreGLSL is not a port of leCore to the GPU. It is a set of **ten verified shader kernels** —
each one differentially tested against an exact NumPy reference and shipped with the number that
verified it — plus five self-verifying browser pages that show them running. The engine stays
NumPy, and if you can run Python you should. This guide is for when you cannot — browser only — and
its main job is to tell you which half of your workload belongs in GLSL and which half belongs in
plain JavaScript, because **the measurements do not put the dividing line where most people assume
it is.**

Everything here has run in Chrome on real hardware (WebGL 2.0, RTX A4500). The numbers are from
that machine, not from a simulator.

---

## Start here: open the pages

```
pages/field_demo.html               START HERE -- type words, watch the field they describe
pages/lecore_search_webgl2.html     BM25 + containment + answer/set/abstain, 500 real passages
pages/lecore_webgl2_vsa.html        bind · bundle · unbind · cleanup, 6 shader programs
pages/lecore_webgl2_full.html       3-tier beam recall, every decision on the GPU
pages/lecore_webgl2_typed.html      type a query; it is encoded and recalled entirely in WebGL2
pages/lecore_webgl2.html            the smallest one: bind · score · tiled-argmax
```

Double-click any of them. No server, no build step, no network calls — the data is embedded. Each
page runs a PASS table against **f64 answers computed by the engine and embedded in the file**, so
a page that opens green has proved itself on your hardware, not on mine.

Read one of the generators next: `research/shader_retrieval/make_search_page.py` is the canonical
example of the whole pattern — take a corpus, build an index, emit a page that carries its own
reference answers.

To rebuild after changing a shader: `build_pages.bat` (or `.sh`). It makes a throwaway venv,
rebuilds all five, and opens the search page.

---

## Which substrate should you use?

Three real options, and the answer is not the same for every workload.

| you have | use |
|---|---|
| a Python backend | **the engine.** It is the reference implementation, it is faster, and everything else here is checked against it. |
| browser only, **retrieval** | **plain JavaScript.** Measured below — it beats the GPU path by two to three orders of magnitude at this scale. |
| browser only, **fields / physics / generative** | **GLSL.** This is where the GPU earns two orders of magnitude and JavaScript would be hopeless. |

If Python is available, use it. This guide exists for when it is not — a static site, a browser
extension, an offline app, someone else's runtime — and its job is to tell you honestly which of the
two remaining options fits your workload.

### The numbers, on hardware

GPU figures are RTX A4500 in Chrome (WebGL 2.0), median of repeated runs, correctness verified
against the engine first. JavaScript figures are Node on a different machine, so treat the
JS-vs-GPU comparison as **orders of magnitude, not a controlled benchmark** — the gap is far too
large for the machine difference to explain it.

| kernel | vs NumPy | verdict |
|---|---|---|
| `hdrift_grad` — sum of plane waves | **307×** | build on this |
| `diffuse` — five-point Laplacian, ping-pong | **134×** | build on this |
| `pbd_scatter_vs` — cloth constraints, Jacobi | **51×** | build on this |
| `raster_form` — linear image formation | 0.34× | it is a GEMV; only worth it if the result stays on the GPU |
| `scatter_bm25_vs` — inverted index | 0.26× | see below |
| `bm25_score` — full scan | 0.32× | see below |

### Retrieval in the browser: write it in JavaScript

The search page runs BM25 + exact containment on the GPU at **1.55 ms/query** (500 passages, 6,767
terms, including a readback). The same algorithm in plain JavaScript:

| passages | terms | index build | query |
|---|---|---|---|
| 500 | 6,767 | 33 ms | **0.002 ms** |
| 5,000 | 10,954 | 237 ms | **0.016 ms** |
| 20,000 | 15,533 | 545 ms | **0.076 ms** |

**JavaScript is roughly 300× faster than the GPU here, and it is still 20× faster at forty times the
corpus.** The reason is structural, not a tuning problem: a postings walk for a 2–4 term query is a
few thousand indexed multiply-adds — far too little work to cover a draw call and a readback. On the
A4500 an *empty* shader that reads back 262,144 floats costs 1.02 ms, which is most of the GPU
number above before any arithmetic happens.

So: **build browser retrieval in JavaScript.** The engine's BM25, the answer/set/abstain policy and
the proximity reranker are all a few hundred lines of straightforward code, and the search page
already contains a working JS port of the tokenizer, the packing and the reranker to copy from.

**How far batching moves it.** Scoring one query at a time pays the per-call floor every time, and
that floor is per-CALL, not per-byte -- an empty shader reading 4 floats costs 0.009 ms and 65,536
floats costs 0.249 ms. `research/shader_retrieval/glsl_batch.py` puts a whole batch in ONE draw
(query as a second axis, verdicts reduced on the GPU so nothing but the answers comes back) and then
takes three of the six levers: tile the reduce (T4 proves a tiled max equals the single-pass max),
pack two queries per RGBA texel, bake the idf table. Per-query cost went **0.46 -> 0.21 ms** with
top-1 identical to the engine at every step. JavaScript is linear in corpus and the GPU is flat, so
there IS a crossover -- extrapolating puts it near 10^5 passages on a software rasteriser, and it
has moved down with every lever taken. Treat that number as provisional until it is measured on
hardware.

**Then why do the retrieval shaders exist?** Two honest reasons, neither of them speed:

1. **They are differential tests.** They proved the algebra, the containment count, the policy and
   the reranker all reproduce the f64 engine exactly — 60/60 on mode, 60/60 on exact ambiguity
   integers, 58/58 on the reranked answer. That is what makes the JavaScript version trustworthy.
2. **They are the shape that scales past a browser tab.** The scatter kernel does 63× less *work*
   than the full scan; the wall-clock loss is per-call overhead, which amortises when the corpus
   lives on the GPU across many queries and the result never comes back to the CPU. Nobody here has
   built that; the honest status is "plausible, unmeasured".

> A caution earned the hard way: an earlier draft of this document would have claimed the scatter
> index was **106×** faster. That number came from a software rasteriser, where the full-scan
> baseline was a CPU pretending to be a GPU. On real hardware it is **1.3×**, and against JavaScript
> it loses outright. Measure on the device you ship on, against the alternative you would actually
> write.

### The demo: `pages/field_demo.html`

Open it and type words. Each one is hashed to a frequency and a phase, and the field you see is the
sum of those plane waves -- **fractional-power encoding, drawn instead of stored**. 16,384 particles
follow the field's gradient, repel each other off their own density, and leave a diffusing trail.

Four passes per frame and **zero readbacks**: drift (one fragment per particle, summing plane
waves), splat (one point per particle, additive blending), diffuse (five-point Laplacian
ping-pong), colour blit. Those are the three kernels that measure 307x, 51x and 134x against NumPy
on an A4500 -- and the demo exists because it is the honest showcase. The same page's *retrieval*
kernels lose to plain JavaScript, so a search demo would have been showing off the weak half.

Three details worth reading in the source:

* **The vocabulary is a function.** Type a word that has never existed and its wave appears
  immediately. Nothing is stored. The demo uses the ENGINE'S hash (`hash32_pcg`) and that is
  checked against Python -- substituting a different 32-bit mixer draws a field orthogonal to what
  leCore would draw, which is a bug that already cost this project a day of blaming the driver.
* **Repulsion is not decoration.** Turn it off with the button and watch every particle collapse
  onto one ridge. Attraction alone memorises; that is the same finding the HDRIFT work measured
  numerically, made visible.
* **`EXT_float_blend` is checked, not assumed.** Without it the accumulation buffer would silently
  be zeros, so the demo disables trails and says so in the status line rather than showing you
  nothing and calling it art.

### Where GLSL is the only sane answer

Everything above is about retrieval. For **fields, physics and generative visuals** the comparison
inverts completely — 8,064 cloth constraints × 40 iterations in 1.51 ms, a 512×512 diffusion for 40
steps in 2.20 ms, 4,096 particles × 2,048 plane waves in 1.22 ms. Writing any of those in
JavaScript means a per-element loop over hundreds of thousands of elements per frame, every frame.
That is the workload GLSL is for, and it is why this layer exists.

## The primitives

### 1 · Atoms are functions, not tables

The single most useful idea in the GLSL layer. An atom is not stored — it is **computed from its
name**:

```glsl
uint x = fnv1a_of_token ^ dimension_index;
uint s = x * 747796405u + 2891336453u;
uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
x = (w >> 22u) ^ w;
float value = ((x >> 31u) == 1u) ? 1.0 : -1.0;   // Rademacher ±1
```

That is the whole vocabulary. **Zero bytes shipped.** A 6,767-term vocabulary costs nothing because
every atom is regenerated on demand, bit-identically, in NumPy, JavaScript and GLSL.

**Porting trap, paid for once:** you must use *those* constants (`hash32_pcg`). Shipping a
different 32-bit mixer produces atoms orthogonal to the reference — recall collapses to chance and
it looks exactly like a driver bug. `mind.glsl_kernel('...')` hands you the verified source; use it
rather than retyping it.

### 2 · The three shader shapes

Every kernel in the set is one of three shapes. Learn these and you can write your own.

**Gather** — one fragment per output, reading many inputs.
Used by: `bm25_score`, `raster_form`, `hdrift_grad`, the beam walk.

```glsl
void main(){
  int d = int(gl_FragCoord.x) + int(gl_FragCoord.y) * uWidth;   // which output am I?
  float acc = 0.0;
  for (int k = 0; k < uK; ++k) acc += texelFetch(uData, at(d*uK + k), 0).r * ...;
  fragOut = acc;
}
```

**Ping-pong** — two textures, alternate read and write, one pass per step.
Used by: `diffuse`, `hdrift_step` (which pairs with `hdrift_spectrum`, a gather over particles
that rebuilds the batch field every step), any iterative solver.

**Scatter-add** — one *point primitive* per contribution, placed by the **vertex** stage, summed by
additive blending (`blendFunc(ONE, ONE)`).
Used by: `scatter_bm25_vs` (with `scatter_bm25_fs` as its fragment half),
`pbd_scatter_vs`.

```glsl
// vertex shader: I am contribution #gl_VertexID; where do I land and what do I add?
vDelta = my_contribution;
gl_Position = vec4(texel_of(target_index), 0.0, 1.0);
gl_PointSize = 1.0;
```

A fragment shader cannot scatter. A **vertex** shader can place a primitive anywhere, and blending
is a hardware scatter-add. This is the trick that makes physics constraints and inverted indexes
expressible at all.

**Two costs that come with scatter-add, both mandatory reading:**
- Blend order is unspecified and float addition is not associative, so **you give up
  bit-reproducibility**. Decisions survive (the margins are enormous); bit-identity does not.
- You must request `EXT_float_blend` and **branch on the answer**. A driver can offer float render
  targets without float blending, and blending to one anyway produces zeros with no error. Every
  page here does this check and says which path ran.

### 3 · Flat 2D addressing

Textures have a size limit (commonly 16,384). Any array longer than that must be addressed as a
grid:

```glsl
ivec2 at(int i){ return ivec2(i % uTexW, i / uTexW); }
```

This sounds trivial and is the single most common source of silent corruption in this codebase — a
too-tall texture does **not** raise an error, it samples garbage. Use `at()` everywhere.

### 4 · Differential testing is the method, not a nicety

Every page embeds f64 answers computed by the engine, then checks itself. Every kernel ships with a
NumPy reference. This is not ceremony — three separate bugs in this arc presented as plausible
output and were caught only by a differential test:

- A wrong hash gave *plausible-looking* retrieval hits (every result in the corpus shares a prefix,
  so random ranking still looks like ranking). The **margin** was the tell: 0.0215 broken vs 0.1277
  correct.
- A texture taller than the device limit gave 100% wrong pixels with no error.
- A benchmark harness's own reference was wrong while the shader was right.

**Write the reference first. Compare on every run. Report the margin, not just the answer.**

---

## What you can build

### Games and interactive simulation — yes, this is the strong case

The physics kernels are the ones that win on hardware.

- **Cloth, rope, soft bodies:** `pbd_scatter_vs` + the apply pass. Position-based dynamics, Jacobi
  style — 8,064 constraints × 40 iterations in **1.51 ms**, constraint residual falling
  0.4717 → 0.0311 identically to the CPU reference. Jacobi, not Gauss-Seidel: all constraints see
  the same input state, which is what makes it parallel. Compensate with more iterations.
- **Fluids, heat, smoke, reaction-diffusion:** `diffuse` is a five-point Laplacian with insulated
  edges, 512×512 × 40 steps in **2.20 ms**. Error does not compound: 2.35e-07 after a hundred
  steps, flat across grid sizes. Note the boundary detail — a boundary texel's missing neighbour is
  *itself* (zero flux); clamping the fetch samples the interior twice and slowly leaks heat inward.
- **Particle systems with structure:** `hdrift_step` is attraction to a learned field minus batch
  self-repulsion, both passes on the GPU. The repulsion is what stops particles collapsing onto the
  data they were fitted to.
- **Procedural texturing:** write the kernel in Python, emit GLSL with `mind.emit_kernel(src,
  'glsl')`, run it as a fragment shader. Six of eight menu entries (stripes, rings, marble, fbm,
  wood, checker) emit and execute within f32 of the Python original. The emitter **refuses** what it
  cannot translate exactly — a variable loop bound, or `fmod`, whose sign convention differs between
  C and GLSL. A refusal is a feature: it means no silently-wrong kernel ever ships.

### Exact answers without a model — worth knowing about even though it is not a GPU win

`perfect_recall_candidates` is the exception to "don't put retrieval on the GPU", and only because
of what it is *for*. It runs the candidate half of the exact-containment index — tile probes culled
by an AND of query bits, then a per-doc Bloom test — and culls **5.5× harder** than a doc-filter-only
pass while never dropping a true candidate (verified: candidates ⊇ the exact answer, 25/25 in two
regimes).

**The verify half is deliberately not ported.** Exact sha256 membership is what buys zero false
positives, and moving a correctness guarantee onto a float substrate would trade the one exact thing
in the module for speed. It stays on the host and only ever sees candidates.

Why you care: when the AND-set is a singleton, **you have proved the answer and never need to call a
model.** That is worth more than any speedup on this list.

It has no hardware timing and, like the other retrieval kernels, would very likely lose to a
JavaScript implementation of the same Bloom-and-verify structure. **Correctness is the claim, and
the claim is the valuable part** — a singleton AND-set is a proof, and a proof means you can skip a
model call entirely. Port the structure to JavaScript; keep the shader as the test that says your
port is right.

### Apps — yes, and the browser pages are the template

The search page is a complete application in one file: an index, a scorer, a policy that answers /
returns a set / refuses, and a self-test. 232 KB, no server, no account, no network.

The architecture worth copying:
- **Ship one representation, derive the others.** The page ships a packed token stream and builds
  the scorer's index in JavaScript at load. Duplicate state that can drift is worse than the cycles
  to rebuild it.
- **Pack to the entropy.** 6,767 terms is 13 bits, not 32. The packing is 2.42× and lossless — with
  an assertion on the width, because a vocabulary that outgrew the width would silently *alias*
  terms rather than fail.
- **Carry your own reference.** The page verifies its fast path against its own slow path at load.

For corpora past ~100k tokens, shard: fit each shard with the corpus statistics and the shards are
**bit-identical** to a single index (verified 4 through 256 shards, and the reassembly verified
under Node against the Python reference at 5.8e-13 with top-1 identical 60/60). Without those
statistics the merge silently ranks by which shard a document landed in. A browser-only app fetches
shards on demand and never needs the whole index resident — that is how the same design carries a
corpus far larger than a tab.

Note what is GPU and what is not in that page: the **scoring** is a shader because the page exists
to prove the shader, but if you are building this for production in a browser, **write the scorer in
JavaScript** and keep GLSL for the visual and simulation work. See the substrate table above.

### 3D — partly, and here is the honest boundary

- **Image formation / deferred-style shading:** `raster_form` works and is exact, but it is a GEMV
  and NumPy beats it. Use it when the result **stays on the GPU** and feeds the next pass; the
  moment you read it back you have lost.
- **Mesh transforms in the vertex stage:** written, differentially tested, and **blocked** — the
  harness needs transform feedback, which produced nothing on the development stack. Transform
  feedback is core in WebGL2, so this most likely works in a browser and simply has not been run
  there. The shader and its test ship in `research/shader_retrieval/glsl_meshvs.py`; if you need it,
  run it and tell us.
- **Byte-exact rendering:** possible but **conditional**. `raster_program_pgm` now reports a
  `rounding_margin` — byte-exactness against another substrate holds only while that substrate's
  error is smaller than every pixel's distance to a `.5` rounding boundary. On a real scene the f32
  error *exceeded* that margin at three of four light counts and the images matched anyway, by luck.
  Check the margin; don't assume it.

### VR — not tested, and I will not pretend otherwise

Nothing here has been run under WebXR. The kernels are ordinary WebGL2 fragment and vertex shaders,
so there is no structural reason they would not work in an XR render loop, but *no structural reason
not to* is not evidence. The specific unknowns: per-eye render targets doubling the readback cost
(readback is already 46% of the diffusion kernel's time), and frame budget — at 90 Hz you have
11 ms total, and the diffusion kernel alone is 2.2 ms.

**If you are trying this:** keep everything on the GPU. The readback is what will kill you, not the
compute.

---

## Persistence: the index survives the tab

`mind.index_save(tokens, path)` writes a portable `lecore-index/1` bundle; `pages/idb_store.js`
reads the same bytes out of IndexedDB. One format for disk, browser and page.

* **The bundle stores the GENERATOR** -- the bit-packed token stream, the vocabulary as u32 hashes,
  and the global statistics. Postings, tf tables and document lengths are DERIVED on load in
  milliseconds. Storing them too would be duplicate state that can drift from the stream it
  describes.
* **It verifies itself.** Every load recomputes a sha256 and REFUSES a mismatch, because a
  truncated write or an aborted transaction leaves a payload that parses cleanly and answers
  wrongly. The search page ignores a cache that fails and says so in its PASS table.
* **IndexedDB, not localStorage, and the reason is arithmetic.** An index is ~1.9 MB per million
  tokens packed; localStorage is synchronous, string-only and capped near 5 MB, so it runs out
  around two million tokens and blocks the main thread getting there. localStorage ships as a
  fallback with a hard cap that refuses rather than truncating.

Bake a corpus on a workstation, drag the file onto the search page, and it is cached for every
later visit with no rebuild.

## Factoring on the GPU: the resonator is PBD in a different costume

`project_onto_constraints`'s own docstring says it -- the SBC resonator, the PnP denoiser and a PBD
constraint sweep are ONE ENGINE, alternating projection. `research/shader_retrieval/glsl_resonator.py`
is that engine pointed at factoring a bound product: probe, score, reduce, project, four passes per
iteration regardless of factor count because all factors update from the same state (Jacobi, which
is what PBD taught).

**It works at three factors and fails completely at four** -- 20/20 at F=3, 0/20 at F=4, at D=512
*and* D=1024, so more dimensions did not help. That is the resonator's own capacity cliff, not a
shader limit. Past it, use `recursive_factor`, whose catalog entry is literally "past the
resonator's cliff".

Two things that are load-bearing and easy to get wrong: a **hard argmax does not resonate** (it is
omega=1 with no damping and limit-cycles on the first step -- use the soft projection
`sign(C^T(C x))`), and **restarts are the method, not a tuning knob** -- keep the run whose
RECONSTRUCTION matches, because a cycling resonator produces a confident wrong answer.

## Where the ceiling is

- **Readback dominates more than you expect.** Measured on the A4500: an empty shader reading back
  262,144 floats costs 1.02 ms. That is 46% of the diffusion kernel's total time — and diffusion
  still wins 134×. Design so results stay on the device.
- **No compute shaders, no SSBOs.** WebGL2 has neither. Everything here is fragment and vertex
  shaders with `texelFetch`. That constraint is why the scatter-add trick exists.
- **WebGPU is a rewrite for three of the ten kernels.** The six gather-shaped ones port
  mechanically. The three scatter kernels rely on point primitives plus blending; WGSL's native
  answer is a compute shader with atomics into a storage buffer — likely *faster*, but a different
  implementation needing its own differential test. Also: WebGPU has no synchronous readback, so
  every "render, read, compare" harness becomes async.
- **The install boundary is real.** Of 17 VM units, 10 install as arithmetic and 7 do not (control
  flow and storage). Fragment shaders are arithmetic. Same boundary — which is a useful predictor of
  what will port before you try.

---

## Reference

```python
mind.glsl_kernels()          # the ten verified kernels by name
mind.glsl_kernel('diffuse')  # {'does': ..., 'verified': ..., 'source': ...}
```

The `verified` field is not documentation — it is the measurement that earned the kernel its place,
including its kept negative. Read it before you port anything.

| where | what |
|---|---|
| `pages/` | the five self-verifying demos + `encoder_probe.html` |
| `research/shader_retrieval/glsl_*.py` | one kernel each, with its differential test |
| `research/shader_retrieval/bench_gpu.py` | the hardware benchmark; refuses to time a software rasteriser |
| `run_bench.bat` / `build_pages.bat` | throwaway venv, no system Python touched |
| `holographic/io_and_interop/holographic_glslkernels.py` | the shader sources, inside the engine |
| `docs/NOTES_concepts.md` | every measurement above, with its kept negatives |

**Two standing rules if you contribute a kernel:** it ships with a NumPy reference and the number
that verified it, and it ships with its boundary stated. A kernel that is exact in one regime and
not another is a trap unless the limit travels with the code.

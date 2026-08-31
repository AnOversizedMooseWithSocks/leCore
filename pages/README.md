# pages/ — the browser demos, checked in

**Open `lecore_search_webgl2.html` in Chrome.** Double-click it, or drag it onto a Chrome window.
No server is needed: zero network calls, no external scripts, all data embedded.

These are generated files that are **deliberately checked in**, like `CAPABILITIES.md` and
`REFERENCE.md`. They were briefly gitignored as "build artefacts", which classified them by how
they are produced rather than what they are for — they are the deliverable of the WebGL2 work, and
nobody should have to run Python to look at one.

| page | what it exercises |
|---|---|
| `volume_three.html` | **a volumetric cloud with no volume.** three.js uploads a 128³ texture (2.10 MB); leCore evaluates a closed-form sum of 20 plane waves in the raymarch loop — 0 bytes, no grid, no voxels to zoom into. |
| `zeroasset_three.html` | **the one to show someone.** 24,000 towers, every position/height/rotation/colour a deterministic function of the district's NAME. A glTF with `EXT_mesh_gpu_instancing` would carry 1.25 MB of instance data; this ships **62 bytes**. Type a name that has never existed. |
| `cloth_three.html` | **the one to show someone.** three.js renders (PBR material, lights, shadows, ACES tone mapping); leCore solves. 9,202 constraints stepped on the GPU by the catalogued `pbd_scatter_vs` kernel, pulled from the engine at build time. Self-contained: three.js r185 is inlined, zero network requests. |
| `cloth_demo.html` | **physics you can drag.** 4,096 particles, 8,064 distance constraints, position-based dynamics solved entirely on the GPU. Checks its own solver against a CPU reference on load and reports whether the constraint residual actually falls. |
| `field_demo.html` | **the one to show someone.** Type words; each is hashed to a frequency and phase, and the field is the sum of those plane waves. 16,384 particles drift along its gradient, repel off their own density, and leave a diffusing trail. Four passes per frame, zero readbacks. |
| `lecore_search_webgl2.html` | the canonical one: BM25 + exact containment in one shader pass, answer/set/abstain policy, proximity reranking, **both** scorers behind an `EXT_float_blend` gate |
| `lecore_webgl2_vsa.html` | VSA algebra — bind / unbind / cleanup, phasor atoms, 6 shaders |
| `lecore_webgl2_full.html` | real corpus, 3-tier beam walk, all decisions on the GPU |
| `lecore_webgl2_typed.html` | typed queries via Rademacher encoding, beam recall |
| `lecore_webgl2.html` | the smallest one: bind / score / argmax on synthetic atoms |

For how to BUILD with these — the kernels, the shader shapes, and what the GPU is
measurably good at — see [`docs/GLSL_GUIDE.md`](../docs/GLSL_GUIDE.md).

## What to look at

Each page runs a PASS table on load. On the search page the rows that matter are
`EXT_float_blend`, `SCORER IN USE` (scatter or full scan), `scatter == full scan` (the page
checking its own fast path against its own slow path), and the three agreement rows — verdict mode
**60/60**, ambiguity **60/60** as exact integers, reranked answer **45/45** against
`mind.retrieval_verdict`.

**If a row is false, that is the useful result.** Press F12, read the Console, and report the red
text plus which rows failed. Every shader here compiles AND links under a real GLSL ES 3.20
compiler, but that validates syntax and linking — it says nothing about how Chrome's ANGLE layer
handles the texture formats, the blending path, or `gl_PointSize`.

## Rebuilding

Only needed if you changed a shader, the corpus, or the retrieval policy:

```
build_pages.bat          rebuild all five, then open the search page
build_pages.bat --clean  rebuild the venv first
build_pages.sh           the MINGW64 / Linux twin
```

It creates a throwaway venv at `.venv-bench/` (shared with `run_bench.bat`), installs numpy into
it, and writes the pages here. Nothing touches your system Python.

# Render path audit — where the pipeline left the substrate, and how it was found

*Sweep 149, 2026-09-05. Written because Moose said "we went off path somewhere and got lost," and he was right.
This doc is the map back, with the numbers. Read it before touching anything in `holographic/rendering/`.*

## The thesis this engine renders under

leCore renders by **projecting from a superposition**. The scene — vertices, faces, materials, lights, camera,
all just numbers — is carried as hypervectors, and an image is a *readout* of that field, the same operation that
recalls a record or a sentence. A 3D render is the proving ground: if the substrate can turn a scene into a
coherent 2D image, it can turn context into an answer. The optical correspondence note says it plainly: `bind` is
a lens, `bundle` is multiplexing, bake-and-query is exposing a hologram once and projecting it at any size.
**Staying in the Fourier domain is the whole win.** Every round-trip out to a spatial grid and back is pulling
the light out of focus and paying, in FFTs, for what a lens does passively.

## The lineage that already exists (all still wired)

| Faculty | What it is | Measured |
|---|---|---|
| `radiance_transfer` (PRT) | "collapse the wave function" — visibility integral once, shading = dot product | 57× per relight |
| `render_dispatch` / `bake_scene` / `render_baked` | collapse on diffuse, trace on mirror; bake once, every frame is a relight | 15×; first frame is a relight |
| `holographic_fog_volume` (volint) | density field as ONE vector; ray integral in closed form; empty space *known* | 90× over marching, exact |
| `holographic_radiance` | radiance as FPE bundles — titled **RENDER = QUERY** | free-viewpoint query |
| `render_adaptive` / `plan_render` | one call that picks bake/collapse/trace from measured break-evens, with reasons | — |
| semantic scene | the scene as ONE bundled vector | 12/12 attributes decode |

## Where it diverged: sweep 138

"Caustics + dispersion." From here on: Monte-Carlo path tracing with NEE, hero-wavelength spectral sampling,
manifold NEE, photon-splat caustics into a histogram, progressive MC buckets, a scanned mesh baked to a voxel
grid and sphere-traced. **Every one is how Cycles does it. Not one hypervector.** Renders took 30–100 minutes
and were hard to make look good — which is the tell that the pipeline had stopped using the substrate.

## Why: the holographic lineage had zero catalog cards

`render_baked`, `bake_scene`, `radiance_transfer`, `render_dispatch`, `plan_render`, `holographic_fog_volume`,
`HolographicRadianceField`, `TiledRadianceField`, `HolographicField` — wired as verbs, **no cards**. Every
conventional path had cards with generous aliases. Measured on the memory-booted engine:

- "precomputed radiance transfer" → *Rendering (path trace)* first
- "relight a scene with a dot product instead of tracing" → nothing of the kind
- "render an image by projecting from a superposition" → image editing, MCP doors

The semantic surface steered every render question toward Monte Carlo. By the engine's own rule — *a capability
that `find_capability` can't surface does not exist* — the holographic path did not exist, and eleven sweeps
walked past it. **Fixed: five cards restored, two added; 9/9 phrasings now route to the holographic path.**

Trap found on the way: `capdoc.generate_json` drops every card whose *name* starts with `holographic_`, and
`mind.suggest` ranks over that curated set. Name cards descriptively.

## What was built: the caustic as one hypervector

A caustic is refraction (nonlinear, cheap, one pass) glued to **accumulation** (a pure sum — a bundle). Every cost
of the conventional version is in the accumulation. So: refract once with a *per-ray* wavelength, bundle the
landings into an FPE field over (x, z, λ), and read the image out. RGB is **three unbinds** — the colour-matching
integral folds into the query because a multi-axis encode is a bind of per-axis encodes.

Measured: one vector reads consistently at any resolution (0.9997); agrees with the histogram 0.80 (dim 1024) →
0.95 (dim 16384); 5× smoother at equal rays; translate = bind to 1e-8. The separable outer-product read
(`img = Re(E_z · diag(G) · E_xᵀ)`) took a 256² RGB read from 223.5 s to **1.4 s** — identical to 1e-15.

## What it costs (loud)

- **Capacity-bounded.** A dim-d vector resolves ~√d cells per axis. The default kernel is set to that; sharper
  and agreement *falls*. Colour separation is capacity-bounded too: same-geometry reads at two wavelengths
  correlate at only ~0.75 (dim 2048) — the λ crosstalk floor — against a ~3% physical dispersion effect.
- **Tiling** (mirroring `TiledRadianceField`) cuts the build 11× and holds 0.87 against the true caustic at
  equal ray budget, but with landings/tile above the tile's dim the read is a *patchwork*. More tiles, not dim.
- **The histogram won on speed this round**: `spectral_caustics` on an analytic SDF is 2.3 s with clean rainbow
  arcs. The substrate's wins here are resolution independence, composability, one-pass spectrum — not raw speed.

## The result that answers the complaint

The **1-fold inverted Mandelbox as glass shows clear prismatic dispersion in 2.3 seconds** and renders as clean
crystal at 192 spp in under five minutes. Dispersion was never the hard part; the scanned soup was. `fold_fractal`
also needed a fix to be a solid at all (no bailout → estimate collapsed 16× per 4 iterations; all-positive → no
interior). Both opt-in, defaults byte-identical.

## The next rung — BUILT in sweep 150 (`holographic_glassbake`)

Measured: bake 1.6 s + relight 7 s per light on the 1-fold Mandelbox at 320×200, against ~5 min at 192 spp for the
Monte-Carlo frame — and a second light is another relight from the same bake, no tracing. Two bugs found on the
way (first-order TIR loss 65.5% → 3.0% once `_march_through` stopped re-exiting the face it reflected off) are in
the sweep-150 notes. The original plan follows.

### The plan as written before it was built

The 30–100 minute cost lives in the **view** render: MC-tracing a solid glass interior × bounces × wavelengths ×
spp. The holographic answer is already in the lineage — `bake_scene`-style: trace primary visibility and the
refracted exit **once** per pixel, store a per-pixel transfer over (light, wavelength) as the field, and make every
relight and every wavelength a dot-product read. PRT generalised from diffuse-SH to spectral-refractive transfer.
That is the build that attacks the actual number.

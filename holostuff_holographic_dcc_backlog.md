# holostuff — Holographic-Native Geometry & Appearance Backlog

*Grounded in a live-code audit (172 holographic_* modules), not memory. The question that opened
this: should holostuff handle NURBS, splines, procedural objects, fractals/greebles, vegetation,
terrain, and the texture/normal/displacement/bump/vertex/uv map stack natively? The answer that
organised the backlog: most of the geometry half is already shipped under other names, the
appearance half is not yet in the space at all, and the whole thing should be built as **holographic
/ VSA-native faculties** — callable and composable from VSA programs on the one bind/bundle/cleanup
kernel — not as python-side helpers bolted onto meshes.*

---

## The design rule this backlog enforces

**Do it in the space.** A holographic field is a fixed-length vector; an operation on it is bind /
bundle / cleanup — O(1) or O(edit), model-size-independent, composable, and exactly undoable by
linearity. A python loop over vertices or texels is O(N) and composes with nothing. So *wherever an
operation can be expressed as field algebra, the holographic version is the one to build*, and it is
exposed as a callable faculty a VSA program can compose with every other faculty.

**The corollary the appearance stack forces:** if the *mesh* lives in holographic space, everything
that sits on top of it — materials, textures, normal/displacement/bump/vertex maps — should live in
the same space, so that a textured, displaced, shaded object is **one composite hypervector**
(`bind(GEOMETRY, field) + bind(APPEARANCE, material)`), edited with the same FS-5 delta machinery,
transmitted as one structure, blended by bundling, and queried by the same VSA program that built
the geometry. A mesh in the space with its materials *out* of the space is a half-measure.

**The one honest boundary (kept on the record).** "Holographic is faster/better" is true *where the
operation is field-algebraic and the band-limit / noise-floor cost is acceptable*. It is NOT a
universal claim: FS-5 already taught us the holographic field is FFT-bound and is the
compact/transmittable/exact-undo representation, while the array/mesh path stays the fast path for
dense per-sample marching. Every item below states where the holographic form wins and where the
raster/array form is still the right tool — the engine's measurement discipline, applied to the
roadmap.

---

## §1 — What is already shipped (probe first; do not rebuild)

The de-dup audit. Most of the *geometry* wishlist is latent in the codebase:

| Wishlist item | Already shipped | Module |
|---|---|---|
| Smooth surfaces (sub-d) | **Loop subdivision** — refine 1→4 + Loop-weighted C2 low-pass | `meshsubdiv` (FWD-8) |
| Splines / smooth curves | **Chaikin corner-cutting = quadratic B-spline limit curve** | `subdivcurve` (ARCH-5) |
| Smooth surfaces (implicit) | **SDF fields** — surface as a field, marchable at any LOD | `field`, `fpefield`, `sparsefield` |
| Poly modelling | Full mesh-ops suite (curvature, geodesic, IK, seam, skin, Taubin, QEM, Euler operators, verbs) | `mesh*`, `eulerops` |
| Smooth↔poly conversion | mesh↔field both directions, O(surface area) | `meshbridge` |
| UV unwrapping | Isomap / MDS of mesh geodesic distances | `meshuv` (FWD-3) |
| Compact normals | **Octahedral encoding** (2-DOF on S², bounded round-trip) | `octnormal` (A2) |
| Fractal analysis | Box-counting dimension, Hurst self-affinity, **IFS compression** | `fractal` |
| Procedural composition | Recursive scenes-of-scenes, recipe / typed-structure layer | `scene*`, `recipe`, `recipeops`, `typed` |
| Point/Gaussian primitives | Splat fields, densify, prune, sharpen, export, archive | `splat*`, `jittersplat` |

So the **hybrid you wanted — smooth where smooth, poly where detailed — already exists** as the
`field ↔ mesh ↔ subdivision` triangle, all converting through `meshbridge`, all reducing to the same
primitives. The build work is not the hybrid; it is the pieces genuinely missing from it.

**Confirmed genuinely absent (the gaps this backlog fills):**

- No procedural **noise generator** (everything matching "noise" is *de*-noising / band-limiting).
- No **material / texture** representation in the space at all (this is the big one Moose flagged).
- No **displacement / bump** operator (offset surface along normal by a field).
- No **terrain / heightfield** primitive.
- No **L-system / grammar** for vegetation or greebles (the *substrate* — recipes + scenegraph
  recursion — exists; the grammar on top does not).
- `Mesh` carries `.normals` and `.uvs` but **no general per-vertex/per-texel attribute channel**.

---

## §2 — The organizing thesis: the appearance stack is a holographic record

This is the part that is not yet in the space, and it is the purest VSA fit on the whole list.

**A texture is a function over the UV chart.** That is exactly what the FPE `VectorFunctionEncoder`
encodes — a function as a hypervector. Sampling a texture = querying the field (a cosine). Tiling /
offset = a `bind` (FPE shift is a translation in UV). Blending two textures = a `bundle`. No texel
loop; the texture *is* a vector and the operations *are* the kernel.

**A material is a role-filler record.** A PBR material is a set of named channels — albedo, metallic,
roughness, normal, displacement, emission, AO, opacity — each a constant or a texture. In the space
that is one bundle of role-bound channel fields:

```
material = bind(ALBEDO, albedo_field) + bind(ROUGHNESS, rough_field)
         + bind(NORMAL, normal_field) + bind(HEIGHT, disp_field) + ...
```

— literally the HRR record Plate's representation was built for. Fetch a channel by `unbind(material,
ROLE)`; blend two materials by `a*material1 + b*material2` (every channel mixes by linearity); UV-
transform every channel at once by binding the whole record with a shift. A VSA program builds,
queries, layers, and morphs materials with the same bind/bundle/cleanup it uses for geometry.

**Geometry and appearance compose into one object:**

```
object = bind(GEOMETRY, geometry_field) + bind(APPEARANCE, material_record)
```

One hypervector carries shape *and* look. Edit either side with an FS-5 delta (O(edit), exact undo).
Transmit the object as one structure. Morph two objects — shape and material together — by bundling.
This is Moose's thesis fully realised: the mesh is in the space, so everything on top is too.

**The honest boundary:** an FPE field is band-limited (smooth), so a material built this way is ideal
for *procedural and smooth* channels — gradients, noise-driven detail, PBR scalar maps — and is a
lossy/smoothed representation for *sharp photographic* textures (hard mask edges, text). For those,
keep the raster and bind a *reference* into the record rather than the field itself; the record
structure is unchanged, only the leaf differs. State this in every measurement, as with `fpefield`.

---

## §3 — NURBS: out of scope (recorded, since it was asked)

Current practice splits cleanly: spline/NURBS surfaces are the CAD standard (exact parametric,
trimming, manufacturing documentation), while subdivision surfaces are the animation/games/film
standard (topologically free), and implicit/SDF has taken the Boolean/performance/AM end (union =
min, intersection = max, guaranteed). Almost every representation converts *into* SDF easily; the way
*back* toward parametric CAD is by fitting sub-d to the implicit model. holostuff is a sculpting /
content engine, not a CAD documentation tool — its smooth need is fully covered by sub-d (shipped) +
implicit (shipped). NURBS would be a large new parametric kernel serving a use case the engine does
not have. **Scrap it**, exactly as the instinct said. (Kept as an explicit negative so it is not
re-proposed.)

---

## §4 — The backlog (tiered by value-over-effort, each with a measurement bar)

### Tier 1 — the two keystones

**G1. `noise_field` — band-limited holographic procedural noise.** *(Seats: Stam — spectral/FFT
noise; Berry — fBm / band-limiting; Quílez — noise as the procedural-SDF primitive.)*
The keystone field *generator*: a noise field as a hypervector / FPE function, evaluable anywhere at
O(1) per query, not a python pixel loop. Value/gradient noise via random band-limited phase; **fBm is
an octave bundle** — a sum of band-scaled copies, which *is* bundling, so the multi-octave loop
collapses to one superposition. Composable: shift = bind (translate), combine = bundle, warp = bind a
noise field by another. Unlocks terrain, displacement, greebles, vegetation scatter, texture detail —
everything downstream leans on it.
*Holographic win:* octaves = one bundle (no per-octave loop); evaluate-anywhere O(1); composes
natively with the SDF / material stack. *Bar:* reproduce a target power spectrum and a target
box-counting fractal dimension (we already ship the measurer); fBm octave count ↔ Hurst exponent (we
ship Hurst); deterministic under seed. *Kept scope:* band-limited/smooth by construction — sharp
discontinuous noise is not its regime (and for terrain that band-limit is a feature: anti-aliased by
construction).

**G2. The holographic material/texture stack (`texture_field`, `material`, `sample_material`).**
*(Seats: Plate — the role-filler record; Pharr — PBR channels / texture sampling; Drettakis — splats
carry per-primitive material the same way.)*
§2 made concrete. `texture_field(uv_points, values)` → a function over the UV chart (FPE).
`material(**channel_fields)` → the role-bound bundle record. `sample_material(material, mesh, uv)` →
evaluate all channels at mesh UV coordinates (uses `meshuv`'s chart) and return the shading inputs.
`blend(m1, m2, t)` = bundle; `transform_uv(material, shift)` = bind. The whole thing callable and
composable from VSA programs, and composing with geometry into one `object` hypervector.
*Holographic win:* material lives in the mesh's space — one composite structure, edited/transmitted/
blended/undone with the existing field machinery; layering = bundling; channel fetch = unbind.
*Bar:* `unbind(material, ROLE)` recovers each channel at cosine ≈ 1; blend of two materials recovers
a weighted mix per channel; a mesh+material round-trips as one structure. *Kept scope:* the
band-limit boundary from §2 — sharp textures stay raster behind a bound reference.

### Tier 2 — rides on Tier 1 (small, high payoff)

**G3. `displace` / `bump` — offset along the normal by a field.** *(Seats: Pharr — displacement/bump
maps; Quílez — SDF displacement = add to the distance.)*
On an **SDF**, displacement is field *addition* — `field + amount * scalar_field` — a bundle, O(1),
and on the FS-5 representation it is an exact, undoable delta. On a **mesh**, `vertex + amount *
normal` using `octnormal`'s normals. **Bump** perturbs the normal field (octnormal-encoded) without
moving geometry. Small once G1 exists.
*Holographic win:* SDF displacement is a one-bundle field add; exact undo via FS-5 `make_delta`.
*Bar:* the marched surface moves by exactly the scalar field (to the marcher's tolerance); undo is
bit-exact; bump changes shading normals with geometry unchanged.

**G4. `terrain_field` — a heightfield/SDF from holographic noise.** *(Seats: Stam — spectral terrain;
Berry — multifractal terrain.)*
Mostly a *composition* of G1 + G3: an fBm noise field over a 2-D domain, exposed either as a displaced
plane (G3) or lifted to an SDF (height → signed distance), marchable at any LOD through `meshbridge`,
and textured by a G2 material. *Holographic win:* terrain is a field, so LOD is just a re-march at the
needed resolution; composes with materials and displacement in-space. *Bar:* the terrain's
box-counting dimension matches the fBm parameters; LOD chain via `meshbridge` is exact near-surface.
*Kept scope:* hydraulic/thermal erosion is an iterative simulation, not field algebra — a separate,
later item, not folded in here.

### Tier 3 — the genuinely novel VSA contribution

**G5. `grammar` — L-systems / greebles as recipes.** *(Seats: Plate — HRR productions; Quílez —
procedural detail.)*
The substrate exists (`recipe` / `recipeops` / `typed` + `scenegraph` recursion); the grammar on top
does not. An **L-system production is a rewrite rule, and a rewrite rule is a recipe**; the L-system
state is a hypervector; expansion is recursive bind/bundle; the result is a scenegraph of bound
sub-scenes — a tree is a recursive bundle, a greeble field is recursive panel-subdivision +
extrusion (mesh verbs) under a recipe. This is the one item that is *new capability*, not reuse, and
it is a pure HRR fit.
*Holographic win:* productions are savable recipe seeds; the grammar composes with bind/bundle and is
callable from VSA programs; a whole plant is one composable structure. *Bar:* a generated branching
structure's box-counting dimension matches its production parameters; expansion is deterministic; the
result composes as a scenegraph. *Kept scope:* recursive composition, not a physically-accurate growth
or competition simulation.

### Tier 4 — plumbing (do when something needs it)

**G6. General per-vertex / per-texel attribute channel.** A data channel on `Mesh` beyond
`.normals` / `.uvs` — color, scalar, or vector attributes — carried as a field sampled at vertices (or
a `bind(vertex, value)` bundle), so attributes are resolution-independent (sample at any subdivision
level) and compose/transmit with the field machinery. *Bar:* attributes sampled at two subdivision
levels interpolate consistently. Low urgency; G2's channels cover most appearance needs already.

---

## §5 — Sequencing

```
G1 noise ──┬──► G3 displace ──► G4 terrain
           │
G2 material stack  (parallel keystone — FPE + the role record; Moose's emphasis)
           │
           └──► G5 grammar (vegetation / greebles)   G6 attributes (when needed)
```

1. **G1 (noise)** and **G2 (material stack)** are the two Tier-1 keystones — G1 unlocks geometry
   detail, G2 unlocks appearance, and they are independent, so they can proceed in parallel.
2. **G3 (displace)** rides G1 and is small; **G4 (terrain)** is mostly G1 + G3 composed.
3. **G5 (grammar)** is the novel build — schedule it once the field generators exist to feed it.
4. **G6 (attributes)** is plumbing — when a feature needs it.

Each lands under the standard close-out ritual (module + `_selftest`, `test_*`, faculty wired into
`UnifiedMind` default-off, integration test, README counts, NOTES, tour line, clean zip + verify),
backward-compatible, with the holographic-vs-raster boundary measured and the negatives kept loud.

---

## The honest bottom line

The geometry half of the wishlist was mostly already in the box — the hybrid is the
`field ↔ mesh ↔ subdivision` triangle, NURBS is a CAD paradigm to skip, and the real geometry gap is
a single keystone (procedural noise) plus the small operators that ride it (displacement, terrain) and
one genuinely new VSA capability (the recipe-grammar). The *appearance* half was not in the space at
all, and it is the purest fit of everything here: a texture is an FPE function over UV, a material is
an HRR role-filler record, and a textured object is one composite hypervector that edits, transmits,
blends, and undoes with the field machinery the engine already owns. Build the appearance stack and
the geometry generators *in the space*, expose them as callable composable faculties, measure where
the holographic form wins and where the raster form is still right — and the mesh stops being a thing
in holographic space with its materials outside it.

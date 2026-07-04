# holostuff — 3D Output Quality & the Modulate/Demodulate Primitive

*One backlog for two threads that share the rendering/output pipeline: **(A)** the quality of the 3D artifacts
we export — Gaussian splats and GLB mesh, including fluid/smoke; and **(B)** a cross-cutting primitive —
**modulate/demodulate** (which is `bind`/`unbind`, spent as **bake-and-query**) — that already lives five times
in the codebase and, once elevated, improves denoising, upscaling, relighting, and compression everywhere. The
two threads meet at splats (relightable) and upscaling. Grounded in a probe of the LIVE code (latest repo, 333
modules). Items are tagged **PROMOTE** (already ships; consolidate/expose) or **BUILD** (new — usually reusing
shipped parts). The constitution rule shapes it: the modern SOTA here is mostly **learned**, so we take the
**classical ancestors**, several of which are already half-built. Kept negatives are loud; code sketches are
short and commented.*

---

## 1. What already ships (so we don't re-request it)

**Export & mesh:** GLB/glTF export (`gltf`), splat export (`splatexport`); **QEM decimation** (`meshqem` —
Garland & Heckbert, SIGGRAPH 1997, feature-preserving, and the quadric is an outer-product accumulation, i.e. a
**bundle**); a full mesh toolkit (`meshsmooth`, `meshsubdiv`, `meshcurvature`, `meshuv`, `meshseam`, guarded
collapse via `eulerops`); a marching-cubes-style **SDF→mesh** (`mesh`/`meshbridge`).

**Splats:** `splat.splat_fit` (matching-pursuit placement + scale selection), `splat.adaptive_fit`
(content-driven count to a noise floor), `splatdensify` (the 3DGS clone-vs-split density control, *measured*),
`photo3d` (per-pixel splats sized from local spacing), and — newly landed — **`splatprune`** (prune negligible
splats, **merge** redundant ones, build a **LOD chain** for a quality budget). Plus `sampling` (Poisson-disk /
blue-noise), `lowdiscrepancy` (jitter), `octree`, `sparsefield` (narrow band), and `steering` (**anisotropic
kernels** — a reuse for surfels below).

**The modulate/demodulate move, already implemented five times** (see Part 4): `matcompile` (fold constant
channels), `matbake` (bake position-dependent channels, look them up), `viewlut` (bake the view-dependent part),
`prt` (bake the light transport), `radiance` (render = query). Each is "split a signal by how it varies, bake the
slow part, recombine" — which is `demodulate` → bake → `remodulate`.

---

## 2. Gaussian splats — the right amount, at the correct size and location (and relightable)

**Honest scope first.** holostuff's splats come from **geometry** (an SDF / scene / depth), **not** image-fitting
optimisation (autodiff is banned). So we can't do photometric adaptive density control the way optimised 3DGS
does — but we *can* place, size, and prune splats excellently from geometry, which is the right scope.

### SP1 (BUILD). Surface-aligned (anisotropic) splats — surfels *(the biggest quality win)*
Today's splats are **isotropic** (`splat.py`). A round blob is the wrong shape for a thin surface — it
over-covers and needs many splats. Collapse it into an **oriented planar disk** aligned to the surface (a
*surfel*), sized by two in-plane axes from the local surface:
```python
# SURFEL SPLAT: an oriented DISK on the surface, not a fat isotropic blob.
# Orientation is the SDF normal; the two in-plane axes are sized to the local surface --
# a big disk on flat regions, a small one where curvature is high.
def surfel_at(point, sdf):
    n      = sdf.normal(point)                 # surface normal (already available from the SDF gradient)
    t1, t2 = tangent_frame(n)                  # two axes spanning the tangent plane
    r      = local_spacing(point)              # nearest-neighbour spacing -> base disk size
    k1, k2 = principal_curvatures(point, sdf)  # curvature along t1, t2 (from meshcurvature)
    s1 = r / (1.0 + abs(k1) * r)               # flat -> large; sharp -> small
    s2 = r / (1.0 + abs(k2) * r)
    return Splat(center=point, axes=(t1, t2), scales=(s1, s2), normal=n)   # an oriented disk, no volume
```
*Real basis:* surfels (Pfister et al., SIGGRAPH 2000), EWA surface splatting (Zwicker et al., SIGGRAPH 2001), 2D
Gaussian Splatting (Huang et al., **SIGGRAPH 2024**), and the flat-Gaussian / Gaussian-Surfel line (2024).
*Reuses:* the SDF normal, `steering` (anisotropic kernels), `meshcurvature`. *Result:* far better surface fit,
**fewer splats** for the same coverage, cleaner mesh extraction. *Kept negative (loud):* oriented disks are
**less flexible in thin/complex regions** (2DGS notes this) — keep the isotropic splat as a fallback. *Effort:
medium.*

### SP2 (PROMOTE). The right amount — count control + prune/merge
Largely **shipped**: `adaptive_fit` (count to a noise floor) + `splatdensify` (clone-vs-split) + **`splatprune`**
(prune negligible, merge redundant, LOD chain). The remaining work is *promotion*: wire this 3D-count pipeline
onto the from-geometry splats end to end (fit → densify where under-covered → prune/merge to a budget), and
expose the quality-budget knob. *Reuses:* `adaptive_fit`, `splatdensify`, `splatprune`. *Effort: low–medium.*
*Kept negative:* prune/merge thresholds trade count against fidelity — expose them, measure against no-prune.

### SP3 (BUILD). The right size across scales — a Mip-style min-size clamp (anti-aliasing)
Splats correct up close **alias and shimmer** when zoomed out, and bloat when zoomed in. Clamp each splat's
**minimum screen-space size to the local sampling rate** — a low-pass that says "no splat may be smaller than a
pixel footprint at its distance," computed from the camera and the splat's depth. *Real basis:* Mip-Splatting (Yu
et al. 2024) — a 3D smoothing filter + a 2D Mip filter to bound the max sampling frequency. *Effort: low–medium.*
*Kept negative:* the clamp slightly over-smooths the finest detail — the honest trade for alias-free zoom.

### SP4 (PROMOTE). The right location — surface projection + blue-noise distribution
The parts ship: **project each splat onto the SDF zero level set** (a Newton step along the gradient — already
used) so it sits *on* the surface, and **distribute by Poisson-disk / farthest-point** via `sampling` so coverage
is even (no clumps, no gaps). This is mostly wiring those onto the splat placement. *Reuses:* the SDF projection,
`sampling`, `lowdiscrepancy`. *Effort: low.* *Kept negative:* even distribution is right for *uniform* detail —
pair it with SP1's curvature sizing so detailed regions still get smaller, denser splats.

### SP5 (BUILD). Relightable splats *(the splat↔modulation meeting point; was M8)*
A splat carries a colour, i.e. an **albedo**. **Demodulate** its view-dependent shading from the base colour so a
splat **scene can be relit** — and since a splat scene is a `bundle`, the demodulated albedo is one more role in
the record. *Reuses:* the M1 primitive from Part 4 + `splat`. *Kept negative:* the engine's splats are
geometry-derived, not photometrically fit, so the view-dependent model is limited. *Effort: medium.*

---

## 3. GLB mesh — topology optimization, decimation, and fluid/smoke

### GL1 (PROMOTE). QEM decimation in the GLB export — target count / error budget
`meshqem` is built and feature-preserving; **expose it in the glTF export** as a decimation control — "decimate to
N triangles" or "to error budget ε," sharp features preserved. Promotion, not construction. *Reuses:* `meshqem`.
*Effort: low.*

### GL2 (BUILD, reuses the shipped quadric). Dual Contouring — feature-preserving SDF → mesh
Marching cubes **rounds off sharp edges** and yields poor-valence triangles. **Dual Contouring** (Ju et al.,
**SIGGRAPH 2002**) places one dual vertex *inside* each cell by minimising the **quadratic error function** — the
*same* QEF `meshqem` already implements — using the SDF's edge crossings and normals, so **sharp edges and corners
survive**, adapting triangle count via `octree`. So DC is largely "point the shipped quadric at isosurface vertex
placement." *Reuses:* the `meshqem` quadric, the SDF normal, `octree`. *Kept negative (loud):* standard DC can
yield **non-manifold** meshes on multi-sheet cells — use **Manifold Dual Contouring** (Schaefer et al. 2007); and
MC-plus-remeshing (GL3) sometimes beats DC — offer both. *Effort: medium.*

### GL3 (BUILD, reuses shipped mesh ops). Isotropic remeshing — triangle *quality*
Marching-cubes output is full of valence-4 vertices that behave badly under smoothing, decimation, and UV.
**Isotropic remeshing** (Botsch & Kobbelt 2004): split long edges, collapse short edges, **flip edges toward
valence-6 / Delaunay**, then **tangential relaxation** → uniform, well-shaped triangles. The engine has collapse
(`eulerops`), curvature, and smoothing already; the new bits are **edge flip + valence balancing**. Optionally
curvature/local-feature-size adaptive via `meshcurvature`. *Reuses:* `eulerops` collapse, `meshsmooth`,
`meshcurvature`. *Effort: medium.* *Kept negative:* remeshing moves vertices — protect feature edges/creases (tag
them, as QEM already does).

### GL4 (BUILD). Fluid / liquid → mesh (level-set isosurface, narrow-band)
The standard, borrowable pipeline: build a **level-set scalar field**, extract its zero-isosurface as a mesh. For
a **grid** sim the level-set/density *is* the field; for **particles**, use Zhu & Bridson's distance-to-smoothed-
cloud:
```python
# LIQUID SURFACE from particles (Zhu & Bridson, SIGGRAPH 2005): the level set is a distance to a
# SMOOTHED particle cloud; marching-cubes its zero-crossing to get the surface mesh.
def liquid_phi(x, particles, radii):
    w    = kernel_weights(x, particles)        # weights of nearby particles
    xbar = weighted_avg(particles, w)          # smoothed surface position
    rbar = weighted_avg(radii, w)              # smoothed radius
    return norm(x - xbar) - rbar               # signed distance; the zero-set IS the liquid surface
# build phi ONLY in a narrow band around the particles (sparsefield), then extract + clean (GL2/GL3/GL1).
```
Build the field in a **narrow band** (`sparsefield`) so cost scales with **surface area, not volume**, then
extract via GL2/marching-cubes and clean via GL3/GL1. *Real basis:* Zhu & Bridson (SIGGRAPH 2005); narrow-band MC
(Akinci et al. 2012). *Reuses:* `sparsefield`, the SDF→mesh, the mesh tools. *Effort: medium.* *Kept negative:*
MC-on-a-level-set gives poor-valence triangles (GL3 fixes them); thin sheets/droplets under-resolve unless the
band grid is fine.

### GL5 (BUILD + honest boundary). Smoke → GLB
Smoke is a **density volume**; GLB is a **surface** format — a *lossy* mapping. Three options: (a) **marching
cubes on a density isovalue** → a shell mesh — solid and standard, but a hard shell loses the soft volumetric
look; (b) a **set of billboards / splats** sampling the density — cheaper and more smoke-like, but not a solid;
(c) **bake to a 3D texture on a box** — glTF's volume extensions (`KHR_materials_volume`) are for absorption on a
mesh, **not** true participating media, so support is limited. Recommend (a) for a solid deliverable and (b) for a
lighter look, and **state plainly that neither is true volumetric smoke.** *Reuses:* the density field, MC, the
splat pipeline. *Effort: medium.*

### The big GLB animation boundary (kept loud — read before promising animated fluid)
A fluid/smoke surface **changes topology every frame** — marching cubes yields a different vertex count and
connectivity each step. glTF animation (**morph targets, skinning**) requires **fixed topology**. So a
topology-changing fluid mesh **cannot** be a standard glTF animated mesh. Honest options: (a) a **sequence of
separate meshes** with visibility keyframes (works, but heavy — N full meshes); (b) a **fixed-topology proxy** where
topology is stable (a fixed grid deformed by the field); (c) a **point/splat** animation. This is a **format
limitation**, not something we can engineer away — say so up front.

---

## 4. The modulate / demodulate primitive (bind · unbind · bake-and-query)

**The move, in one line.** For a diffuse surface the rendered pixel is a **product** — `radiance = albedo ×
irradiance` (a crisp, structured factor times a smooth, expensive one); the general case is a **convolution** of
lighting with the reflection kernel (Ramamoorthi & Hanrahan, SIGGRAPH 2001), and **convolution is `bind`**. So
**modulate = `bind`**, **demodulate = `unbind`**, and the optimal way to spend it is **bake-and-query** —
precompute the slow factor, multiply the fast factor at query (exactly PRT: Sloan, Kautz & Snyder, SIGGRAPH 2002).

**The audit — already here, five times, unshared.** The probe found the bake-and-query separation implemented
per-domain, each classifying a signal *by how its factors vary* and baking the slow ones: `matcompile` (fold
**constant** channels), `matbake` (bake **position-dependent** channels, trilinear lookup, reuses `sdfbake`),
`viewlut` (bake the **view-dependent** part into a (position, view) LUT), `prt` (bake the **light-transport**
integral, relight by a dot product), `radiance` (render = query of a radiance field). The pattern is identical
each time — **but there is no shared primitive**, and two sites that *should* demodulate don't: `svgf` uses albedo
only as an edge-stop **guide**, and `superres` uses the G-buffer only as an upsampling **guide**. So: name the move
once, make the five costumes delegate, add it where it's missing.

### M1 (BUILD). `demodulate` / `remodulate` — the primitive, recognised as `unbind` / `bind`
```python
# Split a signal into a CRISP carrier and a SMOOTH residual, and put it back.
#   modulate (combine) = bind ;  demodulate (split) = unbind
def demodulate(signal, carrier, eps=1e-4):
    return signal / (carrier + eps)          # smooth residual: denoises / upscales / compresses cleanly
def remodulate(residual, carrier):
    return residual * carrier                 # multiply the crisp detail back in, undamaged
# Round-trip is exact where the carrier is known: remodulate(demodulate(x, c), c) == x.
# In the FFT/HRR domain the same pair IS unbind/bind -- the engine already owns it.
```
*Kept negative:* the divide needs the `eps` guard and a **known, non-zero carrier**; where the carrier is *not*
known (a photo, where albedo is unknown), you must *recover* it first — that's M6, and it's ill-posed. *Effort:
low.*

### M2 (BUILD). `bake_and_query(factor, vary=...)` — the optimisation, generalised
```python
# Structure a computation as (slow factor, BAKED once) x (fast factor, applied at QUERY).
def bake_and_query(evaluate, bounds, vary="position", res=64):
    if vary == "constant":                    # matcompile's fold: evaluate once, broadcast
        return const_kernel(evaluate())
    if vary in ("position", "time"):          # matbake / iterate: sample to a grid, trilinear lookup (O(1))
        grid = sample_over(evaluate, bounds, res)
        return lambda p: trilinear(grid, p)
    if vary == "view":                        # viewlut: a (position, view) LUT
        return build_view_lut(evaluate, bounds, res)
# prt is this with vary='light-transport'; radiance is this applied to the whole render.
```
*Kept negative:* baking **trades memory for speed** and only pays for slow-varying factors — bake a high-frequency
one and you blur it or blow the grid. Measure the crossover (the discipline `matbake` already lives under).
*Effort: medium.*

### M3 (PROMOTE). Make the five existing bakes delegate to M1/M2 — the real generalisation win
The §5.1 consolidation: `matcompile` (constant), `matbake` (position), `viewlut` (view), `prt` (transfer),
`radiance` (all) are all `bake_and_query` with a different `vary`, and their recombine step is a `remodulate`.
Route them through M1/M2 (or minimally share the grid/LUT code and document the one pattern). Result: one place to
fix, one place to optimise, and every future bake inherits it. *Effort: medium (thin the call sites; keep
behaviour bit-identical — pin it).*

### M4 (BUILD). SVGF **demodulation** — the denoise refinement
`svgf` guides by albedo but filters the colour directly, so it can soften texture. Add the demodulation path:
`demodulate` the albedo out, denoise the **smooth irradiance** (which denoises cleanly with no texture to smear),
`remodulate`. Standard in real-time path-tracing denoisers; the missing quality step. *Reuses:* M1 + the shipped
à-trous bilateral. *Kept negative:* clean only for **diffuse**; keep the guide-only path for glossy. *Effort:
low–medium.*

### M5 (BUILD). Super-res **demodulation** — the upscale refinement
`superres` (which ships) does guided upsampling; add demodulation: upscale the **smooth irradiance**, `remodulate`
with the **crisp high-res albedo** (cheap to make — it needs no light transport). *Real basis:* radiance-
demodulation SR (Zhuang et al. 2023). *Reuses:* M1 + `superres`/`fsr`. *Kept negative:* diffuse-clean; glossy needs
the partial-separation formulation. *Effort: low–medium.*

### M6 (BUILD). Intrinsic decomposition — **recover** the carrier when it's unknown
Everything above assumes the carrier (albedo) is known; from a **photo** it isn't, so recover it: split the image
into reflectance (albedo, piecewise-constant) and shading (smooth) — Retinex (Land & McCann): large gradients are
reflectance edges, small smooth ones are shading; threshold the gradient field and **reintegrate with the
FFT/Poisson solver** the inverse-render work already uses. Then M1/M4/M5 apply to captured photos too. *Reuses:*
the gradient-domain solver; `inverserender`/`perception`. *Kept negative (loud):* single-image intrinsic
decomposition is **ill-posed** — a heuristic that fails on soft reflectance edges and hard cast shadows; the
learned SOTA is better and non-borrowable. *Effort: medium.*

### M7 (BUILD). Compression by separation
`demodulate`, then compress the **crisp** factor (structured) and the **smooth** factor by their *own* statistics —
the same reasoning as luma/chroma separation in image codecs, squarely Duda's ANS seat. *Reuses:* M1 +
`compress`/`ratedistortion`. *Kept negative:* the split only helps when the two factors really have different
statistics (a flat-lit textured surface, yes; pure noise, no). *Effort: medium.*

---

## 5. The honest boundaries (kept loud)

- **Geometry-derived splats, not photometric.** No image-fit optimisation (autodiff banned) — we place/size from
  geometry, which is the right scope, but won't match optimised 3DGS *appearance* fit to photos.
- **Anisotropic splats are less flexible in thin/complex regions** (2DGS's own caveat) — keep an isotropic fallback.
- **Dual Contouring can be non-manifold** — use the manifold variant; and MC-plus-remeshing sometimes wins, so
  offer both meshers.
- **Marching-cubes triangle quality is poor** — always remesh (GL3) or decimate (GL1) after.
- **Smoke → GLB is lossy** (surface format vs volume), and **topology-changing fluid animation does not fit glTF's
  fixed-topology animation** — the two format limits to be honest about, not to oversell.
- **Clean demodulation is a diffuse fact.** `radiance = albedo × irradiance` is a plain product only for diffuse
  surfaces; glossy/view-dependent doesn't separate by a division — keep the guide-only paths as the fallback.
- **The carrier must be known — recovering it is ill-posed** (M6). And **baking trades memory for speed**, paying
  only for slow-varying factors — the same crossover to measure throughout.

---

## 6. Sequencing (value over effort)

**Splats:** SP4 (surface projection + blue-noise — cheap promote, fixes location) → SP2 (count control + prune —
mostly shipped, wire end-to-end) → SP3 (Mip min-size clamp — anti-aliasing) → SP1 (surfels — the big quality
build) → SP5 (relightable — after M1 lands).

**Mesh / GLB:** GL1 (expose QEM decimation — it's built) → GL3 (remeshing — triangle quality) → GL2 (dual
contouring — reuses the QEM quadric) → GL4 (fluid → mesh — reuses `sparsefield`) → GL5 (smoke → GLB, with the
lossy/animation boundaries stated).

**Modulate/demodulate:** M1 (the primitive) → M3 (consolidate the five existing bakes onto it — the biggest
generalisation win) → M2 (the `bake_and_query` wrapper) → M4 / M5 (denoise + upscale demodulation — cheap quality
wins reusing shipped SVGF/superres) → M6 (intrinsic — the recover-albedo gap, which unlocks M1/M4/M5 on photos) →
M7 (compression). **SP5** rides on M1.

---

## 7. The recurring lesson

Both threads shortened the usual way. Most of the splat machinery (adaptive count, clone-vs-split, and now prune/
merge/LOD) and the mesh **decimation** (QEM) already ship — the splat gaps are specific (**surfels**, a **Mip
min-size clamp**), and the mesh gaps reuse shipped parts (**dual contouring** on the QEM quadric, **remeshing** on
the collapse/curvature ops, **narrow-band level-set surfacing** on `sparsefield`). And the modulate/demodulate
move was in the box **five times** — `matcompile`, `matbake`, `viewlut`, `prt`, `radiance` are all "split a signal
by how it varies, bake the slow part, recombine," which is `demodulate` → bake → `remodulate`, which is `unbind`
→ bake → `bind`. Naming it once and making everything delegate is the elevation you asked for; applying it to the
three places that lack it (denoise, upscale, recover-the-carrier) is the payoff. The two things to be honest about
are **format limits** (GLB is a surface format; volumetric smoke and topology-changing fluid animation don't map
cleanly) and the **diffuse assumption** behind clean demodulation — neither an engineering failure, both just the
truth stated plainly. As above, so below: a product of a crisp thing and a smooth thing is a `bind`; pulling them
apart is an `unbind`; baking the smooth one is how you make it cheap.

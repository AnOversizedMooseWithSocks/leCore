# leCore — Wet Cloth, Wicking & Bleed Backlog
### soak liquid into cloth · wick it through the fabric · drop it in a tank · watch the dye seep out and mix

*Can cloth soak up liquid and bleed dye into water? Yes — and after re-probing the current repo (331 modules),
most of it is now **reuse**, not new code: the cloth sim, the mesh-Laplacian operator wicking needs, and the
whole dye-in-water mixture model all already exist. The genuinely new work is two small, contained pieces — a
saturation/dye field that lives on the cloth, and the flux that hands dye from the cloth to the water. Written
reuse-first and wired into the main system, not siloed. Readable, commented sketches use the real API.*

---

## The idea, as three coupled fields

1. **The cloth** carries, per vertex, a **saturation** `s` (0 = bone dry, 1 = soaked) and a **dye
   concentration** `d` (the red). Dipping it in red liquid sets both high where it touches.
2. **Wicking** = the wet front spreading through the weave = **diffusion of `s` (and the dye it carries) along
   the cloth mesh**, pulled downward a little by gravity.
3. **Bleed + mix** = where the soaked cloth sits in the tank, a **flux** hands dye from the cloth into the water
   (high concentration → low), and the water then **advects + diffuses** it — the dye plumes out and tints the
   clear water.

The pleasant part: wicking (on the cloth) and mixing (in the water) are the *same operation* — a concentration
diffusing — just on two different substrates. The dye is one field, coupled at the boundary.

## What already exists (reuse — confirmed by probe)

- **The cloth sim is `SoftBody`.** `SoftBody.cloth(rows, cols, spacing, compliance)` (and `.cloth3d(...)`) build a
  PBD cloth sheet. `cloth.x` is the `(N, D)` per-vertex positions, and **`cloth.constraints` is the edge list**
  `(i, j, rest, compliance)` — so the fabric's connectivity (what wicks to what) is already there.
- **The wicking operator exists.** `spectral.graph_laplacian(adjacency)` returns `L = D − A` — exactly the
  operator that diffuses a scalar along a mesh. Build the adjacency from `cloth.constraints` and you have the
  cloth's Laplacian.
- **The dye-in-water model exists** (built since we last discussed it): `holographic_mixture` — `Component`
  (density + diffusivity), `Mixture` (concentration channels sharing one flow), and `matter_step(mix, vx, vy,
  dt)` (advect + diffuse + buoyancy + drift). A dye is just a `Mixture` channel; releasing dye into the water and
  letting it mix is `matter_step`.
- **Surface machinery** — `meshgeodesic` (geodesic distances / soft selection) and the spectral eigenbasis, if
  you want implicit (unconditionally stable) diffusion instead of explicit steps.

So the cloth, the wicking operator, and the water-mixing are all in the box. Only the *wet* part is new.

## What to build (small, contained)

### W1 — Saturation + dye on the cloth, and the wicking step *(build; reuses `graph_laplacian`)*
Hang two per-vertex arrays on the cloth and diffuse them along its own Laplacian.
```python
# WICKING = heat-diffusion of the wet front ALONG the fabric, on the cloth's graph Laplacian.
# Reuses SoftBody.cloth() (the mesh) and spectral.graph_laplacian() (the operator) -- nothing re-implemented.
import numpy as np
from holographic_spectral import graph_laplacian

def cloth_laplacian(cloth):
    # the fabric's EDGES are already in cloth.constraints as (i, j, rest, compliance) -> adjacency -> Laplacian
    A = np.zeros((cloth.N, cloth.N))
    for (i, j, _rest, _c) in cloth.constraints:
        A[i, j] = A[j, i] = 1.0
    return graph_laplacian(A)                                  # L = D - A            [REUSE: holographic_spectral]

def wick_step(cloth, L, dt, kappa=0.5, gravity=0.2):
    # cloth.s = saturation (0 dry .. 1 soaked);  cloth.d = dye concentration, riding the wet fabric
    down = np.maximum(0.0, -cloth.x[:, 1])                     # a simple "downhill" weight (−y is down)
    cloth.s = cloth.s - dt * kappa * (L @ cloth.s)            # capillary spread: wet -> dry (graph diffusion)
    cloth.s = cloth.s + dt * gravity * down * (cloth.s > 0)   # gravity pulls the wet front downward
    wet = (cloth.s > 1e-3).astype(float)
    cloth.d = cloth.d - dt * kappa * (L @ (cloth.d * wet))    # the dye diffuses only where the fabric is wet
    np.clip(cloth.s, 0.0, 1.0, out=cloth.s)
```
*(For a large cloth, precompute `L`'s eigenbasis once — `spectral.laplacian_eigenbasis(L)` — and diffuse in it
for an unconditionally stable, closed-form step, the same "diagonalise once, evaluate any t" move as the physics
backlog. Explicit is fine to start.)* *Effort: LOW.*

### W2 — The cloth↔water exchange (the "seeping out") *(build; reuses the `Mixture` dye channel)*
```python
# The tank is a Mixture with ONE "dye" channel, starting CLEAR (all zeros).  REUSE: holographic_mixture
from holographic_mixture import Mixture, matter_step
tank = Mixture(shape=(H, W)).add("dye", np.zeros((H, W)), density=1.0, diffusivity=0.02)

def seep_and_mix(cloth, tank, vx, vy, dt, rate=0.3):
    dye = tank.channels["dye"]
    for v in submerged_vertices(cloth, tank):                # cloth vertices sitting inside the water grid
        cell = cell_of(cloth.x[v])                           # which tank cell this vertex is in
        flux = rate * (cloth.d[v] - dye[cell])               # Fick's law: lots of dye in cloth, ~0 in clear water
        cloth.d[v] -= flux * dt                              # cloth loses dye (fades toward the water's level)
        dye[cell]  += flux * dt                              # water gains it -> the red SEEPS OUT here
    matter_step(tank, vx, vy, dt)                            # the flow advects + diffuses it -> it MIXES  [REUSE]
```
The flux is symmetric, so it also runs the *other* way: dip a dry cloth into dyed water and it soaks the colour
*up*. Same code, both directions. *Effort: LOW–MED (the geometry glue — which vertices are in which cell — is
the only fiddly bit).*

### W3 — Wire it in, don't silo it *(the anti-silo close-out)*
A `UnifiedMind` faculty (e.g. `wet_cloth` / `wick_step`) that **delegates** to `SoftBody`, `spectral.graph_
laplacian`, and `holographic_mixture` — it adds the saturation field and the flux, nothing it can borrow. Proven
by a **cross-faculty integration test** in `test_integration.py`, not a private module test:
*soak a cloth patch red (set `s=1`, `d=red` on some vertices), wick it a few steps (assert the red spreads and
sinks), drop it in a clear `Mixture` tank, step, and assert the tank's `dye` channel becomes non-zero exactly
where the cloth is submerged and then spreads outward.* *Effort: LOW.*

## The holographic angle (why it's cheap here)

- **Wicking and mixing are one diffusion on two substrates.** The cloth diffuses in its **graph-Laplacian**
  eigenbasis; the water diffuses in **Fourier** (inside `matter_step`). Both are diagonal in their own basis, so
  both are the "advance = per-mode decay" spectral step from the physics backlog. As above (the water's FFT
  diffusion), so below (the cloth's Laplacian diffusion) — same move, different graph.
- **The dye is one field wearing two coats** — a per-vertex channel on the cloth, a `Mixture` channel in the
  water — joined by the boundary flux. Adding it doesn't fork the dye model; it reuses `holographic_mixture`.
- **Saturation is a second channel the dye rides on** — the same "extra dimension carries an extra property"
  lever. Wet where `s>0`, dyed where `d>0`; the dye only moves through wet fabric.
- **The coupling is a bind at the shared boundary** — a shaped, local transfer where the cloth's vertices sit in
  the water grid; the read-dual of the same "where two fields meet, exchange" pattern the FieldEffect/Sampler
  family already uses.

So the whole effect is: the cloth mesh you already build + one diffusion you already have (`graph_laplacian`) +
the dye-in-water model you already have (`Mixture`/`matter_step`) + a flux the size of a dozen lines.

## Honest scope (kept negatives)

- **This is believable VFX wicking, not first-principles porous flow.** Modelling the wet front as Fickian
  diffusion is the standard, good-looking approximation (it's how ink-in-absorbent-paper is done). True capillary
  dynamics — pressure-driven Darcy/Washburn flow through the weave's pores, anisotropic along warp vs weft — is a
  bigger model and almost certainly more than you need. State which regime a scene is in.
- **It's a grid+mesh coupling.** The one fiddly piece is the geometry glue in W2 — mapping cloth vertices to tank
  cells each step (and back-coupling the cloth's drag on the water, if you want two-way). Keep it explicit and
  readable; a spatial hash over the tank grid makes "which cell is this vertex in" cheap.
- **Mass bookkeeping.** Dye that leaves the cloth must equal dye that enters the water (the flux is symmetric, so
  it conserves by construction) — assert it in the test so a bug can't quietly create or destroy dye.
- **Explicit diffusion has a step limit.** `kappa*dt` too large on a fine cloth goes unstable; either cap it or
  switch to the eigenbasis (implicit) step noted in W1. Deterministic and seeded either way.
- **Saturation caps at 1.** A vertex can't hold more than "soaked"; clamp, and let excess run off (or drip — a
  future emitter trigger where `s` hits 1 at a downward edge).

## Sequencing

1. **W1 wick_step** on a standalone `SoftBody.cloth` — soak a patch, watch the red spread and sink. Reuses
   `graph_laplacian`; no water yet. *Low.*
2. **W2 seep_and_mix** — couple to a `Mixture` tank; the dye bleeds out and mixes. Reuses `matter_step`. *Low–med.*
3. **W3 faculty + integration test** — the soak→drop→mix pipeline through `UnifiedMind`. *Low.*
4. *(optional)* two-way drag (cloth pushes the water), drip triggers, warp/weft anisotropy (a directional `L`).

## Close-out ritual

Module + `_selftest`; a pytest pinning behaviour (a soaked patch's saturation spreads and conserves; the dye only
moves through wet fabric; the tank dye appears at the submerged cells and then spreads; dye is conserved across
the flux); a default-off `UnifiedMind` faculty delegating to `SoftBody`/`spectral`/`mixture`; the cross-faculty
integration test (soak red → drop in clear tank → water tints where it bleeds); README count via `sed`;
`NOTES_concepts.md` with the kept negatives (Fickian-not-Darcy, grid+mesh coupling, step limit); a `tour.py`
block (a red cloth bleeding into a tank); clean zip rebuild + verify. Readable, WHY-commented code throughout.

---

### Bottom line

Yes — cloth can soak up liquid, wick it through the weave, and bleed dye into a tank that mixes into the clear
water, and after the latest build most of it is reuse: the cloth is `SoftBody.cloth` (with its edges in
`cloth.constraints`), the wicking operator is `spectral.graph_laplacian`, and the dye-in-water mixing is the
`holographic_mixture` model (`Mixture` + `matter_step`) you already have. The new work is a saturation+dye field
that lives on the cloth (diffused along its Laplacian — the same spectral diffusion the water uses, on a
different graph) and a Fick's-law flux where the cloth meets the water (which, being symmetric, soaks *up* dyed
water too). It's a believable VFX coupling rather than first-principles Darcy flow, it wires into `UnifiedMind`
as a faculty proven by a soak→drop→mix integration test, and it's about a dozen new lines over machinery that's
already there.

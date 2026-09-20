---
name: lecore-simulate
description: "Simulate and generate with the leCore engine: rigid/soft/cloth bodies with SDF contact, particles, fluids, smoke, fire, gases, waves, water, acoustics, heat, phase change, the physical-material DB, fields, noise, textures, crystals, lattices, growth, trees, terrain, fractals — and how each reaches a picture. Use when leCore should simulate, drop, collide, drape, burn, heat, grow, erode or generate geometry/materials/fields."
---

# Simulating and generating with leCore

Everything here is what the repo's own code, cards (`CAPABILITIES.md`, `holographic_catalog_p*.py`)
and notes (`docs/NOTES_concepts.md`) say, checked against a lab (`sim_lab.py`, `render_lab.py`
physics stage: one cube on a plane, 2 CPU cores, 160×120) — numbers say which. Two engine bugs
were found and fixed while building the lab; both are in section 3. **`find_capability(problem)`
first**: the catalog answered every domain below by name in one call.

## 1. Mental model

**Fields are the lingua franca.** A solver holds a grid or a particle cloud; a renderer wants a
callable `field(P) → density` or an SDF `eval(P)` or points. The typed pipeline (`IO_KINDS`:
field 199 members, mesh 151, sdf 95, image 78, points 51…) and `suggest_pipeline(start_kind,
goal_kind)` chain them. Two unrelated classes are both named `Field`: `holographic_fieldhome.Field`
(spatial: `.grid/.sparse/.callable`, `sample(points)`) and `holographic_field.Field` (hypervector
space). Don't conflate them.

**Every solver has its own coordinate space.** `ParticleSystem` and `fields.*` live in cell
indices with `(x=col, y=row)` and periodic wrap; `MPMSnow` in grid units (x≈19–29); the ocean in
metres; `StableFluid` in its own periodic box; SDFs and cameras in scene units. `body_animation`
auto-frames precisely because of this. Convert explicitly.

**Determinism is a CPU property and a seed argument.** Every seeded door uses
`np.random.default_rng(seed)`; `StableFluid` has no RNG; collision and constraint code is RNG-free;
`spatial_hash_pairs` sorts its output. Reproducible within a run/build, not across NumPy builds.

**Sim is cheap, pictures are not.** Lab: rigid step 1 ms, particle step 0.4 ms, 128² fluid step
10 ms, cloth step 114 ms — against 100–300 ms per preview frame. Step many times per rendered frame.

## 2. Getting a picture out of a sim

| Sim output | Door | Lab timing |
|---|---|---|
| density grid | `render_volume(field_callable, cam, bounds, w, h, steps, mode="smoke"\|"fire"\|"density")` → `(rgb, alpha)`; `field` **must be callable** `P(N,3) → density ≥ 0` — wrap a grid (nearest or trilinear) | 0.03 s/frame at 160×120, 48 steps, 32³ grid |
| points | `splat_points(P, cam, w, h, colors, radius_px)` → `(image, alpha)` | 5 ms/frame for 600 points |
| a moved rigid body | rebuild its SDF per frame (`box().rotate(axis, angle).translate(c)`), render with `render_sdf` / `RenderSession.preview()` | 100–300 ms/frame |
| cloth / soft mesh | `to_mesh()` → `render_mesh`, or splat the nodes | — |
| height field | `terrain_to_mesh`, `WaterBody.mesh` / `.render('fast'\|'final')` | ocean fast 0.61 s |
| ready-made clips | `particle_animation`, `smoke_animation`, `body_animation`, `render_animation(keys)` — frames + GIF | door-quality only |

Composite volumes over the stage by alpha; write clips with an adaptive-palette GIF or an MP4
(section 9) — **not `save_gif`**, whose fixed 6×7×6 palette recolours greys mauve and copper yellow.

## 3. Bodies and contact

**Rigid** — `RigidBody(positions, inv_mass=None, velocities=None)` (shape matching; live state is
`.x`, **`.rest` is the rest pose and animates nothing**), `.step(dt, gravity=(gx,gy,gz), stiffness,
floor=, restitution=)`. `.step` knows only a flat floor. **Lab finding (fall-through):** a
per-particle `resolve_sdf_collision` after the step leaves the fall velocity intact; the body sinks
a little more each frame, and once the nearest surface is a *side* face the particles are kicked
sideways (vx 1.2 m/s at first contact) — the cube walked out through the other cube. The working
recipe is `rigid_contact.rigid_step_contact(body, dt, gravity, sdf_eval, restitution, friction)`:
`advance_ccd` (swept, no tunnelling, into-surface velocity cancelled) → shape match → **move the
whole body out along the contact normal of the deepest stopped particle** (never per-particle near
an edge) → residual resolve → contact velocity response with friction. Measured: a 4×4×4 lattice
cube dropped on another rests at y = 0.950 exactly, worst penetration 0.00000, 1 ms/step; an
overhanging one tips off and settles beside. Kept negative: a body whose centre of mass is inside
the support still tips when fewer than half its contact particles are supported — shape matching
is not a torque-balanced contact solver. **There is no rigid-vs-rigid solver**: the other body is
a static SDF.

**Soft / cloth / rope / hair** — `SoftBody` (XPBD; `cloth(rows, cols, spacing, compliance)`,
`cloth3d(..., bending=)`, `rope(n)`, `soft_box`, `from_mesh`), `.step(dt, gravity, iterations=20,
substeps=1, solver="xpbd"|"pbd", collider=sdf_eval, collide_radius, continuous=True for swept
contact, floor, restitution, damping, sleep, stiffness=(hertz, zeta))`, `.add_distance/bending/
volume/self_collision`, `.pin(i)`, `.constraint_residual()`, `.islands()`, `.to_mesh()`.
`cloth3d` **pins its first row by default** (`w=0`, a curtain) — set `w=1` for a drop. **Engine
fix (2026-09-14, `holographic_softbody.py`)**: environment contact used to subtract the push-out
from the velocity, so a sheet resting on a cube kept free-fall velocity (vy −9.8 after one second
at rest), penetration grew per substep, and the sheet slid off the side to the floor; velocity is
now read from the resolved position. After the fix an 8×8 sheet rests at y = 0.700 with |vy| 0 and
a 14×14 sheet drapes (y 0.32–0.72, residual 0.02); 114 ms/step at 12 iterations × 2 substeps. All
19 softbody/collide tests pass. XPBD compliance is timestep-independent (0.005 between 1 and 6
substeps); PBD stiffness depends on iteration count. Hair: `CosseratStrand` / `groom_hair`,
`simulate_hair`, `hair_wind`.

**Collision primitives** (`holographic_collide.py`, all point-vs-SDF, RNG-free):
`resolve_sdf_collision(X, sdf_eval, radius)` (medial-axis escape built in),
`time_of_impact(X, V, dt, sdf_eval)` (conservative advancement = sphere tracing),
`advance_ccd(X, V, dt, sdf_eval, radius, restitution)` → `(X, V, hit)` (a 30 m/s body through a
0.1 m wall: discrete tunnels, swept stops exactly), `resolve_swept_collision` (module-only, no
door), `sdf_collision_projection` (for `project_onto_constraints`), `classify_contact` (a label,
not a response), `sdf_offset` (a margin *detects* proximity, it does not prevent tunnelling — it
pushed a fast body out the wrong side). Neighbours: `spatial_hash_pairs`, `pairwise_repulsion`.

**One solver, many uses** — `project_onto_constraints(x, projections, iters, tol, omega,
sweep="sequential"|"simultaneous", average, stiffness=(hertz, zeta), return_residual)` is the same
iterate-a-projection engine the resonator and denoiser use; `stiffness=(inf, zeta)` is the hard
projection bit-identically. Integrators: semi-implicit Euler ("symplectic"), explicit Euler,
Verlet — **no Runge-Kutta, no adaptive step**.

## 4. Particles, emitters, fields of motion

`emit_from_surface(sdf_eval, n, bounds, speed, weight, seed)` → **`(P, normals, vel)`** (Newton-
projected onto the zero set; lab 600 points in 0.04 s). `advance_particles(pos, vel, force, dt,
damping, wrap_to)` (N-D, stateless), `particle_sim(pos, vel, force_fn, integrator)`,
`ParticleSystem` (2-D only, grid units). Couple to a flow with `sample_field` / `scatter_to_field`
(exact adjoints), `drag_force`, `attractor_force` — measured but **no coupled two-way door: you
write the loop**. Lab fountain: gravity + `advance_ccd` against the floor 0.37 ms/step, splat 5
ms/frame. **Trap:** the `particle_animation` door unpacks emit's tuple as `(pos, vel, normals)`,
so its `speed=` has no effect — every particle starts at unit speed.

Fields: `diffuse`, `advect` (semi-Lagrangian, `boundary="wrap"|"wall"`, `solid=` masks),
`project_divergence_free` (FFT), `curl`, `vorticity_confinement`, `buoyancy_force`, 3-D twins
(`*_3d`; **3-D up is +y = axis 1**, while `StableFluid` defaults `up_axis=0`), `spectral_field
(shape, beta, seed)` — the only tileable generator (64³ in 0.02 s, seam continuity 1.009), and
`curl_noise(res, bounds, octaves, seed, obstacle_sdf)` — divergence-free to exactly 0.0 (res 48:
0.68 s), no-slip not enforced.

## 5. Fluids, smoke, fire, gases

`fluid_solver(shape, **kw)` → `StableFluid((ny, nx)` or `(n, n, n), dt, viscosity, diffusion,
dissipation, cooling, buoyancy_alpha/beta, vorticity, up_axis, ignition, burn_rate, smoke_yield,
device="cpu"|"gpu")`; state `.vel .density .temperature .fuel`; `add_source(region_slices_or_mask,
density, temperature, fuel, vel)`; `.step()`; `.divergence()`. FFT Helmholtz projection with the
centred-difference symbol (divergence ~5e-16), semi-Lagrangian advection (**no CFL limit**, dt 1.0
stable), **periodic boundaries, no obstacles** — put the box above the object and emit at its
bottom; obstacles exist only on the functional `fluid_step`/`smoke_step` path via `solid=` masks
(approximate no-slip). Lab: 128² 9.8 ms/step; 32³ with combustion 49 ms/step; notes: 64³ ≈ 0.5
s/step, ~20 % smoke mass lost over 60 steps, vorticity confinement keeps 88× more swirl; float32 +
`roi=` halves advection. `device="gpu"` is CuPy/CUDA, wired, **unmeasured**. Mixture model:
`make_mixture` / `matter_step(mix, vx, vy, dt)` — the field lives in `mix.channels[name]`,
buoyancy needs component density < 1, thread `vx, vy` between calls (five probes to learn that).

Fire: `fire(material, fuel_kg, temp_K)` → `Fire.step(dt)` → `{temperature_K, flame_color,
burned_kg, heat_J, smoke_color, soot_mass, fuel_left}` (lab wood: 1400 K, flame (1, 0.38, 0));
`configure_fluid(fluid, material)` sets the solver's ignition/burn/yield from the 8-material
`COMBUSTION` table (art-directable, not a lab dataset); `emit_smoke`, `burn_object`,
`oxidation_field`, `combustion_products("wood", 1 kg)` → 18 MJ, 0.55 kg soot. `render_volume
(mode="fire")` is a hand-authored density→colour ramp, **not blackbody**. Gases:
`gas_pressure(density, T, name)` (1.2 kg/m³ air at 293 K → 100,983 Pa), `gas_density`,
`speed_of_sound`, `adiabatic` — an ideal-gas parcel; **no compressible flow, no shocks**.
`nebula_volume` is an artist's nebula, not hydro. Absent, loudly: SPH, FLIP/PIC/APIC, LBM.

## 6. Waves, water, acoustics

`water_body(container=None|'glass'|'pool'|'bowl'|sdf, level, preset="ocean", extent, res, t, seed,
ripple, material)` → `WaterBody` (`.render('fast'|'final')`, `.mesh`, `.camera()`, `.at_time(t)`);
Gerstner, metres, exact analytic normals, deterministic in `(seed, t)`, steepness bound enforced;
kinematic — no breaking, no mass. Lab: build 0.04 s, fast render 0.61 s; notes: contained water is
a refractive path trace ~60 s at 20 spp. `gerstner_waves(...)`, `wave_packets` (packets that
shoal and diffract), `plan_waves` / `solve_waves` (per-tile method choice). `wave_field(shape, c,
dx, damping, absorb_border)` → `WaveField` (`pulse`, `source`, `energy`; `stable_dt` = 0.9·dx/
(c√ndim), lab 96² at c = 343: 0.00186; **any requested dt is auto-subdivided** — 20 steps of 0.05 in
0.05 s); scalar/linear only. `spectral_wave` / `spectral_ocean` advance to any `t` in closed form.
`free_surface` / `break_wave` — ballistic particles, the only multi-valued surface
(`is_overturning`, `is_multivalued` both True in the lab). Acoustics: `acoustic_interface(a, b)`
→ `(R, T)` energy fractions from Z = ρc (water→steel 0.881 / 0.119), `acoustic_impedance`,
`levitation_chamber` (Gor'kov nodes at λ/2), `chladni_plate`. No elastic/vector waves, no PML.

## 7. Heat, phase, materials

**Physical DB** — `PHYSICAL_MATERIALS`: 116 materials, 12 categories, CRC-style, fields omitted
rather than guessed (`field_coverage()`: thermal_conductivity 87, sound_speed 77, youngs 76,
refractive 55, melting_point 34, viscosity 28, thermal_expansion 21). Doors `physical_material
(name)`, `material_data(name|category)`, `material_info(name)` (the render↔physics bridge:
copper is in both), `find_materials` (word overlap, not semantic), `validate_materials`. Render
presets: 141 in `matlib`. Units in `UNITS` (SI, 20 °C, 1 atm).

**Heat** — `diffuse_heat(temp, alpha, dx, dt, steps, bc="neumann"|"periodic", operator=)`: explicit
stencil, **auto-substeps** so any dt is stable, Neumann conserves heat (lab: 96² steel plate, one
real second in 7 ms, total heat conserved); periodic is the exact spectral form (2e-16 vs 1.5e-4
iterative); a prebuilt `operator=` is 1.9× and bit-identical. `heat_body(material, mass, T)` →
`HeatBody.add_energy / newton_cool(ambient, hA, dt)` (steel 1 kg from 800 K: 537, 411, 350 … 294 K
per minute at hA = 5). `steady_heat(sdf, boundary_T, points)` walk-on-spheres. `material_thermal
(name)` returns **`thermal_conductivity: None` for diamond, tungsten, oak_wood** — it reads a
7-entry `enrich.json` plus 9 fallbacks, not the 87-entry DB; use `physical_material(name)
['thermal_conductivity']`. `diffuse_spectral`, `diffusion_transfer`, `solve_poisson_periodic`.

**Phase and chemistry** — `phase_state(material, mass, T)` (water/iron/aluminum only; latent heat
holds the plateau: +100 kJ × 12 on 1 kg water → 317, 341, 365, then 373.1 K flat); `boiling_point_at
(P)`. `reaction_diffusion_cells` (Gray-Scott on a cell graph), `morphogenesis_grow/relax/
differentiate`. **No FEM heat, no emissivity, no convection, thermal_expansion is dead data, nothing
melts geometry.**

**Heat → picture** — `matlib.heat(name, T)` stamps `temperature_K`; `shade(mat, n_points)` adds a
blackbody-shaped emission above 500 K (steel 1400 K → (0.36, 0.14, 0)); feed it as the emission
channel of the path tracer (lab: glowing cube, `render_auto draft` 5.3 s). `blackbody_color(T)` is
Planck × CIE, **scalar only** (1000 K (1, 0.19, 0); 6500 K white) — loop it for a field. No heat-map
colormap faculty; `ramp()` is the stand-in.

## 8. Fields, noise, textures

`proc_texture(name)` → `f(P (M,3)) → (M,)`, 14 names; lab per 100k points: checker 3 ms, magic 20,
marble 88, wood 91, fbm 100, musgrave 109, voronoi 210. **Ranges are not [0,1]** (voronoi max 1.11,
fbm min 0.06, `procedural_noise` [−0.35, 0.23]): `saturate` before a mix weight (an unsaturated fbm
mix measured RGB (1.30, 0, −0.30)). `texture_image` / `texture_volume` **min-max normalise** — absolute
values are lost. `white_noise` has no scale (constant over [0,1)³). VSA band-limited noise
(`procedural_noise`, `FractalNoise`) is ~1000× slower than `proc_texture('fbm')` per point
(`Terrain.heightmap(128)` took 11.5 s through it) — `sample_grid_fast` is 38× the loop.
Hash-lattice `pattern_field(name)` (noise, fbm, checker, stripes, gradient, dots; `*32` GPU-
reproducible twins; `pattern_to_glsl` emits only the closed-form ones). Texture graph
(`texture_leaf`, `texture_op` over mix/add/multiply/scale/over/remap/min/max/clamp/saturate,
`bake_texture(res)` 40× per sample after ~1000 samples). Feeds: `Param(field=)` material
channels, `bake_material`, `displace(target, fn, amount)` / `SDF.displace(amount, freq)`,
`field_displace(mesh, field, amount)` (28 ms on a 24² grid), `auto_displace`/`auto_bump` from an
image (abstain gate), `synthesize_texture`. Field-as-hypervector: `mesh_to_field_vector`, `bake_field
(_nd)`, `gabor_volume` (+14 dB over Gaussians at equal count), `holographic_fog_volume`. **Absent:
simplex, true Perlin, `worley()` (use `voronoi`), tileable textures (only `spectral_field`), triplanar
field sampling, filtered/mip lookups, roughness/metallic map generation, normal maps from fields.**

## 9. Crystals, lattices, growth, procedural geometry

**Lattices** — `lattice_basis(system, a, b, c, α, β, γ, centring)`, `lattice_sites(system, centring,
a, extent)` (lab FCC extent 2: 500 sites, nearest a/√2, coordination 12), `lattice_bonds`,
`crystal_systems()` (7 systems, exactly 14 Bravais lattices; illegal centrings raise),
`crystal_form_faces`. **`crystal_habit(form=False)` is the default and {100} without `form=True` is
a slab** (volume 5.76 vs 1.00).

**Habits and growth** — `crystal_single(habit, size, real_cell=True)` (12 habits: quartz, beryl,
cube, octahedron, dodecahedron, needle, druse, amethyst_druse, quartz_point, quartz_long,
amethyst_pyramid, `quartz_rz` = class-32 quartz with r/z terminations; `real_cell` is **off by
default** and the default cell gives m^r 139.1° instead of 141.8°), `crystal_grow_on(sdf, bounds,
count, habit, size, size_jitter, inward, tilt, where, seed, cull=True, pack=k, real_cell,
clip_to_substrate, seeds)` (a crystal grows perpendicular to its substrate; `inward=True` is a
geode; lab: 14 points on a cube top in 0.02 s; `batched=True` is a refuted optimisation, `cull` is
the speed knob, `pack≈0.55` stops shard fields), `crystal_cluster`, `crystal_geode`, `crystal_cut`,
flaws (`crystal_cloudiness/inclusions/phantom/fractures` are optics; only `crystal_chipped` changes
geometry). **Verify a habit before rendering it** — the facet probe: bisection along random
directions from the centre, gradient = facet normal, **merge neighbouring normal bins, then
threshold** (thresholding first counted 6 faces on quartz_rz; merged first: 12 = 6 prism + 6
terminal; octahedron 8). The hexagonal form bug that shipped four renders of rhombic blades was
caught by exactly this probe. Absent: twin laws, striations, re-entrant intergrowth (unions only),
3-D DLA.

**Growth** — `grow_stages(kind, spec, n)` / `grow_at(kind, spec, t)` / `growth_report` for
`crystal | dendrite | plant | tree`; `grow_ice(shape, eta, steps, seed)` / `grow_lightning` /
`dielectric_breakdown` (2-D DLA; eta 1 = dendrite, large = stringy; lab 81² in 0.3 s, fractal
dimension 1.05); `variant(spec, seed)` pools. Trees: `crown_attractors(n, centre, radius, shape)`
→ `grow_tree(attractors, root, step, influence, kill)` (space colonisation; **check `terminated ==
"attractors_consumed"`**; lab 76 nodes in 0.03 s) → `taper_radii` (da Vinci, 1e-9) → `tree_mesh`
(12 verts per branch) / `tree_instanced` (420×). The tree grows along **+z**; swap axes for a y-up
stage; tip radius 0.006 is sub-pixel at 160×120. Plants: `lsystem` → `grow_plant`. IFS:
`ifs_generate` (5 systems), `ifs_fit` (snap-to-library, not rotation-invariant). Cells:
`crystal_material` (Voronoi colour/cracks, not unit cells), `morphogenesis_*`.

**SDF and parametric geometry** — primitives are functions (`sphere, box(bx,by,bz), torus,
cylinder, plane, capsule, cone, ellipsoid, octahedron, menger, fold_fractal` (Mandelbox),
`mandelbulb`), combinators methods (`union, intersect, subtract, smooth_union(k), fillet_union,
translate, scale, rotate(axis, angle), mirror, fold, repeat, rounded, onion, elongate, displace,
twist, bend`); `.cost()` (menger alu 39 cheap, mandelbox 95, mandelbulb 253 expensive), `.to_dsl()`
/ `parse_dsl`, `.to_glsl()`, `.to_jit_expr()` (refuses twist/displace/bend/ellipsoid/fractals).
**Fractal DEs are bounds**: `render_sdf` marched at full step and landed inside (black speckle);
wrap `eval` to return 0.5× and it renders (0.5 s at 160×120). 2-D profiles (`circle2d, box2d,
rounded_box2d, ngon2d, polygon2d`) → `sdf_extrude(sd2d, height)` / `sdf_revolve(sd2d)` — these
return **callables with `eval`, not nodes** (no `.translate`; wrap them). `sweep_tube(points,
profile, radius)`, `sweep_profile`. Terrain: `terrain(seed)` (`.heightmap(res)`, `terrain_to_mesh/
sdf`), `terrain_erode(height, droplets=2000)` (0.5 s at 128²; **positive-gain instability above
~20 000 droplets — documented limit, not fixed**), `scatter_on_terrain`, `scatter_mesh`,
`grass_blade`, `greeble_mesh`, `procedural_object`, `metaball_mesh`, `voxel_remesh`, `auto_retopo`
(cubic knob, capped), `convolution_field` (no bulge at joints). **Absent: loft, gears/involutes,
text/glyph geometry, lattice-infill generators, a general (u,v) parametric surface door.**

## 10. Infrastructure

`Simulation(solver, step_fn, field_fn, lo, hi)` / `Simulation.for_fluid` / `for_automaton` →
`.step / .run / .grid / .field / .render`; `run_simulation(kind='fluid'|'automaton'|'smoke',
steps, grid, seed)` is the stateless JSON twin. `run_until_settled(step, state, steps, residual,
settle_tol)` (i.i.d. guard **and** residual ≤ tol × peak; 600-frame fluid settled at step 96, 4.7×,
max error 0). Islands + sleep (`islands`, `island_sleep_tracker`, `step_islands`; skipping a
sleeping island is bit-identical; the closed-form fixed point is the mean, not rest).
`stream_meter` (online = batch). `Timeline / Transport / FrameCache` are playback, not a sim
clock. **No sim job/checkpoint system**: `render_progressive` exists because MC buckets commute;
a time-stepped solver has no pause/resume, `Simulation` has no save/load (only `GameShard.
save_state`), and no solver consults the frame budget. Long runs: your own loop, detached, with
per-frame outputs on disk. Benchmarks: only `bench_fluid`; no softbody/collision/MPM bench.

**Writing clips**: adaptive palette (Pillow median cut + Floyd–Steinberg, one palette per clip)
or MP4 via `imageio_ffmpeg.write_frames(path, (w16, h16), fps, pix_fmt_in="rgb24")` with frames
**padded to 16-px macroblocks** (imageio otherwise resizes silently); `render_lab.write_gif_adaptive`
/ `write_mp4` are the reference implementations. Pillow merges identical consecutive GIF frames
into longer durations (a resting body yields fewer frames — that is fine).

## 11. Measured, in one place (lab, 2 cores)

| Thing | Number |
|---|---|
| RigidBody + swept contact | 1 ms/step; rests at y = 0.950, penetration 0 |
| cloth3d 14×14, 12 iters × 2 substeps, SDF collider | 114 ms/step; drapes after the fix |
| emit 600 / advance_ccd / splat | 0.04 s / 0.37 ms per step / 5 ms per frame |
| StableFluid 128² / 32³ + fire | 9.8 / 49 ms per step; divergence 5e-16 |
| render_volume fire 160×120, 48 steps | 0.03 s/frame |
| diffuse_heat 96², 1 s real time | 7 ms; heat conserved |
| spectral_field 64³ / curl_noise 48 | 0.02 s / 0.68 s; seam 1.009 / div 0.0 |
| proc_texture per 100k points | 3 ms (checker) … 210 ms (voronoi) |
| Terrain.heightmap(128) via VSA noise | 11.5 s (the loud one) |
| crystal_grow_on 14 / grow_tree 250 / grow_ice 81² | 0.02 s / 0.03 s / 0.3 s |
| water_body ocean res 96 fast render | 0.61 s |
| WaveField 96², 20 steps auto-substepped | 0.05 s |

## 12. Kept negatives and absences (do not re-litigate)

No SPH, FLIP, LBM, compressible gas, two-way coupling door, rigid-vs-rigid contact, FEM heat,
emissivity, convection, melting geometry, RK integrators, sim checkpoints, GPU-measured solver,
simplex/Perlin/Worley by name, tileable textures beyond `spectral_field`, loft, gears, text
geometry, 3-D DLA, twin laws, erosion above ~20 k droplets. `save_gif` for colour. `material_thermal`
for conductivity. `particle_animation`'s `speed=`. `cloth3d`'s pinned first row. `.rest` vs `.x`.
`form=False` and `real_cell=False` defaults. Thresholding facet bins before merging. Full-step
marching of a fractal DE. Per-particle push-out of a rigid body near an edge. Subtracting the
environment push-out from a cloth's velocity (fixed).

## 13. Teach what you measured

Before: `ask`, `find_capability`, `wisdom`. After: `teach(question, answer-with-numbers)`,
`bequeath(lesson)`, `learning_save(root, path=<root>/learning/state-<stamp>Z.lecore)`.
`sim_lab_teach.py` is the worked example (13 rows + 2 lessons; re-ask returns T0 after a reload).
Deliver the lab's `results.json`, the clips, the engine diff and the stamped state file together.

## 14. Decide, don't guess: typed decisions, records and validated termination in a sim session (sweep 176)

The Jev-shaped doors are built into the verbs a sim session already uses; the syntax reference with
live-generated examples is `docs/TYPED_DECISIONS.md`. What they measured on exactly this kind of work:

- **Pick a solver, a preset or a material as a TYPED decision, not a model guess.** `typed(state,
  [options], examples=...)` builds the schema, lints it and the state (findings attached to the answer),
  scores it, and returns one answer with an `id`. Measured in the 3-D session: picking a material from the
  141-name library by a one-line part description was 8 of 8; picking a *tool* from the same descriptions was
  1–3 of 8 until the parts were described in the question's own vocabulary (geometry, mechanics) with
  balanced example budgets, then 8 of 8. Describe the shape or the physics, not the function; `systemone_lint`
  now flags both failures (imbalanced budgets, multi-clause or contrastive states) before you ask.
- **`route(task)` is tiered** (act / choose / abstain with `tier`, `z`, `id`); `plan_change("do we have a
  solver for X")` answers Rule 0 with the catalog tier and `code_search` as evidence (reuse / extend / build,
  11 of 12 on history) — run it before hand-rolling any solver, and report `decision_outcome(id, what you
  did)` so the next plan learns.
- **Every step of a swarm sim is a `swarm_step`** — refused without `done_when` and `evidence`; with
  `verify={"verb": ..., "args": ...}` the check runs BEFORE the step is accepted. For a solver, done_when
  is a number this skill already produces: `run_until_settled`'s settled step and residual, divergence
  (StableFluid 5e-16), penetration (0), heat conserved. `swarm_evaluate()` is the exit.
- **No sim checkpoint system exists (§10) — so checkpoint yourself, per frame.** A 30-minute render died at
  a turn boundary with nothing written; the fix that worked (`speaker_render_strips.py`) is the pattern for
  a long sim: your own loop, detached, one output per frame or bucket on disk, a rerun that skips what is
  on disk. Two more sandbox lessons: a script run from another directory imports whatever `lecore` is first
  on `sys.path` (a PyPI `leos-core` installed by a studio app shadowed the checkout — `PYTHONPATH=.`), and
  `pgrep -f` from a shell whose own command line contains the pattern kills that shell.
- **The reflex learns from use.** After `decision_outcome`, a repeated request is answered from experience
  (`route(..., reflex=True)`: repeats 99.3% at 0.982), and `verify_decision(state, answer)` catches a served
  answer that contradicts what was learned (forward/backward lookup AUROC 1.000). For typed decisions the
  count table stays the learner — the reflex measured worse there, so it is off by default.
- **Teach as before (§13), and report outcomes by id as well:** `teach` is prose the next session recalls at
  T0; `decision_outcome(id, truth)` is the structured path the table, the reflex and the calibration learn
  from. Both.


# 3D, Rendering & Simulation — a practical guide

This guide is the map I wish I'd had: how to actually *make a picture* with leCore — a scene, a cloud, smoke,
a lit sphere — without reverse-engineering the call chain first. Every code block here has been run as written.

> **Import convention.** Everything starts from a `UnifiedMind` (the `mind`), and the engine modules live under the
> `holographic` package. Two equivalent entry points:
> ```python
> import lecore
> mind = lecore.UnifiedMind(dim=512, seed=0)      # `import lecore` is the shim; this is the object you use
> ```
> Low-level pieces are imported by package path, e.g.
> `from holographic.rendering.holographic_render import Camera, volume_render, save_png`.

---

## 1. The mental model (read this first)

There are **two kinds of thing** you can render, and they go down **two different pipelines**. Knowing which is
which is 90% of the battle.

| You want… | It's a… | Rendered by | Example materials |
|---|---|---|---|
| A solid object with a surface (ball, box, glass, metal) | **Surface** (an SDF) | the **surface renderer** (`render_scene`) | metal, glass, matte, gold, mirror, wax |
| Something made of *stuff in the air* (cloud, smoke, fog, fire) | **Volume** (a density field) | the **volume renderer** (`volume_render`) | cloud, smoke, fog, fire |

A **surface** is defined by a signed-distance function (SDF): "how far am I from the object." A **volume** is
defined by a density field: a plain callable `field(points) -> density>=0` that says "how much stuff is at each
point in space." Clouds are volumes. This is why a cloud is not just "a fuzzy sphere" — it has no surface at all.

The good news: the **semantic scene system knows this split already**. When you describe a scene, each object is
tagged surface or volumetric and sent down the right pipeline automatically. You only need the low-level pieces
when you want direct control.

---

## 2. Clouds — three ways, easiest first

### 2a. Just describe it (semantic scene)

The scene parser understands cloud words (`cloud`, `cloudy`, `fluffy`, `cumulus`, `puffy`) and builds a
self-shadowed volumetric cloud for you:

```python
import lecore
mind = lecore.UnifiedMind(dim=512, seed=0)

scene = mind.build_scene("a large fluffy cloud")
img = scene.render(width=384, height=384)          # (H, W, 3) float image in [0, 1]
mind.save_render("cloud.png", img)
```

That's the whole thing. `build_scene` returns a `SemanticScene` you can also talk to — `scene.adjust("make it
bigger")`, `scene.describe()` — before rendering.

### 2b. One call, low-level shortcut (`mind.make_cloud`)

When you don't need a whole scene, `make_cloud` builds the density and renders it self-shadowed in one call:

```python
img = mind.make_cloud(radius=1.0, seed=3)          # default 3/4 camera, 384x384
mind.save_render("cloud.png", img)
```

Useful knobs: `center`, `radius`, `density`, `seed` (change the noise → a different cloud), `camera`,
`width`/`height`, `sky` (background colour), `sun_dir`, `grid` (noise detail — see the cost note below).

### 2c. Full manual control (build the field yourself)

Reach for this when you want a custom density, custom lighting, or to composite the cloud into your own frame.
The two ingredients are a **density field** and the **volume renderer**:

```python
import numpy as np
from holographic.simulation_and_physics.holographic_semantic import cloud_field
from holographic.rendering.holographic_render import Camera, Light, volume_render, save_png

field  = cloud_field(center=(0, 0, 0), radius=1.3, density=3.4, seed=7)     # multi-lobe cumulus density
camera = Camera(eye=(2.5, 0.18, 3.4), target=(0, 0.05, 0), fov_deg=40, aspect=1.4)
bounds = (np.array([-2., -1.1, -2]), np.array([2., 1.4, 2]))               # the box the field lives in

img, alpha = volume_render(field, camera, bounds, width=560, height=400, steps=190,
                           mode="smoke", sigma=6.5, albedo=(1.0, 0.99, 0.97),
                           lights=[Light("directional", direction=(-0.42, -0.5, -0.62))],
                           self_shadow=True, shadow_steps=28, shadow_sigma=9.0,
                           ambient=(0.42, 0.53, 0.72))          # <-- the settings that make it look real
# composite over a sky gradient
yy = np.linspace(0, 1, 400)[:, None, None]
sky = np.array([0.18, 0.38, 0.72]) * (1 - yy) + np.array([0.66, 0.80, 0.95]) * yy
save_png("cloud.png", np.clip(img * alpha[..., None] + sky * (1 - alpha[..., None]), 0, 1))
```

**What actually makes a cloud look real** (all of this is baked into `cloud_field`, but worth understanding when
you tune it):

1. **A lumpy body, not one ellipsoid.** A single blob renders as an *egg*. `cloud_field` builds the shape as a
   **smooth-union of many spherical lobes** — a wide low body with a cluster of billows rising on top — so the
   puffs merge into one cauliflower mass with a bumpy crown.
2. **A flat base.** Cumulus sit on a level bottom (the condensation line). The density is hard-cut below the
   cloudbase and feathered just above it.
3. **Eroded, wispy edges.** Domain-warped fBm carves the billows, with *stronger* erosion where the cloud is thin
   (the rim), so edges break into wisps instead of a clean outline.
4. **`self_shadow=True` with light that penetrates.** This is the lighting half of realism. Self-shadowing marches
   each sample toward the sun so the crown is bright and the base falls into shadow. But it only shows if light
   can *get in*: keep `sigma` modest (~6–8) and density moderate. Crank them too high and the cloud becomes an
   opaque white shell with no internal modelling — the #1 reason a volumetric cloud looks flat. `ambient` is the
   sky-blue fill that colours the shadowed underside (real clouds have blue-grey, not black, undersides).

> ### ⚠️ Cost note — read before you pick a `grid`
> `cloud_field` builds fractal noise by **baking it onto a `grid`³ lattice once**, because the underlying noise
> query is per-point (~2 ms each). Baking dominates the cost; the render itself is ~1 s.
>
> | `grid` | bake time | use for |
> |---|---|---|
> | 24 | ~30 s | quick previews |
> | **32** (default) | ~60 s | normal use |
> | 48 | ~4 min | hero stills |
>
> If you're rendering many frames of the *same* cloud (e.g. an orbit), build the `field` **once** and reuse it —
> the bake is in `cloud_field`, not in `volume_render`.

---

## 3. Building scenes from words (surfaces)

The semantic scene system turns a description into named, adjustable objects. Vocabulary is a **controlled**
(deliberately small, deterministic) set:

- **Shapes:** `sphere` (ball, orb, globe), `box` (cube, block, crate)
- **Colours:** red, green, blue, yellow, orange, purple, pink, cyan, white, black, gray
- **Materials:** metal, glass, gold, copper, matte, plastic, ceramic, mirror, glossy, emissive, wax (subsurface),
  translucent — plus the volumetrics: `cloud`, `smoke`, `fog`, `fire`
- **Sizes:** tiny, small, big/large, huge (and stretches)
- **Relations:** "next to", "on", "inside", "under", "beside"

```python
scene = mind.build_scene("a big red metal sphere next to a small blue glass box")
img = scene.render(width=256, height=256, quality="fast")
mind.save_render("scene.png", img)
```

Each object gets a human name from its words ("big red metal sphere"), so you can adjust it the way you'd say it:

```python
scene.adjust("make the sphere gold")
scene.adjust("make everything matte")
img = scene.render()
```

`scene.render(quality=...)` has two levels:
- `quality="fast"` (default) — the single-pass surface renderer, seconds. Great for iterating.
- `quality="hyperreal"` — the Monte-Carlo path tracer (`render_scene_pbr`), slower, physically-based.

Mixing works: describe `"a metal sphere and a puffy cloud"` and the renderer draws the sphere as a surface and
composites the cloud as a volume over it, in one `render()`.

---

## 4. Cameras and lights

A `Camera` is a pinhole looking from `eye` at `target`:

```python
from holographic.rendering.holographic_render import Camera, Light
camera = Camera(eye=(3.2, 0.9, 4.0), target=(0, 0, 0), up=(0, 1, 0), fov_deg=42, aspect=1.0)
```
- `eye` / `target` — where you stand / what you look at. A ¾ view (offset in x, y and z) reads as more 3-D than
  a head-on shot.
- `fov_deg` — vertical field of view; smaller = more "zoomed"/telephoto, larger = wider/more dramatic.
- `aspect` — width/height; set it to `width/height` if you render non-square.

A `Light` is usually directional (the sun):

```python
sun = Light("directional", direction=(-0.5, -0.72, -0.48), color=(1.0, 0.96, 0.88), intensity=1.0)
```
`direction` is the way the light *travels* (so `(-0.5, -0.72, -0.48)` comes from the upper-right, heading down
and left). Warm, slightly-off-white sun colour plus a downward angle is what makes lighting feel natural.

---

## 5. Volumes in depth (smoke, fog, fire, custom density)

Any callable `field(points_Nx3) -> density_N (>=0)` can be volume-rendered. The engine ships two field builders:

| Builder | Look | Use for |
|---|---|---|
| `cloud_field(center, radius, ...)` | fBm carved by an envelope — puffy, wispy | **clouds** |
| `volumetric_field(center, radius, density, turbulence, seed)` | soft sphere + cheap turbulence | thin **smoke / fog** |

```python
from holographic.simulation_and_physics.holographic_semantic import volumetric_field
smoke = volumetric_field(center=(0, 0, 0), radius=1.0, density=1.6, turbulence=0.7, seed=1)
img, alpha = volume_render(smoke, camera, bounds, mode="smoke", sigma=14.0)
```

`volume_render` modes:
- `mode="smoke"` — grey/`albedo` absorption (add `self_shadow=True` for clouds).
- `mode="fire"` — emissive blackbody ramp (dark-red → orange → white as density rises); no lighting needed.
- `mode="density"` — raw density as luminance (debugging/scientific).

Key parameters: `sigma` (optical density — higher = thicker/more opaque), `steps` (march resolution — more =
smoother but slower), `albedo` (the medium's colour). Two production optimisations are **on by default** and
result-preserving: `empty_skip` (skip empty space) and `early_term` (stop opaque rays).

**Writing your own density** — a spherical blob, for reference:
```python
def blob(P):
    P = np.asarray(P, float)
    return np.clip(1.0 - np.linalg.norm(P, axis=1) / 0.6, 0, 1)
img, alpha = volume_render(blob, camera, bounds, mode="smoke", sigma=12.0)
```

---

## 6. Adaptive surface rendering (`render_auto`)

For SDF **surfaces**, the engine has an auto-calibrating path tracer: you give it a *quality target* and it
measures how many samples each pixel needs and how hard to denoise — no hand-set spp or denoise strength.

```python
from holographic.rendering.holographic_gbuffer import render_auto
img, stats = render_auto(scene_sdf, camera, width=256, height=256, material=mat,
                         quality="high", return_stats=True)   # 'draft' | 'medium' | 'high' | 'ultra'
```
The same call renders a sphere, glass, or a fractal to the same quality bar because it adapts per scene. `quality`
is the one knob (a target confidence-interval half-width; smaller = cleaner and slower — halving it costs ~4×
the samples). `return_stats=True` reports samples used, passes, and converged fraction.

Note the surface **scene** renderer (`render_scene`, used by `scene.render()`) is *also* adaptive in its
anti-aliasing: it finds edges in a cheap base pass and supersamples only there (`adaptive=True` by default), so
AA cost scales with edge length, not pixel count.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Cloud looks like a flat grey disc / white shell | self-shadowing off, **or** density/`sigma` so high light can't penetrate | pass `self_shadow=True`, and keep `sigma`~6–8 with moderate density so the interior is lit (`make_cloud`/`build_scene` are tuned) |
| Cloud looks like a smooth egg | single-ellipsoid density | use `cloud_field` (multi-lobe cumulus: flat base + eroded edges), not a hand-rolled sphere |
| Cloud render takes minutes | the fBm **bake**, not the render | lower `grid` (24 for previews); build the `field` once and reuse across frames |
| Cloud is too faint / too solid | optical density | adjust `sigma` (lower = thinner, higher = denser) |
| Volume is black | no light / wrong mode | pass a `Light`, use `mode="smoke"` with `self_shadow`, or `mode="fire"` for self-emissive |
| Nothing in frame | camera not pointed at it, or object outside `bounds` | check `Camera(target=...)` and that `bounds` enclose the field |
| `build_scene("a cloud")` makes two clouds | "fluffy"/"puffy" and "cloud" are both cloud words | describe with one cloud word, or de-duplicate `scene.objects` |
| Surface render noisy | too few samples | use `render_auto` with a higher `quality`, or `scene.render(quality="hyperreal")` |

---

## 8. Quick reference

```python
# --- the objects ---
import lecore
mind = lecore.UnifiedMind(dim=512, seed=0)

# --- clouds ---
mind.build_scene("a fluffy cloud").render()              # semantic, easiest
mind.make_cloud(radius=1.0, seed=0)                      # one call, low-level
cloud_field(center, radius, density=3.8, seed=0)         # the density (holographic_semantic)

# --- scenes (surfaces) ---
scene = mind.build_scene("a red metal sphere and a glass box")
scene.adjust("make the sphere gold")
scene.render(width=256, height=256, quality="fast")      # or "hyperreal"

# --- low-level rendering (holographic.rendering.holographic_render) ---
Camera(eye=(3,1,4), target=(0,0,0), fov_deg=42)
Light("directional", direction=(-0.5,-0.72,-0.48))
volume_render(field, camera, bounds, mode="smoke", sigma=8.0, self_shadow=True)   # volumes
save_png("out.png", img)                                 # img is (H,W,3) in [0,1]

# --- adaptive surface path tracer (holographic.rendering.holographic_gbuffer) ---
render_auto(sdf, camera, width, height, material, quality="high")
```

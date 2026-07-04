# holostuff -- render composability audit & backlog

*Grounded in a probe of the current repo. Your read is correct: composability is real in some places and
flat in others, and -- the good news -- the *substrate* to fix it is already built and strong; the texture
and material layers just aren't built on it yet. The unifying idea, very much in the engine's own spirit:
**everything you listed is the same thing -- a typed tree of nodes where a node is either a leaf (a number,
a color) or a composition of typed child inputs, with a schema saying which types may go where.** A texture
map, a layered material, a multi-material, a scene node, and a pipeline stage are five costumes of that one
structure. The engine already has that structure (`typed`, `template`, `planshape`); the job is to dress the
texture/material/scene layers in it. Readable-first: keep each graph a plain object tree you can read, and use
the hypervector encoding only where it earns its keep (search, blend, cache).*

---

## 1. The scorecard (what's composable today)

| Layer | State | What's there | What's missing |
|---|---|---|---|
| **Pipeline** | **STRONG** | `Stage` declares `needs`/`produces`; render methods are strategies; volume/particle compositing; adaptive picks the plan | it composes *passes*, but doesn't yet orchestrate the texture/material graphs below |
| **Scene** | **PARTIAL** | `scenegraph`: transform hierarchy, nested-transform composition, leaf instancing via `flatten_scene`; surface-vs-volumetric type awareness; tag grouping | schema-enforced **type-correct material binding** (surface->mesh, volumetric->volume); explicit **shared-definition instancing** (edit-once) |
| **Material** | **WEAK** | `Material.blend(other, t)` -- a 2-way linear channel mix | **layered** materials with a type-**order** (no reflection under diffuse); **multi-material** selected/blended by a map or field |
| **Texture** | **WEAK / MISSING** | textures are FPE functions over UV; source library in `texturehome` (fbm, voronoi, synth, weathering) | a composable **map graph**: a map whose inputs may be `map \| color \| field \| number`, with **child maps** |
| **Substrate** | **STRONG (unused here)** | `typed` (typed tree / encode-tree), `template` (recipes with typed **holes**), `planshape` (schema-guided typed structures), `scenegraph`, `compose` | nothing -- it just needs to be the thing textures/materials are built on |

So: the pipeline is genuinely composable (that's why it's adaptive), the scene is half-there, and the
material/texture layers are the real gap -- and the fix is not new machinery, it's building them on `typed` +
`template` + `planshape`.

---

## 2. The one model under all of it

A composable value is a **node** that is either a **leaf** or a **composition of typed inputs** -- the classic
expression/shader graph, which in this engine is `typed.encode_tree`. Keep the graph an explicit, readable
object tree; the VSA encoding is for the operations that benefit (cache the evaluated result, search a library,
blend two graphs), not something to force the whole tree into (that would hit the capacity cliff -- see the
boundaries).

```python
# The single shape under textures, materials, scene nodes, and pipeline stages.
class Leaf:                          # the base case: a number, a color, or a field
    def __init__(self, value):
        self.value = value
    def sample(self, uv):
        return self.value            # a constant returns itself; a field samples at uv

class Map:                           # a texture map: an op over TYPED inputs, each of which may be another Map
    def __init__(self, op, **inputs):
        self.op = op                 # 'mix' | 'noise' | 'multiply' | ...
        self.inputs = inputs         # name -> Leaf | Map | Field | number   (the typed inputs)
    def sample(self, uv):
        vals = {k: (v.sample(uv) if hasattr(v, "sample") else v)  # recursively evaluate child inputs
                for k, v in self.inputs.items()}
        return OPS[self.op](uv, vals)
# A Material is the same shape (an ORDERED list of typed layers); a Scene node is the same shape (typed
# children + a transform); a Pipeline stage is the same shape (needs/produces). One tree, four costumes.
```

The **type rules** -- "input must be map/color/field/number", "reflection layer can't sit under diffuse",
"a surface material binds to a mesh, a volumetric material to a volume" -- are a **schema**, which is exactly
what `planshape` is for. That is where "with respect to hierarchy" lives, and it's enforced at *compose* time,
not discovered at render time.

---

## 3. The backlog (build the graphs on the substrate)

### CMP1 (BUILD on `typed`/`template`). Texture map graph -- typed inputs + child maps
A `Map` node (the sketch above) whose inputs are `map | color | field | number`, recursively. Sampling
evaluates the tree. The input-type variant is a small **schema** (an input slot accepts those four kinds and
nothing else). *Reuses:* `texturehome` (the leaf sources: fbm, voronoi, synth), `fieldhome` (field inputs),
`template` (the typed holes = the inputs), `typed` (encode the graph when you want to cache/search it).
*Effort: medium.* This is the foundation everything else pulls from.

### CMP2 (BUILD on `typed`/`planshape`). Layered materials with a layer-order schema
An **ordered stack** of typed layers (base/diffuse -> specular/reflection -> coat/clearcoat), where a
`PlanShape` schema enforces the order so you *can't* put reflection under diffuse. Each layer combines with the
one below via a per-layer op (the existing `Material.blend` is the seed for the "over" mix). *Reuses:* `typed`
(the ordered tree), `planshape` (the order schema), `Material.blend`. *Effort: medium.* *Kept negative (loud):*
**ordering is necessary but not sufficient for physical correctness** -- true layered BRDFs conserve energy
across layers (a coat darkens/tints what's under it); the schema fixes the *stacking*, not the *radiometry*.
Ship correct ordering first; note the energy-conservation approximation honestly.

### CMP3 (PROMOTE + build). Multi-material selected/blended by a map or field
Generalize `Material.blend` (2-way, scalar `t`) to **N materials chosen/mixed by a map/field** (a mask):
`sum_i field_i(uv) * material_i`, the field being any CMP1 map. This is a `bundle` weighted by a field --
squarely on-substrate. *Reuses:* `Material.blend` (the per-channel mix), `fieldhome` + CMP1 (the selector).
*Effort: low-medium.* *Kept negative:* the weights should partition (sum ~1) or you get brightness drift --
normalize the mask.

### CMP4 (PROMOTE + build on `scenegraph`). Type-correct scene binding + shared-definition instancing
Two things: (1) a **scene schema** that binds a *surface* material only to a *mesh* and a *volumetric* material
only to a *volume* (the surface/volumetric type awareness already exists in `semantic`; make the binding
schema-checked, not conventional); (2) an explicit **Instance** node that references **one shared definition**
(mesh + material), so editing the definition updates every instance -- today `flatten_scene` instances by
transform and materializes copies, which works but isn't edit-once. *Reuses:* `scenegraph` (hierarchy +
transform instancing), `planshape` (the binding schema), the existing surface/volumetric split. *Effort:
medium.* *Kept negative:* `flatten_scene` still materializes at flatten time (it merges to one mesh); the shared
definition is at the *graph* level (edit-once), and flatten is where instances become geometry.

### CMP5 (PROMOTE on `pipeline`). Let the pipeline compose the graphs -- this is the rest of "adaptive"
The pipeline is already a `Stage` graph with `needs`/`produces`; extend it to orchestrate the graphs below as
stages: **bake a texture graph** to a grid (via the `Cache` home) when it's static, **resolve a layered
material** stack, **bind the scene** by the schema, then render. Then "adaptive" reaches all the way down: the
plan can decide to bake an expensive map graph vs sample it live, pick a material LOD, etc. *Reuses:*
`pipeline` (Stage/needs/produces), `Cache`/`bake_and_query` (bake an evaluated graph), CMP1-CMP4. *Effort:
medium.*

---

## 4. The type-hierarchy discipline (your "with respect to hierarchy", made concrete)

Every composable kind gets **one schema**, and that schema is the whole answer to "not overly associative,
type-correct":

- **Texture input schema** -- a slot accepts `map | color | field | number`, nothing else.
- **Material layer schema** -- an allowed order (base < diffuse < specular/reflection < coat), rejected if violated.
- **Scene binding schema** -- surface material <-> mesh, volumetric material <-> volume.
- **Pipeline stage schema** -- `needs`/`produces` (already there).

`planshape` is the mechanism for all four. Validate at **compose** time -- a bad graph is refused when you build
it, with a clear message, not rendered wrong. That is the difference between "composable" and "composable
*correctly*."

---

## 5. Honest boundaries (kept loud)

- **Keep the graph explicit; encode where it earns it.** A `Map`/`Layer`/`Node` object tree is readable Python
  and evaluates directly. Encode it to a hypervector for the ops that benefit -- caching an evaluated map,
  searching a material library, blending two graphs -- but **don't force the whole deep tree into one vector**:
  that hits the HRR capacity cliff (the `scenegraph` docstring already records "a node with very many children
  loses per-child" fidelity). Deep/wide compositions stay explicit trees; the vector is the encoded/cached form,
  factored by the resonator or tiled when you need the whole thing holographic.
- **Layer ordering != energy conservation** (CMP2) -- stacking order is a type rule; physically-correct layered
  BRDF radiometry is a separate, harder thing. Be honest which one you've shipped.
- **Multi-material masks should partition** (CMP3) or brightness drifts -- normalize.
- **Instancing is edit-once at the graph, materialized at flatten** (CMP4) -- not instanced *rendering*; say so.
- **Validate at compose time, not render time** -- the schema refuses a bad graph up front; a type error found
  mid-render is a bug, not a feature.

---

## 6. Sequencing

1. **CMP1 -- texture map graph** *(the foundation; maps feed materials, masks, displacement -- build it first).*
2. **CMP3 -- multi-material by field** *(cheap; reuses `blend` + CMP1's maps as the selector).*
3. **CMP2 -- layered materials + order schema** *(reuses `typed` + `planshape`; ship ordering, note radiometry).*
4. **CMP4 -- type-correct scene binding + shared-definition instancing** *(reuses `scenegraph` + a schema).*
5. **CMP5 -- pipeline composes the graphs** *(reuses the Stage pipeline; this is what makes "adaptive" reach all
   the way down to the maps and materials).*

The through-line is the project's own: the composition primitive already exists (`typed` + `template` +
`planshape`), and the render/material/texture layers should be **built on it** rather than each carrying its own
flat representation. A texture map is a typed tree; a material is a typed ordered tree; a scene is a typed tree
with transforms; a pipeline is a typed tree of stages -- one recursive, schema-checked composition, four
costumes. Generalize on contact, keep the trees readable, and let the schema keep the hierarchy honest.

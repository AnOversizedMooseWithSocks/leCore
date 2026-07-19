# SDF Cookbook

Signed distance fields in leCore, from the integrator's point of view. Every snippet here was run against the
live engine; the gotchas are the ones that cost an afternoon if you have to rediscover them from source.

The one-sentence mental model: **constructors are module functions, combinators are methods on the object they
return, and the object is a callable distance field.** Build a tree by chaining, then hand it to a field consumer.

```python
import lecore
import holographic.mesh_and_geometry.holographic_sdf as sdf

m = lecore.UnifiedMind(dim=64, seed=0)

shape = (sdf.sphere(0.6)
         .smooth_union(sdf.box(0.4, 0.4, 0.4).translate((0.3, 0.0, 0.0)), k=0.2))

d = shape([[0.0, 0.0, 0.0]])                       # evaluate: negative inside, positive outside
mesh = m.mesh_from_sdf(shape, ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)), res=48)
```

## Constructors are module functions

Primitives live on the module, not as methods. Each returns an `SDF` you then combine:

```python
sdf.sphere(r=1.0)
sdf.box(bx=1.0, by=1.0, bz=1.0)          # THREE scalars, not one tuple -- box(0.5, 0.5, 0.5)
sdf.cylinder(h=1.0, r=0.5)
sdf.torus(R=1.0, r=0.3)                    # R = ring radius, r = tube radius
sdf.plane(h=0.0)
sdf.capsule(h=1.0, r=0.3)
sdf.cone(h=1.0, r=0.5)
```

The easy mistake is `sdf.box((0.5, 0.5, 0.5))` — `box` takes three separate arguments, so that passes a tuple as
`bx` and raises. Use `sdf.box(0.5, 0.5, 0.5)`.

## Combinators are methods on the SDF

Booleans, transforms, and deformations are methods on the object, so they chain left to right. `smooth_union`
takes a blend radius `k`; `rotate` takes an **axis vector and an angle** (not Euler angles):

```python
a.union(b)                 a.intersect(b)              a.subtract(b)
a.smooth_union(b, k=0.3)                                # k is the blend width in world units
a.translate((tx, ty, tz))  a.scale(s)                  a.rotate((0, 1, 0), angle)   # axis, then radians
a.repeat((px, py, pz))     a.rounded(r)                a.onion(thickness)
a.twist(k)                 a.displace(amount, freq)    a.elongate(hx, hy, hz)
```

So a rounded, twisted, subtracted tree reads as one chain:

```python
part = (sdf.box(1, 1, 1).rounded(0.1)
        .subtract(sdf.cylinder(2.0, 0.3))
        .twist(0.5))
```

`rotate((0, 1, 0), angle)` is the one people trip on: the first argument is the axis to spin around, the second
is the angle in radians. `rotate(angle)` alone will not work — there is no default axis.

## The SDF is a callable distance field

`shape.eval(P)` and `shape(P)` are identical — an `SDF` is callable, so any consumer that wants a
`func(points) -> distances` takes the object directly:

```python
shape([[0, 0, 0], [1, 0, 0]])              # -> array of signed distances, one per point
m.mesh_from_sdf(shape, bounds, res=48)     # bounds = ((minx,miny,minz), (maxx,maxy,maxz))
```

`mesh_from_sdf`'s signature is `mesh_from_sdf(sdf, bounds, res=24, level=0.0, vectorized=False)`. `bounds` is a
pair of corner points, `res` is the grid resolution per axis, `level` is the iso-level (0.0 = the surface).
Higher `res` is finer and slower (it samples `res^3` points).

## The NodeGraph route (data-driven, serializable)

For a saved / editable graph rather than a Python expression, build a `NodeGraph`. The socket names are the
non-obvious part: **single-input SDF nodes name their input `'a'`; two-input nodes use `'a'` and `'b'`; every
SDF node's output socket is `'out'`.**

```python
g = m.node_graph()
a = g.add('sdf_sphere', {'radius': 1.0})
b = g.add('sdf_box')
u = g.add('sdf_smooth_union', {'k': 0.2})
g.connect(a, 'out', u, 'a')                # source node, source socket, dest node, dest socket
g.connect(b, 'out', u, 'b')
```

A transform node (`sdf_translate`, `sdf_rotate`, `sdf_scale`, `sdf_repeat`, `sdf_twist`, …) has a single input
socket `'a'`, so you wire the upstream shape into `'a'`, not `'in'` or `'0'`:

```python
t = g.add('sdf_translate', {'t': (0.3, 0, 0)})
g.connect(b, 'out', t, 'a')
```

`NodeGraph.remove(id)` deletes a node and its wires; `NodeGraph.collapse([ids])` contracts a selection into one
reusable subgraph node (and `expand(id)` reverses it).

## Analytic vs. meshed: which representation you hold

A common integration question: "if I run mesh operations, do I keep the exact SDF?" The answer is a deliberate
architectural choice, so it is worth stating plainly.

**A `Mesh` never carries an SDF.** It is a pure geometric snapshot — vertices, faces, uvs, normals — and has no
`sdf_tree` or analytic field to lose. So no mesh verb can "silently drop" the analytic form; there was never one
attached to drop. The analytic tree lives on the SDF object you built and, for a full scene, on the scene/session
(`session.sdf`, reachable via `sdf_tree()`), which by convention keeps its tree reachable for the renderer.

The practical consequence: **to stay exact, keep the SDF and re-mesh from it; do not expect a mesh to round-trip
back to analytic.** `mesh_from_sdf(shape, ...)` samples the field into geometry — a one-way projection. Going the
other way, `mesh_to_sdf(mesh, points)` and `mesh_to_sdf_grid(mesh, bounds)` build a *sampled* SDF from a mesh
(distances to the surface), which is an approximation, not a recovery of the original tree.

```python
shape = sdf.sphere(0.6).smooth_union(sdf.box(0.4, 0.4, 0.4), k=0.2)   # analytic: exact everywhere
mesh  = m.mesh_from_sdf(shape, ((-1.5,)*3, (1.5,)*3), res=64)         # meshed: an approximation at this res
# keep `shape` if you need exactness later -- re-mesh at a higher res, boolean more shapes onto it, etc.
# `mesh` alone cannot reconstruct `shape`; it is geometry, not the field.
```

So there is no per-verb "preserves analytic: yes/no" flag because the two representations are simply held in
different objects. Track exactness by tracking which object you are holding: an `SDF` (exact) or a `Mesh`
(sampled). If a workflow needs both, carry the `SDF` alongside the `Mesh` it produced.

## Gotchas, collected

- `box` takes three scalars, not a tuple.
- `rotate` takes `(axis_vector, angle)`, angle in radians. No default axis.
- Constructors are `sdf.sphere(...)` (module functions); combinators are `shape.union(...)` (methods). Mixing
  the two up — looking for `sdf.union(a, b)` or `shape.sphere()` — is the usual first wrong guess.
- `mesh_from_sdf` wants `bounds` as a corner pair and `res`, not `resolution`.
- NodeGraph SDF sockets: inputs `'a'` (and `'b'` for booleans), output `'out'`.
- The SDF object is callable, so you never need to pass `.eval` explicitly — but `.eval(P)` still works if you
  prefer to be explicit.

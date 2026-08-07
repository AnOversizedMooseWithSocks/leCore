# SDF COOKBOOK -- every constructor, every combinator, one worked scene

**Generated from the live module by `tools/gen_sdf_cookbook.py` -- signatures cannot drift.**

The convention, stated once (client P-3): **constructors are module functions** returning an
`SDF` node (`sdf.sphere(0.5)`, `sdf.box(0.4, 0.4, 0.4)` -- three scalars, NOT a tuple);
**combinators and transforms are methods on the node** (`node.union(other)`,
`node.translate((x, y, z))`). Everything returns another `SDF`, so chains read like math and
`result.preserves_analytic` is True at every step (see PACKAGING.md, the analytic contract).

## Constructors (module functions)

- **`sdf.box(bx=1.0, by=1.0, bz=1.0)`** -- An axis-aligned box with half-extents (bx, by, bz) centred at the origin -- so the box spans [-bx, bx] on x,
- **`sdf.capsule(h=1.0, r=0.3)`** -- A capsule (a cylinder with hemispherical caps) along Y: segment from -h to +h on the Y axis, radius `r`.
- **`sdf.cone(h=1.0, r=0.5)`** -- A capped cone along Y: height `h` (apex at +h/2, base at -h/2), base radius `r`. iq's exact cone distance
- **`sdf.cylinder(h=1.0, r=0.5)`** -- A capped cylinder of half-height `h` and radius `r`, axis along Y, centred at the origin. Returns an SDF.
- **`sdf.ellipsoid(ax=1.0, ay=0.7, az=0.5)`** -- An ellipsoid with semi-axes (`ax`,`ay`,`az`). Uses iq's BOUNDED APPROXIMATION k1*(k1-1)/k2 -- the ellipsoid
- **`sdf.fold_fractal(iterations=12, scale=2.0, min_radius=0.5, fold_limit=1.0)`** -- The KALEIDOSCOPIC-IFS / MANDELBOX distance-estimator SDF -- the general 'fold engine' behind the fractal-forums
- **`sdf.mandelbulb(power=8.0, iterations=8, bailout=2.0)`** -- The MANDELBULB distance-estimator SDF (White & Nylander's polar-power fractal, the 3D Mandelbrot analogue).
- **`sdf.menger(iterations=3, size=1.0)`** -- The Menger sponge: the classic recursive fractal cube, carved `iterations` deep at the given `size`. Returns an
- **`sdf.octahedron(s=1.0)`** -- A regular octahedron of 'radius' `s` (vertex distance along each axis). iq's exact octahedron distance.
- **`sdf.plane(h=0.0)`** -- An infinite ground plane at height y = `h` (points above are outside). Returns an SDF -- handy as a floor.
- **`sdf.sphere(r=1.0)`** -- A sphere of radius `r`, centred at the origin. Returns an SDF you can transform (translate/rotate/scale) and
- **`sdf.torus(R=1.0, r=0.3)`** -- A torus in the XZ plane: `R` is the ring radius (centre to tube centre), `r` the tube radius. Returns an SDF.

## Combinators & transforms (methods on the node)

- **`node.bend(k, axis=0)`** -- Bend space by `k` radians per unit along `axis` (iq's opCheapBend) -- curl a straight beam into an arc.
- **`node.cost()`** -- Estimate the per-ray evaluation COST of this SDF tree (W2) -- a machine-model annotation for deciding
- **`node.displace(amount, freq)`** -- (see module)
- **`node.elongate(hx=0.0, hy=0.0, hz=0.0)`** -- Stretch this shape by pulling it apart along the axes by half-extents (`hx`,`hy`,`hz`) -- iq's
- **`node.fillet_union(other, r=0.1)`** -- (see module)
- **`node.fold(plane=0.0)`** -- Mirror all three axes about `plane` -- map the world into one octant (an 8-fold kaleidoscope). Composes
- **`node.intersect(other)`** -- (see module)
- **`node.mirror(axis=0, plane=0.0)`** -- Fold space across a plane on one axis (kaleidoscopic symmetry from abs()). A DSL-tree node so it
- **`node.onion(thickness)`** -- (see module)
- **`node.repeat(period)`** -- (see module)
- **`node.rotate(axis, angle)`** -- (see module)
- **`node.rounded(r)`** -- (see module)
- **`node.scale(s)`** -- (see module)
- **`node.smooth_union(other, k=0.3)`** -- (see module)
- **`node.subtract(other)`** -- (see module)
- **`node.to_dsl()`** -- A compact s-expression: (kind p0 p1 ... child0 child1 ...). Round-trips via parse_dsl.
- **`node.to_glsl(name='map', camera='fixed')`** -- Emit a complete Shadertoy-ready fragment shader for this SDF (see _emit_shader). camera="fixed" (default)
- **`node.to_jit_expr()`** -- Emit this tree as a SINGLE symbolic expression string in (x, y, z) -- the `jit_expr=`
- **`node.to_tree()`** -- A nested tuple where the op name folds in the params (e.g. 'sphere(1.0)') so a leaf/op is a
- **`node.translate(t)`** -- (see module)
- **`node.twist(k)`** -- (see module)
- **`node.union(other)`** -- (see module)

## Emitters (also methods)

- **`node.to_glsl(name="map")`** -- a complete GLSL distance function for shaders.
- **`node.to_jit_expr()`** -- the single symbolic expression `render_sdf(..., jit_expr=...)`
  compiles (~9-15x). Exact kinds only; bound-only kinds (twist/bend/ellipsoid/fractals) and
  branchy kinds (octahedron, menger) refuse with the reason -- a shader that disagrees with the
  numpy field is worse than no shader.

## Worked example (EXECUTED by the generator before this page is written -- it cannot rot)

```python
import numpy as np
from holographic.mesh_and_geometry.holographic_sdf import sphere, box, cylinder, plane
from holographic.rendering.holographic_render import Camera
from holographic.rendering.holographic_raymarch import render_sdf

# a tabletop scene: rounded box, a sphere resting on it, a hole drilled through, on a floor
body  = box(0.5, 0.25, 0.35).rounded(0.05)
ball  = sphere(0.22).translate((0.0, 0.47, 0.0))
hole  = cylinder(1.0, 0.12).rotate((1, 0, 0), 1.5707963)
scene = body.union(ball).subtract(hole).union(plane(-0.25))

cam = Camera(eye=(1.6, 1.1, 2.2), target=(0, 0.1, 0), fov_deg=45)
img = render_sdf(scene, cam, 96, 96, ao=True, shadows=True, reflect=0.2)
assert img.shape == (96, 96, 3)

d = scene.eval(np.array([[0.0, 0.47, 0.0]]))     # inside the resting ball -> negative
assert d[0] < 0
```

#!/usr/bin/env python3
"""Regenerate docs/SDF_COOKBOOK.md from the live SDF module (client P-3). Signatures are
introspected -- the page cannot drift -- and the worked example is EXECUTED before the file is
written: a page whose example fails does not ship. Run: PYTHONPATH=. python3 tools/gen_sdf_cookbook.py"""
import inspect
import textwrap

import numpy as np

import holographic.mesh_and_geometry.holographic_sdf as S

EXAMPLE = '''
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
'''


def main():
    cons = [(n, str(inspect.signature(getattr(S, n))),
             ((getattr(S, n).__doc__ or "").split("\n")[0]).strip())
            for n in dir(S)
            if not n.startswith("_") and inspect.isfunction(getattr(S, n))
            and getattr(S, n).__module__ == S.__name__ and n in S.ARITY]
    methods = [(n, str(inspect.signature(f)).replace("(self, ", "(").replace("(self)", "()"),
                (inspect.getdoc(f) or "").split("\n")[0].strip())
               for n, f in inspect.getmembers(S.SDF, predicate=inspect.isfunction)
               if not n.startswith("_") and n != "eval"]

    exec(compile(textwrap.dedent(EXAMPLE), "<cookbook-example>", "exec"), {})  # rot gate FIRST

    L = []
    L.append("# SDF COOKBOOK -- every constructor, every combinator, one worked scene\n")
    L.append("**Generated from the live module by `tools/gen_sdf_cookbook.py` -- signatures cannot drift.**\n")
    L.append("The convention, stated once (client P-3): **constructors are module functions** returning an")
    L.append("`SDF` node (`sdf.sphere(0.5)`, `sdf.box(0.4, 0.4, 0.4)` -- three scalars, NOT a tuple);")
    L.append("**combinators and transforms are methods on the node** (`node.union(other)`,")
    L.append("`node.translate((x, y, z))`). Everything returns another `SDF`, so chains read like math and")
    L.append("`result.preserves_analytic` is True at every step (see PACKAGING.md, the analytic contract).\n")
    L.append("## Constructors (module functions)\n")
    for name, sig, doc in sorted(cons):
        L.append("- **`sdf.%s%s`** -- %s" % (name, sig, doc or "(see module)"))
    L.append("\n## Combinators & transforms (methods on the node)\n")
    for name, sig, doc in sorted(methods):
        L.append("- **`node.%s%s`** -- %s" % (name, sig, doc or "(see module)"))
    L.append("""
## Emitters (also methods)

- **`node.to_glsl(name="map")`** -- a complete GLSL distance function for shaders.
- **`node.to_jit_expr()`** -- the single symbolic expression `render_sdf(..., jit_expr=...)`
  compiles (~9-15x). Exact kinds only; bound-only kinds (twist/bend/ellipsoid/fractals) and
  branchy kinds (octahedron, menger) refuse with the reason -- a shader that disagrees with the
  numpy field is worse than no shader.

## Worked example (EXECUTED by the generator before this page is written -- it cannot rot)

```python""" + EXAMPLE + "```\n")
    open("docs/SDF_COOKBOOK.md", "w").write("\n".join(L))
    print("wrote docs/SDF_COOKBOOK.md -- %d constructors, %d methods, example executed" % (
        len(cons), len(methods)))


if __name__ == "__main__":
    main()

"""holographic_scatterlayer.py -- SURFACE SCATTER LAYER (fluids/matter backlog item 5, part 1).

`procgen.scatter_on_terrain` already scatters instances on a terrain; `emit_from_surface` does the same on ANY
surface (points + normals + a weight map). So a scatter layer is just that emission, wired as a surface MATERIAL
layer -- a channel that emits GEOMETRY instead of colour, and the write-dual of the sampler (same surface sampling,
opposite direction). Grass on a hill, barnacles on a hull, craters on a moon: one layer, any surface.

Holographically each placement is a BIND (an instance bound to a position/region code) and the whole layer is a
BUNDLE (a superposition of those placements) -- so the scattered field is content-addressable: you can ask "what's
scattered near here?" by unbinding the region code. And a scatter layer BAKES: generate the instances once, cache
them, and LOD them (a distant grass field becomes one card, not a million blades) -- which is exactly the ScaleNode's
job in part 2.

KEPT NEGATIVE: the region query is a coarse cell hash (region-addressable, not exact per-instance recall); dense
scatter needs the bake+LOD path (part 2 / the performance half) to stay affordable -- the layer itself just emits.
Deterministic (seeded emission + hashed cell codes).
"""
import hashlib

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, unbind, bundle, cosine
from holographic.simulation_and_physics.holographic_emitter import emit_from_surface


def _cell_code(point, dim, cell_size, seed=0):
    """A deterministic unit vector for the grid CELL containing `point` -- so nearby placements share a code and the
    bundle is region-addressable. Uses hashlib (not Python hash) for determinism, per the engine's rule."""
    cell = tuple(int(np.floor(c / cell_size)) for c in np.asarray(point, float))
    h = hashlib.sha256(("%d:%s" % (seed, cell)).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.standard_normal(dim)
    return v / (np.linalg.norm(v) + 1e-12)


class ScatterLayer:
    """A surface layer that EMITS geometry: sample points on a surface (weighted by a density map), place an instance
    at each (aligned to the normal), and bundle them into one content-addressable layer vector."""

    def __init__(self, instance, count, scale=1.0, density=None, align="normal", cell_size=0.25, seed=0):
        self.instance = np.asarray(instance, float)          # the hypervector to scatter (dim D)
        self.dim = self.instance.shape[0]
        self.count = int(count)
        self.scale = float(scale)
        self.density = density                               # a weight map (callable P->weights) or None = uniform
        self.align = align
        self.cell_size = float(cell_size)
        self.seed = int(seed)

    def apply(self, sdf_eval, bounds):
        """Scatter onto the surface of `sdf_eval` within `bounds`=(lo,hi). Returns the placements (point, normal,
        scale, and the per-placement bound vector) plus the bundled `layer` vector. Reuses emit_from_surface.
        `sdf_eval` may be a raw callable P->distance or an SDF object (its .eval is used)."""
        fn = sdf_eval.eval if hasattr(sdf_eval, "eval") else sdf_eval
        pts, normals, _ = emit_from_surface(fn, self.count, bounds, weight=self.density, seed=self.seed)
        placements = []
        vecs = []
        for p, n in zip(pts, normals):
            code = _cell_code(p, self.dim, self.cell_size, self.seed)
            vec = bind(self.instance, code)                  # a placement = instance bound to its region code
            placements.append({"pos": p, "normal": n, "scale": self.scale, "vec": vec})
            vecs.append(vec)
        layer = bundle(vecs) if vecs else np.zeros(self.dim)  # the layer = a bundle (superposition of placements)
        return {"points": pts, "normals": normals, "placements": placements, "layer": layer, "count": len(pts)}

    def recall_region(self, layer, point):
        """Ask the bundled layer "is an instance scattered near `point`?" -- unbind the region's cell code and read
        its cosine to the instance. High = yes, low = no. The content-addressable read the bundle buys."""
        code = _cell_code(point, self.dim, self.cell_size, self.seed)
        return float(cosine(unbind(layer, code), self.instance))


def _selftest():
    """Scatter a grass instance on a sphere and a box (any surface), check the points land ON the surface with unit
    normals, that a density map biases placement, and that the bundled layer answers a region query."""
    from holographic.mesh_and_geometry.holographic_sdf import sphere, box
    from holographic.agents_and_reasoning.holographic_ai import Vocabulary

    voc = Vocabulary(512, seed=0)
    grass = voc.get("grass")

    for name, sdf, bnds in [("sphere", sphere(1.0), ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5))),
                            ("box", box(1.0, 0.6, 0.8), ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)))]:
        layer = ScatterLayer(grass, count=60, scale=0.1, seed=0)
        res = layer.apply(sdf, bnds)
        assert res["count"] > 20, (name, res["count"])                # scattered a decent number
        d = np.abs(np.asarray([sdf.eval(p[None, :])[0] for p in res["points"]]))
        assert d.max() < 0.05, (name, float(d.max()))                 # every instance is ON the surface
        norms = np.linalg.norm(res["normals"], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), name               # unit normals
        # the bundled layer answers "is grass near a scattered point?" -- the near cell reads clearly above a far one
        # (region-addressable), though a single unbind is crosstalk-limited to ~1/sqrt(N) for a dense bundle
        near = layer.recall_region(res["layer"], res["points"][0])
        far = layer.recall_region(res["layer"], np.array([100.0, 100.0, 100.0]))
        assert near > far + 0.05 and near > 0.0, (name, near, far)

    # a density map that only allows the top hemisphere (y>0) puts (almost) all instances up top
    top_only = lambda P: (np.asarray(P)[:, 1] > 0).astype(float)
    layer = ScatterLayer(grass, count=120, density=top_only, seed=1)
    res = layer.apply(sphere(1.0), ((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)))
    ys = res["points"][:, 1]
    assert (ys > 0).mean() > 0.9, float((ys > 0).mean())              # density map steered the scatter

    print("holographic_scatterlayer selftest OK: grass scattered on a sphere AND a box (any surface via "
          "emit_from_surface) -- all instances ON the surface (|sdf|<0.05) with unit normals; the bundled layer is "
          "region-queryable (near %.2f > far); a top-hemisphere density map put >90%% of instances up top" % near)


if __name__ == "__main__":
    _selftest()

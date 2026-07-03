"""Region fields (REGIONS): a boundary that says how to REGARD what is inside it -- as a material, or a behaviour, or a
biome -- and that composes with other boundaries by priority. This is the one primitive under "treat anything as
mesh / particle / smoke / fluid / light": at the superposition layer it is all a field of vectors; a region is a
boundary (an SDF) plus a LABEL that tells the pipeline how to interpret the points it contains, and a PRIORITY that
resolves overlaps. Everything downstream -- which material to shade, which simulation to run, whether a point is even
there to be processed -- reads from the same classification.

WHY THIS UNIFIES SO MUCH
------------------------
  * MATERIAL by region: everything inside a boundary shades as its material -- and because regions LAYER by priority,
    slicing the volume open shows the layers (crust over mantle over core), no special case.
  * BEHAVIOUR by region: the label can be "cloth" / "fire" / "smoke" / "fluid" instead of a colour, so the SAME
    classification that picks a material picks a SIMULATION. A cloth-on-fire disintegrating into smoke is three region
    labels over one field, with the boundaries changing over time -- not three separate engines bolted together.
  * CULLING is trivial and PRECISE: a point outside every region is known-empty (the SDFs say so up front), so it is
    skipped without marching -- the same "empty space is known, not discovered" property the density/radiance fields
    have. No occlusion queries, no guesswork: classify < 0 means "not there".

WHAT THIS IS AND IS NOT (kept honest)
-------------------------------------
  This is the composable SUBSTRATE -- the labelled-region algebra and its classify / slice / cull. It is NOT the
  simulations themselves (a real cloth solver, a real combustion model): those are the APPLICATION layer that reads a
  region's behaviour label and runs. What this earns is that they compose over ONE field with ONE classification,
  instead of each being a siloed pipeline. Deterministic, NumPy/stdlib only, O(regions) per point, vectorised.
"""
import numpy as np


def _resolve_color(mat, points):
    """Resolve a region's albedo to per-point (M,3): a constant rgb is tiled; a callable / Param(field) is evaluated
    (a texture). Kept small and local -- the general resolve_param handles scalars; colour is the vector case."""
    if callable(mat):
        rgb = np.asarray(mat(points), float)
    elif hasattr(mat, "resolve"):                                 # a Param socket
        if mat.field is not None:
            rgb = np.asarray(mat.field(points), float)
        else:
            rgb = np.asarray(mat.value, float)
    else:
        rgb = np.asarray(mat, float)
    if rgb.ndim == 1:                                             # a single rgb -> tile to every point
        rgb = np.tile(rgb, (len(points), 1))
    return rgb


class Region:
    """A boundary (SDF, negative inside) plus how to REGARD what it contains: a label, a priority (higher wins on
    overlap), an optional material (rgb) for rendering, an optional behaviour tag for simulation, and free-form data."""

    def __init__(self, sdf, label, priority=0.0, material=None, behavior=None, data=None, reflect=None, roughness=None):
        self.sdf = sdf; self.label = label; self.priority = float(priority)
        # material may be a constant rgb OR a socket (callable / Param) for a textured albedo -- keep non-constants as-is
        if material is None or callable(material) or hasattr(material, "resolve"):
            self.material = material
        else:
            self.material = np.asarray(material, float)
        self.behavior = behavior; self.data = data or {}
        self.reflect = reflect                                   # optional per-region reflectivity (0..1), for the shader
        self.roughness = roughness                              # optional per-region glossy lobe angle


class RegionField:
    """A composable set of labelled regions. `classify(points)` returns, per point, the index of the highest-priority
    region containing it (or -1 = empty). That one call drives material lookup, behaviour lookup, and culling."""

    def __init__(self, regions):
        self.regions = list(regions)

    def classify(self, points):
        """Per-point winning region index (-1 = outside every region). Highest priority wins where regions overlap --
        so an inner core (high priority) shows through the shells (low priority) around it. Vectorised, O(regions)."""
        P = np.atleast_2d(np.asarray(points, float))
        best = -np.ones(len(P), int); best_pri = np.full(len(P), -np.inf)
        for i, reg in enumerate(self.regions):
            inside = reg.sdf.eval(P) < 0.0                      # SDF is negative inside the boundary
            take = inside & (reg.priority > best_pri)
            best[take] = i; best_pri[take] = reg.priority
        return best

    def cull(self, points):
        """Boolean mask of points that are actually THERE (inside some region). Points outside every region are
        known-empty and can be skipped with no marching -- precise culling for free from the boundaries themselves."""
        return self.classify(points) >= 0

    def material_at(self, points, empty=(0.0, 0.0, 0.0)):
        """Per-point material rgb from the winning region (empty colour where a point is outside every region). The
        region's colour is resolved through the parameter socket, so albedo can be a constant rgb OR a field / callable
        returning per-point (M,3) -- a TEXTURE across the region, consistent with reflect/roughness now taking maps."""
        idx = self.classify(points)
        out = np.tile(np.asarray(empty, float), (len(idx), 1))
        for i, reg in enumerate(self.regions):
            m = idx == i
            if np.any(m) and reg.material is not None:
                out[m] = _resolve_color(reg.material, points[m])
        return out

    def behavior_at(self, points):
        """Per-point behaviour tag from the winning region (None where empty) -- which simulation a point belongs to."""
        idx = self.classify(points)
        return [self.regions[i].behavior if i >= 0 else None for i in idx]

    def _scalar_at(self, points, attr, default):
        """Per-point scalar (reflect / roughness) from the winning region, `default` where a region doesn't set it.
        The region's value is resolved through the parameter socket, so reflect/roughness can be a bare number OR a
        map / field / wired output -- e.g. a roughness TEXTURE that varies across the region, like any DCC material."""
        from holographic_param import resolve_param
        idx = self.classify(points)
        out = np.full(len(idx), float(default))
        for i, reg in enumerate(self.regions):
            val = getattr(reg, attr)
            if val is not None:
                m = idx == i
                if np.any(m):
                    out[m] = resolve_param(val, points[m])       # constant -> itself; map/field -> sampled per point
        return out

    def reflect_at(self, points, default=0.0):
        """Per-point reflectivity from the winning region -- lets ONE object be mirror here, matte there (the shader
        reads this to give a single body multiple materials)."""
        return self._scalar_at(points, "reflect", default)

    def roughness_at(self, points, default=0.0):
        """Per-point glossy-lobe roughness from the winning region."""
        return self._scalar_at(points, "roughness", default)

    def slice(self, origin, u, v, extent=2.0, res=200):
        """Cut the volume with a plane (origin + s*u + t*v, s,t in [-extent, extent]) and return an (res, res) image of
        the region MATERIAL at each point -- i.e. slice it open and SEE THE LAYERS. Also returns the label-index grid."""
        u = np.asarray(u, float); u = u / (np.linalg.norm(u) + 1e-12)
        v = np.asarray(v, float); v = v / (np.linalg.norm(v) + 1e-12)
        ts = np.linspace(-extent, extent, res)
        S, T = np.meshgrid(ts, ts)
        pts = np.asarray(origin, float) + S.reshape(-1, 1) * u + T.reshape(-1, 1) * v
        idx = self.classify(pts)
        img = self.material_at(pts).reshape(res, res, 3)
        return img, idx.reshape(res, res)


def layered_sphere(center=(0, 0, 0), radii_labels_materials=None):
    """Convenience: nested spheres as priority-ordered shells (innermost highest priority) -- the canonical
    'slice reveals layers' object (a planet: core / mantle / crust). Each entry is (radius, label, rgb)."""
    from holographic_semantic import _SphereSDF
    radii_labels_materials = radii_labels_materials or [
        (1.0, "crust", (0.45, 0.32, 0.20)), (0.7, "mantle", (0.80, 0.35, 0.12)), (0.35, "core", (0.98, 0.85, 0.30))]
    regions = []
    for pri, (r, label, mat) in enumerate(sorted(radii_labels_materials, key=lambda e: -e[0])):
        regions.append(Region(_SphereSDF(center, r), label, priority=pri, material=mat))
    return RegionField(regions)


def _selftest():
    """Classify resolves layers by priority; a slice through a layered sphere shows concentric materials; culling marks
    the empty exterior."""
    field = layered_sphere()
    # a point at the centre is core; at r=0.5 is mantle; at r=0.9 is crust; at r=2 is empty
    pts = np.array([[0, 0, 0], [0.5, 0, 0], [0.9, 0, 0], [2.0, 0, 0]])
    labels = [field.regions[i].label if i >= 0 else None for i in field.classify(pts)]
    assert labels == ["core", "mantle", "crust", None], labels
    img, idx = field.slice((0, 0, 0), (1, 0, 0), (0, 1, 0), extent=1.5, res=64)
    n_layers = len(set(idx[idx >= 0].tolist()))
    assert n_layers == 3                                        # the cut shows all three layers
    keep = field.cull(np.array([[0, 0, 0], [3, 0, 0]]))
    assert keep.tolist() == [True, False]                      # interior kept, exterior culled
    print("regionfield selftest ok: layers %s, slice shows %d materials, culling precise" % (labels[:3], n_layers))


if __name__ == "__main__":
    _selftest()

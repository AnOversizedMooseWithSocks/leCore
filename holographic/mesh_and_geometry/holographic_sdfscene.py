"""holographic_sdfscene.py -- a small, documented base class for "a scene is a set of SDF parts".

WHY THIS EXISTS. Every demo that ray-marches a custom scene ends up hand-writing the same three methods:
`.eval` (nearest-surface distance = min over the parts), `.part_ids` (which part owns a point, for material
lookup), and `.ids` (the SAME thing under the name the splat exporter/`to_splats` happens to expect). You only
discover that contract by reverse-engineering the exporters. This base class writes that contract down ONCE:
subclass it, implement `parts()`, and you get `.eval`, `.part_ids`, and `.ids` for free.

SDF SIGN CONVENTION (stated once, load-bearing): each part's sdf_fn returns NEGATIVE inside the surface,
ZERO on it, POSITIVE outside -- the engine-wide convention. `eval` returns the min over parts, so the scene's
own zero-level-set is the union of the parts.
"""

import numpy as np


class SDFScene:
    """Subclass and implement `parts()` -> list of (sdf_fn, material_name).

    Each `sdf_fn` takes an (N, 3) array of points and returns an (N,) array of signed distances (negative
    inside). You then get, for free:

        .eval(P)      -> (N,) nearest-surface distance = min over parts        (what the ray-marcher calls)
        .part_ids(P)  -> (N,) index of the owning part = argmin over parts     (for material lookup)
        .ids(P)       -> alias of .part_ids -- the name to_splats / exporters expect
        .material_at(P) -> (N,) the material_name of the owning part           (convenience)

    That is the whole required surface. The naive min/argmin below is O(parts) per query and is the readable,
    ALWAYS-CORRECT default. For a scene with many parts, override `bounds()` and use `parts_near()` to cull --
    see those methods. (Kept honest: automatic broadphase for an SDF *min* needs per-part bounds AND an
    upper-bound pass to prune safely, so it is opt-in, not silently baked into `eval`.)"""

    def parts(self):
        """Return a list of (sdf_fn, material_name). Override this -- it is the only required method."""
        raise NotImplementedError("An SDFScene subclass must implement parts() -> [(sdf_fn, material), ...]")

    @classmethod
    def from_parts(cls, parts, bounds=None):
        """Build an SDFScene DIRECTLY from a parts list, without writing a subclass -- the ergonomic door for a
        caller who just has some (sdf_fn, material) pairs and (optionally) their bounding spheres. `parts` is a list
        of (sdf_fn, material_name); `bounds`, if given, is a matching list of (center_xyz, radius) so `parts_near`
        culling works. Returns a ready SDFScene with all the free methods (eval / part_ids / material_at). This is
        why the base class exists: compose an SDF scene from parts the same way a splat scene bundles primitives."""
        _parts = list(parts)
        _bounds = list(bounds) if bounds is not None else None

        class _FromParts(cls):
            def parts(self):
                return _parts

            def bounds(self):
                if _bounds is None:
                    return super().bounds()
                return _bounds

        return _FromParts()

    # -- cached views of parts() so a subclass's parts() can build lazily and we don't re-call it per query ----

    def _parts(self):
        if not hasattr(self, "_parts_cache"):
            self._parts_cache = list(self.parts())
        return self._parts_cache

    def _fns(self):
        if not hasattr(self, "_fns_cache"):
            self._fns_cache = [fn for fn, _ in self._parts()]
        return self._fns_cache

    def _materials(self):
        if not hasattr(self, "_mat_cache"):
            self._mat_cache = [mat for _, mat in self._parts()]
        return self._mat_cache

    # -- the required contract ---------------------------------------------------------------------------------

    def _stack(self, P):
        """Evaluate every part at points P and stack into (N, num_parts). The one place that touches all parts."""
        P = np.atleast_2d(np.asarray(P, float))
        fns = self._fns()
        if not fns:                                          # an empty scene: everything is "far outside"
            return P, np.full((P.shape[0], 0), np.inf)
        return P, np.stack([np.asarray(fn(P), float) for fn in fns], axis=1)

    def eval(self, P):
        """Nearest-surface signed distance at each point = min over parts. Empty scene -> +inf everywhere."""
        P, D = self._stack(P)
        if D.shape[1] == 0:
            return np.full(P.shape[0], np.inf)
        return np.min(D, axis=1)

    def part_ids(self, P):
        """Which part owns each point = argmin over parts (for material lookup). Empty scene -> all -1.

        On an exact tie, argmin returns the FIRST (lowest-index) part -- a stable, deterministic rule, so
        part ordering in parts() is the tie-break. Keep it stable if materials must not flicker at seams."""
        P, D = self._stack(P)
        if D.shape[1] == 0:
            return np.full(P.shape[0], -1, dtype=np.int64)
        return np.argmin(D, axis=1)

    ids = part_ids                                           # the name to_splats / exporters expect

    def material_at(self, P):
        """The material_name of the owning part at each point (or None where the scene is empty)."""
        mats = self._materials()
        return np.array([mats[i] if i >= 0 else None for i in self.part_ids(P)], dtype=object)

    # -- optional acceleration for MANY-part scenes ------------------------------------------------------------
    #
    # These wire in holographic_spatial.SpatialGrid so a demo does not re-write a broadphase index. They are a
    # HELPER, not automatic: the base eval stays naive-and-correct, and a subclass that wants culling supplies
    # bounding spheres via bounds() and calls parts_near() itself. This keeps the correctness-sensitive argmin
    # exact (no silent pruning) while still saving you from hand-rolling a grid.

    def bounds(self):
        """Override to return a list of (center_xyz, radius) bounding spheres, one per part, each enclosing that
        part's surface. Default: None (no bounds known -> parts_near() returns all parts, i.e. no culling)."""
        return None

    def _grid(self):
        """A SpatialGrid over the part-bounding-sphere centers, built once. None if bounds() is not provided."""
        if not hasattr(self, "_grid_cache"):
            b = self.bounds()
            if not b:
                self._grid_cache = None
            else:
                from holographic.misc.holographic_spatial import SpatialGrid
                centers = np.array([c for c, _ in b], float)
                self._radii = np.array([r for _, r in b], float)
                # cell size ~ the typical bounding radius keeps each query touching only a few cells.
                cell = float(np.median(self._radii)) if len(self._radii) else 1.0
                self._grid_cache = SpatialGrid(centers, max(cell, 1e-6))
        return self._grid_cache

    def parts_near(self, point, radius):
        """Indices of parts whose bounding sphere lies within `radius` of `point`, using the SpatialGrid.
        Falls back to ALL part indices when bounds() is not provided (so a caller can always rely on it).

        This is the culling primitive: a caller that only needs the field NEAR a point (e.g. building a local
        set of candidate parts for a region query) can evaluate just these instead of every part. It does NOT
        replace eval()'s exact min -- see the class docstring for why that stays naive."""
        grid = self._grid()
        n = len(self._fns())
        if grid is None:
            return list(range(n))
        # a part is "near" if the distance to its center is within radius + its own bounding radius.
        hits = grid.radius(point, radius + float(np.max(self._radii)) if n else radius)
        return sorted(hits)


def _selftest():
    """Prove the contract on a tiny three-sphere scene: min/argmin are correct, .ids aliases .part_ids,
    material lookup follows the argmin, parts_near matches a brute-force sphere test, and an empty scene is
    well-behaved (+inf distance, -1 ids)."""
    def sph(c, r):
        c = np.asarray(c, float)
        return lambda P: np.linalg.norm(P - c, axis=1) - r

    class S(SDFScene):
        def parts(self):
            return [(sph((0, 0, 0), 1.0), "a"), (sph((3, 0, 0), 1.0), "b"), (sph((0, 3, 0), 0.5), "c")]
        def bounds(self):
            return [((0, 0, 0), 1.0), ((3, 0, 0), 1.0), ((0, 3, 0), 0.5)]

    s = S()
    P = np.array([[0, 0, 0], [3, 0, 0], [1.5, 0, 0], [0, 3, 0]], float)
    assert np.allclose(s.eval(P), [-1.0, -1.0, 0.5, -0.5])          # nearest-surface signed distance
    assert list(s.part_ids(P)) == [0, 1, 0, 2]                      # owning part = argmin
    assert np.array_equal(s.ids(P), s.part_ids(P))                  # exporter alias
    assert list(s.material_at(P)) == ["a", "b", "a", "c"]           # material follows the argmin
    assert s.parts_near((0.2, 0.1, 0.0), 0.5) == [0]                # grid cull vs brute force

    class Empty(SDFScene):
        def parts(self):
            return []
    e = Empty()
    assert np.all(np.isinf(e.eval(P))) and list(e.part_ids(P)) == [-1, -1, -1, -1]

    # from_parts: build the SAME scene without subclassing -> identical eval/materials, and bounds enable culling.
    fp = SDFScene.from_parts([(sph((0, 0, 0), 1.0), "a"), (sph((3, 0, 0), 1.0), "b"), (sph((0, 3, 0), 0.5), "c")],
                             bounds=[((0, 0, 0), 1.0), ((3, 0, 0), 1.0), ((0, 3, 0), 0.5)])
    assert np.allclose(fp.eval(P), s.eval(P)) and list(fp.material_at(P)) == ["a", "b", "a", "c"]
    assert fp.parts_near((0.2, 0.1, 0.0), 0.5) == [0]
    print("holographic_sdfscene selftest OK (min/argmin/materials/parts_near correct; empty scene well-behaved; "
          "from_parts builds an identical scene without subclassing and supports culling)")


if __name__ == "__main__":
    _selftest()

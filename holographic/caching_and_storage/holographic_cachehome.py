"""holographic_cachehome.py -- the CACHE home (consolidation backlog H2): bake a slow evaluator over the thing that
VARIES, then look it up cheaply. The engine's core performance lever ("bake once, query O(1)"), in one place.

WHY THIS EXISTS
---------------
The same move is written in several modules: sample an expensive function over a regular grid and read it back by
trilinear interpolation (holographic_matbake bakes material channels; holographic_sdfbake bakes a distance field),
sample a BRDF over view/roughness into a small table (holographic_viewlut), sample a deformation over frames
(holographic_anim). Each re-writes the SAME grid-generation (np.linspace -> meshgrid(indexing='ij') -> stack), then
its own lookup. This home owns that shared core so the bakes stop duplicating it:

    pts, res3 = Cache.grid_points(lo, hi, res)     # the ONE position-grid generator the bakes share
    grid, pts = Cache.bake_grid(evaluator, lo, hi, res)   # sample an evaluator over that grid
    Cache.bake(evaluator, vary="position"|"view"|"time"|"constant", ...)   # the dispatcher by what varies

Route, don't rewrite: this owns the shared grid-sample-and-store core; each domain keeps its own lookup reader
(BakedField, GridSDF, the view LUT, the frame table) with its own clamping/interpolation. What is unified is the
precompute, not the readers.

NOT holographic_cache.py, which is Ward's irradiance-GRADIENT cache (value + Jacobian at sparse anchors, first-order
interpolation) -- a different, complementary caching scheme. This home is the dense bake-over-a-grid one.
"""
import numpy as np


def _res3(res):
    """Normalise a resolution to a length-3 int array (scalar -> cube, or a per-axis triple)."""
    return np.broadcast_to(np.asarray(res, int), (3,)).astype(int)


class BakedGrid:
    """A value sampled onto a regular 3-D grid over [lo,hi], read back by trilinear interpolation -- O(1) per point.
    Holds a scalar grid (rx,ry,rz) or a channel grid (rx,ry,rz,C). Points outside the box clamp to the edge. This
    mirrors matbake.BakedField's reader so a position bake can share the whole path when it wants to."""

    def __init__(self, grid, lo, hi):
        self.grid = np.asarray(grid, float)
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.res = np.array(self.grid.shape[:3])

    def sample(self, P):
        P = np.asarray(P, float)
        frac = (P - self.lo) / np.maximum(self.hi - self.lo, 1e-12)
        coord = np.clip(frac * (self.res - 1), 0.0, self.res - 1)
        i0 = np.floor(coord).astype(int)
        i1 = np.minimum(i0 + 1, self.res - 1)
        w = coord - i0
        out = None
        for dx in (0, 1):                                          # accumulate the 8 surrounding corners (same order
            for dy in (0, 1):                                      # as matbake.BakedField, so results agree)
                for dz in (0, 1):
                    wx = w[:, 0] if dx else 1.0 - w[:, 0]
                    wy = w[:, 1] if dy else 1.0 - w[:, 1]
                    wz = w[:, 2] if dz else 1.0 - w[:, 2]
                    ix = i1[:, 0] if dx else i0[:, 0]
                    iy = i1[:, 1] if dy else i0[:, 1]
                    iz = i1[:, 2] if dz else i0[:, 2]
                    corner = self.grid[ix, iy, iz]
                    wgt = wx * wy * wz
                    contrib = wgt[:, None] * corner if corner.ndim > 1 else wgt * corner
                    out = contrib if out is None else out + contrib
        return out


class Cache:
    """Bake-and-query, chosen by what VARIES. All staticmethods -- Cache is a namespace of the shared precompute."""

    @staticmethod
    def grid_points(lo, hi, res):
        """The ONE position-grid generator the bakes share: points of a regular grid spanning [lo,hi] at `res` per
        axis (scalar or triple), row-major over indexing='ij'. Returns (points (P,3), res3 (3,)). Deterministic --
        a bake that generates its grid HERE is bit-for-bit identical to the inline np.linspace/meshgrid it replaced.
        """
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        r = _res3(res)
        axes = [np.linspace(lo[k], hi[k], int(r[k])) for k in range(3)]
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        return pts, r

    @staticmethod
    def bake_grid(evaluator, lo, hi, res):
        """Sample `evaluator(points)->values` over the position grid and reshape to (rx,ry,rz[,C]). Returns
        (grid, points). The convenience wrapper over grid_points for the common single-evaluator case."""
        pts, r = Cache.grid_points(lo, hi, res)
        vals = np.asarray(evaluator(pts))
        grid = vals.reshape(tuple(int(x) for x in r) + vals.shape[1:])
        return grid, pts

    @staticmethod
    def bake(evaluator, vary="position", **kw):
        """Dispatch by what varies:
          vary='constant' : return evaluator()               -- compute once, no table.
          vary='position' : bake over [lo,hi] at res -> a BakedGrid (kw: lo, hi, res=24).
          vary='view'     : delegate to holographic_viewlut.bake_view_lut (kw passed through).
          vary='time'     : delegate to holographic_anim.bake_deformation (kw: base, n_frames).
        """
        if vary == "constant":
            return evaluator()
        if vary == "position":
            grid, _ = Cache.bake_grid(evaluator, kw["lo"], kw["hi"], kw.get("res", 24))
            return BakedGrid(grid, kw["lo"], kw["hi"])
        if vary == "view":
            from holographic.rendering.holographic_viewlut import bake_view_lut
            return bake_view_lut(**kw)
        if vary == "time":
            from holographic.misc.holographic_anim import bake_deformation
            return bake_deformation(kw["base"], kw["n_frames"], evaluator)
        raise ValueError("Cache.bake: unknown vary=%r (constant/position/view/time)" % vary)


def cache_backends():
    """What varies (the strategies Cache dispatches over), for the catalog / discovery."""
    return ("constant", "position", "view", "time")


def _selftest():
    lo = np.array([-1.0, -1.0, -1.0]); hi = np.array([1.0, 1.0, 1.0]); res = 8

    # the shared grid generator matches the inline np.linspace/meshgrid the bakes used
    pts, r = Cache.grid_points(lo, hi, res)
    axes = [np.linspace(lo[k], hi[k], res) for k in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    ref = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    assert np.array_equal(pts, ref) and tuple(r) == (res, res, res)

    # bake a scalar field and a colour field; trilinear lookup at a node returns the stored value
    scalar = lambda P: P[:, 0] ** 2 + P[:, 1]
    colour = lambda P: np.stack([P[:, 0], P[:, 1], P[:, 2]], axis=1)
    bg = Cache.bake(scalar, vary="position", lo=lo, hi=hi, res=res)
    node = pts[100]
    assert abs(float(bg.sample(node[None, :])[0]) - float(scalar(node[None, :])[0])) < 1e-9
    cg = Cache.bake(colour, vary="position", lo=lo, hi=hi, res=res)
    assert np.allclose(cg.sample(node[None, :])[0], colour(node[None, :])[0], atol=1e-9)

    # constant strategy computes once
    assert Cache.bake(lambda: 42.0, vary="constant") == 42.0
    print("OK: holographic_cachehome self-test passed (shared grid matches inline bake; trilinear exact at nodes; "
          "dispatch over %s)" % ", ".join(cache_backends()))


if __name__ == "__main__":
    _selftest()

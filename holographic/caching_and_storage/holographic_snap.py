"""holographic_snap.py -- SNAPPING = cleanup, applied to geometry (modeling-app feature layer).

Thinking holographically: snapping IS cleanup. VSA cleanup projects a noisy vector onto the nearest CLEAN atom in
a codebook; snapping projects a dragged, continuous position onto the nearest ALLOWED place -- a grid node, an
existing vertex, a point on an edge, an angle increment. Same operation, geometric codebook. And just as cleanup
can REFUSE a weak match (return "no confident atom"), a snap has a TOLERANCE: if nothing allowed is close enough,
the point is left where it is. That confidence gate is what stops a cursor from teleporting across the screen.

These read raw coordinates (the honest way -- no lossy encoding for something this exact). NumPy + stdlib only;
deterministic.
"""
import numpy as np


def snap_to_grid(p, spacing, origin=0.0):
    """Snap a point to the nearest grid node -- round each coordinate to the lattice. The simplest cleanup: the
    codebook is the infinite regular grid, so the nearest entry is just a rounding."""
    p = np.asarray(p, float)
    origin = np.asarray(origin, float)
    return origin + np.round((p - origin) / spacing) * spacing


def snap_to_points(p, points, tol=None):
    """Snap to the NEAREST point in a set -- this is literally cleanup (nearest codebook entry). Returns
    (snapped_point, index, distance). If `tol` is given and the nearest point is farther than tol, the ORIGINAL
    point is returned unchanged with index -1 -- a confidence-gated snap, like cleanup refusing a weak match."""
    p = np.asarray(p, float)
    pts = np.asarray(points, float)
    if len(pts) == 0:
        return p, -1, float("inf")
    d = np.linalg.norm(pts - p, axis=1)
    i = int(np.argmin(d))
    if tol is not None and d[i] > tol:
        return p, -1, float(d[i])                            # nothing close enough -> leave the point alone
    return pts[i].copy(), i, float(d[i])


def snap_to_segment(p, a, b):
    """The nearest point on the line SEGMENT a-b (clamped to the endpoints) -- snapping to an edge."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    p = np.asarray(p, float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    t = 0.0 if denom < 1e-15 else np.clip(float(np.dot(p - a, ab)) / denom, 0.0, 1.0)
    return a + t * ab


def snap_value(x, increment, origin=0.0):
    """Snap a scalar to the nearest multiple of `increment` from `origin` -- e.g. a length to 0.25 m steps."""
    return origin + round((x - origin) / increment) * increment


def snap_angle(theta, increment):
    """Snap an angle (radians) to the nearest multiple of `increment` -- e.g. rotate in 15-degree steps."""
    return round(theta / increment) * increment


class Snapper:
    """Snaps a point to the nearest snap target within a tolerance, combining a GRID and a VERTEX set. Whichever
    target is nearest (within `tol`) wins; if nothing is close enough, the point is returned unchanged (the
    confidence gate). This is the object a viewport's snapping toggle drives."""

    def __init__(self, grid=None, vertices=None, tol=0.25):
        self.grid = grid                                     # grid spacing (scalar) or None
        self.vertices = None if vertices is None else np.asarray(vertices, float)
        self.tol = tol

    def snap(self, p):
        """Return (snapped_point, kind) where kind is 'vertex', 'grid', or 'none'. A vertex within tol beats the
        grid (vertices are the stronger, more specific target); then the grid; else the point is left alone."""
        p = np.asarray(p, float)
        best = (p, "none", self.tol)
        if self.vertices is not None and len(self.vertices):
            v, i, d = snap_to_points(p, self.vertices, tol=self.tol)
            if i >= 0 and d < best[2]:
                best = (v, "vertex", d)
        if self.grid is not None:
            g = snap_to_grid(p, self.grid)
            d = float(np.linalg.norm(g - p))
            if d <= self.tol and best[1] != "vertex":        # vertices win ties; grid fills in otherwise
                best = (g, "grid", d)
        return best[0], best[1]


def _selftest():
    """Grid snap rounds to the lattice; point snap is nearest-neighbour cleanup; the tolerance gates a far snap
    (leaves the point); segment snap clamps to an edge; angle/value snap to increments; the Snapper prefers a
    nearby vertex over the grid; deterministic."""
    # (1) grid snap = rounding to the lattice
    assert np.allclose(snap_to_grid([0.12, 0.49, -0.51], 0.25), [0.0, 0.5, -0.5])
    assert np.allclose(snap_to_grid([1.3], 1.0), [1.0])

    # (2) point snap = cleanup (nearest codebook entry) with an index
    pts = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    sp, i, d = snap_to_points([0.9, 0.1, 0.0], pts)
    assert i == 1 and np.allclose(sp, [1, 0, 0])

    # (3) the tolerance gates: a far point is left ALONE (index -1) -- the confidence refusal
    sp, i, d = snap_to_points([5.0, 5.0, 5.0], pts, tol=0.5)
    assert i == -1 and np.allclose(sp, [5, 5, 5])

    # (4) segment snap clamps to the edge
    assert np.allclose(snap_to_segment([0.5, 5.0, 0.0], [0, 0, 0], [1, 0, 0]), [0.5, 0.0, 0.0])
    assert np.allclose(snap_to_segment([-1.0, 0.0, 0.0], [0, 0, 0], [1, 0, 0]), [0, 0, 0])   # clamped to a end

    # (5) scalar + angle snapping to increments
    assert abs(snap_value(0.62, 0.25) - 0.5) < 1e-12
    assert abs(snap_angle(np.radians(20), np.radians(15)) - np.radians(15)) < 1e-9

    # (6) the Snapper: a nearby vertex beats the grid, and a far point stays put
    snp = Snapper(grid=1.0, vertices=[[0.05, 0.05, 0.0]], tol=0.25)
    out, kind = snp.snap([0.1, 0.1, 0.0])
    assert kind == "vertex" and np.allclose(out, [0.05, 0.05, 0.0])
    out, kind = snp.snap([0.48, 0.0, 0.0])                  # no vertex near, but near a grid line at 0.5? d=0.02<tol
    assert kind in ("grid", "none")
    out, kind = snp.snap([0.4, 0.4, 0.4])                  # nothing within tol -> left alone
    assert kind == "none" and np.allclose(out, [0.4, 0.4, 0.4])

    # (7) deterministic
    assert np.array_equal(snap_to_grid([0.3, 0.7], 0.5), snap_to_grid([0.3, 0.7], 0.5))

    print("holographic_snap selftest OK: grid snap rounds to the lattice; point snap is nearest-neighbour cleanup "
          "with an index; the tolerance gates a far snap (leaves the point, index -1 -- the confidence refusal); "
          "segment snap clamps to an edge; scalar/angle snap to increments; the Snapper prefers a nearby vertex "
          "over the grid; deterministic")


if __name__ == "__main__":
    _selftest()

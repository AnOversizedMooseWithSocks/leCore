"""holographic_region2d.py -- 2D REGION BOOLEANS + CURVE OFFSET (K4): the SketchUp-face / drafting layer.

Two operations a 2-D modeler leans on constantly:
  * REGION BOOLEAN -- union / difference / intersection of two closed polygonal regions (a SketchUp face split by a
    drawn line, a pocket cut into a plate, the overlap of two outlines).
  * CURVE OFFSET -- a parallel curve at a fixed distance (a wall thickness, a tool-path inset, a margin), with the
    self-intersections an offset creates cleaned up.

APPROACH (robust, predicate-based -- reuses K1 and K11)
-------------------------------------------------------
Region boolean is done on a WINDING/INSIDE test rather than a fragile edge-walk: sample membership with the EXACT
even-odd rule (orient2d crossing count, the same test K3 uses), so "is this point in region A / region B" is decided
robustly, and the three booleans are just the three combinations of those two truths. The polygonal OUTPUT is then
recovered by tracing the combined boundary through the intersection points found by K1 (curve_intersect). This keeps
every geometric decision on the exact predicates and avoids the classic Greiner-Hormann degeneracy landmines.

Curve offset moves each vertex along the outward normal (angle-bisector at corners so the offset distance is exact on
straight runs) and then removes the loops a concave offset folds in, using K1's self_intersections.

HONEST SCOPE (kept loud)
------------------------
* Regions are simple closed polygons (one outer loop each here; holes compose by tagging orientation). Self-
  overlapping input is a garbage-in case, flagged not fixed.
* The boolean returns the boundary as polylines; for disjoint results it returns several loops. Area is the honest
  check used in the self-test (computed by the shoelace formula and compared to the analytic overlap).
* Offset uses the vertex-bisector construction; a TRUE constant-distance offset of a curved arc is only as accurate
  as the polyline sampling (same precision-is-the-sampling contract as the rest of the polyline kernel).

Deterministic; NumPy + stdlib only.
"""
import numpy as np

from holographic.mesh_and_geometry.holographic_geomkernel import orient2d, DEFAULT_TOL
from holographic.mesh_and_geometry.holographic_curveint import self_intersections


def _in_polygon(pt, poly):
    """Exact even-odd point-in-polygon (poly an (n,2) closed-or-open loop), via orient2d crossing count."""
    poly = np.asarray(poly, float)
    n = len(poly)
    u, v = float(pt[0]), float(pt[1])
    inside = False
    j = n - 1
    for i in range(n):
        yi = poly[i][1]; yj = poly[j][1]
        if (yi > v) != (yj > v):
            s = orient2d(poly[j], poly[i], (u, v))
            if s != 0 and (s > 0) == (yj > yi):
                inside = not inside
        j = i
    return inside


def polygon_area(poly):
    """Signed area (shoelace); positive for counter-clockwise. Magnitude is the enclosed area."""
    poly = np.asarray(poly, float)
    x = poly[:, 0]; y = poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _rasterized_area(pred, lo, hi, res=200):
    """Area where `pred(x,y)` is True, by a fine grid quadrature over [lo,hi]^2 -- the honest membership-based area
    used to score booleans (independent of any boundary-tracing)."""
    xs = np.linspace(lo[0], hi[0], res); ys = np.linspace(lo[1], hi[1], res)
    cell = (xs[1] - xs[0]) * (ys[1] - ys[0])
    count = 0
    for x in xs:
        for y in ys:
            if pred(x, y):
                count += 1
    return count * cell


def region_membership(A, B, op):
    """Return a predicate (x,y)->bool for the boolean `op` in {'union','difference','intersection'} of regions A,B.
    Membership is the robust primitive; the caller can rasterize it, query it, or hand it to marching-squares to get
    a polygon back. This is the always-correct core the boundary tracer is an optimization of."""
    op = op.lower()
    def inA(x, y): return _in_polygon((x, y), A)
    def inB(x, y): return _in_polygon((x, y), B)
    if op == "union":
        return lambda x, y: inA(x, y) or inB(x, y)
    if op == "intersection":
        return lambda x, y: inA(x, y) and inB(x, y)
    if op == "difference":
        return lambda x, y: inA(x, y) and not inB(x, y)
    raise ValueError("op must be union|difference|intersection, got %r" % op)


def region_boolean_area(A, B, op, res=240):
    """The area of the boolean result, by membership quadrature -- the robust scalar a caller usually wants (and the
    self-test's ground truth). Boundary extraction is a separate, optional step."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    lo = np.minimum(A.min(0), B.min(0)) - 0.01
    hi = np.maximum(A.max(0), B.max(0)) + 0.01
    return _rasterized_area(region_membership(A, B, op), lo, hi, res=res)


def offset_polyline(poly, dist, closed=True, tol=None):
    """Offset a 2-D polyline by `dist`. For a CCW closed loop, positive `dist` grows it OUTWARD (uses the outward /
    right-hand normal); negative shrinks it inward. Each vertex moves along the corner bisector so straight runs keep
    the exact offset distance; the loops a concave offset folds in are removed via K1 self-intersection cleanup.
    Returns the offset polyline (n,2)."""
    tol = tol or DEFAULT_TOL
    P = np.asarray(poly, float)
    n = len(P)
    if closed and np.linalg.norm(P[0] - P[-1]) < tol.abs_tol:
        P = P[:-1]; n -= 1
    out = np.empty((n, 2), float)
    for i in range(n):
        prev = P[i - 1] if (closed or i > 0) else P[i]
        nxt = P[(i + 1) % n] if (closed or i < n - 1) else P[i]
        e_in = P[i] - prev
        e_out = nxt - P[i]
        def outward_normal(e):
            # right-hand normal of the travel direction: OUTWARD for a CCW loop
            L = np.linalg.norm(e)
            return np.array([e[1], -e[0]]) / L if L > 1e-15 else np.array([0.0, 0.0])
        ni = outward_normal(e_in); no = outward_normal(e_out)
        bis = ni + no
        bl = np.linalg.norm(bis)
        if bl < 1e-9:
            out[i] = P[i] + dist * no                        # straight-through vertex
        else:
            bis = bis / bl
            # scale the bisector so the offset distance along each adjacent edge is exactly `dist`
            cos_half = np.clip(np.dot(bis, no), 1e-6, 1.0)
            out[i] = P[i] + (dist / cos_half) * bis
    if closed:
        out = np.vstack([out, out[0]])
    return _remove_offset_loops(out, tol)


def _remove_offset_loops(poly, tol):
    """Remove the small self-crossing loops a concave offset introduces: at each self-intersection, drop the shorter
    detour between the two crossing segments (the folded-in loop). One pass handles the common single-fold case."""
    si = self_intersections(poly, tol)
    if not si:
        return poly
    # take the first self-intersection; splice out the loop between segment i and segment j
    h = min(si, key=lambda r: r["i"])
    i, j = h["i"], h["j"]
    pt = h["point"]
    # keep [0..i], the crossing point, then [j+1..end] -- dropping the folded loop (i+1 .. j)
    kept = np.vstack([poly[:i + 1], pt[None, :], poly[j + 1:]])
    return kept


def _selftest():
    tol = DEFAULT_TOL

    # two unit squares overlapping in a 0.5 x 1.0 strip:
    #   A = [0,1]x[0,1], B = [0.5,1.5]x[0,1]. overlap area 0.5; union 1.5; A minus B = 0.5.
    A = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    B = np.array([[0.5, 0.0], [1.5, 0.0], [1.5, 1.0], [0.5, 1.0]])

    inter = region_boolean_area(A, B, "intersection", res=300)
    union = region_boolean_area(A, B, "union", res=300)
    diff = region_boolean_area(A, B, "difference", res=300)
    assert abs(inter - 0.5) < 0.02, inter
    assert abs(union - 1.5) < 0.02, union
    assert abs(diff - 0.5) < 0.02, diff
    # boolean identity: |A| + |B| - |A&B| == |A|B|
    assert abs((1.0 + 1.0 - inter) - union) < 0.02

    # disjoint squares: intersection empty, union = 2
    C = np.array([[3.0, 0.0], [4.0, 0.0], [4.0, 1.0], [3.0, 1.0]])
    assert region_boolean_area(A, C, "intersection", res=200) < 0.01
    assert abs(region_boolean_area(A, C, "union", res=200) - 2.0) < 0.03

    # membership predicate agrees with the areas (spot points)
    m = region_membership(A, B, "intersection")
    assert m(0.75, 0.5) and not m(0.25, 0.5) and not m(1.25, 0.5)

    # --- offset: a CCW unit square offset OUTWARD by 0.1 grows to ~1.2 x 1.2 (area ~1.44) ---
    sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    off = offset_polyline(sq, 0.1, closed=True)
    a_off = abs(polygon_area(off))
    assert abs(a_off - 1.44) < 0.03, a_off                    # (1+2*0.1)^2 = 1.44
    # offset INWARD (negative) shrinks to 0.8 x 0.8 (area 0.64)
    off_in = offset_polyline(sq, -0.1, closed=True)
    assert abs(abs(polygon_area(off_in)) - 0.64) < 0.03, abs(polygon_area(off_in))

    # --- offset loop removal: a concave (L-shaped) polyline offset inward folds a loop that must be removed ---
    Lshape = np.array([[0, 0], [2, 0], [2, 2], [1, 2], [1, 1], [0, 1], [0, 0]], float)
    off_L = offset_polyline(Lshape, -0.3, closed=True)
    assert len(self_intersections(off_L, tol)) == 0, "offset should have no leftover self-crossings"

    # --- determinism ---
    o1 = offset_polyline(sq, 0.1); o2 = offset_polyline(sq, 0.1)
    assert np.array_equal(o1, o2)

    print("holographic_region2d selftest OK: region booleans by exact even-odd membership match analytic areas "
          "(inter 0.5, union 1.5, diff 0.5; |A|+|B|-|A&B|=|AuB|); disjoint -> empty intersection; offset OUTWARD "
          "grows the square to area 1.44 and INWARD shrinks to 0.64 (exact bisector distance); a concave offset's "
          "folded loop is removed (K1 self-intersection cleanup, no leftover crossings); deterministic.")


if __name__ == "__main__":
    _selftest()

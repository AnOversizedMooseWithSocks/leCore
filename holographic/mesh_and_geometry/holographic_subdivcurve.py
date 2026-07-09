"""Subdivision curves on hypervector sequences (ARCH-5): Loop subdivision (FWD-8), turned inward.

WHY THIS MODULE EXISTS
----------------------
FWD-8 subdivided a MESH -- a 2-manifold -- refining it (1 triangle -> 4) and smoothing it toward a limit surface,
and its honest frame was "subdivision = a topological REFINE + a graph-signal LOW-PASS smooth". ARCH-5 turns that
inward onto the engine's own 1-D structure: a SEQUENCE of hypervectors is a polyline (a 1-manifold) through vector
space -- exactly what the sequence faculties encode -- and the same idea refines it into a smooth limit CURVE.

The scheme is CHAIKIN'S corner-cutting (Chaikin 1974), the curve analogue of Loop and the classic generator of a
quadratic B-spline limit: each edge (p_i, p_{i+1}) is replaced by two points that cut its corner,
    q_i = 3/4 p_i + 1/4 p_{i+1},   r_i = 1/4 p_i + 3/4 p_{i+1},
which both REFINES (doubles the point count) and SMOOTHS (corner-cutting is a low-pass filter). The mesh
properties map across exactly:
    Loop: faces x4 each level            <->  Chaikin: points x2 each level
    Loop: flat stays flat (affine)       <->  Chaikin: a straight line of vectors stays straight (affine)
    Loop: converges to a limit surface   <->  Chaikin: converges to a limit curve
    Loop: low-pass made geometric        <->  Chaikin: roughness (2nd-differences) shrinks each level

WHAT IT PROVIDES
  * chaikin_subdivide(points, closed) -- one level of corner-cutting.
  * subdivide_sequence(points, levels, closed) -- `levels` of Chaikin on a sequence of vectors. Returns the refined
    sequence (an (M, dim) array).

THE MEASUREMENT BAR (checked exactly in the self-test)
  * REFINE: an open polyline of n points goes to 2(n-1) per level; a closed one to 2n.
  * AFFINE REPRODUCTION: subdividing a straight line of vectors (a linear ramp t*a + (1-t)*b) keeps every point ON
    the line to machine precision -- the exact analogue of FWD-8's "flat stays flat".
  * CONVERGENCE: the curve length settles (successive-level changes shrink) -- the polyline approaches a limit.
  * LOW-PASS: a zig-zag sequence's roughness (sum of squared second differences) shrinks every level -- corner
    cutting removes the high frequencies, just as Loop made a cube's dihedral spread shrink.

DETERMINISM (per ISA.md)
  Fixed affine combinations in fixed order; no RNG. Same sequence -> identical refinement (asserted).

KEPT NEGATIVES (loud)
  * Chaikin is APPROXIMATING, not interpolating: the limit curve does NOT pass through the original interior
    control points (it cuts their corners). This is the exact mirror of FWD-8's negative -- Loop approximates, so a
    curved icosphere smooths to Loop's own limit, not the exact sphere. An INTERPOLATING scheme (the Dyn-Levin-
    Gregory 4-point rule) would keep the control points but is less smooth and needs >=4 points with boundary
    special-casing -- the classic approximating/interpolating trade-off, deferred.
  * Boundaries: the open scheme cuts the corners at the ends too, so the very first/last control points are not
    preserved (the standard simple Chaikin boundary; an endpoint-preserving variant is a separate rule).
"""

import numpy as np


def chaikin_subdivide(points, closed=False):
    """One level of Chaikin corner-cutting on a sequence of vectors. Each edge (p_i, p_{i+1}) becomes two points,
    3/4 p_i + 1/4 p_{i+1} and 1/4 p_i + 3/4 p_{i+1}. Doubles the point count (open: 2(n-1); closed: 2n) and smooths.
    Returns an (M, dim) array."""
    P = np.asarray(points, float)
    n = len(P)
    if n < 2:
        return P.copy()
    out = []
    edges = range(n) if closed else range(n - 1)          # closed wraps the last->first edge
    for i in edges:
        a, b = P[i], P[(i + 1) % n]
        out.append(0.75 * a + 0.25 * b)                   # the point nearer a
        out.append(0.25 * a + 0.75 * b)                   # the point nearer b
    return np.array(out)


def subdivide_sequence(points, levels=1, closed=False):
    """Apply `levels` of Chaikin corner-cutting to a sequence of hypervectors -- refining the polyline into a smooth
    limit curve through vector space (the 1-D inward mirror of FWD-8's mesh subdivision). Returns the refined (M,
    dim) sequence."""
    P = np.asarray(points, float)
    for _ in range(int(levels)):
        P = chaikin_subdivide(P, closed=closed)
    return P


def _roughness(P):
    """A polyline's high-frequency content: the sum of squared second differences. Low-pass smoothing shrinks it."""
    if len(P) < 3:
        return 0.0
    return float(np.sum(np.diff(P, n=2, axis=0) ** 2))


# =====================================================================================================
# Self-test -- refine (count doubling), affine reproduction, convergence, low-pass smoothing.
# =====================================================================================================
def _selftest():
    rng = np.random.default_rng(0)
    dim = 64

    # --- REFINE: counts double exactly (open 2(n-1), closed 2n) ---
    P = rng.standard_normal((6, dim))
    open_counts = [len(subdivide_sequence(P, l)) for l in range(4)]
    closed_counts = [len(subdivide_sequence(P, l, closed=True)) for l in range(4)]
    assert open_counts == [6, 10, 18, 34], open_counts        # 2(n-1) per level
    assert closed_counts == [6, 12, 24, 48], closed_counts    # 2n per level

    # --- AFFINE REPRODUCTION: a straight line of vectors stays straight (FWD-8's "flat stays flat") ---
    a, b = rng.standard_normal(dim), rng.standard_normal(dim)
    ramp = np.array([a + (b - a) * t for t in np.linspace(0, 1, 6)])
    sub = subdivide_sequence(ramp, 3)
    dn = (b - a) / np.linalg.norm(b - a)
    max_resid = max(np.linalg.norm((p - a) - np.dot(p - a, dn) * dn) for p in sub)
    assert max_resid < 1e-12, f"a linear ramp must stay on the line, got residual {max_resid:.1e}"

    # --- CONVERGENCE: the curve length settles (successive-level changes shrink) ---
    Q = rng.standard_normal((8, dim))
    lengths = [float(np.sum(np.linalg.norm(np.diff(subdivide_sequence(Q, l), axis=0), axis=1))) for l in range(6)]
    deltas = [abs(lengths[i + 1] - lengths[i]) for i in range(len(lengths) - 1)]
    assert all(deltas[i + 1] < deltas[i] for i in range(len(deltas) - 1)), "level-to-level change must shrink (converge)"

    # --- LOW-PASS: a zig-zag's roughness shrinks every level ---
    zig = np.zeros((10, dim)); zig[::2, 0] = 1.0; zig[1::2, 0] = -1.0
    rough = [_roughness(subdivide_sequence(zig, l)) for l in range(5)]
    assert all(rough[i + 1] < rough[i] for i in range(len(rough) - 1)), "corner-cutting must reduce roughness"

    # --- KEPT NEGATIVE: Chaikin is APPROXIMATING -- the original interior points are NOT preserved ---
    interior = ramp[2]                                        # an original interior control point
    nearest = min(np.linalg.norm(interior - p) for p in subdivide_sequence(ramp, 2))
    assert nearest > 1e-6, "Chaikin approximates -- it cuts the corners, so control points are not interpolated"

    # --- determinism ---
    assert np.array_equal(subdivide_sequence(P, 2), subdivide_sequence(P, 2))

    print(f"holographic_subdivcurve selftest: ok (Chaikin corner-cutting on vector sequences -- REFINE: open "
          f"{open_counts} (2(n-1)/level), closed {closed_counts} (2n/level); AFFINE: a straight line stays straight "
          f"(residual {max_resid:.0e}); CONVERGES: length deltas {[f'{d:.1f}' for d in deltas]} shrink; LOW-PASS: "
          f"zig-zag roughness {[f'{r:.2f}' for r in rough]} shrinks; KEPT NEGATIVE: approximating, control points "
          f"NOT interpolated (nearest {nearest:.2f}); deterministic)")


if __name__ == "__main__":
    _selftest()

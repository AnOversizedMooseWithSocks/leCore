"""holographic_geomkernel.py -- the shared GEOMETRY-KERNEL foundation: ONE tolerance authority + EXACT-sign
geometric predicates. Everything robust in the CAD stack (curve/curve, surface/surface, booleans, snapping, trim)
consults these, so the answer to "are these two points the same / is this point left of that line" has a SINGLE
source instead of a per-op magic epsilon.

WHY A TOLERANCE AUTHORITY (modeling-app backlog K11)
----------------------------------------------------
Robust intersection is only as consistent as its notion of "equal". If curve/curve uses 1e-6, the boolean uses
1e-9, and snapping uses 1e-3, the three disagree at a shared vertex and the topology tears. `ModelTolerance` is the
one object the document owns; a boolean and a snap consult the SAME numbers, so a point that is "on" an edge for one
is "on" it for the other. It carries an absolute length tolerance, a relative one (for large models where absolute
is too tight), and an angular tolerance (for tangency / parallel tests).

WHY EXACT-SIGN PREDICATES (the thing that makes intersection ROBUST)
--------------------------------------------------------------------
The single operation under every intersection and boolean is a SIGN: is c left of the line a->b (orient2d), is d
above the plane a-b-c (orient3d). Computed naively in float, a near-degenerate sign flips at random and the boolean
crashes or self-intersects. We compute the determinant in float first (fast path); only when it is within a
conservative error filter do we recompute it EXACTLY with stdlib `fractions.Fraction` (rational, no rounding) and
return the true sign. This is the adaptive-precision idea (Shewchuk) with Fraction as the exact stage instead of
expansion arithmetic -- stdlib-only, deterministic, and exact on the ties that matter. leCore also carries native
exact integer arithmetic in holographic_rns (phasor residues); Fraction is chosen here because the predicate inputs
are arbitrary rationals from coordinates, not the bounded integers RNS federates -- the honest tool for THIS job.

Deterministic; NumPy + stdlib only.
"""
from fractions import Fraction

import numpy as np


class ModelTolerance:
    """The document's single tolerance authority. `abs_tol` is the length below which two coordinates are "the same";
    `rel_tol` scales that by the size of the numbers involved (so a 10 km model isn't held to nanometres); `ang_tol`
    is the angle (radians) below which directions count as parallel/tangent.

    Consult it through its helpers so every op agrees: `length_eq`, `point_eq`, `is_zero`, `parallel`."""

    def __init__(self, abs_tol=1e-9, rel_tol=1e-12, ang_tol=1e-9):
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)
        self.ang_tol = float(ang_tol)

    def is_zero(self, x):
        """Is a scalar within absolute tolerance of zero."""
        return abs(float(x)) <= self.abs_tol

    def length_eq(self, a, b):
        """Are two lengths equal, using absolute OR relative tolerance (whichever is looser at this scale)."""
        a = float(a); b = float(b)
        return abs(a - b) <= max(self.abs_tol, self.rel_tol * max(abs(a), abs(b)))

    def point_eq(self, p, q):
        """Are two points the same within tolerance (Euclidean distance)."""
        p = np.asarray(p, float); q = np.asarray(q, float)
        d = float(np.linalg.norm(p - q))
        scale = max(float(np.linalg.norm(p)), float(np.linalg.norm(q)), 1.0)
        return d <= max(self.abs_tol, self.rel_tol * scale)

    def parallel(self, u, v):
        """Are two direction vectors parallel (or anti-parallel) within the angular tolerance."""
        u = np.asarray(u, float); v = np.asarray(v, float)
        nu = np.linalg.norm(u); nv = np.linalg.norm(v)
        if nu == 0 or nv == 0:
            return False
        c = abs(float(np.dot(u, v)) / (nu * nv))
        c = min(1.0, c)
        return (1.0 - c) <= (self.ang_tol ** 2) / 2.0 + 1e-15   # 1-cos(t) ~ t^2/2 for small t

    def __repr__(self):
        return "ModelTolerance(abs=%g, rel=%g, ang=%g)" % (self.abs_tol, self.rel_tol, self.ang_tol)


# a shared default so callers that don't thread a document tolerance still agree with each other
DEFAULT_TOL = ModelTolerance()


# ---- exact-sign predicates (float fast path + Fraction exact fallback) ---------------------------------------
def _sign(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def orient2d(a, b, c):
    """Sign of the orientation of triangle (a, b, c) in 2D: +1 counter-clockwise (c left of a->b), -1 clockwise,
    0 collinear. EXACT: a float estimate is returned immediately when it is safely away from zero; otherwise the
    determinant is recomputed with Fraction and its true sign returned (so collinearity is decided exactly)."""
    ax, ay = float(a[0]), float(a[1]); bx, by = float(b[0]), float(b[1]); cx, cy = float(c[0]), float(c[1])
    detleft = (ax - cx) * (by - cy)
    detright = (ay - cy) * (bx - cx)
    det = detleft - detright
    # conservative error bound for the 2x2 determinant of these magnitudes (Shewchuk-style static filter)
    err = (abs(detleft) + abs(detright)) * 4e-16
    if abs(det) > err:
        return _sign(det)
    # exact fallback
    A = (Fraction(a[0]) - Fraction(c[0])) * (Fraction(b[1]) - Fraction(c[1]))
    B = (Fraction(a[1]) - Fraction(c[1])) * (Fraction(b[0]) - Fraction(c[0]))
    return _sign(A - B)


def orient3d(a, b, c, d):
    """Sign of the orientation of tetrahedron (a, b, c, d): +1 if d is above the plane a-b-c (by the right-hand
    rule), -1 below, 0 coplanar. Exact via the same float-filter-then-Fraction scheme."""
    a = [float(x) for x in a]; b = [float(x) for x in b]; c = [float(x) for x in c]; d = [float(x) for x in d]
    adx, ady, adz = a[0] - d[0], a[1] - d[1], a[2] - d[2]
    bdx, bdy, bdz = b[0] - d[0], b[1] - d[1], b[2] - d[2]
    cdx, cdy, cdz = c[0] - d[0], c[1] - d[1], c[2] - d[2]
    det = (adx * (bdy * cdz - bdz * cdy)
           - ady * (bdx * cdz - bdz * cdx)
           + adz * (bdx * cdy - bdy * cdx))
    perm = ((abs(bdy * cdz) + abs(bdz * cdy)) * abs(adx)
            + (abs(bdx * cdz) + abs(bdz * cdx)) * abs(ady)
            + (abs(bdx * cdy) + abs(bdy * cdx)) * abs(adz))
    if abs(det) > perm * 8e-16:
        return _sign(det)
    fa = [Fraction(x) for x in a]; fb = [Fraction(x) for x in b]; fc = [Fraction(x) for x in c]; fd = [Fraction(x) for x in d]
    Adx, Ady, Adz = fa[0] - fd[0], fa[1] - fd[1], fa[2] - fd[2]
    Bdx, Bdy, Bdz = fb[0] - fd[0], fb[1] - fd[1], fb[2] - fd[2]
    Cdx, Cdy, Cdz = fc[0] - fd[0], fc[1] - fd[1], fc[2] - fd[2]
    D = (Adx * (Bdy * Cdz - Bdz * Cdy) - Ady * (Bdx * Cdz - Bdz * Cdx) + Adz * (Bdx * Cdy - Bdy * Cdx))
    return _sign(D)


def _selftest():
    rng = np.random.default_rng(0)

    # --- tolerance authority ---
    tol = ModelTolerance(abs_tol=1e-9)
    assert tol.point_eq([0, 0, 0], [0, 0, 5e-10])           # within abs tol
    assert not tol.point_eq([0, 0, 0], [0, 0, 1e-6])        # outside
    assert tol.length_eq(1.0, 1.0 + 5e-10)
    assert tol.parallel([1, 0, 0], [2, 0, 0]) and tol.parallel([1, 0, 0], [-3, 0, 0])
    assert not tol.parallel([1, 0, 0], [0, 1, 0])

    # --- orient2d: sign is correct on clear cases and EXACT on collinear ---
    assert orient2d((0, 0), (1, 0), (0, 1)) == 1            # ccw
    assert orient2d((0, 0), (1, 0), (0, -1)) == -1          # cw
    assert orient2d((0, 0), (1, 0), (2, 0)) == 0            # exactly collinear
    assert orient2d((0, 0), (1, 1), (2, 2)) == 0            # exactly collinear on a diagonal
    # a point a hair off the line must NOT read as collinear (exactness, not a fuzzy epsilon)
    assert orient2d((0, 0), (1, 1), (2, 2 + 1e-11)) == 1

    # --- orient3d: coplanar exact, and sign correct off-plane ---
    assert orient3d((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)) != 0
    assert orient3d((0, 0, 0), (1, 0, 0), (0, 1, 0), (0.3, 0.3, 0.0)) == 0   # exactly in the z=0 plane
    assert orient3d((0, 0, 0), (1, 0, 0), (0, 1, 0), (0.3, 0.3, 1e-11)) != 0 # a hair above -> nonzero

    # --- determinism: predicates are pure and reproducible ---
    pts = rng.standard_normal((20, 2))
    s1 = [orient2d(pts[i], pts[i + 1], pts[i + 2]) for i in range(18)]
    s2 = [orient2d(pts[i], pts[i + 1], pts[i + 2]) for i in range(18)]
    assert s1 == s2

    print("holographic_geomkernel selftest OK: ONE ModelTolerance authority (abs/rel/ang; point_eq, length_eq, "
          "parallel); orient2d/orient3d return the correct sign on clear cases and the EXACT sign on ties -- "
          "collinear/coplanar decided by Fraction, and a 1e-11 perturbation off the line/plane is NOT swallowed "
          "(exact, not a fuzzy epsilon); deterministic")


if __name__ == "__main__":
    _selftest()

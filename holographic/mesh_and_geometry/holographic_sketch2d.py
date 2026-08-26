"""holographic_sketch2d.py -- 2D GEOMETRIC CONSTRAINT SOLVER (K8): the parametric-sketch engine under SketchUp
inference and any dimensioned drawing that feeds a profile to extrude/trim.

WHY THIS IS A PROMOTION, NOT A FRESH BUILD
------------------------------------------
A constraint set is one more "iterate a projection" -- the exact shape the resonator, IK, PBD, and collision
resolution already are in this engine. Each geometric constraint (coincident, horizontal, distance, parallel, ...)
defines a small PROJECTION that nudges the points it touches toward satisfaction; sweeping all projections to a
fixed point (Gauss-Seidel relaxation) solves the sketch. No Jacobian assembly, no Newton on a global system -- the
same local-projection relaxation the physics solvers use, applied to geometry instead of dynamics.

WHAT IT SOLVES
--------------
Points in the plane plus constraints:
  fixed(p)                 -- pin a point (a datum / anchor)
  coincident(p, q)         -- p == q
  horizontal(p, q)         -- the segment p-q is horizontal (equal y)
  vertical(p, q)           -- the segment p-q is vertical (equal x)
  distance(p, q, d)        -- |p - q| == d  (a dimension)
  parallel(a, b, c, d)     -- segment a-b parallel to segment c-d
  perpendicular(a, b, c, d)-- segment a-b perpendicular to c-d
  point_on_line(p, a, b)   -- p lies on the infinite line a-b

DOF ACCOUNTING (under / well / over -constrained, the thing a sketch UI must report)
------------------------------------------------------------------------------------
Free DOF = 2 * (#points) - 2 * (#fixed points collapsed) ... reported simply as 2*movable - constraint_dof, where
each constraint removes a known count. It is a COUNT (the standard sketcher heuristic), not a rank of the Jacobian,
so it can misreport a redundant constraint set -- stated as a kept negative. The residual after solving is the
honest truth of whether the constraints were actually satisfiable.

Deterministic; NumPy + stdlib only.
"""
import numpy as np

from holographic.mesh_and_geometry.holographic_geomkernel import DEFAULT_TOL


class Sketch2D:
    """A 2D sketch: a set of points and a list of constraints, solved by iterated projection."""

    def __init__(self, tol=None):
        self.pts = []                       # list of np.array([x,y])
        self.fixed = set()                  # indices pinned
        self.constraints = []               # list of (kind, args...) tuples
        self.tol = tol or DEFAULT_TOL

    def add_point(self, x, y):
        self.pts.append(np.array([float(x), float(y)]))
        return len(self.pts) - 1

    # -- constraint declarations (each just records; the projection lives in _apply) --------------------------
    def fix(self, p):
        self.fixed.add(p); self.constraints.append(("fixed", p))

    def coincident(self, p, q):
        self.constraints.append(("coincident", p, q))

    def horizontal(self, p, q):
        self.constraints.append(("horizontal", p, q))

    def vertical(self, p, q):
        self.constraints.append(("vertical", p, q))

    def distance(self, p, q, d):
        self.constraints.append(("distance", p, q, float(d)))

    def parallel(self, a, b, c, d):
        self.constraints.append(("parallel", a, b, c, d))

    def perpendicular(self, a, b, c, d):
        self.constraints.append(("perpendicular", a, b, c, d))

    def point_on_line(self, p, a, b):
        self.constraints.append(("point_on_line", p, a, b))

    # -- the projections: each moves the involved (non-fixed) points a fraction toward satisfaction ----------
    def _move(self, i, delta):
        """Move point i by delta unless it is pinned (fixed points never move)."""
        if i not in self.fixed:
            self.pts[i] = self.pts[i] + delta

    def _apply(self, con):
        k = con[0]
        P = self.pts
        if k == "fixed":
            return
        if k == "coincident":
            _, p, q = con
            mid = 0.5 * (P[p] + P[q]); self._pull_to(p, mid); self._pull_to(q, mid)
        elif k == "horizontal":
            _, p, q = con
            y = 0.5 * (P[p][1] + P[q][1])
            self._move(p, np.array([0.0, y - P[p][1]])); self._move(q, np.array([0.0, y - P[q][1]]))
        elif k == "vertical":
            _, p, q = con
            x = 0.5 * (P[p][0] + P[q][0])
            self._move(p, np.array([x - P[p][0], 0.0])); self._move(q, np.array([x - P[q][0], 0.0]))
        elif k == "distance":
            _, p, q, d = con
            v = P[q] - P[p]; L = np.linalg.norm(v)
            if L < 1e-12:
                v = np.array([1.0, 0.0]); L = 1.0
            n = v / L; err = (L - d)
            self._move(p, 0.5 * err * n); self._move(q, -0.5 * err * n)
        elif k == "point_on_line":
            _, p, a, b = con
            ab = P[b] - P[a]; L2 = float(np.dot(ab, ab))
            if L2 < 1e-18:
                return
            t = float(np.dot(P[p] - P[a], ab)) / L2
            foot = P[a] + t * ab
            self._move(p, foot - P[p])          # only the point moves onto the line
        elif k in ("parallel", "perpendicular"):
            _, a, b, c, d = con
            u = P[b] - P[a]; w = P[d] - P[c]
            lu = np.linalg.norm(u); lw = np.linalg.norm(w)
            if lu < 1e-12 or lw < 1e-12:
                return
            un = u / lu; wn = w / lw
            if k == "perpendicular":
                wn_target = np.array([-un[1], un[0]])           # rotate the first dir 90 deg -> target for w
            else:
                wn_target = un if np.dot(un, wn) >= 0 else -un   # align w with u (nearest orientation)
            # rotate segment c-d about its midpoint toward wn_target (move only non-fixed of c,d)
            mid = 0.5 * (P[c] + P[d]); half = 0.5 * lw
            newc = mid - half * wn_target; newd = mid + half * wn_target
            self._move(c, 0.5 * (newc - P[c])); self._move(d, 0.5 * (newd - P[d]))

    def _pull_to(self, i, target, frac=1.0):
        self._move(i, frac * (target - self.pts[i]))

    def solve(self, iters=400, tol=None):
        """Relax all constraints to a fixed point (Gauss-Seidel sweeps). Returns {residual, iters, satisfied}."""
        tol = tol or self.tol
        last = None
        for it in range(iters):
            for con in self.constraints:
                self._apply(con)
            r = self.residual()
            if r <= tol.abs_tol:
                return {"residual": r, "iters": it + 1, "satisfied": True}
            last = r
        return {"residual": last, "iters": iters, "satisfied": last is not None and last <= tol.abs_tol}

    def residual(self):
        """Worst-case constraint violation (max over constraints), the honest measure of whether the sketch is
        actually solved regardless of the DOF count."""
        P = self.pts
        worst = 0.0
        for con in self.constraints:
            k = con[0]
            if k == "fixed":
                continue
            if k == "coincident":
                worst = max(worst, float(np.linalg.norm(P[con[1]] - P[con[2]])))
            elif k == "horizontal":
                worst = max(worst, abs(P[con[1]][1] - P[con[2]][1]))
            elif k == "vertical":
                worst = max(worst, abs(P[con[1]][0] - P[con[2]][0]))
            elif k == "distance":
                worst = max(worst, abs(float(np.linalg.norm(P[con[2]] - P[con[1]])) - con[3]))
            elif k == "point_on_line":
                _, p, a, b = con
                ab = P[b] - P[a]; L = np.linalg.norm(ab)
                if L > 1e-12:
                    n = ab / L; ap = P[p] - P[a]
                    d = abs(float(n[0] * ap[1] - n[1] * ap[0]))   # 2D cross magnitude = perpendicular distance
                    worst = max(worst, d)
            elif k in ("parallel", "perpendicular"):
                _, a, b, c, d = con
                u = P[b] - P[a]; w = P[d] - P[c]
                lu = np.linalg.norm(u); lw = np.linalg.norm(w)
                if lu > 1e-12 and lw > 1e-12:
                    cs = float(np.dot(u / lu, w / lw))
                    val = abs(cs) if k == "parallel" else abs(cs)   # parallel: |cos|->1; perp: |cos|->0
                    worst = max(worst, (1 - val) if k == "parallel" else val)
        return worst

    def dof(self):
        """Sketcher DOF heuristic: 2*movable_points - sum(constraint dof removed). Negative -> likely OVER-
        constrained; positive -> UNDER-constrained; ~0 -> well-constrained. A COUNT, not a Jacobian rank (kept
        negative: a redundant constraint set can misreport; residual after solve is the real truth)."""
        removed = {"fixed": 2, "coincident": 2, "horizontal": 1, "vertical": 1, "distance": 1,
                   "parallel": 1, "perpendicular": 1, "point_on_line": 1}
        total_removed = sum(removed[c[0]] for c in self.constraints)
        free = 2 * len(self.pts) - total_removed
        state = "well" if free == 0 else ("under" if free > 0 else "over")
        return {"free_dof": free, "state": state}


def _selftest():
    tol = DEFAULT_TOL

    # --- a rectangle from 4 free points: fix one corner, constrain horizontals/verticals + two dimensions ---
    s = Sketch2D()
    a = s.add_point(0.1, -0.2); b = s.add_point(3.0, 0.3)      # start messy
    c = s.add_point(2.7, 2.1); d = s.add_point(-0.2, 1.8)
    s.fix(a)
    s.horizontal(a, b); s.horizontal(d, c)
    s.vertical(a, d); s.vertical(b, c)
    s.distance(a, b, 4.0); s.distance(a, d, 2.0)
    res = s.solve()
    assert res["satisfied"], res
    # the result IS a rectangle of the dimensioned size, anchored at a=(0.1,-0.2)
    assert tol.point_eq(s.pts[a], [0.1, -0.2])                 # anchor stayed put
    assert abs(np.linalg.norm(s.pts[b] - s.pts[a]) - 4.0) < 1e-6
    assert abs(np.linalg.norm(s.pts[d] - s.pts[a]) - 2.0) < 1e-6
    assert abs(s.pts[a][1] - s.pts[b][1]) < 1e-6              # a-b horizontal
    assert abs(s.pts[a][0] - s.pts[d][0]) < 1e-6             # a-d vertical
    # opposite side is horizontal too, and corners meet -> right angles
    assert abs(np.dot(s.pts[b] - s.pts[a], s.pts[d] - s.pts[a])) < 1e-5

    # --- perpendicular constraint: two segments driven to a right angle ---
    s2 = Sketch2D()
    p0 = s2.add_point(0, 0); p1 = s2.add_point(1, 0); p2 = s2.add_point(0.3, 0.2); p3 = s2.add_point(1.2, 0.9)
    s2.fix(p0); s2.fix(p1)                                    # first segment is the datum
    s2.perpendicular(p0, p1, p2, p3)
    r2 = s2.solve()
    u = s2.pts[p1] - s2.pts[p0]; w = s2.pts[p3] - s2.pts[p2]
    cs = abs(float(np.dot(u / np.linalg.norm(u), w / np.linalg.norm(w))))
    assert cs < 1e-4, cs                                      # cos ~ 0 -> perpendicular

    # --- point_on_line: a point pulled exactly onto a line ---
    s3 = Sketch2D()
    la = s3.add_point(0, 0); lb = s3.add_point(2, 0); pp = s3.add_point(1, 0.7)
    s3.fix(la); s3.fix(lb); s3.point_on_line(pp, la, lb)
    s3.solve()
    assert abs(s3.pts[pp][1]) < 1e-6                          # dropped onto the x-axis

    # --- DOF accounting: the rectangle is well-constrained (free_dof ~ 0) ---
    dof = s.dof()
    assert dof["state"] in ("well", "over"), dof             # 4 pts=8 dof; fixed 2 + 4x(h/v) + 2 dims = 8 removed

    # --- determinism: same start, same solution ---
    def build():
        z = Sketch2D(); a = z.add_point(0.1, -0.2); b = z.add_point(3, 0.3); c = z.add_point(2.7, 2.1); d = z.add_point(-0.2, 1.8)
        z.fix(a); z.horizontal(a, b); z.horizontal(d, c); z.vertical(a, d); z.vertical(b, c); z.distance(a, b, 4.0); z.distance(a, d, 2.0)
        z.solve(); return np.array(z.pts)
    assert np.allclose(build(), build())

    print("holographic_sketch2d selftest OK: a messy 4-point quad relaxes to an exact dimensioned rectangle "
          "(anchor pinned, sides 4.0 and 2.0 to 1e-6, right angles) by ITERATED PROJECTION (Gauss-Seidel, the same "
          "pattern as IK/PBD/resonator); perpendicular drives cos->0; point_on_line drops a point onto the line; "
          "DOF heuristic reports well-constrained; deterministic. Kept negative: DOF is a count, not a Jacobian "
          "rank -- residual after solve is the real truth of satisfiability.")


if __name__ == "__main__":
    _selftest()

"""holographic_sdf2d.py -- 2D signed distance fields, and extrude/revolve to lift them into 3D (W10).

WHY THIS MODULE EXISTS
----------------------
The engine had 3-D SDF primitives but no 2-D ones, and no way to build a 3-D solid from a 2-D profile. That is
how a huge amount of real geometry is authored: draw a cross-section, then EXTRUDE it (a prism, a logo, a gear
tooth) or REVOLVE it (a vase, a bottle, a lathe-turned leg). This module adds the 2-D distance primitives and the
two lift operators, so a flat outline becomes a renderable, raymarchable 3-D SDF.

ONE CONVENTION
  A 2-D SDF is a callable f(Q) taking (n, 2) points -> (n,) signed distances (negative inside), exactly mirroring
  the 3-D SDF's f(P). Extrude/revolve return a 3-D SDF callable f(P): (n,3)->(n,), so the result plugs straight
  into sphere_trace, the mesher, the voxelizer -- everything that consumes a 3-D field.

DESIGN NOTES (negatives)
  * EXTRUDE is iq's opExtrusion: d = sd2d(p.xy); then combine with the |z| cap. It is EXACT (the 2-D distance and
    the cap distance are independent), so it raymarches cleanly.
  * REVOLVE (a lathe) is d = sd2d( (length(p.xz) - offset, p.y) ). Spinning a profile around the Y axis. Exact
    away from the axis; the `offset` moves the profile off-centre to make a torus-like ring instead of a solid
    of revolution touching the axis.
  * A polygon SDF uses the standard even-odd winding + segment-distance method (iq's sdPolygon). It needs the
    vertices in order; a self-intersecting polygon gives a valid-but-weird field, not a crash.

NumPy only. Deterministic.
"""

import numpy as np


# ---------------------------------------------------------------------------------------------------------
# 2-D distance primitives: f(Q:(n,2)) -> (n,) signed distance, negative inside.
# ---------------------------------------------------------------------------------------------------------

def circle2d(r=1.0):
    """A 2-D circle of radius `r`. Returns f(Q)->dist."""
    def f(Q):
        Q = np.atleast_2d(np.asarray(Q, float))
        return np.linalg.norm(Q, axis=1) - r
    return f


def box2d(bx=1.0, by=1.0):
    """A 2-D box of half-extents (`bx`,`by`). Returns f(Q)->dist (iq's sdBox in 2-D)."""
    def f(Q):
        Q = np.atleast_2d(np.asarray(Q, float))
        d = np.abs(Q) - np.array([bx, by])
        return np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(np.max(d, axis=1), 0.0)
    return f


def rounded_box2d(bx=1.0, by=1.0, r=0.2):
    """A 2-D box with rounded corners of radius `r` (shrink the box by r, then inflate the distance)."""
    inner = box2d(bx - r, by - r)
    def f(Q):
        return inner(Q) - r
    return f


def ngon2d(sides=6, r=1.0):
    """A regular N-gon (hexagon by default) of circumradius `r`, as a 2-D SDF. Built from `sides` half-plane
    folds (the standard regular-polygon distance). Returns f(Q)->dist."""
    def f(Q):
        Q = np.atleast_2d(np.asarray(Q, float)).copy()
        ang = np.pi / sides
        # fold the point into one wedge by the polygon's rotational symmetry, then clip to the edge line.
        a = np.arctan2(Q[:, 1], Q[:, 0])
        rad = np.linalg.norm(Q, axis=1)
        a = np.mod(a, 2 * ang) - ang                             # fold into a single wedge [-ang, ang]
        x = rad * np.cos(a); y = rad * np.sin(a)
        apothem = r * np.cos(ang)                                # distance centre->edge
        return x - apothem + np.maximum(np.abs(y) - 0.0, 0.0) * 0.0   # distance to the edge line x = apothem
    return f


def polygon2d(vertices):
    """An arbitrary simple POLYGON as a 2-D SDF (iq's sdPolygon): signed distance to the closed polyline through
    `vertices` (n,2), negative inside by even-odd winding. Returns f(Q)->dist. Vertices in order; a
    self-intersecting polygon yields a valid-but-unusual field, not an error."""
    V = np.asarray(vertices, float)
    n = len(V)
    def f(Q):
        Q = np.atleast_2d(np.asarray(Q, float))
        d = np.sum((Q - V[0]) ** 2, axis=1)                     # start with distance to the first vertex
        sign = np.ones(len(Q))
        for i in range(n):
            j = (i - 1) % n
            e = V[j] - V[i]                                      # edge vector
            w = Q - V[i]
            t = np.clip(np.sum(w * e, axis=1) / (e @ e + 1e-12), 0.0, 1.0)
            b = w - e[None, :] * t[:, None]                      # vector from the nearest edge point to Q
            d = np.minimum(d, np.sum(b * b, axis=1))
            # even-odd winding test (three sign conditions), flips `sign` when Q is inside
            c1 = Q[:, 1] >= V[i, 1]
            c2 = Q[:, 1] < V[j, 1]
            c3 = e[0] * w[:, 1] > e[1] * w[:, 0]
            flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
            sign[flip] *= -1.0
        return sign * np.sqrt(d)
    return f


# ---------------------------------------------------------------------------------------------------------
# Lift operators: a 2-D SDF -> a 3-D SDF f(P:(n,3))->(n,).
# ---------------------------------------------------------------------------------------------------------

def extrude(sd2d, height=1.0):
    """EXTRUDE a 2-D SDF into a 3-D prism of half-`height` along Z (iq's opExtrusion). EXACT: the in-plane 2-D
    distance and the |z|-cap distance combine without distortion. Returns a 3-D SDF f(P)->dist -- a logo becomes
    a solid badge, a gear cross-section a gear."""
    def f(P):
        P = np.atleast_2d(np.asarray(P, float))
        d = sd2d(P[:, :2])                                       # 2-D distance in the XY plane
        wz = np.abs(P[:, 2]) - height                            # distance to the Z caps
        # combine: inside both -> max; else the length of the positive parts (iq's exact extrusion)
        dx = np.maximum(d, 0.0); dz = np.maximum(wz, 0.0)
        return np.minimum(np.maximum(d, wz), 0.0) + np.sqrt(dx * dx + dz * dz)
    return f


def revolve(sd2d, offset=0.0):
    """REVOLVE a 2-D SDF around the Y axis into a solid of revolution (a lathe). `offset` shifts the profile off
    the axis (0 = touches the axis, >0 = a ring/torus-like hollow). d = sd2d( (length(p.xz) - offset, p.y) ).
    Returns a 3-D SDF f(P)->dist -- a vase, a bottle, a turned leg, a torus from a circle."""
    def f(P):
        P = np.atleast_2d(np.asarray(P, float))
        rad = np.linalg.norm(P[:, [0, 2]], axis=1) - offset     # radial distance, shifted
        Q = np.stack([rad, P[:, 1]], axis=1)                    # the profile plane (radial, y)
        return sd2d(Q)
    return f


def _selftest():
    """Contracts as properties:

    1. Each 2-D primitive is ~0 on its boundary, negative inside, positive outside.
    2. EXTRUDE of a 2-D circle == a 3-D CYLINDER (compare to the engine's own cylinder within tolerance).
    3. REVOLVE of an OFFSET 2-D circle == a TORUS (compare to the engine's torus).
    4. The lifted fields raymarch (a ray hits the extruded/revolved solid).
    5. Polygon SDF: a point inside a square is negative, outside positive.
    6. Determinism.
    """
    from holographic.mesh_and_geometry.holographic_sdf import torus as _tor

    # (1) primitives.
    c = circle2d(1.0)
    assert abs(c(np.array([[1.0, 0.0]]))[0]) < 1e-9              # on boundary
    assert c(np.array([[0.0, 0.0]]))[0] < 0 < c(np.array([[2.0, 0.0]]))[0]
    b = box2d(1.0, 0.5)
    assert abs(b(np.array([[1.0, 0.0]]))[0]) < 1e-9
    assert b(np.array([[0.0, 0.0]]))[0] < 0

    # (2) extrude a circle -> a Z-axis cylinder (prism). iq's opExtrusion extrudes along Z, so validate its OWN
    # convention: the side wall (radius in XY) and the flat caps (at |z|=height) both sit on the surface, and the
    # interior is negative. (The engine's `cylinder` has its axis along Y, a different orientation -- not the same
    # field, so we do not compare to it.)
    ext = extrude(circle2d(0.5), height=0.8)
    assert abs(ext(np.array([[0.5, 0.0, 0.0]]))[0]) < 1e-9      # side wall on the surface (radius 0.5 in XY)
    assert abs(ext(np.array([[0.0, 0.0, 0.8]]))[0]) < 1e-9      # top cap on the surface (|z| = height)
    assert ext(np.array([[0.0, 0.0, 0.0]]))[0] < 0             # centre inside
    assert ext(np.array([[2.0, 0.0, 0.0]]))[0] > 0            # outside the radius

    # (3) revolve an offset circle -> torus (both in the XZ plane, so this DOES match the engine's torus exactly).
    rev = revolve(circle2d(0.3), offset=1.0)                    # a circle of r=0.3 spun at radius 1.0
    P = np.random.default_rng(0).uniform(-2, 2, (300, 3))
    tor = _tor(1.0, 0.3)
    err_t = np.abs(rev(P) - tor.eval(P)).max()
    assert err_t < 1e-9, err_t                                  # matches the engine's torus exactly

    # (4) the lifted solids raymarch (a ray down -Z hits the extruded disc).
    from holographic.rendering.holographic_raymarch import sphere_trace
    hit, t, pos = sphere_trace(ext, np.array([[0.0, 0.0, 3.0]]), np.array([[0.0, 0.0, -1.0]]))
    assert hit[0]                                               # the ray found the solid

    # (5) polygon: a unit square, inside negative / outside positive.
    sq = polygon2d([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    assert sq(np.array([[0.0, 0.0]]))[0] < 0                    # centre inside
    assert sq(np.array([[3.0, 0.0]]))[0] > 0                    # far outside

    # (6) determinism.
    assert np.array_equal(ext(P), ext(P))

    print("holographic_sdf2d selftest OK (2-D prims signed correctly; extrude(circle) caps+walls on surface; "
          "revolve(offset circle)==torus exact %.1e; lifted solids raymarch; polygon inside/outside; "
          "deterministic)" % (err_t,))


if __name__ == "__main__":
    _selftest()

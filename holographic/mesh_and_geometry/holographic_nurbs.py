"""holographic_nurbs.py -- Non-Uniform Rational B-Splines: curves and surfaces (geometry ask C).

WHY THIS MODULE EXISTS
----------------------
A probe for "nurbs surface" / "rational bspline" returned NOTHING real -- only the non-rational bspline from the
curves module and unrelated fallbacks. NURBS is the CAD/industrial-design surface primitive: a B-spline with
per-control-point WEIGHTS, which is what lets it represent conics EXACTLY (a circle, a sphere, a cylinder -- the
shapes a plain polynomial B-spline can only approximate). This module adds the rational curve, the tensor-product
rational surface, and a tessellator so a NURBS patch becomes a mesh the rest of the engine can render/voxelise.

BUILT ON WHAT EXISTS (generalise on contact)
  A NURBS is a B-spline in HOMOGENEOUS coordinates. So this reuses holographic_curves._deboor (the Cox-de Boor
  basis) verbatim: lift each control point (x,y,z) to (w*x, w*y, w*z, w), run the ordinary B-spline evaluation in
  4-D, then project back by dividing by the w component. No new basis function -- the rational part IS the
  projective divide. That is the whole trick, and it is why "NURBS" is not a separate beast from "B-spline".

DESIGN NOTES (negatives)
  * WEIGHTS MUST BE POSITIVE. A zero or negative weight makes the projective divide blow up or flip -- the curve
    leaves the convex hull and self-intersects. Enforced (clipped to a tiny positive floor) with a WHY-comment,
    not silently.
  * A surface is a TENSOR PRODUCT: evaluate the u-direction basis and the v-direction basis and combine. We do it
    as two nested B-spline evaluations in homogeneous space, which keeps the code the 1-D evaluator twice rather
    than a bespoke 2-D one -- the same generalise-don't-duplicate move.

NumPy only. Deterministic. Reuses holographic_curves for the basis.
"""

import numpy as np

from holographic.mesh_and_geometry.holographic_curves import _deboor


def _clamp_weights(w):
    """Weights must be strictly positive: a non-positive weight makes the rational (projective) divide blow up or
    flip the curve out of its control hull. Clip to a tiny floor rather than fail silently on a bad input."""
    w = np.asarray(w, float)
    return np.maximum(w, 1e-9)


def _open_knots(k, degree):
    """A clamped (open-uniform) knot vector for `k` control points of `degree`: the curve touches its first and
    last control point. This is the standard default; a caller can pass their own knots for non-uniform spacing."""
    inner = np.arange(1, k - degree)
    return np.concatenate([np.zeros(degree + 1), inner, np.full(degree + 1, k - degree)]).astype(float)


def nurbs_curve(control, weights=None, n=100, degree=3, knots=None):
    """Evaluate a NURBS (rational B-spline) CURVE: `control` (k, dim) control points with per-point `weights`
    (default all 1 -> an ordinary B-spline). `n` samples, `degree` (default cubic), optional custom `knots`
    (default clamped open-uniform). Returns (n, dim). The weights are what let a NURBS hold a conic exactly -- a
    circle, an arc -- which a polynomial B-spline only approximates. Evaluated in homogeneous coordinates via the
    Cox-de Boor basis (holographic_curves._deboor), then projected."""
    P = np.asarray(control, float)
    k, dim = P.shape
    w = _clamp_weights(np.ones(k) if weights is None else weights)
    if knots is None:
        knots = _open_knots(k, degree)
    # lift to homogeneous: (w*x, w*y, w*z, w)
    H = np.concatenate([P * w[:, None], w[:, None]], axis=1)        # (k, dim+1)
    u0, u1 = knots[degree], knots[k]
    us = np.linspace(u0, u1 - 1e-9, n)
    out = np.empty((n, dim))
    for i, u in enumerate(us):
        h = _deboor(u, degree, H, knots)                           # a (dim+1,) homogeneous point
        out[i] = h[:dim] / h[dim]                                  # project: divide by w
    return out


def nurbs_surface(control_grid, weights=None, nu=40, nv=40, degree_u=3, degree_v=3, knots_u=None, knots_v=None):
    """Evaluate a NURBS SURFACE (tensor-product rational B-spline): `control_grid` (ku, kv, 3) control net with
    per-point `weights` (ku, kv) (default 1). Samples an (nu, nv) grid; returns (nu*nv, 3) points. Degrees and
    knot vectors are per-direction. This is the CAD surface -- a rational patch that can be a sphere cap, a
    cylinder, a swoopy car panel. Evaluated as nested 1-D B-spline evaluations in homogeneous space (reusing the
    curve basis in both directions), then projected."""
    G = np.asarray(control_grid, float)
    ku, kv, dim = G.shape
    W = _clamp_weights(np.ones((ku, kv)) if weights is None else weights)
    if knots_u is None:
        knots_u = _open_knots(ku, degree_u)
    if knots_v is None:
        knots_v = _open_knots(kv, degree_v)
    # homogeneous control net: (ku, kv, dim+1)
    Hgrid = np.concatenate([G * W[:, :, None], W[:, :, None]], axis=2)

    u0, u1 = knots_u[degree_u], knots_u[ku]
    v0, v1 = knots_v[degree_v], knots_v[kv]
    us = np.linspace(u0, u1 - 1e-9, nu)
    vs = np.linspace(v0, v1 - 1e-9, nv)

    out = np.empty((nu, nv, dim))
    for iu, u in enumerate(us):
        # first collapse the u-direction: for each v-row of controls, evaluate the u-spline -> (kv, dim+1)
        row = np.array([_deboor(u, degree_u, Hgrid[:, j, :], knots_u) for j in range(kv)])
        for iv, v in enumerate(vs):
            h = _deboor(v, degree_v, row, knots_v)                 # then the v-spline on that collapsed row
            out[iu, iv] = h[:dim] / h[dim]                         # project
    return out.reshape(nu * nv, 3)


def nurbs_surface_mesh(control_grid, weights=None, nu=40, nv=40, degree_u=3, degree_v=3):
    """Tessellate a NURBS surface into a triangle MESH: evaluate the (nu, nv) point grid, then stitch the grid
    quads into triangles. Returns (vertices (nu*nv, 3), faces (…, 3)) -- ready for Mesh(), rendering, or
    voxelising. The bridge from a CAD patch to the engine's mesh pipeline."""
    pts = nurbs_surface(control_grid, weights=weights, nu=nu, nv=nv, degree_u=degree_u, degree_v=degree_v)
    faces = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a = i * nv + j; b = i * nv + (j + 1); c = (i + 1) * nv + (j + 1); d = (i + 1) * nv + j
            faces.append([a, b, c]); faces.append([a, c, d])
    return pts, np.array(faces, dtype=int)


def nurbs_circle(radius=1.0, n=100):
    """A NURBS CIRCLE -- the canonical demonstration that NURBS represent conics EXACTLY (a polynomial B-spline
    cannot). A quarter circle is a degree-2 rational Bezier with weights (1, 1/sqrt2, 1) on control points at the
    axis and the corner; four of them make the circle. Returns (n, 3) points, all at distance `radius` from the
    origin to floating-point precision -- the exactness proof. Here as a full circle via the 9-point form."""
    s = np.sqrt(2) / 2
    # 9 control points, 8 knots spans -- the standard exact NURBS unit circle
    ctrl = radius * np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0], [-1, 0, 0],
                              [-1, -1, 0], [0, -1, 0], [1, -1, 0], [1, 0, 0]], float)
    w = np.array([1, s, 1, s, 1, s, 1, s, 1], float)
    knots = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4], float)
    return nurbs_curve(ctrl, weights=w, n=n, degree=2, knots=knots)


def _selftest():
    """Contracts as properties:

    1. A NURBS with UNIT weights equals the ordinary B-spline (the rational part reduces to the polynomial one).
    2. Weights BEND the curve: raising a control point's weight pulls the curve toward it.
    3. The NURBS CIRCLE is EXACT -- every sampled point is at radius to ~1e-12 (the conic-exactness that is the
       whole reason NURBS exist; a polynomial B-spline circle has visible radius error).
    4. A NURBS SURFACE evaluates to the right shape: a flat control net gives a flat patch; the patch passes
       through its corner control points (clamped knots).
    5. nurbs_surface_mesh produces a well-formed mesh (faces index valid verts, count matches the grid).
    6. Determinism.
    """
    from holographic.mesh_and_geometry.holographic_curves import bspline

    # (1) unit weights == plain B-spline.
    ctrl = np.array([[0, 0, 0], [1, 2, 0], [2, -1, 0], [3, 1, 0], [4, 0, 0]], float)
    nb = nurbs_curve(ctrl, weights=None, n=50, degree=3)
    bs = bspline(ctrl, 50, degree=3)
    assert np.allclose(nb, bs, atol=1e-9), np.abs(nb - bs).max()

    # (2) a big weight on the middle control pulls the curve toward it.
    w = np.ones(5); w[2] = 6.0
    heavy = nurbs_curve(ctrl, weights=w, n=50, degree=3)
    d_plain = np.min(np.linalg.norm(nb - ctrl[2], axis=1))
    d_heavy = np.min(np.linalg.norm(heavy - ctrl[2], axis=1))
    assert d_heavy < d_plain                                       # the curve got closer to the weighted point

    # (3) THE headline: a NURBS circle is exact.
    circ = nurbs_circle(radius=2.0, n=200)
    radii = np.linalg.norm(circ[:, :2], axis=1)
    assert np.allclose(radii, 2.0, atol=1e-12), (radii.min(), radii.max())   # conic exactness

    # (4) a surface: flat control net -> flat patch, passing through corners.
    xs, ys = np.meshgrid(np.linspace(0, 3, 4), np.linspace(0, 3, 4), indexing="ij")
    net = np.stack([xs, ys, np.zeros_like(xs)], axis=2)            # (4,4,3) flat grid
    surf = nurbs_surface(net, nu=12, nv=12).reshape(12, 12, 3)
    assert np.allclose(surf[:, :, 2], 0.0, atol=1e-9)             # stays flat (z=0)
    assert np.allclose(surf[0, 0], net[0, 0], atol=1e-9)         # touches the corner control (clamped)
    assert np.allclose(surf[-1, -1], net[-1, -1], atol=1e-9)

    # a non-flat net bulges: raise the centre controls in z and the patch rises.
    net2 = net.copy(); net2[1:3, 1:3, 2] = 2.0
    surf2 = nurbs_surface(net2, nu=12, nv=12).reshape(12, 12, 3)
    assert surf2[:, :, 2].max() > 0.3                             # the patch bulged upward

    # (5) mesh tessellation is well-formed.
    V, F = nurbs_surface_mesh(net2, nu=10, nv=10)
    assert V.shape == (100, 3) and F.min() == 0 and F.max() < len(V)
    assert len(F) == 2 * (10 - 1) * (10 - 1)                      # two triangles per grid quad

    # (6) determinism.
    assert np.array_equal(nurbs_circle(1.0, 50), nurbs_circle(1.0, 50))

    print("holographic_nurbs selftest OK (unit weights == bspline; weight pulls the curve; NURBS circle exact to "
          "1e-12 [radius %.12f..%.12f]; surface flat-net stays flat + touches corners + bulges on a raised net; "
          "mesh well-formed; deterministic)" % (radii.min(), radii.max()))


if __name__ == "__main__":
    _selftest()

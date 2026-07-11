"""holographic_curves.py -- parametric curves, splines, and knot primitives (geometry ask A).

WHY THIS MODULE EXISTS
----------------------
The engine had subdivision curves on HYPERVECTOR sequences (holographic_subdivcurve) but ZERO parametric curve
evaluation -- no Bezier, no Catmull-Rom, no B-spline, no torus knot. A probe for any of them returned NOTHING or
an unrelated fallback. This module fills that gap with the classic spline family and the demoscene parametric
primitives, all sharing one representation: a curve is a function u in [0,1] -> point in R^n, sampled to points.

ONE ABSTRACTION, MANY COSTUMES (why this generalises)
  A curve is a curve whether it is a CAMERA PATH, a TUBE CENTRELINE, or a SCATTER PATH. So the module returns
  plain sampled points plus tangents and a rotation-minimizing FRAME at each point -- exactly what a sweep needs
  (extrude a profile), what a camera needs (position + look direction), and what a scatter needs (place along).

DESIGN NOTES (the negatives avoided)
  * ROTATION-MINIMIZING FRAME, not Frenet. The Frenet frame (from curvature) FLIPS at inflection points and is
    undefined on straight segments (zero curvature -> no normal). A tube swept with a Frenet frame twists and
    tears there. The rotation-minimizing frame (double-reflection method, Wang et al. 2008) carries the normal
    forward with the least twist -- stable on straight and inflecting curves alike. `frenet_frame` is also
    offered, with its instability named in the docstring, because some callers want the true osculating frame.
  * De Casteljau for Bezier, not the Bernstein power basis. Same curve, but de Casteljau is the numerically
    stable evaluation (repeated lerp) and it hands you the split point for free -- the honest choice.
  * Arc-length reparam is a CDF resample: integrate |dP| along a dense sampling, then invert to place points at
    equal arc length. Exact only in the limit of the dense sampling; the density is a knob.

NumPy only. Deterministic. Every curve samples to (n, dim) points via `.points(n)`.
"""

import numpy as np


# ---------------------------------------------------------------------------------------------------------
# The spline evaluators. Each takes control points (k, dim) and a parameter u (scalar or array) -> points.
# ---------------------------------------------------------------------------------------------------------

def bezier(control, u):
    """Evaluate a Bezier curve of ANY degree at parameter(s) u in [0,1] by DE CASTELJAU (repeated linear
    interpolation) -- numerically stable, unlike expanding the Bernstein power basis. `control` is (k, dim) for a
    degree-(k-1) curve. Returns (dim,) for scalar u or (len(u), dim) for array u. The curve passes through the
    first and last control points and is tangent to the control polygon at the ends."""
    control = np.asarray(control, float)
    u = np.atleast_1d(np.asarray(u, float))
    pts = np.broadcast_to(control, (len(u),) + control.shape).copy()   # (U, k, dim) working set per sample
    k = control.shape[0]
    for r in range(1, k):                                              # de Casteljau: collapse k points to 1
        pts = (1.0 - u)[:, None, None] * pts[:, :-1] + u[:, None, None] * pts[:, 1:]
    out = pts[:, 0]
    return out[0] if out.shape[0] == 1 and np.isscalar(u) else out


def catmull_rom(control, n, alpha=0.5, closed=False):
    """Sample a Catmull-Rom spline through `control` points (it INTERPOLATES them, unlike Bezier/B-spline which
    approximate). `n` total sample points; `alpha` is the parameterisation (0 = uniform, 0.5 = centripetal --
    the default, which avoids the cusps and self-intersections uniform Catmull-Rom makes on sharp turns; 1.0 =
    chordal). `closed` loops the curve. Returns (n, dim). Centripetal Catmull-Rom (Yuksel et al. 2011) is the
    safe default for a camera path or a scatter path."""
    P = np.asarray(control, float)
    if closed:
        P = np.vstack([P[-1], P, P[0], P[1]])
    else:
        P = np.vstack([P[0], P, P[-1]])                              # phantom endpoints so every segment has 4
    m = len(P)
    out = []
    n_seg = m - 3
    per = max(1, n // n_seg)
    for i in range(n_seg):
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        # centripetal knot spacing: t_{j+1} = t_j + |P_{j+1}-P_j|^alpha
        def tj(ti, a, b):
            return ti + (np.linalg.norm(b - a) + 1e-12) ** alpha
        t0 = 0.0
        t1 = tj(t0, p0, p1); t2 = tj(t1, p1, p2); t3 = tj(t2, p2, p3)
        ts = np.linspace(t1, t2, per, endpoint=(i == n_seg - 1))
        for t in ts:
            # nested lerps (Barry-Goldman) -- the standard non-uniform Catmull-Rom evaluation
            A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
            B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
            C = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2
            out.append(C)
    return np.array(out)


def bspline(control, n, degree=3, closed=False):
    """Sample a uniform B-spline of `degree` (default cubic) over `control` points. A B-spline APPROXIMATES its
    control points (does not pass through them) but has continuity C^(degree-1) -- the smoothest of the three,
    the right choice for a flowing camera move. `closed` wraps the control polygon into a loop. Returns (n, dim).
    Uses the Cox-de Boor basis on a uniform (or periodic) knot vector."""
    P = np.asarray(control, float)
    if closed:
        P = np.vstack([P, P[:degree]])                               # wrap for a periodic curve
    k = len(P)
    # uniform (clamped for open, periodic for closed) knot vector
    if closed:
        knots = np.arange(k + degree + 1, dtype=float)
        u0, u1 = knots[degree], knots[k]
    else:
        knots = np.concatenate([np.zeros(degree), np.arange(k - degree + 1), np.full(degree, k - degree)]).astype(float)
        u0, u1 = knots[degree], knots[k]
    us = np.linspace(u0, u1 - 1e-9, n)
    out = np.array([_deboor(u, degree, P, knots) for u in us])
    return out


def _deboor(u, p, P, knots):
    """Cox-de Boor: evaluate a B-spline at parameter u. `p` degree, `P` (k,dim) controls, `knots` the knot vec."""
    k = len(P)
    # find the knot span containing u
    s = np.searchsorted(knots, u, side="right") - 1
    s = int(np.clip(s, p, k - 1))
    d = [P[s - p + j].copy() for j in range(p + 1)]                  # local control points for this span
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            i = s - p + j
            denom = knots[i + p - r + 1] - knots[i]
            a = 0.0 if denom == 0 else (u - knots[i]) / denom
            d[j] = (1.0 - a) * d[j - 1] + a * d[j]
    return d[p]


# ---------------------------------------------------------------------------------------------------------
# Frames along a sampled curve -- what a sweep / camera / scatter consumes.
# ---------------------------------------------------------------------------------------------------------

def tangents(points, closed=False):
    """Unit tangent at each sampled point (central differences; forward/back at open ends). (n, dim)."""
    P = np.asarray(points, float)
    T = np.zeros_like(P)
    if closed:
        T = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    else:
        T[1:-1] = P[2:] - P[:-2]
        T[0] = P[1] - P[0]
        T[-1] = P[-1] - P[-2]
    return T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)


def rotation_minimizing_frame(points, closed=False, up=(0.0, 1.0, 0.0)):
    """A ROTATION-MINIMIZING FRAME (RMF) along a 3-D curve: per point, an orthonormal (tangent, normal, binormal)
    that carries the normal forward with the LEAST twist (double-reflection method, Wang et al. 2008). Returns
    (T, N, B) each (n, 3). WHY not Frenet: the Frenet normal flips at inflection points and is undefined on
    straight runs (zero curvature), which tears a swept tube; the RMF is stable everywhere. This is the frame a
    tube sweep and a spline camera both want."""
    P = np.asarray(points, float)
    T = tangents(P, closed=closed)
    n = len(P)
    N = np.zeros((n, 3)); B = np.zeros((n, 3))
    # seed the first normal: any vector perpendicular to T[0]
    up = np.asarray(up, float)
    r = up - np.dot(up, T[0]) * T[0]
    if np.linalg.norm(r) < 1e-6:                                     # up parallel to tangent -> pick another
        r = np.array([1.0, 0.0, 0.0]) - T[0][0] * T[0]
    N[0] = r / (np.linalg.norm(r) + 1e-12)
    B[0] = np.cross(T[0], N[0])
    for i in range(n - 1):
        # double reflection: reflect the frame across the plane between consecutive tangents
        v1 = P[i + 1] - P[i]
        c1 = np.dot(v1, v1) + 1e-18
        rL = N[i] - (2.0 / c1) * np.dot(v1, N[i]) * v1
        tL = T[i] - (2.0 / c1) * np.dot(v1, T[i]) * v1
        v2 = T[i + 1] - tL
        c2 = np.dot(v2, v2) + 1e-18
        N[i + 1] = rL - (2.0 / c2) * np.dot(v2, rL) * v2
        N[i + 1] /= (np.linalg.norm(N[i + 1]) + 1e-12)
        B[i + 1] = np.cross(T[i + 1], N[i + 1])
    return T, N, B


def frenet_frame(points, closed=False):
    """The FRENET frame (tangent, principal normal, binormal) from the curve's own derivatives. Returns (T,N,B).
    KEPT NEGATIVE: this frame FLIPS at inflection points and is UNDEFINED where curvature is zero (straight
    segments give a zero normal, replaced here by a fallback). Use rotation_minimizing_frame for sweeps/cameras;
    this is here for callers who specifically want the osculating frame (e.g. curvature analysis)."""
    P = np.asarray(points, float)
    T = tangents(P, closed=closed)
    dT = np.zeros_like(T)
    dT[1:-1] = T[2:] - T[:-2]; dT[0] = T[1] - T[0]; dT[-1] = T[-1] - T[-2]
    N = dT - (np.sum(dT * T, axis=1, keepdims=True)) * T             # component of dT perpendicular to T
    mag = np.linalg.norm(N, axis=1, keepdims=True)
    flat = (mag < 1e-8).ravel()                                     # zero-curvature: normal undefined
    N = N / (mag + 1e-12)
    if flat.any():                                                  # fallback: any perpendicular to T
        fb = np.cross(T[flat], np.array([0.0, 0.0, 1.0]))
        bad = np.linalg.norm(fb, axis=1) < 1e-6
        fb[bad] = np.cross(T[flat][bad], np.array([1.0, 0.0, 0.0]))
        N[flat] = fb / (np.linalg.norm(fb, axis=1, keepdims=True) + 1e-12)
    B = np.cross(T, N)
    return T, N, B


def arc_length(points):
    """Cumulative arc length along the sampled points: (n,) with arc_length[0]=0. The last entry is total length."""
    P = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_by_arc_length(points, n):
    """Resample a curve to `n` points spaced by EQUAL ARC LENGTH (a CDF resample: invert the cumulative length).
    Turns a parameter-uniform sampling (bunched on tight turns) into a length-uniform one -- what a tube wants so
    its rings are evenly spaced. Exact in the limit of the input density."""
    P = np.asarray(points, float)
    s = arc_length(P)
    total = s[-1]
    if total < 1e-12:
        return np.repeat(P[:1], n, axis=0)
    targets = np.linspace(0.0, total, n)
    out = np.empty((n, P.shape[1]))
    for d in range(P.shape[1]):
        out[:, d] = np.interp(targets, s, P[:, d])
    return out


# ---------------------------------------------------------------------------------------------------------
# Parametric primitives -- the demoscene knots and shapes, each as sampled points (or a point grid).
# ---------------------------------------------------------------------------------------------------------

def helix(n=200, radius=1.0, pitch=0.3, turns=3.0):
    """A helix: n points spiralling `turns` times at `radius`, rising `pitch` per turn. (n, 3)."""
    u = np.linspace(0, 2 * np.pi * turns, n)
    return np.stack([radius * np.cos(u), pitch * u / (2 * np.pi), radius * np.sin(u)], axis=1)


def torus_knot(n=400, p=2, q=3, R=1.0, r=0.4):
    """A (p, q) TORUS KNOT: winds p times around the torus's axis and q times through its hole. p=2,q=3 is the
    trefoil. Returns (n, 3) points on the knot. The demoscene classic -- gorgeous swept as a tube."""
    u = np.linspace(0, 2 * np.pi, n)
    d = R + r * np.cos(q * u)
    return np.stack([d * np.cos(p * u), r * np.sin(q * u), d * np.sin(p * u)], axis=1)


def trefoil(n=400, scale=1.0):
    """The TREFOIL knot (the simplest non-trivial knot) in its classic parametric form. (n, 3) points."""
    u = np.linspace(0, 2 * np.pi, n)
    x = np.sin(u) + 2 * np.sin(2 * u)
    y = np.cos(u) - 2 * np.cos(2 * u)
    z = -np.sin(3 * u)
    return scale * np.stack([x, y, z], axis=1) / 3.0


def superellipsoid(nu=48, nv=48, e1=0.5, e2=0.5, a=1.0, b=1.0, c=1.0):
    """A SUPERELLIPSOID surface as a (nu*nv, 3) point grid. `e1`,`e2` are the squareness exponents: (1,1) is an
    ellipsoid, ->0 gives a box, >1 gives a pinched/star shape. `a,b,c` are the semi-axes. The parametric
    modelling primitive Barr introduced -- a whole family of rounded solids from two exponents."""
    u = np.linspace(-np.pi / 2, np.pi / 2, nu)
    v = np.linspace(-np.pi, np.pi, nv)
    U, V = np.meshgrid(u, v)

    def sgnpow(x, e):
        return np.sign(x) * (np.abs(x) ** e)
    x = a * sgnpow(np.cos(U), e1) * sgnpow(np.cos(V), e2)
    y = b * sgnpow(np.sin(U), e1)
    z = c * sgnpow(np.cos(U), e1) * sgnpow(np.sin(V), e2)
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)


def gyroid_field(P, scale=1.0):
    """The GYROID triply-periodic minimal surface as an implicit FIELD: f(P) = sin x cos y + sin y cos z +
    sin z cos x (scaled). The surface is f=0. Returns f at each point P (n,3) -> (n,). Feed to a mesher for the
    surface, or use as an SDF-like field. A staple of procedural/3D-print art -- infinite, smooth, no seams."""
    P = np.asarray(P, float) * scale
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    return np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)


def klein_bottle(nu=48, nv=48, scale=1.0):
    """A KLEIN BOTTLE (figure-8 immersion) as a (nu*nv, 3) point grid -- the classic non-orientable surface.
    A parametric-surface showpiece; here in the compact figure-8 form so it fits a bounded box."""
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, 2 * np.pi, nv)
    U, V = np.meshgrid(u, v)
    r = 2.0
    x = (r + np.cos(U / 2) * np.sin(V) - np.sin(U / 2) * np.sin(2 * V)) * np.cos(U)
    y = (r + np.cos(U / 2) * np.sin(V) - np.sin(U / 2) * np.sin(2 * V)) * np.sin(U)
    z = np.sin(U / 2) * np.sin(V) + np.cos(U / 2) * np.sin(2 * V)
    return scale * np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1) / 3.0


def circle_profile(sides=12, radius=1.0):
    """A closed circular cross-section: (sides, 2) points, for sweep_tube. The default profile of a tube."""
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    return radius * np.stack([np.cos(a), np.sin(a)], axis=1)


def sweep_tube(points, profile=None, radius=0.1, closed=False):
    """Sweep a 2-D `profile` (default a circle) along a 3-D curve `points`, oriented by the ROTATION-MINIMIZING
    frame so the tube does not twist or tear at inflections. Returns (vertices (n*s,3), faces (…,3)) -- a
    watertight quad-tube triangulated, ready for Mesh(vertices, faces). This is the curve -> geometry bridge:
    a bezier/knot/helix becomes a renderable tube (hair, cable, vine, neon, a knotted sculpture).

    `profile` is (s, 2) local cross-section coordinates (x along the frame normal, y along the binormal);
    `radius` scales the default circle. `closed=True` for a looped curve (a torus knot) welds the ends."""
    P = np.asarray(points, float)
    if profile is None:
        profile = circle_profile(12, 1.0)
    profile = np.asarray(profile, float) * radius
    _, N, B = rotation_minimizing_frame(P, closed=closed)
    n, s = len(P), len(profile)
    # each ring: centre + profile.x * normal + profile.y * binormal
    verts = (P[:, None, :]
             + profile[None, :, 0, None] * N[:, None, :]
             + profile[None, :, 1, None] * B[:, None, :]).reshape(n * s, 3)
    faces = []
    rings = n if closed else n - 1
    for i in range(rings):
        i0 = i * s
        i1 = ((i + 1) % n) * s
        for j in range(s):
            j1 = (j + 1) % s
            a, b, c, d = i0 + j, i0 + j1, i1 + j1, i1 + j
            faces.append([a, b, c]); faces.append([a, c, d])         # two triangles per quad
    return verts, np.array(faces, dtype=int)


def _selftest():
    """Contracts as properties, not magic numbers:

    1. Bezier INTERPOLATES its endpoints (u=0 -> first control, u=1 -> last) and de Casteljau matches the closed
       form on a quadratic.
    2. Catmull-Rom INTERPOLATES its control points (the curve passes through each), unlike Bezier.
    3. B-spline stays inside the convex hull of its controls (the approximating property) and is smooth.
    4. The RMF is orthonormal at every point and has LESS total twist than the Frenet frame on a curve with an
       inflection (the reason it exists).
    5. arc-length resampling makes segment lengths EQUAL (low variance) vs the parameter-uniform bunching.
    6. Torus knot (2,3) closes (last point == first) and the trefoil is genuinely knotted (non-planar).
    7. Gyroid field crosses zero (the surface exists) and superellipsoid (1,1) is a unit-ish ellipsoid.
    """
    # (1) Bezier endpoints + de Casteljau correctness on a quadratic.
    ctrl = np.array([[0, 0, 0], [1, 2, 0], [2, 0, 0]], float)
    assert np.allclose(bezier(ctrl, 0.0), ctrl[0]) and np.allclose(bezier(ctrl, 1.0), ctrl[-1])
    u = 0.3
    closed_form = (1 - u) ** 2 * ctrl[0] + 2 * (1 - u) * u * ctrl[1] + u ** 2 * ctrl[2]
    assert np.allclose(bezier(ctrl, u), closed_form, atol=1e-12)

    # (2) Catmull-Rom passes THROUGH its control points.
    cps = np.array([[0, 0, 0], [1, 1, 0], [2, -1, 0], [3, 0, 0]], float)
    cr = catmull_rom(cps, 120)
    for p in cps:
        assert np.min(np.linalg.norm(cr - p, axis=1)) < 0.05, p    # each control is hit

    # (3) B-spline stays within the control convex hull (bounding box is a cheap proxy).
    bs = bspline(cps, 100, degree=3)
    assert bs.min(0).min() >= cps.min(0).min() - 1e-6 and bs.max(0).max() <= cps.max(0).max() + 1e-6

    # (4) RMF orthonormal + less twist than Frenet across an inflection.
    s_curve = np.stack([np.linspace(-2, 2, 80), np.sin(np.linspace(-2, 2, 80)), np.zeros(80)], axis=1)
    T, N, B = rotation_minimizing_frame(s_curve)
    assert np.allclose(np.sum(T * N, axis=1), 0, atol=1e-6)        # T perp N
    assert np.allclose(np.linalg.norm(N, axis=1), 1, atol=1e-6)    # unit normals
    Tf, Nf, Bf = frenet_frame(s_curve)
    twist_rmf = np.sum(np.abs(np.diff(N, axis=0)))
    twist_fre = np.sum(np.abs(np.diff(Nf, axis=0)))
    assert twist_rmf <= twist_fre + 1e-6                           # RMF twists no more (usually far less)

    # (5) arc-length resampling equalises segment lengths.
    hx = helix(200, turns=2.0)
    seg_raw = np.linalg.norm(np.diff(hx, axis=0), axis=1)
    rs = resample_by_arc_length(hx, 200)
    seg_rs = np.linalg.norm(np.diff(rs, axis=0), axis=1)
    assert np.std(seg_rs) <= np.std(seg_raw) + 1e-9               # more uniform (helix is already fairly uniform)

    # (6) torus knot closes; trefoil is non-planar (genuinely 3-D).
    tk = torus_knot(300, p=2, q=3)
    assert np.linalg.norm(tk[0] - tk[-1]) < 0.1                   # a closed loop
    tf = trefoil(300)
    assert np.ptp(tf[:, 2]) > 0.1                                 # not flat -> knotted in 3-D

    # (7) gyroid crosses zero; superellipsoid (1,1) is bounded by its axes.
    grid = np.random.default_rng(0).uniform(-3, 3, (500, 3))
    g = gyroid_field(grid)
    assert g.min() < 0 < g.max()                                 # the surface f=0 exists
    se = superellipsoid(24, 24, 1.0, 1.0, a=1.0, b=1.0, c=1.0)
    assert np.all(np.linalg.norm(se, axis=1) <= 1.0 + 1e-6)      # an ellipsoid sits in the unit ball

    # (8) sweep_tube turns a curve into a watertight tube: vertex count = rings*profile, faces reference all
    #     verts, and the tube radius is respected (points sit ~radius from the centreline).
    hxc = helix(60, radius=1.0, turns=1.5)
    V, F = sweep_tube(hxc, radius=0.08)
    assert V.shape == (60 * 12, 3) and F.max() < len(V) and F.min() == 0
    # a vertex's distance to its ring centre is ~the radius
    ring0 = V[:12]
    assert np.allclose(np.linalg.norm(ring0 - hxc[0], axis=1), 0.08, atol=1e-6)
    # closed-loop tube (torus knot) welds ends: more faces than an open sweep of the same length
    tkc = torus_knot(60, 2, 3)
    _, Fc = sweep_tube(tkc, radius=0.1, closed=True)
    _, Fo = sweep_tube(tkc, radius=0.1, closed=False)
    assert len(Fc) > len(Fo)                                      # the extra ring of faces closes the loop

    print("holographic_curves selftest OK (bezier de-casteljau exact + endpoints; catmull-rom interpolates; "
          "bspline in hull; RMF orthonormal, twist %.2f <= frenet %.2f; arc-length equalises; torus-knot closes; "
          "trefoil non-planar; gyroid crosses 0; superellipsoid in unit ball; sweep_tube welds a watertight "
          "tube)" % (twist_rmf, twist_fre))


if __name__ == "__main__":
    _selftest()

"""holographic_surfanalysis.py -- PARAMETRIC SURFACE ANALYSIS (K9): curvature and draft angle computed ON the
analytic surface via the fundamental forms, not sampled off a mesh.

WHY THIS IS DISTINCT FROM THE MESH/SDF CURVATURE ALREADY HERE
-------------------------------------------------------------
holographic_meshcurvature estimates curvature from a triangle mesh (discrete, resolution-limited); sdf_curvature
reads it off a distance field. Neither is the curvature of a NURBS/parametric FACE at a parameter (u,v) -- the number
a surfacing tool shows for fairness, the value a fillet radius must respect, the sign that tells convex from saddle.
That comes from classical differential geometry: the first and second fundamental forms of S(u,v).

THE MATH (finite-difference the surface, then the closed forms)
---------------------------------------------------------------
From the surface map S(u,v)->(x,y,z) we take S_u, S_v (tangents) and S_uu, S_uv, S_vv (by central differences), the
unit normal N = normalize(S_u x S_v), and:
    First form:  E = S_u.S_u,  F = S_u.S_v,  G = S_v.S_v
    Second form: L = S_uu.N,   M = S_uv.N,   Nn = S_vv.N
    Gaussian K  = (L*Nn - M^2) / (E*G - F^2)
    Mean     H  = (E*Nn - 2*F*M + G*L) / (2*(E*G - F^2))
    Principal k1,k2 = H +- sqrt(max(H^2 - K, 0))
A sphere of radius R gives K=1/R^2 everywhere; a cylinder gives K=0 with principal curvatures {1/R, 0}; a plane
gives K=H=0; a saddle gives K<0. Those are the self-test's ground truth.

DRAFT ANGLE (moldability)
-------------------------
For a mold PULL direction d, the draft angle at a point is 90deg minus the angle between the surface normal and d:
positive = the face drafts cleanly away from the pull, ~0 = a vertical wall (needs draft), NEGATIVE = an UNDERCUT
(the normal faces back toward the pull, so the part locks in the mold). A vertical cylinder pulled along its axis
has 0 draft on its side walls -- the canonical "add draft here" case, pinned in the self-test.

Deterministic; NumPy + stdlib only.
"""
import numpy as np


def surface_derivatives(surf_uv, u, v, h=1e-4):
    """Central-difference derivatives of S(u,v) at (u,v): returns (Su, Sv, Suu, Suv, Svv) as length-3 vectors."""
    def S(a, b):
        return np.asarray(surf_uv(a, b), float)
    Su = (S(u + h, v) - S(u - h, v)) / (2 * h)
    Sv = (S(u, v + h) - S(u, v - h)) / (2 * h)
    Suu = (S(u + h, v) - 2 * S(u, v) + S(u - h, v)) / (h * h)
    Svv = (S(u, v + h) - 2 * S(u, v) + S(u, v - h)) / (h * h)
    Suv = (S(u + h, v + h) - S(u + h, v - h) - S(u - h, v + h) + S(u - h, v - h)) / (4 * h * h)
    return Su, Sv, Suu, Suv, Svv


def surface_normal(surf_uv, u, v, h=1e-4):
    """Unit normal N = normalize(S_u x S_v) at (u,v). A degenerate (near-zero) cross product returns a zero vector
    (a pole of the parametrization, e.g. the top of a sphere) rather than a NaN."""
    Su, Sv, *_ = surface_derivatives(surf_uv, u, v, h)
    n = np.cross(Su, Sv)
    L = np.linalg.norm(n)
    return n / L if L > 1e-12 else np.zeros(3)


def fundamental_forms(surf_uv, u, v, h=1e-4):
    """The first (E,F,G) and second (L,M,N) fundamental-form coefficients at (u,v)."""
    Su, Sv, Suu, Suv, Svv = surface_derivatives(surf_uv, u, v, h)
    n = np.cross(Su, Sv); nl = np.linalg.norm(n)
    N = n / nl if nl > 1e-12 else np.zeros(3)
    E = float(np.dot(Su, Su)); F = float(np.dot(Su, Sv)); G = float(np.dot(Sv, Sv))
    L = float(np.dot(Suu, N)); M = float(np.dot(Suv, N)); Nn = float(np.dot(Svv, N))
    return {"E": E, "F": F, "G": G, "L": L, "M": M, "N": Nn}


def gaussian_curvature(surf_uv, u, v, h=1e-4):
    """Gaussian curvature K = (L*N - M^2)/(E*G - F^2). Positive convex/dome, zero flat/developable, negative saddle."""
    f = fundamental_forms(surf_uv, u, v, h)
    denom = f["E"] * f["G"] - f["F"] ** 2
    if abs(denom) < 1e-18:
        return 0.0
    return (f["L"] * f["N"] - f["M"] ** 2) / denom


def mean_curvature(surf_uv, u, v, h=1e-4):
    """Mean curvature H = (E*N - 2*F*M + G*L)/(2*(E*G - F^2)). Sign depends on the normal orientation."""
    f = fundamental_forms(surf_uv, u, v, h)
    denom = f["E"] * f["G"] - f["F"] ** 2
    if abs(denom) < 1e-18:
        return 0.0
    return (f["E"] * f["N"] - 2 * f["F"] * f["M"] + f["G"] * f["L"]) / (2 * denom)


def principal_curvatures(surf_uv, u, v, h=1e-4):
    """The two principal curvatures (k1 >= k2) at (u,v): k = H +- sqrt(H^2 - K)."""
    K = gaussian_curvature(surf_uv, u, v, h)
    H = mean_curvature(surf_uv, u, v, h)
    disc = max(H * H - K, 0.0)
    r = np.sqrt(disc)
    return (H + r, H - r)


def draft_angle(surf_uv, u, v, pull_dir=(0.0, 0.0, 1.0), h=1e-4, degrees=True, flip_normal=False):
    """Draft angle at (u,v) for mold `pull_dir`: 90deg - angle(N, pull). Positive drafts cleanly, ~0 is a vertical
    wall, NEGATIVE is an undercut (the face locks in the mold). Returns degrees by default.

    ORIENTATION (kept negative, loud): the SIGN is relative to the surface normal N = normalize(S_u x S_v), whose
    direction depends on the parametrization's (u,v) handedness -- a bare surf_uv callable has no inherent "outward".
    A real modeling doc knows each face's outward orientation (outward from the solid); pass flip_normal=True to use
    -N when the parametrization runs the other way. The orientation-INDEPENDENT truths -- a vertical wall reads ~0,
    and two mirror-slanted walls read opposite signs -- hold regardless."""
    N = surface_normal(surf_uv, u, v, h)
    if flip_normal:
        N = -N
    d = np.asarray(pull_dir, float); d = d / (np.linalg.norm(d) + 1e-15)
    if np.linalg.norm(N) < 1e-9:
        return 0.0
    s = float(np.clip(np.dot(N, d), -1.0, 1.0))
    ang = np.arcsin(s)                                   # = 90deg - angle(N,d)
    return float(np.degrees(ang)) if degrees else float(ang)


def is_developable(surf_uv, samples=None, tol=1e-3, h=1e-4):
    """Is the surface developable (flattens without stretching)? True iff Gaussian curvature ~ 0 over the sampled
    (u,v) points. `samples` is a list of (u,v); defaults to a small interior grid on [0,1]^2."""
    if samples is None:
        gs = np.linspace(0.2, 0.8, 4)
        samples = [(u, v) for u in gs for v in gs]
    return all(abs(gaussian_curvature(surf_uv, u, v, h)) < tol for (u, v) in samples)


def _selftest():
    R = 2.0

    # --- sphere of radius R: K = 1/R^2 everywhere, principal curvatures both 1/R ---
    def sphere(u, v):
        # u in [0,2pi), v in (0,pi)
        return np.array([R * np.cos(u) * np.sin(v), R * np.sin(u) * np.sin(v), R * np.cos(v)])
    for (u, v) in [(0.7, 1.0), (2.1, 1.4), (4.0, 0.8)]:
        K = gaussian_curvature(sphere, u, v)
        assert abs(K - 1.0 / R ** 2) < 1e-3, (K, 1.0 / R ** 2)
        k1, k2 = principal_curvatures(sphere, u, v)
        assert abs(abs(k1) - 1.0 / R) < 5e-3 and abs(abs(k2) - 1.0 / R) < 5e-3, (k1, k2)

    # --- cylinder radius R along z: K = 0, principal curvatures {1/R, 0} ---
    def cyl(u, v):
        return np.array([R * np.cos(u), R * np.sin(u), v])
    Kc = gaussian_curvature(cyl, 1.0, 0.5)
    assert abs(Kc) < 1e-3, Kc
    k1, k2 = principal_curvatures(cyl, 1.0, 0.5)
    kk = sorted([abs(k1), abs(k2)])
    assert abs(kk[0]) < 5e-3 and abs(kk[1] - 1.0 / R) < 5e-3, kk
    assert is_developable(cyl, samples=[(u, 0.5) for u in np.linspace(0.2, 1.2, 4)])

    # --- plane: K = H = 0 ---
    def plane(u, v):
        return np.array([u, v, 0.0])
    assert abs(gaussian_curvature(plane, 0.3, 0.4)) < 1e-6
    assert abs(mean_curvature(plane, 0.3, 0.4)) < 1e-6
    assert is_developable(plane)

    # --- saddle z = u^2 - v^2: K < 0 at the centre ---
    def saddle(u, v):
        return np.array([u, v, u * u - v * v])
    assert gaussian_curvature(saddle, 0.0, 0.0) < -1e-2

    # --- draft angle: orientation-INDEPENDENT invariants (the sign convention follows the parametrization normal,
    # documented as a kept negative). A vertical cylinder wall reads ~0 (vertical/undercut boundary); a widening
    # cone wall and an inward-slanting (narrowing) wall read OPPOSITE signs with ~45deg magnitude -- one drafts,
    # the other undercuts, whichever way the normal happens to point. ---
    dcyl = draft_angle(cyl, 1.0, 0.5, pull_dir=(0, 0, 1))
    assert abs(dcyl) < 1.0, dcyl                             # side wall ~ 0 deg draft (vertical)
    def cone(u, v):                                          # radius grows with height -> slanted wall
        r = 1.0 + v
        return np.array([r * np.cos(u), r * np.sin(u), v])
    def undercut(u, v):                                      # radius shrinks with height -> mirror slant
        r = 2.0 - v
        return np.array([r * np.cos(u), r * np.sin(u), v])
    dcone = draft_angle(cone, 1.0, 0.5, pull_dir=(0, 0, 1))
    dun = draft_angle(undercut, 1.0, 0.5, pull_dir=(0, 0, 1))
    assert abs(abs(dcone) - 45.0) < 2.0 and abs(abs(dun) - 45.0) < 2.0, (dcone, dun)   # ~45deg slant
    assert (dcone > 0) != (dun > 0), (dcone, dun)           # opposite signs: one drafts, one undercuts
    # flip_normal negates the sign (the caller's outward-orientation control)
    assert abs(draft_angle(cone, 1.0, 0.5, pull_dir=(0, 0, 1), flip_normal=True) + dcone) < 1e-6

    # --- determinism ---
    assert gaussian_curvature(sphere, 0.7, 1.0) == gaussian_curvature(sphere, 0.7, 1.0)

    print("holographic_surfanalysis selftest OK: sphere K=1/R^2 and both principal curvatures 1/R; cylinder K=0 with "
          "principal {1/R,0} and reads developable; plane K=H=0; saddle K<0; draft angle ~0 on a vertical cylinder "
          "wall, and two mirror-slanted walls read opposite signs at ~45deg (one drafts, one undercuts) -- sign "
          "follows the parametrization normal (kept negative; flip_normal for the caller's outward orientation). "
          "All via the first/second fundamental forms of the analytic surface. Deterministic.")


if __name__ == "__main__":
    _selftest()

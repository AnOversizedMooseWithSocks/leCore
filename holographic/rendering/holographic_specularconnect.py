"""specularconnect -- connect a shade point to a light THROUGH a refractive surface (the caustic path).

WHY ORDINARY NEXT-EVENT ESTIMATION CANNOT DO CAUSTICS, which is the whole reason this exists. NEE
connects a shade point to a light with a straight SHADOW RAY. A caustic is the path
light -> glass -> floor -> eye: the connection has to pass THROUGH a specular vertex, and a straight
ray cannot, because the glass bends it. So a path tracer with perfect NEE still renders caustics only
by the luck of a bounce landing on the light after refraction -- which is why they arrive as fireflies
or not at all. This is a structural gap in the estimator, not a sampling budget.

THE STATE OF THE ART names the fix: Manifold Next-Event Estimation (Hanika et al.) and Specular
Manifold Sampling (Zeltner, Georgiev & Jakob, SIGGRAPH 2020), which Blender's Cycles uses for its
caustics. Instead of hoping a ray lands correctly, SOLVE for the point on the specular surface that
connects the two endpoints: a Newton iteration on a constraint that is zero exactly when the refracted
direction points at the target.

WHY IT BELONGS HERE RATHER THAN AS NEW MACHINERY. This engine already states that IK, PBD, PnP and the
resonator "are all iterate a projection", and holographic_pdb.project_onto_constraints is described in
its own docstring as "the one engine under three faculties the mind grew separately". A specular
connection is that same shape wearing another costume: a residual that must be driven to zero by
repeatedly snapping a point onto a manifold. What was missing was the CONSTRAINT, not the solver --
which is why this module is small.

THE CONSTRAINT. For a light L, a shade point P, and a candidate vertex X on the refractive surface
with unit normal N and relative index eta:
    d_in  = normalize(X - L)                      the ray arriving at the glass
    d_ref = refract(d_in, N, eta)                 where it goes after bending
    d_out = normalize(P - X)                      where it WOULD have to go to reach P
    C(X)  = d_ref - (d_ref . d_out) * d_out       the part of d_ref that misses P
C is zero exactly when the refracted ray points at P. It is a 3-vector but lives in the 2-D tangent
plane, which is the reduction the Newton step exploits.

SCOPE, DECLARED RATHER THAN DISCOVERED: ONE interface. A real glass object refracts on entry AND exit,
so a full solution solves a 2-vertex chain (4 unknowns). This solves the single-interface case -- a
water surface, a lens seen from inside, or a thin shell -- and is the honest first rung. Two-vertex
chaining is the next build and is NOT quietly approximated here: `connect` reports the residual it
achieved so a caller can reject a connection rather than trust one that did not converge.
"""
import numpy as np


def refract(d, n, eta):
    """Snell's law as a vector operation. Returns (direction, valid) -- valid is False under total
    internal reflection, where no refracted ray exists and the connection has no solution at all."""
    d = np.atleast_2d(np.asarray(d, float))
    n = np.atleast_2d(np.asarray(n, float))
    eta = np.atleast_1d(np.asarray(eta, float))
    cosi = -np.sum(d * n, axis=-1)
    k = 1.0 - eta ** 2 * (1.0 - cosi ** 2)
    valid = k >= 0.0
    out = eta[:, None] * d + (eta * cosi - np.sqrt(np.maximum(k, 0.0)))[:, None] * n
    nrm = np.linalg.norm(out, axis=-1, keepdims=True)
    return np.where(valid[:, None], out / np.maximum(nrm, 1e-12), 0.0), valid


def residual(X, L, P, N, eta):
    """C(X): the component of the refracted direction that MISSES P. Zero iff the path connects.

    Returned as a 3-vector rather than an angle because the Newton step needs its direction, not just
    its size -- an angle would say how wrong the vertex is and not which way to move it."""
    X = np.atleast_2d(np.asarray(X, float))
    d_in = X - np.atleast_2d(np.asarray(L, float))
    d_in /= np.maximum(np.linalg.norm(d_in, axis=-1, keepdims=True), 1e-12)
    d_ref, valid = refract(d_in, N, eta)
    d_out = np.atleast_2d(np.asarray(P, float)) - X
    d_out /= np.maximum(np.linalg.norm(d_out, axis=-1, keepdims=True), 1e-12)
    perp = d_ref - np.sum(d_ref * d_out, axis=-1)[:, None] * d_out
    return np.where(valid[:, None], perp, np.nan), valid


def tangent_basis(N):
    """An orthonormal (T, B) spanning the surface at each normal -- the 2-D space the vertex moves in.

    A specular vertex is constrained to the surface, so the Newton step must live in the tangent plane;
    stepping in full 3-D would walk the point off the geometry and the constraint would stop meaning
    anything. Building the basis from whichever axis is least aligned with N avoids the degenerate
    cross product at the poles."""
    N = np.atleast_2d(np.asarray(N, float))
    a = np.where(np.abs(N[:, 1:2]) < 0.9, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    T = np.cross(a, N)
    T /= np.maximum(np.linalg.norm(T, axis=-1, keepdims=True), 1e-12)
    return T, np.cross(N, T)


def connect(sdf_normal_fn, project_fn, X0, L, P, eta, iters=24, step=1e-3, tol=1e-5, damp=1e-6):
    """Solve for the surface vertex that connects light L to shade point P through refraction.

    Returns {'X', 'residual', 'converged', 'valid'}. `project_fn(X) -> X` snaps a point back onto the
    surface (an SDF walk), `sdf_normal_fn(X) -> N` gives its normal. A Newton step in the 2-D tangent
    plane, re-projected onto the surface each iteration -- iterate a projection, exactly as the engine's
    other solvers do.

    IT REPORTS WHAT IT ACHIEVED. A Newton solve on a non-convex surface does not always converge, and a
    caustic renderer that trusts an unconverged vertex draws light where none goes. `converged` is the
    caller's gate; `residual` is the number behind it. Damping keeps the 2x2 solve from exploding where
    the surface is locally flat and the Jacobian is singular."""
    X = np.atleast_2d(np.asarray(X0, float)).copy()
    L = np.atleast_2d(np.asarray(L, float))
    P = np.atleast_2d(np.asarray(P, float))
    eta = np.atleast_1d(np.asarray(eta, float))
    if eta.size == 1 and X.shape[0] > 1:
        eta = np.repeat(eta, X.shape[0])
    ok = np.ones(X.shape[0], bool)
    for _ in range(int(iters)):
        N = np.atleast_2d(np.asarray(sdf_normal_fn(X), float))
        C, valid = residual(X, L, P, N, eta)
        ok &= valid
        err = np.linalg.norm(np.nan_to_num(C), axis=-1)
        if np.all(err < tol):
            break
        T, B = tangent_basis(N)
        # Numerical Jacobian in the tangent plane: two extra residual evaluations, no autodiff --
        # the same choice holographic_spectralup makes, for the same reason (NumPy only).
        Ct, _ = residual(project_fn(X + T * step), L, P,
                         np.atleast_2d(np.asarray(sdf_normal_fn(project_fn(X + T * step)), float)), eta)
        Cb, _ = residual(project_fn(X + B * step), L, P,
                         np.atleast_2d(np.asarray(sdf_normal_fn(project_fn(X + B * step)), float)), eta)
        dCt = (np.nan_to_num(Ct) - np.nan_to_num(C)) / step
        dCb = (np.nan_to_num(Cb) - np.nan_to_num(C)) / step
        # Normal equations of the 3x2 system, damped (Levenberg): a flat patch makes it singular.
        a11 = np.sum(dCt * dCt, -1) + damp
        a12 = np.sum(dCt * dCb, -1)
        a22 = np.sum(dCb * dCb, -1) + damp
        b1 = np.sum(dCt * np.nan_to_num(C), -1)
        b2 = np.sum(dCb * np.nan_to_num(C), -1)
        det = a11 * a22 - a12 * a12
        det = np.where(np.abs(det) < 1e-18, 1e-18, det)
        du = -(a22 * b1 - a12 * b2) / det
        dv = -(a11 * b2 - a12 * b1) / det
        move = np.clip(np.stack([du, dv], -1), -0.25, 0.25)      # trust region, in surface units
        X = project_fn(X + move[:, :1] * T + move[:, 1:] * B)
    N = np.atleast_2d(np.asarray(sdf_normal_fn(X), float))
    C, valid = residual(X, L, P, N, eta)
    err = np.linalg.norm(np.nan_to_num(C), axis=-1)
    return {"X": X, "residual": err, "converged": (err < tol) & valid & ok, "valid": valid & ok}


def _selftest():
    # A PLANE is the case with a known answer: for a flat refractor the constraint is solvable and the
    # residual must go to zero. Testing against geometry whose solution can be checked independently
    # is the point -- a solver validated only on the shape it was tuned for proves nothing.
    nrm = np.array([0.0, 1.0, 0.0])

    def normal_fn(X):
        return np.repeat(nrm[None, :], np.atleast_2d(X).shape[0], axis=0)

    def project_fn(X):
        X = np.atleast_2d(np.asarray(X, float)).copy()
        X[:, 1] = 0.0                      # snap back onto the y=0 plane
        return X

    L = np.array([[-1.0, 2.0, 0.3]])
    P = np.array([[0.9, -1.5, -0.2]])
    out = connect(normal_fn, project_fn, np.array([[0.0, 0.0, 0.0]]), L, P, 1.0 / 1.5, iters=60)
    assert out["valid"][0], "a plane refraction must exist for this configuration"
    assert out["converged"][0], "did not converge on a PLANE: residual %.3e" % out["residual"][0]
    assert abs(out["X"][0, 1]) < 1e-12, "the vertex left the surface"

    # AND THE SOLUTION IS THE RIGHT ONE, checked by walking the path forward rather than by trusting
    # the residual that the solver itself minimised.
    X = out["X"]
    d_in = X - L; d_in /= np.linalg.norm(d_in, axis=-1, keepdims=True)
    d_ref, ok = refract(d_in, normal_fn(X), np.array([1.0 / 1.5]))
    to_P = P - X; to_P /= np.linalg.norm(to_P, axis=-1, keepdims=True)
    assert ok[0] and float(np.sum(d_ref * to_P)) > 1.0 - 1e-6, \
        "the refracted ray does not actually point at P (cos %.6f)" % float(np.sum(d_ref * to_P))

    # TOTAL INTERNAL REFLECTION HAS NO SOLUTION, and must be reported rather than returned as one.
    d_graze = np.array([[0.999, -0.045, 0.0]])
    d_graze /= np.linalg.norm(d_graze)
    _, tir = refract(d_graze, np.array([[0.0, 1.0, 0.0]]), np.array([1.5]))
    assert not bool(tir[0]), "refract() claimed a solution under total internal reflection"

    # The residual is zero exactly on a connecting path, by construction.
    C, _ = residual(X, L, P, normal_fn(X), np.array([1.0 / 1.5]))
    assert float(np.linalg.norm(C)) < 1e-5, float(np.linalg.norm(C))
    print("specularconnect selftest OK -- plane connection converges to residual %.2e, forward walk "
          "confirms the path, TIR reported as no-solution" % out["residual"][0])


if __name__ == "__main__":
    _selftest()

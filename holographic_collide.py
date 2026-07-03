"""Environment collision -- keep particles / cloth OUTSIDE a scene SDF, as one more projection.

The softbody solver already resolves distance, bending, volume, and node-node self-collision, and the panel's
unification (Macklin) is shipped: `project_onto_constraints` is the one iterate-a-projection engine under the resonator,
the PnP denoiser, and PBD. What was missing was collision with the *environment* -- an arbitrary signed-distance
surface -- so cloth couldn't drape over a scene object and emitted particles couldn't pile on one. This module adds
exactly that, and adds it in the shape the rest of the engine already speaks: a PROJECTION callable that snaps a
position vector onto the feasible set 'outside the collider', so it drops straight into the same unified sweep as every
other constraint. One solver, one more constraint.

WHY a projection (not a force): position-based dynamics resolves contact by moving the point to the surface, not by
integrating a penalty force -- it is unconditionally stable and needs no stiffness tuning, which is the whole reason
PBD/XPBD won in real-time engines. Deterministic; finite-difference normals; vectorised (no per-node Python loop).
"""
import numpy as np
from holographic_emitter import _sdf_normal


def resolve_sdf_collision(X, sdf_eval, radius=0.0, eps=1e-3):
    """Push every point that is inside the collider (signed distance < `radius`) back OUT to the surface offset by
    `radius`, along the outward normal. `X` is (N, D); `sdf_eval` a callable P -> signed distance (negative inside).
    Returns the corrected positions. `radius` > 0 gives the particles a thickness so they rest ON the surface rather
    than exactly at it. One vectorised contact resolve."""
    X = np.asarray(X, float)
    d = np.asarray(sdf_eval(X), float)
    inside = d < radius
    if not np.any(inside):
        return X
    Xn = X.copy()
    n = _sdf_normal(sdf_eval, X[inside], eps)
    Xn[inside] = X[inside] + (radius - d[inside])[:, None] * n     # move out to exactly the offset surface
    return Xn


def sdf_collision_projection(sdf_eval, N, D, radius=0.0, eps=1e-3):
    """A projection callable over the FLAT position vector, for `project_onto_constraints` -- environment collision as
    one more projection in the same sweep as the distance/bend constraints. So cloth drapes over a scene SDF using the
    SAME unified iterate-a-projection engine the resonator and the denoiser use (Macklin's 'one solver, many uses')."""
    def proj(flat):
        X = np.asarray(flat, float).reshape(N, D)
        return resolve_sdf_collision(X, sdf_eval, radius=radius, eps=eps).ravel()
    return proj


def _selftest():
    """Points scattered inside a sphere are all pushed to its surface by the collision projection; and the SAME
    projection, run inside the shipped project_onto_constraints sweeper alongside a distance link, satisfies BOTH."""
    from holographic_denoise import project_onto_constraints
    R = 1.0
    sphere = lambda P: np.linalg.norm(P, axis=1) - R
    X = np.array([[0.2, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.1], [2.0, 0.0, 0.0]])  # 3 inside, 1 outside
    Xc = resolve_sdf_collision(X, sphere, radius=0.0)
    assert (sphere(Xc) >= -1e-6).all()                            # nobody left inside the sphere
    assert np.allclose(Xc[3], X[3])                               # the outside point didn't move
    # unify: two nodes must stay a fixed distance apart AND both stay outside the sphere -> one projection sweep
    N, D = 2, 3
    x0 = np.array([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]]).ravel()    # both inside, opposite sides (non-degenerate)
    dist = 2.4
    def link(flat):
        Xr = flat.reshape(N, D); n = Xr[0] - Xr[1]; d = np.linalg.norm(n)
        if d < 1e-9:
            return flat
        n = n / d; c = d - dist; Xn = Xr.copy()
        Xn[0] = Xr[0] - 0.5 * c * n; Xn[1] = Xr[1] + 0.5 * c * n
        return Xn.ravel()
    coll = sdf_collision_projection(sphere, N, D, radius=0.0)
    out, sweeps, _ = project_onto_constraints(x0, [link, coll], iters=60)
    Xf = out.reshape(N, D)
    assert (sphere(Xf) >= -0.02).all()                            # both nodes outside the collider
    assert abs(np.linalg.norm(Xf[0] - Xf[1]) - dist) < 0.1        # and the link is (nearly) satisfied
    print("collide selftest ok: %d inside-points pushed to the surface; link+collision co-satisfied in %d sweeps" % (3, sweeps))


if __name__ == "__main__":
    _selftest()

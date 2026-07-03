"""Emit particles FROM a surface -- the source that drives a particle system.

A particle system needs somewhere for particles to be born. In a DCC app you pick an object and 'emit from surface':
particles spawn on the mesh, usually shot out along the normal, often with the emission DENSITY or SPEED painted by a
map. holostuff had a `ParticleSystem` that could be pushed by forces and advected by fields, and a Poisson sampler for
a box, but no surface emitter -- so a system had nowhere principled to spawn from. This module fills that gap.

`emit_from_surface(sdf_eval, n, bounds, ...)` samples points on the zero level-set of ANY signed-distance function
(project random candidates onto the surface with a few Newton steps), returns their positions, outward normals, and
initial velocities (normal * speed). Both the emit SPEED and the emit WEIGHT (where on the surface particles prefer to
spawn) are resolved through `holographic_param.resolve_param`, so each can be a bare number OR a map / field / wired
output -- the 'parameter is more than a number' affordance, proven on a real emitter.

WHY project-onto-surface rather than march the surface: it is dependency-free (finite-difference normals + Newton),
works for any callable SDF including a whole scene union, and needs no meshing. Deterministic given the seed.
"""
import numpy as np
from holographic_param import resolve_param


def _sdf_normal(sdf_eval, P, eps=1e-3):
    """Outward unit normal = normalized gradient of the SDF, by central differences. Works for any callable SDF."""
    P = np.asarray(P, float)
    g = np.empty_like(P)
    for k in range(P.shape[1]):
        d = np.zeros(P.shape[1]); d[k] = eps
        g[:, k] = (np.asarray(sdf_eval(P + d), float) - np.asarray(sdf_eval(P - d), float)) / (2 * eps)
    return g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)


def emit_from_surface(sdf_eval, n, bounds, speed=1.0, weight=None, seed=0, project_iters=8, tol=0.02):
    """Spawn up to `n` particles ON the surface of `sdf_eval` (a callable P (M,D) -> signed distance), within `bounds`
    = (lo, hi). Returns (positions (K,D), normals (K,D), velocities (K,D)) with velocity = outward_normal * speed.

    `speed` and `weight` are resolved through the parameter socket, so either can be a constant, a map, a field, or a
    wired output:
      * `speed`  -- per-particle emission speed along the normal (e.g. faster where a curvature map is high).
      * `weight` -- emission DENSITY on the surface: particles are importance-sampled so more spawn where weight is
                    large (e.g. emit only from the top of an object, painted by a map). None = uniform on the surface.
    """
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    D = lo.shape[0]
    rng = np.random.default_rng(seed)
    over = max(n * 12, 400)
    P = rng.uniform(lo, hi, size=(over, D))
    for _ in range(project_iters):                                 # Newton onto the zero level-set: p -= sdf(p)*n(p)
        d = np.asarray(sdf_eval(P), float)
        P = P - d[:, None] * _sdf_normal(sdf_eval, P)
    d = np.asarray(sdf_eval(P), float)
    on = np.abs(d) < tol                                           # keep the ones that actually reached the surface
    P = P[on]
    if len(P) == 0:
        z = np.zeros((0, D)); return z, z, z
    nrm = _sdf_normal(sdf_eval, P)
    if weight is not None:                                         # importance-sample the surface by the weight map
        w = np.clip(np.asarray(resolve_param(weight, P), float), 0.0, None)
        if w.max() > 0:
            keep = rng.random(len(P)) < (w / w.max())
            P, nrm = P[keep], nrm[keep]
    if len(P) > n:                                                 # thin down to n (shuffled, so no spatial bias)
        sel = rng.permutation(len(P))[:n]; P, nrm = P[sel], nrm[sel]
    sp = np.asarray(resolve_param(speed, P), float)                # emit speed: constant OR a map/field
    vel = nrm * sp[:, None]
    return P, nrm, vel


def advance(pos, vel, force=None, dt=0.05, damping=0.0, wrap_to=None):
    """One semi-implicit Euler step for an N-D particle set (the 3-D sibling of ParticleSystem.step). `force` is an
    (N, D) acceleration array (gravity, an attractor, a sampled field). Returns (pos, vel)."""
    pos = np.asarray(pos, float); vel = np.asarray(vel, float)
    if force is not None:
        vel = vel + dt * np.asarray(force, float)
    if damping:
        vel = vel * (1.0 - damping)
    pos = pos + dt * vel
    if wrap_to is not None:
        pos = np.mod(pos, np.asarray(wrap_to, float))
    return pos, vel


def _selftest():
    """Particles emitted from a sphere land ON the sphere, their velocities point along the (radial) normal, and a
    weight map biases WHERE they spawn -- proving both the emitter and the parameter socket driving it."""
    from holographic_param import Param
    R = 1.5
    sphere = lambda P: np.linalg.norm(P, axis=1) - R           # a sphere SDF centred at origin
    bounds = (np.full(3, -2.2), np.full(3, 2.2))
    pos, nrm, vel = emit_from_surface(sphere, 200, bounds, speed=2.0, seed=0)
    assert len(pos) > 50                                        # emitted a healthy number
    assert np.abs(np.linalg.norm(pos, axis=1) - R).max() < 0.05  # every particle is ON the surface
    radial = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    assert (np.abs((nrm * radial).sum(1) - 1.0) < 0.02).all()  # normals are radial (outward)
    assert np.allclose(np.linalg.norm(vel, axis=1), 2.0, atol=1e-6)  # speed applied along the normal
    # WEIGHT AS A MAP: emit only from the top hemisphere (weight = 1 where z>0, else 0)
    top = Param(field=lambda P: (P[:, 2] > 0).astype(float))
    tpos, _, _ = emit_from_surface(sphere, 200, bounds, speed=1.0, weight=top, seed=1)
    assert len(tpos) > 20 and (tpos[:, 2] >= -0.1).all()        # spawns respect the weight map (top only)
    # SPEED AS A FIELD: speed grows with height -> higher particles move faster
    fast = Param(field=lambda P: 1.0 + 2.0 * np.clip(P[:, 2], 0, None))
    fpos, _, fvel = emit_from_surface(sphere, 200, bounds, speed=fast, seed=2)
    sp = np.linalg.norm(fvel, axis=1)
    assert np.corrcoef(fpos[:, 2], sp)[0, 1] > 0.5             # speed tracks the field
    print("emitter selftest ok: %d particles ON the sphere; normals radial; weight-map & speed-field sockets both drive emission" % len(pos))


if __name__ == "__main__":
    _selftest()

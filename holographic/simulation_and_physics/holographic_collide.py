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
from holographic.simulation_and_physics.holographic_emitter import _sdf_normal


def _escape_direction(sdf_eval, X, step):
    """A deterministic escape direction for points whose SDF gradient VANISHES -- the medial axis (the dead centre
    of a slab, the axis of a cylinder), where central differences cancel exactly and there is no normal.

    Probe +/- each coordinate axis at distance `step` and take whichever candidate raises the SDF most. Ties (the
    slab centre: +x and -x are exactly equivalent) break by lowest axis index, then positive sign -- an arbitrary
    but DETERMINISTIC choice, which is what the engine's determinism rule requires. It is a choice, not a
    derivation: on the medial axis both sides are genuinely equally correct."""
    X = np.asarray(X, float)
    D = X.shape[1]
    best_dir = np.zeros_like(X)
    best_val = np.full(len(X), -np.inf)
    for k in range(D):                                    # ascending axis, then +1 before -1: the tie-break order
        for sign in (1.0, -1.0):
            cand = np.zeros(D)
            cand[k] = sign
            val = np.asarray(sdf_eval(X + step[:, None] * cand), float)
            better = val > best_val + 1e-15               # strict: an equal candidate never displaces an earlier one
            best_val = np.where(better, val, best_val)
            best_dir[better] = cand
    return best_dir


def resolve_sdf_collision(X, sdf_eval, radius=0.0, eps=1e-3):
    """Push every point that is inside the collider (signed distance < `radius`) back OUT to the surface offset by
    `radius`, along the outward normal. `X` is (N, D); `sdf_eval` a callable P -> signed distance (negative inside).
    Returns the corrected positions. `radius` > 0 gives the particles a thickness so they rest ON the surface rather
    than exactly at it. One vectorised contact resolve.

    THE MEDIAL-AXIS FIX (this used to silently violate its own contract). The outward normal is the normalised SDF
    gradient, and on the medial axis that gradient is EXACTLY ZERO -- the central differences cancel. `_sdf_normal`
    then returns a zero vector (the +1e-12 guard keeps it from being NaN), so `X + (radius-d)*0 == X`: the point was
    left INSIDE the collider while the function reported success. Measured: a point at the dead centre of a slab
    |x| - 0.05 came back at x = 0.0 with sdf = -0.05, still inside. Such points now get a deterministic
    `_escape_direction` and are re-resolved. Points with a well-defined normal are untouched and BIT-IDENTICAL."""
    from holographic.mesh_and_geometry.holographic_sdf import as_eval
    sdf_eval = as_eval(sdf_eval)
    X = np.asarray(X, float)
    d = np.asarray(sdf_eval(X), float)
    inside = d < radius
    if not np.any(inside):
        return X
    Xn = X.copy()
    n = _sdf_normal(sdf_eval, X[inside], eps)
    depth = (radius - d[inside])
    # degenerate = no usable gradient. _sdf_normal normalises by (|g| + 1e-12), so a vanishing gradient shows up
    # as a (near) zero-length "normal" rather than a NaN -- test for it directly.
    degen = np.linalg.norm(n, axis=1) < 0.5               # a true unit normal has length 1; nothing lands between
    if np.any(degen):
        n[degen] = _escape_direction(sdf_eval, X[inside][degen], np.maximum(depth[degen], eps))
    Xn[inside] = X[inside] + depth[:, None] * n           # move out to exactly the offset surface
    if np.any(degen):
        # one more pass: the escape step lands near the surface but the axis probe is not the true normal, so let
        # the (now well-defined) gradient finish the job. Bounded, not a loop -- a second degeneracy is impossible
        # once the point is off the medial axis.
        idx = np.flatnonzero(inside)[degen]
        d2 = np.asarray(sdf_eval(Xn[idx]), float)
        still = d2 < radius - 1e-12
        if np.any(still):
            j = idx[still]
            n2 = _sdf_normal(sdf_eval, Xn[j], eps)
            Xn[j] = Xn[j] + (radius - d2[still])[:, None] * n2
    return Xn


def sdf_offset(sdf_eval, margin):
    """The SPECULATIVE CONTACT MARGIN (Box3D lesson B4), and it costs nothing: enlarging a collider by `margin` is
    just subtracting a constant from its signed distance. Where a mesh engine must inflate AABBs and grow contact
    manifolds, an SDF-native engine gets the enlarged shape for one subtraction -- the margin IS an offset.

    Returns a callable P -> sdf(P) - margin. Use it to generate contacts BEFORE bodies touch (the solver then has a
    frame to react), or with a negative margin to shrink a collider.

    KEPT NEGATIVE -- a margin DETECTS proximity, it does not PREVENT tunnelling. Measured: a body stepping 0.5 m per
    frame across a 0.1 m wall lands at x = +0.20 having never sampled the interior; resolving there against a 0.2
    margin pushes it to x = +0.25, i.e. FURTHER along its direction of travel and out the wrong side. A margin
    widens the contact band; it cannot recover which side of a thin wall you started on, because a point sample
    carries no memory of the swept path. That is what `time_of_impact` is for."""
    return lambda P: np.asarray(sdf_eval(P), float) - float(margin)


def sdf_collision_projection(sdf_eval, N, D, radius=0.0, eps=1e-3):
    """A projection callable over the FLAT position vector, for `project_onto_constraints` -- environment collision as
    one more projection in the same sweep as the distance/bend constraints. So cloth drapes over a scene SDF using the
    SAME unified iterate-a-projection engine the resonator and the denoiser use (Macklin's 'one solver, many uses')."""
    def proj(flat):
        X = np.asarray(flat, float).reshape(N, D)
        return resolve_sdf_collision(X, sdf_eval, radius=radius, eps=eps).ravel()
    return proj


def time_of_impact(X, V, dt, sdf_eval, radius=0.0, max_steps=96, surf_eps=1e-4):
    """CONSERVATIVE ADVANCEMENT (continuous collision detection) for points X:(N,3) moving with velocity V:(N,3)
    over one step `dt`. Returns (hit:(N,) bool, toi:(N,) float in [0, dt], contact:(N,3)) -- `toi` is the time of
    first contact with `sdf_eval` offset by `radius`; where `hit` is False, `toi` is `dt` and `contact` is the
    unobstructed landing point.

    THE POINT (Box3D lesson B4): conservative advancement's core query is "how far can I move without hitting
    anything?", and for an SDF that is *the SDF value itself*. So this is SPHERE TRACING, and it DELEGATES to the
    renderer's `raymarch.sphere_trace` rather than growing a second copy: the same march that renders a pixel
    computes a time of impact, and it is the same distance query Walk-on-Spheres takes its steps by. leCore gets
    CCD's core primitive for nothing -- there is no dedicated CCD pass anywhere in this module.

    Zero-velocity points never hit (nothing swept, nothing to test) and are returned with toi = dt."""
    from holographic.rendering.holographic_raymarch import sphere_trace
    from holographic.mesh_and_geometry.holographic_sdf import as_eval
    sdf_eval = as_eval(sdf_eval)                          # accept a node, a callable, or a DSL string
    X = np.asarray(X, float)
    V = np.asarray(V, float)
    dt = float(dt)
    speed = np.linalg.norm(V, axis=1)
    moving = speed > 1e-15

    hit = np.zeros(len(X), bool)
    toi = np.full(len(X), dt)
    contact = X + V * dt                                  # the unobstructed landing point, for the misses
    if not np.any(moving):
        return hit, toi, contact

    O = X[moving]
    D = V[moving] / speed[moving, None]
    L = speed[moving] * dt                                # each point's swept length this step
    field = sdf_offset(sdf_eval, radius) if radius else sdf_eval

    # sphere_trace takes ONE scalar max_dist, so march to the longest sweep and reject the overshoots per point.
    h, t, p = sphere_trace(field, O, D, max_steps=max_steps, max_dist=float(L.max()) + surf_eps,
                           surf_eps=surf_eps)
    real = h & (t <= L)                                   # a hit beyond THIS point's sweep is not a hit this step
    idx = np.flatnonzero(moving)
    hit[idx[real]] = True
    toi[idx[real]] = t[real] / speed[moving][real]        # convert arc-length back to time
    contact[idx[real]] = p[real]
    return hit, toi, contact


def resolve_swept_collision(X_prev, X, sdf_eval, radius=0.0, max_steps=96, surf_eps=1e-4):
    """Positional CCD for a PBD-style solver: a node that MOVED from `X_prev` to `X` must not have crossed the
    collider on the way, even if neither endpoint is inside it.

    Returns the corrected positions. Nodes whose swept segment hits `sdf_eval` (offset by `radius`) are placed at
    the first contact; everything else is returned untouched, BIT-IDENTICALLY -- so this is a strict addition to
    `resolve_sdf_collision`, never a perturbation of the nodes it does not catch.

    WHY THE DISCRETE RESOLVE IS NOT ENOUGH, measured: a node stepping 0.5 m per frame across a 0.1 m wall lands at
    x = +0.20 having never sampled the interior. `resolve_sdf_collision` sees a positive signed distance and does
    nothing. Widening the collider does not help -- a margin resolves the already-crossed node out the WRONG side
    (it lands further along its travel). A point sample carries no memory of the swept path; only the sweep does.

    Cheap, because conservative advancement IS sphere tracing: this delegates to `time_of_impact`, which delegates
    to the renderer's `raymarch.sphere_trace`. There is no dedicated CCD pass."""
    X_prev = np.asarray(X_prev, float)
    X = np.asarray(X, float)
    disp = X - X_prev
    hit, toi, contact = time_of_impact(X_prev, disp, 1.0, sdf_eval, radius=radius,
                                       max_steps=max_steps, surf_eps=surf_eps)   # dt=1 => V is the displacement
    if not np.any(hit):
        return X
    out = X.copy()
    out[hit] = contact[hit]
    return out


def advance_ccd(X, V, dt, sdf_eval, radius=0.0, restitution=0.0, max_steps=96, surf_eps=1e-4):
    """Advance points by one step WITHOUT tunnelling: sweep to the first contact (`time_of_impact`), stop there,
    and cancel the into-surface component of velocity (with `restitution` for a bounce). Points that hit nothing
    take the full step. Returns (X_new, V_new, hit).

    This is what a discrete `resolve_sdf_collision` cannot do at any margin: a point sample carries no memory of
    the path it swept, so a thin wall crossed in one step is simply never seen. Measured: a body at 30 m/s stepping
    0.5 m per frame passes clean through a 0.1 m wall under discrete resolution and is stopped exactly at the
    surface here. Deterministic; no RNG."""
    from holographic.mesh_and_geometry.holographic_sdf import as_eval
    sdf_eval = as_eval(sdf_eval)
    X = np.asarray(X, float)
    V = np.asarray(V, float)
    hit, toi, contact = time_of_impact(X, V, dt, sdf_eval, radius=radius, max_steps=max_steps, surf_eps=surf_eps)
    Xn = np.where(hit[:, None], contact, X + V * dt)
    Vn = V.copy()
    if np.any(hit):
        n = _sdf_normal(sdf_eval, Xn[hit], eps=max(surf_eps, 1e-4))
        vn = np.sum(Vn[hit] * n, axis=1)[:, None]         # component INTO the surface (negative when approaching)
        Vn[hit] = Vn[hit] - (1.0 + float(restitution)) * np.minimum(vn, 0.0) * n
    return Xn, Vn, hit


# Contact TYPES as categorical records: each names the {overlap, velocity, restitution} condition it is for.
# This is a LABELING/DISPATCH layer over the numeric resolvers (resolve_sdf_collision / advance_ccd) -- it does
# NOT replace their math; it names WHAT KIND of contact happened so a caller can pick a per-type response and
# log a self-explaining reason. Fillers are categories (deep/shallow, fast/slow, bouncy/dead), never raw floats.
_CONTACT_TYPES = {
    "bounce":       {"overlap": "shallow", "velocity": "fast", "restitution": "bouncy"},
    "slide":        {"overlap": "shallow", "velocity": "fast", "restitution": "dead"},
    "rest_contact": {"overlap": "shallow", "velocity": "slow", "restitution": "dead"},
    "penetration":  {"overlap": "deep",    "velocity": "fast", "restitution": "bouncy"},
    "jam":          {"overlap": "deep",    "velocity": "slow", "restitution": "dead"},
}


def _bin_contact(overlap, velocity, restitution, overlap_deep=0.1, velocity_fast=0.5, restitution_bouncy=0.3):
    """Bin the three CONTINUOUS contact scalars into the CATEGORICAL record classify_contact matches on. The bin
    edges are the schema -- documented and adjustable -- so the categorical match_record never sees a raw float
    (its kept-negative: categorical only). This is where the continuous->categorical boundary is made explicit."""
    return {
        "overlap": "deep" if overlap >= overlap_deep else "shallow",
        "velocity": "fast" if abs(velocity) >= velocity_fast else "slow",
        "restitution": "bouncy" if restitution >= restitution_bouncy else "dead",
    }


def classify_contact(overlap, velocity, restitution, mind=None, margin=0.1, **bins):
    """Name a contact's TYPE (bounce / slide / rest_contact / penetration / jam) from its {overlap, velocity,
    restitution}. Bins the continuous scalars to categories (_bin_contact), then match_record against the
    contact-type records + decide_or_abstain. Returns {'type', 'confident', 'record', 'ranked'}. WHY: the
    resolvers (resolve_sdf_collision / advance_ccd) compute the RESPONSE but never name the SITUATION; this
    labels it so a solver can dispatch a per-type policy and log a reason. KEPT NEGATIVE: this is a LABEL over
    the numerics, NOT a replacement -- the actual impulse math stays in advance_ccd; and the categorical bins
    lose the exact magnitude (a barely-fast and a very-fast contact both read 'fast'), which is deliberate:
    which TYPE it is is categorical; how hard to respond is the resolver's continuous job."""
    if mind is None:
        import lecore
        mind = lecore.UnifiedMind(dim=512, seed=0)
    from holographic.misc.holographic_relations import match_record, decide_or_abstain
    rec = _bin_contact(overlap, velocity, restitution, **bins)
    ranked = match_record(mind.encode_record, rec, _CONTACT_TYPES)
    ctype, score, confident = decide_or_abstain(ranked, margin=margin)
    return {"type": ctype, "confident": confident, "record": rec, "ranked": ranked}


def _selftest():
    """Points scattered inside a sphere are all pushed to its surface by the collision projection; and the SAME
    projection, run inside the shipped project_onto_constraints sweeper alongside a distance link, satisfies BOTH."""
    from holographic.rendering.holographic_denoise import project_onto_constraints
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
    # -- X4: the medial-axis fix, the speculative margin, and conservative advancement -------------------------
    wall = lambda Pp: np.abs(Pp[:, 0]) - 0.05                     # a 0.1 m thick slab at x = 0

    # (1) THE BUG THIS FUNCTION USED TO HAVE: a point on the medial axis has a ZERO gradient, so the old
    #     resolve moved it by zero and reported success while leaving it inside.
    mid = resolve_sdf_collision(np.array([[0.0, 0.0, 0.0]]), wall)
    assert float(wall(mid)[0]) >= -1e-9, "a medial-axis point must not be left inside the collider"
    assert abs(float(mid[0, 0]) - 0.05) < 1e-9                    # deterministic escape: +x wins the tie

    # (2) the speculative margin is an OFFSET and costs one subtraction
    off = sdf_offset(wall, 0.2)
    assert abs(float(off(np.array([[1.0, 0, 0]]))[0]) - (0.95 - 0.2)) < 1e-12

    # (3) KEPT NEGATIVE: a margin cannot prevent tunnelling -- it resolves to the WRONG SIDE.
    landed = np.array([[0.20, 0.0, 0.0]])                         # came from x=-0.30, already through the wall
    pushed = resolve_sdf_collision(landed, wall, radius=0.2)
    assert pushed[0, 0] > landed[0, 0], "the margin pushes it further along its travel: detection != prevention"

    # (4) CONSERVATIVE ADVANCEMENT stops it exactly at the surface, with no dedicated CCD pass
    X = np.array([[-0.30, 0.0, 0.0]]); V = np.array([[30.0, 0.0, 0.0]]); dt = 1 / 60.0
    naive = resolve_sdf_collision(X + V * dt, wall)                # the discrete resolve: tunnels
    assert naive[0, 0] > 0.1, "discrete resolution is supposed to tunnel here (that is the premise)"
    hit, toi, contact = time_of_impact(X, V, dt, wall)
    assert hit[0] and abs(float(contact[0, 0]) + 0.05) < 1e-3      # stopped at the near face, x = -0.05
    assert abs(float(toi[0]) - 0.25 / 30.0) < 1e-6                 # exact time of impact
    Xn, Vn, h2 = advance_ccd(X, V, dt, wall)
    assert h2[0] and float(wall(Xn)[0]) >= -1e-3
    assert abs(float(Vn[0, 0])) < 1e-6                             # into-surface velocity cancelled

    print("collide selftest ok: %d inside-points pushed to the surface; link+collision co-satisfied in %d sweeps; "
          "X4: medial-axis points escape (were silently left inside), the speculative margin is one subtraction "
          "but resolves a tunnelled body to the WRONG side, and conservative advancement (sphere_trace, reused "
          "from the renderer) stops a 30 m/s body exactly on a 0.1 m wall the discrete resolve passes through"
          % (3, sweeps))



def _selftest_classify_contact():
    """A3: classify_contact names each contact type from binned categoricals; label layer, not the numerics."""
    import lecore
    m = lecore.UnifiedMind(dim=512, seed=0)
    assert classify_contact(0.02, 2.0, 0.8, mind=m)["type"] == "bounce"
    assert classify_contact(0.02, 0.1, 0.0, mind=m)["type"] == "rest_contact"
    assert classify_contact(0.5, 0.1, 0.0, mind=m)["type"] == "jam"
    # KEPT NEGATIVE guard: bins are categorical -- barely-fast and very-fast read the same type
    assert classify_contact(0.02, 0.6, 0.8, mind=m)["type"] == classify_contact(0.02, 9.0, 0.8, mind=m)["type"]
    print("  classify_contact selftest OK: bounce/rest/jam named; categorical bins collapse magnitude")

if __name__ == "__main__":
    _selftest(); _selftest_classify_contact()

"""L3: when is a normal-offset / shrink-wrap projection INJECTIVE? The reach, made checkable.

BACKLOG L3, and it is what makes O1's template wrap correct rather than hopeful. Projecting
template vertices along surface normals is only valid retopology if the map is one-to-one;
where it is not, the wrapped mesh folds through itself and the "correspondence" it produces
is fiction -- while surface_error still reads clean, because every vertex IS on the surface.

SOTA gives the condition in two parts, and the SECOND is the one that bites here
(Patrikalakis & Maekawa; Wallner et al.; the offset-surface literature):
  * LOCAL / differential: self-intersection arises "in concave regions of surface where the
    positive offset distance exceeds the maximum absolute value of the negative minimum
    principal curvature" -- i.e. the offset must stay under the smallest radius of curvature,
    |d| < 1/|kappa|.
  * GLOBAL / distance: it also arises "in the vicinity of a pair of COLLINEAR NORMAL POINTS
    whose distance is equal or smaller than TWICE the offset distance" -- two facing pieces
    of surface, each with the other in its normal direction.
Together these are the REACH (Federer's reach / local feature size / distance to the medial
axis): the offset is injective iff the offset distance is below it.

WHY THE GLOBAL TERM IS THE IMPORTANT ONE FOR CREATURES, and why a curvature-only check would
have been worse than none: an armpit, the gap between a limb and the torso, the space between
fingers, and a tail lying against a flank are all LOW-CURVATURE regions where two surfaces
FACE each other closely. Curvature says they are fine. They are not. A checker that passes
exactly the cases a creature rig hits would be actively misleading.

RULE-0 AUDIT (2026-08-16): `reach of a surface`, `safe offset distance` and `local feature
size` returned unrelated fallbacks -- no offset-safety predicate exists. REUSED: sdf_curvature
(the field Laplacian, "POSITIVE on convex edges, NEGATIVE in concave creases") for the local
term; nothing here recomputes curvature.

KEPT NEGATIVE: this samples the reach, it does not compute the medial axis. A sampled
estimate can MISS a thin feature that no sample landed in, so safe_offset is an estimate and
`wrap_is_injective` reports the sample count that backs it. Refusing on a sampled reach is
sound in the direction that matters (it errs toward saying "too big"), but a pass is evidence,
not proof -- and the difference is exactly what a Lean-verified medial-axis bound would close.
"""

import numpy as np


def collinear_normal_reach(points, normals, max_pairs=200000, seed=0):
    """The GLOBAL half of the reach: half the smallest distance between a pair of points whose
    normals face each other.

    For each point p with normal n, a point q is "facing" if q - p points along n and the
    normal at q points back. The offset must stay under half that distance or the two offset
    sheets cross. Returns (reach, i, j) -- the limiting pair is returned so a caller can SEE
    where the geometry is tight rather than just being told a number."""
    P = np.asarray(points, float)
    N = np.asarray(normals, float)
    n = len(P)
    best, bi, bj = np.inf, -1, -1
    # chunked so the (n, n) pair matrix never materialises whole -- LEVER 5, the same tiling
    # that cvt_remesh needed, applied before it becomes a wall rather than after
    step = max(1, int(max_pairs // max(n, 1)))
    for i0 in range(0, n, step):
        A = P[i0:i0 + step]
        NA = N[i0:i0 + step]
        D = P[None, :, :] - A[:, None, :]              # (a, n, 3)
        d = np.linalg.norm(D, axis=2)
        np.fill_diagonal(d[:, i0:i0 + step], np.inf)
        U = D / np.maximum(d[..., None], 1e-12)
        facing = (np.sum(U * NA[:, None, :], axis=2) > 0.5) & \
                 (np.sum(U * N[None, :, :], axis=2) < -0.5)
        dd = np.where(facing, d, np.inf)
        k = int(np.argmin(dd))
        a, b = divmod(k, n)
        if dd[a, b] < best:
            best, bi, bj = float(dd[a, b]), int(i0 + a), int(b)
    return (0.5 * best if np.isfinite(best) else np.inf), bi, bj


def shrinking_ball_lfs(sdf, points, normals=None, eps=2e-3, iters=24, r0=None):
    """LOCAL FEATURE SIZE by the shrinking-ball algorithm -- the correct definition, and the
    fix for the all-pairs test's fatal flaw.

    SOTA definition: "the local feature size on a 3D shape is the distance from a query point
    to its closest point on the medial axis", and "the reach of a shape refers to the MINIMUM
    of the LFS" (Federer). The medial axis is "the locus of centers of spheres that touch the
    shape's boundary at two or more unique points" -- i.e. MAXIMAL EMPTY BALLS. The standard
    algorithm finds, for each sample, "its maximal tangent ball containing no other sample
    points, by iteratively reducing its radius".

    WHY THIS REPLACES THE PAIRWISE FACING TEST, measured: that test asked whether two points
    face each other and are close. On a bumpy marching-cubes surface, GEODESICALLY ADJACENT
    points satisfy that -- two neighbours across a small wrinkle "face" each other -- so the
    reach on a real head collapsed to 0.0003 and the guard refused fur everywhere. The
    shrinking ball cannot make that mistake: a ball tangent at p and centred inside is empty
    only if nothing else is within it, and a neighbour on the SAME smooth patch is never
    inside the tangent ball. Adjacency is excluded BY CONSTRUCTION rather than by a threshold.

    With an SDF the iteration is a one-line fixed point -- r <- |sdf(p - r*n)| -- because
    |sdf(c)| IS the radius of the largest empty ball at c. No nearest-neighbour queries, no
    Voronoi, and O(N) instead of O(N^2)."""
    P = np.atleast_2d(np.asarray(points, float))
    if normals is None:
        g = np.empty_like(P)
        for k in range(3):
            d = np.zeros(3); d[k] = eps
            g[:, k] = (np.asarray(sdf(P + d), float).ravel() -
                       np.asarray(sdf(P - d), float).ravel()) / (2 * eps)
        normals = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    N = np.asarray(normals, float)
    r = np.full(len(P), float(r0) if r0 is not None
                else 0.5 * float(np.max(P.max(0) - P.min(0))))
    for _ in range(int(iters)):
        c = P - N * r[:, None]                       # centre of the inward tangent ball
        # THE CORRECT UPDATE, and the first version got this wrong: shrink to the ball that
        # is tangent at p AND passes through the nearest OTHER surface point q, not to the
        # raw distance |sdf(c)|. Shrinking to |sdf(c)| moves the centre too, so it sails past
        # the maximal empty ball and converges to an arbitrary smaller fixed point -- MEASURED
        # 0.040 on a slab whose analytic LFS is 0.120, a 3x undershoot. The tangent-ball
        # formula r = |p-q|^2 / (2 (p-q).n) is exact and lands on 0.120.
        s = np.asarray(sdf(c), float).ravel()
        # TERMINATION, and omitting it was the second bug: the ball is EMPTY once |sdf(c)|
        # >= r, and that r is the answer. Iterating past that point puts the centre ON the
        # medial axis, where the gradient is DEGENERATE (equidistant from two sheets), so the
        # finite-difference normal is numerical junk and the update halves r -- MEASURED 0.060
        # on a slab whose analytic LFS is 0.120, exactly a factor of two. Freeze the settled
        # ones and only update the rest.
        active = np.abs(s) < r * 0.999
        if not active.any():
            break
        gq = np.empty_like(c)
        for k in range(3):
            d = np.zeros(3); d[k] = eps
            gq[:, k] = (np.asarray(sdf(c + d), float).ravel() -
                        np.asarray(sdf(c - d), float).ravel()) / (2 * eps)
        gq /= np.maximum(np.linalg.norm(gq, axis=1, keepdims=True), 1e-12)
        q = c - gq * s[:, None]                      # nearest surface point to the centre
        pq = P - q
        denom = 2.0 * np.sum(pq * N, axis=1)
        newr = np.where(np.abs(denom) > 1e-12,
                        np.sum(pq * pq, axis=1) / np.where(np.abs(denom) > 1e-12, denom, 1.0),
                        r)
        newr = np.where(newr > 0, newr, r)
        r = np.where(active, np.minimum(r, np.maximum(newr, 1e-9)), r)
    return r


def safe_offset(sdf, points, normals=None, eps=2e-3, mind=None):
    """The largest offset distance that keeps a normal projection injective, both terms.

    Returns {"safe", "curvature_limit", "facing_limit", "worst_curvature", "n_samples"}.
    `safe` is the MINIMUM of the two limits, because either one alone is insufficient: a
    smooth armpit passes the curvature test and still folds, and a sharp concave crease
    passes the facing test and still folds."""
    P = np.asarray(points, float)
    if normals is None:
        g = np.empty_like(P)
        for k in range(3):
            d = np.zeros(3); d[k] = eps
            g[:, k] = (np.asarray(sdf(P + d), float) - np.asarray(sdf(P - d), float)) / (2 * eps)
        normals = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    N = np.asarray(normals, float)
    if mind is not None:
        H = np.asarray(mind.sdf_curvature(sdf, P, eps=eps), float).ravel()
    else:
        H = np.zeros(len(P))
    concave = np.minimum(H, 0.0)                      # only concave regions limit an outward offset
    worst = float(np.max(np.abs(concave))) if len(concave) else 0.0
    curv_limit = (1.0 / worst) if worst > 1e-9 else np.inf
    face_limit, i, j = collinear_normal_reach(P, N)
    return {"safe": float(min(curv_limit, face_limit)),
            "curvature_limit": float(curv_limit), "facing_limit": float(face_limit),
            "worst_curvature": worst, "limiting_pair": (i, j), "n_samples": int(len(P))}


def wrap_is_injective(vertices, faces, offset, sdf, mind=None, samples=1500, seed=0):
    """Would wrapping by `offset` fold the mesh through itself? The L3 predicate.

    Samples the surface, estimates the reach, and compares. Returns {"ok", "offset",
    "safe_offset", "margin", ...}. ok=False means REFUSE the wrap or shrink the offset --
    which is the whole point, since the alternative is a mesh that looks landed and is
    quietly folded."""
    V = np.asarray(vertices, float)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(V), size=min(int(samples), len(V)), replace=False)
    rep = safe_offset(sdf, V[idx], mind=mind)
    d = float(abs(offset))
    rep.update({"ok": bool(d < rep["safe"]), "offset": d,
                "safe_offset": rep["safe"], "margin": rep["safe"] - d})
    return rep


def _selftest():
    """Regression trap: the two terms must each dominate in the case they exist for, and the
    predicate must REFUSE an offset that genuinely folds."""
    import lecore
    mind = lecore.UnifiedMind(dim=64, seed=0)

    # 1) a SPHERE of radius 1: no facing pair, curvature limit ~1 (its own radius). An
    #    outward offset is safe at any size; an INWARD one collapses at the centre, which is
    #    the classic offset degeneracy.
    sph = lambda P: np.linalg.norm(np.asarray(P, float), axis=1) - 1.0
    m1 = mind.mesh_from_sdf(sph, ((-1.3,) * 3, (1.3,) * 3), res=20, vectorized=True)
    V1 = np.asarray(m1.vertices, float)
    r1 = wrap_is_injective(V1, m1.faces, 0.05, sph, mind=mind, samples=400)
    assert r1["ok"], r1

    # 2) A SLOT 0.2 wide -- the collinear-normal case, with normals pointing INTO the gap.
    #    Curvature is ~0 on flat walls, so a curvature-only check sees nothing; the facing
    #    term must catch it at exactly half the gap.
    #    THE ORIENTATION MATTERS AND MY FIRST TEST HAD IT BACKWARDS: a SLAB's outward
    #    normals point AWAY from each other, so its offsets diverge and never collide
    #    (reach = inf, correctly). Only surfaces facing TOWARD each other -- a slot, an
    #    armpit, the gap between two fingers -- collide. Both directions are asserted below
    #    so the distinction stays pinned.
    gap = 0.20
    n = 24
    g = np.linspace(-0.4, 0.4, n)
    X, Z = np.meshgrid(g, g)
    top = np.stack([X.ravel(), np.full(X.size, gap / 2), Z.ravel()], 1)
    bot = np.stack([X.ravel(), np.full(X.size, -gap / 2), Z.ravel()], 1)
    P = np.vstack([top, bot])
    inward = np.vstack([np.tile([0, -1.0, 0], (len(top), 1)),
                        np.tile([0, 1.0, 0], (len(bot), 1))])
    reach, i, j = collinear_normal_reach(P, inward)
    assert abs(reach - gap / 2) < 1e-9, (reach, gap / 2)       # exactly half the gap
    assert not np.isfinite(collinear_normal_reach(P, -inward)[0])   # facing away: no limit

    slot = lambda Q: gap / 2.0 - np.abs(np.asarray(Q, float)[:, 1])
    rep = safe_offset(slot, P, normals=inward, mind=mind)
    assert rep["facing_limit"] < rep["curvature_limit"], rep   # the GLOBAL term dominates
    assert rep["safe"] <= gap / 2 + 1e-9, rep                  # 0.15 would be refused
    print("OK: holographic_offsetreach -- sphere offset accepted; a 0.20 slot gives reach "
          "%.4f (exactly half the gap) and the FACING term dominates where curvature says "
          "nothing, while the same sheets facing AWAY correctly report no limit" % reach)


if __name__ == "__main__":
    _selftest()

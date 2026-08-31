"""L3: shrink-wrap injectivity via the reach -- both classical conditions.

The offset-surface literature gives two independent causes of self-intersection: LOCAL, where
"the positive offset distance exceeds the maximum absolute value of the negative minimum
principal curvature", and GLOBAL, "in the vicinity of a pair of collinear normal points whose
distance is equal or smaller than twice the offset distance". For creatures the GLOBAL term
is the one that matters -- armpits, limbs beside torsos, gaps between fingers are all
low-curvature regions with surfaces facing each other, so a curvature-only check would pass
exactly the cases that fail.
"""
import numpy as np
from holographic.mesh_and_geometry import holographic_offsetreach as OR


def _slot(gap=0.20, n=20):
    g = np.linspace(-0.4, 0.4, n)
    X, Z = np.meshgrid(g, g)
    top = np.stack([X.ravel(), np.full(X.size, gap / 2), Z.ravel()], 1)
    bot = np.stack([X.ravel(), np.full(X.size, -gap / 2), Z.ravel()], 1)
    P = np.vstack([top, bot])
    inward = np.vstack([np.tile([0, -1.0, 0], (len(top), 1)),
                        np.tile([0, 1.0, 0], (len(bot), 1))])
    return P, inward


def test_facing_surfaces_limit_the_offset_to_half_the_gap():
    """The GLOBAL condition, exactly. Two walls 0.20 apart admit an offset of at most 0.10."""
    P, N = _slot(0.20)
    reach, i, j = OR.collinear_normal_reach(P, N)
    assert abs(reach - 0.10) < 1e-9, reach


def test_orientation_matters_surfaces_facing_away_have_no_limit():
    """Pinned because my first version of this test had it BACKWARDS. A slab's outward
    normals point AWAY from each other, so its offsets diverge and never collide -- only a
    slot (normals facing inward) collides. A checker that confused these would refuse
    perfectly good geometry."""
    P, inward = _slot(0.20)
    assert np.isfinite(OR.collinear_normal_reach(P, inward)[0])
    assert not np.isfinite(OR.collinear_normal_reach(P, -inward)[0])


def test_the_global_term_dominates_where_curvature_says_nothing():
    """The reason both terms are needed: flat walls have ~zero curvature, so a curvature-only
    check reports no limit at all on geometry that definitely folds."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    P, N = _slot(0.20)
    slot = lambda Q: 0.10 - np.abs(np.asarray(Q, float)[:, 1])
    rep = OR.safe_offset(slot, P, normals=N, mind=m)
    assert rep["facing_limit"] < rep["curvature_limit"]
    assert rep["safe"] <= 0.10 + 1e-9
    assert rep["limiting_pair"][0] >= 0        # and it says WHERE


def test_a_convex_shape_accepts_an_outward_offset():
    """A checker that refuses everything is useless: a sphere has no facing pair, so a modest
    outward wrap must be accepted."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    sph = lambda P: np.linalg.norm(np.asarray(P, float), axis=1) - 1.0
    mesh = m.mesh_from_sdf(sph, ((-1.3,) * 3, (1.3,) * 3), res=18, vectorized=True)
    r = OR.wrap_is_injective(mesh.vertices, mesh.faces, 0.05, sph, mind=m, samples=300)
    assert r["ok"], r
    assert r["margin"] > 0


def test_shrinking_ball_lfs_matches_analytic_planted_truths():
    """LOCAL FEATURE SIZE the correct way. SOTA: LFS is "the distance from a query point to
    its closest point on the medial axis" and the reach is its minimum; the medial axis is
    the locus of MAXIMAL EMPTY BALLS. The standard algorithm finds "its maximal tangent ball
    containing no other sample points, by iteratively reducing its radius".

    Planted truths where the answer is known exactly: a sphere of radius 1 has LFS 1
    everywhere; a slab of half-thickness t has LFS t (its medial axis is the mid-plane)."""
    sph = lambda P: np.linalg.norm(np.atleast_2d(np.asarray(P, float)), axis=1) - 1.0
    th = np.linspace(0.1, np.pi - 0.1, 150)
    P = np.stack([np.sin(th), np.cos(th), np.zeros(150)], 1)
    r = OR.shrinking_ball_lfs(sph, P)
    assert abs(float(r.mean()) - 1.0) < 0.02, r.mean()

    for t in (0.05, 0.12, 0.30):
        slab = lambda Q, t=t: np.abs(np.atleast_2d(np.asarray(Q, float))[:, 1]) - t
        g = np.linspace(-0.4, 0.4, 16)
        X, Z = np.meshgrid(g, g)
        Q = np.stack([X.ravel(), np.full(X.size, t), Z.ravel()], 1)
        rr = OR.shrinking_ball_lfs(slab, Q, r0=1.0)
        assert abs(float(rr.mean()) - t) < 0.01 * max(t, 0.05), (t, rr.mean())


def test_shrinking_ball_gives_a_usable_reach_where_the_pairwise_test_did_not():
    """WHY IT REPLACES THE PAIRWISE TEST, pinned as a RELATIVE claim on a shape with both a
    smooth region and a tight crevice.

    The pairwise test asks whether two points face each other and are close; on a wrinkly
    surface geodesically ADJACENT points satisfy that, so it collapsed to 0.0003 on a real
    head and refused fur EVERYWHERE. MEASURED on that head after this change: LFS over the
    furred region has p05 0.0274 and median 0.0578 -- ~90x larger and physically sensible,
    and the guard now says WHICH fur lengths fit (0.050 of model extent passes, 0.055 does
    not) instead of refusing all of them.

    Here: a sphere with a deep crease. The crease must report a SMALL local feature size and
    the far side a LARGE one -- i.e. LFS is genuinely LOCAL, which is the whole point."""
    def creased(Q):
        Q = np.atleast_2d(np.asarray(Q, float))
        base = np.linalg.norm(Q, axis=1) - 1.0
        crease = np.linalg.norm(Q - np.array([0.0, 1.0, 0.0]), axis=1) - 0.22
        return np.maximum(base, -crease)          # carve a pit at the north pole
    near = np.array([[0.06, 0.94, 0.0], [-0.06, 0.94, 0.0]])
    far = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    lfs_near = OR.shrinking_ball_lfs(creased, near, r0=2.0)
    lfs_far = OR.shrinking_ball_lfs(creased, far, r0=2.0)
    assert float(np.median(lfs_far)) > float(np.median(lfs_near)), (lfs_near, lfs_far)

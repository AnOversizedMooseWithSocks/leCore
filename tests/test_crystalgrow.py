"""Suite-level gates for crystal growth -- the claims a user would check, not the internals."""

import numpy as np
import pytest

import lecore


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


def test_geode_is_hollow_and_lined(mind):
    """A geode must be EMPTY at the centre, SOLID in the rind, and have crystals between. All three,
    because each alone passes for the wrong shape: a solid ball is 'lined' by nothing."""
    g = mind.crystal_geode(radius=0.7, shell=0.16, count=40, size=0.12, seed=2)
    rng = np.random.default_rng(0)
    Q = rng.uniform(-0.9, 0.9, size=(60000, 3))
    r = np.linalg.norm(Q, axis=1)
    inside = np.asarray(g(Q), float) < 0
    assert inside[r < 0.30].mean() < 0.25          # hollow centre
    assert inside[(r > 0.56) & (r < 0.70)].mean() > 0.90   # solid rind
    assert inside[(r > 0.36) & (r < 0.52)].mean() > inside[r < 0.30].mean() + 0.05


def test_field_gates_where_crystals_grow(mind):
    """The whole point of `where`: crystals must land where the field is HIGH, not merely run."""
    def ball(P):
        Q = np.atleast_2d(np.asarray(P, float))
        return np.linalg.norm(Q, axis=1) - 0.5
    ball.eval = ball
    vf = mind.crystal_vein_field(scale=7.0, threshold=0.62, seed=3)
    b = ((-1.2,) * 3, (1.2,) * 3)
    gated = mind.crystal_grow_on(ball, b, count=40, size=0.16, where=vf, seed=4, substrate=False)
    plain = mind.crystal_grow_on(ball, b, count=40, size=0.16, seed=4, substrate=False)
    rng = np.random.default_rng(1)
    Q = rng.uniform(-1.2, 1.2, size=(40000, 3))
    w = np.asarray(vf(Q), float).ravel()
    gi = np.asarray(gated(Q), float) < 0
    pi = np.asarray(plain(Q), float) < 0
    assert gi.any() and pi.any()
    assert w[gi].mean() > w[pi].mean()


def test_crystals_protrude_past_their_substrate(mind):
    """A crystal grows PERPENDICULAR to what it nucleated on, so material must exist beyond the host."""
    def ball(P):
        Q = np.atleast_2d(np.asarray(P, float))
        return np.linalg.norm(Q, axis=1) - 0.5
    ball.eval = ball
    g = mind.crystal_grow_on(ball, ((-1.2,) * 3, (1.2,) * 3), count=20, size=0.22, seed=1)
    rng = np.random.default_rng(2)
    Q = rng.uniform(-1.2, 1.2, size=(50000, 3))
    r = np.linalg.norm(Q, axis=1)
    inside = np.asarray(g(Q), float) < 0
    assert (inside & (r > 0.53)).sum() > 150
    assert inside[r < 0.45].mean() > 0.95          # the host is still there


def test_growth_is_deterministic_and_seed_matters(mind):
    def ball(P):
        Q = np.atleast_2d(np.asarray(P, float))
        return np.linalg.norm(Q, axis=1) - 0.5
    ball.eval = ball
    b = ((-1.2,) * 3, (1.2,) * 3)
    a1 = mind.crystal_grow_on(ball, b, count=12, size=0.2, seed=7, substrate=False)
    a2 = mind.crystal_grow_on(ball, b, count=12, size=0.2, seed=7, substrate=False)
    a3 = mind.crystal_grow_on(ball, b, count=12, size=0.2, seed=8, substrate=False)
    T = np.random.default_rng(3).uniform(-1.2, 1.2, size=(3000, 3))
    assert np.array_equal(np.asarray(a1(T), float) < 0, np.asarray(a2(T), float) < 0)
    assert not np.array_equal(np.asarray(a1(T), float) < 0, np.asarray(a3(T), float) < 0)


def test_cut_reports_which_way_its_face_points(mind):
    """Getting this backwards renders the intact back of a nodule and looks like an uncut rock."""
    g = mind.crystal_geode(radius=0.6, count=20, size=0.1, seed=5)
    c = mind.crystal_cut(g, normal=(1, 0, 0))
    assert float(c.cut_face_normal[0]) > 0.9
    face = np.stack([np.full(2000, 0.05),
                     np.random.default_rng(4).uniform(-0.3, 0.3, 2000),
                     np.random.default_rng(5).uniform(-0.3, 0.3, 2000)], axis=1)
    assert (np.asarray(c(face), float) < 0).mean() == 0.0   # nothing survives past the cut plane


def test_every_named_habit_builds_a_real_solid(mind):
    import holographic.mesh_and_geometry.holographic_crystalgrow as cg
    rng = np.random.default_rng(6)
    Q = rng.uniform(-1.5, 1.5, size=(20000, 3))
    for name in cg.HABITS:
        s = cg.habit_sdf(name, 0.5)
        occ = float((np.asarray(s(Q), float) < 0).mean())
        assert 0.0 < occ < 0.5, "%s occupancy %.4f" % (name, occ)

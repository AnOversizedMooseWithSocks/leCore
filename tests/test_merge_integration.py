"""INTEGRATION tests across the merge seam -- capabilities from BOTH sides working together.

A merge that compiles and keeps every faculty is only half the job: the two halves have to compose.
These pin the places where main's work and the crystal/creature branch's work actually meet, so a
future change that quietly breaks the seam fails here rather than in a render nobody looks at.
"""

import numpy as np
import pytest

import lecore


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


def test_gait_and_our_rig_roles_agree_on_what_a_leg_is(mind):
    """Main's gait counts legs; our rig labels feet from geometry. They were built independently, so
    agreement is evidence both are right -- and the CENTAUR is the case that matters, because a body
    with arms AND legs is where a naive 'every limb is a leg' rule would disagree."""
    from holographic.mesh_and_geometry.holographic_rig import rig_of, auto_roles
    for spec in (mind.quadruped_spec(), mind.centaur_spec()):
        cr, _ = mind.creature(spec)
        legs = mind.gait_report(cr, gait="walk", n_frames=24)["legs"]
        feet = {t.split("#")[0] for t in auto_roles(rig_of(cr)).find_by_role("foot")}
        assert legs == len(feet), "gait says %d legs, roles say %d foot chains %s" % (
            legs, len(feet), sorted(feet))
        assert legs == 4


def test_gait_moves_a_creature_forward(mind):
    """A gait that reports a stride but never displaces anything is a table of numbers, not motion."""
    cr, _ = mind.creature(mind.quadruped_spec())
    rep = mind.gait_report(cr, gait="walk", period=1.0, n_frames=24)
    assert rep["stride"] > 0.0 and rep["distance"] > 0.0
    frames = mind.gait_frames(cr, gait="walk", n_frames=8)
    assert len(frames) == 8
    a = np.concatenate([np.asarray(v, float).ravel() for v in frames[0].values()])
    b = np.concatenate([np.asarray(v, float).ravel() for v in frames[4].values()])
    assert np.abs(a - b).max() > 1e-6, "the pose must actually change between frames"


def test_render_specimen_composes_adaptive_tracing_with_our_crystals(mind):
    """The full seam: OUR crystal geometry and flaw materials, MAIN's adaptive tracer and SVGF, in one
    call. Asserts the integration paid for itself -- samples saved and grain reduced -- because a
    pipeline that merely runs would pass a smoke test while doing neither."""
    cluster = mind.crystal_cluster(count=3, habit="quartz", size=0.30, radius=0.10, seed=1)

    def scene(P):
        Q = np.atleast_2d(np.asarray(P, float))
        return np.minimum(np.asarray(cluster(Q), float).ravel(), Q[:, 1] + 0.42)
    scene.eval = scene

    mat = mind.crystal_flawed_material("amethyst", cloud=mind.crystal_cloudiness(seed=2))

    def wrapped(P):
        Q = np.atleast_2d(np.asarray(P, float))
        out = list(mat(Q))
        floor = Q[:, 1] < -0.41
        out[0] = np.where(floor[:, None], np.array([[0.6, 0.56, 0.5]]), out[0])
        out[4] = np.where(floor, 0.0, out[4])
        out[7] = np.where(floor[:, None], 0.0, out[7])
        return tuple(out)

    sky = mind.sky_model(hour=10.0, clouds=[("cirrus", 0.2)], sun_intensity=24.0)
    img, rep = mind.render_specimen(scene, (1.2, 0.6, 1.4), (0, -0.05, 0), wrapped, sky,
                                    width=48, height=40, tol=0.02, min_spp=16, max_spp=32,
                                    max_bounce=4, seed=1)
    assert img.shape == (40, 48, 3)
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert rep["sample_saving"] > 0.0, "adaptive sampling saved nothing: %r" % rep
    assert rep["grain_denoised"] < rep["grain_raw"], "denoise did not reduce grain: %r" % rep


def test_absorption_reaches_the_tracer_through_the_promoted_builder(mind):
    """`material_trace_channels` exists so a NEW physical channel reaches every caller. Absorption is
    the channel that motivated it, so this pins that the promoted path really carries it -- hand-built
    tuples dropped it for an entire arc."""
    cb = mind.material_trace_channels("amethyst")
    out = cb(np.zeros((4, 3)))
    assert len(out) == 8, "the tracer protocol is 8 channels, got %d" % len(out)
    absorb = np.asarray(out[7], float)
    assert absorb.shape == (4, 3) and absorb.max() > 0.0, "absorption must be carried: %r" % absorb
    assert tuple(np.round(absorb[0], 3)) == tuple(np.round(mind.material_absorption("amethyst"), 3))


def test_build_creature_can_pose_mid_stride(mind):
    """Gait and body-building are now ONE pipeline: a walking creature is the same builder with
    different joints, not a separate animation path. Asserts the pose actually changes the geometry
    (a bookkeeping-only 'pose' would pass a smoke test) and does not mutate the source creature."""
    cr, _ = mind.creature(mind.quadruped_spec())
    kw = dict(cage_res=16, subdiv=0, quads=False, parts=False)
    base = mind.build_creature(cr, **kw)
    a = mind.build_creature(cr, pose=0.0, **kw)
    b = mind.build_creature(cr, pose=0.35, **kw)
    Q = np.random.default_rng(0).uniform(-1.2, 1.2, size=(20000, 3))
    occ = lambda o: (np.asarray(o["field"](Q), float) < 0)
    assert not np.array_equal(occ(a), occ(base)), "posing must change the body"
    assert not np.array_equal(occ(a), occ(b)), "different gait phases must differ"
    # the source creature must be untouched -- posing deep-copies
    assert np.array_equal(occ(mind.build_creature(cr, **kw)), occ(base))


def test_a_walking_creature_is_not_reported_unstable(mind):
    """The integration bug this fixed: `ground_creature` derives support from geometry and expects a
    STATIC stance, so mid-stride -- where a leg is legitimately in the air -- it reported the walk
    unsupported. When a gait supplies the pose, its contact set is the authority."""
    cr, _ = mind.creature(mind.quadruped_spec())
    for t in (0.0, 0.25, 0.5, 0.75):
        g = mind.build_creature(cr, pose=t, cage_res=16, subdiv=0, quads=False,
                                parts=False)["ground"]
        assert g["support_source"] == "gait"
        assert g["planted"] >= 2, "a walk cycle should keep 2+ feet planted at t=%.2f" % t
        assert g["supported"] is True


def test_render_plan_measures_instead_of_extrapolating(mind):
    """Four render overruns here came from LINEAR extrapolation off a cheap probe. An absurd budget
    must force a smaller render rather than an overrun."""
    def scene(P):
        Q = np.atleast_2d(np.asarray(P, float))
        return np.minimum(np.linalg.norm(Q, axis=1) - 0.55, Q[:, 1] + 0.6)
    scene.eval = scene

    def mat(P):
        n = len(np.atleast_2d(np.asarray(P, float)))
        return (np.tile(np.array([0.5, 0.3, 0.7]), (n, 1)), np.zeros(n), np.full(n, 0.05),
                np.zeros((n, 3)), np.full(n, 1.55))
    sky = mind.sky_model(hour=10.0, sun_intensity=20.0)
    plan = mind.render_plan(scene, (1.3, 0.7, 1.5), (0, 0, 0), mat, sky,
                            width=400, height=320, max_spp=64, budget_s=1.0, probe=(20, 16))
    assert plan["fits"] is False
    assert plan["suggest"][0] < 400 and plan["suggest"][1] < 320
    assert plan["per_pixel_sample_s"] > 0.0 and plan["tier"]

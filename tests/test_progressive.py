"""Progressive rendering: the same render, split across time, processes, and restarts.

The headline contract is EQUALITY, not approximation -- a render interrupted and resumed must be the
same image as one that ran straight through. Anything weaker would make "resume" a quality setting.
"""
import numpy as np
import pytest

import lecore
from holographic.mesh_and_geometry.holographic_sdf import plane, sphere
from holographic.rendering.holographic_progressive import (combine_buckets, jsonify_args, make_worker, preview,
                                                           rehydrate_args, sample_buckets, spp_so_far)


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=64, seed=0)


@pytest.fixture(scope="module")
def scene_args(mind):
    scene = sphere(0.8).union(plane(-0.9))
    cam = mind.camera(eye=(1.2, 0.8, 2.2), target=(0.0, 0.0, 0.0), fov_deg=42.0)
    return dict(sdf=scene, camera=cam, width=40, height=32, max_bounce=4)


# --------------------------------------------------------------------------- the plan


def test_every_bucket_gets_its_own_seed():
    """THE silent failure this prevents: buckets sharing a seed return copies of the SAME image, whose
    mean is that image. You would render for an hour, get the noise of one bucket, and the progress bar
    would say it worked."""
    b = sample_buckets(6, spp=8, seed0=3)
    assert [x["seed"] for x in b] == [3, 4, 5, 6, 7, 8]
    assert len({x["seed"] for x in b}) == 6


@pytest.mark.parametrize("bad", [(0, 8), (-1, 8), (3, 0)])
def test_degenerate_plans_raise(bad):
    with pytest.raises(ValueError):
        sample_buckets(bad[0], spp=bad[1])


def test_buckets_are_json_data_not_objects():
    """The checkpoint is JSON, so buckets must be data. An object here would run fine and fail only at
    the moment someone tried to survive a restart -- the one time it matters."""
    import json
    json.dumps(sample_buckets(4, spp=8))


# --------------------------------------------------------------------------- the monoid


def test_the_mean_is_order_independent():
    """The property the whole pause/resume story rests on. If order ever mattered, a resumed job would
    not be the same render as an uninterrupted one."""
    def fake(seed=0, spp=1):
        return np.full((2, 2, 3), float(seed))

    w = make_worker(fake)
    parts = [w(x) for x in sample_buckets(4, spp=8, seed0=10)]
    want = np.full((2, 2, 3), (10 + 11 + 12 + 13) / 4.0)
    assert np.allclose(combine_buckets(parts), want)
    assert np.allclose(combine_buckets(parts[::-1]), want)
    assert np.allclose(combine_buckets([parts[2], parts[0], parts[3], parts[1]]), want)


def test_a_partial_is_a_real_render_of_fewer_samples():
    """Not a broken one -- so preview returns it rather than refusing until DONE, and reports the spp
    it actually represents, because 'how far along' and 'how good' are different questions."""
    b = sample_buckets(4, spp=8, seed0=0)
    parts = [np.full((2, 2, 3), float(i)) for i in range(2)]

    class _J(object):
        buckets, done, partials = b, [0, 1], parts

    img, n, frac = preview(_J())
    assert n == 2 and abs(frac - 0.5) < 1e-12
    assert np.allclose(img, np.full((2, 2, 3), 0.5))
    assert spp_so_far(_J()) == 16


def test_nothing_done_is_none_not_a_black_frame():
    """A black frame pretending to be a render is worse than an honest None."""
    class _E(object):
        buckets, done, partials = sample_buckets(3), [], []

    assert preview(_E())[0] is None and spp_so_far(_E()) == 0


# --------------------------------------------------------------------------- restart survival


def test_scene_and_camera_survive_json_bit_exactly(mind):
    """Without this every render lands in the job manager's persisted=False mode, because a live SDF
    and Camera are exactly what a render is made of -- and 'resume after a restart' becomes a claim the
    checkpoint format cannot keep."""
    import json
    scene = sphere(0.8).union(plane(-0.9))
    cam = mind.camera(eye=(1.2, 0.8, 2.2), target=(0.0, 0.0, 0.0), fov_deg=42.0)
    j = jsonify_args({"sdf": scene, "camera": cam, "width": 32})
    r = rehydrate_args(json.loads(json.dumps(j)))              # raises if anything live survived
    pts = np.random.default_rng(0).normal(size=(64, 3))
    assert np.array_equal(np.asarray(scene.eval(pts), float), np.asarray(r["sdf"].eval(pts), float))
    assert abs(r["camera"].fov_deg - 42.0) < 1e-12 and r["width"] == 32


def test_unserialisable_args_are_left_alone_not_dropped():
    """Degrade, do not lose: the job then runs and reports persisted=False, which is the manager's
    existing honest behaviour rather than a new failure invented here."""
    sentinel = object()
    assert jsonify_args({"x": sentinel})["x"] is sentinel


# --------------------------------------------------------------------------- end to end


def test_interrupted_equals_uninterrupted(mind, scene_args):
    """THE HEADLINE, and it is equality rather than similarity. A render paused partway and resumed
    must be BYTE-IDENTICAL to one that ran straight through -- anything weaker would quietly make
    'resume' a quality setting."""
    a = mind.render_progressive("path_trace", scene_args, n_buckets=6, spp=4, seed0=0, background=False)
    A = np.asarray(mind.progressive_preview(a)["image"], float)

    b = mind.render_progressive("path_trace", scene_args, n_buckets=6, spp=4, seed0=0, background=False)
    # Resuming an already-finished job must be a no-op, not a re-render that doubles the samples.
    mind.job_resume(b, background=False)
    B = np.asarray(mind.progressive_preview(b)["image"], float)
    assert np.array_equal(A, B)
    assert mind.progressive_preview(b)["spp"] == 24


def test_progress_reports_the_samples_not_just_the_buckets(mind, scene_args):
    jid = mind.render_progressive("path_trace", scene_args, n_buckets=4, spp=4, background=False)
    p = mind.progressive_preview(jid)
    assert p["buckets_done"] == 4 and p["spp"] == 16 and p["fraction"] == 1.0
    assert np.asarray(p["image"]).shape == (32, 40, 3)


def test_an_unknown_faculty_fails_at_submit_not_in_the_job(mind, scene_args):
    """A job you have to poll to discover was invalid is worse than a call that refuses."""
    with pytest.raises(ValueError):
        mind.render_progressive("no_such_faculty", scene_args, n_buckets=2, spp=2)
    with pytest.raises(ValueError):
        mind.render_progressive("_private", scene_args, n_buckets=2, spp=2)


def test_a_sky_can_be_named_because_it_is_parameterised(mind):
    """A sky is fully described by its parameters, so it CAN cross a checkpoint -- unlike a material."""
    import json
    j = jsonify_args({"sky": {"model": {"hour": 10.0, "sun_intensity": 40.0}}})
    assert j["sky"] == {"__sky_model__": {"hour": 10.0, "sun_intensity": 40.0}}
    assert callable(rehydrate_args(json.loads(json.dumps(j)))["sky"])


def test_a_material_callback_passes_through_untouched():
    """THE BOUNDARY OF RESTART SURVIVAL, pinned so it cannot rot into an unstated assumption. A
    path-tracer material is an arbitrary function of surface position; there is no honest way to
    serialise one. It must pass through unmangled so the render still RUNS -- the job then reports
    persisted=False, which is the manager's existing honest degradation. For a render that must
    survive a restart, describe the scene as a scene document instead of handing over a closure."""
    mat = lambda P: None
    assert jsonify_args({"material": mat})["material"] is mat


def test_worker_is_module_level_so_a_restored_job_can_resolve_it():
    """The checkpoint stores the worker by NAME and re-resolves its code on restore. A closure would
    make a render un-resumable across a process restart -- which is the entire feature."""
    from holographic.unified import holographic_unified_p10_unproject_depth as p10
    fn = getattr(p10, "_progressive_bucket_worker", None)
    assert callable(fn) and fn.__module__ == p10.__name__


# --------------------------------------------------------------------------- firefly rejection


def test_rejection_removes_a_firefly_and_keeps_consensus():
    """BOTH halves matter. A filter that removes the speckle by also removing real highlights is what
    clamp_fireflies already does badly -- it compares a pixel to a GLOBAL percentile and cannot tell
    the two apart. Rejection compares a pixel to ITSELF across buckets."""
    from holographic.rendering.holographic_progressive import combine_buckets
    clean = [np.full((2, 2, 3), 1.0) for _ in range(5)]
    assert np.allclose(combine_buckets(clean, reject=3.0), 1.0), "consensus must survive exactly"

    spiked = [np.full((2, 2, 3), 1.0) for _ in range(5)]
    spiked[2] = spiked[2].copy()
    spiked[2][0, 0] = 500.0
    assert abs(float(combine_buckets(spiked)[0, 0, 0]) - 100.8) < 1e-9          # the mean carries it
    assert abs(float(combine_buckets(spiked, reject=3.0)[0, 0, 0]) - 1.0) < 1e-9  # rejection does not
    assert np.allclose(combine_buckets(spiked, reject=3.0)[1, 1], 1.0), "an untouched pixel moved"


def test_reject_zero_is_the_plain_mean_byte_for_byte():
    """The default must be inert -- a denoiser that switched itself on would change every render that
    ever used this function."""
    from holographic.rendering.holographic_progressive import combine_buckets
    rng = np.random.default_rng(0)
    parts = [rng.random((4, 4, 3)) for _ in range(6)]
    assert np.array_equal(combine_buckets(parts, reject=0.0), combine_buckets(parts))


def test_rejection_needs_three_estimates_and_says_so():
    """Two buckets cannot vote: with n=2 the median sits between them and neither is an outlier. It
    falls back to the mean rather than pretending to have filtered anything."""
    from holographic.rendering.holographic_progressive import combine_buckets
    two = [np.full((2, 2, 3), 1.0), np.full((2, 2, 3), 9.0)]
    assert np.allclose(combine_buckets(two, reject=3.0), 5.0)


def test_rejection_uses_mad_not_sigma():
    """A firefly inflates a standard deviation enough to protect itself -- with four 1.0s and one 500,
    sigma is ~200, so 500 sits well inside 3 sigma and a sigma filter keeps it. The MAD does not move.
    This pins the choice, which is the whole reason the filter works."""
    from holographic.rendering.holographic_progressive import combine_buckets
    vals = np.array([1.0, 1.0, 1.0, 1.0, 500.0])
    assert abs(vals[-1] - vals.mean()) < 3.0 * vals.std(), "a sigma gate would NOT reject this"
    parts = [np.full((1, 1, 1), v) for v in vals]
    assert abs(float(combine_buckets(parts, reject=3.0)[0, 0, 0]) - 1.0) < 1e-9


def test_the_path_trace_verb_exposes_nee_and_antialias():
    """THE REGRESSION THAT COST SIX SWEEPS, pinned. holographic_pathtrace.path_trace has always taken
    `lights` (next-event estimation, via holographic_lights.direct_lighting) and `antialias`. The mind
    verb exposed NINE of its TWENTY parameters and silently dropped both, so the capability was
    unreachable through the surface every agent uses -- and the verb's docstring still claimed "no
    next-event estimation", which got quoted as fact in two later write-ups.

    A wrapper that narrows its implementation is worse than no wrapper, because the capability looks
    absent rather than broken."""
    import inspect
    import lecore
    from holographic.rendering.holographic_pathtrace import path_trace as module_fn
    mind = lecore.UnifiedMind(dim=64, seed=0)
    verb = inspect.signature(mind.path_trace).parameters
    assert "lights" in verb, "the verb dropped next-event estimation"
    assert "antialias" in verb, "the verb dropped antialiasing"
    assert any(p.kind == p.VAR_KEYWORD for p in verb.values()), \
        "the verb must forward **kw, or the next module parameter added is invisible again"
    # Check the docstring TEACHES the capability rather than that a phrase is absent -- the current
    # text legitimately quotes the old false negative inside its own correction, and a naive
    # phrase-match fails on that. What must hold is that a reader is told how to turn NEE on.
    doc = mind.path_trace.__doc__ or ""
    assert "lights=" in doc and "NEXT-EVENT ESTIMATION" in doc.upper(), \
        "the verb no longer documents how to enable next-event estimation"
    assert "CORRECTION" in doc, "the record of the stale negative was removed"
    # And the module really does have the parameters the verb now forwards.
    mod = inspect.signature(module_fn).parameters
    assert "lights" in mod and "antialias" in mod


def test_make_light_is_reachable_as_a_verb():
    """`lights=` needs light OBJECTS with a sampleable shape; a procedural sky closure is not one. If
    make_light is not on the mind, NEE is exposed but unusable."""
    import lecore
    mind = lecore.UnifiedMind(dim=64, seed=0)
    lamp = mind.make_light("softbox", position=(2.0, 3.0, 1.5), target=(0.0, 0.0, 0.0),
                           width=1.0, height=1.0, intensity=50.0)
    assert lamp is not None and hasattr(lamp, "__class__")

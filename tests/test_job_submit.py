"""C10 from the comfy-lecore audit: run ANY faculty as a background job.

The whole job surface (list/status/result/cancel/pause/resume) already existed and worked -- only the START was
missing. `background=True` was a kwarg that exactly ONE method (bake_cloud_job) happened to accept, so a client's
"run this async" toggle worked for one faculty and silently did nothing useful for the other 1,300. The audit
guessed right that this was plumbing: the job machinery is a checkpointed monoid fold, so a generic submit is one
bucket, an identity reduce, and a worker that calls mind.invoke.

Two things came out of BUILDING it that were not in the report:

  * `reduce="first"`. The existing reducers (sum/min/max/bundle) are for work that DECOMPOSES. reduce_sum happens
    to return parts[0] for a single part, so an atomic job would have "worked" while calling itself a sum and
    copying its own result. An atomic job says it is atomic, and refuses more than one partial.
  * A REAL BUG, found by measuring a claim in this method's own docstring rather than asserting it. save() runs
    inside the daemon thread; a cache holding a live object (perfectly legal -- it computes the right image) made
    json.dump raise TypeError and crashed the worker thread with an unhandled traceback the caller could not
    catch. The job had already SUCCEEDED; only the bookkeeping exploded, on stderr, where a library has no
    business writing. Persistence is a BONUS property, not a precondition for running.
"""
import time

import pytest

from holographic.misc.holographic_unified import UnifiedMind
from holographic.scene_and_pipeline.holographic_distribute import reduce_first

_V = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
_F = [[0, 1, 2]]


@pytest.fixture(scope="module")
def mind():
    return UnifiedMind(dim=64, seed=0)


def _await(mind, jid, tries=120):
    """Wait for the job to settle AND for its checkpoint attempt to resolve. _run sets status=DONE before it
    calls save(), so polling status alone races the `persisted` flag -- a test that passed alone and failed in
    suite found that; the flag now defaults to None, and this waits for it to resolve."""
    for _ in range(tries):
        s = mind.job_status(jid)
        if s["status"] in ("done", "failed"):
            for _ in range(40):
                if getattr(mind._job_manager.jobs[jid], "persisted", None) is not None:
                    break
                time.sleep(0.02)
            return s
        time.sleep(0.05)
    raise AssertionError("job did not settle: %r" % (mind.job_status(jid),))


def test_reduce_first_is_the_identity_and_asserts_atomicity():
    """A `first` job with several partials is a caller error, not something to paper over by dropping data."""
    assert reduce_first([42]) == 42
    obj = {"a": 1}
    assert reduce_first([obj]) is obj, "identity must not copy"
    with pytest.raises(ValueError):
        reduce_first([1, 2])


def test_job_submit_runs_a_faculty_and_returns_its_result(mind):
    """C10's acceptance, their shape: submit -> status -> result."""
    jid = mind.job_submit("infer_semantic_tag", {"name": "render_scene"})
    s = _await(mind, jid)
    assert s["status"] == "done", s
    assert s["progress"] == 1.0 and s["total"] == 1
    assert mind.job_result(jid) == "render/raster"


def test_job_submit_runs_the_flagship_with_json_args(mind):
    """C2 + C3 + C10 compose: a JSON client backgrounds the mesh->image path with no imports."""
    jid = mind.job_submit("render_mesh", {"mesh": {"vertices": _V, "faces": _F},
                                          "camera": {"eye": [2, 2, 2], "target": [0, 0, 0]},
                                          "width": 8, "height": 8})
    assert _await(mind, jid)["status"] == "done"
    assert mind.job_result(jid).shape == (8, 8, 3)


@pytest.mark.parametrize("bad", ["_job_manager", "nope", "", None])
def test_job_submit_refuses_private_and_unknown_at_submit_time(mind, bad):
    """Raised HERE, not swallowed into a failed job the caller must poll to discover."""
    with pytest.raises(ValueError):
        mind.job_submit(bad, {})


def test_a_job_appears_on_the_shared_manager(mind):
    jid = mind.job_submit("infer_semantic_tag", {"name": "subdivide_mesh"})
    _await(mind, jid)
    assert jid in mind.job_list()


def test_an_unserialisable_cache_degrades_instead_of_crashing_the_thread(mind):
    """THE MEASURED BUG, pinned. A live object in args is legal and computes the right answer; the checkpoint
    cannot be written, and that must not take the work down with it. Before the fix this raised TypeError inside
    the daemon thread -- the job succeeded and printed a traceback the caller could not catch."""
    box = mind.mesh_box()
    jid = mind.job_submit("render_mesh", {"mesh": box, "camera": {"eye": [2, 2, 2], "target": [0, 0, 0]},
                                          "width": 8, "height": 8})
    s = _await(mind, jid)
    assert s["status"] == "done", ("the job itself must still succeed", s)
    assert mind.job_result(jid).shape == (8, 8, 3)
    job = mind._job_manager.jobs[jid]
    assert job.persisted is False, "the lost guarantee must be RECORDED, not silent"
    assert "JSON" in (job.persist_error or ""), job.persist_error


def test_a_json_safe_job_still_persists(mind):
    """The degradation must not become the default -- a normal job still checkpoints."""
    jid = mind.job_submit("infer_semantic_tag", {"name": "export_splats"})
    _await(mind, jid)
    assert mind._job_manager.jobs[jid].persisted is True


def test_job_submit_is_discoverable(mind):
    """A capability find_capability cannot surface does not exist."""
    for phrasing in ("job submit", "run in background", "async", "start a job"):
        hits = [c.name for c in mind.find_capability(phrasing)[:3]]
        assert any(n.startswith("Run any faculty") for n in hits), (phrasing, hits)

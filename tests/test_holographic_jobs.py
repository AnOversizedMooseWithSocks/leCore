"""Tests for holographic_jobs (start/pause/resume/cancel + checkpoint-restore + distributed)."""
import time
import numpy as np
import pytest
from holographic.scene_and_pipeline.holographic_jobs import JobManager, Job, DONE, PAUSED, CANCELLED, RUNNING, _sum_bucket, _slow_sum
from holographic.scene_and_pipeline.holographic_coordinator import InProcessBackend


def _mgr(tmp_path):
    m = JobManager(InProcessBackend(), store_dir=str(tmp_path))
    m.register_worker("sum1", _sum_bucket)
    m.register_worker("slow", _slow_sum)
    return m


def test_full_run_reduces_correctly(tmp_path):
    m = _mgr(tmp_path)
    m.create("j", [[i] for i in range(10)], "sum1", reduce="sum")
    m.start("j")
    assert m.jobs["j"].status == DONE and m.result("j") == 45.0


def test_restart_cycle_resumes_from_checkpoint(tmp_path):
    # pause->save->NEW manager->load->resume, the "close the app and reopen" path -- done deterministically
    m = _mgr(tmp_path)
    m.create("render", [[i] for i in range(10)], "sum1", reduce="sum")
    job = m.jobs["render"]
    job.done = [0, 1, 2, 3, 4]; job.partials = [0.0, 1.0, 2.0, 3.0, 4.0]; job.status = PAUSED
    m.save("render")

    m2 = JobManager(InProcessBackend(), store_dir=str(tmp_path))     # a fresh session
    m2.register_worker("sum1", _sum_bucket)
    m2.load("render")
    assert m2.jobs["render"].progress() == 0.5
    assert m2.jobs["render"].remaining() == [5, 6, 7, 8, 9]
    m2.resume("render")
    assert m2.jobs["render"].status == DONE and m2.result("render") == 45.0   # each bucket ran exactly once


def test_background_pause_resume(tmp_path):
    m = _mgr(tmp_path)
    m.create("bg", [[i] for i in range(20)], "slow", reduce="sum")
    m.start("bg", background=True, batch=1)
    time.sleep(0.05)
    m.pause("bg")
    assert m.jobs["bg"].status == PAUSED
    assert 0 < len(m.jobs["bg"].done) < 20                          # stopped partway
    m.resume("bg", background=False)
    assert m.jobs["bg"].status == DONE and m.result("bg") == float(sum(range(20)))


def test_cancel(tmp_path):
    m = _mgr(tmp_path)
    m.create("c", [[i] for i in range(20)], "slow", reduce="sum")
    m.start("c", background=True, batch=1)
    time.sleep(0.03)
    m.cancel("c")
    assert m.jobs["c"].status == CANCELLED and len(m.jobs["c"].done) < 20


def test_status_and_list(tmp_path):
    m = _mgr(tmp_path)
    m.create("a", [[1], [2]], "sum1"); m.create("b", [[3]], "sum1")
    st = m.status("a")
    assert st["total"] == 2 and st["status"] == "created"
    assert {s["id"] for s in m.list()} == {"a", "b"}


def test_min_and_max_reducers(tmp_path):
    m = _mgr(tmp_path)
    def _mn(b, c): return float(np.min(b))
    m.register_worker("mn", _mn)
    m.create("j", [[5.0, 1.0], [3.0], [2.0, 0.5]], "mn", reduce="min")
    m.start("j")
    assert m.result("j") == 0.5


def test_auto_checkpoint_on_stop(tmp_path):
    import os
    m = _mgr(tmp_path)
    m.create("j", [[i] for i in range(4)], "sum1")
    m.start("j")                                                    # DONE -> auto-saved
    assert os.path.exists(m._path("j"))


def test_unregistered_worker_local_raises(tmp_path):
    m = JobManager(InProcessBackend(), store_dir=str(tmp_path))     # no register_worker
    m.create("j", [[1]], "ghost")
    m.start("j")
    assert m.jobs["j"].status == "failed" and "not registered" in m.jobs["j"].error


def test_reducer_name_validated():
    try:
        Job("x", [[1]], "w", reduce="median"); assert False
    except ValueError:
        pass


def test_distributed_pause_resume_restart():
    # the full distributed scenario over the real farm: pause mid-job, restore in a fresh manager, resume
    from holographic.misc.holographic_farm import WorkerDaemon, NetworkFarm
    import tempfile

    def slow_sum(bucket, cache):
        time.sleep(0.03)
        return float(np.sum([cache[i] for i in bucket])) if cache is not None else float(np.sum(bucket))

    store = tempfile.mkdtemp(prefix="jobs_farm_")
    node = WorkerDaemon(port=0); node.register_worker("slow_sum", slow_sum); addr = node.start()
    try:
        cache = np.arange(40, dtype=float) ** 2
        buckets = [list(range(i, i + 4)) for i in range(0, 40, 4)]
        m = JobManager(NetworkFarm([addr]), store_dir=store)
        m.create("r", buckets, "slow_sum", reduce="sum", cache=cache)
        m.start("r", background=True, batch=1)
        time.sleep(0.10)
        m.pause("r")
        assert m.jobs["r"].status == PAUSED and 0 < len(m.jobs["r"].done) < len(buckets)
        m2 = JobManager(NetworkFarm([addr]), store_dir=store)       # fresh session
        m2.load("r")
        m2.resume("r", background=False)
        assert m2.jobs["r"].status == DONE
        assert abs(m2.result("r") - float(np.sum(cache))) < 1e-6    # each tile computed exactly once
    finally:
        node.stop()

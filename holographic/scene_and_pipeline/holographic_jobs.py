"""holographic_jobs.py -- start / pause / resume / cancel long-running work (renders, sims, dataset processing), with
checkpoints that survive an app restart, across any coordinator backend (local pool / network farm).

THE IDEA
--------
A long job here is distributed MONOID work: a set of BUCKETS processed by a worker and combined by a commutative,
associative reducer (sum / min / max / bundle -- from holographic_distribute). That structure is exactly what makes
pause/resume clean:
  * Completed buckets fold into `partials`; because the reduce is a MONOID, the order does not matter, so you can stop
    after any bucket and combine what you have with whatever you finish later.
  * PAUSE = stop dispatching at the next bucket boundary and checkpoint the partials. RESUME = process only the
    REMAINING buckets and reduce everything. CANCEL = stop and mark cancelled.
  * Because a checkpoint is just (buckets, worker NAME, reducer NAME, cache, done-indices, partials), it serialises to
    one JSON file -- so you can pause a render, save, CLOSE the app, reopen it, load the checkpoint, and RESUME.

Workers are referenced by NAME (registered on the manager for local backends, or on the daemon for the farm) so a
restored job re-resolves its code -- the same "buckets are data, workers are registered code" rule as the farm.

COOPERATIVE CONTROL (kept honest): pause/cancel take effect at the next BUCKET BOUNDARY -- in-flight remote work is not
killed mid-bucket. Pick the bucket size (and the dispatch `batch`) for the responsiveness you want: small batch = quick
pause + less concurrency; large batch = more concurrency + a coarser pause. Default batch=1 (most responsive).
"""
import base64
import json
import os
import threading

import numpy as np

from holographic.scene_and_pipeline.holographic_distribute import reduce_sum, reduce_min, reduce_max, reduce_bundle

# -- job states --------------------------------------------------------------------------------------------
CREATED = "created"
RUNNING = "running"
PAUSED = "paused"
DONE = "done"
CANCELLED = "cancelled"
FAILED = "failed"

# reducers referenced by NAME so a job survives serialization (a function object cannot be saved to JSON)
_REDUCERS = {"sum": reduce_sum, "min": reduce_min, "max": reduce_max, "bundle": reduce_bundle}


def _pack(v):
    """Make a partial/cache value JSON-safe: arrays -> base64+shape+dtype, numpy scalars -> float, else as-is."""
    if isinstance(v, np.ndarray):
        return {"__ndarray__": base64.b64encode(np.ascontiguousarray(v).tobytes()).decode("ascii"),
                "shape": list(v.shape), "dtype": str(v.dtype)}
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    return v


def _unpack(v):
    if isinstance(v, dict) and "__ndarray__" in v:
        return np.frombuffer(base64.b64decode(v["__ndarray__"]), dtype=np.dtype(v["dtype"])).reshape(v["shape"])
    return v


class _Control:
    """A job's live control flags, checked between buckets. Cooperative: they take effect at the next boundary."""

    def __init__(self):
        self.paused = False
        self.cancelled = False


class Job:
    """A resumable, checkpointable unit of distributed monoid work."""

    def __init__(self, job_id, buckets, worker, reduce="sum", cache=None, meta=None):
        if reduce not in _REDUCERS:
            raise ValueError("reduce must be one of %s" % sorted(_REDUCERS))
        self.id = job_id
        self.buckets = list(buckets)
        self.worker = worker                            # a NAME (string) -- resolved to code by the manager/farm
        self.reduce = reduce                            # a reducer NAME (survives serialization)
        self.cache = cache                              # a shared read-only array (or None)
        self.meta = dict(meta or {})
        self.done = []                                  # completed bucket indices (order-independent -- monoid)
        self.partials = []                              # their results, aligned with self.done
        self.status = CREATED
        self.error = None

    def remaining(self):
        """The bucket indices not yet completed -- the work a resume still has to do."""
        d = set(self.done)
        return [i for i in range(len(self.buckets)) if i not in d]

    def progress(self):
        """Fraction complete, 0..1."""
        return len(self.done) / max(1, len(self.buckets))

    def result(self):
        """The reduced result. Valid once DONE; reduces all partials in any order (monoid)."""
        return _REDUCERS[self.reduce](self.partials) if self.partials else None

    # -- checkpoint <-> JSON-safe dict -----------------------------------------------------------------------
    def to_state(self):
        return {"id": self.id, "buckets": self.buckets, "worker": self.worker, "reduce": self.reduce,
                "cache": _pack(self.cache), "meta": self.meta, "done": list(self.done),
                "partials": [_pack(p) for p in self.partials], "status": self.status, "error": self.error}

    @classmethod
    def from_state(cls, s):
        j = cls(s["id"], s["buckets"], s["worker"], s.get("reduce", "sum"), _unpack(s.get("cache")), s.get("meta"))
        j.done = list(s.get("done", []))
        j.partials = [_unpack(p) for p in s.get("partials", [])]
        j.status = s.get("status", CREATED)
        j.error = s.get("error")
        return j


class JobManager:
    """Owns jobs, their control flags, a worker registry, and (optionally) a checkpoint directory. Runs a job on a
    coordinator backend with start/pause/resume/cancel, and can save/load a job so it survives an app restart."""

    def __init__(self, backend, store_dir=None):
        self.backend = backend
        self.store_dir = store_dir                      # a directory for checkpoints (enables restart-survival)
        self.workers = {}                               # name -> callable(bucket, cache), for local backends
        self.jobs = {}                                  # id -> Job
        self._controls = {}                             # id -> _Control
        self._threads = {}                              # id -> Thread (when started in the background)
        if store_dir:
            os.makedirs(store_dir, exist_ok=True)

    # -- setup ----------------------------------------------------------------------------------------------
    def register_worker(self, name, fn):
        """Register a worker's CODE by name (needed for local backends; the farm registers on its daemons instead)."""
        self.workers[name] = fn
        return name

    def create(self, job_id, buckets, worker, reduce="sum", cache=None, meta=None):
        """Define a job (does not start it). `worker` is a registered name; `reduce` is sum/min/max/bundle."""
        job = Job(job_id, buckets, worker, reduce, cache, meta)
        self.jobs[job_id] = job
        self._controls[job_id] = _Control()
        return job

    # -- lifecycle ------------------------------------------------------------------------------------------
    def start(self, job_id, background=False, batch=1):
        """Start (or resume) a job. background=True runs it in a daemon thread so the app stays responsive; otherwise
        it runs to its next stopping point and returns. `batch` = buckets dispatched per control check (concurrency
        vs pause-responsiveness)."""
        ctl = self._controls[job_id]
        ctl.paused = False
        ctl.cancelled = False
        if background:
            t = threading.Thread(target=self._run, args=(job_id, batch), daemon=True)
            self._threads[job_id] = t
            t.start()
            return self.jobs[job_id]
        return self._run(job_id, batch)

    def resume(self, job_id, background=False, batch=1):
        """Resume a paused (or restored) job -- it processes only the REMAINING buckets."""
        return self.start(job_id, background=background, batch=batch)

    def pause(self, job_id):
        """Ask a job to pause at the next bucket boundary; wait for it to stop (if backgrounded) and checkpoint."""
        self._controls[job_id].paused = True
        self._join(job_id)
        return self.jobs[job_id]

    def cancel(self, job_id):
        """Ask a job to cancel at the next bucket boundary; wait for it to stop and checkpoint its (partial) state."""
        self._controls[job_id].cancelled = True
        self._join(job_id)
        return self.jobs[job_id]

    def status(self, job_id):
        j = self.jobs[job_id]
        return {"id": j.id, "status": j.status, "progress": round(j.progress(), 4),
                "done": len(j.done), "total": len(j.buckets), "error": j.error}

    def list(self):
        return [self.status(jid) for jid in self.jobs]

    def wait(self, job_id, timeout=60):
        """Block until a backgrounded job stops (done/paused/cancelled/failed), then return it."""
        self._join(job_id, timeout)
        return self.jobs[job_id]

    def result(self, job_id):
        return self.jobs[job_id].result()

    # -- persistence (survive an app restart) ---------------------------------------------------------------
    def save(self, job_id, path=None):
        """Checkpoint a job to JSON. With store_dir set this is called automatically at every stopping point."""
        path = path or self._path(job_id)
        with open(path, "w") as f:
            json.dump(self.jobs[job_id].to_state(), f)
        return path

    def load(self, job_id=None, path=None):
        """Restore a job from a checkpoint into this manager (a fresh app session does this on reopen)."""
        path = path or self._path(job_id)
        with open(path) as f:
            job = Job.from_state(json.load(f))
        self.jobs[job.id] = job
        self._controls[job.id] = _Control()
        return job

    def load_all(self):
        """Restore EVERY checkpointed job in store_dir -- what an app calls on startup so paused jobs reappear ready
        to resume. Returns the list of restored job ids. A running-at-crash job is left as-is (resume re-runs only its
        remaining buckets, so no work is lost or duplicated)."""
        if not self.store_dir or not os.path.isdir(self.store_dir):
            return []
        restored = []
        for fname in sorted(os.listdir(self.store_dir)):
            if fname.startswith("job_") and fname.endswith(".json"):
                try:
                    self.load(path=os.path.join(self.store_dir, fname))
                    restored.append(fname[4:-5])
                except Exception:
                    pass                                    # a corrupt checkpoint is skipped, not fatal
        return restored

    def _path(self, job_id):
        if not self.store_dir:
            raise ValueError("no store_dir set -- pass store_dir=... to persist jobs across restarts")
        return os.path.join(self.store_dir, "job_%s.json" % job_id)

    # -- internals ------------------------------------------------------------------------------------------
    def _resolve_worker(self, name):
        """A farm dispatches by worker NAME; a local backend needs the registered CALLABLE."""
        if getattr(self.backend, "by_name", False):
            return name
        if name not in self.workers:
            raise KeyError("worker %r is not registered -- call register_worker() before starting (local backend)"
                           % name)
        return self.workers[name]

    def _run(self, job_id, batch):
        """The core loop: dispatch remaining buckets in `batch`-sized rounds, checking the control flags between
        rounds. Stops at DONE, or early on pause/cancel; auto-checkpoints at every stop if a store_dir is set."""
        job = self.jobs[job_id]
        ctl = self._controls[job_id]
        job.status = RUNNING
        handle = None
        try:
            worker = self._resolve_worker(job.worker)                  # may raise (unregistered) -> FAILED, resumable
            handle = self.backend.publish_cache(job.cache)
            remaining = job.remaining()
            batch = max(1, batch or 1)
            pos = 0
            stopped = False
            while pos < len(remaining):
                if ctl.cancelled:
                    job.status = CANCELLED
                    stopped = True
                    break
                if ctl.paused:
                    job.status = PAUSED
                    stopped = True
                    break
                chunk = remaining[pos:pos + batch]                          # dispatch this round concurrently
                futures = [(i, self.backend.submit(worker, job.buckets[i], handle)) for i in chunk]
                for i, f in futures:                                        # collect the round's results
                    job.done.append(i)
                    job.partials.append(f.result())
                pos += batch
            if not stopped:
                job.status = DONE
        except Exception as e:                                             # a real failure -> FAILED (resumable)
            job.status = FAILED
            job.error = "%s: %s" % (type(e).__name__, e)
        finally:
            if handle is not None:
                self.backend.release_cache(handle)
        if self.store_dir:                                                 # auto-checkpoint at every stopping point
            self.save(job_id)
        return job

    def _join(self, job_id, timeout=60):
        t = self._threads.get(job_id)
        if t is not None and t.is_alive():
            t.join(timeout)


# ---- module-level workers for the self-test (top-level so they are registrable + picklable) ----------------
def _sum_bucket(bucket, cache):
    return float(np.sum(bucket))


def _slow_sum(bucket, cache):
    import time
    time.sleep(0.02)                                    # simulate a heavy tile/step so a background pause can land
    return float(np.sum(bucket))


def _selftest():
    import tempfile
    import time

    store = tempfile.mkdtemp(prefix="lecore_jobs_")
    from holographic.scene_and_pipeline.holographic_coordinator import InProcessBackend

    # (1) a full run reduces correctly (10 buckets of one index each -> sum 0..9 = 45)
    mgr = JobManager(InProcessBackend(), store_dir=store)
    mgr.register_worker("sum1", _sum_bucket)
    mgr.create("j1", [[i] for i in range(10)], "sum1", reduce="sum")
    mgr.start("j1")
    assert mgr.jobs["j1"].status == DONE and mgr.result("j1") == 45.0

    # (2) the RESTART cycle, deterministically: a half-done PAUSED job, saved, reopened in a FRESH manager, resumed
    mgr.create("render", [[i] for i in range(10)], "sum1", reduce="sum")
    job = mgr.jobs["render"]
    job.done = [0, 1, 2, 3, 4]                           # pretend the first five tiles rendered, then we paused
    job.partials = [0.0, 1.0, 2.0, 3.0, 4.0]
    job.status = PAUSED
    mgr.save("render")                                  # checkpoint to disk (as pause would)
    # ... app closes; a brand-new manager (new session) reopens the checkpoint and resumes ...
    mgr2 = JobManager(InProcessBackend(), store_dir=store)
    mgr2.register_worker("sum1", _sum_bucket)           # the same code, registered by the same name
    mgr2.load("render")
    assert mgr2.jobs["render"].progress() == 0.5        # half done, as saved
    mgr2.resume("render")
    assert mgr2.jobs["render"].status == DONE and mgr2.result("render") == 45.0   # only tiles 5..9 ran; total correct

    # (3) background start + pause + resume (a real "start a render, pause it, resume it")
    mgr.register_worker("slow", _slow_sum)
    mgr.create("bg", [[i] for i in range(20)], "slow", reduce="sum")
    mgr.start("bg", background=True, batch=1)
    time.sleep(0.05)                                    # let a few tiles finish
    mgr.pause("bg")
    paused_done = len(mgr.jobs["bg"].done)
    assert mgr.jobs["bg"].status == PAUSED and 0 < paused_done < 20   # stopped partway
    mgr.resume("bg", background=False)                  # finish it synchronously
    assert mgr.jobs["bg"].status == DONE and mgr.result("bg") == float(sum(range(20)))

    # (4) cancel stops a job
    mgr.create("c", [[i] for i in range(20)], "slow", reduce="sum")
    mgr.start("c", background=True, batch=1)
    time.sleep(0.03)
    mgr.cancel("c")
    assert mgr.jobs["c"].status == CANCELLED and len(mgr.jobs["c"].done) < 20

    print("OK: holographic_jobs self-test passed (full run reduces correctly; PAUSE->save->NEW manager->load->RESUME "
          "completes across a restart; background start/pause/resume; cancel -- lifecycle over a coordinator backend)")


if __name__ == "__main__":
    _selftest()

"""holographic_renderjobs.py -- turn the SLOW part of making a cloud (baking its fractal-noise density grid,
~30s-4min depending on resolution -- see cloud_field) into a real, resumable background JOB: start it, check its
progress, pause it, walk away, come back (even after restarting the process) and resume it, using the engine's
existing job infrastructure (holographic_jobs.JobManager) rather than a one-off thread.

WHY THIS EXISTS
---------------
cloud_field's fBm bake loops a per-point noise query over grid^3 points -- that IS the cost. But those points are
INDEPENDENT (each is just "evaluate the noise here"), which makes the bake naturally BUCKETABLE: split the grid
into Z-SLICE BANDS, have each bucket compute a full-size zero array with only its own slice filled in, and let the
job's `reduce="sum"` assemble the complete grid -- since buckets never overlap, summing is exactly "each bucket
writes its slice, `sum` because zero everywhere else stays zero." That is the same "scatter-via-zero-padded-sum"
trick used elsewhere in the engine's distribute/reduce machinery.

The worker's `cache` is a plain JSON-safe dict (bounds/octaves/seed/... -- everything needed to RECONSTRUCT the
FractalNoise), never the live object -- so a paused job's checkpoint is a small JSON file that a brand-new process
can load and resume, exactly like the render/render lifecycle in holographic_jobs's own selftest.

USAGE (see UnifiedMind.bake_cloud_job / job_status / job_pause / job_resume / job_cancel / job_result for the
one-call versions -- this module is the plumbing underneath them):

    from holographic.scene_and_pipeline.holographic_renderjobs import make_noise_bake_job
    mgr, job_id = make_noise_bake_job(bounds=[(-2,2),(-1.5,1.5),(-2,2)], grid=32, octaves=4, gain=0.58, seed=0,
                                      store_dir=".lecore_jobs")
    mgr.start(job_id, background=True)     # or background=False to block
    mgr.status(job_id)                     # {"status": "running", "progress": 0.4, ...}
    mgr.pause(job_id)                      # stop at the next slice-band boundary, checkpoint to store_dir
    # ... later, even a different process: ...
    mgr2 = JobManager(InProcessBackend(), store_dir=".lecore_jobs")
    mgr2.register_worker("noise_bake_slice", _noise_bake_slice_worker)
    mgr2.load(job_id)
    mgr2.resume(job_id)
    grid_array = mgr2.result(job_id)       # the finished (grid,grid,grid) density-noise array
"""
import numpy as np

from holographic.scene_and_pipeline.holographic_jobs import JobManager
from holographic.scene_and_pipeline.holographic_coordinator import InProcessBackend

WORKER_NAME = "noise_bake_slice"


def _noise_bake_slice_worker(bucket, cache):
    """A single bucket: bake z-slices [z0, z1) of a grid^3 fBm lattice, in a full-size (mostly-zero) array so the
    job's sum-reduce assembles the whole grid across non-overlapping buckets. `cache` is the plain-JSON params
    dict (never the live FractalNoise), so this worker is fully restart-safe."""
    from holographic.sampling_and_signal.holographic_noise import FractalNoise
    z0, z1 = bucket
    p = cache
    fbm = FractalNoise(3, dim=p["dim"], bounds=[tuple(b) for b in p["bounds"]], octaves=p["octaves"],
                       lacunarity=p["lacunarity"], gain=p["gain"], base_bandwidth=p["base_bandwidth"],
                       seed=p["seed"])
    grid = p["grid"]
    axes = [np.linspace(lo, hi, grid) for (lo, hi) in p["bounds"]]
    X, Y = np.meshgrid(axes[0], axes[1], indexing="ij")
    flat_xy = np.stack([X.ravel(), Y.ravel()], axis=1)
    out = np.zeros((grid, grid, grid))
    for z in range(z0, z1):
        pts = np.concatenate([flat_xy, np.full((flat_xy.shape[0], 1), axes[2][z])], axis=1)
        vals = np.array([fbm.query(pt) for pt in pts])
        out[:, :, z] = vals.reshape(grid, grid)
    return out


def make_noise_bake_job(bounds, grid=32, octaves=4, lacunarity=2.1, gain=0.58, base_bandwidth=2.0, seed=0,
                        dim=1024, n_buckets=8, job_id=None, store_dir=None, manager=None):
    """Set up (but do not start) a resumable job that bakes an fBm grid^3 noise lattice, bucketed into `n_buckets`
    z-slice bands. Returns (manager, job_id); call manager.start(job_id, background=True) to run it. `bounds` is
    the same [(lo,hi), (lo,hi), (lo,hi)] format FractalNoise takes. `store_dir`, if given, makes the job's
    checkpoints (and therefore pause/resume) survive a process restart -- see the module docstring for the
    reopen-in-a-new-process pattern. Pass an existing `manager` (already carrying the "noise_bake_slice" worker
    registration) to add this job to it instead of creating a fresh one -- e.g. UnifiedMind.bake_cloud_job does
    this so all its jobs live on one shared manager."""
    import time
    if manager is None:
        manager = JobManager(InProcessBackend(), store_dir=store_dir)
        manager.register_worker(WORKER_NAME, _noise_bake_slice_worker)
    mgr = manager
    grid = int(grid)
    n_buckets = max(1, min(int(n_buckets), grid))
    edges = np.linspace(0, grid, n_buckets + 1).astype(int)
    buckets = [[int(edges[i]), int(edges[i + 1])] for i in range(n_buckets) if edges[i] < edges[i + 1]]
    cache = {"bounds": [list(b) for b in bounds], "grid": grid, "octaves": octaves, "lacunarity": lacunarity,
            "gain": gain, "base_bandwidth": base_bandwidth, "seed": seed, "dim": dim}
    if job_id is None:
        job_id = "noise_bake_%d" % int(time.time() * 1000)
    mgr.create(job_id, buckets, WORKER_NAME, reduce="sum", cache=cache,
              meta={"kind": "noise_bake", "grid": grid})
    return mgr, job_id


def _selftest():
    import tempfile
    store = tempfile.mkdtemp(prefix="lecore_renderjobs_")
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    kw = dict(lacunarity=2.0, gain=0.5, base_bandwidth=2.0)   # explicit (not defaulted) so both sides truly match

    # (1) a small bake completes and matches a direct (non-bucketed) bake, bucket-order independent
    mgr, jid = make_noise_bake_job(bounds, grid=8, octaves=2, n_buckets=3, seed=1, store_dir=store, **kw)
    mgr.start(jid)
    assert mgr.jobs[jid].status == "done"
    baked = mgr.result(jid)
    assert baked.shape == (8, 8, 8)

    from holographic.sampling_and_signal.holographic_noise import FractalNoise
    direct = FractalNoise(3, dim=1024, bounds=bounds, octaves=2, seed=1, **kw).sample_grid(8)
    assert np.allclose(baked, direct, atol=1e-8), "bucketed bake must exactly match a direct bake"

    # (2) pause -> checkpoint -> a BRAND NEW manager (simulating a process restart) -> resume -> same result
    mgr2, jid2 = make_noise_bake_job(bounds, grid=8, octaves=2, n_buckets=4, seed=2, store_dir=store, **kw)
    j = mgr2.jobs[jid2]
    j.done = [0]                                   # pretend the first slice-band finished, then we paused
    j.partials = [_noise_bake_slice_worker(j.buckets[0], j.cache)]
    j.status = "paused"
    mgr2.save(jid2)
    mgr3 = JobManager(InProcessBackend(), store_dir=store)
    mgr3.register_worker(WORKER_NAME, _noise_bake_slice_worker)
    mgr3.load(jid2)
    assert 0 < mgr3.jobs[jid2].progress() < 1.0
    mgr3.resume(jid2)
    assert mgr3.jobs[jid2].status == "done"
    resumed = mgr3.result(jid2)
    direct2 = FractalNoise(3, dim=1024, bounds=bounds, octaves=2, seed=2, **kw).sample_grid(8)
    assert np.allclose(resumed, direct2, atol=1e-8), "resumed-after-restart bake must match a direct bake"

    print("OK: holographic_renderjobs self-test passed (bucketed bake matches a direct bake; "
          "pause -> checkpoint -> new-process resume produces the identical grid)")


if __name__ == "__main__":
    _selftest()

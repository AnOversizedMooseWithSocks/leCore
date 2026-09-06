"""progressive -- a render as CHECKPOINTED MONOID WORK, so it can be paused, resumed and distributed.

THE PROBLEM. A path trace is one long call: it either finishes or it did not happen. In a limited
environment that caps the quality you can ever reach -- if a 4K render needs forty minutes and the
process gets ten, you never get the render, no matter how many times you try. The job machinery
already solved this shape (holographic_jobs: buckets, a monoid reducer, a JSON checkpoint that
survives an app restart), and `job_submit` says so in its own docstring: "ATOMIC. One bucket, so
progress is 0 then 1... do not expect a partial render." That is the gap this closes, and it closes it
by CHANGING THE SHAPE OF THE RENDER rather than by adding machinery.

WHY A RENDER IS ALREADY A MONOID, which is the whole reason this is thin. Monte Carlo samples are
independent and identically distributed, so an image is a MEAN of per-sample contributions, and a mean
of equal-sized batches is the mean of their sum. Order does not matter; batches can be computed
anywhere, in any order, at any time, and combined by addition. That is exactly the (buckets, sum,
checkpoint) structure the job manager already runs -- so pause, resume, restart-survival and the
network farm all come for free rather than being rebuilt here.

MEASURED, against a 256-spp reference, on a sphere-and-plane scene: accumulating 8 buckets of 16 spp
tracks a single-shot render of the same total samples at every step (mean abs error 0.0368 / 0.0263 /
0.0220 / 0.0197 / 0.0180 / 0.0167 / 0.0156 / 0.0150 progressively, against 0.0372 / 0.0272 / 0.0228 /
0.0203 / 0.0185 / 0.0173 / 0.0163 / 0.0157 single-shot). Progressive is very slightly AHEAD at every
count -- around 4%, which is not claimed as a win, only as evidence that splitting costs nothing.

THE CORRECTNESS POINT, and it is the one thing that can silently ruin this: EVERY BUCKET NEEDS ITS OWN
SEED. The tracer is deterministic in its seed (verified: same seed, byte-identical image), which is
what the whole engine's reproducibility rests on -- and it means N buckets at the same seed return N
copies of the SAME image, whose mean is that image. You would render for an hour and get the noise of
one bucket, with a progress bar saying it worked. `sample_buckets` exists to make that impossible to
get wrong by hand.

KEPT NEGATIVE -- EQUAL BUCKETS, ON PURPOSE. Every bucket carries the same spp so the readout is a plain
mean over completed buckets and the reducer stays the stock "sum". Unequal buckets would need a
weighted mean, which means carrying a weight through the checkpoint and reducing pairs instead of
arrays -- a different reducer, a wider checkpoint format, and a new way to be subtly wrong. The cost of
the restriction is that `spp` granularity is the bucket size; the benefit is that this module adds no
new reducer and no new state. Choose the bucket size for the pause responsiveness you want.
"""
import numpy as np


def sample_buckets(n_buckets, spp=16, seed0=0):
    """The bucket plan: `n_buckets` batches of `spp` samples, each with a DISTINCT seed.

    Returns JSON-safe dicts (the checkpoint is JSON, so buckets must be data, not objects). Seeds are
    seed0, seed0+1, ... -- consecutive rather than random, so a plan is reproducible and a resumed job
    re-derives exactly the buckets it had. See the module docstring for why the seed is load-bearing."""
    n_buckets = int(n_buckets)
    if n_buckets < 1:
        raise ValueError("n_buckets must be >= 1, got %r" % (n_buckets,))
    if int(spp) < 1:
        raise ValueError("spp must be >= 1, got %r" % (spp,))
    return [{"seed": int(seed0) + i, "spp": int(spp), "index": i} for i in range(n_buckets)]


def make_worker(trace_fn):
    """Wrap a tracer as a job worker: bucket -> one batch's image.

    `trace_fn(seed=..., spp=..., **cache)` must return an (H,W,3) array. The worker signature is the
    job manager's (bucket, cache), and the cache carries everything shared across buckets -- the scene,
    camera and size -- so it is published once per run rather than per bucket."""
    def worker(bucket, cache=None):
        kw = dict(cache or {})
        img = trace_fn(seed=int(bucket["seed"]), spp=int(bucket["spp"]), **kw)
        return np.asarray(img, float)
    return worker


def combine_buckets(partials, n_done=None, reject=0.0):
    """The readout: the MEAN of the completed buckets -- valid at any point, not just at the end.

    This is what makes the render watchable while it runs. `partials` is the job's list of completed
    bucket results; because every bucket is the same size, the estimator is a plain mean and the count
    is just how many finished.

    `reject=k` TURNS THE BUCKETS INTO A FIREFLY FILTER, which is a capability the split gives away for
    free and a plain single-shot render cannot have. A firefly is one rare bright path -- so it lands in
    ONE bucket, and the other buckets at that pixel disagree with it wildly. Having independent
    estimates per pixel means the disagreement is measurable: drop any bucket value more than k median
    absolute deviations from the per-pixel median, then average what is left. reject=0.0 (default) is
    the plain mean and is byte-identical to before.

    WHY MAD AND NOT A STANDARD DEVIATION: a firefly is exactly the kind of outlier that inflates a
    standard deviation enough to protect itself -- one 500x sample raises sigma so far that it falls
    inside 3 sigma. The median absolute deviation does not move, which is the whole point of using it.

    WHY THIS BEATS clamp_fireflies FOR A SPECTRAL RENDER: clamping works on ONE image against a global
    percentile, so it cannot tell a genuine specular highlight from a firefly of the same brightness --
    push it hard enough to remove the speckle and it removes the highlights too (measured in sweep 142:
    pct 97 cut speckle 2.6x and clipped p99 from 0.920 to 0.377). Rejection compares a pixel against
    ITSELF across buckets, so a real highlight -- which every bucket agrees on -- is never touched."""
    if not partials:
        return None
    n = int(n_done) if n_done is not None else len(partials)
    if n < 1:
        return None
    if not reject:
        acc = None
        for p in partials:
            a = np.asarray(p, float)
            acc = a.copy() if acc is None else acc + a
        return acc / float(n)

    stack = np.stack([np.asarray(p, float) for p in partials[:n]], axis=0)   # (n, ...)
    if stack.shape[0] < 3:
        return stack.mean(axis=0)          # fewer than 3 estimates cannot vote; do not pretend otherwise
    med = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - med), axis=0)
    # A pixel every bucket agrees on has mad == 0; the epsilon keeps it from rejecting everything there.
    keep = np.abs(stack - med) <= (float(reject) * mad + 1e-9)
    kept = keep.sum(axis=0)
    out = np.where(kept > 0, (stack * keep).sum(axis=0) / np.maximum(kept, 1), med)
    return out


def preview(job):
    """The image SO FAR from a (possibly unfinished) job -> (image, buckets_done, fraction).

    A partially complete render is a real render of fewer samples, not a broken one -- so this returns
    it rather than refusing until DONE. Noise falls as 1/sqrt(buckets done)."""
    done = len(getattr(job, "done", []) or [])
    total = max(1, len(getattr(job, "buckets", []) or []))
    return combine_buckets(getattr(job, "partials", []) or [], done), done, done / float(total)


def spp_so_far(job):
    """How many samples per pixel the current preview actually represents -- the honest quality label.

    Reported rather than inferred from the bucket count, because 'how far along' and 'how good' are
    different questions and only this one answers the second."""
    done = getattr(job, "done", []) or []
    buckets = getattr(job, "buckets", []) or []
    return int(sum(int(buckets[i]["spp"]) for i in done if i < len(buckets)))


def jsonify_args(args):
    """Make a render's arguments JSON-safe so the checkpoint SURVIVES A PROCESS RESTART.

    This is what turns "pause and resume in one session" into "render across as many sessions as it
    takes", which is the whole point in a limited environment. The job manager already degrades
    gracefully -- a cache it cannot serialise runs fine and just records persisted=False -- but a live
    SDF and Camera in the cache means EVERY render lands in that degraded mode, because those are what
    a render is made of.

    Two conversions do it, and both are exact rather than approximate:
      * an SDF -> its DSL string via to_dsl(), which parse_dsl() rebuilds. Verified exact on a
        twisted/rotated/translated torus, not merely on a sphere: the round-tripped SDF evaluates
        bit-identically over random points.
      * a Camera -> the numbers it was built from (eye, target, fov, aspect).
      * a sky given as {"model": {...}} -> rebuilt by mind.sky_model(**those params), because a sky is
        already fully parameterised and so is exactly the kind of callable that CAN be named.

    Anything already JSON-safe passes through untouched, and anything that is neither is LEFT ALONE --
    the job then runs and reports persisted=False, which is the manager's existing honest degradation
    rather than a new failure mode invented here.

    KEPT NEGATIVE -- A MATERIAL CALLBACK CANNOT BE CHECKPOINTED, and this is the real boundary of
    restart survival. A path-tracer material is an arbitrary Python function of surface position; there
    is no honest way to serialise one, and inventing a mini-language for it here would be a worse
    version of something the engine already has. A render passing `material=<function>` therefore runs
    fine, pauses and resumes fine WITHIN a process, and reports persisted=False -- it will not survive a
    restart. For a render that must survive one, describe the scene as a SCENE DOCUMENT (which is JSON
    by construction) and trace that, rather than handing a closure to a checkpoint."""
    out = {}
    for k, v in dict(args or {}).items():
        if hasattr(v, "to_dsl") and callable(getattr(v, "to_dsl")):
            try:
                out[k] = {"__sdf_dsl__": v.to_dsl()}
                continue
            except Exception:
                pass                                   # un-serialisable node: fall through, keep the object
        if isinstance(v, dict) and "model" in v and len(v) == 1:
            out[k] = {"__sky_model__": dict(v["model"])}
            continue
        if hasattr(v, "eye") and hasattr(v, "target") and hasattr(v, "fov_deg"):
            out[k] = {"__camera__": {"eye": [float(x) for x in np.ravel(v.eye)],
                                     "target": [float(x) for x in np.ravel(v.target)],
                                     "fov_deg": float(v.fov_deg),
                                     "aspect": float(getattr(v, "aspect", 1.0))}}
            continue
        out[k] = v
    return out


def rehydrate_args(args):
    """The inverse of jsonify_args, run inside the worker. Unrecognised values pass through."""
    out = {}
    for k, v in dict(args or {}).items():
        if isinstance(v, dict) and "__sdf_dsl__" in v:
            from holographic.mesh_and_geometry.holographic_sdf import parse_dsl
            out[k] = parse_dsl(v["__sdf_dsl__"])
        elif isinstance(v, dict) and "__sky_model__" in v:
            import lecore
            out[k] = lecore.UnifiedMind(dim=64, seed=0).sky_model(**v["__sky_model__"])
        elif isinstance(v, dict) and "__camera__" in v:
            from holographic.rendering.holographic_render import Camera
            c = v["__camera__"]
            out[k] = Camera(eye=tuple(c["eye"]), target=tuple(c["target"]),
                            fov_deg=c["fov_deg"], aspect=c.get("aspect", 1.0))
        else:
            out[k] = v
    return out


def _selftest():
    # A fake tracer whose "image" is a known function of the seed, so the monoid arithmetic can be
    # checked EXACTLY rather than statistically -- a statistical check would pass a broken mean.
    def fake(seed=0, spp=1, scale=1.0):
        return np.full((2, 2, 3), float(seed) * float(scale)) * float(spp)

    b = sample_buckets(4, spp=8, seed0=10)
    assert [x["seed"] for x in b] == [10, 11, 12, 13], b
    assert all(x["spp"] == 8 for x in b)

    # 1. SEEDS MUST DIFFER. The failure this guards against is silent: identical seeds average to one
    #    bucket's image while the progress bar says the render worked.
    assert len({x["seed"] for x in b}) == len(b), "buckets must not share a seed"

    w = make_worker(fake)
    parts = [w(x, {"scale": 1.0}) for x in b]
    got = combine_buckets(parts)
    assert np.allclose(got, np.full((2, 2, 3), 8.0 * (10 + 11 + 12 + 13) / 4.0)), got

    # 2. ORDER DOES NOT MATTER -- the monoid property the whole pause/resume story rests on. If this
    #    ever fails, a resumed job is not the same render as an uninterrupted one.
    assert np.allclose(combine_buckets(parts[::-1]), got)
    assert np.allclose(combine_buckets([parts[2], parts[0], parts[3], parts[1]]), got)

    # 3. A PARTIAL IS A REAL RENDER of fewer samples, not a broken one.
    class _J(object):
        buckets, done, partials = b, [0, 1], parts[:2]
    img, n, frac = preview(_J())
    assert n == 2 and abs(frac - 0.5) < 1e-12
    assert np.allclose(img, np.full((2, 2, 3), 8.0 * (10 + 11) / 2.0))
    assert spp_so_far(_J()) == 16, spp_so_far(_J())

    # 4. Nothing done yet is None, not a divide-by-zero or a black frame pretending to be a render.
    class _E(object):
        buckets, done, partials = b, [], []
    assert preview(_E())[0] is None and spp_so_far(_E()) == 0

    for bad in ((0, 8), (3, 0)):
        try:
            sample_buckets(bad[0], spp=bad[1])
            raise AssertionError("sample_buckets%s must raise" % (bad,))
        except ValueError:
            pass
    # 5. THE CHECKPOINT MUST BE JSON, or "resume after a restart" is a claim the format cannot keep.
    #    An SDF and a Camera are what a render is MADE of, so leaving them live means every render
    #    lands in the manager's persisted=False mode.
    import json
    from holographic.mesh_and_geometry.holographic_sdf import sphere, plane
    from holographic.rendering.holographic_render import Camera
    scene = sphere(0.8).union(plane(-0.9))
    cam = Camera(eye=(1.2, 0.8, 2.2), target=(0.0, 0.0, 0.0), fov_deg=42.0)
    j = jsonify_args({"sdf": scene, "camera": cam, "width": 32, "height": 24})
    json.dumps(j)                                       # raises if anything live survived
    r = rehydrate_args(json.loads(json.dumps(j)))
    pts = np.random.default_rng(0).normal(size=(64, 3))
    assert np.array_equal(np.asarray(scene.eval(pts), float), np.asarray(r["sdf"].eval(pts), float)), \
        "the scene did not survive the round trip EXACTLY"
    assert r["width"] == 32 and abs(r["camera"].fov_deg - 42.0) < 1e-12
    # Anything unserialisable is left alone rather than dropped -- degrade, do not lose. A MATERIAL
    # CALLBACK is the case that matters: it is an arbitrary function and there is no honest way to
    # serialise one, so it must pass through untouched and let the manager report persisted=False.
    sentinel = object()
    assert jsonify_args({"x": sentinel})["x"] is sentinel
    mat = lambda P: None
    assert jsonify_args({"material": mat})["material"] is mat, "a material must not be mangled"
    # A sky CAN be named, because it is fully parameterised.
    sky = jsonify_args({"sky": {"model": {"hour": 10.0, "sun_intensity": 40.0}}})["sky"]
    assert sky == {"__sky_model__": {"hour": 10.0, "sun_intensity": 40.0}}, sky
    assert callable(rehydrate_args({"sky": sky})["sky"])

    # 6. OUTLIER REJECTION: a firefly in ONE bucket must not survive, and a value every bucket agrees
    #    on must not be touched. Both halves matter -- a filter that also eats real highlights is the
    #    thing clamp_fireflies already does badly.
    clean = [np.full((2, 2, 3), 1.0) for _ in range(5)]
    assert np.allclose(combine_buckets(clean, reject=3.0), 1.0), "consensus must be preserved exactly"
    spiked = [np.full((2, 2, 3), 1.0) for _ in range(5)]
    spiked[2] = spiked[2].copy(); spiked[2][0, 0] = 500.0
    plain = combine_buckets(spiked)
    robust = combine_buckets(spiked, reject=3.0)
    assert abs(float(plain[0, 0, 0]) - 100.8) < 1e-9, float(plain[0, 0, 0])   # the mean carries it
    assert abs(float(robust[0, 0, 0]) - 1.0) < 1e-9, float(robust[0, 0, 0])   # rejection does not
    assert np.allclose(robust[1, 1], 1.0), "an untouched pixel changed"
    assert np.array_equal(combine_buckets(spiked, reject=0.0), plain), "reject=0 must be the plain mean"
    # Under three buckets there is nothing to vote on, and it says so by not pretending.
    assert np.allclose(combine_buckets(spiked[:2], reject=3.0), np.mean(spiked[:2], axis=0))

    print("progressive selftest OK -- distinct seeds, order-independent mean, partial previews exact, "
          "scene+camera round-trip through JSON bit-exact")


if __name__ == "__main__":
    _selftest()

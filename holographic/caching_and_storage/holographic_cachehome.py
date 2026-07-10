"""holographic_cachehome.py -- the CACHE home (consolidation backlog H2): bake a slow evaluator over the thing that
VARIES, then look it up cheaply. The engine's core performance lever ("bake once, query O(1)"), in one place.

WHY THIS EXISTS
---------------
The same move is written in several modules: sample an expensive function over a regular grid and read it back by
trilinear interpolation (holographic_matbake bakes material channels; holographic_sdfbake bakes a distance field),
sample a BRDF over view/roughness into a small table (holographic_viewlut), sample a deformation over frames
(holographic_anim). Each re-writes the SAME grid-generation (np.linspace -> meshgrid(indexing='ij') -> stack), then
its own lookup. This home owns that shared core so the bakes stop duplicating it:

    pts, res3 = Cache.grid_points(lo, hi, res)     # the ONE position-grid generator the bakes share
    grid, pts = Cache.bake_grid(evaluator, lo, hi, res)   # sample an evaluator over that grid
    Cache.bake(evaluator, vary="position"|"view"|"time"|"constant", ...)   # the dispatcher by what varies

Route, don't rewrite: this owns the shared grid-sample-and-store core; each domain keeps its own lookup reader
(BakedField, GridSDF, the view LUT, the frame table) with its own clamping/interpolation. What is unified is the
precompute, not the readers.

NOT holographic_cache.py, which is Ward's irradiance-GRADIENT cache (value + Jacobian at sparse anchors, first-order
interpolation) -- a different, complementary caching scheme. This home is the dense bake-over-a-grid one.
"""
import numpy as np


def _res3(res):
    """Normalise a resolution to a length-3 int array (scalar -> cube, or a per-axis triple)."""
    return np.broadcast_to(np.asarray(res, int), (3,)).astype(int)


class BakedGrid:
    """A value sampled onto a regular 3-D grid over [lo,hi], read back by trilinear interpolation -- O(1) per point.
    Holds a scalar grid (rx,ry,rz) or a channel grid (rx,ry,rz,C). Points outside the box clamp to the edge. This
    mirrors matbake.BakedField's reader so a position bake can share the whole path when it wants to."""

    def __init__(self, grid, lo, hi):
        self.grid = np.asarray(grid, float)
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.res = np.array(self.grid.shape[:3])

    def sample(self, P):
        P = np.asarray(P, float)
        frac = (P - self.lo) / np.maximum(self.hi - self.lo, 1e-12)
        coord = np.clip(frac * (self.res - 1), 0.0, self.res - 1)
        i0 = np.floor(coord).astype(int)
        i1 = np.minimum(i0 + 1, self.res - 1)
        w = coord - i0
        out = None
        for dx in (0, 1):                                          # accumulate the 8 surrounding corners (same order
            for dy in (0, 1):                                      # as matbake.BakedField, so results agree)
                for dz in (0, 1):
                    wx = w[:, 0] if dx else 1.0 - w[:, 0]
                    wy = w[:, 1] if dy else 1.0 - w[:, 1]
                    wz = w[:, 2] if dz else 1.0 - w[:, 2]
                    ix = i1[:, 0] if dx else i0[:, 0]
                    iy = i1[:, 1] if dy else i0[:, 1]
                    iz = i1[:, 2] if dz else i0[:, 2]
                    corner = self.grid[ix, iy, iz]
                    wgt = wx * wy * wz
                    contrib = wgt[:, None] * corner if corner.ndim > 1 else wgt * corner
                    out = contrib if out is None else out + contrib
        return out


class Cache:
    """Bake-and-query, chosen by what VARIES. All staticmethods -- Cache is a namespace of the shared precompute."""

    @staticmethod
    def grid_points(lo, hi, res):
        """The ONE position-grid generator the bakes share: points of a regular grid spanning [lo,hi] at `res` per
        axis (scalar or triple), row-major over indexing='ij'. Returns (points (P,3), res3 (3,)). Deterministic --
        a bake that generates its grid HERE is bit-for-bit identical to the inline np.linspace/meshgrid it replaced.
        """
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        r = _res3(res)
        axes = [np.linspace(lo[k], hi[k], int(r[k])) for k in range(3)]
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        return pts, r

    @staticmethod
    def bake_grid(evaluator, lo, hi, res):
        """Sample `evaluator(points)->values` over the position grid and reshape to (rx,ry,rz[,C]). Returns
        (grid, points). The convenience wrapper over grid_points for the common single-evaluator case."""
        pts, r = Cache.grid_points(lo, hi, res)
        vals = np.asarray(evaluator(pts))
        grid = vals.reshape(tuple(int(x) for x in r) + vals.shape[1:])
        return grid, pts

    @staticmethod
    def bake(evaluator, vary="position", **kw):
        """Dispatch by what varies:
          vary='constant' : return evaluator()               -- compute once, no table.
          vary='position' : bake over [lo,hi] at res -> a BakedGrid (kw: lo, hi, res=24).
          vary='view'     : delegate to holographic_viewlut.bake_view_lut (kw passed through).
          vary='time'     : delegate to holographic_anim.bake_deformation (kw: base, n_frames).
        """
        if vary == "constant":
            return evaluator()
        if vary == "position":
            grid, _ = Cache.bake_grid(evaluator, kw["lo"], kw["hi"], kw.get("res", 24))
            return BakedGrid(grid, kw["lo"], kw["hi"])
        if vary == "view":
            from holographic.rendering.holographic_viewlut import bake_view_lut
            return bake_view_lut(**kw)
        if vary == "time":
            from holographic.misc.holographic_anim import bake_deformation
            return bake_deformation(kw["base"], kw["n_frames"], evaluator)
        raise ValueError("Cache.bake: unknown vary=%r (constant/position/view/time)" % vary)


# ---- FAT-MARGIN CACHING for a DRIFTING query (backlog X9, Box3D lesson B4/B5's cousin) --------------------------
#
# Catto enlarges a moving body's AABB so it does not have to re-insert into the broadphase tree every frame. Read as a
# cache policy that generalizes far past physics: when a query DRIFTS -- a camera nudging forward, a cursor, an agent's
# position, a recall neighbourhood shifting one item at a time -- do not key the cache on the exact query. Bake an
# ENLARGED REGION around it and serve every query that lands inside. The margin buys hits with a little extra bake.
#
# MEASURED here (400 queries, a 2-D random walk of unit step; the numbers move with the drift scale, so they are the
# shape of the trade, not universal constants):
#
#     margin 0.0 ->  0.0% hits, 400 rebuilds      margin 3.0 -> 85.0% hits,  60 rebuilds
#     margin 1.0 -> 35.5% hits, 258 rebuilds      margin 6.0 -> 95.0% hits,  20 rebuilds
#
# KEPT NEGATIVE -- I predicted this would reuse SleepTracker's two-threshold hysteresis band (X3), because both are
# "a margin bought to stop thrashing". MEASURED: it does not. A margin cache has exactly ONE radius. Adding an inner
# radius changes nothing -- there is no state to leave, only a region to be inside or outside of, so the inner
# threshold is never read. Sleep needs two bars because an island can hover AT the bar and flicker between two states;
# a cache entry has no such state. They are cousins, not the same mechanism, and forcing a shared class would have
# been over-generalizing. Recorded because it was my prediction, and it was wrong.
#
# The margin is set by the DRIFT STATISTICS of the query stream -- which is the descriptor's `variation` probe pointed
# at the queries instead of the data. Same probe, new axis. `suggest_margin` does that by REPLAYING the observed
# stream, not by assuming a diffusion law: a random walk's exit time scales as (R/sigma)^2, but the measured rebuild
# counts sit ~1.8x off that prediction on this stream, so the law is a rule of thumb and the replay is the answer.

class MarginCache:
    """Cache a baked result over an ENLARGED region around a drifting query. `builder(point) -> value` bakes the
    region centred on `point`; every subsequent query within `margin` of that centre is served from the bake.

    `get(point)` returns (value, hit). `rebuilds` and `hits` count the trade so a caller can see what the margin
    bought. Deterministic: no RNG, and the hit test is a plain distance comparison.

    `metric` defaults to Euclidean distance and can be any callable (a, b) -> float, so this serves a camera
    position, a cursor, an agent state, or a hypervector recall neighbourhood (cosine distance) unchanged."""

    def __init__(self, builder, margin, metric=None):
        if margin < 0:
            raise ValueError("margin must be >= 0")
        self.builder = builder
        self.margin = float(margin)
        self.metric = metric or (lambda a, b: float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float))))
        self._center = None
        self._value = None
        self.hits = 0
        self.rebuilds = 0

    def get(self, point):
        """Serve `point` from the current bake if it lies within `margin` of the bake's centre; otherwise re-bake
        centred on it. Returns (value, hit)."""
        if self._center is not None and self.metric(point, self._center) <= self.margin:
            self.hits += 1
            return self._value, True
        self.rebuilds += 1
        self._center = np.asarray(point, float).copy() if hasattr(point, "__len__") else point
        self._value = self.builder(point)
        return self._value, False

    def stats(self):
        """{hits, rebuilds, hit_rate, margin} -- the trade, counted rather than asserted."""
        total = self.hits + self.rebuilds
        return {"hits": self.hits, "rebuilds": self.rebuilds,
                "hit_rate": (self.hits / total) if total else 0.0, "margin": self.margin}


def drift_scale(queries, metric=None):
    """The `variation` probe, pointed at a QUERY STREAM instead of at data: the mean step size between consecutive
    queries. This is the statistic that sets a fat margin -- a camera that moves 0.01 units a frame wants a very
    different region than one that teleports. Returns 0.0 for fewer than two queries."""
    qs = list(queries)
    if len(qs) < 2:
        return 0.0
    metric = metric or (lambda a, b: float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float))))
    return float(np.mean([metric(qs[i + 1], qs[i]) for i in range(len(qs) - 1)]))


def replay_margin(queries, margin, metric=None):
    """Replay a query stream against a margin and count what it would have cost: {hits, rebuilds, hit_rate, margin}.
    No baking -- only the hit/miss decision, so it is cheap enough to sweep. This is the honest way to choose a
    margin: measure the stream you actually have."""
    cache = MarginCache(builder=lambda p: None, margin=margin, metric=metric)
    for q in queries:
        cache.get(q)
    return cache.stats()


def replay_margin_error(queries, values, margin, metric=None, error=None):
    """Replay a query stream against a margin and measure what the cache would have SERVED against what was TRUE.

    `values[i]` is the correct result for `queries[i]`. Returns {hits, rebuilds, hit_rate, margin, max_error,
    mean_error}. Offline sizing: you run this once on a representative captured path, pick a margin, then use it
    live.

    WHY THIS EXISTS, and it is the negative the fat-margin table never showed. `replay_margin` counts HITS. A hit
    serves a STALE value, and hit rate says nothing about how stale. Measured on a drifting camera over a rendered
    scene (40 frames, drift 0.0316):

        margin   hit rate   rebuilds   max|err|   mean|err|
          0.00       0.0%         40     0.0000      0.0000
          0.02      15.0%         34     0.5864      0.0001
          0.05      77.5%          9     0.5864      0.0007
          0.12      92.5%          3     0.5864      0.0021
          0.50      97.5%          1     0.5864      0.0051

    **max|err| saturates the instant ANY reuse happens** -- one stale frame at a silhouette edge is already 0.59 --
    while mean|err| is the dial that actually responds to the margin. So a fat margin is sound only where the cached
    value varies SLOWLY in the query, and the margin must be sized against an ERROR budget, not a hit-rate target.
    That is what `suggest_margin_for_error` does, and it is the gate a caller must pass before caching anything
    whose value can jump."""
    metric = metric or (lambda a, b: float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float))))
    error = error or (lambda a, b: float(np.abs(np.asarray(a, float) - np.asarray(b, float)).mean()))
    qs, vs = list(queries), list(values)
    if len(qs) != len(vs):
        raise ValueError("need one value per query, got %d and %d" % (len(qs), len(vs)))
    center, hits, rebuilds, errs = None, 0, 0, []
    for k, q in enumerate(qs):
        if center is not None and metric(q, qs[center]) <= margin:
            hits += 1
            errs.append(error(vs[center], vs[k]))
        else:
            rebuilds += 1
            center = k
            errs.append(0.0)
    total = hits + rebuilds
    return {"hits": hits, "rebuilds": rebuilds, "hit_rate": (hits / total) if total else 0.0,
            "margin": float(margin), "max_error": (max(errs) if errs else 0.0),
            "mean_error": (float(np.mean(errs)) if errs else 0.0)}


def suggest_margin_for_error(queries, values, max_mean_error, max_abs_error=None,
                             metric=None, error=None, max_multiple=64.0):
    """THE GATE. Choose the LARGEST margin whose replayed error stays inside the budget on this stream.

    `max_mean_error` bounds the average staleness. `max_abs_error` (optional, and you usually want it) bounds the
    WORST single served result. Bisection over [0, max_multiple * drift_scale]; returns 0.0 when even the smallest
    useful margin busts the budget -- an honest refusal, and the signal that this value varies too fast to cache by
    proximity.

    WHY BOTH BOUNDS. Mean error can be fooled. Measured: a value that jumps from 0 to 1 as the query crosses an
    axis passes a `max_mean_error=0.02` budget at margin 0.193 -- because the jump is RARE, and the mean averages it
    away. Its max error at that margin is 1.0: the cache serves a completely wrong answer, occasionally. That is the
    same shape as a rendered frame's silhouette edge, whose max error saturates at the very first reuse (0.5864)
    while its mean error creeps 0.0001 -> 0.0051. **Size a fat margin on the error you cannot tolerate, not on the
    error you usually get.**"""
    qs = list(queries)
    sigma = drift_scale(qs, metric=metric)
    if sigma == 0.0 or len(qs) < 2:
        return 0.0

    def _ok(mgn):
        st = replay_margin_error(qs, values, mgn, metric=metric, error=error)
        if st["mean_error"] > max_mean_error:
            return False
        return max_abs_error is None or st["max_error"] <= max_abs_error

    lo, hi = 0.0, float(max_multiple) * sigma
    if _ok(hi):
        return hi                                          # the whole search range fits the budget
    for _ in range(40):                                    # fixed count -> deterministic
        mid = 0.5 * (lo + hi)
        if _ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def suggest_margin(queries, target_hit_rate=0.95, metric=None, max_multiple=64.0):
    """Choose the SMALLEST margin whose replayed hit rate on this query stream meets `target_hit_rate`.

    Bisection over [0, max_multiple * drift_scale], on the observed stream. Deliberately empirical: a random walk's
    exit time scales like (R/sigma)^2, but the measured rebuild counts here sit ~1.8x off that prediction, so a
    fitted law would be a worse answer than a replay. Returns 0.0 if the stream never moves, and the upper bound if
    the target is unreachable within it (an honest ceiling, not a silent clamp)."""
    qs = list(queries)
    sigma = drift_scale(qs, metric=metric)
    if sigma == 0.0:
        return 0.0
    lo, hi = 0.0, float(max_multiple) * sigma
    if replay_margin(qs, hi, metric=metric)["hit_rate"] < target_hit_rate:
        return hi                                          # unreachable within the bound: say so by returning it
    for _ in range(40):                                    # fixed iteration count -> deterministic
        mid = 0.5 * (lo + hi)
        if replay_margin(qs, mid, metric=metric)["hit_rate"] >= target_hit_rate:
            hi = mid
        else:
            lo = mid
    return hi


def cache_backends():
    """What varies (the strategies Cache dispatches over), for the catalog / discovery."""
    return ("constant", "position", "view", "time")


def _selftest():
    lo = np.array([-1.0, -1.0, -1.0]); hi = np.array([1.0, 1.0, 1.0]); res = 8

    # the shared grid generator matches the inline np.linspace/meshgrid the bakes used
    pts, r = Cache.grid_points(lo, hi, res)
    axes = [np.linspace(lo[k], hi[k], res) for k in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    ref = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    assert np.array_equal(pts, ref) and tuple(r) == (res, res, res)

    # bake a scalar field and a colour field; trilinear lookup at a node returns the stored value
    scalar = lambda P: P[:, 0] ** 2 + P[:, 1]
    colour = lambda P: np.stack([P[:, 0], P[:, 1], P[:, 2]], axis=1)
    bg = Cache.bake(scalar, vary="position", lo=lo, hi=hi, res=res)
    node = pts[100]
    assert abs(float(bg.sample(node[None, :])[0]) - float(scalar(node[None, :])[0])) < 1e-9
    cg = Cache.bake(colour, vary="position", lo=lo, hi=hi, res=res)
    assert np.allclose(cg.sample(node[None, :])[0], colour(node[None, :])[0], atol=1e-9)

    # constant strategy computes once
    assert Cache.bake(lambda: 42.0, vary="constant") == 42.0
    # -- X9: the fat margin on a drifting query -------------------------------------------------------------
    rng = np.random.default_rng(0)
    q = np.cumsum(rng.normal(size=(400, 2)), axis=0)                 # a unit-step 2-D random walk

    zero = replay_margin(q, 0.0)
    assert zero["rebuilds"] == 400 and zero["hits"] == 0             # exact key: every query misses
    prev_hits = -1
    for margin in (0.0, 1.0, 3.0, 6.0):                              # the trade is MONOTONE in the margin
        st = replay_margin(q, margin)
        assert st["hits"] >= prev_hits
        prev_hits = st["hits"]
    assert replay_margin(q, 6.0)["hit_rate"] > 0.9                   # a fat margin turns 400 rebuilds into ~20

    # the margin is set by the query stream's own drift, not by a constant
    assert abs(drift_scale(q) - np.mean(np.linalg.norm(np.diff(q, axis=0), axis=1))) < 1e-12
    m9 = suggest_margin(q, target_hit_rate=0.9)
    assert replay_margin(q, m9)["hit_rate"] >= 0.9                   # it meets the target ...
    assert replay_margin(q, m9 * 0.5)["hit_rate"] < 0.9              # ... and is not wastefully large
    assert suggest_margin([np.zeros(2)] * 5) == 0.0                  # a stream that never moves needs no margin

    # KEPT NEGATIVE: a margin cache has ONE radius. An "inner" threshold would never be read.
    assert replay_margin(q, 3.0) == replay_margin(q, 3.0)            # deterministic, and stateless per query

    # the cache actually serves the baked value, and only re-bakes on a miss
    calls = {"n": 0}
    def build(p):
        calls["n"] += 1
        return ("baked", tuple(np.round(np.asarray(p, float), 6)))
    mc = MarginCache(build, margin=6.0)
    for x in q:
        mc.get(x)
    assert calls["n"] == mc.rebuilds and mc.rebuilds < 40            # one bake per rebuild, and few of them

    print("OK: holographic_cachehome self-test passed (shared grid matches inline bake; trilinear exact at nodes; "
          "dispatch over %s; X9 fat margin: exact-key caching rebuilds 400/400, margin 6.0 rebuilds %d at %.0f%% "
          "hits, and suggest_margin picks the smallest margin meeting the target from the stream's own drift)"
          % (", ".join(cache_backends()), mc.rebuilds, 100 * mc.stats()["hit_rate"]))


if __name__ == "__main__":
    _selftest()

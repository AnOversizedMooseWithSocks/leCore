"""holographic_temporal.py -- the TEMPORAL-REUSE LOOP: reuse last frame's result, reproject it (backward-warp,
the unbind form), accumulate, and re-solve ONLY the dirty region. Amortizes a per-cell solve across frames.

WHY THIS EXISTS (Above/Below Sweep 3, item 5 -- the render/solve SPEED discipline)
----------------------------------------------------------------------------------
`holographic_backwardwarp` ships the reproject PRIMITIVE (backward_gather: read last frame at inverse-warped
positions, hole-free because it is the unbind direction). But the sweep's finding was blunt: "the LOOP is the
work, not the primitive." This module is that loop. It is one discipline that serves several stacks:
  * a path tracer -- ACCUMULATE samples across frames (running average), re-shooting only where the image changed;
  * a Walk-on-Spheres solve -- it is POINTWISE, so "re-solve only the dirty region" is native (each cell is
    independent, reuse the rest verbatim);
  * a fluid / wave field -- advance from the last state, re-solve where a source moved.
The move is §5.3's "compute, don't store" pointed at TIME: last frame is a free prior for this one, so only the
delta costs anything. `dirtyfield` is the per-tile sibling of exactly this.

HONEST SCOPE (kept loud): reuse is only correct where the clean cells GENUINELY did not change -- the caller owns
the dirty mask, and a wrong mask reuses stale values (the honest failure mode, so the caller must be right about
what changed). For a NOISY estimator (path tracing) accumulation converges the reused cells toward truth over
frames; for an EXACT pointwise solve (WoS/analytic) reuse is verbatim. Deterministic; NumPy + stdlib; the
reproject step delegates to holographic_backwardwarp.
"""
import numpy as np

from holographic.misc.holographic_backwardwarp import backward_gather


class TemporalReuse:
    """Holds last frame's per-cell result and reuses it: reproject -> keep clean cells -> re-solve dirty cells ->
    optionally accumulate. `frame` is the current (n,) field; `count` tracks per-cell samples for accumulation."""

    def __init__(self):
        self.frame = None                                    # last per-cell values (n,)
        self.count = None                                    # per-cell sample counts (accumulation)
        self.solves = 0                                      # total per-cell solve_fn calls (the cost we amortize)

    def solve(self, solve_fn, n, dirty=None, reproject=None, accumulate=False):
        """Produce this frame's (n,) field.
          * first call, or `dirty is None`: solve every cell (the full, un-amortized cost).
          * otherwise: start from last frame (reprojected if `reproject` is given), and re-solve ONLY the cells in
            `dirty`; the rest are reused verbatim (exact) or as the running prior (accumulate).
        `solve_fn(i)` returns cell i's value; `reproject(frame)->frame` warps last frame into this frame's
        coordinates (e.g., backward_gather); `accumulate=True` folds each re-solve into a running average (for
        noisy estimators). Returns (values, n_resolved) -- n_resolved is how many solve_fn calls this frame cost."""
        if self.frame is None or dirty is None:
            vals = np.array([solve_fn(i) for i in range(n)], float)   # full solve
            self.solves += n
            self.frame = vals.copy()
            self.count = np.ones(n)
            return vals.copy(), n

        # reuse: reproject last frame into this frame's coordinates, then re-solve only the dirty cells
        base = np.asarray(reproject(self.frame), float) if reproject is not None else self.frame.copy()
        cnt = self.count.copy() if self.count is not None else np.ones(n)
        dirty = list(dirty)
        for i in dirty:
            v = float(solve_fn(i))
            self.solves += 1
            if accumulate:                                   # running average: fold the new sample in
                cnt[i] += 1.0
                base[i] = base[i] + (v - base[i]) / cnt[i]
            else:                                            # exact pointwise: overwrite with the fresh solve
                base[i] = v
                cnt[i] = 1.0
        self.frame = base.copy()
        self.count = cnt
        return base.copy(), len(dirty)


def _selftest():
    """A pointwise solve reused across frames: re-solving only the dirty cells (a) costs far fewer solve calls
    than a full re-solve, and (b) is IDENTICAL to a from-scratch full solve when the clean cells truly didn't
    change; reproject (backward-warp) resamples a shifted field hole-free; accumulation converges a noisy
    estimator; deterministic."""
    n = 200
    # a field where cell i's value is an (expensive) function of a scene parameter; only a few cells change
    scene = np.sin(np.linspace(0, 6, n))

    def make_solver(sc):
        def solve_fn(i):
            return float(sc[i] * sc[i] + 0.5 * sc[i])        # stand-in for an expensive per-cell solve
        return solve_fn

    tr = TemporalReuse()
    frame0, cost0 = tr.solve(make_solver(scene), n)          # first frame: full solve
    assert cost0 == n

    # change a handful of cells; the caller knows which (the dirty mask)
    scene2 = scene.copy()
    dirty = [10, 11, 12, 99, 150]
    scene2[dirty] += 0.3
    frame1, cost1 = tr.solve(make_solver(scene2), n, dirty=dirty)
    # (a) SPEED: we re-solved only the dirty cells, not all n
    assert cost1 == len(dirty) and cost1 < n // 10
    # (b) CORRECTNESS: identical to a full from-scratch solve of the changed scene (clean cells truly unchanged)
    full = np.array([make_solver(scene2)(i) for i in range(n)])
    assert np.allclose(frame1, full, atol=1e-12)

    # reproject: a field shifted by a known warp is resampled hole-free by backward_gather
    positions = np.linspace(0, 1, n)
    shifted = TemporalReuse(); shifted.solve(make_solver(scene), n)
    def reproj(frame):                                       # inverse-warp: read the source shifted left by 0.02
        return backward_gather(frame, positions, np.clip(positions + 0.02, 0, 1))
    out, _ = shifted.solve(make_solver(scene), n, dirty=[0, 1], reproject=reproj)
    assert out.shape == (n,) and not np.isnan(out).any()     # hole-free reproject

    # accumulation: a NOISY estimator's reused cells converge toward truth over frames
    rng = np.random.default_rng(0)
    truth = 2.0
    acc = TemporalReuse()
    acc.solve(lambda i: truth + rng.standard_normal(), 1)    # noisy first sample
    for _ in range(200):
        acc.solve(lambda i: truth + rng.standard_normal(), 1, dirty=[0], accumulate=True)
    assert abs(acc.frame[0] - truth) < 0.2                   # the running average converged

    # deterministic
    a = TemporalReuse(); fa, _ = a.solve(make_solver(scene), n)
    b = TemporalReuse(); fb, _ = b.solve(make_solver(scene), n)
    assert np.array_equal(fa, fb)
    print("holographic_temporal selftest OK: reuse re-solves only the dirty region (%d vs %d cells) and matches a "
          "full re-solve exactly; reproject is hole-free; accumulation converges a noisy estimator; deterministic"
          % (len(dirty), n))


if __name__ == "__main__":
    _selftest()

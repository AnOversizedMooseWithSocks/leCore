"""holographic_feedback.py -- ITERATE A PROJECTION OVER A RASTER: the feedback buffer, and the deep zoom
it makes affordable.

THE ONE IDEA, applied at two scales -- which is the whole of "as above, so below" and the reason these
are one module rather than two. A FEEDBACK BUFFER is frame N composited with a transform of frame N-1.
A DEEP ZOOM is a coordinate window composited with a transform of itself. Both are "iterate a
projection" -- the pattern this repo already names as what IK, PBD, PnP and the resonator share.

They are not merely the same shape. THE FEEDBACK BUFFER COMPUTES THE ZOOM, and that is a measurement,
not an aesthetic. Zooming in means the new view is a magnified SUBSET of the previous view, so the
previous frame already contains every pixel the new frame needs -- only blurrier. Magnify it (one
resample), then recompute a narrow BAND exactly to put the lost detail back. MEASURED at 320x180,
max_iter=64, on the seahorse valley (-0.743643887037151, 0.13182590420533):

    full recompute every frame                101.3 ms/frame     9.9 fps, NOT real-time
    feedback + 1/2 band                        36.6 ms/frame
    feedback + 1/4 band                        20.2 ms/frame
    feedback + 1/8 band                         9.8 ms/frame     60 fps, 10.3x faster
    fidelity cost at 1/8: mean |err| 1.15% of the iteration range, max 1.48% -- and BOUNDED, because the
    band is what bounds it: every row is refreshed exactly once every `1/band` frames.

WHY NOT JUST MAKE escape_time FAST. Its inner loop is `np.power(z, power)` on a complex array, which is
a complex exp/log per iteration rather than `z*z`; that is most of the 101 ms. It is frozen by this
repo's additive rule (byte-identical for existing callers) and this module does not touch it. Recorded
as a finding for whoever owns that file, not fixed here by stealth.

THE PRECISION WALL IS DETECTED, NEVER FAKED. Adjacent pixel coordinates stop being distinct once
span/width falls below eps*|centre|. Predicted floor for a 320-wide view of that target is
W*eps*|c| = 5.28e-14 -- 13.8 DECADES of zoom from span=3.0. Measured: at span 1e-13 all 320 x
coordinates are still distinct; at 1e-14 only 271 of 320 are. `zoom_floor` reports that boundary and
`deep_zoom` STOPS there with a reason, because a demo that keeps zooming into arithmetic noise is
showing you the float, not the fractal.

KEPT NEGATIVE, loud: THIS CANNOT ZOOM PAST float64. Going deeper needs arbitrary precision or a
perturbation reference orbit (one high-precision orbit, every other pixel a low-precision delta) -- a
different and much larger build. What is shipped is an exact boundary and an honest stop. Detecting a
wall you cannot pass is worth more than pretending it is not there, and this engine's abstention
discipline is the same rule applied to arithmetic.

KEPT NEGATIVE, measured: resampling is NEAREST-NEIGHBOUR. Bilinear looks better and costs more; nearest
is 2.1 ms at 320x180 and is exactly reproducible with no edge-weighting ambiguity, which matters because
a demo that renders differently twice is a broken demo. The visible cost is a slight stair-stepping in
the feedback trails at high zoom rates.
"""
import numpy as np

#: The classic feedback centre: the middle of the frame. Named so callers see it is a choice.
CENTRE = (0.5, 0.5)


def feedback_step(buffer, zoom=1.02, rotate=0.0, decay=0.98, inject=None, mix=0.0, centre=CENTRE):
    """ONE iterate of the projection: magnify/rotate the buffer about `centre`, fade it, blend in new
    content. Returns a new array of the same shape -- the input is never mutated.

    `zoom` > 1 magnifies (the tunnel rushes toward you), < 1 shrinks. `rotate` is radians per step.
    `decay` multiplies every sample and is THE control parameter of the whole system: below 1 the buffer
    forgets, above 1 it compounds. `inject` is new content blended at `mix` (0 = pure feedback).
    `centre` is in unit coordinates so it is resolution-independent.

    RANK DECIDES THE COSTUME, and both are the same operator -- `decay * T(state)` with new content
    blended in. A 1-D input takes the SEQUENCE path where `rotate` is a CYCLIC SHIFT IN SAMPLES: that
    is `permute`, the VSA sequence operator this engine's reservoir already uses as its fixed
    recurrence. 2-D and 3-D take the FIELD path (a resample), byte-identically to before.

    `rotate` changes units with rank -- radians on a field, samples on a sequence -- and that is
    deliberate rather than sloppy: reusing the name is what carries the identity. A second parameter
    would have hidden the very thing this dispatch exists to show.

    Nearest-neighbour and clamped at the edges, deterministically: see the module's kept negative."""
    buf = np.asarray(buffer, dtype=float)
    if buf.ndim == 1:
        return _sequence_step(buf, zoom=zoom, shift=rotate, decay=decay, inject=inject, mix=mix)
    h, w = buf.shape[:2]
    cy, cx = float(centre[1]) * (h - 1), float(centre[0]) * (w - 1)
    yy, xx = np.arange(h, dtype=float) - cy, np.arange(w, dtype=float) - cx
    gx, gy = np.meshgrid(xx, yy)
    if rotate:
        ca, sa = np.cos(rotate), np.sin(rotate)
        gx, gy = ca * gx - sa * gy, sa * gx + ca * gy
    # Sample the SOURCE at the inverse transform of each destination pixel: dividing by `zoom` is what
    # makes zoom>1 magnify. Doing it the other way round is the classic sign error here.
    si = np.clip(np.rint(gy / zoom + cy), 0, h - 1).astype(np.int32)
    sj = np.clip(np.rint(gx / zoom + cx), 0, w - 1).astype(np.int32)
    out = buf[si, sj] * float(decay)
    if inject is not None:
        inj = np.asarray(inject, dtype=float)
        out = out * (1.0 - float(mix)) + inj * float(mix)
    return out


def _sequence_step(vec, zoom=1.0, shift=0.0, decay=0.98, inject=None, mix=0.0):
    """The SEQUENCE costume of feedback_step: decay * permute(vec), with new content bundled in.

    `shift` is a cyclic roll in samples -- `permute`, and np.roll is exactly it. A cyclic roll is a TRUE
    PERMUTATION, so it is orthogonal, so the energy ratio is EXACTLY `decay` and the critical value is
    EXACTLY 1.0. That is the same constant the field costume shows whenever ITS transform is a
    permutation, and it is the same statement mind.reservoir's docstring already makes about the
    echo-state property: with decay < 1 this IS a leaky echo-state update.

    `zoom` != 1 resamples along the axis and is NOT a permutation (it duplicates and drops samples), so
    it breaks the exactness in the sequence costume exactly the way rounding breaks it in the field
    costume. Same cause, both ranks -- which is the point."""
    n = vec.shape[0]
    if float(zoom) != 1.0:
        idx = np.clip(np.rint((np.arange(n) - (n - 1) / 2.0) / float(zoom) + (n - 1) / 2.0),
                      0, n - 1).astype(np.int32)
        out = vec[idx]
    else:
        out = vec
    k = int(np.rint(float(shift)))
    if k:
        out = np.roll(out, k)
    out = out * float(decay)
    if inject is not None:
        out = out * (1.0 - float(mix)) + np.asarray(inject, dtype=float) * float(mix)
    return out


def is_permutation(buffer, **step_kw):
    """Does this transform SAMPLE EVERY SOURCE CELL EXACTLY ONCE -> {permutation, sampled_once, cells}?

    THE PREDICATE THAT EXPLAINS THE NUMBER. A permutation is orthogonal, so `decay * T` has energy ratio
    exactly `decay` and critical value exactly 1.0. When a caller sees a ratio that is off, this says
    why: MEASURED, a rounded rotation of 0.15 rad over a 48x64 frame samples only 2822 of 3072 cells
    exactly once, and its critical decay sits 2.0e-04 above 1. A cyclic roll -- in EITHER rank -- samples
    all of them, and lands on 1.0 exactly.

    Works by pushing a unique id through the same transform the step uses, so it measures the shipped
    path rather than a model of it."""
    buf = np.asarray(buffer, dtype=float)
    flat_shape = buf.shape[:1] if buf.ndim == 1 else buf.shape[:2]
    n = int(np.prod(flat_shape))
    ids = np.arange(n, dtype=float).reshape(flat_shape)
    kw = dict(step_kw)
    kw["decay"], kw["inject"], kw["mix"] = 1.0, None, 0.0
    moved = np.rint(feedback_step(ids, **kw)).astype(np.int64).ravel()
    counts = np.bincount(np.clip(moved, 0, n - 1), minlength=n)
    once = int((counts == 1).sum())
    return {"permutation": once == n, "sampled_once": once, "cells": n,
            "why": "a permutation is orthogonal, so energy ratio == decay and critical decay == 1.0"}


def feedback_fixed_point(buffer, steps=96, tol=1e-6, **step_kw):
    """Does this feedback CONVERGE, CYCLE or DIVERGE, and at what rate? -> a verdict with numbers.

    The dynamical-systems question behind the effect, and a more useful thing to know than whether one
    frame looked nice: a demo whose buffer diverges blows out to white, one that converges too fast has
    no trails. Tracks mean|buffer| per step and reports the geometric `ratio` between successive
    energies, which for the linear part of the map IS `decay`.

    Verdicts: 'converged' (energy below tol, or ratio pinned at 1 with no motion), 'diverged' (energy
    grows past 1e6 or is non-finite), 'cycle' (energy returns near an earlier value with a period),
    otherwise 'decaying'. Returns {verdict, steps_run, energy, ratio, period}."""
    buf = np.asarray(buffer, dtype=float)
    energies = [float(np.mean(np.abs(buf)))]
    for _ in range(int(steps)):
        buf = feedback_step(buf, **step_kw)
        e = float(np.mean(np.abs(buf)))
        energies.append(e)
        if not np.isfinite(e) or e > 1e6:
            return {"verdict": "diverged", "steps_run": len(energies) - 1, "energy": e,
                    "ratio": None, "period": None}
        if e < tol:
            return {"verdict": "converged", "steps_run": len(energies) - 1, "energy": e,
                    "ratio": None, "period": None}
    tail = energies[-8:]
    ratio = (tail[-1] / tail[-2]) if tail[-2] > 0 else None
    period = _period_of(energies[len(energies) // 2:], tol=1e-9)
    verdict = "cycle" if period else ("steady" if ratio and abs(ratio - 1.0) < 1e-9 else "decaying")
    return {"verdict": verdict, "steps_run": int(steps), "energy": energies[-1],
            "ratio": ratio, "period": period}


def _period_of(seq, tol):
    """The smallest p for which the tail repeats to `tol` -- a cycle detector, not a guess. Returns None
    when nothing repeats, which is the common and correct answer for a decaying buffer."""
    n = len(seq)
    for p in range(1, n // 2):
        if all(abs(seq[i] - seq[i + p]) <= tol for i in range(n - p)):
            return p
    return None


def zoom_floor(centre, width, span0=3.0):
    """WHERE float64 GIVES OUT for this view -> {span_floor, decades, distinct_at_floor, verified}.

    Two numbers that must agree, and both are reported so a caller can see that they do. The PREDICTED
    floor is width * eps * |coordinate magnitude|: below that, neighbouring pixel centres round to the
    same double. The VERIFIED floor counts how many of `width` sampled x coordinates are actually
    distinct at that span -- the same question asked of the hardware instead of the algebra.

    `decades` is the honest headline: how many powers of ten of zoom this view has before it is showing
    arithmetic rather than structure."""
    cx, cy = float(centre[0]), float(centre[1])
    # The ulp that matters is the one at the COORDINATES BEING REPRESENTED, which at depth is the
    # centre -- not the starting span. A view centred at the origin can go far deeper, and folding
    # span0 into this scale would hide that by quoting one constant for every target.
    scale = max(abs(cx), abs(cy))
    eps = float(np.finfo(np.float64).eps)
    w = int(width)
    # WHERE YOU LOOK SETS THE WALL. Pixels stop separating when span/width falls below the ulp of the
    # largest coordinate in play, which at depth is the CENTRE. Centred exactly on the origin the
    # largest coordinate IS span/2, and span/width > eps*span/2 holds for every width below 2/eps --
    # so there is no eps wall at all there, only the smallest normal double. Quoting one constant for
    # every target would hide that, and it is the difference between "13.8 decades" and "essentially
    # unbounded" for two views of the same set.
    span_floor = (float(width) * eps * scale) if scale > 0.0 else float(np.finfo(np.float64).tiny)

    def _distinct(span):
        half = span * 0.5
        return int(len(np.unique(np.linspace(cx - half, cx + half, w))))

    # BRACKET the wall rather than sampling one side of it: at the floor the pixels must still be
    # distinct (or the floor is too pessimistic and we are refusing usable depth), and a decade below
    # it they must collide (or it is too optimistic and we would render arithmetic noise). An earlier
    # draft asserted only the second and failed -- at exactly the floor, all 320 are still distinct.
    at_floor, below = _distinct(span_floor), _distinct(span_floor * 0.1)
    return {"span_floor": span_floor,
            "decades": float(np.log10(float(span0) / span_floor)),
            "distinct_at_floor": at_floor, "distinct_below_floor": below, "width": w,
            "verified": at_floor == w and below < w,
            "why": "below span_floor adjacent pixel centres round to the same float64"}


def zoom_spans(span0, rate, frames):
    """The zoom schedule as a plain list -- span0 * rate**k. Separated from the render so a caller can
    plan, budget and clamp a flight path without rendering a pixel."""
    return [float(span0) * float(rate) ** k for k in range(int(frames))]


def deep_zoom(escape_fn, centre=(-0.743643887037151, 0.13182590420533), span0=3.0, rate=0.85,
              frames=24, width=320, height=180, max_iter=64, band=8, verify=False):
    """A deep zoom rendered BY the feedback buffer: magnify the last frame, refresh one band exactly.

    `escape_fn(width, height, centre, span, max_iter)` is passed in rather than imported so this stays a
    pure operator over any escape-time field -- and so the faculty can hand it `mind.escape_time` and
    keep the application's "everything through faculties" rule true.

    `band` is the refresh divisor: 1/`band` of the rows are recomputed exactly each frame, so every row
    is exact once per `band` frames and the error cannot accumulate. band=1 is a full recompute (exact,
    101 ms/frame); band=8 is 9.8 ms/frame at 1.15% mean error. STOPS at the float64 floor and says so.

    Returns {frames_rendered, ms_per_frame, stopped, final, spans, mean_abs_error}. `verify=True`
    additionally full-recomputes each frame to measure the error, which is ~10x slower BY DESIGN -- it
    is the instrument, not the effect."""
    import time
    floor = zoom_floor(centre, width, span0)
    spans = zoom_spans(span0, rate, frames)
    rows = max(1, int(height) // int(band))
    buf = np.asarray(escape_fn(width, height, centre, spans[0], max_iter), dtype=float)
    errs, stopped, rendered = [], "completed", 1
    t0 = time.time()
    for k in range(1, len(spans)):
        if spans[k] < floor["span_floor"]:
            stopped = "precision floor: span %.3e < %.3e (float64)" % (spans[k], floor["span_floor"])
            break
        buf = feedback_step(buf, zoom=1.0 / float(rate), decay=1.0, centre=CENTRE)
        r0 = min(((k * rows) % int(height)), int(height) - rows)
        buf[r0:r0 + rows] = _band(escape_fn, centre, spans[k], width, height, r0, rows, max_iter)
        rendered += 1
        if verify:
            truth = np.asarray(escape_fn(width, height, centre, spans[k], max_iter), dtype=float)
            errs.append(float(np.mean(np.abs(buf - truth))) / float(max_iter))
    ms = (time.time() - t0) / max(1, rendered - 1) * 1000.0
    return {"frames_rendered": rendered, "ms_per_frame": ms, "stopped": stopped, "final": buf,
            "spans": spans[:rendered], "band_rows": rows, "floor": floor,
            "mean_abs_error": (float(np.mean(errs)) if errs else None),
            "max_abs_error": (float(np.max(errs)) if errs else None)}


def _band(escape_fn, centre, span, width, height, r0, rows, max_iter):
    """Rows [r0, r0+rows) of the full frame, computed DIRECTLY rather than by cropping a full render.

    This is what makes the acceleration real: escape_fn derives its y extent from span*height/width, so
    a contiguous band is exactly a shorter render with a shifted centre_y. (A dithered every-k-th-row
    refresh is NOT expressible this way -- it was the author's first attempt and it silently cost a full
    frame per frame, making the "accelerated" path slower than the baseline.)"""
    half_y = float(span) * 0.5 * float(height) / float(width)
    ys = np.linspace(centre[1] - half_y, centre[1] + half_y, int(height))
    cy = 0.5 * (ys[r0] + ys[r0 + rows - 1])
    return np.asarray(escape_fn(width, rows, (centre[0], cy), span, max_iter), dtype=float)


def _selftest():
    rng = np.random.default_rng(0)
    buf = rng.random((48, 64))

    # 1. THE OPERATOR IS SHAPE- AND DETERMINISM-SAFE, and it does not mutate its input.
    before = buf.copy()
    out = feedback_step(buf, zoom=1.05, decay=0.9)
    assert out.shape == buf.shape and np.array_equal(buf, before), "feedback_step mutated its input"
    assert np.array_equal(out, feedback_step(buf, zoom=1.05, decay=0.9)), "not deterministic"

    # 2. DECAY IS THE CONTROL PARAMETER, and the critical value is exactly 1. Pinned on both sides,
    #    because a classifier that only ever says one thing would pass a one-sided test.
    assert feedback_fixed_point(buf, steps=200, zoom=1.0, decay=0.9)["verdict"] == "converged"
    assert feedback_fixed_point(buf, steps=400, zoom=1.0, decay=1.6)["verdict"] == "diverged"
    steady = feedback_fixed_point(buf, steps=24, zoom=1.0, decay=1.0)
    assert steady["verdict"] in ("steady", "cycle"), steady
    assert abs(steady["ratio"] - 1.0) < 1e-9, steady

    # 3. ENERGY FALLS AT THE RATE decay SAYS IT WILL -- the measurement, not just the label.
    r = feedback_fixed_point(buf, steps=12, zoom=1.0, decay=0.95, tol=0.0)
    assert abs(r["ratio"] - 0.95) < 0.02, r

    # 4. THE PRECISION WALL, both ways: predicted and verified must agree, and the headline is decades.
    f = zoom_floor((-0.743643887037151, 0.13182590420533), 320, 3.0)
    assert 13.0 < f["decades"] < 14.5, f
    assert f["verified"], f
    assert f["distinct_at_floor"] == 320 and f["distinct_below_floor"] < 320, f
    # THE FLOOR IS A PROPERTY OF WHERE YOU LOOK, not a constant. Centred on the origin there is no eps
    # wall (the largest coordinate shrinks with the span), so it goes vastly deeper -- a fixed constant
    # would be a lie in both directions. Measured: 13.8 decades at the seahorse target, >250 at 0.
    assert zoom_floor((0.0, 0.0), 320, 3.0)["decades"] > 100.0, zoom_floor((0.0, 0.0), 320, 3.0)
    near = zoom_floor((1e-3, 0.0), 320, 3.0)
    assert near["decades"] > f["decades"], (near, f)   # closer to 0 == deeper, monotonically

    # 5. THE ACCELERATION, end to end on a real escape-time field, against the full-recompute truth.
    from holographic.mesh_and_geometry.holographic_sdf import escape_time

    def fn(w, h, c, s, it):
        return escape_time(w, h, center=c, span=s, max_iter=it)

    z = deep_zoom(fn, frames=8, width=160, height=90, max_iter=48, band=8, verify=True)
    assert z["frames_rendered"] == 8 and z["stopped"] == "completed", z["stopped"]
    assert z["mean_abs_error"] < 0.05, z["mean_abs_error"]      # bounded by the refresh band
    assert z["final"].shape == (90, 160)

    # 6. IT STOPS AT THE WALL instead of zooming into arithmetic noise -- the abstention, pinned.
    deep = deep_zoom(fn, span0=1e-12, rate=0.1, frames=8, width=64, height=36, max_iter=16, band=4)
    assert deep["stopped"].startswith("precision floor"), deep["stopped"]
    assert deep["frames_rendered"] < 8, deep

    # 7. THE SIDEWAYS DIRECTION (sweep 134). The operator wears the SEQUENCE costume, and the claim is
    #    empirical rather than aesthetic: the critical decay must be the SAME NUMBER in both costumes.
    vec = rng.random(256)
    assert feedback_step(vec, zoom=1.0, rotate=3, decay=0.9).shape == vec.shape
    assert feedback_fixed_point(vec, steps=200, zoom=1.0, rotate=3, decay=0.9)["verdict"] == "converged"
    assert feedback_fixed_point(vec, steps=400, zoom=1.0, rotate=3, decay=1.6)["verdict"] == "diverged"
    seq = feedback_fixed_point(vec, steps=16, zoom=1.0, rotate=3, decay=0.95, tol=0.0)
    assert abs(seq["ratio"] - 0.95) < 1e-12, seq       # EXACT: a cyclic roll is a true permutation

    # 8. AND THE CONDITION IS PERMUTATION-NESS, NOT RANK -- which is the actual finding. A 2-D INTEGER
    #    roll is exact too; a rounded ROTATION is not, and is_permutation says why.
    assert is_permutation(vec, zoom=1.0, rotate=3)["permutation"] is True
    assert is_permutation(buf, zoom=1.0, rotate=0.0)["permutation"] is True
    rot = is_permutation(buf, zoom=1.0, rotate=0.15)
    assert rot["permutation"] is False and rot["sampled_once"] < rot["cells"], rot
    # the exactness follows the permutation, in BOTH ranks
    f_exact = feedback_fixed_point(buf, steps=16, zoom=1.0, rotate=0.0, decay=0.95, tol=0.0)
    assert abs(f_exact["ratio"] - 0.95) < 1e-12, f_exact

    print("holographic_feedback selftest OK: decay critical at 1.0 (converge 0.9 / diverge 1.6, ratio "
          "matched to 2%%), float64 floor %.1f decades verified, deep_zoom %d frames at %.1f ms with "
          "mean err %.3f, it STOPS at the wall, and the SEQUENCE costume shares the critical value "
          "exactly (ratio %.12f for decay 0.95, permutation=%s)"
          % (f["decades"], z["frames_rendered"], z["ms_per_frame"], z["mean_abs_error"],
             seq["ratio"], is_permutation(vec, zoom=1.0, rotate=3)["permutation"]))


if __name__ == "__main__":
    _selftest()

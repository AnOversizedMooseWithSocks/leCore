"""holographic_reclock.py -- sample when an AXIS moves, not when time passes ("make the boring property the
ruler").

WHY THIS MODULE EXISTS
----------------------
Uniform-time sampling spends most of its samples on nothing happening and then under-samples the moments that
matter. Re-clocking inverts that: pick an axis (cumulative movement of the series itself, photon count, error
growth, progress), and emit one event each time that axis traverses a fixed `step`. Quiet stretches become
cheap, busy stretches become dense, and the per-event DURATION (how much source time each step took) becomes a
clean "activity" channel with magnitude divided out. The renko/brick "price clock" is the canonical special
case (axis = cumulative |dx| of the tracked series); a photon-count clock or a flux-level-crossing clock is the
same abstraction pointed at a telescope.

THE TRAP, MEASURED, AND WHY null_persistence() IS BUILT IN
The re-clocking machinery MANUFACTURES structure. In the campaign this module comes from, the brick transform
produced 72% direction persistence ON PURE NOISE -- read naively, a spectacular momentum effect; referenced to
the null the same machinery produces, the real series was significantly ANTI-persistent (z=-7.3), the OPPOSITE
sign from the naive reading. The manufactured DIRECTION is a property of the mechanism, not of the data: the
campaign's net-displacement renko favoured continuation (+72% fake momentum), while THIS module's
total-variation price clock favours alternation (~25% agreement on the same pure noise -- a fake -25-point
"reversion effect", measured in _selftest). Two clocks, two confident opposite stories, one input of pure
noise. Neither number means anything until it is referenced to its own machinery's null. For that reason this module
does not merely document the trap -- `null_persistence()` runs the honest measurement (the full reclock on
surrogates, via holographic_honesty.pipeline_null) as one call, and the naive persistence number is never
returned without a route to its null.

MEASURED SCOPE, kept honest (the B1 lesson): re-clocking SHARPENED structure only when axis and rotation were
the same physical quantity (price-on-price: z=-7.3). Foreign axes -- an RSI clock or a volume clock carrying
price rotation -- added nothing (|z| < 1.4), and self-clocked foreign properties carried no cross-information
into the target. Re-clocking CONCENTRATES structure a quantity already has about itself; it does not transport
structure between quantities. Choose the axis accordingly, and expect nothing from a borrowed ruler.

NumPy + stdlib only. Deterministic throughout (the transform itself draws no randomness; the null helper is
seeded).
"""

import numpy as np


def reclock(series, step, axis=None):
    """Emit one event each time `axis` traverses `step`, carrying where and when it happened and which way the
    tracked series was going.

      series : the 1-D quantity whose ROTATION (direction of change) each event reports.
      step   : the axis distance per event, in the axis's own units. For the price clock this is the brick
               size, in the series' units.
      axis   : a 1-D NON-DECREASING cumulative quantity to clock against (photon count, cumulative volume,
               cumulative error). Default None = the PRICE CLOCK: axis is the cumulative |diff| of `series`
               itself -- the canonical case, and the only configuration whose sharpening effect is measured
               (see the module docstring's scope note).

    Returns a dict of aligned arrays, one entry per event:
      source_index : index into `series` at which the event completed (the first sample at or past the
                     threshold -- so acting on an event at its own source_index uses no future data).
      duration     : source samples consumed since the previous event (B2's activity channel). The FIRST
                     event's duration counts from sample 0.
      rotation     : sign (+1/-1/0) of the tracked series' NET change over the event's span.
      value        : the series value at the event.
      axis_value   : the axis value at the event.
    plus `n_events`, `step`, and `skipped_gap` -- how many events completed INSIDE a single source sample
    (axis jumped more than one step between samples). Those events are NOT synthesised: inventing them would
    fabricate rotations no sample ever witnessed. They are counted, and duration_resolution_check() turns the
    count into a warning when it matters.

    KEPT NEGATIVE (measured, the -inf log-duration incident): with `step` too small for the sampling rate,
    many events complete inside one sample -- durations quantise to a handful of values (or would be zero for
    the skipped ones), and any duration statistic becomes an artifact of the grid, not the process. 25bp
    bricks on 5-minute bars produced exactly this. Run duration_resolution_check() before trusting any
    duration-channel readout; it is one call and it names the failure."""
    x = np.asarray(series, float).ravel()
    if x.size < 2:
        raise ValueError("reclock needs at least 2 samples (got %d)" % x.size)
    step = float(step)
    if not (step > 0):
        raise ValueError("step must be > 0 (got %r)" % (step,))
    if axis is None:
        ax = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(x)))])
    else:
        ax = np.asarray(axis, float).ravel()
        if ax.size != x.size:
            raise ValueError("axis has %d samples but series has %d" % (ax.size, x.size))
        if np.any(np.diff(ax) < 0):
            raise ValueError("axis must be non-decreasing (a cumulative quantity); got a decrease -- pass "
                             "np.cumsum(np.abs(np.diff(...))) style axes, not raw signed ones")

    src, dur, rot, val, axval = [], [], [], [], []
    skipped = 0
    next_thr = ax[0] + step
    prev_i = 0
    for i in range(1, x.size):
        if ax[i] < next_thr:
            continue
        # the axis crossed at least one threshold inside (prev sample, this sample]. Emit ONE event at this
        # sample -- the first index where the crossing is knowable, so the event is causal -- and count any
        # additional whole steps as skipped rather than fabricating rotations no sample witnessed.
        n_cross = int((ax[i] - next_thr) // step) + 1
        skipped += n_cross - 1
        src.append(i)
        dur.append(i - prev_i)
        d = x[i] - x[prev_i]
        rot.append(0 if d == 0 else (1 if d > 0 else -1))
        val.append(x[i])
        axval.append(ax[i])
        prev_i = i
        next_thr += n_cross * step
    return {"source_index": np.array(src, int), "duration": np.array(dur, int),
            "rotation": np.array(rot, int), "value": np.array(val, float),
            "axis_value": np.array(axval, float), "n_events": len(src),
            "step": step, "skipped_gap": int(skipped)}


def rotation_persistence(events):
    """Fraction of consecutive events whose rotation agrees -- the naive momentum readout, provided so it has
    ONE definition instead of five hand-rolled ones. Zero-rotation events are dropped before pairing.

    THIS NUMBER MEANS NOTHING ON ITS OWN. The reclock machinery manufactures ~72% agreement on pure noise
    (module docstring); quote it only next to null_persistence()'s z, which is one call."""
    r = np.asarray(events["rotation"], int)
    r = r[r != 0]
    if r.size < 2:
        return float("nan")
    return float(np.mean(r[1:] == r[:-1]))


def null_persistence(series, step, axis=None, surrogate="iid_shuffle", n=200, seed=0, **surrogate_kwargs):
    """The HONEST persistence measurement: run the FULL reclock -> rotation_persistence chain on `series` and
    on `n` surrogates, and report the observed value against the null the machinery itself produces
    (holographic_honesty.pipeline_null, two-sided -- the campaign's real series came out on the ANTI-persistent
    side of its null, the opposite of the naive reading).

    Default surrogate is iid_shuffle: the persistence claim is about the ORDERING of moves, and a shuffle
    destroys exactly that while keeping the move-size distribution the brick quantisation feeds on. Pass
    "sign_flip" instead when the claim is specifically about direction with magnitude clustering kept.

    Returns pipeline_null's dict: {observed, null_mean, null_std, z, p, null_ci, collapsed, n, surrogate}.
    Expect null_mean far above 0.5 -- that gap IS the manufactured structure, on display.

    KEPT NEGATIVE: a custom `axis` is NOT re-derived per surrogate (the surrogate reorders the series, and an
    external axis has no defined reordering), so this helper only supports the price clock (axis=None), where
    the axis is recomputed from each surrogate by construction. A foreign-axis persistence claim needs a
    hand-built pipeline_null with a joint resampling story -- and the measured scope note says to expect
    nothing from it anyway."""
    if axis is not None:
        raise ValueError("null_persistence supports only the price clock (axis=None): a surrogate reorders "
                         "the series and an external axis has no defined reordering. Build a custom "
                         "pipeline_null with a joint resampling story if you really need a foreign axis.")
    from holographic.agents_and_reasoning.holographic_honesty import pipeline_null

    def chain(v):
        ev = reclock(v, step)
        p = rotation_persistence(ev)
        # a surrogate too tame to produce 2 nonzero-rotation events scores chance, not NaN -- NaNs would
        # poison the null's mean/std silently.
        return 0.5 if p != p else p

    return pipeline_null(chain, series, surrogate=surrogate, n=n, seed=seed, side="two-sided",
                         **surrogate_kwargs)


def duration_stats(events):
    """B2's duration channel in one report: the per-event durations as a series, their lag-1 autocorrelation
    in LOG space (activity clustering -- slow periods follow slow periods), and the up/down duration asymmetry
    (do rises take longer than falls?).

    Returns {durations, n, mean, median, log_ac1, mean_up, mean_down, updown_ratio, updown_z}. `updown_z` is
    a Welch z on mean log-duration up vs down; log space because durations are ratio-scaled (a 2x-slower event
    is the same "amount slower" at any base rate) and heavy-tailed.

    KEPT NEGATIVE: every number here is only as good as the duration RESOLUTION -- run
    duration_resolution_check() first; a quantised duration grid makes log_ac1 and the asymmetry artifacts of
    the step/sampling ratio (the -inf log-duration incident, module docstring)."""
    d = np.asarray(events["duration"], float)
    r = np.asarray(events["rotation"], int)
    if d.size == 0:
        raise ValueError("no events -- nothing to report; is `step` larger than the series' total movement?")
    logd = np.log(np.maximum(d, 1.0))                        # durations are >= 1 by construction; guard anyway
    if logd.size > 2 and np.std(logd) > 0:
        a = logd - logd.mean()
        log_ac1 = float(np.dot(a[:-1], a[1:]) / (np.dot(a, a) + 1e-300))
    else:
        log_ac1 = float("nan")
    up, down = logd[r > 0], logd[r < 0]
    mean_up = float(np.exp(up.mean())) if up.size else float("nan")
    mean_down = float(np.exp(down.mean())) if down.size else float("nan")
    if up.size > 1 and down.size > 1:
        se = np.sqrt(up.var(ddof=1) / up.size + down.var(ddof=1) / down.size)
        updown_z = float((up.mean() - down.mean()) / se) if se > 0 else float("nan")
    else:
        updown_z = float("nan")
    ratio = mean_up / mean_down if (mean_down == mean_down and mean_down > 0) else float("nan")
    return {"durations": d.astype(int), "n": int(d.size), "mean": float(d.mean()),
            "median": float(np.median(d)), "log_ac1": log_ac1,
            "mean_up": mean_up, "mean_down": mean_down, "updown_ratio": float(ratio),
            "updown_z": updown_z}


def duration_resolution_check(events, min_distinct=5, max_unit_frac=0.5):
    """Is the duration channel RESOLVED, or an artifact of the step/sampling grid? Warns when (a) too few
    DISTINCT duration values exist (`min_distinct`), (b) too many events complete in a single source sample
    (`max_unit_frac` of durations == 1), or (c) events completed INSIDE one sample (`skipped_gap` > 0 -- the
    -inf log-duration incident's exact signature: 25bp bricks completing inside single 5-minute bars).

    Returns {ok, n_distinct, unit_frac, skipped_gap, warnings}. `ok` is the gate; `warnings` names each
    failure and the fix (a larger step, or a finer source sampling). Cheap enough to run always -- and
    duration_stats' docstring tells you to."""
    d = np.asarray(events["duration"], int)
    warnings = []
    n_distinct = int(np.unique(d).size) if d.size else 0
    unit_frac = float(np.mean(d <= 1)) if d.size else 1.0
    skipped = int(events.get("skipped_gap", 0))
    if n_distinct < min_distinct:
        warnings.append("only %d distinct duration value(s) (< %d): the duration channel is quantised by the "
                        "step/sampling ratio -- use a larger step or finer sampling" % (n_distinct, min_distinct))
    if unit_frac > max_unit_frac:
        warnings.append("%.0f%% of events complete in a single source sample (> %.0f%%): durations carry the "
                        "grid, not the process" % (100 * unit_frac, 100 * max_unit_frac))
    if skipped > 0:
        warnings.append("%d event(s) completed INSIDE one source sample and were skipped rather than "
                        "fabricated: the step is small relative to per-sample movement" % skipped)
    return {"ok": not warnings, "n_distinct": n_distinct, "unit_frac": unit_frac,
            "skipped_gap": skipped, "warnings": warnings}


def _selftest():
    """Contracts:

    1. MECHANICS: on a straight ramp with unit steps, events land exactly every `step` samples, all rotations
       +1, all durations equal; the transform is deterministic and causal by construction (source_index is the
       first sample at which the crossing is knowable).
    2. THE MANUFACTURED-PERSISTENCE TRAP, on the record numerically: on PURE NOISE the naive rotation
       persistence is far above 0.5, and null_persistence reports it as NOTHING against the machinery's own
       null (|z| small) -- while the null's own mean sits far above 0.5, displaying the manufacturing.
    3. POWER: a genuinely trending-then-mean-reverting construction separates from its null through the same
       chain.
    4. B2: the resolution check catches a step chosen too small (the -inf log-duration incident's signature)
       and passes a well-chosen one; duration asymmetry detects a series built slow-up / fast-down.
    5. Refusals name their fix.
    """
    rng = np.random.default_rng(0)

    # (1) mechanics on a ramp: axis = cumulative |diff| = arange, so step=5 -> an event every 5 samples.
    ramp = np.arange(101, dtype=float)
    ev = reclock(ramp, step=5)
    assert ev["n_events"] == 20 and ev["skipped_gap"] == 0
    assert np.all(ev["duration"] == 5) and np.all(ev["rotation"] == 1)
    assert np.array_equal(ev["source_index"], np.arange(5, 101, 5))
    ev2 = reclock(ramp, step=5)
    assert all(np.array_equal(ev[k], ev2[k]) for k in ("source_index", "duration", "rotation"))  # deterministic

    # explicit-axis path: clock the same ramp on a photon-count-style external axis.
    photons = np.cumsum(rng.integers(0, 4, size=101)).astype(float)
    evp = reclock(ramp, step=10, axis=photons)
    assert evp["n_events"] > 0 and np.all(np.diff(evp["axis_value"]) > 0)

    # (2) THE TRAP. Pure noise -> naive persistence far from 0.5; honest z ~ 0; the null itself carries the
    #     same bias. NOTE the direction: THIS clock (total-variation axis) manufactures ANTI-persistence
    #     (~0.25 at step=2), where the campaign's net-displacement renko manufactured +72% persistence -- the
    #     manufactured DIRECTION is a property of the mechanism, and both naive readings are equally fake.
    noise = rng.normal(size=4000)
    ev_noise = reclock(noise, step=2.0)
    naive = rotation_persistence(ev_noise)
    honest = null_persistence(noise, step=2.0, n=100, seed=0)
    assert abs(naive - 0.5) > 0.15, naive                          # measured 0.25: a fake -25pt "reversion"
    assert abs(honest["z"]) < 2.5, honest                          # and it is NOTHING vs the machinery's null
    assert abs(honest["null_mean"] - 0.5) > 0.15, honest           # the null ITSELF displays the manufacturing
    assert abs(honest["null_mean"] - naive) < 0.05, honest         # ...and matches the naive reading closely
    assert not honest["collapsed"], honest

    # (3) POWER: an alternating-drift series (up-leg, down-leg, repeat) has real structure the chain sees.
    legs = np.concatenate([np.full(50, s) for s in ([+0.5, -0.5] * 40)])
    trend = np.cumsum(legs + 0.3 * rng.normal(size=legs.size))
    honest_t = null_persistence(trend, step=2.0, n=100, seed=0)
    assert abs(honest_t["z"]) > 3.0, honest_t                     # measured z = +101
    assert honest_t["collapsed"], honest_t

    # (4) B2: resolution. A step far below per-sample movement -> events complete inside single samples.
    jumpy = np.cumsum(rng.normal(0, 5.0, size=400))
    bad = duration_resolution_check(reclock(jumpy, step=0.5))
    assert not bad["ok"] and bad["skipped_gap"] > 0 and any("INSIDE" in w for w in bad["warnings"]), bad
    good = duration_resolution_check(reclock(jumpy, step=25.0))
    assert good["ok"], good

    # duration asymmetry on a slow-up / fast-down sawtooth: rises take ~4x the samples falls do.
    saw = np.concatenate([np.concatenate([np.linspace(0, 8, 40, endpoint=False),
                                          np.linspace(8, 0, 10, endpoint=False)]) for _ in range(20)])
    ds = duration_stats(reclock(saw, step=2.0))
    assert ds["mean_up"] > 2.0 * ds["mean_down"], ds               # measured ratio ~4
    assert ds["updown_z"] > 5.0, ds

    # (5) refusals.
    for bad_call, needle in ((lambda: reclock(ramp, step=0), "step must be > 0"),
                             (lambda: reclock(ramp, step=1, axis=np.sin(ramp)), "non-decreasing"),
                             (lambda: null_persistence(ramp, 1, axis=photons), "price clock"),
                             (lambda: duration_stats(reclock(ramp, step=1e9)), "nothing to report")):
        try:
            bad_call()
            raise AssertionError("expected ValueError (%s)" % needle)
        except ValueError as e:
            assert needle in str(e), (needle, str(e))

    print("holographic_reclock selftest OK (ramp mechanics exact: 20 events, dur 5, rot +1; THE TRAP pinned: "
          "naive persistence %.2f on pure noise -- a fake 'reversion effect' this clock manufactures where "
          "renko manufactured fake momentum -- honest z=%+.2f against the machinery's own null (null mean "
          "%.2f, the manufacturing on display); alternating-drift separates at z=%+.1f; resolution check "
          "catches step-too-small (skipped %d) and passes a sane step; slow-up/fast-down asymmetry ratio "
          "%.1fx z=%.1f; refusals name their fix)"
          % (naive, honest["z"], honest["null_mean"], honest_t["z"], bad["skipped_gap"],
             ds["updown_ratio"], ds["updown_z"]))


if __name__ == "__main__":
    _selftest()

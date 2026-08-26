"""holographic_eventstudy.py -- H2: what ACTUALLY happens after the signal fires, measured against a null
that keeps everything about the events except their alignment with the outcome.

WHY THIS MODULE EXISTS
----------------------
"Average the next H steps after each event" is the most natural analysis in sequential data and one of the
easiest to fool yourself with, in three specific ways this module closes one by one:

  1. THE NULL. Comparing the post-event mean to the unconditional mean assumes events are placed like an iid
     sample, which they never are (they cluster, they respect a refractory period, they avoid the edges).
     The null here is the CIRCULAR SHIFT: slide the whole event pattern by a random offset (mod n). This
     preserves the event COUNT and every INTER-EVENT SPACING exactly -- the pattern keeps its clustering, its
     refractory structure, everything -- and destroys only the alignment with the outcome, which is precisely
     the thing under test. (The same reasoning as pipeline_null: the null must share the machinery.)

  2. OVERLAP. When events sit closer together than the horizon, their forward windows share samples and the
     per-event "observations" are correlated -- the naive across-events standard error is too small and the
     naive CI too tight. Measured in _selftest rather than asserted: at heavy overlap the naive CI's false-
     alarm rate on pure noise is a multiple of nominal, while the circular-shift null (which inherits the
     SAME overlap through the preserved spacings) stays calibrated. The report carries n_events,
     n_overlapping and the shared-sample fraction so the reader can see the correlation structure.

  3. PRE-TREND. Events selected BECAUSE of what just happened (a threshold on recent runup, a breakout rule)
     carry their selection into the backward window: the pre-event path is not flat, and any post-event
     "drift" must be read against that. The window here is two-sided by default and `pre_trend` reports the
     backward slope with its own shift-null z -- a large pre-trend z is the "your event definition already
     contains the move" alarm, the event-study cousin of target_shift_probe's not-ahead dominance.

The path convention: paths are aligned to the event at offset 0, cumulative from the event (path[k] = sum of
the outcome over offsets 1..k for the forward side, and over -k..-1 for the backward side), so "drift" reads
directly as height. Events whose window would cross either edge are DROPPED and counted, never truncated --
a truncated window changes what the average means at exactly the offsets where it is most quoted.

NumPy + stdlib only. Deterministic given seed.
"""

import numpy as np


def event_study(outcome, events, horizon=20, pre=None, n_null=500, seed=0, alpha=0.05):
    """The aligned-window study. `outcome` is the per-step series the events are supposed to move (a diff, a
    return, an error, a load); `events` are integer indices into it.

    Returns a dict:
      offsets            [-pre .. +horizon] (0 = the event step itself, excluded from both cumulations)
      mean_path          cumulative mean outcome at each offset (see module docstring for the convention)
      n_events, n_dropped  usable events / dropped for edge-crossing (dropped, never truncated)
      n_overlapping      events closer than `horizon` to their successor
      shared_fraction    fraction of forward-window samples shared with another event's window
      forward            {stat, z, p} -- the summary statistic (mean cumulative forward outcome at +horizon)
                         against the circular-shift null
      pre_trend          {stat, z, p} -- the same for the backward window; |z| large means the event
                         definition already contains a move (selection, not prediction)
      null               {mean, std, n} for the forward stat, for the record

    KEPT NEGATIVE (measured in _selftest): the naive across-events t-CI is anticonservative under overlap --
    do not rebuild it from mean_path and n_events; the z here is the calibrated one BECAUSE the shifted
    pattern inherits the same overlap. And the shift null assumes the outcome series is (cyclo)stationary
    enough that a shifted alignment is exchangeable; on a strongly trending outcome, difference it first --
    the same rule as every surrogate in the null layer."""
    y = np.asarray(outcome, float).ravel()
    ev = np.asarray(sorted(int(e) for e in events), int)
    n = y.size
    horizon = int(horizon)
    pre = int(pre) if pre is not None else horizon
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if ev.size == 0:
        raise ValueError("no events given -- an event study of nothing is not a null result, it is no study")
    if np.any(ev < 0) or np.any(ev >= n):
        raise ValueError("event indices must lie inside the outcome series")

    keep = ev[(ev - pre >= 0) & (ev + horizon < n)]
    n_dropped = int(ev.size - keep.size)
    if keep.size < 3:
        raise ValueError("only %d event(s) fit inside the series with pre=%d, horizon=%d -- too few to "
                         "average (dropped %d at the edges)" % (keep.size, pre, horizon, n_dropped))

    gaps = np.diff(keep)
    n_overlapping = int(np.sum(gaps < horizon))
    covered = np.zeros(n, bool)
    shared = 0
    total = 0
    for e in keep:
        seg = slice(e + 1, e + 1 + horizon)
        shared += int(np.sum(covered[seg]))
        total += horizon
        covered[seg] = True

    def paths_for(evts):
        fwd = np.stack([np.cumsum(y[e + 1:e + 1 + horizon]) for e in evts])
        bwd = np.stack([np.cumsum(y[e - pre:e][::-1]) for e in evts])          # cumulative going backwards
        return fwd, bwd

    fwd, bwd = paths_for(keep)
    mean_fwd = fwd.mean(axis=0)
    mean_bwd = bwd.mean(axis=0)
    offsets = list(range(-pre, horizon + 1))
    mean_path = np.concatenate([mean_bwd[::-1] * -1.0, [0.0], mean_fwd])
    # backward cumulation sign: path[-k] is drawn so that a positive pre-event runup plots as rising INTO
    # the event -- the picture a reader expects from an event-study figure.

    stat_fwd = float(mean_fwd[-1])
    stat_pre = float(mean_bwd[-1])

    rng = np.random.default_rng(seed)
    null_fwd = np.empty(int(n_null))
    null_pre = np.empty(int(n_null))
    for i in range(int(n_null)):
        off = int(rng.integers(1, n - 1))
        shifted = (ev + off) % n                                # shift the ORIGINAL pattern: spacings intact
        k = shifted[(shifted - pre >= 0) & (shifted + horizon < n)]
        while k.size < 3:                                        # a shift can push the pattern into the edges;
            off = int(rng.integers(1, n - 1))                    # redraw -- count stays comparable
            shifted = (ev + off) % n
            k = shifted[(shifted - pre >= 0) & (shifted + horizon < n)]
        f, b = paths_for(k)
        null_fwd[i] = float(f.mean(axis=0)[-1])
        null_pre[i] = float(b.mean(axis=0)[-1])

    def _z(stat, null):
        mu, sd = float(null.mean()), float(null.std())
        z = (stat - mu) / sd if sd > 0 else 0.0
        p = float((np.sum(np.abs(null - mu) >= abs(stat - mu)) + 1) / (null.size + 1))
        return {"stat": float(stat), "z": float(z), "p": p}

    return {"offsets": offsets, "mean_path": mean_path,
            "n_events": int(keep.size), "n_dropped": n_dropped,
            "n_overlapping": n_overlapping, "shared_fraction": float(shared / max(total, 1)),
            "forward": _z(stat_fwd, null_fwd), "pre_trend": _z(stat_pre, null_pre),
            "null": {"mean": float(null_fwd.mean()), "std": float(null_fwd.std()), "n": int(n_null)}}


def _selftest():
    """Contracts:
    1. POWER: a planted post-event drift is detected against the shift null; the mean path shows it.
    2. CALIBRATION: random events on pure noise read z ~ 0, and across many noise draws the shift-null p is
       ~uniform (false-alarm at nominal) EVEN AT HEAVY OVERLAP -- while the naive across-events t-test's
       false-alarm rate at the same overlap is a multiple of nominal. The overlap negative, measured.
    3. PRE-TREND: events defined by a threshold on recent runup show a large pre_trend z and (on noise) no
       forward z -- selection made visible, prediction correctly absent.
    4. Edge events are dropped and counted; refusals name their reason.
    """
    rng = np.random.default_rng(0)

    # (1) planted drift: +0.3 per step for 10 steps after each of 40 well-spaced events.
    n = 6000
    y = rng.standard_normal(n)
    ev = np.arange(100, 5900, 145)
    for e in ev:
        y[e + 1:e + 11] += 0.3
    r = event_study(y, ev, horizon=20, seed=0)
    assert r["forward"]["z"] > 2.5 and r["forward"]["p"] < 0.01, r["forward"]
    assert r["mean_path"][r["offsets"].index(10)] > 2.0          # ~3.0 planted, visible in the path
    assert r["pre_trend"]["p"] > 0.05, r["pre_trend"]            # nothing planted before the event

    # (2a) calibration on noise, sparse events
    y0 = rng.standard_normal(n)
    r0 = event_study(y0, ev, horizon=20, seed=1)
    assert abs(r0["forward"]["z"]) < 3, r0["forward"]

    # (2b) THE OVERLAP NEGATIVE, measured: dense events (spacing 6 < horizon 20) on pure noise, 200 draws.
    #      naive across-events t on the cumulative forward stat vs the shift-null p.
    import math
    ev_dense = np.arange(50, 1950, 6)
    naive_fp = 0
    shift_fp = 0
    draws = 200
    for d in range(draws):
        yd = np.random.default_rng(1000 + d).standard_normal(2000)
        fwd = np.stack([np.cumsum(yd[e + 1:e + 21]) for e in ev_dense if e + 21 < 2000 and e - 20 >= 0])
        stat = fwd[:, -1]
        t = stat.mean() / (stat.std(ddof=1) / math.sqrt(len(stat)))
        naive_fp += (abs(t) > 1.96)
        rs = event_study(yd, ev_dense, horizon=20, n_null=200, seed=d)
        shift_fp += (rs["forward"]["p"] < 0.05)
    naive_rate, shift_rate = naive_fp / draws, shift_fp / draws
    assert naive_rate > 0.15, naive_rate                         # ~4x nominal: overlap wrecks the naive CI
    assert shift_rate < 0.12, shift_rate                         # the shift null inherits the overlap: calibrated
    r_dense = event_study(np.random.default_rng(7).standard_normal(2000), ev_dense, horizon=20, seed=7)
    assert r_dense["n_overlapping"] > 200 and r_dense["shared_fraction"] > 0.5

    # (3) selection made visible: events fire when the trailing 10-step sum exceeds a threshold. On noise the
    #     backward window then contains the very runup that defined the event; forward contains nothing.
    y3 = rng.standard_normal(n)
    # trailing sum: full-convolution index t is sum of y3[t-9..t]. The FIRST draft sliced [9:n], which is the
    # FORWARD sum -- the fixture itself leaked the future, and the pre-trend came out negative noise while
    # the leak hid in the event definition. Caught because the assert disagreed with the story; kept because
    # an event-study fixture leaking the future is exactly the bug this module hunts in the wild.
    run = np.convolve(y3, np.ones(10), mode="full")[:n]
    ev3 = np.where((run > 4.0) & (np.arange(n) >= 9))[0]
    ev3 = ev3[np.concatenate([[True], np.diff(ev3) > 25])]       # refractory: one event per excursion
    r3 = event_study(y3, ev3, horizon=20, pre=12, seed=0)
    assert r3["pre_trend"]["z"] > 4, r3["pre_trend"]             # the definition IS the pre-trend
    assert r3["forward"]["p"] > 0.05, r3["forward"]              # and it predicts nothing

    # (4) edge events are DROPPED (never truncated) and the drop can leave too few -- which refuses by name
    #     rather than averaging one path and calling it a study. And a wide-enough series keeps its middle
    #     events while still counting the dropped edges.
    r4 = event_study(rng.standard_normal(400), [2, 100, 200, 300, 398], horizon=20, pre=20, seed=0)
    assert r4["n_dropped"] == 2 and r4["n_events"] == 3, (r4["n_dropped"], r4["n_events"])
    try:
        event_study(rng.standard_normal(200), [2, 100, 198], horizon=20, pre=20, seed=0)
        raise AssertionError("expected too-few refusal")
    except ValueError as e:
        assert "too few" in str(e)
    try:
        event_study(rng.standard_normal(100), [], horizon=5)
        raise AssertionError("expected no-events refusal")
    except ValueError as e:
        assert "no events" in str(e)

    print("holographic_eventstudy selftest OK (planted +0.3x10 drift: forward z=%.1f p=%.3f with a flat pre-trend; "
          "noise calibrated; THE OVERLAP NEGATIVE MEASURED over 200 draws at spacing 6 vs horizon 20 -- naive "
          "across-events t false-alarms at %.0f%% where the circular-shift null holds %.0f%% (nominal 5%%), "
          "because the shifted pattern inherits the same overlap; threshold-defined events show pre_trend "
          "z=%.1f with forward p=%.2f -- selection visible, prediction absent; edge events dropped and "
          "counted, never truncated)"
          % (r["forward"]["z"], r["forward"]["p"], 100 * naive_rate, 100 * shift_rate, r3["pre_trend"]["z"], r3["forward"]["p"]))


if __name__ == "__main__":
    _selftest()

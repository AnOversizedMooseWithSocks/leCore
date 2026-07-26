"""holographic_actioncost.py -- the ACTION layer's two honesty gates: the cost wall (G1) and the
emission-vs-actionable fill discipline (G2). Signal quality and action cost are different measurements, and a
detector's worth is decided at the moment of ACTION, at the price/state actually reachable then -- not at the
moment of detection.

WHY THIS MODULE EXISTS (two measured graves)
  * THE COST WALL: every intraday gross edge the source campaign found (+1.9, +4.5, +9.6, +10.6 bp per event)
    died at a 17 bp round-trip cost wall, approached from both directions. Not one was a bad detection; all
    were real, and none survived acting. The wall is a property of the ACTION channel, so it belongs in the
    readout next to the gross number, not in a footnote.
  * EMISSION vs ACTIONABLE: a composed brick battery measured z=+20 continuation FROM BRICK EMISSION -- and the
    tradeable version (enter at the raw price at the moment the brick is KNOWN) had NEGATIVE gross. The signal
    was late by construction: measuring from the emission price credits the move that happened while the event
    was completing, which no actor could have captured. The same failure generalises to any
    detection-vs-action latency (a telescope trigger fires after the transient's onset; a scheduler reacts
    after the load spike began).

Both gates are evaluators over (events, path): domain-neutral, NumPy + stdlib, deterministic. "Price" below is
any acted-on state variable.
"""

import math

import numpy as np


def net_of_costs(event_values, cost, per_side=None):
    """G1 -- THE COST WALL: gross per-event value against the round-trip cost of acting, as one readout.

    event_values : per-event GROSS values, in the same units as `cost` (e.g. bp per trade; joules per
                   actuation; seconds of compute per intervention).
    cost         : round-trip cost per event. Give EITHER `cost` (total) or `per_side=(entry, exit)`, which is
                   summed -- the split exists because entry and exit often live on different fee schedules
                   (maker/taker; act-now vs rest-and-wait).

    Returns {n, gross_mean, gross_t, net_mean, net_t, cost, wall_ratio, survives, breakeven_cost}.
    `wall_ratio` = gross_mean / cost: below 1.0 the wall wins on average before variance is even discussed.
    `survives` = net mean positive AND significant (normal-approx t, the engine's NumPy-only convention).
    `breakeven_cost` is the cost at which the mean edge exactly dies -- quote it, because "survives at 5 bp,
    dies at 9" is a portable fact about the signal while "survives" alone is a fact about today's fees.

    KEPT NEGATIVE: a constant cost is itself a model. Real action costs are state-dependent (spreads widen in
    storms -- exactly when many signals fire), so a signal that "clears the wall" at the average cost can lose
    to the cost it actually meets. For state-dependent costs pass per-event costs as an ARRAY in `cost`; the
    readout then uses each event's own cost, and the conditional split of net values by regime is one
    mind.conditional call away."""
    v = np.asarray(event_values, float).ravel()
    if v.size < 2:
        raise ValueError("net_of_costs needs at least 2 events (got %d)" % v.size)
    if per_side is not None:
        if cost is not None:
            raise ValueError("give either cost or per_side=(entry, exit), not both")
        cost = float(per_side[0]) + float(per_side[1])
    c = np.asarray(cost, float)
    if c.ndim == 0:
        c = np.full(v.size, float(c))
    elif c.size != v.size:
        raise ValueError("per-event cost array has %d entries for %d events" % (c.size, v.size))
    if np.any(c < 0):
        raise ValueError("costs must be >= 0")
    net = v - c

    def t_of(w):
        sd = float(w.std(ddof=1))
        return float(w.mean() / (sd / math.sqrt(w.size))) if sd > 0 else 0.0

    gross_mean, net_mean = float(v.mean()), float(net.mean())
    net_t = t_of(net)
    p_net = math.erfc(abs(net_t) / math.sqrt(2.0))
    mean_cost = float(c.mean())
    return {"n": int(v.size), "gross_mean": gross_mean, "gross_t": t_of(v),
            "net_mean": net_mean, "net_t": net_t, "cost": mean_cost,
            "wall_ratio": gross_mean / mean_cost if mean_cost > 0 else float("inf"),
            "survives": bool(net_mean > 0 and p_net <= 0.05),
            "breakeven_cost": gross_mean}


def realizable_fills(event_index, path, horizon, lag=1, cost=0.0, side=None, emission_price=None):
    """G2 -- EMISSION vs ACTIONABLE, both computed, actionable the headline. For each event, the forward value
    over `horizon` is measured TWICE:

      actionable : entry at path[i + lag] -- the first state reachable AFTER the event is KNOWN (`lag` >= 1;
                   lag=0 is refused, because acting in the same sample that defines the event is the idealised
                   fill this function exists to forbid as a default).
      idealized  : entry at the event's own emission price -- path[i], or the supplied `emission_price` array
                   (e.g. a brick's close, which may sit far from any tradable sample).

    side : optional +1/-1 per event (the acted direction); values are side * (exit - entry). Default long.
    cost : round-trip cost per event, subtracted from both readouts (G1 inside G2, so the two gates cannot be
           quoted separately by accident).

    Returns {n, actionable_mean, actionable_t, idealized_mean, idealized_t, latency_cost, verdict}.
    `latency_cost` = idealized_mean - actionable_mean: the part of the "edge" that is pure detection latency --
    the move that completed while the event was being recognised, which no actor gets. The measured canon:
    z=+20 at emission, NEGATIVE gross at the actionable price; latency was the entire finding.

    KEPT NEGATIVE: lag=1 is the FLOOR, not the truth -- real actuation adds queueing, transmission, and
    decision time on top. A signal that survives lag=1 and dies at lag=2 is a statement about your
    infrastructure budget, so sweep the lag before believing the edge (one loop; the function is cheap)."""
    p = np.asarray(path, float).ravel()
    idx = np.asarray(event_index, int).ravel()
    if idx.size < 2:
        raise ValueError("realizable_fills needs at least 2 events (got %d)" % idx.size)
    lag = int(lag)
    if lag < 1:
        raise ValueError("lag must be >= 1: lag=0 is the idealized emission fill, which is reported separately "
                         "on purpose -- an actor cannot enter at the price that defines the event")
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be >= 1 (got %d)" % horizon)
    s = np.ones(idx.size) if side is None else np.asarray(side, float).ravel()
    if s.size != idx.size:
        raise ValueError("side has %d entries for %d events" % (s.size, idx.size))
    em = p[idx] if emission_price is None else np.asarray(emission_price, float).ravel()
    if em.size != idx.size:
        raise ValueError("emission_price has %d entries for %d events" % (em.size, idx.size))

    keep = idx + lag + horizon < p.size
    if not np.any(keep):
        raise ValueError("no event has room for lag+horizon=%d samples of forward path -- shorten the horizon "
                         "or supply a longer path" % (lag + horizon))
    idx, s, em = idx[keep], s[keep], em[keep]
    entry_act = p[idx + lag]
    exit_ = p[idx + lag + horizon]
    act = s * (exit_ - entry_act) - cost
    ide = s * (exit_ - em) - cost

    def t_of(w):
        sd = float(w.std(ddof=1))
        return float(w.mean() / (sd / math.sqrt(w.size))) if sd > 0 else 0.0

    a_m, i_m = float(act.mean()), float(ide.mean())
    a_t = t_of(act)
    latency = i_m - a_m
    p_a = math.erfc(abs(a_t) / math.sqrt(2.0))
    if i_m > 0 and a_m <= 0:
        verdict = ("LATENCY ARTIFACT: the edge exists only at the emission price (idealized %+.4g vs actionable "
                   "%+.4g). The %+.4g difference is the move that completed while the event was being "
                   "recognised -- no actor gets it." % (i_m, a_m, latency))
    elif a_m > 0 and p_a <= 0.05:
        verdict = ("ACTIONABLE: %+.4g per event at the reachable price (t=%.1f), %+.4g of latency cost already "
                   "excluded." % (a_m, a_t, latency))
    else:
        verdict = ("NOT ESTABLISHED at the actionable price (%+.4g, t=%.1f) -- and the idealized number "
                   "(%+.4g) is not evidence; it contains the latency." % (a_m, a_t, i_m))
    return {"n": int(idx.size), "actionable_mean": a_m, "actionable_t": a_t,
            "idealized_mean": i_m, "idealized_t": t_of(ide),
            "latency_cost": float(latency), "verdict": verdict}


def _selftest():
    """Contracts:

    1. G1 mechanics: net = gross - cost exactly; wall_ratio and breakeven read straight off the inputs; a
       +10 bp edge dies at a 17 bp wall (the campaign's shape) and survives a 5 bp one; per-event cost arrays
       (state-dependent costs) shift the answer when costs concentrate where the signal fires.
    2. G2's canon, reproduced structurally: events defined by the COMPLETION of a move (a threshold-crossing
       detector on a mean-reverting path) show a strong POSITIVE idealized edge and a NON-POSITIVE actionable
       one -- the latency IS the finding. A genuinely predictive event (drift begins at the event) survives
       the actionable fill.
    3. Refusals: lag=0 is refused by name; both functions name their fixes.
    """
    rng = np.random.default_rng(0)

    # (1) G1.
    edge = rng.normal(10.0, 20.0, size=400)                       # +10 bp gross, honest spread
    dead = net_of_costs(edge, cost=17.0)
    alive = net_of_costs(edge, cost=5.0)
    assert not dead["survives"] and alive["survives"], (dead, alive)
    assert abs(dead["net_mean"] - (dead["gross_mean"] - 17.0)) < 1e-9
    assert abs(dead["breakeven_cost"] - dead["gross_mean"]) < 1e-9
    assert 0 < dead["wall_ratio"] < 1 < alive["wall_ratio"]
    # state-dependent costs: same AVERAGE cost, different answer -- a constant cost is a model, and the sign
    # of the error follows the cost-value COVARIANCE. Measured surprise, kept: costs landing on the GOOD
    # events RAISED net_t (4.28 -> 5.29; subtracting more from larger values compresses the net spread), the
    # opposite of the first draft's guess. The harmful case -- the storm shape, costs high exactly where
    # values are already bad -- is the one pinned as worse.
    on_bad = np.where(edge < 10.0, 10.0, 0.2)
    on_bad = on_bad * (5.0 / on_bad.mean())
    r_const, r_bad = net_of_costs(edge, cost=5.0), net_of_costs(edge, cost=on_bad)
    assert abs(r_const["cost"] - r_bad["cost"]) < 1e-9             # same average cost...
    assert r_bad["net_t"] < r_const["net_t"], (r_bad, r_const)     # ...worse when it lands on the bad events

    # (2) G2. Mean-reverting path + completion detector = the latency trap.
    n = 20000
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.9 * ar[i - 1] + rng.normal()
    # event: the path has just RISEN by >2 over 5 samples (a completed move -- brick-like by construction).
    rise = ar[5:] - ar[:-5]
    ev = np.nonzero(rise > 2.0)[0] + 5
    ev = ev[(ev > 10) & (ev < n - 30)][::3]                        # thin overlapping triggers
    side = np.ones(ev.size)                                        # FOLLOW the detected rise (continuation)
    # idealized entry: the price at the START of the detected move -- the emission-style anchor the detector's
    # own statistic is measured from, 5 samples before anything was knowable. Long-from-there "captures" the
    # completed rise itself: the canon's z=+20 continuation, manufactured out of the detection window.
    trap = realizable_fills(ev, ar, horizon=10, lag=1, side=side, emission_price=ar[ev - 5])
    assert trap["idealized_mean"] > 0.5, trap                      # a spectacular fake edge...
    assert trap["actionable_mean"] < 0.0, trap                     # ...that is NEGATIVE at the reachable price
    assert trap["latency_cost"] > 0.5 * trap["idealized_mean"], trap
    # a genuinely predictive event: drift starts AT the event, so the reachable entry still captures it.
    drift = np.cumsum(rng.normal(size=n) * 0.1)
    starts = np.arange(50, n - 200, 400)
    for s0 in starts:
        drift[s0:s0 + 30] += np.linspace(0, 3.0, 30)               # a 3-unit rise beginning at the event
    real = realizable_fills(starts, drift, horizon=20, lag=1)
    assert real["actionable_mean"] > 1.0 and real["actionable_t"] > 5.0, real
    assert real["verdict"].startswith("ACTIONABLE"), real

    # (3) refusals.
    for bad, needle in ((lambda: realizable_fills(ev, ar, horizon=10, lag=0), "lag must be >= 1"),
                        (lambda: realizable_fills(ev, ar, horizon=0), "horizon must be >= 1"),
                        (lambda: net_of_costs(edge, cost=5.0, per_side=(2, 3)), "not both"),
                        (lambda: net_of_costs(edge, cost=-1.0), ">= 0")):
        try:
            bad()
            raise AssertionError("expected ValueError (%s)" % needle)
        except ValueError as e:
            assert needle in str(e), (needle, str(e))

    print("holographic_actioncost selftest OK (G1: +%.1f bp gross dies at a 17 bp wall (net t=%.1f) and "
          "survives 5 bp (t=%.1f); equal-average state-dependent costs landing on the BAD events cut net t "
          "%.1f->%.1f (and on the good events RAISED it -- the covariance surprise, kept); G2: completion-detector edge %+.2f idealized "
          "vs %+.2f actionable -- %.0f%% of it was latency -- while a genuinely predictive event survives at "
          "%+.2f, t=%.1f; lag=0 refused by name)"
          % (dead["gross_mean"], dead["net_t"], alive["net_t"], r_const["net_t"], r_bad["net_t"],
             trap["idealized_mean"], trap["actionable_mean"],
             100 * trap["latency_cost"] / trap["idealized_mean"],
             real["actionable_mean"], real["actionable_t"]))


if __name__ == "__main__":
    _selftest()

"""holographic_paperbook.py -- G3 + G4: what PASSIVE fills actually cost, and the walk-forward paper account
that keeps every honesty gate attached while a strategy runs forward.

G3 -- RESTING-ORDER ADVERSE SELECTION (resting_fill_sim)
--------------------------------------------------------
A resting (limit) order does not choose when it trades; the MARKET does, and the market chooses to hit it
exactly when the price is moving through it. So conditioning on "my resting buy filled" IS conditioning on
"the price just fell through my level" -- and on a pure random walk, with no signal anywhere, the post-fill
mark-out of passive fills is NEGATIVE. That is not a bug in the strategy; it is a property of the fill
mechanism, the passive-side twin of realizable_fills' latency artifact (there the aggressive side paid for
arriving late; here the passive side pays for being chosen). The simulator measures it: fill rate, mark-out
at a horizon, and the SELECTION COST -- the mark-out of filled orders minus the mark-out of the same entries
had they all filled unconditionally.

The path's character sets HOW BAD, and the honest quantity is the EXTRA adverse beyond the discount
(selection_cost + delta): measured at -2.45 on a momentum path, -0.53 on a random walk, -0.21 under mean
reversion (delta = 1 sigma, horizon 10). The first draft claimed reversion FLIPS the mark-out positive --
refuted: reversion refunds most of the adverse excursion but NOT the discount; the filled mark-out sits near
zero, never meaningfully positive at this depth. Passive resting is punished everywhere and punished worst
where the flow that hits you keeps going. DEPTH, third refuted narrative: on a martingale the per-fill extra
adverse SHRINKS with depth (-0.58 / -0.53 / -0.31 / -0.16 at delta = 0.5 / 1 / 2 / 3) -- it is essentially
the discrete OVERSHOOT of the level, not a toxicity that deepens -- while the fill rate collapses (0.71 ->
0.26). The cost of resting deep is OPPORTUNITY, not per-fill toxicity; both curves pinned.

G4 -- THE PAPER BOOK (PaperBook)
--------------------------------
The forward-test harness with the gates built in, so "let's just paper trade it" cannot quietly drop them:
  * entries are ACTIONABLE (lag >= 1 enforced; lag=0 refused with realizable_fills' words),
  * per-side costs applied to every trade,
  * an optional CAUSAL GATE mask stands the book aside (and the mask's causality can be audited upstream),
  * SLEEVES: each strategy tracked separately AND combined, with the cross-sleeve MEDIAN reported beside the
    mean -- one lucky sleeve drags a mean; it does not drag a median.
Report per sleeve: net, t, max drawdown, trade count, equity curve; plus the median-across-sleeves summary.

KEPT NEGATIVE, structural: a paper book proves PLUMBING (costs, lags, gates wired and honest), not EDGE --
its numbers are one path's realisation, and the split-half / ledger discipline still applies to anything it
appears to show. The report says so in its own verdict field.

NumPy + stdlib only, deterministic given seed.
"""

import math

import numpy as np


def resting_fill_sim(path, events, delta, side=1, horizon=10):
    """Rest a limit order `delta` BELOW the current price for buys (side=+1) / above for sells (side=-1) at
    each event time; it fills at the first step in the next `horizon` steps whose price trades through the
    level; mark out the filled position `horizon` steps after the fill, at the level (the fill price).

    Returns {n_orders, n_filled, fill_rate, markout_mean, markout_t, selection_cost, verdict} where
    selection_cost = markout of FILLED orders minus the mark-out of ALL orders had each filled at its level
    unconditionally at event time -- the pure price of being chosen.

    KEPT NEGATIVE: this is a price-path simulator -- no queue, no partial fills, no competing size. Queue
    position generally makes real adverse selection WORSE than measured here (you fill last, at the most
    informed moments), so read these numbers as the OPTIMISTIC bound and say so when quoting them."""
    p = np.asarray(path, float).ravel()
    ev = [int(e) for e in events]
    side = int(side)
    if side not in (-1, 1):
        raise ValueError("side must be +1 (resting buy below) or -1 (resting sell above)")
    if delta <= 0:
        raise ValueError("delta must be positive -- a resting order AT the touch is a different instrument")
    filled_mark, uncond_mark = [], []
    n_orders = 0
    for e in ev:
        if e + horizon + horizon >= p.size:
            continue
        n_orders += 1
        level = p[e] - side * float(delta)
        window = p[e + 1:e + 1 + horizon]
        hit = np.where((window <= level) if side == 1 else (window >= level))[0]
        uncond_mark.append(side * (p[e + horizon] - level))
        if hit.size:
            t_fill = e + 1 + int(hit[0])
            filled_mark.append(side * (p[t_fill + horizon] - level))
    if n_orders < 8:
        raise ValueError("only %d orders fit the path with horizon=%d -- too few to measure" % (n_orders, horizon))
    n_filled = len(filled_mark)
    if n_filled < 5:
        return {"n_orders": n_orders, "n_filled": n_filled, "fill_rate": n_filled / n_orders,
                "markout_mean": float("nan"), "markout_t": 0.0, "selection_cost": float("nan"),
                "verdict": "only %d fills -- the level is too deep for this path to reach; that scarcity is "
                           "the measurement" % n_filled}
    fm = np.asarray(filled_mark)
    um = np.asarray(uncond_mark)
    se = fm.std(ddof=1) / math.sqrt(n_filled) if n_filled > 1 else 1.0
    t = float(fm.mean() / se) if se > 0 else 0.0
    sel = float(fm.mean() - um.mean())
    verdict = ("passive fills mark out %+.4f (t=%.1f) vs %+.4f unconditional -- selection cost %+.4f per "
               "fill; being CHOSEN is the mechanism, not the strategy" % (fm.mean(), t, um.mean(), sel))
    return {"n_orders": n_orders, "n_filled": n_filled, "fill_rate": float(n_filled / n_orders),
            "markout_mean": float(fm.mean()), "markout_t": t, "selection_cost": sel, "verdict": verdict}


class PaperBook:
    """The walk-forward paper account (G4). Construct, add sleeves, run once over a path.

        book = PaperBook(lag=1, cost=0.02)
        book.add_sleeve("momo", decisions_momo)     # per-step side in {-1, 0, +1}, aligned with the path
        book.add_sleeve("meanrev", decisions_mr)
        report = book.run(path, gate_mask=storm_gate_mask)

    Semantics per step t where a sleeve's decision d[t] != 0 and the gate (if any) is OPEN at t: enter at
    path[t + lag] (the ACTIONABLE price -- lag >= 1 enforced), exit one step later at path[t + lag + 1], pay
    `cost` per trade. One-step holding keeps the accounting exact and composable; longer horizons are the
    caller's aggregation.

    The report carries, per sleeve: net, mean, t, n_trades, max_drawdown, equity (cumulative); combined (all
    sleeves summed); and across-sleeves MEDIANS beside the means. The verdict repeats the structural kept
    negative: plumbing, not edge."""

    def __init__(self, lag=1, cost=0.0):
        if int(lag) < 1:
            raise ValueError("lag=0 would enter at the price that emitted the signal -- simultaneous is not "
                             "past; use lag >= 1")
        self.lag = int(lag)
        self.cost = float(cost)
        self._sleeves = {}

    def add_sleeve(self, name, decisions):
        if name in self._sleeves:
            raise ValueError("duplicate sleeve name %r -- names must be unique so the medians are honest" % name)
        self._sleeves[name] = np.sign(np.asarray(decisions, float).ravel())
        return self

    def run(self, path, gate_mask=None):
        p = np.asarray(path, float).ravel()
        if not self._sleeves:
            raise ValueError("no sleeves -- an empty book has nothing to run, which is a statement about the "
                             "workflow, not the data")
        gate = np.ones(p.size, bool) if gate_mask is None else np.asarray(gate_mask, bool).ravel()
        if gate.size != p.size:
            raise ValueError("gate_mask has %d flags for %d prices" % (gate.size, p.size))
        out = {"sleeves": {}, "lag": self.lag, "cost": self.cost}
        combined = None
        for name, d in self._sleeves.items():
            if d.size != p.size:
                raise ValueError("sleeve %r has %d decisions for %d prices" % (name, d.size, p.size))
            valid = np.arange(p.size - self.lag - 1)
            act = (d[valid] != 0) & gate[valid]
            entries = valid[act]
            pnl = d[entries] * (p[entries + self.lag + 1] - p[entries + self.lag]) - self.cost
            equity = np.cumsum(pnl) if pnl.size else np.zeros(0)
            peak = np.maximum.accumulate(equity) if equity.size else np.zeros(0)
            maxdd = float((equity - peak).min()) if equity.size else 0.0
            se = pnl.std(ddof=1) / math.sqrt(pnl.size) if pnl.size > 1 else 0.0
            out["sleeves"][name] = {"net": float(pnl.sum()), "mean": float(pnl.mean()) if pnl.size else 0.0,
                                    "t": float(pnl.mean() / se) if se > 0 else 0.0,
                                    "n_trades": int(pnl.size), "max_drawdown": maxdd,
                                    "equity": equity}
            step_pnl = np.zeros(p.size)
            step_pnl[entries] = pnl
            combined = step_pnl if combined is None else combined + step_pnl
        eq = np.cumsum(combined)
        peak = np.maximum.accumulate(eq)
        nets = [s["net"] for s in out["sleeves"].values()]
        means = [s["mean"] for s in out["sleeves"].values()]
        out["combined"] = {"net": float(combined.sum()), "max_drawdown": float((eq - peak).min()),
                           "equity": eq}
        out["across_sleeves"] = {"median_net": float(np.median(nets)), "mean_net": float(np.mean(nets)),
                                 "median_mean": float(np.median(means))}
        out["verdict"] = ("paper book: plumbing proven (lag=%d actionable entries, cost %.4g/trade, gate %s)"
                          " -- these numbers are ONE path's realisation; edge claims still owe split_half "
                          "and the ledger" % (self.lag, self.cost,
                                              "applied" if gate_mask is not None else "none"))
        return out


def _selftest():
    """Contracts:
    G3.1  STRUCTURAL adverse selection on a pure random walk, and the naive-backtest trap it exposes: the
          UNCONDITIONAL mark-out is ~+delta BY CONSTRUCTION (a fill at delta below spot pockets the discount
          -- which is exactly what a fill-anything backtest credits you), while the FILLED mark-out is
          NEGATIVE: reality claws back MORE than the whole discount. selection_cost ~ -(delta + adverse).
    G3.2  The path-character ORDERING of the extra adverse (selection_cost + delta): momentum << random walk
          < mean reversion ~ 0 -- reversion refunds the adverse excursion but NOT the discount. The first
          draft's "reversion flips it positive" is REFUTED and pinned refuted.
    G3.3  DEPTH: per-fill extra adverse SHRINKS with depth on a martingale (overshoot, not deepening
          toxicity) while fill rate collapses -- the deep-resting cost is opportunity. A level too deep to
          fill returns a scarcity verdict, not a number from 4 fills.
    G4.1  Sleeves separate and combined; a planted profitable sleeve reads positive net while a coin-flip
          sleeve reads ~0, and the across-sleeves MEDIAN is not dragged by the winner the way the mean is.
    G4.2  The gate stands the book aside: gating a planted storm removes the storm's losses (net and maxdd
          both improve) with fewer trades.
    G4.3  lag is applied: the same decisions at lag=1 vs an idealized entry differ exactly as
          realizable_fills teaches; lag=0 refused by name.
    """
    rng = np.random.default_rng(0)

    # G3.1 random walk
    rw = np.cumsum(rng.standard_normal(20000))
    ev = list(range(50, 19500, 37))
    r_rw = resting_fill_sim(rw, ev, delta=1.0, side=1, horizon=10)
    assert r_rw["markout_t"] < -2 and r_rw["markout_mean"] < -0.2, r_rw
    uncond = r_rw["markout_mean"] - r_rw["selection_cost"]
    assert abs(uncond - 1.0) < 0.3, uncond                        # ~+delta: the discount a naive backtest banks
    assert r_rw["selection_cost"] < -1.0, r_rw                    # ...and reality claws back MORE than all of it
    assert 0.2 < r_rw["fill_rate"] < 0.95

    # G3.2 the path-character ordering, all three measured
    ou = np.zeros(20000)
    for t in range(1, 20000):
        ou[t] = 0.9 * ou[t - 1] + rng.standard_normal()
    r_ou = resting_fill_sim(ou, ev, delta=1.0, side=1, horizon=10)
    dr = np.sign(rng.standard_normal(20000 // 200 + 1)).repeat(200)[:20000]
    mom = np.cumsum(0.35 * dr + rng.standard_normal(20000))
    r_mom = resting_fill_sim(mom, ev, delta=1.0, side=1, horizon=10)
    extra = lambda r, d=1.0: r["selection_cost"] + d              # extra adverse beyond the delta discount
    assert extra(r_mom) < extra(r_rw) < extra(r_ou), (extra(r_mom), extra(r_rw), extra(r_ou))
    assert extra(r_mom) < -1.5, extra(r_mom)                     # momentum: flow that hits you keeps going
    assert abs(r_ou["markout_mean"]) < 0.25, r_ou                # reversion: near-zero mark-out, NOT positive
    assert r_ou["markout_mean"] < 0.15                           # the refuted "flips positive" stays refuted

    # G3.3 depth: extra adverse shrinks (overshoot), fill rate collapses; scarcity refuses a thin number
    shallow = resting_fill_sim(rw, ev, delta=0.5, side=1, horizon=10)
    deep = resting_fill_sim(rw, ev, delta=3.0, side=1, horizon=10)
    assert extra(shallow, 0.5) < extra(deep, 3.0) < 0, (extra(shallow, 0.5), extra(deep, 3.0))
    assert deep["fill_rate"] < 0.6 * shallow["fill_rate"], (deep["fill_rate"], shallow["fill_rate"])
    scarce = resting_fill_sim(rw, ev, delta=25.0, side=1, horizon=10)
    assert "scarcity" in scarce["verdict"]

    # G4.1 sleeves + medians
    n = 6000
    # PERSISTENT drift regimes (blocks of 50): a sleeve that knows today's regime then genuinely predicts
    # tomorrow's increment. The first draft used iid per-step drift -- knowing step t's drift says NOTHING
    # about the t+lag increment the book actually trades, and the "good" sleeve measured t=-0.2. The lag is
    # doing its job; the fixture had no edge that survives one step. Kept as the reminder that an actionable
    # harness kills exactly the edges that only existed at lag zero.
    drift_sign = np.sign(rng.standard_normal(n // 50 + 1)).repeat(50)[:n]
    px = np.cumsum(0.25 * drift_sign + rng.standard_normal(n))
    good = drift_sign.copy()                                      # knows the (persistent) regime
    coin = np.sign(rng.standard_normal(n))
    book = PaperBook(lag=1, cost=0.01)
    book.add_sleeve("good", good).add_sleeve("coin_a", coin).add_sleeve("coin_b", np.sign(rng.standard_normal(n)))
    rep = book.run(px)
    assert rep["sleeves"]["good"]["t"] > 4, rep["sleeves"]["good"]
    assert abs(rep["sleeves"]["coin_a"]["t"]) < 3
    assert rep["across_sleeves"]["median_net"] < 0.5 * rep["across_sleeves"]["mean_net"], rep["across_sleeves"]
    # ^ one winner and two coins: the mean is dragged up, the median stays with the coins. The point of medians.

    # G4.2 the gate
    storm = np.zeros(n, bool)
    for s in range(500, n - 200, 1500):
        storm[s:s + 300] = True
    px2 = np.cumsum(np.where(storm, rng.normal(0, 3.0, n), 0.2 * drift_sign + rng.normal(0, 0.5, n)))
    b2 = PaperBook(lag=1, cost=0.01).add_sleeve("s", good)
    open_rep = b2.run(px2)
    gated_rep = b2.run(px2, gate_mask=~storm)
    assert gated_rep["sleeves"]["s"]["n_trades"] < open_rep["sleeves"]["s"]["n_trades"]
    assert gated_rep["sleeves"]["s"]["max_drawdown"] > open_rep["sleeves"]["s"]["max_drawdown"]  # less negative
    assert gated_rep["sleeves"]["s"]["t"] > open_rep["sleeves"]["s"]["t"]

    # G4.3 lag semantics + refusals
    try:
        PaperBook(lag=0)
        raise AssertionError("expected lag=0 refusal")
    except ValueError as e:
        assert "simultaneous is not past" in str(e)
    lag2 = PaperBook(lag=2, cost=0.01).add_sleeve("s", good).run(px)
    assert lag2["sleeves"]["s"]["t"] < rep["sleeves"]["good"]["t"]   # later entry, weaker capture -- latency costs
    for bad, needle in ((lambda: PaperBook().run(px), "no sleeves"),
                        (lambda: PaperBook().add_sleeve("a", good).add_sleeve("a", coin), "duplicate sleeve"),
                        (lambda: resting_fill_sim(rw, ev, delta=-1), "positive"),
                        (lambda: resting_fill_sim(rw, ev, delta=1, side=2), "+1")):
        try:
            bad()
            raise AssertionError("expected ValueError (%s)" % needle)
        except ValueError as e:
            assert needle in str(e), (needle, str(e))

    print("holographic_paperbook selftest OK (G3: unconditional mark-out is +delta BY CONSTRUCTION -- the "
          "discount a naive backtest banks -- while filled mark-out on a random walk is %.3f (t=%.1f): "
          "reality claws back more than the whole discount; extra adverse beyond the discount orders "
          "momentum %.2f << rw %.2f < reversion %.2f, and the first draft's 'reversion flips it positive' "
          "is REFUTED (ou mark-out %.3f, near zero); depth SHRINKS the per-fill extra (%.2f at 0.5 vs %.2f at "
          "3.0 -- overshoot, not deepening toxicity) while fills collapse; "
          "a 25-sigma level returns scarcity, not a number. G4: planted sleeve t=%.1f beside coins ~0 with "
          "the median staying with the coins; gating the storm lifts t %.1f->%.1f and maxdd %.1f->%.1f on "
          "fewer trades; lag=2 weaker than lag=1; lag=0 refused)"
          % (r_rw["markout_mean"], r_rw["markout_t"], extra(r_mom), extra(r_rw), extra(r_ou),
             r_ou["markout_mean"], extra(shallow, 0.5), extra(deep, 3.0),
             rep["sleeves"]["good"]["t"], open_rep["sleeves"]["s"]["t"], gated_rep["sleeves"]["s"]["t"],
             open_rep["sleeves"]["s"]["max_drawdown"], gated_rep["sleeves"]["s"]["max_drawdown"]))


if __name__ == "__main__":
    _selftest()

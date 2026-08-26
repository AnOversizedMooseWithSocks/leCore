"""holographic_forecastvalue.py -- D4: CALIBRATION IS NOT VALUE. Score a probabilistic forecast twice -- once
as a statistician (is p honest?) and once as a decision-maker (does acting on p pay?) -- and keep the two
verdicts separate, because they answer different questions and the campaign kept catching itself trading one
for the other.

THE TWO FACTS THIS MODULE MAKES UNAVOIDABLE
  1. A PERFECTLY CALIBRATED forecast can be WORTHLESS: forecast the base rate every time and your reliability
     error is ~0 while your decisions are exactly the always/never coin you started with. Calibration without
     RESOLUTION (the forecast actually separating the outcomes) buys nothing.
  2. A MISCALIBRATED forecast can be VALUABLE: any monotone distortion of an informative p wrecks reliability
     and leaves the RANKING -- hence the achievable decision value at the right threshold -- intact.
     Calibration is a REPAIR (a monotone remap fixes it); resolution is the SOURCE and no remap creates it.

The statistician's score is the Brier score with the Murphy decomposition, brier = reliability - resolution
+ uncertainty (binned): reliability is the miscalibration penalty, resolution is the separation earned,
uncertainty is the base rate's own entropy-like term that no forecast controls.

The decision-maker's score is realized: act when p >= tau, earn `payoff_act` on hits, lose `loss_act` on
false alarms, pay `cost` per action; swept over taus and reported net, against the honest baselines (never
act = 0, always act = the climatology decision). value_best is the sweep's max -- WHICH IS ITSELF A
SELECTION over taus, so the report says so and the honest quote is value_at(tau) for a tau chosen on other
data (or the ledger records the sweep).

NumPy + stdlib only, deterministic.
"""

import numpy as np


def calibration_vs_value(probs, outcomes, payoff_act=1.0, loss_act=1.0, cost=0.0,
                         taus=None, n_bins=10):
    """The double scoring. `probs` in [0,1], `outcomes` in {0,1}, aligned.

    Returns:
      brier, reliability, resolution, uncertainty   Murphy decomposition (binned, `n_bins` equal-width)
      bins                per-bin {p_mean, y_rate, n} -- the reliability diagram's numbers
      value_curve         [{tau, n_act, hits, false_alarms, net}] with net = hits*payoff_act
                          - false_alarms*loss_act - n_act*cost
      value_best          {tau, net} -- max over the sweep, WITH the selection warning in `note`
      baselines           {never: 0.0, always: net of acting on everything}
      verdicts            plain-language: calibration verdict AND value verdict, separately

    KEPT NEGATIVES: (1) value_best is an argmax over taus -- quoting it as achieved performance is a
    selection; pick tau elsewhere or put the sweep on the SelectionLedger. (2) the decomposition is BINNED:
    with few samples per bin, reliability is biased upward (noise reads as miscalibration) -- bins carry
    their n so a thin diagram is visibly thin. (3) payoffs here are per-event constants; state-dependent
    payoffs are net_of_costs' ground and the two compose rather than duplicate."""
    p = np.asarray(probs, float).ravel()
    y = np.asarray(outcomes, float).ravel()
    if p.size != y.size:
        raise ValueError("probs and outcomes must align (got %d vs %d)" % (p.size, y.size))
    if p.size < 20:
        raise ValueError("need at least 20 samples to say anything about calibration (got %d)" % p.size)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probs must lie in [0, 1] -- a score is not a probability until it is mapped there")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("outcomes must be 0/1")

    base = float(y.mean())
    brier = float(np.mean((p - y) ** 2))

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    bins = []
    reliability = 0.0
    resolution = 0.0
    for b in range(int(n_bins)):
        m = which == b
        nb = int(m.sum())
        if nb == 0:
            continue
        pm, yr = float(p[m].mean()), float(y[m].mean())
        bins.append({"p_mean": pm, "y_rate": yr, "n": nb})
        reliability += nb * (pm - yr) ** 2
        resolution += nb * (yr - base) ** 2
    reliability /= p.size
    resolution /= p.size
    uncertainty = float(base * (1.0 - base))

    if taus is None:
        taus = np.round(np.linspace(0.05, 0.95, 19), 2)
    curve = []
    for tau in taus:
        act = p >= float(tau)
        n_act = int(act.sum())
        hits = int(np.sum(y[act] == 1))
        fa = n_act - hits
        net = hits * float(payoff_act) - fa * float(loss_act) - n_act * float(cost)
        curve.append({"tau": float(tau), "n_act": n_act, "hits": hits, "false_alarms": fa,
                      "net": float(net)})
    best = max(curve, key=lambda r: r["net"])
    always_net = float(np.sum(y) * payoff_act - np.sum(1 - y) * loss_act - y.size * cost)

    # The calibration line: reliability under 10% of uncertainty. A first draft used 25% and waved through a
    # squash whose reliability was HALF the diagram's visible miscalibration -- a threshold lenient enough to
    # never fire is decoration. 10% still passes genuinely calibrated forecasts (measured 0.3% of uncertainty
    # on the Bernoulli fixture) with a wide margin.
    cal_ok = reliability < 0.10 * max(uncertainty, 1e-12)
    val_ok = best["net"] > max(0.0, always_net)
    verdicts = {
        "calibration": ("calibrated (reliability %.4f << uncertainty %.4f)" % (reliability, uncertainty))
                       if cal_ok else
                       ("MIScalibrated (reliability %.4f) -- a monotone remap may repair it; check whether "
                        "resolution (%.4f) survived, because THAT is what a remap cannot create"
                        % (reliability, resolution)),
        "value": ("acting at tau=%.2f nets %+.1f vs never=0 / always=%+.1f" %
                  (best["tau"], best["net"], always_net)) if val_ok else
                 ("NO decision value over the trivial policies (best net %+.1f, always %+.1f, never 0) -- "
                  "if calibration looks fine, this is the calibrated-but-worthless case: resolution %.4f "
                  "is the number that failed" % (best["net"], always_net, resolution)),
    }
    return {"brier": brier, "reliability": float(reliability), "resolution": float(resolution),
            "uncertainty": uncertainty, "bins": bins, "value_curve": curve,
            "value_best": {"tau": best["tau"], "net": best["net"],
                           "note": "argmax over the tau sweep -- a SELECTION; choose tau on other data or "
                                   "ledger the sweep"},
            "baselines": {"never": 0.0, "always": always_net}, "verdicts": verdicts}


def _selftest():
    """Contracts -- the module's two facts, planted and measured:
    1. CALIBRATED-BUT-WORTHLESS: p = base rate always. Reliability ~0, resolution ~0, and the best net does
       not beat the trivial policies. The value verdict names resolution as the number that failed.
    2. SHARP AND CALIBRATED: p drawn so outcomes follow Bernoulli(p). Reliability ~0, resolution high, value
       clearly positive.
    3. MISCALIBRATED-BUT-VALUABLE: the SAME informative p pushed through a monotone squash. Reliability
       blows up; resolution and the achievable value survive (within tolerance of the calibrated version) --
       calibration is a repair, resolution is the source, MEASURED.
    4. Murphy identity holds on the binned quantities; refusals name their reason.
    """
    rng = np.random.default_rng(0)
    n = 4000

    # (2) sharp + calibrated first (its p feeds case 3)
    p_true = np.clip(rng.beta(2, 2, n), 0.01, 0.99)
    y = (rng.random(n) < p_true).astype(float)
    r_good = calibration_vs_value(p_true, y, cost=0.05)
    assert r_good["reliability"] < 0.01, r_good["reliability"]
    assert r_good["resolution"] > 0.02, r_good["resolution"]
    assert r_good["value_best"]["net"] > r_good["baselines"]["always"], r_good["value_best"]
    assert r_good["value_best"]["net"] > 0

    # (1) calibrated but worthless: constant base rate on the same outcomes.
    r_flat = calibration_vs_value(np.full(n, y.mean()), y, cost=0.05)
    assert r_flat["reliability"] < 1e-3
    assert r_flat["resolution"] < 1e-6
    assert r_flat["value_best"]["net"] <= max(0.0, r_flat["baselines"]["always"]) + 1e-9
    assert "resolution" in r_flat["verdicts"]["value"], r_flat["verdicts"]["value"]

    # (3) miscalibrated but valuable: squash the informative p toward 0.5 monotonically.
    p_squash = 0.5 + 0.08 * np.tanh(3.0 * (p_true - 0.5))        # monotone; ranking preserved exactly
    r_squash = calibration_vs_value(p_squash, y, cost=0.05)
    assert r_squash["reliability"] > 5 * r_good["reliability"], (r_squash["reliability"], r_good["reliability"])
    # the achievable value survives the distortion (threshold moves; the ranking did not):
    assert r_squash["value_best"]["net"] > 0.8 * r_good["value_best"]["net"], \
        (r_squash["value_best"], r_good["value_best"])
    assert "remap" in r_squash["verdicts"]["calibration"]

    # (4) Murphy identity on the binned quantities: brier ~= reliability - resolution + uncertainty.
    #     Binning makes it approximate (within-bin variance is attributed to none of the three terms), so the
    #     tolerance is the honest one for 10 bins, not 1e-12 theater.
    for r in (r_good, r_flat, r_squash):
        lhs = r["brier"]
        rhs = r["reliability"] - r["resolution"] + r["uncertainty"]
        assert abs(lhs - rhs) < 0.02, (lhs, rhs)

    # refusals
    for bad, needle in ((lambda: calibration_vs_value([0.5] * 10, [0, 1] * 5), "at least 20"),
                        (lambda: calibration_vs_value([1.5] * 30, [1] * 30), "[0, 1]"),
                        (lambda: calibration_vs_value([0.5] * 30, [2] * 30), "0/1"),
                        (lambda: calibration_vs_value([0.5] * 30, [1] * 29), "align")):
        try:
            bad()
            raise AssertionError("expected ValueError (%s)" % needle)
        except ValueError as e:
            assert needle in str(e), (needle, str(e))

    print("holographic_forecastvalue selftest OK (calibrated-constant: reliability %.4f, resolution %.6f, "
          "best net %.1f vs always %.1f -- worthless and the verdict names resolution; sharp+calibrated nets "
          "%+.1f; the SAME forecast monotone-squashed: reliability %.4f (%.0fx worse) yet best net %+.1f -- "
          "%.0f%% of the calibrated value survives, calibration is a repair and resolution is the source; "
          "Murphy identity within binning tolerance)"
          % (r_flat["reliability"], r_flat["resolution"], r_flat["value_best"]["net"],
             r_flat["baselines"]["always"], r_good["value_best"]["net"], r_squash["reliability"],
             r_squash["reliability"] / max(r_good["reliability"], 1e-12),
             r_squash["value_best"]["net"],
             100 * r_squash["value_best"]["net"] / r_good["value_best"]["net"]))


if __name__ == "__main__":
    _selftest()

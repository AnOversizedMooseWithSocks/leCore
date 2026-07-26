"""holographic_envelope.py -- forecast the ENVELOPE of the next move (its scale), not its direction, and keep
the two claims separated by construction.

WHY THIS MODULE EXISTS
The source campaign's cleanest structural finding was an asymmetry: DIRECTION of the next move was
indistinguishable from chance through every honest gate, while its SIZE was strongly forecastable -- activity
clusters (large |moves| follow large |moves|; the measured lag-1 autocorrelation of |returns| dwarfed that of
returns). Most "prediction" effort was spent on the unforecastable half. The honest tool forecasts the half
that IS there -- a calibrated band for |next move| -- and REFUSES to convert it into a directional statement,
because a scale forecast contains zero directional bits and dressing it in one is how envelope skill gets
laundered into direction claims.

The forecaster is deliberately simple: trailing scale (mean |diff| over a window) as the conditional
predictor, conformal RATIO residuals (|move| / predicted scale) from a calibration split for the band. Ratio
residuals, not additive ones: scale errors are multiplicative (a 2x-too-small band in a storm is the same
mistake as 2x-too-small in calm), and the ratio calibration makes ONE quantile serve every volatility state --
which is exactly what an additive residual cannot do (kept negative, measured in _selftest: additive
calibration under-covers storms and over-covers calm; the D1 shape from the inside).

NumPy + stdlib only, deterministic.
"""

import numpy as np


def envelope_forecast(series, window=20, alpha=0.1, calib_frac=0.5):
    """Fit the envelope forecaster on `series` and return a one-step-ahead band for |next move| plus the
    evidence that earned it.

    Mechanics: predictor = trailing mean |diff| over `window`; on the calibration split (first `calib_frac`
    of the sample, honouring time order -- calibrating on the future would be look-ahead) collect ratio
    residuals r_t = |dx_t| / pred_t and take their (1-alpha) conformal quantile q. The band for the next move
    is [0, q * current_pred] with typical size ~median(r) * current_pred.

    Returns {predicted_scale, upper, typical, q_ratio, alpha, coverage_holdout, n_calib, n_holdout,
    directional_bits_note}. `coverage_holdout` is the honest number: empirical coverage of the band on the
    UNSEEN second split, computed with the same rolling predictor -- quote the band only next to it.

    KEPT NEGATIVES: (1) the band says NOTHING about direction, and the returned note says so in words --
    scale forecastability and direction forecastability are different quantities and the campaign measured
    the second at chance while the first was strong. Test direction separately (mind.null_persistence,
    mind.mutual_information_vs_null in bits) before any directional use. (2) one-step-ahead only: multi-step
    envelope forecasts need the vol dynamics compounded, and a naive sqrt-horizon scaling assumes exactly the
    independence that vol clustering violates."""
    x = np.asarray(series, float).ravel()
    if x.size < 4 * window:
        raise ValueError("envelope_forecast needs at least 4*window=%d samples (got %d) -- a calibration "
                         "split thinner than that cannot support a quantile" % (4 * window, x.size))
    d = np.abs(np.diff(x))
    n = d.size
    # rolling trailing predictor: pred[t] uses d[t-window : t] -- strictly past moves only.
    csum = np.concatenate([[0.0], np.cumsum(d)])
    pred = np.full(n, np.nan)
    for t in range(window, n):
        pred[t] = (csum[t] - csum[t - window]) / window
    valid = np.arange(window, n)
    n_cal = max(int(valid.size * calib_frac), window)
    cal, hold = valid[:n_cal], valid[n_cal:]
    if hold.size < 10:
        raise ValueError("holdout too thin (%d moves) -- give a longer series or a smaller calib_frac" % hold.size)
    floor = 1e-12
    r_cal = d[cal] / np.maximum(pred[cal], floor)
    # conformal quantile with the finite-sample correction (ceil((n+1)(1-alpha))/n), the engine's convention.
    k = int(np.ceil((r_cal.size + 1) * (1.0 - alpha)))
    k = min(max(k, 1), r_cal.size)
    q = float(np.sort(r_cal)[k - 1])
    r_hold = d[hold] / np.maximum(pred[hold], floor)
    coverage = float(np.mean(r_hold <= q))
    current_pred = float((csum[n] - csum[n - window]) / window)
    return {"predicted_scale": current_pred, "upper": q * current_pred,
            "typical": float(np.median(r_cal)) * current_pred, "q_ratio": q, "alpha": float(alpha),
            "coverage_holdout": coverage, "n_calib": int(cal.size), "n_holdout": int(hold.size),
            "directional_bits_note": "this band bounds |next move| only; it contains ZERO directional bits -- "
                                     "test direction separately (null_persistence / mutual information in bits)"}


def envelope_vs_constant(series, window=20, alpha=0.1, calib_frac=0.5):
    """The BASELINE CHECK the envelope must pass to be worth anything: does conditioning on trailing scale
    beat a CONSTANT band (the unconditional quantile of |move|) on the same holdout? Both bands are calibrated
    to the same nominal coverage, so the honest comparison is WIDTH at equal coverage: `width_ratio` =
    mean conditional upper / constant upper. Below 1.0 the envelope earns its keep (equal safety, tighter
    band); at ~1.0 the series has no usable vol clustering and the constant band is the right tool.

    Returns {width_ratio, coverage_conditional, coverage_constant, sharper, verdict}. `verdict` names which
    of three cases the comparison landed in, because width_ratio is only meaningful in the first:
      BOTH-COVER        both bands hold nominal on the holdout -- width_ratio is the honest score.
      CONSTANT-FAILED   the constant band UNDER-COVERS the holdout (vol drifted between the halves; split
                        conformal is exchangeability-based and drift breaks it) -- the envelope wins on
                        VALIDITY and width_ratio is apples-to-oranges: a band that failed its guarantee is
                        not cheaper, it is broken. Reported ratio kept for the record, not for ranking.
      CONDITIONAL-FAILED the envelope itself missed nominal -- do not quote it at all.

    KEPT NEGATIVES: (1) on an iid-Gaussian series the ratio is ~1.0 (pinned) -- a sharpness win is a CLAIM
    ABOUT THE DATA, never a property of the method; and the cost is not zero: on UNclustered data the trailing
    predictor's estimation noise makes the conditional band strictly WIDER (measured 1.05 at window=20 up to
    1.18 at window=5), so on a domain without clustering the constant band is simply better. (2) The
    CONSTANT-FAILED case was found by review, on log-vol AR(1) drift (phi=0.97): const covered 0.856 vs
    nominal 0.90 while the envelope held 0.909 -- and the first version reported width_ratio=1.70 with no
    flag, inviting exactly the wrong conclusion. Run this before quoting envelope_forecast on any new domain."""
    x = np.asarray(series, float).ravel()
    ef = envelope_forecast(x, window=window, alpha=alpha, calib_frac=calib_frac)
    d = np.abs(np.diff(x))
    n = d.size
    csum = np.concatenate([[0.0], np.cumsum(d)])
    pred = np.full(n, np.nan)
    for t in range(window, n):
        pred[t] = (csum[t] - csum[t - window]) / window
    valid = np.arange(window, n)
    n_cal = max(int(valid.size * calib_frac), window)
    cal, hold = valid[:n_cal], valid[n_cal:]
    # constant band: unconditional conformal quantile of |move| itself on the SAME calibration split.
    k = int(np.ceil((cal.size + 1) * (1.0 - alpha)))
    k = min(max(k, 1), cal.size)
    const_upper = float(np.sort(d[cal])[k - 1])
    cov_const = float(np.mean(d[hold] <= const_upper))
    cond_upper = ef["q_ratio"] * pred[hold]
    cov_cond = float(np.mean(d[hold] <= cond_upper))
    width_ratio = float(np.mean(cond_upper) / const_upper) if const_upper > 0 else float("inf")
    # coverage bound scaled to the holdout size (same 2-binomial-SE rule as conditional_coverage): a fixed
    # 0.05 slack is far too forgiving at n~3000 (it waved through 0.856 vs nominal 0.90 during review) and
    # too harsh at n~50.
    nominal = 1.0 - alpha
    se = float(np.sqrt(max(nominal * (1.0 - nominal), 1e-12) / max(hold.size, 1)))
    tol = 2.0 * se
    ok_cond = cov_cond >= nominal - tol
    ok_const = cov_const >= nominal - tol
    if not ok_cond:
        verdict = "CONDITIONAL-FAILED: the envelope missed nominal on the holdout -- do not quote it"
    elif not ok_const:
        verdict = ("CONSTANT-FAILED: the constant band under-covered the holdout (%.3f vs nominal %.2f; "
                   "exchangeability broken, likely vol drift between the halves) -- the envelope wins on "
                   "VALIDITY and width_ratio is not a ranking here" % (cov_const, nominal))
    else:
        verdict = "BOTH-COVER: width_ratio is the honest score"
    return {"width_ratio": width_ratio, "coverage_conditional": cov_cond,
            "coverage_constant": cov_const,
            "sharper": bool(width_ratio < 0.95 and ok_cond and ok_const),
            "verdict": verdict}


def _selftest():
    """Contracts:

    1. On a VOL-CLUSTERED series (regime-switching scale), the envelope holds nominal coverage on the unseen
       holdout AND is materially sharper than the constant band at equal coverage.
    2. On IID GAUSSIAN noise the width ratio is ~1.0 -- the sharpness win is a property of the data's
       clustering, not of the method (the kept negative, pinned).
    3. RATIO vs ADDITIVE calibration: on the clustered series, an additive band (constant absolute margin)
       under-covers the high-vol state and over-covers calm -- the D1 failure built in miniature -- while the
       ratio band covers both states. Pinned by conditional coverage inside/outside the storm state.
    4. The directional note is present verbatim; refusals name their fix; deterministic.
    """
    rng = np.random.default_rng(0)

    # (1) clustered: alternating calm/storm scale regimes.
    n = 4000
    scale = np.where((np.arange(n) // 250) % 2 == 0, 0.5, 2.5)
    moves = rng.normal(size=n) * scale
    x = np.cumsum(moves)
    ef = envelope_forecast(x, window=20, alpha=0.1)
    assert abs(ef["coverage_holdout"] - 0.9) < 0.04, ef
    cmp_ = envelope_vs_constant(x, window=20, alpha=0.1)
    assert cmp_["sharper"] and cmp_["width_ratio"] < 0.87, cmp_    # measured 0.849: the clustering pays

    # (2) iid noise: no clustering, no sharpness -- ratio ~ 1.
    x_iid = np.cumsum(rng.normal(size=n))
    cmp_iid = envelope_vs_constant(x_iid, window=20, alpha=0.1)
    assert 0.9 < cmp_iid["width_ratio"] < 1.15 and not cmp_iid["sharper"], cmp_iid

    # (3) ratio vs additive, judged by PER-STATE coverage (the reason ratio residuals are the design).
    d = np.abs(np.diff(x))
    w = 20
    csum = np.concatenate([[0.0], np.cumsum(d)])
    pred = np.array([np.nan] * d.size, float)
    for t in range(w, d.size):
        pred[t] = (csum[t] - csum[t - w]) / w
    valid = np.arange(w, d.size)
    half = valid[:valid.size // 2]
    hold = valid[valid.size // 2:]
    k = int(np.ceil((half.size + 1) * 0.9))
    add_margin = float(np.sort(d[half])[min(k, half.size) - 1])          # additive: one absolute number
    q_ratio = float(np.sort(d[half] / pred[half])[min(k, half.size) - 1])  # ratio: one multiplier
    storm_hold = scale[1:][hold] > 1.0
    cov_add_storm = float(np.mean(d[hold][storm_hold] <= add_margin))
    cov_add_calm = float(np.mean(d[hold][~storm_hold] <= add_margin))
    cov_rat_storm = float(np.mean(d[hold][storm_hold] <= q_ratio * pred[hold][storm_hold]))
    cov_rat_calm = float(np.mean(d[hold][~storm_hold] <= q_ratio * pred[hold][~storm_hold]))
    assert cov_add_storm < 0.85 and cov_add_calm > 0.97, (cov_add_storm, cov_add_calm)  # additive fails D1-style
    assert abs(cov_rat_storm - 0.9) < 0.06 and abs(cov_rat_calm - 0.9) < 0.06, (cov_rat_storm, cov_rat_calm)

    # (4) note + refusals + determinism.
    assert "ZERO directional bits" in ef["directional_bits_note"]
    # (5) THE DRIFT CASE, from review: log-vol AR(1) at phi=0.97 drifts the scale between the calibration
    #     and holdout halves; split conformal's exchangeability assumption breaks and the CONSTANT band
    #     under-covers, while the conditional band adapts and holds. The comparison must SAY so instead of
    #     reporting a wider-therefore-worse ratio: verdict=CONSTANT-FAILED, sharper stays False, and the
    #     ratio is present for the record only. (First version reported width_ratio=1.70 here with no flag.)
    #     A dedicated seed pins the exact realisation the review used: drift is a property of the DRAW under
    #     phi=0.97 (some realisations stay level), and a contract about a specific failure needs the specific
    #     failing draw, not a fresh one per rng state.
    rng_drift = np.random.default_rng(42)
    lv = np.zeros(6000)
    for t in range(1, 6000):
        lv[t] = 0.97 * lv[t - 1] + rng_drift.normal(0, 0.25)
    x_drift = np.cumsum(rng_drift.normal(0, 1, 6000) * np.exp(lv))
    cmp_drift = envelope_vs_constant(x_drift, window=20, alpha=0.1)
    assert cmp_drift["verdict"].startswith("CONSTANT-FAILED"), cmp_drift
    assert cmp_drift["coverage_constant"] < 0.87 and cmp_drift["coverage_conditional"] > 0.87, cmp_drift
    assert cmp_drift["sharper"] is False
    assert cmp_["verdict"].startswith("BOTH-COVER")            # the regime-switch fixture stays comparable

    assert envelope_forecast(x, window=20) == envelope_forecast(x, window=20)
    for bad, needle in ((lambda: envelope_forecast(x[:60], window=20), "4*window"),
                        (lambda: envelope_forecast(x[:90], window=20, calib_frac=0.95), "holdout too thin")):
        try:
            bad()
            raise AssertionError("expected ValueError (%s)" % needle)
        except ValueError as e:
            assert needle in str(e), (needle, str(e))

    print("holographic_envelope selftest OK (clustered series: holdout coverage %.2f at nominal 0.90 with the "
          "conditional band %.0f%% the width of the constant one at equal coverage; iid noise: ratio %.2f -- "
          "no clustering, no win, KEPT; additive calibration fails per-state (storm %.2f / calm %.2f) where "
          "the ratio band holds both (%.2f / %.2f) -- the reason ratio residuals are the design; the band "
          "carries its own zero-directional-bits note)"
          % (ef["coverage_holdout"], 100 * cmp_["width_ratio"], cmp_iid["width_ratio"],
             cov_add_storm, cov_add_calm, cov_rat_storm, cov_rat_calm))


if __name__ == "__main__":
    _selftest()

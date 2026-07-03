"""holographic_conformal.py -- CALIBRATED FORECAST CONFIDENCE: distribution-free prediction intervals that wrap
ANY producer, know when to ABSTAIN, adapt to drift over time, and score their own quality. The forecasting-side
twin of RecallNull -- RecallNull calibrates RECOGNITION ("is this cue a real match or noise?"); this calibrates
a FORECAST ("how wide is my interval, and should I trust it?").

WHY THIS EXISTS (Forecasting & Prediction backlog, F1 / F2 / F8)
---------------------------------------------------------------
The engine can already PRODUCE the next vector four ways (Propagator / reservoir / predictive / generate), but it
had no *calibrated confidence* on a forecast and no single scoring instrument. Conformal prediction is that
layer, and it fits the constitution almost too well: run a predictor on held-out data where the answer is known,
collect how wrong it was each time (the residuals), sort them, and take the (1-alpha) quantile. Any new forecast
then gets an interval "point +/- that error", with a guarantee the truth lands inside about 1-alpha of the time
-- distribution-free, finite-sample, PURE NUMPY, NO LEARNED WEIGHTS. It wraps any producer as a black box, so ONE
confidence layer serves all data and all producers.

PROBE-FIRST NOTE: holographic_reasoning.py already has a SCALAR split-conformal `ConformalPredictor` (leOS's
reflex gate). This module GENERALIZES that same finite-sample discipline rather than duplicating it, adding the
four things it lacks: (a) a VSA-NATIVE nonconformity score, `1 - cosine(pred, actual)`, so a VECTOR forecast is
scored in the engine's own metric; (b) wrap-any-producer + ABSTAIN when the interval is wider than the caller
trusts; (c) TEMPORAL conformal (ACI + weighting) for time series, which break the exchangeability split
conformal assumes; (d) a coverage instrument and proper scores (pinball / CRPS) so an interval can be *trusted*,
not just produced.

KEPT NEGATIVES (loud -- forecasting is where over-claiming is easiest):
* Calibrated != correct. The interval COVERS at 1-alpha; it does NOT make the point forecast accurate. A useless
  predictor gets honest-but-wide intervals -- the width is the signal, not a substitute for a good producer.
* Coverage is MARGINAL, not conditional (Barber, Candes, Ramdas, Tibshirani 2021): 1-alpha holds on AVERAGE over
  inputs, never guaranteed for one specific input. We promise the PROCEDURE is 1-alpha covered, never a forecast.
* Exchangeability breaks on time series, so plain split conformal silently under-covers under drift -- that is
  why ACI/weighting (F2) exist. And under a FUNDAMENTAL regime change (~0% overlap between the calibration past
  and the present) NO conformal variant holds; the correct behaviour is to ABSTAIN and flag drift (that is the
  honest earthquake boundary), not to forecast confidently.
* Vanilla split conformal gives CONSTANT-width intervals (over-covers calm regions, under-covers volatile ones);
  the normalized/CQR fix is a follow-on, noted not built here.

Real basis: Vovk, Gammerman, Shafer (2005); Papadopoulos et al. (2002); Lei et al. (2018) -- split conformal.
Gibbs & Candes (2021) -- Adaptive Conformal Inference. Barber et al. (2023) -- beyond exchangeability (weighting).
Gneiting & Raftery (2007) -- proper scoring rules (CRPS, pinball). Deterministic; NumPy + stdlib only.
"""
from collections import deque

import numpy as np

from holographic_ai import cosine


# --- nonconformity scores (how wrong was a forecast) -----------------------------------------------------------

def nonconformity_scalar(pred, actual):
    """Scalar residual |pred - actual| -- the score for a numeric forecast (matches the reasoning-module core)."""
    return float(abs(float(pred) - float(actual)))


def nonconformity_vector(pred, actual):
    """VSA-NATIVE residual 1 - cosine(pred, actual) -- how wrong a VECTOR forecast is, in the engine's OWN metric.
    0 when the forecast points exactly at the truth, up to 2 when it points the opposite way. This is what makes
    the whole layer speak the substrate's language instead of an external one."""
    return float(1.0 - cosine(np.asarray(pred, float), np.asarray(actual, float)))


def conformal_quantile(residuals, alpha):
    """The split-conformal half-width: the (1-alpha) quantile of the residuals with the FINITE-SAMPLE correction.
    Take the ceil((n+1)(1-alpha))-th smallest residual -- the order statistic that guarantees >= 1-alpha coverage
    on exchangeable new data (Vovk et al.). Same rule as holographic_reasoning's scalar ConformalPredictor,
    generalized here to feed both the scalar and the vector paths."""
    r = np.sort(np.abs(np.asarray(residuals, float)))
    n = len(r)
    if n == 0:
        return float("inf")                                      # no calibration data -> infinitely cautious
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))                # the finite-sample order statistic
    return float(r[min(rank - 1, n - 1)])                        # clamp: alpha smaller than 1/(n+1) -> the max


# --- the split-conformal forecaster (F1) -----------------------------------------------------------------------

class ConformalForecaster:
    """Wraps ANY predictor's residuals into a calibrated interval, and abstains when the interval is too wide to
    trust. `kind='scalar'` scores by |error| (numeric forecast); `kind='vector'` scores by 1 - cosine (a vector
    forecast -- the interval is then a cosine-RADIUS, i.e. the prediction SET of all vectors within that radius).
    `abstain_width` (optional) is the caller's trust tolerance: a wider interval returns abstain=True."""

    def __init__(self, alpha=0.1, kind="scalar", abstain_width=None):
        self.alpha = float(alpha)                                # miscoverage target: 0.1 -> 90% coverage
        self.kind = kind
        self.abstain_width = abstain_width
        self.q = None                                            # calibrated half-width / cosine-radius

    def _residual(self, pred, actual):
        return nonconformity_scalar(pred, actual) if self.kind == "scalar" else nonconformity_vector(pred, actual)

    def calibrate(self, preds, actuals):
        """Fit on a calibration set the predictor did NOT learn from: score each (pred, actual), take the
        conformal quantile. `preds`/`actuals` are paired scalars or paired vectors."""
        residuals = [self._residual(p, a) for p, a in zip(preds, actuals)]
        self.q = conformal_quantile(residuals, self.alpha)
        return self.q

    def calibrate_residuals(self, residuals):
        """Fit directly from precomputed residuals (when you already have the errors)."""
        self.q = conformal_quantile(residuals, self.alpha)
        return self.q

    def predict(self, point):
        """Return the calibrated forecast for a point prediction: a dict with the point, the half-width, the
        interval (scalar) or cosine-radius (vector), the target coverage, and an ABSTAIN flag."""
        if self.q is None:
            raise RuntimeError("call calibrate() first")
        out = {"point": point, "half_width": self.q, "coverage": 1.0 - self.alpha,
               "abstain": (self.abstain_width is not None and self.q > self.abstain_width)}
        if self.kind == "scalar":
            out["interval"] = (float(point) - self.q, float(point) + self.q)
        else:
            out["cosine_radius"] = self.q                        # the set is {v : 1 - cosine(point, v) <= q}
        return out

    def covers(self, point, actual):
        """Did the calibrated interval/set around `point` contain `actual`? (The check coverage is measured with.)"""
        if self.q is None:
            raise RuntimeError("call calibrate() first")
        return self._residual(point, actual) <= self.q


def wrap(producer, calib_inputs, calib_targets, alpha=0.1, kind="scalar", abstain_width=None):
    """Turn any `producer(input) -> point forecast` into a producer that ALSO emits a calibrated interval and an
    abstain flag. Runs the producer on a held-out calibration set, calibrates on the residuals, and returns a
    callable `wrapped(input) -> the predict() dict`. One layer, any producer -- the model-agnostic promise."""
    cf = ConformalForecaster(alpha=alpha, kind=kind, abstain_width=abstain_width)
    preds = [producer(x) for x in calib_inputs]
    cf.calibrate(preds, calib_targets)

    def wrapped(x):
        out = cf.predict(producer(x))
        return out
    wrapped.forecaster = cf                                      # expose the fitted forecaster (its q, alpha)
    return wrapped


# --- temporal conformal (F2): the exchangeability fixes for time series ----------------------------------------

class AdaptiveConformal:
    """Adaptive Conformal Inference (Gibbs & Candes 2021): a feedback rule that keeps LONG-RUN coverage at
    1-alpha even under drift, without modelling the drift. Each step: form the interval at the CURRENT effective
    alpha_t from recent residuals, observe whether it covered, then nudge alpha_t -- widen after a miss, narrow
    after a hit. `gamma` is the step size; `window` bounds how much recent history feeds the quantile.

    KEPT NEGATIVE: ACI reacts AFTER errors and can only widen/narrow -- it cannot shift a biased point forecast's
    CENTER, and under a ~0%-overlap regime change no feedback rule recovers coverage (abstain + flag drift)."""

    def __init__(self, alpha=0.1, gamma=0.05, window=200):
        self.target = float(alpha)                               # the coverage we want to hold long-run
        self.alpha_t = float(alpha)                              # the effective, adapting miscoverage
        self.gamma = float(gamma)
        self.residuals = deque(maxlen=window)                    # recent residuals -> the running quantile
        self.history = []                                        # (q, covered) per step, for the coverage report

    def q(self):
        """Current half-width: the conformal quantile of recent residuals at the ADAPTED level (clamped to [0,1])."""
        a = min(max(self.alpha_t, 0.0), 1.0)
        return conformal_quantile(list(self.residuals), a) if self.residuals else float("inf")

    def step(self, residual):
        """Process one observed residual: score coverage with the current q, then apply the ACI update. Returns
        (q_used, covered)."""
        q = self.q()
        covered = residual <= q
        err = 0.0 if covered else 1.0
        # ACI: alpha_{t+1} = alpha_t + gamma*(target - err). Miss (err=1) -> alpha down -> wider next; hit -> narrower.
        self.alpha_t = self.alpha_t + self.gamma * (self.target - err)
        self.residuals.append(float(residual))
        self.history.append((q, covered))
        return q, covered

    def realized_coverage(self):
        """The empirical coverage seen so far -- should track 1-target under ACI even as the series drifts."""
        if not self.history:
            return float("nan")
        return float(np.mean([c for _, c in self.history]))


def weighted_conformal_quantile(residuals, alpha, halflife=50):
    """Weighted split conformal (Barber et al. 2023): exponentially decay the calibration residuals toward the
    MOST RECENT ones so the quantile tracks drift. `halflife` sets how fast old residuals fade. A weighted
    quantile at level 1-alpha over the residual distribution."""
    r = np.asarray(residuals, float)
    n = len(r)
    if n == 0:
        return float("inf")
    ages = np.arange(n)[::-1]                                    # 0 = most recent (assumes residuals in time order)
    w = 0.5 ** (ages / float(halflife))                         # exponential decay by half-life
    order = np.argsort(np.abs(r))
    rs = np.abs(r)[order]; ws = w[order]
    cw = np.cumsum(ws) / np.sum(ws)                             # weighted CDF
    idx = int(np.searchsorted(cw, 1.0 - alpha))                # the (1-alpha) weighted quantile
    return float(rs[min(idx, n - 1)])


# --- coverage instrument (F8) ----------------------------------------------------------------------------------

def empirical_coverage(covered_flags):
    """The fraction of held-out cases the interval actually covered -- the single number the whole guarantee is
    about. Should sit near 1-alpha."""
    c = np.asarray(covered_flags, dtype=float)
    return float(c.mean()) if len(c) else float("nan")


def coverage_report(residuals_calib, residuals_test, alphas=(0.01, 0.05, 0.1, 0.2)):
    """Split residuals into a calibration half and a test half, and for each alpha check that the empirical
    coverage on the test half tracks the nominal 1-alpha. The forecasting twin of the mind's calibration_report.
    Returns a list of {alpha, nominal, empirical} rows."""
    rc = np.abs(np.asarray(residuals_calib, float))
    rt = np.abs(np.asarray(residuals_test, float))
    rows = []
    for a in alphas:
        q = conformal_quantile(rc, a)
        cov = empirical_coverage(rt <= q)
        rows.append({"alpha": a, "nominal": 1.0 - a, "empirical": cov})
    return rows


# --- proper scoring rules (F8): coverage says the interval is WIDE ENOUGH; scoring says the forecast is GOOD ----

def pinball_loss(pred, actual, q):
    """The quantile (pinball) loss at quantile level q -- the proper score for an interval endpoint. Lower is
    better; it penalises being on the wrong side asymmetrically by q."""
    pred = float(pred); actual = float(actual); q = float(q)
    d = actual - pred
    return float(max(q * d, (q - 1.0) * d))


def crps_sample(samples, actual):
    """CRPS (Continuous Ranked Probability Score) from an empirical predictive SAMPLE (Gneiting & Raftery 2007):
    CRPS = E|X - y| - 0.5 * E|X - X'|, estimated from the samples. Lower is better; it rewards a forecast
    distribution that is both accurate AND sharp, so it discriminates a good producer from a lucky-but-vague one.
    This is exactly the score for the analog-recall forecaster's empirical successor distribution."""
    x = np.asarray(samples, float)
    m = len(x)
    if m == 0:
        return float("nan")
    term1 = np.mean(np.abs(x - float(actual)))                 # accuracy: how far samples sit from the truth
    term2 = 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))     # spread: half the mean pairwise distance
    return float(term1 - term2)


def sharpness(half_widths):
    """Mean interval half-width -- the calibration-and-sharpness pair's second half. Among forecasters that hold
    coverage, the SHARPER (smaller) one is better; reported alongside coverage so neither is gamed alone."""
    h = np.asarray(half_widths, float)
    return float(h.mean()) if len(h) else float("nan")


def _selftest():
    """Split conformal hits nominal coverage across alphas (scalar AND vector); the wrapper abstains on a wide
    interval; ACI holds coverage on a DRIFTING series where plain split conformal drifts off it; CRPS ranks a
    better producer strictly below a worse one; coverage_report tracks nominal."""
    rng = np.random.default_rng(0)

    # (1) scalar split conformal hits nominal coverage on exchangeable data
    truth = rng.standard_normal(4000)
    pred = truth + rng.standard_normal(4000) * 0.5             # a predictor with Gaussian error
    resid = np.abs(pred - truth)
    rep = coverage_report(resid[:2000], resid[2000:], alphas=(0.05, 0.1, 0.2))
    for row in rep:
        assert abs(row["empirical"] - row["nominal"]) < 0.03, row   # empirical tracks nominal within 3%

    # (2) VSA-native vector conformal: a noisy vector forecast, coverage at 1-alpha
    dim = 256
    base = rng.standard_normal((3000, dim))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    noisy = base + 0.3 * rng.standard_normal((3000, dim))     # the "forecast" is a noisy version of the truth
    cf = ConformalForecaster(alpha=0.1, kind="vector")
    cf.calibrate(list(noisy[:1500]), list(base[:1500]))
    covered = [cf.covers(noisy[i], base[i]) for i in range(1500, 3000)]
    cov = empirical_coverage(covered)
    assert abs(cov - 0.9) < 0.03, cov                         # ~90% of vector forecasts inside the cosine radius

    # (3) wrap + abstain: a tight tolerance on a noisy producer abstains; a loose one does not
    producer = lambda x: x + rng.standard_normal() * 2.0      # a noisy scalar producer
    xs = rng.standard_normal(500); ys = xs.copy()
    tight = wrap(producer, xs, ys, alpha=0.1, kind="scalar", abstain_width=0.5)
    loose = wrap(producer, xs, ys, alpha=0.1, kind="scalar", abstain_width=100.0)
    assert tight(0.0)["abstain"] is True                      # interval wider than 0.5 -> abstain
    assert loose(0.0)["abstain"] is False

    # (4) ACI holds long-run coverage on a DRIFTING series where fixed split conformal drifts off
    n = 3000
    scale = np.linspace(0.5, 4.0, n)                          # error variance GROWS over time (drift)
    stream = np.abs(rng.standard_normal(n) * scale)          # residual stream
    fixed_q = conformal_quantile(stream[:300], 0.1)          # plain CP: calibrate once on the calm early part
    fixed_cov = empirical_coverage(stream[300:] <= fixed_q)  # ... then never adapt
    aci = AdaptiveConformal(alpha=0.1, gamma=0.05, window=300)
    for r in stream:
        aci.step(r)
    aci_cov = aci.realized_coverage()
    assert aci_cov > fixed_cov                                # ACI tracks the drift; fixed CP under-covers
    assert abs(aci_cov - 0.9) < 0.05                          # and holds near the 90% target

    # (5) CRPS discriminates: a sharp-and-accurate forecast scores strictly better than a vague one
    good = crps_sample(rng.standard_normal(500) * 0.3 + 1.0, 1.0)    # tight around the truth
    bad = crps_sample(rng.standard_normal(500) * 3.0 + 1.0, 1.0)     # vague around the truth
    assert good < bad

    print("holographic_conformal selftest OK: scalar coverage tracks nominal (within 3 pct); VSA-native vector "
          "conformal covers %.2f at 90%%; wrap abstains on a wide interval; ACI holds %.2f coverage on a drifting "
          "series where fixed CP drifts to %.2f; CRPS ranks sharp<vague (%.3f<%.3f)"
          % (cov, aci_cov, fixed_cov, good, bad))


if __name__ == "__main__":
    _selftest()

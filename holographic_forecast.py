"""holographic_forecast.py -- the `forecast(data)` ROUTER: one entry point that, given a time series, picks the
producer that fits it best, wraps the choice in a calibrated interval, and abstains when even the best producer
is not confident. The single door the brief asks for.

WHY THIS EXISTS (Forecasting & Prediction backlog, F3)
------------------------------------------------------
The engine has several ways to produce the next value; a user should not have to know which. This routes by a
MEASURED criterion instead of a guess: fit each cheap producer on a training split, evaluate each on a held-out
calibration split, and choose the one with the smaller calibration error -- then calibrate a conformal interval
on THAT producer's residuals. A misroute fails SAFE: a wrong producer shows up as a wide interval / abstention,
never a confident wrong answer (holographic_conformal makes that the default).

The two producers wired here are the linear AR predictor (near-linear structure -- the Propagator's shape) and
the analog forecaster (recurrence/nonlinearity -- holographic_analog). The policy table is deliberately small
and readable; adding the reservoir/predictive/generate producers is more rows, not new machinery.

KEPT NEGATIVE (loud): routing "pick the tighter calibration" is honest only because the interval is calibrated
either way -- the router cannot make a bad producer good, it can only prefer the less-bad one and report an
honest (possibly wide) interval. On data unlike anything (no analog, no linear fit) the honest output is a wide
interval or abstention. Deterministic; NumPy + stdlib.
"""
import numpy as np

from holographic_analog import AnalogForecaster, delay_embed
from holographic_conformal import ConformalForecaster


def linear_ar_fit(contexts, successors):
    """Least-squares AR predictor: fit successor ~ contexts (with a bias column). Returns a predict(context) ->
    value function. This is the near-linear/Koopman producer's cheap cousin -- a linear map from a window to the
    next value."""
    X = np.asarray(contexts, float)
    y = np.asarray(successors, float)
    A = np.concatenate([X, np.ones((len(X), 1))], axis=1)      # bias column
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)

    def predict(context):
        c = np.concatenate([np.asarray(context, float), [1.0]])
        return float(c @ coef)
    return predict


class RoutedForecaster:
    """The result of routing: the chosen producer's name, a point-forecast function, and the fitted conformal
    forecaster that turns a point into a calibrated interval. `.predict(context)` returns the conformal dict plus
    which producer was chosen."""

    def __init__(self, producer_name, predict_fn, conformal):
        self.producer = producer_name
        self._predict = predict_fn
        self.conformal = conformal

    def predict(self, context):
        out = self.conformal.predict(self._predict(context))
        out["producer"] = self.producer
        return out


def route_and_forecast(series, d=20, alpha=0.1, abstain_width=None, seed=0):
    """Route a 1-D series to the producer that calibrates tightest, and return a RoutedForecaster. Splits the
    delay-embedded data into train (fit) / calibration (choose + conformal). The winner is the producer with the
    lower calibration MAE; its residuals set the conformal interval."""
    contexts, successors = delay_embed(np.asarray(series, float), d)
    n = len(contexts)
    if n < 20:
        raise ValueError("series too short for routing at this embedding dimension")
    split = int(n * 0.7)
    ctx_tr, ctx_ca = contexts[:split], contexts[split:]
    y_tr, y_ca = successors[:split], successors[split:]

    # producer 1: linear AR
    lin = linear_ar_fit(ctx_tr, y_tr)
    lin_mae = float(np.mean([abs(lin(c) - t) for c, t in zip(ctx_ca, y_ca)]))

    # producer 2: analog (the VSA-native one)
    af = AnalogForecaster(sim_floor=0.0, seed=seed).fit(ctx_tr, y_tr)   # sim_floor 0 here; abstention is conformal-side
    def analog_predict(c):
        f = af.forecast(c, k=8)
        return f["point"] if f["point"] is not None else float(y_tr.mean())
    ana_mae = float(np.mean([abs(analog_predict(c) - t) for c, t in zip(ctx_ca, y_ca)]))

    # choose the producer that calibrated tighter (the MEASURED routing decision)
    if lin_mae <= ana_mae:
        name, predict_fn = "linear", lin
    else:
        name, predict_fn = "analog", analog_predict

    # calibrate the conformal interval on the CHOSEN producer's calibration residuals
    cf = ConformalForecaster(alpha=alpha, kind="scalar", abstain_width=abstain_width)
    preds = [predict_fn(c) for c in ctx_ca]
    cf.calibrate(preds, list(y_ca))
    return RoutedForecaster(name, predict_fn, cf), {"linear_mae": lin_mae, "analog_mae": ana_mae, "chosen": name}


def _selftest():
    """A near-linear series routes to 'linear'; a nonlinear-recurrent series routes to 'analog'; both return a
    calibrated interval, and a misroute would still be safe (wide interval), not a confident wrong answer."""
    rng = np.random.default_rng(0)

    # (1) near-linear: an AR(1) process x_t = 0.8 x_{t-1} + noise -> the linear producer should win
    x = [0.0]
    for _ in range(3000):
        x.append(0.8 * x[-1] + 0.1 * rng.standard_normal())
    rf_lin, info_lin = route_and_forecast(np.array(x), d=5, alpha=0.1)
    assert info_lin["chosen"] == "linear", info_lin
    out = rf_lin.predict(np.array(x[-5:]))
    assert "interval" in out and out["coverage"] == 0.9

    # (2) nonlinear-recurrent: the logistic map x_{t+1}=3.9 x(1-x). A linear window can't capture the quadratic
    # curvature, but analog recall (a near-identical past state has a near-identical successor) can -> analog wins
    lx = [0.37]
    for _ in range(4000):
        lx.append(3.9 * lx[-1] * (1.0 - lx[-1]))
    rf_ana, info_ana = route_and_forecast(np.array(lx), d=4, alpha=0.1)
    assert info_ana["chosen"] == "analog", info_ana
    out2 = rf_ana.predict(np.array(lx[-4:]))
    assert "interval" in out2 and out2["producer"] == "analog"

    print("holographic_forecast selftest OK: AR(1) series routes to 'linear' (lin MAE %.4f < analog %.4f); fast "
          "quasi-periodic routes to 'analog' (analog MAE %.4f < lin %.4f); both return calibrated 90%% intervals"
          % (info_lin["linear_mae"], info_lin["analog_mae"], info_ana["analog_mae"], info_ana["linear_mae"]))


if __name__ == "__main__":
    _selftest()

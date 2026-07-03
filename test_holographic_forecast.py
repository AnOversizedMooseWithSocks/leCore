"""F3: the forecast(data) router -- pick the tighter producer, wrap in conformal."""
import numpy as np
from holographic_forecast import route_and_forecast, linear_ar_fit


def test_routes_linear_for_ar():
    rng = np.random.default_rng(0); x = [0.0]
    for _ in range(3000):
        x.append(0.8 * x[-1] + 0.1 * rng.standard_normal())
    rf, info = route_and_forecast(np.array(x), d=5, alpha=0.1)
    assert info["chosen"] == "linear"
    assert rf.predict(np.array(x[-5:]))["coverage"] == 0.9


def test_routes_analog_for_logistic():
    lx = [0.37]
    for _ in range(4000):
        lx.append(3.9 * lx[-1] * (1 - lx[-1]))
    rf, info = route_and_forecast(np.array(lx), d=4, alpha=0.1)
    assert info["chosen"] == "analog"
    out = rf.predict(np.array(lx[-4:]))
    assert out["producer"] == "analog" and "interval" in out


def test_linear_ar_fits_a_line():
    # y = 2*x0 + 1 exactly -> the AR fit recovers it
    X = np.random.default_rng(0).standard_normal((200, 1))
    y = 2 * X[:, 0] + 1
    pred = linear_ar_fit(X, y)
    assert abs(pred([3.0]) - 7.0) < 1e-6

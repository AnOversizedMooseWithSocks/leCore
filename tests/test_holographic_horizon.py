"""F6: multi-horizon forecast with a trusted-horizon gate."""
import numpy as np
from holographic.misc.holographic_horizon import MultiHorizonForecaster


def test_smooth_widens_and_trusts_far():
    theta = 0.15
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) * 0.99
    true = lambda st, H: np.array([(np.linalg.matrix_power(R, i + 1) @ st)[0] for i in range(H)])
    roll = lambda st, H: np.array([(np.linalg.matrix_power(R * 1.001, i + 1) @ st)[0] for i in range(H)])
    rng = np.random.default_rng(0); H = 30
    states = [rng.standard_normal(2) for _ in range(300)]
    mh = MultiHorizonForecaster(roll, alpha=0.1)
    q = mh.calibrate(states, [true(s, H) for s in states], H)
    assert q[-1] >= q[0]
    assert mh.forecast(rng.standard_normal(2), tolerance=float(np.median(q)))["trusted_horizon"] >= 3


def test_chaos_short_horizon():
    true = lambda st, H: _logi(st, H, 0.0)
    roll = lambda st, H: _logi(st, H, 1e-4)
    rng = np.random.default_rng(0); H = 30
    st = [rng.uniform(0.1, 0.9) for _ in range(300)]
    mh = MultiHorizonForecaster(roll, alpha=0.1)
    qc = mh.calibrate(st, [true(s, H) for s in st], H)
    assert qc[-1] > qc[0] * 3                       # chaotic width blows up with horizon
    assert mh.forecast(0.5, tolerance=0.05)["trusted_horizon"] < H


def _logi(st, H, eps):
    x = float(st) + eps; out = []
    for _ in range(H):
        x = 3.9 * x * (1 - x); out.append(x)
    return np.array(out)

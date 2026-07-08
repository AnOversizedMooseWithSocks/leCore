"""A deterministic Kolmogorov-Arnold readout on holostuff encoders (adaptive grid + LS additive fit)."""
import numpy as np
from holographic.agents_and_reasoning.holographic_kan import HolographicKAN, AdaptiveScalarEncoder


def _r2(y, yh):
    return 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)


def test_fits_additive_target_and_recovers_univariate_parts():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (1500, 2))
    g1 = lambda t: np.sin(2 * np.pi * t)
    g2 = lambda t: 4 * (t - 0.5) ** 2
    y = g1(X[:, 0]) + g2(X[:, 1]) + 0.02 * rng.standard_normal(1500)
    k = HolographicKAN(2, seed=0).fit(X[:1000], y[:1000])
    assert _r2(y[1000:], k.predict(X[1000:])) > 0.99
    ts = np.linspace(0.05, 0.95, 40)
    p1 = k.feature_function(0, ts); p2 = k.feature_function(1, ts)
    assert abs(np.corrcoef(p1, g1(ts))[0, 1]) > 0.97          # psi_1 recovers sin (KAN interpretability)
    assert abs(np.corrcoef(p2, g2(ts))[0, 1]) > 0.97          # psi_2 recovers the quadratic


def test_beats_linear_readout_on_nonlinear_additive():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (1200, 2))
    y = np.sin(2 * np.pi * X[:, 0]) + 4 * (X[:, 1] - 0.5) ** 2 + 0.02 * rng.standard_normal(1200)
    k = HolographicKAN(2, seed=0).fit(X[:900], y[:900])
    Xtr = np.hstack([X[:900], np.ones((900, 1))])
    coef = np.linalg.lstsq(Xtr, y[:900], rcond=None)[0]
    ylin = np.hstack([X[900:], np.ones((300, 1))]) @ coef
    assert _r2(y[900:], k.predict(X[900:])) > _r2(y[900:], ylin) + 0.3


def test_adaptive_grid_beats_uniform_on_skewed_feature():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, 1500) ** 4                          # clustered near 0
    y = np.sin(12 * np.pi * x) + 0.02 * rng.standard_normal(1500)
    X = x.reshape(-1, 1)
    adaptive = HolographicKAN(1, seed=1).fit(X[:1000], y[:1000])
    uniform = HolographicKAN(1, seed=1)
    for e in uniform.encoders:
        e._sorted = None                                      # force the uniform (identity) grid
    Phi = uniform._design(X[:1000])
    uniform.coef = np.linalg.solve(Phi.T @ Phi + 1e-2 * np.eye(Phi.shape[1]), Phi.T @ y[:1000])
    assert _r2(y[1000:], adaptive.predict(X[1000:])) > _r2(y[1000:], uniform.predict(X[1000:])) + 0.05


def test_kept_negative_additive_cannot_do_interactions():
    rng = np.random.default_rng(3)
    X = (rng.uniform(-1, 1, (1500, 2)) + 1) / 2
    y_prod = (2 * X[:, 0] - 1) * (2 * X[:, 1] - 1) + 0.02 * rng.standard_normal(1500)
    y_add = (2 * X[:, 0] - 1) ** 2 + (2 * X[:, 1] - 1) ** 2 + 0.02 * rng.standard_normal(1500)
    kp = HolographicKAN(2, seed=3).fit(X[:1000], y_prod[:1000])
    ka = HolographicKAN(2, seed=4).fit(X[:1000], y_add[:1000])
    assert _r2(y_prod[1000:], kp.predict(X[1000:])) < 0.2     # interaction: additive form can't
    assert _r2(y_add[1000:], ka.predict(X[1000:])) > 0.9      # additive control: succeeds (the boundary)


def test_adaptive_warp_maps_skewed_data_toward_uniform():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 1, 4000) ** 4
    enc = AdaptiveScalarEncoder(seed=0).fit(x)
    w = enc.warp(x)
    # the CDF warp turns any distribution into ~uniform ranks in [0,1]
    assert abs(w.mean() - 0.5) < 0.05 and w.min() >= 0.0 and w.max() <= 1.0


def test_warp_is_identity_before_fit():
    enc = AdaptiveScalarEncoder(seed=0)
    assert np.allclose(enc.warp(np.array([0.2, 0.7])), np.array([0.2, 0.7]))

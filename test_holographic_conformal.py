"""F1/F2/F8: calibrated forecast confidence -- conformal intervals, temporal ACI, proper scoring."""
import numpy as np
from holographic_conformal import (ConformalForecaster, wrap, AdaptiveConformal, conformal_quantile,
                                    coverage_report, empirical_coverage, crps_sample, pinball_loss,
                                    weighted_conformal_quantile, nonconformity_vector)


def test_scalar_coverage_tracks_nominal():
    rng = np.random.default_rng(0)
    truth = rng.standard_normal(4000); pred = truth + rng.standard_normal(4000) * 0.5
    resid = np.abs(pred - truth)
    for row in coverage_report(resid[:2000], resid[2000:], alphas=(0.05, 0.1, 0.2)):
        assert abs(row["empirical"] - row["nominal"]) < 0.03


def test_finite_sample_quantile_matches_reasoning_rule():
    # the (n+1)(1-alpha) order statistic -- same rule as holographic_reasoning's scalar ConformalPredictor
    r = np.arange(1, 101).astype(float)                         # residuals 1..100
    q = conformal_quantile(r, 0.1)                              # ceil(101*0.9)=91 -> the 91st smallest = 91
    assert q == 91.0


def test_vector_conformal_covers_at_nominal():
    rng = np.random.default_rng(1); dim = 256
    base = rng.standard_normal((3000, dim)); base /= np.linalg.norm(base, axis=1, keepdims=True)
    noisy = base + 0.3 * rng.standard_normal((3000, dim))
    cf = ConformalForecaster(alpha=0.1, kind="vector")
    cf.calibrate(list(noisy[:1500]), list(base[:1500]))
    covered = [cf.covers(noisy[i], base[i]) for i in range(1500, 3000)]
    assert abs(empirical_coverage(covered) - 0.9) < 0.03


def test_wrap_abstains_on_wide_interval():
    rng = np.random.default_rng(2)
    producer = lambda x: x + rng.standard_normal() * 2.0
    xs = rng.standard_normal(500); ys = xs.copy()
    tight = wrap(producer, xs, ys, alpha=0.1, abstain_width=0.5)
    loose = wrap(producer, xs, ys, alpha=0.1, abstain_width=100.0)
    assert tight(0.0)["abstain"] is True
    assert loose(0.0)["abstain"] is False


def test_aci_holds_coverage_under_drift():
    rng = np.random.default_rng(3); n = 3000
    stream = np.abs(rng.standard_normal(n) * np.linspace(0.5, 4.0, n))    # variance grows -> drift
    fixed_q = conformal_quantile(stream[:300], 0.1)
    fixed_cov = empirical_coverage(stream[300:] <= fixed_q)
    aci = AdaptiveConformal(alpha=0.1, gamma=0.05, window=300)
    for r in stream:
        aci.step(r)
    assert aci.realized_coverage() > fixed_cov                  # ACI adapts; fixed CP under-covers
    assert abs(aci.realized_coverage() - 0.9) < 0.05


def test_weighted_quantile_tracks_recent():
    # residuals small early, large late; the weighted quantile leans toward the large recent tail
    resid = np.concatenate([np.full(200, 1.0), np.full(200, 5.0)])
    plain = conformal_quantile(resid, 0.1)
    weighted = weighted_conformal_quantile(resid, 0.1, halflife=30)
    assert weighted >= plain                                    # recent-weighted -> at least as wide


def test_crps_and_pinball_discriminate():
    rng = np.random.default_rng(4)
    good = crps_sample(rng.standard_normal(500) * 0.3 + 1.0, 1.0)
    bad = crps_sample(rng.standard_normal(500) * 3.0 + 1.0, 1.0)
    assert good < bad
    # pinball: an under-prediction at the median is penalised symmetrically to an equal over-prediction
    assert abs(pinball_loss(0.0, 1.0, 0.5) - pinball_loss(2.0, 1.0, 0.5)) < 1e-9


def test_deterministic():
    r = np.arange(1, 51).astype(float)
    assert conformal_quantile(r, 0.1) == conformal_quantile(r, 0.1)
    a = ConformalForecaster(alpha=0.1, kind="scalar"); a.calibrate_residuals(r)
    b = ConformalForecaster(alpha=0.1, kind="scalar"); b.calibrate_residuals(r)
    assert a.q == b.q

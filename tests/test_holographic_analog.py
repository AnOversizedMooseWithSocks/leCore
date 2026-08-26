"""F4: analog forecasting -- find the similar past, return the successor distribution."""
import numpy as np
from holographic.misc.holographic_analog import AnalogForecaster, delay_embed


def _signal(n=4000, seed=0):
    rng = np.random.default_rng(seed); t = np.arange(n)
    return np.sin(t * 0.55) + 0.5 * np.sin(t * 0.27) + 0.03 * rng.standard_normal(n)


def test_beats_persistence_and_mean():
    series = _signal(); d = 20
    ctx, succ = delay_embed(series, d); ntr = 3000
    af = AnalogForecaster(sim_floor=0.5, seed=0).fit(ctx[:ntr], succ[:ntr])
    ea, ep, em = [], [], []; tm = float(succ[:ntr].mean())
    for i in range(ntr, len(ctx)):
        f = af.forecast(ctx[i], k=8)
        if f["abstain"]:
            continue
        ea.append(abs(f["point"] - succ[i])); ep.append(abs(ctx[i][-1] - succ[i])); em.append(abs(tm - succ[i]))
    assert np.mean(ea) < np.mean(ep) and np.mean(ea) < np.mean(em)


def test_yields_distribution():
    series = _signal(); ctx, succ = delay_embed(series, 20)
    af = AnalogForecaster(seed=0).fit(ctx[:3000], succ[:3000])
    f = af.forecast(ctx[3000], k=8)
    assert len(f["samples"]) > 1 and f["confidence"] > 0.5


def test_confidence_ordering_and_strict_abstain():
    series = _signal(); ctx, succ = delay_embed(series, 20)
    af = AnalogForecaster(sim_floor=0.5, seed=0).fit(ctx[:3000], succ[:3000])
    rng = np.random.default_rng(1); alien = rng.standard_normal(20) * 10.0
    assert af.forecast(alien)["confidence"] < af.forecast(ctx[3000])["confidence"]
    strict = AnalogForecaster(sim_floor=0.95, seed=0).fit(ctx[:3000], succ[:3000])
    assert strict.forecast(alien)["abstain"] is True


def test_deterministic():
    series = _signal(); ctx, succ = delay_embed(series, 20)
    af = AnalogForecaster(seed=0).fit(ctx[:3000], succ[:3000])
    assert af.forecast(ctx[3000])["point"] == af.forecast(ctx[3000])["point"]

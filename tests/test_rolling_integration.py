"""Rolling kit (H1) at the seams: through the mind, closing the TRAILING_STATS seam the conditioning layer
declared (a gate built from the lambda and a gate thresholded on the rolling series must agree decision for
decision), streaming warm-start feeding a live gate, and the lint as the standing contract."""
import numpy as np

import lecore
from holographic.agents_and_reasoning.holographic_conditioning import TRAILING_STATS, trailing_gate
from holographic.sampling_and_signal.holographic_rolling import rolling_std


def test_the_faculty_returns_the_requested_series_and_refuses_unknown_stats():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    x = np.cumsum(np.random.default_rng(0).standard_normal(300))
    r = mind.rolling_stats(x, 20, stats=("mean", "std", "quantile", "drawdown"), q=0.9)
    assert set(r) == {"mean", "std", "quantile", "drawdown"}
    assert all(v.shape == (300,) for v in r.values())
    assert np.isnan(r["std"][:19]).all() and not np.isnan(r["std"][19:]).any()
    try:
        mind.rolling_stats(x, 20, stats=("std", "sharpe"))
        raise AssertionError("expected refusal")
    except ValueError as e:
        assert "sharpe" in str(e)


def test_the_trailing_stats_seam_gate_and_series_agree_decision_for_decision():
    """The seam the conditioning module declared, closed and pinned: a storm gate built from
    TRAILING_STATS['std'] and a mask built by thresholding rolling_std must FLIP AT THE SAME INDICES. Any
    numerical daylight between the lambda and the series would let the same threshold mean two different
    gates -- the tie-sensitivity rule applied to conditions."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 0.5, 300), rng.normal(0, 2.5, 200), rng.normal(0, 0.5, 300)])
    w, thr = 50, 1.2
    # min_periods=w so both sides define the warm-up the same way: the gate stays closed until a full
    # window exists, matching the series' NaN (and NaN >= thr is False). With the gate's default
    # min_periods=1 the two would legitimately differ in the warm-up -- partial-window stds vs closed --
    # which is a POLICY difference, not numerical daylight; the pin is about the numbers.
    gate = trailing_gate("std", window=w, threshold=thr, compare="ge", min_periods=w)
    gate_mask = np.asarray(gate.mask(x), bool)
    series_mask = rolling_std(x, w) >= thr
    assert (gate_mask == series_mask).all()
    assert series_mask[320:480].mean() > 0.9                     # the gate actually opened in the storm
    assert series_mask[:300].mean() < 0.05


def test_streaming_warm_start_drives_a_live_gate_identically_to_the_backtest():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.standard_normal(500))
    w = 30
    live = mind.streaming_stats(window=w).warm_start(x[:400])
    vec = mind.rolling_stats(x, w, stats=("std",))["std"]
    for i, v in enumerate(x[400:], start=400):
        live.push(v)
        assert abs(live.std() - vec[i]) < 1e-9


def test_every_kit_stat_stays_lint_causal_through_the_mind():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    x = np.cumsum(np.random.default_rng(3).standard_normal(300))
    for s in ("mean", "std", "min", "max", "range", "drawdown"):
        fn = lambda z, s=s: np.nan_to_num(mind.rolling_stats(z, 20, stats=(s,))[s])
        assert mind.lookahead_lint(fn, x)["max_drift"] == 0.0, s

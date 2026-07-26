"""Event study (H2) at the seams: through the mind, composed with the reclock (events from a re-clocked
series studied on the original), with trailing_gate (gate transitions AS the events), and the pre-trend
diagnostic agreeing with target_shift_probe about the same selection leak seen from two angles."""
import numpy as np

import lecore


def test_the_faculty_finds_a_planted_drift_and_stays_quiet_on_noise():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(0)
    y = rng.standard_normal(4000)
    ev = list(range(120, 3800, 130))
    for e in ev:
        y[e + 1:e + 9] += 0.4
    r = mind.event_study(y, ev, horizon=15, seed=0)
    assert r["forward"]["p"] < 0.01 and r["forward"]["z"] > 2.5
    y0 = rng.standard_normal(4000)
    r0 = mind.event_study(y0, ev, horizon=15, seed=1)
    assert r0["forward"]["p"] > 0.02


def test_gate_transitions_as_events_the_storm_onset_study():
    """The composition the campaign used constantly: study what the outcome does after a GATE OPENS. Storm
    onsets (trailing-std gate rising edge) on a vol-clustered series must show elevated |outcome| forward --
    the envelope module's clustering fact re-derived through two other faculties."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(1)
    blocks = []
    for k in range(10):
        blocks.append(rng.normal(0, 0.5, 400))
        blocks.append(rng.normal(0, 2.5, 250))
    x = np.concatenate(blocks + [rng.normal(0, 0.5, 400)])
    std = mind.rolling_stats(x, 40, stats=("std",))["std"]
    mask = np.nan_to_num(std) >= 1.2
    onsets = np.where(mask[1:] & ~mask[:-1])[0] + 1              # ten storm onsets, one per storm
    r = mind.event_study(np.abs(x), list(onsets), horizon=60, pre=60, seed=0)
    assert r["forward"]["z"] > 2, r["forward"]                   # |x| elevated after a storm onset
    # and the onset detector is itself trailing, so the pre-window is calmer than the forward one.
    assert r["mean_path"][-1] > r["mean_path"][0], (r["mean_path"][0], r["mean_path"][-1])


def test_pre_trend_and_shift_probe_see_the_same_selection_leak():
    """One selection leak, two instruments: events thresholded on trailing runup. The event study's pre_trend
    z flags it; target_shift_probe on the same trigger-vs-outcome reads not-ahead dominant. Agreement between
    independent diagnostics is the point of having both."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(2)
    y = rng.standard_normal(5000)
    run = np.convolve(y, np.ones(10), mode="full")[:5000]
    trigger = (run > 4.0) & (np.arange(5000) >= 9)
    ev = np.where(trigger)[0]
    ev = ev[np.concatenate([[True], np.diff(ev) > 25])]
    r = mind.event_study(y, list(ev), horizon=20, pre=12, seed=0)
    assert r["pre_trend"]["z"] > 3 and r["forward"]["p"] > 0.05
    probe = mind.target_shift_probe(run, y, max_lag=3)
    assert probe["corr_not_ahead"] > 2 * probe["corr_ahead"]


def test_reclocked_events_studied_on_the_original_axis():
    """Reclock a noisy series by movement, take the emitted events, and study the ORIGINAL series around
    them: on pure noise the reclock's events must show no forward drift under the shift null -- the reclock
    negative (manufactured reversion is a property of its own emitted sequence, not of the underlying data)
    confirmed through the event-study lens."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.standard_normal(4000))
    rc = mind.reclock(x, step=2.0)
    ev = [int(i) for i in rc["source_index"] if 30 <= i < 3960]
    d = np.concatenate([[0.0], np.diff(x)])
    r = mind.event_study(d, ev, horizon=25, seed=0)
    assert r["forward"]["p"] > 0.01, r["forward"]                # no real forward structure on noise

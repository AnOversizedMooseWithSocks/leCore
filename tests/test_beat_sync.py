"""Regression traps for SYNC (sweep 137): onsets, tempo, the beat grid, and the abstention.

A beat detector has no ground truth unless you make one, so every fixture here is GENERATED and its beat
times are exact by construction. That is what turns "it found some beats" into precision and recall.
Two things rot quietly in a detector like this: the numbers drift while still looking plausible, and the
failure envelope gets "fixed" by a threshold that quietly ruins the good case. Both are pinned.
"""
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

RATE = 22050
SECS = 8.0


@pytest.fixture(scope="module")
def mind():
    import lecore
    return lecore.UnifiedMind(dim=256, seed=0)


def clicks(times, secs=SECS, attack=0.001, decay=0.03, noise=0.0, seed=0, gain=1.0):
    """The fixture IS the ground truth -- the only reason any number in this file means anything."""
    n = int(secs * RATE)
    x = np.zeros(n)
    for t0 in times:
        i = int(t0 * RATE)
        et = np.arange(int(0.25 * RATE)) / RATE
        env = np.minimum(et / max(attack, 1e-9), 1.0) * np.exp(-et / decay)
        seg = (np.sin(2 * np.pi * 180.0 * et) * env)[: max(0, n - i)]
        x[i:i + len(seg)] += seg
    if noise:
        x = x + np.random.default_rng(seed).normal(0, noise, n)
    return x * gain


def pr(det, truth, tol=0.05):
    """Precision/recall at the MIREX +/-50 ms tolerance, greedy nearest-match, each truth used once."""
    used, tp = set(), 0
    for d in det:
        cand = [j for j, t in enumerate(truth) if j not in used and abs(d - t) <= tol]
        if cand:
            used.add(min(cand, key=lambda j: abs(d - truth[j])))
            tp += 1
    return (tp / len(det) if len(det) else 0.0), (tp / len(truth) if len(truth) else 1.0)


@pytest.mark.parametrize("bpm", [60, 90, 120, 140, 175])
def test_clean_clicks_are_found_exactly_across_the_tempo_range(mind, bpm):
    """The easy case has to be perfect, or nothing harder is worth reporting."""
    beats = np.arange(0.5, SECS, 60.0 / bpm)
    det = mind.onset_detect(clicks(beats), RATE)
    p, r = pr(det["times"], beats)
    assert (p, r) == (1.0, 1.0), (bpm, p, r, det["n"])


def test_tempo_is_accurate_and_confidence_means_something(mind):
    """The least-squares refinement is what buys the accuracy: the median IOI alone is quantised to the
    analysis frame and read 117.45 BPM for a true 120, a miss that compounded into a 40 ms grid error."""
    beats = np.arange(0.5, SECS, 0.5)
    det = mind.onset_detect(clicks(beats), RATE)
    tem = mind.tempo(det["times"])
    assert abs(tem["bpm"] - 120.0) / 120.0 < 0.005, tem
    assert tem["confidence"] > 0.95, tem


def test_the_beat_grid_fits_far_better_than_chance(mind):
    """A grid needs its own null. A RANDOM phase averages a quarter of the period -- 125 ms at 120 BPM --
    so only a number far below that is evidence of a fit."""
    beats = np.arange(0.5, SECS, 0.5)
    det = mind.onset_detect(clicks(beats), RATE)
    grid = mind.beat_grid(det["times"], SECS)
    assert grid["mean_abs_error"] < 0.04, grid
    assert grid["mean_abs_error"] < 0.125 / 3.0, "no better than a third of chance alignment"


def test_noise_is_survived_up_to_a_measured_point(mind):
    """Stated as a range, so the envelope is a measurement rather than a boast."""
    beats = np.arange(0.5, SECS, 0.5)
    assert pr(mind.onset_detect(clicks(beats, noise=0.2), RATE)["times"], beats) == (1.0, 1.0)
    p, r = pr(mind.onset_detect(clicks(beats, noise=0.5), RATE)["times"], beats)
    assert r == 1.0 and p > 0.8, (p, r)


def test_it_abstains_on_anything_without_transient_structure(mind):
    """THE PROPERTY THIS DETECTOR IS JUDGED ON. Normalisation erases loudness, so noise fills 0..1
    exactly as music does -- without a structure gate the detector fired 62 times on silence-with-noise.
    Loud noise is in here on purpose: a gate that only caught QUIET noise would be a loudness test
    wearing a structure test's name."""
    for label, sig in (("silence", np.zeros(int(4 * RATE))),
                       ("quiet noise", np.random.default_rng(0).normal(0, 0.001, int(4 * RATE))),
                       ("loud noise", np.random.default_rng(1).normal(0, 0.3, int(4 * RATE)))):
        det = mind.onset_detect(sig, RATE)
        assert det["n"] == 0 and det["abstained"] is True, (label, det["n"])
        assert "structure" in det["why"]


def test_the_structure_gate_is_gain_invariant(mind):
    """What makes it a structure test and not a volume knob: the same music at 1/100th the level must
    give the same verdict and the same ratio."""
    beats = np.arange(0.5, SECS, 0.5)
    loud = mind.onset_detect(clicks(beats), RATE)
    quiet = mind.onset_detect(clicks(beats, gain=0.01), RATE)
    assert abs(loud["structure_ratio"] - quiet["structure_ratio"]) < 1e-9
    assert quiet["n"] == loud["n"]


def test_the_failure_envelope_is_pinned_as_a_failure(mind):
    """SLOW ATTACKS ARE WHERE THIS BREAKS, and it is asserted as a RANGE so the limit cannot be quietly
    'fixed' by a threshold that ruins the clean case. A flux detector keys on a sharp rise; give it a
    30 ms attack and it finds every beat and roughly as many again."""
    beats = np.arange(0.5, SECS, 0.5)
    p, r = pr(mind.onset_detect(clicks(beats, attack=0.030), RATE)["times"], beats)
    assert r == 1.0, "recall should hold; it over-fires rather than missing"
    assert 0.4 < p < 0.8, "precision on a slow attack is the documented envelope: %.2f" % p


def test_an_onset_at_time_zero_cannot_be_detected(mind):
    """Structural, not a tuning failure: flux[0] is 0 by construction because there is no previous frame
    to difference against. If this ever passes, the docstrings claiming it are stale."""
    beats = np.arange(0.0, SECS, 0.5)
    _, r = pr(mind.onset_detect(clicks(beats), RATE)["times"], beats)
    assert r < 1.0


def test_a_syncopated_pattern_reports_a_subdivision_and_says_it_is_unsure(mind):
    """The classic tempo octave error, and the honest part is that `confidence` catches it. Reporting
    184.6 BPM as fact would be wrong; reporting it at confidence 0.52 is a different claim."""
    sync = np.sort(np.concatenate([np.arange(0.5, SECS, 0.5), np.arange(0.83, SECS, 0.5)]))
    det = mind.onset_detect(clicks(sync), RATE)
    tem = mind.tempo(det["times"])
    assert tem["confidence"] < 0.8, tem


def test_the_decision_layer_is_cheap_enough_to_run_in_a_frame(mind):
    """Real-time, gated as a RATIO against the analysis rather than in milliseconds -- CI is a different
    machine every time and a wall-clock assertion is a claim about a machine. Measured on this box the
    per-frame decision was 0.026 ms, 654x inside a 16.7 ms budget."""
    import time
    beats = np.arange(0.5, SECS, 0.5)
    bus = mind.audio_param_bus(clicks(beats), RATE, hop=512, size=2048, smooth=1)
    t0 = time.time()
    for _ in range(5):
        mind.onset_detect(bus)
    decision = time.time() - t0
    t0 = time.time()
    for _ in range(5):
        mind.audio_param_bus(clicks(beats), RATE, hop=512, size=2048, smooth=1)
    analysis = time.time() - t0
    assert decision < analysis, (decision, analysis)


def test_the_sync_loop_ships_as_a_runnable_application(mind):
    """A demo nobody can run is not a demo. The whole wire, end to end, through the dispatcher."""
    r = mind.app_run("beat_sync")
    p = r["proved"]
    assert p["precision"] == 1.0 and p["recall"] == 1.0
    assert p["sync_error_ms"] <= p["frame_ms"], p      # the visual lands within one frame of the beat

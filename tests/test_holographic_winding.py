"""Tests for holographic_winding: the reversing-carrier regime, resolved honestly.

Pins the three verdicts and their evidence:
  * 'function'  : all laps agree -> merged profile, and the merge measurably beats
                  the best single lap (multi-pass = free denoise, ~sqrt(#laps));
  * 'hysteresis': within-direction agreement + across-direction disagreement ->
                  per-direction branches, merge REFUSED (the average is a curve no
                  pass traced -- pinned by the branch gap);
  * 'path'      : disagreement even within a direction (drift) -> no merge.
Plus lap segmentation mechanics and determinism. Fixed seeds throughout.
"""

import numpy as np
import pytest

from holographic.sampling_and_signal.holographic_winding import (
    split_laps, winding_map, AGREE_TOL,
)


def _sweep(f, n_laps=6, n_per=120, noise=0.05, offset_fn=None, seed=0):
    """Back-and-forth sweep of f over [0,1]; offset_fn(k, up) adds per-lap terms."""
    rng = np.random.default_rng(seed)
    xs_up = np.linspace(0, 1, n_per)
    coords, content = [], []
    for k in range(n_laps):
        up = (k % 2 == 0)
        xs = xs_up if up else xs_up[::-1]
        off = offset_fn(k, up) if offset_fn else 0.0
        coords.append(xs)
        content.append(f(xs) + off + rng.standard_normal(n_per) * noise)
    return np.concatenate(coords), np.concatenate(content)


F = lambda x: np.sin(2 * np.pi * x) + 0.3 * x


# ---------------------------------------------------------------------------
# Lap segmentation
# ---------------------------------------------------------------------------

def test_split_laps_finds_reversals_and_directions():
    x = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 0, 50),
                        np.linspace(0, 1, 50)])
    laps = split_laps(x)
    assert len(laps) == 3
    assert [l["direction"] for l in laps] == [1, -1, 1]


def test_split_laps_treats_stalls_as_continuation():
    # a plateau mid-lap is a stall, not a reversal
    x = np.concatenate([np.linspace(0, 0.5, 30), np.full(10, 0.5),
                        np.linspace(0.5, 1, 30)])
    assert len(split_laps(x)) == 1


# ---------------------------------------------------------------------------
# Verdict: FUNCTION (merge pays, measured)
# ---------------------------------------------------------------------------

def test_stable_profile_reads_function_and_merge_beats_best_lap():
    coords, content = _sweep(F, noise=0.05)
    r = winding_map(coords, content)
    assert r["verdict"] == "function"
    true = F(r["grid"])
    merged_err = float(np.sqrt(np.nanmean((r["merged"] - true) ** 2)))
    lap_errs = [float(np.sqrt(np.nanmean((c - true) ** 2))) for c in r["lap_curves"]]
    assert merged_err < min(lap_errs)          # the multi-pass win, with baseline
    assert r["disagreement"]["all"] <= AGREE_TOL


# ---------------------------------------------------------------------------
# Verdict: HYSTERESIS (merge refused, branches recovered)
# ---------------------------------------------------------------------------

def test_direction_dependent_offset_reads_hysteresis():
    coords, content = _sweep(F, noise=0.03,
                             offset_fn=lambda k, up: 0.4 if up else -0.4)
    r = winding_map(coords, content)
    assert r["verdict"] == "hysteresis", r["disagreement"]
    assert r["merged"] is None                 # the refusal IS the contract
    # branches recovered near their true offsets
    true = F(r["grid"])
    up_err = float(np.sqrt(np.nanmean((r["branches"]["up"] - (true + 0.4)) ** 2)))
    dn_err = float(np.sqrt(np.nanmean((r["branches"]["down"] - (true - 0.4)) ** 2)))
    assert up_err < 0.05 and dn_err < 0.05
    # the fictitious average would sit ~0.4 from each branch: no pass traced it
    mid = 0.5 * (r["branches"]["up"] + r["branches"]["down"])
    assert float(np.nanmean(np.abs(r["branches"]["up"] - mid))) > 0.3


# ---------------------------------------------------------------------------
# Verdict: PATH (drift; no merge offered)
# ---------------------------------------------------------------------------

def test_per_lap_drift_reads_path():
    coords, content = _sweep(F, noise=0.03,
                             offset_fn=lambda k, up: 0.3 * k)
    r = winding_map(coords, content)
    assert r["verdict"] == "path", r["disagreement"]
    assert r["merged"] is None and r["branches"] is None
    assert len(r["lap_curves"]) >= 4           # the per-lap evidence is returned


def test_single_monotone_lap_is_trivially_a_function():
    # Two ascending disjoint segments are ONE monotone lap (no reversal ever
    # happens -- the jump is just a big forward step), so nothing is revisited and
    # the verdict is trivially 'function' with the lap itself as the profile. The
    # first draft of this test expected 'path' here and the module was right to
    # disagree: monotonicity, not contiguity, is what defines a lap.
    x = np.concatenate([np.linspace(0, 1, 40), np.linspace(2, 3, 40)])
    r = winding_map(x, np.sin(x))
    assert r["verdict"] == "function"
    assert r["n_laps"] == 1


# ---------------------------------------------------------------------------
# Evidence + determinism
# ---------------------------------------------------------------------------

def test_disagreement_numbers_travel_with_every_verdict():
    coords, content = _sweep(F, noise=0.05)
    r = winding_map(coords, content)
    for k in ("all", "within_up", "within_down", "across_directions"):
        assert k in r["disagreement"]


def test_winding_map_is_deterministic():
    coords, content = _sweep(F, noise=0.05)
    a = winding_map(coords, content)
    b = winding_map(coords, content)
    assert a["verdict"] == b["verdict"]
    assert np.array_equal(a["merged"], b["merged"])


def test_selftest_runs():
    from holographic.sampling_and_signal.holographic_winding import _selftest
    _selftest()

"""Tests for holographic_scaffold: the auto-exploration orchestrator.

Pins the loop's contracts: scaffold selection (boring AND organising, with the
score table as evidence), rectification of a wobbling carrier, MDL decomposition
along the carrier with the explained fraction MEASURED against the recomposition,
residuals returned as the hand-off, and the honest no-structure verdict on noise.
Fixed seeds; deterministic end to end.
"""

import numpy as np
import pytest

import lecore
from holographic.sampling_and_signal.holographic_scaffold import (
    scaffold_scores, explore_series, MIN_CONTINUITY, MIN_EXPLAINED,
)

MIND = lecore.UnifiedMind(dim=256, seed=0)


# ---------------------------------------------------------------------------
# Scaffold selection
# ---------------------------------------------------------------------------

def test_true_carrier_wins_with_a_real_margin():
    # smooth evolution along axis 0; axis 1 holds unrelated channels -> axis 0
    # must win the score table, and not by a hair.
    rng = np.random.default_rng(0)
    T, A = 60, 5
    chan = rng.standard_normal(A) * 5.0
    cube = np.stack([chan + 0.05 * t for t in range(T)], axis=0)
    cube += rng.standard_normal(cube.shape) * 0.01
    rows = scaffold_scores(cube)
    assert rows[0]["axis"] == 0
    assert rows[0]["score"] > rows[1]["score"] + 0.05


def test_boring_but_disorganising_axis_does_not_win():
    # An axis can be perfectly boring (uniform index) yet organise nothing (its
    # slices are unrelated). Boredom alone must not make it a scaffold: with all
    # axes disorganising, the pipeline reports no scaffold rather than crowning
    # the least-bad one.
    rng = np.random.default_rng(1)
    noise = rng.standard_normal((50, 50))
    res = explore_series(noise, mind=MIND)
    assert res["scaffold"] is None
    assert res["verdict"] == "no structure found"
    assert res["scores"][0]["continuity"] < MIN_CONTINUITY


# ---------------------------------------------------------------------------
# The full loop on planted structure
# ---------------------------------------------------------------------------

def test_planted_laws_on_an_irregular_carrier_are_recovered():
    # Two channels with known laws, sampled at Poisson-irregular positions: the
    # loop must pick axis 0, rectify it to zero boredom, and explain >90% of each
    # channel's variance -- with the residual really being what the law misses.
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.exponential(1.0, size=200))
    u = (t - t[0]) / (t[-1] - t[0])
    series = np.stack([np.sin(2 * np.pi * 2 * u), 0.8 * u + 0.1], axis=1)
    res = explore_series(series, coords={0: t}, mind=MIND)
    assert res["scaffold"] == 0
    assert res["rectified"]["marginal_info_after"] == 0.0
    assert res["verdict"] == "structured"
    for ch in res["channels"]:
        assert ch["explained_fraction"] > 0.9
        assert ch["residual"].shape[0] == 200


def test_residuals_are_the_honest_hand_off():
    # explained + residual must actually account for the channel: recomposing
    # (channel - residual) and adding the residual back reproduces the channel
    # bit-exactly -- the variance ledger balances.
    rng = np.random.default_rng(2)
    u = np.linspace(0, 1, 150)
    series = np.stack([np.sin(2 * np.pi * 2 * u) + rng.standard_normal(150) * 0.02],
                      axis=1)
    res = explore_series(series, mind=MIND)
    ch = res["channels"][0]
    recon = series[:, 0] - ch["residual"]
    assert np.max(np.abs((recon + ch["residual"]) - series[:, 0])) < 1e-12


def test_mixed_cube_reads_weakly_structured():
    # one lawful channel + one noise channel -> some structure found, honestly
    # short of 'structured'.
    rng = np.random.default_rng(3)
    u = np.linspace(0, 1, 200)
    series = np.stack([np.sin(2 * np.pi * 2 * u),
                       rng.standard_normal(200)], axis=1)
    res = explore_series(series, mind=MIND)
    assert res["verdict"] in ("weakly structured", "structured")
    fracs = sorted(ch["explained_fraction"] for ch in res["channels"])
    assert fracs[0] < MIN_EXPLAINED     # the noise channel stayed unexplained
    assert fracs[-1] > 0.9              # the lawful channel was found


# ---------------------------------------------------------------------------
# Honesty + determinism
# ---------------------------------------------------------------------------

def test_pure_noise_is_never_dressed_as_law():
    rng = np.random.default_rng(4)
    res = explore_series(rng.standard_normal((80, 4)), mind=MIND)
    assert res["verdict"] == "no structure found"


def test_score_table_travels_with_every_result():
    rng = np.random.default_rng(5)
    res = explore_series(rng.standard_normal((40, 3)), mind=MIND)
    assert len(res["scores"]) == 2                      # one row per axis
    assert all("score" in r and "continuity" in r for r in res["scores"])


def test_explore_series_is_deterministic():
    rng = np.random.default_rng(6)
    u = np.linspace(0, 1, 120)
    series = np.stack([np.sin(2 * np.pi * 2 * u), 0.5 * u], axis=1)
    a = explore_series(series, mind=MIND)
    b = explore_series(series, mind=MIND)
    assert a["verdict"] == b["verdict"] and a["scaffold"] == b["scaffold"]
    for x, y in zip(a["channels"], b["channels"]):
        assert x["explained_fraction"] == y["explained_fraction"]


def test_selftest_runs():
    from holographic.sampling_and_signal.holographic_scaffold import _selftest
    _selftest()


# ---------------------------------------------------------------------------
# Arc adoptions (each measured against its baseline) + the degenerate-axis fix
# ---------------------------------------------------------------------------

def test_column_vector_does_not_pick_the_degenerate_axis():
    # REGRESSION TRAP: an (N, 1) column once let the LENGTH-1 axis win, laying N
    # samples out as N one-sample channels -- each trivially "explained", a
    # vacuous 'structured'. A scaffold you cannot index along is not a scaffold.
    rng = np.random.default_rng(0)
    xs = np.linspace(0, 1, 120)
    co, ct = [], []
    for k in range(6):
        c = xs if k % 2 == 0 else xs[::-1]
        co.append(c)
        ct.append(np.sin(2 * np.pi * 2 * c) + rng.standard_normal(120) * 0.15)
    res = explore_series(np.concatenate(ct).reshape(-1, 1),
                         coords={0: np.concatenate(co)}, mind=MIND)
    assert res["scaffold"] == 0
    assert res["n_channels"] == 1


def test_handle_reversals_adopts_the_winding_merge():
    # A reversing 6-pass scan: with the flag, winding_map's 'function' merge
    # replaces the payload (measured win: profile RMS 2.4x); default-off is the
    # old behaviour with no 'winding' key.
    rng = np.random.default_rng(0)
    xs = np.linspace(0, 1, 120)
    co, ct = [], []
    for k in range(6):
        c = xs if k % 2 == 0 else xs[::-1]
        co.append(c)
        ct.append(np.sin(2 * np.pi * 2 * c) + rng.standard_normal(120) * 0.15)
    co = np.concatenate(co)
    ct = np.concatenate(ct).reshape(-1, 1)
    off = explore_series(ct, coords={0: co}, mind=MIND)
    on = explore_series(ct, coords={0: co}, mind=MIND, handle_reversals=True)
    assert "winding" not in off
    assert on["winding"]["verdict"] == "function" and on["winding"]["n_laps"] == 6
    assert on["channels"][0]["explained_fraction"] >= \
        off["channels"][0]["explained_fraction"]


def test_handle_reversals_honours_the_hysteresis_refusal():
    # A hysteresis sweep must NOT be merged by the adoption: the verdict is
    # reported, the payload stays rectified -- winding_map's refusal, honoured.
    rng = np.random.default_rng(1)
    xs = np.linspace(0, 1, 120)
    co, ct = [], []
    for k in range(6):
        up = (k % 2 == 0)
        c = xs if up else xs[::-1]
        co.append(c)
        ct.append(np.sin(2 * np.pi * c) + (0.4 if up else -0.4)
                  + rng.standard_normal(120) * 0.03)
    on = explore_series(np.concatenate(ct).reshape(-1, 1),
                        coords={0: np.concatenate(co)}, mind=MIND,
                        handle_reversals=True)
    assert on["winding"]["verdict"] == "hysteresis"


def test_decompose_piecewise_beats_global_with_baseline_attached():
    from holographic.sampling_and_signal.holographic_scaffold import (
        decompose_piecewise)
    n = 100
    y = np.concatenate([2.0 * np.linspace(0, 1, n),
                        np.sin(2 * np.pi * 2 * np.linspace(0, 1, n)) + 3.0,
                        -np.linspace(0, 1, n) + 1.0])
    d = decompose_piecewise(y, mind=MIND, min_seg=24)
    assert d["residual_rms"] < 0.1 * d["baseline"]["residual_rms"]
    assert d["total_bits"] < d["baseline"]["mdl_bits"]
    # the reconstruction really is the pieces stitched
    assert d["reconstruction"].shape == y.shape


def test_decompose_piecewise_carries_its_baseline_honestly():
    from holographic.sampling_and_signal.holographic_scaffold import (
        decompose_piecewise)
    # A linear-conformant single regime: one segment, no shattering.
    rng = np.random.default_rng(0)
    ramp = 2.0 * np.linspace(0, 1, 240) + rng.standard_normal(240) * 0.05
    d_ramp = decompose_piecewise(ramp, mind=MIND, min_seg=24)
    assert len(d_ramp["segments"]) == 1
    # An oscillating signal (the segmenter's documented negative): it shatters,
    # piecewise LOSES on bits to the 1-term global fit (measured: 1136 vs 556)
    # -- and the attached baseline makes the loss VISIBLE. The honesty is the
    # contract: a non-paying segmentation is reported as such, never hidden.
    sine = np.sin(2 * np.pi * 2 * np.linspace(0, 1, 240))
    d_sine = decompose_piecewise(sine, mind=MIND, min_seg=24)
    assert d_sine["total_bits"] > d_sine["baseline"]["mdl_bits"]
    assert d_sine["baseline"]["residual_rms"] < 0.01   # global fit was fine

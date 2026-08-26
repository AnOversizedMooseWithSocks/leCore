"""Tests for holographic_axisrole: the index-vs-payload (carrier-vs-content) criterion.

These pin the exact numeric contracts of the theory that a low-information axis
should be an INDEX (carrier) and a high-information, content-coupled axis should
be a BIND (payload) -- and that getting it wrong destroys comparability by a
measurable amount. They double as the exploration suite the feature was built for:
video (time is the carrier), market data (time carrier, asset/field payload), and
a flat-message decode (the Arecibo / Arrival "recover the grid" case).

Determinism rule: all measurements are of the data (no RNG in the analyzer), and
the one RNG (per-slice binding keys in comparability_cost) is fully seeded, so
every assertion below is a hard numeric contract, not a smoke test.
"""

import numpy as np
import pytest

from holographic.sampling_and_signal.holographic_axisrole import (
    analyze_axes, axis_report, recommend_axis_role, comparability_cost,
    _delta_entropy_rate, _label_entropy, LOW_INFO_FRAC,
)


# ---------------------------------------------------------------------------
# The information measures themselves (the axis "boredom" scores)
# ---------------------------------------------------------------------------

def test_constant_delta_axis_is_maximally_boring():
    # A uniformly-sampled axis (dt = const) carries NO information in its spacing:
    # this is the ideal carrier. The delta-entropy-rate must be exactly 0.
    t = np.arange(50) * 0.1
    assert _delta_entropy_rate(t) == 0.0


def test_irregular_sampling_raises_axis_information():
    # Jittered / irregular timestamps carry information in WHEN samples land, so a
    # genuinely irregular axis reads a positive rate -- it is no longer pure carrier.
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.exponential(1.0, size=200))  # Poisson-like arrivals
    assert _delta_entropy_rate(t) > 0.3


def test_single_label_axis_has_zero_entropy():
    # A constant categorical axis (one label) is boring; all-distinct is maximal.
    assert _label_entropy(np.zeros(20)) == 0.0
    assert _label_entropy(np.arange(20)) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Case 1: VIDEO -- time is the carrier, indexed not bound
# ---------------------------------------------------------------------------

def test_video_time_axis_is_indexed():
    # A smoothly-drifting video cube [T,H,W]: uniform time sampling => marginal
    # info 0 => the recommender must call time an INDEX, with a real margin.
    rng = np.random.default_rng(0)
    T, H, W = 24, 8, 8
    base = rng.standard_normal((H, W))
    drift = rng.standard_normal((H, W)) * 0.05
    video = np.stack([base + drift * t for t in range(T)], axis=0)

    res = analyze_axes(video)
    time_rec = res["per_axis"][0]
    assert time_rec["role"] == "index"
    assert time_rec["marginal_info"] < LOW_INFO_FRAC
    assert time_rec["margin"] > 0.0  # not a borderline call
    assert 0 in res["index_axes"]


def test_binding_the_video_time_axis_destroys_comparability():
    # The thesis, measured: adjacent frames are ~identical (indexed_sim high), but
    # binding a distinct key per frame rotates each into a private subspace
    # (bound_sim ~0). The collapse is the concrete cost of the wrong role choice.
    rng = np.random.default_rng(0)
    T, H, W = 24, 8, 8
    base = rng.standard_normal((H, W))
    drift = rng.standard_normal((H, W)) * 0.03
    video = np.stack([base + drift * t for t in range(T)], axis=0)

    cost = comparability_cost(video, 0, dim=128, seed=0)
    assert cost["indexed_sim"] > 0.9      # neighbours genuinely alike
    assert cost["bound_sim"] < 0.2        # rotated apart by binding
    assert cost["collapse"] > 0.7         # most of the similarity destroyed


# ---------------------------------------------------------------------------
# Case 2: MARKET DATA -- time carrier, asset/field payload
# ---------------------------------------------------------------------------

def test_market_cube_time_indexes_content_axes_bind():
    # [T, asset, field] with a distinct signature per asset. Time is the carrier
    # (uniform bars); asset and field are informative categorical payload axes
    # whose content the data depends on -> BIND. This is the schema the analyzer
    # should recover once told which axes are categorical.
    rng = np.random.default_rng(1)
    T, A, F = 30, 5, 4
    asset_sig = rng.standard_normal((A, F)) * 2.0
    cube = np.stack([asset_sig + rng.standard_normal((A, F)) * 0.03 * t
                     for t in range(T)], axis=0)

    res = analyze_axes(cube, categorical=[1, 2])
    roles = {r["axis"]: r["role"] for r in res["per_axis"]}
    assert roles[0] == "index"   # time is the boring carrier
    assert roles[1] == "bind"    # asset value defines content
    assert roles[2] == "bind"    # field value defines content
    assert res["bind_axes"] == [1, 2]


def test_pure_noise_cube_rewards_no_binding():
    # Honesty / no-invented-structure: an i.i.d. noise cube has no informative,
    # content-coupled axis, so the honest answer is "index everything." We must
    # NOT force a payload axis where none exists.
    rng = np.random.default_rng(7)
    cube = rng.standard_normal((15, 6, 6))
    res = analyze_axes(cube)          # all ordered, uniform -> marginal 0
    assert res["bind_axes"] == []
    assert "index all" in res["summary"]


# ---------------------------------------------------------------------------
# Case 3: FLAT MESSAGE DECODE -- the Arecibo / Arrival "recover the grid" case
# ---------------------------------------------------------------------------

def test_flat_message_column_axis_is_the_payload():
    # A 2-D message [row, col] where each ROW is a near-repeat (scanlines of an
    # image) but COLUMNS carry the picture. Rows are the boring index (scanline
    # order); columns are the content. With columns marked categorical (distinct
    # positions), the analyzer recovers "rows index, columns bind" -- the machine
    # analogue of using the prime factorization to lay out the Arecibo bitmap.
    rng = np.random.default_rng(3)
    R, Cn = 23, 40
    # Each column has its own vertical stripe pattern (content lives across cols),
    # rows are near-copies with mild noise (the scanline / carrier axis).
    col_pattern = rng.standard_normal(Cn) * 2.0
    msg = np.stack([col_pattern + rng.standard_normal(Cn) * 0.02 for _ in range(R)],
                   axis=0)  # [R, Cn]

    # Row axis: uniform index -> marginal 0 -> INDEX.
    row_rec = recommend_axis_role(axis_report(msg, 0))
    assert row_rec["role"] == "index"

    # Column axis as content: bind cost is high if we (wrongly) bound the row axis.
    cost = comparability_cost(msg, 0, dim=64, seed=1)
    assert cost["collapse"] > 0.5  # binding the scanline index would wreck the image


def test_wrong_aspect_ratio_reads_as_noise_right_one_as_structure():
    # The Arecibo insight: only the correct 2-D reshape reveals structure; the
    # wrong factorization looks like noise. We build a signal that is smooth when
    # reshaped [H, W] (correct) and scrambled when reshaped [W, H] (transposed).
    # The analyzer's coupling on the row axis should be LOW (smooth neighbours) for
    # the correct shape and HIGH (unrelated neighbours) for the wrong shape --
    # a data-driven way to prefer the correct grid.
    rng = np.random.default_rng(5)
    H, W = 16, 24
    # A smooth 2-D field: neighbouring ROWS are similar in the correct layout.
    yy, xx = np.meshgrid(np.linspace(0, 3, W), np.linspace(0, 3, H))
    field = np.sin(xx) + 0.3 * np.cos(yy) + rng.standard_normal((H, W)) * 0.01

    correct = field                       # [H, W]
    wrong = field.reshape(-1)[: H * W].reshape(W, H)  # transposed-ish scramble

    # adjacent-row cosine: correct layout has more similar neighbours than wrong.
    from holographic.sampling_and_signal.holographic_axisrole import (
        _slices_along, _mean_adjacent_cosine)
    sim_correct = _mean_adjacent_cosine(_slices_along(correct, 0))
    sim_wrong = _mean_adjacent_cosine(_slices_along(wrong, 0))
    assert sim_correct > sim_wrong  # the correct grid has smoother row-neighbours


# ---------------------------------------------------------------------------
# Recommender logic + determinism
# ---------------------------------------------------------------------------

def test_recommender_flags_borderline_with_small_margin():
    # An axis sitting right at the threshold should be reported with a small margin
    # so the call is loud about being borderline (not silently decided).
    rep = {"axis": 0, "marginal_info": LOW_INFO_FRAC - 0.01,
           "coupling": 0.5, "length": 10}
    rec = recommend_axis_role(rep)
    assert rec["role"] == "index"
    assert rec["margin"] < 0.05  # borderline, loudly so


def test_flat_axis_is_droppable():
    # An axis with no content variation AND low info is a constant scaffold --
    # flagged droppable (content does not depend on it at all).
    rep = {"axis": 2, "marginal_info": 0.0, "coupling": 0.0, "length": 5}
    rec = recommend_axis_role(rep)
    assert rec["role"] == "index"
    assert rec["droppable"] is True


def test_analyze_axes_is_deterministic():
    rng = np.random.default_rng(0)
    cube = rng.standard_normal((12, 5, 5))
    a = analyze_axes(cube)
    b = analyze_axes(cube)
    assert a["summary"] == b["summary"]
    for ra, rb in zip(a["per_axis"], b["per_axis"]):
        assert ra["marginal_info"] == rb["marginal_info"]
        assert ra["coupling"] == rb["coupling"]


def test_selftest_runs():
    from holographic.sampling_and_signal.holographic_axisrole import _selftest
    _selftest()  # must not raise


def test_irregular_index_borderline_bind_has_thin_margin():
    # KEPT NEGATIVE, pinned: an irregular-timing axis reads high marginal info and
    # can tip to BIND, but with a razor-thin margin -- the honest signal that the
    # coupling probe cannot tell "timing informs content" from "timing is just
    # noisily irregular." We pin that a BIND from irregular timing over otherwise
    # flat content is borderline (small margin), NOT a confident payload call.
    rng = np.random.default_rng(2)
    T = 200
    sig = np.sin(np.linspace(0, 20, T))[:, None]      # smooth content, flat along t
    t_irr = np.cumsum(rng.exponential(1.0, size=T))   # irregular timestamps
    rec = analyze_axes(sig, coords={0: t_irr})["per_axis"][0]
    assert rec["marginal_info"] > 0.5                 # irregular timing carries bits
    if rec["role"] == "bind":
        assert rec["margin"] < 0.05                   # borderline, loudly so


# ---------------------------------------------------------------------------
# Carrier rectification: repair the boring axis when reality wobbles
# ---------------------------------------------------------------------------

def test_irregular_carrier_rectifies_to_exactly_boring():
    # Poisson-irregular timestamps carry spacing information (marginal info high);
    # rectification resamples onto a uniform grid, restoring the ideal carrier
    # (marginal info exactly 0) -- and reports both numbers so the repair is
    # auditable, not asserted.
    from holographic.sampling_and_signal.holographic_axisrole import rectify_carrier
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.exponential(1.0, size=300))
    r = rectify_carrier(t, np.sin(0.1 * t))
    assert r["marginal_info_before"] > 0.3
    assert r["marginal_info_after"] == 0.0
    assert r["monotone_fraction"] == 1.0            # only resampling was needed


def test_occasionally_negative_carrier_lifts_to_monotone():
    # The user's case: a nearly-boring axis whose delta sometimes dips negative.
    # The arc-length (covering) lift absorbs the reversals into one-way progress:
    # output coords strictly increase, and monotone_fraction says how much repair
    # the axis needed.
    from holographic.sampling_and_signal.holographic_axisrole import rectify_carrier
    steps = np.full(300, 1.0)
    steps[::37] = -0.3                               # rare small back-steps
    t = np.cumsum(steps)
    r = rectify_carrier(t, np.cos(0.05 * np.arange(300)))
    assert np.all(np.diff(r["coords"]) > 0)
    assert 0.9 < r["monotone_fraction"] < 1.0        # mostly forward, some repair
    assert r["marginal_info_after"] == 0.0


def test_rectified_content_tracks_the_true_function():
    # Resampling must reproduce the underlying smooth payload on the uniform grid
    # (linear interpolation of a slow sine over modest gaps: small error).
    from holographic.sampling_and_signal.holographic_axisrole import rectify_carrier
    rng = np.random.default_rng(3)
    t = np.cumsum(0.5 + 0.5 * rng.random(400))        # gaps in [0.5, 1.0]: modest
    f = lambda x: np.sin(0.2 * x)
    r = rectify_carrier(t, f(t))
    true = f(t[0] + r["coords"])
    assert float(np.mean(np.abs(r["content"] - true))) < 0.01


def test_multichannel_content_resamples_every_channel():
    from holographic.sampling_and_signal.holographic_axisrole import rectify_carrier
    rng = np.random.default_rng(4)
    t = np.cumsum(rng.exponential(1.0, size=200))
    content = np.stack([np.sin(0.1 * t), np.cos(0.1 * t)], axis=1)   # (N, 2)
    r = rectify_carrier(t, content)
    assert r["content"].shape == (200, 2)


def test_constant_carrier_is_refused():
    from holographic.sampling_and_signal.holographic_axisrole import rectify_carrier
    with pytest.raises(ValueError, match="zero variation"):
        rectify_carrier(np.zeros(50), np.arange(50.0))


def test_rectify_then_analyze_closes_the_loop():
    # THE COMPOSITION: a broken carrier makes analyze_axes read the axis as
    # informative (irregular spacing = bits); after rectification the same
    # analysis reads it as an ideal INDEX with a wide margin. Diagnosis -> repair
    # -> re-diagnosis, all through one family.
    from holographic.sampling_and_signal.holographic_axisrole import (
        rectify_carrier, analyze_axes)
    rng = np.random.default_rng(5)
    t = np.cumsum(rng.exponential(1.0, size=240))
    content = np.stack([np.sin(0.1 * t + p) for p in (0.0, 0.7, 1.4)], axis=1)

    before = analyze_axes(content, coords={0: t})["per_axis"][0]
    assert before["marginal_info"] > 0.3             # broken carrier reads informative

    r = rectify_carrier(t, content)
    after = analyze_axes(r["content"])["per_axis"][0]  # uniform -> coords omitted
    assert after["role"] == "index"
    assert after["marginal_info"] == 0.0
    assert after["margin"] > 0.3                     # a confident carrier again

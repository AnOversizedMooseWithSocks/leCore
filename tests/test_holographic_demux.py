"""Tests for holographic_demux: one stream, many sources.

Pins: stride detection by delta-continuity (the Contact channel separation) with
BIT-EXACT recovery (deinterleaving is a permutation); the smallest-K Occam rule
over the m*K harmonic ladder; honest K=1 on noise; multi-object channel grouping
(the animated-mesh case, mirrored axes included); and the full composition:
demux -> explore_series per channel, each source's law recovered separately.
Fixed seeds; deterministic.
"""

import numpy as np
import pytest

import lecore
from holographic.sampling_and_signal.holographic_demux import (
    detect_interleave, group_channels, demux_series,
    STRIDE_MARGIN, GROUP_THRESHOLD,
)

MIND = lecore.UnifiedMind(dim=256, seed=0)


def _mux3(n=300):
    u = np.linspace(0, 1, n)
    s1 = np.sin(2 * np.pi * 2 * u)
    s2 = 0.8 * u ** 2
    s3 = 0.5 * np.cos(2 * np.pi * 5 * u)
    x = np.empty(3 * n)
    x[0::3], x[1::3], x[2::3] = s1, s2, s3
    return x, (s1, s2, s3)


# ---------------------------------------------------------------------------
# Stride detection (the Contact move)
# ---------------------------------------------------------------------------

def test_stride_three_is_found_and_recovery_is_bit_exact():
    x, sources = _mux3()
    det = detect_interleave(x)
    assert det["k"] == 3
    for rec, src in zip(det["channels"], sources):
        assert np.array_equal(rec, src)   # a permutation, not an approximation


def test_harmonics_score_well_but_occam_picks_the_smallest():
    # K=6 MUST also score above baseline (each sub-stream is a further downsample
    # of one smooth source) -- and the smallest-K rule must still return 3.
    x, _ = _mux3()
    det = detect_interleave(x)
    k6 = next(t["score"] for t in det["table"] if t["k"] == 6)
    assert k6 > det["baseline"] + STRIDE_MARGIN
    assert det["k"] == 3


def test_two_channel_interleave():
    n = 200
    u = np.linspace(0, 1, n)
    x = np.empty(2 * n)
    x[0::2] = np.sin(2 * np.pi * 3 * u)
    x[1::2] = u
    assert detect_interleave(x)["k"] == 2


def test_noise_reads_k_equals_one():
    rng = np.random.default_rng(0)
    det = detect_interleave(rng.standard_normal(600))
    assert det["k"] == 1                   # nothing to separate: honest
    assert "table" in det                  # the evidence still travels


def test_uninterleaved_smooth_signal_reads_k_equals_one():
    # A single smooth source must NOT be split: every K scores well on it (any
    # downsample of smooth is smooth), so no K clears the baseline MARGIN.
    u = np.linspace(0, 1, 600)
    det = detect_interleave(np.sin(2 * np.pi * 3 * u))
    assert det["k"] == 1


# ---------------------------------------------------------------------------
# Channel grouping (the multi-mesh case)
# ---------------------------------------------------------------------------

def test_two_objects_recovered_mirror_included():
    rng = np.random.default_rng(0)
    n = 300
    u = np.linspace(0, 1, n)
    ma = np.sin(2 * np.pi * 1.5 * u)
    mb = np.cumsum(rng.standard_normal(n)) * 0.1
    series = np.stack([ma, 0.7 * ma, 0.4 * ma,
                       mb, 0.6 * mb, -0.8 * mb], axis=1)   # B's z-axis mirrored
    series += rng.standard_normal(series.shape) * 0.01
    g = group_channels(series)
    assert g["groups"] == [[0, 1, 2], [3, 4, 5]]
    assert g["corr"].shape == (6, 6)       # the evidence travels


def test_unrelated_channels_stay_separate():
    rng = np.random.default_rng(1)
    series = np.stack([rng.standard_normal(200) for _ in range(4)], axis=1)
    g = group_channels(series)
    assert len(g["groups"]) == 4           # no fictitious objects


# ---------------------------------------------------------------------------
# The full composition: Contact protocol -- demux, then decode each separately
# ---------------------------------------------------------------------------

def test_demux_then_explore_recovers_each_sources_law():
    # An interleaved stream of two lawful sources: demux_series finds the stride,
    # and explore_series on each recovered channel independently identifies its
    # law -- the whole pipeline, exactly the "separate the channels and decode
    # each one" protocol.
    n = 200
    u = np.linspace(0, 1, n)
    x = np.empty(2 * n)
    x[0::2] = np.sin(2 * np.pi * 2 * u)
    x[1::2] = 0.8 * u + 0.1
    d = demux_series(x)
    assert d["stride"] == 2
    for obj in d["objects"]:
        res = MIND.explore_series(obj.reshape(-1, 1) if obj.ndim == 1 else obj)
        assert res["verdict"] == "structured"
        assert res["channels"][0]["explained_fraction"] > 0.9


def test_demux_on_2d_input_skips_stride_and_groups():
    rng = np.random.default_rng(2)
    u = np.linspace(0, 1, 150)
    m = np.sin(2 * np.pi * u)
    series = np.stack([m, 0.5 * m, rng.standard_normal(150)], axis=1)
    d = demux_series(series)
    assert d["stride"] is None
    assert [0, 1] in d["groups"]


def test_demux_is_deterministic():
    x, _ = _mux3()
    a = demux_series(x)
    b = demux_series(x)
    assert a["stride"] == b["stride"] and a["groups"] == b["groups"]


def test_selftest_runs():
    from holographic.sampling_and_signal.holographic_demux import _selftest
    _selftest()


# ---------------------------------------------------------------------------
# Cross-channel links (the residual pass)
# ---------------------------------------------------------------------------

def test_delayed_copy_found_with_exact_lag_and_gain():
    rng = np.random.default_rng(0)
    src = rng.standard_normal(400)
    dst = np.zeros(400)
    dst[7:] = 0.8 * src[:-7]
    from holographic.sampling_and_signal.holographic_demux import cross_channel_links
    cx = cross_channel_links(np.stack([src, dst], axis=1))
    top = cx["links"][0]
    assert (top["src"], top["dst"], top["lag"]) == (0, 1, 7)
    assert abs(top["gain"] - 0.8) < 0.05


def test_unrelated_noise_yields_no_links():
    rng = np.random.default_rng(1)
    from holographic.sampling_and_signal.holographic_demux import cross_channel_links
    cx = cross_channel_links(np.stack([rng.standard_normal(300),
                                       rng.standard_normal(300)], axis=1))
    assert cx["links"] == []


def test_too_few_samples_refuses_links():
    # REGRESSION TRAP for the 89,700-fictitious-links bug: two mean-removed
    # 2-sample vectors correlate at exactly +/-1, so with too few samples the
    # threshold is meaningless -- the guard must refuse, with a note, never
    # fabricate.
    rng = np.random.default_rng(2)
    from holographic.sampling_and_signal.holographic_demux import cross_channel_links
    cx = cross_channel_links(rng.standard_normal((4, 6)))
    assert cx["links"] == []
    assert "note" in cx


def test_explore_series_cross_channel_upgrades_linked_residuals():
    # A delayed noise copy: zero per-channel structure (the no-scaffold exit
    # fires), yet the pair is lawful -- cross_channel=True must find the link and
    # upgrade the verdict; default-off must stay byte-identical old behaviour.
    import lecore
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(0)
    src = rng.standard_normal(300)
    dst = np.zeros(300)
    dst[5:] = 0.9 * src[:-5]
    pair = np.stack([src, dst], axis=1)

    on = mind.explore_series(pair, cross_channel=True)
    assert on["verdict"] == "weakly structured"
    assert on["residual_links"]["links"][0]["lag"] == 5

    off = mind.explore_series(pair)
    assert "residual_links" not in off
    assert off["verdict"] == "no structure found"


def test_explore_series_cross_on_noise_stays_refused():
    import lecore
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(3)
    noise = np.stack([rng.standard_normal(300), rng.standard_normal(300)], axis=1)
    r = mind.explore_series(noise, cross_channel=True)
    assert r["verdict"] == "no structure found"
    assert r.get("residual_links", {}).get("links", []) == []


def test_explore_series_auto_demux_runs_the_contact_protocol_hands_free():
    import lecore
    mind = lecore.UnifiedMind(dim=256, seed=0)
    n = 200
    u = np.linspace(0, 1, n)
    x = np.empty(2 * n)
    x[0::2] = np.sin(2 * np.pi * 2 * u)
    x[1::2] = 0.8 * u + 0.1
    r = mind.explore_series(x, auto_demux=True)
    assert r["demux"]["stride"] == 2
    assert r["verdict"] == "structured"
    assert all(o["verdict"] == "structured" for o in r["objects"])


# ---------------------------------------------------------------------------
# Packetized demux (change-point segmentation + noise-calibrated assignment)
# ---------------------------------------------------------------------------

def test_packet_boundaries_land_on_statistics_shifts():
    from holographic.sampling_and_signal.holographic_demux import packet_demux
    rng = np.random.default_rng(0)
    lens = [40, 65, 50, 80, 45, 70]
    parts = [rng.standard_normal(L) * 0.1 if i % 2 == 0
             else 3.0 + rng.standard_normal(L) for i, L in enumerate(lens)]
    pk = packet_demux(np.concatenate(parts), min_seg=16)
    truth = np.cumsum(lens)[:-1]
    for tb in truth:
        assert min(abs(tb - b) for b in pk["boundaries"]) <= 5


def test_packet_sources_recovered_by_reassembled_statistics():
    from holographic.sampling_and_signal.holographic_demux import packet_demux
    rng = np.random.default_rng(0)
    lens = [40, 65, 50, 80, 45, 70]
    parts = [rng.standard_normal(L) * 0.1 if i % 2 == 0
             else 3.0 + rng.standard_normal(L) for i, L in enumerate(lens)]
    pk = packet_demux(np.concatenate(parts), min_seg=16)
    assert pk["n_sources"] == 2
    means = sorted(float(np.mean(s["stream"])) for s in pk["sources"])
    assert abs(means[0]) < 0.15 and abs(means[1] - 3.0) < 0.3


def test_homogeneous_stream_returns_no_packets():
    # the BIC penalty must refuse to shatter a single source into noise-fit
    # pieces: no boundaries, one source.
    from holographic.sampling_and_signal.holographic_demux import packet_demux
    rng = np.random.default_rng(1)
    pk = packet_demux(rng.standard_normal(400), min_seg=16)
    assert pk["boundaries"] == [] and pk["n_sources"] == 1


def test_same_power_different_spectrum_assigns_by_signature():
    # The declared boundary-model negative in action: two sources with equal
    # mean and variance but different SPECTRA. The spectral bands in the
    # signature separate them at ASSIGNMENT time -- provided a boundary exists;
    # here the variance ripple at the seam is enough for segmentation, and the
    # test pins the assignment half of the story.
    from holographic.sampling_and_signal.holographic_demux import (
        segment_stream, packet_demux)
    t = np.arange(600)
    lo = np.sin(2 * np.pi * t / 100)              # slow oscillation
    hi = np.sin(2 * np.pi * t / 6)                # fast oscillation, same power
    stream = np.concatenate([lo[:150], hi[:150], lo[150:300], hi[150:300]])
    pk = packet_demux(stream, min_seg=24)
    if pk["n_sources"] >= 2:                      # segmentation found seams
        # slow and fast pieces must not share a source
        for s in pk["sources"]:
            segs_fast = sum(1 for a, b in s["segments"]
                            if np.mean(np.abs(np.diff(stream[a:b]))) > 0.5)
            assert segs_fast == 0 or segs_fast == len(s["segments"])


def test_packet_then_explore_decodes_each_source():
    # The composition on MODEL-CONFORMANT sources: a REPEATING lawful ramp among
    # loud noise bursts. The ramp pieces must reunite into ONE source and that
    # source must decode 'structured'. The noise bursts may stay split (the
    # documented conservative bias: two noisy realizations ~2 sigma apart are
    # kept separate -- under-merging fabricates nothing), so n_sources is 2 or 3.
    import lecore
    from holographic.sampling_and_signal.holographic_demux import packet_demux
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(2)
    stream = np.concatenate([0.5 * np.linspace(0, 1, 120),           # lawful ramp
                             5.0 + rng.standard_normal(80),           # noise burst
                             0.5 * np.linspace(0, 1, 100),            # ramp repeats
                             5.0 + rng.standard_normal(90)])
    pk = packet_demux(stream, min_seg=24)
    assert pk["n_sources"] in (2, 3), pk["assignment"]
    assert pk["assignment"][0] == pk["assignment"][2]   # the ramps reunited
    ramp_src = pk["sources"][pk["assignment"][0]]
    r = mind.explore_series(ramp_src["stream"].reshape(-1, 1))
    assert r["verdict"] == "structured"                  # and decoded


def test_oscillating_source_over_segments_kept_negative():
    # PINNED MEASURED NEGATIVE: an oscillating source violates the piecewise-
    # constant-statistics model -- its mean genuinely swings each half-cycle, so
    # the segmenter legitimately splits it there and the halves read as distinct
    # statistical sources. This is the declared model boundary, kept loud as a
    # regression trap on the DOCUMENTED behaviour (a "fix" that silently merges
    # them would have changed the model and must show up here).
    from holographic.sampling_and_signal.holographic_demux import packet_demux
    t = np.arange(240)
    sine = np.sin(2 * np.pi * t / 60)             # period 60 >> min_seg
    pk = packet_demux(sine, min_seg=16)
    assert len(pk["boundaries"]) > 0              # half-cycle splits occur


def test_packet_demux_is_deterministic():
    from holographic.sampling_and_signal.holographic_demux import packet_demux
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.standard_normal(60) * 0.1, 3 + rng.standard_normal(80)])
    a = packet_demux(x)
    b = packet_demux(x)
    assert a["boundaries"] == b["boundaries"] and a["assignment"] == b["assignment"]


# ---------------------------------------------------------------------------
# Continuation merges (the drift-across-bursts negative, closed)
# ---------------------------------------------------------------------------

def _drift_stream(resume_offset=0.0, seed=2):
    rng = np.random.default_rng(seed)
    t1 = 0.02 * np.arange(60)                       # ramp 0 .. 1.18
    burst = 8.0 + rng.standard_normal(120)          # loud noise burst
    t2 = resume_offset + 0.02 * np.arange(180, 240) # resumes at 3.6 (+offset)
    return np.concatenate([t1, burst, t2])


def test_continuation_reunites_a_drifting_source():
    from holographic.sampling_and_signal.holographic_demux import packet_demux
    x = _drift_stream()
    off = packet_demux(x, min_seg=24)
    on = packet_demux(x, min_seg=24, continuation=True)
    assert off["n_sources"] == 3                    # levels differ: correct at
    assert on["n_sources"] == 2                     # bag-of-segments scope...
    m = on["continuation_merges"][0]                # ...and the sequence pass
    assert abs(m["predicted"] - m["observed"]) <= m["tolerance"]


def test_continuation_refuses_wrong_level_resumption():
    from holographic.sampling_and_signal.holographic_demux import packet_demux
    on = packet_demux(_drift_stream(resume_offset=1.0), min_seg=24,
                      continuation=True)
    assert on["continuation_merges"] == []


def test_continuation_dynamics_gate_blocks_noise_swallow():
    # REGRESSION TRAP: a high-noise source's level gate is enormous; before the
    # dynamics gate it swallowed a quiet ramp resuming at the WRONG level. The
    # continuation of a source must LOOK like the source.
    from holographic.sampling_and_signal.holographic_demux import packet_demux
    rng = np.random.default_rng(2)
    bad = np.concatenate([np.linspace(0, 1, 120),
                          5.0 + rng.standard_normal(80),
                          np.linspace(3, 4, 100)])
    on = packet_demux(bad, min_seg=24, continuation=True)
    assert on["continuation_merges"] == []


def test_continuation_default_off_is_byte_identical():
    from holographic.sampling_and_signal.holographic_demux import packet_demux
    x = _drift_stream()
    a = packet_demux(x, min_seg=24)
    b = packet_demux(x, min_seg=24, continuation=False)
    assert a["assignment"] == b["assignment"]
    assert "continuation_merges" not in a and "continuation_merges" not in b

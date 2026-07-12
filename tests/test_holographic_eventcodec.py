"""X7 -- the physics event codec (Box3D lesson B8).

THE BAR (from the backlog): beat delta-compression of raw state. It does, by 13.7x, losslessly -- but not by the
mechanism the backlog names, and the tests below pin the difference:

  * the win is event SPARSITY (663 events replace 9,600 state rows), not an edit codebook;
  * a quantized impulse codebook adds ~2x and is LOSSY, and the loss AMPLIFIES because events decide which events
    happen next -- at q=0.1 the reconstruction leaves the box;
  * `DeltaChain`, the obvious reuse, LOSES on this workload: it skips unchanged rows, and a sim moves every body
    every frame.
"""

import zlib

import numpy as np
import pytest

from holographic.simulation_and_physics.holographic_eventcodec import (
    EventTrace, bouncing_box_trace, replay_bouncing_box, compression_report)


def test_selftest_runs():
    from holographic.simulation_and_physics import holographic_eventcodec as mod
    mod._selftest()


def test_the_trace_is_deterministic_and_has_the_expected_shape():
    T1, e1 = bouncing_box_trace()
    T2, e2 = bouncing_box_trace()
    assert T1.shape == (600, 16, 6)
    assert np.array_equal(T1, T2) and np.array_equal(e1.impulses, e2.impulses)
    assert len(e1.impulses) == 663                                # the events really are sparse
    T3, _ = bouncing_box_trace(seed=1)
    assert not np.array_equal(T1, T3)                             # ... and the seed actually matters


def test_replay_is_bit_identical_not_merely_close():
    # This is the correctness contract. A contact sequence is a chaotic map, so "close" is a different simulation.
    T, ev = bouncing_box_trace()
    assert np.array_equal(replay_bouncing_box(ev), T)
    assert np.abs(replay_bouncing_box(ev) - T).max() == 0.0


def test_the_blob_round_trips_and_a_bad_magic_is_refused():
    T, ev = bouncing_box_trace(n=8, frames=120)
    back = EventTrace.decode(ev.encode())
    assert np.array_equal(back.base, ev.base)
    assert np.array_equal(back.keys, ev.keys)
    assert np.array_equal(back.impulses, ev.impulses)
    assert back.frames == ev.frames
    assert np.array_equal(replay_bouncing_box(back), T)           # decode -> replay is STILL bit-exact

    with pytest.raises(ValueError):
        EventTrace.decode(zlib.compress(b"NOPE" + b"\x00" * 40))  # a stale format must raise, not misread


def test_event_trace_validates_its_inputs():
    T, ev = bouncing_box_trace(n=4, frames=30)
    with pytest.raises(ValueError):
        EventTrace(ev.base, ev.keys, ev.impulses[:-1], ev.frames)  # one impulse per key


def test_the_bar_beat_delta_compression_of_raw_state():
    T, ev = bouncing_box_trace()
    rep = compression_report(T, ev)

    # the baselines, strongest last -- the codec must beat the strongest, not the weakest
    assert rep["zlib_frame_deltas"] < rep["zlib_raw"] < rep["raw"]
    assert rep["event_codec"] < rep["zlib_frame_deltas"]           # THE BAR
    assert rep["ratio_vs_frame_deltas"] > 10.0
    assert rep["event_codec"] < 8000 and rep["zlib_frame_deltas"] > 80_000


def test_the_win_is_sparsity_not_a_codebook():
    # 663 events stand in for 600 x 16 = 9,600 state rows. THAT ratio is the compression, and it needs no codebook.
    T, ev = bouncing_box_trace()
    rep = compression_report(T, ev)
    assert rep["rows"] == 9600 and rep["events"] == 663
    assert rep["rows"] > 14 * rep["events"]

    # a quantized impulse codebook buys under ~2.5x on top -- an order of magnitude less than the sparsity itself
    q = 0.01
    coarse = EventTrace(ev.base, ev.keys, np.round(ev.impulses / q) * q, ev.frames)
    assert coarse.nbytes() < ev.nbytes()
    assert ev.nbytes() / coarse.nbytes() < 2.5
    assert rep["ratio_vs_frame_deltas"] > 5 * (ev.nbytes() / coarse.nbytes())


@pytest.mark.parametrize("q,floor", [(1e-3, 1e-3), (1e-2, 1e-2), (1e-1, 1.0)])
def test_kept_negative_a_lossy_impulse_codebook_wrecks_the_trajectory(q, floor):
    # An event is not a sample: it decides WHICH events happen next. So quantization error does not stay small.
    # Measured final-frame error in a box of half-extent 2.0: q=1e-3 -> 5.7e-02, 1e-2 -> 4.4e-01, 1e-1 -> 4.47.
    T, ev = bouncing_box_trace()
    coarse = EventTrace(ev.base, ev.keys, np.round(ev.impulses / q) * q, ev.frames)
    R = replay_bouncing_box(coarse)
    assert np.abs(R[-1] - T[-1]).max() > floor
    assert np.abs(R[0] - T[0]).max() == 0.0                       # the base is exact; the divergence is dynamic

    # the error GROWS: it is amplification, not a fixed offset
    errs = np.abs(R - T).max(axis=(1, 2))
    assert errs[-1] > 5 * errs[100]


def test_kept_negative_deltachain_is_the_wrong_tool_for_a_dense_mutation_trace():
    # The obvious reuse, measured before being rejected. DeltaChain skips rows that did not change; a sim moves
    # every body every frame, so it stores every row PLUS index bookkeeping and comes out larger than the raw array.
    T, ev = bouncing_box_trace()
    rep = compression_report(T, ev)
    assert rep["deltachain"] > rep["raw"]                          # a LOSS, not a saving
    assert rep["deltachain"] > 10 * rep["event_codec"]

    # ... and it is not a tuning problem: DeltaChain is right for SPARSE mutation. Prove it on that workload.
    from holographic.agents_and_reasoning.holographic_deltachain import DeltaChain
    base = np.zeros((64, 8))
    dc = DeltaChain(base)
    for f in range(50):
        chunk = base.copy()
        chunk[f % 64] = float(f)                                   # exactly one row changes per chunk
        dc.append(chunk)
    assert dc.memory_bytes() < dc.full_bytes() / 5                 # on sparse mutation it wins handily


def test_the_report_carries_its_own_baselines_so_an_agent_can_rerun_it():
    T, ev = bouncing_box_trace(n=8, frames=200)
    rep = compression_report(T, ev)
    for k in ("raw", "zlib_raw", "zlib_frame_deltas", "deltachain", "event_codec",
              "ratio_vs_frame_deltas", "events", "rows"):
        assert k in rep
    assert rep["ratio_vs_frame_deltas"] == pytest.approx(rep["zlib_frame_deltas"] / rep["event_codec"])


def test_the_codec_is_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    trace, ev = m.record_physics_trace(n=8, frames=200)
    assert np.array_equal(m.replay_physics_trace(ev), trace)
    rep = m.physics_compression_report(trace, ev)
    assert rep["event_codec"] < rep["zlib_frame_deltas"]
    assert "Physics event codec" in str(m.find_capability("compress a physics simulation trace")[:3])


def test_cross_faculty_a_sleeping_island_generates_no_events():
    # X7 meets X3: the codec's compression IS the sleep probe's claim, seen from the other side. An island at rest
    # emits no events, so it costs the codec nothing -- "sleep" and "incompressible" are the same fact.
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    # gravity=0 alone is NOT enough -- the bodies keep their initial velocity and still hit walls (10 events,
    # measured, when I first wrote this). An island truly at rest needs speed=0 too. The test said so before I did.
    trace, ev = m.record_physics_trace(n=4, frames=300, gravity=0.0, speed=0.0)
    assert len(ev.impulses) == 0                                       # ... so no events at all
    assert np.array_equal(m.replay_physics_trace(ev), trace)           # and replay is still exact
    rep = m.physics_compression_report(trace, ev)
    # NB: with nothing happening, the DELTA baseline is also tiny (an all-zero difference array compresses to
    # nothing), so the codec's edge here is small in absolute terms. The invariant is that it still wins, and
    # that its cost is the base state alone -- not that the ratio grows. Asserting a big ratio here would have
    # been asserting an artifact of zlib on zeros.
    assert rep["events"] == 0 and rep["event_codec"] < rep["zlib_frame_deltas"]
    assert rep["event_codec"] < rep["raw"] / 100


def test_the_law_travels_with_the_data_not_just_the_events():
    # BUG FIX, pinned. The first EventTrace stored base + events but NOT dt/gravity/box, so a decoder replayed
    # with its own defaults. Encoding a zero-gravity trace and replaying it under the default -9.81 reconstructed
    # a completely different simulation, silently -- fatal for the "sync physics over the network" claim.
    # A trace is base + events + THE LAW THAT CONNECTS THEM.
    T, ev = bouncing_box_trace(n=4, frames=120, gravity=0.0, speed=1.0, box=1.0, dt=1 / 90.0)
    assert (ev.dt, ev.gravity, ev.box) == (1 / 90.0, 0.0, 1.0)

    back = EventTrace.decode(ev.encode())
    assert (back.dt, back.gravity, back.box) == (ev.dt, ev.gravity, ev.box)
    assert np.array_equal(replay_bouncing_box(back), T)          # replays under the RECORDED law, not the default

    # and an explicit override still works -- it just must not be the silent default
    wrong = replay_bouncing_box(back, gravity=-9.81)
    assert not np.array_equal(wrong, T)


def test_a_truncated_blob_is_refused_not_a_struct_error():
    with pytest.raises(ValueError):
        EventTrace.decode(zlib.compress(b"short"))

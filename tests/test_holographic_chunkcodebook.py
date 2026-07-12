"""R1 -- learned chunk codebooks by iterated pair promotion, and R3's "one codebook family".

The claims under test, each with its baseline:
  1. the round-trip is LOSSLESS and the codebook DETERMINISTIC (it is useless to R2/W5/DL8 otherwise)
  2. structure produces deep chunks; a uniform control stalls at depth 2  -- the recursion dividend's precondition
  3. KEPT NEGATIVE: this is not a byte compressor. zlib beats it on the very stream it "compresses 4.3x".
"""

import numpy as np
import pytest

from holographic.agents_and_reasoning.holographic_chunkcodebook import (
    ChunkCodebook, learn_chunks, structure_score, byte_report, workflow_stream, uniform_stream)


def test_selftest_runs():
    from holographic.agents_and_reasoning import holographic_chunkcodebook as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# 1. lossless + deterministic
# ---------------------------------------------------------------------------------------------------------

def test_round_trip_is_lossless_on_structure_and_on_noise():
    for s in (workflow_stream(), uniform_stream()):
        cb = learn_chunks(s)
        assert cb.decode(cb.encode(s)) == s


def test_round_trip_edge_cases():
    assert learn_chunks([]).encode([]) == []
    assert learn_chunks([]).decode([]) == []
    one = learn_chunks([7])
    assert len(one) == 0 and one.encode([7]) == [7] and one.decode([7]) == [7]
    # a stream of a single repeated symbol collapses into a deep chunk and still round-trips
    rep = [3] * 64
    cb = learn_chunks(rep)
    assert cb.decode(cb.encode(rep)) == rep
    assert cb.stats(rep)["max_depth"] >= 32


def test_the_codebook_is_deterministic_and_not_at_the_mercy_of_dict_order():
    s = workflow_stream()
    a, b = learn_chunks(s), learn_chunks(s)
    assert a.to_dict() == b.to_dict()

    # Ties on count must break on the PAIR, not on the order the pairs were first SEEN. Two pairs tie at 2 here;
    # the smallest must win regardless of which appears first in the stream.
    #   (NB: reversing the stream is not the right test -- it changes the pairs themselves, from (1,2)/(3,4) to
    #    (2,1)/(4,3). I wrote that first and it "failed" while the tie-break was working perfectly.)
    assert learn_chunks([1, 2, 1, 2, 3, 4, 3, 4], max_merges=1).merges[0][0] == (1, 2)
    assert learn_chunks([3, 4, 3, 4, 1, 2, 1, 2], max_merges=1).merges[0][0] == (1, 2)   # order seen: irrelevant
    assert learn_chunks([9, 8, 9, 8, 1, 2, 1, 2], max_merges=1).merges[0][0] == (1, 2)


def test_the_codebook_is_plain_data_and_survives_serialization():
    s = workflow_stream(n_workflows=200)
    cb = learn_chunks(s)
    d = cb.to_dict()
    import json
    back = ChunkCodebook.from_dict(json.loads(json.dumps(d)))   # must survive a JSON boundary
    assert back.encode(s) == cb.encode(s)
    assert back.decode(back.encode(s)) == s


def test_encoding_replays_the_merges_in_learning_order():
    # A later merge can consume a token an earlier one produced. Applying them out of order gives a different
    # (and worse) tokenization -- pinned, because "the order IS the codebook" is easy to lose in a refactor.
    s = workflow_stream(n_workflows=300)
    cb = learn_chunks(s, max_merges=50)
    shuffled = ChunkCodebook(list(reversed(cb.merges)), cb.depth)
    assert shuffled.encode(s) != cb.encode(s)
    assert len(cb.encode(s)) <= len(shuffled.encode(s))


# ---------------------------------------------------------------------------------------------------------
# 2. the separation: structure vs noise
# ---------------------------------------------------------------------------------------------------------

def test_structure_produces_deep_chunks_and_noise_stalls_at_depth_two():
    s, u = workflow_stream(), uniform_stream()
    st = learn_chunks(s).stats(s)
    us = learn_chunks(u).stats(u)

    assert st["token_ratio"] > 3.0                 # measured 4.31
    assert us["token_ratio"] < 1.6                 # measured 1.34
    assert st["max_depth"] >= 8                    # measured 16
    assert us["max_depth"] <= 2                    # noise never nests: THE condition on the recursion dividend
    assert st["mean_depth"] > 3.0 > us["mean_depth"]


def test_structure_score_is_the_one_number_gate():
    assert structure_score(workflow_stream()) > 3.0
    assert structure_score(uniform_stream()) < 1.6
    assert structure_score([]) == 0.0
    assert structure_score([5] * 32) > 8.0         # maximal structure


def test_the_dividend_tracks_the_reuse_rate_monotonically():
    # The claim is not "BPE finds pairs" (it always will) -- it is that the dividend is PAID BY REUSE.
    scores = [structure_score(workflow_stream(n_workflows=400, reuse=r, seed=1))
              for r in (0.0, 0.3, 0.6, 0.9)]
    assert scores == sorted(scores), scores
    assert scores[-1] > 2 * scores[0]


def test_min_count_halts_on_noise_rather_than_inventing_chunks():
    u = uniform_stream()
    assert len(learn_chunks(u, min_count=10_000)) == 0     # nothing repeats often enough: learn nothing
    assert learn_chunks(u, min_count=10_000).encode(u) == u


# ---------------------------------------------------------------------------------------------------------
# 3. KEPT NEGATIVE: not a byte compressor
# ---------------------------------------------------------------------------------------------------------

def test_kept_negative_this_loses_to_zlib_as_a_byte_codec():
    # The 4.3x is a TOKEN-count ratio. Sold as compression it would be a false claim, and the report says so.
    s = workflow_stream()
    cb = learn_chunks(s)
    rep = byte_report(s, cb)
    assert rep["zlib_raw"] < rep["bpe_bytes"]              # codebook + raw tokens: bigger than zlib
    assert rep["zlib_raw"] < rep["zlib_bpe_bytes"]         # even after zlib-ing the tokens
    assert rep["beats_zlib"] is False

    # ... while the TOKEN ratio really is 4.3x. Both are true; only one is a compression claim.
    assert cb.stats(s)["token_ratio"] > 4.0


def test_the_byte_report_travels_with_the_capability():
    s = workflow_stream(n_workflows=200)
    rep = byte_report(s, learn_chunks(s))
    for k in ("raw", "zlib_raw", "bpe_bytes", "zlib_bpe_bytes", "beats_zlib"):
        assert k in rep


# ---------------------------------------------------------------------------------------------------------
# wiring + cross-faculty
# ---------------------------------------------------------------------------------------------------------

def test_fully_wired_to_the_mind_as_plain_data():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    s = workflow_stream(n_workflows=300)

    cb = m.learn_chunks(s)
    assert isinstance(cb, dict) and "merges" in cb and "depth" in cb   # plain data: crosses HTTP
    toks = m.chunk_encode(s, cb)
    assert m.chunk_decode(toks, cb) == s                                # lossless through the mind
    assert m.chunk_stats(s, cb)["token_ratio"] > 3.0
    assert m.structure_score(s) > 3.0 > m.structure_score(uniform_stream())
    assert m.chunk_byte_report(s, cb)["beats_zlib"] is False

    live = m.chunk_codebook(cb)                                         # the in-process twin
    assert live.encode(s) == toks
    assert "Learned chunk codebook" in str(m.find_capability("byte pair encoding")[:3])


def test_cross_faculty_the_probe_reads_a_real_physics_event_stream():
    # R1 meets X7. The event codec's key stream (body, axis, wall) is a real trace, not a synthetic one. Does it
    # have reusable structure? Measure rather than assume -- a body bouncing in a corner repeats its wall pattern.
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    _, ev = m.record_physics_trace(n=8, frames=600)
    # pack (body, axis, wall) into one symbol per event, in event order
    stream = [int(b) * 6 + int(k) * 2 + int(w) for (_f, b, k, w) in ev.keys]
    assert len(stream) > 100

    score = m.structure_score(stream)
    cb = m.learn_chunks(stream)
    assert m.chunk_decode(m.chunk_encode(stream, cb), cb) == stream     # lossless on real data too
    # a real contact stream sits BETWEEN the synthetic extremes -- it has structure, but far less than a
    # workflow trace. Reported, not asserted as a headline: this is what "measure the stream you have" means.
    assert 1.0 < score < m.structure_score(workflow_stream())

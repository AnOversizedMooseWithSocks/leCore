"""R2 -- recursive factoring over learned chunk levels, and R3's one-codebook-family made real.

Claims under test, each against the strongest honest baseline (the flat resonator, on the SAME composites):

  1. the flat resonator is a CLIFF, not a slope: it works to depth 4 and is gone at depth 5.
  2. with PROMOTED chunks a depth-8 composite factors, where flat scores exactly zero -- and faster.
  3. below the cliff recursion is a modest gain at real cost. Said out loud, not buried.
  4. the verify gate REFUSES an unexpressible composite rather than guessing.
  5. MAP binding is self-inverse, so "correct" and "minimal" are different things (reduce_involution).

Sizes are kept small so the suite stays fast; the headline numbers live in the module docstring, measured at
D=4096 with a 32-symbol vocabulary.
"""

import itertools

import numpy as np
import pytest

from holographic.misc.holographic_resonator import (
    ResonatorNetwork, map_codebook, map_bind,
    chunk_vector, level_codebook, available_levels, recursive_factor, reduce_involution)
from holographic.agents_and_reasoning.holographic_chunkcodebook import ChunkCodebook


D, V = 2048, 12


def _vocab():
    return map_codebook(V, D, seed=0)


def _two_level_codebook():
    """Six DISJOINT pairs promoted to three quads -- the shape `learn_chunks` produces from a structured stream.
    Disjoint on purpose: overlapping pairs cancel under the MAP involution."""
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    merges = [((a, b), V + i) for i, (a, b) in enumerate(pairs)]
    depth = {t: 1 for t in range(V)}
    for (a, b), nid in merges:
        depth[nid] = depth[a] + depth[b]
    for i, (a, b) in enumerate([(V + 0, V + 1), (V + 2, V + 3), (V + 4, V + 5)]):
        nid = V + len(pairs) + i
        merges.append(((a, b), nid))
        depth[nid] = depth[a] + depth[b]
    return ChunkCodebook(merges, depth)


def test_selftest_runs():
    from holographic.misc import holographic_resonator as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# the cliff, and crossing it
# ---------------------------------------------------------------------------------------------------------

def _flat_solve(vocab, n_sym, depth, seed, restarts=6, iters=200):
    r = np.random.default_rng(seed)
    idx = sorted(int(x) for x in r.choice(n_sym, depth, replace=False))
    c = map_bind(*[vocab[i] for i in idx])
    res = ResonatorNetwork([vocab] * depth).factor(c, restarts=restarts, iters=iters)
    return res["solved"] and sorted(int(x) for x in res["factors"]) == idx


def test_the_flat_resonator_falls_off_a_cliff_not_a_slope():
    # THE BASELINE, measured here rather than taken from the backlog (which reports 36.7% at depth 4; this code
    # scores far better than that, so the backlog's number would have been a strawman baseline).
    # NB the cliff needs the REAL vocabulary: at V=12 the depth-6 search space is only 3e6 and the flat resonator
    # still lands it sometimes. Use V=32.
    big_v, big_d = 32, 4096
    vocab = map_codebook(big_v, big_d, seed=0)
    assert sum(_flat_solve(vocab, big_v, 2, s) for s in range(5)) >= 4     # depth 2 is easy
    assert sum(_flat_solve(vocab, big_v, 5, s) for s in range(5)) == 0     # depth 5 is not "hard": it is gone


def test_the_cliff_is_set_by_the_search_space_not_by_the_depth():
    # A correction to the module's own first draft, pinned so it cannot drift back. Depth 6 is fatal at V=32
    # (V^d = 1.1e9) and survivable at V=12 (V^d = 3.0e6). "Past the cliff" is a search-space budget, not a depth.
    small = map_codebook(12, 2048, seed=0)
    big = map_codebook(32, 4096, seed=0)
    assert sum(_flat_solve(small, 12, 6, s, restarts=20, iters=400) for s in range(5)) >= 2
    assert sum(_flat_solve(big, 32, 6, s) for s in range(4)) == 0


def test_recursive_factoring_crosses_the_cliff_where_flat_scores_zero():
    vocab = _vocab()
    cb = _two_level_codebook()
    _, ids4 = level_codebook(cb, vocab, 4)

    ok = 0
    for a, b in itertools.combinations(range(len(ids4)), 2):
        truth = reduce_involution(cb.decode([ids4[a]]) + cb.decode([ids4[b]]))
        assert len(truth) == 8                                       # disjoint quads: a real depth-8 composite
        comp = map_bind(chunk_vector(ids4[a], cb, vocab), chunk_vector(ids4[b], cb, vocab))

        got = recursive_factor(comp, cb, vocab, restarts=8, iters=200)
        ok += got["solved"] and got["verified"] and got["leaves"] == truth
        assert got["level"] in (4, None)                             # solved at the chunk level, not by luck below

        # the SAME composite, flat over the base vocabulary: zero
        flat = ResonatorNetwork([vocab] * 8).factor(comp, restarts=4, iters=150)
        assert not (flat["solved"] and sorted(int(x) for x in flat["factors"]) == truth)

    assert ok == 3                                                   # every pairing of the three quads


def test_the_search_space_is_smaller_at_the_chunk_level_which_is_why_it_is_faster():
    vocab = _vocab()
    cb = _two_level_codebook()
    _, ids4 = level_codebook(cb, vocab, 4)
    comp = map_bind(chunk_vector(ids4[0], cb, vocab), chunk_vector(ids4[1], cb, vocab))
    got = recursive_factor(comp, cb, vocab, restarts=8, iters=200)
    assert got["solved"]
    assert got["search_space"] == len(ids4) ** 2                     # 3^2 = 9, versus V^8 for the flat problem
    assert got["search_space"] < V ** 8


def test_honest_scope_below_the_cliff_the_flat_resonator_is_already_working():
    # Recursion is not free and this test says so: a depth-2 composite of base symbols is solved by the flat
    # resonator directly. Recursive factoring exists for what is PAST the cliff.
    vocab = _vocab()
    comp = map_bind(vocab[3], vocab[7])
    flat = ResonatorNetwork([vocab] * 2).factor(comp, restarts=6, iters=200)
    assert flat["solved"] and sorted(int(x) for x in flat["factors"]) == [3, 7]


# ---------------------------------------------------------------------------------------------------------
# the verify gate
# ---------------------------------------------------------------------------------------------------------

def test_the_verify_gate_refuses_an_unexpressible_composite_instead_of_guessing():
    vocab = _vocab()
    cb = _two_level_codebook()
    junk = map_bind(vocab[0], vocab[2], vocab[4])                    # depth 3: no level expresses it
    bad = recursive_factor(junk, cb, vocab, restarts=4, iters=120)
    assert bad["solved"] is False and bad["verified"] is False and bad["leaves"] == []
    assert bad["tried"] == [4, 2, 1]                                 # it walked the whole ladder before refusing


def test_a_solved_answer_always_reconstructs_the_composite():
    # The gate's contract: if solved, binding the leaves reproduces the composite exactly. No exceptions.
    vocab = _vocab()
    cb = _two_level_codebook()
    _, ids4 = level_codebook(cb, vocab, 4)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        comp = map_bind(chunk_vector(ids4[a], cb, vocab), chunk_vector(ids4[b], cb, vocab))
        got = recursive_factor(comp, cb, vocab, restarts=8, iters=200)
        if got["solved"]:
            assert np.allclose(map_bind(*[vocab[i] for i in got["leaves"]]), comp)


# ---------------------------------------------------------------------------------------------------------
# MAP is self-inverse: correct != minimal
# ---------------------------------------------------------------------------------------------------------

def test_reduce_involution_cancels_duplicate_leaves():
    assert reduce_involution([0, 0, 3, 7]) == [3, 7]
    assert reduce_involution([5, 5, 5]) == [5]
    assert reduce_involution([1, 1, 2, 2]) == []
    assert reduce_involution([]) == []


def test_a_non_minimal_expansion_can_be_exactly_correct():
    # bind(v0,v3) * bind(v0,v7) == v3 * v7, because bind(x,x) is all-ones. The expansion [0,0,3,7] therefore
    # RECONSTRUCTS the composite perfectly -- the verify gate is right to pass it, and reduction is what makes it
    # minimal. Measured: this is exactly what the resonator returned before reduce_involution existed.
    vocab = _vocab()
    comp = map_bind(vocab[3], vocab[7])
    assert np.allclose(map_bind(*[vocab[i] for i in [0, 0, 3, 7]]), comp)
    assert reduce_involution([0, 0, 3, 7]) == [3, 7]


# ---------------------------------------------------------------------------------------------------------
# level machinery
# ---------------------------------------------------------------------------------------------------------

def test_level_codebook_and_available_levels():
    vocab = _vocab()
    cb = _two_level_codebook()
    assert available_levels(cb, vocab) == [4, 2, 1]
    assert level_codebook(cb, vocab, 1)[0].shape == (V, D)
    assert level_codebook(cb, vocab, 2)[0].shape == (6, D)
    assert level_codebook(cb, vocab, 4)[0].shape == (3, D)
    assert level_codebook(cb, vocab, 3)[1] == []                     # an empty level, not a crash
    ids2 = level_codebook(cb, vocab, 2)[1]
    assert ids2 == sorted(ids2)                                      # ascending ids: deterministic index -> token


def test_chunk_vector_binds_the_leaf_expansion():
    vocab = _vocab()
    cb = _two_level_codebook()
    _, ids4 = level_codebook(cb, vocab, 4)
    leaves = cb.decode([ids4[0]])
    assert np.allclose(chunk_vector(ids4[0], cb, vocab), map_bind(*[vocab[i] for i in leaves]))
    assert np.allclose(chunk_vector(5, cb, vocab), vocab[5])         # a leaf is its own vector


# ---------------------------------------------------------------------------------------------------------
# wiring + R3: one codebook family
# ---------------------------------------------------------------------------------------------------------

def test_fully_wired_to_the_mind():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    vocab = m.map_codebook(V, D, seed=0)
    assert np.array_equal(vocab, map_codebook(V, D, 0))              # a SEED, not a megabyte of vectors
    assert np.allclose(m.map_bind(vocab[1], vocab[1]), np.ones(D))   # self-inverse, through the mind
    assert m.reduce_involution([0, 0, 3, 7]) == [3, 7]

    cb = _two_level_codebook().to_dict()
    assert m.chunk_levels(cb, vocab) == [4, 2, 1]
    _, ids4 = level_codebook(ChunkCodebook.from_dict(cb), vocab, 4)
    comp = m.map_bind(*[vocab[i] for i in ChunkCodebook.from_dict(cb).decode([ids4[0]])],
                      *[vocab[i] for i in ChunkCodebook.from_dict(cb).decode([ids4[1]])])
    got = m.recursive_factor(comp, cb, vocab, restarts=8, iters=200)
    assert got["solved"] and got["verified"] and len(got["leaves"]) == 8

    assert "Recursive factoring" in str(m.find_capability("my resonator fails past four factors")[:3])


def test_r3_the_codebook_learned_by_r1_is_the_one_r2_factors_against():
    # THE POINT OF R3, end to end through the mind: a stream is observed, chunks are PROMOTED from it (R1), and
    # those very chunks become the levels a deep composite is factored against (R2). One structure, two consumers.
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    vocab = m.map_codebook(16, D, seed=0)

    rng = np.random.default_rng(3)
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    stream = []
    for _ in range(120):
        a, b = pairs[rng.integers(0, 4)]
        c, d = pairs[rng.integers(0, 4)]
        stream += [a, b, c, d]

    assert m.structure_score(stream) > 2.0                           # R1's gate: this stream HAS structure
    cb = m.learn_chunks(stream, max_merges=12)
    assert 4 in m.chunk_levels(cb, vocab)                            # quads were promoted from the stream

    comp = m.map_bind(*[vocab[i] for i in [0, 1, 2, 3, 4, 5, 6, 7]])
    got = m.recursive_factor(comp, cb, vocab, restarts=8, iters=200)
    assert got["solved"] and got["verified"]
    assert got["leaves"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert got["level"] == 4                                         # solved at the LEARNED chunk level

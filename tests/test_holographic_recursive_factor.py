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


# --------------------------------------------------------------------------------------
# Per-level failure reasons (work plan item 5.3).
# --------------------------------------------------------------------------------------

def test_a_failure_now_says_which_level_failed_and_why():
    """THE DEFECT: three different things produced an IDENTICAL record -- a level too small to run at all,
    a resonator that did not converge, and a candidate the verify gate rejected. "Nothing fits your arity"
    and "I tried and the coherence was 0.31" are OPPOSITE diagnoses (raise the arity vs the composite is
    not factorable here), and collapsing them made a solvable problem read as impossible."""
    vocab = _vocab()
    cb = _two_level_codebook()
    junk = np.random.default_rng(3).standard_normal(D)
    junk /= np.linalg.norm(junk)
    got = recursive_factor(junk, cb, vocab, restarts=2, iters=40)
    assert not got["solved"]
    assert got["reasons"], "a failure with no reasons is the defect this fixes"
    for row in got["reasons"]:
        assert row["reason"] in ("level-too-small", "arity-unreachable",
                                 "resonator-unconverged", "verify-failed")
        assert row["detail"], "a reason without a detail is not a diagnosis"
        assert isinstance(row["level"], int)


def test_arity_unreachable_is_named_distinctly():
    # The one arity mismatch the code can diagnose with CERTAINTY: fewer distinct tokens at a level than
    # the arity being searched. The fix is to lower the arity or promote more chunks -- the opposite advice
    # from "not factorable here", which is exactly why it needs its own name.
    vocab = _vocab()
    cb = _two_level_codebook()
    _, ids4 = level_codebook(cb, vocab, 4)
    comp = map_bind(chunk_vector(ids4[0], cb, vocab), chunk_vector(ids4[1], cb, vocab))
    got = recursive_factor(comp, cb, vocab, arity=5, restarts=2, iters=40)
    unreachable = [r for r in got["reasons"] if r["reason"] == "arity-unreachable"]
    assert unreachable, got["reasons"]
    assert 4 not in got["tried"], "a level that could not be attempted was counted as tried"


def test_a_level_too_small_to_run_is_recorded_not_skipped_silently():
    # Previously this level never appeared in `tried` OR anywhere else, so a run that attempted nothing was
    # indistinguishable from one that attempted everything and was rejected.
    vocab = _vocab()
    cb = _two_level_codebook()
    junk = np.random.default_rng(5).standard_normal(D)
    junk /= np.linalg.norm(junk)
    got = recursive_factor(junk, cb, vocab, arity=2, restarts=1, iters=20)
    levels_accounted = {r["level"] for r in got["reasons"]} | set(got["tried"])
    assert levels_accounted, "no level was accounted for at all"


def test_a_solved_run_also_carries_reasons_for_the_levels_it_rejected():
    # The levels tried BEFORE the winning one are evidence about the search, not noise -- a solve at level 4
    # after rejecting level 8 means something different from a solve at level 4 reached first.
    vocab = _vocab()
    cb = _two_level_codebook()
    _, ids4 = level_codebook(cb, vocab, 4)
    comp = map_bind(chunk_vector(ids4[0], cb, vocab), chunk_vector(ids4[1], cb, vocab))
    got = recursive_factor(comp, cb, vocab, restarts=8, iters=200)
    assert got["solved"] and "reasons" in got


# --------------------------------------------------------------------------------------
# The 5.3b gate: is the F=4 failure a capacity limit or a search budget?
# --------------------------------------------------------------------------------------

@pytest.mark.slow                       # a restart sweep at N=2048
def test_the_f4_cliff_is_a_search_budget_not_a_capacity_limit():
    """THE GATE RESULT FOR THE PROPOSED ALGORITHMIC REPLACEMENT. It was proposed that this resonator needs a
    new update rule because it fails at F=4. Measured at N=2048, V=16, F=4 (search space 65,536): the SAME
    network on the SAME codebooks goes from 25% at restarts=4 to 100% at restarts=256.

    So a published method must be benchmarked against a BUDGET-MATCHED baseline. Comparing a new update rule
    against this one at restarts=4 would be a strawman, and a win without a proper baseline is not a result.
    """
    from holographic.misc.holographic_resonator import ResonatorNetwork, map_bind, map_codebook

    F, V, N = 4, 16, 2048
    books = [map_codebook(V, N, seed=100 + i) for i in range(F)]

    def rate(restarts, iters, trials=8):
        ok = 0
        for t in range(trials):
            rng = np.random.default_rng(t)
            idx = [int(rng.integers(V)) for _ in range(F)]
            comp = map_bind(*[books[i][idx[i]] for i in range(F)])
            got = ResonatorNetwork(books).factor(comp, restarts=restarts, iters=iters)
            ok += bool(got["solved"] and [int(x) for x in got["factors"]] == idx)
        return ok / trials

    assert rate(4, 150) < 0.6, "the low-budget arm no longer fails; re-read the budget finding"
    assert rate(256, 600) > 0.9, "the high-budget arm no longer succeeds; the cliff may be real after all"


def test_the_budget_finding_is_recorded_where_it_will_be_reproposed():
    import inspect
    from holographic.misc import holographic_resonator as mod
    src = inspect.getsource(mod.ResonatorNetwork.factor)
    assert "SEARCH BUDGET, NOT A CAPACITY LIMIT" in src
    assert "NOT SUFFICIENT" in src, "the Df/N insufficiency finding left the comment"


# --------------------------------------------------------------------------------------
# Restart budget advisor, and why the default was not simply raised.
# --------------------------------------------------------------------------------------

def test_the_restart_sequence_is_prefix_stable():
    """The determinism question, settled: raising the budget cannot flip an existing answer. On every
    already-solved case, restarts=64 returns the identical factors AND the identical restart count as
    restarts=20 -- so the objection to a higher default is COST, not correctness."""
    from holographic.misc.holographic_resonator import ResonatorNetwork, map_bind, map_codebook

    N = 1024
    for F, V in ((2, 8), (3, 8)):
        books = [map_codebook(V, N, seed=100 + i) for i in range(F)]
        for t in range(3):
            rng = np.random.default_rng(t)
            idx = [int(rng.integers(V)) for _ in range(F)]
            comp = map_bind(*[books[i][idx[i]] for i in range(F)])
            low = ResonatorNetwork(books).factor(comp, restarts=20, iters=300)
            high = ResonatorNetwork(books).factor(comp, restarts=64, iters=300)
            if low["solved"]:
                assert low["factors"] == high["factors"]
                assert low["restarts"] == high["restarts"], "a bigger cap changed the winning restart"


@pytest.mark.slow                       # a budget sweep over several codebooks
def test_the_advisor_asks_for_more_restarts_as_factors_grow():
    from holographic.misc.holographic_resonator import advise_restarts, map_codebook

    easy = advise_restarts([map_codebook(16, 2048, seed=100 + i) for i in range(2)],
                           targets=(0.9,), budgets=(4, 16, 64, 256), trials=6)[0]
    hard = advise_restarts([map_codebook(16, 2048, seed=100 + i) for i in range(4)],
                           targets=(0.9,), budgets=(4, 16, 64, 256), trials=6)[0]
    assert easy["restarts"] is not None and hard["restarts"] is not None
    assert hard["restarts"] > easy["restarts"], \
        "F=4 no longer needs a bigger budget than F=2 (%r vs %r)" % (hard["restarts"], easy["restarts"])


def test_the_advisor_reports_the_curve_not_just_a_number():
    # A budget without its curve is another number-without-its-variable; the caller needs to see where it
    # saturates to know whether more would help.
    from holographic.misc.holographic_resonator import advise_restarts, map_codebook

    got = advise_restarts([map_codebook(8, 512, seed=1), map_codebook(8, 512, seed=2)],
                          targets=(0.9,), budgets=(4, 16), trials=4)[0]
    assert "measured" in got and len(got["measured"]) == 2
    assert all(isinstance(b, int) and 0.0 <= r <= 1.0 for b, r in got["measured"])


def test_the_attention_update_negative_is_recorded():
    """KEPT NEGATIVE, PINNED (backlog item 5.3b). A softmax/attention weighting was raced against the shipped
    linear one at EQUAL BUDGET and does not win — it ties inside a narrow tuned band and is worse outside.
    If someone proposes it again, this points them at the measurement rather than the paper."""
    import inspect

    from holographic.misc import holographic_resonator as mod
    doc = inspect.getdoc(mod.ResonatorNetwork._cleanup) or ""
    assert "ATTENTION (SOFTMAX) WEIGHTING WAS RACED HERE AND DOES NOT WIN" in doc
    assert "BETA IS MEANINGLESS WITHOUT THE SCALE IT MULTIPLIES" in doc


def test_a_one_hot_cleanup_would_destroy_the_search():
    """THE FINDING WORTH MORE THAN THE NEGATIVE. The shipped cleanup superposes codevectors weighted by raw
    similarity; as a softmax sharpens toward one-hot — committing to a single codevector each iteration —
    the solve rate goes to ZERO. Holding a weighted blend is what lets the search escape a wrong commitment.

    Pinned as a property of the shipped rule: its cleanup must NOT be a bare argmax."""
    import inspect

    from holographic.misc import holographic_resonator as mod
    src = inspect.getsource(mod.ResonatorNetwork._cleanup)
    assert "B.T @ (B @ est)" in src, "the cleanup became something other than a weighted superposition"
    assert "argmax" not in src, "the cleanup committed to a single codevector; measured at 0% solve rate"

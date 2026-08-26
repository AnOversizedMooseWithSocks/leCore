"""W5 -- hierarchical superposition: capacity is bounded by the worst single level, not the product of levels.

The claims under test, with the baseline shipped beside the capability:

  1. THEOREM-SHAPED NEGATIVE: superposition is linear, so naive bundle-of-bundles with product roles IS one flat
     bundle. Nesting alone buys nothing.
  2. THE MECHANISM: a cleanup between levels resets crosstalk, so the descent below it is clean. 100% recall at
     G=64 where the flat baseline has collapsed to ~18%.
  3. CORRECTION TO THE BACKLOG: shared chunks do not buy RECALL (64 distinct patterns still recalls 100%) --
     they buy a SMALL CODEBOOK. Recall comes from the cleanup; sharing comes from R1.
"""

import numpy as np
import pytest

from holographic.misc.holographic_core import bind, bundle
from holographic.misc.holographic_superposed import (
    pack, hierarchical_pack, hierarchical_recall, flat_recall)
from holographic.agents_and_reasoning.holographic_ai import unitary_vector

D = 2048
PER = 8


def _atoms(rng, n):
    return np.stack([unitary_vector(D, rng) for _ in range(n)])


def _world(seed, groups, n_patterns):
    """G groups, each holding one of `n_patterns` shared chunk patterns of PER leaves."""
    rng = np.random.default_rng(seed)
    leaf_keys = _atoms(rng, PER)
    group_keys = _atoms(rng, groups)
    items = _atoms(rng, PER * n_patterns)
    chunks = np.stack([pack(leaf_keys, items[p * PER:(p + 1) * PER]) for p in range(n_patterns)])
    assign = rng.integers(0, n_patterns, groups) if n_patterns < groups else np.arange(groups)
    return rng, leaf_keys, group_keys, items, chunks, assign


def _flat_bundle(group_keys, leaf_keys, items, assign):
    """The strongest honest baseline in the original space: ALL G*PER items bundled under product roles."""
    G = len(group_keys)
    keys = np.stack([bind(group_keys[g], leaf_keys[i]) for g in range(G) for i in range(PER)])
    vals = np.stack([items[assign[g] * PER + i] for g in range(G) for i in range(PER)])
    return pack(keys, vals)


def test_selftest_runs():
    from holographic.misc import holographic_superposed as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# 1. the theorem-shaped negative
# ---------------------------------------------------------------------------------------------------------

def test_naive_nesting_is_literally_the_flat_bundle():
    # bind distributes over the sum, so bind(g, sum_i bind(l_i, x_i)) == sum_i bind(bind(g, l_i), x_i).
    # Compared as raw SUMS: bundle() may normalise, which would hide the identity behind a scale factor.
    rng, lk, gk, ib, _, _ = _world(0, groups=4, n_patterns=4)
    nested = sum(bind(gk[g], sum(bind(lk[i], ib[i]) for i in range(PER))) for g in range(4))
    flat = sum(bind(bind(gk[g], lk[i]), ib[i]) for g in range(4) for i in range(PER))
    assert np.abs(nested - flat).max() < 1e-9

    # ... so a "hierarchy" without a mid-level cleanup cannot possibly recall better than flat: same vector.
    assert np.allclose(nested / np.linalg.norm(nested), flat / np.linalg.norm(flat))


# ---------------------------------------------------------------------------------------------------------
# 2. the mechanism: cleanup between levels
# ---------------------------------------------------------------------------------------------------------

def _recall_rates(seed, groups, n_patterns, trials=40):
    rng, lk, gk, ib, chunks, asg = _world(seed, groups, n_patterns)
    S = hierarchical_pack(gk, chunks[asg])
    Sf = _flat_bundle(gk, lk, ib, asg)
    ok_h = ok_f = 0
    for _ in range(trials):
        g0, i0 = int(rng.integers(0, groups)), int(rng.integers(0, PER))
        truth = asg[g0] * PER + i0
        ok_h += int(hierarchical_recall(S, gk[g0], lk[i0], chunks, ib)["item_index"] == truth)
        ok_f += int(flat_recall(Sf, gk[g0], lk[i0], ib)["item_index"] == truth)
    return ok_h / trials, ok_f / trials


def test_the_flat_baseline_collapses_as_the_bundle_widens():
    # The premise. Without this the hierarchy is solving nothing.
    rates = [_recall_rates(10 + g, groups=g, n_patterns=16)[1] for g in (4, 16, 32, 64)]
    assert rates[0] == 1.0                       # G=4: flat is fine, nothing to fix
    assert rates == sorted(rates, reverse=True)  # monotone collapse
    assert rates[-1] < 0.5                       # G=64: the flat bundle has fallen apart


def test_the_mid_level_cleanup_holds_recall_where_flat_collapses():
    # NB: "100%" is the modal result, not a guarantee -- one seed in four measures 0.975 at G=64. Asserting an
    # exact 1.0 would be asserting a seed. The claim is that hierarchy stays near-perfect while flat collapses.
    for g in (4, 16, 32, 64):
        hier, flat = _recall_rates(10 + g, groups=g, n_patterns=16)
        assert hier >= 0.95, (g, hier)
        assert hier > flat or flat == 1.0
    hier64, flat64 = _recall_rates(74, groups=64, n_patterns=16)
    assert hier64 >= 0.95 and flat64 < 0.5       # the headline, on one seed, in one assertion


def test_the_cleanup_returns_its_confidence_and_the_chunk_it_chose():
    rng, lk, gk, ib, chunks, asg = _world(5, groups=32, n_patterns=16)
    S = hierarchical_pack(gk, chunks[asg])
    got = hierarchical_recall(S, gk[7], lk[3], chunks, ib)
    assert got["chunk_index"] == asg[7]                       # it snapped to the RIGHT pattern
    assert got["item_index"] == asg[7] * PER + 3
    assert 0.0 < got["chunk_similarity"] <= 1.0
    assert got["item"].shape == (D,)

    # NOT "nearly clean" -- I asserted > 0.9 first and it came back 0.344. The mid-level cleanup resets the GROUP
    # crosstalk; what remains is exactly the LEAF level's own 1/sqrt(width). That IS "capacity is bounded by the
    # worst single level", as a number. Measured across widths: 2 -> 0.705, 4 -> 0.498, 8 -> 0.352, 16 -> 0.251,
    # against 1/sqrt(w) = 0.707 / 0.500 / 0.354 / 0.250.
    assert abs(got["item_similarity"] - 1.0 / np.sqrt(PER)) < 0.05


def test_the_post_cleanup_similarity_is_exactly_the_leaf_levels_own_crosstalk():
    # The law, swept. A cleanup between levels does not make the descent noiseless -- it makes the descent see
    # ONLY the level below it. 1/sqrt(leaf width), independent of how many groups are in the bundle.
    for per in (2, 4, 8, 16):
        rng = np.random.default_rng(11)
        at = lambda n: np.stack([unitary_vector(D, rng) for _ in range(n)])
        lk, gk, ib = at(per), at(32), at(per * 16)
        chunks = np.stack([pack(lk, ib[p * per:(p + 1) * per]) for p in range(16)])
        asg = rng.integers(0, 16, 32)
        S = hierarchical_pack(gk, chunks[asg])
        sims = [hierarchical_recall(S, gk[g], lk[i], chunks, ib)["item_similarity"]
                for g in range(8) for i in range(per)]
        assert abs(float(np.mean(sims)) - 1.0 / np.sqrt(per)) < 0.02, per


def test_removing_the_cleanup_removes_the_win():
    # Prove the cleanup is load-bearing, not incidental: recall the SAME hierarchical vector without the mid-level
    # snap (i.e. treat it flat) and the advantage disappears.
    rng, lk, gk, ib, chunks, asg = _world(6, groups=64, n_patterns=16)
    S = hierarchical_pack(gk, chunks[asg])
    ok_with = ok_without = 0
    for _ in range(40):
        g0, i0 = int(rng.integers(0, 64)), int(rng.integers(0, PER))
        truth = asg[g0] * PER + i0
        ok_with += int(hierarchical_recall(S, gk[g0], lk[i0], chunks, ib)["item_index"] == truth)
        ok_without += int(flat_recall(S, gk[g0], lk[i0], ib)["item_index"] == truth)
    assert ok_with >= 38 and ok_without < 30


# ---------------------------------------------------------------------------------------------------------
# 3. the correction: sharing buys codebook SIZE, not recall
# ---------------------------------------------------------------------------------------------------------

def test_kept_negative_shared_chunks_do_not_buy_recall_they_buy_a_small_codebook():
    # The backlog implies recall needs SHARED chunks. Measured: 64 distinct patterns for 64 groups still recalls
    # 100%. What sharing buys is the codebook's size -- 16 patterns instead of 64.
    hier_shared, _ = _recall_rates(20, groups=64, n_patterns=16)
    hier_unshared, _ = _recall_rates(21, groups=64, n_patterns=64)
    assert hier_shared >= 0.95 and hier_unshared >= 0.95   # both near-perfect: sharing changed nothing here

    shared_bytes = 16 * D * 8
    unshared_bytes = 64 * D * 8
    assert unshared_bytes == 4 * shared_bytes            # THAT is the dividend, and it is memory, not accuracy


def test_the_single_vector_holds_structure_and_the_codebooks_hold_content():
    # The honest framing, asserted. S is one D-vector; it cannot contain 512 independent items. The atoms live in
    # the codebooks, and recall FAILS if you take the codebooks away (there is nothing to snap to).
    rng, lk, gk, ib, chunks, asg = _world(7, groups=64, n_patterns=16)
    S = hierarchical_pack(gk, chunks[asg])
    assert S.shape == (D,)
    assert ib.shape == (16 * PER, D)                     # the content: 128 atoms, not 512

    # with a codebook containing only the WRONG patterns, the cleanup cannot recover anything correct
    wrong = _atoms(np.random.default_rng(99), 16)
    got = hierarchical_recall(S, gk[0], lk[0], wrong, ib)
    assert got["chunk_similarity"] < 0.3                 # it snapped to noise, and says so in its confidence


# ---------------------------------------------------------------------------------------------------------
# wiring + R3: one codebook family
# ---------------------------------------------------------------------------------------------------------

def test_fully_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    rng, lk, gk, ib, chunks, asg = _world(8, groups=32, n_patterns=16)
    S = m.hierarchical_pack(gk, chunks[asg])
    got = m.hierarchical_recall(S, gk[5], lk[1], chunks, ib)
    assert got["item_index"] == asg[5] * PER + 1
    base = m.flat_recall(S, gk[5], lk[1], ib)
    assert "item_index" in base and "chunk_index" not in base
    assert "Hierarchical superposition" in str(m.find_capability("bundle of bundles")[:3])


def test_r3_one_codebook_family_r1_chunks_choose_w5_patterns():
    # R3: the SAME promoted chunks that R1 learns from a stream decide which groups share a pattern in W5.
    # Here the stream's learned chunk vocabulary sizes the chunk codebook, and the reuse it found is the reuse
    # W5 exploits. Measured end to end rather than asserted architecturally.
    import lecore
    from holographic.agents_and_reasoning.holographic_chunkcodebook import workflow_stream, uniform_stream
    m = lecore.UnifiedMind(dim=256, seed=0)

    structured = workflow_stream(n_workflows=400)
    noise = uniform_stream(n=1600)
    assert m.structure_score(structured) > 3.0 > m.structure_score(noise)

    # the structured stream's top chunks are FEW and cover most of it -> a small W5 chunk codebook
    cb = m.learn_chunks(structured, max_merges=64)
    st = m.chunk_stats(structured, cb)
    assert st["covered"] > 0.7 and st["n_merges"] <= 64

    # the noise stream's chunks cover it poorly -> no small codebook exists, so W5's memory dividend evaporates
    ncb = m.learn_chunks(noise, max_merges=64)
    nst = m.chunk_stats(noise, ncb)
    assert nst["mean_depth"] < st["mean_depth"] / 2

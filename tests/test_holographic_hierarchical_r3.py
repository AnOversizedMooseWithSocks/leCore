"""W5 + R3 -- hierarchical superposition fed by R1's LEARNED chunk codebook, and the coverage failure it exposes.

W5 itself already shipped (hierarchical_pack / hierarchical_recall / flat_recall). What is added and tested here:

  * `chunk_codebook_vectors` -- THE R3 BRIDGE: R1's learned chunk ids realized as packed chunk vectors. R1 learns
    WHICH chunks recur; R2 realizes each as a map_bind product; W5 realizes each as a pack superposition. Same
    identities, different vectors. That is what "one codebook family" means, and this is its third consumer.
  * `chunk_coverage` -- the precondition, measured rather than assumed.
  * `min_chunk_similarity` -- the ABSTAIN GATE, added because of a measured failure: an UNCOVERED group's
    mid-level cleanup snaps to the nearest codebook entry (the wrong chunk) and returns an item with every
    appearance of success.
"""

import numpy as np
import pytest

from holographic.misc.holographic_superposed import (
    pack, hierarchical_pack, hierarchical_recall, flat_recall, chunk_codebook_vectors, chunk_coverage)
from holographic.agents_and_reasoning.holographic_chunkcodebook import learn_chunks

D = 1024
N_ITEMS = 32
LEAVES = 4


def _fixture(seed=0, n_distinct=8, merges=200):
    """A stream whose groups are `n_distinct` recurring patterns of LEAVES items -- the shared-chunk condition."""
    rng = np.random.default_rng(seed)
    items = rng.normal(size=(N_ITEMS, D))
    items /= np.linalg.norm(items, axis=1, keepdims=True)

    def unit(n):
        ph = rng.uniform(0.0, 2 * np.pi, (n, D // 2 + 1))
        ph[:, 0] = 0.0
        ph[:, -1] = 0.0
        return np.fft.irfft(np.exp(1j * ph), n=D, axis=1)

    leaf_keys, group_keys = unit(LEAVES), unit(64)
    groups = [list(rng.choice(N_ITEMS, LEAVES, replace=False)) for _ in range(n_distinct)]
    stream = []
    for _ in range(400):
        stream += groups[rng.integers(0, n_distinct)]
    cb = learn_chunks(stream, max_merges=merges)
    return items, leaf_keys, group_keys, groups, cb, rng


def test_selftest_runs():
    from holographic.misc import holographic_superposed as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# R3: the learned codebook IS the chunk codebook
# ---------------------------------------------------------------------------------------------------------

def test_the_bridge_realizes_learned_chunks_as_packed_vectors():
    items, leaf_keys, _, groups, cb, _ = _fixture()
    vecs, ids, leaves = chunk_codebook_vectors(cb, items, leaf_keys)

    assert vecs.shape == (len(ids), D) and len(leaves) == len(ids)
    assert len(ids) >= 2
    # every promoted chunk's leaves are a group that ACTUALLY occurred -- BPE did not invent one
    known = {tuple(g) for g in groups}
    assert all(tuple(ls) in known for ls in leaves)
    # each vector is the pack of its own leaves
    for v, ls in zip(vecs, leaves):
        assert np.allclose(v, pack(leaf_keys, items[np.asarray(ls, int)]))


def test_the_bridge_returns_nothing_when_no_chunk_fills_the_slots():
    # A chunk must fill every slot. Pick a width with NO tokens -- note BPE also promotes depth-3 chunks
    # (a pair merged with a leaf), so "3" is not such a width; ask the codebook rather than guessing.
    items, leaf_keys, _, _, cb, rng = _fixture()
    present = set(cb.depth.values())
    absent_width = next(w for w in range(2, 40) if w not in present)

    ph = rng.uniform(0.0, 2 * np.pi, (absent_width, D // 2 + 1))
    ph[:, 0] = 0.0
    ph[:, -1] = 0.0
    keys = np.fft.irfft(np.exp(1j * ph), n=D, axis=1)

    vecs, ids, leaves = chunk_codebook_vectors(cb, items, keys)
    assert len(ids) == 0 and vecs.shape[0] == 0 and leaves == []


def test_r3_end_to_end_hierarchical_recall_on_a_learned_codebook():
    items, leaf_keys, group_keys, _, cb, rng = _fixture()
    vecs, ids, leaves = chunk_codebook_vectors(cb, items, leaf_keys)

    G = 16
    picks = rng.choice(len(ids), G, replace=True)
    S = hierarchical_pack(group_keys[:G], vecs[picks])
    ok = 0
    for _ in range(30):
        g = int(rng.integers(0, G))
        l = int(rng.integers(0, LEAVES))
        truth = leaves[picks[g]][l]
        got = hierarchical_recall(S, group_keys[g], leaf_keys[l], vecs, items)
        ok += got["item_index"] == truth
    assert ok == 30                                    # every recall exact, on a codebook nobody handed us


def test_the_capacity_win_over_the_flat_baseline_grows_with_the_group_count():
    # Reproduces the backlog's table on a LEARNED codebook. flat 100/95/70/30 at G=4/16/32/64 (D=2048);
    # here at D=1024 the flat baseline collapses sooner, which is the point -- the wall is dimensional.
    items, leaf_keys, group_keys, _, cb, rng = _fixture()
    vecs, ids, leaves = chunk_codebook_vectors(cb, items, leaf_keys)

    def rate(G, n=30):
        h = f = 0
        for s in range(n):
            r = np.random.default_rng(500 + s)
            picks = r.choice(len(ids), G, replace=True)
            S = hierarchical_pack(group_keys[:G], vecs[picks])
            g, l = int(r.integers(0, G)), int(r.integers(0, LEAVES))
            truth = leaves[picks[g]][l]
            h += hierarchical_recall(S, group_keys[g], leaf_keys[l], vecs, items)["item_index"] == truth
            f += flat_recall(S, group_keys[g], leaf_keys[l], items)["item_index"] == truth
        return h / n, f / n

    h4, f4 = rate(4)
    h32, f32 = rate(32)
    assert h4 == 1.0 and h32 == 1.0                    # hierarchical: flat across the range
    assert f32 < f4                                    # flat: degrades with width
    assert f32 < 0.9 and h32 > f32                     # ... and the hierarchy is strictly better where it matters


# ---------------------------------------------------------------------------------------------------------
# coverage, and the dangerous negative
# ---------------------------------------------------------------------------------------------------------

def test_coverage_is_set_by_how_many_merges_r1_was_allowed():
    # Not bookkeeping: an uncovered group is a wrong answer waiting to happen. Measured on the real fixture --
    # too few merges and only some of the groups become chunks.
    _, _, _, groups, few, _ = _fixture(merges=12)
    _, _, _, groups2, many, _ = _fixture(merges=400)
    c_few = chunk_coverage(few, groups, LEAVES)
    c_many = chunk_coverage(many, groups2, LEAVES)
    assert c_few["total"] == c_many["total"] == 8
    assert c_few["fraction"] < c_many["fraction"]
    assert c_many["fraction"] == 1.0 and c_many["missing"] == []
    assert len(c_few["missing"]) == c_few["total"] - c_few["covered"]


def test_kept_negative_an_uncovered_group_snaps_to_the_wrong_chunk_and_answers_confidently():
    items, leaf_keys, group_keys, groups, cb, _ = _fixture(merges=12)
    vecs, ids, leaves = chunk_codebook_vectors(cb, items, leaf_keys)
    cov = chunk_coverage(cb, groups, LEAVES)
    assert cov["missing"], "this fixture is supposed to leave some groups uncovered"

    absent = list(cov["missing"][0])
    S = hierarchical_pack(group_keys[:2], np.stack([pack(leaf_keys, items[np.asarray(absent, int)]), vecs[0]]))

    ungated = hierarchical_recall(S, group_keys[0], leaf_keys[2], vecs, items)
    assert ungated["abstained"] is False
    assert ungated["item_index"] != absent[2]                  # a CONFIDENT WRONG ANSWER
    assert ungated["chunk_similarity"] < 0.15                  # ... and the signal that it is wrong was right there

    gated = hierarchical_recall(S, group_keys[0], leaf_keys[2], vecs, items, min_chunk_similarity=0.15)
    assert gated["abstained"] is True and gated["item_index"] is None and gated["item"] is None


def test_the_abstain_gate_does_not_fire_on_a_covered_group():
    items, leaf_keys, group_keys, _, cb, _ = _fixture()
    vecs, ids, leaves = chunk_codebook_vectors(cb, items, leaf_keys)
    S = hierarchical_pack(group_keys[:4], vecs[[0, 1, 2, 3]])
    got = hierarchical_recall(S, group_keys[1], leaf_keys[3], vecs, items, min_chunk_similarity=0.15)
    assert got["abstained"] is False
    assert got["item_index"] == leaves[1][3]
    assert got["chunk_similarity"] >= 0.15


def test_the_gate_is_default_off_and_backward_compatible():
    items, leaf_keys, group_keys, _, cb, _ = _fixture()
    vecs, _, leaves = chunk_codebook_vectors(cb, items, leaf_keys)
    S = hierarchical_pack(group_keys[:4], vecs[[0, 1, 2, 3]])
    old = hierarchical_recall(S, group_keys[0], leaf_keys[0], vecs, items)
    assert old["abstained"] is False                            # the new key is present but never blocks
    assert old["item_index"] == leaves[0][0]


# ---------------------------------------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------------------------------------

def test_fully_wired_to_the_mind():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    items, leaf_keys, group_keys, groups, cb, _ = _fixture()

    vecs, ids, leaves = m.chunk_codebook_vectors(cb.to_dict(), items, leaf_keys)
    assert vecs.shape[0] == len(ids) == len(leaves)

    cov = m.chunk_coverage(cb.to_dict(), groups, LEAVES)
    assert cov["fraction"] == 1.0

    S = m.hierarchical_pack(group_keys[:4], np.asarray(vecs)[[0, 1, 2, 3]])
    got = m.hierarchical_recall(S, group_keys[2], leaf_keys[1], vecs, items, min_chunk_similarity=0.15)
    assert got["abstained"] is False and got["item_index"] == leaves[2][1]

    flat = m.flat_recall(S, group_keys[2], leaf_keys[1], items)
    assert "item_index" in flat                                 # the baseline is still shipped beside it

    assert "Hierarchical superposition" in str(m.find_capability("use my learned chunks as a memory hierarchy")[:3])


def test_r3_one_codebook_three_consumers():
    # R1 learns the chunks; R2 factors against them (map_bind product); W5 recalls against them (pack
    # superposition). The IDENTITIES are shared; the vectors are not, and that distinction is the whole claim.
    import lecore
    from holographic.misc.holographic_resonator import level_codebook, map_codebook
    m = lecore.UnifiedMind(dim=256, seed=0)
    items, leaf_keys, _, _, cb, _ = _fixture()

    w5_vecs, w5_ids, _ = chunk_codebook_vectors(cb, items, leaf_keys)     # consumer 3: superposition
    vocab = map_codebook(N_ITEMS, D, seed=0)
    r2_book, r2_ids = level_codebook(cb, vocab, LEAVES)                   # consumer 2: binding

    assert w5_ids == r2_ids                                              # SAME chunk identities ...
    assert not np.allclose(w5_vecs[0], r2_book[0])                       # ... DIFFERENT vector realizations
    assert m.structure_score([i for ls in [cb.decode([t]) for t in w5_ids] for i in ls]) > 1.0

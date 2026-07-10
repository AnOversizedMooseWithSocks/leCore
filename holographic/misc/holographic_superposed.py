"""
holographic_superposed.py  --  parallel computation in superposition (the WIDTH faculty).

PORTED FROM leOS (`superposed_compute.py`, "one processor, many states simultaneously").
holostuff already owns the DEPTH side of holographic computing -- recursive structure
(`encode_tree`), cleanup-gated traversal (`peel.traverse`), the measured inception depth law.
What it lacked was a first-class WIDTH primitive: a single faculty for *evaluating many
computations at once inside one vector*, which is the literal thing the Path D capacity
experiment measured. This module is that faculty, on holostuff's frozen kernel.

The idea (Kanerva / Kleyko VSA "computing in superposition"):
    Bundle K candidate computations into one vector, each tagged by a key:
        S = bundle( bind(key_i, item_i)  for i )
    Then ONE unbind recovers any candidate's contribution, so K results come out of a single
    superposed structure -- data-parallelism from superposition rather than from silicon cores.
    Resolve the winner by cleanup (nearest known symbol).

The honest capacity wall (measured on this kernel, see experiment_superposed_forward_pass.py):
    A D-dim vector holds only ~0.1-0.2 x D items at *discrete cleanup-gated* recall, and only
    ~0.02 x D when the recovered values feed *continuous* downstream math with no cleanup to
    absorb the crosstalk. So `width` here is bounded; to go bigger you spend DEPTH (recurse,
    cleanup-gate each level -- the inception law) rather than widening one flat bundle. This
    module is the width half; holostuff's recursion is the depth half; together they are the
    width x depth surface the theory predicts.
"""
import numpy as np
from holographic.misc.holographic_core import bind, unbind, bundle, cosine
from holographic.agents_and_reasoning.holographic_ai import bind_batch, bind_fixed, involution


def _involution_stack(A):
    """Row-wise involution (a[0] stays, the rest flips) -- matches holographic_ai.involution."""
    return np.concatenate([A[:, :1], A[:, :0:-1]], axis=1)


def pack(keys, items):
    """Superpose K keyed computations into ONE vector: S = bundle( bind(key_i, item_i) ).

    `keys`  -- (K, D) distinct tag vectors (use unitary atoms for EXACT unbinding, so the only
               readback error is superposition crosstalk).
    `items` -- (K, D) the K things being held in parallel (values, weights, candidate results).
    Returns a single (D,) vector that stands for all K keyed items at once.
    """
    keys = np.asarray(keys, float); items = np.asarray(items, float)
    return bundle(bind_batch(keys, items))


def recover_all(S, keys):
    """Pull every keyed item back out of the superposition in one shot (the parallel readout).

    recover_all(S, keys)[i] ~= items[i] + crosstalk, computed as unbind(S, key_i) for all i but
    vectorised over the whole key stack -- so K recoveries cost one batched FFT, not a Python
    loop. The estimates are NOISY; follow with cleanup (resolve) when the items are symbols.
    """
    keys = np.asarray(keys, float)
    return bind_fixed(S, _involution_stack(keys))     # bind(S, involution(key_i)) for all i


def score_all(S, keys, query):
    """Evaluate <item_i, query> for ALL i at once, out of the single superposed vector S.

    This is the parallel inner-product: instead of K separate dot products, recover the K items
    from one structure and dot each with `query`. Returns a length-K score array. (This is the
    operation whose capacity the Path D experiment charted -- noise on each score grows with K.)
    """
    rec = recover_all(S, keys)                        # (K, D) noisy items
    return rec @ np.asarray(query, float)             # (K,) parallel scores


def resolve(noisy, codebook):
    """Snap a noisy recovered vector to the nearest clean candidate (cleanup / the winner).

    `codebook` -- (M, D) the known clean items. Returns (index, similarity). This is the
    discrete decision that RESETS crosstalk -- the reason cleanup-gated recall (~0.1 x D) holds
    so much more than raw continuous superposition (~0.02 x D), and the reason recursion has to
    cleanup-gate between levels to stay alive.
    """
    cb = np.asarray(codebook, float)
    n = np.linalg.norm(noisy)
    if n == 0:
        return 0, 0.0
    sims = (cb @ noisy) / (n * np.linalg.norm(cb, axis=1).clip(1e-12))
    j = int(sims.argmax())
    return j, float(sims[j])


def evaluate_candidates(keys, candidate_results, query, codebook=None):
    """End-to-end: hold K candidate computations in superposition, score them all against a
    query in parallel, and return the winning candidate index + its score.

    With `codebook` given, each recovered candidate is also cleaned to a known symbol first
    (cleanup-gated, higher capacity). Without it, scores come from the raw continuous recovery.
    This is the leOS 'evaluate many hypotheses simultaneously, then resolve which path won'
    pattern, made a holostuff faculty.
    """
    S = pack(keys, candidate_results)
    if codebook is None:
        scores = score_all(S, keys, query)
    else:
        rec = recover_all(S, keys)
        cleaned = np.stack([codebook[resolve(r, codebook)[0]] for r in rec])
        scores = cleaned @ np.asarray(query, float)
    win = int(np.argmax(scores))
    return win, float(scores[win]), scores


# ---- W5: HIERARCHICAL SUPERPOSITION over a shared chunk codebook (backlog W5; R3's third consumer) -------------
#
# THE THEOREM-SHAPED NEGATIVE, first, because it kills the obvious design. Superposition is LINEAR, so a naive
# bundle-of-bundles with product roles IS one flat bundle -- nesting alone buys exactly nothing. Measured:
#
#     sum_g bind(g_key, sum_i bind(l_key_i, item_i))   vs   sum_{g,i} bind(bind(g_key, l_key_i), item_i)
#     max|difference| = 2.78e-16     <- the same vector, to machine precision
#
# The escape is a CLEANUP BETWEEN LEVELS. Unbind the group key, snap the noisy chunk back to its exact pattern in a
# chunk codebook, and the descent below that point is crosstalk-free -- so capacity is bounded by the WORST SINGLE
# LEVEL, not by the product of levels. Measured (D=2048, 8 items/group, 16 shared patterns, 60 trials/point):
#
#     G groups   M leaves   flat recall   hier + mid-cleanup
#          4         32        100.0%          100.0%
#         16        128         90.0%          100.0%
#         32        256         56.7%          100.0%
#         64        512         18.3%          100.0%
#
# CORRECTION TO THE BACKLOG, measured. It says the shared codebook is what "buys the win", implying recall fails
# without shared chunks. It does not: with 64 DISTINCT patterns for 64 groups, hierarchical recall is still 100.0%.
# What sharing buys is a SMALL CODEBOOK -- 16 patterns (256 KB) instead of 64 (1024 KB) at D=2048, a 4x saving that
# grows with the reuse rate. The honest statement is therefore:
#
#     * the mid-level cleanup buys RECALL, and it needs a chunk codebook (of any size) to snap against;
#     * SHARED chunks buy the codebook's SIZE, and that is where R1's promoted chunks pay.
#
# AND THE LAW, MADE NUMERIC. The cleanup does not make the descent noiseless -- it makes the descent see ONLY the
# level below it. After the snap, the leaf unbind's similarity to the truth is exactly the LEAF level's own
# crosstalk, 1/sqrt(leaf width), independent of how many groups are in the bundle:
#
#     leaves/group     2       4       8      16
#     measured      0.705   0.498   0.352   0.251
#     1/sqrt(w)     0.707   0.500   0.354   0.250
#
# "Capacity is bounded by the worst single level, not the product of levels" is not a slogan. It is this number.
# (Recall of ~100% is the MODAL result at these widths, not a guarantee: one seed in four measures 0.975 at G=64.
# The tests assert >= 0.95, because asserting 1.0 would be asserting a seed.)
#
# And the deeper honesty this makes unavoidable: hierarchical superposition does not store 512 items in 2048 floats.
# The atoms live in the codebooks. What the single vector holds is the STRUCTURE -- which group sits where -- and
# the codebooks hold the content. Say it that way and the capacity claim is true; say it the other way and it is a
# conjuring trick.

def hierarchical_pack(group_keys, chunk_vectors):
    """Level-2 pack: superpose G group-keyed CHUNKS into one vector. Each chunk is itself a `pack` of its leaves.

    This is deliberately just `pack` -- the hierarchy is not in the packing (it cannot be; superposition is linear)
    but in the RECALL, which cleans up between the levels. Kept as a named function so the two levels read
    symmetrically at the call site and nobody is tempted to invent a nesting operator that does not exist."""
    return pack(group_keys, chunk_vectors)


def chunk_codebook_vectors(codebook, item_codebook, leaf_keys):
    """THE R3 BRIDGE: realize a LEARNED chunk codebook (holographic_chunkcodebook, R1) as the chunk vectors this
    module's hierarchy cleans up against. Returns (chunk_vectors (n, D), chunk_ids, leaf_lists).

    Only tokens whose leaf-expansion has exactly `len(leaf_keys)` leaves are usable -- a chunk must fill the slots.

    WHAT IS SHARED AND WHAT IS NOT. R1 learns WHICH chunks recur; R2 (`resonator.recursive_factor`) realizes each
    as a `map_bind` PRODUCT of its leaves; this module realizes each as a `pack` SUPERPOSITION of its keyed leaves.
    Same chunk IDENTITIES, different vector realizations -- and that is what R3's "one codebook family" actually
    means. Saying the three consumers share a *vector* would be false; they share the promoted structure."""
    from holographic.agents_and_reasoning.holographic_chunkcodebook import ChunkCodebook
    cb = codebook if isinstance(codebook, ChunkCodebook) else ChunkCodebook.from_dict(codebook)
    leaf_keys = np.asarray(leaf_keys, float)
    n_leaves = leaf_keys.shape[0]
    items = np.asarray(item_codebook, float)
    ids = sorted(t for t, d in cb.depth.items() if d == n_leaves)
    if not ids:
        return np.empty((0, items.shape[1])), [], []
    leaf_lists = [cb.decode([t]) for t in ids]
    vecs = np.stack([pack(leaf_keys, items[np.asarray(ls, int)]) for ls in leaf_lists])
    return vecs, ids, leaf_lists


def chunk_coverage(codebook, groups, n_leaves):
    """What fraction of the groups actually used are PRESENT in the learned chunk codebook? Returns
    {covered, total, fraction, missing}.

    This is not bookkeeping. An uncovered group's mid-level cleanup snaps to the WRONG chunk -- see
    `hierarchical_recall`'s `min_chunk_similarity` -- so coverage is the precondition of the whole capacity claim,
    and it is set by how many merges R1 was allowed. Measured on a 16-group stream: 60 merges promoted 8 of 16
    groups; 150 merges promoted all 16."""
    from holographic.agents_and_reasoning.holographic_chunkcodebook import ChunkCodebook
    cb = codebook if isinstance(codebook, ChunkCodebook) else ChunkCodebook.from_dict(codebook)
    have = {tuple(cb.decode([t])) for t, d in cb.depth.items() if d == int(n_leaves)}
    groups = [tuple(int(i) for i in g) for g in groups]
    missing = [g for g in groups if g not in have]
    total = len(groups)
    return {"covered": total - len(missing), "total": total,
            "fraction": (total - len(missing)) / total if total else 0.0, "missing": missing}


def hierarchical_recall(S, group_key, leaf_key, chunk_codebook, item_codebook, min_chunk_similarity=None):
    """Descend one hierarchical superposition with a CLEANUP at the middle level.

        1. unbind the group key   -> a noisy chunk
        2. RESOLVE it against `chunk_codebook`  -> the EXACT chunk, crosstalk reset
        3. unbind the leaf key from the exact chunk -> a noisy item
        4. RESOLVE it against `item_codebook`   -> the recovered item

    Returns {item_index, item, chunk_index, chunk_similarity, item_similarity, abstained}. Step 2 is the whole
    mechanism: the discrete decision resets the noise, so the leaf unbind sees a clean chunk rather than a chunk
    plus G-1 others. Without it, this function is algebraically identical to a flat recall (see the module note).

    MEASURED at D=2048, G=64 groups x 8 leaves: 18.3% flat, 100.0% here. Reproduced against `flat_recall` on a
    LEARNED codebook (R1): flat 100/95/70/30 at G=4/16/32/64, hierarchical 100/100/100/100.

    `min_chunk_similarity` -- THE ABSTAIN GATE, and it exists because of a measured failure. If the group being
    recalled is NOT in `chunk_codebook` (R1 promoted fewer chunks than the workload uses), step 2 snaps the noisy
    chunk to the NEAREST codebook entry, which is simply the wrong chunk, and step 4 then returns a wrong item with
    every appearance of success. Measured: an uncovered group cleaned up at chunk_similarity **0.036**, against
    **0.502** for a covered one -- the signal is there, it just has to be read. Set the threshold (0.15 separates
    those cleanly) and the recall ABSTAINS instead of lying: `abstained=True`, `item_index=None`.

    Default `None` keeps the historical behaviour exactly, so no existing caller changes."""
    S = np.asarray(S, float)
    noisy_chunk = unbind(S, np.asarray(group_key, float))
    ci, csim = resolve(noisy_chunk, chunk_codebook)                # the MID-LEVEL CLEANUP: crosstalk reset
    if min_chunk_similarity is not None and csim < float(min_chunk_similarity):
        return {"item_index": None, "item": None, "chunk_index": ci,
                "chunk_similarity": csim, "item_similarity": 0.0, "abstained": True}
    exact_chunk = np.asarray(chunk_codebook, float)[ci]
    noisy_item = unbind(exact_chunk, np.asarray(leaf_key, float))
    ii, isim = resolve(noisy_item, item_codebook)
    return {"item_index": ii, "item": np.asarray(item_codebook, float)[ii],
            "chunk_index": ci, "chunk_similarity": csim, "item_similarity": isim, "abstained": False}


def flat_recall(S, group_key, leaf_key, item_codebook):
    """The BASELINE hierarchical_recall must beat, and the strongest honest one: unbind both roles from the single
    bundle and clean up ONCE at the bottom, with no mid-level snap. Returns the same dict shape (no chunk fields).

    Shipped beside its competitor on purpose, so the comparison can be re-run rather than taken on trust."""
    noisy = unbind(unbind(np.asarray(S, float), np.asarray(group_key, float)), np.asarray(leaf_key, float))
    ii, isim = resolve(noisy, item_codebook)
    return {"item_index": ii, "item": np.asarray(item_codebook, float)[ii], "item_similarity": isim}


def _selftest():
    """Two checks: (1) the UNAMBIGUOUS property -- a single keyed item comes back EXACTLY with a
    unitary key; (2) the honest capacity behaviour -- parallel readout tracks direct computation
    at small width and decays as width grows past the wall (informational, not asserted)."""
    rng = np.random.default_rng(0)
    from holographic.agents_and_reasoning.holographic_ai import unitary_vector, random_vector
    D = 512
    # (1) exact single-item recovery -- the contract the module rests on
    k = unitary_vector(D, rng); v = random_vector(D, rng)
    S1 = pack(k[None, :], v[None, :])
    rec1 = recover_all(S1, k[None, :])[0]
    assert cosine(rec1, v) > 0.999, "single keyed item must recover exactly with a unitary key"
    # (2) parallel readout fidelity across width -- mean per-item recovery cosine (stable,
    #     monotone), the capacity wall shown not hidden
    print(f"[superposed selftest] D={D}: single-item recovery cosine = {cosine(rec1, v):.4f} (exact)")
    for K in (2, 4, 8, 16, 32, 64, 128):
        fids = []
        for s in range(8):
            r = np.random.default_rng(100 + s)
            keys = np.stack([unitary_vector(D, r) for _ in range(K)])
            items = np.stack([random_vector(D, r) for _ in range(K)])
            rec = recover_all(pack(keys, items), keys)
            fids.append(np.mean([cosine(rec[i], items[i]) for i in range(K)]))
        print(f"    K={K:4d}: mean recovery cosine = {np.mean(fids):.3f}")
    print("[superposed selftest] OK -- exact at K=1, decays smoothly with width as the wall predicts")

    # -- W5: hierarchical superposition --------------------------------------------------------------------
    from holographic.agents_and_reasoning.holographic_ai import unitary_vector
    D = 2048
    r = np.random.default_rng(3)
    u = lambda n: np.stack([unitary_vector(D, r) for _ in range(n)])
    per, G, npat = 8, 64, 16
    lk, gk, ib = u(per), u(G), u(per * npat)
    chunks = np.stack([pack(lk, ib[p * per:(p + 1) * per]) for p in range(npat)])
    asg = r.integers(0, npat, G)

    # (1) THE THEOREM-SHAPED NEGATIVE: nesting alone IS the flat bundle. Binding distributes over the bundle, so
    #     bind(g, sum_i bind(l_i, x_i)) == sum_i bind(bind(g, l_i), x_i). Compared as raw SUMS (bundle() may
    #     normalise, which would hide the identity behind a scale factor). Pinned so nobody ships a "nesting
    #     operator" and reports the flat number as a hierarchy win.
    small = 4
    nested_sum = sum(bind(gk[g], sum(bind(lk[i], ib[i]) for i in range(per))) for g in range(small))
    flat_sum = sum(bind(bind(gk[g], lk[i]), ib[i]) for g in range(small) for i in range(per))
    assert np.abs(nested_sum - flat_sum).max() < 1e-9, "nesting must be literally the flat bundle"

    # (2) THE MECHANISM: the mid-level cleanup. Hierarchy recalls where flat has collapsed.
    S = hierarchical_pack(gk, chunks[asg])
    Sf = pack(np.stack([bind(gk[g], lk[i]) for g in range(G) for i in range(per)]),
              np.stack([ib[asg[g] * per + i] for g in range(G) for i in range(per)]))
    ok_h = ok_f = 0
    for _ in range(40):
        g0, i0 = int(r.integers(0, G)), int(r.integers(0, per))
        truth = asg[g0] * per + i0
        ok_h += int(hierarchical_recall(S, gk[g0], lk[i0], chunks, ib)["item_index"] == truth)
        ok_f += int(flat_recall(Sf, gk[g0], lk[i0], ib)["item_index"] == truth)
    assert ok_h >= 38, ("mid-level cleanup must give near-exact recall", ok_h)   # 1.0 is modal, not guaranteed
    assert ok_f < 30, ("the flat baseline is supposed to have collapsed at G=64", ok_f)

    # -- R3: the LEARNED codebook (R1) as the chunk codebook, and the coverage gate it forces ------------------
    from holographic.agents_and_reasoning.holographic_chunkcodebook import learn_chunks

    _r = np.random.default_rng(11)
    _D = 512
    _items = _r.normal(size=(32, _D))
    _items /= np.linalg.norm(_items, axis=1, keepdims=True)

    def _unit(n):
        ph = _r.uniform(0.0, 2 * np.pi, (n, _D // 2 + 1))
        ph[:, 0] = 0.0
        ph[:, -1] = 0.0
        return np.fft.irfft(np.exp(1j * ph), n=_D, axis=1)

    _leaf_keys, _group_keys = _unit(4), _unit(8)
    _groups = [list(_r.choice(32, 4, replace=False)) for _ in range(4)]
    _stream = []
    for _ in range(200):
        _stream += _groups[_r.integers(0, 4)]

    _cb = learn_chunks(_stream, max_merges=40)
    _vecs, _ids, _leaves = chunk_codebook_vectors(_cb, _items, _leaf_keys)
    assert len(_ids) >= 2 and _vecs.shape[1] == _D
    # every learned chunk's leaves are one of the groups that actually occurred
    assert all(tuple(ls) in {tuple(g) for g in _groups} for ls in _leaves)

    _cov = chunk_coverage(_cb, _groups, 4)
    assert _cov["total"] == 4 and 0 < _cov["fraction"] <= 1.0

    # a COVERED group recalls exactly; the abstain gate does not fire on it
    _S = hierarchical_pack(_group_keys[:2], _vecs[[0, 1]])
    _hit = hierarchical_recall(_S, _group_keys[0], _leaf_keys[2], _vecs, _items, min_chunk_similarity=0.15)
    assert _hit["abstained"] is False and _hit["item_index"] == _leaves[0][2]

    # AN UNCOVERED GROUP: without the gate the cleanup snaps to the WRONG chunk and answers confidently.
    _absent = [g for g in _groups if tuple(g) not in {tuple(ls) for ls in _leaves}]
    if _absent:
        _bad = hierarchical_pack(_group_keys[:2], np.stack([pack(_leaf_keys, _items[np.asarray(_absent[0], int)]),
                                                            _vecs[0]]))
        _open = hierarchical_recall(_bad, _group_keys[0], _leaf_keys[2], _vecs, _items)
        assert _open["abstained"] is False and _open["item_index"] != _absent[0][2]     # a confident WRONG answer
        _gated = hierarchical_recall(_bad, _group_keys[0], _leaf_keys[2], _vecs, _items, min_chunk_similarity=0.15)
        assert _gated["abstained"] is True and _gated["item_index"] is None             # ... refused instead

    # backward compatibility: no threshold => the historical behaviour, and abstained is always False
    assert hierarchical_recall(_S, _group_keys[0], _leaf_keys[0], _vecs, _items)["abstained"] is False

    print("[superposed selftest] R3 OK -- a LEARNED chunk codebook (R1) realized as packed chunk vectors: "
          "%d chunks promoted, coverage %.0f%%; a covered group recalls exactly, and an uncovered group snaps to "
          "the wrong chunk and answers CONFIDENTLY unless min_chunk_similarity abstains"
          % (len(_ids), 100 * _cov["fraction"]))

    print("[superposed selftest] W5 OK -- naive nesting IS the flat bundle (linearity); with a mid-level cleanup "
          "against a %d-pattern chunk codebook, G=%d groups x %d leaves recalls %d/40 vs the flat baseline's %d/40"
          % (npat, G, per, ok_h, ok_f))


if __name__ == "__main__":
    _selftest()

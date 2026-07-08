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


if __name__ == "__main__":
    _selftest()

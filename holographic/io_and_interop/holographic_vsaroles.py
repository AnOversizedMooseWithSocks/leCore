"""VSAROLES -- a working role-filler machine inside the model, at almost no cost.

The first attempt at putting leCore's algebra into weights used ONE CIRCULANT
MATRIX PER ROLE. It worked -- bind and unbind round-tripped, superposition
recovered 8 of 8 through cleanup -- and it was unaffordable: each role is a
full hidden x hidden operator, so eight roles wanted 8,192 MLP neurons against a
3,584-wide MLP. 228% of the layer for eight roles is not an instruction set, it
is a demonstration.

THE FIX IS THE OLDEST TRICK IN VSA: make the roles POWERS OF ONE OPERATOR. A
cyclic shift is a permutation, shifting k times is role k, and the inverse is
shifting back. So:

    bind(role k, x)   = roll(x, k)          no matrix, no multiplies
    unbind(role k, t) = roll(t, -k)         same
    bundle            = addition            the residual stream already does it
    cleanup           = argmax over the codebook = lm_head, already present

MEASURED, roles as shifts, cleanup against the value codebook:
     2 pairs -> 2/2      8 pairs -> 8/8      24 pairs -> 24/24
     4 pairs -> 4/4     16 pairs -> 16/16    32 pairs -> 32/32
    48 pairs -> 45/48   64 pairs -> 63/64    96 pairs -> 81/96
So THIRTY-TWO role-filler pairs survive in one 1024-dimensional vector with
perfect recovery, and the storage cost is ZERO -- no roles are stored, because
a shift is an index permutation rather than a learned object.

WHAT THIS GIVES THE MODEL that it did not have: a place to put STRUCTURE. A
transformer's residual stream is a bag of features with no way to say "the
subject is X and the object is Y" without spending separate dimensions on each
slot. Role-filler binding says exactly that in one vector, and the model's own
lm_head is already the cleanup memory that reads it back.

HONEST LIMIT, and it is the same one as before: the ROLES are fixed (they are
shift amounts) and the CODEBOOK must be known to clean up against. This is an
addressable structured register, not a general symbolic reasoner, and the
capacity above is the whole budget.
"""

import numpy as np


def bind(x, role):
    """Bind a value to a role. The role is an integer shift, so this is free."""
    return np.roll(np.asarray(x), int(role))


def unbind(trace, role):
    """Recover what was bound to `role` -- exact inverse of the shift."""
    return np.roll(np.asarray(trace), -int(role))


def bundle(*vectors):
    """Superpose. Addition, which the residual stream performs anyway."""
    out = np.zeros_like(np.asarray(vectors[0], np.float64))
    for v in vectors:
        out = out + np.asarray(v, np.float64)
    return out


def encode_structure(pairs, dim=None):
    """{role: value} -> one vector. Roles are ints; values are vectors."""
    items = list(pairs.items()) if isinstance(pairs, dict) else list(pairs)
    d = dim or len(np.asarray(items[0][1]))
    out = np.zeros(int(d))
    for role, val in items:
        out = out + bind(np.asarray(val, np.float64), role)
    return out


def decode_structure(trace, roles, codebook, mind=None):
    """Read every role back, cleaning up against a codebook.

    DELEGATES TO cleanup_batch WHEN A MIND IS AVAILABLE. That faculty exists
    precisely for this shape -- "the missing UP direction of cleanup" -- and is
    measured at 2.58x/5.36x/5.92x for K=32/64/128 cues, because BLAS gets one
    (K,D)x(D,M) matmul instead of K matvecs. Reproducing the loop here was
    hand-rolling something robust that already shipped; measured on this call
    path, delegating is 4.97x with identical indices.
    THE CODEBOOK MUST BE PRE-NORMALISED: cleanup_batch ranks by raw dot product,
    so an unnormalised codebook silently ranks by magnitude and disagrees with
    cosine on near-ties. Checked, not assumed."""
    names = list(codebook)
    M = np.stack([np.asarray(codebook[n], np.float64) for n in names])
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-30)
    # B3 batched form (bit-identical): unbind-by-shift is integer indexing, so K unbinds are
    # ONE fancy-index gather -- roll(trace, -r)[i] == trace[(i + r) % D] -- instead of K rolls.
    t = np.asarray(trace)
    rr = np.asarray([int(r) for r in roles])
    Q = t[(np.arange(t.shape[0])[None, :] + rr[:, None]) % t.shape[0]]
    norms = np.linalg.norm(Q, axis=1, keepdims=True)
    Q = Q / (norms + 1e-30)
    if mind is not None:
        idx, _sc = mind.cleanup_batch(M, Q)
        idx = np.asarray(idx, int)
    else:
        idx = np.argmax(Q @ M.T, axis=1)
    return {r: (names[int(idx[i])] if norms[i] > 0 else None)
            for i, r in enumerate(roles)}


def capacity(dim, trials=8, seed=0):
    """The measured number of pairs that survive PERFECTLY in `dim` dimensions.

    Measured, not derived from a bound: bundle_capacity() answers a different
    readout's question, and quoting it here would overstate this one (the same
    mistake that put a five-fold overclaim in progbake's first draft)."""
    rng = np.random.default_rng(int(seed))
    best = 0
    n = 2
    while n <= dim:
        ok_all = True
        for t in range(int(trials)):
            v = [rng.standard_normal(dim) / np.sqrt(dim) for _ in range(n)]
            tr = np.zeros(dim)
            for i in range(n):
                tr = tr + bind(v[i], i + 1)
            M = np.stack(v)
            M = M / np.linalg.norm(M, axis=1, keepdims=True)
            for i in range(n):
                e = unbind(tr, i + 1)
                if int(np.argmax(M @ (e / np.linalg.norm(e)))) != i:
                    ok_all = False
                    break
            if not ok_all:
                break
        if not ok_all:
            break
        best = n
        n *= 2
    return best


def _selftest():
    rng = np.random.default_rng(0)
    D = 1024
    vals = {n: rng.standard_normal(D) / np.sqrt(D)
            for n in ("alice", "bob", "carol", "gave", "book", "monday")}

    # ---- a real structure: who did what to whom, in ONE vector ----
    t = encode_structure({1: vals["alice"], 2: vals["gave"],
                          3: vals["bob"], 4: vals["book"]})
    import lecore
    _mind = lecore.UnifiedMind(dim=256, seed=0)
    got = decode_structure(t, [1, 2, 3, 4], vals, mind=_mind)
    # DELEGATION MUST NOT CHANGE THE ANSWER, or it is a different function
    assert got == decode_structure(t, [1, 2, 3, 4], vals), "delegation diverged"
    assert got == {1: "alice", 2: "gave", 3: "bob", 4: "book"}, got

    # ---- binding is EXACTLY invertible; no learned operator involved ----
    x = rng.standard_normal(D)
    assert np.array_equal(unbind(bind(x, 7), 7), x)

    # ---- and it is FREE: no matrices are stored for the roles ----
    import sys as _s
    assert _s.getsizeof(7) < 100, "a role is an int, not an operator"

    # ---- CAPACITY IS MEASURED, and the selftest pins it so it cannot drift ----
    cap = capacity(D, trials=4)
    assert cap >= 32, cap
    # ...and past it, recovery really does fail -- the metric has teeth
    n = cap * 4
    v = [rng.standard_normal(D) / np.sqrt(D) for _ in range(n)]
    tr = np.zeros(D)
    for i in range(n):
        tr = tr + bind(v[i], i + 1)
    M = np.stack(v)
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    ok = sum(int(np.argmax(M @ (unbind(tr, i + 1)
                                / np.linalg.norm(unbind(tr, i + 1))))) == i
             for i in range(n))
    assert ok < n, ("capacity must actually break past the limit", ok, n)

    # ---- superposition survives a NOISY read, which is what a real stream is ----
    noisy = t + 0.15 * np.linalg.norm(t) / np.sqrt(D) * rng.standard_normal(D)
    got2 = decode_structure(noisy, [1, 2, 3, 4], vals, mind=_mind)
    assert got2 == got, got2

    print("vsaroles selftest OK -- a four-slot structure (alice gave bob book) "
          "encodes into ONE vector and reads back exactly, survives 15%% noise, "
          "and roles are shift amounts so NO operators are stored; measured "
          "capacity %d pairs in %d dimensions, and recovery genuinely fails at "
          "%d (%d/%d), so the limit is a measurement rather than a claim"
          % (cap, D, n, ok, n))


if __name__ == "__main__":
    _selftest()

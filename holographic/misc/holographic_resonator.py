"""Factoring a composite back into its parts -- the inverse of binding, solved by
searching in superposition.

Binding combines several vectors into one (the cat, the colour red, the position
top-left -> one vector for "a red cat at top-left"). The hard inverse is
FACTORIZATION: given only that one composite vector and the codebooks of possible
parts, recover which part came from each codebook. Brute force is the product of the
codebook sizes -- for three codebooks of 100 that is a million combinations, and it
explodes with more factors. A Resonator Network (Frady, Kent, Olshausen & Sommer,
2020) solves it without enumerating that space, by "computing in superposition":
each factor's estimate is a weighted blend of ALL its codebook's vectors at once,
and the estimates are refined together until the true factors resonate out of the
mixture and the rest cancel.

THE LOOP (MAP / bipolar binding, where binding is elementwise product and is its own
inverse -- the regime the resonator is defined for; our circular-convolution bind
amplifies noise too much to factor stably, a kept finding):
  * hold all factors but one fixed; unbind them from the composite (multiply them
    back in) to get a noisy estimate of the remaining factor
  * CLEAN UP that estimate toward its codebook: project onto the codebook
    (similarity to every codevector, then superpose them back by those similarities)
    and take the sign -- a superposition that sharpens toward the best match
  * do this for every factor each step, from the previous step's estimates
    (simultaneous update), and iterate until re-binding the nearest codevectors
    reproduces the composite exactly
A resonator is not guaranteed to converge, but when it does it lands on the correct
factorization, so RANDOM RESTARTS turn it into a reliable solver: restart from a new
random state on failure and keep the first run that re-binds to the target.

WHAT WAS MEASURED, honestly, on this substrate:
  * It WORKS, and searches far more than it enumerates: 3 codebooks of 50 (125,000
    combinations) solved 20/20 with a median of ~2 restarts; 3 of 100 (1,000,000
    combinations) solved ~11/20 at dimension 3000 -- the classic dimension-vs-
    capacity tradeoff (capacity grows ~quadratically with dimension).
  * A KEPT NEGATIVE: with the engine's native circular-convolution binding the
    resonator does NOT converge -- unbinding by involution amplifies the
    cross-term noise each step and the search never settles (0-1/20). Factorization
    needs a self-inverse, noise-stable bind, so this module uses MAP (bipolar)
    binding internally and is explicit that it is a different bind from the rest of
    the engine. The lesson: the operation you can invert in superposition depends on
    the algebra you bind with.

This is the decomposition primitive the engine was missing: not segmenting a stream
(holographic_segment) but pulling a single bound representation apart into the
independent factors that composed it -- combinatorial search done by projection and
settling, the same loop-until-resolved shape as cleanup, coarse_to_fine and the
meaning settler, now solving an exponential search.

Needs: numpy.
"""
import numpy as np
from holographic.misc.holographic_determinism import argmax_tiebreak


def map_codebook(n_codes, dim, seed):
    """A codebook of n_codes random bipolar (+/-1) vectors of length dim."""
    rng = np.random.default_rng(seed)
    return np.where(rng.random((n_codes, dim)) < 0.5, -1.0, 1.0)


def map_bind(*vectors):
    """MAP binding: elementwise product. Self-inverse (binding twice cancels), which
    is what makes stable factorization possible."""
    out = np.ones_like(vectors[0], dtype=float)
    for v in vectors:
        out = out * v
    return out


class ResonatorNetwork:
    """Factor a composite vector into one codevector from each codebook, by searching
    in superposition with random restarts."""

    def __init__(self, codebooks):
        # codebooks: list of (n_f, dim) bipolar matrices, rows = codevectors
        self.books = [np.asarray(B, float) for B in codebooks]
        self.F = len(self.books)
        self.dim = self.books[0].shape[1]

    def _cleanup(self, est, B):
        """Project an estimate onto a codebook and sharpen: similarity to each
        codevector, superpose them back weighted by those similarities, take sign."""
        return np.sign(B.T @ (B @ est))

    def _run(self, c, iters, seed):
        rng = np.random.default_rng(seed)
        ests = [np.sign(rng.standard_normal(self.dim)) for _ in range(self.F)]
        # UNIFIER (P7): one iteration of this loop IS an iterate-a-projection sweep -- each factor is projected
        # onto its own codebook (cleanup is idempotent, measured) while the others are held fixed. Because the
        # factors are DISJOINT blocks, the "simultaneous" (Jacobi) sweep of `project_onto_constraints` sums the
        # block moves and reproduces this update EXACTLY -- verified bit-for-bit. We delegate the sweep and keep
        # our own exit conditions (exact reconstruction / stuck detection), which the generic engine has no
        # opinion about.
        from holographic.rendering.holographic_denoise import project_onto_constraints

        def _projection(f):
            def proj(x):
                blocks = x.reshape(self.F, self.dim)
                others = c.copy()
                for g in range(self.F):
                    if g != f:
                        others = others * blocks[g]
                out = blocks.copy()
                out[f] = self._cleanup(others, self.books[f])
                return out.reshape(-1)
            return proj

        projections = [_projection(f) for f in range(self.F)]
        for t in range(iters):
            stacked, _sweeps, _conv = project_onto_constraints(
                np.concatenate(ests), projections, iters=1, sweep="simultaneous")
            new = list(stacked.reshape(self.F, self.dim))
            # DETERMINISM CONTRACT (ISA-1): cite argmax_tiebreak, don't hand-roll. The scores' last bits are not
            # stable across backends/orders, so the WINNER must come from the named rule (ties -> lowest index).
            idx = tuple(argmax_tiebreak(self.books[f] @ new[f]) for f in range(self.F))
            rec = np.ones(self.dim)
            for f in range(self.F):
                rec = rec * self.books[f][idx[f]]
            if np.array_equal(rec, c):
                return idx, t, True
            if all(np.array_equal(new[f], ests[f]) for f in range(self.F)):
                return idx, t, False                 # stuck at a wrong fixed point
            ests = new
        return idx, iters, False

    def factor(self, composite, restarts=20, iters=400):
        """Recover the factor indices (one per codebook) whose binding equals the
        composite. Returns {'factors', 'solved', 'restarts', 'iterations',
        'search_space'}. Tries random restarts because a single run may not converge,
        but a converged run is always correct."""
        c = np.asarray(composite, float)
        space = 1
        for B in self.books:
            space *= B.shape[0]
        for r in range(restarts):
            idx, t, ok = self._run(c, iters, seed=r)
            if ok:
                return {"factors": idx, "solved": True, "restarts": r + 1,
                        "iterations": t, "search_space": space}
        return {"factors": idx, "solved": False, "restarts": restarts,
                "iterations": iters, "search_space": space}

# ---------------------------------------------------------------------------------------------------------
# RECURSIVE FACTORING over learned chunk levels (backlog R2; R3's "one codebook family", second consumer).
#
# The flat resonator has a combinatorial cliff. Measured (D=4096, 32-symbol vocab, MAP binding, distinct symbols,
# 10 restarts x 300 iters):
#
#     depth 2   93.3%      depth 4   60.0%      depth 5   0.0%      depth 6   0.0%      depth 8   0.0%
#
# It is a CLIFF, not a slope: at this vocabulary depth 5 is not "harder", it is gone.
#
# THE CLIFF IS SET BY THE SEARCH SPACE V^depth, NOT BY DEPTH -- a correction to my own first draft, which said
# "depth 5 is gone" without qualification and was wrong. Measured, 6 restarts x 200 iters:
#
#     V=12, depth 6  (V^d = 3.0e6)   ->  2/5 solved      <- a small vocabulary survives depth 6
#     V=32, depth 5  (V^d = 3.4e7)   ->  0/5
#     V=32, depth 6  (V^d = 1.1e9)   ->  0/5
#
# So "past the cliff" means past a search-space budget, and a caller with a 12-symbol alphabet should not expect
# to need this at depth 6. Dimension buys some of it back (D=4096 beats D=2048 at the margin) but not the shape.
#
# THE FRACTAL MOVE: factor depth-k as a depth-2 problem over a codebook of composed CHUNKS, then expand each
# chunk by LOOKUP instead of by search. Every level self-verifies by re-composition, so the rare failure is
# DETECTED, never silent, and the search falls back one level down.
#
# MEASURED, and both halves matter:
#
#     depth 4, all-pairs macro codebook (496 entries)    recursive 93.3%   vs flat 86.7%   -- but 5x SLOWER
#     depth 8, promoted chunks (62 pairs -> 64 quads)    recursive 90.0%   vs flat  0.0%   -- and 3x FASTER
#
# So the honest statement is conditional. Below the cliff, recursion is a modest accuracy gain paid for with real
# time, because the macro codebook (O(V^2)) is larger than the vocabulary and the flat search was working. PAST
# the cliff it is the difference between a result and nothing, and it is faster besides, because a 64-entry
# promoted codebook is a smaller search space than V^8.
#
# THE CONDITION, and it is the program's recurring law: the depth-8 win needs PROMOTED chunks -- a codebook of the
# chunks that actually recur, which is what `holographic_chunkcodebook.learn_chunks` (R1) produces from a stream.
# An all-pairs codebook grows quadratically and cannot reach depth 8; a learned one covers only the structure that
# is there. No structure, no recursion dividend -- and `structure_score` measures that before this is attempted.
#
# NOTE ON THE MACRO CODEBOOK'S SIZE: the backlog says 528 = C(33,2) all-pairs entries for a 32-vocab. That counts
# the 32 self-pairs. Under MAP binding bind(x, x) is the all-ones vector for EVERY x, so those 32 entries are one
# degenerate atom, not 32 distinguishable ones. The usable all-pairs codebook is C(32,2) = 496.
# ---------------------------------------------------------------------------------------------------------

def chunk_vector(token, codebook, vocab):
    """The MAP vector of a chunk token: bind its leaf expansion. A leaf is its own vocabulary row.

    `codebook` is a holographic_chunkcodebook.ChunkCodebook (R3: the SAME object that R1 learns and W5/DL8 use);
    `vocab` is the (V, D) base codebook."""
    leaves = codebook.decode([int(token)])
    return map_bind(*[np.asarray(vocab)[int(i)] for i in leaves])


def level_codebook(codebook, vocab, depth):
    """Every token in `codebook` whose leaf-expansion has exactly `depth` leaves, as (matrix (n, D), token ids).

    `depth=1` is the base vocabulary itself. Tokens are returned in ascending id order, so the matrix is
    deterministic and a factor index maps back to a token reproducibly."""
    vocab = np.asarray(vocab, float)
    if int(depth) == 1:
        return vocab, list(range(vocab.shape[0]))
    ids = sorted(t for t, d in codebook.depth.items() if d == int(depth))
    if not ids:
        return np.empty((0, vocab.shape[1])), []
    return np.stack([chunk_vector(t, codebook, vocab) for t in ids]), ids


def available_levels(codebook, vocab):
    """The chunk depths this codebook can factor against, DEEPEST FIRST -- the ladder `recursive_factor` descends.
    Only depths with at least `arity` distinct tokens are useful, but that gate lives in the caller."""
    depths = sorted({d for d in codebook.depth.values()}, reverse=True)
    return [d for d in depths if d >= 1]


def reduce_involution(leaves):
    """MAP binding is SELF-INVERSE (`x * x` is the all-ones vector), so a leaf that appears twice CANCELS. The
    recoverable object is therefore the leaf multiset **modulo pairs of duplicates**, and a factorization can be
    exactly correct while carrying redundant pairs.

    Measured, and it surprised me: factoring `bind(v3, v7)` against a pair codebook returned leaves [0, 0, 3, 7]
    -- because `bind(v0,v3) * bind(v0,v7) == v3 * v7`. The re-composition gate PASSED, correctly: that expansion
    really does reproduce the composite. It is a different route to the same vector, not an error. Reducing modulo
    the involution recovers the minimal multiset without weakening the gate (dropping a cancelling pair leaves the
    product unchanged, exactly)."""
    from collections import Counter
    counts = Counter(int(i) for i in leaves)
    return sorted(t for t, n in counts.items() for _ in range(n % 2))


def recursive_factor(composite, codebook, vocab, arity=2, restarts=10, iters=300, tol=1e-6):
    """Factor a deep composite by searching a SHALLOW problem over composed chunks, then expanding by lookup.

    Tries each chunk level deepest-first: run an `arity`-way resonator over that level's codebook, expand each
    recovered token to its leaves, and VERIFY by re-composition -- accept only if binding the leaves reproduces
    `composite`. On failure, fall back one level down, ending at the flat base vocabulary. So the answer is either
    verified correct or reported unsolved; it is never a silent guess.

    Returns {leaves, solved, level, verified, tried, search_space}. `leaves` is sorted and reduced modulo the MAP
    involution (see `reduce_involution`): binding is commutative AND self-inverse, so the recoverable object is the
    leaf multiset with duplicate pairs cancelled -- not the sequence, and not the raw expansion.

    MEASURED: depth 8 with promoted chunks -- 90.0% here, 0.0% flat. Depth 4 with an all-pairs codebook -- 93.3%
    here, 86.7% flat, at 5x the time. Use it past the cliff; below it, the flat resonator is already working."""
    c = np.asarray(composite, float)
    vocab = np.asarray(vocab, float)
    tried = []
    for depth in available_levels(codebook, vocab):
        book, ids = level_codebook(codebook, vocab, depth)
        if len(ids) < 2:
            continue
        tried.append(int(depth))
        res = ResonatorNetwork([book] * int(arity)).factor(c, restarts=restarts, iters=iters)
        if not res["solved"]:
            continue
        leaves = []
        for f in res["factors"]:
            leaves.extend(codebook.decode([int(ids[int(f)])]) if depth > 1 else [int(ids[int(f)])])
        # THE VERIFY GATE: re-compose and compare. A wrong pairing cannot survive this, so a failure at one level
        # costs a fallback, not a wrong answer.
        # THE VERIFY GATE runs on the RAW expansion (that is what the resonator actually claimed), and only then
        # is the answer reduced modulo the involution -- reducing first would be assuming the thing being checked.
        if np.max(np.abs(map_bind(*[vocab[int(i)] for i in leaves]) - c)) <= tol:
            return {"leaves": reduce_involution(leaves), "solved": True, "level": int(depth),
                    "verified": True, "tried": tried, "search_space": int(res["search_space"])}
    return {"leaves": [], "solved": False, "level": None, "verified": False, "tried": tried, "search_space": 0}


def _selftest():
    """Regression trap for R2: the cliff is real, recursion crosses it, and the verify gate refuses rather than
    guesses. Small sizes so the self-test stays fast; the full measurement lives in the tests."""
    import itertools
    from holographic.agents_and_reasoning.holographic_chunkcodebook import ChunkCodebook

    D, V = 2048, 12
    vocab = map_codebook(V, D, seed=0)

    # a hand-built codebook: pairs (depth 2) then quads (depth 4) -- exactly what learn_chunks promotes.
    # DISJOINT pairs, deliberately: overlapping ones cancel under the MAP involution and would make the truth
    # a reduced multiset rather than the leaves themselves (which is the point `reduce_involution` documents).
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    merges = [((a, b), V + i) for i, (a, b) in enumerate(pairs)]
    depth = {t: 1 for t in range(V)}
    for (a, b), nid in merges:
        depth[nid] = depth[a] + depth[b]
    quads = [(V + 0, V + 1), (V + 2, V + 3), (V + 4, V + 5)]
    for i, (a, b) in enumerate(quads):
        nid = V + len(pairs) + i
        merges.append(((a, b), nid))
        depth[nid] = depth[a] + depth[b]
    cb = ChunkCodebook(merges, depth)

    assert available_levels(cb, vocab) == [4, 2, 1]
    book4, ids4 = level_codebook(cb, vocab, 4)
    assert book4.shape == (3, D) and len(ids4) == 3
    assert level_codebook(cb, vocab, 1)[0].shape == (V, D)
    assert level_codebook(cb, vocab, 3)[1] == []                     # no depth-3 tokens: an empty level

    # a depth-8 composite of two learned quads: the flat vocabulary cannot touch this
    q0, q1 = ids4[0], ids4[2]
    truth = reduce_involution(cb.decode([q0]) + cb.decode([q1]))     # disjoint quads: nothing cancels
    comp = map_bind(chunk_vector(q0, cb, vocab), chunk_vector(q1, cb, vocab))

    got = recursive_factor(comp, cb, vocab, restarts=8, iters=200)
    assert got["solved"] and got["verified"] and got["level"] == 4
    assert got["leaves"] == truth == [0, 1, 2, 3, 8, 9, 10, 11]      # all 8 leaves, none cancelled

    # THE VERIFY GATE: a composite built from atoms OUTSIDE every level must be refused, not guessed at.
    junk = map_bind(vocab[0], vocab[1], vocab[2])                    # depth 3: no level can express it
    bad = recursive_factor(junk, cb, vocab, restarts=4, iters=120)
    assert bad["solved"] is False and bad["leaves"] == []

    # A depth-2 composite still solves -- but note WHERE. It is solved at the PAIR level as bind(v0,v3)*bind(v0,v7),
    # whose v0's cancel. The verify gate passes because that really is the composite; `reduce_involution` then
    # recovers the minimal multiset. MAP's self-inverse property means "correct" and "minimal" are different things.
    flat2 = map_bind(vocab[3], vocab[7])
    got2 = recursive_factor(flat2, cb, vocab, restarts=8, iters=200)
    assert got2["solved"] and got2["verified"] and got2["leaves"] == [3, 7]
    assert reduce_involution([0, 0, 3, 7]) == [3, 7]                  # the measured case: v0*v0 cancels
    assert reduce_involution([5, 5, 5]) == [5]                        # odd count: one survives
    assert reduce_involution([]) == []

    print("OK: holographic_resonator self-test passed (recursive factoring: a depth-8 composite of two learned "
          "quads is recovered exactly at level %d after trying %s -- the flat vocabulary is measured at 0%% past "
          "depth 4 -- and the re-composition verify gate REFUSES an unexpressible composite instead of guessing)"
          % (got["level"], got["tried"]))


if __name__ == "__main__":
    _selftest()

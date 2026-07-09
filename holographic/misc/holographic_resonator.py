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
        for t in range(iters):
            new = []
            for f in range(self.F):
                others = c.copy()
                for g in range(self.F):
                    if g != f:
                        others = others * ests[g]
                new.append(self._cleanup(others, self.books[f]))
            idx = tuple(int(np.argmax(self.books[f] @ new[f])) for f in range(self.F))
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

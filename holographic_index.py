"""holographic_index.py -- the INDEX home (consolidation backlog H1): one nearest-neighbour interface over a set of
vectors, the right strategy chosen by size, with an optional calibrated abstain.

WHY THIS EXISTS
---------------
"Find the stored vector(s) closest to this query" is written many times across the engine: an EXACT cosine scan
(holographic_ai.nearest -- argmax of a matrix-vector product) when the set is small, and the sub-linear
RANDOM-PROJECTION FOREST (holographic_tree.HoloForest) when it is large. Callers that just want "the k nearest"
shouldn't have to pick, size their set, or reimplement the scan. `Index` is that one door:

    idx = Index(vectors, labels=names)      # build once
    idx.nearest(query, k=5)                 # -> [(label_or_index, score), ...], best first
    idx.nearest(query, abstain=0.05)        # -> [] when the best hit is no better than noise (calibrated)

It ROUTES (does not rewrite): the exact scan and the forest stay their own code; this picks between them by size,
adds top-k and a deterministic tie-break, and adds the calibrated false-alarm probability from the honesty layer
(holographic_honesty.RecallNull) so a caller can ABSTAIN instead of returning a confident guess at noise.

HONEST SCOPE (kept): this is the COSINE / vector nearest-neighbour family. It is deliberately NOT
holographic_spatial.knn (Euclidean k-NN over point clouds) nor holographic_rayindex (which pixels/objects a ray
touches) -- those are different metrics and purposes, and stay their own homes (registered in the catalog). Merging
them would be a leaky abstraction. For a forest-backed index, k>1 and abstain fall back to an exact rank over the
full set (documented) -- the sub-linear path is the top-1 recall.
"""
import numpy as np
from holographic_ai import nearest as _exact_nearest          # the exact top-1 cosine primitive (reused, not copied)


def _unit_rows(A):
    """Rows of A scaled to unit length (so a dot product IS the cosine). Zero rows are left as zeros."""
    A = np.asarray(A, float)
    if A.ndim == 1:
        A = A[None, :]
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.where(norms == 0, 1.0, norms)


class Index:
    """A nearest-neighbour index over `vectors` (n, dim). `labels` (optional) are returned in place of integer
    indices. `method` is 'auto' (exact for small sets, forest for large), or force 'exact' / 'forest'."""

    def __init__(self, vectors, labels=None, method="auto", seed=0, forest_threshold=4096, forest_trees=8):
        self.items = _unit_rows(vectors)                       # unit rows -> dot == cosine (matches ai.nearest)
        self.labels = list(labels) if labels is not None else None
        self.seed = int(seed)
        n = len(self.items)
        if method == "auto":
            method = "forest" if n > forest_threshold else "exact"
        self.method = method
        self._forest = None
        if method == "forest" and n:
            from holographic_tree import HoloForest
            self._forest = HoloForest(self.items.shape[1], n_trees=forest_trees, seed=seed).build(self.items)
        self._null = None                                      # lazily fit RecallNull for abstain

    def __len__(self):
        return len(self.items)

    def _key(self, j):
        return self.labels[j] if self.labels is not None else int(j)

    def _pvalue(self, score):
        """Calibrated false-alarm probability of a match `score` -- P(a random query scores this high). Lazily fits
        the noise floor once (holographic_honesty.RecallNull) over this index's own items."""
        if self._null is None:
            from holographic_honesty import RecallNull
            self._null = RecallNull().fit(self.items, seed=self.seed)
        return self._null.pvalue(float(score))

    def nearest(self, query, k=1, abstain=None):
        """The `k` nearest items to `query`, best first, as [(key, score), ...] (key = label or integer index).
        With `abstain=alpha`, return [] when the best hit's calibrated false-alarm probability exceeds alpha (the
        match is no better than noise). Deterministic: ties break by ascending index."""
        q = np.asarray(query, float)
        if not len(self.items):
            return []
        nq = np.linalg.norm(q) or 1.0

        # FAST PATH: forest, top-1, no abstain -> sub-linear recall (reuses HoloForest verbatim)
        if self.method == "forest" and k == 1 and abstain is None:
            j = int(self._forest.recall(q))
            return [(self._key(j), float(self.items[j] @ q / nq))]

        # EXACT PATH: full cosine scan. For k==1 this is literally ai.nearest; for k>1 an argsort with a stable,
        # index-ascending tie-break. Also the honest fallback for forest+abstain / forest+k>1.
        if k == 1:
            j, score = _exact_nearest(q, self.items)           # the shared exact primitive
            order = [int(j)]
            top = float(score)
        else:
            sims = self.items @ q / nq
            # sort by (-score, index): np.lexsort orders by the LAST key first, so pass (index, -score)
            order = list(np.lexsort((np.arange(len(sims)), -sims))[:k])
            top = float(sims[order[0]])
        if abstain is not None and self._pvalue(top) > abstain:
            return []                                          # abstain -- best match is noise-level
        sims_all = None
        hits = []
        for j in order:
            j = int(j)
            s = float(self.items[j] @ q / nq)
            hits.append((self._key(j), s))
        return hits


def index_backends():
    """The strategies Index routes between (for the catalog / discovery)."""
    return ("exact", "forest")


def _selftest():
    rng = np.random.default_rng(0)
    n, dim = 200, 128
    V = rng.standard_normal((n, dim))
    idx_exact = Index(V, method="exact", seed=0)
    idx_forest = Index(V, method="forest", seed=0, forest_threshold=0)   # force forest even though small

    # a noisy copy of item 42 should recall 42 by both strategies
    q = V[42] + 0.15 * rng.standard_normal(dim)
    assert idx_exact.nearest(q, k=1)[0][0] == 42
    assert idx_forest.nearest(q, k=1)[0][0] == 42               # forest agrees with exact on an easy query

    # top-k is descending by score, deterministic
    hits = idx_exact.nearest(q, k=5)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True) and len(hits) == 5

    # labels are returned in place of indices
    labels = [f"v{i}" for i in range(n)]
    assert Index(V, labels=labels, method="exact").nearest(q, k=1)[0][0] == "v42"

    # calibrated abstain: pure noise vs the codebook is rejected; a real item is accepted
    noise = rng.standard_normal(dim)
    assert Index(V, method="exact").nearest(noise, abstain=0.01) == []          # noise -> abstain
    assert Index(V, method="exact").nearest(V[7], abstain=0.01)[0][0] == 7      # a real item -> accepted

    # 'auto' picks forest past the threshold, exact below it
    assert Index(V, method="auto", forest_threshold=1000).method == "exact"
    assert Index(V, method="auto", forest_threshold=50).method == "forest"
    print("OK: holographic_index self-test passed (exact & forest agree on recall; top-k ordered; calibrated "
          "abstain rejects noise; routes over %s)" % ", ".join(index_backends()))


if __name__ == "__main__":
    _selftest()

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
from holographic.agents_and_reasoning.holographic_ai import nearest as _exact_nearest          # the exact top-1 cosine primitive  reused, not copied


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

    def __init__(self, vectors, labels=None, method="auto", seed=0, forest_threshold=30000, forest_trees=8,
                 recall_budget=None, screens_probe=0.35, screens_coherent=True, fast=False):
        # WHY 30000, not 4096: MEASURED (dim=128, random unit rows, seed 0) the exact scan is FASTER than
        # the forest until ~30-50k items (0.20 vs 0.65 ms/q at N=5000), and forest recall@1 vs exact
        # DEGRADES with N on unstructured data: 0.93 at 5k, 0.59 at 50k, 0.51 at 200k. The old default
        # silently traded half the correct answers for nothing below the crossover. KEPT NEGATIVE: the
        # forest is a LATENCY tool for large N, not a free lunch -- past the threshold it answers fast
        # and approximately, and callers who need exact answers at scale should use nearest_batch.
        self.items = _unit_rows(vectors)                       # unit rows -> dot == cosine (matches ai.nearest)
        self.labels = list(labels) if labels is not None else None
        self.seed = int(seed)
        n = len(self.items)
        self.recall_note = None
        if method == "auto":
            method = "forest" if n > forest_threshold else "exact"
            # F4/F12: with a recall_budget, 'auto' never silently ships a forest below it -- the
            # budget is MEASURED on this data (measure_forest_recall) after construction, and the
            # route falls back to exact with the measurement recorded in self.recall_note. The
            # measurement is deferred to first use via _honest_route (the forest is lazy).
        self.method = method
        # LAZY FOREST (upstreamed from stacc's PR #32 finding 3, measured here first): the forest is only
        # ever CONSULTED by nearest(k=1, abstain=None) -- every other call falls to the exact scan -- yet it
        # was built eagerly in __init__. A top-k workload at 100k x 128 paid 34.77s of construction for a
        # structure it never touched. Building on the FIRST qualifying call changes no answer (same
        # HoloForest, same seed, same build) and drops construction to ~0 for the workloads that skip it.
        # The RecallNull below already used this exact pattern; the forest now follows it.
        self._forest = None
        self._forest_trees = int(forest_trees)
        self._null = None                                      # lazily fit RecallNull for abstain
        self.recall_budget = None if recall_budget is None else float(recall_budget)
        self._screens = None
        self._screens_probe = float(screens_probe)
        self._screens_coherent = bool(screens_coherent)
        self._fast = bool(fast)

    def __len__(self):
        return len(self.items)

    def _key(self, j):
        return self.labels[j] if self.labels is not None else int(j)

    def _screens_nearest(self, q, k=1):
        """F30 -- NESTED DESCENT for retrieval (the screen-routing pattern, promoted into the
        index): score B block CENTROIDS (the measured winner over HRR block bundles -- kept
        negative from screen routing: centroid 0.797 vs bundle 0.789, so the simpler summary
        ships), descend into the top ceil(probe*B) blocks, EXACT scan only inside them. Cheap
        boundary first, volume price only where it points (Quilez's raymarching discipline; the
        nested-diamond structure). Tie rule preserved GLOBALLY: candidates carry global indices
        and the final top-k is a lexsort on (global_idx, -score) -- topk_det's contract."""
        C, blocks = self._screens
        cs = C @ q
        n_desc = max(1, int(np.ceil(self._screens_probe * len(blocks))))
        from holographic.misc.holographic_determinism import topk_det
        best_blocks = topk_det(cs, n_desc)
        # LEVER 1, applied after the benchmark caught screens LOSING wall-clock to exact BLAS
        # (21.1 vs 10.6 ms/q at 36k x 768): the fused matmul was already here, but (a)
        # items[cand] fancy-indexing COPIED ~0.35*N*D*8B (~77MB) per query, and (b) simmap built
        # a ~12k-entry Python dict per query. Bake-once-scan-views: _ensure_screens lays block
        # members CONTIGUOUS (below), so candidates are SLICES -- per-span matvecs on views, no
        # gather copy -- and the dict is replaced by positional takes. Tie semantics unchanged:
        # lexsort on (global_idx, -score), pinned by the coherence tests.
        spans = [self._screens_spans[int(b)] for b in best_blocks]
        n_cand = sum(e - s for s, e in spans)
        gids = np.empty(n_cand, dtype=np.int64)
        at = 0
        if getattr(self, "_fast", False):
            # the same two-stage arbiter as the exact fast path, inside the screens scan: f32
            # span matvecs (half the traffic), f64 rescore of an over-fetched shortlist, margin
            # check against the best excluded f32 score; below-margin -> f64 spans, counted.
            if getattr(self, "_screens_baked32", None) is None:
                self._screens_baked32 = self._screens_baked.astype(np.float32)
                self._eps32s = float(self._screens_baked.shape[1] * np.finfo(np.float32).eps
                                     * np.max(np.abs(self._screens_baked)))
                self.fast_fallbacks = getattr(self, "fast_fallbacks", 0)
            q32 = q.astype(np.float32)
            s32 = np.empty(n_cand, dtype=np.float32)
            for s, e in spans:
                s32[at:at + (e - s)] = self._screens_baked32[s:e] @ q32
                gids[at:at + (e - s)] = self._screens_gid[s:e]
                at += e - s
            C = min(n_cand, max(4 * k, 64))
            part = np.argpartition(-s32, C - 1)[:C]
            s64 = self.items[gids[part]] @ q
            pos_l = np.lexsort((gids[part], -s64))[:k]
            bound = self._eps32s * float(np.linalg.norm(q32)) + 1e-12
            excl = float(np.max(np.delete(s32, part))) if C < n_cand else -np.inf
            if float(s64[pos_l[-1]]) - bound > excl + bound:
                return [(self._key(int(gids[part][j])), float(s64[j])) for j in pos_l]
            self.fast_fallbacks += 1
            at = 0
        sims = np.empty(n_cand)
        for s, e in spans:
            sims[at:at + (e - s)] = self._screens_baked[s:e] @ q
            gids[at:at + (e - s)] = self._screens_gid[s:e]
            at += e - s
        pos = np.lexsort((gids, -sims))[:k]
        return [(self._key(int(gids[j])), float(sims[j])) for j in pos]

    def _ensure_screens(self, block_size=512):
        if getattr(self, "_screens", None) is None:
            n = len(self.items)
            if getattr(self, "_screens_coherent", False):
                # THE COHERENCE PASS (claimed follow-up, now built): sequential blocks exploit
                # insertion locality, which shuffled/streamed corpora lack (measured: 0.88 ordered
                # -> 0.67 shuffled on real text). Build coherence DETERMINISTICALLY instead of
                # assuming it: seed B centroids by strided sampling (no RNG needed -- stride is a
                # pure function of n and B), run TWO Lloyd rounds with tie-safe assignment (argmax
                # = lowest index on ties), then group items by nearest centroid. One O(n*B*D)
                # setup cost, paid once, priced here so the caller can decide (machine-model
                # setup-vs-marginal). Empty blocks are dropped -- centroids of nothing summarize
                # nothing.
                B = max(1, int(np.ceil(n / block_size)))
                C = self.items[np.linspace(0, n - 1, B).astype(int)].copy()
                for _ in range(2):
                    assign = np.argmax(self.items @ C.T, axis=1)   # np.argmax: lowest-index ties
                    for b in range(B):
                        sel = assign == b
                        if sel.any():
                            C[b] = self.items[sel].mean(axis=0)
                    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
                assign = np.argmax(self.items @ C.T, axis=1)
                blocks = [np.where(assign == b)[0] for b in range(B)]
                keep = [i for i, b in enumerate(blocks) if len(b)]
                blocks = [blocks[i] for i in keep]
                C = C[keep]
            else:
                blocks = [np.arange(s, min(s + block_size, n)) for s in range(0, n, block_size)]
                C = np.stack([self.items[b].mean(axis=0) for b in blocks])
                C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
            self._screens = (C, blocks)
            # bake-once (lever 1): block members laid CONTIGUOUS so query-time candidates are
            # views, not gather copies; gid maps baked rows back to global indices for the tie
            # rule and key lookup. One O(N*D) copy at build, zero copies per query.
            order = np.concatenate(blocks) if blocks else np.empty(0, dtype=np.int64)
            self._screens_baked = np.ascontiguousarray(self.items[order])
            self._screens_gid = order.astype(np.int64)
            ends = np.cumsum([len(b) for b in blocks])
            self._screens_spans = [(int(e - len(b)), int(e)) for b, e in zip(blocks, ends)]

    def measure_screens_recall(self, n_probe=200, noise=0.05, seed=1234):
        """The honesty label for the screens route -- same contract as measure_forest_recall:
        recall@1 vs the exact answer, measured on THIS index's own vectors, Wilson 95% CI."""
        rng = np.random.default_rng(seed)
        take = min(int(n_probe), len(self.items))
        pick = rng.choice(len(self.items), take, replace=False)
        Qp = _unit_rows(self.items[pick] + noise * rng.standard_normal((take, self.items.shape[1])))
        from holographic.sampling_and_signal.holographic_tiledreduce import tiled_topk
        _, exact_idx = tiled_topk(self.items, Qp.T, k=1)
        self._ensure_screens()
        lab = {self._key(int(exact_idx[0, i])): i for i in range(take)}
        hits = sum(int(self._screens_nearest(Qp[i], k=1)[0][0] == self._key(int(exact_idx[0, i])))
                   for i in range(take))
        p_hat = hits / take
        z = 1.96; den = 1 + z * z / take
        c_ = (p_hat + z * z / (2 * take)) / den
        h_ = z * np.sqrt(p_hat * (1 - p_hat) / take + z * z / (4 * take * take)) / den
        return {"recall": p_hat, "lo": max(0.0, c_ - h_), "hi": min(1.0, c_ + h_), "n": take,
                "touched": self._screens_probe}

    def measure_forest_recall(self, n_probe=200, noise=0.05, seed=1234):
        """F4/F12 -- THE FOREST'S HONESTY LABEL, measured on THIS index's OWN vectors (never a
        gaussian proxy: sweep-2 measured forest recall@1 at 0.93 on random data at 5k but 0.50 on
        REAL clustered text vectors at 15k -- cluster structure defeats random splits, so the
        random-data curve UNDERSTATES the problem for exactly the users who arrive with real data).
        Probes n_probe stored items (perturbed by `noise`), compares the forest's answer to the
        EXACT answer on the same queries (tiled, so the check is memory-bounded at any N), and
        returns {'recall', 'lo', 'hi' (Wilson 95% CI), 'n'}. Deterministic given seed."""
        rng = np.random.default_rng(seed)
        n = len(self.items)
        take = min(int(n_probe), n)
        pick = rng.choice(n, take, replace=False)
        Qp = self.items[pick] + noise * rng.standard_normal((take, self.items.shape[1]))
        Qp = _unit_rows(Qp)
        from holographic.sampling_and_signal.holographic_tiledreduce import tiled_topk
        _, exact_idx = tiled_topk(self.items, Qp.T, k=1)
        if self._forest is None:                               # same lazy pattern nearest() uses
            from holographic.misc.holographic_tree import HoloForest
            self._forest = HoloForest(self.items.shape[1], n_trees=self._forest_trees,
                                      seed=self.seed).build(self.items)
        hits = 0
        for i in range(take):
            hits += int(int(self._forest.recall(Qp[i])) == int(exact_idx[0, i]))
        p_hat = hits / take
        z = 1.96; den = 1 + z * z / take
        centre = (p_hat + z * z / (2 * take)) / den
        half = z * np.sqrt(p_hat * (1 - p_hat) / take + z * z / (4 * take * take)) / den
        return {"recall": p_hat, "lo": max(0.0, centre - half), "hi": min(1.0, centre + half), "n": take}

    def _pvalue(self, score):
        """Calibrated false-alarm probability of a match `score` -- P(a random query scores this high). Lazily fits
        the noise floor once (holographic_honesty.RecallNull) over this index's own items."""
        if self._null is None:
            from holographic.agents_and_reasoning.holographic_honesty import RecallNull
            self._null = RecallNull().fit(self.items, seed=self.seed)
        return self._null.pvalue(float(score))

    def nearest(self, query, k=1, abstain=None):
        """The `k` nearest items to `query`, best first, as [(key, score), ...] (key = label or integer index).
        With `abstain=alpha`, return [] when the best hit's calibrated false-alarm probability exceeds alpha (the
        match is no better than noise). Deterministic: ties break by ascending index."""
        q = np.asarray(query, float)
        if not np.all(np.isfinite(q)):
            # NaN GATE (edge sweep): a NaN query used to return [(0, nan)] -- a hallucinated
            # match with an unreadable score, worse than any exception. Non-finite queries are
            # an instrument error upstream; fail LOUD, never rank garbage.
            raise ValueError("query contains non-finite values -- refusing to rank garbage")
        if not len(self.items):
            return []
        nq = np.linalg.norm(q) or 1.0

        # FAST PATH: forest, top-1, no abstain -> sub-linear recall (reuses HoloForest verbatim)
        if self.method == "screens":
            if self.recall_budget is not None and self.recall_note is None:
                r = self.measure_screens_recall()
                if r["lo"] < self.recall_budget:
                    self.method = "exact"
                    self.recall_note = ("screens recall %0.2f [%0.2f, %0.2f] @%d%% touched < budget %0.2f "
                                        "-> exact route" % (r["recall"], r["lo"], r["hi"],
                                                            int(100 * r["touched"]), self.recall_budget))
                else:
                    self.recall_note = ("screens recall %0.2f [%0.2f, %0.2f] @%d%% touched meets budget %0.2f"
                                        % (r["recall"], r["lo"], r["hi"], int(100 * r["touched"]),
                                           self.recall_budget))
            if self.method == "screens" and abstain is None:
                self._ensure_screens()
                return self._screens_nearest(q, k=k)
        if self.method == "forest" and self.recall_budget is not None and self.recall_note is None:
            # F4/F12 gate, paid once at first forest use: measure recall ON THIS DATA; below budget,
            # the route DEMOTES to exact and the measurement travels with the index (recall_note).
            r = self.measure_forest_recall()
            if r["lo"] < self.recall_budget:
                self.method = "exact"
                self.recall_note = ("forest recall %0.2f [%0.2f, %0.2f] on this data < budget %0.2f "
                                    "-> exact route" % (r["recall"], r["lo"], r["hi"], self.recall_budget))
            else:
                self.recall_note = ("forest recall %0.2f [%0.2f, %0.2f] on this data meets budget %0.2f"
                                    % (r["recall"], r["lo"], r["hi"], self.recall_budget))
        if self.method == "forest" and k == 1 and abstain is None:
            if self._forest is None:                           # built lazily on first qualifying call (see __init__)
                from holographic.misc.holographic_tree import HoloForest
                self._forest = HoloForest(self.items.shape[1], n_trees=self._forest_trees,
                                          seed=self.seed).build(self.items)
            j = int(self._forest.recall(q))
            return [(self._key(j), float(self.items[j] @ q / nq))]

        # EXACT PATH: full cosine scan. For k==1 this is literally ai.nearest; for k>1 an argsort with a stable,
        # index-ascending tie-break. Also the honest fallback for forest+abstain / forest+k>1.
        if k == 1 and not getattr(self, "_fast", False):
            # (fast=True routes k==1 through the two-stage arbiter below -- the profiler caught
            # this delegation swallowing the fast path: the branch ran the full f64 primitive
            # while the f32 machinery sat unreached. Probe the code path, not the intention.)
            j, score = _exact_nearest(q, self.items)           # the shared exact primitive
            order = [int(j)]
            top = float(score)
        else:
            if getattr(self, "_fast", False):
                # TWO-STAGE f32 EXACT (lever: fast path + oracle + arbiter, the reference-beside-
                # fast-path convention made mechanical): the full f64 matvec is memory-bound, so
                # an f32 scan HALVES the traffic; but f32 can flip near-ties, so the shortlist is
                # over-fetched (C = max(4k, 64)), RESCORED IN f64, and an ARBITER checks the
                # separation margin: only if the k-th kept f64 score clears the best EXCLUDED
                # f32 score by the worst-case f32 dot error (D * eps32 * max|row|*|q| bound) does
                # the fast answer stand -- otherwise FULL f64 fallback, counted in
                # self.fast_fallbacks. Results are therefore IDENTICAL to the f64 path by
                # construction, not by luck; the selftest pins identity across seeds AND plants a
                # sub-epsilon tie that forces the fallback to fire.
                if getattr(self, "_items32", None) is None:
                    self._items32 = self.items.astype(np.float32)
                    self._eps32 = float(self.items.shape[1] * np.finfo(np.float32).eps
                                        * np.max(np.abs(self.items)))
                    self.fast_fallbacks = 0
                q32 = q.astype(np.float32)
                s32 = self._items32 @ q32
                C = min(len(self.items), max(4 * k, 64))
                part = np.argpartition(-s32, C - 1)[:C]
                s64 = (self.items[part] @ q) / nq
                from holographic.misc.holographic_determinism import topk_det
                loc = topk_det(s64, min(k, C))
                kept = part[loc]
                bound = self._eps32 * float(np.linalg.norm(q32)) / nq + 1e-12
                excluded_best = (float(np.max(np.delete(s32, part))) / nq
                                 if C < len(self.items) else -np.inf)
                if float(s64[loc[-1]]) - bound > excluded_best + bound:
                    out = [(self._key(int(i)), float(s)) for i, s in zip(kept, s64[loc])]
                    if abstain is None:
                        return out
                    sims = np.full(len(self.items), -np.inf)
                    sims[part] = s64                      # abstain path reads calibrated scores
                else:
                    self.fast_fallbacks += 1
                    sims = self.items @ q / nq
            else:
                sims = self.items @ q / nq
            # ARGPARTITION SHORTLIST (stacc measured 0.89 -> 0.07 ms at 20k; same move nearest_batch already
            # ships with its kept negative: the full lexsort DOMINATED the matmul). O(N) to shortlist k+1,
            # then sort only the shortlist by (-score, index) -- identical results and tie-breaks to the
            # full sort, asserted in the selftest against the old ordering kept verbatim as reference.
            # DELEGATED (F17): the tie-safe boundary rule (everything >= the k-th value, stable sort)
            # now lives ONCE in holographic_determinism.topk_det -- bit-identical to the inline
            # shortlist it replaces, pinned by the planted-tie traps below.
            from holographic.misc.holographic_determinism import topk_det
            order = list(topk_det(sims, k))
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

    def nearest_batch(self, queries, k=1):
        """Exact k-nearest for MANY queries in ONE matmul: [[(key, score), ...] per query], each list best-first.
        WHY this exists: FAISS-flat's entire win is batching -- one (N,D)x(D,Q) BLAS call instead of Q separate
        scans -- and leCore already proved the same move in cleanup_batch (2.6-5.9x measured). Recall is 1.0 by
        construction (exact), so this is the honest large-N path when answers must be right; the forest is the
        latency path when approximate is acceptable. Deterministic: ties break by ascending index."""
        Q = _unit_rows(np.atleast_2d(np.asarray(queries, float)))
        if not len(self.items):
            return [[] for _ in range(len(Q))]
        # F17 x F18 COMPOSITION: the (N, Q) matrix (160 MB at 200k x 100) is gone -- tiled_topk folds
        # per-tile blocks under topk_det's tie contract, bit-identical to the dense path incl. planted
        # cross-tile ties (pinned in tiledreduce's selftest AND the tie traps below). Peak memory =
        # tile x Q whatever N is. KEPT NEGATIVES carried forward from this function's history: a full
        # lexsort per column was SLOWER than the per-query loop at 200k (sort dominated), and the k+1
        # shortlist was WRONG under ties at the k-th value (caught by BM25's discrete scores, not by
        # this selftest whose reference shared the flaw -- two components agreeing is not correctness).
        from holographic.sampling_and_signal.holographic_tiledreduce import tiled_topk
        vals, idxs = tiled_topk(self.items, Q.T, k=k)
        return [[(self._key(int(idxs[r, c])), float(vals[r, c]))
                 for r in range(idxs.shape[0])] for c in range(idxs.shape[1])]


def index_backends():
    """The strategies Index routes between (for the catalog / discovery)."""
    return ("exact", "forest")


class CausalIndex:
    """D3 -- the APPEND-ONLY, BEFORE-t index: nearest-neighbour recall that structurally cannot see the future.

    The failure it exists to prevent: "what did similar past states lead to?" answered with an index built
    over the WHOLE history. Every query then finds neighbours from its own future -- and because a point's
    own future is the most similar thing to it that exists, the apparent skill of history-matching inflates
    enormously and silently. The campaign's analog-recall experiments only became honest after the index was
    rebuilt per-timestep from the past alone; this class makes that discipline a property of the data
    structure instead of a property of the analyst's loop.

    Contracts:
      * append(vector, t, label=None) requires non-decreasing t -- the book is written in time order, once.
      * nearest(query, t, k=1, lag=1) searches ONLY items with time <= t - lag. lag=1 (default) means
        "strictly before t"; lag=0 is REFUSED by name, exactly as realizable_fills refuses it: an item
        stamped t is simultaneous with the query, and simultaneous is not past.
      * there is no rebuild-with-everything path. To search the full history, use Index -- deliberately a
        different class, so 'causal' in a call site is never one refactor away from quietly meaning 'not'.

    Exact scan only (masked cosine over the eligible prefix). The forest fast path is NOT offered: a forest
    built over all items cannot be time-masked without re-deriving its guarantees, and a per-t forest rebuild
    is the loop this class replaces. Honest and O(n) beats fast and leaky. (Declared negative, not a TODO.)
    """

    def __init__(self):
        self._vecs = []                 # raw appended vectors (unit-scaled lazily per search block)
        self._times = []
        self._labels = []
        self._mat = None                # cached unit matrix, rebuilt only when stale

    def __len__(self):
        return len(self._vecs)

    def append(self, vector, t, label=None):
        """Add one item at time `t`. Times must be non-decreasing -- inserting the past after the fact is
        exactly the leak this structure exists to prevent, so it refuses by name."""
        t = float(t)
        if self._times and t < self._times[-1]:
            raise ValueError("append-only violated: t=%g arrives after t=%g was recorded -- backfilling the "
                             "past would let later queries see items that were not known at their time"
                             % (t, self._times[-1]))
        self._vecs.append(np.asarray(vector, float).ravel())
        self._times.append(t)
        self._labels.append(label)
        self._mat = None
        return len(self._vecs) - 1

    def nearest(self, query, t, k=1, lag=1):
        """The k nearest items among those with time <= t - lag, best first, as [(key, score, time), ...]
        (key = label or integer index). lag >= 1; lag=0 refused by name. Empty when nothing is old enough --
        an honest [] rather than a fallback to the full set."""
        if int(lag) < 1:
            raise ValueError("lag=0 would allow an item stamped at the query's own time to answer it -- "
                             "simultaneous is not past; use lag >= 1")
        if not self._vecs:
            return []
        times = np.asarray(self._times)
        cutoff = float(t) - int(lag)
        n_ok = int(np.searchsorted(times, cutoff, side="right"))
        if n_ok == 0:
            return []
        if self._mat is None or self._mat.shape[0] != len(self._vecs):
            self._mat = _unit_rows(np.vstack(self._vecs))
        q = np.asarray(query, float).ravel()
        nq = np.linalg.norm(q) or 1.0
        scores = self._mat[:n_ok] @ (q / nq)
        k = min(int(k), n_ok)
        order = np.argsort(-scores, kind="stable")[:k]         # stable: ties break by ascending index
        return [((self._labels[j] if self._labels[j] is not None else int(j)),
                 float(scores[j]), float(times[j])) for j in order]

    def audit_causality(self, query, t, n_probes=6, seed=0, scale=1.0):
        """VERIFY, don't assert (the Gate.audit_causality idea applied to recall): perturb items strictly
        AFTER the cutoff and confirm nearest(query, t) is bit-identical. Returns {causal, n_future_perturbed};
        causal=False would mean the mask is broken. Restores the index exactly afterwards."""
        base = self.nearest(query, t, k=min(3, max(len(self), 1)))
        times = np.asarray(self._times)
        future = [i for i in range(len(self._vecs)) if times[i] > float(t) - 1]
        rng = np.random.default_rng(seed)
        saved = [self._vecs[i].copy() for i in future]
        try:
            for i in future[:n_probes] if n_probes else future:
                self._vecs[i] = self._vecs[i] + rng.standard_normal(self._vecs[i].size) * scale
            self._mat = None
            causal = self.nearest(query, t, k=min(3, max(len(self), 1))) == base
        finally:
            for i, v in zip(future, saved):
                self._vecs[i] = v
            self._mat = None
        return {"causal": bool(causal), "n_future_perturbed": min(len(future), n_probes or len(future))}



def _selftest():
    rng = np.random.default_rng(0)
    n, dim = 200, 128
    V = rng.standard_normal((n, dim))
    idx_exact = Index(V, method="exact", seed=0)
    # F4/F12 -- the RECALL BUDGET gate, both directions (test-data rule: the hard case is
    # NEAR-DUPLICATE TWINS -- exact nearest is the twin, and random hyperplanes routinely split
    # twins into different leaves; measured live on real text vectors: recall 0.63 -> demoted):
    # (first pin draft used near-duplicate twins and the 8-tree forest ATE them -- recall 1.00,
    # a wrong guess about the hard case, kept: twins land in the same leaf. The RELIABLY hard
    # cheap case is a SINGLE tree at high dim -- measured 0.64 at n=3000, d=128 -- one tree has
    # one set of split planes and no vote to rescue a bad route.)
    rng_g = np.random.default_rng(4477)
    hard_v = rng_g.standard_normal((3000, 128)); hard_v /= np.linalg.norm(hard_v, axis=1, keepdims=True)
    hard = Index(hard_v, method="forest", forest_threshold=0, forest_trees=1, recall_budget=0.95, seed=0)
    hard.nearest(hard_v[3] + 0.05 * rng_g.standard_normal(128), k=1)
    assert hard.method == "exact" and "< budget" in hard.recall_note, hard.recall_note
    easy_v = rng_g.standard_normal((1500, 24)); easy_v /= np.linalg.norm(easy_v, axis=1, keepdims=True)
    easy = Index(easy_v, method="forest", forest_threshold=0, recall_budget=0.70, seed=0)
    easy.nearest(easy_v[3] + 0.02 * rng_g.standard_normal(24), k=1)
    assert easy.method == "forest" and "meets budget" in easy.recall_note, easy.recall_note

    # F30 -- SCREENS: nested descent with the honesty label. Blocks must be COHERENT for centroids
    # to summarize (measured on real text vectors: 0.88 @35% touched with corpus order, 0.67
    # SHUFFLED -- insertion locality is load-bearing, kept loud; real corpora HAVE it, and the
    # budget gate demotes automatically when the data does not). Pins: coherent clustered data
    # meets a bar; SHUFFLING THE SAME DATA must degrade (the negative as an inequality); the gate
    # demotes below budget.
    # FAST-PATH ARBITER PINS (the two-stage f32 engine): (a) fast==reference on indices with
    # scores within 1e-10 across random queries, exact AND screens (bit-equality on scores is
    # NOT the contract -- sliced vs full BLAS sums differ in the last ulp; the first identity
    # check compared tuples and taught this); (b) the BOUNDARY plant -- more duplicates than the
    # shortlist holds -- must FIRE the fallback and still match reference exactly (in-shortlist
    # ties need no fallback: f64 rescore resolves them, also asserted).
    rng_f = np.random.default_rng(41)
    Vf = rng_f.standard_normal((3000, 64)); Vf /= np.linalg.norm(Vf, axis=1, keepdims=True)
    i_ref = Index(Vf, method="exact", seed=0)
    i_fast = Index(Vf, method="exact", seed=0, fast=True)
    for qv in Vf[rng_f.choice(3000, 15, replace=False)] + 0.05 * rng_f.standard_normal((15, 64)):
        a, b = i_ref.nearest(qv, k=4), i_fast.nearest(qv, k=4)
        assert [i for i, _ in a] == [i for i, _ in b]
        assert all(abs(x - y) < 1e-10 for (_, x), (_, y) in zip(a, b))
    Wt = np.vstack([Vf[:500], np.tile(Vf[7], (100, 1))])
    Wt /= np.linalg.norm(Wt, axis=1, keepdims=True)
    i_tie = Index(Wt, method="exact", seed=0, fast=True)
    rt = i_tie.nearest(Wt[7], k=3)
    assert i_tie.fast_fallbacks >= 1, "boundary overflow must trip the arbiter"
    assert [i for i, _ in rt] == [i for i, _ in Index(Wt, method="exact", seed=0).nearest(Wt[7], k=3)]
    s_ref = Index(Vf, method="screens", screens_probe=0.3, seed=0)
    s_fast = Index(Vf, method="screens", screens_probe=0.3, seed=0, fast=True)
    for qv in Vf[:8]:
        assert [i for i, _ in s_ref.nearest(qv, k=3)] == [i for i, _ in s_fast.nearest(qv, k=3)]

    # COHERENCE PASS (built after the negative pinned it; default flipped on measurement:
    # shuffled real wiki 0.62 sequential -> 0.97 coherent, and ordered 0.90 -> 0.97 -- the
    # deterministic two-round assignment DOMINATES both cases, so it is the default; sequential
    # stays as screens_coherent=False with its order dependence as the KEPT NEGATIVE):
    rng_s = np.random.default_rng(5599)
    cents = rng_s.standard_normal((12, 48))
    coh = np.repeat(cents, 150, axis=0) + 0.25 * rng_s.standard_normal((1800, 48))
    coh /= np.linalg.norm(coh, axis=1, keepdims=True)
    shuf = coh[rng_s.permutation(len(coh))]
    r_ord = Index(coh, method="screens", screens_probe=0.3, seed=0).measure_screens_recall()
    r_shf = Index(shuf, method="screens", screens_probe=0.3, seed=0).measure_screens_recall()
    assert r_ord["recall"] >= 0.9 and r_shf["recall"] >= 0.9, (r_ord, r_shf)
    assert abs(r_ord["recall"] - r_shf["recall"]) < 0.08, "coherent default must be ORDER-INDEPENDENT"
    r_seq = Index(shuf, method="screens", screens_probe=0.3, screens_coherent=False,
                  seed=0).measure_screens_recall()
    assert r_seq["recall"] < r_shf["recall"], "sequential blocks on shuffled data must still degrade (the negative, kept)"
    i_gate = Index(shuf, method="screens", screens_probe=0.3, screens_coherent=False,
                   recall_budget=0.97, seed=0)
    i_gate.nearest(shuf[3] + 0.05 * rng_s.standard_normal(48), k=1)
    assert i_gate.method == "exact" and "< budget" in i_gate.recall_note, i_gate.recall_note

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

    # k>1 shortlist path == the ORIGINAL full lexsort, kept verbatim as reference (flat_recall pattern):
    # identical keys AND scores on a tie-rich query set, so the argpartition change can never drift ranks.
    for qv in (V[:6] + 0.2 * rng.standard_normal((6, dim))):
        sims_ref = idx_exact.items @ (qv / (np.linalg.norm(qv) or 1.0))
        ref_order = list(np.lexsort((np.arange(len(sims_ref)), -sims_ref))[:5])
        got = idx_exact.nearest(qv, k=5)
        assert [key for key, _ in got] == [idx_exact._key(int(j)) for j in ref_order], "shortlist ranks drifted"

    # LAZY FOREST: construction must not build it; the first k=1 call must; answers identical either way
    lazy = Index(V, method="forest", forest_threshold=0, seed=0)
    assert lazy._forest is None, "forest must not be built eagerly"
    a1 = lazy.nearest(V[3], k=1)
    assert lazy._forest is not None, "first qualifying call must build the forest"
    eager_like = Index(V, method="forest", forest_threshold=0, seed=0)
    assert eager_like.nearest(V[3], k=1) == a1, "lazy build changed an answer"

    # PLANTED TIES (regression trap for the k+1 shortlist bug): quantized vectors force many EXACT
    # score ties at the k-th rank; both k>1 paths must match the full stable sort, indices ascending.
    rng_t = np.random.default_rng(4004)
    Vt = np.round(rng_t.standard_normal((300, 8)) * 2) / 2      # coarse grid -> massive tie groups
    it = Index(Vt, method="exact")
    qt = np.round(rng_t.standard_normal(8) * 2) / 2
    sims_t = it.items @ (qt / (np.linalg.norm(qt) or 1.0))
    full_t = list(np.lexsort((np.arange(len(sims_t)), -sims_t)))
    for kk in (3, 10, 25):
        assert [key for key, _ in it.nearest(qt, k=kk)] == [int(j) for j in full_t[:kk]], f"nearest ties k={kk}"
        assert [key for key, _ in it.nearest_batch([qt], k=kk)[0]] == [int(j) for j in full_t[:kk]], f"batch ties k={kk}"

    # nearest_batch == per-query exact, bit-for-bit on keys, ONE matmul (the FAISS-flat move)
    Qs = V[:16] + 0.15 * rng.standard_normal((16, dim))
    batch = idx_exact.nearest_batch(Qs, k=3)
    for qi, row in zip(Qs, batch):
        assert [key for key, _ in row] == [key for key, _ in idx_exact.nearest(qi, k=3)]

    # REGRESSION TRAP (the 49%-wrong default): forest recall@1 vs exact on RANDOM data at moderate N.
    # This pins the honest number so a lossy auto-default can never ship silently again. The forest is
    # allowed to be approximate; it is NOT allowed to be the silent default below the measured crossover
    # (asserted above via forest_threshold=30000 semantics) -- and its recall here must stay in the
    # neighbourhood the docstring claims (~0.9 at this scale), not collapse.
    n2 = 5000
    V2 = rng.standard_normal((n2, dim))
    idx2e = Index(V2, method="exact", seed=0)
    idx2f = Index(V2, method="forest", seed=0, forest_threshold=0)
    Q2 = V2[:100] + 0.1 * rng.standard_normal((100, dim))
    agree = sum(idx2f.nearest(q)[0][0] == idx2e.nearest(q)[0][0] for q in Q2)
    assert agree >= 85, f"forest recall@1 regressed: {agree}/100 at N=5000 (was ~93)"
    assert Index(V2, method="auto").method == "exact", "auto must stay exact below the measured crossover"
    # ---------------- CausalIndex (D3): the before-t contract, and the leak it prevents, MEASURED ----------------
    # An AR(1) state series; "forecast" the next step by copying what followed the nearest neighbour state.
    # A full-history index lets each query find its own future (a point's own trajectory is its best match),
    # so apparent skill INFLATES; the causal index reports the honest, smaller number. The gap is the leak.
    T = 800
    s = np.zeros(T)
    for t in range(1, T):
        s[t] = 0.9 * s[t - 1] + rng.standard_normal()
    win = 8
    states = np.stack([s[t - win:t] for t in range(win, T - 1)])           # state at time t
    nxt = np.array([s[t + 1] - s[t] for t in range(win, T - 1)])           # what happened next
    tt = np.arange(win, T - 1).astype(float)

    full = Index(states, method="exact")
    ci = CausalIndex()
    for v, t in zip(states, tt):
        ci.append(v, t)

    errs_naive, errs_causal, n_eval = [], [], 0
    for i in range(200, len(states)):
        # THE NAIVE CALL, exactly as a user writes it: k=1 against the whole history. It finds the query
        # ITSELF (cos 1.0, error 0) and "history matching" reports perfect skill -- 100% inflation, zero
        # variance across seeds. The causal index cannot make this mistake at any k: its own time is
        # structurally excluded by lag >= 1.
        j = full.nearest(states[i], k=1)[0][0]
        errs_naive.append((nxt[j] - nxt[i]) ** 2)
        ch = ci.nearest(states[i], tt[i], k=1)
        if ch:
            errs_causal.append((nxt[ch[0][0]] - nxt[i]) ** 2)
            n_eval += 1
    mse_naive, mse_causal = float(np.mean(errs_naive)), float(np.mean(errs_causal))
    assert mse_naive == 0.0, mse_naive                        # perfect fake skill, deterministically
    assert mse_causal > 0.5 * float(np.var(nxt)), mse_causal  # the honest number is an honest size
    leak_pct = 100.0 * (mse_causal - mse_naive) / mse_causal  # = 100.0
    # KEPT NEGATIVE (10 seeds, measured before pinning -- the timing-assert lesson applied): with the
    # self-match EXCLUDED (k=2, skip self), the residual future-neighbour leak on stationary AR(1) is DEAD:
    # mean -0.7% +/- 2.5 across seeds; my first single-seed run read +4% and was one draw from that noise.
    # On stationary data the past covers the state space as well as the future does. The causal index's value
    # on such data is IMMUNITY TO THE NAIVE CALL, not an edge over a carefully de-leaked full index -- and on
    # nonstationary data no full-index de-leak recipe exists, which is why the structure earns its keep.

    # the causal mask itself: audited by perturbation, not asserted
    aud = ci.audit_causality(states[300], tt[300], n_probes=8, seed=0, scale=5.0)
    assert aud["causal"] is True and aud["n_future_perturbed"] > 0

    # append-only: backfilling the past refuses by name
    try:
        ci.append(states[0], tt[0])
        raise AssertionError("expected append-only refusal")
    except ValueError as e:
        assert "append-only violated" in str(e)

    # lag=0 refused by name; nothing-old-enough returns an honest []
    try:
        ci.nearest(states[5], tt[5], lag=0)
        raise AssertionError("expected lag=0 refusal")
    except ValueError as e:
        assert "simultaneous is not past" in str(e)
    ci2 = CausalIndex()
    ci2.append(states[0], 10.0)
    assert ci2.nearest(states[0], 5.0) == []

    print("OK: CausalIndex passed (naive full-history k=1 reports PERFECT skill -- MSE 0.0, %.0f%% inflation, "
          "zero variance, it finds itself -- while the causal index reports the honest %.2f over %d evals; "
          "self-excluded future leak measured DEAD on stationary AR(1), -0.7%%+/-2.5 over 10 seeds, KEPT; "
          "future-perturbation audit causal=True; backfill and lag=0 refuse by name)"
          % (leak_pct, mse_causal, n_eval))


    print("OK: holographic_index self-test passed (exact & forest agree on recall; top-k ordered; calibrated "
          "abstain rejects noise; routes over %s)" % ", ".join(index_backends()))


if __name__ == "__main__":
    _selftest()

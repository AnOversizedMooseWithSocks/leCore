"""
holographic_tree.py -- breaking a too-big memory into a deterministic tree.

The honest limitation we keep running into: a single holographic store is a
*bundle*, and a bundle has finite capacity.  Superpose too many associations into
one trace and recall collapses -- measured here, a 2048-d associative memory
recalls 100% of 64 pairs but only ~1% of 2048.  Slime mould has the same problem
at the cell scale: pure diffusion can't service a large body, so Physarum
"overcomes the size limitations of purely diffusive transport" by resolving its
broad foraging mass into a *hierarchical vein network* -- a few thick trunks,
many thin twigs, reinforced by flux.  (Dasgupta & Freund's random projection tree
is the same idea in computer-science clothes; SQL's B-tree indexes and hash
partitions are the same idea again -- never scan the whole table, descend an index.)

So instead of one flat trace we build a tree:

  * Each internal node owns one deterministic random hyperplane (seeded by the
    node's id, so the whole tree is reproducible from a single seed -- very
    demoscene) and splits its items at the median of their projection onto it.
    Balanced, log-depth, and adapts to the data the way an RP-tree does.
  * Each leaf holds a small HolographicMemory -- few enough pairs to stay well
    inside capacity, so recall there is reliable.
  * A query routes down the tree; with a beam it can back-track into the nearest
    sibling cells (the standard RP-tree trick) for robustness to noisy cues.
  * Every leaf counts the "flux" of queries that reach it -- the thick-vein /
    thin-vein signal, exposed for inspection.

Net effect: recall stays high as the dataset grows, and a query touches
O(leaf_size . log N) items instead of all N.  Pure numpy, built on the engine's
HolographicMemory (bind/unbind) and random_vector.
"""

import itertools
import heapq
import numpy as np

from holographic_ai import HolographicMemory, random_vector, cosine


class HoloTree:
    """A deterministic recursive partition of key vectors with a small
    holographic associative memory at every leaf."""

    def __init__(self, dim, leaf_size=64, seed=0, max_depth=40):
        self.dim = dim
        self.leaf_size = leaf_size
        self.seed = seed
        self.max_depth = max_depth
        self.root = None
        self.keys = None
        self.values = None
        self.last_comparisons = 0          # set by recall(), for honest cost stats

    # ---- construction ----
    def _hyperplane(self, node_id):
        """A reproducible random unit direction for this node (seed + node id)."""
        rng = np.random.default_rng([self.seed, node_id])
        h = rng.standard_normal(self.dim)
        return h / np.linalg.norm(h)

    def build(self, keys, values=None):
        """keys: (N, dim) unit vectors.  values: (N, dim) atoms to associate with
        each key, or None for content-addressing (the key is its own value)."""
        self.keys = np.asarray(keys, float)
        self.values = np.asarray(values, float) if values is not None else None
        self.root = self._build(np.arange(len(self.keys)), depth=0, node_id=1)
        return self

    def _build(self, idx, depth, node_id):
        if len(idx) <= self.leaf_size or depth >= self.max_depth:
            return self._leaf(idx)
        h = self._hyperplane(node_id)
        proj = self.keys[idx] @ h
        thr = float(np.median(proj))
        left, right = idx[proj <= thr], idx[proj > thr]
        if len(left) == 0 or len(right) == 0:           # everything on one side
            return self._leaf(idx)
        return {"leaf": False, "h": h, "thr": thr,
                "left": self._build(left, depth + 1, 2 * node_id),
                "right": self._build(right, depth + 1, 2 * node_id + 1)}

    def _leaf(self, idx):
        mem = HolographicMemory(self.dim)
        for i in idx:
            mem.learn(self.keys[i], self.values[i] if self.values is not None else self.keys[i])
        return {"leaf": True, "idx": np.asarray(idx), "mem": mem, "flux": 0}

    # ---- query ----
    def _route(self, key, beam):
        """Best-first descent collecting up to `beam` leaves.  When we pick a side
        of a hyperplane we remember the other side, prioritised by how close the
        key fell to the boundary -- so a beam revisits the most plausible
        alternatives first (RP-tree back-tracking)."""
        leaves, cnt = [], itertools.count()
        pq = [(0.0, next(cnt), self.root)]
        while pq and len(leaves) < beam:
            _, _, node = heapq.heappop(pq)
            while not node["leaf"]:
                d = float(key @ node["h"]) - node["thr"]
                near, far = (node["left"], node["right"]) if d <= 0 else (node["right"], node["left"])
                heapq.heappush(pq, (abs(d), next(cnt), far))
                node = near
            leaves.append(node)
        return leaves

    def recall(self, key, beam=1):
        """Return the global index of the best-matching stored item.  Routes to
        leaf(s), reads the holographic memory there, and cleans up against just
        those candidates."""
        key = np.asarray(key, float)
        comps, best, bid = 0, -2.0, -1
        for leaf in self._route(key, beam):
            leaf["flux"] += 1
            cand = leaf["idx"]
            if self.values is not None:                 # key->value associative recall
                est, ref = leaf["mem"].recall(key), self.values[cand]
            else:                                        # content addressing: cosine to the cue
                est, ref = key, self.keys[cand]
            sims = ref @ est                             # cleanup over this leaf only
            comps += len(cand)
            j = int(np.argmax(sims))
            if sims[j] > best:
                best, bid = float(sims[j]), int(cand[j])
        self.last_comparisons = comps
        return bid

    # ---- inspection ----
    def stats(self):
        depths, sizes = [], []

        def walk(n, d):
            if n["leaf"]:
                depths.append(d); sizes.append(len(n["idx"]))
            else:
                walk(n["left"], d + 1); walk(n["right"], d + 1)
        walk(self.root, 0)
        return dict(leaves=len(sizes), depth=max(depths), avg_leaf=float(np.mean(sizes)),
                    max_leaf=int(max(sizes)), min_leaf=int(min(sizes)))

    def flux(self):
        """Per-leaf query counts gathered so far -- the vein-thickness signal."""
        out = []

        def walk(n):
            if n["leaf"]:
                out.append(n["flux"])
            else:
                walk(n["left"]); walk(n["right"])
        walk(self.root)
        return out


class HoloForest:
    """A handful of HoloTrees grown from different seeds.  A single tree can miss
    the true neighbour when the cue falls just across one of its hyperplanes;
    independent trees draw their boundaries elsewhere, so the *union* of their
    candidate leaves almost always contains it.  This is what lets the structure
    reach exact-scan recall at a fraction of the comparisons -- the cap a single
    tree couldn't pass without simply visiting every leaf."""

    def __init__(self, dim, n_trees=4, leaf_size=64, seed=0):
        self.dim = dim
        self.trees = [HoloTree(dim, leaf_size=leaf_size, seed=seed + t) for t in range(n_trees)]
        self.items = None
        self.last_comparisons = 0

    def build(self, items):
        self.items = np.asarray(items, float)
        for t in self.trees:
            t.build(self.items)
        return self

    def recall(self, query, beam=4):
        """Union the candidate leaves from every tree, then cosine-rank once over
        that combined (de-duplicated) candidate set."""
        query = np.asarray(query, float)
        cand = set()
        for t in self.trees:
            for leaf in t._route(query, beam):
                leaf["flux"] += 1
                cand.update(int(i) for i in leaf["idx"])
        cand = np.fromiter(cand, int)
        self.last_comparisons = len(cand)
        sims = self.items[cand] @ query
        return int(cand[int(sims.argmax())])


# ======================================================================
# benchmarks
# ======================================================================
def capacity_curve(Ns, dim=2048, leaf_size=64, seed=0, probes=None):
    """Recall@1 of a flat memory vs a HoloTree as the dataset grows.  This is the
    headline: flat collapses past capacity, the tree holds.  `probes` optionally
    subsamples the recall estimate for speed."""
    rows = []
    for N in Ns:
        rng = np.random.default_rng(seed)
        keys = np.stack([random_vector(dim, rng) for _ in range(N)])
        vals = np.stack([random_vector(dim, rng) for _ in range(N)])
        probe_idx = range(N) if not probes else rng.choice(N, min(probes, N), replace=False)
        flat = HolographicMemory(dim)
        for k, v in zip(keys, vals):
            flat.learn(k, v)
        f_ok = sum(int((vals @ flat.recall(keys[i])).argmax() == i) for i in probe_idx)
        tree = HoloTree(dim, leaf_size=leaf_size, seed=seed).build(keys, vals)
        t_ok = sum(int(tree.recall(keys[i]) == i) for i in probe_idx)
        n = len(list(probe_idx)) if probes else N
        rows.append(dict(N=N, flat=f_ok / n, tree=t_ok / n, leaves=tree.stats()["leaves"],
                         depth=tree.stats()["depth"]))
    return rows


def nn_benchmark(N=2000, dim=512, leaf_size=64, beam=8, noise=0.5, seed=0):
    """Approximate nearest-neighbour search on noisy cues: exact O(N) scan vs the
    tree.  Reports recall@1 and how many comparisons each did.  `noise` adds a
    unit-vector perturbation of that magnitude (0.5 -> cue cosine ~0.9 to truth)."""
    rng = np.random.default_rng(seed)
    items = np.stack([random_vector(dim, rng) for _ in range(N)])
    tree = HoloTree(dim, leaf_size=leaf_size, seed=seed).build(items)
    q_ok = t_ok = t_cmp = 0
    T = 300
    for _ in range(T):
        i = int(rng.integers(N))
        q = items[i] + noise * random_vector(dim, rng)
        q = q / np.linalg.norm(q)
        q_ok += int((items @ q).argmax() == i)          # exact scan, N comparisons
        t_ok += int(tree.recall(q, beam=beam) == i); t_cmp += tree.last_comparisons
    return dict(N=N, dim=dim, beam=beam, exact_recall=q_ok / T, exact_cmp=N,
                tree_recall=t_ok / T, tree_cmp=round(t_cmp / T))


def forest_benchmark(N=2000, dim=512, leaf_size=64, n_trees=4, beam=4, noise=0.5, seed=0):
    """Approximate NN with a forest: recall and comparisons vs an exact scan."""
    rng = np.random.default_rng(seed)
    items = np.stack([random_vector(dim, rng) for _ in range(N)])
    forest = HoloForest(dim, n_trees=n_trees, leaf_size=leaf_size, seed=seed).build(items)
    ok = cmp = exact = 0; T = 300
    for _ in range(T):
        i = int(rng.integers(N)); q = items[i] + noise * random_vector(dim, rng); q = q / np.linalg.norm(q)
        exact += int((items @ q).argmax() == i)
        ok += int(forest.recall(q, beam=beam) == i); cmp += forest.last_comparisons
    return dict(N=N, n_trees=n_trees, beam=beam, exact_recall=exact / T, exact_cmp=N,
                forest_recall=ok / T, forest_cmp=round(cmp / T))


# ======================================================================
# demo
# ======================================================================

def _demo():
    print("holographic_tree -- a deterministic recursive memory\n" + "-" * 52)
    print("capacity: recall@1 as the dataset grows (dim=2048, leaf=64)")
    print(f"  {'N':>6} {'flat':>7} {'tree':>7}  leaves")
    for r in capacity_curve([64, 256, 1024, 4096], dim=2048, leaf_size=64):
        print(f"  {r['N']:>6} {r['flat']:>7.0%} {r['tree']:>7.0%}  {r['leaves']}")
    print("\napproximate nearest-neighbour on noisy cues (exact scan vs tree)")
    b = nn_benchmark(N=2000, dim=512, leaf_size=64, beam=8)
    print(f"  exact: recall {b['exact_recall']:.0%}  ({b['exact_cmp']} comparisons/query)")
    print(f"  tree : recall {b['tree_recall']:.0%}  ({b['tree_cmp']} comparisons/query, "
          f"{b['exact_cmp'] / b['tree_cmp']:.0f}x fewer)")
    f = forest_benchmark(N=2000, dim=512, leaf_size=64, n_trees=4, beam=4)
    print(f"  forest (4 trees): recall {f['forest_recall']:.0%}  ({f['forest_cmp']} comparisons/query, "
          f"{f['exact_cmp'] / f['forest_cmp']:.1f}x fewer) -- past the single tree's ceiling")
    t = HoloTree(512, leaf_size=64, seed=0).build(np.stack([random_vector(512, np.random.default_rng(i)) for i in range(2000)]))
    print(f"\ntree shape: {t.stats()}")


if __name__ == "__main__":
    _demo()

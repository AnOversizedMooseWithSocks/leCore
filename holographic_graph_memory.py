"""Hierarchical holographic memory: the maze lesson, applied to store/retrieve.

The flat prototype memory (SubPrototypeMemory) classifies by scanning EVERY prototype --
one big field, O(prototypes) comparisons. That is the flat maze field: fine while small,
but it is the same shape that hit the capacity wall, and the cost grows with everything
you have ever learned. The fix is the one the maze taught us: PARTITION into a routing
tree, keep each node's field bounded to its CHILDREN, and route a query DOWN by hop-and-
snap -- cosine to a handful of child centroids at each level, never to the whole memory.

  * store     = fold a vector into its nearest leaf sub-prototype (same as the flat memory)
  * organize  = cluster the leaves into a bounded-branching tree (recursively)
  * retrieve  = hop from the root, snapping to the nearest child centroid each level
  * the BRANCHING factor IS the per-field capacity budget, held fixed as the memory grows

This deliberately mirrors SubPrototypeMemory's leaf representation, so a head-to-head test
changes only ONE thing: flat scan vs routed descent. The honest expectation is that it
MATCHES the flat memory's accuracy (it cannot beat an exact nearest-neighbour) while doing
far fewer comparisons as the memory scales -- and that it beats a flat HOLOGRAPHIC store,
which blurs at scale exactly like the overloaded maze field.
"""

import numpy as np

from holographic_ai import cosine


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _cosine_kmeans(vectors, k, seed=0, iters=12):
    """Tiny cosine k-means used to split a node's items into <=k child clusters.
    Returns a list of index-lists (one per non-empty cluster)."""
    rng = np.random.default_rng(seed)
    n = len(vectors)
    if n <= k:
        return [[i] for i in range(n)]
    V = np.array([_unit(v) for v in vectors])
    centroids = V[rng.choice(n, k, replace=False)]
    assign = np.zeros(n, dtype=int)
    for _ in range(iters):
        sims = V @ centroids.T                      # cosine (unit rows)
        new = sims.argmax(axis=1)
        if np.array_equal(new, assign):
            break
        assign = new
        for c in range(k):
            members = V[assign == c]
            if len(members):
                centroids[c] = _unit(members.sum(axis=0))
    groups = {}
    for i, c in enumerate(assign):
        groups.setdefault(c, []).append(i)
    return list(groups.values())


class _Node:
    __slots__ = ("centroid", "children", "protos")

    def __init__(self):
        self.centroid = None      # unit summary vector (the node's field key)
        self.children = None      # list[_Node] for internal nodes
        self.protos = None        # list of leaf prototype indices for leaves


class GraphMemory:
    """A routing-tree classifier over per-label sub-prototypes. Same leaves as the flat
    memory; the difference is retrieval descends a bounded-branching tree instead of
    scanning everything."""

    def __init__(self, dim, branching=6, beam=1, seed=0):
        self.dim = dim
        self.b = branching            # per-node field size budget (children per node)
        self.beam = beam              # how many branches to keep while descending (1 = greedy)
        self.seed = seed
        self._p = []                  # leaves: [label, sum_vector, unit_vector, count]
        self._root = None
        self.last_comparisons = 0     # cosine comparisons used by the last classify (cost)

    # -- store (identical online rule to the flat prototype memory) ------------
    def observe_vector(self, v, label):
        best_i, best_s = -1, -2.0
        for i, p in enumerate(self._p):
            if p[0] == label:
                s = float(p[2] @ v)
                if s > best_s:
                    best_i, best_s = i, s
        if best_i < 0:
            self._p.append([label, v.copy(), _unit(v), 1])
        else:
            p = self._p[best_i]
            p[1] = p[1] + v
            p[2] = _unit(p[1])
            p[3] += 1
        self._root = None             # tree is stale until re-organized

    # -- organize: build the bounded-branching routing tree --------------------
    def organize(self):
        idxs = list(range(len(self._p)))
        self._root = self._build(idxs, depth=0)
        return self._root

    def _build(self, idxs, depth):
        node = _Node()
        if len(idxs) <= self.b:
            node.protos = idxs
            node.centroid = _unit(np.sum([self._p[i][2] for i in idxs], axis=0))
            return node
        vecs = [self._p[i][2] for i in idxs]
        groups = _cosine_kmeans(vecs, self.b, seed=self.seed + depth)
        if len(groups) <= 1:                       # degenerate split: stop here as a leaf
            node.protos = idxs
            node.centroid = _unit(np.sum(vecs, axis=0))
            return node
        node.children = [self._build([idxs[i] for i in g], depth + 1) for g in groups]
        node.centroid = _unit(np.sum([c.centroid for c in node.children], axis=0))
        return node

    # -- retrieve: hop-and-snap down the tree ----------------------------------
    def classify_vector(self, v, among=None):
        if self._root is None:
            self.organize()
        comps = 0
        # beam of frontier nodes; expand the most promising until all are leaves
        frontier = [self._root]
        while any(n.children for n in frontier):
            cand = []
            for n in frontier:
                if n.children:
                    for c in n.children:
                        comps += 1
                        cand.append((float(c.centroid @ v), c))
                else:
                    cand.append((float(n.centroid @ v), n))
            cand.sort(key=lambda t: t[0], reverse=True)
            frontier = [n for _, n in cand[:self.beam]]
        # score the leaf prototypes we landed on
        best_label, best_s = None, -2.0
        for n in frontier:
            for i in n.protos:
                label, _, unit, _ = self._p[i]
                if among is not None and label not in among:
                    continue
                comps += 1
                s = float(unit @ v)
                if s > best_s:
                    best_label, best_s = label, s
        self.last_comparisons = comps
        return best_label, best_s

    def size(self):
        return len(self._p)

    def counts_by_label(self):
        d = {}
        for p in self._p:
            d[p[0]] = d.get(p[0], 0) + 1
        return d

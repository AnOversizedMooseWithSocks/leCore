"""
holographic_pivot.py  --  a recursive pivot-tree index for sublinear nearest-item recall (Path D, Pharr's seat).

A naive content index that summarizes items UPWARD into a bundle hits the capacity wall -- the bundle blurs as
it grows, and recall collapses (the kept negative the crash taught: ~0.23). A B-tree never does that: internal
nodes hold PIVOTS (separators), stored explicitly, so the wall never bites. In VSA a node is then a small
CLEANUP memory of (pivot -> child), and routing a query is a nearest-pivot cleanup applied RECURSIVELY --
inception as the addressing fabric, the same primitive the mind's `cleanup` already is, one level per hop.

Greedy top-1 routing matches an exhaustive scan while touching only O(fanout * depth) ~ O(log N) pivots; a
wider beam trades a few more comparisons for near-perfect recall of the true leaf into the candidate set, after
which an exact key-unbind (or a final scan of the candidates) finishes. This is the sublinear retrieval the
forest seat wanted, built as a tree of cleanups rather than a flat scan.

Dependency note: the build uses a small NumPy k-means (Lloyd's with k-means++ seeding) rather than sklearn --
the engine's minimal-frameworks rule. KEPT NEGATIVE: each hop is an approximate nearest-pivot decision, so a
wrong turn at beam=1 can lose a query on overlapping data; the beam is the honest knob that buys recall back,
and the build cost is the recursive k-means.
"""
import numpy as np


def _kmeans(X, k, iters=12, seed=0):
    """Lloyd's k-means with k-means++ seeding, NumPy only. Returns (labels, centroids)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    k = min(k, n)
    c0 = int(rng.integers(n))
    centers = [X[c0]]
    d2 = ((X - centers[0]) ** 2).sum(1)
    for _ in range(1, k):                                       # k-means++ distance-weighted seeding
        total = d2.sum()
        pick = c0 if total <= 0 else int(rng.choice(n, p=d2 / total))
        centers.append(X[pick])
        d2 = np.minimum(d2, ((X - centers[-1]) ** 2).sum(1))
    C = np.stack(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        D2 = np.stack([((X - C[j]) ** 2).sum(1) for j in range(k)], axis=1)   # (n, k) distances
        labels = D2.argmin(1)
        newC = np.stack([X[labels == j].mean(0) if np.any(labels == j) else C[j] for j in range(k)])
        if np.allclose(newC, C):
            C = newC
            break
        C = newC
    return labels, C


def build_pivot_tree(items, fanout=7, seed=0, _idx=None):
    """Recursively k-means the items into `fanout` groups; each node holds its children, each child a pivot
    (the centroid of its subtree). Leaves hold the actual item index. Pivots are stored EXPLICITLY -- the
    B-tree move that avoids the summarize-upward capacity wall."""
    items = np.asarray(items, float)
    if _idx is None:
        _idx = np.arange(len(items))
    if len(_idx) <= fanout:
        return {"children": [{"leaf": int(i), "pivot": items[i]} for i in _idx]}
    labels, _ = _kmeans(items[_idx], fanout, seed=seed)
    kids = []
    for c in range(fanout):
        sub = _idx[labels == c]
        if len(sub) == 0:
            continue
        child = build_pivot_tree(items, fanout, seed, sub)
        child["pivot"] = items[sub].mean(0)
        kids.append(child)
    return {"children": kids}


def route(node, q, beam=1):
    """Route query `q` from a node: keep the `beam` nearest child pivots and recurse, collecting reached leaf
    indices and counting pivot comparisons. beam=1 is greedy top-1; a wider beam follows more branches."""
    if "leaf" in node:
        return [node["leaf"]], 0
    piv = np.stack([c["pivot"] for c in node["children"]])
    order = np.argsort(((piv - q) ** 2).sum(1))[:beam]
    hit, comps = [], len(piv)
    for i in order:
        sub_hit, sub_comps = route(node["children"][i], q, beam)
        hit += sub_hit
        comps += sub_comps
    return hit, comps


class PivotIndex:
    """A built pivot tree over a set of items, queried sublinearly. `.query(q, beam)` returns the nearest
    reached item and the pivot comparisons used; `.reached(q, beam)` returns the candidate leaf set."""

    def __init__(self, items, fanout=7, seed=0):
        self.items = np.asarray(items, float)
        self.fanout = fanout
        self.n = len(self.items)
        self.root = build_pivot_tree(self.items, fanout, seed)

    def query(self, q, beam=1):
        q = np.asarray(q, float)
        hit, comps = route(self.root, q, beam)
        ha = np.array(hit)
        best = int(ha[((self.items[ha] - q) ** 2).sum(1).argmin()])   # nearest among the reached candidates
        return best, comps

    def reached(self, q, beam=1):
        return route(self.root, q, beam=beam)[0]


def _selftest():
    rng = np.random.default_rng(3)
    D = 256

    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n else v

    def gen(depth, F, center, scale, decay, out):
        if depth == 0:
            out.append(center)
            return
        for _ in range(F):
            gen(depth - 1, F, center + unit(rng.standard_normal(D)) * scale, scale * decay, decay, out)

    leaves = []
    g = 3.0
    gen(3, 6, np.zeros(D), 0.6 * g ** 2, 1.0 / g, leaves)         # 6^3 = 216 well-separated leaves
    leaves = np.stack(leaves)
    K = len(leaves)
    qr = np.random.default_rng(77)
    NQ = 300
    tgt = qr.integers(0, K, size=NQ)
    Q = leaves[tgt] + 0.22 * qr.standard_normal((NQ, D))
    exhaustive = np.mean([int(((leaves - Q[i]) ** 2).sum(1).argmin()) == tgt[i] for i in range(NQ)])

    idx = PivotIndex(leaves, fanout=6, seed=0)
    top1 = np.mean([idx.query(Q[i], beam=1)[0] == tgt[i] for i in range(NQ)])
    rec5 = np.mean([tgt[i] in idx.reached(Q[i], beam=5) for i in range(NQ)])
    comps = np.mean([idx.query(Q[i], beam=1)[1] for i in range(NQ)])
    print(f"[pivot selftest] {K} leaves, D={D}: exhaustive top-1 = {exhaustive:.3f}, "
          f"tree greedy top-1 = {top1:.3f} ({comps:.0f} comparisons vs {K} -- {K / comps:.0f}x fewer)")
    print(f"[pivot selftest] beam-5 recall (true leaf in candidate set) = {rec5:.3f}")
    assert top1 >= exhaustive - 0.05, "greedy top-1 should match the exhaustive scan"
    assert comps < K / 2, "the tree must touch far fewer pivots than an exhaustive scan"
    assert rec5 >= 0.95, "a beam should land the true leaf in the candidate set nearly always"
    print("[pivot selftest] OK")


if __name__ == "__main__":
    _selftest()

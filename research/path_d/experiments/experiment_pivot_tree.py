"""
Path D finale: does recursive pivot-tree routing survive DEPTH? (clean isolation)

The lesson from the crash: the naive index summarized content UPWARD into a bundle (capacity wall,
recall 0.23). A B-tree never does that -- nodes hold PIVOTS (separators), stored explicitly, so the
wall never bites. In VSA a node is then a small CLEANUP memory of (pivot -> child): cleanup applied
recursively, inception as the addressing fabric. The one open risk: each hop is an approximate
nearest-pivot decision, and a wrong turn loses the query -- so does top-1 recall compound as r^d?

Clean test: ONE fixed dataset (a well-separated hierarchical cluster set, ~2400 shards), build pivot
trees of DIFFERENT depth over the SAME leaves by recursive k-means, route the SAME queries. Now the
exhaustive ceiling is FIXED, so any change with depth is the routing, not the data.
"""
import sys, os, time
import numpy as np
from sklearn.cluster import KMeans
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "repo"))
D = 1024
rng = np.random.default_rng(3)
def unit(v):
    n = np.linalg.norm(v); return v / n if n else v

# ---- ONE fixed hierarchical leaf set (well-separated; fixed leaf spacing) ---------------------
def gen_leaves(depth, F, center, scale, decay, out):
    if depth == 0:
        out.append(center); return
    for _ in range(F):
        gen_leaves(depth - 1, F, center + unit(rng.standard_normal(D)) * scale, scale * decay, decay, out)
g, s_fine = 3.0, 0.6
leaves = []
gen_leaves(4, 7, np.zeros(D), s_fine * g ** 3, 1.0 / g, leaves)   # 7^4 = 2401 leaves, fixed spacing
leaves = np.stack(leaves); K = len(leaves)
SIGMA = 0.22
print(f"fixed dataset: {K} leaf-shards in D={D}, leaf spacing ~{s_fine}, query noise sigma={SIGMA}")

# queries: a stored item = leaf + noise (fixed set, reused for every tree)
qrng = np.random.default_rng(77)
NQ = 800
tgt = qrng.integers(0, K, size=NQ)
Q = leaves[tgt] + SIGMA * qrng.standard_normal((NQ, D))
# fixed exhaustive ceiling (nearest of ALL leaves), computed once
exh = np.mean([int(((leaves - Q[i]) ** 2).sum(1).argmin()) == tgt[i] for i in range(NQ)])
print(f"exhaustive nearest-centroid (the fixed ceiling, scans all {K}): {exh:.3f}\n")

# ---- recursive k-means pivot tree over the FIXED leaves; depth set by fanout ------------------
def build(idx, F):
    node = {"idx": idx}
    if len(idx) <= F:
        node["children"] = [{"leaf": int(i), "pivot": leaves[i]} for i in idx]
        return node
    lab = KMeans(F, n_init=3, random_state=0).fit(leaves[idx]).labels_
    kids = []
    for c in range(F):
        sub = idx[lab == c]
        if len(sub) == 0: continue
        ch = build(sub, F); ch["pivot"] = leaves[sub].mean(0); kids.append(ch)
    node["children"] = kids
    return node

def route(node, q, beam):
    if "leaf" in node:
        return [node["leaf"]], 0
    piv = np.stack([c["pivot"] for c in node["children"]])
    order = np.argsort(((piv - q) ** 2).sum(1))[:beam]
    leaves_hit, comps = [], len(piv)
    for i in order:
        l, c = route(node["children"][i], q, beam); leaves_hit += l; comps += c
    return leaves_hit, comps

print("depth |  fanout | tree top-1 (=ceiling?) |   recall@beam5  | comparisons/query (b1 / b5)")
print("-" * 86)
results = []
for F, dlabel in [(K, 1), (49, 2), (13, 3), (7, 4)]:
    root = build(np.arange(K), F)
    row = {"depth": dlabel, "F": F, "exhaustive": exh}
    for b in (1, 3, 5):
        top1, rec, comp = 0, 0, 0
        for i in range(NQ):
            hit, c = route(root, Q[i], b)
            ha = np.array(hit)
            best = int(ha[((leaves[ha] - Q[i]) ** 2).sum(1).argmin()])  # nearest among reached
            top1 += (best == tgt[i]); rec += (tgt[i] in hit); comp += c
        row[f"top1_b{b}"] = top1 / NQ; row[f"rec_b{b}"] = rec / NQ; row[f"comp_b{b}"] = comp / NQ
    results.append(row)
    print(f"  {dlabel}   | {F:6d}  |   b1={row['top1_b1']:.3f}  b5={row['top1_b5']:.3f}    |"
          f"   {row['rec_b5']:.3f}      |   {row['comp_b1']:.0f}  /  {row['comp_b5']:.0f}", flush=True)
print("-" * 86)
deep = results[-1]
print(f"depth-4 tree: greedy top-1 = {deep['top1_b1']:.3f} vs exhaustive {exh:.3f}  "
      f"({deep['comp_b1']:.0f} comparisons vs {K} -- {K/deep['comp_b1']:.0f}x fewer)")
print(f"depth-4 beam-5: true shard in candidate set {deep['rec_b5']:.3f} of the time "
      f"(then the array's exact key-unbind finishes) -- vs the naive summary index's 0.23")
import json
json.dump({"results": results, "exhaustive": float(exh), "K": int(K)}, open("_tree_cache.json", "w"))
print("cached.")

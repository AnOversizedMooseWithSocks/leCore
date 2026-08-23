"""Re-test the design rule on the HARD corpus. The rule was derived on orthogonal atoms.

WHAT CHANGES AND WHY IT SHOULD. On synthetic atoms every document is near-orthogonal, so a cell
holding k of them superposes k INDEPENDENT vectors -- that is the regime the ~D/9 capacity figure
comes from. Real passages share vocabulary heavily (top-20 terms are 7.3% of all occurrences, and
13.3% of passages have a peer above 0.5 Jaccard), so a cell's contents are CORRELATED. Correlated
items superpose more compactly but are also harder to tell apart afterwards. Which effect wins is
an empirical question, so it gets measured rather than argued.

QUERIES ARE HELD-OUT SUBSETS, small on purpose. A whole-document query is a lookup: the target is
the only vector containing every term. The interesting and realistic regime is a handful of terms.

AND THE SCORE IS RECALL@k AGAINST THE FLAT SCAN, NOT AGAINST TRUTH. The question a tree has to
answer is "did the index lose anything the exhaustive scan would have found", which is separable
from "is bag-of-atoms a good retriever at all". Conflating the two would let a weak retriever
flatter the index, or vice versa.
"""
import numpy as np

import hard_corpus as HC
import holographic.agents_and_reasoning.holographic_hashatom as HA


def encode_all(docs, D):
    return np.stack([HA.encode_hash(t, D) for _, t in docs])


def make_queries(docs, D, n, terms, seed=0):
    rng = np.random.default_rng(seed)
    qs, gold = [], []
    for i in rng.choice(len(docs), n, replace=False):
        u = sorted(set(docs[i][1]))
        k = min(terms, len(u))
        pick = [u[j] for j in rng.choice(len(u), k, replace=False)]
        qs.append(HA.encode_hash(pick, D, normalise=False))
        gold.append(int(i))
    return np.stack(qs), gold


def tree_walk(V, top, g, q, beam):
    cand = np.argsort(top @ q)[::-1][:beam]
    rows = np.concatenate([np.arange(c * g, min((c + 1) * g, len(V))) for c in cand])
    order = rows[np.argsort(V[rows] @ q)[::-1]]
    return order


def evaluate(V, g, Q, beam, topk=5):
    top = np.stack([V[i:i + g].sum(0) for i in range(0, len(V), g)])
    agree1 = agreek = 0
    for q in Q:
        flat = np.argsort(V @ q)[::-1]
        tree = tree_walk(V, top, g, q, beam)
        agree1 += int(tree[0] == flat[0])
        agreek += len(set(tree[:topk]) & set(flat[:topk])) / topk
    dots = len(top) + beam * g
    return agree1 / len(Q), agreek / len(Q), dots


if __name__ == "__main__":
    docs = HC.load_passages(target=3000)
    K = len(docs)
    print("HARD CORPUS: %d real overlapping source passages\n" % K)

    for terms in (8,):
        print("== held-out queries of %d terms ==" % terms)
        print("   D     g    cap~D/9  rule D>=9*sqrt(K)=%d   flat-top1 recall   tree==flat top1   top5 overlap   dots"
              % int(9 * np.sqrt(K)))
        for D in (256, 1024, 2048):
            V = encode_all(docs, D)
            Q, gold = make_queries(docs, D, 150, terms)
            flat_acc = np.mean([int(np.argmax(V @ q)) == gg for q, gg in zip(Q, gold)])
            for g in (int(round(np.sqrt(K))), max(2, int(D / 9))):
                a1, ak, dots = evaluate(V, g, Q, beam=16)
                print("   %-5d %-4d %-8.0f %-21s %-18.3f %-17.3f %-14.3f %d"
                      % (D, g, D / 9, "yes" if D >= 9 * np.sqrt(K) else "NO",
                         flat_acc, a1, ak, dots))
        print()

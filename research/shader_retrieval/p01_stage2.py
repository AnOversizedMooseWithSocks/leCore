"""P0.1 continued -- and my own conclusion needed correcting once I checked the literature.

WHAT THE PUBLISHED RECORD SAYS (searched, not recalled):
  * BM25 remains a strong zero-shot baseline, and on corpora with precise domain terminology it
    BEATS strong 2026 commercial dense embeddings outright. Source code is exactly that kind of
    corpus, so BM25 winning here is the expected result, not an anomaly.
  * BM25+RM3 pseudo-relevance feedback UNDER-PERFORMS the lexical baseline in a 2026 matched
    evaluation -- so naive query expansion is not the move, which matches this project's own
    refutation of naive fusion.
  * Hybrid reciprocal-rank fusion "increases coverage but offers limited gains in ranking
    accuracy" -- Recall@100 near 0.997 while nDCG barely moves.
  * The best pipelines are TWO-STAGE: broad candidate recall, then rerank. The reranker that wins
    is a cross-encoder, at >50x the runtime -- and a learned cross-encoder is out of bounds here.

THE CORRECTION TO MY OWN CONCLUSION. I measured fusion on TOP-1 ONLY and concluded "delete RRF".
The literature says fusion's gain is in COVERAGE, which top-1 cannot see. An instrument that
reports one number about a two-number phenomenon will conclude the wrong thing, so recall@k is
added here and the "delete RRF" recommendation is re-examined against it.

THE RERANKER THAT IS IN BOUNDS: TERM PROXIMITY. BM25 is a bag of words -- it cannot tell two
passages apart when they share vocabulary but arrange it differently, which is exactly the R2
near-duplicate failure. Proximity/span scoring (Tao & Zhai-style: reward query terms appearing
close together, and reward preserved order) is unsupervised, has no learned weights, and attacks
precisely that blind spot. It is the constitutional stand-in for the cross-encoder slot.
"""
import numpy as np
from collections import defaultdict

import hard_corpus as HC
import p01_retriever as P
import holographic.agents_and_reasoning.holographic_hashatom as HA


def min_window_span(positions):
    """Smallest window containing one occurrence of each present query term.

    Classic proximity feature: a passage where the query terms cluster is a better match than one
    where they are scattered, even at identical bag-of-words score. Returns (covered, span).
    """
    lists = [p for p in positions if p]
    if not lists:
        return 0, 10 ** 9
    idx = [0] * len(lists)
    best = 10 ** 9
    while True:
        cur = [lists[i][idx[i]] for i in range(len(lists))]
        lo, hi = min(cur), max(cur)
        best = min(best, hi - lo + 1)
        k = int(np.argmin(cur))
        idx[k] += 1
        if idx[k] >= len(lists[k]):
            break
    return len(lists), best


def proximity_score(doc_tokens, qterms):
    """Coverage first, then tightness, then ordered adjacency. No weights are learned; the
    ordering is lexicographic by design so a tie in coverage is broken by span, not by a
    hand-tuned mixture that would need held-out selection to justify."""
    pos = defaultdict(list)
    for i, t in enumerate(doc_tokens):
        pos[t].append(i)
    positions = [pos[t] for t in qterms]
    covered, span = min_window_span(positions)
    bigram = 0
    qs = set(qterms)
    for a, b in zip(doc_tokens, doc_tokens[1:]):
        if a in qs and b in qs:
            bigram += 1
    return covered, -span, bigram


def recall_at(order, gold, k):
    return int(gold in list(order)[:k])


if __name__ == "__main__":
    import lecore
    mind = lecore.UnifiedMind(dim=256, seed=0)
    dn = HC.load_passages(target=800)
    docs = [t for _, t in dn]
    sets = [set(t) for t in docs]
    txt = [" ".join(t) for t in docs]
    K, D = len(docs), 1024
    V = np.stack([HA.encode_hash(t, D) for t in docs])

    vocab = sorted({t for s in sets for t in s})
    dic = {}
    for w in vocab:
        try:
            e = mind.lookup(w)
        except Exception:
            e = None
        if e and e.get("definition"):
            dic[w] = e["definition"]

    print("K=%d passages, 40 queries x 3 seeds, engine bm25_rank + hash dense + RRF + proximity\n" % K)
    print("  regime  arm            top-1    recall@10   recall@100")
    pooled = {}
    for regime in ("R1", "R2", "R3"):
        acc = defaultdict(lambda: defaultdict(list))
        for seed in range(3):
            for gold, q in P.build_queries(dn, sets, dic, regime, 40, seed):
                lex = [int(d) for d, _ in mind.bm25_rank(" ".join(q), txt)]
                qv = HA.encode_hash(q, D, normalise=False)
                dense = [int(x) for x in np.argsort(V @ qv)[::-1]]
                fused = P.rrf([lex[:200], dense[:200]])
                cand = fused[:100]
                rer = sorted(cand, key=lambda d: proximity_score(docs[d], q), reverse=True)
                rer = rer + [d for d in fused if d not in set(cand)]
                for name, o in (("bm25", lex), ("dense", dense), ("rrf", fused),
                                ("rrf+prox", rer)):
                    acc[name]["t1"].append(int(list(o)[0] == gold))
                    acc[name]["r10"].append(recall_at(o, gold, 10))
                    acc[name]["r100"].append(recall_at(o, gold, 100))
        for name in ("bm25", "dense", "rrf", "rrf+prox"):
            a = acc[name]
            print("  %-7s %-14s %-8.3f %-11.3f %.3f"
                  % (regime, name, np.mean(a["t1"]), np.mean(a["r10"]), np.mean(a["r100"])))
        pooled[regime] = acc
        print()

    print("PAIRED PERMUTATION (10k) + BH-FDR at q=0.05")
    labels, ps, diffs = [], [], []
    for regime in ("R1", "R2"):
        for metric in ("t1", "r100"):
            for a, b in (("rrf", "bm25"), ("rrf+prox", "bm25"), ("rrf+prox", "rrf")):
                d, p = P.paired_perm(pooled[regime][a][metric], pooled[regime][b][metric])
                labels.append("%s %-4s %s vs %s" % (regime, metric, a, b))
                ps.append(p); diffs.append(d)
    passed = P.bh(ps)
    for lab, d, p, ok in zip(labels, diffs, ps, passed):
        print("   %-32s diff %+0.4f  p=%.4f  %s" % (lab, d, p, "SIG" if ok else "ns"))

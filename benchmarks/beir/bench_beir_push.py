"""Phase 3: SciFact push -- add a char-n-gram hypervector arm (morphology channel) to the tuned fusion.

WHY: SciFact claims share terminology with their abstracts, so word-level BM25 is near-ceiling and the
context arm adds noise. What word-level BM25 structurally lacks is SUB-WORD signal: 'phosphorylation' /
'phosphorylated' / 'phosphorylates' are distinct BM25 terms (the derivational stemmer bridges some, not
all). A char-3/4-gram hypervector bundle scores morphological kinship continuously -- the classic VSA
n-gram encoding (Kanerva). Complementary by construction, so fusion has something new to use.
Protocol unchanged: sweep weights on TRAIN, touch TEST once.
"""
import sys, time
import numpy as np

sys.path.insert(0, "/home/claude/bench")
sys.path.insert(0, "/home/claude/repo")
from bench_beir import (load_scifact, load_nfcorpus, build_matrices, encode_query, eval_run,
                        bootstrap_delta, atom, DIM)
from bench_beir_tuned import load_scifact_train, load_nfcorpus_train, arm_orders
from holographic.semantic_router.holographic_bm25 import BM25, reciprocal_rank_fusion, tokenize


def chargram_token_matrix(vocab):
    """(V, DIM) matrix: each token = unit-normalized bundle of its char 3- and 4-gram atoms ('^word$'
    padded so prefixes/suffixes are distinct grams). Deterministic via the same sha256 atom rule."""
    M = np.zeros((len(vocab), DIM), dtype=np.float32)
    for i, t in enumerate(vocab):
        s = "^" + t + "$"
        grams = [s[j:j + n] for n in (3, 4) for j in range(len(s) - n + 1)]
        v = np.zeros(DIM, dtype=np.float32)
        for g in grams:
            v += atom("cg:" + g)
        nrm = np.linalg.norm(v)
        M[i] = v / nrm if nrm > 0 else v
    return M


def run(name, loaders):
    (ids, docs, test_q, test_qrels), (train_q, train_qrels) = loaders
    print("\n===", name)
    t0 = time.time()
    bm = BM25(docs)
    idf, vindex, A, C, D_bow, D_ctx = build_matrices(docs)
    vocab = sorted(vindex, key=vindex.get)
    G = chargram_token_matrix(vocab)
    # doc chargram vectors: reuse the SAME idf/log1p weighting -- built CHUNKED to avoid a second dense
    # (N, V) allocation (the phase-3 first run OOMed exactly there; V=31k x N=5.2k float32 x2 + G was the
    # kill. Sparse per-doc accumulation is O(total tokens), the doc-major lesson again).
    D_cg = np.zeros((len(docs), DIM), dtype=np.float32)
    for i, d in enumerate(docs):
        c = {}
        for t in tokenize(d):
            c[t] = c.get(t, 0) + 1
        v = np.zeros(DIM, dtype=np.float32)
        for t, f in c.items():
            v += (np.log1p(f) * idf[t]) * G[vindex[t]]
        D_cg[i] = v
    D_cg /= (np.linalg.norm(D_cg, axis=1, keepdims=True) + 1e-12)
    tr_o, tr_s = arm_orders(train_q, bm, idf, vindex, A, C, D_bow, D_ctx, ids)
    te_o, te_s = arm_orders(test_q, bm, idf, vindex, A, C, D_bow, D_ctx, ids)
    for tag, qs, o, s in (("tr", train_q, tr_o, tr_s), ("te", test_q, te_o, te_s)):
        o["holo-cg"], s["holo-cg"] = [], []
        n = len(ids)
        for qid, qtext in qs:
            qv = encode_query(qtext, idf, vindex, G)
            sc = D_cg @ qv
            order = np.lexsort((np.arange(n), -sc))[:200]
            o["holo-cg"].append([ids[i] for i in order]); s["holo-cg"].append(np.asarray(sc, np.float64))
    print("  build+score %.1fs" % (time.time() - t0))

    arms = ("bm25", "holo-bow", "holo-ctx", "holo-cg")
    grid = [0.0, 0.2, 0.5, 1.0]
    n = len(ids)

    def combsum(scores, qn, w):
        out = []
        for qi in range(qn):
            tot = np.zeros(n)
            for arm, wa in zip(arms, w):
                sc = scores[arm][qi]
                r = sc.max() - sc.min()
                tot += wa * ((sc - sc.min()) / r if r > 0 else np.zeros(n))
            order = np.lexsort((np.arange(n), -tot))[:200]
            out.append([ids[i] for i in order])
        return out

    def rrf(orders, qn, w):
        out = []
        for qi in range(qn):
            fused = reciprocal_rank_fusion([orders[a][qi] for a in arms], k=60, weights=list(w))
            out.append([d for d, _ in fused[:200]])
        return out

    best = None
    for method, fn, src in (("combsum", combsum, tr_s), ("rrf", rrf, tr_o)):
        for wb in grid:
            for wc in grid:
                for wg in grid:
                    fused = fn(src, len(train_q), (1.0, wb, wc, wg))
                    nd = eval_run(fused, train_q, train_qrels)[0].mean()
                    if best is None or nd > best[0]:
                        best = (nd, method, (1.0, wb, wc, wg))
    nd_tr, method, w = best
    print("  TRAIN champion: %s weights(bm25,bow,ctx,cg)=%s  train nDCG@10 %.4f" % (method, w, nd_tr))

    fused_te = (combsum(te_s, len(test_q), w) if method == "combsum" else rrf(te_o, len(test_q), w))
    r_bm = eval_run(te_o["bm25"], test_q, test_qrels)
    r_hy = eval_run(fused_te, test_q, test_qrels)
    print("  %-14s nDCG@10 %.4f  R@100 %.4f  MRR@10 %.4f" % ("bm25", *[x.mean() for x in r_bm]))
    print("  %-14s nDCG@10 %.4f  R@100 %.4f  MRR@10 %.4f" % ("tuned-hybrid4", *[x.mean() for x in r_hy]))
    for mi, mname in ((0, "nDCG@10"), (1, "R@100")):
        mu, lo, hi = bootstrap_delta(r_hy[mi], r_bm[mi])
        tag = "SIGNIFICANT WIN" if lo > 0 else ("significant LOSS" if hi < 0 else "not significant")
        print("  hybrid4 - bm25 %s: %+.4f [%+.4f, %+.4f] %s" % (mname, mu, lo, hi, tag))


if __name__ == "__main__":
    root = "/home/claude/bench"
    run("SciFact", (load_scifact(root), load_scifact_train(root)))
    run("NFCorpus", (load_nfcorpus(root), load_nfcorpus_train(root)))

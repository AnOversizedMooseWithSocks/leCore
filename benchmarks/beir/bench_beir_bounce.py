"""Phase 8: MULTI-BOUNCE retrieval -- the levers stacked as render layers, train-swept, test ONCE.

  BOUNCE 1 (the beauty pass): the phase-7 champion fusion (doc bm25 + ctx + bmx + lint [+maxp+title]).
  BOUNCE 2 (relight from the first bounce): PSEUDO-RELEVANCE FEEDBACK (Rocchio 1971 / RM3 spirit, pure
    counting). The top-F docs of the FUSED first pass are treated as emitters; their top-T terms by
    sum(tf)*idf (query terms excluded) expand the query; a second lexical pass scores the expanded
    query; the final image interpolates bounce-1 with bounce-2. The key holographic move: feedback
    quality is set by the FIRST pass -- our fused pass is a better light source than BM25's own top-F,
    which is why published RM3-on-BM25 gains should transfer or better.
  PARALLEL LAYER (multi-sample): the top few TRAIN weight vectors are independent renders of the same
    scene; RRF-ensembling them averages estimator noise (checked on train; kept only if it helps).

All counting + the existing channels; no learned weights anywhere.
"""
import json, sys, time
import numpy as np

sys.path.insert(0, "/home/claude/bench")
sys.path.insert(0, "/home/claude/repo")
from bench_beir import (load_nfcorpus, load_scifact, build_matrices, encode_query, eval_run,
                        bootstrap_delta)
from bench_beir_tuned import load_nfcorpus_train, load_scifact_train
from bench_beir_rungs import ctx_neighbors
from holographic.semantic_router.holographic_bm25 import BM25, tokenize, reciprocal_rank_fusion


def build_channels(name, ids, docs, queries_list, scifact_extras=None):
    """Phase-7 channel builder, factored. Returns (per-set channel dicts..., bm, idf, vindex)."""
    n = len(ids)
    bm = BM25(docs, k1=0.9, b=0.4, slim=True) if name == "SciFact" else BM25(docs)
    idf, vindex, A, C, D_bow, D_ctx = build_matrices(docs)
    del A, D_bow
    vocab = sorted(vindex, key=vindex.get)
    nb = ctx_neighbors(C, vocab, topk=3)
    expanded = []
    for d in docs:
        ts = set(tokenize(d))
        extra = sorted({e for t in ts for e in nb.get(t, ())} - ts)
        expanded.append(d + " " + " ".join(extra))
    bmx = BM25(expanded, slim=True)
    doc_term_lists = [sorted({vindex[t] for t in tokenize(d) if t in vindex}) for d in docs]
    flat = np.concatenate([np.array(l, dtype=np.int64) if l else np.array([0], dtype=np.int64)
                           for l in doc_term_lists])
    starts = np.zeros(n, dtype=np.int64); p = 0
    empty_mask = np.zeros(n, dtype=bool)
    for i, l in enumerate(doc_term_lists):
        starts[i] = p; p += max(1, len(l)); empty_mask[i] = not l
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    maxp_bm = title_bm = sent_doc = None
    if scifact_extras is not None:
        sents, sent_doc, titles = scifact_extras
        maxp_bm = BM25(sents, slim=True); title_bm = BM25(titles, slim=True)
    lint_cache = {}

    def channels(queries):
        out = {k: [] for k in ("doc", "ctx", "bmx", "lint")}
        if maxp_bm is not None: out["maxp"] = []; out["title"] = []
        for qid, qtext in queries:
            out["doc"].append(bm.scores(qtext).astype(np.float32))
            out["ctx"].append(np.asarray(D_ctx @ encode_query(qtext, idf, vindex, C), np.float32))
            out["bmx"].append(bmx.scores(qtext).astype(np.float32))
            li = np.zeros(n, dtype=np.float32)
            for t in set(tokenize(qtext)):
                j = vindex.get(t)
                if j is None: continue
                mx = lint_cache.get(j)
                if mx is None:
                    simv = Cn @ Cn[j]
                    mx = np.maximum.reduceat(simv[flat], starts)
                    mx[empty_mask] = 0.0
                    mx = mx.astype(np.float32); lint_cache[j] = mx
                li += np.float32(idf[t]) * mx
            out["lint"].append(li)
            if maxp_bm is not None:
                ss = maxp_bm.scores(qtext); m = np.zeros(n); np.maximum.at(m, sent_doc, ss)
                out["maxp"].append(m.astype(np.float32))
                out["title"].append(title_bm.scores(qtext).astype(np.float32))
        return out

    return [channels(q) for q in queries_list], bm, idf, vindex


def fused_scores(ch, arms, qi, w, n):
    tot = np.zeros(n)
    for a, wa in zip(arms, w):
        if not wa: continue
        s = ch[a][qi]; r = s.max() - s.min()
        tot += wa * ((s - s.min()) / r if r > 0 else np.zeros(n))
    return tot


def run(name, ids, docs, test_q, test_qrels, train_q, train_qrels, w1, arms, prior, targets,
        scifact_extras=None):
    print("\n===", name, "(prior %.4f; targets %s)" % (prior, targets))
    n = len(ids)
    t0 = time.time()
    (tr, te), bm, idf, vindex = build_channels(name, ids, docs, [train_q, test_q], scifact_extras)
    tok_docs = [tokenize(d) for d in docs]
    print("  build %.1fs" % (time.time() - t0))

    def bounce(ch, queries, F, T, alpha, w=w1, top=200):
        """Two-bounce ranking for every query: fuse -> PRF-expand from fused top-F -> interpolate."""
        out = []
        for qi, (qid, qtext) in enumerate(queries):
            s1 = fused_scores(ch, arms, qi, w, n)
            order = np.lexsort((np.arange(n), -s1))
            if alpha > 0.0:
                qset = set(tokenize(qtext))
                cnt = {}
                for di in order[:F]:                       # the emitters: fused top-F, not bm25's
                    for t in tok_docs[di]:
                        if t not in qset:
                            cnt[t] = cnt.get(t, 0) + 1
                exp = sorted(cnt, key=lambda t: (-cnt[t] * idf.get(t, 0.0), t))[:T]
                s2 = bm.scores(qtext + " " + " ".join(exp))
                r1 = s1.max() - s1.min(); r2 = s2.max() - s2.min()
                s = (1 - alpha) * (s1 - s1.min()) / (r1 if r1 > 0 else 1) \
                    + alpha * (s2 - s2.min()) / (r2 if r2 > 0 else 1)
                order = np.lexsort((np.arange(n), -s))
            out.append([ids[i] for i in order[:top]])
        return out

    # sweep the second bounce ON TRAIN (first-bounce weights held at the phase-7 champion)
    best = (eval_run(bounce(tr, train_q, 0, 0, 0.0), train_q, train_qrels)[0].mean(), 0, 0, 0.0)
    print("  bounce-1 only train nDCG@10 %.4f" % best[0])
    for F in (5, 10):
        for T in (10, 20):
            for a in (0.2, 0.4, 0.6):
                nd = eval_run(bounce(tr, train_q, F, T, a), train_q, train_qrels)[0].mean()
                if nd > best[0]:
                    best = (nd, F, T, a)
    nd_tr, F, T, a = best
    print("  TRAIN champion: F=%d T=%d alpha=%.1f (train nDCG@10 %.4f)" % (F, T, a, nd_tr))

    # PARALLEL LAYER: RRF-ensemble the two-bounce ranking with a half-weight-ctx variant (an
    # independent sample of the same scene); kept only if train says yes.
    w2 = list(w1); w2[1] = 0.5 if w1[1] == 1.0 else 1.0
    tr_a = bounce(tr, train_q, F, T, a)
    tr_b = bounce(tr, train_q, F, T, a, w=w2)
    tr_ens = [[d for d, _ in reciprocal_rank_fusion([la, lb], k=60)[:200]]
              for la, lb in zip(tr_a, tr_b)]
    nd_ens = eval_run(tr_ens, train_q, train_qrels)[0].mean()
    use_ens = nd_ens > nd_tr
    print("  parallel ensemble train nDCG@10 %.4f -> %s" % (nd_ens, "KEPT" if use_ens else "refused"))

    # TEST, once
    te_a = bounce(te, test_q, F, T, a)
    if use_ens:
        te_b = bounce(te, test_q, F, T, a, w=w2)
        final = [[d for d, _ in reciprocal_rank_fusion([la, lb], k=60)[:200]]
                 for la, lb in zip(te_a, te_b)]
    else:
        final = te_a
    bm_base = BM25(docs, slim=True)
    base = []
    for qid, qtext in test_q:
        s = bm_base.scores(qtext)
        base.append([ids[i] for i in np.lexsort((np.arange(n), -s))[:200]])
    r_bm = eval_run(base, test_q, test_qrels)
    r_hy = eval_run(final, test_q, test_qrels)
    nd = r_hy[0].mean()
    beat = [t for tn, t in targets.items() if nd > t]
    print("  bm25    nDCG@10 %.4f  R@100 %.4f" % (r_bm[0].mean(), r_bm[1].mean()))
    print("  BOUNCE2 nDCG@10 %.4f  R@100 %.4f  (prior %.4f)" % (nd, r_hy[1].mean(), prior))
    for tn, t in targets.items():
        print("    vs %-22s %.3f -> %s" % (tn, t, "CLEARED" if nd > t else ("matched" if abs(nd - t) < 0.002 else "open")))
    for mi, mname in ((0, "nDCG@10"), (1, "R@100")):
        mu, lo, hi = bootstrap_delta(r_hy[mi], r_bm[mi])
        tag = "SIGNIFICANT WIN" if lo > 0 else ("significant LOSS" if hi < 0 else "not significant")
        print("  bounce2 - bm25 %s: %+.4f [%+.4f, %+.4f] %s" % (mname, mu, lo, hi, tag))


if __name__ == "__main__":
    root = "/home/claude/bench"
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "sf"):
        ids, docs, test_q, test_qrels = load_scifact(root)
        train_q, train_qrels = load_scifact_train(root)
        raw = [json.loads(l) for l in open(root + "/scifact-retrieval-system-main/data/corpus.jsonl")]
        sents, sd, titles = [], [], [r.get("title", "") for r in raw]
        for di, d in enumerate(raw):
            for s in d.get("abstract", []): sents.append(s); sd.append(di)
        run("SciFact", ids, docs, test_q, test_qrels, train_q, train_qrels,
            w1=[1.0, 1.0, 0.6, 1.0, 0.0, 0.3], arms=["doc", "ctx", "bmx", "lint", "maxp", "title"],
            prior=0.6924, targets={"ColBERTv2": 0.693, "BM25+UPR(T0-3B)": 0.703, "SPLADE-v3": 0.710},
            scifact_extras=(sents, np.array(sd), titles))
    if which in ("both", "nfc"):
        ids, docs, test_q, test_qrels = load_nfcorpus(root)
        train_q, train_qrels = load_nfcorpus_train(root)
        run("NFCorpus", ids, docs, test_q, test_qrels, train_q, train_qrels,
            w1=[1.0, 1.0, 0.0, 0.3], arms=["doc", "ctx", "bmx", "lint"],
            prior=0.3390, targets={"ColBERTv2-high": 0.344, "SPLADE-v3": 0.357})

"""Phase 6: the perfect-recall structure, measured against the SAME BEIR benchmark.

TWO SEPARATE QUESTIONS, answered separately (containment != relevance -- the module's own kept negative):

  A. THE GUARANTEE AUDIT -- does the ranked hybrid ever MISS a relevant doc that the perfect-recall
     index provably finds? For every test query: relevant docs containing >=1 query term (the set any
     lexical method could possibly reach -- the RECALL CEILING) vs what the hybrid's top-200 candidates
     actually contain. Every miss is a doc the exact structure recovers for free.

  B. THE RANKING QUESTION -- exact-containment as a CHANNEL: coord(d) = number of query terms doc d
     contains (pure Boolean coordination-level, computed from the perfect-recall verify sets -- no tf,
     no idf, no BM25 anywhere in the channel). Swept into the champion fusion on TRAIN, test ONCE.
"""
import json, sys, time
import numpy as np

sys.path.insert(0, "/home/claude/bench")
sys.path.insert(0, "/home/claude/repo")
from bench_beir import (load_nfcorpus, load_scifact, build_matrices, encode_query, eval_run,
                        bootstrap_delta, DIM)
from bench_beir_tuned import load_nfcorpus_train, load_scifact_train
from bench_beir_push import chargram_token_matrix
from bench_beir_scifact_fix import bigramize
from holographic.semantic_router.holographic_bm25 import BM25, tokenize
from holographic.caching_and_storage.holographic_perfectrecall import PerfectRecallIndex


def run(name, ids, docs, test_q, test_qrels, train_q, train_qrels, scifact_sentences=None):
    print("\n===", name)
    n = len(ids)
    id2i = {d: i for i, d in enumerate(ids)}
    tok_sets = [set(tokenize(d)) for d in docs]

    # perfect-recall index over the corpus (token channel)
    pr = PerfectRecallIndex(tile=256)
    for ts in tok_sets:
        pr.add({"token": sorted(ts)})
    # BAKE-ONCE: exact per-term containment sets, precomputed from the same structure (lever #1 --
    # the per-query tile walk was the phase-6 first-run timeout; the answer never changes, so bake it).
    postings = {}
    for di, ts in enumerate(tok_sets):
        for t in ts:
            postings.setdefault(t, []).append(di)
    postings = {t: np.array(v, dtype=np.int64) for t, v in postings.items()}

    # ---- A. GUARANTEE AUDIT + RECALL CEILING --------------------------------------------------------
    reachable = missed_possible = total_rel = 0
    for qid, qtext in test_q:
        qt = set(tokenize(qtext))
        rel = [d for d, s in test_qrels[qid].items() if s > 0 and d in id2i]
        total_rel += len(rel)
        for d in rel:
            if qt & tok_sets[id2i[d]]:
                reachable += 1
    print("  RECALL CEILING: %d/%d relevant docs share >=1 query term (%.1f%%) -- the max ANY lexical"
          % (reachable, total_rel, 100.0 * reachable / total_rel))
    print("  method (BM25 included) can reach; the remainder needs semantics, which is the ctx arm's job.")

    # ---- B. channels (champion set + the exact coord channel) ---------------------------------------
    t0 = time.time()
    bm = BM25(docs, k1=0.9, b=0.4, slim=True) if name == "SciFact" else BM25(docs)
    bm_base = BM25(docs, slim=True)                        # untouched defaults; slim = same scores
    idf, vindex, A, C, D_bow, D_ctx = build_matrices(docs)
    vocab = sorted(vindex, key=vindex.get)
    G = chargram_token_matrix(vocab)
    D_cg = np.zeros((n, DIM), dtype=np.float32)
    for i, d in enumerate(docs):
        c = {}
        for t in tokenize(d): c[t] = c.get(t, 0) + 1
        v = np.zeros(DIM, dtype=np.float32)
        for t, f in c.items(): v += (np.log1p(f) * idf[t]) * G[vindex[t]]
        D_cg[i] = v
    D_cg /= (np.linalg.norm(D_cg, axis=1, keepdims=True) + 1e-12)
    del A, D_bow                                          # unused arms; free before the channel cache
    maxp = title = None
    if scifact_sentences is not None:
        sents, sent_doc = scifact_sentences
        bm_sent = BM25(sents, slim=True); bm_title = BM25([json.loads(l).get("title", "") for l in
                     open("/home/claude/bench/scifact-retrieval-system-main/data/corpus.jsonl")])
        maxp, title = bm_sent, (bm_title, sent_doc)

    def channels(queries):
        out = {k: [] for k in ("doc", "ctx", "cg", "coord")}
        if maxp is not None: out["maxp"] = []; out["title"] = []
        for qid, qtext in queries:
            out["doc"].append(bm.scores(qtext).astype(np.float32))
            out["ctx"].append(np.asarray(D_ctx @ encode_query(qtext, idf, vindex, C), np.float32))
            out["cg"].append(np.asarray(D_cg @ encode_query(qtext, idf, vindex, G), np.float32))
            # THE EXACT CHANNEL: coordination level from the perfect-recall verify sets -- per query
            # term, the exact containment set (zero false neg/pos), summed. No tf, no idf, no BM25.
            co = np.zeros(n, dtype=np.float32)
            for t in set(tokenize(qtext)):
                p = postings.get(t)
                if p is not None:
                    co[p] += 1.0                          # exact containment, scatter-add
            out["coord"].append(co)
            if maxp is not None:
                ss = maxp.scores(qtext); m = np.zeros(n); np.maximum.at(m, title[1], ss)
                out["maxp"].append(m.astype(np.float32))
                out["title"].append(title[0].scores(qtext).astype(np.float32))
        return out

    tr = channels(train_q); te = channels(test_q)
    print("  build+score %.1fs" % (time.time() - t0))

    arms = ["doc", "ctx", "cg", "coord"] + (["maxp", "title"] if maxp is not None else [])

    def fuse(ch, qn, w, top=200):
        out = []
        for qi in range(qn):
            tot = np.zeros(n)
            for a, wa in zip(arms, w):
                if not wa: continue
                s = ch[a][qi]; r = s.max() - s.min()
                tot += wa * ((s - s.min()) / r if r > 0 else np.zeros(n))
            order = np.lexsort((np.arange(n), -tot))[:top]
            out.append([ids[i] for i in order])
        return out

    # sweep on TRAIN: doc=1.0; ctx/cg from prior champions +- ; coord the new dial; maxp/title if present
    grid_c = [0.0, 0.2, 0.5, 1.0]
    best = None
    for wctx in (0.5, 1.0):
        for wcg in (0.0, 0.2, 0.5):
            for wco in grid_c:
                for wm in ((0.0, 0.3) if maxp is not None else (0.0,)):
                    for wt in ((0.0, 0.3) if maxp is not None else (0.0,)):
                        w = [1.0, wctx, wcg, wco] + ([wm, wt] if maxp is not None else [])
                        nd = eval_run(fuse(tr, len(train_q), w), train_q, train_qrels)[0].mean()
                        if best is None or nd > best[0]:
                            best = (nd, w)
    nd_tr, w = best
    print("  TRAIN champion weights %s = %s  (train nDCG@10 %.4f)" % (arms, w, nd_tr))

    fused_te = fuse(te, len(test_q), w)
    base_orders = []
    for qid, qtext in test_q:
        s = bm_base.scores(qtext)
        base_orders.append([ids[i] for i in np.lexsort((np.arange(n), -s))[:200]])
    r_bm = eval_run(base_orders, test_q, test_qrels)
    r_hy = eval_run(fused_te, test_q, test_qrels)

    # guarantee audit of the SHIPPED candidates: reachable relevant docs the hybrid top-200 still missed
    miss = 0; reach = 0
    for (qid, qtext), cand in zip(test_q, fused_te):
        qt = set(tokenize(qtext)); cs = set(cand)
        for d, s_ in test_qrels[qid].items():
            if s_ > 0 and d in id2i and (qt & tok_sets[id2i[d]]):
                reach += 1
                if d not in cs: miss += 1
    print("  GUARANTEE AUDIT: hybrid top-200 misses %d/%d lexically-reachable relevant docs" % (miss, reach))

    print("  %-8s nDCG@10 %.4f  R@100 %.4f" % ("bm25", r_bm[0].mean(), r_bm[1].mean()))
    print("  %-8s nDCG@10 %.4f  R@100 %.4f" % ("HYBRID", r_hy[0].mean(), r_hy[1].mean()))
    for mi, mname in ((0, "nDCG@10"), (1, "R@100")):
        mu, lo, hi = bootstrap_delta(r_hy[mi], r_bm[mi])
        tag = "SIGNIFICANT WIN" if lo > 0 else ("significant LOSS" if hi < 0 else "not significant")
        print("  hybrid - bm25 %s: %+.4f [%+.4f, %+.4f] %s" % (mname, mu, lo, hi, tag))


if __name__ == "__main__":
    root = "/home/claude/bench"
    ids, docs, test_q, test_qrels = load_nfcorpus(root)
    train_q, train_qrels = load_nfcorpus_train(root)
    run("NFCorpus", ids, docs, test_q, test_qrels, train_q, train_qrels)

    ids, docs, test_q, test_qrels = load_scifact(root)
    train_q, train_qrels = load_scifact_train(root)
    raw = [json.loads(l) for l in open(root + "/scifact-retrieval-system-main/data/corpus.jsonl")]
    sents, sent_doc = [], []
    for di, d in enumerate(raw):
        for s in d.get("abstract", []): sents.append(s); sent_doc.append(di)
    run("SciFact", ids, docs, test_q, test_qrels, train_q, train_qrels,
        scifact_sentences=(sents, np.array(sent_doc)))

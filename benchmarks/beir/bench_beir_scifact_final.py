"""Phase 5: SciFact final -- ALL levers in one sweep. Baseline untouched (k1=1.5,b=0.75, the published
reference); the FUSED SYSTEM may tune its own internal BM25 (k1,b) on train like any other weight.
Channels: doc-bm25(k1,b) + sentence-MaxP + title-F + bigram + holo-ctx + holo-cg. Train-swept, test once.
"""
import json, sys, time
import numpy as np

sys.path.insert(0, "/home/claude/bench")
sys.path.insert(0, "/home/claude/repo")
from bench_beir import load_scifact, build_matrices, encode_query, eval_run, bootstrap_delta, DIM, atom
from bench_beir_tuned import load_scifact_train
from bench_beir_push import chargram_token_matrix
from holographic.semantic_router.holographic_bm25 import BM25, tokenize
from bench_beir_scifact_fix import bigramize


def run():
    root = "/home/claude/bench"
    ids, docs, test_q, test_qrels = load_scifact(root)
    train_q, train_qrels = load_scifact_train(root)
    raw = [json.loads(l) for l in open(root + "/scifact-retrieval-system-main/data/corpus.jsonl")]
    titles = [d.get("title", "") for d in raw]
    sents, sent_doc = [], []
    for di, d in enumerate(raw):
        for s in d.get("abstract", []):
            sents.append(s); sent_doc.append(di)
    sent_doc = np.array(sent_doc); n = len(ids)

    t0 = time.time()
    kb_grid = [(0.9, 0.4), (0.9, 0.75), (1.2, 0.6), (1.5, 0.75), (2.0, 0.75)]
    bm_kb = {kb: BM25(docs, k1=kb[0], b=kb[1], slim=True) for kb in kb_grid}
    bm_sent = BM25(sents, slim=True); bm_title = BM25(titles); bm_bi = BM25([bigramize(d) for d in docs], slim=True)
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
    print("build %.1fs" % (time.time() - t0))

    def channels(queries):
        out = {("doc", kb): [] for kb in kb_grid}
        out.update({k: [] for k in ("maxp", "title", "bigram", "ctx", "cg")})
        for qid, qtext in queries:
            for kb in kb_grid:
                out[("doc", kb)].append(bm_kb[kb].scores(qtext).astype(np.float32))
            ss = bm_sent.scores(qtext); m = np.zeros(n); np.maximum.at(m, sent_doc, ss)
            out["maxp"].append(m.astype(np.float32))
            out["title"].append(bm_title.scores(qtext).astype(np.float32))
            out["bigram"].append(bm_bi.scores(bigramize(qtext)).astype(np.float32))
            out["ctx"].append(np.asarray(D_ctx @ encode_query(qtext, idf, vindex, C), np.float32))
            out["cg"].append(np.asarray(D_cg @ encode_query(qtext, idf, vindex, G), np.float32))
        return out

    t0 = time.time(); tr = channels(train_q); te = channels(test_q)
    print("score %.1fs" % (time.time() - t0))

    def normed(ch, key, qi):
        s = ch[key][qi]; r = s.max() - s.min()
        return (s - s.min()) / r if r > 0 else np.zeros(n)

    def fuse(ch, qn, kb, w, top=200):
        keys = [("doc", kb), "maxp", "title", "bigram", "ctx", "cg"]
        out = []
        for qi in range(qn):
            tot = np.zeros(n)
            for key, wa in zip(keys, w):
                if wa: tot += wa * normed(ch, key, qi)
            order = np.lexsort((np.arange(n), -tot))[:top]
            out.append([ids[i] for i in order])
        return out

    best = None
    for kb in kb_grid:
        for wm in (0.0, 0.3, 0.6):
            for wt in (0.0, 0.3):
                for wb in (0.0, 0.3, 0.6):
                    for wc in (0.5, 1.0):
                        for wg in (0.0, 0.5):
                            w = (1.0, wm, wt, wb, wc, wg)
                            nd = eval_run(fuse(tr, len(train_q), kb, w), train_q, train_qrels)[0].mean()
                            if best is None or nd > best[0]:
                                best = (nd, kb, w)
    nd_tr, kb, w = best
    print("TRAIN champion: k1=%.1f b=%.2f  weights(doc,maxp,title,bigram,ctx,cg)=%s  train nDCG@10 %.4f"
          % (kb[0], kb[1], w, nd_tr))

    fused_te = fuse(te, len(test_q), kb, w)
    base = [[ids[i] for i in np.lexsort((np.arange(n), -te[("doc", (1.5, 0.75))][qi]))[:200]]
            for qi in range(len(test_q))]
    r_bm = eval_run(base, test_q, test_qrels)
    r_hy = eval_run(fused_te, test_q, test_qrels)
    print("%-8s nDCG@10 %.4f  R@100 %.4f  MRR@10 %.4f" % ("bm25", *[x.mean() for x in r_bm]))
    print("%-8s nDCG@10 %.4f  R@100 %.4f  MRR@10 %.4f" % ("HYBRID", *[x.mean() for x in r_hy]))
    for mi, mname in ((0, "nDCG@10"), (1, "R@100"), (2, "MRR@10")):
        mu, lo, hi = bootstrap_delta(r_hy[mi], r_bm[mi])
        tag = "SIGNIFICANT WIN" if lo > 0 else ("significant LOSS" if hi < 0 else "not significant")
        print("hybrid - bm25 %s: %+.4f [%+.4f, %+.4f] %s" % (mname, mu, lo, hi, tag))


if __name__ == "__main__":
    run()

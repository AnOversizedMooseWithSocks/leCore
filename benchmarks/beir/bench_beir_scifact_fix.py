"""Phase 4: SciFact nDCG fix -- the demoscene answer (domain operators), applied to text.

THE DIAGNOSIS (Quilez seat, domain operators / the tiling lever): BM25 over the whole abstract averages
the match over ~9 sentences, but a SciFact claim is evidenced by ONE sentence -- doc-level length
normalization dilutes exactly the signal nDCG@10 needs. The fix is not a bigger scorer; it is evaluating
the field where the detail lives:

  * SENT-MAXP  -- tile the domain: BM25 over the corpus's own SENTENCES, doc score = max over its
                  sentences (passage MaxP -- Dai & Callan 2019, here with BM25 not BERT). The tiling lever.
  * TITLE-F    -- LOD fields: a separate BM25 over titles fused in (BM25F spirit -- Robertson 2004).
                  A title hit is a coarse-mip hit: few words, high signal.
  * BIGRAM     -- more dimensions: BM25 over ordered adjacent-pair terms 'a_b' (sequential dependence --
                  Metzler & Croft 2005). Word ORDER is the dimension bag-of-words lacks.

All pure counting + the existing holographic arms; weights swept on TRAIN only, TEST touched once.
"""
import json, sys, time
import numpy as np

sys.path.insert(0, "/home/claude/bench")
sys.path.insert(0, "/home/claude/repo")
from bench_beir import load_scifact, build_matrices, encode_query, eval_run, bootstrap_delta
from bench_beir_tuned import load_scifact_train, arm_orders
from bench_beir_push import chargram_token_matrix
from holographic.semantic_router.holographic_bm25 import BM25, tokenize


def bigramize(text):
    t = tokenize(text)
    return " ".join(a + "_" + b for a, b in zip(t, t[1:]))


def run():
    root = "/home/claude/bench"
    ids, docs, test_q, test_qrels = load_scifact(root)
    train_q, train_qrels = load_scifact_train(root)
    raw = [json.loads(l) for l in open(root + "/scifact-retrieval-system-main/data/corpus.jsonl")]
    titles = [d.get("title", "") for d in raw]
    sents, sent_doc = [], []                                  # flattened sentence corpus + owner map
    for di, d in enumerate(raw):
        for s in d.get("abstract", []):
            sents.append(s); sent_doc.append(di)
    sent_doc = np.array(sent_doc)
    n = len(ids)
    print("docs %d, sentences %d, titles %d" % (n, len(sents), n))

    t0 = time.time()
    bm_doc = BM25(docs)
    bm_sent = BM25(sents)
    bm_title = BM25(titles)
    bm_bi = BM25([bigramize(d) for d in docs])
    idf, vindex, A, C, D_bow, D_ctx = build_matrices(docs)
    print("build %.1fs" % (time.time() - t0))

    def channels(queries):
        """Per-query score arrays for every channel, length-n each."""
        out = {k: [] for k in ("doc", "maxp", "title", "bigram", "ctx")}
        for qid, qtext in queries:
            out["doc"].append(bm_doc.scores(qtext))
            ss = bm_sent.scores(qtext)                         # sentence field, then MAX back onto docs:
            m = np.zeros(n)                                    # the tile whose local match is strongest
            np.maximum.at(m, sent_doc, ss)                     # speaks for the doc (MaxP)
            out["maxp"].append(m)
            out["title"].append(bm_title.scores(qtext))
            out["bigram"].append(bm_bi.scores(bigramize(qtext)))
            qv = encode_query(qtext, idf, vindex, C)
            out["ctx"].append(np.asarray(D_ctx @ qv, np.float64))
        return out

    t0 = time.time()
    tr = channels(train_q); te = channels(test_q)
    print("score %.1fs" % (time.time() - t0))

    arms = ("doc", "maxp", "title", "bigram", "ctx")

    def combsum(ch, qn, w, top=200):
        out = []
        for qi in range(qn):
            tot = np.zeros(n)
            for a, wa in zip(arms, w):
                if wa == 0.0:
                    continue
                s = ch[a][qi]
                r = s.max() - s.min()
                tot += wa * ((s - s.min()) / r if r > 0 else np.zeros(n))
            order = np.lexsort((np.arange(n), -tot))[:top]
            out.append([ids[i] for i in order])
        return out

    # sweep ON TRAIN: doc fixed 1.0; maxp/title/bigram/ctx over a coarse grid (the lever mix)
    grid = [0.0, 0.3, 0.6, 1.0]
    best = None
    for wm in grid:
        for wt in grid:
            for wb in grid:
                for wc in (0.0, 0.5, 1.0):
                    w = (1.0, wm, wt, wb, wc)
                    nd = eval_run(combsum(tr, len(train_q), w), train_q, train_qrels)[0].mean()
                    if best is None or nd > best[0]:
                        best = (nd, w)
    nd_tr, w = best
    print("TRAIN champion weights (doc,maxp,title,bigram,ctx) = %s  train nDCG@10 %.4f" % (w, nd_tr))

    fused_te = combsum(te, len(test_q), w)
    bm_order_te = [[ids[i] for i in np.lexsort((np.arange(n), -te["doc"][qi]))[:200]]
                   for qi in range(len(test_q))]
    r_bm = eval_run(bm_order_te, test_q, test_qrels)
    r_hy = eval_run(fused_te, test_q, test_qrels)
    # per-channel solo scores for the record (which lever did the work?)
    print("%-10s nDCG@10 %.4f  R@100 %.4f  MRR@10 %.4f" % ("bm25-doc", *[x.mean() for x in r_bm]))
    for a in ("maxp", "title", "bigram", "ctx"):
        solo = [[ids[i] for i in np.lexsort((np.arange(n), -te[a][qi]))[:200]] for qi in range(len(test_q))]
        r = eval_run(solo, test_q, test_qrels)
        print("%-10s nDCG@10 %.4f  R@100 %.4f  MRR@10 %.4f  (solo)" % (a, *[x.mean() for x in r]))
    print("%-10s nDCG@10 %.4f  R@100 %.4f  MRR@10 %.4f" % ("HYBRID", *[x.mean() for x in r_hy]))
    for mi, mname in ((0, "nDCG@10"), (1, "R@100"), (2, "MRR@10")):
        mu, lo, hi = bootstrap_delta(r_hy[mi], r_bm[mi])
        tag = "SIGNIFICANT WIN" if lo > 0 else ("significant LOSS" if hi < 0 else "not significant")
        print("hybrid - bm25 %s: %+.4f [%+.4f, %+.4f] %s" % (mname, mu, lo, hi, tag))


if __name__ == "__main__":
    run()

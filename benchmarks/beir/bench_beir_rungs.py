"""Phase 7: the two no-weights rungs from the SOTA ladder, measured. Train-swept, test ONCE.

  RUNG 1 -- DOC EXPANSION (the SPLADE-v3 analogue, target NFCorpus 0.357): SPLADE's edge is LEARNED
    expansion -- a doc gains terms it implies but never says. The zero-weights analogue: at index time,
    append each doc term's top-3 CTX-NEIGHBOR terms (random-indexing cosine -- meaning measured from the
    corpus itself), then BM25 over the EXPANDED text as a channel ('bmx'). Attacks exactly the measured
    71% lexically-unreachable NFCorpus relevance.

  RUNG 2 -- TERM-LEVEL LATE INTERACTION (the ColBERTv2 analogue, target SciFact 0.693): ColBERT scores
    sum_q max_d sim(t_q, t_d) over token embeddings. The zero-weights analogue ('lint'): the same maxsim
    over our ctx token vectors, idf-weighted per query term -- the maxp lever at TERM granularity.
    Computed with one reduceat per query term over a flattened doc-term index (no per-doc Python loop).
"""
import json, sys, time
import numpy as np

sys.path.insert(0, "/home/claude/bench")
sys.path.insert(0, "/home/claude/repo")
from bench_beir import (load_nfcorpus, load_scifact, build_matrices, encode_query, eval_run,
                        bootstrap_delta, DIM)
from bench_beir_tuned import load_nfcorpus_train, load_scifact_train
from bench_beir_push import chargram_token_matrix
from holographic.semantic_router.holographic_bm25 import BM25, tokenize


def ctx_neighbors(C, vocab, topk=3, chunk=512):
    """Per term: its topk nearest OTHER terms by ctx cosine. Chunked matmul (the phase-5 OOM lesson)."""
    V = len(vocab)
    nb = {}
    for s in range(0, V, chunk):
        sims = C[s:s + chunk] @ C.T                       # (chunk, V) float32
        for r in range(sims.shape[0]):
            row = sims[r]
            row[s + r] = -2.0                             # not yourself
            idx = np.argpartition(-row, topk)[:topk]
            idx = idx[np.argsort(-row[idx], kind="stable")]
            nb[vocab[s + r]] = [vocab[j] for j in idx]
    return nb


def run(name, ids, docs, test_q, test_qrels, train_q, train_qrels, prior_champ, target,
        scifact_extras=None):
    print("\n===", name, "(prior champion %.4f; ladder target %.3f)" % (prior_champ, target))
    n = len(ids)
    t0 = time.time()
    bm = BM25(docs, k1=0.9, b=0.4, slim=True) if name == "SciFact" else BM25(docs)
    bm_base = BM25(docs, slim=True)
    idf, vindex, A, C, D_bow, D_ctx = build_matrices(docs)
    del A, D_bow
    vocab = sorted(vindex, key=vindex.get)

    # RUNG 1: expanded-doc BM25. Neighbors from ctx cosine; each distinct doc term contributes its
    # top-3 neighbors ONCE (presence, not tf -- expansion should whisper, not shout).
    nb = ctx_neighbors(C, vocab, topk=3)
    expanded = []
    for d in docs:
        ts = set(tokenize(d))
        extra = sorted({e for t in ts for e in nb.get(t, ())} - ts)
        expanded.append(d + " " + " ".join(extra))
    bmx = BM25(expanded, slim=True)

    # RUNG 2 prep: flattened doc-term index for the reduceat maxsim
    doc_term_lists = [sorted({vindex[t] for t in tokenize(d) if t in vindex}) for d in docs]
    flat = np.concatenate([np.array(l, dtype=np.int64) if l else np.array([0], dtype=np.int64)
                           for l in doc_term_lists])
    starts = np.zeros(n, dtype=np.int64)
    p = 0
    empty_mask = np.zeros(n, dtype=bool)
    for i, l in enumerate(doc_term_lists):
        starts[i] = p
        p += max(1, len(l))
        empty_mask[i] = not l
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)

    # SciFact keeps its earlier winning channels
    maxp_bm = title_bm = sent_doc = None
    if scifact_extras is not None:
        sents, sent_doc, titles = scifact_extras
        maxp_bm = BM25(sents, slim=True); title_bm = BM25(titles, slim=True)

    lint_cache = {}
    def channels(queries):
        out = {k: [] for k in ("doc", "ctx", "cg_skip", "bmx", "lint")}
        out.pop("cg_skip")
        if maxp_bm is not None: out["maxp"] = []; out["title"] = []
        for qid, qtext in queries:
            out["doc"].append(bm.scores(qtext).astype(np.float32))
            out["ctx"].append(np.asarray(D_ctx @ encode_query(qtext, idf, vindex, C), np.float32))
            out["bmx"].append(bmx.scores(qtext).astype(np.float32))
            # lint: sum_q idf(t) * max over doc terms of ctx-cosine(t, term)
            li = np.zeros(n, dtype=np.float32)
            for t in set(tokenize(qtext)):
                j = vindex.get(t)
                if j is None: continue
                mx = lint_cache.get(j)
                if mx is None:                             # bake-once per vocab term (queries share vocab
                    simv = Cn @ Cn[j]                      # heavily; the timeout was recomputing this)
                    mx = np.maximum.reduceat(simv[flat], starts)
                    mx[empty_mask] = 0.0
                    mx = mx.astype(np.float32)
                    lint_cache[j] = mx
                li += np.float32(idf[t]) * mx
            out["lint"].append(li)
            if maxp_bm is not None:
                ss = maxp_bm.scores(qtext); m = np.zeros(n); np.maximum.at(m, sent_doc, ss)
                out["maxp"].append(m.astype(np.float32))
                out["title"].append(title_bm.scores(qtext).astype(np.float32))
        return out

    tr = channels(train_q); te = channels(test_q)
    print("  build+score %.1fs" % (time.time() - t0))

    arms = ["doc", "ctx", "bmx", "lint"] + (["maxp", "title"] if maxp_bm is not None else [])

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

    # solo scores first (which rung carries?)
    for a in arms:
        solo = [[ids[i] for i in np.lexsort((np.arange(n), -te[a][qi]))[:200]] for qi in range(len(test_q))]
        r = eval_run(solo, test_q, test_qrels)
        print("  %-6s solo nDCG@10 %.4f  R@100 %.4f" % (a, r[0].mean(), r[1].mean()))

    grid = (0.0, 0.3, 0.6, 1.0)
    best = None
    for wctx in (0.5, 1.0):
        for wbx in grid:
            for wli in grid:
                for wm in ((0.0, 0.3) if maxp_bm is not None else (0.0,)):
                    for wt in ((0.0, 0.3) if maxp_bm is not None else (0.0,)):
                        w = [1.0, wctx, wbx, wli] + ([wm, wt] if maxp_bm is not None else [])
                        nd = eval_run(fuse(tr, len(train_q), w), train_q, train_qrels)[0].mean()
                        if best is None or nd > best[0]:
                            best = (nd, w)
    nd_tr, w = best
    print("  TRAIN champion %s = %s (train nDCG@10 %.4f)" % (arms, w, nd_tr))

    fused_te = fuse(te, len(test_q), w)
    base = []
    for qid, qtext in test_q:
        s = bm_base.scores(qtext)
        base.append([ids[i] for i in np.lexsort((np.arange(n), -s))[:200]])
    r_bm = eval_run(base, test_q, test_qrels)
    r_hy = eval_run(fused_te, test_q, test_qrels)
    nd = r_hy[0].mean()
    print("  bm25    nDCG@10 %.4f  R@100 %.4f" % (r_bm[0].mean(), r_bm[1].mean()))
    print("  HYBRID7 nDCG@10 %.4f  R@100 %.4f   (prior champ %.4f, ladder target %.3f -> %s)"
          % (nd, r_hy[1].mean(), prior_champ, target,
             "TARGET CLEARED" if nd > target else ("improved" if nd > prior_champ else "no gain")))
    for mi, mname in ((0, "nDCG@10"), (1, "R@100")):
        mu, lo, hi = bootstrap_delta(r_hy[mi], r_bm[mi])
        tag = "SIGNIFICANT WIN" if lo > 0 else ("significant LOSS" if hi < 0 else "not significant")
        print("  hybrid7 - bm25 %s: %+.4f [%+.4f, %+.4f] %s" % (mname, mu, lo, hi, tag))


if __name__ == "__main__":
    root = "/home/claude/bench"
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "nfc"):
        ids, docs, test_q, test_qrels = load_nfcorpus(root)
        train_q, train_qrels = load_nfcorpus_train(root)
        run("NFCorpus", ids, docs, test_q, test_qrels, train_q, train_qrels,
            prior_champ=0.3371, target=0.357)
    if which in ("both", "sf"):
        ids, docs, test_q, test_qrels = load_scifact(root)
        train_q, train_qrels = load_scifact_train(root)
        raw = [json.loads(l) for l in open(root + "/scifact-retrieval-system-main/data/corpus.jsonl")]
        sents, sd, titles = [], [], [r.get("title", "") for r in raw]
        for di, d in enumerate(raw):
            for s in d.get("abstract", []): sents.append(s); sd.append(di)
        run("SciFact", ids, docs, test_q, test_qrels, train_q, train_qrels,
            prior_champ=0.6854, target=0.693, scifact_extras=(sents, np.array(sd), titles))

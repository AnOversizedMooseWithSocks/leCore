"""Phase 2: TRAIN-TUNED hybrid -- the honest SOTA protocol (tune on train qrels, touch test ONCE).

Phase 1 finding: holo-ctx (random-indexing semantics) BEATS bm25 on NFCorpus (vocab-mismatch regime) and
LOSES on SciFact (shared-terminology regime); a fixed-weight hybrid therefore cannot win both. The fix is
the dispatcher lesson at the WEIGHT level: per-dataset fusion weights chosen on the train split.

Also swept on train: CombSUM (min-max score fusion) vs RRF, and a bow+ctx doc-vector mix.
"""
import json, csv, sys, time
import numpy as np

sys.path.insert(0, "/home/claude/bench")
sys.path.insert(0, "/home/claude/repo")
from bench_beir import (load_nfcorpus, load_scifact, build_matrices, encode_query,
                        eval_run, bootstrap_delta)
from holographic.semantic_router.holographic_bm25 import BM25, reciprocal_rank_fusion, tokenize


def load_nfcorpus_train(root):
    qrels = {}
    for r in csv.DictReader(open(root + "/RAG_nfcorpus-main/assets/train.csv")):
        qrels.setdefault(r["query-id"], {})[r["corpus-id"]] = int(r["score"])
    qtext = {r["_id"]: (r["title"] + " " + r["text"]).strip()
             for r in csv.DictReader(open(root + "/RAG_nfcorpus-main/assets/queries.csv"))}
    return [(q, qtext[q]) for q in sorted(qrels) if q in qtext], qrels


def load_scifact_train(root):
    qrels = {}
    for line in open(root + "/retrieval-evolution-study-main/data/datasets/scifact/qrels/train.tsv").read().strip().splitlines()[1:]:
        q, c, s = line.split("\t"); qrels.setdefault(q, {})[c] = int(s)
    qtext = {json.loads(l)["_id"]: json.loads(l)["text"]
             for l in open(root + "/retrieval-evolution-study-main/data/datasets/scifact/queries.jsonl")}
    return [(q, qtext[q]) for q in sorted(qrels) if q in qtext], qrels


def arm_orders(queries, bm, idf, vindex, A, C, D_bow, D_ctx, ids, top=200):
    """Per-query ranked doc-id lists AND raw score arrays for every arm (scores needed for CombSUM)."""
    out = {"bm25": [], "holo-bow": [], "holo-ctx": []}
    sc = {"bm25": [], "holo-bow": [], "holo-ctx": []}
    n = len(ids)
    for qid, qtext in queries:
        s_bm = bm.scores(qtext)
        order = np.lexsort((np.arange(n), -s_bm))[:top]
        out["bm25"].append([ids[i] for i in order]); sc["bm25"].append(s_bm)
        for arm, M, D in (("holo-bow", A, D_bow), ("holo-ctx", C, D_ctx)):
            qv = encode_query(qtext, idf, vindex, M)
            s = D @ qv
            order = np.lexsort((np.arange(n), -s))[:top]
            out[arm].append([ids[i] for i in order]); sc[arm].append(np.asarray(s, np.float64))
    return out, sc


def fuse_rrf(orders, qn, w_bow, w_ctx, top=200):
    fused_out = []
    for qi in range(qn):
        fused = reciprocal_rank_fusion(
            [orders["bm25"][qi], orders["holo-bow"][qi], orders["holo-ctx"][qi]],
            k=60, weights=[1.0, w_bow, w_ctx])
        fused_out.append([d for d, _ in fused[:top]])
    return fused_out


def fuse_combsum(scores, ids, qn, w_bow, w_ctx, top=200):
    """CombSUM over min-max-normalized scores -- the score-level alternative RRF's docstring warns is
    brittle; swept here so the warning is measured on real data, not assumed."""
    fused_out = []
    n = len(ids)
    for qi in range(qn):
        tot = np.zeros(n)
        for arm, w in (("bm25", 1.0), ("holo-bow", w_bow), ("holo-ctx", w_ctx)):
            s = scores[arm][qi]
            rng_ = s.max() - s.min()
            tot += w * ((s - s.min()) / rng_ if rng_ > 0 else np.zeros(n))
        order = np.lexsort((np.arange(n), -tot))[:top]
        fused_out.append([ids[i] for i in order])
    return fused_out


def run(name, ids, docs, test_q, test_qrels, train_q, train_qrels):
    print("\n===", name, "-- train queries", len(train_q), "test queries", len(test_q))
    t0 = time.time()
    bm = BM25(docs)
    idf, vindex, A, C, D_bow, D_ctx = build_matrices(docs)
    tr_orders, tr_scores = arm_orders(train_q, bm, idf, vindex, A, C, D_bow, D_ctx, ids)
    te_orders, te_scores = arm_orders(test_q, bm, idf, vindex, A, C, D_bow, D_ctx, ids)
    print("  build+score %.1fs" % (time.time() - t0))

    # ---- sweep ON TRAIN ONLY ------------------------------------------------------------------------
    grid = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
    best = None
    for method in ("rrf", "combsum"):
        for wb in grid:
            for wc in grid:
                fused = (fuse_rrf(tr_orders, len(train_q), wb, wc) if method == "rrf"
                         else fuse_combsum(tr_scores, ids, len(train_q), wb, wc))
                nd = eval_run(fused, train_q, train_qrels)[0].mean()
                if best is None or nd > best[0]:
                    best = (nd, method, wb, wc)
    nd_tr, method, wb, wc = best
    print("  TRAIN champion: %s  w_bow=%.2f w_ctx=%.2f  (train nDCG@10 %.4f)" % (method, wb, wc, nd_tr))

    # ---- evaluate champion ON TEST, once ------------------------------------------------------------
    fused_te = (fuse_rrf(te_orders, len(test_q), wb, wc) if method == "rrf"
                else fuse_combsum(te_scores, ids, len(test_q), wb, wc))
    runs = {"bm25": eval_run(te_orders["bm25"], test_q, test_qrels),
            "holo-ctx": eval_run(te_orders["holo-ctx"], test_q, test_qrels),
            "tuned-hybrid": eval_run(fused_te, test_q, test_qrels)}
    hdr = "%-14s %8s %8s %8s" % ("arm", "nDCG@10", "R@100", "MRR@10")
    print("  " + hdr); print("  " + "-" * len(hdr))
    for arm in ("bm25", "holo-ctx", "tuned-hybrid"):
        nd, rec, mrr = runs[arm]
        print("  %-14s %8.4f %8.4f %8.4f" % (arm, nd.mean(), rec.mean(), mrr.mean()))
    for arm in ("holo-ctx", "tuned-hybrid"):
        mu, lo, hi = bootstrap_delta(runs[arm][0], runs["bm25"][0])
        tag = "SIGNIFICANT WIN" if lo > 0 else ("significant LOSS" if hi < 0 else "not significant")
        print("  %s - bm25 nDCG@10: %+.4f [%+.4f, %+.4f] %s" % (arm, mu, lo, hi, tag))
        mu, lo, hi = bootstrap_delta(runs[arm][1], runs["bm25"][1])
        tag = "SIGNIFICANT WIN" if lo > 0 else ("significant LOSS" if hi < 0 else "not significant")
        print("  %s - bm25 R@100  : %+.4f [%+.4f, %+.4f] %s" % (arm, mu, lo, hi, tag))


if __name__ == "__main__":
    root = "/home/claude/bench"
    ids, docs, test_q, test_qrels = load_nfcorpus(root)
    train_q, train_qrels = load_nfcorpus_train(root)
    run("NFCorpus", ids, docs, test_q, test_qrels, train_q, train_qrels)
    ids, docs, test_q, test_qrels = load_scifact(root)
    train_q, train_qrels = load_scifact_train(root)
    run("SciFact", ids, docs, test_q, test_qrels, train_q, train_qrels)

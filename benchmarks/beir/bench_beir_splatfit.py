"""Phase 9: SPLAT-FIT TERM WEIGHTS -- the 3DGS move pointed at retrieval.

THE INSIGHT (the user's): a Gaussian splat's parameters are fit by DETERMINISTIC gradient descent to
reproduce a target image -- seeded, reproducible, no neural network. 'Learned term weighting' (the edge
behind SPLADE's open rungs) is the SAME object: give each vocabulary term a scalar weight w_t on its
BM25 contribution and fit w by seeded gradient descent on the TRAIN qrels. Protocol unchanged from every
prior phase (train-only fitting, test touched once); capacity raised from ~6 fusion weights to |V| term
weights, disciplined by:
  * L2 pull toward w=1 (lambda swept) -- the prior IS BM25; fitting only reweights where train evidence
    pushes,
  * a HELD-OUT train slice (15% of train queries) gating epochs and lambda -- overfit dies there, never
    on test,
  * full determinism: seeded rng for negatives/init order, fixed iteration order, float64 accumulation.

MODEL: s_w(q, d) = sum over shared terms of w_t * contrib_t(q, d), where contrib is exactly BM25's
per-term posting weight (w=1 recovers BM25 bit-for-bit). LOSS: per query, listwise softmax
cross-entropy of the relevant docs among a candidate set (BM25 top-C union relevants). GRADIENT:
dL/dw_t = sum_d (softmax_d - y_d) * contrib_t(d) -- one small dense (terms x cands) matmul per query.

Then the reweighted channel replaces 'doc' in the phase-8 stack; fusion + PRF re-swept on train; test once.
"""
import json, sys, time
import numpy as np

sys.path.insert(0, "/home/claude/bench")
sys.path.insert(0, "/home/claude/repo")
from bench_beir import load_nfcorpus, load_scifact, eval_run, bootstrap_delta
from bench_beir_tuned import load_nfcorpus_train, load_scifact_train
from bench_beir_bounce import build_channels, fused_scores
from holographic.semantic_router.holographic_bm25 import BM25, tokenize


def build_query_mats(queries, qrels, bm, id2i, ids, n, cand=150):
    """Per query: (term_ids, cand_doc_ids, M[t,c] contribution matrix, y relevance mask). Candidates =
    BM25 top-`cand` union the relevant docs (so every positive has gradient)."""
    from collections import Counter
    mats = []
    for qid, qtext in queries:
        s = bm.scores(qtext)
        order = np.lexsort((np.arange(n), -s))[:cand]
        rel = [id2i[d] for d, sv in qrels[qid].items() if sv > 0 and d in id2i]
        cands = sorted(set(order.tolist()) | set(rel))
        cpos = {d: i for i, d in enumerate(cands)}
        terms, rows = [], []
        for t, c in Counter(tokenize(qtext)).items():
            post = bm._postings.get(t)
            if post is None: continue
            idxs, wts = post
            row = np.zeros(len(cands))
            hit = False
            for ix, wv in zip(idxs, wts):
                j = cpos.get(int(ix))
                if j is not None:
                    row[j] = c * wv; hit = True
            if hit:
                terms.append(t); rows.append(row)
        if not terms or not rel: continue
        M = np.array(rows)                                # (T, C)
        y = np.zeros(len(cands)); 
        for d in rel:
            y[cpos[d]] = 1.0
        y /= y.sum()
        mats.append((terms, np.array(cands), M, y))
    return mats


def fit_weights(mats_fit, mats_val, vindex, lam, epochs=200, lr=0.3, seed=0):
    """Seeded full-batch gradient descent, ADAGRAD-scaled per coordinate (the 3DGS lesson: parameters
    with wildly different gradient frequencies -- a term seen in one query vs a thousand -- need
    per-splat step sizes; a single global lr left the first fit at |w-1|=0.001, a measured negative).
    Early stop on the held-out slice; deterministic (fixed order, no sampling)."""
    V = len(vindex)
    w = np.ones(V)
    G2 = np.zeros(V) + 1e-8                               # adagrad accumulator
    js_cache = [np.array([vindex[t] for t in terms]) for terms, _, _, _ in mats_fit]
    js_val = [np.array([vindex[t] for t in terms]) for terms, _, _, _ in mats_val]

    def val_loss(w):
        L = 0.0
        for (terms, cands, M, y), js in zip(mats_val, js_val):
            s = w[js] @ M; s = s - s.max()
            p = np.exp(s); p /= p.sum()
            L += -float(np.sum(y * np.log(p + 1e-12)))
        return L

    best = (val_loss(w), w.copy())
    for ep in range(epochs):
        g = lam * (w - 1.0)
        for (terms, cands, M, y), js in zip(mats_fit, js_cache):
            s = w[js] @ M; s = s - s.max()
            p = np.exp(s); p /= p.sum()
            g[js] += M @ (p - y)
        G2 += g * g
        w = w - lr * g / np.sqrt(G2)
        np.clip(w, 0.0, 4.0, out=w)
        if ep % 10 == 9:
            Lv = val_loss(w)
            if Lv < best[0]:
                best = (Lv, w.copy())
    return best[1]


def reweighted_scores(qtext, bm, w, vindex, n):
    """Full-corpus reweighted lexical scores: BM25's scatter-add with w_t on each term (w=1 == BM25)."""
    from collections import Counter
    out = np.zeros(n)
    for t, c in Counter(tokenize(qtext)).items():
        post = bm._postings.get(t)
        if post is None: continue
        j = vindex.get(t)
        wt = w[j] if j is not None else 1.0
        idxs, wts = post
        out[idxs] += wt * c * wts
    return out


def run(name, ids, docs, test_q, test_qrels, train_q, train_qrels, w1, arms, prior, targets,
        prf, scifact_extras=None):
    print("\n===", name, "(prior %.4f)" % prior)
    n = len(ids); id2i = {d: i for i, d in enumerate(ids)}
    t0 = time.time()
    (tr, te), bm, idf, vindex = build_channels(name, ids, docs, [train_q, test_q], scifact_extras)
    print("  channels %.1fs" % (time.time() - t0))

    # split train 85/15 for the held-out gate, deterministically
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(train_q))
    cut = int(0.85 * len(train_q))
    fit_q = [train_q[i] for i in perm[:cut]]; val_q = [train_q[i] for i in perm[cut:]]
    t0 = time.time()
    mats_fit = build_query_mats(fit_q, train_qrels, bm, id2i, ids, n)
    mats_val = build_query_mats(val_q, train_qrels, bm, id2i, ids, n)
    print("  mats %.1fs (%d fit / %d val queries usable)" % (time.time() - t0, len(mats_fit), len(mats_val)))

    # lambda swept on the VALIDATION slice via full nDCG of the reweighted channel alone
    best = None
    for lam in (10.0, 3.0, 1.0, 0.3):
        w = fit_weights(mats_fit, mats_val, vindex, lam)
        val_rank = []
        for qid, qtext in val_q:
            s = reweighted_scores(qtext, bm, w, vindex, n)
            val_rank.append([ids[i] for i in np.lexsort((np.arange(n), -s))[:200]])
        nd = eval_run(val_rank, val_q, train_qrels)[0].mean()
        moved = float(np.abs(w - 1.0).mean())
        print("    lambda %.1f: val nDCG@10 %.4f (mean |w-1| %.3f)" % (lam, nd, moved))
        if best is None or nd > best[0]:
            best = (nd, lam, w)
    _, lam, w = best
    print("  splat-fit champion lambda %.1f" % lam)

    # replace the lexical channel with the reweighted one everywhere
    for ch, qs in ((tr, train_q), (te, test_q)):
        ch["doc"] = [reweighted_scores(qtext, bm, w, vindex, n).astype(np.float32) for _, qtext in qs]

    # light re-sweep of fusion weights around the champion + the PRF second bounce (NFC only, per phase-8)
    tok_docs = [tokenize(d) for d in docs]

    def bounce(ch, queries, F, T, alpha, wv, top=200):
        out = []
        for qi, (qid, qtext) in enumerate(queries):
            s1 = fused_scores(ch, arms, qi, wv, n)
            order = np.lexsort((np.arange(n), -s1))
            if alpha > 0:
                qset = set(tokenize(qtext)); cnt = {}
                for di in order[:F]:
                    for t in tok_docs[di]:
                        if t not in qset: cnt[t] = cnt.get(t, 0) + 1
                exp = sorted(cnt, key=lambda t: (-cnt[t] * idf.get(t, 0.0), t))[:T]
                s2 = reweighted_scores(qtext + " " + " ".join(exp), bm, w, vindex, n)
                r1 = s1.max() - s1.min(); r2 = s2.max() - s2.min()
                s1 = (1 - alpha) * (s1 - s1.min()) / (r1 or 1) + alpha * (s2 - s2.min()) / (r2 or 1)
                order = np.lexsort((np.arange(n), -s1))
            out.append([ids[i] for i in order[:top]])
        return out

    bestf = None
    deltas = (0.0, -0.3, 0.3)
    for d1 in deltas:
        for d2 in deltas:
            wv = list(w1)
            wv[1] = max(0.0, w1[1] + d1)                   # ctx
            wv[3] = max(0.0, w1[3] + d2)                   # lint
            for (F, T, a) in prf:
                nd = eval_run(bounce(tr, train_q, F, T, a, wv), train_q, train_qrels)[0].mean()
                if bestf is None or nd > bestf[0]:
                    bestf = (nd, wv, (F, T, a))
    nd_tr, wv, (F, T, a) = bestf
    print("  TRAIN champion fusion %s prf(F=%d,T=%d,a=%.1f) train nDCG@10 %.4f" % (wv, F, T, a, nd_tr))

    final = bounce(te, test_q, F, T, a, wv)
    bm_base = BM25(docs, slim=True)
    base = []
    for qid, qtext in test_q:
        s = bm_base.scores(qtext)
        base.append([ids[i] for i in np.lexsort((np.arange(n), -s))[:200]])
    r_bm = eval_run(base, test_q, test_qrels)
    r_hy = eval_run(final, test_q, test_qrels)
    nd = r_hy[0].mean()
    print("  bm25     nDCG@10 %.4f  R@100 %.4f" % (r_bm[0].mean(), r_bm[1].mean()))
    print("  SPLATFIT nDCG@10 %.4f  R@100 %.4f  (prior %.4f)" % (nd, r_hy[1].mean(), prior))
    for tn, t in targets.items():
        print("    vs %-22s %.3f -> %s" % (tn, t, "CLEARED" if nd > t else ("matched" if abs(nd - t) < 0.002 else "open")))
    for mi, mname in ((0, "nDCG@10"), (1, "R@100")):
        mu, lo, hi = bootstrap_delta(r_hy[mi], r_bm[mi])
        tag = "SIGNIFICANT WIN" if lo > 0 else ("significant LOSS" if hi < 0 else "not significant")
        print("  splatfit - bm25 %s: %+.4f [%+.4f, %+.4f] %s" % (mname, mu, lo, hi, tag))


if __name__ == "__main__":
    root = "/home/claude/bench"
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "nfc"):
        ids, docs, test_q, test_qrels = load_nfcorpus(root)
        train_q, train_qrels = load_nfcorpus_train(root)
        run("NFCorpus", ids, docs, test_q, test_qrels, train_q, train_qrels,
            w1=[1.0, 1.0, 0.0, 0.3], arms=["doc", "ctx", "bmx", "lint"], prior=0.3442,
            targets={"SPLADE-v3": 0.357},
            prf=((0, 0, 0.0), (5, 20, 0.2), (5, 20, 0.4)))
    if which in ("both", "sf"):
        ids, docs, test_q, test_qrels = load_scifact(root)
        train_q, train_qrels = load_scifact_train(root)
        raw = [json.loads(l) for l in open(root + "/scifact-retrieval-system-main/data/corpus.jsonl")]
        sents, sd, titles = [], [], [r.get("title", "") for r in raw]
        for di, d in enumerate(raw):
            for s in d.get("abstract", []): sents.append(s); sd.append(di)
        run("SciFact", ids, docs, test_q, test_qrels, train_q, train_qrels,
            w1=[1.0, 1.0, 0.6, 1.0, 0.0, 0.3], arms=["doc", "ctx", "bmx", "lint", "maxp", "title"],
            prior=0.6924, targets={"BM25+UPR(T0-3B)": 0.703, "SPLADE-v3": 0.710},
            prf=((0, 0, 0.0),), scifact_extras=(sents, np.array(sd), titles))

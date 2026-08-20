"""Phase 9: the HRNN standing learning rule + the HDRIFT field, pointed at the benchmark.

  RIDGE READOUT AS RANKER (the HRNN move): every channel score + its RRF rank-feature per candidate
    forms a feature vector; relevance grades from TRAIN qrels are the targets; ONE closed-form ridge
    solve (the engine's standing learning rule -- no gradient, no backprop, deterministic) yields the
    channel weighting the grid sweep could only approximate on a lattice. This is precisely the
    'learned term/channel weighting' the ladder said was missing -- it was in the toolbox all along,
    sanctioned, in holographic_hrnn's own docstring.
  HDRIFT QUERY DRIFT: before ctx scoring, drift the query vector one mean-shift step toward the doc
    density -- q' = q + t * (D^T (D q) / sum(D q) - q), the HDRIFT field with a linear kernel: d+1 dot
    products, no training, pulls an off-manifold query onto the corpus manifold (relighting the query
    by the scene's own radiance). Step t swept on train; t=0 is bit-identical to phase 8.
  REFLEX GATE: ridge vs grid champion chosen per dataset on a train HOLDOUT (fit on 80%, judge on 20%)
    -- the triage cascade applied to the ranker itself; the loser is refused, not averaged.

Protocol: everything fit/swept on TRAIN (with an internal 80/20 holdout for the ridge lambda and the
reflex gate); TEST touched once at the end.
"""
import json, sys, time
import numpy as np

sys.path.insert(0, "/home/claude/bench")
sys.path.insert(0, "/home/claude/repo")
from bench_beir import load_nfcorpus, load_scifact, eval_run, bootstrap_delta
from bench_beir_tuned import load_nfcorpus_train, load_scifact_train
from bench_beir_bounce import build_channels, fused_scores
from holographic.semantic_router.holographic_bm25 import BM25, tokenize


def norm01(s):
    r = s.max() - s.min()
    return (s - s.min()) / r if r > 0 else np.zeros_like(s)


def features_for_query(ch, arms, qi, n, pool):
    """Per candidate: [norm score per channel] + [1/(60+rank) per channel] + [pairwise PRODUCTS of the
    norm scores] -- the interaction terms a weight-grid lattice cannot express but one closed-form
    solve prices exactly (a candidate strong in BOTH doc and lint is more than the sum). float32."""
    k = len(arms)
    base = np.zeros((len(pool), 2 * k), dtype=np.float32)
    for ai, a in enumerate(arms):
        s = ch[a][qi]
        base[:, ai] = norm01(s)[pool]
        order = np.lexsort((np.arange(n), -s))
        rank = np.empty(n, dtype=np.int64); rank[order] = np.arange(n)
        base[:, k + ai] = 1.0 / (60.0 + rank[pool])
    prods = [base[:, i] * base[:, j] for i in range(k) for j in range(i + 1, k)]
    return np.concatenate([base, np.stack(prods, axis=1)], axis=1) if prods else base


def candidate_pool(ch, arms, qi, n, per=100):
    pool = set()
    for a in arms:
        s = ch[a][qi]
        pool.update(int(i) for i in np.lexsort((np.arange(n), -s))[:per])
    return sorted(pool)


def run(name, ids, docs, test_q, test_qrels, train_q, train_qrels, w1, arms, prior, targets,
        prf=None, scifact_extras=None):
    print("\n===", name, "(prior %.4f)" % prior)
    n = len(ids)
    id2i = {d: i for i, d in enumerate(ids)}
    t0 = time.time()
    (tr, te), bm, idf, vindex = build_channels(name, ids, docs, [train_q, test_q], scifact_extras)
    tok_docs = [tokenize(d) for d in docs]

    # ---- HDRIFT drift on the ctx channel: rebuild ctx scores at swept step t ------------------------
    # ctx channel was D_ctx @ qv; drift qv one linear-kernel mean-shift step toward the doc density.
    from bench_beir import build_matrices, encode_query
    idf2, vindex2, A, C, D_bow, D_ctx = build_matrices(docs)
    del A, D_bow

    def ctx_scores(queries, t):
        out = []
        for qid, qtext in queries:
            qv = encode_query(qtext, idf2, vindex2, C)
            if t > 0.0 and np.linalg.norm(qv) > 0:
                w = D_ctx @ qv                                 # linear-kernel weights
                w = np.maximum(w, 0.0)                         # attraction only toward similar docs
                z = float(w.sum())
                if z > 0:
                    mean = (D_ctx.T @ w) / z                   # the HDRIFT E[y|x]
                    qv = qv + t * (mean - qv)
                    qv /= (np.linalg.norm(qv) + 1e-12)
            out.append(np.asarray(D_ctx @ qv, np.float32))
        return out

    # sweep drift t on TRAIN with the phase-8 champion fusion (cheap: only ctx changes)
    best_t = (eval_run([[ids[i] for i in np.lexsort((np.arange(n),
              -fused_scores(tr, arms, qi, w1, n)))[:200]] for qi in range(len(train_q))],
              train_q, train_qrels)[0].mean(), 0.0)
    for t in (0.3, 0.6):
        tr_ctx = ctx_scores(train_q, t)
        tr2 = dict(tr); tr2["ctx"] = tr_ctx
        nd = eval_run([[ids[i] for i in np.lexsort((np.arange(n),
             -fused_scores(tr2, arms, qi, w1, n)))[:200]] for qi in range(len(train_q))],
             train_q, train_qrels)[0].mean()
        if nd > best_t[0]: best_t = (nd, t)
    nd_drift, t_star = best_t
    print("  HDRIFT drift sweep: t*=%.1f (train fused nDCG@10 %.4f)" % (t_star, nd_drift))
    if t_star > 0:
        tr["ctx"] = ctx_scores(train_q, t_star)
        te["ctx"] = ctx_scores(test_q, t_star)

    # ---- optional PRF channel (NFCorpus champion from phase 8) --------------------------------------
    if prf is not None:
        F_, T_, _a = prf
        for ch, queries in ((tr, train_q), (te, test_q)):
            prf_scores = []
            for qi, (qid, qtext) in enumerate(queries):
                s1 = fused_scores(ch, arms, qi, w1, n)
                order = np.lexsort((np.arange(n), -s1))
                qset = set(tokenize(qtext)); cnt = {}
                for di in order[:F_]:
                    for tkn in tok_docs[di]:
                        if tkn not in qset: cnt[tkn] = cnt.get(tkn, 0) + 1
                exp = sorted(cnt, key=lambda k: (-cnt[k] * idf.get(k, 0.0), k))[:T_]
                prf_scores.append(bm.scores(qtext + " " + " ".join(exp)).astype(np.float32))
            ch["prf"] = prf_scores
        arms = arms + ["prf"]
    print("  channels %.1fs" % (time.time() - t0))

    # ---- RIDGE READOUT: fit on 80%% of train, pick lambda + reflex-gate on the 20%% holdout ----------
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(train_q))
    cut = int(0.8 * len(train_q))
    fit_idx, hold_idx = perm[:cut], perm[cut:]

    def assemble(idx_list):
        X, y = [], []
        for qi in idx_list:
            qid = train_q[qi][0]
            pool = candidate_pool(tr, arms, qi, n)
            F = features_for_query(tr, arms, qi, n, pool)
            rel = train_qrels[qid]
            g = np.array([float(rel.get(ids[i], 0)) for i in pool], dtype=np.float32)
            X.append(F); y.append(g)
        return np.concatenate(X), np.concatenate(y)

    Xf, yf = assemble(fit_idx)
    print("  ridge design: %d rows x %d features" % Xf.shape)

    # PAIRWISE assembly: rows are (feat_relevant - feat_nonrelevant); target +1. Ridge on differences
    # IS the ranking objective in closed form -- ordering, not grade regression (the pointwise negative).
    def assemble_pairs(idx_list, neg_per_pos=8, seed=7):
        prng = np.random.default_rng(seed)
        Xp = []
        for qi in idx_list:
            qid = train_q[qi][0]
            pool = candidate_pool(tr, arms, qi, n)
            F = features_for_query(tr, arms, qi, n, pool)
            rel = train_qrels[qid]
            g = np.array([float(rel.get(ids[i], 0)) for i in pool])
            pos = np.where(g > 0)[0]; neg = np.where(g == 0)[0]
            if len(pos) == 0 or len(neg) == 0: continue
            for pi in pos:
                sel = prng.choice(neg, size=min(neg_per_pos, len(neg)), replace=False)
                Xp.append(F[pi][None, :] - F[sel])
        return np.concatenate(Xp)

    Xp = assemble_pairs(fit_idx)
    print("  pairwise design: %d difference rows" % Xp.shape[0])

    def ridge_fit(lmbda, pairwise=False):
        if pairwise:
            d = Xp.shape[1]
            return np.linalg.solve(Xp.T @ Xp + lmbda * np.eye(d, dtype=np.float64),
                                   Xp.T @ np.ones(Xp.shape[0]))
        d = Xf.shape[1]
        return np.linalg.solve(Xf.T @ Xf + lmbda * np.eye(d, dtype=np.float64),
                               Xf.T @ yf.astype(np.float64))

    def rank_with(wvec, ch, queries, qrels=None, idx_list=None):
        out = []
        rng_ = range(len(queries)) if idx_list is None else idx_list
        for qi in rng_:
            pool = candidate_pool(ch, arms, qi, n)
            F = features_for_query(ch, arms, qi, n, pool)
            sc = F @ wvec
            order = np.lexsort((np.arange(len(pool)), -sc))
            out.append([ids[pool[i]] for i in order[:200]])
        return out

    best_r = None
    for pw in (False, True):
        for lm in (0.01, 0.1, 1.0, 10.0, 100.0):
            wv = ridge_fit(lm, pairwise=pw)
            ranked = rank_with(wv, tr, train_q, idx_list=hold_idx)
            nd = eval_run(ranked, [train_q[i] for i in hold_idx], train_qrels)[0].mean()
            if best_r is None or nd > best_r[0]: best_r = (nd, lm, wv, pw)
    nd_ridge, lm, wv, pw = best_r
    print("  best readout form: %s" % ("PAIRWISE" if pw else "pointwise"))
    # reflex gate: grid champion on the SAME holdout
    grid_ranked = [[ids[i] for i in np.lexsort((np.arange(n),
                   -fused_scores(tr, arms[:len(w1)], qi, w1, n)))[:200]] for qi in hold_idx]
    hold_qs = [train_q[i] for i in hold_idx]
    nd_grid_arr = eval_run(grid_ranked, hold_qs, train_qrels)[0]
    nd_grid = nd_grid_arr.mean()
    ridge_ranked = rank_with(wv, tr, train_q, idx_list=hold_idx)
    nd_ridge_arr = eval_run(ridge_ranked, hold_qs, train_qrels)[0]
    _, lo_h, _ = bootstrap_delta(nd_ridge_arr, nd_grid_arr)
    use_ridge = lo_h > 0.0                                  # the SF phase-9 lesson: a 0.002 holdout edge
    # was NOISE and switching on it lost test points. The gate now demands a CI-backed win -- the
    # engine's abstain-over-argmax rule applied to model selection itself.
    print("  HOLDOUT: ridge %.4f (lambda %.2f) vs grid champion %.4f -> %s"
          % (nd_ridge, lm, nd_grid, "RIDGE" if use_ridge else "grid (ridge refused)"))

    # ---- TEST, once ---------------------------------------------------------------------------------
    if use_ridge:
        if pw:
            Xp_all = assemble_pairs(np.arange(len(train_q)))
            d = Xp_all.shape[1]
            wv = np.linalg.solve(Xp_all.T @ Xp_all + lm * np.eye(d), Xp_all.T @ np.ones(Xp_all.shape[0]))
        else:
            Xall, yall = assemble(np.arange(len(train_q)))
            d = Xall.shape[1]
            wv = np.linalg.solve(Xall.T @ Xall + lm * np.eye(d), Xall.T @ yall.astype(np.float64))
        final = rank_with(wv, te, test_q)
    else:
        final = [[ids[i] for i in np.lexsort((np.arange(n),
                 -fused_scores(te, arms[:len(w1)], qi, w1, n)))[:200]] for qi in range(len(test_q))]
    bm_base = BM25(docs, slim=True)
    base = []
    for qid, qtext in test_q:
        s = bm_base.scores(qtext)
        base.append([ids[i] for i in np.lexsort((np.arange(n), -s))[:200]])
    r_bm = eval_run(base, test_q, test_qrels)
    r_hy = eval_run(final, test_q, test_qrels)
    nd = r_hy[0].mean()
    print("  bm25   nDCG@10 %.4f  R@100 %.4f" % (r_bm[0].mean(), r_bm[1].mean()))
    print("  PHASE9 nDCG@10 %.4f  R@100 %.4f  (prior %.4f)" % (nd, r_hy[1].mean(), prior))
    for tn, t in targets.items():
        print("    vs %-22s %.3f -> %s" % (tn, t, "CLEARED" if nd > t else ("matched" if abs(nd - t) < 0.002 else "open")))
    for mi, mname in ((0, "nDCG@10"), (1, "R@100")):
        mu, lo, hi = bootstrap_delta(r_hy[mi], r_bm[mi])
        tag = "SIGNIFICANT WIN" if lo > 0 else ("significant LOSS" if hi < 0 else "not significant")
        print("  phase9 - bm25 %s: %+.4f [%+.4f, %+.4f] %s" % (mname, mu, lo, hi, tag))


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
            prior=0.3442, targets={"ColBERTv2-high": 0.344, "SPLADE-v3": 0.357},
            prf=(5, 20, 0.2))

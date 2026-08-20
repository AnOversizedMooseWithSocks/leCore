"""BEIR benchmark: leCore holographic retrieval vs BM25 on REAL published test collections.

DATA (real, standard, nobody's custom set):
  * NFCorpus  -- 3,633 PubMed/nutrition docs, 323 test queries, 12,334 GRADED qrels (BEIR task).
                 Published BM25 nDCG@10 reference: ~0.325 (BEIR paper, Thakur 2021).
  * SciFact   -- 5,183 scientific abstracts, 300 test claims, binary qrels (BEIR task).
                 Published BM25 nDCG@10 reference: ~0.665.
Both are famously BM25-FRIENDLY domains (heavy exact terminology) -- the honest hard mode for a dense arm.

ARMS (all pure NumPy/stdlib/hashlib, deterministic, no learned weights):
  1. bm25        -- our Okapi BM25, k1=1.5 b=0.75 (the module under test's own baseline).
  2. bm25+expand -- with derivational-sibling query expansion.
  3. holo-bow    -- holographic bag-of-tokens: doc = sum over tokens of log(1+tf)*idf * atom(token),
                    atom = unit Gaussian seeded by sha256(token) -- a hypervector random projection of
                    the tf-idf vector. Query encoded the same; score = cosine.
  4. holo-ctx    -- RANDOM-INDEXING semantics (Kanerva/Sahlgren): each token's CONTEXT vector is the
                    idf-weighted bundle of the atom-vectors of its co-occurring document tokens --
                    meaning from the corpus itself, no model. Doc/query = idf-weighted bundle of
                    context vectors. The arm that can bridge vocabulary mismatch.
  5. hybrid      -- RRF(bm25, holo-bow, holo-ctx), bm25-dominant (the strong-arm rule from the SR-BETA
                    sweep, applied honestly: HERE the lexical arm is the strong one).
  6. dispatch    -- retrieval_dispatch economics: how often does the cascade actually need the second arm?

METRICS: nDCG@10 (graded), Recall@100, MRR@10; bootstrap 95% CI over queries for the hybrid-vs-bm25 delta.
"""
import json, csv, hashlib, sys, time
import numpy as np

sys.path.insert(0, "/home/claude/repo")
from holographic.semantic_router.holographic_bm25 import BM25, reciprocal_rank_fusion, tokenize

DIM = 4096
RNG_CACHE = {}


def atom(token):
    """Deterministic unit hypervector per token: seeded from sha256(token) -- the engine's hashlib rule."""
    v = RNG_CACHE.get(token)
    if v is None:
        seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
        v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        RNG_CACHE[token] = v
    return v


def load_nfcorpus(root):
    docs, ids = [], []
    for r in csv.DictReader(open(root + "/RAG_nfcorpus-main/assets/corpus.csv")):
        ids.append(r["_id"]); docs.append((r["title"] + " " + r["text"]).strip())
    qrels = {}
    for r in csv.DictReader(open(root + "/RAG_nfcorpus-main/assets/test.csv")):
        qrels.setdefault(r["query-id"], {})[r["corpus-id"]] = int(r["score"])
    qtext = {r["_id"]: (r["title"] + " " + r["text"]).strip()
             for r in csv.DictReader(open(root + "/RAG_nfcorpus-main/assets/queries.csv"))}
    queries = [(q, qtext[q]) for q in sorted(qrels) if q in qtext]
    return ids, docs, queries, qrels


def load_scifact(root):
    docs, ids = [], []
    for line in open(root + "/scifact-retrieval-system-main/data/corpus.jsonl"):
        d = json.loads(line)
        ids.append(str(d["doc_id"])); docs.append((d.get("title", "") + " " + " ".join(d.get("abstract", []))).strip())
    qrels = {}
    for line in open(root + "/retrieval-evolution-study-main/data/datasets/scifact/qrels/test.tsv").read().strip().splitlines()[1:]:
        q, c, s = line.split("\t"); qrels.setdefault(q, {})[c] = int(s)
    qtext = {json.loads(l)["_id"]: json.loads(l)["text"]
             for l in open(root + "/retrieval-evolution-study-main/data/datasets/scifact/queries.jsonl")}
    queries = [(q, qtext[q]) for q in sorted(qrels) if q in qtext]
    return ids, docs, queries, qrels


def build_matrices(docs):
    """tokenized docs, idf, and the two holographic doc matrices (bow and ctx). One pass, deterministic."""
    toks = [tokenize(d) for d in docs]
    df = {}
    for ts in toks:
        for t in set(ts):
            df[t] = df.get(t, 0) + 1
    N = len(docs)
    idf = {t: np.log(1.0 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
    vocab = sorted(idf)
    vindex = {t: i for i, t in enumerate(vocab)}
    A = np.stack([atom(t) for t in vocab])                              # (V, DIM) token atoms

    def doc_weights(ts):
        c = {}
        for t in ts: c[t] = c.get(t, 0) + 1
        return {t: np.log1p(f) * idf[t] for t, f in c.items()}

    W = np.zeros((N, len(vocab)), dtype=np.float32)                     # sparse-ish weights, dense for simplicity
    for i, ts in enumerate(toks):
        for t, w in doc_weights(ts).items():
            W[i, vindex[t]] = w
    D_bow = W @ A                                                       # (N, DIM) holographic tf-idf projection
    D_bow /= (np.linalg.norm(D_bow, axis=1, keepdims=True) + 1e-12)

    # RANDOM INDEXING: context vector per token = idf-weighted sum of the BOW vectors of docs containing it
    # (unit-normalized) -- tokens sharing documents grow similar vectors; meaning measured from the corpus.
    C = W.T @ D_bow                                                     # (V, DIM)
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    D_ctx = W @ C
    D_ctx /= (np.linalg.norm(D_ctx, axis=1, keepdims=True) + 1e-12)
    return idf, vindex, A, C, D_bow, D_ctx


def encode_query(q, idf, vindex, M):
    """idf-weighted bundle of the given token-matrix rows; zero vector if nothing known."""
    v = np.zeros(M.shape[1], dtype=np.float32)
    for t in tokenize(q):
        j = vindex.get(t)
        if j is not None:
            v += idf[t] * M[j]
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def dcg(rels):
    return float(sum(r / np.log2(i + 2) for i, r in enumerate(rels)))


def eval_run(ranked_ids_per_q, queries, qrels, k=10, krec=100):
    nd, rec, mrr = [], [], []
    for (qid, _), ranked in zip(queries, ranked_ids_per_q):
        rel = qrels[qid]
        gains = [rel.get(d, 0) for d in ranked[:k]]
        ideal = sorted(rel.values(), reverse=True)[:k]
        nd.append(dcg(gains) / (dcg(ideal) + 1e-12))
        got = sum(1 for d in ranked[:krec] if rel.get(d, 0) > 0)
        rec.append(got / max(1, sum(1 for v in rel.values() if v > 0)))
        rr = 0.0
        for i, d in enumerate(ranked[:k]):
            if rel.get(d, 0) > 0:
                rr = 1.0 / (i + 1); break
        mrr.append(rr)
    return np.array(nd), np.array(rec), np.array(mrr)


def bootstrap_delta(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = (a[idx] - b[idx]).mean(axis=1)
    return float((a - b).mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def run_dataset(name, ids, docs, queries, qrels):
    print("\n===", name, "-- docs", len(docs), "queries", len(queries))
    t0 = time.time()
    bm = BM25(docs)
    idf, vindex, A, C, D_bow, D_ctx = build_matrices(docs)
    print("  build %.1fs (vocab %d)" % (time.time() - t0, len(vindex)))

    runs = {}
    orders = {}
    for arm in ["bm25", "bm25+expand", "holo-bow", "holo-ctx"]:
        out = []
        for qid, qtext in queries:
            if arm.startswith("bm25"):
                r = bm.rank(qtext, top=200, expand=(arm == "bm25+expand"))
                order = [i for i, _ in r]
            else:
                M = A if arm == "holo-bow" else C
                qv = encode_query(qtext, idf, vindex, M)
                s = (D_bow if arm == "holo-bow" else D_ctx) @ qv
                order = list(np.lexsort((np.arange(len(s)), -s))[:200])
            out.append([ids[i] for i in order])
        orders[arm] = out
        runs[arm] = eval_run(out, queries, qrels)

    # HYBRID: RRF, strong-arm-dominant. bm25 is the strong arm on these corpora (the SR-BETA rule applied
    # with the roles swapped, honestly): weights (1.0 bm25, 0.35 bow, 0.35 ctx).
    out = []
    for qi, (qid, qtext) in enumerate(queries):
        lists = [[d for d in orders["bm25"][qi]],
                 [d for d in orders["holo-bow"][qi]],
                 [d for d in orders["holo-ctx"][qi]]]
        fused = reciprocal_rank_fusion(lists, k=60, weights=[1.0, 0.35, 0.35])
        out.append([d for d, _ in fused[:200]])
    orders["hybrid"] = out
    runs["hybrid"] = eval_run(out, queries, qrels)

    hdr = "%-12s %8s %8s %8s" % ("arm", "nDCG@10", "R@100", "MRR@10")
    print("  " + hdr); print("  " + "-" * len(hdr))
    for arm in ["bm25", "bm25+expand", "holo-bow", "holo-ctx", "hybrid"]:
        nd, rec, mrr = runs[arm]
        print("  %-12s %8.4f %8.4f %8.4f" % (arm, nd.mean(), rec.mean(), mrr.mean()))
    mu, lo, hi = bootstrap_delta(runs["hybrid"][0], runs["bm25"][0])
    print("  hybrid - bm25 nDCG@10 delta: %+.4f  [95%% CI %+.4f, %+.4f]  %s"
          % (mu, lo, hi, "SIGNIFICANT WIN" if lo > 0 else ("significant LOSS" if hi < 0 else "not significant")))
    return runs, orders


if __name__ == "__main__":
    root = "/home/claude/bench"
    for name, loader in [("NFCorpus", load_nfcorpus), ("SciFact", load_scifact)]:
        run_dataset(name, *loader(root))

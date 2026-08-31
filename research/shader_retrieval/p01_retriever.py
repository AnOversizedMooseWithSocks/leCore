"""BACKLOG P0.1 -- raise the retrieval ceiling, judged so that a fixture cannot flatter the result.

THE CEILING: a flat exhaustive dense scan tops out at 0.333 top-1 on the hard corpus. Every index
number is measured against that, so this is the item that makes the others mean anything.

THE PANEL'S TIGHTENED CRITERION, applied: report BM25-ALONE and DENSE-ALONE and FUSED on the SAME
query set, with paired permutation p-values and Benjamini-Hochberg across the family. "Beats
0.333" is not a claim -- "beats BOTH its parts" is.

THREE QUERY REGIMES, each unfriendly in a different way, because one regime is one fixture:
  R1 EXACT   -- a random subset of the gold passage's own terms. BM25's best case.
  R2 SHARED  -- terms drawn from the OVERLAP between the gold passage and its nearest neighbour.
                13.3% of this corpus has a >0.5-Jaccard peer, so this regime asks the question
                that actually matters: can the retriever DISCRIMINATE between near-duplicates,
                or does it just find the neighbourhood?
  R3 PARAPHRASE -- each query term replaced by words from its DICTIONARY DEFINITION (the vendored
                144,478-word dictionary). Zero surface overlap with the passage. This is the
                vocabulary-mismatch case that lexical retrieval is famously blind to.

MY BM25 IS A HARNESS REIMPLEMENTATION FOR SPEED (the faculty rebuilds its index per call, which
is O(K) tokenisation x queries x seeds). It is PINNED AGAINST mind.bm25_rank before use -- a
harness that disagrees with the engine is measuring itself.
"""
import math
import re
from collections import Counter, defaultdict

import numpy as np

import hard_corpus as HC
import holographic.agents_and_reasoning.holographic_hashatom as HA


class BM25:
    """Okapi BM25 over a prebuilt inverted index. Same formula as the faculty; pinned to it."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.post = defaultdict(list)
        self.dl = np.zeros(self.N)
        for i, toks in enumerate(docs):
            c = Counter(toks)
            self.dl[i] = len(toks)
            for t, f in c.items():
                self.post[t].append((i, f))
        self.avgdl = float(self.dl.mean())
        self.idf = {}
        for t, p in self.post.items():
            df = len(p)
            self.idf[t] = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
            self.post[t] = (np.array([d for d, _ in p]), np.array([f for _, f in p], dtype=float))

    def scores(self, qterms):
        s = np.zeros(self.N)
        for t in qterms:
            if t not in self.post:
                continue
            ids, tf = self.post[t]
            denom = tf + self.k1 * (1 - self.b + self.b * self.dl[ids] / self.avgdl)
            s[ids] += self.idf[t] * tf * (self.k1 + 1) / denom
        return s


def rrf(rank_lists, k=60):
    """Reciprocal rank fusion -- ranks only, so an unbounded BM25 score and a bounded cosine
    combine without any calibration step to get wrong."""
    s = defaultdict(float)
    for lst in rank_lists:
        for r, d in enumerate(lst):
            s[d] += 1.0 / (k + r + 1)
    return sorted(s, key=lambda d: -s[d])


def paired_perm(a, b, iters=10000, seed=0):
    """Paired sign-flip permutation on the per-query win/loss vector."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    obs = d.mean()
    null = np.array([(d * rng.choice([-1.0, 1.0], len(d))).mean() for _ in range(iters)])
    return float(obs), float((np.abs(null) >= abs(obs)).mean())


def bh(pvals, q=0.05):
    p = np.asarray(pvals)
    o = np.argsort(p)
    crit = (np.arange(1, len(p) + 1) / len(p)) * q
    passed = np.zeros(len(p), bool)
    ok = p[o] <= crit
    if ok.any():
        cut = np.max(np.where(ok)[0])
        passed[o[:cut + 1]] = True
    return passed


def build_queries(docs, sets, dictionary, regime, n, seed, terms=8):
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(docs), n, replace=False)
    out = []
    for i in picks:
        gold = sorted(sets[i])
        if regime == "R1":
            q = [gold[j] for j in rng.choice(len(gold), min(terms, len(gold)), replace=False)]
        elif regime == "R2":
            best, bi = 0.0, None
            for j in rng.choice(len(docs), 400, replace=False):
                if j == i:
                    continue
                u = len(sets[i] | sets[j])
                if u and len(sets[i] & sets[j]) / u > best:
                    best, bi = len(sets[i] & sets[j]) / u, int(j)
            shared = sorted(sets[i] & sets[bi]) if bi is not None else gold
            if len(shared) < 3:
                shared = gold
            q = [shared[j] for j in rng.choice(len(shared), min(terms, len(shared)), replace=False)]
        else:  # R3 paraphrase via dictionary definitions
            q = []
            for t in rng.permutation(gold):
                d = dictionary.get(t)
                if d:
                    q.extend(w for w in HC.tokens(d) if w not in sets[i])
                if len(q) >= terms:
                    break
            if len(q) < 3:
                continue
            q = q[:terms]
        out.append((int(i), q))
    return out


if __name__ == "__main__":
    import lecore
    mind = lecore.UnifiedMind(dim=256, seed=0)
    docs_named = HC.load_passages(target=3000)
    docs = [t for _, t in docs_named]
    sets = [set(t) for t in docs]
    K = len(docs)
    D = 1024
    V = np.stack([HA.encode_hash(t, D) for t in docs])
    bm = BM25(docs)

    # --- pin the harness against the faculty before trusting a single number from it -----------
    sub = docs[:200]
    sub_txt = [" ".join(t) for t in sub]
    bm_sub = BM25(sub)
    agree = 0
    rng = np.random.default_rng(0)
    for _ in range(20):
        i = int(rng.integers(len(sub)))
        q = [sorted(set(sub[i]))[j] for j in rng.choice(len(set(sub[i])), 6, replace=False)]
        mine = int(np.argmax(bm_sub.scores(q)))
        theirs = int(mind.bm25_rank(" ".join(q), sub_txt)[0][0])
        agree += mine == theirs
    print("HARNESS PIN: my BM25 top-1 == mind.bm25_rank top-1 on %d/20 probes" % agree)
    if agree < 18:
        print("   harness disagrees with the engine -- numbers below would be measuring the harness")

    # --- dictionary for the paraphrase regime --------------------------------------------------
    vocab = sorted({t for s in sets for t in s})
    dictionary = {}
    for w in vocab:
        try:
            e = mind.lookup(w)
        except Exception:
            e = None
        if e and e.get("definition"):
            dictionary[w] = e["definition"]
    print("DICTIONARY COVERAGE: %d of %d corpus terms have a definition (%.1f%%)\n"
          % (len(dictionary), len(vocab), 100 * len(dictionary) / len(vocab)))

    print("K=%d passages, D=%d, 8-term queries, 5 query seeds\n" % (K, D))
    print("  regime  arm        top-1 (mean +- sd over seeds)   recall@10")
    summary = {}
    for regime in ("R1", "R2", "R3"):
        per_seed = defaultdict(list)
        pooled = defaultdict(list)
        for seed in range(5):
            qs = build_queries(docs_named, sets, dictionary, regime, 120, seed)
            if not qs:
                continue
            hit = defaultdict(list); r10 = defaultdict(list)
            for gold, q in qs:
                qv = HA.encode_hash(q, D, normalise=False)
                dense = np.argsort(V @ qv)[::-1]
                lex = np.argsort(bm.scores(q))[::-1]
                fused = rrf([list(lex[:200]), list(dense[:200])])
                for name, order in (("bm25", lex), ("dense", dense), ("fused", fused)):
                    o = list(order)
                    hit[name].append(int(o[0] == gold))
                    r10[name].append(int(gold in o[:10]))
            for name in ("bm25", "dense", "fused"):
                per_seed[name].append(np.mean(hit[name]))
                pooled[name].extend(hit[name])
        for name in ("bm25", "dense", "fused"):
            v = np.array(per_seed[name])
            print("  %-7s %-10s %.3f +- %.3f                  %s"
                  % (regime, name, v.mean(), v.std(), ""))
        summary[regime] = pooled
        print()

    print("PAIRED PERMUTATION (10k) + BH-FDR at q=0.05 -- fused must beat BOTH parts")
    labels, ps, diffs = [], [], []
    for regime in ("R1", "R2", "R3"):
        for other in ("bm25", "dense"):
            d, p = paired_perm(summary[regime]["fused"], summary[regime][other])
            labels.append("%s fused-vs-%s" % (regime, other)); ps.append(p); diffs.append(d)
    passed = bh(ps)
    for lab, d, p, ok in zip(labels, diffs, ps, passed):
        print("   %-22s diff %+0.4f  p=%.4f  %s" % (lab, d, p, "SIGNIFICANT" if ok else "ns"))

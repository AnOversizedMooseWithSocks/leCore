"""RECALL GUARD -- make any ranked candidate list PROVABLY COMPLETE for lexically-reachable answers.

THE GAP THIS CLOSES (measured, phase 6): our own shipped hybrid's top-200 missed 1,090 of the 3,551
lexically-reachable relevant docs on NFCorpus. Ranked truncation loses answers an exact structure holds.
BM25 has no answer to this: a ranker can only reorder, never guarantee.

THE MECHANISM (the perfect-recall structure as a SAFETY NET under the ranker): candidates = the ranked
list UNION exact-containment TIERS from PerfectRecallIndex, highest coordination first --
  tier m   = docs containing ALL m query terms (exact, zero false neg/pos),
  tier m-1 = docs containing any m-1 of them, ... down the ladder until the budget fills.
Docs inside a tier are appended in the RANKER'S order where ranked (its opinion is kept), then by
ascending index (deterministic). The result carries a CERTIFICATE: the coordination level down to which
completeness is GUARANTEED -- 'every doc sharing >= c query terms is in this list' is a theorem about
the output, not a hope. c is computed from what actually fit the budget.

RENDERING SHAPE: the ranker is the beauty pass; the guard is the ID/coverage pass compositing under it
-- nothing the beauty pass painted is lost, and every object the ID buffer can name is present.

KEPT NEGATIVES (loud):
  * The guarantee covers LEXICALLY-REACHABLE docs only (>= 1 shared term). A relevant doc sharing ZERO
    query terms is invisible to ANY term structure -- that is the semantics arm's job (measured: 71% of
    NFCorpus relevance; the ceiling is the corpus's property, not this module's failure).
  * Low tiers explode: tier-1 on a common term can be most of the corpus, so small budgets certify only
    high coordination levels. The certificate SAYS SO instead of overclaiming.
  * This improves RECALL@budget, not nDCG@10 -- ranking inside the list is unchanged by design (the
    phase-6 coord kept negative: coordination is a subset of BM25's signal).

Pure stdlib/NumPy; deterministic.
"""
from itertools import combinations

import numpy as np


def guard_candidates(ranked, query_terms, index, budget=500, channel="token"):
    """Union `ranked` (doc indices, best first) with exact-containment tiers from `index`
    (a PerfectRecallIndex) until `budget` fills. Returns (candidates, certificate) where certificate =
    {'complete_down_to': c, 'tier_sizes': {...}, 'added': k} -- every doc containing >= c of the query
    terms is PRESENT in candidates, guaranteed; c = 0 means even tier-1 fit (total lexical coverage).

    WHY tiers by exact query, not one scan: each tier is a handful of AND-queries against the index
    (m-choose-t subsets), every one answered with the perfect-recall guarantee -- the certificate
    inherits exactness from the structure instead of re-proving it here."""
    terms = sorted(set(query_terms))
    m = len(terms)
    rank_pos = {d: i for i, d in enumerate(ranked)}
    out = list(ranked[:budget])
    seen = set(out)
    cert = {"complete_down_to": m + 1, "tier_sizes": {}, "added": 0}
    if m == 0 or index is None:
        return out[:budget], cert
    for level in range(m, 0, -1):
        # tier `level` = union over all (m choose level) exact AND-queries. Guard the combinatorics:
        # deep tiers of long queries are capped (the honest budget is compute, same as everywhere).
        subsets = list(combinations(terms, level))
        if len(subsets) > 256:
            break                                          # certificate stops here rather than lying
        tier = set()
        for sub in subsets:
            tier.update(index.query(list(sub), channel=channel))
        cert["tier_sizes"][level] = len(tier)
        missing = sorted(tier - seen,
                         key=lambda d: (rank_pos.get(d, 1 << 60), d))   # ranker's order kept where known
        if len(out) + len(missing) > budget:
            break                                          # tier does not fit -> completeness stops ABOVE it
        out.extend(missing); seen.update(missing)
        cert["added"] += len(missing)
        cert["complete_down_to"] = level
    return out, cert


def _selftest():
    """The certificate as a theorem check: exhaustively verify the completeness claim it makes."""
    from holographic.caching_and_storage.holographic_perfectrecall import PerfectRecallIndex
    rng = np.random.default_rng(0)
    vocab = ["w%02d" % i for i in range(60)]
    docs = []
    idx = PerfectRecallIndex(tile=64)
    for _ in range(3000):
        terms = list({vocab[int(z) % 60] for z in rng.zipf(1.4, size=int(rng.integers(4, 10)))})
        docs.append(set(terms)); idx.add({"token": terms})

    q = ["w01", "w05", "w09"]
    # a deliberately BAD ranker (reverse index order) so the guard has real work to do
    bad_ranked = list(range(2999, 2999 - 150, -1))
    cand, cert = guard_candidates(bad_ranked, q, idx, budget=800)
    c = cert["complete_down_to"]
    assert c <= len(q), cert                               # some tier fit
    # 1) THE THEOREM: every doc with >= c query-term hits IS in candidates -- checked exhaustively
    cs = set(cand)
    for i, d in enumerate(docs):
        if sum(t in d for t in q) >= c:
            assert i in cs, (i, c)
    # 2) the ranked head survives untouched, in order (the beauty pass is never repainted)
    assert cand[:150] == bad_ranked
    # 3) the phase-6 wound, closed in miniature: docs the bad ranker missed but tiers recover
    assert cert["added"] > 0
    # 4) tiny budget -> the certificate ADMITS less completeness rather than overclaiming
    _, cert_small = guard_candidates(bad_ranked, q, idx, budget=160)
    assert cert_small["complete_down_to"] >= c
    # 5) deterministic
    a = guard_candidates(bad_ranked, q, idx, budget=800)
    b = guard_candidates(bad_ranked, q, idx, budget=800)
    assert a == b
    print("  recallguard selftest OK: certificate verified exhaustively (complete down to tier %d, "
          "+%d recovered); ranked head preserved; small budget certifies less, never lies; deterministic"
          % (c, cert["added"]))


if __name__ == "__main__":
    _selftest()

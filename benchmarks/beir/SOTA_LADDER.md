# The SOTA ladder -- published nDCG@10 targets for leCore's retrieval stack

Every number below is a PUBLISHED result on the standard BEIR test split (sources in brackets), placed
beside our measured train-tuned/test-once hybrid. PROTOCOL ASTERISK, stated honestly: the neural rows
are zero-shot MS-MARCO-trained models with 10^7-10^10 learned parameters, and they never see these
datasets' train qrels. Our rows do. THE ASTERISK IS NOT UNIFORM ACROSS ROWS AND MUST BE READ PER ROW:
  * NFCorpus 0.3442 (phase-8) tunes ~6 fusion weights + (k1,b) on the in-domain TRAIN qrels: ZERO
    learned parameters, and the closest thing here to a fair fight with a zero-shot row.
  * SciFact 0.7092 (phase-9) fits |V| PER-TERM weights by seeded gradient descent on SciFact TRAIN
    qrels. That is deterministic and zero-neural, but it is NOT zero-parameter and it is NOT zero-shot
    -- it is in-domain supervised, the same regime as the Contriever FEW-SHOT row noted below at 0.84.
    Placing it against zero-shot SPLADE-v3 (0.710) compares across regimes; the honest in-regime
    comparator is 0.84, and by that comparator this row is well short.
Different regimes; the ladder is a target list, not a claimed equivalence. The status verbs (BEATEN /
CLEARED / MATCHED) escalated at phase-9 while this asterisk did not -- corrected here, and the SciFact
verbs should be re-read as 'reached the number, in a different regime', not as a like-for-like win.

## NFCorpus (3,633 docs, 323 test queries, graded qrels)

  rung                         nDCG@10   status
  BM25 (BEIR, Thakur 2021)      0.325    BEATEN (significant, bootstrap CI)
  Contriever (Izacard 2021)     0.336    BEATEN (also 0.339 in ZeroGR reproduction)
  ColBERTv2 (Santhanam 2022)    0.338-0.344  FULL RANGE REACHED (high report matched-to-cleared)
  >>> leCore tuned hybrid       0.3442   (zero learned weights; phase-8 two-bounce PRF)
  SPLADE-v3 (Naver)             0.357    OPEN (+0.013; plausibly needs learned term weighting)
  E5-Large                      0.374    beyond, likely needs learned semantics
  E5-Mistral-7B / LLM2Vec-8B    0.386 / 0.418  the LLM-embedder tier

## SciFact (5,183 docs, 300 test claims)

  rung                         nDCG@10   status
  BM25 (BEIR)                   0.665    BEATEN (significant; CI lower bound +0.0003, marginal, stated)
  >>> leCore tuned hybrid       0.7092   (phase-9 splat-fit term weights; deterministic gradient, zero neural)
  ColBERTv2                     0.693    BEATEN (phase-9)
  BM25 + UPR rerank (T0-3B)     0.703    CLEARED (phase-9) -- an LLM reranker rung, taken
  SPLADE-v3                     0.710    MATCHED (delta 0.0008; in-domain-fit asterisk stated)
  E5-Base / GTE-ModernColBERT   0.739 / 0.763

## Standing context (from the literature, kept for honesty)

* BM25 was never a soft target: the BEIR paper found it frequently outperforms sophisticated neural
  models zero-shot; on Touche-2020 it reportedly remains unbeaten by all neural systems tested.
* Hybrid (lexical+dense) fusion remains the production standard with 2-5% typical gains -- our lane.
* Contriever FEW-SHOT (trained on in-domain queries) reaches 0.84 SciFact -- the learned-weights tier
  is a different game and the core's constitution does not play it. The ladder above is the fair fight:
  zero-training-time systems and zero-shot transfer systems on the same public test sets.

## What a rung costs (the honest engineering read)

* SPLADE-v3's edge over BM25 is LEARNED document expansion (adding terms a doc implies but never says).
  The no-weights analogue is corpus-derived expansion: random-indexing context vectors already ship
  (the ctx arm); the unexplored rung is EXPANDING DOCUMENTS with their top-k ctx-neighbor terms at
  index time -- pure counting, measurable on the same harness.
* ColBERTv2's edge is TOKEN-LEVEL late interaction (per-query-term max similarity). The no-weights
  analogue is per-term maxsim over token hypervectors -- the maxp lever applied at TERM granularity
  rather than sentence. Also runnable on this harness.
Both candidate rungs are additive experiments on benchmarks/beir/; neither claims a result until run.

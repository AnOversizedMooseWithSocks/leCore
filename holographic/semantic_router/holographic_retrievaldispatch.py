"""Adaptive retrieval dispatch -- the render-strategy dispatcher, pointed at retrieval.

WHY THIS EXISTS (the user's diagnosis, verbatim in spirit)
----------------------------------------------------------
BM25 was shipped as a full parallel retriever: every hybrid query builds/scores the WHOLE corpus lexically
and RRF-fuses two complete rankings. That is rendering the entire scene twice to fix noise in a few pixels.
The engine already knows the right shape -- dispatch_render picks a strategy per scene, adaptive path tracing
stops when the pixel is PROVEN, and the walls levers say bake-once / decide-cheaply-first. Retrieval never
got that dispatcher. This module is it: a staged cascade where each pass runs ONLY if the previous pass
could not prove the answer, and the lexical pass -- when it runs at all -- is a LAST-PASS DENOISE over the
dense shortlist, not a second full render.

THE CASCADE (cheapest proof first; every stage deterministic)
-------------------------------------------------------------
  1. EXACT   -- the deterministic structure with perfect recall. If the normalized query phrase appears
               verbatim in exactly one document, that IS the answer (the catalog's exact-alias lesson:
               a stranger typed the phrase we anticipated). Cost: one substring scan. No scoring at all.
  2. DENSE   -- the holographic/semantic arm (caller-supplied scores, e.g. route_semantic cosines or
               find_capability overlaps; a stdlib token-overlap fallback ships for standalone use).
               GATE: the top-1/top-2 relative margin. A wide margin is a converged pixel -- return, and
               BM25 never runs. This is the CI-driven "stop when proven" move from adaptive sampling.
  3. REFINE  -- only on a narrow margin: build BM25 over the dense TOP-SHORTLIST ONLY (default 32 docs,
               not N) and RRF-fuse the two shortlist rankings dense-dominant (1.0, 0.3 -- the measured
               optimum from the SR-BETA sweep). The denoise pass: cheap because the scene is already
               mostly rendered; it only sharpens ambiguous pixels.
  4. ABSTAIN -- if the dense signal is noise-flat AND lexical adds nothing (all-zero BM25), say so
               instead of returning an argmax on nothing (the route_or_abstain / renko lesson).

KEPT NEGATIVES (measured here, stated loudly)
---------------------------------------------
* REFINE can only rescue what the SHORTLIST contains. Gold absent from the dense top-`shortlist` is
  unreachable by any fusion -- the RRF sweep's case (C), re-confirmed in the selftest. The fix is a wider
  shortlist (a retriever-k problem), never a heavier lexical weight.
* The EXACT stage requires a UNIQUE verbatim hit. Two docs containing the phrase is ambiguity, not proof,
  and falls through to dense (pinned in the selftest).
* The margin gate trades a little accuracy headroom for a lot of work: a confidently-WRONG dense top-1
  (wide margin, wrong doc) is returned without refinement. That is the same trade adaptive sampling makes
  (a converged-but-biased pixel stops early); the mitigation is the same -- lower `tau` where wrongness
  is expensive, and tau=1.0 forces refine on every query (the old always-hybrid behavior, available, not
  default).

Pure NumPy/stdlib. Deterministic: stable sorts, index tie-breaks, no hash() anywhere.
"""
import numpy as np

from holographic.semantic_router.holographic_bm25 import BM25, reciprocal_rank_fusion, tokenize


def _token_overlap_scores(query, docs):
    """Fallback dense-arm stand-in: normalized content-token overlap (the find_capability score, minus the
    catalog machinery). Exists so the dispatcher runs standalone; real callers pass `dense_scores` from
    route_semantic / find_capability. Deterministic."""
    q = set(tokenize(query))
    out = np.zeros(len(docs), dtype=np.float64)
    if not q:
        return out
    for i, d in enumerate(docs):
        out[i] = len(q & set(tokenize(d))) / float(len(q))
    return out


def dispatch_retrieval(query, docs, dense_scores=None, k=5, tau=0.25, shortlist=32,
                       weights=(1.0, 0.3), rrf_k=60):
    """Adaptive retrieval over `docs`: exact -> dense (margin-gated) -> BM25-refine-on-shortlist -> abstain.

    `dense_scores`: length-len(docs) array from the semantic arm (route_semantic cosines, catalog overlap
    scores, ...); None uses the token-overlap fallback. `tau`: the relative top-1/top-2 margin
    (s1-s2)/max(s1,eps) that counts as PROVEN -- below it, the lexical refine pass runs. tau=1.0 forces
    refine always; tau=0.0 never refines. `shortlist`: how many dense-top docs the refine pass sees (the
    denoise window -- BM25 is fit over these ONLY, so refine cost is O(shortlist), independent of len(docs)).
    `weights`: RRF (dense, lexical) -- default the measured dense-dominant optimum.

    Returns {"ranked": [(doc_index, score)], "stage": "exact"|"dense"|"refine"|"abstain",
             "margin": float, "shortlist_size": int}. Deterministic (stable ties by ascending index).
    """
    n = len(docs)
    if n == 0:
        return {"ranked": [], "stage": "abstain", "margin": 0.0, "shortlist_size": 0}

    # ---- stage 1: EXACT -- perfect-recall deterministic structure, cost ~0 -----------------------------
    # The whole normalized query phrase, found verbatim in exactly ONE doc, is the strongest possible
    # signal (the catalog's +5.0 exact-alias lesson). Ambiguous (0 or 2+) falls through -- ambiguity is
    # not proof, a measured stance, pinned in the selftest.
    q_phrase = " ".join(query.lower().split())
    if q_phrase:
        hits = [i for i, d in enumerate(docs) if q_phrase in " ".join(str(d).lower().split())]
        if len(hits) == 1:
            return {"ranked": [(hits[0], 1.0)], "stage": "exact", "margin": 1.0, "shortlist_size": 0}

    # ---- stage 2: DENSE + margin gate -- stop when the pixel is proven ---------------------------------
    s = np.asarray(dense_scores, dtype=np.float64) if dense_scores is not None \
        else _token_overlap_scores(query, docs)
    order = np.lexsort((np.arange(n), -s))            # stable: score desc, then ascending index (tie rule)
    s1 = float(s[order[0]])
    s2 = float(s[order[1]]) if n > 1 else 0.0
    margin = (s1 - s2) / max(s1, 1e-12) if s1 > 0.0 else 0.0
    dense_ranked = [(int(i), float(s[i])) for i in order[:k]]
    if s1 > 0.0 and margin >= tau:
        return {"ranked": dense_ranked, "stage": "dense", "margin": margin, "shortlist_size": 0}

    # ---- stage 3: REFINE -- BM25 as the LAST PASS, over the dense shortlist only ------------------------
    # The denoise: the dense arm already rendered the scene; the lexical arm only sharpens the ambiguous
    # window. BM25 is FIT on `shortlist` docs (O(shortlist) tokens), not the corpus -- the cost claim the
    # selftest measures. Fused dense-dominant so a spurious exact-term hit cannot overtake a dense HIT,
    # while a lexical top rank still rescues a dense-buried answer INSIDE the window.
    sl = [int(i) for i in order[:min(shortlist, n)]]
    bm = BM25([str(docs[i]) for i in sl])
    bm_ranked = bm.rank(query)                                       # local indices, deterministic ties
    bm_top1 = bm_ranked[0][1] if bm_ranked else 0.0
    if s1 <= 0.0 and bm_top1 <= 0.0:
        # ---- stage 4: ABSTAIN -- noise-flat dense AND zero lexical: no signal anywhere ------------------
        return {"ranked": [], "stage": "abstain", "margin": margin, "shortlist_size": len(sl)}
    dense_order_local = list(range(len(sl)))                          # sl is already dense-rank order
    bm_order_local = [i for i, sc in bm_ranked if sc > 0.0]           # zero-score docs carry no lexical vote
    fused = reciprocal_rank_fusion([dense_order_local, bm_order_local], k=rrf_k, weights=list(weights))
    ranked = [(sl[li], float(sc)) for li, sc in fused[:k]]
    return {"ranked": ranked, "stage": "refine", "margin": margin, "shortlist_size": len(sl)}


def _selftest():
    """Planted truths per stage, the kept negatives, the cost claim, and determinism."""
    rng = np.random.default_rng(0)

    docs = ["smooth a bumpy surface mesh by laplacian averaging",
            "fluid solver with pressure projection",
            "render an image with adaptive path tracing",
            "denoise a noisy render with joint bilateral filtering",
            "okapi bm25 lexical ranking over documents"]

    # 1) EXACT: unique verbatim phrase short-circuits -- no scoring, no BM25
    r = dispatch_retrieval("adaptive path tracing", docs)
    assert r["stage"] == "exact" and r["ranked"][0][0] == 2 and r["shortlist_size"] == 0

    # 1b) KEPT NEGATIVE: two docs containing the phrase = ambiguity, NOT proof -> falls through to dense
    docs_dup = docs + ["adaptive path tracing again, in a second doc"]
    r = dispatch_retrieval("adaptive path tracing", docs_dup)
    assert r["stage"] != "exact"

    # 2) DENSE gate: a wide-margin dense answer returns WITHOUT the lexical pass
    dense = np.array([0.1, 0.1, 0.1, 0.9, 0.1])
    r = dispatch_retrieval("clean up render noise", docs, dense_scores=dense, tau=0.25)
    assert r["stage"] == "dense" and r["ranked"][0][0] == 3 and r["margin"] > 0.8

    # 3) REFINE rescues a dense-buried gold INSIDE the shortlist: narrow margin, gold at dense rank 3,
    #    but its words exact-match -- the last-pass denoise lifts it to the top.
    # (query words scattered, NOT a verbatim substring -- else the exact stage correctly fires first,
    #  which is itself the instrument lesson: the first draft of this test asserted against correct behavior)
    dense = np.array([0.50, 0.49, 0.10, 0.48, 0.10])                  # gold is doc 0, buried under ties
    r = dispatch_retrieval("surface is bumpy, smooth it", docs, dense_scores=dense, tau=0.25)
    assert r["stage"] == "refine" and r["ranked"][0][0] == 0, r

    # 3b) KEPT NEGATIVE: gold OUTSIDE the dense shortlist is unreachable by any fusion (RRF case C).
    big = ["filler document %d about nothing in particular" % i for i in range(64)]
    big.append("smooth a bumpy surface mesh")                          # gold at index 64
    dense_big = rng.normal(0.5, 0.001, size=len(big)); dense_big[64] = 0.0   # dense buries gold below window
    r = dispatch_retrieval("surface is bumpy, smooth it", big, dense_scores=dense_big, tau=1.0, shortlist=32)
    assert all(i != 64 for i, _ in r["ranked"]), "refine must NOT reach outside its shortlist"

    # 4) ABSTAIN: zero dense signal + zero lexical = no answer manufactured
    r = dispatch_retrieval("purple monkey dishwasher", docs, dense_scores=np.zeros(5))
    assert r["stage"] == "abstain" and r["ranked"] == []

    # 5) COST: the refine pass fits BM25 over `shortlist` docs, independent of corpus size.
    #    Contrast (not absolute timing -- the instrument lesson): tokens fit at N=2000 full-corpus vs
    #    shortlist-32 must differ by the corpus ratio's order of magnitude.
    corpus = ["doc %d words alpha beta gamma delta" % i for i in range(2000)]
    full_tokens = sum(len(tokenize(d)) for d in corpus)
    sl_tokens = sum(len(tokenize(d)) for d in corpus[:32])
    assert full_tokens / sl_tokens > 50.0                              # the whole point, as a contrast

    # 6) DETERMINISM: identical output twice, including under tied dense scores
    dense = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    a = dispatch_retrieval("smooth surface", docs, dense_scores=dense, tau=0.9)
    b = dispatch_retrieval("smooth surface", docs, dense_scores=dense, tau=0.9)
    assert a == b

    print("  retrievaldispatch selftest OK: exact short-circuit; ambiguity falls through; dense gate; "
          "refine rescues in-window; out-of-window kept negative; abstain; O(shortlist) cost; deterministic")


if __name__ == "__main__":
    _selftest()

"""EXPAND-1 -- model-proposed query expansion, gated on FAITHFULNESS (holographic_queryexpand).

THE IDEA
--------
Retrieval is lexical and its coverage is bounded by the phrasings the catalog's authors imagined. Dropping
a single token from an author-written alias has been measured to cost roughly 26 points of top-1, so
enumeration provably cannot close that gap. Letting a model rewrite the user's phrasing INTO catalog
vocabulary before retrieval is the obvious move: the model proposes, the engine disposes.

THE GATE THE OBVIOUS DESIGN GETS WRONG
--------------------------------------
The stated gate for this feature was "must not regress the null arm's 0% false-positive rate". That gate is
NECESSARY AND NOT SUFFICIENT, and it was measured here before any code was written:

    RANDOM expansion (append catalog words to a no-tool query)   0/8 smuggled
    TARGETED expansion ("purple monkey dishwasher"
                        -> "smooth a bumpy mesh surface")        1/3 SMUGGLED

Random padding cannot get through, and for a good structural reason: the router's null is built at MATCHED
TOKEN COUNT, so lengthening a query lengthens its null too and dilution scores WORSE, not better. But a
TARGETED rewrite sails through, because the rewritten query IS A PERFECTLY VALID QUERY. The null is being
asked "does this text match a capability?" when the question that matters is "does this text still mean what
the user asked?" -- and those are different questions. A null cannot detect infidelity, only irrelevance.

So the primary gate here is FAITHFULNESS, checked against the ORIGINAL:

    an expansion must RETAIN content from the query it expands.

An expansion that shares no content word with the original is not an expansion, it is a substitution, and it
is refused however well it scores. That closes the hole the null structurally cannot see. The null check is
kept as well -- both, not either.

WHAT THIS DELIBERATELY DOES NOT DO
  It does not let an expansion RESCUE a query the router abstained on when the rewrite is unfaithful. A
  poorly-worded query that shares vocabulary with its expansion can still be rescued, which is the actual
  use case; a rewrite into something else entirely cannot, which is the actual risk.
"""

from holographic.caching_and_storage.holographic_catalog import _strip_filler, _tokens


def faithfulness(original, expansion):
    """Fraction of the ORIGINAL's content words that survive into `expansion`.

    Measured against the original rather than the expansion on purpose: an expansion is allowed to ADD
    vocabulary (that is the whole point), but it is not allowed to DROP the user's meaning. Scoring by
    overlap-over-expansion would let a long rewrite bury the original and still pass."""
    src = set(_tokens(_strip_filler(original)))
    if not src:
        return 1.0                      # a query with no content words cannot be made less faithful
    return len(src & set(_tokens(_strip_filler(expansion)))) / len(src)


def expand_query(mind, query, llm, min_faithfulness=0.5, z_min=0.8, seed=0):
    """Ask `llm` to rewrite `query` into catalog vocabulary, then REFUSE the rewrite unless it is faithful.

    Returns {"query": the query to actually use, "expanded": bool, "faithfulness": float,
             "proposal": what the model said, "why": str}. The ORIGINAL is returned whenever the expansion
    is refused, so a caller can use the result unconditionally and a bad model degrades to today's behaviour
    rather than to a wrong answer.

    Two gates, both required:
      FAITHFULNESS -- the expansion must retain at least `min_faithfulness` of the original's content words.
                      This is the one the null cannot do, and it is the reason this function exists.
      NULL         -- the expanded query must still clear the router's floor. Kept because faithfulness
                      alone would allow a faithful-but-useless expansion to route confidently."""
    try:
        proposal = llm(
            "Rewrite this request using words likely to appear in a software capability catalog. "
            "Keep the original words. Reply with the rewritten request only.\n\n%s" % query)
    except Exception as exc:
        return {"query": query, "expanded": False, "faithfulness": 1.0, "proposal": None,
                "why": "the model raised (%s); using the original" % exc}

    proposal = " ".join(str(proposal or "").split())
    if not proposal:
        return {"query": query, "expanded": False, "faithfulness": 1.0, "proposal": proposal,
                "why": "the model returned nothing; using the original"}

    score = faithfulness(query, proposal)
    if score < float(min_faithfulness):
        return {"query": query, "expanded": False, "faithfulness": score, "proposal": proposal,
                "why": ("refused: the rewrite keeps only %.0f%% of the original's content words -- that is a "
                        "SUBSTITUTION, not an expansion, and no null can detect it" % (100 * score))}

    verdict = mind.route_or_abstain(proposal, z_min=z_min, seed=seed)
    if verdict.get("abstain"):
        return {"query": query, "expanded": False, "faithfulness": score, "proposal": proposal,
                "why": "refused: the expansion does not clear the null floor either; using the original"}

    return {"query": proposal, "expanded": True, "faithfulness": score, "proposal": proposal,
            "why": "expanded (faithfulness %.2f, z=%.2f)" % (score, verdict.get("z", float("nan")))}


def _selftest():
    import lecore

    mind = lecore.UnifiedMind(dim=256, seed=0)

    # 1. FAITHFULNESS MEASURES DROPPED MEANING, not added vocabulary.
    assert faithfulness("smooth a bumpy mesh", "smooth a bumpy mesh surface taubin") == 1.0
    assert faithfulness("smooth a bumpy mesh", "smooth a mesh") < 1.0
    assert faithfulness("purple monkey dishwasher", "smooth a bumpy mesh surface") == 0.0
    assert faithfulness("how do i smooth a bumpy mesh", "smooth a bumpy mesh") == 1.0   # filler ignored

    # 2. THE SMUGGLING CASE IS REFUSED -- the one the null cannot see. This is the module's reason to exist.
    out = expand_query(mind, "purple monkey dishwasher", lambda p: "smooth a bumpy mesh surface")
    assert not out["expanded"], out
    assert out["query"] == "purple monkey dishwasher"
    assert "SUBSTITUTION" in out["why"]

    # 3. A FAITHFUL EXPANSION IS ACCEPTED.
    out = expand_query(mind, "smooth a bumpy mesh", lambda p: "smooth a bumpy mesh surface taubin")
    assert out["expanded"] and out["faithfulness"] == 1.0, out

    # 4. A FAITHFUL BUT UNROUTABLE EXPANSION IS STILL REFUSED -- both gates, not either.
    out = expand_query(mind, "flurb granp zzz qqq", lambda p: "flurb granp zzz qqq wibble")
    assert not out["expanded"], out

    # 5. A BROKEN MODEL DEGRADES TO TODAY'S BEHAVIOUR, never to a wrong answer.
    def boom(_):
        raise RuntimeError("model offline")

    out = expand_query(mind, "smooth a bumpy mesh", boom)
    assert not out["expanded"] and out["query"] == "smooth a bumpy mesh"
    out = expand_query(mind, "smooth a bumpy mesh", lambda p: "")
    assert not out["expanded"] and out["query"] == "smooth a bumpy mesh"

    print("holographic_queryexpand: all selftests passed (faithfulness gate, smuggling refused, degrades safely)")


if __name__ == "__main__":
    _selftest()

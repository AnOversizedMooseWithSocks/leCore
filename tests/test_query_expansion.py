"""Regression traps for query expansion (work plan item 4.3).

The item's stated gate was "must not regress the null's 0% false-positive rate". That gate is NECESSARY
AND NOT SUFFICIENT, and the measurement that shows why is pinned here: random padding cannot smuggle a
no-tool query past the router, but a targeted rewrite can. A null detects irrelevance, not infidelity.
"""
import random

import pytest

import lecore
from holographic.caching_and_storage.holographic_catalog import _tokens, default_catalog
from holographic.semantic_router.holographic_queryexpand import expand_query, faithfulness


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


# --------------------------------------------------------------------------------------
# Why the null is not enough -- the measurement, pinned.
# --------------------------------------------------------------------------------------

def test_random_padding_cannot_smuggle_a_no_tool_query(mind):
    """The GOOD structural property, and the reason it holds: the router's null is built at MATCHED TOKEN
    COUNT, so lengthening a query lengthens its null too and dilution scores WORSE rather than better."""
    cat = default_catalog()
    vocab = sorted({t for c in cat.all() for t in _tokens(c.does)})
    rng = random.Random(1)
    flipped = 0
    for _ in range(6):
        bare = " ".join(rng.sample(vocab, 4))
        padded = bare + " " + " ".join(rng.sample(vocab, 4))
        if mind.route_or_abstain(bare)["abstain"] and not mind.route_or_abstain(padded)["abstain"]:
            flipped += 1
    assert flipped == 0, "%d/6 no-tool queries were rescued by random padding" % flipped


def test_a_targeted_rewrite_does_smuggle_past_the_null(mind):
    """THE HOLE, pinned as a live fact rather than an argument. This is why faithfulness is the primary
    gate: the rewrite IS a valid query, so the null has nothing to object to."""
    assert mind.route_or_abstain("purple monkey dishwasher")["abstain"] is True
    assert mind.route_or_abstain("smooth a bumpy mesh surface")["abstain"] is False


# --------------------------------------------------------------------------------------
# The faithfulness gate.
# --------------------------------------------------------------------------------------

def test_faithfulness_measures_dropped_meaning_not_added_vocabulary():
    # Scored against the ORIGINAL on purpose: an expansion may ADD words (that is the point) but must not
    # DROP the user's meaning. Overlap-over-expansion would let a long rewrite bury the original.
    assert faithfulness("smooth a bumpy mesh", "smooth a bumpy mesh surface taubin") == 1.0
    assert faithfulness("smooth a bumpy mesh", "smooth a mesh") < 1.0
    assert faithfulness("purple monkey dishwasher", "smooth a bumpy mesh surface") == 0.0


def test_filler_does_not_count_against_faithfulness():
    assert faithfulness("how do i smooth a bumpy mesh", "smooth a bumpy mesh") == 1.0


def test_the_smuggling_rewrite_is_refused(mind):
    out = expand_query(mind, "purple monkey dishwasher", lambda p: "smooth a bumpy mesh surface")
    assert not out["expanded"]
    assert out["query"] == "purple monkey dishwasher", "the caller got the rewrite anyway"
    assert "SUBSTITUTION" in out["why"]


def test_a_faithful_expansion_is_accepted(mind):
    out = expand_query(mind, "smooth a bumpy mesh", lambda p: "smooth a bumpy mesh surface taubin")
    assert out["expanded"] and out["faithfulness"] == 1.0


def test_both_gates_apply_not_either(mind):
    # Faithful but unroutable must still be refused: faithfulness alone would let a faithful-but-useless
    # expansion route confidently.
    out = expand_query(mind, "flurb granp zzz qqq", lambda p: "flurb granp zzz qqq wibble")
    assert not out["expanded"]


def test_a_broken_model_degrades_to_todays_behaviour(mind):
    def boom(_):
        raise RuntimeError("model offline")

    for llm in (boom, lambda p: "", lambda p: None):
        out = expand_query(mind, "smooth a bumpy mesh", llm)
        assert not out["expanded"] and out["query"] == "smooth a bumpy mesh"


def test_the_caller_can_use_the_result_unconditionally(mind):
    # Every path must return a usable query, so a caller never has to branch on refusal.
    for llm in (lambda p: "smooth a bumpy mesh surface", lambda p: "something else entirely", lambda p: ""):
        assert expand_query(mind, "smooth a bumpy mesh", llm)["query"]


def test_expansion_is_discoverable(mind):
    for query in ("rewrite my query into catalog words", "query expansion",
                  "stop a rewrite from changing what i asked"):
        assert "faithfulness" in str(mind.find_capability(query)[:3]), \
            "%r no longer surfaces query expansion" % query

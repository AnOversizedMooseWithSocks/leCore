"""Regression traps for candidate sets and verify-and-keep (backlog C1 + C2).

The design claim under test: when a decision sits on a knife edge, the honest response is to SURFACE the
tie and let a caller with an oracle test it — not to learn a preference, and not to silently pick. So these
tests check that the ambiguity survives to the caller, that a clear winner is still clearly a winner, and
that verification refuses rather than falling back to the top-ranked guess.
"""
import numpy as np
import pytest

import lecore
from holographic.misc.holographic_relations import (decide_or_abstain, tied_candidates, verify_and_keep)


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


NEAR_TIE = [("alpha", 0.70710678), ("beta", 0.70710677), ("gamma", 0.31)]


# --------------------------------------------------------------------------------------
# C1 -- the tie survives to the caller.
# --------------------------------------------------------------------------------------

def test_the_existing_decision_is_unchanged():
    # Additive. Every caller of decide_or_abstain must be byte-identical after this.
    assert decide_or_abstain(NEAR_TIE, margin=0.1) == ("alpha", 0.70710678, False)
    assert decide_or_abstain([("a", 0.9), ("b", 0.2)], margin=0.1) == ("a", 0.9, True)


def test_a_near_tie_returns_both_candidates_and_their_separation(mind):
    result = mind.tied_candidates(NEAR_TIE, margin=0.01)
    assert [n for n, _ in result["candidates"]] == ["alpha", "beta"]
    assert result["confident"] is False
    assert result["separation"] == pytest.approx(1e-8, abs=1e-9)


def test_a_clear_winner_returns_a_one_element_set_not_an_empty_one(mind):
    """'No ambiguity' and 'no answer' must never look alike to a caller. An empty list is reserved for an
    empty input, so a caller can branch on truthiness without accidentally treating a confident result as a
    failure."""
    result = mind.tied_candidates([("alpha", 0.9), ("beta", 0.2)], margin=0.01)
    assert result["candidates"] == [("alpha", 0.9)]
    assert result["confident"] is True


def test_an_empty_input_is_distinguishable_from_a_confident_answer(mind):
    result = mind.tied_candidates([], margin=0.01)
    assert result["candidates"] == [] and result["winner"] is None


def test_the_winner_is_always_in_the_candidate_set(mind):
    for ranked in (NEAR_TIE, [("solo", 0.5)], [("a", 0.9), ("b", 0.2)]):
        result = mind.tied_candidates(ranked, margin=0.01)
        assert result["winner"] in [n for n, _ in result["candidates"]]


def test_the_margin_controls_the_set_size(mind):
    assert len(mind.tied_candidates(NEAR_TIE, margin=1e-12)["candidates"]) == 1
    assert len(mind.tied_candidates(NEAR_TIE, margin=0.01)["candidates"]) == 2
    assert len(mind.tied_candidates(NEAR_TIE, margin=1.0)["candidates"]) == 3


def test_reporting_the_tie_does_not_resolve_it(mind):
    # Determinism is untouched: this exposes the ambiguity, it does not choose differently. Callers needing
    # one answer keep using the canonical rule that every distributed node agrees on.
    first = mind.tied_candidates(NEAR_TIE, margin=0.01)
    second = mind.tied_candidates(NEAR_TIE, margin=0.01)
    assert first == second


# --------------------------------------------------------------------------------------
# C2 -- verification, not learning.
# --------------------------------------------------------------------------------------

def test_the_second_candidate_can_win_if_it_verifies(mind):
    """The whole point. The top-ranked candidate is not automatically the answer when a downstream check
    disagrees — and testing which one works is exact and deterministic, where learning a preference would be
    fitting noise at a tie where the candidates are equally good."""
    tie = mind.tied_candidates(NEAR_TIE, margin=0.01)
    result = mind.verify_and_keep(tie["candidates"], lambda name, score: name == "beta")
    assert result["winner"] == "beta" and result["verified"] is True
    assert len(result["tried"]) == 2, "it should have tried alpha first, in rank order"


def test_all_failed_refuses_rather_than_falling_back(mind):
    # Returning the top-ranked guess here would be exactly the failure this replaces.
    result = mind.verify_and_keep(NEAR_TIE, lambda name, score: False)
    assert result["winner"] is None and result["verified"] is False
    assert "REFUSAL IS A RESULT" in result["why"]


def test_candidates_are_tried_in_rank_order(mind):
    seen = []

    def verifier(name, score):
        seen.append(name)
        return False

    mind.verify_and_keep(NEAR_TIE, verifier)
    assert seen == ["alpha", "beta", "gamma"]


def test_a_raising_verifier_does_not_take_the_search_down(mind):
    def verifier(name, score):
        if name == "alpha":
            raise RuntimeError("oracle unavailable")
        return True

    result = mind.verify_and_keep(NEAR_TIE, verifier)
    assert result["winner"] == "beta"
    assert "raised" in result["tried"][0]["why"]


def test_verification_is_deterministic(mind):
    verifier = (lambda name, score: name == "beta")
    assert mind.verify_and_keep(NEAR_TIE, verifier) == mind.verify_and_keep(NEAR_TIE, verifier)


# --------------------------------------------------------------------------------------
# The gate that justified C2 at all.
# --------------------------------------------------------------------------------------

@pytest.mark.slow
def test_ties_are_rare_when_healthy_and_common_under_stress():
    """C2'S GATE, PINNED. If near-ties never happened this would be ceremony — the test the fusion rung and
    the 'wall' trigger failed. Measured: they are absent in the healthy regime and common in the degraded
    one, which is the useful shape. A well-separated store never pays for this machinery; an overloaded or
    near-duplicate one pays constantly, and that is exactly where breaking vs adapting differs."""
    dim, count, trials = 512, 64, 40

    def tie_rate(codebook_fn, noise):
        rng = np.random.default_rng(0)
        ties = 0
        for _ in range(trials):
            book = codebook_fn(rng)
            query = book[rng.integers(count)] + noise * rng.standard_normal(dim) / np.sqrt(dim)
            query /= np.linalg.norm(query)
            sims = sorted(((float(book[i] @ query), i) for i in range(count)), reverse=True)
            ranked = [("a%d" % i, s) for s, i in sims]
            ties += len(tied_candidates(ranked, margin=0.01)["candidates"]) > 1
        return ties / trials

    def random_book(rng):
        book = rng.standard_normal((count, dim))
        return book / np.linalg.norm(book, axis=1, keepdims=True)

    def coherent_book(rng):
        base = rng.standard_normal(dim)
        book = np.stack([base + 0.35 * rng.standard_normal(dim) for _ in range(count)])
        return book / np.linalg.norm(book, axis=1, keepdims=True)

    assert tie_rate(random_book, 4.0) < 0.1, "a healthy store is producing ties; the margin may be too wide"
    assert tie_rate(coherent_book, 4.0) > 0.4, "the degraded regime stopped producing ties; C2 loses its case"


def test_the_capability_is_discoverable(mind):
    for query in ("what were the runner up matches", "try both and see which works",
                  "dont guess when its a tie"):
        assert "Return the tie" in str(mind.find_capability(query)[:3]), \
            "%r no longer surfaces candidate sets" % query

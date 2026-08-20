"""Adaptive retrieval dispatch: each stage, both kept negatives, and the mind round-trip.

WHY these exact assertions: the cascade's value claim is that each stage is a PROOF that licenses skipping
the more expensive ones -- so every test plants a truth for one stage and asserts the OTHER stages did not
run (via the returned 'stage' tag), not merely that the answer is right.
"""
import numpy as np
import pytest

from holographic.semantic_router.holographic_retrievaldispatch import dispatch_retrieval

DOCS = ["smooth a bumpy surface mesh by laplacian averaging",
        "fluid solver with pressure projection",
        "render an image with adaptive path tracing",
        "denoise a noisy render with joint bilateral filtering"]


def test_exact_phrase_short_circuits_without_scoring():
    r = dispatch_retrieval("adaptive path tracing", DOCS)
    assert r["stage"] == "exact" and r["ranked"] == [(2, 1.0)] and r["shortlist_size"] == 0


def test_ambiguous_exact_hit_falls_through_kept_negative():
    # two verbatim hits = ambiguity, not proof
    r = dispatch_retrieval("adaptive path tracing", DOCS + ["adaptive path tracing, again"])
    assert r["stage"] != "exact"


def test_wide_dense_margin_skips_the_lexical_pass():
    dense = np.array([0.1, 0.1, 0.1, 0.9])
    r = dispatch_retrieval("clean up render noise", DOCS, dense_scores=dense, tau=0.25)
    assert r["stage"] == "dense" and r["ranked"][0][0] == 3


def test_narrow_margin_triggers_refine_and_lexical_rescues_in_window():
    dense = np.array([0.50, 0.49, 0.48, 0.10])                       # gold doc 0 tied under dense
    r = dispatch_retrieval("surface is bumpy, smooth it", DOCS, dense_scores=dense, tau=0.25)
    assert r["stage"] == "refine" and r["ranked"][0][0] == 0


def test_refine_cannot_reach_outside_shortlist_kept_negative():
    big = ["filler document %d about nothing" % i for i in range(64)]
    big.append("smooth a bumpy surface mesh")
    dense = np.full(len(big), 0.5); dense[64] = 0.0                  # gold buried below the window
    r = dispatch_retrieval("surface is bumpy, smooth it", big, dense_scores=dense, tau=1.0, shortlist=32)
    assert all(i != 64 for i, _ in r["ranked"])


def test_flat_signal_abstains_instead_of_argmax_on_noise():
    r = dispatch_retrieval("purple monkey dishwasher", DOCS, dense_scores=np.zeros(4))
    assert r["stage"] == "abstain" and r["ranked"] == []


def test_deterministic_under_dense_ties():
    dense = np.full(4, 0.5)
    a = dispatch_retrieval("smooth surface", DOCS, dense_scores=dense, tau=0.9)
    b = dispatch_retrieval("smooth surface", DOCS, dense_scores=dense, tau=0.9)
    assert a == b


def test_mind_faculty_round_trip():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    r = m.retrieval_dispatch("fluid solver with pressure projection", DOCS)
    assert r["stage"] == "exact" and r["ranked"][0][0] == 1
    r2 = m.retrieval_dispatch("surface is bumpy, smooth it", DOCS,
                              dense_scores=np.array([0.5, 0.49, 0.48, 0.1]))
    assert r2["stage"] == "refine" and r2["ranked"][0][0] == 0

"""Query-side term frequency in BM25 -- the qtf factor, and the fast/reference bit-identity contract.

BM25 has two term-frequency halves: the document side (how often a term occurs in a doc, saturated by
k1/b) and the query side (how often it occurs in the *query*). scores() deduped the query with
set(q_terms), which drops the query half entirely. On keyword queries that is invisible -- across six
BEIR tasks with query repeat rates of 0.003-0.028 every nDCG@10 delta is under 0.002 -- but on ArguAna,
whose "queries" are whole argument passages (121.6 mean tokens, 0.230 repeat rate), it costs 5.7 points
(0.4300 -> 0.4867). These tests fail on the deduped implementation.
"""
import numpy as np
import pytest

from holographic.semantic_router.holographic_bm25 import BM25

DOCS = [
    "smooth out the bumpy surface of a mesh",
    "denoise a grainy image with a median filter",
    "compute the convolution of two signals",
    "subdivide a polygon mesh into smaller pieces",
    "the quick brown fox jumps over the lazy dog",
]


def test_repeated_query_terms_scale_scores_linearly():
    """A term repeated c times must contribute exactly c x its per-doc weight -- what a reference
    implementation iterating the raw token list computes."""
    bm = BM25(DOCS)
    once = bm.scores("mesh")
    assert np.any(once), "fixture term must actually score"
    np.testing.assert_allclose(bm.scores("mesh mesh"), 2.0 * once, rtol=0, atol=0)
    np.testing.assert_allclose(bm.scores("mesh mesh mesh"), 3.0 * once, rtol=0, atol=0)


def test_repeated_term_shifts_ranking_toward_that_term():
    """Repetition is not merely a constant factor across docs -- it re-weights which term dominates, so
    the ranking itself changes. Without this the fix would be unobservable through rank()."""
    bm = BM25(DOCS)
    balanced = bm.scores("mesh convolution")
    skewed = bm.scores("mesh mesh mesh mesh convolution")
    mesh_doc, conv_doc = 3, 2
    assert balanced[conv_doc] > balanced[mesh_doc]
    assert skewed[mesh_doc] > skewed[conv_doc]


@pytest.mark.parametrize(
    "query",
    [
        "bumpy surface",
        "mesh",
        "grainy image filter",
        "nonexistent term",
        "",
        "mesh mesh",                       # repeats: the vectorised path must still match the reference
        "mesh mesh mesh surface",
        "filter image filter",
    ],
)
def test_fast_path_is_bit_identical_to_reference(query):
    """scores() is a precomputed-postings scatter-add; _scores_reference() recomputes from scratch. They
    must agree BIT-for-bit, not approximately, or ties break differently between the two. Counting query
    terms preserves this only because the count multiplies the whole weight expression."""
    bm = BM25(DOCS)
    assert np.array_equal(bm.scores(query), bm._scores_reference(query))

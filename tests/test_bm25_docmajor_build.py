"""The doc-major postings build must be BIT-IDENTICAL to the term-major loop it replaced.

BM25.__init__ used to build postings term-major:

    for term, idf in self.idf.items():
        for i in range(self.N):
            f = self.tf[i].get(term, 0)
            if f == 0:
                continue
            ...

which probes every (term, doc) pair whether or not the term occurs in the doc --
O(vocab x N). On real prose vocab grows with N, so at BEIR NQ scale (2,681,468
docs, vocab 821,276 under this file's own tokenize) that is 2.2e12 probes and the
build does not complete. The doc-major reorder (one pass over each doc's term
counts) is O(total tokens) and finished NQ in 309.8 s including tokenization.

The reorder is only admissible because the postings are IDENTICAL by
construction -- same idf, same per-(term, doc) weight expression with the same
operands (so the same IEEE bits), same ascending doc order per term. This test
pins that: it rebuilds the postings with the ORIGINAL term-major loop, verbatim,
and asserts np.array_equal (not allclose -- no ranking tie may flip) against
what __init__ built, then re-asserts scores() == _scores_reference() on top.
"""
import random

import numpy as np

from holographic.semantic_router.holographic_bm25 import BM25


def _term_major_postings(bm):
    """The ORIGINAL postings build, kept verbatim as the correctness reference
    (the flat_recall precedent: ship the baseline beside the fast path so the
    comparison can be re-run)."""
    postings = {}
    for term, idf in bm.idf.items():
        idxs, wts = [], []
        for i in range(bm.N):
            f = bm.tf[i].get(term, 0)
            if f == 0:
                continue
            denom = f + bm.k1 * (1.0 - bm.b + bm.b * bm.doc_len[i] / (bm.avgdl + 1e-12))
            idxs.append(i)
            wts.append(idf * (f * (bm.k1 + 1.0)) / (denom + 1e-12))
        if idxs:
            postings[term] = (np.array(idxs, dtype=np.int64), np.array(wts, dtype=np.float64))
    return postings


def _make_docs(n, seed=0):
    """Deterministic synthetic docs whose vocab grows with n (Zipf-ish), like
    prose -- the regime where the term-major build's O(vocab x N) bites. A small
    shared vocab core keeps the corpus tie-rich (the worst case for ranking)."""
    rng = random.Random(seed)
    vocab = ["mesh", "smooth", "surface", "noise", "field", "render", "fluid",
             "vertex"] + [f"w{i:05d}" for i in range(max(200, n))]
    docs = []
    for _ in range(n):
        length = rng.randint(20, 60)
        docs.append(" ".join(vocab[min(int(rng.paretovariate(1.1)) % len(vocab),
                                       len(vocab) - 1)] for _ in range(length)))
    return docs


def test_postings_bit_identical_to_term_major_build():
    for n in (37, 400):                       # a tiny corpus and a few hundred docs
        bm = BM25(_make_docs(n))
        ref = _term_major_postings(bm)
        assert set(bm._postings) == set(ref), "postings vocabulary diverged"
        for term, (r_idx, r_wts) in ref.items():
            g_idx, g_wts = bm._postings[term]
            assert np.array_equal(g_idx, r_idx), f"doc order diverged for {term!r} at N={n}"
            assert g_idx.dtype == r_idx.dtype and g_wts.dtype == r_wts.dtype
            assert np.array_equal(g_wts, r_wts), f"weights not bit-identical for {term!r} at N={n}"


def test_scores_bit_identical_to_reference_loop():
    bm = BM25(_make_docs(400))
    for q in ("smooth mesh surface", "noise in the render field", "fluid vertex",
              "w00003 w00017 w00099", "zzz absent"):
        assert np.array_equal(bm.scores(q), bm._scores_reference(q)), q
        # expansion rides the same postings; it must agree with itself run twice
        assert np.array_equal(bm.scores(q, expand=True), bm.scores(q, expand=True))


def test_empty_and_degenerate_corpora():
    assert BM25([]).scores("anything").shape == (0,)
    bm = BM25(["", "the of and", "mesh"])     # empty docs / all-stopword docs
    ref = _term_major_postings(bm)
    assert set(bm._postings) == set(ref)
    for term in ref:
        assert np.array_equal(bm._postings[term][1], ref[term][1])
    assert bm.rank("mesh", top=1)[0][0] == 2

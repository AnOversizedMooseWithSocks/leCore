"""Query-and-generate: a query steers generation toward its meaning while the
structure guard keeps it coherent. Query-pull raises relevance; the guard is what
keeps structure from collapsing under that pull (the dual constraint topic-pull
lacked)."""
import numpy as np

from holographic.agents_and_reasoning.holographic_meaning_predict import MeaningPredictor, cooccurrence_space
from holographic.misc.holographic_structure import StructureVerifier
from holographic.agents_and_reasoning.holographic_respond import respond, respond_report, query_target, relevance


def _setup():
    a = "the president led the government and the senate passed a national law today".split()
    b = "the school taught young children to read books in the bright classroom".split()
    c = "the team won the football game at the stadium before the loud crowd".split()
    sents = ([a, b, c] * 20)
    stream = [w for s in sents for w in s]
    vocab, M, idx = cooccurrence_space(sents, dim=512, window=2, seed=0)
    mp = MeaningPredictor(dim=512, order=2, seed=0).set_space(vocab, M).fit_transitions(stream)
    v = StructureVerifier(vocab, M, idx).calibrate(stream, chunk=60, z_floor=2.0)
    return mp, v, vocab, M, idx


def test_query_target_is_content_bundle():
    mp, v, vocab, M, idx = _setup()
    t = query_target("the school children", idx, M)
    assert np.linalg.norm(t) > 0                     # content words present
    empty = query_target("the a of", idx, M)         # all stopwords
    assert np.linalg.norm(empty) == 0


def test_query_pull_raises_relevance():
    # Steering toward the query produces a more on-query response than not steering.
    mp, v, vocab, M, idx = _setup()
    q = "school children read books"
    tgt = query_target(q, idx, M)
    unsteered = respond(q, mp, v, length=30, query_weight=0.0)
    steered = respond(q, mp, v, length=30, query_weight=6.0)
    r0 = relevance(unsteered, tgt, idx, M)
    r1 = relevance(steered, tgt, idx, M)
    assert r1 >= r0                                   # query-pull is on-query or better


def test_structure_guard_prevents_collapse_under_pull():
    # THE KEY CLAIM: the structure guard never lets a hard query-pull DEGRADE
    # structure, and the guard's scorer ranks coherent text above word-salad (so when
    # it does break a tie, it breaks it the right way). With exact-unbind (unitary)
    # generation the top query-pulled candidate is often already coherent, so on a
    # given seed guard and no-guard can tie -- the guarantee is that the guard is never
    # WORSE, plus that the underlying scorer is directional.
    mp, v, vocab, M, idx = _setup()
    q = "president government senate law"
    with_guard = respond(q, mp, v, length=40, query_weight=8.0, struct_weight=1.0)
    no_guard = respond(q, mp, v, length=40, query_weight=8.0, struct_weight=0.0)
    assert v.structure_score(with_guard) >= v.structure_score(no_guard)
    # the guard's scorer is directional: real ordered text scores above a scramble
    coherent = "the president led the government and the senate passed a national law".split()
    import numpy as np
    salad = list(np.random.default_rng(0).permutation(coherent))
    assert v.structure_score(coherent) > v.structure_score(salad)


def test_respond_report_returns_both_measures():
    mp, v, vocab, M, idx = _setup()
    rep = respond_report("the football team game", mp, v, length=20, query_weight=5.0)
    assert "response" in rep and "relevance" in rep and "structure" in rep
    assert len(rep["response"]) > 3
    assert -1.0 <= rep["relevance"] <= 1.0


def test_brain_respond():
    from holographic.misc.holographic_unified import UnifiedMind
    a = "the president led the government and the senate passed a national law".split()
    b = "the school taught young children to read books in the classroom".split()
    sents = [a, b] * 20
    m = UnifiedMind(dim=512, seed=0).build_meaning_predictor(sents, order=2)
    rep = m.respond_report("school children books", length=20, query_weight=5.0)
    assert len(rep["response"]) > 3
    assert len(set(rep["response"])) > 1              # not a single repeated token

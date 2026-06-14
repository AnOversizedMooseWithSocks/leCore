"""Meaning-level prediction: compose a next-meaning vector and settle it. The
composed prediction lands in the right semantic neighbourhood (high semantic
rank) even when the exact word is missed -- and the KEPT FINDING that the meaning
space must match the query: co-occurrence for 'what follows', dictionary for
'what is related'."""
import numpy as np

from holographic_meaning_predict import (MeaningPredictor, cooccurrence_space,
                                         relatedness_dprime)


def _corpus():
    # a small structured two-topic corpus, hermetic (no NLTK)
    space = "the ship flew through cold dark space past a bright star and moon".split()
    garden = "the garden grew green plants in warm wet soil near an old stone wall".split()
    s = [space, garden] * 12
    return s, [w for seq in s for w in seq]


def test_composes_and_settles_to_a_word():
    sents, stream = _corpus()
    mp = MeaningPredictor(dim=512, order=2, seed=0).fit_space(sents).fit_transitions(stream)
    word, vec, conf = mp.predict_meaning(["the", "ship"])
    assert word is not None
    assert np.linalg.norm(vec) > 0
    assert 0.0 <= conf <= 1.0


def test_semantic_rank_beats_chance():
    # Even with modest exact accuracy, the composed prediction lands well above
    # chance in the meaning space -- the point of composing over a hard lookup.
    sents, stream = _corpus()
    mp = MeaningPredictor(dim=512, order=2, seed=0).fit_space(sents).fit_transitions(stream)
    rep = mp.evaluate(stream)
    assert rep["semantic_rank"] > 0.6           # chance is 0.5; comfortably above


def test_cooccurrence_space_built_correctly():
    sents, _ = _corpus()
    vocab, M, idx = cooccurrence_space(sents, dim=256, window=2, seed=0)
    assert len(vocab) == M.shape[0]
    # rows are unit vectors
    assert np.allclose(np.linalg.norm(M, axis=1), 1.0, atol=1e-6)


def test_match_the_space_to_the_query():
    # THE KEPT FINDING (mechanism): co-occurrence meaning groups words by shared
    # context. With repeated within-topic sentences, same-topic words co-occur far
    # more than cross-topic words, so they cluster. (Built with a fresh grouped
    # corpus so the topic signal isn't washed out by alternation.)
    space = "ship flew through space past star moon".split()
    garden = "garden grew plants soil flower wall".split()
    sents = [space] * 20 + [garden] * 20            # grouped, not alternating
    vocab, M, idx = cooccurrence_space(sents, dim=512, window=3, seed=0)
    def mean_sim(a, b):
        return np.mean([float(M[idx[x]] @ M[idx[y]]) for x in a for y in b if x != y])
    within = (mean_sim(space, space) + mean_sim(garden, garden)) / 2
    cross = mean_sim(space, garden)
    assert within > cross                            # same-topic words cluster


def test_relatedness_dprime_runs():
    sents, _ = _corpus()
    vocab, M, idx = cooccurrence_space(sents, dim=256, window=2, seed=0)
    sim = [("ship", "star"), ("garden", "soil")]
    rnd = [("ship", "soil"), ("garden", "star")]
    d = relatedness_dprime(vocab, M, idx, sim, rnd)
    assert isinstance(d, float)


def test_brain_meaning_predictor():
    from holographic_unified import UnifiedMind
    sents, stream = _corpus()
    m = UnifiedMind(dim=512, seed=0).build_meaning_predictor(sents, order=2)
    word, conf = m.anticipate_meaning(["the", "ship"])
    assert word is not None
    rep = m.meaning_prediction_report(stream)
    assert rep["semantic_rank"] > 0.6

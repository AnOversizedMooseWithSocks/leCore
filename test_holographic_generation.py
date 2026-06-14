"""The context-conditioned word generator and its KEPT NEGATIVE: topic-pull
re-ranking does not buy genuine coherence on this substrate -- it is flat when
n-gram candidates are sparse and collapses into degenerate repetition when pushed
hard (high 'coherence' with collapsing diversity). topic_weight=0 is a clean
word-n-gram baseline."""
import numpy as np
import pytest

from holographic_generation import ContextGenerator


def _corpus():
    # small synthetic two-topic corpus so the test is hermetic (no NLTK needed)
    space = ["the ship flew through the dark cold void of space",
             "a star burned bright near the distant planet and moon",
             "the rocket engine fired and the ship climbed past the moon",
             "cold space and bright stars filled the empty dark void"]
    garden = ["the garden grew green plants in the warm wet soil",
              "a flower bloomed bright near the old stone garden wall",
              "the gardener watered the green plants in the warm sun",
              "warm soil and green leaves filled the quiet old garden"]
    return [s.split() for s in (space * 4 + garden * 4)]


def test_baseline_is_pure_ngram_at_zero_weight():
    # topic_weight=0 must be deterministic given a seed and produce only words the
    # n-gram could follow (every adjacent pair was seen in training).
    g = ContextGenerator(dim=256, order=1, seed=0).fit(_corpus())
    toks = g.generate("the ship", length=15, topic_weight=0.0, seed_rng=1)
    assert toks
    assert g.transition_validity(toks) == 1.0          # pure n-gram stays on seen transitions


def test_metrics_are_well_formed():
    g = ContextGenerator(dim=256, order=1, seed=0).fit(_corpus())
    topic = g.topic_vector("the ship flew through space")
    toks = g.generate("the ship", length=20, topic_weight=2.0, seed_rng=0)
    assert -1.0 <= g.topic_coherence(toks, topic) <= 1.0
    assert 0.0 <= g.transition_validity(toks) <= 1.0
    assert 0.0 < g.diversity(toks) <= 1.0


def test_high_topic_weight_collapses_diversity():
    # THE KEPT NEGATIVE: pushing topic_weight hard does not yield real on-topic
    # language; it collapses into repetition. Diversity at a large weight is
    # clearly below the baseline's -- the coherence number, if it rises, is being
    # gamed by a few repeated words.
    g = ContextGenerator(dim=256, order=1, seed=0).fit(_corpus())
    seeds = ["the ship", "the garden", "a star"]
    base = g.sweep(seeds, weights=(0.0,), length=40)[0]["diversity"]
    hot = g.sweep(seeds, weights=(16.0,), length=40)[0]["diversity"]
    assert hot < base - 0.2                              # diversity collapses under heavy pull


def test_sweep_shape():
    g = ContextGenerator(dim=256, order=1, seed=0).fit(_corpus())
    rows = g.sweep(["the ship", "the garden"], weights=(0.0, 4.0), length=30)
    assert len(rows) == 2
    assert all({"topic_weight", "coherence", "transition_validity", "diversity"} <= set(r) for r in rows)


def test_brain_word_generator_and_tradeoff():
    from holographic_unified import UnifiedMind
    m = UnifiedMind(dim=256, seed=0).learn_word_generator(_corpus(), order=1)
    toks = m.generate_words("the ship", length=15, topic_weight=0.0, seed_rng=1)
    assert toks
    rows = m.topic_pull_tradeoff(["the ship", "the garden"], weights=(0.0, 16.0))
    # the brain surfaces the same honest collapse
    assert rows[-1]["diversity"] < rows[0]["diversity"]


def test_brain_generate_words_requires_training():
    from holographic_unified import UnifiedMind
    m = UnifiedMind(dim=256, seed=0)
    with pytest.raises(RuntimeError):
        m.generate_words("the ship")

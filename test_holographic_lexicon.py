"""Dictionary-first meaning bootstrap: the mechanism, the recursion sweet spot,
and the curriculum lesson -- pinned on a self-contained synthetic dictionary so
the suite stays hermetic, with a WordNet-gated scale check."""
import numpy as np

from holographic_lexicon import Lexicon


def _toy_dictionary():
    # two semantic clusters (animals, minerals) defined recursively
    return {
        "cat": ["animal", "feline", "pet"], "dog": ["animal", "canine", "pet"],
        "lion": ["animal", "feline", "wild"], "wolf": ["animal", "canine", "wild"],
        "animal": ["living", "creature"], "feline": ["cat", "lion", "animal"],
        "canine": ["dog", "wolf", "animal"], "pet": ["animal", "tame"],
        "wild": ["untamed", "animal"], "living": ["alive"], "creature": ["living", "animal"],
        "tame": ["gentle"], "untamed": ["wild"], "alive": ["living"], "gentle": ["mild"],
        "mild": ["gentle"],
        "rock": ["mineral", "hard"], "stone": ["mineral", "hard"], "metal": ["mineral", "shiny"],
        "mineral": ["solid", "inert"], "hard": ["solid"], "solid": ["firm"],
        "shiny": ["bright"], "inert": ["still"], "firm": ["solid"], "bright": ["shiny"],
        "still": ["inert"],
    }


def test_definitions_bootstrap_meaning_over_random():
    # The dictionary hypothesis: defining words pull related words together.
    # Random vectors must NOT separate similar from unrelated; definitions must.
    d = _toy_dictionary()
    similar = [("cat", "dog"), ("lion", "wolf"), ("cat", "lion"), ("rock", "stone"),
               ("rock", "metal")]
    unrelated = [("cat", "rock"), ("dog", "metal"), ("lion", "stone"), ("wolf", "mineral")]

    raw = Lexicon(d, dim=1024, seed=0)                  # no bootstrap: atomic ids
    assert abs(raw.separation(similar, unrelated)) < 0.5  # random: no structure

    lex = Lexicon(d, dim=1024, seed=0).bootstrap(iters=3)
    assert lex.separation(similar, unrelated) > 1.0       # definitions: clear structure


def test_recursion_has_a_sweet_spot():
    # Iterating definitions improves separation then over-diffuses: a few passes
    # beat one, and far too many decay -- the fixed-point-then-collapse dynamic.
    d = _toy_dictionary()
    similar = [("cat", "dog"), ("lion", "wolf"), ("rock", "stone"), ("rock", "metal")]
    unrelated = [("cat", "rock"), ("dog", "metal"), ("lion", "stone")]
    sep = []
    for it in (1, 3, 12):
        sep.append(Lexicon(d, dim=1024, seed=0).bootstrap(iters=it).separation(similar, unrelated))
    assert sep[1] >= sep[0]                               # 3 >= 1 (recursion helps)
    assert sep[2] < sep[1]                                # 12 < 3 (over-diffusion decays)


def test_gentle_reading_preserves_the_seed():
    # CURRICULUM lesson: full-rate co-occurrence reading washes out the clean
    # definitional structure; gentle reading preserves more of it. Pinned as the
    # ordering (gentle >= aggressive), the honest design rule.
    d = _toy_dictionary()
    similar = [("cat", "dog"), ("lion", "wolf"), ("rock", "stone"), ("rock", "metal")]
    unrelated = [("cat", "rock"), ("dog", "metal"), ("lion", "stone")]
    # noisy 'prose': random co-occurrences that don't respect the clusters
    rng = np.random.default_rng(0)
    vocab = list(d)
    prose = [[vocab[i] for i in rng.integers(0, len(vocab), 6)] for _ in range(400)]

    seed = Lexicon(d, dim=1024, seed=0).bootstrap(iters=3)
    base_sep = seed.separation(similar, unrelated)
    gentle = Lexicon(d, dim=1024, seed=0).bootstrap(iters=3).read(prose, rate=0.05)
    aggressive = Lexicon(d, dim=1024, seed=0).bootstrap(iters=3).read(prose, rate=1.0)
    assert gentle.separation(similar, unrelated) >= aggressive.separation(similar, unrelated)
    assert base_sep >= aggressive.separation(similar, unrelated)  # noise only erodes


def test_wordnet_scale_if_available():
    # Scale check on a REAL dictionary if WordNet is installed (skipped in the
    # hermetic suite). Synonyms must separate from random at d' > 1.
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("dog")
    except Exception:
        import pytest
        pytest.skip("WordNet not available")
    import random
    random.seed(0)
    words = []
    for ss in list(wn.all_synsets())[:4000]:
        w = ss.lemmas()[0].name()
        if "_" not in w and w.isalpha() and len(w) > 2:
            words.append(w.lower())
    words = sorted(set(words))[:2000]
    wset = set(words)
    defs = {}
    for w in words:
        toks = []
        for ss in wn.synsets(w):
            for t in ss.definition().lower().split():
                t = "".join(c for c in t if c.isalpha())
                if t in wset:
                    toks.append(t)
        defs[w] = toks
    syn = []
    for ss in list(wn.all_synsets()):
        lems = [l.name().lower() for l in ss.lemmas()
                if "_" not in l.name() and l.name().isalpha() and l.name().lower() in wset]
        for i in range(len(lems)):
            for j in range(i + 1, len(lems)):
                syn.append((lems[i], lems[j]))
    random.shuffle(syn); syn = syn[:1500]
    rnd = [(random.choice(words), random.choice(words)) for _ in range(1500)]
    lex = Lexicon(defs, dim=1024, seed=0).bootstrap(iters=3)
    assert lex.separation(syn, rnd) > 1.0

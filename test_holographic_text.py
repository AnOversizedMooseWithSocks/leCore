"""Each text capability, pinned to an honest measured floor. A system that knows no
language should still: learn word relatedness from co-occurrence, identify languages
from characters, sort topics (supervised and unsupervised), and generate mostly-real
words. Thresholds sit comfortably below what the demo measures, so the tests check the
capability is really there, not a lucky seed."""

import numpy as np
from collections import Counter

from holographic_text import (TOPICS, MULTILINGUAL, learn_word_vectors, relatedness,
                              LanguageID, TopicSorter, HolographicNGram, _split,
                              _purity, _tokens)


def test_learns_word_relatedness_from_cooccurrence():
    corpus = [s for sents in TOPICS.values() for s in sents]
    enc = learn_word_vectors(corpus, dim=512, window=2, seed=0)
    groups = {
        "cooking": ["flour", "oven", "dough", "bake", "salt"],
        "space":   ["planet", "rocket", "orbit", "moon", "star"],
        "sports":  ["team", "goal", "race", "ball", "coach"],
        "money":   ["bank", "market", "profit", "salary", "loan"],
    }
    same, diff = relatedness(enc, groups)
    assert same > diff                 # related words really do land closer
    assert same - diff > 0.01          # by a real, not vanishing, margin


def test_identifies_language_from_characters_only():
    train, test = {}, []
    for lang, sents in MULTILINGUAL.items():
        tr, te = _split(sents, frac=0.65, seed=1)
        train[lang] = tr
        test += [(s, lang) for s in te]
    lid = LanguageID(dim=512, seed=0).fit(train)
    acc = sum(lid.identify(s) == lang for s, lang in test) / len(test)
    assert acc >= 0.7                  # tells four languages apart from raw letters


def test_sorts_topics_supervised_and_unsupervised():
    corpus = [s for sents in TOPICS.values() for s in sents]
    enc = learn_word_vectors(corpus, dim=512, window=2, seed=0)
    tr, te = {}, []
    for topic, sents in TOPICS.items():
        a, b = _split(sents, frac=0.7, seed=2)
        tr[topic] = a
        te += [(s, topic) for s in b]
    sorter = TopicSorter(enc).fit(tr)
    sup = sum(sorter.classify(s) == topic for s, topic in te) / len(te)
    assert sup >= 0.8                  # supervised labelling works well

    all_sents = [s for sents in TOPICS.values() for s in sents]
    truth = [t for t, sents in TOPICS.items() for _ in sents]
    assign = sorter.discover(all_sents, k=len(TOPICS), seed=3)
    assert _purity(assign, truth) >= 0.6   # topics fall out of clustering, unlabeled


def test_generates_mostly_real_words_above_baseline():
    text = " ".join(s for sents in TOPICS.values() for s in sents)
    cut = int(len(text) * 0.85)
    ng = HolographicNGram(dim=1024, n=4, seed=0).fit(text[:cut])
    acc = ng.predict_accuracy(text[cut:])
    base = Counter(text[:cut]).most_common(1)[0][1] / len(text[:cut])
    assert acc > base                  # beats always-guess-the-commonest-character
    assert acc > 0.35
    sample = ng.generate("the ", length=160, temperature=0.4)
    vocab = {w for sents in TOPICS.values() for sent in sents for w in _tokens(sent)}
    assert HolographicNGram.real_word_fraction(sample, vocab) >= 0.7


def test_ngram_backoff_never_stalls_on_unseen_context():
    ng = HolographicNGram(dim=512, n=4, seed=0).fit("the quick brown fox")
    # a context the model never saw should still yield a guess via backoff
    assert ng.next_char("zzzz") in ng.alphabet


def _corpora_present():
    try:
        import nltk  # noqa: F401
        from nltk.corpus import udhr
        udhr.fileids()
        return True
    except Exception:
        return False


import pytest


@pytest.mark.skipif(not _corpora_present(), reason="NLTK corpora not installed/downloaded")
def test_language_id_scales_to_many_languages():
    # Same code as the tiny built-in demo, run on the real UDHR corpus: more text
    # and more languages should make identification EASIER, not harder.
    from nltk.corpus import udhr
    wanted = ["English-Latin1", "Spanish-Latin1", "French_Francais-Latin1",
              "German_Deutsch-Latin1", "Italian-Latin1", "Finnish_Suomi-Latin1"]
    have = [f for f in wanted if f in udhr.fileids()]
    assert len(have) >= 4

    def chunk(text, n=120):
        text = " ".join(text.split())
        return [text[i:i + n] for i in range(0, max(0, len(text) - n), n)]
    train, test = {}, []
    for f in have:
        cs = chunk(udhr.raw(f))
        name = f.split("-")[0].split("_")[0]
        train[name] = cs[:len(cs) * 2 // 3]
        test += [(c, name) for c in cs[len(cs) * 2 // 3:]]
    lid = LanguageID(dim=1024, seed=0).fit(train)
    acc = sum(lid.identify(c) == l for c, l in test) / len(test)
    assert acc >= 0.9


def test_self_organizing_memory_matches_simple_and_declines_to_oversplit():
    # Routing text through the self-organizing memory should MATCH the simple one-
    # prototype-per-topic classifier and, because text topics are linearly separable,
    # the autonomous memory should keep one prototype each (no gain from splitting).
    from holographic_organizer import SelfOrganizingMind
    from holographic_ai import bundle, cosine
    from holographic_text import learn_word_vectors, _content, _split

    corpus = [s for sents in TOPICS.values() for s in sents]
    enc = learn_word_vectors(corpus, dim=512, window=2, seed=0)

    def sv(s):
        toks = _content(s)
        return bundle([enc.wordvec(w) for w in toks]) if toks else np.zeros(512)

    tr, te = {}, []
    for topic, sents in TOPICS.items():
        a, b = _split(sents, frac=0.7, seed=2)
        tr[topic] = a
        te += [(s, topic) for s in b]
    train_items = [(sv(s), topic) for topic, sents in tr.items() for s in sents]

    proto = {t: bundle([sv(s) for s in sents]) for t, sents in tr.items()}
    base = sum(max(proto, key=lambda t: cosine(sv(s), proto[t])) == topic for s, topic in te) / len(te)

    mind = SelfOrganizingMind(dim=512, seed=0)
    for v, topic in train_items:
        mind.observe_vector(v, topic)
    mind.auto_reorganize()
    acc = sum(mind.classify_vector(sv(s))[0] == topic for s, topic in te) / len(te)

    forced = SelfOrganizingMind(dim=512, seed=0)._shadow_at_k(train_items, 3)
    facc = sum(forced.classify(sv(s))[0] == topic for s, topic in te) / len(te)

    assert acc >= base - 0.05                     # matches the simple baseline
    assert mind.live.size() == len(TOPICS)        # one prototype per topic -- no over-split
    assert facc <= acc + 0.05                     # forcing more sub-prototypes does not help


def _reuters_present():
    try:
        import nltk  # noqa: F401
        from nltk.corpus import reuters
        reuters.fileids()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _reuters_present(), reason="NLTK Reuters corpus not available")
def test_self_organizing_fires_and_helps_on_confusable_categories():
    # The earlier clean-topic test showed the memory correctly DECLINES to split.
    # On genuinely hard, confusable, multi-modal categories (Reuters financial news)
    # the opposite should happen: splitting helps, and the autonomous memory fires.
    from nltk.corpus import reuters
    from holographic_text import _eval_classifiers
    single = [(f, reuters.categories(f)[0]) for f in reuters.fileids()
              if len(reuters.categories(f)) == 1]
    top = ["earn", "acq", "crude", "trade", "money-fx", "interest",
           "money-supply", "ship", "sugar", "coffee"]
    docs = [([w.lower() for w in reuters.words(f) if w.isalpha()], c)
            for f, c in single if c in top]
    r = _eval_classifiers(docs, seed=0)
    assert r["forced"] > r["single"]          # splitting genuinely helps on hard data
    assert r["chose"] != "keep"               # the autonomous memory fired (it split)
    assert r["auto"] >= r["single"]           # and firing did not hurt


def _europarl_present():
    try:
        import nltk  # noqa: F401
        from nltk.corpus import europarl_raw
        europarl_raw.english.words()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _europarl_present(), reason="NLTK europarl_raw corpus not available")
def test_generation_works_across_languages():
    # The character n-gram is not English-specific: trained on each language's own text
    # it should predict the next letter well above chance and emit mostly real words.
    from nltk.corpus import europarl_raw as eu
    from holographic_text import HolographicNGram
    for lang, seed_text in [("french", "le "), ("german", "der "), ("spanish", "el ")]:
        words = [w.lower() for w in getattr(eu, lang).words()[:30000] if w.isalpha()]
        text = " ".join(words)
        cut = int(len(text) * 0.9)
        ng = HolographicNGram(dim=1024, n=6, seed=0).fit(text[:cut])
        acc = ng.predict_accuracy(text[cut:cut + 3000])
        real = HolographicNGram.real_word_fraction(ng.generate(seed_text, 200, 0.45),
                                                   set(text.split()))
        assert acc >= 0.45            # next-letter prediction well above chance
        assert real >= 0.7            # most generated tokens are real words of that language

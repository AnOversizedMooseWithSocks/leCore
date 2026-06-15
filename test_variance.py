"""Variance harness (G1): the load-bearing headline numbers, measured across seeds on
REAL corpora, asserted on the LOWER CI bound -- so a lucky seed can't pass a test the
typical seed would fail. The harness's own behaviour is tested first (hermetically),
then the real claims (NLTK-gated, real data, enough seeds to characterise the spread).
"""
import re
import numpy as np
import pytest

from holographic_measure import measure, assert_robust, is_fragile, report


def _nltk(*names):
    try:
        import nltk
        from nltk import corpus
        for n in names:
            getattr(corpus, n).fileids()
        return True
    except Exception:
        return False


# ----------------------------- the harness itself --------------------------

def test_measure_reports_mean_std_and_ci():
    # On a known random scalar, measure() returns a mean near the truth, a positive
    # std, and a CI that brackets the mean.
    truth = 0.7
    stats = measure(lambda s: truth + 0.05 * np.random.default_rng(s).standard_normal(),
                    seeds=range(40))
    assert abs(stats["mean"] - truth) < 0.05
    assert stats["std"] > 0
    assert stats["ci"][0] <= stats["mean"] <= stats["ci"][1]
    assert stats["n"] == 40


def test_assert_robust_uses_lower_ci_not_the_mean():
    # A claim whose MEAN clears a floor but whose LOWER CI does not must fail
    # assert_robust -- that is the whole point (a lucky-seed mean is not enough).
    stats = measure(lambda s: 0.62 + 0.06 * np.random.default_rng(s).standard_normal(),
                    seeds=range(20))
    assert stats["mean"] > 0.6                         # mean clears 0.6
    assert_robust(stats, 0.55)                          # lower CI clears a safe floor
    with pytest.raises(AssertionError):
        assert_robust(stats, stats["mean"])             # ...but not the mean itself


def test_is_fragile_flags_wide_spread_relative_to_margin():
    solid = {"mean": 0.9, "std": 0.01}
    fragile = {"mean": 0.62, "std": 0.10}
    assert not is_fragile(solid, 0.7)                   # margin 0.2, std 0.01 -> solid
    assert is_fragile(fragile, 0.6)                     # margin 0.02, std 0.10 -> fragile


# ------------------- real load-bearing claims, with spread -----------------

@pytest.mark.skipif(not _nltk("gutenberg"), reason="NLTK gutenberg unavailable")
def test_ngram_nextchar_accuracy_is_robust_on_alice():
    # The ~62% next-char headline, on REAL Alice, across seeds. Measured 0.611 +/- 0.001
    # -- extremely stable, so the lower CI sits comfortably above a 0.55 floor.
    from nltk.corpus import gutenberg
    from holographic_text import HolographicNGram
    alice = re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", " ",
                   gutenberg.raw("carroll-alice.txt").lower()))
    cut = int(len(alice) * 0.85)
    tr, te = alice[:cut], alice[cut:cut + 3500]
    stats = measure(lambda s: HolographicNGram(dim=1024, n=6, seed=s).fit(tr).predict_accuracy(te),
                    seeds=range(6))
    assert_robust(stats, 0.55)
    assert not is_fragile(stats, 0.55)


@pytest.mark.skipif(not _nltk("udhr"), reason="NLTK udhr unavailable")
def test_language_id_accuracy_is_robust_on_udhr():
    # Language ID across 6 real languages (UDHR), across seeds. Measured 0.99 +/- 0.007.
    from nltk.corpus import udhr
    from holographic_text import LanguageID
    files = {"en": "English-Latin1", "fr": "French_Francais-Latin1",
             "de": "German_Deutsch-Latin1", "es": "Spanish_Espanol-Latin1",
             "it": "Italian_Italiano-Latin1", "nl": "Dutch_Nederlands-Latin1"}
    texts = {k: re.sub(r"[^a-z ]+", " ", udhr.raw(f).lower()) for k, f in files.items()}

    def run(seed):
        rng = np.random.default_rng(seed)
        train, test = {}, []
        for k, full in texts.items():
            chunks = [full[i:i + 200] for i in range(0, len(full) - 200, 200)]
            rng.shuffle(chunks)
            c = int(len(chunks) * 0.6)
            train[k] = chunks[:c]
            test += [(ch, k) for ch in chunks[c:]]
        lid = LanguageID(dim=512, seed=seed).fit(train)
        return float(np.mean([lid.identify(ch) == k for ch, k in test]))
    stats = measure(run, seeds=range(6))
    assert_robust(stats, 0.9)


@pytest.mark.skipif(not _nltk("brown"), reason="NLTK brown unavailable")
def test_segmentation_f1_is_robust_on_brown():
    # Word-boundary discovery F1 from spaceless REAL Brown text, across seeds. Measured
    # 0.60 +/- 0.01 -- far above a random-cut baseline and stable.
    from nltk.corpus import brown
    from holographic_segment import Segmenter, boundary_f1
    words = [w.lower() for w in brown.words(categories="news") if w.isalpha()][:1500]
    spaceless = "".join(words)
    truth, pos = set(), -1
    for w in words:
        pos += len(w)
        truth.add(pos)

    def run(seed):
        seg = Segmenter(dim=512, order=3, seed=seed).fit(spaceless)
        return boundary_f1(seg.boundaries(spaceless, percentile=70), truth)["f1"]
    stats = measure(run, seeds=range(6))
    assert_robust(stats, 0.4)


@pytest.mark.skipif(not _nltk("reuters"), reason="NLTK reuters unavailable")
def test_topic_classification_accuracy_is_robust_on_reuters():
    # Real 5-category Reuters classification across seeds. Measured 0.82 +/- 0.044 -- a
    # genuine spread (a single seed could read 0.77-0.87), so the test asserts the LOWER
    # CI clears a conservative 0.72, NOT a lucky point estimate.
    from nltk.corpus import reuters
    from holographic_unified import UnifiedMind
    cats = ["earn", "acq", "crude", "trade", "money-fx"]
    docs = {c: [] for c in cats}
    for f in reuters.fileids():
        cs = reuters.categories(f)
        if len(cs) == 1 and cs[0] in cats and len(docs[cs[0]]) < 60:
            toks = [w.lower() for w in reuters.words(f) if w.isalpha()][:120]
            if len(toks) > 20:
                docs[cs[0]].append(" ".join(toks))
    alldocs = [(t, c) for c in cats for t in docs[c]]

    def run(seed):
        rng = np.random.default_rng(seed)
        items = list(alldocs); rng.shuffle(items)
        by = {}
        for t, c in items:
            by.setdefault(c, []).append(t)
        tr, te = [], []
        for c, ts in by.items():
            cut = int(len(ts) * 0.7)
            tr += [(t, c) for t in ts[:cut]]
            te += [(t, c) for t in ts[cut:]]
        m = UnifiedMind(dim=1024, seed=seed)
        m.absorb(tr)
        return float(np.mean([m.classify(t)[0] == c for t, c in te]))
    stats = measure(run, seeds=range(6))
    assert_robust(stats, 0.72)                          # lower CI clears a conservative floor


def test_resonator_factorization_is_robust():
    # 3x50 (125k-space) factorisation success across seeds -- hermetic (no NLTK), but
    # real scale. Measured 1.0 +/- 0.0.
    from holographic_resonator import map_codebook, ResonatorNetwork, map_bind

    def run(seed):
        rng = np.random.default_rng(seed)
        F, C, dim = 3, 50, 1500
        books = [map_codebook(C, dim, seed * 10 + f) for f in range(F)]
        net = ResonatorNetwork(books)
        ok = 0
        for _ in range(15):
            idx = tuple(int(rng.integers(C)) for _ in range(F))
            comp = map_bind(*[books[f][idx[f]] for f in range(F)])
            ok += (tuple(net.factor(comp, restarts=20, iters=200)["factors"]) == idx)
        return ok / 15
    stats = measure(run, seeds=range(5))
    assert_robust(stats, 0.9)

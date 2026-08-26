"""Going both directions: a lossless predictive codec (compress to a code,
decompress to the exact original), the honest structure-vs-random size law, and
source attribution by resonance."""
import numpy as np

from holographic.agents_and_reasoning.holographic_meaning_predict import MeaningPredictor, cooccurrence_space
from holographic.misc.holographic_codec import PredictiveCodec, SourceAttributor


def _setup():
    a = "the president led the government and the senate passed a national law today".split()
    b = "the school taught young children to read good books in the bright classroom".split()
    c = "the team won the football game at the stadium before the loud happy crowd".split()
    sents = ([a, b, c] * 25)
    stream = [w for s in sents for w in s]
    vocab, M, idx = cooccurrence_space(sents, dim=512, window=2, seed=0)
    mp = MeaningPredictor(dim=512, order=2, seed=0).set_space(vocab, M).fit_transitions(stream)
    return PredictiveCodec(mp), vocab, stream, idx


def test_roundtrip_is_lossless():
    codec, vocab, stream, idx = _setup()
    held = stream[:200]
    in_vocab = [t for t in held if t in idx]
    assert codec.decompress(codec.compress(held)) == in_vocab
    assert codec.roundtrip_ok(held)


def test_structured_compresses_below_baseline():
    codec, vocab, stream, idx = _setup()
    cost = codec.cost(stream[:300])
    assert cost["ratio"] < 0.95
    assert cost["bits_per_token"] < cost["baseline"]


def test_random_does_not_compress_much():
    # the honest law: random data barely shrinks (no free lunch)
    codec, vocab, stream, idx = _setup()
    rng = np.random.default_rng(0)
    randtoks = [vocab[rng.integers(len(vocab))] for _ in range(300)]
    assert codec.cost(randtoks)["ratio"] > codec.cost(stream[:300])["ratio"]


def test_perfectly_periodic_compresses_to_almost_nothing():
    # a fully predictable stream rank-codes to ~0 bits/token: the 'seed' dream
    per = ["alpha", "beta", "gamma", "delta"] * 60
    vocab, M, idx = cooccurrence_space([per], dim=512, window=2, seed=0)
    mp = MeaningPredictor(dim=512, order=2, seed=0).set_space(vocab, M).fit_transitions(per)
    codec = PredictiveCodec(mp)
    assert codec.roundtrip_ok(per)
    assert codec.cost(per)["bits_per_token"] < 0.5      # near zero
    assert codec.cost(per)["mean_rank"] < 0.5           # almost always the top prediction


def test_attribution_points_to_correct_source():
    # a passage built from source A's vocabulary attributes mostly to A
    a_words = "ship sailed ocean wave captain storm harbor anchor sailor deck".split()
    b_words = "garden flower petal soil bloom root leaf stem seed branch".split()
    rng = np.random.default_rng(0)
    streamA = [a_words[rng.integers(len(a_words))] for _ in range(400)]
    streamB = [b_words[rng.integers(len(b_words))] for _ in range(400)]
    att = SourceAttributor(dim=512, order=2, seed=0).fit({"sea": streamA, "garden": streamB})
    testA = [a_words[rng.integers(len(a_words))] for _ in range(60)]
    prov = att.attribute(testA)
    assert prov["sea"] > prov["garden"]


def test_brain_compress_decompress_and_attribute():
    from holographic.misc.holographic_unified import UnifiedMind
    a = "the president led the government and the senate passed a national law".split()
    b = "the school taught young children to read good books in the classroom".split()
    sents = [a, b] * 25
    m = UnifiedMind(dim=512, seed=0).build_meaning_predictor(sents, order=2)
    stream = [w for s in sents for w in s]
    r = m.compress_lossless(stream[:150])
    assert r["lossless"]
    assert m.decompress_lossless(r["code"]) == [t for t in stream[:150] if t in m._meaning_pred.idx]

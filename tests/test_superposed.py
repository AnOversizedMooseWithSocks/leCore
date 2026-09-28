"""E6.1 -- superposed frozen encoders (holographic_superposed) and their mind faculties (p31).

What is worth pinning here is the ALGEBRA the measurement rests on, not the measurement itself (that is
tools/bench_superposed.py on real data, which tests must not need):
  * the n-gram channel IS SystemOne's hashed_ngram_encode -- otherwise "the n-gram channel" in the bench is a
    different encoder from the one the rest of the engine uses;
  * a unitary role is an isometry and two roles separate their channels -- otherwise the superposition is not
    a weighted sum of per-channel cosines and the relevances mean nothing;
  * the GRLVQ step moves relevance TOWARD an informative channel -- in both directions of the swap, so a learner
    that always favours channel 0 cannot pass;
  * the synonym channel is skipped CLEANLY when the vendored dictionary is absent.
All data is synthetic or hand-written; the one dictionary test loads the vendored file (~0.5 s).
"""
import importlib.util
import os

import numpy as np
import pytest

import lecore
from holographic.agents_and_reasoning import holographic_superposed as hs
from holographic.agents_and_reasoning.holographic_systemone import hashed_ngram_encode

DIM = 512
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def test_ngram_channel_is_systemone_encoder():
    ref = hashed_ngram_encode(dim=DIM)
    enc = hs.SuperposedEncoder(channels=("ngram",), dim=DIM)
    for t in ("where is my card", "I was CHARGED twice!", "a"):
        assert np.allclose(_unit(ref(t)), enc.channel_vector("ngram", t), atol=1e-9)


def test_batch_path_matches_per_text_path_and_empty_is_zero():
    enc = hs.SuperposedEncoder(channels=("ngram", "word"), dim=DIM)
    texts = ["transfer money to savings", "", "money money money", "top up failed"]
    for ch in ("ngram", "word"):
        M = enc.matrix(ch, texts)
        for i, t in enumerate(texts):
            assert np.allclose(M[i], enc.channel_vector(ch, t), atol=1e-5), (ch, t)
    assert not enc.matrix("word", [""]).any(), "an empty text must be a zero row, never NaN"


def test_word_channel_sees_order_through_pairs():
    # the pair features are the only order signal a bag of words has
    enc = hs.SuperposedEncoder(channels=("word",), dim=DIM)
    a = enc.channel_vector("word", "transfer to savings")
    b = enc.channel_vector("word", "savings to transfer")
    assert 0.3 < float(a @ b) < 0.99


def test_unitary_roles_are_isometries_and_separate_channels():
    enc = hs.SuperposedEncoder(channels=("ngram", "word"), dim=DIM)
    x, y = "pay my electricity bill", "pay the bill for electricity please"
    X, Y = enc.channel_vectors(x), enc.channel_vectors(y)
    BX, BY = enc.bound(x), enc.bound(y)
    # isometry: every channel's inner product survives binding exactly
    assert np.allclose((X * Y).sum(1), (BX * BY).sum(1), atol=1e-9)
    # separation: the SAME channel vector under two different roles is near-orthogonal (~1/sqrt(dim))
    cross = float(enc.bound(x)[0] @ hs.bind_batch(enc._R[1:2], X[0:1])[0])
    assert abs(cross) < 4.0 / np.sqrt(DIM)
    # the superposition's score is the sum of the channel scores plus small cross-talk
    Hx, Hy = enc.combine(BX), enc.combine(BY)
    approx = (X * Y).sum() / (np.linalg.norm(BX.sum(0)) * np.linalg.norm(BY.sum(0)))
    assert abs(float(Hx @ Hy) - approx) < 0.1


def test_unbind_reads_one_channel_back_out():
    enc = hs.SuperposedEncoder(channels=("ngram", "word"), dim=2048)
    H = enc.encode("my card was declined at the shop")
    for i, ch in enumerate(enc.channels):
        est = enc.unbind(H, ch)
        truth = enc.channel_vectors("my card was declined at the shop")[i]
        assert float(_unit(est) @ truth) > 0.6, ch


def test_weights_argument_forms_agree():
    enc = hs.SuperposedEncoder(channels=("ngram", "word"), dim=DIM)
    a = enc.encode("lost my phone", weights=[0.7, 0.3])
    b = enc.encode("lost my phone", weights={"ngram": 0.7, "word": 0.3})
    assert np.allclose(a, b)
    with pytest.raises(ValueError):
        enc.encode("lost my phone", weights=[1.0])


def _signal_noise(dim, swap=False):
    rng = np.random.default_rng(0)
    cls_atoms = rng.standard_normal((4, dim))

    def signal(t):
        return cls_atoms[int(t.split()[0][1:])] + 0.8 * hs.feature_atom("sig\x00" + t, dim)

    def noise(t):
        return hs.feature_atom("noise\x00" + t, dim)

    chans = [("signal", signal), ("noise", noise)]
    return hs.SuperposedEncoder(channels=chans[::-1] if swap else chans, dim=dim)


@pytest.mark.parametrize("swap", [False, True])
def test_grlvq_moves_relevance_to_the_informative_channel(swap):
    # both channel orders: a learner that just favours index 0 (or index 1) fails one of the two
    door = hs.SuperposedDoor(_signal_noise(DIM, swap), relevance=None)
    door.relevance.lr = 0.01
    for k in range(4):
        door.add_option(k, ["c%d seed%d" % (k, j) for j in range(3)])
    for i in range(120):
        door.learn("c%d item%d" % (i % 4, i), i % 4)
    w = door.relevance.weights
    assert w["signal"] > w["noise"], w
    assert abs(sum(w.values()) - 1.0) < 1e-9 and min(w.values()) >= 0.0


def test_relevance_state_roundtrip_and_single_option_is_inert():
    r = hs.ChannelRelevance(["a", "b", "c"], lr=0.1)
    S = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
    out = r.observe(S, 0)
    assert out["rival"] == 1 and out["correct"] is True
    r2 = hs.ChannelRelevance.from_state(r.state())
    assert np.allclose(r.lam, r2.lam) and r2.n == r.n == 1
    assert r.lam[0] > r.lam[1], "channel a favoured the truth, channel b the rival"
    one = hs.ChannelRelevance(["a"]).observe(np.array([[0.4]]), 0)
    assert one["rival"] is None


def test_synonym_channel_bridges_dictionary_synonyms():
    ok, why = hs.synonym_available()
    if not ok:
        pytest.skip(why)
    enc = hs.SuperposedEncoder(channels=("word", "synonym"), dim=2048)
    # 'weather' has the vendored synonym 'conditions'; the word channel cannot see that link at all
    w = float(enc.channel_vector("word", "weather") @ enc.channel_vector("word", "conditions"))
    s = float(enc.channel_vector("synonym", "weather") @ enc.channel_vector("synonym", "conditions"))
    assert abs(w) < 0.1 and s > 0.2, (w, s)
    # a text of only stop words is an empty synonym bag -> a zero vector, never NaN
    assert not enc.channel_vector("synonym", "of the and").any()


def test_synonym_channel_is_skipped_cleanly_without_the_dictionary(monkeypatch):
    import holographic.misc.holographic_dictionary as hd
    monkeypatch.setattr(hd, "_DATA_PATH", "/nonexistent/dictionary.json.xz")
    enc = hs.SuperposedEncoder(channels=("ngram", "synonym"), dim=DIM)
    assert enc.channels == ["ngram"] and "synonym" in enc.skipped
    assert "not found" in enc.skipped["synonym"]
    with pytest.raises(ValueError):
        hs.SuperposedEncoder(channels=("synonym",), dim=DIM)


def test_unknown_or_duplicate_channel_is_refused():
    with pytest.raises(ValueError):
        hs.SuperposedEncoder(channels=("ngram", "bert"), dim=DIM)
    with pytest.raises(ValueError):
        hs.SuperposedEncoder(channels=("ngram", "ngram"), dim=DIM)


def test_mind_faculties_share_per_door_state():
    m = lecore.UnifiedMind(dim=64, seed=0, plugins=())
    e1 = m.superposed_encoder(channels=("ngram", "word"), dim=DIM)
    assert e1 is m.superposed_encoder(channels=("ngram", "word"), dim=DIM)
    door = m.superposed_door("intent-test", channels=("ngram", "word"), dim=DIM)
    assert door.store is m.protostore("intent-test") and door.relevance is m.superposed_relevance("intent-test")
    door.add_option("card", ["my card is lost", "card stolen", "block my card"])
    door.add_option("transfer", ["send money to a friend", "transfer to savings", "wire money abroad"])
    r = door.learn("someone stole my card", "card")
    assert r["pred"] == "card" and r["relevance"] is not None
    assert door.rank("please transfer money to my savings")[0][0] == "transfer"
    with pytest.raises(ValueError):
        m.superposed_relevance("intent-test", channels=("ngram",))


def test_door_is_deterministic():
    def run():
        enc = hs.SuperposedEncoder(channels=("ngram", "word"), dim=DIM)
        d = hs.SuperposedDoor(enc)
        d.add_option("a", ["alpha one", "alpha two"])
        d.add_option("b", ["beta one", "beta two"])
        for t, y in (("alpha three", "a"), ("beta three", "b"), ("alpha beta", "b")):
            d.learn(t, y)
        return d.store.digest(), d.relevance.state()["lam"]
    assert run() == run()


def test_bench_aurc_helper():
    spec = importlib.util.spec_from_file_location("bench_superposed", os.path.join(ROOT, "tools", "bench_superposed.py"))
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    ok = np.array([1, 1, 0, 0], bool)
    assert b.aurc([0.9, 0.8, 0.2, 0.1], ok) == pytest.approx(0.0 + 0 + (1 / 3) / 4 + (1 / 2) / 4)
    assert b.aurc([0.1, 0.2, 0.8, 0.9], ok) > b.aurc([0.9, 0.8, 0.2, 0.1], ok)   # a worse ranking scores worse
    lo, hi = b.boot_ci(lambda idx: float(ok[idx].mean()), 4, 200)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0

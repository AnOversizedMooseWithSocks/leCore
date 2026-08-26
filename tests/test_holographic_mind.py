"""Tests for holographic_mind: the universal encoder lands every modality in one
space, names the modality it would use (self-discovery), and the recall index
stays exact until the store is genuinely big. (The old Mind facade this file
used to test was retired in favour of UnifiedMind -- see holographic_unified.)"""

import numpy as np

from holographic.agents_and_reasoning.holographic_mind import UniversalEncoder, _Index


def test_universal_encoder_is_consistent_and_similarity_preserving():
    enc = UniversalEncoder(dim=512, seed=0)
    # same input -> same vector (deterministic)
    a1 = enc.encode("the quick brown fox")
    a2 = enc.encode("the quick brown fox")
    assert np.allclose(a1, a2)
    # every modality lands on a unit vector in the one space
    for x, mod in [("hello world", None), (3.5, None), ({"a": 1, "b": "x"}, None),
                   (np.linspace(0, 1, 40), "vector"), (np.ones((8, 8)), "image"),
                   ([1, "two", 3.0], None)]:
        v = enc.encode(x, mod)
        assert v.shape == (512,)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6 or np.linalg.norm(v) == 0.0
    # a feature vector close to another encodes more similarly than to a far one
    base = np.linspace(0, 1, 32)
    near = base + 0.02 * np.random.default_rng(1).standard_normal(32)
    far = np.random.default_rng(2).standard_normal(32)
    vb, vn, vf = (enc.encode(z, "vector") for z in (base, near, far))
    assert float(vb @ vn) > float(vb @ vf)


def test_infer_names_the_modality_for_every_type():
    enc = UniversalEncoder(dim=256, seed=0)
    assert enc.infer("a sentence") == "text"
    assert enc.infer(3.5) == "number"
    assert enc.infer(7) == "number"
    assert enc.infer(True) == "category"          # bool is an int -- must not be a number
    assert enc.infer({"k": 1}) == "record"
    assert enc.infer(np.ones((8, 8))) == "image"
    assert enc.infer(np.ones(8)) == "vector"
    assert enc.infer([1, "two", 3.0]) == "sequence"


def test_infer_treats_token_lists_as_text_not_sequence():
    # The measured bug: a list of tokens falling into the order-sensitive sequence
    # encoder silently wrecks topic similarity. Inference must send it to the
    # order-insensitive text bundle -- and the encoding must agree with the tag.
    enc = UniversalEncoder(dim=512, seed=0)
    toks = ["holographic", "memory", "engine"]
    assert enc.infer(toks) == "text"
    assert np.allclose(enc.encode(toks), enc.encode(toks, "text"))
    # order-insensitive: a shuffled token list encodes to the same bundle
    assert np.allclose(enc.encode(toks), enc.encode(list(reversed(toks)), "text"))
    # an explicit sequence is still order-SENSITIVE -- the escape hatch works
    s1 = enc.encode(toks, "sequence")
    s2 = enc.encode(list(reversed(toks)), "sequence")
    assert not np.allclose(s1, s2)


def test_encode_and_infer_cannot_disagree():
    # encode(x) with no declared modality must equal encode(x, infer(x)) for every
    # type -- the single-source-of-truth property that routing depends on.
    enc = UniversalEncoder(dim=256, seed=3)
    for x in ["words here", 2.5, {"a": 1}, np.ones((4, 4)), np.ones(6),
              ["tok", "list"], [1, 2, 3], True]:
        assert np.allclose(enc.encode(x), enc.encode(x, enc.infer(x)))


def test_index_recall_is_exact_below_the_forest_crossover():
    # the recall index must do an exact scan in the small/medium regime (the
    # forest there was measured to cost MORE wall-clock for LESS accuracy)
    rng = np.random.default_rng(0)
    idx = _Index(128)
    vecs = rng.standard_normal((500, 128))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    for i, v in enumerate(vecs):
        idx.add(v, i)
    assert idx._forest is None                    # no forest built in this regime
    for i in (0, 123, 499):
        q = vecs[i] + 0.2 * rng.standard_normal(128)
        payload, _ = idx.recall(q / np.linalg.norm(q))
        assert payload == int((vecs @ (q / np.linalg.norm(q))).argmax())
    assert idx._forest is None                    # still exact after recalls


def test_code_modality_encodes_like_text_not_opaque_symbol():
    # The pinned foot-gun: a declared "code" modality used to fall through to the
    # opaque-symbol path, so two nearly identical snippets encoded ORTHOGONALLY
    # (measured cosine 0.04). Code must encode like text -- bag of token vectors --
    # with the distinct name existing only for routing.
    enc = UniversalEncoder(dim=512, seed=0)
    a = enc.encode("def foo(x): return x + 1", "code")
    b = enc.encode("def foo(x): return x + 2", "code")
    assert float(a @ b) > 0.6                       # near-identical snippets are near
    t = enc.encode("def foo(x): return x + 1", "text")
    assert np.allclose(a, t)                        # same encoding, different routing tag


def test_index_big_regime_reflex_and_matrix_cache():
    # The big-regime recall path, pinned: (a) the stacked matrix is cached -- before
    # this, every recall re-stacked the whole store at a measured 54 ms PER CALL at
    # 16k items; (b) the slime-mould ReflexCache fronts the forest -- on a Zipf
    # stream it answered 70% of queries with recall@1 RISING 96.8% -> 99.0% at 3x
    # lower cost, while a uniform stream's flux guard deactivates it (cost a wash).
    from holographic.agents_and_reasoning.holographic_navigator import _zipf_workload
    rng = np.random.default_rng(0)
    N, DIM = 4500, 256
    items = rng.standard_normal((N, DIM))
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    idx = _Index(DIM)
    for i, v in enumerate(items):
        idx.add(v, i)

    r = np.random.default_rng(7)
    ok = 0
    wl = _zipf_workload(N, 800, 1.3, seed=5)
    for tgt in wl:
        q = items[tgt] + 0.5 * r.standard_normal(DIM) / np.sqrt(DIM)
        q /= np.linalg.norm(q)
        truth = int((items @ q).argmax())
        ok += (idx.recall(q)[0] == truth)
    assert idx._forest is not None                  # big regime engaged
    assert idx._mat is not None                     # matrix cached, not re-stacked
    assert idx._reflex is not None and idx._reflex.t == len(wl)
    assert ok / len(wl) >= 0.90                     # reflex never costs recall on skew
    assert len(idx._reflex.hot) > 0                 # veins thickened toward the popular

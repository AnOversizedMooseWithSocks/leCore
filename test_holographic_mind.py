"""Tests for holographic_mind: the universal encoder lands every modality in one
space, and the Mind assembles the right machine for classify / recall / decide."""

import numpy as np

from holographic_mind import UniversalEncoder, Mind, assemble


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


def test_mind_classifies_text_and_records():
    m = Mind(seed=1)
    for s in ["happy joy wonderful", "great love delight", "awful hate terrible",
              "sad gloom misery"]:
        m.teach(s, "pos" if s[0] in "hg" else "neg")
    # (the four above split pos/neg by first letter h/g vs a/s -- fix labels)
    m = Mind(seed=1)
    pos = ["happy joy wonderful", "great love delight", "joy happy great"]
    neg = ["awful hate terrible", "sad gloom misery", "hate awful sad"]
    for s in pos: m.teach(s, "pos")
    for s in neg: m.teach(s, "neg")
    assert m.classify("joy and delight")[0] == "pos"
    assert m.classify("terrible misery")[0] == "neg"

    r = Mind(seed=2)
    for rec, lab in [({"cover": "feathers", "fly": "yes"}, "bird"),
                     ({"cover": "scales", "fly": "no"}, "fish"),
                     ({"cover": "fur", "fly": "no"}, "mammal")]:
        r.teach(rec, lab)
    assert r.classify({"cover": "feathers", "fly": "yes"})[0] == "bird"


def test_mind_recall_returns_right_payload():
    m = Mind(seed=3)
    facts = {"paris": "France", "tokyo": "Japan", "cairo": "Egypt"}
    for k, v in facts.items():
        m.store(k, v)
    ans, score = m.recall("tokyo")
    assert ans == "Japan" and score > 0.9


def test_mind_decide_learns_contextual_bandit():
    m = Mind(seed=4).actions(["a", "b"])
    rng = np.random.default_rng(0)
    best = {"x": "a", "y": "b"}
    for _ in range(120):
        c = "x" if rng.random() < 0.5 else "y"
        act = m.act({"c": c}, explore=True, epsilon=0.3)
        m.reinforce({"c": c}, act, 1.0 if act == best[c] else 0.0)
    hits = np.mean([m.act({"c": c}) == best[c] for c in ("x", "y") for _ in range(20)])
    assert hits >= 0.9


def test_assemble_infers_the_task():
    clf = assemble([("good nice", "pos"), ("bad awful", "neg")])
    assert clf.classify("nice")[0] in ("pos", "neg")          # built a classifier
    idx = assemble(["alpha", "beta", "gamma"])
    assert idx.recall("alpha")[0] == "alpha"                  # built an index
    dec = assemble([({"c": "x"}, "a", 1.0), ({"c": "x"}, "b", 0.0)] * 20)
    assert dec.act({"c": "x"}) in ("a", "b")                  # built a brain

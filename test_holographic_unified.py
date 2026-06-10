"""Tests for the top-level UnifiedMind: one encoder, one self-organizing memory, one
decision brain over a single holographic space. The point is that the pieces share the
substrate -- so one memory should classify several modalities at once, the recall view
should search the same vectors, and the same mind should be able to decide."""

import numpy as np

from holographic_unified import UnifiedMind, _patterns


def _records(dept, rng, n):
    return [({"dept": dept, "level": int(rng.integers(1, 6))}, f"rec:{dept}", "record")
            for _ in range(n)]


def test_one_memory_holds_and_classifies_several_modalities():
    rng = np.random.default_rng(0)
    mind = UnifiedMind(dim=1024, seed=0)
    train, test = [], []
    for kind in ("rows", "cols", "diag", "check"):
        train += [(_patterns(kind, rng), f"img:{kind}", "image") for _ in range(20)]
        test += [(_patterns(kind, rng), f"img:{kind}", "image") for _ in range(8)]
    for d in ("eng", "sales", "ops"):
        train += _records(d, rng, 20)
        test += _records(d, rng, 8)
    rng.shuffle(train)
    for x, label, mod in train:
        mind.learn(x, label, mod)
    mind.maintain_now()

    # one memory now holds every label from every modality
    labels = set(mind.memory.live.counts_by_label())
    assert {f"img:{k}" for k in ("rows", "cols", "diag", "check")} <= labels
    assert {f"rec:{d}" for d in ("eng", "sales", "ops")} <= labels

    acc = sum(mind.classify(x, mod)[0] == lab for x, lab, mod in test) / len(test)
    assert acc >= 0.85          # both modalities classify well from the single store


def test_recall_view_searches_the_same_vectors():
    rng = np.random.default_rng(1)
    mind = UnifiedMind(dim=1024, seed=1)
    for kind in ("rows", "check"):
        for _ in range(15):
            mind.learn(_patterns(kind, rng), f"img:{kind}", "image")
    (label, _), sim = mind.recall(_patterns("rows", rng), "image")
    assert label == "img:rows"   # nearest stored individual is the right kind
    assert sim > 0.5


def test_same_mind_decides_over_the_same_space():
    mind = UnifiedMind(dim=1024, seed=2).actions(["left", "right"])
    rng = np.random.default_rng(2)
    for _ in range(400):
        n = float(rng.uniform(-3, 3))
        good = "right" if n > 0 else "left"
        choice = mind.decide(n, explore=True, epsilon=0.3, modality="number")
        mind.reinforce(n, choice, 1.0 if choice == good else 0.0, modality="number")
    acc = sum(mind.decide(float(v), modality="number") == ("right" if v > 0 else "left")
              for v in np.linspace(-3, 3, 40)) / 40
    assert acc >= 0.7            # the shared-substrate brain learned the contextual choice


def test_routing_removes_cross_modal_interference():
    # A flat store of several modalities can mistake a query for a foreign-modality
    # concept; restricting the query to its own modality (a cheap router, since the
    # modality is known) should never hurt and should fix those cross-modal errors.
    from holographic_mind import UniversalEncoder
    from holographic_text import TOPICS, _content, _split
    rng = np.random.default_rng(0)
    corpus = [s for ss in TOPICS.values() for s in ss]
    mind = UnifiedMind(dim=1024, seed=0).read([_content(s) for s in corpus])
    text_te = []
    for topic, ss in TOPICS.items():
        a, b = _split(ss, frac=0.7, seed=2)
        for s in a:
            mind.learn(_content(s), topic, "text")
        text_te += [(_content(s), topic) for s in b]
    for kind in ("rows", "cols", "diag", "check"):
        for _ in range(20):
            img = np.zeros((8, 8)); img[::2, :] = 1.0
            mind.learn(img + 0.15 * rng.standard_normal((8, 8)), f"img:{kind}", "image")
    for d in ("eng", "sales", "ops"):
        for _ in range(20):
            mind.learn({"dept": d, "level": int(rng.integers(1, 6))}, f"rec:{d}", "record")
    mind.maintain_now()

    flat = sum(mind.classify(t, "text", route=False)[0] == lab for t, lab in text_te) / len(text_te)
    routed = sum(mind.classify(t, "text", route=True)[0] == lab for t, lab in text_te) / len(text_te)
    assert routed >= flat          # routing never hurts
    # every routed prediction is a text label (no cross-modal leakage)
    assert all(mind.classify(t, "text", route=True)[0] in set(TOPICS) for t, _ in text_te)


def test_same_mind_generates_over_the_shared_space():
    # Generation is the fourth operation on the one model: learn to continue a sequence,
    # then produce more of it. The next-symbol prediction is holographic cleanup (the
    # same primitive the classifier uses); only the context key is exact.
    from holographic_text import TOPICS
    text = " ".join(s for ss in TOPICS.values() for s in ss).lower()
    mind = UnifiedMind(dim=1024, seed=0).learn_sequence(text)          # default fractal engine
    out = mind.generate("the ", length=80, temperature=0.4)
    assert len(out) > 50                                  # it produced a continuation
    assert set(out) <= set(text)                          # only symbols it actually learned
    # next-symbol prediction is a flat-engine primitive; it should beat uniform random
    cut = int(len(text) * 0.9)
    m2 = UnifiedMind(dim=1024, seed=0).learn_sequence(text[:cut], n=5, hierarchical=False)
    held = text[cut:cut + 800]
    ok = sum(m2.next_symbol(held[max(0, j - 4):j]) == held[j] for j in range(1, len(held)))
    assert ok / (len(held) - 1) > 1.0 / len(set(text))    # well above chance

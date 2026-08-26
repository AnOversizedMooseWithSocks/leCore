"""Tests for holographic_moe: a learned holographic gate (the creature's brain)
routes each input to the right specialist, beating any single expert and
approaching the oracle -- the genuine mixture-of-experts payoff."""

import numpy as np

from holographic.agents_and_reasoning.holographic_moe import GatedMixture


def _tone(freq, rng):
    t = np.linspace(0, 1, 256, endpoint=False)
    return np.abs(np.fft.rfft(np.sin(2 * np.pi * freq * t) + 0.3 * rng.standard_normal(256)))


def test_learned_gate_routes_crossmodal_and_beats_single():
    rng = np.random.default_rng(0)
    text = [("the cat sat", "animal", None), ("a dog ran", "animal", None),
            ("the kitten purred", "animal", None), ("the car drove", "vehicle", None),
            ("a truck honked", "vehicle", None), ("the engine roared", "vehicle", None)]
    audio = [(_tone(f, rng), "tone_" + n, "vector")
             for n, f in {"lo": 5, "hi": 40}.items() for _ in range(8)]

    moe = GatedMixture(dim=1024, seed=0)
    moe.encoder.learn_text([s for s, _, _ in text])
    moe.add_expert("text", text)
    moe.add_expert("audio", audio)
    moe.train_gate(text + audio, epochs=12)

    test = [("the dog sat", "animal", None), ("the sedan parked", "vehicle", None),
            ("a cat ran", "animal", None), ("the truck drove", "vehicle", None),
            ("the kitten slept", "animal", None), ("the engine stalled", "vehicle", None)]
    for n, f in {"lo": 5, "hi": 40}.items():
        for _ in range(3):
            test.append((_tone(f, rng), "tone_" + n, "vector"))

    def acc(fn):
        return np.mean([fn(x, lab, mod) == lab for x, lab, mod in test])

    learned = acc(lambda x, lab, mod: moe.predict(x, mod)[0])
    oracle = acc(lambda x, lab, mod: moe.oracle_label(x, lab, mod))
    single = max(acc(lambda x, lab, mod, i=i: moe.predict_with(i, x, mod)) for i in range(2))

    assert learned >= 0.85               # routes nearly everything correctly
    assert learned > single + 0.25       # clearly beats any single expert
    assert learned >= oracle - 0.05       # approaches the oracle upper bound


def test_learned_gate_routes_by_content_not_type():
    # Same modality (numbers); two experts own different halves of the line. The
    # gate must route by VALUE, which a simple type check could never do.
    rng = np.random.default_rng(1)
    lo = [(rng.uniform(0.02, 0.46), "L", None) for _ in range(60)]
    hi = [(rng.uniform(0.54, 0.98), "H", None) for _ in range(60)]
    moe = GatedMixture(dim=1024, seed=2, number_range=(0.0, 1.0))
    moe.add_expert("low", lo)
    moe.add_expert("high", hi)
    moe.train_gate(lo + hi, epochs=14)

    test = ([(rng.uniform(0.02, 0.46), "L", None) for _ in range(40)]
            + [(rng.uniform(0.54, 0.98), "H", None) for _ in range(40)])
    learned = np.mean([moe.predict(x, mod)[0] == lab for x, lab, mod in test])
    single = max(np.mean([moe.predict_with(i, x, mod) == lab for x, lab, mod in test])
                 for i in range(2))
    assert learned >= 0.85
    assert learned > single + 0.25


def test_learned_gate_beats_confidence_under_miscalibration():
    # Heterogeneous bank: a calibrated holographic specialist on region A, and an
    # OVERCONFIDENT linear specialist on region B that is confidently wrong on A.
    # Confidence routing is fooled; the outcome-trained gate is not.
    rng = np.random.default_rng(0)
    L = 40
    moe = GatedMixture(dim=512, seed=0)
    cA, cB = rng.standard_normal(L) * 3, rng.standard_normal(L) * 3
    sub = rng.standard_normal((4, L)) * 0.6
    def sample(c):
        return (cA if c < 2 else cB) + sub[c] + 0.4 * rng.standard_normal(L)
    A = [(sample(c := int(rng.integers(2))), c, "vector") for _ in range(250)]
    B = [(sample(c := 2 + int(rng.integers(2))), c, "vector") for _ in range(250)]
    moe.add_expert("holo-A", A)
    moe.add_linear_expert("linear-B", B, temp=25.0)
    moe.train_gate(A + B, epochs=7)

    test = [(sample(c := int(rng.integers(4))), c, "vector") for _ in range(400)]
    def acc(fn):
        return np.mean([fn(x, lab, mod) == lab for x, lab, mod in test])
    single = max(acc(lambda x, lab, mod, i=i: moe.predict_with(i, x, mod)) for i in range(2))
    conf = acc(lambda x, lab, mod: moe.predict_by_confidence(x, mod))
    learned = acc(lambda x, lab, mod: moe.predict(x, mod)[0])

    assert single < 0.65                  # neither expert is good on both regions
    assert learned >= 0.9                 # the gate routes nearly everything right
    assert learned > conf + 0.12          # and clearly beats confidence routing here


def test_schema_gate_routes_to_the_expert_that_understands_the_input():
    # Three experts over distinct text; the schema gate routes a held-out snippet to the
    # expert whose schema compresses it best -- deterministic, no gate training needed.
    import numpy as np
    from holographic.agents_and_reasoning.holographic_moe import GatedMixture
    streams = {
        "weather": "the storm front moved east bringing rain wind and cold over the coast ",
        "finance": "the market closed higher as shares rose on strong quarterly earnings news ",
        "cooking": "fold the butter into the flour then chill the dough before you roll it out ",
    }
    moe = GatedMixture(dim=512, seed=0)
    moe.encoder.learn_text(" ".join(streams.values()).split())
    names = list(streams)
    for nm in names:
        moe.add_expert(nm, [(s, nm, None) for s in (streams[nm] * 8).split(". ")])
        moe.set_expert_schema(len(moe.experts) - 1, streams[nm] * 40, cuts=(0, 40, 110))
    for nm in names:
        held = (streams[nm] * 3)[80:240]
        assert names[moe.schema_route(held)] == nm


def test_learned_gate_beats_schema_gate_under_miscalibration():
    # The schema gate's standing assumption is understanding ~= competence. Break it: the
    # expert that models the input BEST is the one that answers WRONG. Compression routing and
    # confidence routing both get fooled (they trust understanding); only the reward-trained
    # gate, which sees outcomes, learns to avoid the saboteur. This maps the boundary precisely.
    import pytest
    try:
        import nltk
        nltk.data.path.insert(0, "/home/claude/nltk_data")
        from nltk.corpus import gutenberg
        txt = " ".join(w.lower() for w in gutenberg.words("austen-emma.txt") if w.isalpha())
        assert len(txt) > 50000
    except Exception:
        pytest.skip("needs the gutenberg corpus for a rich-enough domain")
    from holographic.agents_and_reasoning.holographic_moe import GatedMixture

    class Fixed:  # controlled competence: a fixed answer at a fixed confidence
        def __init__(self, name, label, conf):
            self.name, self.label, self.conf, self.labels, self.schema = name, label, conf, [label], None
        def predict(self, vec):
            return self.label, self.conf

    train_text = txt[:30000]
    probes = [txt[i:i + 160] for i in range(40000, 40000 + 100 * 160, 160)]
    moe = GatedMixture(dim=1024, seed=0)
    moe.encoder.learn_text(train_text.split())
    moe.add_prebuilt_expert(Fixed("honest", "A", 0.80))     # right, modest confidence, thin model
    moe.add_prebuilt_expert(Fixed("saboteur", "X", 0.95))   # wrong, overconfident, deep model
    moe.set_expert_schema(0, train_text[:1500], cuts=(0, 120, 350))   # thin understanding
    moe.set_expert_schema(1, train_text, cuts=(0, 120, 350))          # deep understanding -> fewest bits

    moe.train_gate([(s, "A", None) for s in probes[:60]], epochs=16, seed=0)
    test = [(s, "A") for s in probes[60:]]
    def acc(p):
        return sum(p(s) == lab for s, lab in test) / len(test)

    assert acc(lambda s: moe.predict(s)[0]) > 0.9            # reward training avoids the saboteur
    assert acc(lambda s: moe.predict_by_schema(s)) < 0.1     # compression gate is fooled
    assert acc(lambda s: moe.predict_by_confidence(s)) < 0.1  # confidence gate is fooled too


def test_hybrid_gate_recovers_where_the_schema_gate_was_fooled():
    # The synthesis: route by compression, but a thin reward signal demotes the saboteur. With
    # no feedback it is exactly the (fooled) schema gate; one calibration pass recovers it.
    import pytest
    try:
        import nltk
        nltk.data.path.insert(0, "/home/claude/nltk_data")
        from nltk.corpus import gutenberg
        txt = " ".join(w.lower() for w in gutenberg.words("austen-emma.txt") if w.isalpha())
        assert len(txt) > 50000
    except Exception:
        pytest.skip("needs the gutenberg corpus for a rich-enough domain")
    from holographic.agents_and_reasoning.holographic_moe import GatedMixture

    class Fixed:
        def __init__(self, name, label, conf):
            self.name, self.label, self.conf, self.labels, self.schema = name, label, conf, [label], None
        def predict(self, vec):
            return self.label, self.conf

    train_text = txt[:30000]
    probes = [txt[i:i + 160] for i in range(40000, 40000 + 120 * 160, 160)]
    moe = GatedMixture(dim=1024, seed=0)
    moe.encoder.learn_text(train_text.split())
    moe.add_prebuilt_expert(Fixed("honest", "A", 0.80))
    moe.add_prebuilt_expert(Fixed("saboteur", "X", 0.95))
    moe.set_expert_schema(0, train_text[:1500], cuts=(0, 120, 350))
    moe.set_expert_schema(1, train_text, cuts=(0, 120, 350))

    test = [(s, "A") for s in probes[80:]]
    def acc(p):
        return sum(p(s) == lab for s, lab in test) / len(test)

    assert all(moe.route_hybrid(s) == moe.schema_route(s) for s, _ in test)  # no feedback == compression
    assert acc(lambda s: moe.predict_by_schema(s)) < 0.1                     # which is fooled
    moe.calibrate_hybrid([(s, "A", None) for s in probes[:80]])              # thin reward pass
    assert acc(lambda s: moe.predict_by_hybrid(s)) > 0.9                     # recovered

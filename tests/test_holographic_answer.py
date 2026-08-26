"""Grounded answering -- construct a short, accurate, non-verbatim sentence from retrieved
knowledge, abstaining honestly when the mind does not know.

Most tests pin the pure realizer on hand-made answer() structs (fast, deterministic); two run
end-to-end through the mind to prove the wiring (above/below) and the no-fabrication discipline.
"""
import numpy as np
from holographic.agents_and_reasoning.holographic_answer import realize_answer, _english_list, _art
from holographic.misc.holographic_unified import UnifiedMind


# ---- pure realizer: form, accuracy of phrasing, abstention --------------------
def test_is_a_yes_renders_chain():
    s = realize_answer({"kind": "is_a", "subject": "dog", "ancestor": "animal",
                        "answer": True, "chain": ["dog", "mammal", "animal", "organism"]})
    assert s == "Yes -- a dog is a mammal, which is an animal."


def test_is_a_no_shows_what_it_is():
    s = realize_answer({"kind": "is_a", "subject": "salmon", "ancestor": "bird",
                        "answer": False, "chain": ["salmon", "fish", "animal"]})
    assert s.lower().startswith("no") and "fish" in s and "not a bird" in s


def test_is_a_unknown_subject_abstains():
    # the subject has no known parent (chain is just itself) -> must NOT assert yes/no, must abstain
    s = realize_answer({"kind": "is_a", "subject": "dragon", "ancestor": "animal",
                        "answer": False, "chain": ["dragon"]})
    assert "don't have dragon" in s.lower() and not s.lower().startswith("no")


def test_role_renders_value_and_gates_low_confidence():
    assert realize_answer({"kind": "role", "concept": "france", "role": "capital",
                           "value": "paris", "confidence": 0.6}) == "The capital of france is paris."
    low = realize_answer({"kind": "role", "concept": "france", "role": "capital",
                          "value": "paris", "confidence": 0.01})
    assert "not sure" in low.lower()                       # below the floor -> abstain, don't assert


def test_define_renders_class_and_relatives():
    s = realize_answer({"kind": "define", "word": "dog",
                        "is_a_chain": ["dog", "mammal", "animal"],
                        "meaning": [("cat", 0.8), ("wolf", 0.7), ("rock", 0.05)]})
    assert s.startswith("A dog is a mammal") and "cat" in s and "wolf" in s
    assert "rock" not in s                                 # below similarity floor -> dropped


def test_recall_and_classify_gate_low_scores():
    assert "don't have anything" in realize_answer(
        {"kind": "recall", "label": "x", "score": 0.05}).lower()
    assert "not sure what kind" in realize_answer(
        {"kind": "classify", "label": "poem", "score": 0.05}).lower()
    assert realize_answer({"kind": "classify", "label": "poem", "score": 0.9}) == "That looks like poem."


def test_completion_and_unknown_abstain():
    for k in ("completion", "unknown", "none"):
        s = realize_answer({"kind": k, "text": "whatever"})
        assert "can't answer that from what i know" in s.lower()


def test_helpers():
    assert _english_list(["a"]) == "a" and _english_list(["a", "b"]) == "a and b"
    assert _english_list(["a", "b", "c"]) == "a, b and c"
    assert _art("oak") == "an" and _art("dog") == "a"


# ---- end-to-end through the mind: delegation + no fabrication -----------------
def _mind():
    m = UnifiedMind(dim=512, seed=0)
    m.learn_encyclopedia({"dog": {"is_a": "mammal"}, "mammal": {"is_a": "animal"},
                          "animal": {"is_a": "organism"},
                          "france": {"is_a": "country", "capital": "paris"}})
    return m


def test_answer_text_delegates_to_answer():
    # ABOVE/BELOW: answer_text is exactly realize_answer(answer(q)), no separate logic
    m = _mind()
    for q in ("is a dog an animal?", "what is the capital of france?", "what is a dog?"):
        assert m.answer_text(q) == realize_answer(m.answer(q))


def test_answer_text_is_accurate_and_does_not_fabricate():
    m = _mind()
    assert m.answer_text("is a dog an animal?").lower().startswith("yes")
    assert "paris" in m.answer_text("what is the capital of france?").lower()
    # unknowns must abstain, never invent a fact
    a = m.answer_text("what is the capital of atlantis?").lower()
    assert "atlantis" not in a or "can't answer" in a or "not sure" in a
    assert "paris" not in m.answer_text("is a dragon an animal?").lower()   # no spurious content

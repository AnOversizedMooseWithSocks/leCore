"""VSA-native question routing -- intent from a blend of word meanings, arguments from a
concept-scan with order. Pins: intent classification of natural phrasings; the ORDER logic that
fixes is_a direction; end-to-end answers on phrasings the regex misses; backward-compat (templates
still win); and honest abstention (unknown subject / unknown concept -> no fabrication).
"""
from holographic.agents_and_reasoning.holographic_intent import route_intent, route_question
from holographic.misc.holographic_unified import UnifiedMind


def _kb():
    facts = {"dog": {"is_a": "mammal"}, "wolf": {"is_a": "mammal"}, "mammal": {"is_a": "animal"},
             "animal": {"is_a": "organism"}, "bird": {"is_a": "animal"},
             "oak": {"is_a": "tree"}, "tree": {"is_a": "plant"}, "plant": {"is_a": "organism"},
             "france": {"is_a": "country", "capital": "paris"},
             "japan": {"is_a": "country", "capital": "tokyo"}}
    return UnifiedMind(dim=1024, seed=0).learn_encyclopedia(facts)


def test_route_intent_natural_phrasings():
    m = _kb()
    assert route_intent(m, "could you tell me whether a dog is an animal")[0] == "IS_A"
    assert route_intent(m, "do you happen to know the capital of japan")[0] == "ROLE"
    assert route_intent(m, "what exactly is a wolf, in your own words")[0] in ("DEFINE", "SIMILAR")


def test_route_question_direction_from_order():
    # the order-scan assigns subject (first) and ancestor (last) -- the direction the blend cannot
    m = _kb()
    yes = route_question(m, "is a dog an animal")
    no = route_question(m, "is an animal a dog")
    assert yes["kind"] == "is_a" and yes["subject"] == "dog" and yes["answer"] is True
    assert no["kind"] == "is_a" and no["subject"] == "animal" and no["answer"] is False


def test_route_question_role_and_define_natural():
    m = _kb()
    r = route_question(m, "do you happen to know the capital of japan")
    assert r["kind"] == "role" and r["value"] == "tokyo"
    d = route_question(m, "what exactly is a wolf in your own words")
    assert d["kind"] == "define" and d["word"] == "wolf"


def test_route_question_abstains_when_subject_unknown():
    # IS_A intent but the SUBJECT is unknown (only the ancestor is a known concept) -> abstain,
    # do NOT describe the lone known concept (that would answer the wrong thing)
    m = _kb()
    assert route_question(m, "could you tell me whether a dragon is an animal") is None


def test_answer_text_answers_natural_phrasing_via_fallback():
    m = _kb()
    res = m.answer("could you tell me whether a dog is an animal")
    assert res.get("via") == "vsa" and res["answer"] is True            # routed by the VSA fallback
    assert m.answer_text("could you tell me whether a dog is an animal").lower().startswith("yes")
    assert "tokyo" in m.answer_text("do you happen to know the capital of japan").lower()


def test_templated_questions_unchanged_backward_compatible():
    # an exact-template question is answered by the regex FIRST -- the VSA fallback never runs
    m = _kb()
    r = m.answer("is a dog an animal?")
    assert r.get("via") != "vsa"
    assert m.answer_text("is a dog an animal?").lower().startswith("yes")
    assert "paris" in m.answer_text("what is the capital of france?").lower()


def test_abstention_preserved_on_natural_unknown():
    m = _kb()
    # natural phrasing, unknown concept -> still abstains, never fabricates
    a = m.answer_text("could you tell me whether a griffin is a mammal").lower()
    assert "can't answer" in a or "don't" in a
    b = m.answer_text("do you know the capital of atlantis").lower()
    assert "atlantis" not in b or "can't answer" in b

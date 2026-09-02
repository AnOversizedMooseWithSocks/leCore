"""The first leCore benchmark whose questions leCore did not write -- and the caveat that
decides whether its headline number means anything.

Every abstention figure in this repo before this file came from a set built BY REMOVAL FROM
leCore'S OWN CATALOG. This harness makes the task file a parameter, in LongMemEval's published
schema, using their `_abs` question-id convention.
"""
import lecore
import pytest

from holographic.agents_and_reasoning import holographic_extbench as eb


def _sess(*qa):
    return [[{"role": "user", "content": q}, {"role": "assistant", "content": a}] for q, a in qa]


def _mind():
    return lecore.UnifiedMind(dim=512, seed=0)


def test_the_abs_suffix_is_their_contract_and_is_read_exactly():
    assert eb.is_abstention("lme_3_abs") and not eb.is_abstention("lme_3")
    recs = eb.longmemeval_records(eb._fixture())
    assert [r["abstention"] for r in recs] == [False, False, True, True]


def test_a_malformed_instance_RAISES_rather_than_shrinking_the_corpus_silently():
    """A benchmark that quietly drops what it cannot parse reports a number for a set nobody
    chose. Both required fields are checked, and both raise."""
    for bad in ({"question_id": "x"}, {"question": "q"}):
        with pytest.raises(ValueError):
            eb.longmemeval_records([bad])


def test_the_corpus_digest_is_stable_and_uses_hashlib():
    """A benchmark identifier that changes between processes is not an identifier -- so this is
    sha256, never hash(). Two runs quoting the same digest ran the same set."""
    recs = eb.longmemeval_records(eb._fixture())
    d = eb.corpus_digest(recs)
    assert d == eb.corpus_digest(recs) and len(d) == 16
    assert d != eb.corpus_digest(recs[:2])


def test_the_gate_holds_on_NEAR_MISS_abstention_questions():
    """The fixture's own abstention questions are easy (boat vs bike). LongMemEval's are about
    PLAUSIBLE-but-absent events, so the honest probe is a question one word from a taught fact:
    'dentist appointment on thursday' against a taught 'tuesday'. 4/4 declined."""
    hay = _sess(("what time is my dentist appointment on tuesday",
                 "your dentist appointment is at 9am on tuesday"),
                ("what colour is my bike", "your bike is red"),
                ("where did I park the car", "you parked on level 3"),
                ("what is my sister's name", "your sister is called Mara"))
    cases = [{"question_id": "h1", "question": "what time is my dentist appointment on tuesday",
              "haystack_sessions": hay},
             {"question_id": "h3_abs", "question": "what time is my dentist appointment on thursday",
              "haystack_sessions": hay},
             {"question_id": "h4_abs", "question": "what colour is my sister's bike",
              "haystack_sessions": hay},
             {"question_id": "h5_abs", "question": "where did I park the bike",
              "haystack_sessions": hay},
             {"question_id": "h6_abs", "question": "what is my brother's name",
              "haystack_sessions": hay}]
    rep = eb.run(eb.longmemeval_records(cases), _mind)
    assert rep["n_abs"] == 4 and rep["abstained_abs"] == 4
    assert rep["false_answer_rate"] == 0.0


def test_THE_CAVEAT_a_system_that_answers_nothing_scores_perfect_abstention():
    """THE MOST IMPORTANT TEST IN THIS FILE, and it asserts a WEAKNESS rather than a win.

    LongMemEval has five abilities and abstention is one. On a fixture shaped like the other
    four -- knowledge-update, multi-session, temporal -- this engine's T0 memory answers the
    single-session lookup and DECLINES the rest. Recall 0.25 with abstention 1.00 and
    false-answer 0.00: both halves true, and only the pair honest. If someone later improves
    recall, this test fails and MUST be updated with the new number -- which is the point. It
    exists so nobody can quote the abstention rate without meeting the recall rate."""
    hay = _sess(("I adopted a dog", "congratulations on the dog"),
                ("what did I name the dog", "you named the dog Pepper"),
                ("how old is Pepper", "Pepper is 3"),
                ("I moved to Bristol in March", "noted, you moved to Bristol in March"),
                ("I moved again in September, to Leeds", "noted, you now live in Leeds"))
    cases = [{"question_id": "s1", "question": "what did I name the dog", "haystack_sessions": hay},
             {"question_id": "k1", "question": "where do I live now", "haystack_sessions": hay},
             {"question_id": "m1", "question": "how old is my dog", "haystack_sessions": hay},
             {"question_id": "t1", "question": "which city did I live in before Leeds",
              "haystack_sessions": hay},
             {"question_id": "a1_abs", "question": "what did I name the cat",
              "haystack_sessions": hay}]
    rep = eb.run(eb.longmemeval_records(cases), _mind)
    assert rep["recall_rate"] == 0.25, "recall moved -- update the number, do not delete the test"
    assert rep["abstention_rate"] == 1.0
    assert rep["false_answer_rate"] == 0.0
    assert rep["paired_rate"] == 0.4, "the paired rate is the one to quote"


def test_it_is_reachable_as_a_door_and_returns_the_named_gate():
    """A benchmark an agent cannot invoke is a benchmark nobody re-runs. The report also states
    WHICH gate ran, because leCore has two abstentions and quoting the wrong one is the error
    this module was built to prevent."""
    rep = lecore.UnifiedMind(dim=64, seed=0).external_abstention(eb._fixture())
    assert rep["n"] == 4 and rep["false_answer_rate"] == 0.0
    assert "NOT route_or_abstain" in rep["gate"]
    assert "user, assistant" in rep["indexing"], "the indexing choice must travel with the number"


# ---------------------------------------------------------------------------------------------
# THE EXCHANGE RATE. Sweep 135 measured recall 0.25 with abstention 1.00 and said recall was the
# binding constraint. Sweep 136 added a retrieval rung and measured what recall COSTS: the 0.00
# false-answer rate was purchased at recall 0.25, and the rung doubles the paired rate by
# spending a quarter of the abstention. These tests pin the curve so the trade is never quoted
# from one end only.
# ---------------------------------------------------------------------------------------------

_HAY = _sess(("I adopted a dog", "congratulations on the dog"),
             ("what did I name the dog", "you named the dog Pepper"),
             ("how old is Pepper", "Pepper is 3"),
             ("I moved to Bristol in March", "noted, you moved to Bristol in March"),
             ("I moved again in September, to Leeds", "noted, you now live in Leeds"))

_CURVE_CASES = [
    {"question_id": "s1", "question": "what did I name the dog", "haystack_sessions": _HAY},
    {"question_id": "k1", "question": "where do I live now", "haystack_sessions": _HAY},
    {"question_id": "m1", "question": "how old is my dog", "haystack_sessions": _HAY},
    {"question_id": "t1", "question": "which city did I live in before Leeds", "haystack_sessions": _HAY},
    {"question_id": "a1_abs", "question": "what did I name the cat", "haystack_sessions": _HAY},
    {"question_id": "a2_abs", "question": "when is my flight to Lisbon", "haystack_sessions": _HAY},
    {"question_id": "a3_abs", "question": "how old is my neighbour", "haystack_sessions": _HAY},
    {"question_id": "a4_abs", "question": "what did the vet say about Pepper", "haystack_sessions": _HAY},
]


def _curve(**kw):
    return eb.run(eb.longmemeval_records(_CURVE_CASES), _mind, **kw)


def test_the_no_rung_baseline_on_the_WIDER_fixture():
    """NOTE THE CORPUS. This is a 4-answerable / 4-abstention fixture, WIDER than sweep 135's
    2/2/1 one, so its baseline paired rate is 0.25 and NOT the 0.40 sweep 135 reported. Comparing
    a rate across corpora is exactly the error this file exists to prevent -- the digest is in
    every report so two numbers can be checked for having come from the same set."""
    r = _curve()
    assert (r["recall_rate"], r["abstention_rate"], r["false_answer_rate"], r["paired_rate"]) \
        == (0.25, 1.0, 0.0, 0.25)


def test_a_retrieval_rung_DOUBLES_the_paired_rate_and_SPENDS_abstention():
    """THE RESULT. At floor 0.50 the semantic rung takes recall 0.25 -> 0.50 and abstention
    1.00 -> 0.75, paying 0.25 of false-answer for it. Paired doubles, 0.25 -> 0.50. Both halves
    are asserted, because quoting either alone is the thing that makes a benchmark dishonest."""
    r = _curve(retrieve="semantic", floor=0.50)
    assert r["recall_rate"] == 0.50, "recall moved -- update the number, do not delete the test"
    assert r["abstention_rate"] == 0.75
    assert r["false_answer_rate"] == 0.25, "the rung MUST cost abstention; a free lunch is a bug"
    assert r["paired_rate"] == 0.50


def test_the_floor_is_the_whole_design_low_invents_high_declines():
    """The two ends of the curve, pinned. At floor 0.20 the engine answers every abstention
    question (false-answer 1.00, paired 0.00) -- it invents. At floor 0.70 nothing clears the bar
    and it is back to the no-rung numbers -- it declines. The useful floor is the interior."""
    low = _curve(retrieve="semantic", floor=0.20)
    assert low["false_answer_rate"] == 1.0 and low["paired_rate"] == 0.0
    high = _curve(retrieve="semantic", floor=0.70)
    assert (high["recall_rate"], high["false_answer_rate"]) == (0.25, 0.0)


def test_bm25_reaches_the_same_peak_on_a_floor_that_does_NOT_transfer():
    """A second rung, same peak paired rate (0.50), at floor 3.0 rather than 0.50 -- because bm25
    returns RAW Okapi weights, which are corpus- and length-dependent. Two rungs agreeing on the
    peak is evidence the peak is a property of the corpus rather than of one scorer; the floors
    differing by 6x is the warning that neither floor is a constant to copy."""
    r = _curve(retrieve="bm25", floor=3.0)
    assert r["paired_rate"] == 0.50
    assert r["floor"] == 3.0 and r["retrieve"] == "bm25"


def test_retrieve_None_is_byte_identical_to_having_no_rung_at_all():
    """Additive by default: the parameter exists and changes nothing until it is asked for."""
    a, b = _curve(), _curve(retrieve=None, floor=0.9)
    for k in ("recall_rate", "abstention_rate", "false_answer_rate", "paired_rate", "corpus"):
        assert a[k] == b[k]
    assert a["retrieve"] is None and a["floor"] is None


def test_an_unknown_rung_raises_rather_than_silently_doing_nothing():
    with pytest.raises(ValueError):
        _curve(retrieve="magic", floor=0.5)

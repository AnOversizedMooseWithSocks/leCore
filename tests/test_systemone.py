"""Contract pins for holographic_systemone (sweep 171) -- the typed-decision (System One) door.

WHAT IS PINNED AND WHY (sweep 169's rule applied: pin mechanisms with a controlled vocabulary,
never a live calibration artifact -- no assertion here depends on catalog density or on any
natural sentence clearing a threshold "today"):
  1. Schema strictness IS the type guarantee: bad specs raise before any state is scored.
  2. The noul==choice(yes/no) identity: one code path, exact to 1e-12.
  3. The mind wiring is DELEGATING and composes cross-faculty: mind.systemone_decide runs the
     SAME encoder as mind.build_prototypes/match_prototype (perceive text), so a prototype built
     by one faculty and a decision made by the other agree -- the hard lesson on record is that a
     shared kernel is not a shared manifold, so the agreement is asserted, not assumed.
  4. Honesty gates: no probability before calibration; batch row-identical to single; off-schema
     abstention with the why named.
"""
import numpy as np
import pytest

import lecore
from holographic.agents_and_reasoning.holographic_systemone import (
    SchemaError, SystemOne, _hash_bow_encode, _pav_fit, validate_questions)

# Controlled vocabulary: disjoint token sets per class so the pins measure the MECHANISM,
# not an encoder's opinion about English.
QS = {"cat": {"type": "choice", "options": ["billing", "shipping", "bug"], "examples": {
        "billing": ["invoice charge refund payment", "card charged twice billing"],
        "shipping": ["package delivery tracking courier", "parcel arrives late shipping"],
        "bug": ["crash stack trace error", "segfault bug crash report"]}}}


def test_schema_rejects_bad_specs():
    for bad in [{}, {"q": {"type": "pick"}},
                {"q": {"type": "choice", "options": ["only-one"]}},
                {"q": {"type": "choice", "options": ["a", "b"], "surprise": 1}},
                {"q": {"type": "score", "min": 2, "max": 2}},
                {"q": {"type": "noul", "examples": {"maybe": ["x"]}}}]:
        with pytest.raises(SchemaError):
            validate_questions(bad)


def test_noul_is_choice_yes_no_exactly():
    enc = _hash_bow_encode()
    ex = {"yes": ["approve ship now"], "no": ["reject deny request"]}
    a = SystemOne(enc); a.fit({"q": {"type": "noul", "examples": ex}})
    b = SystemOne(enc); b.fit({"q": {"type": "choice", "options": ["yes", "no"], "examples": ex}})
    ra = a.decide("approve and ship")["q"]["ranked"]
    rb = b.decide("approve and ship")["q"]["ranked"]
    assert [x[0] for x in ra] == [x[0] for x in rb]
    assert max(abs(x[1] - y[1]) for x, y in zip(ra, rb)) < 1e-12


def test_mind_wiring_shares_the_prototype_manifold():
    """Cross-faculty: build_prototypes (relations faculty) and systemone (this faculty) must
    live on the SAME encoder manifold. A prototype the relations door builds for 'billing' has
    to be the vector the systemone door scores against, cosine 1.0 -- if a future refactor gives
    systemone its own encoder, this is the test that names it."""
    m = lecore.UnifiedMind(dim=512, seed=0)
    protos = m.build_prototypes({"billing": QS["cat"]["examples"]["billing"]})
    so = m.systemone(QS)
    row = so._mat["cat"][so._meta["cat"]["labels"].index("billing")]
    assert float(np.dot(protos["billing"], row)) > 1.0 - 1e-9


def test_mind_decide_types_and_honesty():
    m = lecore.UnifiedMind(dim=512, seed=0)
    a = m.systemone_decide("my card was charged twice on the invoice", QS)
    # value is schema-typed or None -- nothing else is constructible.
    assert a["cat"]["value"] in QS["cat"]["options"] + [None]
    # No probability before calibration: honesty is part of the contract, not a mood.
    assert a["cat"]["p"] is None and a["cat"]["calibrated"] is False
    assert a["cat"]["ranked"][0][0] in QS["cat"]["options"]


def test_batch_identical_to_single():
    m = lecore.UnifiedMind(dim=512, seed=0)
    states = ["package tracking lost courier", "crash on startup stack trace"]
    assert m.systemone_map(states, QS) == [m.systemone_decide(s, QS) for s in states]


def test_calibration_gives_bounded_probability():
    m = lecore.UnifiedMind(dim=512, seed=0)
    labeled = [("invoice refund charge", {"cat": "billing"}),
               ("card charged twice", {"cat": "billing"}),
               ("parcel courier tracking", {"cat": "shipping"}),
               ("delivery late package", {"cat": "shipping"}),
               ("stack trace crash", {"cat": "bug"}),
               ("segfault error report", {"cat": "bug"})]
    a = m.systemone_decide("refund the invoice charge", QS, labeled=labeled)
    # In (0,1) strictly: the PAV clip forbids claiming a certainty n=6 outcomes cannot support.
    assert a["cat"]["p"] is not None and 0.0 < a["cat"]["p"] < 1.0
    assert a["cat"]["calibrated"] is True


def test_pav_hand_case_and_monotone():
    xs, ys = _pav_fit([0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 1.0, 1.0])
    assert np.allclose(ys, [0.125, 0.125, 0.875, 0.875], atol=1e-12)
    xs2, ys2 = _pav_fit([0.1, 0.2, 0.3, 0.4], [1.0, 0.0, 1.0, 1.0])
    assert abs(ys2[0] - 0.5) < 1e-12 and np.all(np.diff(ys2) >= -1e-15)


def test_score_abstains_off_manifold_with_why():
    m = lecore.UnifiedMind(dim=512, seed=0)
    q = {"sev": {"type": "score", "min": 0, "max": 100, "anchors": [
        ("total outage meltdown broken", 95.0), ("cosmetic typo no rush", 5.0)]}}
    a = m.systemone_decide("qqq zzz unrelated nonsense tokens", q)
    assert a["sev"]["value"] is None and a["sev"]["abstained"] and "floor" in a["sev"]["why"]


# ---- sweep 172: the loop (min_support, observe, escalate, drift) ----

def test_min_support_floor_via_mind():
    """The Milanfar gap, closed: choice abstains off-manifold like score always did."""
    m = lecore.UnifiedMind(dim=512, seed=0)
    a = m.systemone_decide("qqq zzz unrelated nonsense", QS, min_support=0.35)
    assert a["cat"]["abstained"] and "min_support" in a["cat"]["why"]
    # default-off: without the floor, sweep-171 behaviour is unchanged (ranked still present)
    b = m.systemone_decide("qqq zzz unrelated nonsense", QS)
    assert "min_support" not in b["cat"].get("why", "")


def test_prequential_learning_beats_frozen_on_label_flip():
    """Controlled concept drift: the SAME vocabulary flips label mid-stream. Frozen prototypes
    keep answering yesterday's truth; the AdaptHD update must recover. Deterministic encoder,
    constructed schedule -- a mechanism pin, not a benchmark claim."""
    from holographic.agents_and_reasoning.holographic_systemone import _hash_bow_encode
    m = lecore.UnifiedMind(dim=512, seed=0)
    q = {"cat": {"type": "choice", "options": ["alpha", "beta"], "examples": {
        "alpha": ["red crimson scarlet"], "beta": ["blue azure navy"]}}}
    stream = [("red crimson scarlet item", {"cat": "alpha"})] * 10 \
           + [("red crimson scarlet item", {"cat": "beta"})] * 20
    enc = _hash_bow_encode()
    frozen = m.systemone_stream(stream, q, lr=0.0, encoder=enc)
    online = m.systemone_stream(stream, q, lr=1.0, encoder=enc)
    assert frozen["prequential_accuracy"]["cat"] < 0.5          # keeps the stale answer
    assert online["prequential_accuracy"]["cat"] > frozen["prequential_accuracy"]["cat"]
    assert online["drift"]["cat"]["correctness"]["status"] in ("ok", "insufficient")


def test_escalation_seam_holds_the_schema():
    m = lecore.UnifiedMind(dim=512, seed=0)
    so = m.systemone(QS, min_support=0.99)      # force abstention: everything escalates
    good = so.decide_or_escalate("whatever", escalate=lambda p: "bug")
    assert good["answers"]["cat"]["via"] == "escalated" and good["answers"]["cat"]["value"] == "bug"
    bad = so.decide_or_escalate("whatever", escalate=lambda p: "not-an-option")
    assert bad["answers"]["cat"]["via"] == "refused" and bad["answers"]["cat"]["value"] is None
    # the payload carries the substrate's evidence: the model end sees the prior
    seen = {}
    so.decide_or_escalate("whatever", escalate=lambda p: seen.update(p) or "bug")
    assert "evidence" in seen and seen["question"] == "cat"


def test_drift_report_two_channels_honest():
    from holographic.agents_and_reasoning.holographic_systemone import SystemOne, _hash_bow_encode
    so = SystemOne(_hash_bow_encode())
    so.fit({"cat": {"type": "choice", "options": ["a", "b"], "examples": {
        "a": ["one two three"], "b": ["four five six"]}}})
    so._support["cat"] = [0.8] * 40 + [0.2] * 40      # controlled collapse
    rep = so.drift_report()["cat"]
    assert rep["support"]["drift"] is True and rep["support"]["boundaries"] == [40]
    assert rep["correctness"]["status"] == "insufficient"   # no labels arrived: says so, no verdict


# ---------------- sweep 174: the nb scorer ----------------

def test_nb_scorer_contract_and_default_off():
    """scorer='nb' keeps the typed contract (posteriors sum to 1, abstention, determinism) and the
    DEFAULT stays the prototype path with no count table at all."""
    from holographic.agents_and_reasoning.holographic_systemone import (SystemOne, SchemaError,
                                                                        _hash_bow_encode)
    enc = _hash_bow_encode()
    q = {"cat": {"type": "choice", "options": ["billing", "shipping"], "examples": {
        "billing": ["card charged twice", "refund my invoice fee"],
        "shipping": ["parcel lost in transit", "courier delivery late"]}}}
    nb = SystemOne(enc, scorer="nb", margin=0.05); nb.fit(q)
    a = nb.decide("card charged a fee twice")["cat"]
    assert a["value"] == "billing" and abs(sum(s for _, s in a["ranked"]) - 1.0) < 1e-9
    assert nb.decide("parcel lost by the courier")["cat"]["value"] == "shipping"
    assert nb.decide("x")["cat"]["ranked"] == nb.decide("x")["cat"]["ranked"]
    default = SystemOne(enc); default.fit(q)
    assert default.scorer == "prototype" and default._nb == {}
    with pytest.raises(SchemaError):
        SystemOne(enc, scorer="bayes")


def test_nb_scorer_learns_by_counting():
    """observe() with scorer='nb' adds the state's tokens to the TRUE option; a novel phrase that
    was wrong becomes right after one observation, and lr=0 records without touching the table."""
    from holographic.agents_and_reasoning.holographic_systemone import SystemOne, _hash_bow_encode
    q = {"cat": {"type": "choice", "options": ["billing", "shipping"], "examples": {
        "billing": ["card charged twice"], "shipping": ["parcel lost"]}}}
    nb = SystemOne(_hash_bow_encode(), scorer="nb", margin=0.0); nb.fit(q)
    before = dict(nb._nb["cat"]["tot"])
    nb.observe("tracking shows no movement", {"cat": "shipping"}, lr=0.0)
    assert nb._nb["cat"]["tot"] == before                       # frozen baseline: nothing learned
    nb.observe("tracking shows no movement", {"cat": "shipping"}, lr=2.0)
    assert nb._nb["cat"]["tot"]["shipping"] > before["shipping"]
    assert nb.decide("tracking shows no movement")["cat"]["ranked"][0][0] == "shipping"


def test_mind_wiring_passes_scorer_through():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    q = {"cat": {"type": "choice", "options": ["billing", "shipping"], "examples": {
        "billing": ["card charged fee", "refund the invoice"],
        "shipping": ["package courier delivery", "parcel lost"]}}}
    r = m.systemone_decide("my card was charged twice", q, scorer="nb", margin=0.05)
    assert r["cat"]["value"] == "billing"
    # ("the parcel never came" abstains here with nb_transform: one known token in four is weak
    # evidence at two examples per class, and the length-normalised posterior says so honestly.)
    rows = m.systemone_map(["my card was charged twice", "parcel lost with the courier"], q,
                           scorer="nb", margin=0.05)
    assert [x["cat"]["value"] for x in rows] == ["billing", "shipping"]
    assert "prequential_accuracy" in m.systemone_stream(
        [("invoice charge", {"cat": "billing"})], q, scorer="nb")


# ---------------- sweep 175: the transform, and Banking77 ----------------

def test_nb_transform_is_default_and_flag_reproduces_sweep174():
    """nb_transform=True is the nb default (measured best on all three tasks); False must reproduce
    the sweep-174 untransformed table exactly (no IDF stored, raw counts)."""
    from holographic.agents_and_reasoning.holographic_systemone import SystemOne, _hash_bow_encode
    q = {"cat": {"type": "choice", "options": ["billing", "shipping"], "examples": {
        "billing": ["card charged twice", "refund my invoice fee"],
        "shipping": ["parcel lost in transit", "courier delivery late"]}}}
    on = SystemOne(_hash_bow_encode(), scorer="nb"); on.fit(q)
    off = SystemOne(_hash_bow_encode(), scorer="nb", nb_transform=False); off.fit(q)
    assert on.nb_transform is True and on._nb["cat"]["idf"] is not None
    assert off._nb["cat"]["idf"] is None
    assert off._nb["cat"]["tot"]["billing"] == 7.0          # raw word counts, untransformed
    assert on._nb["cat"]["tot"]["billing"] != 7.0           # transformed weights, not counts
    for phrase in ("card charged a fee twice", "parcel lost by the courier"):
        assert on.decide(phrase)["cat"]["ranked"][0][0] == off.decide(phrase)["cat"]["ranked"][0][0]


def test_banking77_cache_present_and_shaped():
    """The Jev-shaped task ships with the repo so the bench is reproducible offline."""
    import json, os
    path = os.path.join(os.path.dirname(__file__), "..", "tools", "_banking77_cache.json")
    d = json.load(open(path))
    assert d["n_labels"] == 77 and len(d["train"]) >= 10000 and len(d["eval"]) >= 3000
    assert all(0 <= y < 77 for _, y in d["eval"][:100])

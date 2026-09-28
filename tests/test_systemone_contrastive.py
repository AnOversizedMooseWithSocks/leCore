"""tests/test_systemone_contrastive.py -- SystemOne(scorer="contrastive"), the CLM backlog's E1.1 first consumer.

The scorer keeps one holographic_protostore.ProtoStore per choice / noul question, initialised EXACTLY like the
prototype scorer (the unit mean of each option's examples), and moves it by the shared InfoNCE rule on EVERY
labelled observation; its isotonic calibrator is fed p_top (tau 0.02) instead of the margin gap.

MEASURED (not asserted here -- tests must not need the datasets): Banking77 one pass, K = 5, 3,000 verdicts,
through SystemOne itself: 0.8263 / 0.8218 vs this class's own miss-only AdaptHD (scorer="prototype",
observe(lr=0.3)) 0.7253 / 0.7377 -- `tools/bench_contrastive.py --data DIR systemone`.

What IS pinned here, on a controlled vocabulary (mechanism, never a live calibration artifact):
  1. before any verdict the contrastive scorer decides exactly as the prototype scorer does;
  2. it learns from a CORRECT decision too (the prototype scorer only moves on a miss), lr=0 records only;
  3. the calibrator's feature is p_top at tau 0.02, and p is the isotonic map of it;
  4. its store writes the SAME arrays decide() reads, before and after a state() / from_state() round trip,
     and a reloaded scorer keeps learning bit-identically;
  5. the prototype and nb scorers carry no p_top and keep the margin gap as their feature (bit-identity with the
     code before this scorer existed was checked against a snapshot: one sha256 over decisions, observe outputs
     and state for prototype / nb / nb untransformed -- identical);
  6. the mind: systemone_decide(scorer="contrastive") -> decision_outcome -> learning_save / learning_load keeps the
     learned store and decides identically after a cold reload.
"""
import json
import os

os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np

from holographic.agents_and_reasoning.holographic_systemone import SchemaError, SystemOne, _hash_bow_encode

QS = {"cat": {"type": "choice", "options": ["billing", "shipping", "bug"], "examples": {
    "billing": ["invoice charge refund payment", "card charged twice billing"],
    "shipping": ["package delivery tracking courier", "parcel arrives late shipping"],
    "bug": ["crash stack trace error", "segfault bug crash report"]}}}
STREAM = [("card charged invoice %d" % i, "billing") for i in range(5)] + \
         [("parcel courier late %d" % i, "shipping") for i in range(5)] + \
         [("crash trace segfault %d" % i, "bug") for i in range(5)]


def _fit(**kw):
    so = SystemOne(_hash_bow_encode(), **kw)
    so.fit(QS)
    return so


def test_the_scorer_name_is_accepted_and_an_unknown_one_still_refused():
    assert SystemOne(_hash_bow_encode(), scorer="contrastive").scorer == "contrastive"
    try:
        SystemOne(_hash_bow_encode(), scorer="infonce")
        raise AssertionError("unknown scorer accepted")
    except SchemaError:
        pass


def test_before_any_verdict_it_decides_exactly_like_the_prototype_scorer():
    a, b = _fit(), _fit(scorer="contrastive")
    for s in ("my card was charged twice", "the parcel is late", "crash on start", "zzz qqq"):
        ra, rb = a.decide(s)["cat"], b.decide(s)["cat"]
        assert ra["ranked"] == rb["ranked"] and ra["value"] == rb["value"]
    # the store IS the SystemOne's tables (not a copy): what the rule writes is what decide() reads
    st = b._store["cat"]
    assert st.P is b._mat["cat"] and st.A is b._acc["cat"] and st.labels == ["billing", "shipping", "bug"]


def test_it_learns_from_a_narrow_win_and_lr_zero_records_only():
    proto, con, frozen = _fit(), _fit(scorer="contrastive"), _fit(scorer="contrastive")
    s = "card charged parcel late"                         # shipping wins by 0.0003 cosine: RIGHT, but barely
    assert con.decide(s)["cat"]["ranked"][0][0] == "shipping" and con.decide(s)["cat"]["p_top"] < 0.6
    m0p, m0c, m0f = proto._mat["cat"].copy(), con._mat["cat"].copy(), frozen._mat["cat"].copy()
    assert proto.observe(s, {"cat": "shipping"})["cat"] == {"was_correct": True, "updated": False}
    rc = con.observe(s, {"cat": "shipping"})["cat"]
    assert rc["was_correct"] is True and rc["updated"] is True
    assert np.array_equal(proto._mat["cat"], m0p)          # AdaptHD: a narrow win teaches nothing
    assert not np.array_equal(con._mat["cat"], m0c)        # InfoNCE: it pulls the truth and pushes the close rival
    assert con.decide(s)["cat"]["margin_gap"] > proto.decide(s)["cat"]["margin_gap"]
    assert frozen.observe(s, {"cat": "shipping"}, lr=0)["cat"]["updated"] is False
    assert np.array_equal(frozen._mat["cat"], m0f) and len(frozen._outcomes["cat"]) == 1
    # a CONFIDENT win moves nothing (p_truth ~ 1, so t - p ~ 0): the tau -> 0 limit of the same rule is AdaptHD
    m1 = con._mat["cat"].copy()
    assert con.observe("invoice charge refund payment again", {"cat": "billing"})["cat"]["updated"] is False
    assert np.array_equal(con._mat["cat"], m1)


def test_the_rule_flips_a_wrong_decision():
    so = _fit(scorer="contrastive")
    s = "invoice charge refund"                            # billing words, labelled shipping over and over
    assert so.decide(s)["cat"]["ranked"][0][0] == "billing"
    for _ in range(8):
        so.observe(s, {"cat": "shipping"})
    assert so.decide(s)["cat"]["ranked"][0][0] == "shipping"
    assert so._store["cat"].n_updates == 8


def test_the_calibrator_is_fed_p_top_not_the_margin_gap():
    so = _fit(scorer="contrastive")
    feats = []
    for s, t in STREAM:
        a = so.decide(s)["cat"]
        z = np.array([x for _, x in a["ranked"]]) / SystemOne.CONTRASTIVE_CAL_TAU
        z = np.exp(z - z.max())
        assert abs(a["p_top"] - z.max() / z.sum()) < 1e-12   # softmax weight of the top option at tau 0.02
        feats.append(a["p_top"])
        so.observe(s, {"cat": t})
    # the outcome stream records the feature observe() decided with (p_top), never the gap
    assert [g for g, _ in so._outcomes["cat"]] == feats
    a = so.decide("card charged twice invoice")["cat"]
    assert a["calibrated"] and a["p"] == so._calib["cat"].predict(a["p_top"])
    # the other scorers are untouched: no p_top, the margin gap stays their feature
    for kw in ({}, {"scorer": "nb"}):
        o = _fit(**kw)
        a = o.decide("card charged twice")["cat"]
        assert "p_top" not in a
        o.observe("card charged twice", {"cat": "billing"})
        assert o._outcomes["cat"][0][0] == a["margin_gap"]


def test_state_round_trip_keeps_the_store_live_and_learning_bit_identical():
    so = _fit(scorer="contrastive")
    for s, t in STREAM:
        so.observe(s, {"cat": t})
    meta, arrays = so.state()
    meta = json.loads(json.dumps(meta))                   # it must survive the container's JSON meta
    assert meta["contrastive"] == {"tau": 0.05, "lr": 0.3} and meta["q"]["0"]["store"]["n_updates"] == 15
    back = SystemOne.from_state(_hash_bow_encode(), so.questions, meta, arrays)
    for s in ("invoice refund", "courier late", "segfault", "zzz"):
        assert back.decide(s)["cat"] == so.decide(s)["cat"]
    st = back._store["cat"]
    assert st.P is back._mat["cat"] and st.A is back._acc["cat"] and st.n_updates == 15
    for x in (so, back):
        x.observe("refund my card charge", {"cat": "billing"})
    assert np.array_equal(back._mat["cat"], so._mat["cat"]) and np.array_equal(back._acc["cat"], so._acc["cat"])


def test_a_noul_question_learns_the_string_no_as_no():
    so = SystemOne(_hash_bow_encode(), scorer="contrastive")
    so.fit({"ok": {"type": "noul", "examples": {"yes": ["approve ship now"], "no": ["reject deny request"]}}})
    for _ in range(6):
        so.observe("deny it please", {"ok": "no"})
    assert so.decide("deny it please")["ok"]["value"] is False
    try:
        so.observe("deny it", {"ok": "maybe"})
        raise AssertionError("a non-yes/no noul truth was learned")
    except SchemaError:
        pass


def test_the_mind_door_learns_persists_and_reloads(tmp_path):
    import lecore
    q = {"intent": {"type": "choice", "options": ["billing", "shipping"], "examples": {
        "billing": ["card charged twice", "refund my invoice fee", "charge on my statement"],
        "shipping": ["parcel lost in transit", "courier delivery late", "package never arrived"]}}}
    m = lecore.UnifiedMind(dim=256, seed=0)
    a = m.systemone_decide("my card shows a double charge", q, scorer="contrastive", encoder="ngram")["intent"]
    rep = m.decision_outcome(a["id"], "billing")
    assert rep["forwarded"]["updated"] is True                  # a CORRECT verdict still trains this door
    so = next(iter(m._systemone_cache.values()))
    assert so.scorer == "contrastive" and so._store["intent"].n_updates == 1
    before = m.systemone_decide("where is my parcel", q, scorer="contrastive", encoder="ngram")["intent"]
    root = str(tmp_path / "p")
    m.learning_save(root)
    m2 = lecore.UnifiedMind(dim=256, seed=0)
    m2.learning_load(root)
    so2 = next(iter(m2._systemone_cache.values()))
    assert so2.scorer == "contrastive" and so2._store["intent"].n_updates == 1
    assert so2._store["intent"].P is so2._mat["intent"]
    after = m2.systemone_decide("where is my parcel", q, scorer="contrastive", encoder="ngram")["intent"]
    assert after["ranked"] == before["ranked"]

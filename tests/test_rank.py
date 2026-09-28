"""tests/test_rank.py -- mind.rank(state, candidates): free-form System One ranking (CLM backlog E3.3, wave 2).

Pins the contract: ONE typed record (value, ranked, margin, p_correct, p_null, set, id, evidence); the ORDER is the
absolute cosine with the given order as the tie rule; an absolute floor refuses; candidate encodings are cached by
sha256 of the text and caching never changes a result; the encoder is hashed_ngram_encode's space bit for bit with a
bounded atom cache; decision_outcome(id, truth) feeds the rank door's calibrator, its ProtoStore (restricted to the
call's candidates) and the conformal stream -- and survives a restart; a secret is never learned or persisted; rank
outcomes never write the shared reflex trace. Numbers on real wording live in tools/bench_rank.py. Hand-written data.
"""
import hashlib
import json
import os

import numpy as np
import pytest

os.environ.setdefault("PYTHONHASHSEED", "0")

CANDS = [{"id": "billing", "text": "billing", "examples": ["my card was charged twice", "refund the invoice fee"]},
         {"id": "shipping", "text": "shipping", "examples": ["where is my parcel", "the courier lost the package"]},
         {"id": "account", "text": "account", "examples": ["reset my password", "change my email address"]},
         {"id": "cancel", "text": "cancel order", "examples": ["cancel my order please", "i want to stop the order"]}]


def _mind():
    import lecore
    return lecore.UnifiedMind(dim=256, seed=0)


def test_rank_returns_one_typed_record_best_first():
    m = _mind()
    r = m.rank("the courier never delivered my package", CANDS)
    assert r["via"] == "rank" and r["value"] == "shipping" and r["abstained"] is False
    scores = [s for _, s in r["ranked"]]
    assert [l for l, _ in r["ranked"]][0] == "shipping" and scores == sorted(scores, reverse=True)
    assert r["margin"] == pytest.approx(scores[0] - scores[1]) and r["top"] == pytest.approx(scores[0])
    assert r["p_correct"] is None and r["set"] is None                  # uncalibrated doors say so
    assert 0.0 < r["p_null"] <= 1.0 and r["evidence"]["null"]["n"] == 64
    rec = m.decision_ledger().get(r["id"])
    assert rec.via == "rank" and rec.options[0] == "shipping" and rec.meta["hook"] == {"kind": "rank"}
    assert json.loads(json.dumps(r))["value"] == "shipping"              # a plain dict on the wire
    assert "p" not in r                                                  # no deprecated bare p on a new door


def test_ties_keep_the_given_order_and_k_cuts_the_list():
    m = _mind()
    r = m.rank("zzz", ["alpha one", "alpha one ", "beta"], floor=-1.0, k=2)   # identical encodings tie
    assert [l for l, _ in r["ranked"]] == ["alpha one", "alpha one "] or r["ranked"][0][1] > r["ranked"][1][1]
    assert len(r["ranked"]) == 2


def test_the_floor_refuses_but_still_ranks():
    m = _mind()
    r = m.rank("purple elephants dance quietly", CANDS)
    assert r["value"] is None and r["abstained"] is True and "floor" in r["why"]
    assert len(r["ranked"]) == 4 and r["id"] is not None               # the ranking and the record still exist
    assert m.rank("purple elephants dance quietly", CANDS, floor=-1.0)["value"] is not None


def test_candidates_are_validated():
    m = _mind()
    for bad in ([], "just a string", [""], ["a", "a"], [{"id": "x"}], [3]):
        with pytest.raises((ValueError, TypeError)):
            m.rank("state", bad)
    with pytest.raises(ValueError):
        m.rank("", ["a", "b"])
    r = m.rank("reset my password", ["billing", "reset password"])   # plain strings: the label is the text
    assert r["value"] == "reset password"


def test_the_cache_is_keyed_by_sha256_of_the_text_and_never_changes_a_result():
    m = _mind()
    on1 = m.rank("i was charged twice", CANDS, record=False)
    on2 = m.rank("i was charged twice", CANDS, record=False)
    off = m.rank("i was charged twice", CANDS, record=False, cache=False)
    assert on1["evidence"]["cache"]["encoded"] == 4 and on2["evidence"]["cache"]["hits"] == 4
    assert off["evidence"]["cache"]["encoded"] == 4 and off["ranked"] == on1["ranked"] == on2["ranked"]
    key = hashlib.sha256(json.dumps({"text": "billing", "examples": CANDS[0]["examples"]}, sort_keys=True)
                         .encode()).hexdigest()
    assert key in m._rank_state()["cands"]
    m.rank("x", ["plain candidate text"], floor=-1, record=False)
    assert hashlib.sha256(b"plain candidate text").hexdigest() in m._rank_state()["cands"]


def test_the_bounded_encoder_is_hashed_ngram_encode_bit_for_bit():
    from holographic.agents_and_reasoning.holographic_systemone import hashed_ngram_encode
    from holographic.unified.holographic_unified_p29_contrastive import _bounded_ngram
    ref = hashed_ngram_encode(dim=2048)
    tiny = _bounded_ngram(cap=8)                                     # constant eviction: results must not move
    for t in ("the courier lost my parcel", "Refund THE invoice!", "a"):
        assert np.array_equal(tiny(t), ref(t))
    assert len(tiny.cache) <= 8 and tiny.stats["misses"] > 8


def test_outcomes_calibrate_the_door_and_learning_fixes_a_wrong_ranking():
    """A paraphrase the static texts rank wrong ('my shipment is missing' reads nearer 'cancel order' / 'account'
    than 'shipping'?) -- after a few reported truths the rank door's prototypes rank it right, and p_correct stops
    being None once both outcome kinds were seen."""
    m = _mind()
    stream = [("my shipment went missing", "shipping"), ("lost shipment again", "shipping"),
              ("the shipment is missing", "shipping"), ("double charge on my card", "billing"),
              ("please cancel it, i changed my mind", "cancel"), ("forgot my login password", "account"),
              ("shipment missing for a week", "shipping"), ("charged two times this month", "billing")]
    first = m.rank("where did my shipment go missing", CANDS, record=False)["ranked"]
    for state, truth in stream * 2:
        r = m.rank(state, CANDS)
        rep = m.decision_outcome(r["id"], truth)
        assert rep["door"] == "rank" and rep["forwarded"]["learned"] is True
        assert rep["reflex"]["learned"] is False                     # never the shared trace
    after = m.rank("where did my shipment go missing", CANDS, record=False)
    assert after["value"] == "shipping" and after["evidence"]["learned_rows"] >= 1
    assert dict(after["ranked"])["shipping"] > dict(first)["shipping"]
    cal = m.door_calibration_report()["rank"]
    assert cal["labels"] == 16
    if 0 < cal["correct"] < cal["labels"]:
        assert after["p_correct"] is not None
    assert not getattr(m, "_reflex_labels", None)                    # no candidate label entered the reflex book


def test_the_conformal_set_appears_after_eight_truths_and_holds_the_truth():
    m = _mind()
    stream = [("card charged twice", "billing"), ("parcel lost", "shipping"), ("reset password", "account"),
              ("cancel my order", "cancel")] * 3
    assert m.rank("parcel lost", CANDS)["set"] is None
    for st, truth in stream:
        m.decision_outcome(m.rank(st, CANDS)["id"], truth)
    r = m.rank("the courier lost my parcel", CANDS)
    assert r["set"] is not None and "shipping" in r["set"] and r["set"][0] == r["ranked"][0][0]
    assert len(m.rank("the courier lost my parcel", CANDS, alpha=0.5)["set"]) <= len(r["set"])


def test_p_null_is_lower_for_a_real_match_than_for_salad():
    m = _mind()
    good = m.rank("the courier lost the package", CANDS, record=False)
    salad = m.rank("purple quantum giraffe", CANDS, record=False, floor=-1)
    assert good["p_null"] < salad["p_null"]
    again = m.rank("the courier lost the package", CANDS, record=False)
    assert again["evidence"]["null"]["built"] is False and again["p_null"] == good["p_null"]   # cached per set/length


def test_the_rank_door_survives_a_restart(tmp_path):
    """The learned rows, the candidate-set registry and the conformal stream travel in the decisions section: a
    decision after the reload ranks exactly as before it, and an id issued before the restart still learns."""
    m = _mind()
    for st, truth in [("my shipment went missing", "shipping"), ("double charge on my card", "billing")] * 2:
        m.decision_outcome(m.rank(st, CANDS)["id"], truth)
    pending = m.rank("lost shipment again", CANDS)
    before = m.rank("where did my shipment go missing", CANDS, record=False)["ranked"]
    root = str(tmp_path / "partition")
    m.learning_save(root)
    m2 = _mind()
    rep = m2.learning_load(root)
    assert rep["decisions"]["protostores"] >= 1
    assert m2.rank("where did my shipment go missing", CANDS, record=False)["ranked"] == before
    out = m2.decision_outcome(pending["id"], "shipping")
    assert out["door"] == "rank" and out["forwarded"]["learned"] is True


def test_a_secret_is_never_learned_or_persisted(tmp_path):
    m = _mind()
    secret = "my password: Hunter2-FAKE-9c1d please reset it"
    r = m.rank(secret, CANDS)
    rep = m.decision_outcome(r["id"], "account")
    assert rep["forwarded"]["learned"] is False and "secret" in rep["forwarded"]["why"]
    opt = m.rank("anything", ["password: Hunter2-FAKE-9c1d", "billing"], floor=-1)   # a secret as an OPTION
    root = str(tmp_path / "p")
    m.learning_save(root)
    blob = json.dumps(m._decisions_section()["meta"])
    assert "Hunter2" not in blob and opt["id"] is not None

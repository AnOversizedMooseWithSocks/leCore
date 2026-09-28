"""tests/test_arc_substrate.py -- backlog Phase A (+ E5.3): the reflex arc's SUBSTRATE defects the CLM panel found.

Each test pins one fix, with the defect it closes (docs/TYPED_DECISIONS.md section 8d has the numbers):
  E0.5  decisions survive a restart: decision_outcome(<an id issued before learning_rollover + learning_save and a
        COLD reload in a fresh mind>) used to raise KeyError, and SystemOne's learned tables were lost; a hook that
        raised skipped the reflex; the string "no" was learned as YES by a yes/no decision
  E0.6  one meaning of p: p_correct (calibrated P(correct), HIGH = confident) and p_null (a null p-value, LOW =
        significant) on every door's record; the bare p still reads (deprecated, with a warning) and still JSONs
  E0.7  calibration streams per door: the ladder's veto no longer moves when the reflex bridge reports, and it is
        live right after a reload (it used to wait for 8 new reports)
  E1.5  a correction is READ BACK: the reflex trace unlearns the wrong answer by a verbatim-replayed raw delta
  E5.3  verify_decision can vouch for a meaning serve that was confirmed by decision_outcome
All data is hand-written; every secret is FAKE.
"""
import json
import os
import warnings

import pytest

os.environ.setdefault("PYTHONHASHSEED", "0")

Q = {"cat": {"type": "choice", "options": ["billing", "shipping"],
             "examples": {"billing": ["card charged twice", "refund my invoice fee"],
                          "shipping": ["parcel lost in transit", "courier delivery late"]}}}
STREAM = [("my card was charged twice this month", "billing"), ("the courier never showed up", "shipping"),
          ("refund the double charge on my invoice", "billing"), ("where is my parcel, tracking is frozen", "shipping"),
          ("i was billed twice for one order", "billing"), ("the package arrived late and damaged", "shipping")]


def _mind():
    import lecore
    return lecore.UnifiedMind(dim=256, seed=0)


def _typed(m, state, q=Q, **kw):
    return m.systemone_decide(state, q, scorer="nb", encoder="ngram", **kw)[next(iter(q))]


# =================================================================================================== E0.5 ======

def test_an_old_decision_id_is_reported_after_rollover_save_and_a_cold_reload(tmp_path):
    """THE ACCEPTANCE: an id issued before the restart -- and reported after it, in a FRESH mind -- trains the same
    door. Before E0.5 this raised KeyError (the ledger lived in process memory only)."""
    m = _mind()
    for st, truth in STREAM:
        m.decision_outcome(_typed(m, st)["id"], truth)
    pending = _typed(m, "my card got charged two times")["id"]      # issued, NOT reported before the restart
    reported = _typed(m, STREAM[1][0])["id"]
    root = str(tmp_path / "partition")
    m.learning_rollover(root)
    m.learning_save(root)
    m2 = _mind()
    boot = m2.learning_rollover(root)                                # what a cold boot does
    assert boot["rolled"] is True
    rep = m2.decision_outcome(pending, "billing")                    # KeyError before E0.5
    assert rep["was_correct"] is not None and rep["forwarded"]["updated"] is True   # the SystemOne learned it
    assert m2.decision_ledger().get(reported).outcome == "shipping"  # a reported outcome survived too


def test_the_rollover_merges_decisions_from_older_generations(tmp_path):
    """learning_rollover full-loads the NEWEST generation; an older generation's decision records (durable text,
    like taught rows) are merged in too -- newest wins by id, older records sort first."""
    root = tmp_path / "p"
    (root / "learning").mkdir(parents=True)
    old_m, new_m = _mind(), _mind()
    a = _typed(old_m, STREAM[0][0])["id"]
    b = _typed(new_m, STREAM[1][0])["id"]
    old_m.learning_save(str(root), path=str(root / "learning" / "state-20260101-000000Z.lecore"))
    new_m.learning_save(str(root), path=str(root / "learning" / "state-20260102-000000Z.lecore"))
    m = _mind()
    rep = m.learning_rollover(str(root))
    assert rep["rolled"] is True and rep["decisions_merged"] == 1
    L = m.decision_ledger()
    assert L.get(a) is not None and L.get(b) is not None and L._order.index(a) < L._order.index(b)
    assert m.decision_outcome(a, "billing")["forwarded"]["updated"] is True


def test_systemone_learned_tables_survive_a_reload(tmp_path):
    """A typed decision made after the reload equals the one made before it (bit-identical ranking), and differs
    from a fresh fit -- i.e. the lessons came back, not just the schema."""
    m = _mind()
    for _ in range(2):
        for st, truth in STREAM:
            m.decision_outcome(_typed(m, st)["id"], truth)
    probes = [s for s, _ in STREAM] + ["card charged again, parcel late"]
    before = [_typed(m, s) for s in probes]
    root = str(tmp_path / "p")
    m.learning_save(root)
    m2 = _mind()
    rep = m2.learning_load(root)
    assert rep["decisions"]["systemone"] == 1 and not rep["decisions"]["systemone_refused"]
    after = [_typed(m2, s) for s in probes]
    assert [a["ranked"] for a in before] == [a["ranked"] for a in after]
    assert [a["p_correct"] for a in before] == [a["p_correct"] for a in after]
    fresh = [_typed(_mind(), s) for s in probes]
    assert [a["ranked"] for a in fresh] != [a["ranked"] for a in after]


def test_a_record_carrying_a_secret_is_never_persisted(tmp_path):
    from holographic.io_and_interop.holographic_container import load_container
    m = _mind()
    secret = _typed(m, "my password is Hunter2-FAKE-9c1d and my card was charged twice")
    ok = _typed(m, "the courier never showed up")
    m.decision_outcome(ok["id"], "shipping")
    m.decision_outcome(secret["id"], "billing")                      # the typed hook refuses to learn it, too
    so = next(iter(m._systemone_cache.values()))
    assert "hunter2" not in so._nb["cat"]["vocab"]
    out = m.learning_save(str(tmp_path / "p"))
    got = load_container(open(out["path"], "rb").read())
    blob = json.dumps([s["meta"] for s in got["sections"]], default=str)
    assert "Hunter2" not in blob
    sec = next(s for s in got["sections"] if s["kind"] == "lecore.learning.decisions")
    assert sec["meta"]["dropped_sensitive"] >= 1
    m2 = _mind()
    m2.learning_load(str(tmp_path / "p"))
    assert m2.decision_ledger().get(ok["id"]) is not None and m2.decision_ledger().get(secret["id"]) is None


def test_a_hook_that_raises_is_reported_and_the_reflex_still_learns():
    """SystemOne.observe raises SchemaError for an outcome that is not an option. Before E0.5 the exception escaped
    Ledger.report and decision_outcome never reached reflex_learn."""
    m = _mind()
    a = _typed(m, "the courier never showed up")
    n0 = len(getattr(m, "_reflex_seen", None) or [])
    rep = m.decision_outcome(a["id"], "wrong")                      # not an option of the schema
    assert "SchemaError" in rep["hook_error"] and rep["reflex"]["learned"] is True
    assert len(m._reflex_seen) == n0 + 1


def test_the_string_no_teaches_a_yes_no_decision_no():
    """The noul fix: observe computed want = "yes" if truth else "no", so the STRING "no" (truthy) was learned as
    yes."""
    m = _mind()
    qn = {"ok": {"type": "noul", "examples": {"yes": ["approve and ship it now"], "no": ["reject and deny this"]}}}
    a = _typed(m, "deny that request please", q=qn)
    so = next(iter(m._systemone_cache.values()))
    before = dict(so._nb["ok"]["tot"])
    rep = m.decision_outcome(a["id"], "no")
    assert so._nb["ok"]["tot"]["no"] > before["no"] and so._nb["ok"]["tot"]["yes"] == before["yes"]
    # the record's answer is the bool the door returned: "no" CONFIRMS False (it used to count as a correction)
    assert a["value"] is False and rep["was_correct"] is True and rep["reflex"]["was_correct"] is True


# =================================================================================================== E0.6 ======

def _monotone(cal, lo, hi, n=25):
    """(is p non-decreasing over [lo, hi], the p's) for a DoorCalibrator or an IsotonicCalibrator."""
    f = cal.p_correct if hasattr(cal, "p_correct") else cal.predict
    xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    ps = [f(x) for x in xs]
    return all(b >= a - 1e-12 for a, b in zip(ps, ps[1:])), ps


def test_the_bare_p_is_deprecated_but_still_reads_and_serialises():
    from holographic.agents_and_reasoning.holographic_decisionrecord import DecisionDict
    m = _mind()
    r = m.route_tiered("smooth a bumpy mesh")
    assert {"p_correct", "p_null"} <= set(r) and r["p_null"] == dict.get(r, "p")   # the null p-value moved
    DecisionDict._warned.clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = (r["p_correct"], r.get("p_null"), "p" in r, dict(r))
        assert not [x for x in w if issubclass(x.category, DeprecationWarning)]
    with pytest.warns(DeprecationWarning):
        assert r["p"] == r["p_null"]
    assert json.loads(json.dumps(r, default=str))["p_null"] == r["p_null"]
    # the HTTP service's wire conversion walks items() -- it neither warns nor drops the new keys
    from holographic_service import _jsonable
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        wire = _jsonable(r)
    assert wire["p_null"] == r["p_null"] and "p_correct" in wire and type(wire) is dict


def test_route_p_correct_is_high_when_confident():
    """route_tiered: the catalog's p is a NULL p-value (low = confident) and now travels as p_null; p_correct comes
    from the route door's own calibrator, fed by decision_outcome -- here a caller who used the top capability when
    the route was confident (z >= 0) and the runner-up when it was not."""
    m = _mind()
    qs = ["smooth a bumpy mesh", "grow crystals on a surface", "shortest path", "compress a video",
          "signed distance field", "render a scene with path tracing",
          "fft spectral analysis", "purple monkey dishwasher", "the weather is nice today", "my cat likes boxes",
          "please sort out stuff", "banana telephone orbit"]
    first = [m.route_tiered(q) for q in qs]
    assert all(r["p_correct"] is None for r in first)               # uncalibrated door: None, not a guess
    for r in first:
        top = r["answer"] or r["options"][0]["name"]
        m.decision_outcome(r["id"], top if r["z"] >= 0 else r["options"][1]["name"])
    cal = m.door_calibrator("route")
    ok, ps = _monotone(cal, -2.0, 4.0)
    assert ok and ps[-1] > 0.8 and ps[0] < 0.2
    hi, lo = m.route_tiered("smooth a bumpy mesh"), m.route_tiered("my cat likes boxes")
    assert hi["p_correct"] > 0.8 > 0.2 > lo["p_correct"]            # HIGH = confident
    assert hi["p_null"] < lo["p_null"]                              # ...while the null p-value is LOW = significant


def test_typed_p_correct_is_high_when_confident():
    """typed: SystemOne's isotonic calibrator (fed by decision_outcome through the re-derived hook) is p_correct."""
    m = _mind()
    for _ in range(2):
        for st, truth in STREAM:
            m.decision_outcome(_typed(m, st)["id"], truth)
    for st, truth in [("card", "shipping"), ("late", "billing")]:     # two ambiguous states the door got wrong
        m.decision_outcome(_typed(m, st)["id"], truth)
    so = next(iter(m._systemone_cache.values()))
    cal = so._calib["cat"]
    ok, ps = _monotone(cal, min(cal.xs), max(cal.xs))
    assert ok and ps[-1] > ps[0]
    a = _typed(m, "my card was charged twice this month")
    assert a["p_correct"] == pytest.approx(cal.predict(a["margin_gap"])) and a["p_correct"] >= ps[0]
    assert m.decision_ledger().get(a["id"]).p_correct == a["p_correct"]


def test_reflex_p_correct_is_high_when_confident():
    """The reflex bridge: p_correct from door_calibrator('bridge'), fed ONLY by reported reflex answers. A real
    stream: the reflex's first answers are corrected (the caller used another capability), later ones confirmed."""
    m = _mind()
    names = ["smooth a bumpy mesh", "grow crystals on a surface", "shortest path", "compress a video"]
    for n in names:
        r = m.route_tiered(n, reflex=True)
        m.decision_outcome(r["id"], r["answer"])
    served = []
    for rnd in range(3):
        for n in names:
            r = m.route_tiered(n, reflex=True)
            assert r["via"] == "reflex"
            served.append(r)
            m.decision_outcome(r["id"], "some other capability" if rnd == 0 else r["answer"])
    cal = m.door_calibrator("bridge")
    assert cal.calibrated() and len(cal.pairs) == 12
    confs = [c for c, _ in cal.pairs]
    ok, ps = _monotone(cal, min(confs), max(confs))
    assert ok and ps[-1] > ps[0]                                      # HIGH = confident
    assert served[0]["p_correct"] is None                            # uncalibrated at first: None, not a guess
    rf = m.reflex_decide("smooth a bumpy mesh")
    assert rf["p_correct"] == m._door_p("bridge", rf["confidence"]) is not None
    assert rf["error_prob"] == pytest.approx(1.0 - rf["p_correct"])
    assert m.decision_ledger().get(rf["id"]).p_correct == rf["p_correct"]


def test_reflex_door_monotone_on_a_scripted_bridge():
    """The bridge calibrator itself, fed the (confidence, correct) labels reflex_learn used to push into the shared
    list: high confidence right, low wrong -> p_correct rises with confidence (HIGH = confident)."""
    m = _mind()
    cal = m.door_calibrator("bridge")
    for i in range(16):
        cal.observe(0.05 * i, i >= 8)
    ok, ps = _monotone(cal, 0.0, 0.75)
    assert ok and ps[-1] > 0.8 and ps[0] < 0.2
    assert m.reflex_error_prob(0.7, door="bridge") == pytest.approx(1.0 - cal.p_correct(0.7))


def test_meaning_p_correct_is_high_when_confident():
    m = _mind()
    m.teach("how do i check my account balance", "Open the app and tap Accounts.")
    rid = list(m.meaning.rows)[0]
    for k in range(24):
        m.meaning.observe(1.0 + k * 0.05, True)
        m.meaning.observe(0.1 + k * 0.02, False)
    m.meaning_resolve("how much money do i have left", {"verdict": "same", "row": rid})
    out = m.ask("how much money do i have left")
    assert out["via"] == "meaning" and out["p_correct"] is not None and out.get("p_null") is None
    assert m.meaning.p_correct(2.0) > 0.8 > 0.2 > m.meaning.p_correct(0.1)
    assert m.decision_ledger().get(out["id"]).p_correct == out["p_correct"]


def _tool_mind():
    m = _mind()
    m.api_use = lambda service, endpoint, params=None, headers=None: {"ok": True, "data": {"endpoint": endpoint}}
    m.tool_reflex_teach("convert fahrenheit temperature into celsius degrees units", "convertd", "f_to_c")
    return m


PATTERN_WORDS = {"convert", "fahrenheit", "temperature", "into", "celsius", "degrees", "units"}


def _judge(q, tool, data):
    """A scripted verifier: the conversion was the right tool when the request shared 3+ words with its pattern."""
    import re
    return len(set(re.findall(r"[a-z]{4,}", q.lower())) & PATTERN_WORDS) >= 3


def test_the_tool_reflex_has_a_score_and_p_correct_is_high_when_confident():
    """serve()'s tool reflex had no margin and no p. Its score is the word overlap that picked the tool; a judged
    call is a label for door_calibrator('tool:overlap') -- a verified call through decision_outcome, a failed
    verdict straight to the calibrator (no outcome: a 'failed' outcome would paint the failure field)."""
    m = _tool_mind()
    strong = ["please convert this fahrenheit reading to celsius degrees", "what is 70 fahrenheit in celsius degrees",
              "fahrenheit to celsius for the oven temperature", "oven temperature fahrenheit celsius chart"]
    weak = ["convert units please", "temperature units now", "degrees units thanks", "convert celsius maybe"]
    for q in strong + weak:
        r = m.serve(q, verify=_judge)
        assert r["via"] == "tool-reflex" and r["picked_by"] == "overlap" and r["score"] >= 2 and "p_correct" in r
    cal = m.door_calibrator("tool:overlap")
    ok, ps = _monotone(cal, 2.0, 5.0)
    assert cal.calibrated() and ok and ps[-1] > 0.8 > 0.2 > ps[0]
    hi = m.serve("boil water fahrenheit versus celsius degrees")
    lo = m.serve("units convert tomorrow")
    assert hi["picked_by"] == lo["picked_by"] == "overlap" and hi["score"] > lo["score"]
    assert hi["p_correct"] > 0.8 > 0.2 > lo["p_correct"]           # HIGH = confident


def test_the_typed_reflex_path_keeps_a_zero_p_correct():
    """p08's `err is not None and (1 - err) or None` turned a calibrated 0.0 into None. p_correct is read now."""
    m = _mind()
    st, truth = STREAM[0]
    a = _typed(m, st)
    m.decision_outcome(a["id"], truth)
    m._door_p = lambda door, score: 0.0 if door == "bridge" else None
    b = _typed(m, st, reflex=True)
    assert b["via"] == "reflex" and b["p_correct"] == 0.0 and b["value"] == truth


# =================================================================================================== E0.7 ======

TAUGHT = [("what colour is the sky on mars", "butterscotch by day"), ("who wrote the lever seven module", "the owner"),
          ("which service answers the bus", "the leCore service"), ("why are tiles kept small", "readback falls off a cliff"),
          ("which encoder keys typed states", "hashed n-grams"), ("what holds the reflex lessons", "the experience trace"),
          ("who sat on the panel", "ten seats in four workers"), ("what does p_null mean", "a null p-value"),
          ("what does p_correct mean", "calibrated probability of correct"), ("where do plugins live", "in the plugins folder")]
# (answers are multi-word and carry no number: a bare 6+ char token or a measurement-shaped answer sends the learning
#  guard to its SEMANTIC layer, whose one-time build costs ~8-19 s -- tests/test_learnguard.py's subject, not this one's)


def _ladder_fed(m, bad_every=3):
    """Ladder feedback the way a person gives it: ask each taught question, mark every third answer bad."""
    for q, a in TAUGHT:
        m.teach(q, a)
    for i, (q, _) in enumerate(TAUGHT):
        m.ask(q)
        m.answer_feedback(q, ok=(i % bad_every != 0))
    return m


def test_the_ladder_calibration_does_not_move_when_the_bridge_reports():
    m = _ladder_fed(_mind())
    lad = m.door_calibrator("ladder")
    snap = (list(lad.pairs), [lad.p_correct(c / 10.0) for c in range(11)],
            [m.zoo["ladder"]._error_prob(c / 10.0) for c in range(11)])
    assert len(snap[0]) == len(TAUGHT)
    for n in ("smooth a bumpy mesh", "grow crystals on a surface"):   # the BRIDGE now reports, right and wrong
        r = m.route_tiered(n, reflex=True)
        m.decision_outcome(r["id"], r["answer"])
        r = m.route_tiered(n, reflex=True)
        m.decision_outcome(r["id"], "some other capability")
    assert len(m.door_calibrator("bridge").pairs) >= 2
    after = (list(lad.pairs), [lad.p_correct(c / 10.0) for c in range(11)],
             [m.zoo["ladder"]._error_prob(c / 10.0) for c in range(11)])
    assert after == snap


def test_the_ladder_veto_is_live_right_after_a_reload(tmp_path):
    m = _ladder_fed(_mind(), bad_every=1)      # every served answer was marked bad: the ladder should not be trusted
    m.door_calibrator("ladder").observe(0.99, True)                  # one good label: both outcomes, calibrated
    m.door_calibrator("ladder").observe(0.98, True)
    assert m.door_calibrator("ladder").calibrated()
    m.learning_save(str(tmp_path / "p"))
    m2 = _mind()
    m2.learning_load(str(tmp_path / "p"))
    ep = getattr(m2.zoo["ladder"], "_error_prob", None)
    assert ep is not None and ep(0.5) == pytest.approx(m.zoo["ladder"]._error_prob(0.5)) and ep(0.5) > 0.5
    assert m2.door_calibrator("ladder").pairs == m.door_calibrator("ladder").pairs
    # ...and it VETOES: a near-exact rewording (the trace's fuzzy arm, not the exact sidecar) is refused by the
    # reflex rung (the meaning rung may still serve it -- that door has its own calibration)
    q = "what is the name of the new release"
    m2.teach(q, "the tessellate release")
    v0 = m2.zoo["ladder"].ledger.by_tier.get("calib_veto", 0)
    r2 = m2.ask(q + " ?")
    assert m2.zoo["ladder"].ledger.by_tier.get("calib_veto", 0) == v0 + 1 and r2.get("via") != "reflex"
    fresh = _mind()                                                  # control: an uncalibrated ladder serves it
    fresh.teach(q, "the tessellate release")
    assert fresh.ask(q + " ?").get("via") == "reflex" and not fresh.zoo["ladder"].ledger.by_tier.get("calib_veto")


def test_a_legacy_untagged_calibration_migrates_into_both_streams():
    m = _mind()
    legacy = [[0.1 * i, i >= 5] for i in range(10)]
    rep = m._calibration_restore({"pairs": legacy, "fit": {"conf": [0.0], "err": [0.5]}})
    assert rep["format"] == "legacy" and rep["migrated_pairs"] == 10
    for door in ("ladder", "bridge"):
        assert [list(p) for p in m.door_calibrator(door).pairs] == [[float(c), float(y)] for c, y in legacy]
    assert m.zoo["ladder"]._error_prob(0.95) < 0.5 < m.zoo["ladder"]._error_prob(0.05)


# =================================================================================================== E1.5 ======

@pytest.mark.parametrize("design", ["lms_apa", "provenance"])
def test_a_correction_is_read_back_and_survives_retile_and_reload(tmp_path, design):
    """A wrong lesson at a key (a noisy verdict), then the correction: the reflex answers the TRUTH afterwards, and
    keeps answering it after a re-tile and a save/load (the raw correction replays verbatim)."""
    m = _mind()
    m.reflex_correction_mode(design)
    for other, lab in (("refund the double charge on my invoice", "billing"), ("where is my parcel", "shipping")):
        m.decision_outcome(_typed(m, other)["id"], lab)          # both labels are known to the reflex's codebook
    st = "the courier never showed up"
    a = _typed(m, st)
    m.decision_outcome(a["id"], "billing")                       # the noisy verdict: learned as billing
    first = m.reflex_decide(st, key="ngram")
    assert first["value"] == "billing"
    m.decision_outcome(first["id"], "shipping")                  # the correction, against the REFLEX's record
    assert m.reflex_decide(st, key="ngram")["value"] == "shipping"
    assert sum(len(t._audit_raw) for t in m.experience.tiles) >= 1
    m.reflex_retile(advisory_load=0.03)
    assert m.reflex_decide(st, key="ngram")["value"] == "shipping"
    m.learning_save(str(tmp_path / "p"))
    m2 = _mind()
    m2.reflex_correction_mode(design)
    m2.learning_load(str(tmp_path / "p"))
    assert sum(len(t._audit_raw) for t in m2.experience.tiles) >= 1
    assert m2.reflex_decide(st, key="ngram")["value"] == "shipping"


def test_a_reloaded_trace_keeps_the_mind_tile_size():
    """experience_from_state rebuilt the tiled trace WITHOUT advisory_load=0.03, so tiles split after a reload
    reverted to the old 0.10 (split at n=205, past the readback cliff)."""
    m = _mind()
    m.reflex_write([1.0] + [0.0] * 2047, [0.0, 1.0] + [0.0] * 2046)
    m2 = _mind()
    m2.experience_from_state({"tiles": [t.to_state() for t in m.experience.tiles]})
    assert m2.experience.kw.get("advisory_load") == 0.03


# =================================================================================================== E5.3 ======

def test_verify_decision_vouches_for_a_confirmed_meaning_serve():
    m = _mind()
    m.teach("how do i check my account balance", "Open the app and tap Accounts.")
    m.teach("how do i change my pin", "Settings > Card > Change PIN.")
    rid = next(r for r, row in m.meaning.rows.items() if row["canonical"] == "how do i check my account balance")
    other = next(r for r in m.meaning.rows if r != rid)
    m.meaning_resolve("how much money do i have left", {"verdict": "same", "row": rid})
    q = "how much money do i have left"
    out = m.ask(q)
    assert out["via"] == "meaning" and out["row"] == rid
    assert m.verify_decision(q, rid, key="meaning")["valid"] is not True    # nothing reported yet: no vouching
    m.decision_outcome(out["id"], rid)                                       # confirmed by the one outcome path
    v = m.verify_decision(q, rid, key="meaning")
    assert v["valid"] is True and v["checks"]["seen"] > 0.99
    assert m.verify_decision(q, other, key="meaning")["valid"] is False      # a different row is not vouched for
    assert m.ask(q)["via"] == "meaning"                                      # the meaning door still serves it
    assert rid not in (getattr(m, "_reflex_labels", None) or {})             # a row id never became a trace label

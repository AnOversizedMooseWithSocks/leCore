"""tests/test_negatives.py -- E2.1 ONE NEGATIVE STORE: a negative TRAINS the door it belongs to.

Before (the CLM panel, defect D4): negatives lived in three unconnected forms and none trained anything --
  * the reflex trace's region-wide failure / success fields (lost by to_state, a split, a retile, a reload),
  * the meaning index's exact blocked wordings (_meaning_neg: that one wording, that one row, nothing near it),
  * the ladder's bad payloads (_payload_bad), whose checks in the exact arm and the meaning rung compared an id the
    exact store never wrote (pid was always None) -- dead code.
Now: a wrong meaning serve (decision_outcome or answer_feedback) and a ladder payload vetoed for a meaning row become
a LABELLED NEGATIVE on that row's learned prototype (MeaningIndex.proto_negative) plus the exact veto; the trace's
outcome fields travel through to_state / a split / a partition save; duplicate rows that share an answer key are
merged at creation (the rule is in MeaningIndex.add_row); the dead pid checks are live and cannot leak across a
reused slot. Synthetic, hand-written wording; every "model" is scripted.
"""
import json
import os

os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np

BAL = "how do i check my account balance"
BAL_A = "Open the app and tap Accounts."
PIN = "how do i change my pin"
PIN_A = "Settings > Card > Change PIN."


def _mind(protos=True):
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    if protos:
        m.meaning.enable_protos()
    return m


def _rid(m, text):
    return next(r for r, row in m.meaning.rows.items() if row["canonical"] == text)


def _calibrated(m):
    for k in range(24):
        m.meaning.observe(1.0 + k * 0.05, True)
        m.meaning.observe(0.1 + k * 0.02, False)
    return m


def _proto_cos(mi, text, rid):
    """cos(question, row's learned prototype) as rank() reads it (the centroid when no delta)."""
    q, qn = mi._query_vec(text)
    A, _, _, cen = mi._row_scores(q, qn)
    return float(cen[A["rix"][rid]])


# ---------------------------------------------------------------- the meaning door
def test_a_wrong_meaning_serve_becomes_a_labelled_negative_on_the_row():
    m = _calibrated(_mind())
    m.teach(BAL, BAL_A)
    m.teach(PIN, PIN_A)
    m.meaning_resolve("how can i see my account balance", {"verdict": "same", "row": _rid(m, BAL)})
    q = "how can i see my account balance today"
    out = m.ask(q)
    assert out["via"] == "meaning" and out["row"] == _rid(m, BAL)
    mi = m.meaning
    near = "can i see my account balance today"                # a NEIGHBOUR wording the exact veto cannot reach
    before_q, before_near = _proto_cos(mi, q, _rid(m, BAL)), _proto_cos(mi, near, _rid(m, BAL))
    n0 = mi.protos.n_negatives
    rep = m.decision_outcome(out["id"], "wrong")
    assert rep["forwarded"]["meaning"] == "corrected"
    assert mi.protos.n_negatives == n0 + 1                                     # it TRAINED the row
    assert (q.lower(), _rid(m, BAL)) in {(a, b) for a, b in m._meaning_neg}    # ...and the exact veto stays
    assert _proto_cos(mi, q, _rid(m, BAL)) < before_q
    assert _proto_cos(mi, near, _rid(m, BAL)) < before_near                   # the neighbour moved too
    assert m.ask(q).get("via") != "meaning"


def test_answer_feedback_on_a_meaning_serve_trains_the_row():
    m = _calibrated(_mind())
    m.teach(BAL, BAL_A)
    m.meaning_resolve("how can i see my account balance", {"verdict": "same", "row": _rid(m, BAL)})
    q = "how can i see my account balance now"
    assert m.ask(q)["via"] == "meaning"
    n0 = m.meaning.protos.n_negatives
    assert m.answer_feedback(q, ok=False)["via"] == "meaning"
    assert m.meaning.protos.n_negatives == n0 + 1
    assert m.ask(BAL)["answer"] == BAL_A                                      # the row's own answer is untouched


def test_an_escalated_verdict_does_not_push_twice():
    """A `same` naming another row for an ESCALATED question is not a wrong serve: the InfoNCE update of its link
    already pushed the rival -- an explicit negative would count one label twice. Only the exact veto is kept."""
    m = _mind()
    m.teach(BAL, BAL_A)
    m.teach(PIN, PIN_A)
    q = "how do i change what my balance pin is"
    ranked = m.meaning.answers(q)
    assert len(ranked) == 2
    other = ranked[1][0]
    n0 = m.meaning.protos.n_negatives
    m.meaning_resolve(q, {"verdict": "same", "row": other}, decision={"action": "escalate", "ranked": ranked},
                      _ranked=ranked)
    assert m.meaning.protos.n_negatives == n0 and (" ".join(q.split()), ranked[0][0]) in m._meaning_neg
    assert m.meaning.protos.n_updates >= 1                                    # the link trained the rows


def test_the_ladders_bad_payload_trains_its_meaning_row_and_the_dead_pid_checks_are_live():
    m = _mind()
    m.teach(BAL, BAL_A)
    lad = m.zoo["ladder"]
    ex = lad._exact[BAL]
    assert ex["pid"] is not None and lad._payload_qs[ex["pid"]] == BAL        # the exact entry knows its payload
    # a fuzzy reflex hit on the taught payload, reported bad
    q = "how do i check my account balance ?"
    rep = m.answer_feedback(q, ok=False)
    assert rep["located"] and rep["payload"] == ex["pid"]
    assert m.meaning.protos.n_negatives == 1                                  # the meaning row was trained
    # THE DEAD CHECK, LIVE: the exact arm no longer serves the vetoed payload for its own taught question, and the
    # meaning rung no longer serves it for a rewording (before: pid None never matched -- both kept serving it)
    assert m.ask(BAL).get("answer") != BAL_A
    m.meaning_resolve("show me my account balance", {"verdict": "same", "row": _rid(m, BAL)}) \
        if any(r["canonical"] == BAL for r in m.meaning.rows.values()) else None
    assert m.ask("show me my account balance").get("answer") != BAL_A
    # a deliberate re-teach lifts the veto (the documented recovery path)
    m.teach(BAL, "Open the app, tap Accounts, then Balance.")
    assert m.ask(BAL)["answer"].startswith("Open the app, tap Accounts")


def test_a_reused_slot_never_leaks_another_questions_veto():
    """After a tile split a payload id ('tile:slot') can name a different question's payload. The veto check
    requires the bad id to still belong to THIS question, so an unrelated exact entry is never vetoed by it."""
    m = _mind(protos=False)
    m.teach(BAL, BAL_A)
    lad = m.zoo["ladder"]
    pid = lad._exact[BAL]["pid"]
    lad._payload_bad = {pid}
    lad._payload_qs[pid] = "some other question entirely"                   # the slot now belongs to another q
    assert m.ask(BAL)["answer"] == BAL_A


# ---------------------------------------------------------------- the merge rule
def test_the_merge_rule_is_off_by_default_and_measured_why():
    """MEASURED (tools/bench_meaning.py online, perfect stand-in): the merge rule gains CLINC150 (52.7% vs 48.3%) but
    costs Banking77 a third of its coverage (25.5% vs 37.2%) -- so it ships OFF; the mechanism is a per-mind switch."""
    from holographic.agents_and_reasoning.holographic_meaning import MeaningIndex
    assert MeaningIndex.MERGE_SAME_ANSWER is False
    m = _mind()
    m.teach(BAL, BAL_A)
    m.meaning_resolve("show me what is in my account", {"verdict": "new", "answer": BAL_A})
    assert len(m.meaning.rows) == 2 and m.meaning.merged == 0            # off: a duplicate row, as before


def test_a_new_answer_that_matches_a_row_joins_it_instead_of_splitting_the_intent():
    m = _mind()
    m.meaning.MERGE_SAME_ANSWER = True                                       # the switch (off by default)
    m.teach(BAL, BAL_A)
    out = m.meaning_resolve("show me what is in my account", {"verdict": "new", "answer": BAL_A})
    assert out["learned"] and len(m.meaning.rows) == 1 and m.meaning.merged == 1
    row = m.meaning.rows[_rid(m, BAL)]
    assert "show me what is in my account" in row["phrasings"]
    assert m.ask("show me what is in my account")["answer"] == BAL_A           # the exact store still has it
    # different answer -> its own row; another SESSION -> never merged across the boundary
    m.teach(PIN, PIN_A)
    assert len(m.meaning.rows) == 2
    m.session_open("alice")
    m.teach("what is in my account", BAL_A)
    m.session_close()
    assert len(m.meaning.rows) == 3                                          # never merged across a session


def test_methods_and_loads_are_never_merged():
    from holographic.agents_and_reasoning.holographic_meaning import MeaningIndex
    m = _mind()
    for q, frm in (("price of solana", "solana"), ("price of ether", "ether")):
        m.meaning_resolve(q, {"verdict": "new", "answer": "-", "method": {
            "verb": "price", "args": {"symbol": frm.upper()[:3]}, "from_question": {"symbol": frm}, "live": True}})
    assert sum(1 for r in m.meaning.rows.values() if r["kind"] == "method") == 2   # same verb, still two rows
    mi = MeaningIndex()
    mi.MERGE_SAME_ANSWER = False
    mi.add_row("q one", akey="a:x")
    mi.add_row("q two", akey="a:x")
    back = MeaningIndex.from_state(json.loads(json.dumps(mi.state())))
    assert len(back.rows) == 2 and back.merged == 0                           # a load restores rows as saved


def test_a_merged_row_survives_its_oldest_wordings_veto():
    m = _mind(protos=False)
    m.meaning.MERGE_SAME_ANSWER = True
    m.teach(BAL, BAL_A)
    m.teach("show me what is in my account", BAL_A)                          # merged into BAL's row
    rid = _rid(m, BAL)
    assert len(m.meaning.rows) == 1
    lad = m.zoo["ladder"]
    lad._exact.pop(BAL)                                                       # the canonical's answer is gone
    ex = m._meaning_answer(m.meaning.rows[rid])
    assert ex is not None and ex["answer"] == BAL_A                           # served from the other wording


# ---------------------------------------------------------------- the rows never push their own answer
def test_rows_sharing_an_answer_key_are_never_pushed():
    from holographic.agents_and_reasoning.holographic_meaning import MeaningIndex
    def build(bkey):
        mi = MeaningIndex()
        mi.MERGE_SAME_ANSWER = False
        a = mi.add_row("how do i check my balance", akey="a:bal")
        b = mi.add_row("what is my account balance", akey=bkey)
        mi.add_row("how do i change my pin", akey="a:pin")
        mi.enable_protos()
        return mi, a, b
    q = "balance on my account"                                              # b is the CLOSER row for this wording
    mi, a, b = build("a:bal")                                                # b: a duplicate row, same answer
    r = mi.learn_verdict(q, a)
    rp = mi.protos
    assert r["touched"] >= 1 and a in rp._ix and np.any(rp.D[rp._ix[a]])    # the truth was pulled
    assert b not in rp._ix or not np.any(rp.D[rp._ix[b]])                    # never a rival of its own answer
    mi2, a2, b2 = build("a:other")                                          # the counterfactual: b IS a rival
    mi2.learn_verdict(q, a2)
    assert b2 in mi2.protos._ix and np.any(mi2.protos.D[mi2.protos._ix[b2]])


# ---------------------------------------------------------------- the reflex trace's outcome fields
def test_trace_outcome_fields_survive_state_a_split_and_a_partition_save(tmp_path):
    from holographic.agents_and_reasoning.holographic_lever7 import DisplacementTrace, TiledDisplacementTrace
    rng = np.random.default_rng(0)
    t = DisplacementTrace(256, seed=0)
    k_bad, k_good = rng.standard_normal(256), rng.standard_normal(256)
    t.record_outcome(k_bad, False)
    t.record_outcome(k_good, True)
    back = DisplacementTrace.from_state(json.loads(json.dumps(t.to_state())))
    assert np.array_equal(back._fail_field, t._fail_field) and np.array_equal(back._succ_field, t._succ_field)
    assert "fail_field" not in DisplacementTrace(256).to_state()               # no outcomes: the old state shape
    # a SPLIT keeps them on both halves (it used to replay the audit only -- every failure was forgotten)
    tt = TiledDisplacementTrace(dim=256, seed=0, advisory_load=0.05)
    tt.tiles[0].record_outcome(k_bad, False)
    tt.tiles[0].volatility.mark("price")
    fail0 = tt.tiles[0]._fail_field.copy()
    for i in range(40):
        tt.write(rng.standard_normal(256), rng.standard_normal(256))
    assert len(tt.tiles) >= 2
    assert all(np.array_equal(x._fail_field, fail0) for x in tt.tiles)
    assert all("price" in x.volatility._marks for x in tt.tiles)              # found on the way: marks survived
    s_, f_ = tt.outcome_fields()
    assert np.allclose(f_, len(tt.tiles) * fail0)
    # the mind: a reported failure survives learning_save / learning_load
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    m.teach("what is the release branch", "main")
    key = m.zoo["ladder"]._qkey("what is the release branch")
    m.reflex_outcome(key, False)
    before = [x._fail_field.copy() for x in m.experience.tiles]
    root = str(tmp_path / "p")
    m.learning_save(root)
    m2 = lecore.UnifiedMind(dim=256, seed=0)
    m2.learning_load(root)
    assert any(np.any(x._fail_field) for x in m2.experience.tiles)
    assert all(np.array_equal(a_, b_._fail_field) for a_, b_ in zip(before, m2.experience.tiles))


def test_the_learned_rows_persist_exactly(tmp_path):
    import lecore
    m = _calibrated(_mind())
    m.teach(BAL, BAL_A)
    m.teach(PIN, PIN_A)
    for q in ("how can i see my account balance", "how much money is in my account"):
        m.meaning_resolve(q, {"verdict": "same", "row": _rid(m, BAL)})
    m.meaning_resolve("i want a new pin", {"verdict": "same", "row": _rid(m, PIN)})
    m.meaning.proto_negative("my pin and my balance", _rid(m, BAL))      # a labelled negative: a delta for sure
    assert m.meaning.protos.active()
    probes = ["what is my balance", "reset my pin", "money in my account"]
    before = [m.meaning.answers(p) for p in probes]
    root = str(tmp_path / "p")
    m.learning_save(root)
    m2 = lecore.UnifiedMind(dim=256, seed=0)
    m2.learning_load(root)
    assert m2.meaning.protos is not None and m2.meaning.protos.n_updates == m.meaning.protos.n_updates
    assert np.array_equal(m2.meaning.protos.D, m.meaning.protos.D)
    m.meaning._arrays = None                                                  # both read from a fresh build
    assert [m2.meaning.answers(p) for p in probes] == [m.meaning.answers(p) for p in probes] == before

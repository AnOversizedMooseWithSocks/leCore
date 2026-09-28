"""tests/test_ledger_hooks.py -- Ledger.add COMBINES hooks for one record id (CLM backlog, wave 2).

THE DEFECT (found by wave-1 worker B-encoders, recorded in holographic/plugins/clm.py): a record id is sha256 over
(state, question, options, answer, via), so two doors can file ONE decision under ONE id -- an escalated typed answer
(systemone_decide, via model_end) and the CLM plugin's own record of the same answer. Ledger.add kept one hook per id,
so the second door silently replaced the first door's learning hook (the plugin now files under "clm:<question>" as a
workaround). Now hooks combine: every door's hook runs, each caught, the same code site never runs twice, and a
record may carry several hook SPECS that survive a restart (each re-derived on load, as wave 1's single spec was).
All data is hand-written; nothing here needs the network or the datasets.
"""
import json
import os

os.environ.setdefault("PYTHONHASHSEED", "0")

from holographic.agents_and_reasoning.holographic_decisionrecord import DecisionRecord, Ledger, hook_specs  # noqa: E402

Q = {"answer": {"type": "choice", "options": ["billing", "shipping"],
                "examples": {"billing": ["card charged twice", "refund my invoice fee"],
                             "shipping": ["parcel lost in transit", "courier delivery late"]}}}


def _rec(meta=None):
    return DecisionRecord("the courier lost my parcel", "answer", ["billing", "shipping"], "shipping", "model_end",
                          meta=meta)


def _mind():
    import lecore
    return lecore.UnifiedMind(dim=256, seed=0)


# ------------------------------------------------------------------------------------------ the Ledger itself ------

def test_the_module_hooks_selftest():
    from holographic.agents_and_reasoning.holographic_decisionrecord import _selftest, _selftest_hooks
    assert _selftest_hooks() == {"ok": True, "pinned": 1}
    assert _selftest() == {"ok": True, "pinned": 11}              # the original selftest, unchanged



def test_two_doors_filing_one_id_both_learn_in_the_order_attached():
    L = Ledger()
    ran = []

    def typed_door(outcome):
        ran.append(("typed", outcome))
        return {"updated": True}

    def clm_door(outcome):
        ran.append(("clm", outcome))
        return {"door": "clm:answer"}
    rid = L.add(_rec(), hook=typed_door)
    assert L.add(_rec(), hook=clm_door) == rid                     # the SAME id: the collision
    rep = L.report(rid, "shipping")
    assert ran == [("typed", "shipping"), ("clm", "shipping")]    # before wave 2: only ("clm", ...) ran
    assert rep["forwarded"] == {"updated": True}                   # the first hook's result, as before
    assert rep["forwarded_all"] == [{"updated": True}, {"door": "clm:answer"}]
    assert rep["was_correct"] is True and "hook_error" not in rep


def test_the_same_door_deciding_again_replaces_its_hook_and_never_runs_twice():
    L = Ledger()
    ran = []

    def make(tag):
        def hook(outcome):                                         # one code site, a fresh closure per decision
            ran.append(tag)
        return hook
    rid = L.add(_rec(), hook=make("first"))
    L.add(_rec(), hook=make("second"))                             # the same decision made again
    rep = L.report(rid, "shipping")
    assert ran == ["second"] and "forwarded_all" not in rep        # one hook: exactly the old report shape
    # an explicit hook_key separates two hooks that share a code site
    L2 = Ledger()
    rid2 = L2.add(_rec(), hook=make("a"), hook_key="door:a")
    L2.add(_rec(), hook=make("b"), hook_key="door:b")
    ran.clear()
    L2.report(rid2, "billing")
    assert ran == ["a", "b"]


def test_a_raising_hook_is_reported_and_never_stops_the_other_door():
    L = Ledger()
    got = []

    def broken(outcome):
        raise RuntimeError("the typed door broke")

    def calibrator(outcome):
        got.append(outcome)
        return "ok"
    rid = L.add(_rec(), hook=broken)
    L.add(_rec(), hook=calibrator)
    rep = L.report(rid, "billing")
    assert got == ["billing"] and "RuntimeError: the typed door broke" in rep["hook_error"]
    assert rep["forwarded_all"] == [None, "ok"] and L.get(rid).outcome == "billing"


def test_hook_specs_merge_and_survive_the_text_round_trip():
    """Two doors each NAME their hook in meta (restart-proof specs): re-adding merges them into a list; one spec keeps
    the old dict form byte for byte; after to_text / load_text a resolver re-derives BOTH."""
    L = Ledger()
    a = {"kind": "systemone", "key": "k", "q": "answer"}
    b = {"kind": "calibrate", "door": "clm:answer", "score": 0.4, "answer": "shipping"}
    rid = L.add(_rec({"hook": a}))
    assert L.get(rid).meta["hook"] == a                            # one spec: the dict form, unchanged
    L.add(_rec({"hook": a}))                                       # the same spec again: not duplicated
    assert L.get(rid).meta["hook"] == a
    L.add(_rec({"hook": b}))
    assert L.get(rid).meta["hook"] == [a, b] and hook_specs(L.get(rid).meta) == [a, b]
    rows = json.loads(json.dumps(L.to_text()["records"]))
    seen = []
    L2 = Ledger(resolve_hook=lambda r: (lambda outcome: seen.append((r.meta["hook"]["kind"], outcome)) or r.meta["hook"]["kind"]))
    assert L2.load_text(rows)["loaded"] == 1
    assert L2.hooks_for(rid) == ["spec:" + json.dumps(a, sort_keys=True), "spec:" + json.dumps(b, sort_keys=True)]
    rep = L2.report(rid, "shipping")
    assert seen == [("systemone", "shipping"), ("calibrate", "shipping")]
    assert rep["forwarded_all"] == ["systemone", "calibrate"]
    # a MERGING load keeps the live record but still picks up a spec only the stored copy names
    L3 = Ledger()
    L3.add(_rec({"hook": a}))
    assert L3.load_text(rows, merge=True)["kept_live"] == 1 and L3.get(rid).meta["hook"] == [a, b]


def test_a_closure_added_with_its_own_spec_is_not_run_twice():
    """The standalone rule 'an explicit closure wins' now holds per add(): a closure passed together with a record
    that names a spec IS that spec's implementation while it lives -- the resolver is not asked for it again. A spec
    another door added later is still resolved."""
    calls = []
    L = Ledger(resolve_hook=lambda r: (lambda outcome: calls.append(("resolved", r.meta["hook"]["kind"]))))
    rid = L.add(_rec({"hook": {"kind": "typed"}}), hook=lambda outcome: calls.append(("live", "typed")))
    L.report(rid, "shipping")
    assert calls == [("live", "typed")]
    L.add(_rec({"hook": {"kind": "meaning"}}))
    calls.clear()
    L.report(rid, "shipping")
    assert calls == [("live", "typed"), ("resolved", "meaning")]


# --------------------------------------------------------------------------------------------- on the mind ------

def _escalated(m):
    """An escalated typed answer (the model end picks 'shipping'), filed by systemone_decide via model_end."""
    a = m.typed("the courier lost my parcel", ["billing", "shipping"], examples=Q["answer"]["examples"],
                margin=0.99, escalate=lambda payload: "shipping")
    assert a["via"] == "escalated" and a["value"] == "shipping"
    return a


def _second_door_record(m, rid, **meta):
    """What the CLM plugin filed before its workaround: a record with the SAME five fields -> the same id."""
    r = m.decision_ledger().get(rid)
    rec = DecisionRecord(r.state, r.question, list(r.options), r.answer, r.via, margin=0.25, meta=meta)
    assert rec.id == rid
    return rec


def test_an_escalated_typed_answer_and_the_clm_record_on_one_id_both_learn():
    m = _mind()
    a = _escalated(m)
    so = m._systemone_from_key(m.decision_ledger().get(a["id"]).meta["hook"]["key"])
    before = len(so._outcomes.get("answer", []))
    cal = m.door_calibrator("clm:answer")

    def clm_hook(outcome):                                         # the plugin's live calibration closure
        cal.observe(0.25, outcome == "shipping")
        return {"door": "clm:answer", "n": len(cal.pairs)}
    m.decision_ledger().add(_second_door_record(m, a["id"], tier="clm"), hook=clm_hook)
    rep = m.decision_outcome(a["id"], "shipping")
    assert len(cal.pairs) == 1                                     # the CLM door learned ...
    typed_rep = [f for f in rep["forwarded_all"] if isinstance(f, dict) and "was_correct" in f]
    assert typed_rep and typed_rep[0]["was_correct"] is True       # ... AND the typed SystemOne did (the lost hook):
    assert len(so._outcomes["answer"]) == before + 1               # its outcome stream took the label
    assert "hook_error" not in rep


def test_several_hook_specs_survive_a_restart(tmp_path):
    """The same collision filed with restart-proof specs: the typed door's systemone spec + a 'calibrate' spec (the
    form a closure-free CLM hook takes). learning_save -> a FRESH mind -> learning_load -> the OLD id trains both."""
    m = _mind()
    a = _escalated(m)
    spec = {"kind": "calibrate", "door": "clm:answer", "score": 0.25, "answer": "shipping"}
    m.decision_ledger().add(_second_door_record(m, a["id"], hook=spec))
    specs = hook_specs(m.decision_ledger().get(a["id"]).meta)
    assert [s["kind"] for s in specs] == ["systemone", "calibrate"]
    root = str(tmp_path / "partition")
    m.learning_save(root)
    m2 = _mind()
    m2.learning_load(root)
    rep = m2.decision_outcome(a["id"], "shipping")
    assert m2.door_calibration_report()["clm:answer"]["labels"] == 1
    assert rep["forwarded"]["was_correct"] is True                 # the typed door's re-derived hook ran first
    assert rep["forwarded_all"][1]["door"] == "clm:answer" and "hook_error" not in rep

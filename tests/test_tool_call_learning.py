"""tests/test_tool_call_learning.py -- sweep 178: tool calls feed the decision learners.

Before this sweep the decision trees / typed decisions / route_tiered all minted DecisionRecords and said
"report the outcome with decision_outcome(id, ...)", but nothing that actually CALLS a tool ever did:
  * AgentLoop had the route id, the tool and whether it raised -- and dropped all three;
  * serve()'s tool reflex noted successes in a side trace, nothing on failure, no record at all;
  * that side trace (tool_usage) was never saved, so a restart forgot it.
The rule carried over from sweep 177: ONLY A JUDGED CALL TEACHES. Unjudged calls are recorded, not learned.

Tools are stubbed on the mind instance (invoke / api_use): these tests pin the LEARNING LOOP, not the tools.
"""
import os

import pytest

os.environ.setdefault("PYTHONHASHSEED", "0")

TASK = "smooth a bumpy mesh"


def _mind():
    import lecore
    return lecore.UnifiedMind(dim=256, seed=0)


def _stub_invoke(m, raise_on=None):
    """Replace invoke with a recorder: tools 'run' without touching the engine. raise_on names a tool that raises."""
    calls = []

    def invoke(name, args=None):
        calls.append(name)
        if name == raise_on:
            raise RuntimeError("stub failure")
        return {"ran": name}
    m.invoke = invoke
    return calls


def _scripted_llm(order):
    """A 'model' that calls the manifest's tools in the given order (by index), then says DONE."""
    state = {"i": 0}

    def llm(prompt):
        tools = [ln.strip().split(" -- ")[0] for ln in prompt.split("TOOLS:")[1].split("\n\n")[0].splitlines() if ln.strip()]
        if state["i"] >= len(order):
            return "DONE: ok"
        pick = tools[min(order[state["i"]], len(tools) - 1)]
        state["i"] += 1
        return '{"tool": "%s", "args": {}}' % pick
    return llm


def _loop(m, llm, **kw):
    from holographic.agents_and_reasoning.holographic_agentloop import AgentLoop
    return AgentLoop(m, llm, max_steps=4, **kw)


# ------------------------------------------------------------------------ 1. AgentLoop closes the loop ----

def test_verified_tool_call_teaches_the_route_and_the_repeat_is_answered_from_experience():
    m = _mind()
    _stub_invoke(m)
    out = _loop(m, _scripted_llm([0]), verify=lambda task, tool, args, result: True).run(TASK)
    assert out["gate"]["tier"] == "answer"
    step = out["steps"][0]
    assert step["verified"] is True and step["taught"] is True and step["taught_on"] == out["gate"]["id"]
    again = m.route_tiered(TASK, reflex=True)
    assert again["via"] == "reflex" and again["answer"] == step["card"]


def test_outcomes_are_reported_by_card_name_not_method_name():
    """One label space: the router speaks card names, so the loop must report the card behind the method."""
    m = _mind()
    _stub_invoke(m)
    out = _loop(m, _scripted_llm([0]), verify=lambda *a: True).run(TASK)
    step = out["steps"][0]
    assert step["card"] != step["tool"]                       # a method name and a card name differ
    assert m.decision_ledger().get(step["taught_on"]).outcome == step["card"]


def test_unjudged_call_is_recorded_but_teaches_nothing():
    m = _mind()
    _stub_invoke(m)
    out = _loop(m, _scripted_llm([0])).run(TASK)               # no verify
    step = out["steps"][0]
    rec = m.decision_ledger().get(step["id"])
    assert rec is not None and rec.meta["tool"] == step["tool"] and rec.meta["verified"] is None
    assert step["taught"] is False and step["taught_on"] is None
    assert m.reflex_decide(TASK)["value"] is None


def test_failed_verdict_and_raised_call_do_not_teach_and_do_not_poison_the_region():
    """A rejected tool and a crashing tool are recorded; neither marks the failure field, so the RIGHT tool,
    judged afterwards on the same task, still teaches and fires (the sweep-177 trap, pinned for tools)."""
    m = _mind()
    calls = _stub_invoke(m, raise_on=None)
    from holographic.agents_and_reasoning.holographic_agentloop import AgentLoop
    first = [r["tool"] for r in AgentLoop(m, lambda _: "x", max_steps=1).manifest(TASK)]
    # the manifest can list one METHOD under two cards (mesh_smooth appears twice) -- pick by distinct tool
    good = first[0]
    bad = next(t for t in first if t != good)
    rejected = next(t for t in first if t not in (good, bad))
    _stub_invoke(m, raise_on=bad)
    judge = lambda task, tool, args, result: tool == good
    # step 0 calls a tool the judge rejects, step 1 the crashing one -> nothing taught
    out = _loop(m, _scripted_llm([first.index(rejected), first.index(bad)]), verify=judge).run(TASK)
    whys = [s["why"] for s in out["steps"]]
    assert any("raised" in w for w in whys) and not any(s["taught"] for s in out["steps"])
    assert m.reflex_decide(TASK)["value"] is None
    # now the judged-good tool on the same task: it must still teach, and the repeat must fire
    out2 = _loop(m, _scripted_llm([0]), verify=judge).run(TASK)
    assert out2["steps"][0]["taught"] is True
    assert m.route_tiered(TASK, reflex=True)["via"] == "reflex"


def test_later_steps_learn_under_their_context():
    """Step 2's decision is 'which tool after step 1' -- keyed on task + what just happened."""
    m = _mind()
    _stub_invoke(m)
    out = _loop(m, _scripted_llm([0, 1]), verify=lambda *a: True).run(TASK)
    s0, s1 = out["steps"][0], out["steps"][1]
    assert s1["taught"] is True and s1["taught_on"] == s1["id"]
    ctx = m.decision_ledger().get(s1["id"]).state
    assert ctx == "%s | after %s" % (TASK, s0["card"])
    assert m.reflex_decide(ctx)["value"] == s1["card"]


def test_tool_loop_passes_verify_and_reflex_through():
    m = _mind()
    _stub_invoke(m)
    out = m.tool_loop(TASK, llm=_scripted_llm([0]), verify=lambda *a: True, reflex=True)
    assert out["steps"][0]["taught"] is True
    out2 = m.tool_loop(TASK, llm=_scripted_llm([0]), reflex=True)
    assert out2["gate"].get("id") and m.route_tiered(TASK, reflex=True)["via"] == "reflex"


# ------------------------------------------------------------------------ 2. serve()'s tool reflex ----

def _tool_mind(results=None):
    """Two tool reflexes whose patterns tie on word overlap -- so word overlap alone picks the FIRST taught."""
    m = _mind()
    results = results or {}
    seen = []

    def api_use(service, endpoint, params=None, headers=None):
        seen.append(endpoint)
        return results.get(endpoint, {"ok": True, "data": {"endpoint": endpoint}})
    m.api_use = api_use
    m.tool_reflex_teach("convert 77 fahrenheit into celsius units", "convertd", "f_to_c")
    m.tool_reflex_teach("convert 25 celsius into fahrenheit units", "convertd", "c_to_f")
    return m, seen


Q = "convert celsius fahrenheit units please"


def test_every_tool_reflex_call_is_a_decision_record():
    m, _ = _tool_mind()
    r = m.serve(Q)
    rec = m.decision_ledger().get(r["id"])
    assert r["via"] == "tool-reflex" and rec.via == "tool" and rec.answer == r["tool"]
    assert set(rec.options) == {"convertd.c_to_f", "convertd.f_to_c"}


def test_a_failed_tool_call_is_recorded_and_counted():
    m, _ = _tool_mind(results={"f_to_c": {"ok": False, "error": "500"}})
    r = m.serve(Q)
    assert r["via"] == "escalate" and r["id"]
    assert m.decision_ledger().get(r["id"]).meta["ok"] is False
    # counts = successful uses (the teach itself noted one); failures are their OWN tally
    assert m.tool_usage.counts["convertd.f_to_c"] == 1 and m.tool_usage.failures["convertd.f_to_c"] == 1


def test_a_corrected_call_changes_the_next_pick():
    """Overlap picks f_to_c (first taught). The judge says c_to_f was right: report it. The next identical
    query is routed by EXPERIENCE to c_to_f, not by overlap."""
    m, seen = _tool_mind()
    r = m.serve(Q)
    assert r["tool"] == "convertd.f_to_c" and r["picked_by"] == "overlap"
    m.decision_outcome(r["id"], "convertd.c_to_f")
    r2 = m.serve(Q)
    assert r2["tool"] == "convertd.c_to_f" and r2["picked_by"] == "experience" and seen[-1] == "c_to_f"


def test_verify_teaches_and_unverified_does_not():
    m, _ = _tool_mind()
    r = m.serve(Q)                                              # no verify: nothing taught
    assert r["verified"] is None and m.reflex_decide(Q, key="ngram")["value"] is None
    r2 = m.serve(Q, verify=lambda q, tool, data: True)
    assert r2["verified"] is True and m.reflex_decide(Q, key="ngram")["value"] == r2["tool"]


# ------------------------------------------------------------------------ 3. the usage trace persists ----

def test_tool_usage_trace_survives_learning_save_and_load(tmp_path):
    import numpy as np
    m = _mind()
    rng = np.random.default_rng(0)
    tasks = [rng.standard_normal(m.tool_usage.dim) for _ in range(3)]
    for i, t in enumerate(tasks):
        m.tool_note(t, "tool_%d" % i, success=True)
    m.tool_note(tasks[0], "tool_bad", success=False)
    before = [m.tool_predict(t, k=2) for t in tasks]
    root = str(tmp_path / "p")
    os.makedirs(os.path.join(root, "learning"), exist_ok=True)
    m.learning_save(root)
    m2 = _mind()
    m2.learning_load(root)
    assert [m2.tool_predict(t, k=2) for t in tasks] == before
    assert m2.tool_usage.counts == m.tool_usage.counts
    assert m2.tool_usage.failures == {"tool_bad": 1}


# ------------------------------------------------------------------------ 4. ONE tool learner (CLM backlog E4.3 / E4.4) ----
# The tool door's ProtoStore (holographic_unified_p33_router) is the one learner of tool choice: taught patterns seed
# it, a verified call pulls, a reported correction pulls the right tool and pushes the wrong one, a failed verify is a
# labelled negative. Cold start (no judged verdict yet) keeps the word-overlap pick exactly as before.

Q2 = "please convert these celsius temperatures to fahrenheit units"      # a PARAPHRASE of Q (not a repeat)


def test_a_corrected_call_changes_the_next_pick_for_a_paraphrase_through_the_tool_door():
    """E4.3 acceptance: overlap picks f_to_c at cold start; one reported correction (c_to_f was right) and a
    PARAPHRASE -- below the reflex bridge's seen gate, so experience cannot answer it -- is picked by the door."""
    m, seen = _tool_mind()
    r = m.serve(Q)
    assert r["tool"] == "convertd.f_to_c" and r["picked_by"] == "overlap"            # cold start: unchanged
    rep = m.decision_outcome(r["id"], "convertd.c_to_f")
    assert m._tool_audit()["verdicts"] == 1
    fw = rep.get("forwarded")
    assert any(isinstance(f, dict) and f.get("tool_door") for f in (fw if isinstance(fw, list) else [fw]))
    r2 = m.serve(Q2)
    assert r2["tool"] == "convertd.c_to_f" and r2["picked_by"] == "proto" and seen[-1] == "c_to_f"
    assert m.decision_ledger().get(r2["id"]).meta["picked_by"] == "proto"


def test_a_failed_verify_is_a_labelled_negative_that_moves_the_next_pick():
    m, _ = _tool_mind()
    judge = lambda q, tool, data: tool == "convertd.c_to_f"
    r = m.serve(Q, verify=judge)
    assert r["tool"] == "convertd.f_to_c" and r["verified"] is False
    st = m._tool_store(create=False)
    assert st.n_negatives == 1                                   # the rejected pick was pushed away
    r2 = m.serve(Q, verify=judge)                                # no outcome was reported: the bridge knows nothing
    assert r2["picked_by"] == "proto" and r2["tool"] == "convertd.c_to_f" and r2["verified"] is True


def test_unjudged_calls_teach_the_tool_door_nothing():
    """Only a judged call teaches (sweep 177's rule). The old UsageTrace strengthened on EVERY ok call."""
    m, _ = _tool_mind()
    st = m._tool_store(create=False)
    before = st.digest()
    for _ in range(3):
        assert m.serve(Q)["picked_by"] == "overlap"
    assert st.digest() == before and m._tool_audit()["verdicts"] == 0
    assert m.tool_usage.counts["convertd.f_to_c"] == 1 + 3      # the audit still counts uses


def test_tool_predict_reads_the_tool_door_and_refuses_a_foreign_key_space():
    m, _ = _tool_mind()
    top = m.tool_predict(m.tool_key("convert 30 celsius into fahrenheit units"), k=2)
    assert [t for t, _ in top] == ["convertd.c_to_f", "convertd.f_to_c"]
    with pytest.raises(ValueError):
        m.tool_predict([0.0] * 17)                                # wrong dimension: say so, never guess
    assert m.present_tools(m.tool_key("convert 30 celsius into fahrenheit units"), k=1)[0][0] == "convertd.c_to_f"


def test_the_tool_door_survives_a_restart_and_a_legacy_usage_trace_still_loads(tmp_path):
    """E4.4: the door's section round-trips (predictions bit-identical, audit equal), and a partition written by the
    sweep-178 code -- a lecore.learning.toolusage section, no tool door -- still loads: its counts migrate into the
    audit, and the prototypes are re-seeded from the taught patterns on the first serve."""
    import glob
    import numpy as np
    from holographic.agents_and_reasoning.holographic_lever7 import UsageTrace
    from holographic.io_and_interop.holographic_container import load_container, save_container
    m, _ = _tool_mind()
    r = m.serve(Q)
    m.decision_outcome(r["id"], "convertd.c_to_f")
    root = str(tmp_path / "p")
    os.makedirs(os.path.join(root, "learning"), exist_ok=True)
    m.learning_save(root)
    m2 = _mind()
    m2.learning_load(root)
    k = m.tool_key(Q2)
    assert m2.tool_predict(k, k=2) == m.tool_predict(k, k=2)
    assert m2.tool_usage.counts == m.tool_usage.counts and m2._tool_audit()["verdicts"] == 1
    # rewrite the partition the way the OLD code wrote it
    path = sorted(glob.glob(os.path.join(root, "learning", "*.lecore")))[-1]
    got = load_container(open(path, "rb").read())
    old = UsageTrace(dim=2048, seed=0)
    old.note(np.ones(2048), "convertd.f_to_c", success=True)
    old.note(np.ones(2048), "convertd.f_to_c", success=False)
    st = old.to_state()
    tr = np.asarray(st.pop("trace"), np.float64)
    secs = [s for s in got["sections"] if s["kind"] != "lecore.learning.tooldoor"]
    secs.append({"kind": "lecore.learning.toolusage", "id": "v1", "meta": st, "arrays": {"trace": tr}})
    open(path, "wb").write(save_container(secs, meta=got["meta"]))
    m3 = _mind()
    m3.learning_load(root, force=True)
    assert m3.tool_usage.counts == {"convertd.f_to_c": 1} and m3.tool_usage.failures == {"convertd.f_to_c": 1}
    m3.api_use = lambda service, endpoint, params=None, headers=None: {"ok": True, "data": {}}
    assert m3.serve(Q)["via"] == "tool-reflex"                   # the reflexes rebuild from the taught rows ...
    assert set(m3._tool_store(create=False).labels) == {"convertd.f_to_c", "convertd.c_to_f"}   # ... and re-seed

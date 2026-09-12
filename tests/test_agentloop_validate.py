"""THE EXIT GATE: typed return validation with bounded retry in `mind.agent_loop` (sweep 122).

WHAT IS UNDER TEST. `goal_work` scores a step with `out is not None`, so a tool that returns
`{"error": "backend down"}` or a half-built dict is marked DONE -- the quiet failure BENCH-3
measures at 0.50. `agent_loop(contracts=...)` holds each step to a contract the CALLER declared,
retries a bounded number of times with the typed violation handed back, and fails the step loudly
when the contract is never met.

THE NUMBER THAT MATTERS IS `false_retries`, AND IT MUST BE 0. leCore's headline property is
null-referenced abstention: `route_or_abstain` returns `hits: []` and that empty result is the
CORRECT answer. An exit gate that reads empty-as-failure would spend the entry gate's whole
result on retry storms -- it would make the system worse while looking like a feature. Every
test here that touches an abstaining executor asserts the retry count, not merely the verdict.

WHY THE NUMBERS ARE EXACT. "The loop marked it done" is the thing under test, so it cannot be the
measure. The harness below records what each executor last returned and scores a step as a TRUE
pass only if that value satisfies the declared contract; `false_pass` (marked done, contract
violated) is the quiet failure counted directly.

Complements tests/test_agent_loop.py, which pins the ENTRY gate (the model is never consulted on
a no-tool task). Entry: should you have started. Exit: did you prove you finished.
"""
import json
import subprocess
import sys

import pytest

import lecore
from holographic.agents_and_reasoning.holographic_postcheck import (
    MAX_RETRIES, check_contract, contract_digest, normalize_contract, validated_call,
    validation_report, wrap_executors)

# The pre-change return contract of agent_loop, hard-coded. If a future sweep adds a key it must
# come here deliberately -- that is the point of writing the set out rather than computing it.
LEGACY_KEYS = {"goal", "gather", "rounds", "status"}

TYPED = {"expect": "nonempty", "require": ["evidence", "verify"], "retries": 1}
STEPS = ["good", "nullish", "errshape", "abstain", "flaky"]
CONTRACTS = {"good": TYPED, "nullish": TYPED, "errshape": TYPED,
             "abstain": {"expect": "any"},      # the caller KNOWS empty is legitimate here
             "flaky": TYPED}


def _mind():
    m = lecore.UnifiedMind(dim=256, seed=0)
    m.zoo_attach(lambda p: "rung")
    return m


def _executors(rec):
    """Five fake executors covering every verdict the gate must tell apart."""
    st = {"flaky": 0}

    def _mk(name, fn):
        def _run(feedback=None):
            rec["calls"][name] = rec["calls"].get(name, 0) + 1
            rec["fb"][name] = feedback
            v = fn(feedback)
            rec["last"][name] = v
            return v
        return _run

    def _flaky(fb):
        st["flaky"] += 1
        if st["flaky"] == 1:
            return {"result": None}            # malformed: no evidence, no verify
        return {"result": 9, "evidence": ["reran after %s" % (fb or {}).get("error")],
                "verify": "pytest -q"}

    return {"good": _mk("good", lambda fb: {"result": 7, "evidence": ["read the file"],
                                            "verify": "pytest -q"}),
            "nullish": _mk("nullish", lambda fb: None),
            "errshape": _mk("errshape", lambda fb: {"error": "backend down", "result": None}),
            "abstain": _mk("abstain", lambda fb: []),
            "flaky": _mk("flaky", _flaky)}


def _run(with_contracts, objective):
    """One loop over the five fakes, scored against what the executors actually returned."""
    m = _mind()
    rec = {"calls": {}, "last": {}, "fb": {}}
    kw = {"contracts": CONTRACTS} if with_contracts else {}
    r = m.agent_loop(objective, executors=_executors(rec), rounds=2, budget_steps=5,
                     plan=list(STEPS), **kw)
    status = {s["name"]: s["status"] for s in m.goal_book.goals[r["goal"]]["steps"]}
    true_pass = false_pass = false_fail = 0
    for n in STEPS:
        sat = check_contract(rec["last"].get(n), CONTRACTS[n])["ok"]
        if status[n] == "done":
            true_pass, false_pass = true_pass + int(sat), false_pass + int(not sat)
        else:
            false_fail += int(sat)
    return {"loop": r, "status": status, "rec": rec,
            "invocations": sum(rec["calls"].values()),
            "marked_done": sum(1 for n in STEPS if status[n] == "done"),
            "true_pass": true_pass, "false_pass": false_pass, "false_fail": false_fail}


# --------------------------------------------------------------------------------------
# 1. THE SAFETY PROPERTY. A correct abstention must never cost a retry.
# --------------------------------------------------------------------------------------

def test_abstention_costs_exactly_zero_retries():
    # Eight abstaining steps, each with a THREE-retry budget declared. If the gate read empty as
    # failure this would issue 24 retries; leCore's abstention result is worth more than this
    # whole feature, so the assertion is on the count, not on the verdict.
    names = ["ab%d" % i for i in range(8)]
    r = _mind().agent_loop("abstention probe objective for the exit gate",
                           executors={n: (lambda: []) for n in names},
                           rounds=2, budget_steps=8, plan=names,
                           contracts={"*": {"expect": "any", "retries": 3}})
    v = r["validation"]
    assert v["calls"] == 8 and v["abstentions"] == 8
    assert v["retries_issued"] == 0, "the exit gate retried a correct abstention"
    assert v["false_retries"] == 0, "false-retry rate must be 0.000, got %d/8" % v["false_retries"]
    assert v["attempts"] == 8, "8 steps must cost exactly 8 executor calls"


def test_an_empty_result_is_accepted_and_reported_as_empty_not_as_a_failure():
    # ok and empty are DIFFERENT FACTS and collapsing them is the whole hazard: "the tool failed"
    # and "the tool correctly returned nothing" must stay tellable apart in the record.
    v = check_contract([], None)
    assert v["ok"] is True and v["empty"] is True
    assert "abstention" in v["reason"]
    # ...and it fails only where the caller SAID empty means failure.
    assert check_contract([], "nonempty")["ok"] is False


# --------------------------------------------------------------------------------------
# 2. BACKWARD COMPATIBILITY. contracts=None must be the old loop, not a fast path through
#    the new one.
# --------------------------------------------------------------------------------------

def test_default_run_is_the_old_loop_exactly():
    e = {"s1": lambda: {"result": 1, "evidence": ["a"], "verify": "v"},
         "s2": lambda: None, "s3": lambda: []}
    seen = {}
    m = _mind()
    real = m.goal_work

    def spy(gid, executors=None, **kw):
        seen["ex"] = executors
        return real(gid, executors=executors, **kw)

    m.goal_work = spy
    out = m.agent_loop("backward compatibility objective for the exit gate", executors=e,
                       rounds=2, budget_steps=3, plan=["s1", "s2", "s3"])
    assert set(out) == LEGACY_KEYS, "the default return shape changed: %s" % sorted(out)
    assert "validation" not in out, "the audit record leaked into an undeclared run"
    # not merely equal -- the SAME callables. A wrapper that behaved identically would still be a
    # behaviour change (identity is what `stateless` caching and repr-based debugging see).
    assert all(seen["ex"][k] is e[k] for k in e), "an undeclared executor was wrapped"
    assert out["status"] == m.goal_book.goals[out["goal"]]["status"]


def test_two_default_runs_are_identical():
    e = {"s1": lambda: "a", "s2": lambda: "b"}
    kw = dict(executors=e, rounds=1, budget_steps=2, plan=["s1", "s2"])
    a = _mind().agent_loop("determinism objective for the exit gate", **kw)
    b = _mind().agent_loop("determinism objective for the exit gate", **kw)
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


# --------------------------------------------------------------------------------------
# 3. THE MEASUREMENT. Exact counts, with the quiet failure counted directly.
# --------------------------------------------------------------------------------------

def test_without_validation_two_of_five_steps_are_false_passes():
    a = _run(False, "harness objective alpha for the exit gate measurement")
    assert a["invocations"] == 5, "the old loop calls each executor exactly once"
    assert a["marked_done"] == 4
    # errshape returned {"error": ...} and flaky returned a half-built dict; both were scored DONE.
    assert a["false_pass"] == 2, "the quiet-failure baseline moved: %r" % a["status"]
    assert a["true_pass"] == 2
    assert a["status"]["errshape"] == "done" and a["status"]["flaky"] == "done"


def test_with_validation_the_false_passes_go_to_zero_and_the_flaky_step_recovers():
    b = _run(True, "harness objective alpha for the exit gate measurement")
    assert b["false_pass"] == 0, "a contract violation was still scored done: %r" % b["status"]
    assert b["true_pass"] == 3 and b["false_fail"] == 0
    assert b["status"]["errshape"] == "failed" and b["status"]["nullish"] == "failed"
    assert b["status"]["flaky"] == "done", "the flaky step must recover on attempt 2"
    # THE PRICE, stated: 5 executor calls become 8. 1 each for good/abstain, 2 each for the three
    # that failed their contract once. A gate whose cost is not written down is not measured.
    assert b["invocations"] == 8
    assert b["rec"]["calls"] == {"good": 1, "nullish": 2, "errshape": 2, "abstain": 1, "flaky": 2}
    v = b["loop"]["validation"]
    assert (v["calls"], v["passed"], v["failed"]) == (5, 3, 2)
    assert v["retries_issued"] == 3 and v["informed_retries"] == 3
    assert v["abstentions"] == 1 and v["false_retries"] == 0


def test_the_same_loop_twice_gives_the_same_attempt_sequence():
    o = "harness objective alpha for the exit gate measurement"
    x = _run(True, o)["loop"]["validation"]
    y = _run(True, o)["loop"]["validation"]
    assert json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True)


# --------------------------------------------------------------------------------------
# 4. THE RETRY IS BOUNDED, TYPED, AND AUDITABLE.
# --------------------------------------------------------------------------------------

def test_a_broken_executor_is_called_exactly_retries_plus_one_times():
    n = {"n": 0}

    def broken():
        n["n"] += 1
        return None

    r = validated_call(broken, contract={"retries": 2})
    assert n["n"] == 3 and r["attempts"] == 3 and r["ok"] is False
    assert len(r["trace"]) == 3, "an attempt with no trace row is a silent retry"
    assert r["value"] is None, "a rejected value must not leak out as a result"


def test_the_retry_budget_is_refused_above_the_cap_not_clamped():
    # A clamped bound is a bound the caller does not know about.
    with pytest.raises(ValueError):
        validated_call(lambda: None, retries=MAX_RETRIES + 1)
    with pytest.raises(ValueError):
        validated_call(lambda: None, retries=-1)


def test_nothing_declared_means_nothing_retried():
    n = {"n": 0}

    def broken():
        n["n"] += 1
        return None

    r = validated_call(broken)
    assert n["n"] == 1 and r["attempts"] == 1 and r["ok"] is False


def test_the_executor_is_told_which_field_it_missed():
    seen = {}

    def needs_feedback(feedback=None):
        seen["fb"] = feedback
        if feedback is None:
            return {"answer": 42}
        return {"answer": 42, "evidence": ["ran the test"], "verify": "pytest -q"}

    r = validated_call(needs_feedback, contract=TYPED)
    assert r["attempts"] == 2 and r["ok"] is True and r["informed_retries"] == 1
    fb = seen["fb"]
    assert fb["error"] == "contract_violation" and fb["attempt"] == 1
    assert fb["violations"] == ["required field 'evidence' is missing",
                                "required field 'verify' is missing"]
    assert fb["returned_type"] == "dict" and "answer" in fb["returned"]
    # the feedback is ERROR-SHAPED on purpose: an executor that echoes it back is caught next pass.
    assert check_contract(fb, None)["ok"] is False


def test_every_violation_is_reported_not_just_the_first():
    # One round-trip should buy the whole fix, not one third of it.
    v = check_contract({"error": "x"}, TYPED)
    assert len(v["violations"]) == 3 and v["ok"] is False


def test_the_failure_reason_reaches_the_caller():
    b = _run(True, "harness objective alpha for the exit gate measurement")
    fails = {f["name"]: f for f in b["loop"]["validation"]["failures"]}
    assert set(fails) == {"nullish", "errshape"}
    assert "returned None" in fails["nullish"]["reason"]
    assert "error-shaped return" in fails["errshape"]["reason"]
    assert fails["errshape"]["attempts"] == 2 and fails["errshape"]["raised"] is None


def test_a_raising_executor_is_distinguished_from_a_quiet_one():
    def boom():
        raise RuntimeError("backend unavailable")

    r = validated_call(boom, contract={"retries": 1})
    assert r["ok"] is False and r["attempts"] == 2
    assert "RuntimeError" in r["raised"] and r["reason"] == "call raised"
    q = validated_call(lambda: None, contract={"retries": 1})
    assert q["ok"] is False and q["raised"] is None, "a quiet failure is not a raise"


# --------------------------------------------------------------------------------------
# 5. THE CONTRACT ITSELF.
# --------------------------------------------------------------------------------------

def test_true_is_not_an_int_and_zero_is_not_empty():
    # bool is an int subclass: a step declared to return a COUNT must not pass with True.
    assert check_contract({"n": True}, {"types": {"n": "int"}})["ok"] is False
    assert check_contract({"n": 0}, {"types": {"n": "int"}})["ok"] is True
    assert check_contract({"evidence": []}, {"require": ["evidence"]})["ok"] is False
    assert check_contract({"evidence": 0}, {"require": ["evidence"]})["ok"] is True


@pytest.mark.parametrize("bad", [{"expct": "nonempty"}, {"expect": "nonemty"},
                                 {"types": {"k": "integer"}}, 7])
def test_a_malformed_contract_is_refused(bad):
    # A contract that silently drops the half you cared about is a gate that stopped gating.
    with pytest.raises(ValueError):
        normalize_contract(bad)


def test_the_contract_digest_is_hashlib_stable_across_processes():
    # hashlib, never hash(): the id must survive a different PYTHONHASHSEED, or the audit record
    # cannot be compared between two runs of the same loop.
    here = contract_digest(TYPED)
    code = ("import json,sys;"
            "sys.path.insert(0, %r);"
            "from holographic.agents_and_reasoning.holographic_postcheck import contract_digest;"
            "print(contract_digest(json.loads(%r)))" % (_repo_root(), json.dumps(TYPED)))
    for seed in ("0", "1", "12345"):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env=_env(seed), check=True).stdout.strip()
        assert out == here, "digest moved with PYTHONHASHSEED=%s" % seed
    assert contract_digest("nonempty") == contract_digest({"expect": "nonempty"})
    assert contract_digest("nonempty") != contract_digest({"expect": "truthy"})


def _repo_root():
    import os
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(seed):
    import os
    e = dict(os.environ)
    e["PYTHONHASHSEED"] = seed
    e["PYTHONPATH"] = _repo_root() + os.pathsep + e.get("PYTHONPATH", "")
    return e


def test_an_undeclared_executor_is_the_same_object_and_a_ghost_contract_is_reported():
    plain = lambda: "x"
    w, j = wrap_executors({"a": plain}, {"ghost": "nonempty"})
    assert w["a"] is plain and j["wrapped"] == []
    # A contract naming a step with no executor is a typo the caller wants to hear about.
    assert validation_report(j)["declared_without_executor"] == ["ghost"]


# --------------------------------------------------------------------------------------
# 6. WIRING. A module reachable only by import does not exist by this repo's rule.
# --------------------------------------------------------------------------------------

def test_the_faculty_is_wired_to_the_mind_and_documented():
    m = lecore.UnifiedMind(dim=256, seed=0)
    for name in ("result_contract", "validated_call"):
        fn = getattr(m, name, None)
        assert callable(fn), "%s is not wired to UnifiedMind" % name
        assert (fn.__doc__ or "").strip(), "%s has no docstring (undiscoverable)" % name
    assert m.result_contract({"evidence": ["e"], "verify": "v"}, TYPED)["ok"] is True
    assert m.result_contract(None, TYPED)["ok"] is False
    r = m.validated_call(lambda: None, contract={"retries": 2})
    assert r["attempts"] == 3 and r["ok"] is False
    # the JSON forms cross /invoke intact -- a contract a client cannot send is not agent-facing.
    assert json.loads(json.dumps(m.result_contract([], "any")))["empty"] is True

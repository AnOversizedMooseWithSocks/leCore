"""POSTCHECK -- did that call actually produce anything USABLE? The silent half of runtime discovery.

WHY, and it is measured rather than assumed. BENCH-3 scores abandon 0.50 / SILENT-FAIL 0.50: a tool that
RAISES is caught by ordinary exception flow, while a tool that returns nothing usable is treated as
success, because nothing in the loop asks the question. AgentAbstain names that the most dangerous of its
three failure modes -- acting, then claiming restraint.

THE TENSION THAT SHAPES THE WHOLE DESIGN, and it is leCore's own case. **An empty result is often the
CORRECT answer.** `route_or_abstain` returns `hits: []` on a deliberate abstention; `alias_gaps` returns
an empty list when there are no gaps; a search that finds nothing has succeeded at searching. A checker
that flags empty-as-failure would flag every correct abstention as a failure -- turning a safety property
into a bug report.

You cannot tell those apart FROM THE VALUE. `[]` is identical whether it means "nothing matched, and that
is the answer" or "the backend died and returned a default". So the CALLER declares the postcondition,
exactly as the caller owns verdict labels in the outcome-feedback seam: leCore has many loops with
different notions of a good result, and one built-in rule would be wrong for most of them.

Default `expect="any"` therefore catches only the unambiguous cases -- None, and error-shaped returns --
and says so. Anything stronger is opt-in.

THE EXIT GATE (sweep 122). `result_usable` answers the question once; `check_contract` /
`validated_call` / `wrap_executors` hold a step to a CALLER-DECLARED contract and retry a bounded
number of times, handing the executor a typed violation instead of accepting the bad return. That is
the half NOOA has and leCore did not: leCore gates the ENTRY (null-referenced abstention -- should
you have started?), this gates the EXIT (did you prove you finished?). The tension above is what
makes the pairing safe rather than self-defeating: the exit gate must never fire on a correct
abstention, so `validation_report` reports `false_retries` as a first-class number and the selftest
pins it at 0."""
import math


def _is_error_shaped(v):
    """A dict carrying an error field, or an Exception instance. Deliberately narrow: guessing that a
    dict with a False in it means failure would misread half the reports in this codebase."""
    if isinstance(v, BaseException):
        return True
    if isinstance(v, dict):
        for k in ("error", "err", "exception", "traceback"):
            if v.get(k):
                return True
    return False


def _empty(v):
    """Empty CONTAINER, not empty meaning. An array of all zeros is not empty -- it is a result."""
    if v is None:
        return True
    if isinstance(v, (str, bytes, list, tuple, dict, set, frozenset)):
        return len(v) == 0
    n = getattr(v, "size", None)
    return n == 0 if isinstance(n, int) else False


def _all_nan(v):
    try:
        import numpy as np
        a = np.asarray(v, dtype=float)
        return a.size > 0 and bool(np.all(~np.isfinite(a)))
    except Exception:
        return isinstance(v, float) and (math.isnan(v) or math.isinf(v))


def result_usable(value, expect="any"):
    """Is `value` a usable result? Returns {usable, reason, expect}. The CALLER names the postcondition.

    `expect`:
      * "any"      (default) -- only None and error-shaped returns fail. An empty list PASSES, because
                    it is a legitimate answer and this checker must not turn abstention into a bug.
      * "nonempty" -- the caller states that an empty container means the call failed.
      * "numeric"  -- finite numbers required; all-NaN/inf fails. Catches a solver that "succeeded"
                    into garbage, which raises nothing and looks like a result.
      * "truthy"   -- the strictest, for calls whose whole purpose is to return something falsy-if-broken.

    KEPT NEGATIVE: this cannot detect a WRONG answer, only an absent or malformed one. A tool that
    confidently returns the wrong number passes every check here, and no value-shaped test will catch it
    -- that needs a verifier, which is a different and much harder instrument."""
    if expect not in ("any", "nonempty", "numeric", "truthy"):
        raise ValueError("expect must be any/nonempty/numeric/truthy, got %r" % (expect,))
    if _is_error_shaped(value):
        return {"usable": False, "reason": "error-shaped return", "expect": expect}
    if value is None:
        return {"usable": False, "reason": "returned None", "expect": expect}
    if expect == "any":
        return {"usable": True, "reason": "not None and not error-shaped", "expect": expect}
    if expect == "nonempty" and _empty(value):
        return {"usable": False, "reason": "empty container (caller declared nonempty)", "expect": expect}
    if expect == "numeric":
        if _empty(value):
            return {"usable": False, "reason": "empty where numbers were expected", "expect": expect}
        if _all_nan(value):
            return {"usable": False, "reason": "all values non-finite", "expect": expect}
    if expect == "truthy" and not value:
        return {"usable": False, "reason": "falsy result (caller declared truthy)", "expect": expect}
    return {"usable": True, "reason": "meets the declared postcondition", "expect": expect}


def guarded_call(fn, *args, expect="any", **kw):
    """Call `fn`, then CHECK the result -- both failure modes in one place.

    Returns {ok, value, reason, raised}. An exception is the LOUD failure and was already caught by
    ordinary flow; an unusable return is the QUIET one that BENCH-3 measures at 0.50 and nothing else
    catches. Reporting them through one verdict is what lets a loop abandon on either."""
    try:
        v = fn(*args, **kw)
    except Exception as e:
        return {"ok": False, "value": None, "raised": "%s: %s" % (type(e).__name__, str(e)[:80]),
                "reason": "call raised"}
    chk = result_usable(v, expect=expect)
    return {"ok": bool(chk["usable"]), "value": v, "raised": None, "reason": chk["reason"]}


# ---------------------------------------------------------------------------
# RETURN CONTRACTS -- the EXIT gate (sweep 122, NOOA item 2).
#
# WHY here and not in a sibling module: this is the same instrument as
# result_usable with a memory. A separate module would repeat the caller-declares
# rule above, and two faculties that do 80% of the same thing is a discoverability
# tax the audits cannot see. The tension in the module docstring is UNCHANGED and
# load-bearing: a contract is something the CALLER declares, never something this
# file assumes, because an empty return is often the correct answer.
# ---------------------------------------------------------------------------

MAX_RETRIES = 8            # a bound, refused rather than clamped -- see validated_call
DEFAULT_RETRIES = 1        # one retry, and only when a contract was actually declared


def _type_ok(v, tname):
    """Does `v` satisfy the declared type name? Unknown names raise -- see normalize_contract.

    KEPT NEGATIVE, pinned in the selftest: bool is a subclass of int in Python, so a naive
    isinstance lets a step declared to return a COUNT return True and pass. A malformed
    return that type-checks is exactly what this gate exists to catch, so int/number
    exclude bool explicitly."""
    if tname == "any":
        return True
    if tname == "int":
        return isinstance(v, int) and not isinstance(v, bool)
    if tname == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if tname == "float":
        return isinstance(v, float)
    if tname == "bool":
        return isinstance(v, bool)
    if tname == "str":
        return isinstance(v, str)
    if tname == "bytes":
        return isinstance(v, (bytes, bytearray))
    if tname == "list":
        return isinstance(v, list)
    if tname == "tuple":
        return isinstance(v, tuple)
    if tname == "dict":
        return isinstance(v, dict)
    if tname == "callable":
        return callable(v)
    if tname == "array":
        # duck-typed on purpose: this module stays import-light (numpy is only pulled in
        # lazily by _all_nan), and anything carrying shape+dtype answers the question.
        return hasattr(v, "shape") and hasattr(v, "dtype")
    raise ValueError("unknown type name %r in contract" % (tname,))


def _preview(v, limit=140):
    """A BOUNDED rendering of a rejected value, for the error handed back to the executor.

    Deliberately local and tiny: the feedback string exists to tell a model WHAT it returned,
    and a full repr of a rejected million-row array is the context blow-up the whole exit gate
    is meant to avoid. True length is reported alongside so the bound is never mistaken for
    the value."""
    try:
        n = len(v)
    except Exception:
        n = None
    try:
        s = repr(v)
    except Exception:
        s = "<unreprable %s>" % type(v).__name__
    if len(s) > limit:
        s = s[:limit] + "..."
    return s if n is None else "%s (len %d)" % (s, n)


def normalize_contract(contract):
    """Canonicalise a caller-declared return contract. Returns the full dict form.

    Accepted forms, all of which say the same thing:
      * None                 -- nothing declared. expect="any", and therefore NO retries:
                                a caller who declared nothing has not asked for a second try.
      * "nonempty"           -- shorthand for {"expect": "nonempty"}.
      * {"expect": "nonempty", "require": ["evidence", "verify"],
         "types": {"evidence": "list", "verify": "str"}, "retries": 2}

    Keys:
      expect    -- one of any/nonempty/numeric/truthy, handed straight to result_usable.
      require   -- dict keys that must be PRESENT and neither None nor an empty container.
                   This is NOOA's typed return: a step that must show its evidence and the
                   command that verifies it declares require=["evidence", "verify"].
                   Note 0 and False are NOT empty containers, so a required count of 0 passes.
      types     -- {key: type-name}, checked only when the key is present; pair with `require`
                   to demand both. Type names: any/int/float/number/bool/str/bytes/list/tuple/
                   dict/callable/array.
      retries   -- how many EXTRA attempts validated_call may make (default 1, hard cap 8).
      predicate -- an optional callable(value) -> bool for anything the dict form cannot say.
                   IN-PROCESS ONLY: it cannot cross /invoke, and it is excluded from the
                   digest, so two contracts differing only by predicate share an id.

    A misspelled key is REFUSED rather than ignored -- a contract that silently drops the half
    you cared about is a gate that has quietly stopped gating."""
    if contract is None:
        return {"expect": "any", "require": (), "types": {}, "retries": 0, "predicate": None}
    if isinstance(contract, str):
        contract = {"expect": contract}
    if not isinstance(contract, dict):
        raise ValueError("contract must be None, an expect-string or a dict, got %r"
                         % (type(contract).__name__,))
    unknown = set(contract) - {"expect", "require", "types", "retries", "predicate"}
    if unknown:
        raise ValueError("unknown contract keys %s" % (sorted(unknown),))
    expect = contract.get("expect", "any")
    if expect not in ("any", "nonempty", "numeric", "truthy"):
        raise ValueError("expect must be any/nonempty/numeric/truthy, got %r" % (expect,))
    require = tuple(contract.get("require") or ())
    types = dict(contract.get("types") or {})
    for k, tn in types.items():
        _type_ok(None, tn)                 # raises now, at declaration, not mid-retry
    retries = contract.get("retries", DEFAULT_RETRIES)
    pred = contract.get("predicate")
    if pred is not None and not callable(pred):
        raise ValueError("contract predicate must be callable")
    return {"expect": expect, "require": require, "types": types,
            "retries": int(retries), "predicate": pred}


def contract_digest(contract):
    """A stable short id for a contract -- hashlib over its canonical JSON form, never hash().

    WHY: the audit record names WHICH contract was enforced without re-dumping it on every
    attempt, and the id must be identical across processes and runs (PYTHONHASHSEED-proof).
    The predicate is excluded because a callable has no canonical form; contracts differing
    only by predicate therefore share an id, which is stated here so nobody reads the id as
    proof of full equality."""
    import hashlib
    import json as _json
    spec = normalize_contract(contract)          # idempotent, so an already-normalised
                                                 # spec digests to the same id as its source
    body = {"expect": spec["expect"],
            "require": sorted(spec["require"]),
            "types": {k: spec["types"][k] for k in sorted(spec["types"])},
            "retries": spec["retries"],
            "predicate": bool(spec["predicate"])}
    return hashlib.sha1(_json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]


def check_contract(value, contract=None):
    """Does `value` satisfy the caller's declared return contract?

    Returns {ok, reason, violations, empty, expect, contract}. `empty` is the distinction that
    keeps abstention safe: a value can be ACCEPTED and EMPTY at the same time (the tool
    correctly returned nothing), which is a different fact from a tool that failed, and the
    two must never be collapsed into one verdict.

    The checks run expect -> require -> types -> predicate and report EVERY violation, not the
    first: a retry whose error message names one of three problems buys one round-trip per
    problem."""
    spec = normalize_contract(contract)
    base = result_usable(value, expect=spec["expect"])
    viol = []
    if not base["usable"]:
        viol.append(base["reason"])
    for k in spec["require"]:
        if not isinstance(value, dict):
            viol.append("required field %r: return is %s, not a dict"
                        % (k, type(value).__name__))
        elif k not in value:
            viol.append("required field %r is missing" % (k,))
        elif value[k] is None:
            viol.append("required field %r is None" % (k,))
        elif _empty(value[k]):
            viol.append("required field %r is empty" % (k,))
    for k, tn in spec["types"].items():
        if isinstance(value, dict) and k in value and not _type_ok(value[k], tn):
            viol.append("field %r must be %s, got %s"
                        % (k, tn, type(value[k]).__name__))
    if spec["predicate"] is not None and not viol:
        try:
            if not spec["predicate"](value):
                viol.append("caller predicate rejected the value")
        except Exception as e:
            viol.append("caller predicate raised %s: %s" % (type(e).__name__, str(e)[:60]))
    empty = _empty(value)
    if viol:
        reason = "; ".join(viol)
    elif empty:
        reason = "accepted AND empty -- a deliberate abstention is a result, not a failure"
    else:
        reason = "meets the declared contract"
    return {"ok": not viol, "reason": reason, "violations": viol, "empty": empty,
            "expect": spec["expect"], "contract": contract_digest(spec)}


def contract_error(spec, verdict, attempt, value, raised=None):
    """The TYPED error handed back to a retrying executor. Error-shaped ON PURPOSE.

    It carries an `error` key, so an executor that gives up and echoes this dict straight back
    is caught by _is_error_shaped on the next check instead of being scored as a pass. That is
    the one failure mode a feedback channel invents, and it is closed here."""
    return {"error": "contract_violation",
            "attempt": int(attempt),
            "reason": verdict.get("reason"),
            "violations": list(verdict.get("violations") or ()),
            "expect": spec.get("expect"),
            "require": list(spec.get("require") or ()),
            "types": dict(spec.get("types") or {}),
            "contract": contract_digest(spec),
            "raised": raised,
            "returned_type": type(value).__name__,
            "returned": _preview(value)}


def _accepts_feedback(fn):
    """Can this executor be TOLD why it was rejected? True for a `feedback` parameter or **kw.

    An executor that cannot take the error is still retried -- a flaky backend often succeeds
    on the second call -- but a plain retry and an informed one are different things, and the
    record says which happened."""
    import inspect
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False                       # builtins and C callables have no signature
    for p in sig.parameters.values():
        if p.kind is p.VAR_KEYWORD:
            return True
        if p.name == "feedback" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            return True
    return False


def validated_call(fn, contract=None, retries=None, name="step", record=None):
    """Call `fn`, hold its RETURN to a contract, and retry a bounded number of times.

    This is guarded_call plus the exit gate: guarded_call answers "was that usable ONCE",
    this answers "make it usable or say loudly why it never was". Returns
    {ok, value, name, attempts, retries_used, informed_retries, abstained, reason, raised,
     trace, contract} -- `trace` has one row per attempt, so a retry is never silent.

    Determinism: attempt 1 is ALWAYS a plain zero-argument call, byte-identical to calling
    `fn()` yourself; retries pass the typed error as `feedback=` when the executor accepts one.
    No sleeps, no wall clock, no randomness -- same inputs give the same attempt sequence.

    `retries` bounds the EXTRA attempts (total = retries + 1). It is refused above MAX_RETRIES
    rather than clamped: a clamped bound is a bound the caller does not know about."""
    spec = normalize_contract(contract)
    n_retry = spec["retries"] if retries is None else int(retries)
    if n_retry < 0 or n_retry > MAX_RETRIES:
        raise ValueError("retries must be 0..%d, got %r" % (MAX_RETRIES, n_retry))
    informed = _accepts_feedback(fn)
    trace, fb, value = [], None, None
    for i in range(n_retry + 1):
        raised = None
        try:
            value = fn() if (i == 0 or not informed) else fn(feedback=fb)
        except Exception as e:
            value = None
            raised = "%s: %s" % (type(e).__name__, str(e)[:100])
        if raised is not None:
            verdict = {"ok": False, "reason": "call raised", "violations": ["raised " + raised],
                       "empty": False, "expect": spec["expect"]}
        else:
            verdict = check_contract(value, spec)
        trace.append({"attempt": i + 1, "ok": bool(verdict["ok"]), "reason": verdict["reason"],
                      "raised": raised, "empty": bool(verdict.get("empty")),
                      "violations": list(verdict.get("violations") or ()),
                      "informed": bool(informed and i > 0)})
        if verdict["ok"]:
            break
        fb = contract_error(spec, verdict, i + 1, value, raised)
        value = None                       # a rejected value must never leak out as a result
    last = trace[-1]
    out = {"ok": bool(last["ok"]), "value": value if last["ok"] else None, "name": str(name),
           "attempts": len(trace), "retries_used": len(trace) - 1,
           "informed_retries": sum(1 for t in trace if t["informed"]),
           "abstained": bool(last["ok"] and last["empty"]),
           "reason": last["reason"], "raised": last["raised"],
           "contract": contract_digest(spec), "trace": trace}
    if record is not None:
        record.append(out)
    return out


def wrap_executors(executors, contracts, record=None):
    """Put the exit gate under an agent loop's executors. Returns (wrapped_executors, journal).

    `contracts` maps STEP NAME -> contract (any form normalize_contract accepts); the key "*"
    declares a default for every executor that has no entry of its own. An executor with NO
    contract is returned UNTOUCHED -- not wrapped, not recorded -- so the old path stays the
    old path down to object identity.

    Each wrapper is a zero-argument callable, because that is exactly what goal_work calls, and
    it returns None when the contract is finally unmet. WHY None and not the error dict: goal_work
    reads `out is not None` as the step's verdict, so returning the typed error would mark a
    contract FAILURE as a completed step -- the precise quiet failure this module exists to stop.
    The reason is not lost, it is in the journal.

    KEPT LIMITATION, on record: a step served from the trajectory tool cache never reaches its
    executor, so it is not re-validated. Only values that passed a contract are ever cached
    (a failed wrapper returns None, which goal_work refuses to cache and invalidates), but a
    cache filled BEFORE contracts were declared is not retro-checked."""
    ex = dict(executors or {})
    specs = dict(contracts or {})
    default = specs.pop("*", None)
    journal = {"calls": [], "wrapped": [], "contracts": {},
               "declared_without_executor": sorted(set(specs) - set(ex))}
    if record is not None:
        journal["calls"] = record
    wrapped = {}
    for nm in sorted(ex):                  # sorted: construction order is deterministic
        spec = specs.get(nm, default)
        if spec is None:
            wrapped[nm] = ex[nm]
            continue
        norm = normalize_contract(spec)
        journal["wrapped"].append(nm)
        journal["contracts"][nm] = contract_digest(norm)
        wrapped[nm] = _contract_wrapper(ex[nm], norm, nm, journal["calls"])
    return wrapped, journal


def _contract_wrapper(fn, spec, name, calls):
    """One zero-argument executor with its contract and its journal attached."""
    def _validated():
        r = validated_call(fn, contract=spec, name=name, record=calls)
        return r["value"] if r["ok"] else None
    _validated.__doc__ = "contract-validated %r (%s)" % (name, contract_digest(spec))
    return _validated


def validation_report(journal):
    """Summarise a wrap_executors journal into the audit record a loop returns to its caller.

    Reports `false_retries` explicitly, and it is the number to watch: a call that ENDED as an
    accepted abstention but spent a retry getting there means the gate treated a correct empty
    answer as a failure. It must be 0. A retry on a caller-declared `nonempty` contract is NOT
    counted here -- that retry is the caller's own instruction, not a false positive."""
    calls = list((journal or {}).get("calls") or ())
    by_step = {}
    for c in calls:
        s = by_step.setdefault(c["name"], {"calls": 0, "passed": 0, "failed": 0,
                                           "attempts": 0, "retries": 0, "abstained": 0})
        s["calls"] += 1
        s["passed" if c["ok"] else "failed"] += 1
        s["attempts"] += c["attempts"]
        s["retries"] += c["retries_used"]
        s["abstained"] += 1 if c["abstained"] else 0
    return {"wrapped": list((journal or {}).get("wrapped") or ()),
            "contracts": dict((journal or {}).get("contracts") or {}),
            "declared_without_executor": list(
                (journal or {}).get("declared_without_executor") or ()),
            "calls": len(calls),
            "passed": sum(1 for c in calls if c["ok"]),
            "failed": sum(1 for c in calls if not c["ok"]),
            "attempts": sum(c["attempts"] for c in calls),
            "retries_issued": sum(c["retries_used"] for c in calls),
            "informed_retries": sum(c["informed_retries"] for c in calls),
            "abstentions": sum(1 for c in calls if c["abstained"]),
            "false_retries": sum(1 for c in calls if c["abstained"] and c["retries_used"] > 0),
            "by_step": by_step,
            "failures": [{"name": c["name"], "reason": c["reason"], "raised": c["raised"],
                          "attempts": c["attempts"],
                          "violations": c["trace"][-1]["violations"]}
                         for c in calls if not c["ok"]],
            "trace": [{"name": c["name"], "attempts": c["attempts"], "ok": c["ok"],
                       "reason": c["reason"]} for c in calls]}


def _selftest():
    import numpy as np

    # 1. THE LOAD-BEARING CASE: an EMPTY result must PASS under the default. route_or_abstain returns
    #    hits: [] on a deliberate abstention -- flagging that as failure would turn leCore's core safety
    #    property into a bug report, which is the single way this module could do harm.
    assert result_usable([])["usable"] is True
    assert result_usable({"hits": [], "abstain": True})["usable"] is True
    #    ...and it fails ONLY when the caller says so.
    assert result_usable([], expect="nonempty")["usable"] is False

    # 2. THE QUIET FAILURE BENCH-3 MEASURES: None is unusable at every level.
    assert result_usable(None)["usable"] is False
    assert "None" in result_usable(None)["reason"]

    # 3. ERROR-SHAPED returns are caught, and the test is DELIBERATELY NARROW -- a dict containing a
    #    False must not be read as failure, or half the reports in this codebase become errors.
    assert result_usable({"error": "backend down"})["usable"] is False
    assert result_usable(ValueError("x"))["usable"] is False
    assert result_usable({"ok": False, "abstain": True})["usable"] is True, "over-eager error detection"
    assert result_usable({"error": None, "value": 3})["usable"] is True, "a null error field is not an error"

    # 4. NUMERIC: a solver that "succeeds" into NaN raises nothing and looks like a result.
    assert result_usable(np.array([np.nan, np.nan]), expect="numeric")["usable"] is False
    assert result_usable(np.array([0.0, 0.0]), expect="numeric")["usable"] is True, \
        "all-zero is a RESULT, not an absence"

    # 5. guarded_call reports BOTH failure modes through one verdict, which is what lets a loop abandon
    #    on either without two code paths.
    def boom():
        raise RuntimeError("backend unavailable")
    r = guarded_call(boom)
    assert r["ok"] is False and "raised" in r["reason"] and "RuntimeError" in r["raised"]
    q = guarded_call(lambda: None)
    assert q["ok"] is False and q["raised"] is None, "a quiet failure must not be reported as a raise"
    g = guarded_call(lambda: {"v": 1})
    assert g["ok"] is True and g["value"] == {"v": 1}

    # 6. An unknown postcondition is REFUSED, not silently treated as "any" -- a typo'd expect that
    #    degrades to the weakest check is a guard that quietly stops guarding.
    try:
        result_usable(1, expect="nonemty")
        raise AssertionError("a typo'd postcondition was accepted")
    except ValueError:
        pass
    # ---- RETURN CONTRACTS (sweep 122): the exit gate. Every assertion below pins a NUMBER
    #      or an exact verdict; "it did not raise" would not have caught a single bug here.

    # 7. THE HEADLINE SAFETY PROPERTY, and the one that must never regress: an executor that
    #    CORRECTLY ABSTAINS costs ZERO retries. If this number is ever non-zero, the exit gate
    #    has started eating leCore's entry gate, and that trade is never worth making.
    abstain = lambda: []
    r = validated_call(abstain, contract={"retries": 3})
    assert r["attempts"] == 1 and r["retries_used"] == 0, "abstention triggered a retry storm"
    assert r["ok"] is True and r["abstained"] is True and r["value"] == []

    # 8. FLAKY EXECUTOR: fails once, then succeeds -- EXACTLY 2 attempts, no more, no fewer.
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        return None if state["n"] == 1 else {"rows": [1, 2]}
    r = validated_call(flaky, contract={"expect": "nonempty"})
    assert r["attempts"] == 2 and r["retries_used"] == 1 and r["ok"] is True, r
    assert r["value"] == {"rows": [1, 2]} and state["n"] == 2

    # 9. THE BOUND IS A BOUND: a permanently broken executor is called retries+1 times and
    #    then STOPS, and the count is exact.
    calls = {"n": 0}

    def broken():
        calls["n"] += 1
        return None
    r = validated_call(broken, contract={"retries": 2})
    assert calls["n"] == 3 and r["attempts"] == 3 and r["ok"] is False, (calls, r)
    assert len(r["trace"]) == 3, "a retry that leaves no trace row is a silent retry"
    try:
        validated_call(broken, retries=MAX_RETRIES + 1)
        raise AssertionError("an unbounded retry budget was accepted")
    except ValueError:
        pass

    # 10. THE TYPED RETURN (NOOA's shape): evidence + a verification command, and the rejected
    #     executor is TOLD which field it missed rather than merely being called again.
    seen = {}

    def needs_feedback(feedback=None):
        seen["fb"] = feedback
        if feedback is None:
            return {"answer": 42}
        return {"answer": 42, "evidence": ["ran the test"], "verify": "pytest -q"}
    c = {"expect": "nonempty", "require": ["evidence", "verify"],
         "types": {"evidence": "list", "verify": "str"}}
    r = validated_call(needs_feedback, contract=c)
    assert r["attempts"] == 2 and r["ok"] is True and r["informed_retries"] == 1
    assert seen["fb"]["error"] == "contract_violation"
    assert seen["fb"]["violations"] == ["required field 'evidence' is missing",
                                        "required field 'verify' is missing"], seen["fb"]
    #     ...and the feedback is ERROR-SHAPED, so an executor that echoes it back is caught.
    assert result_usable(seen["fb"])["usable"] is False

    # 11. bool is an int subclass: a step declared to return a COUNT must not pass with True.
    v = check_contract({"n": True}, {"types": {"n": "int"}})
    assert v["ok"] is False and "must be int" in v["reason"], v
    assert check_contract({"n": 0}, {"types": {"n": "int"}})["ok"] is True, \
        "0 is a count, not an absence"
    #     a required field that is present-but-empty is a violation; 0 is NOT empty.
    assert check_contract({"evidence": []}, {"require": ["evidence"]})["ok"] is False
    assert check_contract({"evidence": 0}, {"require": ["evidence"]})["ok"] is True

    # 12. NOTHING DECLARED, NOTHING RETRIED: contract=None gives 0 retries and the old verdict.
    calls["n"] = 0
    r = validated_call(broken)
    assert calls["n"] == 1 and r["attempts"] == 1 and r["ok"] is False
    #     and a misspelled contract key is refused, not ignored.
    for bad in ({"expct": "nonempty"}, {"expect": "nonemty"}, {"types": {"k": "integer"}}):
        try:
            normalize_contract(bad)
            raise AssertionError("a malformed contract was accepted: %r" % (bad,))
        except ValueError:
            pass
    #     the digest is content-addressed and stable across processes (hashlib, never hash()).
    assert contract_digest({"expect": "nonempty"}) == contract_digest("nonempty")
    assert contract_digest({"expect": "nonempty"}) != contract_digest({"expect": "truthy"})

    # 13. wrap_executors: an executor with NO contract is the SAME OBJECT, not a wrapper --
    #     the old path stays the old path down to object identity.
    plain = lambda: "x"
    w, j = wrap_executors({"a": plain, "b": abstain}, {"b": {"expect": "any"}})
    assert w["a"] is plain, "an undeclared executor was wrapped anyway"
    assert w["b"] is not abstain and j["wrapped"] == ["b"]
    w["b"]()
    rep = validation_report(j)
    assert rep["calls"] == 1 and rep["abstentions"] == 1 and rep["false_retries"] == 0, rep
    #     a contract naming a step with no executor is REPORTED, never silently dropped.
    _, j2 = wrap_executors({"a": plain}, {"ghost": "nonempty"})
    assert validation_report(j2)["declared_without_executor"] == ["ghost"]

    print("postcheck selftest OK -- empty PASSES by default (abstention is not a bug); None and "
          "error-shaped fail; all-zero is a result and all-NaN is not; guarded_call reports loud and "
          "quiet failures through one verdict; a typo'd expect is refused")
    print("postcheck contracts OK -- abstention costs 0 retries; the flaky executor takes exactly "
          "2 attempts and the broken one exactly retries+1; a rejected executor is handed an "
          "error-shaped typed violation; True is not an int and 0 is not empty; an undeclared "
          "executor is the same object")


if __name__ == "__main__":
    _selftest()

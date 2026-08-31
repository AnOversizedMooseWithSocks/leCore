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
and says so. Anything stronger is opt-in."""
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
    print("postcheck selftest OK -- empty PASSES by default (abstention is not a bug); None and "
          "error-shaped fail; all-zero is a result and all-NaN is not; guarded_call reports loud and "
          "quiet failures through one verdict; a typo'd expect is refused")


if __name__ == "__main__":
    _selftest()

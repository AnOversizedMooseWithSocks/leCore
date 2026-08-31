"""holographic_soprunner.py -- FOLLOW ORDERS: run an AUTHORED text SOP through the mind.

THE DIVISION OF LABOUR (the doctrine this module exists to enforce): planning is
a good model task; execution is not. A model writes the SOP -- a plain-text
standard operating procedure -- and the substrate runs it: invoking its own
faculties, sandboxing code, verifying every step with its own instruments, and
consulting the model ONLY at declared guidance points or on a declared
escalation. On the happy path the model is called exactly as many times as the
SOP declares guidance steps -- zero, usually -- and the selftest pins that count.

PORTED DOCTRINE, CREDITED (leOS): bone chains (linear steps, a hard step cap,
graceful per-step degradation with an honest log -- sdol/app_executor); bots'
escalate-to-Director (the model is consulted on a SIGNAL, never per step --
bots/bot_runner_act); macro_registry (named procedures saved and reloaded by
name -- sdol/macro_registry).

THE SOP FORMAT (all a model needs to know to give orders):

    # any title
    ## step: map the tree
    invoke: repo_map {"root": ".", "budget_lines": 40}
    store: tree
    verify: result["files"] > 0

    ## step: try the snippet
    python: print(6*7)
    verify: "42" in result["stdout"]
    on_fail: retry 2

    ## step: sanity-check direction
    guidance: does the map above look like the right tree to edit?

One directive per line. `invoke:` calls a PUBLIC mind faculty with JSON kwargs;
`python:`/`javascript:`/`c:` run through sandbox_run; `shell:` runs through the
allowlisted run_command; `verify:` is a boolean expression over `result` (this
step's output) and `state` (everything `store:`d so far), evaluated by a
whitelisted AST walker -- never eval/exec; `on_fail:` is abort (default),
continue, retry N, or escalate; `guidance:`/`escalate` are the ONLY roads to the
model. An SOP that cannot be FULLY parsed refuses to run at all -- refusing to
follow half-understood orders is the abstain-not-error rule applied to orders.

KEPT NEGATIVES (loud):
- eval()/exec() for verify expressions REJECTED: the expression comes from the
  model. The AST walker admits comparisons, boolean/arith ops, literals,
  subscripts, and len/str/int/float/min/max/abs on the two names -- nothing
  else. Attribute access is refused OUTRIGHT (dunder-crawling to object() is
  the classic escape).
- Executing a partially-parsed SOP REJECTED: a skipped unknown directive is an
  order silently dropped -- worse than a refusal.
- Wall-clock step budgets REJECTED (leOS activity_monitor lesson via
  agent_loop): the cap is a STEP count, deterministic by construction.
- goto/branching NOT in v1: a linear SOP's log reads top to bottom; the leOS
  goto machinery is noted for v2 only if a real SOP needs it.
"""
import ast
import json
import re
import time


# One directive per line: keyword, colon, payload. Anything that looks like a
# directive but is not in this table is a PARSE REFUSAL, not a skip.
_DIRECTIVES = ("invoke", "python", "javascript", "c", "shell", "verify",
               "on_fail", "store", "guidance")
_STEP_RE = re.compile(r"^##\s*step\s*:\s*(.+)$", re.IGNORECASE)
_DIRECTIVE_RE = re.compile(r"^([a-z_]+)\s*:\s*(.*)$")

# The verify-expression whitelist. Names beyond these two and calls beyond this
# table are refused by the walker below.
_VERIFY_NAMES = ("result", "state")
_VERIFY_CALLS = {"len": len, "str": str, "int": int, "float": float,
                 "min": min, "max": max, "abs": abs}
_VERIFY_NODES = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp,
                 ast.Not, ast.USub, ast.Compare, ast.Eq, ast.NotEq, ast.Lt,
                 ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is,
                 ast.IsNot, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
                 ast.Mod, ast.Constant, ast.Subscript, ast.Index, ast.Name,
                 ast.Load, ast.Call, ast.Tuple, ast.List)


def safe_verify(expr, result, state):
    """Evaluate a verify expression with the whitelisted AST walker.

    DELIBERATELY SEPARATE from holographic_mathcheck.evaluate (read in the
    dedup sweep): that walker is numerically hardened (Pow bounds, literal
    caps, refuse-by-raise) for arithmetic claims in prose; this one speaks a
    BOOLEAN grammar (comparisons, names, subscripts, whitelisted calls) and
    reports refusals as data. One body would couple the two contracts.

    Returns (ok, why): ok is the boolean outcome; why is "" on a clean
    evaluation or the refusal/error reason. A refused expression is NOT a
    failed verification -- the caller reports it as a blocked step, because
    "your expression is outside the grammar" and "your check came back false"
    demand different fixes from the SOP's author.
    """
    try:
        tree = ast.parse(str(expr), mode="eval")
    except SyntaxError as e:
        return False, "verify does not parse: %s" % e
    for node in ast.walk(tree):
        if not isinstance(node, _VERIFY_NODES):
            # Attribute lands here on purpose: refusing the whole node class
            # closes the dunder escape without enumerating dunders.
            return False, "verify refused: %s is outside the whitelist" % (
                type(node).__name__)
        if isinstance(node, ast.Name) and node.id not in _VERIFY_NAMES \
                and node.id not in _VERIFY_CALLS:
            return False, "verify refused: unknown name %r" % node.id
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) \
                    or node.func.id not in _VERIFY_CALLS:
                return False, "verify refused: only %s may be called" % (
                    sorted(_VERIFY_CALLS))
    def _run(n):
        if isinstance(n, ast.Expression):
            return _run(n.body)
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            return {"result": result, "state": state}.get(n.id,
                                                          _VERIFY_CALLS.get(n.id))
        if isinstance(n, ast.Tuple) or isinstance(n, ast.List):
            return [_run(x) for x in n.elts]
        if isinstance(n, ast.Subscript):
            sl = n.slice
            if isinstance(sl, ast.Index):        # < py3.9 shape, kept for zip portability
                sl = sl.value
            return _run(n.value)[_run(sl)]
        if isinstance(n, ast.UnaryOp):
            v = _run(n.operand)
            return (not v) if isinstance(n.op, ast.Not) else -v
        if isinstance(n, ast.BoolOp):
            vals = [_run(v) for v in n.values]
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.BinOp):
            a, b = _run(n.left), _run(n.right)
            return {ast.Add: lambda: a + b, ast.Sub: lambda: a - b,
                    ast.Mult: lambda: a * b, ast.Div: lambda: a / b,
                    ast.Mod: lambda: a % b}[type(n.op)]()
        if isinstance(n, ast.Compare):
            left = _run(n.left)
            for op, cmp_ in zip(n.ops, n.comparators):
                right = _run(cmp_)
                ok = {ast.Eq: lambda: left == right,
                      ast.NotEq: lambda: left != right,
                      ast.Lt: lambda: left < right, ast.LtE: lambda: left <= right,
                      ast.Gt: lambda: left > right, ast.GtE: lambda: left >= right,
                      ast.In: lambda: left in right,
                      ast.NotIn: lambda: left not in right,
                      ast.Is: lambda: left is right,
                      ast.IsNot: lambda: left is not right}[type(op)]()
                if not ok:
                    return False
                left = right
            return True
        if isinstance(n, ast.Call):
            return _VERIFY_CALLS[n.func.id](*[_run(a) for a in n.args])
        raise ValueError("unreachable node %s" % type(n).__name__)
    try:
        return bool(_run(tree)), ""
    except Exception as e:
        return False, "verify raised: %s: %s" % (type(e).__name__, str(e)[:120])


def parse_sop(text):
    """Parse an authored SOP into steps, or REFUSE with every problem named.

    Returns {"ok": True, "title", "steps": [...]} or {"ok": False, "errors":
    ["line N: ..."]}. All-or-nothing on purpose: an SOP with one
    ununderstood line has NOT been understood -- see the module docstring's
    kept negative on partially-parsed orders.
    """
    steps, errors, title = [], [], ""
    cur = None
    for ln, raw in enumerate(str(text).split("\n"), 1):
        line = raw.strip()
        if not line or line.startswith("#") and not line.startswith("##"):
            if line.startswith("# ") and not title:
                title = line[2:].strip()
            continue
        mstep = _STEP_RE.match(line)
        if mstep:
            cur = {"name": mstep.group(1).strip(), "line": ln, "action": None,
                   "verify": None, "on_fail": ("abort", 0), "store": None}
            steps.append(cur)
            continue
        m = _DIRECTIVE_RE.match(line)
        if not m:
            errors.append("line %d: not a directive or step header: %r"
                          % (ln, line[:60]))
            continue
        key, payload = m.group(1), m.group(2).strip()
        if key not in _DIRECTIVES:
            errors.append("line %d: unknown directive %r (know: %s)"
                          % (ln, key, ", ".join(_DIRECTIVES)))
            continue
        if cur is None:
            errors.append("line %d: %r before any '## step:' header" % (ln, key))
            continue
        if key == "invoke":
            name, _, rest = payload.partition(" ")
            try:
                kwargs = json.loads(rest) if rest.strip() else {}
                assert isinstance(kwargs, dict)
            except Exception:
                errors.append("line %d: invoke kwargs must be a JSON object, "
                              "got %r" % (ln, rest[:40]))
                continue
            if name.startswith("_"):
                errors.append("line %d: invoke of private %r refused" % (ln, name))
                continue
            _set_action(cur, errors, ln, ("invoke", name, kwargs))
        elif key in ("python", "javascript", "c"):
            _set_action(cur, errors, ln, ("code", key, payload))
        elif key == "shell":
            _set_action(cur, errors, ln, ("shell", payload, None))
        elif key == "guidance":
            _set_action(cur, errors, ln, ("guidance", payload, None))
        elif key == "verify":
            cur["verify"] = payload
        elif key == "store":
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", payload):
                errors.append("line %d: store key %r is not an identifier"
                              % (ln, payload))
            else:
                cur["store"] = payload
        elif key == "on_fail":
            mm = re.fullmatch(r"(abort|continue|escalate|retry\s+(\d+))", payload)
            if not mm:
                errors.append("line %d: on_fail must be abort|continue|"
                              "escalate|retry N, got %r" % (ln, payload))
            elif mm.group(2):
                cur["on_fail"] = ("retry", int(mm.group(2)))
            else:
                cur["on_fail"] = (mm.group(1), 0)
    for s in steps:
        if s["action"] is None:
            errors.append("line %d: step %r has no action" % (s["line"], s["name"]))
    if not steps and not errors:
        errors.append("no '## step:' headers found")
    if errors:
        return {"ok": False, "errors": errors}
    return {"ok": True, "title": title, "steps": steps}


def _set_action(cur, errors, ln, action):
    # Two actions in one step is an ambiguous order -- refused, not last-wins.
    if cur["action"] is not None:
        errors.append("line %d: step %r already has an action" % (ln, cur["name"]))
    else:
        cur["action"] = action


class SOPRunner:
    """Execute a parsed SOP against a mind. One runner per run; the log is
    the product. See the module docstring for the format and the doctrine."""

    def __init__(self, mind, llm=None, max_steps=50):
        self.mind = mind
        self.llm = llm
        # leOS bone-chain lesson kept: a hard STEP cap (which retries also
        # spend), never a wall clock.
        self.max_steps = int(max_steps)
        self.llm_calls = 0

    def run(self, sop_text):
        parsed = parse_sop(sop_text)
        if not parsed["ok"]:
            return {"ok": False, "refused": True, "errors": parsed["errors"],
                    "log": [], "llm_calls": 0}
        state, log, spent, aborted = {}, [], 0, False
        for step in parsed["steps"]:
            if aborted:
                log.append({"step": step["name"], "status": "skipped",
                            "detail": "after abort"})
                continue
            tries = 1 + (step["on_fail"][1] if step["on_fail"][0] == "retry" else 0)
            entry = None
            for attempt in range(tries):
                if spent >= self.max_steps:
                    entry = {"step": step["name"], "status": "blocked",
                             "detail": "step budget %d exhausted" % self.max_steps}
                    aborted = True
                    break
                spent += 1
                entry = self._attempt(step, state, attempt)
                if entry["status"] == "fired":
                    break
            if entry["status"] != "fired" and not aborted:
                mode = step["on_fail"][0]
                if mode == "escalate" and self.llm is not None:
                    verdict = self._escalate(step, entry, state)
                    entry["escalation"] = verdict
                    if not str(verdict).lower().startswith("continue"):
                        aborted = True
                elif mode != "continue":
                    aborted = True
            if entry["status"] == "fired" and step["store"]:
                state[step["store"]] = entry.get("result")
            # The stored log carries VERDICTS and small details, not payloads --
            # results live in `state` for the caller.
            log.append({k: v for k, v in entry.items() if k != "result"})
        return {"ok": not aborted, "title": parsed["title"], "log": log,
                "state_keys": sorted(state), "state": state,
                "steps_spent": spent, "llm_calls": self.llm_calls}

    def _attempt(self, step, state, attempt):
        kind, a, b = step["action"]
        t0 = time.time()
        entry = {"step": step["name"], "attempt": attempt, "status": "fired",
                 "detail": ""}
        try:
            if kind == "invoke":
                fn = getattr(self.mind, a, None)
                if fn is None or not callable(fn) or a.startswith("_"):
                    entry.update(status="blocked",
                                 detail="no public faculty %r" % a)
                    return entry
                entry["result"] = fn(**b)
            elif kind == "code":
                entry["result"] = self.mind.sandbox_run(b, lang=a)
                if not entry["result"].get("ok"):
                    entry.update(status="failed", detail="sandbox: %s"
                                 % entry["result"].get("why",
                                                       entry["result"].get("stderr", ""))[:160])
                    return entry
            elif kind == "shell":
                # run_command's contract: the allowlist gates the program NAME;
                # arguments travel separately (no shell, injection-proof). A
                # whole command line passed as the name failed the gate --
                # measured, sweep 64 -- so the line is split here.
                import shlex
                parts = shlex.split(a)
                entry["result"] = self.mind.run_command(
                    parts[0], args=parts[1:] or None)
            elif kind == "guidance":
                if self.llm is None:
                    entry.update(status="blocked",
                                 detail="guidance step but no llm attached")
                    return entry
                self.llm_calls += 1
                entry["result"] = {"guidance": str(self.llm(
                    "GUIDANCE REQUEST in SOP step %r: %s\nState keys: %s"
                    % (step["name"], a, sorted(state))))}
        except Exception as e:
            entry.update(status="failed",
                         detail="%s: %s" % (type(e).__name__, str(e)[:160]))
            return entry
        if step["verify"] is not None:
            ok, why = safe_verify(step["verify"], entry.get("result"), state)
            if why:
                entry.update(status="blocked", detail=why)
            elif not ok:
                entry.update(status="failed",
                             detail="verify false: %s" % step["verify"][:120])
        entry["elapsed"] = round(time.time() - t0, 4)
        return entry

    def _escalate(self, step, entry, state):
        # The bot_runner escalation, live: ONE model call carrying the failed
        # step's context; the model answers with a verdict whose FIRST WORD is
        # continue or abort. Anything else is treated as abort -- an
        # ambiguous order from the escalation path is not followed either.
        self.llm_calls += 1
        return str(self.llm(
            "SOP step %r failed (%s). Reply 'continue' to proceed to the next "
            "step or 'abort' to stop. Detail: %s\nState keys: %s"
            % (step["name"], entry["status"], entry["detail"], sorted(state))))


def _selftest():
    import numpy as np

    class FakeMind:
        """Planted faculties with exact known behaviour -- the SOP runner is
        the instrument under test, so the mind must not be one."""
        def __init__(self):
            self.calls = []
        def greet(self, who="x"):
            self.calls.append(("greet", who))
            return {"msg": "hello %s" % who}
        def flaky(self):
            # fails twice, then works: the retry counter's planted truth
            self.calls.append(("flaky",))
            return {"n": len([c for c in self.calls if c[0] == "flaky"])}
        def sandbox_run(self, code, lang="python", **k):
            self.calls.append(("sandbox", lang))
            return {"ok": True, "stdout": "42\n", "returncode": 0}
        def run_command(self, name, args=None):
            # mirrors the REAL contract: name gated, args separate
            self.calls.append(("shell", name, tuple(args or ())))
            return {"cmd": name, "args": list(args or ())}

    # parse refusals: unknown directive, action-less step, private invoke,
    # double action -- each names its line; NOTHING runs
    bad = parse_sop("## step: a\nfrobnicate: x\n## step: b\n"
                    "invoke: _secret {}\n## step: c\npython: 1\nshell: ls\n")
    # 5, not 3: the refused directives ALSO leave steps a and b action-less --
    # every consequence of a bad order is named, not just its first cause
    assert not bad["ok"] and len(bad["errors"]) == 5, bad["errors"]
    r = SOPRunner(FakeMind()).run("## step: a\nfrobnicate: x\n")
    assert r["refused"] and r["log"] == [] and r["llm_calls"] == 0

    # the happy path: invoke + store + verify-over-state + code + shell,
    # and THE COUNT THAT IS THE POINT: zero model calls
    sop = ("# planted run\n"
           "## step: greet\ninvoke: greet {\"who\": \"moose\"}\nstore: g\n"
           "verify: result[\"msg\"] == \"hello moose\"\n"
           "## step: code\npython: print(6*7)\n"
           "verify: \"42\" in result[\"stdout\"] and state[\"g\"][\"msg\"] == \"hello moose\"\n"
           "## step: sh\nshell: echo hi\n")
    fm = FakeMind()
    out = SOPRunner(fm, llm=None).run(sop)
    assert out["ok"] and out["llm_calls"] == 0, out
    assert [e["status"] for e in out["log"]] == ["fired"] * 3
    assert out["state"]["g"]["msg"] == "hello moose"

    # determinism: identical logs modulo elapsed
    out2 = SOPRunner(FakeMind(), llm=None).run(sop)
    strip = lambda lg: [{k: v for k, v in e.items() if k != "elapsed"} for e in lg]
    assert strip(out["log"]) == strip(out2["log"])

    # verify failure -> abort by default; later steps SKIPPED not silently run
    out = SOPRunner(FakeMind()).run(
        "## step: a\ninvoke: greet {}\nverify: result[\"msg\"] == \"nope\"\n"
        "## step: b\nshell: echo never\n")
    assert not out["ok"] and out["log"][0]["status"] == "failed"
    assert out["log"][1]["status"] == "skipped"

    # on_fail continue: leOS graceful degradation, honestly logged
    out = SOPRunner(FakeMind()).run(
        "## step: a\ninvoke: greet {}\nverify: result[\"msg\"] == \"nope\"\n"
        "on_fail: continue\n## step: b\nshell: echo still\n")
    assert out["ok"] and [e["status"] for e in out["log"]] == ["failed", "fired"]

    # retry: flaky fires on the 3rd attempt; spent counts every attempt
    out = SOPRunner(FakeMind()).run(
        "## step: f\ninvoke: flaky {}\nverify: result[\"n\"] >= 3\non_fail: retry 2\n")
    assert out["ok"] and out["log"][0]["attempt"] == 2 and out["steps_spent"] == 3

    # escalate: exactly ONE model call; 'continue' proceeds, anything else aborts
    calls = []
    def llm_continue(p, **k):
        calls.append(p)
        return "continue -- acceptable"
    out = SOPRunner(FakeMind(), llm=llm_continue).run(
        "## step: a\ninvoke: greet {}\nverify: result[\"msg\"] == \"nope\"\n"
        "on_fail: escalate\n## step: b\nshell: echo on\n")
    assert out["ok"] and len(calls) == 1 and out["llm_calls"] == 1
    out = SOPRunner(FakeMind(), llm=lambda p, **k: "hmm, unclear").run(
        "## step: a\ninvoke: greet {}\nverify: result[\"msg\"] == \"nope\"\n"
        "on_fail: escalate\n## step: b\nshell: echo off\n")
    assert not out["ok"] and out["log"][1]["status"] == "skipped"

    # guidance: the ONLY happy-path model call; blocked without an llm
    out = SOPRunner(FakeMind(), llm=lambda p, **k: "looks right").run(
        "## step: g\nguidance: sane?\n")
    assert out["ok"] and out["llm_calls"] == 1
    out = SOPRunner(FakeMind(), llm=None).run("## step: g\nguidance: sane?\n")
    assert not out["ok"] and out["log"][0]["status"] == "blocked"

    # the verify walker's kept negatives: dunder crawl, unknown names, and
    # arbitrary calls are REFUSED (blocked), not evaluated-and-false
    for expr in ("result.__class__", "__import__('os')", "open('x')",
                 "result['a'].__dict__"):
        ok, why = safe_verify(expr, {"a": 1}, {})
        assert not ok and "refused" in why, (expr, why)
    ok, why = safe_verify("state[\"k\"] + 1 == 2 and len(result) >= 0", [], {"k": 1})
    assert ok and why == ""
    # a refusal is distinguishable from a false check
    ok, why = safe_verify("result == 5", 4, {})
    assert not ok and why == ""

    # step budget: retries spend it; exhaustion is a NAMED block
    out = SOPRunner(FakeMind(), max_steps=2).run(
        "## step: f\ninvoke: flaky {}\nverify: result[\"n\"] >= 99\non_fail: retry 5\n")
    assert not out["ok"] and "budget" in out["log"][0]["detail"]
    print("holographic_soprunner selftest OK")
    return True


if __name__ == "__main__":
    _selftest()

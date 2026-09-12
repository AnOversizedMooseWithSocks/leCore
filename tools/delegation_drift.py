"""delegation_drift -- catch a FACULTY whose signature has drifted from the function it delegates to.

WHY THIS TOOL EXISTS
--------------------
The engine's rule is that a UnifiedMind faculty DELEGATES and never reimplements, and the convention
is that its docstring ends with "See holographic_<module>.<function>". That convention is machine-
readable, and nothing was reading it.

The bug it catches is specific and quiet: a parameter is added to the module function, the faculty is
not updated, and the capability is now REACHABLE BUT CRIPPLED -- the feature exists, the docs and the
catalog describe it, and calling it through the mind raises TypeError. Every existing audit passes,
because the module has a docstring (reachability), the catalog example still runs (skill_lint), and
nothing is unwired (wiring_report). The failure is in the SEAM, and no tool was looking at seams.

It has now happened twice in one arc:
    scatter_mesh        gained `holographic`/`dim`/`cell_size` in the module; the faculty did not
    creature_material   gained `iridescence`/`film_nm`/`n_film`; the faculty did not
Both were caught by an integration test happening to call the new argument -- i.e. by luck, late.

WHAT IT REPORTS
    MISSING   the delegate accepts a parameter the faculty does not expose and cannot forward.
              This is the real bug: the argument is unreachable through the mind.
    EXTRA     the faculty accepts something the delegate does not. Usually deliberate (a faculty that
              composes two calls), so it is reported separately and does not gate.

WHAT IT DELIBERATELY DOES NOT FLAG
  * A faculty with **kwargs: it can forward anything, so nothing is unreachable.
  * A parameter the faculty BINDS AT ITS OWN CALL SITE -- `mind=self`, `seed=self.seed`,
    `aspect=float(width) / float(height)`. That is not unreachable, it is DECIDED, and since sweep 131
    this tool reads the wrapper's AST to tell the two apart instead of comparing signatures alone.
    Those are listed under SUPPLIED with their binding, so the decision is visible without inflating
    the score. MEASURED: 45 of the original 99 findings were this, or the two classes below it.
  * A parameter whose name starts with an underscore: recursion and plumbing arguments
    (holo_octree's `_depth`, logic_prove's `_return_table`) are never public API, and a faculty that
    exposed one would be the bug rather than the fix.
  * A faculty that intentionally exposes a SUBSET for a reason no call-site read can see -- those live
    in BUDGET below, named, with a reason. A budget entry is a decision on the record; silence would be
    a decision nobody can see.

KEPT NEGATIVE (loud)
    This checks NAMES, not semantics. A faculty that forwards `seed` to the delegate's `rng_seed`
    reads as drift, and a faculty that forwards a parameter to the WRONG delegate parameter reads as
    clean. It is a seam-shaped net, not a proof of correctness.
"""

import argparse
import ast
import importlib
import json
import inspect
import re
import sys
import textwrap

#: Faculties that intentionally expose fewer parameters than their delegate, with the REASON. Each
#: entry is a decision on the record; adding one should feel like a small commitment, not a mute button.
BUDGET = {
    # "faculty_name": "why the narrower signature is deliberate",
    #
    # A faculty that SUPPLIES the missing parameter itself is not drift. These three delegate to
    # functions taking `mind=` for the FABRIK reach, and the faculty passes `self` -- exposing it
    # would let a caller hand a DIFFERENT mind to a method reached through this one, which is worse
    # than not exposing it. Recorded here rather than left silent, so the exemption is a decision on
    # the record instead of a hole in the audit.
    "gait_pose": "supplies mind=self; exposing it would let a caller pass a foreign mind",
    "gait_frames": "supplies mind=self; exposing it would let a caller pass a foreign mind",
    "gait_report": "supplies mind=self; exposing it would let a caller pass a foreign mind",
}

_SEE = re.compile(r"See\s+(holographic_[A-Za-z0-9_]+)\.([A-Za-z_][A-Za-z0-9_]*)")

#: The recorded drift count. `--gate` enforces MAY ONLY SHRINK against this, not zero.
#: WHY A BUDGET AND NOT A ZERO GATE, which is what --gate meant before sweep 131: this file's own
#: docstring says "a tool that fails the build on day one gets disabled rather than fixed", and that
#: is exactly what a zero gate would have been while 99 findings stood. A shrink-only budget is the
#: pattern doc_coverage already uses in this repo and the one that survives contact with a backlog:
#: it cannot be satisfied by ignoring the tool, and it cannot block a build on somebody else's debt.
BUDGET_FILE = __file__.replace("delegation_drift.py", "delegation_drift_budget.json")


def _budget():
    """The recorded ceiling, or None when no budget file exists -- in which case --gate keeps its old,
    stricter meaning (fail on ANY finding), so nothing silently loosens."""
    import json as _json
    import os as _os
    if not _os.path.exists(BUDGET_FILE):
        return None
    try:
        return _json.load(open(BUDGET_FILE))["missing_budget"]
    except Exception:
        return None


def _find_module(short_name):
    """Resolve 'holographic_foo' to its real dotted path -- the engine keeps modules in family
    packages, and the docstring convention only records the leaf name."""
    for pkg in ("mesh_and_geometry", "materials_and_texture", "rendering", "simulation_and_physics",
                "sampling_and_signal", "agents_and_reasoning", "caching_and_storage", "io_and_interop",
                "scene_and_pipeline", "semantic_router", "misc", "unified"):
        try:
            return importlib.import_module("holographic.%s.%s" % (pkg, short_name))
        except Exception:
            continue
    try:
        return importlib.import_module(short_name)
    except Exception:
        return None


def _bound_at_call_site(fac, fn_name, tgt_params):
    """Which of the delegate's parameters does the faculty BIND when it calls it? -> {param: expression}.

    WHY THIS EXISTS, and it is the difference between an audit and a nag. This tool defines the bug as
    "the argument is UNREACHABLE through the mind". A parameter the wrapper binds itself is not
    unreachable -- it is DECIDED, which is a design choice with a visible reason, and a signature-only
    comparison cannot tell the two apart. MEASURED on the engine as this landed: of 107 (faculty,
    parameter) findings, 45 were bound at the call site -- `seed=self.seed` eighteen times and `mind=self`
    sixteen (the gait_* exemption already in BUDGET, times sixteen), plus computed values like
    `aspect=float(width) / float(height)`, where the faculty exposes a BETTER parameter than the one it
    hides. Reporting those as drift for a year is how a report-only tool stays report-only.

    Positional arguments count: a delegate's required first parameters are usually passed positionally
    under a different local name (`simulate(positions, ...)` for a delegate whose first parameter is
    `points`), and calling that "missing" is a false positive of a NAME-based check -- the kept negative
    at the top of this file, now partly paid off.

    CONSERVATIVE IN ONE DIRECTION ON PURPOSE. Every case it cannot read resolves to "not bound", so an
    unreadable wrapper is REPORTED as drift rather than silently excused. A false positive is visible and
    can be budgeted in one line; a false negative is the exact silence this tool exists to break, and
    both of its founding bugs (scatter_mesh, creature_material) were found late by luck."""
    try:
        src = textwrap.dedent(inspect.getsource(fac))
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError, IndentationError):
        return {}
    names = _delegate_names(tree, fn_name)
    bound = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        called = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
        if called not in names:
            continue
        for i, arg in enumerate(node.args):
            if isinstance(arg, ast.Starred):
                # *args could cover anything from here on; claiming otherwise would invent coverage.
                return {p: "*args" for p in tgt_params}
            if i < len(tgt_params):
                bound[tgt_params[i]] = _expr(arg)
        for kw in node.keywords:
            if kw.arg is None:
                # `**kw` FORWARDS AN UNKNOWN SET -- and the first draft assumed it covered everything,
                # which HID two real drifts (creature_tree's tip_inset and mount_flare travel nowhere:
                # `kw` is a dict literal built two lines above and neither key is in it). So read the
                # dict when it is readable, and when it is not, bind NOTHING.
                keys = _local_dict_keys(tree, kw.value)
                if keys is None:
                    continue
                for k in keys:
                    bound[k] = "**%s" % _expr(kw.value)
                continue
            bound[kw.arg] = _expr(kw.value)
    return bound


def _delegate_names(tree, fn_name):
    """Every local name that refers to the delegate -- `from mod import fn as _d` is idiomatic in these
    wrappers and matching only `fn_name` missed it, reading `_d(self, ...)` as a call to something else
    and reporting three faculties that plainly supply `mind=self` as drift."""
    names = {fn_name}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == fn_name and a.asname:
                    names.add(a.asname)
    return names


def _local_dict_keys(tree, node):
    """The keys of a dict this function builds, for a `**kw` forward -> a set, or None if not readable.

    Only `kw = dict(a=..., b=...)` and `kw = {"a": ...}` count. Any other assignment to that name, or any
    mutation of it, gives up and returns None -- which reports the parameter rather than excusing it."""
    if not isinstance(node, ast.Name):
        return None
    target, keys, seen = node.id, set(), 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == target:
            return None                                   # kw.update(...) / kw.pop(...): unreadable
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == target:
            return None                                   # kw["x"] = ...
        if not (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == target):
            continue
        seen += 1
        v = n.value
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "dict":
            if any(k.arg is None for k in v.keywords):
                return None
            keys |= {k.arg for k in v.keywords}
        elif isinstance(v, ast.Dict):
            for k in v.keys:
                if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                    return None
                keys.add(k.value)
        else:
            return None
    return keys if seen else None


def _expr(node):
    """The source of one argument expression, for the report -- the READER decides whether
    `seed=self.seed` is a deliberate narrowing or a bug, and cannot without seeing the binding."""
    try:
        return " ".join(ast.unparse(node).split())[:60]
    except Exception:
        return "?"


def _params(fn):
    """Named parameters of a callable, plus whether it takes **kwargs. `self` is dropped so a method
    and a plain function compare on equal terms."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None, False
    names, has_kw = [], False
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            has_kw = True
        elif p.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        else:
            names.append(p.name)
    return names, has_kw


def audit(verbose=False, min_overlap=0.8):
    """Compare every faculty that names a delegate against that delegate's signature."""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # sweep 124: runnable from anywhere
    import lecore
    mind = lecore.UnifiedMind(dim=32, seed=0)

    missing, extra, unresolved, supplied, checked = [], [], [], [], 0
    for name in sorted(dir(mind)):
        if name.startswith("_"):
            continue
        try:
            fac = getattr(type(mind), name, None) or getattr(mind, name)
        except Exception:
            continue
        if not callable(fac):
            continue
        doc = inspect.getdoc(fac) or ""
        m = _SEE.search(doc)
        if not m:
            continue                                          # no declared delegate: nothing to compare
        mod_name, fn_name = m.group(1), m.group(2)
        mod = _find_module(mod_name)
        if mod is None:
            unresolved.append((name, "%s.%s" % (mod_name, fn_name), "module not importable"))
            continue
        target = getattr(mod, fn_name, None)
        if target is None:                                    # often a Class.method reference
            for attr in dir(mod):
                obj = getattr(mod, attr, None)
                if inspect.isclass(obj) and hasattr(obj, fn_name):
                    target = getattr(obj, fn_name)
                    break
        if target is None or not callable(target):
            unresolved.append((name, "%s.%s" % (mod_name, fn_name), "function not found"))
            continue

        fac_p, fac_kw = _params(fac)
        tgt_p, _tgt_kw = _params(target)
        if fac_p is None or tgt_p is None:
            continue
        checked += 1
        if fac_kw:
            continue                                          # **kwargs forwards everything
        # A LEADING UNDERSCORE IS NEVER PUBLIC API. holo_octree's `_depth` and logic_prove's
        # `_return_table` are recursion/plumbing arguments; a faculty that exposed them would be the
        # bug, not the fix. Dropped before anything else so they never reach a budget line either.
        tgt_p = [p for p in tgt_p if not p.startswith("_")]
        miss = [p for p in tgt_p if p not in fac_p]
        if miss:
            # Bound at the call site == reachable-by-decision, not unreachable. See _bound_at_call_site.
            at_call = _bound_at_call_site(fac, fn_name, tgt_p)
            still = [p for p in miss if p not in at_call]
            for p in miss:
                if p in at_call:
                    supplied.append((name, "%s.%s" % (mod_name, fn_name), p, at_call[p]))
            miss = still
        # OVERLAP is what separates drift from a different calling convention. A faculty that already
        # forwards most of the delegate's parameters by name IS a 1:1 forwarder, so a small gap is
        # almost certainly a parameter added later and never plumbed through. A faculty that shares
        # few names is doing something else entirely (building a class, computing its arguments), and
        # flagging it would be noise. MEASURED across the engine: 238 raw findings fall to 42 at 0.8
        # overlap, and the ones that survive are single missing parameters on obvious forwarders.
        overlap = (len([p for p in tgt_p if p in fac_p]) / len(tgt_p)) if tgt_p else 1.0
        if miss and overlap >= min_overlap and name not in BUDGET:
            missing.append((name, "%s.%s" % (mod_name, fn_name), miss, round(overlap, 2)))
        ext = [p for p in fac_p if p not in tgt_p]
        if ext:
            extra.append((name, "%s.%s" % (mod_name, fn_name), ext))

    print("DELEGATION DRIFT over %d faculties that declare a delegate" % checked)
    print()
    print("  MISSING -- the delegate takes a parameter the faculty cannot forward "
          "(reachable but crippled): %d" % len(missing))
    for n, tg, ps, ov in missing:
        print("    %-32s -> %-42s [overlap %.2f] missing: %s" % (n, tg, ov, ", ".join(ps)))
    if verbose:
        print()
        print("  EXTRA -- faculty takes what the delegate does not (usually deliberate): %d" % len(extra))
        for n, t, ps in extra[:40]:
            print("    %-34s -> %-46s extra: %s" % (n, t, ", ".join(ps)))
        print()
        print("  UNRESOLVED delegate references (not gating): %d" % len(unresolved))
        for n, t, why in unresolved[:25]:
            print("    %-34s -> %-46s %s" % (n, t, why))
    print()
    print("  SUPPLIED -- the faculty BINDS the parameter itself, so nothing is unreachable: %d "
          "across %d faculties" % (len(supplied), len({s[0] for s in supplied})))
    if verbose:
        for n, tg, p, expr in supplied:
            print("    %-32s -> %-42s %s = %s" % (n, tg, p, expr))
    else:
        # The SHAPE of the supplied set is the finding, not the list: two conventions, not N accidents.
        pat = {}
        for n, tg, p, expr in supplied:
            pat.setdefault("%s = %s" % (p, expr if expr.startswith("self") else "<computed>"), []).append(n)
        for k in sorted(pat, key=lambda k: -len(pat[k]))[:8]:
            print("    %-38s x%-3d  e.g. %s" % (k, len(pat[k]), ", ".join(sorted(pat[k])[:3])))
        print("    (--verbose lists every one with its binding)")
    if BUDGET:
        print()
        print("  BUDGETED (deliberately narrower, and NOT detectable from the call site): %d" % len(BUDGET))
        for k, why in sorted(BUDGET.items()):
            print("    %-34s %s" % (k, why))
    print()
    print("TOTAL: %d likely-drifted faculty signature(s) at overlap >= %.2f." % (len(missing), min_overlap))
    print("(--gate enforces MAY ONLY SHRINK against tools/delegation_drift_budget.json, not zero: a "
          "tool that fails the build on somebody else's backlog gets disabled rather than fixed. "
          "Fix a drift and --rebase, WITH A REASON.)")
    return {"missing": missing, "extra": extra, "unresolved": unresolved,
            "supplied": supplied, "checked": checked, "budgeted": sorted(BUDGET)}


def as_records(result):
    """The audit result with every row NAMED -- {faculty, delegate, missing, overlap} instead of a bare
    4-tuple.

    WHY: a list of tuples is precisely the shape holographic_shapeprobe was built after six instrument
    errors to warn about -- `hits` was a list, and the bug was what the list held. The print path can
    unpack positionally because it is three lines away from the definition; an agent reading this over
    /invoke is not, and a caller that reads row[2] as the overlap gets a wrong answer that looks right."""
    return {
        "checked": result["checked"],
        "missing": [{"faculty": n, "delegate": t, "missing": list(ps), "overlap": ov}
                    for n, t, ps, ov in result["missing"]],
        "supplied": [{"faculty": n, "delegate": t, "parameter": p, "bound_to": e}
                     for n, t, p, e in result["supplied"]],
        "extra": [{"faculty": n, "delegate": t, "extra": list(ps)} for n, t, ps in result["extra"]],
        "unresolved": [{"faculty": n, "delegate": t, "why": w} for n, t, w in result["unresolved"]],
        "budgeted": [{"faculty": k, "why": BUDGET[k]} for k in result["budgeted"]],
        "total_missing": len(result["missing"]),
    }


def audit_quiet(min_overlap=0.8):
    """`audit` with its printed report suppressed, results as records -- the form a faculty returns.

    The report is written for a terminal; a faculty that printed 200 lines into a service log every time
    an agent asked for the drift count would be a different kind of unreachable."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        result = audit(min_overlap=min_overlap)
    return as_records(result)


def _selftest():
    """The tool must CATCH a synthetic drift and PASS a matching pair -- an audit that cannot fail is
    decoration, which is the same failure this tool exists to catch in others."""
    def delegate(a, b, c=1, d=2):
        """target"""
    def good(a, b, c=1, d=2):
        """See holographic_x.delegate."""
    def drifted(a, b):
        """See holographic_x.delegate."""
    def kwargged(a, **kw):
        """See holographic_x.delegate."""

    gp, gk = _params(good); dp, dk = _params(drifted); kp, kk = _params(kwargged)
    tp, _ = _params(delegate)
    assert [p for p in tp if p not in gp] == [], "a matching signature must show no drift"
    assert [p for p in tp if p not in dp] == ["c", "d"], "drift must be detected by name"
    assert kk is True and gk is False, "**kwargs must be recognised as forwarding everything"
    assert _SEE.search(good.__doc__).group(2) == "delegate", "the See-convention must parse"
    # OVERLAP: a near-identical forwarder scores high (real drift); a totally different convention
    # scores low and must be filtered out, or the audit drowns in noise and gets ignored.
    def different(x, y, z):
        """See holographic_x.delegate."""
    dp2, _ = _params(different)
    assert len([p for p in tp if p in dp2]) / len(tp) < 0.5, "a different convention must score LOW"
    assert len([p for p in tp if p in dp]) / len(tp) >= 0.5, "a near-forwarder must score HIGH"
    # ---- THE CALL-SITE READER (sweep 131). Each of these was a REAL false positive in the 99, and each
    # is pinned with the exact parameter set so a regression names itself rather than moving a total.
    def bound_kw(a, b):
        """See holographic_x.delegate."""
        return delegate(a, b, c=self_c, d=7)
    assert set(_bound_at_call_site(bound_kw, "delegate", tp)) == {"a", "b", "c", "d"}, \
        "keyword-bound parameters are DECIDED, not unreachable"

    def bound_positionally(a, b):
        """See holographic_x.delegate."""
        return delegate(a, b, 3, 4)
    got = _bound_at_call_site(bound_positionally, "delegate", tp)
    assert set(got) == {"a", "b", "c", "d"} and got["c"] == "3", got
    # ^ the RENAME class: a delegate's parameter passed positionally under another local name reads as
    #   "missing" to a name-based check. Three of the 99 were exactly this (fem_simulate's points).

    # The alias resolver is exercised on PARSED SOURCE rather than a real function, because a genuine
    # `from holographic_x import ...` in this file -- even inside a never-called body -- is an import that
    # resolves to nothing, and tools/audit_imports.py is right to call that broken. An audit that breaks
    # another audit to test itself has not understood the assignment.
    aliased = ast.parse("from holographic_x import delegate as _d\n_d(a, b, c=1, d=2)\n")
    assert _delegate_names(aliased, "delegate") == {"delegate", "_d"}, \
        "an aliased import IS the delegate -- missing this reported 3 faculties that plainly supply mind=self"

    def via_kw_dict(a, b):
        """See holographic_x.delegate."""
        kw = dict(c=1)
        return delegate(a, b, **kw)
    got = _bound_at_call_site(via_kw_dict, "delegate", tp)
    assert "c" in got and "d" not in got, \
        "a readable **kw dict binds ONLY its own keys -- assuming it covered everything hid creature_tree's " \
        "tip_inset and mount_flare, a silent no-op that is worse than the drift it masked"

    def via_opaque_kw(a, b, extra):
        """See holographic_x.delegate."""
        kw = extra
        return delegate(a, b, **kw)
    assert "c" not in _bound_at_call_site(via_opaque_kw, "delegate", tp), \
        "an UNREADABLE **kw must bind nothing: report a false positive, never hide a real one"

    assert _local_dict_keys(ast.parse("kw = dict(x=1, y=2)"), ast.Name(id="kw")) == {"x", "y"}
    assert _local_dict_keys(ast.parse("kw = other"), ast.Name(id="kw")) is None

    print("delegation_drift selftest OK: detects missing params, ignores **kwargs, parses the "
          "See-convention, separates drift from a different calling convention by overlap, and reads the "
          "CALL SITE (keyword, positional, aliased import, readable **kw) so a bound parameter is never "
          "reported as unreachable -- the 45 of 99 that were never drift")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verbose", action="store_true", help="also list EXTRA and UNRESOLVED")
    ap.add_argument("--min-overlap", type=float, default=0.8,
                    help="only flag forwarders sharing at least this fraction of the delegate's params")
    ap.add_argument("--gate", action="store_true",
                    help="CI gate: exit non-zero if the drift count GREW past the recorded budget "
                         "(tools/delegation_drift_budget.json); with no budget file, fail on any finding")
    ap.add_argument("--rebase", action="store_true",
                    help="record today's count as the budget -- lower it when drifts are fixed, and "
                         "say WHY in the file, because a budget without a reason is a mute button")
    ap.add_argument("--reason", default=None,
                    help="REQUIRED with --rebase: why the budget is moving. The report already tells "
                         "you to rebase 'WITH A REASON' and there was no way to give one, so every "
                         "rebase wrote the placeholder -- the mute button this tool warns against, "
                         "installed by the tool itself.")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest(); sys.exit(0)
    r = audit(verbose=a.verbose, min_overlap=a.min_overlap)
    n, budget = len(r["missing"]), _budget()
    if a.rebase:
        if not a.reason:
            print("REFUSED: --rebase needs --reason. A budget without a reason is a mute button, "
                  "and the next reader cannot tell a cleared backlog from a silenced one.")
            sys.exit(2)
        json.dump({"missing_budget": n, "why": a.reason},
                  open(BUDGET_FILE, "w"), indent=1)
        print("budget recorded:", n)
        sys.exit(0)
    if a.gate:
        if budget is None:
            sys.exit(1 if r["missing"] else 0)
        if n > budget:
            print("FAIL: delegation drift grew %d -> %d. Plumb the parameter through, or record the "
                  "narrowing in BUDGET with a reason." % (budget, n))
            sys.exit(1)
        print("OK: within budget (%d)" % budget)
    sys.exit(0)

"""holographic_codehealth.py -- complexity crossed with EXPOSURE and EXERCISE, which is the only form in
which complexity is worth measuring.

WHY NOT JUST A COMPLEXITY NUMBER
--------------------------------
The previous session deliberately refused to register a "cyclomatic complexity" capability, on the grounds
that we did not have one and a known false friend (the query resolved to the renderer's `scene_cost`) beats
a claim we cannot back. Building it needed a reason beyond "radon prints a number", and measurement supplied
one by REFUTING the plan it was supposed to serve.

The plan was: refactor the highest-complexity functions. Crossing radon's scores against whether any test so
much as MENTIONS the function killed that plan outright:

    the top-CC functions are ALL exercised     parse_description 65, mesh_parts 57, rebake_texture 54,
                                               query.run 48, rasterize_mesh 42, extract_quads 42
    the risk is somewhere else entirely        761 public functions no test mentions at all,
                                               11 of them at CC >= 20, 44 at CC >= 10

So the big scary numbers are the SAFE ones -- they are big precisely because they are load-bearing, and
load-bearing code got tests. Raw complexity ranks the wrong thing. What this module reports instead is the
CROSS PRODUCT:

    RISK = complexity  x  exposure (is it advertised?)  x  exercise (does any test touch it?)

and the worst cell in that matrix is not the biggest function. It is `catmull_clark`: CC 46, REGISTERED IN
THE CATALOG as an advertised capability, and not mentioned by a single test. An advertised, complex,
unguarded surface is worth more attention than a complex one that fifty tests sit on.

RELATIONSHIP TO radon (third-party, and it gets the credit)
-----------------------------------------------------------
radon found this. This module reimplements only the McCabe count, in stdlib `ast`, because core is
NumPy/Flask/stdlib/hashlib and an audit CI cannot run without a third-party wheel is an audit that quietly
stops running. The count is NOT expected to match radon exactly -- radon makes its own defensible choices
about `with`, `assert` and boolean operators -- so `_selftest` pins RANK AGREEMENT against radon when radon
is installed, and simply skips that check when it is not. Agreeing on the ORDER is the property we actually
use; agreeing on the integer is not.

THE MENTION SCAN WAS REFUTED BY MEASUREMENT -- READ THIS BEFORE TRUSTING THE ATTENTION LIST
-------------------------------------------------------------------------------------------
The first version used "does any test NAME this function?" as the exercise axis. Both of its headline
findings turned out to be FALSE POSITIVES when checked against coverage.py:

    reproject_uv       CC 67, reported unexercised -> 163 of 272 lines EXECUTE (via mind.mesh_reproject_uv)
    interpret_command  CC 27, reported unexercised ->  55 of  91 lines EXECUTE (via a tested caller)

Two for two. The first was fixable -- resolve one hop through the facade, since this engine delegates
faculty-to-module under a different name, and that alone removed 129 false positives. The second was not:
it is reached through an ordinary call chain, and no amount of name-scanning finds that. A NAME SCAN IS
NOT A REACHABILITY ANALYSIS, and in a codebase built on delegation it over-reports badly.

So `exercised` now prefers REAL DATA: pass `coverage_file=` (or leave the default and have a .coverage
database present) and the exercise axis becomes executed-line counts. Without one it falls back to the
mention scan AND SAYS SO, loudly, in the report and in the returned dict -- because the fallback's own
track record is two false headlines out of two.
  * DEMO AND SELFTEST FUNCTIONS ARE NOT RISK. `demo_text` at CC 22 is a demo; nobody's render depends on it.
    They are counted but tagged, so they stop crowding out the real findings.
  * NO REFACTORING ADVICE, and no threshold that calls a function "too complex". A long if/elif chain in a
    parser scores terribly and is often the clearest possible way to write it. The output ranks attention,
    it does not issue verdicts.
"""
import ast
import glob
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# A function at or above this, with no test mention, is what the report leads with. Not a verdict on the
# function -- a threshold on where a reader's attention is worth spending first.
ATTENTION_CC = 20

# Names that are complex on purpose and carry no production weight. Tagged, not hidden.
_DEMO = re.compile(r"^(demo_|example_|_selftest)")


def complexity(node):
    """McCabe cyclomatic complexity of one function node: 1 + the number of independent decision points.

    Counted: if / elif, for, while, except handlers, boolean operators (each extra operand is another path),
    ternaries, comprehension conditions, match cases, and `assert`. Not counted: `with`, which introduces no
    branch. These choices are conventional but not universal, which is why _selftest pins agreement with
    radon on RANK rather than on the integer."""
    score = 1
    for n in ast.walk(node):
        if isinstance(n, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler,
                          ast.IfExp, ast.Assert)):
            score += 1
        elif isinstance(n, ast.BoolOp):
            score += len(n.values) - 1
        elif isinstance(n, comprehension_types):
            score += 1 + len(n.ifs)
        elif hasattr(ast, "match_case") and isinstance(n, ast.match_case):
            score += 1
    return score


comprehension_types = (ast.comprehension,)


def _iter_functions(path, trees=None):
    """Every named function in a file, top-level or one class deep, with its class prefix for reporting."""
    from holographic.io_and_interop.holographic_orphanaudit import _tree as _shared_tree
    tree = _shared_tree(path, trees)           # index resolved ONCE by the caller -- see _tree's docstring
    if tree is None:
        return
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            yield node.name, node, ""
        elif isinstance(node, ast.ClassDef):
            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    yield n.name, n, node.name + "."


def delegation_map():
    """faculty method name -> the module-level function names it calls.

    WHY THIS IS LOAD-BEARING. This engine's architecture is FACULTY-DELEGATES-TO-MODULE: `mind.mesh_reproject_uv`
    imports and calls `holographic_meshtools.reproject_uv`. A test naturally names the FACULTY, so a name scan
    for the module function finds nothing and reports it unexercised.

    That is not hypothetical -- it was this audit's own number-one finding. reproject_uv (CC 67) topped the
    attention list as "unmentioned", and coverage.py then showed 163 of its 272 lines executing under two
    tests in test_cad_backlog.py that call it through mind.mesh_reproject_uv. The single hairiest "unguarded"
    function in the engine was well guarded, under another name.

    So the mention scan resolves one hop through the facade before concluding anything. Import aliases are
    followed by their REAL name (`from x import foo as _f` registers `foo`, not `_f`), because the alias is
    exactly what hides the delegation."""
    try:
        from holographic.misc.holographic_unified import unified_sources
        paths = unified_sources()
    except Exception:
        return {}
    out = {}
    for p in paths:
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                tree = ast.parse(f.read(), filename=p)
        except (OSError, SyntaxError):
            continue
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
                called = set()
                for n in ast.walk(fn):
                    if isinstance(n, ast.Call):
                        f = n.func
                        if isinstance(f, ast.Name):
                            called.add(f.id)
                        elif isinstance(f, ast.Attribute):
                            called.add(f.attr)
                    elif isinstance(n, ast.ImportFrom):
                        for a in n.names:
                            called.add(a.name)          # the REAL name; the alias is what hid it
                            if a.asname:
                                called.add(a.asname)
                out.setdefault(fn.name, set()).update(called)
    return out


def test_mentions():
    """The set of identifiers named anywhere under tests/. A cheap upper bound on what is exercised."""
    names = set()
    for f in glob.glob(os.path.join(REPO, "tests", "**", "*.py"), recursive=True):
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                src = fh.read()
        except OSError:
            continue
        names.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", src))
    return names


def coverage_hits(coverage_file=None):
    """{module_path: set of executed line numbers} from a coverage.py database, or ({}, note) if there is none.

    Optional by design: coverage.py is a dev dependency, not a core one, and this module must run in CI
    without it. When a database IS present the exercise axis stops being a guess."""
    path = coverage_file or os.path.join(REPO, ".coverage")
    if not os.path.exists(path):
        return {}, ("MENTION SCAN (no coverage database at %s) -- over-reports; see the module docstring"
                    % os.path.relpath(path, REPO))
    try:
        import coverage as _cov
    except ImportError:
        return {}, "MENTION SCAN (coverage.py not installed) -- over-reports; see the module docstring"
    try:
        c = _cov.Coverage(data_file=path)
        c.load()
        data = c.get_data()
        hits = {f: set(data.lines(f) or ()) for f in data.measured_files()}
        # SAY HOW BIG THE RUN WAS. A database from three tests marks almost everything unexercised, which
        # looks exactly like a catastrophic finding instead of what it is -- a partial run. The file count
        # is the reader's only cue, so it goes in the evidence string rather than being left to be guessed.
        note = "COVERAGE (%s, %d file(s) measured)" % (os.path.basename(path), len(hits))
        if len(hits) < 200:
            note += " -- PARTIAL RUN: unexercised counts below are inflated, not a finding"
        return hits, note
    except Exception as exc:
        return {}, "MENTION SCAN (coverage database unreadable: %s)" % type(exc).__name__


def health_report(limit=20, attention_cc=ATTENTION_CC, coverage_file=None):
    """Rank the engine's functions by RISK = complexity x exposure x exercise, not by complexity.

    Returns {totals, attention, most_complex, buckets}. `attention` is the list worth reading: functions at
    or above `attention_cc` that no test mentions, tagged with how they are exposed (a catalogued or
    faculty-exposed function that nothing exercises outranks a purely internal one at the same score)."""
    try:
        from holographic.io_and_interop.holographic_orphanaudit import audit as _reach
        buckets = {}
        for kind, rows in _reach().items():
            for n, _p, _l in rows:
                buckets[n] = kind
    except Exception as exc:                      # loud, never silent -- an empty oracle looks like a finding
        print("  WARNING: reachability oracle unavailable (%s); exposure column will read '?'"
              % type(exc).__name__, file=sys.stderr)
        buckets = {}

    try:
        from holographic.io_and_interop.holographic_srcindex import parsed_trees
        trees = parsed_trees()                # ONE resolve for the whole report; never per file
    except Exception:
        trees = None
    covered, cov_note = coverage_hits(coverage_file)
    mentioned = test_mentions()
    # RESOLVE ONE HOP THROUGH THE FACADE: a module function is exercised if a TESTED FACULTY delegates to it.
    # Without this the audit reports the engine's entire delegated surface as unexercised -- see delegation_map.
    delegated = set()
    for faculty, calls in delegation_map().items():
        if faculty in mentioned:
            delegated |= calls
    mentioned = mentioned | delegated
    rows = []
    for path in sorted(glob.glob(os.path.join(REPO, "holographic", "**", "*.py"), recursive=True)):
        for name, node, prefix in _iter_functions(path, trees):
            if name.startswith("__"):
                continue
            cc = complexity(node)
            # REAL COVERAGE WHEN WE HAVE IT: a function counts as exercised if any line inside it ran.
            # The mention scan is only consulted for files coverage never measured.
            hits = covered.get(path)
            if hits is not None:
                end = getattr(node, "end_lineno", node.lineno)
                exercised = any(node.lineno <= l <= end for l in hits)
            else:
                exercised = name in mentioned
            rows.append({"name": name, "qual": prefix + name, "cc": cc, "exercised": exercised,
                         "module": os.path.basename(path)[:-3],
                         "line": node.lineno,
                         "exposure": buckets.get(name, "internal"),
                         "mentioned": exercised,
                         "demo": bool(_DEMO.match(name))})
    rows.sort(key=lambda r: -r["cc"])
    attention = [r for r in rows
                 if r["cc"] >= attention_cc and not r["mentioned"] and not r["demo"]]
    # advertised surfaces first: a catalogued or faculty function nothing exercises is the worst cell
    rank = {"catalog": 0, "faculty": 0, "test_only": 1, "engine": 2, "internal": 3, "orphan": 3}
    attention.sort(key=lambda r: (rank.get(r["exposure"], 3), -r["cc"]))
    return {
        "evidence": cov_note,
        "totals": {"functions": len(rows),
                   "mean_cc": round(sum(r["cc"] for r in rows) / max(len(rows), 1), 2),
                   "unmentioned": sum(1 for r in rows if not r["mentioned"]),
                   "unmentioned_over_threshold": len(attention)},
        "attention": attention[:limit],
        "most_complex": rows[:limit],
    }


def main(argv):
    r = health_report(limit=25)
    t = r["totals"]
    print("CODE HEALTH  --  %d functions, mean CC %.2f" % (t["functions"], t["mean_cc"]))
    print("  exercise evidence: %s" % r["evidence"])
    print("  %d never mentioned by any test; %d of those at CC >= %d"
          % (t["unmentioned"], t["unmentioned_over_threshold"], ATTENTION_CC))
    print("\nATTENTION -- complex, and no test names them (advertised surfaces first):")
    print("   CC  exposure    module                    function")
    for x in r["attention"]:
        print("  %3d  %-10s  %-24s  %s :%d" % (x["cc"], x["exposure"], x["module"][12:], x["qual"], x["line"]))
    if "--complex" in argv:
        print("\nMOST COMPLEX overall (mostly exercised -- that is the point):")
        for x in r["most_complex"]:
            print("  %3d  %-10s  %-24s  %s%s" % (x["cc"], x["exposure"], x["module"][12:], x["qual"],
                                                 "" if x["mentioned"] else "   <- UNMENTIONED"))
    return 0


def _selftest():
    """Pins the counting rules, and pins AGREEMENT WITH radon ON RANK rather than on the integer.

    The rank check is the honest one: two tools with different but defensible rules about `with`, `assert`
    and boolean operators will disagree on scores while agreeing on which functions are the hairy ones, and
    the ordering is the only thing this module's output actually uses."""
    src = ("def flat():\n    return 1\n\n"
           "def branchy(x):\n"
           "    if x > 0:\n        return 1\n"
           "    elif x < 0:\n        return 2\n"
           "    for i in range(3):\n        pass\n"
           "    return [i for i in range(3) if i]\n")
    tree = ast.parse(src)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert complexity(fns["flat"]) == 1, "a straight-line function must score 1"
    got = complexity(fns["branchy"])
    assert got >= 5, "two ifs, a for and a filtered comprehension should score >= 5, got %d" % got

    r = health_report(limit=5)
    assert r["totals"]["functions"] > 3000, "the scan missed the tree (%d)" % r["totals"]["functions"]
    assert all(a["cc"] >= ATTENTION_CC and not a["mentioned"] for a in r["attention"]), \
        "the attention list admitted a mentioned or low-complexity function"
    assert not any(a["demo"] for a in r["attention"]), "a demo leaked into the attention list"

    # cross-validation against radon, when it is available; skipped rather than faked when it is not
    try:
        import json as _json, subprocess as _sp
        out = _sp.run([sys.executable, "-m", "radon", "cc",
                       os.path.join(REPO, "holographic"), "-j"], capture_output=True, text=True).stdout
        ext = {}
        for path, blocks in _json.loads(out).items():
            if isinstance(blocks, dict):
                continue
            for b in blocks:
                ext[b["name"]] = max(ext.get(b["name"], 0), b["complexity"])
        mine = {}
        for path in glob.glob(os.path.join(REPO, "holographic", "**", "*.py"), recursive=True):
            for name, node, _p in _iter_functions(path):
                mine[name] = max(mine.get(name, 0), complexity(node))
        both = sorted(set(ext) & set(mine))
        assert len(both) > 1000, "too few shared names to validate against (%d)" % len(both)
        top_ext = {n for n in sorted(both, key=lambda n: -ext[n])[:100]}
        top_mine = {n for n in sorted(both, key=lambda n: -mine[n])[:100]}
        overlap = len(top_ext & top_mine) / 100.0
        assert overlap >= 0.7, "top-100 rank agreement with radon fell to %.2f" % overlap
        print("holographic_codehealth selftest OK -- %d functions, top-100 rank agreement with radon %.2f"
              % (r["totals"]["functions"], overlap))
        return
    except ImportError:
        pass
    print("holographic_codehealth selftest OK -- %d functions (radon absent; rank cross-check skipped)"
          % r["totals"]["functions"])


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        raise SystemExit(main(sys.argv[1:]))

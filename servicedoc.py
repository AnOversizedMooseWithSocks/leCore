#!/usr/bin/env python3
"""
servicedoc.py -- keep SERVICE.md's endpoint list honest against the code.

WHY THIS EXISTS (and how it differs from capdoc.py). capdoc.py fully REGENERATES CAPABILITIES.md, because that file is
nothing but a generated menu. SERVICE.md is different: most of it is hand-written prose worth keeping -- curl examples,
security notes, the job-lifecycle explanation. Two parts silently rot when the code changes: the ENDPOINT TABLE (add
or rename a route and the table is quietly wrong) and the LAUNCH section's CLI FLAGS (add or rename an argparse flag
and the launch instructions are wrong). So this does not regenerate the whole doc; it CHECKS that the documented
endpoints match the routes the service registers AND that every user-facing CLI flag is documented, naming anything
missing or stale. Run it in CI as a gate, the same spirit as tools/catalog_gaps.py.

It also has a `--print` helper that emits a fresh Markdown table (built from the live routes + each handler's one-line
docstring), so when the check fails you can paste an up-to-date table into the Endpoints section instead of hand-editing.

OLD-SCHOOL AND DEPENDENCY-FREE: standard library only. It imports holographic_service to read the route registry (a
cheap, side-effect-free construction -- no server is started).

    Check it (CI):   python servicedoc.py            # exits non-zero if the endpoint list drifted
    Fresh table:     python servicedoc.py --print     # prints a table to paste into SERVICE.md
"""
import ast
import os
import re
import sys


def routes():
    """Every registered endpoint as (method, path, description, body, returns), read from a live Service instance and
    its handler docstrings. Constructing Service() only builds the route table; it does not open a socket."""
    from holographic_service import Service
    svc = Service()
    out = []
    for (method, path), handler in sorted(svc._routes.items()):
        first = (handler.__doc__ or "").strip().split("\n")[0].strip()
        out.append((method, path) + _split_doc(first))
    return out


def _split_doc(doc):
    """Split a handler's one-line docstring 'What it does. Body: {...}. Returns: {...}.' into (desc, body, returns).
    Missing pieces come back as an em dash, so the table always has something to show."""
    body = returns = "—"
    desc = doc
    if "Returns:" in desc:
        desc, returns = desc.split("Returns:", 1)
        returns = returns.strip().rstrip(".") or "—"
    if "Body:" in desc:
        desc, body = desc.split("Body:", 1)
        body = body.strip().rstrip(".") or "—"
    return desc.strip().rstrip("."), body, returns


def doc_endpoints(path):
    """The set of (method, path) currently listed in SERVICE.md's endpoint table. A row looks like
    `| GET | ` + backtick + `/health` + backtick + ` | ... |`, so we pull the method and the back-ticked path."""
    text = open(path, encoding="utf-8").read()
    eps = set()
    for line in text.splitlines():
        m = re.match(r"\|\s*(GET|POST|PUT|DELETE|PATCH)\s*\|\s*`([^`]+)`\s*\|", line)
        if m:
            eps.add((m.group(1), m.group(2)))
    return eps


def check(path=None):
    """Compare the routes the service registers with the endpoints documented in SERVICE.md.
    Returns (missing, stale): `missing` are routes in the code but not the doc; `stale` are documented but no longer
    registered. Both empty means the list is in sync."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SERVICE.md")
    registered = {(m, p) for (m, p, *_rest) in routes()}
    documented = doc_endpoints(path)
    return sorted(registered - documented), sorted(documented - registered)


def generated_table():
    """A fresh Markdown endpoint table built from the live routes + handler docstrings -- paste-ready for SERVICE.md."""
    rows = ["| Method | Path | Body | Returns |", "|---|---|---|---|"]
    for method, path, desc, body, returns in routes():
        rows.append("| %s | `%s` | %s | %s |" % (method, path, body, returns if returns != "—" else desc))
    return "\n".join(rows)


# CLI flags the service accepts but that SERVICE.md need NOT document (internal / test-only). Everything else the
# argparse defines is user-facing and must appear in the doc, or the launch instructions are lying by omission.
_INTERNAL_FLAGS = {"--selftest"}

# --flags that appear in SERVICE.md but belong to the doc TOOLING, not the service (e.g. `servicedoc.py --print`).
# They must be excluded from the "stale" check, which otherwise reads them as service flags that no longer exist.
_DOC_ONLY_FLAGS = {"--print"}


def cli_flags(service_path=None):
    """The set of command-line flags the service's argparse defines, read from holographic_service.py with AST (no
    execution). We look for add_argument("--flag", ...) calls -- the same read-the-code approach as routes()."""
    if service_path is None:
        service_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holographic_service.py")
    tree = ast.parse(open(service_path, encoding="utf-8").read())
    flags = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                    flags.add(arg.value)
    return flags


def doc_flags(path):
    """The set of --flags mentioned anywhere in SERVICE.md (they appear inline in prose and code blocks, not a table)."""
    text = open(path, encoding="utf-8").read()
    return set(re.findall(r"--[a-z][a-z0-9-]+", text))


def check_flags(path=None, service_path=None):
    """Compare the argparse flags with those documented in SERVICE.md. Returns (stale, undocumented): `stale` are
    documented but no longer defined (the doc references a flag that's gone); `undocumented` are user-facing flags the
    argparse defines that SERVICE.md never mentions (internal/test flags in _INTERNAL_FLAGS are exempt)."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SERVICE.md")
    actual = cli_flags(service_path)
    documented = doc_flags(path)
    stale = sorted(f for f in documented if f not in actual and f not in _DOC_ONLY_FLAGS)
    undocumented = sorted(f for f in actual if f not in documented and f not in _INTERNAL_FLAGS)
    return stale, undocumented


def _selftest():
    r = routes()
    assert len(r) > 10 and all(len(t) == 5 for t in r)
    d, b, ret = _split_doc("Do a thing. Body: {x}. Returns: {ok}.")
    assert d == "Do a thing" and b == "{x}" and ret == "{ok}"
    assert "--host" in cli_flags() and "--port" in cli_flags()
    print("OK: servicedoc self-test passed (%d routes; %d cli flags; docstring split works)" % (len(r), len(cli_flags())))


if __name__ == "__main__":
    if "--print" in sys.argv:
        print(generated_table())
        sys.exit(0)
    if "--selftest" in sys.argv:
        _selftest()
        sys.exit(0)
    missing, stale = check()
    flag_stale, flag_undoc = check_flags()
    total = len(routes())
    problems = len(missing) + len(stale) + len(flag_stale) + len(flag_undoc)
    if not problems:
        print("SERVICE.md is in sync with the service (%d endpoints, %d cli flags)." % (total, len(cli_flags())))
        sys.exit(0)
    for m, p in missing:
        print("MISSING from SERVICE.md -- the service registers %s %s but it isn't documented." % (m, p))
    for m, p in stale:
        print("STALE in SERVICE.md -- documents %s %s but the service no longer registers it." % (m, p))
    for f in flag_undoc:
        print("MISSING from SERVICE.md -- the service accepts %s but the Launch docs never mention it." % f)
    for f in flag_stale:
        print("STALE in SERVICE.md -- documents %s but the service no longer accepts it." % f)
    if missing or stale:
        print("\nFix the Endpoints table in SERVICE.md. Tip: `python servicedoc.py --print` prints a fresh table.")
    if flag_stale or flag_undoc:
        print("Fix the Launch section's flags in SERVICE.md to match holographic_service.py's argparse.")
    sys.exit(problems)

#!/usr/bin/env python3
"""
servicedoc.py -- keep SERVICE.md's endpoint list honest against the code.

WHY THIS EXISTS (and how it differs from capdoc.py). capdoc.py fully REGENERATES CAPABILITIES.md, because that file is
nothing but a generated menu. SERVICE.md is different: most of it is hand-written prose worth keeping -- curl examples,
security notes, the job-lifecycle explanation. The one part that silently rots is the ENDPOINT TABLE: add or rename a
route in holographic_service.py and the table is quietly wrong. So this does not regenerate the whole doc; it CHECKS
that the set of endpoints in the table matches the routes the service actually registers, and names any that are
missing or stale. Run it in CI as a gate, the same spirit as tools/catalog_gaps.py.

It also has a `--print` helper that emits a fresh Markdown table (built from the live routes + each handler's one-line
docstring), so when the check fails you can paste an up-to-date table into the Endpoints section instead of hand-editing.

OLD-SCHOOL AND DEPENDENCY-FREE: standard library only. It imports holographic_service to read the route registry (a
cheap, side-effect-free construction -- no server is started).

    Check it (CI):   python servicedoc.py            # exits non-zero if the endpoint list drifted
    Fresh table:     python servicedoc.py --print     # prints a table to paste into SERVICE.md
"""
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


def _selftest():
    r = routes()
    assert len(r) > 10 and all(len(t) == 5 for t in r)
    d, b, ret = _split_doc("Do a thing. Body: {x}. Returns: {ok}.")
    assert d == "Do a thing" and b == "{x}" and ret == "{ok}"
    print("OK: servicedoc self-test passed (%d routes; docstring split works)" % len(r))


if __name__ == "__main__":
    if "--print" in sys.argv:
        print(generated_table())
        sys.exit(0)
    if "--selftest" in sys.argv:
        _selftest()
        sys.exit(0)
    missing, stale = check()
    total = len(routes())
    if not missing and not stale:
        print("SERVICE.md endpoints are in sync with the service (%d routes)." % total)
        sys.exit(0)
    for m, p in missing:
        print("MISSING from SERVICE.md -- the service registers %s %s but it isn't documented." % (m, p))
    for m, p in stale:
        print("STALE in SERVICE.md -- documents %s %s but the service no longer registers it." % (m, p))
    print("\nFix the Endpoints table in SERVICE.md. Tip: `python servicedoc.py --print` prints a fresh table to paste.")
    sys.exit(len(missing) + len(stale))

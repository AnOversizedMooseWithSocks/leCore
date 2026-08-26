"""holographic_deptrace.py -- what does this entry point ACTUALLY need at import time? (Poly Studio backlog D2)

WHY THIS MODULE EXISTS
----------------------
Embedding leCore elsewhere (a Pyodide bundle, a vendored subset) needs one number the repo could not produce:
the TRUE minimal module footprint of an entry point. The Poly Studio build reported that static import-tracing a
demo's slice ballooned from ~29 runtime modules to ~496, because modules name `numba`, `cupy`, `pyfftw`,
`matplotlib`, `sympy`, `nltk` inside `try/except` or inside functions that never run. A naive tracer follows every
`import` it can see, so it reports the union of everything the engine COULD ever touch -- useless for bundling.

The existing neighbours answer different questions, deliberately not extended here:
  * `holographic_backend.accelerator_report` -- what is INSTALLED in this environment and what it buys. A runtime
    probe of the machine, not an analysis of the code.
  * `tools/audit_imports.py` -- does an import RESOLVE (broken / flat)? A correctness gate, not a footprint.
This module reuses their machinery where it can (`select_tests.discover_modules` stays the single source of truth
for name->path; the flat-vs-packaged basename fallback mirrors audit_imports) and adds the missing axis: WHERE an
import sits, and therefore whether it runs at import time and whether its failure is survivable.

THE CLASSIFICATION (the whole point)
------------------------------------
  * HARD     -- module top level, not inside try. Runs on import; ImportError is fatal. A real dependency.
  * GUARDED  -- lexically inside a `try`. Runs on import, but failure is caught: an OPTIONAL accelerator.
  * DEFERRED -- inside a function/method body. Does not run at import AT ALL; only if that function is called.
So the import-time REQUIRED footprint is the transitive closure over HARD edges only. Everything else is reachable
but not required -- which is exactly the distinction a bundler, a dependency audit, and a subset build all need.

KEPT SIMPLIFICATION, stated rather than hidden: "inside a try" is judged LEXICALLY (any descendant of ast.Try),
so an import in a try body, an except handler, or an else/finally all count as GUARDED. A module-level import
wrapped in try purely for a fallback path (not for optionality) is therefore reported as GUARDED too. That is the
conservative direction -- it never calls a real dependency optional the other way round, and the alternative
(inferring intent from the handler) would be guessing.

Deterministic: `ast` only, never imports the code it analyses (no side effects, no import-order dependence).
"""

import ast
import os


def import_edges(path):
    """Every import in one file, classified by WHERE it sits: [{name, lineno, kind}] with kind in
    hard/guarded/deferred. Parses with `ast` -- the file is never imported, so analysing a module that needs a
    missing accelerator is safe. Returns [] on a syntax error (a broken file is audit_imports' job, not ours)."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            tree = ast.parse(fh.read(), filename=path)
    except (SyntaxError, ValueError):
        return []
    out = []

    def visit(node, in_func, in_try):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Import):
                for a in child.names:
                    out.append({"name": a.name, "lineno": child.lineno,
                                "kind": _kind(in_func, in_try)})
            elif isinstance(child, ast.ImportFrom):
                # level > 0 is a relative import; this tree uses absolute ones, so record the module as-is
                if child.module:
                    out.append({"name": child.module, "lineno": child.lineno,
                                "kind": _kind(in_func, in_try)})
            visit(child,
                  in_func or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)),
                  in_try or isinstance(child, ast.Try))

    visit(tree, False, False)
    return out


def _kind(in_func, in_try):
    # DEFERRED wins over GUARDED: an import inside a function does not run at import time whether or not it is
    # also wrapped in try, and "does it run at import" is the question a bundler is actually asking.
    if in_func:
        return "deferred"
    return "guarded" if in_try else "hard"


def _is_stdlib(name):
    """Is this top-level package part of the standard library? Uses sys.stdlib_module_names (3.10+), the
    interpreter's own answer -- no hand-maintained list to rot. Matters because a bundler cares about PIP
    dependencies; lumping `os` and `numpy` together as 'third-party' makes the one number anyone wants unreadable."""
    import sys
    return name.split(".")[0] in sys.stdlib_module_names


def _is_ours(name):
    """Is this one of OUR modules (worth resolving), or stdlib/third-party? Mirrors tools/audit_imports."""
    head = name.split(".")[0]
    return head.startswith("holographic") or head in (
        "holographic", "app", "tools", "lecore", "capdoc", "docgen", "apiquickref", "servicedoc")


def _module_index(root):
    """{dotted: relpath} plus {basename: {dotted, ...}} -- reuses select_tests.discover_modules so there is ONE
    module-discovery walk in the repo, not a second source of truth that can drift from it.

    WHY THE IMPORT DANCE: select_tests lives in tools/, which is on sys.path when tools are run as scripts (how
    audit_imports gets it for free) but not when this module is imported from inside the package. Both names below
    are the SAME file. Deliberately NO local-walk fallback: a fallback would be exactly the second source of truth
    this reuse exists to prevent, and it would drift silently the first time _IGNORE_DIRS changed. Analysing a
    source tree without the repo's own tools/ in it is out of scope, and says so."""
    try:
        from tools.select_tests import discover_modules
    except ImportError:
        try:
            from select_tests import discover_modules
        except ImportError:
            raise ImportError("deptrace needs the repo's tools/select_tests.py importable "
                              "(run from the repo root, or put tools/ on sys.path)")
    modules = discover_modules(root)
    by_base = {}
    for dotted in modules:
        by_base.setdefault(dotted.split(".")[-1], set()).add(dotted)
    return modules, by_base


def _resolve(name, modules, by_base):
    """Dotted name -> the canonical dotted module in this tree, or None. Handles the flat/packaged duality the
    backlog flags: a bare `holographic_foo` resolves to its packaged home, because that is what flatcompat's
    meta-path finder does at runtime and a tracer that disagreed with the runtime would be lying."""
    if name in modules:
        return name
    base = name.split(".")[-1]
    homes = by_base.get(base, set())
    if len(homes) == 1:
        return next(iter(homes))
    if name in homes:
        return name
    return None


def trace(entry, root=".", follow=("hard",)):
    """The footprint of `entry`: which of OUR modules it pulls in, and which third-party packages, split by kind.

    `entry` is a dotted module name (flat or packaged) or a path to a .py file. `follow` chooses which edge kinds
    the transitive walk crosses -- the default ("hard",) answers the bundler's question (what must exist for
    `import entry` to succeed). Pass ("hard", "guarded", "deferred") to reproduce a naive tracer's answer and see
    the balloon for yourself.

    Returns a dict:
      modules            -- sorted dotted names of OUR modules in the closure (includes `entry`)
      third_party        -- {"hard": [...], "guarded": [...], "deferred": [...]}, packages that are not ours
      edges_by_kind      -- counts of every classified edge seen while walking
      unresolved         -- ours-looking names that resolve to no file (audit_imports' territory; reported, not raised)
    """
    modules, by_base = _module_index(root)
    if entry.endswith(".py"):
        rel = os.path.relpath(os.path.abspath(entry), os.path.abspath(root))
        hits = [d for d, r in modules.items() if r == rel]
        if not hits:
            raise ValueError("%r is not a module under %r" % (entry, root))
        start = hits[0]
    else:
        start = _resolve(entry, modules, by_base)
        if start is None:
            raise ValueError("cannot resolve entry %r in %r" % (entry, root))

    follow = tuple(follow)
    seen, stack = {start}, [start]
    third = {"hard": set(), "guarded": set(), "deferred": set()}
    counts = {"hard": 0, "guarded": 0, "deferred": 0}
    unresolved = set()
    while stack:
        dotted = stack.pop()
        for e in import_edges(os.path.join(root, modules[dotted])):
            counts[e["kind"]] += 1
            if not _is_ours(e["name"]):
                third[e["kind"]].add(e["name"].split(".")[0])
                continue
            tgt = _resolve(e["name"], modules, by_base)
            if tgt is None:
                unresolved.add(e["name"])
                continue
            if e["kind"] in follow and tgt not in seen:
                seen.add(tgt)
                stack.append(tgt)
    external = {k: sorted(n for n in v if not _is_stdlib(n)) for k, v in third.items()}
    stdlib = {k: sorted(n for n in v if _is_stdlib(n)) for k, v in third.items()}
    return {"modules": sorted(seen),
            "third_party": {k: sorted(v) for k, v in third.items()},
            "external": external,                        # the PIP dependencies -- what a bundle must ship
            "stdlib": stdlib,                            # free, but listed so nothing is hidden
            "edges_by_kind": counts,
            "unresolved": sorted(unresolved)}


def footprint_report(entry, root="."):
    """The bundler's answer in one call: the REQUIRED closure vs what a naive follow-everything tracer reports.

    Returns {entry, required, naive, ratio, required_modules, required_external, optional_external,
    required_stdlib, edges_by_kind}. `ratio` is the balloon factor (naive/required) -- exactly how much a tracer
    that ignores WHERE an import sits over-reports by. `required_external` is the number that matters: the pip
    packages that must exist for `import entry` to succeed."""
    req = trace(entry, root=root, follow=("hard",))
    naive = trace(entry, root=root, follow=("hard", "guarded", "deferred"))
    n_req, n_naive = len(req["modules"]), len(naive["modules"])
    all_ext = set().union(*[set(v) for v in naive["external"].values()])
    return {"entry": entry, "required": n_req, "naive": n_naive,
            "ratio": (n_naive / n_req) if n_req else 0.0,
            "required_modules": req["modules"],
            "required_external": req["external"]["hard"],
            "optional_external": sorted(all_ext - set(req["external"]["hard"])),
            "required_stdlib": req["stdlib"]["hard"],
            "edges_by_kind": naive["edges_by_kind"]}


def _selftest():
    """Regression trap: the CLASSIFIER is pinned on a synthetic file with one import of each kind (that is the
    part everything else rests on), and the tracer is pinned against the live tree's real numbers."""
    import tempfile

    src = (
        "import os\n"                                   # hard
        "try:\n"
        "    import numba\n"                            # guarded
        "except ImportError:\n"
        "    numba = None\n"
        "from json import dumps\n"                      # hard (ImportFrom)
        "def f():\n"
        "    import cupy\n"                             # deferred
        "    try:\n"
        "        import pyfftw\n"                       # deferred (deferred beats guarded)
        "    except ImportError:\n"
        "        pass\n"
        "class C:\n"
        "    def m(self):\n"
        "        from sympy import Symbol\n"            # deferred, inside a method
    )
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "probe.py")
        with open(p, "w") as fh:
            fh.write(src)
        got = {(e["name"], e["kind"]) for e in import_edges(p)}
    assert ("os", "hard") in got, got
    assert ("json", "hard") in got, got
    assert ("numba", "guarded") in got, got
    assert ("cupy", "deferred") in got, got
    assert ("pyfftw", "deferred") in got, got           # KEPT RULE: deferred beats guarded, both are "not at import"
    assert ("sympy", "deferred") in got, got
    assert len(got) == 6, got

    # a syntax error is somebody else's problem, not a crash here
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "bad.py")
        with open(p, "w") as fh:
            fh.write("def (:\n")
        assert import_edges(p) == []

    # live tree: a real module's REQUIRED closure must be far smaller than the naive one, and numpy must be
    # required while numba/cupy must NOT be (the engine's contract: NumPy required, accelerators opt-in)
    rep = footprint_report("holographic.mesh_and_geometry.holographic_sdf2d", root=".")
    assert rep["required"] < rep["naive"], rep
    assert "numpy" in rep["required_external"], rep["required_external"]

    # THE CONSTITUTION, MEASURED: importing lecore must require NumPy and nothing else off pip. This turns hard
    # constraint #1 ("NumPy/Flask/stdlib/hashlib only in core") from a discipline into a GATE -- the day someone
    # top-level-imports torch or scipy anywhere in the import-time closure, this fails loudly and by name.
    core = footprint_report("lecore", root=".")
    assert core["required_external"] == ["numpy"], (
        "core import-time closure must be NumPy-only, got %s" % core["required_external"])
    for acc in ("numba", "cupy", "sympy", "torch", "scipy", "sklearn"):
        assert acc not in core["required_external"], (acc, core["required_external"])
    print("deptrace selftest OK (classifier 6/6 exact; sdf2d %d vs %d = %.0fx; CONSTITUTION PINNED: "
          "import lecore requires external=%s across %d modules, naive tracer says %d, edges %s)"
          % (rep["required"], rep["naive"], rep["ratio"], core["required_external"],
             core["required"], core["naive"], core["edges_by_kind"]))


if __name__ == "__main__":
    _selftest()

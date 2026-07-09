#!/usr/bin/env python3
"""wiring_report.py -- find modules nothing calls (T-5).

Discoverability and WIRING are different axes, and the engine has been bitten by both:

  * a DARK module has zero engine references of any kind. It has tests, it works, and nothing in the engine can
    reach it (`compose` and `diffusion` were here: real capability, no door).
  * a CATALOG-ONLY module is referenced by exactly one engine file -- the catalog. It is discoverable and never
    called (`hardening` and `ablate` were here, both load-bearing for standing plans).

Neither shows up as a test failure, which is why they persist. This reports both, from the same static import graph
`select_tests` uses (ast, no imports executed).

A module may legitimately have no callers: a KEPT NEGATIVE is kept precisely so nobody re-invents it, and a
standalone tool or entry point is not meant to be imported. Those are honoured:
  * a module whose docstring says "KEPT NEGATIVE" is exempt (and listed separately, so the annotation stays visible);
  * anything in EXEMPT below is exempt with a stated reason.

    python tools/wiring_report.py            # human report
    python tools/wiring_report.py --check    # exit 1 if a NEW dark/catalog-only module appeared
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_tests import discover_modules  # one module-discovery walk, not a second source of truth

CATALOG = "holographic_catalog"

# A module may be a documented DEAD END rather than a silo: it is kept so nobody re-invents it. The engine writes
# these two ways, so honour both (misgen says "KEPT NO-OP"; ldexplore says "KEPT NEGATIVE").
_NEGATIVE_MARKERS = ("KEPT NEGATIVE", "KEPT NO-OP")

# Modules with no engine callers ON PURPOSE. Each needs a reason -- "nothing imports it" is not one.
EXEMPT = {
    "holographic_unified": "the top-level facade: it imports everything, nothing imports it",
    "holographic_catalog": "the discoverability registry itself",
    "holographic_reference": "definitional reference implementations, used by the conformance harness (tests)",
    "benchmark_holographic": "a benchmark entry point",
    "stress_holographic": "an adversarial benchmark entry point",
}


# The dark modules that exist TODAY, each with an honest status. `--check` fails only on a module that is NOT here,
# so the debt is frozen and cannot grow. Wiring one means deleting its line -- progress recorded, like PENDING.
KNOWN_DARK = {
    "holographic_creature_mind": "reference DEMO of building a specialised mind ON UnifiedMind -- meant to be read, not imported.",
    "holographic_extras": "grab-bag; triage each function into a home, then delete the module.",
    "holographic_farm": "superseded in practice by coordinator.NetworkFarm (R3 prototype); archive or fold in.",
    "holographic_lexicon": "a curriculum EXPERIMENT (dictionary-first word meaning), not a library.",
    "holographic_photos": "a testing harness against real photographs, not a library.",
    "holographic_reanchor": "the AUDIT that proves re-anchoring is load-bearing; evidence, not a callable faculty.",
    "holographic_sdfscene": "a documented BASE CLASS for SDF scenes -- meant to be subclassed by callers, not imported by the engine.",
}


# Every file this run could not parse. AN AUDIT THAT DEGRADES SILENTLY IS WORSE THAN NO AUDIT: `ast.parse` failing
# used to return an empty import set, so ONE unparseable file made every module it alone references look DARK. And
# `holographic_unified.py` alone references 342 of them. The failure never printed anything; the report just got
# quietly, confidently wrong. Failures are now collected and reported, and `--check` exits non-zero on any of them.
PARSE_FAILURES = []


def _parse(path):
    """Parse a module, RECORDING any failure rather than swallowing it. Returns None if it could not be parsed."""
    try:
        return ast.parse(open(path, "r", encoding="utf-8", errors="ignore").read(), filename=path)
    except SyntaxError as exc:
        PARSE_FAILURES.append((path, "%s (line %s)" % (exc.msg, exc.lineno)))
        return None


def _module_docstring(path):
    tree = _parse(path)
    if tree is None:
        return ""
    return ast.get_docstring(tree) or ""


def _imports(path):
    """The project module BASENAMES this file imports (top-level and nested), e.g. 'holographic_render'.

    An unparseable file yields NO imports, which is indistinguishable from a file that imports nothing -- so the
    failure is recorded in PARSE_FAILURES and surfaced by the report. Do not make this quiet again."""
    tree = _parse(path)
    if tree is None:
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[-1])
    return names


def analyse(root="."):
    """Returns (dark, catalog_only, kept_negative): each a sorted list of module basenames."""
    modules = {}                                          # basename -> path, engine modules only
    for dotted, rel in discover_modules(root).items():
        rel_slash = rel.replace("\\", "/")
        base = dotted.split(".")[-1]
        if not rel_slash.startswith("holographic/") or base.startswith("test_"):
            continue
        if rel_slash.endswith("__init__.py"):             # a package, not a module -- nothing "imports" it by name
            continue
        modules[base] = os.path.join(root, rel)

    importers = {b: set() for b in modules}
    for base, path in modules.items():
        for target in _imports(path):
            if target in importers and target != base:
                importers[target].add(base)

    dark, catalog_only, kept_negative = [], [], []
    for base in sorted(modules):
        if base in EXEMPT:
            continue
        who = importers[base]
        doc = _module_docstring(modules[base]).upper()
        if any(mark in doc for mark in _NEGATIVE_MARKERS):
            if not who:
                kept_negative.append(base)                # documented dead end -- fine, but keep it visible
            continue
        if not who:
            dark.append(base)
        elif who == {CATALOG}:
            catalog_only.append(base)
    return dark, catalog_only, kept_negative


def main(argv):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dark, catalog_only, kept = analyse(root)

    print("DARK modules (zero engine references -- capability with no door): %d" % len(dark))
    for m in dark:
        print("   " + m)
    print("\nCATALOG-ONLY modules (discoverable, never called): %d" % len(catalog_only))
    for m in catalog_only:
        print("   " + m)
    print("\nKEPT NEGATIVES with no callers (expected, annotation honoured): %d" % len(kept))

    if PARSE_FAILURES:
        print("\nPARSE FAILURES (%d) -- this report is NOT trustworthy until these are fixed. A file that will not"
              % len(PARSE_FAILURES))
        print("parse contributes ZERO references, so every module it alone reaches is falsely reported DARK:")
        for path, why in PARSE_FAILURES:
            print("   %s: %s" % (path, why))

    if "--check" in argv:
        if PARSE_FAILURES:
            print("\nFAIL: %d module(s) could not be parsed; the import graph is incomplete." % len(PARSE_FAILURES))
            return 1
        new_dark = [m for m in dark if m not in KNOWN_DARK]
        fixed = [m for m in KNOWN_DARK if m not in dark]
        if new_dark or catalog_only:
            print("\nFAIL: these modules have no caller and are not recorded as known debt:")
            for m in new_dark + catalog_only:
                print("   " + m)
            print("Wire it, add it to EXEMPT/KNOWN_DARK with a reason, or annotate it KEPT NEGATIVE in its docstring.")
            return 1
        if fixed:
            print("\nThese are no longer dark -- delete them from KNOWN_DARK so the progress is recorded:")
            for m in fixed:
                print("   " + m)
            return 1
        print("\nOK: no new dark modules; no catalog-only modules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

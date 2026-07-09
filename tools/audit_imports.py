#!/usr/bin/env python3
"""audit_imports.py -- find imports that DON'T RESOLVE, so a restructure can't leave silent landmines.

WHY THIS EXISTS
---------------
When modules move (e.g. the flat `holographic_render.py` -> `holographic/rendering/holographic_render.py`), an import
that still names the OLD location keeps working by accident as long as the old file is lying around or the repo root
happens to be on sys.path -- and then breaks the day it isn't (a pip install, a different pytest invocation, CI).
`pytest` only catches these when the offending line actually runs, and an import inside a function may never run.

This walks EVERY .py in the repo with `ast` (never importing anything -- no side effects, fast, safe) and checks each
import against the modules that actually exist on disk. It reports:

  * BROKEN   -- an import of something that LOOKS like ours (holographic*, app, tools.*, a known root module) but
                resolves to no file on disk. These are real bugs waiting to happen.
  * FLAT     -- an import of a bare `holographic_foo` when the module actually lives in a package now
                (`holographic.x.holographic_foo`). Works only while the repo root is on sys.path; fragile.

Third-party and stdlib imports are ignored (they aren't ours to resolve).

Usage:
    python tools/audit_imports.py               # report; exit 1 if anything BROKEN
    python tools/audit_imports.py --flat        # also list the FLAT (fragile-but-working) imports
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_tests import discover_modules  # reuse the one module-discovery walk (no second source of truth)

# a module we consider "ours" -- worth resolving. Anything else is stdlib/third-party and skipped.
_OURS_PREFIXES = ("holographic", "app", "tools", "lecore", "capdoc", "docgen", "apiquickref", "servicedoc")


def _is_ours(name):
    head = name.split(".")[0]
    return head.startswith("holographic") or head in _OURS_PREFIXES


def _imports_with_lines(path):
    """[(dotted_name, lineno), ...] for every import in this file -- top-level AND nested inside functions."""
    try:
        tree = ast.parse(open(path, "r", encoding="utf-8", errors="ignore").read(), filename=path)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append((a.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                out.append((node.module, node.lineno))
    return out


def audit(root="."):
    """Returns (broken, flat): lists of (file, lineno, imported_name[, suggestion])."""
    modules = discover_modules(root)                      # {dotted_name: relpath}
    known = set(modules)
    # basename -> the dotted names it lives under, e.g. 'holographic_render' -> {'holographic.rendering.holographic_render'}
    by_base = {}
    for dotted in known:
        by_base.setdefault(dotted.split(".")[-1], set()).add(dotted)

    broken, flat = [], []
    for dotted, rel in sorted(modules.items()):
        path = os.path.join(root, rel)
        for name, line in _imports_with_lines(path):
            if not _is_ours(name) or name in known:
                continue                                  # not ours, or resolves fine
            base = name.split(".")[-1]
            homes = by_base.get(base, set()) - {name}
            if name == base and homes:
                # a bare `holographic_foo` that actually lives inside a package -> works only via sys.path luck
                flat.append((rel, line, name, sorted(homes)[0]))
            else:
                broken.append((rel, line, name, sorted(homes)[0] if homes else ""))
    return broken, flat


def main(argv):
    show_flat = "--flat" in argv
    broken, flat = audit(".")

    if broken:
        print("BROKEN imports (resolve to nothing on disk) -- %d:" % len(broken))
        for rel, line, name, hint in broken:
            print("  %s:%d  imports %r%s" % (rel, line, name, ("   -> did you mean %r?" % hint) if hint else ""))
    else:
        print("BROKEN imports: none")

    print("\nFLAT imports (bare name, module now lives in a package; works only while repo root is on sys.path) -- %d"
          % len(flat))
    if show_flat:
        for rel, line, name, hint in flat:
            print("  %s:%d  imports %r   -> %r" % (rel, line, name, hint))
    elif flat:
        print("  (re-run with --flat to list them)")

    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

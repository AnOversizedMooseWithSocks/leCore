#!/usr/bin/env python3
"""
docgen.py -- generate a human-readable code reference for leCore.

OLD-SCHOOL AND DEPENDENCY-FREE. This uses only the Python standard library (`ast`, `os`, `pathlib`). It reads
the plain-language docstring at the top of every `holographic_*.py` module, plus each module's public classes
and functions, and writes it all into one navigable REFERENCE.md. Because every module in this repo already
opens with a "why this exists" docstring, that IS most of the documentation -- this tool just gathers it into
one place a newcomer can read top to bottom.

    Run it:   python docgen.py
    Output:   REFERENCE.md   (a file/module map + a breakdown of every module and its public API)

Nothing here is clever; it is meant to be read. If you want it to say more, teach `summarise()` to pull more
out of the docstrings you already write.
"""

import ast
import os
from pathlib import Path

# ------------------------------------------------------------------------------------------------------------
# 1. FIND THE MODULES.  We document the engine modules (holographic_*.py) and skip the test files.
# ------------------------------------------------------------------------------------------------------------

def find_modules(root):
    """Return the sorted list of engine module paths -- every holographic_*.py that is not a test, wherever it
    lives under the holographic/ package (or, for backward compatibility, flat at the repo root)."""
    mods = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", ".venv", "venv") and not d.startswith(".")]
        # only descend into the holographic package and the repo root itself
        rel = os.path.relpath(dirpath, root)
        if rel != "." and not (rel == "holographic" or rel.startswith("holographic" + os.sep)):
            dirnames[:] = []  # don't descend further (skip tests/, tools/, docs/, etc.)
            continue
        for name in sorted(filenames):
            if name.startswith("holographic_") and name.endswith(".py") and not name.startswith("test_"):
                mods.append(Path(dirpath) / name)
    return sorted(mods, key=lambda p: p.name)


# ------------------------------------------------------------------------------------------------------------
# 2. READ ONE MODULE.  Parse it with `ast` (no importing, so it is safe and fast) and pull out what a reader
#    wants: the module's own docstring, and its PUBLIC functions/classes (names not starting with "_").
# ------------------------------------------------------------------------------------------------------------

def signature(fn):
    """A simple, readable argument list for a function/method: just the names, e.g. step(self, dt, gravity)."""
    a = fn.args
    names = [arg.arg for arg in a.posonlyargs + a.args]      # ordinary positional args
    if a.vararg:   names.append("*" + a.vararg.arg)          # *args
    if a.kwonlyargs: names += [k.arg for k in a.kwonlyargs]  # keyword-only args
    if a.kwarg:    names.append("**" + a.kwarg.arg)          # **kwargs
    return "%s(%s)" % (fn.name, ", ".join(names))


def first_line(docstring):
    """The first non-empty line of a docstring -- our one-line summary. Empty string if there is no docstring."""
    if not docstring:
        return ""
    for line in docstring.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def read_module(path):
    """Parse a module and return a small dict describing it: name, summary, full docstring, public API, LOC."""
    source = path.read_text(encoding="utf-8", errors="replace")
    loc = source.count("\n") + 1
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return dict(name=path.name, summary="(could not parse)", doc="", api=[], loc=loc)

    mod_doc = ast.get_docstring(tree) or ""
    api = []
    for node in tree.body:                                   # only TOP-LEVEL defs -- the module's public surface
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):                    # skip private helpers (leading underscore)
                continue
            kind = "class" if isinstance(node, ast.ClassDef) else "def"
            sig = node.name if kind == "class" else signature(node)
            api.append((kind, sig, first_line(ast.get_docstring(node))))
    return dict(name=path.name, summary=first_line(mod_doc), doc=mod_doc, api=api, loc=loc)


# ------------------------------------------------------------------------------------------------------------
# 3. GROUP FOR NAVIGATION.  The repo is a flat folder of ~280 modules; a newcomer needs it clustered. We group
#    automatically by the leading word of the module name (mesh*, splat*, ray*, ...), which discovers the
#    families without a hand-maintained list. Anything that doesn't cluster goes under "Core & standalone".
# ------------------------------------------------------------------------------------------------------------

# The families we cluster by name prefix -- purely for navigation. A module like holographic_meshqem.py joins
# the "mesh" family. Add a prefix here if a new cluster grows big enough to deserve its own table.
FAMILY_PREFIXES = ("mesh", "splat", "ray", "sdf", "scene")


def family_of(module_name):
    """Group key from a module name: the first matching family prefix, else the module's own name."""
    stem = module_name[len("holographic_"):-len(".py")]      # "holographic_meshqem.py" -> "meshqem"
    for prefix in FAMILY_PREFIXES:
        if stem.startswith(prefix):
            return prefix
    return stem                                              # no family -> keys on itself (folded in below)


def group_modules(mods):
    """Return {family: [modules]}. A prefix family with 3+ members gets its own group; everything else folds
    into one 'Core & standalone' group, so the map stays short and readable."""
    by_family = {}
    for m in mods:
        by_family.setdefault(family_of(m["name"]), []).append(m)

    grouped = {}
    standalone = []
    for family, members in by_family.items():
        if len(members) >= 3:
            grouped[family] = sorted(members, key=lambda m: m["name"])
        else:
            standalone.extend(members)                       # singletons and pairs go together
    if standalone:
        grouped["Core & standalone"] = sorted(standalone, key=lambda m: m["name"])
    return grouped


# ------------------------------------------------------------------------------------------------------------
# 4. WRITE THE MARKDOWN.  A newcomer intro, a family map, and then one section per module: its "why" docstring
#    and its public API. Kept plain so it renders anywhere and reads cleanly.
# ------------------------------------------------------------------------------------------------------------

NEWCOMER_INTRO = """\
> **New here? Read this first.** leCore represents *everything* -- memory, geometry, physics, rendering -- as
> points in one very high-dimensional space (hypervectors), and combines them with a tiny algebra: **bind**
> (glue two things together), **bundle** (overlay many into one), and **cleanup** (snap a noisy result to the
> nearest known thing). Almost every module below is one capability built from that algebra. Each module opens
> with a plain-language "why this exists" note -- this reference just gathers those, plus each module's public
> functions and classes, into one place. Start with the [README](README.md) for the big picture, then use the
> map below to find the area you care about.
"""

def write_reference(mods, out_path):
    """Assemble REFERENCE.md from the parsed modules."""
    grouped = group_modules(mods)
    total_loc = sum(m["loc"] for m in mods)
    lines = []
    w = lines.append

    # -- header --
    w("# leCore -- Code Reference")
    w("")
    w("*Auto-generated by `docgen.py` -- do not edit by hand; edit the module docstrings instead and re-run it.*")
    # No date here: REFERENCE.md is drift-checked; a timestamp would make it "stale" daily. Counts are
    # deterministic (they change only when code changes), which is exactly what the drift check wants.
    w("*%d modules, %s lines of engine code.*" % (len(mods), f"{total_loc:,}"))
    w("")
    w(NEWCOMER_INTRO)
    w("")

    # -- the map: a table of every module + its one-line summary, grouped by family --
    w("## Module map")
    w("")
    for fam in sorted(grouped, key=lambda f: (f == "Core & standalone", f)):   # families first, "Core" last
        members = sorted(grouped[fam], key=lambda m: m["name"])
        title = "`%s*` family" % fam if fam != "Core & standalone" else fam
        w("### %s (%d)" % (title, len(members)))
        w("")
        w("| module | what it is | lines |")
        w("|---|---|---|")
        for m in members:
            anchor = m["name"].replace(".py", "").replace("_", "-")
            summary = (m["summary"] or "").replace("|", "\\|")[:110]
            w("| [`%s`](#%s) | %s | %d |" % (m["name"], anchor, summary, m["loc"]))
        w("")

    # -- per-module detail: the full "why" docstring + the public API --
    w("---")
    w("")
    w("## Modules in detail")
    w("")
    for m in sorted(mods, key=lambda m: m["name"]):
        w("### %s" % m["name"])
        w("")
        if m["doc"]:
            # show the module docstring verbatim in a quote block (it is the hand-written "why")
            for line in m["doc"].strip().splitlines():
                w("> " + line if line.strip() else ">")
            w("")
        if m["api"]:
            w("**Public API:**")
            w("")
            for kind, sig, summary in m["api"]:
                w("- `%s %s`%s" % (kind, sig, (" -- " + summary) if summary else ""))
            w("")
        else:
            w("*(no public functions or classes -- internal or data-only)*")
            w("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(mods), total_loc


# ------------------------------------------------------------------------------------------------------------
# 5. MAIN.
# ------------------------------------------------------------------------------------------------------------

def main():
    root = Path(__file__).resolve().parent          # run from the repo root (docgen.py lives there)
    mods = [read_module(p) for p in find_modules(root)]
    n, loc = write_reference(mods, root / "REFERENCE.md")
    print("wrote REFERENCE.md -- %d modules, %d lines of code documented" % (n, loc))


if __name__ == "__main__":
    main()

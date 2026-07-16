#!/usr/bin/env python3
"""ONE DOOR for the generated docs: run every generator, in the canonical order, or --check them for drift.

WHY THIS EXISTS (a measured failure, not a tidy-up). The canonical list of doc generators lived in exactly one
place -- the step list inside .github/workflows/docs.yml -- and nowhere a human or an agent would look while
closing out a change. The predictable happened: a close-out ran `capdoc.py` and `docgen.py` (the two everybody
remembers), skipped `apiquickref.py`, and CI went red on the API_QUICKREF drift gate. The list was not hard;
it was just invisible. A list that lives only in a CI file is a list that gets half-run.

So: the list lives HERE, in code, importable and testable, and docs.yml + every close-out call the same thing.
tests/test_regen_docs.py pins the other half -- that every file ci.yml drift-GATES is actually produced by a
generator in this list. That pin is the point: a gate on a file nobody regenerates is a guaranteed red build,
and that is exactly the bug this module was born from.

NOT A GENERATOR, DELIBERATELY OMITTED: `servicedoc.py`. It writes nothing -- it CHECKS SERVICE.md's endpoint
table and CLI flags against the live service and exits non-zero, because most of SERVICE.md is hand-written
prose worth keeping (curl examples, security notes) and regenerating it wholesale would destroy that. ci.yml
runs it as its own gate. This list owns only files that are REGENERATED IN FULL; a checker in here would have
to either lie about "regenerating" or start overwriting prose. Recorded so it is not "helpfully" added later.

MISSING GENERATORS FAIL LOUDLY. An earlier version of this idea would have `continue`d past a generator that
was not on disk -- which is how a reconciliation that silently drops a file (it has happened here before, to
`facultymap.py`/`docmap.py` in a working tree, and to UnifiedMind helpers) turns into a green run that
regenerated nothing. Absent generator = non-zero exit, named on stderr.

Usage:
    python tools/regen_docs.py            # regenerate everything (what a close-out runs)
    python tools/regen_docs.py --check    # regenerate, report which tracked outputs CHANGED, exit 1 if any
    python tools/regen_docs.py --list     # print the canonical (generator -> outputs) table and exit

KEPT NEGATIVE / DETERMINISM CONTRACT: every generator here must be a PURE FUNCTION OF THE SOURCE. Two of them
(`apiquickref.py`, `docgen.py`) once stamped `date.today()` into their output, which made `--check` -- and
ci.yml's identical `git diff --exit-code` gate -- a measurement of the CALENDAR: red on any push made on a
later day than the last docs commit, with zero code changed. `capdoc.py`, same gate and no timestamp, never
false-failed; it was the control that proved the diagnosis. Both stamps are gone. If you add a generator that
writes a date, a hostname, a path, or a dict iteration order, you have not added a doc -- you have added a
flaky gate. `--check` re-running clean twice in a row is the contract; test_regen_docs.py pins it.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# The canonical list. Order matters only in that REFERENCE/CAPABILITIES are the ones humans read first; each
# generator is independent. `outputs` are repo-relative paths the generator OWNS -- used by --check and by the
# test that cross-references ci.yml's drift gates. Keep this in sync with .github/workflows/docs.yml, which
# calls this module rather than repeating the list.
GENERATORS = [
    ("docgen.py",      ["REFERENCE.md"]),
    ("capdoc.py",      ["CAPABILITIES.md", "capabilities.json"]),
    ("apiquickref.py", ["API_QUICKREF.md"]),
    ("facultymap.py",  ["docs/FACULTY_MAP.md"]),
    ("docmap.py",      ["docs/DOC_MAP.md"]),
    ("pipelinemap.py", ["docs/PIPELINE_MAP.md", "pipelines.json"]),
    # A generator MAY carry args: the string is split on whitespace, so a tool whose default CLI does
    # something else (unifiers.py prints a wiring table) can still own its doc through the one door.
    ("tools/unifiers.py --write", ["docs/UNIFIERS.md"]),
]


def repo_root():
    """The repo root (this file lives in tools/), so the generators run from where they expect to."""
    return Path(__file__).resolve().parent.parent


def outputs():
    """Every tracked file the generators own, as a flat list of repo-relative paths."""
    return [o for _g, outs in GENERATORS for o in outs]


def _read(path):
    """Bytes of `path`, or None when it does not exist yet -- a first run has nothing to compare against."""
    try:
        return Path(path).read_bytes()
    except OSError:
        return None


def run(check=False, root=None):
    """Run every generator from the repo root. Returns (missing, failed, changed) as lists of names/paths.

    `check=True` snapshots each output first and reports which ones the regeneration CHANGED -- the same
    question ci.yml's `git diff --exit-code` asks, but answerable without a git worktree (and so usable from a
    test or an extracted zip).
    """
    root = Path(root) if root else repo_root()
    before = {o: _read(root / o) for o in outputs()} if check else {}
    missing, failed, changed = [], [], []

    env = dict(os.environ, PYTHONHASHSEED="0")           # the constitution applies to the generators too
    for gen, _outs in GENERATORS:
        argv = gen.split()                               # a generator entry may carry args (see GENERATORS)
        if not (root / argv[0]).exists():
            missing.append(gen)                          # NEVER silently skip -- see the module docstring
            continue
        proc = subprocess.run([sys.executable] + argv, cwd=str(root), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            failed.append(gen)
            sys.stderr.write("FAILED: %s\n%s\n" % (gen, proc.stdout.decode("utf-8", "replace")[-2000:]))

    if check:
        for o in outputs():
            if before.get(o) != _read(root / o):
                changed.append(o)
    return missing, failed, changed


def main(argv=None):
    """CLI entry: regenerate (default), --check for drift, or --list the table."""
    ap = argparse.ArgumentParser(description="Run every generated-docs generator, or check them for drift.")
    ap.add_argument("--check", action="store_true", help="report drift and exit 1 if any output changed")
    ap.add_argument("--list", action="store_true", help="print the canonical generator -> outputs table")
    ap.add_argument("--outputs", action="store_true",
                    help="print just the output paths, space-separated -- so docs.yml's `git add` reads THIS "
                         "list instead of repeating it (a second copy of a list is a second copy of the bug)")
    args = ap.parse_args(argv)

    if args.outputs:
        print(" ".join(outputs()))
        return 0
    if args.list:
        for gen, outs in GENERATORS:
            print("%-16s -> %s" % (gen, ", ".join(outs)))
        return 0

    missing, failed, changed = run(check=args.check)
    for gen in missing:
        sys.stderr.write("MISSING GENERATOR: %s (expected at the repo root)\n" % gen)
    if missing or failed:
        sys.stderr.write("regen_docs: %d missing, %d failed\n" % (len(missing), len(failed)))
        return 1
    if args.check:
        if changed:
            sys.stderr.write("STALE (regenerated copy differs from the committed one):\n")
            for o in changed:
                sys.stderr.write("    %s\n" % o)
            sys.stderr.write("Fix: run `python tools/regen_docs.py` and commit the result.\n")
            return 1
        print("generated docs are up to date (%d outputs checked)" % len(outputs()))
        return 0
    print("regenerated %d outputs from %d generators" % (len(outputs()), len(GENERATORS)))
    return 0


def _selftest():
    """Assert the real contract: the table is well-formed, every gated file is covered, and --check is honest."""
    assert GENERATORS, "the canonical list must not be empty"
    outs = outputs()
    assert len(outs) == len(set(outs)), "two generators claim the same output file -- they would fight"
    for gen, o in GENERATORS:
        assert gen.endswith(".py") and o, gen

    # the load-bearing one: every file ci.yml drift-gates must be regenerated by something in this list,
    # otherwise that gate is a guaranteed red build nobody can fix. This is the bug this module exists for.
    ci = repo_root() / ".github" / "workflows" / "ci.yml"
    if ci.exists():
        import re
        text = ci.read_text(errors="replace")
        gated = set()
        for m in re.finditer(r"git diff --exit-code ([^\n|&]+)", text):
            gated.update(p.strip() for p in m.group(1).split() if p.strip().endswith((".md", ".json")))
        uncovered = gated - set(outs)
        assert not uncovered, "ci.yml gates files no generator here produces: %s" % sorted(uncovered)
        print("  ci.yml drift gates covered: %s" % sorted(gated))

    print("OK: regen_docs self-test passed (%d generators, %d outputs)" % (len(GENERATORS), len(outs)))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        sys.exit(main())

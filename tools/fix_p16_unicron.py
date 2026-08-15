"""Dedupe and split holographic_unified_p16_unicron.py -- the two test_unified_split failures.

WHY THIS IS MECHANICAL AND SAFE:
  * test_no_method_name_is_defined_in_two_parts reported each duplicate with owner == current
    module: the names are defined TWICE INSIDE p16 ITSELF. In a single class body Python keeps
    the LAST definition, so the FIRST body of each duplicated name is dead code that has never
    executed. Deleting it cannot change behaviour -- it is the additive-safe direction.
  * test_the_shim_stays_a_shim caps parts at 2000 LOC; p16 hit 3250. The split moves the TAIL
    methods (everything after the ~1600-LOC cumulative boundary, on a method edge) into a new
    _UnifiedPart17 in holographic_unified_p17_unicron2.py, then registers it in the shim's
    bases and unified_sources(). Moving whole methods between mixin bases is behaviour-neutral
    as long as no name lands in two parts -- which the dedupe step guarantees first, and the
    audit step re-checks last.

USAGE (from repo root, git-bash is fine):
    python tools/fix_p16_unicron.py --dry-run     # prints the plan, writes nothing
    python tools/fix_p16_unicron.py               # applies, then re-runs the AST audits
Idempotent: a second run finds nothing to do. Line endings of each file are detected and
preserved (the CRLF discipline: never let a tool quietly rewrite a CRLF file as LF).
"""
import argparse
import ast
import io
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P16 = os.path.join(REPO, "holographic", "unified", "holographic_unified_p16_unicron.py")
P17 = os.path.join(REPO, "holographic", "unified", "holographic_unified_p17_unicron2.py")
SHIM = os.path.join(REPO, "holographic", "misc", "holographic_unified.py")
SPLIT_BUDGET = 1600     # keep p16 comfortably under the 2000 gate; the tail goes to p17
PART17_CLASS = "_UnifiedPart17"


def read(path):
    raw = open(path, "rb").read()
    nl = "\r\n" if b"\r\n" in raw else "\n"
    return raw.decode("utf-8"), nl


def write(path, text, nl):
    # Preserve the file's own newline convention byte-for-byte (CRLF discipline).
    data = text.replace("\r\n", "\n").replace("\n", nl).encode("utf-8")
    open(path, "wb").write(data)


def class_and_methods(src):
    tree = ast.parse(src)
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    assert len(cls) == 1, "expected exactly one part class in p16, found %d" % len(cls)
    meths = [n for n in cls[0].body if isinstance(n, ast.FunctionDef)]
    return cls[0], meths


def dedupe(src):
    """Drop every non-final definition of a duplicated method name. Returns (new_src, dropped)."""
    lines = src.splitlines(keepends=True)
    cls, meths = class_and_methods(src)
    last_of = {}
    for m in meths:
        last_of[m.name] = m                       # later definition overwrites: LAST wins, as Python does
    doomed = [m for m in meths if last_of[m.name] is not m]
    # Delete from the bottom up so earlier line numbers stay valid. end_lineno includes decorators? No:
    # decorators sit ABOVE lineno, so start from the first decorator line if any.
    doomed.sort(key=lambda m: m.lineno, reverse=True)
    dropped = []
    for m in doomed:
        start = min([d.lineno for d in m.decorator_list] + [m.lineno]) - 1
        end = m.end_lineno                        # inclusive, 1-based -> slice end is fine
        dropped.append((m.name, start + 1, end))
        del lines[start:end]
    return "".join(lines), dropped


def plan_split(src):
    """Choose the method boundary where cumulative LOC crosses SPLIT_BUDGET. Returns
    (head_src_for_p16, tail_method_names, tail_block_text, header_end_line)."""
    lines = src.splitlines(keepends=True)
    cls, meths = class_and_methods(src)
    assert meths, "no methods found -- nothing to split"
    cut = None
    for m in meths:
        if m.end_lineno > SPLIT_BUDGET:
            cut = m
            break
    if cut is None:
        return src, [], "", None                  # already small enough
    start = min([d.lineno for d in cut.decorator_list] + [cut.lineno]) - 1
    tail_names = [m.name for m in meths if m.lineno >= cut.lineno]
    head = "".join(lines[:start]).rstrip() + "\n"
    tail = "".join(lines[start:])
    return head, tail_names, tail, start


P17_HEADER = '''"""Unicron faculties, part two of two -- split from holographic_unified_p16_unicron.py.

WHY THIS FILE EXISTS: test_the_shim_stays_a_shim caps every part at 2000 lines (the gate that
stopped holographic_unified.py growing back to 17k lines one faculty at a time). p16 crossed it
at 3250; the tail methods moved here verbatim. Same rules as every part: NOT a standalone
module, no __init__, assumes the state UnifiedMind.__init__ builds; it only exists as a base of
UnifiedMind. See tests/test_unified_split.py for the full contract.
"""


class %s:
''' % PART17_CLASS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(P16):
        print("p16 not found at %s -- run from the tree that has the unicron work." % P16)
        return 1

    src, nl = read(P16)

    # ---- step 1: dedupe (must come first; the split must not scatter a duplicate across parts)
    deduped, dropped = dedupe(src)
    for name, a, b in dropped:
        print("DEDUPE: dropping dead first definition of %-24s (lines %d-%d; the later body "
              "is what Python was already running)" % (name, a, b))
    if not dropped:
        print("DEDUPE: no duplicate method names in p16 -- nothing to drop")

    # ---- step 2: split at a method boundary near SPLIT_BUDGET
    head, tail_names, tail, at = plan_split(deduped)
    if tail_names:
        print("SPLIT: p16 keeps %d lines; %d methods move to p17: %s"
              % (head.count("\n"), len(tail_names), ", ".join(tail_names)))
    else:
        print("SPLIT: p16 already under budget after dedupe -- no split needed")

    if args.dry_run:
        print("dry-run: nothing written")
        return 0

    write(P16, head, nl)
    if tail_names:
        # tail methods are class-body indented already; the new class header takes them verbatim
        write(P17, P17_HEADER + tail, nl)
        print("wrote %s" % P17)

        # ---- step 3: register p17 in the shim (bases + unified_sources), CRLF-preserving
        shim_src, shim_nl = read(SHIM)
        did = 0
        p16_mod = "holographic_unified_p16_unicron"
        p17_mod = "holographic_unified_p17_unicron2"
        for old, new in [
            # wiring-honesty style: import PKG.MODULE as X so reachability_audit can see it
            ("import holographic.unified.%s" % p16_mod,
             "import holographic.unified.%s" % p16_mod),  # anchor check only
        ]:
            pass
        # Find the p16 import line and mirror it for p17 directly below; then insert Part17 into the
        # bases tuple ON the `class UnifiedMind(...)` line (order is irrelevant while no name is
        # duplicated -- pinned by test_no_method_name_is_defined_in_two_parts -- so appending after
        # Part16 is safe). unified_sources() derives from the live bases, so no third edit exists.
        lines = shim_src.splitlines(keepends=True)
        out, done_import, done_base = [], False, False
        for ln in lines:
            if (not done_base) and "class UnifiedMind(" in ln and "_UnifiedPart16" in ln \
                    and PART17_CLASS not in ln:
                ln = ln.replace("_UnifiedPart16", "_UnifiedPart16, %s" % PART17_CLASS, 1)
                done_base = True
            out.append(ln)
            if (not done_import) and p16_mod in ln and ("import" in ln) and p17_mod not in shim_src:
                out.append(ln.replace(p16_mod, p17_mod).replace("_UnifiedPart16", PART17_CLASS))
                done_import = True
        if not (done_import and done_base):
            print("WARN: could not auto-wire the shim (import mirrored: %s, base mirrored: %s)."
                  % (done_import, done_base))
            print("      Add %s next to every occurrence of _UnifiedPart16 in %s by hand:"
                  % (PART17_CLASS, SHIM))
            print("      the import, the UnifiedMind bases tuple, and unified_sources().")
        else:
            write(SHIM, "".join(out), shim_nl)
            print("shim wired: import + bases entry mirrored from p16")
            did = 1

    # ---- step 4: re-run the exact AST audits the tests run
    for path in ([P16, P17] if tail_names else [P16]):
        s, _ = read(path)
        tree = ast.parse(s)
        cls = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        assert len(cls) == 1, "%s: expected one class" % path
        names = [n.name for n in cls[0].body if isinstance(n, ast.FunctionDef)]
        assert len(names) == len(set(names)), "%s still has duplicate methods" % path
        loc = s.count("\n") + 1
        print("AUDIT: %-45s %5d lines, %3d methods, no dupes"
              % (os.path.basename(path), loc, len(names)))
        assert loc < 2000, "%s still over the 2000-line gate" % path
    print("done. Now run: python -m pytest tests/test_unified_split.py -q")
    return 0


if __name__ == "__main__":
    sys.exit(main())

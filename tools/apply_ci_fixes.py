#!/usr/bin/env python3
"""Apply the two outstanding CI fixes to whatever state this checkout is in. IDEMPOTENT -- safe to run twice.

WHY A SCRIPT AND NOT A PATCH: a unified diff needs the exact surrounding context, and the tree these fixes
target has drifted (a delivery may or may not have been applied, in whole or in part). A script that finds
its targets by NAME and no-ops when the work is already done applies correctly from either state, which a
context patch cannot.

    python3 tools/apply_ci_fixes.py            # apply
    python3 tools/apply_ci_fixes.py --check    # report only, change nothing

FIX 1  tests/test_holographic_catalog.py -- test_field_query_regression_is_recorded_not_hidden was a
       TRIPWIRE asserting a BROKEN state ("Field is NOT in the top 3 for 'represent a density volume over
       space'"). The ranking was FIXED by giving the Field capability the phrase a user actually types, so
       the tripwire now fires exactly as its author designed. Its own message says to close J-3D-26; this
       replaces it with the mirror-image pin so the fixed state stays pinned from the suite (the module
       _selftest only runs under `python -m`, which is how the original rot sat green).

FIX 2  holographic/unified/ -- p09 grew past the 2000-line part cap while recovering scene/asset faculties.
       fetch_asset / load_hdr / load_image move to p01_READ, the part that is literally about input. The cap
       is NOT raised: the budget is the entire point of the split.
"""
import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MOVE = ("fetch_asset", "load_hdr", "load_image")

_PIN = '''def test_the_field_query_ranking_stays_fixed():
    """J-3D-26, CLOSED. This slot used to hold a TRIPWIRE that asserted the BROKEN state: 'represent a
    density volume over space' did not surface the Field capability, the module _selftest asserted that it
    did, and the selftest had been failing unnoticed. The tripwire existed so that whoever fixed the ranking
    would be told, rather than the regression being quietly relaxed into noise. It did its job.

    THE FIX WAS ADDITIVE, which is the part worth keeping: Field carried only single-word aliases ('field',
    'grid', 'volume', ...), and single words lose to descriptively-titled siblings as a catalog grows -- so
    the PHRASE a person actually types was added, not a threshold lowered and not a neighbour demoted."""
    from holographic.caching_and_storage.holographic_catalog import default_catalog
    hits = [h.name for h in default_catalog().find_capability("represent a density volume over space")]
    assert any("Field" in n for n in hits[:3]), \\
        "the Field ranking regressed again (top-3: %r) -- strengthen Field's aliases with the phrasing a " \\
        "user types; do NOT relax this or demote the neighbour that outranked it" % hits[:3]

'''


def _cut_method(text, name):
    """Return (text_without_method, method_block) or (text, None) when the method is not there."""
    m = re.search(r'\n    def %s\(self[^\n]*\n' % re.escape(name), text)
    if not m:
        return text, None
    start = m.start() + 1
    # BOUND BY THE NEXT SIBLING **OR** BY MODULE LEVEL. Searching only for the next `    def ` runs to EOF
    # when the method is the LAST in the class -- which silently swallows the module-level _selftest and the
    # __main__ guard below it. Found by running this script against a reconstruction of the broken tree
    # rather than only against an already-fixed one: a fix script must be tested on the state it repairs.
    nxt = re.compile(r'\n    def ').search(text, m.end())
    mod = re.compile(r'\n(?=[^\s#])').search(text, m.end())          # first line back at column 0
    ends = [x.start() + 1 for x in (nxt, mod) if x]
    end = min(ends) if ends else len(text)
    return text[:start] + text[end:], text[start:end]


def fix_tripwire(check=False):
    p = ROOT / "tests" / "test_holographic_catalog.py"
    t = p.read_text(encoding="utf-8")
    if "def test_field_query_regression_is_recorded_not_hidden():" not in t:
        return "already done"
    start = t.index("def test_field_query_regression_is_recorded_not_hidden():")
    nxt = re.compile(r'\ndef test_', re.S).search(t, start + 10)
    end = nxt.start() + 1 if nxt else len(t)
    if not check:
        p.write_text(t[:start] + _PIN + t[end:], encoding="utf-8")
    return "replaced the tripwire with the fixed-state pin"


def fix_part_size(check=False):
    p09 = ROOT / "holographic" / "unified" / "holographic_unified_p09_navigate_cost_field.py"
    p01 = ROOT / "holographic" / "unified" / "holographic_unified_p01_read.py"
    s9, s1 = p09.read_text(encoding="utf-8"), p01.read_text(encoding="utf-8")
    blocks = []
    for name in MOVE:
        s9, b = _cut_method(s9, name)
        if b is not None:
            blocks.append(b.rstrip("\n") + "\n")
    if not blocks:
        return "already done"
    anchor = re.search(r'\n\ndef _selftest\(\)', s1)
    if not anchor:
        raise SystemExit("p01 has no module-level _selftest to anchor against -- aborting rather than guessing")
    s1 = s1[:anchor.start()] + "\n\n" + "\n".join(blocks).rstrip("\n") + "\n" + s1[anchor.start():]
    if not check:
        p09.write_text(s9, encoding="utf-8")
        p01.write_text(s1, encoding="utf-8")
    return "moved %s from p09 to p01_read" % ", ".join(MOVE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report what would change; write nothing")
    a = ap.parse_args()
    for label, fn in (("tripwire", fix_tripwire), ("part size", fix_part_size)):
        print("  %-10s %s" % (label, fn(check=a.check)))
    if not a.check:
        big = [(p.name, sum(1 for _ in p.open(encoding="utf-8")))
               for p in sorted((ROOT / "holographic" / "unified").glob("holographic_unified_p*.py"))]
        worst = max(big, key=lambda kv: kv[1])
        print("  largest part now: %s at %d lines (cap 2000)" % worst)
        if worst[1] >= 2000:
            print("  STILL OVER -- rebalance another coherent group out of that part")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

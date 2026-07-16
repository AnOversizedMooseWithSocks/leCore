"""Structure audit -- is the tree ORGANIZED, or drifting toward spaghetti?

The other audits guard wiring (wiring_report), the catalog (catalog_gaps, skill_lint) and reachability
(reachability_audit). None of them watch STRUCTURE: family placement, monolith growth, the junk-drawer share of
misc/, and whether the 1,200+ methods on UnifiedMind keep any navigable internal sectioning. Organization is a
discoverability property for HUMANS (find_capability doesn't care what folder a module sits in; a person grepping
the tree does), so it deserves the same treatment as the rest: measure it, budget the current state, and FAIL only
on regression past the budget -- the same budgeted-baseline pattern skill_lint uses for over-long does-fields, so
the gate catches rot without demanding a rewrite of history.

WHY report-only cohesion: a low shared-vocabulary score does NOT prove a module is a grab-bag. Measured negative,
kept loud: holographic_vision (rgb/edges/hough/kmeans -- 8/23 token coverage) and holographic_dictionary
(load/define/synonyms -- 4/18) both LOOK scattered by token metrics and are in fact perfectly cohesive one-topic
toolkits; classic CV and dictionary vocabularies are just diverse. So cohesion is REPORTED for human eyes, never
gated. What IS gated: (1) misc/ must not grow past its budget (new modules should land in a real family), (2)
unified.py's section-marker count must not shrink (navigation markers are load-bearing), (3) the giant-module set
must not gain members past budget (a new 2,000+ line monolith is a review event, not a habit).

Run:  python3 tools/structure_audit.py            # report + gate (exit 1 on regression past budget)
"""
import ast
import glob
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- BUDGETS (the measured state when this audit landed; regressions past these fail) --------------------
# WHY budgets, not zeros: the tree already has 149 misc modules and 8 giants -- history is paid for. The gate's
# job is to stop the NEXT one from landing silently, exactly like skill_lint's 58 budgeted over-600 does-fields.
MISC_BUDGET = 149            # holographic/misc/*.py module count must not exceed this
GIANTS_BUDGET = 3            # modules over GIANT_LOC lines (measured: unified, catalog, creature)
GIANT_LOC = 2000
UNIFIED_MARKERS_MIN = 27     # "# ----" section markers inside unified.py must not drop below this


def module_rows():
    """(path, n_classes, n_public_fns, loc) for every holographic_* module -- the raw structural facts."""
    rows = []
    for p in glob.glob(os.path.join(REPO, "holographic", "**", "holographic_*.py"), recursive=True):
        try:
            tree = ast.parse(open(p, encoding="utf-8", errors="ignore").read())
        except SyntaxError:
            continue                                    # the syntax gate is reachability_audit's job, not ours
        n_cls = sum(1 for n in tree.body if isinstance(n, ast.ClassDef))
        n_fns = sum(1 for n in tree.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("_"))
        loc = sum(1 for _ in open(p, encoding="utf-8", errors="ignore"))
        rows.append((os.path.relpath(p, REPO), n_cls, n_fns, loc))
    return rows


def family_distribution(rows):
    """{family: module_count} -- where the tree's mass sits."""
    fam = {}
    for path, _c, _f, _l in rows:
        parts = path.split(os.sep)
        family = parts[1] if len(parts) > 2 else "(root)"
        fam[family] = fam.get(family, 0) + 1
    return fam


def unified_marker_count():
    """How many '# ----' section markers unified.py still carries -- its only human navigation aid."""
    p = os.path.join(REPO, "holographic", "misc", "holographic_unified.py")
    txt = open(p, encoding="utf-8", errors="ignore").read()
    return len(re.findall(r"^\s*# ----", txt, flags=re.M))


def prefix_clusters(min_size=3):
    """UnifiedMind's public methods grouped by their first name token -- the topical map facultymap.py renders.
    Imported lazily so the audit stays runnable even if a mind can't boot (it then reports fs-only facts)."""
    sys.path.insert(0, REPO)
    import lecore                                        # noqa: deferred -- booting a mind is the expensive part
    m = lecore.UnifiedMind(dim=64, seed=0)
    meths = [n for n in dir(m) if not n.startswith("_") and callable(getattr(type(m), n, None))]
    groups = {}
    for n in meths:
        groups.setdefault(n.split("_")[0], []).append(n)
    big = {k: v for k, v in groups.items() if len(v) >= min_size}
    singles = sum(len(v) for k, v in groups.items() if len(v) < min_size)
    return len(meths), big, singles


def main():
    rows = module_rows()
    fam = family_distribution(rows)
    total = sum(fam.values())
    misc = fam.get("misc", 0)
    giants = sorted([r for r in rows if r[3] > GIANT_LOC], key=lambda r: -r[3])
    markers = unified_marker_count()

    print("STRUCTURE AUDIT")
    print("  modules: %d across %d families" % (total, len(fam)))
    for k in sorted(fam, key=lambda k: -fam[k]):
        print("    %-28s %4d  (%d%%)" % (k, fam[k], 100 * fam[k] // max(total, 1)))
    print("  misc/ share: %d (budget %d) -- new modules should land in a REAL family" % (misc, MISC_BUDGET))
    print("  giants > %d loc: %d (budget %d)" % (GIANT_LOC, len(giants), GIANTS_BUDGET))
    for path, c, f, loc in giants:
        print("    %-56s %6d loc  classes=%d pubfns=%d" % (path, loc, c, f))
    print("  unified.py section markers: %d (min %d)" % (markers, UNIFIED_MARKERS_MIN))

    try:
        n_meth, big, singles = prefix_clusters()
        print("  UnifiedMind methods: %d; prefix clusters (>=3): %d; singletons: %d" % (n_meth, len(big), singles))
        print("  (run facultymap.py to regenerate docs/FACULTY_MAP.md from these clusters)")
    except Exception as e:                               # fs-only mode: still useful without a bootable mind
        print("  (mind boot skipped: %s)" % e)

    fails = []
    if misc > MISC_BUDGET:
        fails.append("misc/ grew to %d (> budget %d): put the new module in a real family" % (misc, MISC_BUDGET))
    if len(giants) > GIANTS_BUDGET:
        fails.append("giant modules grew to %d (> budget %d): a new %d+ line monolith needs review"
                     % (len(giants), GIANTS_BUDGET, GIANT_LOC))
    if markers < UNIFIED_MARKERS_MIN:
        fails.append("unified.py markers fell to %d (< %d): section markers are load-bearing navigation"
                     % (markers, UNIFIED_MARKERS_MIN))
    if fails:
        print("FAIL:")
        for f in fails:
            print("  - " + f)
        return 1
    print("OK: no structural regression past budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

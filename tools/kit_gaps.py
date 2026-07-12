"""tools/kit_gaps.py -- audit the INVARIANT KIT across an abstraction-ladder tower, sibling of catalog_gaps.py.

The abstraction ladder (holographic_ladder) says every LEVEL carries an invariant kit (§3a): delta, chunk, scale,
decompose, cleanup, bake, seed, tile, lod, canonical, cache, fuse, superpose, guide. Each slot must be either
WIRED (names the primitive it delegates to) or a DECLARED NEGATIVE (a reason it does not apply, prefixed 'no ').
An EMPTY slot with no reason is a SILENT GAP -- the thing this tool exists to forbid, exactly as the reachability
audit forbids unwired modules and catalog_gaps forbids undiscoverable capabilities.

Run it after any change to the ladder's level-building or the kit definition:

    python tools/kit_gaps.py

It climbs a few representative corpora (sequence + structure lenses, planted + noise), reports every level's kit,
and exits non-zero if ANY level has a silent gap. Target: 0 silent gaps.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holographic.agents_and_reasoning.holographic_ladder import (
    climb, kit_report, _make_planted_corpus, _make_instanced_scene_corpus,
)


def _audit_tower(name, tower):
    """Print each level's kit summary and return the total silent-gap count for this tower."""
    total_gaps = 0
    print("== %s (%d level(s)) ==" % (name, len(tower)))
    for lv in tower:
        rep = kit_report(lv)
        gaps = rep["silent_gaps"]
        total_gaps += len(gaps)
        flag = ("  SILENT GAPS: " + ", ".join(gaps)) if gaps else ""
        print("  depth %d: %d wired, %d declared-negative, %d silent%s"
              % (lv["depth"], len(rep["wired"]), len(rep["declared_negative"]), len(gaps), flag))
        if lv["depth"] == 0:
            # show the declared negatives once, so the reasons are visible and reviewable.
            for slot in rep["declared_negative"]:
                print("      declared-negative[%s]: %s" % (slot, lv["kit"][slot]))
    return total_gaps


def main():
    towers = {
        "planted sequence corpus (sequence lens)": climb(_make_planted_corpus(seed=0)),
        "instanced scene (structure lens)": climb(_make_instanced_scene_corpus(seed=0), lens="structure"),
    }
    total = 0
    for name, tower in towers.items():
        total += _audit_tower(name, tower)
        print()
    print("TOTAL SILENT KIT GAPS: %d (target 0)" % total)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())

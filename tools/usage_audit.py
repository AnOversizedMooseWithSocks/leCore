#!/usr/bin/env python3
"""USAGE AUDIT -- which capabilities does NOTHING call?

reachability_audit asks whether a capability is DISCOVERABLE. This asks whether
anything USES it, and they are different questions: this repo read 0/0/0 on
reachability while most of an arc's work sat as library code nothing invoked.

A module that only its own selftest imports is a module that will rot, because
nothing else fails when it breaks.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Phrases a module uses to DECLARE that nothing should call it. This mirrors
#: reachability_audit, which distinguishes "IMPORT-ONLY" from "IMPORT-ONLY, not
#: a declared negative" -- an audit that cannot be told "this is deliberate"
#: becomes noise, and noise gets ignored, and then it catches nothing.
#: BARE "KEPT NEGATIVE" IS NOT ENOUGH, and the first version of this audit was
#: wrong because of it. That phrase marks a REFUTED IDEA in hundreds of modules
#: -- "what this deliberately does NOT do" -- which is a different claim from
#: "nothing should call this". holographic_objectref says the first and got a
#: FALSE PASS, taking the count from 1 unused to 0 and making the audit lie in
#: exactly the direction that feels like progress.
DECLARED = (
    "no engine door on purpose",
    "SUPERSEDED BY holographic_",
    "TEST/RESEARCH HARNESS",
)

#: Modules that are ENTRY POINTS by design -- a harness, a tool, a planner.
#: Nothing importing them is correct, so they are not gaps.
ENTRYPOINTS = {
    "holographic_lecorerun",     # the runtime loop; harnesses call it
    "holographic_install_lecore",  # the installer; install.py calls it
    "holographic_devicerun", "holographic_modelvault", "holographic_proglib",
    "holographic_vminstall", "holographic_unlocked", "holographic_installorder",
    "holographic_billionctx", "holographic_actr", "holographic_statetrack",
    "holographic_hybrid", "holographic_selfheal", "holographic_adapt",
}


def main():
    mods, imports = {}, {}
    # SCAN THE ROOT TOO. The first version walked only holographic/ and reported
    # holographic_objectref as called by nothing -- while holographic_service.py,
    # a ROOT-LEVEL module, imports it and passes an ObjectRefs registry into
    # _jsonable on every /invoke. A FALSE POSITIVE, after the earlier false
    # negative, from an audit whose SCOPE was narrower than the thing it audits.
    # An audit is only as honest as the set it walks.
    roots = [os.path.join(ROOT, "holographic"), ROOT]
    for base in roots:
      for dp, _dn, fns in (os.walk(base) if base.endswith("holographic")
                           else [(base, [], os.listdir(base))]):
        for fn in fns:
            if not fn.startswith("holographic_") or not fn.endswith(".py"):
                continue
            name = fn[:-3]
            path = os.path.join(dp, fn)
            # a module found in BOTH places is the same module; keep the first
            if name not in mods:
                mods[name] = path
            try:
                src = open(path, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            for m in re.finditer(r"holographic_(\w+)", src):
                other = "holographic_" + m.group(1)
                if other != name:
                    imports.setdefault(other, set()).add(name)
    # the unified facade counts as a caller: it is how agents reach things
    orphans, declared = [], []
    for name in sorted(mods):
        callers = imports.get(name, set())
        if callers or name in ENTRYPOINTS:
            continue
        try:
            head = open(mods[name], encoding="utf-8",
                        errors="ignore").read(4000)
        except Exception:
            head = ""
        if any(p in head for p in DECLARED):
            declared.append(name)
        else:
            orphans.append(name)
    print("USAGE AUDIT: %d modules, %d called by nothing, %d DECLARED"
          % (len(mods), len(orphans), len(declared)))
    for d in declared:
        print("   declared %s" % d)
    for o in orphans[:20]:
        print("   UNUSED  %s" % o)
    if len(orphans) > 20:
        print("   ... and %d more" % (len(orphans) - 20))
    print("TOTAL: %d unused module(s)" % len(orphans))
    return 0


if __name__ == "__main__":
    sys.exit(main())

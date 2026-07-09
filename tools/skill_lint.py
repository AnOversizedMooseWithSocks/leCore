"""tools/skill_lint.py -- flag engine faculties an agent can't invoke confidently from their docstring.

WHY
---
holographic_skills turns every public UnifiedMind method into a "skill card" whose summary is the first line of the
method's docstring. If that line is missing or too thin, an agent (or a person reading mind.suggest / mind.describe_skill)
gets a signature but no idea what the method DOES or what it returns -- so it can't invoke it with confidence. This is
the invocation-quality twin of tools/catalog_gaps.py: that tool finds capabilities you can't FIND; this one finds
capabilities you can find but can't confidently CALL.

WHAT IT CHECKS (per public UnifiedMind method)
  * CRITICAL -- no docstring at all: the agent gets nothing but the bare signature.
  * TERSE    -- a summary under MIN_SUMMARY_WORDS words: too thin to tell the method apart or know its effect.
  * NO_RETURN (note only) -- the docstring never hints at a return value AND the method isn't obviously a mutator;
    an agent often needs to know what comes back. This is advisory, not a failure.

The check is deterministic and stdlib-only. Run it after wiring new faculties; fix anything CRITICAL/TERSE.

    python tools/skill_lint.py            # human-readable report; exit code = number of CRITICAL+TERSE gaps
    python tools/skill_lint.py --strict   # also list the NO_RETURN notes
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MIN_SUMMARY_WORDS = 5              # a one-line summary shorter than this tells an agent almost nothing

# words that show the docstring says something about what comes back
_RETURN_HINTS = ("return", "returns", "->", "yields", "gives", "produces", "a list", "a dict", "self")


def _summary_words(summary):
    return [w for w in summary.replace("-", " ").split() if any(ch.isalpha() for ch in w)]


def audit():
    """Return {"critical": [...], "terse": [(name, summary)], "no_return": [name]} over public UnifiedMind methods.

    critical/terse look at the SUMMARY (first line, what a skill card shows). no_return scans the FULL docstring -- a
    method that never once hints at what it returns leaves an agent guessing about the result it will get back."""
    import holographic.misc.holographic_skills as sk
    from holographic.misc.holographic_unified import UnifiedMind
    ms = sk.mind_methods()
    critical, terse, no_return = [], [], []
    for name in sorted(ms):
        summary = ms[name]["summary"]
        if not summary:
            critical.append(name)
            continue
        if len(_summary_words(summary)) < MIN_SUMMARY_WORDS:
            terse.append((name, summary))
        # advisory: scan the WHOLE docstring (not just the first line) for any hint of a return value
        full = (getattr(UnifiedMind, name).__doc__ or "").lower()
        if not any(h in full for h in _RETURN_HINTS):
            no_return.append(name)
    return {"critical": critical, "terse": terse, "no_return": no_return}


def audit_home_examples():
    """Audit the MODULE-LEVEL functions named in curated catalog homes' `example` strings -- the ones an agent calls
    directly by copying the example -- both "from holographic_X import Y" module refs AND "mind.method(" refs. Returns
    {"broken": [(mod, name, home, why)], "no_doc": [...], "terse": [...]}.

    'broken' is the worst: the example references a name that does not exist (or a module that won't import), so an
    agent copying it gets an ImportError/AttributeError before it even runs. no_doc/terse mirror the method checks."""
    import re
    import importlib
    from holographic.caching_and_storage.holographic_catalog import default_catalog
    cat = default_catalog()
    # Examples now use dotted package paths (holographic.<family>.holographic_x), so the module pattern must
    # allow dots -- the old holographic_\w+ only matched the pre-reorg FLAT names and silently stopped checking
    # the dotted ones. [\w.]+ matches both the flat legacy form and the current dotted form.
    pat_import = re.compile(r"from ([\w.]*holographic_\w+) import ([\w, ]+)")
    pat_dotted = re.compile(r"([\w.]*holographic_\w+)\.(\w+)\(")
    pat_mind = re.compile(r"\bmind\.(\w+)\(")                   # mind.method( references an agent copies directly
    refs = {}                                                   # (module, name) -> home name
    mind_refs = {}                                             # method_name -> home name
    for c in cat.all():
        ex = c.example or ""
        for m in pat_import.finditer(ex):
            for fn in m.group(2).split(","):
                fn = fn.strip()
                if fn and not fn.startswith("_"):
                    refs[(m.group(1), fn)] = c.name
        for m in pat_dotted.finditer(ex):
            refs[(m.group(1), m.group(2))] = c.name
        for m in pat_mind.finditer(ex):
            mind_refs[m.group(1)] = c.name

    broken, no_doc, terse = [], [], []
    # 1. module-level function references
    for (mod, fn), home in sorted(refs.items()):
        try:
            M = importlib.import_module(mod)
        except Exception as e:                                  # module won't import at all
            broken.append((mod, fn, home, "module import failed: %s" % type(e).__name__))
            continue
        obj = getattr(M, fn, None)
        if obj is None:
            broken.append((mod, fn, home, "name not found in module"))
            continue
        summary = ((getattr(obj, "__doc__", None) or "").strip().split("\n") or [""])[0].strip()
        if not summary:
            no_doc.append((mod, fn, home))
        elif len(_summary_words(summary)) < MIN_SUMMARY_WORDS:
            terse.append((mod, fn, home, summary))
    # 2. mind.method references -- must name a real, public UnifiedMind method
    import holographic.misc.holographic_skills as sk
    methods = sk.mind_methods()
    for fn, home in sorted(mind_refs.items()):
        if fn not in methods:
            broken.append(("mind", fn, home, "not a UnifiedMind method"))
    return {"broken": broken, "no_doc": no_doc, "terse": terse, "checked": len(refs) + len(mind_refs)}


def report(strict=False):
    """Print a human report; return the count of hard gaps (CRITICAL + TERSE) so callers/CI can gate on it."""
    a = audit()
    import holographic.misc.holographic_skills as sk
    total = len(sk.mind_methods())

    print("skill-lint over %d public UnifiedMind methods\n" % total)

    print("CRITICAL -- no docstring (agent gets only the signature): %d" % len(a["critical"]))
    for n in a["critical"]:
        print("   %s" % n)

    print("\nTERSE -- summary under %d words: %d" % (MIN_SUMMARY_WORDS, len(a["terse"])))
    for n, s in a["terse"]:
        print("   %-28s %r" % (n, s))

    if strict:
        print("\nNO_RETURN (note) -- first line doesn't hint at a return value: %d" % len(a["no_return"]))
        for n in a["no_return"]:
            print("   %s" % n)

    gaps = len(a["critical"]) + len(a["terse"])
    print("\n%d hard gap(s) (CRITICAL + TERSE)%s." % (gaps, "" if not strict else
          "; %d NO_RETURN notes" % len(a["no_return"])))

    # -- the module-level functions an agent reaches via a home's `example` --
    h = audit_home_examples()
    print("\n-- curated-home example references (%d checked) --" % h["checked"])
    print("BROKEN -- example names a function/module that doesn't resolve: %d" % len(h["broken"]))
    for mod, fn, home, why in h["broken"]:
        print("   %s.%s   [%s]  <- %s" % (mod, fn, home, why))
    print("NO DOC -- referenced function has no docstring: %d" % len(h["no_doc"]))
    for mod, fn, home in h["no_doc"]:
        print("   %s.%s   [%s]" % (mod, fn, home))
    print("TERSE  -- referenced function's summary under %d words: %d" % (MIN_SUMMARY_WORDS, len(h["terse"])))
    for mod, fn, home, s in h["terse"]:
        print("   %s.%s   %r" % (mod, fn, s))

    example_gaps = len(h["broken"]) + len(h["no_doc"]) + len(h["terse"])
    total = gaps + example_gaps
    print("\nTOTAL: %d invocation gap(s) -- %d method (CRITICAL+TERSE) + %d example (BROKEN+NODOC+TERSE)."
          % (total, gaps, example_gaps))
    return total


if __name__ == "__main__":
    sys.exit(report(strict="--strict" in sys.argv))

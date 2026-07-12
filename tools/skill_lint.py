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

WHAT IT CHECKS (over catalog search aliases -- the words find_capability matches on)
  * INERT (a hard gap) -- an alias that tokenizes to ZERO content words (all stopwords, or pure punctuation like
    "o(n^2)") can never be matched, so it is dead weight that looks correct in the source. Reword with content words.
  * REDUNDANT (--strict note only) -- every surviving token already appears in the entry name, so the alias adds
    no new matchable word. Advisory, never a failure: a lone name-echo is free, and a phrase whose distinctive word
    was stopworded away is a tokenizer casualty worth keeping.

The check is deterministic and stdlib-only. Run it after wiring new faculties; fix anything CRITICAL/TERSE.

    python tools/skill_lint.py            # human-readable report; exit code = number of CRITICAL+TERSE gaps
    python tools/skill_lint.py --strict   # also list the NO_RETURN notes
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MIN_SUMMARY_WORDS = 5              # a one-line summary shorter than this tells an agent almost nothing

# T3: a catalog `does` field is a SEARCH-INDEX entry, not a NOTES entry. Measured cause of two rev.9 routing
# failures: an essay-length `does` is a "token sponge" -- it carries enough incidental words to out-rank the
# entry that is actually the best match for a short query, purely by volume. So a `does` over this many chars is
# flagged (a WARNING, not a hard gate -- length is a judgement call, not a defect like an inert alias). The real
# content of a long entry belongs in the module's own docstring, which docgen.py already publishes. The threshold
# (600) is ~5x the median entry (113) -- comfortably clear of a normal 2-4 sentence index card.
MAX_DOES_CHARS = 600

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


def audit_aliases():
    """Flag catalog search aliases that cannot do their job. Two classes, deliberately weighted differently:

      * INERT (a hard gap) -- the alias tokenizes to ZERO content words, so `find_capability` can never match it.
        Measured cause: it is entirely stopwords ("what can you do", "as of") or punctuation ("o(n^2)"). This is
        the little sibling of the 827-inert-aliases tokenization bug: an alias written in good faith that silently
        matches nothing. It is a defect with no judgement call, so it gates.
      * REDUNDANT (a note only) -- every surviving token already appears in the entry NAME, so the alias adds no
        new matchable word. This is ADVISORY, not a gap: a single word that merely echoes a name word is free
        (it cannot hurt ranking), and a multi-word phrase whose distinctive word was stopworded away is a tokenizer
        casualty worth KEEPING, not deleting -- it still expresses intent a future tokenizer could honour. So we
        surface these for a human to read (the backlog's rule: do not auto-fix aliases), and never fail on them.

    Returns {"inert": [(entry, alias)], "redundant": [(entry, alias)]}."""
    import re
    from holographic.caching_and_storage.holographic_catalog import _tokens
    import holographic.misc.holographic_skills as sk
    _word = re.compile(r"[a-z0-9]+")
    cat = sk._catalog()
    inert, redundant = [], []
    for name, cap in cat._by_name.items():
        name_toks = set(_tokens(name))
        for a in cap.aliases:
            toks = set(_tokens(a))
            if not toks:
                inert.append((name, a))
                continue
            # redundant = nothing NEW survives, and nothing DISTINCTIVE was lost to stopwording (a lost distinctive
            # word means the alias intends to match it -- a casualty, not dead weight, so we do NOT flag it).
            if not (toks - name_toks):
                raw = set(_word.findall(a.lower()))
                distinctive_lost = {w for w in (raw - name_toks - toks) if len(w) > 2}
                if not distinctive_lost:
                    redundant.append((name, a))
    return {"inert": inert, "redundant": redundant}


def audit_does_length():
    """T3: flag catalog `does` fields long enough to act as token sponges (see MAX_DOES_CHARS). Returns
    {"over": [(name, chars)], "budget": <the shrink-only allowlist>, "regressions": [names over budget]}.

    Like the duplicate and no-selftest budgets, `_DOES_BUDGET` MAY SHRINK AND MUST NEVER GROW: it is the set of
    entries that were already over-length when T3 landed, recorded so the check can gate on NEW offenders without
    forcing a 58-entry rewrite in one sitting (that would violate the alias-perturbation rule -- rewriting a
    search-index entry can shift what OTHER queries match, so each must be redone one at a time with the routing
    cluster re-run). Trimming a budgeted entry below the threshold, or deleting its line here after moving its
    prose to the module docstring, is the only correct way to change this set."""
    import holographic.misc.holographic_skills as sk
    cat = sk._catalog()
    over = sorted(((len(cap.does), n) for n, cap in cat._by_name.items() if len(cap.does) > MAX_DOES_CHARS),
                  reverse=True)
    over_names = {n for _, n in over}
    regressions = sorted(over_names - _DOES_BUDGET)          # NEW long entries -- these gate
    return {"over": [(n, ln) for ln, n in over], "regressions": regressions,
            "budget_stale": sorted(_DOES_BUDGET - over_names)}   # budgeted entries since trimmed -> remove their line


# The 58 entries already over MAX_DOES_CHARS when T3 landed. SHRINK-ONLY. See audit_does_length.__doc__.
_DOES_BUDGET = {
    'Asset relocation / relink (external files)',
    'Bake a function into one vector (texture unit)',
    'Bake an N-D function into one vector (n-D texture unit)',
    'Blend M shader variants into one transfer',
    'Canonical affine recovery (Fourier-Mellin + refine)',
    'Canonical element + delta chain (instancing, generalised)',
    'Cloud stack (closed-form shadow rays)',
    'Coarse-first refine (re-enable)',
    'Code / file editing (agentic)',
    'Code as canonical shape + name delta (exact, not a codec)',
    'Cold storage (compress inactive data)',
    'Compressed-domain compute (never touch the decompressed field)',
    'Conformal UV unwrap (LSCM) + the metric that sees folds',
    'Cross field (smoothest 4-RoSy) + the bar that was vacuous',
    'Dependency-keyed cache (key on what the operator reads)',
    'Describe a scene (scene from description, semantic)',
    'Detrend before you bake (non-periodic functions)',
    'Dialect emitters (WGSL / C / JS / Zig from the Python kernel)',
    'Dictionary + taxonomy (vendored)',
    'Encyclopedia (relational knowledge)',
    'Equivariance table (the cache policy, measured)',
    'Fat-margin cache (for a query that drifts)',
    'File map ingest (folder / zip -> queryable)',
    'Fill the gaps in a field (inpaint / impute)',
    'Frame-to-frame motion by one unbind (reprojection velocity)',
    'Frequency-lifted (Gabor) splats',
    'Gather N lookups in one dot product (superposed gather)',
    'Graph-colour waves (lock-free deterministic parallelism)',
    'Hierarchical superposition (cleanup between levels)',
    'Import artist file formats (OBJ/glTF/textures/volume)',
    'Information-rate rendering (shade the news, reproject the rest)',
    'Islands + sleep (solve only what is still moving)',
    'Learned chunk codebook (iterated pair promotion)',
    'Learned navigator (adaptive search budget)',
    'Lossless set-packing for image families',
    'Material library (render + physical)',
    'Memoize a pure function (the purity gate is the point)',
    'Message bus + agent (LLM) bridge',
    'Modal jump solver (skip the substeps)',
    'Multi-way tensor compression (Tucker / TT)',
    'Partition-invariant sums (same answer at any bucket count)',
    'Physics event codec (a trace as base + interruptions)',
    'Points to mesh (isosurface / surface reconstruction)',
    'Post-effect kernel fusion (N linear passes, one FFT pair)',
    'Progressive LOD stream (rank-ordered TT cores)',
    'Purity & effect analysis (the gate a cache needs)',
    'Realtime session (draft frames, refine pass, multi-format payload)',
    "Recursive factoring (past the resonator's cliff)",
    'Run an allowlisted external command',
    'Scatter / gather (any rank, any kernel, exact on demand)',
    'Soft constraints (hertz + damping ratio)',
    'Subdivision limit surface (closed form)',
    "The machine model (leCore's hardware units + memory tiers)",
    'The projective ceiling (where the transform tower stops)',
    "The scene's own SDF, emitted (brain/muscle, realised)",
    'The transform tower (which layer of the affine group)',
    'Transform bank (a prebuilt map of hypervector transforms)',
    'Tunnelling & CCD (speculative margins, conservative advancement)',
}


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

    # -- catalog search aliases: inert ones are a hard gap (match nothing), redundant ones a note (T2) --
    al = audit_aliases()
    print("\n-- catalog search aliases --")
    print("INERT -- alias tokenizes to zero content words, so find_capability can NEVER match it: %d" % len(al["inert"]))
    for entry, alias in al["inert"]:
        print("   %-45s %r" % (entry[:45], alias))
    if strict:
        print("REDUNDANT (note) -- every surviving token already in the entry name, adds no new match: %d"
              % len(al["redundant"]))
        for entry, alias in al["redundant"]:
            print("   %-45s %r" % (entry[:45], alias))
    else:
        print("REDUNDANT (note, --strict to list): %d" % len(al["redundant"]))

    # -- T3: catalog `does` fields long enough to act as token sponges (a WARNING tier, not a hard gate) --
    dl = audit_does_length()
    print("\n-- catalog does-field length (T3) --")
    print("OVER %d chars: %d total (%d budgeted, %d NEW regression(s))"
          % (MAX_DOES_CHARS, len(dl["over"]), len(dl["over"]) - len(dl["regressions"]), len(dl["regressions"])))
    for name in dl["regressions"]:
        chars = dict((n, c) for n, c in dl["over"])[name]
        print("   REGRESSION  %-52s %d chars -- shorten, or move prose to the module docstring" % (name[:52], chars))
    if dl["budget_stale"]:
        print("   %d budgeted entr(y/ies) now under threshold -- delete their _DOES_BUDGET line(s): %s"
              % (len(dl["budget_stale"]), ", ".join(s[:30] for s in dl["budget_stale"][:5])))

    total = gaps + example_gaps + len(al["inert"]) + len(dl["regressions"])
    print("\nTOTAL: %d invocation gap(s) -- %d method (CRITICAL+TERSE) + %d example (BROKEN+NODOC+TERSE) + "
          "%d inert alias(es) + %d does-length regression(s).%s"
          % (total, gaps, example_gaps, len(al["inert"]), len(dl["regressions"]),
          "" if not strict else "  (%d redundant-alias notes)" % len(al["redundant"])))
    return total


if __name__ == "__main__":
    sys.exit(report(strict="--strict" in sys.argv))

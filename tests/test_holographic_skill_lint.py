"""Tests for tools/skill_lint.py -- the docstring/invocation-quality linter over UnifiedMind faculties."""
import importlib.util
import os

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "skill_lint.py")


def _lint():
    spec = importlib.util.spec_from_file_location("skill_lint", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_critical_or_terse_gaps():
    """Every public UnifiedMind method has a non-trivial docstring summary an agent can read."""
    a = _lint().audit()
    assert a["critical"] == [], a["critical"]
    assert a["terse"] == [], a["terse"]


def test_report_returns_zero():
    assert _lint().report() == 0


def test_the_core_faculties_are_documented():
    """The four faculties this linter first caught (learn/next_symbol/reinforce/describe) now describe well."""
    import holographic.misc.holographic_skills as sk
    ms = sk.mind_methods()
    for n in ("learn", "next_symbol", "reinforce", "describe"):
        assert len(ms[n]["summary"].split()) >= 5, (n, ms[n]["summary"])


def test_home_example_references_resolve_and_document():
    """Every module-level function named in a curated home's `example` (what an agent copies) exists, imports, and
    has a usable docstring -- no BROKEN references, no missing/terse docs."""
    h = _lint().audit_home_examples()
    assert h["broken"] == [], "broken example references: %s" % h["broken"]
    assert h["no_doc"] == [], "referenced functions with no docstring: %s" % h["no_doc"]
    assert h["terse"] == [], "referenced functions with a thin docstring: %s" % h["terse"]
    assert h["checked"] > 50


def test_no_inert_aliases():
    """Every catalog search alias tokenizes to at least one content word -- an alias that reduces to zero tokens
    (all stopwords, or pure punctuation like 'o(n^2)') can NEVER be matched by find_capability, the little sibling
    of the 827-inert-aliases tokenization bug. This gates: a new inert alias fails CI here with its name."""
    al = _lint().audit_aliases()
    assert al["inert"] == [], "aliases that match nothing (reword with content words): %s" % al["inert"]


def test_inert_alias_detector_actually_bites():
    """Prove the detector is not vacuously green: a synthetic all-stopword alias must be reported inert, and a
    real content-word alias must not. A lint that cannot fail is worse than no lint."""
    from holographic.caching_and_storage.holographic_catalog import _tokens
    assert _tokens("what can you do") == []          # the exact class the detector must catch
    assert _tokens("point in time") != []            # a reworded alias survives -- not flagged


def test_no_new_does_length_regressions():
    """T3: no NEW catalog `does` field over MAX_DOES_CHARS beyond the shrink-only budget of ones that were already
    long when the check landed. An essay-length `does` is a token sponge (it out-ranks better matches by word
    volume -- the measured cause of two rev.9 routing failures). This gates: ship a new over-length entry and CI
    fails here with its name. Trim a budgeted one below threshold and it moves to `budget_stale` -- delete its
    _DOES_BUDGET line."""
    dl = _lint().audit_does_length()
    assert dl["regressions"] == [], ("new over-length does field(s) -- shorten, or move prose to the module "
                                     "docstring: %s" % dl["regressions"])


def test_does_length_detector_actually_bites():
    """Prove the T3 gate is not vacuously green: a fresh over-length, un-budgeted entry must be reported as a
    regression, and trimming a budgeted entry must surface it as stale. Uses a frozen catalog so the mutation
    reaches the audit (the builder rebuilds fresh each call by design)."""
    lint = _lint()
    import holographic.misc.holographic_skills as sk
    cat = sk._catalog()
    saved = sk._catalog
    try:
        sk._catalog = lambda: cat                     # freeze so the mutation persists into audit_does_length
        name = next(n for n, c in cat._by_name.items()
                    if len(c.does) < 200 and n not in lint._DOES_BUDGET)
        orig = cat._by_name[name].does
        cat._by_name[name].does = "x " * 400          # 800 chars, over threshold, not budgeted
        assert name in lint.audit_does_length()["regressions"]     # the gate must catch it
        cat._by_name[name].does = orig
        budgeted = sorted(lint._DOES_BUDGET)[0]
        cat._by_name[budgeted].does = "short now"
        assert budgeted in lint.audit_does_length()["budget_stale"]   # a trimmed budgeted entry is flagged
    finally:
        sk._catalog = saved

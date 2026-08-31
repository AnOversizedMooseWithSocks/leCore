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


def test_catalog_examples_do_not_pass_unknown_keywords():
    """**An example that passes a kwarg the faculty does not accept cannot run.**

    skill_lint executes examples, so a crashing one is normally caught -- but only
    where the capability's own example is the thing being run. Three capabilities
    documented `m.attach_llm(fn, cache=True, batch_fn=...)` while attach_llm takes
    (llm, name): the cache and batching live on MeteredLLM, one layer down. Every
    one of those examples was copy-pasteable and every one raised TypeError.
    PARSED, NOT GREPPED: a regex over "word=" matches local assignments in the
    example body and reported 541 false hits. Reading the AST and looking only at
    keywords on the call to THIS capability's own method gives the real number."""
    import ast
    import inspect

    import lecore

    m = lecore.UnifiedMind(dim=64, seed=0)
    bad = []
    for cap in m._capability_catalog().all():
        meth = getattr(cap, "method", None)
        ex = (getattr(cap, "example", "") or "").strip()
        if not meth or not ex or not hasattr(m, meth):
            continue
        try:
            params = set(inspect.signature(getattr(m, meth)).parameters)
        except (TypeError, ValueError):
            continue
        # DETECT **kwargs BY KIND, NOT BY NAME. A hardcoded name list
        # ({"kw","kwargs","params","cell"}) missed texture_op(op, **inputs) and
        # reported it as a broken example -- the name of a VAR_KEYWORD parameter
        # is arbitrary, and guessing at it is how a lint invents work.
        sigp = inspect.signature(getattr(m, meth)).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sigp.values()):
            continue                      # **kwargs accepts anything
        try:
            tree = ast.parse(ex)
        except SyntaxError:
            continue                      # a snippet, not a program; skill_lint owns that
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == meth):
                unknown = [k.arg for k in node.keywords
                           if k.arg and k.arg not in params]
                if unknown:
                    bad.append("%s: example passes %r, signature is %s"
                               % (meth, unknown, sorted(params)))
    assert not bad, "\n".join(sorted(set(bad)))


def test_examples_do_not_call_methods_that_do_not_exist():
    """**An example that calls m.<missing> is a promise with nothing behind it.**

    skill_lint checks that the example's SYMBOLS resolve -- and the capability NAME
    resolving is not the same as the METHOD INSIDE the example existing. `prf_rank`
    carried a full description with MEASURED BEIR numbers (NFCorpus nDCG@10
    0.3371 -> 0.3442) and `m.prf_rank(...)` in its example, and no prf_rank or
    prf_expand exists anywhere in the tree.
    THAT IS THE WORST SHAPE A CATALOG ENTRY CAN HAVE: it is discoverable, it is
    documented, it cites results, and it cannot be called. A missing capability is
    a gap; a documented missing capability is a false claim.
    Known absences are listed with a reason -- add to KNOWN, do not delete the
    entry, because the description is the record of what was measured."""
    import ast

    import lecore

    #: Methods named in examples that genuinely do not exist yet. Each needs a
    #: reason, and each is a debt: either the method lands or the entry says so.
    KNOWN = {
        "_llm",                    # private attribute, set by attach_llm at runtime
        "llm_prefix_route",        # documented alongside _llm; same seam
        "predict_streaming_ms",    # roofline predictor, named in prose not wired
        "stocked_part_library",    # part-library entry, not a faculty
        "prf_rank",                # PSEUDO-RELEVANCE FEEDBACK -- the entry cites
                                   # measured BEIR gains and NO IMPLEMENTATION
                                   # EXISTS. Kept so the measurement is not lost.
    }
    m = lecore.UnifiedMind(dim=64, seed=0)
    ghosts = set()
    for cap in m._capability_catalog().all():
        ex = (getattr(cap, "example", "") or "")
        if not ex.startswith("import "):
            continue
        try:
            tree = ast.parse(ex)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and getattr(node.func.value, "id", None) in ("m", "mind")
                    and not hasattr(m, node.func.attr)
                    and node.func.attr not in KNOWN):
                ghosts.add("%s (in %r)" % (node.func.attr, cap.name[:44]))
    assert not ghosts, "\n".join(sorted(ghosts))


def test_examples_bind_the_mind_they_use():
    """**A copy-pasteable example that raises NameError on line one is not an example.**

    76 examples opened with `import numpy as np; ...` and then called
    `mind.<something>` without ever building a mind. skill_lint passed all of them:
    every symbol it checks resolves, and `mind` is not one of them."""
    import re

    import lecore

    m = lecore.UnifiedMind(dim=64, seed=0)
    bad = [cap.name for cap in m._capability_catalog().all()
           if (getattr(cap, "example", "") or "").startswith("import ")
           and re.search(r"\bmind\s*\.", cap.example)
           and not re.search(r"\bmind\s*=", cap.example)]
    assert not bad, "examples using an unbound `mind`: %r" % sorted(bad)[:12]


#: Examples that reference a name they never bind. These are SKETCHES, not
#: programs -- `mind.apply(other, v)` shows the CALL SHAPE without inventing a
#: fixture for `other`. That is a legitimate kind of example, and the budget
#: exists so the count cannot GROW silently while nobody looks.
UNBOUND_EXAMPLE_BUDGET = 97


def test_the_unbound_example_budget_does_not_grow():
    """**A budget that may shrink and must never grow.**

    Running the catalog's 418 full-program examples end to end found 100 that
    reference an undefined name. Twenty-eight were mechanical (used `np.` without
    importing numpy) and are fixed. THE REST NEED A HUMAN TO INVENT A FIXTURE, and
    inventing one silently is worse than the sketch: it makes an untested snippet
    LOOK like a verified program.
    WHY THIS IS A BUDGET AND NOT A GATE: an example is allowed to be a sketch. What
    is not allowed is the number quietly climbing, which is what happened here --
    the first 200 examples had 3 failures and the tail had 58, so ordering hid the
    problem from every sampled check that came before."""
    import ast
    import builtins

    import lecore

    m = lecore.UnifiedMind(dim=64, seed=0)
    built = set(dir(builtins))
    bad = []
    for cap in m._capability_catalog().all():
        ex = (getattr(cap, "example", "") or "")
        if not ex.startswith("import "):
            continue
        try:
            tree = ast.parse(ex)
        except SyntaxError:
            continue
        bound = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    bound.add((a.asname or a.name).split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.Lambda)):
                for a in getattr(node.args, "args", []):
                    bound.add(a.arg)
            elif isinstance(node, ast.comprehension):
                for t in ast.walk(node.target):
                    if isinstance(t, ast.Name):
                        bound.add(t.id)
        if any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
               and n.id not in bound and n.id not in built
               for n in ast.walk(tree)):
            bad.append(cap.method or cap.name)
    assert len(bad) <= UNBOUND_EXAMPLE_BUDGET, (
        "unbound-name examples rose to %d (budget %d) -- bind the name or make "
        "the example a sketch on purpose, but do not let this climb: %r"
        % (len(bad), UNBOUND_EXAMPLE_BUDGET, sorted(bad)[:8]))


def test_booting_twice_does_not_re_teach_the_doctrine():
    """**boot() is called more than once, so registering doctrine must be idempotent.**

    _remember APPENDS unconditionally, and boot() taught the same 14 doctrine facts
    every time:
        boot 1 -> taught 14 | boot 3 -> taught 42 | boot 5 -> taught 70
    A long-running service that re-boots grew its taught store WITHOUT BOUND, 14
    identical rows at a time.
    WHY NOTHING CAUGHT IT: recall was unaffected -- measured, the same answer after
    one boot and after ten -- so it was pure bloat and no wrong answer ever
    appeared. A DEFECT THAT NEVER PRODUCES A WRONG ANSWER IS INVISIBLE TO EVERY
    TEST THAT ONLY CHECKS ANSWERS. Found by calling a faculty, churning others, and
    calling it again."""
    import warnings

    import lecore

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = lecore.UnifiedMind(dim=128, seed=0)
        counts = [m.boot()["inventory"]["taught"] for _ in range(4)]
    assert len(set(counts)) == 1, (
        "the taught store grew across repeated boots: %r -- doctrine is being "
        "re-registered" % counts)
    assert counts[0] > 0, counts


def test_save_load_round_trips_do_not_grow_the_partition(tmp_path):
    """**A partition saved and re-mounted N times must not grow N times.**

    The previous sweep made register_doctrine idempotent with a marker on the
    mind. THAT FIXED ONE PATH AND NOT THE OTHER: autoboot MOUNTS a partition that
    already contains doctrine, and boot()'s order is POST -> mount -> doctrine, so
    a fresh mind has no marker, the facts arrive from disk, and boot teaches all
    14 again on top. Measured across save/reboot cycles before the fix:
        taught 30 -> 100 -> 240 -> 520, every row a duplicate of the same 14.
    THE SAME BUG ONE LAYER OUT, and the in-memory fix could not see it because
    the second path never touched the marker. The check now asks the STORE.
    This test uses the round trip rather than the marker, so it holds whatever the
    mechanism becomes."""
    import json
    import warnings
    import zipfile

    import lecore

    root = str(tmp_path / "part")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = lecore.autoboot(partition=root, llm=None)
        m.teach("a probe fact", "the answer")
        counts = []
        for _ in range(4):
            m.learning_save(root)
            m = lecore.autoboot(partition=root, llm=None)
            z = zipfile.ZipFile(root + "/learning/state.lecore")
            man = json.loads(z.read("manifest.json"))
            meta = [s for s in man["sections"]
                    if s["kind"].endswith("taught")][0]["meta"]
            counts.append(len(meta["texts"]))
        assert m.ask("a probe fact")["tier"] == "T0", "the probe stopped recalling"
    assert len(set(counts)) == 1, (
        "the taught store grew across save/reboot cycles: %r -- doctrine is "
        "being re-taught on top of a partition that already has it" % counts)


def test_learning_compact_removes_duplicates_without_losing_recall(tmp_path):
    """**Repair for a partition that grew, and the repair must not cost an answer.**

    Two sweeps fixed doctrine duplication -- once in memory, once through the mount.
    NEITHER RECLAIMS WHAT WAS ALREADY WRITTEN: my own audit partition held 381 taught
    rows of which 73 were distinct, the same doctrine facts twenty-three times each.
    A bug whose cost outlives the fix needs a repair path, and there was none.
    The dangerous failure here is not leaving a duplicate -- it is DROPPING A REAL
    ANSWER, so this asserts recall survives, not just that the count fell."""
    import warnings

    import lecore

    root = str(tmp_path / "part")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = lecore.autoboot(partition=root, llm=None)
        m.teach("a unique probe fact", "the probe answer")
        lad = m.zoo["ladder"]
        # simulate a partition that lived through the bug
        lad.taught_log = list(lad.taught_log) * 4
        before = len(lad.taught_log)
        dry = m.learning_compact(dry_run=True)
        assert len(lad.taught_log) == before, "dry_run mutated the store"
        rep = m.learning_compact()
        assert rep["removed"] > 0, rep
        assert rep["after"] == rep["distinct"] == dry["distinct"], (rep, dry)
        assert m.ask("a unique probe fact")["tier"] == "T0", (
            "compaction dropped a real answer -- the one thing it must never do")

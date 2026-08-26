"""Tests for holographic_catalog -- the capability catalog (C1: 'search before you build')."""
from holographic.caching_and_storage.holographic_catalog import Catalog, Capability, default_catalog, seed_from_mind, _tokens


def test_register_and_find():
    c = Catalog()
    c.register_capability("Widget", "does a special widget job", example="widget()", aliases=("gadget",))
    hits = c.find_capability("I need a widget job")
    assert hits and hits[0].name == "Widget"
    assert c.find_capability("gadget thing")[0].name == "Widget"    # alias match


def test_headline_query_finds_search_home():
    c = default_catalog()
    assert "Index" in c.find_capability("search a big pile of vectors")[0].name


def test_native_flag_carried():
    c = default_catalog()
    assert c.get("Index (search)").native is True
    assert c.get("holographic_catalog").native is False             # the catalog hops to Python


def test_seeded_with_named_homes():
    c = default_catalog()
    names = " ".join(x.name for x in c.all()).lower()
    for home in ("index", "cache", "field"):                        # the three big consolidation homes are present
        assert home in names


def test_deterministic_and_stable_ranking():
    c = default_catalog()
    a = [h.name for h in c.find_capability("bake a slow factor and look it up", k=3)]
    b = [h.name for h in c.find_capability("bake a slow factor and look it up", k=3)]
    assert a == b                                                   # same query -> same order, run to run


def test_no_match_returns_empty():
    c = Catalog()
    c.register_capability("X", "handles quaternions")
    assert c.find_capability("zzzzz nonsense qqqqq") == []


def test_tokeniser_drops_stopwords():
    toks = _tokens("how do I search a big pile of vectors")
    assert "search" in toks and "vectors" in toks and "how" not in toks and "do" not in toks


def test_seed_from_mind_registers_faculties():
    class FakeMind:
        def render_scene(self):
            "Render a scene."
        def _private(self):
            "hidden"
    c = seed_from_mind(Catalog(), FakeMind())
    assert c.get("render_scene") is not None and c.get("_private") is None


def test_every_engine_module_has_a_docstring():
    """Guard against buried functionality: EVERY engine module must have a docstring, or the catalog's
    find_capability can't surface it. (AST-only, never imports.)"""
    import ast, glob, os
    import holographic.caching_and_storage.holographic_catalog as holographic_catalog
    # the catalog module lives in holographic/caching_and_storage/; the package root is one level up, and it
    # holds every family subpackage. Walk the WHOLE tree, not just this one directory.
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(holographic_catalog.__file__)))
    missing = []
    for path in sorted(glob.glob(os.path.join(pkg_root, "**", "holographic_*.py"), recursive=True)):
        if os.path.basename(path).startswith("test_"):
            continue
        try:
            tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
        except SyntaxError:
            continue
        if not (ast.get_docstring(tree) or "").strip():
            missing.append(os.path.basename(path))
    assert not missing, "modules with no docstring (undiscoverable by the catalog): %s" % missing


def test_seed_from_modules_makes_all_domains_findable():
    from holographic.caching_and_storage.holographic_catalog import default_catalog, seed_from_modules
    cat = seed_from_modules(default_catalog())
    assert len(cat) > 300                                        # homes + every module registered
    for probe, expect in [("thin film iridescence on a bubble", "thinfilm"),
                          ("simulate smoke and fire", "simulation"),
                          ("build a mesh with extrude and bevel", "mesh"),
                          ("gaussian splat scene", "splat"),
                          ("oil and water separating mixture model", "mixture"),
                          ("procedural noise texture", "texture")]:
        names = " ".join(h.name for h in cat.find_capability(probe, k=3)).lower()
        assert expect in names, (probe, names)


def test_domain_homes_registered():
    from holographic.caching_and_storage.holographic_catalog import default_catalog
    homes = " ".join(x.name for x in default_catalog().all())
    for dom in ("Lighting", "Geometry", "Texture", "Simulation", "Physics & chemistry", "Denoise", "Query"):
        assert dom in homes, dom


def test_exact_alias_phrase_ranks_into_top_k():
    # Regression: a query that IS an exact alias of a capability must surface that capability, even when siblings
    # scatter the same content words across their prose and tie on the word-overlap score. Before the exact-alias
    # bonus, ascii_view (exact alias 'render image to terminal') tied at 3.0 with ascii_animate/field/sdf and lost
    # the alphabetical tie-break, falling to position 4 and off the default k=3 list. The bonus makes an exact
    # phrase match the strongest signal, so it wins decisively. Additive: only exact matches are boosted.
    from holographic.caching_and_storage.holographic_catalog import Catalog
    cat = Catalog()
    # two entries that both mention the same words in their does; only one has the exact alias.
    cat.register_capability("sibling_a", "render an image to the terminal as animation frames", aliases=("animate",))
    cat.register_capability("the_target", "render an image somewhere", aliases=("render image to terminal",))
    cat.register_capability("sibling_b", "render an image to the terminal from a field", aliases=("field",))
    cat.register_capability("sibling_c", "render an image to the terminal from an sdf", aliases=("sdf",))
    hits = cat.find_capability("render image to terminal", k=3)
    assert hits[0].name == "the_target", [h.name for h in hits]     # exact-alias match wins position 0

    # find_scored agrees (same scoring), and the exact match outscores the tied siblings by the bonus margin
    scored = dict((c.name, s) for c, s in cat.find_scored("render image to terminal", k=6))
    assert scored["the_target"] >= max(v for k, v in scored.items() if k != "the_target") + 4.0

    # a NON-exact query still ranks by overlap only (no bonus fires) -- the bonus is surgical, not a blanket boost
    hits2 = cat.find_capability("render an image", k=4)
    assert "the_target" in [h.name for h in hits2] or len(hits2) == 4   # still findable, not specially boosted


# ---------------------------------------------------------------------------
# J-3D-25: the catalog is split across parts. These pin what the split promised.
# ---------------------------------------------------------------------------

def test_every_part_is_registered_and_in_order():
    """A part that exists on disk and is not called by default_catalog() registers NOTHING -- its
    capabilities silently cease to exist, which is the exact failure mode this repo calls a gap. And the
    ORDER is contractual: find_capability ranks by score and ties break by registration order, so calling
    the parts in a different sequence would quietly move search results."""
    import re
    from pathlib import Path
    from holographic.caching_and_storage import holographic_catalog as CAT

    here = Path(CAT.__file__).parent
    on_disk = sorted(p.stem for p in here.glob("holographic_catalog_p*.py"))
    src = Path(CAT.__file__).read_text()
    # each part exports register_pNN, not a shared `register`: six modules with the same public name is a
    # name-collision the budget must not grow to absorb (tools/name_collisions), and distinct names make
    # a traceback name its own part. The pattern below pins THAT contract too -- part pNN must be entered
    # through register_pNN, so a copy-paste that calls the wrong part's entry point fails here.
    called = [mod for mod, tag in re.findall(r"(holographic_catalog_p(\d+))\.register_p\2\(c\)", src)]
    assert called == sorted(called), "the parts must be called in sorted order"
    assert called == on_disk, "part files and register() calls disagree: %s vs %s" % (on_disk, called)


def test_no_part_exceeds_the_agent_read_cap():
    """The whole reason for the split. The file that makes capabilities discoverable must stay openable by
    the agents doing the discovering -- holographic_catalog.py had reached 81% of 1 MB before this."""
    from pathlib import Path
    from holographic.caching_and_storage import holographic_catalog as CAT

    here = Path(CAT.__file__).parent
    for path in [Path(CAT.__file__)] + sorted(here.glob("holographic_catalog_p*.py")):
        size = path.stat().st_size
        assert size < 800_000, "%s is %d bytes -- at 80%% of the 1 MB cap, split again" % (path.name, size)


def test_the_split_preserved_every_capability():
    """The split's only promise was that it changes NOTHING. 522 capabilities were registered before it.

    THIS TEST WAS WRONG WHEN FIRST WRITTEN and is kept as a lesson rather than quietly rewritten: it
    asserted `== 522` exactly, and the very next item that registered a capability failed it. An exact count
    is a CHANGE-DETECTOR, not a contract -- it fires on the intended, routine act of adding a capability,
    which trains people to edit the number instead of reading the failure. The real contract is that the
    split never LOSES one and never registers the same name twice; a floor plus a duplicate check says that
    without punishing normal work."""
    from holographic.caching_and_storage.holographic_catalog import default_catalog
    caps = default_catalog().all()
    assert len(caps) >= 522, "capability count FELL to %d -- the split baseline was 522, so a part is " \
                             "no longer registering" % len(caps)
    names = [c.name for c in caps]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, "the same capability name is registered twice: %s" % dupes


def test_every_part_has_a_real_selftest():
    """The split shipped six modules with no `__main__` and no `_selftest`, and the selftest-budget test
    caught it on the next run -- correctly, because a module that asserts nothing is a false green.

    Budgeting them would have silenced the alarm without testing anything. The parts have a real, cheap
    contract, so they assert it. This pins that a future part is not added to the budget instead."""
    import pathlib
    from holographic.caching_and_storage import holographic_catalog as CAT

    here = pathlib.Path(CAT.__file__).parent
    parts = sorted(here.glob("holographic_catalog_p*.py"))
    assert parts, "no catalog parts found -- the split is gone?"
    for path in parts:
        text = path.read_text()
        assert "def _selftest():" in text and '__name__ == "__main__"' in text, \
            "%s has no runnable selftest -- do NOT add it to _NO_SELFTEST_BUDGET, give it the three-line " \
            "contract the others have" % path.name


def test_the_field_query_ranking_stays_fixed():
    """J-3D-26, CLOSED. This slot used to hold a TRIPWIRE that asserted the BROKEN state: 'represent a
    density volume over space' did not surface the Field capability, the module _selftest asserted that it
    did, and the selftest had been failing unnoticed. The tripwire existed so that whoever fixed the ranking
    would be told, rather than the regression being quietly relaxed into noise. It did its job.

    THE FIX WAS ADDITIVE, which is the part worth keeping: Field carried only single-word aliases ('field',
    'grid', 'volume', ...), and single words lose to descriptively-titled siblings as a catalog grows -- so
    the PHRASE a person actually types was added, not a threshold lowered and not a neighbour demoted. The
    assertion now lives in holographic_catalog._selftest() where it started; this pins it from the suite too,
    because a selftest that only runs under `python -m` is exactly how the original rot went unseen."""
    from holographic.caching_and_storage.holographic_catalog import default_catalog
    hits = [h.name for h in default_catalog().find_capability("represent a density volume over space")]
    assert any("Field" in n for n in hits[:3]), \
        "the Field ranking regressed again (top-3: %r) -- strengthen Field's aliases with the phrasing a " \
        "user types; do NOT relax this or demote the neighbour that outranked it" % hits[:3]


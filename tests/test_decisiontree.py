"""tests/test_decisiontree.py -- sweep 173: the suggestion node grown into a tree, the outcome audit
log, the routing fingerprint, and the STOLEN alias audit. Every pin is a contract that was actually
measured, not a smoke test. The module's own _selftest carries the fine-grained numeric pins; these
tests pin the WIRING (mind verbs, catalog cards, examples) and the two catalog-level facts the
session found and fixed.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("PYTHONHASHSEED", "0")


def test_module_selftest_pins():
    from holographic.agents_and_reasoning import holographic_decisiontree as dt
    r = dt._selftest()
    assert r["ok"] is True and r["pinned"] == 13


def test_tree_grows_from_the_live_catalog_and_round_trips():
    """Live catalog, not a stub: the tree must grow, carry an abstain branch, and decode EXACTLY."""
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    r = m.decision_tree("turn a point cloud into a mesh", depth=2, encode=True)
    assert r["root"].action, r
    assert "abstain" in r["root"].branches
    assert r["coverage"]["nodes"] >= 2
    from holographic.agents_and_reasoning.holographic_decisiontree import read_tree
    assert read_tree(r["vector"], r["shape"], r["vocab"]) == r["root"]


def test_depth_one_reproduces_todays_flat_suggestion_node():
    """depth=1 is exactly the node route() already builds: options at the root, no children."""
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    r = m.decision_tree("make the thing do the stuff with the wobbly bits", depth=1)
    assert r["coverage"]["nodes"] == 1
    assert r["options"][()]["confident"] is False       # unparsed -> ask, same as route()
    for child in r["root"].branches.values():
        assert not child.branches                        # leaves only


def test_tree_encoding_is_bit_deterministic():
    """Tie-sensitive path: the same context must encode to the same bits, every time."""
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    a = m.decision_tree("turn a point cloud into a mesh", depth=2, encode=True)["vector"]
    b = m.decision_tree("turn a point cloud into a mesh", depth=2, encode=True)["vector"]
    assert np.array_equal(a, b)


def test_decision_memory_is_an_audit_log_that_abstains():
    """The audit role: recorded facts compare exactly; unfamiliar inputs abstain instead of guessing;
    a single recorded class is never confident without a stated floor (the false-confidence hole)."""
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    om = m.decision_memory()
    om.record("turn a point cloud into a mesh", "points_to_mesh", None, label="a")
    om.record("build a surface from these points", "points_to_mesh", None, label="b")
    om.record("render a scene with path tracing", "path_trace", None, label="c")
    assert om.classes() == {"points_to_mesh": ["a", "b"], "path_trace": ["c"]}
    assert om.compare("turn a point cloud into a mesh",
                      "build a surface from these points")["quadrant"] == "equivalent"
    assert om.compare("turn a point cloud into a mesh",
                      "render a scene with path tracing")["quadrant"] == "distinct"
    om2 = type(om)(dim=256, seed=0, encoder=om._encode)
    om2.record("smooth this mesh", "only_one", None)
    assert om2.recall("photosynthesis in coastal algae", None)["why"] == "single_class"


def test_catalog_cards_exist_run_and_are_discoverable():
    """The rule that governs this repo: a capability find_capability cannot surface does not exist."""
    import lecore
    from holographic.caching_and_storage.holographic_catalog import default_catalog
    cat = default_catalog()
    for name in ("decision_tree", "decision_memory"):
        card = cat.get(name)
        assert card is not None and len(card.does) <= 600, name
        exec(card.example, {})                                # the example must actually RUN
    m = lecore.UnifiedMind(dim=256, seed=0)
    for q, want in [("build a decision tree on the fly", "decision_tree"),
                    ("what can I do after this step", "decision_tree"),
                    ("which inputs lead to the same result", "decision_memory"),
                    ("alarm when similar inputs give different results", "decision_memory")]:
        assert m.find_capability(q)[0].name == want, q


def test_no_curated_alias_is_stolen_outright():
    """Sweep 173 found 118 curated aliases routing to a card whose NAME did not even contain the alias
    words -- including 'bind', 'bundle' and 'cleanup', which lost to the Hypervector datatype card on an
    alphabetical tie. Each was extended with context words (Moose's rule: add words, remove nothing),
    measured 118/118 fixed with 0 regressions and 0 natural-query flips. This pin keeps it at zero."""
    from holographic.caching_and_storage.holographic_catalog import default_catalog, _tokens
    cat = default_catalog()
    outright = []
    for c in cat.all():
        for a in c.aliases:
            top = cat.find_capability(a)
            if top and top[0].name != c.name and not set(_tokens(a)) <= set(_tokens(top[0].name)):
                outright.append((c.name, a, top[0].name))
    assert outright == [], outright[:5]
    assert cat.find_capability("bind primitive")[0].name == "kernel verbs"


def test_skill_lint_reports_the_stolen_class():
    import tools.skill_lint as sl
    al = sl.audit_aliases()
    assert "stolen" in al
    assert not [s for s in al["stolen"] if s[3] == "outright"]

"""The semantic action menu (`browse_capabilities(by='semantic')`) OMITS untagged capabilities, so tag coverage IS
the menu. It was 108/2095 = 5.2%: every one of the 1,203 auto-registered mind faculties -- the engine's actual verb
surface -- arrived with `semantic=None`, which meant the File -> Export -> PNG tree could not see 95% of what the
engine does. `render/` was an EMPTY branch while 50 render faculties existed.

The fix derives the tag from the verb in the name at registration (holographic_semantictag), rather than asking a
human to hand-write 1,200 tags that then rot. These tests pin the three things that make that safe:

  1. the table may APPLY the taxonomy, never mint it -- every emitted root is one of the ten in
     SEMANTIC_TAXONOMY.md, kept in sync with the catalog's S4.1 drift gate BY TEST rather than by hope;
  2. it ABSTAINS on unknown verbs -- a wrong branch files a capability under a verb nobody looks for and, unlike a
     missing tag, looks finished;
  3. coverage does not regress, and the menu reads THIS MIND's catalog (the bug that made the new tags invisible:
     the faculty called browse_semantic() bare, which silently falls back to the ~400-entry default_catalog).
"""
import re

import pytest

from holographic.caching_and_storage.holographic_semantictag import (ROOTS, _RULES, coverage, infer_semantic)
from holographic.misc.holographic_unified import UnifiedMind


def test_roots_match_the_catalog_drift_gate():
    """The S4.1 gate fails any tag whose root is unknown. If this module's ROOTS ever drift from that gate's set,
    a derived tag would redden the build -- so the two are pinned equal here rather than kept in sync by memory."""
    import holographic.caching_and_storage.holographic_catalog as cat

    src = open(cat.__file__, encoding="utf-8").read()
    m = re.search(r'_ROOTS = \{([^}]+)\}', src, re.S)
    gate_roots = set(re.findall(r'"([a-z]+)"', m.group(1)))
    assert gate_roots == set(ROOTS), ("semantictag.ROOTS drifted from the catalog's S4.1 gate", gate_roots ^ set(ROOTS))


def test_every_rule_emits_a_documented_root():
    """The table applies taxonomy; it never invents a root. SEMANTIC_TAXONOMY.md is explicit that a new root is a
    design conversation, not a registration."""
    for stems, tag in _RULES:
        assert tag.split("/")[0] in ROOTS, (tag, "unknown root")
        assert stems, (tag, "a rule with no stems can never fire")


def test_inference_is_deterministic_and_pure():
    """Same name -> same tag, always. The engine's determinism rule applies to its own metadata too."""
    for name in ("render_scene", "export_splats", "subdivide_mesh", "frobnicate", "damage_mask"):
        assert infer_semantic(name) == infer_semantic(name)


@pytest.mark.parametrize("name,expected", [
    ("render_scene", "render/raster"),
    ("subdivide_mesh", "modify/subdivide"),
    ("export_splats", "io/export"),          # io beats render: rule ORDER is the disambiguation strategy
    ("skin_mesh", "animate/skin"),
    ("measure_area", "measure/area"),        # specific stem must beat the generic measure/ catch-all
    ("snap_to_midpoints", "transform/snap"),
    ("mesh_curvature", "measure/curvature"),
])
def test_known_verbs_land_in_the_right_branch(name, expected):
    assert infer_semantic(name) == expected


@pytest.mark.parametrize("name", ["frobnicate", "", "xyzzy_plugh", "svg_canvas", "planet_field"])
def test_abstains_rather_than_guess(name):
    """KEPT NEGATIVE, pinned: noun-named faculties have no verb to file under, and unknown verbs are not guessed.
    Coverage is a MEASURED outcome; loosening the table to hit a number would buy wrong menu branches."""
    assert infer_semantic(name) is None


def test_module_names_abstain_by_design():
    """A module is not an action. 'holographic_raypick' must never appear in a File->Export->PNG verb menu, so the
    ~500 module-name entries abstaining is CORRECT, not missing coverage."""
    for name in ("holographic_raypick", "holographic_iokinds", "holographic_param"):
        assert infer_semantic(name) is None, (name, "a module name must not be filed as a verb")


def test_docstring_is_a_weak_fallback_only():
    """The doc is consulted ONLY when the name says nothing, and never overrides it -- 'Return the widget' must not
    become a tag off boilerplate."""
    assert infer_semantic("frobnicate", "Return the widget") is None
    assert infer_semantic("export_splats", "Render the splats") == "io/export"
    assert infer_semantic("frobnicate", "Render the thing") == "render/raster"


def test_coverage_does_not_regress_and_the_menu_sees_the_minds_own_catalog():
    """The whole point, end to end: coverage well above the 5.2% it started at, every root populated, and the menu
    reading THIS mind's catalog rather than the module-level default."""
    m = UnifiedMind(dim=64, seed=0)
    cov = m.semantic_tag_coverage()
    assert cov["tagged"] > 500, ("tag coverage regressed", cov)
    assert cov["pct"] > 25.0, ("tag coverage regressed", cov)

    menu = m.browse_capabilities(by="semantic")
    assert sum(menu.values()) == cov["tagged"], ("the menu must show every tagged capability", menu, cov)
    # render/ was an EMPTY branch before the fix while 50 render faculties existed -- the regression that matters.
    assert menu.get("render/", 0) > 10, ("render/ branch lost its members", menu)
    assert set(menu) >= {"analyze/", "create/", "io/", "measure/", "modify/", "render/", "simulate/", "transform/"}


# ---------------------------------------------------------------- S5 (sub-branch drift gate)

def _documented_branches():
    """Every `root/sub` the taxonomy doc admits exists."""
    import pathlib
    import re

    doc = (pathlib.Path(__file__).resolve().parent.parent / "docs" / "SEMANTIC_TAXONOMY.md").read_text(encoding="utf-8")
    return set(re.findall(r"`([a-z]+/[a-z]+)`", doc))


def _used_branches():
    m = UnifiedMind(dim=64, seed=0)
    return {c.semantic for c in m._capability_catalog().all() if getattr(c, "semantic", None)}


def test_no_tag_uses_an_undocumented_branch():
    """S5, the drift that S4.1 could not see. S4.1 gates the ROOT only, so `create/emit` x21 and `analyze/measure`
    x15 -- 102 capabilities under 15 branches the taxonomy never listed -- passed CI for as long as they existed.
    A menu path nobody documented is the discovery equivalent of a dark module: real, reachable, and absent from
    the map everyone reads.

    The fix is doc-first (the branches HAD members, so by the doc's own rule they earned their place and the DOC
    was what drifted) and this test is what makes it stick. Deliberately NOT a hardcoded list in the gate: that
    would be a third copy to drift. The doc and the live engine are compared directly."""
    undocumented = _used_branches() - _documented_branches()
    assert not undocumented, (
        "semantic tag(s) use a branch SEMANTIC_TAXONOMY.md does not list: %s -- either document the branch (if it "
        "has members it has earned its place) or retag." % sorted(undocumented))


def test_no_documented_branch_is_empty():
    """The other direction, which the doc itself declares a bug: "a root or sub-branch exists here only because
    real capabilities populate it. An empty menu branch is the discovery equivalent of a dark module."

    This caught two REAL misses rather than aspirational prose: convert/isosurface and simulate/pbd were empty
    because their members were UNTAGGED -- occupancy_to_mesh / mesh_from_sdf / points_to_mesh are exactly
    "points<->mesh, sdf<->mesh", and resolve_swept_collision is a PBD collision step that the stem "resolve" had
    mis-filed under analyze/pipeline. An empty branch is evidence, not decoration."""
    empty = _documented_branches() - _used_branches()
    assert not empty, (
        "documented branch(es) with NO members: %s -- either tag the capabilities that belong there (an empty "
        "branch usually means its members are untagged, not that the branch is wrong) or remove it." % sorted(empty))


def test_overrides_win_over_inference_and_stay_small():
    """The override map exists because a stem is a strong signal that is sometimes simply WRONG -- "resolve" is a
    capability lookup nine times out of ten and a PBD solver the tenth. The cure is an explicit exception, not a
    cleverer table (every special case in the table costs accuracy on every OTHER name). It must stay small: a
    large override list means the table is wrong, not the names."""
    from holographic.caching_and_storage.holographic_catalog import _SEMANTIC_OVERRIDES

    m = UnifiedMind(dim=64, seed=0)
    caps = {c.name: c for c in m._capability_catalog().all()}
    for name, tag in _SEMANTIC_OVERRIDES.items():
        assert caps[name].semantic == tag, (name, "override did not win", caps[name].semantic)
        assert tag in _documented_branches(), (name, tag, "an override may not invent a branch either")
    assert len(_SEMANTIC_OVERRIDES) < 30, "override list is growing -- fix the stem table instead"


def test_a_background_job_is_not_a_simulation():
    """KEPT NEGATIVE, mine: I tagged "Run any faculty as a background job" simulate/step. simulate/ is "evolve a
    physical field over time"; a job evolves nothing. Caught by READING the branch's members rather than trusting
    the coverage number -- a tag can be counted and still be wrong, which is exactly the failure abstention is
    supposed to prevent."""
    m = UnifiedMind(dim=64, seed=0)
    for c in m._capability_catalog().all():
        if c.name.startswith("Run any faculty"):
            assert c.semantic == "analyze/pipeline", c.semantic
            break
    else:
        raise AssertionError("the job_submit capability vanished")

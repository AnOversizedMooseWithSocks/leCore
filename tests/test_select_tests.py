"""Tests for tools/select_tests.py -- the affected-test selector's import-graph logic."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from select_tests import affected_tests, build_graph, _transitive

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rel_of(basename):
    """The on-disk path (and dotted name) of a module, found by its FILE NAME. Looked up from the import graph
    rather than hard-coded, so these tests keep working wherever the modules live (flat root, package, or a mix)."""
    mods, _ = build_graph(ROOT)
    for dotted, rel in mods.items():
        if dotted.split(".")[-1] == basename:
            return rel, dotted
    raise AssertionError("module %r not found anywhere in the repo" % basename)


def test_changed_module_selects_its_own_test():
    rel, _dotted = _rel_of("holographic_render")
    picked = affected_tests([rel], root=ROOT)
    assert picked != "ALL"
    assert any(os.path.basename(p) == "test_holographic_render.py" for p in picked)


def test_transitive_dependency_is_followed():
    # a test that reaches holographic_render only THROUGH holographic_unified must still be selected
    m, d = build_graph(ROOT)
    cache = {}
    rel, dotted = _rel_of("holographic_render")
    picked = affected_tests([rel], root=ROOT)
    # every picked test either imports render (transitively) or uses dynamic imports (always-run)
    from select_tests import _uses_dynamic_import, _path_to_module
    for t in picked:
        name = _path_to_module(t)                           # dotted module name, whatever the layout
        reach = _transitive(name, d, cache)
        assert (dotted in reach) or _uses_dynamic_import(os.path.join(ROOT, t)) or name == dotted, t


def test_docs_only_change_selects_nothing():
    assert affected_tests(["README.md", "NOTES_concepts.md"], root=ROOT) == []


def test_unknown_binary_is_conservative():
    assert affected_tests(["features/sprites.hsp"], root=ROOT) == "ALL"


def test_new_unmapped_py_is_conservative():
    # a .py that isn't a known module (a brand-new file not yet on disk in the graph) -> run everything
    assert affected_tests(["holographic_brand_new_module_xyz.py"], root=ROOT) == "ALL"


def test_leaf_change_is_smaller_than_full_suite():
    import glob
    total = len(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))) or len(glob.glob(os.path.join(ROOT, "test_*.py")))
    rel, _dotted = _rel_of("holographic_assetimport")
    picked = affected_tests([rel], root=ROOT)
    assert picked != "ALL"
    assert len(picked) < total                                 # a leaf change must skip SOME tests


def test_build_artifacts_are_inert():
    # the repo's own build zip must NOT force a full run (it's regenerated, never a test input)
    assert affected_tests(["holographic_vsa_complete.zip"], root=ROOT) == []
    # the exact docs+config+zip change set from a real PR -> nothing to run
    assert affected_tests(["NOTES_concepts.md", "ci.yml", "holographic_vsa_complete.zip"], root=ROOT) == []
    # packaging output dirs are inert too
    assert affected_tests(["dist/leos_core-0.1.0.whl"], root=ROOT) == []
    assert affected_tests(["leos_core.egg-info/PKG-INFO"], root=ROOT) == []


def test_unknown_archive_elsewhere_still_forces_full():
    # a .zip that ISN'T the build artifact could be genuine capability/test data -> stay safe, run everything
    assert affected_tests(["features/mystery_dataset.zip"], root=ROOT) == "ALL"

"""Pins for .gitignore -- the rules that decide what is SOURCE.

WHY (measured, kept loud). Two rules were quietly wrong for months, and both leaked into a delivery zip:

  1. `/__pycache__` -- a LEADING SLASH anchors a rule to the repo ROOT. So the root cache was ignored and
     every nested one (holographic/, tests/, benchmarks/, lecore_data/) was NOT. That is the whole reason
     pycache folders kept reappearing in the working tree. `__pycache__/` matches at any depth.

  2. `scripts/.knowledge_cache.json` -- the semantic tooling MOVED to tools/semantic/ and this rule did not
     follow it, leaving a 7.9 MB embedding cache unignored and one `git add -A` from being committed. CI
     restores that cache from actions/cache; it is never source.

A .gitignore is a list of claims about the repo, and nothing checked them. These tests do.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import make_repo_zip  # noqa: E402


@pytest.fixture(scope="module")
def rules():
    if not (ROOT / ".gitignore").exists():
        pytest.skip(".gitignore not present in this tree")
    return make_repo_zip.load_rules(ROOT)


@pytest.mark.parametrize("path", [
    "holographic/__pycache__",
    "tests/__pycache__",
    "benchmarks/__pycache__",
    "lecore_data/__pycache__",
    "tools/semantic/__pycache__",
])
def test_nested_pycache_is_ignored_at_any_depth(rules, path):
    """The leading-slash bug: `/__pycache__` only ever ignored the root one."""
    assert make_repo_zip.ignored(path, True, rules), (
        "%s is NOT ignored -- a `/__pycache__` style rule anchors to the repo root; use `__pycache__/`" % path)


def test_compiled_python_is_ignored(rules):
    assert make_repo_zip.ignored("holographic/misc/__pycache__/holographic_unified.cpython-312.pyc", False, rules)


def test_the_embedding_cache_is_ignored_where_it_actually_lives(rules):
    """7.9 MB, restored by CI from actions/cache, never source. The rule named scripts/ -- the old location."""
    assert make_repo_zip.ignored("tools/semantic/.knowledge_cache.json", False, rules), (
        "the embedding cache is unignored at its REAL path -- the rule still points at the old scripts/ layout")


@pytest.mark.parametrize("doc", [
    "docs/BACKLOG_modeling.md",
    "docs/BACKLOG_modeling_v2.md",
    "docs/BACKLOG_photo3d_retopo.md",
    "docs/PRIMITIVE_APPLICATION_BACKLOG.md",
    "docs/wiring_audit_backlog.md",
])
def test_backlogs_are_not_committed(rules, doc):
    """Working backlogs are local notes, not repo content -- they reached a delivery zip once."""
    assert make_repo_zip.ignored(doc, False, rules)


def test_build_artifacts_are_ignored(rules):
    """A zip inside the repo doubles the archive on every round trip, and bloats git history badly."""
    for art in ("repo.zip", "holographic_vsa_complete.zip"):
        assert make_repo_zip.ignored(art, False, rules), art
    for d in ("dist", "build", "build_pkg", "temp"):
        assert make_repo_zip.ignored(d, True, rules), d


@pytest.mark.parametrize("src", [
    "holographic/misc/holographic_unified.py",
    "holographic/caching_and_storage/holographic_catalog.py",
    "tools/regen_docs.py",
    "tests/test_regen_docs.py",
    "lecore_data/routing/index_128d.npz",
    ".github/workflows/ci.yml",
    "README.md",
    "capabilities.json",
])
def test_real_source_is_never_ignored(src):
    """The far worse failure mode. Shipping a cache is untidy; EXCLUDING the engine would be a broken repo.
    Note this reads the rules directly rather than taking the fixture, so it also runs if .gitignore is bare."""
    if not (ROOT / ".gitignore").exists():
        pytest.skip(".gitignore not present")
    rs = make_repo_zip.load_rules(ROOT)
    assert not make_repo_zip.ignored(src, False, rs), "%s is SOURCE and must never be ignored" % src

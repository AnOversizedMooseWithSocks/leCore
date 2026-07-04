"""Tests for capdoc.py -- the CAPABILITIES.md generator (catalog-driven, deterministic, drift-check friendly)."""
import os
import importlib.util


def _capdoc():
    path = os.path.join(os.path.dirname(__file__), "capdoc.py")
    spec = importlib.util.spec_from_file_location("capdoc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generates_and_is_nontrivial(tmp_path):
    mod = _capdoc()
    dest = mod.generate(root=str(tmp_path))
    text = open(dest, encoding="utf-8").read()
    assert "# leCore Capabilities" in text
    assert "## Core algebra" in text and "## Run it as a service" in text
    assert "mind.route(" in text                       # the runtime-discovery snippet is present
    assert len(text) > 3000


def test_is_deterministic(tmp_path):
    mod = _capdoc()
    a = open(mod.generate(root=str(tmp_path / "a")), encoding="utf-8").read() if False else None
    os.makedirs(tmp_path / "a"); os.makedirs(tmp_path / "b")
    t1 = open(mod.generate(root=str(tmp_path / "a")), encoding="utf-8").read()
    t2 = open(mod.generate(root=str(tmp_path / "b")), encoding="utf-8").read()
    assert t1 == t2                                    # no timestamp -> drift-check stable


def test_no_timestamp_in_output(tmp_path):
    mod = _capdoc()
    text = open(mod.generate(root=str(tmp_path)), encoding="utf-8").read()
    # would break the CI drift check if present
    import re
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text)


def test_every_curated_home_is_placed(tmp_path):
    """Every curated home lands somewhere in the doc (a theme or 'More capabilities') -- nothing silently dropped."""
    mod = _capdoc()
    from holographic_catalog import default_catalog
    homes = [c for c in default_catalog().all() if not c.name.startswith("holographic_")]
    text = open(mod.generate(root=str(tmp_path)), encoding="utf-8").read()
    for c in homes:
        assert ("### " + c.name) in text, c.name

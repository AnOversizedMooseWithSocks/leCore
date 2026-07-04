"""Tests that the runtime data is packaged correctly, so a pip-installed wheel is not missing its data.

These guard the exact gap we fixed: holographic_dictionary needs a data file at runtime, and the flat py_modules
layout does not ship a loose data/ folder -- so the data lives in the importable lecore_data package. If someone
moves the data or forgets package_data, these fail in plain pytest (ci.yml), before a broken wheel is ever built."""
import os
import ast


def test_lecore_data_package_resolves_dictionary():
    import lecore_data
    assert lecore_data.exists("knowledge", "dictionary.json.xz")
    assert lecore_data.exists("knowledge", "LICENSE_WORDNET.txt")     # provenance ships too


def test_dictionary_loads_via_the_package():
    import holographic_dictionary as d
    assert d.size() > 100000
    assert "force" in d.define("gravity").lower()


def test_heat_enrichment_data_is_packaged():
    import lecore_data
    assert lecore_data.exists("definitions", "native", "materials", "enrich.json")
    import holographic_heat as h
    assert len(h._load_enrichment()) > 0                             # the file is found and parsed


def test_setup_declares_lecore_data_and_its_files():
    """setup.py must list lecore_data as a package and include its data globs -- else the wheel drops the data."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, "setup.py"), encoding="utf-8").read()
    assert '"lecore_data"' in src and "package_data" in src
    assert "knowledge/*" in src                                      # the dictionary glob is present


def test_build_script_stages_the_data_package():
    """build_package.sh must copy lecore_data into the staging folder, or the wheel is built without it."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, "build_package.sh"), encoding="utf-8").read()
    assert "lecore_data" in src

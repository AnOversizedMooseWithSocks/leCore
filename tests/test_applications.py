"""Regression traps for the applications library (sweep 132, backlog A).

Two ways a library like this dies. It rots -- an application stops running and nobody notices because
nothing runs it. Or it drifts into a folder of scripts that reach past the engine into modules, at which
point it stops being a demonstration of the faculty surface and starts being a liability that breaks the
day a faculty changes. Both are pinned here, the second by AST rather than by convention.
"""
import ast
import os
import sys
import time

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import applications  # noqa: E402


@pytest.fixture(scope="module")
def mind():
    import lecore
    return lecore.UnifiedMind(dim=256, seed=0)


def _sources():
    for entry in applications.apps():
        path = os.path.join(_REPO, *entry["module"].split(".")) + ".py"
        yield entry["name"], path, open(path, encoding="utf-8").read()


def test_the_registry_and_the_library_cannot_disagree():
    """apps() reads each application's own declared metadata, so a name in the listing that does not
    import, or an application missing PROVES, fails here rather than in front of a reader."""
    applications._selftest()


@pytest.mark.parametrize("name", sorted(applications.REGISTRY))
def test_every_application_runs_end_to_end(name, mind):
    """The whole claim. Not "it imports" -- it runs, through the dispatcher, and returns the numbers it
    says it proves."""
    r = mind.app_run(name)
    assert r["name"] == name and r["proves"] and isinstance(r["proved"], dict) and r["proved"]
    assert r["seconds"] >= 0.0


@pytest.mark.parametrize("name", sorted(applications.REGISTRY))
def test_every_application_asserts_its_own_number(name):
    """Each application's _selftest is the thing that makes it a demonstration rather than a script that
    exits 0. Run it, in-process, and let its assertions speak."""
    import importlib
    importlib.import_module(applications.REGISTRY[name][0])._selftest()


def test_applications_reach_the_engine_only_through_faculties():
    """THE LOAD-BEARING RULE, enforced by AST so it cannot decay into a comment.

    An application that imports holographic.* is a script: it bypasses the faculty surface every other
    audit in this repo protects, it stops demonstrating the thing it was written to demonstrate, and it
    breaks silently the first time a module moves. numpy and the stdlib are fine -- an application has to
    build its own inputs -- but every engine call must go through the mind."""
    # The detector is checked against a KNOWN violation first: a rule nobody has watched fail is a rule
    # that quietly stopped being enforced, which is the failure mode this whole file exists against.
    assert _engine_imports("from holographic.rendering import x") == ["holographic.rendering"]
    assert _engine_imports("import holographic.mesh_and_geometry as g") == ["holographic.mesh_and_geometry"]
    assert _engine_imports("import numpy as np\nimport lecore") == []

    offenders = [(name, m) for name, _path, src in _sources() for m in _engine_imports(src)]
    assert not offenders, "applications must call mind.<verb>(), never import the engine: %s" % offenders


def _engine_imports(src):
    """Every `holographic.*` module a source imports. numpy, stdlib and lecore itself are not engine
    internals -- an application must be able to build its inputs and (in its selftest) a mind."""
    found = []
    for node in ast.walk(ast.parse(src)):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        found += [n for n in names if n.split(".")[0] == "holographic"]
    return found


def test_lecore_is_imported_only_inside_selftests():
    """`import lecore` is legitimate in an application's _selftest (it must build a mind to test itself)
    and nowhere else: run() takes the mind it is given, so an application that builds its own would be
    ignoring the caller's configuration -- dim, seed, and any state the caller taught it."""
    for name, path, src in _sources():
        tree = ast.parse(src)
        for node in tree.body:                       # module level only
            assert not (isinstance(node, ast.Import) and any(a.name == "lecore" for a in node.names)), \
                "%s imports lecore at module scope; run(mind) must use the mind it is handed" % name


def test_every_application_declares_what_it_proves():
    """A listing that says only what an application is called tells a reader nothing about whether to
    run it. PROVES is the field that makes mind.apps() worth reading."""
    for entry in applications.apps():
        assert len(entry["proves"]) > 40, entry
        assert entry["domain"] and entry["name"]


def test_the_library_is_affordable(mind):
    """An example nobody can afford to run is a rotting example. The whole library is budgeted in
    single-digit seconds; MEASURED at 0.29 s when it landed, and the ceiling is generous so this fails
    on a real regression rather than on a slow box."""
    t0 = time.time()
    for name in sorted(applications.REGISTRY):
        mind.app_run(name)
    elapsed = time.time() - t0
    assert elapsed < 20.0, "the library took %.1fs; it was 0.29s when it landed" % elapsed


def test_an_unknown_application_says_what_exists(mind):
    """The failure a caller actually hits, and it must arrive as a CALLER error.

    ValueError specifically: the service turns ValueError into {ok: false, error} and everything else
    into a 500, and this was a real 500 over HTTP until the type was changed. An agent that mistypes an
    application name should be told the name is wrong and what the names are -- not that the engine
    failed."""
    with pytest.raises(ValueError) as e:
        mind.app_run("no_such_application")
    assert "spectral_heat" in str(e.value)
    assert not isinstance(e.value, KeyError), "KeyError becomes an HTTP 500 at the /invoke boundary"


def test_the_artefact_application_is_deterministic(mind, tmp_path):
    """The GALLERY artefact is a claim about reproducibility: the same program on the same seed must
    produce the same pixels, or the picture in the gallery is not the picture the code makes."""
    a = mind.app_run("texture_composite", res=64, out_dir=str(tmp_path))
    b = mind.app_run("texture_composite", res=64, out_dir=str(tmp_path))
    assert a["proved"]["digest"] == b["proved"]["digest"]
    assert os.path.getsize(a["path"]) > 1000

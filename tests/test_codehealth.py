"""Traps for the code-health audit, and CHARACTERIZATION tests for the risks it found.

Two halves, and the second is the point. The first pins the audit. The second pins the three functions the
audit ranked as the worst cells in the complexity x exposure x exercise matrix -- a catalogued capability
and two mind faculties, all complex, none previously named by any test. These are characterization tests:
they assert what the code DOES today, so a future refactor of a CC-46 function has something to fail against.
"""
import numpy as np
import pytest

import lecore
from holographic.io_and_interop.holographic_codehealth import (
    complexity, health_report, ATTENTION_CC)
import ast


# ---------------------------------------------------------------- the audit itself
def test_complexity_counts_decision_points():
    src = ("def flat():\n    return 1\n\n"
           "def branchy(x):\n    if x:\n        return 1\n"
           "    for i in range(3):\n        pass\n"
           "    return [i for i in range(3) if i]\n")
    fns = {n.name: n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    assert complexity(fns["flat"]) == 1
    assert complexity(fns["branchy"]) >= 4


def test_attention_list_admits_only_complex_unmentioned_non_demos():
    """The contract that makes the list worth reading. A demo at CC 22 is complex on purpose and carries no
    production weight; letting it in would crowd out the real findings."""
    r = health_report(limit=25)
    for a in r["attention"]:
        assert a["cc"] >= ATTENTION_CC, "%s admitted at CC %d" % (a["qual"], a["cc"])
        assert not a["mentioned"], "%s is mentioned by a test and should not be flagged" % a["qual"]
        assert not a["demo"], "%s is a demo and should be excluded" % a["qual"]


def test_advertised_surfaces_are_ranked_first():
    """An advertised capability nothing exercises outranks a more complex internal one. That ordering IS the
    finding -- ranking by raw complexity puts the safest functions at the top."""
    r = health_report(limit=25)
    exposures = [a["exposure"] for a in r["attention"]]
    if "catalog" in exposures and "internal" in exposures:
        assert exposures.index("catalog") < exposures.index("internal")


def test_mind_faculty_round_trip():
    m = lecore.UnifiedMind(dim=128, seed=0)
    r = m.audit_complexity(limit=3)
    assert set(r) == {"evidence", "totals", "attention", "most_complex"}
    assert r["totals"]["functions"] > 3000 and r["totals"]["mean_cc"] > 1
    assert "Code health" in str(m.find_capability("cyclomatic complexity")[0]), \
        "the false friend reopened: 'cyclomatic complexity' must not resolve to the renderer's scene_cost"


# ---------------------------------------------------------------- characterization: the found risks
def _box_and_tris():
    """The engine's own unit box, plus a triangulated copy. Built through the mind so these tests exercise
    the real construction path rather than a hand-rolled cage."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    box = lecore.UnifiedMind(dim=128, seed=0).mesh_box()
    V = np.asarray(box.vertices, float)
    F = np.asarray(box.faces, int)
    T = np.array([[a, b, c] for a, b, c, d in F] + [[a, c, d] for a, b, c, d in F], int)
    return box, Mesh(V, T)


def test_catmull_clark_characterization():
    """THE WORST CELL THE AUDIT FOUND: CC 46, registered in the catalog as an advertised capability, and not
    named by a single test until now.

    Pins the STRUCTURAL invariants rather than coordinates, so the test survives a legitimate refactor while
    still catching a real regression. The vertex count is the sharp one: Catmull-Clark produces exactly one
    new vertex per original vertex, edge and face, so a cube must go 8 -> 8+12+6 = 26. An off-by-one in the
    edge-point pass changes that number and nothing else would notice."""
    from holographic.mesh_and_geometry.holographic_meshsubdiv import catmull_clark
    box, _ = _box_and_tris()
    V0, F0 = np.asarray(box.vertices, float), np.asarray(box.faces, int)
    out = catmull_clark(box, levels=1)
    V, F = np.asarray(out.vertices, float), np.asarray(out.faces, int)
    assert F.shape[1] == 4, "Catmull-Clark must produce quads, got %d-gons" % F.shape[1]
    assert len(F) == 4 * len(F0), "one level must quadruple faces (%d -> %d)" % (len(F0), len(F))
    assert len(V) == 26, "expected V+E+F = 8+12+6 = 26 vertices for a cube, got %d" % len(V)
    assert V.min() >= V0.min() - 1e-9 and V.max() <= V0.max() + 1e-9, \
        "subdivided vertices left the convex hull of the control cage"
    assert set(np.unique(F)) == set(range(len(V))), "orphaned vertices: the result is not watertight"
    assert np.array_equal(np.asarray(catmull_clark(box, levels=1).vertices, float), V), "not deterministic"


def test_shrinkwrap_characterization():
    """CC 26, a mind faculty, previously unnamed by any test.

    Pins the defining behaviour: factor=1.0 lands every vertex ON the target surface, factor=0.0 is an EXACT
    no-op, and the reported distances match the movement actually performed."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    from holographic.mesh_and_geometry.holographic_meshtools import shrinkwrap
    _, tri = _box_and_tris()
    src = Mesh(np.array([[0., 0., 2.], [1., 0., 2.], [0., 1., 2.]]), np.array([[0, 1, 2]]))
    moved, dists = shrinkwrap(src, tri, factor=1.0)
    V = np.asarray(moved.vertices, float)
    assert V.shape == (3, 3)
    assert np.allclose(V[:, 2], 0.5), "factor=1.0 must land vertices on the target surface (z=0.5)"
    assert np.asarray(dists).shape == (3,), "one distance per vertex"
    assert np.allclose(np.asarray(dists), np.linalg.norm(V - np.asarray(src.vertices, float), axis=1)), \
        "reported distances do not match the movement performed"
    noop = np.asarray(shrinkwrap(src, tri, factor=0.0)[0].vertices, float)
    assert np.array_equal(noop, np.asarray(src.vertices, float)), "factor=0.0 must be an EXACT no-op"


def test_trace_streamlines_characterization():
    """CC 27, a mind faculty, previously unnamed by any test.

    Pins determinism and containment -- the two properties a field integrator most easily loses. A curve that
    escapes the mesh means the walk left the surface; a curve that differs between runs means a seed leaked."""
    from holographic.mesh_and_geometry.holographic_crossfield import trace_streamlines
    _, tri = _box_and_tris()
    n_faces = len(np.asarray(tri.faces))
    field = np.tile(np.array([1.0, 0.0, 0.0]), (n_faces, 1))
    a = trace_streamlines(tri, field, seed=0, n_seeds=4, max_steps=20)
    b = trace_streamlines(tri, field, seed=0, n_seeds=4, max_steps=20)
    assert len(a) == len(b) and len(a) > 0, "streamline count is not deterministic"
    for pa, pb in zip(a, b):
        assert np.array_equal(np.asarray(pa, float), np.asarray(pb, float)), "streamlines are not deterministic"
    pts = np.concatenate([np.asarray(p, float) for p in a if len(p)])
    V = np.asarray(tri.vertices, float)
    assert pts.min() >= V.min() - 1e-9 and pts.max() <= V.max() + 1e-9, \
        "a streamline left the mesh: the walk came off the surface"


# ---------------------------------------------------------------- the exercise axis, and its refutation
def test_facade_delegation_is_resolved_one_hop():
    """THE NAMED REGRESSION. reproject_uv (CC 67) was this audit's number-one finding until coverage.py showed
    163 of its 272 lines executing, reached through mind.mesh_reproject_uv. The engine delegates
    faculty-to-module under a DIFFERENT NAME, so a scan for the module function finds nothing.

    If the delegation map ever stops resolving that hop, the audit goes back to leading with its own biggest
    false positive."""
    from holographic.io_and_interop.holographic_codehealth import delegation_map
    dmap = delegation_map()
    assert "mesh_reproject_uv" in dmap, "the facade delegation map lost a known faculty"
    assert "reproject_uv" in dmap["mesh_reproject_uv"], \
        "import aliases must be recorded by their REAL name -- the alias is exactly what hides delegation"
    r = health_report(limit=40)
    flagged = {a["name"] for a in r["attention"]}
    assert "reproject_uv" not in flagged, "reproject_uv is exercised via the facade and must not be flagged"


def test_report_states_its_evidence_source():
    """The mention scan went two-for-two on false headlines, so the report must never present its verdict
    without saying which kind of evidence produced it."""
    r = health_report(limit=1)
    assert "evidence" in r and isinstance(r["evidence"], str) and r["evidence"]
    assert r["evidence"].startswith("MENTION SCAN") or r["evidence"].startswith("COVERAGE")
    if r["evidence"].startswith("MENTION SCAN"):
        assert "over-reports" in r["evidence"], "the fallback must carry its own health warning"


def test_missing_coverage_database_degrades_to_the_scan_not_to_a_crash():
    from holographic.io_and_interop.holographic_codehealth import coverage_hits
    hits, note = coverage_hits("/nonexistent/path/.coverage")
    assert hits == {} and note.startswith("MENTION SCAN")

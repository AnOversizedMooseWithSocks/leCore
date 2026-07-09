"""Tests for representation routing (ARCH-7): the capability table routes operations to the representation that
supports them, and CSG (the flagship) routes meshes through the SDF to compute booleans -- merging or keeping
topology separate, and geometrically correct by inclusion-exclusion. The booleans are extracted once and cached
(the field sampling is the expensive part)."""

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import Mesh
from holographic.mesh_and_geometry.holographic_meshsmooth import _icosphere
from holographic.scene_and_pipeline.holographic_route import REPRESENTATION_CAPABILITIES, representation_for, route_csg, connected_components, mesh_volume, _translate

_CACHE = {}


def _booleans():
    """Overlapping spheres A,B and their union/intersection/difference -- computed once."""
    if "ov" not in _CACHE:
        sph = _icosphere(2)
        A = _translate(sph, [-0.5, 0, 0]); B = _translate(sph, [0.5, 0, 0])
        _CACHE["ov"] = dict(A=A, B=B,
                            uni=route_csg("union", A, B),
                            inter=route_csg("intersection", A, B),
                            diff=route_csg("difference", A, B))
    return _CACHE["ov"]


# ---- the routing policy ---------------------------------------------------------------------------
def test_routing_table_sends_booleans_to_sdf():
    for op in ("union", "intersection", "difference"):
        assert representation_for(op) == "sdf"


def test_routing_table_sends_boundary_to_mesh():
    assert representation_for("boundary") == "mesh"
    assert representation_for("render") == "mesh"


def test_union_is_not_a_mesh_capability():
    # the reason routing exists: the mesh kernel can't do booleans natively
    assert "union" not in REPRESENTATION_CAPABILITIES["mesh"]


def test_representation_for_unknown_operation_raises():
    try:
        representation_for("teleport")
        assert False
    except ValueError:
        pass


# ---- CSG topology ---------------------------------------------------------------------------------
def test_overlapping_union_merges_to_one_component():
    assert connected_components(_booleans()["uni"]) == 1


def test_union_result_is_a_closed_manifold():
    uni = _booleans()["uni"]
    assert uni.is_closed() and uni.is_manifold()


def test_separate_union_stays_two_components():
    sph = _icosphere(2)
    A = _translate(sph, [-1.6, 0, 0]); B = _translate(sph, [1.6, 0, 0])
    assert connected_components(route_csg("union", A, B)) == 2


# ---- CSG geometric correctness --------------------------------------------------------------------
def test_intersection_is_smaller_than_either_input():
    d = _booleans()
    v_int = mesh_volume(d["inter"])
    assert v_int < mesh_volume(d["A"]) and v_int < mesh_volume(d["uni"])


def test_difference_is_smaller_than_the_minuend():
    d = _booleans()
    assert mesh_volume(d["diff"]) < mesh_volume(d["A"])


def test_inclusion_exclusion_for_union():
    d = _booleans()
    vA, vB = mesh_volume(d["A"]), mesh_volume(d["B"])
    assert abs(mesh_volume(d["uni"]) - (vA + vB - mesh_volume(d["inter"]))) / vA < 0.05


def test_intersection_plus_difference_recovers_A():
    d = _booleans()
    vA = mesh_volume(d["A"])
    assert abs(vA - (mesh_volume(d["inter"]) + mesh_volume(d["diff"]))) / vA < 0.05


# ---- helpers / determinism ------------------------------------------------------------------------
def test_connected_components_of_a_single_sphere_is_one():
    assert connected_components(_icosphere(2)) == 1


def test_csg_is_deterministic():
    d = _booleans()
    again = route_csg("union", d["A"], d["B"])
    assert np.array_equal(again.vertices, d["uni"].vertices)

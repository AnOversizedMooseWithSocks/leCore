"""P2 (client): LoadedMesh.split_by_material -- one submesh per material, reindexed.

A .glb import merges a whole multi-material scene into ONE mesh, so a consumer that samples it with a single
texture paints most faces with the wrong image (the reported fishing-spider file). Every such consumer was
re-implementing face grouping + vertex reindexing + UV subsetting; this makes it one call.
"""
import numpy as np
import pytest

from holographic.io_and_interop.holographic_assetimport import LoadedMesh
from holographic.misc.holographic_unified import UnifiedMind


@pytest.fixture(scope="module")
def mind():
    return UnifiedMind(dim=64, seed=0)


def _two_material_quad():
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    faces = np.array([[0, 1, 2], [1, 3, 2]], int)
    return pos, faces


def test_split_reindexes_each_material_to_a_compact_vertex_set():
    pos, faces = _two_material_quad()
    lm = LoadedMesh(pos, faces, face_material=["red", "blue"])
    parts = lm.split_by_material()
    assert list(parts) == ["red", "blue"], "first-seen order, deterministic"
    for name, p in parts.items():
        assert len(p.faces) == 1
        assert int(p.faces.max()) < len(p.positions), "faces must index only this submesh's own vertices"
        assert len(p.positions) == 3, "a triangle touches exactly 3 verts after reindex"
        assert p.face_material == [name] * len(p.faces)


def test_glTF_per_vertex_uv_is_subset_by_the_vertex_remap():
    pos, faces = _two_material_quad()
    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)          # (Nv,2) per-vertex = glTF
    lm = LoadedMesh(pos, faces, uv=uv, face_material=["red", "blue"])
    parts = lm.split_by_material()
    for p in parts.values():
        assert p.uv is not None and p.uv.shape == (len(p.positions), 2)


def test_OBJ_per_corner_uv_is_subset_by_the_face_selection():
    pos, faces = _two_material_quad()
    uv = np.zeros((2, 3, 2), float)                                # (Nf,3,2) per-corner = OBJ
    uv[0] = [[0, 0], [1, 0], [0, 1]]
    uv[1] = [[1, 0], [1, 1], [0, 1]]
    lm = LoadedMesh(pos, faces, uv=uv, face_material=["red", "blue"])
    parts = lm.split_by_material()
    for p in parts.values():
        assert p.uv is not None and p.uv.shape == (len(p.faces), 3, 2), "per-corner UV subsets by FACE, not vertex"


def test_no_material_returns_the_mesh_as_one_group():
    """A face with no material name must not be silently dropped; no material at all is one group, not an error."""
    pos, faces = _two_material_quad()
    lm = LoadedMesh(pos, faces, face_material=[])
    parts = lm.split_by_material()
    assert list(parts) == [""] and parts[""] is lm


def test_wired_and_discoverable(mind):
    pos, faces = _two_material_quad()
    lm = LoadedMesh(pos, faces, face_material=["a", "b"])
    parts = mind.split_by_material(lm)                              # the faculty delegates to the method
    assert list(parts) == ["a", "b"]
    hits = [c.name for c in mind.find_capability("split a mesh by material")[:3]]
    assert any("per-material" in n for n in hits), hits

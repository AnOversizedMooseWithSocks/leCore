"""Tests for the binary glTF (.glb) boundary (FWD-2): structural conformance (the things a GLTFLoader checks
before touching geometry), the position/normal/uv round-trip, byte-reproducibility, and rejection of a
malformed container. Proves offline that what we hand three.js is well-formed and lossless."""

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import box, tetrahedron, grid
from holographic.io_and_interop.holographic_gltf import mesh_to_glb, glb_to_mesh, validate_glb, write_glb, read_glb


def _cube_with_uvs():
    m = box(2.0, 2.0, 2.0)
    m.uvs = np.zeros((m.n_vertices, 2))          # trivial UVs so TEXCOORD_0 is exercised
    return m


def test_glb_is_structurally_valid():
    # every structural check a loader makes must pass (magic, version, lengths, chunk order/alignment, bounds)
    checks = validate_glb(mesh_to_glb(_cube_with_uvs()))
    assert all(checks.values()), checks


def test_glb_total_length_is_4_aligned():
    data = mesh_to_glb(box(1, 1, 1))
    assert len(data) % 4 == 0


def test_glb_round_trip_positions_exact():
    m = _cube_with_uvs()
    back = glb_to_mesh(mesh_to_glb(m))
    assert back.n_vertices == 8
    # we write+read float32, so positions come back exactly at float32 precision
    assert np.allclose(back.vertices.astype(np.float32), m.vertices.astype(np.float32))


def test_glb_round_trips_normals_and_uvs():
    back = glb_to_mesh(mesh_to_glb(_cube_with_uvs()))
    assert back.normals is not None
    assert back.uvs is not None
    assert back.normals.shape == (8, 3)
    assert back.uvs.shape == (8, 2)


def test_glb_triangle_count_matches():
    # 6 quads -> 12 triangles across the boundary
    back = glb_to_mesh(mesh_to_glb(box(2, 2, 2)))
    assert back.n_faces == 12


def test_glb_position_accessor_has_bounds():
    # glTF REQUIRES the POSITION accessor to carry min/max -- validate_glb checks it specifically
    assert validate_glb(mesh_to_glb(box(2, 2, 2)))["position_has_bounds"]


def test_glb_is_byte_reproducible():
    # a serialised artifact is the EXACT class: the same mesh yields identical bytes run to run
    assert mesh_to_glb(box(2, 2, 2)) == mesh_to_glb(box(2, 2, 2))


def test_glb_handles_triangle_and_open_meshes():
    for mk in (tetrahedron(), grid(3, 3)):
        data = mesh_to_glb(mk)
        assert all(validate_glb(data).values())
        back = glb_to_mesh(data)
        assert np.allclose(back.vertices.astype(np.float32), mk.vertices.astype(np.float32))


def test_glb_uses_uint16_indices_for_small_meshes():
    # a small mesh should use the compact unsigned-short index type (componentType 5123)
    import json, struct
    data = mesh_to_glb(box(1, 1, 1))
    clen, _ = struct.unpack_from("<II", data, 12)
    gltf = json.loads(data[20:20 + clen].decode("utf-8"))
    idx_acc = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["indices"]]
    assert idx_acc["componentType"] == 5123      # UNSIGNED_SHORT


def test_glb_rejects_bad_magic():
    try:
        glb_to_mesh(b"NOPE" + b"\x00" * 32)
        assert False, "should reject non-glb bytes"
    except ValueError:
        pass


def test_glb_file_round_trip(tmp_path):
    m = _cube_with_uvs()
    p = tmp_path / "cube.glb"
    write_glb(m, str(p))
    back = read_glb(str(p))
    assert back.n_vertices == 8 and back.n_faces == 12


def test_read_accessor_honours_byteoffset_and_stride():
    """The REAL-WORLD layout regression (found by a Sketchfab .glb that imported 0 verts): accessors packed into ONE
    bufferView at byteOffsets, and interleaved accessors with a byteStride. The first _read_accessor ignored both and
    silently read all-zeros. Build both layouts by hand and assert exact recovery."""
    import numpy as np
    from holographic.io_and_interop.holographic_gltf import _read_accessor

    pos = np.arange(12, dtype="<f4").reshape(4, 3)            # 4 VEC3 floats
    nrm = (np.arange(12, dtype="<f4") + 100).reshape(4, 3)
    # layout A: two accessors share one bufferView via accessor.byteOffset
    blob = pos.tobytes() + nrm.tobytes()
    views = [{"byteOffset": 0, "byteLength": len(blob)}]
    a_pos = {"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 4, "type": "VEC3"}
    a_nrm = {"bufferView": 0, "byteOffset": pos.nbytes, "componentType": 5126, "count": 4, "type": "VEC3"}
    assert np.allclose(_read_accessor(a_pos, views, blob), pos)
    assert np.allclose(_read_accessor(a_nrm, views, blob), nrm)          # was all-zeros before the fix
    # layout B: interleaved pos+nrm with byteStride 24
    inter = np.hstack([pos, nrm]).astype("<f4").tobytes()
    views_b = [{"byteOffset": 0, "byteLength": len(inter), "byteStride": 24}]
    b_pos = {"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 4, "type": "VEC3"}
    b_nrm = {"bufferView": 0, "byteOffset": 12, "componentType": 5126, "count": 4, "type": "VEC3"}
    assert np.allclose(_read_accessor(b_pos, views_b, inter), pos)
    assert np.allclose(_read_accessor(b_nrm, views_b, inter), nrm)

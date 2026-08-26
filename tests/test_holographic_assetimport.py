"""Tests for holographic_assetimport.py -- OBJ/MTL, glTF/GLB (with materials), Substance texture sets, volumes."""
import os
import numpy as np
import pytest
from holographic.io_and_interop.holographic_assetimport import load_obj, load_glb, load_texture_set, load_volume, import_asset, GridField, LoadedMesh, _classify_map


def test_load_obj_geometry_uv_and_material(tmp_path):
    (tmp_path / "m.obj").write_text(
        "mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\n"
        "usemtl red\nf 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n")
    (tmp_path / "m.mtl").write_text("newmtl red\nKd 0.8 0.1 0.1\nPr 0.4\nPm 0.0\n")
    lm = load_obj(str(tmp_path / "m.obj"))
    assert lm.positions.shape == (4, 3) and lm.faces.shape == (2, 3)
    assert lm.uv.shape == (2, 3, 2)                            # per-corner UVs recovered
    assert lm.face_material == ["red", "red"]
    assert abs(lm.materials["red"].base_color[0] - 0.8) < 1e-6
    assert lm.mesh().vertices.shape == (4, 3)                  # hands back an engine Mesh


def test_obj_polygon_fan_triangulation(tmp_path):
    # a quad face (4 verts) should fan-triangulate into 2 triangles
    (tmp_path / "q.obj").write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
    lm = load_obj(str(tmp_path / "q.obj"))
    assert lm.faces.shape == (2, 3)


def test_obj_negative_indices(tmp_path):
    (tmp_path / "n.obj").write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nf -3 -2 -1\n")
    lm = load_obj(str(tmp_path / "n.obj"))
    assert lm.faces.tolist() == [[0, 1, 2]]                    # -3,-2,-1 -> the three verts


def test_glb_round_trip_with_material(tmp_path):
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.io_and_interop.holographic_gltf import mesh_to_glb
    from holographic.materials_and_texture.holographic_materialio import PBRMaterial
    blob = mesh_to_glb(box(), material=PBRMaterial(name="steel", base_color=(0.2, 0.3, 0.9, 1.0),
                                                   metallic=1.0, roughness=0.3))
    p = tmp_path / "b.glb"; p.write_bytes(blob)
    glm = load_glb(str(p))
    assert len(glm.positions) > 0 and glm.materials
    mat = list(glm.materials.values())[0]
    assert abs(mat.metallic - 1.0) < 1e-6 and abs(mat.base_color[2] - 0.9) < 1e-6


def test_classify_map_naming():
    assert _classify_map("brick_BaseColor.png") == "base_color"
    assert _classify_map("wall_Normal_OpenGL.png") == "normal"
    assert _classify_map("x_Roughness.jpg") == "roughness"
    assert _classify_map("y_Metallic.png") == "metallic"
    assert _classify_map("readme.txt") is None


def test_load_texture_set(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    d = tmp_path / "painter"; d.mkdir()
    for nm, tint in [("t_basecolor.png", (200, 80, 60)), ("t_roughness.png", (180, 180, 180)),
                     ("t_normal.png", (128, 128, 255)), ("t_metallic.png", (10, 10, 10))]:
        a = np.zeros((8, 8, 3), np.uint8); a[:] = tint
        Image.fromarray(a).save(str(d / nm))
    mat = load_texture_set(str(d))
    assert mat.base_color_map is not None and mat.roughness_map is not None
    assert mat.normal_map is not None and mat.metallic_map is not None
    assert set(mat.channels_found) >= {"base_color", "roughness", "normal", "metallic"}


def test_load_volume_npy_and_field(tmp_path):
    grid = np.zeros((16, 16, 16), np.float32); grid[6:10, 6:10, 6:10] = 1.0
    p = tmp_path / "v.npy"; np.save(str(p), grid)
    field, bounds = load_volume(str(p), bounds=((-1, -1, -1), (1, 1, 1)))
    assert isinstance(field, GridField)
    assert field(np.array([[0.0, 0.0, 0.0]]))[0] > 0.5        # origin is inside the dense cube
    assert field(np.array([[0.95, 0.95, 0.95]]))[0] < 0.5     # corner is empty


def test_load_volume_raw(tmp_path):
    grid = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    p = tmp_path / "v.raw"; grid.tofile(str(p))
    field, _ = load_volume(str(p), dims=(2, 3, 4))
    assert field.grid.shape == (2, 3, 4)


def test_vdb_abstains_honestly(tmp_path):
    p = tmp_path / "x.vdb"; p.write_bytes(b"fake")
    with pytest.raises(ValueError):
        load_volume(str(p))                                    # proprietary sparse format -> refuse, don't guess


def test_import_asset_dispatch(tmp_path):
    (tmp_path / "m.obj").write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nf 1 2 3\n")
    assert isinstance(import_asset(str(tmp_path / "m.obj")), LoadedMesh)
    with pytest.raises(ValueError):
        import_asset(str(tmp_path / "unknown.xyz"))


# ---- glTF animation, skins, UVs, channels -----------------------------------------------------------------
def _pack_glb(gltf, blob):
    import struct, json as _json
    jb = _json.dumps(gltf).encode(); jb += b" " * ((4 - len(jb) % 4) % 4)
    bb = blob + b"\x00" * ((4 - len(blob) % 4) % 4)
    total = 12 + 8 + len(jb) + 8 + len(bb)
    out = struct.pack("<III", 0x46546C67, 2, total)
    out += struct.pack("<II", len(jb), 0x4E4F534A) + jb
    out += struct.pack("<II", len(bb), 0x004E4942) + bb
    return out


def _animated_glb(tmp_path, with_rotation=False, with_skin=False):
    """A minimal hand-built animated GLB: node 0 translated over 2 keyframes (+ optional rotation + skin)."""
    import json as _json
    times = np.array([0.0, 1.0], np.float32)
    trans = np.array([[0, 0, 0], [2, 0, 0]], np.float32)
    parts = [times.tobytes(), trans.tobytes()]
    accessors = [{"bufferView": 0, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0.0], "max": [1.0]},
                 {"bufferView": 1, "componentType": 5126, "count": 2, "type": "VEC3"}]
    views = [{"buffer": 0, "byteOffset": 0, "byteLength": 8}, {"buffer": 0, "byteOffset": 8, "byteLength": 24}]
    channels = [{"sampler": 0, "target": {"node": 0, "path": "translation"}}]
    samplers = [{"input": 0, "output": 1, "interpolation": "LINEAR"}]
    off = 32
    if with_rotation:
        # identity -> 90deg about Z, as quaternions (x,y,z,w)
        rots = np.array([[0, 0, 0, 1], [0, 0, 0.7071, 0.7071]], np.float32)
        parts.append(rots.tobytes())
        views.append({"buffer": 0, "byteOffset": off, "byteLength": 32}); off += 32
        accessors.append({"bufferView": 2, "componentType": 5126, "count": 2, "type": "VEC4"})
        samplers.append({"input": 0, "output": 3, "interpolation": "LINEAR"})
        channels.append({"sampler": 1, "target": {"node": 0, "path": "rotation"}})
        accessors.append({"bufferView": 0, "componentType": 5126, "count": 2, "type": "SCALAR"})  # reuse times as acc 3
        samplers[1]["input"] = 0; samplers[1]["output"] = 2
    blob = b"".join(parts)
    gltf = {"asset": {"version": "2.0"}, "nodes": [{"name": "bone"}],
            "buffers": [{"byteLength": len(blob)}], "bufferViews": views, "accessors": accessors,
            "animations": [{"name": "clip", "samplers": samplers, "channels": channels}]}
    if with_skin:
        gltf["nodes"].append({"name": "joint"})
        gltf["skins"] = [{"joints": [1]}]
    p = tmp_path / "anim.glb"; p.write_bytes(_pack_glb(gltf, blob))
    return str(p)


def test_gltf_animation_translation_interpolates():
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    lm = load_glb(_animated_glb(d))
    assert len(lm.animations) == 1
    clip = lm.animations[0]
    assert abs(clip.duration - 1.0) < 1e-6
    assert np.allclose(clip.sample(0.0)[0][:3, 3], [0, 0, 0])
    assert np.allclose(clip.sample(0.5)[0][:3, 3], [1, 0, 0])       # linear midpoint
    assert np.allclose(clip.sample(1.0)[0][:3, 3], [2, 0, 0])


def test_gltf_animation_rotation_slerps():
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    lm = load_glb(_animated_glb(d, with_rotation=True))
    clip = lm.animations[0]
    m0 = clip.sample(0.0)[0][:3, :3]
    m1 = clip.sample(1.0)[0][:3, :3]
    assert np.allclose(m0, np.eye(3), atol=1e-3)                    # identity at t=0
    # at t=1, ~90deg about Z: x-axis maps to y
    assert np.allclose(m1 @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-2)


def test_gltf_skin_imported():
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    lm = load_glb(_animated_glb(d, with_skin=True))
    assert len(lm.skins) == 1 and lm.skins[0]["joints"] == [1]


def test_gltf_carries_uv_and_normals_when_present():
    # a plain box has normals but no UVs; assert the mechanism carries what's there without crashing
    import tempfile, pathlib
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.io_and_interop.holographic_gltf import mesh_to_glb
    d = pathlib.Path(tempfile.mkdtemp())
    p = d / "b.glb"; p.write_bytes(mesh_to_glb(box()))
    lm = load_glb(str(p))
    assert lm.normals is not None                                   # per-vertex normals recovered


def test_gltf_occlusion_and_all_channels():
    # a material with all factor channels round-trips; the import exposes the map slots (None here, but present)
    import tempfile, pathlib
    from holographic.mesh_and_geometry.holographic_mesh import box
    from holographic.io_and_interop.holographic_gltf import mesh_to_glb
    from holographic.materials_and_texture.holographic_materialio import PBRMaterial
    d = pathlib.Path(tempfile.mkdtemp())
    p = d / "b.glb"
    p.write_bytes(mesh_to_glb(box(), material=PBRMaterial(name="m", metallic=0.5, roughness=0.4,
                                                          emissive=(0.1, 0.2, 0.3))))
    lm = load_glb(str(p))
    mat = list(lm.materials.values())[0]
    assert abs(mat.metallic - 0.5) < 1e-6 and abs(mat.roughness - 0.4) < 1e-6
    assert all(hasattr(mat, a) for a in ("base_color_map", "metallic_map", "roughness_map",
                                         "emissive_map", "normal_map", "ao_map"))

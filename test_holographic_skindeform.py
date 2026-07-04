"""Tests for holographic_skindeform.py -- linear-blend skinning + morph blending on imported rigs."""
import numpy as np
from holographic_assetimport import LoadedMesh, AnimationClip
from holographic_skindeform import (global_transforms, skin_positions, apply_morph, deform, deformed_positions,
                                    _dense_weights)


def _one_bone_rig():
    node_graph = [{"name": "root", "local": np.eye(4), "children": [1]},
                  {"name": "bone", "local": np.eye(4), "children": []}]
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    joints = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
    weights = np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    skins = [{"joints": [1], "inverse_bind": np.eye(4)[None]}]
    return LoadedMesh(positions, [(0, 1, 0)], joints=joints, weights=weights, skins=skins, node_graph=node_graph)


def _translate_clip():
    return AnimationClip("wave", {1: {"translation": (np.array([0.0, 1.0]),
                                                      np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))}})


def test_rest_pose_is_identity():
    lm = _one_bone_rig()
    assert np.allclose(deformed_positions(lm, clip=None), lm.positions)


def test_skinning_follows_animated_bone():
    lm, clip = _one_bone_rig(), _translate_clip()
    assert np.allclose(deformed_positions(lm, clip=clip, t=1.0), lm.positions + [0, 2, 0])
    assert np.allclose(deformed_positions(lm, clip=clip, t=0.5), lm.positions + [0, 1, 0])


def test_global_transforms_inherit_parent():
    lm = _one_bone_rig()
    T = np.eye(4); T[:3, 3] = [1, 0, 0]
    G = global_transforms(lm.node_graph, overrides={0: T})
    assert np.allclose(G[1][:3, 3], [1, 0, 0])                 # child inherits parent's translation


def test_dense_weights_expansion():
    joints = np.array([[2, 0, 0, 0], [1, 2, 0, 0]])
    weights = np.array([[1.0, 0, 0, 0], [0.5, 0.5, 0, 0]])
    W = _dense_weights(joints, weights, 3)
    assert W.shape == (2, 3)
    assert np.allclose(W[0], [0, 0, 1.0])                      # vertex 0 -> joint 2
    assert np.allclose(W[1], [0, 0.5, 0.5])                    # vertex 1 -> joints 1 and 2


def test_morph_blends_base_plus_weighted_delta():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    targets = np.array([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])   # lift by 1 on Y
    lm = LoadedMesh(positions, [(0, 1, 0)], morph_targets=targets, morph_weights=np.array([0.5]))
    assert np.allclose(deformed_positions(lm, clip=None), positions + [0, 0.5, 0])
    assert np.allclose(apply_morph(positions, targets, [1.0]), positions + [0, 1, 0])


def test_unrigged_mesh_is_untouched():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    lm = LoadedMesh(positions, [(0, 1, 0)])
    assert np.allclose(deform(lm).vertices, positions)         # no skin, no morph -> identity


def _skinned_glb():
    """Hand-build a minimal but COMPLETE skinned GLB: 2 verts both bound to one joint (node 1), which an animation
    translates +2 on Y over 1s. Returns the .glb bytes."""
    import struct, json

    def region(arr):                                          # bytes + 4-byte pad
        b = arr.tobytes()
        return b + b"\x00" * ((4 - len(b) % 4) % 4)

    pos = np.array([[0, 0, 0], [1, 0, 0]], np.float32)
    idx = np.array([0, 1, 0], np.uint16)
    jnt = np.array([[0, 0, 0, 0], [0, 0, 0, 0]], np.uint16)
    wgt = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], np.float32)
    ibm = np.eye(4, dtype=np.float32).reshape(1, 16)          # column-major identity == identity
    times = np.array([0, 1], np.float32)
    trans = np.array([[0, 0, 0], [0, 2, 0]], np.float32)

    arrs = [pos, idx, jnt, wgt, ibm, times, trans]
    blob = b""; views = []
    for a in arrs:
        off = len(blob)
        blob += region(a)
        views.append({"buffer": 0, "byteOffset": off, "byteLength": a.nbytes})

    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": 2, "type": "VEC3"},          # POSITION
        {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},         # indices
        {"bufferView": 2, "componentType": 5123, "count": 2, "type": "VEC4"},           # JOINTS_0 (u16)
        {"bufferView": 3, "componentType": 5126, "count": 2, "type": "VEC4"},           # WEIGHTS_0
        {"bufferView": 4, "componentType": 5126, "count": 1, "type": "MAT4"},           # inverse-bind
        {"bufferView": 5, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0.0], "max": [1.0]},  # times
        {"bufferView": 6, "componentType": 5126, "count": 2, "type": "VEC3"},           # translation values
    ]
    gltf = {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}], "scene": 0,
        "nodes": [{"name": "mesh_node", "mesh": 0, "skin": 0, "children": [1]},
                  {"name": "bone"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3}, "indices": 1}]}],
        "skins": [{"joints": [1], "inverseBindMatrices": 4}],
        "buffers": [{"byteLength": len(blob)}], "bufferViews": views, "accessors": accessors,
        "animations": [{"name": "wave", "samplers": [{"input": 5, "output": 6, "interpolation": "LINEAR"}],
                        "channels": [{"sampler": 0, "target": {"node": 1, "path": "translation"}}]}],
    }
    jb = json.dumps(gltf).encode(); jb += b" " * ((4 - len(jb) % 4) % 4)
    bb = blob + b"\x00" * ((4 - len(blob) % 4) % 4)
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(jb) + 8 + len(bb))
    out += struct.pack("<II", len(jb), 0x4E4F534A) + jb + struct.pack("<II", len(bb), 0x004E4942) + bb
    return out


def test_end_to_end_import_then_deform(tmp_path):
    """Import a real skinned GLB and deform it: at t=1 the vertices follow the bone's +2-on-Y translation."""
    from holographic_assetimport import load_glb
    p = tmp_path / "rig.glb"; p.write_bytes(_skinned_glb())
    lm = load_glb(str(p))
    assert lm.skins and lm.joints is not None and lm.weights is not None and lm.node_graph
    rest = deformed_positions(lm, clip=None)
    moved = deformed_positions(lm, clip=lm.animations[0], t=1.0)
    assert np.allclose(moved - rest, [0, 2, 0], atol=1e-4), (rest, moved)

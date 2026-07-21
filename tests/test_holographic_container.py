"""Tests for the app-neutral typed-section container (leStudio backlog item 11).

These pin the two acceptance criteria from the backlog: (1) leStudio's section model works against the CORE
container -- documents/masks/brushes packed as typed sections round-trip through the mind -- and (2) a foreign-kind
fixture survives save->load->save BYTE-IDENTICALLY.
"""
import numpy as np

from holographic.io_and_interop.holographic_container import save_container, load_container
from holographic.misc.holographic_unified import UnifiedMind


def _lestudio_like_sections():
    """A leStudio-style workspace expressed as core sections: a document (layers + a mask), a brush, and a node
    graph carried in meta -- exactly the data leStudio's .lews packs, but now as {kind, id, meta, arrays}."""
    rng = np.random.default_rng(0)
    return [
        {"kind": "lestudio.document", "id": "D1",
         "meta": {"name": "untitled", "w": 32, "h": 24,
                  "layers": [{"id": "L0", "name": "bg", "opacity": 1.0, "blend": "normal"}],
                  "graph": [{"id": "N1", "type": "Fractal", "params": {"julia": 1}, "inputs": {}}]},
         "arrays": {"layer_L0": rng.random((24, 32, 3)).astype(np.float32),
                    "mask_M0": (rng.random((24, 32)) > 0.5)}},
        {"kind": "lestudio.brush", "id": "B0",
         "meta": {"name": "soft", "spacing": 0.1, "builtin": True},
         "arrays": {"tip": rng.random((9, 9)).astype(np.float32)}},
    ]


def test_lestudio_document_and_brush_round_trip_through_the_core_container():
    m = UnifiedMind(dim=64, seed=0)
    secs = _lestudio_like_sections()
    blob = m.save_container(secs, meta={"app": "lestudio", "active": "D1"})
    got = m.load_container(blob)
    assert got["meta"] == {"app": "lestudio", "active": "D1"}
    by_id = {s["id"]: s for s in got["sections"]}
    # document reconstructs: nested meta (layers + graph) preserved, arrays bit-identical + dtype-preserving
    doc = by_id["D1"]
    assert doc["kind"] == "lestudio.document"
    assert doc["meta"]["graph"][0]["type"] == "Fractal"
    assert doc["arrays"]["layer_L0"].dtype == np.float32
    assert np.array_equal(doc["arrays"]["layer_L0"], secs[0]["arrays"]["layer_L0"])
    assert doc["arrays"]["mask_M0"].dtype == np.bool_
    # brush reconstructs
    assert by_id["B0"]["meta"]["spacing"] == 0.1
    assert np.array_equal(by_id["B0"]["arrays"]["tip"], secs[1]["arrays"]["tip"])


def test_foreign_kind_survives_save_load_save_byte_identically():
    """The keystone: a reader that understands only SOME kinds must carry the rest through losslessly. Here an app
    that only knows 'lestudio.document' loads a file that also holds a '3dapp.mesh' and a 'video.clip' it cannot
    open, and writes it back BYTE-IDENTICALLY."""
    m = UnifiedMind(dim=64, seed=0)
    rng = np.random.default_rng(1)
    secs = [
        {"kind": "lestudio.document", "id": "D1", "meta": {"w": 8, "h": 8},
         "arrays": {"layer_L0": rng.random((8, 8, 3)).astype(np.float32)}},
        {"kind": "3dapp.mesh", "id": "MESH", "meta": {"n_faces": 2, "material": "steel"},
         "arrays": {"verts": rng.random((6, 3)), "faces": np.array([[0, 1, 2], [3, 4, 5]], np.int64)}},
        {"kind": "video.clip", "id": "CLIP", "meta": {"fps": 24, "frames": 3},
         "arrays": {"thumb": rng.integers(0, 255, (4, 4, 3), np.uint8)}},
    ]
    blob = m.save_container(secs, meta={"created_by": "3dapp"})

    # the image editor loads the whole file (it does not interpret mesh/clip, but it must keep them)...
    loaded = m.load_container(blob)
    assert [s["kind"] for s in loaded["sections"]] == ["lestudio.document", "3dapp.mesh", "video.clip"]
    # ...and writes it straight back -> byte-for-byte identical (the empty-diff / lossless-carry property).
    assert m.save_container(loaded["sections"], meta=loaded["meta"]) == blob
    # the foreign sections are bit-identical, meta and all
    mesh = [s for s in loaded["sections"] if s["kind"] == "3dapp.mesh"][0]
    assert mesh["meta"] == {"n_faces": 2, "material": "steel"}
    assert mesh["arrays"]["faces"].dtype == np.int64
    assert np.array_equal(mesh["arrays"]["verts"], secs[1]["arrays"]["verts"])


def test_cross_item_workspace_composes_all_backlog_pieces():
    """THE ARC'S CAPSTONE: one container carries a leStudio document, the emitted SDF/postfx/pattern+palette shaders
    (items 8/9/10) as sections, and a frame-source-graded clip (item 12, map_frames driving color_transfer) -- and a
    reader that understands none of it writes it back byte-identically. The multi-app story, end to end."""
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    m = UnifiedMind(dim=64, seed=0)

    sdf_glsl = m.to_shadertoy(sphere(1.0), camera="uniforms")
    grade_glsl = m.postfx_to_glsl([("exposure", {"ev": 0.3}), ("vignette", {"strength": 0.35}), ("aces", {})])
    # compose the background pattern + palette into ONE shader via the shared assembler (B1) rather than string-concat
    bg_glsl = m.compose_shader([m.pattern_to_glsl("checker", scale=3.0),
                                m.cosine_palette_to_glsl(*m.random_palette(seed=7))])

    src = m.synthetic_frame_source(kind="gradient", size=(16, 16), frames=3)
    ref = np.zeros((8, 8, 3)); ref[..., 0] = 0.8
    graded = []
    for _ in range(3):
        out, _seq = m.map_frames(src, lambda fr: m.color_transfer(fr, ref, mode="meanstd"))
        graded.append(out)
        src.advance()

    def s2a(s):                                        # text -> uint8 (the container is numeric-only, by design)
        return np.frombuffer(s.encode("utf-8"), dtype=np.uint8).copy()

    sections = [
        {"kind": "lestudio.document", "id": "D1", "meta": {"w": 16, "h": 16},
         "arrays": {"layer0": np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)}},
        {"kind": "lecore.shader", "id": "SDF", "meta": {"lang": "glsl", "camera": "uniforms"},
         "arrays": {"src": s2a(sdf_glsl)}},
        {"kind": "lecore.shader", "id": "GRADE", "meta": {"lang": "glsl"}, "arrays": {"src": s2a(grade_glsl)}},
        {"kind": "lecore.shader", "id": "BG", "meta": {"lang": "glsl"}, "arrays": {"src": s2a(bg_glsl)}},
        {"kind": "video.gradedclip", "id": "CLIP", "meta": {"frames": 3},
         "arrays": {"f%d" % i: graded[i] for i in range(3)}},
    ]
    blob = m.save_container(sections, meta={"app": "crossitem"})
    got = m.load_container(blob)
    assert m.save_container(got["sections"], meta=got["meta"]) == blob      # lossless carry of ALL kinds
    back = {s["id"]: s for s in got["sections"]}
    assert "uniform float uAngle;" in bytes(back["SDF"]["arrays"]["src"]).decode("utf-8")
    assert "vec3 postfx" in bytes(back["GRADE"]["arrays"]["src"]).decode("utf-8")
    assert "float pattern(vec3 p)" in bytes(back["BG"]["arrays"]["src"]).decode("utf-8")
    assert np.array_equal(back["CLIP"]["arrays"]["f1"], graded[1])


def test_determinism_and_safety_guards():
    m = UnifiedMind(dim=64, seed=0)
    secs = [{"kind": "k", "id": "1", "meta": {"a": 1}, "arrays": {"x": np.arange(6)}}]
    # two saves of the same input are identical (no wall-clock timestamp leaks in)
    assert m.save_container(secs) == m.save_container(secs)
    # object-dtype arrays are refused (no pickle can enter the format), as is a non-container blob
    import pytest
    with pytest.raises(ValueError):
        m.save_container([{"kind": "k", "arrays": {"o": np.array([{"bad": 1}], object)}}])
    with pytest.raises(ValueError):
        m.load_container(b"definitely not a container")

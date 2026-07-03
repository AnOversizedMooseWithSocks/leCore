import numpy as np
from holographic_materialio import PBRMaterial, materials_from_mtl


def test_mtl_round_trip_exact():
    m = PBRMaterial("brass", base_color=(0.9, 0.7, 0.2, 1.0), metallic=1.0, roughness=0.3, emissive=(0.0, 0.0, 0.0))
    back = materials_from_mtl(m.to_mtl())[0]
    assert back.name == "brass"
    assert np.allclose(back.base_color, m.base_color)
    assert abs(back.metallic - m.metallic) < 1e-6 and abs(back.roughness - m.roughness) < 1e-6


def test_gltf_dict_round_trip():
    m = PBRMaterial("plastic", base_color=(0.2, 0.4, 0.9, 1.0), metallic=0.0, roughness=0.6, emissive=(0.1, 0.0, 0.0))
    g = PBRMaterial.from_gltf_dict(m.to_gltf_dict())
    assert np.allclose(g.base_color, m.base_color) and abs(g.roughness - m.roughness) < 1e-6
    assert np.allclose(g.emissive, m.emissive)


def test_vsa_record_carries_material():
    """A material round-trips through a single hypervector (bind+bundle) to the encoder's resolution."""
    from holographic_encoders import ScalarEncoder
    m = PBRMaterial("gold", base_color=(1.0, 0.84, 0.0, 1.0), metallic=1.0, roughness=0.2)
    enc = ScalarEncoder(8192, lo=0.0, hi=1.0, seed=0, kernel="rbf")
    v = PBRMaterial.from_vsa_record(m.to_vsa_record(enc), enc)
    err = max(abs(a - b) for a, b in zip(v.base_color, m.base_color))
    err = max(err, abs(v.metallic - m.metallic), abs(v.roughness - m.roughness))
    assert err < 0.05


def test_mesh_to_gltf_embeds_material():
    """The .glb writer embeds the PBR material's factors, not the default."""
    import json, struct
    from holographic_mesh import box
    from holographic_gltf import mesh_to_glb
    m = PBRMaterial("gold", base_color=(1.0, 0.84, 0.0, 1.0), metallic=1.0, roughness=0.2)
    glb = mesh_to_glb(box(), material=m)
    jlen = struct.unpack("<I", glb[12:16])[0]
    gj = json.loads(glb[20:20 + jlen].decode("utf-8"))
    pbr = gj["materials"][0]["pbrMetallicRoughness"]
    assert pbr["metallicFactor"] == 1.0 and pbr["roughnessFactor"] == 0.2
    assert abs(pbr["baseColorFactor"][1] - 0.84) < 1e-6


# --- Sweep 3 local completion: texture maps (bilinear UV sampling) ---
import numpy as _np_tex
from holographic_materialio import TextureMap, PBRMaterial as _PBR


def test_texture_bilinear():
    img = _np_tex.array([[[1.0], [0.0]], [[0.0], [1.0]]])            # 2x2 checker
    t = TextureMap(img, wrap="clamp")
    assert abs(float(t.sample(0.0, 0.0)[0]) - 1.0) < 1e-9            # corner clamps to texel (0,0)
    assert abs(float(t.sample(0.5, 0.5)[0]) - 0.5) < 1e-9           # centre = bilinear average of all four


def test_texture_repeat_wrap():
    img = _np_tex.array([[[0.2], [0.8]]])                           # 1x2 row
    t = TextureMap(img, wrap="repeat")
    a = t.sample(0.25, 0.5); b = t.sample(1.25, 0.5)               # u and u+1 wrap to the same texel
    assert abs(float(a[0]) - float(b[0])) < 1e-9


def test_material_uses_map_and_backward_compatible():
    red = _np_tex.zeros((4, 4, 3)); red[:, :, 0] = 1.0
    m = _PBR(base_color=(1, 1, 1, 1), base_color_map=TextureMap(red))
    s = m.sample(0.3, 0.7)
    assert tuple(round(x, 2) for x in s["base_color"]) == (1.0, 0.0, 0.0, 1.0)     # sampled red x white factor
    m0 = _PBR(base_color=(0.2, 0.4, 0.6, 1.0), metallic=0.5, roughness=0.7)        # no maps
    s0 = m0.sample()
    assert tuple(round(x, 2) for x in s0["base_color"]) == (0.2, 0.4, 0.6, 1.0)    # factors unchanged
    assert s0["metallic"] == 0.5 and s0["roughness"] == 0.7


def test_deterministic_sampling():
    rng = _np_tex.random.default_rng(0); img = rng.random((8, 8, 3))
    t = TextureMap(img)
    assert _np_tex.array_equal(t.sample(0.37, 0.62), t.sample(0.37, 0.62))

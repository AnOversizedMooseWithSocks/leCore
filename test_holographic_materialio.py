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

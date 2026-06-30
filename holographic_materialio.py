"""PBR material interchange (G2-IO): the standard-format bridge for materials, and a VSA-native carrier.

WHY THIS MODULE EXISTS
----------------------
The engine already has a VSA-native `Material` (holographic_material): per-channel TEXTURE FIELDS bound into
one hypervector -- the right representation for spatially-varying, composable, blendable materials. What was
missing is the bridge to the OUTSIDE world: the popular interchange formats a modeler imports from and exports
to. The current 3-D ecosystem has converged on one factor model -- glTF 2.0's **metallic-roughness** PBR
(baseColorFactor, metallicFactor, roughnessFactor, emissiveFactor), which is ISO/IEC 12113:2022, maps ~1:1 to
MaterialX's glTF-PBR node and to USD's UsdPreviewSurface, and is identical to Blender/Unreal/Unity's Principled
BSDF. So that is the canonical factor model this module adopts.

WHAT IT PROVIDES
  * `PBRMaterial` -- the standard factor set (base colour RGBA, metallic, roughness, emissive RGB) + a name.
    The single representation every export/import below maps through.
  * glTF: `to_gltf_dict()` -- the `materials[]` entry the .glb writer (holographic_gltf) embeds; export a mesh
    WITH its material instead of the old hard-coded default.
  * MTL: `to_mtl()` / `materials_from_mtl(text)` -- the OBJ companion, using the modern PBR extension keywords
    (Pr roughness, Pm metallic, Ke emissive) alongside legacy Kd/d/Ns so it round-trips with PBR-aware tools and
    still opens in old ones.
  * VSA bridge: `to_vsa_record(scalar_encoder)` / `from_vsa_record(record, scalar_encoder)` -- carry the whole
    material as ONE hypervector (each factor scalar-encoded and bound to its channel role, then bundled), so a
    material transmits, composes, and BLENDS with the engine's own bind/bundle algebra. Round-trips to the
    encoder's resolution. This is the "keep it VSA-native / exposed" path: a material is a bundle, like a splat
    scene or a typed record.

KEPT HONEST
  Factor-level only for now: constant base colour / metallic / roughness / emissive, not yet image TEXTURE maps
  (baseColorTexture etc.). Factors cover a large share of real materials and map cleanly across glTF/MTL/USD;
  packing texture images into the .glb is the natural next step and is flagged, not faked. The VSA record
  round-trips to the scalar encoder's resolution (a continuous-decode tolerance), not bit-exactly.
"""

import hashlib
import numpy as np

from holographic_ai import bind, unbind, bundle, random_vector

# The nine scalar factors the VSA record carries, in a fixed order (so the bundle is deterministic).
_VSA_CHANNELS = ("base_r", "base_g", "base_b", "base_a", "metallic", "roughness", "emis_r", "emis_g", "emis_b")


def _role(name, dim):
    """A deterministic ~unit role vector for a channel name (hashlib-seeded, so it is reproducible and does not
    depend on Python's salted hash())."""
    seed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big") % (2 ** 32)
    return random_vector(dim, np.random.default_rng(seed))


class PBRMaterial:
    """A glTF 2.0 metallic-roughness PBR material (factor level). Values are in [0,1]."""

    def __init__(self, name="material", base_color=(0.8, 0.8, 0.8, 1.0), metallic=0.0, roughness=0.8,
                 emissive=(0.0, 0.0, 0.0)):
        self.name = str(name)
        self.base_color = tuple(float(c) for c in base_color)        # RGBA
        self.metallic = float(metallic)
        self.roughness = float(roughness)
        self.emissive = tuple(float(c) for c in emissive)            # RGB

    def __repr__(self):
        return (f"PBRMaterial(name={self.name!r}, base_color={tuple(round(c,3) for c in self.base_color)}, "
                f"metallic={self.metallic:.3f}, roughness={self.roughness:.3f}, "
                f"emissive={tuple(round(c,3) for c in self.emissive)})")

    # ----- glTF -------------------------------------------------------------------------------------
    def to_gltf_dict(self):
        """The glTF 2.0 `materials[]` entry (pbrMetallicRoughness + emissiveFactor)."""
        return {
            "name": self.name,
            "pbrMetallicRoughness": {
                "baseColorFactor": list(self.base_color),
                "metallicFactor": self.metallic,
                "roughnessFactor": self.roughness,
            },
            "emissiveFactor": list(self.emissive),
        }

    @classmethod
    def from_gltf_dict(cls, d):
        """Parse a glTF `materials[]` entry back to a PBRMaterial."""
        pbr = d.get("pbrMetallicRoughness", {})
        return cls(
            name=d.get("name", "material"),
            base_color=tuple(pbr.get("baseColorFactor", [0.8, 0.8, 0.8, 1.0])),
            metallic=pbr.get("metallicFactor", 1.0),
            roughness=pbr.get("roughnessFactor", 1.0),
            emissive=tuple(d.get("emissiveFactor", [0.0, 0.0, 0.0])),
        )

    # ----- MTL (OBJ companion) ----------------------------------------------------------------------
    def to_mtl(self):
        """An MTL block: modern PBR keywords (Pr roughness, Pm metallic, Ke emissive) + legacy Kd/d/Ns so it
        round-trips with PBR-aware tools and still opens in old viewers. Ns is a Phong stand-in from roughness."""
        r, g, b, a = self.base_color
        er, eg, eb = self.emissive
        ns = (1.0 - self.roughness) ** 2 * 1000.0                     # roughness -> legacy shininess (approx)
        return (f"newmtl {self.name}\n"
                f"Kd {r:.6f} {g:.6f} {b:.6f}\n"
                f"d {a:.6f}\n"
                f"Ke {er:.6f} {eg:.6f} {eb:.6f}\n"
                f"Pr {self.roughness:.6f}\n"
                f"Pm {self.metallic:.6f}\n"
                f"Ns {ns:.6f}\n"
                f"illum 2\n")

    # ----- VSA-native carrier -----------------------------------------------------------------------
    def to_vsa_record(self, scalar_encoder):
        """The whole material as ONE hypervector: each factor scalar-encoded and bound to its channel role, then
        bundled. Transmits / composes / BLENDS with bind+bundle -- a material is a role-filler record, like a
        typed scene. `scalar_encoder` is a ScalarEncoder over [0,1]."""
        dim = scalar_encoder.dim
        vals = self._factor_dict()
        return bundle([bind(_role(ch, dim), scalar_encoder.encode(vals[ch])) for ch in _VSA_CHANNELS])

    @classmethod
    def from_vsa_record(cls, record, scalar_encoder, name="material"):
        """Recover a PBRMaterial from its hypervector record: unbind each channel role and decode the scalar.
        Round-trips to the encoder's resolution (a continuous decode, not bit-exact)."""
        dim = scalar_encoder.dim
        v = {ch: float(np.clip(scalar_encoder.decode(unbind(record, _role(ch, dim))), 0.0, 1.0))
             for ch in _VSA_CHANNELS}
        return cls(name=name,
                   base_color=(v["base_r"], v["base_g"], v["base_b"], v["base_a"]),
                   metallic=v["metallic"], roughness=v["roughness"],
                   emissive=(v["emis_r"], v["emis_g"], v["emis_b"]))

    def _factor_dict(self):
        r, g, b, a = self.base_color
        er, eg, eb = self.emissive
        return {"base_r": r, "base_g": g, "base_b": b, "base_a": a,
                "metallic": self.metallic, "roughness": self.roughness,
                "emis_r": er, "emis_g": eg, "emis_b": eb}


def materials_from_mtl(text):
    """Parse an MTL file's text into a list of PBRMaterial (Kd/d/Ke + the PBR extensions Pr/Pm). Unknown lines
    are ignored. Returns one PBRMaterial per `newmtl` block."""
    mats = []
    cur = None
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0]
        if key == "newmtl":
            cur = PBRMaterial(name=parts[1] if len(parts) > 1 else "material")
            mats.append(cur)
        elif cur is None:
            continue
        elif key == "Kd" and len(parts) >= 4:
            cur.base_color = (float(parts[1]), float(parts[2]), float(parts[3]), cur.base_color[3])
        elif key == "d" and len(parts) >= 2:
            r, g, b, _ = cur.base_color
            cur.base_color = (r, g, b, float(parts[1]))
        elif key == "Ke" and len(parts) >= 4:
            cur.emissive = (float(parts[1]), float(parts[2]), float(parts[3]))
        elif key == "Pr" and len(parts) >= 2:
            cur.roughness = float(parts[1])
        elif key == "Pm" and len(parts) >= 2:
            cur.metallic = float(parts[1])
    return mats


def _selftest():
    from holographic_encoders import ScalarEncoder
    m = PBRMaterial(name="copper", base_color=(0.95, 0.6, 0.3, 1.0), metallic=1.0, roughness=0.25,
                    emissive=(0.0, 0.0, 0.0))
    # MTL round-trip
    back = materials_from_mtl(m.to_mtl())[0]
    assert np.allclose(back.base_color, m.base_color) and abs(back.metallic - m.metallic) < 1e-6
    assert abs(back.roughness - m.roughness) < 1e-6
    # glTF dict round-trip
    g = PBRMaterial.from_gltf_dict(m.to_gltf_dict())
    assert np.allclose(g.base_color, m.base_color) and abs(g.roughness - m.roughness) < 1e-6
    # VSA record round-trip (to encoder resolution; crosstalk-limited -> use a wide vector for a tight read)
    enc = ScalarEncoder(8192, lo=0.0, hi=1.0, seed=0, kernel="rbf")
    rec = m.to_vsa_record(enc)
    v = PBRMaterial.from_vsa_record(rec, enc)
    err = max(abs(a - b) for a, b in zip(v.base_color, m.base_color))
    err = max(err, abs(v.metallic - m.metallic), abs(v.roughness - m.roughness))
    assert err < 0.05, err                                        # ~0.024 at dim 8192; ~0.06 at 2048 (kept honest)
    print(f"materialio selftest ok: MTL + glTF exact round-trip; VSA-record round-trip max factor err {err:.4f}")


if __name__ == "__main__":
    _selftest()

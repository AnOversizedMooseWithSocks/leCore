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

from holographic.agents_and_reasoning.holographic_ai import bind, unbind, bundle, random_vector

# The nine scalar factors the VSA record carries, in a fixed order (so the bundle is deterministic).
_VSA_CHANNELS = ("base_r", "base_g", "base_b", "base_a", "metallic", "roughness", "emis_r", "emis_g", "emis_b")


def _role(name, dim):
    """A deterministic ~unit role vector for a channel name (hashlib-seeded, so it is reproducible and does not
    depend on Python's salted hash())."""
    seed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big") % (2 ** 32)
    return random_vector(dim, np.random.default_rng(seed))


class TextureMap:
    """An image-based texture: an (H, W, C) array sampled by UV coordinates with BILINEAR interpolation -- the
    per-texel detail a factor-level material can't carry. UVs are in [0,1] (glTF convention; v runs top-down).
    `wrap` decides what happens outside [0,1]: 'repeat' tiles the image, 'clamp' holds the edge texel."""

    def __init__(self, image, wrap="repeat"):
        img = np.asarray(image, float)
        if img.ndim == 2:
            img = img[:, :, None]                            # (H,W) grayscale -> (H,W,1)
        self.image = img
        self.h, self.w, self.c = img.shape
        self.wrap = wrap

    def _wrap(self, x, n):
        # 'clamp' holds the last texel; 'repeat' wraps around (the two conventions almost everything uses)
        return int(np.clip(x, 0, n - 1)) if self.wrap == "clamp" else int(x % n)

    def sample(self, u, v):
        """Bilinear sample at UV (u,v). Returns a (C,) vector. Uses the pixel-CENTER convention: texel i covers
        [i, i+1), so u*w-0.5 lands between the two nearest texel centres, and we blend the four that surround it."""
        fx = u * self.w - 0.5
        fy = v * self.h - 0.5
        x0 = int(np.floor(fx)); y0 = int(np.floor(fy))
        tx = fx - x0; ty = fy - y0                            # fractional position between texels (the blend weights)
        x0w, x1w = self._wrap(x0, self.w), self._wrap(x0 + 1, self.w)
        y0w, y1w = self._wrap(y0, self.h), self._wrap(y0 + 1, self.h)
        top = self.image[y0w, x0w] * (1 - tx) + self.image[y0w, x1w] * tx      # blend the top two texels
        bot = self.image[y1w, x0w] * (1 - tx) + self.image[y1w, x1w] * tx      # blend the bottom two
        return top * (1 - ty) + bot * ty                     # blend top and bottom


class PBRMaterial:
    """A glTF 2.0 metallic-roughness PBR material. Factor level by default; optionally carries TEXTURE MAPS
    (base colour / metallic / roughness / emissive) sampled by UV. Values are in [0,1]."""

    def __init__(self, name="material", base_color=(0.8, 0.8, 0.8, 1.0), metallic=0.0, roughness=0.8,
                 emissive=(0.0, 0.0, 0.0), base_color_map=None, metallic_map=None, roughness_map=None,
                 emissive_map=None, ior=1.5, transmission=0.0, attenuation_color=(1.0, 1.0, 1.0),
                 attenuation_distance=None, fiber=False, fiber_roughness=0.10, fiber_tilt_deg=-3.0, sss=0.0,
                 temperature_K=0.0, iridescence_nm=0.0):
        self.name = str(name)
        self.base_color = tuple(float(c) for c in base_color)        # RGBA
        self.metallic = float(metallic)
        self.roughness = float(roughness)
        self.emissive = tuple(float(c) for c in emissive)            # RGB
        # optional TextureMaps -- glTF multiplies the factor by the sampled texture, so a None map leaves the
        # factor unchanged (fully backward compatible: a material with no maps behaves exactly as before)
        self.base_color_map = base_color_map
        self.metallic_map = metallic_map
        self.roughness_map = roughness_map
        self.emissive_map = emissive_map
        # ---- PHYSICAL properties the RENDERER needs (real glTF extensions; defaults keep old behaviour) ----
        # KHR_materials_ior: index of refraction of the surface (1.0=air, 1.33=water, 1.5=glass, 2.4=diamond)
        self.ior = float(ior)
        # KHR_materials_transmission: fraction of light that passes THROUGH (0=opaque, 1=clear dielectric/glass)
        self.transmission = float(transmission)
        # KHR_materials_volume: Beer-Lambert absorption inside a transmissive body -- the colour the transmitted
        # light tends toward, and the distance over which it is absorbed (None = no volumetric tint)
        self.attenuation_color = tuple(float(c) for c in attenuation_color)
        self.attenuation_distance = None if attenuation_distance is None else float(attenuation_distance)
        # ---- FIBER (hair/fur) descriptor: a Marschner strand BSDF is not a surface BRDF, so it carries its own
        # physical params. `fiber` flags the material as hair; the base_color drives absorption (dark hair absorbs
        # more), fiber_roughness is the longitudinal lobe width, fiber_tilt_deg the cuticle scale tilt. ----
        self.fiber = bool(fiber)
        self.fiber_roughness = float(fiber_roughness)
        self.fiber_tilt_deg = float(fiber_tilt_deg)
        # SUBSURFACE strength (0 = opaque; >0 = translucent, glows where the object is thin toward the light --
        # wax/jade/skin/marble). Read by holographic_matlib.shade and consumed by path_trace's SSS term.
        self.sss = float(sss)
        # TEMPERATURE (Kelvin, 0 = not glowing). A hot material EMITS blackbody radiation whose colour is set by
        # its temperature (Planck's law): ~800K dull red, ~1200K orange, ~3000K yellow-white, ~6000K white. shade()
        # turns this into the emission colour via holographic_blackbody, so "glowing hot metal" is a physical
        # property of the material, not a hand-typed emissive colour.
        self.temperature_K = float(temperature_K)
        # IRIDESCENCE film thickness in NANOMETRES (0 = not iridescent). A thin transparent film (soap ~300 nm,
        # oil ~450 nm) on the surface produces a rainbow sheen that shifts with the view angle -- thin-film
        # interference. The path tracer reads this and tints the reflection via holographic_thinfilm.
        self.iridescence_nm = float(iridescence_nm)

    # -- alias: `emission` <-> `emissive` ---------------------------------------------------------------------
    # WHY: half the surrounding vocabulary (Blender's node, most UIs, users' instinct) says "emission"; glTF and
    # this class say "emissive". Writing `m.emission = ...` used to silently create a DEAD attribute the renderer
    # never reads -- the single most time-wasting naming mismatch in the Poly Studio build. A property makes the
    # wrong-but-obvious name a working synonym instead of a silent no-op. Backward compatible: `emissive` stays
    # the stored name, serialization untouched.
    @property
    def emission(self):
        """Synonym for `emissive` (RGB). Reads and writes the same stored value."""
        return self.emissive

    @emission.setter
    def emission(self, value):
        self.emissive = tuple(float(c) for c in value)

    def sample(self, u=0.5, v=0.5):
        """The effective PBR values at UV (u,v): each channel is its factor multiplied by its texture map where one
        is set (glTF's factor x texture rule). With no maps this returns the factor-level values, so old callers
        are unaffected. Returns a dict {base_color (RGBA), metallic, roughness, emissive (RGB)}."""
        bc = np.array(self.base_color, float)
        if self.base_color_map is not None:
            s = np.asarray(self.base_color_map.sample(u, v), float)
            s = np.concatenate([s, np.ones(4 - len(s))]) if len(s) < 4 else s[:4]   # pad to RGBA if grayscale/RGB
            bc = bc * s
        metallic = self.metallic * (float(self.metallic_map.sample(u, v)[0]) if self.metallic_map is not None else 1.0)
        roughness = self.roughness * (float(self.roughness_map.sample(u, v)[0]) if self.roughness_map is not None else 1.0)
        em = np.array(self.emissive, float)
        if self.emissive_map is not None:
            se = np.asarray(self.emissive_map.sample(u, v), float)[:3]
            em = em * (se if len(se) == 3 else np.full(3, se[0]))
        return {"base_color": tuple(float(x) for x in bc), "metallic": metallic,
                "roughness": roughness, "emissive": tuple(float(x) for x in em),
                "ior": self.ior, "transmission": self.transmission,
                "attenuation_color": self.attenuation_color, "attenuation_distance": self.attenuation_distance,
                "fiber": self.fiber, "fiber_roughness": self.fiber_roughness, "fiber_tilt_deg": self.fiber_tilt_deg}

    def __repr__(self):
        return (f"PBRMaterial(name={self.name!r}, base_color={tuple(round(c,3) for c in self.base_color)}, "
                f"metallic={self.metallic:.3f}, roughness={self.roughness:.3f}, "
                f"emissive={tuple(round(c,3) for c in self.emissive)})")

    # ----- glTF -------------------------------------------------------------------------------------
    def to_gltf_dict(self):
        """The glTF 2.0 `materials[]` entry (pbrMetallicRoughness + emissiveFactor), plus the standard KHR
        extension blocks for the physical properties when they differ from the defaults (ior / transmission /
        volume-absorption). Opaque default materials emit exactly the old entry -- backward compatible."""
        d = {
            "name": self.name,
            "pbrMetallicRoughness": {
                "baseColorFactor": list(self.base_color),
                "metallicFactor": self.metallic,
                "roughnessFactor": self.roughness,
            },
            "emissiveFactor": list(self.emissive),
        }
        ext = {}
        if abs(self.ior - 1.5) > 1e-9:
            ext["KHR_materials_ior"] = {"ior": self.ior}
        if self.transmission > 0.0:
            ext["KHR_materials_transmission"] = {"transmissionFactor": self.transmission}
        if self.attenuation_distance is not None or any(abs(c - 1.0) > 1e-9 for c in self.attenuation_color):
            vol = {"attenuationColor": list(self.attenuation_color)}
            if self.attenuation_distance is not None:
                vol["attenuationDistance"] = self.attenuation_distance
            ext["KHR_materials_volume"] = vol
        if ext:
            d["extensions"] = ext
        return d

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
    from holographic.io_and_interop.holographic_encoders import ScalarEncoder
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

"""holographic_matlib.py -- a comprehensive RENDER material library: plain diffuse -> a fractal planet.

WHY THIS MODULE EXISTS
----------------------
The engine has the render *machinery* -- a metallic-roughness PBR shader (holographic_brdf), a VSA-native
`Material` record of texture fields (holographic_material), the glTF-standard `PBRMaterial` factor set
(holographic_materialio), fractal noise (holographic_noise), terrain, and region fields that layer by
priority (holographic_regionfield, whose `layered_sphere` already slices a planet into core/mantle/crust).
What it did NOT have is a *library* -- a broad, ready-to-use catalog a user can reach for, and the builders
that climb from a single flat colour all the way up to a whole procedural world. This module is that.

THE RANGE (Moose's brief), each tier built on the one below
-----------------------------------------------------------
  1. PRESETS      -- ~130 named materials as glTF metallic-roughness factors: diffuse, plastic, metal,
                     wood, stone, glass/gem, organic, fabric, liquid, emissive, ground, biome, planetary
                     layer, and ore/mineral deposit classes. `material(name)` -> PBRMaterial.
  2. PROCEDURAL   -- fractal-noise-driven albedo/roughness that vary over space (marble, granite, rust);
                     returned as a callable socket usable directly as a region's material.
  3. BLEND/LAYER  -- lerp two presets, or mask one over another by a noise field (weathering, moss, wear).
  4. BIOMES       -- a Whittaker-style classifier: (elevation, temperature, moisture) -> a surface
                     material, so a sphere's skin paints itself ocean/desert/forest/tundra/ice by climate.
  5. PLANET       -- `fractal_planet(...)`: ONE RegionField whose crust is a fBm-DISPLACED sphere painted
                     by biomes, wrapped around interior shells (crust/mantle/outer core/inner core), with
                     ORE/MINERAL DEPOSITS placed as high-priority noise-blob pockets inside their host
                     layer. Slice it and the layers, pockets, and biome-tinted surface all show at once.

'AS ABOVE, SO BELOW': a planet is a bundle of biomes over a stack of layers; a layer can hold deposits;
every one of these is the same region-with-a-material primitive at a different scale. No new machinery --
the planet is just region composition all the way down, which is exactly why it is cheap once the parts
exist.

HONEST SCOPE (kept negatives)
-----------------------------
  * The preset numbers are HAND-AUTHORED, physically plausible metallic-roughness values, not measured
    spectra -- a good artist default set, not a spectrophotometer. Dielectric transmission (glass/gems)
    is carried as an `alpha`/`ior` hint on the preset; the base `PBRMaterial` factor set stays glTF-exact.
  * The planet's noise is fBm sampled on the sphere -- believable relief and biome bands, NOT a plate-
    tectonic or climate simulation. Deposits are noise-threshold pockets, not a geological genesis model.
  * fBm queries are per-point Python loops, so cross-sections are rasterised at modest resolution; this is
    the honest Python-loop cost, mitigated by keeping the noise dim/octaves small and the raster coarse.

Deterministic (hashlib/seeded RNG, fBm is seed-reproducible); NumPy + stdlib only.
"""

import struct
import zlib

import numpy as np

from holographic_materialio import PBRMaterial


# ==================================================================================================
# TIER 1 -- the preset catalog. Each entry: (class, base_rgb, metallic, roughness, emissive_or_None,
# alpha). Values in [0,1]; alpha < 1 marks a dielectric that transmits (glass/gems/liquids). These are
# hand-authored artist defaults, physically plausible in the metallic-roughness model.
# ==================================================================================================

# (class, (r,g,b), metallic, roughness, emissive|None, alpha)
RENDER_MATERIALS = {
    # --- iridescent thin-film materials (soap bubble / oil slick / beetle shell) ---------------------
    # base colour is dark so the rainbow SHEEN dominates; the film thickness (nm) is set in _IRIDESCENT below.
    "soap_bubble":  ("diffuse", (0.05, 0.05, 0.06), 0.0, 0.06, None, 1.0),
    "oil_slick":    ("diffuse", (0.03, 0.03, 0.04), 0.1, 0.10, None, 1.0),
    "beetle_shell": ("metal",   (0.10, 0.10, 0.10), 1.0, 0.18, None, 1.0),
    # --- plain diffuse / dielectric mattes ---------------------------------------------------------
    "matte_white":  ("diffuse", (0.90, 0.90, 0.90), 0.0, 0.90, None, 1.0),
    "matte_gray":   ("diffuse", (0.50, 0.50, 0.50), 0.0, 0.90, None, 1.0),
    "matte_black":  ("diffuse", (0.03, 0.03, 0.03), 0.0, 0.90, None, 1.0),
    "chalk":        ("diffuse", (0.95, 0.95, 0.93), 0.0, 0.96, None, 1.0),
    "clay":         ("diffuse", (0.72, 0.55, 0.45), 0.0, 0.85, None, 1.0),
    "plaster":      ("diffuse", (0.92, 0.90, 0.86), 0.0, 0.88, None, 1.0),
    "concrete":     ("diffuse", (0.55, 0.55, 0.53), 0.0, 0.80, None, 1.0),
    "cardboard":    ("diffuse", (0.72, 0.60, 0.44), 0.0, 0.90, None, 1.0),
    "paper":        ("diffuse", (0.95, 0.95, 0.93), 0.0, 0.85, None, 1.0),
    # --- rubber / plastic --------------------------------------------------------------------------
    "rubber_black": ("plastic", (0.05, 0.05, 0.05), 0.0, 0.90, None, 1.0),
    "rubber_red":   ("plastic", (0.40, 0.03, 0.03), 0.0, 0.85, None, 1.0),
    "plastic_white":("plastic", (0.90, 0.90, 0.90), 0.0, 0.35, None, 1.0),
    "plastic_red":  ("plastic", (0.70, 0.05, 0.05), 0.0, 0.30, None, 1.0),
    "plastic_green":("plastic", (0.05, 0.50, 0.10), 0.0, 0.30, None, 1.0),
    "plastic_blue": ("plastic", (0.05, 0.15, 0.60), 0.0, 0.30, None, 1.0),
    "plastic_black":("plastic", (0.02, 0.02, 0.02), 0.0, 0.30, None, 1.0),
    # --- ceramic / porcelain -----------------------------------------------------------------------
    "ceramic_white":("ceramic", (0.90, 0.90, 0.88), 0.0, 0.15, None, 1.0),
    "porcelain":    ("ceramic", (0.93, 0.93, 0.90), 0.0, 0.10, None, 1.0),
    "tile_glazed":  ("ceramic", (0.80, 0.80, 0.85), 0.0, 0.12, None, 1.0),
    # --- metals (metallic = 1) ---------------------------------------------------------------------
    "iron":         ("metal", (0.56, 0.57, 0.58), 1.0, 0.35, None, 1.0),
    "steel":        ("metal", (0.66, 0.67, 0.69), 1.0, 0.25, None, 1.0),
    "steel_brushed":("metal", (0.64, 0.65, 0.67), 1.0, 0.45, None, 1.0),
    "aluminum":     ("metal", (0.91, 0.92, 0.92), 1.0, 0.30, None, 1.0),
    "chrome":       ("metal", (0.55, 0.56, 0.57), 1.0, 0.05, None, 1.0),
    "silver":       ("metal", (0.97, 0.96, 0.91), 1.0, 0.08, None, 1.0),
    "gold":         ("metal", (1.00, 0.77, 0.34), 1.0, 0.15, None, 1.0),
    "copper":       ("metal", (0.95, 0.64, 0.54), 1.0, 0.20, None, 1.0),
    "brass":        ("metal", (0.91, 0.73, 0.35), 1.0, 0.25, None, 1.0),
    "bronze":       ("metal", (0.80, 0.50, 0.30), 1.0, 0.30, None, 1.0),
    "titanium":     ("metal", (0.62, 0.60, 0.59), 1.0, 0.35, None, 1.0),
    "nickel":       ("metal", (0.66, 0.61, 0.53), 1.0, 0.25, None, 1.0),
    "lead":         ("metal", (0.53, 0.54, 0.56), 1.0, 0.50, None, 1.0),
    "rust":         ("metal", (0.45, 0.22, 0.12), 0.3, 0.85, None, 1.0),
    # --- wood --------------------------------------------------------------------------------------
    "wood_oak":     ("wood", (0.55, 0.40, 0.24), 0.0, 0.60, None, 1.0),
    "wood_pine":    ("wood", (0.72, 0.56, 0.36), 0.0, 0.60, None, 1.0),
    "wood_walnut":  ("wood", (0.30, 0.20, 0.12), 0.0, 0.55, None, 1.0),
    "wood_mahogany":("wood", (0.40, 0.16, 0.10), 0.0, 0.50, None, 1.0),
    "wood_bamboo":  ("wood", (0.78, 0.68, 0.40), 0.0, 0.55, None, 1.0),
    "wood_ebony":   ("wood", (0.06, 0.05, 0.05), 0.0, 0.50, None, 1.0),
    # --- stone -------------------------------------------------------------------------------------
    "granite":      ("stone", (0.52, 0.50, 0.48), 0.0, 0.70, None, 1.0),
    "granite_pink": ("stone", (0.72, 0.55, 0.50), 0.0, 0.65, None, 1.0),
    "marble":       ("stone", (0.90, 0.89, 0.86), 0.0, 0.35, None, 1.0),
    "basalt":       ("stone", (0.20, 0.20, 0.22), 0.0, 0.70, None, 1.0),
    "sandstone":    ("stone", (0.78, 0.66, 0.46), 0.0, 0.75, None, 1.0),
    "limestone":    ("stone", (0.82, 0.80, 0.72), 0.0, 0.70, None, 1.0),
    "slate":        ("stone", (0.30, 0.32, 0.35), 0.0, 0.55, None, 1.0),
    "obsidian":     ("stone", (0.04, 0.04, 0.05), 0.0, 0.15, None, 1.0),
    # --- glass / gems (dielectric, alpha < 1 = transmits) ------------------------------------------
    "glass_clear":  ("glass", (0.95, 0.97, 0.97), 0.0, 0.02, None, 0.10),
    "glass_frosted":("glass", (0.90, 0.93, 0.93), 0.0, 0.40, None, 0.40),
    "glass_tinted": ("glass", (0.40, 0.55, 0.50), 0.0, 0.05, None, 0.25),
    "ice":          ("glass", (0.85, 0.92, 0.97), 0.0, 0.15, None, 0.50),
    "diamond":      ("gem", (0.98, 0.98, 0.98), 0.0, 0.00, None, 0.05),
    "ruby":         ("gem", (0.70, 0.05, 0.10), 0.0, 0.05, None, 0.20),
    "emerald":      ("gem", (0.05, 0.60, 0.25), 0.0, 0.05, None, 0.20),
    "sapphire":     ("gem", (0.06, 0.15, 0.60), 0.0, 0.05, None, 0.20),
    "amethyst":     ("gem", (0.50, 0.30, 0.70), 0.0, 0.08, None, 0.25),
    "quartz":       ("gem", (0.90, 0.90, 0.92), 0.0, 0.10, None, 0.30),
    "jade":         ("gem", (0.35, 0.60, 0.42), 0.0, 0.30, None, 0.40),
    # --- organic -----------------------------------------------------------------------------------
    "skin_light":   ("organic", (0.85, 0.60, 0.50), 0.0, 0.50, None, 1.0),
    "skin_dark":    ("organic", (0.35, 0.22, 0.16), 0.0, 0.50, None, 1.0),
    "leather":      ("organic", (0.40, 0.25, 0.16), 0.0, 0.60, None, 1.0),
    "wax":          ("organic", (0.90, 0.85, 0.70), 0.0, 0.40, None, 1.0),
    "bone":         ("organic", (0.90, 0.88, 0.80), 0.0, 0.60, None, 1.0),
    "flesh":        ("organic", (0.60, 0.20, 0.20), 0.0, 0.55, None, 1.0),
    # --- fabric ------------------------------------------------------------------------------------
    "cotton":       ("fabric", (0.85, 0.85, 0.82), 0.0, 0.90, None, 1.0),
    "denim":        ("fabric", (0.20, 0.30, 0.50), 0.0, 0.85, None, 1.0),
    "silk":         ("fabric", (0.85, 0.80, 0.70), 0.0, 0.40, None, 1.0),
    "velvet_red":   ("fabric", (0.40, 0.03, 0.06), 0.0, 0.80, None, 1.0),
    "wool":         ("fabric", (0.70, 0.68, 0.60), 0.0, 0.90, None, 1.0),
    "canvas":       ("fabric", (0.78, 0.72, 0.58), 0.0, 0.88, None, 1.0),
    # --- liquids -----------------------------------------------------------------------------------
    "water":        ("liquid", (0.02, 0.08, 0.12), 0.0, 0.05, None, 0.40),
    "water_deep":   ("liquid", (0.01, 0.04, 0.09), 0.0, 0.03, None, 0.60),
    "oil":          ("liquid", (0.05, 0.04, 0.02), 0.0, 0.10, None, 1.0),
    "honey":        ("liquid", (0.60, 0.35, 0.05), 0.0, 0.15, None, 0.50),
    "milk":         ("liquid", (0.95, 0.95, 0.92), 0.0, 0.40, None, 1.0),
    "mud":          ("liquid", (0.25, 0.18, 0.12), 0.0, 0.80, None, 1.0),
    "blood":        ("liquid", (0.35, 0.02, 0.02), 0.0, 0.30, None, 1.0),
    # --- emissive ----------------------------------------------------------------------------------
    "lamp_warm":    ("emissive", (0.90, 0.80, 0.60), 0.0, 0.60, (1.00, 0.85, 0.60), 1.0),
    "lamp_cool":    ("emissive", (0.80, 0.85, 0.90), 0.0, 0.60, (0.80, 0.90, 1.00), 1.0),
    "neon_pink":    ("emissive", (0.30, 0.05, 0.20), 0.0, 0.40, (1.00, 0.10, 0.60), 1.0),
    "neon_blue":    ("emissive", (0.05, 0.15, 0.30), 0.0, 0.40, (0.10, 0.60, 1.00), 1.0),
    "neon_green":   ("emissive", (0.05, 0.30, 0.10), 0.0, 0.40, (0.20, 1.00, 0.30), 1.0),
    "led_white":    ("emissive", (0.95, 0.95, 0.95), 0.0, 0.30, (1.00, 1.00, 1.00), 1.0),
    "lava":         ("emissive", (0.20, 0.05, 0.02), 0.0, 0.80, (1.00, 0.35, 0.05), 1.0),
    "ember":        ("emissive", (0.15, 0.04, 0.02), 0.0, 0.85, (1.00, 0.30, 0.05), 1.0),
    "plasma":       ("emissive", (0.10, 0.05, 0.20), 0.0, 0.30, (0.60, 0.40, 1.00), 1.0),
    "sun_surface":  ("emissive", (1.00, 0.90, 0.60), 0.0, 1.00, (1.00, 0.85, 0.45), 1.0),
    # --- ground / nature ---------------------------------------------------------------------------
    "grass":        ("ground", (0.20, 0.45, 0.12), 0.0, 0.85, None, 1.0),
    "grass_dry":    ("ground", (0.55, 0.50, 0.22), 0.0, 0.85, None, 1.0),
    "dirt":         ("ground", (0.35, 0.25, 0.16), 0.0, 0.90, None, 1.0),
    "sand":         ("ground", (0.76, 0.68, 0.48), 0.0, 0.85, None, 1.0),
    "sand_red":     ("ground", (0.60, 0.32, 0.18), 0.0, 0.85, None, 1.0),
    "snow":         ("ground", (0.95, 0.96, 0.98), 0.0, 0.50, None, 1.0),
    "moss":         ("ground", (0.18, 0.35, 0.15), 0.0, 0.80, None, 1.0),
    "rock":         ("ground", (0.40, 0.38, 0.35), 0.0, 0.70, None, 1.0),
    "gravel":       ("ground", (0.45, 0.43, 0.40), 0.0, 0.80, None, 1.0),
    "ash":          ("ground", (0.30, 0.30, 0.30), 0.0, 0.90, None, 1.0),
    # --- biome surfaces (planetary skin) -----------------------------------------------------------
    "ocean":        ("biome", (0.02, 0.15, 0.35), 0.0, 0.05, None, 1.0),
    "ocean_deep":   ("biome", (0.01, 0.06, 0.20), 0.0, 0.03, None, 1.0),
    "beach":        ("biome", (0.82, 0.74, 0.55), 0.0, 0.80, None, 1.0),
    "desert":       ("biome", (0.80, 0.68, 0.42), 0.0, 0.85, None, 1.0),
    "savanna":      ("biome", (0.55, 0.50, 0.25), 0.0, 0.85, None, 1.0),
    "grassland":    ("biome", (0.30, 0.50, 0.18), 0.0, 0.85, None, 1.0),
    "shrubland":    ("biome", (0.42, 0.44, 0.24), 0.0, 0.85, None, 1.0),
    "forest":       ("biome", (0.12, 0.32, 0.12), 0.0, 0.80, None, 1.0),
    "rainforest":   ("biome", (0.06, 0.28, 0.10), 0.0, 0.80, None, 1.0),
    "taiga":        ("biome", (0.16, 0.30, 0.20), 0.0, 0.80, None, 1.0),
    "tundra":       ("biome", (0.55, 0.55, 0.48), 0.0, 0.80, None, 1.0),
    "bare_rock":    ("biome", (0.40, 0.38, 0.35), 0.0, 0.70, None, 1.0),
    "mountain_snow":("biome", (0.90, 0.92, 0.95), 0.0, 0.50, None, 1.0),
    "polar_ice":    ("biome", (0.90, 0.94, 0.98), 0.0, 0.30, None, 1.0),
    # --- planetary interior layers -----------------------------------------------------------------
    "crust_rock":   ("layer", (0.42, 0.34, 0.26), 0.0, 0.75, None, 1.0),
    "oceanic_crust":("layer", (0.20, 0.22, 0.24), 0.0, 0.70, None, 1.0),
    "mantle":       ("layer", (0.55, 0.22, 0.10), 0.0, 0.60, None, 1.0),
    "mantle_lower": ("layer", (0.65, 0.28, 0.10), 0.0, 0.60, None, 1.0),
    "outer_core":   ("layer", (0.98, 0.75, 0.30), 0.0, 0.50, (0.40, 0.18, 0.02), 1.0),
    "inner_core":   ("layer", (1.00, 0.90, 0.55), 0.0, 0.40, (0.50, 0.28, 0.05), 1.0),
    "magma":        ("layer", (1.00, 0.40, 0.08), 0.0, 0.70, (1.00, 0.35, 0.05), 1.0),
    # --- ore / mineral deposits (pockets) ----------------------------------------------------------
    "iron_ore":     ("deposit", (0.45, 0.30, 0.24), 0.3, 0.70, None, 1.0),
    "gold_ore":     ("deposit", (0.85, 0.68, 0.30), 0.6, 0.50, None, 1.0),
    "copper_ore":   ("deposit", (0.30, 0.55, 0.45), 0.2, 0.60, None, 1.0),
    "coal":         ("deposit", (0.05, 0.05, 0.05), 0.0, 0.90, None, 1.0),
    "diamond_ore":  ("deposit", (0.90, 0.95, 0.98), 0.0, 0.10, None, 1.0),
    "crystal":      ("deposit", (0.70, 0.80, 0.95), 0.0, 0.10, None, 0.40),
    "quartz_vein":  ("deposit", (0.90, 0.90, 0.92), 0.0, 0.20, None, 1.0),
    "aquifer":      ("deposit", (0.05, 0.20, 0.40), 0.0, 0.10, None, 0.40),
    "oil_pocket":   ("deposit", (0.05, 0.04, 0.02), 0.0, 0.10, None, 1.0),
    "gas_pocket":   ("deposit", (0.60, 0.70, 0.50), 0.0, 0.20, None, 0.30),
    "magma_chamber":("deposit", (1.00, 0.35, 0.06), 0.0, 0.70, (1.00, 0.30, 0.05), 1.0),
    "salt_deposit": ("deposit", (0.90, 0.88, 0.85), 0.0, 0.60, None, 1.0),
    "uranium_ore":  ("deposit", (0.25, 0.45, 0.15), 0.1, 0.60, (0.05, 0.15, 0.03), 1.0),
    # --- fiber (hair / fur): shaded by a Marschner strand BSDF, not a surface BRDF. The rgb is the perceived
    #     hair colour (it drives absorption -- dark hair absorbs the transmitted lobes); the fiber roughness and
    #     cuticle tilt are filled in by material() from _FIBER_PHYS below. ---------------------------------------
    "fur_brown":    ("fiber", (0.36, 0.22, 0.12), 0.0, 0.30, None, 1.0),
    "fur_ginger":   ("fiber", (0.62, 0.30, 0.12), 0.0, 0.30, None, 1.0),
    "fur_gray":     ("fiber", (0.45, 0.44, 0.42), 0.0, 0.30, None, 1.0),
    "hair_blonde":  ("fiber", (0.85, 0.65, 0.35), 0.0, 0.30, None, 1.0),
    "hair_black":   ("fiber", (0.06, 0.05, 0.04), 0.0, 0.30, None, 1.0),
    "hair_brown":   ("fiber", (0.28, 0.18, 0.10), 0.0, 0.30, None, 1.0),
    "hair_red":     ("fiber", (0.50, 0.16, 0.08), 0.0, 0.30, None, 1.0),
}


def names():
    """All preset names, sorted."""
    return sorted(RENDER_MATERIALS)


def by_class(cls):
    """All preset names in a class (diffuse, metal, wood, stone, glass, gem, organic, fabric, liquid,
    emissive, ground, biome, layer, deposit)."""
    return sorted(n for n, e in RENDER_MATERIALS.items() if e[0] == cls)


def classes():
    return sorted(set(e[0] for e in RENDER_MATERIALS.values()))


def material(name):
    """Look up a preset -> a glTF-standard PBRMaterial, now carrying the PHYSICAL properties the renderer needs.
    Beyond the metallic-roughness factors, transmissive classes (glass / gem / clear liquids) get a real index of
    refraction + transmission + a volumetric attenuation colour, and fibers (hair/fur) get their strand-BSDF
    params -- so a material out of this library is the single physical source of truth the renderer reads from
    (see shade() / fiber_params()). Raises a clear KeyError naming near matches."""
    if name not in RENDER_MATERIALS:
        near = [n for n in RENDER_MATERIALS if name.split("_")[0] in n][:6]
        raise KeyError("unknown material %r%s" % (name, (" -- did you mean: %s" % near) if near else ""))
    cls, rgb, metallic, rough, emis, alpha = RENDER_MATERIALS[name]
    mat = PBRMaterial(name=name, base_color=(rgb[0], rgb[1], rgb[2], alpha),
                      metallic=metallic, roughness=rough, emissive=emis or (0.0, 0.0, 0.0))
    _apply_physical(mat, cls, name, rgb)
    return mat


# reference indices of refraction, keyed by material name where it matters, else a per-class default
_IOR = {"diamond": 2.42, "sapphire": 1.77, "ruby": 1.77, "emerald": 1.58, "amethyst": 1.55, "quartz": 1.55,
        "jade": 1.66, "ice": 1.31, "water": 1.33, "water_deep": 1.33, "oil": 1.47, "honey": 1.50}
_CLASS_IOR = {"glass": 1.50, "gem": 1.55, "liquid": 1.33}
_TRANSMISSIVE_LIQUIDS = {"water", "water_deep", "oil", "honey"}     # the rest (milk/blood/mud) read as opaque
# fiber strand-BSDF params (longitudinal roughness = lobe width; cuticle tilt in degrees), keyed by name
_FIBER_PHYS = {"fur_brown": (0.28, -4.0), "fur_ginger": (0.26, -4.0), "fur_gray": (0.30, -4.0),
               "hair_blonde": (0.15, -3.0), "hair_black": (0.12, -3.0), "hair_brown": (0.16, -3.0),
               "hair_red": (0.18, -3.0)}


# subsurface strength for translucent materials (0 = opaque). Wax/skin glow softly; jade/marble a bit; honey/milk too.
_SSS = {"wax": 1.0, "skin_light": 0.9, "skin_dark": 0.7, "jade": 0.8, "marble": 0.5, "milk": 0.9,
        "honey": 0.6, "flesh": 0.9, "leaf": 1.0}

# IRIDESCENT thin-film materials: film thickness in NANOMETRES (0 = not iridescent). Soap ~ 300 nm, oil slick a
# bit thicker, a beetle's cuticle in the strong-colour band. The path tracer tints the reflection by view angle.
_IRIDESCENT = {"soap_bubble": 300.0, "oil_slick": 420.0, "beetle_shell": 250.0}


def _apply_physical(mat, cls, name, rgb):
    """Set the physical dielectric / fiber / subsurface properties on a freshly-built preset, by class and (where
    it matters) by name. Transmissive materials refract; fibers carry their strand params; translucent materials
    get a subsurface strength. Opaque presets are untouched."""
    # a named subsurface material is a translucent SOLID (wax/jade/skin), not refractive glass -- SSS instead of
    # transmission, so it takes precedence over the gem/liquid class default.
    translucent_sss = name in _SSS
    transmissive = (not translucent_sss) and ((cls in ("glass", "gem")) or
                                              (cls == "liquid" and name in _TRANSMISSIVE_LIQUIDS))
    if transmissive:
        mat.transmission = 1.0
        mat.ior = _IOR.get(name, _CLASS_IOR.get(cls, 1.5))
        mat.attenuation_color = tuple(float(c) for c in rgb)     # the tint transmitted light picks up
    if cls == "fiber":
        mat.fiber = True
        r, tilt = _FIBER_PHYS.get(name, (0.20, -3.0))
        mat.fiber_roughness = r
        mat.fiber_tilt_deg = tilt
    if name in _SSS:
        mat.sss = _SSS[name]                                     # translucent -> glows where thin (path_trace SSS term)
    if name in _IRIDESCENT:
        mat.iridescence_nm = _IRIDESCENT[name]                   # thin film -> rainbow sheen (path_trace irid term)


# --------------------------------------------------------------------------- renderer adapters
def shade(mat, n):
    """Hand the path tracer its per-hit tuple (albedo(n,3), metallic(n,), roughness(n,), emission(n,3), ior(n,))
    for `n` points all of material `mat` -- physical data read straight off the material instead of a hand-typed
    tuple. A TRANSMISSIVE material reports ior=mat.ior (the tracer's smooth-dielectric BSDF then refracts) and its
    'albedo' is the attenuation colour (the coloured-glass tint on transmitted light); an OPAQUE material reports
    ior=0 and is shaded by the GGX/diffuse BRDF from base_color / metallic / roughness."""
    base = np.array(mat.base_color[:3], float)
    transmissive = getattr(mat, "transmission", 0.0) > 0.0
    alb = np.tile(np.array(mat.attenuation_color, float) if transmissive else base, (n, 1)).astype(float)
    met = np.full(n, 0.0 if transmissive else mat.metallic)
    rough = np.full(n, mat.roughness)
    emis = np.tile(np.array(mat.emissive, float), (n, 1)).astype(float)
    # HOT material: a temperature > 0 EMITS blackbody radiation -- colour from Planck's law, brightness rising
    # steeply with temperature (Stefan-Boltzmann is ~T^4; we use a gentler visible-range curve so it tonemaps).
    # This makes "glowing hot metal" a physical property (temperature) rather than a hand-typed emissive colour.
    T = float(getattr(mat, "temperature_K", 0.0))
    if T > 500.0:
        from holographic_blackbody import blackbody_rgb
        col = np.array(blackbody_rgb(T), float)                 # the hue for this temperature (deep red -> white)
        bright = ((T - 500.0) / 1500.0) ** 2                    # visible glow ramps from ~500K, quadratic in T
        emis = emis + col * bright                              # add the thermal glow on top of any base emissive
    ior = np.full(n, mat.ior if transmissive else 0.0)
    sss = float(getattr(mat, "sss", 0.0))
    irid = float(getattr(mat, "iridescence_nm", 0.0))
    # IRIDESCENT material: a thin film (nm) on the surface -> the path tracer tints the reflection by view angle.
    # Emit the full 7-tuple whenever a film is present, carrying sss alongside (0 if the surface isn't translucent).
    if not transmissive and irid > 0.0:
        return alb, met, rough, emis, ior, np.full(n, sss), np.full(n, irid)
    if not transmissive and sss > 0.0:
        return alb, met, rough, emis, ior, np.full(n, sss)      # translucent opaque -> carry the subsurface strength
    return alb, met, rough, emis, ior


def iridesce(name_or_mat, thickness_nm):
    """Return a copy of a material with a thin iridescent film of `thickness_nm` nanometres -- it will show a
    rainbow sheen that shifts with the view angle (soap ~300 nm, oil ~420 nm). A physical way to make anything
    iridescent: iridesce('steel', 350) is oil-on-metal. Accepts a material name or an already-built material."""
    import copy
    mat = material(name_or_mat) if isinstance(name_or_mat, str) else copy.deepcopy(name_or_mat)
    mat.iridescence_nm = float(thickness_nm)
    return mat


def heat(name_or_mat, temperature_K):
    """Return a copy of a material heated to `temperature_K` -- it will EMIT blackbody radiation of the colour that
    temperature implies (see shade). A physical way to make anything glow: heat('iron', 1400) is orange-hot iron,
    heat('iron', 900) is a dull red. Accepts a material name or an already-built material."""
    import copy
    mat = material(name_or_mat) if isinstance(name_or_mat, str) else copy.deepcopy(name_or_mat)
    mat.temperature_K = float(temperature_K)
    return mat


def fiber_params(mat):
    """The physical strand parameters the Marschner hair shader needs, read off a fiber material: the hair COLOUR
    (drives absorption -- dark hair absorbs the transmitted TT/TRT lobes), the longitudinal roughness (lobe width),
    and the cuticle tilt (degrees). Raises if the material isn't a fiber."""
    if not getattr(mat, "fiber", False):
        raise ValueError("%r is not a fiber material (its class must be 'fiber')" % mat.name)
    return {"hair_color": tuple(float(c) for c in mat.base_color[:3]),
            "roughness": mat.fiber_roughness, "tilt_deg": mat.fiber_tilt_deg}


def albedo(name):
    """Just the base rgb of a preset (for socket-level colour work)."""
    return np.asarray(RENDER_MATERIALS[name][1], float)


def catalog():
    """{class: [names...]} -- the whole library grouped, for a UI picker."""
    out = {}
    for n, e in RENDER_MATERIALS.items():
        out.setdefault(e[0], []).append(n)
    for k in out:
        out[k].sort()
    return out


# ==================================================================================================
# TIER 2/3 -- procedural (noise-driven) and blended albedo sockets. A socket is f(points)->(M,3) rgb,
# exactly what a region's `material` accepts, so procedural materials drop straight into a RegionField.
# ==================================================================================================

def _fbm_batch(noise, points):
    """Query a FractalNoise at many points (per-point Python loop -- the honest cost), returned in [0,1]."""
    pts = np.atleast_2d(np.asarray(points, float))
    raw = np.array([noise.query(p) for p in pts])
    # fBm output is roughly symmetric around 0; squash to [0,1] for use as a blend factor
    return 0.5 + 0.5 * np.clip(raw / (np.abs(raw).max() + 1e-9), -1.0, 1.0)


def noise_blend_albedo(color_a, color_b, noise, lo=0.0, hi=1.0):
    """A procedural albedo: lerp between two colours by a fBm field (marble/granite/rust look). Returns a
    socket f(points)->rgb. color_a/color_b may be preset names or rgb triples."""
    ca = albedo(color_a) if isinstance(color_a, str) else np.asarray(color_a, float)
    cb = albedo(color_b) if isinstance(color_b, str) else np.asarray(color_b, float)

    def _socket(points):
        t = _fbm_batch(noise, points)
        t = np.clip((t - lo) / max(hi - lo, 1e-6), 0.0, 1.0)[:, None]
        return (1.0 - t) * ca + t * cb
    return _socket


def mask_blend_albedo(base, overlay, noise, threshold=0.5, softness=0.1):
    """Mask `overlay` over `base` where a fBm field exceeds a threshold (weathering, moss, wear, veins).
    Returns a socket f(points)->rgb."""
    cb = albedo(base) if isinstance(base, str) else np.asarray(base, float)
    co = albedo(overlay) if isinstance(overlay, str) else np.asarray(overlay, float)

    def _socket(points):
        n = _fbm_batch(noise, points)
        m = np.clip((n - threshold) / max(softness, 1e-6), 0.0, 1.0)[:, None]
        return (1.0 - m) * cb + m * co
    return _socket


def blend_presets(name_a, name_b, t):
    """Lerp two PRESETS at the factor level -> a new PBRMaterial (constant blend, e.g. worn gold)."""
    a, b = material(name_a), material(name_b)
    lerp = lambda x, y: tuple((1 - t) * xi + t * yi for xi, yi in zip(x, y))
    return PBRMaterial(name="%s_%s_%.2f" % (name_a, name_b, t),
                       base_color=lerp(a.base_color, b.base_color),
                       metallic=(1 - t) * a.metallic + t * b.metallic,
                       roughness=(1 - t) * a.roughness + t * b.roughness,
                       emissive=lerp(a.emissive, b.emissive))


# ==================================================================================================
# TIER 4 -- biomes. A Whittaker-style classifier: given elevation (relative to sea level), temperature
# (0 cold .. 1 hot) and moisture (0 dry .. 1 wet), pick a surface material NAME from the catalog.
# ==================================================================================================

def biome_at(elevation, temperature, moisture, sea_level=0.0):
    """Elevation/temperature/moisture -> a biome preset name. Ordered rules: water first, then a coastal
    band, then land by climate (Whittaker cells), with cold/high overrides for snow and ice."""
    e = elevation - sea_level
    if e < -0.12:
        return "ocean_deep"
    if e < 0.0:
        return "ocean"
    if e < 0.03:
        return "beach"
    # land: cold/high overrides
    if temperature < 0.15 or e > 0.55:
        return "polar_ice" if temperature < 0.08 else ("mountain_snow" if e > 0.45 else "tundra")
    if temperature < 0.35:
        return "taiga" if moisture > 0.4 else "tundra"
    if temperature < 0.7:
        if moisture > 0.6:
            return "forest"
        return "grassland" if moisture > 0.3 else "shrubland"
    # hot
    if moisture > 0.66:
        return "rainforest"
    return "savanna" if moisture > 0.33 else "desert"


# ==================================================================================================
# TIER 5 -- the fractal planet. ONE RegionField: a fBm-displaced surface sphere painted by biomes, over
# interior shells, with ore/mineral deposits as high-priority noise-blob pockets inside a host layer.
# ==================================================================================================

DEFAULT_LAYERS = [   # (relative outer radius, preset name); innermost wins by priority
    (1.00, "crust_rock"),
    (0.90, "mantle"),
    (0.55, "mantle_lower"),
    (0.35, "outer_core"),
    (0.18, "inner_core"),
]

DEFAULT_DEPOSITS = [  # (preset, host layer index, abundance 0..1, scale) -- pockets inside a host shell
    ("iron_ore",      0, 0.35, 6.0),
    ("copper_ore",    0, 0.25, 7.0),
    ("gold_ore",      0, 0.10, 9.0),
    ("coal",          0, 0.30, 5.0),
    ("aquifer",       0, 0.25, 5.5),
    ("diamond_ore",   1, 0.06, 10.0),
    ("magma_chamber", 2, 0.20, 4.0),
]


class _PlanetSurfaceSDF:
    """A sphere whose radius is DISPLACED by an fBm elevation field along each direction -- fractal terrain
    relief on a globe. eval(P) < 0 inside the (bumpy) surface. This is the crust region's boundary."""

    def __init__(self, center, radius, noise, relief):
        self.c = np.asarray(center, float)
        self.r = float(radius)
        self.noise = noise
        self.relief = float(relief)

    def elevation(self, points):
        """Elevation in [-1,1]-ish at each point's DIRECTION (sampled on the reference sphere)."""
        d = np.atleast_2d(np.asarray(points, float)) - self.c
        rad = np.linalg.norm(d, axis=1, keepdims=True)
        dirs = d / np.maximum(rad, 1e-9) * self.r      # sample the noise on the reference sphere
        raw = np.array([self.noise.query(p) for p in dirs])
        m = np.abs(raw).max() + 1e-9
        e = raw / m
        # shape: concentrate toward 0 so most terrain is gently rolling and high peaks are rare
        return np.sign(e) * np.abs(e) ** 1.6

    def eval(self, P):
        d = np.atleast_2d(np.asarray(P, float)) - self.c
        rad = np.linalg.norm(d, axis=1)
        surf = self.r * (1.0 + self.relief * self.elevation(P))
        return rad - surf


class _DepositSDF:
    """A pocket: the intersection of a host layer SHELL (between inner_r and outer_r) and a fBm 'blob'
    (where noise > threshold). The intersection SDF is max(shell_sdf, blob_sdf) -- negative only where a
    point is inside the shell AND inside the blob, so ore/mineral pockets stay within their host layer."""

    def __init__(self, center, outer_r, inner_r, noise, scale, threshold):
        self.c = np.asarray(center, float)
        self.outer = float(outer_r)
        self.inner = float(inner_r)
        self.noise = noise
        self.scale = float(scale)
        self.th = float(threshold)

    def eval(self, P):
        d = np.atleast_2d(np.asarray(P, float)) - self.c
        rad = np.linalg.norm(d, axis=1)
        n = np.array([self.noise.query(p * self.scale) for p in d])
        n = n / (np.abs(n).max() + 1e-9)                # -> ~[-1,1]
        shell_sdf = np.maximum(rad - self.outer, self.inner - rad)   # < 0 inside the [inner,outer] shell
        blob_sdf = self.th - n                          # < 0 where noise exceeds threshold
        return np.maximum(shell_sdf, blob_sdf)


class Planet:
    """A fractal planet as one RegionField: displaced biome crust + interior shells + ore deposits.

    material_at(points) -> per-point rgb (biome-painted crust on the outside; layers and pockets within).
    A cross-section through the centre reveals all three at once."""

    def __init__(self, radius=1.0, seed=0, dim=256, octaves=4, relief=0.10, sea_level=0.0,
                 layers=None, deposits=None, moisture_seed=None):
        from holographic_noise import FractalNoise
        from holographic_regionfield import Region, RegionField

        self.radius = float(radius)
        self.center = np.zeros(3)
        self.sea_level = float(sea_level)
        b = [(-radius * 1.2, radius * 1.2)] * 3
        # elevation, moisture, and deposit noise fields -- different seeds so they are independent
        self.elev_noise = FractalNoise(3, dim=dim, bounds=b, octaves=octaves, base_bandwidth=2.0, seed=seed)
        self.moist_noise = FractalNoise(3, dim=dim, bounds=b, octaves=max(2, octaves - 1),
                                        base_bandwidth=1.5, seed=(moisture_seed if moisture_seed is not None else seed + 101))
        self.dep_noise = FractalNoise(3, dim=dim, bounds=[(-radius * 8, radius * 8)] * 3,
                                      octaves=2, base_bandwidth=4.0, seed=seed + 202)

        layers = layers or DEFAULT_LAYERS
        deposits = DEFAULT_DEPOSITS if deposits is None else deposits
        self.layer_names = [nm for _, nm in layers]

        surf = _PlanetSurfaceSDF(self.center, radius, self.elev_noise, relief)
        self._surf = surf
        regions = [Region(surf, "crust", priority=0.0, material=self._biome_socket())]
        # inner shells: full balls, higher priority the deeper they are (so they win in their shell)
        for pri, (rel_r, nm) in enumerate(layers[1:], start=1):
            regions.append(Region(_SphereBall(self.center, rel_r * radius), nm,
                                   priority=float(pri), material=albedo(nm)))
        # deposits: highest priority so they show through, clipped to their host layer's SHELL and kept
        # just below the surface (a thin biome skin remains above crust deposits)
        for k, (nm, host_idx, abundance, scale) in enumerate(deposits):
            outer = layers[host_idx][0] * radius * 0.97
            inner = (layers[host_idx + 1][0] * radius) if host_idx + 1 < len(layers) else 0.0
            threshold = float(np.clip(1.0 - 2.0 * abundance, -0.9, 0.95))   # rarer -> higher threshold
            regions.append(Region(_DepositSDF(self.center, outer, inner, self.dep_noise, scale, threshold),
                                   nm, priority=100.0 + k, material=albedo(nm)))
        self.field = RegionField(regions)

    def _biome_socket(self):
        """The crust's albedo socket: each point -> its biome colour from (elevation, temperature by
        latitude+altitude, moisture)."""
        def _socket(points):
            P = np.atleast_2d(np.asarray(points, float))
            elev = self._surf.elevation(P)
            d = P - self.center
            rad = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            lat = np.abs(d[:, 2] / rad)                                   # 0 equator .. 1 pole
            raw_m = np.array([self.moist_noise.query(p) for p in d])
            moist = 0.5 + 0.5 * (raw_m / (np.abs(raw_m).max() + 1e-9))
            temp = np.clip(1.0 - 0.8 * lat - 0.4 * np.clip(elev, 0, 1), 0.0, 1.0)  # warm equator, cold poles/peaks
            out = np.zeros((len(P), 3))
            for i in range(len(P)):
                out[i] = albedo(biome_at(float(elev[i]), float(temp[i]), float(moist[i]), self.sea_level))
            return out
        return _socket

    def material_at(self, points):
        """Per-point rgb: crust biome on the outside, interior layers and deposits within."""
        P = np.atleast_2d(np.asarray(points, float))
        return self.field.material_at(P, empty=(0.0, 0.0, 0.0))

    def surface_directions(self, n):
        """n roughly-even directions on the sphere (Fibonacci lattice), for a surface render / stats."""
        i = np.arange(n) + 0.5
        phi = np.arccos(1.0 - 2.0 * i / n)
        gold = np.pi * (1.0 + 5 ** 0.5)
        theta = gold * i
        return np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1)

    def biome_histogram(self, n=256):
        """Fraction of the surface in each biome -- a cheap, verifiable summary of the generated world."""
        dirs = self.surface_directions(n) * self.radius
        rgb = self._biome_socket()(dirs)
        # map each surface point back to a biome name by nearest catalog biome colour
        biomes = by_class("biome")
        cols = np.array([albedo(b) for b in biomes])
        idx = np.argmin(((rgb[:, None, :] - cols[None, :, :]) ** 2).sum(-1), axis=1)
        hist = {}
        for j in idx:
            hist[biomes[j]] = hist.get(biomes[j], 0) + 1
        return {k: v / float(n) for k, v in sorted(hist.items(), key=lambda kv: -kv[1])}

    def cross_section(self, res=48, axis=2):
        """A res x res rgb image of a slice through the centre (plane normal = axis). Reveals the displaced
        biome crust, the interior shells, and the ore pockets in one picture."""
        g = np.linspace(-self.radius * 1.1, self.radius * 1.1, res)
        A, B = np.meshgrid(g, g)
        pts = np.zeros((res * res, 3))
        ax = [i for i in range(3) if i != axis]
        pts[:, ax[0]] = A.ravel()
        pts[:, ax[1]] = B.ravel()
        rgb = self.material_at(pts).reshape(res, res, 3)
        return np.clip(rgb, 0, 1)


class _SphereBall:
    """A plain solid-ball SDF (negative inside), used for the interior shells."""
    def __init__(self, c, r):
        self.c = np.asarray(c, float); self.r = float(r)

    def eval(self, P):
        return np.linalg.norm(np.atleast_2d(np.asarray(P, float)) - self.c, axis=1) - self.r


def fractal_planet(radius=1.0, seed=0, dim=256, octaves=4, relief=0.10, sea_level=0.0,
                   layers=None, deposits=None):
    """Build a Planet (see the class). The top of the library's range: one call yields a whole world whose
    surface, layers, and deposits are all region composition over shared noise."""
    return Planet(radius=radius, seed=seed, dim=dim, octaves=octaves, relief=relief,
                  sea_level=sea_level, layers=layers, deposits=deposits)


# ==================================================================================================
# a tiny deterministic PNG writer (stdlib zlib) so a cross-section is a verifiable deliverable
# ==================================================================================================

def write_png(path, rgb01):
    """Write an (H,W,3) float[0,1] image to a PNG using only stdlib (no PIL)."""
    img = (np.clip(np.asarray(rgb01, float), 0, 1) * 255).astype(np.uint8)
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def _chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, 9)) + _chunk(b"IEND", b""))
    return path


# ==================================================================================================
# SELF TEST
# ==================================================================================================

def _selftest():
    # 1. the catalog spans the range
    cls = classes()
    print("catalog: %d materials across %d classes" % (len(RENDER_MATERIALS), len(cls)))
    print("  classes:", ", ".join(cls))
    assert len(RENDER_MATERIALS) >= 120
    for need in ("matte_white", "gold", "glass_clear", "lava", "iron_ore", "inner_core", "forest"):
        assert need in RENDER_MATERIALS
    # plain diffuse -> a glTF material
    m = material("matte_white")
    assert m.metallic == 0.0 and m.roughness > 0.5
    # metals are metallic, emissives emit
    assert material("gold").metallic == 1.0
    assert material("lava").emissive[0] > 0.5
    # dielectric transmission carried as alpha
    assert material("glass_clear").base_color[3] < 0.2

    # 2. procedural socket returns per-point colour between two presets
    from holographic_noise import FractalNoise
    n = FractalNoise(3, dim=128, bounds=[(-1, 1)] * 3, octaves=2, base_bandwidth=3.0, seed=1)
    marble = noise_blend_albedo("marble", "slate", n)
    pts = np.random.default_rng(0).uniform(-1, 1, (20, 3))
    cols = marble(pts)
    assert cols.shape == (20, 3) and cols.min() >= 0 and cols.max() <= 1
    print("procedural marble socket: 20 points ->", tuple(round(float(x), 2) for x in cols.mean(0)), "mean rgb")

    # 3. biome classifier picks sensible cells
    assert biome_at(-0.3, 0.8, 0.5) == "ocean_deep"
    assert biome_at(0.2, 0.9, 0.1) == "desert"
    assert biome_at(0.2, 0.9, 0.9) == "rainforest"
    assert biome_at(0.7, 0.05, 0.5) == "polar_ice"
    print("biome classifier: hot+dry->desert, hot+wet->rainforest, cold->polar_ice, low->ocean")

    # 4. THE FRACTAL PLANET -- surface biomes + interior layers + deposits, all in one RegionField
    planet = fractal_planet(radius=1.0, seed=3, dim=160, octaves=3, relief=0.12)
    # surface has multiple biomes
    hist = planet.biome_histogram(n=200)
    print("\nfractal planet surface biomes:", {k: round(v, 2) for k, v in list(hist.items())[:6]})
    assert len(hist) >= 3, "a fractal world should have several biomes"
    # a slice reveals the interior: sample points at decreasing radius should hit different layers
    core = planet.material_at([[0.0, 0.0, 0.0]])[0]           # centre -> inner core (glows)
    mid = planet.material_at([[0.0, 0.0, 0.5]])[0]            # mid-depth -> a mantle
    print("centre material rgb:", tuple(round(float(x), 2) for x in core),
          "| mid-depth rgb:", tuple(round(float(x), 2) for x in mid))
    assert not np.allclose(core, mid), "layers must differ through the interior"
    # deposits appear as pockets inside the interior: scan a cross-section for deposit colours
    sec = planet.cross_section(res=64)
    dep_cols = np.array([albedo(nm) for nm in by_class("deposit")])
    flat = sec.reshape(-1, 3)
    is_dep = np.array([np.any(np.all(np.abs(c - dep_cols) < 1e-6, axis=1)) for c in flat])
    print("cross-section pixels that are an ORE/MINERAL DEPOSIT pocket: %d / %d" % (is_dep.sum(), len(flat)))
    assert is_dep.sum() > 0, "the planet should have visible deposits in a slice"

    # 5. render a verifiable cross-section PNG (reuse the slice we just analysed)
    write_png("/tmp/planet_section.png", sec)
    print("cross-section PNG written: /tmp/planet_section.png (%dx%d)" % (sec.shape[0], sec.shape[1]))

    # 6. determinism
    p2 = fractal_planet(radius=1.0, seed=3, dim=160, octaves=3, relief=0.12)
    assert np.allclose(planet.material_at([[0.0, 0.0, 0.5]]), p2.material_at([[0.0, 0.0, 0.5]]))

    print("\nOK: holographic_matlib self-test passed")


if __name__ == "__main__":
    _selftest()

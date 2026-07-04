"""holographic_materialindex.py -- one discoverable index over the engine's TWO material libraries.

The engine already ships two complementary material libraries, but they were not connected:
  * RENDER  (holographic_matlib)                 -- ~141 PBR APPEARANCE presets (metals, gems, woods, stones,
                                                    liquids, biomes...) grouped by class, each a render-ready
                                                    PBRMaterial.
  * PHYSICAL (holographic_definitions.MATERIALS) -- ~33 materials with the numbers a SOLVER / SCIENTIST needs:
                                                    density, refractive index, viscosity, Young's modulus, sound
                                                    speed, specific heat, phase.
15 materials appear in BOTH (gold, water, diamond, copper, ...). This module is the bridge + discovery layer over
them: ONE place to ask "what materials do we have?", "everything about gold" (appearance AND physics), and "find me a
clear liquid" -- WITHOUT moving or duplicating the data, which stays in its home library.

Users can supply their own materials in either home library (matlib.RENDER_MATERIALS / definitions.MATERIALS); this
index simply reflects whatever is registered there, so user data shows up here automatically.
"""
import holographic_matlib as _render
from holographic_definitions import MATERIALS as _physical

# physical field -> (unit, human label) so science output is self-describing
PHYSICAL_FIELDS = {
    "density": ("kg/m^3", "density"),
    "refractive": ("", "refractive index"),
    "viscosity": ("Pa*s", "dynamic viscosity"),
    "youngs": ("GPa", "Young's modulus"),
    "sound_speed": ("m/s", "speed of sound"),
    "specific_heat": ("J/(kg*K)", "specific heat"),
    "phase": ("", "phase"),
}


# -- membership + simple listings ----------------------------------------------------------------------------
def render_names():
    return _render.names()


def render_classes():
    return _render.classes()


def physical_names():
    return sorted(_physical)


def physical_categories():
    """The categories present in the physical library (metal/liquid/gas/polymer/ceramic/stone/wood/...)."""
    return sorted({e.get("category") for e in _physical.values() if e.get("category")})


def physical_by_category(category):
    """Physical materials in a category."""
    return sorted(n for n, e in _physical.items() if e.get("category") == category)


def physical_units():
    """Field -> (unit, description) for the physical properties -- so a value like density=19300 reads as kg/m^3."""
    from holographic_materialdata import UNITS
    return dict(UNITS)


def validate_physical():
    """Plausibility-check the physical database (right units, sane ranges, known category/phase). Empty list = clean."""
    from holographic_materialdata import validate
    return validate()


def has_render(name):
    return name in set(_render.names())


def has_physical(name):
    return name in _physical


def _class_of(name):
    """Which render class a preset belongs to (metal / gem / liquid / ...), or None if it isn't a render preset."""
    for cls in _render.classes():
        if name in _render.by_class(cls):
            return cls
    return None


# -- the two home libraries, reached through one door --------------------------------------------------------
def render_material(name):
    """The render-ready PBRMaterial for a name (delegates to matlib; raises if the name isn't a render preset)."""
    return _render.material(name)


def physical_properties(name):
    """The physical-property dict for a name (density/refractive/viscosity/youngs/sound_speed/specific_heat/phase).
    Raises KeyError if the material has no physical definition."""
    if name not in _physical:
        raise KeyError("no physical material %r (try materialindex.physical_names())" % name)
    return dict(_physical[name])


def material_info(name):
    """Everything the engine knows about a named material: its RENDER appearance (if it is a preset) AND its PHYSICAL
    properties (if defined) -- the unified 'tell me about gold' view. Raises KeyError if in neither library."""
    in_r, in_p = has_render(name), has_physical(name)
    if not (in_r or in_p):
        raise KeyError("no material %r in either library (try materialindex.find_materials(...))" % name)
    info = {"name": name, "in_render": in_r, "in_physical": in_p}
    if in_r:
        m = _render.material(name)
        info["render"] = {"class": _class_of(name), "base_color": list(m.base_color), "metallic": m.metallic,
                          "roughness": m.roughness, "ior": m.ior, "transmission": getattr(m, "transmission", 0.0),
                          "emissive": list(m.emissive)}
    if in_p:
        info["physical"] = dict(_physical[name])
        # attach the units for exactly the fields present, so the block is self-describing for a scientist
        from holographic_materialdata import UNITS
        info["physical_units"] = {k: UNITS[k][0] for k in info["physical"] if k in UNITS}
    return info


# -- discovery ----------------------------------------------------------------------------------------------
def find_materials(query, k=10):
    """Search across BOTH libraries: score each known material by query-word overlap against its name, render class,
    and phase (a readable word-overlap match, not semantic). Returns [{name, class, phase, in_render, in_physical}]."""
    words = [w for w in str(query).lower().replace(",", " ").split() if w]
    if not words:
        return []
    all_names = set(_render.names()) | set(_physical)
    scored = []
    for name in all_names:
        cls = _class_of(name) if has_render(name) else None
        phase = _physical[name].get("phase") if has_physical(name) else None
        hay = " ".join([name, cls or "", phase or ""]).lower().replace("_", " ")
        score = sum(1 for w in words if w in hay)
        if score:
            scored.append((score, name, cls, phase))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [{"name": n, "class": c, "phase": p, "in_render": has_render(n), "in_physical": has_physical(n)}
            for _, n, c, p in scored[:k]]


def all_materials():
    """The unified roster: every material name with which library/libraries it lives in."""
    names = sorted(set(_render.names()) | set(_physical))
    return [{"name": n, "in_render": has_render(n), "in_physical": has_physical(n)} for n in names]


def summary():
    """A quick 'what do we have': render-preset and physical counts, the overlap, and the class/category breakdown."""
    both = set(_render.names()) & set(_physical)
    return {"render_presets": len(_render.names()), "render_classes": len(_render.classes()),
            "physical_materials": len(_physical), "in_both": len(both), "classes": _render.classes(),
            "physical_categories": physical_categories()}


def _selftest():
    # both libraries are reachable and counted
    s = summary()
    assert s["render_presets"] >= 100 and s["physical_materials"] >= 30 and s["in_both"] >= 10

    # unified 'tell me about gold' -> appearance AND physics
    gold = material_info("gold")
    assert gold["in_render"] and gold["in_physical"]
    assert gold["render"]["metallic"] == 1.0 and gold["render"]["class"] == "metal"
    assert gold["physical"]["density"] == 19300                 # kg/m^3, from the physical library

    # a science-only material (no render preset) still resolves its physics
    merc = material_info("mercury")
    assert merc["in_physical"] and not merc["in_render"]
    assert merc["physical"]["phase"] == "liquid"

    # a render-only preset (no physical def) still resolves its appearance
    assert has_render("chrome") and not has_physical("chrome")
    assert material_info("chrome")["render"]["metallic"] == 1.0

    # the render material is the real PBRMaterial (feeds the render pipeline)
    from holographic_materialio import PBRMaterial
    assert isinstance(render_material("copper"), PBRMaterial)

    # physical properties for a scientist (refractive index of water)
    assert abs(physical_properties("water")["refractive"] - 1.333) < 1e-6

    # discovery across both libraries
    assert any(r["name"] == "water" for r in find_materials("water"))       # name match
    liquids = find_materials("liquid", k=40)                                 # phase/class match (many liquids)
    assert any(r["name"] == "water" for r in liquids) and len(liquids) > 3
    gems = find_materials("gem crystal")
    assert any(r["name"] == "diamond" for r in gems)

    # the unified roster covers the union of both libraries
    roster = all_materials()
    names = {r["name"] for r in roster}
    assert "gold" in names and "mercury" in names and "chrome" in names

    print("OK: holographic_materialindex self-test passed (bridges %d render presets + %d physical materials, %d in "
          "both; material_info gives appearance AND physics; find/list/summary discovery over both libraries)"
          % (s["render_presets"], s["physical_materials"], s["in_both"]))


if __name__ == "__main__":
    _selftest()

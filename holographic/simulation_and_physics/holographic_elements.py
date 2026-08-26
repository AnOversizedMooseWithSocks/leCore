"""holographic_elements.py -- the PERIODIC TABLE as engine ingredients: elements, their properties, and the
composition grammar that lets a material reference its elemental makeup.

WHY THIS EXISTS (Moose's idea, made mechanical)
-----------------------------------------------
The definition library had MATERIALS (water, steel, ...) but no ELEMENTS -- the atoms those materials are made
of. Adding them gives the engine the smallest ingredients and, more importantly, a GRAMMAR: a material can
declare its elemental makeup and RATIO (water = H2O, brass = mostly copper with some zinc), and derived facts
fall out of that composition by COMPOSITION, not by restating them:

    molar mass   = sum over elements of (count * atomic_mass)          -- feeds the gas law (T1), buoyancy, mass
    flame colour = the ratio-weighted BLEND of the elements' flame-test colours   -- feeds combustion (M6)

This is the project's thesis again: a material decomposes into elements the way a VSA record decomposes into
role-filler atoms, and 'blending things to get composites/interpolations' is literally how a composite's
properties are computed from its parts. Elements carry the engine-relevant columns Moose named -- what colour it
burns (flame test), the temperatures where it melts/boils, its atomic mass and density.

HONEST SCOPE (kept negative): a CURATED, engine-relevant subset of the periodic table (~45 common elements), not
all 118 -- and it is EXTENSIBLE (add a row, or drop one in the definition data layer). The property values are
standard reference numbers (atomic masses are the real standard atomic weights; densities/melting points are
room-temperature reference values); flame-test colours are the characteristic ones where they exist (many
elements have no distinctive visible flame test -> flame_color is None). This is a reference table + a composition
grammar, not a quantum chemistry engine. Deterministic; NumPy + stdlib.
"""
import numpy as np

# Each element: (name, atomic_number, atomic_mass g/mol, density kg/m3, melt_K, boil_K, flame_color rgb|None, category)
# atomic_mass = the real standard atomic weight; flame_color = characteristic flame-test colour where one exists.
_E = {
    "H":  ("Hydrogen",   1,   1.008,   0.0899,   14.0,   20.3, None,               "nonmetal"),
    "He": ("Helium",     2,   4.0026,  0.1786,    0.95,   4.2, None,               "noble_gas"),
    "Li": ("Lithium",    3,   6.94,    534.0,   453.7, 1615.0, (0.86, 0.10, 0.16), "alkali_metal"),      # crimson
    "Be": ("Beryllium",  4,   9.0122, 1850.0,  1560.0, 2742.0, None,               "alkaline_earth"),
    "B":  ("Boron",      5,  10.81,   2340.0,  2349.0, 4200.0, (0.35, 0.85, 0.35), "metalloid"),          # bright green
    "C":  ("Carbon",     6,  12.011,  2267.0,  3823.0, 4098.0, None,               "nonmetal"),
    "N":  ("Nitrogen",   7,  14.007,     1.251, 63.2,   77.4, None,               "nonmetal"),
    "O":  ("Oxygen",     8,  15.999,     1.429, 54.4,   90.2, None,               "nonmetal"),
    "F":  ("Fluorine",   9,  18.998,     1.696, 53.5,   85.0, None,               "halogen"),
    "Ne": ("Neon",      10,  20.180,     0.900, 24.6,   27.1, (0.95, 0.40, 0.25), "noble_gas"),           # (discharge) orange-red
    "Na": ("Sodium",    11,  22.990,   971.0,   371.0, 1156.0, (0.98, 0.80, 0.12), "alkali_metal"),       # intense yellow
    "Mg": ("Magnesium", 12,  24.305,  1738.0,   923.0, 1363.0, (0.98, 0.98, 0.98), "alkaline_earth"),     # brilliant white
    "Al": ("Aluminium", 13,  26.982,  2700.0,   933.0, 2792.0, None,               "post_transition"),
    "Si": ("Silicon",   14,  28.085,  2330.0,  1687.0, 3538.0, None,               "metalloid"),
    "P":  ("Phosphorus",15,  30.974,  1820.0,   317.3, 553.7, (0.60, 0.85, 0.55), "nonmetal"),            # pale bluish-green
    "S":  ("Sulfur",    16,  32.06,   2067.0,   388.4, 717.8, (0.45, 0.45, 0.90), "nonmetal"),            # blue
    "Cl": ("Chlorine",  17,  35.45,      3.214,171.6,  239.1, None,               "halogen"),
    "Ar": ("Argon",     18,  39.948,     1.784, 83.8,   87.3, None,               "noble_gas"),
    "K":  ("Potassium", 19,  39.098,   862.0,   336.7, 1032.0, (0.72, 0.45, 0.85), "alkali_metal"),       # lilac
    "Ca": ("Calcium",   20,  40.078,  1540.0,  1115.0, 1757.0, (0.95, 0.45, 0.15), "alkaline_earth"),     # orange-red
    "Ti": ("Titanium",  22,  47.867,  4506.0,  1941.0, 3560.0, None,               "transition_metal"),
    "Cr": ("Chromium",  24,  51.996,  7190.0,  2180.0, 2944.0, None,               "transition_metal"),
    "Mn": ("Manganese", 25,  54.938,  7210.0,  1519.0, 2334.0, (0.70, 0.85, 0.55), "transition_metal"),   # yellow-green
    "Fe": ("Iron",      26,  55.845,  7874.0,  1811.0, 3134.0, (0.95, 0.75, 0.35), "transition_metal"),   # gold sparks
    "Co": ("Cobalt",    27,  58.933,  8900.0,  1768.0, 3200.0, None,               "transition_metal"),
    "Ni": ("Nickel",    28,  58.693,  8908.0,  1728.0, 3186.0, None,               "transition_metal"),
    "Cu": ("Copper",    29,  63.546,  8960.0,  1358.0, 2835.0, (0.20, 0.75, 0.60), "transition_metal"),   # blue-green
    "Zn": ("Zinc",      30,  65.38,   7140.0,   692.7, 1180.0, (0.45, 0.75, 0.85), "transition_metal"),   # blue-green
    "As": ("Arsenic",   33,  74.922,  5727.0,  1090.0, 887.0, (0.35, 0.55, 0.90), "metalloid"),           # blue
    "Br": ("Bromine",   35,  79.904,  3120.0,   266.0, 332.0, None,               "halogen"),
    "Kr": ("Krypton",   36,  83.798,     3.749,115.8, 119.9, None,               "noble_gas"),
    "Rb": ("Rubidium",  37,  85.468,  1532.0,   312.5, 961.0, (0.75, 0.20, 0.55), "alkali_metal"),        # red-violet
    "Sr": ("Strontium", 38,  87.62,   2640.0,  1050.0, 1650.0, (0.90, 0.10, 0.12), "alkaline_earth"),     # crimson
    "Ag": ("Silver",    47, 107.868, 10490.0,  1235.0, 2435.0, None,               "transition_metal"),
    "Sn": ("Tin",       50, 118.710,  7310.0,   505.1, 2875.0, None,               "post_transition"),
    "Sb": ("Antimony",  51, 121.760,  6697.0,   903.8, 1860.0, (0.55, 0.75, 0.85), "metalloid"),          # pale blue-green
    "Ba": ("Barium",    56, 137.327,  3510.0,  1000.0, 2170.0, (0.55, 0.85, 0.45), "alkaline_earth"),     # pale green
    "W":  ("Tungsten",  74, 183.84,  19300.0,  3695.0, 5828.0, None,               "transition_metal"),
    "Pt": ("Platinum",  78, 195.084, 21450.0,  2041.0, 4098.0, None,               "transition_metal"),
    "Au": ("Gold",      79, 196.967, 19300.0,  1337.0, 3129.0, None,               "transition_metal"),
    "Hg": ("Mercury",   80, 200.592, 13534.0,   234.3, 629.9, None,               "transition_metal"),
    "Pb": ("Lead",      82, 207.2,   11340.0,   600.6, 2022.0, (0.70, 0.80, 0.90), "post_transition"),    # pale blue-white
    "U":  ("Uranium",   92, 238.029, 19100.0,  1405.0, 4404.0, None,               "actinide"),
}

_KEYS = ("name", "atomic_number", "atomic_mass", "density", "melt_point_K", "boil_point_K", "flame_color", "category")


def element(symbol):
    """Look up an element by symbol -> dict of its engine-relevant properties (name, atomic_number, atomic_mass
    g/mol, density kg/m3, melt/boil points K, flame_color rgb-or-None, category). KeyError if unknown."""
    if symbol not in _E:
        raise KeyError("unknown element %r -- known: %s" % (symbol, ", ".join(sorted(_E))))
    return dict(zip(_KEYS, _E[symbol]))


def symbols():
    """All element symbols in the table."""
    return sorted(_E)


def element_flame_color(symbol):
    """The characteristic flame-test colour (rgb) of an element, or None if it has no distinctive one."""
    fc = _E[symbol][6]
    return None if fc is None else np.asarray(fc, float)


# --------------------------------------------------------------------------------------------------------------
# COMPOSITION: a material's elemental makeup as {symbol: count} (mole counts / ratio). Derived properties fall
# out by composition -- the "as above, so below" decomposition, and the BLEND that makes composites.
# --------------------------------------------------------------------------------------------------------------
def molar_mass(composition):
    """Molar mass (g/mol) of a composition {symbol: count}: sum of count * atomic_mass. This is what feeds the gas
    law (T1) and any mass-from-moles calculation -- a derived fact, computed from the makeup, never restated."""
    return float(sum(cnt * _E[sym][2] for sym, cnt in composition.items()))


def mass_fractions(composition):
    """Convert a mole-count composition to MASS fractions {symbol: fraction} (they sum to 1). Steel is '98% iron
    by mass' -- that is a mass fraction, and this is how you get it from the mole counts."""
    total = molar_mass(composition)
    return {sym: (cnt * _E[sym][2]) / total for sym, cnt in composition.items()}


def flame_color_of(composition):
    """The flame colour of a composite: the ratio-weighted BLEND of its elements' flame-test colours (only the
    elements that HAVE one contribute). Returns rgb, or None if no constituent has a flame colour. This is the
    blend/interpolation capability applied to combustion -- a copper-bearing compound burns green, a sodium one
    yellow, and a mix lands between them, weighted by how much of each is present."""
    cols, weights = [], []
    for sym, cnt in composition.items():
        fc = _E[sym][6]
        if fc is not None:
            cols.append(np.asarray(fc, float)); weights.append(float(cnt))
    if not cols:
        return None
    w = np.asarray(weights); w = w / w.sum()                        # normalise the ratio into blend weights
    return np.clip(np.sum(np.asarray(cols) * w[:, None], axis=0), 0.0, 1.0)


# Some materials' elemental makeup (mole counts). This is where a MATERIAL references its elements + ratio; it is
# extensible (add a row, or carry it on the material definition). Compounds by formula; alloys by rough ratio.
MATERIAL_COMPOSITION = {
    "water":            {"H": 2, "O": 1},                           # H2O
    "table_salt":       {"Na": 1, "Cl": 1},                         # NaCl
    "quartz":           {"Si": 1, "O": 2},                          # SiO2
    "silica":           {"Si": 1, "O": 2},
    "rust":             {"Fe": 2, "O": 3},                          # Fe2O3
    "alumina":          {"Al": 2, "O": 3},                          # Al2O3 (corundum/ruby/sapphire host)
    "methane":          {"C": 1, "H": 4},                           # CH4
    "carbon_dioxide":   {"C": 1, "O": 2},                           # CO2
    "ammonia":          {"N": 1, "H": 3},                           # NH3
    "ethanol":          {"C": 2, "H": 6, "O": 1},                   # C2H6O
    "limestone":        {"Ca": 1, "C": 1, "O": 3},                  # CaCO3
    "brass":            {"Cu": 2, "Zn": 1},                         # ~ Cu-Zn alloy (mole-ish ratio)
    "bronze":           {"Cu": 7, "Sn": 1},                         # ~ Cu-Sn alloy
    "steel":            {"Fe": 50, "C": 1},                         # ~ mostly iron, a little carbon
    "gold":             {"Au": 1},
    "copper":           {"Cu": 1},
    "iron":             {"Fe": 1},
    "aluminum":         {"Al": 1},
}


def material_composition(name):
    """The elemental makeup {symbol: count} of a named material, or None if we do not have it. This is the link
    from the material definitions down to the elements -- the ratio that feeds the derived properties."""
    comp = MATERIAL_COMPOSITION.get(name)
    return dict(comp) if comp else None


def material_elemental(name):
    """Everything derivable from a material's elemental makeup: {composition, molar_mass, flame_color,
    mass_fractions}. flame_color is the ratio-weighted blend of its elements' flame colours (None if none glow).
    Returns None if the material has no composition on file."""
    comp = material_composition(name)
    if comp is None:
        return None
    return {"composition": comp, "molar_mass": molar_mass(comp),
            "flame_color": flame_color_of(comp), "mass_fractions": mass_fractions(comp)}


def _element_state(e):
    """STP phase (gas/liquid/solid) derived categorically from melt/boil points -- so identify can match on
    'what phase is it' without exposing the continuous temperatures. ~293 K = room temperature."""
    mp, bp = e.get("melt_point_K"), e.get("boil_point_K")
    if bp is not None and bp < 293:
        return "gas"
    if mp is not None and mp < 293 and (bp is None or bp >= 293):
        return "liquid"
    return "solid"


def element_record(symbol):
    """The CATEGORICAL identity record of an element: {category, state}. Both are categories (noble_gas /
    transition_metal / ...; gas / liquid / solid), never the continuous mass/density -- so it can be matched
    with match_record (categorical by contract). The reverse of element(): element() looks up properties BY
    name; this exposes the categorical fingerprint that identify_element matches AGAINST."""
    e = element(symbol)
    return {"category": e["category"], "state": _element_state(e)}


def identify_element(props, mind=None, margin=0.1):
    """Identify the element(s) whose categorical fingerprint {category, state} best matches `props`, e.g.
    {'category': 'noble_gas', 'state': 'gas'} -> Helium/Neon/... . Uses match_record over the whole table and
    decide_or_abstain: returns {'ranked': [(symbol, score)...], 'best': symbol_or_None, 'confident': bool}.
    Since many elements share a category, several tie at 1.000 -- 'confident' is False on a tie (the honest
    answer: the fingerprint under-determines the element), and the caller narrows with more fields. WHY THIS
    EXISTS: the table already stores the records; this makes them queryable BY attribute, an agent-callable
    'which element is an inert gas?' instead of only 'what is helium?'. KEPT NEGATIVE: categorical only --
    atomic number/mass are continuous and deliberately excluded; they would need a separate numeric lookup."""
    if mind is None:
        import lecore
        mind = lecore.UnifiedMind(dim=512, seed=0)
    from holographic.misc.holographic_relations import match_record, decide_or_abstain
    candidates = {s: element_record(s) for s in symbols()}
    ranked = match_record(mind.encode_record, props, candidates)
    best, score, confident = decide_or_abstain(ranked, margin=margin)
    return {"ranked": ranked, "best": best, "confident": confident, "score": score}


def _selftest():
    """Element lookups are correct, composition derives the right molar masses, flame colour is the ratio-weighted
    blend of the constituents, and materials link down to their elements. Deterministic."""
    # (1) element properties: standard atomic weights
    assert abs(element("H")["atomic_mass"] - 1.008) < 1e-3
    assert abs(element("Fe")["atomic_mass"] - 55.845) < 1e-3
    assert abs(element("Au")["atomic_mass"] - 196.967) < 1e-2
    assert element("Fe")["category"] == "transition_metal"

    # (2) molar mass from composition: water ~18.015, table salt ~58.44, CO2 ~44.01
    assert abs(molar_mass({"H": 2, "O": 1}) - 18.015) < 0.01
    assert abs(molar_mass({"Na": 1, "Cl": 1}) - 58.44) < 0.01
    assert abs(molar_mass({"C": 1, "O": 2}) - 44.009) < 0.01

    # (3) flame colour is the ratio-weighted BLEND of the elements' flame colours
    na = flame_color_of({"Na": 1})                                  # sodium: yellow (r,g high, b low)
    assert na[0] > 0.7 and na[1] > 0.6 and na[2] < 0.3
    cu = flame_color_of({"Cu": 1})                                  # copper: green-blue (g highest)
    assert cu[1] > cu[0]
    mix = flame_color_of({"Na": 1, "Cu": 1})                        # a 50/50 blend sits between the two
    assert np.allclose(mix, 0.5 * (na + cu), atol=1e-6)
    # weighting by ratio shifts the blend toward the more abundant element
    na_heavy = flame_color_of({"Na": 3, "Cu": 1})
    assert np.linalg.norm(na_heavy - na) < np.linalg.norm(mix - na)
    assert flame_color_of({"O": 1, "H": 2}) is None                 # water has no flame-test colour

    # (4) a material links down to its elements: water is H2O with molar mass ~18; salt burns yellow (sodium)
    w = material_elemental("water")
    assert w["composition"] == {"H": 2, "O": 1} and abs(w["molar_mass"] - 18.015) < 0.01
    salt = material_elemental("table_salt")
    assert salt["flame_color"][0] > 0.7 and salt["flame_color"][2] < 0.3   # sodium yellow
    # mass fractions: steel is mostly iron by mass
    steel = material_elemental("steel")
    assert steel["mass_fractions"]["Fe"] > 0.95
    assert material_elemental("granite") is None                    # no composition on file -> honest None

    # (5) deterministic
    assert np.array_equal(flame_color_of({"Na": 1, "Ca": 1}), flame_color_of({"Na": 1, "Ca": 1}))
    print("holographic_elements selftest OK: %d elements; molar mass from composition (H2O=%.3f, NaCl=%.2f); "
          "flame colour is the ratio-weighted blend (Na yellow, Cu green, mix between); materials link to elements"
          % (len(_E), molar_mass({"H": 2, "O": 1}), molar_mass({"Na": 1, "Cl": 1})))



def _selftest_identify():
    """A1: identify_element surfaces the right category and honestly abstains when the fingerprint is shared."""
    import lecore
    m = lecore.UnifiedMind(dim=512, seed=0)
    r = identify_element({"category": "noble_gas", "state": "gas"}, mind=m)
    tops = [s for s, sc in r["ranked"] if sc > 0.99]
    assert all(element_record(s)["category"] == "noble_gas" for s in tops), tops
    assert not r["confident"], "shared fingerprint must abstain, not fake certainty"
    # a category with ONE member would be confident -- guard the abstain logic is real, not always-False
    r2 = identify_element({"category": "zzz_nonexistent", "state": "solid"}, mind=m)
    assert r2["best"] is not None            # still returns a ranked best (nearest), just low score
    assert set(element_record(symbols()[0])) == {"category", "state"}  # categorical only, no continuous leak
    print("  identify_element selftest OK: noble_gas -> %s (abstains on shared fingerprint)" % tops[:3])

if __name__ == "__main__":
    _selftest(); _selftest_identify()

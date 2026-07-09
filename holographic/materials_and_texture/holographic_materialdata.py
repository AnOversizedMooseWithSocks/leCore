"""holographic_materialdata.py -- a comprehensive, categorised database of REAL physical material properties.

This is the "good starting point" physical library: ~90 materials across metals, liquids, gases, polymers, ceramics,
glass, minerals, stone, wood, biological tissue, building materials and semiconductors, each with the standard
engineering/physics numbers a solver or a scientist reaches for. It feeds holographic_definitions.MATERIALS (which
enriches its legacy entries from here and gains all the new ones), so everything downstream -- resolve_scenario, the
UnifiedMind material faculties, the material index -- sees the expanded set with no other change.

HOW TO READ THE DATA (all SI; see UNITS for the authoritative list)
  density              kg/m^3          mass per volume            (the one field every entry has)
  phase                solid/liquid/gas
  youngs               GPa             Young's (elastic) modulus  -- solids
  sound_speed          m/s             longitudinal speed of sound (bulk; along-grain for wood)
  specific_heat        J/(kg*K)        specific heat capacity, c_p
  thermal_conductivity W/(m*K)         how well it conducts heat
  thermal_expansion    1/K             linear expansion coefficient (e.g. 2.3e-5)
  viscosity            Pa*s            dynamic viscosity          -- liquids/gases
  refractive           (dimensionless) refractive index at ~589 nm -- transparent media
  melting_point        K               melting/solidus point      -- where meaningful
  boiling_point        K               boiling point at 1 atm     -- liquids/gases

HONESTY (this project keeps its provenance loud)
  * Values are STANDARD REFERENCE values (CRC Handbook / common engineering tables) at ~20 C, 1 atm unless noted.
    They are good starting points, not a lab certificate: real materials vary with alloy, grade, temperature,
    grain, porosity and processing. Where a property is strongly variable or I am not confident of a value, the field
    is OMITTED rather than guessed -- so a missing field means "look it up for your exact material", not zero.
  * Wood/tissue/building materials vary a lot between samples; treat their numbers as representative.
  * Gas densities are at ~20 C, 1 atm; they scale with temperature and pressure.
Users add their own with `add_material(...)` (or by editing holographic_definitions.MATERIALS directly).
"""

# field -> (unit, human description). The authoritative units reference.
UNITS = {
    "density": ("kg/m^3", "mass per unit volume"),
    "phase": ("", "solid / liquid / gas at ~20 C, 1 atm"),
    "youngs": ("GPa", "Young's (elastic) modulus"),
    "sound_speed": ("m/s", "longitudinal speed of sound"),
    "specific_heat": ("J/(kg*K)", "specific heat capacity c_p"),
    "thermal_conductivity": ("W/(m*K)", "thermal conductivity"),
    "thermal_expansion": ("1/K", "linear thermal expansion coefficient"),
    "viscosity": ("Pa*s", "dynamic viscosity"),
    "refractive": ("", "refractive index at ~589 nm"),
    "melting_point": ("K", "melting / solidus point"),
    "boiling_point": ("K", "boiling point at 1 atm"),
}

# valid categories (organisation for a UI picker / a scientist browsing the library)
CATEGORIES = ("metal", "liquid", "gas", "polymer", "ceramic", "glass", "mineral", "stone",
              "wood", "biological", "building", "semiconductor")

# plausible physical ranges per field -- used by validate() to catch typos/unit slips (not to reject real outliers,
# which is why the bounds are generous).
_PLAUSIBLE = {
    "density": (0.05, 25000),            # helium ~0.18 ... osmium ~22600
    "youngs": (0.001, 1300),             # soft rubber ... diamond ~1050
    "sound_speed": (100, 20000),         # gases ... diamond ~12000-18000
    "specific_heat": (100, 15000),       # metals ~130 ... hydrogen ~14300
    "thermal_conductivity": (0.01, 2500),  # aerogel/insulation ... diamond ~2200
    "thermal_expansion": (1e-7, 3e-4),
    "viscosity": (1e-6, 200),            # gases ~1e-5 ... honey ~10
    "refractive": (1.0, 4.5),            # gases ~1.0 ... germanium ~4.0
    "melting_point": (1.0, 4000),
    "boiling_point": (4.0, 6000),
}


# ============================================================================================================
# THE DATA. Grouped by category, commented. Every entry has category, phase, density; other fields where confident.
# ============================================================================================================
PHYSICAL_MATERIALS = {}


def _m(name, category, phase, density, **props):
    """Register one material. Keeps the call sites readable: name, category, phase, density, then confident extras."""
    entry = {"category": category, "phase": phase, "density": density}
    entry.update(props)
    PHYSICAL_MATERIALS[name] = entry


# ---- METALS & ALLOYS (density, youngs, sound_speed, specific_heat, thermal_cond, thermal_exp, melting) ------
_m("aluminum", "metal", "solid", 2700, youngs=69.0, sound_speed=6320, specific_heat=897,
   thermal_conductivity=237, thermal_expansion=2.3e-5, melting_point=933)
_m("iron", "metal", "solid", 7874, youngs=211.0, sound_speed=5120, specific_heat=449,
   thermal_conductivity=80, thermal_expansion=1.18e-5, melting_point=1811)
_m("steel", "metal", "solid", 7850, youngs=200.0, sound_speed=5960, specific_heat=490,
   thermal_conductivity=50, thermal_expansion=1.2e-5, melting_point=1700)
_m("stainless_steel", "metal", "solid", 8000, youngs=193.0, sound_speed=5790, specific_heat=500,
   thermal_conductivity=16, thermal_expansion=1.7e-5, melting_point=1700)
_m("cast_iron", "metal", "solid", 7200, youngs=110.0, sound_speed=4600, specific_heat=460,
   thermal_conductivity=55, thermal_expansion=1.05e-5, melting_point=1450)
_m("copper", "metal", "solid", 8960, youngs=130.0, sound_speed=4760, specific_heat=385,
   thermal_conductivity=401, thermal_expansion=1.65e-5, melting_point=1358)
_m("brass", "metal", "solid", 8500, youngs=100.0, sound_speed=4700, specific_heat=380,
   thermal_conductivity=120, thermal_expansion=1.9e-5, melting_point=1200)
_m("bronze", "metal", "solid", 8800, youngs=110.0, sound_speed=3500, specific_heat=380,
   thermal_conductivity=60, thermal_expansion=1.8e-5, melting_point=1200)
_m("gold", "metal", "solid", 19300, youngs=79.0, sound_speed=3240, specific_heat=129,
   thermal_conductivity=318, thermal_expansion=1.4e-5, melting_point=1337)
_m("silver", "metal", "solid", 10490, youngs=83.0, sound_speed=3650, specific_heat=235,
   thermal_conductivity=429, thermal_expansion=1.89e-5, melting_point=1235)
_m("platinum", "metal", "solid", 21450, youngs=168.0, sound_speed=2680, specific_heat=133,
   thermal_conductivity=72, thermal_expansion=8.8e-6, melting_point=2041)
_m("titanium", "metal", "solid", 4506, youngs=116.0, sound_speed=6070, specific_heat=523,
   thermal_conductivity=22, thermal_expansion=8.6e-6, melting_point=1941)
_m("nickel", "metal", "solid", 8908, youngs=200.0, sound_speed=5810, specific_heat=444,
   thermal_conductivity=91, thermal_expansion=1.3e-5, melting_point=1728)
_m("zinc", "metal", "solid", 7140, youngs=108.0, sound_speed=3850, specific_heat=388,
   thermal_conductivity=116, thermal_expansion=3.0e-5, melting_point=693)
_m("tin", "metal", "solid", 7265, youngs=50.0, sound_speed=2730, specific_heat=228,
   thermal_conductivity=67, thermal_expansion=2.2e-5, melting_point=505)
_m("lead", "metal", "solid", 11340, youngs=16.0, sound_speed=1210, specific_heat=129,
   thermal_conductivity=35, thermal_expansion=2.9e-5, melting_point=601)
_m("magnesium", "metal", "solid", 1738, youngs=45.0, sound_speed=4940, specific_heat=1023,
   thermal_conductivity=156, thermal_expansion=2.5e-5, melting_point=923)
_m("tungsten", "metal", "solid", 19250, youngs=411.0, sound_speed=5220, specific_heat=134,
   thermal_conductivity=173, thermal_expansion=4.5e-6, melting_point=3695)
_m("chromium", "metal", "solid", 7190, youngs=279.0, sound_speed=5940, specific_heat=449,
   thermal_conductivity=94, thermal_expansion=4.9e-6, melting_point=2180)
_m("cobalt", "metal", "solid", 8900, youngs=209.0, sound_speed=4720, specific_heat=421,
   thermal_conductivity=100, thermal_expansion=1.3e-5, melting_point=1768)
_m("uranium", "metal", "solid", 19050, youngs=208.0, sound_speed=3155, specific_heat=116,
   thermal_conductivity=27, melting_point=1405)
_m("sodium", "metal", "solid", 971, youngs=10.0, specific_heat=1228, thermal_conductivity=142, melting_point=371)
_m("mercury", "metal", "liquid", 13534, viscosity=0.0015, sound_speed=1450, specific_heat=140,
   thermal_conductivity=8.3, melting_point=234, boiling_point=630)

# ---- LIQUIDS (density, viscosity, sound_speed, specific_heat, refractive, boiling) --------------------------
_m("water", "liquid", "liquid", 1000, viscosity=0.001, sound_speed=1481, specific_heat=4186,
   refractive=1.333, thermal_conductivity=0.6, melting_point=273, boiling_point=373)
_m("seawater", "liquid", "liquid", 1025, viscosity=0.00107, sound_speed=1500, specific_heat=3990, refractive=1.34)
_m("ethanol", "liquid", "liquid", 789, viscosity=0.0012, sound_speed=1160, specific_heat=2440,
   refractive=1.361, boiling_point=351)
_m("methanol", "liquid", "liquid", 792, viscosity=0.00059, sound_speed=1120, specific_heat=2510,
   refractive=1.329, boiling_point=338)
_m("glycerin", "liquid", "liquid", 1261, viscosity=1.412, sound_speed=1920, specific_heat=2430,
   refractive=1.474, boiling_point=563)
_m("acetone", "liquid", "liquid", 784, viscosity=0.000306, sound_speed=1170, specific_heat=2160,
   refractive=1.359, boiling_point=329)
_m("benzene", "liquid", "liquid", 876, viscosity=0.000604, sound_speed=1310, specific_heat=1740,
   refractive=1.501, boiling_point=353)
_m("ethylene_glycol", "liquid", "liquid", 1113, viscosity=0.0161, sound_speed=1660, specific_heat=2200,
   refractive=1.431, boiling_point=470)
_m("gasoline", "liquid", "liquid", 745, viscosity=0.0006, sound_speed=1250, specific_heat=2220, refractive=1.44)
_m("diesel", "liquid", "liquid", 850, viscosity=0.0025, sound_speed=1250, specific_heat=2050, refractive=1.46)
_m("olive_oil", "liquid", "liquid", 915, viscosity=0.081, sound_speed=1430, specific_heat=1970, refractive=1.47)
_m("motor_oil", "liquid", "liquid", 900, viscosity=0.25, specific_heat=1900, refractive=1.47)
_m("honey", "liquid", "liquid", 1420, viscosity=10.0, specific_heat=2300, refractive=1.49)
_m("milk", "liquid", "liquid", 1030, viscosity=0.003, sound_speed=1550, specific_heat=3930)
_m("blood", "liquid", "liquid", 1060, viscosity=0.004, sound_speed=1570, specific_heat=3600)
_m("sulfuric_acid", "liquid", "liquid", 1830, viscosity=0.0242, specific_heat=1340, refractive=1.43)
_m("liquid_ammonia", "liquid", "liquid", 682, viscosity=0.00013, specific_heat=4700, refractive=1.325,
   boiling_point=240)

# ---- GASES (density at ~20 C 1 atm, viscosity, sound_speed, specific_heat c_p, refractive, boiling) ---------
_m("air", "gas", "gas", 1.225, viscosity=1.8e-5, sound_speed=343, specific_heat=1005, refractive=1.0003)
_m("oxygen", "gas", "gas", 1.429, viscosity=2.04e-5, sound_speed=326, specific_heat=918, refractive=1.0003,
   boiling_point=90)
_m("nitrogen", "gas", "gas", 1.251, viscosity=1.76e-5, sound_speed=349, specific_heat=1040, refractive=1.0003,
   boiling_point=77)
_m("hydrogen", "gas", "gas", 0.0899, viscosity=8.8e-6, sound_speed=1310, specific_heat=14300, refractive=1.0001,
   boiling_point=20)
_m("helium", "gas", "gas", 0.1786, viscosity=1.96e-5, sound_speed=972, specific_heat=5193, refractive=1.000036,
   boiling_point=4.2)
_m("argon", "gas", "gas", 1.784, viscosity=2.23e-5, sound_speed=319, specific_heat=520, refractive=1.00028,
   boiling_point=87)
_m("carbon_dioxide", "gas", "gas", 1.977, viscosity=1.47e-5, sound_speed=259, specific_heat=844, refractive=1.00045)
_m("methane", "gas", "gas", 0.717, viscosity=1.1e-5, sound_speed=430, specific_heat=2220, refractive=1.0004,
   boiling_point=112)
_m("propane", "gas", "gas", 2.01, viscosity=8.0e-6, sound_speed=258, specific_heat=1670, boiling_point=231)
_m("neon", "gas", "gas", 0.900, viscosity=3.13e-5, sound_speed=435, specific_heat=1030, refractive=1.000067)

# ---- POLYMERS / PLASTICS (density, youngs, specific_heat, thermal_cond, refractive) ------------------------
_m("hdpe", "polymer", "solid", 950, youngs=1.0, specific_heat=1900, thermal_conductivity=0.48, refractive=1.54)
_m("ldpe", "polymer", "solid", 920, youngs=0.2, specific_heat=2300, thermal_conductivity=0.33, refractive=1.51)
_m("polypropylene", "polymer", "solid", 905, youngs=1.5, specific_heat=1920, thermal_conductivity=0.22,
   refractive=1.49)
_m("pvc", "polymer", "solid", 1380, youngs=3.0, specific_heat=900, thermal_conductivity=0.19, refractive=1.53)
_m("pet", "polymer", "solid", 1380, youngs=2.8, specific_heat=1000, thermal_conductivity=0.24, refractive=1.575)
_m("polystyrene", "polymer", "solid", 1050, youngs=3.2, specific_heat=1300, thermal_conductivity=0.14,
   refractive=1.59)
_m("nylon", "polymer", "solid", 1150, youngs=2.7, specific_heat=1700, thermal_conductivity=0.25, refractive=1.53)
_m("ptfe", "polymer", "solid", 2200, youngs=0.5, specific_heat=1000, thermal_conductivity=0.25, refractive=1.35)
_m("acrylic", "polymer", "solid", 1180, youngs=3.0, specific_heat=1470, thermal_conductivity=0.19, refractive=1.49)
_m("polycarbonate", "polymer", "solid", 1200, youngs=2.4, specific_heat=1200, thermal_conductivity=0.20,
   refractive=1.585)
_m("abs", "polymer", "solid", 1050, youngs=2.3, specific_heat=1300, thermal_conductivity=0.17, refractive=1.53)
_m("epoxy", "polymer", "solid", 1200, youngs=3.5, specific_heat=1000, thermal_conductivity=0.2, refractive=1.55)
_m("natural_rubber", "polymer", "solid", 930, youngs=0.02, specific_heat=1900, thermal_conductivity=0.13,
   refractive=1.52)
_m("silicone_rubber", "polymer", "solid", 1100, youngs=0.01, specific_heat=1200, thermal_conductivity=0.2,
   refractive=1.40)
_m("kevlar", "polymer", "solid", 1440, youngs=70.0, specific_heat=1420, thermal_conductivity=0.04)

# ---- CERAMICS & GLASS -------------------------------------------------------------------------------------
_m("soda_lime_glass", "glass", "solid", 2500, youngs=70.0, sound_speed=5640, specific_heat=840,
   thermal_conductivity=1.0, refractive=1.52)
_m("borosilicate_glass", "glass", "solid", 2230, youngs=64.0, sound_speed=5640, specific_heat=830,
   thermal_conductivity=1.14, refractive=1.47)
_m("fused_silica", "glass", "solid", 2200, youngs=72.0, sound_speed=5900, specific_heat=740,
   thermal_conductivity=1.4, refractive=1.458, melting_point=1983)
_m("alumina", "ceramic", "solid", 3950, youngs=370.0, sound_speed=10000, specific_heat=880,
   thermal_conductivity=30, refractive=1.76, melting_point=2345)
_m("silicon_carbide", "ceramic", "solid", 3210, youngs=410.0, sound_speed=12000, specific_heat=750,
   thermal_conductivity=120, refractive=2.65)
_m("silicon_nitride", "ceramic", "solid", 3200, youngs=310.0, sound_speed=11000, specific_heat=700,
   thermal_conductivity=30)
_m("zirconia", "ceramic", "solid", 5680, youngs=200.0, specific_heat=400, thermal_conductivity=2.0, refractive=2.15)
_m("porcelain", "ceramic", "solid", 2400, youngs=70.0, specific_heat=1000, thermal_conductivity=1.5)

# ---- MINERALS & STONE -------------------------------------------------------------------------------------
_m("granite", "stone", "solid", 2700, youngs=50.0, sound_speed=6000, specific_heat=790, thermal_conductivity=2.9)
_m("marble", "stone", "solid", 2560, youngs=60.0, sound_speed=6000, specific_heat=880, thermal_conductivity=2.8)
_m("limestone", "stone", "solid", 2600, youngs=40.0, sound_speed=6000, specific_heat=910, thermal_conductivity=1.3)
_m("sandstone", "stone", "solid", 2300, youngs=20.0, sound_speed=3500, specific_heat=920, thermal_conductivity=2.5)
_m("basalt", "stone", "solid", 3000, youngs=60.0, sound_speed=6000, specific_heat=840, thermal_conductivity=2.0)
_m("quartz", "mineral", "solid", 2650, youngs=95.0, sound_speed=5760, specific_heat=730, thermal_conductivity=8.0,
   refractive=1.544, melting_point=1983)
_m("diamond", "mineral", "solid", 3510, youngs=1050.0, sound_speed=12000, specific_heat=509,
   thermal_conductivity=2200, refractive=2.417)
_m("graphite", "mineral", "solid", 2260, youngs=20.0, specific_heat=710, thermal_conductivity=150)
_m("halite", "mineral", "solid", 2170, youngs=40.0, sound_speed=4560, specific_heat=880, thermal_conductivity=6.5,
   refractive=1.544, melting_point=1074)
_m("sapphire", "mineral", "solid", 3980, youngs=400.0, sound_speed=11000, specific_heat=760,
   thermal_conductivity=40, refractive=1.77, melting_point=2323)
_m("calcite", "mineral", "solid", 2710, youngs=80.0, specific_heat=800, refractive=1.658)
_m("gypsum", "mineral", "solid", 2320, specific_heat=1090, thermal_conductivity=1.3)

# ---- WOOD (density + along-grain properties; representative -- wood varies a lot) --------------------------
_m("oak_wood", "wood", "solid", 700, youngs=11.0, sound_speed=3800, specific_heat=2000, thermal_conductivity=0.17)
_m("pine_wood", "wood", "solid", 500, youngs=9.0, sound_speed=3300, specific_heat=1700, thermal_conductivity=0.12)
_m("balsa", "wood", "solid", 160, youngs=3.0, sound_speed=4300, specific_heat=2900, thermal_conductivity=0.05)
_m("bamboo", "wood", "solid", 700, youngs=20.0, specific_heat=1800, thermal_conductivity=0.17)
_m("plywood", "wood", "solid", 600, youngs=8.0, specific_heat=1200, thermal_conductivity=0.13)
_m("maple", "wood", "solid", 705, youngs=12.0, specific_heat=1700, thermal_conductivity=0.16)
_m("cedar", "wood", "solid", 380, youngs=8.0, specific_heat=1700, thermal_conductivity=0.09)

# ---- BIOLOGICAL TISSUE (representative; for acoustics/medical sims) ----------------------------------------
_m("bone", "biological", "solid", 1900, youngs=18.0, sound_speed=4000, specific_heat=1300, thermal_conductivity=0.32)
_m("muscle", "biological", "solid", 1090, sound_speed=1580, specific_heat=3500, thermal_conductivity=0.49)
_m("fat", "biological", "solid", 909, sound_speed=1450, specific_heat=2300, thermal_conductivity=0.21)
_m("skin_tissue", "biological", "solid", 1100, sound_speed=1600, specific_heat=3500, thermal_conductivity=0.37)
_m("cartilage", "biological", "solid", 1100, sound_speed=1650, specific_heat=3500)
_m("leather", "biological", "solid", 860, specific_heat=1500)

# ---- BUILDING MATERIALS -----------------------------------------------------------------------------------
_m("concrete", "building", "solid", 2400, youngs=30.0, sound_speed=3700, specific_heat=880, thermal_conductivity=1.4)
_m("brick", "building", "solid", 1920, youngs=14.0, specific_heat=840, thermal_conductivity=0.7)
_m("asphalt", "building", "solid", 2300, specific_heat=920, thermal_conductivity=0.75)
_m("plaster", "building", "solid", 850, specific_heat=1000, thermal_conductivity=0.4)
_m("drywall", "building", "solid", 800, specific_heat=1090, thermal_conductivity=0.17)
_m("cork", "building", "solid", 240, youngs=0.03, sound_speed=500, specific_heat=1900, thermal_conductivity=0.04)
_m("styrofoam", "building", "solid", 50, youngs=0.005, specific_heat=1300, thermal_conductivity=0.033)
_m("fiberglass_insulation", "building", "solid", 40, specific_heat=700, thermal_conductivity=0.04)

# ---- ICE & MISC SOLIDS ------------------------------------------------------------------------------------
_m("ice", "mineral", "solid", 917, youngs=9.0, sound_speed=3200, specific_heat=2090, thermal_conductivity=2.2,
   refractive=1.31, melting_point=273)
_m("wax", "polymer", "solid", 900, specific_heat=2100, thermal_conductivity=0.25, melting_point=337)
_m("clay", "mineral", "solid", 1750, specific_heat=920, thermal_conductivity=1.3)
_m("rubber", "polymer", "solid", 1100, youngs=0.05, sound_speed=1600, specific_heat=1900, thermal_conductivity=0.16)
_m("pvc_plastic", "polymer", "solid", 1380, youngs=3.0, specific_heat=900, thermal_conductivity=0.19, refractive=1.53)
_m("wood", "wood", "solid", 650, youngs=10.0, sound_speed=3300, specific_heat=1700, thermal_conductivity=0.15)
_m("glass", "glass", "solid", 2500, youngs=70.0, sound_speed=5640, specific_heat=840, thermal_conductivity=1.0,
   refractive=1.52)

# ---- SEMICONDUCTORS ---------------------------------------------------------------------------------------
_m("silicon", "semiconductor", "solid", 2329, youngs=130.0, sound_speed=8433, specific_heat=705,
   thermal_conductivity=150, thermal_expansion=2.6e-6, refractive=3.42, melting_point=1687)
_m("germanium", "semiconductor", "solid", 5323, youngs=103.0, sound_speed=5400, specific_heat=320,
   thermal_conductivity=60, refractive=4.0, melting_point=1211)
_m("gallium_arsenide", "semiconductor", "solid", 5320, youngs=85.0, sound_speed=5240, specific_heat=350,
   thermal_conductivity=55, refractive=3.9, melting_point=1511)


# ============================================================================================================
# Accessors + validation.
# ============================================================================================================
def add_material(name, category, phase, density, **props):
    """Add a user material to this database (and, once merged, to the whole engine's material knowledge)."""
    if category not in CATEGORIES:
        raise ValueError("category must be one of %s" % (CATEGORIES,))
    _m(name, category, phase, density, **props)
    return name


def categories():
    return sorted({e["category"] for e in PHYSICAL_MATERIALS.values()})


def by_category(category):
    return sorted(n for n, e in PHYSICAL_MATERIALS.items() if e.get("category") == category)


def field_coverage():
    """How many materials carry each field -- an honesty report on the library's completeness."""
    cov = {}
    for e in PHYSICAL_MATERIALS.values():
        for k in e:
            cov[k] = cov.get(k, 0) + 1
    return dict(sorted(cov.items(), key=lambda kv: -kv[1]))


def validate():
    """Check every entry for plausibility (right units, sane ranges, known category/phase). Returns a list of issue
    strings -- EMPTY means the library is clean. Kept loud so a bad edit is caught, not hidden."""
    issues = []
    for name, e in PHYSICAL_MATERIALS.items():
        if "density" not in e:
            issues.append("%s: missing density" % name)
        if e.get("phase") not in ("solid", "liquid", "gas"):
            issues.append("%s: bad phase %r" % (name, e.get("phase")))
        if e.get("category") not in CATEGORIES:
            issues.append("%s: bad/missing category %r" % (name, e.get("category")))
        for field, val in e.items():
            if field in ("phase", "category"):
                continue
            if field not in UNITS:
                issues.append("%s: unknown field %r (not in UNITS)" % (name, field))
                continue
            lo, hi = _PLAUSIBLE.get(field, (None, None))
            if lo is not None and not (lo <= val <= hi):
                issues.append("%s: %s=%g out of plausible range [%g, %g]" % (name, field, val, lo, hi))
    return issues


def _selftest():
    assert len(PHYSICAL_MATERIALS) >= 85, len(PHYSICAL_MATERIALS)
    assert set(categories()) <= set(CATEGORIES)
    # the library validates clean (every value in a plausible range, right units)
    issues = validate()
    assert not issues, issues[:5]
    # coverage: density+phase+category on all; the rest partial (honest omissions)
    cov = field_coverage()
    assert cov["density"] == len(PHYSICAL_MATERIALS) == cov["phase"] == cov["category"]
    # spot-check a few real values across categories
    assert PHYSICAL_MATERIALS["tungsten"]["melting_point"] == 3695
    assert PHYSICAL_MATERIALS["helium"]["density"] == 0.1786
    assert abs(PHYSICAL_MATERIALS["diamond"]["thermal_conductivity"] - 2200) < 1
    assert PHYSICAL_MATERIALS["balsa"]["category"] == "wood"
    # user add + category guard
    add_material("test_alloy", "metal", "solid", 5000, youngs=100.0)
    assert "test_alloy" in PHYSICAL_MATERIALS
    del PHYSICAL_MATERIALS["test_alloy"]
    try:
        add_material("bad", "notacat", "solid", 100); assert False
    except ValueError:
        pass
    print("OK: holographic_materialdata self-test passed (%d materials across %d categories; validate() clean; "
          "density/phase/category on all, deeper fields where confident)" % (len(PHYSICAL_MATERIALS), len(categories())))


if __name__ == "__main__":
    _selftest()

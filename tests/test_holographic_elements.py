"""Elements: standard atomic weights, molar mass from composition, ratio-weighted flame-colour blend, material link."""
import numpy as np
from holographic.simulation_and_physics.holographic_elements import element, molar_mass, mass_fractions, flame_color_of, material_elemental, element_flame_color, symbols


def test_element_properties_are_standard():
    assert abs(element("H")["atomic_mass"] - 1.008) < 1e-3
    assert abs(element("Fe")["atomic_mass"] - 55.845) < 1e-3
    assert abs(element("Au")["atomic_mass"] - 196.967) < 1e-2
    assert element("Cu")["category"] == "transition_metal"
    assert len(symbols()) >= 40


def test_molar_mass_from_composition():
    assert abs(molar_mass({"H": 2, "O": 1}) - 18.015) < 0.01       # water
    assert abs(molar_mass({"Na": 1, "Cl": 1}) - 58.44) < 0.01      # salt
    assert abs(molar_mass({"C": 1, "O": 2}) - 44.009) < 0.01       # CO2


def test_flame_colour_is_ratio_weighted_blend():
    na = flame_color_of({"Na": 1}); cu = flame_color_of({"Cu": 1})
    assert na[0] > 0.7 and na[2] < 0.3                             # sodium yellow
    assert cu[1] > cu[0]                                           # copper green
    mix = flame_color_of({"Na": 1, "Cu": 1})
    assert np.allclose(mix, 0.5 * (na + cu), atol=1e-6)            # 50/50 blend
    na_heavy = flame_color_of({"Na": 3, "Cu": 1})
    assert np.linalg.norm(na_heavy - na) < np.linalg.norm(mix - na)   # ratio shifts the blend
    assert flame_color_of({"H": 2, "O": 1}) is None               # water has no flame colour


def test_material_links_to_elements():
    w = material_elemental("water")
    assert w["composition"] == {"H": 2, "O": 1} and abs(w["molar_mass"] - 18.015) < 0.01
    salt = material_elemental("table_salt")
    assert salt["flame_color"][0] > 0.7 and salt["flame_color"][2] < 0.3   # burns sodium-yellow
    assert material_elemental("steel")["mass_fractions"]["Fe"] > 0.95      # mostly iron by mass
    assert material_elemental("granite") is None                  # honest None when unknown

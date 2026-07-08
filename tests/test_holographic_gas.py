"""Gas (T1): ideal gas law reproduces air density & speed of sound (cross-checks definitions), adiabatic, boiling."""
import numpy as np
from holographic.simulation_and_physics.holographic_gas import gas_density, speed_of_sound, adiabatic, boiling_point, IdealGas, specific_gas_constant


def test_air_density_and_sound_speed_match_definitions():
    from holographic.misc.holographic_definitions import MATERIALS
    assert abs(gas_density(101325.0, 293.15, "air") - 1.204) < 0.02
    a = speed_of_sound(293.15, "air")
    assert abs(a - 343.0) < 3.0
    assert abs(a - MATERIALS["air"]["sound_speed"]) < 5.0          # derived == tabulated (two roads, one number)
    assert speed_of_sound(293.15, "helium") > 2.5 * a             # light gas -> faster sound


def test_adiabatic_compression_heats():
    p2, t2 = adiabatic(101325.0, 293.15, 0.5, "air")
    assert t2 > 293.15 and p2 > 101325.0
    _, t3 = adiabatic(101325.0, 293.15, 2.0, "air")
    assert t3 < 293.15                                             # expansion cools


def test_boiling_point_tracks_pressure():
    assert abs(boiling_point(101325.0) - 373.15) < 0.5            # 100 C at 1 atm
    assert 355.0 < boiling_point(70000.0) < 373.15                # cooler up a mountain
    assert boiling_point(2 * 101325.0) > 373.15                   # hotter in a pressure cooker


def test_ideal_gas_object_and_determinism():
    g = IdealGas("air", 293.15, 101325.0)
    assert abs(g.density() - 1.204) < 0.02 and abs(g.sound_speed() - 343.0) < 3.0
    assert specific_gas_constant("air") > 280 and specific_gas_constant("air") < 290
    assert speed_of_sound(300.0, "air") == speed_of_sound(300.0, "air")


def test_molar_mass_derived_from_composition():
    """Wrap-up: a gas not in GAS_PROPERTIES gets its molar mass from its elemental composition; table gases
    cross-check against composition."""
    from holographic.simulation_and_physics.holographic_gas import molar_mass_of, GAS_PROPERTIES
    from holographic.simulation_and_physics.holographic_elements import material_elemental
    assert "methane" not in GAS_PROPERTIES
    assert abs(molar_mass_of("methane") - material_elemental("methane")["molar_mass"] / 1000.0) < 1e-9
    assert abs(molar_mass_of("carbon_dioxide") - 0.04401) < 1e-4   # table value
    assert abs(material_elemental("carbon_dioxide")["molar_mass"] / 1000.0 - 0.04401) < 1e-4   # composition agrees

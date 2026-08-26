"""Phase change (M5): the boiling plateau, melt/freeze, pressure-dependent boiling point, determinism."""
import numpy as np
from holographic.misc.holographic_phase import PhaseState, boiling_point_at, PHASE_DATA


def test_boiling_plateau_holds_temperature():
    ps = PhaseState("water", 1.0, temp_K=372.15)                   # just below boiling
    temps, gas = [], []
    for _ in range(40):
        ps.add_heat(1.0e5)                                          # 100 kJ steps
        temps.append(ps.T); gas.append(ps.gas)
    temps = np.array(temps)
    at_boil = np.abs(temps - 373.15) < 0.2
    assert at_boil.sum() >= 15                                      # temperature holds at 100 C
    assert gas[-1] > gas[0]                                         # liquid became steam during the hold
    assert 2.0e6 < at_boil.sum() * 1.0e5 < 2.6e6                    # ~ the latent heat of vaporization


def test_melting_and_freezing_hold_at_zero():
    ice = PhaseState("water", 1.0, temp_K=272.15)
    assert ice.dominant_phase() == "solid"
    ice.add_heat(3.34e5 * 0.5 + 2090.0)                             # reach 0 C then melt ~half
    assert abs(ice.T - 273.15) < 0.5 and 0.2 < ice.phase_fractions()["liquid"] < 0.8
    water = PhaseState("water", 1.0, temp_K=273.15); water.liquid = 1.0; water.solid = 0.0
    water.add_heat(-3.34e5 * 0.5)                                   # freeze half
    assert abs(water.T - 273.15) < 0.5 and water.solid > 0.3


def test_boiling_point_tracks_pressure():
    assert abs(boiling_point_at("water", 101325.0) - 373.15) < 0.5
    assert boiling_point_at("water", 70000.0) < 373.15             # boils cooler up a mountain


def test_heating_water_fully_to_steam_conserves_mass():
    ps = PhaseState("water", 2.0, temp_K=293.15)
    ps.add_heat(1e7)                                               # plenty of energy
    assert abs(ps.total_mass() - 2.0) < 1e-9                       # mass conserved across the transitions
    assert ps.gas > 1.0                                            # most of it boiled to steam


def test_deterministic():
    a = PhaseState("water", 1.0, 372.15); a.add_heat(5e5)
    b = PhaseState("water", 1.0, 372.15); b.add_heat(5e5)
    assert abs(a.T - b.T) < 1e-12 and abs(a.gas - b.gas) < 1e-12

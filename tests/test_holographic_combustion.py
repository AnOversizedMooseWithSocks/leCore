"""Combustion (M6): per-material ignition thresholds, wood!=plastic smoke, lit fire sustains then burns out."""
import numpy as np
from holographic.simulation_and_physics.holographic_combustion import COMBUSTION, ignites, flame_color, combustion_products, Fire, configure_fluid, emit_smoke


def test_ignition_respects_each_material_threshold():
    assert ignites("wood", 600.0) and not ignites("wood", 500.0)
    assert not ignites("pvc_plastic", 600.0) and ignites("pvc_plastic", 750.0)   # PVC needs more heat
    assert not ignites("wood", 293.15)                              # room temperature: nothing burns
    assert not ignites("granite", 5000.0)                           # non-flammable material never ignites


def test_wood_smoke_differs_from_plastic_smoke():
    wood = combustion_products("wood", 1.0); pvc = combustion_products("pvc_plastic", 1.0)
    assert wood["smoke_color"].mean() > 0.4                          # wood: pale grey
    assert pvc["smoke_color"].mean() < 0.2                           # PVC: black
    assert pvc["soot_mass"] > wood["soot_mass"]                      # PVC much sootier
    assert COMBUSTION["ethanol"]["soot_yield"] < COMBUSTION["coal"]["soot_yield"]   # clean vs dirty


def test_flame_colour_from_blackbody():
    cool = flame_color(900.0); hot = flame_color(1600.0)
    assert cool[0] > cool[2]                                         # cool smoulder red-dominant
    assert hot[2] >= cool[2]                                         # blue rises with flame temperature


def test_lit_fire_sustains_then_burns_out():
    cold = Fire("wood", 1.0, temp_K=293.15)
    r = cold.step(1.0)
    assert not r["burning"] and r["burned_kg"] == 0.0               # too cold to light
    lit = Fire("wood", 1.0, temp_K=900.0)
    fuel = []
    for _ in range(60):
        fuel.append(lit.step(0.5)["fuel_left"])
    assert fuel[0] < 1.0 and fuel[-1] < fuel[0] * 0.2              # burned down over time
    assert all(fuel[i + 1] <= fuel[i] + 1e-12 for i in range(len(fuel) - 1))   # monotonic


def test_couplings_configure_fluid_and_colour_emitter():
    class _Fluid:                                                   # a stand-in with the same attributes
        ignition = 0.0; burn_rate = 0.0; smoke_yield = 0.0
    f = configure_fluid(_Fluid(), "gasoline")
    assert f.ignition == COMBUSTION["gasoline"]["autoignition_K"] and f.burn_rate == COMBUSTION["gasoline"]["burn_rate"]
    sphere = lambda P: np.linalg.norm(np.atleast_2d(P), axis=1) - 1.0
    pos, vel, colors, soot = emit_smoke(sphere, "pvc_plastic", 40, ((-2, -2, -2), (2, 2, 2)))
    assert len(pos) > 0 and colors.shape[1] == 3 and colors.mean() < 0.2 and soot > 0.8   # black, sooty


def test_deterministic():
    a = Fire("gasoline", 1.0, 900.0).step(0.3); b = Fire("gasoline", 1.0, 900.0).step(0.3)
    assert a["burned_kg"] == b["burned_kg"] and np.array_equal(a["smoke_color"], b["smoke_color"])


def test_flame_tint_from_element_composition():
    """Wrap-up: a copper-bearing fire tints green (emission lines) over the pure thermal blackbody glow."""
    import numpy as np
    from holographic.simulation_and_physics.holographic_combustion import flame_color
    thermal = flame_color(1500.0)
    copper = flame_color(1500.0, material="copper")
    assert copper[1] > thermal[1]                                  # greener than the thermal glow
    assert not np.array_equal(thermal, copper)
    assert np.array_equal(flame_color(1500.0), flame_color(1500.0, material="granite"))  # no composition -> unchanged

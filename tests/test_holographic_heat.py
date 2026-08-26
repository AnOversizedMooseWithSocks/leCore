"""Heat (T4): Q=mcdT, conduction conserves heat & smooths (stable at big dt), Newton cooling; conductivity from data."""
import numpy as np
from holographic.simulation_and_physics.holographic_heat import heat_energy, temperature_change, thermal_diffusivity, diffuse_heat, HeatBody, material_thermal


def test_specific_heat_roundtrip():
    Q = heat_energy(1.0, 4186.0, 20.0)
    assert abs(Q - 83720.0) < 1.0
    assert abs(temperature_change(Q, 1.0, 4186.0) - 20.0) < 1e-9


def test_conduction_conserves_heat_and_smooths():
    T = np.full((21, 21), 300.0); T[10, 10] = 800.0
    t0, v0 = T.sum(), T.var()
    T2 = diffuse_heat(T, alpha=1e-4, dx=0.01, dt=0.5, steps=20)
    assert abs(T2.sum() - t0) < 1e-6                                # insulated -> conserved
    assert T2.var() < v0 and T2.max() < 800.0 and np.isfinite(T2).all()


def test_big_dt_stays_stable():
    T = np.full((15, 15), 300.0); T[7, 7] = 900.0
    T2 = diffuse_heat(T, alpha=1e-3, dx=0.01, dt=100.0, steps=1)    # far past the explicit limit
    assert np.isfinite(T2).all() and abs(T2.sum() - T.sum()) < 1e-6


def test_newton_cooling_relaxes_toward_ambient():
    b = HeatBody(1.0, material_thermal("steel")["specific_heat"], temp_K=800.0)
    t1 = b.newton_cool(300.0, 2.0, 10.0)
    t2 = b.newton_cool(300.0, 2.0, 10.0)
    assert 300.0 < t2 < t1 < 800.0                                 # monotonic toward ambient
    assert (800.0 - t1) > (t1 - t2)                                # faster while hotter


def test_conductivity_read_from_enrichment_data():
    assert abs(material_thermal("steel")["thermal_conductivity"] - 50.0) < 1e-9    # from enrich.json, not restated
    assert abs(material_thermal("water")["thermal_conductivity"] - 0.6) < 1e-9
    a = thermal_diffusivity(50.0, 7850.0, 490.0)
    assert a > 0

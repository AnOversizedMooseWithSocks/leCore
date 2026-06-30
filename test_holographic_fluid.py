"""Tests for the stable-fluids solver (Stam 1999): smoke, buoyancy, fire, incompressibility (FLUID-1)."""
import numpy as np
from holographic_fluid import StableFluid


def test_projection_makes_velocity_divergence_free():
    rng = np.random.default_rng(0)
    f = StableFluid((32, 32), dt=0.1)
    f.vel = rng.standard_normal(f.vel.shape)
    before = f.divergence()
    f.vel = f.project(f.vel)
    assert f.divergence() < 1e-9 < before                          # incompressible to machine precision


def test_projection_divergence_free_in_3d():
    rng = np.random.default_rng(1)
    f = StableFluid((16, 16, 16), dt=0.1)
    f.vel = rng.standard_normal(f.vel.shape)
    f.vel = f.project(f.vel)
    assert f.divergence() < 1e-9


def test_semi_lagrangian_is_unconditionally_stable():
    f = StableFluid((48, 48), dt=2.0, vorticity=3.0)               # a wildly large timestep
    f.add_source((slice(20, 28), slice(20, 28)), density=1.0, temperature=2.0)
    for _ in range(50):
        f.step()
    assert np.isfinite(f.vel).all()                                # never blows up to NaN/inf (an explicit solver would)


def test_buoyancy_lifts_a_hot_plume():
    f = StableFluid((48, 48), dt=0.5, buoyancy_beta=0.4, vorticity=0.0,
                    dissipation=0.0, cooling=0.0, up_axis=0)
    f.add_source((slice(34, 40), slice(20, 28)), density=1.0, temperature=3.0)
    rows = np.arange(48)[:, None]
    com0 = (f.density * rows).sum() / f.density.sum()
    for _ in range(30):
        f.step()
    com1 = (f.density * rows).sum() / f.density.sum()
    assert com1 < com0 - 1.0                                       # centre of mass rose (row index decreased)


def test_combustion_burns_fuel_into_heat_and_smoke():
    f = StableFluid((48, 48), dt=0.3, ignition=0.5, burn_rate=2.0, smoke_yield=0.4)
    f.add_source((slice(20, 28), slice(20, 28)), fuel=1.0, temperature=1.0)
    fuel0 = f.fuel.sum()
    for _ in range(15):
        f.step()
    assert f.fuel.sum() < 0.2 * fuel0                              # fuel consumed
    assert f.density.sum() > 0.0                                   # smoke produced by the reaction


def test_vorticity_confinement_preserves_swirl():
    def enstrophy(g):
        w = g._d(g.vel[1], 0) - g._d(g.vel[0], 1)
        return float((w ** 2).sum())
    out = {}
    for eps in (0.0, 4.0):
        g = StableFluid((48, 48), dt=0.5, vorticity=eps, buoyancy_beta=0.4, dissipation=0.0)
        g.add_source((slice(30, 38), slice(20, 28)), density=1.0, temperature=3.0)
        for _ in range(30):
            g.step()
        out[eps] = enstrophy(g)
    assert out[4.0] > 2.0 * out[0.0]                               # confinement keeps substantially more swirl


def test_semi_lagrangian_mass_dissipation_kept_negative():
    f = StableFluid((48, 48), dt=0.5, vorticity=0.0, dissipation=0.0, cooling=0.0, buoyancy_beta=0.3)
    f.add_source((slice(30, 38), slice(20, 28)), density=1.0, temperature=1.0)
    m0 = f.density.sum()
    for _ in range(50):
        f.step()
    assert f.density.sum() < m0                                    # documents the known dissipation (mass drifts down)


def test_device_cpu_is_numpy_and_runs():
    import numpy as np
    f = StableFluid((32, 32), dt=0.5, vorticity=3.0, buoyancy_beta=0.4, device="cpu")
    assert f.xp is np                                              # cpu device == NumPy (byte-identical path)
    f.add_source((slice(20, 26), slice(12, 20)), density=1.0, temperature=2.0)
    for _ in range(15):
        f.step()
    assert np.isfinite(f.vel).all()
    assert isinstance(f.to_numpy("density"), np.ndarray)

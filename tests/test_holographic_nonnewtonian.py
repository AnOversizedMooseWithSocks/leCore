"""Non-Newtonian (cornstarch): power law thickens under shear for n>1, thins for n<1; viscous step; fluid mode."""
import numpy as np
from holographic.simulation_and_physics.holographic_nonnewtonian import power_law_viscosity, strain_rate_magnitude, strain_rate_tensor, viscous_step, effective_viscosity, _shear_flow


def test_power_law_thickens_thins_and_clamps():
    slow, fast = 0.5, 50.0
    assert power_law_viscosity(fast, 1.0, 1.8) > power_law_viscosity(slow, 1.0, 1.8) * 5    # cornstarch stiffens
    assert power_law_viscosity(fast, 1.0, 0.5) < power_law_viscosity(slow, 1.0, 0.5)        # shear-thinning
    assert abs(power_law_viscosity(fast, 2.0, 1.0) - power_law_viscosity(slow, 2.0, 1.0)) < 1e-9   # Newtonian flat
    assert power_law_viscosity(1e9, 1.0, 2.0) <= 1e4 + 1e-6                                  # clamped


def test_strain_rate_field():
    uniform = np.stack([np.ones((8, 8)), np.zeros((8, 8))])
    assert strain_rate_magnitude(uniform).max() < 1e-9                                       # no deformation
    interior = strain_rate_magnitude(_shear_flow(8, 2.0))[1:-1, 1:-1]
    assert interior.std() < 1e-6 and interior.mean() > 0.5                                   # uniform, nonzero


def test_effective_viscosity_signature():
    assert effective_viscosity(_shear_flow(12, 20.0), 1.0, 1.8) > effective_viscosity(_shear_flow(12, 0.5), 1.0, 1.8) * 5
    assert abs(effective_viscosity(_shear_flow(12, 20.0), 1.0, 1.0) - effective_viscosity(_shear_flow(12, 0.5), 1.0, 1.0)) < 1e-9


def test_viscous_step_cornstarch_resists_more():
    grid = 20; yy = np.arange(grid, dtype=float)[:, None] * np.ones((1, grid))
    v0 = np.stack([15.0 * np.sin(2 * np.pi * 2 * yy / grid), np.zeros((grid, grid))])
    E0 = float((v0 ** 2).sum())
    v_corn, eta = viscous_step(v0.copy(), 0.05, 1.8, 0.1)
    v_newt, _ = viscous_step(v0.copy(), 0.05, 1.0, 0.1)
    assert np.isfinite(v_corn).all()
    assert (v_corn ** 2).sum() < E0 and (v_corn ** 2).sum() < (v_newt ** 2).sum()            # cornstarch damps more
    assert eta.max() > eta.min()                                                             # eta is a field


def test_fluid_mode_and_backward_compat():
    from holographic.simulation_and_physics.holographic_fluid import StableFluid
    def one_step(n):
        f = StableFluid((40, 40), dt=0.1, viscosity=0.02, vorticity=0.0, dissipation=0.0, cooling=0.0,
                        power_law_n=n, consistency_K=0.06)
        rows = np.arange(40)[:, None] * np.ones((1, 40))
        f.vel = f.xp.asarray(np.stack([np.zeros((40, 40)), 12.0 * np.sin(2 * np.pi * 3 * rows / 40)]))
        E0 = float((np.asarray(f.vel) ** 2).sum()); f.step()
        return E0, float((np.asarray(f.vel) ** 2).sum())
    E0n, En = one_step(1.0); E0c, Ec = one_step(1.8)
    assert np.isfinite(En) and np.isfinite(Ec) and Ec < En                                   # cornstarch resists more, stable
    # n=1 is byte-identical to not passing the param (backward-compatible)
    a = StableFluid((20, 20), viscosity=0.03, vorticity=0.5)
    b = StableFluid((20, 20), viscosity=0.03, vorticity=0.5, power_law_n=1.0)
    a.density[10, 10] = b.density[10, 10] = 1.0; a.vel[0, 5, 5] = b.vel[0, 5, 5] = 1.0
    for _ in range(3):
        a.step(); b.step()
    assert np.array_equal(np.asarray(a.density), np.asarray(b.density))
    assert np.array_equal(np.asarray(a.vel), np.asarray(b.vel))


def test_deterministic():
    a, _ = viscous_step(_shear_flow(10, 5.0), 0.02, 1.8, 0.05)
    b, _ = viscous_step(_shear_flow(10, 5.0), 0.02, 1.8, 0.05)
    assert np.array_equal(a, b)

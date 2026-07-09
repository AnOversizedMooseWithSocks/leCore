"""holographic_nonnewtonian.py -- NON-NEWTONIAN viscosity: cornstarch and friends, where thickness depends on how
fast you shear.

WHY THIS EXISTS
---------------
The A5 cymatics cornstarch is a PLATE-SURFACE phenomenology (it holds peaks). But to honestly claim we can
"simulate cornstarch", the FLUID solver itself needs non-Newtonian rheology: its viscosity must stop being a
single constant and become a FUNCTION of the local shear rate. That is the power-law (Ostwald-de Waele) model,
and it is exactly the snippet Moose supplied:

    eta = K * shear_rate^(n-1),  clamped

  * n > 1  -> DILATANT / shear-THICKENING: stiffens the harder you shear it. This is cornstarch-in-water
    ("oobleck") -- punch it and it resists like a solid; let your hand rest and it flows. (n ~ 1.5-2.0)
  * n < 1  -> shear-THINNING: thins under shear -- ketchup, paint, blood, quicksand. (n ~ 0.3-0.7)
  * n = 1  -> Newtonian: eta = K, a constant (water, air) -- the solver's original behaviour, unchanged.

This module supplies the viscosity law, the shear-rate (strain-rate) field it depends on, and a variable-
viscosity viscous step  dv = dt * (1/rho) * div( eta * (grad v + grad v^T) )  that replaces the solver's single
constant-viscosity diffuse. It plugs into holographic_fluid.StableFluid as an opt-in mode (n != 1), so the
incompressible solver can carry a real shear-thickening or shear-thinning fluid while staying byte-identical for
plain Newtonian flow.

THE PHYSICS (readable)
----------------------
The rate-of-strain tensor e_ij = 1/2 (dv_i/dx_j + dv_j/dx_i) measures how fast the fluid is deforming; its
magnitude gamma_dot = sqrt(2 e_ij e_ij) is the "shear rate" the viscosity law reads. The viscous stress is
tau_ij = 2 eta e_ij, and the force per volume is its divergence div(tau) -- momentum diffusing between cells,
but now FASTER where eta is high (a thickened region locks up) and slower where eta is low (a thin region
flows). An explicit step has a diffusion stability limit ~ dx^2/(2*ndim*eta); this AUTO-SUBSTEPS to stay stable
when eta spikes (which it does, for cornstarch).

HONEST SCOPE (kept negative): the power-law (Ostwald-de Waele) model -- a good, standard fit for the
shear-thickening/thinning REGIME, but it has no yield stress (Herschel-Bulkley) and no time-dependence
(thixotropy); the clamp [eta_min, eta_max] keeps it numerically finite (real oobleck's stress can run away).
2-D, unit density by default. Deterministic; NumPy + stdlib.
"""
import numpy as np


def power_law_viscosity(shear_rate, K, n, eta_min=1e-4, eta_max=1e4):
    """The Ostwald-de Waele power law: eta = K * shear_rate^(n-1), clamped so it can neither blow up nor vanish
    numerically. Returns a viscosity FIELD (one value per cell). n>1 thickens with shear (cornstarch), n<1 thins
    (ketchup), n=1 is the constant K (Newtonian). This is Moose's snippet, verbatim in spirit."""
    eta = K * np.power(np.maximum(np.asarray(shear_rate, float), 1e-6), n - 1.0)
    return np.clip(eta, eta_min, eta_max)


def _grad(field, dx):
    """(d/dx, d/dy) of a 2-D field. np.gradient returns [d/d(row)=d/dy, d/d(col)=d/dx]; we return (d/dx, d/dy)."""
    dfy, dfx = np.gradient(field, dx)
    return dfx, dfy


def strain_rate_tensor(vel, dx=1.0):
    """The rate-of-strain tensor components (e_xx, e_yy, e_xy) of a 2-D velocity field vel=(vx,vy). e_ij measures
    how fast the fluid is stretching/shearing -- the thing the viscosity law responds to."""
    vx, vy = vel[0], vel[1]
    dvx_dx, dvx_dy = _grad(vx, dx)
    dvy_dx, dvy_dy = _grad(vy, dx)
    exx = dvx_dx
    eyy = dvy_dy
    exy = 0.5 * (dvx_dy + dvy_dx)
    return exx, eyy, exy


def strain_rate_magnitude(vel, dx=1.0):
    """The scalar shear rate gamma_dot = sqrt(2 e_ij e_ij) -- zero for a uniform flow (no deformation), large in a
    strong shear. This is what power_law_viscosity reads to decide how thick the fluid is, cell by cell."""
    exx, eyy, exy = strain_rate_tensor(vel, dx)
    return np.sqrt(2.0 * (exx ** 2 + eyy ** 2 + 2.0 * exy ** 2))


def viscous_step(vel, K, n, dt, dx=1.0, rho=1.0, eta_min=1e-4, eta_max=1e4):
    """One variable-viscosity viscous update of a 2-D velocity field: dv = dt*(1/rho)*div(eta*(grad v+grad v^T)),
    with eta = power_law_viscosity(shear_rate). Momentum diffuses FAST where the fluid has thickened (high shear
    for cornstarch) and slow where it is thin. Auto-substeps to stay stable when eta spikes. Returns the new vel
    and the eta field (so callers can see where it thickened)."""
    vel = np.asarray(vel, float)
    gamma = strain_rate_magnitude(vel, dx)
    eta = power_law_viscosity(gamma, K, n, eta_min, eta_max)
    # explicit diffusion stability: dt_stable ~ dx^2 / (2*ndim*eta_max); subdivide a big/stiff step
    diff_max = float(eta.max()) / rho
    dt_stable = 0.9 * dx * dx / (2.0 * 2.0 * max(diff_max, 1e-12))
    n_sub = max(1, int(np.ceil(dt / dt_stable)))
    h = dt / n_sub
    for _ in range(n_sub):
        # recompute eta each substep (shear changes as momentum diffuses); tau_ij = 2 eta e_ij
        gamma = strain_rate_magnitude(vel, dx)
        eta = power_law_viscosity(gamma, K, n, eta_min, eta_max)
        exx, eyy, exy = strain_rate_tensor(vel, dx)
        txx = 2.0 * eta * exx; tyy = 2.0 * eta * eyy; txy = 2.0 * eta * exy
        dtxx_dx, _ = _grad(txx, dx)
        dtxy_dx, dtxy_dy = _grad(txy, dx)
        _, dtyy_dy = _grad(tyy, dx)
        fx = (dtxx_dx + dtxy_dy) / rho                             # div(tau) x-component
        fy = (dtxy_dx + dtyy_dy) / rho                             # div(tau) y-component
        vel = np.stack([vel[0] + h * fx, vel[1] + h * fy])
    return vel, eta


def effective_viscosity(vel, K, n, dx=1.0):
    """The mean viscosity the fluid currently shows under its own flow -- a single honest number for 'how thick is
    it right now'. Rises with shear for cornstarch (n>1), falls for shear-thinning (n<1)."""
    return float(power_law_viscosity(strain_rate_magnitude(vel, dx), K, n).mean())


def _shear_flow(grid, rate):
    """A test velocity field: simple shear vx = rate * y, vy = 0 -- a uniform shear rate of `rate` everywhere.
    The textbook flow for probing a viscosity law."""
    y = np.arange(grid, dtype=float)[:, None] * np.ones((1, grid))
    vx = rate * y
    return np.stack([vx, np.zeros((grid, grid))])


def _selftest():
    """The power law thickens under shear for n>1 (cornstarch) and thins for n<1; the shear-rate field is right;
    a thickened fluid damps its shear faster; n=1 is Newtonian. Deterministic."""
    # (1) THE cornstarch property: eta RISES with shear rate when n>1, FALLS when n<1, is FLAT when n=1
    slow, fast = 0.5, 50.0
    assert power_law_viscosity(fast, 1.0, 1.8) > power_law_viscosity(slow, 1.0, 1.8) * 5    # dilatant: much stiffer fast
    assert power_law_viscosity(fast, 1.0, 0.5) < power_law_viscosity(slow, 1.0, 0.5)        # shear-thinning
    assert abs(power_law_viscosity(fast, 2.0, 1.0) - power_law_viscosity(slow, 2.0, 1.0)) < 1e-9   # Newtonian flat
    assert power_law_viscosity(1e9, 1.0, 2.0) <= 1e4 + 1e-6                                  # clamped, no blow-up

    # (2) shear-rate field: a uniform flow has ZERO strain rate; a simple shear has a constant nonzero one
    uniform = np.stack([np.ones((8, 8)), np.zeros((8, 8))])
    assert strain_rate_magnitude(uniform).max() < 1e-9
    sh = _shear_flow(8, rate=2.0)
    interior = strain_rate_magnitude(sh)[1:-1, 1:-1]
    assert interior.std() < 1e-6 and interior.mean() > 0.5                                  # uniform, nonzero

    # (3) cornstarch shows a HIGHER effective viscosity under fast shear than slow -- the signature (Newtonian doesn't)
    corn_fast = effective_viscosity(_shear_flow(12, 20.0), K=1.0, n=1.8)
    corn_slow = effective_viscosity(_shear_flow(12, 0.5), K=1.0, n=1.8)
    assert corn_fast > corn_slow * 5.0
    newt_fast = effective_viscosity(_shear_flow(12, 20.0), K=1.0, n=1.0)
    newt_slow = effective_viscosity(_shear_flow(12, 0.5), K=1.0, n=1.0)
    assert abs(newt_fast - newt_slow) < 1e-9                                                # Newtonian: same thickness

    # (4) the viscous step damps a CURVED shear (a linear shear has no curvature to diffuse); cornstarch, thicker
    # in the high-shear bands, damps it MORE than Newtonian
    grid = 20
    yy = np.arange(grid, dtype=float)[:, None] * np.ones((1, grid))
    vx = 15.0 * np.sin(2 * np.pi * 2 * yy / grid)                  # a sinusoidal shear profile (has curvature)
    v0 = np.stack([vx, np.zeros((grid, grid))])
    E0 = float((v0 ** 2).sum())
    v_corn, eta_corn = viscous_step(v0.copy(), K=0.05, n=1.8, dt=0.1)
    v_newt, _ = viscous_step(v0.copy(), K=0.05, n=1.0, dt=0.1)
    E_corn = float((v_corn ** 2).sum()); E_newt = float((v_newt ** 2).sum())
    assert np.isfinite(v_corn).all()                                                        # stayed stable (substepped)
    assert E_corn < E0 and E_corn < E_newt                                                  # cornstarch resists/damps more
    assert eta_corn.max() > eta_corn.min()                                                  # eta is a FIELD, not a constant

    # (5) deterministic
    a, _ = viscous_step(_shear_flow(10, 5.0), 0.02, 1.8, 0.05)
    b, _ = viscous_step(_shear_flow(10, 5.0), 0.02, 1.8, 0.05)
    assert np.array_equal(a, b)
    print("holographic_nonnewtonian selftest OK: eta=K*gamma^(n-1) thickens under shear for n>1 (cornstarch: %.1fx "
          "stiffer fast vs slow), thins for n<1, flat for n=1; viscous step locks up a fast-sheared cornstarch "
          "more than water; deterministic" % (corn_fast / corn_slow))


if __name__ == "__main__":
    _selftest()

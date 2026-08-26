"""A1 -- one discrete Laplacian, boundary condition as a parameter (was duplicated in heat and wave)."""
import numpy as np
import pytest

from holographic.simulation_and_physics.holographic_laplacian import laplacian, is_circular


def test_linear_in_every_dimension_and_bc():
    rng = np.random.default_rng(0)
    for bc in ("neumann", "periodic", "dirichlet"):
        for shape in [(9,), (7, 8), (4, 5, 6)]:
            a, b = rng.standard_normal(shape), rng.standard_normal(shape)
            assert np.allclose(laplacian(a + b, bc), laplacian(a, bc) + laplacian(b, bc))
            assert np.allclose(laplacian(2.5 * a, bc), 2.5 * laplacian(a, bc))


def test_neumann_conserves_the_total():
    f = np.random.default_rng(1).standard_normal((12, 12))
    assert abs(laplacian(f, "neumann").sum()) < 1e-9      # zero-flux: nothing leaves the domain


def test_constant_field_has_zero_laplacian_except_at_a_dirichlet_wall():
    c = np.full((6, 6), 3.0)
    assert np.allclose(laplacian(c, "neumann"), 0.0)
    assert np.allclose(laplacian(c, "periodic"), 0.0)
    assert not np.allclose(laplacian(c, "dirichlet"), 0.0)   # the wall is a step


def test_only_periodic_is_a_circular_convolution():
    """This is the measurement that explains why iterate.step_k cannot be wired into the PDE solvers."""
    x = np.random.default_rng(2).standard_normal(16)
    kern = np.zeros(16); kern[0] = -2.0; kern[1] = 1.0; kern[-1] = 1.0
    circ = np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(kern), n=16)
    assert np.allclose(laplacian(x, "periodic"), circ)
    assert not np.allclose(laplacian(x, "neumann"), circ)
    assert is_circular("periodic") and not is_circular("neumann")


def test_heat_and_wave_delegate_and_are_unchanged():
    from holographic.simulation_and_physics.holographic_heat import _laplacian as h
    from holographic.simulation_and_physics.holographic_wave import _laplacian as w
    rng = np.random.default_rng(3)
    for shape in [(9,), (7, 8), (4, 5, 6)]:
        f = rng.standard_normal(shape)
        assert np.allclose(h(f), laplacian(f, "neumann"))
        assert np.allclose(w(f), laplacian(f, "neumann"))


def test_unknown_bc_raises():
    with pytest.raises(ValueError):
        laplacian(np.zeros(4), bc="nonsense")


# ---------------------------------------------------------------------------------------------------------------
# L5/M2 -- the closed form. A linear, CIRCULAR operator diagonalises in the FFT, so the solve needs no iteration.
# ---------------------------------------------------------------------------------------------------------------
def _mode_2d(n=64):
    xs = np.arange(n) / n
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    return xs, u - u.mean()


def test_spectral_poisson_is_exact_on_band_limited_data():
    from holographic.simulation_and_physics.holographic_laplacian import solve_poisson_spectral
    n = 64
    _, u_true = _mode_2d(n)
    f = -8.0 * np.pi ** 2 * u_true                       # the exact continuous laplacian of u_true
    u = solve_poisson_spectral(f, dx=1.0 / n)
    assert np.max(np.abs(u - u_true)) < 1e-12            # measured 6.7e-16


def test_spectral_poisson_enforces_solvability_and_zero_mean():
    from holographic.simulation_and_physics.holographic_laplacian import solve_poisson_spectral
    f = np.ones((16, 16))                                # a non-zero-mean source has no periodic solution
    u = solve_poisson_spectral(f)                        # the mean is subtracted explicitly, not silently ignored
    assert abs(u.mean()) < 1e-12
    assert np.isfinite(u).all()


def test_spectral_heat_is_exact_for_any_t_in_one_evaluation():
    from holographic.simulation_and_physics.holographic_laplacian import diffuse_spectral
    n = 64
    xs = np.arange(n) / n
    T0 = np.sin(2 * np.pi * xs)
    alpha = 0.01
    for t in (0.5, 2.0, 50.0):
        exact = np.exp(-alpha * (2 * np.pi) ** 2 * t) * T0
        assert np.max(np.abs(diffuse_spectral(T0, alpha, t, dx=1.0 / n) - exact)) < 1e-12
    # a huge t has no stability limit and costs exactly the same
    assert np.isfinite(diffuse_spectral(T0, alpha, 1e6, dx=1.0 / n)).all()


def test_spectral_beats_iterative_stepping():
    """The point of the closed form: an iterative stepper accumulates truncation error at every step."""
    from holographic.simulation_and_physics.holographic_laplacian import diffuse_spectral
    n = 64
    xs = np.arange(n) / n
    T0 = np.sin(2 * np.pi * xs)
    alpha, t, dx = 0.01, 2.0, 1.0 / n
    exact = np.exp(-alpha * (2 * np.pi) ** 2 * t) * T0

    def iterative(steps):
        T = T0.copy()
        r = alpha * (t / steps) / dx ** 2
        sub = max(1, int(np.ceil(r / 0.45)))
        rs = r / sub
        for _ in range(steps * sub):
            T = T + rs * (np.roll(T, 1) + np.roll(T, -1) - 2 * T)
        return T

    e_spectral = np.max(np.abs(diffuse_spectral(T0, alpha, t, dx=dx) - exact))
    e_iter = np.max(np.abs(iterative(1000) - exact))
    assert e_spectral < 1e-12 and e_iter > 1e-5
    assert e_spectral < e_iter / 1e6                     # measured: 2e-16 vs 1.5e-4


def test_spectral_solvers_through_the_mind():
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    n = 32
    _, u_true = _mode_2d(n)
    u = m.solve_poisson_periodic(-8.0 * np.pi ** 2 * u_true, dx=1.0 / n)
    assert np.max(np.abs(u - u_true)) < 1e-10
    T = m.diffuse_periodic(np.sin(2 * np.pi * np.arange(n) / n), alpha=0.01, t=1.0, dx=1.0 / n)
    assert np.isfinite(T).all() and np.max(np.abs(T)) < 1.0     # it decayed


def test_heat_periodic_boundary_uses_the_exact_closed_form():
    """A DEFERRED item, built. bc='periodic' makes the Laplacian circular, so the whole evolution to any t is ONE
    exact evaluation. The Neumann default is unchanged (its eigenbasis is the DCT, not the DFT)."""
    from holographic.simulation_and_physics.holographic_heat import diffuse_heat
    n = 64
    xs = np.arange(n) / n
    T0 = np.sin(2 * np.pi * xs)
    alpha, t = 0.01, 2.0
    exact = np.exp(-alpha * (2 * np.pi) ** 2 * t) * T0
    got = diffuse_heat(T0, alpha, dx=1.0 / n, dt=t, steps=1, bc="periodic")
    assert np.max(np.abs(got - exact)) < 1e-12                  # measured 2.2e-16
    assert np.isfinite(diffuse_heat(T0, alpha, dx=1.0 / n, dt=1e6, steps=1, bc="periodic")).all()  # no stability cap

    default = diffuse_heat(T0, alpha, dx=1.0 / n, dt=t, steps=1)
    assert abs(default.sum() - T0.sum()) < 1e-9                 # Neumann default unchanged: heat conserved
    with pytest.raises(ValueError):
        diffuse_heat(T0, alpha, bc="nonsense")


def test_dynamics_supplies_a_true_projection_for_the_constraint_engine():
    """The other DEFERRED item, built. I retracted `dynamics` from project_onto_constraints because bind() is not
    idempotent -- true, but the module DOES contain a projector: masking the persistent (|eigenvalue|~1) eigenspace.
    `limit()` is NOT that projector (it keeps the eigenvalue, so applying it twice multiplies by H^2)."""
    from holographic.misc.holographic_iterate import limit
    from holographic.rendering.holographic_denoise import project_onto_constraints
    from holographic.simulation_and_physics.holographic_dynamics import Propagator

    rng = np.random.default_rng(0)
    dim = 256
    nf = dim // 2 + 1
    mag = np.full(nf, 0.6)
    mag[:20] = 1.0                                              # a genuine persistent subspace
    ph = rng.uniform(-np.pi, np.pi, nf); ph[0] = 0.0; ph[-1] = 0.0
    U = np.fft.irfft(mag * np.exp(1j * ph), n=dim)
    P = Propagator(U, U)
    x = rng.standard_normal(dim)

    p1 = P.persistent_projection(x)
    assert np.max(np.abs(P.persistent_projection(p1) - p1)) < 1e-12      # idempotent -> a projection
    assert np.max(np.abs(P.persistent_projection(P.step(p1)) - P.step(p1))) < 1e-12   # subspace is invariant
    assert np.max(np.abs(limit(limit(x, U), U) - limit(x, U))) > 0.1     # limit() is NOT idempotent

    # and it composes with other constraints in the shared iterate-a-projection engine
    out, _sweeps, _conv = project_onto_constraints(
        x, [P.constraint(), lambda v: v / (np.linalg.norm(v) + 1e-12)], iters=20)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-6
    assert np.max(np.abs(P.persistent_projection(out) - out)) < 1e-9

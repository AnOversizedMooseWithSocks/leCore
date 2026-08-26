"""Physics backbone: SpectralField -- closed-form field evolution (diffusion, waves, ocean, electrostatics)."""
import numpy as np
from holographic.sampling_and_signal.holographic_spectralfield import SpectralField, diffusion_field, wave_field, ocean_field, phillips_spectrum, poisson_solve, wavenumbers


def test_parabolic_closed_form_equals_stepped():
    N = 128; x = np.arange(N)
    f0 = np.exp(-((x - N / 2) ** 2) / (2 * 4.0 ** 2))
    one = diffusion_field(f0.copy(), D=0.5, dx=1.0).advanced(20.0)
    st = diffusion_field(f0.copy(), D=0.5, dx=1.0); st.step(0.5, steps=40)
    assert np.max(np.abs(one - st.field)) < 1e-9


def test_diffusion_matches_analytic_and_beats_grid():
    from holographic.simulation_and_physics.holographic_heat import diffuse_heat
    N = 128; x = np.arange(N); s0 = 4.0; D = 0.5; T = 20.0
    f0 = np.exp(-((x - N / 2) ** 2) / (2 * s0 ** 2))
    analytic_peak = f0.max() * s0 / np.sqrt(s0 ** 2 + 2 * D * T)
    spectral = diffusion_field(f0.copy(), D=D, dx=1.0).advanced(T)
    grid = diffuse_heat(f0.copy().reshape(1, N), alpha=D, dx=1.0, dt=0.05, steps=400).ravel()
    # spectral is exact (closed form), grid accumulates step error -> spectral is strictly closer to analytic
    assert abs(spectral.max() - analytic_peak) < 1e-12
    assert abs(grid.max() - analytic_peak) > abs(spectral.max() - analytic_peak)


def test_steady_state_is_the_mean():
    f0 = np.random.default_rng(0).standard_normal((16, 16))
    ss = diffusion_field(f0.copy(), D=0.3, dx=1.0).steady_state()
    assert np.allclose(ss, f0.mean())


def test_wave_mode_returns_after_one_period():
    N = 128; x = np.arange(N); c = 2.0
    k = 2 * np.pi * 3 / N
    f0 = np.cos(k * x)
    period = 2 * np.pi / (c * k)
    back = wave_field(f0.copy(), c=c, dx=1.0).advanced(period)[0]
    assert np.max(np.abs(back - f0)) < 1e-8


def test_wave_closed_form_equals_stepped():
    N = 64; x = np.arange(N)
    f0 = np.sin(2 * np.pi * 2 * x / N)
    one = wave_field(f0.copy(), c=1.5, dx=1.0).advanced(1.3)[0]
    st = wave_field(f0.copy(), c=1.5, dx=1.0); st.step(0.13, steps=10)
    assert np.max(np.abs(one - st.field)) < 1e-8


def test_superposition_is_additive():
    N = 128; x = np.arange(N)
    a = np.exp(-((x - 40) ** 2) / 8.0); b = np.exp(-((x - 90) ** 2) / 8.0)
    s = diffusion_field((a + b).copy(), 0.5, 1.0).advanced(5.0)
    sa = diffusion_field(a.copy(), 0.5, 1.0).advanced(5.0)
    sb = diffusion_field(b.copy(), 0.5, 1.0).advanced(5.0)
    assert np.max(np.abs(s - (sa + sb))) < 1e-9


def test_ocean_real_deterministic_dispersive():
    h1 = phillips_spectrum((64, 64), wind=(15.0, 0.0), seed=1)
    h2 = phillips_spectrum((64, 64), wind=(15.0, 0.0), seed=1)
    assert np.isrealobj(h1) and np.array_equal(h1, h2)
    surf = ocean_field(h1.copy(), g=9.81, dx=1.0).advanced(2.0)[0]
    assert np.isrealobj(surf) and np.max(np.abs(surf)) > 0


def test_poisson_extremum_at_charge():
    src = np.zeros((64, 64)); src[32, 32] = 1.0; src -= src.mean()
    phi = poisson_solve(src, dx=1.0)
    assert phi[32, 32] == phi.max() or phi[32, 32] == phi.min()


def test_wavenumbers_shape():
    ks, kmag = wavenumbers((8, 16), dx=1.0)
    assert kmag.shape == (8, 16) and len(ks) == 2

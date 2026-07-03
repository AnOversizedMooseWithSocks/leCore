"""Fluids/matter item 2: Mixture + matter_step -- the multi-channel matter model (dye/milk)."""
import numpy as np
from holographic_mixture import Mixture, Component, matter_step, _blob, _spatial_spread


def _two_dye():
    shape = (48, 48)
    mix = Mixture(shape, solvent_density=1.0, buoyancy=0.0)
    mix.add("red", _blob(shape, 24, 16, 4.0), density=1.2, diffusivity=0.05)
    mix.add("blue", _blob(shape, 24, 32, 4.0), density=0.8, diffusivity=0.05)
    return mix


def test_density_blend_reads_components():
    mix = _two_dye()
    rho = mix.density()
    r = np.unravel_index(np.argmax(mix.channels["red"]), mix.shape)
    b = np.unravel_index(np.argmax(mix.channels["blue"]), mix.shape)
    assert rho[r] > 1.0 > rho[b]                              # heavy dye denser, light dye lighter than solvent


def test_pure_solvent_reads_solvent_density():
    mix = Mixture((16, 16), solvent_density=1.0)
    mix.add("x", np.zeros((16, 16)), density=5.0)            # channel present but empty
    assert np.allclose(mix.density(), 1.0)                    # all solvent


def test_channels_advect_and_diffuse():
    mix = _two_dye()
    vx = np.full(mix.shape, 0.5); vy = np.zeros(mix.shape)
    before = mix.channels["red"].copy(); spread0 = _spatial_spread(before)
    for _ in range(20):
        vx, vy = matter_step(mix, vx, vy, dt=0.1)
    assert not np.allclose(mix.channels["red"], before)       # moved
    assert _spatial_spread(mix.channels["red"]) > spread0      # spread


def test_mass_roughly_conserved():
    mix = _two_dye()
    vx = np.zeros(mix.shape); vy = np.zeros(mix.shape)
    m0 = mix.channels["red"].sum()
    for _ in range(15):
        vx, vy = matter_step(mix, vx, vy, dt=0.1)
    assert abs(mix.channels["red"].sum() - m0) / m0 < 0.1     # diffusion conserves mass on the torus


def test_renormalise_keeps_valid_partition():
    mix = Mixture((16, 16))
    mix.add("a", np.full((16, 16), 0.9), density=1.0)
    mix.add("b", np.full((16, 16), 0.9), density=1.0)         # sum 1.8 > 1 -> must scale back
    mix.renormalise()
    assert mix.total_fraction().max() <= 1.0 + 1e-6
    for phi in mix.channels.values():
        assert phi.min() >= -1e-9


def test_per_channel_diffusivity_differs():
    # a fast-diffusing channel spreads more than a slow one over the same steps (the salt-fingering precondition)
    shape = (40, 40)
    mix = Mixture(shape, buoyancy=0.0)
    mix.add("fast", _blob(shape, 20, 20, 3.0), diffusivity=0.2)
    mix.add("slow", _blob(shape, 20, 20, 3.0), diffusivity=0.001)
    vx = np.zeros(shape); vy = np.zeros(shape)
    for _ in range(15):
        vx, vy = matter_step(mix, vx, vy, dt=0.1)
    assert _spatial_spread(mix.channels["fast"]) > _spatial_spread(mix.channels["slow"])


def test_deterministic():
    a = _two_dye(); b = _two_dye()
    va = np.full(a.shape, 0.3); vb = np.full(b.shape, 0.3); z = np.zeros(a.shape)
    for _ in range(10):
        va, _ = matter_step(a, va, z.copy(), dt=0.1)
        vb, _ = matter_step(b, vb, z.copy(), dt=0.1)
    assert np.array_equal(a.channels["red"], b.channels["red"])

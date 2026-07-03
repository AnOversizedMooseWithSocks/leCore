"""Performance PW4: iterate sim readout -- diagonalize the linear diffusion sub-step, evaluate any t directly."""
import numpy as np
from holographic_fields import diffuse
from holographic_simreadout import diffusion_transfer, diffuse_at, diffuse_limit


def _field():
    return np.random.default_rng(0).standard_normal((32, 32))


def test_readout_matches_marching():
    field, amount = _field(), 0.15
    for k in (1, 3, 7, 20):
        marched = field.copy()
        for _ in range(k):
            marched = diffuse(marched, amount)
        assert np.allclose(diffuse_at(field, amount, k), marched, atol=1e-9)


def test_semigroup_k_steps_equals_one_big_step():
    # k diffusions by `amount` == one diffusion by k*amount (the heat semigroup)
    field, amount = _field(), 0.1
    assert np.allclose(diffuse_at(field, amount, 5), diffuse(field, amount * 5), atol=1e-9)


def test_fractional_step_interpolates():
    field, amount = _field(), 0.15
    half = np.abs(diffuse_at(field, amount, 2.5)).max()
    assert np.abs(diffuse_at(field, amount, 3.0)).max() <= half + 1e-9 <= np.abs(diffuse_at(field, amount, 2.0)).max() + 1e-9


def test_limit_is_mean():
    field = _field()
    assert np.allclose(diffuse_limit(field), field.mean())


def test_long_diffusion_approaches_limit():
    field = _field()
    assert np.abs(diffuse_at(field, 0.15, 5000) - diffuse_limit(field)).max() < 1e-6


def test_mass_conserved():
    field, amount = _field(), 0.2
    for k in (1, 10, 100):
        assert abs(diffuse_at(field, amount, k).mean() - field.mean()) < 1e-9


def test_transfer_dc_is_one():
    t = diffusion_transfer((16, 16), 0.3)
    assert abs(t[0, 0] - 1.0) < 1e-12                         # DC preserved -> mass conserved


def test_zero_steps_is_identity():
    field = _field()
    assert np.allclose(diffuse_at(field, 0.15, 0), field, atol=1e-9)

"""G1 -- the shader algebra doing non-graphics work: the heat equation's periodic propagator as a Pipeline.

THE AUDIT'S CORRECTION. The backlog proposed routing periodic `heat` to `filter_k`, matching its stepped loop to
6.66e-16 at 80x. Reading the code first: `heat`'s periodic path ALREADY had a closed form -- `diffuse_spectral` --
and it is BETTER, because it solves the CONTINUOUS PDE (each Fourier mode decays as exp(-alpha |k|^2 t)) rather
than reproducing the discrete Euler stepper. There was nothing to port.

What it lacked was COMPOSITION. `diffuse_spectral` re-exponentiates the transfer on every call.
`laplacian.diffusion_operator` builds a `shader.Pipeline` from it: bit-identical, ~1.9x faster on reuse, and the
transfer composes with any other diagonal operator by multiplication -- which a bare array does not.

THE GATE, third appearance: the operator must be LINEAR *and* CIRCULAR. Applying the periodic transfer to a Neumann
(edge-replicated) problem is measured 4.76e-02 wrong. Boundary condition is the gate, not the domain.
"""

import numpy as np
import pytest

from holographic.rendering.holographic_shader import Pipeline
from holographic.simulation_and_physics.holographic_heat import diffuse_heat
from holographic.simulation_and_physics.holographic_laplacian import (
    diffuse_spectral, diffusion_operator, diffusion_transfer, laplacian)


ALPHA, DX = 0.2, 1.0


def _field(shape=(64, 64), seed=0):
    """A REAL diffusible field: smooth, periodic-friendly, with structure at several scales. Not an outer product."""
    rng = np.random.default_rng(seed)
    g = [np.linspace(0, 2 * np.pi, n, endpoint=False) for n in shape]
    XX, YY = np.meshgrid(*g, indexing="ij")
    return (np.sin(XX) * np.cos(2 * YY) + 0.4 * np.sin(3 * XX + 1.1) + 0.2 * rng.normal(size=shape))


# ---------------------------------------------------------------------------------------------------------
# Pipeline gains a general transfer injection
# ---------------------------------------------------------------------------------------------------------

def test_pipeline_stage_injects_an_arbitrary_transfer():
    shape = (32, 32)
    f = _field(shape)
    T = diffusion_transfer(shape, ALPHA, 0.35, dx=DX)
    p = Pipeline(shape).stage(T)
    assert np.allclose(p.apply(f), diffuse_spectral(f, ALPHA, 0.35, dx=DX))


def test_pipeline_stage_refuses_a_mismatched_transfer():
    with pytest.raises(ValueError):
        Pipeline((32, 32)).stage(np.ones((16, 16)))


def test_pipeline_stage_is_identity_for_a_transfer_of_ones():
    shape = (16, 16)
    f = _field(shape)
    assert np.allclose(Pipeline(shape).stage(np.ones(shape)).apply(f), f)


# ---------------------------------------------------------------------------------------------------------
# the diffusion operator: bit-identical, reusable, composable
# ---------------------------------------------------------------------------------------------------------

def test_the_operator_is_bit_identical_to_diffuse_spectral():
    for shape in ((32, 32), (64, 64)):
        f = _field(shape)
        op = diffusion_operator(shape, ALPHA, 0.35, dx=DX)
        assert np.array_equal(op.apply(f), diffuse_spectral(f, ALPHA, 0.35, dx=DX))   # bit-for-bit


def test_the_transfer_is_the_exact_mode_decay():
    shape = (32, 32)
    T = diffusion_transfer(shape, ALPHA, 0.5, dx=DX)
    assert T.shape == shape
    assert T.max() == pytest.approx(1.0)          # the DC mode never decays: heat is conserved
    assert (T > 0).all() and (T <= 1.0).all()     # every mode decays, none grows -- no stability limit
    # a longer time decays strictly more
    assert (diffusion_transfer(shape, ALPHA, 1.0, dx=DX) <= T + 1e-15).all()


def test_the_operator_composes_semigroup_style_for_free():
    # THE POINT of holding a transfer rather than an array: two half-steps MULTIPLY into one full step.
    shape = (32, 32)
    f = _field(shape)
    half = diffusion_operator(shape, ALPHA, 0.25, dx=DX)
    full = diffusion_operator(shape, ALPHA, 0.5, dx=DX)
    assert np.abs(half.apply(half.apply(f)) - full.apply(f)).max() < 1e-12

    # ... and composing the transfers directly is the same operator
    composed = Pipeline(shape).stage(diffusion_transfer(shape, ALPHA, 0.25, dx=DX)) \
                              .stage(diffusion_transfer(shape, ALPHA, 0.25, dx=DX))
    assert np.abs(composed.apply(f) - full.apply(f)).max() < 1e-12


def test_diffusion_conserves_total_heat_on_a_periodic_domain():
    # A physical invariant, not a numerical one: the DC mode has |k|=0, so its transfer is exactly 1.
    f = _field((48, 48))
    op = diffusion_operator(f.shape, ALPHA, 2.0, dx=DX)
    assert op.apply(f).sum() == pytest.approx(f.sum(), rel=1e-12)


def test_a_long_time_diffuses_to_the_mean():
    f = _field((32, 32))
    out = diffusion_operator(f.shape, ALPHA, 5000.0, dx=DX).apply(f)
    assert out.std() < 1e-9                       # heat death
    assert out.mean() == pytest.approx(f.mean(), rel=1e-12)


# ---------------------------------------------------------------------------------------------------------
# heat delegates, and the boundary condition is the gate
# ---------------------------------------------------------------------------------------------------------

def test_heat_accepts_the_prebuilt_operator_and_agrees_with_its_own_closed_form():
    f = _field((32, 32))
    dt = (0.9 / 4.0) * DX * DX / ALPHA
    steps = 50
    op = diffusion_operator(f.shape, ALPHA, dt * steps, dx=DX)
    a = diffuse_heat(f, ALPHA, dx=DX, dt=dt, steps=steps, bc="periodic")
    b = diffuse_heat(f, ALPHA, dx=DX, dt=dt, steps=steps, bc="periodic", operator=op)
    assert np.array_equal(a, b)


def test_the_operator_is_ignored_under_neumann_which_keeps_stepping():
    # Passing an operator to the Neumann path must not silently apply a wrong closed form.
    f = _field((32, 32))
    op = diffusion_operator(f.shape, ALPHA, 1.0, dx=DX)
    a = diffuse_heat(f, ALPHA, steps=5)
    b = diffuse_heat(f, ALPHA, steps=5, operator=op)
    assert np.array_equal(a, b)                   # bc="neumann" is the default, and the operator is ignored


def test_kept_negative_the_periodic_transfer_is_wrong_on_a_neumann_problem():
    # The gate, third appearance (after `iterate`'s scope line and `postfx`'s clamped stages). A Neumann stencil is
    # NOT shift-equivariant, so no Fourier transfer represents it.
    f = _field((64, 64))
    a = laplacian(np.roll(f, 3, axis=0), bc="neumann")
    b = np.roll(laplacian(f, bc="neumann"), 3, axis=0)
    assert np.abs(a - b).max() > 1e-3             # not shift-equivariant, by a wide margin

    assert np.abs(laplacian(np.roll(f, 3, axis=0), bc="periodic")
                  - np.roll(laplacian(f, bc="periodic"), 3, axis=0)).max() == 0.0   # periodic IS

    dt = (0.9 / 4.0) * DX * DX / ALPHA
    steps = 200
    stepped = diffuse_heat(f, ALPHA, dx=DX, dt=dt, steps=steps)                     # Neumann, stepped
    wrong = diffusion_operator(f.shape, ALPHA, dt * steps, dx=DX).apply(f)          # periodic transfer, misapplied
    assert np.abs(wrong - stepped).max() > 1e-2                                     # measured 4.76e-02


def test_the_closed_form_beats_stepping_the_periodic_problem():
    # The exact solution is not merely faster: the stepped scheme carries Euler truncation error.
    f = _field((64, 64))
    dt = (0.9 / 4.0) * DX * DX / ALPHA
    steps = 200
    stepped = f.copy()
    r = ALPHA * dt / (DX * DX)
    for _ in range(steps):
        stepped = stepped + r * laplacian(stepped, bc="periodic")
    exact = diffuse_spectral(f, ALPHA, dt * steps, dx=DX)
    err = float(np.abs(stepped - exact).max())
    assert 1e-6 < err < 1e-2                       # the STEPPER is the approximate one; measured 9.97e-05


# ---------------------------------------------------------------------------------------------------------
# the registry records the retarget
# ---------------------------------------------------------------------------------------------------------

def test_the_shader_unifier_now_has_a_non_graphics_client():
    import os
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo, "tools"))
    from unifiers import PENDING, REGISTRY, cites

    key = "shader algebra (bake once, compose passes)"
    assert "holographic_laplacian" in REGISTRY[key]["clients"]
    assert "holographic_heat" not in REGISTRY[key]["clients"]     # heat delegates to laplacian; that is the layering
    assert cites("holographic_laplacian", key, repo)
    assert (key, "holographic_laplacian") not in PENDING

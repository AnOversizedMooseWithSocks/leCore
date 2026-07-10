"""G8 -- Pipeline learns the half-spectrum, and the postfx silo closes.

THE SHAPE OF THIS FIX. `postfx` composed its linear passes into one transfer -- the shader algebra's whole idea --
but would not call `Pipeline`, because `Pipeline.transfer` lived on the FULL `fftn` grid and a real image wants the
half-spectrum (`rfft`). Delegating cost 2.2x for identical output, so the duplication was filed in `DEFERRED` with
its measurement rather than papered over.

`Pipeline(shape, real=True)` builds the transfer on `rfftn`'s grid. Delegating is then **bit-identical**
(`max|diff| = 0.0e+00`) at the same speed. The silo closed by **generalizing the primitive**, not by paying for the
wrong spectrum -- which is what a DEFERRED entry is *for*: a debt with a stated interest rate, waiting for the lever
that clears it.
"""

import numpy as np
import pytest

from holographic.rendering.holographic_postfx import (
    PostChain, _fft_blur, _gaussian_transfer, apply_transfer, cinematic_chain, default_chain)
from holographic.rendering.holographic_shader import Pipeline, blur_kernel


def _real_field(shape, seed=0):
    return np.random.default_rng(seed).normal(size=shape)


# ---------------------------------------------------------------------------------------------------------
# the half-spectrum mode
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(256,), (64, 64), (16, 16, 16)])
def test_the_real_transfer_lives_on_the_half_spectrum(shape):
    p = Pipeline(shape, real=True)
    assert p.tshape == shape[:-1] + (shape[-1] // 2 + 1,)
    assert Pipeline(shape).tshape == shape                     # the default is unchanged


@pytest.mark.parametrize("shape", [(256,), (64, 64), (16, 16, 16)])
def test_real_and_complex_pipelines_agree(shape):
    f = _real_field(shape)
    k = blur_kernel(shape)
    a = Pipeline(shape).blur(k, 3).apply(f)
    b = Pipeline(shape, real=True).blur(k, 3).apply(f)
    assert np.abs(a - b).max() < 1e-12                         # measured 5e-16 .. 9e-16


def test_the_default_mode_is_bit_identical_to_before():
    # BACKWARD COMPATIBILITY. `real` defaults to False, and nothing about the complex path changed.
    f = _real_field((64, 64))
    k = blur_kernel((64, 64))
    p = Pipeline((64, 64)).blur(k, 2)
    assert np.array_equal(p.apply(f), np.real(np.fft.ifftn(np.fft.fftn(f, axes=(0, 1)) * p.transfer, axes=(0, 1))))


def test_translate_is_exact_on_the_half_spectrum():
    # The phase ramp must be built on rfftfreq for the last axis, or a fractional shift is silently wrong.
    x = _real_field((64,), seed=1)
    two_halves = Pipeline((64,), real=True).translate(0.5).translate(0.5)
    one_full = Pipeline((64,), real=True).translate(1.0)
    assert np.abs(two_halves.apply(x) - one_full.apply(x)).max() < 1e-12
    assert np.abs(one_full.apply(x) - np.roll(x, 1)).max() < 1e-9


def test_unsharp_and_gain_work_on_the_half_spectrum():
    x = _real_field((64,), seed=2)
    k = blur_kernel((64,))
    a = Pipeline((64,)).unsharp(k, 0.5).gain(2.0).apply(x)
    b = Pipeline((64,), real=True).unsharp(k, 0.5).gain(2.0).apply(x)
    assert np.abs(a - b).max() < 1e-12


def test_stage_checks_the_transfer_against_the_right_grid():
    shape = (32, 32)
    good = np.ones(Pipeline(shape, real=True).tshape)
    Pipeline(shape, real=True).stage(good)                     # ok
    with pytest.raises(ValueError):
        Pipeline(shape, real=True).stage(np.ones(shape))       # a full-grid transfer is refused
    with pytest.raises(ValueError):
        Pipeline(shape).stage(good)                            # ... and vice versa


def test_from_transfer_is_the_zero_overhead_entry_point():
    shape = (32, 32)
    H = _gaussian_transfer(shape, 2.0)
    p = Pipeline.from_transfer(shape, H, real=True)
    assert np.array_equal(p.transfer, H.astype(complex))       # no identity, no composing multiply
    f = _real_field(shape)
    assert np.array_equal(p.apply(f), Pipeline(shape, real=True).stage(H).apply(f))

    with pytest.raises(ValueError):
        Pipeline.from_transfer(shape, np.ones((8, 8)), real=True)


def test_the_real_pipeline_still_composes_a_whole_graph_into_one_transfer():
    shape = (64, 64)
    f = _real_field(shape)
    k = blur_kernel(shape)
    graph = Pipeline(shape, real=True).blur(k, 4).translate((0.5, -0.25)).unsharp(k, 0.3)
    # ... which must equal doing it the long way through the complex pipeline
    ref = Pipeline(shape).blur(k, 4).translate((0.5, -0.25)).unsharp(k, 0.3)
    assert np.abs(graph.apply(f) - ref.apply(f)).max() < 1e-12


# ---------------------------------------------------------------------------------------------------------
# postfx now delegates, and nothing about it changed
# ---------------------------------------------------------------------------------------------------------

def test_apply_transfer_delegates_and_is_bit_identical_to_the_engine_operator():
    img = np.random.default_rng(0).uniform(0.2, 0.6, size=(64, 64, 3))
    G = _gaussian_transfer((64, 64), 2.0)
    assert np.array_equal(apply_transfer(img, G), _fft_blur(img, 2.0))   # bit-for-bit, through the Pipeline


def test_apply_transfer_matches_a_hand_rolled_rfft2():
    img = np.random.default_rng(1).uniform(0, 1, size=(64, 64, 3))
    G = _gaussian_transfer((64, 64), 1.3)
    manual = np.empty_like(img)
    for c in range(3):
        manual[:, :, c] = np.fft.irfft2(np.fft.rfft2(img[:, :, c]) * G, s=(64, 64))
    assert np.array_equal(apply_transfer(img, G), manual)


def test_the_fused_chain_is_unchanged_by_the_delegation():
    img = np.random.default_rng(0).uniform(0.2, 0.6, size=(64, 64, 3))
    chain = PostChain().then("denoise", sigma=1.0).then("sharpen", amount=0.3, sigma=1.5)
    assert np.abs(chain.apply(img) - chain.apply(img, fuse=True)).max() < 1e-12


def test_the_shipped_chains_are_still_a_bit_identical_no_op_under_fusion():
    # `default_chain` can be applied blind; `cinematic_chain` needs a depth buffer, so it is checked structurally.
    img = np.random.default_rng(0).uniform(0.2, 0.6, size=(48, 48, 3))
    d = default_chain()
    assert np.array_equal(d.apply(img), d.apply(img, fuse=True))

    from holographic.rendering.holographic_postfx import fusable_runs
    for chain in (default_chain(), cinematic_chain()):
        assert not any(is_fused for is_fused, _grp in fusable_runs(chain.steps))


def test_postfx_keeps_its_per_channel_loop():
    # KEPT NEGATIVE, still true after the delegation: batching the 3 channels into one FFT is 0.66x SLOWER
    # (non-contiguous strides), bit-identical output. The loop stays, and the Pipeline is built once outside it.
    import inspect

    from holographic.rendering import holographic_postfx as pf

    src = inspect.getsource(pf.apply_transfer)
    assert "for c in range(img.shape[2])" in src
    assert "from_transfer" in src                              # ... and constructed once, not per channel


# ---------------------------------------------------------------------------------------------------------
# the registry: a DEFERRED debt, paid
# ---------------------------------------------------------------------------------------------------------

def test_postfx_is_now_a_wired_client_and_no_longer_deferred():
    import os
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo, "tools"))
    from unifiers import DEFERRED, NOT_APPLICABLE, PENDING, REGISTRY, cites

    key = "shader algebra (bake once, compose passes)"
    assert "holographic_postfx" in REGISTRY[key]["clients"]
    assert cites("holographic_postfx", key, repo)
    assert (key, "holographic_postfx") not in DEFERRED
    assert (key, "holographic_postfx") not in NOT_APPLICABLE
    assert not any(u == key for u, _c in PENDING)

    # the OTHER postfx deferral -- LowRankField -- is untouched: streaming a frame still cannot amortize an SVD
    assert ("tucker.LowRankField (compressed-domain compute)", "holographic_postfx") in DEFERRED


def test_the_registry_records_how_the_debt_was_paid():
    import os
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo, "tools"))
    from unifiers import REGISTRY

    why = REGISTRY["shader algebra (bake once, compose passes)"]["why"]
    assert "real=True" in why and "0.0e+00" in why             # the mechanism and the evidence

"""B1 -- postfx on the shader algebra: compose a run of linear passes into ONE transfer.

`_fft_blur` already runs the engine's core operator (a 2-D circular convolution IS `bind`, one dimension up). What
it did not do is COMPOSE: a run of linear stages paid one FFT pair each, when the composed operator is just the
elementwise product of their transfers. Diagonal operators commute and multiply.

MEASURED (256x256x3, three linear stages): 14.76 ms sequential vs 5.03 ms fused -- 2.9x, max|diff| 4.44e-16.

THREE KEPT NEGATIVES, and the first decides how this ships:
  1. the SHIPPED chains have no adjacent linear stages, so `fuse=True` is correctly a NO-OP on them;
  2. `sharpen` clips internally, so fusing DEFERS its clamp -- a semantic change, hence opt-in;
  3. batching the 3 channels into one FFT is measured 0.66x SLOWER (non-contiguous strides), not faster.
"""

import numpy as np
import pytest

from holographic.rendering.holographic_postfx import (
    EFFECTS, PostChain, _fft_blur, _gaussian_transfer, apply_transfer, cinematic_chain, default_chain,
    denoise, fusable_runs, fuse_transfers, linear_transfer, sharpen)


def _img(h=64, w=64, lo=0.2, hi=0.6, seed=0):
    return np.random.default_rng(seed).uniform(lo, hi, size=(h, w, 3))


def test_selftest_runs():
    from holographic.rendering import holographic_postfx as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# the transfers
# ---------------------------------------------------------------------------------------------------------

def test_the_gaussian_transfer_is_the_one_fft_blur_multiplies_by():
    img = _img()
    T = _gaussian_transfer(img.shape[:2], 2.0)
    assert np.array_equal(apply_transfer(img, T), _fft_blur(img, 2.0))   # bit-identical, not merely close


def test_denoise_transfer_is_exact():
    img = _img()
    T = linear_transfer(img.shape[:2], "denoise", {"sigma": 1.3})
    assert np.abs(apply_transfer(img, T) - denoise(img, sigma=1.3)).max() < 1e-12


def test_sharpen_transfer_matches_its_unclipped_math():
    img = _img()
    a, s = 0.5, 1.5
    T = linear_transfer(img.shape[:2], "sharpen", {"amount": a, "sigma": s})
    unclipped = (1.0 + a) * img - a * _fft_blur(img, s)
    assert np.abs(apply_transfer(img, T) - unclipped).max() < 1e-12


def test_nonlinear_and_edge_clamping_effects_are_refused_not_approximated():
    shape = (64, 64)
    for name in ("gamma", "aces", "bloom", "glare", "motion_blur", "vignette", "dof"):
        assert linear_transfer(shape, name, {}) is None, name
    # ... and a run containing one of them cannot be fused
    assert fuse_transfers(shape, [("denoise", {}), ("glare", {})]) is None


def test_transfers_compose_by_multiplication_and_commute():
    shape = (64, 64)
    a = [("denoise", {"sigma": 1.0}), ("sharpen", {"amount": 0.3, "sigma": 2.0})]
    T1 = fuse_transfers(shape, a)
    T2 = fuse_transfers(shape, list(reversed(a)))
    assert np.abs(T1 - T2).max() < 1e-15          # diagonal operators commute


# ---------------------------------------------------------------------------------------------------------
# the fusion, and its exactness
# ---------------------------------------------------------------------------------------------------------

def test_a_pure_linear_run_fuses_exactly():
    img = _img(128, 128)
    chain = (PostChain().then("denoise", sigma=1.0)
                        .then("denoise", sigma=0.7)
                        .then("denoise", sigma=0.5))
    assert np.abs(chain.apply(img) - chain.apply(img, fuse=True)).max() < 1e-12


def test_a_run_with_sharpen_fuses_when_the_intermediate_clamp_never_fires():
    # values comfortably inside [0,1] and a gentle amount: the deferred clamp changes nothing
    img = _img(128, 128, lo=0.35, hi=0.55)
    chain = PostChain().then("denoise", sigma=1.0).then("sharpen", amount=0.3, sigma=1.5)
    assert np.abs(chain.apply(img) - chain.apply(img, fuse=True)).max() < 1e-9


def test_kept_negative_a_deferred_clamp_only_matters_when_a_stage_follows_it():
    # The finding is sharper than "fusing sharpen changes the result". A deferred clamp only matters when the
    # clamped stage is FOLLOWED by another stage inside the run -- with `sharpen` last, the chain's own final clip
    # does the same job. (I asserted the coarse version first and it failed: denoise->sharpen is EXACT.)
    img = _img(128, 128, lo=0.0, hi=1.0, seed=3)

    last = PostChain().then("denoise", sigma=1.0).then("sharpen", amount=1.5, sigma=2.0)
    assert np.abs(last.apply(img) - last.apply(img, fuse=True)).max() < 1e-9      # exact: 1.33e-15

    first = PostChain().then("sharpen", amount=1.5, sigma=2.0).then("denoise", sigma=1.0)
    assert np.abs(first.apply(img) - first.apply(img, fuse=True)).max() > 1e-3    # differs: 2.81e-01

    # ... and this is WHY: the unsharp mask drives values below zero before the deferred clamp would have caught them
    T = fuse_transfers(img.shape[:2], [("sharpen", {"amount": 1.5, "sigma": 2.0})])
    assert (apply_transfer(img, T) < 0.0).any()


def test_kept_negative_the_shipped_chains_have_no_adjacent_linear_stages():
    # The finding that decides how this ships. Fusing "the chain" would deliver exactly zero on the real chains.
    for chain in (default_chain(), cinematic_chain()):
        runs = fusable_runs(chain.steps)
        assert not any(is_fused for is_fused, _ in runs), [n for n, _ in chain.steps]

    # ... so fuse=True is a bit-identical no-op on them
    img = _img(64, 64)
    d = default_chain()
    assert np.array_equal(d.apply(img), d.apply(img, fuse=True))


def test_fuse_is_default_off_and_a_single_linear_stage_is_not_fused():
    # a run of ONE would pay exactly the FFT pair it already pays; fusing it buys nothing and is reported as such
    runs = fusable_runs([("denoise", {}), ("gamma", {}), ("sharpen", {})])
    assert [f for f, _ in runs] == [False, False, False]
    runs2 = fusable_runs([("denoise", {}), ("sharpen", {}), ("gamma", {})])
    assert [f for f, _ in runs2] == [True, False]
    assert [n for n, _ in runs2[0][1]] == ["denoise", "sharpen"]


def test_fusable_runs_preserves_every_step_exactly_once():
    steps = [("exposure", {}), ("denoise", {}), ("sharpen", {}), ("gamma", {}), ("denoise", {})]
    flat = [s for _f, grp in fusable_runs(steps) for s in grp]
    assert flat == steps


def test_a_chain_with_depth_still_runs_under_fusion():
    img = _img(64, 64)
    depth = np.full((64, 64), 2.0)
    chain = PostChain().then("denoise", sigma=1.0).then("sharpen", amount=0.2).then("dof", aperture=1.0)
    out = chain.apply(img, depth=depth, fuse=True)
    assert out.shape == img.shape and np.isfinite(out).all()


def test_kept_negative_batching_the_channels_into_one_fft_is_slower():
    # Filed so nobody "optimizes" the per-channel loop away. Bit-identical, and 0.66x at 256^2.
    img = _img(128, 128)
    H, W = img.shape[:2]
    G = _gaussian_transfer((H, W), 2.0)
    batched = np.fft.irfft2(np.fft.rfft2(img, axes=(0, 1)) * G[:, :, None], s=(H, W), axes=(0, 1))
    assert np.abs(batched - apply_transfer(img, G)).max() < 1e-12       # same answer ...
    # ... the reason we keep the loop is speed, measured in the module note, not correctness


# ---------------------------------------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------------------------------------

def test_fully_wired_to_the_mind():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    img = _img(64, 64)

    T = m.postfx_fuse_transfers((64, 64), [("denoise", {"sigma": 1.0}), ("sharpen", {"amount": 0.3, "sigma": 1.5})])
    assert T is not None and T.shape == (64, 33)
    out = m.postfx_apply_transfer(img, T)
    assert out.shape == img.shape

    runs = m.postfx_fusable_runs([("denoise", {}), ("sharpen", {}), ("gamma", {})])
    assert runs[0][0] is True and runs[1][0] is False
    assert m.postfx_fuse_transfers((64, 64), [("denoise", {}), ("glare", {})]) is None


def test_the_postfx_deferral_was_PAID_not_forgotten():
    # THE HONEST BOOKKEEPING, and its close. postfx composed its passes into one transfer -- the shader algebra's
    # whole idea -- but would not call `Pipeline`, because `Pipeline.transfer` lived on the full fftn grid while a
    # real image wants the half-spectrum. Delegating was EXACT (7.8e-16) and a measured 2.2x LOSS, so it went to
    # DEFERRED with its evidence rather than being papered over.
    #
    # G8 gave `Pipeline` a `real=True` mode. Delegating is now BIT-IDENTICAL (0.0e+00) at the same speed, and
    # postfx is a wired client. **A DEFERRED entry is a debt with a stated interest rate, not a shrug** -- and this
    # one was paid by generalizing the primitive, not by paying for the wrong spectrum.
    import os
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo, "tools"))
    from unifiers import DEFERRED, NOT_APPLICABLE, PENDING, REGISTRY, cites

    key = ("shader algebra (bake once, compose passes)", "holographic_postfx")
    assert key not in DEFERRED and key not in NOT_APPLICABLE and key not in PENDING
    assert "holographic_postfx" in REGISTRY[key[0]]["clients"]
    assert cites("holographic_postfx", key[0], repo)

    # the OTHER postfx deferral stands: streaming a frame still cannot amortize an SVD (53.7x)
    lr = ("tucker.LowRankField (compressed-domain compute)", "holographic_postfx")
    assert lr in DEFERRED and "53.7x" in DEFERRED[lr]

"""Tests for holographic_splatprune: prune the negligible splats, merge the redundant ones, and pick a level for a
quality budget -- the splat-domain twin of the mesh LOD policy. Contribution-ranked prune+refit dominates naive
pruning; the chain degrades gracefully."""

import numpy as np

from holographic.rendering.holographic_splat import splat_fit, splat_render, splat_refit, psnr
from holographic.rendering.holographic_splatprune import splat_prune, splat_merge, splat_lod_chain, select_splat_lod


def _gauss2d(shape, cy, cx, s):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))


_SHAPE = (64, 64)
_TARGET = (1.0 * _gauss2d(_SHAPE, 18, 20, 5) + 0.8 * _gauss2d(_SHAPE, 40, 44, 7)
           + 0.6 * _gauss2d(_SHAPE, 48, 15, 4) + 0.4 * _gauss2d(_SHAPE, 12, 50, 3))
_FULL = splat_fit(_TARGET, 60, refit=True)
_CHAIN = splat_lod_chain(_FULL, _TARGET, keeps=(40, 20, 10, 5))


def _p(splats):
    return psnr(splat_render(splats, _SHAPE), _TARGET)


def test_contribution_prune_beats_random():
    rng = np.random.default_rng(0)
    rand = splat_refit([_FULL[i] for i in rng.permutation(len(_FULL))[:20]], _TARGET)
    assert _p(splat_prune(_FULL, _TARGET, 20)) > _p(rand) + 8


def test_contribution_prune_beats_keeping_smallest():
    worst = splat_refit([_FULL[i] for i in sorted(range(len(_FULL)), key=lambda i: abs(_FULL[i][2]))[:20]], _TARGET)
    assert _p(splat_prune(_FULL, _TARGET, 20)) > _p(worst) + 8


def test_prune_keep_all_returns_full():
    assert len(splat_prune(_FULL, _TARGET, 999)) == len(_FULL)


def test_prune_reduces_count():
    assert len(splat_prune(_FULL, _TARGET, 12)) == 12


def test_lod_chain_counts_decrease():
    counts = [c[1] for c in _CHAIN]
    assert all(counts[i] > counts[i + 1] for i in range(len(counts) - 1))


def test_lod_chain_psnr_degrades_gracefully():
    psnrs = [c[2] for c in _CHAIN]
    assert all(psnrs[i] >= psnrs[i + 1] - 1e-6 for i in range(len(psnrs) - 1))


def test_merge_reduces_count():
    assert len(splat_merge(_FULL, _TARGET, radius=4.0)) < len(_FULL)


def test_merge_quality_loss_bounded():
    merged = splat_merge(_FULL, _TARGET, radius=4.0)
    assert _p(merged) > _p(_FULL) - 12


def test_select_tighter_budget_keeps_more():
    loose = select_splat_lod(_CHAIN, min_psnr=30.0)
    tight = select_splat_lod(_CHAIN, min_psnr=43.0)
    assert _CHAIN[tight][1] >= _CHAIN[loose][1]


def test_select_meets_budget():
    pick = select_splat_lod(_CHAIN, min_psnr=30.0)
    assert _CHAIN[pick][2] >= 30.0


def test_deterministic():
    a = splat_render(splat_prune(_FULL, _TARGET, 15), _SHAPE)
    b = splat_render(splat_prune(_FULL, _TARGET, 15), _SHAPE)
    assert np.array_equal(a, b)

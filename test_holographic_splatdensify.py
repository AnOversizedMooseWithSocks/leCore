"""Tests for clone-vs-split density control (holographic_splatdensify): scale-aware splat densification -- CLONE a
small high-error splat to COVER an under-served region, SPLIT a wide high-error splat to RESOLVE fine structure -- the
3DGS densification distinction the engine's scale-blind residual placement was missing."""

import numpy as np

from holographic_splat import splat_render, splat_refit
from holographic_splatdensify import clone_splat, split_splat, clone_split_densify


def _mse(a, b):
    return float(((a - b) ** 2).mean())


_YS, _XS = np.mgrid[0:64, 0:64]


def test_clone_wins_for_cover():
    ridge = np.exp(-(((_XS - 24) ** 2) / 6.0 + ((_YS - 32) ** 2) / 120.0))
    s = splat_refit([(32, 24, 0.0, 1.0)], ridge)
    base = _mse(splat_render(s, ridge.shape), ridge)
    res = ridge - splat_render(s, ridge.shape)
    clone = _mse(splat_render(splat_refit(s + clone_splat(s[0], res, ridge.shape), ridge), ridge.shape), ridge)
    split = _mse(splat_render(splat_refit(split_splat(s[0], res, ridge.shape), ridge), ridge.shape), ridge)
    assert clone < base and clone < split


def test_split_worse_than_baseline_on_small_splat():
    # the wrong move can be WORSE than nothing -- splitting a small splat loses coverage
    ridge = np.exp(-(((_XS - 24) ** 2) / 6.0 + ((_YS - 32) ** 2) / 120.0))
    s = splat_refit([(32, 24, 0.0, 1.0)], ridge)
    base = _mse(splat_render(s, ridge.shape), ridge)
    res = ridge - splat_render(s, ridge.shape)
    split = _mse(splat_render(splat_refit(split_splat(s[0], res, ridge.shape), ridge), ridge.shape), ridge)
    assert split > base


def test_split_wins_for_resolve():
    twin = (np.exp(-(((_XS - 30) ** 2 + (_YS - 30) ** 2) / 4.0)) + np.exp(-(((_XS - 38) ** 2 + (_YS - 30) ** 2) / 4.0)))
    s = splat_refit([(30, 34, 0.0, 3.5)], twin)
    res = twin - splat_render(s, twin.shape)
    clone = _mse(splat_render(splat_refit(s + clone_splat(s[0], res, twin.shape), twin), twin.shape), twin)
    split = _mse(splat_render(splat_refit(split_splat(s[0], res, twin.shape), twin), twin.shape), twin)
    assert split < clone


def test_scale_aware_beats_both_blind_strategies():
    ridge = np.exp(-(((_XS - 14) ** 2) / 6.0 + ((_YS - 32) ** 2) / 120.0))
    twin = (np.exp(-(((_XS - 46) ** 2 + (_YS - 30) ** 2) / 4.0)) + np.exp(-(((_XS - 52) ** 2 + (_YS - 30) ** 2) / 4.0)))
    target = ridge + twin
    splats = splat_refit([(32, 14, 0.0, 1.0), (30, 49, 0.0, 3.5)], target)
    shape = target.shape
    res = target - splat_render(splats, shape)

    def blind(strategy):
        out = []
        for sp in splats:
            if strategy == "split":
                out += split_splat(sp, res, shape)
            else:
                out += [sp] + clone_splat(sp, res, shape)
        return _mse(splat_render(splat_refit(out, target), shape), target)

    scale_mse = _mse(splat_render(clone_split_densify(splats, target), shape), target)
    assert scale_mse < blind("clone") and scale_mse < blind("split")


def test_split_removes_original_clone_keeps_it():
    # split replaces a wide splat with two narrower; clone keeps the original and adds one
    target = np.exp(-(((_XS - 30) ** 2 + (_YS - 30) ** 2) / 30.0))
    s = splat_refit([(30, 30, 0.0, 3.5)], target)
    res = target - splat_render(s, target.shape)
    assert len(split_splat(s[0], res, target.shape)) == 2
    assert all(sp[3] < 3.5 for sp in split_splat(s[0], res, target.shape))   # narrower
    assert len(clone_splat(s[0], res, target.shape)) == 1
    assert clone_splat(s[0], res, target.shape)[0][3] == 3.5                 # same scale


def test_deterministic():
    target = np.exp(-(((_XS - 30) ** 2 + (_YS - 30) ** 2) / 30.0))
    splats = splat_refit([(30, 30, 0.0, 2.0), (20, 20, 0.0, 1.0)], target)
    d1 = clone_split_densify(splats, target)
    d2 = clone_split_densify(splats, target)
    assert all(np.allclose(a[:3], b[:3]) and a[3] == b[3] for a, b in zip(d1, d2))

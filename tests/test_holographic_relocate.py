"""Tests for MCMC birth-death relocation (holographic_relocate): the successor to evict-rarest -- relocate a dead /
low-weight splat to an under-represented region instead of DROPPING it, conserving the budget. The principled
destination (the residual peak) is what wins; random relocation does not."""

import numpy as np

from holographic.rendering.holographic_splat import splat_render, splat_refit
from holographic.misc.holographic_relocate import birth_death_relocate

_YS, _XS = np.mgrid[0:64, 0:64]


def _target():
    return sum(np.exp(-(((_XS - cx) ** 2 + (_YS - cy) ** 2) / 12.0))
               for cx, cy in [(16, 16), (48, 16), (16, 48), (48, 48), (32, 32), (32, 10)])


def _with_dead():
    target = _target()
    useful = [(16, 16, 0.0, 3.5), (48, 16, 0.0, 3.5), (16, 48, 0.0, 3.5),
              (48, 48, 0.0, 3.5), (32, 32, 0.0, 3.5), (32, 10, 0.0, 3.5)]
    dead = [(2, 2, 0.0, 1.0)] * 6
    return splat_refit(useful + dead, target), target


def _mse(a, b):
    return float(((a - b) ** 2).mean())


def test_relocate_beats_drop():
    splats, target = _with_dead()
    thr = 0.05 * np.abs([s[2] for s in splats]).max()
    drop = _mse(splat_render(splat_refit([s for s in splats if abs(s[2]) >= thr], target), target.shape), target)
    reloc = _mse(splat_render(birth_death_relocate(splats, target), target.shape), target)
    assert reloc < drop * 0.6


def test_residual_target_beats_random():
    splats, target = _with_dead()
    shape = target.shape
    a = np.abs([s[2] for s in splats])
    t = 0.05 * a.max()
    rng = np.random.default_rng(0)
    sp = [list(s) for s in splats]
    residual = target - splat_render(sp, shape)
    from holographic.rendering.holographic_splat import _gaussian
    for i in [k for k in range(len(sp)) if a[k] < t]:
        py, px = int(rng.integers(shape[0])), int(rng.integers(shape[1]))
        g = _gaussian(shape, py, px, sp[i][3])
        amp = float((residual * g).sum())
        sp[i] = [int(py), int(px), amp, sp[i][3]]
        residual = residual - amp * g
    rand = _mse(splat_render(splat_refit([tuple(s) for s in sp], target), shape), target)
    reloc = _mse(splat_render(birth_death_relocate(splats, target), shape), target)
    assert reloc < rand * 0.6


def test_count_conserved():
    splats, target = _with_dead()
    assert len(birth_death_relocate(splats, target)) == len(splats)


def test_no_dead_is_noop():
    target = (np.exp(-(((_XS - 16) ** 2 + (_YS - 16) ** 2) / 12.0)) +
              np.exp(-(((_XS - 48) ** 2 + (_YS - 48) ** 2) / 12.0)))
    splats = splat_refit([(16, 16, 0.0, 3.5), (48, 48, 0.0, 3.5)], target)
    out = birth_death_relocate(splats, target)
    assert len(out) == len(splats)


def test_relocate_improves_reconstruction():
    splats, target = _with_dead()
    keep = _mse(splat_render(splats, target.shape), target)
    reloc = _mse(splat_render(birth_death_relocate(splats, target), target.shape), target)
    assert reloc < keep


def test_deterministic():
    splats, target = _with_dead()
    a = birth_death_relocate(splats, target)
    b = birth_death_relocate(splats, target)
    assert all(np.allclose(x[:3], y[:3]) and x[3] == y[3] for x, y in zip(a, b))

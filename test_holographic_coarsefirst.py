"""Tests for holographic_coarsefirst -- the coarse-first residual pass (the Group-B unlocker)."""
import numpy as np
from holographic_coarsefirst import (escalate_mask, refine_where_uncertain, gradient_uncertainty, concentration)


def test_escalate_mask_by_frac_is_conservative():
    u = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    m = escalate_mask(u, frac=0.4)              # top 40% = top 2 cells
    assert m.tolist() == [False, False, False, True, True]


def test_escalate_mask_by_threshold():
    u = np.array([0.1, 0.5, 0.9])
    assert escalate_mask(u, threshold=0.5).tolist() == [False, True, True]   # >= threshold (conservative)


def test_refine_only_touches_flagged_cells():
    coarse = np.zeros((4, 4))
    unc = np.zeros((4, 4)); unc[0, 0] = 10.0
    refined, mask, n = refine_where_uncertain(coarse, unc, lambda m: np.full(int(m.sum()), 9.0), threshold=5.0)
    assert n == 1 and refined[0, 0] == 9.0 and refined.sum() == 9.0        # only the flagged cell changed


def test_field_approximation_wins_when_concentrated():
    # smooth field + a thin ridge (concentrated structure): refine 20% -> big error drop at a fraction of the cost
    H = W = 64
    ys, xs = np.mgrid[0:H, 0:W] / float(H)
    def f(Y, X): return 0.3 * np.sin(3 * Y) + 0.3 * np.cos(3 * X) + np.exp(-((X - 0.51) ** 2) / 0.0008)
    truth = f(ys, xs)
    cs = 4
    cg = f(ys[::cs, ::cs], xs[::cs, ::cs])
    coarse = np.repeat(np.repeat(cg, cs, axis=0), cs, axis=1)[:H, :W]      # blocky upsample
    unc = gradient_uncertainty(coarse)
    assert concentration(unc) > 0.3
    refined, mask, n = refine_where_uncertain(coarse, unc, lambda m: f(ys, xs), frac=0.2)
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    assert rmse(refined, truth) < 0.5 * rmse(coarse, truth)               # refinement recovered most of the error


def test_concentration_low_for_uniform_uncertainty():
    rng = np.random.default_rng(0)
    uniform = np.abs(rng.standard_normal((32, 32))) + 5.0                 # roughly uniform -> low concentration
    spike = np.ones((32, 32)) * 0.01; spike[0, 0] = 100.0                # one hot cell -> high concentration
    assert concentration(uniform) < 0.2
    assert concentration(spike) > 0.8


def test_gradient_uncertainty_flags_edges():
    field = np.zeros((16, 16)); field[:, 8:] = 1.0                       # a vertical edge at column 8
    u = gradient_uncertainty(field)
    assert u[:, 7:9].sum() > u[:, :6].sum()                              # gradient concentrated at the edge

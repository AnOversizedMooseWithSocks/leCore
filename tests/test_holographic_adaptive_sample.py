"""Forecasting sweep: calibrated adaptive-sampling stop (variance-CI budget, the renderer delegation)."""
import numpy as np
from holographic.sampling_and_signal.holographic_adaptive_sample import converged_mask, samples_to_target, sample_budget, ci_half_width


def _varmap():
    H = W = 16
    v = np.zeros((H, W)); v[:, :W // 2] = 1e-5; v[:, W // 2:] = 4e-3
    return v


def test_converged_mask_splits_low_and_high_variance():
    v = _varmap(); W = 16
    m = converged_mask(v, tolerance=0.05)
    assert m[:, :W // 2].all() and not m[:, W // 2:].any()


def test_budget_zero_for_converged_positive_for_noisy():
    v = _varmap(); W = 16
    b = sample_budget(v, current_n=64, target_half_width=0.05)
    assert b[:, :W // 2].sum() == 0 and b[:, W // 2:].min() > 0


def test_mc_law_quadruples_on_halving():
    nw = samples_to_target(np.array([4e-3]), 64, 0.10)[0]
    nh = samples_to_target(np.array([4e-3]), 64, 0.05)[0]
    assert 3.5 <= nh / max(nw, 1) <= 4.5


def test_ci_half_width_and_determinism():
    assert abs(float(ci_half_width(np.array([1.0]))[0]) - 1.959963984540054) < 1e-9
    v = _varmap()
    assert np.array_equal(converged_mask(v, 0.05), converged_mask(v, 0.05))

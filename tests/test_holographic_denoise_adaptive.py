"""Tests for the adaptive-rank (noise-thresholded) manifold denoiser -- cashing the fixed-rank
denoiser's low-noise over-smoothing negative on real SOL windows."""

import numpy as np

from holographic.rendering.holographic_denoise import fit_manifold, manifold_denoise, fit_manifold_full, adaptive_manifold_denoise, estimate_sigma


def _windows():
    px = np.load("data/sol_5min.npz")["px"].astype(float)
    W, step = 64, 16
    wins = np.stack([px[i:i + W] for i in range(0, len(px) - W, step)])
    wins = (wins - wins.mean(1, keepdims=True)) / (wins.std(1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(0); rng.shuffle(wins)
    return wins[:600], wins[600:900]


def _gain(clean, noisy, est):
    s = lambda c, e: 10 * np.log10(np.var(c) / (np.mean((c - e) ** 2) + 1e-12))
    return s(clean, est) - s(clean, noisy)


def test_adaptive_does_not_harm_at_low_noise_where_fixed_rank_does():
    tr, te = _windows()
    b8, m8 = fit_manifold(tr, rank=8)
    Vf, _, mf = fit_manifold_full(tr, rank=32)
    rng = np.random.default_rng(1)
    fixed, adap = [], []
    for c in te:
        n = c + 0.3 * rng.standard_normal(len(c))
        fixed.append(_gain(c, n, manifold_denoise(n, b8, m8)))
        adap.append(_gain(c, n, adaptive_manifold_denoise(n, Vf, mf)))
    assert np.mean(fixed) < -0.3                      # fixed rank-8 over-smooths -> harms at low noise
    assert np.mean(adap) > np.mean(fixed) + 0.3       # adaptive does materially less harm (cashes it)
    assert np.mean(adap) > -0.25                       # essentially neutral


def test_adaptive_still_denoises_at_high_noise():
    tr, te = _windows()
    Vf, _, mf = fit_manifold_full(tr, rank=32)
    rng = np.random.default_rng(2)
    adap = [_gain(c, c + 0.8 * rng.standard_normal(len(c)),
                  adaptive_manifold_denoise(c + 0.8 * rng.standard_normal(len(c)), Vf, mf)) for c in te]
    # recompute cleanly (same noise draw) to avoid double-draw bias
    adap = []
    for c in te:
        n = c + 0.8 * rng.standard_normal(len(c))
        adap.append(_gain(c, n, adaptive_manifold_denoise(n, Vf, mf)))
    assert np.mean(adap) > 2.0                          # solid denoising at high noise


def test_estimate_sigma_tracks_the_true_noise():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.standard_normal(200)) * 0.05      # a smooth-ish signal
    for sig in (0.1, 0.3, 0.6):
        est = estimate_sigma(x + sig * rng.standard_normal(len(x)))
        assert 0.5 * sig < est < 1.6 * sig              # within a reasonable factor


def test_adaptive_is_near_identity_when_noise_is_tiny():
    tr, _ = _windows()
    Vf, _, mf = fit_manifold_full(tr, rank=32)
    x = tr[0]
    out = adaptive_manifold_denoise(x, Vf, mf, sigma=1e-6)   # ~no noise -> keep all -> identity-on-subspace
    proj = mf + (x - mf) @ Vf.T @ Vf                         # full projection onto the (generous) basis
    assert np.max(np.abs(out - proj)) < 1e-6


def test_fit_manifold_full_returns_singular_values():
    tr, _ = _windows()
    V, S, m = fit_manifold_full(tr, rank=16)
    assert V.shape[0] == 16 and S.shape[0] == 16 and m.shape[0] == V.shape[1]
    assert np.all(np.diff(S) <= 1e-9)                        # singular values non-increasing

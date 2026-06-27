"""Tests for holographic_graphsignal: Taubin graph-Laplacian denoising of a codebook over its k-NN graph
(reverse-transfer RT-III1), with the honest high-noise win and the low-noise kept negative."""

import numpy as np

from holographic_graphsignal import knn_graph, laplacian_filter, taubin_filter, graph_denoise, _selftest


def _manifold(seed=0, D=512, N=120, rel=1.2):
    """A curved high-rank 1-D manifold plus additive noise of relative norm `rel`. Returns (clean, noisy)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, N)
    omega = rng.uniform(1, 10, D)
    phi = rng.uniform(0, 2 * np.pi, D)
    clean = np.cos(2 * np.pi * np.outer(t, omega) + phi)
    clean /= np.linalg.norm(clean, axis=1, keepdims=True)
    nz = rng.standard_normal((N, D))
    nz /= np.linalg.norm(nz, axis=1, keepdims=True)
    return clean, clean + rel * nz


def _quality(X, clean):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    return float(np.mean(np.sum(Xn * clean, axis=1)))


def _consolidate(X, r):
    m = X.mean(0)
    Vt = np.linalg.svd(X - m, full_matrices=False)[2]
    return m + (X - m) @ Vt[:r].T @ Vt[:r]


def _per_vector(X, clean):
    return max(_quality(_consolidate(X, r), clean) for r in (8, 16, 24, 32))


def test_module_selftest():
    _selftest()


def test_taubin_denoises_and_avoids_the_shrink():
    clean, noisy = _manifold(rel=1.2)
    idx, w = knn_graph(noisy, k=8)
    taub = taubin_filter(noisy, idx, w)
    naive = laplacian_filter(noisy, idx, w)
    assert _quality(taub, clean) > _quality(noisy, clean)          # it denoises
    taub_norm = float(np.linalg.norm(taub, axis=1).mean())
    naive_norm = float(np.linalg.norm(naive, axis=1).mean())
    assert taub_norm > 0.8                                          # Taubin keeps its norm (no shrink)
    assert naive_norm < 0.65                                        # naive Laplacian collapses
    assert taub_norm > naive_norm + 0.2                             # the no-shrink gap is unambiguous


def test_graph_beats_per_vector_at_high_noise_across_seeds():
    wins = 0
    for s in range(6):
        clean, noisy = _manifold(seed=s, rel=1.2)
        taub = graph_denoise(noisy, k=8, method="taubin")
        if _quality(taub, clean) > _per_vector(noisy, clean):
            wins += 1
    assert wins >= 5                                                # robust high-noise win (measured 6/6)


def test_low_noise_kept_negative_per_vector_wins():
    # The honest regime boundary: when the signal is already clean, the global linear denoiser is better and
    # the graph filter OVER-SMOOTHS. Pinned so the win is not overclaimed.
    clean, noisy = _manifold(seed=0, rel=0.5)
    taub = graph_denoise(noisy, k=8, method="taubin")
    assert _per_vector(noisy, clean) > _quality(taub, clean)


def test_knn_graph_is_deterministic_and_row_normalized():
    _, noisy = _manifold()
    idx1, w1 = knn_graph(noisy, k=8)
    idx2, w2 = knn_graph(noisy, k=8)
    assert np.array_equal(idx1, idx2) and np.allclose(w1, w2)       # deterministic
    assert np.allclose(w1.sum(axis=1), 1.0)                         # rows are a probability (random-walk) graph

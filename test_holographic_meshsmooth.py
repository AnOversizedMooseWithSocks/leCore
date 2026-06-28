"""Tests for mesh smoothing (FWD-4): the Taubin no-shrink denoise wired onto the shipped graphsignal filter,
the cotangent-vs-uniform adjacency, the no-shrink property versus the Laplacian baseline, connectivity/chi
preservation (smoothing moves vertices only), and determinism. The cotangent-isn't-uniformly-better finding is
a docstring/selftest negative, mirrored here only as 'both weightings denoise' -- no false superiority claim."""

import numpy as np

from holographic_mesh import Mesh, box
from holographic_meshsmooth import (taubin_smooth, laplacian_smooth,
                                    cotangent_adjacency, uniform_adjacency, _icosphere)


def _noisy_sphere(subdiv=3, sigma=0.05, seed=0):
    clean = _icosphere(subdiv)
    rng = np.random.default_rng(seed)
    noisy = Mesh(clean.vertices + rng.normal(0.0, sigma, clean.vertices.shape), list(clean.faces))
    return clean, noisy


def _radial_err(m):
    return float(np.abs(np.linalg.norm(m.vertices, axis=1) - 1.0).mean())


def _mean_radius(m):
    return float(np.linalg.norm(m.vertices, axis=1).mean())


# ---- the core measured bar --------------------------------------------------------------------------
def test_taubin_smooth_denoises():
    clean, noisy = _noisy_sphere()
    out = taubin_smooth(noisy, iters=10)
    assert _radial_err(out) < 0.6 * _radial_err(noisy)                  # markedly closer to the true sphere


def test_taubin_smooth_does_not_shrink():
    _, noisy = _noisy_sphere()
    out = taubin_smooth(noisy, iters=10)
    assert _mean_radius(out) > 0.95                                     # overall extent preserved (no shrink)


def test_laplacian_baseline_shrinks():
    _, noisy = _noisy_sphere()
    taub = taubin_smooth(noisy, iters=10)
    lap = laplacian_smooth(noisy, iters=10)
    assert _mean_radius(lap) < _mean_radius(taub) - 0.05               # the naive baseline collapses inward


def test_smoothing_preserves_connectivity_and_chi():
    clean, noisy = _noisy_sphere()
    out = taubin_smooth(noisy, iters=10)
    assert out.faces == clean.faces                                    # only vertices moved
    assert out.euler_characteristic() == clean.euler_characteristic()
    assert out.is_closed() and out.is_manifold()


def test_both_weightings_denoise():
    # both cotangent and uniform reduce the noise; we do NOT claim cotangent is better (it isn't, here)
    _, noisy = _noisy_sphere()
    assert _radial_err(taubin_smooth(noisy, iters=10, weights="cotangent")) < 0.6 * _radial_err(noisy)
    assert _radial_err(taubin_smooth(noisy, iters=10, weights="uniform")) < 0.6 * _radial_err(noisy)


# ---- adjacency format -------------------------------------------------------------------------------
def test_adjacency_is_row_normalised_and_rectangular():
    clean = _icosphere(2)
    for builder in (cotangent_adjacency, uniform_adjacency):
        nbr_idx, nbr_w = builder(clean)
        assert nbr_idx.shape == nbr_w.shape
        assert nbr_idx.shape[0] == clean.n_vertices                    # one row per vertex
        # every row sums to 1 (a proper weighted-average operator) since every sphere vertex has neighbours
        assert np.allclose(nbr_w.sum(axis=1), 1.0, atol=1e-9)


def test_cotangent_weights_are_nonnegative():
    nbr_idx, nbr_w = cotangent_adjacency(_icosphere(2))
    assert np.all(nbr_w >= 0.0)                                         # clamped (obtuse-triangle mitigation)


# ---- works on an n-gon mesh (quads), keeping it a quad mesh -----------------------------------------
def test_smoothing_keeps_quad_topology():
    quad = box(2.0, 2.0, 2.0)                                          # quad faces
    out = taubin_smooth(quad, iters=4)
    assert out.faces == quad.faces                                     # quads preserved (only vertices move)
    assert all(len(f) == 4 for f in out.faces)


# ---- determinism ------------------------------------------------------------------------------------
def test_smoothing_is_deterministic():
    _, noisy = _noisy_sphere()
    a = taubin_smooth(noisy, iters=10)
    b = taubin_smooth(noisy, iters=10)
    assert np.array_equal(a.vertices, b.vertices)

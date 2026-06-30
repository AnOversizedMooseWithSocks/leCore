"""Tests for the landmark (Nystrom) spectral embedding (SCALE-1)."""

import numpy as np
from holographic_nystrom import (farthest_point_landmarks, gaussian_affinity, dense_embedding,
                                 nystrom_embedding, subspace_alignment)


def _blobs(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(c, 0.25, (n, 3)) for c in ([0, 0, 0], [5, 0, 0], [0, 5, 0])])


def test_farthest_point_landmarks_cover_all_clusters():
    P = _blobs(100)
    lm = farthest_point_landmarks(P, 9, seed=0)
    cl = (P[:, 0] > 2.5).astype(int) + 2 * (P[:, 1] > 2.5).astype(int)   # 3 clusters -> labels {0,1,2}
    assert len(np.unique(cl[lm])) == 3                        # every cluster got at least one landmark
    assert len(np.unique(lm)) == 9                            # distinct landmarks


def test_nystrom_matches_dense_on_separable_data():
    P = _blobs(120)
    _, Pd = dense_embedding(P, n_basis=3, sigma=1.0)
    _, Pn = nystrom_embedding(P, n_basis=3, m=48, sigma=1.0)
    assert subspace_alignment(Pd, Pn) > 0.9                   # landmark embedding ~ the exact dense one


def test_nystrom_returns_full_length_embedding():
    P = _blobs(80)
    val, Phi = nystrom_embedding(P, n_basis=4, m=32, sigma=1.0)
    assert Phi.shape == (len(P), 4) and val.shape == (4,)     # one row per point, no N x N ever formed


def test_quality_improves_with_more_landmarks_on_a_manifold():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 3 * np.pi, 600)
    roll = np.stack([t * np.cos(t), rng.uniform(0, 4, 600), t * np.sin(t)], axis=1)
    _, Pd = dense_embedding(roll, n_basis=4, sigma=2.0)
    a_few = subspace_alignment(Pd, nystrom_embedding(roll, 4, m=16, sigma=2.0)[1])
    a_many = subspace_alignment(Pd, nystrom_embedding(roll, 4, m=96, sigma=2.0)[1])
    assert a_many >= a_few                                    # more landmarks -> closer to dense (kept negative: not exact)


def test_fps_more_stable_than_random_on_imbalanced_data():
    rng = np.random.default_rng(2)
    imb = np.vstack([rng.normal([0, 0, 0], 0.5, (400, 3)), rng.normal([6, 6, 6], 0.2, (20, 3))])
    _, Pd = dense_embedding(imb, n_basis=3, sigma=1.0)
    rand = [subspace_alignment(Pd, nystrom_embedding(imb, 3, m=14, sigma=1.0, landmarks="random", seed=s)[1])
            for s in range(6)]
    fps = [subspace_alignment(Pd, nystrom_embedding(imb, 3, m=14, sigma=1.0, landmarks="fps", seed=s)[1])
           for s in range(6)]
    assert np.std(fps) <= np.std(rand) + 1e-6                 # FPS coverage -> lower variance (random can miss the small cluster)

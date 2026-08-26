"""Tests for holographic_chart: the nonlinear manifold chart (reverse-transfer RT-II1). Isomap recovers a
curved manifold's intrinsic structure better than a linear SVD chart, deterministically."""

import numpy as np

from holographic.misc.holographic_chart import knn_graph_euclidean, geodesic_distances, classical_mds, isomap, laplacian_eigenmaps, manifold_chart, _selftest


def _swiss_roll(seed=0, N=300, D=256):
    """A swiss roll (curved 2-manifold) lifted into D dimensions, with its true (u,v) parameter and a 4-class
    banding that a linear projection folds together."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, N)
    v = rng.uniform(0, 1, N)
    ang = 1.5 * np.pi * (1 + 2 * u)
    roll = np.stack([ang * np.cos(ang), 21 * v, ang * np.sin(ang)], 1)
    Q = np.linalg.qr(rng.standard_normal((D, 3)))[0]
    X = roll @ Q.T + 0.05 * rng.standard_normal((N, D))
    lab = np.clip((u * 4).astype(int), 0, 3)
    return X, lab


def _svd_chart(X, dim=2):
    m = X.mean(0)
    return (X - m) @ np.linalg.svd(X - m, full_matrices=False)[2][:dim].T


def _geo_corr(Y, Gtrue):
    iu = np.triu_indices(len(Y), 1)
    dy = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))[iu]
    return float(np.corrcoef(dy, Gtrue[iu])[0, 1])


def _sep(Y, lab):
    Yc = Y - Y.mean(0)
    cen = np.stack([Yc[lab == c].mean(0) for c in range(lab.max() + 1)])
    return float((np.argmin(((Yc[:, None, :] - cen[None, :, :]) ** 2).sum(-1), 1) == lab).mean())


def test_module_selftest():
    _selftest()


def test_isomap_beats_svd_on_geodesic_fidelity_across_seeds():
    wins = 0
    for s in range(5):
        X, _ = _swiss_roll(seed=s)
        Gt = geodesic_distances(X, k=10)
        if _geo_corr(isomap(X, k=10), Gt) > _geo_corr(_svd_chart(X), Gt):
            wins += 1
    assert wins >= 4                                       # robust: the curved-manifold chart preserves geodesics


def test_isomap_separates_classes_the_linear_chart_folds():
    X, lab = _swiss_roll(seed=0)
    assert _sep(isomap(X, k=10), lab) > _sep(_svd_chart(X), lab)


def test_chart_is_deterministic():
    X, _ = _swiss_roll(seed=1)
    a = manifold_chart(X, dim=2, method="isomap")
    b = manifold_chart(X, dim=2, method="isomap")
    assert np.allclose(a, b)                               # eigenvector signs pinned -> bit-stable
    assert a.shape == (len(X), 2)


def test_spectral_method_runs_and_preserves_local_structure():
    # Laplacian Eigenmaps is the honest secondary -- it should at least keep neighbours mostly local even if its
    # global geo-corr trails Isomap. Here we only assert it produces a valid, deterministic chart.
    X, _ = _swiss_roll(seed=0)
    s1 = manifold_chart(X, dim=2, method="spectral")
    s2 = manifold_chart(X, dim=2, method="spectral")
    assert s1.shape == (len(X), 2) and np.allclose(s1, s2)


def test_geodesics_finite_even_when_knn_graph_starts_disconnected():
    # Two well-separated blobs with small k make the raw k-NN graph disconnected; the connectivity repair must
    # bridge them so every geodesic is finite.
    rng = np.random.default_rng(0)
    A = rng.standard_normal((40, 32))
    B = rng.standard_normal((40, 32)) + 50.0
    X = np.vstack([A, B])
    G = geodesic_distances(X, k=4)
    assert np.all(np.isfinite(G))

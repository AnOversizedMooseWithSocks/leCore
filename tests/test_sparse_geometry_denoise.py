"""Sparse cleanup readout + geometry-aware denoise selector -- the VSA-native fix to the
cleanup-ties/loses-to-NN finding (panel review, June 2026).

Pins: (1) softmax readout is bit-for-bit the old path; (2) both readouts collapse to hard NN at
high beta; (3) the MEASURED continuous-manifold win (sparse >> softmax, sparse >= NN); (4) sparse
does not regress discrete recall; (5) effective_rank separates the two geometries; (6) the geometry
router picks the right map AND projection forced on high-rank data is worse (the kept negative);
(7) ABOVE/BELOW -- the mind delegates to the kernel (no divergent reimplementation); (8) Cranmer's
variance harness on the thin sparse-vs-NN margin.
"""
import numpy as np
from holographic.agents_and_reasoning.holographic_ai import cosine
from holographic.agents_and_reasoning.holographic_hopfield import dense_cleanup, _sparsemax
from holographic.rendering.holographic_denoise import effective_rank, manifold_denoise, fit_manifold
from holographic.misc.holographic_unified import UnifiedMind

D = 256


def _slerp(a, b, t):
    a1, b1 = a / np.linalg.norm(a), b / np.linalg.norm(b)
    om = np.arccos(np.clip(a1 @ b1, -1, 1)); so = np.sin(om)
    return ((1 - t) * a + t * b) if so < 1e-6 else (np.sin((1 - t) * om) / so) * a + (np.sin(t * om) / so) * b


def test_sparsemax_simplex():
    w = _sparsemax(np.array([3.0, 1.0, 0.2, -1.0, -2.0]))
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w == 0).any() and (w > 0).any()              # genuinely sparse


def test_softmax_readout_unchanged():
    rng = np.random.default_rng(0)
    V = rng.standard_normal((12, 64)); q = V[3] + 0.3 * rng.standard_normal(64)
    a = dense_cleanup(q, V, beta=25, steps=3)
    b = dense_cleanup(q, V, beta=25, steps=3, readout="softmax")
    assert np.array_equal(a, b)                          # default == explicit softmax, bit-for-bit


def test_high_beta_pins_to_hard_nn():
    rng = np.random.default_rng(1)
    V = rng.standard_normal((10, 48)); Vu = V / np.linalg.norm(V, axis=1, keepdims=True)
    for _ in range(20):
        i = rng.integers(10); q = V[i] + 0.2 * rng.standard_normal(48)
        nn = int((Vu @ (q / np.linalg.norm(q))).argmax())
        for ro in ("softmax", "sparsemax"):
            z = dense_cleanup(q, V, beta=1e4, steps=1, readout=ro)
            assert int((Vu @ (z / np.linalg.norm(z))).argmax()) == nn


def test_sparse_beats_softmax_on_continuous_manifold():
    rng = np.random.default_rng(2)
    A = rng.standard_normal(D); B = rng.standard_normal(D)
    coarse = np.stack([_slerp(A, B, t) for t in np.linspace(0, 1, 6)])
    Cu = coarse / np.linalg.norm(coarse, axis=1, keepdims=True); norm = np.linalg.norm(coarse, axis=1).mean()
    soft = []; sparse = []; nn = []
    for _ in range(300):
        t = rng.uniform(0, 1); clean = _slerp(A, B, t)
        noisy = clean + 1.0 * norm / np.sqrt(D) * rng.standard_normal(D)
        soft.append(cosine(dense_cleanup(noisy, coarse, beta=8, steps=3, readout="softmax"), clean))
        sparse.append(cosine(dense_cleanup(noisy, coarse, beta=8, steps=3, readout="sparsemax"), clean))
        nn.append(cosine(coarse[int((Cu @ (noisy / np.linalg.norm(noisy))).argmax())], clean))
    assert np.mean(sparse) > np.mean(soft) + 0.005       # clear robust win over the softmax blend
    assert np.mean(sparse) >= np.mean(nn) - 0.002        # not worse than NN (the thin-margin claim, guarded)


def test_sparse_does_not_regress_discrete_recall():
    rng = np.random.default_rng(3); V = rng.standard_normal((16, 128))
    Vu = V / np.linalg.norm(V, axis=1, keepdims=True); ok = 0
    for _ in range(200):
        i = rng.integers(16); q = V[i] + 1.0 * rng.standard_normal(128)
        z = dense_cleanup(q, V, beta=40, steps=3, readout="sparsemax")
        ok += int((Vu @ (z / np.linalg.norm(z))).argmax() == i)
    assert ok / 200 > 0.98


def test_effective_rank_distinguishes_geometries():
    rng = np.random.default_rng(4)
    A = rng.standard_normal(D); B = rng.standard_normal(D)
    path = np.stack([_slerp(A, B, t) for t in np.linspace(0, 1, 6)])   # continuous -> low rank
    atoms = rng.standard_normal((6, D))                                # distinct -> high rank
    assert effective_rank(path) <= 3
    assert effective_rank(atoms) >= 5


def test_geometry_router_matches_the_right_map():
    rng = np.random.default_rng(5); mind = UnifiedMind(dim=D, seed=0)
    A = rng.standard_normal(D); B = rng.standard_normal(D)
    path = np.stack([_slerp(A, B, t) for t in np.linspace(0, 1, 6)])
    norm = np.linalg.norm(path, axis=1).mean()
    clean = _slerp(A, B, 0.37); noisy = clean + 1.0 * norm / np.sqrt(D) * rng.standard_normal(D)
    rec = mind.denoise(noisy, method="geometry", samples=path)         # low-rank -> projects
    assert cosine(rec, clean) > 0.95                                   # recovers an UN-stored in-between point

    atoms = rng.standard_normal((12, D)); Au = atoms / np.linalg.norm(atoms, axis=1, keepdims=True)
    i = 4; nm = np.linalg.norm(atoms, axis=1).mean()
    q = atoms[i] + 1.0 * nm / np.sqrt(D) * rng.standard_normal(D)
    g = mind.denoise(q, method="geometry", codebook=atoms, beta=40, readout="sparsemax")  # high-rank -> recalls
    assert int((Au @ (g / np.linalg.norm(g))).argmax()) == i
    basis, mean = fit_manifold(atoms, rank=3); proj = manifold_denoise(q, basis, mean)    # WRONG map on high rank
    assert cosine(proj, atoms[i]) < cosine(g, atoms[i])                # the kept negative: projection is worse


def test_mind_delegates_to_kernel_above_below():
    rng = np.random.default_rng(6); V = rng.standard_normal((10, D)); q = V[2] + 0.4 * rng.standard_normal(D)
    mind = UnifiedMind(dim=D, seed=0)
    for ro in ("softmax", "sparsemax"):
        a = mind.denoise(q, method="codebook", codebook=V, beta=25, steps=3, readout=ro)
        b = dense_cleanup(q, V, beta=25, steps=3, readout=ro)
        assert np.array_equal(a, b)                                    # mind IS the kernel, not a reimplementation


def test_variance_harness_interpolation_margin():
    soft_d = []; nn_d = []
    for seed in range(12):
        rng = np.random.default_rng(100 + seed)
        A = rng.standard_normal(D); B = rng.standard_normal(D)
        coarse = np.stack([_slerp(A, B, t) for t in np.linspace(0, 1, 6)])
        Cu = coarse / np.linalg.norm(coarse, axis=1, keepdims=True); norm = np.linalg.norm(coarse, axis=1).mean()
        s_acc = []; so_acc = []; nn_acc = []
        for _ in range(120):
            t = rng.uniform(0, 1); clean = _slerp(A, B, t)
            noisy = clean + 1.0 * norm / np.sqrt(D) * rng.standard_normal(D)
            s_acc.append(cosine(dense_cleanup(noisy, coarse, beta=8, steps=3, readout="sparsemax"), clean))
            so_acc.append(cosine(dense_cleanup(noisy, coarse, beta=8, steps=3, readout="softmax"), clean))
            nn_acc.append(cosine(coarse[int((Cu @ (noisy / np.linalg.norm(noisy))).argmax())], clean))
        soft_d.append(np.mean(s_acc) - np.mean(so_acc)); nn_d.append(np.mean(s_acc) - np.mean(nn_acc))
    soft_d = np.array(soft_d); nn_d = np.array(nn_d)
    assert soft_d.min() > 0                              # sparse beats softmax on EVERY seed (robust)
    assert nn_d.mean() > -0.002                          # the thin sparse-vs-NN margin is not materially negative

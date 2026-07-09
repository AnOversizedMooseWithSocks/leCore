"""Tests for the landmark (Nystrom) spectral embedding (SCALE-1)."""

import numpy as np
from holographic.sampling_and_signal.holographic_nystrom import farthest_point_landmarks, gaussian_affinity, dense_embedding, nystrom_embedding, subspace_alignment


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


# --- SIM-1: nystrom kernel-field approximation for large smooth fields -----------------------------
def test_nystrom_field_matches_exact_on_smooth_field():
    import numpy as np
    from holographic.sampling_and_signal.holographic_nystrom import nystrom_kernel_apply, exact_kernel_apply
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((600, 3)); w = rng.standard_normal(600)
    ex = exact_kernel_apply(pts, pts, w, sigma=1.0)
    ap = nystrom_kernel_apply(pts, pts, w, sigma=1.0, m=64)
    assert np.corrcoef(ex, ap)[0, 1] > 0.99                        # smooth (low-rank) field: faithful


def test_nystrom_field_degrades_on_high_frequency():
    import numpy as np
    from holographic.sampling_and_signal.holographic_nystrom import nystrom_kernel_apply, exact_kernel_apply
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((600, 3)); w = rng.standard_normal(600)
    ex = exact_kernel_apply(pts, pts, w, sigma=0.08)
    ap = nystrom_kernel_apply(pts, pts, w, sigma=0.08, m=64)
    assert np.corrcoef(ex, ap)[0, 1] < 0.8                         # kept negative: high-freq field is full-rank


# ======================================================================================================
# The landmark rule is DATA-DEPENDENT, and the shipped default is not uniformly better.
# ======================================================================================================
def _sigma_of(P, m=24):
    from holographic.sampling_and_signal.holographic_nystrom import farthest_point_landmarks
    L = P[farthest_point_landmarks(P, m, seed=0)]
    DL = np.sqrt(((L[:, None, :] - L[None, :, :]) ** 2).sum(-1))
    return float(np.median(DL[DL > 0])) or 1.0


def _mean_align(P, sig, rule, m=24, k=6, seeds=5):
    from holographic.sampling_and_signal.holographic_nystrom import (dense_embedding, nystrom_embedding,
                                                                     subspace_alignment)
    _, Pd = dense_embedding(P, n_basis=k, sigma=sig)
    return float(np.mean([subspace_alignment(Pd, nystrom_embedding(
        P, n_basis=k, m=m, sigma=sig, landmarks=rule, seed=s)[1]) for s in range(seeds)]))


def test_fps_wins_when_there_is_nothing_to_cover():
    """The regime the default was chosen for. Measured 0.9889 +- 0.0033 against random's 0.9743 +- 0.0126 over 50
    trials; random won 2 of them."""
    rng = np.random.default_rng(0)
    blob = rng.uniform(-1, 1, (320, 2))
    sig = _sigma_of(blob)
    assert _mean_align(blob, sig, "fps") > _mean_align(blob, sig, "random")


def test_with_outliers_the_coverage_argument_inverts_and_random_wins():
    """KEPT NEGATIVE against the module's own former docstring, which said FPS covers every cluster "unlike
    uniform-random". Coverage is real; it is not what a spectral embedding wants. FPS spends its budget on the
    stragglers while the leading eigenvectors carry their mass in the DENSE regions. Measured 0.8692 +- 0.0216
    against random's 0.9512 +- 0.0315 over 50 trials -- random wins 47."""
    rng = np.random.default_rng(0)
    core = np.vstack([rng.normal(c, 0.12, (120, 2)) for c in ([0, 0], [2.5, 0.3], [1.2, 2.2])])
    P = np.vstack([core, rng.uniform(-4, 6, (40, 2))])
    sig = _sigma_of(P)
    fps, rand = _mean_align(P, sig, "fps"), _mean_align(P, sig, "random")
    assert rand > fps + 0.02, (rand, fps)


def test_coarse_first_landmark_selection_loses_and_the_gate_cannot_see_it():
    """The last coarse-first retirement, and the most interesting: the GATE PASSES and the method still loses.
    Pivoting the next landmark onto the maximum diagonal residual picks the most ISOLATED point -- the residual is a
    COVERAGE signal, and a spectral embedding needs a MASS signal. `concentration` scores it 'concentrated' anyway,
    which is exactly what 'necessary, not sufficient' means."""
    from holographic.misc.holographic_coarsefirst import concentration
    from holographic.sampling_and_signal.holographic_nystrom import (dense_embedding, farthest_point_landmarks,
                                                                     gaussian_affinity, nystrom_embedding,
                                                                     subspace_alignment)

    def residual_landmarks(P, m, sigma, seed=0):
        idx = [int(np.random.default_rng(seed).integers(len(P)))]
        r = None
        while len(idx) < m:
            L = P[idx]
            Wnm = gaussian_affinity(P, L, sigma)
            r = 1.0 - np.einsum("ij,jk,ik->i", Wnm, np.linalg.pinv(gaussian_affinity(L, L, sigma)), Wnm)
            r[idx] = -np.inf
            idx.append(int(np.argmax(r)))
        return np.array(idx), r

    rng = np.random.default_rng(0)
    core = np.vstack([rng.normal(c, 0.12, (120, 2)) for c in ([0, 0], [2.5, 0.3], [1.2, 2.2])])
    P = np.vstack([core, rng.uniform(-4, 6, (40, 2))])
    sig = _sigma_of(P)
    _, Pd = dense_embedding(P, n_basis=6, sigma=sig)

    lm, resid = residual_landmarks(P, 24, sig)
    # a hand-rolled embedding on the residual landmarks, using nystrom's own machinery for the rest
    from holographic.sampling_and_signal.holographic_nystrom import gaussian_affinity as ga
    L = P[lm]
    Wmm, Wnm = ga(L, L, sig), ga(P, L, sig)
    deg = np.maximum(Wnm @ (np.linalg.pinv(Wmm) @ (Wnm.T @ np.ones(len(P)))), 1e-12)
    dnm, dmm = 1 / np.sqrt(deg), 1 / np.sqrt(deg[lm])
    val, U = np.linalg.eigh(dmm[:, None] * Wmm * dmm[None, :])
    val, U = np.maximum(val[::-1], 1e-12), U[:, ::-1]
    Phi = (dnm[:, None] * Wnm * dmm[None, :]) @ (U[:, :6] / val[:6])
    Phi /= np.linalg.norm(Phi, axis=0, keepdims=True) + 1e-12

    a_resid = subspace_alignment(Pd, Phi)
    a_rand = _mean_align(P, sig, "random")
    assert a_resid < a_rand - 0.05, (a_resid, a_rand)      # coarse-first loses to a random control

    assert concentration(np.clip(resid, 0, None)) > 0.2    # ...and the gate says CANDIDATE anyway

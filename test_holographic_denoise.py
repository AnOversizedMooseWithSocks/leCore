"""Denoising as manifold projection + Plug-and-Play restoration (B7)."""
import numpy as np
from holographic_denoise import (fit_manifold, manifold_denoise, pnp_restore,
                                 codebook_denoise, nlm_denoise)


def _low_rank_signals(n=400, dim=32, rank=5, seed=0):
    rng = np.random.default_rng(seed)
    B = np.linalg.svd(rng.standard_normal((rank, dim)), full_matrices=False)[2]
    coeffs = rng.standard_normal((n, rank))
    return coeffs @ B                          # genuinely rank-`rank` signals


def _snr(clean, est):
    return 10 * np.log10(np.sum(clean ** 2) / (np.sum((clean - est) ** 2) + 1e-12))


def test_manifold_projection_denoises_high_noise_low_rank_signal():
    rng = np.random.default_rng(1)
    X = _low_rank_signals()
    basis, mean = fit_manifold(X[:300], rank=6)
    raw = proj = 0.0
    for x in X[300:380]:
        noisy = x + 0.7 * rng.standard_normal(x.shape[0])
        raw += _snr(x, noisy); proj += _snr(x, manifold_denoise(noisy, basis, mean))
    assert proj > raw                          # projection helps when noise dominates off-manifold


def test_manifold_projection_does_not_help_random_data():
    # honest control: no low-rank manifold -> projecting onto a spurious subspace cannot denoise.
    rng = np.random.default_rng(2)
    X = rng.standard_normal((400, 32))
    basis, mean = fit_manifold(X[:300], rank=6)
    raw = proj = 0.0
    for x in X[300:380]:
        noisy = x + 0.7 * rng.standard_normal(32)
        raw += _snr(x, noisy); proj += _snr(x, manifold_denoise(noisy, basis, mean))
    assert proj < raw + 0.5                     # no real gain (and typically a loss)


def test_pnp_restore_recovers_an_inpainting_problem():
    # use the manifold denoiser as the prior to fill erased entries (A = a binary mask, A^T == A).
    rng = np.random.default_rng(3)
    X = _low_rank_signals(dim=40, rank=5)
    basis, mean = fit_manifold(X[:300], rank=6)
    x = X[350]
    mask = (rng.random(40) > 0.4).astype(float)        # keep ~60% of entries
    A = lambda v: mask * v
    y = A(x)
    den = lambda v: manifold_denoise(v, basis, mean)
    rec = pnp_restore(y, A, A, den, mu=0.8, steps=60)
    assert _snr(x, rec) > _snr(x, y)                   # restoration beats the masked measurement


def _motif_signal(M, R, D=24, sigma=0.6, seed=0):
    rng = np.random.default_rng(seed)
    motifs = rng.standard_normal((M, D))
    motifs /= np.linalg.norm(motifs, axis=1, keepdims=True)
    clean = np.repeat(motifs, R, axis=0)
    return clean, clean + sigma * rng.standard_normal(clean.shape)


def test_nlm_beats_projection_on_self_similar_signal():
    # repeated motifs -> NLM averages the near-duplicates and cancels noise; projection cannot.
    clean, noisy = _motif_signal(M=20, R=8)
    basis, mean = fit_manifold(noisy, rank=8)
    proj = np.stack([manifold_denoise(x, basis, mean) for x in noisy])
    nlm = nlm_denoise(noisy, k=8, use_forest=True)
    s = lambda A: np.mean([_snr(clean[i], A[i]) for i in range(len(clean))])
    assert s(nlm) > s(proj) and s(nlm) > s(noisy) + 3.0


def test_projection_beats_nlm_without_self_similarity():
    # KEPT NEGATIVE / complementarity: low-rank but every patch unique -> NLM has no duplicates to
    # average, projection captures the subspace and wins. The two denoisers cover different worlds.
    rng = np.random.default_rng(3)
    X = _low_rank_signals(n=400, dim=32, rank=5)        # all-unique low-rank patches
    noisy = X + 0.6 * rng.standard_normal(X.shape)
    basis, mean = fit_manifold(noisy, rank=6)
    proj = np.stack([manifold_denoise(x, basis, mean) for x in noisy])
    nlm = nlm_denoise(noisy, k=8, use_forest=True)
    s = lambda A: np.mean([_snr(X[i], A[i]) for i in range(len(X))])
    assert s(proj) > s(nlm)


def test_forest_recall_k_finds_near_duplicates():
    # the recall step: a duplicated patch's k-nearest should be dominated by its other copies.
    from holographic_tree import HoloForest
    clean, noisy = _motif_signal(M=10, R=8, sigma=0.2, seed=5)
    f = HoloForest(noisy.shape[1], n_trees=4, leaf_size=8, seed=0).build(noisy)
    idx, sims = f.recall_k(noisy[0], k=6)
    assert len(idx) >= 1 and sims[0] >= sims[-1]        # ranked descending
    # the closest neighbours should come from the same motif block (the first 8 rows)
    assert np.mean(idx[:4] < 8) >= 0.5


# ---- trajectory denoise: lone-1-D-signal prior, promoted out of the pipeline (above/below sweep) -----

def test_trajectory_denoise_cleans_a_lone_1d_signal():
    """trajectory_denoise gives a LONE 1-D signal the prior it lacks, from its own sliding windows (SSA):
    a smooth/periodic signal's Hankel matrix is low-rank, so the windows project onto their own subspace and
    the signal rebuilds by anti-diagonal averaging. On such a signal the error drops well below the noisy
    input. (No free lunch: the prior IS the signal's structure, so a structureless signal has nothing to
    recover -- the method can only shrink it, not restore a signal that was never there.)"""
    from holographic_denoise import trajectory_denoise
    t = np.linspace(0, 1, 256)
    clean = np.sin(2 * np.pi * 3 * t) + 0.5 * t
    noisy = clean + 0.4 * np.random.default_rng(0).standard_normal(256)
    den = trajectory_denoise(noisy)
    assert np.linalg.norm(den - clean) < 0.7 * np.linalg.norm(noisy - clean)


def test_trajectory_method_is_the_pipeline_denoiser_promoted():
    """The denoise faculty exposes the trajectory denoiser as method='trajectory', and the pipeline's private
    _denoise_signal is now a thin delegate to it -- one shared implementation, bit-identical."""
    from holographic_unified import UnifiedMind
    m = UnifiedMind(dim=256, seed=0)
    t = np.linspace(0, 1, 200)
    sig = 1 + 2 * t + 3 * t ** 2 + 0.3 * np.random.default_rng(0).standard_normal(200)
    faculty = np.asarray(m.denoise(sig, method="trajectory"))
    private = np.asarray(m._denoise_signal(sig))
    assert np.array_equal(faculty, private)


def test_denoise_gate_routes_but_is_opt_in():
    """The projection-denoise re-enable ROUTES on the residual ratio: clearly-noise-dominated -> project, else
    fall back to the no-op. NOTE (kept negative): this gate is OPT-IN, not an auto-default -- a fixed threshold
    can't robustly separate off-manifold detail from noise (measured harm-leak at strong detail). It's safe only
    when the caller knows the signal is low-rank. Here we test the mechanical routing + the fallback identity."""
    import numpy as np
    from holographic_denoise import fit_manifold, manifold_denoise, denoise_gated
    rng = np.random.default_rng(0); D, rank = 128, 8
    Q = np.linalg.qr(rng.standard_normal((D, D)))[0]; base, det = Q[:, :rank], Q[:, rank:rank+24]
    sc = lambda n: (rng.standard_normal((n, rank)) @ base.T) + 0.35 * (rng.standard_normal((n, 24)) @ det.T)
    basis, mean = fit_manifold(sc(300), rank=rank)
    lo = sc(1)[0] + 0.05 * rng.standard_normal(D)                 # low noise -> fallback (identity)
    hi = sc(1)[0] + 0.8 * rng.standard_normal(D)                  # high noise -> project
    r_lo, i_lo = denoise_gated(lo, basis, mean)
    r_hi, i_hi = denoise_gated(hi, basis, mean)
    assert i_lo["used"] == "fallback" and np.array_equal(r_lo, np.asarray(lo, float))   # safe no-op
    assert i_hi["used"] == "superior" and np.array_equal(r_hi, manifold_denoise(hi, basis, mean))

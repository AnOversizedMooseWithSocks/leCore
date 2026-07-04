"""Denoising as manifold projection, and the Plug-and-Play / RED restoration loop.

WHY THIS EXISTS
---------------
Milanfar's thesis: a denoiser is a MAP OF THE MANIFOLD that clean signals live on. holostuff
already owns two such maps -- `cleanup` (snap to the codebook manifold) and `consolidation` (the
low-rank SVD subspace real states occupy). This module exposes them as a callable denoiser and
wraps the standard Plug-and-Play / Regularization-by-Denoising loop (Venkatakrishnan et al. 2013;
Romano, Elad, Milanfar 2017) around them, so the SAME map that denoises also solves any inverse
problem (inpainting, deblurring, reconstruction-under-erasure).

MEASURED (on real SOL price windows; manifold = the consolidation/SVD subspace of clean windows)
  * Projection denoising WINS increasingly as noise grows: +1.7 dB SNR at sigma=0.5, +3.85 dB at
    sigma=0.8 -- the off-manifold directions become mostly noise.
  * KEPT NEGATIVE: it HURTS at low noise (-1.4 dB at sigma=0.3) by discarding real signal detail
    -- projecting onto a fixed low rank over-smooths a clean signal. The method needs the noise
    level (or an adaptive rank/threshold) to be applied well -- exactly the Donoho/Milanfar
    threshold-selection problem.
  * HONEST CONTROL: on random data with no low-rank manifold, projection DESTROYS signal (-5 dB).
    The map only helps where real structure exists. `manifold_denoise` is therefore only a
    denoiser to the extent the fitted basis captures the signal.

DESIGN NOTES
  * `fit_manifold` is the same operation as the engine's consolidation (SVD), kept standalone here
    so denoising does not depend on a creature/brain being present.
  * `pnp_restore` accepts ANY denoiser callable -- pass `manifold_denoise` (projection) or the
    modern-Hopfield `dense_cleanup` (codebook) -- so the loop is agnostic to which map you use.
  * Pure NumPy, deterministic.
"""

import numpy as np


def fit_manifold(samples, rank=8):
    """Learn a signal manifold from clean `samples` (rows) as a rank-`rank` SVD subspace.

    Returns (basis, mean) where basis is (rank x dim) orthonormal -- this IS the consolidation
    step, used here as the map a denoiser projects onto."""
    X = np.asarray(samples, float)
    mean = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
    rank = int(min(rank, Vt.shape[0]))
    return Vt[:rank], mean


def manifold_denoise(x, basis, mean):
    """Denoise `x` by projecting onto the affine manifold (mean + span(basis)).

    The holostuff denoiser: keep only the components that lie in the signal subspace, drop the
    rest as noise. Only as good as the basis -- see the kept negative in the module docstring."""
    x = np.asarray(x, float)
    return mean + (x - mean) @ basis.T @ basis


def fit_manifold_full(samples, rank=None):
    """Like fit_manifold but keeps a GENEROUS basis AND its singular values, so a per-signal noise
    threshold (adaptive_manifold_denoise) can decide how many components to keep at denoise time.
    rank=None keeps every component. Returns (basis, sv, mean)."""
    X = np.asarray(samples, float)
    mean = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    if rank is not None:
        Vt, S = Vt[:int(rank)], S[:int(rank)]
    return Vt, S, mean


def effective_rank(samples, energy=0.95):
    """Effective rank of a row set: how many SVD directions hold `energy` of the centred variance.

    LOW relative to the row count => the rows lie near a low-dimensional CONTINUOUS manifold, where a
    projection denoiser is right (projecting removes the off-manifold noise; measured cosine ~1.0 on a
    rank-2 SD-latent path). HIGH (~= the row count) => the rows are distinct, high-rank atoms, where
    projection DESTROYS signal -- measured: projecting the high-rank mountain-photo latents collapsed
    recall to 67% -- and codebook recall is right instead. This is the knee the geometry-aware denoiser
    routes on. Returns an int in [0, n_rows]; pure NumPy, deterministic."""
    X = np.asarray(samples, float)
    if X.ndim == 1 or len(X) < 2:
        return min(1, len(X))
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    if s.size == 0 or s[0] == 0:
        return 0
    cum = np.cumsum(s ** 2) / float(np.sum(s ** 2))
    return int(np.searchsorted(cum, energy) + 1)


def estimate_sigma(x):
    """Donoho's robust noise estimate: the MAD of the finest detail (successive differences),
    rescaled. Parameter-free; good when the clean signal is smoother than the noise."""
    d = np.diff(np.asarray(x, float))
    if d.size == 0:
        return 0.0
    return float(np.median(np.abs(d - np.median(d))) / 0.6745 / np.sqrt(2.0))


def adaptive_manifold_denoise(x, basis, mean, sigma=None, kappa=1.0):
    """Adaptive denoiser: project x onto a GENEROUS manifold basis, then HARD-THRESHOLD the projection
    coefficients at a NOISE-DRIVEN level (Donoho-Johnstone shrinkage in the manifold basis). With an
    orthonormal basis each coefficient carries noise of std ~sigma, so dropping |c| <= kappa*sigma*sqrt(
    2 ln r) removes noise-dominated directions and keeps signal-bearing ones.

    This cashes the fixed-rank denoiser's kept negative -- at LOW noise the threshold is tiny so nearly
    all detail survives (no over-smoothing), while a fixed rank-k always truncates and discards real
    detail there; at HIGH noise only strong signal components survive (full denoising). sigma is
    estimated from x if not given (the Donoho/Milanfar threshold-selection step)."""
    x = np.asarray(x, float)
    r = basis.shape[0]
    if sigma is None:
        sigma = estimate_sigma(x)
    thr = kappa * sigma * np.sqrt(2.0 * np.log(max(r, 2)))   # universal threshold, scaled
    c = (x - mean) @ basis.T                                 # projection coefficients
    c = np.where(np.abs(c) > thr, c, 0.0)                    # keep only signal-bearing directions
    return mean + c @ basis


def codebook_denoise(x, codebook, beta=25.0, steps=3, readout="softmax"):
    """Denoise `x` by snapping toward the codebook manifold via the modern-Hopfield update.
    Thin re-export of holographic_hopfield.dense_cleanup so callers can pick a manifold (subspace)
    or a codebook denoiser without importing two modules. `readout='sparsemax'` selects the sparse
    (Hopfield-Fenchel-Young) readout that does not over-smooth a continuous codebook manifold."""
    from holographic_hopfield import dense_cleanup
    return dense_cleanup(x, codebook, beta=beta, steps=steps, readout=readout)


def project_onto_constraints(x, projections, iters=30, tol=None, omega=1.0):
    """Iterated projection -- satisfy a set of constraints by repeatedly projecting onto each in turn. This
    is the structure three things the engine grew separately all share (Macklin's observation -- the same
    object he builds in position-based dynamics):

      * the SBC resonator's alternating cleanup -- project each factor onto its codebook holding the others;
      * the PnP/RED denoise loop -- a data-fidelity projection then a manifold/codebook denoise (below);
      * a position-based-dynamics constraint sweep -- project each particle onto each constraint in turn.

    `projections` is a list of callables x->x', each snapping x onto one constraint set / manifold; they are
    swept in order. With `omega` < 1 the update is UNDER-RELAXED (x <- x + omega*(proj(x)-x)), PBD's trick for
    stability when many constraints fight. When the projections are onto convex sets this IS von Neumann /
    POCS alternating projection and converges to a point in their intersection.

    `iters` sweeps; `tol` (if set) stops early once a full sweep moves x by less than tol (relative); `tol=None`
    runs all `iters` -- what the PnP loop wants. Returns (x, n_sweeps, converged) where `converged` is True only
    when `tol` triggered an early stop. Deterministic given deterministic projections (no RNG of its own)."""
    x = np.asarray(x, float).copy()
    for it in range(iters):
        prev = x.copy()
        for proj in projections:
            px = np.asarray(proj(x), float)
            x = px if omega == 1.0 else x + omega * (px - x)
        if tol is not None and np.linalg.norm(x - prev) <= tol * (np.linalg.norm(prev) + 1e-12):
            return x, it + 1, True
    return x, iters, False


def pnp_restore(y, forward, adjoint, denoiser, mu=0.5, steps=30, x0=None):
    """Plug-and-Play / RED restoration: recover x from a degraded measurement y = forward(x)+noise
    by alternating a data-fidelity gradient step with a denoise step.

        x <- x - mu * adjoint(forward(x) - y)     # pull toward agreement with the measurement
        x <- denoiser(x)                          # pull toward the signal manifold (the prior)

    `forward`/`adjoint` are callables for the degradation operator A and its transpose A^T (for
    inpainting, A is a binary mask and A^T == A; for plain denoising, both are identity). Any
    denoiser callable works. Returns the restored vector. Deterministic given a deterministic
    denoiser and x0. This IS `project_onto_constraints` with two projections -- data-fidelity, then the
    denoiser -- so the PnP loop and the resonator are literally the same iterated-projection engine."""
    y = np.asarray(y, float)
    x0v = np.asarray(x0, float).copy() if x0 is not None else adjoint(y).astype(float)
    data_fidelity = lambda x: x - mu * adjoint(forward(x) - y)   # the projection toward the measurement
    x, _, _ = project_onto_constraints(
        x0v, [data_fidelity, lambda x: np.asarray(denoiser(x), float)], iters=steps, tol=None)
    return x


def nlm_denoise(patches, k=12, h=0.5, use_forest=True):
    """Non-local-means denoising (Buades, Coll, Morel 2005; BM3D, Dabov et al. 2007) running on
    holostuff's OWN content-addressable recall.

    "Find the patches that look like this one and average them" -- averaging k near-duplicate
    patches cancels the independent noise in each (a ~1/sqrt(k) reduction). The neighbour search
    is exactly recall, so it runs sub-linearly through `HoloForest.recall_k`; with use_forest=False
    it falls back to exact cosine kNN (handy for small sets and for a deterministic reference).

    COMPLEMENTARY to manifold_denoise, not a replacement -- measured:
      * self-similar signals (repeated motifs): NLM wins big (averages the duplicates) -- e.g. on
        real SOL motif-windows, ~11 dB vs ~7 dB for rank-8 projection.
      * low-rank but NOT self-similar (every patch unique): projection wins, NLM has nothing to
        average -- ~2.8 dB vs ~0.5 dB. KEPT NEGATIVE: NLM only helps where near-duplicates exist.

    `patches` is (N, D). Returns the denoised (N, D). Weights are softmax(cosine / h) over the
    k nearest (including self), so a closer neighbour counts more; smaller h = more selective."""
    X = np.asarray(patches, float)
    N = len(X)
    k = min(k, N)
    out = np.empty_like(X)

    if use_forest:
        from holographic_tree import HoloForest
        forest = HoloForest(X.shape[1], n_trees=4, leaf_size=max(8, N // 16), seed=0).build(X)
        for i in range(N):
            idx, sims = forest.recall_k(X[i], k=k)
            if len(idx) == 0:                       # nothing routed -> keep the patch as-is
                out[i] = X[i]; continue
            w = np.exp(sims / h); w /= w.sum()
            out[i] = w @ X[idx]
        return out

    # exact reference path: all-pairs cosine, top-k per patch
    U = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = U @ U.T
    for i in range(N):
        idx = np.argsort(S[i])[::-1][:k]
        w = np.exp(S[i, idx] / h); w /= w.sum()
        out[i] = w @ X[idx]
    return out


def trajectory_denoise(x, window=None, rank=8):
    """Denoise a lone 1-D SIGNAL against its OWN low-rank trajectory -- the prior a single vector lacks,
    built from the signal itself (so this is the second prior-free denoiser beside nlm, which needs a patch
    SET; this takes a raw 1-D signal). A smooth/structured signal's sliding-window Hankel matrix is LOW-RANK
    (Broomhead-King / Cadzow SSA), so projecting those windows onto their dominant subspace (the adaptive,
    noise-thresholded manifold map) removes noise; reconstruct the signal by averaging anti-diagonals. Roughly
    idempotent on a clean low-rank signal, so iterating it CONVERGES. Returns the denoised 1-D signal.

    SCOPE/NEGATIVE: helps where the signal really is locally low-rank (smooth, periodic, polynomial); on a
    structureless signal the trajectory matrix is full-rank and nothing is removed (no free lunch -- the
    prior is the signal's own structure, and a structureless signal has none)."""
    x = np.asarray(x, float).ravel()
    n = x.size
    if n < 8:                                           # too short to form a trajectory matrix -- pass through
        return x
    L = window or max(4, n // 4)                        # window length; K = n-L+1 windows (the signal's own)
    L = min(L, n - 1)
    K = n - L + 1
    H = np.stack([x[i:i + L] for i in range(K)])        # (K, L) trajectory (Hankel) matrix
    # the GENEROUS-basis + noise-threshold map (exactly what denoise(method='adaptive') uses): a wide
    # subspace whose coefficients are Donoho-Johnstone thresholded, so the rank adapts to the noise.
    basis, _, mean = fit_manifold_full(H, rank=min(4 * rank, H.shape[1]))
    Hd = adaptive_manifold_denoise(H, basis, mean)      # project + threshold the windows
    out = np.zeros(n); cnt = np.zeros(n)                # anti-diagonal averaging -> back to a 1-D signal
    for i in range(K):
        out[i:i + L] += Hd[i]; cnt[i:i + L] += 1.0
    return out / np.maximum(cnt, 1.0)


# ---------------------------------------------------------------------------------------------------------------
# RE-ENABLE (consolidation, adaptive-dispatch audit): the fixed-rank PROJECTION denoiser was a kept negative --
# it WINS increasingly as noise grows (+3.85 dB @ sigma=0.8) but HURTS at low noise (over-smooths a near-clean
# signal, discarding real detail that lives beyond the fitted rank). With a catalog + adaptive dispatch, we can
# gate it: project only when a cheap, deterministic detector says the discarded directions are noise-dominated.
#
# THE DETECTOR (measured, robust). Raw estimate_sigma is marginal here because off-manifold DETAIL looks like
# noise to a difference-based estimator. The reliable signal is the RESIDUAL RATIO: of the energy the projection
# would remove (the residual r = x - project(x), i.e. the discarded off-manifold directions), how much is
# explained by NOISE ALONE? Under noise of std sigma, the discarded (D-rank)-dim subspace carries expected energy
# sigma^2 * (D-rank). So
#         ratio = ||x - project(x)||^2 / (sigma_est^2 * (D - rank))
# is ~1 when the discarded directions are pure noise (projecting is safe -> DO it) and well below 1 when they hold
# structured signal (projecting would over-smooth -> DON'T). Measured, the ratio rises monotonically with noise and
# crosses the hurt/help boundary at ~0.55; we gate CONSERVATIVELY at 0.6 (project only when clearly noise-dominated,
# so a misfire falls back to the harmless no-op rather than the -dB negative).

def projection_residual_ratio(x, basis, mean):
    """The regime detector for projection denoising: energy the projection would discard, over the energy noise
    ALONE would put in that discarded subspace. ~1 => discarded directions are noise (safe to project); <<1 =>
    they hold real signal detail (projecting would over-smooth). Cheap (one projection) and deterministic."""
    x = np.asarray(x, float)
    resid = x - manifold_denoise(x, basis, mean)          # the off-manifold component projection removes
    sigma = estimate_sigma(x)
    n_discarded = max(int(x.shape[-1]) - int(np.asarray(basis).shape[0]), 1)
    noise_energy = sigma * sigma * n_discarded + 1e-9     # expected residual energy under noise alone
    return float(np.sum(resid * resid) / noise_energy)


def denoise_gated(x, basis, mean, threshold=0.6):
    """Projection denoise, RE-ENABLED behind its regime detector. Projects x onto the fitted manifold ONLY when the
    residual ratio says the off-manifold directions are noise-dominated (ratio >= threshold); otherwise returns x
    UNCHANGED -- the safe fallback that avoids the low-noise over-smoothing negative. Returns (result, info) where
    info records the score / threshold / which path ran, so the re-enabled method stays measurable. Deterministic.

    `threshold` is biased conservative (0.6): borderline signals fall back to the no-op. Lower it toward the measured
    knee (~0.55) to also capture the small wins just past the crossover, at a little more risk of over-smoothing."""
    from holographic_regimegate import RegimeGate
    gate = RegimeGate("projection_denoise", detect=projection_residual_ratio, threshold=threshold,
                      superior=lambda z, b, m: manifold_denoise(z, b, m),   # the shelved aggressive projection
                      fallback=lambda z, b, m: np.asarray(z, float))        # safe default: leave it alone
    return gate.apply(x, basis, mean)

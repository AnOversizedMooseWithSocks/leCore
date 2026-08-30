"""Part 03b of UnifiedMind's faculty surface -- the denoise verbs (smooth_sharp_split ..
denoise), split out of part 03 in sweep 114 when that part crossed the 2,000-line
budget test_unified_split pins (the whole point of the split was file size; a part
that grows past the cap gets split again, not a raised cap).

NOT A STANDALONE MODULE. One slice of the single `UnifiedMind` class, assembled by
holographic/misc/holographic_unified.py, which is still the only import path anyone
uses. Carries no `__init__`; assumes the state UnifiedMind.__init__ sets up.
"""
import numpy as np

from holographic.unified import check_part


class _UnifiedPart03B:

    def smooth_sharp_split(self, x, k_smooth, k_sharp):
        """Split a signal into a SMOOTH layer (its k_smooth lowest-frequency coefficients) and a SHARP layer (the
        k_sharp largest residual samples -- sparse in the sample domain) (CACHE-2). At a budget covering both
        layers this beats any single basis, because no single basis is cheap across smooth-plus-sharp content (the
        spikes are broadband in frequency but sparse in samples). Returns a TwoLayerCode; reconstruct with
        smooth_sharp_reconstruct. The right sharp basis matches the sharp content (sample-sparse for spikes)."""
        from holographic.misc.holographic_twolayer import smooth_sharp_split
        return smooth_sharp_split(np.asarray(x, float), k_smooth, k_sharp)

    def smooth_sharp_reconstruct(self, code):
        """Reconstruct a signal from a two-layer code (CACHE-2): the smooth layer everywhere plus the exact sharp
        residual at the stored sharp positions."""
        from holographic.misc.holographic_twolayer import smooth_sharp_reconstruct
        return smooth_sharp_reconstruct(code)

    def graph_denoise(self, vectors, k=8, method="taubin", lam=0.55, mu=-0.58, iters=8, sublinear=False):
        """Denoise / regularize a SET of vectors (a noisy codebook, an embedding, a value function) over its
        own k-NN similarity graph -- the graph-signal filter the stack lacked (reverse-transfer RT-III1; mesh
        smoothing mapped back onto the concept graph). `method='taubin'` is Taubin's lambda|mu no-shrink
        low-pass; 'laplacian' is the naive shrinking baseline. Where `denoise` cleans ONE vector against a
        manifold, this cleans a whole set USING its own redundancy (non-local means on the graph).

        Helps most at HIGH noise on a curved manifold whose local neighbourhoods survive when the global linear
        subspace is corrupted (measured: beats per-vector consolidation 6/6 seeds at rel-noise 1.2, and Taubin
        keeps its norm where the naive Laplacian collapses). KEPT NEGATIVE: at low noise a per-vector
        consolidation denoiser is better and this over-smooths. `sublinear=True` builds the k-NN graph from a
        HoloForest's recall_k instead of the O(n^2) dense scan -- reuse the index for large sets."""
        from holographic.misc.holographic_graphsignal import graph_denoise
        forest = None
        if sublinear:
            from holographic.misc.holographic_tree import HoloForest
            V = np.asarray(vectors, float)
            forest = HoloForest(V.shape[1], seed=self.seed).build(V)   # index over the vectors' own dim
        return graph_denoise(vectors, k=k, method=method, lam=lam, mu=mu, iters=iters, forest=forest)

    def denoise(self, x, method="auto", samples=None, codebook=None, sigma=None,
                rank=8, beta=25.0, steps=3, forward=None, adjoint=None, mu=0.5, pnp_steps=30,
                readout="softmax", points=None, spectral_k=10, spectral_nbasis=12, check_manifold=False):
        """Clean a noisy signal by projecting it onto a manifold -- Milanfar's thesis that a denoiser
        IS a map of the manifold clean signals live on. One call over holographic_denoise +
        holographic_hopfield, picking the map by the structure you supply a prior for:

          method='adaptive' : project onto a low-rank SVD subspace fit from `samples`, then
                              noise-THRESHOLD the coefficients (Donoho-Johnstone). The safe default for
                              low-rank signals -- estimates the noise level itself, so it does not
                              over-smooth at low noise.
          method='manifold' : plain FIXED-rank projection onto the subspace fit from `samples`.
          method='codebook' : modern-Hopfield cleanup of `x` toward a discrete `codebook` manifold.
          method='nlm'       : non-local means -- `x` is a (N, dim) patch set; average each patch with
                              its near-duplicates via the engine's own content-addressable recall.
          method='trajectory': clean a LONE 1-D signal with no external prior -- its sliding-window Hankel
                              matrix is low-rank for a smooth/structured signal (SSA/Cadzow), so project the
                              windows onto their own subspace and reconstruct. The second prior-free method
                              beside nlm (nlm needs a patch SET; this takes a raw 1-D signal).
          method='spectral' : clean a lone scalar FIELD living on a known manifold GEOMETRY -- pass the point
                              coordinates as points=<(N, d)> and x as the field value at each of those N points.
                              Builds the kNN graph-Laplacian eigenbasis (EXP-5/6) and projects the field onto
                              its low-frequency modes. The NONLINEAR-manifold map the linear methods lack: it is
                              the only denoiser here that needs no example set and no codebook, just the cloud's
                              own geometry. Measured on a smooth field over a 2-sphere, it cleans error 4.1->0.9
                              where the geometry-blind options barely move it (trajectory 3.1, DCT 4.2) -- a
                              linear/1-D prior cannot see a curved manifold's smoothness.
          method='pnp'       : Plug-and-Play / RED restoration of a degraded measurement x = forward(clean)
                              + noise, using the adaptive manifold map as the prior (needs forward/adjoint).
          method='auto'      : codebook if a `codebook` is given, else adaptive manifold if `samples`
                              are given. NLM and PnP stay OPT-IN: deciding self-similar-vs-low-rank
                              automatically is itself a measurement we will not fake -- name them.
          method='geometry'  : route by the GEOMETRY of the set you hand (samples= or codebook=). Read its
                              effective rank; if LOW relative to the row count (a continuous manifold)
                              project onto that subspace; if HIGH (distinct atoms) do codebook recall. This
                              is the measured 'match the map to the manifold' rule -- projection is
                              near-perfect on a low-rank manifold and FAILS (67% recall) on high-rank atoms,
                              so the rank knee picks the right one.

        `readout='sparsemax'` switches the codebook/recall branches from the softmax blend (which
        over-smooths a continuous manifold) to the sparse Hopfield-Fenchel-Young readout; the default
        'softmax' leaves every path bit-for-bit unchanged.

        `check_manifold=True` (method='spectral' only) first verifies the points form a single connected manifold
        via is_manifold and raises if they do not -- the spectral map's premise -- rather than silently returning
        graph low-pass on a blob. Default False keeps the path overhead-free and backward-compatible.

        A denoiser needs a PRIOR; a single vector with no manifold cannot be cleaned (no free lunch), so
        `samples` (clean rows) or `codebook` (atoms) is required for every method but 'nlm' (which uses
        `x`'s own redundancy). Returns the cleaned vector (or, for 'nlm', the cleaned (N, dim) set).

        KEPT NEGATIVES (the modules', surfaced not hidden): FIXED-rank projection over-smooths at low
        noise -- use 'adaptive', which is ~neutral there; manifold projection only helps where real
        low-rank structure exists (it destroys structureless signal); NLM only helps where near-duplicates
        exist."""
        from holographic.rendering.holographic_denoise import fit_manifold, manifold_denoise, fit_manifold_full, adaptive_manifold_denoise, codebook_denoise, nlm_denoise, pnp_restore, effective_rank, trajectory_denoise
        x = np.asarray(x, float)

        if method == "auto":                          # pick by the prior you handed me, conservatively
            method = "codebook" if codebook is not None else ("adaptive" if samples is not None else None)
            if method is None:
                raise ValueError("denoise needs a prior: pass samples=<clean rows> or codebook=<atoms> "
                                 "(a denoiser is a map of a manifold; a lone vector has none)")

        if method == "nlm":                           # self-similarity: x IS the patch set to clean
            P = np.atleast_2d(x)
            return nlm_denoise(P, k=min(12, len(P)))

        if method == "trajectory":                    # lone 1-D signal: prior built from its OWN windows (SSA)
            return trajectory_denoise(x, window=None, rank=rank)

        if method == "spectral":              # lone scalar FIELD on a known manifold GEOMETRY -> graph-Laplacian map
            if points is None:
                raise ValueError("method='spectral' needs points=<(N, d) coordinates>; x is the field over "
                                 "those N points (the manifold's own geometry IS the prior)")
            pts = np.atleast_2d(np.asarray(points, float))
            if check_manifold:                # opt-in premise check (cheap now PH is fast): the spectral map
                chk = self.is_manifold(pts)   # assumes a smooth field on a CONNECTED manifold; on a blob it is
                if not chk["is_manifold"]:     # only graph low-pass, so refuse loudly unless overridden
                    raise ValueError(
                        f"method='spectral' premise fails: the points are not a single connected manifold "
                        f"(topology={chk['topology']!r}, dense_scales={chk['dense_scales']}). The spectral "
                        f"denoiser would be graph low-pass, not manifold denoising. Pass check_manifold=False "
                        f"to proceed anyway.")
            from holographic.sampling_and_signal.holographic_spectral import SpectralBasis
            sb = SpectralBasis(pts, k=spectral_k, n_basis=spectral_nbasis)
            return sb.denoise(x)

        if method == "codebook":
            if codebook is None:
                raise ValueError("method='codebook' needs codebook=<(n, dim) atoms>")
            return codebook_denoise(x, np.asarray(codebook, float), beta=beta, steps=steps, readout=readout)

        if method == "geometry":              # route by the set's geometry (measured: match map to manifold)
            M = codebook if codebook is not None else samples
            if M is None:
                raise ValueError("method='geometry' needs samples= or codebook= (the manifold/atom set "
                                 "whose geometry decides the map)")
            M = np.atleast_2d(np.asarray(M, float))
            er = effective_rank(M)
            if er <= 0.5 * len(M):            # LOW-rank continuous -> the manifold map: project onto its span
                basis, mean = fit_manifold(M, rank=max(1, er))
                return manifold_denoise(x, basis, mean)
            return codebook_denoise(x, M, beta=beta, steps=steps, readout=readout)   # HIGH-rank discrete -> recall

        if method in ("manifold", "adaptive", "pnp"):
            if samples is None:
                raise ValueError(f"method='{method}' needs samples=<clean rows> to fit the manifold")
            S = np.atleast_2d(np.asarray(samples, float))
            if method == "manifold":
                basis, mean = fit_manifold(S, rank=rank)
                return manifold_denoise(x, basis, mean)
            # 'adaptive' and 'pnp' both want a GENEROUS basis whose coefficients get noise-thresholded
            basis, _, mean = fit_manifold_full(S, rank=min(4 * rank, S.shape[1]))
            if method == "adaptive":
                return adaptive_manifold_denoise(x, basis, mean, sigma=sigma)
            if forward is None or adjoint is None:    # pnp
                raise ValueError("method='pnp' needs forward and adjoint callables (the operator A and A^T)")
            prior = lambda v: adaptive_manifold_denoise(v, basis, mean, sigma=sigma)
            return pnp_restore(x, forward, adjoint, prior, mu=mu, steps=pnp_steps)

        raise ValueError(f"unknown denoise method: {method!r}")



def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p03b_denoise", "_UnifiedPart03B")
    print("holographic_unified_p03b_denoise selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()

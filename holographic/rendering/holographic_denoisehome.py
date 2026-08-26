"""holographic_denoisehome.py -- the DENOISE home (consolidation backlog R5): one place to clean a render or a
signal, routing to the shipped denoisers.

WHY THIS EXISTS
---------------
Denoising is written in several spots: the edge-aware a-trous bilateral for rendered IMAGES (holographic_svgf),
the demodulated variant that divides albedo out first (holographic_modulate), the unsharp/Van-Cittert SHARPEN
(holographic_sharpen), and the manifold-projection denoisers for SIGNALS/vectors (holographic_denoise +
holographic_hopfield -- adaptive/manifold/codebook/nlm/trajectory). A caller shouldn't have to know which module
each lives in.

`Denoise` is that one home, split by what you're cleaning:

    Denoise.image(img, normal, albedo, depth, method='svgf'|'demodulated', ...)   # a rendered frame + its G-buffer
    Denoise.sharpen(x, ...)                                                        # recover detail an over-blur lost
    Denoise.signal(x, samples=..., codebook=..., method='auto', ...)              # a signal/vector onto its manifold

It is THIN -- it ROUTES to the shipped modules (route, don't rewrite). The Milanfar reframe is why these belong
together: a denoiser is a MAP of the manifold clean signals live on, so cleanup and consolidation ARE denoisers.
UnifiedMind.denoise is the full-featured signal dispatcher over the same holographic_denoise family (it adds pnp /
spectral / an auto noise estimate); Denoise.signal is the lighter library entry over the same primitives.
"""
import numpy as np


class Denoise:
    """A namespace of staticmethods over the engine's denoisers. Split by domain: image / sharpen / signal."""

    # --- IMAGE: a rendered frame, denoised using its G-buffer (normal/albedo/depth) as edge-stopping features ---
    @staticmethod
    def image(image, normal, albedo, depth, method="svgf", variance=None, levels=5, **kw):
        """Denoise a rendered `image` (H,W,3) with its G-buffer.
          method='svgf'        : edge-aware a-trous bilateral (holographic_svgf.atrous_bilateral), variance-guided
                                 when a per-pixel variance is supplied. Edges stop on feature-vector cosine.
          method='demodulated' : divide the albedo out, denoise the smooth irradiance, multiply it back
                                 (holographic_modulate.denoise_demodulated) -- best where texture would otherwise
                                 be smeared. Diffuse-clean path.
        """
        if method == "svgf":
            from holographic.rendering.holographic_svgf import atrous_bilateral
            return atrous_bilateral(image, normal, albedo, depth, levels=levels, variance=variance, **kw)
        if method == "demodulated":
            from holographic.misc.holographic_modulate import denoise_demodulated
            return denoise_demodulated(image, normal, albedo, depth, variance=variance, levels=levels, **kw)
        raise ValueError("Denoise.image: unknown method %r (svgf/demodulated)" % method)

    # --- SHARPEN: recover detail an over-smoothing lost (the inverse of a blur) ---
    @staticmethod
    def sharpen(x, blur=None, sigma=3.0, lam=1.0, iters=60, noise_level=0.0):
        """Loop a converging negative-lobe (Van Cittert) sharpen to undo an over-smoothing `blur` (a callable;
        default a Gaussian low-pass with `sigma`). Stops by the discrepancy principle when `noise_level`>0.
        Routes to holographic_sharpen.sharpen_loop. (Kept negative: it can't add detail that was never there;
        too-large `lam` rings.)"""
        from holographic.rendering.holographic_sharpen import sharpen_loop
        return sharpen_loop(np.asarray(x, float), blur=blur, sigma=sigma, lam=lam, iters=iters,
                            noise_level=noise_level)

    # --- SIGNAL / vector: project onto the manifold clean signals live on (Milanfar's reframe) ---
    @staticmethod
    def signal(x, samples=None, codebook=None, method="auto", rank=8, sigma=None, beta=25.0, steps=3,
               readout="softmax"):
        """Denoise a signal/vector by projecting onto its manifold. Routes to holographic_denoise:
          'adaptive'   : low-rank SVD subspace fit from `samples`, then noise-threshold the coefficients
                         (estimates sigma itself -- safe at low noise).
          'manifold'   : plain FIXED-rank projection onto the subspace fit from `samples`.
          'codebook'   : modern-Hopfield cleanup toward a discrete `codebook` manifold.
          'trajectory' : PRIOR-FREE -- the signal's own sliding-window (Hankel/SSA) subspace; needs no examples.
          'auto'       : codebook if a codebook is given, adaptive if samples are given, else trajectory.
        UnifiedMind.denoise is the fuller dispatcher over this same family (adds nlm / pnp / spectral)."""
        from holographic.rendering.holographic_denoise import fit_manifold, manifold_denoise, adaptive_manifold_denoise, codebook_denoise, trajectory_denoise
        x = np.asarray(x, float)
        if method == "auto":
            method = "codebook" if codebook is not None else ("adaptive" if samples is not None else "trajectory")
        if method in ("adaptive", "manifold"):
            if samples is None:
                raise ValueError("Denoise.signal(method=%r) needs `samples` to fit the manifold" % method)
            basis, mean = fit_manifold(samples, rank=rank)
            return (adaptive_manifold_denoise(x, basis, mean, sigma=sigma) if method == "adaptive"
                    else manifold_denoise(x, basis, mean))
        if method == "codebook":
            if codebook is None:
                raise ValueError("Denoise.signal(method='codebook') needs a `codebook`")
            return codebook_denoise(x, codebook, beta=beta, steps=steps, readout=readout)
        if method == "trajectory":
            return trajectory_denoise(x, rank=rank)
        raise ValueError("Denoise.signal: unknown method %r (auto/adaptive/manifold/codebook/trajectory)" % method)


def denoise_backends():
    """The denoising facilities the home exposes (for the catalog / discovery)."""
    return ("image:svgf", "image:demodulated", "sharpen", "signal:adaptive", "signal:manifold",
            "signal:codebook", "signal:trajectory")


def _selftest():
    rng = np.random.default_rng(0)

    # IMAGE svgf: denoise a noisy flat image toward its clean value using a G-buffer; error must drop
    H = W = 24
    clean = np.ones((H, W, 3)) * 0.5
    noisy = clean + 0.1 * rng.standard_normal((H, W, 3))
    normal = np.tile([0., 0., 1.], (H, W, 1))
    albedo = np.ones((H, W, 3)) * 0.5
    depth = np.ones((H, W))
    den = Denoise.image(noisy, normal, albedo, depth, method="svgf", levels=4)
    assert np.abs(den - clean).mean() < np.abs(noisy - clean).mean()          # cleaner than the input
    # svgf via the home == calling svgf directly (bit-identical routing)
    from holographic.rendering.holographic_svgf import atrous_bilateral
    assert np.array_equal(den, atrous_bilateral(noisy, normal, albedo, depth, levels=4, variance=None))

    # SIGNAL trajectory (prior-free): a smooth 1-D signal + noise -> cleaner
    t = np.linspace(0, 4 * np.pi, 256)
    sig = np.sin(t)
    noisy_sig = sig + 0.3 * rng.standard_normal(256)
    clean_sig = Denoise.signal(noisy_sig, method="trajectory", rank=4)
    assert np.abs(clean_sig - sig).mean() < np.abs(noisy_sig - sig).mean()

    # SIGNAL adaptive routes and matches the module for a low-rank vector
    samples = rng.standard_normal((40, 8)) @ rng.standard_normal((8, 32))       # rank-8 in 32-D
    from holographic.rendering.holographic_denoise import fit_manifold, manifold_denoise
    basis, mean = fit_manifold(samples, rank=8)
    x = samples[0] + 0.05 * rng.standard_normal(32)
    assert np.array_equal(Denoise.signal(x, samples=samples, method="manifold", rank=8),
                          manifold_denoise(x, basis, mean))
    print("OK: holographic_denoisehome self-test passed (image svgf cleans + routes bit-identical; signal "
          "trajectory/manifold clean & route; over %d facilities)" % len(denoise_backends()))


if __name__ == "__main__":
    _selftest()

"""A LEARNED energy memory -- training the cleanup's attractors instead of storing them.

WHY THIS EXISTS
---------------
Every cleanup in the engine is a FIXED associative memory. The classical cleanup snaps a noisy
vector to the nearest STORED atom; the modern-Hopfield `dense_cleanup` (holographic_hopfield) -- the
"energy memory" -- relaxes a query against a FIXED codebook V, whose minima ARE the data you stored.
holographic_equilibrium's own docstring says Equilibrium Propagation "is the rule that LEARNS those
attractors", but until now nothing actually trained a cleanup's energy: EP ran as a standalone
classifier and the cleanup stayed fixed. This module makes good on that claim. It uses EP (delegating
to EquilibriumNet -- it does NOT re-implement a learner) to TRAIN an energy whose attractors form a
LEARNED manifold, turning the fixed cleanup into a learned, continuous denoiser: clamp a noisy vector,
let the net relax, read the cleaned output -- a denoising auto-associator whose energy was shaped to
the data rather than handed the data.

The honest question this answers: WHEN does learning the energy beat storing the codebook? The answer
is geometric, and it is the whole point.

MEASURED (honest picture; ring-bump and torus-bump manifolds, isotropic noise)
  * Against the FIXED SOFT energy cleanup it is an unconditional win. On a continuous manifold the
    learned energy beats `dense_cleanup` at every codebook size -- 1-D: ~0.33 vs 0.43-0.51; 2-D: ~0.43
    vs 0.45-0.56 -- because the fixed cleanup returns a softmax MIXTURE that blurs on a continuum while
    the learned net projects. This is the apples-to-apples result: a learned energy memory beats the
    fixed one. Deterministic (seed in -> identical weights out).
  * Against storing DATA (a hard 1-NN codebook of random manifold samples) the win is DIMENSIONAL. On a
    2-D manifold, at MATCHED MEMORY (the EP net's weights vs an equal-byte codebook, K~=49), the learned
    energy wins ~0.43 vs ~0.50 -- because tiling a d-dimensional manifold with samples costs ~grid^d
    points (the curse of dimensionality), while a fixed-size learned projector scales with the
    manifold's intrinsic structure, not its volume. The codebook needs 2-4x more memory to catch up.

KEPT NEGATIVES (loud -- they define exactly where this helps)
  * DISCRETE atoms are the wrong job for it. When queries are noisy versions of a finite set of stored
    atoms, the HARD 1-NN cleanup returns the EXACT atom (rel-error ~0.02 here) and is unbeatable -- a
    learned, approximate energy cannot beat exact recovery. This is B1's "single-item identity is a
    tie", sharpened: against the hard cleanup it is a loss, not a tie. Use the existing cleanup for
    discrete recall; use this for continuous manifolds.
  * In 1-D the curse of dimensionality does not bite, so a matched-memory codebook beats the learned
    energy (1-D, K=32 random samples ~0.27 vs EP ~0.33). The learned energy's advantage over storing
    data REQUIRES manifold dimension >= 2. Said plainly: in 1-D, just store the samples.
  * The win over a codebook is at MATCHED memory, not unbounded -- give the codebook several times more
    bytes and it wins even in 2-D. And EP inherits its weakness at very high output dimension, so this
    targets moderate D with a low intrinsic-dimensional manifold, not arbitrary high-D fields.

DESIGN NOTES
  * Pure delegation to EquilibriumNet; pure NumPy; deterministic. Patterns and the cleaned output live
    in [0,1] (EP's hard-sigmoid output range); the caller normalises arbitrary data into that box.
  * A learned-energy cleanup is the natural prior for a Plug-and-Play / RED loop (Milanfar): a learned
    map of a continuous manifold, exactly what that framework asks for in place of a fixed codebook.
"""

import numpy as np
from holographic.simulation_and_physics.holographic_equilibrium import EquilibriumNet


def torus_bump_manifold(n_grid=6, latent_dim=2, sigma=0.13, n_samples=2000, noise=0.30, seed=0):
    """Sample a continuous NONLINEAR manifold: a Gaussian bump at a continuous position on a
    `latent_dim`-D torus, embedded in D = n_grid**latent_dim dims. Returns (clean, noisy, D). The
    manifold is curved (not low-rank, so SVD/consolidation cannot denoise it) and a finite codebook can
    only QUANTIZE its continuous position -- which is exactly why the selftest uses it to separate
    learning from storing. WHY here: gives the selftest a known continuous manifold with no external
    data dependency, the way lorenz_trajectory does for the chaos module.
    """
    rng = np.random.default_rng(seed)
    g = np.arange(n_grid) / n_grid

    def bump(p):
        # product of per-axis ring-Gaussians -> a bump on the latent_dim-torus, flattened.
        axes = []
        for a in range(latent_dim):
            d = np.minimum(np.abs(g - p[a]), 1.0 - np.abs(g - p[a]))
            axes.append(np.exp(-(d ** 2) / (2.0 * sigma ** 2)))
        out = axes[0]
        for a in range(1, latent_dim):
            out = out[..., None] * axes[a]
        return out.ravel()

    P = rng.uniform(0.0, 1.0, (n_samples, latent_dim))
    clean = np.stack([bump(p) for p in P])
    noisy = np.clip(clean + noise * rng.standard_normal(clean.shape), 0.0, 1.0)
    return clean, noisy, clean.shape[1]


class LearnedEnergyMemory:
    """A denoising associative memory whose attractors are LEARNED by Equilibrium Propagation, rather
    than a fixed codebook. `cleanup(x)` clamps x, relaxes the EP net, and returns the cleaned vector --
    the projection onto the learned manifold. The learned companion to the fixed modern-Hopfield
    `dense_cleanup`: use that for discrete-atom recall, this for a continuous manifold.

    SCOPE (measured, kept negative): this is a DENOISER, not a GENERATOR. It earns its place where a noisy
    point already sits NEAR the manifold, beating a matched-byte codebook (the selftest's relative win on a
    2-D bump manifold) -- but the absolute fidelity is modest. It does NOT generate: descending the energy from
    pure noise, or even 'cleaning' an already-clean point on a low-dimensional smooth manifold, COLLAPSES and
    distorts, because the free-state attractors are not reliably ON the manifold (so they cannot be sampled to
    produce novel valid points). For GENERATION use the composed-manifold diffusion
    (holographic_hopfield.generate_structure), or -- on a locally-convex manifold -- simple interpolation in the
    composed/parameter space, which already lands on the manifold (the SVG morph does exactly this). A learned
    GENERATIVE manifold would need a denoiser trained ACROSS noise levels (a real diffusion model); this
    single-noise EP autoencoder is not one. See backlog VG-2.
    """

    def __init__(self, net):
        self.net = net                    # the trained EquilibriumNet -- the delegated learner

    @classmethod
    def learn(cls, patterns, noise=0.30, n_hidden=None, epochs=80, beta=0.5,
              dt=0.4, t_free=30, t_nudge=10, lr=0.3, batch=128, seed=0):
        """Train the energy on manifold samples. `patterns` (rows, values in [0,1]) are clean samples of
        the manifold; we form (sample + noise -> sample) pairs so the net learns to project noisy points
        back onto the manifold. The hidden layer is a bottleneck (default ~ D/2) that forces the energy's
        attractor set onto the low-dimensional manifold rather than memorising points. Deterministic.
        """
        X = np.clip(np.asarray(patterns, float), 0.0, 1.0)
        d = X.shape[1]
        h = n_hidden if n_hidden is not None else max(8, d // 2)
        rng = np.random.default_rng(seed)
        # The auto-associative training set: corrupt each pattern, ask the net to restore it. Drawing
        # fresh noise per pattern (not reusing the caller's) makes the energy robust across the noise ball.
        noisy = np.clip(X + noise * rng.standard_normal(X.shape), 0.0, 1.0)
        net = EquilibriumNet(n_in=d, n_hidden=h, n_out=d, beta=beta, dt=dt,
                             t_free=t_free, t_nudge=t_nudge, seed=seed)
        net.fit(noisy, X, epochs=epochs, lr=lr, batch=batch, seed=seed)
        return cls(net)

    def cleanup(self, x):
        """Denoise x onto the learned manifold (the free-phase output of the relaxed net). Accepts a
        single vector or a batch; returns the same shape."""
        X = np.clip(np.atleast_2d(np.asarray(x, float)), 0.0, 1.0)
        cleaned = self.net.free_state(X)[1]          # (batch, D) free-phase output
        return cleaned[0] if np.ndim(x) == 1 else cleaned


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / (np.linalg.norm(b) + 1e-12))


def _selftest():
    """CI-fast: prove (1) the learned energy beats the fixed SOFT energy cleanup on a continuous
    manifold, (2) it beats a matched-memory codebook of random samples on a 2-D manifold (the curse of
    dimensionality), (3) it is deterministic -- and keep the negative honest: on discrete atoms the hard
    1-NN cleanup wins by exact recovery, which the learned energy cannot beat."""
    from holographic.agents_and_reasoning.holographic_hopfield import dense_cleanup

    def hard(q, cb):
        return cb[int(np.argmax(cb @ q))]            # 1-NN, returns the exact stored atom

    rng = np.random.default_rng(0)

    # ---- (1)+(2) continuous 2-D torus manifold ----
    n_grid, latent, sigma, noise = 6, 2, 0.13, 0.30
    clean_tr, noisy_tr, D = torus_bump_manifold(n_grid, latent, sigma, n_samples=2000, noise=noise, seed=0)
    mem = LearnedEnergyMemory.learn(clean_tr, noise=noise, n_hidden=24, epochs=80, beta=0.5, seed=0)

    clean_te, noisy_te, _ = torus_bump_manifold(n_grid, latent, sigma, n_samples=300, noise=noise, seed=7)
    ep_err = np.mean([_rel(mem.cleanup(noisy_te[i]), clean_te[i]) for i in range(len(noisy_te))])

    # matched-memory baseline: the EP net's weight count vs an equal-byte codebook of RANDOM samples.
    ep_weights = 2 * D * 24
    K = max(2, round(ep_weights / D))                # ~49 atoms == matched memory
    g = np.arange(n_grid) / n_grid

    def bump(p):
        dx = np.minimum(np.abs(g - p[0]), 1 - np.abs(g - p[0]))
        dy = np.minimum(np.abs(g - p[1]), 1 - np.abs(g - p[1]))
        return np.exp(-(dx[:, None] ** 2 + dy[None, :] ** 2) / (2 * sigma ** 2)).ravel()

    Pc = rng.uniform(0, 1, (K, latent))
    cb = np.stack([bump(p) for p in Pc])
    hard_err = np.mean([_rel(hard(noisy_te[i], cb), clean_te[i]) for i in range(len(noisy_te))])
    soft_err = np.mean([_rel(dense_cleanup(noisy_te[i], cb, beta=25.0, steps=3), clean_te[i]) for i in range(len(noisy_te))])

    assert ep_err < soft_err, f"learned energy should beat the soft energy cleanup: ep={ep_err:.3f} soft={soft_err:.3f}"
    assert ep_err < hard_err, f"learned energy should beat a matched-memory codebook on a 2-D manifold: ep={ep_err:.3f} hard(K={K})={hard_err:.3f}"

    # determinism
    mem2 = LearnedEnergyMemory.learn(clean_tr, noise=noise, n_hidden=24, epochs=80, beta=0.5, seed=0)
    assert np.allclose(mem.net.Who, mem2.net.Who), "training must be deterministic for a fixed seed"

    # ---- (3) kept negative: discrete atoms -> hard 1-NN exact recovery is unbeatable ----
    M = 8
    atoms = np.stack([bump(p) for p in rng.uniform(0, 1, (M, latent))])
    didx = rng.integers(0, M, 300)
    clean_d = atoms[didx]
    noisy_d = np.clip(clean_d + noise * rng.standard_normal(clean_d.shape), 0, 1)
    tix = rng.integers(0, M, 1500)
    mem_d = LearnedEnergyMemory.learn(atoms[tix], noise=noise, n_hidden=24, epochs=60, beta=0.5, seed=0)
    ep_d = np.mean([_rel(mem_d.cleanup(noisy_d[i]), clean_d[i]) for i in range(len(noisy_d))])
    hard_d = np.mean([_rel(hard(noisy_d[i], atoms), clean_d[i]) for i in range(len(noisy_d))])
    assert hard_d < 0.1, f"hard 1-NN should recover discrete atoms near-exactly, got {hard_d:.3f}"
    assert hard_d < ep_d, f"on discrete atoms the fixed hard cleanup must win (kept negative): hard={hard_d:.3f} ep={ep_d:.3f}"

    print("holographic_energy selftest OK")
    print(f"  continuous 2-D manifold: learned energy {ep_err:.3f}  vs  soft cleanup {soft_err:.3f}  vs  "
          f"matched-memory hard-1NN(K={K}) {hard_err:.3f}  -> learned wins")
    print(f"  discrete atoms (kept negative): hard-1NN exact {hard_d:.3f}  <  learned energy {ep_d:.3f}  -> fixed wins")


if __name__ == "__main__":
    _selftest()

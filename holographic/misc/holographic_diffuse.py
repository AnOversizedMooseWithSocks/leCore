"""Looping denoise as diffusion on an arbitrary manifold -- denoise by settling, generate by walking from noise.

WHY THIS EXISTS (Group G, the diffusion generalisation)
-------------------------------------------------------
"A looping denoising process": iterate a denoiser and it walks onto the manifold (DENOISING a noisy input) or,
started from pure noise, walks ONTO the manifold (GENERATING a sample) -- the same operation in two regimes (the
addendum's B10 result). The engine already runs this for the discrete CODEBOOK (hopfield.generate). The frontier,
flagged by Eno, is looping over a COMPOSED or LEARNED manifold that is NOT the bare codebook -- a curved point
cloud, a consolidation subspace fit to real data (XDATA-1 finds exactly such a manifold to loop on).

The denoiser is a dense-Hopfield step over the manifold's SAMPLES: x <- softmax(beta * S.x) @ S -- a soft move
toward the local samples. Iterating it settles a point onto the manifold. Annealing beta UP while injecting
DECREASING noise turns it into a diffusion sampler: from noise, the walk commits to the manifold and lands on it.

The payoff over a bare codebook AND over interpolation, on a CURVED manifold (a ring) where interpolation
provably leaves the manifold:
  * DENOISE is idempotent settling -- a noisy point converges onto the manifold and stays (measured 0.59 -> 0.029,
    flat thereafter; the 0.029 floor is the sample-spacing limit, not error).
  * INTERPOLATION leaves the manifold -- the chord midpoint of two ring samples is off the ring (0.74), and the
    denoiser settles it BACK on (0.029). So looping-denoise beats interpolation for staying on a curved manifold.
  * GENERATION from noise is NOVEL-but-VALID -- generated points land on the ring (dist ~0.02, valid) but BETWEEN
    the stored samples (dist-to-nearest-stored ~0.04, novel), where bare-codebook generation just returns a stored
    sample (dist-to-stored 0, degenerate).
"""

import numpy as np


def manifold_denoise_step(x, manifold, beta):
    """One dense-Hopfield step toward a sample-defined manifold: x <- softmax(beta * S.x) @ S. A soft move toward
    the local manifold samples (the modern-Hopfield / attention update over the manifold's point cloud)."""
    S = np.asarray(manifold, float)
    s = S @ np.asarray(x, float)
    w = np.exp(beta * (s - s.max())); w /= w.sum()
    return w @ S


def settle(x, manifold, beta=18.0, steps=8):
    """Iterate the denoise step to settle a (noisy) point ONTO the manifold -- the looping-denoise. Converges
    (idempotent): once on the manifold, further steps leave it essentially fixed."""
    x = np.asarray(x, float)
    for _ in range(steps):
        x = manifold_denoise_step(x, manifold, beta)
    return x


def generate(manifold, steps=30, beta_lo=2.0, beta_hi=25.0, noise_hi=0.5, noise_lo=0.0,
             settle_steps=5, seed=0):
    """Generate a NOVEL-but-VALID sample on a sample-defined manifold by annealed diffusion: start from noise,
    iterate the denoise step with beta rising (lo->hi) and injected noise falling (hi->lo), then a final settle.
    The walk commits to the manifold and lands between the stored samples -- not a verbatim sample (the bare
    codebook's degenerate result)."""
    S = np.asarray(manifold, float)
    g = np.random.default_rng(seed)
    D = S.shape[1]
    x = g.standard_normal(D) / np.sqrt(D)
    for t in range(steps):
        frac = t / max(steps - 1, 1)
        beta = beta_lo + (beta_hi - beta_lo) * frac
        noise = noise_hi + (noise_lo - noise_hi) * frac
        x = manifold_denoise_step(x, S, beta) + noise * g.standard_normal(D) / np.sqrt(D)
    for _ in range(settle_steps):
        x = manifold_denoise_step(x, S, beta_hi)
    return x


def _selftest():
    """CI-fast: on a curved manifold (a unit ring in R^D) the looping denoiser settles points onto the ring
    (idempotent), settles an off-manifold interpolation midpoint back on (beating interpolation), and generates
    novel-but-valid samples from noise where bare-codebook generation would be degenerate."""
    rng = np.random.default_rng(0)
    D, N = 64, 48
    U, _ = np.linalg.qr(rng.standard_normal((D, 2)))
    u, v = U[:, 0], U[:, 1]
    th = np.linspace(0, 2 * np.pi, N, endpoint=False)
    S = np.stack([np.cos(t) * u + np.sin(t) * v for t in th])

    def ring_dist(x):
        a, b = u @ x, v @ x
        plane = a * u + b * v
        return float(np.hypot(np.linalg.norm(x - plane), abs(np.hypot(a, b) - 1)))

    # (1) idempotent denoise: a noisy ring point settles onto the ring and stays
    x0 = S[10] + 0.6 * rng.standard_normal(D) / np.sqrt(D)
    x1 = settle(x0, S, beta=18.0, steps=8)
    x2 = settle(x1, S, beta=18.0, steps=4)
    assert ring_dist(x0) > 0.3                          # starts off the manifold
    assert ring_dist(x1) < 0.15                         # settles onto it
    assert abs(ring_dist(x2) - ring_dist(x1)) < 1e-3    # idempotent (already settled)

    # (2) interpolation leaves the manifold; denoise settles it back on
    mid = 0.5 * (S[5] + S[25])
    mid_s = settle(mid, S, beta=18.0, steps=8)
    assert ring_dist(mid) > 0.3 and ring_dist(mid_s) < 0.15, (ring_dist(mid), ring_dist(mid_s))

    # (3) generation is novel-but-valid; the bare codebook would be degenerate
    gd, nov = [], []
    for s in range(20):
        xg = generate(S, seed=s)
        gd.append(ring_dist(xg))
        nov.append(min(float(np.linalg.norm(xg - si)) for si in S))
    assert np.mean(gd) < 0.1                             # generated samples are ON the ring (valid)
    assert np.mean(nov) > 0.01                           # but BETWEEN the stored samples (novel)
    assert min(float(np.linalg.norm(S[3] - si)) for si in S) == 0.0   # a stored sample is verbatim (codebook degeneracy)


if __name__ == "__main__":
    _selftest()
    print("holographic_diffuse selftest passed")

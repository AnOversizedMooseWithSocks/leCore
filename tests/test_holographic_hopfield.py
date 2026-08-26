"""Modern Hopfield cleanup (B1) + generation by denoising (B10)."""
import numpy as np
from holographic.agents_and_reasoning.holographic_hopfield import dense_cleanup, HopfieldCleanup, generate
from holographic.agents_and_reasoning.holographic_ai import Vocabulary, random_vector


def _codebook(dim=256, n=64, seed=1):
    voc = Vocabulary(dim, seed=seed)
    for i in range(n):
        voc.get(f"a{i}")
    return voc._matrix()[1]


def test_high_beta_reproduces_hard_nearest_neighbour():
    # backward-compatibility: at high beta the energy cleanup makes the SAME identity decision as
    # holostuff's argmax NN -- a strict superset, no behaviour change for the classification path.
    rng = np.random.default_rng(0)
    V = _codebook()
    hop = HopfieldCleanup(beta=80.0, steps=2).fit(V)
    for _ in range(200):
        i = rng.integers(len(V))
        q = V[i] + 0.8 * rng.standard_normal(V.shape[1]) / np.sqrt(V.shape[1])
        nn = int((V @ q).argmax())            # holostuff's hard cleanup decision
        assert hop.cleanup(q)[0] == nn        # energy cleanup agrees exactly


def test_denoise_cleans_a_corrupted_vector():
    # the real win: a badly-recovered vector is cleaned back onto the manifold (cosine -> ~1).
    # averaged over trials (a single high-noise draw can occasionally miss; the mean is what we
    # measured and ship).
    rng = np.random.default_rng(1)
    V = _codebook()
    dim = V.shape[1]
    raw = clean = 0.0
    trials = 40
    for _ in range(trials):
        i = rng.integers(len(V))
        noisy = V[i] + 1.5 * rng.standard_normal(dim) / np.sqrt(dim)
        raw += float(V[i] @ noisy / (np.linalg.norm(noisy) + 1e-12))
        z = dense_cleanup(noisy, V, beta=25.0, steps=3)
        clean += float(V[i] @ z / (np.linalg.norm(z) + 1e-12))
    raw /= trials; clean /= trials
    assert clean > 0.95 and clean > raw + 0.2


def test_generation_by_denoising_reaches_the_manifold():
    # iterating the denoiser from pure noise walks onto the codebook manifold = generation.
    V = _codebook()
    for seed in range(3):
        z = generate(V, steps=12, seed=seed)
        assert float((V @ z).max()) > 0.9    # ended on (near) a stored pattern

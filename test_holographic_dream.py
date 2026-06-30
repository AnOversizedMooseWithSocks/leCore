"""Tests for consolidation + dreaming: nystrom subspace + generative replay (DREAM-1)."""
import numpy as np
from holographic_dream import dream_subspace, subspace_alignment, dream, on_manifold


def _lowrank_memory(D=256, k=8, N=500, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((k, D)); mem = rng.standard_normal((N, k)) @ B + 0.02 * rng.standard_normal((N, D))
    return mem / np.linalg.norm(mem, axis=1, keepdims=True), B


def test_nystrom_subspace_aligns_to_full():
    mem, _ = _lowrank_memory()
    full, _ = dream_subspace(mem, k=8)
    lm, _ = dream_subspace(mem, k=8, landmarks=64)
    assert subspace_alignment(full, lm) > 0.9                      # landmark subspace ~ full (large-memory sketch)


def test_dreamed_samples_are_valid_and_novel():
    mem, _ = _lowrank_memory()
    basis, mean = dream_subspace(mem, k=8)
    samples = dream(basis, mean, n=16, seed=1)
    val = np.mean([on_manifold(s, basis, mean) for s in samples])
    nov = np.mean([1.0 - max(abs(float(s @ m)) for m in mem) for s in samples])
    assert val > 0.9                                               # on the manifold (valid)
    assert nov > 0.05                                              # not a verbatim stored item (novel)


def test_nystrom_subspace_degrades_on_full_rank():
    rng = np.random.default_rng(0)
    mem = rng.standard_normal((500, 256)); mem /= np.linalg.norm(mem, axis=1, keepdims=True)  # full-rank
    full, _ = dream_subspace(mem, k=8)
    lm, _ = dream_subspace(mem, k=8, landmarks=40)
    assert subspace_alignment(full, lm) < 0.8                      # kept negative: no low-rank structure to sketch

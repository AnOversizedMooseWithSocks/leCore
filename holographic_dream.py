"""Consolidation + dreaming (DREAM-1): the memory's low-rank manifold, approximated cheaply with Nystrom for a
LARGE store, and 'dreaming' = generative replay over that manifold.

THE TIE-IN (the request): consolidation already finds the low-rank SUBSPACE real states live on (an SVD); the
B10 result already showed that iterating the denoiser FROM NOISE generates structure. This wires them together
and adds the two asks:
  * NYSTROM for large memory: compute the consolidation subspace from m farthest-point LANDMARK memories instead
    of all N -- O(m D k) not O(N D k) -- the sketch that keeps it affordable as the store grows.
  * DREAMING = generative replay over the CONSOLIDATED subspace (not the bare codebook): draw noise, project onto
    the subspace (the manifold denoiser run from noise), optionally clean -> a sample that is ON the manifold
    (valid) yet NOVEL (not a verbatim stored item). Over the composed/continuous consolidated manifold this
    produces novel COMPOSITIONS, which is exactly the regime B10 flagged as the interesting one (bare-codebook
    generation just returns stored atoms).

HONEST: the Nystrom subspace is exact only when the memory is genuinely LOW-RANK (the case consolidation is for);
on full-rank noise a landmark subset misses directions. Dreaming generates only within the SPAN of what was
consolidated -- it recombines, it does not invent outside the manifold. Both measured, negatives kept.
"""

import numpy as np
from holographic_nystrom import farthest_point_landmarks


def dream_subspace(memories, k=8, landmarks=None, seed=0):
    """The consolidated low-rank SUBSPACE of stored memories (top-k principal directions) + the mean. With
    `landmarks`=m, compute it from m farthest-point-sampled memories instead of all N (the Nystrom/sketch
    approximation for a LARGE store); else use all N. Returns (basis (k, D), mean (D,))."""
    X = np.asarray(memories, float)
    if landmarks is not None and landmarks < len(X):
        X = X[farthest_point_landmarks(X, landmarks, seed=seed)]      # a covering subset, not a random one
    mean = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return Vt[:k], mean


def subspace_alignment(A, B):
    """Mean cosine of the principal angles between two subspaces (rows = basis vectors); 1.0 = identical span.
    How well the Nystrom (landmark) subspace matches the full one."""
    Qa = np.linalg.qr(np.asarray(A, float).T)[0]
    Qb = np.linalg.qr(np.asarray(B, float).T)[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.mean(np.clip(s, 0.0, 1.0)))


def dream(basis, mean, n=8, seed=0, noise=1.0, codebook=None, beta=25.0):
    """DREAM = generative replay: draw noise, PROJECT onto the consolidated subspace (the manifold denoiser run
    from noise), optionally clean toward a codebook -> samples that are ON the manifold (valid) yet NOVEL. Over
    the consolidated subspace (composed/continuous), this produces novel COMPOSITIONS, not stored atoms. Returns
    an (n, D) array of unit-norm samples."""
    rng = np.random.default_rng(seed)
    basis = np.asarray(basis, float); mean = np.asarray(mean, float)
    out = []
    for _ in range(n):
        z = mean + noise * rng.standard_normal(mean.size)
        proj = mean + basis.T @ (basis @ (z - mean))                 # project onto the subspace (denoise-from-noise)
        if codebook is not None:
            from holographic_hopfield import dense_cleanup
            proj = dense_cleanup(proj, codebook, beta=beta, steps=2)
        out.append(proj / (np.linalg.norm(proj) + 1e-12))
    return np.array(out)


def on_manifold(sample, basis, mean):
    """How much of a sample lies IN the subspace: ||proj(sample)|| / ||sample|| (centred). 1.0 = fully on-manifold."""
    basis = np.asarray(basis, float); mean = np.asarray(mean, float)
    c = np.asarray(sample, float) - mean
    p = basis.T @ (basis @ c)
    return float(np.linalg.norm(p) / (np.linalg.norm(c) + 1e-12))


def _selftest():
    rng = np.random.default_rng(0)
    D, k, N = 256, 8, 600
    B = rng.standard_normal((k, D))                                  # the true rank-k manifold
    coeffs = rng.standard_normal((N, k))
    mem = coeffs @ B + 0.02 * rng.standard_normal((N, D))           # low-rank memories + a little noise
    mem /= np.linalg.norm(mem, axis=1, keepdims=True)
    full_basis, mean = dream_subspace(mem, k=k)
    lm_basis, _ = dream_subspace(mem, k=k, landmarks=64)            # Nystrom: from 64 landmarks
    align = subspace_alignment(full_basis, lm_basis)
    assert align > 0.9, align                                       # landmark subspace ~ full subspace
    samples = dream(full_basis, mean, n=16, seed=1)
    val = np.mean([on_manifold(s, full_basis, mean) for s in samples])
    nov = np.mean([1.0 - max(abs(float(s @ m)) for m in mem) for s in samples])   # novelty vs every stored item
    assert val > 0.9 and nov > 0.1, (val, nov)                     # on-manifold (valid) yet not verbatim (novel)
    print(f"dream selftest ok: Nystrom subspace aligns {align:.3f} to full; dreamed samples on-manifold {val:.2f}, "
          f"novelty {nov:.2f} (valid yet not a stored item)")


if __name__ == "__main__":
    _selftest()

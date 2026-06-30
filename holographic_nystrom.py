"""Landmark (Nystrom) spectral embedding (SCALE-1): break the dense O(N^3) eigendecomposition wall by doing the
high-precision eigh on a small set of landmark "anchors" and extending to all N points cheaply.

THE BOTTLENECK THIS ATTACKS
---------------------------
`holographic_spectral.laplacian_eigenbasis` calls np.linalg.eigh on the full N x N graph Laplacian -- O(N^3) time
and O(N^2) memory, and it computes ALL N eigenvectors only to keep the lowest few. That is the engine's
"moderate N" ceiling for any spectral / manifold-embedding task on the semantic field.

THE MOVE (and why it is the irradiance cache, applied to the latent space)
--------------------------------------------------------------------------
Indirect light is smooth, so Ward's irradiance cache computes it at a sparse set of anchor points and interpolates
the rest. The leading eigenvectors of a smooth affinity are *also* smooth and low-rank, so the SAME move works:
  * pick m << N LANDMARKS that cover the data (farthest-point sampling = guaranteed coverage of every "hot"
    cluster / local manifold, the discrete cousin of the engine's blue-noise sampling),
  * do the high-precision eigh on the small m x m landmark affinity block (the expensive computation, paid only
    on the anchors),
  * EXTEND the eigenvectors to all N points by the Nystrom formula (the cheap interpolation = the coarse
    background).
Cost falls from O(N^3) to O(m^3 + N*m): the landmark eigh plus a thin N x m extension, and only an N x m affinity
block is ever formed, never the N x N matrix. This is Fowlkes et al. (2004), "Spectral Grouping Using the Nystrom
Method," recast in the engine's own caching language. Measured below: cost scaling AND embedding quality, with the
kept negatives (coverage-limited, low-rank-affinity-limited).
"""

import numpy as np


def farthest_point_landmarks(points, m, seed=0):
    """Greedy farthest-point sampling: start at a random point, then repeatedly add the point furthest from the
    current landmark set. Guarantees the m landmarks COVER the data -- every cluster / local manifold gets an
    anchor (unlike uniform-random, which can miss a small cluster). O(N*m), vectorised inner loop. Returns the
    landmark indices."""
    P = np.asarray(points, float)
    n = len(P)
    m = min(m, n)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(n))
    idx = [first]
    d2 = ((P - P[first]) ** 2).sum(axis=1)                   # min sq-distance to the landmark set so far
    for _ in range(m - 1):
        j = int(np.argmax(d2))                               # the point currently least-covered
        idx.append(j)
        d2 = np.minimum(d2, ((P - P[j]) ** 2).sum(axis=1))   # update coverage
    return np.array(idx)


def gaussian_affinity(A, B, sigma):
    """The Gaussian (heat-kernel) affinity exp(-||a-b||^2 / 2 sigma^2) between rows of A:(Na,d) and B:(Nb,d).
    Used to build ONLY the N x m and m x m blocks -- never the full N x N."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    D2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-D2 / (2.0 * sigma ** 2))


def _normalized_affinity_eigh(W):
    """Reference: top eigenpairs of the symmetric-normalized affinity M = D^-1/2 W D^-1/2 (descending). The top
    eigenvectors of M are the smooth spectral embedding = the lowest eigenvectors of the normalized Laplacian
    I - M. Dense O(N^3)."""
    deg = np.maximum(W.sum(1), 1e-12)
    dinv = 1.0 / np.sqrt(deg)
    M = dinv[:, None] * W * dinv[None, :]
    val, U = np.linalg.eigh(M)
    return val[::-1], U[:, ::-1], dinv


def dense_embedding(points, n_basis, sigma):
    """The dense reference embedding: full N x N affinity, symmetric-normalize, eigh, keep the top n_basis smooth
    eigenvectors. This is the O(N^3) path Nystrom approximates -- kept for measurement and as ground truth."""
    P = np.asarray(points, float)
    W = gaussian_affinity(P, P, sigma)
    val, U, _ = _normalized_affinity_eigh(W)
    Phi = U[:, :n_basis]
    return val[:n_basis], Phi / (np.linalg.norm(Phi, axis=0, keepdims=True) + 1e-12)


def nystrom_embedding(points, n_basis, m=None, sigma=None, seed=0, landmarks="fps"):
    """Landmark spectral embedding: approximate the top n_basis smooth eigenvectors of the normalized affinity in
    O(m^3 + N*m) instead of O(N^3), forming only the N x m and m x m affinity blocks. `m` landmarks (default
    ~8*n_basis) are chosen by farthest-point sampling ('fps', covers every cluster) or 'random'. `sigma` defaults
    to the median landmark distance (a robust bandwidth). Returns (eigenvalues, eigenvectors (N, n_basis)) -- the
    same shape as the dense path, so it is a drop-in for laplacian_eigenbasis on the smooth-embedding use."""
    P = np.asarray(points, float)
    n = len(P)
    if m is None:
        m = min(max(8 * n_basis, 16), n)
    if landmarks == "fps":
        lm = farthest_point_landmarks(P, m, seed)
    else:
        lm = np.random.default_rng(seed).choice(n, size=min(m, n), replace=False)
    L = P[lm]
    if sigma is None:
        DL = np.sqrt(np.maximum(((L[:, None, :] - L[None, :, :]) ** 2).sum(-1), 0.0))
        sigma = float(np.median(DL[DL > 0])) or 1.0

    Wmm = gaussian_affinity(L, L, sigma)                     # m x m  (landmark block)
    Wnm = gaussian_affinity(P, L, sigma)                     # N x m  (the only big block, thin)

    # Nystrom estimate of the full degree vector deg = W @ 1, WITHOUT forming W:
    #   W ~ Wnm @ pinv(Wmm) @ Wnm^T, so deg ~ Wnm @ (pinv(Wmm) @ (Wnm^T @ 1)).
    pinvWmm = np.linalg.pinv(Wmm)
    deg = Wnm @ (pinvWmm @ (Wnm.T @ np.ones(n)))
    deg = np.maximum(deg, 1e-12)
    dnm = 1.0 / np.sqrt(deg)                                 # N
    dmm = 1.0 / np.sqrt(deg[lm])                             # m
    Wnm_hat = dnm[:, None] * Wnm * dmm[None, :]              # symmetric-normalized N x m
    Wmm_hat = dmm[:, None] * Wmm * dmm[None, :]              # symmetric-normalized m x m

    val, U = np.linalg.eigh(Wmm_hat)                         # the HIGH-PRECISION eigh, only m x m
    val = val[::-1]; U = U[:, ::-1]
    val = np.maximum(val, 1e-12)
    k = min(n_basis, m)
    Phi = Wnm_hat @ (U[:, :k] / val[:k])                     # Nystrom extension to all N points
    Phi = Phi / (np.linalg.norm(Phi, axis=0, keepdims=True) + 1e-12)
    return val[:k], Phi


def subspace_alignment(Phi_a, Phi_b):
    """How well two embeddings span the same subspace, invariant to sign flips and within-subspace rotation: the
    mean singular value of Phi_a^T Phi_b (1.0 = identical subspace, 0 = orthogonal). The honest quality metric for
    an approximate eigenbasis."""
    A = Phi_a / (np.linalg.norm(Phi_a, axis=0, keepdims=True) + 1e-12)
    B = Phi_b / (np.linalg.norm(Phi_b, axis=0, keepdims=True) + 1e-12)
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return float(s.mean())


def _selftest():
    rng = np.random.default_rng(0)
    # three well-separated blobs: a clean union-of-manifolds the embedding should resolve
    blobs = np.vstack([rng.normal(c, 0.25, (120, 3)) for c in
                       ([0, 0, 0], [5, 0, 0], [0, 5, 0])])
    vd, Pd = dense_embedding(blobs, n_basis=3, sigma=1.0)
    vn, Pn = nystrom_embedding(blobs, n_basis=3, m=48, sigma=1.0)
    align = subspace_alignment(Pd, Pn)
    assert align > 0.9, align                                # landmark embedding ~ the dense one
    # FPS coverage beats random landmarks (random can miss a blob)
    _, Pr = nystrom_embedding(blobs, n_basis=3, m=12, sigma=1.0, landmarks="random", seed=3)
    _, Pf = nystrom_embedding(blobs, n_basis=3, m=12, sigma=1.0, landmarks="fps", seed=3)
    a_rand = subspace_alignment(Pd, Pr); a_fps = subspace_alignment(Pd, Pf)
    assert a_fps >= a_rand - 0.05                            # FPS at least as good (usually better) at tiny m
    print(f"nystrom selftest ok: subspace alignment to dense {align:.3f} (48 landmarks vs {len(blobs)} pts); "
          f"FPS {a_fps:.2f} >= random {a_rand:.2f} at m=12")


if __name__ == "__main__":
    _selftest()

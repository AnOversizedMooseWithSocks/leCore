"""Tensor-product binding and its tensor-train (MPS) truncation -- the uncompressed cousins of HRR's
circular-convolution bind, for the capacity comparison the tensor-networks seat (Stoudenmire) asked for.

HRR's bind(a, b) = circular convolution is a *compressed projection* of Smolensky's tensor-product binding
a (X) b. This module keeps the uncompressed tensor product (an outer product) and the tensor-train view
(a low-rank, 2-site matrix-product truncation of it), so all three points on the rank spectrum can be
measured against each other:

    HRR (rank-1-ish, D numbers)  <  tensor train (rank r, ~2rD numbers)  <  full tensor product (D^2 numbers)

The honest finding the comparison surfaces (see test_integration): at a fixed LOAD the tensor product recalls
far more accurately than HRR (it spends D x the storage), and with ORTHOGONAL keys it is EXACT up to M = D,
which circular convolution is not; an MPS truncation losslessly compresses a low-rank binding matrix. But on
the capacity-per-STORED-NUMBER frontier HRR gives up nothing, and a generic (full-rank) binding cannot be
MPS-compressed without losing recall -- so the tensor route is a different storage/fidelity tradeoff point,
not a free improvement over HRR.
"""

import numpy as np


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


class TensorBindMemory:
    """A heteroassociative memory built from TENSOR-PRODUCT binding (the outer product), optionally truncated
    to a rank-r 2-site tensor train (MPS).

    The bundle is the (D x D) matrix  M = sum_i outer(key_i, value_i); recall(key) = normalise(M^T key) =
    normalise(sum_i (key_i . key) value_i). Crosstalk is suppressed by the key inner products (~1/sqrt(D)),
    which is why this recalls far better than HRR at a fixed load -- at the cost of D^2 rather than D numbers.

    With `rank=r`, M is replaced by its truncated SVD  M ~ A @ B  with A (D x r), B (r x D) -- a two-core
    matrix-product (tensor-train) state with bond dimension r. This is LOSSLESS exactly when rank(M) <= r
    (a low-entanglement / low-rank binding), and lossy otherwise. `n_numbers` reports the honest storage."""

    def __init__(self, keys, values, rank=None):
        K = np.asarray(keys, float)
        V = np.asarray(values, float)
        M = K.T @ V                                          # sum_i outer(k_i, v_i), shape (D, D)
        self.D = int(M.shape[0])
        self.rank = rank
        if rank is None:
            self.M = M
            self.factors = None
            self.n_numbers = int(M.size)                     # D * D
        else:
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            r = int(min(rank, S.size))
            self.factors = ((U[:, :r] * S[:r]), Vt[:r, :])   # A (D x r), B (r x D); M ~ A @ B
            self.M = None
            self.n_numbers = int(self.factors[0].size + self.factors[1].size)   # ~ 2 r D

    def recall(self, key):
        """Recover the value bound to `key`: normalise(M^T key). From the factored form, M^T k = B^T (A^T k),
        so recall never materialises the full matrix."""
        k = np.asarray(key, float)
        if self.factors is None:
            out = self.M.T @ k
        else:
            A, B = self.factors
            out = B.T @ (A.T @ k)
        return _unit(out)


def outer_bind(a, b):
    """A single tensor-product binding: the outer product a (X) b (a D x D matrix). The uncompressed form
    that HRR's circular convolution projects down to a single D-vector."""
    return np.outer(np.asarray(a, float), np.asarray(b, float))

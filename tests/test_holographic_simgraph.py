"""Tests for geometry-weighted graph operations on hypervectors (ARCH-3): the cotangent-Laplacian mirror via
cosine-similarity weighting. The weighted similarity-graph eigenmap recovers a ring; weighting corrects
non-uniform sampling (where it beats binary); under uniform sampling weighted and binary tie (the kept negative,
high-D concentration of measure)."""

import numpy as np

from holographic.misc.holographic_simgraph import similarity_adjacency, spectral_embedding, ring_order, _ring_vectors, _circ_corr


# ---- the positive: ring recovery ------------------------------------------------------------------
def test_weighted_eigenmap_recovers_a_ring():
    V, th = _ring_vectors(nonuniform=False, seed=0)
    assert _circ_corr(ring_order(V, weighted=True), th) > 0.99


# ---- the geometry is in the weights ---------------------------------------------------------------
def test_weighted_edges_carry_varying_similarities():
    V, _ = _ring_vectors(seed=0)
    A = similarity_adjacency(V, k=6, weighted=True)
    nz = A[A > 0]
    assert nz.min() > 0.0 and nz.max() <= 1.0000001 and nz.std() > 0.0


def test_binary_edges_are_all_one():
    V, _ = _ring_vectors(seed=0)
    A = similarity_adjacency(V, k=6, weighted=False)
    assert set(np.unique(A[A > 0])) == {1.0}


def test_similarity_adjacency_is_symmetric():
    V, _ = _ring_vectors(seed=0)
    A = similarity_adjacency(V, k=6, weighted=True)
    assert np.allclose(A, A.T)


# ---- where weighting wins, and where it ties ------------------------------------------------------
def test_weighting_wins_under_nonuniform_sampling():
    # the cotangent-Laplacian's role: correcting irregular sampling density
    V, th = _ring_vectors(nonuniform=True, seed=0)
    rec_w = _circ_corr(ring_order(V, weighted=True), th)
    rec_b = _circ_corr(ring_order(V, weighted=False), th)
    assert rec_w > rec_b


def test_weighting_ties_binary_under_uniform_sampling():
    # the kept negative: high-D concentration makes the kNN graph robust to weighting (unlike a mesh)
    V, th = _ring_vectors(nonuniform=False, seed=0)
    rec_w = _circ_corr(ring_order(V, weighted=True), th)
    rec_b = _circ_corr(ring_order(V, weighted=False), th)
    assert abs(rec_w - rec_b) < 0.01


# ---- shapes / determinism -------------------------------------------------------------------------
def test_spectral_embedding_shape():
    V, _ = _ring_vectors(n=80, seed=1)
    emb = spectral_embedding(V, k=6, dims=2, weighted=True)
    assert emb.shape == (80, 2)


def test_ring_order_length():
    V, _ = _ring_vectors(n=100, seed=2)
    assert ring_order(V, weighted=True).shape == (100,)


def test_geometry_weighted_graph_is_deterministic():
    V, _ = _ring_vectors(seed=0)
    assert np.array_equal(ring_order(V, weighted=True), ring_order(V, weighted=True))

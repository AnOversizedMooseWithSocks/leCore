"""Tests for holographic_index -- the Index home (H1: one nearest-neighbour interface, exact/forest + abstain)."""
import numpy as np
from holographic.caching_and_storage.holographic_index import Index, index_backends


def _data(n=200, dim=128, seed=0):
    return np.random.default_rng(seed).standard_normal((n, dim))


def test_exact_and_forest_agree_on_easy_query():
    V = _data()
    rng = np.random.default_rng(1)
    q = V[42] + 0.15 * rng.standard_normal(V.shape[1])
    assert Index(V, method="exact").nearest(q)[0][0] == 42
    assert Index(V, method="forest", forest_threshold=0).nearest(q)[0][0] == 42


def test_topk_descending_and_deterministic():
    V = _data()
    q = V[10] + 0.1 * np.random.default_rng(2).standard_normal(V.shape[1])
    hits = Index(V, method="exact").nearest(q, k=6)
    scores = [s for _, s in hits]
    assert len(hits) == 6 and scores == sorted(scores, reverse=True)
    assert Index(V, method="exact").nearest(q, k=6) == hits          # deterministic run-to-run


def test_labels_returned():
    V = _data()
    labels = [f"v{i}" for i in range(len(V))]
    assert Index(V, labels=labels, method="exact").nearest(V[7])[0][0] == "v7"


def test_calibrated_abstain_rejects_noise():
    V = _data()
    rng = np.random.default_rng(3)
    assert Index(V, method="exact").nearest(rng.standard_normal(V.shape[1]), abstain=0.01) == []
    assert Index(V, method="exact").nearest(V[5], abstain=0.01)[0][0] == 5


def test_auto_routes_by_size():
    V = _data()
    assert Index(V, method="auto", forest_threshold=1000).method == "exact"
    assert Index(V, method="auto", forest_threshold=50).method == "forest"
    assert set(index_backends()) == {"exact", "forest"}


def test_empty_index():
    assert Index(np.zeros((0, 8)), method="exact").nearest(np.ones(8)) == []

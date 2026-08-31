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


def test_sphere_is_exact_in_both_regimes_and_prunes_only_where_structure_exists():
    # the sphere contract: bit-identical to exact ALWAYS; sublinear ONLY where cluster mass
    # exists. The whitened-dust negative is pinned so nobody 'fixes' concentration of measure.
    import numpy as np
    from holographic.caching_and_storage.holographic_index import Index
    rng = np.random.default_rng(0)
    clustered = np.repeat(rng.standard_normal((80, 48)), 60, 0) + 0.1 * rng.standard_normal((4800, 48))
    dust = rng.standard_normal((4800, 48))
    for X, expect_prune in ((clustered, True), (dust, False)):
        ex, sp = Index(X, method="exact"), Index(X, method="sphere")
        for t in range(10):
            q = X[rng.integers(len(X))] + 0.05 * rng.standard_normal(48)
            assert [i for i, _ in ex.nearest(q, k=8)] == [i for i, _ in sp.nearest(q, k=8)]
        if expect_prune:
            assert sp.sphere_touched < 0.2, sp.sphere_touched
        else:
            assert sp.sphere_touched > 0.9, sp.sphere_touched


def test_index_merge_ablate_monoid_laws():
    # HDRIFT's compose/ablate on retrieval: exact over the union, round-trip identity,
    # commutative up to tie order. Bounds survive merge because each block's radius is a
    # fact about its own members -- if any law breaks, someone re-optimized what must only
    # concatenate.
    import numpy as np
    from holographic.caching_and_storage.holographic_index import Index
    rng = np.random.default_rng(3)
    A = np.repeat(rng.standard_normal((30, 32)), 40, 0) + 0.1 * rng.standard_normal((1200, 32))
    B = np.repeat(rng.standard_normal((20, 32)), 40, 0) + 0.1 * rng.standard_normal((800, 32))
    ia = Index(A, labels=["a%d" % i for i in range(len(A))], method="sphere")
    ib = Index(B, labels=["b%d" % i for i in range(len(B))], method="sphere")
    ia.nearest(A[0]); ib.nearest(B[0])
    iab = ia.merge(ib, "A", "B")
    ref = Index(np.vstack([A, B]), labels=ia.labels + ib.labels, method="exact")
    for t in range(8):
        q = (A if t % 2 else B)[rng.integers(800)] + 0.05 * rng.standard_normal(32)
        assert set(l for l, _ in iab.nearest(q, k=6)) == set(l for l, _ in ref.nearest(q, k=6))
    back = iab.ablate("B")
    for t in range(8):
        q = A[rng.integers(1200)] + 0.05 * rng.standard_normal(32)
        assert [l for l, _ in back.nearest(q, k=6)] == [l for l, _ in ia.nearest(q, k=6)]
    ba = ib.merge(ia, "B", "A")
    q = A[5] + 0.05 * rng.standard_normal(32)
    assert sorted((l, round(s, 10)) for l, s in iab.nearest(q, k=6)) == \
           sorted((l, round(s, 10)) for l, s in ba.nearest(q, k=6))


def test_compact_storage_is_selfconsistent_and_f32_primary():
    # compact contract: f32-normalized rows ARE the index (no f64 full copy -- the fast
    # machinery aliases them zero-copy), answers bit-equal to f64 arithmetic over those same
    # f32 rows, deterministic run to run. Compact is its OWN tie domain, opt-in; the default
    # index stays bit-stable.
    import numpy as np
    from holographic.caching_and_storage.holographic_index import Index
    rng = np.random.default_rng(5)
    X = rng.standard_normal((5000, 64)).astype(np.float32)
    c = Index(X, method="exact", compact=True)
    ref = Index(np.asarray(c.items, np.float64), method="exact")
    for t in range(12):
        q = X[rng.integers(5000)] + 0.05 * rng.standard_normal(64).astype(np.float32)
        assert [i for i, _ in c.nearest(q, k=8)] == [i for i, _ in ref.nearest(q, k=8)]
    assert c.items.dtype == np.float32
    c.nearest(X[0], k=2)
    assert c._items32 is c.items          # zero-copy alias, not a duplicate


def test_screens_state_roundtrip_guard_and_bulk_finish():
    # persistence: restored bake answers bit-equal; mismatched corpus refused loudly;
    # bulk-finish fires on dust (bounds prune nothing -> delegate to exact) and stays off
    # where structure prunes. The worst case must cost the exact path, never 170x it.
    import numpy as np
    from holographic.caching_and_storage.holographic_index import Index
    rng = np.random.default_rng(6)
    X = np.repeat(rng.standard_normal((40, 32)), 40, 0) + 0.1 * rng.standard_normal((1600, 32))
    a = Index(X, method="sphere")
    q = X[3] + 0.05 * rng.standard_normal(32)
    r1 = a.nearest(q, k=6)
    st = a.screens_state()
    b = Index(X, method="sphere").screens_restore(st)
    assert b.nearest(q, k=6) == r1
    assert b.sphere_bulk is False
    try:
        Index(X + 1e-5, method="sphere").screens_restore(st)
        assert False, "mismatch guard must fire"
    except ValueError:
        pass
    # dust must have MORE than 32 blocks for the guard to be reachable (it is a large-N
    # device; 8 span matvecs need no rescue) -- 20000/512 = 40 blocks
    dust = rng.standard_normal((20000, 32))
    d = Index(dust, method="sphere")
    ex = Index(dust, method="exact")
    qd = dust[7] + 0.05 * rng.standard_normal(32)
    assert [i for i, _ in d.nearest(qd, k=6)] == [i for i, _ in ex.nearest(qd, k=6)]
    assert d.sphere_bulk is True and d.sphere_touched == 1.0


def test_int8_rung_is_certified_exact_in_both_regimes():
    # the precision-ladder rung: exact indices vs the f64 path on clustered AND dust data,
    # planted near-duplicate ties included. numba absent -> the route must not exist (skip).
    import numpy as np
    import pytest
    from holographic.caching_and_storage.holographic_index import Index
    if Index._int8_kernel() is None:
        pytest.skip("numba absent -- the int8 rung correctly does not exist")
    rng = np.random.default_rng(8)
    clustered = np.repeat(rng.standard_normal((50, 64)), 40, 0) + 0.1 * rng.standard_normal((2000, 64))
    dust = rng.standard_normal((2000, 64))
    twin = dust.copy(); twin[7] = twin[3] + 1e-9 * rng.standard_normal(64)  # boundary tie plant
    for X in (clustered, dust, twin):
        ex = Index(X, method="exact")
        i8 = Index(X, method="int8")
        for t in range(12):
            q = X[rng.integers(len(X))] + 0.05 * rng.standard_normal(64)
            assert [i for i, _ in i8.nearest(q, k=8)] == [i for i, _ in ex.nearest(q, k=8)]


def test_merge_carries_the_int8_rung():
    # per-row int8 facts survive union like block radii: both sides baked -> the merged
    # index serves the certified int8 route over the union with ZERO requantization.
    import numpy as np
    import pytest
    from holographic.caching_and_storage.holographic_index import Index
    if Index._int8_kernel() is None:
        pytest.skip("numba absent")
    rng = np.random.default_rng(9)
    A = np.repeat(rng.standard_normal((20, 32)), 30, 0) + 0.1 * rng.standard_normal((600, 32))
    B = np.repeat(rng.standard_normal((15, 32)), 30, 0) + 0.1 * rng.standard_normal((450, 32))
    ia = Index(A, labels=["a%d" % i for i in range(600)], method="int8")
    ib = Index(B, labels=["b%d" % i for i in range(450)], method="int8")
    ia.nearest(A[0]); ib.nearest(B[0])
    iab = ia.merge(ib, "A", "B")
    iab.method = "int8"
    assert getattr(iab, "_items8", None) is not None and len(iab._items8) == 1050
    ref = Index(np.vstack([A, B]), labels=ia.labels + ib.labels, method="exact")
    for t in range(10):
        q = (A if t % 2 else B)[rng.integers(450)] + 0.05 * rng.standard_normal(32)
        assert [i for i, _ in iab.nearest(q, k=6)] == [i for i, _ in ref.nearest(q, k=6)]

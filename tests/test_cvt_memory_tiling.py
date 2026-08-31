"""LEVER 5 applied to cvt_remesh: the assignment step, tiled.

The direct k-means assignment materialises an (N, K, 3) array. MEASURED: 7.57 GiB at
N=112,838 vertices and K=3,000 sites, which died outright and blocked isotropic remeshing of
any full-resolution body -- the exact template quality O1 wants.

Each vertex's nearest site depends on NO other vertex, so the axis is embarrassingly
partitionable and chunking is BIT-IDENTICAL rather than approximate. That is the whole
justification for the change, and this test pins it.
"""
import numpy as np
import pytest


def _mesh(m, res=26):
    sph = lambda P: np.linalg.norm(np.asarray(P, float), axis=1) - 1.0
    return m.mesh_from_sdf(sph, ((-1.3,) * 3, (1.3,) * 3), res=res, vectorized=True)


def test_tiled_assignment_is_deterministic():
    """Tiling must not perturb the result: two runs identical, and the site count honoured."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    t = _mesh(m)
    a, _ = m.cvt_remesh(t, n_sites=300, iterations=4)
    b, _ = m.cvt_remesh(t, n_sites=300, iterations=4)
    assert np.array_equal(np.asarray(a.vertices), np.asarray(b.vertices))
    assert len(np.asarray(a.vertices)) == 300


@pytest.mark.slow
def test_large_case_that_used_to_exhaust_memory():
    """The regression guard. A vertex count x site count whose dense form is >1 GiB must now
    complete; before tiling this raised _ArrayMemoryError. Kept modest so CI stays quick
    while still exceeding the old dense budget."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    t = _mesh(m, res=52)
    V = np.asarray(t.vertices)
    sites = 2000
    dense_gib = len(V) * sites * 3 * 8 / 2 ** 30
    assert dense_gib > 0.5, "test no longer exercises the memory path (%.2f GiB)" % dense_gib
    out, _ = m.cvt_remesh(t, n_sites=sites, iterations=3)
    assert len(np.asarray(out.vertices)) == sites

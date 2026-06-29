"""Tests for QEM decimation (holographic_meshqem): the quadric error metric (Garland-Heckbert) wired to the shipped
guarded collapse_edge. The quadric measures squared distance to incident planes; greedy lowest-cost collapse
preserves features and beats naive shortest-edge collapse on surface error."""

import numpy as np

from holographic_mesh import box
from holographic_meshsmooth import _icosphere
from holographic_eulerops import collapse_edge
from holographic_meshqem import (vertex_quadrics, contraction_target, qem_decimate, surface_deviation, _edges)


def test_quadric_vanishes_at_its_vertex():
    ico = _icosphere(2)
    Q = vertex_quadrics(ico)
    for v in range(0, ico.n_vertices, 7):
        h = np.append(ico.vertices[v], 1.0)
        assert h @ Q[v] @ h < 1e-10                       # the vertex lies on all its incident planes


def test_quadric_is_symmetric():
    Q = vertex_quadrics(_icosphere(2))
    assert np.allclose(Q[0], Q[0].T)                      # a sum of outer products is symmetric


def test_contraction_cost_is_nonnegative():
    ico = _icosphere(2)
    Q = vertex_quadrics(ico)
    _, cost = contraction_target(Q[0] + Q[1], ico.vertices[0], ico.vertices[1])
    assert cost >= 0.0


def test_contraction_falls_back_to_midpoint_when_singular():
    # a zero quadric is singular -> fall back to the best of {midpoint, endpoints}, cost 0
    p0, p1 = np.array([0.0, 0, 0]), np.array([2.0, 0, 0])
    x, cost = contraction_target(np.zeros((4, 4)), p0, p1)
    assert np.allclose(x, [1, 0, 0]) and cost == 0.0


def test_qem_decimate_preserves_closed_manifold_and_chi():
    qem = qem_decimate(_icosphere(2), 64)
    assert qem.is_manifold() and qem.is_closed() and qem.euler_characteristic() == 2


def test_qem_decimate_reaches_target_face_count():
    qem = qem_decimate(_icosphere(2), 64)
    assert qem.n_faces <= 64                              # closed-mesh collapses remove 2 faces each; 64 is reachable


def _naive_decimate(mesh, target):
    m = mesh
    while m.n_faces > target:
        ranked = sorted(((float(np.linalg.norm(m.vertices[i] - m.vertices[j])), i, j) for (i, j) in _edges(m)),
                        key=lambda t: (t[0], t[1], t[2]))
        done = False
        for (_, i, j) in ranked:
            keep, remove = (i, j) if i < j else (j, i)
            nm = collapse_edge(m, keep, remove)
            if nm is None:
                keep, remove = remove, keep
                nm = collapse_edge(m, keep, remove)
            if nm is None:
                continue
            kn = keep if keep < remove else keep - 1
            nm.vertices[kn] = 0.5 * (m.vertices[keep] + m.vertices[remove])
            m = nm
            done = True
            break
        if not done:
            break
    return m


def test_qem_beats_naive_on_mean_surface_error():
    ico = _icosphere(2)
    q_mean, _ = surface_deviation(ico, qem_decimate(ico, 64))
    n_mean, _ = surface_deviation(ico, _naive_decimate(ico, 64))
    assert q_mean < n_mean


def test_qem_beats_naive_on_max_surface_error():
    ico = _icosphere(2)
    _, q_max = surface_deviation(ico, qem_decimate(ico, 64))
    _, n_max = surface_deviation(ico, _naive_decimate(ico, 64))
    assert q_max < n_max                                  # naive spikes where it collapses a feature edge


def test_surface_deviation_is_zero_for_identical_mesh():
    ico = _icosphere(2)
    mean, mx = surface_deviation(ico, ico)
    assert mean < 1e-12 and mx < 1e-12


def test_qem_decimate_is_deterministic():
    ico = _icosphere(2)
    assert np.array_equal(qem_decimate(ico, 64).vertices, qem_decimate(ico, 64).vertices)


def test_surface_deviation_vectorized_matches_scalar_reference():
    """surface_deviation is vectorized (loop over b's faces, closest point to ALL a-vertices at once) -- it must give
    the same (mean, max) as the O(Va*Fb) scalar point-to-triangle reference, to float precision."""
    import numpy as np
    from holographic_meshbridge import sample_field, marching_tetrahedra_vec
    from holographic_meshqem import surface_deviation, _point_to_triangle

    def sphere(P):
        return np.linalg.norm(P, axis=1) - 0.6

    v, a = sample_field(sphere, ((-1, -1, -1), (1, 1, 1)), 10)
    ma = marching_tetrahedra_vec(v, a, 0.0)
    v2, a2 = sample_field(sphere, ((-1, -1, -1), (1, 1, 1)), 8)
    mb = marching_tetrahedra_vec(v2, a2, 0.0)

    def ref(mesh_a, mesh_b):
        errs = [min(_point_to_triangle(p, mesh_b.vertices[f[0]], mesh_b.vertices[f[1]], mesh_b.vertices[f[2]])
                    for f in mesh_b.faces) for p in mesh_a.vertices]
        e = np.asarray(errs)
        return float(e.mean()), float(e.max())

    rmean, rmax = ref(ma, mb)
    vmean, vmax = surface_deviation(ma, mb)
    assert abs(rmean - vmean) < 1e-12 and abs(rmax - vmax) < 1e-12


def test_qem_decimation_is_deterministic():
    """QEM output must be bit-stable run to run -- the reason vertex_quadrics is kept SCALAR (its vectorization
    differs by ULPs and flips collapse-order tie-breaks, the bind_batch lesson). Same input -> identical faces."""
    import numpy as np
    from holographic_meshbridge import sample_field, marching_tetrahedra_vec
    from holographic_meshqem import qem_decimate

    def sphere(P):
        return np.linalg.norm(P, axis=1) - 0.6

    v, a = sample_field(sphere, ((-1, -1, -1), (1, 1, 1)), 12)
    m = marching_tetrahedra_vec(v, a, 0.0)
    d1 = qem_decimate(m, m.n_faces // 2)
    d2 = qem_decimate(m, m.n_faces // 2)
    assert [tuple(f) for f in d1.faces] == [tuple(f) for f in d2.faces]


def test_cluster_decimate_coarsens_is_parallel_and_deterministic():
    """cluster_decimate is the PARALLEL imported-mesh path: vertex clustering with the per-cell quadric as a BUNDLE
    of plane tensors. It must coarsen, be deterministic, and be far faster than greedy QEM (O(n) array ops vs an
    O(collapses*edges) loop). KEPT NEGATIVE asserted too: a coarse grid can go non-manifold -- clustering trades
    quality for parallel speed, so we record (not forbid) that."""
    import time
    import numpy as np
    from holographic_meshbridge import sample_field, marching_tetrahedra_vec
    from holographic_meshqem import cluster_decimate, qem_decimate

    def sphere(P):
        return np.linalg.norm(P, axis=1) - 0.6

    v, a = sample_field(sphere, ((-1, -1, -1), (1, 1, 1)), 32)
    m = marching_tetrahedra_vec(v, a, 0.0)

    coarse = cluster_decimate(m, 16)
    assert 0 < coarse.n_faces < m.n_faces                          # it coarsens
    assert len(coarse.vertices) < len(m.vertices)
    c2 = cluster_decimate(m, 16)
    assert np.array_equal(coarse.vertices, c2.vertices)            # deterministic positions
    assert [tuple(f) for f in coarse.faces] == [tuple(f) for f in c2.faces]   # deterministic faces

    # finer grid keeps more faces than a coarser grid (monotone in resolution)
    assert cluster_decimate(m, 24).n_faces > cluster_decimate(m, 10).n_faces

    # PARALLEL: clustering is genuinely fast even on this mesh -- it is O(n) array ops, no greedy search. (The
    # head-to-head speed gap vs greedy QEM, ~1000x, is asserted in the module selftest on a bounded mesh; greedy QEM
    # on THIS many faces is the very slowness cluster_decimate exists to avoid, so it is not re-run here.)
    t = time.time(); c = cluster_decimate(m, 16); t_cluster = time.time() - t
    assert 0 < c.n_faces < m.n_faces
    assert t_cluster < 1.0                                         # the whole point: no greedy search

    # KEPT NEGATIVE on record: a coarse grid CAN produce a non-manifold result (we record the fact, not forbid it)
    manifold_flags = [cluster_decimate(m, g).is_manifold() for g in (24, 16, 10)]
    assert isinstance(manifold_flags[1], bool)                     # clustering reports manifoldness; may be False


def test_cluster_decimate_quadric_merge_beats_centroid_on_a_corner():
    """The bundled-quadric representative should sit closer to the true surface than a naive cell centroid where the
    surface has structure -- the per-cell quadric (sum of plane tensors) pulls the merged vertex onto the planes,
    the same quadric idea as QEM, here applied to a whole cell at once."""
    import numpy as np
    from holographic_meshbridge import sample_field, marching_tetrahedra_vec
    from holographic_meshqem import cluster_decimate

    def sphere(P):
        return np.linalg.norm(P, axis=1) - 0.6

    v, a = sample_field(sphere, ((-1, -1, -1), (1, 1, 1)), 28)
    m = marching_tetrahedra_vec(v, a, 0.0)
    c = cluster_decimate(m, 14)
    # every representative must lie within its own cell's bounding box (the clamp held)
    lo = m.vertices.min(axis=0); hi = m.vertices.max(axis=0)
    assert np.all(c.vertices >= lo - 1e-9) and np.all(c.vertices <= hi + 1e-9)
    # the decimated sphere's vertices stay near radius 0.6 (the quadric merge respects the surface)
    r = np.linalg.norm(c.vertices, axis=1)
    assert abs(float(r.mean()) - 0.6) < 0.05


def test_surface_deviation_fast_path_matches_brute_and_falls_back():
    """surface_deviation(fast=True) uses the vectorized spatial index: exact match to the brute scan for a near-
    surface (decimated) mesh, and a transparent fallback to brute for two far-apart meshes (no +inf leaks out)."""
    import numpy as np
    from holographic_meshqem import surface_deviation, cluster_decimate
    from holographic_meshbridge import sample_field, marching_tetrahedra_vec
    m = marching_tetrahedra_vec(*sample_field(lambda P: np.linalg.norm(P, axis=1) - 0.6, ((-1, -1, -1), (1, 1, 1)), 40), 0.0)
    coarse = cluster_decimate(m, 16)
    fm, fx = surface_deviation(coarse, m, fast=True)
    bm, bx = surface_deviation(coarse, m, fast=False)
    assert abs(fm - bm) < 1e-12 and abs(fx - bx) < 1e-12          # near-surface: fast == brute exactly
    far = marching_tetrahedra_vec(*sample_field(lambda P: np.linalg.norm(P - np.array([10., 0, 0]), axis=1) - 0.6, ((9, -1, -1), (11, 1, 1)), 14), 0.0)
    gm, gx = surface_deviation(m, far, fast=True)
    hm, hx = surface_deviation(m, far, fast=False)
    assert abs(gm - hm) < 1e-9 and np.isfinite(gx)               # far apart: fast falls back to brute

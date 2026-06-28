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

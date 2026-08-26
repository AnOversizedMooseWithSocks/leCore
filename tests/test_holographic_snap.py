"""Modeling-app feature layer: snapping = cleanup to nearest grid/vertex/edge/angle."""
import numpy as np
from holographic.caching_and_storage.holographic_snap import snap_to_grid, snap_to_points, snap_to_segment, snap_value, snap_angle, Snapper


def test_grid_snap():
    assert np.allclose(snap_to_grid([0.12, 0.49, -0.51], 0.25), [0.0, 0.5, -0.5])


def test_point_snap_is_cleanup():
    pts = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    sp, i, d = snap_to_points([0.9, 0.1, 0.0], pts)
    assert i == 1 and np.allclose(sp, [1, 0, 0])


def test_tolerance_gates_far_snap():
    pts = [[0, 0, 0], [1, 0, 0]]
    sp, i, d = snap_to_points([5, 5, 5], pts, tol=0.5)
    assert i == -1 and np.allclose(sp, [5, 5, 5])            # left alone -- the confidence refusal


def test_segment_snap_clamps():
    assert np.allclose(snap_to_segment([0.5, 5, 0], [0, 0, 0], [1, 0, 0]), [0.5, 0, 0])
    assert np.allclose(snap_to_segment([-1, 0, 0], [0, 0, 0], [1, 0, 0]), [0, 0, 0])


def test_value_and_angle_snap():
    assert abs(snap_value(0.62, 0.25) - 0.5) < 1e-12
    assert abs(snap_angle(np.radians(20), np.radians(15)) - np.radians(15)) < 1e-9


def test_snapper_prefers_vertex():
    snp = Snapper(grid=1.0, vertices=[[0.05, 0.05, 0.0]], tol=0.25)
    out, kind = snp.snap([0.1, 0.1, 0.0])
    assert kind == "vertex" and np.allclose(out, [0.05, 0.05, 0.0])
    out, kind = snp.snap([0.4, 0.4, 0.4])
    assert kind == "none"

"""Tests for holographic_pointsplat -- projecting and splatting 3D points (particles) into a camera image."""
import numpy as np
from holographic.rendering.holographic_pointsplat import splat_points, _project
from holographic.rendering.holographic_render import Camera


def _cam(aspect=1.0):
    return Camera(eye=(0.0, 0.0, 3.0), target=(0.0, 0.0, 0.0), fov_deg=45.0, aspect=aspect)


def test_point_at_origin_lands_centre():
    img, alpha = splat_points(np.array([[0.0, 0.0, 0.0]]), _cam(), 64, 64, colors=(1, 1, 1), radius_px=2.0)
    cy, cx = np.unravel_index(np.argmax(alpha), alpha.shape)
    assert abs(cy - 32) <= 2 and abs(cx - 32) <= 2
    assert 0.0 <= alpha.min() and alpha.max() <= 1.0                # alpha stays a valid coverage


def test_point_behind_camera_not_drawn():
    _, alpha = splat_points(np.array([[0.0, 0.0, 5.0]]), _cam(), 64, 64)   # behind the eye at z=3
    assert alpha.max() == 0.0


def test_offscreen_point_not_drawn():
    _, alpha = splat_points(np.array([[100.0, 0.0, 0.0]]), _cam(), 64, 64)
    assert alpha.max() == 0.0


def test_near_point_occludes_far():
    # two points on the same screen spot; the nearer (painted last) wins the centre pixel
    pts = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]])
    cols = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    img, _ = splat_points(pts, _cam(), 64, 64, colors=cols, radius_px=2.0)
    c = img[32, 32]
    assert c[0] > c[2]                                              # red (near) over blue (far)


def test_depth_fade_dims_far_points():
    _, a_far = splat_points(np.array([[0.0, 0.0, -2.0]]), _cam(), 64, 64, depth_fade=(1.0, 4.0))
    _, a_near = splat_points(np.array([[0.0, 0.0, 0.0]]), _cam(), 64, 64, depth_fade=(1.0, 4.0))
    assert a_far.max() < a_near.max()


def test_empty_cloud_is_blank():
    img, alpha = splat_points(np.zeros((0, 3)), _cam(), 32, 32)
    assert alpha.max() == 0.0 and img.shape == (32, 32, 3)


def test_deterministic():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, (50, 3))
    a = splat_points(pts, _cam(), 48, 48, colors=(1, 0.6, 0.2), radius_px=1.5)[0]
    b = splat_points(pts, _cam(), 48, 48, colors=(1, 0.6, 0.2), radius_px=1.5)[0]
    assert np.array_equal(a, b)                                    # no RNG -> bit-identical

"""Modeling-app feature layer: the camera controller -- orbit/pan/dolly/zoom/frame."""
import numpy as np
from holographic.rendering.holographic_camera import CameraController, _norm


def test_orbit_preserves_distance_and_wraps():
    cam = CameraController(eye=(0, 0, 5), target=(0, 0, 0))
    d0 = cam.distance
    cam.orbit(np.radians(90), 0.0)
    assert abs(cam.distance - d0) < 1e-9 and not np.allclose(cam.eye, [0, 0, 5])
    cam.orbit(np.radians(270), 0.0)
    assert np.allclose(cam.eye, [0, 0, 5], atol=1e-9)


def test_elevation_clamps():
    cam = CameraController(eye=(0, 0, 5), target=(0, 0, 0))
    cam.orbit(0.0, np.radians(200))
    elev = np.arcsin(np.clip(np.dot(_norm(cam.eye - cam.target), cam.up), -1, 1))
    assert elev <= np.radians(89.0) + 1e-6


def test_pan_moves_rig_together():
    cam = CameraController(eye=(0, 0, 5), target=(0, 0, 0))
    f0 = cam._forward().copy(); d0 = cam.distance
    cam.pan(2.0, 1.0)
    assert abs(cam.distance - d0) < 1e-9 and np.allclose(cam._forward(), f0, atol=1e-9)


def test_dolly_no_overshoot():
    cam = CameraController(eye=(0, 0, 5), target=(0, 0, 0))
    cam.dolly(2.0); assert abs(cam.distance - 3.0) < 1e-9
    cam.dolly(100.0); assert cam.distance >= cam.min_distance


def test_zoom_scales_distance():
    cam = CameraController(eye=(0, 0, 4), target=(0, 0, 0))
    cam.zoom(0.5); assert abs(cam.distance - 2.0) < 1e-9


def test_frame_fits_box():
    cam = CameraController(eye=(0, 0, 10), target=(0, 0, 0))
    cam.frame([0, 0, 0], [2, 2, 2], fov_deg=45.0)
    assert np.allclose(cam.target, [1, 1, 1], atol=1e-9)
    assert abs(cam.distance - 0.5 * np.linalg.norm([2, 2, 2]) / np.sin(np.radians(22.5))) < 1e-9

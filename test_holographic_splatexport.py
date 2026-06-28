"""Tests for FS-3 splat export (holographic_splatexport): the .ply / JSON adapter that writes the engine's Gaussian
splats to formats a browser splat renderer reads. The core math is principal_axes -- L (Cholesky of the inverse
covariance) -> scale + rotation quaternion -- verified by round-trip; a degenerate covariance is surfaced, not faked."""

import os
import tempfile

import numpy as np

from holographic_splatexport import (principal_axes, rotation_to_quaternion, quaternion_to_rotation,
                                     splats_to_ply, splats_from_ply, splats_to_json, splats_from_json,
                                     field_to_splats)


def _spd_covariance(seed):
    A = np.random.default_rng(seed).standard_normal((3, 3))
    return A @ A.T + 0.5 * np.eye(3)


def test_principal_axes_reconstructs_covariance():
    Sigma = _spd_covariance(0)
    P = np.linalg.inv(Sigma)
    scales, R = principal_axes(P)
    assert np.allclose(R @ np.diag(scales ** 2) @ R.T, Sigma, atol=1e-9)
    assert np.linalg.det(R) > 0                              # proper rotation


def test_quaternion_roundtrip():
    Sigma = _spd_covariance(1)
    _scales, R = principal_axes(np.linalg.inv(Sigma))
    q = rotation_to_quaternion(R)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9              # unit quaternion
    assert np.allclose(quaternion_to_rotation(q), R, atol=1e-9)


def test_ply_roundtrip_recovers_covariance_and_attributes():
    Sigma = _spd_covariance(2)
    L = np.linalg.cholesky(np.linalg.inv(Sigma))
    splats = [(np.array([0.1, -0.2, 0.3]), 0.8, L)]
    tmp = os.path.join(tempfile.gettempdir(), "holo_splat_wrap.ply")
    n = splats_to_ply(splats, tmp, colors=[[0.9, 0.1, 0.2]])
    recs = splats_from_ply(tmp)
    os.remove(tmp)
    assert n == 1 and len(recs) == 1
    s = np.array(recs[0]["scale"]); R = quaternion_to_rotation(recs[0]["rotation"])
    assert np.allclose(R @ np.diag(s ** 2) @ R.T, Sigma, atol=1e-5)
    assert np.allclose(recs[0]["position"], [0.1, -0.2, 0.3], atol=1e-5)
    assert np.allclose(recs[0]["color"], [0.9, 0.1, 0.2], atol=1e-4)
    assert abs(recs[0]["opacity"] - 0.8) < 1e-4


def test_json_roundtrip():
    L = np.linalg.cholesky(np.linalg.inv(_spd_covariance(3)))
    splats = [(np.array([1.0, 0.0, 0.5]), 0.5, L)]
    back = splats_from_json(splats_to_json(splats))
    assert len(back) == 1
    assert np.allclose(back[0]["position"], [1.0, 0.0, 0.5], atol=1e-9)
    assert "scale" in back[0] and "rotation" in back[0]


def test_field_to_splats_isotropic_std_equals_radius():
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    splats = field_to_splats(centers, radius=0.4)
    assert len(splats) == 2
    scales, _ = principal_axes(splats[0][2] @ splats[0][2].T)
    assert np.allclose(scales, 0.4, atol=1e-9)


def test_degenerate_covariance_raises():
    flat = np.diag([1.0, 1.0, 0.0])    # a rank-deficient precision -> no clean axes
    try:
        principal_axes(flat)
        assert False
    except ValueError:
        pass


def test_deterministic():
    L = np.linalg.cholesky(np.linalg.inv(_spd_covariance(4)))
    splats = [(np.array([0.0, 0.0, 0.0]), 0.7, L)]
    assert splats_to_json(splats) == splats_to_json(splats)

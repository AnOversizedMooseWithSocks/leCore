"""Forecasting sweep (sec.5) / photo-to-3D: abstaining depth lift -- observe the front, abstain on the rest."""
import numpy as np
from holographic_photo3d import unproject, depth_confidence, photo_to_gaussians


def _two_planes():
    H = W = 40
    depth = np.empty((H, W)); depth[:, :W // 2] = 1.0; depth[:, W // 2:] = 3.0
    colour = np.zeros((H, W, 3)); colour[:, :W // 2] = [0.8, 0.2, 0.2]; colour[:, W // 2:] = [0.2, 0.3, 0.8]
    return depth, colour, 40.0, 40.0, 20.0, 20.0


def test_unproject_pinhole():
    depth = np.full((3, 3), 2.0)
    pts = unproject(depth, fx=10.0, fy=10.0, cx=1.0, cy=1.0)
    # centre pixel (u=cx, v=cy) lands on the optical axis at (0,0,z)
    assert np.allclose(pts[1, 1], [0.0, 0.0, 2.0])
    # a pixel one to the right of centre has x = (2-1)*2/10 = 0.2
    assert np.isclose(pts[1, 2, 0], 0.2)


def test_abstains_at_occlusion_edge_and_holes():
    depth, colour, fx, fy, cx, cy = _two_planes()
    depth[0, 0] = 0.0                                            # a hole
    conf, abstain = depth_confidence(depth, fx, fy, cx, cy)
    e = 20
    assert abstain[20, e] or abstain[20, e - 1]                 # the occlusion edge abstains
    assert abstain[0, 0]                                        # the hole abstains
    assert not abstain[20, 5] and conf[20, 5] > 0.5            # a flat interior pixel is confident


def test_confident_splats_reconstruct_front_planes():
    depth, colour, fx, fy, cx, cy = _two_planes()
    g = photo_to_gaussians(depth, colour, fx, fy, cx, cy, confidence_floor=0.3)
    z = g["positions"][:, 2]
    assert np.isclose(z, 1.0, atol=0.1).any() and np.isclose(z, 3.0, atol=0.1).any()
    assert 0.0 < g["coverage"] < 1.0                            # honest: not every pixel is emitted
    assert g["n_abstained"] >= 1


def test_deterministic():
    depth, colour, fx, fy, cx, cy = _two_planes()
    a = photo_to_gaussians(depth, colour, fx, fy, cx, cy)
    b = photo_to_gaussians(depth, colour, fx, fy, cx, cy)
    assert np.array_equal(a["positions"], b["positions"]) and a["n_observed"] == b["n_observed"]

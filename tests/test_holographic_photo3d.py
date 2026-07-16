"""Forecasting sweep (sec.5) / photo-to-3D: abstaining depth lift -- observe the front, abstain on the rest."""
import numpy as np
from holographic.rendering.holographic_photo3d import unproject, depth_confidence, photo_to_gaussians


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


def test_image_to_mesh_repair_flag_honest():
    """image_to_mesh(repair=False) is byte-identical (default); repair=True now runs weld+SPLIT-nonmanifold+fill and
    MAKES THE MESH MANIFOLD (non-manifold edges -> 0), so the cross-field retopo accepts it -- the split closed the gap
    the earlier weld/fill-only repair could not."""
    import lecore
    from collections import Counter
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    m = lecore.UnifiedMind(dim=64)
    H = W = 40
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.exp(-(((yy - 20) / 12.0) ** 2 + ((xx - 20) / 12.0) ** 2))[..., None] * np.array([0.8, 0.7, 0.6])

    a = m.image_to_mesh(img, res=20, repair=False)
    b = m.image_to_mesh(img, res=20, repair=False)
    assert np.array_equal(a[0], b[0]) and [tuple(q) for q in a[1]] == [tuple(q) for q in b[1]]  # deterministic + off

    def nm_edges(faces):
        und = Counter()
        for f in faces:
            f = [int(i) for i in f]; n = len(f)
            for k in range(n):
                und[tuple(sorted((f[k], f[(k + 1) % n])))] += 1
        return sum(1 for c in und.values() if c > 2)

    raw = Mesh(a[0], [tuple(int(i) for i in q) for q in a[1]])
    assert nm_edges(a[1]) > 0 and not raw.is_manifold()                       # raw output IS non-manifold

    v1, q1, _f, _g = m.image_to_mesh(img, res=20, repair=True)                # opt-in repair splits non-manifold verts
    rep = Mesh(v1, [tuple(int(i) for i in f) for f in q1])
    assert nm_edges(q1) == 0 and rep.is_manifold()                           # repaired output IS manifold (the payoff)

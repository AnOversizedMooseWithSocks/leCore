"""O2: procedural blendshape basis with DECLARED local support, plus L2 (partition of unity
survives blending, including extrapolated weights).

SMPL's dense pose correctives "relate every vertex on the mesh to all the joints in the
kinematic tree, capturing spurious long-range correlations" -- artifacts STAR calls
"unappealing for animators". STAR fixes it by TRAINING on scans to learn each joint's
activation region. Authoring a basis, we DECLARE the region, so the fix is free and the
locality is exact rather than learned. These tests pin that exactness.
"""
import numpy as np
from holographic.mesh_and_geometry import holographic_blendbasis as BB


def _sphere_mesh(m):
    sph = lambda P: np.linalg.norm(np.asarray(P, float), axis=1) - 1.0
    return m.mesh_from_sdf(sph, ((-1.3,) * 3, (1.3,) * 3), res=24, vectorized=True)


def test_support_is_exactly_local_no_overreach():
    """THE STAR PROPERTY, by declaration rather than training: a corrective influences no
    vertex beyond its declared geodesic radius. Measured overreach must be exactly zero --
    'small' is not good enough, because any leak is the long-range coupling itself."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    mesh = _sphere_mesh(m)
    V = np.asarray(mesh.vertices, float)
    srcs = [int(np.argmax(V[:, 1])), int(np.argmin(V[:, 1]))]
    radii = [0.8, 0.5]
    targets = [BB.make_corrective(mesh, s, r, "normal", 0.25, m)
               for s, r in zip(srcs, radii)]
    rep = BB.locality_report(V, targets, mesh, srcs, radii, m)
    assert rep["max_overreach"] == 0.0, rep
    assert rep["local"]
    # and SPARSE: each corrective moves a minority of the mesh, which is STAR's other claim
    assert all(t["fraction_moved"] < 0.6 for t in rep["targets"]), rep


def test_disjoint_supports_do_not_overlap():
    """Two correctives on opposite poles must share no moved vertex. If they did, a shoulder
    shrug would tug the hip -- the exact SMPL artifact."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    mesh = _sphere_mesh(m)
    V = np.asarray(mesh.vertices, float)
    a = BB.make_corrective(mesh, int(np.argmax(V[:, 1])), 0.8, "normal", 0.25, m)
    b = BB.make_corrective(mesh, int(np.argmin(V[:, 1])), 0.5, "normal", 0.25, m)
    ma = np.linalg.norm(a - V, axis=1) > 1e-9
    mb = np.linalg.norm(b - V, axis=1) > 1e-9
    assert ma.any() and mb.any()
    assert not (ma & mb).any()


def test_L2_partition_of_unity_survives_blending_and_extrapolation():
    """BACKLOG L2, verified empirically. Animators drive blendshape weights outside [0,1]
    constantly; if the blend+skin chain broke the partition of unity the posed mesh would
    shrink toward the origin. Measured exactly 1.0 at nominal AND extrapolated weights."""
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    mesh = _sphere_mesh(m)
    V = np.asarray(mesh.vertices, float)
    srcs = [int(np.argmax(V[:, 1])), int(np.argmin(V[:, 1]))]
    targets = [BB.make_corrective(mesh, s, r, "normal", 0.2, m)
               for s, r in zip(srcs, [0.8, 0.5])]
    bones = V[::400]
    for wts in ([1.0, 0.5], [1.8, -0.6], [0.0, 0.0]):
        mixed = np.asarray(m.blend_shapes(V, targets, wts), float)
        assert np.all(np.isfinite(mixed)), wts
        W = np.asarray(m.skin_bind_weights(mixed, bones, falloff=2.0, max_influences=4),
                       float)
        s = W.sum(1)
        assert abs(s.min() - 1.0) < 1e-8 and abs(s.max() - 1.0) < 1e-8, (wts, s.min(), s.max())

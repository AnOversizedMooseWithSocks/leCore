"""Structural duplicate scan (rev. 8): three bodies were bit-identical and got ONE owner; one look-alike was
semantically DIFFERENT and got a warning instead. Guards both, so a future edit cannot re-fork them or wrongly merge
the look-alike.
"""
import numpy as np


def test_the_one_trilinear_reader():
    # cachehome OWNS `trilinear_sample`; matbake DELEGATES. Their readers agreed bit for bit before, and must stay
    # so -- on cubic AND anisotropic grids, and OFF the nodes (where the original selftest was blind).
    import importlib.util
    import os

    from holographic.caching_and_storage.holographic_cachehome import trilinear_sample
    import holographic.materials_and_texture.holographic_matbake as mb

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "holographic", "materials_and_texture", "holographic_matbake.py"),
               encoding="utf-8").read()
    assert "trilinear_sample" in src            # it delegates, not re-implements

    rng = np.random.default_rng(0)
    for shape in ((6, 6, 6), (6, 6, 6, 3), (5, 7, 9), (5, 7, 9, 3)):
        grid = rng.normal(size=shape)
        lo, hi = np.zeros(3), np.ones(3)
        P = rng.uniform(-0.3, 1.3, (300, 3))              # includes out-of-box clamping
        field = mb.BakedField(grid, lo, hi)
        assert np.array_equal(np.asarray(field.sample(P)),
                              np.asarray(trilinear_sample(grid, lo, hi, field.res, P)))


def test_the_trilinear_reader_needs_all_eight_corners():
    # THE REGRESSION THE ORIGINAL SELFTEST MISSED: it checked only grid NODES, where the weights are 0/1, so a
    # `return` inside the corner loop still passed. Off a node, all eight corners must contribute.
    from holographic.caching_and_storage.holographic_cachehome import trilinear_sample

    g = np.zeros((3, 3, 3))
    g[0, 0, 0] = 1.0
    mid = trilinear_sample(g, np.zeros(3), np.ones(3), np.array([3, 3, 3]), np.array([[0.25, 0.25, 0.25]]))
    assert abs(float(mid[0]) - 0.125) < 1e-12            # (1-0.5)^3, and it needs the far corner


def test_the_one_quaternion_product():
    from holographic.misc.holographic_transform import quat_mul
    from holographic.simulation_and_physics.holographic_cosserat import qmul

    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.normal(size=4), rng.normal(size=4)
        assert np.array_equal(np.asarray(qmul(a, b)), np.asarray(quat_mul(a, b)))


def test_the_one_newell_normal():
    from holographic.mesh_and_geometry.holographic_meshcurvature import _newell_normal
    from holographic.mesh_and_geometry.holographic_meshverbs import newell_normal

    rng = np.random.default_rng(0)
    V = rng.normal(size=(8, 3))
    for face in ([0, 3, 5, 6], [1, 2, 4], [0, 1, 2, 3, 4]):
        assert np.array_equal(np.asarray(_newell_normal(V, face)), np.asarray(newell_normal(V, face)))


def test_the_lookalike_is_NOT_merged():
    # meshbridge._face_normal has the same AST shape as Newell but is the first-three-vertices normal. On a bent
    # polygon they diverge. Structurally identical, semantically different -- a merge would be a bug.
    from holographic.mesh_and_geometry.holographic_meshbridge import _face_normal as first_three
    from holographic.mesh_and_geometry.holographic_meshverbs import newell_normal

    rng = np.random.default_rng(1)
    V = rng.normal(size=(6, 3))
    bent_quad = [0, 1, 2, 3]
    assert np.abs(np.asarray(newell_normal(V, bent_quad)) - np.asarray(first_three(V, bent_quad))).max() > 0.1

    tri = [0, 1, 2]
    assert np.abs(np.asarray(newell_normal(V, tri)) - np.asarray(first_three(V, tri))).max() < 1e-12   # agree on a tri


def test_the_docstrings_warn_against_re_merging():
    import holographic.mesh_and_geometry.holographic_meshbridge as mb
    assert "NOT Newell" in (mb._face_normal.__doc__ or "")

"""Part 4: FieldEffect -- shaped, decaying, composable, attachable influence."""
import numpy as np
from holographic.mesh_and_geometry.holographic_sdf import sphere
from holographic.misc.holographic_fieldeffect import FieldEffect, FieldGroup, AttachedFieldEffect, attract_to, repel_from, uniform_force


def test_weight_zero_outside_full_inside():
    fx = FieldEffect(sphere(2.0), attract_to([0, 0, 0]), radius=2.0)
    assert fx.weight(np.array([[5.0, 0, 0]]))[0] == 0.0            # outside
    assert fx.weight(np.array([[2.0, 0, 0]]))[0] == 0.0           # on the surface
    assert fx.weight(np.array([[0.0, 0, 0]]))[0] > 0.9            # deep inside


def test_apply_pulls_inward():
    fx = FieldEffect(sphere(2.0), attract_to([0, 0, 0]), radius=2.0)
    assert fx.apply(np.array([[1.0, 0, 0]]))[0][0] < 0            # pulled toward origin


def test_repel_pushes_outward():
    fx = FieldEffect(sphere(2.0), repel_from([0, 0, 0]), radius=2.0)
    assert fx.apply(np.array([[1.0, 0, 0]]))[0][0] > 0


def test_group_is_sum():
    a = FieldEffect(sphere(2.0), attract_to([0, 0, 0]), radius=2.0)
    b = FieldEffect(sphere(2.0), uniform_force([0, 1, 0]), radius=2.0)
    g = FieldGroup([a, b])
    pts = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    assert np.allclose(g.apply(pts), a.apply(pts) + b.apply(pts))


def test_attached_rides_moving_node():
    class _Node:
        def __init__(self, xyz): self.transform = np.eye(4); self.transform[:3, 3] = xyz
    node = _Node([0.0, 0.0, 0.0])
    att = AttachedFieldEffect(node, sphere(2.0), attract_to([0, 0, 0]), radius=2.0)
    world = np.array([[3.0, 0.0, 0.0]])
    w0 = att.weight(world)[0]
    node.transform[:3, 3] = [3.0, 0.0, 0.0]
    w1 = att.weight(world)[0]
    assert w1 > w0 and w1 > 0.9


def test_texture_modulates():
    base = FieldEffect(sphere(2.0), attract_to([0, 0, 0]), radius=2.0)
    tex = FieldEffect(sphere(2.0), attract_to([0, 0, 0]), radius=2.0, texture=lambda P: np.full(len(P), 0.5))
    p = np.array([[0.0, 0, 0]])
    assert np.allclose(tex.weight(p), 0.5 * base.weight(p))


def test_deterministic():
    fx = FieldEffect(sphere(2.0), attract_to([0, 0, 0]), radius=2.0)
    pts = np.array([[0.5, 0.5, 0.0]])
    assert np.array_equal(fx.apply(pts), fx.apply(pts))

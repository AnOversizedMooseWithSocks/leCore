"""holographic_fieldeffect.py -- a FIELD EFFECT: a shaped zone of influence (attractor, wind, drag, stickiness, a
density source) that acts on whatever reads it, decaying toward the edge, composable, and attachable to a moving
object.

WHY THIS EXISTS (Render/Sim Pipeline backlog, Part 4)
-----------------------------------------------------
Every ingredient was already in the tree -- SDF shapes for the region, the SDF's own distance as the falloff
coordinate, the `fields` force library, `noise`/`SparseField` for 3-D textures, SDF CSG + the force monoid for
composition, `scenegraph` transforms for attachment -- but nothing PACKAGED them into a first-class "shaped
influence." This does. The SDF is the SHAPE, its signed distance IS the falloff coordinate (0 at the surface,
growing inside), and `effect` is what the field DOES to the points that fall inside it. It renders nothing and
has no mesh interaction; it only acts on what reads it.

KEY IDEAS:
  * `weight(points)`: -d/radius clips to 0 at/outside the surface and rises to 1 a `radius` deep inside, then a
    falloff curve shapes the edge -- the SDF distance is the falloff for free.
  * `FieldGroup`: effects ADD. Force superposition is a commutative monoid, so a group is order-independent
    (the same reason `distribute` reassembles by sum).
  * `AttachedFieldEffect`: the field rides a moving object -- transform world points into the object's frame
    ("a rigid transform is a single bind") so the zone travels with it (a planet's gravity, a sticky surface).

HONEST SCOPE (kept loud): a field effect is a soft, weighted FORCE, not a hard constraint -- "sticky" is a
strong short-range attractor, not a positional lock (a true stick is a `project_onto_constraints` job). A
mesh-volume shape via `mesh_to_sdf` inherits its sign caveat (magnitude right, inside/outside can flip on deep
concavities/thin sheets). For a point-shaped case, the plain point attractor in `fields` is cheaper -- use this
when the SHAPE matters. Deterministic; NumPy + stdlib; falloff reuses holographic_sculpt.
"""
import numpy as np

from holographic_sculpt import falloff as _sculpt_falloff


def _falloff(t, kind):
    """Shape the edge: `t` in [0,1] is 0 at/outside the surface, 1 deep inside. Reuses the sculpt falloff curves
    (smooth/linear/sharp/...) by treating t as a normalized distance-in (radius 1)."""
    return _sculpt_falloff(1.0 - np.clip(t, 0.0, 1.0), 1.0, kind)     # sculpt.falloff is 1 at d=0, 0 at d=r


class FieldEffect:
    """A shaped zone of influence. `sdf` (anything with `.eval(P)`) is the shape; its signed distance is the
    falloff coordinate; `effect(points, weight) -> per-point vectors/scalars` is what it does. `radius` sets how
    deep the influence reaches full strength; `texture(points) -> scalar` optionally modulates it in 3-D."""

    def __init__(self, sdf, effect, radius=1.0, falloff="smooth", strength=1.0, texture=None):
        self.sdf = sdf
        self.effect = effect
        self.radius = float(radius)
        self.falloff = falloff
        self.strength = float(strength)
        self.texture = texture

    def weight(self, points):
        """Per-point influence in [0, strength]: 0 at/outside the surface, rising to `strength` a `radius` deep
        inside, shaped by the falloff curve, optionally modulated by a 3-D texture."""
        P = np.asarray(points, float)
        d = self.sdf.eval(P)                                         # the SDF already IS the falloff coordinate
        t = np.clip(-d / self.radius, 0.0, 1.0)                     # 0 outside -> 1 a radius deep inside
        w = self.strength * _falloff(t, self.falloff)
        if self.texture is not None:
            w = w * np.asarray(self.texture(P), float)              # 3-D texture: noise.fbm3(P) or sparsefield.sample(P)
        return w

    def apply(self, points):
        """The field's contribution at `points`: `effect(points, weight(points))`."""
        P = np.asarray(points, float)
        return self.effect(P, self.weight(P))


class FieldGroup:
    """A set of field effects whose contributions ADD. Because force superposition is a commutative monoid, the
    result is independent of the order the effects were listed in -- compose freely."""

    def __init__(self, effects):
        self.effects = list(effects)

    def apply(self, points):
        P = np.asarray(points, float)
        total = 0.0
        for fx in self.effects:
            total = total + fx.apply(P)                             # ADD (the monoid)
        return total


def _apply_inverse(matrix, world_points):
    """Transform world points into a node's LOCAL frame by the inverse of its 4x4 world transform (homogeneous)."""
    inv = np.linalg.inv(np.asarray(matrix, float))
    P = np.asarray(world_points, float)
    homog = np.concatenate([P, np.ones((len(P), 1))], axis=1)      # (N,4)
    local = homog @ inv.T
    return local[:, :3]


class AttachedFieldEffect(FieldEffect):
    """A field effect constrained to a MOVING object: its shape is defined in the object's LOCAL frame, and
    `weight` transforms world query points into that frame first -- so the zone rides the object as it moves ("a
    rigid transform is a single bind"). `node` is anything with a 4x4 `.transform` (a scenegraph SceneNode), or
    pass a 4x4 matrix / a `world_matrix()` callable directly."""

    def __init__(self, node, local_sdf, effect, **kw):
        super().__init__(local_sdf, effect, **kw)
        self.node = node

    def _world_matrix(self):
        if hasattr(self.node, "transform"):
            return self.node.transform                             # a SceneNode
        if callable(self.node):
            return self.node()                                     # a world_matrix() closure
        return np.asarray(self.node, float)                        # a raw 4x4

    def weight(self, world_points):
        local = _apply_inverse(self._world_matrix(), world_points) # world -> the object's moving frame
        return super().weight(local)


# --- a few ready effect factories (effect(points, weight) -> per-point force) -----------------------------------

def attract_to(center):
    """Pull points toward `center`, scaled by the field weight -- a gravity/attractor well."""
    center = np.asarray(center, float)
    def effect(points, weight):
        dirs = center - np.asarray(points, float)
        return dirs * np.asarray(weight, float)[:, None]
    return effect


def repel_from(center):
    """Push points away from `center`, scaled by the field weight."""
    center = np.asarray(center, float)
    def effect(points, weight):
        dirs = np.asarray(points, float) - center
        return dirs * np.asarray(weight, float)[:, None]
    return effect


def uniform_force(vector):
    """A constant force direction (wind), scaled by the field weight."""
    vector = np.asarray(vector, float)
    def effect(points, weight):
        return vector[None, :] * np.asarray(weight, float)[:, None]
    return effect


def _selftest():
    """weight is 0 outside the shape, rises to full strength deep inside, and is smooth; a FieldGroup equals the
    SUM of its members; an AttachedFieldEffect rides a moving node; a texture modulates the weight; deterministic."""
    from holographic_sdf import sphere

    s = sphere(2.0)                                                 # a sphere of radius 2 centred at origin
    fx = FieldEffect(s, attract_to([0, 0, 0]), radius=2.0, strength=1.0)

    # (1) weight: 0 well outside, ~full deep inside (at the centre), 0 exactly at/beyond the surface
    outside = fx.weight(np.array([[5.0, 0, 0]]))[0]
    centre = fx.weight(np.array([[0.0, 0, 0]]))[0]
    surface = fx.weight(np.array([[2.0, 0, 0]]))[0]
    assert outside == 0.0 and surface == 0.0                       # nothing outside or on the surface
    assert centre > 0.9                                            # ~full strength a radius deep in

    # (2) apply pulls toward the centre inside the field
    p = np.array([[1.0, 0, 0]])                                    # inside, off-centre
    force = fx.apply(p)[0]
    assert force[0] < 0                                            # pulled back toward origin (negative x)

    # (3) FieldGroup == sum of members
    fx2 = FieldEffect(sphere(2.0), uniform_force([0, 1, 0]), radius=2.0)
    group = FieldGroup([fx, fx2])
    pts = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    assert np.allclose(group.apply(pts), fx.apply(pts) + fx2.apply(pts))

    # (4) AttachedFieldEffect rides a moving node: the same WORLD point has different weight as the node moves
    class _Node:
        def __init__(self, xyz): self.transform = np.eye(4); self.transform[:3, 3] = xyz
    node = _Node([0.0, 0.0, 0.0])
    att = AttachedFieldEffect(node, sphere(2.0), attract_to([0, 0, 0]), radius=2.0)
    world_pt = np.array([[3.0, 0.0, 0.0]])
    w_here = att.weight(world_pt)[0]                               # node at origin: 3.0 is outside r=2 -> 0
    node.transform[:3, 3] = [3.0, 0.0, 0.0]                        # move the node onto the point
    w_moved = att.weight(world_pt)[0]                              # now the point is at the node's centre -> full
    assert w_moved > w_here and w_moved > 0.9

    # (5) texture modulates the weight
    ftex = FieldEffect(sphere(2.0), attract_to([0, 0, 0]), radius=2.0, texture=lambda P: np.full(len(P), 0.5))
    assert np.allclose(ftex.weight(np.array([[0.0, 0, 0]])), 0.5 * fx.weight(np.array([[0.0, 0, 0]])))

    # (6) deterministic
    assert np.array_equal(fx.apply(pts), fx.apply(pts))
    print("holographic_fieldeffect selftest OK: weight 0 outside / %.2f at centre, 0 at surface; apply pulls "
          "inward; FieldGroup == sum; AttachedFieldEffect rides a moving node (%.2f->%.2f); texture modulates; "
          "deterministic" % (centre, w_here, w_moved))


if __name__ == "__main__":
    _selftest()

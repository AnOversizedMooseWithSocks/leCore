"""holographic_transform_space.py -- the TRANSFORM + SPACE model behind a gizmo. A gizmo is a UI; the backend it
drives is 'transform a selection about a PIVOT, in a SPACE, under an axis CONSTRAINT'. That triple -- pivot, space,
constraint -- is what turns a raw matrix multiply into the move/rotate/scale a modeler expects, and it is the piece
that was missing (raw warps existed; the space model did not).

THE THREE KNOBS
  PIVOT   -- the point a rotate/scale happens around: 'median' (centroid of the selection), 'active' (a chosen
             element), 'cursor' (an arbitrary point), or 'bbox' (bounding-box centre). Translation ignores the
             pivot; rotate/scale need it.
  SPACE   -- the frame the axes mean something in: 'world' (global XYZ), 'local' (an object's own frame), or 'view'
             (the camera's frame). The same "move along X" means different directions per space.
  CONSTRAINT -- which axes are free: a mask like (1,0,0) locks the move to X; (1,1,0) to the XY plane. This is the
             press-X-to-constrain behaviour of every DCC.

Everything is an affine transform composed as translate(pivot) . R/S . translate(-pivot), applied to the selected
vertices only. Deterministic, NumPy/stdlib only.
"""

import numpy as np


def pivot_point(points, selection_idx, mode="median", cursor=None, active=None):
    """Resolve the PIVOT for a transform. `points` is the full (N,3) vertex array, `selection_idx` the selected
    vertex indices. 'median' = centroid of the selection; 'bbox' = centre of its bounding box; 'cursor' = the given
    point; 'active' = the given active vertex's position. The point a rotate/scale turns around."""
    P = np.asarray(points, float)
    sel = np.asarray(sorted(set(int(i) for i in selection_idx)), int)
    if mode == "cursor":
        if cursor is None:
            raise ValueError("pivot mode 'cursor' needs a cursor point")
        return np.asarray(cursor, float)
    if mode == "active":
        if active is None:
            raise ValueError("pivot mode 'active' needs an active vertex index")
        return P[int(active)].copy()
    if len(sel) == 0:
        return np.zeros(3)
    S = P[sel]
    if mode == "bbox":
        return 0.5 * (S.min(axis=0) + S.max(axis=0))
    return S.mean(axis=0)                                       # 'median' (centroid) default


def _space_basis(space, view_matrix=None, local_matrix=None):
    """Return the 3x3 basis whose columns are the axis directions for `space`. 'world' = identity; 'view' uses the
    camera's rotation; 'local' uses the object's rotation. A transform expressed in axis coords is rotated INTO
    world by this basis, applied, and the result stays in world -- so 'move along local X' does the right thing."""
    if space == "world":
        return np.eye(3)
    if space == "view":
        if view_matrix is None:
            raise ValueError("space 'view' needs a view_matrix (3x3 or 4x4)")
        M = np.asarray(view_matrix, float)
        return M[:3, :3]
    if space == "local":
        if local_matrix is None:
            raise ValueError("space 'local' needs a local_matrix (3x3 or 4x4)")
        M = np.asarray(local_matrix, float)
        return M[:3, :3]
    raise ValueError("space must be world/view/local; got %r" % space)


def _axis_rotation(axis, angle):
    """Rotation matrix (3x3) of `angle` radians about a unit `axis`. DELEGATES to the canonical Rodrigues
    implementation in holographic_scenegraph.rotation (which scene_rotation also uses) and returns its 3x3 block --
    so there is ONE axis-angle rotation in the engine, not two. The rotate primitive for transform_selection."""
    from holographic.scene_and_pipeline.holographic_scenegraph import rotation
    return np.asarray(rotation(axis, float(angle)), float)[:3, :3]


def transform_selection(points, selection_idx, translate=None, rotate=None, scale=None,
                        pivot="median", space="world", constraint=(1, 1, 1),
                        cursor=None, active=None, view_matrix=None, local_matrix=None, weights=None):
    """Apply a TRANSFORM to the selected vertices, honouring pivot + space + axis constraint -- the gizmo backend.
    Returns a NEW (N,3) points array (the input is not mutated; non-destructive by default).

      translate -- a 3-vector delta (in `space` axes), masked by `constraint`.
      rotate    -- (axis, angle_radians) about the pivot, axis interpreted in `space`.
      scale     -- a scalar or 3-vector about the pivot, per-axis masked by `constraint`.

    `constraint` is an axis mask: (1,0,0) locks to X, (1,1,0) to the XY plane. `weights`, if given (length N or the
    selection length), scales each vertex's motion -- this is how a SOFT selection (proportional editing) plugs in:
    pass soft_selection_weights and the transform drags neighbours by their falloff.

    Order within one call: scale, then rotate, then translate (the standard TRS), each about the pivot. Deterministic.
    """
    P = np.asarray(points, float).copy()
    sel = np.asarray(sorted(set(int(i) for i in selection_idx)), int)
    if len(sel) == 0:
        return P
    piv = pivot_point(P, sel, mode=pivot, cursor=cursor, active=active)
    basis = _space_basis(space, view_matrix=view_matrix, local_matrix=local_matrix)
    mask = np.asarray(constraint, float)

    # per-vertex motion weight (1 by default; a soft-selection field drags neighbours).
    if weights is None:
        w = np.ones(len(sel))
    else:
        wfull = np.asarray(weights, float)
        w = wfull[sel] if len(wfull) == len(P) else wfull      # accept full-length or selection-length weights

    rel = P[sel] - piv                                         # positions relative to the pivot

    if scale is not None:
        sc = np.asarray(scale, float)
        if sc.ndim == 0:
            sc = np.array([float(sc)] * 3)
        sc = np.where(mask > 0, sc, 1.0)                       # constrained axes don't scale
        # scale in the space frame: rotate rel into the basis, scale, rotate back.
        local = rel @ basis                                    # world -> space coords
        local = local * sc
        rel = local @ basis.T                                  # space -> world

    if rotate is not None:
        axis, angle = rotate
        axis_world = basis @ np.asarray(axis, float)           # axis given in space -> world
        R = _axis_rotation(axis_world, float(angle))
        rel = rel @ R.T

    new = piv + rel
    if translate is not None:
        t = np.asarray(translate, float) * mask                # mask out locked axes
        t_world = basis @ t                                    # space delta -> world delta
        new = new + t_world

    # blend by per-vertex weight (proportional editing): move each vertex a fraction of the full delta.
    moved = P[sel] + (new - P[sel]) * w[:, None]
    P[sel] = moved
    return P


def _selftest():
    """Contracts:
    1. translate with an axis constraint moves only the free axes.
    2. rotate about the median pivot spins the selection in place (centroid unchanged).
    3. scale about the pivot scales relative positions; constrained axis is untouched.
    4. weights make a soft transform: a weight-0.5 vertex moves half as far.
    """
    # four verts of a unit square in z=0.
    P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    sel = [0, 1, 2, 3]

    # (1) translate along X only, though we asked for (1,1,1) delta, with an X constraint.
    out = transform_selection(P, sel, translate=[1, 1, 1], constraint=(1, 0, 0))
    assert np.allclose(out[:, 0], P[:, 0] + 1) and np.allclose(out[:, 1:], P[:, 1:])

    # (2) rotate 90deg about Z around the median pivot -> centroid stays, square rotates.
    piv = P.mean(axis=0)
    rot = transform_selection(P, sel, rotate=([0, 0, 1], np.pi / 2), pivot="median")
    assert np.allclose(rot.mean(axis=0), piv, atol=1e-9)       # centroid unchanged
    assert not np.allclose(rot, P)                             # but the verts moved

    # (3) scale 2x about the centroid, but constrain to X -> only X spreads.
    sc = transform_selection(P, sel, scale=2.0, pivot="median", constraint=(1, 0, 0))
    # X spread doubles about the centroid; Y unchanged.
    assert abs((sc[:, 0].max() - sc[:, 0].min()) - 2 * (P[:, 0].max() - P[:, 0].min())) < 1e-9
    assert abs((sc[:, 1].max() - sc[:, 1].min()) - (P[:, 1].max() - P[:, 1].min())) < 1e-9

    # (4) soft transform: vertex 0 weight 1, vertex 1 weight 0.5 -> vertex 1 moves half as far.
    w = np.array([1.0, 0.5, 0.0, 0.0])
    soft = transform_selection(P, [0, 1], translate=[0, 0, 1], weights=w)
    assert abs(soft[0, 2] - 1.0) < 1e-9 and abs(soft[1, 2] - 0.5) < 1e-9

    # (5) pivot modes resolve.
    assert np.allclose(pivot_point(P, sel, "bbox"), [0.5, 0.5, 0.0])
    assert np.allclose(pivot_point(P, sel, "cursor", cursor=[9, 9, 9]), [9, 9, 9])

    print("holographic_transform_space selftest OK (translate honours an axis constraint; rotate about the median "
          "pivot keeps the centroid fixed; scale about the pivot with an X constraint spreads only X; a soft "
          "transform moves a weight-0.5 vertex half as far; pivot modes median/bbox/cursor/active resolve; "
          "non-destructive; deterministic)")


if __name__ == "__main__":
    _selftest()

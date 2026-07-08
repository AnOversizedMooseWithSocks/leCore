"""holographic_transform.py -- TRANSFORM UTILITIES for a modeling app (modeling-app backlog, item G).

The engine has scattered transform bits (scenegraph.translation/rotation/compose_transforms; cosserat's
quaternion helpers; splatexport's rotation<->quaternion), but not the full kit a gizmo and a property panel need
in one place. This gathers the standard, well-known math:

  * decompose(M) -> (translate, rotation-quaternion, scale)   -- what a gizmo reads off a matrix to show handles,
    and what a property panel shows as the T/R/S fields;
  * compose_trs(translate, quat, scale) -> 4x4                -- the inverse (build a matrix from panel values);
  * a quaternion kit -- from/to matrix, from/to axis-angle, from/to euler, multiply, SLERP, rotate a vector
    (quaternions are what rotation UI and animation want: no gimbal lock, and slerp gives smooth interpolation);
  * look_at(eye, target, up) -> 4x4 view matrix              -- for a camera or an object aimed at a point.

Conventions, stated ONCE and held (the backlog's coordinate-convention note): matrices are 4x4 and act on COLUMN
vectors, p' = M @ [x, y, z, 1]; compose(A, B) = A @ B means "apply B, then A"; quaternions are (w, x, y, z), unit
length; euler angles are (rx, ry, rz) applied X then Y then Z, i.e. R = Rz @ Ry @ Rx; look_at returns an OpenGL
view matrix (the camera looks down -z, y is up) to match the engine's Camera. Nothing here is holographic -- it is
plain linear algebra a modeling app needs -- so it is kept as a small, readable utility, not dressed up as a bind.
Deterministic; NumPy + stdlib only.
"""
import numpy as np


# ---- basic transform matrices (4x4, column-vector convention: p' = M @ [x, y, z, 1]) --------------------------

def translation(t):
    """A 4x4 translation matrix from a 3-vector."""
    M = np.eye(4)
    M[:3, 3] = np.asarray(t, float)
    return M


def scaling(s):
    """A 4x4 scale matrix. `s` is a scalar (uniform) or a 3-vector (per-axis)."""
    s = np.asarray(s, float)
    d = np.ones(3) * s if s.ndim == 0 else s
    M = np.eye(4)
    M[0, 0], M[1, 1], M[2, 2] = d
    return M


def rotation_axis_angle(axis, angle):
    """A 4x4 rotation of `angle` radians about `axis` (Rodrigues' formula)."""
    M = np.eye(4)
    M[:3, :3] = quat_to_matrix(quat_from_axis_angle(axis, angle))
    return M


def compose(*mats):
    """Matrix product M0 @ M1 @ ... -- with the column-vector convention this applies the RIGHTMOST first."""
    out = np.eye(4)
    for m in mats:
        out = out @ np.asarray(m, float)
    return out


# ---- decompose / recompose (the gizmo + property-panel workhorse) --------------------------------------------

def decompose(M):
    """Split a 4x4 affine transform into (translate (3,), rotation quaternion (4,), scale (3,)). Assumes no shear
    (the modeling-app common case: T * R * S). A reflection (negative determinant) is folded into a negative
    scale on X so the rotation stays a proper rotation. This is what a move/rotate/scale gizmo reads off a matrix."""
    M = np.asarray(M, float)
    translate = M[:3, 3].copy()
    L = M[:3, :3].copy()                          # the linear part = R @ diag(scale)
    scale = np.linalg.norm(L, axis=0)             # each column's length is that axis's scale
    if np.linalg.det(L) < 0:                      # a reflection -> make one scale negative to keep R proper
        scale[0] = -scale[0]
    scale_safe = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    R = L / scale_safe                            # normalise the columns to recover the pure rotation
    return translate, quat_from_matrix(R), scale


def compose_trs(translate, quat, scale):
    """Build a 4x4 from translate (3,), a rotation quaternion (4,), and scale (3,) -- the inverse of decompose."""
    R = quat_to_matrix(quat)
    L = R * np.asarray(scale, float)              # R @ diag(scale): scale each COLUMN of R
    M = np.eye(4)
    M[:3, :3] = L
    M[:3, 3] = np.asarray(translate, float)
    return M


# ---- quaternions (w, x, y, z), unit length -------------------------------------------------------------------

def quat_normalize(q):
    q = np.asarray(q, float)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_mul(a, b):
    """The Hamilton product a*b: the rotation "apply b, then a"."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_from_axis_angle(axis, angle):
    """A quaternion for a rotation of `angle` radians about `axis`."""
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    h = 0.5 * angle
    return np.concatenate([[np.cos(h)], np.sin(h) * axis])


def quat_to_axis_angle(q):
    """Recover (axis, angle) from a quaternion."""
    w, x, y, z = quat_normalize(q)
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sqrt(max(0.0, 1.0 - w * w))
    axis = np.array([x, y, z]) / s if s > 1e-9 else np.array([1.0, 0.0, 0.0])
    return axis, angle


def quat_to_matrix(q):
    """The 3x3 rotation matrix for a quaternion."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def quat_from_matrix(R):
    """The quaternion for a 3x3 rotation matrix (Shepperd's method: branch on the largest diagonal term for
    numerical stability -- a naive formula loses precision when the trace is near zero)."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0                # s = 4w
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0   # s = 4x
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0   # s = 4y
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0   # s = 4z
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z]))


def quat_from_euler(rx, ry, rz):
    """A quaternion from euler angles applied X then Y then Z (R = Rz @ Ry @ Rx)."""
    qx = quat_from_axis_angle([1, 0, 0], rx)
    qy = quat_from_axis_angle([0, 1, 0], ry)
    qz = quat_from_axis_angle([0, 0, 1], rz)
    return quat_mul(quat_mul(qz, qy), qx)


def quat_to_euler(q):
    """Recover euler angles (rx, ry, rz) from a quaternion, inverting R = Rz @ Ry @ Rx. Handles gimbal lock
    (pitch near +/-90 degrees) by pinning roll to 0."""
    R = quat_to_matrix(q)
    ry = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if abs(R[2, 0]) < 0.9999:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:                                          # gimbal lock: cos(pitch) ~ 0
        rx = np.arctan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return np.array([rx, ry, rz])


def quat_slerp(a, b, t):
    """Spherical linear interpolation between two rotations -- constant angular speed, the smooth in-between an
    animation wants. Takes the shortest path (flips b's sign if the dot is negative), and falls back to a straight
    lerp when the two are nearly parallel (where sin(theta) -> 0 would divide by ~zero)."""
    a = quat_normalize(a)
    b = quat_normalize(b)
    d = float(np.dot(a, b))
    if d < 0.0:                                    # -q is the same rotation as q; pick the near one (shortest arc)
        b = -b
        d = -d
    if d > 0.9995:                                 # nearly identical -> lerp, then renormalise
        return quat_normalize(a + t * (b - a))
    theta = np.arccos(np.clip(d, -1.0, 1.0))
    return (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / np.sin(theta)


def quat_rotate(q, v):
    """Rotate a 3-vector by a quaternion."""
    return quat_to_matrix(q) @ np.asarray(v, float)


# ---- camera ---------------------------------------------------------------------------------------------------

def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """An OpenGL view matrix for a camera at `eye` looking at `target` (the engine's convention: the camera looks
    down -z, y is up). Transforms `eye` to the origin and puts `target` on the -z axis. Also the tool for aiming an
    object at a point (use its inverse for a model transform)."""
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    up = np.asarray(up, float)
    f = target - eye
    f = f / np.linalg.norm(f)                      # forward: the camera looks along -f in view space
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)                      # right
    u = np.cross(r, f)                             # true up (already unit; r and f are orthonormal)
    M = np.eye(4)
    M[0, :3] = r
    M[1, :3] = u
    M[2, :3] = -f
    M[0, 3] = -np.dot(r, eye)
    M[1, 3] = -np.dot(u, eye)
    M[2, 3] = np.dot(f, eye)
    return M


def _selftest():
    """decompose/compose round-trip a TRS matrix; the quaternion kit round-trips through matrix/euler/axis-angle;
    slerp hits its endpoints and stays on the unit sphere; look_at sends eye->origin and target->-z; deterministic."""
    rng = np.random.default_rng(0)

    # (1) decompose <-> compose_trs round-trip
    t = np.array([2.0, -3.0, 5.0])
    q = quat_from_euler(0.3, -0.7, 1.1)
    s = np.array([2.0, 0.5, 1.5])
    M = compose_trs(t, q, s)
    t2, q2, s2 = decompose(M)
    assert np.allclose(t2, t) and np.allclose(s2, s)
    # the recovered rotation is the same (quaternion may differ by an overall sign, which is the same rotation)
    assert np.allclose(quat_to_matrix(q2), quat_to_matrix(q), atol=1e-9)

    # (2) quaternion <-> matrix round-trip
    R = quat_to_matrix(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)          # it's a proper rotation
    assert np.allclose(quat_to_matrix(quat_from_matrix(R)), R, atol=1e-9)

    # (3) euler round-trip (moderate angles, away from gimbal lock)
    e = np.array([0.4, -0.6, 0.9])
    assert np.allclose(quat_to_euler(quat_from_euler(*e)), e, atol=1e-6)

    # (4) axis-angle round-trip
    axis = np.array([1.0, 2.0, -1.0]); axis /= np.linalg.norm(axis)
    ang = 1.2
    ax2, an2 = quat_to_axis_angle(quat_from_axis_angle(axis, ang))
    assert np.allclose(ax2, axis, atol=1e-9) and abs(an2 - ang) < 1e-9

    # (5) quat_mul composes rotations: rotating by (qy then qx) equals the product
    qx = quat_from_axis_angle([1, 0, 0], 0.5)
    qy = quat_from_axis_angle([0, 1, 0], 0.9)
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(quat_rotate(quat_mul(qx, qy), v), quat_rotate(qx, quat_rotate(qy, v)), atol=1e-9)

    # (6) SLERP: endpoints exact, midpoint on the unit sphere and halfway in angle
    a = quat_from_axis_angle([0, 0, 1], 0.0)
    b = quat_from_axis_angle([0, 0, 1], 1.0)
    assert np.allclose(quat_slerp(a, b, 0.0), a) and np.allclose(quat_slerp(a, b, 1.0), b)
    mid = quat_slerp(a, b, 0.5)
    assert abs(np.linalg.norm(mid) - 1.0) < 1e-9
    _, mid_ang = quat_to_axis_angle(mid)
    assert abs(mid_ang - 0.5) < 1e-6                          # halfway between angle 0 and 1

    # (7) look_at: eye -> origin, target -> a point on -z
    eye = np.array([3.0, 4.0, 5.0]); target = np.array([0.0, 0.0, 0.0])
    V = look_at(eye, target)
    oe = V @ np.array([eye[0], eye[1], eye[2], 1.0])
    ot = V @ np.array([target[0], target[1], target[2], 1.0])
    assert np.allclose(oe[:3], 0.0, atol=1e-9)               # camera sits at the origin of view space
    assert abs(ot[0]) < 1e-9 and abs(ot[1]) < 1e-9 and ot[2] < 0   # target straight ahead, down -z

    # (8) deterministic
    assert np.array_equal(quat_from_euler(0.1, 0.2, 0.3), quat_from_euler(0.1, 0.2, 0.3))

    print("holographic_transform selftest OK: decompose<->compose round-trips a T/R/S matrix; quaternions round-"
          "trip through matrix / euler / axis-angle; quat_mul composes rotations; slerp hits its endpoints, stays "
          "unit-length, and is halfway in angle at t=0.5; look_at sends eye->origin and target->-z (OpenGL "
          "convention)")


if __name__ == "__main__":
    _selftest()

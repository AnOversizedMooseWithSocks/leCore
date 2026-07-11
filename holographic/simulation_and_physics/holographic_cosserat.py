"""holographic_cosserat.py -- H2b: TWIST for hair, via a Cosserat rod with orientation frames.

WHY THIS EXISTS (Hair & Fur backlog, item H2b -- the deferred quality rung)
---------------------------------------------------------------------------
Plain PBD strands (H2) have distance + bend springs: they resist STRETCHING and FLATTENING, but they have no
notion of TWIST, and a curl held only by bend springs tends to sag straight under gravity. The graphics answer
(Kugelstadt-Schomer 2016, "Position and Orientation Based Cosserat Rods"; VIPER 2019) is to give each segment
an ORIENTATION -- a material frame carried as a quaternion -- and constrain the frames, not just the points:

  * a STRETCH-SHEAR coupling keeps each edge aligned with its frame's tangent axis, so rotating a frame rotates
    the strand;
  * a BEND-TWIST constraint drives the relative rotation between consecutive frames back toward its REST value
    (the rest "Darboux vector"), which stores the curl AND the twist -- so a curl springs back instead of
    sagging, and a twist propagates along the fibre.

This is the PBD route to what Discrete Elastic Rods does with a manifold projection, and it rides on the same
predict -> project -> update loop the rest of the softbody uses. Opt-in, because frames cost more and carry a
sign/tie convention (the bind_batch lesson) that plain points don't.

HONEST SCOPE (kept negative): a readable Cosserat rod that holds curl and carries twist -- the frame's roll is
tracked and coupled, but the visible CENTERLINE follows the tangent (roll matters for an oriented cross-section
like a ribbon/elliptical fibre, not a round one). Positions are reconstructed from the frames each step (a
Follow-The-Leader-with-orientations pass), which is stable and inextensible but is the simplified coupling, not
the full simultaneous stretch-shear solve. Deterministic; NumPy + stdlib. Quaternion helpers are written out
here (small and readable) since the engine had none to reuse.
"""
import numpy as np


# --- quaternion helpers (unit quaternions [w, x, y, z]; small, readable, no dependency) --------------------------

def qmul(a, b):
    """Hamilton quaternion product `(w, x, y, z)`. **DELEGATES to `holographic_transform.quat_mul`** -- the engine's
    canonical quaternion kit, which `transformhome.Transform` already routes to.

    A structural duplicate scan found the two bodies identical, and a numeric check confirmed they agree bit for bit
    (0.0e+00). Two implementations of one convention will eventually disagree on a sign, and the one that disagrees
    will be the one nobody is testing."""
    from holographic.misc.holographic_transform import quat_mul
    return quat_mul(a, b)


def _qmul_original(a, b):
    """Hamilton product a (x) b -- compose two rotations."""
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def qconj(q):
    """Conjugate = inverse for a unit quaternion (the opposite rotation)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qnorm(q):
    """Renormalize to a unit quaternion (drift control after corrections)."""
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_from_axis_angle(axis, angle):
    """Rotation of `angle` radians about (a normalized) `axis`. **DELEGATES to
    `holographic_transform.quat_from_axis_angle`** -- the same unification rev. 8 made for `qmul`, finished for its
    sibling constructor. The rev. 9 organization audit found the two bodies structurally identical and a numeric
    check confirmed bit-identity (`np.array_equal`) over 2,000 random (axis, angle) draws AND the degenerate
    zero-axis branch. Two copies of one convention will eventually disagree, and the copy that disagrees will be
    the one nobody is testing."""
    from holographic.misc.holographic_transform import quat_from_axis_angle as _canonical
    return _canonical(axis, angle)


def quat_rotate(q, v):
    """Rotate 3-vector v by unit quaternion q (v' = q v q*)."""
    qv = np.array([0.0, *v])
    return qmul(qmul(q, qv), qconj(q))[1:]


def quat_between(u, v):
    """Shortest rotation taking unit vector u to unit vector v."""
    u = u / (np.linalg.norm(u) + 1e-12); v = v / (np.linalg.norm(v) + 1e-12)
    d = float(np.clip(np.dot(u, v), -1.0, 1.0))
    if d > 1.0 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])                       # already aligned
    if d < -1.0 + 1e-9:                                             # opposite: 180 deg about any perpendicular
        perp = np.cross(u, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(u, [0.0, 1.0, 0.0])
        return quat_from_axis_angle(perp, np.pi)
    axis = np.cross(u, v); angle = np.arccos(d)
    return quat_from_axis_angle(axis, angle)


def _slerp_toward(rel, target, t):
    """Move rotation `rel` a fraction t toward `target` (used to relax bend-twist toward its rest value). A
    short-arc nlerp -- cheaper than full slerp and plenty for a stiffness blend."""
    if np.dot(rel, target) < 0.0:
        target = -target                                            # take the short way round (double cover)
    return qnorm((1.0 - t) * rel + t * target)


E3 = np.array([0.0, 0.0, 1.0])                                      # the frame's tangent director at rest


class CosseratStrand:
    """A hair as a Cosserat rod: points plus a per-segment orientation frame (quaternion). The frames store the
    rest curvature/twist, so the strand HOLDS its curl under gravity and can carry a twist. Root pinned."""

    def __init__(self, points, bend_stiffness=0.5, shape_stiffness=0.6):
        self.x = np.asarray(points, float).copy()                   # (n, 3)
        self.n = len(self.x)
        self.v = np.zeros_like(self.x)
        seg = np.diff(self.x, axis=0)
        self.L = np.linalg.norm(seg, axis=1)                        # rest segment lengths (inextensible target)
        # build a rest frame per segment: rotate the reference tangent E3 onto the rest edge direction, carried
        # by PARALLEL TRANSPORT so neighbouring frames differ only by the strand's own bend (no spurious twist).
        self.q = self._frames_from_edges(seg, parallel_transport=True)
        # rest Darboux = the relative rotation between consecutive rest frames; this is what stores the curl+twist
        self.rest_rel = [qmul(qconj(self.q[i]), self.q[i + 1]) for i in range(self.n - 2)]
        self.bend_stiffness = float(bend_stiffness)
        self.shape_stiffness = float(shape_stiffness)
        self._arc = float(self.L.sum())
        self._rest_extension = float(np.linalg.norm(self.x[-1] - self.x[0])) / max(self._arc, 1e-9)  # rest end/arc

    @staticmethod
    def _frames_from_edges(seg, parallel_transport=True, prev_frames=None):
        """One quaternion per segment whose tangent (d3) points along the edge. With parallel_transport, each
        frame is the previous one rotated minimally onto the new edge (no added roll) -- the natural rest frame.
        With prev_frames given, we instead rotate each EXISTING frame minimally onto its new edge, preserving the
        roll/twist it already carried (used every step so twist is not thrown away)."""
        dirs = seg / (np.linalg.norm(seg, axis=1, keepdims=True) + 1e-12)
        q = []
        for i in range(len(dirs)):
            if prev_frames is not None:
                d3 = quat_rotate(prev_frames[i], E3)                # where this frame currently points
                q.append(qnorm(qmul(quat_between(d3, dirs[i]), prev_frames[i])))
            elif parallel_transport and i > 0:
                prev_d3 = quat_rotate(q[i - 1], E3)
                q.append(qnorm(qmul(quat_between(prev_d3, dirs[i]), q[i - 1])))
            else:
                q.append(quat_between(E3, dirs[i]))
        return q

    def curl_amount(self):
        """0 = straight, higher = curlier: how much shorter the straight-line end-to-end distance is than the
        arc length. A clean scalar for 'did the curl hold'."""
        arc = float(self.L.sum())
        end = float(np.linalg.norm(self.x[-1] - self.x[0]))
        return 1.0 - end / max(arc, 1e-9)

    def twist_of(self, i):
        """The roll (twist) angle carried between segment i and i+1 about the tangent -- the twist DOF the plain
        bend model does not have. Returns radians."""
        rel = qmul(qconj(self.q[i]), self.q[i + 1])
        return 2.0 * np.arctan2(np.linalg.norm(rel[1:]), abs(rel[0])) * np.sign(rel[0] if rel[0] != 0 else 1.0)

    def set_root_twist(self, angle):
        """Twist the first frame about its own tangent by `angle` -- a torque at the root that the bend-twist
        constraint will then propagate down the strand."""
        d3 = quat_rotate(self.q[0], E3)
        self.q[0] = qnorm(qmul(quat_from_axis_angle(d3, angle), self.q[0]))

    def step(self, dt=1.0 / 60.0, gravity=(0.0, -9.8, 0.0), external_force=None, iters=8, damping=0.02):
        """One Cosserat step: predict under gravity/wind, keep segments inextensible, align frames to the edges
        (preserving their roll), relax the bend-twist toward the REST curvature, then reconstruct the centerline
        from the corrected frames -- which is what springs the curl back. Root stays pinned."""
        g = np.asarray(gravity, float)
        x_prev = self.x.copy()
        # 1. predict positions (root pinned)
        acc = np.tile(g, (self.n, 1))
        if external_force is not None:
            acc = acc + np.asarray(external_force, float)
        self.v *= (1.0 - damping)
        self.v[1:] += dt * acc[1:]
        self.x[1:] += dt * self.v[1:]
        # 2. inextensibility: Follow-The-Leader from the pinned root (each point at rest distance from the last)
        for _ in range(iters):
            for i in range(1, self.n):
                d = self.x[i] - self.x[i - 1]
                self.x[i] = self.x[i - 1] + d / (np.linalg.norm(d) + 1e-12) * self.L[i - 1]
        # 3. align frames to the (moved) edges, PRESERVING each frame's roll so twist is not lost
        seg = np.diff(self.x, axis=0)
        self.q = self._frames_from_edges(seg, prev_frames=self.q)
        # 4. bend-twist: pull each joint's relative rotation back toward its rest value (stores curl + twist)
        for i in range(self.n - 2):
            rel = qmul(qconj(self.q[i]), self.q[i + 1])
            relaxed = _slerp_toward(rel, self.rest_rel[i], self.bend_stiffness)
            self.q[i + 1] = qnorm(qmul(self.q[i], relaxed))         # move the next frame toward rest-relative
        # 5. reconstruct the centerline from the corrected frames (the curl-restoring feedback), blended with the
        #    gravity-bent positions. The curl memory is TENSION-AWARE: when the strand is pulled toward full
        #    extension (a taut fibre), it cannot hold its curl, so the reconstruction weight fades to zero.
        extension = float(np.linalg.norm(self.x[-1] - self.x[0])) / max(self._arc, 1e-9)
        slack = np.clip((1.0 - extension) / max(1.0 - self._rest_extension, 1e-9), 0.0, 1.0)  # 1 at rest, 0 when taut
        eff_shape = self.shape_stiffness * slack
        rebuilt = self.x.copy()
        for i in range(self.n - 1):
            rebuilt[i + 1] = rebuilt[i] + self.L[i] * quat_rotate(self.q[i], E3)
        self.x[1:] = (1.0 - eff_shape) * self.x[1:] + eff_shape * rebuilt[1:]
        # blending two configurations can nudge segment lengths, so re-assert inextensibility once more (FTL)
        for i in range(1, self.n):
            d = self.x[i] - self.x[i - 1]
            self.x[i] = self.x[i - 1] + d / (np.linalg.norm(d) + 1e-12) * self.L[i - 1]
        # 6. velocities from the actual position change
        self.v = (self.x - x_prev) / dt
        self.v[0] = 0.0
        return self

    def settle(self, steps=80, **kw):
        for _ in range(int(steps)):
            self.step(**kw)
        return self


def from_strand(strand, bend_stiffness=0.5, shape_stiffness=0.6):
    """Build a CosseratStrand from a groom Strand (H1/H2)."""
    return CosseratStrand(strand.points, bend_stiffness=bend_stiffness, shape_stiffness=shape_stiffness)


def _selftest():
    """Quaternion helpers are sane; a curly Cosserat rod HOLDS its curl under gravity far better than a plain
    inextensible chain with no bend-twist; pulling the tip straight UN-CURLS it; a root twist propagates down
    the frames; deterministic."""
    # (0) quaternion sanity: rotate E3 by 90 deg about x -> -y-ish; round-trip with conjugate
    q = quat_from_axis_angle([1, 0, 0], np.pi / 2)
    assert np.allclose(quat_rotate(q, [0, 0, 1]), [0, -1, 0], atol=1e-6)
    assert np.allclose(quat_rotate(qconj(q), quat_rotate(q, [0.3, 0.4, 0.5])), [0.3, 0.4, 0.5], atol=1e-6)

    # a curly rest strand (a helix), rooted at the origin, growing up
    n = 16
    s = np.linspace(0, 1, n)
    curl_r = 0.12
    pts = np.stack([curl_r * (np.cos(2 * np.pi * 2 * s) - 1.0) * s,
                    s * 0.8,
                    curl_r * np.sin(2 * np.pi * 2 * s) * s], axis=1)
    rest_curl = 1.0 - np.linalg.norm(pts[-1] - pts[0]) / np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()

    # (1) the Cosserat rod holds its curl under gravity -- its shape stays close to the REST curl, whereas a
    # plain inextensible chain with no bend-twist drifts away from it (it sags/bunches into a different shape)
    rod = CosseratStrand(pts, bend_stiffness=0.6, shape_stiffness=0.7)
    rod.settle(steps=120, gravity=(0.0, -9.8, 0.0))
    held = rod.curl_amount()
    plain = CosseratStrand(pts, bend_stiffness=0.0, shape_stiffness=0.0)   # no frame constraints = plain chain
    plain.settle(steps=120, gravity=(0.0, -9.8, 0.0))
    sagged = plain.curl_amount()
    assert abs(held - rest_curl) < abs(sagged - rest_curl)         # Cosserat preserves the rest curl better
    assert held > 0.5 * rest_curl                                  # and keeps most of the curl

    # (2) STRETCHING it taut un-curls it: pin the root and pull the tip to near-full extension; the curl opens
    rod2 = CosseratStrand(pts, bend_stiffness=0.6, shape_stiffness=0.7)
    axis = (pts[-1] - pts[0]); axis = axis / np.linalg.norm(axis)
    target = pts[0] + axis * 0.96 * rod2._arc                       # almost fully extended
    for _ in range(150):
        rod2.step(gravity=(0.0, 0.0, 0.0))
        rod2.x[-1] = target; rod2.v[-1] = 0.0; rod2.x[0] = pts[0]; rod2.v[0] = 0.0
    assert rod2.curl_amount() < 0.5 * rest_curl                     # pulled taut, it straightened out

    # (3) TWIST DOF: twisting the root frame propagates roll down the strand (plain bend has no such DOF)
    rod3 = CosseratStrand(pts, bend_stiffness=0.8, shape_stiffness=0.5)
    tw_before = abs(rod3.twist_of(n // 2))
    rod3.set_root_twist(1.2)
    for _ in range(40):
        rod3.step(gravity=(0.0, 0.0, 0.0))
    assert abs(rod3.twist_of(n // 2)) > tw_before                   # twist travelled down the frames

    # (4) deterministic
    a = CosseratStrand(pts).settle(steps=20); b = CosseratStrand(pts).settle(steps=20)
    assert np.array_equal(a.x, b.x)
    print("holographic_cosserat selftest OK: orientation frames hold a curl under gravity (%.2f vs plain chain "
          "%.2f, rest %.2f); tip tension un-curls it; a root twist propagates down the frames; deterministic"
          % (held, sagged, rest_curl))


if __name__ == "__main__":
    _selftest()

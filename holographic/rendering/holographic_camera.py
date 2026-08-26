"""holographic_camera.py -- the CAMERA CONTROLLER: viewport navigation (modeling-app feature layer).

The interactive controls a modeling viewport needs -- ORBIT (turntable), PAN, DOLLY, ZOOM, and FRAME-a-box --
built on the item-G transform utilities (look_at, quaternions). It holds eye / target / up and hands back either a
4x4 view matrix (look_at) or a render Camera. Nothing holographic here; it is the plain vector math of moving a
camera around a pivot, written to be read.

Conventions (item G's): the camera looks from `eye` toward `target`, forward = normalize(target - eye), and the
view basis is right = forward x up, view-up = right x forward. Orbit keeps the eye on a sphere around the target
(distance preserved, a rotation), clamped near the poles so a turntable never flips upside down. Deterministic;
NumPy + stdlib only.
"""
import numpy as np

from holographic.misc.holographic_transform import look_at, quat_from_axis_angle, quat_rotate


def _norm(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


class CameraController:
    """Orbit / pan / dolly / zoom / frame around a target. Mutates eye/target in place; read view_matrix() or
    to_camera() for the result."""

    def __init__(self, eye=(0.0, 0.0, 5.0), target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0), min_distance=1e-3):
        self.eye = np.asarray(eye, float)
        self.target = np.asarray(target, float)
        self.up = _norm(up)                                  # the WORLD up (the turntable axis)
        self.min_distance = min_distance

    # -- derived view basis --------------------------------------------------------------------------------
    @property
    def distance(self):
        """The eye-to-target distance (the orbit radius)."""
        return float(np.linalg.norm(self.eye - self.target))

    def _forward(self):
        return _norm(self.target - self.eye)                 # the camera looks along +forward

    def _right(self):
        return _norm(np.cross(self._forward(), self.up))

    def _view_up(self):
        return np.cross(self._right(), self._forward())      # orthonormal, so already unit length

    # -- orbit: turntable rotation about the target -------------------------------------------------------
    def orbit(self, d_azimuth, d_elevation, elevation_limit=np.radians(89.0)):
        """Rotate the eye around the target: `d_azimuth` about the world up axis, `d_elevation` about the current
        right axis. The distance is preserved (it's a rotation of the offset), and the elevation is CLAMPED to
        +/- elevation_limit so a turntable never tips over the pole (the classic orbit-camera gimbal guard)."""
        offset = self.eye - self.target

        # azimuth: spin the offset around the world up axis
        offset = quat_rotate(quat_from_axis_angle(self.up, d_azimuth), offset)

        # elevation: tilt around the right axis, but first clamp so we stay below the pole
        right = _norm(np.cross(_norm(self.target - (self.target + offset)), self.up))  # right for this offset
        cur_elev = float(np.arcsin(np.clip(np.dot(_norm(offset), self.up), -1.0, 1.0)))
        new_elev = float(np.clip(cur_elev + d_elevation, -elevation_limit, elevation_limit))
        offset = quat_rotate(quat_from_axis_angle(right, new_elev - cur_elev), offset)

        self.eye = self.target + offset

    # -- pan: slide eye AND target together in the view plane ---------------------------------------------
    def pan(self, dx, dy):
        """Slide the camera in its own right/up plane -- both eye and target move by the same vector, so the view
        direction and distance are unchanged (a translation of the whole rig)."""
        shift = dx * self._right() + dy * self._view_up()
        self.eye = self.eye + shift
        self.target = self.target + shift

    # -- dolly: move the eye along the view direction (target fixed) --------------------------------------
    def dolly(self, distance):
        """Move the eye toward (+) or away from (-) the target along the view direction. Won't cross the target
        (keeps min_distance) -- dollying past your pivot is never what you want."""
        f = self._forward()
        new_eye = self.eye + f * distance
        if float(np.dot(self.target - new_eye, f)) > 0.0 and np.linalg.norm(self.target - new_eye) >= self.min_distance:
            self.eye = new_eye

    # -- zoom: scale the eye-target distance --------------------------------------------------------------
    def zoom(self, factor):
        """Scale the orbit radius by `factor` (0<factor<1 moves closer, >1 pulls back). Keeps min_distance."""
        offset = (self.eye - self.target) * factor
        if np.linalg.norm(offset) >= self.min_distance:
            self.eye = self.target + offset

    # -- frame: fit an axis-aligned box in view -----------------------------------------------------------
    def frame(self, bbox_min, bbox_max, fov_deg=45.0):
        """Aim at the box centre and back off just far enough that its bounding SPHERE fills the vertical field of
        view: distance = radius / sin(fov/2). Keeps the current viewing direction. This is the 'frame selected' /
        'zoom to fit' command."""
        lo = np.asarray(bbox_min, float)
        hi = np.asarray(bbox_max, float)
        center = 0.5 * (lo + hi)
        radius = 0.5 * float(np.linalg.norm(hi - lo))
        f = self._forward()                                  # keep looking the same way
        dist = radius / np.sin(np.radians(fov_deg / 2.0)) if radius > 1e-12 else self.distance
        self.target = center
        self.eye = center - f * dist                         # back off along -forward so we look AT the centre

    # -- output -------------------------------------------------------------------------------------------
    def view_matrix(self):
        """The 4x4 OpenGL view matrix for the current pose (via item-G look_at)."""
        return look_at(self.eye, self.target, self.up)

    def to_camera(self, fov_deg=45.0, aspect=1.0):
        """A render Camera at the current pose (for handing straight to the path tracer / session)."""
        from holographic.rendering.holographic_render import Camera
        return Camera(eye=tuple(self.eye), target=tuple(self.target), up=tuple(self.up),
                      fov_deg=fov_deg, aspect=aspect)


def _selftest():
    """Orbit preserves the target distance and wraps at 360 degrees; elevation clamps at the pole; pan moves eye
    and target together (distance unchanged); dolly shortens the distance without crossing the target; zoom scales
    it; frame fits a box's bounding sphere; the view matrix sends the target down -z; deterministic."""
    cam = CameraController(eye=(0, 0, 5), target=(0, 0, 0))

    # (1) orbit preserves distance; a full 360-degree azimuth returns to the start
    d0 = cam.distance
    cam.orbit(np.radians(90), 0.0)
    assert abs(cam.distance - d0) < 1e-9                     # a rotation preserves the radius
    assert not np.allclose(cam.eye, [0, 0, 5])              # ...but the eye actually moved
    cam.orbit(np.radians(270), 0.0)
    assert np.allclose(cam.eye, [0, 0, 5], atol=1e-9)       # 90 + 270 = 360 -> back to start

    # (2) elevation clamps at the pole (no flip-over)
    cam.orbit(0.0, np.radians(200))                         # ask for way past vertical
    elev = np.arcsin(np.clip(np.dot(_norm(cam.eye - cam.target), cam.up), -1, 1))
    assert elev <= np.radians(89.0) + 1e-6                   # clamped, not flipped

    # (3) pan slides eye AND target together -> distance and direction unchanged
    cam = CameraController(eye=(0, 0, 5), target=(0, 0, 0))
    f0 = cam._forward().copy(); d0 = cam.distance
    cam.pan(2.0, 1.0)
    assert abs(cam.distance - d0) < 1e-9 and np.allclose(cam._forward(), f0, atol=1e-9)
    assert np.allclose(cam.target, cam.eye + f0 * d0, atol=1e-9)

    # (4) dolly moves closer but never crosses the target
    cam = CameraController(eye=(0, 0, 5), target=(0, 0, 0))
    cam.dolly(2.0)
    assert abs(cam.distance - 3.0) < 1e-9                    # 5 - 2 = 3
    cam.dolly(100.0)                                         # would overshoot -> refused
    assert cam.distance >= cam.min_distance

    # (5) zoom scales the distance
    cam = CameraController(eye=(0, 0, 4), target=(0, 0, 0))
    cam.zoom(0.5)
    assert abs(cam.distance - 2.0) < 1e-9

    # (6) frame fits a box: target at centre, distance = radius / sin(fov/2)
    cam = CameraController(eye=(0, 0, 10), target=(0, 0, 0))
    cam.frame([0, 0, 0], [2, 2, 2], fov_deg=45.0)
    center = np.array([1, 1, 1]); radius = 0.5 * np.linalg.norm([2, 2, 2])
    assert np.allclose(cam.target, center, atol=1e-9)
    assert abs(cam.distance - radius / np.sin(np.radians(22.5))) < 1e-9

    # (7) the view matrix sends the target down -z (looks at it)
    V = cam.view_matrix()
    ot = V @ np.array([center[0], center[1], center[2], 1.0])
    assert abs(ot[0]) < 1e-9 and abs(ot[1]) < 1e-9 and ot[2] < 0

    # (8) deterministic
    c1 = CameraController(); c1.orbit(0.3, 0.1)
    c2 = CameraController(); c2.orbit(0.3, 0.1)
    assert np.array_equal(c1.eye, c2.eye)

    print("holographic_camera selftest OK: orbit preserves the target distance and wraps at 360 degrees; "
          "elevation clamps at the pole (no flip); pan moves eye+target together (distance/direction unchanged); "
          "dolly shortens the distance without crossing the target; zoom scales it; frame fits a box's bounding "
          "sphere (distance = radius/sin(fov/2)); the view matrix looks at the target; deterministic")


if __name__ == "__main__":
    _selftest()

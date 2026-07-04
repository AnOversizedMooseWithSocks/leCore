"""holographic_lights -- placed light objects and NEXT-EVENT ESTIMATION for the path tracer.

Until now the path tracer only got light when a bounce ray happened to escape and hit the emissive SKY. That works
for a big bright environment, but a small bright lamp is almost never hit by a random bounce, so it shows up as
extreme noise (a few pixels that got lucky, the rest black). The fix every renderer uses is NEXT-EVENT ESTIMATION
(NEE), also called direct light sampling:

    at each diffuse surface hit, instead of only bouncing randomly and hoping, we ALSO look STRAIGHT AT each light,
    cast a short "shadow ray" to check nothing is in the way, and if the light is visible add its contribution
    directly. Small bright lights then converge almost instantly, and we get real, correctly-shaped shadows.

This module provides the light objects and the direct-lighting evaluation. The path tracer calls `direct_lighting`
once per surface hit (in addition to its existing BRDF bounce, which still carries the INDIRECT light -- colour
bleeding, ambient in concavities -- so nothing is lost).

The full set of light types (what a real DCC rig has):

  * PointLight       -- an infinitely small point. 1/distance^2 falloff, hard shadows. The simplest lamp.
  * DirectionalLight -- the sun: parallel rays from a fixed direction, no falloff, hard shadows.
  * AmbientLight     -- a constant fill from every direction. No shadow ray (it comes from everywhere); it just
                        lifts the whole scene by albedo * colour, the classic ambient term.
  * SpotLight        -- a point light inside a CONE: full brightness within an inner angle, fading to zero at an
                        outer angle. Can carry a GOBO (a projected pattern / light cookie).
  * RectLight        -- a rectangular AREA emitter (a softbox / panel). One-sided; soft shadows; can carry a gobo.
  * SphereLight      -- a glowing sphere. AREA -> soft shadows (a penumbra).
  * MeshLight        -- an emissive TRIANGLE MESH: sample a triangle by area, then a point on it. Soft shadows
                        shaped by the mesh -- any emitter geometry you like.
  * IESLight         -- a point light with a REAL-WORLD measured beam shape (an IES photometric profile): the
                        intensity varies with angle from the aim direction, from a candela table. `load_ies`
                        parses a standard .ies file into that profile.

FIELDS / MAPS (parameters that vary): every light's `color` and `intensity` may be a constant OR a FIELD -- a
callable f(points)->values that varies the parameter across the scene (a coloured-gradient lamp, an intensity map).
The projectors (SpotLight, RectLight) additionally take a `gobo` -- a callable over the projected coordinate that
multiplies the light, i.e. a light cookie / projected slide. See `_resolve` and each light's `sample`.

The maths (the rendering equation's direct term, one light):
    Lo += throughput * f_r(V, L) * L_light * cos(theta) * visibility / pdf
where f_r is the surface BRDF (holographic_brdf.cook_torrance), cos(theta) = N.L, visibility is 0/1 from the
shadow ray, and area lights sample a point on the emitter.

Everything is NumPy + stdlib. Deterministic given the tracer's seeded rng.
"""

import numpy as np
from holographic_raymarch import sphere_trace
from holographic_brdf import cook_torrance


# =====================================================================================================
# helpers -- shared by all the light types
# =====================================================================================================
def _unit(v):
    """Normalise a vector (or rows of an array) to unit length."""
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _resolve(param, P):
    """A light parameter (colour or intensity) may be a plain constant OR a FIELD -- a callable f(points)->values
    that varies it across the scene (a coloured gradient lamp, an intensity map, ...). Return the concrete value at
    the shade points P (M,3). A constant just passes through; a callable is evaluated at P."""
    if callable(param):
        return np.asarray(param(P), float)
    return np.asarray(param, float)


def _emit(color, intensity, P):
    """Combine a light's colour and intensity into a per-point emitted radiance (M,3). Either may be a constant or
    a field (see _resolve); this broadcasts them together so a scalar colour+intensity gives one RGB for all
    points, while a colour-field or intensity-field varies per point."""
    M = len(P)
    col = _resolve(color, P)
    if col.ndim == 1:                                            # one RGB for all points -> broadcast
        col = np.broadcast_to(col, (M, 3)).astype(float)
    inten = _resolve(intensity, P)
    if inten.ndim == 0:                                          # one scalar intensity
        return col * float(inten)
    return col * inten[:, None]                                  # per-point intensity field


def _tangent_frame(N):
    """Two orthonormal vectors (T, B) perpendicular to each normal in N (k,3). Used to spread samples across a
    disk or a hemisphere in the surface's own frame."""
    N = np.asarray(N, float)
    # pick a helper axis that isn't nearly parallel to N, so the cross product is well-conditioned
    helper = np.where(np.abs(N[:, 1:2]) < 0.99, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    T = _unit(np.cross(helper, N))
    B = np.cross(N, T)
    return T, B


def _cosine_hemisphere(N, rng):
    """Directions drawn COSINE-weighted about each normal in N (k,3) -> (k,3) unit dirs. Cosine weighting is the
    right importance sampling for a Lambertian surface: the cosine in the rendering equation and the sample pdf
    cancel, so a dome-light estimate is simply albedo * radiance averaged over these directions."""
    k = len(N)
    u1 = rng.random(k); u2 = rng.random(k)
    r = np.sqrt(u1); phi = 2.0 * np.pi * u2
    x = r * np.cos(phi); y = r * np.sin(phi); z = np.sqrt(np.clip(1.0 - u1, 0.0, 1.0))
    T, B = _tangent_frame(N)
    return x[:, None] * T + y[:, None] * B + z[:, None] * np.asarray(N, float)


# =====================================================================================================
# the light types
# =====================================================================================================
class PointLight:
    """A point light at `position` radiating `color * intensity`. Falls off as 1/distance^2 (physically correct
    for a point source). Hard shadows (it has no area). `color`/`intensity` may be fields (callables of P)."""

    def __init__(self, position=(2.0, 3.0, 2.0), color=(1.0, 1.0, 1.0), intensity=8.0):
        self.position = np.asarray(position, float)
        self.color = color
        self.intensity = intensity

    def sample(self, P, rng):
        """From shade points P (k,3), return (L_dir (k,3) toward the light, dist (k,), radiance (k,3)).
        `rng` is unused for a point light (nothing random) but kept for a uniform interface with area lights."""
        to_light = self.position[None, :] - P                   # (k,3) vector to the lamp
        dist = np.linalg.norm(to_light, axis=1)                 # (k,)
        L = to_light / (dist[:, None] + 1e-9)                   # unit direction to the light
        radiance = _emit(self.color, self.intensity, P) / (dist[:, None] ** 2 + 1e-6)   # 1/r^2 falloff
        return L, dist, radiance


class DirectionalLight:
    """The sun: parallel rays coming FROM `direction` (so light travels along -direction). No distance falloff;
    hard shadows. `direction` is the direction TO the light (up toward the sun)."""

    def __init__(self, direction=(0.4, 0.8, 0.5), color=(1.0, 1.0, 1.0), intensity=3.0):
        self.direction = _unit(direction)
        self.color = color
        self.intensity = intensity

    def sample(self, P, rng):
        k = len(P)
        L = np.tile(self.direction, (k, 1))                     # same direction everywhere (parallel rays)
        dist = np.full(k, 1e9)                                  # effectively infinite -> shadow ray goes far
        radiance = _emit(self.color, self.intensity, P)        # no falloff
        return L, dist, radiance


class AmbientLight:
    """A constant fill light -- light arriving equally from every direction (a cheap stand-in for skylight / global
    fill). It casts NO shadow (it comes from everywhere), so it is handled specially in direct_lighting: it simply
    adds albedo * colour * intensity, the classic ambient term. Use it to lift the darkest shadows off pure black."""

    is_ambient = True                                           # direct_lighting checks this and skips the shadow ray

    def __init__(self, color=(1.0, 1.0, 1.0), intensity=0.1):
        self.color = color
        self.intensity = intensity

    def fill(self, P):
        """The ambient irradiance at points P (k,3) -> (k,3). Multiplied by albedo in direct_lighting."""
        return _emit(self.color, self.intensity, P)


class SpotLight:
    """A point light restricted to a CONE aimed along `direction`. Full brightness within `inner_deg` of the axis,
    smoothly fading to zero at `outer_deg` (the penumbra of the cone). 1/distance^2 falloff, hard shadows at the
    geometry. Optional `gobo`: a callable f(uv)->multiplier over the cone's cross-section (uv in [-1,1]^2), i.e. a
    projected pattern / light cookie -- shine a shaped or textured beam. color/intensity may be fields."""

    def __init__(self, position=(0.0, 3.0, 0.0), direction=(0.0, -1.0, 0.0), color=(1.0, 1.0, 1.0), intensity=20.0,
                 inner_deg=15.0, outer_deg=25.0, gobo=None):
        self.position = np.asarray(position, float)
        self.direction = _unit(direction)                      # the way the beam points (down the cone)
        self.color = color
        self.intensity = intensity
        self.cos_inner = float(np.cos(np.radians(inner_deg)))   # full brightness inside this angle
        self.cos_outer = float(np.cos(np.radians(outer_deg)))   # zero brightness outside this angle
        self.gobo = gobo
        # a local frame perpendicular to the beam, so a gobo has stable (u,v) coordinates across the cone. For a
        # near-vertical beam we fall back to world +z as "up" so the gobo's u-axis lands on world +x (what you'd
        # expect: for a downlight, moving the pattern's u moves it along world x).
        up = np.array([0.0, 1.0, 0.0]) if abs(self.direction[1]) < 0.99 else np.array([0.0, 0.0, 1.0])
        self._u = _unit(np.cross(up, self.direction))
        self._v = _unit(np.cross(self.direction, self._u))

    def sample(self, P, rng):
        to_light = self.position[None, :] - P
        dist = np.linalg.norm(to_light, axis=1)
        L = to_light / (dist[:, None] + 1e-9)                   # direction from the point TO the light
        beam = -L                                               # direction from the light toward the point
        cos_ang = np.sum(beam * self.direction[None, :], axis=1)   # how close to the cone axis this point is
        # smooth cone attenuation: 1 inside the inner angle, 0 outside the outer angle, smoothstep between
        t = np.clip((cos_ang - self.cos_outer) / (self.cos_inner - self.cos_outer + 1e-9), 0.0, 1.0)
        cone = t * t * (3.0 - 2.0 * t)                          # smoothstep for a soft-edged spot
        radiance = _emit(self.color, self.intensity, P) / (dist[:, None] ** 2 + 1e-6)
        radiance = radiance * cone[:, None]
        if self.gobo is not None:                              # project a pattern across the cone cross-section
            # (u,v) of the beam direction in the cone's local frame, scaled by the outer cone size -> ~[-1,1]
            bu = np.sum(beam * self._u[None, :], axis=1)
            bv = np.sum(beam * self._v[None, :], axis=1)
            scale = np.sqrt(max(1.0 - self.cos_outer * self.cos_outer, 1e-6))
            uv = np.stack([bu / scale, bv / scale], axis=1)     # (k,2) light-cookie coordinates
            g = np.asarray(self.gobo(uv), float)                # (k,) or (k,3) multiplier
            radiance = radiance * (g[:, None] if g.ndim == 1 else g)
        return L, dist, radiance


class RectLight:
    """A rectangular AREA light (a softbox / panel) centred at `position`, spanning +/- `u_vec` and +/- `v_vec`
    (its two half-edges). It emits from ONE side (the normal = u_vec x v_vec), so it doesn't leak out the back.
    Having area, it gives SOFT shadows: we sample a random point on the panel each call and averaging over samples
    gives the penumbra. Optional `gobo`: a callable f(uv)->multiplier over the panel (uv in [-1,1]^2) -- a textured
    or shaped panel. color/intensity may be fields."""

    def __init__(self, position=(0.0, 3.0, 0.0), u_vec=(1.0, 0.0, 0.0), v_vec=(0.0, 0.0, 1.0),
                 color=(1.0, 1.0, 1.0), intensity=30.0, gobo=None):
        self.position = np.asarray(position, float)
        self.soft = True                                       # area source -> soft penumbra; NEE multi-samples it
        self.u_vec = np.asarray(u_vec, float)                  # half-width edge
        self.v_vec = np.asarray(v_vec, float)                  # half-height edge
        self.normal = _unit(np.cross(self.u_vec, self.v_vec))  # the emitting face direction
        self.color = color
        self.intensity = intensity
        self.gobo = gobo

    def sample(self, P, rng):
        k = len(P)
        a = rng.uniform(-1.0, 1.0, k)                           # random position across the panel width
        b = rng.uniform(-1.0, 1.0, k)                           # ... and height
        sample_pt = self.position[None, :] + a[:, None] * self.u_vec[None, :] + b[:, None] * self.v_vec[None, :]
        to_light = sample_pt - P
        dist = np.linalg.norm(to_light, axis=1)
        L = to_light / (dist[:, None] + 1e-9)
        # ONE-SIDED: the panel only emits toward points in front of its normal. cos on the light's own face:
        cos_light = np.clip(np.sum((-L) * self.normal[None, :], axis=1), 0.0, 1.0)
        radiance = _emit(self.color, self.intensity, P) / (dist[:, None] ** 2 + 1e-6)
        radiance = radiance * cos_light[:, None]               # dark from behind / edge-on
        if self.gobo is not None:                              # a pattern painted on the panel
            g = np.asarray(self.gobo(np.stack([a, b], axis=1)), float)
            radiance = radiance * (g[:, None] if g.ndim == 1 else g)
        return L, dist, radiance


class DiskLight:
    """A round AREA light (a disk) -- the very common circular softbox / lamp. Centred at `position`, facing along
    `normal`, with a given `radius`. One-sided (emits along +normal). Having area it gives SOFT shadows: each call
    it samples a uniform random point on the disk and averaging over samples gives the penumbra. color/intensity
    may be fields."""

    soft = True                                                # an area light -> multi-sampled for soft shadows

    def __init__(self, position=(0.0, 3.0, 0.0), normal=(0.0, -1.0, 0.0), radius=0.5,
                 color=(1.0, 1.0, 1.0), intensity=30.0):
        self.position = np.asarray(position, float)
        self.soft = True                                      # area source -> soft penumbra; NEE multi-samples it
        self.normal = _unit(normal)
        self.radius = float(radius)
        self.color = color
        self.intensity = intensity
        # a frame in the disk's plane so we can place sample points on it
        self._t, self._b = _tangent_frame(self.normal[None, :])
        self._t = self._t[0]; self._b = self._b[0]

    def sample(self, P, rng):
        k = len(P)
        # a uniform point on the disk: radius scales as sqrt(u) so points spread evenly (not bunched at the centre)
        rr = self.radius * np.sqrt(rng.random(k))
        th = 2.0 * np.pi * rng.random(k)
        sample_pt = (self.position[None, :]
                     + (rr * np.cos(th))[:, None] * self._t[None, :]
                     + (rr * np.sin(th))[:, None] * self._b[None, :])
        to_light = sample_pt - P
        dist = np.linalg.norm(to_light, axis=1)
        L = to_light / (dist[:, None] + 1e-9)
        cos_light = np.clip(np.sum((-L) * self.normal[None, :], axis=1), 0.0, 1.0)   # one-sided
        radiance = _emit(self.color, self.intensity, P) / (dist[:, None] ** 2 + 1e-6)
        radiance = radiance * cos_light[:, None]
        return L, dist, radiance


class SphereLight:
    """A glowing sphere of `radius` at `position`. Because it has AREA, sampling a random point on it each call and
    averaging gives SOFT shadows (a penumbra) -- the realistic look. Falls off as 1/distance^2 to the sampled
    point. Set radius small for a near-point lamp, larger for softer light. color/intensity may be fields."""

    soft = True                                                # an area light -> multi-sampled for soft shadows

    def __init__(self, position=(2.0, 3.0, 2.0), radius=0.5, color=(1.0, 1.0, 1.0), intensity=12.0):
        self.position = np.asarray(position, float)
        self.soft = True                                      # area source -> soft penumbra; NEE multi-samples it
        self.radius = float(radius)
        self.color = color
        self.intensity = intensity

    def sample(self, P, rng):
        k = len(P)
        # a uniformly random point on the sphere's surface (per shade point, so shadows soften on averaging).
        # random unit vectors via normalised gaussians -- the standard trick, uniform on the sphere.
        g = rng.standard_normal((k, 3))
        g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
        sample_pt = self.position[None, :] + self.radius * g    # (k,3) a point on the light's surface
        to_light = sample_pt - P
        dist = np.linalg.norm(to_light, axis=1)
        L = to_light / (dist[:, None] + 1e-9)
        radiance = _emit(self.color, self.intensity, P) / (dist[:, None] ** 2 + 1e-6)
        return L, dist, radiance


class DomeLight:
    """An environment / DOME light: light arriving from the whole sky around the scene, like an overcast sky, a
    studio softbox dome, or an image-based-lighting (IBL) environment. Unlike AmbientLight -- which is a flat,
    UN-shadowed fill -- the dome is SHADOWED: at each shade point we sample directions over the hemisphere and cast
    a shadow ray for each; a direction that ESCAPES the scene sees the sky, one that hits geometry does not. So
    points out in the open receive the full dome and points tucked in crevices receive less -- soft
    ambient-occlusion contact shadows fall out for free, which is exactly what makes a dome look better than a flat
    ambient.

    `color` is the sky radiance. It may be a constant RGB, or a FIELD over DIRECTION -- a callable f(dirs (k,3))->
    rgb (k,3) -- so you can give it a gradient sky or an environment map sampled by direction. `ground_color`, if
    set, is used for sample directions pointing below the horizon (dir.y < 0), like the ground half of a sky dome.

    NOTE on double-counting: the dome provides the environment term by DIRECT sampling (fast, low-noise, shadowed).
    The tracer also gathers the plain `sky` on escaped bounce rays. Use a dome with a DARK sky (as the other placed
    lights do), or a bright sky with no dome -- using both counts the environment twice for the diffuse term.

    Handled specially in direct_lighting (it needs the surface normal and the SDF), like AmbientLight. Cosine-
    weighted hemisphere sampling makes the Lambertian estimate simply albedo * sky_radiance, averaged over the
    escaped sample directions."""

    is_dome = True                                             # direct_lighting samples the hemisphere + shadow rays
    soft = True

    def __init__(self, color=(0.6, 0.7, 0.9), intensity=1.0, ground_color=None):
        self.color = color
        self.intensity = intensity
        self.ground_color = None if ground_color is None else np.asarray(ground_color, float)

    def radiance(self, dirs):
        """Sky radiance seen along each sampled world direction `dirs` (k,3) -> (k,3). Constant, a direction field,
        or split into sky / ground by the horizon."""
        d = np.atleast_2d(np.asarray(dirs, float))
        if callable(self.color):
            base = np.asarray(self.color(d), float)             # environment map / gradient by direction
        else:
            base = np.broadcast_to(np.asarray(self.color, float), (len(d), 3)).astype(float)
        rad = base * float(self.intensity) if np.ndim(self.intensity) == 0 else base * np.asarray(self.intensity)[:, None]
        if self.ground_color is not None:
            below = d[:, 1] < 0.0                               # directions pointing down see the ground, not sky
            rad = rad.copy()
            rad[below] = self.ground_color * (float(self.intensity) if np.ndim(self.intensity) == 0 else 1.0)
        return rad


class MeshLight:
    """An emissive TRIANGLE MESH -- any emitter shape you like (a ring, a logo, a strip light). Give it `vertices`
    (V,3) and `faces` (F,3) integer indices. Each call it picks a triangle with probability proportional to its
    AREA, then a uniform random point inside that triangle, so the whole surface emits evenly and shadows are soft.
    One-sided (emits along each triangle's normal). color/intensity may be fields."""

    soft = True                                                # an area light -> multi-sampled for soft shadows

    def __init__(self, vertices, faces, color=(1.0, 1.0, 1.0), intensity=15.0):
        self.vertices = np.asarray(vertices, float)
        self.soft = True                                      # area source -> soft penumbra; NEE multi-samples it
        self.faces = np.asarray(faces, int)
        self.color = color
        self.intensity = intensity
        # precompute per-triangle geometry: a corner and two edges, the normal, and the area
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        self._v0, self._e1, self._e2 = v0, v1 - v0, v2 - v0     # a corner and two edges per triangle
        cross = np.cross(self._e1, self._e2)
        self._area = 0.5 * np.linalg.norm(cross, axis=1)        # triangle areas
        self._normal = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12)
        self._cdf = np.cumsum(self._area) / (self._area.sum() + 1e-12)   # area-weighted triangle picker

    def sample(self, P, rng):
        k = len(P)
        # pick a triangle per shade point, weighted by area (bigger triangles get sampled more -> uniform surface)
        tri = np.searchsorted(self._cdf, rng.random(k))
        tri = np.clip(tri, 0, len(self.faces) - 1)
        # a uniform random point in the chosen triangle (the standard sqrt barycentric sampling)
        r1 = np.sqrt(rng.random(k)); r2 = rng.random(k)
        bary_u = (1.0 - r1)[:, None]; bary_v = (r1 * (1.0 - r2))[:, None]
        sample_pt = self._v0[tri] + bary_u * self._e1[tri] + bary_v * self._e2[tri]
        to_light = sample_pt - P
        dist = np.linalg.norm(to_light, axis=1)
        L = to_light / (dist[:, None] + 1e-9)
        cos_light = np.clip(np.sum((-L) * self._normal[tri], axis=1), 0.0, 1.0)   # one-sided emission
        radiance = _emit(self.color, self.intensity, P) / (dist[:, None] ** 2 + 1e-6)
        radiance = radiance * cos_light[:, None]
        return L, dist, radiance


class IESLight:
    """A point light with a REAL-WORLD beam shape: a photometric (IES) profile that gives the relative intensity as
    a function of the ANGLE from the aim direction. Real luminaires aren't uniform -- a downlight is bright straight
    down and dim to the sides; an IES file measures exactly that. `profile` is either a callable f(angle_rad)->
    relative_intensity, or a 1-D array of candela values sampled uniformly from 0 to `profile_max_deg` (interpolated
    over angle). Use `load_ies(text)` to build a profile from a standard .ies file. 1/distance^2 falloff, hard
    shadows. color/intensity may be fields."""

    def __init__(self, position=(0.0, 3.0, 0.0), direction=(0.0, -1.0, 0.0), profile=None,
                 color=(1.0, 1.0, 1.0), intensity=20.0, profile_max_deg=180.0):
        self.position = np.asarray(position, float)
        self.direction = _unit(direction)
        self.color = color
        self.intensity = intensity
        self.profile_max_deg = float(profile_max_deg)
        if profile is None:
            self._profile = lambda ang: np.ones_like(np.asarray(ang, float))   # uniform fallback
        elif callable(profile):
            self._profile = profile
        else:
            table = np.asarray(profile, float)                  # candela samples over [0, profile_max_deg]
            peak = table.max() + 1e-12
            angles = np.radians(np.linspace(0.0, self.profile_max_deg, len(table)))

            def _interp(ang):
                return np.interp(np.asarray(ang, float), angles, table / peak)   # normalised to peak 1
            self._profile = _interp

    def sample(self, P, rng):
        to_light = self.position[None, :] - P
        dist = np.linalg.norm(to_light, axis=1)
        L = to_light / (dist[:, None] + 1e-9)
        beam = -L                                               # from the light toward the point
        cos_ang = np.clip(np.sum(beam * self.direction[None, :], axis=1), -1.0, 1.0)
        angle = np.arccos(cos_ang)                              # angle off the aim direction (rad)
        shape = np.asarray(self._profile(angle), float)         # relative intensity from the photometric profile
        radiance = _emit(self.color, self.intensity, P) / (dist[:, None] ** 2 + 1e-6)
        radiance = radiance * shape[:, None]
        return L, dist, radiance


def load_ies(text):
    """Parse a standard IESNA LM-63 .ies photometric file into a vertical-angle profile array usable by IESLight.

    Returns (candela_by_vertical_angle, max_vertical_deg). Pass the array as IESLight(profile=..., profile_max_deg=
    max_vertical_deg). We read the vertical angles and the candela values of the FIRST horizontal plane (the common
    case: rotationally-symmetric distributions, where one plane is the whole story). This is a readable subset of
    the format -- enough to get a real luminaire's beam shape -- not a full LM-63 parser (it ignores TILT geometry,
    multi-plane azimuthal variation, and unit/geometry keywords)."""
    lines = text.splitlines()
    start = 0
    for i, ln in enumerate(lines):
        if ln.strip().upper().startswith("TILT"):
            start = i + 1
            break
    nums = []
    for ln in lines[start:]:
        for tok in ln.replace(",", " ").split():
            try:
                nums.append(float(tok))
            except ValueError:
                pass                                            # skip stray non-numeric tokens
    nums = np.asarray(nums, float)
    # header: [num_lamps, lumens/lamp, multiplier, num_vert, num_horiz, photometric_type, units, w, l, h,
    #          ballast, future, watts]  then num_vert vertical angles, num_horiz horizontal angles, then candela.
    num_vert = int(nums[3]); num_horiz = int(nums[4])
    off = 13                                                    # the two header lines carry 13 numbers total
    vert = nums[off:off + num_vert]
    off += num_vert + num_horiz                                # skip the horizontal-angle list
    candela = nums[off:off + num_vert]                         # first horizontal plane's candela vs vertical angle
    return candela, float(vert[-1] if len(vert) else 180.0)


# =====================================================================================================
# next-event estimation -- the direct-light term the path tracer adds at each hit
# =====================================================================================================
def _one_light_sample(light, sdf, P, N, V, albedo, metallic, roughness, rng, shadow_eps):
    """One next-event-estimation sample of ONE ordinary (non-ambient, non-dome) light -> (k,3) contribution.
    Sample the light, gate to points that face it and that it actually reaches, shadow-ray, then add f_r * L for
    the visible ones. Area lights call this several times (each with a fresh random light point) and average."""
    out = np.zeros_like(P)
    L, dist, radiance = light.sample(P, rng)
    ndl = np.sum(N * L, axis=1)                                 # cos(theta); <=0 -> light is below the horizon here
    lit = (ndl > 1e-4) & (np.max(radiance, axis=1) > 1e-9)      # skip back-facing and unreached (e.g. outside a cone)
    if not lit.any():
        return out
    li = np.where(lit)[0]
    O = P[li] + N[li] * shadow_eps                              # start the shadow ray just off the surface
    hit, t, _ = sphere_trace(sdf, O, L[li])
    visible = (~hit) | (t > dist[li] - shadow_eps)             # clear, or the hit is beyond the light
    vi = li[visible]
    if vi.size == 0:
        return out
    f_cos = cook_torrance(N[vi], V[vi], L[vi], albedo[vi], metallic[vi], roughness[vi])   # BRDF * N.L
    out[vi] = f_cos * radiance[vi]
    return out


def direct_lighting(sdf, P, N, V, albedo, metallic, roughness, lights, rng, shadow_eps=3e-3,
                    area_samples=2, dome_samples=4):
    """Next-event estimation: the DIRECT light reaching shade points P from all `lights`, with shadow rays.

    Parameters
      sdf          : the scene SDF (for shadow-ray occlusion tests).
      P            : (k,3) surface points being shaded.
      N            : (k,3) faced surface normals.
      V            : (k,3) view directions (toward the camera/eye).
      albedo, metallic, roughness : the surface material at P.
      lights       : list of light objects (any of the classes above).
      rng          : the tracer's seeded numpy Generator (area / dome lights sample it).
      shadow_eps   : how far to push the shadow ray off the surface so it doesn't self-intersect.
      area_samples : how many light-point samples to average for a SOFT (area) light -- more = smoother penumbra
                     and less speckle. Point/spot/directional/IES lights are deterministic and always use 1.
      dome_samples : how many hemisphere directions to average for a DOME light (its soft ambient-occlusion look).

    Returns (k,3) the summed direct radiance to ADD to each point's result (already includes the BRDF and cosine).
    Three kinds of light are handled: AmbientLight fills everywhere with no shadow ray; DomeLight samples the sky
    hemisphere with shadow rays (shadowed ambient); every other light points shadow rays at itself. Area lights are
    averaged over `area_samples` random points so their soft shadows don't come out speckly."""
    P = np.asarray(P, float); N = np.asarray(N, float); V = np.asarray(V, float)
    albedo = np.asarray(albedo, float)
    out = np.zeros_like(P)
    if not lights:
        return out

    for light in lights:
        # AMBIENT: no direction, no shadow -- a flat fill of albedo * ambient irradiance (the classic ambient term)
        if getattr(light, "is_ambient", False):
            out += albedo * light.fill(P)
            continue

        # DOME / environment: sample the sky hemisphere and shadow-ray each direction. A direction that ESCAPES the
        # scene sees the sky; one that hits geometry is blocked -> soft ambient-occlusion contact shadows. Cosine-
        # weighted sampling of a Lambert surface makes the estimate just albedo * sky_radiance over escaped dirs.
        if getattr(light, "is_dome", False):
            acc = np.zeros_like(P)
            for _ in range(max(1, dome_samples)):
                d = _cosine_hemisphere(N, rng)                  # a sky direction per point
                O = P + N * shadow_eps
                hit, t, _ = sphere_trace(sdf, O, d)             # does this direction reach the sky?
                escaped = ~hit
                if not escaped.any():
                    continue
                sky = light.radiance(d)                         # (k,3) dome radiance in each direction
                acc[escaped] += albedo[escaped] * sky[escaped]  # cosine/pdf cancel -> albedo * radiance
            out += acc / max(1, dome_samples)
            continue

        # ORDINARY lights: average several samples for soft (area) lights, one for hard (point-type) lights.
        n = max(1, area_samples) if getattr(light, "soft", False) else 1
        acc = np.zeros_like(P)
        for _ in range(n):
            acc += _one_light_sample(light, sdf, P, N, V, albedo, metallic, roughness, rng, shadow_eps)
        out += acc / n

    return out


def _selftest():
    """Every light type samples sane directions/falloff; NEE lights and shadows correctly; fields and gobos work."""
    from holographic_sdf import sphere
    rng = np.random.default_rng(0)

    # (1) point light falloff: twice as far -> a quarter the radiance
    pl = PointLight(position=(0, 0, 0), intensity=1.0)
    _, _, r_near = pl.sample(np.array([[0.0, 0.0, 1.0]]), rng)
    _, _, r_far = pl.sample(np.array([[0.0, 0.0, 2.0]]), rng)
    assert abs(float(r_near.mean()) / float(r_far.mean()) - 4.0) < 0.1     # 1/r^2

    # (2) directional light: same direction and radiance everywhere, no falloff
    dl = DirectionalLight(direction=(0, 1, 0), intensity=2.0)
    L, dist, rad = dl.sample(np.random.default_rng(1).standard_normal((5, 3)), rng)
    assert np.allclose(L, L[0]) and np.allclose(rad, rad[0])

    # (3) NEE lights a facing point (no occluder) and shadows one behind a sphere
    scene = sphere(0.4)
    light = PointLight(position=(0, 3, 0), intensity=20.0)
    N = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    V = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    P = np.array([[1.5, 0.0, 0.0], [0.0, -0.5, 0.0]])
    alb = np.full((2, 3), 0.8); met = np.zeros(2); rough = np.full(2, 0.5)
    lit = direct_lighting(scene, P, N, V, alb, met, rough, [light], rng)
    assert lit[0].sum() > 1e-3 and lit[1].sum() < lit[0].sum() * 0.2

    # (4) spot light: bright on-axis, dark outside the cone
    spot = SpotLight(position=(0, 3, 0), direction=(0, -1, 0), inner_deg=10, outer_deg=20, intensity=40.0)
    _, _, on_axis = spot.sample(np.array([[0.0, 0.0, 0.0]]), rng)          # straight below -> on axis
    _, _, off_axis = spot.sample(np.array([[3.0, 0.0, 0.0]]), rng)         # far to the side -> outside cone
    assert on_axis.max() > 1e-3 and off_axis.max() < 1e-6

    # (5) rect light: one-sided (a point behind the panel gets nothing)
    rect = RectLight(position=(0, 3, 0), u_vec=(1, 0, 0), v_vec=(0, 0, 1), intensity=30.0)  # emits along +/-y
    _, _, below = rect.sample(np.array([[0.0, 0.0, 0.0]]), rng)            # below the panel -> lit
    _, _, above = rect.sample(np.array([[0.0, 6.0, 0.0]]), rng)           # above the panel -> dark (back side)
    assert below.max() > 1e-4 and above.max() < 1e-6

    # (6) sphere light softens: sampled directions vary
    sl = SphereLight(position=(0, 3, 0), radius=1.0)
    dirs = np.array([sl.sample(np.array([[0.0, 0.0, 0.0]]), rng)[0][0] for _ in range(50)])
    assert dirs.std(axis=0).mean() > 1e-3

    # (7) mesh light: a two-triangle quad emits a finite radiance toward a point in front of it
    verts = np.array([[-1, 3, -1.0], [1, 3, -1], [1, 3, 1], [-1, 3, 1]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    ml = MeshLight(verts, faces, intensity=20.0)
    _, _, mrad = ml.sample(np.array([[0.0, 0.0, 0.0]]), rng)
    assert np.isfinite(mrad).all()

    # (8) IES light: a narrow downward beam is brighter below than to the side
    prof = np.cos(np.linspace(0, np.pi / 2, 20)) ** 4                     # a narrow downlight beam
    ies = IESLight(position=(0, 3, 0), direction=(0, -1, 0), profile=prof, profile_max_deg=90, intensity=40.0)
    _, _, i_below = ies.sample(np.array([[0.0, 0.0, 0.0]]), rng)          # straight down -> peak
    _, _, i_side = ies.sample(np.array([[3.0, 2.9, 0.0]]), rng)          # nearly sideways -> dim
    assert i_below.max() > i_side.max()

    # (9) FIELD parameters: a colour field varies the light's colour across the scene
    def red_gradient(P):
        x = np.atleast_2d(P)[:, 0]
        return np.stack([np.clip(x, 0, 1), np.zeros_like(x), np.zeros_like(x)], axis=1)  # redder to the right
    pf = PointLight(position=(0, 3, 0), color=red_gradient, intensity=1.0)
    _, _, cf = pf.sample(np.array([[0.2, 0, 0], [0.9, 0, 0]]), rng)
    assert cf[1, 0] > cf[0, 0]                                            # the right point is redder

    # (10) AMBIENT: fills with albedo*colour, no shadow ray (lights even an "enclosed" point)
    amb = AmbientLight(color=(1, 1, 1), intensity=0.2)
    af = direct_lighting(sphere(0.4), np.array([[0.0, -0.5, 0.0]]), np.array([[0.0, -1.0, 0.0]]),
                         np.array([[0.0, 0.0, 1.0]]), np.full((1, 3), 0.5), np.zeros(1), np.full(1, 0.5), [amb], rng)
    assert af.sum() > 1e-3

    # (11) GOBO: a spot with a gobo that lights only the +u half darkens the -u side
    def half_block(uv):
        return (uv[:, 0] > 0).astype(float)
    spg = SpotLight(position=(0, 3, 0), direction=(0, -1, 0), inner_deg=30, outer_deg=40, intensity=40.0,
                    gobo=half_block)
    _, _, g_pass = spg.sample(np.array([[0.4, 0.0, 0.0]]), rng)
    _, _, g_block = spg.sample(np.array([[-0.4, 0.0, 0.0]]), rng)
    assert g_pass.max() > g_block.max()

    # (12) DISK light: one-sided round area light, samples land within the radius
    disk = DiskLight(position=(0, 3, 0), normal=(0, -1, 0), radius=0.5, intensity=30.0)
    _, _, d_below = disk.sample(np.array([[0.0, 0.0, 0.0]]), rng)          # in front -> lit
    _, _, d_above = disk.sample(np.array([[0.0, 6.0, 0.0]]), rng)         # behind -> dark (one-sided)
    assert d_below.max() > 1e-4 and d_above.max() < 1e-6

    # (13) DOME light: shadowed ambient -- an open point sees the sky, a point under a big occluder sees less
    dome = DomeLight(color=(0.6, 0.7, 0.9), intensity=1.0)
    open_scene = sphere(0.3)                                              # small occluder
    P_open = np.array([[3.0, 0.0, 0.0]]); N_up = np.array([[0.0, 1.0, 0.0]]); Vv = np.array([[0.0, 0.0, 1.0]])
    lit_open = direct_lighting(open_scene, P_open, N_up, Vv, np.full((1, 3), 0.8), np.zeros(1), np.full(1, 0.5),
                               [dome], rng, dome_samples=32)
    # a point in a deep pit sees much less sky (surround it with an occluding shell via a big inverted sphere)
    from holographic_sdf import box as _box
    pit = _box(6.0, 6.0, 0.3).translate((0, 0, -0.2))                     # a wall right above blocks most sky
    P_pit = np.array([[0.0, 0.0, 0.0]])
    lit_pit = direct_lighting(pit, P_pit, np.array([[0.0, 0.0, 1.0]]), Vv, np.full((1, 3), 0.8), np.zeros(1),
                              np.full(1, 0.5), [dome], rng, dome_samples=32)
    assert lit_open.sum() > 1e-3                                          # the open point receives sky
    assert lit_pit.sum() < lit_open.sum()                                # the occluded point receives less (AO)

    # (14) area lights are MULTI-SAMPLED: in a PENUMBRA (a point partly occluded from the panel) averaging many
    # samples has lower run-to-run variance than a single sample (which randomly lands fully lit or fully shadowed)
    rectL = RectLight(position=(0, 3, 0), u_vec=(0.6, 0, 0), v_vec=(0, 0, 0.6), intensity=40.0)
    occluder = sphere(0.5).translate((0.0, 1.5, 0.0))                    # sits between the panel and the point
    Pp = np.array([[0.35, 0.0, 0.0]])                                    # in the soft edge of the sphere's shadow
    Np = np.array([[0.0, 1.0, 0.0]]); Vp = np.array([[0.0, 0.0, 1.0]])
    a1 = np.array([direct_lighting(occluder, Pp, Np, Vp, np.full((1, 3), 0.8), np.zeros(1), np.full(1, 0.5),
                                   [rectL], np.random.default_rng(s), area_samples=1).sum() for s in range(16)])
    a8 = np.array([direct_lighting(occluder, Pp, Np, Vp, np.full((1, 3), 0.8), np.zeros(1), np.full(1, 0.5),
                                   [rectL], np.random.default_rng(s), area_samples=8).sum() for s in range(16)])
    assert a8.std() <= a1.std()                                         # more samples -> no more variance (usually less)

    print("OK: holographic_lights self-test passed "
          "(point/dir/ambient/spot/rect/disk/sphere/mesh/IES/dome + fields + gobo + multi-sample)")


if __name__ == "__main__":
    _selftest()

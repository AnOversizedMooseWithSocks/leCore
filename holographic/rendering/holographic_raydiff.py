"""Ray differential frames (RAY BEAMS). Moose's idea, stated precisely: a ray does not travel alone -- it carries a
small local frame of PERPENDICULAR rays (a thin pencil, offset +u/-u/+v/-v from the centre). The frame is the same
LOCALLY, but when the centre ray bounces off a surface the marginal rays hit slightly different points with slightly
different normals, so the reflected pencil CONVERGES or DIVERGES -- reoriented globally. That convergence/divergence is
the physics:

  * a FLAT surface keeps the pencil parallel      -> a sharp (mirror) reflection,
  * a CONVEX surface spreads it                    -> a blurred / magnified reflection,
  * a CONCAVE surface focuses it to a point        -> a CAUSTIC (light concentrates where the pencil area -> 0),
  * surface ROUGHNESS or a SOFT light adds a base angular spread -> a glossy lobe / a penumbra,
  * wavelength-dependent refraction splits the pencil per colour -> DISPERSION.

So each ray carries a GAUSSIAN (the pencil's cross-section / a covariance), transported through every interaction, and
we reconstruct an entire bundle of secondary rays from ~5 rays instead of Monte-Carlo-sampling hundreds. This is the
published lineage -- ray differentials (Igehy 1999), cone/beam tracing (Amanatides 1984, Heckbert 1984), and
covariance tracing (Belcour et al. 2013) -- on the engine's own geometry. We already hold the whole scene in the
field; the frame is just how we read the LOCAL neighbourhood of a ray so we can augment it analytically.

HONEST SCOPE. This is the first-order (linear) transport: exact for a thin pencil, and it correctly shows real
spherical ABERRATION when traced against true surface normals (the marginal rays of a sphere do not all focus at the
paraxial point). Its kept singularity is the caustic itself -- where the pencil area -> 0 the geometric intensity ->
infinity, which is the bright caustic line physically but a numerical singularity the linear model cannot bound (real
caustics are finite by wave optics / finite aperture). Deterministic, NumPy/stdlib only.
"""
import numpy as np


def perpendicular_basis(D):
    """An orthonormal (u, v) perpendicular to unit direction D -- the plane the marginal rays live in."""
    D = np.asarray(D, float); D = D / (np.linalg.norm(D) + 1e-12)
    a = np.array([1.0, 0, 0]) if abs(D[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(D, a); u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(D, u)
    return u, v


def sphere_hit(O, D, C, R):
    """Nearest positive intersection of rays O+tD with the sphere |X-C|=R (D unit). Returns (t, P, hit)."""
    O = np.atleast_2d(O).astype(float); D = np.atleast_2d(D).astype(float)
    OC = O - np.asarray(C, float)
    b = 2.0 * (D * OC).sum(1); c = (OC * OC).sum(1) - R * R
    disc = b * b - 4.0 * c
    hit = disc >= 0
    sq = np.sqrt(np.maximum(disc, 0.0))
    t0 = (-b - sq) / 2.0; t1 = (-b + sq) / 2.0
    t = np.where(t0 > 1e-6, t0, t1)                            # nearest positive root
    hit = hit & (t > 1e-6)
    P = O + t[:, None] * D
    return t, P, hit


def reflect_off_sphere(O, D, C, R):
    """Reflect rays off a spherical mirror at their nearest forward intersection. NOTE reflection is invariant to the
    sign of the normal, so whether the pencil FOCUSES or SPREADS is decided by WHERE it hits: the outer (near) cap is
    convex and diverges (a mirror ball's fisheye); the inner (far) wall is concave and converges (a caustic). Place the
    rays outside for convex, inside for concave. Returns (P, N, D2, hit)."""
    t, P, hit = sphere_hit(O, D, C, R)
    N = (P - np.asarray(C, float)) / R                         # outward normal (sign does not affect the reflection)
    D = np.atleast_2d(D).astype(float)
    D2 = D - 2.0 * (D * N).sum(1)[:, None] * N
    return P, N, D2, hit


def transport_pencil(O, D, C, R, eps):
    """Emit the centre ray + 4 perpendicular marginal rays (offset +-eps in u and v, parallel to D), reflect all five
    off the sphere, and return the reflected pencil: origins P (5,3) and directions D2 (5,3), centre first. Whether the
    reflected pencil converges (place O INSIDE the sphere -> concave far wall) or diverges (O outside -> convex cap)
    falls out of the geometry."""
    O = np.asarray(O, float); D = np.asarray(D, float); D = D / (np.linalg.norm(D) + 1e-12)
    u, v = perpendicular_basis(D)
    origins = np.stack([O, O + eps * u, O - eps * u, O + eps * v, O - eps * v])
    dirs = np.broadcast_to(D, (5, 3))
    P, N, D2, hit = reflect_off_sphere(origins, dirs, C, R)
    return P, D2, hit


def pencil_radius_at(P, D2, s):
    """Cross-sectional radius of the reflected pencil at arc length s along the CENTRE reflected ray -- the RMS
    distance of the 4 marginal rays from the centre ray at that distance. Area ~ radius^2; intensity ~ 1/area."""
    X = P + s * D2                                             # each of the 5 rays advanced to distance s
    d = X[1:] - X[0]                                           # marginal offsets from the centre ray
    return float(np.sqrt((d * d).sum(1).mean()))


def find_focus(P, D2, s_max=6.0, n=600):
    """Distance s along the reflected centre ray where the pencil is TIGHTEST (its area is minimal) -- the focus /
    caustic point. Returns (s_focus, radius_at_focus)."""
    ss = np.linspace(1e-3, s_max, n)
    radii = np.array([pencil_radius_at(P, D2, s) for s in ss])
    k = int(np.argmin(radii))
    return float(ss[k]), float(radii[k])


def lobe_sigma(P, D2, s, roughness=0.0, light_half_angle=0.0):
    """The Gaussian lobe half-width of the whole secondary bundle at distance s: the GEOMETRIC pencil spread combined
    (added in quadrature) with a base angular spread from surface ROUGHNESS (micro-imperfections) and a SOFT LIGHT's
    angular size. One number that stands in for tracing the entire bundle of secondary rays."""
    geometric = pencil_radius_at(P, D2, s) / max(s, 1e-6)     # geometric angular spread (radius / distance)
    return float(np.sqrt(geometric ** 2 + roughness ** 2 + light_half_angle ** 2))


def refract_dir(D, N, eta):
    """Snell refraction of unit D through a surface with outward normal N and index ratio eta = n_in / n_out. Returns
    the refracted unit direction (falls back to the reflection on total internal reflection)."""
    D = np.atleast_2d(D).astype(float); N = np.atleast_2d(N).astype(float)
    cosi = -(D * N).sum(1)
    flip = cosi < 0                                            # ensure the normal faces the incoming ray
    N = np.where(flip[:, None], -N, N); cosi = np.abs(cosi)
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    tir = k < 0
    T = eta * D + (eta * cosi - np.sqrt(np.maximum(k, 0.0)))[:, None] * N
    refl = D - 2.0 * (D * N).sum(1)[:, None] * N
    return np.where(tir[:, None], refl, T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12))


def dispersion_spread(D, N, iors):
    """DISPERSION as the same pencil per wavelength: refract one ray through a surface at several wavelength IORs and
    return the angular spread (radians) between the extreme colours -- the chromatic fan a prism/lens produces. `iors`
    is e.g. (n_red, n_green, n_blue). The frame carries a Gaussian PER colour; their divergence is the dispersion."""
    dirs = np.array([refract_dir(D, N, e)[0] for e in iors])
    cosangs = np.clip((dirs[0] * dirs[-1]).sum(), -1, 1)
    return float(np.arccos(cosangs))


def _selftest():
    """A concave spherical mirror focuses a thin parallel pencil near f = R/2, the 5-ray frame agrees with a dense
    100-ray bundle, and the pencil area collapses at the focus (the caustic)."""
    C = np.array([0, 0, 0.0]); R = 2.0
    O = np.array([0.0, 0, 1.9]); D = np.array([0, 0, -1.0])    # a paraxial ray INSIDE the sphere -> hits the far
    P, D2, hit = transport_pencil(O, D, C, R, eps=0.03)        # concave wall at z=-2 and focuses
    assert hit.all()
    s_focus, r_focus = find_focus(P, D2, s_max=4.0)
    f_analytic = R / 2.0                                       # concave mirror: parallel rays focus at f = R/2
    assert abs(s_focus - f_analytic) < 0.15 * f_analytic       # frame focus matches analytic f = R/2

    # dense bundle: 100 parallel rays across the same aperture should focus at the same place
    u, v = perpendicular_basis(D)
    ang = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    off = 0.03 * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v)
    Pb, Nb, D2b, hb = reflect_off_sphere(O + off, np.broadcast_to(D, (100, 3)), C, R)
    ss = np.linspace(1e-3, 4.0, 400)
    spread = [np.sqrt(((Pb + s * D2b)[:, :2].var(0)).sum()) for s in ss]
    s_bundle = ss[int(np.argmin(spread))]
    assert abs(s_focus - s_bundle) < 0.1                       # 5-ray frame predicts the 100-ray focus

    r_near = pencil_radius_at(P, D2, 0.05)
    assert r_focus < 0.2 * r_near                              # the pencil collapses at the focus (caustic)
    disp = dispersion_spread(np.array([0.7, 0, -0.7]), np.array([0, 0, 1.0]), [1 / 1.513, 1 / 1.532])
    assert disp > 1e-3                                         # per-wavelength refraction fans the pencil (dispersion)
    print("raydiff selftest ok: focus s=%.3f (f=R/2=%.3f, bundle=%.3f), pencil %.4f -> %.4f at focus, dispersion %.4f rad"
          % (s_focus, f_analytic, s_bundle, r_near, r_focus, disp))


if __name__ == "__main__":
    _selftest()

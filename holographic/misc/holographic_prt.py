"""Precomputed Radiance Transfer (PRT) -- collapse the light-transport integral into a per-point operator once, then
relight with a DOT PRODUCT. The "don't path-trace, just read out" idea, grounded.

THE THESIS ("path tracing shouldn't be necessary if we can just collapse")
--------------------------------------------------------------------------
What makes global illumination expensive is the VISIBILITY INTEGRAL: at every surface point you must ask, over the
whole hemisphere of incoming directions, "does light actually reach me from there, or is something in the way?" Path
tracing answers that by shooting many rays PER FRAME, and re-shoots them every time the light changes. Precomputed
Radiance Transfer (Sloan, Kautz & Snyder, SIGGRAPH 2002) observes that for a STATIC scene the answer -- how a point
turns incident lighting into outgoing radiance, INCLUDING its own soft self-shadowing -- depends only on geometry, not
on the light. So compute it ONCE as a transfer vector in a spherical-harmonic basis, and at runtime the shading integral
collapses to a DOT PRODUCT of two ~9-element vectors (Sloan's result). Change the light, rotate the environment, animate
the sun -- each is a new light vector and a fresh dot product, no rays.

WHY THIS IS VSA-NATIVE ("as above, so below" / not limited by 3 or 4 dimensions)
-------------------------------------------------------------------------------
The transfer vector is a small codebook entry bound to each surface point; runtime lighting is a projection (a readout)
of the light onto that codebook -- exactly the bind/unbind-and-read shape the rest of the engine uses. And the basis is
a knob on dimensionality: 9 coefficients (3 SH bands) capture diffuse irradiance to ~1% (Ramamoorthi & Hanrahan 2001);
16 (4 bands) sharpen it. Higher bands = a higher-dimensional readout space = more angular detail, at more precompute.
The scene isn't living in 3D here -- each point carries a vector in a 9- or 16-D transport space.

HONEST LIMITS (kept loud)
-------------------------
* LOW FREQUENCY: truncating SH blurs sharp shadow edges -- PRT gives soft ambient shadows, not crisp contact shadows.
* STATIC geometry: the transfer is tied to fixed sample points; move the geometry and it must be recomputed.
* DIFFUSE here: this module does the classic shadowed-diffuse transfer (the 9/16-coeff case). Glossy transfer needs a
  matrix per point and higher order (noted, not built).
* The PRECOMPUTE is a Monte-Carlo visibility integral per point -- genuinely expensive. The whole point is that it is
  paid ONCE and amortized over every relight; a single still frame under one light is cheaper to shade directly.

Deterministic (seeded sampling, PYTHONHASHSEED=0); NumPy only.
"""
import numpy as np

# Real spherical-harmonic normalization constants (Cartesian form), bands l=0..3 -> 16 coefficients.
# These are the standard graphics constants (e.g. Sloan's "Stupid SH Tricks"); the selftest checks orthonormality.
_K0 = 0.2820947918                                               # l=0
_K1 = 0.4886025119                                               # l=1  (x, y, z)
_K2a = 1.0925484306                                              # l=2  (xy, yz, xz)
_K2b = 0.3153915653                                              # l=2  (3z^2-1)
_K2c = 0.5462742153                                              # l=2  (x^2-y^2)
_K3a = 0.5900435900                                             # l=3  y(3x^2-y^2), x(x^2-3y^2)
_K3b = 2.8906114426                                             # l=3  xyz
_K3c = 0.4570457995                                             # l=3  y(5z^2-1), x(5z^2-1)
_K3d = 0.3731763326                                             # l=3  z(5z^2-3)
_K3e = 1.4453057213                                             # l=3  z(x^2-y^2)


def sh_eval(dirs, order=3):
    """Evaluate the real spherical harmonics for unit directions `dirs` (M,3), returning (M, order^2) coefficients.
    `order` is the number of BANDS: order=3 -> 9 coeffs (the diffuse-irradiance standard), order=4 -> 16 (sharper)."""
    dirs = np.asarray(dirs, float)
    x = dirs[:, 0]; y = dirs[:, 1]; z = dirs[:, 2]
    cols = [np.full(len(dirs), _K0)]                             # band 0
    if order >= 2:
        cols += [_K1 * y, _K1 * z, _K1 * x]                     # band 1
    if order >= 3:
        cols += [_K2a * x * y, _K2a * y * z, _K2b * (3 * z * z - 1),
                 _K2a * x * z, _K2c * (x * x - y * y)]          # band 2
    if order >= 4:
        cols += [_K3a * y * (3 * x * x - y * y), _K3b * x * y * z, _K3c * y * (5 * z * z - 1),
                 _K3d * z * (5 * z * z - 3), _K3c * x * (5 * z * z - 1), _K3e * z * (x * x - y * y),
                 _K3a * x * (x * x - 3 * y * y)]                 # band 3
    return np.stack(cols, axis=1)


def _sphere_dirs(n, seed=0):
    """`n` deterministic near-uniform directions on the unit sphere (a spherical Fibonacci lattice -- low-discrepancy,
    so the Monte-Carlo integrals converge with far fewer samples than white-noise sampling)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)                          # polar angle, equal-area in cos
    golden = np.pi * (1.0 + 5.0 ** 0.5)                        # golden-angle azimuth
    theta = golden * i
    return np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1)


def project_env_to_sh(env_fn, order=3, n=2048):
    """Project an environment lighting function env_fn(dirs (M,3)) -> radiance (M,3) onto the SH basis.
    Returns the light vector (order^2, 3). Monte-Carlo over the sphere: coeff = (4*pi/N) * sum env(w) * Y(w)."""
    w = _sphere_dirs(n)
    Y = sh_eval(w, order)                                        # (n, ncoeff)
    E = np.asarray(env_fn(w), float)                            # (n, 3)
    return (4.0 * np.pi / n) * (Y.T @ E)                        # (ncoeff, 3)


def precompute_transfer(sdf, points, normals, order=3, n=512, eps=3e-3, max_dist=20.0):
    """The 'global transport simulator': for each surface point, integrate its SHADOWED, cosine-weighted visibility over
    the hemisphere and project onto SH -> a transfer vector per point (len(points), order^2). This is the one expensive
    precompute; it captures soft self-shadowing (a point tucked in a crevice sees less sky) without any runtime rays.

    T_k(p) = (1/pi) * integral_hemisphere V(p,w) * max(0, n.w) * Y_k(w) dw
    where V(p,w) = 1 if a ray from p along w escapes the SDF unobstructed. Diffuse (Lambert 1/pi) convention, so a fully
    open white point under a unit-DC light returns ~albedo."""
    from holographic.rendering.holographic_raymarch import sphere_trace
    points = np.asarray(points, float); normals = np.asarray(normals, float)
    P = len(points)
    w = _sphere_dirs(n)                                         # shared sample directions (M dirs)
    Y = sh_eval(w, order)                                       # (n, ncoeff)
    ncoeff = Y.shape[1]
    transfer = np.zeros((P, ncoeff))
    solid = 4.0 * np.pi / n                                     # weight per sample direction
    # process points in blocks to bound memory: for each block, shadow-test every (point, direction) pair
    block = max(1, int(4000 / max(n, 1)) * 8)
    for s in range(0, P, block):
        pb = points[s:s + block]; nb = normals[s:s + block]; B = len(pb)
        cosw = nb @ w.T                                         # (B, n) cosine of each direction at each point
        upper = cosw > 0.0                                      # only the hemisphere above the surface
        # visibility: shoot a ray from each point along each upper-hemisphere direction; blocked if it hits the SDF
        bi, di = np.where(upper)
        origins = pb[bi] + nb[bi] * eps                        # lift off the surface to avoid self-hit
        rays = w[di]
        hit, _, _ = sphere_trace(sdf, origins, rays, max_dist=max_dist)
        vis = np.zeros((B, n)); vis[bi, di] = (~hit).astype(float)   # 1 where the ray escaped (lit)
        weight = vis * np.clip(cosw, 0.0, None) * solid / np.pi     # cosine * visibility * dw / pi
        transfer[s:s + block] = weight @ Y                     # project the weighted visibility onto SH
    return transfer


def shade_prt(transfer, light_sh, albedo=None):
    """Runtime shading -- the collapse. Radiance per point = albedo * (transfer . light_sh), a dot product of the
    point's transfer vector with the light's SH vector. No rays. `transfer` (P,ncoeff), `light_sh` (ncoeff,3)."""
    rad = transfer @ np.asarray(light_sh, float)                # (P,3) -- the shading integral, as a matrix-vector product
    rad = np.clip(rad, 0.0, None)
    if albedo is not None:
        rad = rad * np.asarray(albedo, float)
    return rad


def _selftest():
    """The SH basis is orthonormal, a fully-open point recovers the ambient DC light, and a point under an occluder
    receives measurably less -- soft self-shadowing, with relighting as a pure dot product."""
    # 1) orthonormality: (1/N) sum Y_i Y_j * 4pi ~= identity
    w = _sphere_dirs(20000)
    Y = sh_eval(w, order=4)
    G = (4.0 * np.pi / len(w)) * (Y.T @ Y)
    off = np.abs(G - np.eye(Y.shape[1])).max()
    assert off < 0.02, off                                      # orthonormal to Monte-Carlo tolerance

    # 2) transfer + relight on a two-sphere scene: the point between the spheres is self-shadowed
    class TwoSpheres:
        def eval(s, Pp):
            a = np.linalg.norm(Pp - np.array([-1.0, 0, 0]), axis=1) - 0.9
            b = np.linalg.norm(Pp - np.array([1.0, 0, 0]), axis=1) - 0.9
            return np.minimum(a, b)
    sdf = TwoSpheres()
    # an open point on top of the left sphere vs a point on its right flank (facing the other sphere)
    p_open = np.array([[-1.0, 0.9, 0.0]]); n_open = np.array([[0.0, 1.0, 0.0]])
    p_tuck = np.array([[-0.15, 0.0, 0.0]]); n_tuck = np.array([[1.0, 0.0, 0.0]])   # faces the right sphere, blocked
    T_open = precompute_transfer(sdf, p_open, n_open, order=3, n=800)
    T_tuck = precompute_transfer(sdf, p_tuck, n_tuck, order=3, n=800)
    white = lambda d: np.ones((len(d), 3))                      # uniform white environment
    L = project_env_to_sh(white, order=3, n=4000)
    r_open = float(shade_prt(T_open, L)[0, 0])
    r_tuck = float(shade_prt(T_tuck, L)[0, 0])
    assert r_open > r_tuck * 1.3, (r_open, r_tuck)             # the tucked point is self-shadowed -> darker
    # a fully-open upward point under a white sky returns ~1 (albedo) -- the DC recovery
    assert 0.7 < r_open < 1.2, r_open
    print("prt selftest ok: SH orthonormal (off-diag %.4f); open point %.3f vs self-shadowed %.3f (soft shadow, "
          "relight = dot product)" % (off, r_open, r_tuck))


if __name__ == "__main__":
    _selftest()

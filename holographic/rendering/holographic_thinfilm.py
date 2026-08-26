"""holographic_thinfilm -- thin-film interference iridescence (soap bubble, oil slick, beetle shell).

Why a surface goes rainbow-coloured: a very thin transparent film (soap, oil, an insect's cuticle) sits on top of
another surface. Light reflects off BOTH the top of the film and the bottom. Those two reflected beams travel
slightly different distances -- the bottom beam makes an extra round trip through the film -- so when they
recombine they INTERFERE. Whether that interference is constructive (bright) or destructive (dark) depends on how
the extra path length compares to the light's WAVELENGTH. Red, green and blue have different wavelengths, so at a
given film thickness and viewing angle some colours reinforce and others cancel -- and as you tilt the surface the
path length changes and the colours shift. That angle-dependent rainbow sheen is iridescence.

This module computes the effect from first principles and boils it down to one thing a shader can use: a per-pixel
RGB TINT that multiplies the surface's reflection. It reuses the CIE colour-matching fit already in
holographic_blackbody (no second copy of that table), so the spectrum -> colour step is grounded and consistent
with the rest of the engine.

The core physics (two-beam interference, the standard textbook model):
  * A ray hits the film at angle theta_i (cos_theta from the surface normal). Inside the film it refracts to
    theta_t via Snell's law:  sin(theta_t) = sin(theta_i) / n_film.
  * The extra optical path the bottom reflection travels is:  OPD = 2 * n_film * d * cos(theta_t),
    where d is the film thickness. (cos(theta_t), not cos(theta_i): the beam travels THROUGH the film.)
  * There is also a half-wave (pi) phase flip on reflection at an interface going into a DENSER medium. We include
    the top-surface flip via `phase_flip` so soap-on-air vs oil-on-water come out right.
  * For wavelength lambda the reflected intensity is  R(lambda) = R_amp * sin^2( pi * OPD / lambda + flip/2 ).
    Integrating R(lambda) against the eye's colour response over the visible band gives the tint.

Everything is NumPy + stdlib. Deterministic. No learned weights.
"""

import numpy as np
from holographic.misc.holographic_blackbody import _cie_xyz_bar, _XYZ_TO_RGB, _gamma_encode


# The visible band we integrate over, sampled once. 60 samples is plenty for a smooth tint and stays cheap.
_LAM_NM = np.linspace(380.0, 780.0, 60)                          # wavelengths in nanometres


def interference_reflectance(thickness_nm, cos_theta, n_film=1.33, phase_flip=True, lam_nm=None):
    """Reflected intensity per wavelength from a thin film -- the raw interference spectrum.

    Parameters
      thickness_nm : film thickness in nanometres (soap ~ 100-1000 nm; the visible iridescence range).
      cos_theta    : cosine of the incidence/view angle from the surface normal (scalar or array).
      n_film       : refractive index of the film (soap/water ~1.33, oil ~1.45).
      phase_flip   : include the half-wave (pi) flip on the top reflection (film denser than the medium above).
      lam_nm       : wavelengths to evaluate (nm); default the module's visible-band sampling.

    Returns an array of shape (..., n_lambda): the reflectance at each wavelength, in [0,1]. This is the spectrum
    you'd see; feed it to `spectrum_to_rgb` (or use `thin_film_tint` which does both).
    """
    lam = _LAM_NM if lam_nm is None else np.asarray(lam_nm, float)
    cos_i = np.clip(np.abs(np.asarray(cos_theta, float)), 1e-4, 1.0)
    d_nm = np.asarray(thickness_nm, float)                       # scalar or per-point (...,)

    # Snell inside the film: sin(theta_t) = sin(theta_i)/n. cos(theta_t) is what sets the path length.
    sin_i2 = 1.0 - cos_i * cos_i                                 # sin^2(theta_i)
    sin_t2 = sin_i2 / (n_film * n_film)                          # sin^2(theta_t) via Snell
    cos_t = np.sqrt(np.clip(1.0 - sin_t2, 0.0, 1.0))            # cos(theta_t) in the film

    # Optical path difference between the two reflected beams (nm). The 2x is the down-and-back trip.
    opd = 2.0 * n_film * d_nm * cos_t                            # (...) broadcast over the angle / per-point
    flip = np.pi if phase_flip else 0.0                         # half-wave phase flip on the first reflection

    # phase = 2*pi*OPD/lambda; reflected intensity ~ sin^2(phase/2 + flip/2). Broadcast angle (...) x wavelength.
    phase = 2.0 * np.pi * (np.asarray(opd)[..., None] / lam) + flip   # (..., n_lambda)
    return np.sin(0.5 * phase) ** 2                             # in [0,1] per wavelength


def spectrum_to_rgb(spectrum, lam_nm=None):
    """Convert a per-wavelength reflectance spectrum (..., n_lambda) to a linear-ish sRGB colour (..., 3).

    Integrates the spectrum against the CIE 1931 colour-matching functions (reused from holographic_blackbody),
    converts XYZ -> sRGB, gamma-encodes, and normalises to a unit-ish tint so it can multiply a reflection without
    darkening it overall. This is the eye's answer to 'what colour is this spectrum?'.
    """
    lam = _LAM_NM if lam_nm is None else np.asarray(lam_nm, float)
    S = np.asarray(spectrum, float)
    xb, yb, zb = _cie_xyz_bar(lam)                               # (n_lambda,) each

    # Riemann-sum the spectrum against each colour-matching curve -> XYZ tristimulus (..., 3).
    X = np.sum(S * xb, axis=-1)
    Y = np.sum(S * yb, axis=-1)
    Z = np.sum(S * zb, axis=-1)
    XYZ = np.stack([X, Y, Z], axis=-1)                          # (..., 3)

    rgb = XYZ @ _XYZ_TO_RGB.T                                   # linear sRGB (..., 3)
    rgb = np.clip(rgb, 0.0, None)                               # drop out-of-gamut negatives

    # Normalise by the brightest channel so the result is a HUE/TINT (peak ~1), not an absolute brightness.
    peak = np.max(rgb, axis=-1, keepdims=True)
    rgb = rgb / np.clip(peak, 1e-6, None)
    return _gamma_encode(np.clip(rgb, 0.0, 1.0))


def thin_film_tint(thickness_nm, cos_theta, n_film=1.33, phase_flip=True):
    """The iridescent RGB tint for a film of `thickness_nm` seen at angle `cos_theta` -- the shader-ready value.

    Combines the two steps: compute the interference spectrum, then convert to an sRGB tint (..., 3). Multiply a
    surface's reflected colour by this to give it a soap-bubble/oil-slick sheen. As `cos_theta` sweeps (the
    surface tilts) or `thickness_nm` varies across the surface, the tint cycles through the spectrum -- the
    hallmark of iridescence.
    """
    spec = interference_reflectance(thickness_nm, cos_theta, n_film=n_film, phase_flip=phase_flip)
    return spectrum_to_rgb(spec)


def iridescent_socket(base_color, thickness_nm=320.0, n_film=1.33, strength=0.8, phase_flip=True,
                      thickness_variation=0.0, seed=0):
    """Build an albedo/reflectance SOCKET f(points, normals, view_dirs) -> (M,3) for an iridescent surface.

    This matches the socket pattern the renderer already uses for crystal/inclusion materials, extended with the
    normal and view direction (iridescence is view-dependent, so it needs them). At each shaded point it computes
    the thin-film tint from the angle between the view direction and the surface normal, then blends it over the
    base colour by `strength`.

    Parameters
      base_color          : the underlying surface colour (3,) -- the tint modulates reflections on top of it.
      thickness_nm        : nominal film thickness (nm). ~200-400 nm gives strong visible colour.
      n_film              : film refractive index (1.33 soapy water, 1.45 oil).
      strength            : 0..1, how strongly the iridescence shows over the base colour.
      phase_flip          : half-wave flip on the top reflection (see interference_reflectance).
      thickness_variation : if > 0, add a smooth position-dependent wobble to the thickness (nm), so the film
                            isn't perfectly uniform -- real soap films vary in thickness, which is why the colours
                            swirl. Deterministic (hash of position), no RNG state.
      seed                : offsets the thickness wobble pattern.

    Returns a callable socket(points (M,3), normals (M,3), view_dirs (M,3)) -> (M,3) rgb.
    """
    base = np.asarray(base_color, float)

    def socket(points, normals, view_dirs):
        P = np.atleast_2d(np.asarray(points, float))
        N = np.atleast_2d(np.asarray(normals, float))
        Vd = np.atleast_2d(np.asarray(view_dirs, float))
        # view angle from the surface normal: cos_theta = |N . V|
        cos_theta = np.abs(np.sum(N * Vd, axis=-1))

        d = float(thickness_nm)
        if thickness_variation > 0.0:
            # a smooth deterministic wobble of the film thickness across the surface (hash of position).
            h = np.sin(P @ np.array([12.9898, 78.233, 37.719]) + float(seed)) * 43758.5453
            wobble = (np.modf(h)[0] * 2.0 - 1.0)                # in [-1, 1], deterministic per point
            d = d + thickness_variation * wobble               # (M,) per-point thickness

        tint = thin_film_tint(d, cos_theta, n_film=n_film, phase_flip=phase_flip)   # (M,3)
        # blend the iridescent tint over the base colour by strength
        return (1.0 - strength) * base + strength * tint

    return socket


def _selftest():
    """The tint cycles with thickness and angle, stays in [0,1], and is deterministic."""
    # (1) at a fixed angle, sweeping thickness cycles the hue -> the tint is not constant
    tints = np.array([thin_film_tint(t, 1.0) for t in np.linspace(100, 800, 40)])
    assert tints.min() >= 0.0 and tints.max() <= 1.0
    spread = tints.std(axis=0).mean()
    assert spread > 0.05, spread                                # colour genuinely varies with thickness

    # (2) view-angle dependence: the same film reads a different colour head-on vs grazing
    head_on = thin_film_tint(320.0, 1.0)                        # cos_theta = 1 (normal)
    grazing = thin_film_tint(320.0, 0.2)                        # cos_theta = 0.2 (near grazing)
    assert np.linalg.norm(head_on - grazing) > 0.05            # the colour shifts with angle -> iridescent

    # (3) the socket blends over a base colour and is view-dependent
    base = np.array([0.2, 0.2, 0.25])
    sock = iridescent_socket(base, thickness_nm=300.0, strength=1.0)
    P = np.zeros((5, 3))
    N = np.tile([0.0, 0.0, 1.0], (5, 1))
    V = np.array([[0, 0, 1.0], [0.3, 0, 0.95], [0.6, 0, 0.8], [0.8, 0, 0.6], [0.95, 0, 0.31]])
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    cols = sock(P, N, V)
    assert cols.shape == (5, 3)
    assert cols.std(axis=0).mean() > 0.02                      # different view angles -> different colours

    # (4) deterministic
    a = thin_film_tint(275.0, 0.7)
    b = thin_film_tint(275.0, 0.7)
    assert np.array_equal(a, b)

    print("OK: holographic_thinfilm self-test passed")


if __name__ == "__main__":
    _selftest()

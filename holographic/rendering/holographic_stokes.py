"""holographic_stokes.py -- the STATE OF POLARIZED LIGHT as a Stokes vector (leCore rendering).

WHY THIS EXISTS
---------------
Every radiance value in the engine has, until now, been a scalar (or an RGB triple): "how much light".
But light also has a SHAPE -- the geometry of its electric-field oscillation -- and that shape carries
real information the scalar throws away. A reflection off water is partially linearly polarized; a
mantis shrimp reads the e-vector orientation and even the HANDEDNESS of circular light (Chiou et al.
2008); a radio telescope measures the magnetic field of a galaxy through the polarization of its
synchrotron glow. To render or analyse any of that we must carry the full state, not just S0.

The Stokes vector [S0, S1, S2, S3] is the standard, measurement-based description of that state
(Stokes 1852). It is defined entirely by intensities you could actually measure with filters:

    S0 = total intensity                         (I)
    S1 = I(0 deg)   - I(90 deg)                   (Q -- horizontal vs vertical linear)
    S2 = I(+45 deg) - I(-45 deg)                  (U -- diagonal linear)
    S3 = I(right-circular) - I(left-circular)     (V -- circular / handedness)

We chose the Stokes 4-vector (not the Jones 2-vector) for two constitutional reasons:
  * it is REAL-valued and describes PARTIAL polarization + unpolarised light, which is what real
    sensors and real scenes have (Jones handles only fully-polarised coherent light);
  * every optical element then acts by a real 4x4 Mueller matrix (holographic_mueller), so the whole
    pipeline stays plain numpy with no complex bookkeeping in the transport.

DIRECTIONS (the up/down/sideways wiring, built in from birth -- a missed direction is a missed faculty)
  DOWN   -- a single Stokes vector is the atom; there is no meaningful decomposition below one
            wavelength sample. DECLARED NEGATIVE, not an oversight.
  UP     -- a polarised image is a FIELD of Stokes vectors. Every function here is written to
            broadcast over a leading batch: shape (..., 4). A lone vector is (4,); an image is
            (H, W, 4); a spectral cube is (nlam, 4) or (H, W, nlam, 4). One implementation, all scales.
  SIDEWAYS
    field    -- (..., 4) arrays, above.
    sequence -- the CONVERGENCE wiring: P(lambda^2) = S1 + i*S2 (= Q + iU) is a complex SAMPLED WAVE.
                `complex_linear` returns that phasor. Take its FFT over lambda^2 and you have
                rotation-measure synthesis (Brentjens & de Bruyn 2005) -- the telescope's Faraday
                probe -- for free, using the engine's existing FFT core. Building the state as a wave
                is why one core serves both the mantis eye and the radio dish (U1).
    program  -- the arithmetic is elementwise, so mind.emit_kernel can project it to a shader.

BACKWARD COMPATIBILITY (byte-identical, non-negotiable): `from_radiance(x)` lifts a scalar/RGB radiance
to an unpolarised Stokes vector [x, 0, 0, 0] and `to_radiance(S)` returns S0. Any existing scalar path
that round-trips through here is bit-for-bit unchanged -- polarization is purely additive information.

Determinism: pure closed-form numpy, no RNG, no hashing needed. Exact to floating-point.

Convention note (handedness): we use the OPTICS convention S3 > 0 == RIGHT-circular (RCP), as in most
polarimetry and radio astronomy. This is a labelling choice; it is stated so downstream code and the
mantis retarder (holographic_mueller) agree on a sign rather than silently disagreeing.
"""

import numpy as np

# The four components live on the LAST axis so everything broadcasts over an arbitrary field shape.
_I, _Q, _U, _V = 0, 1, 2, 3


# ----------------------------------------------------------------------------------------------------
# Constructors -- each returns an array whose last axis is the 4 Stokes components.
# ----------------------------------------------------------------------------------------------------

def unpolarized(intensity=1.0):
    """Fully UNPOLARISED light of the given intensity: [I, 0, 0, 0]. `intensity` may be a scalar or an
    array (a field of intensities) -- the result gains a trailing length-4 axis. This is also what a
    scalar radiance IS in Stokes terms, so it is the identity element the old pipeline maps onto."""
    intensity = np.asarray(intensity, float)
    S = np.zeros(intensity.shape + (4,), float)
    S[..., _I] = intensity
    return S


def linear(intensity=1.0, angle=0.0, p=1.0):
    """LINEARLY polarised light: intensity `intensity`, e-vector at `angle` RADIANS (measured from the
    S1/horizontal axis), degree of linear polarization `p` in [0,1] (p<1 mixes in an unpolarised part).

    The 2x on the angle is physical, not a typo: a polarizer at theta and at theta+180 deg are
    indistinguishable, so polarization orientation lives on the DOUBLE-angle circle -- S1=cos(2t),
    S2=sin(2t). (This is exactly why e-vector angle later comes back as 0.5*atan2(U, Q).)"""
    intensity = np.asarray(intensity, float)
    angle = np.asarray(angle, float)
    p = np.asarray(p, float)
    S = np.zeros(np.broadcast(intensity, angle, p).shape + (4,), float)
    S[..., _I] = intensity
    S[..., _Q] = intensity * p * np.cos(2.0 * angle)
    S[..., _U] = intensity * p * np.sin(2.0 * angle)
    return S


def circular(intensity=1.0, handedness=1, p=1.0):
    """CIRCULARLY polarised light: `handedness` +1 = right (RCP, S3>0), -1 = left (LCP, S3<0); degree
    of circular polarization `p` in [0,1]. This is the channel the mantis shrimp uniquely sees and the
    one Faraday rotation shuffles into the linear channels along a line of sight."""
    intensity = np.asarray(intensity, float)
    hnd = np.sign(np.asarray(handedness, float))
    p = np.asarray(p, float)
    S = np.zeros(np.broadcast(intensity, hnd, p).shape + (4,), float)
    S[..., _I] = intensity
    S[..., _V] = intensity * p * hnd
    return S


def from_components(I, Q, U, V):
    """Assemble a Stokes field from four separate component arrays (e.g. the I/Q/U/V planes a radio
    telescope actually delivers). Broadcasts them together; the last axis becomes the 4 components."""
    I, Q, U, V = (np.asarray(x, float) for x in (I, Q, U, V))
    shape = np.broadcast(I, Q, U, V).shape
    S = np.empty(shape + (4,), float)
    S[..., _I], S[..., _Q], S[..., _U], S[..., _V] = I, Q, U, V
    return S


# ----------------------------------------------------------------------------------------------------
# Backward-compatibility bridge -- scalar radiance <-> Stokes, byte-identical round trip.
# ----------------------------------------------------------------------------------------------------

def from_radiance(radiance):
    """Lift an existing scalar (or RGB, or any-shape) radiance to UNPOLARISED Stokes. The old value
    becomes S0; the polarization axes are exactly zero. `to_radiance(from_radiance(x)) == x` bitwise."""
    return unpolarized(radiance)


def to_radiance(S):
    """Collapse a Stokes field back to the plain intensity S0 the scalar pipeline expects. This is the
    'polarization OFF' path: it returns exactly what was put in via from_radiance, no rounding."""
    return np.asarray(S, float)[..., _I]


# ----------------------------------------------------------------------------------------------------
# Derived quantities -- all broadcast over the field; a lone vector and an image use the SAME code.
# ----------------------------------------------------------------------------------------------------

def intensity(S):
    """Total intensity S0 of the field."""
    return np.asarray(S, float)[..., _I]


def _pol_intensity(S):
    # magnitude of the polarised part, sqrt(Q^2+U^2+V^2) -- shared by dop and the physicality check.
    S = np.asarray(S, float)
    return np.sqrt(S[..., _Q] ** 2 + S[..., _U] ** 2 + S[..., _V] ** 2)


def dop(S):
    """DEGREE OF POLARIZATION in [0,1]: sqrt(Q^2+U^2+V^2)/I. 0 = unpolarised, 1 = fully polarised.
    Zero-intensity samples return 0 rather than NaN (an unlit pixel has no polarization to report)."""
    S = np.asarray(S, float)
    I = S[..., _I]
    out = np.zeros_like(I)
    nz = I > 0
    out[nz] = _pol_intensity(S)[nz] / I[nz]
    return out if out.ndim else float(out)


def dolp(S):
    """DEGREE OF LINEAR polarization: sqrt(Q^2+U^2)/I -- the part a plain (non-circular) polarizer or a
    linear-only eye (most linear-polarization animals) can see."""
    S = np.asarray(S, float)
    I = S[..., _I]
    out = np.zeros_like(I)
    nz = I > 0
    out[nz] = np.sqrt(S[..., _Q] ** 2 + S[..., _U] ** 2)[nz] / I[nz]
    return out if out.ndim else float(out)


def docp(S):
    """DEGREE OF CIRCULAR polarization: |V|/I -- the channel the mantis shrimp's R8 retarder unlocks."""
    S = np.asarray(S, float)
    I = S[..., _I]
    out = np.zeros_like(I)
    nz = I > 0
    out[nz] = np.abs(S[..., _V])[nz] / I[nz]
    return out if out.ndim else float(out)


def evector_angle(S):
    """Orientation of the polarization ellipse's major axis, in RADIANS, in (-pi/2, pi/2]. Recovered as
    0.5*atan2(U, Q) -- the inverse of the double-angle in `linear`. This is the e-vector orientation an
    animal or a polarimeter reports; it is also the quantity Faraday rotation advances with lambda^2."""
    S = np.asarray(S, float)
    ang = 0.5 * np.arctan2(S[..., _U], S[..., _Q])
    return ang if ang.ndim else float(ang)


def handedness(S):
    """+1 for right-circular content, -1 for left, 0 for none: sign(V). The diverging channel O3 will
    paint so a human can SEE the handedness the mantis distinguishes."""
    S = np.asarray(S, float)
    h = np.sign(S[..., _V])
    return h if h.ndim else float(h)


def is_physical(S, tol=1e-9):
    """A Stokes vector is PHYSICAL iff I >= sqrt(Q^2+U^2+V^2) (you cannot be more polarised than lit).
    Returns a bool array over the field. Used as a guard when data or a Mueller product could push a
    vector past the Poincare sphere; a violation means an upstream bug, not a rounding artefact."""
    S = np.asarray(S, float)
    return S[..., _I] + tol >= _pol_intensity(S)


# ----------------------------------------------------------------------------------------------------
# SIDEWAYS/sequence -- the convergence view that makes rotation-measure synthesis a one-liner (U1).
# ----------------------------------------------------------------------------------------------------

def complex_linear(S):
    """The complex linear polarization P = Q + iU (Stokes S1 + i*S2), as a phasor. Its MAGNITUDE is the
    linearly-polarised intensity and its PHASE is twice the e-vector angle.

    Why this belongs in the state, not in the telescope code: sampled across many wavelengths, P(lambda^2)
    is a complex WAVE, and Faraday rotation makes its phase advance linearly in lambda^2. So the FFT of
    P over lambda^2 is the Faraday-depth spectrum -- rotation-measure synthesis (Brentjens & de Bruyn
    2005) -- computed by the engine's existing complex-FFT core. Exposing the phasor here is what lets
    the mantis-shrimp arc and the radio-telescope arc share one operator (U1)."""
    S = np.asarray(S, float)
    return S[..., _Q] + 1j * S[..., _U]


def _selftest():
    """Assert the EXACT polarimetric contract (to 1e-12), the field/scalar equivalence (up == down),
    the byte-identical radiance round trip, and the phasor identity that RM synthesis will lean on."""
    rng = np.random.default_rng(0)

    # --- unpolarised: no polarization, exact zeros ---
    u = unpolarized(2.5)
    assert u.shape == (4,)
    assert dop(u) == 0.0 and dolp(u) == 0.0 and docp(u) == 0.0
    assert u[_Q] == 0.0 and u[_U] == 0.0 and u[_V] == 0.0

    # --- linear at a known angle: dolp==p exactly, no circular, angle recovered, components exact ---
    for theta in [0.0, np.pi / 6, np.pi / 4, 1.1, -0.7]:
        s = linear(3.0, theta, p=0.8)
        assert abs(dolp(s) - 0.8) < 1e-12, "dolp must equal the requested p"
        assert docp(s) < 1e-12 and handedness(s) == 0.0
        # angle lives mod pi (double-angle); compare on the circle
        d = (evector_angle(s) - theta) % np.pi
        d = min(d, np.pi - d)
        assert d < 1e-12, "e-vector angle must invert the double-angle construction"
        assert abs(s[_Q] - 3.0 * 0.8 * np.cos(2 * theta)) < 1e-12
        assert abs(s[_U] - 3.0 * 0.8 * np.sin(2 * theta)) < 1e-12

    # --- circular: docp==p exactly, no linear, handedness sign correct both ways ---
    r = circular(1.0, handedness=+1, p=1.0)
    l = circular(1.0, handedness=-1, p=0.5)
    assert abs(docp(r) - 1.0) < 1e-12 and dolp(r) < 1e-12 and handedness(r) == +1.0
    assert abs(docp(l) - 0.5) < 1e-12 and handedness(l) == -1.0

    # --- fully polarised light sits exactly ON the Poincare sphere: I^2 == Q^2+U^2+V^2 ---
    fp = linear(2.0, 0.3, p=1.0)
    assert abs(fp[_I] ** 2 - (fp[_Q] ** 2 + fp[_U] ** 2 + fp[_V] ** 2)) < 1e-12
    assert bool(np.all(is_physical(fp)))
    # a NON-physical vector (over-polarised) is caught
    bad = np.array([1.0, 0.9, 0.9, 0.0])
    assert not bool(np.all(is_physical(bad)))

    # --- BACKWARD COMPAT: scalar radiance round-trips BYTE-IDENTICALLY through Stokes ---
    x = rng.standard_normal((5, 7)) ** 2  # a positive radiance image
    assert np.array_equal(to_radiance(from_radiance(x)), x), "radiance round trip must be exact"

    # --- UP == DOWN: an image (field) of Stokes vectors gives the same per-pixel answers as looping ---
    I = rng.random((4, 6)) + 0.1
    ang = rng.uniform(-1, 1, (4, 6))
    field = linear(I, ang, p=0.7)            # shape (4, 6, 4)
    assert field.shape == (4, 6, 4)
    d_field = dolp(field)
    for i in range(4):
        for j in range(6):
            one = linear(float(I[i, j]), float(ang[i, j]), p=0.7)
            assert abs(dolp(one) - d_field[i, j]) < 1e-12, "field and per-vector must agree exactly"

    # --- SEQUENCE/convergence: P = Q+iU has magnitude = linear intensity and phase = 2*angle ---
    s = linear(1.0, 0.4, p=1.0)
    P = complex_linear(s)
    assert abs(np.abs(P) - 1.0) < 1e-12
    assert abs(((np.angle(P) - 2 * 0.4 + np.pi) % (2 * np.pi)) - np.pi) < 1e-12

    print("OK: holographic_stokes self-test passed (exact polarimetry to 1e-12; unpolarised/linear/"
          "circular constructors; dop/dolp/docp/angle/handedness; Poincare-sphere physicality guard; "
          "BYTE-IDENTICAL radiance round trip; field==per-vector (up==down); P=Q+iU phasor for RM "
          "synthesis)")


if __name__ == "__main__":
    _selftest()

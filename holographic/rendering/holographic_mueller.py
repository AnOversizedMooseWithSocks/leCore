"""holographic_mueller.py -- how optical elements TRANSFORM polarized light: the Mueller matrix (leCore).

WHY THIS EXISTS
---------------
holographic_stokes gives light a STATE (S0..S3). This gives the world its ACTIONS on that state. Every
linear, non-image-forming optical element -- a polarizer, a wave plate, a rotator, a reflecting surface,
a stretch of magnetised plasma -- acts on a Stokes vector by a single real 4x4 matrix M:

        S_out = M @ S_in

That is the whole contract, and it is why we chose Stokes over Jones: the transport stays REAL 4x4
numpy, elements COMPOSE by matrix multiplication, and partial polarization is handled for free.

The retarder is the load-bearing element for this arc. A quarter-wave retarder at 45 degrees turns
linearly polarised light into CIRCULARLY polarised light -- and run backwards it turns circular into
linear. That is exactly the trick the mantis shrimp's R8 rhabdomere plays (Chiou et al. 2008): a
biological quarter-wave plate that converts the circular light its linear detectors cannot read into
linear light they can. So "see circular polarization" is not a new sensor, it is a retarder (here) in
front of a linear channel (holographic_observer, later).

The rotator is the load-bearing element for the OTHER arc. An optical rotator advances the plane of
linear polarization by a fixed angle -- which is precisely what Faraday rotation does to radio waves
crossing a magnetised medium. So the telescope's magnetic-field probe and the bug's retarder are the
same family of 4x4 matrix; one module serves both (U1).

DIRECTIONS (up/down/sideways, built in from birth)
  DOWN   -- a matrix acts on one Stokes vector; that is the atom.
  UP     -- composition is a MONOID: `identity()` is the unit, `compose()` is the associative combine,
            and N copies of the same element is `power()` (matrix power -- retardances/rotations just
            ADD, a closed-form 'iterate'). `apply()` broadcasts one matrix over a whole Stokes FIELD,
            or applies a FIELD of matrices elementwise (a spatially-varying retarder / birefringence map).
  SIDEWAYS
    field   -- `apply` handles (...,4,4) matrix fields onto (...,4) Stokes fields.
    program -- the matrices are constants and `apply` is one einsum, so mind.emit_kernel can project it.

Determinism: pure closed-form numpy, no RNG. Angles in RADIANS throughout, consistent with holographic_stokes.

KEPT NEGATIVE: `fresnel_reflection` handles REAL refractive indices (dielectrics: water, glass, ice,
atmosphere) where the s/p reflection coefficients are real and the only cross-phase is 0 or pi. METALS
and total-internal-reflection need COMPLEX indices (a genuine s-p phase retardance on reflection); that
is a declared later item, not silently approximated here.
"""

import numpy as np

from holographic.rendering import holographic_stokes as stk


# ----------------------------------------------------------------------------------------------------
# Frame rotation -- the building block every oriented element is built from.
# ----------------------------------------------------------------------------------------------------

def rotation(phi):
    """The Mueller matrix that ROTATES THE REFERENCE FRAME by `phi` radians. Note the 2*phi: polarization
    orientation lives on the double-angle circle, so a physical rotation by phi mixes Q and U by 2*phi.
    Oriented elements are R(-t) @ (element at 0) @ R(t); an optical rotator is this alone (see `rotator`)."""
    c, s = np.cos(2.0 * phi), np.sin(2.0 * phi)
    return np.array([[1, 0, 0, 0],
                     [0,  c, s, 0],
                     [0, -s, c, 0],
                     [0, 0, 0, 1]], float)


def identity():
    """Free space / no element: the 4x4 identity, and the UNIT of the composition monoid."""
    return np.eye(4)


# ----------------------------------------------------------------------------------------------------
# Elements.
# ----------------------------------------------------------------------------------------------------

def linear_polarizer(angle=0.0):
    """An ideal LINEAR POLARIZER transmitting the e-vector at `angle` radians. Unpolarised light in gives
    half intensity out, fully polarised along `angle`. Two of these crossed (angle differing by pi/2)
    pass nothing -- the classic extinction, which the selftest pins exactly."""
    c, s = np.cos(2.0 * angle), np.sin(2.0 * angle)
    return 0.5 * np.array([[1, c, s, 0],
                           [c, c * c, c * s, 0],
                           [s, c * s, s * s, 0],
                           [0, 0, 0, 0]], float)


def retarder(delta, angle=0.0):
    """A general linear RETARDER (wave plate): retardance `delta` radians between the fast axis (at
    `angle`) and the slow axis. delta=pi/2 is a quarter-wave plate, delta=pi a half-wave plate. Built by
    rotating a retarder-at-zero into place: R(-angle) @ M_delta @ R(angle)."""
    cd, sd = np.cos(delta), np.sin(delta)
    m0 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0,  cd, sd],
                   [0, 0, -sd, cd]], float)
    if angle == 0.0:
        return m0
    return rotation(-angle) @ m0 @ rotation(angle)


def quarter_wave(angle=0.0):
    """A QUARTER-WAVE plate (delta=pi/2) at `angle`. At 45 degrees it converts linear<->circular -- the
    mantis-shrimp R8 retarder. This is the single most important element in the polarization arc."""
    return retarder(np.pi / 2.0, angle)


def half_wave(angle=0.0):
    """A HALF-WAVE plate (delta=pi) at `angle`. It REFLECTS the e-vector orientation about the fast axis
    (linear at theta -> linear at 2*angle - theta), the standard way to steer linear polarization."""
    return retarder(np.pi, angle)


def rotator(rho):
    """An optical ROTATOR: rotates the plane of linear polarization by `rho` radians (Q,U spin, V and I
    untouched). This IS Faraday rotation of a radio wave by a magnetised medium -- the element rotation-
    measure synthesis inverts. It equals rotation(-rho): a frame rotation by -rho advances the e-vector
    by +rho. Rotators COMPOSE additively: rotator(a) @ rotator(b) == rotator(a+b) (pinned in selftest)."""
    return rotation(-rho)


def depolarizer(factor=0.0):
    """A partial DEPOLARIZER: scales the polarised part (Q,U,V) by `factor` in [0,1] while keeping total
    intensity. factor=1 is a no-op, factor=0 fully unpolarises. Real scattering media (a dusty nebula,
    a rough surface) reduce the degree of polarization; this is the knob for that."""
    return np.diag([1.0, factor, factor, factor])


def fresnel_reflection(n1, n2, theta_i):
    """The polarizing REFLECTION Mueller matrix at a DIELECTRIC interface (real indices n1->n2), incidence
    `theta_i` radians. This is the PHYSICAL origin of polarization in a render: surfaces polarize what
    they reflect, most strongly near Brewster's angle where the reflection is fully linearly polarised
    (Rp -> 0). Below is the real-coefficient form; complex-index metals are a declared later item.

    s and p are the field components perpendicular / parallel to the plane of incidence. At/after total
    internal reflection (no real transmitted angle) we return the perfect-mirror identity scaled to unit
    reflectance (Rs=Rp=1), which is the correct limit for the intensity terms."""
    ci = np.cos(theta_i)
    sin_t = (n1 / n2) * np.sin(theta_i)
    if sin_t >= 1.0:                       # total internal reflection: everything reflects, no diattenuation
        return np.eye(4)
    ct = np.sqrt(1.0 - sin_t * sin_t)
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct)
    Rs, Rp, cross = rs * rs, rp * rp, rs * rp     # real coeffs: s-p cross term is 2*rs*rp (sign carries the phase)
    return 0.5 * np.array([[Rs + Rp, Rs - Rp, 0, 0],
                           [Rs - Rp, Rs + Rp, 0, 0],
                           [0, 0, 2 * cross, 0],
                           [0, 0, 0, 2 * cross]], float)


# ----------------------------------------------------------------------------------------------------
# Composition monoid + application (the UP direction).
# ----------------------------------------------------------------------------------------------------

def compose(*elements):
    """Fold a light path into ONE matrix. Arguments are given IN THE ORDER LIGHT PASSES THROUGH them, so
    the later elements multiply on the LEFT: compose(A, B, C) == C @ B @ A. Empty path == identity. This
    is the monoid's associative combine; compose(identity(), M) == M is pinned in the selftest."""
    M = np.eye(4)
    for e in elements:
        M = np.asarray(e, float) @ M
    return M


def power(M, n):
    """N copies of the SAME element in a row, as one closed-form matrix power (not an n-step loop). For a
    retarder this multiplies retardance (power(retarder(d), n) == retarder(n*d)); for a rotator it adds
    rotation. This is the 'iterate a diagonal-ish operator k times = one evaluation' lever."""
    return np.linalg.matrix_power(np.asarray(M, float), int(n))


def apply(M, S):
    """Transform a Stokes vector or FIELD by a Mueller matrix or matrix-field. Cases handled:
      * one matrix (4,4) onto one vector (4,) or a field (...,4)  -> broadcast the matrix over the field;
      * a matrix FIELD (...,4,4) onto a matching Stokes field (...,4) -> elementwise (a birefringence map).
    Returns the transformed Stokes array of the same field shape."""
    M = np.asarray(M, float)
    S = np.asarray(S, float)
    if M.ndim == 2:
        return np.einsum("ij,...j->...i", M, S)
    return np.einsum("...ij,...j->...i", M, S)      # a field of matrices onto a field of vectors


def _selftest():
    """Pin the EXACT physics (to ~1e-12): identity, polarizer half-power + crossed extinction, the
    QUARTER-WAVE-at-45 linear->circular conversion (the mantis mechanism), Faraday rotator additivity,
    Brewster full polarization, the composition monoid, and the field/broadcast paths."""
    # --- identity is a no-op ---
    s = stk.linear(2.0, 0.3, p=1.0)
    assert np.allclose(apply(identity(), s), s, atol=1e-12)

    # --- polarizer: unpolarised -> half intensity, fully linear along the axis ---
    u = stk.unpolarized(1.0)
    out = apply(linear_polarizer(0.0), u)
    assert abs(stk.intensity(out) - 0.5) < 1e-12
    assert abs(stk.dolp(out) - 1.0) < 1e-12
    # crossed polarizers pass nothing
    crossed = compose(linear_polarizer(0.0), linear_polarizer(np.pi / 2))
    assert stk.intensity(apply(crossed, u)) < 1e-12

    # --- THE LOAD-BEARING TEST: quarter-wave at 45deg turns linear@0 into CIRCULAR ---
    lin0 = stk.linear(1.0, 0.0, p=1.0)
    circ = apply(quarter_wave(np.pi / 4), lin0)
    assert abs(stk.docp(circ) - 1.0) < 1e-12, "QWP@45 must fully circularly polarise linear light"
    assert stk.dolp(circ) < 1e-12
    assert stk.handedness(circ) == 1.0
    # ...and a QWP@45 run again takes that circular light back to linear (the mantis read-out)
    back = apply(quarter_wave(np.pi / 4), circ)
    assert stk.dolp(back) > 1.0 - 1e-9 and stk.docp(back) < 1e-9

    # --- half-wave reflects the e-vector angle: linear@theta -> linear@-theta (axis at 0) ---
    th = 0.4
    hw = apply(half_wave(0.0), stk.linear(1.0, th, p=1.0))
    d = (stk.evector_angle(hw) - (-th)) % np.pi
    assert min(d, np.pi - d) < 1e-12

    # --- rotator == Faraday rotation, and rotators ADD (the RM-synthesis element) ---
    rlin = apply(rotator(0.5), stk.linear(1.0, 0.0, p=1.0))
    dd = (stk.evector_angle(rlin) - 0.5) % np.pi
    assert min(dd, np.pi - dd) < 1e-12, "rotator(rho) must advance the e-vector by rho"
    assert np.allclose(power(rotator(0.3), 4), rotator(1.2), atol=1e-12), "rotators compose additively"
    assert np.allclose(power(retarder(0.2, 0.0), 3), retarder(0.6, 0.0), atol=1e-12), "retardances add"

    # --- Brewster: reflection off glass at atan(n2/n1) is FULLY linearly polarised ---
    n1, n2 = 1.0, 1.5
    brew = np.arctan2(n2, n1)
    refl = apply(fresnel_reflection(n1, n2, brew), stk.unpolarized(1.0))
    assert abs(stk.dolp(refl) - 1.0) < 1e-9, "reflection at Brewster's angle is fully polarised"

    # --- monoid: identity is the unit; compose is associative order (light order, later on the left) ---
    A, B = quarter_wave(0.2), linear_polarizer(0.7)
    assert np.allclose(compose(identity(), A), A, atol=1e-12)
    assert np.allclose(compose(A, B), B @ A, atol=1e-12)

    # --- FIELD paths: one matrix over an image, and a field of matrices elementwise ---
    field = stk.linear(np.ones((3, 3)), np.zeros((3, 3)), p=1.0)     # (3,3,4)
    fout = apply(quarter_wave(np.pi / 4), field)
    assert fout.shape == (3, 3, 4) and abs(stk.docp(fout).mean() - 1.0) < 1e-12
    Mfield = np.broadcast_to(quarter_wave(np.pi / 4), (3, 3, 4, 4))
    assert np.allclose(apply(Mfield, field), fout, atol=1e-12), "matrix-field path must match broadcast"

    print("OK: holographic_mueller self-test passed (identity; polarizer half-power + crossed "
          "extinction; QWP@45 linear<->circular = the mantis retarder; HWP angle flip; rotator = "
          "Faraday, rotators/retardances ADD via power(); Brewster full polarization; composition "
          "monoid; field + matrix-field apply)")


if __name__ == "__main__":
    _selftest()

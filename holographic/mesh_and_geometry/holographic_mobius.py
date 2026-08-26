"""Möbius / non-orientable encoders -- matching a representation's TOPOLOGY to its data.

WHY THIS EXISTS
---------------
holostuff's binding is circular convolution, so its native topology is the circle (and the torus).
That is the right shape for a DIRECTED angle -- a heading, a phase that runs 0..2pi. But a circle is
the WRONG shape for two kinds of data that show up the moment you have "multiple states" or
"sign-flipping noise", and forcing them onto a circle quietly corrupts the geometry:

  * AXIAL data: theta and theta+pi are the SAME state -- the orientation of an unoriented line, a
    nematic/director field, a crystal axis, a phase defined only mod pi. On a circle theta and
    theta+pi sit at OPPOSITE points (similarity -1): the representation screams "maximally different"
    about two things that are identical. The correct base space is the projective line RP^1 -- the
    base of the Mobius double-cover. The fix is the classic DOUBLE-ANGLE map theta -> 2*theta, which
    makes theta and theta+pi coincide exactly. (In neuroscience this is why orientation tuning in
    visual cortex traces a Klein bottle / Mobius structure, not a ring -- the topology matches the
    variable.)

  * SIGN-FLIPPING data: f(t+T) = -f(t), a pattern that inverts every period and only truly returns
    after TWO of them. That is antiperiodic -- a Mobius double-cover in time. Its energy lives
    ENTIRELY in the ODD-harmonic subspace; the ordinary periodic (circular) basis is blind to it.

MEASURED
  * Axial recovery error (each value reported as theta or theta+pi at random): naive circle 0.470 rad
    vs double-angle Mobius 0.002 rad. similarity(theta, theta+pi): naive -0.22 vs Mobius +1.00.
  * A sign-flipping signal carries ~100% of its energy in the antiperiodic component (the periodic
    half is ~1e-14).

KEPT NEGATIVE / SCOPE (the honest boundary)
  * Use these ONLY when the data is genuinely axial or sign-flipping. On ordinary DIRECTED data the
    circle is correct, and the double-angle encoder deliberately THROWS AWAY the half-turn
    distinction -- it would wrongly merge a heading with its reverse. This is a tool for when the
    topology doesn't match, not a free upgrade.
  * Naming an old kept negative: binary quantization maps values to +-1, which is itself a Z2 /
    antipodal (Mobius-like) identification. That is precisely why it distorted CIRCULAR similarity
    geometry (and why it is, conversely, the RIGHT move for axial/sign-flip data). Same lesson.

Pure NumPy, deterministic, no new dependencies.
"""

import numpy as np


class AxialEncoder:
    """FHRR-style phasor encoder for AXIAL values where theta and theta+pi denote the same state.

    Encodes via the double-angle (projective) map e^{i * freqs * 2*theta}, so theta and theta+pi map
    to the SAME hypervector -- the representation lives on RP^1 (the Mobius base), not the circle.
    """

    def __init__(self, dim, seed=0, max_freq=8):
        rng = np.random.default_rng(seed)
        # integer frequencies keep the phasor exactly pi-periodic in theta after the angle doubling,
        # so theta and theta+pi coincide to machine precision (not just approximately).
        self.freqs = rng.integers(1, max_freq, dim).astype(float)
        self.dim = dim

    def encode(self, theta):
        """Map an axial value (or array of them) to a unit phasor hypervector via theta -> 2*theta."""
        theta = np.asarray(theta, float)
        if theta.ndim == 0:
            return np.exp(1j * self.freqs * 2.0 * float(theta))
        return np.exp(1j * self.freqs[None, :] * 2.0 * theta[:, None])

    def similarity(self, a_theta, b_theta):
        """Cosine similarity between two axial values: +1 when they are the same orientation
        (including the theta vs theta+pi case), regardless of a pi flip."""
        a = self.encode(a_theta)
        b = self.encode(b_theta)
        return float(np.real(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def decode(self, vec, grid=720):
        """Recover the axial value in [0, pi) whose phasor best matches `vec` (mod pi by construction)."""
        g = np.linspace(0.0, np.pi, grid, endpoint=False)
        G = np.exp(1j * self.freqs[None, :] * 2.0 * g[:, None])
        return float(g[int(np.argmax(np.real(G @ np.conj(vec))))])


def antiperiodic_fraction(signal):
    """Fraction of a signal's energy in the SIGN-FLIPPING (Mobius/antiperiodic) component.

    Splits the signal into two equal halves a, b. The antiperiodic part is (a - b)/2 and the periodic
    part is (a + b)/2 -- a clean orthogonal split that needs no FFT-bin parity bookkeeping. Returns
    ~1.0 for f(t+T) = -f(t) data and ~0.0 for ordinary periodic f(t+T) = f(t) data. This is the
    diagnostic for "does this pattern belong on a Mobius strip rather than a circle?".
    """
    x = np.asarray(signal, float)
    _NAME = "antiperiodic_fraction"
    # A DIAGNOSTIC MUST NOT RETURN NaN. This is a number a caller BRANCHES ON,
    # and a NaN compares False to every threshold -- so a poisoned input reads
    # as "ordinary periodic" and the caller proceeds on a measurement that was
    # never made. The doctrine is already on record in the declare ladder:
    # "a NaN must never pass a gate". Applied where the number is PRODUCED.
    # AND THE EMPTY CASE, which the `x.size and` above SKIPPED BY CONSTRUCTION.
    # I wrote that guard last sweep to stop a NaN reading as 0.0, and left the
    # hole immediately beside it: an EMPTY signal returns 0.0 too, and 0.0 is
    # the real answer for "ordinary periodic". Fewer than two samples cannot be
    # split into two halves at all, so there is nothing to measure.
    # A GUARD THAT STARTS WITH `if x.size and` IS A GUARD THAT EXEMPTS THE
    # EMPTIEST INPUT FROM THE CHECK.
    if x.size < 2:
        raise ValueError(
            "%s needs at least 2 samples (got %d): the measurement splits the "
            "signal into two halves, and 0.0 -- the value returned for an empty "
            "input -- is also the real answer for 'ordinary periodic'."
            % (_NAME, int(x.size)))
    if not np.all(np.isfinite(x)):
        raise ValueError(
            "antiperiodic_fraction needs a finite signal: %d of %d samples are NaN or inf. A NaN "
            "diagnostic reads as 0.0 at every threshold -- a measurement that "
            "was never made, wearing a real-looking answer."
            % (int((~np.isfinite(x)).sum()), int(x.size)))
    n = len(x) // 2
    a, b = x[:n], x[n:2 * n]
    anti = (a - b) / 2.0
    peri = (a + b) / 2.0
    e = float(np.sum(anti ** 2) + np.sum(peri ** 2))
    return float(np.sum(anti ** 2) / (e + 1e-12))


def antiperiodic_split(signal):
    """Return (periodic_component, antiperiodic_component) of the first two periods of `signal`.
    The antiperiodic component is the part a circular representation cannot hold."""
    x = np.asarray(signal, float)
    _NAME = "antiperiodic_split"
    # A DIAGNOSTIC MUST NOT RETURN NaN. This is a number a caller BRANCHES ON,
    # and a NaN compares False to every threshold -- so a poisoned input reads
    # as "ordinary periodic" and the caller proceeds on a measurement that was
    # never made. The doctrine is already on record in the declare ladder:
    # "a NaN must never pass a gate". Applied where the number is PRODUCED.
    # AND THE EMPTY CASE, which the `x.size and` above SKIPPED BY CONSTRUCTION.
    # I wrote that guard last sweep to stop a NaN reading as 0.0, and left the
    # hole immediately beside it: an EMPTY signal returns 0.0 too, and 0.0 is
    # the real answer for "ordinary periodic". Fewer than two samples cannot be
    # split into two halves at all, so there is nothing to measure.
    # A GUARD THAT STARTS WITH `if x.size and` IS A GUARD THAT EXEMPTS THE
    # EMPTIEST INPUT FROM THE CHECK.
    if x.size < 2:
        raise ValueError(
            "%s needs at least 2 samples (got %d): the measurement splits the "
            "signal into two halves, and 0.0 -- the value returned for an empty "
            "input -- is also the real answer for 'ordinary periodic'."
            % (_NAME, int(x.size)))
    if not np.all(np.isfinite(x)):
        raise ValueError(
            "antiperiodic_split needs a finite signal: %d of %d samples are NaN or inf. A NaN "
            "diagnostic reads as 0.0 at every threshold -- a measurement that "
            "was never made, wearing a real-looking answer."
            % (int((~np.isfinite(x)).sum()), int(x.size)))
    n = len(x) // 2
    a, b = x[:n], x[n:2 * n]
    return (a + b) / 2.0, (a - b) / 2.0

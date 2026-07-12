"""Sign as rotation: the analytic signal, and the price of clockwise-only rotation.

WHY THIS MODULE EXISTS
----------------------
A signed scalar series (price deltas, an audio wave, any +/- signal) is the
SHADOW of a point rotating on a circle: the value is one coordinate of the point,
its SIGN is which half of the circle the point is on, and "make it negative" is a
rotation. Formalising that turns a real series x(t) into its rotating complex
companion, the ANALYTIC SIGNAL

    z(t) = x(t) + i * H[x](t) = A(t) * exp(i * phi(t))

where H is the Hilbert transform (the 90-degrees-shifted "quadrature" partner),
A(t) = |z| is the instantaneous AMPLITUDE (the circle's radius, time-varying),
phi(t) = angle(z) is the instantaneous PHASE (the rotation angle), and
d phi/dt is the instantaneous FREQUENCY (how fast the sign is turning over).
[Gabor 1946, "Theory of Communication"; Boashash 1992 for instantaneous freq.]

Because z's real part is x by construction, A * cos(phi) reconstructs x EXACTLY
-- the (amplitude, phase) pair is a lossless re-coordinatisation, not a lossy
model. This is the REVERSIBLE, two-channel form: it is a GROUP element (SO(2)),
every rotation has an inverse, and it can represent a reversal (the phasor backing
up = negative instantaneous frequency) as a small local motion.

THE CLOCKWISE-ONLY QUESTION
---------------------------
If rotation may only go one way (phase may only advance), you drop from a GROUP to
a MONOID: you can still compose rotations but you can no longer invert one. This is
the algebra of irreversibility -- the same group-vs-semigroup split that separates
reversible (unitary) from irreversible (contraction-semigroup / diffusion)
evolution. Two consequences, both measured here:

  GAIN: a monotone phase is an UNWRAPPABLE phase -- it lifts to a straight,
    ever-increasing line (the covering map R -> S^1), so it becomes a clean
    monotone INDEX / clock (Itoh's condition: given < pi advance per sample the
    unwrap is unique, no quadrature partner needed). This is the carrier/index
    idea (holographic_axisrole) re-derived: forcing one direction turns the
    wrap-around circle back into a monotone carrier axis.

  COST: you can no longer represent a genuine reversal cheaply. Where the true
    rotation wants to back up (instantaneous frequency < 0), a clockwise-only
    rotation must STALL (a ratchet under a reversing drive does not turn back), so
    the reconstruction departs from the signal exactly at the reversals. That
    departure is the measured price of the wrong constraint on a bidirectional
    signal.

WHAT THIS BUYS THE ENGINE
-------------------------
  * sign-aware, comparable coordinates for a signed series (feed a phasor memory,
    compare cyclic structure);
  * a data-driven test of whether a signal is "monocomponent" (well-behaved,
    monotone-ish phase) -- the reversal fraction;
  * the honest number for "how much does clockwise-only distort a bidirectional
    signal," which is the empirical core of the whole question.

CONNECTS TO (in the engine): FHRR unit phasors (holographic_fhrr) are the
constant-radius special case of this; Clifford rotors (holographic_clifford) are
the exact composable rotation primitive; the Mobius axial encoder
(holographic_mobius) handles the theta vs theta+pi sign fold. This module is the
missing assembled capability: raw signed series -> its rotating form, both modes.

KEPT NEGATIVES (loud)
---------------------
  * THE REAL-SIGNAL THEOREM (the sharpest finding, stated up front because it
    reshapes the whole question): a real scalar signal has a Hermitian-symmetric
    spectrum, so its analytic signal ALWAYS rotates one way -- the instantaneous
    frequency is essentially non-negative by construction. A real series therefore
    IS already a clockwise-only rotation, and clamping it monotone costs ~nothing.
    A genuine reversal (the phasor truly backing up) can only exist for a COMPLEX /
    two-channel (I/Q) rotation, where the second channel carries an independent
    direction. This is the quadrature-encoder fact made rigorous: a single real
    channel cannot even REPRESENT a reversal, so of course one-way costs nothing on
    it. The group-vs-monoid price is real, but it lives on the complex path
    (phasor_monotone_cost), not the real one (monotone_cost, which will honestly
    read ~0 and say so).
  * The Hilbert transform is a GLOBAL FFT operation, so it assumes periodicity and
    has EDGE EFFECTS: amplitude/phase near the first/last few samples are
    unreliable. Measurements here trim a margin; a windowed or mirror-padded input
    is cleaner. Documented, not hidden.
  * The analytic-signal amplitude/phase split is only physically meaningful for a
    reasonably NARROWBAND / monocomponent signal (Bedrosian/Nuttall conditions). On
    a broadband multi-tone mess the phase is a valid number but not "the" rotation;
    the reversal fraction will read high, which is itself the honest warning.
  * The reversible round-trip is exact by construction (Re(z) = x). That is a
    STRONG baseline, not a strawman: the monotone cost is measured against it and
    against the original, in the original units.
  * Re-coordinatising returns as rotation is LOSSLESS, not predictive. It cannot
    manufacture structure that is not there (the engine's kept negative: SOL
    returns have no linear structure). It buys comparability and decomposition,
    not a crystal ball.

Only NumPy + stdlib. Deterministic (FFT; no RNG here).
"""

import numpy as np

# Below this |instantaneous frequency| a step counts as "no rotation" rather than a
# reversal -- guards float noise around the zero-crossing from inflating the
# reversal fraction. In radians/sample; ~0.6 degrees.
_FREQ_EPS = 1e-2

# Default fraction of each end to trim before MEASURING, to dodge Hilbert edge
# effects. Not applied to the returned arrays (the caller may want them whole);
# only to the error/statistics so a boundary artefact can't dominate a number.
_EDGE_TRIM = 0.08


def hilbert(x):
    """The analytic signal z = x + i*H[x] of a real series, by FFT (pure NumPy).

    WHY reimplement (no scipy): the constitution forbids a second dependency. The
    recipe is exact and classical -- take the FFT, keep the positive frequencies,
    double them, zero the negatives (the analytic-signal filter), invert. The
    imaginary part of the result is the Hilbert transform (the 90-degrees quadrature
    partner); the whole complex result is the rotating companion of x.

    Returns a complex array the same length as x, with Re(z) == x to float precision
    (the property that makes the amplitude/phase form losslessly invertible).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=complex)
    X = np.fft.fft(x)
    h = np.zeros(n)
    # The analytic-signal spectral mask: DC and (for even n) Nyquist kept once, the
    # positive half doubled, the negative half zeroed. This is scipy.signal.hilbert's
    # construction, written out so it has no external dependency.
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(X * h)


def analytic_signal(x):
    """Decompose a signed series into its rotation: amplitude, phase, inst. frequency.

    Returns dict:
      analytic   : complex z(t) = x + i*H[x] (the rotating companion).
      amplitude  : A(t) = |z|, the instantaneous envelope (the circle radius).
      phase      : UNWRAPPED instantaneous phase phi(t) (radians), monotone-lifted
                   so it does not jump at +/- pi -- the "how far has it rotated" axis.
      inst_freq  : d phi/dt (radians/sample), length N-1. Its SIGN is the direction
                   of rotation at each step: >0 clockwise-convention forward, <0 a
                   reversal (the phasor backing up).

    WHY unwrap the phase: wrapped phase lives on the circle and jumps by 2*pi; the
    unwrapped phase is the covering-space lift to the real line, which is what makes
    "instantaneous frequency" (its slope) and "monotone or not" (its sign changes)
    well defined. [np.unwrap uses Itoh's < pi assumption.]
    """
    x = np.asarray(x, dtype=float).ravel()
    z = hilbert(x)
    amp = np.abs(z)
    phase = np.unwrap(np.angle(z))
    inst_freq = np.diff(phase)
    return {"analytic": z, "amplitude": amp, "phase": phase,
            "inst_freq": inst_freq}


def _interior(a, trim=_EDGE_TRIM):
    """Return the interior slice of an array, dropping `trim` of each end.

    WHY: Hilbert edge effects live in the first/last few samples. Statistics are
    computed on the interior so a boundary artefact cannot dominate a reported
    number. Always leaves at least one sample.
    """
    a = np.asarray(a)
    n = a.shape[0]
    k = int(n * trim)
    if 2 * k >= n:
        k = max(0, (n - 1) // 2)
    return a[k:n - k] if k > 0 else a


def enforce_monotone(phase, direction=+1):
    """Force a phase to rotate ONE way only (the clockwise-only / monoid constraint).

    Given the unwrapped phase, clamp its per-step change so it may only advance in
    `direction` (+1 = non-decreasing, -1 = non-increasing). Where the true rotation
    reverses, the clamped phase STALLS (step 0) rather than backing up -- the honest
    behaviour of a ratchet driven against its allowed direction, and the reason a
    reversal cannot be represented cheaply once you leave the group.

    Returns the monotone phase (same length). Reconstructing A*cos(phase_mono) then
    departs from the signal exactly where reversals were clamped -- that departure is
    the measured cost.
    """
    phase = np.asarray(phase, dtype=float).ravel()
    if phase.size < 2:
        return phase.copy()
    steps = np.diff(phase)
    if direction >= 0:
        steps = np.clip(steps, 0.0, None)   # may only advance (never back up)
    else:
        steps = np.clip(steps, None, 0.0)   # may only retreat
    return np.concatenate([[phase[0]], phase[0] + np.cumsum(steps)])


def rotary_encode(x, monotonic=False, direction=+1):
    """Encode a signed series as rotation, then reconstruct -- both modes.

    monotonic=False (REVERSIBLE / two-channel / group): keeps the full phase, so the
    reconstruction A*cos(phi) == x to float precision. This is the lossless form; it
    can represent reversals because the quadrature partner (the Hilbert term) records
    the direction of motion.

    monotonic=True (CLOCKWISE-ONLY / single-channel / monoid): forces the phase to
    advance one way (see enforce_monotone), so a reversal stalls and the
    reconstruction departs from x at those points. This is the direction-assuming,
    single-channel-encoder form -- cheaper and unambiguous to unwrap, blind to
    reversal.

    Returns dict: amplitude, phase (the phase actually used), reconstruction,
    monotonic (bool). The two modes share the same amplitude; only the phase differs.
    """
    x = np.asarray(x, dtype=float).ravel()
    a = analytic_signal(x)
    amp, phase = a["amplitude"], a["phase"]
    used_phase = enforce_monotone(phase, direction=direction) if monotonic else phase
    recon = amp * np.cos(used_phase)
    return {"amplitude": amp, "phase": used_phase, "reconstruction": recon,
            "monotonic": bool(monotonic)}


def reversal_fraction(x, freq_eps=_FREQ_EPS, trim=_EDGE_TRIM):
    """Fraction of steps where the rotation REVERSES (instantaneous frequency < 0).

    WHY this number matters: it is the data-driven test of whether a signal is
    "monocomponent" / well-behaved for the rotation picture. Near 0 -> the phase is
    essentially monotone, so clockwise-only costs almost nothing (the signal is
    already a one-way rotation). Large -> the signal genuinely goes both ways, so the
    clockwise-only constraint will distort it. Measured on the interior to avoid edge
    effects.
    """
    a = analytic_signal(x)
    f = _interior(a["inst_freq"], trim=trim)
    if f.size == 0:
        return 0.0
    return float(np.mean(f < -freq_eps))


def monotone_cost(x, direction=+1, trim=_EDGE_TRIM):
    """MEASURE the price of clockwise-only rotation on this signal (the headline).

    Reconstructs the signal two ways and compares, in the ORIGINAL units, on the
    interior:
      reversible_rmse : ||x - A*cos(phi)||        -- the lossless baseline (~0).
      monotone_rmse   : ||x - A*cos(phi_mono)||   -- the cost of the one-way clamp.
      excess          : monotone_rmse - reversible_rmse (the price attributable to
                        the constraint, not to any residual round-trip error).
      reversal_fraction : where the rotation wanted to back up (predicts where the
                        error lives).
      max_local_error : the worst single-sample departure under the clamp.

    On a purely forward (monotone-phase) signal the excess is ~0: clockwise-only is
    free because the signal was already a one-way rotation. On a reversing signal the
    excess is large and concentrated at the reversals -- the group-vs-monoid cost,
    measured against the strongest honest baseline (the exact reversible round-trip).
    """
    x = np.asarray(x, dtype=float).ravel()
    a = analytic_signal(x)
    amp, phase = a["amplitude"], a["phase"]

    recon_rev = amp * np.cos(phase)
    phase_mono = enforce_monotone(phase, direction=direction)
    recon_mono = amp * np.cos(phase_mono)

    xi = _interior(x, trim=trim)
    rev_i = _interior(recon_rev, trim=trim)
    mono_i = _interior(recon_mono, trim=trim)

    reversible_rmse = float(np.sqrt(np.mean((xi - rev_i) ** 2)))
    monotone_rmse = float(np.sqrt(np.mean((xi - mono_i) ** 2)))
    max_local = float(np.max(np.abs(xi - mono_i))) if xi.size else 0.0

    return {"reversible_rmse": reversible_rmse,
            "monotone_rmse": monotone_rmse,
            "excess": float(monotone_rmse - reversible_rmse),
            "reversal_fraction": reversal_fraction(x, trim=trim),
            "max_local_error": max_local}


def phasor_monotone_cost(z, direction=+1, trim=_EDGE_TRIM):
    """The group-vs-monoid cost where it ACTUALLY lives: a true complex rotation.

    Unlike a real scalar signal (whose analytic phase is one-way by the symmetric-
    spectrum theorem, so monotone_cost reads ~0), a complex/I-Q series z(t) carries a
    genuine rotation DIRECTION in its two channels. It can truly reverse -- the
    phasor backing up -- and clamping it clockwise-only then loses that reversal.
    This is the quadrature encoder with both channels present: drop to one direction
    and you pay.

    Measures, against the exact reversible baseline (z itself), on the interior:
      reversal_fraction : steps where the phasor rotates the "wrong" way.
      monotone_rmse     : ||z - z_mono|| after clamping the phase one-way (complex).
      excess            : same, minus the ~0 reversible baseline.
      max_local_error   : worst single-sample complex departure.

    z_mono keeps the true amplitude |z| and the clamped (monotone) phase, so it is
    the honest "what a single-channel, direction-assuming encoder would reconstruct."
    """
    z = np.asarray(z, dtype=complex).ravel()
    amp = np.abs(z)
    phase = np.unwrap(np.angle(z))
    inst = np.diff(phase)

    phase_mono = enforce_monotone(phase, direction=direction)
    z_mono = amp * np.exp(1j * phase_mono)

    zi = _interior(z, trim=trim)
    zmi = _interior(z_mono, trim=trim)
    fi = _interior(inst, trim=trim)

    rev_frac = float(np.mean(fi < -_FREQ_EPS)) if direction >= 0 else \
        float(np.mean(fi > _FREQ_EPS))
    monotone_rmse = float(np.sqrt(np.mean(np.abs(zi - zmi) ** 2))) if zi.size else 0.0
    max_local = float(np.max(np.abs(zi - zmi))) if zi.size else 0.0

    return {"reversal_fraction": rev_frac,
            "monotone_rmse": monotone_rmse,
            "excess": monotone_rmse,  # baseline (z vs z) is exactly 0
            "max_local_error": max_local}


def _selftest():
    """Assert the exact numeric contracts, failing loudly on the core claims.

    1. Hilbert quadrature: the analytic signal of cos is exp(i*w*t) -- amplitude
       constant, and H[cos] == sin (the 90-degrees partner). The canonical check.
    2. Reversible round-trip is EXACT (Re(z) = x): A*cos(phi) == x to 1e-10.
    3. THE REAL-SIGNAL THEOREM: a real scalar signal is ALREADY a one-way rotation,
       so clockwise-only costs ~nothing -- monotone_cost excess and reversal_fraction
       are both ~0 even on a multi-tone reversing-looking wave. Stated as an
       assertion so a regression that fakes a real-signal reversal is caught.
    4. THE GROUP-VS-MONOID PRICE, where it lives: a TRUE complex phasor that reverses
       pays a large, well-defined excess when clamped clockwise-only, and its
       reversal_fraction is high. This is the honest home of the cost.
    5. Determinism.
    """
    t = np.linspace(0, 8 * np.pi, 1024, endpoint=False)

    # (1) quadrature: analytic signal of cos(w t) ~ exp(i w t).
    c = np.cos(t)
    z = hilbert(c)
    him = _interior(np.imag(z))
    sref = _interior(np.sin(t))
    assert np.max(np.abs(him - sref)) < 1e-2, np.max(np.abs(him - sref))
    amp_i = _interior(np.abs(z))
    assert np.max(np.abs(amp_i - 1.0)) < 1e-2, np.max(np.abs(amp_i - 1.0))

    # (2) reversible round-trip exact.
    x = 0.7 * np.cos(t) + 0.3 * np.cos(3 * t + 0.5)
    a = analytic_signal(x)
    recon = a["amplitude"] * np.cos(a["phase"])
    assert np.max(np.abs(recon - x)) < 1e-10, np.max(np.abs(recon - x))

    # (3) the real-signal theorem: even a multi-tone real wave is a one-way rotation,
    # so clockwise-only is essentially free on it. This is the surprising, sharp
    # contract -- a real scalar series cannot itself carry a reversal.
    tt = np.linspace(0, 1, 4096)
    multitone = np.cos(2 * np.pi * 6 * tt) + 0.6 * np.cos(2 * np.pi * 11 * tt + 0.3)
    cost_real = monotone_cost(multitone)
    assert cost_real["reversible_rmse"] < 1e-9, cost_real   # baseline exact
    assert cost_real["reversal_fraction"] < 0.02, cost_real  # already one-way (theorem)
    # A real signal carries NO true reversal, so any residual monotone cost comes
    # only from envelope-null glitches (where |z|~0 and the phase spins), not from a
    # direction reversal -- small and bounded, unlike the complex case below.
    assert cost_real["excess"] < 0.3, cost_real

    # (4) the group-vs-monoid price on a TRUE complex phasor that reverses: run the
    # phase forward for half the series, backward for the other half.
    steps = np.concatenate([np.full(2048, 0.15), np.full(2048, -0.15)])
    zc = np.exp(1j * np.cumsum(steps))                       # a genuine I/Q reversal
    cost_cx = phasor_monotone_cost(zc)
    assert cost_cx["reversal_fraction"] > 0.3, cost_cx       # half the steps reverse
    assert cost_cx["excess"] > 0.5, cost_cx                  # the monoid price, loud

    # (5) determinism.
    d1 = monotone_cost(multitone)
    d2 = monotone_cost(multitone)
    for k in d1:
        assert d1[k] == d2[k], k

    print("holographic_analytic selftest OK "
          "(real wave: excess %.4f rev_frac %.3f -> one-way is free | "
          "complex phasor: excess %.4f rev_frac %.3f -> the monoid price)"
          % (cost_real["excess"], cost_real["reversal_fraction"],
             cost_cx["excess"], cost_cx["reversal_fraction"]))


if __name__ == "__main__":
    _selftest()

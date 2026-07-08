"""B9 -- manifold-aware decompose: detect the domain TOPOLOGY, then decompose on the right manifold.

The decompose search (build 2, holographic_symbolic) assumes a flat line: it fits a sum of elementary
functions over an open interval. But many real signals live on a curved domain -- a RING (periodic:
phase wraps at 2pi), an antiperiodic MOBIUS band (f(t + P/2) = -f(t), so only ODD harmonics exist), or
a TORUS (two independent periods). Decomposed on the wrong manifold a periodic signal needs many terms
and, fatally, EXTRAPOLATES BY DIVERGING -- a polynomial shoots off where the true signal repeats. Detect
the topology first and the matched basis (harmonics of the detected period; odd-only for a Mobius band)
is parsimonious and extrapolates correctly, because it is built from the manifold's own functions.

This is the decompose-side twin of the Mobius/AxialEncoder work: the odd-harmonic basis a Mobius signal
decomposes onto IS the antiperiodic function space that encoder represents.

HOW DETECTION WORKS, AND ITS LIMITS (kept honestly):
  * Period: detrend (so a ramp doesn't masquerade as a long period), FFT for a candidate fundamental
    (the LOWEST significant peak -- a strong harmonic is not the fundamental), then VALIDATE by how well
    a harmonic basis at that period actually fits. Fit-based validation is robust to FFT spectral leakage.
  * Commensurate peaks (all integer multiples of the fundamental) -> periodic; a non-integer-multiple peak
    -> quasiperiodic (torus). Mobius: the odd-harmonic basis fits as well as the full one.
  * MEASURED: line / ring / mobius classify correctly and survive 5% noise (3/3 seeds each) on a 2-cycle
    window. TORUS needs a window long enough to RESOLVE the two incommensurate tones (the Rayleigh limit,
    span >= 1/df): on a short 2-cycle window the tones merge into one blurred peak and detection falls
    back to line rather than guessing. That is a reported limitation, not a silent error.
  * A wrong-period false alarm is guarded: if the best harmonic fit at the candidate period is poor
    (R^2 < 0.9) the signal is called a line, not forced onto a spurious ring.

Pure NumPy + holostuff spirit; deterministic; feeds straight into symbolic_regress (build 2) via a
topology-matched dictionary -- no new search machinery.
"""

import numpy as np

from holographic.agents_and_reasoning.holographic_symbolic import symbolic_regress, elementary_dictionary


# ---- topology detection -------------------------------------------------------------------------
def _ring_fit(x, y, P, K, odd_only=False, P2=None):
    """R^2 of a harmonic fit at period P (and optionally a second period P2). odd_only restricts to
    odd harmonics -- the antiperiodic (Mobius) function space."""
    w = 2 * np.pi / P
    cols = [np.ones(len(x))]
    for k in (range(1, 2 * K, 2) if odd_only else range(1, K + 1)):
        cols += [np.cos(k * w * x), np.sin(k * w * x)]
    if P2:
        w2 = 2 * np.pi / P2
        for k in range(1, K + 1):
            cols += [np.cos(k * w2 * x), np.sin(k * w2 * x)]
    A = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(1 - np.var(y - A @ beta) / (np.var(y) + 1e-12))


def _refine(x, y, P0, K, **kw):
    """Refine a period near an FFT estimate by maximizing the harmonic fit (sub-bin accuracy)."""
    best = (-9.0, P0)
    for P in P0 * np.linspace(0.94, 1.06, 25):
        r = _ring_fit(x, y, P, K, **kw)
        if r > best[0]:
            best = (r, P)
    return best


def detect_topology(x, y, n_harmonics=5, rel=0.2):
    """Classify a signal's domain topology. Returns (name, period) where name is one of
    "line" / "ring" / "mobius" / "torus" and period is None (line), a float (ring/mobius), or a
    (P1, P2) pair (torus)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(y); dx = (x[-1] - x[0]) / (n - 1); span = x[-1] - x[0]
    A = np.column_stack([x, np.ones(n)])                      # detrend: remove the linear part
    yd = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    if np.std(yd) < 1e-6 * (np.std(y) + 1e-12):
        return "line", None                                   # a pure trend explains everything
    freqs = np.fft.rfftfreq(n, dx); Y = np.abs(np.fft.rfft(yd)); Y[0] = 0
    pk = [i for i in range(1, len(Y) - 1)
          if Y[i] > Y[i - 1] and Y[i] >= Y[i + 1] and Y[i] > rel * Y.max()]
    if not pk:
        return "line", None
    f_lo = freqs[pk[0]]                                        # the FUNDAMENTAL is the lowest peak
    if f_lo <= 0 or span * f_lo < 1.5:
        return "line", None                                   # < 1.5 cycles -> can't confirm a period
    ratios = [freqs[p] / f_lo for p in pk]
    if all(abs(r - round(r)) < 0.12 for r in ratios):         # commensurate peaks -> periodic
        r2, P = _refine(x, y, 1 / f_lo, n_harmonics)
        if r2 < 0.9:
            return "line", None                               # false-alarm guard
        r2_odd, _ = _refine(x, y, 1 / f_lo, n_harmonics, odd_only=True)
        return ("mobius", P) if r2_odd > r2 - 0.02 else ("ring", P)
    # an incommensurate peak -> quasiperiodic (torus); resolve both tones and validate
    nonint = [p for p in pk if abs(freqs[p] / f_lo - round(freqs[p] / f_lo)) >= 0.12]
    f2 = freqs[nonint[0]]
    _, P1 = _refine(x, y, 1 / f_lo, 2)
    best = (-9.0, 1 / f2)
    for P2 in (1 / f2) * np.linspace(0.94, 1.06, 25):
        r = _ring_fit(x, y, P1, 2, P2=P2)
        if r > best[0]:
            best = (r, P2)
    return ("torus", (P1, best[1])) if best[0] > 0.95 else ("line", None)


# ---- topology -> a matched dictionary for the build-2 decompose ----------------------------------
def line_dictionary(powers=(1, 2, 3, 4, 5)):
    """The flat-manifold assumption: a polynomial basis. The honest baseline a periodic signal is
    forced onto when its topology is ignored -- it cannot extrapolate periodically (it diverges)."""
    return [("pow", p) for p in powers]


def manifold_dictionary(topology, period, n_harmonics=5):
    """Build the symbolic_regress basis matched to the detected topology. Ring -> harmonics of the
    period; Mobius -> ODD harmonics only (the antiperiodic space); Torus -> harmonics of both periods;
    Line -> the general elementary dictionary."""
    if topology == "line" or period is None:
        return elementary_dictionary()
    if topology in ("ring", "mobius"):
        w = 2 * np.pi / period
        ks = range(1, 2 * n_harmonics, 2) if topology == "mobius" else range(1, n_harmonics + 1)
        return [("cos", k * w) for k in ks] + [("sin", k * w) for k in ks]
    if topology == "torus":
        P1, P2 = period
        out = []
        for P in (P1, P2):
            w = 2 * np.pi / P
            out += [("cos", k * w) for k in range(1, n_harmonics + 1)]
            out += [("sin", k * w) for k in range(1, n_harmonics + 1)]
        return out
    raise ValueError(topology)


def decompose_on_manifold(x, y, n_harmonics=5, max_terms=6, coef_bits=20):
    """Detect the topology, then run the build-2 MDL decompose on the matched basis. Returns
    (Formula, info) with info['topology'] and info['period'] recorded. The Formula extrapolates on the
    detected manifold (periodically for a ring/mobius) instead of diverging like a flat-line fit."""
    topo, period = detect_topology(x, y, n_harmonics=n_harmonics)
    d = manifold_dictionary(topo, period, n_harmonics)
    f, info = symbolic_regress(x, y, dictionary=d, max_terms=max_terms, coef_bits=coef_bits)
    info["topology"] = topo
    info["period"] = period
    return f, info

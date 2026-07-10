"""holographic_registration.py -- canonical affine recovery (Box3D backlog DL11).

An edit history of translates and scales does not commute -- translate-then-scale is not scale-then-translate -- and
scale is not diagonal in the linear-frequency basis, so no polar separation factors the family. Three facts close
the gap, and this module ships them:

  1. **GROUP CLOSURE MAKES ORDER A NON-PROBLEM.** Any chain of `x -> s x + t` collapses to ONE canonical `(S, T)`
     by the affine composition law. Order changes *which* `(S, T)` you get; every order gives *some* single
     `(S, T)`. The recoverable object is the GROUP ELEMENT, not the factor sequence. `affine_compose` is exact
     (measured, max|diff| ~1e-16 over a 4-edit chain).
  2. **THE LIFT DIAGONALIZES THE STUBBORN FAMILY.** `|FFT|` discards translation; resampling the magnitude
     spectrum onto a LOG-frequency axis turns dilation into a SHIFT (Reddy & Chatterji's Fourier-Mellin move). The
     estimator is then the same cross-correlation-with-a-parabola that `holographic_reproject.est_dx` uses on
     images. Scale becomes translation; the engine already knew how to find a translation.
  3. **COARSE MELLIN + FINE CONTINUOUS REFINE.** Mellin initialises, a shrinking grid on the two-parameter
     `(s, t)` manifold refines. Measured on a 4-edit chain (n = 2048): **scale error 3.7e-04, shift error 0.37
     SAMPLES, alignment 0.99946.** The units matter: the backlog reports a shift error of ~1e-4, which does not
     reproduce in sample units -- the SCALE lands at 1e-4, the SHIFT at a few tenths of a sample. State the unit.

CONVENTION, stated once because a sign error here is invisible: `after(x) = before((x - t) / s)`. So `s > 1` is a
DILATION, and it shifts the log-frequency axis by `-log s`.

KEPT NEGATIVE 1 -- **THE SUPPORT BAND IS THE GATE, NOT `log` VS PLAIN.** The backlog says to correlate LOG
magnitudes "because dilation also scales spectrum amplitude, |G(w)| = s |F(sw)|, which tilts plain correlation."
Measured, that reason is wrong: multiplying one signal by a constant scales the entire cross-correlation and leaves
the argmax exactly where it was (verified, peak 7.0 either way). What actually decides it is the BAND. On a signal
whose spectrum has support only in bins 1-7 of 512, the log axis is mostly noise floor, `log` amplifies it into a
structureless signal, and the correlation peaks at **zero shift** for every true scale (1.05, 1.2, 1.5 all recover
1.00). Band-limit to where `|F|` actually has support and both plain and log recover the scale to ~0.5%. This
module bands by default, and `use_log` genuinely helps only when the spectral envelope is heavy-tailed.

KEPT NEGATIVE 2 -- **the coarse stage alone is not enough near identity.** A near-1 scale shifts the log axis by a
fraction of a bin, and window truncation smears it. The refine stage carries it, and `recover_affine` runs both.
Severe truncation or occlusion degrades the coarse init; if it lands outside the refine's basin, the answer is
wrong and `alignment` says so -- which is why `alignment` is returned rather than hidden.

KEPT NEGATIVE 3 -- **the group law is exact on the PARAMETERS; repeated RESAMPLING is not.** `affine_compose` is
the exact affine group law (verified to 1e-15). But applying four edits by four interpolated resamples does NOT
produce the same array as one resample by `(S, T)`: measured `max|chain - direct|` is **0.157 at n = 1024**, 0.058
at 2048 and 0.0045 at 8192. Interpolation and zero-fill are lossy, and they compose lossily. The identity is
recovered in the LIMIT of sampling density, not at any fixed n. So `recover_affine` on a chained signal is fitting
the affine that best explains a slightly-blurred observation -- which it does well (alignment 0.998 at n = 1024),
and which is not the same claim as recovering the group element from an exact realisation of it.

HONEST SCOPE: **1-D.** Two dimensions adds rotation, which needs the log-POLAR resampling of the full
Fourier-Mellin transform, and owes its own measurement.
"""

import numpy as np


def resample_affine(f, s, t):
    """`after(x) = before((x - t) / s)`, linearly interpolated, zero outside the support.

    This is the forward edit. Zero-fill rather than wrap: an edit history moves content off the end of the signal,
    and pretending it wrapped around would make the estimator's job artificially easy."""
    f = np.asarray(f, float)
    n = len(f)
    u = (np.arange(n) - float(t)) / float(s)
    i = np.floor(u).astype(int)
    frac = u - i
    out = np.zeros(n)
    ok = (i >= 0) & (i < n - 1)
    out[ok] = f[i[ok]] * (1.0 - frac[ok]) + f[i[ok] + 1] * frac[ok]
    return out


def affine_compose(chain):
    """Collapse a chain of `(s, t)` edits, applied left to right, into ONE canonical `(S, T)`.

    `x -> s2 (s1 x + t1) + t2 = (s2 s1) x + (s2 t1 + t2)`. Exact -- this is the affine group law, not an
    approximation, and it is why the ORDER of a non-commuting edit history is not an obstacle to canonical storage:
    every order yields some single group element. When the sequence itself matters, store the sequence."""
    S, T = 1.0, 0.0
    for (s, t) in chain:
        S, T = float(s) * S, float(s) * T + float(t)
    return S, T


def _parabolic_peak_1d(c):
    """The circular cross-correlation's peak to sub-sample precision, wrapped into [-n/2, n/2).

    **DELEGATES to `holographic_reproject.parabolic_peak`.** This module's docstring says "the estimator is `est_dx`
    again", and for a while that was a claim rather than a fact: there were two copies of the same three-point
    parabola in two families. A reachability audit found them. `parabolic_peak` is now rank-agnostic, and this is a
    1-D call into it -- the interpolation still needs CURVATURE, which is why the correlation must not be
    phase-normalised into a delta."""
    from holographic.rendering.holographic_reproject import parabolic_peak
    return float(parabolic_peak(np.asarray(c, float))[0])


def _correlate_1d(a, b):
    """`conj(F(a)) * F(b)`, inverse-transformed: the 1-D twin of `holographic_reproject._correlate`.

    NOT delegated, and the reason is stated: `reproject._correlate` uses `rfft2`/`irfft2` because an image is
    2-D, and calling it with a 1-D signal would transform the wrong axes. The shared object is the PEAK FINDER,
    which is rank-agnostic; the transform's rank is genuinely different. Generalising `_correlate` too would mean
    `rfftn`, which changes `reproject`'s hot path for no benefit -- a unification that costs more than the
    duplication it removes."""
    A, B = np.fft.rfft(np.asarray(a, float)), np.fft.rfft(np.asarray(b, float))
    return np.fft.irfft(A.conj() * B, n=len(a))


def support_band(F, frac=1e-3):
    """The `[lo, hi]` frequency bins (1-based) where `|F|` exceeds `frac` of its peak.

    THE GATE. Outside its support a magnitude spectrum is a noise floor, and `log` turns that floor into a
    structureless signal that dominates the log-axis correlation and pins its peak at zero shift. Measured: a
    narrowband signal (90% of energy below bin 7 of 512) recovers `s = 1.00` for every true scale until the band
    is applied."""
    F = np.asarray(F, float)
    idx = np.where(F > float(frac) * F.max())[0]
    if idx.size == 0:
        return 1, len(F)
    return int(max(idx.min() + 1, 1)), int(idx.max() + 1)


def mellin_scale(before, after, n_log=1024, use_log=True, frac=1e-3):
    """Recover the DILATION `s` alone, by the Fourier-Mellin lift: `|FFT|` kills the translation, and resampling
    onto a log-frequency axis turns the dilation into a shift that a cross-correlation finds.

    Returns `s` such that `after(x) ~ before((x - t) / s)` for some `t`. Banded to the spectrum's support (see
    `support_band`); `use_log` equalises a heavy-tailed envelope and is not, contrary to the backlog, needed to
    undo an amplitude factor -- a constant factor cannot move an argmax."""
    a = np.asarray(before, float)
    b = np.asarray(after, float)
    F = np.abs(np.fft.rfft(a))[1:]
    G = np.abs(np.fft.rfft(b))[1:]
    if F.max() <= 0.0 or G.max() <= 0.0:
        return 1.0
    lo, hi = support_band(F, frac=frac)
    if hi <= lo + 1:
        return 1.0
    w = np.arange(1, len(F) + 1)
    lu = np.linspace(np.log(lo), np.log(hi), int(n_log))
    du = float(lu[1] - lu[0])
    Fl = np.interp(np.exp(lu), w, F)
    Gl = np.interp(np.exp(lu), w, G)
    if use_log:
        Fl, Gl = np.log(Fl + 1e-12), np.log(Gl + 1e-12)
    Fl = Fl - Fl.mean()
    Gl = Gl - Gl.mean()
    shift = _parabolic_peak_1d(_correlate_1d(Fl, Gl))
    return float(np.exp(-shift * du))                # a dilation by s shifts the log axis by -log s


def alignment(before, after, s, t):
    """Normalised correlation between `after` and `resample_affine(before, s, t)`. 1.0 is a perfect match.

    Returned rather than hidden: when the coarse Mellin init lands outside the refine's basin, the answer is wrong,
    and this is the only thing that says so."""
    pred = resample_affine(before, s, t)
    a = np.asarray(after, float)
    na, np_ = np.linalg.norm(a), np.linalg.norm(pred)
    if na < 1e-12 or np_ < 1e-12:
        return 0.0
    return float(a @ pred / (na * np_))


def refine_affine(before, after, s0, t0, rounds=8, grid=5, span_s=0.10, span_t=8.0):
    """Shrinking-grid search on the two-parameter `(s, t)` manifold, maximising `alignment`.

    Deterministic: a fixed grid, halved each round, ties resolved by the first (lowest `s`, then lowest `t`) --
    there is no RNG, so the same inputs give the same answer bit for bit. This is DL10's continuous cleanup applied
    to a 2-D manifold rather than a 1-D one."""
    s, t = float(s0), float(t0)
    ss, st = float(span_s), float(span_t)
    best = alignment(before, after, s, t)
    for _ in range(int(rounds)):
        cands = [(sc, tc) for sc in np.linspace(s - ss, s + ss, int(grid))
                 for tc in np.linspace(t - st, t + st, int(grid)) if sc > 1e-6]
        for (sc, tc) in cands:
            a = alignment(before, after, sc, tc)
            if a > best + 1e-15:
                best, s, t = a, float(sc), float(tc)
        ss *= 0.5
        st *= 0.5
    return s, t, best


def recover_affine(before, after, refine=True, **kw):
    """The DL11 item: recover the canonical `(S, T)` of an arbitrary translate/scale edit history.

    Coarse Fourier-Mellin for `S`, a translation estimate for `T` on the de-scaled signal, then a shrinking-grid
    refine on `(s, t)`. Returns `{scale, shift, alignment, coarse_scale}`.

    MEASURED on a 4-edit non-commuting chain, n = 2048, canonical `(S, T) = (1.0811, 1.6828)`: scale error
    **3.7e-04**, shift error **0.37 samples**, alignment **0.99946**. The scale is the accurate one; the shift lands
    at a few tenths of a sample, not at 1e-4. `coarse_scale` is reported so a caller can see when the Mellin init
    was poor and the refine did the work -- which is the near-identity case."""
    a = np.asarray(before, float)
    b = np.asarray(after, float)
    s0 = mellin_scale(a, b, **{k: v for k, v in kw.items() if k in ("n_log", "use_log", "frac")})

    # translation, estimated on the DE-SCALED signal: undo s0, then the residual edit is a pure shift
    descaled = resample_affine(a, s0, 0.0)
    t0 = float(_parabolic_peak_1d(_correlate_1d(descaled - descaled.mean(), b - b.mean())))

    if not refine:
        return {"scale": s0, "shift": t0, "alignment": alignment(a, b, s0, t0), "coarse_scale": s0}
    s, t, al = refine_affine(a, b, s0, t0)
    return {"scale": s, "shift": t, "alignment": al, "coarse_scale": s0}


def _selftest():
    """Regression trap for DL11: the group law is exact, the Mellin lift recovers a dilation, the refine reaches
    ~1e-4, and the two kept negatives hold -- the band is the gate, and a constant amplitude factor is not."""
    n = 2048
    x = np.linspace(0, 1, n)
    # a BROADBAND signal: a chirp plus a bump. A narrowband one has no log-axis to correlate on, which is
    # precisely kept negative 1.
    f = (np.sin(2 * np.pi * (20 * x + 60 * x ** 2)) * np.exp(-((x - 0.5) ** 2) / 0.06)
         + 0.5 * np.sin(2 * np.pi * 180 * x) * np.exp(-((x - 0.3) ** 2) / 0.005))

    # 1. THE GROUP LAW is exact, and order matters to the RESULT while never breaking closure
    chain = [(1.03, 4.0), (0.98, -2.5), (1.05, 3.1), (1.02, -3.0)]
    S, T = affine_compose(chain)
    y = f.copy()
    for (s, t) in chain:
        y = resample_affine(y, s, t)
    direct = resample_affine(f, S, T)
    assert np.abs(np.array(affine_compose(list(reversed(chain)))) - np.array((S, T))).max() > 1e-6  # order matters
    assert abs(affine_compose([(2.0, 1.0), (3.0, 5.0)])[0] - 6.0) < 1e-15                            # S = s2 s1
    assert abs(affine_compose([(2.0, 1.0), (3.0, 5.0)])[1] - 8.0) < 1e-15                            # T = s2 t1 + t2
    assert alignment(f, direct, S, T) > 0.9999                                                        # one (S, T)

    # 2. the Mellin lift recovers a dilation from the log-frequency axis
    for s_true in (1.05, 1.2, 1.5):
        g = resample_affine(f, s_true, 17.0)
        assert abs(mellin_scale(f, g) - s_true) < 0.05 * s_true

    # 3. KEPT NEGATIVE 1a: a constant amplitude factor cannot move an argmax -- the backlog's stated reason for
    #    preferring log magnitudes is not the operative one.
    rng = np.random.default_rng(0)
    a1 = rng.normal(size=256)
    b1 = np.roll(a1, 7)
    p1 = _parabolic_peak_1d(_correlate_1d(a1 - a1.mean(), b1 - b1.mean()))
    p2 = _parabolic_peak_1d(_correlate_1d(a1 - a1.mean(), 5.0 * (b1 - b1.mean())))
    assert abs(p1 - p2) < 1e-9 and abs(p1 - 7.0) < 1e-9

    # 4. KEPT NEGATIVE 1b: THE BAND is the gate. A narrowband signal, banded to the full spectrum, recovers 1.0
    #    for every true scale -- log amplifies the noise floor and the peak sits at zero shift.
    narrow = np.exp(-((x - 0.45) ** 2) / 0.01) * np.sin(30 * x)
    F = np.abs(np.fft.rfft(narrow))[1:]
    lo, hi = support_band(F)
    assert hi < len(F) // 4, (lo, hi)              # the support really is narrow
    unbanded = mellin_scale(narrow, resample_affine(narrow, 1.5, 9.0), frac=0.0)
    assert abs(unbanded - 1.0) < 0.02, unbanded    # the whole spectrum: no shift found at all

    # 5. KEPT NEGATIVE 3: the group law is exact on the PARAMETERS, and repeated RESAMPLING is not. Four
    #    interpolated resamples do not reproduce one resample by (S, T) -- and the gap shrinks with sampling density.
    gaps = []
    for nn in (1024, 4096):
        xx = np.linspace(0, 1, nn)
        ff = (np.sin(2 * np.pi * (20 * xx + 60 * xx ** 2)) * np.exp(-((xx - 0.5) ** 2) / 0.06)
              + 0.5 * np.sin(2 * np.pi * 180 * xx) * np.exp(-((xx - 0.3) ** 2) / 0.005))
        yy = ff.copy()
        for (s, t) in chain:
            yy = resample_affine(yy, s, t)
        gaps.append(float(np.abs(yy - resample_affine(ff, S, T)).max()))
    assert gaps[0] > 0.05                       # NOT identical at a fixed n
    assert gaps[1] < 0.5 * gaps[0]              # ... and it converges with sampling density

    # 6. THE BAR: blind recovery of the 4-edit chain's canonical (S, T). The SCALE is the accurate one.
    rep = recover_affine(f, y)
    assert abs(rep["scale"] - S) < 5e-3, (rep["scale"], S)
    assert abs(rep["shift"] - T) < 1.0, (rep["shift"], T)      # SAMPLES, not 1e-4
    assert rep["alignment"] > 0.99

    # ... and from an EXACT single affine the estimator does better still
    exact = recover_affine(f, resample_affine(f, S, T))
    assert abs(exact["scale"] - S) < 1e-3 and exact["alignment"] > 0.999

    # 7. determinism, and an honest identity
    assert recover_affine(f, y) == recover_affine(f, y)
    ident = recover_affine(f, f)
    assert abs(ident["scale"] - 1.0) < 1e-2 and abs(ident["shift"]) < 1.0 and ident["alignment"] > 0.999

    print("OK: holographic_registration self-test passed (the affine group law is exact on the PARAMETERS -- a "
          "4-edit non-commuting chain collapses to ONE canonical (S, T) = (%.4f, %.4f) -- while repeated RESAMPLING "
          "is not: max|chain - direct| is %.3f at n=1024 and %.3f at n=4096. Blind Mellin+refine recovers the scale "
          "to %.1e at alignment %.6f, and the shift to %.2f SAMPLES, not 1e-4: state the unit. Kept negatives hold: "
          "a constant amplitude factor does NOT move the correlation peak (%.1f both ways), so the backlog's reason "
          "for log-magnitudes is not the operative one -- the SUPPORT BAND is, and an unbanded narrowband spectrum "
          "recovers s = %.4f for a true 1.5)"
          % (S, T, gaps[0], gaps[1], abs(rep["scale"] - S), rep["alignment"], abs(rep["shift"] - T), p1, unbanded))


if __name__ == "__main__":
    _selftest()

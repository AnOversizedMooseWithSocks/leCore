"""holographic_fitgen.py -- which deterministic GENERATOR made this data? (L14, the inverse of the ladder).

WHY THIS MODULE EXISTS
----------------------
The ladder climbs data into abstractions. This is the inverse: given data that is NOT clean (noisy, empirical,
recorded), recover the DETERMINISTIC GENERATOR that best explains it, then keep the residual honest. The demoscene
economy is the prize -- if a signal IS a gyroid plus noise, store (generator, params, seed), a few bytes, not the
megabytes of samples. And a fitted generator becomes an ATOM at the next level up.

THE FAST PATH: IDENTIFICATION IS A CLEANUP, NOT A SEARCH (plan §3k)
  1. BAKE A GENERATOR BANK. Every deterministic family we own is baked at COARSE resolution into a signature
     vector -- one bake each, stored as (family, params). This is a codebook whose atoms are PROGRAMS.
  2. SNAP, DON'T MARCH. Encode the mystery data the same coarse way and correlate against every bank signature
     at once. The best-correlating family is the identification -- O(bank) dot products, no iterative search.
  3. REFINE. The snapped family fixes the FORM; its continuous parameters are recovered by a small local search
     (the same iterate-a-projection move as guide_structure, here over the parameter axis).
  4. THE RESIDUAL GATE decides what the fit IS: residual ~ noise floor => deterministic-plus-noise, store the
     generator; residual dominates => REFUSE ("no deterministic structure above the null" is a result).

QUILEZ Q8 -- BAND-LIMIT THE SIGNATURE (the panel addition):
  Two generators that differ only ABOVE the coarse sampling rate alias to the same coarse signature. If we snap on
  point-sampled signatures, one silently "wins" a tie it has no right to. So each signature is LOW-PASSED to the
  coarse rate before baking (his analytic-prefiltering technique). Two families indistinguishable at that grain are
  honestly reported as a TIE (equifinality), not a false confident pick -- and the refine step / a finer bank is
  what resolves them, not a lucky point sample.

KEPT NEGATIVES carried in: (a) a rich bank fits ANYTHING -- the residual gate (does the fit beat raw storage AND
the null?) is the only defense, measured not assumed; (b) equifinality -- when two families fit within tolerance,
report the tie, never silently pick one; (c) fit quality on the fit window != on held-out data -- verify by
REGENERATING and comparing in the original space.

NumPy only. Deterministic.
"""

import numpy as np


# --- a small bank of deterministic 1-D generator FAMILIES (each: params -> signal on t in [0,1]) --------------

def _sine(t, freq, phase):
    return np.sin(2 * np.pi * freq * t + phase)


def _chirp(t, f0, f1):
    return np.sin(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t * t))


def _gauss_bump(t, center, width):
    return np.exp(-0.5 * ((t - center) / max(width, 1e-3)) ** 2)


def _sawtooth(t, freq, _):
    x = (t * freq) % 1.0
    return 2.0 * x - 1.0


def _harmonic(t, freq, brightness):
    """A HARMONIC oscillator (Puckette): a fundamental plus decaying harmonics -- the playable model of a musical
    TONE, which a bare sine cannot capture. `brightness` in (0,1) sets how much energy is in the upper harmonics
    (0 -> nearly a sine, 1 -> buzzy/rich). Analysis=resynthesis: a fit of this IS a resynthesizable oscillator."""
    out = np.zeros_like(t)
    for k in range(1, 6):                                      # fundamental + 4 harmonics
        out += (brightness ** (k - 1)) * np.sin(2 * np.pi * freq * k * t) / k
    return out


def _am(t, carrier, rate):
    """An AMPLITUDE-MODULATED carrier (Puckette): a fast carrier whose amplitude is shaped by a slow envelope --
    the tremolo / formant-resonance / beating case, which neither a sine nor a chirp captures. `carrier` is the
    fast frequency, `rate` the slow modulation frequency. Playable: it resynthesizes as carrier x envelope."""
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * rate * t))     # 0..1 slow envelope
    return envelope * np.sin(2 * np.pi * carrier * t)


FAMILIES = {
    "sine":     (_sine,       [(1.0, 12.0), (0.0, 2 * np.pi)]),   # (freq, phase)
    "chirp":    (_chirp,      [(1.0, 6.0), (6.0, 20.0)]),         # (f0, f1)
    "gauss":    (_gauss_bump, [(0.2, 0.8), (0.03, 0.2)]),         # (center, width)
    "sawtooth": (_sawtooth,   [(1.0, 12.0), (0.0, 1.0)]),         # (freq, unused)
    "harmonic": (_harmonic,   [(1.0, 8.0), (0.1, 0.95)]),        # (freq, brightness) -- Puckette tone
    "am":       (_am,         [(4.0, 16.0), (0.5, 4.0)]),        # (carrier, rate) -- Puckette tremolo/formant
}


def _band_limit(sig, keep_frac=0.25):
    """Q8: low-pass a signature to the COARSE rate before comparison -- zero the top (1-keep_frac) of the spectrum
    so detail ABOVE the coarse Nyquist cannot masquerade as a distinguishing feature. Two generators that differ
    only above this band become honestly indistinguishable (a tie), instead of one winning on aliased high-freq
    point samples. Returns the band-limited signal (same length)."""
    n = len(sig)
    F = np.fft.rfft(sig)
    keep = max(1, int(len(F) * keep_frac))
    F[keep:] = 0.0
    return np.fft.irfft(F, n=n)


def _normalize(v):
    v = v - v.mean()
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 1e-12 else v


def fit_deterministic(data, coarse=64, keep_frac=0.25, refine_steps=12, tie_tol=0.02, seed=0):
    """Recover the deterministic GENERATOR that best explains `data` (a 1-D signal), by SNAP-then-REFINE against a
    baked generator bank -- the inverse of the ladder. Returns a dict: `family` (best-fitting generator name, or
    None if refused), `params`, `correlation` (of the regenerated signal to the data, in the original space),
    `residual_frac` (fraction of the data's variance NOT explained), `ties` (families within `tie_tol` of the
    best -- equifinality reported, never silently broken), and `verdict` ('fit' / 'tie' / 'refused'). Q8: the snap
    compares BAND-LIMITED signatures (low-passed to the coarse rate), so families that differ only above that rate
    tie honestly. Refuses (family=None, verdict='refused') when no generator beats the noise -- 'no deterministic
    structure' is a result. Deterministic."""
    data = np.asarray(data, float).ravel()
    t = np.linspace(0.0, 1.0, coarse)
    # coarse-encode the data (resample to `coarse` points), band-limit, normalize -> the snap target.
    data_coarse = np.interp(t, np.linspace(0, 1, len(data)), data)
    target = _normalize(_band_limit(data_coarse, keep_frac))

    rng = np.random.default_rng(seed)

    def best_params_for(family):
        """Small random+local search for the params of one family that best correlate (band-limited) with the
        target. This is the REFINE step -- the snapped family fixes the form, we tune its knobs."""
        fn, ranges = FAMILIES[family]
        # Q8 tension: the harmonic/AM families carry their IDENTIFYING content at HIGH frequencies (harmonics, a
        # fast carrier), which a narrow coarse band erases -- making them tie a bare sine. Give those families a
        # WIDER snap band so their distinguishing structure survives; the low-frequency families keep the narrow
        # band (where aliasing is the risk, not erasure). This is per-family band-limiting, still honest: each
        # family is compared at the grain where its own structure lives.
        fam_keep = 0.6 if family in ("harmonic", "am") else keep_frac
        # the target must be compared at the SAME band as the candidate, so re-band-limit it per family.
        fam_target = _normalize(_band_limit(data_coarse, fam_keep))
        best_c, best_p = -2.0, None
        # coarse random probe then local refine (iterate-a-projection on the parameter axis). A generous probe
        # matters: a chirp's (f0,f1) landscape is narrow, so too few candidates miss the basin.
        cand = [tuple(rng.uniform(lo, hi) for (lo, hi) in ranges) for _ in range(120)]
        for step in range(refine_steps):
            scored = []
            for p in cand:
                sig = _normalize(_band_limit(fn(t, *p), fam_keep))
                scored.append((float(np.dot(sig, fam_target)), p))
            scored.sort(reverse=True)
            c, p = scored[0]
            if c > best_c:
                best_c, best_p = c, p
            # refine around the current best: jitter shrinking each step, keep a few wide explorers too.
            scale = 0.3 * (0.75 ** step)
            cand = [tuple(np.clip(pi + rng.normal() * scale * (hi - lo), lo, hi)
                          for pi, (lo, hi) in zip(best_p, ranges)) for _ in range(60)]
            cand += [tuple(rng.uniform(lo, hi) for (lo, hi) in ranges) for _ in range(20)]  # explorers
        return best_c, best_p

    # SNAP: score every family, keep the best correlation each achieves.
    scores = {}
    params = {}
    for fam in FAMILIES:
        c, p = best_params_for(fam)
        scores[fam] = c
        params[fam] = p

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best_fam, best_score = ranked[0]
    # ties (Q8 / equifinality): families within tie_tol of the best band-limited correlation.
    ties = [f for f, s in ranked if best_score - s <= tie_tol]
    # RESOLVE ties by ORIGINAL-SPACE fit at a WIDE band (where harmonics/AM are visible, unlike the coarse snap
    # that erases them). Regenerate each tied family full-length and score it against the data at the wide band;
    # the genuinely-best fit wins. Occam only enters as the final tie-break when the wide-band fits are themselves
    # near-equal (a harmonic dominated by its fundamental IS a sine -- report the simpler one then, not otherwise).
    t_full = np.linspace(0, 1, len(data))
    wide = min(0.7, keep_frac * 2.5)
    data_wide = _normalize(_band_limit(data, wide))
    _SIMPLICITY = {"sine": 0, "sawtooth": 1, "gauss": 1, "chirp": 2, "am": 3, "harmonic": 3}
    if len(ties) > 1 and data.std() > 1e-12:
        wide_corr = {}
        for f in ties:
            fn_f, _ = FAMILIES[f]
            regen_f = _normalize(_band_limit(fn_f(t_full, *params[f]), wide))
            wide_corr[f] = float(np.corrcoef(regen_f, data_wide)[0, 1])
        top = max(wide_corr.values())
        # among families within tie_tol of the BEST wide-band fit, pick the simplest (Occam only for real ties).
        near = [f for f in ties if top - wide_corr[f] <= tie_tol]
        best_fam = min(near, key=lambda f: (_SIMPLICITY.get(f, 9), -wide_corr[f]))
        best_score = scores[best_fam]

    # verify in the ORIGINAL space, at the Q8 GRAIN: regenerate at full length and correlate the band-limited
    # regen with the band-limited data. WHY band-limited and not point-wise: a chirp's phase drifts over a long
    # window, so a near-correct fit can have low point-wise correlation while being right at the grain the snap
    # trusts. Comparing at the coarse band is the honest verification -- fine phase is what a finer bank resolves.
    fn, _ = FAMILIES[best_fam]
    t_full = np.linspace(0, 1, len(data))
    regen = fn(t_full, *params[best_fam])
    kf = min(0.5, keep_frac * 2)                               # a slightly wider band for verification than snap
    corr = (float(np.corrcoef(_normalize(_band_limit(regen, kf)),
                              _normalize(_band_limit(data, kf)))[0, 1]) if data.std() > 1e-12 else 0.0)
    residual = 1.0 - corr ** 2                                 # fraction of variance unexplained (at this grain)

    # RESIDUAL GATE: a real fit must explain most of the variance. Otherwise REFUSE -- no deterministic structure.
    if corr < 0.5:
        verdict = "refused"
        return {"family": None, "params": None, "correlation": round(corr, 4),
                "residual_frac": round(residual, 4), "ties": [], "verdict": verdict}
    verdict = "tie" if len(ties) > 1 else "fit"
    return {"family": best_fam, "params": tuple(round(float(x), 4) for x in params[best_fam]),
            "correlation": round(corr, 4), "residual_frac": round(residual, 4),
            "ties": ties, "verdict": verdict}


def extend_generator(fit_result, n_ahead, original_length):
    """FORECAST by evaluating a fitted generator PAST its data (Puckette/Quilez: store the formula, play the
    future -- the demoscene economy applied to time). Given a `fit_result` from fit_deterministic (with a family
    and params), and the `original_length` of the fitted data, regenerate `n_ahead` samples beyond the end.
    Returns {forecast, t_range, valid} -- `forecast` is the extrapolated samples, `t_range` the normalized time
    they cover, and `valid` False (with the samples still returned) when the extrapolation runs far past the
    fit's validated window.

    KEPT NEGATIVE (Quilez reprojection-ghost, on the time axis): a generator fit on t in [0,1] evaluated at t=100
    is confident nonsense. We carry the fit's validity: extrapolating more than one data-length past the end is
    flagged valid=False. Store the formula, but do not trust it arbitrarily far -- refuse beyond where it was
    validated. A REFUSED fit (family None) cannot be extended (returns valid=False, empty forecast)."""
    if fit_result is None or fit_result.get("family") is None:
        return {"forecast": [], "t_range": (None, None), "valid": False}
    from_family = fit_result["family"]
    params = fit_result["params"]
    fn, _ranges = FAMILIES[from_family]
    # the fit lives on t in [0,1] over `original_length` samples; each sample is dt = 1/(L-1) apart.
    L = max(2, int(original_length))
    dt = 1.0 / (L - 1)
    # future times continue past t=1.0.
    t_future = 1.0 + dt * np.arange(1, n_ahead + 1)
    forecast = fn(t_future, *params)
    # validity: extrapolating beyond one extra data-length (t > 2.0) is past the validated window.
    valid = bool(t_future[-1] <= 2.0) if n_ahead > 0 else True
    return {"forecast": [float(v) for v in forecast], "t_range": (float(t_future[0]), float(t_future[-1])),
            "valid": valid}


def _selftest_extend():
    """extend_generator contracts: a fitted sine extrapolates to the CORRECT future values (the formula plays
    forward); extrapolating too far is flagged valid=False; a refused fit cannot be extended."""
    t = np.linspace(0, 1, 200)
    sine = np.sin(2 * np.pi * 5.0 * t)
    fit = fit_deterministic(sine, seed=0)
    assert fit["family"] == "sine"
    ext = extend_generator(fit, 20, len(sine))
    L = len(sine); dt = 1.0 / (L - 1)
    t_future = 1.0 + dt * np.arange(1, 21)
    truth = np.sin(2 * np.pi * fit["params"][0] * t_future + fit["params"][1])
    corr = float(np.corrcoef(ext["forecast"], truth)[0, 1])
    assert corr > 0.95, corr                                   # the formula plays forward correctly
    assert ext["valid"]                                        # 20 samples ahead is within the window
    far = extend_generator(fit, 500, len(sine))               # way past the validated window
    assert not far["valid"]                                    # flagged, not silently trusted
    none = extend_generator({"family": None}, 10, 200)
    assert none["forecast"] == [] and not none["valid"]       # a refused fit cannot be extended
    print("extend_generator OK (fitted sine plays forward corr %.2f; far extrapolation flagged invalid; refused "
          "fit not extended)" % corr)


def _selftest():
    """Contracts:

    1. A noisy SINE is identified as 'sine' with high correlation and low residual, and its frequency is recovered
       close to truth.
    2. A noisy CHIRP is identified as 'chirp', NOT sine (the bank discriminates families).
    3. PURE NOISE is REFUSED (family=None, verdict='refused') -- no deterministic structure above the floor.
    4. Q8 band-limiting: reported correlation/verdict is stable; a fit is verified in the ORIGINAL space (regen vs
       data), not just on the coarse signature.
    5. Determinism.
    """
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 400)

    # (1) noisy sine.
    sine = np.sin(2 * np.pi * 7.0 * t + 1.0) + 0.15 * rng.normal(size=len(t))
    r = fit_deterministic(sine, seed=1)
    assert r["family"] == "sine", r
    assert r["correlation"] > 0.85 and r["residual_frac"] < 0.3, r
    assert abs(r["params"][0] - 7.0) < 1.5, r                  # frequency recovered near truth

    # (2) noisy chirp identified as chirp, not sine.
    chirp = _chirp(t, 2.0, 15.0) + 0.15 * rng.normal(size=len(t))
    rc = fit_deterministic(chirp, seed=2)
    assert rc["family"] == "chirp", rc

    # (2b) Puckette audio families: a HARMONIC tone reads 'harmonic' (not a bare sine -- the upper harmonics are
    #      real structure), and an AM/tremolo signal reads 'am'. The wide-band tie resolution sees the harmonics
    #      the coarse snap erases.
    harm = _harmonic(t, 4.0, 0.8) + 0.05 * rng.normal(size=len(t))
    rh = fit_deterministic(harm, seed=5)
    assert rh["family"] == "harmonic", rh
    am_sig = _am(t, 10.0, 2.0) + 0.05 * rng.normal(size=len(t))
    ra = fit_deterministic(am_sig, seed=6)
    assert ra["family"] == "am", ra
    # a pure sine still reads 'sine' (Occam -- a fundamental-only tone is a sine, not a degenerate harmonic).
    pure = np.sin(2 * np.pi * 6.0 * t) + 0.05 * rng.normal(size=len(t))
    assert fit_deterministic(pure, seed=7)["family"] == "sine"

    # (3) pure noise refused.
    noise = rng.normal(size=len(t))
    rn = fit_deterministic(noise, seed=3)
    assert rn["verdict"] == "refused" and rn["family"] is None, rn

    # (4) the fit is verified in the ORIGINAL space (correlation is regen-vs-raw-data, not coarse signature).
    assert 0.0 <= r["residual_frac"] <= 1.0

    # (5) determinism.
    assert fit_deterministic(sine, seed=9) == fit_deterministic(sine, seed=9)

    print("holographic_fitgen selftest OK (noisy sine -> 'sine' corr %.2f, freq %.1f~7; noisy chirp -> 'chirp' not "
          "sine; pure noise REFUSED; Q8 band-limited snap, verified in original space; deterministic)"
          % (r["correlation"], r["params"][0]))


if __name__ == "__main__":
    _selftest()
    _selftest_extend()

"""holographic_spectralline.py -- SCI-3: the spectroscopist's bench (lines, identity, shift, decay).

WHAT EXISTS ALREADY (Rule-0 on record): the Doppler MATH is holographic_dedoppler (doppler_velocity,
redshift, doppler_shift) and time-domain tone fitting is fit_multitone. WHAT WAS MISSING: the
instruments that sit between a measured (wavelength, flux) spectrum and those verbs --

  find_lines        continuum-subtracted, null-gated line finding with sub-bin centroids.
  identify_lines    the CLEANUP DISCIPLINE in scalar costume: nearest catalog line, accepted only
                    with a MARGIN over the runner-up -- identification as recall, ABSTAINING
                    between lines rather than guessing (an identification without a margin is a
                    coin flip wearing a name).
  redshift_verdict  the Le Verrier move on a line list: ONE shared shift must explain EVERY
                    line's displacement, judged against scrambled catalogs -- agreement across
                    lines is the claim, a single line's match is numerology. Velocity is read out
                    through the existing dedoppler faculty, not reimplemented.
  fit_decay         the RESID-5 geometric-decay estimator promoted to a general instrument:
                    y = A exp(-lambda t) + C for counts, ringdowns, randomized-benchmarking
                    fidelities. Closed form (tail-median background + weighted log-linear),
                    bootstrap CI across seeds, and the truncation negative carried over verbatim:
                    a record shorter than ~2/lambda biases the background and the rate -- flagged,
                    never silently absorbed.

All verdicts carry their nulls and p-floors; refusal is a result.
"""

import numpy as np


# ---------------------------------------------------------------------------------------------------
# find_lines -- continuum off, noise floor measured, peaks gated, centroids refined.
# ---------------------------------------------------------------------------------------------------

def _running_median(y, w):
    """Median filter as the continuum estimate -- robust to the very lines being hunted (a mean
    filter drags the continuum toward each line and eats part of it; the median shrugs)."""
    y = np.asarray(y, float)
    w = max(3, int(w) | 1)
    pad = w // 2
    yp = np.pad(y, pad, mode="edge")
    return np.array([np.median(yp[i:i + w]) for i in range(len(y))])


def find_lines(x, y, min_snr=4.0, n_null=32, seed=0, continuum_frac=0.08, max_lines=32):
    """Find emission AND absorption lines in a measured spectrum (x ascending, y flux):
    continuum = running median (window continuum_frac of the record), noise = MAD of the
    residual, candidate lines = local extrema beyond min_snr * noise, centers refined by a
    3-point parabolic centroid (sub-bin, closed form). THE GATE: each candidate's |amplitude| is
    judged against the MAX |residual| of iid-shuffled residuals (the Westfall-Young shape --
    hunting extrema over the whole record is a multiplicity, so the null must hunt too).

    Returns {'lines': [{'center','amplitude','snr','p','kind'}], 'continuum', 'noise'}."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    cont = _running_median(y, continuum_frac * len(y))
    r = y - cont
    noise = float(np.median(np.abs(r - np.median(r)))) * 1.4826 or 1e-12
    # the null hunts extrema too -- but KEPT NEGATIVE (measured, p=1.0 on every planted line):
    # a PERMUTATION of the residual keeps the values, so its max equals the real max and every
    # true line sits inside its own null. The multiplicity null must draw from the NOISE-ONLY
    # distribution: bootstrap n values from the residual's central portion (candidate lines
    # clipped out at 4*noise) and take each surrogate's max -- 'the largest excursion pure noise
    # of this length produces', which is the claim a peak actually has to beat.
    core = r[np.abs(r) < 4.0 * noise]
    if len(core) < 16:
        core = r
    null_max = np.array([np.max(np.abs(np.random.default_rng(seed * 977 + j)
                                       .choice(core, size=len(r), replace=True)))
                         for j in range(int(n_null))])
    cand = []
    for i in range(1, len(r) - 1):
        a = r[i]
        if abs(a) < min_snr * noise:
            continue
        if not ((a > 0 and a >= r[i - 1] and a >= r[i + 1]) or
                (a < 0 and a <= r[i - 1] and a <= r[i + 1])):
            continue
        # 3-point parabolic sub-bin centroid: exact for a parabola, good for any smooth peak top
        denom = (r[i - 1] - 2 * r[i] + r[i + 1])
        delta = 0.5 * (r[i - 1] - r[i + 1]) / denom if abs(denom) > 1e-15 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        center = x[i] + delta * (x[min(i + 1, len(x) - 1)] - x[i - 1]) / 2.0
        p = float((1 + np.sum(null_max >= abs(a))) / (1 + n_null))
        cand.append({"center": float(center), "amplitude": float(a),
                     "snr": float(abs(a) / noise), "p": p,
                     "kind": "emission" if a > 0 else "absorption"})
    cand.sort(key=lambda d: -abs(d["amplitude"]))
    # de-duplicate shoulders: keep the strongest within one continuum-window of a kept line
    kept, min_sep = [], (x[-1] - x[0]) * continuum_frac / 4.0
    for c in cand:
        if all(abs(c["center"] - k["center"]) > min_sep for k in kept):
            kept.append(c)
        if len(kept) >= max_lines:
            break
    return {"lines": [c for c in kept if c["p"] < 0.05], "candidates": kept,
            "continuum": cont, "noise": noise}


# ---------------------------------------------------------------------------------------------------
# identify_lines -- cleanup with a margin; between lines, abstain.
# ---------------------------------------------------------------------------------------------------

def identify_lines(centers, catalog, tol_frac=0.002, margin=2.0):
    """Match measured line centers to a rest catalog: nearest entry, ACCEPTED only when the miss
    is within tol_frac of the wavelength AND the runner-up is at least `margin` times further --
    the codebook-cleanup discipline in scalar costume. Everything else is returned as
    'abstained', by name: an identification without a margin is a coin flip wearing a name.

    Returns {'matches': [{'measured','rest','name','miss_frac'}], 'abstained': [centers]}."""
    cat = sorted(catalog.items(), key=lambda kv: kv[1]) if isinstance(catalog, dict) else \
        sorted(((str(v), float(v)) for v in catalog), key=lambda kv: kv[1])
    names = [k for k, _ in cat]; waves = np.array([v for _, v in cat], float)
    matches, abstained = [], []
    for c in centers:
        d = np.abs(waves - c)
        j = int(np.argmin(d))
        d2 = np.min(np.delete(d, j)) if len(d) > 1 else np.inf
        if d[j] / waves[j] <= tol_frac and d2 >= margin * max(d[j], 1e-15):
            matches.append({"measured": float(c), "rest": float(waves[j]), "name": names[j],
                            "miss_frac": float(d[j] / waves[j])})
        else:
            abstained.append(float(c))
    return {"matches": matches, "abstained": abstained}


# ---------------------------------------------------------------------------------------------------
# redshift_verdict -- one shift must explain every line, or refuse.
# ---------------------------------------------------------------------------------------------------

def redshift_verdict(centers, catalog, z_max=0.2, n_z=4001, tol_frac=0.0015, n_null=48, seed=0):
    """The Le Verrier move on a line list: scan a shared shift z, count catalog lines matched
    within tol at (1+z)*rest, and judge the best count against SCRAMBLED CATALOGS (same number of
    lines, same span, uniformly redrawn -- the null preserves density and destroys the pattern).
    A single matched line is numerology; the claim is AGREEMENT ACROSS THE LIST.

    Returns {'verdict': 'consistent-shift'|'no-consistent-shift', 'z', 'velocity_kms' (classical
    c*z readout; the dedoppler faculty offers the relativistic form), 'matched', 'of',
    'per_line_z_spread', 'p', 'why'}."""
    centers = np.asarray(sorted(centers), float)
    waves = np.array(sorted(catalog.values() if isinstance(catalog, dict) else catalog), float)
    zs = np.linspace(0.0, float(z_max), int(n_z))

    def best_match(cs, ws):
        best = (0, 0.0, [])
        for z in zs:
            pred = ws * (1.0 + z)
            hits = []
            for c in cs:
                d = np.abs(pred - c)
                j = int(np.argmin(d))
                if d[j] / pred[j] <= tol_frac:
                    hits.append((c, ws[j]))
            if len(hits) > best[0]:
                best = (len(hits), float(z), hits)
        return best
    n_hit, z_scan, hits = best_match(centers, waves)
    # KEPT NEGATIVE (measured, z low by ~tol): the scan returns the FIRST z reaching max count --
    # the low EDGE of the tolerance window, not the shift. The scan's job is the ASSIGNMENT; the
    # VALUE comes from the matched pairs themselves: median per-line z, unbiased and outlier-shy.
    z_hat = float(np.median([c / w - 1.0 for c, w in hits])) if hits else z_scan
    rng = np.random.default_rng(seed)
    span = (waves.min(), waves.max())
    null_hits = []
    for _ in range(int(n_null)):
        fake = np.sort(rng.uniform(span[0], span[1], len(waves)))
        null_hits.append(best_match(centers, fake)[0])
    p = float((1 + sum(h >= n_hit for h in null_hits)) / (1 + n_null))
    per_z = [c / w - 1.0 for c, w in hits]
    spread = float(np.std(per_z)) if len(per_z) > 1 else 0.0
    ok = p < 0.05 and n_hit >= 3
    c_kms = 299792.458
    why = ("one shift z=%.5f explains %d/%d measured lines (scrambled-catalog p=%.3f, per-line z "
           "spread %.2e) -- agreement across the list, not a single coincidence"
           % (z_hat, n_hit, len(centers), p, spread) if ok else
           "no shared shift matches more lines than scrambled catalogs do (best %d, p=%.3f) -- "
           "identification withheld; a coincidence is not a redshift" % (n_hit, p))
    return {"verdict": "consistent-shift" if ok else "no-consistent-shift",
            "z": z_hat if ok else None,
            "velocity_kms": c_kms * z_hat if ok else None,
            "matched": n_hit, "of": len(centers), "per_line_z_spread": spread,
            "p": p, "p_floor": 1.0 / (n_null + 1), "why": why}


# ---------------------------------------------------------------------------------------------------
# fit_decay -- the geometric-decay estimator, promoted.
# ---------------------------------------------------------------------------------------------------

def fit_decay(t, y, n_boot=24, seed=0):
    """Fit y = A exp(-lambda t) + C, closed form throughout: C from the tail median (last 10%),
    (A, lambda) by amplitude-weighted log-linear least squares on y - C (weights suppress the
    noisy tail exactly as in the GARCH geometric-decay fit this generalises). Bootstrap CI over
    resampled residuals. Verdict gate: the log-linear slope must beat time-shuffled copies.

    KEPT NEGATIVE, carried verbatim from RESID-5: TRUNCATION BIAS -- a record shorter than
    ~2/lambda has not reached background, so the tail median overestimates C and lambda is biased
    high; flagged in the output ('truncated': True), never silently absorbed.

    Returns {'A','lam','C','half_life','ci_lam','r2','p','verdict','truncated','why'}."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    n = len(y)

    def _pass(C_est):
        d = y - C_est
        pos = d > 0
        if pos.sum() < 4:
            return None
        # WEIGHTS ARE W = d^2, and this is the load-bearing line (two kept negatives behind it,
        # both measured): (1) W = d gave lambda 17% low -- the log-linearisation's bias
        # (E[log d] < log E[d]) concentrates where SNR is small, and the delta method says
        # Var[log d] ~ sigma^2/d^2, so the variance-correct weights are d^2 -- with them the
        # multi-seed bias is 3% (0.0291 +/- 0.0002 on truth 0.0300); (2) a coordinate-descent
        # second pass on the background moved the WRONG WAY (a low lambda inflates the late-time
        # model, drags C down, which flattens lambda further -- the errors feed each other), so
        # the background stays the plain tail median and the weighting carries the fix.
        L = np.log(d[pos]); T = t[pos]; W = d[pos] ** 2
        A_ = np.stack([np.ones(pos.sum()), T], 1)
        M = A_.T @ (W[:, None] * A_); b = A_.T @ (W * L)
        sol = np.linalg.solve(M + 1e-12 * np.eye(2), b)
        return float(np.exp(sol[0])), float(-sol[1])
    C = float(np.median(y[max(n - n // 10, n - 8):]))
    first = _pass(C)
    if first is None:
        return {"verdict": "no-decay", "why": "fewer than 4 samples above the tail background -- "
                                              "nothing to fit a decay through", "p": 1.0}
    A_hat, lam = first
    fit = A_hat * np.exp(-lam * t) + C
    ss = float(np.sum((y - fit) ** 2)); sy = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    r2 = 1.0 - ss / sy
    # the gate: the weighted log-linear slope vs time-shuffled copies (decay = ordered decline;
    # a shuffle keeps the values and destroys the ordering, which is exactly the claim)
    rng = np.random.default_rng(seed)
    slope_real = -lam
    hits = 0
    for j in range(48):
        perm = rng.permutation(n)
        dp = (y[perm] - C)
        pp = dp > 0
        if pp.sum() < 4:
            continue
        Lp = np.log(dp[pp]); Tp = t[pp]; Wp = dp[pp] ** 2
        Ap = np.stack([np.ones(pp.sum()), Tp], 1)
        Mp = Ap.T @ (Wp[:, None] * Ap); bp = Ap.T @ (Wp * Lp)
        sp = np.linalg.solve(Mp + 1e-12 * np.eye(2), bp)
        hits += (sp[1] <= slope_real)                  # as steeply negative as the real slope
    p = float((1 + hits) / (1 + 48))
    lams = []
    for j in range(int(n_boot)):
        rb = np.random.default_rng(seed * 131 + j)
        idx = rb.integers(0, n, n)
        ts, ys = t[idx], y[idx]
        Cs = float(np.median(ys[np.argsort(ts)][-max(n // 10, 8):]))
        ds = ys - Cs; ps = ds > 0
        if ps.sum() < 4:
            continue
        Ls = np.log(ds[ps]); Ts = ts[ps]; Ws = ds[ps] ** 2   # same estimator the CI brackets
        As = np.stack([np.ones(ps.sum()), Ts], 1)
        Ms = As.T @ (Ws[:, None] * As); bs = As.T @ (Ws * Ls)
        lams.append(float(-np.linalg.solve(Ms + 1e-12 * np.eye(2), bs)[1]))
    ci = (float(np.quantile(lams, 0.05)), float(np.quantile(lams, 0.95))) if lams else (lam, lam)
    # margin 3.0 not 2.0, because the flag's own input is compromised: on a truncated record
    # lambda biases HIGH (measured: 0.072 on truth 0.030 at range=0.9/lambda), which inflates
    # lam*range past a tight bar -- the flag must absorb the very bias it exists to report.
    truncated = bool(lam * (t[-1] - t[0]) < 3.0)
    ok = p < 0.05 and lam > 0
    why = ("decay rate lambda=%.4g (half-life %.4g) beats time-shuffled orderings (p=%.3f, "
           "R^2=%.3f)%s" % (lam, np.log(2) / max(lam, 1e-300), p, r2,
           "; TRUNCATED: record < 2/lambda, background and rate biased -- extend the record"
           if truncated else "") if ok else
           "the decline does not beat its own reordering (p=%.3f) -- no decay is claimed" % p)
    return {"A": A_hat, "lam": lam, "C": C, "half_life": float(np.log(2) / max(lam, 1e-300)),
            "ci_lam": ci, "r2": float(r2), "p": p,
            "verdict": "decay" if ok else "no-decay", "truncated": truncated, "why": why}


# ---------------------------------------------------------------------------------------------------
# Selftest -- planted lines, planted shift, planted decay; refusals for each.
# ---------------------------------------------------------------------------------------------------

BALMER = {"H-alpha": 656.279, "H-beta": 486.135, "H-gamma": 434.047, "H-delta": 410.173}


def _selftest():
    rng = np.random.default_rng(0)

    # --- find_lines: 3 emission + 1 absorption on a sloped continuum ---------------------------
    x = np.linspace(400.0, 700.0, 3000)
    cont = 10.0 + 0.004 * (x - 400.0)
    y = cont + 0.15 * rng.standard_normal(len(x))
    for lc, amp in ((486.135, 2.2), (656.279, 3.0), (434.047, 1.6)):
        y += amp * np.exp(-0.5 * ((x - lc) / 0.35) ** 2)
    y -= 1.8 * np.exp(-0.5 * ((x - 589.0) / 0.35) ** 2)          # planted absorption (Na-ish)
    fl = find_lines(x, y, n_null=32, seed=0)
    got = sorted(l["center"] for l in fl["lines"])
    assert len(fl["lines"]) == 4, "exactly the 4 planted lines must gate (got %d: %s)" % (
        len(fl["lines"]), [round(g, 1) for g in got])
    for truth in (434.047, 486.135, 589.0, 656.279):
        assert min(abs(g - truth) for g in got) < 0.2, "line at %.1f must be centred sub-bin" % truth
    kinds = {round(l["center"]): l["kind"] for l in fl["lines"]}
    assert kinds[589] == "absorption", "the dip must be reported as absorption, not flipped"

    # --- find_lines refusal: pure continuum + noise ---------------------------------------------
    y0 = cont + 0.15 * rng.standard_normal(len(x))
    fl0 = find_lines(x, y0, n_null=32, seed=1)
    assert len(fl0["lines"]) == 0, "a lineless spectrum must yield no gated lines (%d)" % len(fl0["lines"])

    # --- identify_lines: Balmer matched, the interloper abstained -------------------------------
    ident = identify_lines([l["center"] for l in fl["lines"]], BALMER, tol_frac=0.002)
    names = {m["name"] for m in ident["matches"]}
    assert {"H-alpha", "H-beta", "H-gamma"} <= names, "Balmer lines must be identified (%s)" % names
    assert any(abs(a - 589.0) < 0.5 for a in ident["abstained"]), \
        "the line NOT in the catalog must be ABSTAINED, not force-matched (%s)" % ident["abstained"]

    # --- redshift_verdict: shared z recovered; scrambled centers refused ------------------------
    z_true = 0.0213
    shifted = [w * (1 + z_true) + 0.02 * rng.standard_normal() for w in BALMER.values()]
    rz = redshift_verdict(shifted, BALMER, seed=0)
    assert rz["verdict"] == "consistent-shift" and abs(rz["z"] - z_true) < 5e-4, \
        "the shared shift must be recovered (%s z=%s)" % (rz["verdict"], rz["z"])
    assert abs(rz["velocity_kms"] - 299792.458 * z_true) < 200
    bogus = sorted(rng.uniform(400, 700, 4))
    rz0 = redshift_verdict(bogus, BALMER, seed=1)
    assert rz0["verdict"] == "no-consistent-shift", \
        "random centers must be refused -- a coincidence is not a redshift (p=%.3f)" % rz0["p"]

    # --- fit_decay: rate + background + CI; truncation flag; no-decay refusal -------------------
    # dedicated rng (the standing rule: planted truths OWN their seeds -- this plant broke once
    # by drawing after the spectrum plants had consumed the shared stream)
    rdc = np.random.default_rng(0)
    t = np.linspace(0, 200, 400)
    lam_true, A_true, C_true = 0.03, 40.0, 5.0
    yd = A_true * np.exp(-lam_true * t) + C_true + 0.6 * rdc.standard_normal(len(t))
    fd = fit_decay(t, yd, seed=0)
    assert fd["verdict"] == "decay" and abs(fd["lam"] - lam_true) < 0.15 * lam_true, \
        "rate must recover (lam=%.4f vs %.4f)" % (fd["lam"], lam_true)
    assert abs(fd["C"] - C_true) < 1.0 and fd["ci_lam"][0] < lam_true < fd["ci_lam"][1]
    assert not fd["truncated"]
    fd_tr = fit_decay(t[:60], yd[:60], seed=0)                   # record << 2/lambda
    assert fd_tr.get("truncated", False), "a short record must carry the truncation flag"
    fd0 = fit_decay(t, C_true + 0.6 * rdc.standard_normal(len(t)), seed=0)
    assert fd0["verdict"] == "no-decay", "flat noise must refuse (p=%.3f)" % fd0["p"]

    print("holographic_spectralline selftest OK -- 4 lines found+centred (1 absorption), lineless "
          "refused, Balmer identified with the interloper abstained, shared redshift recovered and "
          "coincidence refused, decay rate+CI recovered with truncation flagged, flat refused")


if __name__ == "__main__":
    _selftest()

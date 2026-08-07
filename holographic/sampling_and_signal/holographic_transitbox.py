"""holographic_transitbox.py -- SCI-1: the box-matched period hunter (the grammar that finds planets).

WHAT EXISTS ALREADY (Rule-0 on record, reused not rebuilt): holographic_lombscargle provides the
periodogram and phase_fold. WHAT WAS MISSING, measured before building: Lomb-Scargle is a SINUSOID-
matched filter, and a transit is a BOX -- on an injected box (P=173, duty 5%) LS found the period
but at 6.3x LESS peak power than a matched-rms sinusoid. That factor IS the detection floor: near
the floor, the sinusoid template loses planets the box template keeps. Box Least Squares (Kovacs,
Zucker & Mazeh 2002) is the box-matched filter, and it is exactly engine-shaped: deterministic,
closed form per trial period, no learning anywhere.

THE VERDICT DISCIPLINE (inherited from the RESID arc, verbatim):
  * one claim, one matched null: the claim is PHASE COHERENCE AT P, so the null is the block
    shuffle with block << P -- short-range correlation (red noise) survives, cross-period
    alignment dies. An iid null is also reported but the VERDICT uses the block null, because
    red noise makes iid anticonservative (it flags red noise as planets).
  * p-floor arithmetic: with n_null surrogates the minimum p is 1/(n_null+1); a verdict is only
    offered when the gate is arithmetically passable, else the instrument says so.
  * the harmonic family is REPORTED, not hidden: a box at P also scores at P/2 and 2P (the
    grid's own structure masquerading as discoveries); peaks are grouped into families and the
    family, not the bare peak, is the finding.

HONEST SCOPE: evenly-sampled series in v1 (the fold handles gaps, but the null's block shuffle
assumes near-even cadence); no limb darkening, no eccentricity -- a box is a box. This is a
statistics instrument: it returns verdict + power + null, never a discovery claim.
"""

import numpy as np

from holographic.sampling_and_signal.holographic_surrogate import block_shuffle, iid_shuffle


def bls_power(times, values, period, n_bins=64, max_dur_frac=0.15):
    """Box fit at ONE trial period: phase-fold, bin, and find the contiguous run of bins whose
    mean is most below the out-of-box mean -- the signal residue SR of Kovacs et al., closed form.
    Returns (power, depth, dur_frac, phase0). Power is depth^2 * q(1-q) * n -- the chi^2
    improvement of the box over a constant, so it is comparable across periods."""
    t = np.asarray(times, float); y = np.asarray(values, float)
    y = y - y.mean()
    ph = (t / float(period)) % 1.0
    idx = np.clip((ph * n_bins).astype(int), 0, n_bins - 1)
    s = np.bincount(idx, weights=y, minlength=n_bins)
    c = np.bincount(idx, minlength=n_bins).astype(float)
    max_w = max(1, int(np.ceil(max_dur_frac * n_bins)))
    # prefix sums over a doubled circle so a box can wrap phase 0
    s2 = np.concatenate([s, s]); c2 = np.concatenate([c, c])
    S = np.concatenate([[0.0], np.cumsum(s2)]); C = np.concatenate([[0.0], np.cumsum(c2)])
    n_tot = float(len(y)); best = (0.0, 0.0, 1.0 / n_bins, 0.0)
    for w in range(1, max_w + 1):
        sw = S[w:w + n_bins] - S[:n_bins]
        cw = C[w:w + n_bins] - C[:n_bins]
        ok = (cw > 0) & (cw < n_tot)
        if not ok.any():
            continue
        # SR = s^2 / (r (1 - r)) with r the in-box fraction of points, s the in-box sum of a
        # zero-mean series -- the exact chi^2 gain of a two-level (box) model over a constant.
        r = cw / n_tot
        sr = np.where(ok, sw ** 2 / np.maximum(n_tot * r * (1 - r), 1e-12), 0.0)
        i = int(np.argmax(sr))
        if sr[i] > best[0]:
            depth = -sw[i] / max(cw[i], 1.0)          # positive depth = a DIP
            best = (float(sr[i]), float(depth), w / float(n_bins), i / float(n_bins))
    return best


def period_scan(times, values, min_period, max_period, n_periods=800, n_bins=64,
                max_dur_frac=0.15):
    """BLS over a FREQUENCY-uniform trial grid (uniform in 1/P -- uniform in P oversamples long
    periods and starves short ones). Returns (periods, powers). The grid itself is part of the
    instrument: its spacing bounds which periods are distinguishable, and its harmonics are the
    alias family reported by transit_search."""
    f = np.linspace(1.0 / float(max_period), 1.0 / float(min_period), int(n_periods))
    periods = 1.0 / f
    powers = np.array([bls_power(times, values, p, n_bins=n_bins,
                                 max_dur_frac=max_dur_frac)[0] for p in periods])
    return periods, powers


def _harmonic_family(periods, powers, top_frac=0.5):
    """Group the strong peaks into ONE family when they sit at (near-)integer ratios of the
    strongest -- a box at P also scores at P/2, 2P, 3P: the grid's own structure masquerading as
    separate discoveries. Returns (best_period, family list)."""
    i0 = int(np.argmax(powers)); p0 = periods[i0]
    fam = []
    thresh = top_frac * powers[i0]
    for j in np.argsort(-powers)[:12]:
        if powers[j] < thresh:
            break
        ratio = periods[j] / p0
        r = ratio if ratio >= 1 else 1.0 / ratio
        near_int = abs(r - round(r)) < 0.03 and round(r) <= 4
        fam.append({"period": float(periods[j]), "power": float(powers[j]),
                    "harmonic_of_best": bool(near_int)})
    return float(p0), fam


def transit_search(times, values, min_period, max_period, n_periods=800, n_bins=64,
                   n_null=24, seed=0, alpha=0.05):
    """The full instrument: scan, take the best family, and judge it against the PROCEDURE-MATCHED
    null -- the identical scan run on block-shuffled copies (block ~ P_best/4: red noise survives,
    phase coherence dies). Returns {'period', 'depth', 'dur_frac', 'phase0', 'power', 'p_block',
    'p_iid', 'family', 'verdict': 'periodic'|'not-significant', 'why'}. The p-floor is stated;
    a gate that cannot arithmetically pass refuses to pretend it ran."""
    t = np.asarray(times, float); y = np.asarray(values, float)
    p_floor = 1.0 / (int(n_null) + 1)
    if p_floor > alpha:
        return {"verdict": "underpowered", "p_floor": p_floor, "alpha": alpha,
                "why": "with %d surrogates the minimum possible p is %.3f > alpha=%.2f -- the "
                       "gate cannot arithmetically pass; raise n_null (this is arithmetic, not "
                       "evidence)" % (n_null, p_floor, alpha)}
    periods, powers = period_scan(t, y, min_period, max_period, n_periods=n_periods,
                                  n_bins=n_bins)
    p_best, family = _harmonic_family(periods, powers)
    power, depth, dur_frac, phase0 = bls_power(t, y, p_best, n_bins=n_bins)
    block = max(4, int(round(p_best / 4.0)))
    hits_b = hits_i = 0
    for j in range(int(n_null)):
        yb = block_shuffle(y, block, seed=seed * 977 + j)
        _, pw_b = period_scan(t, yb, min_period, max_period, n_periods=n_periods, n_bins=n_bins)
        hits_b += (pw_b.max() >= power)
        yi = iid_shuffle(y, seed=seed * 977 + j)
        _, pw_i = period_scan(t, yi, min_period, max_period, n_periods=n_periods, n_bins=n_bins)
        hits_i += (pw_i.max() >= power)
    p_block = (1 + hits_b) / (1 + n_null)
    p_iid = (1 + hits_i) / (1 + n_null)
    ok = p_block < alpha
    why = ("box power at P=%.4g beats the identical scan on %d phase-destroying block surrogates "
           "(p=%.3f; iid p=%.3f reported, not used -- red noise makes it anticonservative)"
           % (p_best, n_null, p_block, p_iid) if ok else
           "the best box does not stand out from block-shuffled copies of the same series "
           "(p=%.3f) -- no phase-coherent period is claimed at this power" % p_block)
    return {"period": p_best, "depth": depth, "dur_frac": dur_frac, "phase0": phase0,
            "power": power, "p_block": float(p_block), "p_iid": float(p_iid),
            "family": family, "p_floor": p_floor,
            "verdict": "periodic" if ok else "not-significant", "why": why}


def vsa_fold(times, values, period, dim=2048, seed=0):
    """THE FOLD ON THE HOLOGRAPHIC SUBSTRATE: phase becomes a CircularEncoder hypervector (wrap
    exact by construction -- phase 0.999 and 0.001 are the neighbours they physically are, where
    bin edges call them strangers), and the folded profile is a Nadaraya-Watson kernel readout
    from TWO bundles: B = sum_t y_t * enc(phi_t) (value mass) and C = sum_t enc(phi_t)
    (occupancy). profile(phi) = <B, enc(phi)> / <C, enc(phi)> -- one dot product per query, at
    ANY phase (no bins, no edges), and the time stamps never needed to be even or ordered: uneven
    sampling is the native case, not a special one. This is the same moments-not-samples move as
    HDRIFT's mu (a kernel mean embedding), worn by phase.

    Returns (profile_fn, B, C). Robustness note, honest: the kernel fold is a MEAN, so a single
    outlier bleeds into its phase neighbourhood where the binned MEDIAN shrugs it off -- which is
    why fold_subtract keeps 'median' as its default engine and 'vsa' is the opt-in."""
    from holographic.io_and_interop.holographic_encoders import CircularEncoder
    t = np.asarray(times, float); y = np.asarray(values, float)
    enc = CircularEncoder(dim=int(dim), period=1.0, seed=seed)
    ph = (t / float(period)) % 1.0
    ybar = float(y.mean()); yc = y - ybar
    E = np.stack([enc.encode(p) for p in ph])          # (n, dim)
    B = yc @ E                                         # value-mass bundle (CENTERED -- see below)
    C = E.sum(0)                                       # occupancy bundle
    # KEPT NEGATIVE (measured): the CircularEncoder's similarity is Poisson-MINUS-DC -- a SIGNED,
    # mean-zero kernel -- so the raw occupancy dot hovered around zero and flipped sign (den in
    # [-4.6, +6.9] on 2000 uniform phases) and the Nadaraya-Watson ratio became a spike injector
    # (BLS power 1411 CREATED in the 'residual'). A ratio smoother needs a NON-NEGATIVE window:
    # add back a DC offset c0 >= -min(kernel), which leaves the CENTERED numerator exactly
    # unchanged (sum of centered y times a constant is zero) and makes the denominator
    # C.e + c0*n >= n*margin > 0 everywhere. Exact fix, one scalar, no approximation.
    kmin = min(enc.kernel_at(g) for g in np.linspace(0.0, 1.0, 512, endpoint=False))
    c0 = -kmin + 0.05 * abs(enc.kernel_at(0.0))
    n_tot = float(len(y))
    # SECOND KEPT NEGATIVE (measured, concentration swept 0.85-0.98): with near-uniform phases the
    # occupancy dot vanishes, the denominator is ~c0*n at every phase, and the "ratio" degenerates
    # into a convolution with the arbitrary scale 1/c0 -- SHAPE survives (corr 0.835 with the
    # binned template) but AMPLITUDE does not (depth read 0.0026 on truth 0.010, and narrower
    # kernels made it WORSE). The repair is a decomposition of labour the engine uses elsewhere:
    # shape from the bundle, amplitude from ONE closed-form projection -- alpha = <y_c, g>/<g, g>
    # with g the raw profile at the samples' own phases. Exact least-squares rescale, one pass.
    g = np.array([float(B @ enc.encode(p)) / (float(C @ enc.encode(p)) + c0 * n_tot) for p in ph])
    gc = g - g.mean()
    alpha = float(yc @ gc) / (float(gc @ gc) or 1e-12)

    def profile(phi):
        e = enc.encode(float(phi) % 1.0)
        raw = float(B @ e) / (float(C @ e) + c0 * n_tot)
        return ybar + alpha * (raw - g.mean())
    return profile, B, C


def fold_subtract(times, values, period, n_bins=64, engine="median", dim=2048, seed=0):
    """Subtract the phase-folded template at `period` -- the ladder rung's action. Two engines:
    'median' (default): per-bin median -- one outlier in a bin must not become part of the
    'explanation'; 'vsa': the CircularEncoder kernel fold (vsa_fold) -- smooth, bin-free,
    uneven-sampling-native, evaluated at each sample's own phase. Default stays 'median'
    (backward compatible; the mean-based kernel is the opt-in trade).
    Returns (residual, template) where template is per-bin ('median') or per-sample ('vsa')."""
    t = np.asarray(times, float); y = np.asarray(values, float)
    if engine == "vsa":
        prof, _, _ = vsa_fold(t, y, period, dim=dim, seed=seed)
        ph = (t / float(period)) % 1.0
        tmpl = np.array([prof(p) for p in ph])
        return y - tmpl, tmpl
    ph = (t / float(period)) % 1.0
    idx = np.clip((ph * n_bins).astype(int), 0, n_bins - 1)
    tmpl = np.zeros(n_bins)
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            tmpl[b] = np.median(y[sel])
    return y - tmpl[idx], tmpl


def detection_floor(depths=(0.002, 0.004, 0.006, 0.010), period=173.0, dur=9, n=2000,
                    noise=0.002, n_seeds=4, n_null=24, seed=0):
    """The honest deliverable: the DETECTION-LIMIT CURVE, not a highlight reel. For each injected
    depth, the fraction of seeds where transit_search returns 'periodic' with the true period's
    family. Returns {depth: {'recovered_frac', 'snr_per_transit'}}."""
    out = {}
    t = np.arange(n, dtype=float)
    for d in depths:
        rec = 0
        for s in range(int(n_seeds)):
            rng = np.random.default_rng(seed * 131 + s)
            y = noise * rng.standard_normal(n)
            y[(t % period) < dur] -= d
            r = transit_search(t, y, min_period=period * 0.5, max_period=period * 2.0,
                               n_periods=400, n_null=n_null, seed=s)
            good = (r.get("verdict") == "periodic" and
                    min(abs(r["period"] - period), abs(r["period"] - period / 2),
                        abs(r["period"] * 2 - period)) < 0.03 * period)
            rec += bool(good)
        out[float(d)] = {"recovered_frac": rec / float(n_seeds),
                         "snr_per_transit": float(d / noise * np.sqrt(dur))}
    return out


def _selftest():
    rng = np.random.default_rng(0)
    n = 2000; t = np.arange(n, dtype=float)
    P, depth, dur = 173.0, 0.010, 9

    # --- recovery: injected box found, right family, significant --------------------------------
    y = 0.002 * rng.standard_normal(n)
    y[(t % P) < dur] -= depth
    r = transit_search(t, y, 60, 400, n_periods=600, n_null=24, seed=0)
    assert r["verdict"] == "periodic", "a 5-sigma-per-transit box must be significant: %s" % r["why"]
    fam_ps = [f["period"] for f in r["family"]]
    assert min(abs(r["period"] - P), abs(r["period"] - P / 2)) < 0.03 * P or \
           any(abs(p - P) < 0.03 * P for p in fam_ps), \
        "the true period must be the peak or in its harmonic family (got %.1f, fam %s)" % (
            r["period"], [round(p, 1) for p in fam_ps])
    assert abs(r["depth"] - depth) < 0.5 * depth, \
        "recovered depth must be the right magnitude (%.4f vs %.4f)" % (r["depth"], depth)

    # --- refusal: pure noise (white AND red) -----------------------------------------------------
    r0 = transit_search(t, 0.002 * rng.standard_normal(n), 60, 400, n_periods=400,
                        n_null=24, seed=1)
    assert r0["verdict"] == "not-significant", "white noise must be refused (p=%.3f)" % r0["p_block"]
    red = np.cumsum(rng.standard_normal(n)); red = 0.002 * (red - red.mean()) / red.std()
    rr = transit_search(t, red, 60, 400, n_periods=400, n_null=24, seed=2)
    assert rr["verdict"] == "not-significant", \
        "RED noise must be refused by the block null (p_block=%.3f; iid p=%.3f would have been " \
        "fooled: the anticonservative-iid clause, measured)" % (rr["p_block"], rr["p_iid"])

    # --- p-floor arithmetic refusal --------------------------------------------------------------
    ru = transit_search(t, y, 60, 400, n_null=10, alpha=0.05)
    assert ru["verdict"] == "underpowered" and "arithmetic" in ru["why"]

    # --- fold_subtract consumes the periodicity --------------------------------------------------
    resid, tmpl = fold_subtract(t, y, r["period"] if abs(r["period"] - P) < abs(
        r["period"] - P / 2) else P)
    pw_before = bls_power(t, y, P)[0]; pw_after = bls_power(t, resid, P)[0]
    assert pw_after < 0.2 * pw_before, \
        "subtracting the folded template must consume the box power (%.1f -> %.1f)" % (
            pw_before, pw_after)

    # --- the measured gap that justified this module: BLS vs LS at the floor ---------------------
    from holographic.sampling_and_signal.holographic_lombscargle import lomb_scargle_auto
    d_small = 0.004
    y2 = 0.002 * np.random.default_rng(9).standard_normal(n)
    y2[(t % P) < dur] -= d_small
    _, pw = period_scan(t, y2, 60, 400, n_periods=600)
    pk = pw.max() / np.median(pw)
    fr, lp = lomb_scargle_auto(t, y2 - y2.mean(), min_period=60, max_period=400)
    lk = lp.max() / np.median(lp)
    assert pk > lk, \
        "near the floor the box filter must out-contrast the sinusoid filter (BLS %.1fx vs LS " \
        "%.1fx over their own medians) -- the measured gap this module exists for" % (pk, lk)

    # --- the substrate fold: shape from the bundle, amplitude from one projection ----------------
    res_v, _ = fold_subtract(t, y, P, engine="vsa")
    pw_v = bls_power(t, res_v, P)[0]
    assert pw_v < 0.15 * pw_before, \
        "the CircularEncoder fold must consume the box (%.5f -> %.5f)" % (pw_before, pw_v)
    keep = np.sort(np.random.default_rng(4).choice(n, size=int(0.6 * n), replace=False))
    tu = t[keep] + np.random.default_rng(4).uniform(-0.3, 0.3, len(keep))
    yu = y[keep]
    pw_u0 = bls_power(tu, yu - yu.mean(), P)[0]
    res_u, _ = fold_subtract(tu, yu, P, engine="vsa")
    assert bls_power(tu, res_u, P)[0] < 0.15 * pw_u0, \
        "uneven, jittered sampling is the substrate fold's NATIVE case and must still consume"

    print("holographic_transitbox selftest OK -- box recovered with family, white AND red noise "
          "refused, p-floor stated, fold rung consumes, box-vs-sinusoid gap confirmed at the floor, substrate fold consumes even+uneven")


if __name__ == "__main__":
    _selftest()

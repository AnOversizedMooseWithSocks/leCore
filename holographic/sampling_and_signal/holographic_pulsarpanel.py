"""holographic_pulsarpanel.py -- SCI-2: the Hellings-Downs costume for hidden_drivers.

THE ANCESTOR, exact: a gravitational-wave background does not announce itself in any single
pulsar's timing residuals -- each pulsar alone just looks a little red. The signature lives in the
PANEL: pairwise residual correlations that follow one specific curve in pairwise ANGULAR
SEPARATION, the Hellings-Downs curve chi(theta) = 1/2 + (3/2) x ln x - x/4 with x = (1-cos th)/2
-- the quadrupolar fingerprint of a metric perturbation. This is `hidden_drivers` with geometry:
not just "is there a shared factor" but "do its loadings follow the curve the physics predicts".

TWO NULLS FOR TWO CLAIMS (the one-claim-one-null doctrine, and the discrimination that matters):
  cross-correlation exists    independently phase-destroying surrogates per pulsar (AAFT) --
                              spectra kept, cross-pulsar alignment destroyed.
  ... AND IS SHAPED BY THE SKY  the SKY SCRAMBLE: permute pulsar POSITIONS against their
                              residuals. Every pairwise correlation survives untouched; only the
                              angle-pattern dies. A common CLOCK error (monopole: same correlation
                              at every angle) passes the first null and FAILS this one -- which is
                              exactly how it should be told apart from a GW background. The
                              scramble is the modern PTA discipline (cf. NANOGrav's sky-scramble
                              checks) in engine form.

Verdicts: 'hd-consistent' (both nulls beaten AND the fitted amplitude is positive),
'correlated-not-sky-patterned' (cross fires, scramble does not -- the clock-error/monopole
diagnosis), 'independent' (nothing beats its null). Refusals carry p-floors.

PER-PULSAR RED NOISE FIRST (the trap, stated): every pulsar carries its own red noise, and raw
correlations between two red series are spuriously large. Each series is therefore WHITENED with
the closed-form AR rung (the ladder's grammar, no surrogates needed for the fit itself) before the
panel step. HONEST CAVEAT, kept: whitening filters differ per pulsar, so a shared signal is
attenuated and slightly distorted -- amplitude estimates here are LOWER BOUNDS with per-pulsar
filter bias; the CURVE-SHAPE statistic is what the instrument actually certifies.

This is a statistics instrument on synthetic or supplied residuals; it never claims a detection --
it returns verdict + pattern statistic + both nulls, and the scientist owns the interpretation.
"""

import numpy as np

from holographic.sampling_and_signal.holographic_surrogate import amplitude_adjusted_surrogate
from holographic.sampling_and_signal.holographic_residualvoid import _ar_fit


def hd_curve(theta):
    """The Hellings-Downs cross-correlation as a function of angular separation (radians),
    normalised so chi(0+) -> 0.5 (the standard cross-pulsar convention; the auto term's extra
    1/2 delta is not included -- this curve is for DISTINCT pulsars). Closed form, exact."""
    theta = np.asarray(theta, float)
    x = np.clip((1.0 - np.cos(theta)) / 2.0, 1e-12, 1.0)
    return 0.5 + 1.5 * x * np.log(x) - 0.25 * x


def pairwise_angles(positions):
    """Pairwise angular separations from unit sky vectors (k, 3) or (ra, dec) pairs in radians
    (k, 2). Returns the condensed upper-triangle vector, matching np.triu_indices(k, 1) order."""
    P = np.asarray(positions, float)
    if P.shape[1] == 2:                                # (ra, dec) -> unit vectors
        ra, dec = P[:, 0], P[:, 1]
        P = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], 1)
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    iu = np.triu_indices(len(P), 1)
    return np.arccos(np.clip((P @ P.T)[iu], -1.0, 1.0))


def _whiten(y, ar_order=8):
    """Closed-form AR whitening (the ladder's rung, reused): subtract the lag prediction and
    standardise. The first ar_order samples carry through unpredicted."""
    y = np.asarray(y, float)
    _, fitted = _ar_fit(y - y.mean(), order=ar_order)
    r = (y - y.mean()) - fitted
    return r / (r.std() or 1e-12)


def _pattern_stat(C, chi):
    """The angle-pattern statistic: Pearson correlation between the measured pairwise
    correlations and the HD template over pairs, plus the least-squares amplitude. The
    CORRELATION (shape) is the certified quantity; the amplitude is the attenuated estimate."""
    Cc = C - C.mean(); Xc = chi - chi.mean()
    denom = float(np.sqrt((Cc @ Cc) * (Xc @ Xc))) or 1e-12
    shape = float(Cc @ Xc) / denom
    amp = float(Cc @ Xc) / (float(Xc @ Xc) or 1e-12)
    return shape, amp


def hd_search(panel, positions, ar_order=8, n_null=32, seed=0, alpha=0.05):
    """THE PANEL INSTRUMENT: whiten each pulsar's residuals (closed-form AR), correlate every
    pair, and judge the pairwise-correlation vector against the Hellings-Downs template with TWO
    procedure-matched nulls -- AAFT-per-pulsar (does ANY cross-correlation exist?) and the SKY
    SCRAMBLE (is it patterned by geometry, or would any assignment of positions do?).

    Returns {'verdict': 'hd-consistent'|'correlated-not-sky-patterned'|'independent',
             'shape', 'amplitude', 'p_cross', 'p_scramble', 'pair_corr', 'angles', 'why', ...}.
    The p-floor is stated; an impassable gate refuses (the standing arithmetic clause)."""
    P = [np.asarray(s, float) for s in panel]
    k = len(P)
    n = min(len(s) for s in P)
    p_floor = 1.0 / (int(n_null) + 1)
    if p_floor > alpha:
        return {"verdict": "underpowered", "p_floor": p_floor, "alpha": alpha,
                "why": "with %d surrogates the minimum p is %.3f > alpha=%.2f -- arithmetic, "
                       "not evidence; raise n_null" % (n_null, p_floor, alpha)}
    W = np.stack([_whiten(s[:n], ar_order=ar_order) for s in P])       # (k, n) whitened
    iu = np.triu_indices(k, 1)
    C = np.corrcoef(W)[iu]
    theta = pairwise_angles(positions)
    chi = hd_curve(theta)
    shape, amp = _pattern_stat(C, chi)
    # cross-existence statistic: mean squared pairwise correlation (any structure at all)
    cross_stat = float(np.mean(C ** 2))
    rng = np.random.default_rng(seed)
    hits_cross = 0
    for j in range(int(n_null)):
        Ws = np.stack([amplitude_adjusted_surrogate(W[i], seed=seed * 613 + j * 31 + i)
                       for i in range(k)])
        Cs = np.corrcoef(Ws)[iu]
        hits_cross += (float(np.mean(Cs ** 2)) >= cross_stat)
    p_cross = (1 + hits_cross) / (1 + n_null)
    # sky scramble: SAME correlations, permuted positions -- only the pattern is on trial.
    hits_sky = 0
    for j in range(int(n_null)):
        perm = rng.permutation(k)
        theta_s = pairwise_angles(np.asarray(positions, float)[perm])
        shape_s, _ = _pattern_stat(C, hd_curve(theta_s))
        hits_sky += (shape_s >= shape)
    p_scramble = (1 + hits_sky) / (1 + n_null)
    cross_ok = p_cross < alpha
    sky_ok = p_scramble < alpha and amp > 0
    if cross_ok and sky_ok:
        verdict = "hd-consistent"
        why = ("pairwise correlations exist beyond independent surrogates (p=%.3f) AND follow the "
               "Hellings-Downs curve beyond sky scrambles (shape=%.2f, p=%.3f, amplitude=%.3f -- "
               "a lower bound: per-pulsar whitening attenuates shared signal)"
               % (p_cross, shape, p_scramble, amp))
    elif cross_ok:
        verdict = "correlated-not-sky-patterned"
        why = ("the panel co-moves (p=%.3f) but the correlation is NOT organised by pairwise sky "
               "angle (scramble p=%.3f) -- the monopole/clock-error diagnosis, not a GW-background "
               "pattern" % (p_cross, p_scramble))
    else:
        verdict = "independent"
        why = ("pairwise correlations do not exceed independently-surrogated panels (p=%.3f) -- "
               "no shared process is claimed at this power" % p_cross)
    return {"verdict": verdict, "shape": shape, "amplitude": amp,
            "p_cross": float(p_cross), "p_scramble": float(p_scramble),
            "pair_corr": C, "angles": theta, "hd_template": chi,
            "p_floor": p_floor, "why": why}


def make_hd_panel(k=12, n=1500, gw_amp=0.4, red_phi=0.6, red_amp=1.0, seed=0, mode="hd"):
    """Synthetic pulsar panel with planted ground truth, for the verdict experiment: per-pulsar
    AR(1) red noise plus a cross-pulsar process whose spatial covariance is A*chi(theta) ('hd'),
    a MONOPOLE (constant correlation -- the clock-error control, 'mono'), or absent ('none').
    Returns (panel list, positions (k,3) unit vectors). The spatially-correlated process is built
    by a Cholesky factor of the pair covariance applied to white time samples -- white in time on
    purpose, so the per-pulsar AR whitening cannot eat it (the segmenter-eats-boxes lesson,
    pre-applied)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((k, 3))
    pos = v / np.linalg.norm(v, axis=1, keepdims=True)
    if mode == "hd":
        th = np.arccos(np.clip(pos @ pos.T, -1, 1))
        Cov = hd_curve(th)
        np.fill_diagonal(Cov, 1.0)
    elif mode == "mono":
        Cov = np.full((k, k), 0.5); np.fill_diagonal(Cov, 1.0)
    else:
        Cov = np.eye(k)
    # nearest-PSD guard: HD off-diagonals are small; jitter the diagonal rather than trusting luck
    w_eig = np.linalg.eigvalsh(Cov)
    if w_eig.min() < 1e-9:
        Cov = Cov + (1e-9 - w_eig.min()) * np.eye(k)
    L = np.linalg.cholesky(Cov)
    common = L @ rng.standard_normal((k, n))            # spatially HD/mono, temporally white
    panel = []
    for i in range(k):
        e = rng.standard_normal(n); red = np.zeros(n)
        for t in range(1, n):
            red[t] = red_phi * red[t - 1] + e[t]
        red = red / (red.std() or 1e-12)
        panel.append(red_amp * red + (gw_amp * common[i] if mode != "none" else 0.0))
    return panel, pos


def _selftest():
    # --- HD injection: recovered as hd-consistent, curve shape certified ------------------------
    panel, pos = make_hd_panel(k=12, n=1500, gw_amp=0.45, seed=0, mode="hd")
    r = hd_search(panel, pos, n_null=32, seed=0)
    assert r["verdict"] == "hd-consistent", \
        "the planted HD-patterned process must be found (verdict=%s, p_cross=%.3f, p_scr=%.3f)" % (
            r["verdict"], r["p_cross"], r["p_scramble"])
    assert r["shape"] > 0.3, "the angle-pattern correlation must certify the curve (%.2f)" % r["shape"]

    # --- monopole control: co-moving but NOT sky-patterned (the clock-error diagnosis) ----------
    panel_m, pos_m = make_hd_panel(k=12, n=1500, gw_amp=0.45, seed=1, mode="mono")
    rm = hd_search(panel_m, pos_m, n_null=32, seed=1)
    assert rm["verdict"] == "correlated-not-sky-patterned", \
        "a monopole (clock error) must be told apart from HD (verdict=%s, p_scr=%.3f)" % (
            rm["verdict"], rm["p_scramble"])

    # --- no-injection control: independent red noise refused -----------------------------------
    panel_0, pos_0 = make_hd_panel(k=12, n=1500, seed=2, mode="none")
    r0 = hd_search(panel_0, pos_0, n_null=32, seed=2)
    assert r0["verdict"] == "independent", \
        "independent red pulsars must be refused (verdict=%s, p_cross=%.3f)" % (
            r0["verdict"], r0["p_cross"])

    # --- p-floor arithmetic refusal --------------------------------------------------------------
    ru = hd_search(panel, pos, n_null=10, alpha=0.05)
    assert ru["verdict"] == "underpowered" and "arithmetic" in ru["why"]

    # --- the whitening trap, measured and pinned: raw red-vs-red correlations are spurious ------
    raw_C = np.corrcoef(np.stack([np.asarray(s)[:1500] for s in panel_0]))[np.triu_indices(12, 1)]
    wht_C = np.corrcoef(np.stack([_whiten(np.asarray(s)[:1500]) for s in panel_0]))[
        np.triu_indices(12, 1)]
    assert np.mean(raw_C ** 2) > 1.5 * np.mean(wht_C ** 2), \
        "AR whitening must shrink the spurious red-red correlations (%.4f -> %.4f)" % (
            np.mean(raw_C ** 2), np.mean(wht_C ** 2))

    print("holographic_pulsarpanel selftest OK -- HD injection recovered with curve shape, "
          "monopole diagnosed as clock-error-like (not sky-patterned), independent panel refused, "
          "p-floor stated, whitening trap pinned")


if __name__ == "__main__":
    _selftest()

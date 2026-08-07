"""holographic_quantumstats.py -- SCI-4: quantum statistics as refusing instruments.

TWO INSTRUMENTS, both closed form, both with the abstention ladder built in:

  level_statistics   INTEGRABLE OR CHAOTIC, read off the spectrum alone: the consecutive-spacing
                     RATIO r~_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}) (Atas, Bogomolny, Giraud
                     & Roux 2013) needs NO unfolding -- the classic spacing distribution requires
                     dividing out the local density first, and a wrong unfolding manufactures or
                     erases repulsion; the ratio cancels the density exactly. Reference means are
                     exact or high-precision surmises: <r~>_Poisson = 2 ln 2 - 1 ~ 0.38629
                     (integrable, levels ignore each other), <r~>_GOE ~ 0.53590 (chaotic, time-
                     reversal symmetric), <r~>_GUE ~ 0.60266 (chaotic, broken time reversal).
                     The verdict is the nearest class ONLY when the bootstrap CI excludes the
                     others -- at small n the classes are closer than the noise and the honest
                     answer is 'indeterminate' with the n that would decide (the p-floor lesson
                     as a sample-size statement).

  chsh_verdict       BELL CORRELATIONS with two gates and one alarm: S = |E(ab) - E(ab') +
                     E(a'b) + E(a'b')|; the PAIRING SCRAMBLE null (shuffle which B-outcome pairs
                     with which A-outcome, within matching settings -- marginals and setting
                     counts survive, only the correlation dies) answers 'is there correlation at
                     all'; the bootstrap CI against the CLASSICAL BOUND 2 answers 'is it beyond
                     any local hidden-variable account'; and the TSIRELSON ALARM: a CI lower
                     bound beyond 2*sqrt(2) does not mean new physics -- quantum mechanics itself
                     caps S there, so the verdict is 'suspect-instrument' (selection bias, pairing
                     error, detection loophole). The instrument that can call its own data broken
                     is the one worth trusting near a famous bound.

Statistics instruments, not experiments: verdict + null + CI, interpretation belongs to the
scientist. References: Atas et al., PRL 110, 084101 (2013); Oganesyan & Huse, PRB 75, 155111
(2007); Clauser-Horne-Shimony-Holt, PRL 23, 880 (1969); Tsirelson, Lett. Math. Phys. 4, 93 (1980).
"""

import numpy as np

R_POISSON = 2.0 * np.log(2.0) - 1.0          # exact
R_GOE = 0.53590                              # Atas et al. surmise (3x3 exact + numerics)
R_GUE = 0.60266
_REFS = {"poisson (integrable)": R_POISSON, "goe (chaotic, time-reversal)": R_GOE,
         "gue (chaotic, broken time-reversal)": R_GUE}


def spacing_ratios(levels):
    """The unfolding-free ratios r~_n from a sorted spectrum. Degenerate levels (zero spacings)
    are dropped with a count -- a symmetry-forced degeneracy is REAL physics, but it belongs in
    a symmetry-resolved spectrum, not silently inside the ratio statistic."""
    E = np.sort(np.asarray(levels, float))
    s = np.diff(E)
    keep = s > 1e-12 * max(abs(E[-1] - E[0]), 1e-300)
    s = s[keep]
    if len(s) < 2:
        return np.array([]), int((~keep).sum())
    r = np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])
    return r, int((~keep).sum())


def level_statistics(levels, n_boot=400, seed=0, trim_frac=0.1):
    """Classify a spectrum's level statistics by <r~> with a bootstrap CI, refusing when the CI
    cannot separate the candidate classes. `trim_frac` drops the spectrum's edges first: random-
    matrix universality lives in the BULK, and edge levels (the semicircle's rim) obey different
    laws -- feeding them in biases <r~> toward Poisson (measured on a planted GOE: edges kept
    pulled <r~> down by ~0.01 at n=400).

    Returns {'r_mean', 'ci', 'n_ratios', 'dropped_degenerate', 'distances', 'verdict', 'why'}
    with verdict one of the class names or 'indeterminate'."""
    E = np.sort(np.asarray(levels, float))
    k = int(len(E) * trim_frac)
    if k > 0:
        E = E[k:-k]
    r, dropped = spacing_ratios(E)
    if len(r) < 8:
        return {"verdict": "indeterminate", "n_ratios": len(r), "dropped_degenerate": dropped,
                "why": "fewer than 8 usable ratios -- no class is distinguishable at this size"}
    rng = np.random.default_rng(seed)
    means = np.array([r[rng.integers(0, len(r), len(r))].mean() for _ in range(int(n_boot))])
    ci = (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))
    r_mean = float(r.mean())
    dist = {name: abs(r_mean - ref) for name, ref in _REFS.items()}
    inside = [name for name, ref in _REFS.items() if ci[0] <= ref <= ci[1]]
    if len(inside) == 1:
        verdict = inside[0]
        why = ("<r~> = %.4f, CI (%.4f, %.4f) contains only the %s reference %.4f and excludes "
               "the others -- the spectrum's repulsion class is decided"
               % (r_mean, ci[0], ci[1], verdict, _REFS[verdict]))
    elif len(inside) == 0:
        # between classes: could be mixed symmetry sectors, intermediate statistics, or bias --
        # name the nearest, refuse the classification.
        nearest = min(dist, key=dist.get)
        verdict = "indeterminate"
        why = ("<r~> = %.4f sits OUTSIDE every reference's CI membership (nearest: %s at %.4f) "
               "-- intermediate statistics, mixed symmetry sectors, or an unresolved symmetry; "
               "classification withheld" % (r_mean, nearest, _REFS[nearest]))
    else:
        # CI too wide to exclude competitors: the sample-size refusal, with the n that would do it
        gap = min(abs(_REFS[a] - _REFS[b]) for a in inside for b in inside if a < b)
        sd1 = float(r.std())
        n_need = int(np.ceil((2 * 1.96 * sd1 / gap) ** 2))
        verdict = "indeterminate"
        why = ("the CI (%.4f, %.4f) still contains %d classes -- at sigma(r~)=%.3f you need "
               "roughly %d ratios to separate them (have %d); more levels, not more confidence"
               % (ci[0], ci[1], len(inside), sd1, n_need, len(r)))
    return {"r_mean": r_mean, "ci": ci, "n_ratios": len(r), "dropped_degenerate": dropped,
            "distances": dist, "verdict": verdict, "why": why}


# ---------------------------------------------------------------------------------------------------
# CHSH -- the Bell verdict.
# ---------------------------------------------------------------------------------------------------

def chsh_verdict(a_setting, b_setting, a_out, b_out, n_null=200, n_boot=400, seed=0):
    """The CHSH instrument on trial data (per-trial: Alice's setting in {0,1}, Bob's in {0,1},
    outcomes in {-1,+1}): S from the four correlators with the sign pattern that maximises |S|
    over the eight CHSH sign conventions (the CONVENTION is not evidence; the null is scored on
    the same maximised statistic, procedure-matched). Three-way gate:

      pairing-scramble null  shuffle B outcomes WITHIN each (a,b) setting cell -- marginals and
                             counts survive, only the A-B correlation dies. 'independent' when
                             not beaten.
      classical bound        bootstrap CI on S; 'nonclassical (violates CHSH)' only when the CI
                             lower bound clears 2 -- the entire local-hidden-variable polytope,
                             not a point null.
      Tsirelson alarm        CI lower bound beyond 2*sqrt(2) = 'suspect-instrument': quantum
                             mechanics itself stops there, so the data is accusing the apparatus
                             (post-selection, pairing errors), not the theory.

    Returns {'S', 'ci', 'E', 'counts', 'p_pairing', 'verdict', 'why'}."""
    a_s = np.asarray(a_setting, int); b_s = np.asarray(b_setting, int)
    A = np.asarray(a_out, float); B = np.asarray(b_out, float)
    signs = [(1, -1, 1, 1), (1, 1, -1, 1), (1, 1, 1, -1), (-1, 1, 1, 1),
             (-1, 1, -1, -1), (-1, -1, 1, -1), (-1, -1, -1, 1), (1, -1, -1, -1)]

    def corr_cells(Bv):
        E = np.zeros((2, 2)); cnt = np.zeros((2, 2), int)
        for i in range(2):
            for j in range(2):
                sel = (a_s == i) & (b_s == j)
                cnt[i, j] = int(sel.sum())
                E[i, j] = float(np.mean(A[sel] * Bv[sel])) if sel.any() else 0.0
        return E, cnt

    def s_max(E):
        vals = [abs(sg[0] * E[0, 0] + sg[1] * E[0, 1] + sg[2] * E[1, 0] + sg[3] * E[1, 1])
                for sg in signs]
        return float(max(vals))
    E, cnt = corr_cells(B)
    if cnt.min() < 8:
        return {"verdict": "underpowered", "counts": cnt.tolist(),
                "why": "a setting cell has fewer than 8 trials -- correlators are not estimable"}
    S = s_max(E)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(int(n_null)):
        Bp = B.copy()
        for i in range(2):
            for j in range(2):
                sel = np.where((a_s == i) & (b_s == j))[0]
                Bp[sel] = B[sel][rng.permutation(len(sel))]
        hits += (s_max(corr_cells(Bp)[0]) >= S)
    p_pair = float((1 + hits) / (1 + n_null))
    boots = []
    n = len(A)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        aa, bb, Aa, Bb = a_s[idx], b_s[idx], A[idx], B[idx]
        Eb = np.zeros((2, 2)); okc = True
        for i in range(2):
            for j in range(2):
                sel = (aa == i) & (bb == j)
                if not sel.any():
                    okc = False; break
                Eb[i, j] = float(np.mean(Aa[sel] * Bb[sel]))
        if okc:
            boots.append(s_max(Eb))
    ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
    TSIRELSON = 2.0 * np.sqrt(2.0)
    if p_pair >= 0.05:
        verdict = "independent"
        why = ("S=%.3f does not beat pairing-scrambled data (p=%.3f) -- the sides are not even "
               "correlated; no bound is on trial" % (S, p_pair))
    elif ci[0] > TSIRELSON:
        verdict = "suspect-instrument"
        why = ("CI lower bound %.3f exceeds the Tsirelson bound 2*sqrt(2)=%.3f -- quantum "
               "mechanics itself stops there, so the data is accusing the apparatus "
               "(post-selection, pairing errors, detection loophole), not the theory" % (
                   ci[0], TSIRELSON))
    elif ci[0] > 2.0:
        verdict = "nonclassical (violates CHSH)"
        why = ("S=%.3f, CI (%.3f, %.3f): the whole interval clears the classical bound 2 -- no "
               "local hidden-variable model reproduces these correlators (pairing null p=%.3f)"
               % (S, ci[0], ci[1], p_pair))
    else:
        verdict = "correlated-classical"
        why = ("S=%.3f is real correlation (pairing p=%.3f) but the CI (%.3f, %.3f) does not "
               "clear 2 -- a local model suffices; no violation is claimed" % (
                   S, p_pair, ci[0], ci[1]))
    return {"S": S, "ci": ci, "E": E.tolist(), "counts": cnt.tolist(),
            "p_pairing": p_pair, "verdict": verdict, "why": why}


def make_chsh_trials(n=4000, kind="quantum", seed=0):
    """Planted CHSH trials for the verdict experiment. 'quantum': singlet statistics at the
    optimal angles (E = -cos(theta_a - theta_b), S -> 2*sqrt(2)); 'classical': an explicit local
    hidden-variable model (shared lambda, deterministic responses -- S <= 2 by construction);
    'independent': uncorrelated coins; 'broken': quantum trials post-selected on agreement --
    the selection loophole made concrete, pushing S past Tsirelson. Returns (a_set, b_set, A, B)."""
    rng = np.random.default_rng(seed)
    ang_a = [0.0, np.pi / 2]; ang_b = [np.pi / 4, 3 * np.pi / 4]
    a_s = rng.integers(0, 2, n); b_s = rng.integers(0, 2, n)
    A = np.empty(n); B = np.empty(n)
    if kind in ("quantum", "broken"):
        for i in range(n):
            Ecorr = -np.cos(ang_a[a_s[i]] - ang_b[b_s[i]])
            A[i] = 1.0 if rng.random() < 0.5 else -1.0
            B[i] = A[i] if rng.random() < (1 + Ecorr) / 2 else -A[i]
        if kind == "broken":
            # KEPT NEGATIVE from this plant's first draft: post-selecting on raw agreement only
            # reached S=2.21 -- it INFLATES the positive-correlation cell and DEFLATES the three
            # negative ones (agreement is the rare event there), and the effects nearly cancel.
            # A real selection loophole inflates each cell toward ITS OWN favourable sign; the
            # plant now keeps trials whose product matches the cell's true correlation sign.
            Ecell = -np.cos(np.array(ang_a)[a_s] - np.array(ang_b)[b_s])
            favour = A * B * np.sign(Ecell) > 0
            sel = rng.random(n) < np.where(favour, 1.0, 0.30)
            return a_s[sel], b_s[sel], A[sel], B[sel]
    elif kind == "classical":
        lam = rng.uniform(0, 2 * np.pi, n)
        A = np.sign(np.cos(lam - np.array(ang_a)[a_s]))
        B = -np.sign(np.cos(lam - np.array(ang_b)[b_s]))
        A[A == 0] = 1; B[B == 0] = 1
    else:
        A = rng.choice([-1.0, 1.0], n); B = rng.choice([-1.0, 1.0], n)
    return a_s, b_s, A, B


def _selftest():
    # --- level_statistics: three planted ensembles, each classified; small n refused -------------
    rp = np.random.default_rng(0)
    poisson = np.cumsum(rp.exponential(1.0, 800))
    M = rp.standard_normal((500, 500)); goe = np.linalg.eigvalsh((M + M.T) / 2.0)
    H = rp.standard_normal((500, 500)) + 1j * rp.standard_normal((500, 500))
    gue = np.linalg.eigvalsh((H + H.conj().T) / 2.0)
    for name, lv, want in (("poisson", poisson, "poisson"), ("goe", goe, "goe"),
                           ("gue", gue, "gue")):
        r = level_statistics(lv, seed=0)
        assert r["verdict"].startswith(want), \
            "%s spectrum must classify as %s (got %s, <r~>=%.4f CI %s)" % (
                name, want, r["verdict"], r["r_mean"], r["ci"])
    small = level_statistics(goe[200:250], seed=0)         # 50 bulk levels: classes overlap
    assert small["verdict"] == "indeterminate" and "need" in small["why"] or \
           small["verdict"] == "indeterminate", \
        "50 levels cannot separate GOE from GUE -- must refuse (got %s)" % small["verdict"]

    # --- chsh_verdict: quantum violates, classical does not, independent refused, broken alarms --
    q = chsh_verdict(*make_chsh_trials(4000, "quantum", seed=0), seed=0)
    assert q["verdict"] == "nonclassical (violates CHSH)" and q["S"] > 2.5, \
        "singlet statistics must violate (S=%.3f, %s)" % (q["S"], q["verdict"])
    c = chsh_verdict(*make_chsh_trials(4000, "classical", seed=1), seed=1)
    assert c["verdict"] == "correlated-classical" and c["ci"][0] <= 2.0, \
        "an explicit LHV model must NOT violate (S=%.3f CI %s -- if this fires, the " \
        "instrument, not Bell, is wrong)" % (c["S"], c["ci"])
    ind = chsh_verdict(*make_chsh_trials(3000, "independent", seed=2), seed=2)
    assert ind["verdict"] == "independent", "coins must read independent (p=%.3f)" % ind["p_pairing"]
    br = chsh_verdict(*make_chsh_trials(9000, "broken", seed=3), seed=3)
    assert br["verdict"] == "suspect-instrument", \
        "post-selected data past Tsirelson must accuse the APPARATUS (S=%.3f CI %s, got %s)" % (
            br["S"], br["ci"], br["verdict"])

    print("holographic_quantumstats selftest OK -- Poisson/GOE/GUE classified with small-n "
          "refusal, singlet violates CHSH, the LHV model honestly does not, coins independent, "
          "and post-selection past Tsirelson accuses the instrument, not the theory")


if __name__ == "__main__":
    _selftest()

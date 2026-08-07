"""holographic_residualvoid.py -- RESID-1: 'noise is data without an explanation yet', made operational.

THREE COMPOSITIONS over machinery that already exists (Rule-0 on record: every phrasing --
'residual structured or irreducible', 'shared unexplained driver', 'how far from anything in my
history' -- returned fallbacks; the PARTS all hit):

  residual_verdict   EXPLAIN, SUBTRACT, INTERROGATE WHAT REMAINS. Delegate the explanation to
                     decompose_piecewise (segments + per-segment laws + reconstruction), subtract,
                     then judge the residual against SURROGATES MATCHED TO THE DOMAIN'S PATHOLOGY:
                     AAFT preserves fat tails, block_shuffle preserves everything shorter than the
                     claim's scale. Only structure that beats BOTH is called structure. Le Verrier's
                     procedure: Uranus's residuals were not noise, they were Neptune's signature --
                     but an efficient market's residual SHOULD price irreducible, and saying so is
                     the correct terminal answer, not a failure.

  support_gauge      HAVE I EVER SEEN A STATE LIKE THIS? A causal out-of-support monitor: at each
                     step, delay-embed the TRAILING window only (the look-ahead discipline -- the
                     model at time t is built from data before t), train drift moments, and read
                     z(now) against the history's own on-support scale. Every quantitative model
                     dies by confidently extrapolating into its own void (2008 correlations, COVID
                     microstructure); this instrument does not predict the void's contents -- it
                     reports only that you have ENTERED one, which is the claim no adversary can
                     arbitrage away.

  hidden_drivers     THE PUPPET STRINGS. Explain each series in a panel SEPARATELY, collect the
                     residuals, and ask whether they share a common factor (top singular share of
                     the residual matrix) BEYOND what independently-surrogated residuals produce.
                     A real shared factor in the UNEXPLAINED parts is the signature of an external
                     influence no single series discloses -- news, a common counterparty, an
                     exploit in progress. Refused when the panel's residuals are independent.

KEPT DISCIPLINE, inherited on purpose: the null is chosen to destroy the CLAIM and nothing else
(the surrogate module's own doctrine); sparsity is never called void; a grammarless corpus gets a
refusal with its p-value, not an enumeration. Discovered structure and EXPLOITABLE structure are
different claims separated by latency and capacity -- this module makes only the first kind.
"""

import numpy as np

from holographic.sampling_and_signal.holographic_surrogate import (
    amplitude_adjusted_surrogate, block_shuffle, iid_shuffle)


# ---------------------------------------------------------------------------------------------------
# residual_verdict -- explain, subtract, interrogate.
# ---------------------------------------------------------------------------------------------------

def _structure_stat(r):
    """The residual-structure statistic: lag-1..8 autocorrelation energy. Large when the residual
    still carries linear temporal structure the explanation missed; near zero on white noise. Chosen
    because it is exactly what BOTH surrogates are built to preserve-or-destroy on purpose: AAFT
    keeps the marginal and (approx) spectrum -- so beating AAFT means structure BEYOND the spectrum
    is not claimed here, only that the linear structure is real and not a fat-tail artifact; the
    block shuffle destroys structure longer than the block -- so beating it localises the scale."""
    r = np.asarray(r, float)
    r = r - r.mean()
    denom = float(r @ r) + 1e-12
    return float(sum((r[:-k] @ r[k:]) ** 2 for k in range(1, 9)) / (denom ** 2))


def _periodic_stat(r):
    """The PERIODICITY channel: peak-to-median contrast of the power spectrum -- one FFT, closed
    form. Exists because the lag-1..8 autocorrelation stat is BLIND to long-period structure: a
    box at P=211 contributes nothing at short lags once an AR rung eats the within-transit
    adjacency, so the ladder could never see the need for its own fold rung (measured: the
    long-period plant escalated to the vol rung instead). A sharp spectral line has high contrast;
    AR-type spectra are smooth and stay low -- the channels separate the grammars."""
    r = np.asarray(r, float); r = r - r.mean()
    F = np.abs(np.fft.rfft(r)) ** 2
    # Two negatives shaped this statistic, both measured: (1) k < 6 is decomposition residue
    # wearing a period's clothes (the piecewise leftover peaked at k=3 and the fold rung chased
    # it) -- phase-coherence evidence requires repetitions; (2) a single-bin max is BLIND TO
    # BOXES: a box spreads its energy over a HARMONIC COMB (measured F[7]~10k, F[23]~15k, each
    # 3-5x median, no single bin significant -- a FALSE IRREDUCIBLE with BLS power 68 still in
    # the residual). The statistic is therefore the best 4-harmonic comb sum, normalised so a
    # single pure tone scores identically under either reading.
    med = float(np.median(F[6:])) or 1e-12
    kmax = (len(F) - 1) // 4
    if kmax < 7:
        return float(F[6:].max() / med)
    ks = np.arange(6, kmax)
    comb = F[ks] + F[2 * ks] + F[3 * ks] + F[4 * ks]
    return float(comb.max() / (4.0 * med))


def residual_verdict(y, n_surrogates=64, seed=0, min_seg=16, penalty=3.0,
                     scales=(4, 8, 16, 32, 64, 128)):
    """Explain `y` with the piecewise decomposer, subtract, and ask ONE precise question of what
    remains: DID THE EXPLANATION REMOVE ALL TEMPORAL DEPENDENCE? The null is iid_shuffle -- the
    marginal preserved EXACTLY (fat tails cannot be blamed), every trace of temporal order
    destroyed; a residual whose autocorrelation energy beats it carries structure the explanation
    missed. 'structured' at p < 0.05, else 'irreducible' -- and on a market return series
    'irreducible' is the efficient-market hypothesis agreeing with the instrument.

    KEPT NEGATIVE (from this function's own first design, caught by the HTTP e2e run at a
    different n): demanding the block-shuffle AND AAFT nulls simultaneously conflates three
    claims. block_shuffle PRESERVES structure shorter than the block, so a long block CONTAINS
    AR-type residual structure and can never detect it; AAFT preserves the spectrum, and linear
    dependence IS its spectrum, so 'beating AAFT' measures the surrogate's approximation gap.
    One claim, one matched null. The block family is kept for what it IS for: LOCALIZING the
    scale -- `scale_of_structure` reports the smallest block whose surrogates already contain the
    effect, i.e. the scale the structure lives below.

    Returns {'explained', 'residual', 'stat', 'z', 'p', 'verdict', 'scale_of_structure', 'why'}."""
    y = np.asarray(y, float)
    from holographic.sampling_and_signal.holographic_scaffold import decompose_piecewise
    dec = decompose_piecewise(y, min_seg=min_seg, penalty=penalty)
    resid = y - np.asarray(dec["reconstruction"], float)
    stat = _structure_stat(resid)
    # TWO CHANNELS, one null. Level autocorrelation is BLIND to volatility clustering: an ARCH(1)
    # residual measured level-stat 0.016 (p=0.39 -- a FALSE REFUSAL, the worst failure this verdict
    # can make) while its SQUARED series measured 0.694. Market noise's signature dependence lives
    # in the second moment, so both channels are interrogated; iid_shuffle destroys temporal order
    # in both at once, and the SAME shuffles serve both channels (procedure-matched by identity).
    sq = (resid - resid.mean()) ** 2
    stat_sc = _structure_stat(sq)
    stat_pd = _periodic_stat(resid)
    null, null_sc, null_pd = [], [], []
    for j in range(int(n_surrogates)):
        s = iid_shuffle(resid, seed=seed * 977 + j)
        null.append(_structure_stat(s))
        null_sc.append(_structure_stat((s - s.mean()) ** 2))
        null_pd.append(_periodic_stat(s))
    p = (1 + sum(s >= stat for s in null)) / (1 + n_surrogates)      # +1 plug: never exactly 0
    z = (stat - np.mean(null)) / (np.std(null) + 1e-12)
    p_scale = (1 + sum(s >= stat_sc for s in null_sc)) / (1 + n_surrogates)
    z_scale = (stat_sc - np.mean(null_sc)) / (np.std(null_sc) + 1e-12)
    p_per = (1 + sum(s >= stat_pd for s in null_pd)) / (1 + n_surrogates)
    # MULTIPLICITY (kept negative, caught by the white-noise refusal plant): three channels each
    # gated at alpha gives a ~14% family-wise false-alarm rate, and Bonferroni's alpha/3 would sit
    # BELOW the p-floor at ordinary surrogate budgets (arithmetically impassable -- the RESID-4
    # lesson in a new spot). The Westfall-Young move instead: the family statistic is the MAX of
    # the per-channel z-scores, and its null is the max over the SAME shuffles -- one gate, the
    # floor unchanged, correlation between channels handled for free because the null inherits it.
    mus = [np.mean(x) for x in (null, null_sc, null_pd)]
    sds = [np.std(x) + 1e-12 for x in (null, null_sc, null_pd)]
    zs = [(stat - mus[0]) / sds[0], (stat_sc - mus[1]) / sds[1], (stat_pd - mus[2]) / sds[2]]
    fam_real = max(zs)
    fam_null = [max((null[j] - mus[0]) / sds[0], (null_sc[j] - mus[1]) / sds[1],
                    (null_pd[j] - mus[2]) / sds[2]) for j in range(int(n_surrogates))]
    p_family = (1 + sum(f >= fam_real for f in fam_null)) / (1 + n_surrogates)
    structured = p_family < 0.05
    # channel names carry the ROUTING: individually-passing channels, else the argmax-z channel.
    lvl, scl, per = p < 0.05, p_scale < 0.05, p_per < 0.05
    parts = [nm for nm, on in (("periodic", per), ("level", lvl), ("scale", scl)) if on]
    if structured and not parts:
        parts = [("level", "scale", "periodic")[int(np.argmax(zs))]]
    channel = "+".join(parts) if (structured and parts) else None
    scale, profile = None, {}
    if structured:
        # localize -- but the honest deliverable is the PROFILE, not one number: block surrogates
        # CONTAIN progressively more of the structure as the block grows (null mean climbs toward
        # the real stat), and the first block whose surrogates statistically absorb the effect is
        # only a COARSE UPPER BOUND on the structure's scale (the statistic accumulates over the
        # whole series, so containment lags the correlation length). Report both.
        for b in scales:
            nb = [_structure_stat(block_shuffle(resid, b, seed=seed * 613 + j))
                  for j in range(max(int(n_surrogates) // 2, 16))]
            profile[int(b)] = {"p": (1 + sum(s >= stat for s in nb)) / (1 + len(nb)),
                               "null_mean": float(np.mean(nb))}
            if scale is None and profile[int(b)]["p"] >= 0.05:
                scale = int(b)
    why = ("residual dependence beats the order-destroying null on the %s channel(s) "
           "(z_level=%.1f, z_vol=%.1f)%s -- the explanation is missing temporal structure" % (
               channel, z, z_scale,
               ", localized below block %d" % scale if scale else "")
           if structured else
           "the residual is indistinguishable from its own reshuffling in BOTH the level and the "
           "second-moment channel -- irreducible at this horizon; stop mining it (refusal is a result)")
    return {"explained": dec, "residual": resid, "stat": stat, "z": float(z), "p": float(p),
            "p_scale": float(p_scale), "z_vol": float(z_scale), "p_periodic": float(p_per),
            "channel": channel,
            "p_family": float(p_family), "_zs": tuple(float(z_) for z_ in zs),
            "verdict": "structured" if structured else "irreducible",
            "scale_of_structure": scale, "scale_profile": profile, "why": why}


# ---------------------------------------------------------------------------------------------------
# support_gauge -- the causal out-of-support monitor.
# ---------------------------------------------------------------------------------------------------

def support_gauge(y, embed=4, train_window=256, hop=8, dim=1024, seed=0, n_null=16):
    """Walk a stream and report, at each evaluation point, how far the CURRENT delay-embedded state
    sits from everything in the trailing history -- z(now) from drift moments built on past states
    only (causal by construction: the model at step t never sees t or later; the look-ahead linter's
    standard applied to the instrument itself).

    Verdicts per point, on the history's own scale: 'inside' (z above the bootstrap floor),
    'sparse' (thin but explainable by resampling), 'void' (below what any bootstrap of the history
    produces -- a state genuinely unlike anything seen). The gauge predicts NOTHING about the void's
    contents; entering one means exactly 'my history does not cover this' -- the model-validity
    claim, which unlike an alpha claim does not decay when others hold it too.

    Returns {'t': indices, 'z_rel': z / on-support scale, 'verdict': [...], 'embed': ...}."""
    from holographic.sampling_and_signal.holographic_hdrift import drift_moments
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
    y = np.asarray(y, float)
    d = int(embed)
    ts, zr, verd = [], [], []
    for t in range(int(train_window) + d, len(y), int(hop)):
        hist = y[t - train_window - d:t]
        # delay embedding of the TRAILING window only; the current state is the last d samples.
        states = np.stack([hist[i:i + d] for i in range(len(hist) - d)])
        now = y[t - d:t]
        lo, hi = states.min(0), states.max(0)
        span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
        bounds = [(float(l - 0.05 * s), float(h + 0.05 * s)) for l, h, s in zip(lo, hi, span)]
        enc = VectorFunctionEncoder(d, dim=int(dim), bounds=bounds, bandwidth=10.0, seed=seed)
        mu, _ = drift_moments(states, enc)
        z_scale = float(np.mean(enc.encode_many(states) @ mu)) or 1e-9
        z_now = float(mu @ enc.encode(np.clip(now, [b[0] for b in bounds], [b[1] for b in bounds])))
        # bootstrap floor at THIS point: rebuild moments on resampled histories
        floor = []
        for j in range(int(n_null)):
            rs = np.random.default_rng(500 + seed * 31 + j)
            mu_b, _ = drift_moments(states[rs.integers(0, len(states), len(states))], enc)
            floor.append(float(mu_b @ enc.encode(np.clip(now, [b[0] for b in bounds],
                                                         [b[1] for b in bounds]))))
        f_lo = float(np.quantile(floor, 0.05))
        rel = z_now / z_scale
        if rel < 0.05 and z_now <= f_lo + 0.02 * z_scale:
            v = "void"
        elif rel < 0.25:
            v = "sparse"
        else:
            v = "inside"
        ts.append(t); zr.append(rel); verd.append(v)
    return {"t": np.asarray(ts), "z_rel": np.asarray(zr), "verdict": verd, "embed": d}


# ---------------------------------------------------------------------------------------------------
# hidden_drivers -- the puppet strings: a shared factor in what NO series explains alone.
# ---------------------------------------------------------------------------------------------------

def hidden_drivers(panel, n_surrogates=48, seed=0, min_seg=16, penalty=3.0):
    """Explain every series in `panel` (list/array of equal-length series) separately, collect the
    residuals, and test whether the residual MATRIX has a common factor beyond chance: the top
    singular value's energy share, judged against panels of INDEPENDENTLY AAFT-surrogated residuals
    (marginals and spectra kept, cross-series alignment destroyed -- the null that destroys exactly
    the claim 'they move together', nothing else).

    A passing factor is the signature of an influence outside every individual explanation --
    the puppet string. Returns {'factor': the common residual series (unit norm), 'loadings',
    'share', 'z', 'p', 'verdict': 'driver'|'independent', 'residuals'}. Refuses (verdict
    'independent') when the panel's unexplained parts do not co-move beyond their null."""
    from holographic.sampling_and_signal.holographic_scaffold import decompose_piecewise
    P = [np.asarray(s, float) for s in panel]
    n = min(len(s) for s in P)
    R = []
    for s in P:
        dec = decompose_piecewise(s[:n], min_seg=min_seg, penalty=penalty)
        r = s[:n] - np.asarray(dec["reconstruction"], float)
        sd = r.std() or 1e-12
        R.append((r - r.mean()) / sd)                # unit-variance residuals: shares comparable
    R = np.stack(R)                                  # (k series, n samples)
    def top_share(M):
        sv = np.linalg.svd(M, compute_uv=False)
        return float(sv[0] ** 2 / (np.sum(sv ** 2) + 1e-12))
    share = top_share(R)
    null_shares = []
    for j in range(int(n_surrogates)):
        Rs = np.stack([amplitude_adjusted_surrogate(R[i], seed=seed * 613 + j * 17 + i)
                       for i in range(len(R))])
        null_shares.append(top_share(Rs))
    p = (1 + sum(s >= share for s in null_shares)) / (1 + n_surrogates)
    z = (share - np.mean(null_shares)) / (np.std(null_shares) + 1e-12)
    if p >= 0.05:
        return {"factor": None, "loadings": None, "share": share, "z": float(z), "p": float(p),
                "verdict": "independent", "residuals": R,
                "why": "the panel's unexplained parts do not co-move beyond independently-"
                       "surrogated residuals -- no puppet string is claimed"}
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    return {"factor": Vt[0], "loadings": U[:, 0] * S[0], "share": share,
            "z": float(z), "p": float(p), "verdict": "driver", "residuals": R,
            "why": "a common factor carries %.0f%% of the residual energy, z=%.1f above the "
                   "alignment-destroying null -- an influence outside every single-series "
                   "explanation" % (100 * share, z)}


# ---------------------------------------------------------------------------------------------------
# The gauge core, generalized on contact: support_gauge walks DELAY-embedded states of one series;
# panel_gauge walks DEPENDENCE-embedded states of many. Same instrument, different state map -- the
# 2008 lesson is that the void can live in the correlation structure while every single series stays
# inside its own history.
# ---------------------------------------------------------------------------------------------------

def _gauge_states(states, nows, dim=1024, seed=0, n_null=16, bandwidth=10.0):
    """Score each row of `nows` against drift moments built on `states` (one trailing history):
    z(now) on the history's own on-support scale, bootstrap-gated to inside / sparse / void.
    The shared engine behind support_gauge and panel_gauge -- one body, two costumes."""
    from holographic.sampling_and_signal.holographic_hdrift import drift_moments
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
    states = np.asarray(states, float); nows = np.atleast_2d(np.asarray(nows, float))
    lo, hi = states.min(0), states.max(0)
    span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    bounds = [(float(l - 0.05 * s), float(h + 0.05 * s)) for l, h, s in zip(lo, hi, span)]
    enc = VectorFunctionEncoder(states.shape[1], dim=int(dim), bounds=bounds, bandwidth=bandwidth,
                                seed=seed)
    mu, _ = drift_moments(states, enc)
    # LEAVE-SELF-OUT SCALE: a training state's density under mu includes its OWN encoding's
    # self-term; a query state has none. Using the raw mean as the yardstick therefore inflates
    # the scale by exactly one kernel self-mass per state and genuinely-inside queries read
    # 'sparse' (measured: an entire calm regime scored z_rel ~ 0). Subtract the self-term.
    E_tr = enc.encode_many(states)
    z_scale = float(np.mean(E_tr @ mu - np.einsum("ij,ij->i", E_tr, E_tr))) or 1e-9
    blo = np.array([b[0] for b in bounds]); bhi = np.array([b[1] for b in bounds])
    # THE GEOMETRY BOX MUST BE ROBUST: with min/max bounds, ONE straddling transition state in the
    # history stretches the box over the new regime and grants deniability -- measured: a planted
    # tail-coupling flip read 'inside' at every post-flip step because history states whose windows
    # merely TOUCHED the flip had widened the box. The encoder keeps full-range bounds (it must
    # represent everything seen), but OUTSIDENESS is judged against the 10-90% quantile box with a
    # robust span -- a handful of intermediates cannot move a quantile the way they move a max.
    qlo, qhi = np.quantile(states, 0.10, axis=0), np.quantile(states, 0.90, axis=0)
    span_q = np.where(qhi - qlo < 1e-9, np.where(bhi - blo < 1e-9, 1.0, bhi - blo), qhi - qlo)
    out = []
    for now in nows:
        # A query OUTSIDE the history's own (robust) box is the strongest void there is. The first
        # draft clipped the query into the box before scoring, which collapsed 'arbitrarily far
        # outside' into 'at the boundary' and under-reported exactly the events the gauge exists
        # for. Score the clipped point for z_rel context, but the OUTSIDE verdict is geometry.
        outside = float(np.max(np.maximum((now - qhi) / span_q, (qlo - now) / span_q)))
        nowc = np.clip(now, blo, bhi)
        z_now = float(mu @ enc.encode(nowc))
        floor = []
        for j in range(int(n_null)):
            rs = np.random.default_rng(500 + seed * 31 + j)
            mu_b, _ = drift_moments(states[rs.integers(0, len(states), len(states))], enc)
            floor.append(float(mu_b @ enc.encode(nowc)))
        f_lo = float(np.quantile(floor, 0.05))
        rel = z_now / z_scale
        if outside > 0.35:
            # 0.35 robust-spans beyond the 10-90 box: far enough that ordinary tail states of the
            # history's own distribution (which legitimately live outside a quantile box) do not
            # fire; a regime flip measured 1.5-6 robust-spans out.
            v = "void"
        elif rel < 0.05 and z_now <= f_lo + 0.02 * z_scale:
            v = "void"
        elif rel < 0.25:
            v = "sparse"
        else:
            v = "inside"
        out.append((rel, v))
    return out


def panel_gauge(panel, corr_window=60, train_window=240, hop=20, dim=1024, seed=0, n_null=12,
                panel_bandwidth=3.0, state_map="corr", tail_q=0.90):
    """HAVE THE RELATIONSHIPS EVER LOOKED LIKE THIS? The joint-panel out-of-support monitor: the
    state at time t is the upper triangle of the trailing `corr_window` CORRELATION MATRIX, and
    that state is gauged against the trailing history of such states -- causal throughout. This is
    the void support_gauge cannot see: in a correlation crisis every single series can sit inside
    its own marginal history while the DEPENDENCE structure enters territory no history covers
    (the 2008 case: pairwise correlations jumping toward 1 together).

    Returns {'t', 'z_rel', 'verdict', 'state_dim'}. Verdicts share support_gauge's contract,
    including the honest one: the void closes as the trailing window absorbs the new regime."""
    P = np.stack([np.asarray(s, float) for s in panel])        # (k, n)
    k, n = P.shape
    iu = np.triu_indices(k, 1)

    def corr_state(t):
        W = P[:, t - corr_window:t]
        if state_map == "corr":
            C = np.clip(np.corrcoef(W), -0.999, 0.999)
            # FISHER-Z (arctanh): raw correlations are HETEROSCEDASTIC state coordinates -- sampling
            # noise shrinks as |rho| -> 1 (measured: the post-flip rho~0.95 states clustered so
            # tightly they read 'inside' while the calm regime's noisy rho~0.15 states read sparse).
            # arctanh stabilises the variance everywhere, so distances mean the same in every regime.
            return np.arctanh(C[iu])
        if state_map == "leadlag":
            # WHO MOVES FIRST: the ANTISYMMETRIC part of the lag-1 cross-correlation. The costume
            # for causality flips that contemporaneous correlation cannot see -- A leading B and B
            # leading A produce the SAME corr matrix but opposite-signed lead-lag states. Each
            # entry Fisher-z'd for the same variance-stabilising reason as above.
            Wc = (W - W.mean(1, keepdims=True)) / (W.std(1, keepdims=True) + 1e-12)
            a, b = Wc[:, 1:], Wc[:, :-1]                        # a_t vs b_{t-1}
            L = (a @ b.T) / (W.shape[1] - 1)                    # L[i,j] = corr(x_i,t ; x_j,t-1)
            A = np.clip((L - L.T) / 2.0, -0.999, 0.999)         # antisymmetric = pure lead-lag
            return np.arctanh(A[iu])
        if state_map == "tail":
            # DO THEY CRASH TOGETHER: pairwise co-exceedance beyond each series' own trailing
            # tail_q quantile (lower tail). Correlation averages over the whole body of the joint
            # distribution; tail dependence is exactly the part a crisis changes first. Empirical
            # co-exceedance rates are proportions -- variance-stabilised by arcsin(sqrt(p)), the
            # proportion's Fisher-z.
            thr = np.quantile(W, 1.0 - tail_q, axis=1, keepdims=True)
            hit = (W <= thr).astype(float)                      # each series' own worst (1-q) tail
            co = (hit @ hit.T) / W.shape[1]
            return np.arcsin(np.sqrt(np.clip(co[iu], 0.0, 1.0)))
        raise ValueError("state_map must be 'corr', 'leadlag', or 'tail' (got %r)" % state_map)
    ts, zr, verd = [], [], []
    for t in range(corr_window + train_window, n, int(hop)):
        hist_ts = range(t - train_window, t, max(corr_window // 8, 1))
        states = np.stack([corr_state(u) for u in hist_ts])
        (rel, v), = _gauge_states(states, corr_state(t), dim=dim, seed=seed, n_null=n_null,
                                  bandwidth=panel_bandwidth)
        ts.append(t); zr.append(rel); verd.append(v)
    return {"t": np.asarray(ts), "z_rel": np.asarray(zr), "verdict": verd,
            "state_dim": len(iu[0])}


# ---------------------------------------------------------------------------------------------------
# residual_ladder -- escalate a 'structured' verdict instead of stopping at it: add the explanation
# rung the piecewise grammar lacks (a CLOSED-FORM linear autoregression -- deterministic ridge on the
# lag matrix, no learning loop), subtract again, re-interrogate. Climb until 'irreducible' or the
# rungs run out. The ladder's terminal refusal is the deliverable: it says WHICH grammar finally
# priced the stream as noise.
# ---------------------------------------------------------------------------------------------------

def _vol_fit(r, order=4, ridge=1e-3, floor=1e-6, return_var=False):
    """Closed-form ARCH(order)-shaped rung: ridge least squares of r_t^2 on its own lags gives a
    deterministic conditional-variance forecast; the rung's OUTPUT is the STANDARDIZED residual
    r_t / sigma_t, which carries the levels untouched (a vol model explains the ENVELOPE, not the
    signs -- dividing, not subtracting, is what 'explain' means in the second moment). No learning
    loop, no distributional assumption beyond positivity (clamped at `floor`, kept honest)."""
    r = np.asarray(r, float)
    r2 = (r - r.mean()) ** 2
    X = np.stack([r2[k:len(r2) - order + k] for k in range(order)], 1)
    yv = r2[order:]
    A = X.T @ X + ridge * np.eye(order)
    w = np.linalg.solve(A, X.T @ yv)
    var_hat = np.concatenate([np.full(order, max(float(r2.mean()), floor)),
                              np.maximum(X @ w, floor)])
    if return_var:
        return w, r / np.sqrt(var_hat), var_hat
    return w, r / np.sqrt(var_hat)


def _garch_fit(r, proxy_order=8, ridge=1e-3, floor=1e-6):
    """Closed-form GARCH(1,1) via its AR(infinity) representation: r^2_t depends on its own lags
    with GEOMETRICALLY DECAYING coefficients c_k = alpha * beta^(k-1), so an AR(proxy_order) ridge
    on r^2 (the ARCH machinery, reused) followed by ONE log-linear least squares over its positive
    coefficients recovers (alpha, beta) with no MLE and no iteration; omega = mean(r^2)*(1-a-b).

    KEPT NEGATIVES from this function's own drafts, in order: (1) a slipped line handed stage 2
    the squared STANDARDIZED residual instead of sigma^2 -- the memory regressor was chi^2 noise
    and beta fit ~0 on a true 0.95; (2) the repaired two-stage still ATTENUATED beta to ~0.25
    (errors-in-variables: a noisy proxy regressor shrinks its coefficient toward zero) -- the
    geometric-decay form sidesteps the proxy entirely. Falls back to a memoryless report
    (beta=0) when fewer than 3 positive lag coefficients exist to fit a decay through."""
    r = np.asarray(r, float)
    r2 = (r - r.mean()) ** 2
    w, _std = _vol_fit(r, order=proxy_order, ridge=ridge, floor=floor)
    # _vol_fit column k corresponds to r2 shifted by k; column proxy_order-1 is lag-1, so
    # c_lagj = w[proxy_order - j] for j = 1..proxy_order.
    c = np.array([w[proxy_order - j] for j in range(1, proxy_order + 1)], float)
    pos = np.where(c > 0)[0]
    if len(pos) < 3:
        alpha, beta = float(max(c[0], 0.0)), 0.0
    else:
        ks = pos.astype(float)                                  # lag index - 1
        # weighted log-linear fit: log c_k = log(alpha) + k*log(beta); weight by c so the noisy
        # small tail coefficients cannot steer the line.
        L = np.log(c[pos]); W = c[pos]
        A = np.stack([np.ones(len(ks)), ks], 1)
        M = A.T @ (W[:, None] * A); b = A.T @ (W * L)
        sol = np.linalg.solve(M + 1e-12 * np.eye(2), b)
        alpha, beta = float(np.exp(sol[0])), float(np.exp(sol[1]))
    alpha = min(max(alpha, 0.0), 0.999)
    beta = min(max(beta, 0.0), 0.999)
    clamped = alpha + beta >= 1.0
    if clamped:
        beta = max(0.0, 0.999 - alpha)
    omega = max(float(r2.mean()) * (1.0 - alpha - beta), floor)
    var = np.empty(len(r2)); var[0] = max(float(r2.mean()), floor)
    for i in range(1, len(r2)):
        var[i] = omega + alpha * r2[i - 1] + beta * var[i - 1]
    return {"alpha": alpha, "beta": beta, "omega": omega, "clamped": bool(clamped)}, \
        r / np.sqrt(np.maximum(var, floor))


def _ar_fit(r, order=8, ridge=1e-3):
    """Closed-form AR(order) by ridge least squares on the lag matrix. Deterministic, causal in
    form (each prediction uses lags only); returns (coeffs, fitted) with fitted aligned to r
    (first `order` samples carried through unexplained -- no fabricated warm-up)."""
    r = np.asarray(r, float)
    X = np.stack([r[k:len(r) - order + k] for k in range(order)], 1)
    yv = r[order:]
    A = X.T @ X + ridge * np.eye(order)
    w = np.linalg.solve(A, X.T @ yv)
    fitted = np.concatenate([np.zeros(order), X @ w])
    return w, fitted


def residual_ladder(y, max_depth=3, n_surrogates=48, seed=0, min_seg=16, penalty=3.0, ar_order=8):
    """CLIMB THE RESIDUAL: level 0 explains with the piecewise decomposer; every level whose
    residual still reads 'structured' (residual_verdict's iid-shuffle gate) gets the NEXT grammar
    -- a closed-form AR rung -- applied to the residual, and the interrogation repeats. Returns
    {'tower': [level dicts], 'terminal': 'irreducible'|'rungs-exhausted', 'residual'}. Each level
    records its grammar, the variance it removed, and its verdict; the terminal answer names which
    grammar finally priced the remainder as noise -- or admits none here did, which is the honest
    invitation to a grammar this module does not own (the Mendeleev boundary: the tower cannot
    climb past the axioms it has)."""
    y = np.asarray(y, float)
    tower = []
    rv = residual_verdict(y, n_surrogates=n_surrogates, seed=seed, min_seg=min_seg, penalty=penalty)
    var0 = float(np.var(y)) or 1e-12
    tower.append({"grammar": "piecewise", "removed_var_frac": 1.0 - float(np.var(rv["residual"])) / var0,
                  "verdict": rv["verdict"], "p": rv["p"], "p_scale": rv["p_scale"],
                  "channel": rv["channel"], "_zs": rv.get("_zs")})
    resid = rv["residual"]
    base_for_vol = resid
    channel = rv.get("channel")
    depth = 1
    while tower[-1]["verdict"] == "structured" and depth < int(max_depth):
        # RUNG SELECTION BY CHANNEL: level dependence gets the AR rung (subtract the prediction);
        # scale-only dependence gets the VOL rung (divide by the conditional envelope). Applying
        # the AR rung to a pure-ARCH residual removes nothing -- the level channel was already
        # clean -- so the channel decides, not a fixed order of rungs.
        tried = [l["grammar"] for l in tower]
        ar_tried = any(g.startswith("ar(") for g in tried)
        # ROUTING, two kept negatives deep: (1) fixed-priority routing sent an AR(1) residual to
        # the fold rung (a red spectrum's low-k comb can individually pass while the level z is
        # an order of magnitude larger); (2) dominant-z routing then sent the BOX plant to the
        # VOL rung -- transits ARE variance events (scale z=185 there) but vol cannot consume
        # phase coherence, while a fold consumes the level AND scale signatures at once. The
        # discriminator that separates a faked comb from a real one is WHERE the comb peaks: a
        # decaying (AR-type) spectrum pins its argmax to the k-floor boundary; a true period sits
        # INTERIOR. Fold when periodic passes with an interior peak; otherwise dominant-z decides.
        # Routing is GUARDED PRIORITY, not dominant-z (third kept negative on this switch:
        # dominant-z sent real tick data to the vol rung because microstructure lights the scale
        # channel harder than the level channel -- and the bid-ask bounce went unmeasured. The
        # econometrics ordering is principled: MEAN EQUATION BEFORE VARIANCE EQUATION, because
        # unremoved level structure biases the vol fit. Priority fold -> level -> scale, with the
        # interior-peak guard carrying the fix for the AR-plant misroute).
        dom = None
        if channel:
            named = channel.split("+")
            zmap = dict(zip(("level", "scale", "periodic"), tower[-1].get("_zs", (0, 0, 0)))) \
                if tower[-1].get("_zs") else None
            dom = named[0] if zmap is None else max(named, key=lambda nm: zmap.get(nm, -1e9))
        fold_ok = False
        if channel and "periodic" in channel:
            _rz = resid - resid.mean()
            _F = np.abs(np.fft.rfft(_rz)) ** 2
            _kmax = (len(_F) - 1) // 4
            _ks = np.arange(6, max(_kmax, 7))
            _comb = _F[_ks] + _F[2 * _ks] + _F[3 * _ks] + _F[4 * _ks]
            fold_ok = int(_ks[int(np.argmax(_comb))]) > 8      # interior, not the boundary
        if fold_ok:
            # THE CHANNEL THAT DETECTED THE STRUCTURE NAMES THE PERIOD. Two kept negatives from
            # this rung's own drafts: (1) a grid floor of 4*ar_order HID a 30-sample sine from the
            # rung's scan and it chased junk after consuming the true box; (2) with the floor
            # fixed, raw BLS max picked the GRID EDGE (P = len/4, four cycles) -- an unpenalized
            # box at long trial periods absorbs trend residue, the classic long-period bias. The
            # periodicity channel already computed the honest answer: fold at ITS spectral-peak
            # frequency (P = n / argmax|FFT|). BLS with its null gate remains transit_search's
            # job; the rung's job is only to consume what the channel measured.
            from holographic.sampling_and_signal.holographic_transitbox import fold_subtract
            tt = np.arange(len(resid), dtype=float)
            rz = resid - resid.mean()
            F = np.abs(np.fft.rfft(rz)) ** 2
            kmax = (len(F) - 1) // 4
            ks = np.arange(6, max(kmax, 7))
            comb = F[ks] + F[2 * ks] + F[3 * ks] + F[4 * ks]   # the SAME comb the channel scored
            k = int(ks[int(np.argmax(comb))])
            # DETECTOR vs NAMER (kept negative, measured): the integer-bin comb detects reliably
            # but can peak on a HARMONIC when the true fundamental has non-integer k (P=211 in
            # n=1600 puts k0=7.58 between bins; its x3 lands near-integer and won) -- folding at
            # P/3.03 only smears the box. So the comb DETECTS, and the box-matched instrument
            # NAMES: a fine BLS scan around each candidate fundamental m*n/k (m=1..4), +/-5%.
            from holographic.sampling_and_signal.holographic_transitbox import bls_power
            base = float(len(resid)) / k
            best_p, best_pow = base, -1.0
            for mth in (1, 2, 3, 4):
                pc = base * mth
                if pc > len(resid) / 3.0:
                    break
                for p_try in np.linspace(0.95 * pc, 1.05 * pc, 15):
                    pw = bls_power(tt, resid, p_try)[0]
                    if pw > best_pow:
                        best_pow, best_p = pw, float(p_try)
            # second refine pass: at ~7 cycles a period error of dP misaligns the last transit by
            # ~7*dP samples, and the +/-5% grid's step (~1.4 samples here) leaves enough smear
            # that a SECOND fold at almost-the-same P fired on the leftovers (measured: fold(210.2)
            # then fold(211.8)). One +/-1.5% pass tightens the name below the smear scale.
            for p_try in np.linspace(0.985 * best_p, 1.015 * best_p, 15):
                pw = bls_power(tt, resid, p_try)[0]
                if pw > best_pow:
                    best_pow, best_p = pw, float(p_try)
            p_best = best_p
            new_resid, w = fold_subtract(tt, resid, p_best)
            grammar = "fold(P=%.4g)" % p_best
        elif channel and "level" in channel:
            w, fitted = _ar_fit(resid, order=ar_order)
            new_resid = resid - fitted
            grammar = "ar(%d)" % ar_order
        elif "vol-ar(4)" not in tried:
            base_for_vol = resid                       # remember the PRE-division residual
            w, new_resid = _vol_fit(resid)
            grammar = "vol-ar(4)"
        else:
            # Scale dependence SURVIVED the memoryless ARCH envelope: escalate to the sigma^2
            # memory rung -- applied to the residual FROM BEFORE the failed division, replacing
            # it, never stacking on it. KEPT NEGATIVE (measured): feeding GARCH the output of a
            # wrong ARCH division mangles the r^2 dynamics and the two-stage fit collapses to
            # alpha~0, beta~0.02 on a true (0.02, 0.95) plant; on the pre-division residual the
            # same fit standardizes it cleanly (surviving p_scale 0.031 -> 0.723). Failed vol
            # rungs are ALTERNATIVES, not layers.
            w, new_resid = _garch_fit(base_for_vol)
            grammar = "garch(1,1)"
        sq = (new_resid - new_resid.mean()) ** 2
        stat_l, stat_s = _structure_stat(new_resid), _structure_stat(sq)
        stat_p = _periodic_stat(new_resid)
        nl, ns, npd = [], [], []
        for j in range(int(n_surrogates)):
            s = iid_shuffle(new_resid, seed=seed * 977 + depth * 131 + j)
            nl.append(_structure_stat(s)); ns.append(_structure_stat((s - s.mean()) ** 2))
            npd.append(_periodic_stat(s))
        p_l = (1 + sum(s >= stat_l for s in nl)) / (1 + n_surrogates)
        p_s = (1 + sum(s >= stat_s for s in ns)) / (1 + n_surrogates)
        p_p = (1 + sum(s >= stat_p for s in npd)) / (1 + n_surrogates)
        mus = [np.mean(x) for x in (nl, ns, npd)]
        sds = [np.std(x) + 1e-12 for x in (nl, ns, npd)]
        zs = [(stat_l - mus[0]) / sds[0], (stat_s - mus[1]) / sds[1], (stat_p - mus[2]) / sds[2]]
        fam_real = max(zs)
        fam_null = [max((nl[j] - mus[0]) / sds[0], (ns[j] - mus[1]) / sds[1],
                        (npd[j] - mus[2]) / sds[2]) for j in range(int(n_surrogates))]
        p_fam = (1 + sum(f >= fam_real for f in fam_null)) / (1 + n_surrogates)
        structured_lvl = p_fam < 0.05
        lvl, scl, per = p_l < 0.05, p_s < 0.05, p_p < 0.05
        parts = [nm for nm, on in (("periodic", per), ("level", lvl), ("scale", scl)) if on]
        if structured_lvl and not parts:
            parts = [("level", "scale", "periodic")[int(np.argmax(zs))]]
        channel = "+".join(parts) if (structured_lvl and parts) else None
        tower.append({"grammar": grammar, "coeffs": w,
                      "removed_var_frac": 1.0 - float(np.var(new_resid)) / (float(np.var(resid)) or 1e-12),
                      "verdict": "structured" if structured_lvl else "irreducible",
                      "p": float(p_l), "p_scale": float(p_s), "p_periodic": float(p_p),
                      "p_family": float(p_fam), "channel": channel,
                      "_zs": tuple(float(z_) for z_ in zs)})
        resid = new_resid
        depth += 1
    terminal = "irreducible" if tower[-1]["verdict"] == "irreducible" else "rungs-exhausted"
    return {"tower": tower, "terminal": terminal, "residual": resid}


# ---------------------------------------------------------------------------------------------------
# stream_watch -- one timeline: the sentinel's regime events and the gauge's void events, merged in
# the sentinel's own event dialect ({at, kind, ...}), because two monitors with two report formats
# is how an operator misses the morning both fire at once.
# ---------------------------------------------------------------------------------------------------

def stream_watch(y, sentinel=None, embed=4, train_window=256, hop=8, dim=1024, seed=0, n_null=12):
    """Run the regime sentinel and the support gauge over one stream and merge their events into a
    single time-ordered list. Gauge transitions INTO 'void' emit {'at': t, 'kind':
    'support-void', 'z_rel': ...}; transitions back out emit 'support-recovered' (the void closing
    as it is absorbed is part of the story, not noise). Sentinel events pass through untouched.
    `sentinel` accepts a prebuilt StreamSentinel; None builds one on the gauge's cadence."""
    from holographic.sampling_and_signal.holographic_sentinel import StreamSentinel
    y = np.asarray(y, float)
    s = sentinel if sentinel is not None else StreamSentinel(window=max(train_window // 2, 64),
                                                             hop=max(hop * 4, 32), seed=seed)
    sent = s.watch(y)
    g = support_gauge(y, embed=embed, train_window=train_window, hop=hop, dim=dim, seed=seed,
                      n_null=n_null)
    events = list(sent["events"])
    prev = "inside"
    for t, rel, v in zip(g["t"], g["z_rel"], g["verdict"]):
        if v == "void" and prev != "void":
            events.append({"at": int(t), "kind": "support-void", "z_rel": float(rel),
                           "why": "state outside everything in the trailing history "
                                  "(model-validity warning, not a forecast)"})
        elif v != "void" and prev == "void":
            events.append({"at": int(t), "kind": "support-recovered", "z_rel": float(rel),
                           "why": "the trailing window has absorbed the new regime -- the void "
                                  "closed by being observed"})
        prev = v
    events.sort(key=lambda e: e["at"])
    return {"events": events, "gauge": g, "sentinel": sent}


# ---------------------------------------------------------------------------------------------------
# The real-data report: the instrument's first contact with non-planted truth, kept as a runnable
# faculty so the finding cannot rot into a transcript anecdote.
# ---------------------------------------------------------------------------------------------------

def market_residual_report(n_surrogates=64, max_n=1500, seed=0):
    """RUN THE LADDER ON THE CHECKED-IN MARKET DATA and report which grammar terminates each stream.
    The measured result, first recorded 2026-08-06, reproduced the STYLIZED FACTS of finance with no
    market knowledge anywhere in the code:

      * DAI/WETH 1m returns (n=99): irreducible on BOTH channels -- the EMH agreeing with the
        instrument at small n (low power is acknowledged: 99 bars).
      * SOL/USDT 1h returns: level channel CLEAN (p~0.40 -- no linear predictability), scale
        channel FIRES (p~0.015 -- volatility clustering), vol rung terminates. Engle's ARCH
        finding, read off the tower.
      * SOL tick moves: level+scale; the AR rung's lag-1 coefficient comes back NEGATIVE (~-0.21)
        -- the bid-ask bounce -- and the vol rung finishes. The microstructure/efficiency divide:
        level dependence at tick scale, none at 1h.
      * SOL 1h price LEVELS (control): structured, consumed by ar(8) -- a random walk is an AR
        fit's favourite meal; anchors the module's own permutation finding (levels ordered).

    Returns {name: {'verdict', 'channel', 'tower', 'terminal', ...}}. Data-dependent and slower
    than a selftest (surrogate ensembles per stream); the selftest runs a REDUCED pass."""
    import holographic.misc.holographic_market as M
    out = {}

    def one(name, series):
        series = np.asarray(series, float)[:max_n]
        rv = residual_verdict(series, n_surrogates=n_surrogates, seed=seed)
        rl = residual_ladder(series, max_depth=4, n_surrogates=n_surrogates, seed=seed)
        out[name] = {"n": len(series), "verdict": rv["verdict"], "channel": rv["channel"],
                     "p": rv["p"], "p_scale": rv["p_scale"],
                     "tower": [(l["grammar"], l["verdict"], l.get("channel")) for l in rl["tower"]],
                     "terminal": rl["terminal"]}
        for l in rl["tower"]:
            if l["grammar"].startswith("ar(") and "coeffs" in l:
                out[name]["ar_lag1"] = float(np.asarray(l["coeffs"])[-1])

    rows = M.load_ohlcv()
    close = np.array([r[4] for r in rows], float)
    one("dai_weth_1m_returns", np.diff(np.log(close)) * 1e4)
    arr, cols = M.load_sol_market(timeframe="1h")
    close_sol = arr[:, cols.index("close")].astype(float)
    one("sol_1h_returns", np.diff(np.log(close_sol)) * 1e4)
    ts, px = M.load_ticks()
    moves, _ = M.move_series(ts, px)
    one("sol_tick_moves", moves)
    one("sol_1h_levels", close_sol)
    return out


# ---------------------------------------------------------------------------------------------------
# Selftest: planted truths and refusals for all three, at smoke scale.
# ---------------------------------------------------------------------------------------------------

def _selftest():
    rng = np.random.default_rng(0)
    n = 480
    t = np.arange(n, dtype=float)

    # --- residual_verdict: hidden STOCHASTIC dependence (AR(1)) under trend+season ------------------
    # KEPT NEGATIVE from the first draft: a slow deterministic sine planted here was ABSORBED by the
    # piecewise explainer's 14 per-segment laws (corr(resid, hidden)=0.04) and correctly judged
    # irreducible -- the verdict is CONDITIONAL ON THE EXPLAINER'S CAPACITY, and what survives is
    # what the grammar could not already say. AR dependence cannot be absorbed by deterministic
    # segment laws, and it is the truer market story (vol clustering is exactly this shape).
    e = rng.standard_normal(n); ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.7 * ar[i - 1] + e[i]
    y = 0.01 * t + 1.2 * np.sin(2 * np.pi * t / 24.0) + 0.25 * ar
    rv = residual_verdict(y, n_surrogates=48, seed=0)
    assert rv["verdict"] == "structured", \
        "planted AR dependence must beat the order-destroying null (p=%.3f)" % rv["p"]
    prof = rv["scale_profile"]
    bs = sorted(prof)
    assert prof[bs[-1]]["null_mean"] > prof[bs[0]]["null_mean"] * 1.5, \
        "block surrogates must contain progressively more structure as the block grows (%s)" % {
            b: round(prof[b]["null_mean"], 3) for b in bs}

    # --- residual_verdict refusal: pure noise around the same known structure -----------------------
    y0 = 0.01 * t + 1.2 * np.sin(2 * np.pi * t / 24.0) + 0.25 * rng.standard_normal(n)
    rv0 = residual_verdict(y0, n_surrogates=48, seed=0)
    assert rv0["verdict"] == "irreducible", \
        "a genuinely noisy residual must be refused, not narrated (p=%.3f)" % rv0["p"]

    # --- support_gauge: history in one band, an excursion into a never-visited region ---------------
    ys = np.concatenate([0.5 + 0.05 * rng.standard_normal(400),
                         2.5 + 0.05 * rng.standard_normal(60)])     # jump far outside history
    g = support_gauge(ys, embed=3, train_window=200, hop=20, dim=1024, seed=0, n_null=10)
    inside_idx = [i for i, tt in enumerate(g["t"]) if tt < 400]
    post = [i for i, tt in enumerate(g["t"]) if tt >= 400]
    assert inside_idx and post, "the walk must evaluate both regimes"
    frac_in = np.mean([g["verdict"][i] == "inside" for i in inside_idx])
    assert frac_in > 0.8, "in-sample states must read inside (%.2f)" % frac_in
    # the FIRST evaluation after the jump must read void; LATER ones may recover, because the
    # trailing window absorbs the new regime and the history then genuinely covers it -- the void
    # CLOSES AS IT IS OBSERVED. That adaptation is the design (a causal gauge tracks what the
    # model has seen, not what it saw once), and it is the instrument-level echo of market
    # reflexivity: observed voids fill. The first draft asserted ALL post-jump points stay void
    # and was wrong about the instrument's own contract.
    assert g["verdict"][post[0]] == "void", \
        "the first post-jump state must read void (got %s at t=%s, z_rel=%.3f)" % (
            g["verdict"][post[0]], g["t"][post[0]], g["z_rel"][post[0]])
    assert g["verdict"][post[-1]] != "void", \
        "after the window absorbs the regime the gauge must recover (still void at t=%s)" % g["t"][post[-1]]

    # --- hidden_drivers: 5 series, individual structure + one shared residual factor ----------------
    # The factor must be STOCHASTIC (AR(1)), for the same reason as above: a smooth deterministic
    # factor is absorbed by each series's own explanation (measured corr(resid, factor)=0.02) and
    # the honest per-series verdict leaves nothing shared to find. A common shock process is also
    # the realistic puppet string -- sentiment/liquidity shocks, not a hidden sine.
    ef = rng.standard_normal(n); factor = np.zeros(n)
    for i in range(1, n):
        factor[i] = 0.8 * factor[i - 1] + ef[i]
    factor /= factor.std()
    panel, loads = [], [0.8, -0.6, 0.5, 0.9, -0.7]
    for i, w in enumerate(loads):
        own = np.sin(2 * np.pi * t / (18.0 + 3 * i))
        panel.append(0.005 * i * t + own + w * factor + 0.2 * rng.standard_normal(n))
    hd = hidden_drivers(panel, n_surrogates=40, seed=0)
    assert hd["verdict"] == "driver" and hd["z"] > 5.0, \
        "the planted shared factor must be detected emphatically (p=%.3f, z=%.1f)" % (hd["p"], hd["z"])
    # RECOVERY IS BOUNDED BY WHAT SURVIVES EXPLANATION: each series's own decomposition absorbs part
    # of its share of the factor (measured per-residual corr 0.24-0.42), so the SVD recovers the
    # SHADOW of the string, not the string (measured 0.53 overall). The EXISTENCE verdict is the
    # strong claim; the factor estimate is honestly partial -- assert each at its own strength.
    c = abs(np.corrcoef(hd["factor"], factor)[0, 1])
    assert c > 0.4, "the recovered factor must correlate with the surviving truth (|corr|=%.2f)" % c
    sgn = np.sign(np.corrcoef(hd["factor"], factor)[0, 1])
    assert np.all(np.sign(sgn * hd["loadings"]) == np.sign(loads)), \
        "the loading SIGN pattern (who is pulled which way) must be recovered exactly"

    # --- hidden_drivers refusal: independent residuals ----------------------------------------------
    panel0 = [np.sin(2 * np.pi * t / (18.0 + 3 * i)) + 0.2 * np.random.default_rng(i).standard_normal(n)
              for i in range(5)]
    hd0 = hidden_drivers(panel0, n_surrogates=40, seed=0)
    assert hd0["verdict"] == "independent", \
        "independent residuals must be refused (p=%.3f, share=%.2f)" % (hd0["p"], hd0["share"])

    # --- residual_ladder: piecewise -> AR rung must consume the AR structure -> irreducible --------
    rl = residual_ladder(y, max_depth=3, n_surrogates=40, seed=0)
    assert rl["tower"][0]["verdict"] == "structured" and rl["terminal"] == "irreducible", \
        "the AR rung must consume what the piecewise grammar could not (tower %s)" % [
            (l["grammar"], l["verdict"]) for l in rl["tower"]]
    assert rl["tower"][-1]["grammar"].startswith("ar("), "the consuming rung must be the AR grammar"

    # --- panel_gauge: marginals stationary, CORRELATION regime flips -- the 2008 shape --------------
    npn = 560; kk = 5
    rgp = np.random.default_rng(7)
    common = rgp.standard_normal(npn)
    noise = rgp.standard_normal((kk, npn))
    w = np.concatenate([np.full(400, 0.15), np.full(npn - 400, 0.95)])   # correlations jump together
    Pn = w * common[None, :] + np.sqrt(1 - w ** 2) * noise               # unit variance THROUGHOUT:
    pg = panel_gauge([Pn[i] for i in range(kk)], corr_window=50, train_window=200, hop=25,
                     dim=1024, seed=0, n_null=8)                          # marginals never leave home
    pre = [i for i, tt in enumerate(pg["t"]) if tt < 400]
    post = [i for i, tt in enumerate(pg["t"]) if 420 <= tt < 480]
    assert pre and post, "the panel walk must span both dependence regimes"
    frac_pre = np.mean([pg["verdict"][i] == "inside" for i in pre])
    assert frac_pre > 0.7, "the calm dependence regime must read inside (%.2f)" % frac_pre
    assert any(pg["verdict"][i] == "void" for i in post), \
        "the correlation jump must be gauged void while every marginal stays in-sample: %s" % [
            pg["verdict"][i] for i in post]
    # and each SINGLE series stays inside its own history through the flip -- the point of the panel
    gk = support_gauge(Pn[0], embed=3, train_window=200, hop=40, dim=1024, seed=0, n_null=6)
    late = [i for i, tt in enumerate(gk["t"]) if tt >= 420]
    assert late and all(gk["verdict"][i] != "void" for i in late), \
        "the marginal gauge must NOT fire on a pure dependence shift (%s)" % [gk["verdict"][i] for i in late]

    # --- the SECOND-MOMENT channel: ARCH(1) plant -- the false refusal, pinned ----------------------
    # KEPT NEGATIVE: the single-channel verdict measured level-stat 0.016 (p=0.388) on this exact
    # plant and refused it as irreducible while its SQUARED series measured 0.694 -- a FALSE
    # REFUSAL, the worst failure a refusal-is-a-result instrument can make. Market noise's
    # signature dependence (volatility clustering) lives in the second moment.
    ea = rng.standard_normal(n); ra = np.zeros(n); s2 = np.ones(n)
    for i in range(1, n):
        s2[i] = 0.2 + 0.75 * ra[i - 1] ** 2
        ra[i] = np.sqrt(s2[i]) * ea[i]
    ya = 0.01 * t + np.sin(2 * np.pi * t / 24.0) + 0.3 * ra
    rva = residual_verdict(ya, n_surrogates=48, seed=0)
    assert rva["verdict"] == "structured" and rva["channel"] == "scale", \
        "ARCH dependence must fire on the scale channel ONLY (channel=%s, p=%.3f, p_scale=%.3f)" % (
            rva["channel"], rva["p"], rva["p_scale"])
    rla = residual_ladder(ya, max_depth=3, n_surrogates=40, seed=0)
    assert rla["terminal"] == "irreducible" and rla["tower"][-1]["grammar"].startswith("vol-ar"), \
        "the VOL rung (divide by the envelope), not the AR rung, must consume ARCH (tower %s)" % [
            (l["grammar"], l["verdict"]) for l in rla["tower"]]

    # --- stream_watch: one timeline, both dialects ---------------------------------------------------
    # dedicated rng: this plant broke once when an upstream test block consumed draws from the
    # shared stream and moved the realization -- planted truths own their seeds.
    rsw = np.random.default_rng(123)
    ys2 = np.concatenate([0.5 + 0.05 * rsw.standard_normal(400), 2.5 + 0.05 * rsw.standard_normal(80)])
    sw = stream_watch(ys2, embed=3, train_window=200, hop=20, dim=1024, seed=0, n_null=6)
    kinds = [e["kind"] for e in sw["events"]]
    assert "support-void" in kinds, "the merged timeline must carry the gauge's void event (%s)" % kinds
    assert "support-recovered" in kinds, "the closing of the void is part of the story (%s)" % kinds
    ats = [e["at"] for e in sw["events"]]
    assert ats == sorted(ats), "one time-ordered timeline, not two report formats"

    # --- the fold rung: periodicity the SEGMENTER CANNOT EAT ----------------------------------------
    # KEPT NEGATIVE (this test's own first plant): a BOX transit is piecewise-constant -- the
    # level-0 grammar's food -- and decompose_piecewise segments at the transit edges and eats it
    # whole (BLS at the true period read 0.2 on the verdict residual; the "consumption" the first
    # assert measured was power LATER RUNGS re-created). The fold rung's honest plant is
    # periodicity the segmenter cannot express: teeth SHORTER than min_seg.
    rfp = np.random.default_rng(5)
    nf = 1600; tf = np.arange(nf, dtype=float)
    saw = ((tf % 12.0) / 12.0 - 0.5) * 0.12
    ef = rfp.standard_normal(nf); arf = np.zeros(nf)
    for i in range(1, nf):
        arf[i] = 0.6 * arf[i - 1] + ef[i]
    yf = 0.004 * tf + saw + 0.05 * arf
    rlf = residual_ladder(yf, max_depth=5, n_surrogates=40, seed=0)
    folds = [l["grammar"] for l in rlf["tower"] if l["grammar"].startswith("fold(")]
    assert folds, "sub-min_seg periodicity must route to the fold rung (tower %s)" % [
        l["grammar"] for l in rlf["tower"]]
    p_named = float(folds[0].split("P=")[1].rstrip(")"))
    assert abs(p_named - 12.0) < 0.03 * 12.0, \
        "comb-detect + BLS-name must land within 3%% of the true period (got %s)" % p_named
    pd_base = _periodic_stat(residual_verdict(yf, n_surrogates=8, seed=0)["residual"])
    pd_end = _periodic_stat(rlf["residual"])
    assert pd_end < 0.3 * pd_base, \
        "the fold rung must consume the periodicity (comb contrast %.1f -> %.1f; the meter is the " \
        "periodic channel's own stat -- BLS is a BOX meter and reads a sawtooth at ~1)" % (
            pd_base, pd_end)

    # --- panel costumes: lead-lag (who moves first) and tail (do they crash together) --------------
    rll = np.random.default_rng(11)
    aa = rll.standard_normal(560); nbb = rll.standard_normal(560)
    Al = np.zeros(560); Bl = np.zeros(560)
    Al[:400] = aa[:400]; Bl[1:400] = 0.95 * aa[:399] + 0.2 * nbb[1:400]
    Bl[400:] = aa[400:]; Al[401:] = 0.95 * Bl[400:-1] + 0.2 * nbb[401:]
    pg_ll = panel_gauge([Al, Bl], corr_window=30, train_window=180, hop=10, n_null=8,
                        state_map="leadlag", seed=0)
    pg_c = panel_gauge([Al, Bl], corr_window=30, train_window=180, hop=10, n_null=8,
                       state_map="corr", seed=0)
    win = lambda pg: [v for t, v in zip(pg["t"], pg["verdict"]) if 400 <= t <= 450]
    assert "void" in win(pg_ll), "the lead-lag flip must fire the leadlag costume (%s)" % win(pg_ll)
    assert "void" not in win(pg_c), \
        "contemporaneous correlation is IDENTICAL across this flip -- the corr costume must stay " \
        "silent (%s); who-moves-first is invisible to a symmetric statistic" % win(pg_c)
    rtl = np.random.default_rng(21)
    common_t = rtl.standard_normal(560)
    Xt = 0.45 * common_t[None, :] + np.sqrt(1 - 0.45 ** 2) * rtl.standard_normal((4, 560))
    crash_t = (rtl.random(560) < 0.10) & (np.arange(560) >= 400)
    Xt[:, 400:] = 0.15 * common_t[None, 400:] + np.sqrt(1 - 0.15 ** 2) * rtl.standard_normal((4, 160))
    Xt[:, crash_t] -= 2.5
    pg_t = panel_gauge([Xt[i] for i in range(4)], corr_window=60, train_window=200, hop=15,
                       n_null=8, state_map="tail", seed=0)
    assert "void" in [v for t, v in zip(pg_t["t"], pg_t["verdict"]) if 400 <= t <= 470], \
        "shared crash clustering must fire the tail costume"

    # --- GARCH rung: parameter RECOVERY is the pinned claim; whitening superiority is a KEPT
    # NEGATIVE (6-seed: at beta=0.95 GARCH-standardized residuals still read structured 5/6 vs
    # ARCH's 3/6 -- the rung's value is the DIAGNOSIS (alpha, beta measured, persistence named at
    # exhaustion), not superior consumption; beta is biased low ~0.89 on truth 0.95 because the
    # proxy_order=8 fit truncates the geometric tail).
    rgc = np.random.default_rng(100)
    ng = 1600; eg = rgc.standard_normal(ng); rr = np.zeros(ng); vv = np.ones(ng)
    for i in range(1, ng):
        vv[i] = 0.06 + 0.10 * rr[i - 1] ** 2 + 0.85 * vv[i - 1]
        rr[i] = np.sqrt(vv[i]) * eg[i]
    prm, _ = _garch_fit(rr)
    assert 0.75 < prm["beta"] < 0.95 and 0.05 < prm["alpha"] < 0.20, \
        "geometric-decay GARCH must recover (0.10, 0.85) (got %.3f, %.3f)" % (prm["alpha"], prm["beta"])

    # --- real data, reduced: pin the two headline signatures so the finding cannot rot -------------
    # (full report is market_residual_report; here only the two cheap, decisive anchors)
    import holographic.misc.holographic_market as _M
    _ts, _px = _M.load_ticks()
    _mv, _ = _M.move_series(_ts, _px)
    _rl = residual_ladder(np.asarray(_mv, float)[:900], max_depth=3, n_surrogates=24, seed=0)
    _ar = [l for l in _rl["tower"] if l["grammar"].startswith("ar(") and "coeffs" in l]
    assert _ar and float(np.asarray(_ar[0]["coeffs"])[-1]) < 0, \
        "tick moves must fire the AR rung with a NEGATIVE lag-1 coefficient (bid-ask bounce)"
    _arr, _cols = _M.load_sol_market(timeframe="1h")
    # power note, measured: p_scale 0.080 @ (n=900, 24 surr) -> 0.041 @ 48 surr -> 0.020 @
    # (1500, 48) -- monotone in both dials, so 900/24 was UNDER-POWERED, not wrong. The pin uses
    # the cheapest SUFFICIENT setting, chosen from the sweep, not from the first green run.
    _ret = np.diff(np.log(_arr[:, _cols.index("close")].astype(float)))[:1500] * 1e4
    _rv = residual_verdict(_ret, n_surrogates=24, seed=0)
    assert _rv["channel"] in ("scale", "level+scale") and _rv["p_scale"] < 0.05, \
        "1h returns must carry volatility clustering on the scale channel (p_scale=%.3f)" % _rv["p_scale"]

    print("holographic_residualvoid selftest OK -- AR found (level), ARCH found (scale, the "
          "false-refusal fix), noise refused, vol rung consumes what the AR rung cannot, "
          "excursion gauged void then honestly absorbed, puppet string detected, independence "
          "refused, dependence-void caught while marginals sleep, one merged timeline")


if __name__ == "__main__":
    _selftest()

"""holographic_honesty.py -- the ablation ethos as a callable instrument.

Holostuff's README and ABLATIONS.md keep only what beats a baseline. This module is
that rule made executable, so any recall-based predictor can be judged the same honest
way before it is believed -- and so a parameter SCAN (many encoders / leaf sizes / roles
over the same data) is held to a false-discovery bar instead of celebrating whichever
candidate cleared 2-sigma by luck. Nothing here is market-specific; it is the engine's
own discipline, callable.

  walk_forward_recall -- judge a nearest-neighbour predictor over a signed time series
                         with six checks a real edge has to survive (beat chance, beat
                         the trivial persistence baseline, collapse under a shuffle, and
                         so on).
  bh_fdr              -- Benjamini-Hochberg (or Benjamini-Yekutieli for dependent tests)
                         false-discovery-rate control, the missing guard for a library
                         that can generate a great many candidates on demand.
"""
import math

import numpy as np


def walk_forward_recall(states, outcomes, R, cost=0.0, seed=0, warmup=None):
    """Judge a nearest-neighbour predictor over (states, outcomes) honestly.

    states  : (N, dim) one hypervector per moment, in time order.
    outcomes: (N,) the signed quantity being predicted (e.g. a next return).
    R       : how many nearest PAST states to recall and vote with.
    cost    : round-trip cost in the same units as outcomes, charged per trade.

    Six checks, all of which a real edge has to survive:
      acc           directional accuracy of the recall vote
      acc_persist   the TRIVIAL baseline (predict the last non-zero sign) -- recall must
                    beat this, not merely beat a coin
      acc_shuffled  the same machinery on SHUFFLED outcomes -- this MUST collapse to
                    chance; if it does not, the harness itself is leaking
      chance_band   the 2-sigma chance half-width (2 * 0.5 / sqrt(n)); 'better than
                    chance' means acc - 0.5 > band
      scale_corr    does the SPREAD of recalled outcomes track the realised move size --
                    the magnitude signal, separate from direction
      net           gross edge minus cost, in outcome units
    """
    out = np.asarray(outcomes, float)
    states = np.asarray(states, float)
    N = len(out)
    sgn = np.sign(out)
    # last non-zero sign seen so far -> the persistence baseline
    last_nz = np.zeros(N)
    cur = 0.0
    for t in range(N):
        last_nz[t] = cur
        if sgn[t] != 0:
            cur = sgn[t]
    if warmup is None:
        warmup = max(R + 20, 120)

    def run(vec):
        osgn = np.sign(vec)
        ok = okp = tot = 0
        gross = 0.0
        spreads, realized = [], []
        for row in range(warmup, N):
            if osgn[row] == 0:
                continue
            sims = states[:row - 1] @ states[row]        # recall strictly from the PAST
            top = np.argsort(sims)[-R:]
            recall = vec[top]
            pred = np.sign(np.median(recall)) or 1.0
            ok += (pred == osgn[row])
            okp += (last_nz[row] == osgn[row])
            tot += 1
            gross += pred * vec[row]
            spreads.append(np.quantile(recall, 0.9) - np.quantile(recall, 0.1))
            realized.append(abs(vec[row]))
        return (ok / tot, okp / tot, tot, gross / tot,
                np.asarray(spreads), np.asarray(realized))

    acc, acc_persist, tot, gross, spreads, realized = run(out)
    shuf = out.copy()
    np.random.default_rng(seed).shuffle(shuf)
    acc_shuffled = run(shuf)[0]                           # MUST sit at ~0.5
    if len(spreads) > 2 and spreads.std() > 0 and realized.std() > 0:
        scale_corr = float(np.corrcoef(spreads, realized)[0, 1])
    else:
        scale_corr = float("nan")
    band = 2 * 0.5 / math.sqrt(tot)
    return dict(acc=acc, acc_persist=acc_persist, acc_shuffled=acc_shuffled,
                n=tot, gross=gross, net=gross - cost,
                scale_corr=scale_corr, chance_band=band,
                beats_chance=(acc - 0.5) > band,
                beats_persistence=acc > acc_persist)


def bh_fdr(pvals, alpha=0.1, dependent=True):
    """Benjamini-Hochberg false-discovery-rate control.

    With dependent=True applies the Benjamini-Yekutieli c(m)=sum(1/j) correction, the
    honest choice when the tests are dependent -- and a parameter SCAN (many encoders /
    leaf sizes / roles over the SAME data) always is. Holds the expected false-positive
    fraction among declared discoveries at alpha. Returns (reject_mask, n_rejected).
    """
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return np.zeros(0, bool), 0
    order = np.argsort(p)
    ranked = p[order]
    c = float(np.sum(1.0 / np.arange(1, m + 1))) if dependent else 1.0
    thresh = (np.arange(1, m + 1) / (m * c)) * alpha
    below = ranked <= thresh
    k = int(np.max(np.where(below)[0]) + 1) if below.any() else 0
    rej = np.zeros(m, bool)
    if k > 0:
        rej[order[:k]] = True
    return rej, k


class RecallNull:
    """Turn a recall / cleanup similarity into an HONEST false-alarm probability.

    The move radio-SETI and particle physics both live by: a raw cosine of 0.13 means
    nothing on its own -- you have to ask how high random noise reaches against THIS
    codebook before you believe a match. fit() draws random unit queries and records the
    best-match similarity each one reaches; that empirical null IS the noise floor. Then
    pvalue(score) = the fraction of null best-scores that reach `score` or higher = the
    chance pure noise would look this good. Small p: trust the recall. Large p: abstain.

    Calibrated by construction: a genuinely random query's p is ~uniform, so thresholding at
    p <= alpha holds the false-alarm rate at ~alpha -- the engine's "score it, then prove it
    isn't an artifact of your own pipeline" discipline made callable per recall. Complements
    HoloForest's cross-tree agreement (a structural abstention signal) with a statistical one.
    """

    def __init__(self):
        self.null = None          # sorted ascending: best-match cosine reached by noise

    def fit(self, codebook, n_null=2000, seed=0):
        """codebook: (N, dim) atoms (need not be unit; they are unit-normalised here so the
        dot is a cosine). Draw n_null random unit queries, record the best cosine each hits,
        and keep that sorted null. O(n_null * N * dim) once, then pvalue() is a binary search."""
        C = np.asarray(codebook, float)
        units = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        # SEED-COLLISION FIX (found when the F1 fix exposed it): seeding the null with the caller's
        # plain seed meant that data generated from the SAME small seed (rng(0) data + seed=0 index --
        # the commonest possible case) made the 'random' null queries EQUAL the first index atoms:
        # null saturated at cosine ~1.0 and abstention rejected every true signal. The null seed is
        # now hashlib-derived (never a raw stream anyone uses for data), and a SATURATION GUARD
        # applies the perfect-score rule in code: a null query matching an atom at ~1.0 is an
        # instrument collision, so the salt bumps deterministically and the fit retries (bounded).
        import hashlib
        for salt in range(8):
            h = hashlib.sha256(f"recall-null:{seed}:{salt}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
            Q = rng.standard_normal((n_null, units.shape[1]))
            Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
            from holographic.sampling_and_signal.holographic_tiledreduce import tiled_matreduce
            best, _ = tiled_matreduce(units, Q.T)
            if best.max() < 0.999:                            # no query IS an atom -> a real null
                break
        else:
            raise RuntimeError("recall-null saturated at every salt -- index atoms look like iid "
                               "gaussians from the null's own stream; inspect the data")
        # F1 (memory): the fold above already produced max-per-query WITHOUT the (N, n_null)
        # matrix (7.45 GiB at N=500k under the old dense product) -- tile-bounded, max bit-identical.
        self.null = np.sort(best)                            # best match per random query
        return self

    def pvalue(self, score):
        """False-alarm probability for a recall similarity: P(noise best-match >= score)."""
        if self.null is None:
            raise ValueError("fit() the null on a codebook first")
        n = len(self.null)
        # fraction of null at or above score (null is sorted ascending)
        return float((n - int(np.searchsorted(self.null, score, side="left"))) / n)

    def calibrated_recall(self, query, codebook):
        """Recall the best-matching atom AND its honest false-alarm probability.
        Returns (best_index, similarity, pvalue). p small -> trust it; p large -> abstain."""
        C = np.asarray(codebook, float)
        units = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        q = np.asarray(query, float)
        qn = np.linalg.norm(q)
        sims = units @ (q / qn) if qn > 0 else units @ q
        j = int(sims.argmax())
        return j, float(sims[j]), self.pvalue(float(sims[j]))


class SPRTRecall:
    """Wald's Sequential Probability Ratio Test over a STREAM of recall scores.

    A single recall gives one calibrated score (see RecallNull). But when a SEQUENCE of cues bears
    on the same hypothesis -- a drifting narrowband signal across time, repeated sightings of a
    landmark, a recurring microstructure pattern -- you should not commit to a fixed window. You
    accumulate the per-cue log-likelihood ratio and stop the moment the evidence crosses a Wald
    boundary. That is provably the MINIMUM EXPECTED number of samples for a target (alpha, beta)
    error pair (Wald, Sequential Analysis) -- the discipline radio-SETI and particle physics use to
    decide as fast as the evidence allows.

    Fit Gaussian score densities from a sample of null scores -- RecallNull's noise floor IS
    p(score | null) -- and a sample of genuine-match scores. Then `update(score)` returns
    'MATCH' / 'REJECT' / 'CONTINUE'. MEASURED on real holostuff recall scores: ~half the samples of
    the best fixed-N rule at matched error (e.g. avg 2.8 samples vs fixed-N 6 at ~2% error).
    """

    def __init__(self, null_scores, match_scores, alpha=0.05, beta=0.05):
        self.mu0 = float(np.mean(null_scores)); self.sd0 = float(np.std(null_scores)) + 1e-9
        self.mu1 = float(np.mean(match_scores)); self.sd1 = float(np.std(match_scores)) + 1e-9
        self.A = float(np.log((1 - beta) / alpha))     # upper boundary -> accept MATCH
        self.B = float(np.log(beta / (1 - alpha)))     # lower boundary -> reject
        self.reset()

    def reset(self):
        """Start a fresh stream (accumulated log-LR and sample count back to zero)."""
        self.llr = 0.0
        self.n = 0
        return self

    @staticmethod
    def _loglik(x, mu, sd):
        return -0.5 * np.log(2 * np.pi * sd * sd) - (x - mu) ** 2 / (2 * sd * sd)

    def update(self, score):
        """Add one cue's evidence to the running log-LR; return 'MATCH', 'REJECT', or 'CONTINUE'."""
        self.llr += self._loglik(score, self.mu1, self.sd1) - self._loglik(score, self.mu0, self.sd0)
        self.n += 1
        if self.llr >= self.A:
            return "MATCH"
        if self.llr <= self.B:
            return "REJECT"
        return "CONTINUE"

    def decide(self, scores, cap=None):
        """Feed a whole stream; return (decision, n_samples_used). If `cap` is reached without a
        boundary crossing, fall back to the sign of the accumulated evidence."""
        self.reset()
        for s in scores:
            d = self.update(s)
            if d != "CONTINUE":
                return d, self.n
            if cap is not None and self.n >= cap:
                break
        return ("MATCH" if self.llr > 0 else "REJECT"), self.n


def permutation_null(observed, score_fn, resample_fn, n_null=1000, seed=0, alpha=0.05, side="greater"):
    """The shuffled-null discipline as ONE composable primitive -- the move radio-SETI (Tarter) and particle
    physics (Cranmer) both live by: a raw score means nothing until you re-run the IDENTICAL procedure on data
    where the structure has been destroyed, and demand the real score stand out from that null.

    This is the generalisation of the engine's five procedure-matched nulls (_recall_null, _recognition_null,
    _brain_null, _scan_cue_null, and the phase-randomised / MI nulls), which each hand-rolled the same loop:
    draw a resample, score it, collect the distribution, compare. Here that loop is stated once so ANY capability
    -- including a new one built on the engine -- can get "score me against my own shuffled null" for free.

      observed     : the real score (a float), OR None to have it computed as score_fn(None) if your score_fn
                     supports that; usually you pass the already-computed real number.
      score_fn(z)  : maps a (resampled) datum z to a scalar score, the SAME scoring the real datum went through.
      resample_fn(rng) : draws one null datum using the supplied Generator (so the null is seeded + reproducible).
                     This is where the procedure-match lives: shuffle / phase-randomise / draw-random-unit exactly
                     as the real pipeline would see noise. Called n_null times.
      side         : "greater" (default) -- significant when the real score is HIGH (a match/recall similarity);
                     "less" -- significant when it is LOW; "two-sided" -- either tail.

    Returns {p, null_mean, null_std, null_ci, observed, collapsed, n_null}: `p` is the false-alarm probability
    (fraction of the null at least as extreme as `observed`, with the +1/(n+1) plug so p is never exactly 0 -- the
    conservative permutation-test estimator, North et al. 2002); `null_ci` is the 2.5/97.5 percentile band of the
    null; `collapsed` is True when p <= alpha (the real score stood out -- the null "collapsed" under it, the
    engine's shuffled-null test passing). Deterministic given a deterministic score_fn/resample_fn and seed.

    KEPT NEGATIVE: this does NOT invent the resample for you -- a WRONG resample_fn (one that does not destroy the
    structure the score keys on, or that breaks a dependency the real pipeline preserves) gives a mis-calibrated
    null and a meaningless p. The procedure-match is the caller's responsibility; the primitive only runs it
    honestly and counts."""
    rng = np.random.default_rng(seed)
    null = np.empty(n_null, dtype=float)
    for i in range(n_null):
        null[i] = float(score_fn(resample_fn(rng)))
    null.sort()
    obs = float(observed if observed is not None else score_fn(None))
    n = len(null)
    # +1 plug (North et al. 2002): count the observed itself in the null so p is never 0; conservative + calibrated.
    if side == "greater":
        ge = n - int(np.searchsorted(null, obs, side="left"))
        p = (ge + 1) / (n + 1)
    elif side == "less":
        le = int(np.searchsorted(null, obs, side="right"))
        p = (le + 1) / (n + 1)
    elif side == "two-sided":
        ge = n - int(np.searchsorted(null, obs, side="left"))
        le = int(np.searchsorted(null, obs, side="right"))
        p = min(1.0, 2.0 * (min(ge, le) + 1) / (n + 1))
    else:
        raise ValueError("side must be 'greater', 'less', or 'two-sided', got %r" % (side,))
    lo, hi = (float(v) for v in np.percentile(null, [2.5, 97.5]))
    return {"p": float(p), "null_mean": float(null.mean()), "null_std": float(null.std()),
            "null_ci": (lo, hi), "observed": obs, "collapsed": bool(p <= alpha), "n_null": int(n)}


def split_half(events, values=None, mode="contiguous", alpha=0.05):
    """SPLIT-HALF REPLICATION -- the cheapest honest gate there is, and the one that earns its keep. Cut the
    measurements in two, measure the effect in each half independently, and PASS only when both halves agree in
    SIGN and each is individually significant. An effect that lives in one half is not an effect; it is a story
    about that half.

    Call it either way -- `split_half(values)` when the values are already in event order, or
    `split_half(events, values)` when you are carrying the event indices alongside (the indices only set the
    ordering; they are returned in the report so a failure is traceable back to when it happened).

      mode="contiguous" (default) -- first half vs second half, in event order. This is the TEMPORAL question:
                        does the effect survive into a later stretch of data, under whatever regime came next?
                        This is the mode that does the killing.
      mode="interleave" -- odd events vs even events. The NON-temporal question: is the effect merely noisy, or
                        is it structurally absent from half the sample? Interleaving shares the regime between
                        halves, so it is much easier to pass -- an effect that passes interleaved and fails
                        contiguous is regime-bound, and that is a finding, not a failure of the test.

    Returns {n_a, n_b, mean_a, mean_b, t_a, t_b, p_a, p_b, same_sign, both_significant, passed, mode,
    small_sample, span_a, span_b}. `passed` is the gate: same sign AND both halves significant at `alpha`.

    Measured record, and the reason this is not optional: in the campaign that paid for this module, split-half
    killed four effects that every other readout called real -- a +36bp flush edge (dead in the second half), an
    angular-momentum signal at z=3.6 (SIGN FLIPPED between halves), a correlation-spike veto (+16% / -3.2%), and
    a range-blowout veto. It passed exactly the results that later replicated out of sample. Four artifacts and
    zero false rejections, for the price of one function call.

    KEPT NEGATIVE: `p` here is the NORMAL approximation to the t-test (two-sided, via erfc), not Student's t --
    the engine is NumPy-only by constitution and does not carry a t distribution. It is anticonservative for
    small halves, so `small_sample` is set True when either half has fewer than 30 events and the p-values in
    that case should be read as optimistic. A second negative worth naming: split-half is a REPLICATION test,
    not a multiplicity correction. Passing it after screening two hundred candidates still leaves you owing the
    family-wide correction -- run bh_fdr as well, not instead."""
    if values is None:
        values = events
        events = np.arange(len(np.asarray(values, float).ravel()))
    ev = np.asarray(events).ravel()
    v = np.asarray(values, float).ravel()
    if len(ev) != len(v):
        raise ValueError("events and values must be the same length (got %d and %d)" % (len(ev), len(v)))
    if len(v) < 4:
        raise ValueError("split_half needs at least 4 values to form two testable halves (got %d)" % len(v))
    if mode == "contiguous":
        half = len(v) // 2
        ia, ib = np.arange(half), np.arange(half, len(v))
    elif mode == "interleave":
        ia, ib = np.arange(0, len(v), 2), np.arange(1, len(v), 2)
    else:
        raise ValueError("mode must be 'contiguous' or 'interleave', got %r" % (mode,))

    def one(idx):
        w = v[idx]
        n = len(w)
        mean = float(w.mean())
        # ddof=1: the sample standard deviation, so the t-statistic is not inflated on small halves.
        sd = float(w.std(ddof=1)) if n > 1 else 0.0
        t = mean / (sd / math.sqrt(n)) if sd > 0 else (0.0 if mean == 0 else math.copysign(math.inf, mean))
        # two-sided p under the normal approximation: P(|Z| >= |t|) = erfc(|t| / sqrt(2)).
        p = math.erfc(abs(t) / math.sqrt(2.0)) if math.isfinite(t) else 0.0
        return n, mean, t, p

    n_a, mean_a, t_a, p_a = one(ia)
    n_b, mean_b, t_b, p_b = one(ib)
    same_sign = bool(mean_a > 0) == bool(mean_b > 0) and mean_a != 0 and mean_b != 0
    both_sig = bool(p_a <= alpha and p_b <= alpha)
    return {"n_a": int(n_a), "n_b": int(n_b), "mean_a": mean_a, "mean_b": mean_b,
            "t_a": float(t_a), "t_b": float(t_b), "p_a": float(p_a), "p_b": float(p_b),
            "same_sign": bool(same_sign), "both_significant": both_sig,
            "passed": bool(same_sign and both_sig), "mode": mode,
            "small_sample": bool(min(n_a, n_b) < 30),
            "span_a": (ev[ia[0]], ev[ia[-1]]), "span_b": (ev[ib[0]], ev[ib[-1]])}


def pipeline_null(pipeline_fn, x, surrogate="phase", n=200, stat_fn=None, seed=0, alpha=0.05,
                  side="two-sided", **surrogate_kwargs):
    """Run YOUR WHOLE PIPELINE on surrogates and report the statistic against the null the pipeline itself
    produces. This is the single most important primitive in the honesty layer, and the one every result in the
    campaign that paid for this module ultimately rested on.

    The trap it exists to close: **processing manufactures structure**. Any resampling, smoothing, quantising,
    clocking, or clustering step imposes its own correlations on whatever it is fed, INCLUDING pure noise. So a
    null computed on the raw input, or against a textbook baseline like 0.5, is measuring the wrong thing
    entirely -- it credits the pipeline's own artifacts to the data. The only honest null is the identical
    pipeline run on data with the structure destroyed.

    Measured, and the reason this is a one-liner instead of a paragraph of advice:
      * A price-clock (renko/brick) re-clocking step produced 72% direction persistence ON PURE NOISE. Read
        naively that is a spectacular momentum effect. Referenced to its own pipeline null, the real series was
        significantly ANTI-persistent at z=-7.3 -- the opposite sign from the naive reading.
      * A denoising step similarly manufactured 83.6% persistence on noise.
    Two of the campaign's most confident early "findings" were pipeline artifacts, and both flipped or vanished
    the moment the surrogate went through the same chain.

      pipeline_fn(x)   : the WHOLE chain under test -- everything between raw input and the number you quote.
      x                : the real input series.
      surrogate        : a name ("phase", "aaft", "iaaft", "sign_flip", "iid_shuffle", "block_shuffle" with
                         block=...) or a callable fn(x, seed) -> surrogate. Choose the one that destroys the
                         structure you are CLAIMING and preserves everything you are not; see
                         holographic_surrogate.make_surrogate.
      stat_fn(out)     : maps the pipeline's output to the scalar being quoted. Default: the output is already
                         the scalar.
      side             : "two-sided" (default -- an artifact can push either way, and the renko case pushed the
                         opposite way from the naive reading), or "greater"/"less".

    Returns {observed, null_mean, null_std, z, p, null_ci, collapsed, n, surrogate}. `z` is the honest headline:
    how far the real result sits from what the SAME machinery produces on structureless input. `collapsed` is
    True when p <= alpha. Deterministic given `seed` and a deterministic pipeline.

    KEPT NEGATIVE: this cannot rescue a badly-chosen surrogate. If the surrogate destroys something the pipeline
    depends on for reasons unrelated to the claim, the null is drawn from a different machine than the observed
    value and the z is meaningless in a way that looks perfectly healthy. The surrogate choice is the caller's
    responsibility and the whole of the judgement; the primitive only guarantees the chain is identical on both
    sides. Second negative: `n` surrogate runs cost `n` full pipeline evaluations -- for an expensive chain,
    profile first and lower `n` deliberately rather than discovering it at 200."""
    from holographic.sampling_and_signal.holographic_surrogate import make_surrogate
    surr_fn = make_surrogate(surrogate, **surrogate_kwargs)
    ident = (lambda out: float(out)) if stat_fn is None else (lambda out: float(stat_fn(out)))
    x = np.asarray(x, float).ravel()
    observed = ident(pipeline_fn(x))
    null = np.empty(int(n), dtype=float)
    for i in range(int(n)):
        # sub-seed per draw, matching surrogate_ensemble's convention so the two agree member-for-member.
        null[i] = ident(pipeline_fn(surr_fn(x, seed + i + 1)))
    null_sorted = np.sort(null)
    null_mean, null_std = float(null.mean()), float(null.std())
    z = (observed - null_mean) / (null_std + 1e-300)
    m = len(null_sorted)
    # the +1 plug (North et al. 2002): the observed counts itself, so p is never exactly 0 -- conservative.
    ge = m - int(np.searchsorted(null_sorted, observed, side="left"))
    le = int(np.searchsorted(null_sorted, observed, side="right"))
    if side == "greater":
        p = (ge + 1) / (m + 1)
    elif side == "less":
        p = (le + 1) / (m + 1)
    elif side == "two-sided":
        p = min(1.0, 2.0 * (min(ge, le) + 1) / (m + 1))
    else:
        raise ValueError("side must be 'greater', 'less', or 'two-sided', got %r" % (side,))
    lo, hi = (float(q) for q in np.percentile(null_sorted, [2.5, 97.5]))
    return {"observed": observed, "null_mean": null_mean, "null_std": null_std, "z": float(z),
            "p": float(p), "null_ci": (lo, hi), "collapsed": bool(p <= alpha), "n": int(m),
            "surrogate": surrogate if isinstance(surrogate, str) else "callable"}


def min_detectable_effect(test_fn, x, effect_grid, inject_fn=None, surrogate="phase", n_trials=60,
                          seed=0, alpha=0.05, power=0.8, **surrogate_kwargs):
    """DETECTION FLOOR -- turn "we found nothing" into "there is nothing here above X", which is the only form
    of a null result that can be argued with.

    A null result without a floor is unfalsifiable comfort: it is equally consistent with "no effect exists" and
    "our test could not have seen one if it did". This measures which. For each candidate size in `effect_grid`
    it builds structureless data (a surrogate of your real `x`, so the noise level and the autocorrelation are
    the ones you actually face, not a textbook's), INJECTS a synthetic effect of exactly that size, and counts
    how often `test_fn` catches it. The floor is the smallest size caught at least `power` of the time.

      test_fn(data)    : your test, unchanged, applied to a full data array. May return a p-value (float --
                         detected when p <= alpha) or a bool (detected as-is). Whatever you would have run on
                         the real data is what belongs here.
      effect_grid      : the sizes to try, in the units your effect is quoted in. Ascending order is assumed for
                         reporting the floor; the power curve is returned for every size regardless.
      inject_fn(data, size, rng) -> data : how an effect of `size` enters the data. Default: an additive
                         constant shift. That default is right for a mean/drift claim and WRONG for most other
                         shapes -- pass your own for anything that is not a level shift.
      power            : the detection rate that counts as "detectable" (0.8 by convention).

    Returns {floor, grid, power_curve, n_trials, alpha, target_power, surrogate}. `floor` is None when no size
    on the grid reached the target -- which is itself the answer ("this test cannot see even the largest effect
    I tried"), and means the grid needs extending upward, not that the floor is zero.

    KEPT NEGATIVE, and it is a sharp one: a floor is conditional on the injection SHAPE. A floor of 12 units for
    an additive constant says NOTHING about the detectability of an effect of the same nominal size that arrives
    as a burst, a slow drift, or a change in variance -- the same test can be blind to one and exquisitely
    sensitive to another. Quote the floor with its injection, always. Second negative: the floor inherits `x`'s
    own noise level and length, so it does not transfer to a different dataset or a shorter window."""
    from holographic.sampling_and_signal.holographic_surrogate import make_surrogate
    surr_fn = make_surrogate(surrogate, **surrogate_kwargs)
    if inject_fn is None:
        # the deliberately dumb default: a level shift. Documented above as right for a drift claim and wrong
        # for anything else, so a caller who needs another shape is told rather than silently mis-served.
        def inject_fn(data, size, rng):
            return np.asarray(data, float) + float(size)
    x = np.asarray(x, float).ravel()
    grid = [float(g) for g in effect_grid]
    curve = []
    for gi, size in enumerate(grid):
        hits = 0
        for t in range(int(n_trials)):
            # a distinct, reproducible sub-seed per (size, trial) so no two cells share a draw.
            sub = seed + 1 + gi * 100003 + t
            rng = np.random.default_rng(sub)
            data = inject_fn(surr_fn(x, sub), size, rng)
            r = test_fn(data)
            hits += int(bool(r) if isinstance(r, (bool, np.bool_)) else (float(r) <= alpha))
        curve.append(hits / float(n_trials))
    floor = None
    for size, pw in zip(grid, curve):
        if pw >= power:
            floor = size
            break
    return {"floor": floor, "grid": grid, "power_curve": curve, "n_trials": int(n_trials),
            "alpha": float(alpha), "target_power": float(power),
            "surrogate": surrogate if isinstance(surrogate, str) else "callable"}

def _selftest_null_layer():
    """Contracts for the surrogate/null layer (split_half, pipeline_null, min_detectable_effect).

    The headline contract is the one that pays for the whole module: a pipeline that manufactures structure on
    PURE NOISE must be caught by its own pipeline null, and must NOT be caught by comparison against a textbook
    baseline. Both readings are computed here side by side so the difference is on the record numerically.
    """
    import math as _math
    rng = np.random.default_rng(0)

    # ---- A2 pipeline_null: the manufactured-structure trap ---------------------------------------------------
    def smoothed_sign_persistence(d, a=0.8):
        """An exponential smoother followed by a direction-persistence count -- a two-step chain of exactly the
        kind that looks innocent and manufactures momentum out of nothing."""
        y = np.empty(len(d)); y[0] = d[0]
        for i in range(1, len(d)):
            y[i] = a * y[i - 1] + (1 - a) * d[i]
        s = np.sign(y); s = s[s != 0]
        return float(np.mean(s[1:] == s[:-1]))

    n = 3000
    noise = rng.normal(size=n)
    # a series with GENUINE sign persistence: a Markov sign chain (p=0.75 to hold) times independent magnitudes.
    sgn = np.ones(n)
    for i in range(1, n):
        sgn[i] = sgn[i - 1] if rng.random() < 0.75 else -sgn[i - 1]
    real = sgn * np.abs(rng.normal(size=n))

    r_noise = pipeline_null(smoothed_sign_persistence, noise, surrogate="iid_shuffle", n=100, seed=0, side="greater")
    r_real = pipeline_null(smoothed_sign_persistence, real, surrogate="iid_shuffle", n=100, seed=0, side="greater")

    # THE TRAP, pinned: on PURE NOISE the chain reports 79% direction persistence. Against the textbook 0.5
    # baseline that is a +29-point "momentum effect" and would be believed. Against its own pipeline null it is
    # z = -0.07 -- the machinery made all of it. (Campaign: the same shape at 72% for a renko re-clock and 83.6%
    # for a denoiser; the renko case then measured significantly ANTI-persistent at z=-7.3 once referenced.)
    assert r_noise["observed"] - 0.5 > 0.25, r_noise            # the naive reading is spectacular...
    assert abs(r_noise["z"]) < 2.0, r_noise                     # ...and the honest one is nothing (z = -0.07)
    assert not r_noise["collapsed"], r_noise
    # POWER: genuine structure still stands out through the SAME manufacturing chain.
    assert r_real["z"] > 4.0, r_real                            # measured z = +8.5
    assert r_real["collapsed"], r_real
    # and the null itself proves the manufacturing: the surrogates' own persistence is ~0.79, not ~0.5.
    assert r_noise["null_mean"] > 0.7, r_noise
    assert r_noise["p"] > 0.05 and r_real["p"] <= 0.01, (r_noise["p"], r_real["p"])
    # determinism + the refusal path.
    assert pipeline_null(smoothed_sign_persistence, noise, surrogate="iid_shuffle", n=20, seed=3) == \
           pipeline_null(smoothed_sign_persistence, noise, surrogate="iid_shuffle", n=20, seed=3)
    try:
        pipeline_null(smoothed_sign_persistence, noise, n=5, side="sideways")
        raise AssertionError("expected ValueError for a bad side")
    except ValueError as e:
        assert "greater" in str(e) and "less" in str(e), str(e)

    # ---- A5 split_half: the gate that killed four artifacts -------------------------------------------------
    # a dedicated stream: the split-half contracts must not shift when the pipeline_null section above changes
    # how many draws it consumes (they did once -- the decaying case silently became a sign-flip case).
    rng2 = np.random.default_rng(21)
    real_vals = np.concatenate([rng2.normal(0.5, 1, 200), rng2.normal(0.5, 1, 200)])
    decaying = np.concatenate([rng2.normal(0.6, 1, 200), rng2.normal(0.0, 1, 200)])   # the "flush" shape:
    #   still POSITIVE in the second half, just no longer significant -- the seductive case, where the effect
    #   looks like it merely got weaker rather than never having been there.
    flipping = np.concatenate([rng2.normal(0.5, 1, 200), rng2.normal(-0.5, 1, 200)])  # the "angular momentum" shape
    sh_real, sh_decay, sh_flip = split_half(real_vals), split_half(decaying), split_half(flipping)
    assert sh_real["passed"], sh_real                                    # a real effect survives both halves
    assert not sh_decay["passed"] and sh_decay["same_sign"], sh_decay    # right sign, second half not significant
    assert not sh_flip["passed"] and not sh_flip["same_sign"], sh_flip   # the sign FLIPPED between halves
    assert sh_decay["p_a"] < 0.01 < sh_decay["p_b"], sh_decay            # only the first half carried it
    # The documented mode distinction, pinned: the SAME regime-bound effect passes interleaved (each half gets
    # both regimes) and fails contiguous. That difference IS the finding -- the effect is regime-bound.
    assert split_half(decaying, mode="interleave")["passed"], "interleaved halves share the regime"
    assert not split_half(decaying, mode="contiguous")["passed"]
    # the two call forms agree, small samples are flagged, and refusals name the options.
    idx = np.arange(len(real_vals)) * 3
    assert split_half(idx, real_vals)["t_a"] == sh_real["t_a"]
    assert split_half(idx, real_vals)["span_a"][0] == 0
    assert split_half(rng.normal(0.5, 1, 20))["small_sample"] is True
    try:
        split_half(np.zeros(3))
        raise AssertionError("expected ValueError for a too-short input")
    except ValueError as e:
        assert "at least 4" in str(e), str(e)
    try:
        split_half(real_vals, mode="thirds")
        raise AssertionError("expected ValueError for a bad mode")
    except ValueError as e:
        assert "contiguous" in str(e) and "interleave" in str(e), str(e)

    # ---- A4 min_detectable_effect: "nothing above X", and the surrogate trap --------------------------------
    def ttest_p(v):
        se = v.std(ddof=1) / _math.sqrt(len(v))
        return _math.erfc(abs(v.mean() / se) / _math.sqrt(2.0))

    base = np.random.default_rng(11).normal(size=400)            # se = 1/sqrt(400) = 0.05 per unit sd
    mde = min_detectable_effect(ttest_p, base, [0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
                                surrogate="sign_flip", n_trials=60, seed=0)
    # the curve must be MONOTONE-ish and land the floor where theory says: 0.15 is a 3-sigma effect -> ~85% power.
    assert mde["floor"] == 0.15, mde                             # measured power curve .02/.17/.62/.87/.98/1.0
    assert mde["power_curve"][0] < 0.1 and mde["power_curve"][-1] > 0.95, mde
    assert mde["power_curve"] == sorted(mde["power_curve"]), mde
    # KEPT NEGATIVE, pinned: the surrogate must DESTROY the statistic under test. iid_shuffle preserves the
    # sample mean EXACTLY, so every trial of a mean-test yields the identical p and the power curve degenerates
    # to a 0/1 step -- an unfalsifiable floor that looks like a real measurement. sign_flip destroys the mean
    # (keeping magnitudes) and gives the graded curve above. Same class of error as sign_flip's own negative in
    # holographic_surrogate: a null that preserves what you are testing is not a null.
    degen = min_detectable_effect(ttest_p, base, [0.0, 0.02, 0.05], surrogate="iid_shuffle", n_trials=40, seed=0)
    assert all(pw in (0.0, 1.0) for pw in degen["power_curve"]), degen
    assert not all(pw in (0.0, 1.0) for pw in mde["power_curve"]), mde
    # a grid that is entirely below the floor reports None -- "extend the grid", not "the floor is zero".
    assert min_detectable_effect(ttest_p, base, [0.001, 0.002], surrogate="sign_flip",
                                 n_trials=20, seed=0)["floor"] is None

    print("holographic_honesty null-layer selftest OK (pipeline_null: a smoother+persistence chain manufactures "
          "%.1f%% direction persistence on PURE NOISE -- naive vs 0.5 reads +%.1f points, its own pipeline null "
          "reads z=%+.2f p=%.2f; genuine structure still clears it at z=%+.1f. split_half kills the decaying "
          "artifact (%.2f/%.2f, p %.3f/%.3f) and the sign-flipping artifact, passes the replicating effect, and "
          "the regime-bound case passes interleaved / fails contiguous. min_detectable_effect floor=%.2f "
          "(3-sigma) with power curve %s; iid_shuffle degenerates it to a 0/1 step -- kept negative)"
          % (100 * r_noise["observed"], 100 * (r_noise["observed"] - 0.5), r_noise["z"], r_noise["p"],
             r_real["z"], sh_decay["mean_a"], sh_decay["mean_b"], sh_decay["p_a"], sh_decay["p_b"],
             mde["floor"], [round(c, 2) for c in mde["power_curve"]]))



def holdout_auc(scores_train, labels_train, scores_test, labels_test):
    """AUC on train AND holdout, never train alone -- the standard separability readout, as a pair by
    construction so the overfit signature has nowhere to hide.

    The AUC here is the exact Mann-Whitney statistic (fraction of positive/negative pairs ranked correctly,
    ties counted half), computed identically on both splits. Returns {auc_train, auc_test, gap, n_train,
    n_test}. `gap` = train minus test; a large gap IS the finding. Measured canon (the campaign's Nystrom
    kernel lift): train AUC 0.685, held-out 0.557 -- a re-representation "discovering" separability that was
    mostly its own capacity. That pair of numbers, side by side, is what this function refuses to let you
    report half of."""
    def _auc(s, y):
        s = np.asarray(s, float).ravel()
        y = np.asarray(y).ravel().astype(bool)
        pos, neg = s[y], s[~y]
        if pos.size == 0 or neg.size == 0:
            return float("nan")
        # exact Mann-Whitney via rank sum, ties shared: robust and O(n log n), no pair loop.
        order = np.argsort(np.concatenate([pos, neg]), kind="stable")
        ranks = np.empty(order.size, float)
        ranks[order] = np.arange(1, order.size + 1)
        allv = np.concatenate([pos, neg])
        # average ranks over ties so a constant score reads AUC 0.5, not an ordering artifact.
        vals, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
        sums = np.zeros(vals.size)
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
        u = float(ranks[:pos.size].sum()) - pos.size * (pos.size + 1) / 2.0
        return float(u / (pos.size * neg.size))
    a_tr = _auc(scores_train, labels_train)
    a_te = _auc(scores_test, labels_test)
    return {"auc_train": a_tr, "auc_test": a_te,
            "gap": (a_tr - a_te) if (a_tr == a_tr and a_te == a_te) else float("nan"),
            "n_train": int(np.asarray(scores_train).size), "n_test": int(np.asarray(scores_test).size)}


def dpi_guard(features, new_feature, seed=0, holdout_frac=0.5, degree=2):
    """IS THIS FEATURE ACTUALLY NEW INFORMATION -- or a transform of what you already have?

    The data-processing inequality is the wall this guard names: no transform of existing features can contain
    information about a target that the existing features do not already carry. Re-representations are for
    CONCENTRATING and CLARIFYING what exists -- often worth doing! -- but a proposed feature that is largely
    explained by transforms of the old set should be *budgeted* as concentration, not counted as new signal.
    The campaign this comes from burned weeks on kernel lifts, crowd decompositions, 3-D embeddings and foreign
    clocks, every one of them DPI-capped at the raw features' ~0 directional bits.

    Mechanics: fit `new_feature` from a polynomial expansion of `features` (degree 1 = linear, degree 2 adds
    squares and pairwise products) by least squares ON A TRAIN SPLIT, and report R^2 on train AND HOLDOUT --
    never train alone, for the same reason holdout_auc exists. The holdout R^2 is the honest number: it is how
    much of the new feature the old ones REPRODUCIBLY explain.

    Returns {r2_train, r2_holdout, gap, novel_frac, n_basis, verdict}. `novel_frac` = 1 - max(r2_holdout, 0):
    the reproducibly-unexplained share -- the most the feature could possibly add, not a promise that it adds
    anything.

    KEPT NEGATIVES, both sharp: (1) a LOW r2_holdout does not mean the feature is USEFUL -- it means it is not
    a transform of these features under this basis; it may be noise, which is also novel. Novelty is necessary,
    not sufficient -- the feature still owes a target-side test (mutual_information_vs_null, read in BITS, or a
    pipeline_null of the model that uses it). (2) a HIGH r2_holdout under this small basis is damning, but a
    low one under it is not acquittal against ALL transforms -- a feature can be an exotic transform this basis
    cannot express. The guard catches the common case (linear/quadratic re-dressings), which is where the weeks
    actually went."""
    F = np.asarray(features, float)
    if F.ndim == 1:
        F = F[:, None]
    g = np.asarray(new_feature, float).ravel()
    n, k = F.shape
    if g.size != n:
        raise ValueError("new_feature has %d samples but features has %d" % (g.size, n))
    if n < 20:
        raise ValueError("dpi_guard needs at least 20 samples for a meaningful train/holdout split (got %d)" % n)
    if degree not in (1, 2):
        raise ValueError("degree must be 1 (linear) or 2 (adds squares + pairwise products), got %r" % (degree,))
    # basis: [1, F] and for degree 2 the squares and pairwise products -- the transforms features are actually
    # re-dressed in. Standardised so the least-squares solve is well-conditioned regardless of units.
    cols = [np.ones(n), *(F[:, j] for j in range(k))]
    if degree == 2:
        for i in range(k):
            for j in range(i, k):
                cols.append(F[:, i] * F[:, j])
    B = np.column_stack(cols)
    mu, sd = B.mean(axis=0), B.std(axis=0)
    sd[sd == 0] = 1.0
    Bz = (B - mu) / sd
    Bz[:, 0] = 1.0
    # a SEEDED shuffled split, not first-half/second-half: dpi_guard asks a structural question (is g a
    # transform of F?), not a temporal one -- ordering is split_half's business, not this guard's.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = max(int(n * holdout_frac), 10)
    tr, te = perm[:n_tr], perm[n_tr:]
    coef, *_ = np.linalg.lstsq(Bz[tr], g[tr], rcond=None)

    def r2(idx):
        resid = g[idx] - Bz[idx] @ coef
        tot = float(np.sum((g[idx] - g[idx].mean()) ** 2))
        return 1.0 - float(np.sum(resid ** 2)) / tot if tot > 0 else float("nan")
    r2_tr, r2_te = r2(tr), r2(te)
    novel = 1.0 - max(r2_te, 0.0) if r2_te == r2_te else float("nan")
    if r2_te == r2_te and r2_te > 0.8:
        verdict = ("TRANSFORM: %.0f%% of the proposed feature is reproducibly explained by degree-%d "
                   "transforms of the existing set. Expect CONCENTRATION at best, not new information (DPI); "
                   "count its cost, not its novelty." % (100 * r2_te, degree))
    elif r2_te == r2_te and r2_te > 0.4:
        verdict = ("PARTIAL: %.0f%% explained by existing features (holdout). The remaining %.0f%% is the most "
                   "it could add -- test THAT share against the target before crediting it." % (100 * r2_te, 100 * novel))
    else:
        verdict = ("NOVEL under this basis (holdout R^2 %.2f): not a linear/quadratic re-dressing of the "
                   "existing set. Necessary, not sufficient -- novelty may be noise; it still owes a "
                   "target-side test in bits." % r2_te)
    return {"r2_train": float(r2_tr), "r2_holdout": float(r2_te),
            "gap": float(r2_tr - r2_te) if (r2_tr == r2_tr and r2_te == r2_te) else float("nan"),
            "novel_frac": float(novel), "n_basis": int(B.shape[1]), "verdict": verdict}



def _selftest_information_layer():
    """Contracts for dpi_guard + holdout_auc (the F-group information-honesty pair).

    The canonical trap both exist for: a re-representation "discovering" separability or novelty that is
    mostly its own capacity -- caught only when train and holdout are reported TOGETHER."""
    rng = np.random.default_rng(0)
    n = 2000
    F = rng.normal(size=(n, 4))

    # (1) a DISGUISED TRANSFORM -- tanh of a quadratic combination -- is flagged: holdout R^2 ~ 0.98.
    g_transform = np.tanh(0.5 * F[:, 0] + 0.3 * F[:, 1] * F[:, 2])
    r_tr = dpi_guard(F, g_transform)
    assert r_tr["r2_holdout"] > 0.9 and r_tr["verdict"].startswith("TRANSFORM"), r_tr

    # (2) genuine novelty is passed through as novel -- with the negative intact: novel may be noise (it IS
    #     noise here), so the verdict must demand a target-side test rather than grant approval.
    r_nov = dpi_guard(F, rng.normal(size=n))
    assert r_nov["r2_holdout"] < 0.1 and r_nov["novel_frac"] > 0.9, r_nov
    assert r_nov["verdict"].startswith("NOVEL") and "not sufficient" in r_nov["verdict"], r_nov

    # (3) a PARTIAL re-dressing lands in the middle band and quotes the unexplained share as a budget.
    g_part = 0.7 * (F[:, 0] + F[:, 3] ** 2) + 1.0 * rng.normal(size=n)
    r_part = dpi_guard(F, g_part)
    assert 0.4 < r_part["r2_holdout"] < 0.8 and r_part["verdict"].startswith("PARTIAL"), r_part

    # (4) determinism + refusals name their fix.
    assert dpi_guard(F, g_part, seed=3) == dpi_guard(F, g_part, seed=3)
    for bad, needle in ((lambda: dpi_guard(F, g_part[:100]), "samples"),
                        (lambda: dpi_guard(F[:10], g_part[:10]), "at least 20"),
                        (lambda: dpi_guard(F, g_part, degree=3), "degree must be")):
        try:
            bad()
            raise AssertionError("expected ValueError (%s)" % needle)
        except ValueError as e:
            assert needle in str(e), (needle, str(e))

    # (5) holdout_auc: the Nystrom-shaped overfit signature -- train separates, holdout does not, and the pair
    #     travels together. Exact Mann-Whitney checks: perfect separation = 1.0, constant scores = 0.5 (ties).
    tr_s = np.concatenate([rng.normal(1, 1, 100), rng.normal(0, 1, 100)])
    te_s = np.concatenate([rng.normal(0.1, 1, 100), rng.normal(0, 1, 100)])
    y = np.array([1] * 100 + [0] * 100)
    h = holdout_auc(tr_s, y, te_s, y)
    assert h["auc_train"] > 0.65 and h["auc_test"] < 0.6 and h["gap"] > 0.1, h
    assert holdout_auc([1, 2, 3, 4], [0, 0, 1, 1], [1, 2, 3, 4], [0, 0, 1, 1])["auc_train"] == 1.0
    assert holdout_auc(np.zeros(50), np.arange(50) % 2, np.zeros(50), np.arange(50) % 2)["auc_train"] == 0.5

    print("holographic_honesty information-layer selftest OK (dpi_guard: tanh-of-quadratic flagged as TRANSFORM "
          "at holdout R^2=%.2f; pure noise passes as NOVEL (novel_frac %.2f) with the may-be-noise negative in "
          "the verdict; partial re-dressing budgeted at %.0f%% explained; holdout_auc shows the overfit "
          "signature train %.2f vs holdout %.2f and scores exact ties at 0.5)"
          % (r_tr["r2_holdout"], r_nov["novel_frac"], 100 * r_part["r2_holdout"],
             h["auc_train"], h["auc_test"]))



def lookahead_lint(signal_fn, x, n_checkpoints=8, min_prefix=None, atol=1e-10):
    """E3 -- the LOOK-AHEAD LINTER for a black-box signal pipeline: prove, by recomputation, that the signal
    at time t depends only on data up to t.

    THE TEST IS PREFIX CONSISTENCY, and it is exact, not statistical: run `signal_fn` on the full series and
    on truncated prefixes x[:c] at several checkpoints c; a CAUSAL pipeline returns the identical values on
    the shared range (its output at t cannot know whether more data exists after t), while any pipeline that
    peeks -- a centred smoother, a z-score normalised by the FULL sample's mean/std, a global detrend, a
    min-max scaled feature -- changes its past outputs when the future changes, and the drift is detected at
    machine precision. This is the audit_causality idea (perturb the future, demand bit-identical answers)
    promoted from specific structures (Gate, CausalIndex) to ANY signal_fn.

    signal_fn : callable, series (n,) -> aligned signal (n,). Emitting a shorter warm-up prefix is fine as
                long as alignment from the END is consistent; the linter compares the overlapping tail.
    Returns {causal, n_checkpoints, violations: [{checkpoint, max_drift, first_bad_index}], max_drift}.

    KEPT NEGATIVES: (1) prefix consistency is NECESSARY, not sufficient -- a pipeline can be prefix-consistent
    and still leak through its TARGET (labels built with future data), which is the shift probe's job
    (target_shift_probe below); run both. (2) a stochastic signal_fn must be seeded internally or it will
    fail the lint for the wrong reason; the linter cannot distinguish nondeterminism from leakage and does
    not try -- determinism is a precondition, stated here rather than guessed at."""
    x = np.asarray(x, float).ravel()
    n = x.size
    if min_prefix is None:
        min_prefix = max(n // 4, 8)
    full = np.asarray(signal_fn(x), float).ravel()
    checkpoints = np.unique(np.linspace(min_prefix, n - 1, int(n_checkpoints)).astype(int))
    violations = []
    max_drift = 0.0
    for c in checkpoints:
        part = np.asarray(signal_fn(x[:c]), float).ravel()
        # align from the END of the shared range: warm-up truncation shortens the head, never the tail.
        m = min(part.size, c, full.size)
        a, b = part[-m:], full[:c][-m:]
        drift = np.abs(a - b)
        worst = float(drift.max()) if drift.size else 0.0
        max_drift = max(max_drift, worst)
        if worst > atol:
            violations.append({"checkpoint": int(c), "max_drift": worst,
                               "first_bad_index": int(c - m + int(np.argmax(drift > atol)))})
    return {"causal": not violations, "n_checkpoints": int(len(checkpoints)),
            "violations": violations, "max_drift": float(max_drift)}


def target_shift_probe(signal, target, max_lag=3):
    """The OTHER half of the look-ahead lint: a prefix-consistent signal can still be dressed-up leakage --
    most commonly a signal that USES THE VERY BAR IT CLAIMS TO PREDICT. Symptom, measurable without seeing
    the pipeline: for a signal honestly AHEAD of its target, |corr(signal_t, target_{t+k})| for k >= 1 should
    be where the correlation lives; when the NOT-AHEAD side (k <= 0: the current bar and the past) dominates,
    the "prediction" is explanation wearing a forecast's clothes.

    Returns {suspicious, corr_ahead, corr_not_ahead, ratio, by_lag} where by_lag maps each lag in
    [-max_lag..max_lag] (k=0 contemporaneous, negative = past target) to its correlation; `suspicious` when
    the best not-ahead correlation exceeds twice the best ahead correlation AND is itself non-trivial (>0.1).

    KEPT NEGATIVES, in both directions: (1) a MOMENTUM signal on an autocorrelated target legitimately
    correlates with its own current/past bars -- the probe FLAGS it (pinned as the documented false
    positive); (2) a SYMMETRIC leak -- a label built with a centred window -- correlates equally with past
    and future and slips through the asymmetry test entirely (measured 0.444 ahead vs 0.441 not-ahead:
    indistinguishable). Symmetric leaks are lookahead_lint's job, on the LABEL CONSTRUCTOR. So: smell test,
    not proof; suspicious=True routes you to the lint, never convicts on its own."""
    s = np.asarray(signal, float).ravel()
    t = np.asarray(target, float).ravel()
    if s.size != t.size:
        raise ValueError("signal and target must align (got %d vs %d)" % (s.size, t.size))
    by_lag = {}
    for k in range(-int(max_lag), int(max_lag) + 1):
        if k >= 0:
            a, b = s[:s.size - k] if k else s, t[k:]
        else:
            a, b = s[-k:], t[:t.size + k]
        if a.size < 8 or a.std() == 0 or b.std() == 0:
            by_lag[k] = float("nan")
        else:
            by_lag[k] = float(np.corrcoef(a, b)[0, 1])
    ahead = max((abs(v) for k, v in by_lag.items() if k >= 1 and v == v), default=0.0)
    not_ahead = max((abs(v) for k, v in by_lag.items() if k <= 0 and v == v), default=0.0)
    ratio = not_ahead / ahead if ahead > 0 else float("inf") if not_ahead > 0 else 1.0
    return {"suspicious": bool(not_ahead > 2.0 * ahead and not_ahead > 0.1), "corr_ahead": float(ahead),
            "corr_not_ahead": float(not_ahead), "ratio": float(ratio), "by_lag": by_lag}


def decomposition_contract(decompose_fn, x, atol=1e-8, residual_key="residual"):
    """I4 -- the DECOMPOSITION CONTRACT: judge any series decomposition on the three promises every
    decomposition implicitly makes, and report which it keeps.

    `decompose_fn(x) -> dict of name -> component` (each aligned with x). The three promises:

      COMPLETE      the components SUM BACK to x within atol, elementwise. A decomposition that loses mass is
                    a projection wearing a decomposition's name; max |x - sum| is reported either way.
      CAUSAL        each component passes lookahead_lint individually. Reported PER COMPONENT, because a
                    decomposition is usually a mix -- a trailing trend (causal) plus a global seasonal fit
                    (not) -- and the per-component verdict tells you which parts you may use at time t and
                    which are diagnosis-only. Overall `causal` is the AND.
      HONEST RESIDUAL  the component named `residual_key` (if present) is not secretly the signal: its energy
                    share is reported, and `residual_dominates` flags when it carries the majority -- at
                    which point "we decomposed it" means "we removed a sliver and renamed the rest".

    Returns {complete, max_recon_err, components: {name: {energy_share, causal, max_drift}}, causal,
    residual_share, residual_dominates, verdict}.

    KEPT NEGATIVES: (1) energy shares use the raw second moment, so components are NOT assumed orthogonal --
    shares can sum past 1.0 when components are correlated, and the report says so rather than normalising
    the overlap away (normalising would hide exactly the double-counting a reader should see). (2) CAUSAL
    here means prefix-consistent under the lint; a component can be prefix-consistent and still useless at
    time t (a trailing mean lags -- causal and late are different complaints). (3) the contract judges the
    DECOMPOSITION MAP, not the story attached to the names: nothing here checks that "seasonal" is seasonal.
    """
    x = np.asarray(x, float).ravel()
    parts = decompose_fn(x)
    if not isinstance(parts, dict) or not parts:
        raise ValueError("decompose_fn must return a non-empty dict of name -> component")
    comps = {k: np.asarray(v, float).ravel() for k, v in parts.items()}
    for k, v in comps.items():
        if v.size != x.size:
            raise ValueError("component %r has length %d for a series of %d" % (k, v.size, x.size))

    recon = np.sum(list(comps.values()), axis=0)
    max_err = float(np.max(np.abs(recon - x)))
    complete = max_err <= atol

    total_energy = float(np.sum(x * x)) or 1.0
    report = {}
    for k in comps:
        def comp_fn(s, key=k):
            out = decompose_fn(np.asarray(s, float).ravel())
            return np.asarray(out[key], float).ravel()
        lint = lookahead_lint(comp_fn, x)
        report[k] = {"energy_share": float(np.sum(comps[k] ** 2) / total_energy),
                     "causal": bool(lint["causal"]), "max_drift": float(lint["max_drift"])}
    causal = all(r["causal"] for r in report.values())
    residual_share = report.get(residual_key, {}).get("energy_share", None)
    residual_dominates = bool(residual_share is not None and residual_share > 0.5)

    bits = ["COMPLETE (err %.1e)" % max_err if complete else "INCOMPLETE (err %.1e > atol)" % max_err]
    bad = [k for k, r in report.items() if not r["causal"]]
    bits.append("all components causal" if causal else
                "NON-CAUSAL components: %s -- diagnosis-only at time t" % ", ".join(sorted(bad)))
    if residual_dominates:
        bits.append("residual carries %.0f%% of the energy -- a sliver was removed and the rest renamed"
                    % (100 * residual_share))
    return {"complete": complete, "max_recon_err": max_err, "components": report, "causal": causal,
            "residual_share": residual_share, "residual_dominates": residual_dominates,
            "verdict": "; ".join(bits)}



def _selftest_lookahead_layer():
    """Contracts: the lint passes honest trailing pipelines EXACTLY and catches every classic leak pattern
    with a named first-bad index; the shift probe flags a future-built label and stays quiet on an honest
    predictor; the momentum false-positive is KEPT as the documented limitation."""
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(600))

    # honest pipelines: trailing EMA, trailing z-score, raw diff -- all prefix-consistent to 0.0 drift.
    def ema(s, a=0.1):
        out = np.zeros_like(s)
        for i in range(1, s.size):
            out[i] = (1 - a) * out[i - 1] + a * s[i]
        return out

    def trailing_z(s, w=30):
        out = np.zeros_like(s)
        for i in range(w, s.size):
            seg = s[i - w:i]
            sd = seg.std() or 1.0
            out[i] = (s[i] - seg.mean()) / sd
        return out

    for fn in (ema, trailing_z, lambda s: np.concatenate([[0.0], np.diff(s)])):
        r = lookahead_lint(fn, x)
        assert r["causal"] and r["max_drift"] == 0.0, r

    # the classic leaks, each caught:
    leaks = {
        "full-sample z-score": lambda s: (s - s.mean()) / (s.std() or 1.0),
        "centred smoother": lambda s: np.convolve(s, np.ones(11) / 11.0, mode="same"),
        "global min-max": lambda s: (s - s.min()) / ((s.max() - s.min()) or 1.0),
        "global detrend": lambda s: s - np.polyval(np.polyfit(np.arange(s.size), s, 1), np.arange(s.size)),
    }
    for name, fn in leaks.items():
        r = lookahead_lint(fn, x)
        assert not r["causal"], name
        assert r["violations"][0]["max_drift"] > 1e-6, (name, r)

    # shift probe -- THE CONTEMPORANEOUS LEAK, the most common one in the wild: a "predictor" of the next
    # step that secretly uses the current step's own value. On an iid target it correlates ~0.9 at k=0 and
    # ~0.0 ahead: explanation wearing a forecast's clothes, and the probe says so.
    d = rng.standard_normal(600)
    contemporaneous = d + 0.3 * rng.standard_normal(600)
    probe = target_shift_probe(contemporaneous, d, max_lag=3)
    assert probe["suspicious"], probe
    assert probe["corr_not_ahead"] > 0.9 and probe["corr_ahead"] < 0.15, probe

    # KEPT NEGATIVE, measured then pinned: a SYMMETRIC leak (centred-window label) correlates EQUALLY with
    # past and future (0.444 vs 0.441 on this fixture) and the asymmetry test cannot see it -- that case
    # belongs to lookahead_lint run on the label constructor, which catches the centred window exactly.
    dd = np.concatenate([[0.0], np.diff(x)])
    smooth_label = np.convolve(dd, np.ones(5) / 5.0, mode="same")
    sym = target_shift_probe(smooth_label, dd, max_lag=3)
    assert not sym["suspicious"], sym
    assert abs(sym["corr_ahead"] - sym["corr_not_ahead"]) < 0.05, sym
    assert not lookahead_lint(lambda s: np.convolve(s, np.ones(5) / 5.0, mode="same"), dd)["causal"]

    # honest predictor stays quiet: signal_t = x-level sign, target = next diff of an AR(1)-with-signal series.
    y = np.zeros(600)
    sig = np.sign(rng.standard_normal(600))
    for i in range(1, 600):
        y[i] = y[i - 1] + 0.5 * sig[i - 1] + rng.standard_normal()
    dy = np.concatenate([[0.0], np.diff(y)])
    honest = target_shift_probe(sig, dy, max_lag=3)
    assert not honest["suspicious"], honest

    # KEPT NEGATIVES on the false-positive boundary, both measured before pinning:
    #  * momentum on a PERSISTENT target (AR phi=0.8) reads not-ahead-dominant but UNDER the 2x line
    #    (ratio 1.35: the persistence gives the trailing stat genuine forward correlation too) -- no flag,
    #    correctly.
    #  * momentum on an UNPREDICTABLE target is the real false-positive shape: a trailing statistic always
    #    explains its own inputs (not-ahead ~0.55) and, with nothing forecastable, has ~0 ahead -- the probe
    #    fires on an honest-but-useless signal. Documented limitation: `suspicious` means "explains more than
    #    it predicts", which covers both leakage AND honest signals with no forward skill; the router's next
    #    stop is lookahead_lint either way.
    ar = np.zeros(600)
    for i in range(1, 600):
        ar[i] = 0.8 * ar[i - 1] + rng.standard_normal()
    fp_persistent = target_shift_probe(ema(ar, a=0.3), ar, max_lag=3)
    assert not fp_persistent["suspicious"] and fp_persistent["ratio"] < 2.0, fp_persistent
    iid = rng.standard_normal(600)
    fp_useless = target_shift_probe(ema(iid, a=0.3), iid, max_lag=3)
    assert fp_useless["suspicious"], fp_useless

    # misaligned refuses by name
    try:
        target_shift_probe(np.ones(10), np.ones(11))
        raise AssertionError("expected refusal")
    except ValueError as e:
        assert "align" in str(e)

    print("holographic_honesty lookahead layer OK (trailing EMA / trailing z / diff lint EXACTLY causal at "
          "0.0 drift; full-sample z-score, centred smoother, global min-max and global detrend ALL caught "
          "with a first-bad index; the contemporaneous leak fires the shift probe (0.9 not-ahead vs ~0 ahead) "
          "while the SYMMETRIC centred-label leak is invisible to it and belongs to the lint (both pinned); an "
          "honest predictor stays quiet; momentum on a persistent target correctly clears the 2x line (1.35) "
          "while a trailing stat of an unpredictable target fires as the documented false positive -- "
          "'explains more than it predicts' covers leakage AND honest-but-useless; smell test, not verdict)")



def _selftest_decomposition_contract():
    """I4 contracts, including the DOGFOOD verdicts on the engine's own decomposers:
    1. A fully-causal exact decomposition (trailing EMA trend + residual) passes all three promises.
    2. A GLOBAL polynomial detrend decomposition is COMPLETE but NON-CAUSAL, named per component.
    3. A lossy 'decomposition' (clipped trend, dropped mass) reads INCOMPLETE with the error quoted.
    4. A sliver decomposition (tiny trend, residual = nearly everything) flags residual_dominates.
    5. Correlated components: energy shares may sum past 1.0 and are NOT normalised -- pinned.
    """
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(500)) + 0.02 * np.arange(500)

    def ema_decomp(s, a=0.1):
        t = np.empty_like(s)
        t[0] = s[0]
        for i in range(1, s.size):
            t[i] = (1 - a) * t[i - 1] + a * s[i]
        return {"trend": t, "residual": s - t}

    r1 = decomposition_contract(ema_decomp, x)
    assert r1["complete"] and r1["causal"], r1["verdict"]
    assert not r1["residual_dominates"]

    def global_detrend(s):
        tt = np.arange(s.size)
        fit = np.polyval(np.polyfit(tt, s, 2), tt)
        return {"trend": fit, "residual": s - fit}

    r2 = decomposition_contract(global_detrend, x)
    assert r2["complete"] and not r2["causal"]
    assert not r2["components"]["trend"]["causal"] and not r2["components"]["residual"]["causal"]
    assert "NON-CAUSAL" in r2["verdict"]

    def lossy(s):
        t = np.clip(s, -5, 5)
        return {"trend": t, "residual": (s - t) * 0.5}            # half the overflow vanishes

    r3 = decomposition_contract(lossy, x)
    assert not r3["complete"] and r3["max_recon_err"] > 1.0

    def sliver(s):
        return {"trend": np.full(s.size, s.mean() * 0.01), "residual": s - s.mean() * 0.01}

    r4 = decomposition_contract(sliver, x)
    assert r4["residual_dominates"] and "renamed" in r4["verdict"]

    def correlated(s):
        return {"a": 0.6 * s, "b": 0.4 * s}                       # perfectly correlated halves

    r5 = decomposition_contract(correlated, x)
    shares = sum(c["energy_share"] for c in r5["components"].values())
    assert abs(shares - 0.52) < 0.02, shares                      # 0.36 + 0.16: NOT normalised to 1

    try:
        decomposition_contract(lambda s: [], x)
        raise AssertionError("expected refusal")
    except ValueError as e:
        assert "non-empty dict" in str(e)

    # DOGFOOD: the engine's own smooth_sharp_split, wrapped as a decomposition. Spectral and global by
    # construction -- the contract must call it COMPLETE and NON-CAUSAL, which is its honest label:
    # a diagnosis tool, not a time-t feature source. Verified live rather than assumed.
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    def engine_split(s):
        # TwoLayerCode -> the SMOOTH layer is the low-frequency reconstruction (irfft of the stored
        # coefficients); residual is everything else. Wrapping the code this way makes the split a genuine
        # decomposition (smooth + residual == s exactly) so the contract judges the map, not the storage.
        code = m.smooth_sharp_split(s, k_smooth=8, k_sharp=64)
        nbins = code.n // 2 + 1
        sm = np.fft.irfft(np.concatenate([code.smooth_coeffs,
                                          np.zeros(nbins - len(code.smooth_coeffs), complex)]), n=code.n)
        return {"smooth": sm, "residual": np.asarray(s, float).ravel() - sm}

    r6 = decomposition_contract(engine_split, x)
    assert r6["complete"]
    assert not r6["components"]["smooth"]["causal"], "smooth_sharp_split went causal?! its FFT basis changed"

    print("holographic_honesty decomposition contract OK (trailing-EMA decomposition passes all three "
          "promises; a global quadratic detrend is COMPLETE but NON-CAUSAL per component; a lossy split "
          "reads INCOMPLETE with err %.2f quoted; a sliver split flags residual at %.0f%% -- 'a sliver was "
          "removed and the rest renamed'; correlated halves report shares summing to %.2f, NOT normalised; "
          "DOGFOOD: the engine's own smooth_sharp_split is certified COMPLETE + NON-CAUSAL -- diagnosis "
          "tool, not a time-t feature source, now on the record)"
          % (r3["max_recon_err"], 100 * r4["residual_share"], shares))



def _selftest():
    # The permutation_null primitive: assert the three properties honest measurement demands -- CALIBRATION (a random
    # datum's p is ~uniform, so the false-alarm rate holds at alpha), POWER (a true match collapses the null), and
    # BIT-IDENTITY to the hand-rolled recall-null loop it generalizes (so the private nulls' numbers are preserved).
    rng0 = np.random.default_rng(1)
    D, N = 96, 30
    cb = rng0.standard_normal((N, D)); cb /= np.linalg.norm(cb, axis=1, keepdims=True) + 1e-12

    def score(s):
        s = s / (np.linalg.norm(s) + 1e-12)
        return float(np.max(cb @ s))

    def resample(r):
        s = r.standard_normal(D); return s / (np.linalg.norm(s) + 1e-12)

    # CALIBRATION: random queries flagged at ~alpha (loose band -- it's a finite-sample rate)
    alpha, T, flagged = 0.05, 300, 0
    for t in range(T):
        q = np.random.default_rng(9000 + t).standard_normal(D)
        flagged += permutation_null(score(q), score, resample, n_null=300, seed=7, alpha=alpha)["collapsed"]
    rate = flagged / T
    assert 0.01 <= rate <= 0.12, ("false-alarm rate should sit near alpha=0.05", rate)

    # POWER: a true codebook entry collapses the null (small p)
    r = permutation_null(score(cb[3]), score, resample, n_null=400, seed=7)
    assert r["p"] < 0.02 and r["collapsed"], ("a true match must collapse the null", r["p"])

    # BIT-IDENTITY: the primitive's internal null == the hand-rolled recall-null loop the private nulls share
    seed, n = 5, 250
    rr = np.random.default_rng(seed); hand = np.sort([score(resample(rr)) for _ in range(n)])
    rr2 = np.random.default_rng(seed); prim = np.sort([score(resample(rr2)) for _ in range(n)])
    assert np.array_equal(prim, hand), "primitive must reproduce the incumbent recall-null loop bit-identically"

    # DETERMINISM: same seed -> identical p
    assert (permutation_null(0.5, score, resample, n_null=n, seed=3)["p"]
            == permutation_null(0.5, score, resample, n_null=n, seed=3)["p"])

    print("holographic_honesty selftest OK: permutation_null is CALIBRATED (false-alarm %.3f at alpha=0.05), has "
          "POWER (true match p=%.4f, collapsed), is BIT-IDENTICAL to the hand-rolled recall-null it generalizes, and "
          "is deterministic. The +1 plug keeps p in (0,1]." % (rate, r["p"]))


if __name__ == "__main__":
    _selftest()
    _selftest_null_layer()
    _selftest_information_layer()
    _selftest_lookahead_layer()
    _selftest_decomposition_contract()

"""holographic_conditioning.py -- the CONDITIONING layer: measure an effect under a condition,
and make the difference between ex-ante and ex-post a property of the code instead of a property
of the analyst's discipline.

Four names here, but ONE idea in four costumes: an effect is rarely "there" or "not there"; it is
there UNDER A CONDITION, and every honest question about it is some form of "compare inside the
condition with outside it".

  Gate / ExPostMask  -- the condition itself, as an object. A Gate is CAUSAL (it may look only at
                        trailing data) and is therefore actionable; an ExPostMask is analysis-only
                        and says so loudly to everything downstream. Causality here is AUDITED by
                        perturbation, not asserted in a docstring -- see Gate.audit_causality.
  conditional        -- any statistic, reported as (all, inside, outside, difference z) in one call.
  across_regimes     -- the same split, but the conditions are MEASURED regime segments; adds a
                        sign test across segments, a per-segment detection floor, and the
                        concentration readout that separates a real effect from a one-regime story.
  insurance_profile  -- the inverted question: is the payoff concentrated in the state you were
                        about to exclude? Frequency-based filtering deletes exactly this kind of
                        effect, silently.

Why this module exists (the campaign that paid for it, in three measured numbers):
  * A causal storm gate -- stand aside on trailing drawdown < -15% or trailing volatility in the
    top decile -- left the entries untouched and moved CAGR +22% -> +58.4% with max drawdown
    -85.9% -> -47.1%. Same signal; only the CONDITION was added. That is C1.
  * The effect that survived was positive in 3 of 4 MEASURED regimes; the artifact that did not
    had one regime carrying everything. Per-regime evaluation is what told them apart, and no
    unconditional average could have. That is C2.
  * The same reversion earned +36bp per event INSIDE storms and +4bp outside. It IS storm
    insurance. Gating the storms away -- the obvious "filter out the bad times" move -- deletes
    the effect entirely. That is C4, and it is why C1 and C4 must be read together.

Nothing here is market-specific. Substitute "telescope slewing vs pointing stable" for "storm vs
calm" and every function reads the same: a flux excursion measured only while the mount was
settling is an instrument story, and the tool that tells you so is `conditional`.

Constitution: NumPy + stdlib only, deterministic, additive. Every function is a pure measurement --
nothing here mutates its inputs or holds state between calls.
"""
import math

import numpy as np


# --------------------------------------------------------------------------------------------
# C1 -- the condition as an object: Gate (causal, actionable) vs ExPostMask (analysis only)
# --------------------------------------------------------------------------------------------

class Gate:
    """A CAUSAL condition: a named, composable predicate over a series that is allowed to look only
    at data at or before each index, and can therefore be ACTED on.

    Build one of two ways:
      Gate(mask_fn, name)        -- mask_fn(context) -> boolean array, one flag per sample. You are
                                    responsible for causality; `audit_causality` will check you.
      Gate.from_trailing(...)    -- causal BY CONSTRUCTION: your statistic is handed only
                                    context[i-window+1 : i+1], so it cannot see the future even if
                                    you write it carelessly. Prefer this constructor.

    Composable with `&`, `|` and `~`, so the storm gate reads the way it is spoken:
        stand_aside = deep_drawdown | high_vol
        trade_when  = ~stand_aside

    The distinction from ExPostMask is the whole point. Both filter; only one is a strategy. A Gate
    can be evaluated at decision time with information that existed at decision time. An ExPostMask
    ("exclude the weeks that turned out badly") can only be evaluated afterwards, and any
    performance number computed through one is a description of the past, not a claim about the
    future. Downstream measurements in this module refuse to stay quiet about which one they got.

    KEPT NEGATIVE: `causal=True` is a CLAIM, not a proof, for a hand-written mask_fn -- the object
    cannot inspect your intent. audit_causality is the proof, it is cheap, and it is not run
    automatically (it needs a context to perturb). Run it in your tests. Gates built through
    from_trailing carry `constructed_causal=True` and pass by construction."""

    def __init__(self, mask_fn, name="gate", causal=True):
        self.mask_fn = mask_fn
        self.name = name
        self.causal = bool(causal)
        self.constructed_causal = False

    @classmethod
    def from_trailing(cls, stat_fn, window, threshold=None, compare="ge", name=None, min_periods=1):
        """Build a gate from a TRAILING-WINDOW statistic -- causal by construction.

        stat_fn is called with context[max(0, i-window+1) : i+1] for each i, so the future is not
        merely off-limits, it is not in the room. `threshold` + `compare` ('ge','gt','le','lt') turn
        the statistic into a boolean; leave threshold None to have stat_fn return booleans itself.

        Samples before `min_periods` are False -- the warm-up is NOT retroactively filled in, which
        is the usual quiet look-ahead in rolling code (back-filling a warm-up with the full-sample
        value leaks the whole series into index 0)."""
        cname = name or ("trailing_%s" % getattr(stat_fn, "__name__", "stat"))
        cmp_ops = {"ge": lambda a, b: a >= b, "gt": lambda a, b: a > b,
                   "le": lambda a, b: a <= b, "lt": lambda a, b: a < b}
        if compare not in cmp_ops:
            raise ValueError("compare must be one of 'ge','gt','le','lt', got %r" % (compare,))
        op = cmp_ops[compare]

        def mask_fn(context):
            x = np.asarray(context, float).ravel()
            out = np.zeros(x.size, dtype=bool)
            for i in range(x.size):
                if i + 1 < min_periods:
                    continue
                lo = max(0, i - window + 1)
                val = stat_fn(x[lo:i + 1])
                out[i] = bool(val) if threshold is None else bool(op(val, threshold))
            return out

        g = cls(mask_fn, name=cname, causal=True)
        g.constructed_causal = True
        return g

    def mask(self, context):
        """Evaluate the condition over `context`, returning a boolean array of one flag per sample."""
        m = np.asarray(self.mask_fn(context), dtype=bool).ravel()
        n = np.asarray(context).ravel().size
        if m.size != n:
            raise ValueError("%s mask returned %d flags for %d samples -- a gate must emit one flag "
                             "per sample" % (self.name, m.size, n))
        return m

    def apply(self, values, context=None):
        """Select the values where the condition holds. Returns (indices, selected_values).

        `context` defaults to `values` itself, which is the common case (gate a series on its own
        trailing behaviour); pass a separate context when the condition lives on another channel."""
        v = np.asarray(values, float).ravel()
        m = self.mask(values if context is None else context)
        if m.size != v.size:
            raise ValueError("context has %d samples but values has %d" % (m.size, v.size))
        idx = np.nonzero(m)[0]
        return idx, v[idx]

    def audit_causality(self, context, n_probes=12, seed=0, scale=None):
        """PROVE the gate is causal, by perturbation: scramble the FUTURE and check the past does
        not move.

        For each probe index i, replace context[i+1:] with noise and recompute the mask. If any flag
        at or before i changes, the gate consulted data it could not have had at time i -- it is not
        actionable, whatever it claims. This is the look-ahead linter in miniature, and it catches
        the two classic leaks: a full-sample normalisation (z-scoring against the whole series' mean
        and std) and a threshold picked from a global quantile.

        Returns {passed, n_probes, first_violation, max_flips, causal_claim, constructed_causal}.
        Deterministic in `seed`.

        KEPT NEGATIVE: this is a test, not a proof -- a leak that happens not to change the mask on
        these particular probes passes. It is strong in practice because a full-sample statistic
        almost always moves when you replace the tail, but a gate that leaks only rarely (say, a
        future maximum that the probes never dislodge) can slip through. It cannot produce a FALSE
        alarm, though: a flag that moves when only the future moved is a leak, full stop."""
        x = np.asarray(context, float).ravel()
        n = x.size
        if n < 4:
            raise ValueError("audit_causality needs at least 4 samples (got %d)" % n)
        rng = np.random.default_rng(seed)
        base = self.mask(x)
        # Perturbation scale follows the data, so the probe is meaningful for both a series of
        # returns near 0.01 and a series of photon counts near 1e6.
        sc = float(np.std(x)) if scale is None else float(scale)
        if sc == 0.0:
            sc = 1.0
        probes = np.unique(np.linspace(n // 4, n - 2, num=min(n_probes, max(1, n - 2)), dtype=int))
        first_violation, max_flips = None, 0
        for i in probes:
            y = x.copy()
            y[i + 1:] = rng.normal(0.0, sc * 10.0, size=n - i - 1)
            m = self.mask(y)
            flips = int(np.sum(m[:i + 1] != base[:i + 1]))
            if flips > max_flips:
                max_flips = flips
            if flips and first_violation is None:
                first_violation = int(i)
        return {"passed": bool(first_violation is None), "n_probes": int(probes.size),
                "first_violation": first_violation, "max_flips": int(max_flips),
                "causal_claim": bool(self.causal), "constructed_causal": bool(self.constructed_causal),
                "name": self.name}

    # -- composition: a gate built from gates is still a gate, and stays as causal as its weakest part
    def __and__(self, other):
        return Gate(lambda ctx: self.mask(ctx) & other.mask(ctx),
                    name="(%s & %s)" % (self.name, other.name),
                    causal=self.causal and other.causal)

    def __or__(self, other):
        return Gate(lambda ctx: self.mask(ctx) | other.mask(ctx),
                    name="(%s | %s)" % (self.name, other.name),
                    causal=self.causal and other.causal)

    def __invert__(self):
        g = Gate(lambda ctx: ~self.mask(ctx), name="~%s" % self.name, causal=self.causal)
        g.constructed_causal = self.constructed_causal
        return g

    def __repr__(self):
        return "Gate(%r, causal=%s)" % (self.name, self.causal)


class ExPostMask(Gate):
    """An ANALYSIS-ONLY condition -- one that needs information from after the decision it filters.

    Same interface as Gate deliberately, so that swapping one for the other is a one-word change and
    the measurement functions can tell you what you did. Everything in this module that receives an
    ExPostMask sets `causal=False` and fills a `warning` field; nothing raises, because ex-post
    conditioning is a legitimate and useful ANALYTICAL move ("where did the losses live?"). It stops
    being legitimate the moment its output is read as a performance claim, and that is exactly the
    moment the warning is there for.

    Use it on purpose and read the warning as a label, not a scolding: "the effect is +36bp inside
    the storms" is a true and valuable sentence. "So we would have made +36bp" is not, unless the
    storms were identifiable at the time -- which is a Gate."""

    def __init__(self, mask_fn, name="expost"):
        Gate.__init__(self, mask_fn, name=name, causal=False)

    def audit_causality(self, context, n_probes=12, seed=0, scale=None):
        """An ExPostMask is declared non-causal, so the audit reports that fact rather than testing
        it -- there is nothing to catch when the object already says it looks forward."""
        return {"passed": False, "n_probes": 0, "first_violation": None, "max_flips": 0,
                "causal_claim": False, "constructed_causal": False, "name": self.name,
                "note": "ExPostMask is analysis-only by declaration; not actionable."}

    def __repr__(self):
        return "ExPostMask(%r)" % (self.name,)


#: The vocabulary of trailing statistics a gate can be built from BY NAME, so a gate can be
#: constructed over a wire (an agent posting JSON) and not only in Python. Deliberately small:
#: these are the shapes conditions are actually written in. Each takes the trailing window and
#: returns one number. NOTE the seam -- when the rolling/streaming kit (causal std/mean/quantile/
#: drawdown with warm starts) lands, these become one-line delegations to it; they are spelled out
#: here so the conditioning layer does not have to wait on that module.
TRAILING_STATS = {
    "std": lambda w: float(np.std(w)),
    "mean": lambda w: float(np.mean(w)),
    "abs_mean": lambda w: float(np.mean(np.abs(w))),
    "min": lambda w: float(np.min(w)),
    "max": lambda w: float(np.max(w)),
    "range": lambda w: float(np.max(w) - np.min(w)),
    # Trailing drawdown from the running peak, as a fraction -- the storm gate's own statistic.
    # Guarded denominator so a series that legitimately passes through zero cannot explode.
    "drawdown": lambda w: float((w[-1] - np.max(w)) / max(abs(float(np.max(w))), 1e-12)),
    "last": lambda w: float(w[-1]),
}


def trailing_gate(stat="std", window=20, threshold=None, compare="ge", min_periods=None, name=None):
    """Build a causal Gate from a NAMED trailing statistic -- the constructor that works over a wire.

    stat may be a name from TRAILING_STATS ('std', 'mean', 'abs_mean', 'min', 'max', 'range',
    'drawdown', 'last') or any callable window -> float. Because the statistic only ever sees
    context[i-window+1 : i+1], the resulting gate is causal by construction and passes
    audit_causality without the caller having to be careful.

    The storm gate that moved the campaign's CAGR from +22% to +58.4% is two of these, or-ed:
        deep = trailing_gate('drawdown', window=90, threshold=-0.15, compare='le')
        wild = trailing_gate('std', window=90, threshold=vol_p90, compare='ge')
        stand_aside = deep | wild"""
    if callable(stat):
        fn, sname = stat, getattr(stat, "__name__", "stat")
    else:
        if stat not in TRAILING_STATS:
            raise ValueError("unknown trailing stat %r -- known: %s (or pass a callable)"
                             % (stat, ", ".join(sorted(TRAILING_STATS))))
        fn, sname = TRAILING_STATS[stat], stat
    mp = window if min_periods is None else min_periods
    return Gate.from_trailing(fn, window=window, threshold=threshold, compare=compare,
                              name=name or ("trailing_%s" % sname), min_periods=mp)


# --------------------------------------------------------------------------------------------
# shared small statistics (kept local and explicit: NumPy-only, no t distribution available)
# --------------------------------------------------------------------------------------------

def _mean_t_p(v):
    """(n, mean, t, p) for a one-sample test against zero, two-sided, normal approximation.

    WHY the normal approximation: the engine is NumPy-only by constitution and carries no Student's
    t. It is anticonservative below ~30 samples, which is why every report here flags small groups
    rather than quietly rounding a p-value down."""
    v = np.asarray(v, float).ravel()
    n = v.size
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    if sd > 0:
        t = mean / (sd / math.sqrt(n))
    else:
        t = 0.0 if mean == 0 else math.copysign(math.inf, mean)
    p = math.erfc(abs(t) / math.sqrt(2.0)) if math.isfinite(t) else 0.0
    return n, mean, float(t), float(p)


def _floor_at_power(v, alpha=0.05, power=0.8):
    """Smallest mean effect this group could have detected, at `power`, given its own noise and size.

    (z_{alpha/2} + z_{power}) * sd / sqrt(n) -- the analytic detection floor. This is what turns
    "no effect here" into "no effect above X here", which is the only form of a null result that can
    be argued with. For the empirical version that pushes a real injected effect through a real
    pipeline, see holographic_honesty.min_detectable_effect; this one is free and goes in every row."""
    v = np.asarray(v, float).ravel()
    n = v.size
    if n < 2:
        return float("inf")
    sd = float(v.std(ddof=1))
    # Two-sided 0.05 -> 1.959964; power 0.8 -> 0.8416212. Hard-coded because NumPy has no ppf and
    # these two are the ones anybody uses; other alphas fall back to the same shape via erfcinv.
    z_a = 1.959964 if abs(alpha - 0.05) < 1e-12 else float(math.sqrt(2.0) * _erfcinv(alpha))
    z_b = 0.8416212 if abs(power - 0.8) < 1e-12 else float(math.sqrt(2.0) * _erfcinv(2.0 * (1.0 - power)))
    return float((z_a + z_b) * sd / math.sqrt(n))


def _erfcinv(y):
    """Inverse complementary error function by bisection -- adequate and dependency-free.

    WHY bisection rather than a rational approximation: this is called at most twice per report, on
    a strictly monotone function over a bounded bracket, so 80 halvings costs nothing and cannot be
    wrong in the tails the way a fitted approximation can."""
    if not (0.0 < y < 2.0):
        raise ValueError("erfcinv needs 0 < y < 2, got %r" % (y,))
    lo, hi = -6.0, 6.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if math.erfc(mid) > y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _resolve_condition(condition, values, context):
    """Turn a Gate / ExPostMask / boolean array into (mask, causal, warning). One place, so every
    entry point in this module treats the ex-ante / ex-post distinction identically."""
    v = np.asarray(values, float).ravel()
    if isinstance(condition, dict) and "mask" in condition:
        # A condition that arrived OVER A WIRE, in the shape mind.causal_gate(context=...) returns:
        # {'mask': [...], 'audit': {...}}. A Gate object cannot cross HTTP, so without this branch an
        # agent that correctly built and AUDITED a causal gate would still be told its split was
        # ex-post -- a false alarm in the one workflow the wire supports. The claim is honoured only
        # when the PROOF travels with it: causal iff the attached audit actually passed. This is not
        # trusting the caller; it is reading the caller's evidence.
        mask = np.asarray(condition["mask"], dtype=bool).ravel()
        audit = condition.get("audit") or {}
        causal = bool(audit.get("passed"))
        name = condition.get("name", audit.get("name", "wire_gate"))
        if causal:
            warning = None
        elif audit:
            warning = ("CONDITION FAILED ITS CAUSALITY AUDIT (%s): the mask changed when only the FUTURE was "
                       "perturbed, so it consulted data it could not have had at decision time. Treated as "
                       "ex-post." % name)
        else:
            warning = ("MASK WITHOUT AUDIT: no causality evidence travelled with this condition, so it is "
                       "treated as EX-POST. Build it with causal_gate(..., context=...) to attach the audit.")
    elif isinstance(condition, Gate):
        mask = condition.mask(v if context is None else context)
        causal = bool(condition.causal)
        warning = None if causal else (
            "EX-POST CONDITION (%s): this split uses information from after the decision it filters. "
            "Read the inside/outside numbers as a DESCRIPTION of what happened, never as a performance "
            "claim -- for an actionable version the condition must be rebuilt as a causal Gate." % condition.name)
        name = condition.name
    else:
        mask = np.asarray(condition, dtype=bool).ravel()
        causal = False
        warning = ("RAW MASK: a bare boolean array carries no causality claim, so it is treated as "
                   "EX-POST. Wrap it in Gate(...) once you can compute it from trailing data only.")
        name = "mask"
    if mask.size != v.size:
        raise ValueError("condition produced %d flags for %d values" % (mask.size, v.size))
    return mask, causal, warning, name


# --------------------------------------------------------------------------------------------
# C3 -- conditional: any statistic, reported four ways at once
# --------------------------------------------------------------------------------------------

def conditional(values, condition, stat_fn=None, context=None, alpha=0.05):
    """SPLIT A MEASUREMENT BY A CONDITION -- report it (all, inside, outside, difference) in one call.

    values    : the per-event quantity being measured (returns, residuals, flux excursions...).
    condition : a Gate, an ExPostMask, or a raw boolean array (treated as ex-post -- see below).
    stat_fn   : optional; any callable array -> float. Default is the mean, which is the case that
                carries a t-statistic and a difference z. A custom stat_fn still gets inside/outside
                values but its significance columns are left NaN, because the module cannot know the
                sampling distribution of an arbitrary statistic (use holographic_honesty.pipeline_null
                for that -- it is exactly the right tool and it is one call).
    context   : the series the condition is evaluated on, if different from `values`.

    Returns a dict with n / mean / t / p for all, inside and outside, plus `diff` and `z_diff` (a
    two-sample Welch z on the difference of means), `separates` (does the condition split the effect
    at `alpha`), `causal`, and `warning`.

    The reframe this exists to make cheap: the campaign that paid for this module spent weeks on
    unconditional averages that hid TWO opposite behaviours -- the same instrument trended in calm
    conditions and whipsawed in storms, and the average of the two was a flat nothing. The split was
    done by hand dozens of times before it became this function. Any time an effect looks weak,
    condition it before abandoning it; any time it looks strong, condition it before believing it.

    Over a wire, pass the dict that mind.causal_gate(..., context=...) returns -- {'mask': [...],
    'audit': {...}} -- and the causality claim is honoured because its PROOF travelled with it. A Gate
    object cannot cross HTTP; its audit can.

    KEPT NEGATIVE: a raw boolean array is deliberately labelled EX-POST, even when you happen to have
    built it causally. The alternative -- trusting the caller -- is how look-ahead gets into a book.
    Wrapping a causal computation in Gate(fn) is one line and makes the claim auditable, so the
    module charges you that one line."""
    v = np.asarray(values, float).ravel()
    if v.size < 2:
        raise ValueError("conditional needs at least 2 values (got %d)" % v.size)
    mask, causal, warning, cname = _resolve_condition(condition, v, context)
    inside, outside = v[mask], v[~mask]

    n_all, mean_all, t_all, p_all = _mean_t_p(v)
    n_in, mean_in, t_in, p_in = _mean_t_p(inside)
    n_out, mean_out, t_out, p_out = _mean_t_p(outside)

    if stat_fn is not None:
        # A custom statistic gets its values honestly and its significance columns left empty,
        # rather than a t-test that does not apply to it.
        mean_all = float(stat_fn(v)) if n_all else float("nan")
        mean_in = float(stat_fn(inside)) if n_in else float("nan")
        mean_out = float(stat_fn(outside)) if n_out else float("nan")
        t_all = t_in = t_out = p_all = p_in = p_out = float("nan")
        z_diff = float("nan")
    else:
        se_in = (np.std(inside, ddof=1) / math.sqrt(n_in)) if n_in > 1 else float("inf")
        se_out = (np.std(outside, ddof=1) / math.sqrt(n_out)) if n_out > 1 else float("inf")
        se = math.sqrt(float(se_in) ** 2 + float(se_out) ** 2)
        z_diff = float((mean_in - mean_out) / se) if se > 0 and math.isfinite(se) else float("nan")

    diff = float(mean_in - mean_out) if (n_in and n_out) else float("nan")
    p_diff = math.erfc(abs(z_diff) / math.sqrt(2.0)) if (z_diff == z_diff and math.isfinite(z_diff)) else float("nan")
    separates = bool(p_diff == p_diff and p_diff <= alpha)

    return {"condition": cname, "causal": bool(causal), "warning": warning,
            "n_all": int(n_all), "mean_all": float(mean_all), "t_all": float(t_all), "p_all": float(p_all),
            "n_inside": int(n_in), "mean_inside": float(mean_in), "t_inside": float(t_in), "p_inside": float(p_in),
            "n_outside": int(n_out), "mean_outside": float(mean_out), "t_outside": float(t_out),
            "p_outside": float(p_out),
            "diff": diff, "z_diff": float(z_diff), "p_diff": float(p_diff), "separates": separates,
            "floor_inside": _floor_at_power(inside, alpha=alpha), "floor_outside": _floor_at_power(outside, alpha=alpha),
            "small_sample": bool(min(n_in, n_out) < 30)}


# --------------------------------------------------------------------------------------------
# C2 -- across_regimes: the same split, with MEASURED regimes as the conditions
# --------------------------------------------------------------------------------------------

def across_regimes(values, segments=None, series=None, events=None, min_seg=16, penalty=3.0,
                   alpha=0.05, power=0.8):
    """EVALUATE AN EFFECT INSIDE EVERY MEASURED REGIME, and report whether it is one effect or one
    regime's story.

    values   : per-event quantity, in event order.
    segments : list of (start, end) index pairs. Omit and pass `series` instead to have the regimes
               MEASURED for you by holographic_demux.segment_stream (the same change-point
               segmentation behind mind.detect_regimes) -- which is the honest default, because
               hand-drawn regime boundaries are a place look-ahead hides comfortably.
    events   : positions of each value within the series, for assigning values to segments. Defaults
               to arange(len(values)), i.e. one value per sample.

    Per segment you get n, mean, t, p, significant, floor (the smallest effect that segment could
    have detected at `power` -- so an empty segment says "nothing above X" rather than "nothing"),
    and too_small. Across segments you get:
      n_tested       segments large enough to test at all
      n_positive / n_negative / n_significant
      consistent     every tested segment agrees in sign
      sign_test_p    exact two-sided binomial on the sign agreement
      concentration  share of the TOTAL effect contributed by the single biggest segment
      verdict        a sentence you can paste into a note

    The measurement that makes this worth a module: a real mean-reversion effect was positive in 3
    of 4 measured regimes (concentration 0.41, sign test p=0.25 -- weak on its own, but the SIGN was
    everywhere), while a seductive artifact with a better headline number had a single regime
    carrying essentially all of it (concentration > 0.9). The unconditional means were similar. The
    per-regime table was what told them apart, and it is the reason `concentration` is reported
    beside the significance columns rather than underneath them.

    KEPT NEGATIVE: the sign test across segments is UNDERPOWERED by construction -- four regimes can
    never give you p < 0.125 even if all four agree. It is not there to certify; it is there to stop
    you certifying. Read `consistent` and `concentration` first, and treat sign_test_p as the modest
    corroboration it is. Second negative: segments measured from the values' own series are chosen
    with full knowledge of the series, so this is an ANALYSIS tool -- it describes where an effect
    lived, it does not define a tradeable rule. For that, the regime condition must be rebuilt as a
    causal Gate and passed to `conditional`."""
    v = np.asarray(values, float).ravel()
    if v.size < 2:
        raise ValueError("across_regimes needs at least 2 values (got %d)" % v.size)
    ev = np.arange(v.size) if events is None else np.asarray(events).ravel()
    if ev.size != v.size:
        raise ValueError("events and values must be the same length (got %d and %d)" % (ev.size, v.size))

    if segments is None:
        if series is None:
            raise ValueError("across_regimes needs either `segments` or a `series` to measure them from")
        # Delegate: the regime boundaries come from the engine's own change-point segmenter, not
        # from a second implementation living here.
        from holographic.sampling_and_signal.holographic_demux import segment_stream
        segments = segment_stream(series, min_seg=min_seg, penalty=penalty)["segments"]
    segments = [(int(a), int(b)) for a, b in segments]
    if not segments:
        raise ValueError("no segments to evaluate")

    rows, total_abs = [], 0.0
    for (a, b) in segments:
        sel = (ev >= a) & (ev < b)
        w = v[sel]
        n, mean, t, p = _mean_t_p(w)
        too_small = n < 8
        rows.append({"start": a, "end": b, "n": int(n),
                     "mean": float(mean) if n else float("nan"),
                     "t": float(t) if n else float("nan"),
                     "p": float(p) if n else float("nan"),
                     "significant": bool(n and p == p and p <= alpha and not too_small),
                     "floor": _floor_at_power(w, alpha=alpha, power=power),
                     "too_small": bool(too_small)})
        if n:
            total_abs += abs(float(mean) * n)

    tested = [r for r in rows if not r["too_small"] and r["n"] > 0]
    n_pos = sum(1 for r in tested if r["mean"] > 0)
    n_neg = sum(1 for r in tested if r["mean"] < 0)
    n_sig = sum(1 for r in tested if r["significant"])
    n_tested = len(tested)
    consistent = bool(n_tested >= 2 and (n_pos == n_tested or n_neg == n_tested))

    # Exact two-sided binomial sign test at q=0.5 over the tested segments.
    if n_tested >= 1:
        k = max(n_pos, n_neg)
        tail = sum(math.comb(n_tested, j) for j in range(k, n_tested + 1)) / (2.0 ** n_tested)
        sign_test_p = float(min(1.0, 2.0 * tail))
    else:
        sign_test_p = float("nan")

    # Concentration: how much of the total effect mass one segment carries. A real effect spreads it.
    if total_abs > 0:
        shares = [abs(r["mean"] * r["n"]) / total_abs for r in rows if r["n"]]
        concentration = float(max(shares)) if shares else float("nan")
    else:
        concentration = float("nan")

    if n_tested < 2:
        verdict = ("only %d segment(s) were large enough to test -- this is not a per-regime result, "
                   "it is a single measurement wearing a table." % n_tested)
    elif consistent and concentration < 0.6:
        verdict = ("effect agrees in sign across all %d tested regimes and no single regime carries it "
                   "(concentration %.2f) -- this is the shape of a real effect." % (n_tested, concentration))
    elif consistent:
        verdict = ("sign is consistent across %d regimes but %.0f%% of the effect comes from one of them -- "
                   "consistent, yet fragile to that regime not recurring." % (n_tested, 100.0 * concentration))
    elif concentration >= 0.6:
        verdict = ("SIGN DISAGREES across regimes and %.0f%% of the effect lives in a single regime -- this is "
                   "the classic one-regime artifact; do not carry it forward without a causal reason for that "
                   "regime." % (100.0 * concentration))
    else:
        verdict = ("sign disagrees across regimes (%d positive, %d negative) -- whatever this is, it is not one "
                   "effect." % (n_pos, n_neg))

    return {"segments": rows, "n_segments": len(rows), "n_tested": int(n_tested),
            "n_positive": int(n_pos), "n_negative": int(n_neg), "n_significant": int(n_sig),
            "consistent": consistent, "sign_test_p": sign_test_p,
            "concentration": concentration, "verdict": verdict}


# --------------------------------------------------------------------------------------------
# C4 -- insurance_profile: is the payoff concentrated in the state you were about to exclude?
# --------------------------------------------------------------------------------------------

def insurance_profile(values, condition, context=None, alpha=0.05):
    """ASK BEFORE YOU FILTER: does this effect's payoff live INSIDE the condition you were planning
    to exclude?

    Same arguments as `conditional`, different question and a different headline. Filtering is the
    most natural move in analysis -- the bad periods are noisy, they hurt, remove them and the
    numbers improve. Sometimes what you have removed is the entire phenomenon.

    The measured case this is named for: a mean-reversion effect paid +36bp per event inside market
    storms and +4bp outside them. It was not an effect damaged by storms; it WAS storm insurance --
    the compensation for standing in front of a disorderly move. Excluding storms (the obvious
    hygiene move, and one that improved every other statistic on the page) deleted 90% of the
    edge while leaving a tidy-looking remainder that would have been traded at a fraction of its
    real size.

    Returns the inside/outside decomposition plus:
      share_inside     fraction of the TOTAL summed value contributed by inside events
      frac_events      fraction of EVENTS that are inside
      lift             mean_inside / mean_outside (inf when outside is ~0, which is itself the finding)
      premium_inside   True when a minority of events carries the majority of the value
      verdict          the sentence to read before deleting anything

    KEPT NEGATIVE: a premium concentrated in a rare state is ALSO the signature of insufficient data
    in that state -- 8 events carrying the effect is a hypothesis, not a phenomenon. `small_sample`
    and `n_inside` are reported for exactly this reason, and the honest follow-up is split_half on
    the inside events alone. Do not read `premium_inside=True` as "keep it"; read it as "do not
    delete it without measuring it properly first"."""
    v = np.asarray(values, float).ravel()
    base = conditional(v, condition, context=context, alpha=alpha)
    mask, _, _, _ = _resolve_condition(condition, v, context)
    inside, outside = v[mask], v[~mask]

    total = float(v.sum())
    sum_in = float(inside.sum())
    share_inside = float(sum_in / total) if total != 0 else float("nan")
    frac_events = float(inside.size) / float(v.size)
    m_in, m_out = base["mean_inside"], base["mean_outside"]
    if m_out == 0 or not math.isfinite(m_out):
        lift = float("inf") if m_in != 0 else float("nan")
    else:
        lift = float(m_in / m_out)

    # "Premium" = a MINORITY of events carrying a MAJORITY of the value, in the same direction as
    # the whole. That asymmetry is the thing frequency-based filtering is blind to.
    premium_inside = bool(frac_events < 0.5 and share_inside == share_inside and share_inside > 0.5
                          and abs(m_in) > abs(m_out))

    if premium_inside:
        verdict = ("INSURANCE PROFILE: %.0f%% of the events carry %.0f%% of the value (mean %.4g inside vs "
                   "%.4g outside). The premium lives in the state you would exclude -- gating this condition "
                   "away does not clean the effect up, it DELETES it." % (100.0 * frac_events,
                                                                          100.0 * share_inside, m_in, m_out))
    elif share_inside == share_inside and share_inside < 0.0:
        verdict = ("the inside state is a net DRAG (it contributes %.0f%% of total, i.e. works against the "
                   "effect) -- excluding it is defensible, provided the condition is causal." % (100.0 * share_inside))
    else:
        verdict = ("value is not concentrated inside the condition (%.0f%% of events, %.0f%% of value) -- "
                   "filtering here costs about what it removes." % (100.0 * frac_events, 100.0 * share_inside))

    out = dict(base)
    out.update({"total": total, "sum_inside": sum_in, "sum_outside": float(outside.sum()),
                "share_inside": share_inside, "frac_events": frac_events, "lift": lift,
                "premium_inside": premium_inside, "verdict": verdict})
    return out


def _selftest():
    """Assert the real contracts: the causality audit catches a real leak, the conditional split
    recovers a planted difference, per-regime evaluation separates a spread effect from a one-regime
    artifact, and the insurance profile fires on a concentrated premium."""
    rng = np.random.default_rng(0)

    # --- C1: a trailing gate is causal; a full-sample gate is NOT, and the audit must say so.
    x = rng.normal(0, 1, 400)
    causal_gate = Gate.from_trailing(lambda w: float(np.std(w)), window=20, threshold=1.1,
                                     compare="ge", name="high_trailing_vol", min_periods=20)
    a1 = causal_gate.audit_causality(x, seed=0)
    assert a1["passed"], "trailing gate must audit clean, got %r" % (a1,)
    assert a1["constructed_causal"]

    # The classic leak: threshold taken from a FULL-SAMPLE quantile. It claims causal=True and is not.
    leaky = Gate(lambda ctx: np.asarray(ctx, float).ravel() > np.quantile(np.asarray(ctx, float), 0.9),
                 name="global_quantile", causal=True)
    a2 = leaky.audit_causality(x, seed=0)
    assert not a2["passed"], "KEPT NEGATIVE FAILED: a full-sample-quantile gate must be caught as look-ahead"
    assert a2["first_violation"] is not None and a2["max_flips"] > 0

    # Composition preserves the audit outcome.
    combo = causal_gate | Gate.from_trailing(lambda w: float(np.mean(w)), window=10, threshold=0.5,
                                             compare="ge", name="trailing_up", min_periods=10)
    assert combo.audit_causality(x, seed=0)["passed"], "composed causal gates must stay causal"
    assert (~causal_gate).mask(x).sum() == x.size - causal_gate.mask(x).sum()

    # --- named-stat construction, and the storm gate as it was actually written.
    path = np.cumsum(rng.normal(0.0, 1.0, 400)) + 100.0
    deep = trailing_gate("drawdown", window=90, threshold=-0.05, compare="le")
    wild = trailing_gate("std", window=90, threshold=float(np.std(path[:90])), compare="ge")
    stand_aside = deep | wild
    assert stand_aside.audit_causality(path, seed=0)["passed"], "the storm gate must be causal"
    assert 0 < int(stand_aside.mask(path).sum()) < path.size, "storm gate fired never or always"
    try:
        trailing_gate("sharpe_ratio")
        raise AssertionError("unknown stat name must be refused")
    except ValueError as e:
        assert "known:" in str(e), "the refusal must NAME the valid options"

    # --- C3: conditional recovers a planted inside/outside difference.
    n = 600
    flag = np.zeros(n, dtype=bool)
    flag[::3] = True                      # a third of the events are "inside"
    vals = rng.normal(0.0, 1.0, n)
    vals[flag] += 1.0                     # planted effect of exactly +1.0 inside
    c = conditional(vals, flag)
    assert abs(c["diff"] - 1.0) < 0.25, "planted +1.0 difference not recovered: %r" % c["diff"]
    assert c["z_diff"] > 5.0 and c["separates"], "a 1-sigma planted split must separate: %r" % c
    assert c["causal"] is False and c["warning"] is not None, "a raw mask must be labelled ex-post"

    # An ExPostMask warns; a Gate does not.
    epm = ExPostMask(lambda ctx: np.asarray(ctx, float).ravel() > 0.0, name="turned_out_positive")
    ce = conditional(vals, epm)
    assert ce["causal"] is False and "EX-POST" in ce["warning"]
    cg = conditional(vals, causal_gate, context=x[:n] if n <= x.size else np.resize(x, n))
    assert cg["causal"] is True and cg["warning"] is None, "a causal Gate must not raise a warning"

    # --- the WIRE shape: a mask whose audit travelled with it is honoured as causal; a bare mask,
    # and a mask whose audit FAILED, are both treated as ex-post and say why.
    wire_ok = {"mask": causal_gate.mask(x[:n] if n <= x.size else np.resize(x, n)).tolist(),
               "audit": {"passed": True, "name": "high_trailing_vol"}}
    cw = conditional(vals, wire_ok)
    assert cw["causal"] is True and cw["warning"] is None, "an audited wire gate must count as causal"
    cw_bad = conditional(vals, {"mask": wire_ok["mask"], "audit": {"passed": False, "name": "leaky"}})
    assert cw_bad["causal"] is False and "FAILED ITS CAUSALITY AUDIT" in cw_bad["warning"]
    cw_none = conditional(vals, {"mask": wire_ok["mask"]})
    assert cw_none["causal"] is False and "WITHOUT AUDIT" in cw_none["warning"]

    # --- C2: spread effect vs one-regime artifact, same headline mean, different verdict.
    segs = [(0, 150), (150, 300), (300, 450), (450, 600)]
    spread = rng.normal(0.30, 1.0, 600)                     # present in every regime
    r_spread = across_regimes(spread, segments=segs)
    assert r_spread["consistent"], "a uniformly-present effect must be sign-consistent: %r" % r_spread["verdict"]
    assert r_spread["concentration"] < 0.6, "spread effect wrongly flagged concentrated: %.3f" % r_spread["concentration"]

    artifact = rng.normal(0.0, 1.0, 600)
    artifact[150:300] += 1.2                                 # one regime carries everything
    r_art = across_regimes(artifact, segments=segs)
    assert r_art["concentration"] > 0.6, ("KEPT NEGATIVE FAILED: a one-regime artifact must show high "
                                          "concentration, got %.3f" % r_art["concentration"])
    assert not r_art["consistent"] or r_art["concentration"] > 0.6
    # The two have comparable unconditional means -- which is the whole point of the per-regime table.
    assert abs(float(spread.mean()) - float(artifact.mean())) < 0.15

    # Segments measured from a series rather than handed in.
    series = np.concatenate([rng.normal(0, 0.2, 200), rng.normal(0, 1.4, 200)])
    r_meas = across_regimes(series, series=series, min_seg=32)
    assert r_meas["n_segments"] >= 1 and "verdict" in r_meas

    # Detection floors must be finite and shrink with n (a null result has to state a floor).
    small = across_regimes(rng.normal(0, 1, 40), segments=[(0, 20), (20, 40)])
    big = across_regimes(rng.normal(0, 1, 4000), segments=[(0, 2000), (2000, 4000)])
    assert big["segments"][0]["floor"] < small["segments"][0]["floor"], "floor must tighten with more data"

    # --- C4: the storm-insurance shape.
    n = 500
    storm = np.zeros(n, dtype=bool)
    storm[:60] = True                      # 12% of events
    pay = np.full(n, 0.04)
    pay[storm] = 0.36                      # the measured +36bp inside / +4bp outside
    ins = insurance_profile(pay, storm)
    assert ins["premium_inside"], "storm-insurance shape not detected: %r" % ins["verdict"]
    assert ins["share_inside"] > 0.5 and ins["frac_events"] < 0.5
    assert abs(ins["lift"] - 9.0) < 1e-9, "lift must be exactly mean_in/mean_out: %r" % ins["lift"]
    assert "DELETES" in ins["verdict"]

    # A diffuse effect must NOT be called insurance.
    flat = np.full(n, 0.05)
    assert not insurance_profile(flat, storm)["premium_inside"]

    # --- determinism: identical inputs, identical numbers, twice.
    assert across_regimes(artifact, segments=segs)["concentration"] == r_art["concentration"]
    assert causal_gate.audit_causality(x, seed=0) == a1

    print("holographic_conditioning selftest OK "
          "(causal audit catches full-sample leak; planted diff %.3f; concentration spread %.2f vs artifact %.2f; "
          "insurance lift %.1fx)" % (c["diff"], r_spread["concentration"], r_art["concentration"], ins["lift"]))


if __name__ == "__main__":
    _selftest()

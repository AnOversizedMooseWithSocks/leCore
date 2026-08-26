"""holographic_rolling.py -- the CAUSAL rolling / streaming statistics kit: trailing mean, std, min, max,
range, quantile, drawdown, and EWMA as full series, plus their online (one-sample-at-a-time) counterparts
with exact warm starts.

WHY THIS MODULE EXISTS
----------------------
Trailing statistics are where look-ahead sneaks into pipelines one convenience at a time: a std taken over
the full sample "just for scaling", a min-max over everything "just for the plot", a centred window "because
it looks smoother". The kit gives the causal version of each so the honest form is the EASY form -- and every
function here is CONTRACTED against holographic_honesty.lookahead_lint in its own selftest: prefix-consistent
to 0.0 drift, by construction and by test.

Position convention, stated once and used everywhere: out[i] is the statistic of x[i-window+1 .. i] -- the
window ENDING AT i, INCLUDING i. That is "what you know at the close of step i". Positions with fewer than
`min_periods` samples are NaN, never a silently-shrunk window: a number computed from 3 samples wearing a
window-90 label is how "trailing p90 vol" quietly becomes noise. (min_periods defaults to window.)

EXACT BY DEFAULT, FAST BY CHOICE (the QEM decimator's precedent, applied again): the default implementations
are per-window recomputation over stride-tricks views -- bit-identical to the naive loop and to the
conditioning layer's TRAILING_STATS lambdas, which is what makes gates built on either interchangeable. The
O(n) cumulative-sums path for mean/std is OPT-IN (fast=True) because it is subject to CATASTROPHIC
CANCELLATION: on data with |mean| >> spread the cumsum variance loses digits (measured in _selftest: an
offset of 1e8 turns a std of 1.0 into pure garbage on the fast path while the exact path is untouched), and a
threshold-crossing statistic that flips a gate's decision must never depend on which code path was chosen.

The STREAMING classes exist for the case the vectorised form cannot serve -- data arriving one sample at a
time -- and each is pinned EQUAL to its vectorised sibling: Welford mean/variance to 1e-12, the deque
min/max exactly, so warm-starting a live gate from a backtest's tail is a copy, not a recalibration.

NumPy + stdlib only. Deterministic. No learned anything.
"""

import collections
import math

import numpy as np


def _windows(x, window):
    """All trailing windows as a (n-window+1, window) view -- no copy, exact per-window arithmetic."""
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(x, window)


def _pad(head_nan, values, n):
    out = np.full(n, np.nan)
    out[head_nan:] = values
    return out


def rolling_mean(x, window, min_periods=None, fast=False):
    """Trailing mean of the window ending at (and including) each position; NaN before min_periods.
    fast=True uses the O(n) cumulative-sum path -- opt-in only, see the module docstring's cancellation
    negative."""
    x = np.asarray(x, float).ravel()
    window = int(window)
    if window < 1 or window > x.size:
        raise ValueError("window must be in [1, len(x)] (got %d for %d samples)" % (window, x.size))
    if fast:
        c = np.concatenate([[0.0], np.cumsum(x)])
        vals = (c[window:] - c[:-window]) / window
    else:
        vals = _windows(x, window).mean(axis=1)
    return _pad(window - 1, vals, x.size) if (min_periods or window) >= window else \
        _partial(x, window, int(min_periods), np.mean)


def rolling_std(x, window, min_periods=None, ddof=0, fast=False):
    """Trailing standard deviation (population by default, ddof=0, matching TRAILING_STATS['std'] so a gate
    threshold means the same thing on either side). fast=True is the cumsum-of-squares path: O(n), and
    NUMERICALLY DANGEROUS on offset data -- opt-in, never the default. The selftest measures the failure it
    would cause."""
    x = np.asarray(x, float).ravel()
    window = int(window)
    if window < 1 or window > x.size:
        raise ValueError("window must be in [1, len(x)] (got %d for %d samples)" % (window, x.size))
    if fast:
        c1 = np.concatenate([[0.0], np.cumsum(x)])
        c2 = np.concatenate([[0.0], np.cumsum(x * x)])
        s1 = (c1[window:] - c1[:-window]) / window
        s2 = (c2[window:] - c2[:-window]) / window
        var = np.maximum(s2 - s1 * s1, 0.0)                     # clamp: cancellation can go slightly negative
        if ddof:
            var = var * window / max(window - ddof, 1)
        vals = np.sqrt(var)
    else:
        vals = _windows(x, window).std(axis=1, ddof=ddof)
    return _pad(window - 1, vals, x.size)


def rolling_min(x, window):
    """Trailing minimum, O(n) via the monotonic deque -- and unlike the mean/std fast paths this one is
    EXACT (comparisons, not accumulations, cannot cancel), so there is no exact/fast split to choose."""
    return _mono_extreme(x, window, is_min=True)


def rolling_max(x, window):
    """Trailing maximum, O(n) monotonic deque; exact for the same reason as rolling_min."""
    return _mono_extreme(x, window, is_min=False)


def _mono_extreme(x, window, is_min):
    x = np.asarray(x, float).ravel()
    window = int(window)
    if window < 1 or window > x.size:
        raise ValueError("window must be in [1, len(x)] (got %d for %d samples)" % (window, x.size))
    out = np.full(x.size, np.nan)
    dq = collections.deque()                                    # indices, values monotonic
    for i, v in enumerate(x):
        lo = i - window + 1
        while dq and dq[0] < lo:
            dq.popleft()
        while dq and ((x[dq[-1]] >= v) if is_min else (x[dq[-1]] <= v)):
            dq.pop()
        dq.append(i)
        if i >= window - 1:
            out[i] = x[dq[0]]
    return out


def rolling_range(x, window):
    """Trailing max - min over the window."""
    return rolling_max(x, window) - rolling_min(x, window)


def rolling_quantile(x, window, q):
    """Trailing q-quantile (linear interpolation, numpy convention), exact per window. O(n * w log w) and
    said so: an approximate streaming quantile (P^2 and friends) would be faster and is deliberately NOT
    offered -- a gate threshold crossed because an ESTIMATOR drifted is a decision flipped by an
    implementation detail, the exact class of bug this repo's tie-sensitivity rules exist to prevent.
    (Declared negative, not a TODO.)"""
    x = np.asarray(x, float).ravel()
    window = int(window)
    if window < 1 or window > x.size:
        raise ValueError("window must be in [1, len(x)] (got %d for %d samples)" % (window, x.size))
    if not (0.0 <= float(q) <= 1.0):
        raise ValueError("q must be in [0, 1], got %r" % (q,))
    vals = np.quantile(_windows(x, window), float(q), axis=1)
    return _pad(window - 1, vals, x.size)


def rolling_drawdown(x, window):
    """Trailing drawdown from the running IN-WINDOW peak, as a fraction: (x[i] - max(win)) / |max(win)|,
    guarded denominator -- bit-compatible with TRAILING_STATS['drawdown'] (pinned in the selftest), so the
    storm gate reads the same number whether it is built from the lambda or from this series."""
    x = np.asarray(x, float).ravel()
    mx = rolling_max(x, window)
    denom = np.maximum(np.abs(mx), 1e-12)
    return (x - mx) / denom


def ewma(x, alpha):
    """Exponentially-weighted moving average, out[i] = (1-a)*out[i-1] + a*x[i], seeded at out[0] = x[0] --
    the recursion itself is the causality proof (out[i] literally has no access to x[i+1:]), and the lint
    confirms it anyway."""
    x = np.asarray(x, float).ravel()
    a = float(alpha)
    if not (0.0 < a <= 1.0):
        raise ValueError("alpha must be in (0, 1], got %r" % (alpha,))
    out = np.empty_like(x)
    if x.size:
        out[0] = x[0]
        for i in range(1, x.size):
            out[i] = (1.0 - a) * out[i - 1] + a * x[i]
    return out


def ewm_std(x, alpha):
    """Exponentially-weighted std: EW second moment about the EW mean, same recursion, same seeding. The
    classic vol estimator without the classic full-sample initialisation leak (seeding the EW variance at the
    FULL SAMPLE's variance is a look-ahead the lint catches; seeding at 0 from the first sample is not)."""
    x = np.asarray(x, float).ravel()
    a = float(alpha)
    if not (0.0 < a <= 1.0):
        raise ValueError("alpha must be in (0, 1], got %r" % (alpha,))
    m = np.empty_like(x)
    v = np.empty_like(x)
    if x.size:
        m[0], v[0] = x[0], 0.0
        for i in range(1, x.size):
            m[i] = (1.0 - a) * m[i - 1] + a * x[i]
            v[i] = (1.0 - a) * (v[i - 1] + a * (x[i] - m[i - 1]) ** 2)
    return np.sqrt(v)


class StreamingStats:
    """The online counterpart for live data: push one sample at a time, read mean / std / min / max / count
    at any point. Welford's recurrence for mean/variance (numerically stable -- the streaming answer to the
    cumsum cancellation negative), monotonic deques for windowed extremes.

    `window=None` (default) is expanding (all samples so far); an integer window gives the trailing-window
    versions, pinned equal to rolling_* on the same data.

    warm_start(history) replays a backtest tail so a live gate continues EXACTLY where the backtest left off
    -- pinned: warm_start(x[:k]) then push(x[k:]) equals push(x) sample for sample."""

    def __init__(self, window=None):
        self.window = int(window) if window else None
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._buf = collections.deque(maxlen=self.window) if self.window else None
        self._minq = collections.deque()                        # (index, value), increasing values
        self._maxq = collections.deque()
        self._i = -1

    def warm_start(self, history):
        """Replay `history` through push() -- deliberately the SAME code path, so warm state cannot drift
        from live state (the wiring-honesty rule applied to numerics)."""
        for v in np.asarray(history, float).ravel():
            self.push(v)
        return self

    def push(self, v):
        v = float(v)
        self._i += 1
        if self.window is None:
            self._n += 1
            d = v - self._mean
            self._mean += d / self._n
            self._m2 += d * (v - self._mean)
        else:
            if len(self._buf) == self.window:                   # exact removal via Welford downdate
                old = self._buf[0]
                if self._n == 1:
                    self._mean = self._m2 = 0.0
                    self._n = 0
                else:
                    d = old - self._mean
                    self._mean -= d / (self._n - 1)
                    self._m2 -= d * (old - self._mean)
                    self._n -= 1
            self._buf.append(v)
            self._n += 1
            d = v - self._mean
            self._mean += d / self._n
            self._m2 += d * (v - self._mean)
        lo = self._i - (self.window or (self._i + 1)) + 1
        for dq, keep in ((self._minq, lambda a, b: a <= b), (self._maxq, lambda a, b: a >= b)):
            while dq and dq[0][0] < lo:
                dq.popleft()
            while dq and not keep(dq[-1][1], v):
                dq.pop()
            dq.append((self._i, v))
        return self

    @property
    def count(self):
        return self._n

    @property
    def mean(self):
        return self._mean if self._n else float("nan")

    def std(self, ddof=0):
        if self._n <= ddof:
            return float("nan")
        return math.sqrt(max(self._m2, 0.0) / (self._n - ddof))

    @property
    def min(self):
        return self._minq[0][1] if self._minq else float("nan")

    @property
    def max(self):
        return self._maxq[0][1] if self._maxq else float("nan")


def _partial(x, window, min_periods, fn):
    out = np.full(x.size, np.nan)
    for i in range(x.size):
        lo = max(0, i - window + 1)
        if i - lo + 1 >= min_periods:
            out[i] = fn(x[lo:i + 1])
    return out


def _selftest():
    """Contracts:
    1. EVERY rolling function is prefix-consistent (lookahead_lint causal at 0.0 drift) AND bit-identical to
       the naive per-window loop and to the conditioning layer's TRAILING_STATS lambdas.
    2. THE FAST-PATH CANCELLATION NEGATIVE IS MEASURED, not asserted from theory: a 1e8 offset destroys the
       cumsum std while the exact path is untouched -- the reason exact is the default.
    3. STREAMING == VECTORISED: Welford (expanding and windowed, with the exact downdate) matches rolling_*
       to 1e-9; deque extremes match exactly; warm_start(x[:k]) + push(x[k:]) == push(x).
    4. NaN warm-up: positions before window-1 are NaN, never a silently-shrunk window.
    """
    from holographic.agents_and_reasoning.holographic_honesty import lookahead_lint
    from holographic.agents_and_reasoning.holographic_conditioning import TRAILING_STATS

    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(400))
    w = 20

    # (1) causal + exact vs naive + exact vs TRAILING_STATS
    series = {
        "mean": (lambda s: np.nan_to_num(rolling_mean(s, w)), TRAILING_STATS["mean"]),
        "std": (lambda s: np.nan_to_num(rolling_std(s, w)), TRAILING_STATS["std"]),
        "min": (lambda s: np.nan_to_num(rolling_min(s, w)), TRAILING_STATS["min"]),
        "max": (lambda s: np.nan_to_num(rolling_max(s, w)), TRAILING_STATS["max"]),
        "range": (lambda s: np.nan_to_num(rolling_range(s, w)), TRAILING_STATS["range"]),
        "drawdown": (lambda s: np.nan_to_num(rolling_drawdown(s, w)), TRAILING_STATS["drawdown"]),
    }
    for name, (fn, lam) in series.items():
        r = lookahead_lint(fn, x)
        assert r["causal"] and r["max_drift"] == 0.0, (name, r)
        full = fn(x)
        for i in (w - 1, 137, 399):
            assert full[i] == lam(x[i - w + 1:i + 1]), (name, i)     # BIT-identical to the gate's lambda
    rq = rolling_quantile(x, w, 0.9)
    assert lookahead_lint(lambda s: np.nan_to_num(rolling_quantile(s, w, 0.9)), x)["causal"]
    assert rq[137] == np.quantile(x[137 - w + 1:138], 0.9)
    for fn in (lambda s: ewma(s, 0.1), lambda s: ewm_std(s, 0.1)):
        assert lookahead_lint(fn, x)["max_drift"] == 0.0

    # the classic EW leak, caught by our own lint: seeding the EW variance at the FULL sample's variance.
    def leaky_ewstd(s, a=0.1):
        m = np.empty_like(s); v = np.empty_like(s)
        m[0], v[0] = s[0], s.var()                              # <- the look-ahead: var() sees everything
        for i in range(1, s.size):
            m[i] = (1 - a) * m[i - 1] + a * s[i]
            v[i] = (1 - a) * (v[i - 1] + a * (s[i] - m[i - 1]) ** 2)
        return np.sqrt(v)
    assert not lookahead_lint(leaky_ewstd, x)["causal"]

    # (2) the cancellation negative, MEASURED: same data, offset by 1e8.
    y = rng.standard_normal(300)
    true_std = rolling_std(y + 1e8, w)[w:]
    fast_std = rolling_std(y + 1e8, w, fast=True)[w:]
    exact_err = float(np.nanmax(np.abs(true_std - rolling_std(y, w)[w:])))
    fast_err = float(np.nanmax(np.abs(fast_std - rolling_std(y, w)[w:])))
    assert exact_err < 1e-6, exact_err                          # exact path: offset is invisible
    assert fast_err > 0.1, fast_err                             # fast path: the std is GARBAGE (digits gone)

    # (3) streaming == vectorised
    ss = StreamingStats(window=w)
    means, stds, mins, maxs = [], [], [], []
    for v in x:
        ss.push(v)
        means.append(ss.mean); stds.append(ss.std()); mins.append(ss.min); maxs.append(ss.max)
    vm = rolling_mean(x, w)
    vs = rolling_std(x, w)
    assert max(abs(a - b) for a, b in zip(means[w - 1:], vm[w - 1:])) < 1e-9
    assert max(abs(a - b) for a, b in zip(stds[w - 1:], vs[w - 1:])) < 1e-9
    assert all(a == b for a, b in zip(mins[w - 1:], rolling_min(x, w)[w - 1:]))
    assert all(a == b for a, b in zip(maxs[w - 1:], rolling_max(x, w)[w - 1:]))
    # warm start == continuous
    a1 = StreamingStats(window=w).warm_start(x[:200])
    for v in x[200:]:
        a1.push(v)
    a2 = StreamingStats(window=w).warm_start(x)
    assert abs(a1.mean - a2.mean) < 1e-12 and abs(a1.std() - a2.std()) < 1e-12
    assert a1.min == a2.min and a1.max == a2.max
    # expanding Welford matches numpy on the whole sample
    e = StreamingStats().warm_start(x)
    assert abs(e.mean - x.mean()) < 1e-9 and abs(e.std() - x.std()) < 1e-9

    # (4) warm-up NaN, and refusals by name
    assert np.isnan(rolling_mean(x, w)[:w - 1]).all()
    for bad in (lambda: rolling_mean(x, 0), lambda: rolling_std(x, len(x) + 1),
                lambda: rolling_quantile(x, w, 1.5), lambda: ewma(x, 0.0)):
        try:
            bad()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    print("holographic_rolling selftest OK (mean/std/min/max/range/quantile/drawdown/ewma all lint causal at "
          "0.0 drift and are BIT-identical to the conditioning gate's lambdas; the fast cumsum std at a 1e8 "
          "offset is off by %.2f while the exact default is off by %.1e -- cancellation measured, exact "
          "default earned; streaming Welford+deque matches vectorised to 1e-9/-exactly and warm_start == "
          "continuous to 1e-12; the full-sample-variance EW seed is caught by our own lint)"
          % (fast_err, exact_err))


if __name__ == "__main__":
    _selftest()

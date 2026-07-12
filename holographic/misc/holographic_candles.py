"""holographic_candles.py -- treat OHLC price candles as what they actually are: a SAMPLED WAVE.

THE INSIGHT
-----------
A candlestick is not a single price point. Each bar is a SAMPLE of an underlying continuous price wave, and the
Open/High/Low/Close carry FOUR time-ordered facts about where that wave went during the bar:

    * Open  -- the wave's value at the START of the bar,
    * High  -- the highest the wave reached SOMETIME inside the bar,
    * Low   -- the lowest it reached sometime inside the bar,
    * Close -- its value at the END of the bar.

So a candle series is a wave sampled coarsely, plus an ENVELOPE (the high/low band) that tells us the wave
travelled at least that far BETWEEN the endpoint samples. Most tooling throws away three of those four numbers and
plots close as a line -- discarding the within-bar excursions that are the whole point. This module keeps them:
it reconstructs a continuous wave from the OHLC samples so the engine's existing signal machinery (spectrum,
band-limiting, the phase-randomized null, generator fitting, forecasting) applies to price directly.

WHAT "AS A WAVE" MEANS CONCRETELY
  1. carrier(ohlc)  -- the one-value-per-bar signal (typical price (H+L+C)/3, or close), the wave's samples.
  2. envelope(ohlc) -- the (upper, lower) high/low band around the carrier: how far the wave swung intra-bar.
  3. intrabar_path(ohlc) -- a FOUR-points-per-bar reconstruction O -> {H,L in the inferred order} -> C, the
     honest higher-resolution wave. The H/L visiting order is inferred from bar direction: an up bar (close>open)
     most likely dipped to the low first then ran to the high; a down bar did the reverse. This is the standard
     OHLC->path approximation, stated as a wave reconstruction.

Once price IS a wave, everything composes: analytic_signal for its instantaneous amplitude/phase, spectral_bandwidth
for how band-limited it is, phase_randomize for the honest "is this more than autocorrelation" null,
fit_deterministic for "what generator explains this leg", ladder_predict for forecasting.

KEPT NEGATIVE: the intra-bar H/L order is an INFERENCE, not data -- OHLC does not record which extreme came first.
The direction heuristic is right more often than not but is a MODEL; the reconstruction is honest about being a
plausible path, not the true tick path. For anything that depends on the true order, use ticks, not candles.

NumPy / stdlib only. Deterministic. Additive.
"""

import numpy as np


def _as_ohlc(candles):
    """Accept candles as an (N,4) O/H/L/C array, an (N,5) [ts,O,H,L,C] or (N,6) [ts,O,H,L,C,V] array (the market
    loader's format), or a list of dicts with o/h/l/c keys. Return a clean (N,4) float array of O,H,L,C. One
    tolerant front door so callers don't reshape by hand."""
    if isinstance(candles, (list, tuple)) and len(candles) and isinstance(candles[0], dict):
        return np.array([[c["o"], c["h"], c["l"], c["c"]] for c in candles], float)
    arr = np.asarray(candles, float)
    if arr.ndim != 2:
        raise ValueError("candles must be 2-D (N x 4/5/6); got shape %r" % (arr.shape,))
    ncol = arr.shape[1]
    if ncol == 4:
        return arr
    if ncol in (5, 6):
        return arr[:, 1:5]                                     # drop leading timestamp (and trailing volume)
    raise ValueError("each candle needs 4 (OHLC), 5 ([ts,OHLC]) or 6 ([ts,OHLCV]) columns; got %d" % ncol)


def carrier(candles, kind="typical"):
    """The one-value-per-bar WAVE: the carrier signal the candles sample. `kind`='typical' returns (H+L+C)/3 (the
    standard 'typical price', which folds in the intra-bar range), 'close' returns the close, 'median' returns
    (H+L)/2, 'ohlc4' returns (O+H+L+C)/4. Returns a 1-D array, one value per bar -- feed it to any signal op."""
    o, h, l, c = _as_ohlc(candles).T
    if kind == "typical":
        return (h + l + c) / 3.0
    if kind == "close":
        return c
    if kind == "median":
        return (h + l) / 2.0
    if kind == "ohlc4":
        return (o + h + l + c) / 4.0
    raise ValueError("kind must be typical/close/median/ohlc4; got %r" % kind)


def envelope(candles):
    """The high/low BAND around the carrier -- how far the wave swung INTRA-bar, which a close-only line discards.
    Returns (upper, lower) = (highs, lows), each a 1-D array. The band width (upper-lower) is the per-bar range,
    the amplitude of the wave's within-sample excursion."""
    _o, h, l, _c = _as_ohlc(candles).T
    return h, l


def intrabar_path(candles, steps_per_bar=4):
    """Reconstruct a higher-resolution WAVE from the OHLC samples: for each bar emit O -> {H,L in inferred order}
    -> C, so the within-bar excursions become part of the signal. The H/L visiting order is inferred from bar
    DIRECTION -- an up bar (close>open) is modelled as dipping to the low first then running to the high; a down
    bar does the reverse. Returns a 1-D array of length ~ N*steps_per_bar (the reconstructed wave), and is exact
    at the endpoints (every open and close is hit).

    KEPT NEGATIVE: the H/L order is an INFERENCE (OHLC doesn't record it), so this is a PLAUSIBLE path, not the
    true tick path -- honest higher resolution, not invented certainty."""
    ohlc = _as_ohlc(candles)
    path = []
    for (o, h, l, c) in ohlc:
        up = c >= o
        # up bar: open, low, high, close (dip then run). down bar: open, high, low, close (pop then fall).
        pts = [o, l, h, c] if up else [o, h, l, c]
        if steps_per_bar == 4:
            path.extend(pts)
        else:
            # resample the 4 anchor points to steps_per_bar via linear interpolation (keeps endpoints exact).
            anchor_t = np.linspace(0, 1, 4)
            out_t = np.linspace(0, 1, steps_per_bar)
            path.extend(np.interp(out_t, anchor_t, pts))
    return np.asarray(path, float)


def candle_range(candles):
    """Per-bar RANGE (High-Low), the amplitude of each within-bar excursion, and the BODY (|Close-Open|). Returns
    (ranges, bodies). The range/body ratio is a classic 'how much of the swing was noise vs trend' measure -- a
    small body inside a big range is a bar that went nowhere despite swinging."""
    o, h, l, c = _as_ohlc(candles).T
    return (h - l), np.abs(c - o)


def resample_uniform(signal, n_out):
    """Resample a 1-D wave to exactly `n_out` uniformly-spaced points (linear). A convenience so a candle carrier
    or intrabar path drops straight into ops that want a fixed length (spectrum, fit, forecast). Deterministic."""
    signal = np.asarray(signal, float)
    if len(signal) < 2:
        return np.repeat(signal, n_out)[:n_out]
    return np.interp(np.linspace(0, 1, n_out), np.linspace(0, 1, len(signal)), signal)


def _selftest():
    """Contracts -- the candle-as-wave representation is faithful:

    1. carrier/envelope round-trip: a synthetic sine 'price' sampled into OHLC bars recovers a carrier that
       correlates strongly with the underlying sine, and an envelope that BRACKETS the carrier (upper>=carrier>=lower).
    2. intrabar_path is higher-resolution and hits every open and close exactly; it is longer than the carrier.
    3. The path's direction inference is correct: an up bar visits low before high, a down bar the reverse.
    4. Composability: the reconstructed wave feeds spectral analysis (a band-limited sine reads a small bandwidth).
    5. Determinism + tolerant input (4/5/6 columns and dict form all give the same OHLC).
    """
    # build a synthetic underlying wave and sample it into OHLC bars.
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4 * np.pi, 2000)
    underlying = 100 + 5 * np.sin(t)                           # a clean price 'wave'
    bar = 40                                                   # samples per bar
    ohlc = []
    for i in range(0, len(underlying) - bar, bar):
        seg = underlying[i:i + bar]
        ohlc.append([seg[0], seg.max(), seg.min(), seg[-1]])
    ohlc = np.array(ohlc)

    # (1) carrier correlates with the underlying (downsampled to bar centres); envelope brackets it.
    car = carrier(ohlc, "typical")
    centres = underlying[bar // 2::bar][:len(car)]
    assert np.corrcoef(car, centres)[0, 1] > 0.95, "carrier should track the underlying wave"
    up, lo = envelope(ohlc)
    assert np.all(up >= car - 1e-9) and np.all(lo <= car + 1e-9), "envelope must bracket the carrier"

    # (2) intrabar path is higher resolution and hits endpoints.
    path = intrabar_path(ohlc, steps_per_bar=4)
    assert len(path) == 4 * len(ohlc)
    assert abs(path[0] - ohlc[0, 0]) < 1e-9                    # first open hit
    assert abs(path[3] - ohlc[0, 3]) < 1e-9                    # first close hit

    # (3) direction inference: construct one up bar and one down bar explicitly.
    up_bar = np.array([[10.0, 12.0, 9.0, 11.0]])              # close>open -> up -> visits low(9) before high(12)
    up_path = intrabar_path(up_bar)
    assert list(up_path) == [10.0, 9.0, 12.0, 11.0], up_path
    down_bar = np.array([[11.0, 12.0, 9.0, 10.0]])           # close<open -> down -> visits high(12) before low(9)
    down_path = intrabar_path(down_bar)
    assert list(down_path) == [11.0, 12.0, 9.0, 10.0], down_path

    # (4) composability: the carrier of a band-limited sine reads a small spectral bandwidth.
    from holographic.misc.holographic_bandwidth import spectral_bandwidth
    bw = spectral_bandwidth(car, energy_fraction=0.95)
    assert 0.0 <= bw < 0.5, ("a clean sine price should be band-limited, got bandwidth %.3f" % bw)

    # (5) determinism + tolerant input.
    assert np.array_equal(intrabar_path(ohlc), intrabar_path(ohlc))
    # 4-col, 5-col ([ts,OHLC]) and dict forms agree on OHLC.
    five = np.column_stack([np.arange(len(ohlc)), ohlc])
    dicts = [{"o": r[0], "h": r[1], "l": r[2], "c": r[3]} for r in ohlc]
    assert np.allclose(_as_ohlc(ohlc), _as_ohlc(five)) and np.allclose(_as_ohlc(ohlc), _as_ohlc(dicts))

    ranges, bodies = candle_range(ohlc)
    print("holographic_candles selftest OK (carrier tracks the underlying sine r=%.2f; envelope brackets it; "
          "intrabar path is 4x resolution and hits every open/close; up-bar visits low->high, down-bar "
          "high->low; band-limited price reads bandwidth %.2f; tolerant 4/5-col + dict input; deterministic)"
          % (np.corrcoef(car, centres)[0, 1], bw))


if __name__ == "__main__":
    _selftest()

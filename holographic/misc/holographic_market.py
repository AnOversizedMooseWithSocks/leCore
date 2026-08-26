"""Market data on the holographic substrate -- what numeric time series can and
cannot do here, measured on real DEX candles (DAI/WETH, 100 one-minute bars,
data/dai_weth_ohlcv.json).

WHAT WORKS (each measured):

  * A CANDLE IS ONE RECORD. Five roles (open/high/low/close/volume) bound to
    SCALAR codes (ScalarEncoder: near values stay near -- graded similarity,
    which symbols cannot give numbers) and bundled. Round-trip decode error is
    1.6-2.9 bp on prices vs the data's own 8.2 bp one-minute return sd -- the
    record's resolution is finer than the signal, so a candle genuinely lives
    in one vector.

  * THE PERMUTATION TEST REDISCOVERS MARKET STRUCTURE. Price LEVELS are
    provably ordered (z = +6.8 against their own shuffle); RETURN SIGNS are
    indistinguishable from their own shuffle (z = -0.6) -- the efficient-market
    property, found by the same threshold-free test that validates creature
    routes. The engine's own instrument says: levels carry order, returns do
    not (at this sample).

  * MOTIF RECALL works as MEMORY; candle-level NOVELTY works as detection.
    Windows of position-bound candle features retrieve similar past stretches;
    and each candle's similarity to its nearest PRIOR candle, flagged at the
    data's own scale (z > 2 below the mean), catches the real anomalies -- the
    2685-volume / 17bp-range candle (z 4.2) and the +21bp swing (z 5.7).

WHAT DOES NOT (the kept negatives):

  * PREDICTION IS A COIN FLIP HERE. Walk-forward next-sign prediction from the
    most similar past motif scores 49% -- and so does everything else: majority
    55%, persistence 54%, anti-persistence 46%, ALL inside the binomial
    39-61% chance band at n=82. No edge is demonstrable on 100 candles, by any
    method tried, and the honest instrument (the band) says so explicitly.
    Recall is memory, not prophecy.

  * WINDOW-level novelty DILUTES single-candle anomalies (one outlier role
    among 18 in a summed window barely moves the cosine) and is biased early
    (few priors to match). Novelty must run at the granularity of the thing
    that can be anomalous -- candle level here.

AT SCALE (data/sol_5min.npz: 15,793 SOL ticks over 2.2 days -- despite the
upload's name it is ~1-second jupiter ticks in ~50-tick bursts, plus 419
five-minute coingecko points; analysis is within-burst only, never computing a
return across a hole):

  * THE INSTRUMENT RESPONDS TO REGIME. The same sequentiality test that called
    DAI-minute return signs shuffle-like (z=-0.6) calls tick-scale signs
    STRONGLY ORDERED (z=+44, shuffle control -0.9): tick momentum (+0.20 sign
    autocorr) is real structure and the permutation test finds it. Two
    datasets, opposite honest verdicts, one instrument.

  * STRUCTURE IS LOCATED, NOT ASSUMED. Three measurements map where the tick
    data's structure lives. (1) The momentum is INTRA-BURST ONLY: sign
    persistence is 60.5% +/- 2.7 within bursts but 49.4% (chance) across the
    ~5-minute holes, and one burst's drift says nothing about the next
    (45.7% +/- 12). Sub-minute microstructure, dying at every gap. (2) Bursts
    are mildly drifty (13% exceed the binomial z>2 imbalance vs 5% expected).
    (3) MOVE-SHAPES DO NOT RECUR: against order-shuffled within-burst
    surrogates (same marginal, destroyed order), nearest-neighbour window
    similarity shows no excess (z=-0.6). The order structure is momentum and
    drift, not repeating chart patterns.

  * RAY-PROJECTED PRICE TARGETS -- the validated win. At a matched K-move
    pattern, shoot R rays (the R most similar past windows); each ray carries
    the cumulative return that followed it over the next H moves; the bundle's
    quantiles are the target distribution. Walk-forward, proper-scored
    (pinball + coverage), with the selection done honestly (R chosen on the
    first half only): on the untouched second half the rays beat the
    unconditional outcome distribution by 0.134 bp/point, paired z=+3.3, with
    ~13% tighter 80% intervals near nominal coverage (85% vs the baseline's
    over-covering 89%). The pattern's value is not calling direction (the
    motif lost that contest to persistence) but locating the CURRENT CONTEXT'S
    OUTCOME SCALE -- sharper, calibrated targets. Kept honest: ray-similarity
    CONFIDENCE gates difficulty, not skill (the confident quartile improves
    the baseline exactly as much), so it is a when-to-trust gauge, not an
    edge.

  * MOMENTUM IS PROVEN, AND THE SIMPLEST RULE OWNS IT. Direction-of-next-move,
    walk-forward, n=1454, chance band 50 +/- 2.6%: persistence (last nonzero
    sign) scores 60.2% -- OUTSIDE the band, a real measurable edge. The
    holographic motif scores 54.1% -- also outside chance (it captures genuine
    signal) but DECISIVELY BELOW persistence: measurement beats sophistication,
    the flocking lesson in a new domain. And at raw next-tick granularity,
    always-predict-flat (90.2%, ticks are 88% flat) beats everything. The
    motif's value is memory and novelty, not direction-calling; if you want
    direction here, last-nonzero-sign is the measured tool.
"""
import json
import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, bundle, involution, cosine, permute, random_vector
from holographic.io_and_interop.holographic_encoders import ScalarEncoder


def load_ticks(path="data/sol_5min.npz"):
    """The checked-in tick dataset: (timestamps_s, prices), ascending, deduped."""
    d = np.load(path)
    return d["ts"], d["px"]


def load_ohlcv(path="data/dai_weth_ohlcv.json"):
    """The checked-in real dataset: ascending [ts, o, h, l, c, v] rows."""
    data = json.load(open(path))
    return np.array(data["ohlcv"], float)


def load_sol_market(path="data/sol_market.npz", timeframe="1h"):
    """Real SOL/USDT bars from Binance (vendored from a sibling project that exercised
    this engine on market data). Multi-timeframe and richer than the DAI/WETH set above:
    each row is [time, open, high, low, close, volume, taker_buy, ofi] where taker_buy is
    aggressive buy volume and ofi is the order-flow-imbalance sign -- microstructure the
    permutation tests and CandleCoder can actually chew on.

    timeframe is one of "5m", "1h", "1d" for SOL, or "btc"/"eth" for the 1d cross-asset
    series (same column layout), or "funding" for the [time, rate] funding-rate series.
    Returns (rows, field_names). All ascending in time."""
    d = np.load(path, allow_pickle=False)
    fields = [str(x) for x in d["fields"]]
    key = {"5m": "sol_5m", "1h": "sol_1h", "1d": "sol_1d",
           "btc": "btc_1d", "eth": "eth_1d", "funding": "funding"}.get(timeframe, "sol_1h")
    rows = np.asarray(d[key], float)
    if key == "funding":
        return rows, ["time", "rate"]
    return rows, fields


def load_onchain_traders(path="data/onchain_traders.json"):
    """Realized on-chain Jupiter Perpetuals trades, read off Solana's public ledger by the
    sibling project (no live RPC needed here -- the produced data is vendored). Two parts:

      profiles : per-wallet summaries -- trades, net_usd_per_trade, win_rate, edge_t_stat,
                 avg_hold_hours, median_leverage, long_fraction, liquidations. Honest by
                 construction: edge_t_stat sits next to PnL so a wallet green on a handful
                 of trades reads as luck, not skill (the same n-problem the engine keeps
                 flagging elsewhere).
      realized : individual closed trades -- market, side, net_usd, leverage, hold_hours,
                 liquidated, size_usd. Real labelled outcomes for a records mind.

    Returns the parsed dict. The profiles/realized lists are records ready for RecordEncoder
    or the records-world loader."""
    return json.load(open(path))


class CandleCoder:
    """Candles as holographic records, windows as motifs, anomalies by novelty.

    Prices are encoded in BASIS-POINT space around 1.0 (the pair's own working
    range), volume as log10 -- both through ScalarEncoder so nearby values stay
    near, which is what lets a noisy unbind still decode to the right number.
    """

    PRICE_ROLES = ("open", "high", "low", "close")

    def __init__(self, dim=2048, bp_range=60.0, seed=0):
        self.dim = dim
        rng = np.random.default_rng(seed)
        self.roles = {r: random_vector(dim, rng)
                      for r in (*self.PRICE_ROLES, "vol", "ret", "range")}
        self.px = ScalarEncoder(dim, lo=-bp_range, hi=bp_range, seed=seed + 1)
        self.vol = ScalarEncoder(dim, lo=-1.0, hi=3.6, seed=seed + 2)
        self.ret = ScalarEncoder(dim, lo=-30.0, hi=30.0, seed=seed + 3)
        self.rng_enc = ScalarEncoder(dim, lo=0.0, hi=25.0, seed=seed + 4)

    # ---- a candle as ONE record (round-trip measured: 1.6-2.9 bp) ----
    @staticmethod
    def _bp(p):
        return (p - 1.0) * 1e4

    def encode_candle(self, o, h, l, c, v):
        parts = [bind(self.roles["open"], self.px.encode(self._bp(o))),
                 bind(self.roles["high"], self.px.encode(self._bp(h))),
                 bind(self.roles["low"], self.px.encode(self._bp(l))),
                 bind(self.roles["close"], self.px.encode(self._bp(c))),
                 bind(self.roles["vol"], self.vol.encode(np.log10(max(v, 1e-3))))]
        return bundle(parts)

    def decode_candle(self, vec, steps=600):
        out = {}
        for r in self.PRICE_ROLES:
            bp = self.px.decode(bind(vec, involution(self.roles[r])), steps=steps)
            out[r] = 1.0 + bp / 1e4
        out["vol"] = 10 ** self.vol.decode(bind(vec, involution(self.roles["vol"])),
                                           steps=400)
        return out

    # ---- feature vectors for motifs and novelty ----
    def feature_vec(self, ret_bp, range_bp, volume):
        """One candle's behavioural signature: its move, its range, its volume."""
        return (bind(self.roles["ret"], self.ret.encode(float(np.clip(ret_bp, -30, 30))))
                + bind(self.roles["range"], self.rng_enc.encode(min(float(range_bp), 25.0)))
                + bind(self.roles["vol"], self.vol.encode(np.log10(max(float(volume), 1e-3)))))

    def window_vec(self, feature_vecs):
        """A motif: position-bound candle features summed -- order matters."""
        return np.sum([permute(fv, len(feature_vecs) - 1 - k)
                       for k, fv in enumerate(feature_vecs)], axis=0)

    def nearest_motif(self, query_vec, past_vecs):
        """RECALL (memory, not prophecy): the most similar past window."""
        P = np.atleast_2d(np.asarray(past_vecs, float))
        if P.shape[0] == 0:
            return -1, -1.0
        qn = np.asarray(query_vec, float)
        qn = qn / (np.linalg.norm(qn) + 1e-12)
        Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
        sims = Pn @ qn                                   # cosine vs every past window in one matvec (was a loop)
        j = int(sims.argmax())
        return j, float(sims[j])

    def novelty(self, ohlcv, warmup=10):
        """Candle-level anomaly detection at the data's own scale: each candle's
        similarity to its nearest PRIOR candle; flagged when more than two sigma
        BELOW the mean similarity (z > 2 -- the standard bar, the data's own
        spread). Returns [(index, z)] sorted by z descending. `warmup` candles
        are skipped so every scored candle has priors to compare against (the
        measured early-history bias)."""
        a = np.asarray(ohlcv, float)
        c = a[:, 4]
        rets = np.append(0.0, np.diff(np.log(c)) * 1e4)
        rng_bp = (a[:, 2] - a[:, 3]) * 1e4
        feats = [self.feature_vec(rets[i], rng_bp[i], a[i, 5])
                 for i in range(len(a))]
        scored = []
        for i in range(warmup, len(a)):
            best = max(cosine(feats[i], feats[j]) for j in range(i))
            scored.append((i, best))
        sims = np.array([s for _, s in scored])
        mu, sd = sims.mean(), sims.std() + 1e-12
        out = [(i, float((mu - s) / sd)) for i, s in scored if (mu - s) / sd > 2.0]
        return sorted(out, key=lambda t: -t[1])


def move_series(ts, px, max_gap=2.0):
    """The tick data reduced to its MOVES: nonzero within-burst log-returns (bp)
    plus each move's burst id. A burst is a maximal run of ticks no more than
    `max_gap` seconds apart (the data's own dominant spacing); returns are never
    computed across a hole."""
    gaps = np.diff(ts)
    rets = np.diff(np.log(px)) * 1e4
    burst = np.cumsum(np.append(0, (gaps > max_gap)))[:len(rets)]
    within = np.append(gaps <= max_gap, False)[:len(rets)]
    m = (rets != 0) & within
    return rets[m], burst[m]


class RayProjector:
    """Price targets by RAY PROJECTION -- the validated use of pattern matching
    on this data. Encode K-move windows (position-bound scalar codes); at a new
    window, the R most similar PAST windows are rays, each carrying the
    cumulative return that followed it over the next H moves; the bundle's
    quantiles are the target distribution for that horizon. Measured (split-half
    honest): beats the unconditional outcome distribution at proper score
    (pinball, paired z=+3.3 on the held-out half) with ~13% tighter calibrated
    intervals. Direction-calling is NOT this tool's job (persistence owns that);
    the pattern locates the current context's outcome SCALE."""

    def __init__(self, dim=512, K=5, H=3, R=80, seed=1):
        self.dim, self.K, self.H, self.R = dim, K, H, R
        self._se = None
        self.seed = seed

    def fit(self, moves, burst):
        """Encode every eligible window (its K moves and H-move outcome inside
        ONE burst). Walk-forward use: project(row) only consults rows < row."""
        K, H = self.K, self.H
        p99 = np.percentile(np.abs(moves), 99)       # the moves' own scale
        self._se = ScalarEncoder(self.dim, lo=-p99, hi=p99, seed=self.seed)
        self._p99 = p99
        rows = [i for i in range(K, len(moves) - H) if burst[i - K] == burst[i + H - 1]]
        M = np.zeros((len(rows), self.dim), np.float32)
        outc = np.zeros(len(rows))
        for r_, i in enumerate(rows):
            w = np.sum([permute(self._se.encode(float(np.clip(moves[i - K + j], -p99, p99))),
                                K - 1 - j) for j in range(K)], axis=0)
            M[r_] = w / (np.linalg.norm(w) + 1e-12)
            outc[r_] = moves[i:i + H].sum()
        self.M, self.outcomes, self.rows = M, outc, rows
        return self

    def project(self, row, quantiles=(0.1, 0.5, 0.9)):
        """Target distribution at eval row `row` from strictly-past rays.
        Returns (quantile values in bp over the next H moves, mean ray
        similarity). The similarity is a WHEN-TO-TRUST gauge (it gates
        difficulty, not skill -- measured)."""
        if row < self.R + 1:
            raise ValueError("not enough past windows for the ray bundle")
        sims = self.M[:row - 1] @ self.M[row]
        top = np.argsort(sims)[-self.R:]
        qs = [float(np.quantile(self.outcomes[top], q)) for q in quantiles]
        return qs, float(np.mean(sims[top]))

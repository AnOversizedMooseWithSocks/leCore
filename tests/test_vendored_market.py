"""The vendored real market + on-chain datasets (extracted from a sibling project that
exercised this engine on SOL): they must load with the right shapes, feed the existing
CandleCoder, and -- for the trader records -- be genuinely learnable on the unified mind.

Kept separate from test_holographic_market.py on purpose: that file's wins are measured on
the DAI/WETH set and some paths there touch network/data; these checks are self-contained on
the checked-in files so they never go flaky."""
import numpy as np
from collections import defaultdict

from holographic.misc.holographic_market import load_sol_market, load_onchain_traders, CandleCoder


def test_sol_market_loads_every_timeframe_with_the_right_columns():
    # Three SOL timeframes + two cross-assets share an 8-column [time,open,high,low,close,
    # volume,taker_buy,ofi] layout; funding is a [time,rate] series. All ascending, real.
    for tf in ("5m", "1h", "1d", "btc", "eth"):
        rows, fields = load_sol_market(timeframe=tf)
        assert rows.ndim == 2 and rows.shape[1] == 8
        assert fields[:6] == ["time", "open", "high", "low", "close", "volume"]
        assert np.all(np.diff(rows[:, 0]) >= 0)            # time ascending
        assert rows[:, 4].min() > 0                         # real positive closes
    fund, ff = load_sol_market(timeframe="funding")
    assert fund.shape[1] == 2 and ff == ["time", "rate"]


def test_sol_candles_feed_the_existing_candle_coder():
    # The whole point of vendoring as [t,o,h,l,c,v] rows: a SOL candle round-trips through
    # the SAME CandleCoder the DAI/WETH data uses, no new machinery. CandleCoder works in
    # basis-point space around 1.0 (it was built for a tight stablecoin range), so SOL's
    # $60-250 prices are first normalized to the window's own price level -- the realistic
    # use, and then the round-trip is faithful.
    rows, _ = load_sol_market(timeframe="1h")
    win = rows[:200, :6]
    ref = float(win[:, 4].mean())                          # the window's working level
    cc = CandleCoder(bp_range=400.0)                        # wide enough for SOL's swing
    i = 50
    o, h, l, c = (win[i, k] / ref for k in (1, 2, 3, 4))
    rec = cc.encode_candle(o, h, l, c, win[i, 5])
    back = cc.decode_candle(rec)
    decoded_close = back["close"] * ref
    assert abs(decoded_close - win[i, 4]) / win[i, 4] < 0.01   # < 1% after normalization


def test_onchain_traders_load_and_carry_the_honesty_fields():
    data = load_onchain_traders()
    assert len(data["profiles"]) > 0 and len(data["realized"]) > 0
    p = data["profiles"][0]
    # the fields that keep it honest: trade count and per-trade edge t-stat beside PnL
    for k in ("trades", "net_usd_per_trade", "edge_t_stat", "win_rate", "liquidations"):
        assert k in p
    r = data["realized"][0]
    for k in ("market", "side", "net_usd", "leverage", "liquidated"):
        assert k in r


def test_onchain_trader_records_are_learnable_and_classifiable():
    # The wallet records (labelled by honest edge, not raw PnL) must be genuinely learnable:
    # absorb a 70% split, classify the held-out 30% above chance for a 3-class problem.
    import tools.unified_app as ua
    from holographic.misc.holographic_unified import UnifiedMind
    items, _, _ = ua.load_onchain_world()
    assert len(items) > 20
    by = defaultdict(list)
    for x, lab, mod in items:
        by[lab].append((x, mod))
    rng = np.random.default_rng(0)
    train, test = [], []
    for lab, docs in by.items():
        docs = list(docs)
        rng.shuffle(docs)
        cut = int(len(docs) * 0.7)
        train += [(x, lab, m) for x, m in docs[:cut]]
        test += [(x, lab, m) for x, m in docs[cut:]]
    rng.shuffle(train)
    mind = UnifiedMind(dim=2048, seed=0)
    mind.absorb(train)
    acc = sum(mind.classify(x)[0] == lab for x, lab, _ in test) / max(1, len(test))
    assert acc > 0.5                                        # well above 3-class chance (0.33)

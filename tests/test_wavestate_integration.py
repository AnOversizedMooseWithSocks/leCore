"""Wave-state encoder (I3) at the seams -- and the I4 worked example the backlog asked for: the paved-road
chain decompose -> re-encode -> screen -> pipeline_null, run end to end through the mind with the honesty
verdicts attached at every joint."""
import numpy as np

import lecore


def _ohlc_from_path(path, swing, rng):
    o = path.copy()
    c = np.roll(o, -1); c[-1] = o[-1]
    h = np.maximum(o, c) + swing * np.abs(rng.standard_normal(path.size))
    l = np.minimum(o, c) - swing * np.abs(rng.standard_normal(path.size))
    return np.stack([o, h, l, c], axis=1)


def test_encoder_through_the_faculty_and_recall_through_causal_index():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(0)
    enc = mind.wave_state_encoder(512, window=32, seed=0)
    idx = mind.causal_index()
    labels = []
    for k in range(30):
        regime = k % 2
        p = np.linspace(0, 6, 32) + 0.3 * rng.standard_normal(32) if regime == 0 \
            else 1.5 * np.sin(np.linspace(0, 8 * np.pi, 32)) + 0.3 * rng.standard_normal(32)
        idx.append(enc.encode(_ohlc_from_path(p, 0.5, rng)), k)
        labels.append(regime)
    q = enc.encode(_ohlc_from_path(1.5 * np.sin(np.linspace(0, 8 * np.pi, 32)) + 0.3 * rng.standard_normal(32),
                                   0.5, rng))
    hits = idx.nearest(q, t=30, k=5, lag=1)
    assert sum(1 for h in hits if labels[h[0]] == 1) >= 4


def test_the_envelope_channel_is_screenable_where_close_only_is_blind():
    """A target driven by INTRA-BAR SWING with identical closes everywhere: a wave-state check finds it, a
    close-only check on the same windows cannot -- the measured payoff the encoder exists for, at the
    SignalProgram seam with the honesty gates inside the loop."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(1)
    n_ev = 600
    base = np.cumsum(rng.standard_normal(16))
    swings = np.where(rng.random(n_ev) < 0.5, 0.2, 2.0)
    windows = [_ohlc_from_path(base, s, rng) for s in swings]
    target = np.sign(swings - 1.0) * np.abs(rng.standard_normal(n_ev))

    enc = mind.wave_state_encoder(256, window=16, seed=0)
    # Readout via a CONTRAST AXIS (mean wild prototype minus mean calm prototype, held-out draws): cosine
    # to a single reference barely separated (0.753 vs 0.771 -- one draw's random up/dn pattern is a poor
    # probe of amplitude), while the prototype difference isolates the amplitude direction cleanly
    # (calm -0.48 vs wild +0.05, ~2 pooled sd). The lesson kept: read a bundled channel with a contrast,
    # not with similarity to one exemplar.
    proto_rng = np.random.default_rng(99)
    wild_p = np.mean([enc.encode(_ohlc_from_path(base, 2.0, proto_rng)) for _ in range(10)], axis=0)
    calm_p = np.mean([enc.encode(_ohlc_from_path(base, 0.2, proto_rng)) for _ in range(10)], axis=0)
    axis = wild_p - calm_p
    axis = axis / np.linalg.norm(axis)
    states = np.stack([[float(enc.encode(w) @ axis)] for w in windows])
    closes = np.stack([[float(w[:, 3].sum())] for w in windows])  # identical closes: a constant column

    prog = mind.signal_program(dim=256, seed=0)
    prog.add_check("wave_sim", lambda s: s[:, 0])
    rep = prog.screen(states, target)
    assert rep["passed"] == ["wave_sim"], rep["passed"]
    assert float(np.std(closes)) < 1e-9                          # the close-only view carries literally nothing


def test_the_i4_worked_example_decompose_reencode_screen_pipeline_null():
    """The paved road, as one function, wrapped whole in pipeline_null: decompose (trailing EMA trend, the
    causal one the decomposition_contract certifies) -> re-encode residual sign persistence -> the statistic.
    On structured input the statistic is large; the pipeline null on ITS OWN machinery says how much of that
    the machinery manufactures -- the A2 discipline closing over the entire chain in one call."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(2)

    def chain(series):
        s = np.asarray(series, float).ravel()
        t = np.empty_like(s); t[0] = s[0]
        for i in range(1, s.size):
            t[i] = 0.9 * t[i - 1] + 0.1 * s[i]
        resid = s - t
        sg = np.sign(resid); sg = sg[sg != 0]
        return float(np.mean(sg[1:] == sg[:-1])) if sg.size > 1 else 0.5

    contract = mind.decomposition_contract(
        lambda s: (lambda tt: {"trend": tt, "residual": np.asarray(s, float) - tt})(
            (lambda ss: (lambda t0: [t0.__setitem__(i, 0.9 * t0[i - 1] + 0.1 * ss[i]) or t0 for i in range(1, ss.size)][-1])(
                np.concatenate([[ss[0]], np.zeros(ss.size - 1)])))(np.asarray(s, float).ravel())), 
        np.cumsum(rng.standard_normal(300)))
    assert contract["complete"] and contract["causal"]           # the decomposition step is certified first

    markov = np.zeros(1500)
    for i in range(1, 1500):
        markov[i] = markov[i - 1] + (0.6 * np.sign(markov[i - 1] - markov[i - 2] if i > 1 else 1.0)
                                     + rng.standard_normal())
    r = mind.pipeline_null(chain, markov, surrogate="iid_shuffle", n=100, seed=0)
    assert r["z"] > 3, r                                         # real structure clears the machinery's own null
    noise = rng.standard_normal(1500)
    r0 = mind.pipeline_null(chain, noise, surrogate="iid_shuffle", n=100, seed=0)
    assert abs(r0["z"]) < 3, r0                                  # and noise does not

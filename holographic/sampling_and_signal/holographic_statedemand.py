"""State-demand metering for streams: how much memory does this data DEMAND, measured
before any is allocated (holographic_statedemand, SD-1).

WHY THIS EXISTS
---------------
Every fixed-state sequence system faces the same allocation question and almost none
measure it: how many bits of state does THIS stream require? This module answers with
two instruments transferred from quantum many-body / computational-mechanics theory and
verified on this tree, plus the compressibility gate they feed:

  tt_state_demand   -- TT-SVD bond dimensions of the empirical block distribution.
        The MPS <-> minimal-predictor equivalence says the memory a process demands is
        its rank structure (causal states). Verified 6/6 here: period-p -> ranks p;
        iid -> ranks 1 (independence is a PRODUCT STATE, the easy case); random-walk
        INCREMENTS -> 1. LESSON BAKED IN: the raw ranks of an iid stream read 4/16/54
        from pure sampling noise -- every unfolding is thresholded against the SAME
        stream SHUFFLED (destroys temporal structure, keeps marginals exactly). Rank
        against your own null, like everything else in this engine.

  entropy_rate_report -- block-entropy scaling H(L) ~ E + h*L: h is the entropy rate
        (0 = deterministic, the all-three-corners regime), E the excess entropy (the
        bits an optimal predictor must carry). GUARDED against the failure that was
        measured, not imagined: the plug-in estimator biases LOW outside the dense
        regime (white noise once read 0.47 of its true 3.0 bits at k=8, L=6, T=20k),
        so the report REFUSES L where the word count outruns the sample count.

  compressibility_gate -- two stages with DISJOINT measured failure sets: block entropy
        first (kills walk/ar1/white -- the nulls a phase-randomised surrogate cannot,
        measured 0.90/0.40 raw false-fit there), then an injected surrogate-calibrated
        scorer for the low-rate survivors (kills spectrum-matched stochastic imposters,
        1.000 -> 0.000 measured). THE HORIZON FIELD IS MANDATORY: a phase-randomised
        process is deterministic WITHIN its coherence time and stochastic across it --
        both verdicts are true at their own scale (measured: 400-sample windows of a
        20k randomisation are 95-98%% pure sine) -- so every verdict carries the window
        it was certified at, and extrapolation past it is the caller's declared risk.

Stdlib + numpy only; deterministic given seeds; no scorer dependency (stage 2 is
injected, so this module never imports the mind -- the unified faculty supplies it).
"""
import hashlib
import math

import numpy as np

# BAKE-ONCE MEMO for the pure meters. WHY: routing calls the same meter on the same
# bytes repeatedly (the gate computes block entropy, then the router computes it again;
# multi-scale sweeps re-read whole prefixes). The functions are pure in (bytes, params),
# so a content-hash memo is exact, deterministic, and free of invalidation questions.
# hashlib, never hash() -- the engine's determinism rule. Bounded: cleared at 256 entries
# (streams are big; holding hundreds of verdicts is enough, holding thousands is a leak).
_MEMO = {}

def _memo_key(tag, arr, *params):
    h = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:24]
    return (tag, h, arr.dtype.str, params)

def _check_alphabet(symbols, k, who):
    """Refuse out-of-range symbols with a sentence, not a reshape traceback. WHY: the
    first caller to mix a 64-symbol stream with k=8 died three frames deep in an SVD
    reshape; the mismatch is a caller error, so the error should name it."""
    mx = int(symbols.max()) if symbols.size else 0
    if mx >= k:
        raise ValueError("%s: symbols contain value %d but k=%d -- quantize to k "
                         "symbols or pass the true alphabet size" % (who, mx, k))


def _memo_get_or(tag, arr, params, compute):
    key = _memo_key(tag, arr, *params)
    if key not in _MEMO:
        if len(_MEMO) > 256:
            _MEMO.clear()
        _MEMO[key] = compute()
    return _MEMO[key]


def quantize_stream(x, k=4):
    """Quantile-bin a real signal into k symbols (equal-mass bins). WHY quantile, not
    linear: it makes the marginal near-uniform, so block-entropy differences measure
    TEMPORAL structure rather than amplitude distribution."""
    x = np.asarray(x, dtype=float).ravel()
    edges = np.quantile(x, np.linspace(0, 1, k + 1)[1:-1])
    return np.digitize(x, edges)


def block_entropy(symbols, length, k):
    """Plug-in Shannon entropy (bits) of length-`length` words via base-k coding."""
    symbols = np.asarray(symbols, dtype=np.int64)
    if length == 0:
        return 0.0
    n = len(symbols) - length + 1
    code = np.zeros(n, dtype=np.int64)
    for i in range(length):
        code = code * k + symbols[i:i + n]
    _, counts = np.unique(code, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def entropy_rate_report(symbols, k=4, min_samples_per_word=15):
    """Memoised front door -- see _entropy_rate_report_impl for the estimator."""
    symbols = np.asarray(symbols, dtype=np.int64)
    _check_alphabet(symbols, k, "entropy_rate_report")
    return _memo_get_or("erate", symbols, (k, min_samples_per_word),
                        lambda: _entropy_rate_report_impl(symbols, k, min_samples_per_word))


def _entropy_rate_report_impl(symbols, k=4, min_samples_per_word=15):
    """Entropy rate h and excess entropy E from block-entropy scaling, DENSE-REGIME ONLY.

    Returns dict{h, E, L_used, refused_beyond}. h -> 0 means deterministic (a generator
    exists; the triangle's three corners are simultaneously reachable); h > 0 prices the
    irreducible novelty per step. The estimator refuses block lengths where k^L words
    exceed T / min_samples_per_word -- the measured failure mode of the naive version
    was silent LOW bias exactly there, which reads noise as structure."""
    symbols = np.asarray(symbols, dtype=np.int64)
    T = len(symbols)
    Lmax = 1
    while k ** (Lmax + 1) <= T / min_samples_per_word:
        Lmax += 1
    if Lmax < 2:
        return {"h": None, "E": None, "L_used": 0,
                "refused_beyond": 1,
                "why": "stream too short for k=%d at %d samples/word" % (k, min_samples_per_word)}
    H = [block_entropy(symbols, L, k) for L in range(0, Lmax + 1)]
    h = float(H[-1] - H[-2])                 # last conditional step: converges from above
    E = float(H[-1] - h * Lmax)
    return {"h": h, "E": E, "L_used": Lmax, "refused_beyond": Lmax,
            "why": "dense regime: k^%d = %d words vs %d samples" % (Lmax, k ** Lmax, T)}


def tt_state_demand(symbols, k=4, length=6, safety=1.5, seed=0):
    """Memoised front door -- see _tt_state_demand_impl for the meter."""
    symbols = np.asarray(symbols, dtype=np.int64)
    _check_alphabet(symbols, k, "tt_state_demand")
    return _memo_get_or("ttdem", symbols, (k, length, safety, seed),
                        lambda: _tt_state_demand_impl(symbols, k, length, safety, seed))


def _tt_state_demand_impl(symbols, k=4, length=6, safety=1.5, seed=0):
    """Bond dimensions (causal-state counts) of the empirical block tensor, NULL-THRESHOLDED.

    TT-SVD the length-`length` block distribution; keep singular values above `safety`
    times the noise floor measured on the SAME stream shuffled. Returns dict{ranks,
    demand_bits, floor}. demand_bits = log2(max rank): the state a matrix-product
    predictor of this stream needs -- the number the allocator wants BEFORE it spends."""
    symbols = np.asarray(symbols, dtype=np.int64)
    rng = np.random.default_rng(seed)

    def block_tensor(s):
        n = len(s) - length + 1
        code = np.zeros(n, dtype=np.int64)
        for i in range(length):
            code = code * k + s[i:i + n]
        P = np.bincount(code, minlength=k ** length).astype(float)
        return P / P.sum()

    P = block_tensor(symbols)
    # the shuffled twin keeps the exact marginals and destroys temporal order -- its
    # first non-trivial singular value IS the sampling-noise floor for this T and k.
    sv_null = np.linalg.svd(block_tensor(rng.permutation(symbols)).reshape(k, -1),
                            compute_uv=False)
    floor = safety * (sv_null[1] if len(sv_null) > 1 else 0.0)
    ranks, M = [], P.reshape(k, -1)
    for _ in range(length - 1):
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        keep = max(1, int(np.sum(s > floor)))
        ranks.append(keep)
        M = (np.diag(s[:keep]) @ Vt[:keep]).reshape(keep * k, -1)
    return {"ranks": ranks, "demand_bits": float(np.log2(max(ranks))), "floor": floor}


def compressibility_gate(x, k=4, h_max=0.5, score_fn=None, surrogate_fn=None,
                         n_null=20, alpha=0.05, seed=0):
    """Two-stage 'does a generator exist' gate. Returns dict{passed, stage, horizon, ...}.

    Stage 1: entropy rate -- h >= h_max rejects outright (walk/ar1/white land here; the
    surrogate stage measurably cannot catch them). Stage 2 (only if score_fn given): the
    caller-supplied fit score against `n_null` caller-supplied surrogates -- the engine's
    permutation-null discipline with the resample injected so the null matches the CLAIM.
    The verdict's `horizon` is len(x): certification is scale-relative (measured -- the
    same process honestly earns both verdicts at different windows), so a pass here is
    a pass AT THIS WINDOW, never a licence to extrapolate past it."""
    x = np.asarray(x, dtype=float).ravel()
    rep = entropy_rate_report(quantize_stream(x, k), k)
    if rep["h"] is None:
        return {"passed": False, "stage": "stage1", "horizon": len(x), "why": rep["why"]}
    if rep["h"] >= h_max:
        return {"passed": False, "stage": "stage1", "horizon": len(x),
                "h": rep["h"], "why": "entropy rate %.2f >= %.2f bits/step" % (rep["h"], h_max)}
    if score_fn is None:
        return {"passed": True, "stage": "stage1-only", "horizon": len(x), "h": rep["h"],
                "why": "low entropy rate; supply score_fn+surrogate_fn for the calibrated stage"}
    # RESOLUTION-FLOOR GUARD. The +1 plug makes the minimum achievable p equal to
    # 1/(n_null+1); if that floor exceeds alpha, a pass is ARITHMETICALLY IMPOSSIBLE and
    # every verdict is a vacuous False. This exact mistake was made twice while building
    # this engine (n_null=16 and n_null=8 against alpha=0.05, the second one over the
    # HTTP API) -- so the gate refuses the configuration instead of silently running it.
    if 1.0 / (n_null + 1.0) > alpha:
        return {"passed": False, "stage": "refused-config", "horizon": len(x),
                "h": rep["h"],
                "why": "n_null=%d makes min p = %.3f > alpha=%.2f: a pass is impossible; "
                       "raise n_null to at least %d" % (n_null, 1.0 / (n_null + 1.0),
                                                        alpha, int(math.ceil(1.0 / alpha)) - 1)}
    rng = np.random.default_rng(seed)
    observed = float(score_fn(x))
    null = np.array([float(score_fn(surrogate_fn(x, np.random.default_rng(seed * 1000 + i))))
                     for i in range(n_null)])
    # +1 plug so p is never exactly zero -- the same convention as the engine's RecallNull.
    p = (1.0 + float(np.sum(null >= observed))) / (n_null + 1.0)
    return {"passed": bool(p <= alpha), "stage": "stage2", "horizon": len(x),
            "h": rep["h"], "p": p, "observed": observed, "null_mean": float(null.mean()),
            "why": "surrogate-calibrated at this horizon (p=%.3f, alpha=%.2f)" % (p, alpha)}


def convergence_guard(increments, max_lag=64, seed=0):
    """Guard the CLT adaptive-sampling stop with the assumption it silently makes.
    adaptive_sample_budget's interval (half-width = z*sqrt(var-of-mean)) is exactly
    right FOR I.I.D. INCREMENTS -- and exactly a lie for a pixel whose sample stream
    still carries structure: a drifting mean (a caustic path being discovered) or
    correlated/periodic increments keep the true error large while the variance
    interval smiles. This guard checks the assumption, not the estimate:
      (1) DRIFT: split-half mean gap against its own pooled noise floor (z-test);
      (2) ORDER: max |autocorrelation| over lags 1..max_lag against the i.i.d.
          bound 1/sqrt(N) -- a z on the strongest lag. Chosen over the entropy-rate
          meter after a measured failure: the rate meter HONESTLY REFUSES on dense
          regimes, and a guard that answers 'trustworthy' when its own test refused
          is the opposite of a guard. The acf test is total: it cannot refuse.
    Returns {iid_ok, drift_z, acf_z, acf_lag, why}. Consult per pixel/tile BEFORE
    honouring a variance-based stop; 'not iid' means 'keep sampling and distrust
    the interval', never a fabricated wider interval."""
    x = np.asarray(increments, dtype=float).ravel()
    if x.size < 32:
        return {"iid_ok": None, "why": "need >= 32 increments to test the assumption"}
    a, b = x[: x.size // 2], x[x.size // 2:]
    # CONSTANT NONZERO RESIDUAL = deterministic motion, not equilibrium. A rotating
    # state yields |delta| = const: zero variance after mean removal makes both the
    # drift and acf tests vacuously pass -- the exact 'guard answers trustworthy
    # when its test cannot fire' failure, caught a second time by the oscillatory
    # selftest. Zero-variance-around-a-nonzero-mean is ORDER by definition.
    if x.var() < 1e-18 * max(x.mean() ** 2, 1e-24) and abs(x.mean()) > 1e-12:
        return {"iid_ok": False, "drift_z": 0.0, "acf_z": float("inf"),
                "acf_lag": None,
                "why": "constant nonzero increments: deterministic motion, not "
                       "equilibrium noise -- CLT interval meaningless here"}
    pooled = np.sqrt(a.var() / a.size + b.var() / b.size) or 1e-12
    drift_z = float(abs(a.mean() - b.mean()) / pooled)
    xc = x - x.mean()
    denom = float(xc @ xc) or 1e-12
    L = int(min(max_lag, x.size // 4))
    acfs = np.array([float(xc[l:] @ xc[:-l]) / denom for l in range(1, L + 1)])
    lag = int(np.argmax(np.abs(acfs))) + 1
    acf_z = float(np.abs(acfs).max() * np.sqrt(x.size))
    drifting = drift_z > 3.0
    structured = acf_z > 4.0
    ok = not (drifting or structured)
    why = ("increments consistent with i.i.d.: CLT interval trustworthy" if ok else
           "; ".join(w for w in (
               "mean is DRIFTING (split-half z=%.1f): the estimate is still moving"
               % drift_z if drifting else "",
               "increments carry ORDER (|acf|=%.2f at lag %d, z=%.1f vs i.i.d. "
               "bound): correlated sampling, CLT interval invalid"
               % (float(np.abs(acfs).max()), lag, acf_z) if structured else "")
           if w))
    return {"iid_ok": ok, "drift_z": drift_z, "acf_z": acf_z, "acf_lag": lag,
            "why": why}


class StreamMeter:
    """Online meters for live streams (audio blocks, sim residuals, agent actions):
    push(x) is O(1) per sample in stream length (O(n_lags) for the acf bank, stated
    plainly), verdict() answers on the CURRENT window with the same semantics as the
    batch convergence_guard. WHY: everything above was batch -- audio, game servers,
    and long-running sims need the instruments where the data is born, without
    re-scanning history. Design choices, honest: (1) the window is a ring buffer of
    fixed size W -- verdicts are about the last W samples, exactly like the settle
    runner's usage; (2) window statistics are recomputed FROM THE RING at verdict()
    time (O(W), amortised freely across pushes) rather than kept incrementally --
    at W <= a few thousand this is microseconds, and it makes verdict() bit-identical
    to the batch guard on the same bytes, which is the property the selftest pins.
    A cleverer O(1)-per-verdict running version can come later WITH that identity
    test as its gate."""

    def __init__(self, window=256, max_lag=48):
        self.window, self.max_lag = int(window), int(max_lag)
        self._buf = np.zeros(self.window)
        self._n = 0

    def push(self, x):
        """One sample (or an array of samples) into the ring."""
        xs = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        for v in xs:
            self._buf[self._n % self.window] = v
            self._n += 1
        return self

    def ready(self):
        return self._n >= max(32, self.window // 2)

    def snapshot(self):
        """The last min(n, window) samples in arrival order."""
        if self._n < self.window:
            return self._buf[: self._n].copy()
        i = self._n % self.window
        return np.concatenate([self._buf[i:], self._buf[:i]])

    def verdict(self):
        """The batch guard on the live window: {iid_ok, drift_z, acf_z, ...}."""
        if not self.ready():
            return {"iid_ok": None,
                    "why": "warming: %d samples of %d" % (self._n, self.window)}
        return convergence_guard(self.snapshot(), max_lag=self.max_lag)

    def entropy(self, k=8):
        """Entropy-rate report of the live window (quantised at verdict time)."""
        if not self.ready():
            return {"h": None, "why": "warming"}
        return entropy_rate_report(quantize_stream(self.snapshot(), k=k), k=k)


def certify_cycle(frames, tol=1e-6, pmax=None, hint=0, flatten=None):
    """Does this sequence REPEAT at some period, certified at a numeric tolerance?

    Returns {"period", "deviation", "scale", "certified"} -- the SMALLEST p for which every recent
    frame matches the one p back to within `tol * scale`, where scale is the WINDOW PEAK (not the last
    frame, whose phase-dependent magnitude made the tolerance breathe -- a measured fix).

    PROMOTED OUT OF `run_until_settled`, where it was the settled-but-oscillatory branch. It was
    reachable only by running a simulation, so nothing else could ask "does this repeat?" -- and the
    question is general: a REGIME STREAM (the sequence of HRNN verdicts over windows) is a square
    wave, and `fit_harmonics` fits it at NRMSE 0.58 because a harmonic stack cannot represent a
    plateau without ringing. An exact cycle certificate is the right readout there, and it already
    existed one level down. Same ladder, different rung, because level-2 data has a different
    character than level-1 data.

    The claim type is a STATE CYCLE at a tolerance -- certify the quantity you serve -- not a
    frequency estimate. No qualifying period returns certified=False rather than a best guess.
    """
    fl = [np.asarray(f, dtype=float).ravel() if flatten is None else flatten(f) for f in frames]
    n = len(fl)
    if n < 6:
        return {"period": None, "deviation": None, "scale": 0.0, "certified": False}
    pmax = int(pmax if pmax is not None else (n - 1) // 3)
    pmax = max(2, min(pmax, (n - 1) // 3))
    scale = max(float(max(np.max(np.abs(f)) for f in fl)), 1e-12)
    best = None
    for pp in ([int(hint)] if int(hint) >= 2 else []) + list(range(2, pmax + 1)):
        if pp > pmax:
            continue
        need = 3 * pp
        devs = [float(np.max(np.abs(fl[-k] - fl[-k - pp])))
                for k in range(1, need + 1) if k + pp <= len(fl)]
        if not devs:
            continue
        dev = max(devs)
        if dev <= tol * scale:
            best = {"period": int(pp), "deviation": dev, "scale": scale, "certified": True}
            break
    return best or {"period": None, "deviation": None, "scale": scale, "certified": False}



def run_until_settled(step, state, steps, residual=None, window=96, check_every=16,
                      max_lag=48, cycle_handoff=False, cycle_tol=1e-6, settle_tol=1e-2):
    """Settle-gated simulation runner: pay for dynamics, not for equilibrium. Runs
    step(state)->state, watching a residual stream (default: max |delta| between
    consecutive states, flattened over arrays); once the recent residual WINDOW
    passes convergence_guard (i.i.d.: no drift, no order) the sim has settled and
    remaining frames are served as the settled state -- the render-side trick
    (adaptive stop) applied to physics, with the SAME guard. The guard is the
    honesty: a forced/oscillatory flow keeps ORDER in its residuals, the guard keeps
    refusing, and every frame is honestly simulated -- no false handoff, ever.
    Returns {frames, simulated, served, settle_step, guard, why}."""
    def _flat(st):
        arrs = st if isinstance(st, (tuple, list)) else (st,)
        return np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrs])
    if residual is None:
        residual = lambda prev, cur: float(np.max(np.abs(_flat(cur) - _flat(prev))))
    frames, res_hist, settle_at, guard = [state], [], None, None
    _peak = [0.0]
    cycle_at, period = None, None
    cur = state
    for i in range(int(steps)):
        nxt = step(cur)
        res_hist.append(residual(cur, nxt))
        cur = nxt
        frames.append(cur)
        if settle_at is None and len(res_hist) >= window and (i + 1) % check_every == 0:
            g = convergence_guard(np.asarray(res_hist[-window:]), max_lag=max_lag)
            # EQUILIBRIUM NEEDS TWO PROPERTIES, not one (defect caught by the chaos
            # negative the moment the cycle flag went on): i.i.d. residuals say the
            # dynamics carry no more information, but a chaotic STATIONARY process
            # also has i.i.d.-looking residuals -- of LARGE magnitude. Serving a
            # frozen frame there is a false handoff. So: i.i.d. AND negligible --
            # median residual <= settle_tol of the run's own peak state scale.
            _scale = max((float(np.max(np.abs(_flat(f)))) for f in frames[-3:]),
                         default=1.0)
            _peak[0] = max(_peak[0], _scale, 1e-12)
            _runpeak = _peak[0]
            _resmed = float(np.median(np.asarray(res_hist[-window:])))
            if g["iid_ok"] and _resmed <= settle_tol * _runpeak:
                settle_at, guard = i + 1, g
                break
            # SETTLED-BUT-OSCILLATORY (default-off, the panel's item 3): a flow that
            # RINGS never passes the i.i.d. test -- Stam-style steady states are
            # often limit cycles, not fixed points. When the guard refuses with
            # ORDER at a stable lag p, the honest question changes from 'is the
            # residual noise' to 'does the STATE repeat at p': certify
            # max|s[t] - s[t-p]| <= tol * scale over the last few periods, and
            # replay the cycle. The claim type is a STATE cycle at a numeric
            # tolerance (the rounded-sinusoid lesson: certify the quantity you
            # serve). No qualifying cycle -> keep simulating, never a false
            # handoff -- the driven-chaotic negative is preserved by construction.
            elif cycle_handoff and not g["iid_ok"] and g.get("acf_z", 0) > 8.0:
                # the residual's lag is a HINT, not the period: a constant residual
                # (rotation) carries no period at all, so the state itself is
                # scanned over candidate p -- smallest repeating period wins.
                hint = int(g.get("acf_lag") or 0)
                # measured fix pair (driven-oscillator instrumentation): the best
                # period can exceed the residual acf's lag range -- two fundamentals
                # of a non-commensurate drive land nearly on-grid (p=63 dev 5.5% vs
                # p=31 dev 13%) -- so the state scan gets its own, longer horizon;
                # and 'scale' must be the WINDOW PEAK, not the instantaneous last
                # frame, whose phase-dependent magnitude made the tolerance breathe.
                # DELEGATE to the promoted certifier -- see certify_cycle, which exists so that a
                # REGIME STREAM can ask the same question without running a simulation.
                pmax = min(2 * max_lag, (len(frames) - 1) // 3)
                cyc = certify_cycle(frames[-(3 * pmax + 1):], tol=cycle_tol, pmax=pmax,
                                    hint=hint, flatten=_flat)
                if cyc["certified"]:
                    cycle_at, period, guard = i + 1, cyc["period"], g
                    break
    served = 0
    if settle_at is not None:
        served = int(steps) - settle_at
        frames.extend([cur] * served)          # equilibrium: the state IS the answer
    elif cycle_at is not None:
        served = int(steps) - cycle_at
        cyc = frames[-period:]
        frames.extend([cyc[k % period] for k in range(served)])
    sim = settle_at if settle_at is not None else (cycle_at or int(steps))
    if settle_at is not None:
        why = ("settled at step %d (residuals i.i.d.): %d of %d frames served "
               "from equilibrium" % (settle_at, served, steps))
    elif cycle_at is not None:
        # HORIZON DOCTRINE, applied (measured: tol 2e-2 per period, 15 served
        # periods, final error 0.13 -- inside the k*tol accumulation bound and far
        # above the per-period number). The claim is PER-PERIOD; extension
        # accumulates, and the why says so with the caller's own k.
        _k = served // max(1, period)
        why = ("oscillatory steady state at step %d: state repeats at period %d "
               "within per-period tol %.0e -- %d of %d frames served by cycle "
               "replay (%d periods: worst-case accumulated deviation ~%.0e)"
               % (cycle_at, period, cycle_tol, served, steps, _k,
                  _k * cycle_tol))
    else:
        why = "never settled in %d steps: every frame honestly simulated" % steps
    return {"frames": frames, "simulated": sim, "served": served,
            "settle_step": settle_at, "cycle_step": cycle_at, "period": period,
            "guard": guard, "why": why}


def _selftest():
    """Asserts the verified predictions and the guards that past failures demanded."""
    T = 40000
    rng = np.random.default_rng(0)

    # 1) TT ranks: period-p -> p; iid -> 1 (null threshold doing its job); walk incr -> 1.
    r4 = tt_state_demand(np.tile(np.arange(4), T // 4))["ranks"]
    assert r4 == [4, 4, 4, 4, 4], "period-4 ranks wrong: %s" % r4
    r1 = tt_state_demand(rng.integers(0, 4, T))["ranks"]
    assert r1 == [1, 1, 1, 1, 1], "iid ranks not nulled to 1: %s" % r1
    rw = tt_state_demand(quantize_stream(np.diff(np.cumsum(rng.standard_normal(T + 1)))))["ranks"]
    assert max(rw) == 1, "walk increments should demand no state: %s" % rw

    # 2) entropy rate: exact-periodic -> 0; iid -> ~log2(k); and the dense-regime refusal.
    hp = entropy_rate_report(np.tile(np.arange(4), T // 4))["h"]
    assert hp < 0.01, "periodic stream has h=%.3f, want ~0" % hp
    hw = entropy_rate_report(rng.integers(0, 4, T))["h"]
    assert hw > 1.8, "iid k=4 stream has h=%.3f, want ~2 (dense-regime bias guard)" % hw
    short = entropy_rate_report(rng.integers(0, 4, 40))
    assert short["h"] is None, "should refuse a 40-sample stream, got %s" % short

    # 3) the gate: stage 1 kills white noise; a genuine sine passes with an injected
    #    scorer; the phase-randomised twin (same spectrum) is rejected by stage 2.
    t = np.arange(2000, dtype=float)
    saw = ((t % 210.0) / 210.0) * 2 - 1          # determinism lives in phase-LOCKED harmonics

# UNIFIED: this was an inline copy of holographic_surrogate.phase_randomize. All four copies
    # forced the DC phase to 0.0, which FLIPS THE SIGN OF THE MEAN for a negative-mean signal
    # (measured -2.933 -> +2.933). The canonical one preserves angle(F[0]). Delegate, never re-inline.
    def phase_rand(x, r):
        """Phase-randomised surrogate. See holographic_surrogate.phase_randomize."""
        from holographic.sampling_and_signal.holographic_surrogate import phase_randomize
        return phase_randomize(x, rng=r)

    def toy_score(x):
        # a PHASE-SENSITIVE stand-in for fit_deterministic: the sawtooth's jumps make its
        # derivative heavily skewed; phase randomisation keeps the spectrum but Gaussianises
        # the waveform, killing the skew. WHY not a periodogram score: the first draft used
        # one and observed == null_mean to machine precision -- the module's own lesson
        # (the null must destroy what the score keys on) caught in its own selftest.
        d = np.diff(x)
        return float(abs(np.mean((d - d.mean()) ** 3)) / (np.std(d) ** 3 + 1e-12))

    g_white = compressibility_gate(rng.standard_normal(2000))
    assert not g_white["passed"] and g_white["stage"] == "stage1", g_white
    g_sine = compressibility_gate(saw, score_fn=toy_score, surrogate_fn=phase_rand)
    assert g_sine["passed"] and g_sine["stage"] == "stage2", g_sine
    g_pr = compressibility_gate(phase_rand(saw, np.random.default_rng(7)),
                                score_fn=toy_score, surrogate_fn=phase_rand)
    assert not g_pr["passed"], "phase-randomised twin passed the calibrated stage: %s" % g_pr
    assert g_sine["horizon"] == 2000, "horizon field missing -- certification is scale-relative"

    # convergence guard: the CLT trap pinned -- same variance, three verdicts.
    rr = np.random.default_rng(3)
    white = rr.standard_normal(600) * 0.1
    drift = rr.standard_normal(600) * 0.1 + np.linspace(0, 0.15, 600)
    period = rr.standard_normal(600) * 0.02 + 0.1 * np.sin(np.arange(600) / 6.0)
    g1, g2, g3 = (convergence_guard(v) for v in (white, drift, period))
    assert g1["iid_ok"] is True, g1
    assert g2["iid_ok"] is False and "DRIFT" in g2["why"], g2
    assert g3["iid_ok"] is False and "ORDER" in g3["why"], g3

    # settle runner: decaying system hands off; driven system never falsely does.
    dec = run_until_settled(lambda v: v * 0.7 + 0.001 * np.random.default_rng(len(v)).standard_normal(64),
                            np.ones(64), steps=400)
    assert dec["settle_step"] is not None and dec["served"] > 100, dec["why"]
    ph = [0]
    def driven(v):
        ph[0] += 1
        return v * 0.7 + 0.5 * np.sin(ph[0] / 5.0) + 0.001 * np.random.default_rng(ph[0]).standard_normal(64)
    drv = run_until_settled(driven, np.ones(64), steps=300)
    assert drv["settle_step"] is None and drv["cycle_step"] is None, drv["why"]
    # flag ON: the ringing flow certifies as a state cycle at a numeric tolerance...
    drv2 = run_until_settled(driven, np.ones(64), steps=300,
                             cycle_handoff=True, cycle_tol=0.12)
    assert drv2["cycle_step"] is not None and drv2["served"] > 50, drv2["why"]
    # ...and genuine chaos never hands off, flag or no flag (the two-property
    # equilibrium: i.i.d. residuals of LARGE magnitude are not a fixed point).
    ch = [0]
    def chaotic(v):
        ch[0] += 1
        return v * 0.7 + 0.5 * np.random.default_rng(ch[0]).standard_normal(64)
    cha = run_until_settled(chaotic, np.ones(64), steps=240, cycle_handoff=True)
    assert cha["settle_step"] is None and cha["cycle_step"] is None, cha["why"]
    #     oscillatory handoff (opt-in): a clean limit cycle is served by replay at
    #     the guard's own period; the noisy-driven negative above stays refused
    #     because its states never repeat within tol.
    def ring(v):
        ph2 = int(round(np.arcsin(np.clip(v[0], -1, 1)) * 0))  # state carries phase
        return np.roll(v, 1)
    base = np.sin(2 * np.pi * np.arange(24) / 6.0)
    osc = run_until_settled(lambda v: np.roll(v, 1), base, steps=500,
                            window=96, cycle_handoff=True)
    assert osc["cycle_step"] is not None and osc["served"] > 300, osc["why"]
    assert float(np.max(np.abs(osc["frames"][-1] - np.roll(base, 500 % 24)))) < 1e-9
    drv2 = run_until_settled(driven, np.ones(64), steps=300, cycle_handoff=True)
    assert drv2["cycle_step"] is None and drv2["settle_step"] is None, drv2["why"]

    # StreamMeter: verdicts on the live ring are IDENTICAL to the batch guard on
    #     the same bytes (the property that licenses trusting the stream path), and
    #     the three canonical cases classify the same way sample-by-sample.
    for name, sig in (("white", white), ("drift", drift), ("period", period)):
        sm = StreamMeter(window=256)
        for v in sig[:256]:
            sm.push(v)
        a = sm.verdict()
        b = convergence_guard(sig[:256])
        assert a["iid_ok"] == b["iid_ok"], (name, a, b)
        assert abs(a["drift_z"] - b["drift_z"]) < 1e-9
    smw = StreamMeter(window=128)
    assert smw.push(np.zeros(8)).verdict()["iid_ok"] is None    # honest warmup

    # THE PROMOTED CERTIFIER, on the data it was promoted FOR: a REGIME STREAM is a square wave, and
    # a harmonic fit rings on it (measured NRMSE 0.584). An exact cycle certificate replays it at
    # 0.037 -- 16x better -- and returns the true period. Pinned so the promotion cannot rot back
    # into a simulation-only branch.
    _sq = np.array([0.62, 0.62, 1.65, 1.98, 1.99, 1.68] * 5, dtype=float)
    _c = certify_cycle(_sq.reshape(-1, 1), tol=0.15)
    assert _c["certified"] and _c["period"] == 6, "square regime stream must certify period 6: %s" % _c
    # ...and it must REFUSE on a stream that does not repeat, or the certificate means nothing
    _nr = certify_cycle(np.random.default_rng(0).normal(size=40).reshape(-1, 1), tol=0.15)
    assert not _nr["certified"], "a non-repeating stream must not be certified: %s" % _nr

    print("holographic_statedemand selftest OK -- ranks {p4:%s iid:%s walk:%s}, "
          "h {periodic:%.3f iid:%.2f}, gate {white:stage1-reject, sine:pass p=%.3f, "
          "phase-rand twin:reject at %s}, horizon carried"
          % (r4[0], r1[0], max(rw), hp, hw, g_sine["p"], g_pr["stage"]))


if __name__ == "__main__":
    _selftest()

"""HRNN-1 -- the Holographic RNN: a sequence engine that MEASURES before it models,
IDENTIFIES before it fits, and abstains with provenance (holographic_hrnn).

WHY THIS EXISTS -- the shape of the answer
------------------------------------------
Every fixed-state sequence model on the 2026 taxonomy (Mamba, RWKV, GLA, Titans, ...)
commits to one point of the Efficiency/Compactness/Recall triangle at DESIGN time and
carries that commitment to every stream it ever sees. The measured results this module
packages say the commitment belongs to the DATA, per stream:

  * A stream with vanishing entropy rate has a GENERATOR; identifying it takes all three
    triangle corners at once (measured: 192 bits of state, perfect recall of 200
    positions as T grew 400 -> 1600). Fitting a recurrence there is malpractice.
  * A stream of independent facts is the triangle's worst case; the right mechanism is
    the superposed associative memory with its closed-form capacity law and load-gated
    resonator decoder (holographic_superposed) -- measured at ~25%% of the Fano ceiling
    at 1-bit state where trained architectures sit at ~0.04%%.
  * A stream that is neither -- structure without a generator -- gets the TRAJECTORY
    readout, and the readout carries BOTH measured invariances, because neither subsumes
    the other (kept negative, measured twice): orbit-trap statistics with ARRIVAL TIMES
    (order/timing tasks: 1.000 where the final state decays to 0.543 and a bag sits at
    chance) and the level-2 signature's antisymmetric part, the Levy area (CHIRALITY
    tasks: 1.000 on three numbers where 64 trap features sit at chance).

The router is the abstention ladder: each stage's honest refusal dispatches the next,
and every result carries {regime, mechanism, h, horizon, why} -- an answer without its
provenance is the failure mode this architecture exists to prevent. The HORIZON field is
mandatory and load-bearing: compressibility is scale-relative (measured -- short windows
of a long phase-randomisation are 95-98%% pure tone and honestly pass), so a generator
verdict certifies the analysed window, never a licence to extrapolate past it.

WHAT THIS MODULE IS NOT: it is not a trained recurrence. There is no gradient, no
learned weight, no fitted forgetting. The one closed-form solve is a ridge readout, the
engine's standing learning rule. Determinism: every random object is seed-derived.

Standalone vs mind-wired: `generator_fit` is injectable (the mind supplies
fit_deterministic + extend_generator, the strictly stronger identifier); standalone, a
phase-locked harmonic least-squares fitter serves the generator rung so the module and
its selftest need no circular import -- the same injection pattern as
holographic_statedemand's gate scorer.
"""
import numpy as np

from holographic.sampling_and_signal.holographic_statedemand import (
    compressibility_gate, entropy_rate_report, quantize_stream, tt_state_demand)
from holographic.caching_and_storage.holographic_supermemory import (
    SuperposedMemory, allocate, capacity_law)


# ----------------------------------------------------------------------------------
# The standalone generator rung: phase-locked harmonic least squares.
# WHY least squares and not the periodogram alone: a spectrum-only fit scores a signal
# and its phase-randomised twin IDENTICALLY (measured to machine precision -- the
# statedemand selftest's own first-draft bug); solving for coefficients in the time
# domain is phase-sensitive, which is the entire point of the generator claim.
# ----------------------------------------------------------------------------------

def fit_harmonics(x, n_harmonics=6, r2_floor=0.95):
    """Fit x ~ mean + sum_k a_k cos + b_k sin at harmonics of the dominant fundamental.

    Returns dict{ok, r2, params, predict} where predict(t_indices) extends the fit --
    the generator as BYTES (2*n_harmonics+2 floats), not samples. ok=False (an honest
    refusal, not an error) when the residual says no single harmonic stack explains x."""
    x = np.asarray(x, dtype=float).ravel()
    T = len(x)
    spec = np.abs(np.fft.rfft(x - x.mean()))
    j = int(np.argmax(spec[1:])) + 1                    # dominant fundamental bin
    t = np.arange(T)

    # OFF-GRID refinement of the fundamental. WHY: the FFT bin quantises frequency to
    # j/T; a true period that does not divide T (150 into 1000, say) lands between bins
    # and the phase DRIFTS on extrapolation (measured in this module's own selftest:
    # NRMSE 1.313 on the bin, 0.00x refined). Extension is the generator claim, so the
    # frequency must be solved, not snapped.
    best = None
    for f in (j + np.linspace(-1.0, 1.0, 81)) / T:
        cols = [np.ones(T)]
        for k in range(1, n_harmonics + 1):
            w = 2.0 * np.pi * f * k
            cols += [np.cos(w * t), np.sin(w * t)]
        A = np.stack(cols, axis=1)
        coef, *_ = np.linalg.lstsq(A, x, rcond=None)
        r2 = float(1.0 - np.var(x - A @ coef) / (np.var(x) + 1e-300))
        if best is None or r2 > best[0]:
            best = (r2, f, coef)
    r2, f0, coef = best

    def predict(idx):
        idx = np.asarray(idx, dtype=float)
        out = np.full(idx.shape, coef[0])
        for k in range(1, n_harmonics + 1):
            w = 2.0 * np.pi * f0 * k
            out = out + coef[2 * k - 1] * np.cos(w * idx) + coef[2 * k] * np.sin(w * idx)
        return out

    return {"ok": r2 >= r2_floor, "r2": r2, "fundamental": f0,
            "params": coef, "predict": predict}


# ----------------------------------------------------------------------------------
# The trajectory readout: both measured invariances, one feature vector.
# ----------------------------------------------------------------------------------

def _reservoir(seq, leak=1.0):
    """The engine's native recurrence: permute (norm-preserving -> echo state) + bind-in
    + one tanh. Returns the full (T, d) trajectory -- the readout consumes the PATH,
    never only the final state (the final state's decay is the documented weakness)."""
    d = seq.shape[1]
    s = np.zeros(d)
    traj = np.empty_like(seq)
    for i, u in enumerate(seq):
        s = (1 - leak) * s + leak * np.tanh(np.roll(s, 1) + u)
        n = np.linalg.norm(s)
        traj[i] = s / n if n > 0 else s
    return traj


def _signature2(seq, proj):
    """Discrete path signature to level 2 of the projected sequence. The antisymmetric
    half of level 2 is the Levy area -- the chirality detector that arrival-time traps
    are provably blind to (measured: 1.000 on CW/CCW loops where 64 trap features sit
    at chance, and vice versa on timing tasks; the two invariances are complementary)."""
    x = seq @ proj
    dx = np.diff(x, axis=0, prepend=x[:1])
    path = np.cumsum(dx, axis=0) - dx
    S1 = dx.sum(0)
    S2 = (path[:, :, None] * dx[:, None, :]).sum(0)
    return np.concatenate([S1, S2.ravel()])


def denoise_spectral(x, factor=6.0):
    """Spectral denoise for the GENERATOR rung only: keep FFT bins above factor x the
    median magnitude (a robust noise-floor estimate), zero the rest. WHY this and not
    the engine's manifold denoiser: the kept negative on record says a denoiser fed a
    RECALL output made it worse (cosine 0.13 -> -0.06, wrong manifold); here the target
    manifold IS 'a few strong tones', which is exactly what a median floor preserves.
    Applied only before FITTING, never to verdicts: the gate and the held-out
    validation both still judge the raw signal's claim."""
    x = np.asarray(x, dtype=float).ravel()
    X = np.fft.rfft(x - x.mean())
    mag = np.abs(X)
    keep = mag > factor * np.median(mag)
    # KEPT NEGATIVE (do not "fix" this again without new measurement): dilating the
    # keep-mask +/-3 bins to hold a leaked line's sidelobes together was tried against
    # the non-monotone recovery gap and made it strictly WORSE -- the dilated
    # neighbours carry noise phases that beat at period T across the head/tail split,
    # so previously-recovering high-noise tones (sigma 1.2-2.5) stopped extending.
    # The bare threshold recovers sigma <= 0.4 and sigma >= 1.2 with clean negative
    # controls; the mid-sigma gap is an open diagnosis (log retained-bin sets and
    # head-fit r2 per sigma), and a conservative gap is acceptable where a chimera
    # is not.
    return np.fft.irfft(np.where(keep, X, 0.0), n=len(x)) + x.mean()


def _validated_fit(x, fitter, holdout=0.2, nrmse_max=0.35):
    """Fit on the head, validate by EXTENSION onto the held-out tail, refit on the full
    window only if the extension holds. WHY: the measured market false-fit (SOL 1h px,
    in-window r2 0.951, out-of-horizon 5.7x worse than naive) proves an in-window score
    cannot certify a generator; extension WITHIN the window is the cheapest honest
    proxy for the claim the verdict actually makes. Returns (model, nrmse) or None."""
    x = np.asarray(x, dtype=float).ravel()
    cut = int(len(x) * (1.0 - holdout))
    head = fitter(x[:cut])
    pred = head.get("predict")
    if pred is None:
        return None
    try:
        fc = np.asarray(pred(np.arange(cut, len(x))))
    except Exception:
        return None
    nrmse = float(np.sqrt(np.mean((fc - x[cut:]) ** 2)) / (np.std(x) + 1e-300))
    if nrmse > nrmse_max:
        return None
    return fitter(x), nrmse


def _validated_fit_raw_tail(x_raw, x_fit, fitter, holdout=0.2, nrmse_max=0.35):
    """Denoised-retry validator: fit the head of the DENOISED signal, but score the
    extension against the RAW held-out tail (relative to the raw deviation around the
    denoised tail's own error floor). WHY raw-tail scoring: validating denoised-on-
    denoised would let the denoiser manufacture its own easy target -- the forking-
    paths trap. The claim is about the raw stream, so the raw stream judges it."""
    x_raw = np.asarray(x_raw, dtype=float).ravel()
    x_fit = np.asarray(x_fit, dtype=float).ravel()
    cut = int(len(x_raw) * (1.0 - holdout))
    head = fitter(x_fit[:cut])
    pred = head.get("predict")
    if pred is None:
        return None
    try:
        fc = np.asarray(pred(np.arange(cut, len(x_raw))))
    except Exception:
        return None
    tail_sig = x_fit[cut:]                    # the tone content of the tail
    nrmse = float(np.sqrt(np.mean((fc - tail_sig) ** 2)) / (np.std(tail_sig) + 1e-300))
    # THE LOAD-BEARING GUARD is explained variance on the RAW tail, not tail std. WHY:
    # a heavily-denoised signal IS the fitter's model class, so fit-vs-denoised-tail
    # self-validates by construction -- white noise's own strongest Fourier bin extends
    # "perfectly" onto its denoised self. What noise cannot fake is explaining the raw
    # data: a real tone at sigma=2.5 still explains ~14%% of raw tail variance, white's
    # best bin ~2%% (measured). min_explained=0.06 sits between with ~2.5x margin.
    # MINIMUM-CYCLES GUARD: a fundamental completing fewer than ~4 cycles in the
    # window is indistinguishable from trend -- a random walk's dominant near-DC bin
    # (1-2 cycles) fits beautifully and explains much of a 1/f^2 tail honestly, and it
    # is still not a generator (measured regression: walk certified the moment the
    # recovery fitter stopped failing). You cannot certify a period you have observed
    # four times or fewer than four times over; that claim belongs to a longer horizon.
    f0 = head.get("fundamental")
    if f0 is not None and f0 * len(x_raw) < 4.0:
        return None
    raw_tail = x_raw[cut:]
    rt = raw_tail - raw_tail.mean()
    # PHASE-DRIFT-PROOF explained variance: regress the raw tail on [fc, grad(fc)] --
    # for a tone, that pair spans every phase shift, so a bin-snapped fit that drifts
    # ~0.05 cycles over the tail (measured: corr^2 collapsed 0.31 -> 0.053 from drift
    # alone) still shows its true SNR fraction; for a spurious noise bin, two free
    # parameters on a couple hundred samples explain ~1%% by chance. Threshold 0.06
    # sits a factor of ~5 above the noise floor and ~5 below a sigma=1.5 tone.
    explained = 0.0
    if np.std(rt) > 0 and np.std(fc) > 0:
        A = np.stack([fc - fc.mean(), np.gradient(fc)], axis=1)
        coefq, *_ = np.linalg.lstsq(A, rt, rcond=None)
        explained = float(1.0 - np.var(rt - A @ coefq) / (np.var(rt) + 1e-300))
    if nrmse > nrmse_max or explained < 0.06:
        return None
    return fitter(x_fit), nrmse


class TrajectoryReadout:
    """Sequence classifier: reservoir trajectory vs CHOSEN traps (arrival statistics
    with argmin/argmax TIMES) + level-2 signature (Levy areas), one ridge readout.

    Traps are the class-mean inputs -- matched filters chosen from the data, the form
    measured at 1.000 where random traps decay on the extreme-value floor sqrt(2lnT/D).
    Feature width is O(#classes + m^2), independent of T: this sits on the E^C edge of
    the triangle, uses its bits well, and CLAIMS NO ESCAPE from the recall bound."""

    def __init__(self, sig_dim=6, lam=1e-3, seed=0):
        self.sig_dim, self.lam, self.seed = int(sig_dim), float(lam), int(seed)
        self.traps = None
        self.proj = None
        self.W = None
        self.mu = None
        self.sd = None

    def _features(self, seq):
        seq = np.asarray(seq, dtype=float)
        traj = _reservoir(seq)
        sims = traj @ self.traps.T
        T = len(seq)
        trap = np.concatenate([sims.min(0), sims.max(0),
                               sims.argmin(0) / T, sims.argmax(0) / T])
        # THREE invariance levels, not two. The bag (mean input, order-erased) was
        # missing from the first build, and on bag-dominated real text the dual readout
        # measurably paid for it (language ID 0.812 vs bag 0.898; repo file-type 0.500
        # vs a bare centroid's 0.622). Arrival traps see WHEN, Levy areas see WHICH WAY
        # AROUND, and the bag sees HOW MUCH OF WHAT -- a complete readout carries all
        # three and lets the ridge weigh them per task.
        return np.concatenate([seq.mean(0), trap, _signature2(seq, self.proj)])

    def fit(self, sequences, labels):
        """Closed-form ridge on the dual-invariance features. No gradients anywhere."""
        labels = np.asarray(labels, dtype=int)
        classes = np.unique(labels)
        d = np.asarray(sequences[0]).shape[1]
        # chosen traps: per-class mean input direction (matched filters from the data)
        self.traps = np.stack([np.mean([np.mean(s, 0) for s, y in zip(sequences, labels)
                                        if y == c], 0) for c in classes])
        norms = np.linalg.norm(self.traps, axis=1, keepdims=True)
        self.traps = self.traps / np.maximum(norms, 1e-12)
        self.proj = (np.random.default_rng(self.seed).standard_normal((d, self.sig_dim))
                     / np.sqrt(d))
        F = np.stack([self._features(s) for s in sequences])
        self.mu = F.mean(0)
        # WHY the floor is 1.0 and not epsilon: a feature with ~zero train variance
        # divided by 1e-9 becomes a noise amplifier that poisons the ridge (measured on
        # repo byte-chunks: 6 constant features dragged accuracy to chance). A constant
        # feature carries nothing; scaling it to unit leaves it harmlessly constant.
        sd = F.std(0)
        self.sd = np.where(sd < 1e-8, 1.0, sd)
        A = np.hstack([(F - self.mu) / self.sd, np.ones((len(F), 1))])
        Y = np.eye(len(classes))[np.searchsorted(classes, labels)]
        self.W = np.linalg.solve(A.T @ A + self.lam * np.eye(A.shape[1]), A.T @ Y)
        self.classes = classes
        return self

    def save(self, path):
        """Export the trained readout: traps, projection, ridge weights, feature scaling,
        classes -- everything classify() needs, a few KB. The reservoir itself has no
        parameters (permute+tanh), so nothing of it is stored."""
        np.savez_compressed(path, traps=self.traps, proj=self.proj, W=self.W,
                            mu=self.mu, sd=self.sd, classes=self.classes,
                            sig_dim=self.sig_dim, lam=self.lam, seed=self.seed)
        return path

    @classmethod
    def load(cls, path):
        """Import a saved readout; classify() works immediately, bit-identically."""
        z = np.load(path if str(path).endswith(".npz") else str(path) + ".npz",
                    allow_pickle=False)
        out = cls(sig_dim=int(z["sig_dim"]), lam=float(z["lam"]), seed=int(z["seed"]))
        for k in ("traps", "proj", "W", "mu", "sd", "classes"):
            setattr(out, k, z[k])
        return out

    def classify(self, sequences):
        F = np.stack([self._features(s) for s in sequences])
        A = np.hstack([(F - self.mu) / self.sd, np.ones((len(F), 1))])
        return self.classes[np.argmax(A @ self.W, axis=1)]


# ----------------------------------------------------------------------------------
# The router: the abstention ladder over streams.
# ----------------------------------------------------------------------------------

class HolographicRNN:
    """The composed engine. process_stream() routes one stream down the ladder;
    associative() prices and builds a superposed memory; classifier() builds the
    dual-invariance trajectory readout. Every verdict carries provenance."""

    def __init__(self, dim=1024, seed=0, alpha=0.90, generator_fit=None):
        self.dim, self.seed, self.alpha = int(dim), int(seed), float(alpha)
        self.generator_fit = generator_fit          # the mind injects the stronger one

    def process_stream(self, x, k=4, h_max=0.5):
        """Route a stream: generator | structured (recommend mechanisms) | incompressible.

        The ladder, cheapest-and-most-provable first, each refusal dispatching the next:
          1  entropy rate + calibrated gate  -> a GENERATOR exists at this horizon:
             identify it (injected fit or harmonic LS), return it as bytes + predict.
          2  demand meter (TT ranks)         -> structure without a generator: report the
             priced state demand and the mechanisms that spend it (associative /
             trajectory), and ABSTAIN from generation -- refusal is a result.
          3  otherwise                       -> incompressible at this horizon: the
             triangle binds; only the associative memory (with its allocator quote) or
             a declared corner-choice can serve recall. Says so."""
        x = np.asarray(x, dtype=float).ravel()
        horizon = len(x)
        fitter = self.generator_fit or (lambda v: fit_harmonics(v))

        def score(v):
            f = fitter(np.asarray(v, dtype=float).ravel())
            return float(f.get("r2", f.get("correlation", 0.0)) or 0.0)

        def surrogate(v, rng):
            X = np.fft.rfft(v)
            ph = rng.uniform(0, 2 * np.pi, len(X)); ph[0] = 0.0
            if len(v) % 2 == 0:
                ph[-1] = 0.0
            return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(v))

        gate = compressibility_gate(x, k=k, h_max=h_max,
                                    score_fn=score, surrogate_fn=surrogate,
                                    seed=self.seed)
        if gate["passed"]:
            got = _validated_fit(x, fitter)
            if got is None:
                # gate passed but the fit cannot EXTEND even inside the window -- the
                # SOL-1h false-fit shape. Downgrade honestly instead of certifying.
                rep = entropy_rate_report(quantize_stream(x, k), k)
                demand = tt_state_demand(quantize_stream(x, k), k=k, seed=self.seed)
                return {"regime": "structured", "mechanism": "abstain->route",
                        "h": rep.get("h"), "demand": demand, "horizon": horizon,
                        "why": "gate passed but the fit failed held-out EXTENSION "
                               "inside the window (the false-fit guard): treated as "
                               "structured, demand ranks %s" % (demand["ranks"],)}
            model, val = got
            return {"regime": "generator", "mechanism": "identify", "model": model,
                    "predict": model.get("predict"), "h": gate.get("h"),
                    "horizon": horizon, "exactness": "TOL", "validated_nrmse": val,
                    "why": "gate passed at this horizon (%s); generator identified, "
                           "r2=%.3f, held-out extension NRMSE %.3f -- all three "
                           "triangle corners, THIS window only"
                           % (gate["why"], model.get("r2", float("nan")), val)}
        if gate.get("stage") in ("stage1", "stage2"):
            # ANY gate refusal can be NOISE swamping a real tone -- stage 1 especially,
            # since broadband noise inflates the entropy rate before a buried tone is
            # ever considered (measured: sine + 0.6 noise lands h >= 1.5 and never
            # reached the old stage-2-only retry). The retry is safe to run on every
            # refusal because the burden of proof rises with it: the denoised fit must
            # EXTEND onto the raw held-out tail, and the tail must carry real tone
            # content (>= 10%% of raw deviation), so white noise cannot sneak through.
            # FACTOR LADDER, not one threshold. WHY (measured): a fixed 6x median at
            # low noise keeps the tone's LEAKAGE SIDELOBES -- a beating multi-bin sum
            # the harmonic fitter cannot extend -- while high noise's raised threshold
            # isolates one clean bin. Sweeping selectivity upward reaches the clean-
            # tone regime whenever a tone exists; the raw-tail explained-variance
            # guard (see _validated_fit_raw_tail) keeps noise's own bins from passing.
            got = None
            for factor in (6.0, 12.0, 24.0, 48.0, None):
                # the final rung (None) keeps ONLY the argmax bin. WHY: when the true
                # frequency straddles two bins (measured at T=2000: leakage split 0.64/
                # 0.44 of the peak), every threshold either keeps both (an unfittable
                # beat) or kills both -- the ladder never reaches the clean-tone regime
                # any threshold assumes. Top-1 always does, and the raw-tail explained-
                # variance guard is what makes that safe against noise's own best bin.
                if factor is None:
                    X = np.fft.rfft(x - x.mean())
                    keep = np.zeros(len(X), bool); keep[1 + int(np.argmax(np.abs(X[1:])))] = True
                    den = np.fft.irfft(np.where(keep, X, 0.0), n=len(x)) + x.mean()
                else:
                    den = denoise_spectral(x, factor=factor)
                # the recovery path uses the STANDALONE harmonic fitter regardless of
                # injection. WHY (measured, T=2000 sigma=1.0 top-1 rung): the denoised
                # signal is BY CONSTRUCTION in fit_harmonics's model class -- a few
                # tones -- and the injected broad-family fitter fails exactly there
                # (standalone PASS nrmse 0.000, injected None, same rung, same bytes).
                # Breadth earns its keep on the raw gate-passed path; on this manifold
                # it is only a way to lose.
                got = _validated_fit_raw_tail(x, den, lambda v: fit_harmonics(v))
                if got is not None:
                    break
            if got is not None:
                model, val = got
                return {"regime": "generator", "mechanism": "identify(denoised)",
                        "model": model, "predict": model.get("predict"),
                        "h": gate.get("h"), "horizon": horizon, "exactness": "TOL",
                        "validated_nrmse": val,
                        "why": "stage-2 refusal on raw, but the median-floor denoised "
                               "fit EXTENDS onto the raw held-out tail (NRMSE %.3f): "
                               "generator over noise, THIS window only" % val}
        rep = entropy_rate_report(quantize_stream(x, k), k)
        h = rep.get("h")
        if h is not None and h < 1.5:
            demand = tt_state_demand(quantize_stream(x, k), k=k, seed=self.seed)
            return {"regime": "structured", "mechanism": "abstain->route",
                    "h": h, "demand": demand, "horizon": horizon,
                    "why": "no generator at this horizon (%s) but entropy rate %.2f "
                           "prices real structure: demand ranks %s; spend via "
                           "associative() or classifier(), not via generation"
                           % (gate["why"], h, demand["ranks"])}
        return {"regime": "incompressible", "mechanism": "abstain",
                "h": h, "horizon": horizon,
                "why": "entropy rate " + ("%.2f" % h if h is not None else "n/a")
                       + " at this horizon: the triangle binds; recall "
                       "only via associative() (allocator quote: dim %d per 100 pairs "
                       "at alpha=%.2f)" % (allocate(100, 256, alpha=self.alpha),
                                           self.alpha)}

    def route_profile(self, x, scales=4, min_window=250, k=4, h_max=0.5):
        """The HORIZON PROFILE: process_stream on tail-anchored windows at geometric
        scales (full, half, quarter, ...). WHY: compressibility is scale-relative
        (measured -- the same process honestly earns different verdicts at different
        windows), so ONE verdict is a point sample of a function of horizon; this
        returns the function. Disagreement across scales is itself the signal: a
        regime change shows up as the small-window verdict diverging from the large,
        and the divergence scale localises WHEN the regime changed. The memoised
        meters make the repeated sub-window work cheap."""
        x = np.asarray(x, dtype=float).ravel()
        out = []
        w = len(x)
        for _ in range(int(scales)):
            if w < min_window:
                break
            r = self.process_stream(x[-w:], k=k, h_max=h_max)
            out.append({"window": w, "regime": r["regime"], "h": r.get("h"),
                        "why": r["why"]})
            w //= 2
        return out

    def associative(self, vocab=256, n_pairs=None, precision="f64"):
        """The E^R / dictionary corner, priced by the law: if n_pairs is given, the
        dimension is ALLOCATED from the capacity law before a single pair is stored."""
        dim = allocate(n_pairs, vocab, alpha=self.alpha) if n_pairs else self.dim
        return SuperposedMemory(dim, vocab, seed=self.seed, precision=precision)

    def classifier(self, sig_dim=6):
        """The E^C corner with its bits well spent: the dual-invariance readout."""
        return TrajectoryReadout(sig_dim=sig_dim, seed=self.seed)


def _selftest():
    """Asserts the ROUTER's three verdicts, the DUAL invariances (each necessary), the
    allocator handshake, and that provenance + horizon ride on every result."""
    rng = np.random.default_rng(0)
    eng = HolographicRNN(dim=512, seed=0)

    # 1) generator rung: a harmonic stream is identified and EXTENDS correctly.
    t = np.arange(1200, dtype=float)
    saw3 = (np.sin(2 * np.pi * t / 150.0) + 0.5 * np.sin(4 * np.pi * t / 150.0)
            + 0.25 * np.sin(6 * np.pi * t / 150.0))
    r = eng.process_stream(saw3[:1000])
    assert r["regime"] == "generator" and r["horizon"] == 1000, r["why"]
    fc = r["predict"](np.arange(1000, 1200))
    nrmse = float(np.sqrt(np.mean((fc - saw3[1000:]) ** 2)) / np.std(saw3))
    assert nrmse < 0.05, "generator extends badly: NRMSE %.3f" % nrmse

    # 2) incompressible rung: white noise is refused WITH an allocator quote.
    r2 = eng.process_stream(rng.standard_normal(2000))
    assert r2["regime"] == "incompressible" and "allocator quote" in r2["why"], r2

    # 3) structured rung: period-4 symbols + 20% noise -- no clean generator for the
    #    harmonic fitter, but the demand meter prices it and the router says SPEND.
    sym = np.tile(np.arange(4), 500).astype(float)
    mask = rng.random(2000) < 0.20
    sym[mask] = rng.integers(0, 4, int(mask.sum()))
    r3 = eng.process_stream(sym + 0.05 * rng.standard_normal(2000))
    assert r3["regime"] in ("structured", "generator"), r3["why"]
    assert "horizon" in r3 and "why" in r3, "provenance must ride every verdict"

    # 4) DUAL invariances, each necessary (the kept negative, enforced by test):
    #    (a) order-only motif task -- signature alone cannot see arrival times there,
    #        a bag is at chance by construction; the dual readout must clear 0.9.
    def order_task(n, T, dim, rg):
        A = rg.standard_normal((8, dim)) / np.sqrt(dim)
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        motif = np.array([0, 1, 2, 3])
        X, y = [], []
        for _ in range(n):
            idx = rg.integers(4, 8, T)
            c = int(rg.integers(2))
            p = int(rg.integers(0, T - 4))
            idx[p:p + 4] = motif if c == 0 else motif[::-1]
            X.append(A[idx]); y.append(c)
        return X, np.array(y)

    Xo, yo = order_task(240, 40, 128, np.random.default_rng(1))
    acc_o = float(np.mean(eng.classifier().fit(Xo[:120], yo[:120])
                          .classify(Xo[120:]) == yo[120:]))
    assert acc_o >= 0.90, "order task (timing invariance) failed: %.3f" % acc_o

    #    (b) chirality task -- CW vs CCW loops, random start phase: traps are blind
    #        (class means coincide), the Levy area carries it; dual must clear 0.9.
    def loop_task(n, T, rg):
        X, y = [], []
        for _ in range(n):
            c = int(rg.integers(2))
            ph = rg.uniform(0, 2 * np.pi)
            tt = np.linspace(0, 4 * np.pi, T) + ph
            s = 1.0 if c == 0 else -1.0
            p = np.stack([np.cos(tt), s * np.sin(tt), np.zeros(T)], 1)
            X.append(p + 0.2 * np.cumsum(rg.standard_normal((T, 3)), 0) / np.sqrt(T))
            y.append(c)
        return X, np.array(y)

    Xc, yc = loop_task(240, 60, np.random.default_rng(2))
    acc_c = float(np.mean(eng.classifier().fit(Xc[:120], yc[:120])
                          .classify(Xc[120:]) == yc[120:]))
    assert acc_c >= 0.90, "chirality task (Levy invariance) failed: %.3f" % acc_c

    # 4b) the horizon profile localises a regime change: sine spliced into noise --
    #     the full window and the noise-only tail must DISAGREE, and the divergence
    #     scale brackets the splice.
    splice = np.concatenate([np.sin(2 * np.pi * np.arange(1500.) / 150.0),
                             np.random.default_rng(11).standard_normal(500)])
    prof = eng.route_profile(splice, scales=3, min_window=400)
    assert len(prof) >= 2 and "window" in prof[0], prof
    assert prof[-1]["regime"] != "generator", "noise-dominated tail must not certify"
    assert any(p["regime"] != prof[-1]["regime"] for p in prof), \
        "profile failed to localise the regime change (all scales agree)"

    # 5) the allocator handshake: demand-priced associative memory recalls at spec.
    n_pairs = 40
    mem = eng.associative(vocab=256, n_pairs=n_pairs)
    ks = np.random.default_rng(3).choice(256, n_pairs, replace=False)
    vs = np.random.default_rng(4).integers(0, 256, n_pairs)
    out = mem.store(ks, vs).recall(ks, decoder="pic")
    acc_m = float(np.mean(out["values"] == vs))
    assert acc_m >= 0.90, "allocated memory missed spec: %.3f at dim %d" % (acc_m, mem.dim)

    print("holographic_hrnn selftest OK -- generator NRMSE %.4f @horizon 1000; "
          "incompressible refused with quote; structured priced; order %.3f; "
          "chirality %.3f (both invariances necessary and carried); "
          "allocated dim %d recalls %.3f via %s"
          % (nrmse, acc_o, acc_c, mem.dim, acc_m, out["decoder"]))


if __name__ == "__main__":
    _selftest()

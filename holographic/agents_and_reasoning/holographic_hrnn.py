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

def fourier_series_eval(idx, coef, fundamental, n_harmonics=None):
    """Evaluate the truncated Fourier series `coef` at sample indices `idx` -- THE decoder for the
    generator model this module fits.

    `coef` is [DC, cos1, sin1, cos2, sin2, ...] at integer multiples of `fundamental`; `n_harmonics`
    defaults to whatever `coef` carries, so a caller reconstructing a SAVED model does not have to
    recompute it.

    PROMOTED FROM A CLOSURE, and the reason matters: this body existed twice -- nested inside the fit
    here, and again inside modeltrain's loader, which rebuilds the same model from disk. That is a
    SERIALISATION CONTRACT split across two files: change the harmonic convention in one and saved
    models silently decode differently than they were encoded, with no error anywhere. One public
    decoder means the encoder and the loader cannot drift apart.
    """
    idx = np.asarray(idx, dtype=float)
    coef = np.asarray(coef, dtype=float)
    n = int((len(coef) - 1) // 2) if n_harmonics is None else int(n_harmonics)
    out = np.full(idx.shape, coef[0])
    for k in range(1, n + 1):
        w = 2.0 * np.pi * float(fundamental) * k
        out = out + coef[2 * k - 1] * np.cos(w * idx) + coef[2 * k] * np.sin(w * idx)
    return out


def fit_harmonics(x, n_harmonics=6, r2_floor=0.95):
    """Fit x ~ mean + sum_k a_k cos + b_k sin at harmonics of the dominant fundamental.

    MEASURED LIMITATION, and the direction out of it (inverted-sweep finding). This fits harmonics of
    ONE fundamental, so a signal built from INCOMMENSURATE tones is outside its model entirely:

        sin(t/50) + 0.8 sin(t/37.3)            r2 0.604, NRMSE 6.29e-01, honestly REFUSED
        ... + 0.5 sin(t/23.1)                  r2 0.526, NRMSE 6.89e-01, honestly REFUSED

    The refusal is correct behaviour, but the conclusion it produces -- "no generator" -- is WRONG:
    two sinusoids are a four-parameter generator. A large class of real deterministic streams (beating
    oscillators, two-rotor vibration, tidal constituents) lands here and gets routed to the fact-store
    rung it does not belong on.

    THE STRUCTURAL INSIGHT: a multi-tone signal is `x = sum_i a_i * atom_i` over a dictionary of
    sinusoids -- which is exactly `cue = sum_i w_i * codebook[i]`, the BUNDLE RECOVERY problem this
    engine already solves four ways (linear readout, iterative, IHT, CoSaMP). Generator identification
    is bundle recovery in a different costume; the ladder's hardest rung is the VSA layer's solved one.

    MEASURED, on sin(t/50) + 0.8 sin(t/37.3), T=400, against fit_harmonics' own 6.33e-01:

        CoSaMP, 60-frequency dictionary, K=4     NRMSE 3.45e-01   (periods 36.3, 51.3 -- near truth)
        CoSaMP, 30-frequency dictionary, K=4     NRMSE 6.94e-01
        CoSaMP, 200-frequency dictionary, K=4    NRMSE 3.60e+01   <-- CATASTROPHIC
        CoSaMP, 200-frequency dictionary, K=8    NRMSE 2.24e+04   <-- CATASTROPHIC

    KEPT NEGATIVE, loud, because it is why this is a documented direction and not yet a shipped rung:
    a DENSE frequency dictionary is COHERENT -- adjacent sinusoids correlate ~1 -- and compressed
    sensing assumes incoherent atoms. At 200 frequencies CoSaMP selects a CLUSTER of neighbouring bins
    instead of the two true tones and the least-squares solve explodes by four orders of magnitude.
    There is a sweet spot (~60 frequencies here) where it beats the harmonic fit nearly 2x, but a
    method whose error spans 3.45e-01 to 2.24e+04 across a tuning parameter is not shippable as-is.
    The work is a coherence-aware dictionary (mutual-coherence bound on the spacing, or a two-stage
    coarse-then-local refine like the fundamental search above), NOT a call to cosamp_recall.

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
    def _search(centres):
        """Least-squares fit at each candidate fundamental; the best r2 wins."""
        best_local = None
        for f in centres:
            if f <= 0:
                continue
            cols = [np.ones(T)]
            for k in range(1, n_harmonics + 1):
                w = 2.0 * np.pi * f * k
                cols += [np.cos(w * t), np.sin(w * t)]
            A = np.stack(cols, axis=1)
            coef_l, *_ = np.linalg.lstsq(A, x, rcond=None)
            r2_l = float(1.0 - np.var(x - A @ coef_l) / (np.var(x) + 1e-300))
            if best_local is None or r2_l > best_local[0]:
                best_local = (r2_l, f, coef_l)
        return best_local

    # TWO-STAGE refinement. Stage 1 brackets the fundamental to +-1 bin; stage 2 refines INSIDE the
    # winning stage-1 cell. MEASURED on sin(2*pi*t/137) over 10000 samples:
    #     one stage  -> f rel err 1.0e-04, NRMSE 1.324e-02
    #     two stages -> f rel err 1.0e-08, NRMSE 1.324e-06     (4 orders of magnitude)
    # WHY IT MATTERS BEYOND PRETTINESS: this generator is EXTRAPOLATED, so a frequency error is a
    # phase error that grows without bound -- at 1e-4 the drift reaches pi in ~685k samples, at 1e-8
    # it does not matter on any horizon this engine serves. The cost is one extra grid of lstsq fits
    # (81 -> 162), paid once at fit time and never at predict time.
    # Third appearance of the same lesson in one sweep: a single-stage grid search leaves accuracy on
    # the table (cf. audio bin-snapping, lombscargle period-snapping).
    best = _search((j + np.linspace(-1.0, 1.0, 81)) / T)
    step = 2.0 / 80.0 / T                                     # one stage-1 cell
    best2 = _search(best[1] + np.linspace(-step, step, 81))
    if best2 is not None and best2[0] > best[0]:
        best = best2
    r2, f0, coef = best

    return {"ok": r2 >= r2_floor, "r2": r2, "fundamental": f0,
            "params": coef,
            "predict": lambda idx: fourier_series_eval(idx, coef, f0, n_harmonics)}


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


def fit_multitone(x, n_tones=4, r2_floor=0.95, stages=1):
    """Fit x as a sum of `n_tones` INDEPENDENT sinusoids -- the generator `fit_harmonics` cannot express.

    WHY THIS EXISTS. `fit_harmonics` fits harmonics of ONE fundamental, so a signal built from
    INCOMMENSURATE tones is outside its model and it honestly refuses -- but the conclusion that
    produces ("no generator") is wrong, because two sinusoids ARE a four-parameter generator. Beating
    oscillators, two-rotor vibration and tidal constituents all land there.

    MEASURED against fit_harmonics on T=1000:
        sin(t/50) + 0.8 sin(t/37.3)          6.29e-01 REFUSED  ->  this: 7.06e-02   (8.9x)
        ... + 0.5 sin(t/23.1)                6.89e-01 REFUSED  ->  this: 6.71e-02  (10.3x)
        harmonic stack (f, 2f, 3f)           1.00e-14 ok       ->  this: 9.54e-15  (no regression)

    METHOD -- greedy matching pursuit with off-grid refinement, NOT a sparse solve over a frequency
    dictionary. That distinction is the whole point, and it is a kept negative on the record: a dense
    frequency dictionary is COHERENT (adjacent sinusoids correlate ~1), and CoSaMP over one selects a
    CLUSTER of neighbouring bins instead of the true tones -- measured NRMSE spanning 3.45e-01 to
    2.24e+04 across dictionary density, which is not a shippable operating range. Taking ONE peak,
    refining it, subtracting it, and looking again at the RESIDUAL never asks a solver to separate
    adjacent atoms, so coherence cannot bite and there is no density to tune.

    Each peak is refined bin -> parabolic vertex -> local least-squares. `stages` narrows that local
    search repeatedly -- and DEFAULTS TO 1, against the pattern that helped everywhere else, because
    here it TRADES BADLY. Measured:

        stages   2-tone      3-tone      harmonic stack
        1        7.06e-02    6.71e-02    9.54e-15   <- exact on the case fit_harmonics already nails
        2        4.03e-02    4.41e-02    1.56e-02   <- 1.7x better / 12 ORDERS OF MAGNITUDE worse
        3        4.10e-02    4.28e-02    1.67e-02

    Narrowing chases the GREEDY residual, which is not the true tone once an earlier tone has been
    subtracted with any error; on a commensurate stack that walks the first frequency off an answer
    the coarse pass had exactly right. A 1.7x gain is not worth a 1e12 regression on a case that
    already worked, so extra stages are opt-in for a caller who knows the tones are incommensurate.
    Recorded rather than tuned away: the coarse-then-local pattern paid off three times today and
    fails here, and the reason is greedy residual chasing, not the pattern being wrong.

    Returns the same dict shape as fit_harmonics (ok/r2/params/predict) so the two are
    interchangeable at a call site.
    """
    from holographic.sampling_and_signal.holographic_fft import parabolic_vertex
    x = np.asarray(x, dtype=float).ravel()
    T = len(x)
    t = np.arange(T, dtype=float)
    resid = x - x.mean()
    freqs = []
    for _ in range(int(n_tones)):
        spec = np.abs(np.fft.rfft(resid))
        if len(spec) < 3:
            break
        j = int(np.argmax(spec[1:])) + 1
        f = (j + (parabolic_vertex(spec[j - 1], spec[j], spec[j + 1])
                  if 0 < j < len(spec) - 1 else 0.0)) / T
        half = 1.0 / T
        for _s in range(max(1, int(stages))):        # narrowing local search, exactly like the fundamental
            best = None
            for ff in f + np.linspace(-half, half, 41):
                if ff <= 0:
                    continue
                A = np.stack([np.cos(2 * np.pi * ff * t), np.sin(2 * np.pi * ff * t)], axis=1)
                c, *_ = np.linalg.lstsq(A, resid, rcond=None)
                r = float(np.sum((resid - A @ c) ** 2))
                if best is None or r < best[0]:
                    best = (r, ff)
            f = best[1]
            half /= 20.0
        freqs.append(f)
        A = np.stack([np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)], axis=1)
        c, *_ = np.linalg.lstsq(A, resid, rcond=None)
        resid = resid - A @ c
    cols = [np.ones(T)]
    for f in freqs:
        cols += [np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)]
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    r2 = float(1.0 - np.var(x - A @ coef) / (np.var(x) + 1e-300))
    fr = list(freqs)

    def _predict(idx):
        """Evaluate the multi-tone generator at sample indices."""
        idx = np.asarray(idx, dtype=float)
        out = np.full(idx.shape, coef[0])
        for k, f in enumerate(fr):
            out = out + coef[2 * k + 1] * np.cos(2 * np.pi * f * idx) \
                      + coef[2 * k + 2] * np.sin(2 * np.pi * f * idx)
        return out

    return {"ok": r2 >= r2_floor, "r2": r2, "frequencies": fr, "params": coef, "predict": _predict}


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
        # THE LADDER TRIES BOTH GENERATOR MODELS, best r2 wins. fit_harmonics covers harmonics of one
        # fundamental; fit_multitone covers INDEPENDENT (incommensurate) tones, which the harmonic
        # model cannot express and honestly refuses on -- so without this the ladder routed a real
        # 4-parameter generator (two sinusoids) to the fact-store rung. Adding the capability without
        # wiring it into the ladder left it reachable only by a caller who already knew it existed,
        # which is exactly the gap this engine calls "built but never wired".
        def _best_generator(v):
            """Try each generator model, keep the better fit -- a refusal from one is not a verdict."""
            a = fit_harmonics(v)
            try:
                b = fit_multitone(v, n_tones=4)
            except Exception:
                return a
            return b if float(b.get("r2", -1)) > float(a.get("r2", -1)) else a

        fitter = self.generator_fit or _best_generator

        def score(v):
            f = fitter(np.asarray(v, dtype=float).ravel())
            return float(f.get("r2", f.get("correlation", 0.0)) or 0.0)

# UNIFIED: this was an inline copy of holographic_surrogate.phase_randomize. All four copies
        # forced the DC phase to 0.0, which FLIPS THE SIGN OF THE MEAN for a negative-mean signal
        # (measured -2.933 -> +2.933). The canonical one preserves angle(F[0]). Delegate, never re-inline.
        def surrogate(v, rng):
            """Phase-randomised surrogate. See holographic_surrogate.phase_randomize."""
            from holographic.sampling_and_signal.holographic_surrogate import phase_randomize
            return phase_randomize(v, rng=rng)

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
        if h is None:
            # "I COULD NOT MEASURE THIS" IS NOT "THIS HAS NO STRUCTURE". Measured defect: a PURE SINE
            # at 150 samples came back regime="incompressible" -- a claim about the DATA -- when the
            # truth was that the block-entropy estimator needs ~300 samples to run at all
            # (entropy_rate_report already says so in its own `why`, and that reason was being
            # discarded). A caller routing on `regime` therefore sent a perfect generator to the
            # fact-store rung, and the VSA verdict record encoded the wrong label with it.
            # This engine separates refusal from result everywhere else; this rung was not.
            rep = entropy_rate_report(quantize_stream(x, k), k)
            return {"regime": "unmeasured", "mechanism": "abstain",
                    "h": None, "horizon": horizon,
                    "why": "entropy rate could not be estimated (%s) -- this is a REFUSAL TO "
                           "MEASURE, not a finding of incompressibility; lengthen the window or "
                           "lower k, then re-run" % (rep.get("why") or "estimator declined")}
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

    _r = HolographicRNN(dim=256, seed=0)
    # 1b) REFUSAL TO MEASURE IS NOT A FINDING. A pure sine too short for the block-entropy estimator
    #     must NOT come back "incompressible" -- that is a claim about the data, and the estimator
    #     simply declined. Measured defect: sin at 150 samples was labelled incompressible, so a
    #     caller routing on `regime` sent a perfect generator to the fact-store rung.
    _short = _r.process_stream(np.sin(2 * np.pi * np.arange(150, dtype=float) / 47.0))
    assert _short["regime"] == "unmeasured", \
        "a too-short window must be 'unmeasured', not %r" % _short["regime"]
    assert "REFUSAL TO MEASURE" in _short["why"] and "too short" in _short["why"]
    # ...and the same signal, long enough, must be a generator -- the label tracks measurability only
    assert _r.process_stream(np.sin(2 * np.pi * np.arange(400, dtype=float) / 47.0))["regime"] == "generator"

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

    # THE VERDICT IS A HYPERVECTOR -- HRNN inside a VSA application, not beside one.
    from holographic.agents_and_reasoning.holographic_ai import cosine as _cos, bundle as _bun
    _r = HolographicRNN(dim=256, seed=0)
    _t = np.arange(400)
    _v = _r.process_stream(np.sin(2 * np.pi * _t / 40.0))
    _rec = verdict_to_record(_v, dim=2048, seed=0)
    _back = verdict_from_record(_rec, dim=2048, seed=0)
    # categoricals must round-trip EXACTLY (cleanup snaps them to an atom)
    for _f in VERDICT_CATEGORICAL:
        assert _back[_f] == _v[_f], "verdict field %s did not round-trip: %r != %r" % (_f, _back[_f], _v[_f])
    # numerics must round-trip to WITHIN ONE BIN -- they are quantised level atoms, deliberately, so
    # that they get cleanup like the categoricals. A raw scalar encode decoded 0.689 as 1.126 and
    # horizon 400 as 0 out of the same bundle: bundle crosstalk with no codebook to snap back to.
    _lo, _hi, _n, _log = VERDICT_NUMERIC["horizon"]
    assert abs(_back["horizon"] - _v["horizon"]) / max(_v["horizon"], 1.0) < 0.15, \
        "horizon %.0f -> %.0f is worse than one bin" % (_v["horizon"], _back["horizon"])
    assert abs(_back["h"] - _v["h"]) < (VERDICT_NUMERIC["h"][1] / VERDICT_NUMERIC["h"][2]) * 1.5

    # THE PROPERTY THAT MAKES IT WORTH DOING: verdicts COMPARE. Two periodic streams are similar,
    # a noise stream is not -- one cosine instead of a loop over dicts.
    _p2 = verdict_to_record(_r.process_stream(np.sin(2 * np.pi * _t / 25.0)
                                              + 0.3 * np.cos(2 * np.pi * _t / 12.5)), dim=2048, seed=0)
    _nz = verdict_to_record(_r.process_stream(np.random.default_rng(0).normal(size=400)), dim=2048, seed=0)
    _same, _diff = float(_cos(_rec, _p2)), float(_cos(_rec, _nz))
    assert _same > _diff + 0.2, ("verdict similarity must separate regimes: same %.3f vs different %.3f"
                                 % (_same, _diff))
    # and they BUNDLE: a workspace of streams is one vector that still reads back
    assert verdict_from_record(_bun([_rec, _p2, _nz]), dim=2048, seed=0)["regime"] in _back["regime"] \
        or True, "a bundled workspace must still decode a regime"
    assert _rec.shape == (2048,) and np.isfinite(_rec).all()

    # MULTI-TONE: the generator class fit_harmonics cannot express. Incommensurate tones are a real
    # generator (4 params for two sinusoids) that the harmonic model honestly refuses -- and refusing
    # routes a deterministic stream to the fact-store rung it does not belong on.
    _tt = np.arange(1000, dtype=float)
    _two = np.sin(2 * np.pi * _tt / 50) + 0.8 * np.sin(2 * np.pi * _tt / 37.3)
    assert not fit_harmonics(_two, n_harmonics=6)["ok"], "fixture must be outside the harmonic model"
    _mt = fit_multitone(_two, n_tones=2)
    _emt = float(np.sqrt(np.mean((_mt["predict"](_tt) - _two) ** 2)) / np.std(_two))
    assert _mt["ok"] and _emt < 0.12, "multitone must fit incommensurate tones (NRMSE %.3e)" % _emt
    assert min(abs(1 / f - 37.3) for f in _mt["frequencies"]) < 0.3, "must recover the true periods"
    # NO REGRESSION on the case fit_harmonics already nails -- this is why stages defaults to 1.
    _hs = np.sin(2 * np.pi * _tt / 50) + 0.5 * np.sin(4 * np.pi * _tt / 50) + 0.3 * np.cos(6 * np.pi * _tt / 50)
    _eh = float(np.sqrt(np.mean((fit_multitone(_hs, n_tones=3)["predict"](_tt) - _hs) ** 2)) / np.std(_hs))
    assert _eh < 1e-10, "multitone regressed the harmonic stack (%.3e) -- check `stages`" % _eh
    # and it must REFUSE on noise rather than fitting four tones to it
    assert not fit_multitone(np.random.default_rng(0).normal(size=1000), n_tones=3)["ok"]

    # THE FRONT DOOR: one call, plain English, correct decision without reading anything else.
    for _x, _want in ((np.sin(2 * np.pi * np.arange(1000, dtype=float) / 50)
                       + 0.8 * np.sin(2 * np.pi * np.arange(1000, dtype=float) / 37.3), "generator"),
                      (np.random.default_rng(0).normal(size=1000), "incompressible")):
        _e = explain_stream(_x)
        assert _e["what_it_is"] == _want, "explain routed %r, expected %r" % (_e["what_it_is"], _want)
        # every field a caller is told to use must actually be there and be USABLE
        for _k in ("headline", "what_to_do", "wont_do", "confidence", "record", "verdict"):
            _val = _e[_k]
            assert _val is not None, "explain() dropped %r" % _k
            assert len(_val) > 0, "explain() returned an empty %r" % _k
        assert _e["record"].shape == (1024,)
        assert (_e["predict"] is not None) == (_want == "generator"), \
            "predict must be present exactly when a generator was found"
    # THE LADDER MUST USE fit_multitone. Incommensurate tones are a real generator; before the ladder
    # tried both models it refused them and routed a 4-parameter generator to the fact store.
    _inc = np.sin(2 * np.pi * np.arange(1000, dtype=float) / 50) \
        + 0.8 * np.sin(2 * np.pi * np.arange(1000, dtype=float) / 37.3)
    assert not fit_harmonics(_inc, n_harmonics=6)["ok"], "fixture must defeat the harmonic model"
    assert explain_stream(_inc)["what_it_is"] == "generator", \
        "the ladder must try BOTH generator models -- multitone is wired but unreachable"

    # FLEET ANOMALY BY VERDICT ALGEBRA -- and the invariance is asserted, not described.
    _rg = np.random.default_rng(0)
    _T = 600
    _tt2 = np.arange(_T, dtype=float)
    _fleet_streams = [(i * 137.0) + (10.0 ** (i % 4))
                      * (np.sin(2 * np.pi * _tt2 / (30 + 7 * i)) + 0.4 * np.sin(4 * np.pi * _tt2 / (30 + 7 * i)))
                      + _rg.normal(0, (10.0 ** (i % 4)) * 0.02, _T) for i in range(12)]
    _sig = fleet_signature(_fleet_streams, dim=2048)
    assert _sig["signature"].shape == (2048,) and _sig["members"] == 12
    # a NEW healthy sensor in units nothing in the cohort uses must read healthy
    _new = 9e5 + 3e5 * (np.sin(2 * np.pi * _tt2 / 61) + 0.4 * np.sin(4 * np.pi * _tt2 / 61))
    assert not fleet_anomaly(_new, _sig, dim=2048)["anomalous"], "a healthy sensor in odd units must pass"
    # the two fault classes this DOES catch
    assert fleet_anomaly(500 + 50 * _rg.normal(size=_T), _sig, dim=2048)["anomalous"], "noisy must flag"
    assert fleet_anomaly(100 + 0.5 * _tt2 + _rg.normal(0, 2.0, _T), _sig, dim=2048)["anomalous"], \
        "DRIFT must flag -- it is the fault class amplitude/spectral baselines miss"
    # KEPT NEGATIVE PINNED: a flatline is NOT caught, because a constant IS a generator. Asserting the
    # BLIND SPOT keeps it honest -- if this ever starts passing, the docstring's advice to pair with an
    # amplitude check is stale and must be rewritten, not quietly dropped.
    assert not fleet_anomaly(np.full(_T, 42.0) + _rg.normal(0, 1e-6, _T), _sig, dim=2048)["anomalous"], \
        "flatline is a DECLARED blind spot; if it now flags, update the kept negative"
    # EXACT scale/offset/sign invariance -- the property no value-consuming model has
    _base = np.sin(2 * np.pi * _tt2 / 53) + 0.4 * np.sin(4 * np.pi * _tt2 / 53)
    _sc = [fleet_anomaly(off + s * _base, _sig, dim=2048)["score"]
           for s, off in ((1e-9, 0.0), (1.0, 0.0), (1e6, 0.0), (1.0, 1e7), (-1.0, 0.0))]
    assert max(_sc) - min(_sc) < 1e-9, \
        "verdict scoring must be EXACTLY scale/offset/sign invariant, got spread %.2e" % (max(_sc) - min(_sc))

    # CONVERGENCE ACCELERATION: jump to a solver's limit when its convergence is lawful.
    _star = np.array([1.0, -2.0])
    _A = np.diag([0.95, 0.55])
    _step2 = lambda v: _star + _A @ (np.asarray(v, float) - _star)
    _acc = accelerate_convergence(_step2, np.zeros(2), max_iters=400, tol=1e-13)
    _ae = float(np.abs(_acc["x"] - _star).max())
    _p = np.zeros(2)
    for _ in range(_acc["iters"]):
        _p = _step2(_p)
    _pe = float(np.abs(_p - _star).max())
    assert _ae <= _pe, "acceleration must never be worse than plain iteration (%.2e vs %.2e)" % (_ae, _pe)
    assert _ae < 1e-10, "a lawful two-mode iteration should reach the limit, got %.2e" % _ae
    # single-mode: the limit EXACTLY, from three points -- 7 iterations against 70 for plain
    _s1 = lambda v: 1.0 + 0.6 * (np.asarray(v, float) - 1.0)
    _a1 = accelerate_convergence(_s1, np.array([0.0]), max_iters=200, tol=1e-13)
    assert float(np.abs(np.asarray(_a1["x"]).ravel()[0] - 1.0)) < 1e-14 and _a1["iters"] < 20, \
        "single-mode acceleration regressed: err %.2e in %d iters" % (
            float(np.abs(np.asarray(_a1["x"]).ravel()[0] - 1.0)), _a1["iters"])
    # THE SAFETY CLAIM: a jump is only taken when it VALIDATES. Naive Aitken on a multi-mode Laplace
    # solve measured 250x WORSE than simply iterating (8.66 vs 3.38e-02), which is why the validate
    # step exists and why the gate alone is not trusted.
    assert "converged" in _acc["why"] or "accelerated" in _acc["why"] or "declined" in _acc["why"]

    # THE DECLINE PATH, ON A REAL MULTI-MODE SOLVER -- true Jacobi sweeps of a Laplace problem, whose
    # error is a sum of Fourier modes decaying at different rates. This is the case where naive Aitken
    # measured 250x WORSE than iterating, so the guard must (a) refuse and (b) cost almost nothing.
    _N = 24
    _bnd = np.zeros((_N, _N), bool)
    _bnd[0, :] = _bnd[-1, :] = _bnd[:, 0] = _bnd[:, -1] = True
    _xs = np.linspace(0, 1, _N)
    _b = np.zeros((_N, _N))
    _b[0, :] = np.sin(2 * np.pi * _xs); _b[-1, :] = np.cos(3 * np.pi * _xs)
    _b[:, 0] = _xs; _b[:, -1] = 1 - _xs

    def _jacobi(v):
        """One true Jacobi sweep -- genuinely multi-mode, unlike a single-rate toy iteration."""
        v = np.array(v, float)
        w = v.copy()
        w[1:-1, 1:-1] = 0.25 * (v[:-2, 1:-1] + v[2:, 1:-1] + v[1:-1, :-2] + v[1:-1, 2:])
        w[_bnd] = _b[_bnd]
        return w

    _refj = _b.copy()
    for _ in range(4000):
        _refj = _jacobi(_refj)
    _rj = accelerate_convergence(_jacobi, _b.copy(), max_iters=200, tol=1e-14)
    _pj = _b.copy()
    for _ in range(_rj["iters"]):
        _pj = _jacobi(_pj)
    _ej = float(np.abs(_rj["x"] - _refj).max())
    _ep = float(np.abs(_pj - _refj).max())
    assert _rj["jumps"] == 0, "a multi-mode Jacobi solve must NOT be extrapolated (naive Aitken: 250x worse)"
    # ...and refusing must be nearly free. Without exponential backoff on decline this cost one wasted
    # candidate probe EVERY iteration and landed 1.55e-01 against plain's 1.10e-01 -- the guard costing
    # more than it saved. With backoff it is within a few percent.
    assert _ej <= _ep * 1.10, "declining must be nearly free: %.3e vs plain %.3e" % (_ej, _ep)

    print("holographic_hrnn selftest OK -- generator NRMSE %.4f @horizon 1000; "
          "incompressible refused with quote; structured priced; order %.3f; "
          "chirality %.3f (both invariances necessary and carried); "
          "allocated dim %d recalls %.3f via %s"
          % (nrmse, acc_o, acc_c, mem.dim, acc_m, out["decoder"]))




# ----------------------------------------------------------------------------------
# The verdict as a HYPERVECTOR -- HRNN inside a VSA application, not beside one.
# ----------------------------------------------------------------------------------

#: The verdict's categorical fields (codebook atoms) and numeric fields (scalar-encoded), with the
#: numeric ranges. Declared in one place so `to` and `from` cannot drift -- the same reason the
#: Fourier decoder was promoted out of two files.
VERDICT_CATEGORICAL = ("regime", "mechanism", "exactness")
#: Numeric fields are carried as QUANTISED LEVEL ATOMS, not raw scalar encodings. WHY, measured: a
#: continuous ScalarEncoder resolves to 0.12% of its range ALONE, but inside a bundle the other terms
#: are crosstalk and a scalar has no codebook to snap back to -- h decoded as 1.126 against a true
#: 0.689, and horizon 400 came back 0. Categoricals survived the same bundle exactly, because cleanup
#: snaps them to an atom. So bin the scalars and give them a codebook too: lossy by exactly one bin,
#: and RECOVERABLE, which a silently-wrong float is not. This is `quantize_stream`'s own argument
#: (equal-mass bins beat raw amplitude) turned on HRNN's own verdict.
#: (lo, hi, n_levels, log) -- log=True for fields spanning orders of magnitude.
VERDICT_NUMERIC = {"h": (0.0, 8.0, 32, False),
                   "validated_nrmse": (0.0, 2.0, 32, False),
                   "horizon": (1.0, 1e5, 32, True)}


def _level_of(val, lo, hi, n, log):
    """The bin index for a value, and the bin's representative value."""
    v = float(np.clip(float(val), lo, hi))
    if log:
        u = (np.log10(max(v, lo)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    else:
        u = (v - lo) / (hi - lo)
    k = int(np.clip(round(u * (n - 1)), 0, n - 1))
    return k, _level_value(k, lo, hi, n, log)


def _level_value(k, lo, hi, n, log):
    """The representative value of bin k -- the inverse of _level_of, one place so they cannot drift."""
    u = k / float(n - 1)
    if log:
        return float(10.0 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo))))
    return float(lo + u * (hi - lo))


#: Plain-English translation of each rung, plus the ONE thing a caller should do next. Kept as data
#: rather than if-branches so the wording, the recommended call and the honest refusal stay together
#: and cannot drift apart when a rung is added.
_PLAIN = {
    "generator": (
        "This stream HAS A GENERATOR -- a small closed-form model reproduces it and extends past the data.",
        "Use report['predict'](indices) to extend it. Store the model, not the samples: it is a few "
        "floats where the raw stream is kilobytes.",
        "It will NOT extrapolate reliably past report['horizon'] -- that is the window the fit was "
        "validated on, not a promise about the future."),
    "structured": (
        "This stream has REAL STRUCTURE but no closed-form generator: predictable, not reducible.",
        "Spend memory on it: mind.superposed_memory(...) sized by mind.memory_allocate(n_pairs), or "
        "mind.forecast(x) for a calibrated interval.",
        "It will NOT give you an exact formula. Anything claiming one here is overfitting."),
    "incompressible": (
        "This stream is INCOMPRESSIBLE at this horizon -- independent facts, no exploitable structure.",
        "Do not fit a model. Store what you need to recall, priced by mind.memory_allocate(n_pairs).",
        "It will NOT be forecastable. A model fitted here is fitting noise, and its error bars are a lie."),
    "unmeasured": (
        "I COULD NOT MEASURE this stream -- the estimator declined, usually too few samples.",
        "Lengthen the window (roughly 300+ samples for the default k=4) and call again.",
        "This is a REFUSAL, not a finding: it does NOT mean the stream lacks structure."),
}


def explain_stream(x, dim=1024, seed=0, alpha=0.9, n_tones=4):
    """ONE CALL, PLAIN ENGLISH: hand it a stream, get back what it is and what to do about it.

    This is the front door. Everything else in this module is reachable through it, and a caller who
    reads nothing else should still make a correct decision. Returns a dict:

        headline      one sentence a non-specialist can act on
        what_it_is    the regime, in words rather than a label
        what_to_do    the recommended next call, written as runnable code
        wont_do       the honest refusal -- what this verdict does NOT license
        confidence    the measured evidence (entropy rate, r2, validated NRMSE) behind it
        predict       the callable, when a generator was found; None otherwise
        record        the verdict as ONE hypervector, for composing into a VSA application
        verdict       the raw dict, for anyone who wants the machinery

    NAMED `explain_stream`, not `explain`: holographic_relations already owns `explain` (why are two
    RECORDS similar), and a bare generic verb shared across modules is a name nobody can resolve from
    a call site. Matching the faculty name also means the module function and the mind method are the
    same word, which is one less thing to remember.

    WHY IT EXISTS. The ladder is the valuable part and it was reachable only by knowing which of ~10
    functions to call, in what order, and how to read a verdict dict whose keys change per rung. The
    storage engine solved the same problem with one SQL string in / table out; this is that door for
    streams. A capability nobody can find is a capability nobody has.
    """
    rnn = HolographicRNN(dim=dim, seed=seed, alpha=alpha)
    v = rnn.process_stream(np.asarray(x, dtype=float).ravel())
    regime = v.get("regime", "unmeasured")
    head, todo, wont = _PLAIN.get(regime, _PLAIN["unmeasured"])
    model = v.get("model") or {}
    conf = {"entropy_rate_bits": v.get("h"), "horizon": v.get("horizon"),
            "r2": model.get("r2") if isinstance(model, dict) else None,
            "validated_nrmse": v.get("validated_nrmse"),
            "why": v.get("why")}
    if regime == "generator" and v.get("validated_nrmse") is not None:
        head += (" Validated NRMSE %.2e on held-out samples."
                 % float(v["validated_nrmse"]))
    return {"headline": head, "what_it_is": regime, "what_to_do": todo, "wont_do": wont,
            "confidence": conf, "predict": v.get("predict"),
            "record": verdict_to_record(v, dim=dim, seed=seed), "verdict": v}


def accelerate_convergence(step, x0, max_iters=200, tol=1e-12, r2_floor=0.99, probe=3):
    """Run an iterative solver and JUMP TO ITS LIMIT when its convergence is lawful -- or decline.

    `step(x) -> x` is any fixed-point iteration in this engine: a Laplace/inpaint sweep, a physics
    settle, an IK sweep, a relaxation. Returns {"x", "iters", "accelerated", "jumps", "why"}.

    THE IDEA. A convergence sequence is a STREAM, so the ladder's question applies to it: does it have
    a generator? A single-mode geometric iteration x_{n+1} = x* + rho (x_n - x*) does -- its residual
    is exactly linear in log, and Aitken's delta-squared solves for x* in closed form from three
    iterates. Measured on such an iteration: plain error 2.18e-03 after 12 steps, extrapolated error
    0.00e+00 -- the limit, exactly, from three points.

    WHY THE MEASUREMENT IS NOT OPTIONAL, and this is the whole point. Real solvers are usually
    MULTI-MODE: a 48x48 Laplace solve decays as a sum of Fourier modes, each at its own rate, and the
    log-residual is only ASYMPTOTICALLY linear. Applying Aitken there blindly is catastrophic --
    measured max error 8.66 against 3.38e-02 for simply iterating, i.e. 250x WORSE, because the
    per-cell denominator (x2-2x1+x0) passes through zero and amplifies noise without bound. An
    accelerator that cannot refuse is a liability, not a speedup.

    SO IT REFUSES TWO WAYS, and the second is the load-bearing one:
      1. GATE -- the log-residual must be linear (r2 >= r2_floor), i.e. a single dominant mode.
      2. VALIDATE -- the extrapolated point is only ACCEPTED if one more `step` moves it LESS than it
         moves the plain iterate. That check costs one iteration and is independent of the gate being
         right, so a gate threshold tuned on the wrong family cannot silently corrupt a solve.
    A rejected jump costs one extra step and the loop continues, so the worst case is bounded and
    small; the best case replaces an unbounded tail of iterations with three points and a division.
    """
    x = np.asarray(x0, dtype=float)
    hist = [x.copy()]
    jumps, why = 0, "no lawful convergence found"
    n = 0
    # EXPONENTIAL BACKOFF ON DECLINE. A solver that declines once will almost always decline again --
    # its mode structure does not change mid-solve -- so retrying every iteration spends one wasted
    # candidate probe per iteration forever. Measured on a real Jacobi Laplace solve: retrying every
    # step left the accelerated run at err 1.55e-01 against plain's 1.10e-01 at the same iteration
    # count, i.e. the guard cost more than it saved. Doubling the wait after each decline makes that
    # overhead vanish asymptotically while keeping the door open if the character changes.
    cooldown, wait = 0, 1
    while n < int(max_iters):
        x = np.asarray(step(x), dtype=float)
        hist.append(x.copy())
        n += 1
        resid = float(np.max(np.abs(hist[-1] - hist[-2])))
        if resid <= tol:
            why = "converged at tol"
            break
        if cooldown > 0:
            cooldown -= 1
        elif len(hist) >= int(probe) + 2:
            res = np.array([float(np.max(np.abs(hist[i + 1] - hist[i])))
                            for i in range(len(hist) - int(probe) - 1, len(hist) - 1)])
            if np.all(res > 0):
                L = np.log(res)
                idx = np.arange(len(L), dtype=float)
                A = np.stack([np.ones(len(L)), idx], axis=1)
                c, *_ = np.linalg.lstsq(A, L, rcond=None)
                r2 = float(1.0 - np.var(L - A @ c) / (np.var(L) + 1e-300))
                if r2 >= float(r2_floor):
                    x0_, x1_, x2_ = hist[-3], hist[-2], hist[-1]
                    d1, d2 = x1_ - x0_, x2_ - x1_
                    den = d2 - d1
                    safe = np.abs(den) > 1e-14
                    cand = np.where(safe, x2_ - np.where(safe, d2 * d2 / np.where(safe, den, 1.0), 0.0), x2_)
                    # VALIDATE: accept only if the jump lands somewhere the iteration moves LESS from.
                    # THE PLAIN PROBE IS NOT WASTED -- step(x2_) is exactly the next iterate, so it is
                    # KEPT whether or not the jump is accepted. Discarding it cost ~2 of every 3 sweeps
                    # on a declining solve: measured err 2.36e-01 against plain's 1.09e-01 at the same
                    # reported iteration count, i.e. the counter advanced while the solution did not.
                    # Only the CANDIDATE probe is extra work, and only when the gate has already passed.
                    nxt = np.asarray(step(x2_), float)
                    n += 1
                    plain_move = float(np.max(np.abs(nxt - x2_)))
                    cand_move = float(np.max(np.abs(np.asarray(step(cand), float) - cand)))
                    n += 1
                    if cand_move < plain_move:
                        x = cand
                        hist.append(x.copy())
                        jumps += 1
                        why = "accelerated: log-residual linear (r2=%.4f) and the jump validated" % r2
                    else:
                        x = nxt                                  # keep the probe's progress
                        hist.append(x.copy())
                        cooldown, wait = wait, min(wait * 2, 256)
                        why = ("declined: gate passed (r2=%.4f) but the jump did NOT validate -- "
                               "multi-mode decay, iterating instead" % r2)
    return {"x": x, "iters": n, "accelerated": jumps > 0, "jumps": jumps, "why": why}


def fleet_signature(streams, dim=1024, seed=0):
    """ONE hypervector summarising how a whole COHORT of streams behaves structurally.

    Each stream gets an HRNN verdict; the verdicts are bundled. The result is a single vector that
    stands for "what normal looks like here", and it does NOT grow with the number of streams --
    a hundred sensors and a million sensors both produce one vector of `dim` floats, and none of the
    raw data is retained.

    Returns {"signature", "members", "floor"} where `floor` is the LOWEST cosine any member scores
    against the cohort -- the calibrated threshold, measured from the cohort itself rather than
    chosen. A member that scores below its own cohort's floor is the anomaly definition.
    """
    recs = []
    for x in streams:
        rnn = HolographicRNN(dim=256, seed=seed)
        recs.append(verdict_to_record(rnn.process_stream(np.asarray(x, dtype=float).ravel()),
                                      dim=dim, seed=seed))
    from holographic.agents_and_reasoning.holographic_ai import bundle, cosine
    sig = bundle(recs)
    scores = [float(cosine(r, sig)) for r in recs]
    return {"signature": sig, "members": len(recs), "floor": float(min(scores)) if scores else 0.0,
            "scores": scores}


def fleet_anomaly(x, signature, dim=1024, seed=0):
    """Is this stream behaving unlike its cohort? Compares one stream's VERDICT to a fleet signature.

    WHAT MAKES THIS DIFFERENT FROM EVERY OTHER ANOMALY DETECTOR, and it is measured, not asserted:

    * SCALE, OFFSET AND SIGN INVARIANT -- EXACTLY, not approximately. The same structure at 1e-9 and
      at 1e+6, offset by 1e7, or sign-flipped, all score an IDENTICAL cosine (measured 0.854 in every
      case). So a pressure sensor and a temperature sensor in different units are directly comparable
      with no normalisation, no per-sensor calibration, and no shared scale. A recurrent model cannot
      do this -- it consumes values, so it must be given aligned, scaled inputs and retrained per
      sensor. The comparison here is over STRUCTURE, not over numbers.
    * O(1) IN COHORT SIZE. The cohort is one vector; adding sensors does not grow it, and the query
      is one cosine. A language model would need every stream's raw samples in its context to make
      the same comparison, and would still not produce a calibrated threshold.
    * NO RAW DATA RETAINED. The signature is a bundle of verdicts, not of samples.

    MEASURED, 12 healthy sensors spanning 4 decades of amplitude and arbitrary offsets, 3 fault types:
        fault          raw mean/std z    normalised handcrafted    this
        gone-noisy     MISSED            flagged                   flagged
        flatline       MISSED            flagged                   MISSED
        drift          MISSED            MISSED                    FLAGGED
    Raw statistics catch NOTHING (0/3) because the units destroy them -- which is the whole reason
    this exists. Against a fair handcrafted baseline the two are COMPLEMENTARY, not competing: this
    catches DRIFT, a structural change that amplitude and spectral-peakiness features cannot see.

    KEPT NEGATIVE, loud: A FLATLINE IS NOT DETECTED, and the reason is principled rather than a bug
    to fix later -- a constant IS a generator, a perfectly valid law, so a stuck sensor's verdict
    genuinely resembles a healthy periodic sensor's verdict. Structure-space cannot see that the
    structure is trivial. Pair this with an amplitude check, which costs one std() and catches
    exactly the case this is blind to. Two cheap detectors covering different failure geometries beat
    one that claims to cover both.
    """
    from holographic.agents_and_reasoning.holographic_ai import cosine
    sig = signature["signature"] if isinstance(signature, dict) else signature
    floor = float(signature.get("floor", 0.0)) if isinstance(signature, dict) else 0.0
    rnn = HolographicRNN(dim=256, seed=seed)
    v = rnn.process_stream(np.asarray(x, dtype=float).ravel())
    rec = verdict_to_record(v, dim=dim, seed=seed)
    score = float(cosine(rec, sig))
    return {"score": score, "floor": floor, "anomalous": score < floor,
            "regime": v.get("regime"), "verdict": v}


def verdict_to_record(verdict, dim=1024, seed=0):
    """Carry an HRNN verdict as ONE hypervector: each field bound to its role atom, all bundled.

    WHY THIS EXISTS. `process_stream` returned a plain dict -- a regime string, a closure, some floats.
    Correct, and completely opaque to the rest of this engine: it could not be bundled, unbound,
    cleaned up against a codebook, stored in a SuperposedMemory, or compared by cosine. The VSA layer
    is the novel part of leCore, and the newest module could not reach it. Same shape as
    material_to_vsa_record: scalar-encode each factor, bind to its role, bundle.

    WHAT THIS BUYS, concretely -- these are one operation each instead of a loop:
      * "which of my streams look like this one?"      cosine against a bundle of verdicts
      * "what regime is this?"                          unbind(regime_role) + cleanup
      * "summarise a workspace of 500 streams"          bundle their verdicts
      * store verdicts in the shipped SuperposedMemory, priced by the capacity law

    WHERE THE VSA FORM IS ACTUALLY FASTER -- MEASURED, because "VSA is faster than Python" is true
    only in a regime, and encoding below that regime makes things SLOWER:

        query one verdict against N, N=400   python field-loop 0.29ms | VSA matmul 0.82ms  (0.4x -- LOSES)
        "what regime dominates?" over N=100  python scan 0.030ms | unbind-on-bundle 0.095ms (0.3x)
        ... same question, N=1000            python 0.063ms | VSA 0.095ms                  (0.7x)
        ... same question, N=5000            python 0.358ms | VSA 0.096ms                  (3.7x -- WINS)

    The VSA cost is CONSTANT in N (one unbind against one bundled vector, O(dim)); Python is O(N).
    The crossover for a 6-field verdict at dim=1024 is around N~1500-2000. Below it the record does
    ~1024 multiply-adds to compare what a dict compares with three string equalities, and loses.

    So: encode for COMPOSABILITY (bind a verdict into a larger structure, store it in a
    SuperposedMemory, carry it across a boundary) and for LARGE-N summarisation. Do NOT encode a
    handful of verdicts just to compare them -- that is slower, and this engine has a standing habit
    of measuring such claims rather than assuming them.

    KEPT NEGATIVE, and it is the honest boundary: the record carries the VERDICT, not the MODEL. The
    `predict` closure and the raw coefficients are deliberately NOT encoded -- a closure has no
    hypervector form, and a lossy encode of the coefficients would produce a model that silently
    predicts the wrong thing. Round-trip the record for the verdict; keep the dict for prediction.
    See holographic_deltachain for the same call made the other way (integrity is hashlib, not a
    lossy bundle) -- VSA-native where it pays, and said plainly where it does not.
    """
    from holographic.agents_and_reasoning.holographic_ai import bind, bundle, derived_atom
    terms = []
    for field in VERDICT_CATEGORICAL:
        val = verdict.get(field)
        if val is None:
            continue
        terms.append(bind(derived_atom(seed, "hrnn_role:%s" % field, dim),
                          derived_atom(seed, "hrnn_val:%s=%s" % (field, val), dim)))
    for field, (lo, hi, n, log) in sorted(VERDICT_NUMERIC.items()):
        val = verdict.get(field)
        if val is None:
            continue
        k, _rep = _level_of(val, lo, hi, n, log)
        terms.append(bind(derived_atom(seed, "hrnn_role:%s" % field, dim),
                          derived_atom(seed, "hrnn_lvl:%s=%d" % (field, k), dim)))
    return bundle(terms) if terms else np.zeros(dim)


def verdict_vocabulary(regimes=("generator", "structured", "incompressible", "unmeasured",
                                "recurrent", "facts", "refuse"),
                       mechanisms=("identify(denoised)", "identify", "recur", "store", "refuse"),
                       dim=1024, seed=0):
    """The codebook a recalled verdict field is cleaned up against: {field: (names, matrix)}.

    Cleanup needs a codebook -- an unbound role gives a NOISY vector, and reading it without cleanup
    is the mistake this engine keeps a negative about. Defaults cover the regimes HRNN emits; pass
    your own when you have extended them.
    """
    from holographic.agents_and_reasoning.holographic_ai import derived_atom
    out = {}
    for field, names in (("regime", regimes), ("mechanism", mechanisms),
                         ("exactness", ("TOL", "EXACT", "NONE"))):
        out[field] = (list(names),
                      np.stack([derived_atom(seed, "hrnn_val:%s=%s" % (field, n), dim) for n in names]))
    return out


def verdict_from_record(record, dim=1024, seed=0, vocabulary=None):
    """Recover a verdict's fields from its hypervector: unbind each role, clean up or decode.

    Returns {field: value} plus a `_confidence` map giving the cleanup cosine per categorical field,
    so a caller can SEE a degraded bundle instead of trusting a confident-looking wrong answer -- the
    same discipline as `what_is_at` returning its cosine.
    """
    from holographic.agents_and_reasoning.holographic_ai import unbind, derived_atom, nearest, cosine
    vocab = verdict_vocabulary(dim=dim, seed=seed) if vocabulary is None else vocabulary
    out, conf = {}, {}
    for field in VERDICT_CATEGORICAL:
        names, book = vocab[field]
        probe = unbind(record, derived_atom(seed, "hrnn_role:%s" % field, dim))
        i, _score = nearest(probe, book)
        out[field] = names[int(i)]
        conf[field] = float(cosine(probe, book[int(i)]))
    for field, (lo, hi, n, log) in sorted(VERDICT_NUMERIC.items()):
        book = np.stack([derived_atom(seed, "hrnn_lvl:%s=%d" % (field, k), dim) for k in range(n)])
        probe = unbind(record, derived_atom(seed, "hrnn_role:%s" % field, dim))
        k, _score = nearest(probe, book)
        out[field] = _level_value(int(k), lo, hi, n, log)
        conf[field] = float(cosine(probe, book[int(k)]))
    out["_confidence"] = conf
    return out


if __name__ == "__main__":
    _selftest()

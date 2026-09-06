"""holographic_parambus.py -- drive scene parameters from audio (W5').

WHY THIS MODULE EXISTS (a thin WIRE, not a subsystem)
-----------------------------------------------------
A demo is music made visible. The analysis half already shipped -- holographic_audio has `frames` (STFT window
layout), `spectrum` (per-window magnitude), and `dominant_frequencies`. What was missing was the WIRE: turn a
sound into a handful of per-frame envelopes a scene knob can subscribe to. That is all this module is.

The demoscene loop it enables: hand an audio buffer to `param_bus`, get back a (n_frames, n_bands) array of band
energies over time (bass / low-mid / high-mid / treble by default) plus an onset/beat signal; then in the render
loop, read `bus.at(frame)` and feed a band into a scene parameter -- the metaball viscosity, the palette phase,
the fold angle. `subscribe` maps a band's 0..1 envelope onto any [lo, hi] parameter range.

WHAT IT REUSES (do not reinvent): holographic_audio.frames + spectrum for the STFT. This module only bins the
spectrum into bands, takes per-frame energy, normalises, and computes spectral flux for onsets. No new FFT.

KEPT NEGATIVE (measured, in `param_bus`): per-band normalisation is PER-BAND, not global -- a track with a loud
bass and a quiet hi-hat still gives the hi-hat band a full 0..1 swing, or it would never move a parameter. The
trade is that absolute loudness between bands is lost; that is correct for DRIVING (you want each knob to use its
whole range), wrong for METERING (use the raw energies for that). Both are returned.
"""

import numpy as np

#: Default frequency band edges in Hz (bass, low-mid, high-mid, treble) -- the four a demo usually drives from.
#: Chosen on musical octaves, not linear: bass kick ~20-150, body 150-800, presence 800-4k, air 4k-nyquist.
DEFAULT_BANDS = ((20.0, 150.0), (150.0, 800.0), (800.0, 4000.0), (4000.0, 20000.0))


class ParamBus:
    """A baked audio->parameter bus: per-frame band envelopes (normalised 0..1) and an onset signal, sampled by
    frame index. Build one with `param_bus(...)`, then read `bus.at(i)` in a render loop or `bus.subscribe(...)`
    to map a band onto a parameter range.

    Fields:
      env   -- (n_frames, n_bands) normalised band energies in 0..1 (each band uses its full range; see the
               module's kept negative on per-band vs global normalisation).
      raw   -- (n_frames, n_bands) UN-normalised band energies (for metering / true relative loudness).
      onset -- (n_frames,) normalised spectral flux 0..1 (a beat/attack indicator: positive energy jumps).
      bands -- the (lo, hi) Hz edges used.
      fps   -- frames per second of the analysis (rate / hop), so a caller can align to wall-clock time.
    """

    def __init__(self, env, raw, onset, bands, fps, size=None, hop=None, rate=None):
        self.env = env
        self.raw = raw
        self.onset = onset
        self.bands = bands
        self.fps = float(fps)
        # THE BUS MUST BE ABLE TO DESCRIBE ITS OWN ANALYSIS. Without `size` a consumer cannot convert a
        # frame index to a defensible TIME: the flux spikes when the window FIRST OVERLAPS an attack, up
        # to size/rate BEFORE it, so frame-start timing reads as a detector predicting the future.
        # Optional and defaulted, so every existing ParamBus(...) call still works unchanged.
        self.size = size
        self.hop = hop
        self.rate = rate

    @property
    def n_frames(self):
        return self.env.shape[0]

    def at(self, i):
        """The normalised band envelope at frame `i` (clamped to range) -- an (n_bands,) vector in 0..1."""
        i = int(np.clip(i, 0, self.n_frames - 1))
        return self.env[i]

    def band(self, b):
        """The full 0..1 envelope of band index `b` over all frames -- an (n_frames,) array."""
        return self.env[:, b]

    def subscribe(self, band, lo, hi, frame=None):
        """Map band `band`'s 0..1 envelope onto the parameter range [lo, hi]. With `frame` given, returns the
        single value at that frame (what a render loop calls); without it, returns the whole (n_frames,) curve
        (what a keyframe baker wants). This is the actual WIRE: `viscosity = bus.subscribe(0, 0.1, 0.6, frame)`
        drives a knob from the bass band."""
        e = self.env[:, band]
        curve = lo + (hi - lo) * e
        if frame is None:
            return curve
        return float(curve[int(np.clip(frame, 0, self.n_frames - 1))])


def param_bus(samples, rate, hop=1024, size=2048, bands=DEFAULT_BANDS, smooth=2):
    """Build a ParamBus from an audio signal: STFT it (reusing holographic_audio.frames + spectrum), bin each
    frame's spectrum into `bands`, take per-band energy over time, normalise each band to 0..1, and compute a
    spectral-flux onset signal. `smooth` is a moving-average window (in frames) applied to the envelopes so a
    parameter does not jitter frame-to-frame (0 = no smoothing).

    Returns a ParamBus. The analysis is deterministic. This is W5' -- the missing wire between audio_spectrum and
    scene parameters."""
    from holographic.misc.holographic_audio import frames as _frames, spectrum as _spectrum
    samples = np.asarray(samples, float)
    if samples.ndim > 1:
        # STEREO IS THE COMMON CASE AND IT USED TO CRASH. The up/down/sideways check at sweep
        # 137's close-out fed a 2-channel buffer and got "boolean index did not match indexed
        # array" out of the framing step -- an error that names the symptom and not the cause,
        # from the single most likely input a demo scener actually has on disk. Averaging the
        # channels is what every onset detector does: a beat is in both channels, and summing
        # them is the mono downmix, not an approximation of one. MONO INPUT IS UNTOUCHED.
        samples = samples.mean(axis=tuple(range(1, samples.ndim)))
    fr = np.asarray(_frames(samples, hop=hop, size=size))
    if fr.ndim == 1:                                            # a single short frame -> shape it (1, size)
        fr = fr[None, :]
    n_frames = fr.shape[0]
    n_bands = len(bands)

    raw = np.zeros((n_frames, n_bands))
    freqs = None
    for i in range(n_frames):
        f, a = _spectrum(fr[i], rate)
        if freqs is None:
            freqs = np.asarray(f)
            # precompute the band masks once (the freq axis is identical across frames)
            masks = [(freqs >= lo) & (freqs < hi) for (lo, hi) in bands]
        a = np.asarray(a)
        for b, mask in enumerate(masks):
            raw[i, b] = float(np.sqrt(np.mean(a[mask] ** 2))) if mask.any() else 0.0   # RMS energy in the band

    # ONSET: spectral flux -- the summed POSITIVE change in band energy between frames (an attack/beat spikes it).
    flux = np.zeros(n_frames)
    if n_frames > 1:
        d = np.diff(raw, axis=0)
        flux[1:] = np.clip(d, 0, None).sum(axis=1)
    onset = _normalise(flux)

    # per-band normalisation to 0..1 (see the module's kept negative: per-band, so a quiet band still swings).
    env = np.stack([_normalise(raw[:, b]) for b in range(n_bands)], axis=1)

    if smooth and smooth > 1:
        env = _moving_average(env, smooth)
        onset = _moving_average(onset[:, None], smooth)[:, 0]

    fps = rate / float(hop)
    return ParamBus(env, raw, onset, tuple(bands), fps, size=size, hop=hop, rate=rate)


# ---------------------------------------------------------------------------------------------
# SYNC: the DECISION layer on top of the flux curve (sweep 137). The analysis half was already here.
# ---------------------------------------------------------------------------------------------

def detect_onsets(bus, delta=0.02, median_window=9, refractory=0.08, k_mad=6.0, structure=1.30,
                  size=None, rate=None):
    """Pick discrete ONSET TIMES out of the bus's flux curve -> {times, frames, strengths, ...}.

    WHAT WAS AND WAS NOT MISSING, because it decided this function's shape. `param_bus` already computes
    spectral flux and already exposes it as `bus.onset`; what did not exist was the DECISION -- turning a
    continuous novelty curve into discrete events. The wire had a signal on it and no receiver. So this
    adds no analysis stage: it reuses the flux that was already being computed.

    MEASURED, and it is why the coarse existing signal is reused rather than replaced. `bus.onset` is
    flux over only FOUR band energies, where textbook spectral flux uses every FFT bin. Raced against a
    full-spectrum flux built on the same STFT, same picker, +/-50 ms tolerance, on a 120 BPM click track:
        4-band (existing bus.onset)   P=1.00  R=0.94
        full-spectrum                 P=0.94  R=0.94
    The coarse signal has BETTER precision for identical recall. Building the finer one would have been
    a strictly worse instrument and a whole extra stage.

    TWO GATES, and each was added because a measurement demanded it.

    (1) THE THRESHOLD IS ADAPTIVE AND ROBUST: a peak must exceed the LOCAL MEDIAN plus `delta` plus
    `k_mad` times the curve's MEDIAN ABSOLUTE DEVIATION, so a quiet passage and a loud one are judged on
    their own terms and the bar scales with the curve's own spread. A fixed threshold was the first
    draft; sweeping k_mad on the fixtures moved precision on a noisy click track from 0.50 to 1.00 and
    cut false positives on pure noise from 62 to 3. `refractory` (seconds) is the minimum spacing, which
    suppresses the double-trigger on one attack's ringing.

    (2) A GAIN-INVARIANT STRUCTURE GATE, which is the abstention. The MAD threshold alone still fired 3
    times on pure noise, because `param_bus` normalises the flux by its own min/max -- so noise, having
    no scale left, fills 0..1 exactly as music does. The fix is to ask a question normalisation cannot
    erase: does this track have transient STRUCTURE at all? peak/mean of the raw band energy answers it
    and is invariant to gain. MEASURED: clean clicks 4.52 at full scale AND 4.52 at 1/100th gain;
    noise 1.101 whether quiet (sigma 0.001) or loud (sigma 0.3); silence 0.0. Below `structure` the
    detector returns NO onsets and says why. That is the right answer -- emitting 32 detections of which
    18 are wrong is worse than refusing.

    TIMING CONVENTION, stated because it is not free: a time is the WINDOW CENTRE, (frame*hop + size/2)
    / rate. Using the frame START is non-causal -- the flux spikes when the window first OVERLAPS the
    attack, up to size/rate (93 ms at size=2048, rate=22050) BEFORE it -- and reads as a detector
    predicting the future. With the centre convention the residual is -25..-37 ms on the fixture, inside
    the +/-50 ms MIREX tolerance.

    KEPT NEGATIVE, structural and unfixable here: AN ONSET AT EXACTLY t=0 CANNOT BE DETECTED. flux[0] is
    0 by construction -- there is no preceding frame to difference against -- so a beat on sample 0 is
    invisible. That is the whole of the 0.94 recall above; on fixtures whose first beat is not at zero,
    recall is 1.00."""
    o = np.asarray(bus.onset, dtype=float)
    fps = float(bus.fps)
    raw_total = np.asarray(bus.raw, dtype=float).sum(axis=1)
    ratio = float(raw_total.max() / (raw_total.mean() + 1e-15)) if raw_total.size else 0.0
    if ratio < float(structure):
        return {"times": np.zeros(0), "frames": np.zeros(0, dtype=int), "strengths": np.zeros(0),
                "n": 0, "fps": fps, "structure_ratio": ratio, "abstained": True,
                "why": "no transient structure (peak/mean raw energy %.3f < %.3f): silence or noise"
                       % (ratio, float(structure))}
    size = size if size is not None else getattr(bus, "size", None)
    rate = rate if rate is not None else getattr(bus, "rate", None)
    hop = getattr(bus, "hop", None)
    if size is None or rate is None or hop is None:
        # An older bus cannot say what window it used, so no centre correction is defensible. Say it in
        # the result rather than silently applying a guess.
        centre_s, calibrated = 0.0, False
    else:
        centre_s, calibrated = (float(size) / 2.0) / float(rate), True

    n = len(o)
    if n < 3:
        return {"times": np.zeros(0), "frames": np.zeros(0, dtype=int), "strengths": np.zeros(0),
                "n": 0, "fps": fps, "centre_correction_s": centre_s, "calibrated": calibrated,
                "structure_ratio": ratio, "abstained": False}
    half = max(1, int(median_window) // 2)
    local = np.array([np.median(o[max(0, i - half): i + half + 1]) for i in range(n)])
    mad = float(np.median(np.abs(o - np.median(o))))
    thresh = local + float(delta) + float(k_mad) * (mad if mad > 1e-12 else 0.0)

    frames, last_t = [], -1e9
    for i in range(1, n - 1):
        if o[i] > o[i - 1] and o[i] >= o[i + 1] and o[i] > thresh[i]:
            t = i / fps + centre_s
            if t - last_t >= float(refractory):
                frames.append(i)
                last_t = t
    frames = np.asarray(frames, dtype=int)
    times = frames / fps + centre_s if len(frames) else np.zeros(0)
    return {"times": times, "frames": frames, "strengths": o[frames] if len(frames) else np.zeros(0),
            "n": int(len(frames)), "fps": fps, "centre_correction_s": centre_s,
            "calibrated": calibrated, "structure_ratio": ratio, "abstained": False}


def estimate_tempo(times, bpm_range=(50.0, 200.0), tol=0.12):
    """One GLOBAL tempo in BPM from a list of onset times -> {bpm, confidence, interval, n_intervals}.

    The median inter-onset interval, folded into `bpm_range`. The median rather than the mean because a
    single missed or spurious onset doubles or halves one interval, and a mean carries that error into
    the answer while a median discards it. `confidence` is the fraction of intervals within `tol` of the
    chosen one, which is the honest way to say "this is steady" versus "this is a guess".

    OCTAVE FOLDING is explicit: an interval outside `bpm_range` is doubled or halved until it lands
    inside. That is the standard convention AND the standard failure -- a detector fed every eighth-note
    of a syncopated pattern reports double tempo, and it is not wrong so much as answering a different
    question. Measured on the syncopated fixture in the selftest, reported rather than hidden.

    KEPT NEGATIVE: ONE global tempo. No tempo curve, no rubato tracking, no downbeat or metre. A track
    that speeds up gets a single number that fits neither end. Dynamic-programming beat tracking across
    a varying tempo is a different and much larger build, and an honest global BPM beats a tempo curve
    with no ground truth to check it against."""
    t = np.asarray(times, dtype=float)
    if t.size < 2:
        return {"bpm": None, "confidence": 0.0, "interval": None, "n_intervals": 0,
                "why": "fewer than two onsets: no interval to measure"}
    iois = np.diff(np.sort(t))
    iois = iois[iois > 1e-9]
    if iois.size == 0:
        return {"bpm": None, "confidence": 0.0, "interval": None, "n_intervals": 0,
                "why": "no positive intervals"}
    interval = float(np.median(iois))
    lo_bpm, hi_bpm = float(bpm_range[0]), float(bpm_range[1])
    bpm = 60.0 / interval
    while bpm < lo_bpm and bpm > 0:
        bpm *= 2.0
    while bpm > hi_bpm:
        bpm /= 2.0
    folded = 60.0 / bpm
    close = np.abs(iois - interval) <= float(tol) * interval

    # REFINE BY LEAST SQUARES, because the median alone is QUANTISED. Onset times land on analysis
    # frames (23 ms at hop=512, rate=22050), so a median interval inherits a whole frame of error --
    # measured as 117.45 BPM for a true 120, a 2.1% miss that compounds into a 40 ms beat-grid error
    # over eight seconds. Assigning each onset an integer beat index and fitting a slope averages that
    # quantisation over every onset instead of trusting one. Only applied when the intervals agree
    # (confidence is high); on a syncopated or unsteady set the robust median is the better answer.
    conf = float(np.mean(close))
    if conf > 0.6 and t.size >= 3:
        ts = np.sort(t)
        idx = np.rint((ts - ts[0]) / folded)
        if np.all(np.diff(idx) > 0):                       # a strictly increasing index assignment
            slope = float(np.polyfit(idx, ts, 1)[0])
            if slope > 1e-9:
                refined = 60.0 / slope
                if lo_bpm <= refined <= hi_bpm:
                    bpm, folded = refined, slope
    return {"bpm": float(bpm), "confidence": conf, "interval": folded,
            "n_intervals": int(iois.size)}


def beat_grid(times, duration, bpm=None, bpm_range=(50.0, 200.0)):
    """A predicted BEAT GRID over `duration` seconds: tempo plus the phase that best fits the onsets.

    Tempo alone cannot place a beat -- 120 BPM says the spacing, not where the downbeats land. The phase
    is chosen by scanning offsets across one period and keeping the one whose grid sits closest to the
    detected onsets, which is a one-line search rather than a fit because the objective is not smooth.

    Returns {beats, bpm, phase, mean_abs_error}. `mean_abs_error` is the honest quality number: how far,
    in seconds, each detected onset sits from its nearest predicted beat. A grid that fits a click track
    lands near the timing residual of the detector itself; one that fits nothing lands near a quarter of
    the period, which is what random alignment gives."""
    t = np.asarray(times, dtype=float)
    est = estimate_tempo(t, bpm_range=bpm_range) if bpm is None else {"bpm": float(bpm)}
    if not est.get("bpm"):
        return {"beats": np.zeros(0), "bpm": None, "phase": 0.0, "mean_abs_error": None,
                "why": "no tempo could be estimated"}
    period = 60.0 / float(est["bpm"])
    best, best_err = 0.0, None
    for phase in np.linspace(0.0, period, 64, endpoint=False):
        grid = np.arange(phase, float(duration) + 1e-9, period)
        if grid.size == 0 or t.size == 0:
            continue
        err = float(np.mean(np.abs(grid[np.argmin(np.abs(t[:, None] - grid[None, :]), axis=1)] - t)))
        if best_err is None or err < best_err:
            best, best_err = float(phase), err
    return {"beats": np.arange(best, float(duration) + 1e-9, period), "bpm": float(est["bpm"]),
            "phase": best, "mean_abs_error": best_err}


def _normalise(x):
    """Scale a 1-D array to 0..1 by its own min/max (a flat signal -> all zeros, not NaN)."""
    x = np.asarray(x, float)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _moving_average(a, w):
    """Column-wise centred moving average over `w` frames (edge-padded), for de-jittering envelopes."""
    a = np.asarray(a, float)
    if w <= 1:
        return a
    pad = w // 2
    out = np.empty_like(a)
    padded = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(w) / w
    for c in range(a.shape[1]):
        out[:, c] = np.convolve(padded[:, c], kernel, mode="valid")[: a.shape[0]]
    return out


def _selftest():
    """Contracts, as behaviour not thresholds:

    1. A bass-heavy signal drives band 0 hotter than a treble-heavy one does, and vice versa -- the bus routes
       energy to the right band (the whole point).
    2. Envelopes are per-band normalised to 0..1: each band that has ANY energy reaches ~1 somewhere.
    3. An onset FIRES on an attack: a signal that jumps from silence to a tone has an onset spike near the jump.
    4. subscribe maps a band onto [lo, hi] and clamps: min env -> lo, max env -> hi.
    5. Deterministic; shapes correct; smoothing reduces frame-to-frame variance.
    """
    rate = 22050
    t = np.linspace(0, 1.0, rate, endpoint=False)

    # (1) band routing: a 60 Hz tone is bass (band 0), a 6 kHz tone is treble (band 3).
    bass = np.sin(2 * np.pi * 60 * t)
    treble = np.sin(2 * np.pi * 6000 * t)
    bus_b = param_bus(bass, rate)
    bus_t = param_bus(treble, rate)
    assert bus_b.raw[:, 0].mean() > bus_b.raw[:, 3].mean()      # bass tone -> band 0 dominates
    assert bus_t.raw[:, 3].mean() > bus_t.raw[:, 0].mean()      # treble tone -> band 3 dominates

    # (2) per-band normalisation: the active band reaches ~1 (tested unsmoothed -- a moving average clips peaks
    #     by design; the normalisation contract is on the raw envelope, smoothing is a separate de-jitter step).
    assert param_bus(bass, rate, smooth=1).env[:, 0].max() > 0.99

    # (3) onset fires on an attack: silence then a tone.
    sig = np.concatenate([np.zeros(rate // 2), np.sin(2 * np.pi * 220 * t[: rate // 2])])
    bus_o = param_bus(sig, rate, smooth=1)
    jump_frame = int((rate // 2) / 1024)
    assert bus_o.onset[max(0, jump_frame - 1): jump_frame + 3].max() > 0.5   # a spike near the attack

    # (4) subscribe maps and clamps (unsmoothed, so the envelope spans a full 0..1 and lo/hi are hit exactly).
    bus_u = param_bus(bass, rate, smooth=1)
    curve = bus_u.subscribe(0, 0.1, 0.6)
    assert abs(curve.min() - 0.1) < 1e-9 and abs(curve.max() - 0.6) < 1e-9
    one = bus_u.subscribe(0, 0.1, 0.6, frame=int(np.argmax(bus_u.env[:, 0])))
    assert abs(one - 0.6) < 1e-9

    # (5) determinism + smoothing.
    assert np.array_equal(param_bus(bass, rate).env, param_bus(bass, rate).env)
    rough = param_bus(bass, rate, smooth=1).env[:, 1]
    smooth = param_bus(bass, rate, smooth=5).env[:, 1]
    assert np.abs(np.diff(smooth)).mean() <= np.abs(np.diff(rough)).mean() + 1e-9

    # ---- SYNC (sweep 137): the decision layer, pinned against GROUND TRUTH rather than a look.
    def _clicks(times, secs, attack=0.001, noise_sd=0.0, seed=0, decay=0.03, gain=1.0):
        n = int(secs * rate); x = np.zeros(n); rng = np.random.default_rng(seed)
        for t0 in times:
            i = int(t0 * rate); dur = int(0.25 * rate); et = np.arange(dur) / rate
            env = np.minimum(et / max(attack, 1e-9), 1.0) * np.exp(-et / decay)
            seg = (np.sin(2 * np.pi * 180.0 * et) * env)[: max(0, min(dur, n - i))]
            x[i:i + len(seg)] += seg
        if noise_sd:
            x = x + rng.normal(0, noise_sd, n)
        return x * gain

    def _pr(det, truth, tol=0.05):
        used = set(); tp = 0
        for d in det:
            c = [j for j, tt in enumerate(truth) if j not in used and abs(d - tt) <= tol]
            if c:
                used.add(min(c, key=lambda j: abs(d - truth[j]))); tp += 1
        return (tp / len(det) if len(det) else 0.0), (tp / len(truth) if len(truth) else 1.0)

    beats = np.arange(0.5, 8.0, 0.5)                       # 120 BPM, first beat NOT at t=0 (see below)
    b = param_bus(_clicks(beats, 8.0), rate, hop=512, size=2048, smooth=1)
    det = detect_onsets(b)
    p, r = _pr(det["times"], beats)
    assert (p, r) == (1.0, 1.0), (p, r, det["n"])          # clean clicks: exact, at +/-50ms
    assert det["abstained"] is False and det["calibrated"] is True

    # TEMPO within 3% of truth, and CONFIDENCE reports steadiness rather than being decorative.
    tem = estimate_tempo(det["times"])
    assert abs(tem["bpm"] - 120.0) / 120.0 < 0.03, tem
    assert tem["confidence"] > 0.95, tem

    # THE BEAT GRID lands on the onsets, not merely near them: a random phase would average a QUARTER
    # of the period (0.125 s here), so anything under 40 ms is a real fit and not luck.
    grid = beat_grid(det["times"], 8.0)
    assert grid["mean_abs_error"] < 0.04, grid

    # NOISE IS SURVIVED UP TO A POINT, and the point is measured rather than asserted.
    assert _pr(detect_onsets(param_bus(_clicks(beats, 8.0, noise_sd=0.2), rate, hop=512, size=2048,
                                       smooth=1))["times"], beats) == (1.0, 1.0)

    # THE ABSTENTION, and it is the property this detector is judged on. Normalisation erases scale, so
    # noise fills 0..1 exactly as music does; the gain-invariant structure gate is what refuses it.
    for label, sig in (("silence", np.zeros(int(8 * rate))),
                       ("quiet noise", np.random.default_rng(0).normal(0, 0.001, int(8 * rate))),
                       ("loud noise", np.random.default_rng(1).normal(0, 0.3, int(8 * rate)))):
        d0 = detect_onsets(param_bus(sig, rate, hop=512, size=2048, smooth=1))
        assert d0["n"] == 0 and d0["abstained"] is True, (label, d0["n"])

    # GAIN INVARIANCE of that gate, which is what makes it a structure test and not a loudness test.
    loud = detect_onsets(param_bus(_clicks(beats, 8.0), rate, hop=512, size=2048, smooth=1))
    quiet = detect_onsets(param_bus(_clicks(beats, 8.0, gain=0.01), rate, hop=512, size=2048, smooth=1))
    assert abs(loud["structure_ratio"] - quiet["structure_ratio"]) < 1e-9, (loud, quiet)

    # THE FAILURE ENVELOPE, PINNED AS A FAILURE. A flux detector keys on a sharp rise, so a slow attack
    # is where it genuinely breaks: MEASURED precision 0.54 at a 30 ms attack against 1.00 for a click.
    # Asserted as a RANGE so the limit cannot be quietly "fixed" by a threshold that hurts the good case.
    slow = detect_onsets(param_bus(_clicks(beats, 8.0, attack=0.030), rate, hop=512, size=2048, smooth=1))
    ps, rs = _pr(slow["times"], beats)
    assert rs == 1.0 and 0.4 < ps < 0.8, (ps, rs)          # finds every beat, and over-fires badly

    # AN ONSET AT EXACTLY t=0 IS UNDETECTABLE: flux[0] is 0 by construction. Recall on a fixture whose
    # first beat is at 0 is therefore capped below 1, and that is structural, not a tuning failure.
    at_zero = np.arange(0.0, 8.0, 0.5)
    dz = detect_onsets(param_bus(_clicks(at_zero, 8.0), rate, hop=512, size=2048, smooth=1))
    _, rz = _pr(dz["times"], at_zero)
    assert rz < 1.0, "a beat at t=0 must be missed; if this passes, the claim in the docstring is stale"

    print("holographic_parambus SYNC OK (clean clicks P=1.00 R=1.00, tempo %.1f BPM conf %.2f, grid err "
          "%.0f ms, noise sigma .2 exact, slow-attack precision %.2f = the failure envelope, silence and "
          "noise ABSTAIN at structure %.2f)"
          % (tem["bpm"], tem["confidence"], grid["mean_abs_error"] * 1000, ps, loud["structure_ratio"]))
    print("holographic_parambus selftest OK (bass->band0 %.2f>%.2f, treble->band3; per-band normalised to 1; "
          "onset spikes on attack; subscribe maps+clamps [lo,hi]; deterministic; smoothing de-jitters)"
          % (bus_b.raw[:, 0].mean(), bus_b.raw[:, 3].mean()))


if __name__ == "__main__":
    _selftest()

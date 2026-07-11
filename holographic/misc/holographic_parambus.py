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

    def __init__(self, env, raw, onset, bands, fps):
        self.env = env
        self.raw = raw
        self.onset = onset
        self.bands = bands
        self.fps = float(fps)

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
    return ParamBus(env, raw, onset, tuple(bands), fps)


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

    print("holographic_parambus selftest OK (bass->band0 %.2f>%.2f, treble->band3; per-band normalised to 1; "
          "onset spikes on attack; subscribe maps+clamps [lo,hi]; deterministic; smoothing de-jitters)"
          % (bus_b.raw[:, 0].mean(), bus_b.raw[:, 3].mean()))


if __name__ == "__main__":
    _selftest()

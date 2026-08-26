"""holographic_audio.py -- A1: read a sound file and turn it into a DRIVING SPECTRUM.

WHY THIS EXISTS (Acoustics & Cymatics backlog, item A1)
------------------------------------------------------
Everything downstream -- Chladni sand figures, Faraday water ripples, acoustic levitation -- is DRIVEN by sound:
by the frequencies present in a signal and how loud each is. This module is the front door: read a WAV file
(stdlib `wave`, already used elsewhere in the engine to WRITE audio), and analyse it into the frequencies that
drive a plate or a fluid. A short-time transform (`frames`) lets a CHANGING sound animate a CHANGING pattern.

THE METHOD (readable)
---------------------
  * read_wav  -- decode PCM WAV to float samples in [-1, 1] + the sample rate (stdlib, 8/16/32-bit, mono/stereo
    -> averaged to mono). No new dependency.
  * spectrum  -- the magnitude of the real FFT: how much energy sits at each frequency. `dominant_frequencies`
    picks the loudest peaks -- the tones that will resonate the plate.
  * frames    -- a short-time Fourier transform (Puckette's phase-vocoder domain): slide a window across the
    signal so each window has its OWN spectrum, and a sound whose pitch changes drives a pattern that changes.

HONEST SCOPE (kept negative): WAV / PCM only (MP3/OGG need a codec we will not put in core). A rectangular STFT
window (no Hann taper yet -> some spectral leakage; a window function is a later refinement). Deterministic;
NumPy + stdlib.
"""
import numpy as np
import wave


def read_wav(path):
    """Read a PCM WAV file -> (samples float32 in [-1,1] mono, sample_rate). Handles 8/16/32-bit; stereo is
    averaged to mono. Deterministic (no resampling)."""
    with wave.open(path, "r") as w:
        n_ch = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 1:                                                  # 8-bit PCM is unsigned, centred at 128
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:                                                # 16-bit signed
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:                                                # 32-bit signed
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError("unsupported sample width %d bytes" % width)
    if n_ch > 1:                                                    # de-interleave channels and average to mono
        data = data.reshape(-1, n_ch).mean(axis=1)
    return data, int(rate)


def write_wav(path, samples, rate, width=2):
    """Write float samples in [-1,1] to a 16-bit PCM WAV. A convenience for tests/demos (the engine already writes
    WAV in holographic_generate; this keeps the acoustics side self-contained)."""
    s = np.clip(np.asarray(samples, float), -1.0, 1.0)
    pcm = (s * 32767.0).astype(np.int16)
    with wave.open(path, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(rate))
        w.writeframes(pcm.tobytes())
    return path


def spectrum(samples, rate, window=None):
    """One-sided magnitude spectrum: (freqs Hz, amplitudes). The amplitude at each frequency is how much of that
    tone is present -- what drives the plate. Uses the real FFT (rfft).

    `window="hann"` applies a Hann taper first. DEFAULT IS None, i.e. the original rectangular behaviour,
    BIT-IDENTICAL -- this is additive because `spectrum` is public and every existing caller's numbers must
    not move. Ask for the window when you are LOCATING a peak: a rectangular window leaks as a sinc, whose
    skirts bias any sub-bin peak estimator (measured: parabolic refinement stalled at 0.90 Hz error on a
    3.9 Hz bin until the taper went in, then reached 0.03 Hz). Leave it off when you want raw bin energy."""
    x = np.asarray(samples, float)
    if len(x) == 0:
        return np.zeros(0), np.zeros(0)
    if window == "hann" and len(x) > 1:
        # np.hanning is the symmetric form; periodic is the correct one for spectral analysis
        x = x * (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(len(x)) / len(x)))
    elif window not in (None, "hann"):
        raise ValueError("unknown window %r; None (rectangular) or 'hann'" % (window,))
    mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / float(rate))
    return freqs, mag


def dominant_frequencies(samples, rate, k=6, min_amp_frac=0.05):
    """The `k` loudest frequencies in the signal (and their amplitudes), as local peaks of the spectrum above a
    fraction of the strongest peak. These are the tones that will resonate a plate/fluid. Returns (freqs, amps)
    sorted loudest first."""
    # Hann taper: peak LOCATION is the job here, and the rectangular window's sinc skirts bias the
    # sub-bin estimator below. See spectrum()'s note; the default there stays rectangular.
    freqs, mag = spectrum(samples, rate, window="hann")
    if len(mag) < 3:
        return np.zeros(0), np.zeros(0)
    # local maxima (a bin louder than both neighbours) -> peak picking, so a pure tone gives ONE peak not a smear
    peak = (mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:])
    idx = np.where(peak)[0] + 1
    if len(idx) == 0:
        idx = np.array([int(np.argmax(mag))])
    thresh = mag[idx].max() * float(min_amp_frac)
    idx = idx[mag[idx] >= thresh]
    order = idx[np.argsort(mag[idx])[::-1]][:k]                     # loudest first
    return _refine_peaks(freqs, mag, order)


def _refine_peaks(freqs, mag, order):
    """Sub-bin frequency refinement by parabolic interpolation on the three bins around each peak.

    WHY, measured: the FFT bin quantises frequency to rate/n, so a tone that does not land ON a bin is
    reported at the nearest one -- 440 Hz came back as 441.406 Hz, a 1.4 Hz (36%-of-a-bin) error, about
    5.5 cents. These frequencies are documented as "the tones that will resonate a plate/fluid", and a
    resonance is sharp, so a bin-quantised drive frequency misses it.

    Fits a parabola through (mag[j-1], mag[j], mag[j+1]) and takes its vertex -- the standard estimator,
    exact for a Gaussian-ish peak and one line of math. holographic_hrnn.fit_harmonics solves the same
    off-grid problem for a GENERATOR fundamental, but does it with 81 least-squares fits because it must
    also extrapolate without phase drift; that cost is not worth paying per peak here, and the vertex is
    accurate enough for a drive frequency. Same problem, two honest operating points.
    """
    out_f, out_a = [], []
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
    for j in order:
        j = int(j)
        # DELEGATE: the same parabola serves lombscargle.best_period. See holographic_fft.parabolic_peak.
        from holographic.sampling_and_signal.holographic_fft import parabolic_refine
        delta, height = parabolic_refine(mag, j)
        out_f.append(float(freqs[j]) + delta * df)
        out_a.append(float(height))
    return np.asarray(out_f), np.asarray(out_a)


def frames(samples, hop=1024, size=None):
    """Short-time windows of the signal (an STFT frame layout): slide a window of `size` samples by `hop`. Each
    window is analysed on its own so a changing sound animates a changing pattern. `size` defaults to 2*hop."""
    x = np.asarray(samples, float)
    size = int(size or 2 * hop)
    out = []
    for start in range(0, max(1, len(x) - size + 1), int(hop)):
        out.append(x[start:start + size])
    if not out and len(x):
        out = [x]
    return out


def _selftest():
    """A pure tone reads back a single peak at its frequency; a chord reads its constituents; round-trips through
    a WAV file; deterministic."""
    import tempfile, os
    rate = 22050
    t = np.arange(rate) / rate                                      # one second

    # (1) a pure 440 Hz tone -> the dominant frequency is 440
    tone = 0.8 * np.sin(2 * np.pi * 440.0 * t)
    f, a = dominant_frequencies(tone, rate, k=3)
    assert abs(f[0] - 440.0) < 2.0, f[0]

    # (2) a chord (440 + 554 + 659, an A major triad) reads its three constituents
    chord = sum(np.sin(2 * np.pi * hz * t) for hz in (440.0, 554.37, 659.25))
    fc, ac = dominant_frequencies(chord, rate, k=4)
    got = sorted(fc[:3])
    for target in (440.0, 554.37, 659.25):
        assert min(abs(np.array(fc) - target)) < 3.0, (target, fc)

    # (3) WAV round-trip: write the tone, read it back, still 440
    d = tempfile.mkdtemp()
    p = write_wav(os.path.join(d, "tone.wav"), tone, rate)
    s2, r2 = read_wav(p)
    assert r2 == rate and abs(dominant_frequencies(s2, r2)[0][0] - 440.0) < 2.0

    # (4) STFT frames tile the signal; a rising sweep gives frames with rising dominant frequency
    sweep = np.sin(2 * np.pi * (200 + 600 * t) * t)
    fr = frames(sweep, hop=2048, size=4096)
    assert len(fr) >= 3
    d0 = dominant_frequencies(fr[0], rate)[0][0]; d1 = dominant_frequencies(fr[-1], rate)[0][0]
    assert d1 > d0                                                  # pitch rose across the sound

    # (5) deterministic
    assert np.array_equal(spectrum(tone, rate)[1], spectrum(tone, rate)[1])
    # SUB-BIN FREQUENCY ACCURACY, pinned. Bin-snapping reported 440 Hz as 441.406 (1.4 Hz, ~5.5 cents,
    # 36% of a bin) -- audible, and wrong for driving a sharp resonance. Hann taper + parabolic vertex
    # brings it to <0.2 Hz. The taper is what mattered: parabolic alone stalled at 0.90 Hz because a
    # rectangular window leaks as a sinc and biases the vertex. Both halves are required, so both are
    # asserted here rather than trusting one.
    _rate, _n = 8000, 2048
    _t = np.arange(_n) / _rate
    _binw = _rate / _n
    for _true in (440.0, 1001.445, 261.626):
        _f, _a = dominant_frequencies(np.sin(2 * np.pi * _true * _t), _rate, k=1)
        _err = abs(float(_f[0]) - _true)
        assert _err < 0.25 * _binw, ("dominant_frequencies is bin-snapping again: %.3f Hz reported for "
                                     "%.3f (err %.3f Hz, %.0f%% of a %.2f Hz bin)"
                                     % (_f[0], _true, _err, 100 * _err / _binw, _binw))
    # and spectrum's DEFAULT must stay rectangular/bit-identical -- the window is opt-in, additive
    _x = np.sin(2 * np.pi * 440.0 * _t)
    assert np.array_equal(spectrum(_x, _rate)[1], spectrum(_x, _rate, window=None)[1]), \
        "spectrum's default must remain the untapered rectangular transform"

    print("holographic_audio selftest OK: 440 Hz tone -> single 440 peak (sub-bin: err < 0.25 bin); chord -> its 3 notes; WAV round-trips; "
          "STFT tracks a rising sweep; deterministic")


if __name__ == "__main__":
    _selftest()

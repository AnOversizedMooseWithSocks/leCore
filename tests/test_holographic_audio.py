"""Audio (A1): tone->single peak, chord->constituents, WAV round-trip, STFT tracks a sweep."""
import numpy as np, tempfile, os
from holographic.misc.holographic_audio import read_wav, write_wav, spectrum, dominant_frequencies, frames


def test_pure_tone_single_peak():
    rate = 22050; t = np.arange(rate) / rate
    f, a = dominant_frequencies(0.8 * np.sin(2 * np.pi * 440 * t), rate, k=3)
    assert abs(f[0] - 440.0) < 2.0


def test_chord_reads_constituents():
    rate = 22050; t = np.arange(rate) / rate
    chord = sum(np.sin(2 * np.pi * hz * t) for hz in (440.0, 554.37, 659.25))
    fc, _ = dominant_frequencies(chord, rate, k=4)
    for target in (440.0, 554.37, 659.25):
        assert min(abs(np.array(fc) - target)) < 3.0


def test_wav_roundtrip():
    rate = 16000; t = np.arange(rate) / rate
    tone = 0.7 * np.sin(2 * np.pi * 330 * t)
    d = tempfile.mkdtemp(); p = write_wav(os.path.join(d, "t.wav"), tone, rate)
    s, r = read_wav(p)
    assert r == rate and abs(dominant_frequencies(s, r)[0][0] - 330.0) < 2.0


def test_stft_tracks_sweep_and_is_deterministic():
    rate = 22050; t = np.arange(rate) / rate
    sweep = np.sin(2 * np.pi * (200 + 600 * t) * t)
    fr = frames(sweep, hop=2048, size=4096)
    assert len(fr) >= 3
    assert dominant_frequencies(fr[-1], rate)[0][0] > dominant_frequencies(fr[0], rate)[0][0]
    assert np.array_equal(spectrum(sweep, rate)[1], spectrum(sweep, rate)[1])

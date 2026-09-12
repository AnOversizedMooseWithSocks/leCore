"""The up/down/sideways check on sweep 137's beat detection, and the UP direction it closed.

The check is this repo's close-out habit: DOWN -- does it work on a component of its own input?
UP -- when its input is a component of something larger? SIDEWAYS -- which costumes does it wear?
A missed direction is a missed faculty, so the ones that work are pinned before they rot and the
one that does not is named rather than left as a hole.
"""
import numpy as np
import lecore
import pytest

RATE = 8000


def _clicks(seconds=4.0, period=0.5, rate=RATE, seed=0):
    """A click track: an impulse every `period` seconds. 0.5 s spacing is 120 BPM."""
    rng = np.random.default_rng(seed)
    sig = np.zeros(int(seconds * rate))
    for k in range(int(seconds / period)):
        i = int(k * period * rate)
        sig[i:i + 60] += np.exp(-np.arange(60) / 8.0)
    return sig + 0.01 * rng.normal(size=sig.shape)


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=64, seed=0)


def _n(mind, x, rate=RATE):
    out = mind.onset_detect(x, rate=rate)
    return len(out["times"]) if isinstance(out, dict) and "times" in out else len(out)


def test_the_detector_finds_every_click(mind):
    """Eight clicks at 0.5 s over four seconds. If this drifts, every other number here moves."""
    assert _n(mind, _clicks()) == 8


def test_DOWN_a_segment_of_a_track_is_a_track(mind):
    """Half the buffer must analyse without special-casing -- otherwise streaming or windowed
    analysis is impossible and nothing else would have said so."""
    sig = _clicks()
    assert _n(mind, sig[:len(sig) // 2]) >= 3


def test_UP_a_stereo_buffer_is_the_commonest_input_and_it_used_to_CRASH(mind):
    """THE DIRECTION THIS SWEEP CLOSED. Feeding a 2-channel buffer raised 'boolean index did not
    match indexed array along axis 0' out of the framing step -- an error naming the symptom and
    not the cause, from the single most likely input a demo scener has on disk. Averaging the
    channels is what every onset detector does: a beat is in both channels, so the downmix is the
    mono signal and not an approximation of one. Asserted EQUAL to the mono result, because a
    downmix that changed the answer would be a different bug wearing this fix as a disguise."""
    sig = _clicks()
    mono = _n(mind, sig)
    assert _n(mind, np.stack([sig, sig * 0.9], axis=-1)) == mono
    assert _n(mind, sig) == mono, "the mono path must be untouched"


def test_SIDEWAYS_it_runs_on_a_non_audio_series_but_does_not_WORK_there(mind):
    """THE DIRECTION STILL OPEN, asserted so it cannot be quietly assumed.

    Handed a decimated envelope instead of audio-rate samples, the detector accepts the array and
    returns ZERO onsets. Running-but-not-working is worse than raising, because a caller gets a
    number rather than an error. The cause is structural: this is an STFT flux detector and an
    envelope has already thrown away the spectrum it measures. A general 1-D change-point verb is
    a different faculty, not a parameter on this one -- `detect_regimes` already exists for that.
    If someone ever makes this work on an envelope, THIS TEST is the one to change, on purpose."""
    sig = _clicks()
    envelope = np.abs(sig).reshape(-1, 100).max(axis=1)
    assert _n(mind, envelope, rate=RATE // 100) == 0


def _bpm(mind, rate, hop):
    sig = _clicks(rate=rate)
    out = mind.onset_detect(sig, rate=rate, hop=hop)
    times = out["times"] if isinstance(out, dict) and "times" in out else out
    return float(mind.tempo(list(times))["bpm"])


def test_TEMPO_ACCURACY_IS_BOUNDED_BY_THE_ANALYSIS_FRAME_NOT_BY_THE_ESTIMATOR(mind):
    """The first version of this test asserted |bpm - 120| < 2 and FAILED at 117.19 -- and the
    failure was right. 0.5 s spacing is 120 BPM by construction, so the gap is resolution, not
    error: onset times land on frame boundaries, and a frame is hop/rate seconds long.

    MEASURED across the grid, all on the same 120 BPM click track:
        frame 64.0 ms  ->  117.19 BPM   err 2.34%
        frame 32.0 ms  ->  119.86 BPM   err 0.11%
        frame 16.0 ms  ->  120.14 BPM   err 0.11%
        frame  8.0 ms  ->  120.00 BPM   err 0.00%
    So the estimator is fine and the ANALYSIS RATE is the knob. This asserts the RELATIONSHIP --
    coarse frames are bounded, fine frames are accurate -- rather than one magic constant, because
    a single number would have hidden which of the two was being measured. NOTE for anyone quoting
    beat_sync's headline 0.045%: that belongs to ONE configuration, not to the faculty."""
    coarse = _bpm(mind, rate=8000, hop=512)      # 64 ms frames
    fine = _bpm(mind, rate=16000, hop=128)       # 8 ms frames
    assert abs(coarse - 120.0) < 4.0, "coarse frames should be bounded, got %.2f" % coarse
    assert abs(fine - 120.0) < 0.5, "8 ms frames should be near-exact, got %.2f" % fine
    assert abs(fine - 120.0) < abs(coarse - 120.0), "a finer frame must not be worse"

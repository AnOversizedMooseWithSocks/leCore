"""Tests for holographic_dedoppler -- de-Doppler drift detection (permute + bh_fdr), wired as
the UnifiedMind.detect_drifting faculty (Tarter / Siemion / Cranmer seats)."""
import numpy as np

from holographic.sampling_and_signal.holographic_dedoppler import _selftest, detect_drifting
from holographic.misc.holographic_unified import UnifiedMind


def test_dedoppler_module_selftest():
    # the module's own measured claims: recovery vs stationary, look-elsewhere control, ROC at the
    # field's S/N>=10 bar, ON-OFF cadence rejecting strong RFI, and determinism.
    _selftest()


def _waterfall(T, F, drift, chan, amp, seed, rfi_chan=None, rfi_amp=0.0, on=True):
    r = np.random.default_rng(seed)
    wf = r.standard_normal((T, F))
    if on:
        for t in range(T):
            wf[t, int(round(chan + drift * t)) % F] += amp     # the drifting narrowband signal
    if rfi_chan is not None:
        wf[:, rfi_chan] += rfi_amp                             # stationary RFI: every frame, no drift
    return wf


def test_detect_drifting_faculty_finds_signal_controls_noise_and_is_deterministic():
    m = UnifiedMind(dim=256, seed=0)
    T, F = 24, 96
    wf = _waterfall(T, F, drift=1.5, chan=40, amp=2.4, seed=11)     # integrated S/N ~12 (the field's regime)
    det = m.detect_drifting(wf, alpha=0.01)
    assert any(abs(d["drift"] - 1.5) < 0.6 and abs(d["channel"] - 40) <= 1 for d in det)   # found at the right drift
    # pure noise -> the dependent-FDR veto controls false alarms (a naive per-cell threshold would fire ~always)
    noise_det = m.detect_drifting(np.random.default_rng(99).standard_normal((T, F)), alpha=0.01)
    assert len(noise_det) == 0
    # deterministic (Macklin's tie-break discipline): same input -> identical detections
    assert m.detect_drifting(wf, alpha=0.01) == det


def test_detect_drifting_cadence_rejects_persistent_rfi():
    m = UnifiedMind(dim=256, seed=0)
    T, F = 24, 96
    # ON: a drifting signal at channel 30 + a STRONG stationary RFI at channel 70 (detectable on its own)
    on = _waterfall(T, F, drift=1.5, chan=30, amp=2.4, seed=21, rfi_chan=70, rfi_amp=2.4)
    off = _waterfall(T, F, drift=1.5, chan=30, amp=0.0, seed=121, rfi_chan=70, rfi_amp=2.4, on=False)
    no_cadence = m.detect_drifting(on, alpha=0.01)
    assert any(abs(d["drift"]) < 1e-9 and abs(d["channel"] - 70) <= 1 for d in no_cadence)   # RFI fools a no-cadence run
    with_cadence = m.detect_drifting(on, alpha=0.01, off=off)
    assert any(abs(d["drift"] - 1.5) < 0.6 and abs(d["channel"] - 30) <= 1 for d in with_cadence)  # signal kept
    assert not any(abs(d["drift"]) < 1e-9 and abs(d["channel"] - 70) <= 1 for d in with_cadence)    # RFI vetoed

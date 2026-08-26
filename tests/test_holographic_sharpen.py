"""CI wrapper for looping negative-lobe sharpening (XDATA-3). The module ships its asserts in `_selftest`: a
Gaussian-blurred 1-D signal (slow component + a high-frequency burst) is re-sharpened by a looping Van Cittert
correction -- recovering the detail and converging with no noise, with the discrepancy-principle guard stopping
near the optimum and beating an unguarded run under noise (the over-sharpening kept negative), while an over-large
step diverges into ringing. This collects that check into the suite."""
from holographic.rendering.holographic_sharpen import _selftest


def test_holographic_sharpen_selftest():
    _selftest()

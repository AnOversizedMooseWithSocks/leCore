"""CI wrapper for robust accumulation (harmonic weights + firefly clamping). The module ships its asserts in
`_selftest`: the harmonic (1/n) running average converges on a stationary stream where a fixed-alpha EMA plateaus
(with the kept caveat that the EMA wins on a drifting target), and firefly clamping is robust to injected outliers
with no loss on clean data. This collects that check into the suite."""
from holographic.misc.holographic_accumulate import _selftest


def test_holographic_accumulate_selftest():
    _selftest()

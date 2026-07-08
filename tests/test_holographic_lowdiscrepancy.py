"""CI wrapper for low-discrepancy sampling. The module ships its asserts in `_selftest`: Roberts'
generalised golden-ratio (R) sequence covers a domain more evenly than random (lower dispersion),
integrates a smooth function with lower error than plain Monte Carlo at the same sample count (the
coverage payoff), and is deterministic and progressive. This collects that check into the suite."""
from holographic.sampling_and_signal.holographic_lowdiscrepancy import _selftest


def test_holographic_lowdiscrepancy_selftest():
    _selftest()

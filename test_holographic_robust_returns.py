"""CI wrapper for robust reward/value accumulation in the creature brain (D2). The module ships its asserts in
`_d2_selftest`: with robust_returns on, an outlier reward is winsorised before it folds into a prototype's
running-mean value, giving markedly lower value error under outlier rewards with no cost on clean data; the flag
survives save/load. Off by default (bit-identical plain running average). This collects that check."""
from holographic_creature import _d2_selftest


def test_holographic_robust_returns_selftest():
    _d2_selftest()

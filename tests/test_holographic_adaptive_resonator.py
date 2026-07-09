"""CI wrapper for the resonator's opt-in early-stop (ADAPT-2). The SBC module ships the asserts in
`_adapt2_selftest`: stopping the resonator the moment its picks VERIFY matches fixed-count accuracy at far lower
average iteration cost on easily-solved factorizations, and is a no-op (identical result, no harm) on hard /
mostly-unsolved ones. This collects that check into the suite."""
from holographic.misc.holographic_sbc import _adapt2_selftest


def test_holographic_sbc_adapt2_selftest():
    _adapt2_selftest()

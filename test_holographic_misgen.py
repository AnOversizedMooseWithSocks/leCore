"""CI wrapper for the B1 no-op (cross-cutting MIS-1 -> steered text generation). The module ships its asserts in
`_selftest`: on a loop-trap corpus the verifier escapes the greedy loop (the setup is real), but the MIS
balance-heuristic combination of predictor and verifier scores matches verifier-only EXACTLY -- the predictor is
already spent gating the candidate beam, so there is nothing for the balance heuristic to balance. A kept no-op.
This collects that check."""
from holographic_misgen import _selftest


def test_holographic_misgen_noop_selftest():
    _selftest()

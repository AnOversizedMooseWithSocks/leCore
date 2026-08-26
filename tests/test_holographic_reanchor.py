"""CI wrapper for the re-anchoring audit (RAY-2). The module ships its asserts in `_selftest`: on a directed linked
list, the engine's gated_traverse with a re-anchored step recovers every hop, while the same traversal with a raw
(no-cleanup) step collapses almost immediately -- re-anchoring is load-bearing for deep traversal. This collects that
check into the suite."""
from holographic.misc.holographic_reanchor import _selftest


def test_holographic_reanchor_selftest():
    _selftest()

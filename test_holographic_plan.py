"""CI wrapper for corridor planning (holographic_plan): bake a short route on the directed substrate, run
it cheap, re-anchor when the throughput gate trips. The selftest proves an at-cap corridor decodes to the
full direction sequence, an over-cap corridor reports only its reliable prefix (no overclaiming), and
replan_needed fires exactly on exhaustion / low throughput / a blocked tile."""
from holographic_plan import _selftest


def test_plan_selftest():
    _selftest()

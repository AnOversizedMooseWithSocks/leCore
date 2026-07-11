"""Test configuration for the leCore suite.

WHY THIS FILE EXISTS: a handful of tests are irreducibly slow -- they assert a real contract (a bias/variance
crossover that only separates at high dimension, a maze that only starves at full training budget) that cannot be
shrunk without putting the assertion on a numeric knife-edge. Rather than delete the coverage or let it dominate
every local run, they are marked `@pytest.mark.slow` and DESELECTED BY DEFAULT (see `addopts` in pytest.ini).

To run them anyway:
    pytest -m ""            # everything, slow included (this is what CI's weekly/tag "full" run does)
    pytest -m slow          # ONLY the slow ones
Normal runs (`pytest`, and CI's per-change runs) skip them automatically.

The bar for adding `@pytest.mark.slow` is deliberately high: FIRST try to make the test fast while preserving its
contrast (that fixed the 140 s maze test down to 42 s by finding the cheapest config that still starves-then-cracks
with a real margin). Only mark a test slow when shrinking it would make the assertion fragile -- and say so in a
comment on the mark, with the measurement.
"""
import pytest


def pytest_configure(config):
    # Register the marker so `--strict-markers` (if ever enabled) and `-m slow` both know it, and so pytest does
    # not warn about an unknown mark.
    config.addinivalue_line(
        "markers",
        "slow: an irreducibly slow test (high-dimension bake, full-budget training). Deselected by default; "
        "run with `pytest -m \"\"` or `-m slow`. Marking one requires a comment justifying why it can't be sped up.",
    )

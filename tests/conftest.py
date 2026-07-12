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
import os
import signal
import threading

import pytest


_BUDGET_SECONDS = 15
_HAVE_SIGALRM = hasattr(signal, "SIGALRM")


def _run_slow_forced(config=None):
    """True when the long tests are forced on -- via --run-slow or LECORE_RUN_SLOW=1. Then the budget is disabled.

    The 15 s per-test budget (below) is a safety net ON TOP OF the `slow` marker: the marker deselects tests we KNOW
    are slow up front (cheap); the watchdog catches the ones we DON'T know about yet (a new slow test, or a change
    that balloons a runtime) instead of letting them dominate a run. Both switches lift the budget AND select slow."""
    if os.environ.get("LECORE_RUN_SLOW", "").strip() not in ("", "0", "false", "False"):
        return True
    if config is not None:
        try:
            return bool(config.getoption("--run-slow"))
        except (ValueError, KeyError):
            return False
    return False


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="run the irreducibly-slow tests AND disable the 15s per-test timeout (equivalently: LECORE_RUN_SLOW=1).",
    )


class _Timeout(Exception):
    pass


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Wrap each test's call phase in the 15 s budget. On overrun -> pytest.skip (not a failure), unless forced.

    DEPENDENCY-FREE (core is stdlib-only; test infra should be too -- no pytest-timeout wheel). On POSIX we use
    SIGALRM (precise, interrupts even C-bound loops that hold the GIL); elsewhere a timer thread turns a
    finished-too-late call into a skip (it cannot interrupt a stuck call, so POSIX CI gets the hard guarantee)."""
    if _run_slow_forced(item.config):
        yield
        return
    if _HAVE_SIGALRM:
        def _alarm(signum, frame):
            raise _Timeout()
        old_handler = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, _BUDGET_SECONDS)
        try:
            outcome = yield
            excinfo = outcome.excinfo
            if excinfo is not None and issubclass(excinfo[0], _Timeout):
                outcome.force_exception(pytest.skip.Exception(
                    "exceeded the %ds per-test budget; run with --run-slow or LECORE_RUN_SLOW=1 "
                    "(and consider @pytest.mark.slow)" % _BUDGET_SECONDS))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        timed_out = threading.Event()
        timer = threading.Timer(_BUDGET_SECONDS, timed_out.set)
        timer.start()
        try:
            outcome = yield
            if timed_out.is_set() and outcome.excinfo is None:
                outcome.force_exception(pytest.skip.Exception(
                    "exceeded the %ds per-test budget; run with --run-slow or LECORE_RUN_SLOW=1" % _BUDGET_SECONDS))
        finally:
            timer.cancel()


def pytest_configure(config):
    # Register the marker so `--strict-markers` (if ever enabled) and `-m slow` both know it, and so pytest does
    # not warn about an unknown mark.
    config.addinivalue_line(
        "markers",
        "slow: an irreducibly slow test (high-dimension bake, full-budget training). Deselected by default; "
        "run with `pytest -m \"\"` or `-m slow`. Marking one requires a comment justifying why it can't be sped up.",
    )
    # If the long tests are forced on, also SELECT the `slow`-marked ones (override the default `-m 'not slow'`), so
    # a single flag both lifts the budget and includes the known-slow tests -- one switch, not two.
    if _run_slow_forced(config):
        config.option.markexpr = ""

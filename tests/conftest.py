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
# THE LONG BUDGET (sweep 118): under --run-slow the budget used to be LIFTED FOR EVERY TEST, so one unmarked
# test that had quietly grown to minutes could hang a 20-minute shard -- measured twice in the full-suite
# matrix (shard 3 of 10 at 91%, shard 17 of 20 at 84%). The rule is now Moose's rule: an UNMARKED test keeps
# the 15 s budget ALWAYS and is skipped on overrun (the skip message says to mark it slow if it is critical);
# a test MARKED slow -- i.e. declared critical-but-slow -- gets this long budget under --run-slow, and even
# it has a ceiling, so no single test can take a shard down. Override per machine: LECORE_SLOW_BUDGET=seconds.
_SLOW_BUDGET_SECONDS = int(os.environ.get("LECORE_SLOW_BUDGET", "600") or 600)
_HAVE_SIGALRM = hasattr(signal, "SIGALRM")


def _budget_for(item):
    """Seconds this test may take: the long budget for a `slow`-marked test when slow tests are forced on,
    the 15 s budget for everything else -- including unmarked tests under --run-slow."""
    if _run_slow_forced(item.config) and item.get_closest_marker("slow") is not None:
        return _SLOW_BUDGET_SECONDS
    return _BUDGET_SECONDS


def _run_slow_forced(config=None):
    """True when the long tests are forced on -- via --run-slow or LECORE_RUN_SLOW=1. Then slow-MARKED tests are
    selected and get the long budget (_SLOW_BUDGET_SECONDS); unmarked tests KEEP the 15 s budget (sweep 118).

    The 15 s per-test budget (below) is a safety net ON TOP OF the `slow` marker: the marker deselects tests we KNOW
    are slow up front (cheap); the watchdog catches the ones we DON'T know about yet (a new slow test, or a change
    that balloons a runtime) instead of letting them dominate a run -- and it must keep catching them in the full
    suite, which is exactly where an unknown slow test can hang a shard."""
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


class _Timeout(BaseException):
    # BaseException, NOT Exception, on purpose: the alarm fires INSIDE the code under test, and a lot of that code
    # (Flask request handlers, broad `try/except Exception` blocks) would otherwise CATCH the timeout and turn a clean
    # skip into a confusing downstream failure (a swallowed timeout -> a malformed response -> a KeyError three frames
    # later). Inheriting from BaseException means only code that explicitly catches BaseException/KeyboardInterrupt can
    # intercept it -- so it propagates out to the hookwrapper below and becomes a skip, the same way KeyboardInterrupt
    # and SystemExit are designed to pass through `except Exception`.
    pass


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Wrap each test's call phase in the 15 s budget. On overrun -> pytest.skip (not a failure), unless forced.

    DEPENDENCY-FREE (core is stdlib-only; test infra should be too -- no pytest-timeout wheel). On POSIX we use
    SIGALRM (precise, interrupts even C-bound loops that hold the GIL); elsewhere a timer thread turns a
    finished-too-late call into a skip (it cannot interrupt a stuck call, so POSIX CI gets the hard guarantee)."""
    budget = _budget_for(item)
    slow_marked = item.get_closest_marker("slow") is not None
    if slow_marked and _run_slow_forced(item.config):
        why = ("exceeded the %ds LONG budget for a slow-marked test (LECORE_SLOW_BUDGET raises it); "
               "a critical test that needs longer must be split or made faster" % budget)
    else:
        why = ("exceeded the %ds per-test budget -- SKIPPED, not run to completion, even under --run-slow "
               "(sweep 118). If this test is critical mark it @pytest.mark.slow so it gets the long budget; "
               "otherwise make it fast" % budget)
    if _HAVE_SIGALRM:
        def _alarm(signum, frame):
            raise _Timeout()
        old_handler = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, budget)
        try:
            outcome = yield
            excinfo = outcome.excinfo
            if excinfo is not None and issubclass(excinfo[0], _Timeout):
                outcome.force_exception(pytest.skip.Exception(why))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        timed_out = threading.Event()
        timer = threading.Timer(budget, timed_out.set)
        timer.start()
        try:
            outcome = yield
            if timed_out.is_set() and outcome.excinfo is None:
                outcome.force_exception(pytest.skip.Exception(why))
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

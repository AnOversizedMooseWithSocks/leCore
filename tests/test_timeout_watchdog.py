"""The 15s per-test timeout watchdog (tests/conftest.py): a test that overruns is SKIPPED, forced runs complete, and
-- the bug that broke CI -- an overrun inside a broad `except Exception` is NOT swallowed (the timeout is a
BaseException, so it propagates to a clean skip instead of corrupting the response three frames later)."""
import time

import pytest

import conftest


def test_timeout_is_a_baseexception_not_exception():
    """The load-bearing property: `except Exception` must NOT catch the timeout (that swallowing turned CI skips into
    confusing KeyErrors). Only BaseException handlers may -- same design as KeyboardInterrupt/SystemExit."""
    assert issubclass(conftest._Timeout, BaseException)
    assert not issubclass(conftest._Timeout, Exception), \
        "the timeout must sit OUTSIDE Exception so app `except Exception` handlers cannot swallow it"


def test_except_exception_does_not_swallow_the_timeout():
    """Concretely: raising the timeout through a broad `except Exception` re-raises it, it is not caught."""
    caught_by_exception = False
    try:
        try:
            raise conftest._Timeout()
        except Exception:                       # the exact pattern Flask/handlers use
            caught_by_exception = True
    except conftest._Timeout:
        pass                                    # correct: it slipped past `except Exception`
    assert not caught_by_exception, "`except Exception` must not catch the timeout"


def test_budget_constant_is_fifteen_seconds():
    """The documented contract: a hard 15s budget."""
    assert conftest._BUDGET_SECONDS == 15


def test_fast_test_runs_normally():
    """A fast test is unaffected by the watchdog."""
    time.sleep(0.01)
    assert True

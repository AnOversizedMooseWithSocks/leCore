"""Regression traps for the boundary seams (work plan item 6.2).

The item predicted, from the SHAPE of three earlier defects, that the next failures would sit at layer
boundaries -- "a condition fully detectable on one side becomes an answer on the other" -- and named three
in order: the service boundary, the worker boundary, and persistence.

    worker boundary   AUDITED -- the no-code-crosses-the-wire rule held only because JSON could not
                      serialise a function. Now an explicit refusal (see test_llm_tool.py).
    service boundary  AUDITED -- this file. The service emitted BARE NaN / Infinity, which are not in the
                      JSON grammar. Python's parser is lenient, so it looked fine from inside.
    persistence       AUDITED -- CLEAN. Non-finite values survive to_state/from_state exactly. Recorded as
                      a negative result: two of three predicted seams were real, one was not.
"""
import json
import math
import sys

import numpy as np
import pytest

import lecore

sys.path.insert(0, ".")
from holographic_service import _json_default, _jsonable  # noqa: E402


# --------------------------------------------------------------------------------------
# The service boundary.
# --------------------------------------------------------------------------------------

def test_non_finite_results_become_null_not_bare_nan():
    """THE DEFECT. json.dumps emits bare `NaN` / `Infinity` by default and those are NOT in the JSON
    grammar. Python's own json.loads accepts them, which is exactly why this survived: every in-process
    test passed while Go, Java and every browser's JSON.parse would reject the response outright."""
    for value in (float("nan"), float("inf"), float("-inf"), np.float64("nan")):
        assert _jsonable(value) is None, "%r did not become null" % value


def test_non_finite_is_sanitised_recursively():
    assert _jsonable([1.0, float("nan")]) == [1.0, None]
    assert _jsonable({"m": float("inf")}) == {"m": None}
    assert _jsonable({"a": [{"b": float("nan")}]}) == {"a": [{"b": None}]}


def test_finite_values_are_untouched():
    # A sanitiser that also mangles good data is a worse bug than the one it fixes.
    assert _jsonable(1.5) == 1.5
    assert _jsonable(0.0) == 0.0
    assert _jsonable([1, 2.5, "x", True, None]) == [1, 2.5, "x", True, None]
    assert _jsonable(np.float64(2.5)) == 2.5


def test_a_response_body_survives_a_strict_parser():
    # The end-to-end property: what the service writes must parse under allow_nan=False, which is what a
    # non-Python client effectively enforces.
    payload = {"ok": True, "result": _jsonable({"z": float("nan"), "rows": [1.0, float("inf")]})}
    body = json.dumps(payload, default=_json_default, allow_nan=False)
    assert json.loads(body) == payload
    assert "NaN" not in body and "Infinity" not in body


def test_the_writer_enforces_strictness_rather_than_hoping():
    """allow_nan=False on the response writer turns a silent corruption into a loud failure. Pinned because
    the sanitiser above and this flag are belt AND braces: the lesson from the worker boundary is that an
    accidental guarantee is one refactor away from not being a guarantee."""
    import inspect

    import holographic_service
    src = inspect.getsource(holographic_service)
    assert "allow_nan=False" in src, "the strict-JSON enforcement was removed from the response writer"


def test_non_serialisable_results_still_get_a_typed_summary():
    # The guard that already worked, kept working: /invoke must never crash on an odd return value.
    mind = lecore.UnifiedMind(dim=64, seed=0)
    out = _jsonable(mind.database(dim=64))
    assert isinstance(out, dict) and "type" in out


# --------------------------------------------------------------------------------------
# Persistence -- the predicted seam that turned out to be clean.
# --------------------------------------------------------------------------------------

def test_persistence_preserves_non_finite_values():
    """A NEGATIVE RESULT, kept. Persistence was the third predicted seam and it is CLEAN -- non-finite
    values survive to_state/from_state exactly. Recorded so nobody re-audits it on the strength of the
    prediction alone: two of three predicted seams were real, one was not, and that ratio is the honest
    accuracy of the prediction method."""
    mind = lecore.UnifiedMind(dim=64, seed=0)
    db = mind.database(dim=64)
    mind.db_query("CREATE DATABASE d", db)
    mind.db_query("CREATE TABLE d.t (v)", db)
    for value in (1, 2.5, float("nan"), float("inf")):
        db.insert("d.t", {"v": value})

    restored = type(db).from_state(db.to_state())
    before = [r["v"] for r in mind.db_query("SELECT * FROM d.t", db)]
    after = [r["v"] for r in mind.db_query("SELECT * FROM d.t", restored)]
    assert len(before) == len(after) == 4
    for a, b in zip(before, after):
        if isinstance(a, float) and math.isnan(a):
            assert isinstance(b, float) and math.isnan(b), "NaN did not survive persistence"
        else:
            assert a == b, "%r != %r across to_state/from_state" % (a, b)


# --------------------------------------------------------------------------------------
# Placement pre-gate (work plan item 3.3, second half) -- REFUTED, pinned.
# --------------------------------------------------------------------------------------

def test_break_even_n_is_not_a_constant():
    """THE REFUTATION. A pre-gate was proposed that would skip placement -- and skip measuring the baseline
    -- whenever n_calls fell below the tier's break-even, quoted as 1.63 and asserted to be "independent of
    the baseline". It is not: break_even_n is (setup cost)/(per-call saving), and the saving depends
    entirely on how slow the baseline is."""
    mind = lecore.UnifiedMind(dim=64, seed=0)
    values = [mind.machine_place_unit("t2_baked_grid", b, 100).get("break_even_n")
              for b in (1e3, 1e4, 1e5)]
    assert len(set(values)) == len(values), "break_even_n stopped varying with the baseline: %r" % values
    assert values[0] > values[-1], "break_even_n should FALL as the baseline slows: %r" % values


def test_no_n_calls_threshold_can_be_baseline_free():
    """And there is no floor to retreat to. break_even_n tends to 0 as the baseline grows, and use_unit is
    already True at n_calls=1 for a slow enough baseline -- so a gate that skipped at n_calls<1.63 would
    wrongly refuse the exact case where placement pays most: a slow operation called once."""
    mind = lecore.UnifiedMind(dim=64, seed=0)
    for baseline in (1e6, 1e9):
        report = mind.machine_place_unit("t2_baked_grid", baseline, 1)
        assert report["break_even_n"] < 1.0
        assert report["use_unit"] is True, "placement refused at n=1 for a very slow baseline"


def test_the_refutation_is_recorded_where_it_will_be_reproposed():
    import inspect
    from holographic.misc import holographic_machinemodel as mm
    assert "THERE IS NO BASELINE-FREE PRE-GATE" in inspect.getsource(mm)

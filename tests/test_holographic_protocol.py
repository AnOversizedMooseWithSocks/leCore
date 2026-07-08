"""D1: protocol-as-data anti-pattern auditing (holographic_protocol.py) -- the honesty discipline as a
structural property of a program vector, read back holographically and checked for the search-without-null
and select-then-score-without-split anti-patterns."""
import numpy as np

from holographic.agents_and_reasoning.holographic_machine import HoloMachine
from holographic.scene_and_pipeline.holographic_protocol import build_protocol, audit_protocol, protocol_role_sequence, SEARCH, NULL, FDR, SPLIT, DECIDE, _selftest


def test_selftest_passes():
    _selftest()


def _mk():
    return HoloMachine(dim=4096, seed=3)


def test_structure_round_trips_from_the_vector():
    # the audit reads the step structure BACK FROM the program vector (unbind+cleanup), not from the source
    M = _mk()
    steps = ["encode", "combination_search", "oos_split", "calibrated_null", "fdr", "decide"]
    pv, n = build_protocol(M, steps)
    recovered = [f for (f, r) in audit_protocol(M, pv, n)["sequence"]]
    assert recovered == steps


def test_complete_protocol_is_sound():
    M = _mk()
    pv, n = build_protocol(M, ["encode", "combination_search", "oos_split", "calibrated_null", "fdr", "decide"])
    a = audit_protocol(M, pv, n)
    assert a["sound"] and a["violations"] == []
    assert {SEARCH, NULL, FDR, SPLIT, DECIDE}.issubset(set(a["roles"]))


def test_search_without_null_is_flagged():
    # the canonical artifact-factory: a search that proposes candidates with no procedure-matched null
    M = _mk()
    pv, n = build_protocol(M, ["encode", "combination_search", "oos_split", "fdr", "decide"])
    a = audit_protocol(M, pv, n)
    assert not a["sound"]
    assert any(code == "search_without_null" for code, _msg in a["violations"])


def test_select_then_score_without_split_is_flagged():
    # selecting then scoring with no out-of-sample split between them (the data-flow stand-in, order-based)
    M = _mk()
    pv, n = build_protocol(M, ["encode", "combination_search", "calibrated_null", "fdr", "decide"])
    a = audit_protocol(M, pv, n)
    assert not a["sound"]
    assert any(code == "select_then_score_same_data" for code, _msg in a["violations"])


def test_searched_family_without_fdr_is_flagged():
    M = _mk()
    pv, n = build_protocol(M, ["encode", "recall", "oos_split", "calibrated_null", "decide"])
    a = audit_protocol(M, pv, n)
    assert not a["sound"]
    assert any(code == "search_decide_without_fdr" for code, _msg in a["violations"])


def test_no_search_protocol_is_not_flagged():
    # targeted, not trigger-happy: no search step -> no honesty obligation (a restoration loop is fine)
    M = _mk()
    pv, n = build_protocol(M, ["datafit", "denoise"])
    a = audit_protocol(M, pv, n)
    assert a["sound"]

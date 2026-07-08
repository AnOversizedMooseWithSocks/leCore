"""Tests for holographic_queryprog -- VSA programs as database objects (PR1-PR6)."""
import numpy as np
from holographic.agents_and_reasoning.holographic_queryprog import ProgramCatalog, register_system_program
from holographic.agents_and_reasoning.holographic_query import QueryError


def _cat():
    cat = ProgramCatalog(dim=2048, seed=0)
    cat.install("prototype", [("LOAD", "color"), ("HALT", None)],
                doc="build a prototype vector that clusters similar rows by their color",
                inputs=["color"], outputs=["color"], handlers=[], data=["color"])
    cat.install("normalize_tag", [("LOAD", "color"), ("APPLY", "normalize"), ("HALT", None)],
                doc="normalize and tag a group of records for anomaly detection",
                inputs=["color"], outputs=["color"], handlers=["normalize"], faculties=["normalize"])
    return cat


def test_pr5_install_and_pr1_list():
    cat = _cat()
    names = {r["name"] for r in cat.list()}
    assert names == {"prototype", "normalize_tag"}
    assert all("domain" in r and "tier" in r for r in cat.list())
    assert cat.list(tier="user") and cat.list(tier="system") == []


def test_pr4_find_by_meaning():
    cat = _cat()
    hits = cat.find("group a series of similar things into clusters")
    assert hits[0]["name"] == "prototype"
    assert hits[0]["_confidence"] >= hits[-1]["_confidence"]


def test_pr3_explain_is_a_dry_run():
    cat = _cat()
    ex = cat.explain("normalize_tag")
    assert "normalize" in ex["faculties_called"] and ex["n_steps"] >= 1
    assert "trace" in ex


def test_pr6_execute_sandboxed_and_confident():
    cat = _cat()
    out = cat.execute("prototype", [{"color": "red"}, {"color": "red"}])
    assert "result" in out and 0.0 <= out["_confidence"] <= 1.0 and out["n_steps"] >= 1


def test_pr6_sandbox_whitelist():
    # prototype declared no handlers -> its run can never be handed the 'normalize' faculty
    cat = _cat()
    assert cat._programs["prototype"]["handlers"] == []
    assert cat._programs["normalize_tag"]["handlers"] == ["normalize"]


def test_pr6_step_bound():
    cat = _cat()
    out = cat.execute("normalize_tag", [{"color": "red"}], max_steps=1)
    assert out["n_steps"] <= 2                                    # bounded, does not spin


def test_system_program_read_only():
    cat = _cat()
    register_system_program(cat, "builtin", [("LOAD", "color"), ("HALT", None)],
                            doc="a built-in", inputs=["color"], outputs=["color"], handlers=[], data=["color"])
    try:
        cat.uninstall("builtin"); assert False
    except QueryError:
        pass
    assert cat.uninstall("prototype") is True                     # user program removes fine


def test_no_such_program_errors():
    cat = _cat()
    for fn in (lambda: cat.explain("nope"), lambda: cat.execute("nope", []), lambda: cat.uninstall("nope")):
        try:
            fn(); assert False
        except QueryError:
            pass

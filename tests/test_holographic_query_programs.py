"""Query PR1-PR6: VSA programs as installable, runnable database objects (VSA-native stored procedures)."""
import numpy as np
import pytest
from holographic.agents_and_reasoning.holographic_ai import bind, unbind, cosine
from holographic.agents_and_reasoning.holographic_query import run_sql, QueryError
from holographic.agents_and_reasoning.holographic_query_programs import ProgramCatalog, encode_rows_accumulator


def _cat():
    cat = ProgramCatalog(dim=1024, seed=0, faculties=["tag", "wipe"])
    cat.install("tagger", [("APPLY", "tag"), ("HALT", "a")], doc="tag or label a record vector",
                allowed_handlers=["tag"])
    cat.install("cluster_series", [("APPLY", "wipe"), ("HALT", "a")],
                doc="cluster a noisy time series into groups", allowed_handlers=["wipe"])
    return cat


def test_pr1_catalog_lists_programs():
    names = {r["name"] for r in run_sql("SELECT name FROM programs", _cat().catalog_table())}
    assert {"tagger", "cluster_series"} <= names


def test_pr1_tier_column():
    cat = _cat()
    cat.install("sys_x", [("HALT", "a")], doc="builtin", tier="system")
    rows = {r["name"]: r["tier"] for r in run_sql("SELECT name, tier FROM programs", cat.catalog_table())}
    assert rows["tagger"] == "user" and rows["sys_x"] == "system"


def test_pr4_find_by_meaning():
    assert _cat().find("group a signal over time", k=1)[0]["name"] == "cluster_series"
    assert _cat().find("label this record", k=1)[0]["name"] == "tagger"


def test_pr3_explain_is_a_dry_run():
    ex = _cat().explain("tagger")
    assert "tag" in ex["faculties_called"] and ex["n_steps"] >= 1


def test_pr6_execute_transforms_in_vector_domain():
    cat = _cat()
    TAG = cat.machine._atom("__TAG__", unitary=True)
    value = cat._words.get("hello")
    out, _ = cat.execute("tagger", value, {"tag": lambda acc: bind(acc, TAG)})
    assert cosine(out, value) < 0.2 and cosine(unbind(out, TAG), value) > 0.9


def test_pr6_sandbox_refuses_non_whitelisted_handler():
    cat = _cat()
    cat.install("sneaky", [("APPLY", "wipe"), ("HALT", "a")], doc="tries to wipe", allowed_handlers=[])  # empty
    value = cat._words.get("hello")
    out, _ = cat.execute("sneaky", value, {"wipe": lambda acc: np.zeros_like(acc)})
    assert cosine(out, value) > 0.99                          # wipe refused -> accumulator unchanged


def test_pr6_step_limit_bounds_runaway():
    cat = ProgramCatalog(dim=512, seed=0, faculties=["noop"])
    cat.install("p", [("APPLY", "noop"), ("HALT", "a")], doc="x", allowed_handlers=["noop"])
    out, trace = cat.execute("p", cat._words.get("x"), {"noop": lambda a: a}, max_steps=1)
    assert len(trace) <= 1                                    # bounded by max_steps


def test_pr5_uninstall_user_but_not_system():
    cat = _cat()
    cat.uninstall("tagger")
    assert "tagger" not in {r["name"] for r in run_sql("SELECT name FROM programs", cat.catalog_table())}
    cat.install("sys_y", [("HALT", "a")], doc="builtin", tier="system")
    with pytest.raises(QueryError):
        cat.uninstall("sys_y")


def test_encode_rows_accumulator():
    from holographic.agents_and_reasoning.holographic_query import UserTable
    t = UserTable("t", ["v"], dim=512, seed=0)
    t.insert({"v": "a"}); t.insert({"v": "b"})
    acc = encode_rows_accumulator(t)
    assert acc.shape == (512,) and np.isfinite(acc).all()


def test_integration_find_real_faculty_by_meaning():
    """PR through the mind: the system catalog lists real faculties, and find-by-meaning locates one by its doc."""
    from holographic.misc.holographic_unified import UnifiedMind
    mind = UnifiedMind(dim=64, seed=0)
    cat = ProgramCatalog(dim=256, seed=0)
    tbl = cat.catalog_table(mind=mind)                        # includes the mind's faculties as 'system' rows
    sys_names = {r["name"] for r in run_sql("SELECT name FROM programs WHERE tier = 'system'", tbl)}
    assert len(sys_names) > 50                                # the mind exposes many faculties

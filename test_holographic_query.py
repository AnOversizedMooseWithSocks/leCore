"""Query Interface Phases 1-3: projection core, SQL subset, fuzzy WHERE + confidence."""
import numpy as np
from holographic_query import from_rows, project, Query, run_sql, parse_sql


def _table():
    rows = [
        {"name": "gold", "colour": "yellow", "density": 19300},
        {"name": "copper", "colour": "orange", "density": 8960},
        {"name": "silver", "colour": "grey", "density": 10490},
        {"name": "iron", "colour": "grey", "density": 7870},
        {"name": "lead", "colour": "grey", "density": 11340},
    ]
    return from_rows(rows, ["name", "colour", "density"], dim=2048, seed=0)


def test_round_trip_projection():
    t = _table()
    p = project(t.records[0], ["name", "colour"], t)
    assert p["name"] == "gold" and p["colour"] == "yellow"


def test_exact_where_order_limit():
    t = _table()
    res = run_sql("SELECT name, density FROM materials WHERE density > 9000 ORDER BY density LIMIT 2", t)
    assert [r["name"] for r in res] == ["gold", "lead"]


def test_exact_string_where():
    t = _table()
    assert [r["name"] for r in run_sql("SELECT name FROM materials WHERE colour = 'yellow'", t)] == ["gold"]


def test_fuzzy_where_with_confidence():
    t = _table()
    res = Query().select("name", "colour").where("colour", "~", "grey").order_by("similarity").run(t)
    assert {"silver", "iron", "lead"} <= {r["name"] for r in res}
    assert all("_confidence" in r for r in res) and res[0]["_confidence"] > 0.5


def test_sql_parser_subset():
    plan = parse_sql("SELECT a, b FROM t WHERE c ~ 'x' ORDER BY d ASC LIMIT 3")
    assert plan["select"] == ["a", "b"] and plan["where"] == ("pred", "c", "~", "x")
    assert plan["order"] == ("d", False) and plan["limit"] == 3


def test_deterministic():
    t = _table()
    q = "SELECT name FROM materials WHERE density > 9000 ORDER BY density"
    assert run_sql(q, t) == run_sql(q, t)


def test_group_by_count_avg_centroid():
    t = _table()
    grp = run_sql("SELECT colour, COUNT(*), AVG(density) FROM m GROUP BY colour ORDER BY colour ASC", t)
    by = {r["colour"]: r for r in grp}
    assert by["grey"]["COUNT(*)"] == 3 and abs(by["grey"]["AVG(density)"] - 9900.0) < 1e-6
    assert by["grey"]["_centroid"] is not None                # the group's bundle (VSA prototype)


def test_global_aggregates_exact():
    t = _table()
    g = run_sql("SELECT MIN(density), MAX(density), SUM(density) FROM m", t)[0]
    assert g["MIN(density)"] == 7870 and g["MAX(density)"] == 19300 and g["SUM(density)"] == 57960


def test_aggregate_parse():
    plan = parse_sql("SELECT colour, COUNT(*), AVG(density) FROM t GROUP BY colour")
    assert plan["group"] == "colour" and ("COUNT", "*", "COUNT(*)") in plan["aggs"]
    assert ("AVG", "density", "AVG(density)") in plan["aggs"] and plan["select"] == ["colour"]


def test_capability_registry_is_queryable():
    from holographic_unified import UnifiedMind
    m = UnifiedMind(dim=256, seed=0)
    reg = m.capabilities()
    assert len(reg) > 100                                       # the mind has many faculties
    fc = run_sql("SELECT name FROM actions WHERE domain = 'forecasting'", reg)
    names = {r["name"] for r in fc}
    assert "forecast" in names and "analog_forecaster" in names
    census = run_sql("SELECT domain, COUNT(*) FROM actions GROUP BY domain ORDER BY COUNT(*) DESC", reg)
    assert census[0]["COUNT(*)"] >= census[-1]["COUNT(*)"]     # census ordered by count desc


def test_explain_is_a_dry_run():
    from holographic_machine import HoloMachine
    from holographic_query import explain_program
    mac = HoloMachine(dim=1024, seed=0, faculties=["denoise", "recall", "render"])
    prog = mac.assemble([("APPLY", "denoise"), ("APPLY", "recall"), ("HALT", None)])
    info = explain_program(mac, prog)
    assert info["faculties_called"] == ["denoise", "recall"]   # names the work WITHOUT doing it
    assert info["n_steps"] == 2


def test_run_db_sql_update_delete_join_drop():
    """The SQL skin now covers UPDATE / DELETE (WHERE required) / JOIN / DROP -- the writes an app needs day one."""
    from holographic_query import Database, run_db_sql, QueryError
    db = Database(); db.add_namespace("user")
    run_db_sql("CREATE TABLE user.t (id, color)", db)
    for i, c in [(1, "red"), (2, "blue"), (3, "red")]:
        run_db_sql("INSERT INTO user.t (id, color) VALUES (%d, %s)" % (i, c), db)
    assert run_db_sql("UPDATE user.t SET color = 'crimson' WHERE id = 1", db)["updated"] == 1
    assert run_db_sql("DELETE FROM user.t WHERE color = 'blue'", db)["deleted"] == 1
    run_db_sql("CREATE TABLE user.a (id, x)", db); run_db_sql("CREATE TABLE user.b (id, y)", db)
    run_db_sql("INSERT INTO user.a (id, x) VALUES (1, A1)", db)
    run_db_sql("INSERT INTO user.b (id, y) VALUES (1, B1)", db)
    assert run_db_sql("SELECT x, y FROM user.a JOIN user.b ON id", db)[0] == {"x": "A1", "y": "B1"}
    assert run_db_sql("DROP TABLE user.a", db)["dropped_table"] == "user.a"
    try:
        run_db_sql("UPDATE user.t SET color = 'x'", db); assert False       # WHERE required
    except QueryError:
        pass

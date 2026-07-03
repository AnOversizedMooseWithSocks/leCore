"""Query WS1-WS6: workspaces -- durable DB coexists with transient sessions without wiping each other."""
import pytest
from holographic_query import run_sql, QueryError
from holographic_workspace import WorkspaceManager


def _mgr_with_persistent():
    mgr = WorkspaceManager()
    mgr.db.create_namespace("userdb", tier="persistent")
    mgr.db.create_table("userdb.notes", ["txt"], dim=256, seed=0)
    mgr.db.insert("userdb.notes", {"txt": "durable"})
    return mgr


def test_tiers_on_database():
    mgr = WorkspaceManager()
    mgr.new_workspace("s")
    assert mgr.db.tier_of("ws:s") == "workspace" and mgr.db.tier_of("system") == "system"


def test_new_switch_clear():
    mgr = WorkspaceManager()
    mgr.new_workspace("a"); mgr.new_workspace("b")
    assert mgr.active == "b"
    assert mgr.switch_workspace("a").name == "a"
    mgr.clear_workspace("a")
    assert "a" not in mgr.workspaces and mgr.db.tier_of("ws:a") is None


def test_clearing_one_workspace_isolates_others_and_persistent():
    mgr = _mgr_with_persistent()
    mgr.new_workspace("A"); mgr.db.create_table("ws:A.tmp", ["x"], dim=256, seed=0); mgr.db.insert("ws:A.tmp", {"x": "a"})
    mgr.new_workspace("B"); mgr.db.create_table("ws:B.tmp", ["x"], dim=256, seed=0); mgr.db.insert("ws:B.tmp", {"x": "b"})
    mgr.clear_workspace("A")
    assert mgr.db.tier_of("ws:A") is None                                  # A gone
    assert run_sql("SELECT x FROM tmp", mgr.db.resolve("ws:B.tmp"))[0]["x"] == "b"        # B intact
    assert run_sql("SELECT txt FROM notes", mgr.db.resolve("userdb.notes"))[0]["txt"] == "durable"  # persistent intact


def test_reset_to_default_keeps_persistent_drops_workspaces():
    mgr = _mgr_with_persistent()
    mgr.new_workspace("s1"); mgr.new_workspace("s2")
    mgr.reset_to_default()
    assert mgr.workspaces == {} and mgr.db.tier_of("ws:s1") is None
    assert mgr.db.tier_of("userdb") == "persistent"
    assert run_sql("SELECT txt FROM notes", mgr.db.resolve("userdb.notes"))[0]["txt"] == "durable"


def test_export_import_roundtrips():
    mgr = WorkspaceManager()
    mgr.new_workspace("work")
    mgr.db.create_table("ws:work.items", ["name"], dim=256, seed=1)
    mgr.db.insert("ws:work.items", {"name": "widget"})
    blob = mgr.export_workspace("work")
    mgr.clear_workspace("work")
    mgr.import_workspace(blob)
    assert run_sql("SELECT name FROM items", mgr.db.resolve("ws:work.items"))[0]["name"] == "widget"


def test_to_state_tier_scoped():
    mgr = _mgr_with_persistent()
    mgr.new_workspace("scratch"); mgr.db.create_table("ws:scratch.t", ["x"], dim=128, seed=0)
    assert list(mgr.db.to_state(tiers=["persistent"])["namespaces"].keys()) == ["userdb"]
    assert list(mgr.db.to_state(tiers=["workspace"])["namespaces"].keys()) == ["ws:scratch"]


def test_combine_requires_collision_policy():
    mgr = WorkspaceManager()
    mgr.new_workspace("l"); mgr.db.create_table("ws:l.t", ["v"], dim=128, seed=0)
    mgr.new_workspace("r"); mgr.db.create_table("ws:r.t", ["v"], dim=128, seed=0)
    with pytest.raises(QueryError):
        mgr.combine_workspaces("l", "r", "both")                           # default 'error' on clash
    mgr.combine_workspaces("l", "r", "both2", on_collision="suffix")
    assert {"t_a", "t_b"} <= set(mgr.db.namespaces["ws:both2"]["tables"])


def test_combine_left_policy_picks_winner():
    mgr = WorkspaceManager()
    mgr.new_workspace("l"); mgr.db.create_table("ws:l.t", ["v"], dim=128, seed=0); mgr.db.insert("ws:l.t", {"v": "L"})
    mgr.new_workspace("r"); mgr.db.create_table("ws:r.t", ["v"], dim=128, seed=0); mgr.db.insert("ws:r.t", {"v": "R"})
    mgr.combine_workspaces("l", "r", "c", on_collision="left")
    assert run_sql("SELECT v FROM t", mgr.db.resolve("ws:c.t"))[0]["v"] == "L"


def test_system_tier_protected():
    mgr = WorkspaceManager()
    with pytest.raises(QueryError):
        mgr.db.drop_namespace("system")
    with pytest.raises(QueryError):
        mgr.db.create_namespace("x", tier="system")

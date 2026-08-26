"""Query BUILD B10: graph traversal (descendants/ancestors/shortest_path/reachable) on exact adjacency."""
from holographic.agents_and_reasoning.holographic_query import UserTable, delete as _delete
from holographic.agents_and_reasoning.holographic_query_graph import descendants, ancestors, shortest_path, reachable, build_adjacency


def _org():
    edges = UserTable("reports", ["mgr", "emp"], dim=256, seed=0)
    for a, b in [("ceo", "vp1"), ("ceo", "vp2"), ("vp1", "eng1"), ("vp1", "eng2"), ("vp2", "eng3")]:
        edges.insert({"mgr": a, "emp": b})
    return edges


def test_descendants():
    e = _org()
    assert set(descendants(e, "mgr", "emp", "ceo")) == {"vp1", "vp2", "eng1", "eng2", "eng3"}
    assert set(descendants(e, "mgr", "emp", "vp1")) == {"eng1", "eng2"}
    assert descendants(e, "mgr", "emp", "eng1") == []            # a leaf


def test_ancestors():
    assert set(ancestors(_org(), "mgr", "emp", "eng3")) == {"vp2", "ceo"}


def test_shortest_path():
    e = _org()
    assert shortest_path(e, "mgr", "emp", "ceo", "eng3") == ["ceo", "vp2", "eng3"]
    assert shortest_path(e, "mgr", "emp", "ceo", "ceo") == ["ceo"]     # self path
    assert shortest_path(e, "mgr", "emp", "vp1", "vp2") is None        # no cross-branch path


def test_reachable_is_directional():
    e = _org()
    assert reachable(e, "mgr", "emp", "ceo", "eng1") is True
    assert reachable(e, "mgr", "emp", "eng1", "ceo") is False


def test_deleted_edge_respected():
    e = _org()
    _delete(e, "emp = 'vp2'")
    assert reachable(e, "mgr", "emp", "ceo", "eng3") is False    # subtree unhooked


def test_cycle_terminates():
    e = UserTable("g", ["a", "b"], dim=256, seed=0)
    for a, b in [("x", "y"), ("y", "z"), ("z", "x")]:           # a cycle
        e.insert({"a": a, "b": b})
    assert set(descendants(e, "a", "b", "x")) == {"y", "z"}   # excludes start; terminates on the cycle
    assert shortest_path(e, "a", "b", "x", "z") == ["x", "y", "z"]


def test_build_adjacency_skips_deleted():
    e = _org(); _delete(e, "emp = 'eng1'")
    adj = build_adjacency(e, "mgr", "emp")
    assert "eng1" not in adj.get("vp1", [])

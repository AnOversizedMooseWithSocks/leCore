"""Tests for holographic_querygraph (B10 exact graph traversal over table edges)."""
from holographic.agents_and_reasoning.holographic_query import Database, delete, QueryError
from holographic.agents_and_reasoning.holographic_querygraph import EdgeGraph


def _edges(pairs):
    db = Database(); db.add_namespace("user")
    db.create_table("user.edges", ["src", "dst"], dim=256, seed=0)
    t = db.namespaces["user"]["tables"]["edges"]
    for s, d in pairs:
        t.insert({"src": s, "dst": d})
    return db, t


def test_neighbors_and_descendants():
    _, t = _edges([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])
    g = EdgeGraph(t, "src", "dst")
    assert g.neighbors(1) == [2, 3]
    assert set(g.descendants(1)) == {2, 3, 4, 5}
    assert g.descendants(5) == []


def test_reachable_directed():
    _, t = _edges([(1, 2), (2, 3)])
    g = EdgeGraph(t, "src", "dst", directed=True)
    assert g.reachable(1, 3) and not g.reachable(3, 1)
    assert g.reachable(2, 2)                                    # a node reaches itself


def test_shortest_path():
    _, t = _edges([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])
    g = EdgeGraph(t, "src", "dst")
    p = g.path(1, 5)
    assert p[0] == 1 and p[-1] == 5 and len(p) == 4            # shortest is length 4
    assert g.path(1, 99) is None
    assert g.path(5, 5) == [5]


def test_undirected():
    _, t = _edges([(1, 2), (2, 3)])
    g = EdgeGraph(t, "src", "dst", directed=False)
    assert g.reachable(3, 1)


def test_tombstoned_edges_excluded():
    db, t = _edges([(1, 2), (6, 7)])
    delete(t, "src = 6")
    g = EdgeGraph(t, "src", "dst")
    assert 7 not in g.nodes() and not g.reachable(6, 7)


def test_bad_edge_columns_raise():
    _, t = _edges([(1, 2)])
    try:
        EdgeGraph(t, "src", "nope"); assert False
    except QueryError:
        pass


def test_deterministic():
    _, t = _edges([(1, 3), (1, 2), (2, 4), (3, 4)])
    g = EdgeGraph(t, "src", "dst")
    assert g.path(1, 4) == g.path(1, 4)                        # stable neighbour ordering -> stable path
    assert g.neighbors(1) == [2, 3]                            # sorted

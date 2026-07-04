"""holographic_query_graph.py -- GRAPH TRAVERSAL over an edge table (query backlog B10).

SUPERSEDED BY holographic_querygraph -- the wired version (it has the curated catalog home). This earlier implementation of the same
backlog item is kept for its tests but is intentionally NOT wired into any pipeline; use holographic_querygraph instead.

Recursive/graph queries -- descendants, ancestors, shortest path, reachability -- over a table whose rows are EDGES
(a src column and a dst column). Think org charts, category trees, dependency graphs, foreign-key chains.

DELIBERATE DESIGN CHOICE / KEPT NEGATIVE (loud): traversal is done on an EXACT adjacency dict built from the stored
rows, NOT on holographic graph memory. We measured that the holographic graph-memory recall COLLAPSES at scale (its
accuracy falls off as the graph grows), so using it for traversal would silently return wrong neighbours on a big
graph. Exact adjacency is O(E) to build and gives correct, deterministic BFS/DFS -- the right tool. The VSA store
still owns the edge DATA (records you can SELECT, join, and version); the traversal is a plain, correct graph walk
over that data. Tombstoned (deleted) edges are skipped, so traversal always reflects the live graph.
"""
from collections import deque


def build_adjacency(table, src_col, dst_col, reverse=False):
    """Build an adjacency dict {node: [neighbours]} from an edge table's live rows. reverse=True flips edge direction
    (for ancestor queries). Skips tombstoned edges so the graph is always the live one."""
    adj = {}
    for r in table.rows:
        if r.get("_deleted"):
            continue
        a, b = r.get(src_col), r.get(dst_col)
        if a is None or b is None:
            continue
        if reverse:
            a, b = b, a
        adj.setdefault(a, []).append(b)
    return adj


def descendants(table, src_col, dst_col, start, adj=None):
    """Every node reachable FROM `start` (BFS), excluding start itself, in discovery order. Pass a prebuilt `adj` to
    avoid rebuilding it across many queries."""
    adj = adj if adj is not None else build_adjacency(table, src_col, dst_col)
    seen = {start}
    out = []
    q = deque([start])
    while q:
        node = q.popleft()
        for nxt in adj.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                out.append(nxt)
                q.append(nxt)
    return out


def ancestors(table, src_col, dst_col, node):
    """Every node that can reach `node` (descendants on the reversed graph)."""
    radj = build_adjacency(table, src_col, dst_col, reverse=True)
    return descendants(table, src_col, dst_col, node, adj=radj)


def shortest_path(table, src_col, dst_col, start, goal):
    """A shortest edge path from `start` to `goal` as a list of nodes (BFS), or None if `goal` is unreachable. A
    self-path start==goal is [start]."""
    if start == goal:
        return [start]
    adj = build_adjacency(table, src_col, dst_col)
    parent = {start: None}
    q = deque([start])
    while q:
        node = q.popleft()
        for nxt in adj.get(node, []):
            if nxt not in parent:
                parent[nxt] = node
                if nxt == goal:                                  # found -> walk the parent chain back
                    path = [goal]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    return list(reversed(path))
                q.append(nxt)
    return None


def reachable(table, src_col, dst_col, start, goal):
    """True if `goal` is reachable from `start` (including start==goal)."""
    return start == goal or goal in descendants(table, src_col, dst_col, start)


def _selftest():
    """Build a small DAG as an edge table and check descendants, ancestors, shortest path, reachability, and that a
    deleted edge is respected."""
    from holographic_query import UserTable, delete as _delete

    # a tiny org chart: ceo -> vp1, vp2; vp1 -> eng1, eng2; vp2 -> eng3
    edges = UserTable("reports", ["mgr", "emp"], dim=256, seed=0)
    for a, b in [("ceo", "vp1"), ("ceo", "vp2"), ("vp1", "eng1"), ("vp1", "eng2"), ("vp2", "eng3")]:
        edges.insert({"mgr": a, "emp": b})

    assert set(descendants(edges, "mgr", "emp", "ceo")) == {"vp1", "vp2", "eng1", "eng2", "eng3"}
    assert set(descendants(edges, "mgr", "emp", "vp1")) == {"eng1", "eng2"}
    assert set(ancestors(edges, "mgr", "emp", "eng3")) == {"vp2", "ceo"}
    assert shortest_path(edges, "mgr", "emp", "ceo", "eng3") == ["ceo", "vp2", "eng3"]
    assert shortest_path(edges, "mgr", "emp", "vp1", "vp2") is None       # no path across branches
    assert reachable(edges, "mgr", "emp", "ceo", "eng1") is True
    assert reachable(edges, "mgr", "emp", "eng1", "ceo") is False

    # delete an edge -> traversal reflects the live graph
    _delete(edges, "emp = 'vp2'")                                # ceo no longer manages vp2's subtree
    assert reachable(edges, "mgr", "emp", "ceo", "eng3") is False

    print("holographic_query_graph selftest OK: descendants(ceo)=all 5 reports, descendants(vp1)={eng1,eng2}, "
          "ancestors(eng3)={vp2,ceo}, shortest ceo->eng3 = [ceo,vp2,eng3], cross-branch path None, reachability "
          "directional; deleting the ceo->vp2 edge drops eng3 from ceo's reach; EXACT adjacency (not holographic "
          "recall -- that collapses at scale)")


if __name__ == "__main__":
    _selftest()

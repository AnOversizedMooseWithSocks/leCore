"""holographic_querygraph.py -- B10 graph traversal over a query table's edges (descendants / reachable / path).

WHY (and the loud kept negative that shaped this)
-------------------------------------------------
Recursive CTEs are clunky and widely avoided, so graph reachability is a real query pain. The tempting VSA move is to
store the graph HOLOGRAPHICALLY (edges bundled into vectors, traversal by unbind) -- but the holographic graph memory
has a MEASURED recall-accuracy collapse at scale (routed descent overloads exactly like an over-full maze field). So,
per the backlog's own guidance, traversal here is backed by a PLAIN EXACT ADJACENCY INDEX: a dict of node -> neighbours
built from the table's edge rows. It is deterministic, O(V+E), scales cleanly, and returns EXACT answers -- no recall
cliff. The holographic graph memory stays what it is good at (classification by routed descent); it is deliberately NOT
used for traversal. That is the honest split: exact index for reachability, vectors for similarity.

  neighbors(node)        -- the direct out-neighbours (or all neighbours if undirected).
  descendants(node)      -- every node reachable from `node` (BFS).
  reachable(a, b)        -- is there a path a -> b?
  path(a, b)             -- a SHORTEST path a -> b (BFS), or None.

KEPT NEGATIVE (loud): this is the EXACT adjacency index, on purpose -- the holographic graph store is not used for
traversal because its recall collapses at scale. Tombstoned (`_deleted`) edge rows are skipped, so the graph reflects
the live table.
"""
from collections import deque


class EdgeGraph:
    """An exact adjacency index built from a table's edge rows: each live row contributes an edge
    source_col -> target_col. Deterministic traversal; scales as a plain graph, not as an overloaded hypervector."""

    def __init__(self, table, source_col, target_col, directed=True):
        if source_col not in table.roles or target_col not in table.roles:
            from holographic.agents_and_reasoning.holographic_query import QueryError
            raise QueryError("edge columns %r/%r must be columns of the table (%s)"
                             % (source_col, target_col, ", ".join(table.roles)))
        self.directed = directed
        self.adj = {}                                          # node -> set(neighbours)
        for r in table.rows:
            if r.get("_deleted"):                              # respect B2 tombstones -- live edges only
                continue
            s, t = r.get(source_col), r.get(target_col)
            if s is None or t is None:
                continue
            self.adj.setdefault(s, set()).add(t)
            self.adj.setdefault(t, set())                      # ensure the target is a known node too
            if not directed:
                self.adj[t].add(s)

    def nodes(self):
        """Every node in the graph, sorted for determinism."""
        return sorted(self.adj, key=_sortkey)

    def neighbors(self, node):
        """The direct neighbours of `node` (out-neighbours if directed), sorted."""
        return sorted(self.adj.get(node, set()), key=_sortkey)

    def descendants(self, node):
        """Every node reachable from `node` (excluding itself unless it is in a cycle back to itself). BFS."""
        seen, q = set(), deque(self.adj.get(node, set()))
        while q:
            cur = q.popleft()
            if cur in seen:
                continue
            seen.add(cur)
            q.extend(self.adj.get(cur, set()) - seen)
        return sorted(seen, key=_sortkey)

    def reachable(self, a, b):
        """Is there a directed path a -> b? (a reaches itself trivially.)"""
        if a == b:
            return True
        return b in set(self.descendants(a))

    def path(self, a, b):
        """A SHORTEST path a -> b as a list of nodes (inclusive), or None if b is unreachable. BFS, so the first
        time we reach b it is via a shortest path. Deterministic: neighbours are visited in sorted order."""
        if a == b:
            return [a]
        prev, q = {a: None}, deque([a])
        while q:
            cur = q.popleft()
            for nxt in sorted(self.adj.get(cur, set()), key=_sortkey):
                if nxt not in prev:
                    prev[nxt] = cur
                    if nxt == b:                               # reconstruct the path back to a
                        out = [b]
                        while out[-1] != a:
                            out.append(prev[out[-1]])
                        return list(reversed(out))
                    q.append(nxt)
        return None


def _sortkey(x):
    """Sort mixed node types deterministically (numbers before strings, each in natural order)."""
    return (0, x) if isinstance(x, (int, float)) else (1, str(x))


def _selftest():
    from holographic.agents_and_reasoning.holographic_query import Database, delete
    db = Database(); db.add_namespace("user")
    db.create_table("user.edges", ["src", "dst"], dim=256, seed=0)
    t = db.namespaces["user"]["tables"]["edges"]
    # a small DAG: 1->2, 1->3, 2->4, 3->4, 4->5, plus a dead-end 6->7 we will tombstone
    for s, d in [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (6, 7)]:
        t.insert({"src": s, "dst": d})

    g = EdgeGraph(t, "src", "dst", directed=True)
    assert g.neighbors(1) == [2, 3]
    assert set(g.descendants(1)) == {2, 3, 4, 5}
    assert g.reachable(1, 5) and not g.reachable(5, 1)
    assert g.path(1, 5) == [1, 2, 4, 5] or g.path(1, 5) == [1, 3, 4, 5]     # both are length-4 shortest paths
    assert len(g.path(1, 5)) == 4
    assert g.path(1, 99) is None

    # tombstoned edges drop out of the live graph
    delete(t, "src = 6")
    g2 = EdgeGraph(t, "src", "dst", directed=True)
    assert 7 not in g2.nodes() and not g2.reachable(6, 7)

    # undirected: now 5 can reach 1
    gu = EdgeGraph(t, "src", "dst", directed=False)
    assert gu.reachable(5, 1)

    print("OK: holographic_querygraph self-test passed (exact adjacency: neighbors/descendants/reachable/shortest "
          "path, tombstone-aware, directed+undirected -- B10, exact index by design because the holographic graph "
          "store's recall collapses at scale)")


if __name__ == "__main__":
    _selftest()

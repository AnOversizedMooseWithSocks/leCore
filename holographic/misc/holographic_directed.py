"""Directed structure -- a permutation direction role for sequences and graphs in superposition.

WHY THIS EXISTS
---------------
A memory of edges bundled as bind(x_i, x_{i+1}) is UNDIRECTED: unbinding by a node returns BOTH its
neighbours -- predecessor and successor -- at equal strength (measured ~0.33 vs ~0.33), so a traversal cannot
tell which way is forward. This is the "predecessor leak". Binding the successor through a fixed PERMUTATION
first -- bind(x_i, perm(x_{i+1})) -- breaks the symmetry: unbinding by x_i and undoing the permutation
recovers the successor (~0.34), while the predecessor term lands in the permuted subspace as noise (~0.00).
Forward traversal becomes unambiguous at the ENCODING level, with no explain-away needed at decode time.

This is the substrate-correct counterpart to the engine's undirected chain_structure (B7), which carries the
same leak and needs holographic_peel's per-peel history-aware cleanup to suppress it: the permutation does at
ENCODE time what the peel cleanup does at DECODE time. It also generalises past linear chains -- any set of
directed EDGES bundles the same way, a node with several successors returns their superposition (a graph), and
the whole thing composes with the throughput-gated traversal (gated_traverse) for a Russian-roulette walk.

MEASURED (see `_selftest`)
  * directed unbind recovers the successor (~0.34) with the predecessor suppressed to noise (~0.00), where the
    undirected baseline returns predecessor and successor at equal strength (~0.33 each).
  * a branching node (0 -> {1,2,3}) recovers all three successors cleanly above the non-successors.
  * composes with gated_traverse: a stored chain is walked forward and the gate stops when it runs out.
"""

from collections import namedtuple

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, involution

DirectedStructure = namedtuple("DirectedStructure", "memory nodes perm perm_inv")


def chain_edges(n):
    """The edges of a linear chain 0 -> 1 -> ... -> (n-1)."""
    return [(i, i + 1) for i in range(n - 1)]


def perm_pair(dim, seed):
    """A fixed random permutation and its inverse -- the 'next' direction role. Any fixed permutation works (a
    cyclic roll is the cheapest); a full random permutation is the most thorough symmetry-break, so it is the
    default here and matches the directed chain the throughput-gated traversal already used."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(dim)
    return perm, np.argsort(perm)


def encode_directed(nodes, edges, perm):
    """Bundle directed edges into one memory: M = sum_(i,j) bind(nodes[i], nodes[j][perm]). `nodes` is a node
    codebook (rows); `edges` a list of (src, dst) index pairs. A sequence is the chain of consecutive pairs; a
    graph is any edge set. Returns the memory vector."""
    nodes = np.asarray(nodes, float)
    M = np.zeros(nodes.shape[1])
    for i, j in edges:
        M = M + bind(nodes[i], nodes[j][perm])               # the successor is permuted -> the direction role
    return M


def build(nodes, edges=None, seed=0):
    """Assemble a DirectedStructure from node vectors and directed edges (default: the linear chain)."""
    nodes = np.asarray(nodes, float)
    if edges is None:
        edges = chain_edges(len(nodes))
    perm, inv = perm_pair(nodes.shape[1], seed)
    return DirectedStructure(encode_directed(nodes, edges, perm), nodes, perm, inv)


def _unit_rows(nodes):
    return nodes / np.maximum(np.linalg.norm(nodes, axis=1, keepdims=True), 1e-12)


def successors(ds, node_index, topk=1, thresh=None):
    """Recover the successor(s) of a node: perm_inv(unbind(M, node)) cleaned up against the codebook. Returns
    [(index, cosine), ...] -- the strongest `topk`, or every node at/above `thresh` (a branching node's whole
    successor set)."""
    nodes_u = _unit_rows(ds.nodes)
    probe = bind(ds.memory, involution(ds.nodes[node_index]))[ds.perm_inv]   # unbind, then undo the direction role
    cs = nodes_u @ (probe / (np.linalg.norm(probe) + 1e-12))
    order = np.argsort(cs)[::-1]
    if thresh is not None:
        hits = [(int(k), float(cs[k])) for k in order if cs[k] >= thresh]
        return hits or [(int(order[0]), float(cs[order[0]]))]
    return [(int(order[k]), float(cs[order[k]])) for k in range(min(topk, len(order)))]


def make_step(ds):
    """A gated_traverse-ready step over a directed chain: from the current node vector, recover the successor
    (perm_inv . unbind . cleanup) and report the cleanup cosine as throughput. Returns step(cur_vec) ->
    (next_node_vec, throughput, successor_index)."""
    nodes_u = _unit_rows(ds.nodes)

    def step(cur):
        probe = bind(ds.memory, involution(cur))[ds.perm_inv]
        cs = nodes_u @ (probe / (np.linalg.norm(probe) + 1e-12))
        j = int(np.argmax(cs))
        return (ds.nodes[j], float(cs[j]), j)

    return step


def _selftest():
    """CI-fast: prove the direction role (1) suppresses the predecessor leak that sinks the undirected
    baseline, (2) generalises to a branching GRAPH node, and (3) composes with the throughput-gated walk."""
    rng = np.random.default_rng(0)
    D, n = 8192, 8
    def unit():
        v = rng.standard_normal(D)
        return v / np.linalg.norm(v)
    nodes = np.array([unit() for _ in range(n)])

    # (1) the predecessor leak: the directed encoding suppresses it; the undirected baseline does not
    ds = build(nodes, seed=1)
    Mund = np.zeros(D)
    for i in range(n - 1):
        Mund = Mund + bind(nodes[i], nodes[i + 1])           # bind(x_i, x_{i+1}) -- no direction role
    def cos(a, b):
        return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))
    for i in (2, 4, 6):
        succ = successors(ds, i)[0]
        assert succ[0] == i + 1 and succ[1] > 0.2, succ        # directed: successor recovered
        pd = bind(ds.memory, involution(nodes[i]))[ds.perm_inv]
        assert cos(pd, nodes[i - 1]) < 0.1                     # directed: predecessor suppressed to noise
        pu = bind(Mund, involution(nodes[i]))                  # undirected: both neighbours come back equal
        assert cos(pu, nodes[i + 1]) > 0.2 and cos(pu, nodes[i - 1]) > 0.2

    # (2) a branching GRAPH node: 0 -> {1, 2, 3}, all three recovered from one unbind
    g = build(nodes, edges=[(0, 1), (0, 2), (0, 3)], seed=1)
    assert {k for k, _ in successors(g, 0, topk=3)} == {1, 2, 3}

    # (3) composes with the throughput-gated walk: forward over the chain, abstaining when it runs out
    from holographic.misc.holographic_traverse import gated_traverse
    res = gated_traverse(make_step(ds), nodes[0], floor=0.2, max_steps=20)
    assert res.payloads == list(range(1, n)) and res.stopped == "floor"


if __name__ == "__main__":
    _selftest()
    print("holographic_directed selftest passed")

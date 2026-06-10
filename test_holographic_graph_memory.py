"""Hierarchical routing memory: a sanity check on the mechanism, and an honest record
of where it does and does not help.

MEASURED (see the conversation that produced this): routed descent MATCHES a flat exact
prototype scan only at small class counts; as classes grow it trades accuracy for cost
(one wrong turn high in the tree is unrecoverable -- high-dimensional nearest-neighbour
does not tree-route well). So this is NOT adopted for classification, where the flat scan
is already optimal. Its home is sparse, navigable structure (sequence/transition graphs),
the same place the slime maze solver wins. These tests pin down the working behaviour, not
a win over the flat memory."""
import numpy as np
from holographic_graph_memory import GraphMemory


def test_routes_correctly_on_well_separated_classes():
    rng = np.random.default_rng(0)
    dim = 256
    cents = [c / np.linalg.norm(c) for c in (rng.standard_normal((12, dim)))]
    gm = GraphMemory(dim=dim, branching=4, beam=2, seed=0)
    for g, c in enumerate(cents):
        for _ in range(20):
            v = c + 0.12 * rng.standard_normal(dim)
            gm.observe_vector(v / np.linalg.norm(v), g)
    gm.organize()
    ok = 0
    for g, c in enumerate(cents):
        v = c + 0.12 * rng.standard_normal(dim)
        ok += gm.classify_vector(v / np.linalg.norm(v))[0] == g
    assert ok >= 10            # well-separated classes route correctly


def test_descent_is_sublinear_in_comparisons():
    rng = np.random.default_rng(1)
    dim = 256
    gm = GraphMemory(dim=dim, branching=5, beam=1, seed=0)
    for g in range(60):
        c = rng.standard_normal(dim)
        for _ in range(8):
            gm.observe_vector(c / np.linalg.norm(c), g)
    gm.organize()
    gm.classify_vector(rng.standard_normal(dim))
    assert gm.last_comparisons < gm.size()    # routed touches fewer than all prototypes

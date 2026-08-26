"""Tests for B8: per-peel cleanup pushes the decode depth cliff; soft cleanup earns its keep on
continuous payloads; the chain is a B7 typed structure."""

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, unbind, derived_atom, cosine
from holographic.io_and_interop.holographic_encoders import ScalarEncoder
from holographic.rendering.holographic_peel import chain_recipe, traverse, traversal_score, recover_continuous_values


def test_per_peel_cleanup_decodes_full_chain_raw_craters():
    r, nodes = chain_recipe(512, 1, 16)
    M = r.get(r._outputs[0])
    none_c, total = traversal_score(M, nodes, cleanup=None)
    hard_c, _ = traversal_score(M, nodes, cleanup="hard")
    assert hard_c == total                       # per-peel cleanup decodes every hop
    assert none_c <= 3                            # raw decode craters (noise compounds / diverges)


def test_hard_and_soft_tie_on_discrete_pointers():
    r, nodes = chain_recipe(512, 1, 16)
    M = r.get(r._outputs[0])
    hard_c, total = traversal_score(M, nodes, cleanup="hard")
    soft_c, _ = traversal_score(M, nodes, cleanup="soft")
    assert hard_c == total and soft_c == total   # Bayes-optimal tie on identity (B1's kept negative)


def test_traverse_returns_the_correct_sequence():
    r, nodes = chain_recipe(512, 2, 12)
    M = r.get(r._outputs[0])
    rec = traverse(M, nodes, 11, cleanup="hard")
    assert rec == list(range(1, 12))             # from node 0 the path is 1,2,...,11


def test_chain_is_a_typed_structure():
    from holographic.misc.holographic_typed import op_kinds
    r, nodes = chain_recipe(256, 3, 8)
    assert op_kinds(r) <= {"atom", "bind", "bundle", "superpose", "permute", "raw", "normalize"}
    via = r.get(r._outputs[0])
    direct = np.sum([bind(nodes[i], nodes[i + 1]) for i in range(7)], axis=0)
    assert np.max(np.abs(via - direct)) < 1e-9   # the recipe realizes the chain memory bit-exactly


def test_soft_cleanup_beats_hard_on_continuous_values():
    enc = ScalarEncoder(1024, 0.0, 1.0, seed=1, kernel="rbf", bandwidth=8)
    grid = np.linspace(0, 1, 21)
    codebook = np.stack([enc.encode(g) for g in grid])
    roles = np.stack([derived_atom(1, f"role:{i}", 1024, unitary=True) for i in range(6)])
    rng = np.random.default_rng(0)
    trues = rng.uniform(0.05, 0.95, 6)
    M = np.sum([bind(roles[i], enc.encode(trues[i])) for i in range(6)], axis=0)
    tv = [enc.encode(t) for t in trues]
    hard, soft = recover_continuous_values(roles, codebook, M, tv)
    assert soft >= hard                          # soft blend lands between grid points where truth lives


def test_diverged_hop_is_marked_failed():
    # a raw (no-cleanup) traversal eventually carries a non-finite/huge vector; traverse marks -1, no crash
    r, nodes = chain_recipe(256, 4, 20)
    M = r.get(r._outputs[0])
    rec = traverse(M, nodes, 19, cleanup=None)
    assert any(idx != h + 1 for h, idx in enumerate(rec))   # it does not decode the whole chain cleanly

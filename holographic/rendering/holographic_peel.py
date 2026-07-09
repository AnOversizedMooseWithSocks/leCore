"""B8 -- denoised structure decoding: per-peel cleanup pushes the decode depth cliff.

A composed holographic structure is decoded by ITERATED unbinding. Take a linked list
    M = superpose_i bind(node_i, node_{i+1})
-- itself a typed structure in the B7 sense (atom/bind/superpose; see chain_recipe). To traverse it
you unbind the current node to recover the next, then repeat. Crosstalk makes each recovered pointer
noisy, and -- the crux -- without cleanup that noise is carried into the NEXT query and COMPOUNDS:
a raw traversal craters after a hop or two and the carried vector's norm diverges. Snapping each
recovered pointer back onto the node codebook BEFORE the next hop breaks the compounding -- the
per-hop noise stays bounded and the whole chain decodes. Cleaning structure AS it is decoded.

WHAT WAS MEASURED (kept honestly):
  * Per-peel cleanup is the whole game. On a 16-node chain at dim 512, a raw traversal (no cleanup)
    gets ~1-2 hops before it fails and diverges; per-peel cleanup decodes all 15 hops. ~2 -> full.
  * Hard argmax cleanup and the B1 dense-Hopfield cleanup TIE on the discrete pointer (both decode the
    full chain). That is exactly B1's kept negative: snapping to the nearest atom is already
    Bayes-optimal for "which atom is this", so the soft update cannot beat it there.
  * The Hopfield (soft) cleanup earns its keep only on CONTINUOUS payloads: recovering off-grid
    scalar-encoded values from a superposition, the soft blend beats hard snap-to-grid (~0.996 vs
    ~0.990 cosine to truth) -- it returns a continuous mixture of codebook atoms, which lands between
    grid points where the true value lives. See recover_continuous_values.
  * A commutative-bind chain has an INTRINSIC predecessor leak: unbinding node_i surfaces node_{i-1}
    as a clean atom too (node_{i-1} bound it as its value, and node_i*involution(node_i)=delta). A
    forward traversal KNOWS its predecessor, so we explain it away -- standard history-aware decode.
    This is a property of the encoding, reported, not hidden; it is why traverse takes an explain-away
    step. SBC block codes (B2) bind losslessly and have no such leak -- and no cliff to push.

Pure NumPy + holostuff spirit; deterministic; reuses StructureRecipe (B7) and dense_cleanup (B1).
"""

import numpy as np
from holographic.misc.holographic_determinism import argmax_tiebreak

from holographic.agents_and_reasoning.holographic_ai import bind, unbind, derived_atom, cosine
from holographic.misc.holographic_recipe import StructureRecipe
from holographic.agents_and_reasoning.holographic_hopfield import dense_cleanup


def chain_recipe(dim, seed, n):
    """The linked list AS a typed structure (B7): M = superpose_i bind(node_i, node_{i+1}).
    Returns (recipe, node_codebook). Realizing the recipe yields the chain-memory vector; the node
    codebook is what traverse() cleans against. Nodes are unitary so unbinding is exact."""
    r = StructureRecipe(dim, seed)
    handles = [r.atom(f"node:{i}", unitary=True) for i in range(n)]
    nodes = np.stack([derived_atom(seed, f"node:{i}", dim, unitary=True) for i in range(n)])
    r.mark_output(r.superpose([r.bind(handles[i], handles[i + 1]) for i in range(n - 1)]))
    return r, nodes


def traverse(M, nodes, steps, cleanup="hard", beta=8.0):
    """Decode the chain by iterated unbinding from node 0. `cleanup` in {None, "hard", "soft"}:
      * None  -- carry the raw noisy peel forward; noise compounds and the decode craters/diverges.
      * "hard"-- snap each recovered pointer to the nearest node atom (Bayes-optimal for identity).
      * "soft"-- the B1 dense-Hopfield update (ties hard on discrete pointers).
    History-aware explain-away removes the known predecessor's clean leak each hop. Returns the list
    of recovered node indices (-1 marks a diverged/failed hop)."""
    prev = None
    cur = nodes[0]
    rec = []
    for _ in range(steps):
        if not np.all(np.isfinite(cur)) or np.linalg.norm(cur) > 1e6:
            rec.append(-1)                                  # raw decode diverged -> failed
            break
        succ = unbind(M, cur)
        if prev is not None:                                # remove the known predecessor (intrinsic leak)
            succ = succ - float(succ @ prev) * prev
        nearest = argmax_tiebreak(nodes @ succ)          # DETERMINISM CONTRACT (ISA-1)
        rec.append(nearest)
        prev = cur
        if cleanup is None:
            cur = succ
        elif cleanup == "hard":
            cur = nodes[nearest]
        elif cleanup == "soft":
            cur = dense_cleanup(succ, nodes, beta, steps=3)
        else:
            raise ValueError(cleanup)
    return rec


def traversal_score(M, nodes, cleanup="hard", beta=8.0):
    """How many hops a traversal gets right. From node 0 the correct path is 1, 2, 3, ... Returns
    (n_correct, total_hops)."""
    n = len(nodes)
    rec = traverse(M, nodes, n - 1, cleanup=cleanup, beta=beta)
    correct = sum(1 for h, idx in enumerate(rec) if idx == h + 1)
    return correct, n - 1


def recover_continuous_values(roles, codebook, M, true_vecs, beta=12.0):
    """For a superposition M = sum_i bind(role_i, value_i) of CONTINUOUS values, recover each by
    unbinding its role, then compare hard snap-to-codebook vs the soft Hopfield blend (mean cosine to
    truth). Returns (hard_mean, soft_mean). This is the regime where the soft update earns its keep:
    an off-grid value is best matched by a mixture of nearby grid atoms, which the blend returns and a
    hard snap cannot."""
    hard_c, soft_c = [], []
    for i in range(len(roles)):
        noisy = unbind(M, roles[i])
        hard = codebook[argmax_tiebreak(codebook @ noisy)]   # DETERMINISM CONTRACT (ISA-1)
        soft = dense_cleanup(noisy, codebook, beta, steps=3)
        hard_c.append(cosine(hard, true_vecs[i]))
        soft_c.append(cosine(soft, true_vecs[i]))
    return float(np.mean(hard_c)), float(np.mean(soft_c))

"""LEVER 6: a measured limit is a composability boundary, not a wall.

This test exists because an earlier NOTES entry declared the bundled-fact capacity cliff
STRUCTURAL ("indexed rows, never a single bundled trace") after testing lever 4 (widen D)
and finding it refuted. That conclusion was FALSE. Lever 6 says the cliff number is the TILE
SIZE: group at K, add a coordinator level, and clean up BETWEEN levels -- which is exactly
where hierarchical_pack's docstring says the win lives ("the hierarchy is NOT in the packing
... but in the recall").

It also pins the SECOND half of the doctrine: the coordinator has its own measured limit,
and hitting it is the signal to split again, not evidence the lever failed.
"""
import numpy as np
from holographic.agents_and_reasoning.holographic_ai import derived_atom, bind, unbind, nearest

D = 1024
VOCAB = ["v%d" % i for i in range(64)]


def _sym(n):
    return derived_atom(0, "l6:" + n, D)


def _pack(pairs):
    return np.sum([bind(_sym(k), _sym(v)) for k, v in pairs], axis=0)


def _leaf(S, k):
    q = unbind(S, _sym(k))
    M = np.stack([_sym(v) for v in VOCAB])
    j, _ = nearest(q, M)
    return VOCAB[j]


def _flat_recall(n):
    pairs = [("k%d" % i, VOCAB[i % 64]) for i in range(n)]
    S = _pack(pairs)
    return sum(_leaf(S, k) == v for k, v in pairs) / n


def _tiled_recall(n, group=8):
    pairs = [("k%d" % i, VOCAB[i % 64]) for i in range(n)]
    groups = [pairs[i:i + group] for i in range(0, n, group)]
    chunks = [_pack(g) for g in groups]
    gk = ["g%d" % i for i in range(len(groups))]
    S2 = np.sum([bind(_sym(g), c) for g, c in zip(gk, chunks)], axis=0)
    Mc = np.stack(chunks)
    hits = 0
    for gi, g in enumerate(groups):
        j, _ = nearest(unbind(S2, _sym(gk[gi])), Mc)   # cleanup BETWEEN levels
        for k, v in g:
            hits += int(_leaf(chunks[j], k) == v)
    return hits / n


def test_one_coordinator_level_beats_the_flat_capacity_law():
    """The correction. At 128 and 256 items the flat bundle degrades badly while a single
    coordinator level recalls perfectly -- so 'a bundle cannot hold more than ~8 facts' was
    a false negative, not a structural limit."""
    for n in (128, 256):
        flat, tiled = _flat_recall(n), _tiled_recall(n)
        assert tiled > 0.99, (n, tiled)
        assert tiled > flat + 0.25, (n, flat, tiled)


def test_the_coordinator_has_its_own_tile_size():
    """The doctrine's second half, pinned: lever 6 recurses. When the COORDINATOR level
    itself holds too many groups its recall falls too -- which is the signal to split and
    coordinate again, not evidence the lever failed. Still far above flat throughout."""
    flat_1024, tiled_1024 = _flat_recall(1024), _tiled_recall(1024)
    assert tiled_1024 < 0.99          # the coordinator is over ITS limit
    assert tiled_1024 > 3 * flat_1024  # and still multiples better than flat

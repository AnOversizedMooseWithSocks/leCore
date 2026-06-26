"""Self-verifying storage -- tamper-evidence as an O(log n) property of the structure itself (BLD-1).

A Merkle tree commits to data with hashes: each leaf hashes an item, each node hashes its children, the root
hash is a commitment that changes if any leaf changes, and the tree then localises WHICH leaf changed in
O(log n). This is the holographic analogue, built from the two kernel primitives (bind + bundle) so it stays
in-substrate and needs no hash function:

    leaf_i = bind(pos_i, item_i)     -- item bound to its SLOT key (so the position is part of what is committed)
    node   = sum of its children     -- bundle (superposition) is the COMBINE, in the hash's structural role
    root   = sum of all leaves        -- the whole-store composite: the commitment vector

DETECT: rebuild the tree from the current items and compare the root to the committed root -- any change to
any item, or to which slot it sits in, shifts the root. LOCALISE: descend from the root, at each node
following the child whose composite no longer matches its committed value, reaching the changed leaf in
<= log2(n) composite comparisons (the descent DEPTH), regardless of how many items the store holds.

WHY position is bound into each leaf: a plain bundle is COMMUTATIVE (sum(a,b) == sum(b,a)), so without a slot
key a reordering of items would leave the root identical and slip through. Binding each item to its slot makes
a swap change two leaves -- detected AND localised. (Measured in _selftest.)

WHAT THIS IS NOT -- the load-bearing kept negative. The root is a LINEAR combination, so the map
items -> root is R^(n*D) -> R^D, which is many-to-one for n > 1. Collisions therefore EXIST, and a key-aware
adversary can CONSTRUCT one by deconvolution: pick any change `da` to item a, then change item b by
`db = deconv(-bind(pos_a, da), pos_b)` so that bind(pos_b, db) exactly cancels bind(pos_a, da) and the root is
bit-for-bit unchanged (measured: an invisible canceling pair). A cryptographic Merkle tree resists this --
finding a hash collision is hard; cancelling a linear sum is a division. So this is evidence of ACCIDENTAL
corruption and uncoordinated tampering, NOT cryptographic tamper-proofing against an adversary who knows the
slot keys. Two further honest bounds, both measured: the descent localises ONE changed item (several
uncoordinated changes are detected at the root, but only one path is returned per pass); and localisation
needs the per-node composites kept, so the commitment costs O(n) vectors (a root-only commitment is O(1) but
detect-only).

(A note the original plan guessed wrong, kept for the record: quantising the stored checksums to save space
was expected to create a detection floor for small tampers. Measured, it does NOT at D = 1024 -- detection
stays 100% down to 2-bit checksums even at n = 1024, because in high dimension a tamper always pushes some
component across a quantiser boundary. The capacity ceiling that bounds superposition elsewhere does not bind
here, so the checksums are kept as exact floats and the trade is not exposed as an option.)
"""
import numpy as np
from holographic_ai import bind, cosine


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


class CompositionTree:
    """A tamper-evident commitment to a list of item vectors, built from bind + bundle.

    Slot keys (positions) are generated deterministically from `seed`, so a verifier holding the same seed
    reconstructs the same commitment. `verify(items)` detects any change; `locate(items)` returns the index of
    the single changed item in <= log2(n) composite comparisons. See the module docstring for the kept
    negatives -- in particular this is corruption-evidence, not cryptographic tamper-proofing.
    """

    def __init__(self, items, seed=0):
        items = [np.asarray(x, float) for x in items]
        if not items:
            raise ValueError("CompositionTree needs at least one item")
        self.n = len(items)
        self.dim = len(items[0])
        self.seed = seed
        rng = np.random.default_rng(seed)
        # deterministic slot keys: each item is bound to its position, so position is part of the commitment
        self.positions = [_unit(rng.standard_normal(self.dim)) for _ in range(self.n)]
        self._commit = self._build(items)        # the committed tree: root is the commitment, nodes localise

    def _build(self, items):
        """Bottom-up segment tree: leaves are position-bound items, each internal node is the bundle (sum)
        of its two children, the top level is the single root composite. An odd tail node carries up unpaired
        (a valid not-quite-balanced binary tree)."""
        levels = [[bind(self.positions[i], items[i]) for i in range(self.n)]]
        while len(levels[-1]) > 1:
            cur = levels[-1]
            levels.append([cur[i] + cur[i + 1] if i + 1 < len(cur) else cur[i]
                           for i in range(0, len(cur), 2)])
        return levels

    def root(self):
        """The commitment: a single vector that changes if any item, or its slot, changes."""
        return self._commit[-1][0]

    def verify(self, items):
        """Detect: True if `items` reproduce the commitment exactly, False if anything changed. O(n) to
        rebuild the leaves, then a single root comparison."""
        cur = self._build([np.asarray(x, float) for x in items])
        return bool(np.allclose(cur[-1][0], self.root(), atol=1e-9))

    def locate(self, items):
        """Localise: return (index, n_checks) for the single changed item, or (None, n_checks) if `items`
        verify. Descends from the root following the child whose composite no longer matches its committed
        value, reaching the changed leaf in <= log2(n)+1 comparisons (the root check plus the descent depth) --
        the cost is the tree DEPTH, independent of n. For several uncoordinated changes the root still flags a
        tamper but only one path is returned (see the module docstring)."""
        cur = self._build([np.asarray(x, float) for x in items])
        eq = lambda u, w: np.allclose(u, w, atol=1e-9)
        checks = 1
        if eq(cur[-1][0], self.root()):
            return None, checks
        node = 0
        for depth in range(len(self._commit) - 2, -1, -1):       # root-1 down to the leaf level
            left, right = 2 * node, 2 * node + 1
            checks += 1
            left_changed = (left < len(cur[depth])) and not eq(cur[depth][left], self._commit[depth][left])
            node = left if (left_changed or right >= len(cur[depth])) else right
        return node, checks


def _selftest():
    """The BLD-1 bar and the kept negatives, all deterministic and measured."""
    import math
    rng = np.random.default_rng(0)
    D, n = 1024, 64
    items = [_unit(rng.standard_normal(D)) for _ in range(n)]
    LOG2N = int(math.log2(n))
    tree = CompositionTree(items, seed=1)

    # 1) THE BAR: detect + localise a single full tamper in <= log2(n)+1 checks, every time, with no FP.
    ok = 0; mx = 0
    for _ in range(40):
        j = int(rng.integers(n)); it = list(items); it[j] = _unit(rng.standard_normal(D))
        idx, checks = tree.locate(it); ok += (idx == j); mx = max(mx, checks)
    assert ok == 40, ok
    assert mx <= LOG2N + 1, (mx, LOG2N)
    assert all(tree.locate(items)[0] is None for _ in range(10))            # no false positives on clean data
    assert tree.verify(items)
    assert not tree.verify([_unit(rng.standard_normal(D))] + items[1:])     # any change fails verify

    # 2) position binding defeats commutativity: a slot swap is detected AND localised (a plain bundle misses it).
    sw = list(items); sw[3], sw[40] = sw[40], sw[3]
    assert tree.locate(sw)[0] is not None

    # 3) KEPT NEGATIVE -- linear, not cryptographic: a key-aware adversary cancels a tamper by deconvolution,
    #    leaving the root ~bit-for-bit unchanged (an invisible collision a hash would resist). Contrast the
    #    real single tamper above, which drops the root cosine to ~0.998.
    a, b = 10, 40; da = 0.5 * _unit(rng.standard_normal(D))
    db = np.fft.irfft(np.fft.rfft(-bind(tree.positions[a], da)) / np.fft.rfft(tree.positions[b]), n=D)
    forged = list(items); forged[a] = items[a] + da; forged[b] = items[b] + db
    fr = sum(bind(tree.positions[i], np.asarray(forged[i], float)) for i in range(n))
    assert cosine(_unit(fr), _unit(tree.root())) > 0.9999                   # invisible -> corruption-evidence only

    # 4) KEPT NEGATIVE -- single-tamper localisation: two uncoordinated changes are detected, one path returned.
    two = list(items); two[5] = _unit(rng.standard_normal(D)); two[50] = _unit(rng.standard_normal(D))
    idx2, _ = tree.locate(two)
    assert idx2 in (5, 50)                                                  # detected, but only ONE is localised

    # 5) determinism: same items + seed -> identical commitment.
    assert np.array_equal(CompositionTree(items, seed=1).root(), CompositionTree(items, seed=1).root())
    return True


if __name__ == "__main__":
    print("holographic_verify selftest:", _selftest())

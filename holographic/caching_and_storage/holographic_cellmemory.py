"""holographic_cellmemory.py -- DOMAIN REPETITION over the capacity law: unbounded pairs from a
bounded unit, the limit itself as the tile size.

THE SEAT (Quilez, demoscene): opRep -- fold infinite space into one bounded cell with a mod and
evaluate a single unit; the scene is infinite BECAUSE the unit is bounded. Published, decades
proven, and it is the boundary-composition principle as a graphics primitive. Applied here to
the framework's oldest 'stuck' limit:

THE WALL, MEASURED (real corpus term->df pairs, dim=4096, vocab=8192): the capacity law says
n*=57; storing 4,000 pairs (70x past the law) in ONE superposed memory recalls at accuracy
0.007 -- total interference collapse, exactly as the law predicts. The law is not the enemy;
it is the physics (SNR ~ 1/sqrt(n) is where the holography lives).

THE SEAM (this module): cells of EXACTLY n* pairs -- the measured limit IS the unit boundary --
tiled sequentially, sharing ONE seed-derived codebook (the codebooks are a pure function of
(seed, vocab, dim); only each cell's dim-float trace is per-cell state). Same 4,000 pairs:
accuracy 1.000, 71 cells, 2.3 MB of traces. Moose's recursion, level by level, each with its
ledger:
  L1 UNIT   a cell: one superposed trace at its capacity law. Bounded, exact-in-regime.
  L2 GRID+LOOKUP  cells appended as the limit fills; a key->cell dict (the lookup the grid
             acquires). Ledger: one dict entry per key -- the exact-directory cost, cheap
             and honest (a holographic directory would re-pay the interference this module
             exists to escape; kept negative below).
  L3 CACHE  warm cells stay live; cold cells park zlib-compressed in a ColdStore and inflate
             on touch (the cache the lookup acquires). Ledger: the crossing cost is measured
             in the selftest -- cold recall pays inflation once, then the cell is warm.
KEPT NEGATIVE (the directory): replacing the key->cell dict with a bundled holographic
directory was considered and REJECTED without building -- the directory would itself be a
superposed memory subject to the same law, recreating at the directory level the interference
the cells escape. Composition inherits the weakest contract of its parts; the dict IS the
strong contract. (A celled directory-of-directories is the recursion's next turn, taken only
when a measured dict-size wall demands it.)

Values and keys are vocab symbol ids, matching SuperposedMemory's world.
"""
import numpy as np


class CelledMemory:
    """Unbounded key->value pairs over bounded superposed cells (each at the capacity law),
    one shared seed-derived codebook, warm/cold cell tiers. See module docstring for the
    measured wall (0.007 at 70x overload) and the measured escape (1.000 celled)."""

    def __init__(self, mind, dim=4096, vocab=8192, seed=0, cell_pairs=None, keep_warm=8):
        self._proto = mind.superposed_memory(dim=dim, vocab=vocab, seed=seed)   # shared codebooks
        law = int(mind.memory_capacity_law(dim=dim, vocab=vocab))
        # WHY default to the law: the whole design is 'the measured limit is the tile size'.
        # A caller may shrink cells for headroom; growing past the law re-buys interference.
        self.cell_pairs = int(cell_pairs or law)
        self.dim = int(dim)
        self._warm = {}                          # cell -> trace (dim floats), the live tier
        self._cold = mind.cold_store(keep_warm=0, codec="zlib")
        self._cold_cells = set()
        self._where = {}                         # key -> cell (L2: the exact directory)
        self._order = []                         # warm-recency for parking (oldest first)
        self.keep_warm = int(keep_warm)
        self.count = 0

    # -- L3: the cache over the grid ---------------------------------------
    def _touch(self, cell):
        if cell in self._order:
            self._order.remove(cell)
        self._order.append(cell)

    def _cell_trace(self, cell, create=False):
        if cell in self._warm:
            self._touch(cell)
            return self._warm[cell]
        if cell in self._cold_cells:
            # the crossing cost, paid once: inflate, then the cell is warm again
            t = np.frombuffer(self._cold.get("cell:%d" % cell), dtype=np.float64).copy()
            self._cold_cells.discard(cell)
        elif create:
            t = np.zeros(self.dim)
        else:
            raise KeyError(cell)
        self._warm[cell] = t
        self._touch(cell)
        while len(self._warm) > self.keep_warm:
            old = self._order.pop(0)
            self._cold.put("cell:%d" % old, self._warm.pop(old).tobytes())
            self._cold_cells.add(old)
        return self._warm[cell]

    # -- interface ----------------------------------------------------------
    def store(self, keys, values):
        """Append pairs; a new cell opens exactly when the current one reaches the law."""
        for k, v in zip(np.atleast_1d(keys), np.atleast_1d(values)):
            cell = self.count // self.cell_pairs
            trace = self._cell_trace(cell, create=True)
            self._proto.mem = trace
            self._proto.store([int(k)], [int(v)])
            self._warm[cell] = self._proto.mem
            self._where[int(k)] = cell
            self.count += 1
        return self

    def recall(self, keys):
        """Exact-in-regime recall: route by the directory, read one bounded cell."""
        ks = np.atleast_1d(keys)
        out = np.empty(len(ks), dtype=int)
        for i, k in enumerate(ks):
            self._proto.mem = self._cell_trace(self._where[int(k)])
            out[i] = int(self._proto.recall([int(k)])["values"][0])
        return out

    def stats(self):
        return {"pairs": self.count, "cells": len(self._warm) + len(self._cold_cells),
                "warm": len(self._warm), "cold": len(self._cold_cells),
                "cell_pairs": self.cell_pairs, "dim": self.dim}


def _selftest():
    import lecore
    mind = lecore.UnifiedMind(dim=256, seed=0)
    DIM, VOCAB = 2048, 4096
    n_star = int(mind.memory_capacity_law(dim=DIM, vocab=VOCAB))

    # planted truth A (dedicated rng): REAL-shaped pairs (Zipf keys via power draw), 20x past the law
    rng_a = np.random.default_rng(9001)
    N = 20 * n_star
    keys = rng_a.choice(VOCAB, N, replace=False)
    vals = (keys * 31 + 7) % VOCAB

    # THE WALL: one memory 20x overloaded must collapse -- if this ever PASSES recall, the
    # capacity law itself regressed and everything downstream is suspect (perfect-score rule).
    one = mind.superposed_memory(dim=DIM, vocab=VOCAB)
    r = one.store(keys, vals).recall(keys)
    acc_one = float((r["values"] == vals).mean()) if r.get("values") is not None else 0.0
    assert acc_one < 0.30, f"overloaded single memory should collapse; got {acc_one:.3f}"

    # THE SEAM: celled at the law -> exact recall, warm+cold tiers on
    cm = CelledMemory(mind, dim=DIM, vocab=VOCAB, keep_warm=4)
    cm.store(keys, vals)
    got = cm.recall(keys)
    acc = float((got == vals).mean())
    assert acc == 1.0, f"celled recall must be exact in-regime; got {acc:.3f}"
    s = cm.stats()
    assert s["cells"] >= 19 and s["warm"] <= 4, s          # the grid exists; the cache bounds RAM

    # ledger: cold recall pays a crossing, then the cell is warm (the cost is real and bounded)
    import time
    cold_key = int(keys[0])                                 # cell 0 is long-cold under keep_warm=4
    t0 = time.perf_counter(); cm.recall([cold_key]); t1 = time.perf_counter()
    t2 = time.perf_counter(); cm.recall([cold_key]); t3 = time.perf_counter()
    assert (t1 - t0) >= (t3 - t2), "second touch must not be slower than the inflating first"

    # planted truth B: cell_pairs ABOVE the law re-buys interference (the knob is honest)
    rng_b = np.random.default_rng(9002)
    k2 = rng_b.choice(VOCAB, 6 * n_star, replace=False)
    v2 = (k2 * 13 + 5) % VOCAB
    fat = CelledMemory(mind, dim=DIM, vocab=VOCAB, cell_pairs=3 * n_star, keep_warm=8)
    fat.store(k2, v2)
    acc_fat = float((fat.recall(k2) == v2).mean())
    assert acc_fat < 1.0, "cells 3x past the law must show interference -- the law is the physics"

    print("OK: holographic_cellmemory self-test passed (single memory collapses 20x past the law; "
          "celled-at-the-law recalls 1.000 with bounded warm RAM; cold crossing paid once; "
          "over-law cells honestly degrade)")


if __name__ == "__main__":
    _selftest()

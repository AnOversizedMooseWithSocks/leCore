"""
'As below, so above', the algebra floor: the FACTORIZATION wall, broken by distribution.

The exact mirror of the router. ABOVE: looking a key up across K shards by broadcast is O(K) and the
hit erodes as K grows (the broadcast wall) -- fixed by a thin routing layer (sublinear, accuracy
holds). BELOW: pulling a bound product back into its F factors is a search over a V^F space; a single
DENSE resonator (MAP binding over the whole vector) settles into wrong fixed points as that space
grows -- the combinatorial cliff. The SBC resonator runs the SAME search BLOCK-LOCALLY: B small
independent sub-searches with a thin layer combining them. That is the router's move applied to the
algebra. Question: does distributing the search break the wall, matched on dimension?
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "repo"))
from holographic.misc.holographic_resonator import ResonatorNetwork, map_codebook, map_bind
from holographic.misc.holographic_sbc import sbc_codebook, sbc_reconstruct, sbc_resonator

D = 1024                      # matched dimension for both schemes
B, L = 64, 16                 # SBC layout: 64 blocks x 16 positions = 1024 dims
V = 8                         # entries per codebook (per factor)
TRIALS = 24

def dense_solve_rate(F, V, trials):
    ok = 0
    for t in range(trials):
        books = [map_codebook(V, D, seed=1000 * t + f) for f in range(F)]
        rng = np.random.default_rng(7 * t + 1)
        pick = [int(rng.integers(0, V)) for _ in range(F)]
        prod = map_bind(*[books[f][pick[f]] for f in range(F)])
        res = ResonatorNetwork(books).factor(prod, restarts=15, iters=200)
        ok += int(res["solved"] and tuple(res["factors"]) == tuple(pick))
    return ok / trials

def sbc_solve_rate(F, V, trials):
    ok = 0
    for t in range(trials):
        books = [sbc_codebook(B, L, V, seed=1000 * t + f) for f in range(F)]
        rng = np.random.default_rng(7 * t + 1)
        pick = [int(rng.integers(0, V)) for _ in range(F)]
        prod = sbc_reconstruct(pick, books, L)
        picks, validated = sbc_resonator(prod, books, L, restarts=6, iters=50, seed=t)
        ok += int(validated and tuple(picks) == tuple(pick))
    return ok / trials

print(f"Factorization wall: dense (monolithic) vs SBC (block-distributed), matched D={D}")
print(f"sweeping the number of factors F at V={V} per codebook (search space = {V}^F)\n")
print(" F | search space | dense (monolithic) | SBC (block-distributed)")
print("-" * 64)
rows = []
for F in [2, 3, 4, 5, 6, 7, 8]:
    t0 = time.time()
    dr = dense_solve_rate(F, V, TRIALS)
    sr = sbc_solve_rate(F, V, TRIALS)
    rows.append({"F": F, "space": V ** F, "dense": dr, "sbc": sr})
    print(f" {F} | {V**F:>12,d} |       {dr:.2f}         |        {sr:.2f}     "
          f"   ({time.time()-t0:.1f}s)", flush=True)
print("-" * 64)
import json
json.dump({"rows": rows, "D": D, "V": V}, open("_factor_cache.json", "w"))
print("cached.")

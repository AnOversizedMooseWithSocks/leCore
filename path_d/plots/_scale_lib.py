"""Shared helpers for the staged stress test (bounded steps, JSON cache). Atom-cached for speed."""
import sys, os, time, json, functools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "repo"))
import holographic.agents_and_reasoning.holographic_ai as holographic_ai, holographic.misc.holographic_array as holographic_array
from holographic.misc.holographic_array import HoloArray
D = 1024
CACHE = os.path.join(os.path.dirname(__file__), "_scale_cache.json")

# memoize derived_atom: same (seed,name,dim,unitary) -> same vector, so regeneration is free
_orig_atom = holographic_ai.derived_atom
@functools.lru_cache(maxsize=None)
def _cached_atom(seed, name, dim, unitary):
    return _orig_atom(seed, name, dim, unitary)
def derived_atom_cached(seed, name, dim, unitary=False):
    return _cached_atom(seed, name, dim, unitary)
holographic_ai.derived_atom = derived_atom_cached
holographic_array.derived_atom = derived_atom_cached   # the name HoloArray actually calls

def load_cache():
    return json.load(open(CACHE)) if os.path.exists(CACHE) else {}
def save_cache(c):
    json.dump(c, open(CACHE, "w"))

def sample_recall(arr, n=200, broadcast=False, seed=0):
    rng = np.random.default_rng(seed)
    gs = [int(x) for x in rng.choice(len(arr.truth), size=min(n, len(arr.truth)), replace=False)]
    f = arr.broadcast_recall if broadcast else arr.recall
    t0 = time.perf_counter()
    ok = sum(f(g) == arr.truth[g][1] for g in gs)
    dt = (time.perf_counter() - t0) / len(gs)
    return ok / len(gs), dt

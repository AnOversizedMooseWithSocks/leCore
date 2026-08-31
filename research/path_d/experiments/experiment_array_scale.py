"""
Path D stress test: how far does the aligned array go? Two walls, found at very different scales.

  * DIRECTORY-ROUTED storage scales ~unbounded -- each shard is independent and self-upgrade keeps
    every shard's load bounded, so recall stays flat and query cost stays O(1). Limit = RAM.
  * ROUTERLESS BROADCAST query soft-erodes with shard count (the false-alarm tax) and costs
    O(shards) per lookup. It never cliffs to chance, but it loses accuracy as the array grows.

Honest engineering notes (the thin layer's real costs at scale):
  - The coordinator senses capacity on EVERY add -> O(N x probe) to build. At scale, sense
    periodically; here Part 1 drives spin-up on a fixed cadence to measure storage/query in isolation.
  - Unitary keys are an FFT each; we memoize derived_atom so regeneration doesn't dominate.
"""
import sys, os, time, functools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "repo"))
import holographic.agents_and_reasoning.holographic_ai as holographic_ai, holographic.misc.holographic_array as holographic_array
from holographic.misc.holographic_array import HoloArray
D = 1024

# memoize atom generation: same (seed,name,dim,unitary) -> same vector, so it is computed once
_orig = holographic_ai.derived_atom
@functools.lru_cache(maxsize=None)
def _cached(seed, name, dim, unitary):
    return _orig(seed, name, dim, unitary)
_patched = lambda seed, name, dim, unitary=False: _cached(seed, name, dim, unitary)
holographic_ai.derived_atom = _patched
holographic_array.derived_atom = _patched


def sample_recall(arr, n=300, broadcast=False, seed=0):
    rng = np.random.default_rng(seed)
    gs = [int(x) for x in rng.choice(len(arr.truth), size=min(n, len(arr.truth)), replace=False)]
    f = arr.broadcast_recall if broadcast else arr.recall
    t0 = time.perf_counter()
    ok = sum(f(g) == arr.truth[g][1] for g in gs)
    return ok / len(gs), (time.perf_counter() - t0) / len(gs)


def run_scaling(N_MAX=12000, CAP=50, checkpoints=(200, 500, 1000, 2000, 4000, 7000, 12000)):
    print(f"PART 1 -- stream {N_MAX} items, spin up a shard every {CAP} (storage/query at scale)")
    arr = HoloArray(D, seed=42, n_parity=0, add_threshold=0.0)
    rng = np.random.default_rng(2024); rows = []; cp = 0; t0 = time.perf_counter()
    for g in range(N_MAX):
        if g > 0 and g % CAP == 0:
            arr._spin_up()
        arr.add(int(rng.integers(0, arr.n_vals)))
        if cp < len(checkpoints) and (g + 1) == checkpoints[cp]:
            da, dt = sample_recall(arr, broadcast=False, seed=g)
            ba, bt = sample_recall(arr, broadcast=True, seed=g)
            rows.append([g + 1, len(arr.data), da, ba, dt, bt])
            print(f"  N={g+1:6d} shards={len(arr.data):4d} dir={da:.3f} bcast={ba:.3f} "
                  f"| t: dir={dt*1e6:.0f}us bcast={bt*1e6:.0f}us", flush=True)
            cp += 1
    print(f"  built in {time.perf_counter()-t0:.1f}s; {N_MAX} items = ~{N_MAX//D}x a single D={D} vector")
    return rows


def run_ceiling(per_shard=45, Ks=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)):
    print("PART 2 -- broadcast ceiling: routerless recall vs shard count at fixed load")
    out = []
    for K in Ks:
        arr = HoloArray(D, seed=7, n_parity=0, add_threshold=0.0)
        rng = np.random.default_rng(55)
        for s in range(K):
            if s > 0:
                arr._spin_up()
            for _ in range(per_shard):
                arr.add(int(rng.integers(0, arr.n_vals)))
        ba, _ = sample_recall(arr, 300, broadcast=True, seed=K)
        da, _ = sample_recall(arr, 300, broadcast=False, seed=K)
        out.append([K, da, ba])
        print(f"  shards={K:5d} ({K*per_shard:6d} items)  dir={da:.3f}  bcast={ba:.3f}", flush=True)
    return out


def plot(scale, ceil, outpath):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    N=[r[0] for r in scale]; sh=[r[1] for r in scale]; da=[r[2] for r in scale]; ba=[r[3] for r in scale]
    dt=[r[4]*1e6 for r in scale]; bt=[r[5]*1e6 for r in scale]
    K=[r[0] for r in ceil]; cda=[r[1] for r in ceil]; cba=[r[2] for r in ceil]
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))
    a=ax[0]; a.plot(N,da,"o-",color="#239b56",label="directory recall",ms=4)
    a.plot(N,ba,"o-",color="#c0392b",label="broadcast recall",ms=4)
    a.set_xscale("log"); a.set_xlabel("items streamed in"); a.set_ylabel("recall accuracy")
    a.set_ylim(-.03,1.03); a.grid(alpha=.3,which="both")
    a2=a.twinx(); a2.plot(N,sh,"--",color="#2c7fb8",lw=1.2); a2.set_ylabel("# shards",color="#2c7fb8")
    a2.tick_params(axis="y",labelcolor="#2c7fb8"); a.set_title("(a) Streaming: storage holds, broadcast erodes")
    a.legend(loc="lower left",fontsize=8)
    a=ax[1]; a.plot(K,cda,"o-",color="#239b56",label="directory (routed)",ms=4)
    a.plot(K,cba,"o-",color="#c0392b",label="broadcast (routerless)",ms=4)
    a.axhline(.9,color="0.6",ls=":",lw=1); a.axhline(.5,color="0.6",ls=":",lw=1)
    a.set_xscale("log",base=2); a.set_xlabel("# shards"); a.set_ylabel("recall accuracy")
    a.set_ylim(-.03,1.03); a.grid(alpha=.3,which="both"); a.set_title("(b) Broadcast ceiling (directory has none)")
    a.legend(fontsize=8)
    a=ax[2]; a.plot(sh,dt,"o-",color="#239b56",label="directory O(1)",ms=4)
    a.plot(sh,bt,"o-",color="#c0392b",label="broadcast O(shards)",ms=4)
    a.set_xlabel("# shards"); a.set_ylabel("time per query (us)"); a.set_yscale("log")
    a.grid(alpha=.3,which="both"); a.set_title("(c) The price of routerless"); a.legend(fontsize=8)
    fig.suptitle("How far the aligned array goes: directory storage scales ~unbounded; broadcast soft-erodes + O(shards)",
                 fontsize=11.5)
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(outpath,dpi=130,bbox_inches="tight")
    print("plot ->", outpath)


if __name__ == "__main__":
    import sys
    quick = "--smoke" in sys.argv
    if quick:
        s = run_scaling(N_MAX=400, CAP=50, checkpoints=(100, 200, 400))
        c = run_ceiling(per_shard=20, Ks=(1, 2, 4, 8))
    else:
        s = run_scaling()
        c = run_ceiling()
    plot(s, c, os.path.join(os.path.dirname(__file__), "array_scale.png"))
    print("done.")

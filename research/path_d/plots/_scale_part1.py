from _scale_lib import *
N_MAX, CAP = 12000, 50          # spin up a shard every CAP items (~ the per-shard budget at D=1024)
checkpoints = [200, 500, 1000, 2000, 4000, 7000, 12000]
arr = HoloArray(D, seed=42, n_parity=0, add_threshold=0.0)   # drive spin-up manually (fast)
rng = np.random.default_rng(2024)
rows = []; cp = 0; t0 = time.perf_counter()
for g in range(N_MAX):
    if g > 0 and g % CAP == 0:
        arr._spin_up()
    arr.add(int(rng.integers(0, arr.n_vals)))
    if cp < len(checkpoints) and (g + 1) == checkpoints[cp]:
        da, dt = sample_recall(arr, broadcast=False, seed=g)
        ba, bt = sample_recall(arr, broadcast=True, seed=g)
        rows.append([g + 1, len(arr.data), da, ba, dt, bt])
        print(f"  N={g+1:6d} shards={len(arr.data):4d} directory={da:.3f} broadcast={ba:.3f} "
              f"| dir={dt*1e6:.1f}us bcast={bt*1e6:.1f}us", flush=True)
        cp += 1
build_t = time.perf_counter() - t0
print(f"  built {N_MAX} items / {len(arr.data)} shards in {build_t:.1f}s (manual spin-up, no sensing)")
c = load_cache(); c["scale"] = rows; c["build_t"] = build_t; c["N_MAX"] = N_MAX; save_cache(c)
print("cached part1")

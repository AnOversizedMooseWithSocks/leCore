from _scale_lib import *
per_shard = 45
Ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
out = []
for K in Ks:
    arr = HoloArray(D, seed=7, n_parity=0, add_threshold=0.0)
    rng = np.random.default_rng(55)
    for s in range(K):
        if s > 0:
            arr._spin_up()
        for _ in range(per_shard):
            arr.add(int(rng.integers(0, arr.n_vals)))
    ba, _ = sample_recall(arr, n=300, broadcast=True, seed=K)
    da, _ = sample_recall(arr, n=300, broadcast=False, seed=K)
    out.append([K, da, ba])
    print(f"  shards={K:5d} ({K*per_shard:6d} items)  directory={da:.3f}  broadcast={ba:.3f}", flush=True)
k90 = max([k for k, _, b in out if b >= 0.90], default=0)
k50 = max([k for k, _, b in out if b >= 0.50], default=0)
print(f"  broadcast >=90% to {k90} shards, >=50% to {k50} shards; directory flat throughout")
c = load_cache(); c["ceiling"] = out; c["k90"] = k90; c["k50"] = k50; save_cache(c)
print("cached part2")

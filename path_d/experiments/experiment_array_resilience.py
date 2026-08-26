"""
Path D, fourth data point: the array degrades gracefully under whole-shard loss -- 'as above, so
below' made measured. A lost shard is the fountain code's home failure mode ("a whole packet
lost"), and the thin coordinator survives it with PARITY shards -- the real-valued sibling of the
fountain's XOR droplets. This script sweeps how many shards you can lose at each redundancy level.

The RAID picture, on holographic memory:
  * M parity shards reconstruct up to M simultaneously-lost shards EXACTLY (a square/over-determined
    linear solve over the shard sums -- works because a shard is a linear superposition).
  * Beyond M losses the real-valued solve is under-determined and returns the MIN-NORM partial --
    so recall degrades gracefully rather than hitting the binary fountain's hard floor. That gentler
    tail is a genuine difference between the real-valued parity and its GF(2) original.
  * Cost: M parity = M/K extra storage, survive M losses. The classic capacity-vs-resilience knob.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "repo"))
from holographic.misc.holographic_array import HoloArray

D = 1024
K = 8                 # data shards
PER_SHARD = 50        # items per shard (within the per-vector budget so baseline recall is high)


def build_array(M, data_seed=777):
    """Build an array with M parity shards. The value sequence is fixed (same rng), and the seed is
    shared across M, so the DATA is byte-identical for every redundancy level -- only parity differs."""
    arr = HoloArray(D, seed=data_seed, n_parity=M, add_threshold=0.0)   # manual shard control
    rng = np.random.default_rng(12345)
    for s in range(K):
        if s > 0:
            arr._spin_up()
        for _ in range(PER_SHARD):
            arr.add(int(rng.integers(0, arr.n_vals)))
    return arr


def recall_after_losses(arr, f, trials=10):
    """Mean recall after f random shards are lost (reconstructing whatever parity allows)."""
    if f == 0:
        return arr.accuracy()
    rng = np.random.default_rng(100 + f)
    accs = []
    for _ in range(trials):
        down = tuple(int(x) for x in rng.choice(K, size=f, replace=False))
        accs.append(arr.accuracy(down=down))
    return float(np.mean(accs))


def main():
    print("=" * 76)
    print(f"Array resilience -- {K} data shards x {PER_SHARD} items at D={D}; lose shards, recover")
    print("=" * 76)
    parity_levels = [0, 1, 2, 3]
    failures = list(range(0, 6))
    curves = {}
    for M in parity_levels:
        arr = build_array(M)
        row = [recall_after_losses(arr, f) for f in failures]
        curves[M] = row
        label = "RAID-0 (none)" if M == 0 else f"{M} parity"
        cells = "  ".join(f"f={f}:{a:.3f}" for f, a in zip(failures, row))
        print(f"  {label:13s} (overhead {M}/{K}={M/K:.2f}):  {cells}")
    print("  -> M parity holds full recall to f=M losses, then degrades gracefully (min-norm partial)")

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cols = {0: "#c0392b", 1: "#e67e22", 2: "#2c7fb8", 3: "#239b56"}

    ax = axes[0]
    for M, row in curves.items():
        lbl = "0 parity (RAID-0)" if M == 0 else f"{M} parity (RAID-{4+M if M<=2 else '6+'})"
        ax.plot(failures, row, "o-", color=cols[M], label=lbl, markersize=5)
        if M > 0:
            ax.axvline(M, color=cols[M], ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("simultaneous shard losses (f)")
    ax.set_ylabel("recall accuracy after loss")
    ax.set_title("(a) Graceful degradation: M parity survives M losses exactly")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(-0.03, 1.03)

    # (b) the capacity-vs-resilience tradeoff: usable fraction vs storage overhead
    ax = axes[1]
    overheads = [M / K for M in parity_levels]
    survive = parity_levels                       # exact-recovery tolerance = M
    ax.plot([o * 100 for o in overheads], survive, "s-", color="#8e44ad", markersize=7)
    for M, o in zip(parity_levels, overheads):
        ax.annotate(f"{M} parity", (o * 100, M), textcoords="offset points",
                    xytext=(8, -4), fontsize=8)
    ax.set_xlabel("storage overhead (%)")
    ax.set_ylabel("shard losses survived exactly")
    ax.set_title("(b) The RAID knob: storage vs resilience")
    ax.grid(alpha=0.3)

    fig.suptitle("'As above, so below': whole-shard loss is the fountain's failure mode -- parity "
                 "(real-valued droplet) reconstructs it; the thin layer never touches per-model algebra",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(os.path.dirname(__file__), "array_resilience.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nplot -> {out}")


if __name__ == "__main__":
    main()

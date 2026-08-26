"""
Path D, third data point: beating the per-vector ceiling by adding vectors -- the aligned array.

The conserved budget is PER VECTOR (~0.1 x D faithful items). The escape is not cleverer packing
-- it is more vectors. Moose's move: don't cram everything into many partitions of ONE model;
run MANY aligned models in parallel, like a RAID array of disks or a bank of linked GPUs, and when
you near capacity, spin up another and join it. This script measures whether that actually works.

What "aligned" means here, made precise:
  Every shard is its own D-dimensional VSA model, but they SHARE dimension D and the same
  seed-derived alphabet/roles (derived_atom regenerates byte-identical atoms from one seed). So a
  K-shard array is really a K*D-dimensional model with BLOCK-DIAGONAL structure: each block computes
  independently (parallelizable), and the alphabet is shared for free (no per-shard storage). By the
  conservation law, a K*D space holds K * 0.1*D items -- capacity should scale LINEARLY with K.

  PART A -- does capacity scale linearly with shard count? (directory-routed, to isolate capacity)
  PART B -- the self-upgrading array: detect capacity pressure, hot-add an aligned shard, join it.
            Compared against a single fixed model that just hits the wall and collapses.
  PART C -- the honest cost: routerless BROADCAST query (ask every shard at once) carries a
            false-alarm tax that grows with the number of shards. Measured, not hidden.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "repo"))
from holographic.misc.holographic_core import bind, unbind, bundle
from holographic.agents_and_reasoning.holographic_ai import bind_batch, derived_atom

SEED = 23
D = 1024
N_VALS = 256                    # shared value alphabet (the symbols stored)


# --- aligned, seed-regenerated atoms: identical in every shard, stored once as a seed ---------
def value_codebook():
    """The shared symbol alphabet -- regenerated identically in every shard from one seed."""
    return np.stack([derived_atom(SEED, f"val{i}", D, unitary=False) for i in range(N_VALS)])

def key_atom(g):
    """A unitary key for global item index g (exact unbind). Same g -> same key in any shard."""
    return derived_atom(SEED, f"key{g}", D, unitary=True)

def cleanup_conf(V, codebook):
    """Nearest symbol + its cosine confidence. The confidence is what lets a routerless broadcast
    pick the shard that actually holds the item (high conf) over shards that return noise (low)."""
    n = np.linalg.norm(V)
    if n == 0:
        return 0, 0.0
    sims = (codebook @ V) / n
    j = int(sims.argmax())
    return j, float(sims[j])


# --- a shard is just a running superposition of bind(key, value) over its items ---------------
class Shard:
    """One D-dimensional VSA model. Items are added by superposing bind(key_g, value); recall is
    unbind + cleanup. Kept as an unnormalised running sum so adding an item is O(D log D)."""
    def __init__(self):
        self.sum = np.zeros(D)
        self.n = 0
    def add(self, g, val_vec):
        self.sum = self.sum + bind(key_atom(g), val_vec)
        self.n += 1
    def vec(self):
        nrm = np.linalg.norm(self.sum)
        return self.sum / nrm if nrm > 0 else self.sum
    def recall(self, g, codebook):
        return cleanup_conf(unbind(self.vec(), key_atom(g)), codebook)


# ---------------------------------------------------------------------------
# PART A -- capacity scales with shard count (directory-routed)
# ---------------------------------------------------------------------------
def run_part_a():
    print("=" * 76)
    print("PART A -- does faithful capacity scale linearly with the number of aligned shards?")
    print("=" * 76)
    cb = value_codebook()
    Ks = [1, 2, 4, 8]
    Ns = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
    curves = {}
    for K in Ks:
        accs = []
        for N in Ns:
            shards = [Shard() for _ in range(K)]
            truth = {}
            for g in range(N):
                v = np.random.default_rng(g).integers(0, N_VALS)
                truth[g] = (g % K, v)                      # round-robin assignment = the directory
                shards[g % K].add(g, cb[v])
            ok = 0
            for g in range(N):
                s_idx, v_true = truth[g]
                got, _ = shards[s_idx].recall(g, cb)       # directory routes straight to the shard
                ok += (got == v_true)
            accs.append(ok / N)
        curves[K] = accs
        n90 = max([n for n, a in zip(Ns, accs) if a >= 0.90], default=0)
        print(f"  K={K:2d} shard(s) (={K}x D = {K*D:5d} dims):  90%-recall holds to N={n90:4d} "
              f"total items   (~{n90/(K*D):.2f} x (K*D))")
    print("  -> the 90% point should move RIGHT in proportion to K: capacity = K x (per-vector budget)")
    return Ns, curves


# ---------------------------------------------------------------------------
# PART B -- the self-upgrading array vs a single fixed model
# ---------------------------------------------------------------------------
def run_part_b():
    print()
    print("=" * 76)
    print("PART B -- self-upgrading array: detect capacity pressure, spin up a shard, join it")
    print("=" * 76)
    cb = value_codebook()
    N_STREAM = 1100
    ADD_THRESHOLD = 0.90        # when a probe of recent items drops below this, the array is 'full'
    PROBE = 40                  # how many recent items to re-check when sensing capacity

    def broadcast_recall(shards, g):
        """Routerless query: ask EVERY shard, keep the answer from the most-confident one."""
        best_v, best_c = 0, -1.0
        for s in shards:
            v, c = s.recall(g, cb)
            if c > best_c:
                best_v, best_c = v, c
        return best_v, best_c

    # --- the array controller ---
    shards = [Shard()]
    truth = {}
    recent = []
    hist_n, hist_shards, hist_acc = [], [], []
    add_events = []
    for g in range(N_STREAM):
        v = int(np.random.default_rng(10000 + g).integers(0, N_VALS))
        truth[g] = v
        shards[-1].add(g, cb[v])                           # always fill the newest shard
        recent.append(g); recent = recent[-PROBE:]
        # sense capacity on the items just stored; if recall is slipping, the newest shard is full
        if len(recent) >= PROBE:
            hits = sum(broadcast_recall(shards, gg)[0] == truth[gg] for gg in recent)
            if hits / len(recent) < ADD_THRESHOLD:
                shards.append(Shard())                     # spin up a new aligned model...
                add_events.append(g)                       # ...and join it to the array
                recent = []                                # fresh window for the new shard
        if g % 25 == 0 and g > 0:                          # checkpoint: measure on a random sample
            sample = np.random.default_rng(g).choice(g, size=min(60, g), replace=False)
            acc = np.mean([broadcast_recall(shards, int(gg))[0] == truth[int(gg)] for gg in sample])
            hist_n.append(g); hist_shards.append(len(shards)); hist_acc.append(acc)

    # --- the control: ONE fixed shard, no upgrading ---
    single = Shard(); strue = {}
    s_n, s_acc = [], []
    for g in range(N_STREAM):
        v = int(np.random.default_rng(10000 + g).integers(0, N_VALS))
        strue[g] = v
        single.add(g, cb[v])
        if g % 25 == 0 and g > 0:
            sample = np.random.default_rng(g).choice(g, size=min(60, g), replace=False)
            acc = np.mean([single.recall(int(gg), cb)[0] == strue[int(gg)] for gg in sample])
            s_n.append(g); s_acc.append(acc)

    print(f"  streamed {N_STREAM} items; array grew to {len(shards)} aligned shards "
          f"(hot-added at items {add_events[:6]}{'...' if len(add_events) > 6 else ''})")
    print(f"  final recall  -- self-upgrading array: {hist_acc[-1]:.3f}   single fixed model: {s_acc[-1]:.3f}")
    print("  -> the array holds recall by adding capacity; the single model collapses past its wall")
    return (hist_n, hist_shards, hist_acc), (s_n, s_acc), add_events


# ---------------------------------------------------------------------------
# PART C -- the honest cost of routerless broadcast: a false-alarm tax in K
# ---------------------------------------------------------------------------
def run_part_c():
    print()
    print("=" * 76)
    print("PART C -- routerless broadcast query: the false-alarm tax as the array grows")
    print("=" * 76)
    cb = value_codebook()
    per_shard = int(0.04 * D)        # keep every shard comfortably below its wall (~41 items)
    Ks = [1, 2, 4, 8, 16, 32, 64]
    accs = []
    for K in Ks:
        # build K shards, each holding `per_shard` distinct items
        shards = [Shard() for _ in range(K)]
        truth = {}
        g = 0
        for k in range(K):
            for _ in range(per_shard):
                v = int(np.random.default_rng(50000 + g).integers(0, N_VALS))
                truth[g] = (k, v); shards[k].add(g, cb[v]); g += 1
        # query every item by BROADCAST (ask all K shards, trust the most confident)
        ok = 0
        for gg, (k_true, v_true) in truth.items():
            best_v, best_c = 0, -1.0
            for s in shards:
                v, c = s.recall(gg, cb)
                if c > best_c:
                    best_v, best_c = v, c
            ok += (best_v == v_true)
        acc = ok / len(truth)
        accs.append(acc)
        print(f"  K={K:3d} shards x {per_shard} items = {K*per_shard:5d} items:  broadcast recall = {acc:.3f}")
    print("  -> querying more shards = more chances for a spurious match; keep per-shard headroom or use a directory")
    return Ks, per_shard, accs


# ---------------------------------------------------------------------------
def make_plot(a_Ns, a_curves, b_array, b_single, b_adds, c_Ks, c_per, c_accs, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    kcol = {1: "#222222", 2: "#c0392b", 4: "#2c7fb8", 8: "#239b56"}

    ax = axes[0]
    for K, accs in a_curves.items():
        ax.plot(a_Ns, accs, "o-", color=kcol[K], label=f"{K} shard(s)", markersize=4)
    ax.axhline(0.9, color="0.6", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("total items stored across the array")
    ax.set_ylabel("recall accuracy")
    ax.set_title("(a) Capacity scales linearly with shard count")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both"); ax.set_ylim(-0.03, 1.03)

    ax = axes[1]
    hist_n, hist_shards, hist_acc = b_array
    s_n, s_acc = b_single
    ax.plot(hist_n, hist_acc, "-", color="#239b56", lw=2, label="self-upgrading array")
    ax.plot(s_n, s_acc, "-", color="#c0392b", lw=2, label="single fixed model")
    ax.set_xlabel("items streamed in"); ax.set_ylabel("recall accuracy")
    ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(hist_n, hist_shards, "--", color="#2c7fb8", lw=1.3, label="# shards (array)")
    ax2.set_ylabel("# shards in the array", color="#2c7fb8")
    ax2.tick_params(axis="y", labelcolor="#2c7fb8")
    ax.set_title("(b) The array upgrades itself; the single model collapses")
    ax.legend(loc="center left", fontsize=8)

    ax = axes[2]
    ax.plot(c_Ks, c_accs, "o-", color="#8e44ad", markersize=5)
    ax.axhline(1.0, color="0.6", ls=":", lw=1)
    ax.set_xscale("log", base=2); ax.set_xlabel("# shards queried by broadcast")
    ax.set_ylabel(f"broadcast recall ({c_per} items/shard)")
    ax.set_title("(c) The honest cost: broadcast false-alarm tax")
    ax.grid(alpha=0.3, which="both"); ax.set_ylim(-0.03, 1.03)

    fig.suptitle("RAID for holographic memory: beat the per-vector ceiling by adding aligned models "
                 "(capacity = K x budget; alignment free via seed-regeneration)", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    print(f"\nplot -> {outpath}")


if __name__ == "__main__":
    a_Ns, a_curves = run_part_a()
    b_array, b_single, b_adds = run_part_b()
    c_Ks, c_per, c_accs = run_part_c()
    out = os.path.join(os.path.dirname(__file__), "vsa_array.png")
    make_plot(a_Ns, a_curves, b_array, b_single, b_adds, c_Ks, c_per, c_accs, out)
    print("\ndone.")

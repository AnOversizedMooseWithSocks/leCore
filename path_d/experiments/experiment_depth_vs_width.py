"""
Path D, second data point: does DEPTH (recursion) buy back the capacity that WIDTH can't?

Last experiment measured the flat wall: one D-dim vector holds only ~0.1 x D items at
cleanup-gated recall, ~0.02 x D for continuous compute. The four themes Moose named --
demoscene (max structure from a minimal kernel), fractal recursion, inception (nesting),
multiple layers -- are all ONE escape route: stop widening a flat bundle, and instead spend
DEPTH, cleanup-gating each level so crosstalk resets before it compounds. holostuff already
has the depth machinery (encode_tree, peel, the measured inception depth law); leOS gave us
the width primitive (now holographic_superposed.py). This script puts them on the same axes.

Everything is built on the BARE kernel (bind/bundle/cleanup) with seed-regenerated atoms --
the demoscene point: the whole structure expands deterministically from a tiny seed, so the
stored state is minimal and the richness comes from recursion, not from storing more.

  PART 1 -- flat width capacity: K leaf-symbols in ONE bundle (one level). The wall.
  PART 2 -- recursive TREE (inception/fractal): the same leaf-symbols arranged as a B-ary tree
            of depth d, encoded recursively into ONE vector, decoded by recursive cleanup-gated
            peel. Does the max number of leaves recoverable EXCEED the flat wall?
  PART 3 -- clean-nesting chain: the inception depth law -- how deep a nested chain runs before
            the buried value stops decoding, vs dimension D.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "repo"))
from holographic.misc.holographic_core import bind, unbind, bundle
from holographic.agents_and_reasoning.holographic_ai import bind_batch, derived_atom

SEED = 11


# --- seed-regenerated alphabet & roles (demoscene: format the drive from a seed) -----------
def leaf_alphabet(D, n, seed):
    """n clean leaf atoms, each a pure function of (seed, name) -- regenerate-from-seed."""
    return np.stack([derived_atom(seed, f"leaf{i}", D, unitary=False) for i in range(n)])

def position_roles(D, B, seed):
    """B unitary position roles (exact unbind) -- reused at every level, so the whole tree is
    laid down from just (seed, B). Minimal stored state; recursion supplies the structure."""
    return np.stack([derived_atom(seed, f"pos{i}", D, unitary=True) for i in range(B)])

def cleanup_index(V, codebook):
    """Snap a noisy vector to the nearest alphabet atom -> its index. The discrete decision that
    RESETS crosstalk (why cleanup-gated recall holds more than raw continuous superposition)."""
    n = np.linalg.norm(V)
    if n == 0:
        return 0
    return int((codebook @ V).argmax())   # codebook atoms ~unit norm, so dot ~ cosine*||V||


# --- PART 1 & 2 shared: recursive encode / decode of a B-ary tree on the bare kernel --------
def encode_tree(leaf_idx, B, depth, roles, alphabet):
    """Encode a list of B^depth leaf indices into ONE vector, recursively.
       leaf  : the leaf's clean atom.
       node  : bundle( bind(pos_i, encode(child_i)) for i ) -- self-similar at every level."""
    if depth == 0:
        return alphabet[leaf_idx[0]]
    step = len(leaf_idx) // B
    kids = [encode_tree(leaf_idx[i*step:(i+1)*step], B, depth-1, roles, alphabet) for i in range(B)]
    return bundle(bind_batch(roles, np.stack(kids)))

def decode_tree(V, B, depth, roles, alphabet):
    """Recover the leaf indices by recursive unbind, cleaning up at the leaves (cleanup-gating).
       At internal nodes the recovered child is a noisy BUNDLE (no codebook to clean it to), so
       noise compounds with depth -- that compounding, traded against per-level branching B, is
       exactly the depth x width surface."""
    if depth == 0:
        return [cleanup_index(V, alphabet)]
    out = []
    inv_roles = np.concatenate([roles[:, :1], roles[:, :0:-1]], axis=1)   # involution per role
    for i in range(B):
        child = bind(V, inv_roles[i])           # unbind(V, role_i) = bind(V, involution(role_i))
        out += decode_tree(child, B, depth-1, roles, alphabet)
    return out

def tree_recall(D, B, depth, n_alpha=64, trials=4, seed0=0):
    """Mean fraction of leaves correctly recovered for a B-ary depth-d tree at dimension D."""
    fr = []
    for t in range(trials):
        rng = np.random.default_rng(1000 + seed0 + t)
        alphabet = leaf_alphabet(D, n_alpha, seed=SEED + t)
        roles = position_roles(D, B, seed=SEED + t)
        n_leaves = B ** depth
        true = rng.integers(0, n_alpha, size=n_leaves)
        M = encode_tree(list(true), B, depth, roles, alphabet)
        got = decode_tree(M, B, depth, roles, alphabet)
        fr.append(np.mean(np.array(got) == true))
    return float(np.mean(fr))


def run_parts_1_2():
    print("=" * 74)
    print("PARTS 1+2 -- flat width (depth=1) vs recursive tree (depth>1): can depth hold more?")
    print("=" * 74)
    D = 1024
    # (B, depth) combos spanning a range of total leaves; depth=1 is the FLAT baseline (the wall)
    combos = {
        1: [(10, 1), (20, 1), (40, 1), (60, 1), (100, 1), (150, 1), (220, 1)],          # flat
        2: [(4, 2), (6, 2), (8, 2), (12, 2), (16, 2), (22, 2), (30, 2)],                 # B^2 leaves
        3: [(3, 3), (4, 3), (5, 3), (6, 3), (8, 3), (10, 3)],                            # B^3 leaves
        4: [(3, 4), (4, 4), (5, 4), (6, 4)],                                             # B^4 leaves
        5: [(3, 5), (4, 5)],                                                             # B^5 leaves
    }
    results = {}   # depth -> list of (total_leaves, fraction, recovered_count)
    best_flat = 0.0
    best_recursive = (0.0, None)
    for depth, lst in combos.items():
        row = []
        for B, d in lst:
            frac = tree_recall(D, B, d)
            total = B ** d
            count = frac * total
            row.append((total, frac, count))
            tag = "FLAT " if depth == 1 else f"d={depth}"
            print(f"  {tag}  B={B:3d}  leaves={total:5d}  recovered_frac={frac:.3f}  "
                  f"-> faithful_items={count:6.1f}")
            if depth == 1:
                best_flat = max(best_flat, count)
            else:
                if count > best_recursive[0]:
                    best_recursive = (count, (B, depth))
        results[depth] = row
    print("-" * 74)
    print(f"  BEST flat (one wide bundle)      : {best_flat:6.1f} faithful items")
    print(f"  BEST recursive (depth>1)         : {best_recursive[0]:6.1f} faithful items "
          f"at B={best_recursive[1][0]}, depth={best_recursive[1][1]}")
    mult = best_recursive[0] / best_flat if best_flat else float('nan')
    print(f"  -> recursion holds {mult:.1f}x more faithful items than flat at D={D}")
    return D, results, best_flat, best_recursive


# --- PART 3: clean-nesting chain (the inception depth law) ----------------------------------
def chain_depth_law(D, max_depth=40, n_alpha=64, trials=6):
    """A linked list: each level holds ONE value (cleanu-able) + a NEXT pointer to the rest.
       node_j = bundle( bind(VAL, value_j), bind(NEXT, node_{j+1} ) ).
       Decode by peeling: value = cleanup(unbind(node, VAL)); node = unbind(node, NEXT); recurse.
       Returns the depth at which per-level recovery first drops below 90% (the depth law)."""
    accs = np.zeros(max_depth)
    for t in range(trials):
        rng = np.random.default_rng(7000 + t)
        alphabet = leaf_alphabet(D, n_alpha, seed=SEED + 50 + t)
        VAL = derived_atom(SEED + 50 + t, "VAL", D, unitary=True)
        NEXT = derived_atom(SEED + 50 + t, "NEXT", D, unitary=True)
        invVAL = np.concatenate([[VAL[0]], VAL[:0:-1]])
        invNEXT = np.concatenate([[NEXT[0]], NEXT[:0:-1]])
        vals = rng.integers(0, n_alpha, size=max_depth)
        # build from the deepest level up
        node = alphabet[vals[max_depth - 1]]
        for j in range(max_depth - 2, -1, -1):
            node = bundle(np.stack([bind(VAL, alphabet[vals[j]]), bind(NEXT, node)]))
        # peel from the top down, scoring each level's value recovery
        cur = node
        for j in range(max_depth):
            v_est = bind(cur, invVAL)                      # unbind(cur, VAL)
            accs[j] += (cleanup_index(v_est, alphabet) == vals[j])
            cur = bind(cur, invNEXT)                       # unbind(cur, NEXT) -> the rest
    accs /= trials
    depth90 = int(np.argmax(accs < 0.9)) if np.any(accs < 0.9) else max_depth
    return accs, depth90


def run_part3():
    print()
    print("=" * 74)
    print("PART 3 -- clean-nesting chain: the inception depth law (depth scales with dimension)")
    print("=" * 74)
    dims = [256, 512, 1024, 2048]
    curves = {}
    for D in dims:
        accs, depth90 = chain_depth_law(D)
        curves[D] = accs
        print(f"  D={D:5d}:  per-level recovery stays >=90% to nesting depth {depth90:3d}")
    return curves


# --- plot ----------------------------------------------------------------------------------
def make_plot(D, tree_results, flat_best, rec_best, chain_curves, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    dcol = {1: "#222222", 2: "#c0392b", 3: "#e67e22", 4: "#2c7fb8", 5: "#239b56"}

    # (a) faithful items vs total leaves, by depth -- the depth-beats-width surface
    ax = axes[0]
    for depth, row in tree_results.items():
        tot = [r[0] for r in row]; cnt = [r[2] for r in row]
        lbl = "depth=1 (FLAT)" if depth == 1 else f"depth={depth}"
        ax.plot(tot, cnt, "o-", color=dcol[depth], label=lbl, markersize=4)
    ax.plot([1, 5000], [1, 5000], ":", color="0.6", lw=1, label="perfect (frac=1)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("total leaves packed into one vector")
    ax.set_ylabel("faithfully recovered leaves")
    ax.set_title(f"(a) Faithful capacity is ~conserved (~0.1 x D)  (D={D})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # (b) recovered FRACTION vs total leaves -- where each depth's wall sits
    ax = axes[1]
    for depth, row in tree_results.items():
        tot = [r[0] for r in row]; frac = [r[1] for r in row]
        lbl = "depth=1 (FLAT)" if depth == 1 else f"depth={depth}"
        ax.plot(tot, frac, "o-", color=dcol[depth], label=lbl, markersize=4)
    ax.axhline(0.5, color="0.6", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("total leaves packed into one vector")
    ax.set_ylabel("fraction recovered (after cleanup)")
    ax.set_title("(b) Same wall, however you arrange it")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both"); ax.set_ylim(-0.03, 1.03)

    # (c) the inception depth law: per-level recovery vs nesting depth, by dimension
    ax = axes[2]
    ccol = {256: "#c0392b", 512: "#e67e22", 1024: "#2c7fb8", 2048: "#239b56"}
    for D2, accs in chain_curves.items():
        ax.plot(range(1, len(accs) + 1), accs, "-", color=ccol[D2], label=f"D={D2}", lw=1.5)
    ax.axhline(0.9, color="0.6", ls=":", lw=1)
    ax.set_xlabel("nesting depth (clean chain)")
    ax.set_ylabel("per-level value recovery")
    ax.set_title("(c) Inception: structural DEPTH scales with D")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(-0.03, 1.03)

    fig.suptitle("A D-vector has a CONSERVED capacity budget (~0.1 x D): recursion/inception spends it on "
                 "structural DEPTH, not on more items (holostuff kernel, seed-regenerated atoms)", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    print(f"\nplot -> {outpath}")


if __name__ == "__main__":
    D, tree_results, flat_best, rec_best = run_parts_1_2()
    chain_curves = run_part3()
    out = os.path.join(os.path.dirname(__file__), "depth_vs_width.png")
    make_plot(D, tree_results, flat_best, rec_best, chain_curves, out)
    print("\ndone.")

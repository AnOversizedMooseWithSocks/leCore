"""
holographic_array.py  --  the thin layer that coordinates BETWEEN aligned VSA models.

"As above, so below." The fixes that work inside one model work one rung up, between models --
and holostuff's own fountain module already says why: a fountain DROPLET (XOR of a random subset
of blocks) is the binary sibling of a BUNDLE (superposition of a random subset of vectors). The
within-vector robustness (a superposition spreads information, so partial loss degrades gracefully)
becomes across-shard robustness by the SAME linearity: a shard is a running sum, so a PARITY shard
(a linear combination of shards -- the real-valued sibling of an XOR droplet) reconstructs a lost
shard EXACTLY by subtraction. A lost shard is precisely the fountain's home failure mode -- "a whole
packet lost" -- so the array-level fix is the fountain axis applied to whole shards.

This coordinator is deliberately THIN: it never changes the per-model VSA algebra. Each shard stays
a vanilla D-dimensional bind/bundle/cleanup model. The layer only does the four things a RAID
controller does -- align, place, sense-and-grow, and protect:

  * ALIGN  -- every shard regenerates byte-identical atoms from one seed (derived_atom). Free.
  * PLACE  -- new items fill the newest shard (with a directory for clean routing).
  * GROW   -- sense capacity pressure (recall slipping on recent items) and hot-add a shard.
  * PROTECT-- keep `n_parity` parity shards; on losing up to n_parity shards, reconstruct exactly.

Capacity is per-vector (~0.1 x D); the array stacks ceilings linearly (K x budget). Parity costs
one shard's storage per redundancy level and survives that many simultaneous shard losses -- and,
like the fountain it mirrors, it CANNOT recover more losses than it has parity (an information
floor, not a bug).
"""
import numpy as np
from holographic.misc.holographic_core import bind, unbind, bundle
from holographic.agents_and_reasoning.holographic_ai import derived_atom


class HoloArray:
    def __init__(self, dim, seed=0, n_parity=1, add_threshold=0.90, probe=40, n_vals=256):
        self.dim = dim
        self.seed = seed
        self.n_parity = n_parity
        self.add_threshold = add_threshold      # recall on recent items below this -> grow
        self.probe = probe
        self.n_vals = n_vals
        self.codebook = np.stack([derived_atom(seed, f"val{i}", dim) for i in range(n_vals)])
        self.data = [np.zeros(dim)]             # data shards, each a running (unnormalised) sum
        # n_parity parity rows; row 0 is the plain sum (RAID-5), extra rows use fixed pseudo-random
        # coefficients (RAID-6+). Empty when n_parity == 0 (no redundancy).
        self.par_coef = [[self._coef(m, 0)] for m in range(n_parity)]
        self.parity = [np.zeros(dim) for _ in range(n_parity)]  # parity shards = coef-weighted sums
        self.truth = {}                         # g -> (shard, value_index)   (the directory)
        self.recent = []
        self.add_log = []                       # global indices where a shard was hot-added

    # -- aligned atoms (identical in every shard; stored as a seed, not a matrix) --------------
    def _key(self, g):
        return derived_atom(self.seed, f"key{g}", self.dim, unitary=True)   # exact unbind

    def _coef(self, m, shard):
        """A fixed coefficient for parity row m on a given shard -- pseudo-random but reproducible,
        so future items in that shard update parity consistently. Row 0 is always 1 (the sum)."""
        if m == 0:
            return 1.0
        h = derived_atom(self.seed, f"coef{m}-{shard}", 4)   # tiny derived draw -> a stable scalar
        return float(h[0] * 3.0)

    # -- PLACE + GROW --------------------------------------------------------------------------
    def add(self, value_index):
        """Store one symbol. Fills the newest shard; senses capacity and hot-adds a shard if the
        array is filling up. Returns the global item index."""
        g = len(self.truth)
        k = len(self.data) - 1                  # newest data shard
        atom = bind(self._key(g), self.codebook[value_index])
        self.data[k] = self.data[k] + atom
        for m in range(self.n_parity):
            self.parity[m] = self.parity[m] + self.par_coef[m][k] * atom
        self.truth[g] = (k, value_index)
        self.recent.append(g); self.recent = self.recent[-self.probe:]
        # sense capacity on what we just stored; if recall is slipping, the newest shard is full
        if len(self.recent) >= self.probe:
            hits = sum(self._recall_one(gg)[0] == self.truth[gg][1] for gg in self.recent)
            if hits / len(self.recent) < self.add_threshold:
                self._spin_up()
        return g

    def _spin_up(self):
        """Mint a fresh aligned shard and join it to the array (the 'upgrade')."""
        self.data.append(np.zeros(self.dim))
        knew = len(self.data) - 1
        for m in range(self.n_parity):
            self.par_coef[m].append(self._coef(m, knew))   # fix this shard's parity coefficients
        self.recent = []
        self.add_log.append(len(self.truth))

    # -- PROTECT: reconstruct lost shards from parity (the fountain-over-reals) -----------------
    def _live_sums(self, down):
        """Return each data shard's sum, reconstructing any in `down` from the parity shards.
        Solve  C[:,down] X = parity - C[:,live] @ data[live]  for the lost sums X (per dimension).
        Exact when n_parity >= |down|; underdetermined (graceful/partial) when not -- the floor."""
        K = len(self.data)
        if not down:
            return list(self.data)
        if self.n_parity == 0:                                     # no redundancy: lost is lost
            return [np.zeros(self.dim) if i in down else d for i, d in enumerate(self.data)]
        C = np.array([[self.par_coef[m][i] for i in range(K)] for m in range(self.n_parity)])
        live = [i for i in range(K) if i not in down]
        P = np.stack(self.parity)                                   # (n_parity, D)
        rhs = P - (C[:, live] @ np.stack([self.data[i] for i in live]) if live else 0.0)
        X, *_ = np.linalg.lstsq(C[:, list(down)], rhs, rcond=None)  # (|down|, D)
        sums = list(self.data)
        for idx, i in enumerate(down):
            sums[i] = X[idx]
        return sums

    # -- RECALL --------------------------------------------------------------------------------
    def _norm(self, v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _recall_one(self, g, sums=None):
        """Directory-routed recall of item g (uses the live/reconstructed shard sums)."""
        sums = self.data if sums is None else sums
        k, _ = self.truth[g]
        est = unbind(self._norm(sums[k]), self._key(g))
        sims = self.codebook @ est
        j = int(sims.argmax())
        return j, float(sims[j] / (np.linalg.norm(est) + 1e-12))

    def recall(self, g, down=()):
        """Recall item g, transparently reconstructing any shards in `down` from parity first."""
        sums = self._live_sums(list(down)) if down else self.data
        return self._recall_one(g, sums)[0]

    def broadcast_recall(self, g, down=()):
        """Routerless: ask every (live/reconstructed) shard, trust the most confident answer."""
        sums = self._live_sums(list(down)) if down else self.data
        best_v, best_c = 0, -1.0
        for k in range(len(self.data)):
            est = unbind(self._norm(sums[k]), self._key(g))
            sims = self.codebook @ est
            j = int(sims.argmax()); c = float(sims[j] / (np.linalg.norm(est) + 1e-12))
            if c > best_c:
                best_v, best_c = j, c
        return best_v

    # -- ROUTE: content-addressable shard routing by key-sketch (breaks the broadcast wall) ----
    def _shard_members(self):
        """Item indices held by each shard (inverting the directory)."""
        members = [[] for _ in range(len(self.data))]
        for g, (k, _) in self.truth.items():
            members[k].append(g)
        return members

    def _build_sketches(self):
        """One SKETCH per shard = bundle of that shard's keys -- the holographic 'and' of what it holds. This
        is the index that lets a query route to its shard by matching, WITHOUT consulting the directory; it is
        rebuilt lazily whenever the shard count changes."""
        self._sketches = np.stack([
            bundle(np.stack([self._key(g) for g in mem])) if mem else np.zeros(self.dim)
            for mem in self._shard_members()])
        self._sketch_shards = len(self.data)
        return self._sketches

    def routed_recall(self, g, c=8):
        """Content-addressable recall by SKETCH ROUTING: match item g's key against the shard sketches, keep
        the top-c candidate shards, and unbind+cleanup ONLY those -- O(c) unbinds instead of broadcast's
        O(shards). It stays accurate where broadcast erodes because a key sits ~1/sqrt(load) inside its own
        shard's sketch, far above the 1/sqrt(D) noise from the others, so routing is a clean decision while the
        broadcast value-cleanup vote drowns as shards grow."""
        if getattr(self, "_sketches", None) is None or self._sketch_shards != len(self.data):
            self._build_sketches()
        key = self._key(g)
        scores = self._sketches @ key
        c = min(c, len(self.data))
        cand = np.argpartition(-scores, c - 1)[:c]                  # top-c candidate shards (one matmul)
        best_v, best_c = 0, -1.0
        for s in cand:                                             # unbind+cleanup only the candidates
            est = unbind(self._norm(self.data[s]), key)
            sims = self.codebook @ est
            j = int(sims.argmax())
            cf = float(sims[j] / (np.linalg.norm(est) + 1e-12))
            if cf > best_c:
                best_v, best_c = j, cf
        return best_v

    # -- accuracy helper -----------------------------------------------------------------------
    def accuracy(self, sample=None, down=(), broadcast=False):
        gs = list(self.truth) if sample is None else sample
        f = self.broadcast_recall if broadcast else self.recall
        return float(np.mean([f(g, down=down) == self.truth[g][1] for g in gs]))


def _selftest():
    rng = np.random.default_rng(0)
    D = 1024
    # (1) RAID-5: lose one shard, parity restores recall EXACTLY ------------------------------
    arr = HoloArray(D, seed=1, n_parity=1, add_threshold=0.0)   # threshold 0 -> manual shards
    # force 4 shards by hand, ~40 items each (within per-shard capacity)
    for s in range(4):
        if s > 0:
            arr._spin_up()
        for _ in range(40):
            arr.add(int(rng.integers(0, arr.n_vals)))
    base = arr.accuracy()
    lost = 2                                                     # pretend shard 2 is gone
    no_parity = np.mean([arr._recall_one(g, [d if i != lost else np.zeros(D)
                                             for i, d in enumerate(arr.data)])[0] == v
                         for g, (k, v) in arr.truth.items()])
    with_parity = arr.accuracy(down=(lost,))
    print(f"[array selftest] 4 shards, lose 1:  baseline={base:.3f}  "
          f"lost-shard-zeroed={no_parity:.3f}  parity-reconstructed={with_parity:.3f}")
    assert with_parity >= base - 1e-9, "RAID-5 parity must restore recall exactly"
    assert no_parity < base - 0.1, "without parity, a lost shard should drop recall"

    # (2) information floor: 1 parity cannot recover 2 lost shards ----------------------------
    two_down = arr.accuracy(down=(1, 2))
    print(f"[array selftest] same array, lose 2 with only 1 parity: recall={two_down:.3f} "
          f"(information floor -- mirrors the fountain's 'too few droplets -> nothing')")

    # (3) RAID-6: 2 parity survives 2 losses --------------------------------------------------
    arr2 = HoloArray(D, seed=2, n_parity=2, add_threshold=0.0)
    for s in range(4):
        if s > 0:
            arr2._spin_up()
        for _ in range(40):
            arr2.add(int(rng.integers(0, arr2.n_vals)))
    b2 = arr2.accuracy(); r2 = arr2.accuracy(down=(1, 3))
    print(f"[array selftest] RAID-6 (2 parity), lose 2:  baseline={b2:.3f}  reconstructed={r2:.3f}")
    assert r2 >= b2 - 1e-9, "2 parity must reconstruct 2 lost shards exactly"

    # (4) sketch-routed recall: accurate as the directory, touching only c shards (the broadcast-wall fix) --
    arr3 = HoloArray(D, seed=3, n_parity=0, add_threshold=0.0, probe=10 ** 9)
    for s in range(48):
        if s > 0:
            arr3._spin_up()
        for _ in range(30):
            arr3.add(int(rng.integers(0, arr3.n_vals)))
    pool = list(arr3.truth)
    samp = [pool[i] for i in np.random.default_rng(5).choice(len(pool), 200, replace=False)]
    dir_acc = float(np.mean([arr3.recall(g) == arr3.truth[g][1] for g in samp]))
    rt_acc = float(np.mean([arr3.routed_recall(g, c=8) == arr3.truth[g][1] for g in samp]))
    bc_acc = float(np.mean([arr3.broadcast_recall(g) == arr3.truth[g][1] for g in samp]))
    print(f"[array selftest] 48 shards: directory={dir_acc:.3f}  routed(c=8)={rt_acc:.3f}  broadcast={bc_acc:.3f} "
          f"(routed touches 8 of 48 shards)")
    assert rt_acc >= bc_acc - 1e-9, "sketch routing should be at least as accurate as full broadcast"
    assert rt_acc >= dir_acc - 0.05, "sketch routing should track the directory while touching far fewer shards"
    print("[array selftest] OK")


if __name__ == "__main__":
    _selftest()

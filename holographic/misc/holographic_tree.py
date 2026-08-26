"""
holographic_tree.py -- breaking a too-big memory into a deterministic tree.

The honest limitation we keep running into: a single holographic store is a
*bundle*, and a bundle has finite capacity.  Superpose too many associations into
one trace and recall collapses -- measured here, a 2048-d associative memory
recalls 100% of 64 pairs but only ~1% of 2048.  Slime mould has the same problem
at the cell scale: pure diffusion can't service a large body, so Physarum
"overcomes the size limitations of purely diffusive transport" by resolving its
broad foraging mass into a *hierarchical vein network* -- a few thick trunks,
many thin twigs, reinforced by flux.  (Dasgupta & Freund's random projection tree
is the same idea in computer-science clothes; SQL's B-tree indexes and hash
partitions are the same idea again -- never scan the whole table, descend an index.)

So instead of one flat trace we build a tree:

  * Each internal node owns one deterministic random hyperplane (seeded by the
    node's id, so the whole tree is reproducible from a single seed -- very
    demoscene) and splits its items at the median of their projection onto it.
    Balanced, log-depth, and adapts to the data the way an RP-tree does.
  * Each leaf holds a small HolographicMemory -- few enough pairs to stay well
    inside capacity, so recall there is reliable.
  * A query routes down the tree; with a beam it can back-track into the nearest
    sibling cells (the standard RP-tree trick) for robustness to noisy cues.
  * Every leaf counts the "flux" of queries that reach it -- the thick-vein /
    thin-vein signal, exposed for inspection.

Net effect: recall stays high as the dataset grows, and a query touches
O(leaf_size . log N) items instead of all N.  Pure numpy, built on the engine's
HolographicMemory (bind/unbind) and random_vector.
"""

import itertools
import heapq
import hashlib
import numpy as np

from holographic.agents_and_reasoning.holographic_ai import HolographicMemory, random_vector


class HoloTree:
    """A deterministic recursive partition of key vectors with a small
    holographic associative memory at every leaf."""

    def __init__(self, dim, leaf_size=64, seed=0, max_depth=40):
        self.dim = dim
        self.leaf_size = leaf_size
        self.seed = seed
        self.max_depth = max_depth
        self.root = None
        self.keys = None
        self.values = None
        self.last_comparisons = 0          # set by recall(), for honest cost stats

    # ---- construction ----
    def _hyperplane(self, node_id):
        """A reproducible random unit direction for this node (seed + node id)."""
        rng = np.random.default_rng([self.seed, node_id])
        h = rng.standard_normal(self.dim)
        return h / np.linalg.norm(h)

    def build(self, keys, values=None):
        """keys: (N, dim) unit vectors.  values: (N, dim) atoms to associate with
        each key, or None for content-addressing (the key is its own value)."""
        self.keys = np.asarray(keys, float)
        self.values = np.asarray(values, float) if values is not None else None
        self.root = self._build(np.arange(len(self.keys)), depth=0, node_id=1)
        return self

    def _build(self, idx, depth, node_id):
        if len(idx) <= self.leaf_size or depth >= self.max_depth:
            return self._leaf(idx)
        h = self._hyperplane(node_id)
        proj = self.keys[idx] @ h
        thr = float(np.median(proj))
        left, right = idx[proj <= thr], idx[proj > thr]
        if len(left) == 0 or len(right) == 0:           # everything on one side
            return self._leaf(idx)
        return {"leaf": False, "h": h, "thr": thr,
                "left": self._build(left, depth + 1, 2 * node_id),
                "right": self._build(right, depth + 1, 2 * node_id + 1)}

    def _leaf(self, idx):
        mem = HolographicMemory(self.dim)
        for i in idx:
            mem.learn(self.keys[i], self.values[i] if self.values is not None else self.keys[i])
        return {"leaf": True, "idx": np.asarray(idx), "mem": mem, "flux": 0}

    # ---- query ----
    def _route(self, key, beam):
        """Best-first descent collecting up to `beam` leaves.  When we pick a side
        of a hyperplane we remember the other side, prioritised by how close the
        key fell to the boundary -- so a beam revisits the most plausible
        alternatives first (RP-tree back-tracking)."""
        leaves, cnt = [], itertools.count()
        pq = [(0.0, next(cnt), self.root)]
        while pq and len(leaves) < beam:
            _, _, node = heapq.heappop(pq)
            while not node["leaf"]:
                d = float(key @ node["h"]) - node["thr"]
                near, far = (node["left"], node["right"]) if d <= 0 else (node["right"], node["left"])
                heapq.heappush(pq, (abs(d), next(cnt), far))
                node = near
            leaves.append(node)
        return leaves

    def recall(self, key, beam=1):
        """Return the global index of the best-matching stored item.  Routes to
        leaf(s), reads the holographic memory there, and cleans up against just
        those candidates."""
        key = np.asarray(key, float)
        comps, best, bid = 0, -2.0, -1
        for leaf in self._route(key, beam):
            leaf["flux"] += 1
            cand = leaf["idx"]
            if self.values is not None:                 # key->value associative recall
                est, ref = leaf["mem"].recall(key), self.values[cand]
            else:                                        # content addressing: cosine to the cue
                est, ref = key, self.keys[cand]
            sims = ref @ est                             # cleanup over this leaf only
            comps += len(cand)
            j = int(np.argmax(sims))
            if sims[j] > best:
                best, bid = float(sims[j]), int(cand[j])
        self.last_comparisons = comps
        return bid

    # ---- inspection ----
    def stats(self):
        depths, sizes = [], []

        def walk(n, d):
            if n["leaf"]:
                depths.append(d); sizes.append(len(n["idx"]))
            else:
                walk(n["left"], d + 1); walk(n["right"], d + 1)
        walk(self.root, 0)
        return dict(leaves=len(sizes), depth=max(depths), avg_leaf=float(np.mean(sizes)),
                    max_leaf=int(max(sizes)), min_leaf=int(min(sizes)))

    def flux(self):
        """Per-leaf query counts gathered so far -- the vein-thickness signal."""
        out = []

        def walk(n):
            if n["leaf"]:
                out.append(n["flux"])
            else:
                walk(n["left"]); walk(n["right"])
        walk(self.root)
        return out


class HoloForest:
    """A handful of HoloTrees grown from different seeds.  A single tree can miss
    the true neighbour when the cue falls just across one of its hyperplanes;
    independent trees draw their boundaries elsewhere, so the *union* of their
    candidate leaves almost always contains it.  This is what lets the structure
    reach exact-scan recall at a fraction of the comparisons -- the cap a single
    tree couldn't pass without simply visiting every leaf."""

    def __init__(self, dim, n_trees=4, leaf_size=64, seed=0):
        self.dim = dim
        self.trees = [HoloTree(dim, leaf_size=leaf_size, seed=seed + t) for t in range(n_trees)]
        self.items = None
        self.last_comparisons = 0
        self._seed = seed                   # remembered for persistence (trees are seed-derived)
        self._leaf_size = leaf_size
        self._n_trees = n_trees

    def build(self, items):
        self.items = np.asarray(items, float)
        for t in self.trees:
            t.build(self.items)
        return self

    def recall(self, query, beam=4, with_agreement=False):
        """Union the candidate leaves from every tree, then cosine-rank once over
        that combined (de-duplicated) candidate set.

        with_agreement=True also returns a reliability score in [0, 1]: the fraction of
        trees whose OWN best candidate equals the forest's chosen index. 1.0 means every
        tree agreed; near 1/n_trees means they split and the answer is a guess -- the
        trees are independently seeded, so their agreement is a free abstention signal
        (act when they agree, hold back when they don't). The default path
        (with_agreement=False) is unchanged, byte-for-byte, including cosine-tie order."""
        query = np.asarray(query, float)
        cand = set()
        per_tree_best = [] if with_agreement else None
        for t in self.trees:
            tcand = set() if with_agreement else None
            for leaf in t._route(query, beam):
                leaf["flux"] += 1
                ids = [int(i) for i in leaf["idx"]]
                cand.update(ids)
                if with_agreement:
                    tcand.update(ids)
            if with_agreement and tcand:                 # this tree's own pick
                tc = np.fromiter(tcand, int)
                per_tree_best.append(int(tc[int((self.items[tc] @ query).argmax())]))
        cand = np.fromiter(cand, int)
        self.last_comparisons = len(cand)
        best = int(cand[int((self.items[cand] @ query).argmax())])
        if not with_agreement:
            return best
        agree = float(np.mean([b == best for b in per_tree_best])) if per_tree_best else 0.0
        return best, agree

    def recall_k(self, query, k=8, beam=4):
        """Return the top-k nearest stored items to `query` as (indices, cosines), ranked over the
        same unioned candidate set recall() uses -- so it stays sub-linear (it ranks only the
        routed candidates, not every item). This is the neighbour-search step that non-local-means
        denoising needs: "find the patches that look like this one." Fewer than k candidates ->
        returns however many were found."""
        query = np.asarray(query, float)
        cand = set()
        for t in self.trees:
            for leaf in t._route(query, beam):
                leaf["flux"] += 1
                cand.update(int(i) for i in leaf["idx"])
        cand = np.fromiter(cand, int)
        self.last_comparisons = len(cand)
        qn = np.linalg.norm(query) + 1e-12
        sims = (self.items[cand] @ query) / (np.linalg.norm(self.items[cand], axis=1) + 1e-12) / qn
        order = np.argsort(sims)[::-1][:k]
        return cand[order], sims[order]

    # -- persistence: the trees are seed-derived, so we save only the items + config
    # and rebuild deterministically (a saved-then-loaded forest recalls identically).
    def to_state(self):
        """Snapshot the forest: its config and the stored items. The random
        hyperplanes are a pure function of (seed, tree index, node id), so reload
        rebuilds byte-identical trees from the items alone."""
        return {
            "kind": "HoloForest",
            "dim": int(self.dim),
            "n_trees": int(self._n_trees),
            "leaf_size": int(self._leaf_size),
            "seed": int(self._seed),
            "items": (self.items.copy() if self.items is not None else None),
        }

    @classmethod
    def from_state(cls, state):
        """Rebuild a HoloForest from a to_state() snapshot, re-growing its trees."""
        f = cls(int(state["dim"]), n_trees=int(state["n_trees"]),
                leaf_size=int(state["leaf_size"]), seed=int(state["seed"]))
        if state.get("items") is not None:
            f.build(np.asarray(state["items"], float))
        return f


# ======================================================================
# the shared structured index
# ======================================================================
def _tile_bucket(coord, tile):
    """The bounded-load bucket a coordinate falls in: floor-divide each axis by `tile`. The cell's address
    IS its tile -- routing by COMPUTATION (the RAM / page-table regime), shared so the floor-divide tiling
    lives in exactly ONE place: StructuredIndex(keying='spatial'), TiledStore, and the splat tiler all call
    this, instead of each re-deriving `gy // tile, gx // tile` on its own."""
    return tuple(int(c // tile) for c in coord)


class StructuredIndex:
    """One content-addressable structured index that the chunkers and the content store can all draw from.

    The route index, a chunked sequence, and the content store were each growing their own version of the
    same idea -- "given a pile of vectors, find the one this query points at, without scanning all of them,
    and at scale." This is that idea, once: a thin, payload-carrying wrapper over the HoloForest random-
    projection tree (above), so a lookup is sub-linear and what comes back is a meaningful LABEL, not a row
    number. New callers reach for this instead of re-growing a fourth near-copy; the existing ones are
    operating points of it (see the two rules and the note at the end).

    TWO RULES ARE BAKED IN, because both were measured the hard way and a future caller would otherwise
    rediscover them as "limits":

      1. KEY ON THE ITEMS THEMSELVES.  A hyperplane tree only routes a query to the right leaf when the query
         RESEMBLES the key it is filed under (query ~= key).  Filing items under a bundle-SUMMARY the query is
         only weakly correlated with mis-routes them: a tile has cosine only ~0.27 to its chunk summary, which
         an exhaustive argmax still resolves but a greedy tree descent does not (measured: locating by chunk
         summary through a tree collapsed to ~1/200).  So file each item under its OWN vector and carry the
         label as payload.

      2. NEVER STORE THE INDEX AS A BUNDLE.  Superposing the keys into one vector and recovering a key by
         unbind + cleanup is decode-via-cleanup, which caps with the number of keys (measured: 200 -> 127 -> 15
         recovered as the set grows).  The index must be a navigable STRUCTURE (this tree) -- or, at small
         scale, an explicit scanned list -- never a superposition.  (Comparing two whole bundles by cosine is a
         different operation -- an evaluation, not a decode -- and does NOT cap; that is the job of the Merkle
         tree below, not of this index.)

    locate() is the sub-linear path (approximate, beam-tunable, with a free abstention signal).  locate_exact()
    is the flat O(n) guaranteed-nearest, for small sets, for when you need exactness, or for weak keys -- it is
    exactly what RouteIndex's flat summary scan already does, so RouteIndex is this index at its small-n
    operating point, and the content store's per-bucket HoloForest is this index at its at-scale one.  Honest
    crossover: the forest carries a large fixed constant (n_trees x leaf_size x beam candidates), so a flat
    scan WINS until the set is in the thousands -- locate_exact is not a fallback, it is the right call below
    the crossover; reach for locate() when the collection is genuinely large.

    For INTEGRITY of a stored set instead of lookup -- "has anything changed, and which item" -- that is a
    different job: use verify_store (holographic_verify), the holographic Merkle tree, not this.

    PLUGGABLE KEYING (the pivot fits the query -- the RAM / page-table lesson).  How a key routes to its
    bounded-load bucket is the ONLY thing that varied across the chunkers and stores this replaces, so it is a
    parameter, not a fork:
      * keying='projection' (default) -- file each key under its OWN content vector in the RP-tree forest and
        rank the routed candidates.  Sub-linear, approximate, with the agreement signal.  The content-recall
        regime; this is the original behaviour, byte-for-byte.
      * keying='hash' -- the page-table / LBA / DHT regime: bucket = stable_hash(label) % nbuckets.  Routing is
        ADDRESS COMPUTATION (zero cosine comparisons), exact by construction.  Keys are hashable LABELS.  This
        is "RAM": you compute where it is, you do not search for it.
      * keying='spatial', tile=T -- the splat-tiler regime: bucket = floor(coord / T) via _tile_bucket.  Keys
        are integer COORDINATE tuples; a bucket holds at most T**ndim items regardless of the total grid.
      * keying='sequential' -- the route-chunker regime: keys are a SEQUENCE of chunks (each a (chunk_size,
        dim) array).  Route two-level by exact scan -- nearest chunk SUMMARY (a bundle of the chunk's tiles),
        then nearest raw tile within that chunk -- returning the (chunk, position) coordinate.  This is the
        sub-linear random-access RouteIndex grew; RouteIndex now delegates its routing here.
    All keyings carry payloads and share locate / locate_exact; only the routing differs.  (The decode-a-bundle
    store -- the splat tiler's own shape -- is TiledStore below, not a keying: it shares this routing but keeps
    its own storage, because bundling is forbidden in an INDEX yet correct in a bounded-load tile.)
    """

    def __init__(self, dim, n_trees=4, leaf_size=64, seed=0, keying="projection", nbuckets=None, tile=1,
                 normalize=True):
        self.dim = dim
        self.seed = seed
        self.keying = keying
        self.last_comparisons = 0          # set by every lookup, for honest cost stats
        self._payloads = None
        if keying == "projection":
            # content recall -- the original behaviour, unchanged byte-for-byte
            self._forest = HoloForest(dim, n_trees=n_trees, leaf_size=leaf_size, seed=seed)
            self._keys = None
            # normalize=True (default) files keys as unit vectors, so routing/exact-scan rank by cosine. Set
            # False to keep keys RAW -- then the tree splits on raw vectors and locate ranks by raw dot product,
            # which makes this index BYTE-IDENTICAL to a bare HoloForest over the same items (so a site that
            # already wraps a raw forest can delegate here without a behaviour change).
            self._normalize = normalize
        elif keying == "hash":
            # page-table / LBA / DHT regime: a stable hash of the label IS the address (zero-comparison route)
            self._nbuckets = nbuckets       # default chosen at build = len(keys) (~1 item/bucket)
            self._buckets = {}              # bucket id -> [(label, item_index), ...]
        elif keying == "spatial":
            # splat-tiler regime: floor-divide the coordinate -- the cell's address IS its tile
            self._tile = tile
            self._buckets = {}              # bucket id -> [(coord, item_index), ...]
        elif keying == "sequential":
            # route-chunker regime: a SEQUENCE of chunks; route by nearest chunk-summary (level 1), then by
            # nearest tile within that chunk (level 2). This is the two-level summary routing RouteIndex grew,
            # shared here so the chunkers route through one fabric instead of a bespoke copy.
            self._chunks = None
            self._summaries = None
        else:
            raise ValueError(f"keying must be 'projection' | 'hash' | 'spatial' | 'sequential', got {keying!r}")

    # -- computed-address routers (the RAM regime: route by COMPUTATION, not by comparison) --------
    @staticmethod
    def _hash_bucket(label, nbuckets):
        # blake2b, NOT Python's salted hash(): the route must be identical run-to-run (the determinism rule).
        # Python's hash() is randomised per process, which would reshuffle every bucket and break reproducibility.
        return int(hashlib.blake2b(repr(label).encode(), digest_size=8).hexdigest(), 16) % nbuckets

    def _spatial_bucket(self, coord):
        return _tile_bucket(coord, self._tile)

    def build(self, keys, payloads=None):
        """keys -- what each item is filed under, per the active keying:
             projection : an (N, dim) array of CONTENT vectors (file each item under ITSELF, rule 1).
             hash       : a list of hashable LABELS (the address is computed from the label).
             spatial    : a list of integer COORDINATE tuples (the address is the floor-divided cell).
        payloads -- one label per key, returned by locate ((chunk, step) for a route, a URI for the store, ...);
        defaults to the integer position, so a bare build is a plain index."""
        n = len(keys)
        self._payloads = list(payloads) if payloads is not None else list(range(n))
        if len(self._payloads) != n:
            raise ValueError("payloads must have exactly one entry per key")
        if self.keying == "projection":
            K = np.asarray(keys, float)
            if K.ndim != 2:
                raise ValueError("projection keys must be a 2-D array of shape (N, dim)")
            # unit keys -> cosine == dot, so routing and the exact scan agree and stay numerically stable.
            # normalize=False keeps them raw -> identical to a bare HoloForest over the same items.
            if self._normalize:
                K = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-12)
            self._keys = K
            self._forest.build(K)
        elif self.keying == "hash":
            nb = self._nbuckets if self._nbuckets else max(1, n)
            self._nbuckets = nb
            for i, label in enumerate(keys):
                self._buckets.setdefault(self._hash_bucket(label, nb), []).append((label, i))
        elif self.keying == "spatial":
            for i, coord in enumerate(keys):
                self._buckets.setdefault(self._spatial_bucket(coord), []).append((tuple(coord), i))
        elif self.keying == "sequential":
            # keys are the chunks (each an (chunk_size, dim) array of tile vectors). Build one normalised
            # SUMMARY per chunk -- the same normalising bundle RouteIndex used, replicated byte-for-byte -- and
            # keep the tiles RAW (level-2 ranks raw tiles against the query, exactly as RouteIndex does, so no
            # normalisation drift creeps in). Payloads are unused here: a sequential lookup returns a (chunk,
            # position) coordinate, which IS the answer, not a stored label.
            from holographic.agents_and_reasoning.holographic_ai import bundle
            self._chunks = [np.asarray(ch, float) for ch in keys]
            summaries = []
            for ch in self._chunks:
                s = bundle(list(ch))                                       # one summary vector per chunk
                nrm = np.linalg.norm(s)
                summaries.append(s / nrm if nrm > 0 else s)
            self._summaries = np.array(summaries) if summaries else np.zeros((0,))
        return self

    def locate(self, query, beam=4, with_agreement=False):
        """Find the item `query` points at and return its payload; sets last_comparisons (the honest cost).

        projection : route `query` through the forest, rank the routed candidates -- sub-linear, approximate;
                     with_agreement returns the trees' agreement in [0, 1] (a free "am I sure?" signal).
        hash/spatial : COMPUTE the bucket, then exact-match within it -- the RAM regime, ~O(1), and the answer
                     cannot be a wrong neighbour (a computed address is exact); with_agreement returns 1.0 on a
                     hit and 0.0 on a miss (a computed address is either right or simply absent)."""
        if not self._payloads:
            return (None, 0, 0.0) if with_agreement else (None, 0)
        if self.keying == "projection":
            if with_agreement:
                i, agree = self._forest.recall(query, beam=beam, with_agreement=True)
                self.last_comparisons = self._forest.last_comparisons
                return self._payloads[i], self.last_comparisons, agree
            i = self._forest.recall(query, beam=beam)
            self.last_comparisons = self._forest.last_comparisons
            return self._payloads[i], self.last_comparisons
        if self.keying == "sequential":
            # two-level: nearest chunk SUMMARY (level 1), then nearest tile within that chunk (level 2). Both
            # exact argmax scans -- a tile is only weakly correlated with its summary (rule 1), so an argmax
            # resolves it where a greedy tree descent would not. Returns the (chunk, position) coordinate the
            # query routes to; with_agreement returns 1.0 (an exact scan is certain) / handles the empty route.
            if self._summaries is None or len(self._summaries) == 0:
                return ((-1, -1), 0, 0.0) if with_agreement else ((-1, -1), 0)
            q = np.asarray(query, float)
            nq = np.linalg.norm(q)
            q = q / nq if nq > 0 else q
            c = int(np.argmax(self._summaries @ q))                        # level 1: nearest chunk summary
            ch = self._chunks[c]
            pos = int(np.argmax(ch @ q))                                   # level 2: nearest raw tile within it
            self.last_comparisons = len(self._summaries) + len(ch)
            if with_agreement:
                return (c, pos), self.last_comparisons, 1.0
            return (c, pos), self.last_comparisons
        # hash / spatial: compute the address, then scan only that (tiny, bounded) bucket for the exact key
        if self.keying == "hash":
            bucket, target = self._hash_bucket(query, self._nbuckets), query
        else:
            bucket, target = self._spatial_bucket(query), tuple(query)
        comps = 0
        for (k, i) in self._buckets.get(bucket, []):
            comps += 1
            if k == target:
                self.last_comparisons = comps
                return (self._payloads[i], comps, 1.0) if with_agreement else (self._payloads[i], comps)
        self.last_comparisons = comps
        return (None, comps, 0.0) if with_agreement else (None, comps)

    def locate_k(self, query, k=8, beam=4):
        """The k nearest items as a list of (payload, cosine), still sub-linear (it ranks only the routed
        candidates, not every item).  CONTENT search -- defined only for keying='projection' (hash and spatial
        are exact-address, not nearest-neighbour, so a k-NN query is undefined for them)."""
        if self.keying != "projection":
            raise ValueError("locate_k is a nearest-neighbour query; only keying='projection' supports it")
        idxs, sims = self._forest.recall_k(query, k=k, beam=beam)
        self.last_comparisons = self._forest.last_comparisons
        return [(self._payloads[int(i)], float(s)) for i, s in zip(idxs, sims)]

    def locate_exact(self, query):
        """Guaranteed-correct lookup with no routing.
        projection : flat O(n) argmax -- the right call below the forest's crossover, when you need EXACTNESS,
                     or for weak keys.  This is what RouteIndex's flat scan does.
        hash/spatial : a full linear scan for the exact key -- the reference the computed-address locate is
                     checked against (locate is already exact for these, so this is verification / fallback)."""
        if not self._payloads:
            return None, 0
        if self.keying == "sequential":
            return self.locate(query)          # the sequential locate is already an exact two-level scan
        if self.keying == "projection":
            if self._keys is None or len(self._keys) == 0:
                return None, 0
            q = np.asarray(query, float)
            self.last_comparisons = len(self._keys)
            return self._payloads[int((self._keys @ q).argmax())], self.last_comparisons
        # hash / spatial: scan every bucket for the exact key (guaranteed correct, O(n))
        target = query if self.keying == "hash" else tuple(query)
        n = 0
        for members in self._buckets.values():
            for (k, i) in members:
                n += 1
                if k == target:
                    self.last_comparisons = n
                    return self._payloads[i], n
        self.last_comparisons = n
        return None, n

    def __len__(self):
        return len(self._payloads) if self._payloads else 0


class TiledStore:
    """A spatially-tiled accumulator of vectors -- the splat-tiler's core, generalised and shared.

    Where StructuredIndex FINDS an item (explicit keys; its rule 2 forbids bundling the index), this STORES a
    field as bounded per-tile BUNDLES and reads a region back by DECODE. Bundling is legitimate HERE -- and
    only here -- because floor-divide routing caps each tile at <= tile**ndim cells no matter how fine the
    TOTAL grid, so the decode-via-cleanup capacity cliff never bites at any resolution. That is exactly the
    lesson splat_bundle_tiled measured ("per-bundle load fixed, recall ~100% at any resolution"); it now lives
    once, here, and the splat tiler delegates to it.

    This owns the ROUTING (the same _tile_bucket as the spatial index) and the bounded GROUPING -- nothing
    about the encode/decode. The caller bundles a tile's group with its OWN `bundle` and decodes with its OWN
    codebook, so the store stays representation-agnostic (splat occupancy today; any per-cell vector tomorrow).
    Two axes meet cleanly: routing is shared with StructuredIndex; storage (find-by-key vs decode-a-bundle) is
    the one thing that differs, which is why this is a sibling class, not a flag on the index."""

    def __init__(self, tile, dim):
        self.tile = tile
        self.dim = dim
        self._groups = {}                   # bucket id -> [vector, ...]  (a LIST, so the caller bundles it)

    def bucket_of(self, cell):
        """The tile a cell routes to -- COMPUTATION, no search (the RAM regime)."""
        return _tile_bucket(cell, self.tile)

    def add(self, cell, vector):
        """Group `vector` under cell's tile (bounded load by construction). Returns self for chaining."""
        self._groups.setdefault(self.bucket_of(cell), []).append(vector)
        return self

    def group(self, cell):
        """The list of vectors in cell's tile (empty if none) -- the caller bundles this and decodes it."""
        return self._groups.get(self.bucket_of(cell), [])

    def groups(self):
        """All tiles as {bucket id -> [vectors]} -- for building the per-tile bundles once, at the end."""
        return self._groups


# ======================================================================
# benchmarks
# ======================================================================
def capacity_curve(Ns, dim=2048, leaf_size=64, seed=0, probes=None):
    """Recall@1 of a flat memory vs a HoloTree as the dataset grows.  This is the
    headline: flat collapses past capacity, the tree holds.  `probes` optionally
    subsamples the recall estimate for speed."""
    rows = []
    for N in Ns:
        rng = np.random.default_rng(seed)
        keys = np.stack([random_vector(dim, rng) for _ in range(N)])
        vals = np.stack([random_vector(dim, rng) for _ in range(N)])
        probe_idx = range(N) if not probes else rng.choice(N, min(probes, N), replace=False)
        flat = HolographicMemory(dim)
        for k, v in zip(keys, vals):
            flat.learn(k, v)
        f_ok = sum(int((vals @ flat.recall(keys[i])).argmax() == i) for i in probe_idx)
        tree = HoloTree(dim, leaf_size=leaf_size, seed=seed).build(keys, vals)
        t_ok = sum(int(tree.recall(keys[i]) == i) for i in probe_idx)
        n = len(list(probe_idx)) if probes else N
        rows.append(dict(N=N, flat=f_ok / n, tree=t_ok / n, leaves=tree.stats()["leaves"],
                         depth=tree.stats()["depth"]))
    return rows


def nn_benchmark(N=2000, dim=512, leaf_size=64, beam=8, noise=0.5, seed=0):
    """Approximate nearest-neighbour search on noisy cues: exact O(N) scan vs the
    tree.  Reports recall@1 and how many comparisons each did.  `noise` adds a
    unit-vector perturbation of that magnitude (0.5 -> cue cosine ~0.9 to truth)."""
    rng = np.random.default_rng(seed)
    items = np.stack([random_vector(dim, rng) for _ in range(N)])
    tree = HoloTree(dim, leaf_size=leaf_size, seed=seed).build(items)
    q_ok = t_ok = t_cmp = 0
    T = 300
    for _ in range(T):
        i = int(rng.integers(N))
        q = items[i] + noise * random_vector(dim, rng)
        q = q / np.linalg.norm(q)
        q_ok += int((items @ q).argmax() == i)          # exact scan, N comparisons
        t_ok += int(tree.recall(q, beam=beam) == i); t_cmp += tree.last_comparisons
    return dict(N=N, dim=dim, beam=beam, exact_recall=q_ok / T, exact_cmp=N,
                tree_recall=t_ok / T, tree_cmp=round(t_cmp / T))


def forest_benchmark(N=2000, dim=512, leaf_size=64, n_trees=4, beam=4, noise=0.5, seed=0):
    """Approximate NN with a forest: recall and comparisons vs an exact scan."""
    rng = np.random.default_rng(seed)
    items = np.stack([random_vector(dim, rng) for _ in range(N)])
    forest = HoloForest(dim, n_trees=n_trees, leaf_size=leaf_size, seed=seed).build(items)
    ok = cmp = exact = 0; T = 300
    for _ in range(T):
        i = int(rng.integers(N)); q = items[i] + noise * random_vector(dim, rng); q = q / np.linalg.norm(q)
        exact += int((items @ q).argmax() == i)
        ok += int(forest.recall(q, beam=beam) == i); cmp += forest.last_comparisons
    return dict(N=N, n_trees=n_trees, beam=beam, exact_recall=exact / T, exact_cmp=N,
                forest_recall=ok / T, forest_cmp=round(cmp / T))


# ======================================================================
# demo
# ======================================================================

def _demo():
    print("holographic_tree -- a deterministic recursive memory\n" + "-" * 52)
    print("capacity: recall@1 as the dataset grows (dim=2048, leaf=64)")
    print(f"  {'N':>6} {'flat':>7} {'tree':>7}  leaves")
    for r in capacity_curve([64, 256, 1024, 4096], dim=2048, leaf_size=64):
        print(f"  {r['N']:>6} {r['flat']:>7.0%} {r['tree']:>7.0%}  {r['leaves']}")
    print("\napproximate nearest-neighbour on noisy cues (exact scan vs tree)")
    b = nn_benchmark(N=2000, dim=512, leaf_size=64, beam=8)
    print(f"  exact: recall {b['exact_recall']:.0%}  ({b['exact_cmp']} comparisons/query)")
    print(f"  tree : recall {b['tree_recall']:.0%}  ({b['tree_cmp']} comparisons/query, "
          f"{b['exact_cmp'] / b['tree_cmp']:.0f}x fewer)")
    f = forest_benchmark(N=2000, dim=512, leaf_size=64, n_trees=4, beam=4)
    print(f"  forest (4 trees): recall {f['forest_recall']:.0%}  ({f['forest_cmp']} comparisons/query, "
          f"{f['exact_cmp'] / f['forest_cmp']:.1f}x fewer) -- past the single tree's ceiling")
    t = HoloTree(512, leaf_size=64, seed=0).build(np.stack([random_vector(512, np.random.default_rng(i)) for i in range(2000)]))
    print(f"\ntree shape: {t.stats()}")


def _selftest():
    """Regression trap for the sub-linear index (T6 backfill; it had a demo but no assertion). Pins the two
    contracts callers rely on: an approximate tree/forest still RECALLS the right item, and it does so touching
    FEWER than all candidates (the whole reason a tree exists over a flat scan). Uses the module's own
    nn_benchmark at a scale where sub-linearity is visible -- numbers measured against the live code first."""
    import numpy as np

    # 1. SUB-LINEARITY with high recall: at N=3000 the tree touches far fewer than 3000 candidates while still
    #    finding the true nearest neighbour most of the time. The exact scan is the honest baseline it beats.
    r = nn_benchmark(N=3000, dim=512, seed=0)
    assert r["exact_recall"] == 1.0                          # the baseline is exhaustive, so exact
    assert r["tree_recall"] > 0.7                            # approximate, but reliably finds the target
    assert r["tree_cmp"] < r["exact_cmp"] / 2               # and does it touching <half the candidates (measured ~375/3000)

    # 2. DIRECT recall: a forest returns the INDEX of the stored item nearest a noisy cue -- the [BLIND-SPOT]
    #    point is a NOISY query (not the exact stored vector, which any scheme trivially matches).
    rng = np.random.default_rng(0)
    items = rng.standard_normal((300, 512))
    forest = HoloForest(dim=512, seed=0).build(items)
    hits = sum(forest.recall(items[i] + 0.25 * rng.standard_normal(512)) == i for i in range(0, 300, 10))
    assert hits >= 25                                        # of 30 noisy probes, the vast majority land right

    print("OK: holographic_tree self-test passed (forest recalls the right item under noise >=25/30, and the tree "
          "index touches <half the candidates an exhaustive scan would at N=3000 while keeping recall>0.7)")


if __name__ == "__main__":
    import sys
    _selftest()                                     # fast: the contract check the CI walker runs
    if "--demos" in sys.argv:                         # the benchmarks print and are slow -- opt-in
        _demo()

# ---------------------------------------------------------------------------
# ReflexCache: the slime-mould fast path in front of ANY index
# (moved here from holographic_navigator: the reflex fronts index machinery,
#  not navigators -- _Index in holographic_mind uses it too)
# ---------------------------------------------------------------------------

class ReflexCache:
    """A use-reinforced fast path -- the navigator's habits.

    Real query streams are skewed: a handful of items get most of the traffic.
    Slime mould handles exactly this by thickening the veins it travels often and
    letting unused ones wither; the engine's ReflexArc is the same idea (a
    familiar input skips the expensive path). So the navigator keeps a small HOT
    SET of the items it commits to most, and checks them FIRST, before descending
    the tree. If one clearly matches the cue it is recognised instantly -- a
    reflex -- at the cost of a tiny scan instead of a full search.

    Two pieces of self-regulation keep this honest:
      * reinforcement -- every committed item's count goes up, and the hot set is
        periodically rebuilt from the most-committed items (veins thicken with use);
      * a flux guard -- after a warm-up, if the hot set is rarely the answer
        (a uniform, unpredictable stream), it stops scanning it altogether, so the
        habit never costs more than it saves. Veins nobody travels are pruned.
    """

    def __init__(self, n_items, hot_size=48, gate=0.55, margin=0.08, warmup=600, period=200):
        self.n_items = n_items
        self.hot_size = hot_size
        self.gate = gate          # a hot item must match the cue at least this well
        self.margin = margin      # ...and lead the runner-up by at least this much
        self.warmup = warmup
        self.period = period
        self.counts = np.zeros(n_items)
        self.hot = np.array([], dtype=int)
        self.recent_hits = []
        self.active = True
        self.t = 0

    def consider(self, cue, items):
        """Return (item_index, comparisons) if a hot item clearly matches, else
        (None, comparisons_spent_looking)."""
        if not (self.active and len(self.hot)):
            return None, 0
        sims = items[self.hot] @ cue
        order = np.sort(sims)[::-1]
        runner = order[1] if len(order) > 1 else -2.0
        if order[0] > self.gate and (order[0] - runner) > self.margin:
            return int(self.hot[int(sims.argmax())]), len(self.hot)
        return None, len(self.hot)

    def reinforce(self, item_index, was_hit, items):
        """Record a commitment, and periodically rebuild + flux-guard the veins."""
        self.t += 1
        self.counts[item_index] += 1
        self.recent_hits.append(int(was_hit))
        if self.t % self.period == 0:
            self.hot = np.argsort(self.counts)[::-1][:self.hot_size]
            if self.t >= self.warmup and len(self.recent_hits) >= self.period:
                # keep the habit only if it is actually paying off
                self.active = float(np.mean(self.recent_hits[-self.period:])) > 0.08

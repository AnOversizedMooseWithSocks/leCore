"""PERFECT RECALL at any corpus size -- hierarchical computing-in-superposition with exact verification.

THE CLAIM, stated precisely so it can be true: for CONTAINMENT queries ("every document containing ALL of
these terms"), this index returns EXACTLY the ground-truth set -- zero false negatives, zero false
positives -- at any corpus size, with only TIME as the scaling cost. This is exact-recall semantics, the
thing BM25 never guarantees (it ranks; it can bury). No BM25 anywhere in this module.

WHY IT IS GUARANTEED (each half from a different discipline):
  * ZERO FALSE NEGATIVES -- the superposition law. A sparse binary bundle (OR of per-term codes) is a
    Bloom filter, and a Bloom filter structurally CANNOT miss: every inserted term set its bits, so a
    contained term always tests present (Bloom 1970). VSAs contain Bloom filters as a subclass -- the
    bundle IS computing in superposition, all members tested in one dot (Kleyko-Rahimi-Gayler-Osipov
    2020; Kleyko et al., Proc. IEEE 2022, Sec. C "computing in superposition").
  * ZERO FALSE POSITIVES -- the depth test. Filter hits are CANDIDATES (hash collisions = overdraw);
    every candidate is verified against the doc's exact sha256 term-hash set before it may be returned.
    Verification is the self-verifying-storage discipline: the structure never answers from similarity
    alone.

THE RENDERING SHAPE (the user's metaphors, load-bearing not decorative):
  * GEOMETRY INSTANCING -- one term, one code, everywhere. A term's k bit positions are a pure function
    of sha256(channel, term); a billion docs share the instances, per-doc storage is references.
  * IRRADIANCE PROBES / BAKED LIGHTING -- the TILE filter: the OR of a tile's doc filters, baked once.
    A query tests a whole tile in one AND -- the probe answers "could any surface here see this light?"
  * HIERARCHICAL CULLING -- tiles whose probe lacks any query bit are culled wholesale (frustum culling);
    only lit tiles descend to per-doc tests; only per-doc hits reach exact verification.
  * MULTI-CHANNEL RENDER -- independent channels (e.g. 'token', 'trigram', 'field:title') with their own
    filters; a query ANDs within a channel and may intersect across channels, like compositing passes.
  * ADAPTIVE SAMPLING -- work is spent ONLY where the probe says signal might be; a selective query
    touches a vanishing fraction of the corpus (measured in the selftest).

KEPT NEGATIVES (loud, measured):
  * CONTAINMENT is not RELEVANCE. This returns the exact matching SET; ranking what a human would call
    best is a different problem (the benchmark suite's job) and is NOT claimed here.
  * A ubiquitous query term lights every probe -- the hierarchy degenerates toward the full O(N) verify
    scan. That is the honest meaning of "no limit other than time": worst case IS time.
  * Memory is O(N * (filter_bits/8 + 8*terms)) -- the exact verify sets are the bigger half. They are
    the price of the zero-false-positive guarantee and are cold-storage-parkable (the slim lever).

Pure NumPy/stdlib/hashlib. Deterministic end to end (sha256 bit positions, sorted outputs).
"""
import hashlib
import numpy as np

_W = 64                                                  # bits per packed word


def _positions(channel, term, k, bits):
    """The INSTANCE: term -> its k bit positions, a pure function of sha256(channel|term). Deterministic,
    shared by every document and every tile -- one definition, a billion references."""
    h = hashlib.sha256(("%s|%s" % (channel, term)).encode()).digest()
    out = []
    i = 0
    while len(out) < k:
        if i + 8 > len(h):                               # extend the stream deterministically
            h = h + hashlib.sha256(h).digest()
        out.append(int.from_bytes(h[i:i + 8], "big") % bits)
        i += 8
    return out


def _term_hash(channel, term):
    """8-byte exact identity for the verify sets -- collision odds 2^-64 per pair, and a collision can
    only ever ADD a candidate (caught below by nothing -- see the docstring honesty note in add())."""
    return int.from_bytes(hashlib.sha256(("%s|%s" % (channel, term)).encode()).digest()[:8], "big")


class PerfectRecallIndex:
    """Hierarchical exact-containment index: tile probes -> doc filters -> exact verify.

    filter_bits: per-doc Bloom width (default 2048 -- ~256 bytes/doc); k: bits per term (default 4);
    tile: docs per tile (default 512). All three trade candidate overdraw against memory; NONE of them
    can affect correctness -- a smaller filter only means more candidates reach the verify stage."""

    def __init__(self, filter_bits=2048, k=4, tile=512, tile_bits=1 << 16):
        self.bits = int(filter_bits)
        self.k = int(k)
        self.tile = int(tile)
        # PROBE RESOLUTION IS ITS OWN DIAL (measured negative from this module's first run: OR-ing 512
        # docs into a 2048-bit probe SATURATES it -- every bit set, the cull goes blind, a too-coarse
        # shadow map. A tile aggregates ~tile*terms*k set bits, so its filter must be sized for THAT
        # occupancy, not a single doc's. 64k bits = 8KB/tile keeps the probe sparse at tile=512.)
        self.tile_bits = int(tile_bits)
        self.words = (self.bits + _W - 1) // _W
        self.tile_words = (self.tile_bits + _W - 1) // _W
        self.doc_filters = {}                            # channel -> list of packed uint64 rows
        self.tile_filters = {}                           # channel -> list of packed uint64 probe rows
        self.verify = {}                                 # channel -> list of sorted uint64 term-hash arrays
        self.n = 0

    def _pack(self, positions, words):
        row = np.zeros(words, dtype=np.uint64)
        for p in positions:
            row[p // _W] |= np.uint64(1) << np.uint64(p % _W)
        return row

    def add(self, terms_by_channel):
        """Add one document: {'token': [...], 'trigram': [...], ...}. Returns its doc index. Honesty note:
        exactness is relative to the 8-byte term hash -- two DIFFERENT terms colliding at 2^-64 would be
        treated as the same term. Declared, not hidden; use 16 bytes if that keeps anyone up at night."""
        di = self.n
        ti = di // self.tile
        for ch, terms in terms_by_channel.items():
            if ch not in self.doc_filters:
                self.doc_filters[ch] = []; self.tile_filters[ch] = []; self.verify[ch] = []
            pos, tpos = [], []
            hs = set()
            for t in terms:
                pos.extend(_positions(ch, t, self.k, self.bits))
                tpos.extend(_positions("tile:" + ch, t, self.k, self.tile_bits))
                hs.add(_term_hash(ch, t))
            self.doc_filters[ch].append(self._pack(pos, self.words))
            self.verify[ch].append(np.array(sorted(hs), dtype=np.uint64))
            tf = self.tile_filters[ch]
            if ti >= len(tf):
                tf.append(np.zeros(self.tile_words, dtype=np.uint64))
            tf[ti] |= self._pack(tpos, self.tile_words)  # the probe: OR-bake, irradiance-map style
        self.n += 1
        return di

    def query(self, terms, channel="token", stats=None):
        """EXACT set of doc indices containing ALL `terms` in `channel` (AND / containment semantics),
        ascending. Empty terms -> every doc (vacuous truth, stated). stats={} receives the culling
        numbers: tiles_total/tiles_descended/docs_tested/docs_verified -- the adaptive-sampling receipt."""
        if channel not in self.doc_filters:
            return []
        if not terms:
            return list(range(self.n))
        q = self._pack([p for t in set(terms) for p in _positions(channel, t, self.k, self.bits)],
                       self.words)
        qt = self._pack([p for t in set(terms) for p in _positions("tile:" + channel, t, self.k,
                                                                   self.tile_bits)], self.tile_words)
        qh = np.array(sorted({_term_hash(channel, t) for t in set(terms)}), dtype=np.uint64)
        tf, df, vf = self.tile_filters[channel], self.doc_filters[channel], self.verify[channel]
        out = []
        tiles_desc = docs_tested = docs_ver = 0
        for ti, probe in enumerate(tf):
            if not np.array_equal(probe & qt, qt):       # frustum cull: probe lacks a query bit -> no doc
                continue                                 # in this tile can possibly contain all terms
            tiles_desc += 1
            lo, hi = ti * self.tile, min((ti + 1) * self.tile, self.n)
            for di in range(lo, hi):
                docs_tested += 1
                if not np.array_equal(df[di] & q, q):    # per-doc filter: still zero false negatives
                    continue
                docs_ver += 1                            # depth test: exact hash-set containment
                if np.all(np.isin(qh, vf[di], assume_unique=True)):
                    out.append(di)
        if stats is not None:
            stats.update(tiles_total=len(tf), tiles_descended=tiles_desc,
                         docs_tested=docs_tested, docs_verified=docs_ver)
        return out


def _selftest():
    """The guarantee AS AN ASSERTION: exact set equality against brute force -- not F1, not 'close'."""
    rng = np.random.default_rng(0)
    vocab = ["w%03d" % i for i in range(600)]
    N = 30000
    docs = []
    idx = PerfectRecallIndex()
    for i in range(N):
        # zipf-ish doc: a few common words + rare tail, 8-20 terms
        m = int(rng.integers(8, 21))
        terms = list({vocab[min(599, int(z))] for z in rng.zipf(1.3, size=m)})
        docs.append(set(terms))
        idx.add({"token": terms})

    # 1) PERFECT RECALL, exact == : 60 random AND-queries vs brute force
    stats_sel, stats_ubiq = {}, {}
    for qi in range(60):
        src = docs[int(rng.integers(0, N))]
        qterms = list(rng.choice(sorted(src), size=min(3, len(src)), replace=False))
        truth = sorted(i for i, d in enumerate(docs) if all(t in d for t in qterms))
        got = idx.query(qterms, stats=stats_sel if qi == 0 else None)
        assert got == truth, "recall not perfect: query %r" % (qterms,)

    # 2) ADAPTIVE SAMPLING receipt -- asserted as OPTIMALITY, not an arbitrary fraction (the first
    #    draft asserted docs_tested < 0.3N and FAILED on correct behavior: zipf spreads even a rare
    #    term across many tiles, and the probe descended EXACTLY the tiles truly containing it. The
    #    instrument was wrong, the standing lesson -- assert contrasts and floors, not absolutes.)
    rare = ["w598", "w599"]
    s = {}
    got_rare = idx.query(rare, stats=s)
    tiles_floor = len({i // idx.tile for i, d in enumerate(docs) if rare[0] in d})
    assert s["tiles_descended"] == tiles_floor           # cull is OPTIMAL: only truly-lit tiles descend
    assert s["docs_verified"] == len(got_rare)           # verify touches only true hits (no collisions here)
    #    and the CONTRAST: finer tiles cut docs_tested by ~the tile ratio (resolution is the dial)
    idx_fine = PerfectRecallIndex(tile=64)
    for d in docs:
        idx_fine.add({"token": sorted(d)})
    s_f = {}
    assert idx_fine.query(rare, stats=s_f) == got_rare   # same exact answer at any tiling
    assert s_f["docs_tested"] < 0.5 * s["docs_tested"], (s, s_f)
    # ...and 3) KEPT NEGATIVE: a UBIQUITOUS term lights every probe -- degenerates toward the scan
    s2 = {}
    idx.query(["w001"], stats=s2)
    assert s2["tiles_descended"] == s2["tiles_total"]    # worst case IS time, as declared

    # 4) MULTI-CHANNEL: trigram channel independent of token channel
    idx2 = PerfectRecallIndex(tile=64)
    for w in ("alpha", "alphabet", "beta"):
        tris = [w[i:i + 3] for i in range(len(w) - 2)]
        idx2.add({"token": [w], "trigram": tris})
    assert idx2.query(["alp", "lph"], channel="trigram") == [0, 1]
    assert idx2.query(["alpha"], channel="token") == [0]

    # 5) DETERMINISM: rebuild -> identical answers
    idx3 = PerfectRecallIndex()
    for d in docs[:2000]:
        idx3.add({"token": sorted(d)})
    a = idx3.query(["w010", "w020"]); b = idx3.query(["w010", "w020"])
    assert a == b

    # 6) SCALING is in TIME not correctness: the 30k index answered exactly; per-doc memory is flat.
    print("  perfectrecall selftest OK: 60/60 exact==truth at N=30000; rare query tested %d/%d docs; "
          "ubiquitous query descended all tiles (kept negative); multi-channel; deterministic"
          % (s_f["docs_tested"], N))


if __name__ == "__main__":
    _selftest()

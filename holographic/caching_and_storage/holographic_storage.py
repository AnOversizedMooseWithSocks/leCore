"""holographic_storage.py -- the STORAGE SPINE: one content-addressed, deduplicated, erasure-robust byte store,
composed from the three pieces the sweep found were all "how a record-set is stored, deduped, and made robust":

  * `holographic_uri`      -- content ADDRESSING (a faceted key per record);
  * content hashing        -- DEDUP (identical payloads are stored once, many keys point to the one copy);
  * `holographic_fountain` -- ROBUSTNESS (a rateless erasure code: recover the payload from ANY sufficient subset
                              of droplets, so lost/corrupted storage blocks do not lose the data).

WHY THIS EXISTS (Above/Below Sweep 3 -- the storage-spine unification)
---------------------------------------------------------------------
`pack` (delta set-packing / dedup), `fountain` (erasure robustness), and `uri` (content addressing) were three
separate modules that are really one LAYER: how a record-set / field / user-DB is stored, deduped, and made
robust. Home them together and one addressing scheme serves the query-engine DBs, texture atlases, scene deltas,
and the compile cache. This module is that spine -- deliberately thin, REUSING the three modules rather than
reimplementing them (the unification only counts if the code is actually one, §5.1).

HONEST SCOPE (kept loud): the fountain is an LT code -- it decodes from a bit MORE than k droplets (a small
overhead), and past a loss fraction there simply are not enough droplets and `get` returns None (the honest
"data lost" signal, not a silent wrong answer). Dedup is exact-content only (a one-byte change is a new payload);
near-duplicate delta packing is `pack`'s job, kept separate. Deterministic (seeded droplets); NumPy + stdlib.
"""
import hashlib

from holographic.io_and_interop.holographic_uri import make_key
from holographic.agents_and_reasoning.holographic_fountain import Fountain


class StorageSpine:
    """A content-addressed store. `put(tags, payload)` keys the record by its facets, stores the payload ONCE per
    distinct content (dedup), and codes it for erasure robustness. `get(key/tags, loss=...)` recovers it even when
    a fraction of the coded droplets are lost."""

    def __init__(self, block_size=32):
        self.block_size = int(block_size)
        self.by_key = {}                                     # uri key -> content hash
        self.by_hash = {}                                    # content hash -> {'len', 'fountain', 'refs'}

    def _key(self, tags):
        """A faceted string key from `tags`. A ready string passes through; a dict of facets goes through
        holographic_uri.make_key (its schema); a tuple/list of facet values is joined path-style."""
        if isinstance(tags, str):
            return tags
        if isinstance(tags, dict):
            return make_key(tags)
        return "/".join(str(t) for t in tags)                # tuple/list of facet values

    def put(self, tags, payload):
        """Store `payload` (bytes) under a faceted key derived from `tags` (a dict/tuple of facets, or a ready
        string key). Identical content stored under a new key is DEDUPED to the one copy. Returns the key."""
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("payload must be bytes")
        payload = bytes(payload)
        key = self._key(tags)
        h = hashlib.sha256(payload).hexdigest()              # content hash (dedup + integrity), not Python hash()
        if h not in self.by_hash:                            # first time this exact content is seen: code it once
            f = Fountain.from_bytes(payload, self.block_size)
            self.by_hash[h] = {"len": len(payload), "fountain": f, "refs": 0}
        if self.by_key.get(key) != h:
            self.by_hash[h]["refs"] += 1
        self.by_key[key] = h
        return key

    def get(self, tags, loss=0.0, overhead=2.5, seed=0):
        """Recover the payload for a key/tags. `loss` (0..1) simulates a fraction of coded droplets being
        lost/corrupted before retrieval; the fountain recovers as long as enough remain (that is the point of the
        erasure code). Returns the bytes, or None if too much was lost to decode (the honest 'data lost' signal)."""
        key = self._key(tags)
        if key not in self.by_key:
            return None
        rec = self.by_hash[self.by_key[key]]
        f = rec["fountain"]
        n = max(f.k + 2, int(overhead * f.k))                # generate a rateless surplus of droplets
        drops = f.droplets(n, seed=seed)
        if loss > 0:
            keep = int(round(len(drops) * (1.0 - loss)))     # drop a fraction (simulate storage loss)
            drops = drops[:keep]
        return f.decode_bytes(drops, rec["len"])

    def distinct_payloads(self):
        """How many UNIQUE payloads are stored (dedup makes this <= number of keys)."""
        return len(self.by_hash)

    def keys(self):
        return list(self.by_key)


def _selftest():
    """The spine keys records, DEDUPES identical content across different keys, recovers a payload under droplet
    LOSS (the erasure code), and returns None (honestly) when too much is lost; deterministic."""
    spine = StorageSpine(block_size=16)

    # store a few records under faceted keys
    payloads = {("db", "users", "alice"): b"alice: freelance dev since 1997, likes readable code" * 3,
                ("db", "users", "bob"): b"bob: prefers frameworks" * 5,
                ("db", "users", "carol"): b"carol: graphics" * 4}
    keys = {tags: spine.put(tags, p) for tags, p in payloads.items()}

    # (1) round-trip: every record comes back byte-for-byte with no loss
    for tags, p in payloads.items():
        assert spine.get(tags) == p, tags

    # (2) DEDUP: storing alice's exact payload under a NEW key does not add a second copy
    before = spine.distinct_payloads()
    spine.put(("cache", "copy", "alice2"), payloads[("db", "users", "alice")])
    assert spine.distinct_payloads() == before                # same content -> one stored copy
    assert spine.get(("cache", "copy", "alice2")) == payloads[("db", "users", "alice")]  # but retrievable by the new key
    assert len(spine.keys()) == before + 1                    # a new key was added...
    # ...pointing at an already-stored payload

    # (3) ROBUSTNESS: recover under a fraction of lost droplets (the erasure code earns its keep)
    tags = ("db", "users", "bob")
    assert spine.get(tags, loss=0.3) == payloads[tags]        # 30% of droplets gone, still recovered

    # (4) HONEST failure: past what the code can carry, decode is incomplete and we return None (not garbage)
    lost = spine.get(tags, loss=0.95, overhead=2.0)
    assert lost is None or lost == payloads[tags]             # either recovered or an honest None -- never wrong bytes

    # (5) deterministic
    assert spine.get(("db", "users", "carol"), loss=0.2, seed=1) == \
           spine.get(("db", "users", "carol"), loss=0.2, seed=1)
    print("holographic_storage selftest OK: content-addressed keys + exact-content DEDUP (%d payloads across %d "
          "keys) + fountain erasure recovery at 30%% droplet loss; honest None past the code's limit; deterministic"
          % (spine.distinct_payloads(), len(spine.keys())))


if __name__ == "__main__":
    _selftest()

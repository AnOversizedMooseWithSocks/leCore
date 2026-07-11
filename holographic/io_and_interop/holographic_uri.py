"""
holographic_uri.py -- addresses, not folders.

The partitioning idea from the very start of this project, finally in its proper
form.  AWS S3 has no real folders: it is a *flat namespace* of objects whose keys
happen to contain '/' delimiters, and the "folders" you see are just keys that
share a prefix.  Listing with a delimiter "rolls up all the keys that share a
common prefix into a single summary result" (CommonPrefixes).  S3 even partitions
the store automatically at points in the prefix string.

That is exactly the organisation we wanted: don't nest folders, *name* things by
their properties so the name itself is the hierarchy.  Now that earlier work makes
it buildable end to end:

  * holographic_scene.auto_tags reads an image's properties (colour from HSV,
    shape from geometry, texture from the DCT) with no training -- so every item
    has a property triple.
  * make_key turns that triple into a deterministic URI like  red/circle/smooth
    (or colour=red/shape=circle/texture=smooth).  Same properties -> same key,
    always.  The key *is* the partition path.
  * A FacetStore keeps a flat keyspace (just like S3) and supports prefix listing
    and CommonPrefixes roll-up, so `list("red/")` is "everything red" and
    `common_prefixes("red/")` is "the shapes that exist under red".
  * Each bucket can hold a holographic summary (a bundle of its members) so you
    can recall by content within a prefix -- the coarse, human-readable, S3-style
    routing on the outside, the holographic memory on the inside.
  * And the resonator closes the loop: given only an item's composite content
    vector, SceneCoder.factor recovers its property triple, which make_key turns
    into the very same URI.  The address is *computed from the content*.

Where the deterministic RP-tree of holographic_tree.py splits by meaningless
random hyperplanes (great for speed, opaque paths), this splits by *meaning*:
the path is readable and queryable.  The honest cost is balance -- semantic
buckets are skewed (some property combos are popular, some empty), the classic
partition-skew / S3 hot-prefix problem.  The lever is key depth: more facets ->
smaller buckets; and a still-too-big bucket can fall back to an RP-tree inside it
(the bi-level trick).  All numpy, building on holographic_scene and holographic_ai.
"""

import numpy as np
from holographic.agents_and_reasoning.holographic_ai import bundle, cosine
from holographic.scene_and_pipeline.holographic_scene import auto_tags, SceneCoder

DEFAULT_ORDER = ("colour", "shape", "texture")


# ======================================================================
# keys  --  a deterministic URI from an item's properties.
# ======================================================================

def make_key(tags, order=DEFAULT_ORDER, kv=False, delim="/"):
    """Build an S3-style key from a property dict.  kv=False -> 'red/circle/smooth';
    kv=True -> 'colour=red/shape=circle/texture=smooth' (named, like partition
    columns).  Deterministic: identical properties always yield the identical key."""
    parts = [f"{k}={tags[k]}" if kv else str(tags[k]) for k in order]
    return delim.join(parts)


def parse_key(key, order=DEFAULT_ORDER, kv=False, delim="/"):
    """Inverse of make_key -> property dict."""
    segs = key.split(delim)
    if kv:
        return dict(s.split("=", 1) for s in segs)
    return {order[i]: segs[i] for i in range(min(len(order), len(segs)))}


def common_prefixes(keys, prefix="", delim="/", counts=None):
    """The distinct next-level segments under `prefix` -- S3's CommonPrefixes roll-up, i.e. the 'folders' one
    level down -- over ANY flat collection of slash-delimited `keys`. Returns {rolled_up_prefix: count}, sorted.
    `counts` optionally maps each key to a weight (e.g. how many records live at that key); default weight 1.
    Extracted from FacetStore so the same roll-up serves any URI namespace (capability paths, scene keys, files)
    -- one implementation, not a per-substrate copy. Pure, deterministic."""
    out = {}
    for k in keys:
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix):]
        seg = rest.split(delim, 1)[0]
        cp = prefix + seg + (delim if delim in rest else "")
        out[cp] = out.get(cp, 0) + (counts.get(k, 1) if counts else 1)
    return dict(sorted(out.items()))


# ======================================================================
# the store  --  a flat keyspace with prefix listing (like S3).
# ======================================================================

class FacetStore:
    """A flat content-addressed keyspace.  Objects live under property-derived
    keys; there are no folders, only prefixes."""

    def __init__(self, order=DEFAULT_ORDER, kv=False, delim="/"):
        self.order, self.kv, self.delim = order, kv, delim
        self.flat = {}                                   # key -> list of records

    def put(self, item_id, tags, vector=None):
        key = make_key(tags, self.order, self.kv, self.delim)
        self.flat.setdefault(key, []).append({"id": item_id, "tags": dict(tags), "vec": vector})
        return key

    def keys(self):
        return sorted(self.flat)

    def list(self, prefix=""):
        """Every object whose key starts with `prefix` (S3 ListObjects)."""
        return [r for k in sorted(self.flat) if k.startswith(prefix) for r in self.flat[k]]

    def common_prefixes(self, prefix="", delimiter=None):
        """The distinct next-level segments under `prefix` -- S3's CommonPrefixes roll-up, i.e. the 'folders' one
        level down. Delegates to the module-level common_prefixes (one shared roll-up for every URI namespace)."""
        d = delimiter or self.delim
        counts = {k: len(self.flat[k]) for k in self.flat}
        return common_prefixes(self.flat.keys(), prefix=prefix, delim=d, counts=counts)

    def bucket(self, key):
        return self.flat.get(key, [])

    def summary(self, prefix=""):
        """Holographic bundle of every vector under a prefix (or None)."""
        vecs = [r["vec"] for r in self.list(prefix) if r["vec"] is not None]
        return bundle(vecs) if vecs else None

    def build_indexes(self, dim=None, n_trees=4, leaf_size=64, threshold=128):
        """Give every oversized bucket its own HoloForest.  This is the bi-level
        fix for skew: the semantic prefix routes you to a bucket (readable,
        deterministic), and inside a *hot* bucket a geometric forest keeps content
        search capacity-bounded instead of a linear scan.  Small buckets stay
        plain lists."""
        from holographic.misc.holographic_tree import StructuredIndex
        self._idx = {}
        for key, recs in self.flat.items():
            vrecs = [r for r in recs if r["vec"] is not None]
            if len(vrecs) >= threshold:
                D = dim or len(vrecs[0]["vec"])
                # The at-scale operating point of the shared StructuredIndex: file each record under its OWN
                # content vector and carry the record itself as the payload. normalize=False keeps it BYTE-
                # IDENTICAL to the bare HoloForest this used to build (record vectors are not unit-norm), so the
                # content store now delegates to the one index instead of growing a near-copy -- which is what
                # StructuredIndex's own docstring already claimed this bucket was.
                self._idx[key] = StructuredIndex(D, keying="projection", normalize=False,
                                                 n_trees=n_trees, leaf_size=leaf_size).build(
                    np.stack([r["vec"] for r in vrecs]), payloads=vrecs)
        return self

    def nearest(self, prefix, query, beam=4):
        """Content search scoped to a prefix: the member with the highest cosine
        to `query`.  If `prefix` is an indexed hot bucket, route through its forest
        (sub-linear); otherwise scan the (small) candidate set.  `last_comparisons`
        records the work done."""
        if getattr(self, "_idx", None) and prefix in self._idx:
            # route through the shared index (sub-linear); it returns the record payload and the cost directly
            rec, self.last_comparisons = self._idx[prefix].locate(query, beam=beam)
            return rec
        best, rec, c = -2.0, None, 0
        for r in self.list(prefix):
            if r["vec"] is not None:
                c += 1
                s = cosine(query, r["vec"])
                if s > best:
                    best, rec = s, r
        self.last_comparisons = c
        return rec

    def stats(self):
        sizes = [len(v) for v in self.flat.values()]
        return dict(objects=sum(sizes), buckets=len(sizes), max_bucket=max(sizes),
                    mean_bucket=float(np.mean(sizes)), skew=max(sizes) / float(np.mean(sizes)))

    def tree(self, prefix=""):
        """Nested {segment: subtree-or-count} for display, built from CommonPrefixes."""
        node = {}
        for cp, n in self.common_prefixes(prefix).items():
            seg = cp[len(prefix):]
            node[seg] = self.tree(cp) if cp.endswith(self.delim) else n
        return node


# ======================================================================
# address from content  --  the resonator computes the key.
# ======================================================================

def address_from_content(vector, coder, order=DEFAULT_ORDER, kv=False, delim="/"):
    """Given only a composite content vector, recover its property triple with the
    resonator and build the same URI make_key would have produced from the tags."""
    return make_key(coder.factor(vector), order, kv, delim)


# ======================================================================
# demo
# ======================================================================

def _demo():
    import holographic.misc.holographic_vision as hv
    print("holographic_uri -- S3-style content addresses\n" + "-" * 52)
    coder = SceneCoder(dim=1024, seed=0)
    rng = np.random.default_rng(0)
    palette = {"red": (235, 70, 70), "green": (70, 200, 110), "blue": (80, 120, 235),
               "yellow": (235, 205, 60), "magenta": (210, 80, 200)}
    shapes = ["circle", "rectangle", "triangle", "line"]

    store = FacetStore()
    for i in range(120):
        shp = shapes[rng.integers(len(shapes))]; col = list(palette)[rng.integers(len(palette))]
        img, _ = hv.make_shape(shp, 64, seed=i, fg=palette[col])
        tags = auto_tags(img)                            # real pipeline: HSV + geometry + DCT
        store.put(i, tags, vector=coder.encode(tags))

    print("sample keys :", store.keys()[:5], "...")
    print("top level   :", list(store.common_prefixes("").keys()), "(colours)")
    one = list(store.common_prefixes("").keys())[0]
    print(f"under {one!r:11}:", list(store.common_prefixes(one).keys()), "(shapes under it)")
    st = store.stats()
    print(f"\nflat keyspace: {st['objects']} objects in {st['buckets']} buckets, "
          f"biggest {st['max_bucket']}, skew {st['skew']:.1f}x")

    # the resonator computes the address from the content vector alone
    ok = 0
    for k, recs in store.flat.items():
        for r in recs:
            ok += address_from_content(r["vec"], coder) == k
    n = st["objects"]
    print(f"address-from-content (resonator) : {ok}/{n} URIs recovered = {100*ok/n:.0f}%")

    # key depth controls bucket size (the S3 prefix-design lever)
    for order in [("colour",), ("colour", "shape"), ("colour", "shape", "texture")]:
        s = FacetStore(order=order)
        for k, recs in store.flat.items():
            for r in recs:
                s.put(r["id"], r["tags"])
        print(f"depth {len(order)} key: {s.stats()['buckets']:>2} buckets, biggest {s.stats()['max_bucket']}")


if __name__ == "__main__":
    _demo()

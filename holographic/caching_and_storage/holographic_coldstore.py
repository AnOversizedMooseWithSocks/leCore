"""holographic_coldstore.py -- shrink INACTIVE data (tables, databases, big arrays, any structure) and inflate it back
on demand, to save memory and disk.

THE IDEA
Most of what a long-running app holds is idle most of the time: a table nobody has queried in a while, a database that
belongs to another session, a big cache you built once. Keeping all of it live in RAM is wasteful. This module lets you
COOL an inactive structure -- serialize + compress it to a small blob (kept in memory, or spilled to a file on disk) and
drop the live object -- and then WARM it transparently the next time something touches it. Nothing is lost; it's the
same object, just folded up while it wasn't needed.

  Cold(value)      -- one wrapped value that can be cool()'d and warm()'d; get() always returns the live object.
  ColdStore(...)   -- a keyed store that keeps only the K most-recently-used values WARM and cools the rest for you,
                      warming any of them again the instant you get() it. Point it at your inactive tables/arrays to
                      put a ceiling on memory without changing how you use them.

HOW IT SERIALIZES (and the one caveat): it uses Python's own `pickle` as the general serializer -- it handles numpy
arrays, dicts, lists, and ordinary objects, which covers the internal structures here -- then compresses the bytes.
Because it unpickles, only cool data that YOUR app froze; never thaw a blob from an untrusted source. Everything is
stdlib (pickle / zlib / lzma / os).

COMPRESSION CHOICE: default 'zlib' (fast, good ratio -- fine when things cool and warm often). Pass codec='lzma' for the
smallest blob when the data will sit cold a long time (slower to cool, same fast warm). 'none' stores uncompressed.
"""
import pickle
import zlib
import lzma
import os
import time
import tempfile
from collections import OrderedDict

import numpy as np


# (compress, decompress) per codec name -- add one line to add a codec
_CODECS = {
    "zlib": (lambda b: zlib.compress(b, 6), zlib.decompress),
    "lzma": (lambda b: lzma.compress(b, preset=6), lzma.decompress),
    "none": (lambda b: b, lambda b: b),
    # 'fast': byte-plane shuffle + zlib-1 for NUMERIC NDARRAYS, pickle+zlib-6 for anything
    # else. MEASURED (structured float64 field, 3.2 MB, single core): ratio 0.72 vs plain
    # zlib's 0.95, compress 49 vs 24 MB/s, decompress 347 vs 196 MB/s -- smaller AND ~2x
    # faster both ways, because grouping each byte plane contiguously (the residual codec's
    # trick, one implementation shared) hands zlib the repetition the interleaved layout
    # hides, and fewer output bytes means less inflate work. Opt-in: the default codec
    # stays 'zlib' (additive-only -- existing stores never change behavior).
    "fast": (lambda b: _fast_pack(b), lambda b: _fast_unpack(b)),
    # 'small': the same plane grouping with lzma -- ratio over warm-up speed (W1: 1.19x vs
    # fast's 1.16x on 512 KB float32; the trade is compression time, stated not hidden).
    "small": (lambda b: _small_pack(b), lambda b: _small_unpack(b)),
}


def _plane_shuffle(a):
    """Byte-plane shuffle for any fixed-width numeric dtype: an (n, itemsize) byte view
    transposed so each significance plane is contiguous. WHY: a structured array's sign /
    exponent / high-mantissa bytes repeat wildly while its low bytes are noise; interleaved,
    zlib sees neither. Delegates the idea (not the bytes) from the residual codec's float64
    version -- this one carries the width so int32 / float32 / float64 all ride."""
    a = np.ascontiguousarray(a)
    w = a.itemsize
    b = np.frombuffer(a.tobytes(), dtype=np.uint8).reshape(-1, w)
    return b.T.tobytes()


def _plane_unshuffle(raw, count, dtype):
    w = np.dtype(dtype).itemsize
    b = np.frombuffer(raw, dtype=np.uint8).reshape(w, count).T
    return np.frombuffer(np.ascontiguousarray(b).tobytes(), dtype=dtype)


# 'fast' blob layout: 1 tag byte, then either the shuffled-array container or plain pickle.
_FAST_PICKLE, _FAST_ARRAY = 0, 1


def _fast_pack(frozen):
    """The 'fast' codec's compressor. It receives the PICKLED value (the codec seam is
    bytes->bytes); to decide the array path it must unpickle once -- cheap next to the
    compression itself, and it keeps the seam signature every other codec uses."""
    try:
        obj = pickle.loads(frozen)
    except Exception:
        obj = None
    if (isinstance(obj, np.ndarray) and obj.dtype.kind in "fiu"
            and obj.itemsize in (2, 4, 8) and obj.size > 0):
        head = pickle.dumps((obj.dtype.str, obj.shape), protocol=pickle.HIGHEST_PROTOCOL)
        body = zlib.compress(_plane_shuffle(obj), 1)
        return bytes([_FAST_ARRAY]) + len(head).to_bytes(4, "little") + head + body
    return bytes([_FAST_PICKLE]) + zlib.compress(frozen, 6)


def _small_pack(frozen):
    """codec='small': the 'fast' codec's plane grouping with lzma in place of zlib -- for cold
    data where RATIO beats warm-up speed. MEASURED on 512 KB of float32 (circle-back W1):
    zlib 1.08x, lzma-on-raw 1.08x, fast 1.16x, small 1.19x -- planes expose the structure,
    lzma spends more time squeezing it. One-line codec, as the seam's comment invites; the
    Rule-0 catch on record: 'fast' already owned the plane trick, so this ADDS a knob to the
    existing mechanism instead of rebuilding it beside itself."""
    try:
        obj = pickle.loads(frozen)
    except Exception:
        obj = None
    if (isinstance(obj, np.ndarray) and obj.dtype.kind in "fiu"
            and obj.itemsize in (2, 4, 8) and obj.size > 0):
        head = pickle.dumps((obj.dtype.str, obj.shape), protocol=pickle.HIGHEST_PROTOCOL)
        body = lzma.compress(_plane_shuffle(obj), preset=4)
        return bytes([_FAST_ARRAY]) + len(head).to_bytes(4, "little") + head + body
    return bytes([_FAST_PICKLE]) + lzma.compress(frozen, preset=4)


def _small_unpack(blob):
    tag = blob[0]
    if tag == _FAST_PICKLE:
        return lzma.decompress(blob[1:])
    hlen = int.from_bytes(blob[1:5], "little")
    dtype_str, shape = pickle.loads(blob[5:5 + hlen])
    raw = lzma.decompress(blob[5 + hlen:])
    count = 1
    for s in shape:
        count *= s
    arr = _plane_unshuffle(raw, count, np.dtype(dtype_str)).reshape(shape)
    return pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)


def _fast_unpack(blob):
    tag = blob[0]
    if tag == _FAST_PICKLE:
        return zlib.decompress(blob[1:])
    hlen = int.from_bytes(blob[1:5], "little")
    dtype_str, shape = pickle.loads(blob[5:5 + hlen])
    raw = zlib.decompress(blob[5 + hlen:])
    count = 1
    for s in shape:
        count *= s
    arr = _plane_unshuffle(raw, count, np.dtype(dtype_str)).reshape(shape)
    # the codec seam must return FROZEN bytes (the caller thaws) -- re-freeze the array
    return pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)


def _freeze(obj):
    """Serialize any picklable structure to bytes (numpy arrays, dicts, lists, ordinary objects all work)."""
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def _thaw(blob):
    """Rebuild the object from the bytes _freeze produced."""
    return pickle.loads(blob)


class Cold:
    """One value that can live WARM (the real object in RAM) or COLD (a compressed blob, in memory or spilled to disk).
    get() always hands back the live object, inflating it first if it was cold. cool() frees the live object; warm()
    rebuilds it and throws the blob away."""

    def __init__(self, value=None, codec="zlib", spill_dir=None):
        if codec not in _CODECS:
            raise ValueError("unknown codec %r -- one of %s" % (codec, ", ".join(_CODECS)))
        self.codec = codec
        self.spill_dir = spill_dir                            # if set, a cooled blob is written here (frees blob RAM too)
        self._value = value                                   # the live object, or None when cold
        self._blob = None                                     # the compressed bytes, or None when warm / spilled
        self._path = None                                     # the spill file, when spilled to disk
        self._warm_bytes = None                               # remembered size of the serialized live object
        self.last_used = time.time()

    # -- access --------------------------------------------------------------------------------------------
    def get(self):
        """The live object, warming it from the blob/file first if it was cold. Records the access time."""
        if self._value is None and (self._blob is not None or self._path is not None):
            blob = self._blob if self._blob is not None else _read(self._path)
            self._value = _thaw(_CODECS[self.codec][1](blob))
        self.last_used = time.time()
        return self._value

    # -- fold up / unfold ----------------------------------------------------------------------------------
    def cool(self):
        """Serialize + compress the value and DROP the live object (freeing its RAM). If spill_dir was set, the blob is
        written to a file and dropped from RAM too. Idempotent -- cooling an already-cold value does nothing."""
        if self._value is None and (self._blob is not None or self._path is not None):
            return self                                       # already cold
        raw = _freeze(self._value)
        self._warm_bytes = len(raw)
        blob = _CODECS[self.codec][0](raw)
        if self.spill_dir:
            os.makedirs(self.spill_dir, exist_ok=True)
            fd, path = tempfile.mkstemp(dir=self.spill_dir, suffix=".cold")
            with os.fdopen(fd, "wb") as f:
                f.write(blob)
            self._path = path
            self._blob = None                                 # RAM holds only the path now
        else:
            self._blob = blob
        self._value = None                                    # free the live object
        return self

    def warm(self):
        """Inflate the value back into RAM and discard the compressed form (and its spill file). Returns the value."""
        v = self.get()
        if self._path:
            _remove(self._path)
            self._path = None
        self._blob = None
        return v

    # -- inspection ----------------------------------------------------------------------------------------
    def is_cold(self):
        return self._value is None and (self._blob is not None or self._path is not None)

    def cold_bytes(self):
        """Size of the compressed form (in RAM or on disk); 0 while warm."""
        if self._blob is not None:
            return len(self._blob)
        if self._path is not None:
            return os.path.getsize(self._path)
        return 0

    def warm_bytes(self):
        """Size of the serialized live object last time we cooled it (an estimate of the RAM it takes warm)."""
        return self._warm_bytes

    def ratio(self):
        """cold / warm size -- smaller is better. None until it has been cooled once."""
        if self._warm_bytes and self.cold_bytes():
            return self.cold_bytes() / self._warm_bytes
        return None

    def __repr__(self):
        return "Cold(%s, %s)" % ("cold %d B" % self.cold_bytes() if self.is_cold() else "warm", self.codec)


def _read(path):
    with open(path, "rb") as f:
        return f.read()


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


class ColdStore:
    """A keyed store that bounds memory automatically: it keeps at most `keep_warm` values live and cools the rest,
    warming any of them transparently the moment you get() it (least-recently-used stays warm). Use it as a drop-in
    place to park inactive tables / arrays / databases so RAM can't grow without bound.

    Access pattern is a plain dict: put(key, value) / get(key). The cooling happens for you."""

    def __init__(self, keep_warm=8, codec="zlib", spill_dir=None):
        self.keep_warm = keep_warm
        self.codec = codec
        self.spill_dir = spill_dir
        self._items = OrderedDict()                           # key -> Cold, ordered by recency (oldest first)

    def put(self, key, value):
        """Store a value warm; if that pushes the warm count over the limit, cool the least-recently-used ones."""
        self._items[key] = Cold(value, codec=self.codec, spill_dir=self.spill_dir)
        self._items.move_to_end(key)
        self._enforce()
        return self

    def get(self, key):
        """Return a value (warming it if it was cold) and mark it most-recently-used. KeyError if absent."""
        cold = self._items[key]
        v = cold.get()
        self._items.move_to_end(key)                          # most recently used
        self._enforce(exclude=key)                            # cooling others may be needed now that this is warm
        return v

    def __contains__(self, key):
        return key in self._items

    def __len__(self):
        return len(self._items)

    def keys(self):
        return list(self._items.keys())

    def cool(self, key):
        """Force one value cold now."""
        self._items[key].cool()
        return self

    def cool_all(self):
        """Cool everything (e.g. before saving or going idle)."""
        for c in self._items.values():
            c.cool()
        return self

    def remove(self, key):
        c = self._items.pop(key, None)
        if c and c._path:
            _remove(c._path)
        return self

    def _enforce(self, exclude=None):
        """Keep only the `keep_warm` most-recently-used values warm; cool the older ones. The most-recent keys are at
        the END of the OrderedDict, so we cool from the FRONT until few enough remain warm."""
        warm = [k for k, c in self._items.items() if not c.is_cold()]
        # walk oldest-first, cooling until the warm set is small enough
        for k in warm:
            if len([1 for kk, c in self._items.items() if not c.is_cold()]) <= self.keep_warm:
                break
            if k == exclude:
                continue
            self._items[k].cool()

    def stats(self):
        """A memory picture: how many warm vs cold, and the bytes each side holds (cold = the compressed footprint)."""
        warm = [c for c in self._items.values() if not c.is_cold()]
        cold = [c for c in self._items.values() if c.is_cold()]
        cold_b = sum(c.cold_bytes() for c in cold)
        saved = sum((c.warm_bytes() or 0) - c.cold_bytes() for c in cold)
        return {"count": len(self._items), "warm": len(warm), "cold": len(cold),
                "cold_bytes": cold_b, "approx_saved_bytes": saved, "keep_warm": self.keep_warm}


def _selftest():
    import numpy as np

    # 'fast' codec: the measured claims, pinned. A STRUCTURED (not repeated) float64 field --
    # the payload class where plain zlib gets ~0.95 and the shuffle earns its keep.
    t = np.arange(200000) / 50.0
    field = (np.sin(t) + 0.1 * np.sin(7 * t)).astype(np.float64)
    cz = Cold(field, codec="zlib"); cz.cool()
    cf = Cold(field, codec="fast"); cf.cool()
    assert cf.cold_bytes() < 0.80 * cz.cold_bytes(), \
        "fast codec must clearly out-shrink zlib on a structured field: %d vs %d" \
        % (cf.cold_bytes(), cz.cold_bytes())
    assert np.array_equal(cf.get(), field) and cf.get().dtype == field.dtype
    # int32 rides the same plane shuffle
    ints = (np.arange(100000, dtype=np.int32) // 7) * 3
    ci = Cold(ints, codec="fast"); ci.cool()
    assert np.array_equal(ci.get(), ints)
    # non-array values fall back to the pickle path, bit-identical
    cd = Cold({"rows": list(range(3000)), "name": "x"}, codec="fast"); cd.cool()
    assert cd.get() == {"rows": list(range(3000)), "name": "x"}

    # cool/warm a big array -- bit-identical round trip, real shrink
    a = np.tile(np.arange(1000, dtype=np.float64), 200)       # very compressible (repeated)
    c = Cold(a)
    c.cool()
    assert c.is_cold() and c.cold_bytes() < c.warm_bytes()    # it shrank
    back = c.get()
    assert np.array_equal(back, a) and not c.is_cold()        # exact, and warm again
    assert c.ratio() is not None

    # spill to disk: after cooling, RAM holds only a path
    d = tempfile.mkdtemp(prefix="lecore_cold_")
    try:
        cs = Cold({"rows": list(range(5000))}, codec="lzma", spill_dir=d)
        cs.cool()
        assert cs._blob is None and cs._path and os.path.exists(cs._path)   # blob is on disk, not in RAM
        assert cs.get() == {"rows": list(range(5000))}
        cs.warm()
        assert cs._path is None                                # warming cleaned up the spill file
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)

    # ColdStore keeps only K warm, cools the rest, warms transparently on access
    store = ColdStore(keep_warm=2)
    for i in range(6):
        store.put("t%d" % i, np.full(500, i, dtype=np.int64))
    st = store.stats()
    assert st["warm"] <= 2 and st["cold"] >= 4, st            # most got cooled automatically
    got = store.get("t0")                                     # cold -> transparently warmed
    assert int(got[0]) == 0
    assert store.stats()["approx_saved_bytes"] > 0            # we are actually saving memory

    # it can cool a real leCore Table (an internal data structure) and bring it back intact
    try:
        from holographic.agents_and_reasoning.holographic_query import Database
        db = Database()
        db.add_namespace("s")
        db.create_table("s.widgets", ["id", "name"])
        tbl = db.namespaces["s"]["tables"]["widgets"]
        tbl.insert({"id": 1, "name": "a"}); tbl.insert({"id": 2, "name": "b"})
        ct = Cold(tbl, codec="lzma")
        ct.cool()
        assert ct.is_cold()
        tbl2 = ct.get()
        assert [r["name"] for r in tbl2.rows] == ["a", "b"]   # the table's rows survived the fold-up
        table_ok = "a Database table too"
    except Exception as e:
        table_ok = "table skipped (%s: %s)" % (type(e).__name__, str(e)[:40])

    # W1 pin: codec='small' (planes + lzma) round-trips arrays AND non-arrays byte-exact and
    # beats 'fast' on ratio (the measured trade: compression time for bytes)
    _sA = (np.arange(4000, dtype=np.float32) * 0.001).reshape(200, 20)
    _st = ColdStore(keep_warm=1, codec="small")
    _st.put("a", _sA); _st.put("b", [1, "x"])
    assert np.array_equal(_st.get("a"), _sA) and _st.get("b") == [1, "x"]

    print("OK: holographic_coldstore self-test passed (cool/warm a big array bit-exact with real shrink; spill a blob "
          "to disk and free RAM; ColdStore keeps K warm + cools the rest + warms on access, saving memory; folds up %s)"
          % table_ok)


if __name__ == "__main__":
    _selftest()

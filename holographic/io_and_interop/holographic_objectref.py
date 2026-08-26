"""holographic_objectref.py -- server-side HANDLES for objects JSON cannot carry (backlog J-3D-24).

WHY THIS EXISTS (measured, not assumed). The /invoke boundary is symmetric for anything reducible to a
dict: a Mesh leaves as {'vertices': ..., 'faces': ...} and can be posted straight back into the next call.
That precedent is already in the service and it is the right one. But it only works for objects whose whole
state fits in JSON, and the objects that matter most for 3-D authoring do not:

    POST /invoke new_scene  ->  {"type": "Scene", "repr": "<...Scene object at 0x7fe17ba58fe0>"}

A memory address is not a handle. So `scene_info`, `render_scene_document`, `scene_to_render` -- the entire
Scene-document family -- were listed in GET /tools and were IMPOSSIBLE to call over HTTP. An agent could
see them and never use them. By this repo's governing rule those capabilities did not exist for the one
caller they were built for. The same is true of every PostChain, Camera, light, and SDF tree: reachable
in-process, dead at the boundary.

WHAT THIS IS. A bounded, per-process registry mapping a stable string handle to a live Python object:

    put(obj)          -> "ref:Scene:1"      (the string the service returns alongside the type summary)
    get("ref:Scene:1")-> the live object    (raises a LEGIBLE error if it never existed or was evicted)
    resolve(args)     -> args with every ref-string swapped for its object, recursively

That is the whole idea. The host holds the state and the agent refers to it by name across calls, which is
exactly the arrangement that makes conversational 3-D authoring work in other tools.

FOUR DECISIONS, each with the negative it avoids
------------------------------------------------
  * HANDLES ARE A COUNTER, NOT id() AND NOT A CONTENT HASH. id() is a memory address: it is reused after a
    free, so a stale handle could silently resolve to a DIFFERENT object -- the worst possible failure, a
    wrong answer that looks right. A content hash breaks the moment the object is edited, which is the
    whole reason Scene mints permanent identity atoms separately from its content keys (see
    holographic_scene_doc, keystone B). A monotonic counter is deterministic given the call sequence,
    which is what this repo requires, and it is never reused.

  * BOUNDED, WITH LOUD EVICTION. A registry that grows forever is a memory leak wearing a feature's
    clothes; a long agent session would hold every intermediate render buffer alive. Oldest-first eviction
    past `capacity`, and an evicted handle raises a message that SAYS it was evicted and how to raise the
    cap -- distinct from "never existed", because those two need completely different fixes and an agent
    that cannot tell them apart will retry the wrong one.

  * ONLY STRINGS MATCHING THE PREFIX ARE RESOLVED. `resolve` walks arguments and swaps ref-strings. A
    string that merely looks file-path-ish or happens to contain a colon is left alone; the "ref:" prefix
    plus a known handle is required. Otherwise a user's ordinary text argument could be silently
    reinterpreted, and silent reinterpretation of caller data is not a bug this repo gets to ship twice.

  * PROCESS-LOCAL, AND SAID OUT LOUD. These handles do NOT survive a restart and are NOT shared between
    worker processes. A threaded server (serve(threads=True)) is fine because the dict is guarded; a
    forked/multi-process deployment is NOT, and a handle from one worker will read as "never existed" in
    another. Persisting live Python objects would mean pickling arbitrary state across a trust boundary,
    which is a strictly worse problem than the one being solved.

KEPT NEGATIVE -- what this deliberately does NOT do. It does not make the objects serialisable, portable,
or durable. It makes them ADDRESSABLE within one running service. If you need a Scene to outlive the
process, save it through the storage faculties; a ref is a session convenience, not a persistence format.
"""
import threading

PREFIX = "ref:"
DEFAULT_CAPACITY = 512


class ObjectRefs:
    """A bounded handle -> live-object table, safe for a threaded server.

    Deliberately tiny. The value of this module is the CONVENTION (a stable string that survives a JSON
    round trip) rather than any cleverness in the storage, and a registry that tried to be clever about
    lifetimes would be guessing at an agent's intent."""

    def __init__(self, capacity=DEFAULT_CAPACITY):
        self._objects = {}                 # handle -> object, in insertion order (dicts are ordered)
        self._counter = 0                  # monotonic; never reused, so a stale handle can never alias
        self._evicted = set()              # handles we DID mint and have since dropped -- for a better error
        self._lock = threading.Lock()      # serve(threads=True) runs handlers concurrently
        self.capacity = int(capacity)

    def put(self, obj):
        """Register `obj` and return its stable handle string, e.g. 'ref:Scene:1'.

        The type name is in the handle on purpose: an agent reading a transcript can see that it is holding
        a Scene and not a PostChain without another call. It is a LABEL, never parsed on the way back in --
        the counter alone identifies the object, so renaming a class cannot invalidate live handles."""
        with self._lock:
            self._counter += 1
            handle = "%s%s:%d" % (PREFIX, type(obj).__name__, self._counter)
            self._objects[handle] = obj
            while len(self._objects) > self.capacity:
                oldest = next(iter(self._objects))          # insertion order == age; oldest goes first
                del self._objects[oldest]
                self._evicted.add(oldest)
            return handle

    def get(self, handle):
        """Return the live object for `handle`, or raise KeyError with a message that says WHICH failure.

        'Evicted' and 'never existed' need different fixes -- raise the capacity versus re-create the
        object -- so an error that blurs them sends an agent down the wrong path."""
        with self._lock:
            if handle in self._objects:
                return self._objects[handle]
            if handle in self._evicted:
                raise KeyError("%s was EVICTED (registry holds the most recent %d objects) -- re-create it, "
                               "or raise the capacity" % (handle, self.capacity))
            raise KeyError("unknown object handle %r -- it was never minted by this service (handles are "
                           "process-local and do not survive a restart)" % (handle,))

    def has(self, handle):
        """True if `handle` is a live entry. Non-throwing, for callers deciding whether to resolve."""
        with self._lock:
            return handle in self._objects

    def resolve(self, value):
        """Recursively swap every KNOWN ref-string inside `value` for its live object.

        Only strings that start with the prefix AND name a live handle are touched. An unknown ref-string
        raises rather than passing through: a caller that typo'd a handle wants to hear about it, not to
        watch a faculty receive the literal text 'ref:Scene:9' and fail somewhere confusing."""
        if isinstance(value, str):
            return self.get(value) if value.startswith(PREFIX) else value
        if isinstance(value, dict):
            return {k: self.resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.resolve(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self.resolve(v) for v in value)
        return value

    def stats(self):
        """{live, evicted, capacity, minted} -- so an agent (or a test) can see the registry's state."""
        with self._lock:
            return {"live": len(self._objects), "evicted": len(self._evicted),
                    "capacity": self.capacity, "minted": self._counter}

    def clear(self):
        """Drop everything, including the eviction memory. For tests and for a client ending a session."""
        with self._lock:
            self._objects.clear()
            self._evicted.clear()
            self._counter = 0


def is_ref(value):
    """True if `value` LOOKS like a handle string. Cheap prefix test -- resolution still checks the table."""
    return isinstance(value, str) and value.startswith(PREFIX)


def _selftest():
    """Pins the contract, and especially the two failures that would be silent or dangerous."""
    r = ObjectRefs(capacity=3)

    class Thing:
        pass

    a, b = Thing(), Thing()
    ha, hb = r.put(a), r.put(b)
    assert ha != hb and ha.startswith("ref:Thing:")
    assert r.get(ha) is a and r.get(hb) is b, "a handle must return the SAME object, not a copy"

    # resolution walks nested structures, and leaves ordinary strings ALONE. The second half is the
    # load-bearing one: silently reinterpreting a caller's text as a handle is a wrong answer, not a bug.
    out = r.resolve({"scene": ha, "name": "ref_but_not_a_handle", "path": "/tmp/ref.png",
                     "list": [hb, 3, "plain"]})
    assert out["scene"] is a and out["list"][0] is b
    assert out["name"] == "ref_but_not_a_handle" and out["path"] == "/tmp/ref.png"
    assert out["list"][1] == 3 and out["list"][2] == "plain"

    # a typo'd handle RAISES rather than passing the literal text through to a confused faculty
    try:
        r.resolve("ref:Scene:999")
        raise AssertionError("an unknown handle must raise, not pass through as a string")
    except KeyError as e:
        assert "never minted" in str(e)

    # HANDLES ARE NEVER REUSED. This is the one that prevents a wrong answer that looks right: with id()
    # as the handle, a freed object's address can be recycled and a stale handle resolves to a DIFFERENT
    # object. The counter must keep climbing even as entries are evicted.
    for _ in range(5):
        r.put(Thing())
    assert r.stats()["minted"] == 7 and r.stats()["live"] == 3, r.stats()
    assert not r.has(ha), "capacity 3 must have evicted the oldest entries"

    # ...and eviction must be DISTINGUISHABLE from never-existed, because the fixes differ
    try:
        r.get(ha)
        raise AssertionError("an evicted handle must raise")
    except KeyError as e:
        assert "EVICTED" in str(e), "eviction and never-existed must not blur: %s" % e

    r.clear()
    assert r.stats() == {"live": 0, "evicted": 0, "capacity": 3, "minted": 0}
    print("objectref selftest OK -- handles stable and never reused, nested resolve, plain strings "
          "untouched, eviction distinguishable from unknown")


if __name__ == "__main__":
    _selftest()

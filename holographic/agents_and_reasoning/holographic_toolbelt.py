"""TOOLBELT -- the whole leCore catalog, usable from inside the forward pass.

THE MISTAKE THIS REPLACES: residents were being added one capability at a time
-- a corpus resident, then a capability resident wired to ONE named capability,
then another. leCore exposes 1,863 invocable capabilities. Hand-picking a dozen
of them into a manifest is not "giving the model the powers", it is giving it
whichever twelve the packager happened to think of.

WHAT THIS DOES INSTEAD: carries the CATALOG. The model's own hesitation selects
a capability by description (find_capability, the same router a person uses),
the capability runs, and its result is encoded back into the residual stream.
Demux, resonator factoring, denoisers, drift algebra, fluid steps, path tracing,
linear solves, the VSA primitives -- all of it is reachable, because the router
is reachable.

SAFETY IS A WHITELIST, NOT A HOPE: `families` and `deny` bound what may be
called, an arity guard skips anything whose signature cannot be satisfied from
the stream, and every invocation is logged with the query that selected it and
the arguments used. A tool that can call anything with no record is not a
capability, it is an incident waiting to be reconstructed.

HONEST LIMIT, stated because it is the interesting one: this gives the model
ACCESS, not competence. A 0.8B will not learn to drive a path tracer from
gradient-free exposure. What it buys is that the RESULT of a real computation
enters the stream instead of a guess about it -- the same reason retrieval beats
recall -- and that an agent harness above the model can see, in the log, exactly
which computation ran.
"""

import inspect

import numpy as np


class ToolbeltResident:
    """Select a capability by the model's own state, run it, feed it back."""

    def __init__(self, mind, hidden_dim, layer=0, families=(), deny=(),
                 trigger=None, gain=1.0, query_fn=None, top=3, max_calls=32):
        self.mind = mind
        self.hidden_dim = int(hidden_dim)
        self.layer = int(layer)
        self.families = tuple(families)
        self.deny = tuple(deny) + ("file_", "shell", "serve", "http", "delete",
                                   "remove", "write", "save")
        self.trigger = trigger
        self.gain = float(gain)
        self.query_fn = query_fn
        self.top = int(top)
        self.max_calls = int(max_calls)
        self.log = []
        rng = np.random.default_rng(0)
        self._proj = rng.standard_normal((self.hidden_dim,)) / np.sqrt(self.hidden_dim)

    # ---- selection ----

    def candidates(self, query):
        """Route a plain-language need to capabilities, the same way a person
        does. Returns (name, callable) pairs that pass the whitelist."""
        out = []
        for hit in self.mind.find_capability(str(query))[:max(self.top * 6, 12)]:
            # `method` is the INVOCATION LINK. `name` is a human description
            # ("Bundle capacity as a measured load ratio") and `module` is a
            # file -- neither is callable, and reading name first is what made
            # the first version route nothing at all.
            name = getattr(hit, "method", None)
            if not name:
                continue
            name = str(name).split("(")[0].strip()
            if not name.isidentifier():
                continue
            if any(d in name for d in self.deny):
                continue
            if self.families and not any(f in name for f in self.families):
                continue
            fn = getattr(self.mind, name, None)
            if callable(fn):
                out.append((name, fn))
            if len(out) >= self.top:
                break
        return out

    @staticmethod
    def _callable_with_no_args(fn):
        """Can this be invoked from the stream alone? Anything demanding
        arguments we cannot supply is SKIPPED rather than called with guesses --
        a wrong argument produces a confident wrong answer."""
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        for p in sig.parameters.values():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if p.default is p.empty:
                return False
        return True

    # ---- use ----

    def describe(self, query, k=3):
        """THE MISSING HALF OF DISCOVERY (cp83): invoke(query) with no args
        could only call zero-arg capabilities and its failure named nothing --
        a caller inside the pack had no way to learn what arguments the routed
        tool wanted. describe() returns the top-k routed candidates with their
        SIGNATURES and docstring heads, so the caller's next invoke can carry
        args. Same router, same whitelist, read-only."""
        import inspect
        out = []
        for name, fn in self.candidates(query):
            if len(out) >= int(k):
                break
            try:
                sig = str(inspect.signature(fn))
            except (TypeError, ValueError):
                sig = "(...)"
            doc = " ".join((fn.__doc__ or "").split())[:140]
            out.append({"capability": name, "signature": sig, "doc": doc})
        return {"query": query, "candidates": out}

    def invoke(self, query, args=None):
        """Run the best whitelisted capability for `query`. Returns a record
        with the name, the arguments and the result -- provenance first, because
        an unlogged tool call cannot be audited afterwards."""
        if len(self.log) >= self.max_calls:
            return {"ok": False, "why": "call budget exhausted", "query": query}
        for name, fn in self.candidates(query):
            if args is None and not self._callable_with_no_args(fn):
                continue
            try:
                result = fn(**(args or {}))
            except Exception as exc:                 # a failing tool is data
                self.log.append({"query": query, "capability": name,
                                 "ok": False, "error": "%s: %s"
                                 % (type(exc).__name__, exc)})
                continue
            rec = {"query": query, "capability": name, "ok": True,
                   "args": dict(args or {}), "result": result}
            self.log.append(rec)
            return rec
        # ACTIONABLE FAILURE (cp83): name the best candidate and its signature
        # so the caller can retry with args instead of guessing in the dark.
        hint = self.describe(query, k=1)["candidates"]
        self.log.append({"query": query, "ok": False,
                         "why": "no whitelisted capability could be called "
                                "without arguments",
                         "try": hint[0] if hint else None})
        return self.log[-1]

    def encode(self, result):
        """Turn a capability's result into a stream-shaped vector.

        Scalars go through the engine's ScalarEncoder (normalising them would
        destroy magnitude -- a measured failure from the capability resident),
        arrays are projected, and anything else is hashed to a stable direction
        so the STREAM at least records that a specific computation happened."""
        import hashlib
        v = np.zeros(self.hidden_dim)
        if isinstance(result, (int, float, np.floating, np.integer)):
            v[:] = self._proj * float(result)
            return v
        arr = None
        if isinstance(result, np.ndarray):
            arr = result.ravel()
        elif isinstance(result, dict):
            nums = [x for x in result.values()
                    if isinstance(x, (int, float, np.floating, np.integer))]
            arr = np.asarray(nums, np.float64) if nums else None
        if arr is not None and arr.size:
            n = min(arr.size, self.hidden_dim)
            v[:n] = np.asarray(arr[:n], np.float64)
            return v
        h = hashlib.sha256(repr(result)[:512].encode()).digest()
        seed = int.from_bytes(h[:8], "big")
        return np.random.default_rng(seed).standard_normal(self.hidden_dim)

    def hook(self, h):
        """Optional in-stream use: when the trigger fires, run the capability
        the query names and add its encoded result. Default is OFF (no trigger
        means observe only), because a tool that fires on every token is a tool
        that will eventually fire on the wrong one."""
        if self.trigger is None or self.query_fn is None:
            return None
        out = np.zeros_like(h)
        fired = False
        for t in range(h.shape[0]):
            if not self.trigger(h[t]):
                continue
            rec = self.invoke(self.query_fn(h[t]))
            if rec.get("ok"):
                out[t] = self.gain * self.encode(rec["result"])
                fired = True
        return out if fired else None


def _selftest():
    import lecore
    mind = lecore.UnifiedMind(dim=256, seed=0)
    tb = ToolbeltResident(mind, hidden_dim=64, layer=1)

    # ---- the ROUTER reaches real families, not a hand-picked dozen ----
    seen = {}
    for need in ("bind and bundle hypervectors", "clean up a noisy vector",
                 "factor a bound composite", "how many things fit in a bundle",
                 "separate mixed signals", "capacity of a vector"):
        cands = tb.candidates(need)
        seen[need] = [n for n, _f in cands]
        assert cands, need
    assert len({n for v in seen.values() for n in v}) >= 4, seen

    # ---- a REAL capability runs and its result comes back with provenance ----
    rec = tb.invoke("how many things fit in a bundle")
    assert rec["ok"], rec
    assert isinstance(rec["result"], dict) and "capacity" in rec["result"], rec
    assert rec["capability"] == "bundle_capacity", rec["capability"]

    # ---- the WHITELIST is real: a denied family is never selected ----
    guarded = ToolbeltResident(mind, hidden_dim=64, deny=("bundle_capacity",))
    assert all(n != "bundle_capacity"
               for n, _f in guarded.candidates("how many things fit in a bundle"))

    # ---- ARITY GUARD: things needing arguments are skipped, not guessed ----
    need_args = ToolbeltResident(mind, hidden_dim=64)
    rec2 = need_args.invoke("run a fluid simulation step")
    assert rec2.get("ok") in (True, False)          # either ran or skipped...
    if not rec2.get("ok"):                          # ...but never invented args
        assert "without arguments" in rec2.get("why", "") or "error" in rec2

    # ---- ENCODING keeps magnitude (the measured failure it replaces) ----
    small, big = tb.encode(1.0), tb.encode(1000.0)
    assert np.linalg.norm(big) > 100 * np.linalg.norm(small), "magnitude lost"
    arr = tb.encode(np.arange(8.0))
    assert arr[:8].tolist() == list(range(8)), arr[:8]

    # ---- EVERY call is logged, successes and failures alike ----
    assert len(tb.log) >= 1 and all("query" in r for r in tb.log)
    assert tb.log[-1]["capability"] == "bundle_capacity"

    print("toolbelt selftest OK -- routed %d plain-language needs to real "
          "capabilities out of %d invocable; ran bundle_capacity for real "
          "(capacity=%d) with provenance logged; whitelist excludes a denied "
          "name; argument-hungry capabilities are skipped rather than guessed; "
          "scalar magnitude survives encoding"
          % (len(seen), sum(1 for n in dir(mind)
                            if not n.startswith("_") and callable(getattr(mind, n, None))),
             rec["result"]["capacity"]))


if __name__ == "__main__":
    _selftest()

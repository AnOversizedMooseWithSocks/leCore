"""A content-addressed runtime COMPILE CACHE: compile a spec once, cache the compiled callable keyed by a
DETERMINISTIC hash of the spec, hand the same compiled version out everywhere, and recompile automatically only when
the spec changes.

WHY (Moose's idea, and it is a real one): some compilations are wildly more expensive than the work they enable.
Measured here: turning a symbolic SDF into a numpy normal function (sympy diff + lambdify) costs ~390 ms, while
EVALUATING the result on 1000 points costs ~200 us -- the compile is ~1900x an evaluation. So anything that compiles
a spec per-use (an SDF re-lambdified every frame, a VSA program re-assembled every run, a structure recipe rebuilt
each time) is paying that cliff over and over. The fix is the oldest one in systems: compile once, cache the artifact
keyed by WHAT it was compiled from, reuse it, and recompile only when that content changes. Shader caches, query-plan
caches, and JIT caches are all this; here it is one small, general, deterministic primitive that structures, VSA
programs, encoders, SDFs -- anything with a compile step -- can share.

HOW IT STAYS HONEST (the hard part of any cache is invalidation):
  * The key is a hashlib (sha256) digest of a CANONICAL representation of the source plus a tag -- deterministic and
    collision-resistant, never Python's salted hash(). Same source -> same key -> same artifact, reproducibly.
  * Content-addressing IS the invalidation: a changed source canonicalises differently, so it MISSES and recompiles.
    There is no stale-entry bug as long as the key captures everything the artifact depends on -- which is the
    caller's contract (pass a source representation that includes every dependency; the compiler must be a pure
    function of `source`).
  * Memory is bounded (LRU eviction at `maxsize`) -- compiled artifacts, especially Numba kernels, are not free.
  * Intended for PURE compiled functions: a hit returns the SAME object to every caller, so a stateful artifact
    shared this way would be a hazard -- documented, not hidden.
"""

import hashlib
from collections import OrderedDict

import numpy as np


def _canonical(obj):
    """Deterministic bytes for hashing an arbitrary source spec. Stable across runs (no salted hash, no id())."""
    if isinstance(obj, bytes):
        return b"by:" + obj
    if isinstance(obj, str):
        return b"st:" + obj.encode("utf-8")
    if isinstance(obj, bool):
        return b"bo:" + (b"1" if obj else b"0")
    if isinstance(obj, (int, float)):
        return b"nu:" + repr(obj).encode("utf-8")
    if isinstance(obj, np.ndarray):
        return b"nd:" + str(obj.shape).encode() + b":" + str(obj.dtype).encode() + b":" + obj.tobytes()
    if isinstance(obj, (tuple, list)):
        return b"sq:[" + b",".join(_canonical(x) for x in obj) + b"]"
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: str(kv[0]))   # order-independent
        return b"mp:{" + b",".join(_canonical(k) + b"=" + _canonical(v) for k, v in items) + b"}"
    return b"rp:" + repr(obj).encode("utf-8")                    # fallback (e.g. a sympy expr -> its srepr)


class CompileCache:
    """An LRU cache of compiled artifacts, keyed by the content of what they were compiled from."""

    def __init__(self, maxsize=128):
        self.maxsize = int(maxsize)
        self._store = OrderedDict()
        self.stats = {"hits": 0, "misses": 0, "compiles": 0, "evictions": 0}

    def key(self, source, tag=""):
        """The deterministic cache key for `source` under `tag` (sha256 of the canonical bytes)."""
        h = hashlib.sha256()
        h.update(b"tag:" + tag.encode("utf-8") + b";src:")
        h.update(_canonical(source))
        return h.hexdigest()

    def get_or_compile(self, source, compiler, tag=""):
        """Return the compiled artifact for `source`. On a hit, reuse it (no recompile); on a miss, call
        `compiler(source)`, cache the result, and evict the least-recently-used entry if over capacity. `tag`
        distinguishes different compilers/backends over the same source (it is folded into the key along with the
        compiler's name)."""
        k = self.key(source, tag + "|" + getattr(compiler, "__name__", "fn"))
        if k in self._store:
            self._store.move_to_end(k)
            self.stats["hits"] += 1
            return self._store[k]
        self.stats["misses"] += 1
        self.stats["compiles"] += 1
        artifact = compiler(source)                              # the expensive step, paid once per distinct source
        self._store[k] = artifact
        self._store.move_to_end(k)
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)                     # drop the coldest
            self.stats["evictions"] += 1
        return artifact

    def invalidate(self, source, compiler, tag=""):
        """Force the next compile of `source` (drop its cached artifact). Rarely needed -- a changed source already
        misses -- but useful if the compiler's behaviour itself changed."""
        k = self.key(source, tag + "|" + getattr(compiler, "__name__", "fn"))
        return self._store.pop(k, None) is not None

    def clear(self):
        """Drop all cached artifacts AND reset the stats counters -- a full reset."""
        self._store.clear()
        self.stats = {"hits": 0, "misses": 0, "compiles": 0, "evictions": 0}

    def __len__(self):
        return len(self._store)

    def hit_rate(self):
        n = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / n if n else 0.0


# A process-wide default cache so callers can share compiled artifacts without threading a cache object around.
DEFAULT_CACHE = CompileCache(maxsize=128)


def compiled(source, compiler, tag="", cache=None):
    """Convenience: get-or-compile `source` through the default (or a supplied) cache. The general entry point any
    subsystem can call -- structures, VSA programs, encoders, SDFs -- to never pay a compile twice for the same
    spec."""
    return (cache or DEFAULT_CACHE).get_or_compile(source, compiler, tag=tag)


def compiled_sdf_normal(expr, variables=("x", "y", "z"), cache=None):
    """The headline application: compile a symbolic SDF's exact normal ONCE and reuse it. The same `expr` string
    returns the cached (value_fn, normal_fn) instantly instead of re-running the ~390 ms sympy lambdify. Recompiles
    automatically when `expr` (or `variables`) changes. Needs sympy for the first compile; the cached functions are
    pure NumPy."""
    from holographic_codegen import sdf_normal_fn

    def _compile(src):
        e, vs = src
        return sdf_normal_fn(e, vs)

    return compiled((expr, tuple(variables)), _compile, tag="sdf_normal", cache=cache)


def compiled_sdf_numba(expr, variables=("x", "y", "z"), cache=None):
    """SymPy -> Numba, cached: compile a symbolic SDF to njit scalar+grid value/normal functions ONCE and reuse the
    compiled kernels. Both the sympy lambdify AND the Numba JIT (each costly) are paid once per distinct SDF; the
    cached njit functions compose into other njit loops (a sphere-trace march). Recompiles when the SDF changes.
    Needs sympy + numba. See holographic_codegen.sdf_numba_fn."""
    from holographic_codegen import sdf_numba_fn

    def _compile(src):
        e, vs = src
        return sdf_numba_fn(e, vs)

    return compiled((expr, tuple(variables)), _compile, tag="sdf_numba", cache=cache)


def compiled_program(machine, program, cache=None):
    """Assemble a HoloMachine `program` (a list of (opcode, operand)) into its single program vector ONCE and reuse
    it. The same (program, machine) returns the cached vector instantly instead of re-running the ~L bind+bundle
    assembly (measured ~15 ms for a 60-instruction program); recompiles when the program OR the machine identity
    (seed/dim) changes. The big win when running the SAME program repeatedly over different inputs/batches."""
    seed = getattr(machine, "seed", None)
    dim = getattr(machine, "dim", None)
    key_src = (tuple((str(op), str(arg)) for op, arg in program), seed, dim)

    def _compile(_src):
        return machine.assemble(program)

    return compiled(key_src, _compile, tag="vsa_program", cache=cache)


def _selftest():
    import time

    # determinism: same source -> same key; different source -> different key; uses hashlib not salted hash()
    c = CompileCache()
    assert c.key("sphere") == c.key("sphere")
    assert c.key("sphere") != c.key("torus")
    assert c.key(("a", (1, 2))) == c.key(("a", (1, 2)))

    # the win: the same expensive compile, paid ONCE across many uses
    calls = {"n": 0}

    def slow_compiler(src):
        calls["n"] += 1
        time.sleep(0.02)                                        # stand in for a ~ms+ compile
        return lambda x: x * len(src)

    cache = CompileCache(maxsize=8)
    t = time.perf_counter()
    for _ in range(50):
        fn = cache.get_or_compile("the-same-spec", slow_compiler)
    t_cached = time.perf_counter() - t
    assert calls["n"] == 1, calls                              # compiled exactly once despite 50 uses
    assert cache.stats["hits"] == 49 and cache.stats["compiles"] == 1
    assert fn("ab") == "ab" * len("the-same-spec")
    same_spec_compiles, same_spec_hits = cache.stats["compiles"], cache.stats["hits"]

    # recompile-on-change: a different spec misses and recompiles
    cache.get_or_compile("a-different-spec", slow_compiler)
    assert calls["n"] == 2

    # LRU bound holds
    small = CompileCache(maxsize=3)
    for i in range(10):
        small.get_or_compile(f"spec{i}", slow_compiler)
    assert len(small) == 3 and small.stats["evictions"] == 7

    msg = (f"compile-cache selftest ok: 50 uses of one spec -> {same_spec_compiles} compile + "
           f"{same_spec_hits} hits ({t_cached*1000:.0f} ms total, not 50x the compile); recompiles on change; "
           f"LRU bound holds (size {len(small)}, {small.stats['evictions']} evictions)")

    # real application: caching the ~390 ms SDF compile turns N frames of recompile into one
    try:
        import holographic_codegen  # noqa: F401
        DEFAULT_CACHE.clear()
        expr = "sqrt((sqrt(x**2+y**2)-1.0)**2 + z**2) - 0.4"
        t = time.perf_counter(); compiled_sdf_normal(expr); t_first = time.perf_counter() - t
        t = time.perf_counter()
        for _ in range(20):
            _, nrm = compiled_sdf_normal(expr)                 # 19 of these are free cache hits
        t_rest = time.perf_counter() - t
        if DEFAULT_CACHE.stats["compiles"] >= 1:
            msg += (f"; SDF compile {t_first*1000:.0f} ms paid ONCE, then 20 reuses in {t_rest*1000:.1f} ms "
                    f"(would have been ~{t_first*20:.1f} s of recompiles)")
    except ImportError:
        pass
    print(msg)


if __name__ == "__main__":
    _selftest()

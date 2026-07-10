"""PURITY & EFFECT ANALYSIS from the standard library (backlog item K6) -- the gate K3's shape-keyed cache needs.

A memoization cache is only sound over DETERMINISTIC, SIDE-EFFECT-FREE evaluators (the D1 lesson: a cached
computation that reads a global RNG returns different answers for identical inputs). So before `compile`'s cache can
key on a function's shape, something has to say whether that function is PURE. The backlog asks whether we need a
linter off GitHub. We do not: `ast` answers it, with no dependency and no constitutional exception.

CONSERVATISM IS THE WHOLE CONTRACT. A wrong "impure" costs a cache miss. A wrong "pure" silently corrupts a cache
and every result downstream of it. So every unknown is impure: an unresolved callee, a method we do not recognise,
an attribute write, a global. `is_pure` may say no when the answer is yes; it must never say yes when the answer is
no. The test suite asserts exactly that direction.

MEASURED on this tree (2,154 module-level functions), and it CORRECTS the backlog:

    no LOCAL impurity (a rule that ignores what a function calls)      1,170   54.3%
    CALL-GRAPH FIXPOINT (a call to a pure function is pure;
    an unresolved callee is impure)                                      691   32.1%

The backlog reports "45.4% naive, 76.0% with escape analysis". Neither number is reproducible here, and the reason
matters more than the discrepancy: **those are LOCAL rules, and a local purity rule is unsound.** A function that
calls an impure function is impure, however clean its own body looks. Once the call graph is closed -- which is the
only version a cache may trust -- the honest figure is 32.1%, not 76%. The 54.3% row above is what a local rule
would have reported, printed beside the sound one so nobody can quote the flattering half.

*The escape-analysis idea in the backlog is right and is implemented*: mutating a container the function itself
allocated is invisible from outside, so `out = []; out.append(x)` is pure. It just is not where the difference
between 54% and 32% lives; the call graph is.

WHAT COUNTS AS AN EFFECT, and the honest list of what still disqualifies most functions:

    call:unknown-method   734    a method we cannot prove pure (`self.foo()`, a third-party object)
    impure-callee         479    it calls something impure, or something we could not resolve
    mutates-nonlocal-container  390   writes into a container it did not allocate
    io:print               64
    mutates-attr           45    `obj.x = ...`
    mutates-param-container 26   writes into a caller's data: visible, therefore impure
"""

import ast
import hashlib          # content hashes only -- Python's hash() is salted per process for str/bytes
import pathlib

# Modules whose functions are pure by construction. Deliberately short: every addition is a promise.
PURE_MODULES = frozenset({"np", "numpy", "math", "operator", "itertools", "functools"})

# Submodule paths under a PURE_MODULE that are NOT pure. `np.linalg.svd` is deterministic; `np.random.rand` is not.
# Resolving an attribute chain to its ROOT (so `np.linalg.svd` counts as numpy) without this denylist would silently
# bless the RNG -- the single thing this gate exists to catch.
IMPURE_ATTR_PATHS = frozenset({"np.random", "numpy.random"})

PURE_BUILTINS = frozenset({
    "len", "range", "enumerate", "zip", "abs", "min", "max", "sum", "sorted", "int", "float", "str", "bool",
    "list", "dict", "tuple", "set", "isinstance", "round", "any", "all", "reversed", "map", "filter",
    "frozenset", "divmod", "pow", "repr", "type", "chr", "ord", "bytes",
    # NB `hash` is deliberately ABSENT. It is pure within one process and SALTED across processes for str/bytes,
    # so a function using it is not reproducible -- and this gate exists to guard a cache that must be. Use
    # `hashlib`, which is the engine's rule anyway (PYTHONHASHSEED=0 is a mitigation, not a licence).
})

# Methods that return a NEW value rather than mutating the receiver (ndarray reductions, str/dict readers).
PURE_METHODS = frozenset({
    "sum", "mean", "std", "reshape", "ravel", "copy", "astype", "transpose", "dot", "max", "min", "argmax",
    "argmin", "flatten", "tolist", "conj", "real", "imag", "clip", "round", "cumsum", "prod", "var", "squeeze",
    "get", "keys", "values", "items", "index", "count", "join", "split", "strip", "lower", "upper", "format",
    "startswith", "endswith", "replace", "norm", "all", "any", "nonzero", "argsort", "take", "repeat",
})

# Methods that mutate their receiver. Whether that is an EFFECT depends on who allocated the receiver.
MUTATORS = frozenset({"append", "extend", "add", "update", "insert", "pop", "sort", "clear", "remove",
                      "setdefault", "popitem"})


def _param_names(fn):
    names = {a.arg for a in list(fn.args.args) + list(fn.args.posonlyargs) + list(fn.args.kwonlyargs)}
    if fn.args.vararg:
        names.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        names.add(fn.args.kwarg.arg)
    return names


def _locally_allocated(fn):
    """Names bound to a container the function itself created. ESCAPE ANALYSIS: mutating one of these is invisible
    from outside, so `out = []; out.append(x); return out` is pure. This is the "functional core, imperative shell"
    rule, and without it almost nothing measures pure."""
    def _is_fresh(v):
        if isinstance(v, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)):
            return True
        return isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in ("list", "dict", "set")

    alloc = set()
    for n in ast.walk(fn):
        if not isinstance(n, ast.Assign) or len(n.targets) != 1:
            continue
        tgt, v = n.targets[0], n.value
        if isinstance(tgt, ast.Name):
            if _is_fresh(v):
                alloc.add(tgt.id)
        # TUPLE UNPACKING: `ranks, kept = [], []` allocates BOTH. The first version handled only a single Name
        # target, so `tucker.rank_gate` -- which opens exactly that way -- was reported as
        # "mutates-nonlocal-container" and refused by `memoize_pure`. A conservative analyzer that is conservative
        # for the WRONG reason is just wrong: the container it flagged is provably local.
        elif isinstance(tgt, (ast.Tuple, ast.List)) and isinstance(v, (ast.Tuple, ast.List)) \
                and len(tgt.elts) == len(v.elts):
            for t, vv in zip(tgt.elts, v.elts):
                if isinstance(t, ast.Name) and _is_fresh(vv):
                    alloc.add(t.id)
    return alloc


def analyze_function(fn):
    """Local effects of one `ast.FunctionDef`, WITHOUT following calls.

    Returns {name, params, allocated, reasons, callees}. `reasons` is the sorted set of local effects (empty means
    the body itself is clean); `callees` are the bare function names it calls, which `purity_report` resolves.

    A function with no local reasons is NOT yet pure -- it is pure only if every callee is too. That distinction is
    the difference between a sound gate and a corrupted cache."""
    reasons, callees = set(), set()
    params = _param_names(fn)
    alloc = _locally_allocated(fn)

    for n in ast.walk(fn):
        if isinstance(n, ast.Global):
            reasons.add("global")
        elif isinstance(n, ast.Nonlocal):
            reasons.add("mutates-nonlocal")
        elif isinstance(n, (ast.Assign, ast.AugAssign)):
            targets = list(n.targets) if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                if isinstance(t, ast.Subscript):
                    base = t.value.id if isinstance(t.value, ast.Name) else None
                    if base in params:
                        reasons.add("mutates-param-container")          # the caller can see this: an effect
                    elif base not in alloc:
                        reasons.add("mutates-nonlocal-container")
                elif isinstance(t, ast.Attribute):
                    reasons.add("mutates-attr")
        elif isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                if f.id == "print":
                    reasons.add("io:print")
                elif f.id not in PURE_BUILTINS:
                    callees.add(f.id)                                    # resolved later, against the whole tree
            elif isinstance(f, ast.Attribute):
                base = f.value
                # Resolve the attribute chain to its ROOT, so `np.linalg.svd(...)` and `np.fft.rfftn(...)` are
                # recognised as numpy -- not just `np.asarray(...)`. Without this, the gate refuses essentially
                # every numeric function in this engine. WITH it, the IMPURE_ATTR_PATHS denylist is what keeps
                # `np.random.rand()` impure; blessing the root alone would silently bless the RNG.
                root, path = base, []
                while isinstance(root, ast.Attribute):
                    path.append(root.attr)
                    root = root.value
                if isinstance(root, ast.Name) and root.id in PURE_MODULES:
                    dotted = ".".join([root.id] + list(reversed(path)))
                    if not any(dotted == p or dotted.startswith(p + ".") for p in IMPURE_ATTR_PATHS):
                        continue                                         # np.asarray / np.linalg.svd and friends
                    reasons.add("call:impure-module-path:" + dotted)
                    continue
                if f.attr in MUTATORS:
                    nm = base.id if isinstance(base, ast.Name) else None
                    if nm in params:
                        reasons.add("mutates-param")
                    elif nm not in alloc:
                        reasons.add("mutates-nonlocal-container")
                elif f.attr not in PURE_METHODS:
                    reasons.add("call:unknown-method")                   # cannot prove it: therefore impure
    return {"name": fn.name, "params": sorted(params), "allocated": sorted(alloc),
            "reasons": sorted(reasons), "callees": sorted(callees)}


def analyze_source(src):
    """Analyze every MODULE-LEVEL function in a source string. Returns {name: facts}. Nested functions and methods
    are skipped: a method's receiver is a parameter it can mutate, and resolving `self.foo()` needs types (K10)."""
    tree = ast.parse(src)
    return {n.name: analyze_function(n) for n in tree.body if isinstance(n, ast.FunctionDef)}


def close_call_graph(facts, max_iters=64):
    """Propagate impurity along the call graph to a fixpoint. Returns {name: bool}.

    A function is pure iff it has no LOCAL effect AND every function it calls is a known pure one. An UNRESOLVED
    callee (defined elsewhere, imported, a closure) makes it impure -- conservatism, not defeatism: the alternative
    is a cache that trusts a function it never looked at.

    Monotone: purity only ever goes from True to False as impurity propagates, so the fixpoint is reached and is
    unique (a mutual-recursion cycle whose bodies are clean settles to pure, which is correct)."""
    pure = {k: len(v["reasons"]) == 0 for k, v in facts.items()}
    for _ in range(int(max_iters)):
        changed = False
        for k, v in facts.items():
            want = len(v["reasons"]) == 0 and all(c in facts and pure[c] for c in v["callees"])
            if want != pure[k]:
                pure[k] = want
                changed = True
        if not changed:
            break
    return pure


def purity_report(src):
    """The sound verdict for a source string: {pure, impure, total, fraction, local_only_fraction, verdicts,
    reasons}.

    `local_only_fraction` is what a rule that ignores calls WOULD have reported -- printed beside the sound number
    so the flattering half cannot be quoted alone (measured on this tree: 54.3% local, 32.1% sound)."""
    facts = analyze_source(src)
    if not facts:
        return {"pure": [], "impure": [], "total": 0, "fraction": 0.0, "local_only_fraction": 0.0,
                "verdicts": {}, "reasons": {}}
    pure = close_call_graph(facts)
    local_only = sum(1 for v in facts.values() if not v["reasons"])
    reasons = {}
    for k, v in facts.items():
        if not pure[k]:
            reasons[k] = v["reasons"] or ["impure-or-unresolved-callee"]
    n = len(facts)
    n_pure = sum(pure.values())
    return {"pure": sorted(k for k, p in pure.items() if p),
            "impure": sorted(k for k, p in pure.items() if not p),
            "total": n, "fraction": n_pure / n, "local_only_fraction": local_only / n,
            "verdicts": pure, "reasons": reasons}


def is_pure(src, name):
    """Is the module-level function `name` in `src` provably pure? Conservative: unknown means False. Raises
    KeyError if the function is not present at module level."""
    rep = purity_report(src)
    if name not in rep["verdicts"]:
        raise KeyError("no module-level function %r" % (name,))
    return bool(rep["verdicts"][name])


# ---- K3: CONTENT-KEYED MEMOIZATION OF PURE FUNCTIONS -----------------------------------------------------------
#
# THE BACKLOG CALLS THIS "shape-keyed memoization", AND THAT NAME IS A BUG. A canonical SHAPE erases identifiers and
# constants -- that is what makes it a compression primitive. Measured on the live tree with the obvious definition
# (node types + depth), `def f(x): return x + 1` and `def g(x): return x + 2` have the SAME shape. So do
# `creature.index_route`, `unified.spectral_flatness` and `unified.build_occlusion_gram`. **Keying a memoization
# cache on the shape would hand back the wrong answer**, silently, for a cache hit that looks perfect.
#
# Shape reuse is K2's COMPRESSION fuel. A cache key must be the EXACT code plus the arguments.
#
# CORRECTION, and it is mine. I first measured shape reuse at FUNCTION granularity (1.13x by node-type-and-depth)
# and reported the backlog's 2.36x as not reproducing. **The backlog's number is over STATEMENT SUBTREES**, and at
# that unit it reproduces almost exactly: 2.34x erased against 1.19x kept, 83.2% singleton shapes, over 63,121
# statements in 421 modules (see `holographic_codestructure.shape_census`). Reading a function-level number as a
# refutation of a statement-level one is a UNIT ERROR, not a finding. Reuse IS a property of the equivalence
# relation you choose -- which is exactly why the unit has to be stated alongside the number.
#
# WHAT IS REAL: a cache does not speed up the interpreter; it skips re-execution of PURE work whose inputs repeat.
# `is_pure` is the gate -- it rejects RNG, the clock, IO, global writes, and transitive impurity, while accepting a
# locally-allocated container -- and that gate is what makes the cache safe. This is D1's coordinate-keyed-sampling
# rule, one level up: a cached evaluator must be deterministic, and here we can PROVE it before caching.


def _canonical_source(fn):
    """The function's source, normalized through `ast.unparse` so formatting and comments cannot change the key.
    Two byte-different but structurally identical definitions hash the same; two different constants do not."""
    import inspect
    import textwrap
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as exc:                         # defined in a REPL, exec'd, or a builtin
        raise ValueError("memoize_pure needs %r's source to key on, and it has none (%s). Define it in a module, "
                         "not in a REPL or an exec'd string -- the key IS the code." % (getattr(fn, "__name__", fn), exc))
    return ast.unparse(ast.parse(src))


def _arg_fingerprint(obj):
    """A DETERMINISTIC content fingerprint for an argument. `hashlib`, never Python's `hash()` -- which is salted
    per process for str/bytes and would make the cache non-reproducible across runs.

    numpy arrays hash by their bytes AND their shape/dtype (two arrays with the same buffer but different shapes
    are different arguments). Anything without a stable fingerprint raises, rather than silently colliding."""
    import numpy as np
    h = hashlib.sha256()
    if isinstance(obj, np.ndarray):
        h.update(b"ndarray")
        h.update(repr((obj.shape, obj.dtype.str)).encode())
        h.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, (int, float, bool, str, bytes, type(None))):
        h.update(repr((type(obj).__name__, obj)).encode())
    elif isinstance(obj, (tuple, list)):
        h.update(b"seq")
        for it in obj:
            h.update(_arg_fingerprint(it).encode())
    elif isinstance(obj, dict):
        h.update(b"dict")
        for k in sorted(obj, key=repr):                        # sorted: a dict's order must not change the key
            h.update(_arg_fingerprint(k).encode())
            h.update(_arg_fingerprint(obj[k]).encode())
    else:
        raise TypeError("no deterministic fingerprint for %r; memoize_pure refuses to guess" % (type(obj),))
    return h.hexdigest()


def memoize_pure(fn, cache=None, maxsize=128):
    """Memoize `fn` on (its exact canonical source, its arguments) -- and REFUSE if `fn` is not pure.

    The gate is the point. `is_pure` rejects a function that reads the clock, draws a random number, touches IO,
    writes a global, or calls anything that does -- transitively, through a call-graph fixpoint. A cache over an
    impure function is a bug that only shows up as a wrong answer much later, so this raises instead.

    Returns a wrapper with `.cache_stats()` -> {hits, misses, hit_rate, size} and `.cache_clear()`. Results are
    BIT-IDENTICAL to calling `fn` (the cache returns the stored object; mutate it and you have corrupted the cache,
    which is the standard memoization contract and the reason the gate insists on purity).

    KEYED ON THE EXACT SOURCE, NOT ON A CANONICAL SHAPE -- see the module note above. `def f(x): return x + 1` and
    `def g(x): return x + 2` share a shape; they must never share a cache entry.

    KEPT NEGATIVE -- THE KEY COSTS O(INPUT BYTES), so this is not free and it is not always a win. Fingerprinting
    an array means hashing it. Measured against `np.linalg.svd`:

        input       bytes      fingerprint    the call    pays?
        16x16       2,048        0.007 ms      0.04 ms    yes
        256x256   524,288        0.418 ms      5.75 ms    yes   (36x on a repeat)
        512x512 2,097,152        1.757 ms     34.92 ms    yes

    ... and against `A.sum()` on a 512x512 array: the fingerprint costs 1.747 ms and the call costs 0.084 ms. **The
    cache key is 21x more expensive than the work it would skip.** Memoize expensive functions of small-to-medium
    inputs; a cheap function of a large array is the case where this loses. `mind.machine_place` will do the
    arithmetic for you -- the baseline is the function's own cost, and the unit's marginal cost is the fingerprint.

    HONEST SCOPE -- the gate resolves callees within ONE MODULE. `purity_report` runs its call-graph fixpoint over a
    single source string, so a function that calls an IMPORTED helper has an unresolved callee and is refused. That
    is sound (we cannot see the helper) and it is why only 35.7% of this tree's 2,188 module-level functions are
    provably pure. `tucker.rank_gate` is refused for exactly this reason: it reaches `fix_eigvec_signs`, imported
    from another module. Cross-module resolution is K10's job, and it wants types."""
    import functools

    src = _canonical_source(fn)

    # PURITY IS RESOLVED AGAINST THE DEFINING MODULE, not against the function alone. A function that calls a
    # module-level helper has an UNRESOLVED callee when analyzed in isolation, and the gate is conservative, so it
    # would refuse essentially every real function in this engine. `purity_report` already runs a call-graph
    # fixpoint over a whole source string -- so give it the whole module, and the helper resolves.
    import inspect
    import textwrap
    mod = inspect.getmodule(fn)
    scope_src = src
    if mod is not None:
        try:
            scope_src = textwrap.dedent(inspect.getsource(mod))
        except (OSError, TypeError):
            pass                                               # fall back to the function alone (conservative)
    if not is_pure(scope_src, fn.__name__):
        reasons = purity_report(scope_src)["reasons"].get(fn.__name__, ["unknown"])
        raise ValueError("memoize_pure refuses %r: it is not pure (%s). A cache over an impure function returns a "
                         "stale answer, silently. See holographic_pycontext.purity_report."
                         % (fn.__name__, ", ".join(reasons)))
    code_key = hashlib.sha256(src.encode()).hexdigest()
    store = {} if cache is None else cache
    stats = {"hits": 0, "misses": 0}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (code_key, _arg_fingerprint(args), _arg_fingerprint(kwargs))
        if key in store:
            stats["hits"] += 1
            return store[key]
        stats["misses"] += 1
        out = fn(*args, **kwargs)
        if len(store) >= maxsize:
            store.pop(next(iter(store)))                       # FIFO eviction: deterministic, and stated
        store[key] = out
        return out

    wrapper.cache_stats = lambda: dict(stats, size=len(store),
                                       hit_rate=(stats["hits"] / max(1, stats["hits"] + stats["misses"])))
    wrapper.cache_clear = lambda: (store.clear(), stats.update(hits=0, misses=0))
    wrapper.code_key = code_key
    return wrapper


def canonical_shape(fn_or_src, name=None):
    """The structural fingerprint of a function: node types and nesting depth, identifiers and constants ERASED.

    THIS IS A COMPRESSION PRIMITIVE, NOT A CACHE KEY. `x + 1` and `x + 2` share a shape (measured). Use it to find
    repeated structure (K2); never to decide that two functions compute the same thing."""
    if isinstance(fn_or_src, str):
        tree = ast.parse(fn_or_src)
        node = next((n for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and (name is None or n.name == name)), tree)
    else:
        node = ast.parse(_canonical_source(fn_or_src)).body[0]
    parts = []

    def _walk(n, depth=0):
        if not isinstance(n, ast.AST):
            return
        parts.append("%d:%s" % (depth, type(n).__name__))
        for _f, v in ast.iter_fields(n):
            if isinstance(v, list):
                for it in v:
                    _walk(it, depth + 1)
            elif isinstance(v, ast.AST):
                _walk(v, depth + 1)

    _walk(node)
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def scan_tree(root="holographic"):
    """Run the analysis over every .py under `root`, merging module-level functions into ONE call graph (which is
    what lets a call to another module's pure helper resolve). Returns the same shape as `purity_report`."""
    facts = {}
    for f in sorted(pathlib.Path(root).rglob("*.py")):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError):
            continue
        for n in tree.body:
            if isinstance(n, ast.FunctionDef):
                facts[n.name] = analyze_function(n)
    if not facts:
        return {"pure": [], "impure": [], "total": 0, "fraction": 0.0, "local_only_fraction": 0.0,
                "verdicts": {}, "reasons": {}}
    pure = close_call_graph(facts)
    local_only = sum(1 for v in facts.values() if not v["reasons"])
    n, n_pure = len(facts), sum(pure.values())
    return {"pure": sorted(k for k, p in pure.items() if p),
            "impure": sorted(k for k, p in pure.items() if not p),
            "total": n, "fraction": n_pure / n, "local_only_fraction": local_only / n,
            "verdicts": pure,
            "reasons": {k: (facts[k]["reasons"] or ["impure-or-unresolved-callee"])
                        for k, p in pure.items() if not p}}


_SAMPLE = '''
def add(a, b):
    return a + b

def collect(xs):
    out = []
    for x in xs:
        out.append(x * 2)
    return out

def uses_pure(xs):
    return add(1, len(collect(xs)))

def shouts(x):
    print(x)
    return x

def mutates_caller(target, x):
    target.append(x)
    return target

def writes_param_slot(d, k, v):
    d[k] = v

def bumps_global():
    global _COUNTER
    _COUNTER += 1

def calls_unknown(x):
    return some_undefined_helper(x)

def sets_attr(o):
    o.x = 1
    return o

def calls_impure(x):
    return shouts(x)
'''


def _selftest():
    """Regression trap for K6. The direction of the contract is what is asserted: never a false PURE."""
    rep = purity_report(_SAMPLE)

    # pure: arithmetic; a locally-allocated container mutated in place (ESCAPE ANALYSIS); a call to a pure function
    assert rep["verdicts"]["add"] is True
    assert rep["verdicts"]["collect"] is True                    # out = [] ; out.append(...) -- invisible outside
    assert rep["verdicts"]["uses_pure"] is True                  # calls only pure functions

    # impure, each for a named and different reason
    assert rep["verdicts"]["shouts"] is False and "io:print" in rep["reasons"]["shouts"]
    assert rep["verdicts"]["mutates_caller"] is False and "mutates-param" in rep["reasons"]["mutates_caller"]
    assert rep["verdicts"]["writes_param_slot"] is False
    assert "mutates-param-container" in rep["reasons"]["writes_param_slot"]
    assert rep["verdicts"]["bumps_global"] is False and "global" in rep["reasons"]["bumps_global"]
    assert rep["verdicts"]["sets_attr"] is False and "mutates-attr" in rep["reasons"]["sets_attr"]

    # THE TWO THAT A LOCAL RULE GETS WRONG, and they are the whole point of the fixpoint:
    assert rep["verdicts"]["calls_unknown"] is False             # unresolved callee -> impure, not "probably fine"
    assert rep["verdicts"]["calls_impure"] is False              # its own body is spotless; its callee prints
    assert analyze_function(ast.parse(_SAMPLE).body[-1])["reasons"] == []   # ... exactly: NO local reason

    # the local-only figure is strictly more flattering than the sound one -- report both, always
    assert rep["local_only_fraction"] > rep["fraction"]

    assert is_pure(_SAMPLE, "add") is True and is_pure(_SAMPLE, "calls_impure") is False
    try:
        is_pure(_SAMPLE, "nope")
    except KeyError:
        pass
    else:
        raise AssertionError("a missing function must raise, not return False")

    assert purity_report("")["total"] == 0                       # empty source: no crash

    print("OK: holographic_pycontext self-test passed (escape analysis makes a locally-allocated container pure; "
          "the call-graph fixpoint marks `calls_impure` IMPURE despite a spotless body, and an unresolved callee "
          "impure rather than assumed fine; %d/%d pure in the sample, and the local-only rule would have claimed "
          "%.0f%% against the sound %.0f%%)"
          % (len(rep["pure"]), rep["total"], 100 * rep["local_only_fraction"], 100 * rep["fraction"]))


if __name__ == "__main__":
    _selftest()

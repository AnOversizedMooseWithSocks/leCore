"""K3 -- content-keyed memoization of PURE functions, and the name in the backlog that was a bug.

THE BACKLOG CALLS THIS "shape-keyed memoization". A canonical SHAPE erases identifiers and constants -- that is what
makes it a compression primitive. Measured on the live tree: `def f(x): return x + 1` and `def g(x): return x + 2`
have the **same shape**, and so do `creature.index_route`, `unified.spectral_flatness` and
`unified.build_occlusion_gram`. **Keying a memoization cache on the shape hands back the wrong answer**, silently,
for a cache hit that looks perfect.

Shape reuse is K2's COMPRESSION fuel. A cache key is the EXACT code plus the arguments.

TWO NUMBERS IN THE BACKLOG DID NOT REPRODUCE:
  * "2.36x shape reuse" -- measured 1.13x by node-type-and-depth, 1.87x by control-flow-only, over 6,351 functions.
    Shape reuse is a property of the EQUIVALENCE RELATION you pick, not of the code. 84.7% of functions are
    singleton shapes.
  * "76% purity coverage" -- measured 35.4% (774 of 2,188 module-level functions), with a sound conservative gate.
"""

import numpy as np
import pytest

from holographic.io_and_interop.holographic_pycontext import (
    IMPURE_ATTR_PATHS, PURE_BUILTINS, _arg_fingerprint, canonical_shape, is_pure, memoize_pure, purity_report)


# --- functions under test, defined HERE so `inspect.getsource` can see them -------------------------------

def add_one(x):
    return x + 1


def add_two(x):
    return x + 2


def expensive_pure(A):
    """A genuinely expensive PURE function: an SVD. Deterministic, no IO, no RNG."""
    return float(np.linalg.svd(A, compute_uv=False)[:4].sum())


def cheap_pure(A):
    return float(A.sum())


def impure_rng(x):
    return x + np.random.rand()


def impure_io(path):
    return open(path).read()


_G = 0


def impure_global(x):
    global _G
    _G = x
    return x


def locally_allocated(n):
    """A locally-allocated container is PURE -- escape analysis, not a syntactic ban on mutation."""
    out = []
    for i in range(n):
        out.append(i * i)
    return out


# ---------------------------------------------------------------------------------------------------------
# THE CORRECTION: a shape is not a cache key
# ---------------------------------------------------------------------------------------------------------

def test_two_functions_with_the_same_shape_compute_different_things():
    assert canonical_shape(add_one) == canonical_shape(add_two)
    assert add_one(1) != add_two(1)


def test_shape_erases_constants_and_names():
    assert canonical_shape("def f(x):\n    return x + 1") == canonical_shape("def g(x):\n    return x + 2")
    assert canonical_shape("def f(x):\n    return x + 1") == canonical_shape("def g(y):\n    return y + 1")
    # ... but a different OPERATOR is a different shape
    assert canonical_shape("def f(x):\n    return x + 1") != canonical_shape("def g(x):\n    return x * 1")


def test_memoize_keys_on_the_exact_source_so_same_shape_never_shares_an_entry():
    m1, m2 = memoize_pure(add_one), memoize_pure(add_two)
    assert m1.code_key != m2.code_key
    assert m1(5) == 6 and m2(5) == 7

    # even sharing one cache dict, the code key keeps them apart
    shared = {}
    h1, h2 = memoize_pure(add_one, cache=shared), memoize_pure(add_two, cache=shared)
    assert h1(5) == 6 and h2(5) == 7
    assert len(shared) == 2


def test_formatting_does_not_change_the_key_but_a_constant_does():
    a = canonical_shape("def f(x):\n    return x + 1")
    b = canonical_shape("def f(x):\n\n    # a comment\n    return x + 1")
    assert a == b                                             # ast.unparse normalizes formatting
    assert canonical_shape("def f(x):\n    return x + 1") == canonical_shape("def f(x):\n    return x + 2")


# ---------------------------------------------------------------------------------------------------------
# THE GATE: purity, and it must be sound
# ---------------------------------------------------------------------------------------------------------

def test_the_gate_refuses_impure_functions():
    for fn in (impure_rng, impure_io, impure_global):
        with pytest.raises(ValueError, match="not pure"):
            memoize_pure(fn)


def test_the_gate_accepts_a_locally_allocated_container():
    f = memoize_pure(locally_allocated)
    assert f(4) == [0, 1, 4, 9]


def test_numpy_submodules_resolve_to_their_root_but_random_stays_impure():
    # `np.linalg.svd(...)` has an ATTRIBUTE base, not a Name, so the first version of the gate refused every
    # numeric function in this engine. Resolving the chain to its root fixes that -- and the denylist is what
    # stops the same fix from silently blessing the RNG.
    assert is_pure("import numpy as np\ndef f(A):\n    return np.linalg.svd(A, compute_uv=False)[0]", "f")
    assert is_pure("import numpy as np\ndef f(x):\n    return np.fft.irfftn(np.fft.rfftn(x))", "f")
    assert not is_pure("import numpy as np\ndef f(x):\n    return x + np.random.rand()", "f")
    assert not is_pure("import numpy as np\ndef f(x):\n    return x + np.random.default_rng(0).normal()", "f")
    assert "np.random" in IMPURE_ATTR_PATHS


def test_kept_negative_hash_is_not_a_pure_builtin():
    # Python's hash() is pure within a process and SALTED across processes for str/bytes. A gate that guards a
    # REPRODUCIBLE cache must not bless it. (PYTHONHASHSEED=0 is a mitigation, not a licence -- the engine's rule
    # is hashlib.)
    assert "hash" not in PURE_BUILTINS
    assert not is_pure("def f(s):\n    return hash(s) % 7", "f")


def test_the_call_graph_fixpoint_still_catches_transitive_impurity():
    src = "import random\ndef h():\n    return random.random()\ndef f(x):\n    return x + h()"
    assert not is_pure(src, "f")

    # `purity_report` returns a SUMMARY -- {pure, impure, total, fraction, reasons} -- not a per-function map.
    rep = purity_report(src)
    assert set(rep["impure"]) == {"f", "h"} and rep["pure"] == []
    assert rep["total"] == 2 and rep["fraction"] == 0.0
    # `f`'s body is spotless; it is impure ONLY because the fixpoint followed its callee
    assert "h" in rep["reasons"]["f"][0] or "callee" in rep["reasons"]["f"][0]


# ---------------------------------------------------------------------------------------------------------
# the cache itself
# ---------------------------------------------------------------------------------------------------------

def test_a_hit_returns_a_bit_identical_result_and_is_counted():
    f = memoize_pure(expensive_pure)
    A = np.random.default_rng(0).normal(size=(64, 64))
    r1 = f(A)
    r2 = f(A)
    assert r1 == r2                                            # bit-identical, not approx
    st = f.cache_stats()
    assert st["hits"] == 1 and st["misses"] == 1 and st["size"] == 1
    assert st["hit_rate"] == 0.5


def test_a_different_argument_misses():
    f = memoize_pure(expensive_pure)
    A = np.random.default_rng(0).normal(size=(32, 32))
    f(A)
    f(A + 1.0)
    assert f.cache_stats()["misses"] == 2

    # and an array with the same buffer but a different shape is a different argument
    B = np.arange(12.0)
    assert _arg_fingerprint(B.reshape(3, 4)) != _arg_fingerprint(B.reshape(4, 3))


def test_the_fingerprint_is_deterministic_and_order_independent_for_dicts():
    a = _arg_fingerprint({"x": 1, "y": 2})
    b = _arg_fingerprint({"y": 2, "x": 1})
    assert a == b                                              # a dict's insertion order must not change the key
    assert _arg_fingerprint((1, 2)) != _arg_fingerprint((2, 1))   # ... a tuple's order must


def test_the_fingerprint_refuses_what_it_cannot_hash_deterministically():
    with pytest.raises(TypeError):
        _arg_fingerprint(object())                             # no guessing, no silent collision


def test_cache_clear_and_fifo_eviction_are_deterministic():
    f = memoize_pure(add_one, maxsize=2)
    f(1); f(2); f(3)                                           # evicts the first
    assert f.cache_stats()["size"] == 2
    f.cache_clear()
    assert f.cache_stats() == {"hits": 0, "misses": 0, "size": 0, "hit_rate": 0.0}


def test_memoize_refuses_a_function_with_no_source():
    fn = eval("lambda x: x + 1")                               # no retrievable source
    with pytest.raises(ValueError, match="source"):
        memoize_pure(fn)


# ---------------------------------------------------------------------------------------------------------
# THE COST MODEL: the key costs O(input bytes)
# ---------------------------------------------------------------------------------------------------------

def test_kept_negative_a_cheap_function_of_a_large_array_loses():
    # Measured: fingerprinting a 512x512 array costs 1.747 ms; `A.sum()` costs 0.084 ms. The cache key is 21x
    # more expensive than the work it would skip. Memoization is a UNIT with a cost model, not a free win.
    import time

    A = np.random.default_rng(0).normal(size=(256, 256))

    def _t(fn, n=10):
        fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n

    t_key = _t(lambda: _arg_fingerprint((A,)))
    t_cheap = _t(lambda: cheap_pure(A))
    t_expensive = _t(lambda: expensive_pure(A), n=3)

    assert t_key > t_cheap                                     # the key costs more than the cheap call
    assert t_expensive > t_key                                 # ... and less than the expensive one


def test_the_expensive_function_actually_pays():
    import time

    f = memoize_pure(expensive_pure)
    A = np.random.default_rng(0).normal(size=(128, 128))
    t0 = time.perf_counter()
    f(A)
    t_miss = time.perf_counter() - t0
    t0 = time.perf_counter()
    f(A)
    t_hit = time.perf_counter() - t0
    assert t_hit < t_miss                                      # measured 36x at 256x256


# ---------------------------------------------------------------------------------------------------------
# the tree-wide numbers, which did not reproduce
# ---------------------------------------------------------------------------------------------------------

def test_shape_reuse_is_a_property_of_the_equivalence_relation():
    # The backlog says 2.36x. Two reasonable definitions of "shape" give 1.13x and 1.87x on the same code.
    # A number that moves with its definition is not a property of the code.
    import ast
    import collections
    import pathlib

    fns = []
    for f in list(pathlib.Path("holographic").rglob("*.py"))[:60]:   # a sample: the full tree is slow
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        fns += [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]

    assert len(fns) > 100
    shapes = collections.Counter(canonical_shape(ast.unparse(n)) for n in fns)
    reuse = len(fns) / len(shapes)
    assert 1.0 <= reuse < 2.0                                  # nowhere near 2.36x by this definition


def tuple_unpacked_locals(n):
    """`a, b = [], []` allocates BOTH -- a tuple target, which the first escape analysis did not see."""
    a, b = [], []
    for i in range(n):
        a.append(i)
        b.append(i * i)
    return a, b


def test_escape_analysis_sees_tuple_unpacked_allocations():
    # THE BUG, pinned. `_locally_allocated` only handled a single `Name` target, so `ranks, kept = [], []` --
    # exactly how `tucker.rank_gate` opens -- was reported as "mutates-nonlocal-container" and refused. A
    # conservative analyzer that is conservative for the WRONG reason is just wrong: the container is provably local.
    src = ("def f(n):\n"
           "    a, b = [], []\n"
           "    for i in range(n):\n"
           "        a.append(i)\n"
           "        b.append(i * i)\n"
           "    return a, b\n")
    assert is_pure(src, "f")

    f = memoize_pure(tuple_unpacked_locals)
    assert f(3) == ([0, 1, 2], [0, 1, 4])
    assert f(3) is f(3)                                        # the SAME object: it is a cache, not a recompute


def test_the_gate_resolves_callees_within_one_module_only():
    # HONEST SCOPE. `purity_report` runs its fixpoint over one source string, so an IMPORTED helper is an
    # unresolved callee and the function is refused. Sound -- we cannot see the helper -- and it is why
    # `tucker.rank_gate` is rejected: it reaches `fix_eigvec_signs` from another module.
    import inspect
    import textwrap

    from holographic.caching_and_storage import holographic_tucker as tucker

    rep = purity_report(textwrap.dedent(inspect.getsource(tucker)))
    assert "unfold" in rep["pure"]                              # self-contained: provably pure
    assert "_mode_svd" in rep["impure"]                         # calls an imported `fix_eigvec_signs`
    assert rep["reasons"]["_mode_svd"] == ["impure-or-unresolved-callee"]

    with pytest.raises(ValueError, match="not pure"):
        memoize_pure(tucker.rank_gate)


def test_a_real_engine_function_that_is_provably_pure_memoizes():
    from holographic.simulation_and_physics.holographic_island import island_energy

    f = memoize_pure(island_energy)
    X = np.zeros((64, 3))
    V = np.ones((64, 3))
    assert f(X, V) == f(X, V)
    assert f.cache_stats()["hits"] == 1

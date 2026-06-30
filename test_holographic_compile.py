"""Tests for the content-addressed runtime compile cache (COMPILE-1)."""
import numpy as np
from holographic_compile import CompileCache, compiled, compiled_sdf_normal, DEFAULT_CACHE, _canonical
from holographic_codegen import HAS_SYMPY


def test_key_deterministic_and_content_addressed():
    c = CompileCache()
    assert c.key("sphere") == c.key("sphere")                  # same source -> same key
    assert c.key("sphere") != c.key("torus")                   # different source -> different key
    assert c.key(("a", (1, 2), {"k": 3})) == c.key(("a", (1, 2), {"k": 3}))
    assert _canonical(np.zeros(3)) == _canonical(np.zeros(3))   # arrays canonicalise by content


def test_compiles_once_reused_many():
    n = {"c": 0}
    def comp(src):
        n["c"] += 1
        return lambda x: x + len(src)
    c = CompileCache()
    for _ in range(30):
        fn = c.get_or_compile("spec", comp)
    assert n["c"] == 1 and c.stats["hits"] == 29 and c.stats["compiles"] == 1
    assert fn(0) == len("spec")


def test_recompiles_on_change():
    n = {"c": 0}
    def comp(src):
        n["c"] += 1
        return src
    c = CompileCache()
    c.get_or_compile("a", comp); c.get_or_compile("a", comp); c.get_or_compile("b", comp)
    assert n["c"] == 2                                          # "a" once (reused), "b" once


def test_lru_eviction_bounds_memory():
    c = CompileCache(maxsize=3)
    for i in range(10):
        c.get_or_compile(f"s{i}", lambda s: s)
    assert len(c) == 3 and c.stats["evictions"] == 7


def test_invalidate_forces_recompile():
    n = {"c": 0}
    def comp(src):
        n["c"] += 1
        return src
    c = CompileCache()
    c.get_or_compile("x", comp)
    assert c.invalidate("x", comp) is True
    c.get_or_compile("x", comp)
    assert n["c"] == 2


def test_compiled_sdf_normal_cached_and_correct():
    if not HAS_SYMPY:
        return
    DEFAULT_CACHE.clear()
    v1, n1 = compiled_sdf_normal("sqrt(x**2+y**2+z**2) - 1.0")
    v2, n2 = compiled_sdf_normal("sqrt(x**2+y**2+z**2) - 1.0")
    assert n1 is n2                                             # same artifact object reused (cache hit)
    P = np.random.default_rng(0).standard_normal((20, 3)) * 1.4
    analytic = P / np.linalg.norm(P, axis=1, keepdims=True)
    assert np.max(np.abs(n1(P) - analytic)) < 1e-12
    compiled_sdf_normal("sqrt(x**2+y**2+z**2) - 2.0")          # different expr -> recompile
    assert DEFAULT_CACHE.stats["compiles"] >= 2

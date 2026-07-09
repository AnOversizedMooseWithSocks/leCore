"""Tests for the two cache-backed compilers: SymPy->Numba SDF + VSA program assembler (COMPILE-2).
Numba/sympy parts skip cleanly when those optional deps are absent."""
import numpy as np
from holographic.scene_and_pipeline.holographic_compile import compiled_sdf_numba, compiled_program, DEFAULT_CACHE
from holographic.misc.holographic_codegen import HAS_SYMPY
from holographic.misc.holographic_jit import HAS_NUMBA
from holographic.agents_and_reasoning.holographic_machine import HoloMachine


def test_sdf_numba_matches_analytic_and_caches():
    if not (HAS_SYMPY and HAS_NUMBA):
        return
    DEFAULT_CACHE.clear()
    d1 = compiled_sdf_numba("sqrt(x**2+y**2+z**2) - 1.3")
    d2 = compiled_sdf_numba("sqrt(x**2+y**2+z**2) - 1.3")
    assert d1 is d2                                            # cached -> same compiled kernels reused
    P = np.random.default_rng(0).standard_normal((40, 3)) * 1.5
    analytic = P / np.linalg.norm(P, axis=1, keepdims=True)
    assert np.allclose(d1["grid_normal"](P), analytic, atol=1e-10)
    assert np.allclose(d1["grid_value"](P), np.linalg.norm(P, axis=1) - 1.3, atol=1e-10)


def test_njit_sdf_composes_into_njit_march():
    if not (HAS_SYMPY and HAS_NUMBA):
        return
    from numba import njit
    fv = compiled_sdf_numba("sqrt(x**2+y**2+z**2) - 1.3")["scalar_value"]

    @njit
    def march(oz, dz, steps=64):
        t = 0.0
        for _ in range(steps):
            d = fv(0.0, 0.0, oz + t * dz)
            if d < 1e-4:
                return t
            t += d
        return -1.0
    assert abs(march(-5.0, 1.0) - 3.7) < 0.05                 # njit march calling njit SDF hits R=1.3 at ~3.7


def test_sdf_numba_recompiles_on_change():
    if not (HAS_SYMPY and HAS_NUMBA):
        return
    DEFAULT_CACHE.clear()
    compiled_sdf_numba("sqrt(x**2+y**2+z**2) - 1.0")
    compiled_sdf_numba("sqrt(x**2+y**2+z**2) - 2.0")          # different SDF -> recompile
    assert DEFAULT_CACHE.stats["compiles"] >= 2


def test_program_assemble_cached_and_correct():
    m = HoloMachine(dim=512, seed=0)
    names = m.data_names[:4] if getattr(m, "data_names", None) else ["a", "b", "c", "d"]
    prog = [("BIND", names[i % len(names)]) for i in range(40)]
    DEFAULT_CACHE.clear()
    pv1 = compiled_program(m, prog)
    pv2 = compiled_program(m, prog)
    assert pv1 is pv2                                          # cached program vector reused
    assert np.array_equal(pv1, m.assemble(prog))              # cached == fresh assemble (byte-identical)
    assert DEFAULT_CACHE.stats["hits"] >= 1 and DEFAULT_CACHE.stats["compiles"] == 1


def test_program_recompiles_when_program_changes():
    m = HoloMachine(dim=512, seed=0)
    DEFAULT_CACHE.clear()
    compiled_program(m, [("BIND", "a")])
    compiled_program(m, [("BIND", "b")])                      # different program -> recompile
    assert DEFAULT_CACHE.stats["compiles"] == 2

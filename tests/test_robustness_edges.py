"""Panel edge sweep pins: SEMANTICS at the edges, not mere no-throw (the first sweep scored
12/12 'OK' while a NaN query returned a hallucinated match and a NaN map certified at '0.0
residual' via Python's max(0.0, nan) keeping 0.0 -- perfect scores are instrument hypotheses)."""
import numpy as np
import pytest

from holographic.caching_and_storage.holographic_index import Index
from holographic.io_and_interop.holographic_projector import probe_project
from holographic.agents_and_reasoning.holographic_compileinstall import compile_installed
from holographic.agents_and_reasoning.holographic_machine import HoloMachine


def _x(n=50, d=8):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_nan_query_refuses_loudly():
    with pytest.raises(ValueError):
        Index(_x(), method="exact", seed=0).nearest(np.full(8, np.nan), k=1)


def test_nan_map_refuses_with_infinite_residual():
    p = probe_project(lambda v: v * np.nan, 8)
    assert p["kind"] == "refused" and p["residual"] == float("inf")


def test_k_larger_than_n_returns_all_n():
    r = Index(_x(3), method="exact", seed=0).nearest(_x(3)[0], k=10)
    assert len(r) == 3


def test_single_item_index():
    r = Index(_x(1), method="exact", seed=0).nearest(_x(1)[0], k=1)
    assert r and r[0][0] == 0


def test_dim_one():
    X = np.random.default_rng(1).standard_normal((20, 1))
    assert Index(X, method="exact", seed=0).nearest(np.ones(1), k=2)


def test_constant_rows_tie_lowest_index():
    r = Index(np.ones((10, 8)), method="exact", seed=0).nearest(np.ones(8), k=3)
    assert [i for i, _ in r] == [0, 1, 2]


def test_nan_fac_step_refuses_at_compile():
    mach = HoloMachine(dim=8, seed=1, data=["a"])
    mach.functions_symbolic = {}
    with pytest.raises(ValueError):
        compile_installed(mach, [("LOAD", "a"), ("FAC", ("bad", lambda v: v * np.nan)),
                                 ("HALT", None)])


def test_empty_program_compiles_to_none_state():
    mach = HoloMachine(dim=8, seed=1, data=["a"])
    mach.functions_symbolic = {}
    run, man = compile_installed(mach, [("HALT", None)])
    assert run() is None and man["chain"] == []


def test_fast_abstain_decisions_identical():
    # V2: fast=True fills non-candidate scores with -inf; abstention must be unaffected
    rng = np.random.default_rng(0)
    V = rng.standard_normal((1500, 64)); V /= np.linalg.norm(V, axis=1, keepdims=True)
    ref = Index(V, method="exact", seed=0)
    fas = Index(V, method="exact", seed=0, fast=True)
    noise = V[rng.permutation(1500)[:80]].copy()
    for r in noise:
        rng.shuffle(r)
    for alpha in (0.01, 0.05):
        assert [bool(ref.nearest(q, k=1, abstain=alpha)) for q in noise] == \
               [bool(fas.nearest(q, k=1, abstain=alpha)) for q in noise]


def test_fast_tiny_n_and_k_ge_n():
    # V1/V7: N smaller than the shortlist; k >= N
    rng = np.random.default_rng(1)
    V = rng.standard_normal((30, 16)); V /= np.linalg.norm(V, axis=1, keepdims=True)
    f = Index(V, method="exact", seed=0, fast=True)
    e = Index(V, method="exact", seed=0)
    assert [i for i, _ in f.nearest(V[3], k=5)] == [i for i, _ in e.nearest(V[3], k=5)]
    assert len(f.nearest(V[0], k=99)) == 30
    with pytest.raises(ValueError):
        f.nearest(np.full(16, np.nan))


def test_collapse_edge_steps():
    # V8: n=0 identity, n=1 one step, negative n refused toward the time machine
    from holographic.agents_and_reasoning.holographic_compileinstall import collapse_recurrence
    mach = HoloMachine(dim=6, seed=1, data=["a"])
    mach.functions_symbolic = {}
    step = [("FAC", ("d", lambda f: 0.9 * f + 0.01)), ("HALT", None)]
    r0, _ = collapse_recurrence(mach, step, 0)
    assert np.max(np.abs(r0(np.ones(6)) - np.ones(6))) == 0.0
    r1, _ = collapse_recurrence(mach, step, 1)
    assert np.max(np.abs(r1(np.ones(6)) - (0.9 * np.ones(6) + 0.01))) < 1e-15
    with pytest.raises(ValueError):
        collapse_recurrence(mach, step, -3)


def test_byteplane_nan_inf_empty():
    # V3: byte-exactness is a BYTE claim -- NaN/Inf/-0.0 and empty arrays included
    from holographic.io_and_interop.holographic_byteplane import (float_pack_bytes,
                                                                  float_unpack_bytes)
    A = np.array([[1.0, np.nan], [np.inf, -0.0]], dtype=np.float32)
    B = float_unpack_bytes(float_pack_bytes(A))
    assert B.tobytes() == A.tobytes()
    E = np.empty((0, 4))
    assert float_unpack_bytes(float_pack_bytes(E)).shape == (0, 4)


def test_timemachine_odd_dim_and_mismatch():
    # V11: the inferred-dim bug silently returned D-1; the state owns its dim now
    from holographic.simulation_and_physics.holographic_timemachine import (make_unitary_step,
                                                                            time_jump)
    spec = make_unitary_step(129, seed=2)
    x = np.random.default_rng(3).standard_normal(129)
    back = time_jump(time_jump(x, spec, 21), spec, -21)
    assert back.shape == (129,) and np.max(np.abs(back - x)) < 1e-12
    with pytest.raises(ValueError):
        time_jump(x, make_unitary_step(64, seed=1), 3)

"""Regression traps for the VM decoded-instruction cache (holographic_vmplan).

These assert the CONTRACT, not the wish: the plan exists only because it is indistinguishable from the
unplanned interpreter. Anything that makes it faster at the cost of a different answer must fail here.
"""
import numpy as np
import pytest

from holographic.agents_and_reasoning.holographic_machine import HoloMachine
from holographic.agents_and_reasoning.holographic_vmplan import DecodePlan, program_key
from holographic.sampling_and_signal.holographic_fft import rfft as _rfft, irfft as _irfft

# A corpus that reaches every opcode family: values, registers, the stack, both loops, the branch, the host call.
PROGRAMS = [
    [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("PERMUTE", None), ("BIND", "d"), ("HALT", None)],
    [("LOAD", "a"), ("STORE", "R3"), ("LOAD", "d"), ("BIND", "e"), ("RECALL", "R3"), ("HALT", None)],
    [("LOAD", "b"), ("IFMATCH", "b"), ("BIND", "c"), ("PUSH", None), ("LOAD", "d"), ("POP", None), ("HALT", None)],
    [("LOAD", "a"), ("REPEAT", 3), ("CALL", "body"), ("BUNDLE", "c"), ("HALT", None)],
    [("LOAD", "a"), ("CALL", "body"), ("BIND", "c"), ("HALT", None)],
    [("LOAD", "a"), ("ITERATE", "body"), ("HALT", None)],
    [("LOAD", "a"), ("APPLY", "cleanup"), ("BIND", "b"), ("HALT", None)],
]


def _machine(dim, seed, **kw):
    m = HoloMachine(dim=dim, seed=seed, **kw)
    m.define("body", [("BIND", "b"), ("PERMUTE", None), ("HALT", None)])
    return m


@pytest.mark.parametrize("dim", [256, 1024])
@pytest.mark.parametrize("seed", [3, 7])
@pytest.mark.parametrize("fast_cleanup", [False, True])
def test_planned_run_is_bit_identical_to_unplanned(dim, seed, fast_cleanup):
    """The load-bearing claim. Same accumulator BYTES and the same trace, on every program in the corpus.

    array_equal, not allclose: the VM's decode is cleanup-gated and its accumulator feeds tie-sensitive
    consumers, so 'close enough' is not the contract we are selling."""
    plain = _machine(dim, seed, fast_cleanup=fast_cleanup)
    planned = _machine(dim, seed, fast_cleanup=fast_cleanup, decode_plan=True)
    handlers = {"cleanup": lambda v: v / (np.linalg.norm(v) + 1e-12)}
    for prog in PROGRAMS:
        pv = plain.assemble(prog)
        assert np.array_equal(pv, planned.assemble(prog)), "assembly itself diverged -- the fixture is broken"
        a1, t1 = plain.run(pv, max_loop=12, handlers=handlers)
        a2, t2 = planned.run(pv, max_loop=12, handlers=handlers)
        assert t1 == t2, f"trace diverged on {prog[:2]}: {t1} vs {t2}"
        assert (a1 is None and a2 is None) or np.array_equal(a1, a2), f"accumulator diverged on {prog[:2]}"


@pytest.mark.parametrize("dim", [256, 1024])
def test_plan_decode_matches_scalar_decode_including_past_halt(dim):
    """Addresses PAST the end of the program decode to pure bundling crosstalk. That is where a
    disagreement between the batched and scalar classifiers would surface first, so it is exactly where
    we look -- a decode trap that only checks valid addresses is checking the easy case."""
    m = _machine(dim, 7)
    plan = DecodePlan(m)
    for prog in PROGRAMS:
        pv = m.assemble(prog)
        for i in range(len(prog) + 6):
            assert plan.at(pv, i) == m.decode_instruction(pv, i), f"addr {i} of {prog[:2]} disagrees"


def test_batched_address_read_is_bit_identical_to_scalar():
    """The spectral half of the claim, isolated: a batched inverse transform against resident address
    keys returns the SAME BYTES as the scalar _read_addr. If numpy ever stops guaranteeing that, this
    fails here rather than silently downgrading the equivalence above to 'usually'."""
    m = _machine(1024, 7)
    plan = DecodePlan(m)
    pv = m.assemble(PROGRAMS[0])
    spec = _rfft(pv)
    batched = _irfft(spec[None, :] * plan._addr_stack(0, 6), n=1024, axis=-1)
    for i in range(6):
        assert np.array_equal(batched[i], m._read_addr(spec, i, 1024)), f"batched read differs at address {i}"


def test_the_work_happens_once_not_once_per_visit():
    """The reason the module exists. A loop that revisits addresses must not re-sweep."""
    m = _machine(1024, 7, decode_plan=True)
    pv = m.assemble([("LOAD", "a"), ("ITERATE", "body"), ("HALT", None)])
    m.run(pv, max_loop=64)
    st = m.plan().stats()
    assert st["sweeps"] <= 4, f"a 64-iteration loop caused {st['sweeps']} spectral sweeps -- the cache is not holding"
    assert st["hit_rate"] > 0.9, f"hit rate {st['hit_rate']:.3f} -- CALL bodies are missing the content key"


def test_content_addressing_not_identity():
    """Two byte-identical program vectors built as SEPARATE arrays must share one cache entry, and two
    different programs must not. This is the property that makes CALL/ITERATE hit at all: each iteration
    rebuilds the callee body as a fresh array with a fresh id."""
    m = _machine(512, 7)
    plan = DecodePlan(m)
    pv1 = m.assemble(PROGRAMS[0])
    pv2 = m.assemble(PROGRAMS[0])          # same bytes, different object
    assert pv1 is not pv2 and program_key(pv1) == program_key(pv2)
    plan.at(pv1, 0)
    before = plan.sweeps
    plan.at(pv2, 0)
    assert plan.sweeps == before, "a byte-identical program re-swept -- content addressing is broken"
    other = m.assemble(PROGRAMS[1])
    assert plan.at(other, 1) == ("STORE", "R3"), "a distinct program was served from another program's entry"


def test_define_invalidates_the_caches():
    """Defining a function moves the library AND widens the fn_atoms codebook, either of which can
    legitimately change a decode. A cache that survived define() would serve a stale answer."""
    m = _machine(512, 7, decode_plan=True)
    pv = m.assemble([("LOAD", "a"), ("CALL", "body"), ("HALT", None)])
    m.run(pv)
    assert m.plan().stats()["programs"] > 0
    m.define("other", [("BUNDLE", "c"), ("HALT", None)])
    assert m.plan().stats()["programs"] == 0, "define() did not clear the decode cache"
    assert m._body_cache == {}, "define() did not clear the extracted-body cache"
    # and the machine still runs correctly afterwards, against BOTH library entries
    plain = _machine(512, 7)
    plain.define("other", [("BUNDLE", "c"), ("HALT", None)])
    a1, t1 = plain.run(plain.assemble([("LOAD", "a"), ("CALL", "other"), ("HALT", None)]))
    a2, t2 = m.run(m.assemble([("LOAD", "a"), ("CALL", "other"), ("HALT", None)]))
    assert t1 == t2 and np.array_equal(a1, a2)


def test_mind_faculty_round_trip():
    """The faculty is real: reachable from a mind, togglable, and identical to the unplanned mind."""
    import lecore
    prog = [("LOAD", "a"), ("ITERATE", "dbl"), ("HALT", None)]
    plain = lecore.UnifiedMind(dim=512, seed=0)
    plain.learn_procedure("dbl", [("BIND", "b"), ("HALT", None)])
    planned = lecore.UnifiedMind(dim=512, seed=0)
    planned.learn_procedure("dbl", [("BIND", "b"), ("HALT", None)])
    assert planned.vm_plan_stats() is None, "the plan must be OFF by default (the never-flip rule)"
    planned.vm_decode_plan(True)
    a1, t1 = plain.run_procedure(prog)
    a2, t2 = planned.run_procedure(prog)
    assert t1 == t2 and np.array_equal(a1, a2)
    assert planned.vm_plan_stats()["sweeps"] <= 4
    planned.vm_decode_plan(False)
    assert planned.vm_plan_stats() is None


@pytest.mark.parametrize("dim,n_rows", [(512, 8), (1024, 32)])
def test_run_batch_is_bit_identical_under_the_plan(dim, n_rows):
    """The two amortisations must compose without changing an answer: run_batch already decodes once ACROSS
    THE BATCH, the plan makes it free ACROSS THE CALLS. Measured 1.6x-6.1x; the speedup correctly FALLS as
    N grows (at large N the batch arithmetic dominates and there is less decode left to remove), which is the
    honest shape of the curve rather than a defect."""
    prog = [("LOAD", "a"), ("BIND", "b"), ("STORE", "R1"), ("BUNDLE", "c"),
            ("PERMUTE", None), ("RECALL", "R1"), ("BIND", "d"), ("HALT", None)]
    plain = _machine(dim, 7)
    planned = _machine(dim, 7, decode_plan=True)
    init = np.random.default_rng(0).standard_normal((n_rows, dim))
    pv = plain.assemble(prog)
    assert np.array_equal(plain.run_batch(pv, init), planned.run_batch(pv, init))


def test_decode_instruction_is_never_served_from_the_plan():
    """decode_instruction is the REFERENCE ORACLE this whole module is tested against. If it ever starts
    routing through the plan, every equivalence assertion above silently becomes a comparison of the plan
    with itself. This test is the tripwire for that specific well-meaning refactor."""
    m = _machine(512, 7, decode_plan=True)
    pv = m.assemble(PROGRAMS[0])
    m.plan().clear()
    m.decode_instruction(pv, 0)
    assert m.plan().stats()["sweeps"] == 0, "decode_instruction populated the plan -- the oracle is no longer independent"

"""Holographic stored-program machine: programs encoded as hypervectors, executed by VSA ops."""
import numpy as np
from holographic.agents_and_reasoning.holographic_ai import bind, bundle, cosine, permute
from holographic.agents_and_reasoning.holographic_machine import HoloMachine


def test_program_executes_exactly():
    m = HoloMachine(dim=4096, seed=7)
    prog = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("HALT", "")]
    acc, trace = m.run(m.assemble(prog))
    expected = bundle([bind(m.data_atoms["a"], m.data_atoms["b"]), m.data_atoms["c"]])
    assert cosine(acc, expected) > 0.99
    assert trace == [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c")]


def test_halt_stops_execution():
    m = HoloMachine(dim=2048, seed=3)
    prog = [("LOAD", "a"), ("HALT", ""), ("BIND", "b")]   # the BIND must never run
    acc, trace = m.run(m.assemble(prog))
    assert trace == [("LOAD", "a")]
    assert cosine(acc, m.data_atoms["a"]) > 0.99


def test_permute_op():
    m = HoloMachine(dim=2048, seed=1)
    acc, _ = m.run(m.assemble([("LOAD", "a"), ("PERMUTE", ""), ("HALT", "")]))
    assert cosine(acc, permute(m.data_atoms["a"], 1)) > 0.99


def test_modest_program_decodes_fully():
    # a 32-instruction program at dim 4096 reads back perfectly (well under the capacity cliff).
    import random
    m = HoloMachine(dim=4096, seed=5)
    rng = random.Random(0)
    prog = [(rng.choice(["LOAD", "BIND", "BUNDLE", "PERMUTE"]), rng.choice(m.data_names)) for _ in range(32)]
    pv = m.assemble(prog)
    assert all(m.decode_instruction(pv, i) == prog[i] for i in range(32))


def test_inception_clean_nesting_is_deep():
    # a program nested as the ONLY file at each level survives many levels (near-lossless bind chain).
    m = HoloMachine(dim=4096, seed=7)
    base = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("HALT", "")]
    want = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c")]
    v = m.assemble(base)
    for _ in range(6):
        v = m.disk(v)                       # wrap with no other files
    for _ in range(6):
        v = m.open_slot(v)
    _, trace = m.run(v)
    assert trace == want


def test_inception_busy_disk_has_a_depth_floor():
    # KEPT NEGATIVE / the law: with other files on each disk, a buried program corrupts with depth.
    m = HoloMachine(dim=4096, seed=7)
    base = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("HALT", "")]
    want = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c")]

    def nest(depth):
        v = m.assemble(base)
        for d in range(depth):
            v = m.disk(v, m.junk_files(3, d))
        for d in range(depth):
            v = m.open_slot(v)
        return m.run(v)[1]

    assert nest(2) == want          # shallow nesting on a busy disk still works
    assert nest(8) != want          # deep nesting on a busy disk does not -- the floor is real


def test_call_runs_a_library_function():
    # a function embedded in the holographic library, invoked by name, transforms the accumulator.
    m = HoloMachine(dim=4096, seed=7)
    m.define("tag_b", [("BIND", "b"), ("HALT", "")])     # an ACC->ACC function: ACC = bind(ACC, b)
    acc, trace = m.run(m.assemble([("LOAD", "a"), ("CALL", "tag_b"), ("HALT", "")]))
    assert ("CALL", "tag_b") in trace
    assert cosine(acc, bind(m.data_atoms["a"], m.data_atoms["b"])) > 0.99


def test_call_composes_library_functions():
    # two functions from ONE library vector compose like ordinary code.
    m = HoloMachine(dim=4096, seed=7)
    m.define("tag_b", [("BIND", "b"), ("HALT", "")])
    m.define("shift", [("PERMUTE", ""), ("HALT", "")])
    acc, _ = m.run(m.assemble([("LOAD", "a"), ("CALL", "tag_b"), ("CALL", "shift"), ("HALT", "")]))
    from holographic.agents_and_reasoning.holographic_ai import permute as _p
    assert cosine(acc, _p(bind(m.data_atoms["a"], m.data_atoms["b"]), 1)) > 0.99


def test_library_is_one_vector():
    # the whole function library is a single hypervector addressable by name.
    m = HoloMachine(dim=2048, seed=2)
    m.define("f", [("BIND", "b"), ("HALT", "")])
    m.define("g", [("PERMUTE", ""), ("HALT", "")])
    assert m.library.shape == (2048,)


def test_run_backward_compatible_without_call():
    # programs that don't use CALL behave exactly as before (init_acc defaults to None).
    m = HoloMachine(dim=4096, seed=7)
    acc, trace = m.run(m.assemble([("LOAD", "a"), ("BIND", "b"), ("HALT", "")]))
    assert trace == [("LOAD", "a"), ("BIND", "b")]


def _bindchain(m, names):
    """The accumulator a LOAD; BIND...; program over `names` should produce -- the directly-computed bind chain."""
    acc = m.data_atoms[names[0]]
    for n in names[1:]:
        acc = bind(acc, m.data_atoms[n])
    return acc


def test_run_chunked_executes_a_program_past_the_single_program_cap():
    # A 60-instruction program (well past the ~20-32 single-program cap at dim 1024) decodes to garbage as one
    # structure but runs EXACTLY via run_chunked, which threads the accumulator across clean <=14-instr chunks.
    m = HoloMachine(dim=1024, seed=7)
    names = [chr(ord("a") + (i % 6)) for i in range(60)]
    prog = [("LOAD", names[0])] + [("BIND", names[i]) for i in range(1, 60)] + [("HALT", "")]
    expected = _bindchain(m, names)
    flat, _ = m.run(m.assemble(prog))
    chunked, trace = m.run_chunked(prog)                      # default chunk=14
    assert cosine(flat, expected) < 0.5                       # one structure overstuffed -> the cliff
    assert cosine(chunked, expected) > 0.999                 # host-threaded chunks -> exact, past the cap
    assert len(trace) == 60                                  # every instruction executed (LOAD + 59 BIND)


def test_run_chunked_matches_run_on_a_short_program():
    # within the cap, run_chunked must equal run() (one chunk, no threading) -- backward compatible.
    m = HoloMachine(dim=4096, seed=7)
    prog = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("HALT", "")]
    assert cosine(m.run(m.assemble(prog))[0], m.run_chunked(prog)[0]) > 0.9999


def test_run_chunked_keeps_constructs_intact_and_stops_on_mid_halt():
    m = HoloMachine(dim=1024, seed=7)
    m.define("tag_b", [("BIND", "b"), ("HALT", "")])
    # IFMATCH gates the next instruction: even forced to a chunk=1 seam, the gate+target stay in one chunk.
    gated = [("LOAD", "a"), ("IFMATCH", "a"), ("CALL", "tag_b"), ("HALT", "")]
    assert cosine(m.run_chunked(gated, chunk=1)[0], bind(m.data_atoms["a"], m.data_atoms["b"])) > 0.999
    # a HALT in the MIDDLE stops the whole run, like the flat interpreter -- the trailing BIND must never run.
    mid = [("LOAD", "a"), ("BIND", "b"), ("HALT", ""), ("BIND", "c")]
    acc, trace = m.run_chunked(mid, chunk=2)
    assert trace == [("LOAD", "a"), ("BIND", "b")]
    assert cosine(acc, bind(m.data_atoms["a"], m.data_atoms["b"])) > 0.999


def test_call_library_does_NOT_chunk_a_long_program_kept_negative():
    # KEPT NEGATIVE: the obvious "factor into functions and CALL them" does NOT get past the cap -- CALL pulls
    # each sub-program from a BUNDLED library, and bundling several function-vectors re-introduces the cliff.
    # This is exactly why run_chunked threads independent chunk-vectors in the host instead. Documented, not hidden.
    m = HoloMachine(dim=1024, seed=7)
    names = [chr(ord("a") + (i % 6)) for i in range(60)]
    expected = _bindchain(m, names)
    for c, lo in enumerate(range(1, 60, 10)):
        m.define(f"k{c}", [("BIND", names[i]) for i in range(lo, min(lo + 10, 60))] + [("HALT", "")])
    driver = [("LOAD", names[0])] + [("CALL", f"k{c}") for c in range(6)] + [("HALT", "")]
    via_call, _ = m.run(m.assemble(driver))
    assert cosine(via_call, expected) < 0.5                  # the library bundle corrupts -> CALL is the wrong tool
    assert cosine(m.run_chunked(names and [("LOAD", names[0])] + [("BIND", names[i]) for i in range(1, 60)] + [("HALT", "")])[0],
                  expected) > 0.999                          # run_chunked, on the same program, succeeds


# --- WIRE-2: cross-chunk register/stack threading + the composable state continuation -------------
def _state_machine():
    from holographic.agents_and_reasoning.holographic_machine import HoloMachine
    return HoloMachine(dim=1024, seed=7, data=["a", "b", "c", "d", "e", "f", "g", "h"])


def test_run_chunked_threads_registers_across_a_seam():
    from holographic.agents_and_reasoning.holographic_ai import cosine
    M = _state_machine()
    # STORE R0 lands in chunk 1; RECALL R0 lands in a later chunk (chunk=4 forces the seam between them)
    prog = [("LOAD", "a"), ("STORE", "R0")] + [("LOAD", "b"), ("PERMUTE", "")] * 6 + [("RECALL", "R0"), ("HALT", "")]
    out, _ = M.run_chunked(prog, chunk=4)
    assert cosine(out, M.data_atoms["a"]) > 0.999            # the register survived the chunk boundary


def test_run_chunked_threads_stack_across_a_seam():
    from holographic.agents_and_reasoning.holographic_ai import cosine
    M = _state_machine()
    prog = [("LOAD", "a"), ("PUSH", "")] + [("LOAD", "c"), ("BIND", "d")] * 5 + [("POP", ""), ("HALT", "")]
    out, _ = M.run_chunked(prog, chunk=4)
    assert cosine(out, M.data_atoms["a"]) > 0.999            # PUSH in one chunk, POP in a later one restores a


def test_run_return_state_is_backward_compatible():
    M = _state_machine()
    pv = M.assemble([("LOAD", "a"), ("STORE", "R0"), ("HALT", "")])
    two = M.run(pv)                                          # default: the original 2-tuple
    assert len(two) == 2
    four = M.run(pv, return_state=True)                      # opt-in: full state
    assert len(four) == 4 and "R0" in four[2]


def test_state_continuation_roundtrips_exact_for_atoms():
    import numpy as np
    M = _state_machine()
    acc = M.data_atoms["a"]
    regs = {"R0": M.data_atoms["b"], "R1": M.data_atoms["c"], "R2": M.data_atoms["d"]}
    snap = M.state_to_vector(acc, regs)                     # one composable vector
    cb = list(M.data_atoms.values())
    racc, rregs, _ = M.state_from_vector(snap, reg_names=["R0", "R1", "R2"], codebook=cb)
    from holographic.agents_and_reasoning.holographic_ai import cosine
    assert cosine(racc, acc) > 0.999
    assert all(np.allclose(rregs[r], regs[r]) for r in ("R0", "R1", "R2"))


def test_state_continuation_crosstalk_grows_with_slots():
    # honest negative: raw (pre-cleanup) readback degrades as more slots are bundled (the capacity cliff)
    import numpy as np
    from holographic.agents_and_reasoning.holographic_ai import unbind, cosine
    M = _state_machine()
    raws = []
    for k in (2, 8):
        rr = {f"R{i}": M.data_atoms[M.data_names[i]] for i in range(k)}
        sv = M.state_to_vector(M.data_atoms["a"], rr)
        raws.append(np.mean([float(cosine(unbind(sv, M.reg_atoms[r]), rr[r])) for r in rr]))
    assert raws[0] > raws[1]                                 # more slots -> noisier raw read (why per-seam is exact)


def test_run_chunked_records_a_replay_log():
    from holographic.agents_and_reasoning.holographic_ai import cosine
    M = _state_machine()
    prog = [("LOAD", "a"), ("STORE", "R0")] + [("LOAD", "b"), ("PERMUTE", "")] * 6 + [("RECALL", "R0"), ("HALT", "")]
    acc, trace, states = M.run_chunked(prog, chunk=4, record=True)
    assert len(states) >= 2                                        # one state per seam
    assert states[0].shape == (10, M.dim)                         # acc + 8 registers + stack as rows
    assert cosine(acc, M.data_atoms["a"]) > 0.999                # final acc still correct


def test_run_batch_matches_per_item_run():
    import numpy as np
    from holographic.agents_and_reasoning.holographic_ai import cosine
    M = _state_machine()
    prog = [("BIND", "a"), ("PERMUTE", ""), ("BUNDLE", "b"), ("BIND", "c"),
            ("STORE", "R0"), ("PERMUTE", ""), ("BIND", "a"), ("RECALL", "R0"), ("HALT", "")]
    pv = M.assemble(prog)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, M.dim)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    batch = M.run_batch(pv, X)
    per_item = np.stack([M.run(pv, init_acc=X[i])[0] for i in range(len(X))])
    assert batch.shape == X.shape
    assert np.allclose(batch, per_item, atol=1e-8)                 # matches scalar VM to machine epsilon


def test_run_batch_rejects_control_ops():
    import numpy as np, pytest
    M = _state_machine()
    pv = M.assemble([("IFMATCH", "a"), ("BIND", "b"), ("HALT", "")])
    with pytest.raises(ValueError):
        M.run_batch(pv, np.zeros((4, M.dim)))                      # control flow can't batch -> clear error


def test_residency_decode_bit_exact():
    """The machine's residency address-read (rfft(program_vec) computed once and reused) is BIT-IDENTICAL to
    unbind(program_vec, pos(i)) -- so it cannot flip a cleanup-gated decode winner (the whole safety argument
    for wiring Fill 1 into the VM's hottest loop)."""
    import numpy as np
    from holographic.sampling_and_signal.holographic_fft import rfft
    from holographic.agents_and_reasoning.holographic_ai import unbind
    m = HoloMachine(dim=512, seed=0)
    prog = m.assemble([("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("HALT", None)])
    prog_spec = rfft(prog); n = prog.shape[0]
    for i in range(4):
        assert np.array_equal(m._read_addr(prog_spec, i, n), unbind(prog, m.pos(i)))   # exact, not tolerance


def test_fast_cleanup_and_atom_cache_change_nothing_but_time():
    """Two VM optimisations from the virtual-GPU capability audit, both pinned as EXACT-equivalence claims:
    (1) the atom cache -- derived_atom is pure in (seed, name, dim), so memoising it is bit-identical by
        construction (the profile showed the VM re-deriving pos:i atoms every decode: 940 FFTs per 90
        instructions, the REAL hot path; the cleanup loop was only ~12%);
    (2) fast_cleanup=True -- one cached-codebook matmul + argmax instead of a Python loop of cosine calls.
        Both take the FIRST maximum, hammered with noisy + exact-tie decodes; still opt-in under the QEM rule.
    MEASURED (dim 4096, 9-instruction program): 10.6 -> 8.1 ms default (cache alone), 5.2 ms with fast_cleanup."""
    import numpy as np
    from holographic.agents_and_reasoning.holographic_machine import HoloMachine

    prog = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("PERMUTE", None),
            ("STORE", "R1"), ("RECALL", "R1"), ("BIND", "d"), ("BUNDLE", "e"), ("HALT", None)]
    slow = HoloMachine(dim=1024, seed=0)
    fast = HoloMachine(dim=1024, seed=0, fast_cleanup=True)

    vs, vf = slow.assemble(prog), fast.assemble(prog)
    assert np.array_equal(vs, vf), "assembly must be identical -- the flag touches DECODE only"

    def res(x):
        return np.asarray(x[0] if isinstance(x, tuple) else x)
    assert np.array_equal(res(slow.run(vs, max_steps=9)), res(fast.run(vf, max_steps=9))), \
        "fast_cleanup must be result-identical on a full program"

    # tie agreement, hammered: noisy decodes at several noise levels + an exact two-atom tie
    codebook = dict(slow.op_atoms)
    rng = np.random.default_rng(0)
    for trial in range(300):
        i = int(rng.integers(len(slow.data_names)))
        noisy = slow.data_atoms[slow.data_names[i]] + rng.standard_normal(1024) * float(rng.choice([0.1, 0.7, 1.5]))
        assert slow._nearest_loop(slow.data_atoms, noisy) == fast._nearest_fast(fast.data_atoms, noisy)
    two = slow.op_atoms["LOAD"] + slow.op_atoms["BIND"]        # an exact tie between two opcode atoms
    assert slow._nearest_loop(slow.op_atoms, two) == fast._nearest_fast(fast.op_atoms, two)

    # the atom cache really is a cache (same object back), and a cached atom equals a fresh derivation
    from holographic.agents_and_reasoning.holographic_ai import derived_atom
    a1 = slow._atom("pos:3", unitary=True)
    a2 = slow._atom("pos:3", unitary=True)
    assert a1 is a2, "second call must hit the cache"
    assert np.array_equal(a1, derived_atom(slow.seed, "pos:3", slow.dim, unitary=True)), \
        "the cached atom must equal a fresh derivation bit-for-bit"


def test_fast_cleanup_tie_band_survives_any_blas():
    """The CI-sampled knife edge, pinned structurally: EVERY pairwise exact tie in the opcode codebook (plus
    sub-epsilon scale jitters) must agree between the loop and the fast path -- the band re-arbitration makes
    this true by construction, so this test passes on any BLAS, not just the one we developed on."""
    import numpy as np
    from holographic.agents_and_reasoning.holographic_machine import HoloMachine
    slow = HoloMachine(dim=512, seed=0)
    fast = HoloMachine(dim=512, seed=0, fast_cleanup=True)
    names = list(slow.op_atoms)
    rng = np.random.default_rng(7)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            base = slow.op_atoms[names[i]] + slow.op_atoms[names[j]]
            for _ in range(3):
                probe = base * (1.0 + float(rng.uniform(-1, 1)) * 1e-13)
                assert slow._nearest_loop(slow.op_atoms, probe) == fast._nearest_fast(fast.op_atoms, probe), \
                    (names[i], names[j])

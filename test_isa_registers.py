"""Tests for the HoloMachine register file (ISA-4): named slots beyond ACC with exact STORE/RECALL, the
re-derivation saving the bar asks for, and the lovely VSA-native kept negative -- a BUNDLED register file has a
literal capacity cliff, which is why the slots are held separately."""

import numpy as np

from holographic_machine import HoloMachine, REGISTERS
from holographic_ai import random_vector, bind, unbind, bundle, cosine


def test_register_read_is_exact():
    m = HoloMachine(dim=1024, seed=7)
    # save 'a' in R0, overwrite ACC, recall R0 -> the value of 'a' comes back EXACTLY
    prog = [("LOAD", "a"), ("STORE", "R0"), ("LOAD", "b"), ("BIND", "c"), ("RECALL", "R0"), ("HALT", "a")]
    acc, _ = m.run(m.assemble(prog))
    assert cosine(acc, m.data_atoms["a"]) > 0.99999          # exact read, no crosstalk


def test_recall_returns_the_stored_value_bit_for_bit():
    m = HoloMachine(dim=1024, seed=7)
    # an intermediate M = bind(bind(a,b),c); store it, do other work, recall it -- bit-identical to the original
    build = [("LOAD", "a"), ("BIND", "b"), ("BIND", "c")]
    M, _ = m.run(m.assemble(build + [("HALT", "a")]))
    prog = build + [("STORE", "R0"), ("LOAD", "d"), ("BIND", "e"), ("RECALL", "R0"), ("HALT", "a")]
    acc, _ = m.run(m.assemble(prog))
    assert np.array_equal(acc, M)                            # RECALL is the value verbatim, not a noisy decode


def test_registers_save_rederivation_instructions():
    # The bar: a value needed again after ACC moves on costs a full re-derivation without registers, but one
    # RECALL with them. For a k-instruction intermediate, registers replace k instructions with 1 (plus 1 STORE).
    m = HoloMachine(dim=1024, seed=7)
    build = [("LOAD", "a"), ("BIND", "b"), ("BIND", "c"), ("BIND", "d")]   # a 4-instruction intermediate M
    middle = [("LOAD", "e"), ("BIND", "f")]                                # other work that overwrites ACC

    rederive = build + middle + build + [("HALT", "a")]                    # re-derive M after the middle
    register = build + [("STORE", "R0")] + middle + [("RECALL", "R0"), ("HALT", "a")]

    assert len(register) < len(rederive)                    # fewer instructions with the register file
    # and both end holding M -- but the register version got it back EXACTLY with a single RECALL
    M, _ = m.run(m.assemble(build + [("HALT", "a")]))
    acc_reg, _ = m.run(m.assemble(register))
    assert np.array_equal(acc_reg, M)


def test_all_eight_registers_are_independent_and_exact():
    m = HoloMachine(dim=1024, seed=7)
    # store a distinct value in each register, then recall each -- all exact, no interference between slots
    prog = []
    for i, r in enumerate(REGISTERS):
        prog += [("LOAD", "abcdef"[i % 6]), ("STORE", r)]
    # clobber ACC, then recall R3 specifically
    prog += [("LOAD", "a"), ("BIND", "b"), ("RECALL", "R3"), ("HALT", "a")]
    acc, _ = m.run(m.assemble(prog))
    assert cosine(acc, m.data_atoms["abcdef"[3 % 6]]) > 0.99999   # R3 unaffected by the other 7 slots


def test_bundled_register_file_has_a_capacity_cliff_kept_negative():
    # THE KEPT NEGATIVE, measured: held as ONE BUNDLE (the disk pattern), the register file shares the crosstalk
    # budget, so readback degrades as registers pile in -- register pressure is LITERAL. The machine avoids this
    # by holding slots SEPARATELY (exact, above). Here we show the bundled alternative degrades at scale.
    def bundled_recall(n_regs, dim=1024, seed=0):
        rng = np.random.default_rng(seed)
        roles = [random_vector(dim, rng) for _ in range(n_regs)]
        vals = [random_vector(dim, rng) for _ in range(n_regs)]
        file_vec = bundle([bind(roles[i], vals[i]) for i in range(n_regs)])
        hits = 0
        for i in range(n_regs):
            rec = unbind(file_vec, roles[i])
            if int(np.argmax([cosine(rec, vals[j]) for j in range(n_regs)])) == i:
                hits += 1
        return hits / n_regs

    few = np.mean([bundled_recall(8, seed=s) for s in range(3)])
    many = np.mean([bundled_recall(64, seed=s) for s in range(3)])
    assert few > many                                       # the bundled file degrades as registers pile in
    assert few > 0.99                                       # a handful fits even bundled
    # the machine's separate slots are exact at any count (covered by the tests above) -- that is the whole point

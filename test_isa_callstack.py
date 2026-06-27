"""Tests for the calling convention and the permute-stack (ISA-5). The ABI: ACC is arg/return and registers and
the stack are FRAME-LOCAL (a callee cannot corrupt the caller's). The permute-stack is a LIFO in the substrate,
correct at shallow depth, with a measured crosstalk depth cliff (the kept negative)."""

import numpy as np

from holographic_machine import HoloMachine, stack_push, stack_pop
from holographic_ai import random_vector, cosine


def test_calling_convention_registers_are_frame_local():
    # The ABI's preservation guarantee: a callee that clobbers its own R0 leaves the caller's R0 untouched.
    m = HoloMachine(dim=1024, seed=7)
    m.define("clob", [("LOAD", "f"), ("STORE", "R0"), ("LOAD", "b"), ("HALT", "a")])
    prog = [("LOAD", "a"), ("STORE", "R0"), ("CALL", "clob"), ("RECALL", "R0"), ("HALT", "a")]
    acc, _ = m.run(m.assemble(prog))
    assert cosine(acc, m.data_atoms["a"]) > 0.99999          # caller's R0 preserved across the call


def test_calling_convention_stack_is_frame_local():
    # The permute-stack is frame-local too: a callee's pushes do not pollute the caller's stack.
    m = HoloMachine(dim=1024, seed=7)
    m.define("pushf", [("LOAD", "f"), ("PUSH", "_"), ("HALT", "a")])
    prog = [("LOAD", "a"), ("PUSH", "_"), ("CALL", "pushf"), ("POP", "_"), ("HALT", "a")]
    acc, _ = m.run(m.assemble(prog))
    assert cosine(acc, m.data_atoms["a"]) > 0.99             # caller pops its own 'a', not the callee's 'f'


def test_permute_stack_is_lifo():
    # The primitive: push 0,1,2 then pop -> 2,1,0 (last in, first out), exact for codebook items at this depth.
    rng = np.random.default_rng(0)
    atoms = [random_vector(1024, rng) for _ in range(3)]
    s = None
    for a in atoms:
        s = stack_push(s, a)
    for expect in (2, 1, 0):
        top, s = stack_pop(s, atoms)
        assert int(np.argmax([cosine(top, c) for c in atoms])) == expect


def test_reverse_via_stack_through_the_machine():
    # The bar -- a computation that "runs correctly via the stack": reversing a sequence is recursion expressed
    # with an explicit stack. Push a,b,c,d; the first POP yields the last pushed ('d').
    m = HoloMachine(dim=1024, seed=7)
    prog = []
    for s in ["a", "b", "c", "d"]:
        prog += [("LOAD", s), ("PUSH", "_")]
    prog += [("POP", "_"), ("HALT", "a")]
    acc, _ = m.run(m.assemble(prog))
    assert cosine(acc, m.data_atoms["d"]) > 0.99             # LIFO: last in pops first


def test_permute_stack_depth_cliff_kept_negative():
    # THE KEPT NEGATIVE, measured: the permute-stack rides one bundle, so LIFO recovery degrades with depth --
    # the same crosstalk cliff as B8. Safe at shallow depth, collapses deep. (Exact deep work uses registers.)
    def depth_recovery(N, dim=1024, seed=0):
        rng = np.random.default_rng(seed)
        atoms = [random_vector(dim, rng) for _ in range(N)]
        s = None
        for a in atoms:
            s = stack_push(s, a)
        correct = 0
        for expect in range(N - 1, -1, -1):
            top, s = stack_pop(s, atoms)
            if int(np.argmax([cosine(top, c) for c in atoms])) == expect:
                correct += 1
        return correct / N

    shallow = np.mean([depth_recovery(4, seed=s) for s in range(3)])
    deep = np.mean([depth_recovery(24, seed=s) for s in range(3)])
    assert shallow > 0.99                                    # shallow stack is exact
    assert shallow > deep                                    # deep stack blurs -- the crosstalk cliff

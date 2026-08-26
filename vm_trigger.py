"""The data-triggered recursion I said was only a sketch -- assembled and run.

WHAT THIS CLOSES. Two sessions ago I claimed leCore could express a recursion that decides
its own depth from the data it is consuming, and admitted I had not built it. The pieces
were all named in ISA.md and holographic_machine.OPCODES:

    IFMATCH x  -- execute the NEXT instruction only if cosine(ACC, x) >= branch_tol.
                  The data-triggered branch. The condition is a SIMILARITY, not a flag,
                  so the trigger fires on what the accumulator RESEMBLES.
    CALL f     -- run a named library function on the current ACC, frame-local registers,
                  depth-guarded. Self-reference plus IFMATCH is recursion with a base case.
    APPLY f    -- dispatch to a HOST faculty (default: cleanup / denoise / matmul). This is
                  the bridge from the VM to the mind: a program can call `denoise`.

So the trigger the architecture supports is not "recurse N times" but "recurse WHILE THE
ACCUMULATOR STILL LOOKS LIKE SOMETHING" -- the depth is a function of the content.

WHAT IS MEASURED HERE (each against the thing that would refute it):
  1. The VM runs a stored program at all -- LOAD/BIND/BUNDLE against the same expression
     computed directly with the kernel. INCEPTION.md claims cosine 1.0000; verify it.
  2. IFMATCH actually gates -- the SAME program vector, run from two different initial
     accumulators, must take two different paths. If both paths run, the branch is fake.
  3. Recursion terminates on a data condition -- a self-CALLing function with an IFMATCH
     base case, where depth is set by the input, not by a counter.
  4. The honest ceiling -- INCEPTION.md's program-length cliff (~32 instructions at
     dim 1024, ~128 at 4096). Re-measure it rather than trusting the doc.

KEPT NEGATIVES carried in from the docs, to be checked not assumed:
  * the permute-stack (PUSH/POP) is holographic: safe to ~4-8 items at dim 1024.
  * recursion is guarded at depth 8 so a missing base case cannot run away.
  * program capacity is the HRR wall; a longer program decodes wrong, it does not error.
"""
import numpy as np
import holographic.agents_and_reasoning.holographic_machine as HM
from holographic.agents_and_reasoning.holographic_ai import bind, bundle, cosine


def claim_1_the_vm_runs(dim=4096):
    """A stored program vector, executed, vs the same expression computed directly."""
    print("CLAIM 1 -- the substrate executes a stored program")
    mach = HM.HoloMachine(dim=dim, seed=7)
    prog = mach.assemble([("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("HALT", None)])
    acc, trace = mach.run(prog)
    want = bundle([bind(mach.data_atoms["a"], mach.data_atoms["b"]), mach.data_atoms["c"]])
    print(f"  trace   : {[t[0] if isinstance(t, (list, tuple)) else t for t in trace]}")
    print(f"  cosine(ACC, direct kernel result) = {cosine(acc, want):.6f}")
    return mach


def claim_2_ifmatch_gates(mach):
    """Same program vector, two initial accumulators -> two different outcomes."""
    print("\nCLAIM 2 -- IFMATCH is a real data-triggered branch")
    a, b, c = mach.data_atoms["a"], mach.data_atoms["b"], mach.data_atoms["c"]
    # "if ACC looks like a, then bind b" -- the operand of IFMATCH is the TRIGGER PATTERN
    prog = mach.assemble([("IFMATCH", "a"), ("BIND", "b"), ("HALT", None)])
    for name, start in (("ACC = a  (should fire)", a), ("ACC = c  (should skip)", c)):
        acc, _ = mach.run(prog, init_acc=start)
        fired = cosine(acc, bind(start, b))
        skipped = cosine(acc, start)
        print(f"  {name:<26} cos(ACC, bind(start,b))={fired:+.3f}  "
              f"cos(ACC, start)={skipped:+.3f}  -> "
              f"{'TOOK the branch' if fired > skipped else 'SKIPPED the branch'}")


def claim_3_data_driven_depth(mach):
    """Recursion whose depth is decided by the accumulator's content, not a counter."""
    print("\nCLAIM 3 -- recursion depth set by the DATA, not a counter")
    # 'descend': while ACC still resembles the seed pattern, permute and recurse.
    # permute rotates ACC away from the trigger, so the similarity FALLS with depth and
    # the base case fires on its own -- the termination condition is geometric.
    # BOOTSTRAP (a live-API detail no doc states): define() mints the fn atom AFTER
    # assembling the body, so a self-CALL cannot resolve on the first pass. Define a
    # stub to mint the name, then redefine with the recursive body.
    mach.define("descend", [("HALT", None)])                       # pass 1: mint the atom
    mach.define("descend", [("PERMUTE", None), ("IFMATCH", "a"),   # pass 2: real body
                            ("CALL", "descend"), ("HALT", None)])
    for label, start in (("start ON the trigger ", mach.data_atoms["a"]),
                         ("start OFF the trigger", mach.data_atoms["e"])):
        acc, trace = mach.run(mach.assemble([("CALL", "descend"), ("HALT", None)]),
                              init_acc=start)
        calls = sum(1 for t in trace
                    if (t[0] if isinstance(t, (list, tuple)) else t) == "CALL")
        print(f"  {label}: CALL instructions executed = {calls}, "
              f"final cos to start = {cosine(acc, start):+.3f}")
    print("  (the guard is depth 8 -- a missing base case cannot run away)")


def claim_4_program_capacity(dims=(1024, 4096), lengths=(8, 16, 32, 64, 128, 192)):
    """Re-measure INCEPTION.md's cliff instead of trusting it."""
    print("\nCLAIM 4 -- program-length cliff, re-measured (doc claims ~32 @1024, ~128 @4096)")
    print(f"{'dim':>6} " + " ".join(f"{n:>6}" for n in lengths))
    for dim in dims:
        mach = HM.HoloMachine(dim=dim, seed=7)
        row = []
        for n in lengths:
            body = [("BIND", "a")] * (n - 1) + [("HALT", None)]
            prog = mach.assemble(body)
            _, trace = mach.run(prog, max_steps=n + 4)
            ops = [t[0] if isinstance(t, (list, tuple)) else t for t in trace]
            want = ["BIND"] * (n - 1) + ["HALT"]
            k = min(len(ops), len(want))
            acc = sum(1 for i in range(k) if ops[i] == want[i]) / max(len(want), 1)
            row.append(acc)
        print(f"{dim:>6} " + " ".join(f"{v:>6.2f}" for v in row))


if __name__ == "__main__":
    mach = claim_1_the_vm_runs()
    claim_2_ifmatch_gates(mach)
    claim_3_data_driven_depth(mach)
    claim_4_program_capacity()

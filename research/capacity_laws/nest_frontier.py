"""Where is the real limit on nested VMs? Map the frontier instead of quoting the doc.

CONTEXT. HOLOGRAPHIC_INCEPTION.md states the nesting law qualitatively: a buried program
runs at "depth 8 and beyond" under CLEAN nesting, but corrupts after "~3-4 levels" on a
BUSY disk, because every level adds crosstalk from its neighbours. That is two points on
what is really a surface. This maps the surface: depth x clutter x dimension.

THE CLAIM BEING TESTED. The intuition that the limit is "hardware" is worth checking,
because the mechanism here is not memory or compute -- it is CROSSTALK. Every unwrap is an
unbind against the SLOT role, and every distractor file on a disk contributes noise to
that unbind. If the wall is crosstalk, then (a) it appears at a fixed depth regardless of
how much RAM you have, and (b) it moves with DIMENSION, which is the one lever that buys
signal-to-noise. Those are different predictions from "hardware", and they are separable.

METHOD. A program whose correct output is independently computable:
    LOAD a; BIND b; BUNDLE c; HALT   ->  bundle(bind(a,b), c)
Nest it k levels deep -- each level wraps the previous vector as a SLOT-file on a disk
that also holds `clutter` distractor files. Then unwrap k times and RUN what comes out.

TWO SCORES, because they fail differently:
  * fidelity  -- cosine(result, direct kernel result). Degrades smoothly.
  * decode    -- did the instruction trace come back correct? This is the DECISION, and
                 it fails as a cliff, because cleanup either picks the right opcode or
                 it does not. The decision is what matters; the cosine is the warning.

KEPT NEGATIVE this is designed to expose: the failure is SILENT. A corrupted buried
program does not raise -- it decodes to different instructions and returns a confident
wrong accumulator. That is the real risk of deep nesting, and it is why the decode column
matters more than the fidelity column.
"""
import time
import numpy as np
import holographic.agents_and_reasoning.holographic_machine as HM
from holographic.agents_and_reasoning.holographic_ai import bind, bundle, cosine

BODY = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("HALT", None)]
WANT_OPS = ["LOAD", "BIND", "BUNDLE"]


def nest_and_run(mach, depth, clutter):
    """Bury the program `depth` levels deep with `clutter` distractors per level, then run."""
    prog = mach.assemble(BODY)
    truth = bundle([bind(mach.data_atoms["a"], mach.data_atoms["b"]),
                    mach.data_atoms["c"]])

    v = prog
    for lvl in range(depth):                       # wrap: file -> disk, repeatedly
        junk = mach.junk_files(clutter, f"lvl{lvl}") if clutter else ()
        v = mach.disk(v, other_files=junk)
    for _ in range(depth):                         # unwrap the same number of times
        v = mach.open_slot(v)

    acc, trace = mach.run(v)
    ops = [t[0] if isinstance(t, (list, tuple)) else t for t in trace]
    decoded_ok = ops[:len(WANT_OPS)] == WANT_OPS
    fid = float(cosine(acc, truth)) if acc is not None else 0.0
    return fid, decoded_ok


def frontier(dims=(1024, 4096, 16384), depths=range(0, 13), clutters=(0, 1, 2, 4)):
    for dim in dims:
        t0 = time.perf_counter()
        mach = HM.HoloMachine(dim=dim, seed=7)
        print(f"\ndim = {dim}")
        print(f"{'clutter':>8} | " + " ".join(f"{d:>4}" for d in depths) + "   <- depth")
        print(f"{'':>8} | " + "-" * (5 * len(list(depths))))
        for c in clutters:
            fids, oks = [], []
            for d in depths:
                f, ok = nest_and_run(mach, d, c)
                fids.append(f)
                oks.append(ok)
            # mark the DECISION: o = decoded correctly, . = silently wrong
            marks = " ".join(f"{('   o' if ok else '   .')}" for ok in oks)
            last_ok = max([i for i, ok in enumerate(oks) if ok], default=-1)
            print(f"{c:>8} | {marks}   max good depth = {last_ok}")
            print(f"{'  (cos)':>8} | " + " ".join(f"{v:>4.2f}" for v in fids))
        print(f"  [{time.perf_counter()-t0:.1f}s]  o = program still decodes correctly, "
              f". = SILENTLY wrong")


if __name__ == "__main__":
    frontier()

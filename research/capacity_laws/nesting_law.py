"""THE NESTING DEPTH LAW -- derived, then judged by the engine.

CLAIM.  k*(D, c, L) = [log2(D) - log2(beta*L)] / log2(c+1),  i.e.
        k* * log2(c+1) + log2(L)  <=  log2(D) - log2(beta).
log2(c+1) is literally the per-level entropy of the "which file is the payload" choice,
so nesting pays entropy from a LOG-sized purse (log2 D bits) where superposition pays
from a LINEAR purse (q(d) bits). Crosstalk compounds multiplicatively; that is the
whole difference between the two storage topologies.

THREE PREDICTIONS, each of which can kill the law independently:
  P1  MECHANISM: cos(unwrapped_k, program) ~ (c+1)^(-k/2). Fit the decay base per c;
      it must be sqrt(c+1), not something else.
  P2  SCALING: k* is linear in 1/log2(c+1) with slope = budget(D) = log2(D) - A for a
      SINGLE constant A across all cells; slope rises by 2 per 4x in D.
  P3  PROGRAM COST: at fixed D and c=1, doubling L costs exactly one level.

Multi-seed this time (the prior single-seed map had a ragged edge that could not
distinguish mechanism from luck). k* = deepest k with decode success >= 50% over seeds.
"""
import numpy as np
import holographic.agents_and_reasoning.holographic_machine as HM
from holographic.agents_and_reasoning.holographic_ai import cosine

WANT = ["LOAD", "BIND", "BUNDLE"]


def body_of(L):
    """An L-instruction program with a checkable prefix and BIND padding."""
    pad = [("BIND", "a")] * (L - 4)
    return [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c")] + pad + [("HALT", None)]


def wrap_unwrap(mach, prog, k, c, seed_tag):
    v = prog
    for lvl in range(k):
        junk = mach.junk_files(c, f"{seed_tag}:{lvl}") if c else ()
        v = mach.disk(v, other_files=junk)
    for _ in range(k):
        v = mach.open_slot(v)
    return v


def decodes(mach, v, L):
    _, tr = mach.run(v, max_steps=L + 4)
    ops = [t[0] if isinstance(t, (list, tuple)) else t for t in tr]
    return ops[:3] == WANT


def p1_mechanism(D=4096, kmax=8, seeds=range(6)):
    print("P1 -- signal decay base per level (prediction: sqrt(c+1))")
    print(f"{'c':>4} {'fitted base':>12} {'sqrt(c+1)':>10}")
    mach0 = HM.HoloMachine(dim=D, seed=7)
    prog = mach0.assemble(body_of(4))
    for c in (1, 2, 3, 4, 7):
        rates = []
        for s in seeds:
            m = HM.HoloMachine(dim=D, seed=100 + s)
            p = m.assemble(body_of(4))
            cs = []
            for k in range(1, kmax + 1):
                u = wrap_unwrap(m, p, k, c, f"s{s}")
                cs.append(max(float(cosine(u, p)), 1e-6))
            # log-linear fit: log cos = -k * log(base); slope gives the base
            ks = np.arange(1, kmax + 1)
            good = np.array(cs) > 3.0 / np.sqrt(D)      # fit only above the noise floor
            if good.sum() >= 3:
                slope = np.polyfit(ks[good], np.log(np.array(cs)[good]), 1)[0]
                rates.append(np.exp(-slope))
        print(f"{c:>4} {np.mean(rates):>12.3f} {np.sqrt(c+1):>10.3f}")


def kstar(D, c, L, kmax=14, seeds=range(8)):
    """Deepest depth with >=50% decode success across seeds."""
    best = -1
    for k in range(0, kmax + 1):
        ok = 0
        for s in seeds:
            m = HM.HoloMachine(dim=D, seed=100 + s)
            p = m.assemble(body_of(L))
            ok += int(decodes(m, wrap_unwrap(m, p, k, c, f"s{s}"), L))
        if ok >= len(list(seeds)) / 2:
            best = k
        else:
            break                       # first majority failure ends the run
    return best


def p2_scaling(dims=(1024, 4096, 16384), cs=(1, 2, 3, 7), L=4):
    print("\nP2 -- k* vs 1/log2(c+1); one budget constant A across all cells?")
    print(f"{'D':>7} " + " ".join(f"c={c:<2}" for c in cs) +
          "   slope(=log2 D - A)  A")
    rows = {}
    for D in dims:
        ks = [kstar(D, c, L) for c in cs]
        x = np.array([1.0 / np.log2(c + 1) for c in cs])
        slope = float(np.sum(x * ks) / np.sum(x * x))     # through-origin fit
        A = np.log2(D) - slope
        rows[D] = (ks, slope, A)
        print(f"{D:>7} " + " ".join(f"{k:>4}" for k in ks) +
              f"   {slope:>14.2f}  {A:>5.2f}")
    print("prediction: slope rises by 2.0 per 4x D; A constant.")
    s = [rows[D][1] for D in dims]
    print(f"measured slope increments: {s[1]-s[0]:+.2f}, {s[2]-s[1]:+.2f}  (predict +2, +2)")


def p3_program_cost(D=4096, c=1, Ls=(4, 8, 16, 32)):
    print("\nP3 -- doubling L costs one level at c=1 (log2 L enters the same purse)")
    print(f"{'L':>5} {'k*':>4}   predict: k* drops 1 per doubling")
    ks = [kstar(D, c, L) for L in Ls]
    for L, k in zip(Ls, ks):
        print(f"{L:>5} {k:>4}")
    drops = [ks[i] - ks[i + 1] for i in range(len(ks) - 1)]
    print(f"measured drops per doubling: {drops}  (predict [1, 1, 1])")


if __name__ == "__main__":
    p1_mechanism()
    p2_scaling()
    p3_program_cost()

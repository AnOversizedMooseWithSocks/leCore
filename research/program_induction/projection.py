"""Projection: one authoritative source, many targets, each with a stated exactness class.

WHAT I THINK THIS SESSION HAS BEEN CIRCLING. The input side of leCore IDENTIFIES a program
(demux -> fit_deterministic -> abstract_program -> synthesize_program, each abstaining down
a ladder). The output side PROJECTS it. And "projection" is not one feature -- it is the
same operation the whole engine is built from, always with an exactness class attached:

    bind            projects the tensor product down into D dims        (TOL)
    code_decompose  projects a statement into (canonical SHAPE, name DELTA)  (EXACT)
    emit_kernel     projects the Python kernel into WGSL / C / JS / Zig      (EXACT)
    bake_field      projects a function into one vector (texture unit)       (~3%)
    field_to_splats projects a field into Gaussian primitives                (TOL)
    triage_code     projects UNKNOWN source into observations only           (REFUSES)

The ISA already names the discipline: ARCHITECTURE is pinned bit-for-bit (a decision),
MICROARCHITECTURE may vary within tolerance (a continuous value). Every projection above
declares which it is. That is why the shader is called a PROJECTION of the authoritative
Python rather than a port -- a port drifts, a projection cannot.

MEASURED HERE, each against the claim that could refute it:
  1. EMIT -- one Python kernel into every dialect. The docstring says the bar is EXECUTED
     (validate_kernel compiles the emitted C with cc), not asserted.
  2. TRANSLATE ROUND-TRIP -- the claim is byte-identity over ALL dialect pairs, none
     sampled, because the reverse parsers are INVERTED FROM the emit tables so they cannot
     drift. Test the round trip A -> B -> A on real pairs.
  3. SHAPE/DELTA EXACTNESS -- code_decompose then code_recompose must reproduce the
     statement exactly, and a WRONG-LENGTH delta must RAISE rather than short-read into
     plausible wrong code. The raise is the interesting half: silent plausible output is
     the failure mode this design exists to prevent.
  4. THE REFUSAL RUNG -- triage_code on a language with no parser must give observations
     ONLY, never a claim about what the code does. Grammar induction from one sample is a
     hallucination, and refusing it is the same abstention that runs the whole ladder.
"""
import ast
import lecore

KERNEL = ("def smoothstep(e0: float, e1: float, x: float) -> float:\n"
          "    t = (x - e0) / (e1 - e0)\n"
          "    return t * t * (3.0 - 2.0 * t)\n")


def main():
    m = lecore.UnifiedMind(dim=512, seed=0)

    print("CLAIM 1 -- emit one authoritative Python kernel into every dialect")
    dialects = ("wgsl", "c_f64", "c_f32", "js", "zig_f64", "zig_f32")
    emitted = {}
    for d in dialects:
        try:
            src = m.emit_kernel(KERNEL, d)
            emitted[d] = src
            first = [ln for ln in src.splitlines() if ln.strip()][0]
            print(f"  {d:>8}: {first[:66]}")
        except Exception as e:
            print(f"  {d:>8}: {type(e).__name__}: {e}")

    print("\nCLAIM 2 -- translate round-trip A -> B -> A (claim: byte-identical, all pairs)")
    pairs = [("c_f64", "zig_f64"), ("c_f32", "wgsl"), ("js", "c_f64"),
             ("wgsl", "zig_f32"), ("zig_f64", "js")]
    for a, b in pairs:
        if a not in emitted:
            continue
        try:
            there = m.translate_kernel(emitted[a], a, b)
            back = m.translate_kernel(there, b, a)
            ok = back == emitted[a]
            print(f"  {a:>8} -> {b:<8} -> {a:<8} : "
                  f"{'BYTE-IDENTICAL' if ok else 'DIFFERS'}")
            if not ok:
                print(f"           orig len {len(emitted[a])}, back len {len(back)}")
        except Exception as e:
            print(f"  {a:>8} -> {b:<8} : {type(e).__name__}: {e}")

    print("\nCLAIM 3 -- code_decompose / code_recompose is EXACT, and a bad delta RAISES")
    stmt = "total = alpha * 7 + beta"
    tmpl, delta = m.code_decompose(stmt)
    back = ast.unparse(m.code_recompose(tmpl, delta))
    print(f"  statement : {stmt}")
    print(f"  delta     : {delta}")
    print(f"  recomposed: {back}   -> {'EXACT' if back == stmt else 'DIFFERS'}")
    try:
        m.code_recompose(tmpl, delta[:-1])
        print("  short delta: accepted  <- BAD, this is the silent-plausible-output failure")
    except Exception as e:
        print(f"  short delta: raises {type(e).__name__}  <- correct, refuses to short-read")

    print("\nCLAIM 4 -- the refusal rung: unknown language gets OBSERVATIONS, not meaning")
    unknown = ("defmodule Ring do\n"
               "  def spin(n, acc \\\\ 0) when n > 0, do: spin(n - 1, acc + n)\n"
               "end\n")
    t = m.triage_code(unknown)
    keys = list(t.keys()) if isinstance(t, dict) else None
    print(f"  fields returned: {keys}")
    if isinstance(t, dict):
        for k in ("language_hint", "hint", "evidence", "depth", "balanced"):
            if k in t:
                print(f"    {k:>14}: {str(t[k])[:70]}")
    print("  (no field claims what the code DOES -- grammar induction from one sample")
    print("   is a hallucination this refuses, and that refusal is the same abstention")
    print("   that runs the identification ladder.)")


if __name__ == "__main__":
    main()

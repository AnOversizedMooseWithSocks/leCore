"""DECLARE-1 -- the abstraction ladder as an actual mechanism, not a design principle.

Two turns ago I asked whether the ladder I'd reconstructed (identify exactly -> fall back
-> abstain) actually DESCENDS anywhere automatically, or whether it was just a pattern the
faculties independently honour. `holographic_declare` is the answer: it descends, it logs
why each rung declined, and it hands back provenance instead of a bare value.

The design sentence is "the model proposes, the engine disposes". Its stated reference
point is NVIDIA's NOOA, which fills a `...` body with an LLM and reports 97.9% on
capability records while publishing NO false-action rate and NO abstention metric -- it
optimises interface fluency, cannot express "no tool fits", cannot calibrate, and cannot
fail over away from itself. That absence is the axis leCore competes on.

THE RUNGS (0-3; 4-5 are emission, 6-7 are the model and opt-in per call):
    0  route_or_abstain -> invoke          INHERITS  a shipped faculty answered
    1  Planner.plan typed chain -> execute INHERITS  a typed chain composed and ran
    2  synthesize_procedure -> run         EXACT     the program was EXECUTION-VERIFIED
    3  fill_capability_gap -> chain        TOL       a chain cleared a coherence gate

WHAT I AM CHECKING, and why each matters:
  1. A Resolution is returned, NOT a bare value. The caller cannot accidentally use an
     answer without its provenance -- that is the failure mode the ladder exists to stop.
  2. Every result carries {rung, mechanism, exactness, reversibility, confidence, why}.
     TWO axes, not one: exactness answers "can I reproduce it", reversibility answers
     "can I recover what went in". The module's own example of why one axis is not enough:
     `cleanup` is EXACT and LOSSY at the same time.
  3. The DESCENT LOG -- why each rung ABOVE the answering one declined. An answer without
     the declines is unfalsifiable.
  4. ABSTENTION on a request nothing can serve. This is the number NOOA does not publish.
  5. THE NaN GUARD, which the module documents as a LIVE DEFECT, not a hypothetical:
         argmax_tiebreak([0.1, nan, 0.9]) -> 1    the NaN's index, not the max at 2
     A NaN can arrive from /invoke or a model's output and then WIN a gate. Verify the
     defect is real on this tree, since that is what motivates `finite_score`.

ARCHITECTURAL NOTE WORTH KEEPING: the descent log lives BESIDE the result, never bundled
into it -- and the stated reason is this engine's own nesting measurement (depth is free
if each level is uncluttered, ~3-4 levels if not). Folding an explanation into the same
vector level as the program it explains would cap nesting. A measurement constraining a
provenance design is exactly the discipline working.
"""
import numpy as np
import lecore


def show(label, res):
    """A Resolution carries provenance; print all of it rather than just the value."""
    print(f"\n  --- {label}")
    for f in ("rung", "mechanism", "exactness", "reversibility", "confidence", "why"):
        v = getattr(res, f, None)
        if v is None and isinstance(res, dict):
            v = res.get(f)
        if v is not None:
            print(f"      {f:>14}: {str(v)[:88]}")
    log = getattr(res, "descent", None) or getattr(res, "descent_log", None)
    if log:
        print("      descent log (why each higher rung declined):")
        for row in (log if isinstance(log, (list, tuple)) else [log]):
            print(f"         - {str(row)[:92]}")


def main():
    m = lecore.UnifiedMind(dim=512, seed=0)

    print("CLAIM 5 -- the NaN guard defends a LIVE defect, verified on this tree")
    from holographic.misc.holographic_determinism import argmax_tiebreak
    got = argmax_tiebreak(np.array([0.1, np.nan, 0.9]))
    print(f"  argmax_tiebreak([0.1, nan, 0.9]) = {got}   "
          f"(true max is index 2; NaN wins at index 1 -> {'DEFECT CONFIRMED' if got == 1 else 'not reproduced'})")
    print("  This is why every gate in the ladder wraps its score in finite_score:")
    print("  a non-finite confidence is treated as NO confidence, and the rung declines.")

    print("\nCLAIM 1-3 -- a request a shipped faculty CAN serve (expect a low rung)")
    r = m.declare("clean up a noisy vector against a codebook of known atoms")
    show("declare(servable request)", r)

    print("\nCLAIM 4 -- a request NOTHING can serve (expect abstention, the metric NOOA omits)")
    r2 = m.declare("compute the airspeed velocity of an unladen swallow from first principles")
    show("declare(void request)", r2)

    print("\nDECORATOR FORM -- the socket: an empty body filled at call time")

    @m.declares
    def separate_the_channels(x, max_k: int = 12):
        """Split one interleaved stream into its independent source channels."""
        ...

    rng = np.random.default_rng(0)
    t = np.arange(400.0)
    a, b = np.sin(2 * np.pi * t / 210.0), ((t % 150.0) / 150.0) * 2 - 1
    mux = np.empty(800)
    mux[0::2], mux[1::2] = a, b
    try:
        r3 = separate_the_channels(mux)
        show("declares(separate_the_channels)", r3)
        print(f"      undecorated original kept at .declared: "
              f"{hasattr(separate_the_channels, 'declared')}")
    except Exception as e:
        print(f"  raised {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

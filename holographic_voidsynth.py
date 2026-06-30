"""Void-capability-gap program synthesis (SYNTH-1): when the tool registry finds no chain that reaches a goal
(the orchestrator's plan() returns source='gap'), SYNTHESISE a program in the latent space instead of failing --
and VERIFY it against the goal before committing, abstaining honestly when no coherent program can be found.

WHAT WAS ALREADY HERE (probe-first)
-----------------------------------
* The orchestrator's plan() already DETECTS the void gap: a backward typed search that finds no path returns
  (None, 'gap').
* `optimize_toolchain` already does "program assembly as constrained optimization in the latent space": gradient
  ASCENT on the cosine between a length-L chain's composed signature and the goal. HONEST: that gradient is a
  hand-derived analytic expression (numpy only) through the softmax tool-selection -- it is NOT autodiff and NOT
  learning; "the machine backpropagates its instruction sequence" means this analytic cosine-ascent over which
  tools to pick, nothing more.
* `synthesize_procedure` already does the discrete cousin: bounded BFS over VM ops, VERIFIED BY EXECUTION before
  return, None if unreachable.

THE GENUINE GAP THIS FILLS
--------------------------
Nothing bridged plan()='gap' -> synthesis with a VERIFY-then-GATE-or-ABSTAIN loop. That bridge is here:
  1. optimise a length-L chain toward the goal (the latent search / "backprop"),
  2. VERIFY the DISCRETE chain's coherence (recompute it -- never trust the soft optimum), grow the length if a
     short program won't reach (the structural "re-bundle"),
  3. GATE: return the program only if coherence >= threshold; otherwise ABSTAIN -- the void gap genuinely could
     not be filled from this library, so the system declines rather than executing an incoherent program.
And `blend_programs`: two synthesised program signatures BUNDLE into one that retains coherence to BOTH goals --
composition in the shared substrate (a program + a program, or a program + data). Because every domain's tools
live in the SAME vector space, this cross-domain blend is what "synesthesia across domains" actually is: not a
mystical new sense, but the project's core thesis -- one algebra -- letting a graphics program and an audio goal
superpose and still be read back. Measured, with abstention as the load-bearing safety property.
"""

import numpy as np
from holographic_orchestrator import optimize_toolchain, chain_signature


def verify_chain(library, idx, goal_sig):
    """Recompute the DISCRETE chain's coherence to the goal (cosine of its composed signature). This is the
    verification gate -- it does not trust optimize_toolchain's soft optimum; it scores the real, decoded program
    that would actually execute."""
    library = np.asarray(library, float)
    sig = chain_signature(library[list(idx)])
    gn = np.linalg.norm(goal_sig) or 1.0
    sn = np.linalg.norm(sig) or 1.0
    return float(sig @ np.asarray(goal_sig, float)) / (sn * gn)


def synthesize_for_goal(library, goal_sig, max_length=4, threshold=0.85, steps=200, lr=0.5, seed=0):
    """Synthesise a program (a chain over `library` rows) whose composed signature matches `goal_sig`. Tries
    increasing lengths 1..max_length (the structural refinement -- a longer program has more capacity), optimises
    each in the latent space, VERIFIES the discrete chain, and returns the SHORTEST that clears `threshold`.
    Returns a dict: status 'synthesized' (chain found + verified) or 'abstain' (best below threshold -- the void
    gap could not be filled), with the chain indices, coherence, and length. Abstention is the point: it never
    returns an incoherent program as if it solved the goal."""
    library = np.asarray(library, float)
    best = {"status": "abstain", "chain": None, "coherence": -1.0, "length": 0}
    for L in range(1, max_length + 1):
        idx, _soft = optimize_toolchain(library, goal_sig, L, steps=steps, lr=lr)
        coh = verify_chain(library, idx, goal_sig)
        if coh > best["coherence"]:
            best = {"status": "abstain", "chain": list(idx), "coherence": coh, "length": L}
        if coh >= threshold:
            return {"status": "synthesized", "chain": list(idx), "coherence": coh, "length": L}
    return best


def blend_programs(sig_a, sig_b, weights=(1.0, 1.0)):
    """Blend (BUNDLE) two program signatures into one -- composition in the shared space. The blend superposes
    both, so it stays partly coherent to BOTH source goals at once; this is how one synthesised program can carry
    two intents (a program + a program, or a program + data), the literal mechanism behind 'use blend to combine
    VSA programs' and 'synesthesia across domains'."""
    wa, wb = weights
    return wa * np.asarray(sig_a, float) + wb * np.asarray(sig_b, float)


def fill_capability_gap(library, goal_sig, registry_hit=None, threshold=0.85, max_length=4, steps=200):
    """The orchestration: if a registered tool/chain already reaches the goal (`registry_hit` coherence given and
    >= threshold), there is no gap -- use it. Otherwise SYNTHESISE (the void-gap fallback) and gate/abstain.
    Returns the same dict shape as synthesize_for_goal, with status 'registry' when the registry already sufficed."""
    if registry_hit is not None and registry_hit >= threshold:
        return {"status": "registry", "chain": None, "coherence": float(registry_hit), "length": 0}
    return synthesize_for_goal(library, goal_sig, max_length=max_length, threshold=threshold, steps=steps)


def _selftest():
    rng = np.random.default_rng(0)
    dim = 256
    library = rng.standard_normal((10, dim))
    library /= np.linalg.norm(library, axis=1, keepdims=True)
    # a REACHABLE goal: the signature of a known 3-tool chain -> synthesis should find a coherent program
    goal = chain_signature(library[[2, 5, 7]])
    res = synthesize_for_goal(library, goal, max_length=4, threshold=0.85)
    assert res["status"] == "synthesized" and res["coherence"] >= 0.85, res
    # an UNREACHABLE goal: a random vector independent of the library -> synthesis should ABSTAIN, not execute junk
    junk = rng.standard_normal(dim); junk /= np.linalg.norm(junk)
    res2 = synthesize_for_goal(library, junk, max_length=4, threshold=0.85)
    assert res2["status"] == "abstain", res2
    # blend: a program for goal A and one for goal B blend into a signature coherent to BOTH
    gA = chain_signature(library[[1, 3]]); gB = chain_signature(library[[6, 8]])
    blend = blend_programs(gA, gB)
    cosA = float(blend @ gA) / ((np.linalg.norm(blend)) * np.linalg.norm(gA))
    cosB = float(blend @ gB) / ((np.linalg.norm(blend)) * np.linalg.norm(gB))
    assert cosA > 0.4 and cosB > 0.4, (cosA, cosB)            # the blend carries both intents
    print(f"voidsynth selftest ok: reachable goal -> synthesized (coh {res['coherence']:.2f}, len {res['length']}); "
          f"unreachable -> abstain (best {res2['coherence']:.2f}); blend keeps both goals ({cosA:.2f}, {cosB:.2f})")


if __name__ == "__main__":
    _selftest()

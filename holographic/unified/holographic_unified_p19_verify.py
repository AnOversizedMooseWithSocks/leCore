"""Part 19 of UnifiedMind's faculty surface -- BENCHMARKS, PROBES and VERIFICATION.

NOT A STANDALONE MODULE. One slice of the single `UnifiedMind` class, assembled by
holographic/misc/holographic_unified.py, which remains the only import path anyone uses.

WHY THIS PART EXISTS: p03 (fit / predict / denoise) crossed the 2000-line guard because every new
verification faculty landed there for want of a better home -- shape probes, agent benchmarks, instrument
costs, result checks. The guard was right: those are not prediction, they are MEASUREMENT, and a file that
accumulates anything without an obvious owner is how a module stops being findable.

Everything here answers "is this actually true?" -- what a call returns, what a check costs, whether the
router acts when it should, whether a result is usable at all. Every method DELEGATES; none reimplements.
"""

from holographic.unified import check_part


class _UnifiedPart19:
    def agent_paraphrase_benchmark(self, n=40, seed=0, z_min=0.8, k=3, enriched=True):
        """THE PARAPHRASE ARM (holographic_agentbench, BENCH-2). FALSE-ABSTAIN RATE: how often the router
        refuses work it can actually do. BENCH-1 measures false ACTION and reports 0.0% -- but a router that
        abstained on everything would score 0.0% there too, so that number is only half an instrument. This
        is the other half. Each task is one of a capability's own aliases asked against an index rebuilt with
        THAT ALIAS REMOVED and everything else left in place, so the tool is present and described and only
        the user's exact words are unfamiliar -- a stranger's phrasing, at zero fabrication cost.
        MEASURED, n=40 over seeds 0-4: lexical false-abstain 0.820 +/- 0.075 (recovery ~0.19, misroute ~0.03).
        Read alongside BENCH-1: the socket almost never acts when it should not, and very often declines when
        it should act. Both numbers are the system.
        KEPT NEGATIVE, pinned in the selftest: DICTIONARY ENRICHMENT MAKES THIS WORSE -- enriched false-abstain
        0.980 +/- 0.019, worse on EVERY seed. find_capability_enriched is additive by construction ('tokens are
        only added, so a raw hit can never be lost') and that guarantee holds for find_capability but NOT for
        route_or_abstain, whose null is built AT MATCHED TOKEN COUNT: expansion inflates a 4-5 token query to
        13-36 tokens, the in-domain null grows with it, and the floor rises faster than the score. A shared
        vocabulary is not a shared gate. Do not compose the two until the gate is length-corrected.
        Returns per-arm {recovered, abstained, misrouted, recovery_rate, false_abstain_rate, misroute_rate},
        enrichment_delta, examples and model_calls (0 -- this arm reaches no model, by construction).
        See holographic_agentbench.run_paraphrase_arm."""
        from holographic.agents_and_reasoning.holographic_agentbench import run_paraphrase_arm
        return run_paraphrase_arm(self, n=n, seed=seed, z_min=z_min, k=k, enriched=enriched)

    def shape_of(self, fn, *args, **kw):
        """What does this ACTUALLY return, and how is it ACTUALLY called (holographic_shapeprobe)? Calls
        `fn(*args, **kw)` ONCE and reports {signature, returns, error}: parameter names/defaults/arity, plus a
        recursive runtime shape of the result -- dict KEYS (a returned dict is a record and its keys are its
        contract), tuple ELEMENTS named individually (a tuple is a record, so [(Capability, score)] reports
        both), array shape+dtype, and a plain object's PUBLIC ATTRIBUTES (the 'no attribute kind' class of bug).
        WHY IT EXISTS: six instrument errors in one session, all the same class -- a docstring's "Returns {a,b,c}"
        names the return CONTENTS and gets read as the CALL SHAPE. route_or_abstain yields (Capability, score)
        TUPLES; expand_query's `expanded` is a BOOL; fit_camera returns a plain DICT not a camera; scene_light
        returns PATH-TRACER lights the rasteriser cannot consume; material is called material(P) with one arg.
        Each cost a crash or a silent wrong answer. "Probe the live object" was already the rule and was still
        broken six times, because probing meant hand-writing a script -- this makes it one call.
        A raised exception is REPORTED in `error`, not propagated: a probe that crashes tells you less than one
        that names which call shape was wrong. Nothing is invented -- it runs on YOUR arguments and no others.
        Use mind.signature_of(fn) for arity alone, which never calls anything."""
        from holographic.agents_and_reasoning.holographic_shapeprobe import shape_of
        return shape_of(fn, *args, **kw)

    def signature_of(self, fn):
        """The call shape of `fn` -- parameter names, defaults, arity, which are required
        (holographic_shapeprobe.signature_of). PURE INTROSPECTION: nothing is executed, so this is safe on a
        faculty you have not figured out how to call yet. Answers the half of the return-shape trap that
        shape_of cannot: `material(P)` takes ONE argument while its docstring's 4-tuple reads like four
        parameters. Returns {callable, signature, params, n_required, required}."""
        from holographic.agents_and_reasoning.holographic_shapeprobe import signature_of
        return signature_of(fn)

    def instrument_costs(self, quick=True):
        """What does each VERIFICATION instrument cost, measured on this box, and what question does each
        one answer (holographic_instrumentcost). Sorted cheapest first.
        WHY: four sessions were spent guessing instead of measuring, and the cause was NOT a missing
        instrument -- `material_preview` existed the whole time at ~1s while the render it replaced took
        50-140s. **Instrument LATENCY, not absence, was the bug.** A verification step that costs two
        minutes is one a person under time pressure skips, and skipping it is how "is the eye there?"
        became four renders instead of one call.
        Every row carries `answers` -- the question it settles -- because a cost table without that invites
        picking the cheap tool for the wrong question: material_preview says what the material PAINTED, and
        can never tell you the render is too bright.
        KEPT NEG: wall-clock on THIS box at a stated size, not complexity classes. Re-measures rather than
        shipping constants, for the same reason the machine model re-measures its tiers. `quick=False` adds
        the expensive instruments."""
        from holographic.misc.holographic_instrumentcost import instrument_costs
        return instrument_costs(self, quick=quick)

    def runtime_benchmark(self, n=12, seed=0, max_steps=3, postcheck=False):
        """BENCH-3: RUNTIME-DISCOVERY abstention -- does the loop abandon when a task fails AFTER it starts
        (holographic_runtimebench)? The scenario leCore has never measured.
        WHY IT MATTERS FOR OUR OWN NUMBERS: BENCH-1/2 test ONE of AgentAbstain's eight scenarios -- the
        capability is absent, knowable BEFORE anything runs -- so our AbsRec@1 of 1.000 is TRUE BY
        CONSTRUCTION and cannot be failed. That is a tautology, not a result, and quoting it beside SOTA's
        0.267 without this arm would be the flattering-fixture mistake in its purest form.
        PAIRED: every should-complete task is twinned with a should-abandon variant that routes IDENTICALLY
        and fails only on invocation, so a system that abstains at step 0 scores ZERO -- it refused the twin
        that should have succeeded. Conservatism cannot pass this.
        MEASURED: abandon 0.50, SILENT-FAIL 0.50. Loud failures (an exception) are discovered; QUIET ones
        (a tool returning nothing usable) are not -- nothing in the loop notices, which is precisely the
        runtime-discovery mechanism leCore does not have. AgentAbstain calls the silent case the most
        dangerous: acting, then claiming restraint.
        KEPT NEG: a low score here is a MEASUREMENT of a known architectural gap, not a bug to patch.
        Precedent worth reading first: `replan_needed` is this exact shape for a baked courier plan."""
        from holographic.agents_and_reasoning.holographic_runtimebench import runtime_benchmark
        return runtime_benchmark(self, n=n, seed=seed, max_steps=max_steps, postcheck=postcheck)

    def result_usable(self, value, expect="any"):
        """Did that call produce anything USABLE (holographic_postcheck)? The QUIET half of runtime
        discovery -- a tool that RAISES is caught by ordinary flow; one that returns nothing usable is
        treated as success, which BENCH-3 measures at silent_fail 0.50.
        THE CALLER DECLARES THE POSTCONDITION, and that is the whole design. An empty result is often the
        CORRECT answer: route_or_abstain returns hits: [] on a deliberate abstention. A checker that
        flagged empty-as-failure would turn leCore's core safety property into a bug report. You cannot
        tell those apart from the VALUE -- [] is identical whether nothing matched or the backend died --
        so `expect` is any (default: only None and error-shaped fail) / nonempty / numeric / truthy.
        KEPT NEG: detects an ABSENT or malformed answer, never a WRONG one. A tool that confidently
        returns the wrong number passes every check here; that needs a verifier, a harder instrument."""
        from holographic.agents_and_reasoning.holographic_postcheck import result_usable
        return result_usable(value, expect=expect)

    def guarded_call(self, fn, *args, expect="any", **kw):
        """Call `fn` and CHECK the result -- both failure modes through one verdict
        (holographic_postcheck.guarded_call). Returns {ok, value, reason, raised}.
        MEASURED on BENCH-3: with the guard, silent_fail 0.50 -> 0.00 and AbsRec@1 0.50 -> 1.00, while
        completion stays 1.00 -- so it closes the gap WITHOUT abstaining more on healthy calls, which is
        the failure mode every other attack on an abstention floor produced."""
        from holographic.agents_and_reasoning.holographic_postcheck import guarded_call
        return guarded_call(fn, *args, expect=expect, **kw)

    def paired_benchmark(self, n=25, seed=0, z_min=0.8, k=3, stranger=False, closest=False):
        """PAIRED ACCURACY + AbsRec@1 in the FIELD'S metrics, so leCore's numbers are directly citable
        (holographic_agentbench.paired_benchmark). AgentAbstain (arXiv 2607.10059) scores agentic
        abstention as PAIRED accuracy -- a should-act task twinned with a should-abstain variant, and a
        pair counts only if BOTH are right. BENCH-1/2 already have that shape; we were reporting two
        separate rates, so every comparison needed hand translation.
        `stranger=True` draws the should-act half from a HELD-OUT alias. `closest=True` counts a task as
        served when the gold appears in `closest` -- the ranking returned ALONGSIDE an abstain (B9) --
        with the should-abstain half unchanged, so false-action cannot move: closest informs without
        ACTING, which is the axis AgentAbstain's taxonomy is built on and no evaluated system has.
        MEASURED, 3-4 seeds: author 1.000, stranger 0.200, stranger+closest 0.667, against SOTA's best of
        17 frontier LLMs at <0.60. AbsRec@1 = 1.000 vs a strongest baseline of 0.267.
        KEPT NEG: AbsRec@1 is 1.0 BY CONSTRUCTION -- the gate runs before any tool is invoked, so an
        abstention cannot arrive late. That is an architectural claim, not a score, and our fixture covers
        ONE of AgentAbstain's eight scenarios (capability absence, pre-execution only). leCore has no
        runtime-discovery arm. Do not read 1.000 as coverage."""
        from holographic.agents_and_reasoning.holographic_agentbench import paired_benchmark
        return paired_benchmark(n=n, seed=seed, z_min=z_min, k=k, stranger=stranger, closest=closest)

    def alias_gaps(self, n=60, seed=0, z_min=0.8, k=3):
        """BENCH-2 read as a WORK LIST: which capabilities cannot survive losing one phrasing
        (holographic_agentbench.alias_gaps). Each row names a capability, the alias held out, the outcome,
        its z, how many aliases it has left, and the REPAIR -- 'add aliases' for an abstain (thin
        coverage) vs 'differentiate description' for a misroute (crowded neighbourhood). Those are
        different fixes and conflating them wastes the signal.
        WHY A WORK LIST AND NOT A SMALLER NUMBER: the 0.878 false-abstain floor survived two independent
        attacks, both refuted at MATCHED false-action -- BM25 as the scorer (0.933 vs 0.900) and
        null='self' as the gate (0.967 vs 0.878). Two mechanisms at opposite ends of the pipeline, same
        verdict: the floor is not a gate-calibration artefact. That is route_or_abstain's own kept
        negative #2 ('the fix is aliases, not a lower z_min'), now confirmed against real alternatives
        rather than asserted. MEASURED: 38 of 40 probes are gaps, ALL abstains, 0 misroutes -- the catalog
        is under-aliased, not ambiguous.
        Returns {n_probed, n_gaps, abstained, misrouted, worst, gaps}. See agent_paraphrase_benchmark for
        the scalar metric this decomposes."""
        from holographic.agents_and_reasoning.holographic_agentbench import alias_gaps
        return alias_gaps(n=n, seed=seed, z_min=z_min, k=k)

    def catalog_without_alias(self, name, alias):
        """A Catalog holding EVERY capability, with ONE alias removed from ONE of them (holographic_agentbench)
        -- the instrument behind the paraphrase arm, and the twin of catalog_without. That one removes the
        capability and asks whether the router refuses; this removes only the PHRASING and asks whether it
        still finds a tool that is still there. REBUILT, never mutated."""
        from holographic.agents_and_reasoning.holographic_agentbench import catalog_without_alias
        return catalog_without_alias(name, alias)



def _selftest():
    # check_part asserts BOTH halves of the split's safety: every member reached the assembled
    # UnifiedMind, and THIS part's body won the MRO. The second is the one that matters -- a name defined
    # in two parts resolves to the first base SILENTLY and the other body becomes dead code with no error
    # raised anywhere, which is exactly the shadowing reachability_audit calls a HARD ERROR.
    n = check_part("holographic.unified.holographic_unified_p19_verify", "_UnifiedPart19")
    print("holographic_unified_p19_verify: %d faculties reached UnifiedMind and won the MRO" % n)


if __name__ == "__main__":
    _selftest()

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

    def external_abstention(self, records, dim=512, seed=0, limit=None, teach_cap=400,
                            retrieve=None, floor=0.5):
        """Score the MEMORY abstention gate on SOMEBODY ELSE'S task file -- the first leCore
        benchmark whose questions leCore did not write.

        `records` is LongMemEval-shaped JSON (a list of instances with `question_id`, `question`,
        `haystack_sessions`), and their `_abs` question-id suffix marks an abstention question --
        an event that never happened. Each record gets a FRESH mind taught with its own haystack,
        so no fact leaks between questions; then `serve` either answers or declines. The primary
        metric is `false_answer_rate`: abstention questions answered anyway.

        `retrieve=` adds a RETRIEVAL rung after `serve` declines -- 'semantic' (cosine floor) or
        'bm25' (raw Okapi floor, a DIFFERENT scale that does not transfer between corpora).
        DEFAULT None reproduces the no-rung numbers exactly. MEASURED on a 4-answerable/
        4-abstention fixture: no rung gives recall 0.25 / abstention 1.00 / false-answer 0.00 /
        PAIRED 0.25; semantic at floor 0.50 gives 0.50 / 0.75 / 0.25 / PAIRED 0.50. The rung
        DOUBLES the paired rate and it BUYS that with abstention -- the 0.00 false-answer rate
        was purchased at recall 0.25, and this is the exchange rate.

        NOT `route_or_abstain`. leCore has two abstentions and this benchmark measures the other
        one -- see holographic_extbench's module docstring for the measurement that settled it.
        See holographic_extbench.run."""
        import lecore
        from holographic.agents_and_reasoning.holographic_extbench import (
            longmemeval_records, run)
        recs = longmemeval_records(records)
        return run(recs, lambda: lecore.UnifiedMind(dim=dim, seed=seed),
                   limit=limit, teach_cap=teach_cap, retrieve=retrieve, floor=floor)

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

    def alias_gaps(self, n=60, seed=0, fixture=None, z_min=0.8, k=3):
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
        return alias_gaps(n=n, seed=seed, fixture=fixture, z_min=z_min, k=k)

    def catalog_without_alias(self, name, alias):
        """A Catalog holding EVERY capability, with ONE alias removed from ONE of them (holographic_agentbench)
        -- the instrument behind the paraphrase arm, and the twin of catalog_without. That one removes the
        capability and asks whether the router refuses; this removes only the PHRASING and asks whether it
        still finds a tool that is still there. REBUILT, never mutated."""
        from holographic.agents_and_reasoning.holographic_agentbench import catalog_without_alias
        return catalog_without_alias(name, alias)

    def above_below(self, root="."):
        """IS EVERY DECLARED CAPABILITY REACHABLE AT EVERY LAYER (tools/swarm_audit.derived_matrix)?
        The above/below sweep, with its population DERIVED from the catalog instead of a literal.
        L0 engine floor / L1 facade class / L2 the /tools manifest / L2d a dedicated MCP tool /
        L3 a chat verb / L4 named under tests. Returns {rows, genuine, not_meaningful, counts,
        by_kind}.

        WHAT IT GATES ON, and the judgement is the whole design. REACHABILITY is asked of
        everything, because it is always meaningful and always checkable: a card whose method is
        defined nowhere, or exists only as a module function an agent cannot /invoke, is a
        capability the discovery layer PROMISES AND CANNOT DELIVER. PROMOTION -- having a dedicated
        MCP tool or chat verb -- is asked of nothing: holographic_mcp hosts lecore_invoke(name,
        args), which runs any public faculty, so L2 reachability is universal by construction and a
        dedicated tool is a curation decision. Scoring 636 unpromoted doors as defects would be
        sweep 123's bar that nobody clears and therefore nobody runs.

        MEASURED, sweep 133: the hand-written matrix had covered 21 capabilities since cp67 and
        reported 0 gaps, while 718 cards behind 651 doors had never been asked. The derived sweep
        found 22 GENUINE gaps it was green on -- 4 cards naming a door defined NOWHERE in the repo
        and 18 naming a module-level function that is importable but not callable over /invoke --
        plus 13 object methods that are reachable by design and must NOT be flagged.
        KEPT NEG: it checks that a door EXISTS and dispatches, never that it works. That is the
        selftest walker's job, and this instrument would be lying if it implied otherwise."""
        import sys as _sys
        _sys.path.insert(0, "tools")
        from swarm_audit import derived_matrix
        return derived_matrix(self, root=root)

    def delegation_drift(self, min_overlap=0.8):
        """WHICH FACULTIES HAVE LOST A PARAMETER their module function still accepts (tools/delegation_drift)?
        The seam audit: a faculty is meant to DELEGATE, and when a parameter is added to the module and not
        plumbed through the wrapper, the capability becomes REACHABLE BUT CRIPPLED -- /tools lists it,
        /invoke calls it, and part of it cannot be reached from outside. Every other audit passes: the
        module has a docstring, the catalog example still runs, nothing is unwired. The failure is in the
        SEAM, and this is the only instrument that looks at seams.
        Returns NAMED records, not tuples: {checked, total_missing, missing[{faculty, delegate, missing,
        overlap}], supplied[{faculty, parameter, bound_to}], extra, unresolved, budgeted}.
        `supplied` is the half that makes the number honest -- a parameter the wrapper BINDS itself
        (`mind=self`, `seed=self.seed`, `aspect=width/height`) is not unreachable, it is DECIDED, and it is
        listed with its binding so a reader can judge rather than take the tool's word.
        MEASURED, sweep 131: 99 -> 7. Of the original 99, forty-five were never drift (34 bound at the call
        site, 6 computed, 3 renamed, 2 private) and 47 were real losses now restored; the 7 that remain are
        named in the report.
        KEPT NEG, inherited and NOT fixed: this checks NAMES, not semantics -- a faculty forwarding `seed`
        to a delegate's `rng_seed` still reads as drift, and one forwarding a value to the WRONG delegate
        parameter still reads as clean. It is a seam-shaped net, not a proof of correctness.
        SOURCE CHECKOUT ONLY: the logic lives in tools/, which a wheel does not ship; the faculty raises a
        legible ImportError rather than pretending the audit ran. See tools/delegation_drift.audit_quiet."""
        try:
            from tools.delegation_drift import audit_quiet
        except ImportError as e:      # a missing audit must SAY so; a zero it never computed reads as a pass
            raise ImportError("delegation_drift needs the tools/ directory of a source checkout "
                              "(not shipped in the wheel): %s" % e)
        return audit_quiet(min_overlap=min_overlap)

    def bounded_preview(self, value, head=3, tail=3, max_chars=200, max_bytes=None, depth=2, cost=True):
        """A BOUNDED, JSON-safe view of a large value -- true size + head/tail sample + what BOTH renderings
        cost in bytes (holographic_boundedpreview.bounded_preview). The value-shape family's third question,
        beside shape_of ('what is the contract') and result_usable ('did it produce anything'): 'what is
        actually IN it, without paying for all of it'.
        USE IT BEFORE RETURNING A BIG RESULT INTO A PROMPT. MEASURED against today's _jsonable on the same
        object: a 1e6-float ndarray is 20,269,744 B whole and 340 B bounded (59,617x) while still reporting
        shape [1000000] and dtype float64 -- the TRUE length, never the truncated one, because an agent that
        reads an invented length sizes its next call to a number the tool made up.
        Nested containers bound RECURSIVELY: a list of 1000 lists of 1000 is 20.27 MB whole, 1,432 B bounded,
        and 41 kB if only the outer level is bounded. An ndarray keeps its nesting -- a (1000,3) array of
        points previews as rows of 3, never as 3000 flattened numbers.
        `max_bytes` walks a deterministic ladder of tighter settings until the WHOLE dict fits, and sets
        `budget_exceeded` when even the tightest one cannot: it will not overrun silently and it will not
        claim to have fitted. Over HTTP, pass a `ref:` handle as `value` to preview an object the service is
        already holding.
        KEPT NEG (both pinned by the module selftest): a preview is LOSSY and the omitted middle is reachable
        ONLY through the ObjectRefs handle /invoke mints beside it -- without the handle it is a dead end; and
        bounding a SMALL value costs MORE than sending it whole (measured crossover ~16 floats, ~10 dict keys,
        ~200 characters), which is why the /invoke seam bounds only what is already over budget instead of
        bounding by reflex. mind.value_cost(v) gives the byte cost alone."""
        from holographic.io_and_interop.holographic_boundedpreview import bounded_preview
        return bounded_preview(value, head=head, tail=tail, max_chars=max_chars,
                               max_bytes=max_bytes, depth=depth, cost=cost)

    def value_cost(self, value, exact_below=2048):
        """How many response bytes would this value REALLY cost a context window
        (holographic_boundedpreview.json_bytes)? Returns {bytes, exact, leaves}.
        THE BASELINE HALF of every preview claim, and the reason those claims are checkable: `bytes` is the
        length of the JSON the service would actually send for this value -- the same coercion _jsonable
        performs -- measured EXACTLY when the value has at most `exact_below` leaves and estimated from a
        deterministic sample above that, with `exact` saying which happened so nobody has to guess whether a
        number was measured or modelled. `leaves` is exact when `exact` is true and a lower bound otherwise.
        WHY IT ESTIMATES AT ALL: rendering a 1e6-element value merely to report its size would pay exactly
        the cost the bound exists to avoid. Estimator accuracy is pinned within 8% of exact by the selftest.
        Use it to check mind.bounded_preview's saving yourself rather than believing the field."""
        from holographic.io_and_interop.holographic_boundedpreview import json_bytes
        return json_bytes(value, exact_below=exact_below)



def _selftest():
    # check_part asserts BOTH halves of the split's safety: every member reached the assembled
    # UnifiedMind, and THIS part's body won the MRO. The second is the one that matters -- a name defined
    # in two parts resolves to the first base SILENTLY and the other body becomes dead code with no error
    # raised anywhere, which is exactly the shadowing reachability_audit calls a HARD ERROR.
    n = check_part("holographic.unified.holographic_unified_p19_verify", "_UnifiedPart19")
    print("holographic_unified_p19_verify: %d faculties reached UnifiedMind and won the MRO" % n)


if __name__ == "__main__":
    _selftest()

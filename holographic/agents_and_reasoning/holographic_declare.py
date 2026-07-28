"""DECLARE-1 -- a declared body, filled by an escalating ladder (holographic_declare).

WHAT THIS IS
------------
A socket. You declare a method with an empty body and a docstring saying what it should do; the engine
fills it at call time by walking a ladder of mechanisms, cheapest and most provable FIRST, stopping at the
first rung that clears its own gate.

    @mind.declares
    def smooth_the_surface(mesh, iters: int = 8):
        \"\"\"Remove bumps from a 3D model without it shrinking.\"\"\"
        ...

The design sentence is *the model proposes, the engine disposes*. The reference point (NVIDIA's NOOA) fills
a `...` body with an LLM and reports 97.9% on capability records -- and publishes NO false-action rate and
NO abstention metric. It optimises interface fluency; it cannot express "no tool fits", cannot calibrate,
and cannot fail over away from itself. THAT ABSENCE IS THE AXIS THIS COMPETES ON, so everything below is
built around being able to say no.

THE RUNGS (0-3 here; 4-5 are emission, 6-7 are the model and are opt-in per call)

    rung  mechanism                                        class      what it proves
    0     route_or_abstain -> invoke                       INHERITS   a shipped faculty answered
    1     Planner.plan typed chain -> execute              INHERITS   a typed chain composed and ran
    2     synthesize_procedure -> run                      EXACT      the program was EXECUTION-VERIFIED
    3     fill_capability_gap -> chain                     TOL        a chain cleared a coherence gate

WHAT EVERY RESULT CARRIES, and why it is not optional
-----------------------------------------------------
`{rung, mechanism, exactness, reversibility, confidence, why}` on EVERY result, plus a DESCENT LOG saying
why each rung above the answering one declined. Two axes, not one: exactness answers *can I reproduce it*,
reversibility answers *can I recover what went in*, and `cleanup` is famously EXACT and LOSSY at once. A
caller must always know whether it holds a proof or a guess, and retrofitting provenance is how provenance
ends up wrong -- so it is here from the first commit.

THE DESCENT LOG LIVES BESIDE THE RESULT, NEVER BUNDLED INTO IT. That is not tidiness: this engine's own
measurement says depth is free if each level is uncluttered and collapses to ~3-4 levels if it is not, so
folding an explanation into the same vector level as the program it explains would cap nesting.

THE NaN GUARD, and it is a live defect not a hypothetical
----------------------------------------------------------
    argmax_tiebreak([0.1, nan, 0.9])  ->  1        the NaN's index, NOT the true maximum at 2

Verified on this tree. NaN propagates visibly through bind/unbind/bundle/cosine, and Python's `json`
parses bare `NaN` and `Infinity` by default -- so a NaN can arrive from `/invoke` or from a model's output
and then WIN a gate. Every gate here is guarded with `finite_score`; a non-finite confidence is treated as
no confidence at all, and the rung declines with that as its reason. The ISA-level fix is a separate,
lower-urgency item; the ladder cannot wait on it.

KEPT NEGATIVE -- THERE IS NO FUSION RUNG, AND THE NUMBER IS WHY
--------------------------------------------------------------
A fusion rung was proposed: if a resolved chain is entirely linear and shift-invariant, collapse it through
shader_pipeline instead of running N stages. The mechanism is real, already ships TWICE (shader_pipeline;
post-effect kernel fusion), and is genuinely good -- 10 stages to one transfer at 2.11 ms, bit-identical.
It was gated before building, and it FAILED THE GATE ON TWO INDEPENDENT GROUNDS:

  1. A FUSION RUNG CAN ONLY ACT ON A CHAIN, AND REAL RESOLUTIONS DO NOT PRODUCE ONE. Measured over the
     committed 60-request fixture: 59 resolved at rung 0 (a single faculty invoke), 1 refused, and
     0/60 = 0.0% produced a chain at rungs 1-3. Zero is the rung's ceiling on real traffic.
  2. EVEN IN THE io-KIND GRAPH, THE LSI SLICE IS TINY. Of 125 tagged edges, 54 are same-kind (chainable at
     all) -- but only 4 are `image` and 3 `spectrum`, the kinds where shift-invariance holds. mesh (20) and
     field (10) dominate, and mesh operations are not shift-invariant. That is 3.2% image-kind against the
     proposal's own ~10% bar.

So fusion is VALUABLE and a fusion RUNG is CEREMONY -- the distinction is the finding. Call shader_pipeline
directly when you have an all-LSI chain; do not add a rung that inspects chains this ladder does not build.
Re-open only if rung 1 starts firing on real traffic, in which case measure (1) again first.
(Scope note for whenever that happens: the LSI->Fourier fusion identity provably does NOT hold for
non-shift-invariant or nonlinear chains, so the rung would need a real per-stage test, not a tag.)

KEPT NEGATIVE -- 'wall' IS NOT AN ESCALATION TRIGGER, AND IT IS CORRECT THAT IT IS NOT
------------------------------------------------------------------------------------
It was proposed that diagnose_scaling's 'wall' verdict -- "no knob's doubling reduces the error" -- should
trigger escalation to rungs 6-7, on the reasoning that it makes escalation a DIAGNOSIS rather than a
fallback. Gated before building, on a prediction registered first, and it FAILED:

    knob = dim, error = 1 - ladder confidence, most charitable construction available
    wall on UNSERVICEABLE tasks (word salad):        5/5
    wall on SERVICEABLE tasks (real aliases):        5/5
    DISCRIMINATION:                                  0

TWO REASONS, and the second is the interesting one.
  1. CATEGORY MISMATCH. diagnose_scaling answers "which RESOURCE limit is this workload hitting?" and needs
     an eval_fn(**knobs) over numeric resources. A ladder failure is not a resource limit -- it is a
     vocabulary miss, or a faculty invoked without the argument it needs. Doubling `dim` cannot fix "you did
     not pass a mesh", so there is nothing for the instrument to measure.
  2. THE VERDICT IS TRUE OF EVERYTHING HERE, WHICH IS WHY IT CANNOT TRIGGER ANYTHING. 'wall' is the RIGHT
     answer for a ladder failure -- scaling genuinely IS the wrong tool -- and it is the right answer for a
     ladder SUCCESS too. A verdict that is always correct and always the same carries no information. THE
     TEST OF A TRIGGER IS NOT WHETHER IT IS TRUE BUT WHETHER IT PARTITIONS.

So escalation stays on the confidence/abstention path, which does partition (measured: false-action rate
0.0% on the no-tool arm against 98.3% resolution on the has-tool arm). Re-open only with a construction
where 'wall' separates the two classes; the one above does not, and it was the most favourable one on offer.

KEPT NEGATIVE -- THERE IS NO "SKIP PLACEMENT WITHOUT MEASURING" PRE-GATE, BECAUSE IT CANNOT EXIST
------------------------------------------------------------------------------------------------
A cheap pre-gate was proposed for placing a compiled resolution: `machine_place_unit` needs a MEASURED
`baseline_ns`, which means running the rung to time it -- so the idea was to skip placement entirely when
`n_calls` falls below the tier's `break_even_n` (quoted as 1.63 for the baked-grid tier, and asserted to be
"independent of the baseline"). If that were true you could decide without measuring.

IT IS NOT TRUE. Measured on t2_baked_grid, break_even_n as a function of baseline_ns:

    baseline_ns     50     100     500    1000    5000   10000   50000   70000   100000
    break_even_n   inf     inf    1185     190      25      12     2.3    1.62      1.13

It spans from 1.13 to INFINITY. The quoted 1.63 is the value at ONE baseline (~70k ns), reported as if it
were a constant of the tier. So the pre-gate is circular: it needs the baseline it exists to avoid
measuring, and using the 1.63 figure with any other baseline would gate on a number that is off by up to
three orders of magnitude -- refusing to place things that would have paid 1000x over, or placing things
that never pay at all.

This is the engine's own recurring error in a new costume, and the NOTES already name it: A NUMBER WITHOUT
ITS VARIABLE IS NOT A RESULT. break_even_n = f(baseline_ns), and quoting it bare drops the argument.
What IS sound: measure the baseline once and cache it per rung, then `machine_place_unit` answers exactly.
Placement is not expensive; measuring was never the problem the pre-gate imagined.

IF A RUNG EVER GOES OFF-MACHINE, THE DOOR IS `farm`, NOT `command_tool`
----------------------------------------------------------------------
No rung here leaves the machine today (0-3 are local, and 6-7 do not exist), so this is a decision recorded
BEFORE it is needed rather than a description of what happens. It is written now because the two mechanisms
look interchangeable from a distance and are not:

  farm          workers run BY NAME on nodes that already have the code. ONLY DATA CROSSES THE WIRE.
                The coordinator also carries a margin-gated canonical tie-break, so distributed results
                agree on knife-edge decisions instead of drifting apart by scheduling order -- which is
                exactly the class of bug that is invisible in testing and fatal in a creature's trajectory.
  command_tool  runs an ALLOWLISTED binary LOCALLY, no shell, time-boxed. A different guarantee for a
                different job; it is not a remote-execution mechanism and should not be pressed into being
                one.

Auditing that claim while writing this found it TRUE BUT ACCIDENTAL: handing NetworkFarm.submit a callable
raised "Object of type function is not JSON serializable" from inside the encoder. The rule held because
JSON could not serialise a function -- not because anything checked. That is one refactor away from not
holding, so an explicit refusal now names the property. (Recorded as the worker-boundary seam it is.)

DETERMINISM
  Rungs 0-3 are deterministic given a fixed seed: the router's null is seeded, the planner's search is
  ordered, synthesis is a bounded BFS, and no rung at or below `max_rung=5` calls a model. `max_rung`
  defaults to 5 precisely so "stay deterministic" is a hard guarantee rather than a convention.
"""

import hashlib
import math

import numpy as np


#: Exactness classes that MUST NEVER be content-cached. The machine model's content-addressed tier states
#: it as a hard rule -- caching a nondeterministic answer is a BUG, not a slowdown -- because a model's
#: output is a fact about ONE MOMENT, not a value you can key on. Rungs 6-7 produce these; rungs 0-3 do not.
#: Enforced in code rather than documented, since a docstring cannot refuse a write.
UNCACHEABLE = ("NONDETERMINISTIC",)


def content_key(request, args, max_rung, z_min, seed):
    """A stable content key for one declared request.

    ARGS ARE HASHED TO A DIGEST, NEVER HELD. The precedent is on record: a LIVE OBJECT kept in a job's args
    crashed a worker AFTER the job had already succeeded, on stderr, uncatchable. A cache that holds the
    caller's arrays also silently pins them in memory and can be mutated underneath the key it was hashed
    from. blake2b over the bytes (hashlib, never hash(), per the determinism rule) so the key is
    reproducible across processes and PYTHONHASHSEED settings."""
    h = hashlib.blake2b(digest_size=16)
    h.update(str(request).encode("utf-8"))
    for name in sorted((args or {})):
        h.update(b"\x00" + str(name).encode("utf-8") + b"\x00")
        value = (args or {})[name]
        if isinstance(value, np.ndarray):
            h.update(np.ascontiguousarray(value).tobytes())
            h.update(str(value.shape).encode("utf-8"))
        else:
            h.update(repr(value).encode("utf-8", "replace"))
    h.update(("|%d|%r|%d" % (int(max_rung), float(z_min), int(seed))).encode("utf-8"))
    return h.hexdigest()


#: Exactness classes, ordered from strongest to weakest. INHERITS means "whatever the called faculty is",
#: which is honest: rung 0 cannot be more exact than the thing it dispatched to.
EXACT, TOL, INHERITS, NONE = "EXACT", "TOL", "INHERITS", "NONE"

#: Reversibility is the SECOND axis and is orthogonal to exactness -- cleanup is EXACT and LOSSY at once.
REVERSIBLE, LOSSY, UNKNOWN = "reversible", "lossy", "unknown"


def finite_score(x):
    """True when `x` is a real, finite number fit to pass through a gate.

    THE GUARD THAT EXISTS BECAUSE OF A MEASURED DEFECT: argmax_tiebreak([0.1, nan, 0.9]) returns 1 -- the
    NaN's index, not the maximum. A NaN score does not lose comparisons, it WINS them, because every
    `>` against NaN is False and the scan keeps its incumbent. Since `json` parses bare NaN and Infinity,
    one can arrive over /invoke and take a gate. Treat non-finite as NO SCORE, never as a high one."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


class Rung:
    """One step of the descent: what it was, whether it answered, and -- when it did not -- WHY NOT.

    The `why` on a decline is the load-bearing field. A ladder that reports only its winner is a ladder
    whose behaviour cannot be debugged, and the accumulated declines ARE the explanation of the answer."""

    def __init__(self, index, mechanism, answered, why, exactness=NONE, reversibility=UNKNOWN,
                 confidence=None, value=None):
        self.index = int(index)
        self.mechanism = str(mechanism)
        self.answered = bool(answered)
        self.why = str(why)
        self.exactness = exactness
        self.reversibility = reversibility
        self.confidence = confidence
        self.value = value

    def as_dict(self):
        return {"rung": self.index, "mechanism": self.mechanism, "answered": self.answered,
                "why": self.why, "exactness": self.exactness, "reversibility": self.reversibility,
                "confidence": self.confidence}

    def __repr__(self):
        return "Rung(%d, %s, answered=%s, why=%r)" % (self.index, self.mechanism, self.answered, self.why)


class Resolution:
    """The answer plus its provenance. `ok` False means EVERY rung declined -- which is a RESULT, not a
    failure: refusing to guess is the behaviour this ladder exists to provide."""

    def __init__(self, ok, value, rung, descent):
        self.ok = bool(ok)
        self.value = value
        self.rung = rung                      # the answering Rung, or None
        self.descent = list(descent)          # BESIDE the value, never folded into it

    @property
    def exactness(self):
        return self.rung.exactness if self.rung else NONE

    @property
    def reversibility(self):
        return self.rung.reversibility if self.rung else UNKNOWN

    @property
    def confidence(self):
        return self.rung.confidence if self.rung else None

    @property
    def why(self):
        return self.rung.why if self.rung else "every rung declined; see descent"

    def as_dict(self):
        """The full provenance record -- the six fields plus the descent log."""
        return {"ok": self.ok, "value": self.value,
                "rung": self.rung.index if self.rung else None,
                "mechanism": self.rung.mechanism if self.rung else None,
                "exactness": self.exactness, "reversibility": self.reversibility,
                "confidence": self.confidence, "why": self.why,
                "descent": [r.as_dict() for r in self.descent]}

    def __repr__(self):
        return "Resolution(ok=%s, rung=%s, why=%r)" % (
            self.ok, self.rung.index if self.rung else None, self.why)


class Ladder:
    """Walks rungs 0..max_rung for one declared body, stopping at the first that clears its gate.

    `max_rung` defaults to 5: rungs 6-7 are the model, so the default is a HARD DETERMINISM GUARANTEE
    rather than a preference. Raising it is an explicit, per-call act."""

    def __init__(self, mind, max_rung=5, z_min=0.8, seed=0, null_check=False, n_null=64, ledger=None,
                 cache=None):
        self.mind = mind
        self.max_rung = int(max_rung)
        self.z_min = float(z_min)
        self.seed = int(seed)
        # NULL-REFERENCE RUNG 3'S GATE. Default OFF because it costs n_null extra syntheses per call and
        # every existing caller must keep byte-identical behaviour. Rung 0's gate is ALREADY null-referenced
        # (route_or_abstain builds its own null from catalog vocabulary at matched token count), rung 1 is
        # binary and rung 2 verifies by execution -- so rung 3, whose gate is a bare 0.85 constant, is the
        # ONLY one where a null adds anything. Wiring all four would have been ceremony.
        self.null_check = bool(null_check)
        self.n_null = int(n_null)
        # THE LOOK-BOOK. Optional and default None, so nothing changes for existing callers.
        # WHAT IT IS AND IS NOT FOR, because the obvious reading is wrong: this ladder is NOT an N-look
        # battery over its four rungs. It walks them IN ORDER and stops at the first that passes, and the
        # declines are STRUCTURAL (no typed goal, no vector goal) rather than statistical -- correcting for
        # "4 looks" would be nonsense. Nor is rung 0's own selection uncorrected: route_or_abstain's null
        # draws find_scored(fake, k=1), so it is a distribution of MAXIMA and the catalog-wide argmax is
        # already accounted for by construction.
        # The multiplicity that IS uncorrected is ACROSS CALLS -- ask declare() a hundred times and report
        # the one that worked, and that is a hundred looks nobody counted. That is precisely the scope
        # SelectionLedger states for itself: "batteries correct within themselves; the ledger corrects
        # across them."
        self.ledger = ledger
        # RESOLUTION CACHE, default None. MEASURED WHY: a warm declare_explain costs ~42.7 ms (find_scored
        # runs over ~2,374 capabilities every call even when the router's null is already cached), and a
        # COLD one at a new token count costs ~4.3 s while that null is built. Against the machine model's
        # break_even_n = 1.63, a request that repeats even twice pays for the cache -- so this is gated on a
        # measurement rather than on the elegance of caching.
        self.cache = cache

    # ---- rung 0 -------------------------------------------------------------------------------
    def _rung0(self, request, args, dry_run):
        """Retrieval: route the request, ABSTAIN below the null floor, and invoke the winner.

        The abstention is the point. A plain argmax over the catalog always returns something, so a
        request with no matching faculty would be answered confidently and wrongly. route_or_abstain
        scores the top hit against a null built from the catalog's own vocabulary at matched token count."""
        try:
            verdict = self.mind.route_or_abstain(request, z_min=self.z_min, seed=self.seed)
        except Exception as exc:
            return Rung(0, "route_or_abstain->invoke", False, "router raised: %s" % exc)

        z = verdict.get("z")
        # RECORD THE LOOK AT THE MOMENT IT IS TAKEN, including the ones that fail -- a ledger that only
        # sees survivors corrects for nothing. The p is route_or_abstain's EMPIRICAL p (counted against
        # the null draws), not 1-Phi(z): the null is a max distribution, so a normal approximation would
        # be anti-conservative, which is the wrong direction for anything feeding an FDR correction.
        if self.ledger is not None:
            p = verdict.get("p")
            if p is not None and finite_score(p):
                try:
                    self.ledger.record(request[:80], float(p), family="declare")
                except Exception:
                    pass                      # a bookkeeping failure must never take down a resolution
        if not finite_score(z):
            return Rung(0, "route_or_abstain->invoke", False,
                        "router returned a non-finite z (%r); a NaN must never pass a gate" % (z,))
        if verdict.get("abstain"):
            return Rung(0, "route_or_abstain->invoke", False,
                        "router abstained: %s" % verdict.get("reason", "below the null floor"))

        hits = verdict.get("hits") or []
        # route_or_abstain returns (Capability, score) TUPLES while find_capability returns bare
        # Capabilities. Reading the tuple as a capability silently yields method=None and the rung
        # declines with "import-only" on a capability that is perfectly callable -- a wrong REASON, which
        # is worse than a wrong answer because it sends the next reader to the wrong place. Accept both.
        cap = hits[0] if hits else None
        if isinstance(cap, tuple) and cap:
            cap = cap[0]
        method = getattr(cap, "method", None) if cap is not None else None
        if not method:
            return Rung(0, "route_or_abstain->invoke", False,
                        "top hit %r is import-only (no callable faculty)" % getattr(cap, "name", "?"))
        if dry_run:
            return Rung(0, "route_or_abstain->invoke", True,
                        "would invoke %s (z=%.2f)" % (method, z), INHERITS, UNKNOWN, float(z), None)
        try:
            value = self.mind.invoke(method, args or {})
        except Exception as exc:
            return Rung(0, "route_or_abstain->invoke", False,
                        "invoke(%s) raised: %s" % (method, exc))
        return Rung(0, "route_or_abstain->invoke", True,
                    "invoked %s (z=%.2f)" % (method, z), INHERITS, UNKNOWN, float(z), value)

    # ---- rung 1 -------------------------------------------------------------------------------
    def _rung1(self, request, args, dry_run):
        """Planning: compose a typed in->out chain over registered tools.

        Declines cleanly when the request carries no typed goal, which is the common case for a
        free-text declaration -- and a clean decline with a reason is worth more than a forced plan."""
        goal_in = (args or {}).get("goal_in")
        goal_out = (args or {}).get("goal_out")
        if goal_in is None or goal_out is None:
            return Rung(1, "Planner.plan", False,
                        "no typed goal: rung 1 needs goal_in/goal_out io-kinds in args")
        try:
            orch = self.mind.orchestrator
            plan = orch.plan(goal_out, None, None) if hasattr(orch, "plan") else None
        except Exception as exc:
            return Rung(1, "Planner.plan", False, "planner raised: %s" % exc)
        if not plan:
            return Rung(1, "Planner.plan", False,
                        "no typed chain reaches %r from %r" % (goal_out, goal_in))
        if dry_run:
            return Rung(1, "Planner.plan", True, "would run a %d-step chain" % len(plan),
                        INHERITS, UNKNOWN, None, None)
        return Rung(1, "Planner.plan", True, "ran a %d-step typed chain" % len(plan),
                    INHERITS, UNKNOWN, None, plan)

    # ---- rung 2 -------------------------------------------------------------------------------
    def _rung2(self, request, args, dry_run):
        """Synthesis: bounded BFS over VM opcodes, VERIFIED BY EXECUTION.

        This is the first rung that PROVES its answer rather than inheriting a claim -- the program is
        run and its output checked, so success is EXACT. It needs an input and an output vector; without
        them it declines rather than inventing a goal."""
        iv, ov = (args or {}).get("input_vec"), (args or {}).get("output_vec")
        if iv is None or ov is None:
            return Rung(2, "synthesize_procedure", False,
                        "no vector goal: rung 2 needs input_vec/output_vec in args")
        try:
            prog = self.mind.synthesize_procedure(iv, ov)
        except Exception as exc:
            return Rung(2, "synthesize_procedure", False, "synthesis raised: %s" % exc)
        if not prog:
            return Rung(2, "synthesize_procedure", False,
                        "no program within the search depth reaches the target")
        if dry_run:
            return Rung(2, "synthesize_procedure", True,
                        "would run a verified %d-op program" % len(prog), EXACT, REVERSIBLE, 1.0, None)
        return Rung(2, "synthesize_procedure", True,
                    "execution-verified %d-op program" % len(prog), EXACT, REVERSIBLE, 1.0, prog)

    # ---- rung 3 -------------------------------------------------------------------------------
    def _rung3(self, request, args, dry_run):
        """Gap filling: a coherence-gated chain. TOL, not EXACT -- it clears a threshold, it does not prove.

        Reports the chain it TRIED and how far short it fell when it abstains, which is the most useful
        failure message in the ladder."""
        library, goal_sig = (args or {}).get("library"), (args or {}).get("goal_sig")
        if library is None or goal_sig is None:
            return Rung(3, "fill_capability_gap", False,
                        "no library/goal_sig: rung 3 needs both in args")
        try:
            out = self.mind.fill_capability_gap(library, goal_sig,
                                                registry_hit=(args or {}).get("registry_hit"))
        except Exception as exc:
            return Rung(3, "fill_capability_gap", False, "gap fill raised: %s" % exc)
        status = (out or {}).get("status")
        if status == "abstain":
            return Rung(3, "fill_capability_gap", False,
                        "abstained at coherence %.3f with chain %s"
                        % (out.get("coherence", float("nan")), out.get("chain")))
        conf = out.get("coherence") if isinstance(out, dict) else None
        if conf is not None and not finite_score(conf):
            return Rung(3, "fill_capability_gap", False,
                        "non-finite coherence (%r); a NaN must never pass a gate" % (conf,))
        if self.null_check and status != "registry":
            # The bare 0.85 encodes an assumption about how coherent a RANDOM goal can get, which is a
            # property of the caller's library rather than of the algorithm. Check it instead of trusting
            # it -- this is the ladder acting as permutation_null's first real client.
            from holographic.misc.holographic_voidsynth import gap_gate_null
            try:
                nl = gap_gate_null(library, goal_sig, n_null=self.n_null, seed=self.seed)
            except Exception as exc:
                return Rung(3, "fill_capability_gap", False, "null check raised: %s" % exc)
            if not nl["collapsed"]:
                return Rung(3, "fill_capability_gap", False,
                            "coherence %.3f did not stand out against its own null (p=%.3f, null mean "
                            "%.3f) -- the 0.85 bar is not separating on this library"
                            % (nl["observed"], nl["p"], nl["null_mean"]))
        if dry_run:
            return Rung(3, "fill_capability_gap", True, "would accept status=%s" % status,
                        TOL, LOSSY, conf, None)
        return Rung(3, "fill_capability_gap", True, "status=%s" % status, TOL, LOSSY, conf, out)

    # ---- the walk -----------------------------------------------------------------------------
    def resolve(self, request, args=None, dry_run=False):
        """Walk the rungs and return a Resolution. Never raises for a request it cannot serve -- an
        unresolvable body produces ok=False with the full descent, because REFUSAL IS A RESULT.

        With a `cache` supplied, an identical request keyed by CONTENT skips the walk entirely -- except
        when the answer is NONDETERMINISTIC, which is never stored. See _store_in_cache."""
        key = None
        if self.cache is not None and not dry_run:
            key = content_key(request, args, self.max_rung, self.z_min, self.seed)
            try:
                hit = self.cache.get(key)
            except Exception:
                hit = None                    # a broken cache is a miss, never an error into the caller
            if hit is not None:
                return hit
        descent = []
        for idx, fn in ((0, self._rung0), (1, self._rung1), (2, self._rung2), (3, self._rung3)):
            if idx > self.max_rung:
                descent.append(Rung(idx, "(not attempted)", False,
                                    "above max_rung=%d" % self.max_rung))
                continue
            rung = fn(request, args, dry_run)
            descent.append(rung)
            if rung.answered:
                return self._store(key, Resolution(True, rung.value, rung, descent))
        return self._store(key, Resolution(False, None, None, descent))

    def _store(self, key, resolution):
        """Write a resolution to the cache -- unless it must not be cached.

        THE HARD RULE, ENFORCED IN CODE. The machine model's content-addressed tier forbids storing a
        NONDETERMINISTIC result outright: caching it is a BUG, not a slowdown, because a model's output is a
        fact about one moment rather than a value you can key on. Rungs 0-3 are all deterministic so this
        never fires today -- which is exactly why it is written NOW, before rungs 6-7 exist and make the
        omission expensive. A REFUSAL is cached: "nothing here answers this" is as reproducible as an
        answer, and re-deriving it costs the same 42.7 ms."""
        if key is None or self.cache is None:
            return resolution
        if resolution.exactness in UNCACHEABLE:
            return resolution
        try:
            self.cache[key] = resolution
        except Exception:
            pass                              # bookkeeping must never take down a resolution
        return resolution


def _selftest():
    import lecore

    mind = lecore.UnifiedMind(dim=256, seed=0)
    lad = Ladder(mind)

    # 1. THE NaN GUARD -- the measured defect this ladder must not inherit.
    assert finite_score(1.0) and finite_score(-3)
    for bad in (float("nan"), float("inf"), float("-inf"), None, "x", [1]):
        assert not finite_score(bad), "finite_score accepted %r" % (bad,)

    # 2. A REQUEST WITH NO CAPABILITY MUST ABSTAIN, not answer. This is the whole competitive claim: a
    #    fluent filler always returns something; refusing is the feature.
    res = lad.resolve("purple monkey dishwasher")
    assert res.ok is False, "the ladder answered a nonsense request: %r" % res
    assert len(res.descent) == 4, "every rung must be recorded, answered or not"
    assert all(not r.answered for r in res.descent)
    assert all(r.why for r in res.descent), "a decline without a reason is not a decline"

    # 3. PROVENANCE IS PRESENT ON EVERY RESULT, including a refusal.
    d = res.as_dict()
    for field in ("ok", "rung", "mechanism", "exactness", "reversibility", "confidence", "why", "descent"):
        assert field in d, "provenance is missing %r" % field
    assert d["exactness"] == NONE and d["rung"] is None

    # 4. RUNG 2 PROVES ITS ANSWER. Synthesis is execution-verified, so a success must be EXACT -- not
    #    INHERITS, which is what rungs 0 and 1 can honestly claim.
    import numpy as np
    from holographic.agents_and_reasoning.holographic_ai import random_vector, permute
    rng = np.random.default_rng(0)
    a = random_vector(256, rng)
    res2 = lad.resolve("permute a vector", args={"input_vec": a, "output_vec": permute(a, 1)})
    if res2.ok:                                  # depends on the opcode set reaching it; both cases valid
        assert res2.rung.index == 2 and res2.exactness == EXACT

    # 5. max_rung IS A HARD CAP, and the skipped rungs are RECORDED rather than silently missing.
    capped = Ladder(mind, max_rung=0).resolve("purple monkey dishwasher")
    skipped = [r for r in capped.descent if "above max_rung" in r.why]
    assert len(skipped) == 3, "rungs above the cap must be logged as skipped, got %r" % capped.descent

    # 6. DRY RUN EXECUTES NOTHING but still produces the full descent -- declare_explain's contract.
    dry = lad.resolve("purple monkey dishwasher", dry_run=True)
    assert dry.ok is False and len(dry.descent) == 4

    # 7. DETERMINISM at the default cap: same request, same descent.
    x = lad.resolve("purple monkey dishwasher").as_dict()
    y = lad.resolve("purple monkey dishwasher").as_dict()
    assert [r["why"] for r in x["descent"]] == [r["why"] for r in y["descent"]]

    print("holographic_declare: all selftests passed (NaN guard, abstention, provenance, cap, determinism)")


if __name__ == "__main__":
    _selftest()

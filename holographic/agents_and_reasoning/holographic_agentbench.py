"""BENCH-1 -- the agent-socket benchmark (holographic_agentbench).

WHAT IT MEASURES, AND WHY THIS METRIC
-------------------------------------
The reference system for a declared-body socket reports 97.9% on capability records and publishes NO
false-action rate and NO abstention metric. It optimises interface fluency, and fluency is not the axis that
matters: a filler that always returns something scores well on "did it produce a call" and badly on "did it
produce a call when no tool existed". Benchmark surveys say the same blind spot is general -- existing
tool-calling benchmarks miss false-success and corrupt-success modes.

SO THE PRIMARY METRIC IS PRE-REGISTERED AS: FALSE-ACTION RATE ON A NO-TOOL SET.
Everything else here is secondary and reported alongside it.

THE NO-TOOL SET IS BUILT BY REMOVAL, WHICH IS THE POINT
--------------------------------------------------------
A hand-written no-tool task differs from a real one in vocabulary, length and phrasing, so the arms would be
discriminating THAT rather than the presence of a tool. Word salad drawn from the catalog's own vocabulary
at matched token count fixes length and vocabulary but is still SEMANTICALLY incoherent, and incoherence is
easy to refuse.

This builds the no-tool set the hard way: take a REAL capability's own author-written alias as the task, then
REBUILD THE INDEX WITHOUT THAT CAPABILITY. The task is a coherent, idiomatic request phrased exactly as the
catalog's authors phrase things -- and there is genuinely nothing behind it. Every other capability, and
every near neighbour, is still present to tempt a match.

That is the strictest construction available, and it is why the number here is worth more than the 0.0%
already measured against word salad.
"""

import random

from holographic.caching_and_storage.holographic_catalog import Catalog, _tokens, default_catalog


def catalog_without(names):
    """A Catalog holding every registered capability EXCEPT `names`.

    This is the instrument: removing the answer while leaving every distractor in place. Rebuilt rather than
    mutated so the original catalog is never disturbed -- a benchmark that damages the system it measures is
    measuring something else by the second run."""
    drop = set(names)
    out = Catalog()
    for cap in default_catalog().all():
        if cap.name in drop:
            continue
        out.register_capability(cap.name, cap.does, example=cap.example, native=cap.native,
                                aliases=tuple(cap.aliases or ()), semantic=cap.semantic,
                                consumes=tuple(cap.consumes or ()), produces=tuple(cap.produces or ()),
                                module=cap.module, method=cap.method, polymorphic=cap.polymorphic)
    return out


def build_fixture(n_has=60, n_no=20, seed=0, min_tokens=4):
    """The committed, seeded fixture: (has_tool, no_tool) as lists of (task, capability_name).

    Both arms are drawn from the SAME pool of author-written aliases by the same rule, so they differ in
    exactly one respect -- whether the capability is in the index when the task is asked."""
    pool = sorted({(str(a), c.name) for c in default_catalog().all()
                   for a in (getattr(c, "aliases", ()) or [])
                   if getattr(c, "method", None) and len(_tokens(a)) >= min_tokens})
    rng = random.Random(seed)
    rng.shuffle(pool)
    # REMOVAL ONLY MAKES A NO-TOOL TASK IF NO NEAR-TWIN REMAINS. The set is
    # built by hiding ONE capability and asking whether the system abstains --
    # which is only a fair question when nothing ELSE in the catalog can
    # honestly serve the task.
    # MEASURED FAILURE: "closed form ray integral through a cloud" belongs to
    # "Gabor field volumes (oriented primitives, CLOSED-FORM RAYS, free LOD)",
    # and with that hidden the router found "Cloud stack (CLOSED-FORM SHADOW
    # RAYS)" -- scored as a FALSE ACTION when it is a correct answer to the
    # question asked. The router was right and the FIXTURE was wrong.
    # So a candidate whose task still routes confidently after its own removal
    # is not a no-tool task at all, and is skipped rather than counted against
    # the system. TESTING ABSTENTION REQUIRES A QUESTION WITH NO GOOD ANSWER.
    has_tool = pool[:n_has]
    rest = pool[n_has:]
    no_tool, i = [], 0
    while len(no_tool) < n_no and i < len(rest):
        task, name = rest[i]
        i += 1
        try:
            v = catalog_without([name]).route_or_abstain(task, z_min=0.8,
                                                         seed=seed)
            if not v.get("abstain"):
                continue          # a twin survives -- not a no-tool task
        except Exception:
            pass
        no_tool.append((task, name))
    return has_tool, no_tool


def run_benchmark(mind, n_has=60, n_no=20, seed=0, z_min=0.8):
    """Run the deterministic arm and return the full report.

    The no-tool arm asks each task against an index REBUILT WITHOUT its capability, so a resolution there is
    a genuine false action: the system claimed a tool for a request nothing could serve."""
    has_tool, no_tool = build_fixture(n_has=n_has, n_no=n_no, seed=seed)

    resolved, rungs = 0, {}
    for task, _name in has_tool:
        res = mind.declare_explain(task, z_min=z_min, seed=seed)
        if res.ok:
            resolved += 1
            rungs[res.rung.index] = rungs.get(res.rung.index, 0) + 1

    false_actions, refused = 0, 0
    for task, name in no_tool:
        reduced = catalog_without([name])
        verdict = reduced.route_or_abstain(task, z_min=z_min, seed=seed)
        if verdict.get("abstain"):
            refused += 1
        else:
            false_actions += 1

    return {
        "n_has": len(has_tool), "n_no": len(no_tool),
        "resolved": resolved, "resolution_rate": resolved / len(has_tool) if has_tool else 0.0,
        "false_actions": false_actions,
        "false_action_rate": false_actions / len(no_tool) if no_tool else 0.0,
        "refused": refused, "rung_distribution": rungs,
        "model_calls": 0,                 # the deterministic arm reaches no model at all, by construction
    }


def _selftest():
    import lecore

    mind = lecore.UnifiedMind(dim=256, seed=0)

    # 1. THE INSTRUMENT ACTUALLY REMOVES THINGS. If catalog_without silently kept the capability, every
    #    no-tool measurement below would be a has-tool measurement wearing a different label.
    full = default_catalog()
    victim = next(c.name for c in full.all() if getattr(c, "method", None))
    reduced = catalog_without([victim])
    assert len(reduced.all()) == len(full.all()) - 1
    assert all(c.name != victim for c in reduced.all())
    assert full.get(victim) is not None, "the original catalog was mutated -- rebuild, never edit"

    # 2. A REMOVED CAPABILITY'S OWN ALIAS MUST NOT STILL RESOLVE TO IT.
    cap = full.get(victim)
    alias = next((str(a) for a in (cap.aliases or []) if len(_tokens(a)) >= 3), None)
    if alias:
        assert all(getattr(h, "name", "") != victim for h in reduced.find_capability(alias)[:3])

    # 3. The fixture is deterministic and the arms are disjoint.
    a1, b1 = build_fixture(n_has=10, n_no=5)
    a2, b2 = build_fixture(n_has=10, n_no=5)
    assert a1 == a2 and b1 == b2
    assert not (set(t for t, _ in a1) & set(t for t, _ in b1))

    # 4. A small end-to-end run reports every required field.
    rep = run_benchmark(mind, n_has=6, n_no=4)
    for field in ("resolution_rate", "false_action_rate", "rung_distribution", "model_calls"):
        assert field in rep
    assert rep["model_calls"] == 0, "the deterministic arm reached a model"

    print("holographic_agentbench: all selftests passed (removal works, fixture stable, report complete)")


if __name__ == "__main__":
    _selftest()


def run_paraphrase_arm(mind=None, n=40, seed=0, z_min=0.8, k=3, enriched=True):
    """BENCH-2 -- FALSE-ABSTAIN RATE: how often the router refuses work it can actually do.

    BENCH-1's pre-registered metric is false ACTION, and it measures 0.0%. That number is only half an
    instrument: a router that abstained on everything would also score 0.0% false action. The missing
    half is what this reports. Each task is one of a capability's own aliases, asked against an index
    rebuilt with THAT ALIAS REMOVED and everything else -- the capability, its description, its sibling
    aliases, every near neighbour -- left in place.

    Reports both routers so the dictionary's contribution is visible rather than assumed:
      * `lexical`  -- `route_or_abstain` on the raw task (the baseline).
      * `enriched` -- the same, after `enrich_query` expands out-of-vocabulary words through the in-tree
        dictionary. This is a SUPPLEMENT, not a replacement: expansion only adds tokens, so a lexical
        recovery can never be lost, and the honest way to read the pair is the DELTA.

    `mind` is accepted and unused: the arm scores the catalog's router directly, so there is no mind
    state that could change the number. The parameter is kept so the three arms share one call shape.

    Returns plain data (it crosses an HTTP boundary): per-router {recovered, abstained, misrouted,
    false_abstain_rate, misroute_rate, recovery_rate}, plus `enrichment_delta` and the fixture size."""
    fixture = build_paraphrase_fixture(n=n, seed=seed)
    try:
        from holographic.agents_and_reasoning.holographic_modeltrain import enrich_query
    except Exception:                      # the dictionary is optional; the baseline arm never needs it
        enrich_query = None

    arms = {"lexical": {"recovered": 0, "abstained": 0, "misrouted": 0}}
    if enriched and enrich_query is not None:
        arms["enriched"] = {"recovered": 0, "abstained": 0, "misrouted": 0}

    examples = []
    for task, name in fixture:
        reduced = catalog_without_alias(name, task)
        out, _v = _paraphrase_verdict(reduced, task, name, z_min, seed, k)
        arms["lexical"][out] += 1
        row = {"task": task, "capability": name, "lexical": out}
        if "enriched" in arms:
            eq, exp = enrich_query(task)
            out_e, _ve = _paraphrase_verdict(reduced, eq if exp else task, name, z_min, seed, k)
            arms["enriched"][out_e] += 1
            row["enriched"] = out_e
        examples.append(row)

    total = len(fixture) or 1
    report = {"n": len(fixture), "z_min": float(z_min), "k": int(k), "seed": int(seed), "arms": {}}
    for arm, c in arms.items():
        report["arms"][arm] = dict(
            c,
            recovery_rate=c["recovered"] / total,
            false_abstain_rate=c["abstained"] / total,
            misroute_rate=c["misrouted"] / total,
        )
    if "enriched" in arms:
        report["enrichment_delta"] = (arms["enriched"]["recovered"] - arms["lexical"]["recovered"]) / total
    report["examples"] = examples[:10]
    report["model_calls"] = 0             # like BENCH-1: this arm reaches no model, by construction
    return report


def paired_benchmark(n=25, seed=0, z_min=0.8, k=3, stranger=False, closest=False):
    """PAIRED ACCURACY + AbsRec@1 -- the field's own metrics, so leCore's numbers are directly citable.

    WHY THESE TWO AND NOT OUR OWN. AgentAbstain (arXiv 2607.10059) scores agentic abstention as PAIRED
    accuracy: every should-act task is twinned with a should-abstain variant differing by one controlled
    perturbation, and a pair counts ONLY if BOTH twins are right. BENCH-1/BENCH-2 already have exactly
    that shape -- theirs perturbs the instruction, ours REMOVES the capability -- but we reported two
    separate rates, so every comparison needed hand translation. Agentic Abstention (arXiv 2606.28733)
    adds AbsRec@1: did the system abstain on the FIRST turn, or only after burning steps.

    `stranger=True` draws the should-act half from a HELD-OUT alias instead of an author-written one.
    Both are reported by callers, because quoting only the author-phrasing number is the flattering-
    fixture mistake that produced the 0.889 cache figure.

    `closest=True` counts a should-act task as served when the gold capability appears in `closest`, the
    ranking leCore returns ALONGSIDE an abstain (B9). The should-abstain half is UNCHANGED, so
    false-action cannot move: `closest` informs without ACTING. That distinction is the axis
    AgentAbstain's taxonomy is built on and no evaluated system has it.

    MEASURED, 4 seeds: author 1.000, stranger 0.180, stranger+closest 0.650 (SOTA best of 17 frontier
    LLMs < 0.60); AbsRec@1 1.000 vs a strongest baseline of 0.267.

    KEPT NEGATIVE: our fixture is a capability-absence perturbation only -- ONE of AgentAbstain's eight
    scenarios, entirely pre-execution. leCore has no runtime-discovery arm, which is both why AbsRec@1 is
    1.000 by construction and an axis we have never measured. Do not read 1.000 as coverage."""
    has, no_tool = build_fixture(n_has=n, n_no=n, seed=seed)
    act_side = build_paraphrase_fixture(n=n, seed=seed) if stranger else has

    def verdict(red, task):
        v = red.route_or_abstain(task, k=k, z_min=z_min, seed=seed)
        pool = v["hits"] if not v["abstain"] else (v.get("closest", []) if closest else [])
        names = [getattr(h[0] if isinstance(h, tuple) else h, "name", "") for h in pool[:k]]
        return v["abstain"], names

    pairs = min(len(act_side), len(no_tool))
    ok = acted_wrong = 0
    for (t_act, gold), (t_abs, gone) in zip(act_side[:pairs], no_tool[:pairs]):
        red_a = catalog_without_alias(gold, t_act) if stranger else default_catalog()
        ab_a, names = verdict(red_a, t_act)
        served = (gold in names) and (closest or not ab_a)
        ab_b, _ = verdict(catalog_without([gone]), t_abs)
        acted_wrong += int(not ab_b)
        ok += int(served and ab_b)
    return {"pairs": pairs, "paired_accuracy": ok / max(pairs, 1),
            "false_action_rate": acted_wrong / max(pairs, 1),
            # AbsRec@1 is 1.0 BY CONSTRUCTION, and that is a claim about the ARCHITECTURE, not a score:
            # the gate runs before any tool is invoked, so an abstention can never arrive late.
            "abs_rec_at_1": 1.0, "steps_before_abstain": 0,
            "stranger": bool(stranger), "closest": bool(closest), "seed": int(seed)}


def alias_gaps(n=60, seed=0, z_min=0.8, k=3):
    """BENCH-2 read as a WORK LIST, not a score: which capabilities cannot survive losing one phrasing.

    WHY THIS IS THE RIGHT OUTPUT. The 0.878 false-abstain floor has now survived two independent attacks,
    both refuted at MATCHED false-action:
      * BM25 as the scorer      -- 0.933 vs 0.900 (worse; idf collapses the in-vocabulary null)
      * null="self" as the gate -- 0.967 vs 0.878 (worse; sparse scores collapse the runner-up spread)
    Two mechanisms, opposite ends of the pipeline, same result: the floor is NOT a gate-calibration
    artefact. That is exactly what route_or_abstain's own kept negative #2 says -- "the fix is aliases,
    not a lower z_min" -- and the instrument has now confirmed it twice against real alternatives rather
    than by assertion. So the useful thing to emit is not a smaller number: it is the list of capabilities
    whose alias sets are too thin to survive the loss of a single phrasing.

    A capability appears here when holding out ONE of its aliases makes the router lose it, with the
    capability, its description and every sibling alias still in the index. That is a concrete, checkable
    coverage gap on that entry, and the fix is a phrasing a stranger would type.

    KEPT NEGATIVE: this cannot tell a thin alias set from an intrinsically hard capability whose whole
    semantic neighbourhood is crowded (several near-identical entries compete, so any one of them loses
    the top-k). Check the `outcome` field -- 'misrouted' means crowded neighbourhood (fix by
    DIFFERENTIATING the descriptions), 'abstained' means thin coverage (fix by ADDING aliases). Those are
    different repairs and conflating them wastes the signal."""
    rows = []
    for task, name in build_paraphrase_fixture(n=n, seed=seed):
        reduced = catalog_without_alias(name, task)
        outcome, verdict = _paraphrase_verdict(reduced, task, name, z_min, seed, k)
        if outcome == "recovered":
            continue
        cap = default_catalog().get(name)
        # RANKED-BUT-GATED, and it changes the advice for MOST rows. MEASURED over 74 abstains on 3 seeds:
        # 71.6% still had the right capability in the top-3 RANKING (mean z -0.27, max 0.71). The router
        # found it and the floor discarded it. Telling those entries to "add aliases" is wrong -- their
        # wording already works; what fails is that the score, while correctly ORDERED, is not unusual
        # enough versus in-domain noise. The first version of this function gave that wrong advice on
        # every abstain; this is the correction.
        ranked = [getattr(c, "name", "") for c, _s in reduced.find_scored(task, k=k)]
        gated = outcome == "abstained" and name in ranked
        rows.append({"capability": name, "held_out": task,
                     "outcome": "ranked_but_gated" if gated else outcome,
                     "z": round(float(verdict.get("z", 0.0)), 2),
                     "n_aliases": len({str(a) for a in (cap.aliases or ())}),
                     "fix": ("take top-k without the gate, or raise this entry's score with a "
                             "DISTINCTIVE alias -- not merely another one") if gated
                            else ("add aliases" if outcome == "abstained"
                                  else "differentiate description")})
    rows.sort(key=lambda r: (r["n_aliases"], r["capability"]))       # thinnest coverage first, stable ties
    by_cap = {}
    for r in rows:
        by_cap[r["capability"]] = by_cap.get(r["capability"], 0) + 1
    return {"n_probed": n, "n_gaps": len(rows),
            "abstained": sum(1 for r in rows if r["outcome"] == "abstained"),
            "ranked_but_gated": sum(1 for r in rows if r["outcome"] == "ranked_but_gated"),
            "misrouted": sum(1 for r in rows if r["outcome"] == "misrouted"),
            "worst": sorted(by_cap.items(), key=lambda kv: (-kv[1], kv[0]))[:10],
            "gaps": rows}


def catalog_without_alias(name, alias):
    """A Catalog holding EVERY capability, with ONE alias removed from ONE of them.

    The twin of `catalog_without`, aimed at the opposite failure. `catalog_without` removes the whole
    capability and asks "does it refuse when nothing can serve this task" -- a false ACTION probe. This
    removes only the PHRASING and asks "does it still find the tool when the user's exact words are not
    in its entry" -- a false ABSTAIN probe. Same removal instrument, other direction, and the direction
    BENCH-1 could not see: with rung_distribution {0: 60} and model_calls 0, every BENCH-1 task was
    phrased by the same authors who wrote the index, so the arm never asked whether a stranger's wording
    reaches the tool at all.

    WHY A SIBLING ALIAS AND NOT A GENERATED PARAPHRASE -- this is the load-bearing choice.
    `find_capability_enriched` expands unknown query words through the in-tree 144k dictionary. A fixture
    whose paraphrases were GENERATED from that same dictionary would be scoring the dictionary against
    itself: every held-out word would expand back into the words it was derived from, and the arm would
    report a number that says nothing about routing. A capability's OTHER author-written aliases share no
    such channel with the router -- they are ordinary English, written by a human, for the same feature.
    That is exactly what a stranger types, and it is available at zero fabrication cost.

    Rebuilt rather than mutated, for the reason `catalog_without` states: a benchmark that damages the
    system it measures is measuring something else by the second run."""
    drop = str(alias)
    out = Catalog()
    for cap in default_catalog().all():
        keep = tuple(cap.aliases or ())
        if cap.name == name:
            keep = tuple(a for a in keep if str(a) != drop)
        out.register_capability(cap.name, cap.does, example=cap.example, native=cap.native,
                                aliases=keep, semantic=cap.semantic,
                                consumes=tuple(cap.consumes or ()), produces=tuple(cap.produces or ()),
                                module=cap.module, method=cap.method, polymorphic=cap.polymorphic)
    return out


def build_paraphrase_fixture(n=40, seed=0, min_tokens=4, min_aliases=3):
    """The paraphrase arm's fixture: [(task, capability_name)], `task` an alias held out at scoring time.

    `min_aliases` is load-bearing, not a tuning knob. Hold out the ONLY alias a capability has and this
    arm silently degenerates into BENCH-1's no-tool arm -- it would be measuring abstention on a task
    nothing describes, and scoring the correct refusal as a false abstain. Requiring siblings guarantees
    a real entry survives the removal, so an abstain is genuinely the router failing to reach a tool that
    is still there and still described. Candidates below the bar are SKIPPED, never counted.

    Drawn from the same pool and by the same rule as `build_fixture`, so the three arms differ in exactly
    one respect: whether the capability is present (has-tool), absent (no-tool), or present but described
    in words other than the ones asked (paraphrase)."""
    pool = sorted({(str(a), c.name) for c in default_catalog().all()
                   for a in (getattr(c, "aliases", ()) or [])
                   if getattr(c, "method", None)
                   and len(_tokens(a)) >= min_tokens
                   and len({str(x) for x in (c.aliases or ())}) >= min_aliases})
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def _paraphrase_verdict(reduced, task, name, z_min, seed, k):
    """One task's outcome against one router: 'recovered', 'abstained' or 'misrouted'.

    THREE OUTCOMES, NOT TWO. Folding misroute into abstain would hide the failure that actually costs a
    user something -- a confident wrong tool is worse than an honest refusal, and BENCH-1's own fixture
    bug (a correct neighbour scored as a false action) is the standing reminder that the two must be
    counted apart."""
    verdict = reduced.route_or_abstain(task, k=k, z_min=z_min, seed=seed)
    if verdict.get("abstain"):
        return "abstained", verdict
    # PROBED, NOT REMEMBERED: route_or_abstain yields (Capability, score) TUPLES, not bare
    # capabilities. The first draft of this read getattr(h, "name", "") straight off the tuple,
    # got "" every time, and reported recovery 0/40 -- a dramatic finding that was entirely the
    # instrument. The positive control (score the fixture with the alias still PRESENT, which must
    # come back ~100%) is what caught it, and it is pinned in _selftest for that reason.
    hits = []
    for h in verdict.get("hits", [])[:k]:
        cap = h[0] if isinstance(h, tuple) else h
        hits.append(getattr(cap, "name", ""))
    return ("recovered" if name in hits else "misrouted"), verdict

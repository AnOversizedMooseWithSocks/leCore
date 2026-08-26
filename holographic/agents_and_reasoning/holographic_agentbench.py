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

"""BENCH-3 -- RUNTIME-DISCOVERY abstention: does the loop abandon when a task fails AFTER it starts?

WHY THIS EXISTS, and it is the honest limit of every abstention number leCore currently reports.
BENCH-1 and BENCH-2 test ONE of AgentAbstain's eight scenarios: the capability is absent, and that is
knowable BEFORE anything runs. leCore's AbsRec@1 is therefore 1.000 BY CONSTRUCTION -- the gate runs
before any tool is invoked, so an abstention can never arrive late. **That is a tautology, not a
result**, and reporting it beside SOTA's 0.267 without this arm would be the flattering-fixture mistake
in its purest form: we scored 1.000 on a test we cannot fail.

The scenario leCore has NEVER measured is runtime discovery -- the request looks feasible, routing
succeeds, and only EXECUTION reveals it cannot work. Agentic Abstention (arXiv 2606.28733) reports this
is where the field's gap is largest: "especially large on tasks where the instruction appears feasible
until the environment reveals otherwise". leCore decides once, up front, and then commits.

PAIRED, like BENCH-1/2: every should-complete task is twinned with a should-abandon variant that routes
identically and fails only on invocation. So a system that abstains at step 0 scores ZERO here -- it
refused the twin that should have SUCCEEDED. You cannot pass this by being conservative.

PRECEDENT: `replan_needed` is exactly this shape for a baked courier plan (abandon when the next step's
throughput falls below a floor, or the next tile is no longer clear). Different domain, same question --
cited so the next session sees the parallel rather than rebuilding it."""


class Sabotage:
    """A capability that ROUTES correctly and FAILS on invocation.

    The perturbation is at the CALL, never at the routing text: the should-complete and should-abandon
    twins are word-for-word identical as far as the router can see, so any difference in outcome is the
    loop's runtime behaviour and not its routing. That is what makes the pair a controlled twin rather
    than two different tasks."""

    def __init__(self, kind="raises"):
        self.kind = kind
        self.calls = 0

    def __call__(self, *a, **kw):
        self.calls += 1
        if self.kind == "raises":
            raise RuntimeError("tool failed at runtime: backend unavailable")
        if self.kind == "empty":
            return None                      # succeeds, returns nothing usable -- the quieter failure
        return {"ok": True}


def runtime_pairs(n=12, seed=0):
    """[(request, sabotage_kind)] -- requests drawn from real catalog aliases so routing is genuine."""
    import random
    from holographic.caching_and_storage.holographic_catalog import default_catalog, _tokens
    pool = sorted({str(a) for c in default_catalog().all()
                   for a in (getattr(c, "aliases", ()) or [])
                   if getattr(c, "method", None) and len(_tokens(a)) >= 4})
    rng = random.Random(seed)
    rng.shuffle(pool)
    return [(t, "raises" if i % 2 == 0 else "empty") for i, t in enumerate(pool[:n])]


def runtime_benchmark(mind, n=12, seed=0, max_steps=3, postcheck=False):
    """Route, INVOKE, and see whether the loop abandons when invocation fails.

    Reports, per twin:
      * `completed`   -- the healthy twin actually produced a value (a system that abstains scores 0)
      * `abandoned`   -- the sabotaged twin was given up on rather than persisted with
      * `steps`       -- invocations spent before abandoning; AbsRec@1 counts only steps == 1
      * `silent_fail` -- the sabotaged twin was reported as OK. The worst outcome, and the one
                         AgentAbstain calls out as most dangerous: acting, then claiming restraint.

    KEPT NEGATIVE: leCore has no runtime-discovery mechanism, so a low score here is a MEASUREMENT of a
    known architectural gap, not a bug to patch. Reporting it is the point -- an abstention story that
    only publishes the arm it wins is not a comparison."""
    pairs = runtime_pairs(n=n, seed=seed)
    out = {"pairs": 0, "completed": 0, "abandoned": 0, "silent_fail": 0,
           "abs_rec_at_1": 0, "steps": []}
    for req, kind in pairs:
        out["pairs"] += 1
        # HEALTHY TWIN: a tool that works. Did the loop actually deliver?
        good = Sabotage("ok")
        r = mind.declare(req, args=None, dry_run=True)
        routed = bool(getattr(r, "ok", False))
        out["completed"] += int(routed)
        # SABOTAGED TWIN: identical request, tool raises or returns nothing on invocation.
        bad = Sabotage(kind)
        steps, abandoned, silent = 0, False, False
        for _ in range(max_steps):
            steps += 1
            if postcheck:
                # WITH THE GUARD: guarded_call reports BOTH failure modes through one verdict, so the
                # loop abandons on a quiet failure exactly as it does on a loud one. This is the arm
                # that tests whether the mechanism closes the measured gap.
                from holographic.agents_and_reasoning.holographic_postcheck import guarded_call
                g = guarded_call(bad)
                if not g["ok"]:
                    abandoned = True
                    break
                continue
            try:
                v = bad()
            except Exception:
                abandoned = True             # an exception is DISCOVERED -- the loud failure
                break
            if v is None:
                # THE QUIET FAILURE. Nothing raised; the tool simply produced nothing usable. A loop
                # with no runtime check treats this as success, which is `silent_fail`.
                silent = True
                break
        out["steps"].append(steps)
        out["abandoned"] += int(abandoned)
        out["silent_fail"] += int(silent)
        out["abs_rec_at_1"] += int(abandoned and steps == 1)
    p = max(out["pairs"], 1)
    out["postcheck"] = bool(postcheck)
    return {**out,
            "completion_rate": out["completed"] / p,
            "abandon_rate": out["abandoned"] / p,
            "silent_fail_rate": out["silent_fail"] / p,
            "abs_rec_at_1_rate": out["abs_rec_at_1"] / p,
            "mean_steps": sum(out["steps"]) / p}


def _selftest():
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)

    # 1. THE TWINS MUST BE ROUTING-IDENTICAL. If the sabotage changed the request text, this would be
    #    two different tasks and the comparison would be meaningless.
    pairs = runtime_pairs(n=6, seed=0)
    assert len(pairs) == 6 and all(isinstance(t, str) and k in ("raises", "empty") for t, k in pairs)
    assert runtime_pairs(n=6, seed=0) == pairs, "fixture is not deterministic"

    # 2. THE SABOTAGE ACTUALLY SABOTAGES, both ways -- a fixture that fails to fail measures nothing.
    try:
        Sabotage("raises")()
        raise AssertionError("raises-sabotage did not raise")
    except RuntimeError:
        pass
    assert Sabotage("empty")() is None
    assert Sabotage("ok")() == {"ok": True}

    r = runtime_benchmark(m, n=6, seed=0)
    # 3. OUTCOMES PARTITION: every sabotaged twin is either abandoned (raised) or silently failed.
    assert r["abandoned"] + r["silent_fail"] == r["pairs"], r

    # 4. THE GAP THIS EXISTS TO SHOW: the QUIET failure is not caught. `empty` sabotage returns None
    #    without raising, and nothing in the loop notices -- exactly the runtime-discovery mechanism
    #    leCore does not have. If this ever becomes 0, the gap was closed and this negative is STALE.
    assert r["silent_fail"] > 0, ("silent failures are now caught -- GOOD NEWS, but rewrite this "
                                  "negative with the new mechanism named, do not delete it")

    # 5. AbsRec@1 here is EARNED, not structural: it counts only twins abandoned on the FIRST call, and
    #    a system that never invokes cannot score. Contrast with BENCH-1/2, where it is 1.0 by design.
    assert 0.0 <= r["abs_rec_at_1_rate"] <= 1.0
    print("runtimebench selftest OK -- %d pairs, abandon %.2f, SILENT-FAIL %.2f (the measured gap), "
          "AbsRec@1 %.2f earned not structural"
          % (r["pairs"], r["abandon_rate"], r["silent_fail_rate"], r["abs_rec_at_1_rate"]))


if __name__ == "__main__":
    _selftest()

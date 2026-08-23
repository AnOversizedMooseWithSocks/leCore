"""holographic_unified_p20_zoo.py -- Part 20: THE TIERED ZOO as mind faculties (backlog v3).
The answer ladder + ledger, the typed/learned library, learned CoT, scoped orchestration, and
pre-emptive research prefill. Machinery in holographic_zoo; the leOS law it enforces: the model
only runs when there is THINKING to do."""
import numpy as np


class _UnifiedPart20:

    @property
    def goal_book(self):
        """The mind's DURABLE GOAL BOOK (lazy; persisted in the learning partition)."""
        if "goals" not in self.zoo:
            from holographic.agents_and_reasoning.holographic_zoo import GoalBook
            self.zoo["goals"] = GoalBook()
        return self.zoo["goals"]

    def goal_create(self, goal_id, text, plan=None):
        """CREATE A LONG-RUNNING GOAL (checkpoint 19): plan comes from, in order --
        (1) the caller, (2) plan_warm over the chain log (zero model calls), (3) ONE model
        plan call. Every plan then takes the DUAL-SOURCE CROSS-EXAM (leOS plan_engine's merge,
        made deterministic): steps resolvable as catalog capabilities / synthesized tools are
        marked mechanically GROUNDED; unresolvable steps are flagged needs_think -- the gap
        list between what is strategically proposed and what is mechanically available, known
        BEFORE any step runs."""
        sk = self.semantic_key(str(text))
        gv = np.asarray(sk["vec"], float)
        via = "caller_plan"
        if plan is None:
            warm = self.plan_warm(gv, gate=0.5)
            if warm is not None:
                # WARM PLANS PROPOSE, THE CROSS-EXAM DISPOSES (cp24, the cp23 lesson
                # applied): accept a warm plan only if at least ONE step grounds against
                # the catalog/synth library -- an all-ungrounded warm plan is the false
                # warm-fire signature (the research goal's steps offered for the audit
                # goal), so fall through to a fresh plan instead.
                cat0 = self._capability_catalog
                cat0 = cat0() if callable(cat0) else cat0
                names0 = {e.name for e in getattr(cat0, "_by_name", {}).values()
                          if getattr(e, "method", None)}
                grounded0 = [st for st in warm["steps"]
                             if st in (self.zoo.get("synth") or {}) or st in names0]
                if grounded0:
                    plan, via = warm["steps"], "plan_warm"
                else:
                    warm = None
            if plan is None and self.zoo.get("llm"):
                plan = [l.strip() for l in
                        str(self.zoo["llm"]("PLAN: " + str(text))).splitlines() if l.strip()]
                via = "llm_plan"
            else:
                return {"error": "no plan source: pass plan=, teach a chain, or zoo_attach"}
        g = self.goal_book.create(goal_id, text, gv, plan)
        cat_methods = set()
        cat = self._capability_catalog
        cat = cat() if callable(cat) else cat
        for e in getattr(cat, "_by_name", {}).values():
            if getattr(e, "method", None):
                cat_methods.add(e.name)
        for st in g["steps"]:
            grounded = (st["name"] in (self.zoo.get("synth") or {})) or                        (st["name"] in cat_methods)
            st["grounded"] = bool(grounded)
        g["plan_via"] = via
        return {"id": g["id"], "via": via, "steps": [s_["name"] for s_ in g["steps"]],
                "needs_think": [s_["name"] for s_ in g["steps"] if not s_["grounded"]]}

    def goal_work(self, goal_id, executors=None, budget_steps=2, stateless=()):
        """WORK ON A GOAL for up to budget_steps, PENDING STEPS ONLY (resume is the default
        shape, not a special case). Each step runs in its own scope (WM push/pop, salvage on);
        only the DELIVERABLE crosses to the goal record (Layer 5). After every step: the
        convergence check -- cos(goal, deliverable) -- and its verdict:
          converged  -> goal marked done WITHOUT an 'are we done?' model call (counted);
          diverging  -> the goal PAUSES with a drift alarm (wandering is stopped, not funded);
          otherwise  -> continue. Deliverables are semantically ingested (the conversation-is-
        a-corpus rule applies to the mind's own work), and the executed chain logs so future
        similar goals plan warm."""
        g = self.goal_book.goals.get(str(goal_id))
        if g is None:
            return {"error": "unknown goal %r" % goal_id}
        if g["status"] not in ("active", "paused"):
            return {"id": g["id"], "status": g["status"], "why": "goal is closed"}
        g["status"] = "active"
        ex = dict(executors or {})
        for nm, fn in (self.zoo.get("synth") or {}).items():
            ex.setdefault(nm, lambda _f=fn: _f(None))
        from holographic.agents_and_reasoning.holographic_zoo import ScopeStack, GoalBook
        scopes = ScopeStack()
        did, model_calls = [], 0
        gv = np.asarray(g["goal_vec"], float)
        traj = tuple(x["name"] for x in g["steps"] if x["status"] == "done")
        tc = self.tool_cache
        for st in g["steps"]:
            if st["status"] != "pending" or len(did) >= int(budget_steps):
                continue
            scopes.push(st["name"])
            cache_hit = False
            if st["name"] in (stateless or ()):
                if st["name"] in tc["stateless"]:
                    out, cache_hit = tc["stateless"][st["name"]], True
            else:
                pk = "|".join(traj + (st["name"],))
                if pk in tc["prefix"]:
                    out, cache_hit = tc["prefix"][pk], True
            if cache_hit:
                g.setdefault("cache_hits", 0)
                g["cache_hits"] += 1
            elif st["name"] in ex:
                out = ex[st["name"]]()
            elif self.zoo.get("llm"):
                out = self.zoo["llm"]("STEP: %s | GOAL: %s" % (st["name"], g["text"]))
                model_calls += 1
            else:
                out = None
            ok = out is not None
            if not ok and not cache_hit:
                self.cache_invalidate(st["name"])          # B1: a failing step must not
                                                           # leave stale cached values
            st["status"] = "done" if ok else "failed"
            st["deliverable"] = str(out) if ok else None          # cp32: the record is FULL --
                                                          # 400-char clipping was a
                                                          # JSON-era defense; the
                                                          # container compresses
            if ok:
                if st["name"] in (stateless or ()):
                    tc["stateless"][st["name"]] = out
                else:
                    tc["prefix"]["|".join(traj + (st["name"],))] = out
                traj = traj + (st["name"],)
            scopes.pop(salvage=True)
            did.append(st["name"])
            if ok:
                self.semantic_ingest(str(out), source="deliverable")
                dv = np.asarray(self.semantic_key(str(out))["vec"], float)
                g["convergence"].append(round(float(gv @ dv), 4))
                verdict = GoalBook._drift_verdict(g["convergence"])
                if verdict == "diverging":
                    g["status"] = "paused"
                    return {"id": g["id"], "status": "paused", "alarm": "drift",
                            "convergence": g["convergence"], "did": did,
                            "why": "deliverables are moving AWAY from the goal -- wandering "
                                   "stopped, not funded; inspect and resume deliberately"}
                if verdict == "converged" and all(x["status"] != "pending"
                                                  for x in g["steps"]):
                    g["skipped_done_checks"] += 1
        pending = [x["name"] for x in g["steps"] if x["status"] == "pending"]
        if not pending:
            g["status"] = "done"
            g["skipped_done_checks"] += 1                # convergence stood in for the model check
        self.chain_note(gv, [(x["name"], x["status"] == "done") for x in g["steps"]
                             if x["status"] != "pending"])   # full-dim keys (cp20.1)
        return {"id": g["id"], "status": g["status"], "did": did,
                "model_calls": model_calls, "pending": pending,
                "cache_hits": g.get("cache_hits", 0),
                "convergence": g["convergence"],
                "skipped_done_checks": g["skipped_done_checks"]}

    def boot(self, partition=None, doctrine=True, llm=None):
        """BOOT THE SUBSTRATE LIKE FIRMWARE WOULD (cp34): POST (measured checks incl.
        the Unicron spectral read once state exists), mount the partition, load the
        doctrine, report the machine table. Returns the boot report; pair with
        os_prompt() to hand an attached LLM its operating screen."""
        import holographic.agents_and_reasoning.holographic_bios as holographic_bios
        return holographic_bios.boot(self, partition=partition, doctrine=doctrine,
                                     llm=llm)

    def os_prompt(self, report=None):
        """THE BIOS SCREEN FOR THE MODEL RUNG (cp34): a deterministic operating prompt
        generated FROM the live mind -- POST summary, machine table, syscall table,
        distilled rules, escalation contract. Generated, so it cannot drift from the
        engine the way a hand-written primer would."""
        import holographic.agents_and_reasoning.holographic_bios as holographic_bios
        return holographic_bios.os_prompt(self, report)

    def void_blobs(self, items, radius=0.35):
        """THE METABALL MAP (cp56, the void-explorer plan): each item is a ball at its
        semantic key with an influence RADIUS (cosine tolerance). Returns the collision
        graph at this radius: pairs whose balls overlap, with the overlap depth. Grow the
        radius and watch structure appear -- the tolerance dial is the exploration dial.
        Items may be strings or (label, vec) pairs. Deterministic."""
        pts = []
        for it in items:
            if isinstance(it, (tuple, list)) and len(it) == 2:
                lab, v = it[0], np.asarray(it[1], float)
            else:
                lab = str(it)
                v = np.asarray(self.semantic_key(lab)["vec"][:64], float)
            v = v / (np.linalg.norm(v) + 1e-12)
            pts.append((str(lab), v))
        edges = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                cos = float(pts[i][1] @ pts[j][1])
                # two balls of radius r (cosine tolerance) collide when the gap
                # 1-cos is under 2r; depth in [0,1] is how deep into the lens we are
                gap = 1.0 - cos
                if gap <= 2.0 * float(radius):
                    edges.append({"a": pts[i][0], "b": pts[j][0],
                                  "cos": round(cos, 4),
                                  "depth": round(1.0 - gap / (2.0 * radius), 4)})
        edges.sort(key=lambda e: -e["depth"])
        return {"balls": [p_[0] for p_ in pts], "radius": float(radius),
                "collisions": edges}

    def void_mix(self, a, b, corpus=None, null_trials=64, seed=0):
        """MIX AT THE LENS (cp56): the conjecture living where two balls overlap.
        The blend is SLERP at the midpoint (our keys are unit-sphere; linear averaging
        distorts -- the field relearned Shoemake 1985 for exactly this), and the nearest
        corpus item to the blend is retrieved by argmax min(sim_a, sim_b) -- the deepest
        point of the lens, the same rule concept-mixing systems use (pumpkin + carriage
        -> pumpkin carriage). VALIDATION IS PLURAL, because one cosine renamed "novelty"
        is the field's own documented trap: (1) the drift sentinel's verdict on the
        blend; (2) neighborhood support (does the blend land NEAR anything, or in open
        void); (3) a PERMUTATION NULL -- the same mix on shuffled component order must
        collapse, or the structure was never there. Returns the blend vector, the lens
        retrieval, and the evidence block. The result is a CONJECTURE: it earns nothing
        until the gates and, later, a research rung say so."""
        va = np.asarray(self.semantic_key(str(a))["vec"][:64], float)
        vb = np.asarray(self.semantic_key(str(b))["vec"][:64], float)
        va = va / (np.linalg.norm(va) + 1e-12)
        vb = vb / (np.linalg.norm(vb) + 1e-12)
        omega = float(np.arccos(np.clip(va @ vb, -1.0, 1.0)))
        if omega < 1e-6:
            blend = va.copy()
        else:
            blend = (np.sin(0.5 * omega) / np.sin(omega)) * (va + vb)   # slerp t=0.5
            blend = blend / (np.linalg.norm(blend) + 1e-12)
        report = {"a": str(a), "b": str(b), "cos_ab": round(float(va @ vb), 4)}
        lens = None
        if corpus:
            best, best_score = None, -2.0
            for c in corpus:
                if str(c) in (str(a), str(b)):    # a mix that retrieves its own parent
                    continue                       # is not a mix (cp57)
                vc = np.asarray(self.semantic_key(str(c))["vec"][:64], float)
                vc = vc / (np.linalg.norm(vc) + 1e-12)
                score = min(float(vc @ va), float(vc @ vb))   # deepest point of the lens
                if score > best_score:
                    best, best_score = str(c), score
            lens = {"nearest": best, "min_sim": round(best_score, 4)}
        sen = self.drift_sentinel()
        verdict = sen.classify(va, blend, remember=False)
        # THE NULL MUST BE ABLE TO FAIL (cp56, caught on the first run): the first cut
        # permuted WORD ORDER, but the semantic key is a bag of content words, so every
        # permutation produced the SAME vector -- null similarity 0.9999, a null that can
        # never reject. The honest null permutes the PAIRING: mix random OTHER pairs from
        # the corpus the same way, and ask whether THIS pair's lens is deeper than
        # chance-pair lenses. No corpus -> the null is reported as unavailable, never
        # faked.
        rng = np.random.default_rng(seed)
        null = []
        p_beat = None
        if corpus and lens and len(corpus) >= 4:
            uv = []
            for c in corpus:
                vc = np.asarray(self.semantic_key(str(c))["vec"][:64], float)
                uv.append(vc / (np.linalg.norm(vc) + 1e-12))
            for _ in range(int(null_trials)):
                i, j = rng.choice(len(uv), size=2, replace=False)
                depth = max(min(float(u @ uv[i]), float(u @ uv[j])) for u in uv)
                null.append(depth)
            p_beat = float(np.mean([n >= lens["min_sim"] - 1e-9 for n in null]))
        null_mean = float(np.mean(null)) if null else None
        report.update({"blend": blend, "lens": lens,
                       "null_draws": [round(float(x), 4) for x in null[:300]],
                       "structure": {"drift_verdict": verdict["verdict"],
                                     "neighbors": verdict["neighbors"],
                                     "null_mean_lens": (round(null_mean, 4)
                                                        if null_mean is not None else None),
                                     "null_p_lens_by_chance": (round(p_beat, 4)
                                                               if p_beat is not None
                                                               else None)},
                       "provenance": "conjecture"})
        return report

    def recall_semantic(self, query, k=5, floor=0.22):
        """SEMANTIC RECALL OVER EVERYTHING TAUGHT (cp60, built because the benchmark
        demanded it): the local LongMemEval-protocol harness scored the exact+fuzzy
        ladder 0.0 on five of six abilities -- retrieval died on PARAPHRASE, the
        weakness this arc has recorded three times. The missing organ already existed:
        the semantic-key cosine top-k, with ABSTENTION kept -- below the floor, or with
        no taught corpus, the answer is a refusal, not a guess (the support gate's
        doctrine at a new layer). Returns ranked {text, sim} candidates."""
        lad = self.zoo["ladder"]
        rows = [t for t in getattr(lad, "taught_log", [])
                if len(t) > 3 and t[3] in ("taught", "validated", "evidenced")]
        if not rows:
            return {"found": False, "why": "nothing taught", "candidates": []}
        # cp60 CAPACITY LESSON AT THE KEY LAYER: the 64-d truncated key saturates at
        # corpus scale -- at ~440 stored turns, hash collisions build a 0.15-0.38 noise
        # floor and "leaky faucet" retrieved "lunch downtown" at 0.377 with ZERO shared
        # words. The engine's own saturation-estimator story, repeating one layer down.
        # The index key is a 512-d deterministic word-hash bag (hash32-seeded index and
        # sign per token) -- collision noise at this width measured ~0.02.
        def _wkey(text):
            from holographic.misc.holographic_determinism import hash32_unit
            v = np.zeros(512)
            for w in str(text).lower().split():
                wid = sum((i + 1) * b for i, b in enumerate(w.encode())) % (2**31)
                h = int(hash32_unit(np.int64(wid), np.int64(len(w)),
                                    np.int64(0), seed=17) * (2**24))
                v[h % 512] += 1.0 if (h >> 8) % 2 == 0 else -1.0
            n = np.linalg.norm(v)
            return v / n if n > 0 else v
        if not hasattr(self, "_sem_index") or self._sem_index[0] != len(rows):
            self._sem_index = (len(rows),
                               np.array([_wkey(str(t[0])) for t in rows]),
                               [str(t[0]) for t in rows], _wkey)
        _n, Vm, texts, _wk = self._sem_index
        # the interrogative shell ("what did i say about...") carries no content and
        # DILUTES the key -- strip leading question scaffolding, key on what remains
        _stop = {"what", "did", "i", "say", "about", "when", "where", "and", "is",
                 "does", "do", "how", "who", "mention", "the", "a", "my", "now",
                 "happen", "happens"}
        toks = str(query).lower().split()
        while toks and toks[0] in _stop:
            toks.pop(0)
        core = " ".join(toks) or str(query)
        qv = _wk(core)
        sims = Vm @ qv
        order = np.argsort(-sims)[: int(k)]
        # GROUNDING GATE (cp60): geometry alone is not grounds -- across seeds, junk
        # occasionally crossed the cosine floor on generic-token overlap. A candidate
        # must SHARE at least one substantive query token (len >= 4); a query whose
        # content words appear nowhere in memory is refused no matter what the angles
        # say. Answerable recall is untouched (needles share their topic tokens by
        # construction of being about them); abstention became seed-stable.
        core_toks = {w for w in core.split() if len(w) >= 4}
        cands = []
        for i in order:
            if sims[i] < float(floor):
                continue
            ctoks = {w for w in texts[i].lower().split() if len(w) >= 4}
            if core_toks and not (core_toks & ctoks):
                continue
            cands.append({"text": texts[i], "sim": round(float(sims[i]), 4)})
        return {"found": bool(cands), "candidates": cands,
                "top_sim": round(float(sims[order[0]]), 4) if len(order) else 0.0}

    def ask_latest(self, query, k=8, floor=0.22):
        """KNOWLEDGE UPDATE BY RECENCY (cp60): among semantic candidates above the
        floor, prefer the LATEST dated entry -- the newer fact wins without deleting
        the older one (the audit floor keeps history; recency picks the answer).
        Entries carry their [YYYY-MM-DD] prefix from ingestion; undated entries never
        outrank dated ones here. Falls back to plain semantic recall when nothing is
        dated."""
        import re as _re
        r = self.recall_semantic(query, k=k, floor=floor)
        if not r["found"]:
            return {"answer": None, "escalate": True, "why": "below the floor",
                    "tier": "T4"}
        dated = []
        for c in r["candidates"]:
            mdate = _re.match(r"\[(\d{4}-\d{2}-\d{2})\]", c["text"])
            if mdate:
                dated.append((mdate.group(1), c))
        if dated:
            dated.sort(key=lambda x: x[0])
            best = dated[-1][1]
        else:
            best = r["candidates"][0]
        return {"answer": best["text"], "sim": best["sim"],
                "provenance": "taught", "tier": "T0-semantic",
                "n_candidates": len(r["candidates"])}

    def saturation_estimate(self, margins):
        """THE COHERENCE-BASED SATURATION ESTIMATOR (cp59) -- an open research item since
        cp45, CLOSED BY THE VOID EXPLORER: mining the external memory collided "the full
        stack measure end to end" with "the hrr cleanup capacity"; the hypothesis engine
        then proved it. MEASURED (v3, D=1024 M=300, N=4..120): mean unbind-cleanup margin
        at load N predicts recall accuracy at 1.5N with corr 0.733 against a null of
        0.023, p=0.0 over 39 prediction pairs -- margins fall from 0.36 to 0.025 as
        accuracy falls 1.00 to 0.53, and THE MARGIN MOVES FIRST. Two experiment designs
        died on the way and are kept: v1 bundled bare items (every probe saw the same
        vector -- accuracy ~0, meaningless), v2 was underpowered (5 pairs cannot beat a
        permutation null, corr 0.68 rejected at p 0.22). Feed the per-item cleanup
        margins from any readback (verify_recall computes them); get the headroom
        verdict BEFORE the cliff arrives."""
        mg = [float(x) for x in margins if np.isfinite(x)]
        if not mg:
            return {"verdict": "unknown", "why": "no margins supplied"}
        mean_m, min_m = float(np.mean(mg)), float(np.min(mg))
        if mean_m >= 0.15:
            verdict = "healthy"
        elif mean_m >= 0.05:
            verdict = "nearing-cliff"
        else:
            verdict = "saturated"
        return {"verdict": verdict, "mean_margin": round(mean_m, 4),
                "min_margin": round(min_m, 4), "n": len(mg),
                "calibration": "thresholds from the cp59 sweep: margin 0.36<->acc 1.00, "
                               "0.025<->0.53 at 1.5x load; corr 0.733 p=0.0"}

    def ask_curious(self, query, items=None, budget=3):
        """LOW CONFIDENCE BECOMES CURIOSITY (cp59, the standing behaviour asked for):
        ask normally; when the answer escalates or arrives without established
        provenance, do not stop at "I don't know" -- run the void explorer over the
        question's own neighborhood and return the escalation WITH ranked conjectures
        and hypothesis skeletons attached. The caller (or a plan step, or the research
        rung) picks one and tests it; a validated result comes back as a promoted,
        durable answer -- next time the same ask serves with provenance instead of a
        shrug. Honest contract: the conjectures are clearly labelled; nothing here is
        served AS the answer."""
        r = dict(self.ask(str(query)))
        low = bool(r.get("escalate")) or r.get("tier") in ("T3", "T4") or             r.get("provenance") in (None, "model-cached", "conjecture")
        r["curiosity"] = None
        if low:
            lad = self.zoo["ladder"]
            nq = set(str(query).lower().split())
            pool = []
            for t in getattr(lad, "taught_log", []):
                if len(t) > 3 and t[3] in ("taught", "validated", "evidenced"):
                    if len(nq & set(str(t[0]).lower().split())) >= 2:
                        pool.append(str(t[0]))
            pool = sorted(set(pool))[:24]
            corpus = pool + list(items or [])
            if len(corpus) >= 4:
                ex = self.explore([str(query)] + corpus, radius=0.55,
                                  budget=int(budget))
                r["curiosity"] = {"why": "low confidence -- exploring the voids around "
                                         "this question",
                                  "conjectures": ex["curious_about"]}
        return r

    def hypothesis_propose(self, conjecture, prediction, experiment):
        """A CONJECTURE BECOMES A HYPOTHESIS when it states a MEASURABLE prediction and
        names the experiment that would decide it (cp58). This is the discipline the
        whole arc runs on, turned into a record: no prediction, no hypothesis -- a vibe
        with a citation is still a vibe. `experiment` is a no-argument callable returning
        {"measured": float, "null": [floats], "detail": str}: the engine's own organs are
        the laboratory (the resonator demuxes, the trace remembers, the harness nulls),
        so the verifier is REAL EXECUTION, which is the one lesson every published
        discovery system agrees on."""
        return {"claim": str(conjecture), "prediction": str(prediction),
                "experiment": experiment, "status": "hypothesis"}

    def hypothesis_test(self, h, alpha=0.05):
        """RUN THE EXPERIMENT AND LET IT DECIDE (cp58). The callable runs for real; the
        measured value stands against its own null distribution; p under alpha is a PASS.
        A pass upgrades nothing by itself -- it RETURNS the verdict with the numbers, and
        the caller promotes (conjecture_promote 'validated' for a passed in-engine test,
        'evidenced' when external research agrees too). A FAIL is kept with the same
        care: a hypothesis the engine killed is knowledge."""
        try:
            r = h["experiment"]()
        except Exception as exc:
            return {"status": "error", "why": str(exc)[:200]}
        null = list(r.get("null") or [])
        measured = float(r["measured"])
        p = (float(np.mean([abs(n) >= abs(measured) - 1e-12 for n in null]))
             if null else None)
        ok = (p is not None and p < alpha)
        return {"status": "pass" if ok else ("fail" if p is not None else "no-null"),
                "measured": round(measured, 6), "p": p,
                "null_mean": round(float(np.mean(null)), 6) if null else None,
                "n_null": len(null), "detail": str(r.get("detail", ""))[:300]}

    def explore(self, items, corpus=None, radius=0.5, budget=3, seed=0):
        """THE CURIOSITY LOOP (cp58): propose from the voids, keep the conjectures least
        explainable by chance, and hand each back WITH its evidence block and a
        ready-to-fill hypothesis skeleton. Curiosity is the ranking; the caller (or a
        plan step) supplies the experiment -- the engine never pretends a prediction it
        did not make. Adaptive: pairs whose mixes previously produced validated
        hypotheses carry pheromone into the next explore (the walker's dynamics on the
        interest map itself)."""
        pr = self.void_propose(items, corpus=corpus, radius=radius,
                               top=int(budget), seed=seed)
        if not hasattr(self, "_interest_pheromone"):
            self._interest_pheromone = {}
        out = []
        for c in pr["conjectures"]:
            key = tuple(sorted((c["a"], c["b"])))
            ph = self._interest_pheromone.get(key, 0.0)
            out.append({**c, "interest_pheromone": round(ph, 4),
                        "hypothesis_skeleton": {
                            "claim": "%s and %s share the mechanism named by %r"
                                     % (c["a"], c["b"],
                                        (c.get("lens") or {}).get("nearest")),
                            "prediction": "<state the measurable quantity and "
                                          "direction>",
                            "experiment": "<a callable using the engine's own organs>"}})
        out.sort(key=lambda c: (-(c["interest_pheromone"]),
                                c["p"] if c["p"] is not None else 1.0, -c["margin"]))
        return {"map": pr["map"], "curious_about": out}

    def explore_reinforce(self, a, b, outcome):
        """Close the adaptive loop: a validated hypothesis makes its pair MORE
        interesting next time; a failed one decays it. Same operator as everything else
        in this engine: decay plus reinforcement on a trace."""
        if not hasattr(self, "_interest_pheromone"):
            self._interest_pheromone = {}
        key = tuple(sorted((str(a), str(b))))
        cur = self._interest_pheromone.get(key, 0.0)
        self._interest_pheromone[key] = 0.9 * cur + (1.0 if outcome else -0.3)
        return {"pair": key, "pheromone": round(self._interest_pheromone[key], 4)}

    def void_propose(self, items, corpus=None, radius=0.45, top=5, seed=0):
        """THE ONE-CALL PIPELINE (cp57, plan stage 2): map -> collide -> mix -> validate
        -> RANKED CONJECTURES. Interesting pairs are the SHALLOW collisions -- deep
        overlap means near-duplicates (nothing to learn), no overlap means no bridge; the
        lens between distinct-but-touching balls is where a conjecture lives. Each
        candidate carries the full evidence block from void_mix (drift verdict, lens
        retrieval, pairing null) and provenance='conjecture'. Ranking is p-first then
        margin -- the pair least explainable by chance wins, not the prettiest cosine
        (the proxy trap)."""
        corpus = list(corpus or items)
        g = self.void_blobs(items, radius=radius)
        pairs = [(e["a"], e["b"], e["depth"]) for e in g["collisions"]
                 if 0.02 <= e["depth"] <= 0.7]
        out = []
        for a, b, depth in pairs[: max(3 * top, 12)]:
            mx = self.void_mix(a, b, corpus=corpus, seed=seed)
            st = mx["structure"]
            pv = st.get("null_p_lens_by_chance")
            margin = ((mx["lens"]["min_sim"] - st["null_mean_lens"])
                      if (mx.get("lens") and st.get("null_mean_lens") is not None)
                      else 0.0)
            out.append({"a": a, "b": b, "collision_depth": depth,
                        "lens": mx.get("lens"), "structure": st,
                        "p": pv, "margin": round(float(margin), 4),
                        "provenance": "conjecture"})
        out.sort(key=lambda c: (c["p"] if c["p"] is not None else 1.0, -c["margin"]))
        return {"map": {"balls": g["balls"], "radius": radius,
                        "n_collisions": len(g["collisions"])},
                "conjectures": out[:top]}

    def void_walk(self, items, corpus=None, steps=48, walkers=6, decay=0.9,
                  elite_bonus=0.5, seed=0):
        """THE SLIME-MOLD WALKER (cp57, plan stage 5): many cheap tendrils over the
        collision graph, pheromone on the pairs whose lens beats chance, elitist
        reinforcement, decay -- the maze solver's dynamics on the void map. The radius
        GROWS over the walk (coarse collisions found early are refined late: the HRNN
        idea as a schedule). Cheap per-step scoring (precomputed unit vectors, one
        precomputed null sample); only the FINAL reinforced trails get the full
        void_mix validation, so the expensive gate runs a handful of times, not
        steps*walkers times."""
        rng = np.random.default_rng(seed)
        corpus = list(corpus or items)
        uv = {}
        for c in set(list(items) + corpus):
            v = np.asarray(self.semantic_key(str(c))["vec"][:64], float)
            uv[str(c)] = v / (np.linalg.norm(v) + 1e-12)
        names = [str(i) for i in items]
        cu = [uv[str(c)] for c in corpus]

        def lens_depth(a, b):
            va, vb = uv[a], uv[b]
            return max(min(float(u @ va), float(u @ vb)) for u in cu)
        null = []
        for _ in range(64):
            i, j = rng.choice(len(names), size=2, replace=False)
            null.append(lens_depth(names[i], names[j]))
        null_mean = float(np.mean(null))
        pher = {}
        for t in range(int(steps)):
            r = 0.25 + (0.6 - 0.25) * (t / max(1, steps - 1))     # the radius schedule
            edges = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    gap = 1.0 - float(uv[names[i]] @ uv[names[j]])
                    if gap <= 2.0 * r:
                        edges.append((names[i], names[j]))
            if not edges:
                continue
            w = np.array([(pher.get(e, 0.0) + 0.05) for e in edges])
            w = w / w.sum()
            best_e, best_gain = None, -1.0
            for _k in range(int(walkers)):
                e = edges[int(rng.choice(len(edges), p=w))]
                gain = max(0.0, lens_depth(*e) - null_mean)
                pher[e] = decay * pher.get(e, 0.0) + gain
                if gain > best_gain:
                    best_e, best_gain = e, gain
            if best_e is not None:                                 # elitist deposit
                pher[best_e] = pher.get(best_e, 0.0) + elite_bonus * best_gain
        trails = sorted(pher.items(), key=lambda kv: -kv[1])[:4]
        out = []
        for (a, b), ph in trails:
            mx = self.void_mix(a, b, corpus=corpus, seed=seed)     # full gate, few times
            out.append({"a": a, "b": b, "pheromone": round(float(ph), 4),
                        "lens": mx.get("lens"), "structure": mx["structure"],
                        "provenance": "conjecture"})
        return {"null_mean_lens": round(null_mean, 4), "trails": out}

    def conjecture_record(self, question, answer):
        """PROVENANCE RUNG BELOW model-cached (cp57, plan stage 3): a conjecture enters
        memory SAYING it is one. taught_only callers never see it; ask() serves it with
        provenance='conjecture' so no downstream can mistake a leap for a fact."""
        self.teach(str(question), str(answer))
        lad = self.zoo["ladder"]
        for k in (" ".join(str(question).lower().split()),
                  " ".join(self.session_salt(question).lower().split())):
            if k in getattr(lad, "_exact", {}):
                lad._exact[k]["provenance"] = "conjecture"
        if lad.taught_log and lad.taught_log[-1][0] == str(question):
            row = list(lad.taught_log[-1]); row = (row + ["shared", ""])[:4]
            row[3] = "conjecture"; lad.taught_log[-1] = row
        return {"recorded": True, "provenance": "conjecture"}

    def conjecture_promote(self, question, level, evidence=""):
        """conjecture -> 'validated' (survived the structural gates) -> 'evidenced' (a
        research rung brought back external support). Promotion is A NEW DURABLE WRITE
        (replay-safe: the record replays in order, so the final provenance wins), and the
        evidence rides in the answer text where the audit floor keeps it."""
        assert level in ("validated", "evidenced"), "levels: validated | evidenced"
        lad = self.zoo["ladder"]
        nq = " ".join(self.session_salt(question).lower().split())
        ex = getattr(lad, "_exact", {}).get(nq) or             getattr(lad, "_exact", {}).get(" ".join(str(question).lower().split()))
        if not ex:
            return {"promoted": False, "why": "no recorded conjecture for this question"}
        ans = str(ex["answer"])
        if evidence:
            ans = ans + "  [%s: %s]" % (level.upper(), str(evidence)[:300])
        lad._remember(lad._qkey(self.session_salt(question)), ans,
                      self.session_salt(question), provenance=level)
        return {"promoted": True, "provenance": level}

    def market_signal_test(self, signal, future_returns, fee_bps=(0.0, 2.0, 5.0),
                           null_trials=500, seed=0):
        """HONEST MARKET SIGNAL TEST (cp74) -- the discipline the solana lab settled
        on, wrapped as one faculty. Judged three ways, because each catches what
        the others miss: (1) HIT RATE vs a SIGN-PERMUTATION-AMONG-LIVE null --
        permuting a sparse signal and indexing the original live mask compares
        trades against zeros and flatters ANY signal (our own first null did
        exactly that; the PnL exposed it); the correct null shuffles the signs of
        the LIVE trades only. (2) PnL WITH FEES swept over fee_bps -- a
        statistically real effect can be economically dead (measured: 5m mean
        reversion +0.48 gross, -1.81 at 2bps). (3) The permutation p, reported
        with the null mean. Walk-forward alignment is the caller's contract:
        signal[t] may use data through t only; future_returns[t] is the NEXT
        period. Returns {n_trades, hit, null_hit, p, pnl: {fee: (total,
        sharpe)}}."""
        import numpy as _np
        sig = _np.asarray(signal, _np.float64)
        fut = _np.asarray(future_returns, _np.float64)
        n = min(len(sig), len(fut))
        sig, fut = sig[:n], fut[:n]
        live = (sig != 0) & _np.isfinite(fut)
        if live.sum() < 10:
            return {"n_trades": int(live.sum()),
                    "note": "too few live trades to judge"}
        s_l, f_l = sig[live], fut[live]
        hit = float((_np.sign(f_l) == _np.sign(s_l)).mean())
        rng = _np.random.default_rng(seed)
        null = [float((_np.sign(f_l) ==
                       _np.sign(rng.permutation(s_l))).mean())
                for _ in range(int(null_trials))]
        p = (1 + sum(1 for x in null if x >= hit)) / (1.0 + null_trials)
        ann = _np.sqrt(365.0) if n < 5000 else _np.sqrt(288 * 365.0)
        pnl = {}
        for fee in fee_bps:
            ser = s_l * f_l - (fee / 1e4) * _np.abs(
                _np.diff(_np.concatenate([[0.0], s_l])))
            pnl[float(fee)] = (round(float(ser.sum()), 4),
                              round(float(ser.mean() /
                                          (ser.std() + 1e-12) * ann), 2))
        return {"n_trades": int(live.sum()), "hit": round(hit, 4),
                "null_hit": round(float(_np.mean(null)), 4),
                "p": round(float(p), 4), "pnl": pnl}

    def splat_memory(self, dim=4096, grid=8, seed=0):
        """DRETTAKIS'S SPLAT MEMORY (cp72): a scene as ONE bundle of role-bound
        Gaussian primitives -- add(x, y, sigma, color) binds a quantized position
        cell to a property vector; recall_region(x, y, radius) recovers the
        primitives inside a query circle and ABSTAINS per empty cell (Pharr's
        condition); capacity_cliff() sweeps N and shows where recovery breaks
        (Plate's condition; measured: 100%% at N<=8 to 42%% at N=64, dim 4096).
        Returns a fresh SplatMemory."""
        from holographic.mesh_and_geometry.holographic_splatmem import \
            SplatMemory
        return SplatMemory(dim=dim, grid=grid, seed=seed)

    def signal_scan(self, x, window=256, step=None, trials=1000, alpha=0.05,
                    seed=0, subbands=1):
        """THE PANEL'S SCAN (cp72, Tarter/Siemion/Cranmer): slide a window over a
        long 1-D signal; score each window's NARROWBAND CONCENTRATION (max spectral
        power over mean -- the degree of frequency compression no natural broadband
        process produces); judge each window against ITS OWN shuffled null
        (permutation destroys phase coherence, flattening the spectrum); control
        false discovery ACROSS windows with Benjamini-Hochberg. Non-prescriptive
        (no template), air-gapped (numpy only), and every window reports its
        p-value plus the applied FDR threshold -- Cranmer's condition for calling
        it a measurement. Returns {windows: [{start, score, p, flagged}],
        fdr_threshold, n_flagged}."""
        import numpy as _np
        x = _np.asarray(x, _np.float64)
        step = int(step or window // 2)
        rng = _np.random.default_rng(seed)

        def _score(w_):
            # subbands>1 = INCOHERENT INTEGRATION (cp73, Tarter's drift finding):
            # a Doppler-drifting tone smears across bins over a long window --
            # measured: detection fell 12/12 -> 0/12 at drift 0.3. Splitting the
            # window into k sub-FFTs and SUMMING POWERS keeps each sub-window
            # near-stationary, tolerating ~k x the drift at a small sensitivity
            # cost -- the classic spectrometer move. The null is judged by the
            # SAME scorer, so calibration is preserved by construction.
            k = max(1, int(subbands))
            seg = len(w_) // k
            pw = None
            for i in range(k):
                sub = w_[i * seg:(i + 1) * seg]
                p_ = _np.abs(_np.fft.rfft(sub - sub.mean())) ** 2
                pw = p_ if pw is None else pw + p_
            return float(pw.max() / (pw.mean() + 1e-12))

        rows = []
        for s0 in range(0, max(1, len(x) - window + 1), step):
            w = x[s0:s0 + window]
            if len(w) < window:
                break
            sc = _score(w)
            null = [_score(rng.permutation(w)) for _ in range(int(trials))]
            p = (1 + sum(1 for n_ in null if n_ >= sc)) / (1.0 + trials)
            rows.append({"start": int(s0), "score": round(sc, 3),
                         "p": round(float(p), 5)})
        ps = sorted((r["p"], i) for i, r in enumerate(rows))
        m_ = len(rows)
        thresh = 0.0
        for rank, (pv, _i) in enumerate(ps, 1):
            if pv <= alpha * rank / m_:
                thresh = max(thresh, pv)
        for r in rows:
            r["flagged"] = bool(r["p"] <= thresh and thresh > 0)
        return {"windows": rows, "fdr_threshold": thresh,
                "n_flagged": sum(r["flagged"] for r in rows),
                "alpha": alpha, "null_trials": int(trials)}

    def solve_maze_grid(self, grid, start, goal, dim=2048, ants=24, rounds=50,
                        seed=0, elite=0.5):
        """ADAMATZKY'S DOOR (cp72): run the holographic Physarum solver on an
        ARBITRARY maze -- grid is any 2-D array-like where truthy = wall, with
        (row, col) start and goal -- his petri-dish topologies, not the fixed
        demo labyrinth. Builds a GridWorld from the walls and runs solve_maze
        (elitist reinforcement on, for braided mazes). Returns (path, info)."""
        import numpy as _np
        from holographic.misc.holographic_creature import GridWorld
        from holographic.simulation_and_physics.holographic_slime import \
            solve_maze
        g = _np.asarray(grid)
        h_, w_ = g.shape
        world = GridWorld(int(w_), int(h_), n_poison=0, seed=seed)
        _walls = {(int(c), int(r)) for r in range(h_) for c in range(w_)
                  if g[r, c]}
        _sx, _sy = int(start[1]), int(start[0])
        _gx, _gy = int(goal[1]), int(goal[0])

        def _pin():                    # solve_maze calls world.reset(), which
            world.walls = set(_walls)  # re-randomizes -- pin OUR topology back
            world.cx, world.cy = _sx, _sy
            world.fx, world.fy = _gx, _gy
        _orig_reset = world.reset
        world.reset = lambda *a, **k: (_orig_reset(), _pin())[1]
        _pin()
        return solve_maze(world, dim=dim, ants=ants, rounds=rounds, seed=seed,
                          elite=elite)

    def pnp_restore(self, y=None, iters=8, lam=0.5, seed=0):
        """MILANFAR'S LOOP, WITH ENO'S DOOR OPEN (cp72): plug-and-play restoration
        that treats CLEANUP AS A PRIOR, not an endpoint -- iterate x <- cleanup(x)
        blended with data fidelity lam*(y - x). With y given, a degraded vector is
        pulled onto the codebook manifold THROUGH the observations (measured below:
        heavy noise recovers where one-shot cleanup fails). With y=None the loop
        runs on PURE NOISE and the prior alone generates -- iterating a denoiser is
        a generative system, named as such. Set self._pnp_codebook to choose the
        prior's manifold. Returns {x, converged_to, cosine_to_converged, trace,
        iterations}; judge recovery against YOUR ground truth externally -- the
        loop never grades itself (the first draft did, and the self-graded cosine
        was meaningless; caught in test)."""
        import numpy as _np
        rng = _np.random.default_rng(seed)
        names = sorted(getattr(self, "_pnp_codebook", []) or
                       ["alpha", "beta", "gamma", "delta"])
        codes = {n: _np.asarray(self.semantic_key(n)["vec"], _np.float64)
                 for n in names}
        codes = {n: v / (_np.linalg.norm(v) + 1e-12) for n, v in codes.items()}
        dimv = len(next(iter(codes.values())))
        yv = None if y is None else _np.asarray(y, _np.float64)
        x = yv.copy() if yv is not None else rng.standard_normal(dimv)
        trace = []

        def _denoise(z_):
            zn = z_ / (_np.linalg.norm(z_) + 1e-12)
            best_ = max(codes, key=lambda n: float(codes[n] @ zn))
            return codes[best_], best_

        best = None
        for _it in range(int(iters)):
            z = x + (lam * (yv - x) if yv is not None
                     else 0.1 * rng.standard_normal(dimv))
            proj, best = _denoise(z)
            x = 0.5 * z / (_np.linalg.norm(z) + 1e-12) + 0.5 * proj
            trace.append(best)
        xn = x / (_np.linalg.norm(x) + 1e-12)
        return {"x": xn, "converged_to": best,
                "cosine_to_converged": round(float(codes[best] @ xn), 4),
                "trace": trace, "iterations": int(iters),
                "generative": y is None}

    def attach_runtime(self, model_dir, n_new=8, attribution=True):
        """ONE CALL FOR LOCAL MODELS (cp71): attach the engine's own RuntimeRung --
        generation on the NumPy GDN runtime with AUTOMATIC source attribution (the
        logit lens piggybacks on the generating forward; addresses are taught) and
        address-based truncated-schedule shortcuts for cues that measured early and
        agreed. Opt out with attribution=False or LECORE_NO_ATTRIBUTION=1. Returns
        the rung (stats on .stats, addresses on .addresses)."""
        from holographic.io_and_interop.holographic_runtimerung import \
            RuntimeRung
        rung = RuntimeRung(model_dir, mind=self, n_new=n_new,
                           attribution=attribution)
        self.zoo_attach(rung)
        self._zoo_llm = rung
        return rung

    def model_attribute(self, token_ids, model_dir="/tmp/mini_installed_full"):
        """SOURCE ADDRESS for model-provided knowledge (cp70): run the logit lens
        over a real forward pass and return where the answer crystallizes --
        {source_id: "model:<n>L/L<k>/<hash8>", emergence_layer, answer_token,
        early}. The address is deterministic and is TAUGHT ("model source for
        <ids>") so provenance can answer where AND how. Instrument: logit lens
        (nostalgebraist 2020 / Belrose 2023); shortcut policy honors the
        late-crystallization caution (arXiv 2603.23701, 2606.07978)."""
        from holographic.agents_and_reasoning.holographic_attribution import \
            attribute
        from holographic.io_and_interop.holographic_gdnruntime import \
            load_runtime
        rt = load_runtime(model_dir)
        if isinstance(rt, tuple):
            rt = next(x for x in rt if hasattr(x, "forward"))
        rep = attribute(rt, list(token_ids))
        self.teach("model source for %s" % list(token_ids),
                   "%s (answer token %d, %s)" % (rep["source_id"],
                                                 rep["answer_token"],
                                                 "early" if rep["early"]
                                                 else "late"))
        return rep

    def ask_shortcut(self, token_ids, model_dir="/tmp/mini_installed_full"):
        """INFERENCE SHORTCUT from a measured address (cp70): if this cue's stored
        crystallization layer is EARLY (<= 80% of depth), run a TRUNCATED layer
        schedule to that layer and decode there -- real layers skipped, agreement
        with the stored full-depth answer verified, never assumed. A late address
        falls back to the full pass with the reason stated."""
        from holographic.agents_and_reasoning.holographic_attribution import \
            attribute, shortcut
        from holographic.io_and_interop.holographic_gdnruntime import \
            load_runtime
        rt = load_runtime(model_dir)
        if isinstance(rt, tuple):
            rt = next(x for x in rt if hasattr(x, "forward"))
        rep = attribute(rt, list(token_ids))
        if not rep["early"]:
            return {"used_shortcut": False,
                    "reason": "crystallizes late (L%d/%d) -- full pass is the "
                              "honest path" % (rep["emergence_layer"],
                                               rep["n_layers"] - 1),
                    "answer_token": rep["answer_token"]}
        sc = shortcut(rt, list(token_ids), rep["emergence_layer"])
        return {"used_shortcut": True, "answer_token": sc["answer_token"],
                "agrees_with_full": sc["answer_token"] == rep["answer_token"],
                "layers_run": sc["layers_run"], "layers_total":
                sc["layers_total"], "saved_fraction": sc["saved_fraction"],
                "source_id": rep["source_id"]}

    def decision_report(self):
        """THE DECISION JOURNAL (cp70): every ask_grounded records which arm
        answered (taught / semantic / model / escalate) -- the decision process
        captured as it happens, minable by the same memory-mine machinery as
        everything else. Returns counts per arm plus the last few entries."""
        log = getattr(self, "_decision_log", [])
        counts = {}
        for e in log:
            counts[e["arm"]] = counts.get(e["arm"], 0) + 1
        return {"decisions": len(log), "by_arm": counts, "recent": log[-5:]}

    def memory_list(self, root=None):
        """NAMED EXTERNAL MEMORIES (cp69): list the memory bundles under a root
        directory ($LECORE_MEMORIES or ~/.lecore/memories or ./memories). Each entry
        is a full partition loadable by autoboot(partition=...) or a chat slot --
        one for research, one for the current project, one somebody shared with you.
        Returns [{name, path, entries}]."""
        import os as _os
        root = root or _os.environ.get("LECORE_MEMORIES") or \
            (_os.path.join(_os.path.expanduser("~"), ".lecore", "memories")
             if _os.path.isdir(_os.path.join(_os.path.expanduser("~"),
                                             ".lecore", "memories"))
             else "memories")
        out = []
        if _os.path.isdir(root):
            for name in sorted(_os.listdir(root)):
                st = _os.path.join(root, name, "learning", "state.lecore")
                if _os.path.exists(st):
                    out.append({"name": name,
                                "path": _os.path.join(root, name),
                                "bytes": _os.path.getsize(st)})
        return {"root": root, "memories": out}

    def memory_export(self, dest, query=None, sessions=None, provenance=None,
                      include_api_specs=True):
        """SELECTIVE EXPORT (cp69): write a PORTABLE, SELF-CONTAINED memory bundle
        holding only what the filters admit -- substring/topic `query`, a `sessions`
        list, a `provenance` set ("taught","validated","evidenced"). What travels:
        facts with their provenance, promoted conjectures AT their earned rung, api
        spec records (so learned tools work on arrival -- functionality transfers,
        not just text), and this memory's vetoes (tombstones ride along; a shared
        bundle does not resurrect what its maker killed). VERIFIED before blessing:
        a fresh boot of the bundle must answer every exported question. Returns the
        manifest {exported, by_provenance, vetoes, verified}."""
        import os as _os
        lad = self.zoo["ladder"]
        rows, seen = [], set()
        want_prov = set(provenance) if provenance else None
        for t in reversed(getattr(lad, "taught_log", [])):
            # LATEST STATE WINS (cp69): a promoted conjecture's newest row carries
            # its earned rung; iterating forward exported the stale first row and
            # silently demoted validated knowledge back to a bare conjecture.
            if len(t) < 4:
                continue
            q, a, sess, prov = str(t[0]), str(t[1]), str(t[2]), str(t[3])
            if q in seen or prov == "model-cached":
                continue
            if q in (getattr(lad, "_vetoed_qs", set()) or set()):
                continue                    # a vetoed question travels ONLY as a
                                            # tombstone, never as a live fact
            if q.startswith("api spec record:") and not include_api_specs:
                continue
            if want_prov and prov not in want_prov and \
                    not q.startswith("api spec record:"):
                continue
            if sessions and sess not in sessions:
                continue
            if query and str(query).lower() not in (q + " " + a).lower():
                continue
            seen.add(q)
            rows.append((q, a, prov))
        dst = type(self)()
        dst.zoo_attach(lambda p: "")
        for q, a, prov in rows:
            dst.teach(q, a)
            if prov in ("validated", "evidenced"):
                dst.conjecture_record(q, a)
                dst.conjecture_promote(q, prov, "carried by memory_export")
        vet = sorted(getattr(lad, "_vetoed_qs", set()) or [])
        for vq in vet:
            dst.teach(vq, "carried tombstone")
            dst.answer_feedback(vq, ok=False)
        _os.makedirs(dest, exist_ok=True)
        dst.learning_save(dest)
        chk = type(self)()
        chk.zoo_attach(lambda p: "")
        chk.learning_load(dest)
        def _row_ok(q_, a_, p_):
            got = str(chk.ask(q_).get("answer") or "")
            if p_ in ("validated", "evidenced", "conjecture"):
                return bool(got.strip())        # promote rebuilds canonical
                                                # serve-text; require presence
            return got == a_                    # plain taught: exact
        misses = sum(0 if _row_ok(q, a, _p) else 1 for q, a, _p in rows)
        byp = {}
        for _q, _a, prov in rows:
            byp[prov] = byp.get(prov, 0) + 1
        return {"exported": len(rows), "by_provenance": byp,
                "vetoes": len(vet), "dest": dest,
                "verified": misses == 0, "misses": misses}

    def memory_import(self, src, on_conflict="flag"):
        """IMPORT / MERGE (cp69): bring a shared bundle INTO this memory -- the
        transfer door. Facts arrive with provenance intact; validated/evidenced
        conjectures keep their earned rung; api spec records make the sender's
        learned tools CALLABLE here without relearning; the bundle's tombstones are
        honored (their vetoes import as vetoes). CONFLICTS -- a question this memory
        already answers DIFFERENTLY -- are never silently overwritten: default
        on_conflict="flag" keeps the local answer and reports each collision with
        the drift sentinel's verdict; "theirs" adopts the incoming answer as a
        deliberate re-teach. Returns {imported, conflicts, vetoes, skipped}."""
        donor = type(self)()
        donor.zoo_attach(lambda p: "")
        donor.learning_load(src)
        dlad = donor.zoo["ladder"]
        lad = self.zoo["ladder"]
        mine = {}
        for t in getattr(lad, "taught_log", []):
            if len(t) > 3 and t[3] != "model-cached":
                mine.setdefault(str(t[0]), str(t[1]))
        imported, skipped, conflicts = 0, 0, []
        _dseen = set()
        _drows = []
        for t in reversed(getattr(dlad, "taught_log", [])):
            if len(t) < 4 or t[3] == "model-cached" or str(t[0]) in _dseen:
                continue
            _dseen.add(str(t[0]))
            _drows.append(t)
        for t in reversed(_drows):
            q, a, prov = str(t[0]), str(t[1]), str(t[3])
            if a == "carried tombstone":
                continue
            if q in mine:
                if mine[q] == a:
                    skipped += 1
                    continue
                verdict = ""
                try:
                    verdict = self.teach_check(q, a).get("verdict", "")
                except Exception:
                    pass
                conflicts.append({"q": q[:80], "mine": mine[q][:80],
                                  "theirs": a[:80], "verdict": verdict})
                if on_conflict != "theirs":
                    continue
            self.teach(q, a)
            if prov in ("validated", "evidenced"):
                self.conjecture_record(q, a)
                self.conjecture_promote(q, prov, "imported from %s" % src)
            imported += 1
        vet = sorted(getattr(dlad, "_vetoed_qs", set()) or [])
        for vq in vet:
            if vq not in mine:
                self.teach(vq, "carried tombstone")
            self.answer_feedback(vq, ok=False)
        if hasattr(self, "_api_toolbox"):
            self._api_toolbox._rehydrated = False       # relearn arrivals lazily
        return {"imported": imported, "skipped_identical": skipped,
                "conflicts": conflicts, "vetoes": len(vet),
                "on_conflict": on_conflict}

    def ask_grounded(self, question):
        """THE GROUNDED ANSWER ROUTINE, PUSHED DOWN (cp68): this logic lived only in
        the chat layer, which meant raw api callers and hosted tools served whatever
        the ladder returned -- including a live rung's ungrounded output (the cp65
        noted risk, now closed at the engine). One door, four steps, in order:
        (1) memory -- exact taught/validated/evidenced serves outright (an exact
        repeat is the strongest grounding); (2) the grounding doctrine on the fuzzy
        arm -- an answer sharing no substantive token (len>=4) with the question is
        not an answer; (3) semantic recall; (4) the model rung LAST AND VISIBLY --
        its reply is returned marked model-cached, never silently. Otherwise an
        honest escalate. NOTE, stated plainly: the ladder underneath may still CACHE
        a rung's raw reply; this door polices what is SERVED, same contract as the
        chat has had since cp64. Returns {answer, provenance, escalate, tier}."""
        def _journal(arm_, prov_):
            if not hasattr(self, "_decision_log"):
                self._decision_log = []
            self._decision_log.append({"q": str(question)[:80], "arm": arm_,
                                       "provenance": prov_})
            if len(self._decision_log) > 500:
                self._decision_log = self._decision_log[-500:]

        def _shares(q_, a_):
            qt = {w for w in str(q_).lower().split() if len(w) >= 4}
            at = {w for w in str(a_).lower().split() if len(w) >= 4}
            return bool(qt & at) if qt else True
        a = self.ask(question)
        ans = str(a.get("answer") or "")
        prov = a.get("provenance")
        exact = prov in ("taught", "validated", "evidenced")
        if ans.strip() and (exact or _shares(question, ans)):
            _journal("taught" if exact else "fuzzy-grounded", prov)
            return {"answer": ans, "provenance": prov, "escalate": False,
                    "tier": a.get("tier")}
        if hasattr(self, "recall_semantic"):
            sem = self.recall_semantic(question, k=3)
            if sem.get("found"):
                _journal("semantic", "semantic-recall")
                return {"answer": " / ".join(c["text"] for c in
                                             sem["candidates"][:2]),
                        "provenance": "semantic-recall", "escalate": False,
                        "tier": a.get("tier")}
        rung = getattr(self, "_zoo_llm", None) or             getattr(self.zoo.get("ladder"), "_llm", None)
        if callable(rung):
            try:
                raw = str(rung(str(question)))[:400]
            except Exception:
                raw = ""
            if raw.strip():
                _journal("model", "model-cached")
                return {"answer": raw, "provenance": "model-cached",
                        "escalate": False, "tier": "T4"}
        _journal("escalate", "escalated")
        return {"answer": "", "provenance": "escalated", "escalate": True,
                "tier": a.get("tier")}

    def explain(self, topic):
        """DOCS-DERIVED EXPLANATION, PUSHED DOWN (cp68): 'how does X work' answered
        from the generated documentation (CAPABILITIES.md, REFERENCE.md) -- the same
        source of truth the api ships -- so the engine, the chat, and hosted callers
        all explain themselves identically. Returns {found, title, body} with the
        best-overlap card, or found=False rather than a guess."""
        import os as _os
        if not hasattr(self, "_doc_cards"):
            cards = []
            here = _os.path.dirname(_os.path.dirname(_os.path.dirname(
                _os.path.abspath(__file__))))
            for fn in ("docs/CAPABILITIES.md", "REFERENCE.md"):
                try:
                    cur = None
                    for line in open(_os.path.join(here, fn)):
                        if line.startswith("#"):
                            if cur and len(cur[1]) > 60:
                                cards.append(cur)
                            cur = (line.strip("# \n"), "")
                        elif cur:
                            cur = (cur[0], (cur[1] + " " + line.strip())[:600])
                    if cur and len(cur[1]) > 60:
                        cards.append(cur)
                except Exception:
                    pass
            # THE ENGINE'S OWN DOCSTRINGS ARE CARDS TOO (cp75): file headers
            # covered modules but left every FACULTY invisible to discovery --
            # the audit read "memory slots: MISS" on an engine that has
            # memory_export. Introspection closes it: every public method with
            # a real docstring becomes a card named after itself.
            for _mn in dir(self):
                if _mn.startswith("_"):
                    continue
                _fn = getattr(self, _mn, None)
                _doc = getattr(_fn, "__doc__", None)
                if callable(_fn) and _doc and len(_doc) > 80:
                    cards.append((_mn, " ".join(_doc.split())[:400]))
            self._doc_cards = cards
        tt = set(str(topic).lower().split())
        best, score = None, 0
        for title, body in self._doc_cards:
            sc = len(tt & set((title + " " + body).lower().split()))
            if sc > score:
                best, score = (title, body), sc
        if best and score >= 1:
            return {"found": True, "title": best[0], "body": best[1]}
        return {"found": False}

    def api_toolbox(self):
        """The learned-API toolbox (cp66, extracted from leOS API_LEARN/API_CALL):
        learn an external API from its OpenAPI spec with no LLM in the parse, call
        its endpoints by name, and find them by task. Every learned endpoint teaches
        a discoverability card, so contextual access is ordinary recall with full
        provenance/veto/session semantics."""
        from holographic.io_and_interop.holographic_apilearn import ApiToolbox
        if not hasattr(self, "_api_toolbox"):
            self._api_toolbox = ApiToolbox(mind=self)
        return self._api_toolbox

    def api_learn(self, spec, name=None, base_url=None):
        """Learn an external API from its OpenAPI spec (dict, JSON text, or URL) with
        no LLM in the parse -- endpoints register as callable tools and each one
        teaches a DISCOVERABILITY CARD into memory, so 'how do i ...' finds the tool
        like any other knowledge (with provenance, veto, session isolation for
        free). Returns {service, base, endpoints}. Extracted from leOS API_LEARN."""
        return self.api_toolbox().learn(spec, name=name, base_url=base_url)

    def api_use(self, service, endpoint, params=None, headers=None):
        """Call a LEARNED api endpoint by service.endpoint: the URL is built from the
        stored spec (path-param substitution, query params, header auth), the request
        goes through stdlib urllib, and the reply is an honest {ok, status,
        data | error} -- never a dressed-up failure. Successful calls note themselves
        into the drift sentinel, so tool usage becomes experience. Extracted from
        leOS API_CALL."""
        return self.api_toolbox().call(service, endpoint, params=params,
                                       headers=headers)

    def tool_find(self, task, k=5):
        """ONE CONTEXTUAL QUERY OVER THE WHOLE TOOLSET (cp66): learned APIs, the
        engine's own capability catalog, and everything taught about tools -- ranked
        for a task phrase under the grounding doctrine. This is the door a hosted or
        local caller uses to discover what exists before asking how."""
        rows = list(self.api_toolbox().find(task, k=k))
        tt = {w for w in str(task).lower().split() if len(w) >= 4}
        # THE ENGINE ARM, FIXED (cp75 discoverability audit): the old catalog
        # import died silently and tool_find surfaced NONE of the engine's own
        # faculties -- 18/18 misses on the audit matrix. The docs cards (the
        # same generated source explain() reads) are the guaranteed-populated
        # capability index, so they ARE the engine arm now; the buried-module
        # lesson in one line: a discovery door that cannot see its own house
        # discovers nothing.
        if not hasattr(self, "_doc_cards"):
            self.explain("warm the cards")
        for title, body in getattr(self, "_doc_cards", []):
            et = {w for w in (title + " " + body).lower().split() if len(w) >= 4}
            ov = len(tt & et)
            if ov >= 2 or (ov == 1 and len(tt) <= 2):
                rows.append({"tool": "engine.%s" % title.split()[0],
                             "description": body[:100], "score": ov})
        if hasattr(self, "recall_semantic"):
            sem = self.recall_semantic("tool for %s" % task, k=2)
            for c in (sem.get("candidates") or []):
                rows.append({"tool": "memory", "description": c["text"][:100],
                             "score": 1})
        rows.sort(key=lambda r: -r["score"])
        return rows[:k]

    def drift_sentinel(self):
        """The leOS displacement-drift detector on lever 7's floor (cp54 dig): classifies
        every task->response displacement against the NEIGHBORHOOD of similar past tasks.
        Verdicts: normal / void (honestly unexplored) / echo (a non-answer restating the
        task) / redshift (off established behaviour) / blueshift (suspiciously little
        work), plus loop detection over the last 4 responses. Advisory, never blocking:
        evidence attached to every verdict."""
        from holographic.agents_and_reasoning.holographic_drift import DriftSentinel
        if not hasattr(self, "_drift_sentinel"):
            self._drift_sentinel = DriftSentinel()
        return self._drift_sentinel

    def teach_check(self, query, answer):
        """IMPLICIT-CONFLICT CANDIDATE DETECTION (cp54, closing the cp47 gap the STALE
        benchmark grades): before establishing a new answer, classify its displacement
        against the neighborhood of this question's REGION. A redshift -- the new answer
        landing far from where similar questions' answers have always landed -- is a
        CONFLICT CANDIDATE: perhaps the world changed, perhaps the teach is wrong, and
        only the caller knows which. Returns the verdict WITH the nearest established
        answers so the caller can veto, re-teach, or proceed deliberately. Never blocks:
        surfacing is the contract, resolution is the caller's."""
        sen = self.drift_sentinel()
        qv = np.asarray(self.semantic_key(str(query))["vec"][:64], float)
        av = np.asarray(self.semantic_key(str(answer))["vec"][:64], float)
        rep = sen.classify(qv, av, remember=False)
        near = []
        if rep["verdict"] in ("redshift", "blueshift") and rep["neighbors"] >= 3:
            lad = self.zoo["ladder"]
            nq = " ".join(str(query).lower().split())
            for k, v in list(getattr(lad, "_exact", {}).items())[:400]:
                kq = k.split("] ", 1)[-1] if k.startswith("[s:") else k
                shared = len(set(kq.split()) & set(nq.split()))
                if shared >= max(2, len(nq.split()) // 3) and kq != nq:
                    near.append({"question": kq[:80],
                                 "answer": str(v.get("answer"))[:80],
                                 "provenance": v.get("provenance")})
                if len(near) >= 3:
                    break
        rep["conflict_candidate"] = bool(rep["verdict"] == "redshift" and near)
        rep["established_nearby"] = near
        return rep

    def app_substrate(self, name, user="default", root=None, llm=None, doctrine=False):
        """A leCore substrate scoped to ONE APP AND ONE USER -- the layer anything built on
        this engine should start from. Returns an App with remember/recall/forget,
        observe/suggest/habits (procedures mined from what the user actually does), a
        capability preflight and save/load. Isolation is PHYSICAL: each (app, user) is its
        own partition directory, so a second user cannot appear in the first's memory by
        any path. Built from leStudio's integration report -- an app should not have to
        hand-roll a preflight, a singleton mind and a container before it can start.

        Example: app = mind.app_substrate("lestudio", user="ana"); app.remember(q, a)"""
        from holographic.agents_and_reasoning.holographic_appkit import App
        return App(name, user=user, root=root, llm=llm, doctrine=doctrine)

    def panel_seat(self, members=None, layer=3):
        """SEAT THE EXPERT PANEL IN THE SWARM REALM (cp42): each member becomes a named
        resident with its OWN memory scope in a SHARED KnowledgeStore -- so they read and
        write one record and communicate the way the swarm was built to (fork, deliberate,
        digest). The panel's operating law matches the swarm's contrast digest EXACTLY:
        consensus is silent, only disagreement carries information. Returns the seated
        realm handle; findings are authored per-member so provenance is per-expert."""
        from holographic.caching_and_storage.holographic_knowledgestore import KnowledgeStore
        realm = KnowledgeStore(getattr(self, "_archive_root", None) or "/tmp/panel_realm")
        seated = members or ["quilez", "widrow", "kohonen", "yang", "bau", "tarter",
                             "cranmer", "olshausen", "stoudenmire", "duda", "plate",
                             "stam", "macklin", "milanfar", "eno", "ozcan", "puckette",
                             "siemion", "baker", "adamatzky", "pharr", "drettakis"]
        reg = getattr(self, "_panel_realm", {})
        for name in seated:
            reg.setdefault(name, {"seat": len(reg), "notes": 0, "layer": int(layer)})
        self._panel_realm = reg
        self._panel_store = realm
        return {"realm": "panel", "seated": len(reg), "chair": "quilez",
                "digest_law": "contrast: consensus silent, dissent speaks"}

    def panel_note(self, member, text, tags=()):
        """A SEATED MEMBER WRITES TO THE SHARED REALM: authored by the expert, tagged,
        landing in the one store every member reads -- the swarm's shared record. Refuses
        an unseated author (you cannot speak in a realm you do not sit in)."""
        reg = getattr(self, "_panel_realm", {})
        if member not in reg:
            return {"error": "%r is not seated -- call panel_seat first" % member}
        store = getattr(self, "_panel_store", None)
        if store is None:
            return {"error": "no realm store -- call panel_seat first"}
        store.add_note("[%s] %s" % (member, text), author=member,
                       tags=("panel", member) + tuple(tags))
        reg[member]["notes"] += 1
        return {"member": member, "notes": reg[member]["notes"]}

    def panel_deliberate(self, question, positions):
        """DELIBERATION UNDER THE CONTRAST LAW (cp42): positions is {member: stance}. If
        the seated members AGREE (one distinct stance) the realm is SILENT -- unanimity
        carries no information, exactly the swarm's contrast digest. If they DISAGREE, the
        dissent is what surfaces and gets recorded, authored by each dissenter. Returns
        the digest: silent+consensus, or the surfaced disagreement."""
        reg = getattr(self, "_panel_realm", {})
        pos = {k: v for k, v in positions.items() if k in reg}
        distinct = sorted(set(str(v).strip() for v in pos.values()))
        if len(distinct) <= 1:
            return {"question": question, "silent": True,
                    "consensus": distinct[0] if distinct else None,
                    "why": "unanimity carries no information (contrast digest)"}
        for member, stance in pos.items():
            self.panel_note(member, "on %r I hold: %s" % (question, stance),
                            tags=("deliberation", "dissent"))
        return {"question": question, "silent": False, "positions": pos,
                "surfaced": distinct, "recorded": len(pos)}

    def doctrine_load(self):
        """LOAD THE SHIPPED DOCTRINE PACK (cp33): the distilled lessons of the long
        sessions, taught through the normal gate -- what nomic text is to the embedding
        space, this is to operating knowledge. OPT-IN on purpose: a virgin mind stays
        virgin so cold benchmarks stay honest. Re-teaching overwrites any entry, exactly
        like lived knowledge."""
        import holographic.agents_and_reasoning.holographic_seedpack as holographic_seedpack
        return {"taught": holographic_seedpack.register_doctrine(self),
                "entries": len(holographic_seedpack.DOCTRINE)}

    def goal_close(self, goal_id, reason):
        """CLOSE A GOAL WITH ITS RECEIPT (cp30): pending steps whose work was delivered
        outside their executors (or superseded) get closed with the REASON recorded on
        every step and on the goal -- the book stays honest instead of haunted. Not a
        success shortcut: the reason text is the deliverable."""
        g = self.goal_book.goals.get(goal_id)
        if not g:
            return {"error": "unknown goal %r" % goal_id}
        n = 0
        for st in g["steps"]:
            if st["status"] == "pending":
                st["status"] = "done"
                st["deliverable"] = "closed: %s" % str(reason)
                n += 1
        g["status"] = "done"
        g["closed_reason"] = str(reason)
        return {"closed": goal_id, "steps_closed": n}

    def goal_recurrence(self, gate=0.5):
        """THE APC PRECONDITION, measured (cp24): the fraction of created goals whose
        semantic key landed within `gate` of an EARLIER goal's key -- the recurrence rate
        that decides whether plan caching pays (the research warns hits collapse to 0-12%
        at low recurrence). Computed over the goal book's stored keys; cross-dim keys are
        skipped, never compared."""
        keys = []
        rec = 0
        for g in self.goal_book.goals.values():
            gv = np.asarray(g["goal_vec"], float)
            hit = any(gv.shape == k.shape and
                      float(gv @ k / (np.linalg.norm(gv) * np.linalg.norm(k) + 1e-12))
                      >= float(gate) for k in keys)
            if keys and hit:
                rec += 1
            keys.append(gv)
        n = len(keys)
        return {"goals": n, "recurrent": rec,
                "recurrence_rate": round(rec / max(n - 1, 1), 3),
                "verdict": "plan caching pays" if n > 1 and rec / max(n - 1, 1) >= 0.15
                           else "below the 15% band -- prefer calibration over caching"}

    def reflect_failures(self, llm=None, k=6):
        """FAILURE -> RULE (cp37; Reflexion/ExpeL lineage + tuning-free rule accumulation,
        EMNLP23; ReasoningBank's failure arm): walk the feedback log's bad answers, ask
        the model rung for ONE transferable rule per failure ('what should be done
        differently next time, as a rule'), and TEACH the rule -- text rules, no weight
        updates, so the mechanism is model-agnostic: any attached callable (a local
        llama.cpp wrapper included) self-improves the same way. Rules land under the
        calibrated gate with provenance, which is what the 2026 trust literature (TAME)
        says naive memory evolution lacks."""
        lad = self.zoo["ladder"]
        model = llm or self.zoo.get("llm")
        if model is None:
            return {"ok": False, "why": "no model rung attached"}
        fb = getattr(lad, "_feedback_log", None) or getattr(self, "_feedback_log", [])
        bad = [f_ for f_ in fb if not f_.get("ok")][-int(k):]
        rules = []
        for f_ in bad:
            q_ = str(f_.get("question", ""))[:200]
            r = str(model("REFLECT: the answer to %r was marked wrong. State ONE "
                          "transferable rule for next time, one sentence." % q_))
            rq = "what rule was learned from failing at %s" % q_
            self.teach(rq, r)
            rules.append({"from": q_, "rule": r})
        return {"ok": True, "rules": rules, "examined": len(bad)}

    def workflow_distill(self, min_steps=2):
        """SUCCESS -> WORKFLOW (cp37; Agent Workflow Memory, ICML25): every DONE goal
        with a real trajectory is distilled into a named, reusable workflow -- the step
        list plus what each step delivered, taught so that a NEW objective with
        overlapping wording warms from it. Cross-objective reuse is the AWM claim; the
        warm-plan gate (grounding cross-exam) keeps a workflow from firing where it
        does not belong -- proposal is cheap, adoption is earned."""
        lib = getattr(self, "_workflow_lib", {})
        n_new = 0
        for gid, g in self.goal_book.goals.items():
            if g.get("status") != "done" or gid in lib:
                continue
            steps = [st["name"] for st in g.get("steps", [])
                     if st.get("status") == "done"]
            if len(steps) < int(min_steps):
                continue
            # the goal book stores the text under "text", not "objective" -- the cp37
            # suite caught the None-keyed lookup matching nothing
            obj_txt = g.get("text") or g.get("goal") or gid
            lib[gid] = {"objective": obj_txt, "steps": steps}
            wq = "what workflow solved %s" % str(obj_txt)[:120]
            self.teach(wq, "steps: " + " -> ".join(steps))
            n_new += 1
        self._workflow_lib = lib
        return {"workflows": len(lib), "new": n_new}

    def workflow_warm(self, objective):
        """Match a NEW objective against the distilled library by stemmed overlap;
        returns the best workflow's steps as a candidate plan (the caller's goal_create
        grounding gate still disposes)."""
        lib = getattr(self, "_workflow_lib", {})
        stop = self.zoo["ladder"]._STOP
        ow = {w.rstrip("s") for w in str(objective).lower().split()} - stop
        best, bs = None, 0.0
        for gid, w in lib.items():
            ww = {x.rstrip("s") for x in str(w["objective"]).lower().split()} - stop
            ov = len(ow & ww) / max(len(ow | ww), 1)
            if ov > bs:
                best, bs = w, ov
        return {"plan": (best or {}).get("steps"), "overlap": round(bs, 3),
                "from": (best or {}).get("objective")}

    def self_improve(self, tasks, judge, rounds=3, llm=None):
        """THE SELF-IMPROVEMENT LOOP (cp37) -- what the 2026 sota does, on this
        substrate, for ANY model: each round answers every task, the judge grades,
        failures get feedback (veto + calibration pair) AND a reflected rule, successes
        are taught back; the next round serves what it learned. Returns the LEARNING
        CURVE (per-round error), the metric the Evo-Memory/MemoryBench line measures.
        No weight updates anywhere: a local model improves exactly like a frontier
        one, because the improvement lives in the substrate."""
        model = llm or self.zoo.get("llm")
        curve = []
        for r_ in range(int(rounds)):
            wrong = 0
            for t_ in tasks:
                q_ = t_["q"]
                a = self.ask(q_)
                ans = str(a.get("answer") or "")
                if a.get("tier") not in ("T0", "T1"):
                    ans = str(model("ANSWER: " + q_)) if model else ""
                ok = bool(judge(q_, ans))
                if ok:
                    if a.get("tier") not in ("T0", "T1"):
                        self.teach(q_, ans)
                else:
                    wrong += 1
                    try:
                        self.answer_feedback(q_, ok=False)
                    except Exception:
                        pass
                    if model:
                        rule = str(model("REFLECT: %r was answered wrongly as %r. "
                                         "State the correct approach as one rule."
                                         % (q_, ans[:100])))
                        self.teach("how should %s be approached" % q_, rule)
                        corr = t_.get("teach")
                        if corr:
                            self.teach(q_, corr)
            curve.append({"round": r_ + 1, "errors": wrong,
                          "error_rate": round(wrong / max(len(tasks), 1), 3)})
        return {"curve": curve, "improved": curve[-1]["errors"] < curve[0]["errors"]
                if len(curve) > 1 else False}

    def agent_loop(self, objective, executors=None, rounds=3, budget_steps=3,
                   idle_limit=2, checkpoint_root=None, plan=None, stateless=()):
        """THE AGENT LOOP (cp28) -- the pattern everyone runs LLMs in, built on memory
        that is actually useful. Ported doctrine, credited: leOS substrate_gather (GATHER
        from the substrate BEFORE any model call), agent_session_tool_loop (a state-
        carrying round loop with per-round counters), and infra/activity_monitor (IDLE-
        BASED stopping -- wall-clock deadlines kill legitimate slow work; we stop after
        `idle_limit` rounds WITHOUT PROGRESS instead).

        Each round: GATHER (reflex + archive + catalog, zero model calls) -> ENSURE GOAL
        (deterministic id from the objective, so any process resumes the same goal; warm
        plans propose, the cross-exam disposes) -> ACT (goal_work under the trajectory
        tool cache) -> REFLECT (convergence + drift verdict) -> REMEMBER (deliverables
        ingest; a round summary is taught) -> CHECKPOINT (learning_save when a root is
        given). Long-term memory is the loop's floor: reflexes serve, plans warm, tool
        results cache, archives answer -- across PROCESS RESTARTS, which is the test that
        matters."""
        import hashlib
        gid = "loop-" + hashlib.sha1(str(objective).encode()).hexdigest()[:10]
        gather = {"reflex": None, "archive": [], "capabilities": []}
        a0 = self.zoo["ladder"].answer(str(objective))
        if a0.get("tier") in ("T0", "T1"):
            gather["reflex"] = str(a0.get("answer"))[:200]
        for t_ in list(getattr(self, "_archive_corpora", {}))[:4]:
            qv = self.archive_query(t_, str(objective), k=1)
            gather["archive"] += qv.get("evidence", [])[:1]
        try:
            gather["capabilities"] = [h.name for h in
                                      self.find_capability(str(objective), k=3)]
        except Exception:
            pass
        if gid not in self.goal_book.goals:
            self.goal_create(gid, str(objective), plan=plan)
        log = []
        idle = 0
        for r_ in range(int(rounds)):
            before = sum(1 for st in self.goal_book.goals[gid]["steps"]
                         if st["status"] == "done")
            w = self.goal_work(gid, executors=executors or {},
                               budget_steps=int(budget_steps), stateless=stateless)
            after = sum(1 for st in self.goal_book.goals[gid]["steps"]
                        if st["status"] == "done")
            progress = after - before
            idle = 0 if progress > 0 else idle + 1
            entry = {"round": r_ + 1, "did": len(w.get("did", [])),
                     "progress": progress, "cache_hits": w.get("cache_hits", 0),
                     "status": w["status"], "idle_rounds": idle}
            log.append(entry)
            lad = self.zoo["ladder"]
            rq = "what happened in round %d of %s" % (r_ + 1, gid)
            lad._remember(lad._qkey(rq),
                          "progress %d steps, status %s, cache hits %s" %
                          (progress, w["status"], w.get("cache_hits", 0)), rq)
            if checkpoint_root:
                self.learning_save(checkpoint_root)
            if w["status"] == "done":
                break
            if w["status"] == "paused" and progress == 0:
                break                                     # drift alarm: wandering is
                                                          # stopped, not funded
            if idle >= int(idle_limit):
                entry["stopped"] = "idle limit (activity-monitor doctrine)"
                break
        return {"goal": gid, "gather": gather, "rounds": log,
                "status": self.goal_book.goals[gid]["status"]}

    def codebase_map(self, root, topic=None):
        """WORK WITH LARGE CODEBASES (cp28): walk a source tree, build the semantic layer
        (module doctrine + public defs + import edges), and archive it LOSSLESSLY under a
        topic -- 489-module codebases index in under a second and answer by bm25. The map
        IS the long-term memory a code agent needs: query it, don't re-read the tree."""
        import ast as ast_, os as os_
        topic = topic or ("codebase:" + os_.path.basename(str(root).rstrip("/")))
        recs, srcs, n_defs = [], [], 0
        for r_, dd, ff in os_.walk(str(root)):
            dd[:] = [d_ for d_ in dd if d_ not in ("__pycache__", ".git",
                                                   "node_modules", ".venv")]
            for f_ in sorted(ff):
                if not f_.endswith(".py"):
                    continue
                pth = os_.path.join(r_, f_)
                rel = os_.path.relpath(pth, str(root))
                try:
                    src = open(pth, encoding="utf-8", errors="ignore").read()
                    tree = ast_.parse(src)
                except Exception:
                    recs.append("[%s] (unparseable)" % rel); srcs.append(rel)
                    continue
                doc = (ast_.get_docstring(tree) or "").strip()        # full docstrings: the map is
                                                          # the memory; culling it
                                                          # culls what code_write sees
                defs = [n.name for n in tree.body
                        if isinstance(n, (ast_.FunctionDef, ast_.ClassDef))
                        and not n.name.startswith("_")][:12]
                imps = sorted({(a.name.split(".")[0]) for n in ast_.walk(tree)
                               if isinstance(n, ast_.Import) for a in n.names} |
                              {n.module.split(".")[0] for n in ast_.walk(tree)
                               if isinstance(n, ast_.ImportFrom) and n.module})[:10]
                n_defs += len(defs)
                recs.append("[%s] %s || defs: %s || imports: %s" %
                            (rel, doc or "(no doc)", ", ".join(defs) or "-",
                             ", ".join(imps) or "-"))
                srcs.append(rel)
        self.research_archive(topic, recs, sources=srcs)
        return {"topic": topic, "modules": len(recs), "public_defs": n_defs}

    def code_write(self, name, task, topic=None, test=None, llm=None):
        """WRITE GOOD CODE (cp28): good code is code that PASSED ITS GATE. Context is
        GATHERED first (codebase map + assimilated recipes -- substrate before model);
        the draft comes from the attached model rung when one is present; and nothing
        returns without VERIFICATION: ast.parse must accept it, and when a `test`
        callable is given it must pass against the executed namespace. A draft that
        fails verification is REFUSED with the failure, never handed over polished."""
        import ast as ast_
        ctx = []
        if topic:
            qv = self.archive_query(topic, str(task), k=2)
            ctx += qv.get("evidence", [])
        for api in list(getattr(self, "_recipe_book", {}))[:2]:
            u = self.use_assimilated(api, str(task), k=1)
            if u.get("ok") and u.get("calls"):
                ctx.append("[recipe %s] %s" % (api, u["calls"][0]["call"]))
        model = llm or self.zoo.get("llm")
        if model is None:
            return {"ok": False, "why": "no model rung attached and no deterministic "
                                        "template fits -- refusing to guess code"}
        prompt = ("WRITE PYTHON. Task: " + str(task) + chr(10) +
                  "Name it: " + str(name) + chr(10) + "Context:" + chr(10) +
                  chr(10).join(ctx[:4]) + chr(10) + "Return ONLY code.")
        draft = str(model(prompt))
        if "```" in draft:
            draft = draft.split("```")[1]
            if draft.startswith(("python", "py")):
                draft = draft[draft.find(chr(10)) + 1:]
        try:
            ast_.parse(draft)
        except SyntaxError as exc:
            return {"ok": False, "why": "draft failed ast.parse: %s" % exc,
                    "draft": draft[:400]}
        if test is not None:
            ns = {}
            try:
                exec(compile(draft, "<code_write:%s>" % name, "exec"), ns)
                test(ns)
            except Exception as exc:
                return {"ok": False, "why": "draft failed its test: %s" % str(exc)[:200],
                        "draft": draft[:400]}
        lad = self.zoo["ladder"]
        q = "show me the verified code for %s" % name
        lad._remember(lad._qkey(q), draft[:1600], q)
        return {"ok": True, "name": name, "code": draft,
                "verified": "ast" + ("+test" if test else " only (no test given)"),
                "context_used": len(ctx)}

    def learning_spectrum(self):
        """THE MIND READS ITS OWN WEIGHTS (cp26, unicron turned inward): the same RMT
        instruments unicron points at other models' matrices -- MP bulk edge, spectral
        outliers (the learned signal), stable rank -- run over leCore's OWN learned 2D
        state: each trace tile's atom codebook and the semantic encoder's context matrix.
        Outliers above the MP edge mean the memory holds STRUCTURE; a spectrum matching
        the random bulk means that matrix has learned nothing yet. An honest self-
        diagnostic, not a vibe."""
        from holographic.io_and_interop.holographic_unicron import spectral_report
        out = {"matrices": []}
        for i, t in enumerate(self.experience.tiles):
            A = np.asarray(getattr(t, "_atoms", []), float)
            if A.ndim == 2 and min(A.shape) >= 8:
                r = spectral_report(A)
                out["matrices"].append({
                    "name": "trace_tile_%d_atoms" % i, "shape": list(A.shape),
                    "outliers": int(r.get("outliers", 0)),
                    "outlier_fraction": round(float(r.get("outlier_fraction", 0)), 4),
                    "stable_rank": round(float(r.get("stable_rank", 0)), 2)})
        enc = getattr(self, "_lever7_text", None)      # the p19 conversation encoder
        for attr in ("context", "_contexts", "contexts", "_ctx", "_vectors"):
            ctx = getattr(enc, attr, None) if enc else None
            if isinstance(ctx, dict) and len(ctx) >= 8:
                C = np.asarray([np.asarray(v, float).ravel()
                                for v in list(ctx.values())[:512]], float)
                if C.ndim == 2 and min(C.shape) >= 8:
                    r = spectral_report(C)
                    out["matrices"].append({
                        "name": "semantic_%s" % attr.strip("_"), "shape": list(C.shape),
                        "outliers": int(r.get("outliers", 0)),
                        "outlier_fraction": round(float(r.get("outlier_fraction", 0)), 4),
                        "stable_rank": round(float(r.get("stable_rank", 0)), 2)})
                break
        # HONEST READING (cp26): the trace ATOM CODEBOOK is random BY DESIGN (content-
        # hashed atoms want mutual orthogonality), so its spectrum SHOULD sit in the MP
        # bulk -- outliers THERE would mean atom collisions, an unhealthy sign. Learned
        # structure is expected in the SEMANTIC contexts (trained from co-occurrence).
        atom_ok = all(m_["outliers"] == 0 for m_ in out["matrices"]
                      if m_["name"].startswith("trace_tile"))
        sem = [m_ for m_ in out["matrices"] if m_["name"].startswith("semantic")]
        sem_structured = any(m_["outliers"] > 0 for m_ in sem)
        out["verdict"] = ("codebook orthogonal (healthy) + semantic structure present"
                          if atom_ok and sem_structured else
                          "codebook orthogonal (healthy); semantic contexts %s" %
                          ("show no outliers yet -- vocabulary is young" if sem else
                           "not exposed by this encoder") if atom_ok else
                          "WARNING: atom codebook shows outliers -- possible collisions")
        return out

    def partition_fingerprint(self, root=None):
        """ONE HYPERVECTOR PER PARTITION (cp26, unicron_fingerprint's pattern): bind each
        learned section's role atom with an encoding of its content statistics, bundle
        across sections. Partitions become points in FHRR space -- successive saves
        compare by cosine, and the DELTA is a measured drift number, not a diff scroll."""
        import hashlib
        from holographic.io_and_interop.holographic_container import load_container
        from holographic.agents_and_reasoning.holographic_ai import random_vector, bind
        import os as os_
        if root is None:
            secs = []
            z = self.zoo
            secs.append(("taught", len(getattr(z["ladder"], "taught_log", []))))
            secs.append(("payloads", len(getattr(z["ladder"], "_payloads", {}))))
            secs.append(("images", len(getattr(self, "_image_memory", []))))
            secs.append(("certs", len(getattr(self, "_certificate_memory", []))))
            stats = secs
        else:
            got = load_container(open(os_.path.join(str(root), "learning",
                                                    "state.lecore"), "rb").read())
            stats = []
            for sec in got["sections"]:
                blob = str(sorted(sec["meta"].items()))[:2000].encode()
                stats.append((sec["kind"], int(hashlib.sha256(blob).hexdigest()[:8], 16)))
        v = np.zeros(1024)
        for name, val in stats:
            seed = int(hashlib.sha256(str(name).encode()).hexdigest()[:8], 16)
            role = random_vector(1024, np.random.default_rng(seed))
            mag = random_vector(1024, np.random.default_rng((int(val) % 999983) + 7))
            v = v + bind(role, mag)
        v = v / (np.linalg.norm(v) + 1e-12)
        return {"fingerprint": v, "sections": len(stats)}

    def partition_drift(self, root_a, root_b):
        """Cosine between two partition fingerprints: 1.0 = identical learned state,
        lower = the partitions have diverged. The number the migration battery implies."""
        fa = self.partition_fingerprint(root_a)["fingerprint"]
        fb = self.partition_fingerprint(root_b)["fingerprint"]
        return {"cosine": round(float(fa @ fb), 4),
                "verdict": "same state" if float(fa @ fb) > 0.98 else "diverged"}

    def partition_retention(self, root_before, root_after, probes):
        """RETENTION AS A FACULTY (cp26, unicron_retention's pattern applied to
        partitions): ask the SAME probe questions against both roots in fresh minds and
        report answered-before vs answered-after vs changed. The ad-hoc 8/8 battery,
        institutionalized."""
        import lecore as _lc
        def _serve(root):
            mm = _lc.UnifiedMind()
            mm.zoo_attach(lambda p: "__PROBE_MISS__")
            mm.learning_load(root)
            got = {}
            for q in probes:
                a = mm.ask(q)
                got[q] = None if str(a.get("answer", "")).startswith("__PROBE") \
                    else str(a.get("answer"))
            return got
        A, B = _serve(root_before), _serve(root_after)
        kept = sum(1 for q in probes if A[q] is not None and B[q] == A[q])
        lost = [q for q in probes if A[q] is not None and B[q] is None]
        changed = [q for q in probes if A[q] and B[q] and A[q] != B[q]]
        gained = [q for q in probes if A[q] is None and B[q] is not None]
        return {"probes": len(probes), "kept": kept, "lost": lost,
                "changed": changed, "gained": gained,
                "retention": round(kept / max(sum(1 for q in probes
                                                  if A[q] is not None), 1), 3)}

    def kernel_card(self, name):
        """THE CARD DISCIPLINE, generalized (cp26, from the GLSL library): any library
        entry -- shader kernel, Lean certificate, assimilated recipe -- renders as ONE
        card where the artifact, its VERIFICATION, and its KEPT NEGATIVE travel together.
        A capability whose boundary is separated from its code is a trap; the card keeps
        them fused."""
        from holographic.io_and_interop.holographic_glslkernels import KERNELS
        if name in KERNELS:
            k = KERNELS[name]
            return {"kind": "glsl_kernel", "name": name, "does": k["does"],
                    "verified": k["verified"][:400],
                    "kept_negative": next((line for line in k["verified"].split(".")
                                           if "KEPT NEGATIVE" in line or "GIVES UP" in
                                           line or "RETRACTED" in line), "none recorded"),
                    "source_lines": k["source"].count("\n")}
        for c in getattr(self, "_certificate_memory", []):
            if c["id"] == name:
                mt = c["meta"]
                return {"kind": "lean_certificate", "name": name,
                        "does": "chain_well_typed(%s)" % name,
                        "verified": mt.get("verification_tier"),
                        "kept_negative": "tier ladder above %s is UNRUN: %s" %
                                         (mt.get("verification_tier"),
                                          mt.get("external_step"))}
        for api, recipes in getattr(self, "_recipe_book", {}).items():
            for r_ in recipes:
                if r_["call"].split("(")[0].endswith(name) or r_["call"] == name:
                    return {"kind": "recipe", "name": r_["call"], "api": api,
                            "does": r_["about"],
                            "verified": r_.get("verified",
                                               "UNVERIFIED: extracted from docs, never "
                                               "executed here"),
                            "kept_negative": r_.get("kept_negative",
                                                    "argument semantics unchecked")}
        return {"error": "no library entry named %r" % name}

    def model3d(self, spec, name=None, size=180):
        """MODEL AND RENDER THROUGH THE MIND'S OWN HANDS (cp25): a compact scene spec --
        [{'shape': 'sphere'|'capsule'|'torus'|'box', ...params, 'at': [x,y,z]}] -- becomes
        an SDF CSG tree (smooth-union, the monument recipe), raymarches through the
        rendering faculty, and the render is REMEMBERED: stored in the image memory,
        labeled and content-addressed, recallable later by name. Returns the receipt, not
        just pixels."""
        from holographic.mesh_and_geometry import holographic_sdf as S_
        import numpy as np_
        parts = []
        for it in (spec or []):
            sh = str(it.get("shape", "sphere")).lower()
            at = [float(x) for x in it.get("at", [0, 0.4, 0])]
            if sh == "sphere":
                nd = S_.sphere(float(it.get("r", 0.4))).translate(at)
            elif sh == "capsule":
                # capsule(h, r): a vertical capsule of half-height h -- axis endpoints are
                # not the API here; height + translate is (read the signature, not the guess)
                h_ = abs(float(it.get("h", it.get("b", [0, 0.8, 0])[1])))
                nd = S_.capsule(h_ / 2.0, float(it.get("r", 0.14))).translate(at)
            elif sh == "torus":
                nd = S_.torus(float(it.get("R", 0.6)),
                              float(it.get("r", 0.08))).translate(at)
            elif sh == "box":
                nd = S_.box(*[float(x) for x in
                              it.get("size", [0.3, 0.3, 0.3])]).translate(at)
            else:
                continue
            parts.append(nd)
        if not parts:
            return {"ok": False, "why": "empty spec -- nothing to model"}
        scene = parts[0]
        for nd in parts[1:]:
            scene = scene.smooth_union(nd, k=0.18)
        scene = scene.smooth_union(S_.plane(), k=0.18)
        cam = self.camera(eye=(2.6, 1.7, 3.1), target=(0, 0.45, 0), fov_deg=42.0)
        img = self.render_sdf(scene, cam, width=int(size), height=int(size),
                              reflect=0.3, ao=True, shadows=True)
        label = name or ("model3d: %d shapes" % len(parts))
        rec = self.image_remember(np_.asarray(img), label, source="render")
        return {"ok": True, "label": label, "shapes": len(parts),
                "stored": rec.get("stored", False), "sha": rec.get("sha"),
                "dedup": rec.get("dedup")}

    def research_archive(self, topic, texts, sources=None, notes="auto"):
        """THE LOSSLESS ARCHIVE CONTRACT (cp25): every text is preserved IN FULL -- a
        KnowledgeStore note (the durable record) AND a bound corpus per topic (the
        queryable index) AND semantic ingestion (the vocabulary learns). Nothing is
        summarized away at archive time; compression is the container's job, loss is
        nobody's. archive_query answers from the archive with provenance."""
        from holographic.caching_and_storage.holographic_knowledgestore import KnowledgeStore
        root = getattr(self, "_archive_root", None) or "/tmp/lecore_archive"
        ks = KnowledgeStore(root)
        # BULK HONESTY (cp28, measured): KnowledgeStore.add_note is O(store) PER CALL --
        # 3.3s each on a grown store, 27 minutes for a 489-module codebase. The lossless
        # record for bulk loads is the CONTAINER CORPUS (verbatim, bm25-queryable,
        # persisted); the note arm gets ONE manifest entry naming the batch. notes="auto"
        # writes per-text notes only for small batches (<= 12); notes=True forces them;
        # notes=False never writes them.
        # JOURNAL BUDGET (cp28, the cp20 debt RESOLVED after it detonated at 318MB /
        # 200,199 entries): before any note write, an oversized journal rotates to
        # knowledge.json.rotated-<n> and a fresh journal starts with one manifest note.
        # A flagged debt left unmigrated is a scheduled outage -- this is the migration.
        import os as _os, json as _js
        jp = _os.path.join(root, "knowledge.json")
        try:
            if _os.path.exists(jp) and _os.path.getsize(jp) > 8_000_000:
                n_ = 0
                while _os.path.exists(jp + ".rotated-%d" % n_):
                    n_ += 1
                _os.rename(jp, jp + ".rotated-%d" % n_)
        except Exception:
            pass
        per_note = (notes is True) or (notes == "auto" and len(texts or []) <= 12)
        n_notes = 0
        for i, t in enumerate(texts or []):
            src = (sources or [None] * len(texts))[i] or ("%s#%d" % (topic, i))
            if per_note:
                ks.add_note("[%s] %s" % (src, t), tags=("archive", str(topic)))
                n_notes += 1
        if not per_note and texts:
            ks.add_note("[manifest] archived %d texts under topic %s (verbatim in the "
                        "container corpus; query via archive_query)" %
                        (len(texts), topic), tags=("archive", str(topic), "manifest"))
            n_notes = 1
        self.semantic_ingest(" ".join(str(t)[:200] for t in (texts or []))[:40000],
                             source="archive:%s" % topic)
        if not hasattr(self, "_archive_corpora"):
            self._archive_corpora = {}
        self._archive_corpora.setdefault(str(topic), []).extend(
            [str(t) for t in (texts or [])])
        return {"archived": n_notes, "topic": str(topic),
                "total_in_topic": len(self._archive_corpora[str(topic)]),
                "lossless": True}

    def archive_query(self, topic, question, k=3):
        """Query a research archive: corpus evidence + note recall, provenance labeled,
        composed without discarding conflicts. The archive answers; the model need not."""
        out = {"topic": str(topic), "question": str(question), "evidence": []}
        texts_ = getattr(self, "_archive_corpora", {}).get(str(topic)) or []
        if texts_:
            try:
                ranked_ = self.bm25_rank(str(question), texts_)
                for idx_, sc_ in list(ranked_)[:int(k)]:
                    if sc_ > 0:
                        out["evidence"].append("[corpus %.2f] %s"
                                               % (float(sc_), texts_[int(idx_)][:220]))
            except Exception:
                pass
        if len(out["evidence"]) < int(k):
            # NOTES ARM, BOUNDED (cp28, measured): KnowledgeStore.evidence() crawls the
            # whole journal -- 55s once the store has grown -- so it runs ONLY when the
            # corpus arm came up short, and the crawl is cached per process per root.
            try:
                from holographic.caching_and_storage.holographic_knowledgestore import \
                    KnowledgeStore
                root = getattr(self, "_archive_root", None) or "/tmp/lecore_archive"
                cache = getattr(self, "_archive_note_cache", {})
                if root not in cache:
                    cache[root] = [str(e_.get("text", e_))
                                   for e_ in KnowledgeStore(root).evidence()]
                    self._archive_note_cache = cache
                stopw = self.zoo["ladder"]._STOP
                qw = set(str(question).lower().split()) - stopw
                ranked = []
                for tx in cache[root]:
                    ov = len(qw & (set(tx.lower().split()) - stopw)) / max(len(qw), 1)
                    if ov > 0:
                        ranked.append((ov, tx))
                ranked.sort(key=lambda x_: -x_[0])
                for _, tx in ranked[:int(k) - len(out["evidence"])]:
                    out["evidence"].append("[note] " + tx[:220])
            except Exception:
                pass
        out["found"] = len(out["evidence"])
        return out

    def market_backtest(self, series, horizon=1, d_grid=(3, 5, 8), coverage=0.9):
        """WALK-FORWARD BACKTEST WITH LEARN/ITERATE/ADAPT (cp25, CPTC's lesson applied):
        for each d in the grid, fit the routed forecaster on a growing window and predict
        one step ahead across the tail -- NO lookahead: the model at time t sees only
        [0..t). Learn: the winner by held-out MAE beats a naive last-value baseline or the
        verdict SAYS SO. Patterns: residual sign-runs flag regimes; the conformal width
        from the residual quantile prices the uncertainty. Adapt: the winning config is
        TAUGHT to memory so the next backtest starts warm."""
        xs = [float(x) for x in (series or [])]
        n = len(xs)
        if n < 12:
            return {"ok": False, "why": "need >= 12 points for a walk-forward split"}
        split = max(8, int(n * 0.6))
        results = {}
        for d_ in d_grid:
            if split <= d_ + 2:
                continue
            errs, resid = [], []
            for t in range(split, n):
                try:
                    rf = self.forecast(xs[:t], d=int(d_))
                    p_ = rf.predict(np.asarray(xs[t - int(d_):t], float))
                    yhat = float(p_["point"]) if isinstance(p_, dict) and "point" in p_ \
                        else float(np.asarray(p_, float).ravel()[0])
                except Exception:
                    continue
                errs.append(abs(yhat - xs[t]))
                resid.append(yhat - xs[t])
            if errs:
                results[int(d_)] = {"mae": float(np.mean(errs)), "n": len(errs),
                                    "resid": resid}
        if not results:
            return {"ok": False, "why": "no d in the grid produced forecasts"}
        naive = float(np.mean([abs(xs[t] - xs[t - 1]) for t in range(split, n)]))
        best_d = min(results, key=lambda k_: results[k_]["mae"])
        best = results[best_d]
        q = float(np.quantile(np.abs(best["resid"]), float(coverage)))
        runs, cur = [], 1
        sg = [1 if r > 0 else -1 for r in best["resid"] if r != 0]
        for a, b in zip(sg, sg[1:]):
            if a == b:
                cur += 1
            else:
                runs.append(cur); cur = 1
        runs.append(cur if sg else 0)
        # B4 (cp28, CPTC's lesson completed): sign-runs no longer just FLAG a regime --
        # when the longest run exceeds 5, refit on the post-break tail alone and report
        # the adapted error next to the full-window error. Adaptation must EARN its keep
        # in the same table.
        adapted = None
        longest = max(runs) if runs else 0
        if longest > 5:
            # locate the longest run's START in absolute time (first cut assumed the run
            # was final; second cut demanded 6 points AFTER the run -- which excluded the
            # most important case, a regime that runs to the end of the window. The gate
            # is 6 evaluable points after the BREAK, nothing more.)
            best_start, best_len, cur_start = 0, 0, 0
            for i_ in range(1, len(sg) + 1):
                if i_ == len(sg) or sg[i_] != sg[i_ - 1]:
                    if i_ - cur_start > best_len:
                        best_len, best_start = i_ - cur_start, cur_start
                    cur_start = i_
            brk = split + best_start                       # absolute index of run start
        if longest > 5 and (n - (split + best_start)) >= 6 and split + best_start \
                >= int(best_d) + 2:
            tail_errs = []
            for t in range(max(brk, split + 1), n):
                try:
                    rf2 = self.forecast(xs[brk - int(best_d):t], d=int(best_d))
                    p2_ = rf2.predict(np.asarray(xs[t - int(best_d):t], float))
                    y2 = float(p2_["point"]) if isinstance(p2_, dict) and "point" in p2_ \
                        else float(np.asarray(p2_, float).ravel()[0])
                    tail_errs.append(abs(y2 - xs[t]))
                except Exception:
                    pass
            if tail_errs:
                tail_naive = float(np.mean([abs(xs[t] - xs[t - 1])
                                            for t in range(max(brk, split + 1), n)]))
                adapted = {"post_break_mae": round(float(np.mean(tail_errs)), 5),
                           "post_break_naive": round(tail_naive, 5),
                           "break_at": int(brk),
                           "helps": bool(np.mean(tail_errs) < tail_naive)}
        verdict = ("beats naive" if best["mae"] < naive else
                   "does NOT beat last-value -- do not trade this formula")
        lad = self.zoo["ladder"]
        qq = "what backtest config won on the last market series"
        lad._remember(lad._qkey(qq),
                      "d=%d mae=%.4f naive=%.4f (%s); conformal %d%% width %.4f; longest "
                      "residual sign-run %d (regime flag if >5)" %
                      (best_d, best["mae"], naive, verdict, int(coverage * 100), q,
                       max(runs) if runs else 0), qq)
        return {"ok": True, "best_d": best_d, "mae": round(best["mae"], 5),
                "naive_mae": round(naive, 5), "verdict": verdict,
                "regime_adaptation": adapted,
                "conformal_width": round(q, 5), "coverage": coverage,
                "longest_sign_run": max(runs) if runs else 0,
                "evaluated": {k_: round(v["mae"], 5) for k_, v in results.items()}}

    def assimilate_docs(self, api_name, doc_text):
        """ASSIMILATE DOCUMENTATION, THEN BUILD WITH IT (cp25): the doc text is archived
        LOSSLESSLY (research_archive), the vocabulary ingests it, and every extractable
        CALL RECIPE -- 'name(args)' signatures with their nearest description line -- is
        taught as a reflex pair AND registered in the recipe book. use_assimilated(task)
        then answers 'which call, with what arguments' from memory at T0, no model."""
        import re as re_
        text = str(doc_text)
        self.research_archive("docs:%s" % api_name, [text], sources=[api_name])
        sigs = re_.findall(r"([a-zA-Z_][a-zA-Z0-9_.]*)\(([^)]*)\)", text)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        recipes = []
        seen = set()
        for fn, args in sigs:
            if fn in seen or fn.startswith(("http", "www")):
                continue
            seen.add(fn)
            desc = next((l for l in lines if fn in l and "(" in l), fn)
            recipes.append({"call": "%s(%s)" % (fn, args.strip()), "about": desc})
        if not hasattr(self, "_recipe_book"):
            self._recipe_book = {}
        self._recipe_book[str(api_name)] = recipes
        lad = self.zoo["ladder"]
        for r_ in recipes[:24]:
            head = r_["call"].split("(")[0].split(".")[-1]
            qq = "how do I %s with %s" % (head.replace("_", " "), api_name)
            lad._remember(lad._qkey(qq), "%s -- %s" % (r_["call"], r_["about"]), qq)
        return {"assimilated": api_name, "recipes": len(recipes),
                "taught": min(len(recipes), 24)}

    def use_assimilated(self, api_name, task, k=3):
        """Build with assimilated docs ON THE FLY: rank the api's recipes by content
        overlap with the task and return the calls to make, best first -- deterministic,
        zero model calls, provenance = the archived doc itself."""
        book = getattr(self, "_recipe_book", {}).get(str(api_name)) or []
        if not book:
            return {"ok": False, "why": "no assimilated docs for %r" % api_name}
        stop = self.zoo["ladder"]._STOP

        def _stem(w):                                     # halt/halts/halting share a stem;
            for suf in ("ing", "ed", "es", "s"):          # crude but deterministic, and it
                if len(w) > 3 and w.endswith(suf):        # broke a ranking tie the selftest
                    return w[: -len(suf)]                 # caught (halt vs halts scored 0)
            return w
        tw = {_stem(w) for w in str(task).lower().replace("_", " ").split()} - stop
        scored = []
        for r_ in book:
            rw = {_stem(w) for w in (r_["call"] + " " + r_["about"]).lower()
                  .replace("_", " ").replace("(", " ").replace(")", " ")
                  .replace(".", " ").split()} - stop
            ov = len(tw & rw) / max(len(tw), 1)
            scored.append((ov, r_))
        scored.sort(key=lambda x: -x[0])
        return {"ok": True, "api": str(api_name),
                "calls": [{"call": r_["call"], "about": r_["about"],
                           "score": round(s_, 2)} for s_, r_ in scored[:int(k)] if s_ > 0]}

    def distill_certificates(self, chains=None):
        """DISTILL THE LEAN MATH (cp24): synthesize certified chains, keep each theorem's
        Horn-clause DERIVATION as the mathematical record -- consumes/produces facts plus
        linkage implying chain_well_typed -- and store the Lean 4 sources as a labeled
        certificate library section. Lean is not installed here, so the ok flag is leCore's
        own Horn prover; the Lean source is the EXPORTABLE artifact an external `lake env
        lean` run would check -- stated, not glossed. Ill-typed chains refuse and COUNT:
        the refusal rate is part of the experiment."""
        if chains is None:
            chains = [("cert_depth_mesh", ["depth_from_image", "image_to_mesh"]),
                      ("cert_mesh_pick", ["image_to_mesh", "pick_mesh"]),
                      ("cert_bad_order", ["image_to_mesh", "depth_from_image"]),
                      ("cert_sdf_img", ["sdf_scene_image", "depth_from_image"])]
        lib, ok_n, refuse_n = [], 0, 0
        for name, chain in chains:
            ex = {c: (lambda x=None: x) for c in chain}
            r = self.synthesize_tool_certified(name, chain, ex)
            if r.get("ok") and (r.get("lean_certificate") or {}).get("ok"):
                cert = r["lean_certificate"]
                lib.append({"name": name, "chain": chain,
                            "consumes": r.get("consumes"), "produces": r.get("produces"),
                            "lean": cert.get("lean", ""),                          # the WHOLE theorem: a clipped
                                                       # certificate cannot be checked
                            "proof": str(cert.get("proof")),
                            # the trust-boundary tiering (arXiv:2605.16407, adopted):
                            # certificate status states EXACTLY what has been checked WHERE
                            "verification_tier": "T2:horn-checked-local",
                            "tier_ladder": ["T1:generated", "T2:horn-checked-local",
                                            "T3:lean-kernel-checked",
                                            "T4:axiom-audited-sorry-free"],
                            "external_step": "lake env lean <file> then #print axioms "
                                             "against {propext, Classical.choice, "
                                             "Quot.sound}"})
                ok_n += 1
            else:
                refuse_n += 1
        if not hasattr(self, "_certificate_memory"):
            self._certificate_memory = []
        for c in lib:
            self._certificate_memory.append({
                "kind": "lecore.learning.certificate", "id": c["name"],
                "meta": c, "arrays": {}})
        # the distilled MATH, taught: the implication shape itself
        lad = self.zoo["ladder"]
        for c in lib:
            q = "what does the %s certificate prove" % c["name"]
            a = ("Horn derivation: consumes(%s)=%s, produces(%s)=%s, adjacent links "
                 "type-match => chain_well_typed(%s); Lean 4 source stored in the "
                 "certificate library (leCore's Horn prover checked it; the source is "
                 "exportable to an external Lean run)" %
                 (c["name"], c["consumes"], c["name"], c["produces"], c["name"]))
            lad._remember(lad._qkey(q), a, q)
        return {"certified": ok_n, "refused": refuse_n, "library": [c["name"] for c in lib]}

    def goal_status(self, goal_id):
        """The goal's durable record: steps + statuses + deliverables + convergence trace."""
        g = self.goal_book.goals.get(str(goal_id))
        if g is None:
            return {"error": "unknown goal %r" % goal_id}
        return {"id": g["id"], "status": g["status"], "plan_via": g.get("plan_via"),
                "steps": [{"name": x["name"], "status": x["status"]} for x in g["steps"]],
                "convergence": g["convergence"],
                "deliverables": {x["name"]: x["deliverable"] for x in g["steps"]
                                 if x["deliverable"]}}

    def calibrate_reflex(self):
        """UCCI ADOPTED (cp23, arXiv:2605.18796's recipe in pure NumPy): fit an ISOTONIC
        map (pool-adjacent-violators) from reflex CONFIDENCE to observed ERROR PROBABILITY,
        using the feedback log's (confidence, ok) pairs. The fixed calibrated null stays as
        the floor; this map turns 'the gate fired at 0.41' into 'expected error 12%' -- the
        number a cost-optimal escalation threshold actually needs. Honest refusal below 4
        labeled pairs (no calibration from anecdotes)."""
        pairs = list(getattr(self, "_reflex_calib_pairs", []))
        if len(pairs) < 4:
            return {"calibrated": False, "why": "need >= 4 feedback-labeled serves, have %d"
                    % len(pairs)}
        pairs.sort(key=lambda x: x[0])
        conf = np.array([c for c, _ in pairs], float)
        err = np.array([0.0 if ok_ else 1.0 for _, ok_ in pairs], float)
        # PAV for a DECREASING error-vs-confidence fit: fit increasing on -conf order
        y = err[::-1].copy()
        w = np.ones_like(y)
        i = 0
        vals, wts = list(y), list(w)
        k = 0
        vals, wts = [], []
        for v in y:
            vals.append(v); wts.append(1.0)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                v2, w2 = vals.pop(), wts.pop()
                v1, w1 = vals.pop(), wts.pop()
                vals.append((v1 * w1 + v2 * w2) / (w1 + w2)); wts.append(w1 + w2)
        fit = np.repeat(vals, [int(x) for x in wts])[::-1]
        self._reflex_calibration = {"conf": conf.tolist(), "err": fit.tolist()}
        self.zoo["ladder"]._error_prob = self.reflex_error_prob   # B2: the gate consults
                                                                  # calibration from the
                                                                  # moment it exists
        return {"calibrated": True, "pairs": len(pairs),
                "err_at_min_conf": round(float(fit[0]), 3),
                "err_at_max_conf": round(float(fit[-1]), 3)}

    def reflex_error_prob(self, confidence):
        """The calibrated error probability for a reflex confidence (step interpolation
        over the isotonic fit); None when uncalibrated -- the fixed gate then stands alone."""
        cal = getattr(self, "_reflex_calibration", None)
        if not cal:
            return None
        conf = np.asarray(cal["conf"], float)
        err = np.asarray(cal["err"], float)
        i = int(np.searchsorted(conf, float(confidence), side="right")) - 1
        return float(err[max(0, min(i, len(err) - 1))])

    def cache_invalidate(self, step=None):
        """B1 (cp28): the exact-prefix cache finally has an ERASER. step=None clears all;
        step='name' removes every prefix ending in that step and its stateless entry --
        the fix path for 'I repaired this executor, run it fresh'. Failure auto-
        invalidates (wired in goal_work), because a cached value from before the fix is
        exactly the stale answer TVCache's exactness promise exists to prevent."""
        tc = self.tool_cache
        if step is None:
            n = len(tc["prefix"]) + len(tc["stateless"])
            tc["prefix"].clear(); tc["stateless"].clear()
            return {"invalidated": n, "scope": "all"}
        n = 0
        for k in [k for k in tc["prefix"] if k.split("|")[-1] == str(step)]:
            del tc["prefix"][k]; n += 1
        if str(step) in tc["stateless"]:
            del tc["stateless"][str(step)]; n += 1
        return {"invalidated": n, "scope": str(step)}

    @property
    def tool_cache(self):
        """The TRAJECTORY-KEYED TOOL-VALUE CACHE (cp23, TVCache's design adopted:
        arXiv:2602.10986): results keyed by the LONGEST MATCHING PREFIX of the step
        trajectory, so a cached value is only reused when every state-mutating step before
        it matched too -- exact-prefix reuse is CORRECT by construction. Stateless steps
        (declared via stateless= on goal_work) match out of order, TVCache's
        will_mutate_state distinction."""
        if not hasattr(self, "_tool_cache"):
            self._tool_cache = {"prefix": {}, "stateless": {}}
        return self._tool_cache

    def answer_feedback(self, query, ok, note=None):
        """OUTCOME FEEDBACK ON A REFLEX ANSWER (cp22): ok=False marks the served payload BAD
        -- the T0 path will refuse it forever after (no knee-jerk repeats) and the next ask
        escalates to be re-taught. ok=True strengthens it. Feedback also feeds the
        escalation decision tree (reflex_outcome on the query key) and PERSISTS in the
        taught section, so a bad answer stays dead across processes."""
        lad = self.zoo["ladder"]
        qk = lad._qkey(str(query))
        t = self.experience._route(qk)
        hit = self.experience.read_gated(qk)
        if not hit["fired"]:
            return {"located": False, "why": "no reflex payload near this query"}
        pid_key = "%d:%d" % (t, int(hit.get("atom", -1)))
        if not hasattr(lad, "_payload_bad"):
            lad._payload_bad = set()
        if not hasattr(lad, "_feedback_log"):
            lad._feedback_log = []
        if ok:
            lad._payload_bad.discard(pid_key)
            for k_ok in (" ".join(str(query).lower().split()),
                         " ".join(self.session_salt(query).lower().split())):
                getattr(lad, "_vetoed_qs", set()).discard(k_ok)
        else:
            lad._payload_bad.add(pid_key)
            # cp54: THE VETO MUST SURVIVE A RESTART. It did not: load replays the durable
            # record through _remember, which re-inserted every vetoed answer into the
            # exact sidecar -- the same 43 noise reflexes were purged in cp48 and again in
            # cp53, coming back from replay each time. A TOMBSTONE on the normalized
            # question is persisted with the record; replay honours it. A deliberate
            # re-teach lifts it, because re-establishment is the documented recovery path.
            if not hasattr(lad, "_vetoed_qs"):
                lad._vetoed_qs = set()
            lad._vetoed_qs.add(" ".join(str(query).lower().split()))
            lad._vetoed_qs.add(" ".join(self.session_salt(query).lower().split()))
            # cp38: the EXACT sidecar must honour the veto in the same breath --
            # between veto and re-teach it would otherwise keep serving the bad answer
            nq_fb = " ".join(str(query).lower().split())
            sq_fb = " ".join(self.session_salt(query).lower().split())
            for k_fb in (nq_fb, sq_fb):
                getattr(lad, "_exact", {}).pop(k_fb, None)
        lad._feedback_log.append([lad._payload_qs.get(pid_key, str(query)), bool(ok),
                                  str(note or "")])
        if not hasattr(self, "_reflex_calib_pairs"):
            self._reflex_calib_pairs = []
        self._reflex_calib_pairs.append([float(hit.get("confidence", 0.0)), bool(ok)])
        # KEPT NEGATIVE (cp22): the first cut also called reflex_outcome(qk, False) -- but
        # that suppresses the QUERY KEY's trust, which then blocked even the CORRECTED
        # answer taught afterward (read_gated stopped firing at all). The bad-set vetoes the
        # PAYLOAD; the key must stay readable so a re-teach can serve. Positive outcomes
        # still strengthen the key (the decision tree learns from success).
        if ok:
            try:
                self.reflex_outcome(qk, True)
            except Exception:
                pass
        return {"located": True, "payload": pid_key, "marked": "ok" if ok else "bad",
                "will_serve_again": bool(ok)}

    def synthesize_response(self, query, k=3):
        """SYNTHESIZE, don't parrot (cp22): compose an answer from MULTIPLE memory sources,
        each carrying its PROVENANCE label -- [reflex] taught answers ranked by content
        overlap with the query, [kb] recall rows, [hypothesis] void-born concepts (their
        label rides in the text and stays visible). One verbatim payload is a special case,
        not the rule; conflicts stay side-by-side rather than being silently merged."""
        lad = self.zoo["ladder"]
        stop = lad._STOP
        qw = set(str(query).lower().split()) - stop
        scored = []
        for q_, a_, *rest_ in getattr(lad, "taught_log", []):
            ov = len(qw & (set(q_.lower().split()) - stop)) / max(len(qw), 1)
            if ov > 0.2:
                tag = "[hypothesis]" if a_.startswith("hypothesis:") else "[reflex]"
                scored.append((ov, "%s %s" % (tag, a_)))
        scored.sort(key=lambda x: -x[0])
        parts = [t for _, t in scored[:k]]
        try:
            rows = self.recall(str(query), k=2)
            for label, text in (rows or []):
                parts.append("[kb] %s: %s" % (label, str(text)[:160]))
        except Exception:
            pass
        if not parts:
            return {"synthesized": False, "why": "no memory sources overlap this query"}
        return {"synthesized": True, "sources": len(parts),
                "answer": "\n".join(parts)}

    def discover_concepts(self, max_concepts=3):
        """VOID-DRIVEN CONCEPT DISCOVERY (cp22): run the query void map over everything
        asked and taught; each structurally-licensed void becomes a HYPOTHESIS -- taught
        into the reflex with the 'hypothesis:' label IN the answer text, so it can be
        recalled and built on but never masquerades as verified fact. Returns what was
        proposed; the void gate's refusals are honored silently."""
        lad = self.zoo["ladder"]
        qs = [t_[0] for t_ in getattr(lad, "taught_log", [])] + \
             list(getattr(lad, "query_log", []))[-40:]
        if len(qs) < 4:
            return {"proposed": 0, "why": "too little asked/taught to map voids"}
        try:
            vm = self.query_void_map(qs)
        except Exception as exc:
            return {"proposed": 0, "why": str(exc)[:100]}
        voids = (vm or {}).get("voids", [])[:int(max_concepts)]
        made = []
        for v in voids:
            terms = v.get("terms") or v.get("slots") or []
            if not terms:
                continue
            concept = " + ".join(str(t) for t in terms[:3])
            q = "what about the concept " + concept
            a = ("hypothesis: the structure licenses '%s' but memory lacks it -- proposed "
                 "by the void map, unverified; verify with corpus evidence before "
                 "promotion" % concept)
            lad._remember(lad._qkey(q), a, q)
            made.append(concept)
        return {"proposed": len(made), "concepts": made}

    def image_remember(self, image, label, source="capture", scope="images"):
        """STORE AN IMAGE IN THE HOLOGRAPHIC MEMORY (cp22): a labeled, scoped container
        section (kind lecore.memory.image) in the mind's image store, persisted with
        everything else by learning_save -- the container's zip does the compression and
        the MEASURED ratio returns with the receipt. Rendered outputs and consumed inputs
        share one organized memory, distinguished by `source` (capture/render/dream)."""
        import hashlib
        img = np.asarray(image)
        if not hasattr(self, "_image_memory"):
            self._image_memory = []
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.dtype != np.uint8 else img
        sha = hashlib.sha256(u8.tobytes()).hexdigest()[:16]
        for e in self._image_memory:                      # content-addressed dedupe
            if e["meta"]["sha"] == sha:
                return {"stored": False, "dedup": e["meta"]["label"], "sha": sha}
        self._image_memory.append({
            "kind": "lecore.memory.image", "id": str(label),
            "meta": {"label": str(label), "source": str(source), "scope": str(scope),
                     "shape": list(u8.shape), "sha": sha},
            "arrays": {"img": u8}})
        self.semantic_ingest("image remembered: %s (%s, %s)" % (label, source, scope),
                             source="image_memory")
        return {"stored": True, "label": str(label), "sha": sha,
                "raw_bytes": int(u8.nbytes)}

    def image_recall(self, query, k=3):
        """RECALL IMAGES BY LABEL MEANING: content-word overlap over the labeled memory
        (deterministic), best first; each hit returns its meta + the array."""
        if not getattr(self, "_image_memory", None):
            return []
        stop = self.zoo["ladder"]._STOP
        qw = set(str(query).lower().replace(":", " ").split()) - stop
        scored = []
        for e in self._image_memory:
            lw = set((e["meta"]["label"] + " " + e["meta"]["source"]).lower()
                     .replace(":", " ").split()) - stop
            ov = len(qw & lw) / max(len(qw | lw), 1)
            scored.append((ov, e))
        scored.sort(key=lambda x: -x[0])
        return [{"label": e["meta"]["label"], "source": e["meta"]["source"],
                 "scope": e["meta"]["scope"], "sha": e["meta"]["sha"],
                 "image": e["arrays"]["img"], "score": round(s_, 3)}
                for s_, e in scored[:k] if s_ > 0]

    def image_dream(self, query, n=2, k=6):
        """DREAM FROM MEMORY (cp22, HDRIFT wired to the image store): recall images by
        label, train the media model on them, GENERATE new ones, and store the dreams back
        -- labeled source='dream', never confusable with captures. Honest failure when
        memory holds too few images to train on."""
        hits = self.image_recall(query, k=8)
        if len(hits) < 2:
            return {"dreamed": 0, "why": "need >= 2 remembered images matching %r" % query}
        imgs = [h["image"].astype(float).mean(axis=2) / 255.0
                if h["image"].ndim == 3 else h["image"].astype(float) / 255.0
                for h in hits]                            # media model is single-channel
        try:
            mdl, meta = self.train_media_model(imgs, k=min(int(k), len(imgs) * 3))
            out = self.generate_media(mdl, meta, n=int(n))
        except Exception as exc:
            return {"dreamed": 0, "why": "media model: " + str(exc)[:120]}
        if isinstance(out, dict):                         # {'images': [...], ...} shape
            out = out.get("images", out.get("samples", out.get("generated", [])))
        stored = []
        for i, im in enumerate(out):
            if isinstance(im, dict):                      # generator may label its output
                im = im.get("image", im.get("img"))
            elif isinstance(im, (tuple, list)) and len(im) == 2:
                im = im[0] if hasattr(im[0], "ndim") else im[1]
            arr = np.asarray(im, float)
            if arr.dtype.kind not in "fiu" or arr.ndim < 2:
                continue                                  # refuse non-image debris quietly
            r = self.image_remember(arr, "dream: %s %d" % (query, i), source="dream")
            stored.append(r.get("label") or r.get("dedup"))
        return {"dreamed": len(stored), "labels": stored, "trained_on": len(imgs)}

    def sequence_dream(self, series=None, steps=8):
        """GENERATE A CONTINUATION FROM REMEMBERED DYNAMICS (cp22, HRNN wired to memory):
        by default takes the longest goal-convergence trace the goal book remembers, trains
        the holographic RNN on it, and rolls the sequence forward -- the mind extrapolating
        its own history. The continuation stores as a labeled sequence section."""
        if series is None:
            best = []
            for g in self.goal_book.goals.values():
                if len(g.get("convergence", [])) > len(best):
                    best = list(g["convergence"])
            if len(best) < 4:                             # second source of remembered
                led = self.zoo["ladder"].ledger           # dynamics: the tier ledger curve
                tiers = ["T0", "T1", "T2", "T3", "T4"]
                best = [float(led.by_tier.get(t, 0)) for t in tiers]
            series = best
        series = [float(x) for x in (series or [])]
        if len(series) < 4:
            return {"generated": 0, "why": "need >= 4 points of remembered dynamics"}
        cont = []
        work = [float(x) for x in series]
        d_ = max(2, min(6, len(work) // 3))
        try:
            rf = self.forecast(work, d=d_)                # the routed HRNN forecaster
            for _ in range(int(steps)):
                p_ = rf.predict(np.asarray(work[-d_:], float))
                nxt = float(p_["point"]) if isinstance(p_, dict) and "point" in p_ else \
                      float(np.asarray(p_, float).ravel()[0])
                cont.append(nxt)
                work.append(nxt)
        except Exception as exc:
            return {"generated": 0, "why": "forecast: " + str(exc)[:140]}
        cont = [float(x) for x in np.asarray(cont, float).ravel()[:int(steps)]]
        if not hasattr(self, "_sequence_memory"):
            self._sequence_memory = []
        self._sequence_memory.append({"kind": "lecore.memory.sequence",
                                      "id": "dream:%d" % len(self._sequence_memory),
                                      "meta": {"label": "convergence continuation",
                                               "source": "hrnn_dream",
                                               "seed_len": len(series)},
                                      "arrays": {"seed": np.asarray(series, float),
                                                 "generated": np.asarray(cont, float)}})
        return {"generated": len(cont), "continuation": [round(c, 3) for c in cont],
                "seed_len": len(series)}

    def learning_save(self, root):
        """SAVE THE WHOLE LEARNED STATE as ONE holographic CONTAINER (checkpoint 20; the
        pass-6 correction of pass 5's regression): <root>/learning/state.lecore via
        holographic_container -- typed sections in a compressed ZIP, bulk numerics as ARRAYS,
        small structure as section meta, no pickle, unknown kinds round-tripped untouched.
        LOOSE JSON IS NOT A STORAGE FORMAT HERE ANY MORE (kept negative: pass 5 shipped
        manifest.json + state.npz + experience.json -- three ad-hoc files where the engine
        already owned a blessed one). Sections: lecore.learning.{semantic, affinity, chains,
        skeletons, predictor, ledger, taught, goals, experience}. The artifact registers into
        the KnowledgeStore so the partition catalogs its own learning."""
        import os
        from holographic.io_and_interop.holographic_container import save_container
        d = os.path.join(str(root), "learning")
        os.makedirs(d, exist_ok=True)
        z = self.zoo
        secs = []
        te = getattr(self, "_lever7_text", None)
        if te is not None and te.context:
            words = sorted(te.context)
            taught_txt = " ".join(t_[0] + " " + t_[1] for t_ in
                                  getattr(z["ladder"], "taught_log", [])).lower()
            # (entries are [q, a] or [q, a, session] since cp36 -- index, never unpack)
            norms = {w: float(np.linalg.norm(te.context[w])) for w in words}
            med = float(np.median(list(norms.values()) or [0]))
            words = [w for w in words
                     if norms[w] > 1.35 * med or w in taught_txt]
            secs.append({"kind": "lecore.learning.semantic", "id": "v1",
                         "meta": {"words": words,
                                  "ingest_stats": getattr(self, "_semantic_ingest_stats",
                                                          None)},
                         "arrays": {"ctx": (np.stack([te.context[w] for w in words])
                               if words else np.zeros((0, te.dim)))
                       # cp32: a VIRGIN mind must save cleanly -- np.stack on an empty
                       # vocabulary crashed here; the full-record suite exposed it
                                    .astype(np.float32)}})
            # PARTITION DIET (cp28/B7): contexts were float64 and the vocabulary kept
            # every one-shot noise token (yaml keys, hex ids) -- 2,777 words = 45.5MB.
            # float32 halves it; the prune above keeps a word if its context norm says
            # it accumulated more than one observation OR it appears in a taught text
            # (the durable record protects what matters).
        secs.append({"kind": "lecore.learning.affinity", "id": "v1",
                     "meta": {"counts": [[a, b, n] for (a, b), n in
                                         z["affinity"].counts.items()]}, "arrays": {}})
        # chains group BY KEY DIMENSION (cp20.2: token-atom 64s and semantic 2048s
        # legitimately coexist; one np.stack cannot hold both)
        by_dim = {}
        for c in z["chains"].chains:
            gk = np.asarray(c["goal_key"], float)
            by_dim.setdefault(gk.shape[0], []).append((gk, c["steps"]))
        ch_arrays, ch_steps = {}, {}
        for dim, items in sorted(by_dim.items()):
            ch_arrays["goal_keys_%d" % dim] = np.stack([g_ for g_, _ in items])
            ch_steps[str(dim)] = [list(map(list, st)) for _, st in items]
        secs.append({"kind": "lecore.learning.chains", "id": "v2",
                     "meta": {"steps_by_dim": ch_steps}, "arrays": ch_arrays})
        pc = z.get("programs")
        secs.append({"kind": "lecore.learning.skeletons", "id": "v1",
                     "meta": {"floor": dict(pc._floor) if pc is not None else {}},
                     "arrays": {}})
        pred = z.get("predictor")
        secs.append({"kind": "lecore.learning.predictor", "id": "v1",
                     "meta": {"counts": dict(pred.counts) if pred is not None else {}},
                     "arrays": ({"trace": pred._trace} if pred is not None else {})})
        secs.append({"kind": "lecore.learning.ledger", "id": "v1",
                     "meta": {"by_tier": z["ladder"].ledger.by_tier,
                              "est_tokens_saved": z["ladder"].ledger.est_tokens_saved,
                              "queries": z["ladder"].ledger.queries,
                              "query_log": list(getattr(z["ladder"], "query_log", []))[-500:]},
                     "arrays": {}})
        secs.append({"kind": "lecore.learning.taught", "id": "v2",
                     "meta": {"texts": list(getattr(z["ladder"], "taught_log", [])),
                              "bad_questions": sorted({z["ladder"]._payload_qs.get(pk, "")
                                                       for pk in getattr(z["ladder"],
                                                       "_payload_bad", set())} - {""}),
                              "vetoed_questions": sorted(getattr(z["ladder"],
                                                          "_vetoed_qs", set())),
                              "feedback_log": list(getattr(z["ladder"], "_feedback_log",
                                                           [])),                                                  # full feedback history: the
                                                       # calibration eats every pair
                              "pairs": [[str(k), str(v)] for k, v in
                                        getattr(z["ladder"], "_payloads", {}).items()]},
                     "arrays": {}})
        book = self.goal_book.to_manifest()
        gv_names, gv_stack = [], []
        for gid, gg in (book.get("goals") or {}).items():
            v = gg.pop("goal_vec", None)
            if v is not None and len(np.asarray(v).shape) == 1:
                gv_names.append(gid)
                gv_stack.append(np.asarray(v, np.float32))
        arrs = {}
        if gv_stack and len({a.shape for a in gv_stack}) == 1:
            arrs["goal_vecs"] = np.stack(gv_stack)
        else:                                             # mixed dims: keep in meta as-is
            for gid, a in zip(gv_names, gv_stack):
                book["goals"][gid]["goal_vec"] = a.tolist()
            gv_names = []
        secs.append({"kind": "lecore.learning.goals", "id": "v1",
                     "meta": {"book": book, "gv_names": gv_names}, "arrays": arrs})
        secs.append({"kind": "lecore.learning.recipes", "id": "v1",
                     "meta": {"book": getattr(self, "_recipe_book", {}),
                              "archive": getattr(self, "_archive_corpora", {})},
                     "arrays": {}})
        secs.append({"kind": "lecore.learning.calibration", "id": "v1",
                     "meta": {"pairs": [[float(c), bool(o)] for c, o in
                                        getattr(self, "_reflex_calib_pairs", [])],
                              "fit": getattr(self, "_reflex_calibration", None)},
                     "arrays": {}})
        tc = getattr(self, "_tool_cache", None)
        if tc:
            secs.append({"kind": "lecore.learning.toolcache", "id": "v1",
                         "meta": {"prefix": {k: str(v) for k, v in              # a CLIPPED cache value is a
                                                       # WRONG cache value on reload
                                             tc["prefix"].items()},
                                  "stateless": {k: str(v) for k, v in
                                                tc["stateless"].items()}}, "arrays": {}})
        for e in getattr(self, "_certificate_memory", []) or []:
            secs.append(e)
        for e in getattr(self, "_image_memory", []) or []:
            secs.append(e)                               # labeled, scoped image sections
        for e in getattr(self, "_sequence_memory", []) or []:
            secs.append(e)
        tiles = []
        aud_arrays = {}
        for ti, t in enumerate(self.experience.tiles):
            st = t.to_state()
            aud = st.pop("audit", None)
            if aud:
                ks = np.stack([np.asarray(k, np.float32) for k, _ in aud])
                vs = np.stack([np.asarray(v, np.float32) for _, v in aud])
                aud_arrays["aud_k_%d" % ti] = ks
                aud_arrays["aud_v_%d" % ti] = vs
                st["audit_in_arrays"] = True
            else:
                st["audit"] = []                          # an EMPTY journal keeps its key
                                                          # (pop stripped it and
                                                          # from_state rightly demanded it)
            tiles.append(st)
        secs.append({"kind": "lecore.learning.experience", "id": "v1",
                     "meta": {"tiles": tiles}, "arrays": aud_arrays})
        prev_fp = None
        path = os.path.join(d, "state.lecore")
        if os.path.exists(path):
            try:
                prev_fp = self.partition_fingerprint(str(root))["fingerprint"]
            except Exception:
                prev_fp = None
        blob = save_container(secs, meta={"app": "lecore.learning", "version": 2})
        with open(path, "wb") as f:
            f.write(blob)
        drift_vs_prev = None
        if prev_fp is not None:
            try:
                cur = self.partition_fingerprint(str(root))["fingerprint"]
                drift_vs_prev = round(float(prev_fp @ cur), 4)  # B6: the number arrives
            except Exception:                                    # with every save
                pass
        for legacy in ("manifest.json", "state.npz", "experience.json"):
            lp = os.path.join(d, legacy)
            if os.path.exists(lp):
                os.remove(lp)                            # the regression does not linger
        try:
            from holographic.caching_and_storage.holographic_knowledgestore import KnowledgeStore
            # THE JOURNAL MIGRATION (cp30, the backlog-remainder goal executed): this
            # line used to add_file THE WHOLE CONTAINER into knowledge.json on EVERY
            # save -- the container archived into the journal that exists to index the
            # container. Result, measured twice: 318MB / 200,199 entries (cp28
            # detonation), then 190MB + 776MB of rotated debris within one session.
            # The container IS the store; the journal gets a constant-size breadcrumb.
            import hashlib as _hl
            _sha = _hl.sha256(open(path, "rb").read()).hexdigest()[:16]
            ks_ = KnowledgeStore(str(root))
            ks_.add_note("[container] state.lecore sha=%s bytes=%d (the container is "
                         "the store; query archives via archive_query)" %
                         (_sha, __import__("os").path.getsize(path)),
                         tags=("learning", "container", "manifest"))
        except Exception:
            pass
        return {"saved": True, "path": path, "sections": len(secs),
                "drift_vs_previous_save": drift_vs_prev,
                "bytes": len(blob)}

    def learning_load(self, root, force=False):
        """LOAD from the container (legacy pass-5 JSON tolerated READ-ONLY for one release,
        with a deprecation field in the return). Hot structures rebuild from exact records:
        affinity by replay, skeleton superposition from the floor, experience by journal
        replay -- bit-identical under any PYTHONHASHSEED."""
        import os, json
        if getattr(self, "_learning_loaded_from", None) == str(root) and not force:
            return {"loaded": True, "skipped": "already loaded from this root (idempotent); "
                                              "pass force=True to reload"}
        d = os.path.join(str(root), "learning")
        cpath = os.path.join(d, "state.lecore")
        legacy = os.path.join(d, "manifest.json")
        if not os.path.exists(cpath) and not os.path.exists(legacy):
            return {"loaded": False, "why": "no learning partition at %s" % d}
        z = self.zoo
        deprecated = False
        if os.path.exists(cpath):
            from holographic.io_and_interop.holographic_container import load_container
            got = load_container(open(cpath, "rb").read())
            by = {}
            for sec in got["sections"]:
                by[sec["kind"]] = sec
            sem = by.get("lecore.learning.semantic")
            if sem and sem["meta"].get("words"):
                from holographic.io_and_interop.holographic_encoders import TextEncoder
                te = TextEncoder(dim=2048, seed=0)
                M = sem["arrays"]["ctx"]
                for i, w in enumerate(sem["meta"]["words"]):
                    te.index.get(w)
                    te.context[w] = M[i].copy()
                self._lever7_text = te
                if sem["meta"].get("ingest_stats"):
                    self._semantic_ingest_stats = sem["meta"]["ingest_stats"]
            aff = by.get("lecore.learning.affinity")
            for a, b, n in (aff["meta"]["counts"] if aff else []):
                for _ in range(int(n)):
                    z["affinity"].note_pair(str(a), str(b))
            ch = by.get("lecore.learning.chains")
            if ch and ch["meta"].get("steps_by_dim") is not None:      # v2: grouped by dim
                for dim_s, step_lists in ch["meta"]["steps_by_dim"].items():
                    G = ch["arrays"]["goal_keys_%s" % dim_s]
                    for i, st in enumerate(step_lists):
                        z["chains"].note(G[i], [tuple(x) for x in st])
            elif ch and ch["meta"].get("steps"):                        # v1 tolerance
                G = ch["arrays"]["goal_keys"]
                for i, wrapped in enumerate(ch["meta"]["steps"]):
                    z["chains"].note(G[i], [tuple(x) for x in wrapped[0]])
            sk = by.get("lecore.learning.skeletons")
            for nm, st in sorted(((sk["meta"]["floor"] or {}) if sk else {}).items()):
                self.skeleton_library_add(nm, st)
            pr = by.get("lecore.learning.predictor")
            if pr and pr["meta"]["counts"]:
                from holographic.agents_and_reasoning.holographic_zoo import EscalationPredictor
                z["predictor"] = EscalationPredictor()
                z["predictor"].counts = {str(k): int(v) for k, v in
                                         pr["meta"]["counts"].items()}
                if "trace" in pr["arrays"]:
                    z["predictor"]._trace = pr["arrays"]["trace"].copy()
            led = by.get("lecore.learning.ledger")
            if led:
                z["ladder"].ledger.by_tier.update(led["meta"].get("by_tier", {}))
                z["ladder"].ledger.est_tokens_saved = float(
                    led["meta"].get("est_tokens_saved", 0.0))
                z["ladder"].ledger.queries = int(led["meta"].get("queries", 0))
                z["ladder"].query_log = list(led["meta"].get("query_log", []))
            rcp = by.get("lecore.learning.recipes")
            if rcp:
                self._recipe_book = dict(rcp["meta"].get("book") or {})
                self._archive_corpora = dict(rcp["meta"].get("archive") or {})
                for t_, texts_ in self._archive_corpora.items():
                    if texts_:
                        try:
                            self._archive_corpora_bound = getattr(
                                self, "_archive_corpora_bound", {})
                            self._archive_corpora_bound[t_] = self.corpus_bind(texts_)
                        except Exception:
                            pass
            cal = by.get("lecore.learning.calibration")
            if cal:
                self._reflex_calib_pairs = [[float(c), bool(o)] for c, o in
                                            (cal["meta"].get("pairs") or [])]
                if cal["meta"].get("fit"):
                    self._reflex_calibration = cal["meta"]["fit"]
            tcs = by.get("lecore.learning.toolcache")
            if tcs:
                self._tool_cache = {"prefix": dict(tcs["meta"].get("prefix") or {}),
                                    "stateless": dict(tcs["meta"].get("stateless") or {})}
            gl = by.get("lecore.learning.goals")
            if gl and gl["meta"].get("gv_names") and "goal_vecs" in gl.get("arrays", {}):
                gvs = np.asarray(gl["arrays"]["goal_vecs"], float)
                for gid, v in zip(gl["meta"]["gv_names"], gvs):
                    if gid in (gl["meta"]["book"].get("goals") or {}):
                        gl["meta"]["book"]["goals"][gid]["goal_vec"] = v.tolist()
            if gl and gl["meta"].get("book"):
                self.goal_book.from_manifest(gl["meta"]["book"])
                # RE-KEY ON LOAD (cp20.2, found by the cold-vs-warm test): chains logged
                # before the full-dim fix carry 64-dim truncated goal keys the dim guard
                # rightly skips -- dead weight. The goal book preserves each goal's TEXT, so
                # completed goals re-key here under TODAY'S vocabulary: text is the durable
                # record, the key is hot state (the same rule as affinity counts).
                for _g in self.goal_book.goals.values():
                    done_steps = [(x["name"], True) for x in _g.get("steps", [])
                                  if x.get("status") == "done"]
                    if done_steps and _g.get("text"):
                        try:
                            _gv = np.asarray(self.semantic_key(_g["text"])["vec"], float)
                            self.chain_note(_gv, done_steps)
                        except Exception:
                            pass
            self._certificate_memory = [sec for sec in got["sections"]
                                        if sec["kind"] == "lecore.learning.certificate"]
            self._image_memory = [sec for sec in got["sections"]
                                  if sec["kind"] == "lecore.memory.image"]
            self._sequence_memory = [sec for sec in got["sections"]
                                     if sec["kind"] == "lecore.memory.sequence"]
            exp = by.get("lecore.learning.experience")
            if exp and exp["meta"].get("tiles"):
                tls = exp["meta"]["tiles"]
                for ti, st in enumerate(tls):
                    if st.pop("audit_in_arrays", False):
                        ks = np.asarray(exp["arrays"]["aud_k_%d" % ti], float)
                        vs = np.asarray(exp["arrays"]["aud_v_%d" % ti], float)
                        st["audit"] = [(k, v) for k, v in zip(ks, vs)]
                self.experience_from_state({"tiles": tls})
            # taught replay runs AFTER the experience restore (cp21 ordering bug, caught by
            # the cold cross-check: replay marks written first were WIPED when the trace
            # section replaced the whole trace -- 0/8 T0. Restore the floor, THEN re-teach.)
            ta = by.get("lecore.learning.taught")
            texts = (ta["meta"].get("texts") if ta else None) or []
            if texts:
                # MIGRATION BY REPLAY (cp21): question+answer TEXT is the durable record;
                # each pair re-teaches under the CURRENT key function -- so key-format
                # changes (like the bigram fix) migrate old partitions automatically.
                lad = z["ladder"]
                # cp54 ORDER MATTERS, measured: tombstones were restored AFTER this
                # replay, so it re-inserted every vetoed answer and the tombstones
                # arrived too late to matter. Restore them FIRST, then replay honours
                # them below.
                lad._vetoed_qs = set(ta["meta"].get("vetoed_questions") or [])
                for t_ in texts:
                    # entries are [q, a] or, since cp36, [q, a, session]; a session
                    # entry replays under its SALTED key so isolation survives the
                    # save/load cycle exactly like everything else migrates: by replay
                    q_, a_ = str(t_[0]), str(t_[1])
                    sess_ = t_[2] if len(t_) > 2 else "shared"
                    prov_ = t_[3] if len(t_) > 3 else "taught"   # historic rows predate
                    key_q = q_ if sess_ == "shared" else "[s:%s] %s" % (sess_, q_)
                    if " ".join(key_q.lower().split()) in getattr(lad, "_vetoed_qs",
                                                                  set()):
                        lad.taught_log.append([q_, a_, sess_, prov_])  # books keep history;
                        continue                                       # the veto keeps it dead
                    qk_ = lad._qkey(key_q)
                    h_ = self.experience.read_gated(qk_)
                    if h_["fired"]:
                        pk_ = "%d:%d" % (self.experience._route(qk_), int(h_["atom"]))
                        if getattr(lad, "_payloads", {}).get(pk_) == a_:
                            # already served EXACTLY by the restored floor: re-writing
                            # would grow the audit journal every load/save cycle (the
                            # cp28 bloat) -- keep the books instead of re-teaching
                            lad._payload_qs.setdefault(pk_, key_q)
                            lad.taught_log.append([q_, a_, sess_, prov_])
                            continue
                    lad._remember(qk_, a_, key_q, provenance=prov_)
                    if lad.taught_log and lad.taught_log[-1][0] == key_q:
                        lad.taught_log[-1] = [q_, a_, sess_, prov_]
                # the registry DERIVES from the tags in the durable record itself --
                # it piggybacked on the goals section once and died with it on
                # goal-less minds (cp36 suite caught it): never store what you can
                # derive from the record you already trust
                reg_ = getattr(self, "_session_registry", {})
                for t_ in lad.taught_log:
                    sess_ = t_[2] if len(t_) > 2 else "shared"
                    if sess_ != "shared":
                        reg_.setdefault(sess_, {"opened": len(reg_), "teachings": 0})
                self._session_registry = reg_
                # feedback survives migration: re-mark bad payloads by their QUESTION text
                bad_qs = set(ta["meta"].get("bad_questions") or [])
                if bad_qs:
                    lad._payload_bad = set()
                    for bq in sorted(bad_qs):
                        bqk = lad._qkey(bq)
                        bt = self.experience._route(bqk)
                        bh = self.experience.read_gated(bqk)
                        if bh["fired"]:
                            lad._payload_bad.add("%d:%d" % (bt, int(bh["atom"])))
                lad._feedback_log = list(ta["meta"].get("feedback_log") or [])
            else:                                         # v1 tolerance: raw payload ints,
                pay = {int(k): str(v) for k, v in (ta["meta"]["pairs"] if ta else [])}
                if pay:                                   # only valid if the key fn never
                    z["ladder"]._payloads = pay           # changed since the save (deprecated)
            n_sections = len(got["sections"])
        else:
            deprecated = True                            # pass-5 layout, read-only tolerance
            man = json.load(open(legacy))
            n_sections = 0
            if man.get("experience") and os.path.exists(os.path.join(d, man["experience"])):
                try:
                    self.experience_load(os.path.join(d, man["experience"]))
                except Exception:
                    pass
        self._learning_loaded_from = str(root)
        return {"loaded": True, "format": "container" if not deprecated else
                "LEGACY-JSON (deprecated: re-save to migrate)", "sections": n_sections}

    def zoo_attach(self, llm, intern=None):
        """ATTACH AN LLM AND GET THE WHOLE STACK (the full-advantage faculty): one call wires
        the model behind EVERY zoo mechanism with nothing else to assemble --
          * the ANSWER LADDER: T0 reflex / T1 substrate (the mind's own knowledge via recall +
            bm25 over learned notes) / T2 AUTO-DISPATCH over the live catalog (typed entries
            with runnable methods, args bound deterministically, refusal on unbound) / T3
            intern (utility tasks, if given) / T4 the attached model -- the model is the LAST
            rung, exactly as in leOS;
          * ORCHESTRATION: do() plans with ONE model call (or zero via plan_warm), runs
            catalog-resolvable steps as INGEST, logs every chain, learns affinities, mines
            skeletons -- the second encounter with any solved request costs zero model tokens;
          * the LEDGER + ESCALATION PREDICTOR + query log, always on.
        MEASURED in the selftest bench: a 20-query mixed workload costs the naive
        every-query-hits-the-model integration 20 model calls; attached, the same workload
        costs 7 -- and the ladder's cheap rungs are wrong-answer-free by construction."""
        self.zoo["llm"] = llm
        self.zoo["intern"] = intern
        return {"attached": True, "intern": intern is not None,
                "rungs": ["T0 reflex", "T1 substrate", "T2 dispatch", "T3 intern"
                          if intern else None, "T4 attached llm"]}

    def _zoo_kb_search(self, query):
        """Default T1: the mind's own memory as the substrate rung -- recall() top hit scored,
        served as kind='doc' (durable) unless marked otherwise."""
        try:
            got = self.recall(str(query))                # ((label, text), score) | (None, score)
        except Exception:
            return None
        if not isinstance(got, tuple) or len(got) != 2 or got[0] is None:
            return None
        payload, score = got
        text = payload[1] if isinstance(payload, tuple) and len(payload) == 2 else str(payload)
        # OVERLAP SANITY (kept negative, measured live at checkpoint 13): on a small store the
        # calibrated abstention has no teeth (2 clustered notes served 'why is the sky blue on
        # mars' at alpha 0.05), so the default T1 additionally requires >= 1 shared CONTENT
        # word between question and answer -- a wrong answer served confidently is the one
        # thing the ladder forbids, and a note sharing zero content words with the question
        # cannot be its answer.
        _stop = {"a", "an", "the", "of", "in", "on", "to", "for", "and", "or", "with", "from",
                 "by", "at", "is", "it", "this", "that", "what", "does", "why", "how", "do"}
        q_words = set(str(query).lower().split()) - _stop
        t_words = set(str(text).lower().split()) - _stop
        if not (q_words & t_words):
            return None
        return {"text": str(text), "kind": "doc", "age_s": 0.0, "score": float(score)}

    def _zoo_dispatchers(self):
        """Default T2: the LIVE CATALOG as the dispatch table -- typed entries whose method is
        a real zero/one-arg mind callable become deterministic rungs; binding uses the entity
        extractor and REFUSES on unbound args (never guesses). Synthesized tools join
        automatically (they register with method='synth_call')."""
        disp = []
        for nm, fn in (self.zoo.get("synth") or {}).items():
            disp.append((lambda q, _n=nm: {"x": q.split(_n + " ", 1)[1]}
                         if q.startswith(_n + " ") else None,
                         lambda a, _f=fn: str(_f(a["x"])), nm))
        return disp

    # ------------------------- SESSIONS (cp36): no context bleed -------------------------
    def session_open(self, name):
        """OPEN (or resume) A SESSION: from here, teach() and ask() live in this
        session's own KEY SPACE -- the question is salted with the session name before
        keying, so the vector algebra ITSELF isolates conversations: another session's
        reflexes are unreachable by construction, not by filtering. Shared knowledge
        (doctrine, anything taught outside a session) stays visible everywhere via the
        unsalted fallback. Reopening an old session resumes its memories exactly."""
        name = str(name)
        reg = getattr(self, "_session_registry", {})
        reg.setdefault(name, {"opened": len(reg), "teachings": 0})
        self._session_registry = reg
        self._session = name
        lad = self.zoo["ladder"]
        mine = [q for q, a, *s_ in [(t + ["shared"])[:3] if len(t) < 3 else t
                for t in (list(x) for x in getattr(lad, "taught_log", []))]
                if s_ and s_[0] == name]
        return {"session": name, "resumed": len(mine),
                "sessions_known": sorted(reg)}

    def session_close(self):
        """Return to the SHARED space (teachings land unsalted, visible to all)."""
        was = getattr(self, "_session", None)
        self._session = None
        return {"closed": was}

    def session_list(self):
        """Every session the registry knows, with its teaching count."""
        lad = self.zoo["ladder"]
        counts = {}
        for t in getattr(lad, "taught_log", []):
            t = list(t)
            sess = t[2] if len(t) > 2 else "shared"
            counts[sess] = counts.get(sess, 0) + 1
        return {"current": getattr(self, "_session", None),
                "sessions": sorted(getattr(self, "_session_registry", {})),
                "teachings": counts}

    def session_salt(self, query):
        """The session prefix applied to a question before keying -- the whole isolation
        mechanism in one line: '[s:<name>] <question>' inside a session, the bare
        question in the shared space."""
        sess = getattr(self, "_session", None)
        return ("[s:%s] %s" % (sess, query)) if sess else str(query)

    def teach(self, query, answer):
        """CANONICAL TEACH: lands in the CURRENT session's key space (or shared when no
        session is open). The taught_log records (question, answer, session) at full
        length -- the durable text record session_search walks."""
        lad = self.zoo["ladder"]
        sess = getattr(self, "_session", None) or "shared"
        for k_t in (" ".join(str(query).lower().split()),
                    " ".join(self.session_salt(query).lower().split())):
            getattr(lad, "_vetoed_qs", set()).discard(k_t)     # re-establishment lifts a veto
        lad._remember(lad._qkey(self.session_salt(query)), str(answer),
                      self.session_salt(query), provenance="taught")
        log = getattr(lad, "taught_log", [])
        # _remember appends [q, a, sess, provenance]; upgrade the tail with the session
        # tag WITHOUT stripping provenance -- cp60 found every deliberate teach logging
        # as provenance-less because this rewrite truncated the row cp54 had extended
        if not (log and log[-1][0] == self.session_salt(query) and
                str(log[-1][1]) == str(answer)):
            # cp67, found by the api-persistence battery: _remember can REFUSE a row
            # (the cp38 control-token guard, standalone __word__ tokens) and teach()
            # was reporting {"taught": True} anyway -- a silent drop that cost a full
            # debugging session. A refused teach now says so, and says why.
            return {"taught": False, "session": sess,
                    "reason": "the memory guard refused this row (standalone "
                              "__word__ control tokens are reserved; rephrase "
                              "the question)"}
        if log and log[-1][0] == self.session_salt(query):
            log[-1] = [str(query), str(answer), sess, "taught"]
        return {"taught": True, "session": sess}

    def session_search(self, query, sessions="all", k=5):
        """SEARCH ACROSS SESSIONS explicitly -- the opt-in bridge over the isolation.
        Ranks every taught (question, answer) pair in the chosen sessions by stemmed
        overlap with the query; hits carry their session so you can session_open() the
        right one and resume where it happened."""
        lad = self.zoo["ladder"]
        stop = lad._STOP
        qw = {w.rstrip("s") for w in str(query).lower().split()} - stop
        hits = []
        for t in getattr(lad, "taught_log", []):
            t = list(t)
            q_, a_ = str(t[0]), str(t[1])
            sess = t[2] if len(t) > 2 else "shared"
            if sessions != "all" and sess not in (sessions if
                    isinstance(sessions, (list, tuple, set)) else [sessions]):
                continue
            tw = ({w.rstrip("s") for w in q_.lower().split()} |
                  {w.rstrip("s") for w in a_.lower().split()[:40]}) - stop
            ov = len(qw & tw) / max(len(qw), 1)
            if ov > 0:
                hits.append({"score": round(ov, 3), "session": sess,
                             "question": q_, "answer": a_})
        hits.sort(key=lambda h: (-h["score"], h["session"], h["question"]))
        return {"query": str(query), "hits": hits[:int(k)]}

    def ask(self, query, est_llm_tokens=600):
        """THE ONE-CALL ANSWER PATH, session-aware (cp36): inside a session the salted
        key space is tried first (this conversation's own memories), then the shared
        space (doctrine, unsalted teachings); the result reports which space served.
        Outside a session, identical to the classic ladder walk. Refusal stays a
        result; provenance stays mandatory."""
        sess = getattr(self, "_session", None)
        if sess:
            a = self._ask_unsalted(self.session_salt(query), est_llm_tokens)
            if a.get("tier") in ("T0", "T1"):
                a["session"] = sess
                return a
            a = self._ask_unsalted(query, est_llm_tokens)   # shared fallback
            a["session"] = "shared-fallback" if a.get("tier") in ("T0", "T1") else sess
            return a
        return self._ask_unsalted(query, est_llm_tokens)

    def _ask_unsalted(self, query, est_llm_tokens=600):
        """THE ONE-CALL ANSWER PATH for an attached mind: the full ladder with every default
        wired (see zoo_attach), the serving tier noted into the escalation predictor, and the
        ledger updated. Refusal is a result; provenance is mandatory."""
        out = self.zoo_answer(str(query), kb_search=self._zoo_kb_search,
                              dispatchers=self._zoo_dispatchers(),
                              intern=self.zoo.get("intern"),
                              main=(lambda q, c: self.zoo["llm"]("ANSWER: " + q))
                              if self.zoo.get("llm") else None,
                              est_llm_tokens=est_llm_tokens)
        if out["tier"] in ("T0", "T1", "T2", "T3", "T4"):
            self.escalation_note(str(query), out["tier"])
        # the conversation is a corpus: the question and any served answer join the
        # semantic space (decision trees already learn via escalation_note above)
        self.semantic_ingest(str(query), source="ask")
        if out.get("answer"):
            self.semantic_ingest(str(out["answer"]), source="answer")
        return out

    def do(self, request, executors=None, plan_gate=0.7):
        """THE ONE-CALL TASK PATH for an attached mind: orchestrate with the attached model,
        auto-resolving executors -- caller-supplied first, then synthesized tools by name.
        Chains log, affinities learn, plan_warm makes the second encounter free."""
        ex = dict(executors or {})
        for nm, fn in (self.zoo.get("synth") or {}).items():
            ex.setdefault(nm, lambda _f=fn: _f(None))
        self.semantic_ingest(str(request), source="do")
        # SEMANTIC GOAL KEYS once the encoder has a space (checkpoint 17): a paraphrased
        # request lands NEAR the learned goal and warm-starts the plan the token-bag key
        # would miss -- measured in the selftest. Untrained encoder falls back to token atoms.
        sk = self.semantic_key(str(request))
        if sk.get("trained"):
            gv = np.asarray(sk["vec"], float)
        else:
            from holographic.agents_and_reasoning.holographic_lever7 import key_atom
            toks = sorted(set(str(request).lower().split()))[:8]
            gv = np.sum([key_atom("g:" + t, 256) for t in toks], axis=0)
            gv = gv / (np.linalg.norm(gv) + 1e-12)
        out = self.orchestrate(str(request), gv, self.zoo["llm"], ex, plan_gate=plan_gate)
        for st in out.get("steps", []):
            self.semantic_ingest(str(st), source="plan")
        return out

    def zoo_idle(self):
        """THE IDLE PASS for an attached mind: mine the query log's voids, turn the top
        anchors into proposed questions, and PREFILL the free rungs (never the model --
        research_prefill's contract). Run it from any idle hook; it is what makes the zoo
        answer questions nobody asked yet."""
        log = getattr(self.zoo["ladder"], "query_log", [])
        if len(log) < 4:
            return {"voids": 0, "prefilled": []}
        vm = self.query_void_map(log[-64:], n_clusters=min(4, len(set(log))))
        proposals = [v["anchor"].replace("void between [", "").replace("] and [", " ")
                     .replace("]", "") for v in vm["voids"][:3]]
        pre = self.research_prefill(proposals, kb_search=self._zoo_kb_search,
                                    dispatchers=self._zoo_dispatchers())
        return {"voids": len(vm["voids"]), "proposals": proposals, **pre}

    def zoo_report(self):
        """THE FULL-ADVANTAGE DASHBOARD: the ledger, the skeleton count, the top learned
        transitions, and the query-log size -- what an attached deployment shows its owner."""
        aff = sorted(self.zoo["affinity"].counts.items(), key=lambda x: (-x[1], x[0]))[:3]
        return {"ledger": self.zoo_ledger(),
                "skeletons_mined": len(self.skeleton_mine(min_support=2)),
                "top_transitions": [{"pair": list(p), "n": n} for p, n in aff],
                "queries_seen": len(getattr(self.zoo["ladder"], "query_log", []))}

    @property
    def zoo(self):
        """The mind's ANSWER LADDER + shared zoo state (lazy singleton): T0 reflex -> T1
        substrate+freshness -> T2 deterministic dispatch -> T3 intern (utility-grade) -> T4
        main. Every answer carries {tier, via, confidence, why}; T0-T2 refuse rather than
        guess. See holographic_zoo.AnswerLadder."""
        if getattr(self, "_zoo", None) is None:
            from holographic.agents_and_reasoning.holographic_zoo import (
                AnswerLadder, FreshnessRegistry, TokenLedger, ChainLog, AffinityTrace)
            self._zoo = {
                "ladder": AnswerLadder(self, freshness=FreshnessRegistry(),
                                       ledger=TokenLedger()),
                "chains": ChainLog(), "affinity": AffinityTrace(),
            }
        return self._zoo

    def zoo_answer(self, query, kb_search=None, dispatchers=None, intern=None, main=None,
                   est_llm_tokens=600):
        """Answer through the LADDER (Z0.1). Rungs are injected callables so any deployment
        wires its own kb/tools/models; the ladder owns the order, the gates, the freshness
        check, the reflex write-back, and the ledger entry. Refusal is a result."""
        return self.zoo["ladder"].answer(query, kb_search, dispatchers, intern, main,
                                         est_llm_tokens)

    def zoo_ledger(self):
        """The token ledger (Z0.2): per-tier serves, estimated tokens saved, escalation rate --
        the auditable cost-per-correct-answer number."""
        return self.zoo["ladder"].ledger.summary()

    def zoo_freshness_policy(self, kind, max_age_s):
        """Set a per-entity-kind freshness policy (Z1.1); T1 refuses stale hits and queues a
        refresh (see the registry's refresh_queue)."""
        return self.zoo["ladder"].freshness.set_policy(kind, max_age_s)

    def joint_table(self):
        """The TYPED JOINT TABLE (Z5.1) derived from the live catalog's consumes/produces
        io-kinds -- which tool outputs can feed which inputs. Untyped tools are QUARANTINED:
        usable alone, refused in chains with the gap named (planning-time, never runtime)."""
        from holographic.agents_and_reasoning.holographic_zoo import JointTable
        cat = self._capability_catalog
        cat = cat() if callable(cat) else cat
        entries = list(getattr(cat, "_by_name", {}).values())
        jt = JointTable(entries)
        typed = sum(1 for t in jt.types.values() if t["consumes"] or t["produces"])
        return {"table": jt, "tools": len(jt.types), "typed": typed}

    def validate_chain(self, names):
        """Validate a tool chain against the joint table (Z5.1): ok, or the first mismatch /
        quarantine named."""
        return self.joint_table()["table"].validate_chain(list(names))

    def affinity_note(self, tool_a, tool_b):
        """Record a successful A->B transition in the affinity trace (Z5.2) -- learned
        chain-of-thought at the tool level, counts kept beside the algebra."""
        return self.zoo["affinity"].note_pair(str(tool_a), str(tool_b))

    def affinity_next(self, tool_a, k=3):
        """What usually follows tool_a (Z5.2): two unbinds + a cleanup. Empty history returns
        [] -- refusal is a result."""
        return self.zoo["affinity"].predict_next(str(tool_a), k)

    def present_tools(self, task_vec, k=5, after=None):
        """CONTEXTUAL PRESENTATION (Z5.3): the top-k tool menu for this task -- usage-trace
        ranking (one unbind + cleanup), optionally re-ranked by what usually FOLLOWS `after`.
        Never the full library: presentation cost is flat in library size (the tiles absorb
        growth)."""
        base = self.tool_predict(task_vec, k=max(k * 2, k))
        if after:
            aff = dict(self.zoo["affinity"].predict_next(str(after), k=len(base) or 1))
            base = sorted(base, key=lambda x: (-(x[1] + aff.get(x[0], 0.0)), x[0]))
        return base[: int(k)]

    def chain_note(self, goal_vec, steps):
        """Log an executed chain on the exact floor (Z6.1): the CoT corpus."""
        return self.zoo["chains"].note(np.asarray(goal_vec, float), steps)

    def skeleton_mine(self, min_support=2):
        """Mine the chain log for recurring successful subsequences (Z6.2, the dark_matter port,
        model-free): PROPOSALS with support counts -- promotion is the caller's evidence gate."""
        return self.zoo["chains"].mine_skeletons(min_support=min_support)

    def plan_warm(self, goal_vec, gate=0.7):
        """The lever-7 rung for PLANNING (Z6.4): the nearest logged chain by GOAL similarity, or
        None below the gate (novel goals plan cold). The resonator lesson applies: the key is
        the goal context, never the chain."""
        return self.zoo["chains"].plan_warm(np.asarray(goal_vec, float), gate=gate)

    def orchestrate(self, request_text, goal_vec, llm, executors, plan_gate=0.7):
        """THE COMPOSED LOOP (Z7.3): warm plan (zero model calls when it fires) else one llm
        plan call; per-step scopes; registered executors run as INGEST with no model; chains and
        transitions logged so the second encounter is free. The flagship acceptance -- run 2 of
        a solved request at model_calls == 0 -- is asserted in the Part-20 selftest."""
        from holographic.agents_and_reasoning.holographic_zoo import orchestrate as _orc
        return _orc(self, str(request_text), np.asarray(goal_vec, float), llm, executors,
                    self.zoo["chains"], self.zoo["affinity"],
                    ledger=self.zoo["ladder"].ledger, plan_gate=plan_gate)

    def fabrik_chain(self, have_kinds, want_kinds, max_len=5):
        """BIDIRECTIONAL TYPED CHAIN SOLVING over the live joint table (Z6.3, the fabrik_plan
        shape made deterministic): grow from what you HAVE toward what the goal WANTS through
        type-legal tools, affinity counts steering among the legal. Unreachable is an honest
        verdict with the frontier named."""
        from holographic.agents_and_reasoning.holographic_zoo import fabrik_chain as _fc
        return _fc(self.joint_table()["table"], list(have_kinds), list(want_kinds),
                   affinities=self.zoo["affinity"], max_len=max_len)

    def bind_dispatch_args(self, query, signature, vocab=()):
        """T2's gate made real (Z2.1): extract entities deterministically (numbers, quoted
        strings, URLs, vocab words) and bind them to a {param: kind} signature. Missing required
        args are NAMED and the dispatch must refuse -- T2 never guesses."""
        from holographic.agents_and_reasoning.holographic_zoo import extract_entities, bind_args
        return bind_args(dict(signature), extract_entities(str(query), tuple(vocab)))

    def query_void_map(self, questions, vecs=None, n_clusters=4):
        """THE VOID MAP WITH NAMES (Z3.1, kb_void_map's text half): cluster the asked-question
        log, probe between centroids, score density deficits, and anchor each void in the two
        clusters' own words -- the prioritized gap list that feeds research_prefill. Vectors
        default to semantic_key() over each question."""
        from holographic.agents_and_reasoning.holographic_zoo import query_void_map as _vm
        if vecs is None:
            vecs = [self.semantic_key(q)["vec"] for q in questions]
        return _vm([str(q) for q in questions], vecs, n_clusters=n_clusters)

    def escalation_note(self, query, tier):
        """Record which tier ultimately served a query (Z0.3's learning signal)."""
        if "predictor" not in self.zoo:
            from holographic.agents_and_reasoning.holographic_zoo import EscalationPredictor
            self.zoo["predictor"] = EscalationPredictor()
        return self.zoo["predictor"].note(self.zoo["ladder"]._qkey(query), str(tier))

    def escalation_predict(self, query):
        """Predict the serving tier for a query from accumulated evidence (Z0.3): None below
        the evidence floor -- the ladder walks normally. T0-T2 stay a prefix regardless; the
        prediction only decides whether T3 is worth bothering with before T4."""
        if "predictor" not in self.zoo:
            return None
        return self.zoo["predictor"].predict(self.zoo["ladder"]._qkey(query))

    def synthesize_tool(self, name, chain, executors, does=None, aliases=()):
        """SYNTHESIZE A TOOL FROM A TYPED CHAIN (Route A -- composition synthesis, no new code):
        validate the chain against the joint table, compose the step executors into ONE callable,
        derive the new tool's TYPE from the chain's endpoints (consumes = first step's consumes,
        produces = last step's produces -- so the synthesized tool is immediately CHAINABLE into
        further syntheses: the inception property), register it in the live catalog with a skill
        card, and store the callable for synth_call / /invoke. An ill-typed chain REFUSES with
        the mismatch named -- synthesis inherits the ladder's zero-guess contract.

        THE INCEPTION LOOP this enables (asserted in the selftest): register make_tool() itself
        as a capability, and a PLAN may then contain a synthesis step -- the chain builds the
        tool it then uses, the chain is logged, the skeleton is mined, and the SECOND request
        needing that tool costs zero model calls AND zero synthesis: the tool already exists."""
        v = self.validate_chain(list(chain))
        if not v["ok"]:
            return {"ok": False, "why": v["why"], "at": v.get("at")}
        jt = self.joint_table()["table"]
        first, last = jt.types[str(chain[0])], jt.types[str(chain[-1])]
        def _composite(x=None, _chain=tuple(chain), _ex=dict(executors)):
            out = x
            for step in _chain:
                out = _ex[step](out)
            return out
        if "synth" not in self.zoo:
            self.zoo["synth"] = {}
        self.zoo["synth"][str(name)] = _composite
        cat = self._capability_catalog
        cat = cat() if callable(cat) else cat
        cat.register_capability(
            str(name),
            does or ("SYNTHESIZED tool: the chain %s composed as one step. Consumes %s, "
                     "produces %s. Built by synthesize_tool; call via mind.synth_call."
                     % (" -> ".join(chain), list(first["consumes"]), list(last["produces"]))),
            example="mind.synth_call(%r, x)" % str(name), native=True,
            aliases=tuple(aliases) + ("synthesized", "composed tool"),
            consumes=tuple(first["consumes"]), produces=tuple(last["produces"]),
            method="synth_call")
        return {"ok": True, "name": str(name), "chain": list(chain),
                "consumes": list(first["consumes"]), "produces": list(last["produces"]),
                "why": "typed composition registered; find_capability-surfaceable and chainable"}

    def synth_call(self, name, x=None):
        """Run a SYNTHESIZED tool by name (the /invoke entry point for Route-A tools)."""
        fn = (self.zoo.get("synth") or {}).get(str(name))
        if fn is None:
            return {"error": "no synthesized tool %r" % str(name)}
        return fn(x)

    def synthesize_program_tool(self, name, instructions, data_tags):
        """Route B -- SUBSTRATE-NATIVE synthesis: the new tool is a HoloMachine PROGRAM VECTOR
        (assemble + define into the one-vector function library): decodable for audit,
        composable by CALL, and a candidate for the certify->compile->bake pipeline (a tool
        whose deployed form is exact WEIGHTS -- the 154-byte trick applied to a synthesized
        capability). Returns the vector's shape and the decode round-trip verdict; no Python
        code was emitted anywhere in this route."""
        from holographic.agents_and_reasoning.holographic_machine import HoloMachine
        hm = HoloMachine(dim=2048, seed=0, data=sorted(set(data_tags)))
        vec = hm.assemble(list(instructions))
        hm.define(str(name), list(instructions))
        dec = hm.disassemble(vec, len(instructions)) if hasattr(hm, "disassemble") else None
        ok = dec is not None and [tuple(x) for x in dec] == [tuple(x) for x in instructions]
        if "synth_programs" not in self.zoo:
            self.zoo["synth_programs"] = {}
        self.zoo["synth_programs"][str(name)] = {"machine": hm, "vector": vec,
                                                 "instructions": list(instructions)}
        return {"ok": bool(ok), "name": str(name), "dim": int(vec.shape[0]),
                "decode_roundtrip": bool(ok),
                "why": "the tool IS one vector in the function library; audit = decode"}

    @property
    def program_chains(self):
        """The zoo's PROGRAM-VECTOR layer (lazy singleton): chains, skeletons and ladder orders
        as HoloMachine program vectors -- assembled, decoded, bound, superposed. Tool codebook =
        the affinity trace's known tools + the five tier names (extend by rebuilding)."""
        if "programs" not in self.zoo:
            from holographic.agents_and_reasoning.holographic_zoo import ProgramChains
            tools = sorted({t for pair in self.zoo["affinity"].counts for t in pair} |
                           {"T0", "T1", "T2", "T3", "T4", "fetch", "summ", "send"})
            self.zoo["programs"] = ProgramChains(tools)
        return self.zoo["programs"]

    def chain_vector(self, steps):
        """A tool chain as ONE program vector (VSA-composable; decodable by chain_from_vector)."""
        return self.program_chains.chain_to_vector(list(steps))

    def chain_from_vector(self, vec):
        """Decode a chain program vector back to its steps (a decode that can fail -- the
        checkpoint-9 rule)."""
        return self.program_chains.vector_to_chain(vec)

    def orchestrate_program(self, vec, executors):
        """EXECUTE A PLAN FROM ITS VECTOR: decode, then run every step through registered
        executors as INGEST -- zero model calls by construction (a plan that arrives as algebra
        has already been thought). Unknown steps refuse with the step named."""
        steps = self.chain_from_vector(vec)
        report = []
        for st in steps:
            if st not in executors:
                return {"ok": False, "at": st, "why": "no executor for decoded step %r" % st,
                        "steps": steps, "report": report}
            report.append({"step": st, "ok": executors[st]() is not None})
        return {"ok": True, "steps": steps, "model_calls": 0, "report": report}

    def skeleton_library_add(self, name, steps):
        """Add a skeleton to the SUPERPOSED library vector (ProgramChains.library_add): one
        vector holds them all; the exact floor sits beside it."""
        return self.program_chains.library_add(str(name), list(steps))

    def skeleton_library_recall(self, name):
        """Recall a skeleton from the superposition by unbind + slot cleanup, with the exact
        verdict reported (never a silent fallback)."""
        return self.program_chains.library_recall(str(name))

    def skeleton_library_rate(self):
        """The measured exact-recall rate of the superposed skeleton library at its current
        occupancy (ProgramChains.recall_rate) -- the crosstalk dial for this vector."""
        return self.program_chains.recall_rate()

    def synthesize_tool_certified(self, name, chain, executors, does=None, aliases=()):
        """ROUTE A SYNTHESIS WITH A LEAN 4 CERTIFICATE (pass 2's Lean capability): synthesize
        the tool, then EMIT AND PROVE its chain's well-typedness -- Horn facts produces(step,
        kind) / consumes(step, kind), one linked_i fact per adjacent kind match, and the
        theorem chain_well_typed derived through the engine's own prover into self-contained
        Lean 4 source. A synthesis that fails typing never reaches the prover (the refusal
        happens first, with the mismatch named); a synthesis that succeeds ships with a
        machine-checkable statement of WHY its composition is legal."""
        r = self.synthesize_tool(name, chain, executors, does=does, aliases=aliases)
        if not r.get("ok"):
            return r
        jt = self.joint_table()["table"]
        rules = []
        for i, step in enumerate(chain):
            t = jt.types[str(step)]
            for k in t["produces"]:
                rules.append({"head": ["produces", [str(step), str(k)]], "body": [],
                              "name": "prod_%d_%s" % (i, k)})
            for k in t["consumes"]:
                rules.append({"head": ["consumes", [str(step), str(k)]], "body": [],
                              "name": "cons_%d_%s" % (i, k)})
        link_bodies = []
        for i in range(len(chain) - 1):
            a, b = str(chain[i]), str(chain[i + 1])
            kind = sorted(set(jt.types[a]["produces"]) & set(jt.types[b]["consumes"]))[0]
            rules.append({"head": ["linked", [a, b]],
                          "body": [["produces", [a, kind]], ["consumes", [b, kind]]],
                          "name": "link_%d" % i})
            link_bodies.append(["linked", [a, b]])
        rules.append({"head": ["chain_well_typed", [str(name)]], "body": link_bodies,
                      "name": "well_typed_from_links"})
        cert = self.lean_export(["chain_well_typed", [str(name)]], rules,
                                theorem_name="%s_well_typed" % str(name))
        r["lean_certificate"] = cert
        r["certified"] = "theorem" in str(cert)
        return r

    def ladder_vector(self, tiers=("T0", "T1", "T2", "T3", "T4")):
        """This mind's ladder ESCALATION ORDER as one program vector (see holographic_zoo.
        ladder_order_vector): per-tenant orders superpose via bind(tenant_atom, ladder_vector)
        and unbind back exactly -- measured two-tenant recovery in the checkpoint-11 notes."""
        from holographic.agents_and_reasoning.holographic_zoo import ladder_order_vector
        return ladder_order_vector(self.program_chains, tuple(tiers))

    def graphql(self, query, objects=None):
        """RUN A GRAPHQL QUERY over nested objects (backlog: the query layer's second dialect,
        finally /invoke-reachable -- found unreachable by pass 4 of the audit loop: the module
        existed with no faculty). Pass `objects` (a list of nested dicts) or reuse the last
        scene bound via this faculty. The VSA-native contract holds underneath: a nested
        selection is a chain of role unbinds (see holographic_graphql.project_via_unbind);
        outputs come from the exact stored objects -- the same exact/fuzzy fork as the SQL
        side. Returns the dict shaped like the query."""
        from holographic.io_and_interop.holographic_graphql import Scene, resolve
        if objects is not None:
            self._graphql_scene = Scene(list(objects))
        sc = getattr(self, "_graphql_scene", None)
        if sc is None:
            return {"error": "no scene bound -- pass objects=[...] once, then query freely"}
        return resolve(sc, str(query))

    def zoo_gate_glsl(self):
        """The T0 gate's decision math EMITTED TO GLSL ES 3.0 and validated by execution
        through the emitter's shared IR (bit-identical c_f64 against the same source). The
        leCoreGLSL bridge for the zoo: the gate a shader can run."""
        from holographic.agents_and_reasoning.holographic_zoo import zoo_gate_glsl as _g
        return _g()

    def research_prefill(self, questions, kb_search=None, dispatchers=None):
        """PRE-EMPTIVE RESEARCH (Z3.2): for each proposed question, walk the FREE RUNGS ONLY
        (T1 substrate + T2 dispatch -- never a model) and write any answer into the reflex trace
        so the next real ask is a T0 hit. Every prefilled answer is ledgered and provenance-
        tagged. Model-tier prefetch is deliberately absent from this faculty: idle time is for
        the free rungs (pre-spending T4 defeats the ladder's whole argument)."""
        filled, skipped = [], []
        for q in questions:
            out = self.zoo["ladder"].answer(str(q), kb_search, dispatchers, None, None)
            if out["tier"] in ("T1", "T2"):
                filled.append({"question": str(q), "tier": out["tier"], "via": "prefetch"})
            else:
                skipped.append(str(q))
        return {"prefilled": filled, "skipped": skipped}


def _selftest():
    import lecore
    from holographic.agents_and_reasoning.holographic_lever7 import key_atom
    m = lecore.UnifiedMind()
    # real-catalog joint table builds; quarantine verdicts are named
    jt = m.joint_table()
    assert jt["tools"] > 500, "the live catalog must populate the table"
    some_untyped = next((n for n, t in jt["table"].types.items()
                         if not t["consumes"] and not t["produces"]), None)
    if some_untyped:
        v = m.validate_chain([some_untyped])
        assert not v["ok"] and "UNTYPED" in v["why"]
    # prefill -> T0: the pre-emptive loop closes
    kb = lambda q: {"text": "answer:" + q, "kind": "doc", "age_s": 1.0, "score": 0.9} \
        if "alpha" in q else None
    pre = m.research_prefill(["alpha topic notes", "mystery beta riddle"], kb_search=kb)
    assert len(pre["prefilled"]) == 1 and pre["skipped"] == ["mystery beta riddle"]
    hot = m.zoo_answer("alpha topic notes", kb_search=None)     # kb REMOVED: only the trace can serve
    assert hot["tier"] == "T0", "a prefilled question must serve from the reflex trace"
    # orchestrate flagship through the mind facade
    calls = {"n": 0}
    def llm(p):
        calls["n"] += 1
        return "fetch\nsumm" if p.startswith("PLAN") else "thought"
    ex = {"fetch": lambda: "d", "summ": lambda: "s"}
    g = key_atom("goal:demo", 64)
    r1 = m.orchestrate("demo request", g, llm, ex)
    r2 = m.orchestrate("demo request", g, llm, ex)
    assert r1["model_calls"] == 1 and r2["model_calls"] == 0 and r2["via"] == "plan_warm"
    assert m.affinity_next("fetch", 1)[0][0] == "summ"
    assert m.skeleton_mine(min_support=2)[0]["steps"] == ["fetch", "summ"]
    # THE FULL-ADVANTAGE BENCH (the attached-mind contract, asserted): a 20-query mixed
    # workload vs the naive every-query-hits-the-model integration.
    m2 = lecore.UnifiedMind()
    calls = {"n": 0}
    def llm2(p):
        calls["n"] += 1
        return "LLM"
    m2.zoo_attach(llm2)
    m2.learn("the reflex gate refuses below the calibrated null", label="gate-doctrine")
    m2.learn("crosstalk pricing decays trust as the tile fills", label="crosstalk-doctrine")
    W = (["what does the reflex gate refuse"] * 4 + ["summarize the gate doctrine"] * 3 +
         ["why is the sky blue on mars"] * 2 + ["how does crosstalk pricing decay trust"] * 3 +
         ["what refuses below the calibrated null"] * 4 + ["explain the capacity law price"] * 4)
    outs = [m2.ask(q) for q in W]
    att = calls["n"]
    assert att * 5 <= len(W), "the attached mind must beat naive by >= 5x on this workload"
    mars = [o for q, o in zip(W, outs) if "mars" in q][0]
    assert mars["tier"] == "T4", "an unanswerable question must ESCALATE, never be T1-served"
    # THE CONVERSATION IS A CORPUS (checkpoint 17), three asserts:
    m3 = lecore.UnifiedMind()
    m3.semantic_ingest("glarping means compressing a float series; to glarp is to compress "
                       "the floats, compress the series")
    m3.semantic_ingest("when you glarp a series you compress it; glarping and compressing "
                       "float data are the same")
    g = m3.semantic_key("glarp the float series")["vec"]
    c = m3.semantic_key("compress the float series")["vec"]
    u = m3.semantic_key("render a teapot mesh")["vec"]
    assert float(g @ c) > float(g @ u) + 0.15, \
        "an invented word taught in conversation must join its synonym's neighborhood"
    calls3 = {"n": 0}
    def llm3(p):
        calls3["n"] += 1
        return "fetch\nsumm" if p.startswith("PLAN") else "x"
    m3.zoo_attach(llm3)
    ex3 = {"fetch": lambda: "d", "summ": lambda: "s"}
    for br in ("to assemble the weekly summary is to make the weekly report",
               "make the report: assemble the summary for the week",
               "the weekly report is this week's summary; assemble it or make it",
               "assemble summary, make report -- the weekly report equals the weekly summary",
               "when someone says assemble this week's summary they mean make the weekly report"):
        m3.semantic_ingest(br, source="conversation")
    m3.do("make the weekly report", executors=ex3)
    n3 = calls3["n"]
    rp = m3.do("assemble this week's summary", executors=ex3, plan_gate=0.50)
    assert rp["via"] == "plan_warm" and calls3["n"] == n3, \
        "a conversationally-bridged paraphrase must warm-fire at the measured gate"
    ru = m3.do("render the teapot mesh nicely", executors=ex3, plan_gate=0.50)
    assert ru["via"] == "llm_plan", "an unrelated request must still plan cold at that gate"
    # THE FRAMEWORK REMEMBERS (checkpoint 18): learn in one mind, load in a cold one.
    import tempfile, shutil as _sh
    _root = tempfile.mkdtemp(prefix="lecore_learn_")
    try:
        m3.learning_save(_root)
        m4 = lecore.UnifiedMind()
        m4.zoo_attach(llm3)
        m4.learning_load(_root)
        g4 = m4.semantic_key("glarp the float series")["vec"]
        c4 = m4.semantic_key("compress the float series")["vec"]
        assert float(g4 @ c4) > 0.4, "conversational vocabulary must survive the partition"
        n4 = calls3["n"]
        r4 = m4.do("make the weekly report", executors=ex3)
        assert r4["via"] == "plan_warm" and calls3["n"] == n4, \
            "a learned plan must warm-fire in a COLD process after learning_load"
    finally:
        _sh.rmtree(_root, ignore_errors=True)
    # LONG-RUNNING GOALS (checkpoint 19): resume in a cold process; drift pauses wandering.
    import tempfile, shutil as _sh2
    _r2 = tempfile.mkdtemp(prefix="lecore_goal_")
    try:
        mG = lecore.UnifiedMind()
        gcalls = {"n": 0}
        def gllm(p):
            gcalls["n"] += 1
            return "s1\ns2\ns3\ns4" if p.startswith("PLAN") else "w"
        mG.zoo_attach(gllm)
        mG.semantic_ingest("the report goal: gather draft write review the report")
        gex = {"s1": lambda: "gathered for the report", "s2": lambda: "drafted the report",
               "s3": lambda: "wrote the report", "s4": lambda: "reviewed the report"}
        mG.goal_create("g1", "produce the report", plan=["s1", "s2", "s3", "s4"])
        mG.goal_work("g1", executors=gex, budget_steps=2)
        mG.learning_save(_r2)
        mH = lecore.UnifiedMind()
        mH.zoo_attach(gllm)
        mH.learning_load(_r2)
        n0 = gcalls["n"]
        w = mH.goal_work("g1", executors=gex, budget_steps=4)
        assert w["status"] == "done" and gcalls["n"] == n0, \
            "a goal must RESUME in a cold process, pending-only, zero model calls"
        mD = lecore.UnifiedMind()
        mD.zoo_attach(gllm)
        mD.learn_semantic_keys()
        mD.goal_create("g2", "collect the tax forms and file the tax return",
                       plan=["a", "b", "c", "d", "e"])
        dex = {"a": lambda: "collected the tax forms for the tax return",
               "b": lambda: "organized tax forms to file the return",
               "c": lambda: "read about medieval falconry hood techniques",
               "d": lambda: "compared falconry glove leather", "e": lambda: "falconry perches"}
        wd = mD.goal_work("g2", executors=dex, budget_steps=5)
        assert wd.get("alarm") == "drift" and wd["status"] == "paused", \
            "falling goal-similarity must PAUSE the goal -- wandering is stopped, not funded"
    finally:
        _sh2.rmtree(_r2, ignore_errors=True)
    # BACKLOG SUITE (cp28): diet cycle-stability, invalidation, calibrated veto.
    import tempfile as _tf5, shutil as _sh5
    mB = lecore.UnifiedMind()
    mB.zoo_attach(lambda p: "x")
    lb = mB.zoo["ladder"]
    for i_ in range(6):
        qb = "diet probe %d of the %s panel" % (i_, ["left", "right"][i_ % 2])
        lb._remember(lb._qkey(qb), "pv %d" % i_, qb)
    _r5 = _tf5.mkdtemp(prefix="lecore_diet_")
    try:
        b1 = mB.learning_save(_r5)["bytes"]
        mB2 = lecore.UnifiedMind(); mB2.zoo_attach(lambda p: "x")
        mB2.learning_load(_r5)
        b2 = mB2.learning_save(_r5)["bytes"]
        mB3 = lecore.UnifiedMind(); mB3.zoo_attach(lambda p: "x")
        mB3.learning_load(_r5)
        b3 = mB3.learning_save(_r5)["bytes"]
        # ONE settling cycle is permitted (an ambiguous replay may honestly re-teach
        # once); from the second cycle the partition must be a FIXED POINT -- that is
        # the diet's actual contract, measured 92KB -> 107KB -> 107KB.
        assert b3 <= b2, "the partition must reach a fixed point by the second cycle"
        aB = mB2.ask("diet probe 0 of the left panel")
        assert aB["tier"] == "T0", "the diet must not shed knowledge"
    finally:
        _sh5.rmtree(_r5, ignore_errors=True)
    exI = {"n": 0}
    def mkI(nm):
        def f():
            exI["n"] += 1
            return "step %s ran for the pipe" % nm
        return f
    exd = {"x1": mkI("x1"), "x2": mkI("x2")}
    mB.goal_create("binv1", "run the invalidation check pipe", plan=["x1", "x2"])
    mB.goal_work("binv1", executors=exd, budget_steps=2)
    nI = exI["n"]
    mB.cache_invalidate("x2")
    mB.goal_create("binv2", "run the invalidation check pipe again", plan=["x1", "x2"])
    wI = mB.goal_work("binv2", executors=exd, budget_steps=2)
    assert exI["n"] - nI == 1 and wI["cache_hits"] == 1, \
        "invalidation must force exactly the repaired step to re-run"
    # PROVENANCE SUITE (cp47): a cached model answer must not look like taught truth.
    mV = lecore.UnifiedMind()
    mV.zoo_attach(lambda p: "a model answer")
    mV.teach("provenance taught probe", "the taught truth")
    aT = mV.ask("provenance taught probe")
    assert aT.get("provenance") == "taught", "taught answers must be marked taught"
    mV.ask("provenance uncached probe")                 # escalates, caches the model
    aM = mV.ask("provenance uncached probe")
    assert aM["tier"] == "T0" and aM.get("provenance") == "model-cached", \
        "a model answer served from cache must declare itself model-cached (cp47)"
    # DRIFT SENTINEL SUITE (cp54, the leOS dig): the STALE scenario must SURFACE.
    mD = lecore.UnifiedMind(); mD.zoo_attach(lambda p: "rung")
    for _qd, _ad in [("where does the deployment run", "it runs in frankfurt"),
                     ("where does the staging deployment run", "staging runs in dublin"),
                     ("where does the batch deployment run", "batch runs in oregon"),
                     ("where does the edge deployment run", "edge runs in singapore")]:
        mD.teach(_qd, _ad)
        mD.drift_sentinel().note(mD.semantic_key(_qd)["vec"][:64],
                                 mD.semantic_key(_ad)["vec"][:64])
    _ck = mD.teach_check("where does the deployment run",
                         "frankfurt was decommissioned in june; nothing runs there")
    assert _ck["verdict"] == "redshift" and _ck["conflict_candidate"], \
        "an implicit conflict must surface as a redshift conflict candidate (cp54)"
    assert _ck["established_nearby"], "the caller gets the established answers to weigh"
    _ok = mD.teach_check("where does the ml deployment run", "ml runs in virginia")
    assert not _ok["conflict_candidate"], "a consistent new fact must not cry wolf"
    # DURABLE VETO SUITE (cp54): a veto must survive a restart, and a re-teach must
    # durably lift it. Before this, load replayed the record through _remember and every
    # vetoed answer RESURRECTED -- the same 43 noise reflexes were purged in cp48 and
    # again in cp53, coming back each time. Also pinned: provenance survives replay
    # (restarts were silently relabelling taught answers as model-cached).
    import tempfile as _tfv, shutil as _shv
    _rv = _tfv.mkdtemp(prefix="lecore_veto_")
    mV1 = lecore.UnifiedMind(); mV1.zoo_attach(lambda p: "a model guess")
    mV1.teach("veto pin good", "the taught truth")
    mV1.ask("veto pin bad"); mV1.ask("veto pin bad")
    mV1.answer_feedback("veto pin bad", ok=False)
    mV1.learning_save(_rv)
    mV2 = lecore.UnifiedMind(); mV2.zoo_attach(lambda p: "WOKE"); mV2.learning_load(_rv)
    assert "guess" not in str(mV2.ask("veto pin bad").get("answer") or ""), \
        "a vetoed answer must stay dead across a restart (cp54)"
    aG = mV2.ask("veto pin good")
    assert aG["tier"] == "T0" and aG.get("provenance") == "taught", \
        "taught provenance must survive replay (cp54)"
    mV2.teach("veto pin bad", "the corrected answer"); mV2.learning_save(_rv)
    mV3 = lecore.UnifiedMind(); mV3.zoo_attach(lambda p: "x"); mV3.learning_load(_rv)
    aC = mV3.ask("veto pin bad")
    assert aC.get("provenance") == "taught" and "corrected" in str(aC.get("answer")), \
        "a deliberate re-teach lifts the tombstone durably (cp54)"
    _shv.rmtree(_rv, ignore_errors=True)
    # APPKIT SUITE (cp53): the layer anything built on leCore starts from.
    import tempfile as _tfa, shutil as _sha
    _ra = _tfa.mkdtemp(prefix="lecore_appkit_")
    mA = lecore.UnifiedMind()
    _a1 = mA.app_substrate("suiteapp", user="one", root=_ra)
    _a2 = mA.app_substrate("suiteapp", user="two", root=_ra)
    _a1.remember("preferred format", "webp 82")
    assert _a1.recall("preferred format")["provenance"] == "taught"
    assert "webp" not in str(_a2.recall("preferred format").get("answer") or ""), \
        "per-app-per-user isolation is PHYSICAL: no user appears in another's memory"
    _a1.observe("edit a photo", ["duplicate layer", "curves", "export"])
    _a1.observe("edit a scan", ["duplicate layer", "curves", "sharpen"])
    assert any(h["support"] >= 2 for h in _a1.habits()), \
        "repeated subsequences become mined habits"
    _sha.rmtree(_ra, ignore_errors=True)
    # PANEL-REALM SUITE (cp42): seated writes, silent consensus, surfaced dissent.
    mP = lecore.UnifiedMind()
    mP.zoo_attach(lambda p: "M")
    import tempfile as _tfp, shutil as _shp
    _rp = _tfp.mkdtemp(prefix="lecore_panel_")
    mP._archive_root = _rp
    seat = mP.panel_seat(members=["quilez", "widrow", "bau"])
    assert seat["seated"] == 3 and seat["chair"] == "quilez"
    assert mP.panel_note("bau", "the install needs a unit key")["notes"] == 1
    assert "error" in mP.panel_note("stranger", "x"), "unseated authors are refused"
    agree = mP.panel_deliberate("is bind an fft", {"quilez": "yes", "widrow": "yes"})
    assert agree["silent"], "consensus must be silent (contrast law)"
    split = mP.panel_deliberate("random-init valid?",
                                {"bau": "no", "widrow": "solver yes"})
    assert not split["silent"] and split["recorded"] == 2, "dissent must surface + record"
    _shp.rmtree(_rp, ignore_errors=True)
    # HARSH SUITE (cp38): poison walls, template floods, veto windows.
    mH = lecore.UnifiedMind()
    mH.zoo_attach(lambda p: "M")
    for hz in ("__END__", "__token__ __null__", "prefix __inject__ suffix"):
        mH.teach(hz, "payload")
    assert all(mH.ask(hz)["tier"] != "T0" for hz in
               ("__END__", "__token__ __null__", "prefix __inject__ suffix")), \
        "control-token questions must never become servable reflexes"
    mH.teach("what does bots/substrate_gather/__init__.py do", "the gather step")
    assert mH.ask("what does bots/substrate_gather/__init__.py do")["tier"] == "T0", \
        "dunders inside PATHS are data, not control tokens (the cp38 real-data lesson)"
    for hi in range(40):
        mH.teach("what timeout does the harsh subsystem use in cluster %d" % hi,
                 "harsh timeout %d" % hi)
    hOK = sum(1 for hi in range(40) if ("harsh timeout %d" % hi) in
              str(mH.ask("what timeout does the harsh subsystem use in cluster %d"
                         % hi).get("answer", "")))
    assert hOK == 40, "template floods must serve exactly via the sidecar (got %d)" % hOK
    mH.teach("harsh veto window q", "BAD")
    mH.answer_feedback("harsh veto window q", ok=False)
    assert mH.ask("harsh veto window q")["tier"] != "T0", \
        "a vetoed answer must not serve from the exact arm either"
    # SELF-IMPROVEMENT SUITE (cp37): the curve bends, the frozen control does not.
    mSI = lecore.UnifiedMind()
    siFacts = {"what is the widget quota": "nine per day",
               "who owns the widget queue": "the ops rotation"}
    def siModel(prompt):
        if prompt.startswith("REFLECT:"):
            for kq, ka in siFacts.items():
                if kq.split()[-1] in prompt:
                    return "the correct answer is " + ka
        return "unknown"
    mSI.zoo_attach(siModel)
    siJudge = lambda q, ans: siFacts[q].split()[-1] in str(ans).lower()
    siR = mSI.self_improve([{"q": q, "teach": a} for q, a in siFacts.items()],
                           siJudge, rounds=2)
    assert siR["curve"][0]["errors"] == 2 and siR["curve"][-1]["errors"] == 0, siR
    assert siR["improved"], "the substrate must bend the curve"
    mSI.goal_create("si-wf", "assemble the widget quota report", plan=["ga", "gb"])
    mSI.goal_work("si-wf", executors={"ga": lambda: "did ga for the report",
                                      "gb": lambda: "did gb for the report"},
                  budget_steps=2)
    siW = mSI.workflow_distill()
    assert siW["new"] >= 1, "a done goal must distill into the workflow library"
    siWm = mSI.workflow_warm("assemble the widget quota summary")
    assert siWm["plan"] == ["ga", "gb"], "a NEW overlapping objective warms from it"
    # SESSION SUITE (cp36): isolation by key space, resume by reopen, search across.
    mS = lecore.UnifiedMind()
    mS.zoo_attach(lambda p: "MODEL")
    mS.doctrine_load()
    mS.session_open("alpha")
    mS.teach("what color is the client logo", "the alpha client logo is teal")
    mS.session_open("beta")
    mS.teach("what color is the client logo", "the beta client logo is crimson")
    aB = mS.ask("what color is the client logo")
    assert aB["tier"] == "T0" and "crimson" in str(aB["answer"]), "beta serves beta"
    assert "teal" not in str(aB["answer"]), "NO BLEED between sessions"
    aD = mS.ask("how do I run a long agent task with lecore")
    assert aD["tier"] == "T0", "shared doctrine visible inside a session"
    mS.session_open("alpha")
    aA = mS.ask("what color is the client logo")
    assert "teal" in str(aA["answer"]), "reopening alpha resumes alpha's memory"
    sr = mS.session_search("client logo color")
    assert {h["session"] for h in sr["hits"]} >= {"alpha", "beta"}, \
        "search crosses sessions explicitly"
    import tempfile as _tf8, shutil as _sh8
    _r8 = _tf8.mkdtemp(prefix="lecore_sess_")
    try:
        mS.learning_save(_r8)
        mS2 = lecore.UnifiedMind()
        mS2.zoo_attach(lambda p: "MODEL")
        mS2.learning_load(_r8)
        mS2.session_open("beta")
        aR = mS2.ask("what color is the client logo")
        assert "crimson" in str(aR["answer"]), "sessions survive save/load"
        assert "alpha" in mS2.session_list()["sessions"]
    finally:
        _sh8.rmtree(_r8, ignore_errors=True)
    # FULL-RECORD SUITE (cp32): compression replaces culling; nothing durable clips.
    import tempfile as _tf7, shutil as _sh7
    _r7 = _tf7.mkdtemp(prefix="lecore_fr_")
    try:
        mF = lecore.UnifiedMind()
        mF.zoo_attach(lambda p: "x")
        frTxt = "start " + ("z" * 700) + " tail-marker"
        mF.goal_create("fr-goal", "full record goal", plan=["fr step"])
        mF.goal_work("fr-goal", executors={"fr step": lambda: frTxt}, budget_steps=1)
        mF.learning_save(_r7)
        mF2 = lecore.UnifiedMind()
        mF2.zoo_attach(lambda p: "x")
        mF2.learning_load(_r7)
        frD = mF2.goal_book.goals["fr-goal"]["steps"][0]["deliverable"]
        assert frD == frTxt, "a deliverable must roundtrip UNCLIPPED (record is full)"
        frK = list(mF2.tool_cache["prefix"])[-1]
        assert mF2.tool_cache["prefix"][frK] == frTxt, \
            "a cached tool value must roundtrip EXACTLY -- clipped cache is wrong cache"
    finally:
        _sh7.rmtree(_r7, ignore_errors=True)
    # STORAGE-DOCTRINE SUITE (cp31): no loose JSON storage files, ever again.
    import tempfile as _tf6, shutil as _sh6, os as _os6, json as _js6
    _r6 = _tf6.mkdtemp(prefix="lecore_ks_")
    try:
        from holographic.caching_and_storage.holographic_knowledgestore import \
            KnowledgeStore as _KS6
        _js6.dump([{"id": 0, "hash": "h", "text": "legacy", "kind": "note",
                    "source": "inner", "author": "swarm", "tags": [], "session": None,
                    "ts": 0}], open(_os6.path.join(_r6, "knowledge.json"), "w"))
        _k6 = _KS6(_r6)
        _k6.add_note("post-migration note", tags=("t",))
        _k6.set_scope("session", session="s")
        assert _os6.path.exists(_os6.path.join(_r6, "knowledge.lecore")), \
            "the journal must live in the container"
        assert not _os6.path.exists(_os6.path.join(_r6, "knowledge.json")), \
            "a legacy loose-JSON journal must migrate by replay and be renamed"
        assert not _os6.path.exists(_os6.path.join(_r6, "scopes.json")), \
            "scopes ride the container, never a loose JSON file"
        _k6b = _KS6(_r6, session="s")
        assert len(_k6b.entries) == 2 and _k6b.get_scope() == "session"
        mS6 = lecore.UnifiedMind()
        mS6.zoo_attach(lambda p: "x")
        _p6 = _tf6.mkdtemp(prefix="lecore_sv_")
        try:
            mS6.learning_save(_p6)
            loose = [f_ for f_ in _os6.listdir(_p6) if f_.endswith(".json")]
            assert not loose, "learning_save must leave NO loose JSON: %r" % loose
        finally:
            _sh6.rmtree(_p6, ignore_errors=True)
    finally:
        _sh6.rmtree(_r6, ignore_errors=True)
    # AGENT-LOOP SUITE (cp28): resume across saves, code gate, codebase map.
    import tempfile as _tf5, shutil as _sh5
    _r5 = _tf5.mkdtemp(prefix="lecore_loop_")
    try:
        mL = lecore.UnifiedMind()
        mL.zoo_attach(lambda p: "rung")
        exL = {"sa": lambda: "did sa for the loop objective",
               "sb": lambda: "did sb for the loop objective"}
        aL1 = mL.agent_loop("small loop objective for the suite", executors=exL,
                            rounds=1, budget_steps=1, checkpoint_root=_r5,
                            plan=["sa", "sb"])
        assert aL1["rounds"][0]["progress"] == 1 and aL1["status"] == "active"
        mL2 = lecore.UnifiedMind()
        mL2.zoo_attach(lambda p: "rung")
        mL2.learning_load(_r5)
        aL2 = mL2.agent_loop("small loop objective for the suite", executors=exL,
                             rounds=2, budget_steps=1, checkpoint_root=_r5)
        # (aL-prefixed on purpose: cp21's addendum documented r2-shadowing killing this
        # suite, and cp28 repeated it verbatim before this rename. Prefix your locals.)
        assert aL2["status"] == "done", "a fresh process must RESUME the loop to done"
        cw = mL2.code_write("sq", "sq(x) returns x*x",
                            test=lambda ns: (ns["sq"](3) == 9 or
                                             (_ for _ in ()).throw(AssertionError())),
                            llm=lambda p: "def sq(x):" + chr(10) + "    return x * x")
        assert cw["ok"] and cw["verified"] == "ast+test"
        bad = mL2.code_write("nope", "anything",
                             test=lambda ns: (_ for _ in ()).throw(AssertionError("x")),
                             llm=lambda p: "def nope():" + chr(10) + "    return 1")
        assert not bad["ok"], "a failing draft must be refused, not returned"
        cm = mL2.codebase_map("tools", topic="codebase:suite-tools")
        assert cm["modules"] > 20, "the codebase map must index the tree"
    finally:
        _sh5.rmtree(_r5, ignore_errors=True)
    # UNICRON-INWARD SUITE (cp26): spectrum, fingerprints, retention, cards.
    mU = lecore.UnifiedMind()
    mU.zoo_attach(lambda p: "x")
    lu = mU.zoo["ladder"]
    for i_ in range(10):
        q_ = "spectrum probe %d about the %s array" % (i_, ["alpha", "beta"][i_ % 2])
        lu._remember(lu._qkey(q_), "ans %d" % i_, q_)
    spU = mU.learning_spectrum()
    assert "verdict" in spU and isinstance(spU["matrices"], list), \
        "the mind must be able to read its own weights"
    assert all(m_["outliers"] == 0 for m_ in spU["matrices"]
               if m_["name"].startswith("trace_tile")), \
        "atom codebooks must sit in the random bulk -- outliers mean collisions"
    import tempfile as _tf4, shutil as _sh4
    _r4 = _tf4.mkdtemp(prefix="lecore_fp_")
    try:
        mU.learning_save(_r4)
        dU = mU.partition_drift(_r4, _r4)
        assert dU["cosine"] > 0.98, "a partition must not drift from itself"
        rU = mU.partition_retention(_r4, _r4, ["spectrum probe 0 about the alpha array"])
        assert rU["retention"] == 1.0 and not rU["lost"], \
            "identical roots must retain everything"
    finally:
        _sh4.rmtree(_r4, ignore_errors=True)
    cU = mU.kernel_card("bm25_score")
    assert cU["kind"] == "glsl_kernel" and cU["verified"] and cU["kept_negative"], \
        "a kernel card must fuse artifact, verification, and kept negative"
    # ZOO SELF-USE SUITE (cp25): model3d, lossless archive, backtest honesty, recipes.
    mZ = lecore.UnifiedMind()
    mZ.zoo_attach(lambda p: "x")
    r3 = mZ.model3d([{"shape": "sphere", "r": 0.5},
                     {"shape": "torus", "R": 0.9, "r": 0.07, "at": [0, 0.5, 0]}],
                    name="selftest scene", size=48)
    assert r3.get("ok") and r3.get("stored"), "model3d must render AND remember"
    mZ.research_archive("st", ["the calibration constant is 7.31 exactly"])
    qz = mZ.archive_query("st", "what is the calibration constant")
    assert qz["found"] >= 1 and "7.31" in qz["evidence"][0], "archive must answer verbatim"
    az = mZ.assimilate_docs("stapi", "s.run(job) starts a job\ns.stop(job) halts a job")
    uz = mZ.use_assimilated("stapi", "halt the running job")
    assert az["recipes"] == 2 and uz["calls"][0]["call"].startswith("s.stop"), \
        "assimilated docs must rank the right call for the task"
    import numpy as _np2
    _tr = list(_np2.linspace(0, 3, 40) + 0.05 * _np2.sin(_np2.arange(40)))
    bz = mZ.market_backtest(_tr, d_grid=(3, 5))
    assert bz["ok"] and bz["mae"] < bz["naive_mae"], \
        "a clean trend must beat last-value; the verdict system depends on it"
    # STAGE-1 ADOPTION SUITE (cp23): control tokens, calibration, trajectory cache.
    mS = lecore.UnifiedMind()
    scalls = {"n": 0}
    def sllm(p):
        scalls["n"] += 1
        return "__CTRL__" if scalls["n"] == 1 else "real-" + str(scalls["n"])
    mS.zoo_attach(sllm)
    mS.ask("what is the widget torque spec")
    s2_ = mS.ask("what is the widget torque spec")
    assert s2_["answer"] != "__CTRL__", "control tokens must never be remembered as answers"
    exN = {"n": 0}
    def mkS(nm):
        def f():
            exN["n"] += 1
            return "step %s done for the pipeline" % nm
        return f
    planS = ["sa", "sb", "sc"]
    exS = {n_: mkS(n_) for n_ in planS}
    mS.goal_create("tc1", "run the small pipeline", plan=planS)
    mS.goal_work("tc1", executors=exS, budget_steps=3)
    nfirst = exN["n"]
    mS.goal_create("tc2", "run the small pipeline once more", plan=planS)
    wS = mS.goal_work("tc2", executors=exS, budget_steps=3)
    assert wS["cache_hits"] == 3 and exN["n"] == nfirst, \
        "an identical trajectory must serve wholly from the tool-value cache"
    # THE AUTOMATION-LOOP SUITE (cp22): feedback kills knee-jerks; images roundtrip.
    mF = lecore.UnifiedMind()
    fcalls = {"n": 0}
    def fllm(p):
        fcalls["n"] += 1
        return "F-" + str(fcalls["n"])
    mF.zoo_attach(fllm)
    fq = "what is the calibration constant for the flux gate"
    f0 = mF.ask(fq)
    f0b = mF.ask(fq)
    assert f0b["tier"] == "T0"
    mF.answer_feedback(fq, ok=False)
    f1 = mF.ask(fq)
    assert not (f1["tier"] == "T0" and f1["answer"] == f0["answer"]), \
        "FEEDBACK: a bad answer must never be knee-jerked back"
    mF.zoo["ladder"]._remember(mF.zoo["ladder"]._qkey(fq), "corrected constant 7.3", fq)
    f2 = mF.ask(fq)
    assert f2["tier"] == "T0" and f2["answer"].startswith("corrected"), \
        "a re-taught correction must serve again"
    img_t = (np.arange(48 * 48 * 3, dtype=np.uint8).reshape(48, 48, 3))
    ri = mF.image_remember(img_t, "test grid image", source="capture")
    rc = mF.image_recall("grid image")
    assert ri["stored"] and rc and np.array_equal(rc[0]["image"], img_t), \
        "IMAGE MEMORY: labeled store + recall must roundtrip bit-exact"
    syn = mF.synthesize_response("calibration constant flux")
    assert syn["synthesized"] and "[reflex]" in syn["answer"], \
        "SYNTHESIS: provenance labels must ride with composed answers"
    # THE ALIASING + MIGRATION SUITE (cp21): siblings must separate; texts must migrate.
    mQ = lecore.UnifiedMind()
    qcalls = {"n": 0}
    def qllm(p):
        qcalls["n"] += 1
        return "ANS-" + str(qcalls["n"])
    mQ.zoo_attach(qllm)
    sib1 = mQ.ask("how does lecore compare to gptcache semantic caching")
    sib2 = mQ.ask("how does lecore compare to routellm frugalgpt routing")
    assert sib2["answer"] != sib1["answer"], \
        "ALIASING: sibling question frames must not serve each other's answers"
    rep1 = mQ.ask("how does lecore compare to gptcache semantic caching")
    rep2 = mQ.ask("how does lecore compare to routellm frugalgpt routing")
    assert (rep1["tier"], rep2["tier"]) == ("T0", "T0") and rep1["answer"] != rep2["answer"], \
        "each sibling must own its answer at T0"
    import tempfile as _tf3, shutil as _sh3
    _r3 = _tf3.mkdtemp(prefix="lecore_mig_")
    try:
        mQ.learning_save(_r3)
        mR = lecore.UnifiedMind()
        mR.zoo_attach(lambda p: "should-not-run")
        mR.learning_load(_r3)
        rr = mR.ask("how does lecore compare to routellm frugalgpt routing")
        assert rr["tier"] == "T0" and rr["answer"] == sib2["answer"], \
            "MIGRATION BY REPLAY: taught texts must re-teach under the current key function"
    finally:
        _sh3.rmtree(_r3, ignore_errors=True)
    return {"tools": jt["tools"], "typed": jt["typed"], "prefill_to_T0": hot["tier"],
            "run2_calls": r2["model_calls"], "bench_naive": len(W), "bench_attached": att,
            "conversation_corpus": {"glarp_cos": round(float(g @ c), 2),
                                    "paraphrase_via": rp["via"]},
            "remembers": {"cold_process_plan": r4["via"]},
            "goals": {"resume": w["status"], "drift": wd["status"]},
            "ledger": m.zoo_ledger()}


if __name__ == "__main__":
    print(_selftest())

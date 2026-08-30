"""holographic_unified_p20_zoo.py -- Part 20: THE TIERED ZOO as mind faculties (backlog v3).
The answer ladder + ledger, the typed/learned library, learned CoT, scoped orchestration, and
pre-emptive research prefill. Machinery in holographic_zoo; the leOS law it enforces: the model
only runs when there is THINKING to do."""
import numpy as np



def _q8_pack(a):
    """float32 (n, d) -> (uint8 codes, lo, hi) with a PER-ROW range.

    THE SAME SCHEME lecore_data/routing/index_128d.npz already ships, factored out
    the third time it was needed: the routing index, then learning.semantic's ctx,
    now learning.experience's audit arrays. Per-ROW lo/hi rather than a global
    range is what keeps the error small when row magnitudes differ by orders --
    measured cosine min 0.99995 on both the semantic and the audit arrays."""
    a = np.asarray(a, np.float64)
    if a.size == 0:
        return (np.zeros(a.shape, np.uint8), np.zeros((len(a), 1)),
                np.zeros((len(a), 1)))
    lo = a.min(1, keepdims=True)
    hi = a.max(1, keepdims=True)
    q = np.round((a - lo) / np.maximum(hi - lo, 1e-12) * 255.0).astype(np.uint8)
    return q, lo, hi


def _q8_unpack(q, lo, hi):
    """Inverse of _q8_pack. Reads float32 straight through when handed one, so a
    partition written before the change loads unchanged -- additive, not a flip."""
    q = np.asarray(q)
    if q.dtype != np.uint8:
        return np.asarray(q, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    return lo + q.astype(np.float64) * (hi - lo) / 255.0


class _UnifiedPart20:

    @property
    def goal_book(self):
        """The mind's DURABLE GOAL BOOK (lazy; persisted in the learning partition)."""
        if "goals" not in self.zoo:
            from holographic.agents_and_reasoning.holographic_zoo import GoalBook
            self.zoo["goals"] = GoalBook()
        return self.zoo["goals"]

    def model_atlas(self):
        """THE MAP THAT GROWS WITH USE (cp84): consolidates everything the
        engine learns ABOUT an attached model while it runs -- attribution
        addresses (which layer each answer crystallized at, from
        model_attribute / the runtime rung's per-answer lens), the decision
        journal (which arm answered: memory / grounded / semantic / model), and
        the earned speed tiers (cues whose early crystallization + agreement
        licensed truncated schedules). The atlas is why the system gets FASTER
        and MORE GROUNDED as it is used: every model answer mints an address;
        every taught or vetoed correction moves that cue off the model arm
        entirely; every early address extends truncation coverage. Returns
        {addresses, arm_counts, truncation_eligible, coverage}."""
        lad = self.zoo["ladder"]
        addr = []
        for t in lad.taught_log:
            if len(t) > 3 and "model:" in str(t[1])[:40]:
                addr.append({"cue": str(t[0])[:80],
                             "address": str(t[1])[:48]})
        arms = {}
        for rec in getattr(self, "_decision_journal", []) or []:
            k = rec.get("arm", "?")
            arms[k] = arms.get(k, 0) + 1
        elig = [a for a in addr if "/L" in a["address"] and
                self._atlas_layer(a["address"]) is not None and
                self._atlas_layer(a["address"]) <= 0.8 * max(
                    1, self._atlas_layers_total(a["address"]))]
        return {"addresses": addr, "n_addresses": len(addr),
                "arm_counts": arms,
                "truncation_eligible": len(elig),
                "note": ("every address is a place in the model the engine can "
                         "now exit early from; every teach/veto retires a cue "
                         "from the model arm entirely")}

    @staticmethod

    def _atlas_layer(address):
        try:
            return int(address.split("/L")[1].split("/")[0])
        except Exception:
            return None

    @staticmethod

    def _atlas_layers_total(address):
        try:
            return int(address.split("model:")[1].split("L")[0])
        except Exception:
            return 1


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
                if not _os.path.exists(st):
                    # rolled partitions carry state-<ts>.lecore instead of the legacy
                    # name; the shared resolver keeps this listing honest for both
                    st = self._learning_state_path(_os.path.join(root, name)) or st
                if _os.path.exists(st):
                    out.append({"name": name,
                                "path": _os.path.join(root, name),
                                "bytes": _os.path.getsize(st)})
        return {"root": root, "memories": out}

    def contribute(self, dest, author=None):
        """OPT-IN contribution to the openzoo COMMONS (sweep 100): screen this mind's
        SHARED taught rows through a conservative privacy gate and export the survivors
        as a commons bundle, provenance 'commons:<author-or-anon>'. THE RULES, each a
        refusal: (1) session-salted rows never leave -- session isolation IS user
        privacy by construction, only session=='shared' rows are candidates; (2)
        path-shaped text (/, \\, ~/) is rejected -- no file directories travel; (3)
        email shapes, long digit runs (phone/account), and long hex/base64 runs
        (keys/tokens) are rejected; (4) model-cached rows are rejected -- the commons
        takes established knowledge, not unverified cache. Returns the REVIEW SHEET
        {kept, rejected:[(question, reason)]} -- consent is informed or it is not
        consent. KEPT NEG, loud: a lexical screen is a FLOOR, not a proof of
        anonymity; the review sheet is the real gate and the caller reads it before
        shipping the bundle. Opt-out: never call this, or set mind._commons_optout=True
        and even an accidental call refuses."""
        import os, re
        if getattr(self, "_commons_optout", False):
            return {"refused": "this mind is opted out of the commons"}
        lad = self.zoo["ladder"]
        PATH = re.compile(r"(~/|[A-Za-z]:\\|/[\w.-]+/)")
        MAIL = re.compile(r"\S+@\S+\.\S+")
        DIGITS = re.compile(r"[\d][\d\s().-]{6,}[\d]")
        SECRET = re.compile(r"[A-Fa-f0-9]{24,}|[A-Za-z0-9+/=]{32,}")
        kept, rejected, seen = [], [], set()
        for row in reversed(getattr(lad, "taught_log", []) or []):
            q, a = str(row[0]), str(row[1])
            sess = str(row[2]) if len(row) > 2 else "shared"
            prov = str(row[3]) if len(row) > 3 else "taught"
            if q in seen:
                continue
            seen.add(q)
            if sess != "shared":
                rejected.append((q[:60], "session-salted: user-private by construction"))
                continue
            if prov == "model-cached":
                rejected.append((q[:60], "model-cached: unverified"))
                continue
            if a == "carried tombstone":
                continue
            blob = q + " " + a
            if PATH.search(blob):
                rejected.append((q[:60], "path-shaped text"))
                continue
            if MAIL.search(blob):
                rejected.append((q[:60], "email shape"))
                continue
            if DIGITS.search(blob):
                rejected.append((q[:60], "long digit run (phone/account shape)"))
                continue
            if SECRET.search(blob):
                rejected.append((q[:60], "hex/base64 run (key/token shape)"))
                continue
            kept.append((q, a, prov))
        who = str(author) if author else "anon"
        dst = self.__class__(dim=self.dim, seed=0)
        for q, a, prov in reversed(kept):
            dst.teach(q, a)
            _bl = getattr(dst.zoo["ladder"], "taught_log", [])
            if _bl and str(_bl[-1][0]) == q and len(_bl[-1]) > 3:
                keepprov = prov if prov.startswith(("wisdom:", "commons:")) else "commons:%s" % who
                _bl[-1] = [_bl[-1][0], _bl[-1][1], _bl[-1][2], keepprov]
        # lever 3 (sweep 101): a contribution bundle is pure-taught BY CONSTRUCTION
        # (built row-by-row through teach), so the regen guard always passes and the
        # bundle ships as text -- the middle-out curve for the commons itself.
        dst.learning_save(str(dest), audit="regen")
        return {"kept": len(kept), "rejected": rejected, "dest": str(dest),
                "author": who,
                "advice": "read the rejected list AND spot-check the kept rows before "
                          "shipping -- the lexical screen is a floor, not a proof"}

    def commons_pool(self, bundles, root):
        """POOL many contribution bundles into ONE commons partition (sweep 100): a
        fresh mind imports each bundle through the provenance-carrying cp69 pipe
        (conflicts FLAGGED, never silently resolved -- disagreement between users is
        signal), then saves to root. Any contributing mind imports the commons back
        with memory_import(root) -- the give-and-take Moose described: all who
        contribute may draw. Returns per-bundle counts and the flagged conflicts."""
        pool = self.__class__(dim=self.dim, seed=0)
        report = []
        for b in bundles:
            r = pool.memory_import(str(b), on_conflict="flag")
            report.append({"bundle": str(b), "imported": r.get("imported"),
                           "conflicts": r.get("conflicts")})
        # lever 3 (sweep 101): the pooled commons is likewise pure-taught.
        sv = pool.learning_save(str(root), audit="regen")
        return {"bundles": report, "root": str(root),
                "rows": len(getattr(pool.zoo["ladder"], "taught_log", []) or []),
                "saved": sv.get("saved")}

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
            # symmetric with the sweep-99 import fix: teach() writes provenance
            # 'taught'; a bundle that promises 'facts travel with their provenance'
            # must actually carry it, or authorship (wisdom:<model>) dies in transit.
            _bl = getattr(dst.zoo["ladder"], "taught_log", [])
            if prov and prov != "taught" and _bl and str(_bl[-1][0]) == q and len(_bl[-1]) > 3:
                _bl[-1] = [_bl[-1][0], _bl[-1][1], _bl[-1][2], prov]
            if prov in ("validated", "evidenced"):
                dst.conjecture_record(q, a)
                dst.conjecture_promote(q, prov, "carried by memory_export")
        vet = sorted(getattr(lad, "_vetoed_qs", set()) or [])
        for vq in vet:
            dst.teach(vq, "carried tombstone")
            dst.answer_feedback(vq, ok=False)
        _os.makedirs(dest, exist_ok=True)
        # lever 3 (sweep 101): export bundles are pure-taught scratch minds.
        dst.learning_save(dest, audit="regen")
        chk = type(self)()
        chk.zoo_attach(lambda p: "")
        chk.learning_load(dest)
        def _row_ok(q_, a_, p_):
            got = str(chk.ask(q_).get("answer") or "")
            if p_ in ("validated", "evidenced", "conjecture"):
                return bool(got.strip())        # promote rebuilds canonical
                                                # serve-text; require presence
            return got == a_                    # plain taught: exact
        # NAME THE MISSES, DO NOT JUST COUNT THEM. This reported misses=1 out of
        # 497 and nothing else, so "verified: False" told the caller their bundle
        # was imperfect and gave them no way to act -- I had to reload the bundle
        # and diff 497 rows by hand to learn WHICH question did not survive.
        # A verification that cannot name what failed is a smoke alarm with no
        # location: it is right, and useless.
        _missed = [q for q, a, _p in rows if not _row_ok(q, a, _p)]
        misses = len(_missed)
        byp = {}
        for _q, _a, prov in rows:
            byp[prov] = byp.get(prov, 0) + 1
        return {"exported": len(rows), "by_provenance": byp,
                "vetoes": len(vet), "dest": dest,
                "verified": misses == 0, "misses": misses,
                "missed": _missed[:20]}

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
            # AN EMPTY LOCAL ANSWER IS NOT A CONFLICTING ANSWER. `q in mine` was
            # true for questions this memory holds a BLANK for -- an abstention
            # that got written to the taught store -- so incoming knowledge was
            # flagged as a conflict and, under the default on_conflict="flag",
            # SILENTLY NOT IMPORTED.
            # MEASURED on a real bundle: 8 conflicts reported, FOUR of them
            # against questions that answered T4 with answer='' here. Half the
            # shared knowledge was withheld on the grounds of disagreeing with
            # nothing -- which is precisely the failure mode that makes sharing
            # research between memories useless.
            # Treat blank-here as absent: accept theirs, and count it as an
            # import rather than a conflict.
            _mine = str(mine.get(q, "") or "")
            if q in mine and _mine.strip():
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
            # cp69 promised 'facts arrive with provenance intact' but this teach()
            # flattened every row to 'taught' -- measured in sweep 99 when a bequeathed
            # wisdom row imported with its authorship stripped (wisdom() showed no
            # authors while ask() served the lesson). Re-stamp any NON-DEFAULT
            # provenance onto the row teach just appended; plain 'taught' rows are
            # untouched, byte-identical.
            _lg = getattr(self.zoo["ladder"], "taught_log", [])
            if (prov and prov != "taught" and _lg and str(_lg[-1][0]) == q
                    and len(_lg[-1]) > 3):
                _lg[-1] = [_lg[-1][0], _lg[-1][1], _lg[-1][2], prov]
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

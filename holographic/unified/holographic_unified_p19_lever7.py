"""holographic_unified_p19_lever7.py -- Part 19 of the UnifiedMind: THE SEVENTH LEVER as mind
faculties. The displacement trace (superposed experience memory with delta-rule writes and a
cleanup-calibrated gate), the volatility field, warm-started factoring, and the levers() doctrine
entry. Machinery in holographic_lever7; measured evidence in docs (deep-dive Parts 3, 8-13).
"""
import numpy as np


class _UnifiedPart19:

    # -- the trace ------------------------------------------------------------------------------
    @property
    def experience(self):
        """The mind's DISPLACEMENT TRACE (lazy singleton): a superposed experience memory holding
        every accepted (task_key -> response) pair as bind(key, value) in ONE self-tiling vector
        store. Writes are delta-rule (a predicted write is skipped -- the free P-frame), reads are
        gated by cleanup against the response codebook at a calibrated null, priced by the
        capacity-law trust ledger, and suppressed inside the volatility field. Every accepted
        write also lands in an exact audit log; replay is bit-identical (lever 7 stands on
        lever 3). See holographic_lever7.DisplacementTrace / TiledDisplacementTrace."""
        if getattr(self, "_lever7_trace", None) is None:
            from holographic.agents_and_reasoning.holographic_lever7 import TiledDisplacementTrace
            self._lever7_trace = TiledDisplacementTrace(dim=2048, seed=0)
        return self._lever7_trace

    def reflex_write(self, task_vec, response_vec):
        """Record one experience in the displacement trace: task_vec is the SIMILARITY KEY (the
        task CONTEXT -- kept negative from the resonator sweep: never key on the answer/composite,
        binding decorrelates them), response_vec is the move that solved it. Returns
        {accepted, surprise, load, tiles}; a write the trace already predicts is skipped free."""
        return self.experience.write(np.asarray(task_vec, float), np.asarray(response_vec, float))

    def reflex_try(self, task_vec):
        """THE LEVER-7 READ: try to answer a task from accumulated experience WITHOUT the
        expensive path. One unbind, cleanup against the response codebook, then three gates --
        calibrated cosine null (alpha stated), capacity-law trust price, volatility field.
        Returns {fired, prediction, atom, confidence, trust, why, tiles}; refusal is a result.
        Kept negative (measured, deep-dive Part 3): the ungated version of this read served 48
        wrong answers where this gate served 2 -- the gate is not optional. JSON-safe: the
        prediction ships as a plain list so the service/MCP can carry it."""
        out = dict(self.experience.read_gated(np.asarray(task_vec, float)))
        for k in ("prediction", "raw"):
            if out.get(k) is not None and hasattr(out[k], "tolist"):
                out[k] = out[k].tolist()
        return out

    def reflex_outcome(self, task_vec, success):
        """Close the reflex loop: report whether a served answer worked. Failures accumulate in a
        holographic FAILURE FIELD that reflex_try checks with one cosine -- the outcome gate that
        similarity + calibration alone cannot replace (measured: look-alike traps pass the
        calibrated null; the outcome field catches them)."""
        t = np.asarray(task_vec, float)
        return self.experience.tiles[self.experience._route(t)].record_outcome(t, bool(success))

    def reflex_mark_volatile(self, tag, vec=None):
        """Mark a region of task space VOLATILE (prices, live status, anything the world moves):
        reflex_try will never fire there until reflex_unmark_volatile(tag). Volatility is a FIELD
        (one cosine to check), not a pattern list; unmarking is exact subtraction."""
        for t in self.experience.tiles:
            t.volatility.mark(tag, vec)
        return tag

    def reflex_unmark_volatile(self, tag):
        """Exactly remove a volatility mark (ablation is exact in this algebra)."""
        return any([t.volatility.unmark(tag) for t in self.experience.tiles])

    def reflex_stats(self):
        """Telemetry for the displacement trace: writes, P-frame skips, fires, refusals by
        reason, tiles and splits -- the lever's own honesty ledger."""
        return dict(self.experience.stats)

    def experience_save(self, path):
        """Persist the trace's EXACT FLOOR (the audit logs + volatility marks) to a JSON file;
        experience_load rebuilds the trace bit-identically in any process under any
        PYTHONHASHSEED. The floor is what makes the log an address into computation, not a
        souvenir of it."""
        import json
        state = {"tiles": [t.to_state() for t in self.experience.tiles]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f)
        return path

    def experience_from_state(self, state):
        """Rebuild the displacement trace from a STATE DICT -- the shared core of
        experience_load and the container loader (one rebuild, two transports)."""
        from holographic.agents_and_reasoning.holographic_lever7 import (
            DisplacementTrace, TiledDisplacementTrace)
        tt = TiledDisplacementTrace(dim=2048, seed=0)
        tt.tiles = [DisplacementTrace.from_state(st) for st in state["tiles"]]
        tt._centroids = []
        tt._counts = []
        for t in tt.tiles:
            ks = np.asarray([k for k, _ in t._audit], float)
            tt._centroids.append(ks.mean(axis=0) if len(ks) else np.zeros(t.dim))
            tt._counts.append(len(ks))
        self._lever7_trace = tt
        return {"tiles": len(tt.tiles), "writes": sum(tt._counts)}

    def experience_load(self, path):
        """Rebuild the displacement trace from experience_save() output (bit-identical
        replay). KEPT NEGATIVE (pass 6): this def existed TWICE in this file -- a prior patch
        appended instead of replacing, and Python silently kept the later one; invisible
        until a refactor needed a single anchor."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        return self.experience_from_state(state)
    @property
    def tool_usage(self):
        """The mind's TOOL USAGE TRACE (lazy singleton): successful (task -> tool) pairs held in
        superposition; predicting tools for a task is one unbind + cleanup over the tool
        codebook. See holographic_lever7.UsageTrace."""
        if getattr(self, "_lever7_usage", None) is None:
            from holographic.agents_and_reasoning.holographic_lever7 import UsageTrace
            self._lever7_usage = UsageTrace(dim=2048, seed=0)
        return self._lever7_usage

    def tool_note(self, task_vec, tool, success=True):
        """Record a tool use against its task (only successes strengthen the trace; counts are
        kept beside it for audit). The learned prefilter for tool selection -- leOS's
        tool_selection_memory as one bind instead of a JSONL scan."""
        return self.tool_usage.note(task_vec, tool, success)

    def tool_predict(self, task_vec, k=3):
        """Rank known tools for a task from the usage trace: one unbind + cleanup. Empty trace
        returns [] -- refusal is a result, and the caller falls through to the full registry."""
        return self.tool_usage.predict(task_vec, k)

    def orient(self, topic=None):
        """THE FRONT DOOR (sweep 109 -- the anti-hand-roll compass): one screen that
        gives ANY model full capability access at its fingertips, better than a static
        skill file because it is generated LIVE from the catalog and the partition.
        Returns the agentic workflow (the five moves), live counts, and -- when topic=
        is given -- the top capability pointers for that topic so the model is DIRECTED
        to an existing door instead of hand-rolling. The workflow IS the contract:
        (1) ASK FIRST: serve(query) -- memory or a learned tool may already answer;
        (2) FIND, don't build: find_capability('your goal in your words');
        (3) READ the skill card: describe_skill(name) for the real signature;
        (4) DO: call the method / api_use / lecore_invoke over MCP;
        (5) CLOSE THE LOOP: teach() what you learned, answer_feedback() outcomes,
        bequeath() lessons worth outliving you. Rule 0 in one line: a capability
        find_capability cannot surface does not exist -- and one YOU hand-roll
        without asking first is a gap you just dug."""
        cat = self._capability_catalog()
        out = {
            "workflow": ["serve(query) -- ask before anything; escalation is honest",
                         "find_capability('goal in your own words') -- never hand-roll unasked",
                         "describe_skill(name) -- the real signature and contract",
                         "do it: the method itself, api_use, or lecore_invoke over MCP",
                         "close the loop: teach / answer_feedback / bequeath"],
            "rule_0": "a capability find_capability cannot surface does not exist",
            "counts": {"capabilities": len(cat.all()),
                       "taught_rows": len(getattr(self.zoo["ladder"], "taught_log", []) or []),
                       "wisdom_authors": self.wisdom().get("authors", []),
                       "learned_apis": sorted(getattr(self.api_toolbox(), "services", {}))},
        }
        if topic:
            hits = self.find_capability(str(topic))[:3]
            out["directed_to"] = [{"name": h.name,
                                   "does": str(getattr(h, "does", ""))[:140]}
                                  for h in hits]
            out["advice"] = ("use one of directed_to (describe_skill for the "
                             "contract) before writing anything new")
        return out

    def tool_reflex_teach(self, pattern, service, endpoint, params=None,
                          extract_numbers=None):
        """Teach the substrate HOW a tool answers a question shape (sweep 104): pattern
        is an example question; service/endpoint name an api_learn'd (or faculty) tool;
        params are fixed arguments; extract_numbers optionally names params to fill
        from the NUMBERS IN THE QUERY, in order -- a declared, deterministic argument
        rule (no LLM guesses arguments; what cannot be extracted honestly is not
        served). The reflex is stored on the mind AND noted into the usage trace so
        tool_predict ranks it for similar tasks from day one."""
        if not hasattr(self, "_tool_reflexes"):
            self._tool_reflexes = []
        entry = {"pattern": str(pattern), "service": str(service),
                 "endpoint": str(endpoint), "params": dict(params or {}),
                 "extract_numbers": list(extract_numbers or []),
                 "uses": 0, "hits": 0}
        self._tool_reflexes.append(entry)
        # PERSISTENCE FOR FREE (the wisdom-door move): the reflex ALSO lands as a
        # taught row with provenance 'toolreflex' -- it rides every existing rail
        # (save, load, regen, rollover, export/import) and serve() rebuilds the live
        # list lazily from those rows after a restart. No new save/load surgery.
        import json as _json
        r = self.teach("toolreflex: %s" % str(pattern),
                       _json.dumps({"service": str(service), "endpoint": str(endpoint),
                                    "params": dict(params or {}),
                                    "extract_numbers": list(extract_numbers or [])}))
        if r.get("taught"):
            _lg = getattr(self.zoo["ladder"], "taught_log", [])
            if _lg and len(_lg[-1]) > 3:
                _lg[-1] = [_lg[-1][0], _lg[-1][1], _lg[-1][2], "toolreflex"]
        try:
            tv = self.semantic_key(str(pattern))["vec"]
            self.tool_note(tv, "%s.%s" % (service, endpoint), success=True)
        except Exception:
            pass
        return {"taught": True, "reflexes": len(self._tool_reflexes)}

    def serve(self, query, k=3):
        """PREEMPTIVE SERVE (sweep 104 -- the openzoo division of labor, complete): try
        MEMORY first (T0 taught recall); then the USAGE-LEARNED TOOL REFLEX -- when a
        taught tool's pattern shares >= 2 content words with the query, extract the
        declared arguments (numbers in order for extract_numbers params), CALL the tool
        (api_use), and return its live result with provenance 'tool-reflex' and the
        tool named -- NO LLM IN THE LOOP; then ABSTAIN honestly upward ('escalate') so
        the next level (a model, a human, a bigger harness) knows the substrate could
        not serve this alone. Every successful reflex serve strengthens the usage trace
        (tool_note), so routing sharpens WITH USE. KEPT NEG: argument extraction is
        deterministic and declared -- a query whose arguments cannot be extracted is
        escalated, never guessed."""
        import re
        r = self.ask(str(query))
        if r.get("tier") == "T0" and str(r.get("answer") or "").strip():
            return {"served": True, "via": "memory", "tier": "T0",
                    "answer": r.get("answer")}
        if not getattr(self, "_tool_reflexes", None):
            # lazy rebuild from the durable rows (survives restarts on every rail)
            import json as _json
            self._tool_reflexes = []
            for row in getattr(self.zoo["ladder"], "taught_log", []) or []:
                if len(row) > 3 and str(row[3]) == "toolreflex":
                    try:
                        spec = _json.loads(str(row[1]))
                    except ValueError:
                        continue
                    self._tool_reflexes.append(
                        {"pattern": str(row[0])[len("toolreflex: "):],
                         "service": spec["service"], "endpoint": spec["endpoint"],
                         "params": spec.get("params", {}),
                         "extract_numbers": spec.get("extract_numbers", []),
                         "uses": 0, "hits": 0})
        qw = {w for w in re.findall(r"[a-z]{4,}", str(query).lower())}
        best, best_shared = None, 0
        for e in getattr(self, "_tool_reflexes", []) or []:
            pw = {w for w in re.findall(r"[a-z]{4,}", e["pattern"].lower())}
            shared = len(qw & pw)
            if shared > best_shared:
                best, best_shared = e, shared
        if best is not None and best_shared >= 2:
            params = dict(best["params"])
            if best["extract_numbers"]:
                nums = [float(x) if "." in x else int(x)
                        for x in re.findall(r"-?\d+\.?\d*", str(query))]
                if len(nums) < len(best["extract_numbers"]):
                    return {"served": False, "via": "escalate",
                            "reason": "reflex matched %r but the query carries %d of "
                                      "the %d declared numeric arguments -- not "
                                      "guessing" % (best["pattern"][:40], len(nums),
                                                    len(best["extract_numbers"]))}
                for name, val in zip(best["extract_numbers"], nums):
                    params[name] = val
            out = self.api_use(best["service"], best["endpoint"], params=params)
            best["uses"] += 1
            if isinstance(out, dict) and out.get("ok"):
                best["hits"] += 1
                try:
                    tv = self.semantic_key(str(query))["vec"]
                    self.tool_note(tv, "%s.%s" % (best["service"], best["endpoint"]),
                                   success=True)
                except Exception:
                    pass
                return {"served": True, "via": "tool-reflex",
                        "tool": "%s.%s" % (best["service"], best["endpoint"]),
                        "matched_pattern": best["pattern"], "result": out.get("data"),
                        "uses": best["uses"], "hits": best["hits"]}
            return {"served": False, "via": "escalate",
                    "reason": "reflex tool call failed: %s" % str(out)[:120]}
        # ESCALATION LEDGER (sweep 125): every ask the substrate could not serve is
        # recorded so a service swarm can route it to a human and resolve() it back
        # into memory with provenance -- the swarm's honest list of what it does not know.
        led = getattr(self, "_escalations", None) or {}
        e = led.setdefault(str(query), {"reason": "no memory hit and no confident tool reflex",
                                        "count": 0})
        e["count"] += 1
        self._escalations = led
        return {"served": False, "via": "escalate",
                "reason": "no memory hit and no confident tool reflex -- the next "
                          "level up decides"}

    # -- warm-started factoring (E3.1) ------------------------------------------------------------
    def factor_warm(self, composite, context_vec, resonator, solutions, gate=0.55,
                    iters=100, blend=0.75):
        """Warm-start a ResonatorNetwork past its capacity cliff from logged experience (lever 7,
        calibrated rung). `solutions` is the experience log: a list of (context_vec, factor_indices)
        fed by the expensive path. The nearest logged context (cosine >= gate) seeds each shared
        slot with blend*atom + (1-blend)*superposition; below the gate the resonator runs cold.
        Returns {indices, fired}. MEASURED (this tree, same stop rule both sides for fairness):
        past the cliff at noise 0.3, cold 1/30 -> warm 8/30 (8x); novel contexts refused. The
        deep-dive Part 9 harness saw 35-44/100 with an any-sweep oracle stop -- the honest
        production number is the 8x, and the stop='best' rule is what banks it. KEPT NEGATIVE: the key
        must be the TASK CONTEXT -- keying on the composite fired 10/120 because binding with an
        independent factor decorrelates the products."""
        from holographic.agents_and_reasoning.holographic_ai import bundle, cosine
        ctx = np.asarray(context_vec, float)
        best_s, best_t = 0.0, None
        for c, t in solutions:
            s = cosine(ctx, np.asarray(c, float))
            if s > best_s:
                best_s, best_t = s, t
        cold = [bundle(list(cb)) for cb in resonator.codebooks]
        if best_t is None or best_s < gate:
            return {"indices": resonator.factor(np.asarray(composite, float), iters=iters),
                    "fired": False, "context_similarity": best_s}
        est = []
        for f, cb in enumerate(resonator.codebooks):
            if f < len(best_t) and best_t[f] is not None:
                est.append(blend * cb[int(best_t[f])] + (1 - blend) * cold[f])
            else:
                est.append(cold[f])
        return {"indices": resonator.factor(np.asarray(composite, float), iters=iters, init=est, stop="best"),
                "fired": True, "context_similarity": best_s}

    @property
    def stream_recipes(self):
        """The mind's generator-RECIPE CACHE (lazy singleton) for stream_route_warm."""
        if getattr(self, "_lever7_recipes", None) is None:
            from holographic.agents_and_reasoning.holographic_lever7 import RecipeCache
            self._lever7_recipes = RecipeCache()
        return self._lever7_recipes

    def stream_route_warm(self, x):
        """Route a stream with the LEVER-7 fast path in front of the HRNN ladder: the nearest
        logged family recipe is refit closed-form and HOLDOUT-VALIDATED on this very stream (the
        gate); refusal runs the full ladder, and any generator verdict it returns is logged so
        the family's next member is cheap. Every verdict carries provenance ('via'). MEASURED
        (deep-dive Part 11): 9.9x over 40 family streams; white noise never served a generator."""
        warm = self.stream_recipes.try_stream(x)
        if warm is not None:
            return warm
        if getattr(self, "_lever7_hrnn", None) is None:
            from holographic.agents_and_reasoning.holographic_hrnn import HolographicRNN
            self._lever7_hrnn = HolographicRNN(dim=1024, seed=0)
        verdict = self._lever7_hrnn.process_stream(np.asarray(x, float).ravel())
        try:
            mdl = verdict.get("model") or {}
            f0 = mdl.get("fundamental")
            params = mdl.get("params")
            if verdict.get("regime") == "generator" and f0 is not None and params is not None:
                H = max(1, (len(params) - 1) // 2)           # sin/cos pairs + intercept
                self.stream_recipes.note(x, float(f0), int(H))
        except Exception:
            pass                                             # logging is best-effort; the verdict stands
        if isinstance(verdict, dict):
            verdict.setdefault("via", "full_ladder")
        return verdict

    @property
    def working_memory(self):
        """The mind's WORKING MEMORY as a capacity-priced bundle (lazy singleton): allocator-quoted
        admission, relevance by cosine, EXACT eviction by subtraction with salvage, transcript
        floor beside the bundle. See holographic_lever7.WorkingMemory."""
        if getattr(self, "_lever7_wm", None) is None:
            from holographic.agents_and_reasoning.holographic_lever7 import WorkingMemory
            self._lever7_wm = WorkingMemory(dim=2048)
        return self._lever7_wm

    def wm_admit(self, vec, tag, note=None):
        """Admit an item to working memory under the allocator's quote (the capacity law is the
        budget; no token counting). Returns the quote with the admission verdict."""
        return self.working_memory.admit(vec, tag, note)

    def wm_recall(self, task_vec, k=5):
        """The working set ranked by relevance to the live task."""
        return self.working_memory.recall_ranked(task_vec, k)

    def wm_evict_for(self, task_vec):
        """Free capacity: exactly evict the least task-relevant item and RETURN it for salvage
        (add_note it to the KnowledgeStore before it leaves -- the anti-silent-loss rule)."""
        return self.working_memory.evict_least_relevant(task_vec)

    def maintain_experience(self):
        """The consolidation pass (backlog E6.1 mechanism): per tile -- merge near-duplicate
        response atoms and recalibrate the null at current load. Event-driven by design: call it
        from an idle hook or the jobs system; it costs nothing when there is nothing to merge,
        and the audit floor is never touched."""
        return [t.consolidate() for t in self.experience.tiles]

    def wm_compact(self, task_vec, keep=8):
        """COMPACT WORKING MEMORY WITH SALVAGE (backlog E5.4'): evict least-task-relevant items
        beyond `keep`, and before each one leaves, SALVAGE it into the mind's one memory
        (self.learn under its tag) so nothing is silently lost -- eviction from the hot bundle,
        never from the record. Returns the salvage manifest."""
        salvaged = []
        wm = self.working_memory
        while len(wm._items) > int(keep):
            ev = wm.evict_least_relevant(np.asarray(task_vec, float))
            if ev is None:
                break
            try:
                self.learn(ev["vec"], label=str(ev["tag"]))
                sink = "mind.learn"
            except Exception:
                sink = "returned-only"
            salvaged.append({"tag": ev["tag"], "note": ev["note"], "sink": sink})
        return {"salvaged": salvaged, "remaining": len(wm._items), **wm.quote()}

    def schedule_maintenance(self):
        """Register the consolidation pass with the JOBS SYSTEM (backlog E6.1 wiring): the worker
        'experience_maintain' becomes schedulable by any jobs backend (local pool, farm, idle
        hook). Returns the registered worker name and an immediate dry run's report so the
        wiring is verified, not assumed."""
        jm = getattr(self, "_job_manager", None)
        jm = jm() if callable(jm) else jm
        name = "experience_maintain"
        fn = lambda *_a, **_k: self.maintain_experience()
        if jm is not None and hasattr(jm, "register_worker"):
            jm.register_worker(name, fn)
            registered = True
        else:
            registered = False
        return {"worker": name, "registered": registered, "dry_run": fn()}

    def semantic_view_create(self, name, source, column, value, k=10):
        """Create an INCREMENTAL SEMANTIC VIEW on the mind's Database (/invoke-reachable facade
        for Database.create_semantic_view -- a capability the service cannot call does not
        exist). See holographic_query.SemanticView: delta-refresh, bit-identical to cold."""
        return self.db.create_semantic_view(name, source, column, value, k=k)

    def semantic_view_run(self, name):
        """Refresh + return a semantic view (facade for Database.run_semantic_view)."""
        return self.db.run_semantic_view(name)

    def semantic_view_stats(self, name):
        """The view's honesty ledger (facade for Database.semantic_view_stats)."""
        return self.db.semantic_view_stats(name)

    def experience_coverage(self, probes, threshold=0.5):
        """WHERE CAN LEVER 7 NOT HELP YET: coverage of the displacement trace's audited keys over
        the given probe tasks, plus the top void probes ('escalate here on purpose'). The measured
        law this gauge makes live: the lever's win tracks log coverage exactly."""
        from holographic.agents_and_reasoning.holographic_lever7 import experience_coverage as _cov
        return _cov(self.experience, probes, threshold)

    def learn_semantic_keys(self, extra_corpus=None):
        """Train the TextEncoder on the CATALOG'S OWN CORPUS (every capability's name, description
        and aliases) so paraphrase queries acquire shared geometry -- the E3.7' unblock for the
        semantic reflex. KEPT NEGATIVES this exists to beat (measured, deep-dive Part 11): the
        UNLEARNED encoder scored 0/17 paraphrase cache hits (random word atoms have no synonym
        structure -- the catalog's own recall@1 kept negative, reconfirmed), and token-Jaccard
        served only 4/6 hits correctly. Key-law clause 4: the key's kernel must be CHOSEN --
        here it is learned from the corpus the keys will serve. MEASURED after training on the
        catalog's 3,412 cards (145k tokens): paraphrase reflex FIRES 6/6 where the unlearned
        encoder fired 0/17, gibberish stays refused (max cos 0.03); residual hit accuracy (3/6
        served the same top-1 as a fresh suggest) is limited by BOTH the key and suggest()'s own
        paraphrase sensitivity -- the alias-set validation gate remains the follow-on."""
        from holographic.io_and_interop.holographic_encoders import TextEncoder
        te = TextEncoder(dim=2048, seed=0)
        cat = getattr(self, "_capability_catalog", None)
        if callable(cat):
            cat = cat()
        docs = []
        entries = getattr(cat, "_by_name", None) or {}
        it = entries.values() if hasattr(entries, "values") else entries
        for e in it:
            parts = [getattr(e, "name", ""), getattr(e, "does", "")] +                     list(getattr(e, "aliases", ()) or ())
            docs.append(" ".join(str(x) for x in parts))
        for d in (extra_corpus or []):
            docs.append(str(d))
        n_tok = 0
        for d in docs:
            toks = d.lower().replace("-", " ").replace("_", " ").split()
            if toks:
                te.learn(toks)
                n_tok += len(toks)
        self._lever7_text = te
        return {"documents": len(docs), "tokens": n_tok}

    def semantic_ingest(self, text, source="conversation"):
        """THE CONVERSATION IS A CORPUS (checkpoint 17): feed live traffic -- queries, taught
        answers, plans, escalated chain-of-thought -- into the SAME incremental TextEncoder
        that learn_semantic_keys seeds from the catalog. TextEncoder.learn is online by
        construction: an unseen word gets a context vector on first contact and sharpens with
        every co-occurrence, so NEW WORDS AND CONCEPTS from conversation join the semantic
        space exactly like corpus words do. Returns {new_words, tokens, vocab} so the growth
        is visible. The nomic-style embedding model seeds the space; the attached model's own
        traffic keeps growing it."""
        te = getattr(self, "_lever7_text", None)
        if te is None:
            from holographic.io_and_interop.holographic_encoders import TextEncoder
            te = self._lever7_text = TextEncoder(dim=2048, seed=0)
        toks = [w for w in str(text).lower().replace("-", " ").replace("_", " ").split()]
        before = len(te.context)
        if toks:
            te.learn(toks)
        stats = getattr(self, "_semantic_ingest_stats", None) or             {"tokens": 0, "calls": 0, "sources": {}}
        stats["tokens"] += len(toks)
        stats["calls"] += 1
        stats["sources"][str(source)] = stats["sources"].get(str(source), 0) + 1
        self._semantic_ingest_stats = stats
        return {"new_words": len(te.context) - before, "tokens": len(toks),
                "vocab": len(te.context)}

    def semantic_ingest_stats(self):
        """How much conversation has become corpus: tokens, calls, per-source counts, vocab."""
        te = getattr(self, "_lever7_text", None)
        st = dict(getattr(self, "_semantic_ingest_stats", None) or
                  {"tokens": 0, "calls": 0, "sources": {}})
        st["vocab"] = len(te.context) if te is not None else 0
        return st

    def semantic_key(self, text):
        """A similarity key for a text task from the CORPUS-LEARNED encoder (learn_semantic_keys
        first; falls back to the unlearned encoder with a warning field if not yet trained)."""
        te = getattr(self, "_lever7_text", None)
        trained = te is not None
        if te is None:
            from holographic.io_and_interop.holographic_encoders import TextEncoder
            te = TextEncoder(dim=2048, seed=0)
            self._lever7_text = te
        _stop = {"a", "an", "the", "of", "in", "on", "to", "for", "and", "or", "with",
                 "from", "by", "at", "is", "it", "this", "that", "my", "our", "your"}
        toks = [w for w in str(text).lower().replace("-", " ").replace("_", " ").split()
                if w not in _stop] or [str(text).lower()]
        v = np.asarray(te.encode_sentence(" ".join(toks)), float)
        n = float(np.linalg.norm(v))
        return {"vec": v / (n + 1e-12), "trained": trained}

    def policy_lean_export(self, policy, theorem_name="policy_total"):
        """Export a PolicyProgram's route table to LEAN 4 (backlog E4.5'): prove totality (a
        fallback route exists) from the policy's own Horn facts and emit self-contained,
        checkable Lean source via the engine's prover -- a LEARNED-ROUTING policy whose
        statement an external checker can verify. Honest scope inherited from to_horn_rules:
        the export certifies the route TABLE + fallback existence, not first-match priority
        (that semantics lives in the program vector and its decode)."""
        rules = policy.to_horn_rules()
        return self.lean_export(["total", ["policy"]], rules, theorem_name=theorem_name)

    # -- doctrine ---------------------------------------------------------------------------------
    def realize(self, recipe):
        """Replay a StructureRecipe to its output vector(s) -- the single realize path for any structure."""
        outs = recipe.outputs()
        return outs[0] if len(outs) == 1 else outs

    def scene_scaling(self, s):
        """A 4x4 scale transform (uniform scalar or per-axis length-3) for scene_graph nodes."""
        from holographic.scene_and_pipeline.holographic_scenegraph import scaling
        return scaling(s)

    def levers(self, problem=None, measured=None):
        """THE SEVEN LEVERS: what to do when you hit a measured wall, in cost order. Extends the
        six exact levers with the one lever that is NOT exact -- spend accumulated experience.
        See _UnifiedPart18.levers for the six; entry 7 is installed here (Part 19)."""
        # sweep 63: the six-lever base moved to a PRIVATE name -- two parts
        # defining public levers() is the silent-shadow hazard; delegation is
        # now explicit rather than riding the MRO.
        base = self._levers_base(problem=problem, measured=measured)
        entry7 = {
            "n": 7,
            "name": "spend accumulated experience -- amortize across SIMILARITY, not identity, "
                    "under a calibrated bound",
            "when": "every new input pays full price even though the workload is self-similar; "
                    "the wall is per-input cost, and levers 1/3 cannot fire because inputs never "
                    "repeat exactly",
            "do": "log task->response moves as they happen (the displacement trace); answer a new "
                  "task from its NEIGHBORHOOD -- replay the blended move (speed), store only the "
                  "residual against the neighborhood's prediction (compression), or seed the "
                  "search with the neighbor's solution (optimization) -- and GATE the shortcut: "
                  "cleanup against the response codebook at a calibrated null, the capacity-law "
                  "trust price, and the volatility field; refuse where the log is empty, "
                  "neighbors disagree, or the region is marked volatile",
            "evidence": "speed: the gated reflex avoided 52.7% of model calls at 2/158 wrong "
                        "where the ungated cache served 48 wrong (measured). optimization: "
                        "warm-started resonator factoring past the capacity cliff, 4-5/100 cold "
                        "-> 35-44/100 (7-11x, measured). compression: predict-from-neighbor "
                        "residual storage added 2.44x on top of quantization on clustered "
                        "streams and honestly ~1x on structureless ones (measured). exact rungs "
                        "exist too: chain-transported sphere tracing (0 wrong pixels) and "
                        "incremental fuzzy views (bit-identical, 12.1x fewer row-cosines). its "
                        "trained-system contemporaries are the delta rule (DeltaNet/RWKV-7) and "
                        "surprise-gated writes (Titans) -- adopted here, not reinvented",
            "costs": "the ONLY lever that is not exact: a BOUNDED error rate is traded for "
                     "coverage of inputs never seen before -- the bound must be measured "
                     "(calibration), maintained (recalibration as the log grows), and honored "
                     "(volatility marking); plus the log itself, which lever 6 tiles at the "
                     "advisory (self-tiling) and lever 3 floors (bit-identical replay). refuse "
                     "where algebra already dissolved the wall (FFT-exact solves, the "
                     "optimizer-free SQL layer): there, break-even is infinity",
        }
        if isinstance(base, list):
            return base + [entry7]
        if isinstance(base, dict):
            base = dict(base)
            base["lever7"] = entry7
            return base
        return base


def _selftest():
    """Part-19 wiring smoke: run via the module (python -m ...p19_lever7) or the mind selftests."""
    from holographic.misc.holographic_unified import UnifiedMind
    from holographic.agents_and_reasoning.holographic_ai import random_vector, cosine
    rng = np.random.default_rng(0)
    m = UnifiedMind()
    lv = m.levers()
    assert isinstance(lv, list) and len(lv) == 7 and lv[6]["n"] == 7, "levers() must list seven"
    k = random_vector(2048, rng); v = random_vector(2048, rng)
    assert m.reflex_write(k, v)["accepted"]
    out = m.reflex_try(k)
    assert out["fired"] and cosine(out["prediction"], v) > 0.9
    assert not m.reflex_try(random_vector(2048, rng))["fired"], "novel task must be refused"
    m.reflex_mark_volatile("t", k)
    assert not m.reflex_try(k)["fired"], "volatile region must not fire"
    m.reflex_unmark_volatile("t")
    assert m.reflex_try(k)["fired"]
    return {"levers": len(lv), "stats": m.reflex_stats()}


if __name__ == "__main__":
    print(_selftest())

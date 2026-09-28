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
            # advisory_load 0.03 (sweep 176, the reflex bridge): MEASURED readback of an exact repeat vs load in
            # one 2048-d tile -- 1.000 up to n=50, 0.93 at 100, ~0.6 at 150 -- while the default advisory of
            # 0.10 only split at n=205, i.e. after the cliff. 0.03 (n=61) keeps every tile on the flat part,
            # which is what the tiling docstring promised ("the measured cliff number becomes the tile size").
            self._lever7_trace = TiledDisplacementTrace(dim=2048, seed=0, advisory_load=0.03)
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
        # sweep 177: the reflex bridge's label book + seen gate ride along. Without them a loaded trace answers
        # "no experience yet" (measured: 91,561-byte save, 0 fires in the second mind). Keys go as plain lists,
        # like the audit floor above; an older loader ignores the extra key.
        br = self.reflex_bridge_state()
        state["reflex"] = {"labels": br["labels"], "kinds": br["kinds"], "outcomes": br["outcomes"],
                           "keys": br["keys"].tolist()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f)
        return path

    def experience_from_state(self, state):
        """Rebuild the displacement trace from a STATE DICT -- the shared core of
        experience_load and the container loader (one rebuild, two transports)."""
        from holographic.agents_and_reasoning.holographic_lever7 import (
            DisplacementTrace, TiledDisplacementTrace)
        # advisory_load=0.03 -- the SAME tile size the `experience` property builds (sweep 176: readback is flat to
        # n~50 and falls off a cliff past ~100 in one 2048-d tile). Found by the CLM panel (E1.5 pass): this rebuild
        # dropped it, so every tile a RELOADED mind split later reverted to the old 0.10 default (split at n=205,
        # after the cliff) -- a restart silently undid the fix.
        tt = TiledDisplacementTrace(dim=2048, seed=0, advisory_load=0.03)
        tt.tiles = [DisplacementTrace.from_state(st) for st in state["tiles"]]
        tt._centroids = []
        tt._counts = []
        for t in tt.tiles:
            # centroids are the mean of ORDINARY write keys: a raw correction (E1.5) never moves routing -- the same
            # rule TiledDisplacementTrace.write_raw and _split use, so live and reloaded routing agree
            ks = np.asarray([k for k, _, mode in t.audit_entries() if not mode], float)
            tt._centroids.append(ks.mean(axis=0) if len(ks) else np.zeros(t.dim))
            tt._counts.append(len(ks))
        # The split counter seeds the NEXT split's hyperplane (seed + 104729 + splits). A rebuilt trace started it
        # at 0, so a reloaded mind split along a different plane than the live one would have (learning-loop audit,
        # 2026-09-27). Every split adds exactly one tile, so the count is recoverable from the tiles.
        tt.splits = max(0, len(tt.tiles) - 1)
        self._lever7_trace = tt
        out = {"tiles": len(tt.tiles), "writes": sum(tt._counts)}
        # sweep 177: restore the reflex bridge (label book + seen gate) when the state carries it. A state
        # written before this change has no "reflex" key and loads exactly as before.
        if state.get("reflex"):
            out["reflex"] = self.reflex_bridge_restore(state["reflex"])
        return out

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
        """The mind's tool-usage AUDIT view: .counts (successful uses per tool), .failures (failed uses), .dim (the
        tool door's key dimension), .note / .predict (= tool_note / tool_predict). E4.4 (CLM backlog, ONE tool
        learner): this used to be a separate UsageTrace -- a second learner of tool choice that serve() never read.
        The learner is now the tool door's ProtoStore (holographic_unified_p33_router); this view keeps the audit
        tallies where callers and tests read them."""
        # The view lives in its own module: a mixin part holds exactly ONE class (tests/test_unified_split.py -- the
        # cross-part name-collision check reads only a part's single class). Imported HERE, not at module top, so it
        # stays off `import lecore`'s required footprint (tests/test_deptrace.py caps the non-part modules at 35).
        from holographic.agents_and_reasoning.holographic_toolusage import ToolUsageView
        return ToolUsageView(self)

    def tool_note(self, task_vec, tool, success=True):
        """Record a JUDGED tool use against its task, in the tool door's ProtoStore (E4.4: the one tool learner): a
        success pulls the tool's prototype toward task_vec by the shared InfoNCE rule (pushing its close rivals), a
        failure is a labelled negative (push the tool away). task_vec must be in the door's key space -- build it with
        mind.tool_key(text). Audit counts are kept beside it. Returns {tool, uses, rows}."""
        v = np.asarray(task_vec, float).reshape(-1)
        st = self._tool_store()
        if v.size != st.dim:
            raise ValueError("tool_note: task_vec has %d dims, the tool door keys are %d (use mind.tool_key(text))"
                             % (v.size, st.dim))
        a = self._tool_audit()
        a["counts"][tool] = a["counts"].get(tool, 0) + (1 if success else 0)
        if success:
            st.update(v, tool)
        else:
            a["failures"][tool] = a["failures"].get(tool, 0) + 1
            if tool in st:
                st.negative(v, tool)
        return {"tool": tool, "uses": a["counts"][tool], "rows": len(st)}

    def tool_predict(self, task_vec, k=3):
        """Rank the known tools for a task from the tool door's prototypes (cosine, best first) -> [(tool, score)].
        An empty door returns [] -- refusal is a result, and the caller falls through to the full registry.
        task_vec in the door's key space: mind.tool_key(text)."""
        st = self._tool_store(create=False)
        if st is None or not len(st):
            return []
        v = np.asarray(task_vec, float).reshape(-1)
        if v.size != st.dim:
            raise ValueError("tool_predict: task_vec has %d dims, the tool door keys are %d (use mind.tool_key(text))"
                             % (v.size, st.dim))
        return st.rank(v, int(k))

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
        served). The reflex is stored on the mind AND seeds the tool door's prototype
        (E4.3: the pattern is a labelled example of its tool) so tool_predict ranks it
        for similar tasks from day one.
        A CREDENTIAL IN params (an api key, a token) is stored as a ${SERVICE_PARAM} placeholder, never the value
        (2026-09-27): api_use reads it from os.environ at call time and fails loudly naming the variable when it is
        unset. The result's `env` lists the variables the reflex needs. (Before this, a reflex with a key in its
        params was refused by the learning guard on its taught row -- it lived in process memory only, and died.)"""
        from holographic.io_and_interop.holographic_apilearn import placeholderize, placeholders_in
        if not hasattr(self, "_tool_reflexes"):
            self._tool_reflexes = []
        params = placeholderize(dict(params or {}), str(service))
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
        # E4.3 / E4.4: the pattern seeds (or, for a tool already known, moves) the tool door's prototype -- the ONE
        # tool learner. It used to be noted into a separate UsageTrace under a semantic_key vector that serve() never
        # read. The audit counts the seed as one use (the old trace's convention; tests pin it).
        tool = "%s.%s" % (service, endpoint)
        try:
            self._tool_seed(str(pattern), tool)
            a = self._tool_audit()
            a["counts"][tool] = a["counts"].get(tool, 0) + 1
        except Exception:
            pass
        return {"taught": True, "reflexes": len(self._tool_reflexes), "env": placeholders_in(params)}

    def serve(self, query, k=3, verify=None):
        """PREEMPTIVE SERVE (sweep 104 -- the openzoo division of labor, complete): try
        MEMORY first (T0 taught recall); then the USAGE-LEARNED TOOL REFLEX -- when a
        taught tool's pattern shares >= 2 content words with the query, extract the
        declared arguments (numbers in order for extract_numbers params), CALL the tool
        (api_use), and return its live result with provenance 'tool-reflex' and the
        tool named -- NO LLM IN THE LOOP; then ABSTAIN honestly upward ('escalate') so
        the next level (a model, a human, a bigger harness) knows the substrate could
        not serve this alone. KEPT NEG: argument extraction is
        deterministic and declared -- a query whose arguments cannot be extracted is
        escalated, never guessed.
        TOOL CALLS ARE DECISIONS (sweep 178): every tool-reflex call -- success OR failure --
        is a DecisionRecord (via 'tool') whose id comes back as `id`. Pass
        verify=callable(query, tool, result) -> bool and a call it passes is reported as an
        outcome (decision_outcome -> reflex_learn + the tool door), and the next similar query
        picks the judged tool BEFORE word overlap does. Without verify nothing is taught; uses
        and failures are counted (tool_usage).
        WHICH TOOL (E4.3, CLM backlog): (1) a near-repeat of a JUDGED question -> the reflex
        bridge's experience (picked_by 'experience'); (2) once the tool door's ProtoStore has
        learned from at least one judged call -> its prototype argmax over this mind's tools
        (picked_by 'proto' when that OVERRIDES the overlap pick; a failed verify is a labelled
        negative, a reported correction moves both tools); (3) cold start -> the word-overlap
        pick, exactly as before. A query only
        reaches a tool at all through (1) or the overlap gate (>= 2 shared words).
        THE ROUTER (E4.5): a memory miss that no tool reflex takes is ROUTED before it escalates
        -- a route ANSWER whose gate clears the route door's bar is served as "use capability
        X" (via 'route', with the route's id, so the outcome can be reported); a menu or a
        refusal escalates with the menu in the packet. A taught tool reflex that matched keeps
        priority over the router: it is the owner's own, more specific instruction."""
        # MAPLE SWARM RUN (2026-09-27): the escalation ledger is what memory cannot answer NOW. A question escalated
        # once and then taught under ANOTHER wording (or linked by a later verdict on a rewording) serves at T0 the
        # next time it is asked -- MEASURED, two such questions stayed 'open' after they served. A served answer
        # therefore closes that question's entry; an escalation still records it (_serve_body).
        out = self._serve_body(query, k=k, verify=verify)
        if isinstance(out, dict) and out.get("served"):
            led = getattr(self, "_escalations", None)
            if led:
                led.pop(str(query), None)
        return out

    def _serve_body(self, query, k=3, verify=None):
        """The body of serve() (see its docstring); the public door keeps the escalation ledger current."""
        import re
        r = self.ask(str(query))
        if r.get("tier") == "T0" and str(r.get("answer") or "").strip():
            from holographic.agents_and_reasoning.holographic_decisionrecord import door_record, legacy_p
            out0 = {"served": True, "via": "memory", "tier": "T0", "answer": r.get("answer")}
            if r.get("via") == "meaning":                   # sweep 181: found by meaning -- say which row, and
                out0.update(matched_by="meaning", row=r.get("row"), id=r.get("id"), p=legacy_p(r))  # the id to correct it
                # E0.6: the MeaningIndex's p is isotonic P(correct) -> p_correct (no null on this door)
                return door_record(out0, "serve-meaning", p_correct=r.get("p_correct", legacy_p(r)), p_null=None)
            return out0
        # sweep 181: memory knows HOW (a learned method) or knows it must ASK (a bare value, a known unclear wording)
        if r.get("tier") == "clarify":
            return {"served": True, "via": "clarify", "tier": "clarify", "answer": r.get("answer"),
                    "row": r.get("row")}
        if r.get("call") and str(r.get("via", "")).startswith("method"):
            return {"served": r.get("via") != "method-failed", "via": r.get("via"), "tier": "T2",
                    "answer": r.get("answer"), "call": r.get("call"), "live": r.get("live"), "row": r.get("row"),
                    "id": r.get("id"), "why": r.get("why")}
        # THE REFLEX BRIDGE (sweep 176): after a memory miss, a request whose ROUTE outcome was reported before
        # is served from experience -- the capability that was actually used -- before the tool reflexes.
        try:
            if getattr(self, "_reflex_labels", None) and getattr(self, "_reflex_seen", None):
                rf = self.reflex_decide(str(query), key="fingerprint")
                if rf.get("value"):
                    from holographic.agents_and_reasoning.holographic_decisionrecord import door_record
                    # E0.6: the bridge's calibrated P(correct); the deprecated p keeps the same (old: 1 - error) value
                    return door_record({"served": True, "via": "reflex", "tier": "T1", "capability": rf["value"],
                                        "confidence": rf.get("confidence"), "p": rf.get("p_correct"), "id": rf.get("id")},
                                       "serve-reflex", p_correct=rf.get("p_correct"), p_null=None)
        except Exception:
            pass
        if not getattr(self, "_tool_reflexes", None):
            # lazy rebuild from the durable rows (survives restarts on every rail)
            import json as _json
            from holographic.io_and_interop.holographic_apilearn import placeholderize
            self._tool_reflexes = []
            _log = getattr(self.zoo["ladder"], "taught_log", []) or []
            for _i, row in enumerate(_log):
                if len(row) > 3 and str(row[3]) == "toolreflex":
                    try:
                        spec = _json.loads(str(row[1]))
                    except ValueError:
                        continue
                    # MIGRATION ON LOAD (2026-09-27): a reflex row written with a raw credential in its params is
                    # rewritten in place to ${SERVICE_PARAM} placeholders -- the raw text is gone from the next save
                    _pp = placeholderize(dict(spec.get("params", {}) or {}), str(spec.get("service", "")))
                    if _pp != spec.get("params", {}):
                        spec["params"] = _pp
                        _log[_i] = [row[0], _json.dumps(spec)] + list(row[2:])
                    self._tool_reflexes.append(
                        {"pattern": str(row[0])[len("toolreflex: "):],
                         "service": spec["service"], "endpoint": spec["endpoint"],
                         "params": spec.get("params", {}),
                         "extract_numbers": spec.get("extract_numbers", []),
                         "uses": 0, "hits": 0})
            # E4.3: a tool the door has no prototype for (a partition written before the tool door, or one whose
            # door section was lost) is re-seeded from its durable pattern -- text first, like every other rail
            try:
                _st = self._tool_store(create=bool(self._tool_reflexes))
                for e_ in self._tool_reflexes:
                    t_ = "%s.%s" % (e_["service"], e_["endpoint"])
                    if _st is not None and t_ not in _st:
                        self._tool_seed(e_["pattern"], t_)
            except Exception:
                pass
        qw = {w for w in re.findall(r"[a-z]{4,}", str(query).lower())}
        best, best_shared = None, 0
        # JUDGED EXPERIENCE FIRST (sweep 178): if a verified tool call was reported for a query like this one
        # (ngram key, the reflex's seen gate), that tool is the pick -- word overlap only ranks what has not
        # been judged. The learned label must name one of THIS mind's tool reflexes, or it is ignored.
        picked_by = "overlap"
        exp_conf = None
        try:
            if getattr(self, "_reflex_labels", None) and getattr(self, "_reflex_seen", None):
                rf = self.reflex_decide(str(query), key="ngram")
                for e in getattr(self, "_tool_reflexes", []) or []:
                    if rf.get("value") == "%s.%s" % (e["service"], e["endpoint"]):
                        best, best_shared, picked_by = e, 99, "experience"
                        exp_conf = rf.get("confidence")
                        break
        except Exception:
            pass
        for e in (getattr(self, "_tool_reflexes", []) or []) if picked_by == "overlap" else []:
            pw = {w for w in re.findall(r"[a-z]{4,}", e["pattern"].lower())}
            shared = len(qw & pw)
            if shared > best_shared:
                best, best_shared = e, shared
        proto_cos = None
        if picked_by == "overlap" and best is not None and best_shared >= 2:
            # E4.3: the overlap gate says "this is a tool question"; WHICH tool is the tool door's decision once it
            # has learned from a judged call (cold start -> None -> the overlap pick stands, bit-identical).
            try:
                pick = self._tool_pick(str(query), ["%s.%s" % (e["service"], e["endpoint"])
                                                     for e in self._tool_reflexes])
            except Exception:
                pick = None
            if pick is not None:
                cand = next((e for e in self._tool_reflexes if "%s.%s" % (e["service"], e["endpoint"]) == pick[0]),
                            None)
                # the door's choice is the pick. When it names the SAME tool the overlap rule would have called (always
                # so with one tool), nothing was overridden: the record keeps picked_by 'overlap' and its word-overlap
                # score, so the 'tool:overlap' calibrator keeps its one scale (E0.6). 'proto' marks an OVERRIDE.
                if cand is not None and "%s.%s" % (cand["service"], cand["endpoint"]) != \
                        "%s.%s" % (best["service"], best["endpoint"]):
                    best, picked_by, proto_cos = cand, "proto", pick[1]
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
            tool = "%s.%s" % (best["service"], best["endpoint"])
            ok = isinstance(out, dict) and bool(out.get("ok"))
            # THE CALL IS A DECISION (sweep 178). Before this, a tool-reflex call left a trace note on success
            # and NOTHING on failure, and no record anyone could report against -- so the decision trees and
            # typed decisions, which learn from records, never saw a single tool call.
            from holographic.agents_and_reasoning.holographic_decisionrecord import DecisionRecord, door_record
            names = sorted({"%s.%s" % (e["service"], e["endpoint"]) for e in self._tool_reflexes})
            # E0.6: the tool reflex had no margin and no p. Its SCORE is what picked the tool -- the word overlap
            # (an integer >= 2) or, for a judged pick, the experience confidence -- two scales, so two door
            # calibrators ('tool:overlap', 'tool:experience'); p_correct is None until that door has outcomes.
            # E4.3: a door pick's score is the prototype cosine -- a third scale, a third calibrator ('tool:proto')
            if picked_by == "experience" and exp_conf is not None:
                score = float(exp_conf)
            elif picked_by == "proto" and proto_cos is not None:
                score = float(proto_cos)
            else:
                score = float(best_shared)
            tdoor = "tool:" + picked_by
            p_tool = self._door_p(tdoor, score)
            rec = DecisionRecord(str(query), "tool-reflex", names, tool, "tool", margin=score, p_correct=p_tool,
                                 meta={"pattern": best["pattern"], "ok": ok, "picked_by": picked_by})
            # E4.3: a reported outcome on this record (the judge's pass, or a correction naming the right tool) is a
            # labelled verdict for the tool door -- a live hook; the reflex bridge learns from the same report
            _q = str(query)
            self.decision_ledger().add(rec, hook=lambda outcome, _q=_q, _t=tool: self._tool_outcome(_q, _t, outcome),
                                       hook_key="tooldoor")
            # E4.4: the audit tallies (successful uses / failed uses) -- NOT learning: an unjudged call teaches
            # nothing (the old UsageTrace strengthened on every ok call: self-report, sweep 177's trap)
            _a = self._tool_audit()
            _a["counts"][tool] = _a["counts"].get(tool, 0) + (1 if ok else 0)
            if not ok:
                _a["failures"][tool] = _a["failures"].get(tool, 0) + 1
            if ok:
                best["hits"] += 1
                verified = None
                if verify is not None:
                    try:
                        verified = bool(verify(str(query), tool, out.get("data")))
                    except Exception:
                        verified = False        # a judge that crashes has not passed anything
                    if verified:
                        self.decision_outcome(rec.id, tool)
                    else:
                        # a JUDGED miss is a calibration label for this door (score, wrong). It is NOT reported as an
                        # outcome: no truth is known, and a 'failed' outcome would paint the failure field region-wide
                        # (sweep 178's kept negative) -- the calibrator only needs the number.
                        self.door_calibrator(tdoor).observe(score, False)
                        # E4.3: ... and a LABELLED NEGATIVE for the tool door (push this tool away from this question)
                        try:
                            self._tool_verdict(str(query), tool, None)
                        except Exception:
                            pass
                    rec.meta["verified"] = verified
                return door_record({"served": True, "via": "tool-reflex", "tool": tool, "id": rec.id,
                                    "verified": verified, "picked_by": picked_by, "score": score,
                                    "matched_pattern": best["pattern"], "result": out.get("data"),
                                    "uses": best["uses"], "hits": best["hits"]}, "tool", p_correct=p_tool, p_null=None)
            return door_record({"served": False, "via": "escalate", "id": rec.id, "tool": tool, "score": score,
                                "reason": "reflex tool call failed: %s" % str(out)[:120]}, "tool",
                               p_correct=p_tool, p_null=None)
        # E4.5 -- SERVE ASKS THE ROUTER (the panel's G4: all 20 held-out alias probes escalated because this door never
        # consulted the catalog router). A route ANSWER that clears the route door's bar is served as "use capability
        # X"; a menu or a refusal escalates below, with the menu in the packet.
        route_r = None
        try:
            served_r, route_r = self._serve_route(str(query))
            if served_r is not None:
                return served_r
        except Exception:
            route_r = None
        # ESCALATION LEDGER (sweep 125): every ask the substrate could not serve is
        # recorded so a service swarm can route it to a human and resolve() it back
        # into memory with provenance -- the swarm's honest list of what it does not know.
        led = getattr(self, "_escalations", None) or {}
        e = led.setdefault(str(query), {"reason": "no memory hit and no confident tool reflex",
                                        "count": 0})
        e["count"] += 1
        self._escalations = led
        out_e = {"served": False, "via": "escalate",
                 "reason": "no memory hit and no confident tool reflex -- the next "
                           "level up decides"}
        # sweep 181: the TYPED escalation packet -- the prompt and the nearest rows -- so the next level up (a model
        # or person in another process) can answer it and report the verdict with meaning_resolve(query, reply)
        try:
            out_e["meaning"] = self.meaning_packet(str(query))
        except Exception:
            pass
        # E4.5: the route the router DID produce rides in the packet -- the menu (or the answer that missed the bar)
        # and its id, so the next level up can pick a capability and report it with decision_outcome(id, card)
        if route_r is not None:
            out_e["route"] = {"tier": route_r.get("tier"), "id": route_r.get("id"), "z": route_r.get("z"),
                              "options": [o["name"] for o in route_r.get("options", [])]}
        return out_e

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
        # sweep 179: redact BEFORE tokenizing -- every token becomes a persisted vocabulary word, so an API key
        # or seed phrase in live traffic would otherwise be learned verbatim as "new words".
        from holographic.agents_and_reasoning.holographic_learnguard import redact
        text = redact(text)
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

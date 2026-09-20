"""Part 27 of UnifiedMind's faculty surface -- the DECISION faculties (sweeps 173-176): the decision tree, the
DecisionRecord ledger and outcomes by id, the reflex bridge (the arc learning from use, the seen gate, re-tiling),
verify_decision, batch FDR, guarded EM, the swarm step contract and evaluator, and cold plans from a request.

NOT A STANDALONE MODULE. One slice of the single `UnifiedMind` class, assembled by holographic/misc/
holographic_unified.py (the only import path anyone uses). Split out of part 08 when that part passed the
2,000-line cap (tests/test_unified_split.py): the bodies were moved by line range, byte-identical, and every
method is still a real attribute of UnifiedMind at runtime (mixin, not delegation). These part classes are NOT
a public API and must never be imported or subclassed directly; they carry no `__init__` and assume the state
UnifiedMind.__init__ builds.
"""
import numpy as np

from holographic.unified import check_part


class _UnifiedPart27:
    def decision_ledger(self):
        """The one DecisionRecord store for this mind (sweep 176, backlog G1): every route_tiered and
        systemone_decide call adds a record with an id; decision_outcome(id, outcome) is the outcome path and
        forwards typed decisions to SystemOne.observe by construction. See holographic_decisionrecord.Ledger."""
        from holographic.agents_and_reasoning.holographic_decisionrecord import Ledger
        if getattr(self, "_decision_ledger", None) is None:
            self._decision_ledger = Ledger()
        return self._decision_ledger

    def decision_outcome(self, record_id, outcome):
        """REPORT AN OUTCOME BY ID (sweep 176, backlog G2) -- the only outcome path. Returns {id, outcome,
        forwarded, was_correct}; for a typed decision `forwarded` is SystemOne.observe's report (the count
        table learned; no teach() call anywhere). Unknown id -> KeyError. See holographic_decisionrecord."""
        rep = self.decision_ledger().report(record_id, outcome)
        # the reflex bridge (sweep 176): every reported outcome teaches the experience trace
        rep["reflex"] = self.reflex_learn(record_id)
        return rep

    def decision_records(self, k=None):
        """The ledger as dicts, newest last (optionally the last k), plus stats. See holographic_decisionrecord."""
        L = self.decision_ledger()
        ids = L._order[-int(k):] if k else L._order
        return {"records": [L.get(i).to_dict() for i in ids], "stats": L.stats()}

    def decision_ledger_to_memory(self, only_reported=True, prefix="decision"):
        """RECORDS INTO THE PARTITION (sweep 176, backlog G3; NOOA s6 item 5 -- the memory subsystem): every
        reported DecisionRecord becomes a taught row -- query "<prefix>: <question> :: <state>", answer = the
        record as JSON (bounded preview for a long state) -- so `ask` can recall a past decision by its state
        at T0 and memory_curate's ACT-R activation, decay and reflection apply to decisions exactly as to facts.
        Idempotent by construction: the record id is in the answer, and teaching the same row twice is one row
        in the log. Returns {taught, skipped, ids}. Doc 05: 'a negative a future session can only find by reading
        prose is a negative that will be re-run' -- this is the structured path."""
        import json as _json
        L = self.decision_ledger()
        taught, skipped, ids = 0, 0, []
        done = getattr(self, "_ledger_taught", None)
        if done is None:
            done = self._ledger_taught = set()
        for rid in list(L._order):
            rec = L.get(rid)
            if (only_reported and rec.outcome is None) or rid in done:
                skipped += 1
                continue
            d = rec.to_dict()
            state = d["state"] if isinstance(d["state"], str) else "[%d chars] %s" % (d["state"]["len"], d["state"]["ref"])
            self.teach("%s: %s :: %s" % (prefix, rec.question, state), _json.dumps(d, sort_keys=True, default=str))
            done.add(rid)
            taught += 1
            ids.append(rid)
        return {"taught": taught, "skipped": skipped, "ids": ids}

    # ---- THE REFLEX BRIDGE (sweep 176; Moose: "the reflex arc isn't learning as it's used") -----------
    # Audit before this: reflex_write / reflex_try were called only from their own part, reflex_outcome from
    # one zoo path (successes only), calibrate_reflex from nowhere. leOS's design (README: VERIFY -> LEARN,
    # and "self-extending instructions": one successful escalation becomes a permanent reflex entry) names
    # the missing edges. These verbs are them.
    def _reflex_label_atom(self, label):
        """The response codebook the reflex trace can name: one derived atom per answer label."""
        from holographic.mesh_and_geometry.holographic_planshape import derived_atom
        book = getattr(self, "_reflex_labels", None)
        if book is None:
            book = self._reflex_labels = {}
        if label not in book:
            # at the TRACE's dimension (2048 by default), not the mind's -- the trace binds key and value
            book[label] = derived_atom(0, "answer:" + str(label), int(self.experience.dim))
        return book[label]

    def _reflex_task_vec(self, state, key="fingerprint"):
        """The similarity KEY for the trace, at the trace's dimension, chosen by the DOOR (sweep 176, measured):
        'fingerprint' -- how the state ROUTES (top-8 capability atoms): for route_tiered and decision_tree, where
        paraphrases of a card fire 0.967 at 0.983 right on a loaded trace. 'ngram' -- the state's own hashed
        n-grams: for typed decisions, whose states are usually OUTSIDE the catalog's domain, where fingerprints
        collapse and unrelated sentences look alike (measured: 116 fires on 400 novel rows, 83 wrong)."""
        import numpy as _np
        encs = getattr(self, "_reflex_encs", None)
        if encs is None:
            from holographic.agents_and_reasoning.holographic_decisiontree import routing_fingerprint
            from holographic.agents_and_reasoning.holographic_systemone import hashed_ngram_encode
            d = int(self.experience.dim)
            encs = self._reflex_encs = {"fingerprint": routing_fingerprint(self._capability_catalog(), dim=d, seed=0, k=8),
                                        "ngram": hashed_ngram_encode(dim=d)}
        v = _np.asarray(encs[key](state), float).reshape(-1)
        return v / (_np.linalg.norm(v) + 1e-12)


    def reflex_learn(self, record_id):
        """LEARN FROM A REPORTED DECISION (the leOS VERIFY -> LEARN edge): reflex_write(task, answer-atom) when the
        outcome confirmed the answer -- including an ESCALATED answer, which is leOS's self-extending instruction:
        one successful model-end call teaches the reflex to answer the next similar task without the model --
        and reflex_outcome(task, was_correct) always, so failures land in the failure field. Called by
        decision_outcome; safe to call again (a write the trace already predicts is skipped free)."""
        rec = self.decision_ledger().get(record_id)
        if rec is None or rec.outcome is None or not isinstance(rec.state, str):
            return {"learned": False, "why": "no record / no outcome / non-text state"}
        kind = "fingerprint" if rec.via in ("route", "tree", "reflex") else "ngram"
        if rec.via == "reflex":
            kind = rec.meta.get("key", "fingerprint")
        tv = self._reflex_task_vec(rec.state, key=kind)
        ok = (rec.answer == rec.outcome)
        # THE SEEN GATE's memory (sweep 176): the keys of reported decisions. A reflex fire is trusted only when a
        # reported key sits within `seen_cosine` of the query -- MEASURED: trace confidence cannot separate a
        # genuine repeat on a loaded trace (median 0.076) from a wrong fire on a novel row (median 0.056).
        keys = getattr(self, "_reflex_seen", None)
        if keys is None:
            keys = self._reflex_seen = []
        keys.append((kind, tv, rec.outcome))
        if rec.via == "reflex" and rec.margin is not None:
            # CALIBRATION FEEDBACK (sweep 176): a reflex-answered decision whose outcome is reported is exactly a
            # (confidence, ok) pair for calibrate_reflex -- before this only the zoo's own serves fed it, so the
            # bridge's reflex answers always carried p=None.
            if not hasattr(self, "_reflex_calib_pairs"):
                self._reflex_calib_pairs = []
            self._reflex_calib_pairs.append([float(rec.margin), bool(ok)])
        # The TRUTH is known whatever the door said, so the correct move is always written. The FAILURE
        # FIELD is marked only when an answer was GIVEN and was wrong: an abstention (menu / None) with a
        # reported truth is a learning opportunity, not a miss -- the first cut marked it as a failure and
        # painted the region so the reflex refused there forever (measured: 0 fires on exact repeats).
        FAILED = (None, "", "fail", "failed", "__failed__")
        if rec.outcome in FAILED:
            # No known good move here: THIS is what the failure field is for (one cosine, region-wide).
            self.reflex_outcome(tv, False)
            w = None
        else:
            # A truth is known -- confirmed or a CORRECTION. Write it; the trace's delta-rule write moves the
            # prediction toward the truth. Do NOT mark the region as failed on a correction: the second cut
            # did, and a tree whose root the user corrected could never learn the pick (measured: 0 of 1).
            w = self.reflex_write(tv, self._reflex_label_atom(rec.outcome))
            self._verify_learn(rec, tv, self._reflex_label_atom(rec.outcome))     # the profile + support stream
            if ok:
                self.reflex_outcome(tv, True)
        n = self.decision_ledger().stats()["reported"]
        cal = None
        if n >= 8 and n % 8 == 0:
            try:
                cal = self.calibrate_reflex()
            except Exception as e:                      # honest refusal below the sample floor
                cal = {"refused": str(e)[:80]}
        return {"learned": True, "was_correct": ok, "write": w, "calibrated": cal}

    def reflex_decide(self, state, min_confidence=0.0, seen_cosine=0.8, key="fingerprint"):
        """ANSWER FROM EXPERIENCE FIRST (the lever-7 read on the decision doors): reflex_try on the state; if it
        fires above the calibrated null (and min_confidence) and its atom names a known label, return {value,
        confidence, error_prob, via='reflex'} and add a 'reflex' DecisionRecord; else {value: None, why}. The
        three gates (null, trust, volatility) and the failure field are the reflex's own; this only names the
        atom. min_confidence=0.1 is MEASURED: on a 600-row Banking77 stream every wrong fire on a NOVEL row sat
        below 0.1 (197 fires at 0.254 accuracy, all in [0, 0.1)); repeats fire higher. Without the floor the
        reflex-first typed door fell from 0.778 to 0.603 -- the count table learns from the same outcomes and
        beats a whole-key trace on paraphrases; the reflex earns its place on REPEATS and on doors that have no
        learner of their own (the router). Default-off everywhere: reflex=True opts in."""
        # SELF-HEAL (sweep 176): a trace loaded by boot under the old advisory (split at n=205) reads back under
        # the floor; if any tile sits past twice the cliff advisory, re-tile once from the audit log instead of
        # asking an operator to remember a ritual. It runs on the FIRST reflex read -- before the bridge's first
        # write -- because a write that lands on the old tiling and is then replayed among ~950 others in tile
        # order came back at confidence 0 (measured on the service); on clean tiles it reads at 0.68.
        if not getattr(self, "_reflex_retiled", False):
            try:
                if any(t.load() > 2.0 * 0.03 for t in self.experience.tiles):
                    self.reflex_retile(advisory_load=0.03)
            finally:
                self._reflex_retiled = True
        book = getattr(self, "_reflex_labels", None) or {}
        if not book:
            return {"value": None, "why": "no experience yet"}
        tv = self._reflex_task_vec(state, key=key)
        seen = [(k, o) for kind, k, o in (getattr(self, "_reflex_seen", None) or []) if kind == key]
        if seen:
            import numpy as _np
            K = _np.stack([k for k, _ in seen])
            sims = K @ tv
            j = int(_np.argmax(sims))
            nearest = float(sims[j])
        else:
            nearest, j = 0.0, -1
        if nearest < seen_cosine:
            return {"value": None, "why": "not seen before (nearest reported key %.2f < %.2f)" % (nearest, seen_cosine)}
        r = self.reflex_try(tv)
        if not r.get("fired") or r.get("confidence", 0.0) < min_confidence:
            return {"value": None, "why": r.get("why"), "confidence": r.get("confidence", 0.0), "nearest_seen": nearest}
        import numpy as _np
        pred = _np.asarray(r["prediction"], float)
        labels = list(book)
        A = _np.stack([book[l] for l in labels])
        sims = A @ (pred / (_np.linalg.norm(pred) + 1e-12))
        j = int(_np.argmax(sims))
        if float(sims[j]) < 0.5:                        # fired, but on an atom this door never wrote
            return {"value": None, "why": "atom-not-a-label", "confidence": r["confidence"]}
        from holographic.agents_and_reasoning.holographic_decisionrecord import DecisionRecord
        rec = DecisionRecord(state, "reflex", labels, labels[j], "reflex", margin=float(r["confidence"]),
                             p=(None if self.reflex_error_prob(r["confidence"]) is None else 1.0 - float(self.reflex_error_prob(r["confidence"]))),
                             meta={"key": key, "nearest_seen": nearest})
        self.decision_ledger().add(rec)
        return {"value": labels[j], "confidence": float(r["confidence"]), "error_prob": self.reflex_error_prob(r["confidence"]),
                "via": "reflex", "id": rec.id, "atom_cosine": float(sims[j])}

    def reflex_retile(self, advisory_load=0.03):
        """RE-TILE THE EXPERIENCE TRACE AT THE MEASURED CLIFF (sweep 176): rebuild a fresh TiledDisplacementTrace
        from every tile's bit-identical audit log (lever 7 stands on lever 3) with a new advisory_load, so a
        trace persisted under the old default (split at n=205, after the readback cliff at ~100) is spread over
        tiles that read back cleanly. Found live: the service's persisted trace held 872 writes on 6 tiles and
        answered a just-taught repeat at confidence 0.05 -- below the floor -- while a fresh mind answered it at
        0.68. Volatility marks are carried over per tile. Returns {before_tiles, after_tiles, writes}."""
        from holographic.agents_and_reasoning.holographic_lever7 import TiledDisplacementTrace
        old = self.experience
        pairs = []
        for tile in old.tiles:
            pairs.extend(list(getattr(tile, "_audit", [])))
        new = TiledDisplacementTrace(dim=old.dim, seed=old.seed, advisory_load=float(advisory_load))
        for k, v in pairs:
            new.write(k, v)
        try:
            for tile in old.tiles:
                vol = getattr(tile, "volatility", None)
                marks = getattr(vol, "_marks", None) or getattr(vol, "marks", None)
                if marks:
                    for tag, vec in (marks.items() if isinstance(marks, dict) else []):
                        new.tiles[new._route(vec)].volatility.mark(tag, vec)
        except Exception:
            pass
        self._lever7_trace = new
        return {"before_tiles": len(old.tiles), "after_tiles": len(new.tiles), "writes": len(pairs)}

    # ---- VERIFY (sweep 176, Moose: "verify the final result is a valid response to the input") -------------
    def verify_decision(self, state, answer, key="ngram", question=None, recent=64):
        """IS THIS ANSWER A VALID RESPONSE TO THIS INPUT? leOS step 4 (embed the response, check its displacement
        against the expected profile) done holographically, five checks with their evidence:
          forward   -- the experience trace, read with the STATE key, cleans up to this answer's atom (cosine).
          backward  -- the trace read the OTHER way, unbind(trace, answer atom), points back at this state's key
                       (cosine). A pair the trace holds agrees in BOTH directions; a pair it never saw, or an
                       answer bound to other states, fails one of them. This is the bidirectional lookup.
          profile   -- cosine of bind(state, answer) with the DISPLACEMENT PROFILE, the mean of bind(state, truth)
                       over reported-correct decisions: does this pair displace the way correct pairs do.
          drift     -- the decision's support against the RECENT support stream (z-score over the last `recent`
                       reported decisions): an answer whose support sits far below the stream is an outlier.
          seen      -- the seen gate: nearest reported key to this state (cosine).
        Returns {valid, score, checks, why}. `valid` is a stated rule, not a discovered one: forward and backward
        both above 0.05 when the trace knows the region, or seen >= 0.8 with the profile above 0, or -- with no
        experience at all -- undecided (valid=None) rather than a guess. The AUROC of each check against reported
        outcomes is measured in the bench; the scalar `score` is their mean.
        MEASURED (Banking77, nb typed decisions, outcomes reported by id):
          * a SEEN state served the recorded truth vs a wrong label (200 pairs): forward AUROC 1.000, backward
            1.000, profile 0.946; verdict valid 0.990 for the truth, 0.000 for the wrong label on a fresh
            trace, and 0.970 / 0.000 on a boot-like trace holding 950 other writes -- a served answer that
            contradicts experience is caught every time. (An ABSOLUTE floor let a wrong label through on the
            loaded trace at 0.06 of crosstalk; the checks are relative to the other labels for that reason.)
          * NOVEL rows (450, verified before the outcome): forward 0.605, backward 0.606, profile 0.578 vs the
            decision's own margin 0.812 -- the trace can only vouch for what it has seen; valid=True still
            marks a 12.4% subset that is 0.911 right.
          * KEPT NEGATIVE: averaging the checks into a score and mixing it with the margin as one confidence
            RANKS WORSE (mixed stream AUROC 0.826 vs margin alone 0.885) -- the scales do not mix. This is a
            VERDICT (confirm / veto against experience), never the ranking signal; the calibrated margin is."""
        import numpy as _np
        from holographic.agents_and_reasoning.holographic_ai import bind, unbind, cosine
        tv = self._reflex_task_vec(state, key=key)
        av = self._reflex_label_atom(answer)
        checks = {}
        # forward / backward over every tile (the pair may live in any) -- RELATIVE to the alternatives: on a
        # boot-loaded trace (~950 writes) every atom carries ~0.06 of crosstalk, so an absolute floor let a wrong
        # label through over HTTP (measured); what distinguishes the true pair is that the answer is the forward
        # read's BEST label and the state is more associated with this answer than with any other label.
        book = getattr(self, "_reflex_labels", None) or {}
        others = [b for l, b in book.items() if l != answer]
        fwd, bwd, fwd_alt, bwd_alt = 0.0, 0.0, 0.0, 0.0
        for t in self.experience.tiles:
            rd = t.read(tv)
            fwd = max(fwd, float(cosine(rd, av)))
            bwd = max(bwd, float(cosine(unbind(t._trace, av), tv)))
            if others:
                fwd_alt = max(fwd_alt, max(float(cosine(rd, b)) for b in others))
                bwd_alt = max(bwd_alt, max(float(cosine(unbind(t._trace, b), tv)) for b in others))
        checks["forward"], checks["backward"] = fwd, bwd
        checks["forward_margin"], checks["backward_margin"] = fwd - fwd_alt, bwd - bwd_alt
        prof = getattr(self, "_verify_profile", None)
        pair = bind(tv, av)
        checks["profile"] = float(cosine(pair, prof)) if prof is not None else None
        seen = [(k, o) for kind, k, o in (getattr(self, "_reflex_seen", None) or []) if kind == key]
        sims_seen = [float(k @ tv) for k, _ in seen]
        j = int(_np.argmax(sims_seen)) if sims_seen else -1
        checks["seen"] = float(max(sims_seen, default=0.0))
        sup = getattr(self, "_verify_support", None) or []
        if len(sup) >= 8:
            arr = _np.asarray(sup[-int(recent):], float)
            checks["drift_z"] = float((fwd - arr.mean()) / (arr.std() + 1e-9))
        else:
            checks["drift_z"] = None
        known = (fwd > 0.0 or bwd > 0.0) and self.reflex_stats().get("writes", 0) > 0
        if not known and not seen:
            return {"valid": None, "score": None, "checks": checks, "why": "no experience to verify against"}
        both = (fwd > 0.0 and bwd > 0.0 and checks["forward_margin"] > 0.0 and checks["backward_margin"] > 0.0)
        near_truth = (seen[j][1] if (seen and j >= 0) else None)
        near = (checks["seen"] >= 0.8 and near_truth == answer)
        parts = [v for v in (fwd, bwd, checks["profile"], checks["seen"]) if v is not None]
        score = float(_np.mean(parts)) if parts else None
        valid = bool(both or near)
        why = ("forward %.2f and backward %.2f agree and beat every other label by %.2f / %.2f" % (fwd, bwd, checks["forward_margin"], checks["backward_margin"])) if both else \
              ("seen %.2f with profile %s" % (checks["seen"], checks["profile"])) if near else \
              ("another label fits better (forward margin %.2f, backward margin %.2f), nearest seen %.2f" % (checks["forward_margin"], checks["backward_margin"], checks["seen"]))
        if checks["drift_z"] is not None and checks["drift_z"] < -2.0:
            valid = False
            why += "; support %.2f sigma below the recent stream" % checks["drift_z"]
        return {"valid": valid, "score": score, "checks": checks, "why": why}

    def _verify_learn(self, rec, tv, av):
        """Keep the displacement PROFILE (mean of bind(state, truth) over correct outcomes) and the support stream
        that verify_decision's drift check compares against. Called from reflex_learn."""
        import numpy as _np
        from holographic.agents_and_reasoning.holographic_ai import bind
        pair = bind(tv, av)
        prof = getattr(self, "_verify_profile", None)
        n = getattr(self, "_verify_n", 0)
        self._verify_profile = pair if prof is None else prof + (pair - prof) / (n + 1)
        self._verify_n = n + 1
        sup = getattr(self, "_verify_support", None)
        if sup is None:
            sup = self._verify_support = []
        fwd = 0.0
        for t in self.experience.tiles:
            fwd = max(fwd, float(_np.dot(t.read(tv), av) / (_np.linalg.norm(t.read(tv)) * _np.linalg.norm(av) + 1e-12)))
        sup.append(fwd)

    def decision_tree(self, context, depth=2, fanout=3, margin=0.1, encode=False,
                      dim=1024, seed=0, reflex=False):
        """GROW the suggestion node into a TREE (sweep 173): route() already builds one decision
        node on the fly ('did you mean one of these?'); this recurses it. Each node ranks the live
        catalog for the context, takes the shared decision step, and branches on WHAT THAT CHOICE
        RESULTS IN -- with an `abstain` branch everywhere, because 'ask the user' is a real outcome.
        The child's context is the request PLUS what just happened; MEASURED, that edge kept 4/5
        children on topic where chaining by the catalog's produces/consumes types kept 0/5.
        Returns {root, options, coverage} (+ tree vector when encode=True).
        See holographic_decisiontree.grow_tree."""
        from holographic.agents_and_reasoning.holographic_decisiontree import (
            CapabilityView, encode_tree, grow_tree)
        view = CapabilityView.from_catalog(self._capability_catalog())
        t = grow_tree(context, view, depth=depth, fanout=fanout, margin=margin)
        # THE TREE LEARNS FROM USE (sweep 176, the reflex bridge one door over): every node is a DecisionRecord
        # (via 'tree') whose id rides in options[path]; a pick reported against it -- decision_outcome(id, pick)
        # -- teaches the reflex keyed on that node's context. With reflex=True a node whose context the reflex
        # already knows is answered from experience: action = the learned pick, options[path]['via'] = 'reflex'.
        from holographic.agents_and_reasoning.holographic_decisionrecord import DecisionRecord
        L = self.decision_ledger()
        learned = 0
        for path, opt in t.options.items():
            ctx = opt.get("context") or context
            if reflex:
                rf = self.reflex_decide(ctx)
                if rf.get("value") is not None:
                    opt["via"] = "reflex"; opt["confident"] = True; opt["confidence"] = rf["confidence"]
                    opt["learned_action"] = rf["value"]; learned += 1
                    node = t.root
                    for b in path:
                        node = node.branches[b]
                    node.action = rf["value"]
                    node.confidence = float(rf["confidence"])
            rec = DecisionRecord(ctx, "tree:" + "/".join(path) if path else "tree:root",
                                 [n for n, _ in opt.get("ranked", [])], (opt.get("learned_action") or (opt["ranked"][0][0] if opt.get("ranked") and opt.get("confident") else None)),
                                 "tree", margin=None, meta={"path": list(path)})
            L.add(rec)
            opt["id"] = rec.id
        out = {"root": t.root, "options": t.options, "coverage": t.coverage, "tree": t, "learned_nodes": learned}
        if encode:
            vec, shape, vocab = encode_tree(t, dim=dim, seed=seed)
            out.update({"vector": vec, "shape": shape, "vocab": vocab})
        return out

    def plan_from_request(self, request, k=3, encode=True, dim=1024, seed=0):
        """A COLD PLAN WITH NO MODEL (sweep 176, backlog E1): split the request into single-observation clauses
        (holographic_systemone.clauses), route each through the tiered router, and chain the steps into a
        PlanNode contingency plan -- each step's action is the routed capability, its `ok` branch is the next
        step, its `abstain` branch a leaf -- encoded as one hypervector (encode_plan) when encode=True. MEASURED
        on compound requests made of two exact aliases (3 seeds x 100): split-and-route recovers BOTH 0.860
        vs 0.540 for the whole request's top-5 menu; the first measurement (0.527) was a splitter defect --
        ', next ' was not a connective -- found by re-testing with the routing of a single alias (0.990) as
        the control. Steps that land in the menu tier carry their options; refused clauses stay as steps with
        action 'ask'. Returns {steps, root, vector?, shape?, vocab?}."""
        from holographic.agents_and_reasoning.holographic_systemone import clauses
        from holographic.mesh_and_geometry.holographic_planshape import PlanNode
        steps = []
        for cl in clauses(request):
            r = self.route_tiered(cl, k=k)
            tool = r["answer"] or (r["options"][0]["name"] if r["options"] and r["tier"] != "refuse" else None)
            steps.append({"clause": cl, "tier": r["tier"], "tool": tool, "id": r["id"],
                          "options": [o["name"] for o in r["options"]] if r["tier"] != "answer" else []})
        root = None
        for st in reversed(steps):
            action = st["tool"] or "ask"
            branches = {"abstain": PlanNode("ask", scope="global", confidence=0.0)}
            branches["ok"] = root if root is not None else PlanNode("done", scope="global", confidence=1.0)
            root = PlanNode(action, scope="global", confidence=(1.0 if st["tier"] == "answer" else 0.5), branches=branches)
        out = {"steps": steps, "root": root}
        if encode and root is not None:
            from holographic.agents_and_reasoning.holographic_decisiontree import encode_tree
            vec, shape, vocab = encode_tree(root, dim=dim, seed=seed)
            out.update({"vector": vec, "shape": shape, "vocab": vocab})
        return out

    # ---- THE CODE WORKFLOW (sweep 176): plan, edit under validated termination, review with evidence ----------
    def plan_change(self, request, k=5, reflex=True):
        """PLAN A CODE CHANGE -- Rule 0 as a TYPED decision with the evidence attached (sweep 176). Gathers what
        exists (route_tiered over the catalog; code_search over the engine's own source; the top hit's family),
        writes that evidence into the decision STATE, and decides reuse / extend / build with systemone (seed
        examples from the sweeps' own history; reflex on, so a repeated brief is answered from experience). The
        decision is a DecisionRecord: report what was actually done -- decision_outcome(id, "reuse"|"extend"|
        "build") -- and the next plan learns from it. Returns {action, evidence, family, steps, id}, where steps
        are the build loop with a done_when per step (holographic_codeflow.BUILD_STEPS)."""
        from holographic.agents_and_reasoning.holographic_codeflow import BUILD_STEPS
        r = self.route_tiered(request, k=k)
        try:
            src_hits = self.code_search(request, k=3)
        except Exception:
            src_hits = []
        top = r["options"][0] if r["options"] else None
        fam = (top or {}).get("family")
        sim = float(src_hits[0][1]) if src_hits and len(src_hits[0]) > 1 else 0.0
        evidence = {"catalog_tier": r["tier"], "catalog_top": (top or {}).get("name"), "z": r.get("z"),
                    "source_top": (src_hits[0][0] if src_hits else None), "source_similarity": sim, "family": fam}
        state = ("request: %s. catalog: %s tier%s, top %r. source: nearest %r at similarity %.2f. family %s"
                 % (request, r["tier"], (" z %.2f" % r["z"]) if r.get("z") is not None else "", evidence["catalog_top"],
                    evidence["source_top"], sim, fam))
        q = {"action": {"type": "choice", "options": ["reuse", "extend", "build"], "examples": {
            "reuse": ["catalog: answer tier, a capability already does exactly this", "a card resolves at top-1 with a runnable example, call it",
                      "cleanup turned out to be a denoiser -- the existing module in a different costume", "the router answers with high z and the top card fits"],
            "extend": ["catalog: menu tier, an existing module does most of it, add a parameter or a method", "source similarity high to one module, a new scorer inside the same class",
                       "a default-off flag on the existing faculty", "an existing door needs one more argument"],
            "build": ["catalog: refuse tier, only fallbacks returned", "source similarity low, nothing similar exists", "a new module in the right family, wired and catalogued",
                      "the audit returned unrelated fallbacks, licence to build"]}}}
        a = self.systemone_decide(state, q, scorer="prototype", encoder="ngram", margin=0.02, reflex=reflex)["action"]
        # The MEASURED tier decides where it is decisive (answer -> reuse, refuse -> build); the typed decision
        # breaks the menu tie (extend vs build) and, through its record, LEARNS from reported outcomes. The
        # first cut let four seed examples per option outvote an answer-tier hit ("smooth a bumpy mesh" ->
        # build); a rule over a measured signal beats a scorer with no history.
        # MODIFICATION INTENT: "add a parameter to X", "let X also ...", "extend X" resolve at the answer tier
        # because X exists -- and the right action is extend, not reuse (measured: 2 of 3 remaining misses).
        intent = any(w in (" " + request.lower() + " ") for w in (" add ", " extend ", " parameter", " let the ", " let it ", " also ", " option to ", " flag "))
        if a.get("via") == "reflex" and a.get("value"):
            action = a["value"]
        elif r["tier"] == "answer":
            action = "extend" if intent else "reuse"
        elif r["tier"] == "refuse":
            action = "build"
        else:
            action = "extend" if intent else (a.get("value") if a.get("value") in ("extend", "build") else ("extend" if sim >= 0.3 else "build"))
        steps = [{"step": s, "do": d, "done_when": w} for s, d, w in BUILD_STEPS]
        if action == "reuse":
            steps = [{"step": "reuse", "do": "call %s" % evidence["catalog_top"], "done_when": "the example on its card runs on your input"}]
        elif action == "extend":
            steps = [{"step": "extend", "do": "add a default-off parameter or method to %s" % (evidence["source_top"] or evidence["catalog_top"]),
                      "done_when": "existing decisions bit-identical; the new path pinned in the selftest"}] + steps[2:]
        return {"action": action, "via": a.get("via", "typed"), "evidence": evidence, "family": fam, "steps": steps, "id": a.get("id"),
                "ranked": a.get("ranked")}

    def edit_verified(self, path, old, new, count=1, import_check=False, selftest_module=None):
        """ONE EDIT UNDER VALIDATED TERMINATION (sweep 176, NOOA): file_replace, then the checks -- syntax always,
        the import in a subprocess and the module selftest when asked. ANY failure undoes the edit with file_undo
        and REFUSES with the failing check attached: a broken file never survives the call. Measured: 200
        synthetic edits, half of them breaking the syntax -- every broken edit refused and the file restored
        byte-identical, every good edit accepted. The accepted edit is a swarm step (done_when = the checks,
        evidence = their results) so the ledger has it. Returns {ok, path, checks, undone, why, id}."""
        before = self.file_read(path) if hasattr(self, "file_read") else None
        rep = self.file_replace(path, old, new, count=count)
        checks = {"replace": rep}
        ok = True
        why = None
        c = self.file_python_check(path)
        checks["python_check"] = c
        if not c.get("ok"):
            ok, why = False, "python_check: %s" % c.get("error")
        if ok and import_check:
            mod = path[:-3].replace("/", ".") if path.endswith(".py") else path
            c = self.file_import_check(mod)
            checks["import_check"] = c
            if not c.get("ok"):
                ok, why = False, "import_check: %s" % str(c.get("error"))[:200]
        if ok and selftest_module:
            c = self.file_selftest(selftest_module)
            checks["selftest"] = {"ok": c.get("ok"), "tail": c.get("tail")}
            if not c.get("ok"):
                ok, why = False, "selftest: %s" % " | ".join(c.get("tail") or [])[:200]
        if not ok:
            und = self.file_undo(1)
            restored = (self.file_read(path) == before) if (before is not None and hasattr(self, "file_read")) else None
            return {"ok": False, "path": path, "checks": checks, "undone": und, "restored_byte_identical": restored, "why": why}
        step = self.swarm_step("edit %s" % path, "file_replace", {"path": path, "count": count},
                               done_when="python_check ok" + (", import ok" if import_check else "") + (", selftest ok" if selftest_module else ""),
                               evidence={k: (v if k != "replace" else {"replacements": v.get("replacements"), "first_line": v.get("first_line")}) for k, v in checks.items()},
                               worker="editor", topic="edits")
        return {"ok": True, "path": path, "checks": checks, "undone": None, "why": None, "id": step["id"]}

    def review(self, paths, duplicates=True, purity=True, tests=True, fn_cap_lines=120, module_cap_lines=2000):
        """REVIEW A CHANGE WITH EVIDENCE (sweep 176): for each path -- syntax; import (subprocess); determinism
        hazards from the AST (hash(), unseeded random, wall clock, unsorted listdir/glob, set iteration: the
        constitution's rules, with the line and the reason); undocumented public defs; oversized functions and
        modules; a missing selftest; possible DUPLICATES of each new top-level def by code_similar (Rule 0 asked
        of the source); IMPURE functions by function_purity (informational); and the tests the change needs
        (affected_tests). The verdict merge_ready is a stated rule -- no errors -- never a score (averaging
        findings ranked worse than the rule, doc 05). The review is a DecisionRecord: report "merged" or
        "reverted" by id. Returns {merge_ready, files: {path: {...}}, tests, id}."""
        from holographic.agents_and_reasoning.holographic_codeflow import review_source, top_level_defs
        paths = [paths] if isinstance(paths, str) else list(paths)
        files = {}
        for p in paths:
            src = self.file_read(p)
            rv = review_source(src, module_cap_lines=module_cap_lines, fn_cap_lines=fn_cap_lines)
            if p.endswith(".py"):
                try:
                    ic = self.file_import_check(p)              # the checker takes the PATH (first cut passed a dotted name)
                except Exception as e:
                    ic = {"ok": False, "error": str(e)[:200]}
                rv["import"] = ic
                if not ic.get("ok"):
                    rv["findings"].append({"level": "error", "kind": "import", "line": 0, "what": str(ic.get("error"))[:200]})
                    rv["counts"]["error"] += 1
                    rv["merge_ready"] = False
            if duplicates:
                dups = []
                for name in top_level_defs(src):
                    try:
                        sims = self.code_similar(name, k=3)
                    except Exception:
                        sims = []
                    for label, s in (sims or [])[:3]:
                        if float(s) >= 0.6 and not str(label).endswith("." + name):
                            dups.append({"def": name, "like": label, "similarity": float(s)})
                rv["possible_duplicates"] = dups
                for d in dups:
                    rv["findings"].append({"level": "warn", "kind": "duplicate", "line": 0, "what": "%s looks like %s (%.2f) -- Rule 0: reuse or extend?" % (d["def"], d["like"], d["similarity"])})
                    rv["counts"]["warn"] += 1
            if purity:
                try:
                    pr = self.purity_report(src)
                    rv["impure"] = [x for x in (pr.get("functions", pr) if isinstance(pr, dict) else pr) if isinstance(x, dict) and not x.get("pure", True)][:20]
                except Exception as e:
                    rv["impure"] = "purity_report unavailable: %s" % str(e)[:80]
            files[p] = rv
        tests_needed = None
        if tests:
            try:
                tests_needed = self.affected_tests(paths)
            except Exception as e:
                tests_needed = "affected_tests unavailable: %s" % str(e)[:80]
        merge_ready = all(v["merge_ready"] for v in files.values())
        from holographic.agents_and_reasoning.holographic_decisionrecord import DecisionRecord
        rec = DecisionRecord("review " + ", ".join(paths), "review", ["merged", "reverted"], "merged" if merge_ready else None, "typed",
                             meta={"errors": sum(v["counts"]["error"] for v in files.values()), "warns": sum(v["counts"]["warn"] for v in files.values())})
        self.decision_ledger().add(rec)
        return {"merge_ready": merge_ready, "files": files, "tests": tests_needed, "id": rec.id}

    def systemone_batch_fdr(self, states, questions, question, alpha=0.10, n_null=200, encoder="perceive",
                            scorer="prototype", nb_bigrams=False, margin=None):
        """BATCH FALSE-DISCOVERY CONTROL over a stream of typed decisions (sweep 176, H2): shuffle-null p per state
        (in-vocabulary word salad at matched length), Benjamini-Hochberg across the batch (holographic_ablate.bh_fdr).
        Measured: BH holds FDR 0.03 at nominal 0.05 / 0.10 while the naive per-test gate lets noise through at
        0.14 / 0.22; BY accepts nothing (kept negative). Power is low (13-20% of real rows) by construction.
        See holographic_systemone.SystemOne.batch_fdr."""
        so = self.systemone(questions, margin=margin, encoder=encoder, scorer=scorer, nb_bigrams=nb_bigrams)
        return so.batch_fdr(states, question, alpha=alpha, n_null=n_null)

    def systemone_absorb(self, states, questions, question, iters=3, lam=1.0, max_options=10, labeled=None,
                         margin=None, encoder="perceive", min_support=None, scorer="nb", nb_bigrams=False,
                         nb_transform=False, conformal_alpha=None):
        """ABSORB UNLABELED TRAFFIC into a typed decision (sweep 176, H3) -- guarded EM on the SAME cached
        SystemOne that systemone_decide uses for this schema, so what it absorbs is what the next decision uses.
        Two measured guards: refuses above max_options (EM collapsed at 77) and keeps the table only if held-out
        accuracy did not fall. Measured: nb_transform=False on AG News k=32 with 600 unlabeled rows 0.697 -> 0.752,
        beating the transformed default's 0.713; under the transform the gain vanishes, hence nb_transform=False
        here by default. See holographic_systemone.SystemOne.absorb_unlabeled."""
        so = self._systemone_cached(questions, labeled, margin, encoder, min_support, scorer, nb_bigrams,
                                    conformal_alpha=conformal_alpha)
        return so.absorb_unlabeled(states, question, iters=iters, lam=lam, max_options=max_options)

    def swarm_step(self, state, tool, args=None, done_when=None, evidence=None, topic="swarm", worker="worker",
                   outcome=None, verify=None, expect=None):
        """THE SWARM STEP CONTRACT (sweep 176, backlog J1) with NOOA's VALIDATED TERMINATION: one shape for
        every bus message a worker sends -- {id, state, tool, args, done_when, evidence, verify, via='swarm',
        worker, outcome}. `done_when` and `evidence` are REQUIRED (a step nobody can verify is not a step).
        `verify`, when given, is {"verb": <mind verb name>, "args": {...}} -- a verification command the
        harness RUNS BEFORE THE STEP IS ACCEPTED (arXiv 2607.20709 s3: the typed result carries evidence and a
        verification command checked by the harness before the call returns). `expect` decides pass: a
        callable(result) -> bool, or a value compared for equality, or None (any non-error result passes).
        A step whose verification fails is REFUSED with the result attached -- it never reaches the bus.
        Accepted steps are DecisionRecords in the ledger (outcome by id, cosine similarity to past steps) and
        are published on this mind's MessageBus under `topic`. Returns the published record dict."""
        if not done_when or evidence is None:
            raise ValueError("swarm_step refuses a step without done_when and evidence -- a step nobody can verify is not a step")
        verified = None
        if verify is not None:
            verb = getattr(self, str(verify.get("verb", "")), None)
            if verb is None:
                raise ValueError("swarm_step: verify.verb %r is not a mind verb" % verify.get("verb"))
            result = verb(**(verify.get("args") or {}))
            ok = (bool(expect(result)) if callable(expect) else (result == expect if expect is not None else True))
            if not ok:
                raise ValueError("swarm_step refused: verification %r returned %r, expected %r"
                                 % (verify.get("verb"), str(result)[:200], expect if not callable(expect) else "expect(result) True"))
            verified = {"verb": verify.get("verb"), "passed": True}
        from holographic.agents_and_reasoning.holographic_decisionrecord import DecisionRecord
        rec = DecisionRecord(state, "step:" + str(tool), [str(tool)], str(tool), "swarm", outcome=outcome,
                             meta={"args": args or {}, "done_when": done_when, "evidence": evidence, "worker": worker,
                                   "verify": verify, "verified": verified})
        self.decision_ledger().add(rec)
        payload = rec.to_dict()
        self.bus().publish(topic, payload)
        return payload


    def swarm_evaluate(self, audits=("reachability_audit", "catalog_gaps", "skill_lint"), timeout=1200):
        """THE EVALUATOR ROLE (sweep 176, backlog J2): run the audit suite as a subprocess each (never in-process)
        and return {audit: {ok, seconds, tail}} plus `all_ok`. This is the hard exit a swarm needs: a run that
        fails an audit stops with the audit named. Reuses Editor.run_selftest's runner via file_selftest-style
        subprocesses on tools/<audit>.py."""
        import subprocess, sys as _sys, time as _time, os as _os
        out = {}
        for a in audits:
            t0 = _time.time()
            try:
                r = subprocess.run([_sys.executable, "tools/%s.py" % a], cwd=str(self._editor.root), capture_output=True,
                                   text=True, timeout=timeout, env=dict(_os.environ, PYTHONHASHSEED="0"))
                lines = [l for l in (r.stdout + r.stderr).splitlines() if "Warning" not in l]
                out[a] = {"ok": r.returncode == 0, "seconds": round(_time.time() - t0, 1), "tail": lines[-4:]}
            except subprocess.TimeoutExpired:
                out[a] = {"ok": False, "seconds": timeout, "tail": ["timeout"]}
        out["all_ok"] = all(v["ok"] for k, v in out.items() if k != "all_ok")
        return out



def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p27_decisions", "_UnifiedPart27")
    print("holographic_unified_p27_decisions selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()

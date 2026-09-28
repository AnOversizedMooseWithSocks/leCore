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
            # E0.5: hooks are RE-DERIVED from each record's meta (a SystemOne cache key, the meaning door), never
            # held as live closures -- so a record reloaded from the partition still trains the right door.
            # spec_keys (learning-loop audit, 2026-09-26): a live closure registered under hook_key "router" or
            # "tooldoor" is ALSO written into its record as the spec {"kind": key}, which _decision_hook rebuilds after
            # a restart. MEASURED before: a route / tool outcome reported after a restart trained neither the router's
            # nor the tool door's ProtoStore (0 of 1 verdicts each; tools/audit_learning_loop.py).
            self._decision_ledger = Ledger(resolve_hook=self._decision_hook, spec_keys=("router", "tooldoor"))
        return self._decision_ledger

    def decision_outcome(self, record_id, outcome):
        """REPORT AN OUTCOME BY ID (sweep 176, backlog G2) -- the only outcome path. Returns {id, outcome,
        forwarded, was_correct}; for a typed decision `forwarded` is SystemOne.observe's report (the count
        table learned; no teach() call anywhere). Unknown id -> KeyError. See holographic_decisionrecord."""
        L = self.decision_ledger()
        rep = L.report(record_id, outcome)
        # E0.6 / E0.7: the outcome is a (score, correct) label for THAT door's own calibrator -- route, bridge,
        # tool:* -- so p_correct means P(correct) at every door and no door calibrates on another's scores
        rep["door"] = self._door_feed(L.get(record_id))
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
        if key == "meaning":
            # E5.3: a meaning decision's verify key is the question's hashed n-grams (the same encoder as 'ngram'),
            # kept under its OWN kind so the seen gate of the typed/tool doors never counts a meaning wording
            key = "ngram"
        v = _np.asarray(encs[key](state), float).reshape(-1)
        return v / (_np.linalg.norm(v) + 1e-12)


    def reflex_learn(self, record_id):
        """LEARN FROM A REPORTED DECISION (the leOS VERIFY -> LEARN edge): reflex_write(task, answer-atom) when the
        outcome confirmed the answer -- including an ESCALATED answer, which is leOS's self-extending instruction:
        one successful model-end call teaches the reflex to answer the next similar task without the model --
        and reflex_outcome(task, was_correct) always, so failures land in the failure field. Called by
        decision_outcome; safe to call again (a write the trace already predicts is skipped free).
        A CORRECTION (the door answered, the outcome names another truth) is stored by the measured correction
        design (reflex_correction_mode, E1.5); a MEANING decision joins the seen gate under its own kind so
        verify_decision can vouch for it, without writing the trace (E5.3)."""
        rec = self.decision_ledger().get(record_id)
        if rec is None or rec.outcome is None or not isinstance(rec.state, str):
            return {"learned": False, "why": "no record / no outcome / non-text state"}
        if rec.via == "rank":
            # E3.3: a rank decision's lesson lives in the rank door's OWN ProtoStore and calibrator (its hook already
            # ran). Writing its free-form candidate labels into the shared experience trace would put them in the label
            # book and the 'ngram' seen gate the typed and tool doors read -- one door's labels served at another
            # (separate stores per door, panel Q5; the meaning door's E5.3 reason, kept).
            return {"learned": False, "kind": "rank",
                    "why": "rank decisions learn in the rank door's ProtoStore, not the shared trace"}
        # THE LEARNING GUARD (sweep 179). A reported outcome becomes a LABEL the reflex can serve, so a secret
        # must never become one. A reading must not either -- but only when the outcome is FREE-FORM: an outcome
        # that is one of the record's declared options is a CHOICE (a tool, a card, a class), which is exactly
        # what a volatile question should learn ("call the price tool"), even if the choice's name has a digit.
        from holographic.agents_and_reasoning.holographic_learnguard import learning_verdict as _guard_check
        _is_choice = str(rec.outcome) in [str(o) for o in (rec.options or [])]
        # a declared choice ('mesh_smooth') is a bare token by shape: the SEMANTIC credential check would misread
        # it, so choices get the pattern layer only; free-form outcomes get both layers
        _v = _guard_check(rec.state, rec.outcome, allow_volatile=_is_choice,
                          guard=None if _is_choice else self.semantic_guard)
        if not _v["ok"]:
            return {"learned": False, "why": _v["reason"], "guard": _v["kind"]}
        kind = "fingerprint" if rec.via in ("route", "tree", "reflex") else "ngram"
        if rec.via == "reflex":
            kind = rec.meta.get("key", "fingerprint")
        if rec.via == "meaning":
            kind = "meaning"
        tv = self._reflex_task_vec(rec.state, key=kind)
        from holographic.agents_and_reasoning.holographic_decisionrecord import outcome_matches
        ok = outcome_matches(rec.answer, rec.outcome)   # a yes/no record's bool answer matches "yes"/"no" (E0.5)
        # THE SEEN GATE's memory (sweep 176): the keys of reported decisions. A reflex fire is trusted only when a
        # reported key sits within `seen_cosine` of the query -- MEASURED: trace confidence cannot separate a
        # genuine repeat on a loaded trace (median 0.076) from a wrong fire on a novel row (median 0.056).
        keys = getattr(self, "_reflex_seen", None)
        if keys is None:
            keys = self._reflex_seen = []
        keys.append((kind, tv, rec.outcome))
        # CALIBRATION FEEDBACK moved to decision_outcome -> _door_feed (E0.7): a reflex-answered decision's
        # (confidence, ok) now feeds door_calibrator('bridge') ONLY. It used to append to the one list the ladder's
        # veto was fitted on too (_reflex_calib_pairs), so every bridge report moved the ladder's veto.
        if rec.via == "meaning":
            # E5.3 -- MEANING SERVES BECOME VERIFIABLE. Before, this returned early and verify_decision could never
            # vouch for a meaning serve. Now the reported key joins the seen gate under its OWN kind ('meaning': the
            # typed/tool doors never see it) and the displacement profile learns the pair. The experience TRACE is
            # still NOT written: the lesson lives in the MeaningIndex (its hook already ran), and a second copy in
            # the trace would sit outside the meaning rung's vetoes (sweep 181's reason, kept) -- and
            # tests/test_meaning.py pins that corrections stop the wrong serve through the meaning door alone.
            if rec.outcome not in ("wrong",) + (None, "", "fail", "failed", "__failed__"):
                from holographic.mesh_and_geometry.holographic_planshape import derived_atom
                self._verify_learn(rec, tv, derived_atom(0, "answer:" + str(rec.outcome), int(self.experience.dim)),
                                   kind="meaning")
            return {"learned": True, "was_correct": ok, "write": None, "kind": "meaning",
                    "why": "meaning decisions learn in the MeaningIndex; the key joined the seen gate for verify"}
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
            if rec.answer is not None and not ok:
                # E1.5 -- A CORRECTION: the door ANSWERED and the answer was wrong. write(k, truth) alone corrected
                # along the truth atom only and left the wrong atom at full strength (truth read 45% after one
                # correction on real CLINC keys). The chosen design (reflex_correction_mode; lms_apa by measurement)
                # writes a signed, verbatim-replayed delta instead. See holographic_lever7 / section 8d.
                w = self._reflex_correct(rec, tv, kind)
            else:
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
        learner of their own (the router). Default-off everywhere: reflex=True opts in.
        RE-MEASURED WITH THE SEEN GATE (learning-loop audit, 2026-09-26; tools/audit_learning_loop.py --data,
        check reflex_floor): the 0.1 measurement above predates the seen gate, which now refuses novel rows by itself
        -- so the default stays 0.0. 13 Banking77 intents x 5 examples, typed door (nb), truth reported after every
        decision: 300 novel rows -> accuracy 0.700 with the reflex off, at floor 0.0 and at floor 0.1 alike (7 vs 6
        fires); the same stream with 300 exact repeats -> 0.8373 off / 0.8373 at 0.0 (297 fires at 0.976) / 0.8356
        at 0.1 (253 fires at 0.988). The floor only refuses repeats the count table would also have answered right."""
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
        if self.reflex_correction_mode() == "provenance":
            # E1.5 design (b) keeps rejections as outcome-role negatives in the trace; the read must consult them
            r = dict(self.experience.read_gated(tv, neg_role=self._reflex_neg_role()))
        else:
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
        from holographic.agents_and_reasoning.holographic_decisionrecord import DecisionRecord, door_record
        # E0.6/E0.7: the BRIDGE's own calibrator (fed only by reported reflex decisions) -- high = confident.
        # The deprecated bare p keeps its old value, 1 - error, which is the same number.
        pc = self._door_p("bridge", r["confidence"])
        err = None if pc is None else 1.0 - pc
        rec = DecisionRecord(state, "reflex", labels, labels[j], "reflex", margin=float(r["confidence"]),
                             p=pc, p_correct=pc, meta={"key": key, "nearest_seen": nearest})
        self.decision_ledger().add(rec)
        return door_record({"value": labels[j], "confidence": float(r["confidence"]), "error_prob": err,
                            "via": "reflex", "id": rec.id, "atom_cosine": float(sims[j])}, "reflex",
                           p_correct=pc, p_null=None)

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
            pairs.extend(tile.audit_entries())
        new = TiledDisplacementTrace(dim=old.dim, seed=old.seed, advisory_load=float(advisory_load))
        for k, v, mode in pairs:
            # E1.5: a RAW entry (a signed correction) replays VERBATIM -- through write() it would be re-estimated,
            # clamped at s >= 0 and dropped, and every correction would silently vanish at the next re-tile
            if mode:
                new.write_raw(k, v, mode)
            else:
                new.write(k, v)
        # THE OUTCOME FIELDS COME ALONG (learning-loop audit, 2026-09-26; the E2.1 rule a split and a reload already
        # follow). The success / failure fields are reported OUTCOMES, not audit writes, so a rebuild from the audit
        # alone dropped every reported failure and the outcome gate reopened on exactly the look-alike traps it was
        # measured to catch -- MEASURED: failure-field norm 2.23 before the retile, 0.00 after. The keys are re-routed
        # by the new tiling, so the WHOLE trace's evidence (summed over the old tiles) is laid over every new tile.
        new.set_outcome_fields(*old.outcome_fields())
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

    # ---- THE BRIDGE'S OWN STATE TRAVELS WITH THE TRACE (sweep 177) -----------------------------------------
    # MEASURED BUG this fixes (probe 2026-09-22): mind A reported a route outcome and answered the repeat
    # via='reflex'; experience_save wrote the trace (91,561 bytes); mind B loaded it and answered
    # "no experience yet". The trace travelled, but the two things reflex_decide checks BEFORE it reads the
    # trace did not: the label book (_reflex_labels -- empty book = "no experience yet") and the seen gate
    # (_reflex_seen -- the reported keys a query must sit within cosine 0.8 of). Both were plain attributes
    # nothing saved, so every restart and every second agent started blind. Same lesson as the factfiles
    # section in learning_save: A GATE THAT FORGETS WHAT IT WAS GUARDING REFUSES EVERYTHING.
    def reflex_bridge_state(self):
        """The reflex bridge's persistent state as plain data: {labels, kinds, outcomes, keys}. `keys` is an
        (n, dim) float32 array of the reported similarity keys, in report order; labels are NAMES only --
        each label's atom is derived deterministically from its name, so it is rebuilt, never stored."""
        seen = list(getattr(self, "_reflex_seen", None) or [])
        dim = int(self.experience.dim)
        keys = (np.stack([np.asarray(k, np.float32) for _, k, _ in seen]) if seen
                else np.zeros((0, dim), np.float32))
        return {"labels": [str(l) for l in (getattr(self, "_reflex_labels", None) or {})],
                "kinds": [str(kind) for kind, _, _ in seen],
                "outcomes": [None if o is None else str(o) for _, _, o in seen],
                "keys": keys, "verify": self._verify_state()}

    # ---- VERIFY'S OWN LEARNED STATE TRAVELS WITH THE BRIDGE (learning-loop audit, 2026-09-26) ------------------------
    # verify_decision's displacement PROFILE (mean of bind(state, truth) over reported-correct decisions) and its
    # drift stream (the forward support of recent reports) are learned from every reported outcome -- and lived in
    # plain attributes nothing saved. MEASURED (tools/audit_learning_loop.py): after 9 reported typed outcomes a
    # verdict read profile 0.503 and drift_z 0.152; after a restart, profile None and drift_z None -- the drift veto
    # (< -2 sigma) switched off until 8 new reports arrived. Same lesson as the bridge itself (sweep 177).
    _VERIFY_SUPPORT_KEPT = 512      # the drift check reads the last `recent` (64) supports; 512 is plenty and bounded

    def _verify_state(self):
        """verify_decision's learned state as plain data: the two profiles (arrays or None), their counts, and the
        newest _VERIFY_SUPPORT_KEPT supports. None when nothing was learned."""
        prof = getattr(self, "_verify_profile", None)
        mprof = getattr(self, "_verify_meaning_profile", None)
        sup = list(getattr(self, "_verify_support", None) or [])[-self._VERIFY_SUPPORT_KEPT:]
        if prof is None and mprof is None and not sup:
            return None
        return {"profile": None if prof is None else np.asarray(prof, np.float64),
                "n": int(getattr(self, "_verify_n", 0)),
                "meaning_profile": None if mprof is None else np.asarray(mprof, np.float64),
                "meaning_n": int(getattr(self, "_verify_meaning_n", 0)),
                "support": np.asarray(sup, np.float64)}

    def _verify_restore(self, st, merge=False):
        """Inverse of _verify_state. merge=True (importing another agent's experience) combines the profiles as a
        count-weighted mean -- exactly the running mean both sides would have had over the union of their reports --
        and appends the other side's supports; merge=False replaces."""
        if not st:
            return 0
        for pk, nk, attr, nattr in (("profile", "n", "_verify_profile", "_verify_n"),
                                    ("meaning_profile", "meaning_n", "_verify_meaning_profile", "_verify_meaning_n")):
            p_new, n_new = st.get(pk), int(st.get(nk) or 0)
            if p_new is None or n_new <= 0:
                continue
            p_new = np.asarray(p_new, np.float64).reshape(-1)
            p_old, n_old = getattr(self, attr, None), int(getattr(self, nattr, 0) or 0)
            if merge and p_old is not None and n_old > 0:
                setattr(self, attr, (n_old * np.asarray(p_old, np.float64) + n_new * p_new) / float(n_old + n_new))
                setattr(self, nattr, n_old + n_new)
            else:
                setattr(self, attr, p_new.copy())
                setattr(self, nattr, n_new)
        sup = [float(x) for x in np.asarray(st.get("support") if st.get("support") is not None else [],
                                            np.float64).reshape(-1)]
        if sup:
            base = list(getattr(self, "_verify_support", None) or []) if merge else []
            self._verify_support = (base + sup)[-self._VERIFY_SUPPORT_KEPT:]
        return 1

    def reflex_bridge_restore(self, state, merge=False):
        """Rebuild the label book and the seen gate from reflex_bridge_state() output. merge=False replaces
        (a load); merge=True appends (importing another agent's experience on top of your own -- keys already
        present are skipped, so importing twice is a no-op). Returns {labels, seen, added}."""
        if not state:
            return {"labels": 0, "seen": len(getattr(self, "_reflex_seen", None) or []), "added": 0}
        if not merge:
            self._reflex_labels = {}
            self._reflex_seen = []
        for name in state.get("labels") or []:
            self._reflex_label_atom(name)               # derived from the name: bit-identical everywhere
        seen = self._reflex_seen = getattr(self, "_reflex_seen", None) or []
        K = np.asarray(state.get("keys") if state.get("keys") is not None else [], np.float32)
        K = K.reshape(len(state.get("kinds") or []), -1) if len(K) else K
        have = {(kind, np.asarray(k, np.float32).tobytes()) for kind, k, _ in seen} if merge else set()
        added = 0
        for kind, k, o in zip(state.get("kinds") or [], K, state.get("outcomes") or []):
            tag = (kind, np.asarray(k, np.float32).tobytes())
            if tag in have:
                continue
            have.add(tag)
            seen.append((kind, np.asarray(k, float), o))
            if o not in (None, "", "fail", "failed", "__failed__") and kind != "meaning":   # same FAILED set as reflex_learn
                self._reflex_label_atom(o)              # a confirmed outcome is a label the trace may name
                                                        # (a meaning outcome is a ROW id: never a trace label, E5.3)
            added += 1
        # the audit's fix: verify_decision's profile + drift stream come back with the bridge (absent in older
        # partitions and in experience_save files: nothing changes for them)
        self._verify_restore(state.get("verify"), merge=merge)
        return {"labels": len(self._reflex_labels), "seen": len(seen), "added": added}

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
        meaning = (key == "meaning")
        if meaning:
            # E5.3: a meaning answer is a ROW id -- never registered as a trace label (the book is what the reflex
            # may NAME); its atom is derived from the name only for the profile check
            from holographic.mesh_and_geometry.holographic_planshape import derived_atom
            av = derived_atom(0, "answer:" + str(answer), int(self.experience.dim))
        else:
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
        prof = getattr(self, "_verify_meaning_profile" if meaning else "_verify_profile", None)
        pair = bind(tv, av)
        checks["profile"] = float(cosine(pair, prof)) if prof is not None else None
        seen = [(k, o) for kind, k, o in (getattr(self, "_reflex_seen", None) or []) if kind == key]
        sims_seen = [float(k @ tv) for k, _ in seen]
        j = int(_np.argmax(sims_seen)) if sims_seen else -1
        checks["seen"] = float(max(sims_seen, default=0.0))
        # the drift stream is the TRACE's forward support; a meaning answer is never in the trace (E5.3), so its
        # drift is judged by the MeaningIndex's own calibration instead -- comparing it here would be scale-mixing
        sup = (getattr(self, "_verify_support", None) or []) if not meaning else []
        if len(sup) >= 8:
            arr = _np.asarray(sup[-int(recent):], float)
            checks["drift_z"] = float((fwd - arr.mean()) / (arr.std() + 1e-9))
        else:
            checks["drift_z"] = None
        known = (fwd > 0.0 or bwd > 0.0) and self.reflex_stats().get("writes", 0) > 0
        if not known and not seen:
            return {"valid": None, "score": None, "checks": checks, "why": "no experience to verify against"}
        both = (fwd > 0.0 and bwd > 0.0 and checks["forward_margin"] > 0.0 and checks["backward_margin"] > 0.0)
        if meaning:
            # the trace holds no meaning lessons BY DESIGN, so forward/backward on a row atom are pure crosstalk --
            # with an empty label book they would 'agree' by chance. A meaning serve is vouched for only by the seen
            # gate: a reported, confirmed wording at cosine >= 0.8 whose outcome was THIS row.
            both = False
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

    def _verify_learn(self, rec, tv, av, kind=None):
        """Keep the displacement PROFILE (mean of bind(state, truth) over correct outcomes) and the support stream
        that verify_decision's drift check compares against. Called from reflex_learn. kind='meaning' (E5.3) keeps
        its OWN profile and no support stream (the trace never holds a meaning lesson, so its forward read is 0)."""
        import numpy as _np
        from holographic.agents_and_reasoning.holographic_ai import bind
        pair = bind(tv, av)
        if kind == "meaning":
            prof = getattr(self, "_verify_meaning_profile", None)
            n = getattr(self, "_verify_meaning_n", 0)
            self._verify_meaning_profile = pair if prof is None else prof + (pair - prof) / (n + 1)
            self._verify_meaning_n = n + 1
            return
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
        if len(sup) > 4 * self._VERIFY_SUPPORT_KEPT:
            del sup[:-self._VERIFY_SUPPORT_KEPT]        # bounded in memory too (the drift check reads the last 64)

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

    def typed(self, state, options, examples=None, question="answer", labeled=None, scorer=None, margin=None,
              conformal_alpha=None, reflex=False, escalate=None):
        """THE PLAIN FRONT DOOR for a typed decision (sweep 176): `mind.typed("courier lost the package",
        ["billing", "shipping"])`. Builds the schema for you -- a `choice` over `options`, with `examples` per
        option when given (a dict option -> [text]) and the option names themselves otherwise -- runs
        systemone_lint on it and on the state, picks the scorer from the measured regime table when you do not
        (prototype under ~10 examples per option, transformed NB from ~20), decides, and returns the ONE answer
        dict (value, ranked, margin_gap, p, set, via, id) with `lint` attached (its findings; heed the warns:
        the tool experiments' failures were all there). Report the outcome with decision_outcome(id, truth)."""
        from holographic.agents_and_reasoning.holographic_systemone import schema_lint
        opts = list(options)
        ex = {o: list(examples[o]) for o in opts if examples and examples.get(o)} if examples else {}
        for o in opts:
            ex.setdefault(o, [o.replace("_", " ")])
        q = {question: {"type": "choice", "options": opts, "examples": ex}}
        lint = schema_lint(q, states=[state], scorer=scorer)
        if scorer is None:
            scorer = lint["recommended_scorer"]
        out = self.systemone_decide(state, q, labeled=labeled, margin=margin, encoder="ngram", scorer=scorer,
                                    conformal_alpha=conformal_alpha, reflex=reflex, escalate=escalate)
        from holographic.agents_and_reasoning.holographic_decisionrecord import door_record
        a = door_record(dict(out[question]), "typed")  # E0.6: dict() would drop the p deprecation; re-wrap
        a["lint"] = [f for f in lint["findings"] if f["level"] != "note" or "clause" in f["what"] or "contrastive" in f["what"]]
        a["scorer"] = scorer
        return a

    def systemone_batch_fdr(self, states, questions, question, alpha=0.10, n_null=200, encoder="perceive",
                            scorer="prototype", nb_bigrams=False, margin=None):
        """BATCH FALSE-DISCOVERY CONTROL over a stream of typed decisions (sweep 176, H2): shuffle-null p per state
        (in-vocabulary word salad at matched length), Benjamini-Hochberg across the batch (holographic_ablate.bh_fdr).
        Measured: BH holds FDR 0.03 at nominal 0.05 / 0.10 while the naive per-test gate lets noise through at
        0.14 / 0.22; BY accepts nothing (kept negative). Power is low (13-20% of real rows) by construction.
        See holographic_systemone.SystemOne.batch_fdr."""
        from holographic.agents_and_reasoning.holographic_decisionrecord import door_record
        so = self.systemone(questions, margin=margin, encoder=encoder, scorer=scorer, nb_bigrams=nb_bigrams)
        r = so.batch_fdr(states, question, alpha=alpha, n_null=n_null)
        # E0.6: these are NULL p-values (low = significant) -> p_null; there is no P(correct) here
        return door_record(r, "typed-fdr", p_correct=None, p_null=list(r["p"]))

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
        out = so.absorb_unlabeled(states, question, iters=iters, lam=lam, max_options=max_options)
        if out.get("applied"):
            so._learned = True      # E0.5: absorbed tables are learned state -- learning_save must persist them
        return out

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
        are published on this mind's MessageBus under `topic`. Returns the published record dict.
        E5.2: every verified step -- passed OR failed -- is a labelled example for the tool's verifier prototypes
        (verify_precheck orders candidates by them); the pre-check's prediction is recorded in meta['precheck'] and
        NEVER replaces the verify, which runs on every step that carries one."""
        if not done_when or evidence is None:
            raise ValueError("swarm_step refuses a step without done_when and evidence -- a step nobody can verify is not a step")
        verified = None
        precheck = None
        if verify is not None:
            verb = getattr(self, str(verify.get("verb", "")), None)
            if verb is None:
                raise ValueError("swarm_step: verify.verb %r is not a mind verb" % verify.get("verb"))
            # E5.2 -- THE VERIFIER'S PRE-CHECK SCORE, taken BEFORE the verify runs (so its calibration label is honest,
            # never fitted on the result it is about to see). It is recorded, never obeyed: the verify below ALWAYS
            # runs, whatever the pre-check predicts (pinned by tests/test_verifier.py).
            if isinstance(state, str):
                try:
                    pre = self._verifier_score(self._verifier_vec(state), tool)
                    precheck = {"score": pre, "p_correct": self._door_p("verify", pre)}
                except Exception:
                    precheck = None
            result = verb(**(verify.get("args") or {}))
            ok = (bool(expect(result)) if callable(expect) else (result == expect if expect is not None else True))
            # E5.2 -- BOTH outcomes teach the verifier prototypes. Before this, only a PASSED verify taught anything
            # (via decision_outcome below); a failed verify raised with no trace of what it had learned about the
            # tool. It is now a labelled FAILURE example for this tool's failure prototype -- the verifier's
            # prototypes only, NOT the reflex trace or its failure field (region-wide: one refused tool would silence
            # every later, correct tool on the task -- the trap below, kept).
            try:
                self._verifier_learn(state, tool, ok, None if precheck is None else precheck["score"])
            except Exception:
                pass                                    # a learning error never changes whether a step is accepted
            if not ok:
                raise ValueError("swarm_step refused: verification %r returned %r, expected %r"
                                 % (verify.get("verb"), str(result)[:200], expect if not callable(expect) else "expect(result) True"))
            verified = {"verb": verify.get("verb"), "passed": True}
        from holographic.agents_and_reasoning.holographic_decisionrecord import DecisionRecord
        meta = {"args": args or {}, "done_when": done_when, "evidence": evidence, "worker": worker,
                "verify": verify, "verified": verified}
        if precheck is not None:
            meta["precheck"] = precheck                 # what the verifier predicted before the verify ran (E5.2)
        rec = DecisionRecord(state, "step:" + str(tool), [str(tool)], str(tool), "swarm", outcome=outcome, meta=meta)
        self.decision_ledger().add(rec)
        # THE JUDGE TEACHES THE REFLEX (sweep 177; Moose: "the reflex arc isn't connected to the learning stuff").
        # Before this, a step the harness had just VERIFIED went to the ledger and the bus and nowhere else --
        # the only thing the reflex ever learned from was decision_outcome, i.e. whatever string a caller
        # reported. Now a step whose verification RAN AND PASSED is reported through the one outcome path
        # (decision_outcome -> Ledger.report -> reflex_learn), so the swarm's experience is built only from
        # judged results. Deliberately NOT done:
        #   * an unverified step (verify=None) does not teach -- its outcome is self-reported, the thing we fix;
        #   * a REFUSED step does not mark the failure field -- it raised above, before any record exists, and
        #     the failure field is region-wide: one refused tool would silence the reflex for every later,
        #     correct tool on the same task (the same trap the second cut of reflex_learn fell into).
        # A caller-supplied `outcome` that disagrees with the tool is respected as a CORRECTION (it is written).
        learned = None
        if verified is not None:
            rep = self.decision_outcome(rec.id, str(outcome) if outcome is not None else str(tool))
            learned = rep.get("reflex")
        payload = rec.to_dict()
        payload["reflex"] = learned                     # None when the step was not verified
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

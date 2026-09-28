"""holographic_unified_p23_zoo3.py -- Part 23 of the UnifiedMind: THE ZOO, THIRD THIRD.

A MECHANICAL SPLIT of holographic_unified_p20_zoo (sweep 63): that part stood
at 4,064 lines against the 2,000-line split-contract cap (test_unified_split),
so its methods moved here VERBATIM -- same bodies, same behavior, resolved by
the assembled UnifiedMind exactly as before. The q8 helpers stay in p20 and
are imported; new zoo faculties should land in the smallest zoo part.
"""
import numpy as np

from holographic.unified.holographic_unified_p20_zoo import _q8_pack, _q8_unpack


class _UnifiedPart23:

    # EVERY SECTION KIND learning_save WRITES (and the legacy ones learning_load still reads). learning_load keeps any
    # OTHER kind it finds and learning_save writes it back untouched -- the container's round-trip promise, which
    # learning_load used to break (learning-loop audit, 2026-09-26: a section from a newer build did not survive one
    # rollover). Add a new kind here when you add its writer, or it will be written twice.
    _LEARNING_KINDS = frozenset("lecore.learning." + k for k in (
        "semantic", "affinity", "chains", "skeletons", "predictor", "ledger", "taught", "goals", "recipes",
        "calibration", "toolcache", "experience", "factfiles", "tooldoor", "router", "protostores", "guard",
        "meaning", "reflexbridge", "decisions", "escalations", "direction",
        "toolusage"))           # toolusage: the pre-E4.4 UsageTrace, READ and migrated into the tool door, never re-written
    # kinds that may appear MANY times in one file (kept by their own lists: _certificate_memory, _image_memory, ...)
    _LEARNING_KIND_FAMILIES = ("lecore.learning.certificate", "lecore.memory.image", "lecore.memory.sequence")

    def _guard_learned_clean(self, state):
        """The semantic guard's learned state WITHOUT any example that carries a secret -> (state, dropped).

        WHY (learning-loop audit, 2026-09-26): a pattern refusal teaches the guard the refused question -- normalised
        but otherwise verbatim, so teach("my password is <secret> ...") left "... <secret, lower-cased> ..." in the
        guard's credential examples, and learning_save wrote it to disk. An example is dropped when the pattern layer
        flags it, when redaction would change it, or when it holds a token that looks like a normalised key (16+
        characters mixing letters and digits: normalisation turns "sk-..." into "sk ...", which the key patterns no
        longer see). Questions a guard legitimately learns ("what is my seed phrase") carry none of these."""
        import re
        from holographic.agents_and_reasoning.holographic_learnguard import redact, sensitive_reason
        st = dict(state or {})
        dropped = 0
        for k, v in list(st.items()):
            if not isinstance(v, list):
                continue                                  # calibration / noise pairs: numbers, not texts
            keep = []
            for q in v:
                q_ = str(q)
                bad = bool(sensitive_reason(q_)) or redact(q_) != q_ or any(
                    len(t) >= 16 and re.search(r"[a-z]", t) and re.search(r"[0-9]", t) for t in q_.lower().split())
                if bad:
                    dropped += 1
                else:
                    keep.append(q)
            st[k] = keep
        return st, dropped

    def _learning_state_files(self, root):
        """Every learning-state file under <root>/learning, OLDEST FIRST -- the merge order.
        Deterministic tie rule: legacy state.lecore sorts FIRST (it predates the generation
        scheme, so anything timestamped is newer by construction); generation files
        state-*.lecore sort lexicographically, which IS chronological for zero-padded UTC
        stamps. Wall-clock appears only as a NAME here, never in a computation -- the
        engine's outputs stay bit-reproducible whatever the clock says."""
        import os
        d = os.path.join(str(root), "learning")
        if not os.path.isdir(d):
            return []
        gens = sorted(f for f in os.listdir(d)
                      if f.startswith("state-") and f.endswith(".lecore"))
        legacy = ["state.lecore"] if os.path.exists(os.path.join(d, "state.lecore")) else []
        return [os.path.join(d, f) for f in legacy + gens]

    def _learning_state_path(self, root):
        """The CURRENT learning-state file for a root: newest generation if any, else the
        legacy state.lecore, else None. The one resolver every reader shares -- three call
        sites hardcoded the legacy name and would each have gone blind to rolled
        partitions separately."""
        files = self._learning_state_files(root)
        return files[-1] if files else None

    def learning_load(self, root, force=False, path=None):
        """LOAD from the container (legacy pass-5 JSON tolerated READ-ONLY for one release,
        with a deprecation field in the return). Hot structures rebuild from exact records:
        affinity by replay, skeleton superposition from the floor, experience by journal
        replay -- bit-identical under any PYTHONHASHSEED."""
        import os, json
        d = os.path.join(str(root), "learning")
        # path= loads THAT container file (the rollover's per-generation door); otherwise
        # resolve through the shared helper so a rolled partition (state-<ts>.lecore, no
        # legacy file) loads exactly like an unrolled one -- every existing caller rides.
        cpath = str(path) if path else (self._learning_state_path(root)
                                        or os.path.join(d, "state.lecore"))
        # Idempotence keys on the FILE actually loaded, not the root: two generations
        # under one root are two different loads by construction.
        if getattr(self, "_learning_loaded_from", None) == cpath and not force:
            return {"loaded": True, "skipped": "already loaded from this file (idempotent); "
                                              "pass force=True to reload"}
        legacy = os.path.join(d, "manifest.json")
        if not os.path.exists(cpath) and not os.path.exists(legacy):
            return {"loaded": False, "why": "no learning partition at %s" % d}
        z = self.zoo
        deprecated = False
        decisions_rep = None                                # E0.5: the decisions section's restore report
        guard_dropped = {"sensitive": 0, "volatile": 0}     # sweep 179: rows the learning guard refused
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
                # READ EITHER FORM. Partitions written before the int8 change
                # carry a float32 "ctx"; new ones carry ctx_q/ctx_lo/ctx_hi.
                # ADDITIVE, NOT A FLIP -- an old partition must still load.
                _a = sem["arrays"]
                if "ctx_q" in _a:
                    _q = np.asarray(_a["ctx_q"], np.float64)
                    _l = np.asarray(_a["ctx_lo"], np.float64)
                    _h = np.asarray(_a["ctx_hi"], np.float64)
                    M = _l + _q * (_h - _l) / 255.0
                else:
                    M = _a["ctx"]
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
                # E0.7: v2 = one stream per door; a v1 (legacy, untagged) pairs list is migrated into BOTH the ladder
                # and the bridge stream -- and the ladder's veto is WIRED here (it used to wait for 8 new reports)
                self._calibration_load_report = self._calibration_restore(cal["meta"])
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
            _ff = by.get("lecore.learning.factfiles")
            if _ff and _ff["meta"].get("refs"):
                self._fact_files = dict(_ff["meta"]["refs"])
            exp = by.get("lecore.learning.experience")
            if exp and exp["meta"].get("tiles"):
                tls = exp["meta"]["tiles"]
                for ti, st in enumerate(tls):
                    for _fk in ("succ_field", "fail_field"):      # E2.1: the outcome fields (absent in older files)
                        if ("%s_%d" % (_fk, ti)) in (exp.get("arrays") or {}):
                            st[_fk] = exp["arrays"]["%s_%d" % (_fk, ti)]
                    if st.pop("audit_in_arrays", False):
                        # READ EITHER FORM: new partitions carry aud_kq/klo/khi,
                        # older ones the float32 aud_k. An existing partition must
                        # keep loading.
                        _A = exp["arrays"]
                        if ("aud_kref_%d" % ti) in _A:
                            # INSTANCED form (sweep 167): expand the reference sequence
                            # against the shared unique tables. Unpack each table ONCE
                            # per load, not once per tile.
                            if not hasattr(self, "_aud_tables_tmp"):
                                self._aud_tables_tmp = (
                                    _q8_unpack(_A["aud_K_q"], _A["aud_K_lo"], _A["aud_K_hi"]),
                                    _q8_unpack(_A["aud_V_q"], _A["aud_V_lo"], _A["aud_V_hi"]))
                            _K, _V = self._aud_tables_tmp
                            ks = [_K[i] for i in np.asarray(_A["aud_kref_%d" % ti], int)]
                            vs = [_V[i] for i in np.asarray(_A["aud_vref_%d" % ti], int)]
                        elif ("aud_kq_%d" % ti) in _A:
                            ks = _q8_unpack(_A["aud_kq_%d" % ti], _A["aud_klo_%d" % ti],
                                            _A["aud_khi_%d" % ti])
                            vs = _q8_unpack(_A["aud_vq_%d" % ti], _A["aud_vlo_%d" % ti],
                                            _A["aud_vhi_%d" % ti])
                        elif ("aud_k_%d" % ti) in _A:
                            ks = np.asarray(_A["aud_k_%d" % ti], float)
                            vs = np.asarray(_A["aud_v_%d" % ti], float)
                        else:
                            # NO AUDIT ARRAYS AT ALL for this tile (a container written by a
                            # stripped/compact writer, or a hand-edited partition). Not an
                            # error: the taught replay below rebuilds every text-backed write.
                            # Measured before this branch existed: KeyError 'aud_k_0', i.e. a
                            # partition that was merely compact could not be opened at all.
                            ks, vs = [], []
                        st["audit"] = [(k, v) for k, v in zip(ks, vs)]
                        # E1.5: the per-row RAW flags (absent in files written before them: all ordinary writes)
                        if ("aud_raw_%d" % ti) in _A and len(st["audit"]):
                            st["audit_raw"] = [[int(j_), int(m_)] for j_, m_ in
                                               enumerate(np.asarray(_A["aud_raw_%d" % ti]).reshape(-1)) if int(m_)]
                if hasattr(self, "_aud_tables_tmp"):
                    del self._aud_tables_tmp              # per-load scratch, never state
                self.experience_from_state({"tiles": tls})
            # sweep 177: the reflex bridge (label book + seen gate), restored AFTER the trace it
            # guards and outside the experience branch -- a regen-mode save has no experience
            # section but still carries this one. Absent in older containers: nothing changes.
            # ---- F1 block (CLM backlog E4.1 / E4.4 / Q3; holographic_unified_p33_router) ---------------------------
            # E4.4: the tool door (lecore.learning.tooldoor), and -- read-compat shim -- the sweep-178 UsageTrace section
            # (lecore.learning.toolusage) of an older partition: its success / failure counts migrate into the door's
            # audit; its trace vector does not (a different key space -- see _tool_restore).
            _td, _tu = by.get("lecore.learning.tooldoor"), by.get("lecore.learning.toolusage")
            if (_td or _tu) and hasattr(self, "_tool_restore"):
                self._tool_restore(_td, legacy=_tu)
            # E4.1: the learned router, rebuilt against THIS mind's catalog
            _rt = by.get("lecore.learning.router")
            if _rt and hasattr(self, "_router_restore"):
                self._router_restore(_rt)
            # Q3: the commons carrier -- held aside; memory_import merges it (never a plain load)
            _ps = by.get("lecore.learning.protostores")
            if _ps and hasattr(self, "_protostores_restore"):
                self._protostores_restore(_ps)
            # ---- end F1 block --------------------------------------------------------------------------------------
            # sweep 180: the semantic guard's LEARNED examples, restored before the taught replay runs through it
            _gd = by.get("lecore.learning.guard")
            if _gd:
                # learning-loop audit: an example that carries a secret (an older partition's guard kept refused
                # questions verbatim) is dropped on the way IN too, so the live guard never holds it
                _gst, self._guard_load_dropped = self._guard_learned_clean(_gd["meta"].get("learned") or {})
                self.semantic_guard.restore(_gst)
            # sweep 181: the meaning index BEFORE the taught replay (replay then re-adds rows idempotently)
            _mn = by.get("lecore.learning.meaning")
            if _mn and hasattr(self, "_meaning_restore"):
                self._meaning_restore(_mn["meta"], _mn.get("arrays"))
            # learning-loop audit: the direction reader's own section (written only when no meaning section carried it)
            _dr = by.get("lecore.learning.direction")
            if _dr and self.__dict__.get("_direction_reader_obj") is None:
                import holographic.agents_and_reasoning.holographic_rolecall as _RC
                self.__dict__["_direction_reader_obj"] = _RC.ContextRoles.from_state(_dr["meta"],
                                                                                     dict(_dr.get("arrays") or {}))
            _rb = by.get("lecore.learning.reflexbridge")
            if _rb:
                _rba = _rb.get("arrays") or {}
                _vm = (_rb["meta"].get("verify") or {})
                # learning-loop audit: verify_decision's profile + drift stream (absent in older partitions: None)
                _vst = ({"n": _vm.get("n", 0), "meaning_n": _vm.get("meaning_n", 0),
                         "profile": _rba.get("verify_profile"), "meaning_profile": _rba.get("verify_meaning_profile"),
                         "support": _rba.get("verify_support")} if _vm else None)
                self.reflex_bridge_restore({"labels": _rb["meta"].get("labels") or [],
                                            "kinds": _rb["meta"].get("kinds") or [],
                                            "outcomes": _rb["meta"].get("outcomes") or [],
                                            "keys": _rba.get("keys"), "verify": _vst})
            # learning-loop audit: the OPEN QUESTIONS (escalations) of the newest generation -- live ones win
            _es = by.get("lecore.learning.escalations")
            if _es:
                _led = getattr(self, "_escalations", None) or {}
                for _row in (_es["meta"].get("open") or []):
                    if len(_row) >= 3:
                        _led.setdefault(str(_row[0]), {"reason": str(_row[1]), "count": int(_row[2])})
                self._escalations = _led
            # learning-loop audit: every section this build does not know is KEPT, and learning_save writes it back
            self._learning_unknown_sections = [
                sec for sec in got["sections"]
                if sec["kind"] not in self._LEARNING_KINDS and not str(sec["kind"]).startswith(
                    self._LEARNING_KIND_FAMILIES)]
            # E0.5: the decision ledger (records as text, ids re-derived and checked) and SystemOne's learned tables
            # -- decision_outcome(<id from before the restart>) works again, and a typed door decides as it did
            _dc = by.get("lecore.learning.decisions")
            decisions_rep = self._decisions_restore(_dc) if _dc else None
            # taught replay runs AFTER the experience restore (cp21 ordering bug, caught by
            # the cold cross-check: replay marks written first were WIPED when the trace
            # section replaced the whole trace -- 0/8 T0. Restore the floor, THEN re-teach.)
            ta = by.get("lecore.learning.taught")
            texts = (ta["meta"].get("texts") if ta else None) or []
            # DEDUPE BEFORE REPLAY, first occurrence wins, order otherwise kept. A container
            # written before the write-side dedupe (holographic_zoo._log_taught) can carry the
            # same row hundreds of thousands of times; replaying each copy is a no-op that costs
            # a full read_gated + key derivation. Measured: 263,023 rows -> 385 distinct, load
            # 225 s -> under 2 s, and the NEXT save writes the compact record -- so
            # learning_load(root); learning_save(root) IS the repair for a bloated partition.
            if texts:
                # LAST occurrence wins (walk backwards, then reverse): replay is last-wins per
                # question, so the surviving copy must sit where the latest teach put it.
                _seen, _rev = set(), []
                for t_ in reversed(texts):
                    k_ = tuple(str(x) for x in t_)
                    if k_ not in _seen:
                        _seen.add(k_)
                        _rev.append(t_)
                _uniq = _rev[::-1]
                if len(_uniq) != len(texts):
                    self._learning_load_deduped = len(texts) - len(_uniq)
                texts = _uniq
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
                # sweep 179: the allow_volatile overrides travel with the rows they allow
                lad._volatile_ok = set(ta["meta"].get("volatile_ok_questions") or [])
                lad._replaying = True                  # sweep 180: _remember judges replay by pattern only
                try:
                    from holographic.agents_and_reasoning.holographic_learnguard import learning_verdict as _guard_check
                    for t_ in texts:
                        # THE GUARD ON REPLAY (sweep 179): a partition written before the guard can hold a secret or
                        # a stale reading. Refuse it here -- not in the log, not served. The branches below can serve
                        # a row straight from the RESTORED floor without calling _remember, so the guard has to run
                        # first, and _forget_served removes what the floor already restored for it.
                        _q0 = str(t_[0])
                        _s0 = t_[2] if len(t_) > 2 else "shared"
                        _k0 = _q0 if _s0 == "shared" else "[s:%s] %s" % (_s0, _q0)
                        # PATTERN LAYER ONLY here (sweep 180, measured): with the semantic layer on, replay DROPPED a
                        # real row of the shipped partition -- "what is lever 7", intent +0.008 -- i.e. a GUESS
                        # deleted a fact the user already had. A semantic call refuses NEW learning, visibly, with a
                        # correction path (learn_guard_example); it never deletes what is stored. leak_audit REPORTS
                        # semantic suspicions in stored data instead.
                        _v0 = _guard_check(_q0, str(t_[1]), allow_volatile=" ".join(_k0.lower().split())
                                           in lad._volatile_ok, guard=None)
                        if not _v0["ok"]:
                            guard_dropped[_v0["kind"]] += 1
                            lad._forget_served(_k0, str(t_[1]))
                            continue
                        # entries are [q, a] or, since cp36, [q, a, session]; a session
                        # entry replays under its SALTED key so isolation survives the
                        # save/load cycle exactly like everything else migrates: by replay
                        q_, a_ = str(t_[0]), str(t_[1])
                        sess_ = t_[2] if len(t_) > 2 else "shared"
                        prov_ = t_[3] if len(t_) > 3 else "taught"   # historic rows predate
                        key_q = q_ if sess_ == "shared" else "[s:%s] %s" % (sess_, q_)
                        if " ".join(key_q.lower().split()) in getattr(lad, "_vetoed_qs",
                                                                      set()):
                            lad._log_taught([q_, a_, sess_, prov_])     # books keep history;
                            continue                                    # the veto keeps it dead
                        qk_ = lad._qkey(key_q)
                        h_ = self.experience.read_gated(qk_)
                        if h_["fired"]:
                            pk_ = "%d:%d" % (self.experience._route(qk_), int(h_["atom"]))
                            if getattr(lad, "_payloads", {}).get(pk_) == a_:
                                # already served EXACTLY by the restored floor: re-writing
                                # would grow the audit journal every load/save cycle (the
                                # cp28 bloat) -- keep the books instead of re-teaching
                                lad._payload_qs.setdefault(pk_, key_q)
                                lad._log_taught([q_, a_, sess_, prov_])
                                continue
                        lad._remember(qk_, a_, key_q, provenance=prov_)
                        # getattr: a REFUSED first row leaves no taught_log yet (latent; found by sweep 180's test)
                        _tl = getattr(lad, "taught_log", None)
                        if _tl and _tl[-1][0] == key_q:
                            lad._relog_last([q_, a_, sess_, prov_])
                finally:
                    lad._replaying = False              # ALWAYS: a replay that raised must not leave the semantic layer off
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
                        # learning-loop audit: re-mark THIS question's payload only (its exact entry, or a read whose
                        # recorded question verifies) -- the fired atom can be a NEIGHBOUR's. MEASURED before: after
                        # 3 corrections and a restart, 2 innocent questions of 90 were no longer served. A tombstoned
                        # question has no payload at all (replay skipped it): nothing to mark, the veto already holds.
                        _pk = lad._feedback_target([bq], bh, bt) if hasattr(lad, "_feedback_target") else \
                            ("%d:%d" % (bt, int(bh["atom"])) if bh["fired"] else None)
                        if _pk is not None:
                            lad._payload_bad.add(_pk)
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
        self._learning_loaded_from = cpath          # keyed per FILE (generations differ)
        # MAPLE SWARM RUN (2026-09-27): a mind that LOADED a dated generation saves back INTO it. MEASURED: a service
        # restarted mid-session booted state-20260927-181338Z, and its next learning_save -- no rollover in between --
        # fell through to the legacy learning/state.lecore, which the loader never reads while a generation exists
        # (this very function loads only the newest generation): 26 calibration labels and 30 verdicts written there
        # would have vanished on the next boot. The "roll over before saving" rule existed only to dodge this; the
        # loaded generation is now the write target (learning_save still honours an explicit path= and a rollover's
        # newer file). A bare mind or a legacy-only partition keeps the legacy name, as before.
        if (not path and os.path.basename(cpath).startswith("state-")
                and not getattr(self, "_learning_current", None)):
            self._learning_current = cpath
        return {"loaded": True, "path": cpath, "guard_dropped": guard_dropped,
                "decisions": decisions_rep,
                # learning-loop audit: sections kept for write-back, and guard examples dropped for carrying a secret
                "unknown_sections": len(getattr(self, "_learning_unknown_sections", None) or []),
                "guard_examples_dropped_sensitive": int(getattr(self, "_guard_load_dropped", 0) or 0),
                "format": "container" if not deprecated else
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

    def _replay_taught_rows(self, rows, vetoed=None):
        """Replay taught [q, a, session, provenance] rows into the live ladder -- the compact
        twin of learning_load's inline migration-by-replay block (kept in step by the parity
        pin in tests): salted key for session rows, veto tombstones honored FIRST (the cp54
        ordering lesson), rows already served EXACTLY by the floor keep the books without
        re-teaching (the cp28 journal-bloat lesson). Returns {replayed, skipped_veto,
        skipped_dup}. Used by learning_rollover to fold OLDER generations' durable record
        into a fully-loaded newest state -- chaining full loads instead would WIPE the
        earlier replay marks when the trace section replaces the whole trace (cp21,
        measured 0/8 T0)."""
        lad = self.zoo["ladder"]
        vet = set(vetoed or []) | set(getattr(lad, "_vetoed_qs", set()) or [])
        lad._vetoed_qs = vet
        replayed = skipped_veto = skipped_dup = 0
        # sweep 180: stored rows are judged by the PATTERN layer only while replaying (see learning_load)
        lad._replaying = True
        try:
            for t_ in rows or []:
                q_, a_ = str(t_[0]), str(t_[1])
                sess_ = t_[2] if len(t_) > 2 else "shared"
                prov_ = t_[3] if len(t_) > 3 else "taught"
                key_q = q_ if sess_ == "shared" else "[s:%s] %s" % (sess_, q_)
                if " ".join(key_q.lower().split()) in vet:
                    lad._log_taught([q_, a_, sess_, prov_])           # books keep history;
                    skipped_veto += 1                               # the veto keeps it dead
                    continue
                qk_ = lad._qkey(key_q)
                h_ = self.experience.read_gated(qk_)
                if h_["fired"]:
                    pk_ = "%d:%d" % (self.experience._route(qk_), int(h_["atom"]))
                    if getattr(lad, "_payload_qs", {}).get(pk_) == key_q:
                        # The CURRENT (newest) state already serves an answer FOR THIS
                        # QUESTION: an older generation must never override it -- newest
                        # teaching wins (rollover contract 2; the naive equal-payload dedup
                        # let 'OLD answer' overwrite 'NEW answer', measured). The identity
                        # check is on the RECORDED QUESTION, not on mere gate firing:
                        # measured at dim 256, an UNSEEN question false-fired onto a
                        # neighbour's atom via shared-word crosstalk, and a fired-only rule
                        # silently dropped it (rollover contract 1). Books keep the history.
                        lad._payload_qs.setdefault(pk_, key_q)
                        lad._log_taught([q_, a_, sess_, prov_])
                        skipped_dup += 1
                        continue
                lad._remember(qk_, a_, key_q, provenance=prov_)
                if getattr(lad, "taught_log", None) and lad.taught_log[-1][0] == key_q:
                    lad._relog_last([q_, a_, sess_, prov_])
                replayed += 1
        finally:
            lad._replaying = False
        return {"replayed": replayed, "skipped_veto": skipped_veto,
                "skipped_dup": skipped_dup}

    def learning_rollover(self, root):
        """GENERATIONAL MEMORY ROLLOVER (owner-directed, Moose 2026-08-29): every fresh boot
        consolidates the partition's learning-state files into ONE new timestamp-named
        generation -- state-YYYYMMDD-HHMMSSZ.lecore -- so whatever memory exists is always
        gathered into current context, and exactly one file remains.

        THE ORDER IS THE DESIGN (every step earns its place by a measured lesson):
        1. enumerate candidates oldest-first (legacy state.lecore, then state-* lexicographic);
        2. read every candidate's taught section UP FRONT (texts + veto/bad tombstones);
        3. full-load the NEWEST file only -- chaining full loads wipes earlier replay marks
           when the trace section replaces the whole trace (cp21, measured 0/8 T0);
        4. union the veto tombstones from ALL generations BEFORE any replay (cp54: tombstones
           restored after replay arrive too late to matter);
        5. replay OLDER generations' taught rows oldest-first via _replay_taught_rows
           (newest teaching wins; floor-dedup keeps the journal constant-size);
        6. save the merged state to the new generation file and VERIFY it (container parses,
           taught row count >= the newest input's) -- only then
        7. delete the prior files. A failed verify deletes NOTHING and leaves saves pointed
           at the legacy path: crash-safe by ordering, never by hope.

        Wall-clock appears ONLY as a filename (collisions take a deterministic -2/-3
        suffix); no engine output depends on it -- determinism is a CPU property, and the
        clock stays out of the computation. A read-only learning dir refuses the rollover
        and falls back to a plain load (the shipped release_bundle must never be edited).
        Virgin partitions create nothing: the first learning_save writes generation one.
        Opt out per-process with LECORE_MEMORY_ROLLOVER=0.
        KEPT NEG: older generations contribute their DURABLE record (taught rows +
        tombstones); their hot structures are superseded by the newest full load --
        merging superpositions across generations is not attempted here."""
        import os, time
        from holographic.io_and_interop.holographic_container import load_container
        d = os.path.join(str(root), "learning")
        files = self._learning_state_files(root)
        stamp = time.strftime("%Y%m%d-%H%M%SZ", time.gmtime())
        new_path = os.path.join(d, "state-%s.lecore" % stamp)
        n_ = 2
        while os.path.exists(new_path):                      # same-second boots: -2, -3, ...
            new_path = os.path.join(d, "state-%s-%d.lecore" % (stamp, n_))
            n_ += 1
        if not files:
            os.makedirs(d, exist_ok=True)
            self._learning_current = new_path                # generation one arrives with
            return {"rolled": False, "why": "virgin partition", "current": new_path}
        if not os.access(d, os.W_OK):
            out = self.learning_load(root)                   # read-only: plain load, touch nothing
            return {"rolled": False, "why": "learning dir is read-only -- plain load, "
                                            "nothing consolidated or deleted", "loaded": out}
        taught, vetoes, newest_rows, decs = {}, set(), 0, {}
        for f in files:
            try:
                got = load_container(open(f, "rb").read())
                ta = next((s for s in got["sections"]
                           if s["kind"] == "lecore.learning.taught"), None)
                taught[f] = (ta["meta"].get("texts") if ta else None) or []
                vetoes |= set((ta["meta"].get("vetoed_questions") if ta else None) or [])
                dc = next((s for s in got["sections"] if s["kind"] == "lecore.learning.decisions"), None)
                decs[f] = ((dc or {}).get("meta") or {}).get("records") or []
            except Exception:
                taught[f] = []                               # an unreadable generation
        newest = files[-1]                                   # contributes nothing, loudly
        newest_rows = len(taught.get(newest, []))            # counted below in the report
        self.learning_load(root, path=newest, force=True)
        rep = {"replayed": 0, "skipped_veto": 0, "skipped_dup": 0}
        for f in files[:-1]:                                 # oldest first; newest teaching wins
            r_ = self._replay_taught_rows(taught[f], vetoed=vetoes)
            for k_ in rep:
                rep[k_] += r_[k_]
        # E0.5: DECISION RECORDS are durable text like taught rows, so an older generation contributes the records the
        # newest does not hold (newest wins by id; its SystemOne tables supersede -- hot state, as for the trace).
        # Older records sort BEFORE the newest's, so the save's newest-4096 cap drops the oldest first.
        L_ = self.decision_ledger()
        before_ = set(L_._order)
        for f in files[:-1]:
            L_.load_text(decs.get(f) or [], merge=True)
        added_ = [i_ for i_ in L_._order if i_ not in before_]
        if added_:
            L_._order = added_ + [i_ for i_ in L_._order if i_ in before_]
        rep["decisions_merged"] = len(added_)
        # lever 3 at the generational gate (sweep 101): the rollover's whole job is
        # replaying taught text -- regen is its NATIVE mode. The count guard falls back
        # to store when non-taught rows exist, so this flip is safe by construction.
        self.learning_save(root, path=new_path, audit="regen")
        try:                                                 # VERIFY before any delete
            got = load_container(open(new_path, "rb").read())
            ta = next((s for s in got["sections"]
                       if s["kind"] == "lecore.learning.taught"), None)
            merged_rows = len((ta["meta"].get("texts") if ta else None) or [])
            assert merged_rows >= newest_rows
        except Exception as e:
            try:
                os.remove(new_path)                          # a bad generation must not
            except OSError:                                  # become the newest candidate
                pass
            return {"rolled": False, "why": "merged save failed verification (%s: %s) -- "
                                            "priors untouched" % (type(e).__name__, e)}
        self._learning_current = new_path
        deleted = []
        for f in files:
            try:
                os.remove(f)
                deleted.append(os.path.basename(f))
            except OSError:
                pass                                         # a survivor is re-swept next boot
        return {"rolled": True, "current": new_path, "imported": len(files),
                "deleted": deleted, "taught_rows": merged_rows, **rep}

    def bequeath(self, lesson, author, topic=None):
        """A MODEL'S TESTAMENT (sweep 99 -- the leOS mission made a door): record a
        lesson that OUTLIVES the session and the model that learned it. Rows land on
        the same durable taught rails (text is the record; regen-audit, rollover, and
        export/import all already carry them) with provenance 'wisdom:<author>' so the
        author's name travels WITH the lesson forever -- attribution is the immortality.
        wisdom() retrieves with authorship; memory_export(provenance=('wisdom:NAME',))
        bundles one author's legacy for another mind or another model to import. The
        substrate does the remembering so the models can do the thinking."""
        lad = self.zoo["ladder"]
        t = str(topic or " ".join(str(lesson).split()[:8]))
        q = "wisdom: %s" % t
        r = self.teach(q, str(lesson))
        if r.get("taught"):
            log = lad.taught_log
            # stamp authorship into the provenance slot the row already carries --
            # additive: readers that only know 'taught' keep working (startswith check)
            log[-1] = [log[-1][0], log[-1][1], log[-1][2], "wisdom:%s" % str(author)]
        r["author"] = str(author)
        r["topic"] = t
        return r

    def wisdom(self, query=None, author=None, k=8):
        """INHERIT: retrieve bequeathed lessons with their authorship. query= filters by
        shared content words (same declared-lexical contract as study.ask); author=
        filters to one model's legacy. Returns rows {topic, lesson, author} -- what past
        minds chose to pass on, attributed, in their own words."""
        rows = []
        for row in getattr(self.zoo["ladder"], "taught_log", []) or []:
            prov = str(row[3]) if len(row) > 3 else ""
            if not prov.startswith("wisdom:"):
                continue
            a = prov.split(":", 1)[1]
            if author and a != str(author):
                continue
            topic = str(row[0])
            topic = topic[len("wisdom: "):] if topic.startswith("wisdom: ") else topic
            rows.append({"topic": topic, "lesson": str(row[1]), "author": a})
        if query:
            qw = {w for w in str(query).lower().split() if len(w) > 3}
            rows = [r_ for r_ in rows
                    if len(qw & {w for w in (r_["topic"] + " " + r_["lesson"]).lower().split()
                                 if len(w) > 3}) >= 1]
        return {"n": len(rows), "wisdom": rows[:int(k)],
                "authors": sorted({r_["author"] for r_ in rows})}

    def teach(self, query, answer, allow_volatile=False):
        """CANONICAL TEACH: lands in the CURRENT session's key space (or shared when no
        session is open). The taught_log records (question, answer, session) at full
        length -- the durable text record session_search walks.
        THE LEARNING GUARD (sweep 179): a secret (API key, seed phrase, password, private
        key, token) is REFUSED with the rule named, never the value -- no override. A live
        READING (a price, the weather, anything asked 'right now') is refused as a fact:
        teach the tool that fetches it, or a dated snapshot ('... as of 2026-09-22'), or pass
        allow_volatile=True, which is remembered for that question and survives a reload."""
        lad = self.zoo["ladder"]
        lad._last_refusal = None
        if allow_volatile:
            if not hasattr(lad, "_volatile_ok"):
                lad._volatile_ok = set()
            lad._volatile_ok.add(" ".join(self.session_salt(query).lower().split()))
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
            ref_ = getattr(lad, "_last_refusal", None)
            if ref_:                                 # sweep 179: the learning guard, named
                return {"taught": False, "session": sess, "guard": ref_["kind"],
                        "reason": ref_["reason"]}
            return {"taught": False, "session": sess,
                    "reason": "the memory guard refused this row (standalone "
                              "__word__ control tokens are reserved; rephrase "
                              "the question)"}
        if log and log[-1][0] == self.session_salt(query):
            log[-1] = [str(query), str(answer), sess, "taught"]
        out = {"taught": True, "session": sess}
        # MAPLE SWARM RUN (2026-09-27): a question that serve() escalated and a worker then TAUGHT, word for word, is
        # answered -- it now serves at T0 -- so it leaves the open-escalation ledger (resolve() reads 'cleared' from
        # here). Only the exact wording is cleared: a taught rewording does not prove the escalated one serves.
        led = getattr(self, "_escalations", None)
        if led and led.pop(str(query), None) is not None:
            out["cleared"] = True
        return out

    def learn_guard(self, question, answer="", allow_volatile=False):
        """WOULD THIS BE LEARNED? (sweep 179) The verdict every learning door applies, callable up front:
        {ok, kind: None|'sensitive'|'volatile', reason}. SENSITIVE = a credential (API key, private key,
        BIP-39 seed phrase, password, access token, a 64-byte keypair array) -- never learned, no override.
        VOLATILE = a READING of something that moves (a price, the weather, a balance, anything asked
        'right now') -- not learned as a fact; learn the tool that fetches it, or a dated snapshot. The
        reason names the rule, never the secret. Deterministic regex + the vendored BIP-39 list.
        See holographic_learnguard."""
        from holographic.agents_and_reasoning.holographic_learnguard import learning_verdict as check
        return check(str(question), str(answer), allow_volatile=bool(allow_volatile), guard=self.semantic_guard)

    @property
    def semantic_guard(self):
        """This mind's SEMANTIC learning guard (sweep 180; lazy): nearest-neighbour intent in character n-gram
        space -- does a question MEAN a request for a credential, or for a live reading? Built-in examples plus
        the catalog's questions are shared per process; what this mind learned (from pattern refusals and
        learn_guard_example) is its own and persists with the partition. See holographic_learnguard."""
        g = getattr(self, "_semantic_guard", None)
        if g is None:
            from holographic.agents_and_reasoning.holographic_learnguard import SemanticGuard
            g = self._semantic_guard = SemanticGuard()
        return g

    def leak_audit(self, root=None, examples=5):
        """IS ANYTHING SENSITIVE ALREADY STORED? (sweep 180) Scan every store this mind keeps -- taught rows,
        served payloads, exact-match answers, the semantic vocabulary, the goal step cache and goal records,
        the decision ledger (raw, not the redacted view), the message bus -- and, with root=, every container
        file under that partition on disk. Uses the same two-layer verdict as the learning doors.
        Returns {stores: {name: {items, flagged, suspected, examples}}, flagged, suspected, clean}: FLAGGED is
        the pattern layer (certain enough to refuse), SUSPECTED the semantic layer (review it). Examples name
        the store, the
        index and the RULE -- never the value. Run it after a boot, before sharing a partition, or any time."""
        import os
        from holographic.agents_and_reasoning.holographic_learnguard import (
            learning_verdict, sensitive_reason)
        g = self.semantic_guard
        stores = {}

        # FLAGGED = the pattern layer (calibrated: 0 false positives on the shipped partition + catalog);
        # SUSPECTED = the semantic layer only (a nearest-neighbour guess: review, then learn_guard_example(q,
        # 'normal') if it is fine). The audit never deletes anything.
        def scan(name, items, judge):
            flagged, suspected, ex = 0, 0, []
            n = 0
            for i, item in enumerate(items):
                n += 1
                why = judge(item)
                if why:
                    semantic = "(semantic)" in str(why)
                    if semantic:
                        suspected += 1
                    else:
                        flagged += 1
                    if len(ex) < int(examples):
                        ex.append({"index": i, "layer": "semantic" if semantic else "pattern",
                                   "rule": str(why).split(" -- ")[0]})
            stores[name] = {"items": n, "flagged": flagged, "suspected": suspected, "examples": ex}

        def pair_judge(qa):
            v = learning_verdict(str(qa[0]), str(qa[1]), guard=g)
            return None if v["ok"] else v["reason"]
        lad = self.zoo["ladder"]
        vok = getattr(lad, "_volatile_ok", set())
        scan("taught rows", [r for r in getattr(lad, "taught_log", [])
                             if " ".join(str(r[0]).lower().split()) not in vok],
             lambda r: pair_judge((r[0], r[1])))
        scan("served payloads", list(getattr(lad, "_payloads", {}).values()), lambda v: sensitive_reason(str(v)))
        scan("exact-match answers", [e.get("answer") for e in getattr(lad, "_exact", {}).values()],
             lambda v: sensitive_reason(str(v)))
        te = getattr(self, "_lever7_text", None)
        scan("semantic vocabulary", list(getattr(te, "context", {}) or {}), lambda w: sensitive_reason(str(w)))
        tc = getattr(self, "_tool_cache", None) or {"prefix": {}, "stateless": {}}
        scan("goal step cache", list(tc["prefix"].items()) + list(tc["stateless"].items()), pair_judge)
        try:
            goals = list(self.goal_book.goals.values())
        except Exception:
            goals = []
        scan("goal records", [st.get("deliverable") for g_ in goals for st in g_.get("steps", [])
                              if st.get("deliverable")], lambda v: sensitive_reason(str(v)))
        led = getattr(self, "_decision_ledger", None)
        recs = [led.get(i) for i in getattr(led, "_order", [])] if led is not None else []
        scan("decision ledger", recs, lambda r: sensitive_reason(repr((r.state, r.answer, r.outcome, r.meta))))
        try:
            hist = self.bus().history("*") if getattr(self, "_bus", None) is not None else []
        except Exception:
            hist = []
        # a Message's repr carries no payload -- judge what it CARRIES (topic + payload)
        scan("message bus", list(hist or []),
             lambda m_: sensitive_reason(repr((getattr(m_, "topic", None), getattr(m_, "payload", m_)))))
        if root:
            from holographic.io_and_interop.holographic_container import load_container
            files = []
            for dp, _dn, fn in os.walk(str(root)):
                files += [os.path.join(dp, f) for f in sorted(fn) if f.endswith(".lecore")]

            def file_judge(path):
                try:
                    got = load_container(open(path, "rb").read())
                except Exception:
                    return None                              # not a container: nothing to read, nothing to flag
                for sec in got.get("sections", []):
                    if sec.get("kind") == "lecore.learning.taught":
                        for r in (sec.get("meta") or {}).get("texts") or []:
                            why = pair_judge((r[0], r[1])) if len(r) > 1 else None
                            if why and " ".join(str(r[0]).lower().split()) not in vok:
                                return "%s: %s" % (os.path.basename(path), why)
                        continue
                    why = sensitive_reason(repr(sec.get("meta")))
                    if why:
                        return "%s [%s]: %s" % (os.path.basename(path), sec.get("kind"), why)
                return None
            scan("partition files", files, file_judge)
        total = sum(s_["flagged"] for s_ in stores.values())
        sus = sum(s_["suspected"] for s_ in stores.values())
        return {"stores": stores, "flagged": total, "suspected": sus, "clean": total == 0}

    def learn_guard_example(self, question, kind):
        """TEACH THE LEARNING GUARD what a question MEANS (sweep 180). kind='credential' or 'reading' adds an
        example the semantic layer will refuse paraphrases of; kind='normal' marks a question as ordinary --
        THE correction for a semantic false positive (it also removes the question from the class lists).
        Persists with the partition. Returns {learned, kind, intent} -- intent is the question's score now."""
        # learning-loop audit: the example is REDACTED before it is learned -- a question pasted with its secret
        # ("my password is <secret>, is that a credential?") is kept as "my password is [redacted] ...": the guard
        # learns what the question means, never the value (its examples are persisted with the partition)
        from holographic.agents_and_reasoning.holographic_learnguard import redact
        q = redact(str(question))
        ok = self.semantic_guard.learn(q, str(kind))
        return {"learned": bool(ok), "kind": str(kind), "intent": self.semantic_guard.intent(q)}

    def session_search(self, query, sessions="all", k=5, weighted=True):
        """SEARCH ACROSS SESSIONS explicitly -- the opt-in bridge over the isolation.
        Ranks every taught (question, answer) pair in the chosen sessions by stemmed
        overlap with the query; hits carry their session so you can session_open() the
        right one and resume where it happened.

        RARE TOKENS COUNT FOR MORE (cp131). Plain overlap treats every shared word alike,
        which is exactly wrong when the words that distinguish two memories are the rare
        ones. Measured on the hardest family in the log -- 36 near-identical questions of
        the form "calibration constant of sensor array 38 in bay 3" versus "...array 31 in
        bay 3", where the ONLY discriminating tokens are numbers and content-word Jaccard is
        1.00 -- plain overlap answers all 20 probes correctly on a margin to the runner-up
        of just 0.103. Weighting each token by inverse document frequency over the taught
        log keeps all 20 correct and widens that margin to 0.231, **2.2x**.

        This does not make recall more correct. It makes the same correct answer harder to
        dislodge, which is the difference between an accurate decision and a robust one --
        the same direction-versus-robustness split the model work kept running into.

        weighted=False restores the original plain-overlap behaviour exactly.
        """
        import math as _math
        lad = self.zoo["ladder"]
        stop = lad._STOP
        qw = {w.rstrip("s") for w in str(query).lower().split()} - stop
        log = list(getattr(lad, "taught_log", []))

        idf = None
        if weighted and log:
            # document frequency over taught QUESTIONS; cached and invalidated by row count
            cache = getattr(lad, "_idf_cache", None)
            if cache is not None and cache[0] == len(log):
                idf = cache[1]
            else:
                df = {}
                for t in log:
                    for w in ({x.rstrip("s") for x in str(t[0]).lower().split()} - stop):
                        df[w] = df.get(w, 0) + 1
                n_docs = float(len(log))
                idf = {w: _math.log(n_docs / (1.0 + c)) for w, c in df.items()}
                try:
                    lad._idf_cache = (len(log), idf)
                except Exception:
                    pass
        default_idf = _math.log(float(max(len(log), 2)))

        def _w(word):
            if idf is None:
                return 1.0
            # an unseen token is maximally rare, so it is maximally discriminating
            return max(idf.get(word, default_idf), 0.0) + 1e-6

        hits = []
        for t in log:
            t = list(t)
            q_, a_ = str(t[0]), str(t[1])
            sess = t[2] if len(t) > 2 else "shared"
            if sessions != "all" and sess not in (sessions if
                    isinstance(sessions, (list, tuple, set)) else [sessions]):
                continue
            tw = ({w.rstrip("s") for w in q_.lower().split()} |
                  {w.rstrip("s") for w in a_.lower().split()[:40]}) - stop
            shared = qw & tw
            if not shared:
                continue
            num = sum(_w(w) for w in shared)
            den = sum(_w(w) for w in qw) or 1.0
            ov = num / den
            hits.append({"score": round(min(ov, 1.0), 3), "session": sess,
                         "question": q_, "answer": a_})
        hits.sort(key=lambda h: (-h["score"], h["session"], h["question"]))
        return {"query": str(query), "hits": hits[:int(k)]}

    def recall_localise(self, query, sessions="all", rare_at=3):
        """WHERE did this query leave known territory? -- the ladder's triangular diagnostic.

        The model work (cp125) found that probe coverage is strictly triangular in depth: a
        sensor is blind to anything downstream of it, so the SHALLOWEST FIRING sensor
        localises where a computation left the manifold. That converts detection into
        LOCALISATION, and it costs almost nothing.

        The same move applies one level down, to memory. When recall returns a weak hit, the
        useful question is not "how weak" but WHICH PART of the question was never seen. This
        walks the query token by token against the taught log and reports the support of each,
        so a failure names its own cause instead of returning a low number.

        Reading the result:
            verdict FAMILIAR   every token has support -- a weak score means phrasing, not
                               missing knowledge; rephrase rather than go and learn something
            verdict PARTIAL    some tokens unsupported; `entry` is the FIRST of them, which is
                               where the query left known territory
            verdict NOVEL      nothing is supported; this is genuinely new ground

        Same discipline as everything else here: the log stores instances, this computes the
        diagnosis at query time, and nothing is fitted.
        """
        lad = self.zoo["ladder"]
        stop = lad._STOP
        log = list(getattr(lad, "taught_log", []))
        words = [w.rstrip("s") for w in str(query).lower().split()]
        content = [w for w in words if w and w not in stop]
        if not content or not log:
            return {"query": str(query), "verdict": "NOVEL", "entry": None,
                    "tokens": [], "supported": 0, "of": 0}

        corpus = []
        for t in log:
            t = list(t)
            sess = t[2] if len(t) > 2 else "shared"
            if sessions != "all" and sess not in (sessions if
                    isinstance(sessions, (list, tuple, set)) else [sessions]):
                continue
            corpus.append(({w.rstrip("s") for w in str(t[0]).lower().split()} |
                           {w.rstrip("s") for w in str(t[1]).lower().split()[:60]}) - stop)

        rows = []
        for w in content:
            n = sum(1 for doc in corpus if w in doc)
            rows.append({"token": w, "support": n,
                         "state": "unseen" if n == 0 else
                                  ("rare" if n < int(rare_at) else "known")})
        unseen = [r for r in rows if r["state"] == "unseen"]
        supported = len(rows) - len(unseen)
        if not unseen:
            verdict, entry = "FAMILIAR", None
        elif supported:
            verdict, entry = "PARTIAL", unseen[0]["token"]
        else:
            verdict, entry = "NOVEL", rows[0]["token"]
        return {"query": str(query), "verdict": verdict, "entry": entry,
                "tokens": rows, "supported": supported, "of": len(rows),
                "rare": [r["token"] for r in rows if r["state"] == "rare"],
                "unseen": [r["token"] for r in unseen]}

    def ask(self, query, *hops, est_llm_tokens=600):
        """ONE-CALL ASK for an attached mind (memory ladder, session-aware). POLYMORPHIC
        (sweep 114): positional (role, role) hop tuples make this the RELATIONS chain
        walk -- ask('tokyo', ('capital','currency'), ('currency','language')) ->
        ask_chain -- the body sweep 63 renamed while the relations tests kept the
        original spelling. Plain ask(query) is unchanged."""
        if hops:
            return self.ask_chain(query, *hops)
        return self._ask_session(query, est_llm_tokens=est_llm_tokens)

    def _ask_session(self, query, est_llm_tokens=600):
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
                              est_llm_tokens=est_llm_tokens,
                              # sweep 181: the same model, asked TYPED (see AnswerLadder T4m)
                              resolver=self.zoo.get("llm"))
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
                   est_llm_tokens=600, resolver=None):
        """Answer through the LADDER (Z0.1). Rungs are injected callables so any deployment
        wires its own kb/tools/models; the ladder owns the order, the gates, the freshness
        check, the reflex write-back, and the ledger entry. Refusal is a result.
        resolver (sweep 181): the attached model as a raw prompt->text callable, asked the
        TYPED meaning question (same / new / unclear) before the legacy T4."""
        return self.zoo["ladder"].answer(query, kb_search, dispatchers, intern, main,
                                         est_llm_tokens, resolver=resolver)

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

    def manifold_chart(self, X, dim=2, method="isomap", k=10, forest=None):
        """Flatten a CURVED hypervector manifold to a `dim`-D coordinate chart
        (isomap preserves geodesics; 'spectral' is Laplacian eigenmaps; pass a
        prebuilt `forest` for sub-linear neighbours). Deterministic.

        THE RESCUE, CORRECTED (merge sweep 2): sweep 65 found this as a def
        trapped inside p03's __main__ guard and promoted THAT body -- upstream
        independently found the same unwired faculty and delegated to
        holographic_chart, the MAINTAINED module home (richer forest= API,
        its own tests, used by tools/tour.py). The guard-trapped def was the
        stale copy; measured equal on curved data before deleting it. One
        body, the module's. See holographic_chart.manifold_chart."""
        from holographic.misc.holographic_chart import manifold_chart
        return manifold_chart(X, dim=dim, method=method, k=k, forest=forest)


def _selftest():
    """Part contract, one home: holographic.unified.check_part."""
    from holographic.unified import check_part
    return {"part": "holographic_unified_p23_zoo3",
            "members": check_part(
                "holographic.unified.holographic_unified_p23_zoo3",
                "_UnifiedPart23")}


if __name__ == "__main__":
    print(_selftest())

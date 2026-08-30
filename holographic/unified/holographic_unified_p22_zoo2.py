"""holographic_unified_p22_zoo2.py -- Part 22 of the UnifiedMind: THE ZOO, SECOND THIRD.

A MECHANICAL SPLIT of holographic_unified_p20_zoo (sweep 63): that part stood
at 4,064 lines against the 2,000-line split-contract cap (test_unified_split),
so its methods moved here VERBATIM -- same bodies, same behavior, resolved by
the assembled UnifiedMind exactly as before. The q8 helpers stay in p20 and
are imported; new zoo faculties should land in the smallest zoo part.
"""
import numpy as np

from holographic.unified.holographic_unified_p20_zoo import _q8_pack, _q8_unpack


class _UnifiedPart22:

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
                # lever 3 (sweep 101): checkpoints are written often, reloaded rarely
                # (crash recovery) -- exactly the disk-heavy, load-light profile regen
                # wins at. Guard falls back to store if non-taught rows exist.
                self.learning_save(checkpoint_root, audit="regen")
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
            _sp = self._learning_state_path(root) or os_.path.join(
                str(root), "learning", "state.lecore")   # helper: rolled roots resolve too
            got = load_container(open(_sp, "rb").read())
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
            # WHY: callers may pass ONE source for MANY texts (repo_map passes the
            # root once for a whole skeleton). A rigid sources[i] crashed on that
            # honest shape (sweep: openzoo session); short lists now pad with None
            # so the topic#i fallback names the note instead of an IndexError.
            src = (sources[i] if sources and i < len(sources) else None) \
                  or ("%s#%d" % (topic, i))
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
                    # THE JOURNAL IS ks.entries -- ks.evidence() returns the n-gram
                    # EvidenceStore (not iterable); iterating it raised into the
                    # bare except below and cross-session recall silently returned
                    # zero notes forever: the sweep 59-60 "absent result looks
                    # legit" class, in the memory system itself (sweep 62).
                    cache[root] = [str(e_.get("text", e_))
                                   for e_ in KnowledgeStore(root).entries]
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
            except Exception as e_:
                # A swallowed crawl failure is indistinguishable from an empty
                # archive -- the caller must be able to tell "nothing recorded"
                # from "the instrument broke" (sweep 62). Additive key, absent
                # on the healthy path.
                out["note_arm_error"] = "%s: %s" % (type(e_).__name__, str(e_)[:120])
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

    def partition_report(self, root, path=None):
        """WHERE THE PARTITION'S BYTES GO -- the no-code-safari door (sweep 95, built the
        day the question cost a safari): per-section compressed/raw byte census of a
        .lecore container, the fattest sections named, and per-fact growth cost when the
        taught count is recoverable. MEASURED diagnosis it was built on: growth was
        LINEAR at ~3,822 B/fact; 99% of bytes were the lever7 audit K/V hypervector rows
        (already int8-packed at 4.0x); the codec atlas certified they cannot compress
        further (high-entropy by CONSTRUCTION -- that is VSA working). KEPT NEG: more
        codecs is not the fix; the middle-out fix is lever 3 -- do not store what
        determinism regenerates (audit K/V are deterministic encodings of taught text the
        partition already carries and rollover already replays). See NOTES sweep 95 for
        the regen-audit design."""
        import os, zipfile
        p = str(path) if path else self._learning_state_path(root)
        if not p or not os.path.exists(p):
            return {"error": "no partition found under %r" % str(root)}
        z = zipfile.ZipFile(p)
        rows = sorted(((int(zi.compress_size), int(zi.file_size), zi.filename)
                       for zi in z.infolist()), reverse=True)
        total_c = sum(r[0] for r in rows)
        total_u = sum(r[1] for r in rows)
        out = {"path": p, "bytes": int(os.path.getsize(p)),
               "compressed": total_c, "raw": total_u,
               "sections": [{"name": n, "compressed": c, "raw": u,
                             "share": round(c / max(total_c, 1), 4)}
                            for c, u, n in rows[:10]]}
        n_taught = None
        try:
            n_taught = len(getattr(self.zoo["ladder"], "taught_log", []) or [])
        except Exception:
            pass
        if n_taught:
            out["taught_rows"] = int(n_taught)
            out["bytes_per_fact"] = round(total_c / n_taught, 1)
        out["advice"] = ("audit K/V dominate? codecs will not save you -- the vectors are "
                        "high-entropy by construction; the fix is regenerating them from "
                        "the taught text on load (lever 3). learning_compact drops "
                        "duplicate taught rows first.")
        return out

    def learning_save(self, root, path=None, audit="store"):
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
            _ctx = (np.stack([te.context[w] for w in words])
                    if words else np.zeros((0, te.dim)))
            _ctx = np.asarray(_ctx, np.float64)
            _lo = _ctx.min(1, keepdims=True) if len(_ctx) else np.zeros((0, 1))
            _hi = _ctx.max(1, keepdims=True) if len(_ctx) else np.zeros((0, 1))
            _q8 = (np.round((_ctx - _lo) / np.maximum(_hi - _lo, 1e-12) * 255.0)
                   .astype(np.uint8) if len(_ctx) else np.zeros((0, te.dim), np.uint8))
            secs.append({"kind": "lecore.learning.semantic", "id": "v1",
                         "meta": {"words": words,
                                  "ingest_stats": getattr(self, "_semantic_ingest_stats",
                                                          None)},
                         # INT8 WITH PER-ROW lo/hi -- the SAME SCHEME ALREADY
                         # SHIPPING in lecore_data/routing/index_128d.npz, applied
                         # one layer up. This array was 14.1 MB of a 25.5 MB
                         # partition (55%), stored as float32.
                         # MEASURED on the live partition: 14.1 -> 3.5 MB, 4.0x,
                         # cosine to the original min 0.99995 / mean 0.99997.
                         # NOT seed-derivable (lever 3 does not apply -- row norms
                         # run 1.4 to 1172, so this is ACCUMULATED evidence, not a
                         # regenerable codebook). Lever 3's question was asked and
                         # answered NO; this is the precision lever instead.
                         "arrays": {"ctx_q": _q8, "ctx_lo": _lo, "ctx_hi": _hi
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
        # audit='regen' (sweep 96, THE middle-out delivery): the audit K/V rows are
        # deterministic encodings of taught TEXT this partition already stores, and the
        # loader's migration path already REBUILDS the whole floor by replaying taught
        # text when no experience section is present (cp21 machinery). So: when every
        # audit row is taught-attributable (counts match -- the guard), skip the
        # experience section entirely and let replay regenerate trace + audit + atoms
        # through the ORIGINAL code path on load. Lever 3: never store what determinism
        # regenerates. GUARD, honest: rows written by non-teach verbs are not in the
        # taught log; a count mismatch falls back to storing arrays, loudly, in-result.
        self._audit_regen_applied = False
        self._audit_regen_reason = None
        if audit == "regen":
            n_audit = sum(len(getattr(t, "_audit", []) or [])
                          for t in self.experience.tiles)
            n_taught = len(getattr(self.zoo["ladder"], "taught_log", []) or [])
            if n_audit and n_audit == n_taught:
                self._audit_regen_applied = True
            else:
                # MEASURED (sweep 101): a lived-in mind accrues non-taught experience
                # writes (feedback, promotions, bookkeeping) -- 53 audit rows vs 40
                # taught on a plain rollover -- so the coarse count guard mostly
                # engages on PURE-TAUGHT scratch minds (bundles), which is where the
                # flips landed. The reason travels in the result so a fallback is
                # never a mystery; per-row attribution (drop only the matched rows,
                # bitmap the order) is the named next rung.
                self._audit_regen_reason = ("guard fallback: %d audit rows vs %d "
                                            "taught (non-taught writes present); "
                                            "stored arrays instead" % (n_audit, n_taught))
        tiles = []
        aud_arrays = {}
        for ti, t in enumerate(self.experience.tiles):
            st = t.to_state()
            aud = st.pop("audit", None)
            if aud:
                ks = np.stack([np.asarray(k, np.float32) for k, _ in aud])
                vs = np.stack([np.asarray(v, np.float32) for _, v in aud])
                # int8 PER ROW -- the audit arrays were 15.6 MB of a 16.2 MB
                # partition (96%) once the semantic array was packed. Same
                # scheme, third site. Measured 4.0x at cosine min 0.99996.
                _kq, _klo, _khi = _q8_pack(ks)
                _vq, _vlo, _vhi = _q8_pack(vs)
                aud_arrays["aud_kq_%d" % ti] = _kq
                aud_arrays["aud_klo_%d" % ti] = _klo
                aud_arrays["aud_khi_%d" % ti] = _khi
                aud_arrays["aud_vq_%d" % ti] = _vq
                aud_arrays["aud_vlo_%d" % ti] = _vlo
                aud_arrays["aud_vhi_%d" % ti] = _vhi
                st["audit_in_arrays"] = True
            else:
                st["audit"] = []                          # an EMPTY journal keeps its key
                                                          # (pop stripped it and
                                                          # from_state rightly demanded it)
            tiles.append(st)
        if not self._audit_regen_applied:
            secs.append({"kind": "lecore.learning.experience", "id": "v1",
                         "meta": {"tiles": tiles}, "arrays": aud_arrays})
        # regen mode: no experience section -> the loader's cp21 migration replays every
        # taught (q, a) through the current key function, rebuilding trace + audit + atoms
        # bit-identically (write-order preserved by taught_log order; determinism proven
        # by the identical-minds probe). Cost moves from DISK to LOAD TIME, declared.
        # FILE REFERENCES TRAVEL WITH THE FACTS THEY BELONG TO. teach_about
        # records which files a fact describes; without this section they lived in
        # process memory only, so a fact survived a reboot and its provenance did
        # not -- and stale_facts() came back empty on a partition full of facts
        # about code. A STALENESS CHECK THAT FORGETS WHAT IT WAS WATCHING REPORTS
        # EVERYTHING AS FINE.
        _ff = getattr(self, "_fact_files", None)
        if _ff:
            secs.append({"kind": "lecore.learning.factfiles", "id": "v1",
                         "meta": {"refs": _ff}})
        prev_fp = None
        # WRITE TARGET, three rungs: explicit path= wins; else the generation file the
        # rollover registered on this mind (_learning_current -- "just one file" holds
        # mid-session, not only at boot); else the legacy name, byte-for-byte the old
        # behavior for every bare mind and existing test.
        path = str(path) if path else (getattr(self, "_learning_current", None)
                                       or os.path.join(d, "state.lecore"))
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
                "audit_regen": bool(self._audit_regen_applied),
                "audit_regen_reason": getattr(self, "_audit_regen_reason", None),
                "drift_vs_previous_save": drift_vs_prev,
                "bytes": len(blob)}


def _selftest():
    """Part contract, one home: holographic.unified.check_part."""
    from holographic.unified import check_part
    return {"part": "holographic_unified_p22_zoo2",
            "members": check_part(
                "holographic.unified.holographic_unified_p22_zoo2",
                "_UnifiedPart22")}


if __name__ == "__main__":
    print(_selftest())

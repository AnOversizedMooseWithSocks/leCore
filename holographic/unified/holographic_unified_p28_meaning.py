"""UnifiedMind part 28 -- MEANING: find a taught answer by what a question means, let the attached model resolve
what memory cannot, learn every verdict on the spot, and remember HOW an answer was found (sweep 181).

Owner direction (Moose, 2026-09-24): "if an LLM is attached to the back side of leCore, it should be able to
resolve unknown or new phrasing of things, which should then allow us to learn on the go ... Learning the
meaning behind not just the question and the answer, but also how we find the answer is probably the most
important thing."

The index itself is holographic_meaning.MeaningIndex (read its docstring first). This part wires it:
    meaning                    the index (lazy; seeded from the taught record the first time)
    _meaning_rung(q)           the ladder's T0m rung: serve / run the method / clarify, or hand an unsure
                               decision to the model end
    _meaning_escalate(q, d, r) the ladder's T4m: the TYPED model prompt, verdict validated + retried once
    meaning_resolve(q, v)      apply a verdict -- from the attached model, OR from a model/human in another
                               process that got the escalation packet from serve() (the harness case)
    meaning_tool_register      the tools a METHOD row may call in-process (without one, the row returns the
                               call as a plan for the caller to execute)
    meaning_report()           what the index holds and how its decisions went
Every serve/escalation is a DecisionRecord (via "meaning") whose id comes back with the answer:
decision_outcome(id, <row id | "wrong">) is the correction path, same as every typed decision.

CLM backlog phase C (2026-09-26; faculties in part 32, tests in tests/test_negatives.py, tests/test_teacher_noise.py):
    E2.1  ONE NEGATIVE STORE -- a wrong meaning serve (decision_outcome / answer_feedback) is the exact veto PLUS a
          labelled negative that trains the row's learned prototype (_meaning_neg_add); a ladder payload vetoed by
          answer_feedback trains the meaning row it answers (_meaning_ladder_negative); the dead payload-id checks are
          live (_meaning_payload_bad, AnswerLadder._remember records the id); duplicate rows that share an answer key
          CAN be merged at creation (MeaningIndex.add_row's rule -- OFF by default: it cost Banking77 a third of its
          coverage), and a merged row is served from any live wording.
    E2.3  THE NOISY TEACHER -- ~3% of escalations are re-asked with the candidates permuted (_meaning_reask), eps_hat
          -- as a one-sided lower bound (MeaningIndex.TEACHER_EPS_Z) -- corrects the serve gate (MeaningIndex.decide)
          and the learned rows' targets, and a deep single verdict is held back while eps > 0. eps = 0 is
          bit-identical to the door without re-asks (pinned). Every verdict stays a calibration label.
    E1.3  an optional "off_domain": true in a verdict is recorded, never trained on (MeaningIndex.note_off_domain).
    E3.1  DIRECTION -- a verdict's from_question teaches the direction reader (_meaning_direction_learn), and bind reads
          from / to from the question's context words (MeaningIndex.bind(reader=)); "no direction" is clarified."""
import json

from holographic.agents_and_reasoning.holographic_meaning import MeaningIndex, normq


class _UnifiedPart28:

    # E1.1 consumer 3 -- the LEARNED ROW PROTOTYPES (MeaningIndex.enable_protos): every confirmed wording moves the rows by
    # the shared InfoNCE rule. OFF BY DEFAULT: it FAILS one of its four E0.4 bars. MEASURED (tools/bench_meaning.py
    # protos -> docs/research/evidence/bench_meaning_protos.json; one pass over each learned index's wordings, 3 seeds):
    #   top-1   CLINC150 +2.71 / +3.13 / +2.71 points (CI low +1.89 / +2.36 / +2.00); Banking77 +3.32 / +2.55 / +2.87
    #           (CI low +2.04 / +1.15 / +1.59)                                                  -> bar > +1.0: PASS
    #   AURC    CLINC -0.019..-0.022, Banking -0.022..-0.024, every CI below 0                   -> PASS
    #   realised precision at the calibrated P0.95 / P0.97 thresholds within 0.005 of target      -> PASS
    #   out-of-scope served at the calibrated P0.95 threshold: CLINC 3.7-4.1% vs base 2.15%, Banking 2.3-2.5% vs
    #           1.17% (bar: base + 0.5 points)                                                   -> FAIL
    # Why the last one fails: at the same target precision the learned rows serve FAR more (CLINC 63-64% of in-scope
    # vs 45.5%; at P0.97 42-44% vs the base's 3.3% knife edge). At MATCHED in-scope coverage (a symmetric oracle
    # diagnostic) they are strictly better: CLINC out-of-scope 1.0-1.2% vs 2.15% and in-scope wrong 0.6-0.9% vs
    # 1.2%; Banking 1.0% vs 1.2% and 0.7% vs 2.2%. Four passes (consolidate) add +0.4 to +0.8 points with the same
    # out-of-scope pattern. meaning_protos(True) / meaning_consolidate() switch it on per mind.
    MEANING_PROTOS = False
    # E2.3 -- the share of typed escalations re-asked with the candidates PERMUTED (deterministic: sha256 of the
    # wording), the teacher's self-agreement that TeacherNoise turns into eps_hat. 3%, not the panel's ~1%: MEASURED
    # (tools/bench_meaning.py online, CLINC150, noisy:0.1, 3 seeds, the gate on the z = 1 lower bound of eps): at 1%
    # (~100 re-asks per stream) eps_hat read 0.075 / 0.099 / 0.142 for a true 0.10 and the gate landed at 18.4% /
    # 39.8% / 53.9% correct with 0.3% / 1.3% / 2.5% wrong and 0.3% / 1.4% / 3.3% out-of-scope served -- a knife
    # edge; at 3% (~285 re-asks) eps_hat read 0.100 / 0.102 / 0.108 and the gate landed at 27.7% / 45.7% / 43.6%
    # correct with 0.5% / 1.3% / 1.5% wrong and 0.5% / 2.0% / 1.8% out-of-scope. Cost: ~185 more model calls in a
    # 15,000-question stream (~2% of the ~10,000 it makes). A perfect teacher is unaffected (eps 0: bit-identical).
    MEANING_REASK_RATE = 0.03

    # ---------------------------------------------------------------- the index
    @property
    def meaning(self):
        """The mind's MeaningIndex. Created on first use and SEEDED from the durable taught record, so a mind
        that learned before this part existed (or a partition saved without a meaning section) still finds its
        answers by meaning."""
        mi = getattr(self, "_meaning", None)
        if mi is None:
            mi = MeaningIndex()
            if self.MEANING_PROTOS:
                mi.enable_protos()
            self._meaning = mi
            lad = self.zoo["ladder"]
            for row in list(getattr(lad, "taught_log", []) or []):
                q = str(row[0]) if row else ""
                ex = getattr(lad, "_exact", {}).get(normq(q)) if q else None
                if ex and not q.startswith(("toolreflex: ", "decision: ")):
                    import hashlib
                    mi.add_row(q, kind="answer", provenance=str(row[3]) if len(row) > 3 else "taught",
                               akey="a:" + hashlib.sha256(" ".join(str(ex.get("answer", "")).lower().split())
                                                          .encode()).hexdigest()[:12])
        return mi

    def meaning_tool_register(self, verb, fn):
        """Register the callable a METHOD row runs for `verb` (fn(**args) -> result). A method whose verb has
        no in-process tool is returned as a CALL PLAN ({verb, args}) for the caller to execute -- memory
        still knows HOW, the caller owns the tool."""
        if not hasattr(self, "_meaning_tools"):
            self._meaning_tools = {}
        self._meaning_tools[str(verb)] = fn
        return sorted(self._meaning_tools)

    # ---------------------------------------------------------------- the rung
    def _meaning_record(self, query, d, answer):
        """A DecisionRecord for a meaning decision (via 'meaning'); its outcome becomes a calibration label, and a
        named row a learned wording. E0.5: the hook is NAMED in the record (meta hook kind 'meaning') and re-derived
        at report time by _meaning_outcome (p29) -- the closure it replaces died with the process, so a meaning
        decision reloaded from the partition could never be corrected. E0.6: the index's p is isotonic P(correct)."""
        from holographic.agents_and_reasoning.holographic_decisionrecord import DecisionRecord
        opts = [r[0] for r in (d.get("ranked") or [])]
        rec = DecisionRecord(str(query), "meaning", opts, answer, "meaning", margin=d.get("confidence"),
                             p=d.get("p"), p_correct=d.get("p"), meta={"why": d.get("why"), "hook": {"kind": "meaning"}})
        self.decision_ledger().add(rec)
        return rec.id

    def _meaning_neg_add(self, query, rid, train=True):
        """ONE NEGATIVE STORE (backlog E2.1): "this wording does NOT mean row rid". Kept two ways, both deliberate:
          * the EXACT veto (normq(query), rid) -- this wording is never served that row again (sweep 181);
          * a LABELLED NEGATIVE on the row's learned prototype (MeaningIndex.proto_negative, the shared rule's
            ProtoStore.negative form) when the learned rows are on and train=True -- so the correction also moves the
            wordings NEAR this one, which the exact veto never could.
        train=False for a verdict that only named another row for an escalated question: that verdict already
        pushed the rival through the link's InfoNCE update, and pushing it twice would double-count one label.
        Callers: a wrong meaning SERVE reported by decision_outcome / answer_feedback (_meaning_outcome, p29), a
        spot-check that overturned a serve, and a ladder payload marked bad whose question is a meaning row."""
        if rid is None:
            return
        if not hasattr(self, "_meaning_neg"):
            self._meaning_neg = set()
        self._meaning_neg.add((normq(query), rid))
        mi = getattr(self, "_meaning", None)
        if train and mi is not None and mi.protos is not None and rid in mi.rows:
            from holographic.agents_and_reasoning.holographic_learnguard import sensitive_reason
            if not sensitive_reason(str(query)):            # a wording carrying a secret never trains anything
                mi.proto_negative(str(query), rid)

    def _meaning_ladder_negative(self, query, pid_key):
        """answer_feedback(ok=False) on the LADDER's reflex (E2.1): the payload it vetoes was taught for a question;
        when that question is a meaning row (its canonical wording, or a wording the merge rule put in a row), the
        same verdict is a labelled negative for that row -- "this wording does not mean it" (_meaning_neg_add). The
        payload veto itself is unchanged. -> the row id, or None."""
        mi = getattr(self, "_meaning", None)
        q_taught = getattr(self.zoo["ladder"], "_payload_qs", {}).get(pid_key)
        if mi is None or not q_taught:
            return None
        pid = mi._by_norm.get(normq(q_taught))
        rid = mi._pid_row[pid] if pid is not None else None
        if rid is None or rid not in mi.rows or mi.rows[rid]["kind"] != "answer":
            return None
        self._meaning_neg_add(query, rid)
        return rid

    def _meaning_payload_bad(self, entry, nq):
        """Is the ladder's exact entry for question nq a payload marked BAD by answer_feedback?  FIXES THE DEAD CHECK
        (backlog E2.1): this rung compared entry["pid"] against _payload_bad, but the exact store always wrote
        pid None, so the check never fired and a vetoed payload was still served by meaning. The ladder now records the
        entry's payload id when it locates it (AnswerLadder._remember); the id must also still belong to THIS
        question (after a tile split a slot index can be reused by another question -- that one's veto must not
        leak onto this entry)."""
        lad = self.zoo["ladder"]
        pid = (entry or {}).get("pid")
        if pid is None or pid not in getattr(lad, "_payload_bad", set()):
            return False
        return normq(getattr(lad, "_payload_qs", {}).get(pid, "")) == nq

    def _meaning_answer(self, row):
        """The exact-store entry a meaning row serves: its canonical wording's, or -- for a row the merge rule grew
        (E2.1: a new answer with the same answer key joined an existing row) -- the first of its wordings whose own
        exact entry is live, non-blank and not a bad payload. None = the row's answer is gone everywhere."""
        ex_all = getattr(self.zoo["ladder"], "_exact", {})
        # the fallback needs a VERIFIABLE answer key ("a:" + sha of the answer text): a legacy row keyed by its own id
        # cannot prove another wording's answer is its answer, so it serves its canonical wording's or nothing
        words = [row["canonical"]] + ([p for p in row["phrasings"] if p != row["canonical"]]
                                      if str(row.get("akey", "")).startswith("a:") else [])
        for ph in words:
            nq = normq(ph)
            ex = ex_all.get(nq)
            if ex and str(ex.get("answer", "")).strip() and not self._meaning_payload_bad(ex, nq):
                if ph != row["canonical"]:
                    import hashlib
                    ak = "a:" + hashlib.sha256(" ".join(str(ex.get("answer", "")).lower().split())
                                               .encode()).hexdigest()[:12]
                    if ak != row["akey"]:
                        continue                        # a wording that is its own answer elsewhere: not this row's
                return ex
        return None

    def _meaning_rung(self, query):
        """-> (served dict or None, decision or None). See AnswerLadder.answer T0m."""
        mi = self.meaning
        if not mi.rows:
            return None, None
        lad = self.zoo["ladder"]
        nq = normq(query)
        if nq in getattr(lad, "_vetoed_qs", set()):
            return None, None                               # a vetoed question is vetoed through every door
        d = mi.decide(query)
        if d["action"] == "serve" and (nq, d["rid"]) in getattr(self, "_meaning_neg", set()):
            d["action"], d["why"] = "escalate", "this wording was corrected away from that row before"
        if d["action"] == "clarify":
            return {"tier": "clarify", "via": "meaning", "answer": d["clarify"], "row": d.get("rid"),
                    "confidence": d["confidence"], "why": d["why"]}, d
        if d["action"] != "serve":
            return None, d
        rid, row = d["rid"], d["row"]
        if row["kind"] == "answer":
            ex = self._meaning_answer(row)
            if ex is None:
                mi.drop_row(rid)                            # the answer was vetoed/forgotten: the row is dead
                d.update(action="escalate", why="the matched row's answer is no longer in memory")
                return None, d
            rec_id = self._meaning_record(query, d, rid)
            self._meaning_last = {"query": nq, "id": rec_id, "row": rid}
            from holographic.agents_and_reasoning.holographic_decisionrecord import door_record
            return door_record({"tier": "T0", "via": "meaning", "answer": ex["answer"],
                                "provenance": ex.get("provenance", "model-cached"), "row": rid,
                                "matched": d["ranked"][0][2], "confidence": d["confidence"], "p": d["p"],
                                "id": rec_id, "why": d["why"]}, "meaning", p_correct=d["p"], p_null=None), d
        if row["kind"] == "method":
            mi._last_direction = None
            args, missing, unused = mi.bind_checked(rid, query, reader=self._meaning_reader())
            if missing:
                nodir = (mi._last_direction or {}).get("direction") is False
                d.update(action="escalate", need=missing,
                         why=("the question names %s's values but not which way (no direction in its words)"
                              % "/".join(missing)) if nodir else
                         "the question matches a method but does not give %s" % ", ".join(missing))
                return None, d
            if unused:
                # the question says MORE than the learned call uses ("20 yen" when the method has no amount):
                # running it would silently drop part of the request -- ask instead, and learn the argument
                d.update(action="escalate", unused=unused,
                         why="the question gives %s, which the learned call does not use" % ", ".join(unused))
                return None, d
            rec_id = self._meaning_record(query, d, rid)
            self._meaning_last = {"query": nq, "id": rec_id, "row": rid}   # answer_feedback corrects this link too
            return self._meaning_run(rid, args, d, rec_id), d
        return None, d

    def _meaning_run(self, rid, args, d=None, rec_id=None, by=None):
        """Run a METHOD row with bound args. A live value is returned and NEVER cached -- the method is what
        memory keeps. No in-process tool for the verb -> the call itself is the answer (a plan)."""
        row = self.meaning.rows[rid]
        m = row["method"]
        call = {"verb": m["verb"], "args": args}
        base = {"row": rid, "call": call, "live": m.get("live", False), "id": rec_id}
        if d is not None:
            base.update(confidence=d.get("confidence"), p=d.get("p"), p_correct=d.get("p"), p_null=None)   # E0.6
        if by:
            base["by"] = by
        fn = getattr(self, "_meaning_tools", {}).get(m["verb"])
        from holographic.io_and_interop.holographic_apilearn import (placeholders_in, resolve_placeholders,
                                                                       MissingCredential)
        _env = placeholders_in(args)
        if fn is None:
            # the PLAN keeps its ${VAR} placeholders: the caller who makes the call reads its own environment
            return dict(base, tier="T2", via="method-plan", answer="", **({"needs_env": _env} if _env else {}),
                        why="memory knows HOW to answer this: make the call (no in-process tool for %r)" % m["verb"])
        try:
            # a learned credential is a ${VAR} placeholder: read the environment NOW, and never run the tool without
            # it (2026-09-27) -- a missing variable is a loud, named failure
            args_run = resolve_placeholders(args) if _env else args
        except MissingCredential as e:
            return dict(base, tier="T2", via="method-failed", answer="", missing_env=e.names,
                        why="the %s call needs %s -- export it (learned credentials are read from the environment "
                            "at call time, never stored)" % (m["verb"], ", ".join(e.names)))
        try:
            out = fn(**args_run)
        except Exception as e:
            return dict(base, tier="T2", via="method-failed", answer="",
                        why="the %s call failed: %s" % (m["verb"], str(e)[:160]))
        return dict(base, tier="T2", via="method:%s" % m["verb"], answer=str(out),
                    why="answered by running the learned method (live values are never cached)")

    # ---------------------------------------------------------------- spot-checks
    SPOT_CHECK_RATE = 0.05

    def _meaning_should_verify(self, query):
        """Verify every meaning serve until the gate is calibrated, then a deterministic ~5% (a hash of the
        wording, so the same question is always checked or never -- reproducible runs)."""
        if not self.meaning.calibrated():
            return True
        import hashlib
        h = int(hashlib.sha256(normq(query).encode()).hexdigest()[:8], 16)
        return (h % 10000) < int(self.SPOT_CHECK_RATE * 10000)

    def _meaning_verify(self, query, served, d, resolver):
        """Ask the model about a serve memory was confident in. Agreement: the serve stands (verified, and the
        confidence is labelled correct). Disagreement: the model's verdict is applied and returned instead."""
        mi = self.meaning
        ranked = list((d or {}).get("ranked") or [])
        v, why = mi.parse_verdict(resolver(mi.resolution_prompt(query, ranked)), [r[0] for r in ranked])
        if v is None:
            return None                                    # an unusable reply changes nothing
        got = self.meaning_resolve(query, v, by="model", decision=d, _ranked=ranked)
        if got.get("row") == served.get("row") and v.get("verdict") == "same":
            return dict(served, verified=True)
        return dict(got, corrected_from=served.get("row"))

    # ---------------------------------------------------------------- the model end
    def meaning_packet(self, query):
        """The TYPED escalation packet for a question memory cannot serve: {prompt, candidates}. Hand the prompt
        to any model (or person) and report the JSON reply with meaning_resolve(query, reply)."""
        mi = self.meaning
        ranked = mi.answers(str(query)) if mi.rows else []
        # MAPLE SWARM RUN (2026-09-27): remember WHAT THE MODEL WAS SHOWN, so meaning_resolve labels the decision that
        # escalated and not a re-ranking made later. MEASURED: in a 5-worker swarm three workers answered "new" for a
        # wording another worker had taught seconds earlier; re-ranked at resolve time the top row was that very
        # wording (score 1.0, g ~1.7), so each verdict was stored as "the top row was WRONG at g 1.7" -- 3 of the 4
        # labels above g 1.1 were these races, the isotonic curve fell below the 0.95 bar at the top, and the
        # calibrated gate served 1 of 16 fresh rewordings. Bounded: the oldest shown list is dropped past 4096.
        shown = getattr(self, "_meaning_shown", None)
        if shown is None:
            import collections
            shown = self._meaning_shown = collections.OrderedDict()
        shown[str(query)] = list(ranked)
        shown.move_to_end(str(query))
        while len(shown) > 4096:
            shown.popitem(last=False)
        return {"prompt": mi.resolution_prompt(str(query), ranked),
                "candidates": [{"row": r, "score": round(s, 4), "matched": p} for r, s, p in ranked]}

    def _meaning_escalate(self, query, d, resolver):
        """The ladder's T4m. -> answer dict, or None to let the legacy T4 run."""
        mi = self.meaning
        ranked = list((d or {}).get("ranked") or [])
        if d is None and mi.rows:
            ranked = mi.answers(query)
        note = ""
        if (d or {}).get("need"):
            note += ("\nNOTE: the question matches row %s (a method) but does not give: %s. If it is that "
                     "method, reply same with those args; if the person must say, reply unclear."
                     % (d["rid"], ", ".join(d["need"])))
        if (d or {}).get("unused"):
            note += ("\nNOTE: the question matches row %s (a method) but also gives %s, which that method's "
                     "call does not use. If it is that method, reply same with ALL the args this question "
                     "needs (and from_question for each)." % (d["rid"], ", ".join(d["unused"])))
        prompt = mi.resolution_prompt(query, ranked) + note
        reply = resolver(prompt)
        if reply is None:
            return None
        allowed = [r[0] for r in ranked]
        v, why = mi.parse_verdict(reply, allowed)
        if v is None and "{" in str(reply):                 # it TRIED the contract: one retry with the violation
            reply = resolver(prompt + "\nYOUR PREVIOUS REPLY WAS REJECTED: %s. Reply again with exactly one "
                                      "valid JSON object." % why)
            v, why = mi.parse_verdict(reply, allowed)
            if v is None:
                mi.stats["verdict_bad"] += 1
        if v is None:
            text = str(reply or "")
            if not text.strip() or "{" in text:
                return {"tier": "T4", "via": "main", "answer": "" if "{" in text else text, "confidence": None,
                        "why": "the model gave no usable answer (%s)" % (why or "blank")}
            lad = self.zoo["ladder"]
            lad._remember(lad._qkey(query), text, query)    # the legacy T4 cache, guard included
            return {"tier": "T4", "via": "main", "answer": text, "confidence": None,
                    "why": "the model answered in plain text (no typed verdict): cached as model-cached"}
        if len(ranked) >= 2 and self._meaning_should_reask(query):
            self._meaning_reask(query, ranked, v, resolver, note)
        return self.meaning_resolve(query, v, by="model", decision=d, _ranked=ranked)

    # ---------------------------------------------------------------- E2.3: the teacher's own noise
    def _meaning_should_reask(self, query):
        """~MEANING_REASK_RATE of escalations, chosen by sha256 of the wording (the same question is always re-asked
        or never: reproducible runs, and a re-ask can never be steered by call order)."""
        import hashlib
        h = int(hashlib.sha256(("reask:" + normq(query)).encode()).hexdigest()[:8], 16)
        return (h % 10000) < int(round(self.MEANING_REASK_RATE * 10000))

    def _meaning_verdict_key(self, v):
        """What two answers must share to AGREE: the named row's answer key (two rows of one answer are one answer),
        or the verdict kind for new / unclear."""
        mi = self.meaning
        if v.get("verdict") == "same":
            r = mi.rows.get(v.get("row"))
            return "row:" + str((r or {}).get("akey", v.get("row")))
        return str(v.get("verdict"))

    def _meaning_reask(self, query, ranked, v, resolver, note=""):
        """E2.3 NOISY TEACHER BY SELF-AGREEMENT: ask the SAME typed question again with the candidates in a PERMUTED
        order (sha256-seeded, never the identity), and feed the pair of answers to the index's TeacherNoise. If each
        ask is wrong at rate e, P(agree) = (1-e)^2 + e^2/m -- eps_hat is that quadratic solved (TeacherNoise). The
        permutation is what stops POSITION bias from faking agreement. The first verdict is the one applied; the re-ask
        only measures. CAVEAT kept loud: the two asks are independent for the scripted stand-in (fresh noise per
        call); for a real model, re-asking the same question may repeat the same mistake, so eps_hat can READ LOW --
        a hypothesis to measure on a live model end, not a property of this code. -> the second verdict or None."""
        import hashlib
        import numpy as np
        from holographic.agents_and_reasoning.holographic_protostore import TeacherNoise
        mi = self.meaning
        seed = int(hashlib.sha256(("perm:" + normq(query)).encode()).hexdigest()[:8], 16)
        perm = [int(i) for i in np.random.default_rng(seed).permutation(len(ranked))]
        if perm == list(range(len(ranked))):
            perm = perm[1:] + perm[:1]                      # never the same order twice
        permuted = [ranked[i] for i in perm]
        v2, _ = mi.parse_verdict(resolver(mi.resolution_prompt(query, permuted) + note), [r[0] for r in ranked])
        if v2 is None:
            return None
        if mi.teacher is None:
            mi.teacher = TeacherNoise(m=7, min_pairs=20)
        mi.teacher.observe(self._meaning_verdict_key(v), self._meaning_verdict_key(v2))
        mi.stats["reasked"] = mi.stats.get("reasked", 0) + 1
        return v2

    # How a single `same` verdict is judged likely noise (only while eps_hat > 0):
    #   "rank"      TeacherNoise.hold_back: the pick is ranked 5th or lower (the backlog's rule)
    #   "posterior" the noise-corrected TARGET turned into a keep / hold decision: hold when the posterior that the
    #               pick is the truth, p_y(1-eps) / (p_y(1-eps) + (1-p_y) eps/m) with p = softmax(score / 0.05) over
    #               the candidates, is below 1/2. MEASURED (scratch probe on bench_contrastive's static index, 10%
    #               planted noise): CLINC150 catches 59.2% of planted errors holding back 10.5% of genuine disagreeing
    #               corrections (rank: 55.4% / 12.9%); Banking77 62.6% / 11.5% (rank: 61.4% / 22.7%).
    MEANING_HOLD_RULE = "rank"

    def _meaning_hold_back(self, ranked, pick, eps):
        """Is this single `same` verdict (picking row `pick` among `ranked`) held back as likely teacher noise?"""
        from holographic.agents_and_reasoning.holographic_protostore import TeacherNoise
        ids = [r[0] for r in ranked]
        if pick not in ids:
            return False
        if self.MEANING_HOLD_RULE == "posterior":
            import numpy as np
            s = np.array([float(r[1]) for r in ranked])
            p = np.exp((s - s.max()) / 0.05)
            p /= p.sum()
            py = float(p[ids.index(pick)])
            m = 7.0 if self.meaning.teacher is None else float(self.meaning.teacher.m)
            return py * (1 - eps) < 0.5 * (py * (1 - eps) + (1 - py) * eps / m) if ids.index(pick) > 0 else False
        return TeacherNoise.hold_back(ids.index(pick) + 1, eps)

    def meaning_teacher_eps(self):
        """The model end's estimated flip rate eps_hat (0.0 until 20 re-asks exist; then the gate and the learning
        rule correct for it). See _meaning_reask for how it is measured and its caveat."""
        return float(self.meaning.teacher_eps())

    def meaning_resolve(self, query, verdict, by="model", decision=None, _ranked=None):
        """APPLY A TYPED VERDICT and learn it on the spot. `verdict` is the parsed dict or the raw JSON reply:
            same    -> the wording joins that row (and its method's new slot values are learned); served
            new     -> a new row: a plain answer is taught (through the learning guard); a METHOD is kept as
                       how the answer is found (a live value itself is never kept)
            unclear -> the wording is remembered as unclear, with the clarifying question to ask
        The verdict is also a calibration LABEL for the decision that escalated (was its top row right?).
        E2.3: when the model end is measurably noisy (eps_hat > 0), a single `same` that picks a candidate ranked 5th
        or lower is HELD BACK (answered, not learned: TeacherNoise.hold_back -- noise lands uniformly over ranks,
        genuine corrections concentrate on the runner-up), and the learned rows train on the noise-corrected target.
        E1.3: an optional "off_domain": true is RECORDED (never trained on yet; see resolution_prompt).
        Returns the answer dict for this turn."""
        # MAPLE SWARM RUN (2026-09-27): an escalation the model end answered through THIS door stayed open --
        # resolve() cleared its question but meaning_resolve() did not. MEASURED: after a 5-worker swarm the ledger
        # listed 55 open questions of which 26 already SERVED at T0 (24 answered here, 3 taught word for word), so
        # the swarm's 'what we do not know' list could not be read as a to-do. An APPLIED verdict (same / new / unclear)
        # means the question was answered, so its escalation is cleared, exactly as resolve() does -- also when the
        # guard kept the answer out of memory. A rejected reply ({resolved: False}) leaves it open.
        out = self._meaning_resolve_apply(query, verdict, by=by, decision=decision, _ranked=_ranked)
        if isinstance(out, dict) and out.get("resolved", True) is not False:
            led = getattr(self, "_escalations", None) or {}
            out["cleared"] = led.pop(str(query), None) is not None
            self._escalations = led
        return out

    def _meaning_resolve_apply(self, query, verdict, by="model", decision=None, _ranked=None):
        """The body of meaning_resolve (see its docstring); the public door adds the escalation bookkeeping."""
        mi = self.meaning
        shown = (getattr(self, "_meaning_shown", None) or {}).pop(str(query), None)
        if _ranked is None and not (decision or {}).get("ranked") and shown is not None:
            _ranked = shown                                 # label what the model SAW (see meaning_packet)
        ranked = list(_ranked if _ranked is not None else (decision or {}).get("ranked") or mi.answers(str(query)))
        if isinstance(verdict, str):
            v, why = mi.parse_verdict(verdict, [r[0] for r in ranked])
            if v is None:
                return {"resolved": False, "why": why}
        else:
            v = dict(verdict)
        kind = v.get("verdict")
        mi.stats["verdict_" + kind] = mi.stats.get("verdict_" + kind, 0) + 1
        from holographic.agents_and_reasoning.holographic_learnguard import sensitive_reason
        secret_q = sensitive_reason(str(query))              # a wording that carries a secret is never kept
        if v.get("off_domain") is True and not secret_q:
            mi.note_off_domain(str(query), kind)
        eps = self.meaning_teacher_eps() if by == "model" else 0.0
        held = False
        if kind == "same" and eps > 0:
            held = self._meaning_hold_back(ranked, v.get("row"), eps)
            if held:
                mi.stats["held_back"] = mi.stats.get("held_back", 0) + 1
        served = (decision or {}).get("action") == "serve"
        # A "new" verdict when this exact wording is ALREADY the top row (score ~1.0) is a race (taught since the model
        # looked -- only possible when the shown list is gone, e.g. after a restart) or a correction of a stale answer;
        # the two cannot be told apart here, so it is applied but never LABELLED (see meaning_packet for the measured
        # cost of labelling races).
        race = kind == "new" and bool(ranked) and float(ranked[0][1]) >= 0.999
        if race:
            mi.stats["race_unlabelled"] = mi.stats.get("race_unlabelled", 0) + 1
        if ranked and not race:                             # the label for the gate: was the top row right?
            # ALWAYS labelled, held back or not: the gate calibrates P(the MODEL agrees | g), and the noise correction
            # (TeacherNoise.corrected) assumes every disagreement the noisy model produces is IN that estimate.
            # KEPT NEGATIVE (first cut): skipping the label of a held-back verdict removed ~7% of the labels -- almost
            # all disagreements -- which inflated P(agree), which the correction then amplified: the gate over-served.
            g = MeaningIndex.confidence(ranked)
            mi.observe(g, kind == "same" and v.get("row") == ranked[0][0])
        if ranked and not held:
            if kind == "same" and v.get("row") != ranked[0][0]:
                # the exact veto always; a prototype negative only when memory had SERVED that row (a spot-check
                # overturned it) -- otherwise the link below already pushed it through the InfoNCE update
                self._meaning_neg_add(query, ranked[0][0], train=served)
        learn = not secret_q and not held
        if kind == "same":
            rid = v["row"]
            if rid not in mi.rows:
                return {"resolved": False, "why": "row %r is not in memory" % rid}
            row = mi.rows[rid]
            learned = False
            if learn:
                learned = mi.link(rid, str(query), learn=True, eps=eps)
            if row["kind"] == "method":
                fq = v.get("from_question") if isinstance(v.get("from_question"), dict) else {}
                if learn:
                    for name, val in (v.get("args") or {}).items():
                        spec = row["method"]["args"].get(name)
                        if spec is not None and not isinstance(spec, dict) and fq.get(name):
                            # a CONSTANT the question just named a different value for: the method learns that
                            # this argument comes from the question, and the old constant becomes its default
                            old_const = spec
                            row["method"]["args"][name] = spec = {"slot": name}
                            row["method"]["slots"][name] = {"type": "choice", "values": {}, "default": old_const}
                        if isinstance(spec, dict) and "slot" in spec:
                            surf = fq.get(name) or val              # the words in the question that meant it
                            if " ".join(str(surf).lower().split()) in " ".join(
                                    str(query).lower().replace("'", "").split()) or fq.get(name):
                                mi.learn_slot_value(rid, spec["slot"], surf, val)
                    for name, val in (v.get("args") or {}).items():
                        if name not in row["method"]["args"]:
                            mi.extend_method(rid, str(query), name, val, surface=fq.get(name) or None)
                    self._meaning_direction_learn(str(query), row, fq)
                args, missing = mi.bind(rid, str(query), reader=self._meaning_reader())
                for name, val in (v.get("args") or {}).items():
                    if name in row["method"]["args"]:
                        args[name] = val                    # THIS turn: the model's reading of the question wins
                        missing = [x for x in missing if x != name]
                if missing:
                    return {"tier": "clarify", "via": "meaning-resolved", "row": rid, "by": by,
                            "answer": "Which %s do you mean?" % " and ".join(missing), "learned": learned}
                return dict(self._meaning_run(rid, args, by=by), verdict="same", learned=learned,
                            tier="T4" if by == "model" else "T2", **({"held_back": True} if held else {}))
            if row["kind"] == "clarify":
                return {"tier": "clarify", "via": "meaning-resolved", "answer": row.get("clarify"), "row": rid,
                        "by": by, "learned": learned}
            ex = self._meaning_answer(row) or {}
            out = {"tier": "T4", "via": "meaning-resolved", "answer": ex.get("answer", ""), "row": rid,
                   "verdict": "same", "by": by, "learned": learned,
                   "why": "the model matched this wording to a row memory already had"}
            if held:
                out.update(held_back=True, why="answered from the model's verdict; NOT learned -- a single verdict "
                                               "that picks a candidate ranked 5th or lower is held back while the "
                                               "model end is measurably noisy (eps_hat %.3f)" % eps)
            return out
        if kind == "unclear":
            rid = None
            if not secret_q:
                rid = mi.add_row(str(query), kind="clarify", clarify=str(v["clarify"]), provenance=str(by))
            return {"tier": "clarify", "via": "meaning-resolved", "answer": str(v["clarify"]), "row": rid,
                    "by": by, "learned": rid is not None}
        # new
        answer = str(v.get("answer") or "")
        method = v.get("method") if isinstance(v.get("method"), dict) else None
        if method:
            m = mi.generalize_method(str(query), method)
            rid = None
            if not secret_q:
                rid = mi.add_row(str(query), kind="method", method=m, provenance=str(by))
                fq = method.get("from_question") if isinstance(method.get("from_question"), dict) else {}
                self._meaning_direction_learn(str(query), mi.rows[rid], fq)
            return {"tier": "T4", "via": "meaning-resolved", "answer": answer, "row": rid, "verdict": "new",
                    "call": {"verb": m["verb"], "args": dict(method.get("args") or {})}, "live": m["live"],
                    "by": by, "learned": rid is not None,
                    "why": "a new answer, remembered as HOW it was found%s" % (
                        " (the live value itself is not kept)" if m["live"] else "")}
        lad = self.zoo["ladder"]
        lad._last_refusal = None
        lad._remember(lad._qkey(str(query)), answer, str(query), provenance="model:%s" % by if by else None)
        refused = getattr(lad, "_last_refusal", None)
        return {"tier": "T4", "via": "meaning-resolved", "answer": answer, "verdict": "new", "by": by,
                "learned": refused is None,
                "why": ("a new answer, taught" if refused is None
                        else "answered, NOT learned: %s" % refused.get("reason", "the learning guard refused"))}

    # ---------------------------------------------------------------- direction (E3.1 wiring)
    def _meaning_reader(self):
        """The direction reader MeaningIndex.bind consults for slots that share one value pool (from / to): the
        mind's ContextRoles (part 30), or None when this mind's direction reading is switched off
        (_meaning_direction = False -- the positional baseline the bench measures against)."""
        if not getattr(self, "_meaning_direction", True) or not hasattr(self, "_direction_reader"):
            return None
        return self._direction_reader()

    def _meaning_direction_learn(self, query, row, fq):
        """A verdict's from_question for a method whose CHOICE slots share one value pool (two or more: from / to
        currency) teaches the direction reader which context words mark which slot (direction_learn, part 30).
        {"either": [words, words]} is the model's way to say the question names the values but NO direction ("the
        rate between X and Y") -- it teaches the reader's no-direction hypothesis. A mention the reader cannot
        locate in the question (ValueError) teaches nothing: a verdict is never forced onto the wrong words."""
        if not fq or row.get("kind") != "method" or not hasattr(self, "direction_learn"):
            return None
        m = row["method"]
        shared = [a["slot"] for n, a in m["args"].items() if isinstance(a, dict) and "slot" in a
                  and (m["slots"].get(a["slot"]) or {}).get("type") == "choice"]
        if len(shared) < 2:
            return None
        teach = {}
        if isinstance(fq.get("either"), list) and len(fq["either"]) >= 2:
            teach["either"] = [str(w) for w in fq["either"]]
        else:
            teach = {name: str(fq[name]) for name in shared if str(fq.get(name) or "").strip()}
            if len(teach) < 2:
                return None
        try:
            return self.direction_learn(query, teach)
        except ValueError:
            return None

    # ---------------------------------------------------------------- report / persistence
    def meaning_report(self):
        """Rows by kind, wordings, learned associations, calibration state and decision counters."""
        mi = self.meaning
        kinds = {}
        for r in mi.rows.values():
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        return {"rows": len(mi.rows), "by_kind": kinds,
                "wordings": sum(len(r["phrasings"]) for r in mi.rows.values()),
                "associations": sum(len(d) for d in mi.assoc.values()),
                "slot_values": len(mi.slot_vocab), "calibration_labels": len(mi.calib),
                "calibrated": mi.calibrated(), "stats": dict(mi.stats)}

    def _meaning_state(self):
        """The meaning section's meta: the index (rows, wordings, calibration, learned-row bookkeeping, the teacher's
        re-ask pairs), the exact negatives, and the direction reader's meta once it learned (E3.1 wiring)."""
        mi = getattr(self, "_meaning", None)
        if mi is None or not mi.rows:
            return None
        st = mi.state()
        st["neg"] = sorted([list(x) for x in getattr(self, "_meaning_neg", set())])
        rd = self.__dict__.get("_direction_reader_obj")
        if rd is not None and rd.counts:
            st["direction"] = rd.state()[0]
        return st

    def _meaning_arrays(self):
        """The meaning section's arrays: the learned rows' delta (proto_*) and the direction reader's (dir_*)."""
        mi = getattr(self, "_meaning", None)
        out = dict(mi.state_arrays()) if mi is not None else {}
        rd = self.__dict__.get("_direction_reader_obj")
        if rd is not None and rd.counts:
            for k, a in rd.state()[1].items():
                out["dir_" + k] = a
        return out

    def _meaning_restore(self, st, arrays=None):
        arrays = arrays or {}
        self._meaning = MeaningIndex.from_state(st, {k: v for k, v in arrays.items() if k.startswith("proto_")})
        self._meaning_neg = {tuple(x) for x in (st or {}).get("neg") or []}
        if (st or {}).get("direction"):
            import holographic.agents_and_reasoning.holographic_rolecall as RC
            self.__dict__["_direction_reader_obj"] = RC.ContextRoles.from_state(
                st["direction"], {k[4:]: v for k, v in arrays.items() if k.startswith("dir_")})
        return len(self._meaning.rows)


def _selftest():
    """The part's contract end to end on a bare mind: a resolved rewording is served from memory afterwards (never a
    duplicate row); a live method runs fresh each time and caches nothing; a bare value asks back."""
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    m.teach("how do i check my account balance", "Open the app and tap Accounts.")
    rid = list(m.meaning.rows)[0]
    m.meaning_resolve("how much money do i have left", {"verdict": "same", "row": rid})
    assert m.ask("How much money do I have left?")["via"] == "meaning" and len(m.meaning.rows) == 1
    n = []
    m.meaning_tool_register("price", lambda symbol: n.append(symbol) or "%s %d" % (symbol, len(n)))
    for k in range(24):                               # a calibrated history (see tests/test_meaning.py)
        m.meaning.observe(1.0 + k * 0.05, True)
        m.meaning.observe(0.1 + k * 0.02, False)
    m.meaning_resolve("what's the price of solana", {"verdict": "new", "answer": "-", "method": {
        "verb": "price", "args": {"symbol": "SOL"}, "from_question": {"symbol": "solana"}, "live": True}})
    a, b = m.ask("what's the price of solana"), m.ask("what's the price of solana")
    assert a["call"]["args"] == {"symbol": "SOL"} and a["answer"] != b["answer"] and n == ["SOL", "SOL"]
    assert m.ask("solana")["tier"] == "clarify" and n == ["SOL", "SOL"]
    return "ok"


if __name__ == "__main__":
    print(_selftest())

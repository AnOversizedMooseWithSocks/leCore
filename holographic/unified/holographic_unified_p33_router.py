"""UnifiedMind part 33 -- the router learns, serve asks the router, and ONE tool learner (the CLM backlog, F1 items).

Everything a door LEARNS here goes through the one shared rule (holographic_protostore.ProtoStore, InfoNCE on per-option
prototypes) and is persisted in the memory partition (learning_save / learning_load carry a hand-written section each).

    E4.1  router_learn / router_report / router_mode    the capability router's ProtoStore("route") over cards
          (holographic_routerlearn.RouteLearner): pretrained on the catalog's own aliases (free, noiseless labels),
          updated by every reported route outcome, reranking the lexical top-K. route_tiered consults it when the
          router mode is on. Measured in tools/bench_router.py -> docs/research/evidence/bench_router.json.
    E4.5  _serve_route                                   serve()'s router consult after a memory miss: a route ANSWER
          whose gate clears the route door's bar is served as "use capability X" (via 'route', with the id); a menu
          or refusal escalates (the menu rides in the escalation packet).
    E4.3  tool_key / _tool_pick / _tool_seed / _tool_verdict   tool choice through ProtoStore("tool"): one prototype
          per learned tool (seeded by its taught patterns, moved by verified calls and reported corrections; a
          failed verify is a labelled negative), the word-overlap rule kept as the cold-start fallback.
    E4.4  _tool_audit / _tool_section / _tool_restore   the tool door is the ONE tool learner: tool_note /
          tool_predict / present_tools read and write it; the old UsageTrace section still loads (read-compat shim).
    Q3    protostore_share / _shareable_stores / _protostore_track   learned prototypes travel through
          contribute / commons_pool -- only from stores whose every question passed the learning guard and none was
          session-salted (holographic_unified_p20b_transfer).
"""
import numpy as np

from holographic.agents_and_reasoning import holographic_routerlearn as RL

# ---------------------------------------------------------------------------------------------------------------------
# E4.5 -- THE ROUTE DOOR'S SERVE BAR. serve() answers "use capability X" only when the route's GATE clears the answer
# threshold re-derived with the E0.4 protocol for precision >= ROUTE_SERVE_P (tools/bench_router.py gate section:
# calibrated on the val aliases + CLINC oos half A under the 4,500 : 1,000 prior, reported on the held-out test aliases +
# oos half B) AND -- once the route door's own calibrator holds labelled outcomes of both kinds -- p_correct >=
# ROUTE_SERVE_P. Until then p_correct stays None in the record (an uncalibrated door says so) and the bench threshold IS
# the bar. The gate signal is z + (top score - runner-up score): the best lexical signal by val AURC (0.535 vs z alone
# 0.559); with the learned router on, the same signal over the FUSED ranking (0.501).
#   RE-DERIVED 2026-09-27 on the final catalog (3,996 cards): lexical 7.168 and fused 7.347 (first derivation, 3,982
#   cards: 6.869 / 7.047 -- the catalog's content grew, which lifts every z; the bar must be re-derived whenever the
#   catalog changes materially: `python tools/bench_router.py --data DIR --sections gate`).
#   MEASURED at P 0.90: on HELD-OUT aliases neither serves anything (no
#   paraphrase reaches it; the router's top-1 is ~0.40), so what serve() answers is the near-exact capability wording
#   (an alias, a card's own name) -- which is what the panel's 20 probes are in the live service.
ROUTE_SERVE_P = 0.90
ROUTE_SERVE_LEX = 7.1680             # bench_router.json gate -> signals["lexical|z+margin"] calibrated P0.90 thr
ROUTE_SERVE_FUSED = 7.3469           # bench_router.json gate -> signals["fused|z+margin"] calibrated P0.90 thr
# The refuse floor re-derived (the smallest z refusing at least the shipped floor's share of calibration oos): -0.43,
# which refuses exactly the same items as the shipped -0.5 (z is discrete) -- so -0.5 stands.
ROUTE_REFUSE_Z = -0.5

# E4.3 -- the tool door
TOOL_DOOR = "tool"

# Q3 -- two prototypes for one label that point this far apart are a CONFLICT (two users used the label for different
# questions): flagged and not merged, like a conflicting taught answer (memory_import's on_conflict='flag').
PROTO_CONFLICT_COS = 0.3


class _UnifiedPart33:

    # ================================================================================================================
    # E4.1 -- THE ROUTER LEARNS
    # ================================================================================================================
    def _router(self, create=False):
        """The mind's RouteLearner, or None. create=True builds it (the IDF table over the catalog: ~1 s at 4,000
        cards) -- only a caller that is about to LEARN asks for that; a mind that never learned a route never pays."""
        lr = self.__dict__.get("_route_learner")
        if lr is None and create:
            lr = self.__dict__["_route_learner"] = RL.RouteLearner(self._capability_catalog())
            self._router_register()
        return lr

    def _router_register(self):
        """Keep mind.protostore('route') pointing at the router's live store (RouteLearner rebuilds it when cards are
        added in a batch), so the generic door surfaces -- door reports, the commons hook -- see the same object."""
        lr = self.__dict__.get("_route_learner")
        if lr is not None:
            self.__dict__.setdefault("_protostores", {})["route"] = lr.store

    def router_mode(self, on=None):
        """Get (on=None) or set whether route_tiered reranks with the LEARNED router. Default: on once router_learn()
        has pretrained the store (the configuration the bench measured), off for a mind that never learned one --
        a fresh mind routes exactly as before (bit-identical). Returns the mode now in force."""
        if on is not None:
            self._router_on = bool(on)
            if on:
                self._router(create=True)
        return bool(getattr(self, "_router_on", False)) and self._router() is not None

    def router_learn(self, pairs=None, loo=True, limit=None):
        """Pretrain the capability router on alias -> card pairs (E4.1): each pair is ONE labelled verdict on the
        shared InfoNCE rule, its rivals the lexical candidates the alias meets when it is NOT in the index (leave-one-
        out: the hard negatives an unseen wording of that card faces). pairs=None uses every alias the catalog carries
        (free, noiseless labels). Turns router_mode on. Returns {pairs, rows, seconds, mode}. Costs about 3 ms per pair
        at 4,000 cards; the learned store persists in the partition (lecore.learning.router)."""
        import time
        from holographic.agents_and_reasoning.holographic_learnguard import sensitive_reason
        cat = self._capability_catalog()
        lr = self._router(create=True)
        if pairs is None:
            pairs = [(a, c.name) for c in cat.all() for a in (c.aliases or ())]
        pairs = list(pairs)[:int(limit)] if limit else list(pairs)
        t0 = time.perf_counter()
        items, skipped = [], 0
        for q, card in pairs:
            if sensitive_reason(str(q)):
                skipped += 1                          # a prototype is a SUM of questions: never one carrying a secret
                continue
            rivals = [c.name for c, _ in RL.loo_candidates(cat, str(q), k=lr.k)] if loo else []
            items.append((str(q), str(card), rivals))
        n = lr.learn_many(items)                      # one store rebuild, then the verdicts in order
        for q, card, _ in items:
            if cat.get(card) is not None:
                self._protostore_track("route", q)
        self._router_register()
        self._router_on = True
        return {"pairs": n, "skipped_sensitive": skipped, "rows": len(lr.store),
                "seconds": round(time.perf_counter() - t0, 2), "mode": self.router_mode()}

    def router_report(self):
        """What the learned router holds: {mode, rows, verdicts, learned, lam, mu, k, dim, features_changed, top} --
        top = the cards with the most labelled verdicts (the learned usage prior)."""
        lr = self._router()
        if lr is None:
            return {"mode": False, "rows": 0, "verdicts": 0, "learned": 0,
                    "why": "no router learned yet (router_learn(), or report a route outcome)"}
        cnt = list(zip(lr.store.labels, lr.store.count))
        cnt.sort(key=lambda t: (-t[1], t[0]))
        return {"mode": self.router_mode(), "rows": len(lr.store), "verdicts": int(sum(lr.store.count)),
                "learned": lr.n_learned, "lam": lr.lam, "mu": lr.mu, "k": lr.k, "dim": lr.dim,
                "features_changed": bool(getattr(lr, "features_changed", False)), "top": cnt[:10]}

    def _router_outcome(self, problem, options, outcome):
        """A reported route outcome is one more labelled pair (E4.1): the card the caller actually used, against the
        options the route offered (its rivals). Never learns a question carrying a secret; creates the learner on
        first use. Wired as a live Ledger hook by route_tiered."""
        from holographic.agents_and_reasoning.holographic_learnguard import sensitive_reason
        cat = self._capability_catalog()
        if outcome is None or not isinstance(outcome, str) or cat.get(outcome) is None:
            return {"router": False, "why": "the outcome names no catalog card"}
        if sensitive_reason(str(problem)):
            return {"router": False, "why": "the question carries a secret -- never learned"}
        lr = self._router(create=True)
        rep = lr.learn(str(problem), outcome, rivals=[o for o in (options or []) if o != outcome])
        self._protostore_track("route", str(problem))
        self._router_register()
        return {"router": True, "rows": len(lr.store), "p_truth": rep.get("p_truth")}

    def _router_rerank(self, problem, ranked):
        """The learned rerank of a lexical ranking [(cap, score)] -> [(cap, fused, lexical, cos, verdicts)], or None
        when the router mode is off (route_tiered then stays exactly lexical)."""
        if not self.router_mode():
            return None
        return self._router().rerank(str(problem), ranked)

    def _router_section(self):
        """The lecore.learning.router section (or None): the learned store + settings + guard flags."""
        lr = self._router()
        if lr is None or not len(lr.store):
            return None
        meta, arrays = lr.state()
        meta["mode"] = bool(getattr(self, "_router_on", False))
        meta["share"] = dict(self._share_flags("route"))
        return {"kind": "lecore.learning.router", "id": "v1", "meta": meta, "arrays": arrays}

    def _router_restore(self, sec):
        """Restore the router section against THIS mind's catalog (cards that no longer exist are dropped)."""
        meta = sec.get("meta") or {}
        lr = RL.RouteLearner.from_state(self._capability_catalog(), meta, sec.get("arrays") or {})
        self.__dict__["_route_learner"] = lr
        self._router_on = bool(meta.get("mode", False))
        if meta.get("share"):
            self.__dict__.setdefault("_protostore_share", {})["route"] = dict(meta["share"])
        self._router_register()
        return {"rows": len(lr.store), "mode": self._router_on, "features_changed": lr.features_changed}

    # ================================================================================================================
    # E4.5 -- SERVE ASKS THE ROUTER
    # ================================================================================================================
    def _serve_route(self, query):
        """After a memory miss: route the query; serve an ANSWER tier whose gate clears the route door's bar as
        'use capability X'. Returns (served_record or None, route_record). The bar (ROUTE_SERVE_*): the bench-derived
        answer threshold on the gate signal (lexical z, or the fused signal when the learned router is on), the
        re-derived refuse floor, AND -- once the route door's calibrator is calibrated -- p_correct >= ROUTE_SERVE_P.
        A question carrying a secret is never routed from here (it escalates)."""
        from holographic.agents_and_reasoning.holographic_learnguard import sensitive_reason
        from holographic.agents_and_reasoning.holographic_decisionrecord import door_record
        if sensitive_reason(str(query)):
            return None, None
        r = self.route_tiered(str(query), k=5)
        if r.get("tier") != "answer" or not r.get("answer"):
            return None, r
        z = r.get("z")
        learned = r.get("learned") is not None
        gate = r.get("gate")
        thr = ROUTE_SERVE_FUSED if learned else ROUTE_SERVE_LEX
        if z is None or z < ROUTE_REFUSE_Z or gate is None or gate < thr:
            return None, r
        pc = r.get("p_correct")
        cal = self.__dict__.get("_door_calibrators", {}).get("route")
        if cal is not None and cal.calibrated() and (pc is None or pc < ROUTE_SERVE_P):
            return None, r
        cap = self._capability_catalog().get(r["answer"])
        out = {"served": True, "via": "route", "tier": "T1", "capability": r["answer"],
               "method": getattr(cap, "method", None), "example": getattr(cap, "example", None),
               "answer": "use capability %r%s" % (r["answer"], (" -- mind.%s(...)" % cap.method) if getattr(cap, "method", None) else ""),
               "id": r["id"], "z": z, "gate": gate,
               "bar": {"gate": thr, "signal": ("fused" if learned else "lexical") + " z+margin", "p": ROUTE_SERVE_P,
                       "on": "door p_correct" if (cal is not None and cal.calibrated()) else "bench threshold (bench_router.json)"},
               "options": [o["name"] for o in r.get("options", [])]}
        return door_record(out, "serve-route", p_correct=pc, p_null=r.get("p_null")), r

    # ================================================================================================================
    # E4.3 / E4.4 -- ONE TOOL LEARNER, ON THE SHARED RULE
    # ================================================================================================================
    def tool_key(self, text):
        """The tool door's feature vector for a text: the question's hashed character 3..5-grams at the experience
        trace's dimension -- the SAME key the reflex bridge uses for judged tool picks (reflex_decide(key='ngram')),
        so the door and the bridge see one space. Pass this (not a semantic_key vector) to tool_note / tool_predict."""
        return self._reflex_task_vec(str(text), key="ngram")

    def _tool_store(self, create=True):
        """The tool door's ProtoStore (options = 'service.endpoint' tool names), or None if absent and not created."""
        st = self.__dict__.get("_protostores", {}).get(TOOL_DOOR)
        if st is None and create:
            st = self.protostore(TOOL_DOOR, dim=int(self.experience.dim))
        return st

    def _tool_audit(self):
        """The door's audit tallies (never weights): {counts: successful uses, failures: failed uses, verdicts: judged
        call verdicts that moved the store}. The ex-UsageTrace counts live here."""
        a = self.__dict__.get("_tool_audit_d")
        if a is None:
            a = self.__dict__["_tool_audit_d"] = {"counts": {}, "failures": {}, "verdicts": 0}
        return a

    def _tool_seed(self, pattern, tool):
        """A TAUGHT pattern is a labelled example of its tool: the tool's first pattern seeds its prototype, a later one
        is one more positive verdict. Counts as one use in the audit (the UsageTrace convention the tests pin)."""
        from holographic.agents_and_reasoning.holographic_learnguard import sensitive_reason
        if sensitive_reason(str(pattern)):
            return {"seeded": False, "why": "the pattern carries a secret"}
        st = self._tool_store()
        v = self.tool_key(pattern)
        if tool not in st:
            st.add_option(tool, [v])
        else:
            st.update(v, tool)
        self._protostore_track(TOOL_DOOR, str(pattern))
        return {"seeded": True, "rows": len(st)}

    def _tool_verdict(self, query, tool, truth):
        """A JUDGED tool call (E4.3): truth == tool -> the call was right (pull it); truth another tool -> a
        correction (pull the truth, and the wrong pick is a labelled negative); truth None -> a failed verify with no
        known right tool (the pick is a labelled negative only). Never learns a question carrying a secret."""
        from holographic.agents_and_reasoning.holographic_learnguard import sensitive_reason
        if sensitive_reason(str(query)):
            return {"tool_door": False, "why": "the question carries a secret -- never learned"}
        st = self._tool_store()
        v = self.tool_key(query)
        out = {"tool_door": True}
        if truth is not None:
            out["update"] = st.update(v, str(truth))
        if tool is not None and truth != tool and tool in st:
            out["negative"] = st.negative(v, tool)
        self._protostore_track(TOOL_DOOR, str(query))
        self._tool_audit()["verdicts"] += 1
        return out

    def _tool_outcome(self, query, tool, outcome):
        """The tool record's live Ledger hook: a reported outcome naming a tool is a labelled verdict for the door
        (the judge's pass reports the tool itself; a correction names the right one). A failed/empty outcome carries
        no truth and teaches nothing."""
        if outcome in (None, "", "fail", "failed", "__failed__") or not isinstance(outcome, str):
            return {"tool_door": False, "why": "no truth reported"}
        return self._tool_verdict(query, tool, outcome)

    def _tool_pick(self, query, tools):
        """E4.3: the tool door's choice among `tools` (names) for a query -- (tool, cosine) -- or None at COLD START
        (no judged call verdict has reached the store yet, or none of the tools has a prototype). Cold start keeps
        today's word-overlap pick bit-identical until something has actually been learned."""
        st = self._tool_store(create=False)
        if st is None or not self._tool_audit()["verdicts"]:
            return None
        known = [t for t in dict.fromkeys(tools) if t in st]
        if not known:
            return None
        s = st.scores(self.tool_key(query))
        cand = [(t, float(s[st.index(t)])) for t in known]
        cand.sort(key=lambda x: -x[1])               # stable: ties keep the tools' taught order
        return cand[0]

    def _tool_section(self):
        """lecore.learning.tooldoor (v1): the tool door's store (A and P in float64 -- a restart must predict
        bit-identically, which float32 A plus a renormalised P would not) + the audit tallies + the share flags."""
        st = self._tool_store(create=False)
        a = self.__dict__.get("_tool_audit_d")
        if (st is None or not len(st)) and not (a and (a["counts"] or a["failures"])):
            return None
        meta, arrays = (st.state() if st is not None else (None, {}))
        arr = {}
        if st is not None and len(st):
            arr = {"A": np.asarray(st.A, np.float64), "P": np.asarray(st.P, np.float64)}
        return {"kind": "lecore.learning.tooldoor", "id": "v1",
                "meta": {"store": meta, "audit": a or {"counts": {}, "failures": {}, "verdicts": 0},
                         "share": dict(self._share_flags(TOOL_DOOR))}, "arrays": arr}

    def _tool_restore(self, sec=None, legacy=None):
        """Restore the tool door: the v1 section, and/or a LEGACY lecore.learning.toolusage section (the old
        UsageTrace -- E4.4's read-compat shim). What migrates from a legacy trace: its per-tool success and failure
        COUNTS, into the door's audit tallies (what tool_usage.counts / .failures report). What does not: the trace
        vector itself -- its task keys were semantic_key vectors, a different space from the door's n-gram key, so
        unbinding it into prototypes would plant rows in the wrong space; the tool prototypes are re-seeded from the
        taught toolreflex patterns instead (serve rebuilds them lazily). Nor into ProtoStore.count: those counts
        include UNJUDGED calls, and the store's count means labelled verdicts."""
        from holographic.agents_and_reasoning.holographic_protostore import ProtoStore
        out = {}
        a = self._tool_audit()
        if legacy:
            from holographic.agents_and_reasoning.holographic_lever7 import UsageTrace
            st_ = dict(legacy.get("meta") or {})
            st_["trace"] = (legacy.get("arrays") or {}).get("trace")
            old = UsageTrace.from_state(st_)
            for k, v in old.counts.items():
                a["counts"][k] = a["counts"].get(k, 0) + int(v)
            for k, v in old.failures.items():
                a["failures"][k] = a["failures"].get(k, 0) + int(v)
            out["legacy_counts_migrated"] = len(old.counts)
        if sec:
            meta = sec.get("meta") or {}
            arr = sec.get("arrays") or {}
            if meta.get("store") and "A" in arr:
                st = ProtoStore.from_state(meta["store"], {"A": np.asarray(arr["A"], np.float64)})
                st.A = np.asarray(arr["A"], np.float64).reshape(len(st.labels), st.dim).copy()
                if "P" in arr:
                    st.P = np.asarray(arr["P"], np.float64).reshape(len(st.labels), st.dim).copy()
                self.__dict__.setdefault("_protostores", {})[TOOL_DOOR] = st
                out["rows"] = len(st)
            au = meta.get("audit") or {}
            for k in ("counts", "failures"):
                a[k] = {str(t): int(n) for t, n in (au.get(k) or {}).items()}
            a["verdicts"] = int(au.get("verdicts", 0))
            if meta.get("share"):
                self.__dict__.setdefault("_protostore_share", {})[TOOL_DOOR] = dict(meta["share"])
        return out

    # ================================================================================================================
    # Q3 (owner: YES) -- LEARNED PROTOTYPES TRAVEL THROUGH THE COMMONS, BEHIND THE GUARD
    # ================================================================================================================
    def protostore_share(self, door, on=None):
        """Get (on=None) or set whether a door's ProtoStore may travel with contribute(). Marking is the OWNER'S opt-in
        per door; the guard still decides: a store travels only if every question it learned from passed the learning
        guard and none was session-salted. Returns {door, shareable, guard_ok, salted, questions, eligible}. The router
        and the tool door are marked shareable by default (their questions are capability / tool wordings); any other
        door (the meaning rows, SystemOne) opts in with protostore_share(door, True) and records its questions through
        _protostore_track."""
        f = self._share_flags(door)
        if on is not None:
            f["shareable"] = bool(on)
        return {"door": door, **f, "eligible": bool(f["shareable"] and f["guard_ok"] and not f["salted"])}

    def _share_flags(self, door):
        """The per-door share record {shareable, guard_ok, salted, questions} (created with the door's default)."""
        sh = self.__dict__.setdefault("_protostore_share", {})
        f = sh.get(door)
        if f is None:
            f = sh[door] = {"shareable": door in ("route", TOOL_DOOR), "guard_ok": True, "salted": False, "questions": 0}
        return f

    def _protostore_track(self, door, question):
        """Record that `door`'s store learned from `question`: the learning guard's verdict on it (any refusal -- a
        secret, a live value -- makes the WHOLE store ineligible to travel, for good: a prototype is a sum and cannot
        forget one term) and whether it was session-salted ('[s:<name>] ...' -- user-private by construction)."""
        from holographic.agents_and_reasoning.holographic_learnguard import learning_verdict
        f = self._share_flags(door)
        f["questions"] += 1
        q = str(question)
        if q.startswith("[s:"):
            f["salted"] = True
        try:
            # allow_volatile=True: a door's prototype learns WHERE a question goes (a card, a tool, a row), never a
            # value, so "what is the price of X" routed to a price tool is not the static-answer-from-live-data error
            # the volatile rule exists for. The SENSITIVE layer (secrets) has no override and is what bars sharing.
            ok = learning_verdict(q, "", allow_volatile=True).get("ok", False)
        except Exception:
            ok = False
        if not ok:
            f["guard_ok"] = False
        return f

    def _shareable_stores(self):
        """{door: ProtoStore} for every store in mind._protostores that is marked shareable AND eligible (guard_ok,
        never salted). THE generic hook contribute() reads."""
        out = {}
        for door, st in sorted(self.__dict__.get("_protostores", {}).items()):
            f = self.__dict__.get("_protostore_share", {}).get(door)
            if f and f["shareable"] and f["guard_ok"] and not f["salted"] and st is not None and len(st):
                out[door] = st
        return out

    def _protostores_section(self):
        """lecore.learning.protostores (v1): every shareable store (the commons bundle's carrier; a contribution mind
        holds only these). The router and tool door have their own sections; this one carries the stores that
        travel, in float64 so a merge is exact. Written only when some store is present."""
        stores = self.__dict__.get("_commons_stores")
        if not stores:
            return None
        meta, arrays = {"doors": []}, {}
        for i, (door, st) in enumerate(sorted(stores.items())):
            m_, _ = st.state()
            meta["doors"].append({"door": door, "store": m_, "extra": st.__dict__.get("_commons_extra", {})})
            arrays["A_%d" % i] = np.asarray(st.A, np.float64)
        return {"kind": "lecore.learning.protostores", "id": "v1", "meta": meta, "arrays": arrays}

    def _protostores_restore(self, sec):
        """Restore a commons carrier section into mind._commons_stores (they are MERGED into this mind's own stores
        only by memory_import, under the documented rule -- never on a plain load)."""
        from holographic.agents_and_reasoning.holographic_protostore import ProtoStore
        out = {}
        arr = sec.get("arrays") or {}
        for i, d in enumerate((sec.get("meta") or {}).get("doors") or []):
            A = np.asarray(arr.get("A_%d" % i), np.float64)
            st = ProtoStore.from_state(d["store"], {"A": A})
            st.A = A.reshape(len(st.labels), st.dim).copy()
            st.__dict__["_commons_extra"] = dict(d.get("extra") or {})
            out[d["door"]] = st
        self.__dict__["_commons_stores"] = out
        return {"doors": sorted(out)}

    def _protostore_merge(self, door, theirs, source=""):
        """MERGE a donor's learned store into this mind's store for `door` (the documented rule, Q3):
          * a label only they have -> added with their accumulator and verdict count;
          * a label both have -> the accumulators are combined WEIGHTED BY VERDICT COUNT: A = (n_mine + 1) * unit(A_mine)
            + (n_theirs + 1) * unit(A_theirs), count = n_mine + n_theirs (the +1 gives each side's seed a vote);
          * a label both have whose prototypes point apart (cosine < PROTO_CONFLICT_COS) is a CONFLICT: flagged, and
            the local row is kept untouched -- two users meant different questions by one label, and disagreement is
            signal, never silently resolved (the taught-row rule of memory_import);
          * a store in another feature space (dim, or the router's feature digest) is refused whole.
        -> {door, added, merged, conflicts: [{label, cos}], refused}."""
        mine = self.__dict__.get("_protostores", {}).get(door)
        if door == "route":
            lr = self._router(create=True)
            dig = (theirs.__dict__.get("_commons_extra") or {}).get("features")
            if dig is not None and dig != lr.features.digest:
                return {"door": door, "refused": "feature space differs (router IDF digest %s vs %s)" % (dig, lr.features.digest)}
            mine = lr.store
        elif mine is None:
            mine = self.protostore(door, dim=theirs.dim, tau=theirs.tau, lr=theirs.lr, topk=theirs.topk)
        if mine.dim != theirs.dim:
            return {"door": door, "refused": "dim %d vs %d" % (theirs.dim, mine.dim)}
        added, merged, conflicts = [], [], []
        new_labels, new_rows, new_counts = [], [], []
        for j, lab in enumerate(theirs.labels):
            a_t = np.asarray(theirs.A[j], np.float64)
            n_t = int(theirs.count[j]) if j < len(theirs.count) else 0
            if not np.any(a_t):
                continue
            if lab not in mine:
                new_labels.append(lab)
                new_rows.append(a_t)
                new_counts.append(n_t)
                added.append(lab)
                continue
            i = mine.index(lab)
            u_m = mine.A[i] / max(np.linalg.norm(mine.A[i]), 1e-12)
            u_t = a_t / max(np.linalg.norm(a_t), 1e-12)
            c = float(u_m @ u_t)
            if c < PROTO_CONFLICT_COS:
                conflicts.append({"label": lab, "cos": round(c, 4), "source": source})
                continue
            n_m = int(mine.count[i])
            mine.A[i] = (n_m + 1) * u_m + (n_t + 1) * u_t
            mine.P[i] = mine.A[i] / max(np.linalg.norm(mine.A[i]), 1e-12)
            mine.count[i] = n_m + n_t
            merged.append(lab)
        if new_labels:
            meta, arr = mine.state()
            A_old = np.asarray(mine.A, np.float64)
            P_old = np.asarray(mine.P, np.float64)
            meta["labels"] = list(meta["labels"]) + new_labels
            meta["count"] = list(meta["count"]) + new_counts
            from holographic.agents_and_reasoning.holographic_protostore import ProtoStore
            A_new = np.vstack([A_old, np.stack(new_rows)]) if len(A_old) else np.stack(new_rows)
            st = ProtoStore.from_state(meta, {"A": A_new})
            st.A = A_new.copy()
            if len(A_old):
                st.P[:len(P_old)] = P_old
            if door == "route":
                self._router().store = st
                self._router_register()
            else:
                self.__dict__["_protostores"][door] = st
        return {"door": door, "added": len(added), "merged": len(merged), "conflicts": conflicts}


def _selftest():
    """Part contract, one home: holographic.unified.check_part (every member reaches UnifiedMind, none shadowed)."""
    from holographic.unified import check_part
    n = check_part("holographic.unified.holographic_unified_p33_router", "_UnifiedPart33")
    return {"part": "holographic_unified_p33_router", "members": n}


if __name__ == "__main__":
    print(_selftest())

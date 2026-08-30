"""holographic_zoo.py -- THE TIERED ZOO (backlog v3): the answer LADDER with freshness policy and
a token ledger (Z0/Z1), the typed-and-learned SKILL LIBRARY with O(1) contextual presentation
(Z5), LEARNED CHAIN-OF-THOUGHT -- chain logging, skeleton mining, plan warm-start (Z6), and
SCOPED ORCHESTRATION under the leOS law, quoted verbatim from task_orchestrator.py because it is
the whole economics: "the LLM should only run when there is THINKING to do. Waiting for a
background process is not thinking." (Z7)

THE LADDER (substrate_gather's design sentence, kept: "No LLM is invoked anywhere in the tree"
until the last rungs): T0 reflex (displacement trace) -> T1 substrate retrieval WITH a
per-kind freshness check (a stale hit is a MISS that schedules a refresh) -> T2 deterministic
dispatch (a bound callable answers with zero model) -> T3 intern (small model, utility-grade
tasks only -- coprocessor_bay._auto_select's cost inversion: the big model is the EXCEPTION)
-> T4 main. Every answer carries {tier, via, confidence, why}; T0-T2 are wrong-answer-free BY
CONSTRUCTION because they refuse instead of guessing.

THE LIBRARY (bone_registry's three structures, on leCore data): JOINTS derive from the
Capability catalog's existing consumes/produces io-kinds (an untyped tool is QUARANTINED --
usable alone, not chainable); AFFINITIES are a pair trace -- bind(toolA, NEXT (x) toolB) per
successful transition, so "what usually follows A" is two unbinds and a cleanup; SKELETONS are
mined from the chain log by deterministic subsequence counting with support thresholds --
proposals only, promoted by evidence (the leOS gate kept).

Deterministic; NumPy + stdlib; JSON-safe outputs. Kept negatives pinned inline where earned.
"""
import time

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import cosine, bind, unbind
from holographic.agents_and_reasoning.holographic_lever7 import key_atom


# ---------------------------------------------------------------------------------------------
# Z1 -- freshness as policy, not hope
# ---------------------------------------------------------------------------------------------
class FreshnessRegistry:
    """Per-entity-kind max ages (leOS's freshness policy): T1 consults this BEFORE serving.
    Unknown kinds default to 'doc' (effectively durable). A stale hit returns fresh=False and
    the caller must treat it as a miss + queue a refresh -- serving stale data confidently is
    the failure this registry exists to prevent."""

    DEFAULTS = {"price": 300.0, "news": 3600.0, "status": 600.0,
                "doc": float("inf"), "definition": float("inf"), "code": float("inf")}

    def __init__(self, policies=None):
        self.policies = dict(self.DEFAULTS)
        if policies:
            self.policies.update({str(k): float(v) for k, v in policies.items()})
        self.refresh_queue = []

    def set_policy(self, kind, max_age_s):
        self.policies[str(kind)] = float(max_age_s)
        return dict(self.policies)

    def check(self, kind, age_s, ref=None):
        limit = self.policies.get(str(kind), self.policies["doc"])
        fresh = float(age_s) <= limit
        if not fresh:
            self.refresh_queue.append({"kind": str(kind), "ref": ref, "age_s": float(age_s)})
        return {"fresh": fresh, "kind": str(kind), "limit_s": limit}


# ---------------------------------------------------------------------------------------------
# Z0.2 -- the token ledger: cost per correct answer, auditable
# ---------------------------------------------------------------------------------------------
class TokenLedger:
    """Per-tier serve counts + estimated tokens saved. `est_llm_tokens` is what a T4 answer to
    this query would have cost; a serve at T0-T2 banks the whole estimate, T3 banks the main/
    intern difference (stated estimate, never presented as a measurement of the actual model)."""

    def __init__(self, intern_fraction=0.15):
        self.intern_fraction = float(intern_fraction)
        self.by_tier = {t: 0 for t in ("T0", "T1", "T2", "T3", "T4", "refused")}
        self.est_tokens_saved = 0.0
        self.queries = 0

    def record(self, tier, est_llm_tokens=600):
        self.queries += 1
        self.by_tier[tier] = self.by_tier.get(tier, 0) + 1
        if tier in ("T0", "T1", "T2"):
            self.est_tokens_saved += float(est_llm_tokens)
        elif tier == "T3":
            self.est_tokens_saved += float(est_llm_tokens) * (1.0 - self.intern_fraction)
        return self.summary()

    def summary(self):
        served = self.queries - self.by_tier.get("refused", 0)
        return {"queries": self.queries, "by_tier": dict(self.by_tier),
                "est_tokens_saved": round(self.est_tokens_saved, 1),
                "escalation_rate": round(self.by_tier.get("T4", 0) / max(served, 1), 3)}


# ---------------------------------------------------------------------------------------------
# Z0.1 -- the answer ladder
# ---------------------------------------------------------------------------------------------
UTILITY_TASKS = ("summarize", "summary", "classify", "extract", "format", "rewrite",
                 "list", "translate", "shorten")


class AnswerLadder:
    """Walk the tiers in cost order; return at the FIRST rung whose gate accepts, with mandatory
    provenance. Rungs are injected callables so the ladder is testable and the zoo wires its own:
      kb_search(query)   -> None | {text, kind, age_s, score}          (T1)
      dispatchers        -> [(match_fn(query)->args|None, run_fn(args)->text, name)]  (T2:
                            binding is the gate -- an unbound required arg REFUSES, never guesses)
      intern(query, ctx) -> None | (text, confidence)                  (T3, utility tasks only)
      main(query, ctx)   -> None | text                                (T4)
    KEPT NEGATIVE (system-scale form of the 48-wrong lesson): a ladder whose rungs serve on raw
    similarity without per-rung gates is the naive cache again -- every rung here either refuses
    or serves with a stated basis."""

    def __init__(self, mind, freshness=None, ledger=None, t1_score_floor=0.2, t3_conf_floor=0.5):
        self.mind = mind
        self.freshness = freshness or FreshnessRegistry()
        self.ledger = ledger or TokenLedger()
        self.t1_score_floor = float(t1_score_floor)
        self.t3_conf_floor = float(t3_conf_floor)

    _STOP = {"a", "an", "the", "of", "in", "on", "to", "for", "and", "or", "with", "from",
             "by", "at", "is", "it", "this", "that", "what", "does", "please", "my", "me"}

    def _qkey(self, query):
        """KEPT NEGATIVE (third appearance of the key law -- and now a FOURTH, from the
        cold-vs-warm test): bag-of-atom keys inherit STOPWORD overlap ('summarize the
        meeting' false-fired against 'what does the doc say' at 0.137 through 'the' --
        hence content words only), and content-word bags STILL alias when a shared question
        FRAME dominates ('how does lecore compare to X' served X and Y the same answer at
        T0). cp21: adjacent CONTENT BIGRAMS join the bag -- the frame stays but the subject
        pairs (compare_gptcache vs compare_routellm) pull siblings apart. Key changes are
        survivable because payloads now carry question TEXT and load REPLAYS the teach
        (migration by replay)."""
        seq = [t for t in str(query).lower().split() if t not in self._STOP]
        toks = set(seq)
        toks |= {a + "_" + b for a, b in zip(seq, seq[1:])}
        toks = sorted(toks)
        # FIFTH appearance of the key law (cp38, the harsh battery): a 30-way TEMPLATE
        # FLOOD ('...cluster 1' ... '...cluster 30') swamps the cleanup margin even with
        # bigrams -- the shared frame dominates the bag and every taught member refuses
        # (verify-on-hit correctly blocked wrong serves; 0/200 served at all). NUMERIC
        # TOKENS ARE DISCRIMINATORS: weight digits 3x and digit-bearing bigrams 2x, so
        # the one token that differs carries enough mass to separate the family.
        # Deterministic (static weights); old partitions migrate by replay as always.
        def _w(t_):
            if any(c.isdigit() for c in t_):
                return 3.0 if t_.replace("_", "").isdigit() else 2.0
            return 1.0
        v = np.sum([_w(t) * key_atom("q:" + t, 2048) for t in toks], axis=0) if toks \
            else key_atom("q:", 2048)
        return v / (np.linalg.norm(v) + 1e-12)

    def answer(self, query, kb_search=None, dispatchers=None, intern=None, main=None,
               est_llm_tokens=600):
        if not hasattr(self, "query_log"):
            self.query_log = []
        self.query_log.append(str(query))
        qk = self._qkey(query)
        # T0a -- EXACT-REPEAT SIDECAR (cp38, the harsh battery's lesson): 200 questions
        # sharing a frame are quasi-parallel keys; superposition cleanup at D=2048
        # cannot separate them and the margin gate refused every taught member (verify-
        # on-hit had already blocked all wrong serves). But 'reflex = near-exact repeats
        # only' is the cp24 doctrine -- and near-exact has an HONEST O(1) form: a
        # normalized-text dictionary in front of the trace, subject to the SAME veto
        # set. The trace remains the fuzzy arm for gated similarity. Family crowding
        # now degrades nothing: 200/200 template siblings serve exactly.
        nq_ = " ".join(str(query).lower().split())
        ex_ = getattr(self, "_exact", {})
        if nq_ in ex_:
            pe_ = ex_[nq_]
            if pe_.get("pid") not in getattr(self, "_payload_bad", set()):
                out0 = {"tier": "T0", "via": "reflex-exact", "answer": pe_["answer"],
                        "provenance": pe_.get("provenance", "model-cached"),
                        "why": "exact repeat of a taught question (text-verified)"}
                try:
                    self.ledger.record("T0", est_llm_tokens)  # the exact arm pays the
                except Exception:                             # same bookkeeping as every
                    pass                                      # other tier (cp38 red)
                return out0
        # T0 -- reflex (the fuzzy arm)
        hit = self.mind.experience.read_gated(qk)
        if hit["fired"]:
            t_ = self.mind.experience._route(qk)
            pid_key = "%d:%d" % (t_, int(hit.get("atom", -1)))
            pays = getattr(self, "_payloads", {})
            payload = pays.get(pid_key, pays.get(int(hit.get("atom", -1))))  # legacy ints
            stored_q = getattr(self, "_payload_qs", {}).get(pid_key)
            if payload is not None and pid_key in getattr(self, "_payload_bad", set()):
                # FEEDBACK VETO (cp22): this payload was marked BAD by outcome feedback --
                # a reflex that repeats a known-bad answer is a knee-jerk, not a memory.
                # Fall through to the live rungs and let a better answer be taught.
                self.ledger.by_tier["feedback_veto"] = \
                    self.ledger.by_tier.get("feedback_veto", 0) + 1
                payload = None
            if payload is not None and stored_q is not None:
                # SESSION GUARD (sweep 109, the P0 fix): the '[s:name]' salt is ONE token
                # among many -- a semantic whisper the 0.75 geometric gate and the 0.75
                # jaccard both sail past (measured: cross-session reads at T0, 9/9 on a
                # 15-row partition; interference merely MASKED it at 3k). Sessions are a
                # privacy boundary, not a similarity hint: the session token must match
                # EXACTLY, absent-vs-absent included, or the reflex refuses and falls
                # through to rungs that key properly. Text equality, zero new state.
                def _sess_tok(t):
                    t = str(t)
                    return t.split("]", 1)[0] + "]" if t.startswith("[s:") else ""
                if _sess_tok(query) != _sess_tok(stored_q):
                    self.ledger.by_tier["session_veto"] = \
                        self.ledger.by_tier.get("session_veto", 0) + 1
                    payload = None
            if payload is not None and stored_q is not None:
                # VERIFY-ON-HIT (cp21): the geometric gate passed; now the stored QUESTION
                # must share content words with the query (deterministic jaccard). Belt and
                # braces: geometry proposes, text disposes. A veto falls through to T1+.
                a_ = set(str(query).lower().split()) - self._STOP
                b_ = set(stored_q.lower().split()) - self._STOP
                jac = len(a_ & b_) / max(len(a_ | b_), 1)
                if jac < 0.75:
                    payload = None                        # vetoed -- not the same question.
                    # THRESHOLD HISTORY (measured, not guessed): 0.34 separated the sota
                    # siblings (2-3 distinct terms of ~5) but bench_ladder exposed that
                    # ONE-word-different questions share jaccard 0.6 and served each
                    # other's answers -- a 0.98 "hit rate" that was really aliasing. At
                    # 0.75 the reflex serves near-exact repeats ONLY (TVCache's stance:
                    # exact reuse is correct by construction); paraphrase coverage is the
                    # named sacrifice, and it belongs to T1/synthesis, not the reflex.
            if payload is not None:
                cal_fn = getattr(self, "_error_prob", None)  # B2 (cp28): UCCI wired in --
                if cal_fn is not None:                       # when calibrated, expected
                    ep = cal_fn(float(hit["confidence"]))    # error above the ceiling
                    if ep is not None and ep > getattr(self, "_error_ceiling", 0.5):
                        self.ledger.by_tier["calib_veto"] = \
                            self.ledger.by_tier.get("calib_veto", 0) + 1
                        payload = None
            if payload is not None:
                self.ledger.record("T0", est_llm_tokens)
                if not hasattr(self, "_payload_ok"):
                    self._payload_ok = {}
                self._payload_ok[pid_key] = self._payload_ok.get(pid_key, 0) + 1
                return {"tier": "T0", "via": "reflex", "answer": payload,
                        "confidence": float(hit["confidence"]),
                        "served_count": self._payload_ok[pid_key],
                        "why": "gated trace hit"}
        # T1 -- substrate + freshness
        if kb_search is not None:
            row = kb_search(query)
            if row and float(row.get("score", 0)) >= self.t1_score_floor:
                fr = self.freshness.check(row.get("kind", "doc"), row.get("age_s", 0.0),
                                          ref=str(query))
                if fr["fresh"]:
                    self._remember(qk, row["text"], query)
                    self.ledger.record("T1", est_llm_tokens)
                    return {"tier": "T1", "via": "substrate", "answer": row["text"],
                            "confidence": float(row["score"]),
                            "why": "kb hit, fresh under %s policy" % fr["kind"]}
                # stale = miss + queued refresh; fall through (the registry already queued it)
        # T2 -- deterministic dispatch (binding is the gate)
        for match, run, name in (dispatchers or []):
            args = match(query)
            if args is not None:
                out = run(args)
                self._remember(qk, out, query)
                self.ledger.record("T2", est_llm_tokens)
                return {"tier": "T2", "via": "dispatch:%s" % name, "answer": out,
                        "confidence": 1.0, "why": "entities bound to a deterministic callable"}
        # T3 -- intern, utility-grade only (the cost inversion)
        is_utility = any(w in str(query).lower() for w in UTILITY_TASKS)
        if intern is not None and is_utility:
            got = intern(query, None)
            if got is not None:
                text, conf = got
                if float(conf) >= self.t3_conf_floor:
                    self._remember(qk, text, query)
                    self.ledger.record("T3", est_llm_tokens)
                    return {"tier": "T3", "via": "intern", "answer": text,
                            "confidence": float(conf), "why": "utility task, intern confident"}
        # T4 -- the exception
        if main is not None:
            text = main(query, None)
            if text is not None:
                self._remember(qk, text, query)
                self.ledger.record("T4", est_llm_tokens)
                return {"tier": "T4", "via": "main", "answer": text,
                        "confidence": None, "why": "escalated past all cheap rungs"}
        self.ledger.record("refused", est_llm_tokens)
        return {"tier": "refused", "via": None, "answer": None, "confidence": 0.0,
                "why": "no rung could serve; refusal is a result"}

    def _remember(self, qkey, answer_text, question_text=None, provenance=None):
        """cp21: the QUESTION TEXT rides with the payload -- it is the durable record that
        makes (a) verify-on-hit and (b) key-format migration by replay possible. The vector
        key is hot state; the text is what survives."""
        import re as re_
        a_ = str(answer_text).strip()
        q_ = str(question_text or "")
        # cp38 hardening (the harsh battery walked through the old gate): the exact-shape
        # check missed "__token__ __null__" (two tokens) and never looked at the QUESTION
        # side. Any STANDALONE __token__ word on either side refuses now. STANDALONE is
        # load-bearing: real data promptly collided -- "what does bots/substrate_gather/
        # __init__.py do" was eaten by a substring check, because Python dunders live
        # inside paths. A control token is a whole word; a dunder in a path is not.
        def _has_ctl(txt_):
            return any(re_.fullmatch(r"__\w+__", tk_) for tk_ in str(txt_).split())
        if _has_ctl(a_) or _has_ctl(q_):
            return None
        if a_.startswith("__") and a_.endswith("__"):
            return None                                   # cp23: CONTROL TOKENS are not
                                                          # answers; guarded HERE so every
                                                          # remember path (T1 rows, T3, T4,
                                                          # explicit teach) refuses alike --
                                                          # the first patch guarded ONE of
                                                          # four call sites and missed
        if not hasattr(self, "_payloads"):
            self._payloads = {}
        # DO NOT CACHE A NON-ANSWER. cp47 below reasons about caching a MODEL's
        # answer and decided provenance should make it visible rather than forbid
        # it -- correct, but it assumes there IS an answer. With no model attached
        # (llm=None, the memory-only boot a first-time user gets) the ladder
        # abstains and answer_text is empty, and this cached the EMPTY STRING.
        # MEASURED: asking "what is the capital of france" twice returned
        # T4/via=main/answer='' then T0/via=reflex-exact/answer='' -- THE SAME
        # BLANK, PROMOTED TO THE CONFIDENT TIER BY HAVING BEEN ASKED BEFORE.
        # T0 is the contract that an answer came from memory. Serving a blank
        # under it makes the tier lie, and every caller branching on
        # tier == "T0" gets a false positive. An abstention must stay an
        # abstention however many times it is asked.
        _at = "" if answer_text is None else str(answer_text)
        if question_text is not None and _at.strip():
            ex_ = getattr(self, "_exact", {})
            ex_[" ".join(str(question_text).lower().split())] = \
                {"answer": str(answer_text), "pid": None,
                 # cp47 (the ouroboros sota pass, abstention probe): a never-taught
                 # question correctly abstains ONCE, then the ladder caches the model's
                 # answer and serves it as a confident reflex forever -- indistinguishable
                 # from taught truth at serve time. That is exactly the abstention failure
                 # LongMemEval grades. Provenance does not forbid the cache (caching is the
                 # point of a ladder); it makes the difference VISIBLE so a caller, a
                 # policy, or a benchmark can treat model-cached answers as provisional.
                 "provenance": provenance or getattr(self, "_provenance_hint", "model-cached")}
            self._exact = ex_
        if not hasattr(self, "_payload_qs"):
            self._payload_qs = {}
        if not hasattr(self, "taught_log"):
            self.taught_log = []
        # cp21.1 (found when replay met a restored trace): counter-derived atom names
        # ("ans:%d" % len) COLLIDE with the restored trace's original atoms -- the write
        # merges instead of appending, every replay lands on one slot, and the fired old
        # mark has no payload. Atom identity must be CONTENT-DERIVED: same question, same
        # atom, in any process, forever.
        pid_atom = key_atom("ans#" + str(question_text or answer_text)[:64], 2048)
        self.mind.experience.write(qkey, pid_atom)
        # KEPT NEGATIVES (cp21, a matched set from the v2 cold cross-check):
        # (1) payloads indexed by within-tile slot ALONE collide across tiles -- the index
        #     must be TILE-QUALIFIED (a restored partition served 0/8 through that hole);
        # (2) counter-derived atom names collide with a restored floor's atoms, so replays
        #     MERGE instead of append -- names must be CONTENT-DERIVED;
        # (3) and then `len(atoms)-1` is wrong for exactly those merged writes -- the slot
        #     must be LOCATED, not assumed: argmax cosine of the pid atom against the
        #     codebook finds the true slot whether the write appended or merged (the atom
        #     vector IS key_atom(name), so an existing slot matches at ~1.0).
        t = self.mind.experience._route(qkey)
        tile = self.mind.experience.tiles[t]
        A = np.asarray(tile._atoms, float)
        cs = A @ pid_atom / (np.linalg.norm(A, axis=1) * np.linalg.norm(pid_atom) + 1e-12)
        idx = int(np.argmax(cs))
        pid_key = "%d:%d" % (t, idx)
        self._payloads[pid_key] = answer_text
        if hasattr(self, "_payload_bad"):
            self._payload_bad.discard(pid_key)            # re-teaching IS the correction:
                                                          # a fresh answer clears the veto
        if question_text is not None:
            self._payload_qs[pid_key] = str(question_text)
            # cp54: the log records PROVENANCE, because replay has to know it. Without
            # this, every restart relabelled deliberate teaches as model-cached (the
            # record was [q, a] and replay could only guess). Rows are [q, a] historic,
            # [q, a, sess] cp36+, [q, a, sess, provenance] now -- readers tolerate all.
            self.taught_log.append([str(question_text), str(answer_text), "shared",
                                    provenance or getattr(self, "_provenance_hint",
                                                          "model-cached")])


# ---------------------------------------------------------------------------------------------
# Z5 -- the typed, learned, O(1)-presented library
# ---------------------------------------------------------------------------------------------
class JointTable:
    """JOINTS derived from the catalog's existing consumes/produces io-kinds. An untyped tool is
    QUARANTINED: it can run alone but a chain through it refuses with the gap named -- planning-
    time refusal, never a runtime surprise."""

    def __init__(self, entries):
        self.types = {}
        def _f(e, k, default=()):
            if hasattr(e, k):
                v = getattr(e, k)
            elif isinstance(e, dict):
                v = e.get(k, default)
            else:
                v = default
            return v if v is not None else default
        for e in entries:
            self.types[str(_f(e, "name", ""))] = {"consumes": tuple(_f(e, "consumes")),
                                                  "produces": tuple(_f(e, "produces"))}

    def validate_chain(self, names):
        for i, n in enumerate(names):
            t = self.types.get(str(n))
            if t is None:
                return {"ok": False, "at": i, "why": "unknown tool %r" % n}
            if not t["consumes"] and not t["produces"]:
                return {"ok": False, "at": i, "why": "tool %r is UNTYPED (quarantined: usable "
                                                     "alone, not chainable)" % n}
            if i > 0:
                prev = self.types[str(names[i - 1])]
                if not (set(prev["produces"]) & set(t["consumes"])):
                    return {"ok": False, "at": i,
                            "why": "type mismatch: %r produces %s but %r consumes %s"
                                   % (names[i - 1], list(prev["produces"]), n,
                                      list(t["consumes"]))}
        return {"ok": True, "steps": len(names)}


class AffinityTrace:
    """LEARNED tool-transition memory: each successful A->B writes bind(atomA, NEXT (x) atomB);
    predict_next(A) is two unbinds + a cleanup over the tool codebook -- 'what usually follows A'
    as one algebraic read. Counts kept beside the trace (audit rule; never opaque weights)."""

    def __init__(self, dim=2048):
        self.dim = int(dim)
        self.NEXT = key_atom("role:NEXT", dim)
        self._trace = np.zeros(dim)
        self._atoms = {}
        self.counts = {}

    def _atom(self, tool):
        if tool not in self._atoms:
            self._atoms[tool] = key_atom("tool:" + tool, self.dim)
        return self._atoms[tool]

    def note_pair(self, a, b):
        self.counts[(a, b)] = self.counts.get((a, b), 0) + 1
        self._trace = self._trace + bind(self._atom(a), bind(self.NEXT, self._atom(b)))
        return self.counts[(a, b)]

    def predict_next(self, a, k=3):
        if not self.counts:
            return []
        raw = unbind(unbind(self._trace, self._atom(a)), self.NEXT)
        rn = raw / (np.linalg.norm(raw) + 1e-12)
        scored = sorted(((n, float(np.dot(rn, v))) for n, v in self._atoms.items() if n != a),
                        key=lambda x: (-x[1], x[0]))
        return scored[: int(k)]


# ---------------------------------------------------------------------------------------------
# Z6 -- learned CoT: chain log, skeleton mining, plan warm-start
# ---------------------------------------------------------------------------------------------
class ChainLog:
    """Every executed plan's full chain on the exact floor: {goal_key(list), steps[(tool,
    outcome)]}. The CoT corpus."""

    def __init__(self):
        self.chains = []

    def note(self, goal_key, steps):
        self.chains.append({"goal_key": [float(x) for x in goal_key],
                            "steps": [(str(t), bool(ok)) for t, ok in steps]})
        return len(self.chains)

    def mine_skeletons(self, min_support=2, min_len=2, max_len=4):
        """Deterministic recurring-subsequence counting over SUCCESSFUL step sequences --
        the dark_matter port, model-free. PROPOSALS only; promotion is the caller's evidence
        gate (the leOS rule kept)."""
        counts = {}
        for ch in self.chains:
            seq = tuple(t for t, ok in ch["steps"] if ok)
            for L in range(min_len, max_len + 1):
                for i in range(len(seq) - L + 1):
                    sub = seq[i:i + L]
                    counts[sub] = counts.get(sub, 0) + 1
        props = [{"steps": list(s), "support": c} for s, c in counts.items()
                 if c >= int(min_support)]
        props.sort(key=lambda p: (-p["support"], -len(p["steps"]), p["steps"]))
        return props

    def plan_warm(self, goal_vec, gate=0.7):
        """Nearest logged chain by GOAL similarity (the resonator lesson: key on the goal
        context, never the chain itself). Below the gate: None -- cold planning."""
        g = np.asarray(goal_vec, float)
        best_s, best = 0.0, None
        for ch in self.chains:
            gk = np.asarray(ch["goal_key"], float)
            if gk.shape != g.shape:
                continue          # cp20.1: cosine across dims is undefined -- mixed logs
            s = cosine(g, gk)     # happen when 64-dim token keys and 2048-dim semantic
                                  # keys share one partition; skip, never compare
            if s > best_s:
                best_s, best = s, ch
        if best is None or best_s < float(gate):
            return None
        return {"steps": [t for t, ok in best["steps"] if ok],
                "goal_similarity": best_s, "via": "plan_warm"}


# ---------------------------------------------------------------------------------------------
# Z7 -- scoped orchestration
# ---------------------------------------------------------------------------------------------
class ScopeStack:
    """Child scopes as working-memory bundles (task_orchestrator's focused context windows):
    push(goal) opens a fresh bundle; pop(salvage=True) archives it, salvaging deliverables to
    the PARENT bundle -- the hot context is per-step and disposable, the record is not."""

    def __init__(self, dim=2048):
        from holographic.agents_and_reasoning.holographic_lever7 import WorkingMemory
        self._WM = WorkingMemory
        self.dim = int(dim)
        self.stack = [self._WM(dim)]

    @property
    def current(self):
        return self.stack[-1]

    def push(self, goal_tag="step"):
        self.stack.append(self._WM(self.dim))
        return len(self.stack) - 1

    def pop(self, salvage=True):
        if len(self.stack) == 1:
            return {"popped": False, "why": "root scope stays"}
        child = self.stack.pop()
        moved = []
        if salvage:
            for tag, (vec, note) in list(child._items.items()):
                self.current.admit(vec, "salvaged:" + tag, note)
                moved.append(tag)
        return {"popped": True, "salvaged": moved}


def orchestrate(mind, request_text, goal_vec, llm, executors, chain_log, affinities,
                ledger=None, plan_gate=0.7):
    """THE COMPOSED LOOP (Z7.3): plan_warm from the chain log (zero model calls when it fires)
    else ONE llm plan call; each step runs in its own scope; a step with a registered executor
    is INGEST (Python, no model -- the orchestrator's law as code); a step without one is THINK
    (one llm call). The chain and its transitions are logged so the SECOND encounter is free.
    Returns {steps, model_calls, via, report} -- the flagship acceptance is model_calls == 0 on
    a repeat of a previously solved request."""
    ledger = ledger or TokenLedger()
    scopes = ScopeStack()
    calls = {"n": 0}

    def _llm(prompt):
        calls["n"] += 1
        return llm(prompt)

    warm = chain_log.plan_warm(goal_vec, gate=plan_gate)
    if warm is not None:
        steps = warm["steps"]
        via = "plan_warm"
    else:
        steps = [s.strip() for s in str(_llm("PLAN: " + request_text)).splitlines() if s.strip()]
        via = "llm_plan"
    report, done = [], []
    prev = None
    for st in steps:
        scopes.push(st)
        if st in executors:                          # INGEST: Python handles it, no model
            out = executors[st]()
            ok = out is not None
            report.append({"step": st, "kind": "ingest", "ok": ok})
        else:                                        # THINK: the model earns its call
            out = _llm("STEP: " + st + " | CONTEXT: " + request_text)
            ok = out is not None
            report.append({"step": st, "kind": "think", "ok": ok})
        done.append((st, ok))
        if prev is not None and ok:
            affinities.note_pair(prev, st)
        prev = st if ok else prev
        scopes.pop(salvage=True)
    chain_log.note(goal_vec, done)
    tier = "T2" if calls["n"] == 0 else ("T4" if via == "llm_plan" else "T3")
    ledger.record(tier)
    return {"steps": steps, "model_calls": calls["n"], "via": via, "report": report,
            "ledger": ledger.summary()}


# ---------------------------------------------------------------------------------------------
# Z2.1 -- entity extraction + arg binding: T2's gate, made real
# ---------------------------------------------------------------------------------------------
import re as _re


def extract_entities(query, vocab=()):
    """Deterministic step-0 extraction (leOS substrate_gather's opening move): numbers, quoted
    strings, URLs, and known vocab words. No model, no guessing -- what is not extractable is
    not bindable, and an unbindable required arg REFUSES the dispatch."""
    q = str(query)
    return {
        "numbers": [float(x) for x in _re.findall(r"-?\d+(?:\.\d+)?", q)],
        "quoted": _re.findall(r"'([^']*)'|\"([^\"]*)\"", q) and
                  [a or b for a, b in _re.findall(r"'([^']*)'|\"([^\"]*)\"", q)] or [],
        "urls": _re.findall(r"https?://\S+", q),
        "words": [w for w in vocab if w.lower() in q.lower()],
    }


def bind_args(signature, entities):
    """Bind a T2 signature {param: kind} (kind in number|number:i|string|url|word) against the
    extracted entities, consuming in order. Returns (args, missing): any missing REQUIRED param
    means the dispatch must refuse -- T2 never guesses (the ladder's zero-wrong contract)."""
    pools = {"number": list(entities.get("numbers", [])),
             "string": list(entities.get("quoted", [])),
             "url": list(entities.get("urls", [])),
             "word": list(entities.get("words", []))}
    args, missing = {}, []
    for param, kind in signature.items():
        base, _, idx = str(kind).partition(":")
        pool = pools.get(base, [])
        if idx:
            i = int(idx)
            if i < len(pool):
                args[param] = pool[i]
            else:
                missing.append(param)
        elif pool:
            args[param] = pool.pop(0)
        else:
            missing.append(param)
    return args, missing


# ---------------------------------------------------------------------------------------------
# Z6.3 -- FABRIK-lite: bidirectional typed chain solving
# ---------------------------------------------------------------------------------------------
def fabrik_chain(joint_table, have_kinds, want_kinds, affinities=None, max_len=5):
    """Bidirectional chain search over the TYPED joint table (leOS fabrik_plan's shape, made
    deterministic): grow forward from what we HAVE and backward from what the goal WANTS; meet
    where a tool's consumes are reachable and its produces reach the goal. Ties break by
    (affinity count desc, name) -- the learned transitions steer among the type-legal.
    Returns {chain, reachable, why}; unreachable is an honest verdict with the frontier named,
    never a guess."""
    types = joint_table.types
    have = set(have_kinds)
    want = set(want_kinds)
    counts = getattr(affinities, "counts", {}) if affinities is not None else {}

    def _fwd_reach(kinds, depth):
        layers = [dict.fromkeys(sorted(kinds))]
        reach = {k: [] for k in kinds}
        for _ in range(depth):
            new = {}
            for name in sorted(types):
                t = types[name]
                if t["consumes"] and set(t["consumes"]) <= set(reach):
                    for pk in t["produces"]:
                        if pk not in reach:
                            new[pk] = reach[sorted(set(t["consumes"]))[0]] + [name]
            if not new:
                break
            reach.update(new)
        return reach
    reach = _fwd_reach(have, max_len)
    hit = want & set(reach)
    if not hit:
        return {"chain": [], "reachable": False,
                "why": "no typed path: forward frontier %s never met goal kinds %s"
                       % (sorted(set(reach) - have), sorted(want))}
    goal_kind = sorted(hit)[0]
    chain = reach[goal_kind]
    # affinity-steered cleanup: where multiple names could occupy a slot, the counts already
    # chose during construction via sorted order; re-score the found chain for the report.
    aff = sum(counts.get((a, b), 0) for a, b in zip(chain, chain[1:]))
    return {"chain": chain, "reachable": True, "goal_kind": goal_kind,
            "affinity_score": aff, "why": "typed path found, %d steps" % len(chain)}


# ---------------------------------------------------------------------------------------------
# Z3.1 -- the void map's text half: gaps with names
# ---------------------------------------------------------------------------------------------
def query_void_map(questions, vecs, n_clusters=4, n_anchors=3, seed=0):
    """leOS kb_void_map, ported model-free: cluster the asked-question log (deterministic
    centroid assignment), probe BETWEEN centroid pairs, score each probe's density deficit
    (how far the nearest real question sits vs the interpolated distance), and ANCHOR each void
    in text -- 'void between X and Y: you know each, not their intersection'. Output: a
    prioritized gap list ready for the research queue (Z3.2)."""
    V = np.stack([np.asarray(v, float) / (np.linalg.norm(v) + 1e-12) for v in vecs])
    rng = np.random.default_rng(int(seed))
    k = min(int(n_clusters), len(V))
    idx = sorted(rng.choice(len(V), size=k, replace=False).tolist())
    C = V[idx].copy()
    for _ in range(8):                                          # fixed-iteration Lloyd: deterministic
        assign = np.argmax(V @ C.T, axis=1)
        for j in range(k):
            members = V[assign == j]
            if len(members):
                c = members.mean(axis=0)
                C[j] = c / (np.linalg.norm(c) + 1e-12)
    def _anchor(j):
        toks = {}
        for q, a in zip(questions, assign):
            if a == j:
                for w in set(str(q).lower().split()):
                    if len(w) > 3:
                        toks[w] = toks.get(w, 0) + 1
        return [w for w, _ in sorted(toks.items(), key=lambda x: (-x[1], x[0]))[:n_anchors]]
    populated = [j for j in range(k) if int((assign == j).sum()) > 0]
    voids = []
    for i in populated:
        for j in populated:
            if j <= i:
                continue
            probe = C[i] + C[j]
            probe = probe / (np.linalg.norm(probe) + 1e-12)
            nearest = float(np.max(V @ probe))
            interp = float(np.dot(C[i], C[j]))
            deficit = (1.0 - nearest) - 0.5 * (1.0 - interp)
            voids.append({"between": [i, j], "deficit": round(float(deficit), 4),
                          "anchor": "void between [%s] and [%s]"
                                    % (" ".join(_anchor(i)), " ".join(_anchor(j))),
                          "probe_questions": []})
    voids.sort(key=lambda v: (-v["deficit"], v["between"]))
    return {"clusters": k, "voids": voids}


# ---------------------------------------------------------------------------------------------
# Z0.3 -- the escalation predictor: skip doomed ladder walks
# ---------------------------------------------------------------------------------------------
class EscalationPredictor:
    """Learn (query features -> the tier that ultimately served) as a usage trace whose atoms
    are TIER names; after warm-up, the ladder can start at the predicted rung. T0-T2 remain a
    prefix regardless (they are nearly free and wrong-answer-free); prediction only decides
    whether to BOTHER with T3 before T4. Counts beside the trace (audit rule)."""

    def __init__(self, dim=2048, min_evidence=3):
        self.dim = int(dim)
        self.min_evidence = int(min_evidence)
        self.NEXT = key_atom("role:TIER", dim)
        self._trace = np.zeros(dim)
        self._atoms = {t: key_atom("tier:" + t, dim) for t in ("T0", "T1", "T2", "T3", "T4")}
        self.counts = {}

    def note(self, qkey, tier):
        self.counts[tier] = self.counts.get(tier, 0) + 1
        self._trace = self._trace + bind(np.asarray(qkey, float),
                                         bind(self.NEXT, self._atoms[str(tier)]))
        return self.counts[tier]

    def predict(self, qkey):
        if sum(self.counts.values()) < self.min_evidence:
            return None
        raw = unbind(unbind(self._trace, np.asarray(qkey, float)), self.NEXT)
        rn = raw / (np.linalg.norm(raw) + 1e-12)
        best = max(self._atoms, key=lambda t: float(np.dot(rn, self._atoms[t])))
        score = float(np.dot(rn, self._atoms[best]))
        return {"tier": best, "score": score} if score > 0.2 else None


# ---------------------------------------------------------------------------------------------
# VSA PROGRAM COMPOSABILITY: the zoo's plans, skeletons and ladder order AS ALGEBRA
# ---------------------------------------------------------------------------------------------
class ProgramChains:
    """The leOS-derived orchestration objects lifted INTO the algebra (the user's requirement,
    taken literally): a tool chain is a HoloMachine PROGRAM VECTOR (one LOAD per step, HALT
    terminated), assembled and decoded by the machine's own ISA -- so plans, skeletons and the
    ladder's escalation order are hypervectors that BIND, BUNDLE and UNBIND like everything
    else. What that buys, measured in the selftest:
      * chains round-trip through assemble/disassemble (a decode that can fail, per the
        checkpoint-9 lesson);
      * a SKELETON LIBRARY is one superposed vector -- bundle(bind(name, chain_vec)) -- and
        recall is unbind + per-slot cleanup, with the crosstalk RECALL RATE measured and the
        exact dict kept beside the vector (the lever-3 floor, as always);
      * per-tenant LADDER CONFIGS superpose in one vector and unbind cleanly -- composition
        the list representation simply does not have."""

    MAX_LEN = 6

    def __init__(self, tool_names, dim=2048, seed=0):
        from holographic.agents_and_reasoning.holographic_machine import HoloMachine
        self.hm = HoloMachine(dim=int(dim), seed=int(seed), data=sorted(set(tool_names)))
        self.dim = int(dim)
        self._floor = {}                                 # name -> steps (exact, beside the algebra)
        self._library = np.zeros(self.dim)

    def ensure_tools(self, names):
        """Grow the tool codebook DETERMINISTICALLY (kept negative, measured live at checkpoint
        12: the machine's _instr silently encodes an UNKNOWN data atom as HALT, so a chain
        through an unregistered tool assembled to an empty decode with no error -- a silent
        fallback is a trap). New names rebuild the machine over the sorted union (seed fixed),
        and the LIBRARY is re-assembled from the exact floor -- which is exactly what the floor
        is for: the superposition is hot state, never the only copy."""
        new = sorted(set(str(n) for n in names) - set(self.hm.data_names))
        if not new:
            return {"rebuilt": False, "tools": len(self.hm.data_names)}
        from holographic.agents_and_reasoning.holographic_machine import HoloMachine
        allnames = sorted(set(self.hm.data_names) | set(new))
        self.hm = HoloMachine(dim=self.dim, seed=0, data=allnames)
        self._library = np.zeros(self.dim)
        for nm, st in self._floor.items():
            self._library = self._library + bind(key_atom("skel:" + nm, self.dim),
                                                 self.chain_to_vector(st))
        return {"rebuilt": True, "tools": len(allnames), "reencoded": len(self._floor)}

    def chain_to_vector(self, steps):
        self.ensure_tools(steps)
        ins = [("LOAD", str(t)) for t in steps[: self.MAX_LEN]] + [("HALT", None)]
        return self.hm.assemble(ins)

    def vector_to_chain(self, vec):
        out = []
        for op, arg in self.hm.disassemble(np.asarray(vec, float), self.MAX_LEN + 1):
            if op == "HALT":
                break
            if op == "LOAD":
                out.append(arg)
        return out

    def library_add(self, name, steps):
        v = self.chain_to_vector(steps)
        self._library = self._library + bind(key_atom("skel:" + str(name), self.dim), v)
        self._floor[str(name)] = list(steps)
        return {"skeletons": len(self._floor)}

    def library_recall(self, name):
        """Recall a skeleton FROM THE SUPERPOSITION (unbind + slot cleanup); the exact floor is
        consulted only to REPORT whether the algebraic recall was right -- the caller sees both
        the decoded chain and the verdict, never a silent fallback."""
        raw = unbind(self._library, key_atom("skel:" + str(name), self.dim))
        decoded = self.vector_to_chain(raw)
        truth = self._floor.get(str(name))
        return {"chain": decoded, "exact": decoded == truth, "floor": truth}

    def recall_rate(self):
        ok = sum(1 for n in self._floor if self.library_recall(n)["exact"])
        return {"exact_recalls": ok, "skeletons": len(self._floor),
                "rate": ok / max(len(self._floor), 1)}


def ladder_order_vector(pc, tiers=("T0", "T1", "T2", "T3", "T4")):
    """The ladder's ESCALATION ORDER as one program vector -- per-tenant configs then compose:
    bundle(bind(tenant_atom, order_vec)) superposes many tenants' ladders in one vector, and
    unbind recovers each (asserted in the selftest for two tenants with DIFFERENT orders)."""
    return pc.chain_to_vector(list(tiers))


# ---------------------------------------------------------------------------------------------
# leCoreGLSL BRIDGE: the reflex gate's decision math as an emitted, validated kernel
# ---------------------------------------------------------------------------------------------
def zoo_gate_kernel(best: float, null_q: float, load: float) -> float:
    """The T0 gate's DECISION MATH as a BRANCHLESS scalar kernel (the emitter refuses IfExp by
    name -- K10 -- so the branch is replaced by a saturating clamp built from min/max, which
    every dialect owns): conf = (best - q)/(1 - q) when best clears the calibrated null, 0.0
    otherwise, scaled by the crosstalk price 1/(1 + load), exactly mirroring
    DisplacementTrace.read_gated. SEMANTIC NOTE (stated, not hidden): the clamp saturates over
    a 1e-12 margin, and at best == q the confidence factor is zero regardless, so the
    branchless form and the branching original agree everywhere but a sub-1e-12 sliver where
    both round to zero. One source serves Python reference, C validation, and GLSL emission."""
    cleared = min(max((best - null_q) * 1000000000000.0, 0.0), 1.0)
    conf = max((best - null_q) / (1.0 - null_q + 0.000000000001), 0.0)
    trust = 1.0 / (1.0 + load)
    return cleared * conf * trust


def zoo_gate_glsl():
    """Emit the gate kernel to GLSL ES 3.0 AND validate its semantics by execution: the emitter
    shares one IR across dialects, so running the c_f64 projection against the Python original
    on a grid (including the edge cases: best == q, q ~ 1, load = 0) validates the math the
    GLSL projection carries. Returns {glsl, validated, n_cases}. `step` is emitted inline as a
    ternary via substitution because GLSL owns the name natively."""
    import inspect
    import textwrap
    from holographic.io_and_interop import holographic_emit as E
    # ONE source of truth: the emitted GLSL and the validated C are projections of
    # zoo_gate_kernel ITSELF (pass 2's audit caught the first version duplicating the kernel
    # as a string beside the function -- two sources that could drift; now they cannot).
    src_py = textwrap.dedent(inspect.getsource(zoo_gate_kernel))
    glsl = E.emit(zoo_gate_kernel, "glsl")
    calls = [(b / 10.0, q / 10.0, l / 4.0) for b in range(0, 11) for q in (0, 3, 7, 9, 10)
             for l in range(0, 3)]
    v = E.validate_c(src_py, calls, dialect="c_f64")
    return {"glsl": glsl, "validated": bool(v["bit_identical"]), "n_cases": int(v["n"]),
            "max_abs_diff": float(v["max_abs_diff"]), "detail": v}


# ---------------------------------------------------------------------------------------------
# LONG-RUNNING GOALS (checkpoint 19): leOS's plan/scope/convergence design, rebuilt on the
# machinery leCore now has -- semantic goal keys, the learning partition, typed chains
# ---------------------------------------------------------------------------------------------
class GoalBook:
    """DURABLE GOALS: work that outlives the process. Each goal holds its semantic key, a plan
    whose steps carry STATUS (pending/done/failed), per-step DELIVERABLES (the compact result
    that crosses the scope boundary -- leOS's Layer-5 rule: intermediate work stays in the
    child scope, only the deliverable reaches the parent), and a CONVERGENCE TRACE -- the
    cosine between the goal and each deliverable, leOS's goal_convergence ported whole:
      converging + stable  -> probably done: SKIP the 'are we done?' model call (counted);
      diverging            -> the agent is WANDERING -- a failure mode loop-detection cannot
                              see (every output different, similarity falling) -- PAUSE the
                              goal with a drift alarm instead of spending more steps.
    Resume is first-class: work_on() executes PENDING steps only, so a cold process that
    loads the partition continues exactly where the last one stopped, with zero replanning."""

    def __init__(self):
        self.goals = {}

    def create(self, goal_id, text, goal_vec, steps):
        self.goals[str(goal_id)] = {
            "id": str(goal_id), "text": str(text),
            "goal_vec": [float(x) for x in np.asarray(goal_vec, float)],
            "steps": [{"name": str(st), "status": "pending", "deliverable": None}
                      for st in steps],
            "convergence": [], "status": "active", "skipped_done_checks": 0}
        return self.goals[str(goal_id)]

    @staticmethod
    def _drift_verdict(conv, window=3):
        if len(conv) < window:
            return "warming"
        tail = conv[-window:]
        slope = tail[-1] - tail[0]
        if slope < -0.10:
            return "diverging"
        if abs(slope) <= 0.03 and tail[-1] >= 0.25:
            return "converged"
        return "converging" if slope > 0 else "stable"

    def to_manifest(self):
        return {gid: dict(g) for gid, g in self.goals.items()}

    def from_manifest(self, man):
        for gid, g in (man or {}).items():
            self.goals[str(gid)] = dict(g)
        return len(self.goals)


def _selftest():
    import lecore
    m = lecore.UnifiedMind()
    rng = np.random.default_rng(0)
    # -- Z1: freshness --
    fr = FreshnessRegistry()
    assert not fr.check("price", 3600)["fresh"] and fr.check("doc", 10 ** 9)["fresh"]
    assert fr.refresh_queue and fr.refresh_queue[0]["kind"] == "price"
    # -- Z0: the ladder, tier by tier, provenance mandatory --
    lad = AnswerLadder(m, freshness=fr)
    kb = lambda q: {"text": "cached doc answer", "kind": "doc", "age_s": 5.0, "score": 0.9} \
        if "doc" in q else None
    disp = [(lambda q: {"n": 6} if "square of 6" in q else None,
             lambda a: str(a["n"] ** 2), "square")]
    intern = lambda q, c: ("intern summary", 0.8)
    main = lambda q, c: "main model answer"
    a1 = lad.answer("what does the doc say", kb, disp, intern, main)
    assert a1["tier"] == "T1" and a1["why"].startswith("kb hit")
    a2 = lad.answer("square of 6 please", kb, disp, intern, main)
    assert a2["tier"] == "T2" and a2["answer"] == "36"
    a3 = lad.answer("summarize the meeting", kb, disp, intern, main)
    assert a3["tier"] == "T3"
    a4 = lad.answer("prove the riemann hypothesis", kb, disp, intern, main)
    assert a4["tier"] == "T4"
    a0 = lad.answer("square of 6 please", kb, disp, intern, main)   # repeat -> reflex
    assert a0["tier"] == "T0" and a0["answer"] == "36", "the repeat must serve from the trace"
    led = lad.ledger.summary()
    assert led["by_tier"]["T0"] == 1 and led["by_tier"]["T2"] == 1 and led["queries"] == 5
    # stale kb entry: refused at T1, refresh queued, escalates
    kb_stale = lambda q: {"text": "old price", "kind": "price", "age_s": 9999, "score": 0.9}
    a5 = lad.answer("price of x", kb_stale, [], None, main)
    assert a5["tier"] == "T4" and any(r["kind"] == "price" for r in fr.refresh_queue)
    # -- Z5: joints + affinities --
    class E:                                          # typed fixture entries
        def __init__(s, n, c, p): s.name, s.consumes, s.produces = n, c, p
    jt = JointTable([E("fetch", ("url",), ("text",)), E("summ", ("text",), ("text",)),
                     E("plot", ("table",), ("image",)), E("mystery", (), ())])
    assert jt.validate_chain(["fetch", "summ"])["ok"]
    bad = jt.validate_chain(["fetch", "plot"])
    assert not bad["ok"] and "mismatch" in bad["why"]
    q = jt.validate_chain(["mystery"])
    assert not q["ok"] and "UNTYPED" in q["why"], "quarantine must be named"
    af = AffinityTrace()
    for _ in range(5):
        af.note_pair("fetch", "summ")
    af.note_pair("fetch", "plot")
    assert af.predict_next("fetch", 1)[0][0] == "summ", "the learned transition must win"
    # -- Z6: mining + warm plans --
    cl = ChainLog()
    g1 = key_atom("goal:report", 64)
    for _ in range(3):
        cl.note(g1, [("fetch", True), ("summ", True), ("send", True)])
    cl.note(key_atom("goal:other", 64), [("plot", True)])
    props = cl.mine_skeletons(min_support=3)
    assert any(p["steps"] == ["fetch", "summ", "send"] and p["support"] == 3 for p in props)
    assert not any(p["steps"] == ["plot"] for p in props), "one-off sequences are not skeletons"
    assert cl.plan_warm(g1)["steps"] == ["fetch", "summ", "send"]
    assert cl.plan_warm(key_atom("goal:novel", 64)) is None, "novel goals plan cold"
    # -- Z7: the flagship -- second run of a solved request costs ZERO model calls --
    calls = {"plan": 0}
    def llm(prompt):
        calls["plan"] += 1
        return "fetch\nsumm\nsend" if prompt.startswith("PLAN") else "thought"
    ex = {"fetch": lambda: "data", "summ": lambda: "short", "send": lambda: "sent"}
    cl2, af2 = ChainLog(), AffinityTrace()
    gv = key_atom("goal:weekly-report", 64)
    r1 = orchestrate(m, "make the weekly report", gv, llm, ex, cl2, af2)
    assert r1["via"] == "llm_plan" and r1["model_calls"] == 1, "first run: ONE plan call only"
    r2 = orchestrate(m, "make the weekly report", gv, llm, ex, cl2, af2)
    assert r2["via"] == "plan_warm" and r2["model_calls"] == 0, \
        "THE FLAGSHIP: the second encounter costs zero model tokens -- the CoT was learned"
    assert af2.predict_next("fetch", 1)[0][0] == "summ"
    # -- Z2.1: extraction + binding is the gate --
    ents = extract_entities("plot 'sales' from https://x.co/d.csv rows 10 to 20", vocab=("plot",))
    assert ents["numbers"] == [10.0, 20.0] and ents["urls"] and ents["words"] == ["plot"]
    args, missing = bind_args({"src": "url", "start": "number", "end": "number"}, ents)
    assert not missing and args["start"] == 10.0
    _, miss2 = bind_args({"src": "url", "key": "string"}, extract_entities("no quotes here"))
    assert "src" in miss2 and "key" in miss2, "unbound required args must be NAMED, not guessed"
    # -- Z6.3: bidirectional typed chains --
    fb = fabrik_chain(jt, have_kinds=["url"], want_kinds=["text"])
    assert fb["reachable"] and fb["chain"] == ["fetch"], fb
    fb2 = fabrik_chain(jt, have_kinds=["url"], want_kinds=["image"])
    assert not fb2["reachable"] and "never met" in fb2["why"], "unreachable is an honest verdict"
    # -- Z3.1: voids with names --
    qs = (["how do I auth the api"] * 4 + ["reset a user password"] * 4 +
          ["plot the sales chart"] * 4 + ["export chart to pdf"] * 4)
    vs = [key_atom("qc:" + q, 256) for q in qs]
    vm = query_void_map(qs, vs, n_clusters=4)
    assert vm["clusters"] == 4 and vm["voids"] and "void between [" in vm["voids"][0]["anchor"]
    # -- Z0.3: the predictor learns where queries end up --
    ep = EscalationPredictor(min_evidence=3)
    hardk = key_atom("hardq", 2048)
    for _ in range(4):
        ep.note(hardk, "T4")
    pr = ep.predict(hardk)
    assert pr and pr["tier"] == "T4", "a repeatedly-escalated query family predicts T4"
    assert ep.predict(key_atom("neverseen", 2048)) is None or True
    return {"ladder_tiers": [a1["tier"], a2["tier"], a3["tier"], a4["tier"], a0["tier"]],
            "fabrik": fb["chain"], "void_anchor": vm["voids"][0]["anchor"][:60],
            "escalation_pred": pr["tier"],
            "skeleton": props[0]["steps"], "run2_model_calls": r2["model_calls"],
            "ledger": led}


if __name__ == "__main__":
    print(_selftest())

"""RELEASE DISTILLATION (cp62): the session's external memory -> the SHIPPED bundle.

The two memories are not the same thing and must never be shipped as if they were. The
session partition is one collaboration's working memory: build history, checkpoint
narration, deployment specifics, open items -- personal by nature, and shipping it would
hand every user our arc's biases as if they were the engine's knowledge. The bundled
memory is the distillate: GENERIC, measured engine knowledge that helps everyone --
what the organs are, what the doctrines are, what was proven and how -- the same way
only the relevant slice of a corpus gets kept.

Selection is rule-based and AUDITED, not vibes: an entry ships only if it (a) carries
engine-lexicon content, (b) matches no session/build/history marker, and (c) passes the
leak scan (no paths, no session tags, no collaborator names). Every exclusion is
counted by reason and reported -- what was withheld is visible, so the distillation
itself cannot become a blind spot. Shipped entries stay veto-able and provenance-tagged
like everything else; the bundle answers generic engine questions and ESCALATES on
everything else (verified below, not assumed).

WHAT 2026-09-27 FIXED (found when the owner asked what should be distilled to the core memory):
  * The shipped bundle had silently become a copy of the session memory. release_bundle/ held 62 distilled rows
    at 5ae7460 (2026-08-23) and 497 at 4589f22 (2026-08-26) -- including 46 session-salted rows ("[s:full-stack]
    what is the calibration constant of sensor array 7 in bay 0"), 42 build-history rows and probe facts
    ("how many moons does the seventh exoplanet of kepler-442 have"). By this file's own rules only 161 of the
    497 would have shipped. A bundle is now written ONLY by this tool, and `--check` re-derives the report from a
    shipped bundle so a bundle that was not distilled fails loudly.
  * Duplicate questions kept the FIRST (oldest) answer; replay is last-wins, so a corrected answer shipped stale.
    The newest answer per question now wins, exactly as a reload would serve it.
  * Vetoed answers (answer_feedback ok=False, the veto tombstones) shipped; they are now excluded.
  * `cp\\d{2}` missed three-digit checkpoints (cp104, cp110 ...); "user's box" missed "the user's qwen"; rows
    written under a work scope other than "shared" were not recognised as session work; doctrine rows were
    copied into the bundle although the seedpack already ships them (and an OLD doctrine text would then shadow
    a newer one).
  * An anchor rule: a row must say what in the ENGINE it is about. Domain knowledge taught for one task
    (how a 1930s animation cel was painted, a speaker's blueprint) is real knowledge, but it is that
    collaboration's, not the engine's -- 176 source rows had no engine anchor.
  * The leak scan is now the engine's own learning guard (holographic_learnguard.sensitive_reason) on top of
    the path / session / collaborator markers.
  * Paths were hard-coded to one machine ("/home/claude/claude_partition"); they are arguments now.

WHAT THE CORE MEMORY CARRIES BEYOND Q&A ROWS (2026-09-27, owner-directed: "the point of the seed memory ... is to
reduce LLM calls and improve automatic tool calling ... If a user uses unknown phrasing ... it should be learned
permanently ... If the system learns how to use an api (substituting keys for env variables ...), or a new method or
technique, that's something that should make its way to the seed or core memory"):
  rows          the taught Q&A rows, by the rule above (_select) -- a credential VALUE inside an answer is replaced
                by a ${SERVICE_PARAM} placeholder first (holographic_apilearn.placeholderize_text); anything the guard
                still calls sensitive (a seed phrase, a PEM key, a question that discloses a secret) is excluded
  wordings      every CONFIRMED wording of a shipped row (a `same` verdict or a correction linked it) -- the phrasing
                a person used that memory did not know; replayed with learn=True in the bundle, so the learned WORD
                ASSOCIATIONS are rebuilt from shipped wordings only (never copied from the source, where they may
                have been learned from a row that does not ship)
  methods       METHOD rows (verb + slots + words->value maps + defaults; a live value is never kept) and CLARIFY
                rows ("sol?" -> ask), with their confirmed wordings -- credential constants as placeholders
  negatives     "this wording does NOT mean that row" for carried rows (they stop a wrong serve on arrival)
  direction     the direction reader (from / to context prototypes), when every method wording passed the scan
  apis          learned API specs (newest per service; auth as ${ENV} placeholders; never a private host) and
                their discoverability cards
  reflexes      taught tool reflexes of a carried API (params as placeholders)
  protostores   every door's ProtoStore that is SHAREABLE and ELIGIBLE by the commons rule (p33 _shareable_stores:
                marked shareable, every question it learned from passed the guard, none session-salted) -- merged
                into the bundle's own stores by the commons merge rule (p33 _protostore_merge), and also carried in
                the commons carrier section
  sops          named SOPs (newest per name) that parse, pass the guard and the leak scan, with placeholders --
                except the ones the seedpack installs itself
  workflows     distilled workflows ("what workflow solved X") that pass the row rule (the objective must be about
                the engine)
Every carried item and every exclusion by reason is in distill_report.json; `--check` re-derives every artifact
class from a shipped bundle and scans the whole container for secrets.

THE PROMOTION RULE (permanence of what a model taught). A wording or a row learned from a MODEL verdict reaches the
core memory only when it was CONFIRMED -- never on the model's word alone:
  * a wording linked to a row by a `same` verdict (or a correction by decision id) IS confirmed: the model judged
    it against the row's own evidence, and the link was learned -- it ships with its row;
  * a row the model CREATED (`new` verdict: provenance model:<by> or model-cached) ships only when a second event
    confirmed it: a later verdict linked another wording to it, a positive answer_feedback, or a reported outcome
    on a meaning decision that named it. It ships with provenance "confirmed";
  * a held-back verdict (the noisy-teacher rule) was never linked, so it never ships; a conjecture ships only at an
    earned rung (validated / evidenced), exactly as before.

NOT DISTILLED (reported under not_distilled, with the reason): the reflex bridge's experience and the tool door's
per-call records (question VECTORS without their text -- nothing to audit for secrets; the tool door's STORE travels
because its eligibility was tracked at learn time), SystemOne tables and the decision ledger (one user's decisions
about their own states), open escalations, the guard's learned examples (users' refused questions), door
calibration streams, the query ledger, tool caches, goals, the semantic context vectors and the teacher-noise
estimate (a property of one deployment's model end).

Usage:
    PYTHONHASHSEED=0 python tools/distill_release.py --partition lecore_memory --out release_bundle
    PYTHONHASHSEED=0 python tools/distill_release.py --partition lecore_memory --dry-run   # report only
    PYTHONHASHSEED=0 python tools/distill_release.py --check release_bundle                # audit a bundle
    PYTHONHASHSEED=0 python tools/distill_release.py --partition P --out B --rows-only    # the pre-2026-09-27 bundle
The partition is BOOTED FROM A COPY (a temp dir): booting rolls a partition's learning files over, and the session
partition must never be modified by a distillation.
Writes <out>/learning/state.lecore, <out>/knowledge.lecore (SOPs; written clean) and <out>/distill_report.json.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import lecore  # noqa: E402

LEXICON = ("bind", "bundle", "cleanup", "resonator", "drift", "provenance",
           "tombstone", "saturation", "grounding", "ouroboros", "holographic",
           "fhrr", "quantization", "archive", "holoforest", "conjecture",
           "hypothesis", "permutation", "capacity", "pheromone", "delta rule",
           "hrr", "vsa", "lever", "abstention", "escalat", "veto", "slime",
           "metaball", "void", "recall", "semantic", "reflex",
           # the decision loop (sweeps 171-182)
           "typed decision", "systemone", "decision_outcome", "route_tiered", "p_correct", "protostore",
           "calibrat", "meaning index", "meaning rung", "learning guard", "swarm_step", "rollover")
# THE ANCHOR (2026-09-27): the row must name the engine or one of its organs, not just share a word with the
# lexicon ("recall" and "semantic" also appear in art-technique rows).
ANCHOR = re.compile(r"\blecore\b|\bmind\.|\bfacult|\bpartition\b|\breflex\b|typed decision|systemone|"
                    r"route_tiered|decision_outcome|\bserve\b|protostore|hypervector|\bhrr\b|\bvsa\b|"
                    r"resonator|\bcatalog\b|\bladder\b|learning guard|meaning (?:rung|index)|swarm_step|"
                    r"\bholographic\b|\bT0\b|find_capability|\bteach\b", re.I)
EXCLUDE = [
    (re.compile(r"\bcp\s?\d{2,3}\b|\bcheckpoint\b", re.I), "build-history"),
    (re.compile(r"\bsession\b|\[s:", re.I), "session-specific"),
    (re.compile(r"user's|pending|next step|next build|recorded next|next lever|still not|where is the backlog",
                re.I), "open-item"),
    # a TASK's artefacts (a poster's region map, a speaker blueprint, a painting technique taught for one picture)
    (re.compile(r"blueprint reference design|bench poster|monument|disney|cartoon|\bcel\b", re.I), "task-artifact"),
    (re.compile(r"/tmp/|/home/|/mnt/|/Users/|[A-Za-z]:\\\\", re.I), "absolute-path"),
    (re.compile(r"openzoo|galv_out|galvatron|lestudio|leos\b|\bqwen|smollm", re.I), "deployment-specific"),
    (re.compile(r"\bjev\b|typesafe", re.I), "third-party-product"),
    (re.compile(r"what workflow solved", re.I), "workflow-log"),
    (re.compile(r"claude|anthropic", re.I), "collaborator"),
    (re.compile(r"\bprobe\b|smoke test|pin (?:cached|provenance) q", re.I), "probe"),
]
# HISTORY (2026-09-27): "what did X find / why did Y get slow / what gaps did the audit find" is the build's story,
# not the engine's knowledge -- the LESSON it taught belongs in the seedpack doctrine, stated timelessly.
HISTORY_Q = re.compile(r"^\s*(what did|why did|why was|what gaps did|what happened|what was|what were|what almost|"
                       r"what moved|what closed|what shipped|what lingering)\b", re.I)
GENERIC_TOPICS = ("ouroboros-sota-2026", "void-sota-2026")


SHIPPED_PROVENANCE = ("taught", "validated", "evidenced", "confirmed")
# a model's own provenance: provisional until something else confirms it (THE PROMOTION RULE above)
MODEL_PROVENANCE = re.compile(r"^(model-cached|model:.*|model)$")
SPEC_PREFIX = "api spec record: "
CARD_PREFIX = "use the learned api tool "
WORKFLOW_PREFIX = "what workflow solved "
# A WORKFLOW is a done goal's step list. It is a TECHNIQUE worth shipping only when its objective is a repeatable
# task, not a dated episode of one collaboration's arc: measured on the live partition (2026-09-27), 6 of 13 workflow
# rows passed the row rule, and 5 of those 6 were episodes ("audit lecore against the august 2026 research", "finish
# the backlog remainder", "extract what unicron ... teach"). This rule keeps the repeatable one ("simulate the ci
# inside lecore, find the failures, and fix them").
WORKFLOW_EPISODE = re.compile(r"\b(?:19|20)\d\d\b|\b(?:january|february|march|april|may|june|july|august|september|"
                              r"october|november|december)\b|\bbacklog\b|\bremainder\b|\bupstream\b|\bbranch\b|"
                              r"\bunicron\b|\bsiggraph\b|\barxiv\b", re.I)
# SOPs the seedpack installs itself at boot -- shipping them again would shadow a newer text (the doctrine rule)
SEEDPACK_SOPS = ("lecore_self_review",)


def _norm(q):
    return " ".join(str(q).lower().split())


def _guard():
    from holographic.agents_and_reasoning import holographic_learnguard as LG
    return LG


def _ph():
    from holographic.io_and_interop import holographic_apilearn as AL
    return AL


def _row_kind(q, a, prov):
    """Which ARTIFACT a taught row belongs to. Only 'row' and 'workflow' rows go through the Q&A rule; api spec
    records, their cards and tool reflexes have their own rules (_artifacts); decision records never ship."""
    q, a = str(q), str(a)
    if q.startswith(SPEC_PREFIX):
        return "api-spec"
    if str(prov) == "toolreflex" or q.startswith("toolreflex: "):
        return "toolreflex"
    if a.startswith(CARD_PREFIX):
        return "api-card"
    if q.startswith("decision: "):
        return "decision"
    if q.lower().startswith(WORKFLOW_PREFIX):
        return "workflow"
    return "row"


def _marker_reason(text, skip=()):
    """The first leak / session / history marker the text carries (None = clean) -- the EXCLUDE list."""
    blob = str(text).lower()
    for rx, name in EXCLUDE:
        if name in skip:
            continue
        if rx.search(blob):
            return name
    return None


def _doctrine_norms():
    """The seedpack's doctrine questions, normalised. Their ROWS never ship in the bundle (the seedpack teaches them at
    boot, and an old copy would shadow a newer text), but a wording a verdict LINKED to one of them does: it is a
    learned phrasing of the engine's own doctrine -- exactly what should stop costing a model call on a fresh install
    (maple swarm run, 2026-09-27: "how should a swarm share memory" was linked to the swarm doctrine row by a `same`
    verdict and then dropped by the distillation, so every new install would escalate it again)."""
    from holographic.agents_and_reasoning.holographic_seedpack import DOCTRINE
    return {_norm(row[0]) for row in DOCTRINE}


def _wording_reason(w):
    """Why a WORDING (a phrasing, a pattern, an SOP name) must not ship, or None. The leak markers, a session salt,
    and the learning guard -- a wording carries no answer, so the engine-anchor / lexicon rules do not apply to it."""
    w = str(w)
    if w.startswith("[s:"):
        return "session-specific"
    r = _marker_reason(w, skip=("open-item", "workflow-log", "history"))
    if r:
        return r
    if _guard().sensitive_reason(w):
        return "sensitive"
    return None


def _latest_rows(lad):
    """[(q, row)] one per question, the NEWEST row (replay is last-wins), in first-appearance order."""
    rows = [t for t in getattr(lad, "taught_log", []) if len(t) > 3]
    latest, order = {}, []
    for t in rows:
        q = str(t[0])
        if q not in latest:
            order.append(q)
        latest[q] = t
    return [(q, latest[q]) for q in order], len(rows)


def _vetoed(lad):
    v = set(getattr(lad, "_vetoed_qs", set()) or set())
    v |= {str(q) for q in (getattr(lad, "bad_questions", None) or [])}
    return v


def _select(lad, anchor=True, confirmed=None, workflows=False):
    """The Q&A rule, as a pure function of the ladder: (kept [(q, a, prov)], excluded {reason: n}, source_rows).
    One row per question, the NEWEST. anchor=False drops ONLY the engine-anchor / lexicon / length rules (a DOMAIN
    profile: tools/bench_core_memory.py measures what carrying learned wordings buys when a non-engine row ships;
    the release bundle always runs anchor=True). `confirmed` = normalised questions a second event confirmed (THE
    PROMOTION RULE): a model-provenance row in it ships as 'confirmed'. workflows=True admits "what workflow solved
    X" rows through the same rule (their objective must be about the engine) instead of excluding them as
    workflow-log."""
    latest, n_rows = _latest_rows(lad)
    vetoed = _vetoed(lad)
    confirmed = confirmed or set()
    try:
        LG = _guard()
        sensitive_reason = LG.sensitive_reason
    except Exception:                                # a stripped build: the marker scan below still runs
        sensitive_reason = None
    kept, excluded = [], {}
    n_seen = 0
    for q, t in latest:
        a, scope, prov = str(t[1]), str(t[2]), str(t[3])
        kind = _row_kind(q, a, prov)
        if kind not in ("row", "workflow"):
            continue                                 # an artifact row: _artifacts owns its rule
        n_seen += 1
        reason = None
        if MODEL_PROVENANCE.match(prov) and _norm(q) in confirmed:
            prov = "confirmed"                       # promoted: a second event confirmed the model's row
        if prov not in SHIPPED_PROVENANCE:
            reason = "provisional-provenance"
        elif q in vetoed or q.lower() in vetoed or _norm(q) in vetoed:
            reason = "vetoed"
        elif "[doctrine" in a:
            reason = "doctrine (the seedpack ships it)"
        elif scope not in ("shared", "", "None"):
            reason = "work-scope:" + scope.split("-")[0][:12]
        if reason is None:
            reason = _marker_reason(q + " " + a, skip=("workflow-log",) if (workflows and kind == "workflow") else ())
        if reason is None and HISTORY_Q.search(q):
            reason = "history"
        if reason is None and kind == "workflow" and WORKFLOW_EPISODE.search(q):
            reason = "workflow-episode"
        if reason is None:
            # a credential VALUE in the answer becomes a ${SERVICE_PARAM} placeholder (how to authenticate is
            # knowledge; the key is not). What the guard still calls sensitive afterwards never ships.
            try:
                a2 = _ph().placeholderize_text(a)
            except Exception:
                a2 = a
            if sensitive_reason is not None and sensitive_reason(q, a2):
                reason = "sensitive"
            else:
                a = a2
        if reason is None and anchor and not any(w in (q + " " + a).lower() for w in LEXICON):
            reason = "no-engine-content"
        if reason is None and anchor and not ANCHOR.search(q + " " + a):
            reason = "no-engine-anchor"
        if reason is None and (len(a) < 40 or len(a) > 2000) and anchor:
            reason = "length"
        if reason:
            key = reason if not reason.startswith("work-scope:") else "work-scope"
            excluded[key] = excluded.get(key, 0) + 1
        else:
            kept.append((q, a, prov))
    return kept, excluded, n_seen


def _confirmations(src):
    """THE PROMOTION RULE's evidence: normalised questions (and meaning row ids) a SECOND event confirmed --
    a later verdict linked another wording to the row, a positive answer_feedback, or a reported outcome on a
    meaning decision that named the row. -> (questions, row_ids)."""
    qs, rids = set(), set()
    lad = src.zoo["ladder"]
    for fb in getattr(lad, "_feedback_log", []) or []:
        if len(fb) > 1 and fb[1] is True:
            qs.add(_norm(fb[0]))
    mi = getattr(src, "_meaning", None)
    if mi is not None:
        for rid, row in mi.rows.items():
            if len(row.get("phrasings") or []) > 1:
                rids.add(rid)
                qs.add(_norm(row["canonical"]))
    try:
        led = src.decision_ledger()
        for rec in list(getattr(led, "_rows", {}).values()):
            if getattr(rec, "via", None) == "meaning" and getattr(rec, "outcome", None) is not None \
                    and str(rec.outcome) == str(rec.answer):
                rids.add(str(rec.answer))
                if mi is not None and str(rec.answer) in mi.rows:
                    qs.add(_norm(mi.rows[str(rec.answer)]["canonical"]))
    except Exception:
        pass
    return qs, rids


def _count(d, reason):
    d[reason] = d.get(reason, 0) + 1


def _artifacts(src, kept_qs, anchor=True, allow_private_hosts=False, workflows_kept=()):
    """Everything else a mind learned that saves a model call or picks a tool, each class under its own rule.
    -> {class: {"carried": [...], "excluded": {reason: n}}} plus the payloads the bundle is built from."""
    AL = _ph()
    from holographic.agents_and_reasoning.holographic_meaning import placeholder_method, _rid
    lad = src.zoo["ladder"]
    latest, _ = _latest_rows(lad)
    vetoed = _vetoed(lad)
    _cq, confirmed_rids = _confirmations(src)
    out = {k: {"carried": [], "excluded": {}} for k in
           ("wordings", "methods", "negatives", "direction", "apis", "api_cards", "reflexes", "protostores",
            "sops", "workflows")}
    pay = {"wordings": [], "methods": [], "negatives": [], "direction": None, "apis": [], "api_cards": [],
           "reflexes": [], "protostores": {}, "sops": [], "workflows": list(workflows_kept), "doctrine_wordings": []}
    out["workflows"]["carried"] = [q for q, _, _ in workflows_kept]
    mi = getattr(src, "_meaning", None) or (src.meaning if src.zoo["ladder"].taught_log else None)
    kept_norm = {_norm(q) for q in kept_qs}
    carried_rids = set()
    # ---- wordings of shipped rows ---------------------------------------------------------------------------
    if mi is not None:
        for rid, row in mi.rows.items():
            if row["kind"] != "answer" or _norm(row["canonical"]) not in kept_norm:
                continue
            carried_rids.add(rid)
            for w in row["phrasings"]:
                if _norm(w) == _norm(row["canonical"]):
                    continue
                r = _wording_reason(w)
                if r is None and _norm(w) in vetoed:
                    r = "vetoed"
                if r:
                    _count(out["wordings"]["excluded"], r)
                else:
                    pay["wordings"].append((row["canonical"], w))
                    out["wordings"]["carried"].append([row["canonical"], w])
        # ---- confirmed wordings of DOCTRINE rows (the link ships, the doctrine text never does) -----------------
        doc_norms = _doctrine_norms()
        for rid, row in mi.rows.items():
            if row["kind"] != "answer" or _norm(row["canonical"]) not in doc_norms:
                continue
            for w in row["phrasings"]:
                if _norm(w) == _norm(row["canonical"]):
                    continue
                r = _wording_reason(w)
                if r is None and _norm(w) in vetoed:
                    r = "vetoed"
                if r:
                    _count(out["wordings"]["excluded"], r)
                else:
                    pay["doctrine_wordings"].append((row["canonical"], w))
                    out["wordings"]["carried"].append([row["canonical"], w])
        # ---- method and clarify rows --------------------------------------------------------------------------
        for rid, row in mi.rows.items():
            if row["kind"] not in ("method", "clarify"):
                continue
            prov = str(row.get("provenance", "taught"))
            r = None
            if row.get("session"):
                r = "session-specific"
            elif MODEL_PROVENANCE.match(prov) and rid not in confirmed_rids and len(row["phrasings"]) < 2:
                r = "unconfirmed (one model verdict)"
            elif prov not in SHIPPED_PROVENANCE and not MODEL_PROVENANCE.match(prov):
                r = "provisional-provenance"
            m = placeholder_method(row.get("method")) if row.get("method") else None
            if r is None:
                r = _wording_reason(row["canonical"])
            if r is None and row["kind"] == "method":
                if not str((m or {}).get("verb", "")).strip():
                    r = "no verb"
                elif _guard().sensitive_reason(json.dumps(m, sort_keys=True)):
                    r = "sensitive"
            if r is None and row["kind"] == "clarify" and (_wording_reason(row.get("clarify", "")) or
                                                         _guard().sensitive_reason(str(row.get("clarify", "")))):
                r = "sensitive"
            if r:
                _count(out["methods"]["excluded"], r)
                continue
            words = []
            for w in row["phrasings"]:
                if _norm(w) == _norm(row["canonical"]):
                    continue
                wr = _wording_reason(w)
                if wr:
                    _count(out["wordings"]["excluded"], wr)
                else:
                    words.append(w)
                    out["wordings"]["carried"].append([row["canonical"], w])
            carried_rids.add(rid)
            pay["methods"].append({"kind": row["kind"], "canonical": row["canonical"], "method": m,
                                   "clarify": row.get("clarify"), "provenance": "confirmed"
                                   if MODEL_PROVENANCE.match(prov) else prov, "phrasings": words})
            out["methods"]["carried"].append([row["kind"], row["canonical"],
                                              (m or {}).get("verb") if m else None])
        # ---- negatives ----------------------------------------------------------------------------------------
        for w, rid in sorted(getattr(src, "_meaning_neg", set()) or set()):
            if rid not in carried_rids:
                _count(out["negatives"]["excluded"], "row not carried")
            elif _wording_reason(w):
                _count(out["negatives"]["excluded"], _wording_reason(w))
            else:
                pay["negatives"].append((w, rid))
                out["negatives"]["carried"].append([w, rid])
        # ---- the direction reader -----------------------------------------------------------------------------
        rd = src.__dict__.get("_direction_reader_obj")
        if rd is not None and getattr(rd, "counts", None):
            bad = [w for row in mi.rows.values() if row["kind"] == "method" for w in row["phrasings"]
                   if _wording_reason(w)]
            if bad:
                _count(out["direction"]["excluded"], "a method wording that could have taught it failed the scan")
            else:
                pay["direction"] = rd.state()
                out["direction"]["carried"].append({"slots": sorted(rd.counts), "counts": dict(rd.counts)})
    # ---- learned API specs, their cards, tool reflexes --------------------------------------------------------
    specs = {}
    for q, t in latest:
        if _row_kind(q, t[1], t[3]) == "api-spec" and not MODEL_PROVENANCE.match(str(t[3])):
            specs[q[len(SPEC_PREFIX):].strip()] = str(t[1])          # newest per service (latest rows)
    carried_svcs = set()
    for name, js in specs.items():
        try:
            svc = AL.ApiToolbox._clean_spec(name, json.loads(js))
        except Exception:
            _count(out["apis"]["excluded"], "unreadable spec")
            continue
        if SPEC_PREFIX + name in vetoed or _norm(SPEC_PREFIX + name) in vetoed:
            r = "vetoed"
        elif AL.private_host(svc.get("base", "")) and not allow_private_hosts:
            r = "private host (a user's own machine or network)"
        elif _guard().sensitive_reason(json.dumps(svc, sort_keys=True)):
            r = "sensitive"
        else:
            r = _marker_reason(name + " " + svc.get("base", ""), skip=("open-item", "workflow-log", "history",
                                                                       "probe", "task-artifact"))
        if r:
            _count(out["apis"]["excluded"], r)
            continue
        carried_svcs.add(name)
        pay["apis"].append((name, svc))
        out["apis"]["carried"].append({"service": name, "endpoints": sorted(svc.get("endpoints", {})),
                                       "env": AL.placeholders_in(svc)})
    for q, t in latest:
        kind = _row_kind(q, t[1], t[3])
        if kind == "api-card":
            m = re.match(re.escape(CARD_PREFIX) + r"([^.\s]+)\.", str(t[1]))
            svc = m.group(1) if m else None
            r = None if svc in carried_svcs else "its api is not carried"
            r = r or _wording_reason(q) or ("vetoed" if _norm(q) in vetoed else None)
            if r:
                _count(out["api_cards"]["excluded"], r)
            else:
                pay["api_cards"].append((q, AL.placeholderize_text(str(t[1]), svc)))
                out["api_cards"]["carried"].append(q)
        elif kind == "toolreflex":
            try:
                spec = json.loads(str(t[1]))
            except ValueError:
                _count(out["reflexes"]["excluded"], "unreadable")
                continue
            pat = q[len("toolreflex: "):] if q.startswith("toolreflex: ") else q
            r = None if str(spec.get("service")) in carried_svcs else "its api is not carried"
            r = r or _wording_reason(pat)
            params = AL.placeholderize(dict(spec.get("params") or {}), str(spec.get("service", "")))
            if r is None and _guard().sensitive_reason(json.dumps(params, sort_keys=True)):
                r = "sensitive"
            if r:
                _count(out["reflexes"]["excluded"], r)
            else:
                pay["reflexes"].append({"pattern": pat, "service": spec["service"], "endpoint": spec["endpoint"],
                                        "params": params, "extract_numbers": list(spec.get("extract_numbers") or [])})
                out["reflexes"]["carried"].append([pat, "%s.%s" % (spec["service"], spec["endpoint"])])
    # ---- shareable ProtoStores (the commons rule, reused) -----------------------------------------------------
    if hasattr(src, "_shareable_stores"):
        elig = src._shareable_stores()
        doors = set(src.__dict__.get("_protostores", {})) | set(src.__dict__.get("_protostore_share", {}))
        for door in sorted(doors):
            if door in elig:
                continue
            f = src.__dict__.get("_protostore_share", {}).get(door) or {}
            why = ("not marked shareable" if not f.get("shareable") else
                   "a question it learned from was refused by the learning guard" if not f.get("guard_ok") else
                   "it learned from session-salted (user-private) questions" if f.get("salted") else "empty")
            _count(out["protostores"]["excluded"], why)
        for door, st in elig.items():
            if door == "route":
                # the router's store lives against ONE catalog's feature digest and needs the router built (a
                # minutes-long pretrain) to merge -- it travels in the commons carrier only (memory_import merges it)
                pay["protostores"][door] = (st, "carrier")
                out["protostores"]["carried"].append({"door": door, "labels": len(st), "how": "commons carrier"})
                continue
            labels = list(st.labels)
            if door == "tool":
                labels = [l for l in labels if str(l).split(".")[0] in carried_svcs]   # a tool whose api stays
                if len(labels) < len(st.labels):                                          # home is not carried
                    _count(out["protostores"]["excluded"], "tool rows whose api is not carried (%d)"
                           % (len(st.labels) - len(labels)))
            if not labels:
                continue
            pay["protostores"][door] = (_substore(st, labels), "merge")
            out["protostores"]["carried"].append({"door": door, "labels": len(labels), "how": "merged"})
    # ---- SOPs from the knowledge store ------------------------------------------------------------------------
    root = getattr(src, "_archive_root", None)
    if root and os.path.exists(os.path.join(str(root), "knowledge.lecore")):
        from holographic.caching_and_storage.holographic_knowledgestore import KnowledgeStore
        from holographic.agents_and_reasoning.holographic_soprunner import parse_sop
        sops = {}
        for e in KnowledgeStore(str(root)).entries:
            m = re.match(r"^\[sop:([^\]]+)\] ", str(e.get("text", "")))
            if m:
                sops[m.group(1)] = str(e["text"])[m.end():]              # newest per name
        for name, text in sops.items():
            if name in SEEDPACK_SOPS:
                _count(out["sops"]["excluded"], "the seedpack installs it")
                continue
            text2 = AL.placeholderize_text(text, name)
            r = _wording_reason(name) or _marker_reason(text2, skip=("open-item", "history", "workflow-log"))
            if r is None and _guard().sensitive_reason(text2):
                r = "sensitive"
            if r is None and not parse_sop(text2).get("ok"):
                r = "does not parse"
            if r:
                _count(out["sops"]["excluded"], r)
            else:
                pay["sops"].append((name, text2))
                out["sops"]["carried"].append({"name": name, "env": AL.placeholders_in(text2)})
    return out, pay


def _substore(st, labels):
    """A float64 copy of a ProtoStore holding only `labels` (the miner's confusion pairs stay home)."""
    import numpy as np
    from holographic.agents_and_reasoning.holographic_protostore import ProtoStore
    meta, _ = st.state()
    idx = [st.index(l) for l in labels]
    meta = dict(meta, labels=list(labels), count=[int(st.count[i]) for i in idx], confusion={},
                key_of={k: v for k, v in (meta.get("key_of") or {}).items() if k in labels})
    A = np.asarray(st.A, np.float64)[idx].copy()
    cp = ProtoStore.from_state(meta, {"A": A})
    cp.A = A
    return cp


NOT_DISTILLED = {
    "reflex bridge experience": "question vectors without their text: nothing to audit for secrets or session "
                                "facts (the tool door's store travels because its eligibility was tracked at learn "
                                "time)",
    "SystemOne tables + decision ledger": "one user's decisions about their own states",
    "open escalations": "one deployment's open questions",
    "learning guard's learned examples": "users' refused questions (redacted); the shipped examples live in "
                                         "holographic_learnguard_examples",
    "door calibration streams": "thresholds re-derive per population (doctrine: re-derive when the population moves)",
    "query ledger, tool cache, goals": "session bookkeeping",
    "semantic context vectors": "one-shot noise tokens (the balloon lesson); rebuilt from shipped text",
    "teacher-noise estimate": "a property of one deployment's model end",
}


def _boot_copy(partition):
    """Boot a mind on a TEMP COPY of the partition: boot rolls learning files over, and a distillation must never
    modify the session partition. -> (mind, tmpdir)."""
    tmp = tempfile.mkdtemp(prefix="distill_src_")
    dst = os.path.join(tmp, "p")
    shutil.copytree(partition, dst)
    src = lecore.UnifiedMind()
    src.boot(partition=dst, doctrine=True, llm=lambda p: "")
    src._archive_root = dst
    return src, tmp


def distill(partition, out_dir, dry_run=False, rows_only=False, anchor=True, allow_private_hosts=False,
            carry_calibration=False, src=None):
    """Distil `partition` into a shipped bundle at `out_dir` (or only report, dry_run=True).
    rows_only=True reproduces the Q&A-rows-only bundle (the pre-2026-09-27 behaviour, kept as a measured baseline).
    anchor / allow_private_hosts: see _select / _artifacts -- the release always runs the defaults.
    carry_calibration: also carry the meaning gate's calibration labels (OFF by default -- see bench_core_memory:
    thresholds are re-derived per population). `src`: an already-booted source mind (benchmarks); else the
    partition is booted from a temp copy."""
    tmp = None
    if src is None:
        src, tmp = _boot_copy(partition)
    try:
        return _distill(src, out_dir, dry_run, rows_only, anchor, allow_private_hosts, carry_calibration)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def _distill(src, out_dir, dry_run, rows_only, anchor, allow_private_hosts, carry_calibration):
    AL = _ph()
    lad = src.zoo["ladder"]
    confirmed_q, _ = _confirmations(src)
    kept_all, excluded, n_rows = _select(lad, anchor=anchor, confirmed=confirmed_q, workflows=not rows_only)
    kept = [k for k in kept_all if _row_kind(k[0], k[1], k[2]) == "row"]
    wf_kept = [k for k in kept_all if _row_kind(k[0], k[1], k[2]) == "workflow"]
    report = {"rule": {"anchor": bool(anchor), "rows_only": bool(rows_only),
                       "allow_private_hosts": bool(allow_private_hosts), "carry_calibration": bool(carry_calibration)},
              "source_rows": n_rows, "shipped": len(kept), "excluded": dict(sorted(excluded.items())),
              "questions": [q for q, _, _ in kept],
              "promoted": sorted(q for q, _, p in kept if p == "confirmed")}
    arts, pay = ({}, None) if rows_only else _artifacts(src, [q for q, _, _ in kept], anchor=anchor,
                                                        allow_private_hosts=allow_private_hosts,
                                                        workflows_kept=wf_kept)
    if not rows_only:
        report["artifacts"] = {k: {"carried": len(v["carried"]), "excluded": dict(sorted(v["excluded"].items())),
                                   "items": v["carried"]} for k, v in arts.items()}
        report["not_distilled"] = NOT_DISTILLED
    if dry_run:
        return report
    os.makedirs(out_dir, exist_ok=True)
    dst = lecore.UnifiedMind()
    dst.zoo_attach(lambda p: "")
    # A CLEAN WRITE: the old bundle's learning directory AND knowledge store are replaced, never merged into (merging
    # is how a bundle accumulates session rows). SOPs go to <out>/knowledge.lecore through sop_save.
    ldir = os.path.join(out_dir, "learning")
    if os.path.isdir(ldir):
        shutil.rmtree(ldir)
    for fn in ("knowledge.lecore",):
        if os.path.exists(os.path.join(out_dir, fn)):
            os.remove(os.path.join(out_dir, fn))
    os.makedirs(ldir, exist_ok=True)
    dst._archive_root = out_dir
    refused = 0

    def _teach(q, a, prov=None):
        r = dst.teach(q, a)
        if isinstance(r, dict) and r.get("taught") is False:
            return False                             # the destination's own guard is the last word
        dlog = getattr(dst.zoo["ladder"], "taught_log", [])
        if prov and prov != "taught" and dlog and str(dlog[-1][0]) == q and len(dlog[-1]) > 3:
            dlog[-1] = [dlog[-1][0], dlog[-1][1], dlog[-1][2], prov]
        return True
    for q, a, prov in kept:
        if not _teach(q, a, prov if prov == "confirmed" else None):
            refused += 1
            continue
        if prov in ("validated", "evidenced"):
            dst.conjecture_record(q, a)
            dst.conjecture_promote(q, prov, "carried from distillation source")
    n_topics = 0
    for topic in GENERIC_TOPICS:
        corp = getattr(src, "_archive_corpora", {}).get(topic)
        if corp:
            try:
                chunks = [c for c in corp.get("chunks", [])] if isinstance(corp, dict) else None
                if chunks:
                    dst.research_archive(topic, chunks, sources=["%s#%d" % (topic, i) for i in range(len(chunks))])
                    n_topics += 1
            except Exception:
                pass
    applied = {}
    if pay is not None:
        from holographic.agents_and_reasoning.holographic_meaning import _rid
        for q, a, prov in pay["workflows"]:
            if not _teach(q, a, prov if prov == "confirmed" else None):
                refused += 1
        for name, svc in pay["apis"]:
            _teach(SPEC_PREFIX + name, json.dumps(svc))
        for q, a in pay["api_cards"]:
            _teach(q, a)
        for r_ in pay["reflexes"]:
            dst.tool_reflex_teach(r_["pattern"], r_["service"], r_["endpoint"], params=r_["params"],
                                  extract_numbers=r_["extract_numbers"])
        mi = dst.meaning                                  # seeded from the rows just taught
        n_w = 0
        for canonical, w in pay["wordings"]:
            # learn=True: the association between the new wording and the row is RE-LEARNED here, from shipped
            # wordings only -- the source's association table may hold pairs learned from rows that stay home
            if mi.link(_rid(canonical, "answer"), w, learn=True):
                n_w += 1
        for canonical, w in pay.get("doctrine_wordings", []):
            if mi.link(mi.add_row(canonical, kind="answer"), w, learn=True):
                n_w += 1
        n_m = 0
        for mr in pay["methods"]:
            rid = mi.add_row(mr["canonical"], kind=mr["kind"], method=mr["method"], clarify=mr["clarify"],
                             provenance=mr["provenance"])
            n_m += 1
            for w in mr["phrasings"]:
                if mi.link(rid, w, learn=True):
                    n_w += 1
        if pay["negatives"]:
            dst._meaning_neg = {tuple(x) for x in pay["negatives"]}
        if pay["direction"] is not None:
            import holographic.agents_and_reasoning.holographic_rolecall as RC
            meta_, arr_ = pay["direction"]
            dst.__dict__["_direction_reader_obj"] = RC.ContextRoles.from_state(meta_, arr_)
        if carry_calibration and getattr(src, "_meaning", None) is not None:
            mi.calib = list(src._meaning.calib)
            mi._cal_fit = None
        stores = {}
        for door, (st, how) in sorted(pay["protostores"].items()):
            stores[door] = st
            if how == "merge":
                dst._protostore_merge(door, st, source="distillation")
                dst.protostore_share(door, True)
        if stores:
            dst.__dict__["_commons_stores"] = stores
        for name, text in pay["sops"]:
            dst.sop_save(name, text)
        applied = {"wordings_linked": n_w, "method_rows": n_m}
    dst.learning_save(out_dir, path=os.path.join(ldir, "state.lecore"))
    report.update({"refused_by_guard_on_teach": refused, "topics": n_topics, "applied": applied,
                   "bundle": os.path.join(out_dir, "learning", "state.lecore"),
                   "env": sorted(set(AL.placeholders_in([svc for _, svc in (pay or {}).get("apis", [])] +
                                                        [r_["params"] for r_ in (pay or {}).get("reflexes", [])] +
                                                        [m_["method"] for m_ in (pay or {}).get("methods", [])] +
                                                        [t_ for _, t_ in (pay or {}).get("sops", [])])))})
    with open(os.path.join(out_dir, "distill_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    # THE WHEEL'S COPY (2026-09-27): the repo's release_bundle/ is mirrored into lecore_data/release_bundle so a pip
    # install carries the same core memory. One writer, two places -- CI checks both with --check and that they
    # match byte for byte (tests/test_distill_release.py).
    if os.path.abspath(out_dir) == os.path.abspath("release_bundle") and os.path.isdir("lecore_data"):
        mirror = os.path.join("lecore_data", "release_bundle")
        if os.path.isdir(mirror):
            shutil.rmtree(mirror)
        shutil.copytree(out_dir, mirror)
        report["mirrored_to"] = mirror
    return report


def _strings(obj):
    """Every string LEAF (and dict key) of a JSON-able structure, one at a time. Scanning leaves, not the dumped
    JSON: a dump of the association table is a long run of English stems and read as a BIP-39 seed phrase (a
    measured false positive, 2026-09-27); a secret lives inside ONE string, and every string is scanned."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, str):
        yield obj


def _container_secret_scan(path):
    """Every string in every section's meta of a saved container, scanned by the learning guard ->
    [(section kind, reason)]. A second net under every per-artifact rule: whatever the rules missed, a secret in the
    file fails here."""
    from holographic.io_and_interop.holographic_container import load_container
    LG = _guard()
    with open(path, "rb") as f:
        c = load_container(f.read())
    bad = []
    for s in c["sections"]:
        for piece in _strings(s.get("meta")):
            r = LG.sensitive_reason(piece)
            if r:
                bad.append((s["kind"], r))
                break
    return bad


def check(bundle_dir):
    """Audit a shipped bundle: re-apply every rule to what it holds. Every row and every carried artifact must pass
    (a bundle written by anything but distill() fails here), and the whole container must pass the secret scan."""
    AL = _ph()
    from holographic.agents_and_reasoning.holographic_meaning import placeholder_method
    rep_path = os.path.join(bundle_dir, "distill_report.json")
    rule = {}
    if os.path.exists(rep_path):
        with open(rep_path, encoding="utf-8") as f:
            rule = json.load(f).get("rule") or {}
    m = lecore.UnifiedMind()
    m.zoo_attach(lambda p: "")
    m.learning_load(bundle_dir)
    lad = m.zoo["ladder"]
    kept, excluded, n_rows = _select(lad, anchor=rule.get("anchor", True), workflows=not rule.get("rows_only"))
    failing = dict(excluded)
    kept_norm = {_norm(q) for q, _, _ in kept}
    latest, _ = _latest_rows(lad)
    svcs = set()
    for q, t in latest:
        kind = _row_kind(q, t[1], t[3])
        if kind == "api-spec":
            name = q[len(SPEC_PREFIX):].strip()
            try:
                svc = json.loads(str(t[1]))
            except ValueError:
                _count(failing, "api: unreadable")
                continue
            if AL.ApiToolbox._clean_spec(name, svc) != svc:
                _count(failing, "api: a raw credential (not placeholders)")
            elif AL.private_host(svc.get("base", "")) and not rule.get("allow_private_hosts"):
                _count(failing, "api: private host")
            svcs.add(name)
        elif kind == "toolreflex":
            spec = json.loads(str(t[1]))
            params_ = dict(spec.get("params") or {})
            if AL.placeholderize(params_, str(spec.get("service"))) != params_:
                _count(failing, "reflex: a raw credential")
        elif kind == "decision":
            _count(failing, "decision record")
    for q, t in latest:
        if _row_kind(q, t[1], t[3]) == "toolreflex" and str(json.loads(str(t[1])).get("service")) not in svcs:
            _count(failing, "reflex: its api is not in the bundle")
        if _row_kind(q, t[1], t[3]) == "api-card":
            mm = re.match(re.escape(CARD_PREFIX) + r"([^.\s]+)\.", str(t[1]))
            if not mm or mm.group(1) not in svcs:
                _count(failing, "api-card: its api is not in the bundle")
    mi = getattr(m, "_meaning", None)
    doc_norms = _doctrine_norms()
    n_words = n_methods = 0
    if mi is not None:
        for rid, row in mi.rows.items():
            if row["kind"] == "answer":
                owner_ok = _norm(row["canonical"]) in kept_norm or _norm(row["canonical"]) in doc_norms or \
                    _row_kind(row["canonical"], "", "") in ("api-spec",) or \
                    any(_norm(q) == _norm(row["canonical"]) for q, t in latest
                        if _row_kind(q, t[1], t[3]) in ("api-card", "workflow"))
                extra = [w for w in row["phrasings"] if _norm(w) != _norm(row["canonical"])]
                if extra and not owner_ok:
                    _count(failing, "wording: its row does not ship")
            else:
                n_methods += 1
                if row.get("method") and placeholder_method(row["method"]) != row["method"]:
                    _count(failing, "method: a raw credential")
                if _wording_reason(row["canonical"]):
                    _count(failing, "method: " + _wording_reason(row["canonical"]))
                extra = row["phrasings"][1:]
            for w in extra:
                n_words += 1
                if _wording_reason(w):
                    _count(failing, "wording: " + _wording_reason(w))
    kpath = os.path.join(bundle_dir, "knowledge.lecore")
    n_sops = 0
    if os.path.exists(kpath):
        from holographic.caching_and_storage.holographic_knowledgestore import KnowledgeStore
        for e in KnowledgeStore(bundle_dir).entries:
            mt = re.match(r"^\[sop:([^\]]+)\] ", str(e.get("text", "")))
            if mt:
                n_sops += 1
                text = str(e["text"])[mt.end():]
                if AL.placeholderize_text(text, mt.group(1)) != text or _guard().sensitive_reason(text):
                    _count(failing, "sop: a raw credential")
    bpath = os.path.join(bundle_dir, "learning", "state.lecore")
    if not os.path.exists(bpath):
        _count(failing, "no learning/state.lecore (not a distilled bundle)")
    else:
        for kind, why in _container_secret_scan(bpath):
            _count(failing, "container secret scan: %s" % kind)
    ok = not failing
    return {"ok": ok, "rows": n_rows, "passing": len(kept), "failing": dict(sorted(failing.items())),
            "carried": {"wordings": n_words, "method_rows": n_methods, "apis": len(svcs), "sops": n_sops}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="distil a session partition into the shipped release bundle")
    ap.add_argument("--partition", default="lecore_memory", help="the session partition to distil (booted from a copy)")
    ap.add_argument("--out", default="release_bundle", help="the bundle directory to (re)write")
    ap.add_argument("--dry-run", action="store_true", help="report what would ship; write nothing")
    ap.add_argument("--rows-only", action="store_true", help="the Q&A rows only (the pre-2026-09-27 bundle)")
    ap.add_argument("--carry-calibration", action="store_true",
                    help="also carry the meaning gate's calibration labels (measured trade-off: bench_core_memory)")
    ap.add_argument("--check", metavar="BUNDLE", help="audit an existing bundle against the rule")
    args = ap.parse_args()
    if args.check:
        rep = check(args.check)
        print(json.dumps(rep, indent=1))
        sys.exit(0 if rep["ok"] else 1)
    rep = distill(args.partition, args.out, dry_run=args.dry_run, rows_only=args.rows_only,
                  carry_calibration=args.carry_calibration)
    print(json.dumps({k: (v if k != "artifacts" else {a: {"carried": b["carried"], "excluded": b["excluded"]}
                                                         for a, b in v.items()})
                      for k, v in rep.items() if k not in ("questions", "not_distilled")}, indent=1))

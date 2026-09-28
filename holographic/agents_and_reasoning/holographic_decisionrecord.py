"""
holographic_decisionrecord.py -- ONE record for every decision door, and outcomes that enter BY ID.

WHY (doc 05, sweep 176 backlog G1/G2): a route, a typed decision, a decision-tree node, a swarm step
and a model-end resolution are one act in different costumes, and each returned its own shape. The
loop sweep 172 built closed only because a human noticed a tie in a printout and taught the miss in
prose. Here every decision is a DecisionRecord with a deterministic id; reporting an outcome against
that id is the outcome path, and a record that came from a fitted SystemOne forwards the outcome to
observe() itself -- the loop closes by construction, not by discipline.

HOLOGRAPHIC BY CONSTRUCTION: a record is also an HRR record -- role-bound fillers superposed --
so it can be stored in a bundle, queried by unbinding, and compared to another record by cosine.
encode() / decode() round-trip is pinned at exact field recovery against the candidate set.

KEPT NEGATIVES (measured this sweep, on the record so they are not retried):
  * Family-prototype bundles as a ROUTER lose: family top-1 0.464, and routing within the predicted
    family collapses to 0.318 vs the flat router's 0.562 -- a wrong family loses the answer entirely.
    Families serve the tiered router's 'clarify' tier; they do not filter.
  * The module docstring's opening sentence folded into the card haystack changes nothing: top-1
    0.562 -> 0.562, top-3 0.709 -> 0.716, gate rejection unchanged. It carries no words `does` lacks.
"""
import hashlib
import json
import warnings

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, bundle, cosine, unbind
from holographic.agents_and_reasoning.holographic_systemone import hashed_ngram_encode
from holographic.mesh_and_geometry.holographic_planshape import derived_atom

FIELDS = ("state", "question", "options", "answer", "via", "margin", "p", "cost")
VIAS = ("route", "typed", "tree", "swarm", "model_end", "human", "reflex",   # reflex: answered from experience (sweep 176)
        "tool",    # tool: a live tool call made by serve()'s tool reflex (sweep 178); learns on the ngram key
        "meaning",  # meaning: the ladder's find-by-meaning rung picked a row (sweep 181); learns in the MeaningIndex
        "rank")    # rank: mind.rank(state, candidates) -- free-form System One ranking (CLM backlog E3.3); its outcome
                   # feeds door_calibrator("rank") and the rank door's ProtoStore


# ---- ONE MEANING OF p (backlog E0.6, CLM panel defect G2) ------------------------------------------------------
# Before this, the bare key "p" meant OPPOSITE things at different doors: route_tiered's catalog path returned a NULL
# p-value (low = the score beat the noise = confident) while its reflex path returned 1 - error (high = confident).
# A caller thresholding "p > 0.9" was right at one door and exactly wrong at the other. Every door's record now
# carries TWO named numbers:
#     p_correct  calibrated P(this answer is correct): HIGH = confident. None until that door's own calibrator has
#                enough labelled outcomes (an uncalibrated door says so instead of inventing a number).
#     p_null     a significance p-value against a null, where the door has one: LOW = significant. Else None.
# The bare "p" stays ONE release with its OLD per-door value (so nothing that reads it changes behaviour), but reading
# it through [] or .get() emits a DeprecationWarning -- once per process per door, so a hot loop is not flooded.
# WHY a dict subclass and not a new return type: json.dumps / Flask / the MCP bridge serialise a dict subclass exactly
# like a dict (the C encoder iterates items without calling __getitem__), `"p" in r` and dict(r) do not warn, and
# every existing caller keeps working. Only a READ of the old key is flagged, which is exactly the migration signal.
class DecisionDict(dict):
    """A door's returned record: a plain dict whose bare 'p' key is DEPRECATED (read it and a DeprecationWarning fires
    once per process per door). Use 'p_correct' (calibrated P(correct), high = confident) or 'p_null' (a null
    p-value, low = significant). Serialises exactly like a dict."""

    _warned = set()                        # door names already warned in this process (class-level, by design)

    def __init__(self, *args, _door="door", **kw):
        dict.__init__(self, *args, **kw)
        self._door = str(_door)

    def _warn_p(self):
        if self._door not in DecisionDict._warned:
            DecisionDict._warned.add(self._door)
            warnings.warn("the bare 'p' key of a %r decision is deprecated (it meant a null p-value at some doors and "
                          "P(correct) at others): read 'p_correct' (high = confident) or 'p_null' (low = significant)"
                          % self._door, DeprecationWarning, stacklevel=3)

    def __getitem__(self, key):
        if key == "p":
            self._warn_p()
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        if key == "p":
            self._warn_p()
        return dict.get(self, key, default)

    def copy(self):
        """A copy that keeps the door (dict.copy would silently drop the deprecation)."""
        return DecisionDict(self, _door=self._door)


def door_record(d, door, p_correct=None, p_null=None):
    """Wrap a door's result dict as a DecisionDict and set its p_correct / p_null (setdefault: a value the door
    already put there wins). The bare 'p' the door returned before, if any, is kept verbatim (deprecated)."""
    out = d if isinstance(d, DecisionDict) else DecisionDict(d, _door=door)
    out._door = str(door)                   # an inner door's record re-issued by an outer door warns as the outer one
    out.setdefault("p_correct", p_correct)
    out.setdefault("p_null", p_null)
    return out


def legacy_p(d):
    """Read a record's deprecated bare 'p' WITHOUT the warning -- for in-repo code that must pass the old value
    through for one more release (e.g. route() forwarding route_tiered's p). New code reads p_correct / p_null."""
    return dict.get(d, "p") if isinstance(d, dict) else None


def outcome_matches(answer, outcome):
    """Did the reported outcome CONFIRM the answer? Plain equality, except a yes/no (noul) decision, whose record
    answer is a bool while its reported outcome is the option name: True matches "yes"/"true", False "no"/"false"
    (the same mapping as SystemOne's noul fix -- before it, every noul report counted as a correction)."""
    if isinstance(answer, bool) and isinstance(outcome, str):
        return outcome.strip().lower() in (("yes", "true") if answer else ("no", "false"))
    return answer == outcome


def record_id(state, question, options, answer, via):
    """sha256 over the canonical decision fields -- the OUTCOME is not part of the id, so the same
    decision reported twice has one id and the second report is an update, not a new row."""
    canon = json.dumps({"state": (state or "").strip().lower(), "question": question, "options": list(options or []),
                        "answer": answer, "via": via}, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


class DecisionRecord:
    """{state, question, options, answer, via, margin, p_correct, p_null, p (deprecated), cost, outcome, id}. `via`
    names the door (route / typed / tree / swarm / model_end / human / reflex / tool / meaning). outcome is None until
    reported. p_correct / p_null (E0.6) are optional and NOT part of the id -- ids never change."""

    def __init__(self, state, question, options, answer, via, margin=None, p=None, cost=None, outcome=None, meta=None,
                 p_correct=None, p_null=None):
        if via not in VIAS:
            raise ValueError("via must be one of %s, got %r" % (VIAS, via))
        self.state, self.question, self.options, self.answer, self.via = state, question, list(options or []), answer, via
        self.margin, self.p, self.cost, self.outcome = margin, p, cost, outcome
        self.p_correct, self.p_null = p_correct, p_null
        self.meta = dict(meta or {})
        self.id = record_id(state, question, self.options, answer, via)

    def to_dict(self, max_chars=400):
        """JSON view. NOOA's pass-by-reference rule (bounded previews, sweep 130) applies to the STATE: a
        record never ships a huge input -- above max_chars it carries the true length, a head/tail sample and
        the sha256 of the full text, via holographic_boundedpreview. The HRR encoding still used the full
        text, so similarity is unaffected; only what travels over the wire is bounded."""
        state = self.state
        if isinstance(state, str):
            # sweep 179: what TRAVELS is redacted (bus, MCP, decision_ledger_to_memory). The HRR encoding and
            # the ledger's own similarity still use the full text in-process, which never leaves it.
            from holographic.agents_and_reasoning.holographic_learnguard import redact
            state = redact(state)
        if isinstance(state, str) and len(state) > int(max_chars):
            from holographic.io_and_interop.holographic_boundedpreview import bounded_preview
            state = {"ref": hashlib.sha256(state.encode("utf-8")).hexdigest()[:16], "len": len(state),
                     "preview": bounded_preview(state, max_chars=max_chars, cost=False)}
        return DecisionDict({"id": self.id, "state": state, "question": self.question, "options": self.options,
                "answer": self.answer, "via": self.via, "margin": self.margin, "p": self.p, "cost": self.cost,
                "p_correct": self.p_correct, "p_null": self.p_null,
                "outcome": self.outcome, "meta": self.meta,
                # NOOA validated termination (sweep 176): a step carries its evidence and the command that
                # verifies it; the harness runs `verify` BEFORE the step is accepted (see swarm_step).
                "evidence": self.meta.get("evidence"), "verify": self.meta.get("verify")}, _door="record:" + self.via)

    # ---- PERSISTENCE AS TEXT (backlog E0.5) ------------------------------------------------------------------
    # A record is stored as its TEXT fields, never as its HRR vector: the id and the encoding are re-derived on load,
    # so a change of the id function or the codec migrates old partitions BY REPLAY (the taught-row rule, cp21).
    # The stored id travels too, and the loader checks it -- a mismatch is reported and the old id is kept as an
    # ALIAS, so an outcome a caller reports against the id it was given still lands.
    def to_text(self):
        """The record as JSON-able text fields (+ the id it was issued under). Every field goes through one JSON
        round trip: a numpy scalar becomes its Python value, anything else exotic its string -- never a pickle."""
        def plain(o):
            return o.item() if hasattr(o, "item") and callable(o.item) else str(o)
        return json.loads(json.dumps(
            {"id": self.id, "state": self.state, "question": self.question, "options": list(self.options),
             "answer": self.answer, "via": self.via, "margin": self.margin, "p": self.p,
             "p_correct": self.p_correct, "p_null": self.p_null, "cost": self.cost, "outcome": self.outcome,
             "meta": self.meta}, sort_keys=True, default=plain))

    @classmethod
    def from_text(cls, t):
        """Inverse of to_text: rebuild the record (its id re-derived from the fields, NOT copied)."""
        return cls(t.get("state"), t.get("question"), t.get("options"), t.get("answer"), t.get("via"),
                   margin=t.get("margin"), p=t.get("p"), cost=t.get("cost"), outcome=t.get("outcome"),
                   meta=t.get("meta"), p_correct=t.get("p_correct"), p_null=t.get("p_null"))


    def __repr__(self):
        return "DecisionRecord(%s %s -> %r via %s, outcome=%r)" % (self.id, self.question, self.answer, self.via, self.outcome)


class RecordCodec:
    """HRR encoding of a record: sum over fields of bind(role_atom, filler_vec). Text fillers use the
    same deterministic hashed n-gram encoder the typed decisions use; enum-like fillers (answer, via,
    outcome) use derived atoms by name so decode can clean up against a candidate list exactly."""

    def __init__(self, dim=2048, seed=0):
        self.dim, self.seed = int(dim), int(seed)
        self._text = hashed_ngram_encode(dim=self.dim)
        self._roles = {f: derived_atom(seed, "role:" + f, self.dim) for f in FIELDS + ("outcome",)}

    def atom(self, name):
        return derived_atom(self.seed, "val:" + str(name), self.dim)

    def _unit(self, v):
        v = np.asarray(v, float)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def encode(self, rec):
        parts = [bind(self._roles["state"], self._unit(self._text(rec.state or ""))),
                 bind(self._roles["question"], self.atom(rec.question)),
                 bind(self._roles["answer"], self.atom(rec.answer)),
                 bind(self._roles["via"], self.atom(rec.via))]
        if rec.outcome is not None:
            parts.append(bind(self._roles["outcome"], self.atom(rec.outcome)))
        return self._unit(bundle(parts))

    def decode(self, vec, field, candidates):
        """Unbind the role, clean up against `candidates` -> (best, cosine). Exact recovery is pinned
        in the selftest; the cosine is the evidence, not a sticker."""
        probe = unbind(vec, self._roles[field])
        scored = sorted(((float(cosine(probe, self.atom(c))), c) for c in candidates), key=lambda t: (-t[0], str(t[1])))
        return scored[0][1], scored[0][0]


def options_text(options):
    """A record's options as ONE text for the learning guard's secret check. Joined with a non-word separator
    ('zzqsep', not a BIP-39 word), so a list of ordinary labels ('open', 'file', 'save', ...) can never read as one run
    of 12 seed words across the joins -- one check per record, not one per option (save time stays flat)."""
    return "\nzzqsep\n".join(str(o) for o in (options or []))


def hook_specs(meta):
    """The hook SPECS a record's meta names, as a list (CLM backlog, wave 2). meta["hook"] is one spec (a dict -- the
    form every record written before wave 2 carries) or, once two doors filed a decision under one id, a list of
    specs. None / anything else -> []."""
    h = (meta or {}).get("hook")
    if isinstance(h, dict):
        return [h]
    if isinstance(h, list):
        return [x for x in h if isinstance(x, dict)]
    return []


def _spec_key(spec):
    """Canonical text of a hook spec: two specs are the same hook iff their canonical JSON matches."""
    return json.dumps(spec, sort_keys=True, default=str)


def _hook_site(hook, key=None):
    """WHERE a live hook comes from: an explicit key, else the code that made it (module : qualname : first line).
    Two closures made by the same line of code for the same record id are the same hook re-issued (a decision made
    again), so the later one REPLACES the earlier; closures from different code sites COMBINE. Deterministic: no
    id() and no hash() -- a site is a string the source defines."""
    if key is not None:
        return str(key)
    fn = getattr(hook, "func", hook)                   # functools.partial: the site of the wrapped function
    fn = getattr(fn, "__func__", fn)                   # a bound method: its function
    code = getattr(fn, "__code__", None)
    mod = getattr(fn, "__module__", None) or type(fn).__module__
    qn = getattr(fn, "__qualname__", None) or type(fn).__qualname__
    return "%s:%s:%d" % (mod, qn, code.co_firstlineno if code is not None else 0)


class Ledger:
    """Append-only store of records by id, with the outcome path. `hooks` map a record id to
    callables(outcome) -> anything; a typed decision registers its fitted SystemOne here so that
    report(id, outcome) forwards to observe() -- the loop closes without a human in it.

    resolve_hook (backlog E0.5): callable(rec) -> callable(outcome) or None. A LIVE closure dies with the process, so
    a record reloaded from the partition could never train its door again (decision_outcome(<old id>) raised KeyError,
    and even with the record back, nothing would have forwarded the outcome). The mind instead writes WHAT to call
    into rec.meta["hook"] (a SystemOne cache key + question name, or the meaning door) and passes a resolver that
    RE-DERIVES the callable from the record at report time -- a reloaded record trains the same door.

    HOOKS COMBINE PER ID (CLM backlog wave 2, proposed by wave-1 worker B-encoders). A record id is sha256 over
    (state, question, options, answer, via), so two DOORS can file one decision under one id: an escalated typed answer
    (systemone_decide, via model_end) and the CLM plugin's own record of the same answer hashed to the same id, and
    add() kept ONE hook per id -- MEASURED: the typed record's arrival replaced the CLM calibration hook (the plugin
    now files under "clm:<question>" as a workaround). Now:
      * live closures for one id are kept as a list, keyed by their code site (_hook_site): the same site re-adding
        (the same decision made again) REPLACES its own closure -- it never runs twice -- a different site COMBINES;
      * the record's hook SPECS are merged on re-add (old first, duplicates by canonical JSON dropped), so meta["hook"]
        may now be a LIST of specs; a record with one spec keeps the old dict form byte for byte (old partitions and
        old builds read it unchanged);
      * report() runs every closure (in the order attached), then every spec the resolver rebuilds (the resolver is
        handed a shallow copy of the record whose meta["hook"] is that ONE spec -- the resolver contract is
        unchanged), catching each: one door's error never stops another door from learning.
    A closure passed in the SAME add() as a record carrying specs is taken to implement those specs (they are not
    resolved again while it is alive) -- the old 'an explicit closure wins' rule, now scoped to its own add().

    KEYED CLOSURES PERSIST AS SPECS (spec_keys, the learning-loop audit, 2026-09-26). A live closure added with a
    hook_key the resolver knows how to rebuild (the mind passes spec_keys=("router", "tooldoor")) ALSO writes the
    spec {"kind": <hook_key>} into the record's meta, marked as implemented by that closure while it lives. MEASURED
    before (tools/audit_learning_loop.py): route_tiered's router hook and serve()'s tool-door hook were live closures
    keyed "router" / "tooldoor", so a route or tool outcome reported AFTER a restart trained NOTHING in the router's
    or the tool door's ProtoStore (router verdicts 0 of 1, tool door verdicts 0 of 1) while decision_outcome returned
    normally. With the spec in the record, the restarted mind's resolver rebuilds the door from the record itself
    (state, options, answer) -- and in the live process nothing runs twice (the closure covers its own spec)."""

    def __init__(self, codec=None, resolve_hook=None, spec_keys=()):
        self.codec = codec or RecordCodec()
        self._rows = {}
        self._order = []
        self._hooks = {}               # id -> [(site, callable)] in the order attached (wave 2: several per id)
        self._covered = {}             # id -> {canonical spec} a live closure added with it already implements
        self._vecs = {}
        self._alias = {}               # id a record was issued under -> its re-derived id (after a migrating load)
        self.resolve_hook = resolve_hook
        # hook_keys whose closure is ALSO persisted as the spec {"kind": key} (see the class docstring). Only
        # meaningful with a resolve_hook that rebuilds those kinds; empty by default (the old behaviour exactly).
        self.spec_keys = frozenset(str(k) for k in (spec_keys or ()))
        # A LIVE CAP (the learning-loop audit, 2026-09-27): the in-process ledger had no bound, and lookups
        # (find_capability, suggest, serve) add a record each as a side effect -- a long-running service grew it
        # forever (a 150-way rank record is 7,654 bytes, a typed one 705). The partition already keeps only the newest
        # 4,096; the live ledger keeps the newest MAX_LIVE (4x that). An evicted id reports like an id from before a
        # restart that fell off the saved cap: KeyError, loudly -- never a silent wrong door.
        self.max_live = self.MAX_LIVE

    def add(self, rec, hook=None, hook_key=None):
        """Add (or re-add) a record; `hook` = an optional live callable(outcome). Re-adding an id keeps the reported
        outcome and COMBINES hooks (see the class docstring); hook_key names the hook's source explicitly (default:
        its code site). Returns the id."""
        own_specs = hook_specs(rec.meta)               # what THIS add's record names (before merging the old record's)
        if rec.id not in self._rows:
            self._order.append(rec.id)
            if len(self._order) > self.max_live:
                self._evict(len(self._order) - self.max_live)
        else:
            old = self._rows[rec.id]
            if old.outcome is not None and rec.outcome is None:
                # A decision made again is the SAME record; re-adding it must never erase what was reported.
                # (Found by test: the second systemone_decide on a state wiped its outcome.)
                rec.outcome = old.outcome
            if old is not rec:
                # the other door's hook specs travel with the record: old first, duplicates dropped
                merged, seen = [], set()
                for s in hook_specs(old.meta) + own_specs:
                    k = _spec_key(s)
                    if k not in seen:
                        seen.add(k)
                        merged.append(s)
                if merged:
                    rec.meta["hook"] = merged[0] if len(merged) == 1 else merged
        keyed_spec = None
        if hook is not None and hook_key is not None and self.resolve_hook is not None \
                and str(hook_key) in self.spec_keys:
            # A KEYED CLOSURE PERSISTS AS A SPEC (the audit's fix, class docstring): the record names its door, so a
            # reload rebuilds the hook from the record; the closure covers the spec while it lives (below)
            keyed_spec = {"kind": str(hook_key)}
            specs = hook_specs(rec.meta)
            if _spec_key(keyed_spec) not in {_spec_key(s) for s in specs}:
                specs.append(keyed_spec)
                rec.meta["hook"] = specs[0] if len(specs) == 1 else specs
        self._rows[rec.id] = rec
        # LAZY HRR ENCODING (E0.5): the encoding is needed only by similar(). Encoding on every add cost ~1-3.5 ms per
        # decision, i.e. up to ~14 s to reload a 4096-record ledger at boot; it is now computed on first use and
        # invalidated whenever the record changes (same vectors, same similar() results, bit for bit).
        self._vecs.pop(rec.id, None)
        if hook is not None:
            site = _hook_site(hook, hook_key)
            lst = self._hooks.setdefault(rec.id, [])
            for j, (s, _) in enumerate(lst):
                if s == site:
                    lst[j] = (site, hook)              # the same source re-issued its hook: replace, never run twice
                    break
            else:
                lst.append((site, hook))
            if own_specs:
                self._covered.setdefault(rec.id, set()).update(_spec_key(s) for s in own_specs)
            if keyed_spec is not None:
                self._covered.setdefault(rec.id, set()).add(_spec_key(keyed_spec))
        return rec.id

    MAX_LIVE = 16384

    def _evict(self, n):
        """Drop the n OLDEST records (and their hooks, cached encodings and aliases) from the live ledger. One slice,
        not n pop(0)s, so the cost is paid once per overflow."""
        gone, self._order = self._order[:n], self._order[n:]
        for rid in gone:
            self._rows.pop(rid, None)
            self._hooks.pop(rid, None)
            self._covered.pop(rid, None)
            self._vecs.pop(rid, None)
        if self._alias:
            dead = set(gone)
            self._alias = {a: b for a, b in self._alias.items() if b not in dead and a not in dead}
        return len(gone)

    def hooks_for(self, rid):
        """The hooks report(rid, ...) would run, as [source] strings: live closures by code site, then resolvable specs
        ('spec:<canonical json>'). For diagnostics and tests; nothing is called."""
        rid = self._alias.get(rid, rid)
        rec = self._rows.get(rid)
        if rec is None:
            return []
        out = [s for s, _ in self._hooks.get(rid, [])]
        if self.resolve_hook is not None:
            cov = self._covered.get(rid, set())
            out += ["spec:" + _spec_key(s) for s in hook_specs(rec.meta) if _spec_key(s) not in cov]
        return out

    def _vec(self, rid):
        v = self._vecs.get(rid)
        if v is None:
            v = self._vecs[rid] = self.codec.encode(self._rows[rid])
        return v

    def get(self, rid):
        return self._rows.get(self._alias.get(rid, rid))

    def report(self, rid, outcome):
        """THE outcome path. Sets the record's outcome, re-encodes it, and forwards to the hook (a
        SystemOne.observe closure for typed decisions). Returns {id, outcome, forwarded, was_correct}
        (+ hook_error when the hook raised).

        A HOOK THAT RAISES NO LONGER ABORTS THE REPORT (E0.5). Measured before: SystemOne.observe raises SchemaError
        for an outcome that is not an option ("wrong", a typo), the exception escaped report(), and decision_outcome
        never reached reflex_learn -- so the one outcome path silently skipped the reflex, the seen gate and the
        calibration for exactly the reports that were corrections. The error is now caught and carried in the
        report; the outcome itself is recorded either way."""
        rid = self._alias.get(rid, rid)
        rec = self._rows.get(rid)
        if rec is None:
            raise KeyError("no decision record %r" % rid)
        rec.outcome = outcome
        self._vecs.pop(rid, None)                        # re-encoded lazily (the outcome is part of the encoding)
        errors = []
        # 1) the live closures, in the order they were attached (wave 2: several per id)
        fns = list(self._hooks.get(rid, []))
        # 2) every hook SPEC the record names that no live closure already implements, rebuilt by the resolver from
        #    a shallow copy whose meta["hook"] is that one spec (so a resolver written for one spec works unchanged)
        if self.resolve_hook is not None:
            import copy as _copy
            cov = self._covered.get(rid, set())
            for spec in hook_specs(rec.meta):
                k = _spec_key(spec)
                if k in cov:
                    continue
                view = _copy.copy(rec)
                view.meta = dict(rec.meta, hook=spec)
                try:
                    fn = self.resolve_hook(view)
                except Exception as e:                  # a resolver that cannot rebuild the door: say so, go on
                    errors.append("resolve: %s: %s" % (type(e).__name__, str(e)[:160]))
                    continue
                if fn is not None:
                    fns.append(("spec:" + k, fn))
        # 3) run each, catching each: one door's exception never stops another door from learning
        results = []
        for _site, fn in fns:
            try:
                results.append(fn(outcome))
            except Exception as e:
                results.append(None)
                errors.append("%s: %s" % (type(e).__name__, str(e)[:160]))
        was_correct = outcome_matches(rec.answer, outcome) if rec.answer is not None else None
        # `forwarded` stays the FIRST hook's result (one hook -> exactly the old report); several -> forwarded_all too
        out = {"id": rid, "outcome": outcome, "forwarded": results[0] if results else None, "was_correct": was_correct}
        if len(fns) > 1:
            out["forwarded_all"] = results
            out["hooks"] = [s for s, _ in fns]
        if errors:
            out["hook_error"] = "; ".join(errors)
        return out

    # ---- persistence (E0.5): the ledger as TEXT, newest `cap` records, secrets never written ------------------
    CAP = 4096      # newest records kept across a save; a record is ~1 KB of text, so the section stays ~4 MB max

    def to_text(self, cap=None, sensitive=None):
        """-> {records: [record.to_text()], dropped_sensitive: n, dropped_cap: n}. `sensitive` is a callable(*texts)
        -> reason or None (holographic_learnguard.sensitive_reason by default): a record whose state, question,
        answer or outcome carries a secret is NEVER written -- it is counted instead. The newest `cap` survive."""
        if sensitive is None:
            from holographic.agents_and_reasoning.holographic_learnguard import sensitive_reason as sensitive
        cap = self.CAP if cap is None else int(cap)
        ids = list(self._order)
        dropped_cap = max(0, len(ids) - cap)
        rows, dropped_sensitive = [], 0
        for rid in ids[dropped_cap:]:
            rec = self._rows[rid]
            texts = [rec.state, rec.question, rec.answer, rec.outcome]
            if sensitive(*[t for t in texts if isinstance(t, str)]) or \
                    (rec.options and sensitive(options_text(rec.options))):
                # the OPTIONS too (wave 2): mind.rank's candidates are free-form text, so a pasted secret can be one
                dropped_sensitive += 1
                continue
            t = rec.to_text()
            if sensitive(json.dumps(t["meta"], sort_keys=True)):   # evidence / args can carry a pasted key too
                dropped_sensitive += 1
                continue
            rows.append(t)
        return {"records": rows, "dropped_sensitive": dropped_sensitive, "dropped_cap": dropped_cap}

    def load_text(self, rows, merge=True):
        """Rebuild records from to_text() rows (ids RE-DERIVED from the text and checked against the stored id; a
        mismatch keeps the stored id as an alias). merge=True keeps records already in the ledger (a record id
        present in both keeps the live one). -> {loaded, id_mismatch, invalid, kept_live}."""
        loaded = mismatch = invalid = kept = 0
        for t in rows or []:
            try:
                rec = DecisionRecord.from_text(t)
            except (ValueError, TypeError):
                invalid += 1                            # e.g. a via this version no longer knows
                continue
            if t.get("id") and t["id"] != rec.id:
                mismatch += 1
                self._alias[t["id"]] = rec.id
            if merge and rec.id in self._rows:
                # the live record wins, but a hook SPEC only the stored copy names still travels (wave 2: one id can
                # carry several doors' specs; dropping one here would silently cut that door off after the merge)
                live = self._rows[rec.id]
                specs, seen_ = [], set()
                for s in hook_specs(live.meta) + hook_specs(rec.meta):
                    if _spec_key(s) not in seen_:
                        seen_.add(_spec_key(s))
                        specs.append(s)
                if specs:
                    live.meta["hook"] = specs[0] if len(specs) == 1 else specs
                kept += 1
                continue
            self.add(rec)
            loaded += 1
        return {"loaded": loaded, "id_mismatch": mismatch, "invalid": invalid, "kept_live": kept}

    def similar(self, rec_or_id, k=3):
        """Records nearest to this one by cosine of their HRR encodings -- 'have we decided this before'."""
        rid = rec_or_id if isinstance(rec_or_id, str) else self.add(rec_or_id)
        rid = self._alias.get(rid, rid)
        v = self._vec(rid)
        out = sorted(((float(cosine(v, self._vec(o))), o) for o in self._order if o != rid), key=lambda t: (-t[0], t[1]))
        return [(o, c) for c, o in out[:k]]

    def stats(self):
        rows = list(self._rows.values())
        rep = [r for r in rows if r.outcome is not None]
        ok = [r for r in rep if r.answer is not None and outcome_matches(r.answer, r.outcome)]
        by_via = {}
        for r in rows:
            by_via[r.via] = by_via.get(r.via, 0) + 1
        return {"records": len(rows), "reported": len(rep), "correct_when_reported": (len(ok) / len(rep)) if rep else None, "by_via": by_via}


def _selftest():
    codec = RecordCodec(dim=2048, seed=0)
    r = DecisionRecord("card charged twice on my invoice", "cat", ["billing", "shipping"], "billing", "typed", margin=0.3)
    # 1. deterministic id; the outcome is not part of it
    assert r.id == record_id("card charged twice on my invoice", "cat", ["billing", "shipping"], "billing", "typed")
    r2 = DecisionRecord("card charged twice on my invoice", "cat", ["billing", "shipping"], "billing", "typed", outcome="billing")
    assert r.id == r2.id
    # 2. HRR round trip: every enum field recovers EXACTLY against its candidates, before and after an outcome
    v = codec.encode(r2)
    assert codec.decode(v, "answer", ["billing", "shipping", "refund"])[0] == "billing"
    assert codec.decode(v, "via", list(VIAS))[0] == "typed"
    assert codec.decode(v, "question", ["cat", "priority", "tone"])[0] == "cat"
    assert codec.decode(v, "outcome", ["billing", "shipping"])[0] == "billing"
    # 3. a record with no outcome decodes 'outcome' at low cosine (nothing bound there)
    assert codec.decode(codec.encode(r), "outcome", ["billing", "shipping"])[1] < 0.3
    # 4. the ledger: report by id forwards to the hook and scores correctness
    L = Ledger(codec)
    got = {}
    L.add(r, hook=lambda outcome: got.setdefault("seen", outcome))
    rep = L.report(r.id, "shipping")
    assert got["seen"] == "shipping" and rep["was_correct"] is False and L.get(r.id).outcome == "shipping"
    try:
        L.report("deadbeef00000000", "x"); raise AssertionError("unknown id accepted")
    except KeyError:
        pass
    # 5. similarity: the same question on a paraphrased state is nearer than a different question
    a = DecisionRecord("my card was charged two times", "cat", ["billing", "shipping"], "billing", "typed")
    b = DecisionRecord("where is my parcel", "cat", ["billing", "shipping"], "shipping", "typed")
    c = DecisionRecord("card charged twice on my invoice", "tone", ["angry", "calm"], "angry", "typed")
    for x in (a, b, c):
        L.add(x)
    sim = dict(L.similar(r.id, k=3))
    assert sim[a.id] > sim[b.id], sim
    # 6. determinism: encode twice, bit-identical
    assert np.array_equal(codec.encode(r), codec.encode(r))
    # 7. the loop closes by construction: a typed decision's hook is SystemOne.observe
    from holographic.agents_and_reasoning.holographic_systemone import SystemOne, _hash_bow_encode
    so = SystemOne(_hash_bow_encode(), scorer="nb", margin=0.0)
    so.fit({"cat": {"type": "choice", "options": ["billing", "shipping"], "examples": {"billing": ["card charged twice"], "shipping": ["parcel lost"]}}})
    state = "tracking shows no movement"
    ans = so.decide(state)["cat"]
    rec = DecisionRecord(state, "cat", ["billing", "shipping"], ans["ranked"][0][0], "typed")
    L.add(rec, hook=lambda outcome: so.observe(state, {"cat": outcome}, lr=2.0))
    before = dict(so._nb["cat"]["tot"])
    L.report(rec.id, "shipping")
    assert so._nb["cat"]["tot"]["shipping"] > before["shipping"]        # observe() ran, by id, with no teach
    assert L.stats()["reported"] == 2
    # 8. re-adding a decided record keeps its reported outcome (the wipe bug, pinned)
    L.add(DecisionRecord(state, "cat", ["billing", "shipping"], ans["ranked"][0][0], "typed"))
    assert L.get(rec.id).outcome == "shipping" and L.stats()["reported"] == 2
    # 9. (E0.5) a hook that RAISES does not abort the report: the outcome is recorded, the error carried
    bad = DecisionRecord("x", "cat", ["billing", "shipping"], "billing", "typed")
    L.add(bad, hook=lambda outcome: so.observe("x", {"cat": outcome}))
    rep = L.report(bad.id, "wrong")                     # not an option -> SchemaError inside observe
    assert "SchemaError" in rep["hook_error"] and L.get(bad.id).outcome == "wrong"
    # 10. (E0.5) text round trip: ids re-derive; a secret-bearing record is never written; a resolver re-derives
    #     the hook from rec.meta on a ledger that never saw the closure
    sec = DecisionRecord("my password: Hunter2-FAKE-9c1d", "cat", ["a", "b"], "a", "typed")
    L.add(sec)
    txt = L.to_text()
    assert txt["dropped_sensitive"] == 1 and all("Hunter2" not in json.dumps(r) for r in txt["records"])
    seen = []
    L2 = Ledger(resolve_hook=lambda r: (lambda outcome: seen.append((r.meta["hook"], outcome)) or "ok"))
    got2 = L2.load_text(json.loads(json.dumps(txt["records"])))
    assert got2["loaded"] == len(txt["records"]) and got2["id_mismatch"] == 0 and L2.get(rec.id).outcome == "shipping"
    hooked = DecisionRecord("y", "cat", ["billing", "shipping"], "billing", "typed", meta={"hook": {"kind": "t"}})
    L2.add(hooked)
    assert L2.report(hooked.id, "billing")["forwarded"] == "ok" and seen == [({"kind": "t"}, "billing")]
    # 11. (E0.6) the bare 'p' of a door record is deprecated: reading it warns, p_correct / p_null do not
    d = door_record({"value": "a", "p": 0.2}, "selftest", p_correct=0.9, p_null=0.2)
    DecisionDict._warned.discard("selftest")
    with warnings.catch_warnings(record=True) as w_:
        warnings.simplefilter("always")
        assert d["p_correct"] == 0.9 and d.get("p_null") == 0.2 and not w_
        assert d["p"] == 0.2 and any(issubclass(x.category, DeprecationWarning) for x in w_)
    assert json.loads(json.dumps(d)) == {"value": "a", "p": 0.2, "p_correct": 0.9, "p_null": 0.2}
    return {"ok": True, "pinned": 11}


def _selftest_hooks():
    """(wave 2) HOOKS COMBINE PER ID: two doors filing one decision under one id both learn; the same code site
    re-adding its hook replaces it (never runs twice); a raising hook does not stop the other. A separate selftest
    because tests/test_route_tiered.py pins _selftest()'s return value; tests/test_ledger_hooks.py runs this one."""
    L3 = Ledger(resolve_hook=lambda r: (lambda outcome: ("spec", r.meta["hook"]["kind"], outcome)))
    ran = []

    def door_a(outcome):
        ran.append(("a", outcome))
        return "a"

    def door_b(outcome):
        ran.append(("b", outcome))
        raise RuntimeError("b broke")
    one = DecisionRecord("s", "q", ["x", "y"], "x", "model_end", meta={"hook": {"kind": "typed"}})
    for _ in range(2):                                  # the same decision made twice: door_a registered twice
        L3.add(DecisionRecord("s", "q", ["x", "y"], "x", "model_end"), hook=door_a)
    L3.add(one)
    L3.add(DecisionRecord("s", "q", ["x", "y"], "x", "model_end"), hook=door_b)
    rep = L3.report(one.id, "x")
    assert ran == [("a", "x"), ("b", "x")], ran                                  # each once, in attach order
    assert rep["forwarded_all"][0] == "a" and rep["forwarded_all"][2] == ("spec", "typed", "x")
    assert "b broke" in rep["hook_error"] and L3.get(one.id).meta["hook"] == {"kind": "typed"}
    return {"ok": True, "pinned": 1}


def _selftest_spec_keys():
    """(learning-loop audit) A KEYED CLOSURE PERSISTS AS A SPEC: live, the closure runs once (its spec is covered);
    after a text round trip into a ledger that never saw the closure, the spec is resolved and the door learns."""
    ran = []
    resolver = (lambda r: (lambda outcome: ran.append(("spec", r.meta["hook"]["kind"], outcome)) or "spec"))
    L = Ledger(resolve_hook=resolver, spec_keys=("router",))
    rec = DecisionRecord("denoise an image", "route", ["a", "b"], "a", "route", meta={"tier": "answer"})
    L.add(rec, hook=lambda outcome: ran.append(("live", outcome)) or "live", hook_key="router")
    assert L.get(rec.id).meta["hook"] == {"kind": "router"}
    assert L.report(rec.id, "b")["forwarded"] == "live" and ran == [("live", "b")]      # once, not twice
    L2 = Ledger(resolve_hook=resolver, spec_keys=("router",))
    L2.load_text(json.loads(json.dumps(L.to_text(sensitive=lambda *t: None)["records"])))
    assert L2.report(rec.id, "a")["forwarded"] == "spec" and ran[-1] == ("spec", "router", "a")
    L3 = Ledger(resolve_hook=resolver)                                                 # no spec_keys: unchanged
    L3.add(DecisionRecord("x", "route", ["a"], "a", "route"), hook=lambda o: None, hook_key="router")
    assert "hook" not in L3.get(record_id("x", "route", ["a"], "a", "route")).meta
    return {"ok": True, "pinned": 1}


if __name__ == "__main__":
    print(_selftest(), _selftest_hooks(), _selftest_spec_keys())

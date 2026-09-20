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

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, bundle, cosine, unbind
from holographic.agents_and_reasoning.holographic_systemone import hashed_ngram_encode
from holographic.mesh_and_geometry.holographic_planshape import derived_atom

FIELDS = ("state", "question", "options", "answer", "via", "margin", "p", "cost")
VIAS = ("route", "typed", "tree", "swarm", "model_end", "human", "reflex")   # reflex: answered from experience (sweep 176)


def record_id(state, question, options, answer, via):
    """sha256 over the canonical decision fields -- the OUTCOME is not part of the id, so the same
    decision reported twice has one id and the second report is an update, not a new row."""
    canon = json.dumps({"state": (state or "").strip().lower(), "question": question, "options": list(options or []),
                        "answer": answer, "via": via}, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


class DecisionRecord:
    """{state, question, options, answer, via, margin, p, cost, outcome, id}. `via` names the door
    (route / typed / tree / swarm / model_end / human). outcome is None until reported."""

    def __init__(self, state, question, options, answer, via, margin=None, p=None, cost=None, outcome=None, meta=None):
        if via not in VIAS:
            raise ValueError("via must be one of %s, got %r" % (VIAS, via))
        self.state, self.question, self.options, self.answer, self.via = state, question, list(options or []), answer, via
        self.margin, self.p, self.cost, self.outcome = margin, p, cost, outcome
        self.meta = dict(meta or {})
        self.id = record_id(state, question, self.options, answer, via)

    def to_dict(self, max_chars=400):
        """JSON view. NOOA's pass-by-reference rule (bounded previews, sweep 130) applies to the STATE: a
        record never ships a huge input -- above max_chars it carries the true length, a head/tail sample and
        the sha256 of the full text, via holographic_boundedpreview. The HRR encoding still used the full
        text, so similarity is unaffected; only what travels over the wire is bounded."""
        state = self.state
        if isinstance(state, str) and len(state) > int(max_chars):
            from holographic.io_and_interop.holographic_boundedpreview import bounded_preview
            state = {"ref": hashlib.sha256(self.state.encode("utf-8")).hexdigest()[:16], "len": len(self.state),
                     "preview": bounded_preview(self.state, max_chars=max_chars, cost=False)}
        return {"id": self.id, "state": state, "question": self.question, "options": self.options,
                "answer": self.answer, "via": self.via, "margin": self.margin, "p": self.p, "cost": self.cost,
                "outcome": self.outcome, "meta": self.meta,
                # NOOA validated termination (sweep 176): a step carries its evidence and the command that
                # verifies it; the harness runs `verify` BEFORE the step is accepted (see swarm_step).
                "evidence": self.meta.get("evidence"), "verify": self.meta.get("verify")}


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


class Ledger:
    """Append-only store of records by id, with the outcome path. `hooks` map a record id to a
    callable(outcome) -> anything; a typed decision registers its fitted SystemOne here so that
    report(id, outcome) forwards to observe() -- the loop closes without a human in it."""

    def __init__(self, codec=None):
        self.codec = codec or RecordCodec()
        self._rows = {}
        self._order = []
        self._hooks = {}
        self._vecs = {}

    def add(self, rec, hook=None):
        if rec.id not in self._rows:
            self._order.append(rec.id)
        elif self._rows[rec.id].outcome is not None and rec.outcome is None:
            # A decision made again is the SAME record; re-adding it must never erase what was reported.
            # (Found by test: the second systemone_decide on a state wiped its outcome.)
            rec.outcome = self._rows[rec.id].outcome
        self._rows[rec.id] = rec
        self._vecs[rec.id] = self.codec.encode(rec)
        if hook is not None:
            self._hooks[rec.id] = hook
        return rec.id

    def get(self, rid):
        return self._rows.get(rid)

    def report(self, rid, outcome):
        """THE outcome path. Sets the record's outcome, re-encodes it, and forwards to the hook (a
        SystemOne.observe closure for typed decisions). Returns {id, outcome, forwarded, was_correct}."""
        rec = self._rows.get(rid)
        if rec is None:
            raise KeyError("no decision record %r" % rid)
        rec.outcome = outcome
        self._vecs[rid] = self.codec.encode(rec)
        forwarded = None
        if rid in self._hooks:
            forwarded = self._hooks[rid](outcome)
        was_correct = (rec.answer == outcome) if rec.answer is not None else None
        return {"id": rid, "outcome": outcome, "forwarded": forwarded, "was_correct": was_correct}

    def similar(self, rec_or_id, k=3):
        """Records nearest to this one by cosine of their HRR encodings -- 'have we decided this before'."""
        rid = rec_or_id if isinstance(rec_or_id, str) else self.add(rec_or_id)
        v = self._vecs[rid]
        out = sorted(((float(cosine(v, self._vecs[o])), o) for o in self._order if o != rid), key=lambda t: (-t[0], t[1]))
        return [(o, c) for c, o in out[:k]]

    def stats(self):
        rows = list(self._rows.values())
        rep = [r for r in rows if r.outcome is not None]
        ok = [r for r in rep if r.answer is not None and r.answer == r.outcome]
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
    return {"ok": True, "pinned": 8}


if __name__ == "__main__":
    print(_selftest())

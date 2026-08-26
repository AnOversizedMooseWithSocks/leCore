"""The findings registry (backlog D3): a research log as a holographic KNOWLEDGE STRUCTURE you query by
similarity and that detects its own contradictions.

A 'finding' is a structured claim -- a SUBJECT affects an OBJECT with a POLARITY (+1 helps / strengthens,
-1 hurts / backfires), optionally under a CONDITION (a regime: a horizon, a session, an asset). It is
encoded the way every record in this engine is, as role-bound binding:

    finding = bind(SUBJ, subject) + bind(OBJ, object) [+ bind(COND, condition)]

so the same operations the relations layer already runs (explain, name, analogy-as-unbind in
holographic_relations.KnowledgeStore) compose with it. What this module ADDS, which the relations layer
does not have, is the part that makes a research log honest:

  * QUERY by similarity -- 'what do we know about X?' bundles the given slots and recalls the findings
    whose structure matches. It is ROLE-SENSITIVE: querying object=momentum recalls findings where
    momentum is the OBJECT, not ones where it is the subject -- the value of structured encoding over a
    bag of words.
  * TENSION detection -- the headline. Two findings make the SAME claim if their (subject, object)
    bindings match (cosine, holographically); if they then carry OPPOSITE polarity they are in tension.
    The registry distinguishes a FLAT contradiction (same/no condition -- genuinely conflicting, one must
    be wrong) from a CONDITIONED tension (different conditions -- reconcilable: the outcome is conditioned
    on the differing dimension). 'ER strengthens momentum at 10 days' vs 'ER backfires intraday' is the
    latter, and saying so is the point.

THE EXACT-DOOR DISCIPLINE (the engine's rule): retrieval is holographic (cosine over the bound claim),
but the VERDICT -- opposite polarity, same-vs-different condition -- is read from the exact stored fields,
not from approximate similarity. Similarity FINDS the candidate conflicts; exact comparison JUDGES them.

SCOPE / KEPT NEGATIVES:
  * It operates on findings encoded as STRUCTURED claims (subject, object, polarity, condition), NOT on
    free prose. Turning 2300 lines of narrative into structured claims is an NLP step this engine does not
    do (no embeddings, no parser) -- that is the manual / future-LLM boundary, stated plainly.
  * The tension scan is O(n^2) pairwise claim-cosine -- fine for a few thousand findings; a HoloForest
    pre-filter would be the move at larger scale (the engine's standard sublinear-recall answer), noted
    but not needed yet.
"""
import json

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, bundle, cosine, Vocabulary


class FindingRegistry:
    """A holographic store of structured research findings: query by similarity, and detect flat
    contradictions vs conditioned tensions. Owns its vocabularies, so all structure is reproducible from
    the seeds (the demoscene rule). A finding is (subject, object, polarity in {+1,-1}, condition, note)."""

    def __init__(self, dim=2048, seed=0):
        self.dim = int(dim)
        self.seed = int(seed)                       # remembered so a saved registry rebuilds its vectors
        # Unitary role/token atoms: a finding is pure role-filler binding + unbind-by-role, the few-factor
        # path where exact unbinding widens the cleanup margin at no storage cost (the KnowledgeStore lesson).
        self.roles = Vocabulary(self.dim, seed + 1, unitary=True)
        self.tokens = Vocabulary(self.dim, seed + 2, unitary=True)
        self.findings = []                          # list of dicts: the structured records (human-readable)
        self._vecs = []                             # full finding vectors (subject + object + condition) for query
        self._claims = []                           # claim vectors (subject + object only) for tension pairing

    def _bind(self, role, token):
        return bind(self.roles.get(role), self.tokens.get(token))

    def add(self, subject, object, polarity, condition=None, note=None):
        """Record a finding: SUBJECT affects OBJECT with POLARITY (+1 helps/strengthens, -1 hurts/backfires),
        optionally under CONDITION (a regime string). `note` is the human phrasing. Returns the finding index."""
        if int(polarity) not in (1, -1):
            raise ValueError("polarity must be +1 (helps/strengthens) or -1 (hurts/backfires)")
        rec = {"subject": subject, "object": object, "polarity": int(polarity),
               "condition": condition, "note": note}
        claim = bundle([self._bind("SUBJ", subject), self._bind("OBJ", object)])    # the claim identity
        full = [self._bind("SUBJ", subject), self._bind("OBJ", object)]
        if condition is not None:
            full.append(self._bind("COND", condition))
        self.findings.append(rec)
        self._claims.append(claim)
        self._vecs.append(bundle(full))
        return len(self.findings) - 1

    def query(self, subject=None, object=None, condition=None, k=None, floor=0.2):
        """Recall findings by similarity to a PARTIAL claim: bundle the binds for whichever of subject /
        object / condition you give, and rank findings by cosine to that query. Role-sensitive (object=X
        matches findings with X in the OBJECT slot, not the subject). Returns a list of
        {finding, score} ranked high-to-low, keeping those above `floor` (or the top `k`)."""
        parts = []
        if subject is not None:
            parts.append(self._bind("SUBJ", subject))
        if object is not None:
            parts.append(self._bind("OBJ", object))
        if condition is not None:
            parts.append(self._bind("COND", condition))
        if not parts:
            raise ValueError("query needs at least one of subject=, object=, condition=")
        q = bundle(parts)
        scored = [{"index": i, "finding": self.findings[i], "score": float(cosine(q, v))}
                  for i, v in enumerate(self._vecs)]
        scored.sort(key=lambda r: r["score"], reverse=True)
        if k is not None:
            return scored[:k]
        return [r for r in scored if r["score"] >= floor]

    def related(self, subject=None, object=None, condition=None, k=5):
        """Surface the findings most related to a query slot-set -- the 'find analogous / related findings'
        operation, as content-addressable recall over the structured claims. (Deeper relational analogy --
        a:b::c:? -- is holographic_relations.KnowledgeStore's job and composes with these records.)"""
        return self.query(subject=subject, object=object, condition=condition, k=k)

    def tensions(self, claim_tol=0.85):
        """Detect the registry's own contradictions. For every pair of findings whose CLAIMS match
        (cosine of their subject+object bindings >= claim_tol) and whose POLARITIES are OPPOSITE, report a
        tension -- classified FLAT (same or absent condition: genuinely conflicting, one must be wrong) or
        CONDITIONED (different conditions: reconcilable, the outcome is conditioned on the differing
        dimension). Returns a list of dicts {a, b, subject, object, type, conditions, claim_similarity,
        notes}. Retrieval is holographic (the claim cosine); the verdict is exact (polarity sign, condition
        equality) -- similarity finds the candidates, exact comparison judges them."""
        out = []
        n = len(self.findings)
        for i in range(n):
            for j in range(i + 1, n):
                fi, fj = self.findings[i], self.findings[j]
                if fi["polarity"] * fj["polarity"] >= 0:            # same direction -> not a tension
                    continue
                sim = float(cosine(self._claims[i], self._claims[j]))
                if sim < claim_tol:                                 # different claims -> not a contradiction
                    continue
                same_cond = (fi["condition"] == fj["condition"])
                out.append({
                    "a": i, "b": j,
                    "subject": fi["subject"], "object": fi["object"],
                    "type": "flat" if same_cond else "conditioned",
                    "conditions": (fi["condition"], fj["condition"]),
                    "claim_similarity": sim,
                    "notes": (fi["note"], fj["note"]),
                })
        return out

    # -- persistence: a research log should outlive the session ------------------------------------
    # The saved file holds ONLY the structured claims plus dim/seed -- NO vectors. On load the vectors
    # are REBUILT from the seeds (the demoscene / determinism rule), so the file is tiny and a reloaded
    # registry is bit-identical to the original (its query and tension verdicts reproduce exactly).
    def to_state(self):
        """Serialise to a plain dict: dimension, seed, and the list of structured findings. The vectors are
        deliberately NOT stored -- they are a deterministic function of the findings and the seed."""
        return {"dim": self.dim, "seed": self.seed,
                "findings": [dict(f) for f in self.findings]}

    @classmethod
    def from_state(cls, state):
        """Rebuild a registry from a `to_state` dict: re-add every finding, which regenerates its vectors
        from the seeds -- so the restored registry recalls and detects tensions identically to the original."""
        reg = cls(dim=state["dim"], seed=state["seed"])
        for f in state["findings"]:
            reg.add(f["subject"], f["object"], f["polarity"], f.get("condition"), f.get("note"))
        return reg

    def save(self, path):
        """Write the registry to `path` as JSON (the structured claims; vectors rebuild on load)."""
        with open(path, "w") as fh:
            json.dump(self.to_state(), fh, indent=2)
        return path

    @classmethod
    def load(cls, path):
        """Load a registry saved by `save` -- a durable, growable research log."""
        with open(path) as fh:
            return cls.from_state(json.load(fh))


def _selftest():
    reg = FindingRegistry(dim=2048, seed=0)
    # the backlog's own example: a horizon-conditioned tension (same claim, opposite polarity, DIFFERENT
    # conditions -> reconcilable, not a flat contradiction)
    i0 = reg.add("efficiency_ratio", "momentum", +1, condition="horizon_10d",
                 note="high ER strengthens momentum at the 10-day horizon")
    i1 = reg.add("efficiency_ratio", "momentum", -1, condition="intraday",
                 note="high ER backfires intraday")
    # some unconflicted real edges
    reg.add("low_vol", "vol_expansion", +1, note="low vol precedes expansion")
    reg.add("funding_extreme", "reversal", +1, note="funding extreme -> selective bounce")
    # a PLANTED flat contradiction (same claim, opposite polarity, NO condition either side)
    i4 = reg.add("bracket_order", "convexity", +1, note="the bracket looks convex")
    i5 = reg.add("bracket_order", "convexity", -1, note="the bracket's drift masquerades as convexity")
    # an unrelated finding that must NOT be flagged (momentum here is the SUBJECT, not the object)
    reg.add("momentum", "trend", +1, note="momentum tends to trend")

    # (1) similarity query recalls the right findings, and is ROLE-SENSITIVE.
    by_subject = reg.query(subject="efficiency_ratio", k=2)
    assert {r["index"] for r in by_subject} == {i0, i1}, "query by subject did not recall the ER findings"
    by_object = reg.query(object="momentum", floor=0.4)            # momentum as OBJECT -> findings 0,1 only
    got = {r["index"] for r in by_object}
    assert i0 in got and i1 in got, "query by object missed the findings where momentum is the object"
    assert 5 not in got, "query by object wrongly matched a finding where momentum is the SUBJECT (role-blind)"

    # (2) THE HEADLINE: tensions are detected and correctly classified, with no false positives.
    tens = reg.tensions()
    pairs = {(t["a"], t["b"]): t["type"] for t in tens}
    assert (i0, i1) in pairs and pairs[(i0, i1)] == "conditioned", \
        f"ER tension should be CONDITIONED (10d vs intraday), got {pairs.get((i0, i1))}"
    assert (i4, i5) in pairs and pairs[(i4, i5)] == "flat", \
        f"bracket tension should be FLAT (same/no condition), got {pairs.get((i4, i5))}"
    assert len(tens) == 2, f"expected exactly two tensions, got {len(tens)}: {pairs}"

    # (3) PERSISTENCE: save -> load reproduces the registry exactly. The file stores only the structured
    # claims (no vectors); the vectors rebuild from the seeds, so recall and tension verdicts are identical.
    import os, tempfile
    tmp = os.path.join(tempfile.gettempdir(), "_fr_selftest.json")
    reg.save(tmp)
    reg2 = FindingRegistry.load(tmp)
    os.remove(tmp)
    assert [dict(f) for f in reg2.findings] == [dict(f) for f in reg.findings], "save/load lost findings"
    assert all(np.array_equal(a, b) for a, b in zip(reg._vecs, reg2._vecs)), "rebuilt finding vectors differ"
    assert all(np.array_equal(a, b) for a, b in zip(reg._claims, reg2._claims)), "rebuilt claim vectors differ"
    assert {(t["a"], t["b"]): t["type"] for t in reg2.tensions()} == pairs, "tensions did not survive save/load"

    print("holographic_knowledge selftest OK:")
    print(f"  query subject=efficiency_ratio -> findings {sorted(r['index'] for r in by_subject)} (role-sensitive)")
    print(f"  CONDITIONED tension: ER->momentum  {reg.findings[i0]['condition']} vs {reg.findings[i1]['condition']}  "
          f"(reconcilable, not flat)")
    print(f"  FLAT contradiction:  bracket_order->convexity  (no condition, must resolve)")
    print(f"  exactly {len(tens)} tensions; unrelated findings not flagged")
    print(f"  save/load round-trip: {len(reg2.findings)} findings restored from claims-only file, "
          f"vectors rebuilt bit-identical, tensions reproduce")


if __name__ == "__main__":
    _selftest()

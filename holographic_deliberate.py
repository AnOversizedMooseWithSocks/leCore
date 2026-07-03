"""Deliberation: don't emit the first thing generated. Think, iterate, keep the
best, and say it -- with the amount of thinking adapting to how hard it is.

This models the loop you do before speaking: form the gist of what you mean, draft
it as inner speech, judge the draft, and either refine it or -- if it is already
good enough -- say it. Sometimes that is one quick pass; sometimes it takes
several. Built on the parts already here:

    gist        -- the query's meaning target: the abstract anchor of what to say
    draft       -- realize the gist into words (the meaning predictor + the
                   structure guard), the inner-speech surfacing
    judge       -- quality = relevance (on-gist) + a weight x structure (coherent)
    iterate     -- generate DIVERSE drafts (a first greedy one, then stochastic
                   ones), keep the best, and STOP EARLY once quality clears a bar;
                   the iteration count is the 'thinking time', and it adapts: easy
                   queries settle in one or two passes, hard ones use the budget

WHAT WAS MEASURED, INCLUDING TWO KEPT NEGATIVES THAT SHAPED THE DESIGN:
  * THE LOOP HELPS. Deliberated drafts (best of an adaptive N) score higher than a
    single greedy pass (quality ~0.40 vs ~0.34 on Brown queries), and the
    iteration count is genuinely adaptive -- easy queries stop at 1-2, hard ones
    run the full budget. That adaptivity IS the 'sometimes fast, sometimes slow'.
  * ELABORATING THE PLAN DOES NOT HELP (measured, kept). Two richer 'abstract
    thought' plans were tried and both UNDERPERFORMED the flat query target:
    (a) rolling the meaning predictor forward into a trajectory drifts into
    function words ('in', 'the', 'of'), because the predictor is syntagmatic --
    it predicts what FOLLOWS, not a semantic arc; (b) enriching the target with
    its meaning-neighbours is neutral at best. The honest conclusion: on this
    substrate the query's own meaning target is already the right abstract anchor;
    the human-like gain is in the iterate-and-keep-best loop, not in elaborating
    the plan. So the deliberator keeps the gist simple and spends its effort on
    the loop.

The result is generation that pauses to choose its words: it drafts, judges, and
only emits when the draft is good enough or the thinking budget is spent -- and it
exposes the trace (the candidate drafts and their scores), so the inner
deliberation is visible rather than hidden.

A MULTI-JUDGE layer (negotiate) extends this: instead of one quality number,
several judges score each draft -- coherence (structure), relevance (on-query), and
novelty (anti-repetition) -- and the negotiated score is the MINIMUM across them,
so the kept draft is the most BALANCED rather than one that wins a single axis while
failing another. The judges pull against each other (coherence likes common,
sometimes repetitive text; novelty penalises repetition), so this is competing
pressures resolving before something surfaces. Measured: with the structure guard
already suppressing most loops, the novelty judge is mostly a safety net -- it
matches the single-quality loop on repetition in the typical case and rescues the
occasional repetitive draft (type-token ratio 0.85 -> 0.96 on the one query that
needed it). The per-judge trace makes the tension visible.

Needs: numpy, holographic_ai, a fitted MeaningPredictor and StructureVerifier.
"""
import numpy as np

from holographic_ai import cosine
from holographic_respond import query_target, relevance, _STOP


class Deliberator:
    """Iterate over drafts before emitting, with adaptive thinking time."""

    def __init__(self, predictor, verifier, struct_weight=0.15):
        self.mp = predictor
        self.v = verifier
        self.struct_weight = struct_weight
        self._Cn = predictor._matrix()[0]

    def _realize(self, query, gist, query_weight=5.0, length=26, beam=8,
                 lookback=8, temp=0.0, rng=None):
        """Realize the gist into a draft. temp=0 is the greedy first draft; temp>0
        samples among candidates to produce a different draft each call."""
        mp, v, M, idx = self.mp, self.v, self.mp.M, self.mp.idx
        words = query.split() if isinstance(query, str) else list(query)
        seed = [w for w in words if w in idx and w not in _STOP][:2]
        out = list(seed)
        while len(out) < mp.order:
            out = ["the"] + out
        for _ in range(length):
            qv = mp.context_vector(out[-mp.order:])
            qn = qv / (np.linalg.norm(qv) + 1e-12)
            order = np.argsort(self._Cn @ qn)[::-1][:beam]
            cands, seen = [], set()
            for j in order:
                w = mp._next[j]
                if w not in seen:
                    seen.add(w)
                    cands.append(w)
            if not cands:
                break
            sc = np.array([
                v.structure_score((out + [w])[-lookback:])
                + (query_weight * cosine(M[idx[w]], gist) if (w in idx and np.linalg.norm(gist) > 0) else 0.0)
                for w in cands])
            if temp > 0 and rng is not None:
                p = np.exp((sc - sc.max()) / temp)
                p /= p.sum()
                out.append(cands[int(rng.choice(len(cands), p=p))])
            else:
                out.append(cands[int(np.argmax(sc))])
        return out[len(seed):] if seed else out

    def quality(self, draft, gist):
        """How good a draft is: on-gist relevance plus a weighted structure score."""
        if not draft:
            return -1e9
        return (relevance(draft, gist, self.mp.idx, self.mp.M)
                + self.struct_weight * self.v.structure_score(draft))

    # ---- multi-judge negotiation ----------------------------------------
    def judges(self):
        """The default panel of judges, each mapping a draft to a [0, 1] score.
        They pull in different directions -- coherence likes common (often
        repetitive) text, novelty penalises repetition, relevance wants on-topic --
        so a good draft has to satisfy competing pressures at once.

        coherence -- structure score mapped through the verifier's own threshold
        relevance -- cosine of the draft to the query gist
        novelty   -- type-token ratio: the fraction of distinct words, which falls
                     when a draft loops or repeats
        Returns a list of (name, fn(draft, gist) -> [0,1])."""
        def coherence(draft, gist):
            # Logistic sigmoid of the (score - threshold) margin. Written the numerically-stable way: taking
            # exp() of a large POSITIVE argument overflows (the RuntimeWarning this used to emit), so we branch
            # on the sign of z and only ever exp() a value <= 0. Mathematically identical to 1/(1+exp(-z)).
            z = (self.v.structure_score(draft) - self.v.threshold) * 1.5
            if z >= 0.0:
                return float(1.0 / (1.0 + np.exp(-z)))
            ez = np.exp(z)                                    # z < 0 here, so ez is in (0, 1) -- no overflow
            return float(ez / (1.0 + ez))

        def relevance_j(draft, gist):
            return float(max(0.0, min(1.0, relevance(draft, gist, self.mp.idx, self.mp.M))))

        def novelty(draft, gist):
            return float(len(set(draft)) / max(1, len(draft)))

        return [("coherence", coherence), ("relevance", relevance_j), ("novelty", novelty)]

    def negotiate(self, query, judges=None, max_iters=8, target_quality=0.55,
                  length=26, query_weight=5.0, temp=0.5, seed=0):
        """Deliberate under SEVERAL competing judges. Each draft is scored by every
        judge; the negotiated score is the MINIMUM across judges -- the binding
        pressure -- so the kept draft is the one whose WEAKEST dimension is least
        bad (balanced), not one that wins on a single axis while failing another.
        Stops early once the negotiated score clears target_quality. Returns the
        response, its per-judge scores, the negotiated score, the iterations, and a
        trace where every draft's full judge breakdown is visible -- the competing
        pressures resolving, made explicit."""
        if judges is None:
            judges = self.judges()
        rng = np.random.default_rng(seed)
        gist = query_target(query, self.mp.idx, self.mp.M)
        best, best_score, best_scores, trace = None, -1.0, {}, []
        iters = 0
        for i in range(max_iters):
            iters = i + 1
            draft = self._realize(query, gist, query_weight=query_weight,
                                  length=length, temp=(0.0 if i == 0 else temp), rng=rng)
            if not draft:
                continue
            scores = {name: round(fn(draft, gist), 3) for name, fn in judges}
            negotiated = min(scores.values())          # the binding pressure
            trace.append({"draft": draft, "scores": scores, "negotiated": round(negotiated, 3)})
            if negotiated > best_score:
                best, best_score, best_scores = draft, negotiated, scores
            if best_score >= target_quality:
                break
        return {"response": best, "scores": best_scores, "negotiated": float(best_score),
                "iterations": iters, "trace": trace}

    def deliberate(self, query, max_iters=8, target_quality=0.45, length=26,
                   query_weight=5.0, temp=0.5, seed=0):
        """Think before speaking: draft, judge, and refine -- keeping the best --
        stopping early once a draft clears target_quality. Returns a dict with the
        chosen response, its quality, the iterations used (the thinking time), and
        the full trace of drafts and scores (the inner deliberation, made visible)."""
        rng = np.random.default_rng(seed)
        gist = query_target(query, self.mp.idx, self.mp.M)
        best, best_q, trace = None, -1e18, []
        iters = 0
        for i in range(max_iters):
            iters = i + 1
            draft = self._realize(query, gist, query_weight=query_weight,
                                  length=length, temp=(0.0 if i == 0 else temp), rng=rng)
            q = self.quality(draft, gist)
            trace.append({"draft": draft, "quality": round(float(q), 3)})
            if q > best_q:
                best, best_q = draft, q
            if best_q >= target_quality:
                break
        return {"response": best, "quality": float(best_q), "iterations": iters,
                "relevance": relevance(best, gist, self.mp.idx, self.mp.M) if best else 0.0,
                "structure": float(self.v.structure_score(best)) if best else 0.0,
                "trace": trace}

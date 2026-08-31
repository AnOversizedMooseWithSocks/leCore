"""Answer, return the set, or abstain -- one policy object, every threshold from a null.

WHY THIS EXISTS. Measurement this arc produced three behaviours that each have a number, and
three ad-hoc checks scattered across callers is how they drift apart:
  * WELL-POSED query   -> one answer.       Measured 0.875 against a Bayes ceiling of 0.858.
  * AMBIGUOUS query    -> the set + size.   Set-recall 1.000 at a median of TWO passages, where
                                            forcing top-1 scores 0.458 against a 0.444 ceiling.
  * NO LEXICAL MATCH   -> abstain.          99.2% refusal where forcing an answer is wrong 100%
                                            of the time.

THE AMBIGUITY IS NOT ESTIMATED, IT IS COUNTED. If m passages contain every query term, no scorer
seeing only those terms can separate them, and a deterministic selector is right in at most 1 of
the m symmetric cases (machine-checked as tied_selector_correct_at_most_once). So the honest
output above m=1 is the SET, not a ranking. Intersecting postings is cheap; the published
unsupervised QPP estimators (NQC, WIG, sigma_max) were measured against this and NQC in
particular barely beat the base rate -- an inverted index already knows what they try to guess.

NO LEARNED WEIGHTS AND NO TUNING ON THE EVALUATION. The abstain threshold comes from a NULL of
scrambled queries drawn from the corpus's own vocabulary at matched length. A threshold fitted to
the thing it is judged by is not a gate.
"""
import hashlib

import numpy as np



def _min_window_span(positions):
    """Smallest window containing one occurrence of each present query term."""
    lists = [q for q in positions if q]
    if not lists:
        return 0, 10 ** 9
    idx = [0] * len(lists)
    best = 10 ** 9
    while True:
        cur = [lists[i][idx[i]] for i in range(len(lists))]
        best = min(best, max(cur) - min(cur) + 1)
        k = min(range(len(cur)), key=lambda j: cur[j])
        idx[k] += 1
        if idx[k] >= len(lists[k]):
            break
    return len(lists), best


def proximity_key(doc_tokens, qterms, length_norm=True):
    """Rerank key: coverage, then tightness, then ordered adjacency.

    WHY LEXICOGRAPHIC AND NOT A WEIGHTED MIX: a mixture needs its weights chosen on held-out
    queries to be honest, and an unjustified mixture is a tuned number pretending to be a method.
    BM25 is a bag of words and cannot separate passages that share vocabulary but ARRANGE it
    differently; this adds exactly the arrangement signal and nothing else.

    MEASURED at K=20,000 over SIX independent query draws of 200: +0.0533 +- 0.0167 top-1 over
    BM25 alone (pooled paired permutation p=0.00000). Reported multi-draw because a single draw of
    the same experiment gave +0.0733 p=0.0002 and another gave +0.0400 p=0.1363 -- both noise
    around the same true effect.
    KEPT NEGATIVE: it does NOT help the near-duplicate regime (+0.033, ns, with 0.617 of headroom
    available). Those queries are ambiguous, not mis-ranked, and no reranker breaks a tie the data
    does not contain.
    """
    pos = {}
    for i, t in enumerate(doc_tokens):
        pos.setdefault(t, []).append(i)
    covered, span = _min_window_span([pos.get(t, []) for t in qterms])
    qs = set(qterms)
    bigram = sum(1 for a, b in zip(doc_tokens, doc_tokens[1:]) if a in qs and b in qs)
    if not length_norm:
        return (covered, -span, bigram)
    # LENGTH-NORMALISED COVERAGE. Raw `covered` is a DOCUMENT-LENGTH PRIOR in disguise: with a
    # 148-term query the longest candidate covers the most terms, and measured on the held-out
    # docstring benchmark the raw key picked the longest candidate 89% of the time and collapsed
    # to SIX distinct documents across every query -- top-1 accuracy 0.000. BM25 normalises by
    # length for exactly this reason; so does this. Ratios are rounded to 3 decimals so the
    # ordering stays a total order under float noise rather than flapping on ties.
    n = max(1, len(set(doc_tokens)))
    return (round(covered / n, 3), -span, bigram / n)


class RetrievalPolicy:
    """Decides the SHAPE of an answer, not its content. Scoring is the caller's business."""

    def __init__(self, docs, k1=1.5, b=0.75, pretokenized=False):
        """docs: text strings OR token lists.

        SCORING IS DELEGATED to holographic_bm25.BM25 -- there is exactly one BM25 in this engine
        and this is not it. That also means containment must use BM25's OWN tokenizer, which does
        Porter-style suffix normalisation: if the policy split terms differently from the scorer,
        the containment count would describe a different query than the one being scored.
        """
        from holographic.semantic_router.holographic_bm25 import BM25, tokenize
        self._tokenize = tokenize
        texts = [(" ".join(d) if not isinstance(d, str) else d) for d in docs]
        self.bm = BM25(texts, k1=k1, b=b)
        self.docs = self.bm.docs_tokens          # the scorer's view, not a parallel one

        # DOUBLE-TOKENIZATION DETECTOR. tokenize() is NOT idempotent -- 'settings'->'setting'->
        # 'sett', 'classes'->'class'->'clas' -- so handing it already-tokenized input silently
        # OVER-STEMS the whole index. That shipped once: a page built this way passed its own
        # self-test and disagreed with the faculty on 8 of 60 queries. Raw and already-normalised
        # input are NOT distinguishable in general, so this does not guess; it MEASURES how much
        # a second pass would change and exposes the number, and `pretokenized=True` skips the
        # pass entirely for callers who know. Measured violation rate on this repo's vocabulary:
        # 138 of 4,857 terms (2.8%) -- small enough to hide, large enough to change answers.
        self.pretokenized = bool(pretokenized)
        if self.pretokenized and not isinstance(docs[0] if docs else "", str):
            self.docs = [list(d) for d in docs]
            self.bm.docs_tokens = self.docs
        flat = [t for d in self.docs[:200] for t in d]
        changed = sum(1 for t in flat if tokenize(t) != [t]) if flat else 0
        self.double_tokenization_risk = (changed / len(flat)) if flat else 0.0
        self.N = len(self.docs)
        self.inv = {}
        for i, d in enumerate(self.docs):
            for t in set(d):
                self.inv.setdefault(t, set()).add(i)
        self.threshold = None
        self.null = None
        self._sem = None
        self._sem_docs = None
        self.degenerate = False
        self.degenerate_reason = None

    # ---- scoring: DELEGATED, never reimplemented ---------------------------------------------
    def scores(self, qterms):
        """BM25 scores from the engine's own scorer. See holographic_bm25.BM25.scores."""
        q = qterms if isinstance(qterms, str) else " ".join(qterms)
        return np.asarray(self.bm.scores(q), dtype=float)

    def terms(self, query):
        """Normalise a query THE WAY THE SCORER WILL, so containment and scoring agree."""
        return self._tokenize(query if isinstance(query, str) else " ".join(query))

    # ---- optional semantic arm ----------------------------------------------------------------
    def attach_semantic(self, mind, dim=256, seed=0):
        """Add a MEANING arm built from the vendored dictionary -- unsupervised, no learned weights.

        WHY IT IS OPTIONAL AND OFF BY DEFAULT. Measured on realistic source text the lexical arm is
        AT its Bayes ceiling and the semantic arm is far behind (0.085 vs 0.785 top-1 at 20k docs),
        so switching it on there would cost accuracy for nothing. Measured on PARAPHRASE queries --
        where a description shares no surface form with its target -- the lexical arm scores 0.000
        and this one is the only thing that answers at all. Two arms, two regimes; the caller says
        which, and the numbers for both are on the record rather than a default someone inherits.

        A document's meaning vector is the SUM of its terms' meaning vectors, and a term's vector
        comes from `build_semantic_index` (random indexing over dictionary glosses). Terms are taken
        through THE SAME boundary the lexical arm uses, so the two arms cannot disagree about what a
        term is -- the cross-normalisation hazard the audit gate budgets.
        """
        import numpy as _np
        vocab = sorted({t for d in self.docs for t in d})
        self._sem = mind.build_semantic_index(words=vocab, dim=int(dim), seed=int(seed))
        D = int(dim)
        M = _np.zeros((self.N, D))
        for i, d in enumerate(self.docs):
            for t in set(d):
                try:
                    M[i] += self._sem.vector(t)
                except Exception:
                    pass                      # a term the dictionary cannot define contributes nothing
        n = _np.linalg.norm(M, axis=1, keepdims=True)
        self._sem_docs = M / _np.where(n > 0, n, 1.0)
        return self

    def semantic_scores(self, qterms):
        """Cosine of the query's meaning vector against every document's. Empty if no arm attached."""
        import numpy as _np
        if getattr(self, "_sem", None) is None:
            return _np.zeros(self.N)
        v = _np.zeros(self._sem_docs.shape[1])
        for t in set(self.terms(qterms)):     # THE SAME boundary as the lexical arm
            try:
                v += self._sem.vector(t)
            except Exception:
                pass
        nv = _np.linalg.norm(v)
        if nv == 0:
            return _np.zeros(self.N)
        return self._sem_docs @ (v / nv)

    # ---- calibration -------------------------------------------------------------------------
    def calibrate(self, n=200, terms=8, percentile=95.0, seed=0):
        """Threshold from SCRAMBLED queries: vocabulary-matched nonsense that should never be
        answered. The percentile IS the accepted false-answer rate, stated rather than tuned."""
        rng = np.random.default_rng(seed)
        vocab = sorted(self.inv)
        if not vocab:
            self.threshold = 0.0
            return self
        top = []
        for _ in range(n):
            q = [vocab[j] for j in rng.choice(len(vocab), min(terms, len(vocab)), replace=False)]
            s = self.scores(q)
            top.append(float(s.max()) if s.size else 0.0)
        self.null = np.array(top)
        thr = float(np.percentile(self.null, percentile))
        # DEGENERACY GUARD. The null only means something when a scrambled query looks DIFFERENT
        # from a real one. On a tiny corpus the vocabulary is barely larger than the query, so
        # the two distributions coincide and the threshold swallows every real query -- the
        # policy would then abstain on everything while reporting a perfectly healthy number.
        # Found by a 3-document smoke test; refusing everything silently is worse than not
        # gating at all, so the gate DISABLES ITSELF and says why.
        self.degenerate = (self.N < 30) or (len(self.inv) < 8 * max(1, terms))
        if self.degenerate:
            self.threshold = 0.0
            self.degenerate_reason = (
                "corpus too small to calibrate a null: %d passages, %d vocabulary terms against "
                "%d-term queries -- scrambled and real queries are indistinguishable here, so "
                "the abstain gate is DISABLED rather than silently refusing everything"
                % (self.N, len(self.inv), terms))
        else:
            self.threshold = thr
            self.degenerate_reason = None
        return self

    # ---- the decision --------------------------------------------------------------------------
    def containment(self, qterms, topk=50):
        """Documents a presence-based scorer CANNOT separate: those matching the SAME SET of query
        terms as the top-scoring one.

        WHY THIS AND NOT THE TWO THINGS TRIED BEFORE.
          * exact AND (documents holding EVERY query term) is a short-query notion. On 148-term
            prose queries it is empty 100% of the time -- a representation bug, since it cannot
            express "no document has them all".
          * MAXIMUM COVERAGE fixes the emptiness and introduces something worse: on prose it fires
            2.2% of the time with SET-RECALL 0.000, because a coverage tie on a long query is a tie
            on GENERIC vocabulary. A false ambiguity signal is worse than none.
        Matching the same term SET is the honest condition: only tf and length can then separate
        the documents, and those are the weak signals. Measured, it reduces EXACTLY to the old
        behaviour where ambiguity is real (short queries: fires 17.2%, set-recall 1.000, identical
        to max coverage) and stays silent where it is not (prose: 0.000%). Long prose queries are
        NOT ambiguous -- every document matches a different subset -- so "answer" was always the
        right verdict there, and the old code reached it for the wrong reason.
        """
        terms = sorted(set(self.terms(qterms)))
        if not terms:
            return set()
        scores = self.scores(qterms)
        if not scores.size:
            return set()
        import numpy as _np
        order = _np.argsort(-scores)[:int(topk)]
        sig = lambda d: frozenset(t for t in terms if d in self.inv.get(t, ()))
        top = sig(int(order[0]))
        if not top:
            return set()
        return {int(d) for d in order if sig(int(d)) == top}

    def verdict(self, qterms, scores=None, max_set=10, rerank=False, rerank_k=10):
        """(rerank defaults OFF -- it changes an existing decision, and defaults are not flipped
        silently. See proximity_key for the measured effect and its kept negative.)"""
        """Return a shape decision plus every number behind it.

        mode is one of:
          'abstain' -- top score below the null threshold; nothing here matches
          'answer'  -- exactly one passage contains all the terms
          'set'     -- m > 1 passages are INDISTINGUISHABLE; ranking them carries no information
        """
        if self.threshold is None:
            self.calibrate()
        s = self.scores(qterms) if scores is None else np.asarray(scores, dtype=float)
        if s.size == 0:
            return dict(mode="abstain", answer=None, set=[], ambiguity=0,
                        top_score=0.0, margin=0.0, threshold=self.threshold,
                        ceiling=0.0, reason="empty corpus")
        order = np.argsort(-s)
        top_score = float(s[order[0]])
        margin = float(s[order[0]] - s[order[1]]) if self.N > 1 else float(s[order[0]])
        cset = self.containment(qterms)
        m = len(cset)
        if top_score < self.threshold:
            mode, answer, out = "abstain", None, []
            reason = "top score %.3f below null threshold %.3f" % (top_score, self.threshold)
        elif m <= 1:
            top = [int(i) for i in order[:rerank_k]]
            if rerank and len(top) > 1:
                terms = self.terms(qterms)
                top = sorted(top, key=lambda d: proximity_key(self.docs[d], terms), reverse=True)
            mode, answer = "answer", int(top[0])
            out = [int(top[0])]
            reason = "containment set has %d passage(s); the ranking is informative" % m
        else:
            mode, answer = "set", None
            ranked = [i for i in order if i in cset][:max_set]
            out = ranked
            reason = ("%d passages contain every query term and are indistinguishable to a "
                      "term-based scorer; a ranking among them carries no information" % m)
        return dict(mode=mode, answer=answer, set=out, ambiguity=m,
                    top_score=top_score, margin=margin, threshold=self.threshold,
                    ceiling=(1.0 / m if m else 0.0), reason=reason)

    def fingerprint(self):
        """Content hash of the corpus -- hashlib, never hash(), so it is stable across runs."""
        h = hashlib.sha256()
        for d in self.docs:
            h.update(("\x1f".join(d) + "\x1e").encode("utf-8"))
        return h.hexdigest()[:16]


def _selftest():
    rng = np.random.default_rng(0)
    vocab = ["w%03d" % i for i in range(400)]

    # A planted corpus with all three conditions in it ON PURPOSE, each with its own truth.
    docs = []
    for i in range(120):                                   # background
        docs.append([vocab[j] for j in rng.choice(300, 25, replace=False)])
    unique = ["zeta", "kappa", "lambda", "omega", "theta", "sigma"]
    docs.append(unique + [vocab[j] for j in rng.choice(300, 20, replace=False)])   # 120: unique
    shared = ["alpha", "beta", "gamma", "delta", "epsilon", "iota"]
    docs.append(shared + [vocab[j] for j in rng.choice(300, 20, replace=False)])   # 121
    docs.append(shared + [vocab[j] for j in rng.choice(300, 20, replace=False)])   # 122 twin

    pol = RetrievalPolicy(docs).calibrate(n=120, seed=1)

    # 1. WELL-POSED: terms unique to one passage -> a single answer, and the RIGHT one.
    v = pol.verdict(unique[:4])
    assert v["mode"] == "answer", v
    assert v["answer"] == 120, v
    assert v["ambiguity"] == 1, v

    # 2. AMBIGUOUS: terms shared by two passages -> the SET, containing both, never a guess.
    v = pol.verdict(shared[:4])
    assert v["mode"] == "set", v
    assert v["ambiguity"] == 2, v
    assert set(v["set"]) == {121, 122}, v
    assert abs(v["ceiling"] - 0.5) < 1e-12, v

    # 3b. MAX-COVERAGE containment, pinned on a LONG query: the exact-AND form returned nothing
    #     for 100% of 148-term prose queries and the policy silently became "always answer".
    # A long query with terms no document holds must still find the twins -- they match the SAME
    # subset (the shared terms) and nothing else does. And a query whose terms are spread so that
    # every document matches a DIFFERENT subset must return a singleton, not a false tie: prose
    # queries are not ambiguous, and a gate that hedges on them is worse than one that answers.
    _long_q = shared[:4] + ["nowhere%d" % i for i in range(30)]
    _cs = pol.containment(_long_q)
    assert _cs == {121, 122}, "signature must find the twins, not everything: %s" % sorted(_cs)
    _spread = unique[:3] + shared[:1]
    assert len(pol.containment(_spread)) == 1, (
        "a query no two documents match identically must NOT report ambiguity: %s"
        % sorted(pol.containment(_spread)))

    # 3. NO MATCH: terms absent from the corpus -> abstain.
    v = pol.verdict(["qqqq", "wwww", "eeee", "rrrr"])
    assert v["mode"] == "abstain", v

    # 4. KEPT NEGATIVE, PINNED: a near-duplicate scores HIGH, so the CONFIDENCE gate does NOT
    #    catch ambiguity. Anyone tempted to solve ambiguity with a threshold should read this
    #    assertion: the ambiguous query is comfortably ABOVE the abstain threshold and is caught
    #    only by the containment count. Confidence and ambiguity are different questions.
    v = pol.verdict(shared[:4])
    assert v["top_score"] > v["threshold"], "the ambiguous query should NOT look uncertain"

    # 5. determinism -- hashlib fingerprint, stable across processes
    assert pol.fingerprint() == RetrievalPolicy(docs).fingerprint()

    # 6. DEGENERACY GUARD, pinned: a corpus too small for a null must DISABLE the gate, not
    #    abstain on everything. This is the exact failure a 3-document smoke test exposed.
    tiny = RetrievalPolicy([["alpha", "beta", "solo"], ["alpha", "beta", "twin"],
                            ["zeta", "kappa", "unique"]]).calibrate(n=40, seed=2)
    assert tiny.degenerate, "a 3-passage corpus must be flagged degenerate"
    assert tiny.threshold == 0.0, "a degenerate null must disable the gate"
    assert tiny.verdict(["zeta", "kappa"])["mode"] == "answer", tiny.verdict(["zeta", "kappa"])
    assert tiny.verdict(["alpha", "beta"])["mode"] == "set", tiny.verdict(["alpha", "beta"])
    assert not pol.degenerate, "the 123-passage corpus should NOT be degenerate"

    # 7. DOUBLE-TOKENIZATION DETECTOR, pinned. The terms this policy holds are already
    #    normalised, so re-normalising a slice of them must change SOMETHING -- that non-zero
    #    fraction is exactly the signal a caller needs before handing tokens back in.
    assert 0.0 <= pol.double_tokenization_risk <= 1.0
    from holographic.semantic_router.holographic_bm25 import tokenize as _tk
    assert _tk("settings") == ["setting"] and _tk("setting") == ["sett"], \
        "the non-idempotence this detector exists for has changed -- re-check the detector"
    pre = RetrievalPolicy(pol.docs, pretokenized=True)
    assert pre.docs == pol.docs, "pretokenized=True must not re-normalise"

    # 8. RERANK, pinned: default OFF must leave the verdict untouched, and ON must only ever
    #    REORDER what BM25 already returned -- never introduce a document from outside the top-k,
    #    because recall is the scorer's job and reordering is the reranker's.
    # PINNED: raw coverage is a length prior. A long document stuffed with the query's terms must
    # NOT outrank a short exact match once normalisation is on, and must outrank it with it off --
    # so the pin fails if the normalisation is ever quietly removed.
    _short = ["alpha", "beta", "gamma"]
    _long = _short + ["pad%d" % i for i in range(400)]
    _q = ["alpha", "beta", "gamma"]
    assert proximity_key(_long, _q, length_norm=False) >= proximity_key(_short, _q, length_norm=False), \
        "raw coverage should favour the longer document -- if not, the trap has changed"
    assert proximity_key(_short, _q) > proximity_key(_long, _q), \
        "length-normalised coverage must favour the SHORT exact match"

    v_off = pol.verdict(unique[:4])
    v_on = pol.verdict(unique[:4], rerank=True)
    assert v_off["mode"] == v_on["mode"], "rerank must not change the SHAPE of a verdict"
    base_top = list(np.argsort(-pol.scores(unique[:4]))[:10])
    assert v_on["answer"] in base_top, "rerank introduced a document from outside the top-k"

    # 9. the null is a distribution, not a magic number: a stricter percentile abstains more
    strict = RetrievalPolicy(docs).calibrate(n=120, percentile=99.0, seed=1)
    assert strict.threshold >= pol.threshold, "a higher percentile must not lower the threshold"

    print("holographic_retrievalpolicy self-test passed (well-posed -> answer 120; ambiguous -> "
          "set {121,122} with ceiling 0.500; nonsense -> abstain; ambiguous query scores %.2f "
          "ABOVE the %.2f threshold, so confidence does NOT detect ambiguity; fingerprint stable)"
          % (pol.verdict(shared[:4])["top_score"], pol.threshold))


if __name__ == "__main__":
    _selftest()

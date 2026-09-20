"""
holographic_systemone.py -- typed decisions with HONEST probabilities (the native System One door).

WHY THIS MODULE EXISTS
----------------------
TypeSafe's Jev (Sep 2026) popularized a contract: send program state plus a fixed set of TYPED
questions (choice / score / yes-no), get back typed values with a calibrated probability each --
no generated text, nothing to parse, invalid output impossible by construction. That contract is
substrate-native here: leCore already owns the scorer (build_prototypes / match_prototype), the
abstention gate (decide_or_abstain), and the doctrine that an uncalibrated probability is a lie.
This module adds ONLY what the audit found missing (2026-09-18, sweep 171): the typed schema, a
fitted calibrator, the one-pass multi-question decide, and the batch map. Scoring is DELEGATED to
holographic_relations -- this file is the contract layer, not a new classifier.

WHERE WE ARE DELIBERATELY DIFFERENT FROM JEV (kept on record):
  * No probability until calibrated. Jev always returns a confidence; we return p=None with
    calibrated=False until you feed labeled examples. A softmax over raw cosines LOOKS like a
    probability and isn't one -- refusing to dress it up is the honest half of the contract.
  * Abstention is first-class. A tie abstains (value=None, why=...) instead of forcing a pick.
  * Every choice answer carries its ranked evidence and its basis (examples vs labels-only),
    which is the rationale Jev cannot give.
Honest scope: accuracy comes from YOUR examples on YOUR domain. This is a few-shot prototype
classifier under a typed contract -- it does not import a frontier model's judgment, and on open
text it will lose to one. Measure on your task (see tools/systemone_bench.py) before trusting it.

Schema (validated up front -- the validation IS the no-type-error guarantee):
  questions = {name: spec}
    {"type": "choice", "options": [...>=2 unique strings...], "examples": {option: [texts]}?}
    {"type": "score",  "min": lo, "max": hi (lo<hi), "anchors": [(text, value), ...]?}
    {"type": "noul",   "examples": {"yes": [...], "no": [...]}?}   # yes/no probability
Unknown types and unknown keys are REJECTED, not ignored: a silently-dropped key is how a caller
ships a schema that never did what they thought.
"""
import re
import hashlib
import json

import numpy as np

from holographic.misc.holographic_relations import (
    build_prototypes,
    decide_or_abstain,
    match_prototype,
)

# One tie rule everywhere (ISA-1): ranked lists sort by (-score, name) so an exact tie has a
# stated order; decide_or_abstain then abstains on the tie anyway (gap 0 < margin).
_ALLOWED_KEYS = {
    "choice": {"type", "options", "examples"},
    "score": {"type", "min", "max", "anchors"},
    "noul": {"type", "examples"},
}


class SchemaError(ValueError):
    """A question schema that could produce an untyped or ambiguous answer. Raised at validate
    time so a bad schema fails BEFORE any state is scored -- never as a malformed result later."""


def validate_questions(questions):
    """Validate {name: spec} strictly; return a normalized deep copy. Raises SchemaError with the
    offending question named. Strictness is the guarantee: everything a later decide() can emit is
    enumerable from what passes here."""
    if not isinstance(questions, dict) or not questions:
        raise SchemaError("questions must be a non-empty dict of {name: spec}")
    out = {}
    for name, spec in questions.items():
        if not isinstance(name, str) or not name:
            raise SchemaError("question names must be non-empty strings, got %r" % (name,))
        if not isinstance(spec, dict) or "type" not in spec:
            raise SchemaError("%s: spec must be a dict with a 'type'" % name)
        qt = spec["type"]
        if qt not in _ALLOWED_KEYS:
            raise SchemaError("%s: unknown type %r (choice|score|noul)" % (name, qt))
        extra = set(spec) - _ALLOWED_KEYS[qt]
        if extra:
            raise SchemaError("%s: unknown key(s) %s for type %s" % (name, sorted(extra), qt))
        if qt == "choice":
            opts = spec.get("options")
            if (not isinstance(opts, (list, tuple)) or len(opts) < 2
                    or len(set(opts)) != len(opts)
                    or not all(isinstance(o, str) and o for o in opts)):
                raise SchemaError("%s: options must be >=2 unique non-empty strings" % name)
            ex = spec.get("examples", {})
            if not isinstance(ex, dict) or set(ex) - set(opts):
                raise SchemaError("%s: examples keys must be a subset of options" % name)
            for k, v in ex.items():
                if not isinstance(v, (list, tuple)) or not v:
                    raise SchemaError("%s: examples[%r] must be a non-empty list" % (name, k))
            out[name] = {"type": "choice", "options": list(opts),
                         "examples": {k: list(v) for k, v in ex.items()}}
        elif qt == "noul":
            ex = spec.get("examples", {})
            if not isinstance(ex, dict) or set(ex) - {"yes", "no"}:
                raise SchemaError("%s: noul examples keys must be within {yes, no}" % name)
            # WHY reuse: noul IS a two-option choice -- one code path, one tie rule, one calibrator.
            out[name] = {"type": "noul", "options": ["yes", "no"],
                         "examples": {k: list(v) for k, v in ex.items()}}
        else:  # score
            try:
                lo, hi = float(spec["min"]), float(spec["max"])
            except (KeyError, TypeError, ValueError):
                raise SchemaError("%s: score needs numeric min and max" % name)
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise SchemaError("%s: score needs finite min < max" % name)
            anchors = spec.get("anchors", [])
            norm = []
            for a in anchors:
                if (not isinstance(a, (list, tuple)) or len(a) != 2
                        or not isinstance(a[0], str) or not a[0]):
                    raise SchemaError("%s: anchors are (text, value) pairs" % name)
                v = float(a[1])
                if not np.isfinite(v):
                    raise SchemaError("%s: anchor value must be finite" % name)
                norm.append((a[0], v))
            out[name] = {"type": "score", "min": lo, "max": hi, "anchors": norm}
    return out


def _pav_fit(scores, correct):
    """Pool-Adjacent-Violators isotonic fit: (score, 0/1-correct) pairs -> nondecreasing P(correct).
    Returns (xs, ys) for np.interp. WHY isotonic and not Platt: no logistic assumption, exact on
    small data, numpy-only, and monotone by construction so a bigger margin never claims LESS
    confidence. WHY the clip: finite data cannot support p=0 or p=1 -- we clip to [0.5/n, 1-0.5/n],
    the resolution n labeled examples can actually testify to (a calibrator that says 1.0 is the
    overconfidence this module exists to refuse)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    if s.shape != y.shape or s.ndim != 1 or s.size < 4:
        raise ValueError("calibration needs >=4 (score, correct) pairs")
    order = np.argsort(s, kind="stable")   # stable: equal scores keep input order (deterministic)
    s, y = s[order], y[order]
    # Blocks of [sum, count]; merge while the mean sequence decreases anywhere.
    val = list(y)
    wgt = [1.0] * len(val)
    i = 0
    while i < len(val) - 1:
        if val[i] / wgt[i] > val[i + 1] / wgt[i + 1] + 1e-15:
            val[i] += val.pop(i + 1)
            wgt[i] += wgt.pop(i + 1)
            if i > 0:
                i -= 1              # a merge can create a new violation to the LEFT; step back
        else:
            i += 1
    fitted = np.repeat([v / w for v, w in zip(val, wgt)],
                       [int(w) for w in wgt])
    n = float(len(s))
    fitted = np.clip(fitted, 0.5 / n, 1.0 - 0.5 / n)
    return s, fitted


class IsotonicCalibrator:
    """Monotone map: decision margin -> P(decision correct), fitted by PAV on labeled outcomes.
    predict() interpolates linearly between fitted points (still monotone) and clamps at the ends
    (np.interp default) -- outside the seen range we return the nearest supported value rather
    than extrapolate a certainty nobody measured."""

    def __init__(self, scores, correct):
        self.xs, self.ys = _pav_fit(scores, correct)
        self.n = int(len(self.xs))

    def predict(self, score):
        return float(np.interp(float(score), self.xs, self.ys))


def _nb_feats(text, bigrams=False):
    """Word counts for the nb scorer (lowercase [a-z0-9]+ runs; optional adjacent-word bigrams).
    Deliberately the simplest tokenizer that measured well: no stemming, no stop list -- naive
    Bayes with Laplace smoothing is robust to both, and every extra rule is a thing to drift."""
    import re as _re
    w = _re.findall(r"[a-z0-9]+", (text or "").lower())
    f = {}
    for x in w:
        f[x] = f.get(x, 0.0) + 1.0
    if bigrams:
        for a, b in zip(w, w[1:]):
            k = a + "_" + b
            f[k] = f.get(k, 0.0) + 1.0
    return f


def _nb_transform(feats, idf, n_docs):
    """Rennie's three fixes to multinomial NB's poor assumptions, as one closed-form map on a count
    dict: (1) log(1+tf) damps the burstiness of repeated words; (2) IDF from the examples down-weights
    words every option shares; (3) length normalisation stops a long state out-voting a short one.
    idf=None means untransformed (nb_transform=False), which returns the counts unchanged."""
    if idf is None:
        return feats
    g = {tok: float(np.log1p(c)) * idf.get(tok, float(np.log(max(n_docs, 1)))) for tok, c in feats.items()}
    norm = float(np.sqrt(sum(v * v for v in g.values()))) or 1.0
    return {tok: v / norm for tok, v in g.items()}



# ---- the question lint (sweep 176, backlog D1-D3) ------------------------------------------------
# WHY: both tool experiments (doc 04) failed only on INPUT SHAPE -- a paragraph carrying five observations
# abstained on everything; parts described by function routed 1-3/8 and by geometry 8/8; the option
# with the shortest examples owned the smoothing floor; a contrastive sentence ("works LIKE bass reflex
# ports but WITHOUT a port tube") named the thing it was not. None of that is scorer error. Checking
# the question and the state BEFORE asking is cheap, deterministic, and catches every one of them.
# ", next " was missing (found by the E1 re-test: 25% of two-alias requests never split); "next" is
# accepted with a leading comma or space and an optional trailing comma.
_CLAUSE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\s*,?\s+(?:and then|after that|then|once that is done|next,?)\s+", re.I)
# WORD-BOUNDED: the first cut matched "but" inside "Buttons on the top strip" (measured false positive).
_CONTRAST = re.compile(r"\b(?:but|without|unlike|rather than|except|instead of|not|no|never)\b", re.I)


def clauses(text):
    """Split a request into single-observation clauses: sentence ends, semicolons, and sequencing
    connectives (and then / after that / then / next). Model-free and deterministic. A typed decision
    wants ONE observation per state (doc 04): five in one paragraph abstain at a uniform posterior."""
    parts = []
    for p in _CLAUSE_SPLIT.split(text or ""):
        p = re.sub(r"^(?:and then|and|then|next|after that)\b[ ,]*", "", (p or "").strip(" ,"), flags=re.I).strip(" ,")
        if p:                                    # a stranded connective ("and") is not a clause
            parts.append(p)
    return parts if parts else ([text.strip()] if text and text.strip() else [])


def is_contrastive(text):
    """True when a clause carries a contrast or negation marker -- the structure a bag of words cannot
    read (the blueprint miss). Cheapest honest version: a word list; measured on the 8 research
    sentences it flags the one contrastive sentence and no other. Escalate these to the model end."""
    return bool(_CONTRAST.search(text or ""))


def schema_lint(questions, states=None, scorer=None):
    """Lint a typed-decision schema (and optionally its states) BEFORE asking. Returns
    {ok, findings:[{level, where, what}], recommended_scorer, k_min, budgets}. Findings:
      error   an option with no example and no usable label; a question with < 2 options.
      warn    example TOKEN budgets imbalanced > 2x across options (the shortest option owns the Laplace
              floor and captures out-of-vocabulary states -- measured 1-3/8 tool decisions); fewer than 3
              examples for an option.
      note    recommended scorer from the MEASURED regime table (doc 01): under ~10 examples per option
              the char-n-gram prototype and transformed NB are within spread, from ~20 up transformed NB
              wins clearly -- chosen from k, never by leave-one-out (picked wrong at every budget);
              states with > 1 clause (split them: clauses()); contrastive states (escalate).
    Never changes the schema -- it reports; the caller decides."""
    findings = []
    budgets = {}
    k_min = None
    for name, spec in (questions or {}).items():
        opts = list(spec.get("options") or [])
        if spec.get("type") in ("choice", "noul") and len(opts) < 2:
            findings.append({"level": "error", "where": name, "what": "fewer than 2 options"})
        ex = spec.get("examples") or {}
        toks = {}
        for o in opts:
            texts = ex.get(o) or []
            n_ex = len(texts)
            n_tok = sum(len(_nb_feats(t)) for t in texts) or len(_nb_feats(o))
            toks[o] = {"examples": n_ex, "tokens": n_tok}
            if n_ex == 0 and not _nb_feats(o):
                findings.append({"level": "error", "where": "%s.%s" % (name, o), "what": "no examples and no usable label"})
            elif n_ex < 3:
                findings.append({"level": "warn", "where": "%s.%s" % (name, o), "what": "only %d example(s); noisy below 3" % n_ex})
            k_min = n_ex if k_min is None else min(k_min, n_ex)
        budgets[name] = toks
        if toks:
            lo = min(v["tokens"] for v in toks.values()); hi = max(v["tokens"] for v in toks.values())
            if lo > 0 and hi / lo > 2.0:
                short = min(toks, key=lambda o: toks[o]["tokens"])
                findings.append({"level": "warn", "where": name,
                                 "what": "example token budgets imbalanced %.1fx (%s is shortest and would own the smoothing floor)" % (hi / lo, short)})
    rec = "prototype" if (k_min is None or k_min < 10) else "nb"
    if scorer and scorer != rec:
        findings.append({"level": "note", "where": "scorer", "what": "%s requested; regime table (k_min=%s) recommends %s" % (scorer, k_min, rec)})
    for i, s in enumerate(states or []):
        cl = clauses(s)
        if len(cl) > 1:
            findings.append({"level": "note", "where": "state[%d]" % i, "what": "%d clauses -- split into one state each" % len(cl), "clauses": cl})
        if is_contrastive(s):
            findings.append({"level": "note", "where": "state[%d]" % i, "what": "contrastive/negated -- escalate to the model end"})
    return {"ok": not any(f["level"] == "error" for f in findings), "findings": findings,
            "recommended_scorer": rec, "k_min": k_min, "budgets": budgets}


class SystemOne:
    """Fit once, decide many: the typed-decision contract over any text encoder.

    encode: text -> vector (e.g. lambda t: mind.perceive(t, "text")). All scoring is cosine in
    that space; determinism is inherited from the encoder (seeded in the mind). fit() precomputes
    per-question prototype MATRICES so decide() is: encode the state ONCE, then one mat-vec per
    question -- the whole multi-question answer is a single parallel pass, which is the property
    Jev advertises and a vector substrate gets for free."""

    def __init__(self, encode, margin=0.1, score_floor=0.15, score_tau=0.05, score_top_k=5,
                 min_support=None, scorer="prototype", nb_bigrams=False, nb_transform=True):
        self.encode = encode
        # scorer (sweep 174): "prototype" = the sweep-171 cosine-to-mean-bundle path, unchanged.
        # "nb" = multinomial naive Bayes over word counts of the SAME examples -- a count table,
        # fit exactly the way a prototype is (one pass over the examples, no gradient, no
        # autodiff, stdlib+numpy, deterministic). MEASURED (3 seeds, 400-row evals) against the
        # prototype path with identical examples: AG News k=32 0.697 vs 0.630, k=128 0.779 vs
        # 0.722, k=300 0.821 vs 0.723; SST-2 k=600 0.703 vs 0.669 (0.725 with nb_bigrams). The
        # substrate could NOT host it: NB weights as a hypervector prototype scored 0.646 at
        # d=2048 (worse than the plain centroid) and 0.742 at d=8192 -- JL cross-term noise over a
        # vocabulary far larger than the dimension swamps per-token log-odds. Kept negative.
        # Default stays "prototype": additive, off, and every existing decision is untouched.
        if scorer not in ("prototype", "nb"):
            raise SchemaError("scorer must be 'prototype' or 'nb', got %r" % (scorer,))
        self.scorer = scorer
        self.nb_bigrams = bool(nb_bigrams)
        # nb_transform (sweep 175): Rennie et al. 2003 -- log(1+tf), times IDF (from the examples),
        # length-normalised -- still closed-form counts. MEASURED best on all three real tasks at
        # every budget: AG News k=300 0.843 vs 0.821 plain; SST-2 k=600 0.709 vs 0.703; Banking77
        # (77 intents) k=5 0.581 vs 0.387 plain and vs 0.549 for the prototype path -- the sparse
        # regime where plain NB loses to character n-grams is exactly where the transform fixes
        # it. False reproduces the sweep-174 untransformed nb bit-for-bit.
        self.nb_transform = bool(nb_transform)
        self._nb = {}         # qname -> {"counts": {label: {tok: n}}, "tot": {label: n}, "vocab": set}
        self._conformal = {}  # qname -> {"q": gap quantile, "n", "alpha"} after calibrate_conformal (H1)
        self.margin = float(margin)
        self.score_floor = float(score_floor)     # below this best-anchor cosine, a score abstains
        self.score_tau = float(score_tau)         # softmax temperature over anchor similarities
        self.score_top_k = int(score_top_k)
        # min_support (sweep 172, the Milanfar gap): with None, a state unlike ALL options can
        # still luck into a confident pick because one bad match beats the other bad matches.
        # Set it and choice/noul abstain off-manifold exactly as score questions always did.
        # None keeps the sweep-171 behaviour (backward compatible, default-off).
        self.min_support = None if min_support is None else float(min_support)
        self.questions = None
        self._mat = {}        # qname -> (matrix rows=options/anchors, unit) for the one-pass score
        self._meta = {}       # qname -> per-question fitted extras (labels, values, basis)
        self._calib = {}      # qname -> IsotonicCalibrator (absent until calibrate())
        self._acc = {}        # qname -> RAW prototype accumulators (observe() updates these)
        self._support = {}    # qname -> winning-cosine stream (label-FREE drift channel)
        self._outcomes = {}   # qname -> [(margin_gap, correct 0/1)] (label-lagged channel)
        self.schema_sha256 = None

    # ---------------- fitting ----------------
    def _unit(self, text):
        v = np.asarray(self.encode(text), dtype=np.float64).reshape(-1)
        return v / (np.linalg.norm(v) + 1e-12)

    def _nb_posterior(self, name, state):
        """Multinomial naive Bayes with Laplace (alpha=1) smoothing, returned as a POSTERIOR over the
        options in schema order (a softmax over the log-scores), so it drops into the same ranked /
        margin / calibration path as a cosine. Priors are uniform: the examples are the caller's
        evidence about the OPTIONS, not about their base rates, and a prior read off a few-shot
        example count would be a made-up number."""
        tab = self._nb[name]
        labels = self._meta[name]["labels"]
        V = float(len(tab["vocab"]))
        feats = _nb_transform(_nb_feats(state, self.nb_bigrams), tab["idf"], tab["n_docs"])
        logs = np.zeros(len(labels))
        for i, o in enumerate(labels):
            cnt, tot = tab["counts"][o], tab["tot"][o]
            logs[i] = sum(c * np.log((cnt.get(tok, 0.0) + 1.0) / (tot + V))
                          for tok, c in feats.items())
        logs -= logs.max()                 # stable softmax
        post = np.exp(logs)
        return post / post.sum()

    def fit(self, questions):
        """Validate the schema and precompute prototype/anchor matrices. Options WITH examples get
        a real prototype (build_prototypes: mean bundle); options WITHOUT examples fall back to
        encoding the option LABEL itself -- allowed, but flagged basis='labels-only' because a
        label is one word of evidence and the caller deserves to know that's all we have."""
        self.questions = validate_questions(questions)
        # Content hash of the schema: provenance for receipts, and proof two minds fit the same
        # contract. hashlib, never hash() -- the determinism rule.
        self.schema_sha256 = hashlib.sha256(
            json.dumps(self.questions, sort_keys=True).encode("utf-8")).hexdigest()
        for name, spec in self.questions.items():
            if spec["type"] in ("choice", "noul"):
                ex = spec["examples"]
                with_ex = {o: ex[o] for o in spec["options"] if o in ex}
                protos = build_prototypes(self.encode, with_ex) if with_ex else {}
                for o in spec["options"]:
                    if o not in protos:
                        protos[o] = self._unit(o)
                labels = list(spec["options"])   # fixed order: the schema order is the tie order
                self._mat[name] = np.stack([protos[o] for o in labels])
                # Accumulators beside the unit rows, so observe() can apply the AdaptHD
                # miss-update (pull correct toward the state, push wrong away -- the same
                # gradient-free rule holographic_classifier retrains with) and renormalize.
                # Direction is build_prototypes' exactly (mean of raw encodings, ASSERTED so a
                # drift between the faculties fails loudly at fit); the norm starts at UNIT so
                # lr has a stated meaning: the state is unit, so lr=1 makes one miss weigh about
                # as much as the entire prior prototype (raw-mean norms vary with the encoder --
                # measured ~40x the state here -- which silently froze learning; kept negative).
                acc = []
                for o in labels:
                    if o in with_ex:
                        V = np.stack([np.asarray(self.encode(x), dtype=np.float64).reshape(-1)
                                      for x in with_ex[o]])
                        mu = V.mean(0)
                        u = mu / (np.linalg.norm(mu) + 1e-12)
                        assert float(np.dot(u, protos[o])) > 1 - 1e-9, \
                            "accumulator formula diverged from build_prototypes"
                        acc.append(u)
                    else:
                        raw = np.asarray(self.encode(o), dtype=np.float64).reshape(-1)
                        acc.append(raw / (np.linalg.norm(raw) + 1e-12))
                self._acc[name] = np.stack(acc)
                basis = ("examples" if len(with_ex) == len(labels)
                         else "labels-only" if not with_ex else "mixed")
                self._meta[name] = {"labels": labels, "basis": basis}
                if self.scorer == "nb":
                    # One pass over the same examples: per-option token counts. An option with no
                    # examples is counted from its label text, matching the prototype fallback.
                    texts = [(o, t) for o in labels for t in (with_ex.get(o) or [o])]
                    raw = [(o, _nb_feats(t, self.nb_bigrams)) for o, t in texts]
                    idf = None
                    if self.nb_transform:
                        # IDF over the caller's examples, FIXED at fit: observe() adds counts but
                        # does not move the weights, so a decision is a function of the fitted
                        # table plus what was counted -- reproducible, no drifting denominator.
                        df = {}
                        for _, f in raw:
                            for tok in f:
                                df[tok] = df.get(tok, 0) + 1
                        n_docs = len(raw)
                        idf = {tok: float(np.log(n_docs / d)) for tok, d in df.items()}
                    counts = {}
                    tot = {}
                    vocab = set()
                    for o in labels:
                        counts[o] = {}
                        tot[o] = 0.0
                    for o, f in raw:
                        for tok, c in _nb_transform(f, idf, len(raw)).items():
                            counts[o][tok] = counts[o].get(tok, 0.0) + c
                            tot[o] += c
                            vocab.add(tok)
                    self._nb[name] = {"counts": counts, "tot": tot, "vocab": vocab,
                                      "idf": idf, "n_docs": len(raw)}
            else:
                anchors = spec["anchors"]
                if anchors:
                    self._mat[name] = np.stack([self._unit(t) for t, _ in anchors])
                self._meta[name] = {"values": np.array([v for _, v in anchors], dtype=np.float64)}
        return {"questions": len(self.questions), "schema_sha256": self.schema_sha256}

    # ---------------- deciding ----------------
    def _decide_encoded(self, q, state=None):
        """Answer every question for one already-encoded unit state vector q. `state` is the raw
        text, needed only by the nb scorer (it scores counts, not the encoding)."""
        out = {}
        for name, spec in self.questions.items():
            if spec["type"] in ("choice", "noul"):
                if self.scorer == "nb" and name in self._nb:
                    # Posterior over options, so `ranked` scores live on [0,1] like a similarity
                    # and the SAME margin / gap / calibration machinery applies unchanged.
                    sims = self._nb_posterior(name, state if state is not None else "")
                else:
                    sims = self._mat[name] @ q
                ranked = sorted(zip(self._meta[name]["labels"], (float(s) for s in sims)),
                                key=lambda kv: (-kv[1], kv[0]))
                winner, top, confident = decide_or_abstain(ranked, margin=self.margin,
                                                           min_score=self.min_support)
                gap = ranked[0][1] - ranked[1][1]
                # Label-free drift channel (Siemion seat): record the winning cosine for EVERY
                # decision; a falling support quantile flags covariate shift before labels arrive.
                self._support.setdefault(name, []).append(float(ranked[0][1]))
                cal = self._calib.get(name)
                p = cal.predict(gap) if cal is not None else None
                ans = {"type": spec["type"], "confident": bool(confident),
                       "abstained": not confident, "ranked": ranked, "margin_gap": float(gap),
                       "p": p, "calibrated": cal is not None,
                       "basis": self._meta[name]["basis"]}
                if spec["type"] == "noul":
                    # value is a typed bool; p (when calibrated) is P(this yes/no call is correct).
                    ans["value"] = (winner == "yes") if confident else None
                else:
                    ans["value"] = winner if confident else None
                ans["support"] = float(ranked[0][1])
                if not confident:
                    if self.min_support is not None and ranked[0][1] < self.min_support:
                        ans["why"] = ("support %.3f < min_support %.3f (state is off every "
                                      "option's manifold)" % (ranked[0][1], self.min_support))
                    else:
                        ans["why"] = "top-1 does not beat top-2 by margin %.3g" % self.margin
                cq = getattr(self, "_conformal", {}).get(name)
                if cq is not None:
                    # H1: the conformal answer SET -- every option within the calibrated gap of the top score.
                    # Guaranteed to hold the truth with probability >= 1 - alpha (split conformal); a set of
                    # size 1 is a confident answer with a guarantee, a larger set says exactly who to ask.
                    top = float(ranked[0][1])
                    ans["set"] = [n for n, s in ranked if top - float(s) <= cq["q"]]
                    ans["set_alpha"] = cq["alpha"]
                out[name] = ans
            else:
                lo, hi = spec["min"], spec["max"]
                vals = self._meta[name]["values"]
                if vals.size < 2:
                    out[name] = {"type": "score", "value": None, "abstained": True,
                                 "n_anchors": int(vals.size),
                                 "why": "needs >=2 (text, value) anchors to interpolate"}
                    continue
                sims = self._mat[name] @ q
                k = min(self.score_top_k, sims.size)
                # Deterministic top-k: argsort is stable, ties resolved by anchor order.
                idx = np.argsort(-sims, kind="stable")[:k]
                top = float(sims[idx[0]])
                if top < self.score_floor:
                    out[name] = {"type": "score", "value": None, "abstained": True,
                                 "support": top, "n_anchors": int(vals.size),
                                 "why": "best anchor cosine %.3f < floor %.3f (state is off the "
                                        "anchor manifold; guessing a number would be worse than "
                                        "none)" % (top, self.score_floor)}
                    continue
                # Kernel regression over the top-k anchors: softmax weights on similarity. tau is
                # a resolution knob, not magic -- small tau ~= nearest anchor, large ~= mean.
                w = np.exp((sims[idx] - top) / max(self.score_tau, 1e-9))
                w = w / w.sum()
                value = float(np.clip(np.dot(w, vals[idx]), lo, hi))
                spread = float(np.sqrt(np.dot(w, (vals[idx] - value) ** 2)))
                out[name] = {"type": "score", "value": value, "abstained": False,
                             "support": top, "spread": spread, "n_anchors": int(vals.size)}
        return out

    def decide(self, state):
        """One state, all questions, one encoding pass. Returns {qname: typed answer dict}; every
        value is either from the schema or None (abstained) -- nothing else is constructible."""
        if self.questions is None:
            raise RuntimeError("call fit(questions) before decide()")
        return self._decide_encoded(self._unit(state), state=state)

    def decide_map(self, states):
        """Batch: encode each state, then answer with the SAME precomputed matrices. Encoding is
        the O(N) cost; per-question scoring is one (N x d) @ (d x m) matmul's worth of work spread
        across rows. Row i of the result is bit-identical to decide(states[i]) -- pinned by the
        selftest, because a batch path that drifts from the single path is a silent fork."""
        if self.questions is None:
            raise RuntimeError("call fit(questions) before decide_map()")
        return [self._decide_encoded(self._unit(s), state=s) for s in states]

    # ---------------- the loop: observe, escalate, drift (sweep 172) ----------------
    def observe(self, state, truths, lr=1.0, recalibrate_window=256):
        """PREQUENTIAL online learning: decide FIRST (that decision is the honest test), then
        update from the outcome. On a miss, the AdaptHD rule holographic_classifier already
        retrains with: pull the correct option's accumulator toward the state, push the wrongly
        picked one away, renormalize those two unit rows. lr=0 records outcomes without touching
        prototypes (the frozen baseline for measuring what learning adds). Every outcome also
        feeds a rolling isotonic recalibration (last `recalibrate_window` outcomes, >=8 to fit)
        so the probability tracks the CURRENT prototypes, not the ones you fitted last month --
        the thing a frozen hosted decision model cannot do. Returns {q: {was_correct, updated}}.
        Truths for score questions are ignored here (their update story is anchor management,
        recorded as future work, not silently faked)."""
        q = self._unit(state)
        ans = self._decide_encoded(q, state=state)
        out = {}
        for name, truth in truths.items():
            spec = self.questions.get(name)
            if spec is None or spec["type"] == "score":
                continue
            want = ("yes" if truth else "no") if spec["type"] == "noul" else truth
            labels = self._meta[name]["labels"]
            if want not in labels:
                raise SchemaError("%s: observed truth %r is not an option" % (name, truth))
            pred = ans[name]["ranked"][0][0]
            correct = pred == want
            self._outcomes.setdefault(name, []).append(
                (ans[name]["margin_gap"], 1.0 if correct else 0.0))
            updated = False
            if lr > 0 and self.scorer == "nb" and name in self._nb:
                # NB learns by COUNTING, on EVERY labeled observation -- a correct decision is
                # still evidence (the ladder measured every added example helping: AG News 0.697
                # -> 0.821 from k=32 to 300). AdaptHD below moves only on a miss because pulling a
                # prototype toward a state it already matched over-fits it; a count has no such
                # failure mode. Nothing is subtracted from the wrong option: a count cannot go
                # negative, and the push-away has no honest analogue here.
                tab = self._nb[name]
                for tok, c in _nb_transform(_nb_feats(state, self.nb_bigrams), tab["idf"],
                                            tab["n_docs"]).items():
                    tab["counts"][want][tok] = tab["counts"][want].get(tok, 0.0) + lr * c
                    tab["tot"][want] += lr * c
                    tab["vocab"].add(tok)
                updated = True
            elif not correct and lr > 0:
                hi, wi = labels.index(want), labels.index(pred)
                self._acc[name][hi] += lr * q
                self._acc[name][wi] -= lr * q
                for i in (hi, wi):   # renormalize ONLY the touched rows: cheap, deterministic
                    v = self._acc[name][i]
                    self._mat[name][i] = v / (np.linalg.norm(v) + 1e-12)
                updated = True
            oc = self._outcomes[name][-recalibrate_window:]
            if len(oc) >= 8:
                try:
                    self._calib[name] = IsotonicCalibrator([g for g, _ in oc],
                                                           [c for _, c in oc])
                except ValueError:
                    pass    # degenerate window (all one score); keep the previous calibrator
            out[name] = {"was_correct": bool(correct), "updated": updated}
        return out

    def _validate_escalated(self, spec, raw):
        """Hold a model end's answer to the SAME schema the substrate is held to. The seam is
        where type guarantees usually die -- a generated string that is almost an option -- so
        validation here is strict: exact option after strip, real bool/number, or SchemaError."""
        if spec["type"] == "choice":
            v = raw.strip() if isinstance(raw, str) else raw
            if v not in spec["options"]:
                raise SchemaError("escalated answer %r is not one of %s" % (raw, spec["options"]))
            return v
        if spec["type"] == "noul":
            if isinstance(raw, bool):
                return raw
            v = raw.strip().lower() if isinstance(raw, str) else None
            if v in ("yes", "no"):
                return v == "yes"
            raise SchemaError("escalated noul answer %r is not yes/no/bool" % (raw,))
        try:
            v = float(raw)
        except (TypeError, ValueError):
            raise SchemaError("escalated score answer %r is not numeric" % (raw,))
        if not np.isfinite(v):
            raise SchemaError("escalated score answer is not finite")
        return float(np.clip(v, spec["min"], spec["max"]))

    def decide_or_escalate(self, state, escalate=None, p_floor=None, max_retry=1):
        """The full System One + System Two composition with an HONEST gate between them: answer
        from the substrate where confident (and, with p_floor, where the calibrated probability
        clears it); otherwise hand the question -- WITH its ranked evidence, the substrate's
        prior -- to the `escalate` callable (a model end, a human queue, mind.serve). The
        escalated answer is held to the SAME schema, retried once with the named error, then
        refused: the type guarantee survives the seam. escalate=None marks what would escalate
        (escalate_needed) without pretending. Escalated answers carry p=None -- we do not
        launder someone else's judgment through our calibrator. Every answer says its via:
        substrate | escalated | refused. Returns {"answers": ..., "summary": counts}."""
        ans = self.decide(state)
        summary = {"answered": 0, "escalated": 0, "refused": 0}
        for name, a in ans.items():
            needs = a.get("abstained", False) or (
                p_floor is not None and a.get("p") is not None and a["p"] < p_floor)
            if not needs:
                a["via"] = "substrate"
                summary["answered"] += 1
                continue
            if escalate is None:
                a["via"] = "substrate"
                a["escalate_needed"] = True
                summary["refused"] += 1
                continue
            spec = self.questions[name]
            payload = {"question": name,
                       "spec": {k: v for k, v in spec.items() if k != "examples"},
                       "state": state, "evidence": a.get("ranked"), "why": a.get("why"),
                       # NOOA docstring-as-prompt (sweep 176): the four-part prompt generated from the schema,
                       # so a model end receives the contract it will be held to, not a bare question.
                       "prompt": self.escalation_prompt(name, state, a.get("ranked"))}
            value, err = None, None
            for _ in range(max_retry + 1):
                try:
                    raw = escalate(dict(payload, error=err) if err else payload)
                    value = self._validate_escalated(spec, raw)
                    break
                except SchemaError as e:
                    err, value = str(e), None
            if value is None:
                a["via"] = "refused"
                a["why"] = "model end failed schema %d time(s): %s" % (max_retry + 1, err)
                summary["refused"] += 1
            else:
                a["value"] = value
                a["abstained"] = False
                a["via"] = "escalated"
                a["p"] = None            # not our calibration to claim
                a["calibrated"] = False
                summary["escalated"] += 1
        return {"answers": ans, "summary": summary}

    def drift_report(self, min_len=24, min_seg=None):
        """Two-channel drift watch (Tarter/Siemion seats): per question, change-point detect the
        label-FREE support stream (every decide records the winning cosine; covariate shift shows
        here before labels arrive) and the label-lagged correctness stream (from observe). Both
        DELEGATE to holographic_demux.segment_stream -- the engine's located change-point
        detector -- never a hand-rolled CUSUM; a homogeneous stream honestly reports no drift.
        Streams shorter than min_len return status insufficient rather than a verdict."""
        from holographic.sampling_and_signal.holographic_demux import segment_stream
        rep = {}
        for name in (self.questions or {}):
            if self.questions[name]["type"] == "score":
                continue
            chans = {}
            for chan, xs in (("support", self._support.get(name, [])),
                             ("correctness", [c for _, c in self._outcomes.get(name, [])])):
                if len(xs) < min_len:
                    chans[chan] = {"status": "insufficient", "n": len(xs)}
                else:
                    # min_seg (sweep 176, H4 follow-up): None keeps the sweep-172 behaviour (max(16, min_len//2)).
                    # MEASURED on the shift-at-150 protocol, 5 seeds: nb's max-posterior support fired two alarms
                    # ~110 rows EARLY at min_seg 16/32 and none at 48 (latency -4..0), with ~1 false alarm per
                    # 300 stationary rows either way; the prototype's cosine support is stable at every setting.
                    # KEPT NEGATIVE: pre-smoothing the stream (rolling mean 8) is catastrophic -- 17 to 55 false
                    # alarms -- because the segmenter reads the autocorrelation as structure. Use min_seg, never
                    # a smoother. Recommendation: min_seg=48 for scorer="nb".
                    seg = segment_stream(list(xs), min_seg=(int(min_seg) if min_seg else max(16, min_len // 2)))
                    chans[chan] = {"status": "ok", "n": len(xs),
                                   "drift": bool(seg["boundaries"]),
                                   "boundaries": seg["boundaries"],
                                   # H4 (sweep 176): the row the alarm first points at. MEASURED, 150 AG News
                                   # rows then 150 SST-2 rows (true shift at 150), 5 seeds, support channel:
                                   # prototype fired 4/5 within +-2 rows, 1 false alarm in 5 stationary
                                   # 300-row runs; nb fired 5/5 but 2 boundaries landed ~110 rows EARLY and it
                                   # false-alarmed 6 times -- a sharp posterior's max is a noisy support
                                   # statistic. Kept finding: for nb, do not trust the support channel alone.
                                   "first_boundary": (seg["boundaries"][0] if seg["boundaries"] else None),
                                   "n_segments": seg["n_segments"]}
            rep[name] = chans
        return rep

    # ---------------- calibration ----------------
    def calibrate(self, labeled):
        """Fit per-question isotonic calibrators from labeled outcomes.
        labeled = [(state_text, {qname: true_answer}), ...]; choice truth is the option string,
        noul truth is a bool. Feature = the top1-top2 margin gap (the same quantity the abstention
        gate judges, so the calibrated p answers exactly 'given THIS gap, how often is the pick
        right'). Score questions are NOT calibrated -- they carry support+spread instead; a
        fabricated sigma would be the overclaim this module refuses. Needs >=4 outcomes per
        question; fewer -> that question stays uncalibrated (reported, not silent)."""
        feats, hits = {}, {}
        for state, truths in labeled:
            ans = self.decide(state)
            for name, truth in truths.items():
                spec = self.questions.get(name)
                if spec is None or spec["type"] == "score":
                    continue
                a = ans[name]
                want = ("yes" if truth else "no") if spec["type"] == "noul" else truth
                feats.setdefault(name, []).append(a["margin_gap"])
                hits.setdefault(name, []).append(1.0 if a["ranked"][0][0] == want else 0.0)
        report = {}
        for name, xs in feats.items():
            if len(xs) >= 4:
                self._calib[name] = IsotonicCalibrator(xs, hits[name])
                report[name] = {"calibrated": True, "n": len(xs),
                                "accuracy": float(np.mean(hits[name]))}
            else:
                report[name] = {"calibrated": False, "n": len(xs),
                                "why": "needs >=4 labeled outcomes"}
        return report

    def calibrate_conformal(self, labeled, alpha=0.05):
        """CONFORMAL ANSWER SETS (sweep 176, backlog H1): after this, every choice / noul answer also carries
        `set` -- the smallest set of options guaranteed to contain the truth with probability >= 1 - alpha
        under exchangeability (split conformal). Nonconformity is scorer-agnostic: the gap between the top
        score and the TRUE option's score on held-out labeled rows; the quantile step DELEGATES to
        holographic_reasoning.ConformalPredictor (the (n+1)(1-alpha) order statistic, finite-sample). A
        question with no labeled rows gets no set. WHY this beats a single p: a hosted decision API returns
        one label and one number; a set with a coverage guarantee tells the caller exactly when to ask a
        human -- the set has more than one member -- and the guarantee is distribution-free."""
        from holographic.agents_and_reasoning.holographic_reasoning import ConformalPredictor
        gaps = {}
        for state, truths in labeled:
            ans = self._decide_encoded(self._unit(state), state=state)
            for name, truth in truths.items():
                a = ans.get(name)
                if not a or a.get("type") not in ("choice", "noul") or not a.get("ranked"):
                    continue
                sc = dict(a["ranked"])
                if truth not in sc:
                    continue
                gaps.setdefault(name, []).append(float(a["ranked"][0][1]) - float(sc[truth]))
        self._conformal = {}
        for name, g in gaps.items():
            cp = ConformalPredictor(alpha=alpha)
            self._conformal[name] = {"q": cp.calibrate(g), "n": len(g), "alpha": alpha}
        return {k: dict(v) for k, v in self._conformal.items()}

    def batch_fdr(self, states, question, alpha=0.10, n_null=200, seed=0, dependent=False):
        """BATCH FALSE-DISCOVERY CONTROL over a stream of decisions (sweep 176, backlog H2): for each state, a
        shuffle-null p-value of its margin -- the margin re-scored on `n_null` states of the SAME token count
        drawn from the examples' own vocabulary (in-vocabulary word salad is exactly what a misroutable state
        is; the same null construction route_or_abstain uses) -- then Benjamini-Hochberg across the batch
        (DELEGATES to holographic_ablate.bh_fdr). Returns {accepted: [bool], p: [float], q: alpha, n}.
        MEASURED (150 real AG News rows + 150 in-vocabulary noise rows per seed, 3 seeds):
            BH   q=0.05 -> FDR 0.033, real rows accepted 0.129 | q=0.10 -> FDR 0.028, accepted 0.196
            BY   (dependent=True) -> accepts NOTHING at either q (kept negative: the log-harmonic penalty
                 never clears a permutation p over hundreds of tests)
            uncorrected p<q -> FDR 0.144 / 0.215 -- the naive gate lets noise through.
        The guarantee holds; the power is low because a bag model cannot tell in-vocabulary noise from
        text by score alone. n_null below ~200 makes p too coarse for BH to reject anything (32 nulls: 0)."""
        import re as _re
        from holographic.misc.holographic_ablate import bh_fdr
        spec = self.questions[question]
        vocab = sorted({w for v in (spec.get("examples") or {}).values() for t in v for w in _re.findall(r"[a-z0-9]+", t.lower())})
        if not vocab:
            raise SchemaError("batch_fdr needs examples on %r to build the in-vocabulary null" % question)
        cache = {}

        def bucket(n):
            return min(60, 5 * (max(3, n) // 5))

        pvals = []
        for st in states:
            b = bucket(len(_re.findall(r"[a-z0-9]+", (st or "").lower())))
            if b not in cache:
                g = np.random.default_rng(int(seed) + b)
                cache[b] = np.array([self.decide(" ".join(g.choice(vocab, size=b)))[question]["margin_gap"]
                                     for _ in range(int(n_null))])
            obs = self.decide(st)[question]["margin_gap"]
            pvals.append(float((np.sum(cache[b] >= obs) + 1) / (len(cache[b]) + 1)))
        rej, _ = bh_fdr(np.array(pvals), alpha=alpha, dependent=dependent)
        return {"accepted": [bool(r) for r in rej], "p": pvals, "q": float(alpha), "n": len(states)}

    def absorb_unlabeled(self, states, question, iters=3, lam=1.0, max_options=10, holdout=0.25, seed=0):
        """SEMI-SUPERVISED EM over UNLABELED states for the nb scorer (sweep 176, backlog H3) -- WITH THE GUARD
        that the measurement demanded. E-step: posteriors on the unlabeled states; M-step: recount the option
        tables from the labeled examples (weight 1) plus the unlabeled states weighted lam * posterior. Closed
        form both ways, no gradient (Nigam et al. 2000).
        MEASURED before the guard (doc 01): +0.069 on AG News at k=32 (4 options), ~0 on SST-2, and
        CATASTROPHIC on Banking77 -- 0.093 at k=5 across 77 intents, posterior mass collapsing onto a few
        options. MEASURED through this method (3 seeds, AG News k=32, 600 unlabeled): nb_transform=False
        0.697 -> 0.752 (+0.056, spread 0.030) -- which BEATS the transformed default's 0.713; with
        nb_transform=True the gain vanishes (0.713 -> 0.709). So the recommended combination when unlabeled
        traffic is available on a <=10-option topical task is nb_transform=False + this method. SST-2:
        +0.010; Banking77: refused by guard (1). So two guards, both measured: (1) refuse when the question has more than `max_options`
        options; (2) hold out `holdout` of the labeled examples, run EM, and KEEP the new table only if
        held-out accuracy did not drop -- otherwise restore and report why. Returns {applied, reason,
        heldout_before, heldout_after, n_unlabeled}. Never mutates the table when it declines."""
        if self.scorer != "nb" or question not in self._nb:
            return {"applied": False, "reason": "absorb_unlabeled needs scorer='nb' and a fitted table", "n_unlabeled": len(states)}
        spec = self.questions[question]
        labels = self._meta[question]["labels"]
        if len(labels) > int(max_options):
            return {"applied": False, "reason": "guard: %d options > max_options %d (EM collapsed at 77)" % (len(labels), max_options),
                    "n_unlabeled": len(states)}
        ex = spec.get("examples") or {}
        rng = np.random.default_rng(int(seed))
        train, held = {}, []
        for o in labels:
            texts = list(ex.get(o) or [o])
            idx = rng.permutation(len(texts))
            n_h = int(round(len(texts) * float(holdout))) if len(texts) >= 4 else 0
            held += [(texts[i], o) for i in idx[:n_h]]
            train[o] = [texts[i] for i in idx[n_h:]] or texts
        if len(held) < len(labels):          # at least one held-out row per option, or no honest check
            return {"applied": False, "reason": "guard: too few labeled examples to hold out a check", "n_unlabeled": len(states)}

        def table_from(weighted):
            counts = {o: {} for o in labels}
            tot = {o: 0.0 for o in labels}
            vocab = set()
            for o, text, w in weighted:
                for tok, c in _nb_transform(_nb_feats(text, self.nb_bigrams), self._nb[question]["idf"],
                                            self._nb[question]["n_docs"]).items():
                    counts[o][tok] = counts[o].get(tok, 0.0) + w * c
                    tot[o] += w * c
                    vocab.add(tok)
            return {"counts": counts, "tot": tot, "vocab": vocab, "idf": self._nb[question]["idf"],
                    "n_docs": self._nb[question]["n_docs"]}

        def heldout_acc():
            return float(np.mean([self.decide(t)[question]["ranked"][0][0] == o for t, o in held])) if held else float("nan")

        original = self._nb[question]
        base_rows = [(o, t, 1.0) for o in labels for t in train[o]]
        self._nb[question] = table_from(base_rows)
        before = heldout_acc()
        for _ in range(int(iters)):
            post = [self._nb_posterior(question, s) for s in states]
            rows = list(base_rows)
            for s, pv in zip(states, post):
                for o, pw in zip(labels, pv):
                    if pw > 1e-3:
                        rows.append((o, s, float(lam) * float(pw)))
            self._nb[question] = table_from(rows)
        after = heldout_acc()
        if after + 1e-12 < before:
            self._nb[question] = original
            return {"applied": False, "reason": "guard: held-out accuracy fell %.3f -> %.3f; table restored" % (before, after),
                    "heldout_before": before, "heldout_after": after, "n_unlabeled": len(states)}
        # keep the EM table but rebuilt on ALL labeled examples (the held-out rows go back in)
        full_rows = [(o, t, 1.0) for o in labels for t in (ex.get(o) or [o])]
        post = [self._nb_posterior(question, s) for s in states]
        for s, pv in zip(states, post):
            for o, pw in zip(labels, pv):
                if pw > 1e-3:
                    full_rows.append((o, s, float(lam) * float(pw)))
        self._nb[question] = table_from(full_rows)
        return {"applied": True, "reason": "held-out accuracy %.3f -> %.3f" % (before, after),
                "heldout_before": before, "heldout_after": after, "n_unlabeled": len(states)}

    def escalation_prompt(self, name, state, ranked=None):
        """THE MODEL-END PROMPT, GENERATED FROM THE SCHEMA (sweep 176, NOOA's docstring-as-prompt and
        annotations-as-contract, arXiv 2607.20709 s1, and the four-part shape doc 05 s1): goal, return
        format, constraints, verification -- constraints early, explicit permission to abstain, the
        substrate's ranked evidence as the prior, and the question's own schema as the return CONTRACT the
        harness validates (decide_or_escalate holds the reply to it and retries once). The prompt is a pure
        function of (schema, state, ranked): deterministic, no model needed to write it."""
        spec = self.questions[name]
        qtype = spec["type"]
        opts = list(spec.get("options") or (["yes", "no"] if qtype == "noul" else []))
        lines = ["GOAL: answer the typed question %r about the STATE below." % name]
        if qtype == "score":
            lines.append("RETURN FORMAT: one JSON number between %s and %s, or null to abstain." % (spec.get("min"), spec.get("max")))
        else:
            lines.append("RETURN FORMAT: exactly one of %s as a JSON string, or null to abstain." % opts)
        lines.append("CONSTRAINTS: choose only from the options above; do not invent an option; if the state does not "
                     "support any option, return null -- abstaining is correct and is never penalised.")
        ex = spec.get("examples") or {}
        if ex:
            lines.append("EXAMPLES (one per option, from the schema):")
            for o in opts:
                if ex.get(o):
                    lines.append("  %s: %s" % (o, ex[o][0]))
        if ranked:
            lines.append("SUBSTRATE EVIDENCE (its ranked prior, scores in [0,1] or cosine): %s" % [(n, round(float(s), 3)) for n, s in ranked[:5]])
        lines.append("VERIFICATION: the harness validates the reply against the schema %s and retries once on a violation; "
                     "a reply outside the options is refused." % json.dumps({"type": qtype, "options": opts} if qtype != "score" else {"type": "score", "min": spec.get("min"), "max": spec.get("max")}))
        lines.append("STATE: %s" % (state if len(str(state)) <= 2000 else str(state)[:2000] + " ...[truncated %d chars]" % (len(str(state)) - 2000)))
        return "\n".join(lines)

    def calibration_report(self, labeled, bins=10):
        """The honest reliability readout on a labeled set: per calibrated question, accuracy,
        Brier score, and ECE (expected calibration error over equal-width probability bins).
        Run this on data NOT used in calibrate() -- reporting on the fit set flatters the fit."""
        rep = {}
        per_q = {}
        for state, truths in labeled:
            ans = self.decide(state)
            for name, truth in truths.items():
                spec = self.questions.get(name)
                if spec is None or spec["type"] == "score" or name not in self._calib:
                    continue
                a = ans[name]
                want = ("yes" if truth else "no") if spec["type"] == "noul" else truth
                per_q.setdefault(name, []).append(
                    (a["p"], 1.0 if a["ranked"][0][0] == want else 0.0))
        for name, pairs in per_q.items():
            p = np.array([x for x, _ in pairs])
            y = np.array([x for _, x in pairs])
            edges = np.linspace(0.0, 1.0, bins + 1)
            which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
            ece = 0.0
            for b in range(bins):
                m = which == b
                if m.any():
                    ece += (m.mean()) * abs(p[m].mean() - y[m].mean())
            rep[name] = {"n": int(p.size), "accuracy": float(y.mean()),
                         "brier": float(np.mean((p - y) ** 2)), "ece": float(ece)}
        return rep


# ---------------------------------------------------------------------------
def hashed_ngram_encode(dim=2048, lo=3, hi=5):
    """Character 3..5-gram hashed BUNDLE encoder: each n-gram is a hashlib-seeded random
    hypervector, a text is their superposition. Substrate-native (bundle of atoms), deterministic
    (hashlib, never hash()), stdlib+numpy only, and MEASURED (sweep 171, 200-row evals, 3 seeds,
    k=32/class): AG News topic routing forced 0.648 (spread 0.035) vs token-overlap 0.630 and
    majority 0.285; at ~53% coverage the answered subset hits 0.800 -- answer-or-route, which is
    the whole System One composition. KEPT NEGATIVE, CORRECTED (sweep 174): SST-2 sentiment is
    at chance with k=32 examples/class (0.513 in sweep 171; 0.556 +/- 0.057 re-measured), and
    mind.perceive at dim 512 and 2048 did not move it either -- but the negative was k-LIMITED,
    not fundamental: the SAME encoder reaches 0.669 at k=600 and the nb scorer 0.725 (see
    SystemOne(scorer=)). "No bag encoder carries sentiment" was true of 32 examples, not of the
    encoder. Every n-gram here weighs the same (" th" as much as a distinctive word); IDF
    weighting measured +0.02 to +0.03 (char_idf centroid 0.748 vs 0.723 on AG News k=300) --
    small, real, not yet applied here. Cosine gaps in this space are small -- pair with margin
    ~0.02 (systemone's encoder='ngram' path sets that default for you)."""
    import hashlib as _hl
    cache = {}

    def _vec(g):
        if g not in cache:
            seed = int.from_bytes(_hl.sha256(g.encode()).digest()[:8], "big") % (2 ** 32)
            cache[g] = np.random.default_rng(seed).standard_normal(dim)
        return cache[g]

    def enc(text):
        t = " " + text.lower() + " "
        v = np.zeros(dim)
        for n in range(lo, hi + 1):
            for i in range(len(t) - n + 1):
                v += _vec(t[i:i + n])
        return v
    return enc


def _hash_bow_encode(dim=512):
    """Self-contained deterministic text encoder for the selftest ONLY: bag of hashlib-seeded
    token vectors. Real use goes through mind.perceive; this exists so the selftest pins the
    CONTRACT without depending on the mind's encoder of the day (sweep 169's lesson: pin the
    mechanism with a controlled vocabulary, never a live calibration artifact)."""
    def enc(text):
        v = np.zeros(dim)
        for tok in text.lower().split():
            seed = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:8], "big") % (2 ** 32)
            v += np.random.default_rng(seed).standard_normal(dim)
        return v
    return enc


def _selftest():
    """Fail loudly on the things most likely to break: schema leniency, the noul==choice identity,
    single-vs-batch drift, PAV correctness, and the abstention paths."""
    enc = _hash_bow_encode()

    # 1. Schema strictness: each of these MUST raise.
    for bad in [{}, {"q": {"type": "pick"}},
                {"q": {"type": "choice", "options": ["a"]}},
                {"q": {"type": "choice", "options": ["a", "b"], "extra": 1}},
                {"q": {"type": "score", "min": 1, "max": 1}},
                {"q": {"type": "noul", "examples": {"maybe": ["x"]}}}]:
        try:
            validate_questions(bad)
            raise AssertionError("schema accepted bad spec: %r" % (bad,))
        except SchemaError:
            pass

    # 2. noul IS choice(yes/no): identical ranked scores to 1e-12 given the same examples.
    ex = {"yes": ["ship it now", "approve and ship"], "no": ["reject this", "deny the request"]}
    s1 = SystemOne(enc); s1.fit({"q": {"type": "noul", "examples": ex}})
    s2 = SystemOne(enc); s2.fit({"q": {"type": "choice", "options": ["yes", "no"], "examples": ex}})
    r1 = s1.decide("approve and ship it")["q"]["ranked"]
    r2 = s2.decide("approve and ship it")["q"]["ranked"]
    assert all(a[0] == b[0] and abs(a[1] - b[1]) < 1e-12 for a, b in zip(r1, r2)), \
        "noul diverged from its choice(yes/no) identity"

    # 3. Few-shot choice: controlled vocab, held-out phrase; and the ambiguous state abstains.
    so = SystemOne(enc)
    so.fit({"cat": {"type": "choice", "options": ["billing", "shipping", "bug"], "examples": {
        "billing": ["invoice charge refund payment", "card charged twice billing"],
        "shipping": ["package delivery tracking courier", "parcel arrives late shipping"],
        "bug": ["crash stack trace error", "segfault bug crash report"]}},
        "urgency": {"type": "score", "min": 0, "max": 100, "anchors": [
            ("meltdown outage everything broken", 95.0),
            ("annoying but there is a workaround", 40.0),
            ("cosmetic typo no rush", 5.0)]}})
    a = so.decide("my card was charged twice on the invoice")
    assert a["cat"]["value"] == "billing" and a["cat"]["confident"], a["cat"]
    assert a["cat"]["p"] is None and a["cat"]["calibrated"] is False, \
        "uncalibrated decision must not wear a probability"
    tie = so.decide("zzz qqq unrelated nonsense words")   # off-manifold: gap collapses
    assert tie["urgency"]["abstained"], "off-manifold score must abstain, got %r" % tie["urgency"]
    # 4. Score sanity: a state sitting ON an anchor returns (near) its value; clamp holds.
    v = so.decide("meltdown outage everything broken")["urgency"]["value"]
    assert v is not None and abs(v - 95.0) < 5.0, v

    # 5. Single vs batch: bit-identical rows (a drifting batch path is a silent fork).
    states = ["package tracking says lost", "crash on startup stack trace"]
    singles = [so.decide(s) for s in states]
    assert so.decide_map(states) == singles, "decide_map drifted from decide"

    # 6. PAV on a hand case: scores .1..
    xs, ys = _pav_fit([0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 1.0, 1.0])
    # n=4 -> clip band [0.125, 0.875]; ideal fit (0,0,1,1) clips to (.125,.125,.875,.875)
    assert np.allclose(ys, [0.125, 0.125, 0.875, 0.875], atol=1e-12), ys
    assert np.all(np.diff(ys) >= -1e-15), "isotonic fit is not monotone"
    # violator case pools to the block mean: y=[1,0] on rising scores -> both 0.5, then clipped? no:
    xs2, ys2 = _pav_fit([0.1, 0.2, 0.3, 0.4], [1.0, 0.0, 1.0, 1.0])
    assert abs(ys2[0] - 0.5) < 1e-12 and abs(ys2[1] - 0.5) < 1e-12, ys2

    # 7. Calibration end-to-end: labeled outcomes produce p in [0,1], never exactly 0/1.
    labeled = [("invoice refund charge", {"cat": "billing"}),
               ("card charged billing problem", {"cat": "billing"}),
               ("parcel courier tracking", {"cat": "shipping"}),
               ("delivery late package", {"cat": "shipping"}),
               ("stack trace crash", {"cat": "bug"}),
               ("segfault error report", {"cat": "bug"})]
    rep = so.calibrate(labeled)
    assert rep["cat"]["calibrated"] and rep["cat"]["n"] == 6, rep
    p = so.decide("refund my invoice charge")["cat"]["p"]
    assert p is not None and 0.0 < p < 1.0, p
    rel = so.calibration_report(labeled)
    assert 0.0 <= rel["cat"]["ece"] <= 1.0 and rel["cat"]["brier"] <= 0.5, rel

    # 8. min_support (the Milanfar gap): off-manifold choice ABSTAINS instead of lucking a pick.
    sf = SystemOne(enc, min_support=0.35)
    sf.fit({"cat": {"type": "choice", "options": ["billing", "shipping"], "examples": {
        "billing": ["invoice charge refund"], "shipping": ["package tracking courier"]}}})
    off = sf.decide("qqq zzz unrelated nonsense")["cat"]
    assert off["abstained"] and "min_support" in off["why"], off

    # 9. observe: prequential + the AdaptHD flip. Teach the truth against the initial pick and
    # the prototypes must move until the pick flips; lr=0 must record but never update.
    ob = SystemOne(enc)
    ob.fit({"cat": {"type": "choice", "options": ["alpha", "beta"], "examples": {
        "alpha": ["red crimson scarlet"], "beta": ["blue azure navy"]}}})
    state = "red crimson scarlet paint"
    first = ob.observe(state, {"cat": "beta"})["cat"]      # truth says the OTHER class
    assert first["was_correct"] is False and first["updated"] is True, first
    for _ in range(6):
        ob.observe(state, {"cat": "beta"})
    assert ob.decide(state)["cat"]["ranked"][0][0] == "beta", "AdaptHD update failed to flip"
    frozen = SystemOne(enc)
    frozen.fit({"cat": {"type": "choice", "options": ["alpha", "beta"], "examples": {
        "alpha": ["red crimson scarlet"], "beta": ["blue azure navy"]}}})
    fr = frozen.observe(state, {"cat": "beta"}, lr=0)["cat"]
    assert fr["updated"] is False, fr
    assert frozen.decide(state)["cat"]["ranked"][0][0] == "alpha", "lr=0 must not learn"

    # 10. Escalation seam: the schema survives it. Valid answer -> escalated + typed; an answer
    # that fails the schema twice -> refused, never a lie in the value slot.
    es = SystemOne(enc, min_support=0.99)      # force abstention so every question escalates
    es.fit({"cat": {"type": "choice", "options": ["billing", "shipping"], "examples": {
        "billing": ["invoice charge refund"], "shipping": ["package tracking courier"]}},
        "ok": {"type": "noul"}})
    good = es.decide_or_escalate("anything at all",
                                 escalate=lambda p: "billing" if p["question"] == "cat" else True)
    assert good["answers"]["cat"]["via"] == "escalated" and good["answers"]["cat"]["value"] == "billing"
    assert good["answers"]["ok"]["value"] is True and good["summary"]["escalated"] == 2, good
    bad = es.decide_or_escalate("anything", escalate=lambda p: "not-an-option")
    assert bad["answers"]["cat"]["via"] == "refused" and bad["answers"]["cat"]["value"] is None
    none = es.decide_or_escalate("anything")
    assert none["answers"]["cat"].get("escalate_needed") is True, none

    # 11. Drift delegation: homogeneous support -> honestly no drift; a support collapse -> a
    # located boundary; short streams -> insufficient, not a verdict.
    dr = SystemOne(enc)
    dr.fit({"cat": {"type": "choice", "options": ["a", "b"], "examples": {
        "a": ["one two three"], "b": ["four five six"]}}})
    dr._support["cat"] = [0.8] * 40 + [0.2] * 40
    rep = dr.drift_report()["cat"]
    assert rep["support"]["drift"] is True and rep["support"]["boundaries"] == [40], rep
    dr._support["cat"] = [0.8] * 60
    assert dr.drift_report()["cat"]["support"]["drift"] is False
    dr._support["cat"] = [0.8] * 5
    assert dr.drift_report()["cat"]["support"]["status"] == "insufficient"
    # 12. The nb scorer (sweep 174): same schema, same typed answers, posteriors that sum to one,
    #     a count-based update that flips a wrong decision, and bit-determinism. Default-off is
    #     pinned too: a SystemOne built without scorer= must not carry a count table at all.
    nb = SystemOne(enc, scorer="nb", margin=0.05)
    nb.fit({"cat": {"type": "choice", "options": ["billing", "shipping"], "examples": {
        "billing": ["card charged twice", "refund my invoice fee", "charge on my statement"],
        "shipping": ["parcel lost in transit", "courier delivery late", "package never arrived"]}}})
    a = nb.decide("my card shows a double charge")["cat"]
    assert a["value"] == "billing" and abs(sum(s for _, s in a["ranked"]) - 1.0) < 1e-9, a
    # ("where is my parcel" leans billing here: "my" appears twice in the billing examples and
    # "parcel" once in shipping -- a genuine 3-example naive-Bayes artifact, which is why the
    # pin uses an unambiguous phrase; NB quality is measured in the bench, not asserted here.)
    assert nb.decide("parcel lost by the courier")["cat"]["value"] == "shipping"
    r1 = nb.decide("the courier lost my package")["cat"]["ranked"]
    r2 = nb.decide("the courier lost my package")["cat"]["ranked"]
    assert r1 == r2, "nb scorer is not deterministic"
    # a novel phrase the table has never seen is wrong at first, then learned by COUNTING
    novel = "tracking number shows no movement"
    before = nb.decide(novel)["cat"]["ranked"][0][0]
    nb.observe(novel, {"cat": "shipping"}, lr=3.0)
    if before != "shipping":
        assert nb.decide(novel)["cat"]["ranked"][0][0] == "shipping"
    assert SystemOne(enc)._nb == {} and SystemOne(enc).scorer == "prototype"
    try:
        SystemOne(enc, scorer="bayes")
        raise AssertionError("unknown scorer accepted")
    except SchemaError:
        pass
    # 13. The question lint (sweep 176): it must catch every failure the two tool experiments hit --
    #     a multi-clause state, a contrastive clause, a >2x example budget imbalance, a scorer outside the
    #     regime table -- and NOT flag "Buttons on the top strip" as contrastive (the word-boundary bug).
    para = ("Passive radiators sit at each end of the cylinder. They work like bass reflex ports but without "
            "needing a port tube. Buttons on the top strip.")
    cl = clauses(para)
    assert len(cl) == 3 and [is_contrastive(c) for c in cl] == [False, True, False], (cl, [is_contrastive(c) for c in cl])
    assert clauses("make a mesh and then smooth it") == ["make a mesh", "smooth it"]
    lint = schema_lint({"tool": {"type": "choice", "options": ["extrude_profile", "new_primitive"],
                                 "examples": {"extrude_profile": ["a cylinder from a circle profile extruded to a length"] * 3,
                                              "new_primitive": ["cube"] * 3}}},
                       states=[para], scorer="nb")
    whats = " | ".join(f["what"] for f in lint["findings"])
    assert "imbalanced" in whats and "new_primitive" in whats, whats
    assert "3 clauses" in whats and "contrastive" in whats, whats
    assert lint["recommended_scorer"] == "prototype" and lint["k_min"] == 3, lint
    assert schema_lint({"q": {"type": "choice", "options": ["a"], "examples": {"a": ["x"]}}})["ok"] is False
    # 14. Conformal answer sets (sweep 176, H1): after calibrate_conformal every choice carries `set`; the
    #     set always contains the top-1, its size is >= 1, and a calibration with zero gaps (every row
    #     exactly right) yields singleton sets. The COVERAGE claim is measured in the bench (Banking77:
    #     nominal 0.95 -> 0.960 / 0.961; nominal 0.90 -> 0.913 / 0.918), not asserted on toy data.
    cs = SystemOne(enc, scorer="nb", margin=0.0)
    cs.fit({"cat": {"type": "choice", "options": ["billing", "shipping"], "examples": {
        "billing": ["card charged twice", "refund my invoice fee", "charge on my statement"],
        "shipping": ["parcel lost in transit", "courier delivery late", "package never arrived"]}}})
    info = cs.calibrate_conformal([("my card was charged a fee", {"cat": "billing"}), ("the parcel is lost", {"cat": "shipping"}),
                                   ("refund the invoice", {"cat": "billing"}), ("courier is late", {"cat": "shipping"})], alpha=0.1)
    assert info["cat"]["n"] == 4 and info["cat"]["q"] >= 0.0
    a = cs.decide("courier lost the package")["cat"]
    assert a["ranked"][0][0] in a["set"] and 1 <= len(a["set"]) <= 2 and a["set_alpha"] == 0.1, a
    cs2 = SystemOne(enc, scorer="nb", margin=0.0)
    cs2.fit({"cat": {"type": "choice", "options": ["billing", "shipping"], "examples": {"billing": ["card"], "shipping": ["parcel"]}}})
    assert "set" not in cs2.decide("card")["cat"]                                  # no calibration, no set
    # 15. batch_fdr (H2): p-values in (0,1], one verdict per state, BH accepts a clearly in-domain state
    #     and not a scrambled one at q=0.10 when the null is fine enough; the FDR CLAIM lives in the docstring
    #     table and the bench, not in a toy assertion.
    # (An OUT-of-vocabulary state is not the null's noise: its margin comes from the classes' differing
    #  smoothing floors and can be large. The null is IN-vocabulary word salad, so no ordering is asserted.)
    bf = cs.batch_fdr(["my card was charged twice for one invoice", "zzz qqq"], "cat", alpha=0.10, n_null=64)
    assert len(bf["accepted"]) == 2 and all(0 < pv <= 1 for pv in bf["p"]) and bf["n"] == 2, bf
    assert cs.batch_fdr(["a"], "cat", alpha=0.10, n_null=8, seed=1) == cs.batch_fdr(["a"], "cat", alpha=0.10, n_null=8, seed=1)
    # 16. absorb_unlabeled (H3): the option-count guard refuses (never touches the table), and on a small
    #     2-option task the applied path returns its held-out numbers and keeps the table decidable.
    big = SystemOne(enc, scorer="nb"); big.fit({"q": {"type": "choice", "options": [str(i) for i in range(12)],
                                                 "examples": {str(i): ["word%d a b c" % i] * 4 for i in range(12)}}})
    snap = dict(big._nb["q"]["tot"])
    g = big.absorb_unlabeled(["a b c word3", "word7 a"], "q")
    assert g["applied"] is False and "max_options" in g["reason"] and big._nb["q"]["tot"] == snap, g
    small = SystemOne(enc, scorer="nb", nb_transform=False, margin=0.0)
    small.fit({"cat": {"type": "choice", "options": ["billing", "shipping"], "examples": {
        "billing": ["card charged twice", "refund my invoice fee", "charge on my statement", "double charge on the card"],
        "shipping": ["parcel lost in transit", "courier delivery late", "package never arrived", "tracking shows no movement"]}}})
    r = small.absorb_unlabeled(["my invoice was charged twice", "the courier never delivered the parcel"], "cat")
    assert set(r) >= {"applied", "reason", "heldout_before", "heldout_after", "n_unlabeled"} and r["n_unlabeled"] == 2
    assert small.decide("courier lost the parcel")["cat"]["ranked"][0][0] == "shipping"
    # 17. escalation_prompt (NOOA docstring-as-prompt): deterministic, four parts in order, carries the
    #     options as the contract, permission to abstain, the ranked evidence, and the state; and
    #     decide_or_escalate hands it to the model end inside the payload.
    pr = cs.escalation_prompt("cat", "parcel is late", ranked=[("shipping", 0.7), ("billing", 0.3)])
    for part in ("GOAL:", "RETURN FORMAT:", "CONSTRAINTS:", "VERIFICATION:", "STATE:"):
        assert part in pr, part
    assert pr.index("GOAL:") < pr.index("RETURN FORMAT:") < pr.index("CONSTRAINTS:") < pr.index("VERIFICATION:") < pr.index("STATE:")
    assert "null" in pr and "shipping" in pr and pr == cs.escalation_prompt("cat", "parcel is late", ranked=[("shipping", 0.7), ("billing", 0.3)])
    seen = {}
    shy = SystemOne(enc, scorer="nb", margin=0.99)          # abstains on everything -> must escalate
    shy.fit({"cat": {"type": "choice", "options": ["billing", "shipping"], "examples": {"billing": ["card"], "shipping": ["parcel"]}}})
    shy.decide_or_escalate("zzz qqq", escalate=lambda pl: (seen.setdefault("prompt", pl.get("prompt")), "billing")[1])
    assert seen.get("prompt") and "RETURN FORMAT" in seen["prompt"], seen
    return {"ok": True, "pinned": 17}


if __name__ == "__main__":
    print(_selftest())

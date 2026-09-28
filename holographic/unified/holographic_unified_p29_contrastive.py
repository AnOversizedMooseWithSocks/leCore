"""UnifiedMind part 29 -- ONE contrastive learning rule under every door of the reflex arc (the CLM backlog).

The loop this part wires (docs/research/evidence/clm_panel_20260926/README.md; the local backlog note):
    DECIDE  (JEV)  every door returns ONE typed record: value, ranked, margin, p_correct, p_null, id, evidence
    VERIFY  (NOOA) only a verified or reported truth becomes a label: decision_outcome(id, truth)
    LEARN   (CLM)  that label moves the door's OWN prototypes by the InfoNCE rule (holographic_protostore)

Faculties here:
    protostore(door)          the named door's ProtoStore (one class, one instance per door)
    door_calibrator(door)     the named door's DoorCalibrator (p_correct: high means confident, at every door)
    door_calibration_report() every door's calibrator: labels, both outcomes seen, calibrated or not
    reflex_correction_mode()  how the reflex trace stores a CORRECTION (E1.5: 'lms_apa' measured default)
    rank(state, candidates)   E3.3 free-form ranking with a floor, p_correct, p_null, a set and a decision id
    verify_precheck(state, actions)  E5.2 verifier prototypes: order / pre-filter actions, never replace a verify

and the Phase-A plumbing the other parts call (private, one home so the 2,000-line parts stay under the cap):
    E0.5  _decision_hook / _systemone_from_key / _decisions_section / _decisions_restore -- the ledger and SystemOne's
          learned tables survive learning_save / learning_load / learning_rollover; hooks RE-DERIVED from the record
    E0.6  _door_of / _door_feed / _door_p -- p_correct from each door's own calibrator, fed by decision_outcome
    E0.7  _ladder_error_prob / _wire_ladder_calibration / _calibration_streams(_restore) -- ladder and bridge are
          two calibrators, and the ladder veto is wired on load (it used to wait for 8 new reports)
    E1.5  _reflex_correct -- the chosen trace-correction design, called by reflex_learn for a correction

Wave 2 (worker F3) adds two doors here:
    rank(state, candidates)        E3.3 -- free-form System One ranking (CLM's "Score/Choice over arbitrary candidates"
                                   done our way): absolute cosine in the hashed-n-gram space, an absolute floor, a
                                   calibrated p_correct, a procedure-matched p_null, a conformal-style set, a ledger id
    verify_precheck(state, actions) E5.2 -- per-tool SUCCESS / FAILURE prototypes learned from swarm_step's verified
                                   results (a failed verify is now a labelled failure example); orders or pre-filters
                                   candidate actions and NEVER replaces the verify command
Both doors persist through the decisions section (their ProtoStores, the rank set registry and conformal scores).
"""
import hashlib
import json
import re

import numpy as np

from holographic.agents_and_reasoning.holographic_protostore import DoorCalibrator, ProtoStore

# The outcome strings that carry no truth (same set reflex_learn uses): never a calibration label.
_FAILED = (None, "", "fail", "failed", "__failed__")

# E1.5 -- THE MEASURED DEFAULT (owner: "build both, let the measurement choose"). tools/bench_trace_correction.py on
# real CLINC keys, 150 trials per cell -> docs/research/evidence/bench_trace_correction.json:
#                      same key   flip-flop   neighbour kept @ c=0.8   replay
#   before (write only)   0.453     0.453          0.900                1.000
#   lms_apa               1.000     1.000          1.000                1.000   <- meets the bar (>=0.95, >=0.85, 1.0)
#   provenance            0.993     0.993          0.780 (FAILS 0.85)   1.000
# 'provenance' stays selectable with reflex_correction_mode('provenance'); docs/TYPED_DECISIONS.md section 8d.
REFLEX_CORRECTION_DEFAULT = "lms_apa"

# ======================================================================================================================
# E3.3 -- mind.rank: the measured operating point. Every number is reproduced by tools/bench_rank.py (CLINC150 test
# questions against the 150 intents, each candidate = the intent name + 5 train examples; docs/research/evidence/
# bench_rank.json). Read the whole block before changing a constant.
# ======================================================================================================================
RANK_DIM = 2048        # hashed_ngram_encode's space at 2048: the space the typed doors and the ledger codec already use
# KEPT LOUD -- the static door only TIES a plain lexical baseline: top-1 0.697 vs TF-IDF word cosine 0.696 (word
# Jaccard 0.477), and TF-IDF abstains a little better (AURC 0.097 vs 0.106 on the margin); the door wins off-scope
# (AUROC 0.859 vs 0.809). What the door adds is LEARNING from reported outcomes (0.690 -> 0.892 on the last third).
# LATENCY (150 candidates = 900 texts, 2 cores at load ~5-6, recorded per arm): warm cache-on p50 10.4 ms / p95 22.1
# (n 1000; the bare ranking 7.5 / 15.4); cache OFF 688 / 935 ms with every n-gram atom resident (n 1000) and 5.2 s at
# the default atom cap (n 50: 19,921 distinct candidate n-grams thrash a 4,096-atom LRU); cold first call 6.9 s p50
# (n 20 fresh minds) -- atom generation (41 us each) dominates. A first run at load ~4.3: 6.5 / 11.6 ms warm, 4.5 s
# cold. The candidate cache is worth ~65x per call.
# A candidate with examples is encoded as the BUNDLE unit(unit(text) + sum unit(example)): top-1 0.697 vs 0.673 for one
# concatenated string. KEPT NEGATIVE: rolecall's option_encode (OPTION(x)name + TEXT(x)examples) scored through its TEXT
# role reads 0.630 -- the OPTION term's crosstalk costs 6.7 points against the same bundle without roles, so the rank
# door does not use it (it is the right form for COMPOSING options, not for ranking them).
RANK_FLOOR = 0.15      # absolute-cosine floor: below it the door REFUSES (value None). On CLINC it keeps 99.8% of the
                       # right answers and refuses 6.3% of off-scope questions (0.20 would keep 99.1% / refuse 23.9%) --
                       # a gross "nothing here" gate, not an off-scope detector (the top score's off-scope AUROC is
                       # 0.859; p_null's 0.831)
RANK_NULL_N = 64       # null draws per (candidate set, length bucket): p resolution 1/65, the catalog's own null size
RANK_SET_MIN = 8       # reported truths before the conformal-style set is returned (as DoorCalibrator's min_count)
RANK_SET_ALPHA = 0.10  # default miscoverage of the conformal-style set
RANK_TOP_KEPT = 16     # top scores kept in a record's meta (the truth's score for the conformal set; outside -> -1)
_RANK_ATOM_CAP = 4096  # n-gram atoms cached (float64, 16 KB each = 67 MB) -- see _bounded_ngram
_RANK_CAND_CAP = 1024  # candidate encodings cached, keyed by sha256 of the candidate text (CLM's "cached actions")
_RANK_NULL_CAP = 32    # (candidate set, length bucket) null pools cached (float32, 64 x 2048 = 0.5 MB each)
_RANK_SETS_CAP = 256   # candidate sets remembered for the outcome path (labels + digests; persisted)
_RANK_TRUTH_WINDOW = 512
_RANK_TEXTS_CAP = 65536  # characters of dict-candidate (text + examples) kept per registered set (the audit's fix: a
                         # restarted mind re-encodes a set from them). CLINC150's 150 intents x (name + 5 examples) is
                         # ~31,000 characters -- one full-size set fits; 256 sets cap the registry at ~16 M characters.
RANK_CONFIDENCE = "margin"   # the score door_calibrator('rank') calibrates: "margin" (measured, below) or "top" (the
                             # backlog's stated rule). ONE setting per process: the calibrator must see one scale.
# p_correct is calibrated on the MARGIN (top - second), not the absolute top score. MEASURED, and it CONTRADICTS the
# backlog's stated rule for this door (kept loud): the panel's "absolute cosine abstains better" (AURC 0.078 vs 0.094
# for a top-2 softmax) was measured on LEARNED meaning prototypes. For free-form candidates the gap ranks correctness
# better at every set size (AURC, CLINC, static prototypes): k=2 margin 0.0015 vs top 0.0029; k=5 0.0056 vs 0.0098;
# k=20 0.0208 vs 0.0314; k=150 0.1064 vs 0.1385 -- and after learning from outcomes 0.0161 vs 0.0322 (3 seeds, sd
# 0.0004 / 0.0020; the calibrated p_correct itself 0.0172, ECE 0.012). A top-2 softmax
# is a monotone map of the margin, so the panel's 0.094 arm IS this arm; on this door it wins. What stays absolute: the
# ORDER (argmax of the cosine), the FLOOR (the top score's own scale answers "is anything here", where it beats both
# the margin and p_null), and p_null. No candidate-relative softmax is ever served as a probability.
_LEN_BUCKETS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32)

# E5.2 -- the verifier pre-check reorders only on STRONG evidence (|score| >= 0.5; score = cos(state, success) -
# cos(state, failure) for the tool). MEASURED on tools/bench_verifier.py's recorded swarm runs (judge calls, 3 seeds,
# 480 draws, the same 425 solved in every arm): router order 1469 with the reflex answering repeats / 2691 without;
# reorder by the raw score 1583 (+7.8%, KEPT NEGATIVE: every tool that ever failed anywhere reads negative on any
# English state, so the router's good first guess is demoted on a NEW task) / 1869; strong-evidence rule at 0.4-0.8
# 1469 (no cost) / 1755 (-34.8%). 0.5 sits in the middle of that flat region.
# THE ACCEPTANCE (AUROC at telling a step that will pass from one that will fail, vs the reflex's best signal --
# verify_decision's forward margin -- picked post hoc from five): on the recorded run stream (each step scored before
# its verify, then learned) 0.951 vs 0.924 (ECE 0.045; router rank 0.615): PASSES. On held-out WORDINGS (hash split,
# 5 salts) 0.706 +- 0.097 vs 0.675 +- 0.056, 3 of 5 salts -- within noise. KEPT NEGATIVE: on NOVEL TASKS (both
# wordings held out) 0.545 +- 0.027 vs 0.603 +- 0.058 -- the prototypes know tools on tasks they have seen, nothing
# more; there the router's own order (0.604) is the better guess, which is why the order moves only on strong evidence.
VERIFY_STRONG = 0.5


def _bounded_ngram(dim=RANK_DIM, lo=3, hi=5, cap=_RANK_ATOM_CAP):
    """hashed_ngram_encode's space (holographic_systemone) with a PRIVATE bounded atom cache of `cap` atoms. Returns
    enc(text) -> vector, with enc.cache (the LRU dict), enc.stats ({"misses": n}) and enc.cap attached.

    WHY a private cap for the rank door: hashed_ngram_encode cached one float64 atom per distinct n-gram FOREVER.
    Measured on CLINC150: the test + oos questions alone hold 54,426 distinct n-grams = 891 MB of atoms (train adds up
    to 1.43 GB) -- a ranking service would grow until the memory cgroup killed it. A miss regenerates the atom from its
    seed (41 us measured), so a result never depends on what happens to be cached.
    ONE implementation (2026-09-27): this used to be a private copy of the LRU, and the duplication audit
    (tests/test_duplication_audit.py) flagged the copy once holographic_systemone grew its own capped cache; it now
    delegates to hashed_ngram_encode(cap=...), which gives an encoder a private LRU with the same vectors bit for bit
    (verified: identical encodings, misses and cache size at cap=8)."""
    from holographic.agents_and_reasoning.holographic_systemone import hashed_ngram_encode
    return hashed_ngram_encode(dim=int(dim), lo=int(lo), hi=int(hi), cap=int(cap))


def _unit_vec(v):
    """Unit-normalise (float64); a zero vector stays zero."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _len_bucket(text):
    """The null's length bucket for a state: its word count, exact up to 8, then 10 / 12 / 16 / 24 / 32 (capped)."""
    n = max(1, len(re.findall(r"\S+", str(text))))
    for b in _LEN_BUCKETS:
        if n <= b:
            return b
    return _LEN_BUCKETS[-1]


def _rank_candidates(candidates):
    """Normalise rank()'s candidates -> [(label, text, examples, digest)]. A candidate is a string (its label is the
    text) or a dict {text, id?, examples?}. The digest -- the encoding cache key -- is sha256 of the candidate TEXT (of
    the JSON {text, examples} when it has examples). Labels must be unique; nothing may be empty."""
    if isinstance(candidates, (str, bytes, dict)) or candidates is None:
        raise TypeError("rank needs a LIST of candidates (strings or {text, id?, examples?} dicts)")
    out, seen = [], set()
    for c in candidates:
        if isinstance(c, str):
            text, label, ex = c, c, ()
        elif isinstance(c, dict):
            text = c.get("text")
            label = str(c["id"]) if c.get("id") is not None else text
            ex = tuple(c.get("examples") or ())
        else:
            raise TypeError("a rank candidate is a string or a {text, id?, examples?} dict, got %r" % type(c).__name__)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("a rank candidate needs non-empty text, got %r" % (c,))
        if not all(isinstance(e, str) for e in ex):
            raise ValueError("candidate %r: examples must be strings" % label)
        if label in seen:
            raise ValueError("duplicate candidate label %r (give each an 'id')" % label)
        seen.add(label)
        key = json.dumps({"text": text, "examples": list(ex)}, sort_keys=True) if ex else text
        out.append((label, text, ex, hashlib.sha256(key.encode("utf-8")).hexdigest()))
    if not out:
        raise ValueError("rank needs at least one candidate")
    return out


class _UnifiedPart29:

    def protostore(self, door, dim=2048, tau=0.05, lr=0.3, topk=None):
        """The named door's contrastive prototype store (created on first use): per-option unit prototypes that
        every labelled verdict moves by lr*(t - p)*q. One class, one instance per door -- doors never share."""
        stores = self.__dict__.setdefault("_protostores", {})
        st = stores.get(door)
        if st is None:
            st = ProtoStore(dim, tau=tau, lr=lr, topk=topk, name=door)
            stores[door] = st
        return st

    def door_calibrator(self, door, min_count=8):
        """The named door's calibrator: P(correct | that door's score), fed only by that door's outcomes.
        p_correct() is None until it has min_count labels with both outcomes."""
        cals = self.__dict__.setdefault("_door_calibrators", {})
        c = cals.get(door)
        if c is None:
            c = DoorCalibrator(door, min_count=min_count)
            cals[door] = c
        return c

    def door_calibration_report(self):
        """Every door calibrator this mind holds, as {door: {labels, correct, calibrated}} -- which doors can already
        say p_correct (high = confident) and which still return None because they lack labelled outcomes."""
        out = {}
        for door, c in sorted(self.__dict__.get("_door_calibrators", {}).items()):
            ys = [y for _, y in c.pairs]
            out[door] = {"labels": len(ys), "correct": int(sum(ys)), "calibrated": c.calibrated()}
        return out

    def reflex_correction_mode(self, design=None):
        """Get (design=None) or set how the reflex trace stores a CORRECTION: 'lms_apa' (the measured default --
        codebook LMS on an affine-projected key) or 'provenance' (signed along-atom / outcome-role negative).
        Returns the design now in force. See holographic_lever7.correct_lms_apa / correct_by_provenance."""
        from holographic.agents_and_reasoning.holographic_lever7 import CORRECTION_DESIGNS
        if design is not None:
            if design not in CORRECTION_DESIGNS:
                raise ValueError("reflex correction design must be one of %s, got %r" % (CORRECTION_DESIGNS, design))
            self._reflex_correction = design
        return getattr(self, "_reflex_correction", REFLEX_CORRECTION_DEFAULT)

    # ================================================================================================================
    # E3.3 -- mind.rank(state, candidates): free-form System One ranking, our way.
    # ================================================================================================================
    def rank(self, state, candidates, floor=None, k=None, learn=True, cache=True, record=True, null=True,
             alpha=RANK_SET_ALPHA):
        """Rank free-form candidates against a state and return ONE typed record: value, ranked, p_correct, p_null, id.

        candidates = strings, or dicts {"text": ..., "id": optional label, "examples": optional [texts]}. State and
        candidates are encoded in ONE space (hashed character 3..5-grams, dim 2048 -- the typed doors' space); the
        order is the ABSOLUTE cosine (ties keep the given order). Below `floor` (default 0.15) the door refuses:
        value None, abstained True -- the ranking is still returned. Report the truth with decision_outcome(id, label):
        it feeds door_calibrator("rank") (p_correct stops being None after 8 labels of both kinds), the rank door's
        ProtoStore (the InfoNCE rule over THIS call's candidates; learn=True scores learned rows -- measured prequential
        top-1 0.892 vs 0.690 static on CLINC150's last third, 3 seeds) and the conformal-style `set` (after 8 truths;
        measured coverage 0.902 at alpha 0.1, mean size 1.74).

        Returns {value, ranked [(label, cosine)] best first (k of them), margin, top, p_correct (calibrated P(correct)
        from the MARGIN, high = confident), p_null (the top score's p-value against in-vocabulary word salad of the
        state's length, ranked by the same procedure; low = significant), set, id, abstained, why, via='rank',
        evidence}. cache=False re-encodes every candidate (the latency arm); results are identical either way.
        Measured latency and quality: tools/bench_rank.py -> docs/research/evidence/bench_rank.json."""
        from holographic.agents_and_reasoning.holographic_decisionrecord import DecisionRecord, door_record
        if not isinstance(state, str) or not state.strip():
            raise ValueError("rank needs a non-empty text state")
        cands = _rank_candidates(candidates)
        floor = RANK_FLOOR if floor is None else float(floor)
        rs = self._rank_state()
        labels = [c[0] for c in cands]
        digests = [c[3] for c in cands]
        set_digest = hashlib.sha256(json.dumps(sorted([l, d] for l, d in zip(labels, digests))).encode()).hexdigest()[:24]
        M, info = self._rank_matrix(rs, cands, set_digest, learn=learn, cache=cache)
        q = _unit_vec(rs["enc"](state))
        s = M @ q
        order = np.argsort(-s, kind="stable")                       # the one tie rule: the given order
        top = float(s[order[0]])
        second = float(s[order[1]]) if len(order) > 1 else 0.0
        margin = top - second
        score = margin if RANK_CONFIDENCE == "margin" else top      # the calibrated score (see the RANK constants)
        value = labels[int(order[0])] if top >= floor else None
        p_null, null_info = None, None
        if null:
            p_null, null_info = self._rank_p_null(rs, cands, set_digest, state, M, info["stamp"], top)
        p_correct = self._door_p("rank", score)
        conf_set = self._rank_set(rs, labels, s, order, alpha)
        ranked = [(labels[int(j)], float(s[int(j)])) for j in order]
        why = ("top %.3f clears the floor %.2f, margin %.3f" % (top, floor, margin)) if value is not None else \
              ("top %.3f is under the absolute floor %.2f: nothing here fits -- refused" % (top, floor))
        rid = None
        if record:
            # the record's `margin` field carries the CALIBRATED score (the rank door's _door_of reads it back)
            rec = DecisionRecord(state, "rank", [l for l, _ in ranked], value, "rank", margin=score, p=p_correct,
                                 p_correct=p_correct, p_null=p_null,
                                 meta={"hook": {"kind": "rank"}, "set": set_digest, "floor": floor, "learn": bool(learn),
                                       "top": [round(x, 6) for _, x in ranked[:RANK_TOP_KEPT]]})
            rid = self.decision_ledger().add(rec)
            self._rank_register(rs, set_digest, labels, digests, cands)
        rs["calls"] += 1
        evidence = {"encoder": "hashed_ngram", "dim": RANK_DIM, "floor": floor, "top": top, "second": second,
                    "n_candidates": len(cands), "learned_rows": info["learned_rows"],
                    "cache": {"on": bool(cache), "hits": info["hits"], "encoded": info["encoded"]},
                    "null": null_info, "p_correct_from": RANK_CONFIDENCE}
        return door_record({"value": value, "ranked": ranked[:int(k)] if k else ranked, "margin": margin, "top": top,
                            "set": conf_set, "id": rid, "abstained": value is None, "why": why, "via": "rank",
                            "evidence": evidence}, "rank", p_correct=p_correct, p_null=p_null)

    # ================================================================================================================
    # E5.2 -- VERIFIER PROTOTYPES: a cheap pre-check that orders or pre-filters actions, never a substitute for verify.
    # ================================================================================================================
    def verify_precheck(self, state, actions, min_p=None, strong=VERIFY_STRONG):
        """Order candidate actions by learned verify SUCCESS vs FAILURE prototypes -- a pre-check, never a verify.

        Each tool has its own pair of prototypes (a two-option ProtoStore 'verify:<tool>') learned from swarm_step:
        a step whose verify PASSED is a success example of its state, and -- new in wave 2 -- a step whose verify
        FAILED is a labelled failure example (before, a refused step taught nothing). score = cos(state, success) -
        cos(state, failure); an unseen tool scores 0. The ORDER acts only on STRONG evidence: actions scoring >= strong
        (0.5) move to the front (best first), actions scoring <= -strong move to the back, everything else keeps the
        given order (normally the router's) -- measured: reordering by the raw score cost 7.8% MORE judge calls in the
        swarm with the reflex answering repeats (1583 vs 1469), the strong-evidence rule costs nothing there (1469)
        and saves 34.8% without a reflex (1755 vs 2691). strong=None reorders by the raw score (that kept arm).
        p_correct comes from door_calibrator("verify"), fed with the score each step had BEFORE its verify ran.
        min_p (optional) PRE-FILTERS: actions whose calibrated p_correct is under it are dropped -- only once the door
        is calibrated, and never the last action (measured: min_p 0.05 dropped the truth on 19 of 480 draws, so it is
        off by default). THIS NEVER SKIPS A VERIFY: swarm_step still runs the verify command on every step it accepts;
        the pre-check only changes which candidate is tried first. Returns {order, dropped, scores, p_correct,
        calibrated, verify_still_runs: True, why}. Measured (tools/bench_verifier.py -> bench_verifier.json): AUROC
        0.951 vs the reflex's 0.924 on the recorded run stream; no better than chance on NOVEL tasks (0.545, kept
        negative) -- it knows tools on tasks it has seen."""
        acts = [str(a) for a in actions]
        if not acts:
            return {"order": [], "dropped": [], "scores": {}, "p_correct": {}, "calibrated": False,
                    "verify_still_runs": True, "why": "no actions"}
        q = self._verifier_vec(state)
        scores = {a: self._verifier_score(q, a) for a in acts}
        cal = self.door_calibrator("verify")
        pcs = {a: cal.p_correct(scores[a]) for a in acts}
        by_score = [acts[int(j)] for j in np.argsort(-np.array([scores[a] for a in acts]), kind="stable")]
        if strong is None:
            order = by_score
        else:
            s_ = float(strong)
            order = ([a for a in by_score if scores[a] >= s_] + [a for a in acts if -s_ < scores[a] < s_]
                     + [a for a in by_score if scores[a] <= -s_])
        dropped = []
        if min_p is not None and cal.calibrated():
            keep = [a for a in order if pcs[a] is None or pcs[a] >= float(min_p)]
            if not keep:
                keep = order[:1]                        # a pre-filter never empties the list: the best stays
            dropped = [a for a in order if a not in keep]
            order = keep
        seen = sum(1 for a in acts if ("verify:" + a) in self.__dict__.get("_protostores", {}))
        why = "%d of %d actions have verify experience; ordered by success-minus-failure cosine" % (seen, len(acts))
        return {"order": order, "dropped": dropped, "scores": scores, "p_correct": pcs, "calibrated": cal.calibrated(),
                "verify_still_runs": True, "why": why}

    # ================================================================================================================
    # E0.6 -- ONE MEANING OF p. p_correct comes from the door's OWN calibrator, fed by that door's reported outcomes.
    # ================================================================================================================
    def _door_of(self, rec):
        """(door, score, correct) for a reported record whose door calibrates here, else None.
          route   route_tiered's catalog door: score = z; the scored candidate is the answer, or the TOP option of a
                  menu (the menu's p_correct then reads "P(the top option is the one you used | z)" -- exactly what
                  deciding answer-vs-menu needs; a menu is never 'wrong' just because it did not pick)
          bridge  the reflex bridge (via 'reflex'): score = the reflex confidence
          tool:*  serve()'s tool reflex: score = word overlap ('tool:overlap') or the experience confidence
                  ('tool:experience') -- two scales, so two calibrators (one door, one scale)
        typed / model_end calibrate inside their own SystemOne (isotonic, per question); meaning inside its
        MeaningIndex -- both already per door. A failed/empty outcome carries no truth and is not a label."""
        if rec is None or rec.outcome in _FAILED or rec.margin is None:
            return None
        from holographic.agents_and_reasoning.holographic_decisionrecord import outcome_matches
        if rec.via == "route" and rec.question == "route":
            cand = rec.answer if rec.answer is not None else (rec.options[0] if rec.options else None)
            if cand is None:
                return None
            return "route", float(rec.margin), outcome_matches(cand, rec.outcome)
        if rec.via == "reflex":
            return "bridge", float(rec.margin), outcome_matches(rec.answer, rec.outcome)
        if rec.via == "tool":
            return ("tool:" + str(rec.meta.get("picked_by", "overlap")), float(rec.margin),
                    outcome_matches(rec.answer, rec.outcome))
        if rec.via == "rank":
            # E3.3: score = the margin (see the RANK constants: measured to rank correctness better than the top
            # score); like a route menu, a refused rank is scored on its TOP candidate -- "would the top have been right"
            cand = rec.answer if rec.answer is not None else (rec.options[0] if rec.options else None)
            if cand is None:
                return None
            return "rank", float(rec.margin), outcome_matches(cand, rec.outcome)
        return None

    def _door_feed(self, rec):
        """Feed one reported record into its door's calibrator (decision_outcome calls this). -> door or None."""
        d = self._door_of(rec)
        if d is None:
            return None
        door, score, correct = d
        self.door_calibrator(door).observe(score, bool(correct))
        return door

    def _door_p(self, door, score):
        """p_correct for a score at a door: None when the door is uncalibrated or the score is missing."""
        if score is None:
            return None
        cals = self.__dict__.get("_door_calibrators", {})
        c = cals.get(door)
        return None if c is None else c.p_correct(float(score))

    # ================================================================================================================
    # E0.7 -- CALIBRATION STREAMS PER DOOR. The ladder's T0 reflex (answer_feedback on the _qkey confidence) and the
    # reflex bridge (reflex_learn on bridge confidences) used to feed ONE list and ONE isotonic curve, which then
    # drove BOTH the ladder's calib_veto and the bridge's p -- two score distributions forced through one map, so
    # every bridge outcome moved the ladder's veto. Now: door_calibrator('ladder') and door_calibrator('bridge').
    # ================================================================================================================
    def _ladder_error_prob(self, confidence):
        """The ladder's calibrated error probability for a T0 confidence (1 - p_correct), None while uncalibrated --
        the fixed gate then stands alone. This is what AnswerLadder._error_prob calls (calib_veto above 0.5)."""
        p = self._door_p("ladder", confidence)
        return None if p is None else 1.0 - p

    def _wire_ladder_calibration(self):
        """Point the ladder's veto at the LADDER calibrator. Idempotent. Called on every ladder feed, by
        calibrate_reflex, and on learning_load -- before, the veto was wired only inside calibrate_reflex, so after a
        restart it stayed OFF until 8 new reports arrived (found by the panel, w3)."""
        try:
            self.zoo["ladder"]._error_prob = self._ladder_error_prob
        except Exception:
            pass

    def _calibration_streams(self):
        """{door: DoorCalibrator.state()} for every door calibrator (the calibration section's v2 payload)."""
        return {door: c.state() for door, c in sorted(self.__dict__.get("_door_calibrators", {}).items())}

    def _calibration_restore(self, meta):
        """Restore the calibration section. v2 carries per-door `streams`; a LEGACY partition carries one untagged
        `pairs` list fed by both the ladder and the bridge. Nobody recorded which door wrote which pair, so the
        legacy list is MIGRATED INTO BOTH streams (documented: each door starts from the shared history it used to
        be calibrated on, and its own new labels take over through the calibrator's window). -> report."""
        meta = meta or {}
        cals = self.__dict__.setdefault("_door_calibrators", {})
        streams = meta.get("streams")
        if streams:
            for door, st in sorted(streams.items()):
                cals[door] = DoorCalibrator.from_state(st)
            out = {"format": "v2", "doors": sorted(streams)}
        else:
            legacy = [(float(c), bool(o)) for c, o in (meta.get("pairs") or [])]
            for door in ("ladder", "bridge"):
                cal = self.door_calibrator(door)
                for c, o in legacy:
                    cal.observe(c, o, refit=False)
                cal.refit()
            out = {"format": "legacy", "migrated_pairs": len(legacy), "into": ["ladder", "bridge"]}
        self._wire_ladder_calibration()                 # THE FIX: the veto is live from the moment of the load
        return out

    # ================================================================================================================
    # E0.5 -- PERSIST DECISIONS. The ledger travels as TEXT; hooks are re-derived from rec.meta["hook"].
    # ================================================================================================================
    def _systemone_from_key(self, key):
        """The cached SystemOne for a _systemone_cached key -- rebuilt (fit from the schema in the key) when this
        process never made it, e.g. after a reload. Only string encoders ('perceive' / 'ngram') are portable: a
        callable's key is its repr, which names a memory address, so it refuses with ValueError."""
        cache = self.__dict__.setdefault("_systemone_cache", {})
        so = cache.get(key)
        if so is not None:
            return so
        p = json.loads(key)
        if p.get("e") not in ("perceive", "ngram"):
            raise ValueError("SystemOne cache key uses a non-portable encoder %r" % (p.get("e"),))
        so = self._systemone_cached(p["q"], p.get("l"), p.get("m"), p["e"], p.get("s"), p.get("sc", "prototype"),
                                    p.get("b", False), conformal_alpha=p.get("ca"))
        cache.setdefault(key, so)                       # the re-derived key is identical; an alias costs nothing
        return so

    def _decision_hook(self, rec):
        """The Ledger's resolve_hook: rebuild the callable a record's outcome forwards to, FROM THE RECORD.
            {"kind": "systemone", "key": <cache key>, "q": <question>}  -> that SystemOne's observe(state, {q: outcome})
            {"kind": "meaning"}                                        -> the meaning door's calibration + link
            {"kind": "rank"}                                           -> the rank door's conformal stream + ProtoStore
            {"kind": "calibrate", "door", "score", "answer"}           -> door_calibrator(door).observe(score, right)
            {"kind": "router"}                                         -> the learned router (a route record)
            {"kind": "tooldoor"}                                       -> the tool door's ProtoStore (a tool record)
        A record may carry SEVERAL specs (a list) once two doors filed one decision under one id; the Ledger resolves
        each through this function with a view whose meta["hook"] is that one spec (wave 2, Ledger.add combines).
        Both refuse to learn a state/wording that carries a secret (the learning guard's pattern layer): the typed
        door's NB table would otherwise count a pasted password's tokens, and the table is persisted."""
        h = rec.meta.get("hook") or {}
        kind = h.get("kind")
        if kind == "systemone":
            so = self._systemone_from_key(h["key"])
            q, state = h.get("q", rec.question), rec.state

            def hook(outcome):
                from holographic.agents_and_reasoning.holographic_learnguard import sensitive_reason
                why = sensitive_reason(state) if isinstance(state, str) else None
                if why:
                    return {"learned": False, "why": why}
                return so.observe(state, {q: outcome}).get(q)
            return hook
        if kind == "meaning":
            return lambda outcome: self._meaning_outcome(rec, outcome)
        if kind == "rank":
            # E3.3: the conformal stream + the rank door's ProtoStore (the calibrator is fed by _door_feed)
            return lambda outcome: self._rank_outcome(rec, outcome)
        if kind == "router":
            # LEARNING-LOOP AUDIT: route_tiered's router hook, rebuilt FROM THE RECORD after a restart (it was a live
            # closure over (problem, options) -- exactly rec.state and rec.options). _router_outcome refuses a
            # question carrying a secret and an outcome that names no catalog card, as before.
            state, options = rec.state, tuple(rec.options or ())
            return lambda outcome: self._router_outcome(state, options, outcome)
        if kind == "tooldoor":
            # LEARNING-LOOP AUDIT: serve()'s tool-door hook, rebuilt from the record (query = rec.state, the tool it
            # called = rec.answer). _tool_outcome teaches nothing for a failed/empty outcome and never a secret.
            state, tool = rec.state, rec.answer
            return lambda outcome: self._tool_outcome(state, tool, outcome)
        if kind == "calibrate":
            # Wave 2: a RESTART-PROOF calibration hook for any door that only needs "(score, answer == outcome)" fed
            # to its own calibrator -- {"kind": "calibrate", "door": name, "score": s, "answer": a}. A live closure
            # (the CLM plugin's and call_compose's today) dies with the process; this spec is re-derived on load.
            door, score, ans = h.get("door"), h.get("score"), h.get("answer", rec.answer)
            if not door or score is None:
                return None

            def calibrate(outcome):
                from holographic.agents_and_reasoning.holographic_decisionrecord import outcome_matches
                if outcome in _FAILED:
                    return {"calibrated": False, "why": "no truth reported"}
                cal = self.door_calibrator(str(door))
                cal.observe(float(score), outcome_matches(ans, outcome))
                return {"door": cal.door, "calibrated": cal.calibrated(), "n": len(cal.pairs)}
            return calibrate
        return None

    def _meaning_outcome(self, rec, outcome):
        """The meaning door's hook, re-derived from its record (sweep 181's closure, now restart-proof): the outcome
        is a calibration label for the confidence it served at; a correction naming another row teaches that wording
        to that row (never a wording that carries a secret) and blocks this wording from the wrong row."""
        from holographic.agents_and_reasoning.holographic_learnguard import sensitive_reason
        mi = self.meaning
        q, a = str(rec.state), rec.answer
        ok = (outcome == a)
        mi.observe(float(rec.margin or 0.0), ok)
        if not ok:
            self._meaning_neg_add(q, a)
            if outcome in mi.rows and not sensitive_reason(q):
                mi.link(outcome, q, learn=True)
        return {"meaning": "confirmed" if ok else "corrected", "row": outcome if not ok else a}

    def _decisions_section(self):
        """The `lecore.learning.decisions` container section (or None when there is nothing to keep):
        records as text (newest Ledger.CAP; secrets never written -- counted) + every LEARNED SystemOne in the
        cache (one that only ever decided is re-derivable from its key by fit, so it is not stored: lever 3)."""
        L = getattr(self, "_decision_ledger", None)
        txt = L.to_text() if L is not None else {"records": [], "dropped_sensitive": 0, "dropped_cap": 0}
        so_rows, arrays, skipped = [], {}, 0
        for key, so in sorted((self.__dict__.get("_systemone_cache") or {}).items()):
            if not any(so._outcomes.values()) and not getattr(so, "_learned", False):
                continue                                # never learned: a fit from the key reproduces it exactly
            try:
                if json.loads(key).get("e") not in ("perceive", "ngram"):
                    skipped += 1                        # a callable encoder cannot be rebuilt in another process
                    continue
                meta, arr = so.state()
            except Exception:
                skipped += 1
                continue
            i = len(so_rows)
            so_rows.append({"key": key, "state": meta})
            for name, a in arr.items():
                arrays["so%d_%s" % (i, name)] = a
        # WAVE 2: the rank door and the verifier learn from outcomes too, so their state travels in this section:
        # their ProtoStores ('rank', 'verify:<tool>'), the rank door's candidate-set registry (the outcome path of a
        # record issued before a restart needs it) and its conformal truth scores.
        # LEARNING-LOOP AUDIT (2026-09-26): EVERY door's store that no other section carries travels here, not just
        # 'rank' and 'verify:<tool>'. The router ('route') and the tool door ('tool') have sections of their own; any
        # other store in mind._protostores -- a superposed_door's prototypes, a door merged in from the commons by
        # memory_import (_protostore_merge creates it under the donor's door name) -- was learned and then lost at the
        # next restart. The guard that keeps a secret out of a store is the door's own (it refuses the verdict).
        ps_rows = []
        for name, st in sorted((self.__dict__.get("_protostores") or {}).items()):
            if name in ("route", "tool") or st is None or not len(st):
                continue
            m_, a_ = st.state(dtype="float64")
            # float64, not ProtoStore.state()'s float32: a rank after a cold reload must be BIT-IDENTICAL to the one
            # before it (pinned in tests/test_rank.py; float32 rows moved the 10th decimal and broke that).
            # The unit rows P travel too (CI, py3.12, 2026-09-27): recomputing P from A on load moved a verify score by
            # one ulp (0.008086684778032812 -> ...809) -- a row seeded by add_option and a row moved by update are
            # normalised by two different norm computations, and which one differs in the last bit depends on the
            # numpy/BLAS build. ProtoStore.state(dtype="float64") stores both matrices; see its docstring.
            k_ = len(ps_rows)
            arrays["ps%d_A" % k_] = a_["A"]
            arrays["ps%d_P" % k_] = a_["P"]
            ps_rows.append({"name": name, "state": m_})
        rs = self.__dict__.get("_rank_door")
        rank_meta = {"sets": rs["sets"], "truth": rs["truth"]} if rs and (rs["sets"] or rs["truth"]) else None
        # the superposed doors' per-channel relevances (C floats each; holographic_superposed.ChannelRelevance) -- the
        # other half of a superposed_door's learned state, lost at a restart before the audit
        rel = {d: r.state() for d, r in sorted((self.__dict__.get("_superposed_relevances") or {}).items())
               if getattr(r, "n", 0)}
        if not txt["records"] and not so_rows and not ps_rows and rank_meta is None and not rel:
            return None
        meta = {"records": txt["records"], "dropped_sensitive": txt["dropped_sensitive"],
                "dropped_cap": txt["dropped_cap"], "systemone": so_rows, "systemone_skipped": skipped}
        if ps_rows:
            meta["protostores"] = ps_rows
        if rank_meta is not None:
            meta["rank"] = rank_meta
        if rel:
            meta["relevances"] = rel
        return {"kind": "lecore.learning.decisions", "id": "v1", "meta": meta, "arrays": arrays}

    def _decisions_restore(self, sec, merge=True):
        """Inverse of _decisions_section: SystemOne learned tables FIRST (so a reloaded record's hook lands in the
        learned door, not a fresh fit), then the records. -> {records: Ledger.load_text report, systemone: n, ...}."""
        if not sec:
            return None
        meta, arrays = sec.get("meta") or {}, sec.get("arrays") or {}
        restored, refused = 0, []
        for i, row in enumerate(meta.get("systemone") or []):
            pre = "so%d_" % i
            arr = {n[len(pre):]: a for n, a in arrays.items() if n.startswith(pre)}
            try:
                so = self._systemone_from_key(row["key"])
                so.load_state(row["state"], arr)
                so._learned = True                      # stays persisted at the next save (absorbed tables too)
                restored += 1
            except Exception as e:                      # a schema/encoder that no longer fits: refit fresh, say so
                refused.append("%s: %s" % (type(e).__name__, str(e)[:80]))
        # wave 2: the rank / verifier ProtoStores and the rank door's registry + conformal stream
        stores = self.__dict__.setdefault("_protostores", {})
        n_ps = 0
        for i, row in enumerate(meta.get("protostores") or []):
            A = arrays.get("ps%d_A" % i)
            if A is None:
                continue
            try:
                # ps%d_P is absent in partitions written before 2026-09-27: from_state then recomputes the unit rows
                stores[row["name"]] = ProtoStore.from_state(row["state"], {"A": A, "P": arrays.get("ps%d_P" % i)})
                n_ps += 1
            except Exception as e:
                refused.append("protostore %s: %s" % (row.get("name"), str(e)[:80]))
        if meta.get("relevances"):
            from holographic.agents_and_reasoning.holographic_superposed import ChannelRelevance
            rels = self.__dict__.setdefault("_superposed_relevances", {})
            for d, st_ in sorted(meta["relevances"].items()):
                try:
                    rels[d] = ChannelRelevance.from_state(st_)
                except Exception as e:
                    refused.append("relevance %s: %s" % (d, str(e)[:80]))
        rk = meta.get("rank")
        if rk:
            rs = self._rank_state()
            for sd, reg in (rk.get("sets") or {}).items():
                if sd not in rs["sets"]:
                    rs["sets"][sd] = {"labels": list(reg["labels"]), "digests": list(reg["digests"])}
                    if reg.get("texts"):                # the audit's fix: a restarted mind can re-encode the set
                        rs["sets"][sd]["texts"] = {d: [t, list(ex)] for d, (t, ex) in reg["texts"].items()}
                    if reg.get("texts_dropped"):
                        rs["sets"][sd]["texts_dropped"] = True
            if not rs["truth"]:
                rs["truth"] = [float(x) for x in (rk.get("truth") or [])][-_RANK_TRUTH_WINDOW:]
            rs["mats"].clear()                          # learned rows changed under any cached matrix
        rep = self.decision_ledger().load_text(meta.get("records") or [], merge=merge)
        return {"records": rep, "systemone": restored, "systemone_refused": refused, "protostores": n_ps,
                "dropped_sensitive_at_save": meta.get("dropped_sensitive", 0)}

    # ================================================================================================================
    # E1.5 -- the chosen trace-correction design (reflex_learn calls this for a correction).
    # ================================================================================================================
    def _reflex_neg_role(self):
        """The outcome role design (b) binds a rejected label under (unitary, derived from its name)."""
        from holographic.mesh_and_geometry.holographic_planshape import derived_atom
        return derived_atom(0, "role:outcome_neg", int(self.experience.dim), unitary=True)

    def _reflex_correct(self, rec, tv, kind):
        """Store a CORRECTION (the served answer was wrong, rec.outcome is the truth) in the experience trace by the
        design in force (reflex_correction_mode). -> the correction's diagnostics."""
        from holographic.agents_and_reasoning.holographic_lever7 import correct_by_provenance, correct_lms_apa
        from holographic.mesh_and_geometry.holographic_planshape import derived_atom
        design = self.reflex_correction_mode()
        truth_atom = self._reflex_label_atom(rec.outcome)
        if design == "provenance":
            wrong = derived_atom(0, "answer:" + str(rec.answer), int(self.experience.dim))
            return correct_by_provenance(self.experience, tv, truth_atom, wrong, self._reflex_neg_role())
        seen = [(k, o) for kd, k, o in (getattr(self, "_reflex_seen", None) or []) if kd == kind]
        return correct_lms_apa(self.experience, tv, rec.outcome, dict(self._reflex_labels), seen=seen)

    # ================================================================================================================
    # E3.3 plumbing -- the rank door's caches, null, conformal set and outcome path.
    # ================================================================================================================
    def _rank_state(self):
        """The rank door's process state: the bounded encoder, the candidate-encoding cache (sha256 of the text ->
        unit vector), the candidate-set registry (for the outcome path; persisted), the null pools, the matrix
        cache, and the conformal truth scores (persisted)."""
        rs = self.__dict__.get("_rank_door")
        if rs is None:
            rs = self.__dict__["_rank_door"] = {"enc": _bounded_ngram(), "cands": {}, "sets": {}, "nulls": {},
                                                "mats": {}, "truth": [], "calls": 0}
        return rs

    def _rank_encode(self, rs, text, examples):
        """A candidate's unit vector: its text, or the BUNDLE unit(text) + sum unit(example) (measured 0.697 vs one
        concatenated string 0.673 top-1 on CLINC150)."""
        v = _unit_vec(rs["enc"](text))
        if examples:
            v = _unit_vec(v + sum(_unit_vec(rs["enc"](e)) for e in examples))
        return v

    def _rank_store_stamp(self):
        """A version stamp of the rank door's learned rows: any outcome that moved a row changes it."""
        st = self.__dict__.get("_protostores", {}).get("rank")
        return (0, 0) if st is None else (len(st), st.n_updates)

    def _rank_matrix(self, rs, cands, set_digest, learn=True, cache=True):
        """The (n, dim) matrix the state is scored against: each candidate's LEARNED prototype when the rank door's
        ProtoStore holds one (learn=True), else its text encoding (cached by sha256 of the text when cache=True).
        -> (matrix, {stamp, learned_rows, hits, encoded})."""
        stamp = (bool(learn),) + self._rank_store_stamp()
        # the matrix cache is keyed by the candidates IN THE GIVEN ORDER (the set digest is order-free: the same set
        # passed in another order must not reuse rows stacked in the old order)
        mkey = hashlib.sha256(json.dumps([[c[0], c[3]] for c in cands]).encode()).hexdigest()[:24]
        if cache:
            hit = rs["mats"].get(mkey)
            if hit is not None and hit[0] == stamp:
                return hit[1], {"stamp": stamp, "learned_rows": hit[2], "hits": len(cands), "encoded": 0}
        st = self.__dict__.get("_protostores", {}).get("rank") if learn else None
        rows, hits, encoded, learned = [], 0, 0, 0
        for label, text, ex, d in cands:
            if st is not None and d in st:
                rows.append(st.P[st.index(d)])
                learned += 1
                continue
            v = rs["cands"].get(d) if cache else None
            if v is None:
                v = self._rank_encode(rs, text, ex)
                encoded += 1
                if cache:
                    if len(rs["cands"]) >= _RANK_CAND_CAP:
                        rs["cands"].pop(next(iter(rs["cands"])))     # FIFO: the oldest encoding goes first
                    rs["cands"][d] = v
            else:
                hits += 1
            rows.append(v)
        M = np.stack(rows)
        if cache:
            if mkey not in rs["mats"] and len(rs["mats"]) >= 32:
                rs["mats"].pop(next(iter(rs["mats"])))
            rs["mats"][mkey] = (stamp, M, learned)
        # the raw encodings are also what the outcome path needs to learn this set: keep them reachable even with
        # cache=False (the latency arm), or a reported outcome could not move a row the process never cached
        if not cache:
            for (label, text, ex, d), v in zip(cands, rows):
                if st is None or d not in st:
                    rs.setdefault("pending", {})[d] = v
            while len(rs.get("pending", {})) > _RANK_CAND_CAP:
                rs["pending"].pop(next(iter(rs["pending"])))
        return M, {"stamp": stamp, "learned_rows": learned, "hits": hits, "encoded": encoded}

    def _rank_p_null(self, rs, cands, set_digest, state, M, stamp, top):
        """p_null: the top score's p-value against RANK_NULL_N in-vocabulary word-salad states -- words drawn from the
        candidates' own texts, as many as the state has (bucketed) -- encoded and scored by EXACTLY this procedure
        (the same matrix, learned rows included). The salad pool is cached per (candidate set, length bucket); its
        top scores are recomputed only when the learned rows change. p = (1 + #null tops >= top) / (N + 1).
        MEASURED on CLINC (tools/bench_rank.py): off-scope AUROC 0.831 against the raw top score's 0.859 at 150
        candidates (0.980 vs 0.982 at 3, 0.947 vs 0.945 at 10) -- a stated significance, not a better off-scope
        detector (kept loud). Significant at 0.05: 70% of in-scope vs 13.5% of off-scope questions at 150."""
        b = _len_bucket(state)
        key = (set_digest, b)
        ent = rs["nulls"].get(key)
        built = False
        if ent is None:
            vocab = sorted({w for _, text, ex, _ in cands for t in (text,) + tuple(ex)
                            for w in re.findall(r"[a-z0-9']+", t.lower())})
            if not vocab:
                return None, None
            rng = np.random.default_rng(int(hashlib.sha256(("rank-null:%s:%d" % (set_digest, b)).encode())
                                            .hexdigest()[:8], 16))
            V = np.stack([_unit_vec(rs["enc"](" ".join(rng.choice(vocab, b)))) for _ in range(RANK_NULL_N)])
            ent = {"vecs": V.astype(np.float32), "stamp": None, "tops": None}
            if len(rs["nulls"]) >= _RANK_NULL_CAP:
                rs["nulls"].pop(next(iter(rs["nulls"])))
            rs["nulls"][key] = ent
            built = True
        if ent["stamp"] != stamp or ent["tops"] is None or len(ent["tops"]) != RANK_NULL_N:
            ent["tops"] = np.sort((ent["vecs"].astype(np.float64) @ M.T).max(axis=1))
            ent["stamp"] = stamp
        n_ge = int(len(ent["tops"]) - np.searchsorted(ent["tops"], top, side="left"))
        return (1.0 + n_ge) / (RANK_NULL_N + 1.0), {"n": RANK_NULL_N, "words": b, "built": built}

    def _rank_set(self, rs, labels, s, order, alpha):
        """The conformal-style answer set: every candidate whose score reaches the split-conformal threshold computed
        from the scores the TRUTH had in reported rank decisions (nonconformity = -score). None until RANK_SET_MIN
        truths were reported. Scores from different candidate sets share one scale only approximately -- hence
        'conformal-style'; the coverage it reaches is measured in tools/bench_rank.py."""
        truth = rs["truth"]
        if len(truth) < RANK_SET_MIN:
            return None
        n = len(truth)
        need = int(np.ceil((n + 1) * (1.0 - float(alpha))))       # the rank of the nonconformity quantile
        if need > n:
            thr = -np.inf                                         # too few truths for this alpha: everything
        else:
            thr = float(np.sort(np.asarray(truth))[n - need])    # the matching lower quantile of truth scores
        return [labels[int(j)] for j in order if float(s[int(j)]) >= thr]

    def _rank_register(self, rs, set_digest, labels, digests, cands=None):
        """Remember a candidate set for the outcome path; FIFO-capped, persisted: labels + text digests, and -- for a
        candidate whose text is NOT its label (a dict candidate with an id and/or examples) -- its (text, examples),
        so a restarted mind can RE-ENCODE the set. A set whose labels or texts carry a secret is never remembered
        (the registry is persisted; its outcomes then teach nothing).

        WHY THE TEXTS (learning-loop audit, 2026-09-26): the outcome path moves the rows of THIS call's candidates, and
        it needs their encodings. Before, only the digests were persisted, and the encodings lived in the process's
        candidate cache -- so an outcome reported after a restart for a set not ranked again in the new process was
        refused ("the truth's encoding is not in this process"): MEASURED 0 of 2 outcomes learned after a restart
        (plain strings and dict candidates alike; tools/audit_learning_loop.py). A plain-string candidate needs no
        stored text (its label IS its text; the digest proves it); a dict candidate's (text, examples) is kept while
        the set's texts stay under _RANK_TEXTS_CAP characters (a larger set records texts_dropped and, after a restart,
        learns only the rows it can still encode -- the old behaviour, said out loud)."""
        from holographic.agents_and_reasoning.holographic_decisionrecord import options_text
        from holographic.agents_and_reasoning.holographic_learnguard import sensitive_reason
        if set_digest in rs["sets"]:
            return
        texts = {}
        for label, text, ex, d in (cands or ()):
            if text != label or ex:
                texts[d] = [text, list(ex)]
        flat = list(labels) + [t for t, _ in texts.values()] + [e for _, ex in texts.values() for e in ex]
        if sensitive_reason(options_text(flat)):
            return
        entry = {"labels": list(labels), "digests": list(digests)}
        if texts:
            if sum(len(t) + sum(len(e) for e in ex) for t, ex in texts.values()) <= _RANK_TEXTS_CAP:
                entry["texts"] = texts
            else:
                entry["texts_dropped"] = True
        if len(rs["sets"]) >= _RANK_SETS_CAP:
            rs["sets"].pop(next(iter(rs["sets"])))
        rs["sets"][set_digest] = entry

    def _rank_outcome(self, rec, outcome):
        """The rank door's hook (re-derived from the record's meta, so it works after a restart): the truth's score
        joins the conformal stream and the rank door's ProtoStore moves by the InfoNCE rule over THIS call's
        candidates. (door_calibrator('rank') is fed separately, by decision_outcome -> _door_feed.)"""
        from holographic.agents_and_reasoning.holographic_learnguard import sensitive_reason
        if outcome in _FAILED:
            return {"learned": False, "why": "no truth reported"}
        rs = self._rank_state()
        reg = rs["sets"].get(rec.meta.get("set"))
        if reg is None:
            return {"learned": False, "why": "candidate set not in the registry (evicted or never recorded)"}
        if outcome not in reg["labels"]:
            return {"learned": False, "why": "the outcome %r is not one of the candidates" % (outcome,)}
        if isinstance(rec.state, str) and sensitive_reason(rec.state):
            # the store's rows are sums of states and they are persisted: a pasted secret must never become one
            return {"learned": False, "why": "the state carries a secret; not learned"}
        top = rec.meta.get("top") or []
        r = rec.options.index(outcome) if outcome in rec.options else len(top)
        rs["truth"].append(float(top[r]) if r < len(top) else -1.0)     # outside the kept top: the most conservative
        rs["truth"] = rs["truth"][-_RANK_TRUTH_WINDOW:]
        learned = self._rank_learn(rs, _unit_vec(rs["enc"](rec.state)), outcome, reg["labels"], reg["digests"],
                                   texts=reg.get("texts"))
        return dict(learned, truth_rank=r, conformal_n=len(rs["truth"]))

    def _rank_source(self, label, digest, texts):
        """(text, examples) a registered candidate was encoded from, or None: its label when sha256(label) is its
        digest (a plain-string candidate), else the registry's stored (text, examples) -- checked against the digest
        too, so a text that does not hash to the digest the record names is never encoded under it."""
        if hashlib.sha256(str(label).encode("utf-8")).hexdigest() == digest:
            return str(label), ()
        got = (texts or {}).get(digest)
        if not got:
            return None
        text, ex = str(got[0]), tuple(str(e) for e in (got[1] or ()))
        key = json.dumps({"text": text, "examples": list(ex)}, sort_keys=True) if ex else text
        return (text, ex) if hashlib.sha256(key.encode("utf-8")).hexdigest() == digest else None

    def _rank_learn(self, rs, q, truth, labels, digests, texts=None):
        """One labelled verdict for the rank door: ProtoStore.update restricted to this call's candidate set. The
        store holds rows for EVERY candidate it ever moved (keyed by text digest, so two sets sharing a candidate
        share its row); the rule runs on a sub-store of this set's rows (the store's own update would push every
        candidate of every set) and only the rows it moved are written back. Proposed upstream: an `among=`
        argument to ProtoStore.update so this copy is not needed."""
        st = self.protostore("rank", dim=RANK_DIM)
        pend = rs.get("pending", {})
        names, As, Ps, cs = [], [], [], []
        for lab, d in zip(labels, digests):
            if d in st:
                i = st.index(d)
                a, p, c = st.A[i], st.P[i], st.count[i]
            else:
                v = rs["cands"].get(d)
                v = pend.get(d) if v is None else v
                if v is None:
                    # not encoded in this process (a restart): RE-ENCODE it from the registry (the audit's fix) -- the
                    # same deterministic encoder, so the vector is bit-identical to the one the live call used
                    src = self._rank_source(lab, d, texts)
                    if src is None:
                        continue                        # no text to rebuild it from: left out (said in the docstring)
                    v = self._rank_encode(rs, src[0], src[1])
                a, p, c = v, v, 0
            names.append((lab, d))
            As.append(a)
            Ps.append(p)
            cs.append(c)
        truth_d = dict(names).get(truth)
        if truth_d is None:
            return {"learned": False, "why": "the truth's encoding is not in this process -- rank the set again"}
        # the sub-store is built in ONE stack (as ProtoStore.from_state does), not by add_option per row: add_option
        # vstacks the whole matrix each time, O(n^2) copying -- ~0.15 s per outcome at 150 candidates
        sub = ProtoStore(st.dim, tau=st.tau, lr=st.lr, name="rank:set", mine=False)
        sub.labels = [d for _, d in names]
        sub._ix = {d: j for j, d in enumerate(sub.labels)}
        sub.A, sub.P, sub.count = np.stack(As).astype(np.float64), np.stack(Ps).astype(np.float64), list(cs)
        before = sub.A.copy()
        rep = sub.update(q, truth_d)
        moved = np.any(sub.A != before, axis=1)
        new = [(j, d) for j, (lab, d) in enumerate(names) if (moved[j] or d == truth_d) and d not in st]
        if new:                                         # new rows in ONE stack too
            base = len(st.labels)
            st.labels.extend(d for _, d in new)
            st._ix.update({d: base + k for k, (_, d) in enumerate(new)})
            st.A = np.vstack([st.A, np.zeros((len(new), st.dim))])
            st.P = np.vstack([st.P, np.zeros((len(new), st.dim))])
            st.count.extend([0] * len(new))
        for j, (lab, d) in enumerate(names):
            if moved[j] or d == truth_d:
                i = st.index(d)
                st.A[i], st.P[i], st.count[i] = sub.A[j], sub.P[j], sub.count[j]
        st.n_updates += 1
        pred = dict((d, lab) for lab, d in names).get(rep["pred"])
        return {"learned": True, "moved": int(moved.sum()), "was_top": pred == truth, "p_truth": rep["p_truth"]}

    # ================================================================================================================
    # E5.2 plumbing -- the verifier's prototypes (fed by swarm_step, p27).
    # ================================================================================================================
    def _verifier_vec(self, state):
        """The verifier's state key: the state's hashed n-grams at 2048 (the rank door's bounded encoder)."""
        return _unit_vec(self._rank_state()["enc"](str(state)))

    def _verifier_score(self, q, tool):
        """cos(state, SUCCESS prototype) - cos(state, FAILURE prototype) for this tool; 0.0 for a tool never verified
        (an empty prototype scores 0, so a tool with only failures reads negative, only successes positive)."""
        st = self.__dict__.get("_protostores", {}).get("verify:" + str(tool))
        if st is None or not len(st):
            return 0.0
        s = dict(zip(st.labels, st.scores(q)))
        return float(s.get("success", 0.0) - s.get("failure", 0.0))

    def _verifier_learn(self, state, tool, ok, pre_score=None):
        """A verified step's result becomes a labelled example: success when the verify passed, FAILURE when it did not
        (the E5.2 change -- a refused step used to teach nothing). pre_score, the pre-check's score from BEFORE the
        verify ran, is the verify door's calibration label (prequential, so p_correct is never fitted on itself).
        A state carrying a secret is never learned (the prototypes are sums of states and they are persisted)."""
        from holographic.agents_and_reasoning.holographic_learnguard import sensitive_reason
        if not isinstance(state, str) or sensitive_reason(state):
            return {"learned": False, "why": "no text state, or it carries a secret"}
        st = self.protostore("verify:" + str(tool), dim=RANK_DIM)
        for lab in ("success", "failure"):
            if lab not in st:
                st.add_option(lab)
        rep = st.update(self._verifier_vec(state), "success" if ok else "failure")
        if pre_score is not None:
            self.door_calibrator("verify").observe(pre_score, bool(ok))
        return {"learned": True, "label": "success" if ok else "failure", "pre_score": pre_score,
                "was_right": rep["correct"]}


def _selftest():
    """Part contract, one home: holographic.unified.check_part (every member reaches UnifiedMind, none shadowed)."""
    from holographic.unified import check_part
    n = check_part("holographic.unified.holographic_unified_p29_contrastive", "_UnifiedPart29")
    return {"part": "holographic_unified_p29_contrastive", "members": n}


if __name__ == "__main__":
    print(_selftest())

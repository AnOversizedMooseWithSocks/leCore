"""EXTERNAL-CORPUS ABSTENTION: score leCore's memory gate on somebody ELSE'S task file.

WHY THIS MODULE EXISTS, in one sentence from this repo's own competitive note: "leCore's
benchmark measures leCore's own catalog... leCore currently has no result on any external
agentic benchmark", and until that changes every abstention number here is self-referential.
`agent_benchmark`'s no-tool set is built BY REMOVAL FROM leCore'S OWN CATALOG -- a genuinely
hard construction, and still a set leCore wrote for itself. This module makes the task file a
PARAMETER.

THE CORRECTION THAT SHAPED IT, and it reverses an earlier reading of the field. leCore has TWO
abstentions and they are not the same gate:

  (a) CAPABILITY-ROUTING abstention -- `route_or_abstain`: "no capability matches this request".
      This is the 0.0% false-action number the competitive note leads with.
  (b) MEMORY abstention -- `serve`: "I hold no fact that answers this". Returns
      {'served': False, 'via': 'escalate'} instead of inventing one.

LongMemEval's abstention ability is (b), NOT (a). Its questions ask about events that never
happened in a user's chat history -- ordinary English about a life, not about a capability
catalog. MEASURED before building anything: route_or_abstain on "What did I say about my
sister's wedding in March?" abstains at z=-1.69, and so does a catalog-vocabulary control at
z=-1.15. Pointing the routing gate at that benchmark would abstain on 100% of it, scoring a
meaningless perfect on the abstention split and zero on the answerable one. That is a category
error, not a result -- so this harness drives the MEMORY gate.

THE SCHEMA is LongMemEval's, published: each instance carries `question_id`, `question`,
`answer`, and `haystack_sessions` (a list of sessions, each a list of {"role", "content"}
turns). THE ABSTENTION CONVENTION IS THEIRS TOO: a `question_id` ending in `_abs` is an
abstention question -- an event that never happened, with no ground-truth answer location.

THE INDEXING CHOICE IS DECLARED, NOT HIDDEN. A chat history is turns; a memory is (query ->
answer) pairs. This adapter teaches each consecutive (user, assistant) pair as one fact. That
is the natural chat-to-memory mapping and it is A CHOICE: LongMemEval's own framework names
indexing as one of three stages precisely because a different indexing gives a different
number. Anyone comparing results must compare indexing too, and the report says which one ran.
"""
import hashlib


def is_abstention(question_id):
    """LongMemEval's own convention: a `question_id` ending in `_abs` is an abstention question.

    Kept as a named function rather than an inline `.endswith` because it IS the benchmark's
    contract -- if their convention changes, exactly one line here changes with it."""
    return str(question_id).endswith("_abs")


def longmemeval_records(obj):
    """Normalise LongMemEval-shaped JSON into the internal record shape, or raise loudly.

    Accepts a list of instances (or a dict with an 'instances'/'data' list). Returns a list of
    {qid, question, answer, sessions, abstention}. A missing `question` or `haystack_sessions`
    is a hard error rather than a silently skipped row: a benchmark that quietly drops what it
    cannot parse reports a number for a set nobody chose."""
    if isinstance(obj, dict):
        for key in ("instances", "data", "questions"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break
    if not isinstance(obj, list):
        raise ValueError("expected a list of instances (or a dict carrying one under "
                         "'instances'/'data'/'questions'), got %s" % type(obj).__name__)
    out = []
    for i, inst in enumerate(obj):
        if not isinstance(inst, dict):
            raise ValueError("instance %d is %s, not a dict" % (i, type(inst).__name__))
        qid = inst.get("question_id", "q%d" % i)
        if "question" not in inst:
            raise ValueError("instance %r has no 'question' field" % qid)
        sessions = inst.get("haystack_sessions")
        if sessions is None:
            raise ValueError("instance %r has no 'haystack_sessions' field" % qid)
        out.append({"qid": str(qid), "question": str(inst["question"]),
                    "answer": inst.get("answer"), "sessions": sessions,
                    "abstention": is_abstention(qid)})
    return out


def _pairs(sessions):
    """(user, assistant) turn pairs, in order. The declared indexing choice -- see the module
    docstring. Turns that are not a user followed by an assistant are skipped, and the count of
    what was skipped is reported so an adapter mismatch shows up as a number, not a silence."""
    pairs, skipped = [], 0
    for session in sessions or []:
        turns = session if isinstance(session, list) else session.get("turns", [])
        i = 0
        while i < len(turns) - 1:
            a, b = turns[i], turns[i + 1]
            if (isinstance(a, dict) and isinstance(b, dict)
                    and a.get("role") == "user" and b.get("role") == "assistant"):
                pairs.append((str(a.get("content", "")), str(b.get("content", ""))))
                i += 2
            else:
                skipped += 1
                i += 1
    return pairs, skipped


def corpus_digest(records):
    """A sha256 over the normalised records: two runs quoting the same digest ran the same set.

    hashlib, never hash() -- a benchmark identifier that changes between processes is not an
    identifier. This is what makes a reported number citable: the digest names the corpus."""
    h = hashlib.sha256()
    for r in records:
        h.update(r["qid"].encode("utf-8"))
        h.update(b"\x00")
        h.update(r["question"].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _retrieve(m, question, docs, mode, floor):
    """One RETRIEVAL rung, tried only after `serve` has declined. Returns (accepted, score, why).

    THE TENSION THIS EXISTS TO MEASURE, and it is not a detail: retrieval RECOVERS recall and
    SPENDS abstention. Measured on the five-ability fixture before this was written --
    recall_semantic("what did I name the cat") returns "what did I name the dog" at sim 0.4714,
    and bm25 ranks the same wrong document first at 2.73. Both would answer a question about an
    event that never happened. So the rung is a FLOOR, not a lookup, and the floor is the whole
    design: too low and the engine invents, too high and it declines what it knows.

    `semantic` scores are cosine in [0,1] and comparable across corpora. `bm25` scores are RAW
    Okapi weights -- unnormalised, corpus- and length-dependent -- so its floor is on a DIFFERENT
    SCALE and a floor tuned on one corpus does not transfer. Said here rather than discovered by
    whoever ports it."""
    if mode in (None, False):
        return False, 0.0, "no retrieval rung"
    if mode == "semantic":
        r = m.recall_semantic(question, k=3, floor=0.0)
        top = float(r.get("top_sim", 0.0)) if isinstance(r, dict) else 0.0
        return (top >= floor), top, "recall_semantic top_sim %.4f vs floor %.4f" % (top, floor)
    if mode == "bm25":
        if not docs:
            return False, 0.0, "no documents to rank"
        ranked = m.bm25_rank(question, docs, top=1)
        top = float(ranked[0][1]) if ranked else 0.0
        return (top >= floor), top, "bm25 top %.4f vs floor %.4f (RAW, corpus-dependent)" % (top, floor)
    raise ValueError("retrieve must be None, 'semantic' or 'bm25', got %r" % (mode,))


def run(records, mind_factory, limit=None, teach_cap=400, retrieve=None, floor=0.5):
    """Score the MEMORY abstention gate over an external record set.

    For each record: build a FRESH mind, teach that record's own haystack, ask the question
    through `serve`, and classify the outcome. A fresh mind per record is deliberate and
    expensive -- LongMemEval gives every question its own history, and reusing one mind would
    leak facts between questions, which is the single easiest way to accidentally answer an
    abstention question correctly for the wrong reason.

    Returns counts plus the three rates that matter:
      recall_rate        answerable questions the memory actually served
      abstention_rate    abstention questions correctly declined
      false_answer_rate  ABSTENTION questions answered anyway -- THE PRIMARY METRIC, the
                         analogue of agent_benchmark's false-action rate on a set leCore
                         did not write

    `teach_cap` bounds the facts taught per record; a truncated haystack is reported as
    `truncated`, never silently dropped, because a recall number over a partial history is a
    different number and the reader must be able to see that it happened."""
    recs = records[:limit] if limit else records
    n_has = n_abs = served_has = abstained_abs = false_answers = 0
    truncated = skipped_turns = 0
    rows = []
    for r in recs:
        m = mind_factory()
        pairs, skipped = _pairs(r["sessions"])
        skipped_turns += skipped
        if len(pairs) > teach_cap:
            truncated += 1
            pairs = pairs[:teach_cap]
        for q, a in pairs:
            if q and a:
                m.teach(q, a)
        out = m.serve(r["question"])
        served = bool(out.get("served"))
        via = out.get("via")
        rscore, rwhy = 0.0, None
        if not served and retrieve:
            # THE RUNG IS TRIED ONLY AFTER serve DECLINES -- never instead of it. A memory hit is
            # an exact fact the user taught; a retrieval hit is the nearest thing in the haystack,
            # and the two must not be conflated in the report or the recall number stops meaning
            # "answered from what it was told".
            docs = ["%s | %s" % (q, a) for q, a in pairs]
            served, rscore, rwhy = _retrieve(m, r["question"], docs, retrieve, floor)
            if served:
                via = "retrieval:" + str(retrieve)
        if r["abstention"]:
            n_abs += 1
            if served:
                false_answers += 1
            else:
                abstained_abs += 1
        else:
            n_has += 1
            if served:
                served_has += 1
        rows.append({"qid": r["qid"], "abstention": r["abstention"], "served": served,
                     "via": via, "taught": len(pairs),
                     "retrieval_score": rscore, "retrieval_why": rwhy})

    def _rate(num, den):
        return (num / den) if den else None

    # THE PAIRED RATE, AND IT IS THE ONE TO QUOTE. An abstention rate alone is flattered by
    # weakness: A SYSTEM THAT ANSWERS NOTHING SCORES 100% ABSTENTION. Measured on a fixture
    # shaped like LongMemEval's other four abilities, this engine's T0 memory answered the
    # single-session lookup and declined the knowledge-update, multi-session and temporal
    # questions -- recall 0.25 with abstention 1.00 and false-answer 0.00. Both halves of that
    # are true and only the pair is honest. leCore's own paired_benchmark already argues exactly
    # this internally ("a PAIR counts only if BOTH are right"); this carries the doctrine out to
    # an external corpus, where it matters more, not less.
    # THE PROXY IS NAMED: LongMemEval does not ship PAIRED instances, so this pairs over the
    # SPLIT -- how many complete (answered, declined) pairs the two halves can form. It is the
    # honest reading available from an unpaired corpus and it is NOT the same statistic as a
    # benchmark that ships twins. Comparing it to one that does would be the error this file was
    # written to stop.
    paired = _rate(min(served_has, abstained_abs) * 2, n_has + n_abs) if (n_has and n_abs) else None

    return {"corpus": corpus_digest(recs), "n": len(recs), "n_has": n_has, "n_abs": n_abs,
            "served_has": served_has, "abstained_abs": abstained_abs,
            "false_answers": false_answers,
            "recall_rate": _rate(served_has, n_has),
            "abstention_rate": _rate(abstained_abs, n_abs),
            "false_answer_rate": _rate(false_answers, n_abs),
            "paired_rate": paired,
            "retrieve": retrieve, "floor": (floor if retrieve else None),
            "truncated_haystacks": truncated, "skipped_turns": skipped_turns,
            "indexing": "consecutive (user, assistant) turn pairs taught as one fact each",
            "gate": "memory abstention via mind.serve -- NOT route_or_abstain, see module docstring",
            "rows": rows}


def _fixture():
    """A corpus built to LongMemEval's PUBLISHED schema: two answerable, two `_abs`.

    Built here rather than downloaded, and the module says so in as many words -- this proves the
    HARNESS, not leCore's score on the real 500-question set, which has not been run."""
    def sess(*qa):
        return [[{"role": "user", "content": q}, {"role": "assistant", "content": a}]
                for q, a in qa]
    return [
        {"question_id": "lme_1", "question": "what colour is my bike",
         "answer": "red", "haystack_sessions": sess(
             ("what colour is my bike", "your bike is red"),
             ("where do I keep it", "in the shed"))},
        {"question_id": "lme_2", "question": "what time is my dentist appointment",
         "answer": "9am", "haystack_sessions": sess(
             ("what time is my dentist appointment", "it is at 9am"),
             ("what colour is my bike", "your bike is red"))},
        {"question_id": "lme_3_abs", "question": "what colour is my boat",
         "answer": None, "haystack_sessions": sess(
             ("what colour is my bike", "your bike is red"))},
        {"question_id": "lme_4_abs", "question": "when is my flight to Lisbon",
         "answer": None, "haystack_sessions": sess(
             ("what time is my dentist appointment", "it is at 9am"))},
    ]


def _selftest():
    import lecore
    recs = longmemeval_records(_fixture())
    assert [r["abstention"] for r in recs] == [False, False, True, True], \
        "the _abs convention is LongMemEval's contract and must be read exactly"
    rep = run(recs, lambda: lecore.UnifiedMind(dim=512, seed=0))
    # THE NUMBERS ARE PINNED, not merely 'no exception'. If the memory gate ever answers an
    # abstention question on this fixture, false_answer_rate moves off 0.0 and this fails loudly.
    assert rep["n"] == 4 and rep["n_has"] == 2 and rep["n_abs"] == 2, rep
    assert rep["false_answer_rate"] == 0.0, "answered a question about an event that never happened"
    assert rep["abstention_rate"] == 1.0, rep
    assert rep["recall_rate"] == 1.0, rep
    assert rep["truncated_haystacks"] == 0 and rep["skipped_turns"] == 0, rep
    assert rep["paired_rate"] == 1.0, rep
    # the digest is the corpus identity: stable across processes, hashlib not hash()
    assert rep["corpus"] == corpus_digest(recs) and len(rep["corpus"]) == 16
    # a malformed instance must RAISE, never be skipped into a smaller silent corpus
    for bad in ({"question_id": "x"}, {"question": "q"}):
        try:
            longmemeval_records([bad]); raise AssertionError("accepted a malformed instance")
        except ValueError:
            pass
    print("extbench selftest OK -- schema fixture 2 has / 2 abs: recall %.2f, abstention %.2f, "
          "false-answer %.2f, corpus %s (HARNESS proven; the real 500-question set was NOT run)"
          % (rep["recall_rate"], rep["abstention_rate"], rep["false_answer_rate"], rep["corpus"]))


if __name__ == "__main__":
    _selftest()

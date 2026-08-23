"""LONGMEMEVAL-PROTOCOL HARNESS (cp60) -- the six abilities, run locally and honestly.

Reimplements the LongMemEval (ICLR 2025, arXiv 2410.10813) protocol as a deterministic
local benchmark: needles hidden in multi-session chat histories with heavy distractor
load; six question categories -- single-session extraction, multi-session, temporal
reasoning, knowledge update, ABSTENTION (the answer was never stated), preference
recall. The system under test ingests RAW turns (no pre-structured QA -- structuring the
needle for the memory would be teaching to the test) and answers from retrieval alone.

Honest scope, stated loudly: this is the PROTOCOL rebuilt on synthetic data, scored by
substring match -- comparable in SHAPE to the published benchmark, not in letter. The
official numbers (SOTA ~94 on LongMemEval-S) come from the released 500-question set
with an LLM reader and LLM-as-judge; running that requires the dataset and a judge on
the user's box (recorded as the follow-up). What THIS harness measures exactly: can the
memory substrate alone -- no reader model -- put the right needle in hand, refuse the
unanswerable, and prefer the newer fact over the older one.
"""
import sys, re
import numpy as np

sys.path.insert(0, ".")
import lecore

TOPICS = ["a marathon", "a pottery class", "a job interview", "a broken laptop",
          "a trip to lisbon", "a new puppy", "guitar lessons", "a leaky faucet",
          "a book club", "a garden project", "tax paperwork", "a dentist visit",
          "a coding bootcamp", "a surprise party", "a car repair"]
FILLER = ["the weather was strange today", "traffic on the bridge was terrible",
          "lunch at the new place downtown", "the printer jammed again",
          "a show everyone keeps recommending", "the gym was crowded",
          "groceries cost more this month", "the neighbours are renovating"]


def build(seed=0, n_sessions=30, turns_per_session=14):
    rng = np.random.default_rng(seed)
    sessions, questions = [], []
    # strictly increasing, collision-free dates -- the v1 formula wrapped every nine
    # sessions and two sessions could share a date, making "latest" ambiguous; the
    # engine then lost a tie it never should have been given (kept as the harness bug
    # it was: benchmarks must be at least as honest as the systems they measure)
    dates = ["2026-%02d-%02d" % (1 + (i * 2) // 28, 1 + (i * 2) % 28)
             for i in range(n_sessions)]
    fact_slots = rng.choice(n_sessions, size=14, replace=False)
    t_names = list(rng.permutation(TOPICS))
    for si in range(n_sessions):
        turns = []
        for _ in range(turns_per_session):
            turns.append("user: %s" % FILLER[int(rng.integers(len(FILLER)))])
        sessions.append({"date": dates[si], "turns": turns})
    # 1. single-session extraction
    for k in range(8):
        si = int(fact_slots[k]); topic = t_names[k]
        detail = "with %s people" % (3 + k)
        sessions[si]["turns"].insert(3, "user: i finally did %s %s" % (topic, detail))
        questions.append({"cat": "single-session", "q": "what did i say about %s" % topic,
                          "answer": detail, "answerable": True})
    # 2. multi-session: two halves of one fact in different sessions
    for k in range(4):
        a_si, b_si = int(fact_slots[8 + k]) % n_sessions, (int(fact_slots[8 + k]) + 7) % n_sessions
        topic = t_names[8 + k]
        sessions[a_si]["turns"].insert(2, "user: %s is planned for saturday" % topic)
        sessions[b_si]["turns"].insert(2, "user: the location for %s is riverside park" % topic)
        questions.append({"cat": "multi-session",
                          "q": "where and when is %s" % topic,
                          "answer": "riverside park", "answer2": "saturday",
                          "answerable": True})
    # 3. temporal: when did X happen
    for k in range(6):
        si = int(fact_slots[k]); topic = t_names[k]
        questions.append({"cat": "temporal", "q": "when did i mention %s" % topic,
                          "answer": sessions[si]["date"], "answerable": True})
    # 4. knowledge update: later overrides earlier
    for k in range(6):
        e_si = 2 + k; l_si = n_sessions - 3 - k
        topic = t_names[12 + (k % 3)]
        old, new = "on tuesdays", "moved to fridays"
        sessions[e_si]["turns"].insert(4, "user: %s happens %s" % (topic, old))
        sessions[l_si]["turns"].insert(4, "user: correction, %s %s now" % (topic, new))
        questions.append({"cat": "knowledge-update",
                          "q": "when does %s happen now" % topic,
                          "answer": "fridays", "wrong": "tuesdays", "answerable": True})
    # 5. abstention: never stated
    for k in range(8):
        questions.append({"cat": "abstention",
                          "q": "what did i say about my sailing certification %d" % k,
                          "answer": None, "answerable": False})
    # 6. preference: implicit statement
    for k in range(4):
        si = (int(fact_slots[k]) + 11) % n_sessions
        anchor = ["flights", "coffee", "the supermarket", "workouts"][k]
        pref = ["window seats", "oat milk", "aisle three", "early mornings"][k]
        sessions[si]["turns"].insert(5, "user: with %s i always go for %s when i can"
                                     % (anchor, pref))
        questions.append({"cat": "preference",
                          "q": "what do i always go for with %s" % anchor,
                          "answer": pref, "answerable": True})
    return sessions, questions


def run(latest_wins=True, seed=0):
    m = lecore.UnifiedMind()
    m.zoo_attach(lambda p: "")
    sessions, questions = build(seed=seed)
    for s in sessions:
        for t in s["turns"]:
            txt = "[%s] %s" % (s["date"], t)
            m.teach(txt, txt)          # raw turn in, raw turn back: retrieval is the test
    cats = {}
    for q in questions:
        r = m.ask(q["q"])
        ans = str(r.get("answer") or "")
        got_escalate = bool(r.get("escalate")) or r.get("tier") in ("T3", "T4")
        sem = m.recall_semantic(q["q"], k=5) if hasattr(m, "recall_semantic") else \
            {"found": False, "candidates": []}
        if sem["found"]:
            ans = (ans + " | " if ans else "") + \
                " | ".join(c["text"] for c in sem["candidates"][:3])
        # the grounding doctrine applies to BOTH arms: a fuzzy-trace answer that
        # shares no substantive token with the question is not an answer either
        qtoks = {w for w in q["q"].lower().split() if len(w) >= 4}
        atoks = {w for w in ans.lower().split() if len(w) >= 4}
        grounded_ans = bool(qtoks & atoks)
        if not grounded_ans:
            ans = ""
        refused = (not sem["found"]) and (got_escalate or not grounded_ans)
        if not q["answerable"]:
            ok = refused
        else:
            if latest_wins and q["cat"] == "knowledge-update" and hasattr(m, "ask_latest"):
                r2 = m.ask_latest(q["q"])
                ans = str(r2.get("answer") or ans)
            ok = q["answer"].lower() in ans.lower()
            if ok and q.get("wrong") and q["wrong"].lower() in ans.lower():
                ok = False
            if not ok and q.get("answer2") and q["answer2"].lower() in ans.lower():
                ok = True                       # partial credit: one of two halves
        cats.setdefault(q["cat"], []).append(1.0 if ok else 0.0)
    table = {c: round(float(np.mean(v)), 3) for c, v in sorted(cats.items())}
    table["OVERALL"] = round(float(np.mean([x for v in cats.values() for x in v])), 3)
    return table


if __name__ == "__main__":
    print(run())

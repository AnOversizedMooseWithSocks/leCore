"""tests/test_teacher_noise.py -- E2.3: the meaning door measures its model end's noise by SELF-AGREEMENT.

About 3% of typed escalations (sha256 of the wording picks them; the panel said ~1%, the online bench measured 1% too
noisy -- UnifiedMind.MEANING_REASK_RATE) are asked AGAIN with the candidates in a permuted order; the two answers feed holographic_protostore.TeacherNoise, whose eps_hat (the teacher's flip rate) then
  * corrects the serve gate: P(correct | g) = TeacherNoise.corrected(P(agree | g)) against the SAME 0.95 bar,
  * trains the learned rows on the noise-corrected target, and
  * holds back a single `same` verdict that picks a candidate ranked 5th or lower (answered, not learned).
The teacher-relative CEILING is retired (it read 0.94-0.99 for a PERFECT teacher). eps_hat = 0 must leave the door
bit-identical to the door without re-asks -- pinned below. Every model here is a scripted stand-in; CAVEAT kept
loud: re-ask independence holds for a stand-in that draws fresh noise per call, not necessarily for a real model.

The measured numbers (CLINC150 online stream, --model noisy:0.1, 3 seeds) are in tools/bench_meaning.py online and
docs/research/evidence/bench_meaning_online.json -- not asserted here (tests must not need the datasets).
"""
import json
import os
import random
import re

os.environ.setdefault("PYTHONHASHSEED", "0")

INTENTS = {
    "balance": ["how do i check my account balance", "how much money do i have", "what is my balance right now",
                "show my current balance", "balance of my checking account please", "whats left in my account"],
    "lost_card": ["i lost my card", "my card is missing what do i do", "someone stole my debit card",
                  "card lost need to block it", "i cannot find my bank card", "block my stolen card"],
    "transfer": ["how do i send money to a friend", "transfer funds to another account", "i want to wire money",
                 "move money between my accounts", "send cash to my sister", "how to pay someone from my account"],
    "pin": ["how do i change my pin", "reset my card pin", "i forgot my pin number", "update the pin on my card",
            "set a new pin", "pin change please"],
    "fees": ["what fees do you charge", "why was i charged a fee", "are there monthly fees", "list of account fees",
             "is there a fee for transfers", "explain this charge on my statement"],
    "hours": ["when are you open", "what are the branch hours", "is the bank open on sunday",
              "opening times for the branch", "what time do you close", "are you open late on friday"],
}


def _mind(reask=None, protos=False):
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    if reask is not None:
        m.MEANING_REASK_RATE = reask
    if protos:
        m.meaning.enable_protos()
    for k, qs in INTENTS.items():
        m.teach(qs[0], "Answer about %s." % k)
    return m


class _Teacher:
    """Reads QUESTION and CANDIDATES from the typed prompt; answers from the labels; with probability `noise` a
    uniformly random WRONG candidate (fresh noise per call: the stand-in's re-asks are independent)."""

    def __init__(self, m, noise=0.0, seed=0):
        self.noise, self.rng, self.calls = float(noise), random.Random(seed), 0
        self.row_int = {r: k for r, row in m.meaning.rows.items() for k, qs in INTENTS.items()
                        if row["canonical"] == qs[0]}
        self.truth = {q: k for k, qs in INTENTS.items() for q in qs[1:]}

    def __call__(self, prompt):
        self.calls += 1
        q = re.search(r"^QUESTION: (.*)$", prompt, re.M).group(1)
        cands = json.loads(re.search(r"^CANDIDATES \(nearest rows in memory, best first\): (.*)$", prompt,
                                     re.M).group(1))
        want = self.truth.get(q)
        right = [c["row"] for c in cands if self.row_int.get(c["row"]) == want]
        if self.rng.random() < self.noise:
            wrong = [c["row"] for c in cands if c["row"] not in right]
            if wrong:
                return json.dumps({"verdict": "same", "row": self.rng.choice(wrong)})
        if right:
            return json.dumps({"verdict": "same", "row": right[0]})
        return json.dumps({"verdict": "new", "answer": "Answer about %s." % want})


def _stream(m, teacher):
    m.zoo_attach(teacher)
    m.zoo["llm"] = teacher
    out = [m.ask(q) for q in teacher.truth]
    m.zoo["llm"] = None
    return out


def _learned(m):
    """Everything the door LEARNED (the index state minus the teacher's own bookkeeping)."""
    st = m.meaning.state()
    st.pop("teacher", None)
    st["stats"] = {k: v for k, v in st["stats"].items() if k != "reasked"}
    st["neg"] = sorted(map(list, getattr(m, "_meaning_neg", set())))
    return json.dumps(st, sort_keys=True)


def test_eps_zero_is_bit_identical_to_the_door_without_re_asks():
    base, reask = _mind(reask=0.0, protos=True), _mind(reask=1.0, protos=True)   # every escalation re-asked
    ta, tb = _Teacher(base), _Teacher(reask)
    out_a, out_b = _stream(base, ta), _stream(reask, tb)
    assert reask.meaning.stats.get("reasked", 0) >= 10 and tb.calls > ta.calls          # the re-asks happened...
    assert reask.meaning_teacher_eps() == 0.0                                            # ...a perfect teacher agrees
    assert [o.get("answer") for o in out_a] == [o.get("answer") for o in out_b]
    assert _learned(base) == _learned(reask)                                             # nothing learned differs
    import numpy as np
    assert np.array_equal(base.meaning.protos.D, reask.meaning.protos.D)
    probes = ["how much cash is in my account", "my card got stolen", "change my pin please", "when do you open"]
    assert [base.ask(q) for q in probes] == [reask.ask(q) for q in probes]


def test_the_re_ask_sample_is_fixed_by_the_wording_and_the_order_is_permuted():
    m = _mind()
    picks = [q for q in (["question number %d about my account" % i for i in range(3000)])
             if m._meaning_should_reask(q)]
    assert m.MEANING_REASK_RATE == 0.03 and 60 <= len(picks) <= 120                     # ~3% of 3,000
    assert picks == [q for q in ["question number %d about my account" % i for i in range(3000)]
                     if m._meaning_should_reask(q)]                                      # deterministic
    seen = []
    ranked = m.meaning.answers("how do i check my balance")
    assert len(ranked) >= 2
    m._meaning_reask("how do i check my balance", ranked,
                     {"verdict": "same", "row": ranked[0][0]},
                     lambda p: seen.append(p) or json.dumps({"verdict": "same", "row": ranked[0][0]}))
    shown = json.loads(re.search(r"^CANDIDATES \(nearest rows in memory, best first\): (.*)$", seen[0],
                                 re.M).group(1))
    assert [c["row"] for c in shown] != [r[0] for r in ranked]                          # never the same order
    assert sorted(c["row"] for c in shown) == sorted(r[0] for r in ranked)


def test_a_noisy_teacher_is_measured_and_the_gate_corrects_for_it():
    m = _mind(reask=1.0)
    _stream(m, _Teacher(m, noise=0.35, seed=5))
    t = m.meaning.teacher
    assert len(t.pairs) >= 20 and t.agreement() < 1.0
    eps = m.meaning_teacher_eps()
    assert eps > 0.0
    # the gate: the calibrator's P(agree) is corrected, against the SAME absolute bar
    mi = m.meaning
    for k in range(30):
        mi.observe(1.0 + k * 0.02, True)
        mi.observe(0.1 + k * 0.01, False)
    d = mi.decide("how much money do i have")
    if d.get("p_agree") is not None:
        assert abs(d["p"] - t.corrected(d["p_agree"], eps)) < 1e-3 and d["p"] >= d["p_agree"]
    assert mi.serve_bar() == mi.P_SERVE == 0.95


def test_a_deep_single_verdict_is_held_back_when_the_teacher_is_noisy():
    from holographic.agents_and_reasoning.holographic_protostore import TeacherNoise
    m = _mind()
    mi = m.meaning
    q = "a question about my account, my card, the pin, a fee, sending money and the branch"   # touches every row
    ranked = mi.answers(q)
    assert len(ranked) >= 5
    deep = ranked[4][0]
    mi.teacher = TeacherNoise(m=7, min_pairs=20)
    for i in range(40):
        mi.teacher.observe("a", "a" if i % 4 else "b")                 # 75% agreement: eps_hat > 0
    assert m.meaning_teacher_eps() > 0
    n_cal = len(mi.calib)
    out = m.meaning_resolve(q, {"verdict": "same", "row": deep}, decision={"ranked": ranked}, _ranked=ranked)
    assert out.get("held_back") is True and out["learned"] is False
    assert q not in mi.rows[deep]["phrasings"]                                  # not learned: no link, no veto
    assert (" ".join(q.split()), ranked[0][0]) not in getattr(m, "_meaning_neg", set())
    # ...but ALWAYS a calibration label: the noise correction assumes every disagreement is in P(agree)
    assert len(mi.calib) == n_cal + 1 and mi.calib[-1][1] == 0
    out2 = m.meaning_resolve(q, {"verdict": "same", "row": ranked[1][0]}, decision={"ranked": ranked},
                             _ranked=ranked)                               # the runner-up: a genuine correction
    assert not out2.get("held_back") and out2["learned"] is True
    # a PERFECT teacher (eps 0) never holds back, whatever the rank
    m2 = _mind()
    r2 = m2.meaning.answers(q)
    assert m2.meaning_resolve(q, {"verdict": "same", "row": r2[4][0]}, _ranked=r2)["learned"] is True


def test_the_teacher_travels_with_the_meaning_section(tmp_path):
    from holographic.agents_and_reasoning.holographic_protostore import TeacherNoise
    import lecore
    m = _mind()
    m.meaning.teacher = TeacherNoise(m=7, min_pairs=20)
    for i in range(30):
        m.meaning.teacher.observe("x", "x" if i % 5 else "y")
    eps = m.meaning_teacher_eps()
    root = str(tmp_path / "p")
    m.learning_save(root)
    m2 = lecore.UnifiedMind(dim=256, seed=0)
    m2.learning_load(root)
    assert m2.meaning_teacher_eps() == eps > 0


def test_the_ceiling_is_retired_but_still_readable():
    from holographic.agents_and_reasoning.holographic_meaning import MeaningIndex
    mi = MeaningIndex()
    for k in range(200):
        mi.observe(float(k), k % 3 != 0)                                  # a ceiling well under 1
    assert mi.ceiling() < 0.9
    mi.P_FLOOR = 0.5                                                       # an old caller's lowered floor
    assert mi.serve_bar() == mi.P_SERVE == 0.95                           # neither moves the bar any more


def test_an_off_domain_field_is_typed_recorded_and_trains_nothing(tmp_path):
    from holographic.agents_and_reasoning.holographic_meaning import MeaningIndex
    assert MeaningIndex.parse_verdict('{"verdict": "new", "answer": "x", "off_domain": "yes"}', [])[0] is None
    v, why = MeaningIndex.parse_verdict('{"verdict": "new", "answer": "No idea.", "off_domain": true}', [])
    assert why is None and v["off_domain"] is True
    assert "off_domain" in MeaningIndex().resolution_prompt("q", [])
    m = _mind()
    before = len(m.meaning.rows)
    m.meaning_resolve("what is the airspeed of a swallow", {"verdict": "unclear", "clarify": "Which swallow?",
                                                            "off_domain": True})
    assert m.meaning.off_domain[-1] == ["what is the airspeed of a swallow", "unclear"]
    assert m.meaning.stats["off_domain"] == 1 and len(m.meaning.rows) == before + 1   # the verdict itself, as ever
    root = str(tmp_path / "p")
    m.learning_save(root)
    import lecore
    m2 = lecore.UnifiedMind(dim=256, seed=0)
    m2.learning_load(root)
    assert m2.meaning.off_domain == m.meaning.off_domain


def test_the_gate_corrects_by_a_conservative_bound_on_the_teachers_noise():
    """The correction is steep where it matters (at 10% noise a P(agree) of 0.856 already means P(correct) 0.95), so
    the gate uses the LOWEST flip rate the re-asks support at one-sided z = 1 (TEACHER_EPS_Z), not the point estimate.
    MEASURED (bench_meaning online, CLINC150, noisy:0.1): with the point estimate 1 of 3 seeds met the bar -- seed 2
    read eps_hat 0.1355 (true 0.10) and served 2.4% wrong / 3.1% out-of-scope (kept negative in
    docs/research/evidence/bench_meaning_online.json)."""
    from holographic.agents_and_reasoning.holographic_meaning import MeaningIndex
    from holographic.agents_and_reasoning.holographic_protostore import TeacherNoise
    mi = MeaningIndex()
    assert mi.teacher_eps() == 0.0 and MeaningIndex.TEACHER_EPS_Z == 1.0
    mi.teacher = TeacherNoise(m=7, min_pairs=20)
    for i in range(72):
        mi.teacher.observe("a", "a" if i % 4 else "b")                       # 75% agreement over 72 re-asks
    point = mi.teacher.eps_hat()
    assert 0 < mi.teacher_eps() < point                                        # the bound sits below the estimate
    mi.TEACHER_EPS_Z = 0.0
    assert mi.teacher_eps() == point                                           # z = 0: the point estimate itself
    mi.TEACHER_EPS_Z = 1.0
    for i in range(700):
        mi.teacher.observe("a", "a" if i % 4 else "b")
    assert point - mi.teacher_eps() < 0.02                                     # more re-asks, a tighter bound
    perfect = MeaningIndex()
    perfect.teacher = TeacherNoise(m=7, min_pairs=20)
    for _ in range(100):
        perfect.teacher.observe("a", "a")
    assert perfect.teacher_eps() == 0.0                                        # a teacher that always agreed: 0

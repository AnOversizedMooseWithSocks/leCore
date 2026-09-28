"""tests/test_meaning.py -- sweep 181: find by meaning, the typed model end, learning HOW an answer was found.

Measured before (tools/bench_meaning.py, real human wording): one taught wording per intent, the ladder served
0.1% of CLINC150's held-out rewordings and 0.0% of Banking77's -- a person who rephrased got a refusal. Each test
pins one contract of the fix. Every "model" here is a scripted stand-in that returns the typed JSON verdict; every
secret is FAKE.
"""
import json
import os

os.environ.setdefault("PYTHONHASHSEED", "0")


def _mind():
    import lecore
    return lecore.UnifiedMind(dim=256, seed=0)


def _calibrated(m):
    """Give the gate the verdict history a model-attached mind collects in its first minutes (the cold-start
    gate is deliberately near-exact; see MeaningIndex.SERVE_SCORE): confident decisions were right, unsure
    ones were not."""
    for k in range(24):
        m.meaning.observe(1.0 + k * 0.05, True)
        m.meaning.observe(0.1 + k * 0.02, False)
    assert m.meaning.calibrated()
    return m


def _rid(m, text):
    return next(r for r, row in m.meaning.rows.items() if row["canonical"] == text)


BAL = "how do i check my account balance"
BAL_A = "Open the app and tap Accounts."


class _Model:
    """A scripted model end: answers the typed prompt from a table {question: verdict}; counts its calls."""

    def __init__(self, table):
        self.table, self.calls, self.prompts = table, 0, []

    def __call__(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        q = [ln for ln in prompt.splitlines() if ln.startswith("QUESTION: ")][-1][len("QUESTION: "):]
        v = self.table.get(q)
        return v(prompt) if callable(v) else v


def test_the_cold_start_gate_serves_only_near_exact_rewordings():
    m = _mind()
    m.teach(BAL, BAL_A)
    m.teach("how do i change my pin", "Settings > Card > Change PIN.")
    assert m.ask("how do i check my account balance please")["answer"] == BAL_A      # near-exact: served
    assert m.ask("where can i see my account balance")["tier"] == "refused"           # a guess: never served


def test_a_resolved_rewording_is_learned_and_then_served_without_the_model():
    m = _mind()
    m.teach(BAL, BAL_A)
    assert m.ask("how much money do i have left")["tier"] == "refused"      # before: memory alone refuses
    model = _Model({"how much money do i have left":
                    lambda p: json.dumps({"verdict": "same", "row": _rid(m, BAL)})})
    m.zoo_attach(model)
    m.zoo["llm"] = model
    first = m.ask("how much money do i have left")
    assert first["answer"] == BAL_A and first["via"] == "meaning-resolved" and first["learned"]
    assert "CANDIDATES" in model.prompts[0] and _rid(m, BAL) in model.prompts[0]   # typed, with evidence
    m.zoo["llm"] = None                                                        # detach: memory alone now
    again = m.ask("How much money do I have left?")
    assert again["tier"] == "T0" and again["via"] == "meaning" and again["answer"] == BAL_A
    assert len(m.meaning.rows) == 1                                           # a wording, never a duplicate row


def test_what_was_learned_survives_save_and_a_cold_reload(tmp_path):
    m = _mind()
    m.teach(BAL, BAL_A)
    m.meaning_resolve("how much money do i have left", {"verdict": "same", "row": _rid(m, BAL)})
    root = str(tmp_path / "p")
    m.learning_rollover(root)
    m.learning_save(root)
    m2 = _mind()
    m2.learning_rollover(root)
    out = m2.ask("how much money do i have left")
    assert out["via"] == "meaning" and out["answer"] == BAL_A
    assert m2.meaning_report()["wordings"] == 2


def test_a_live_answer_is_remembered_as_a_method_and_never_cached():
    m = _calibrated(_mind())
    ticks = []

    def price(symbol):
        ticks.append(symbol)
        return "%s is $%d" % (symbol, 100 + len(ticks))
    m.meaning_tool_register("price", price)
    out = m.meaning_resolve("what's the price of solana right now", {
        "verdict": "new", "answer": "SOL is $100",
        "method": {"verb": "price", "args": {"symbol": "SOL"}, "from_question": {"symbol": "solana"}, "live": True}})
    assert out["learned"] and out["live"]
    rid = out["row"]
    m.meaning.learn_slot_value(rid, "symbol", "eth", "ETH")
    a1, a2 = m.ask("price of eth right now"), m.ask("price of eth right now")
    assert a1["call"] == {"verb": "price", "args": {"symbol": "ETH"}} and a1["answer"] != a2["answer"]
    assert ticks == ["ETH", "ETH"]                                            # two asks, two fresh calls
    lad = m.zoo["ladder"]
    assert not any("$1" in str(r[1]) for r in getattr(lad, "taught_log", []))    # no price value was ever taught
    assert not any("price of eth" in k for k in getattr(lad, "_exact", {}))


def test_a_bare_coin_is_clarified_never_acted_on():
    m = _mind()
    calls = []
    m.meaning_tool_register("price", lambda symbol: calls.append(symbol) or "x")
    m.meaning_resolve("what's the price of solana", {
        "verdict": "new", "answer": "…",
        "method": {"verb": "price", "args": {"symbol": "SOL"}, "from_question": {"symbol": "solana"}, "live": True}})
    for q in ("solana", "Solana?", "SOLANA!"):
        out = m.ask(q)
        assert out["tier"] == "clarify" and "solana" in out["answer"].lower()
    assert calls == []


def test_a_call_that_would_ignore_part_of_the_question_is_not_made():
    m = _calibrated(_mind())
    calls = []
    m.meaning_tool_register("fx", lambda **kw: calls.append(kw) or "rate")
    m.meaning_resolve("exchange rate between dollars and yen", {
        "verdict": "new", "answer": "…",
        "method": {"verb": "fx", "args": {"from": "USD", "to": "JPY"},
                   "from_question": {"from": "dollars", "to": "yen"}, "live": True}})
    out = m.ask("how many yen for 20 dollars")                              # an amount the method has no slot for
    assert not out.get("call") and calls == []
    # the model says it is the same method WITH an amount: the method grows an optional amount slot
    rid = next(iter(r for r, row in m.meaning.rows.items() if row["kind"] == "method"))
    got = m.meaning_resolve("how many yen for 20 dollars", {"verdict": "same", "row": rid,
                                                             "args": {"from": "USD", "to": "JPY", "amount": "20"}})
    assert got["call"]["args"]["amount"] == "20"
    assert m.ask("what's the exchange rate between dollars and yen for 50")["call"]["args"] == \
        {"from": "USD", "to": "JPY", "amount": "50"}


def test_a_model_that_ignores_the_contract_still_answers():
    m = _mind()
    plain = _Model({"what is the capital of peru": "Lima."})
    m.zoo_attach(plain)
    m.zoo["llm"] = plain
    out = m.ask("what is the capital of peru")
    assert out["tier"] == "T4" and out["answer"] == "Lima." and plain.calls == 1   # no retry for plain text
    assert m.ask("what is the capital of peru")["tier"] == "T0"                   # cached as before (model-cached)


def test_a_broken_verdict_is_retried_once_and_a_foreign_row_is_refused():
    m = _mind()
    m.teach(BAL, BAL_A)
    seq = iter(['{"verdict": "same", "row": "not-a-candidate"}',
                json.dumps({"verdict": "same", "row": _rid(m, BAL)})])
    model = _Model({"how much money is in my account": lambda p: next(seq)})
    m.zoo_attach(model)
    m.zoo["llm"] = model
    out = m.ask("how much money is in my account")
    assert model.calls == 2 and "REJECTED" in model.prompts[1] and out["answer"] == BAL_A


def test_negation_is_meaning():
    m = _calibrated(_mind())
    m.teach("that is right", "Great, glad I got it right.")
    m.meaning_resolve("yes that's correct", {"verdict": "same", "row": _rid(m, "that is right")})
    assert m.ask("that's correct")["via"] == "meaning"
    assert m.ask("that isn't right").get("answer") != "Great, glad I got it right."


def test_a_correction_by_decision_id_teaches_and_stops_the_wrong_serve():
    m = _mind()
    m.teach(BAL, BAL_A)
    m.teach("how do i change my pin", "Settings > Card > Change PIN.")
    m.meaning_resolve("how can i see my account balance", {"verdict": "same", "row": _rid(m, BAL)})
    out = m.ask("how can i see my account pin")
    if out.get("via") == "meaning" and out["row"] == _rid(m, BAL):         # served the wrong row: correct it
        rep = m.decision_outcome(out["id"], _rid(m, "how do i change my pin"))
        assert rep["forwarded"]["meaning"] == "corrected"
        again = m.ask("how can i see my account pin")
        assert again.get("row") != _rid(m, BAL)
    assert m.meaning.calib                                                      # every verdict is a label


def test_a_vetoed_answer_is_not_served_through_meaning():
    m = _mind()
    m.teach(BAL, BAL_A)
    m.meaning_resolve("how much money do i have left", {"verdict": "same", "row": _rid(m, BAL)})
    assert m.ask("how much money do i have left")["via"] == "meaning"
    m.answer_feedback(BAL, ok=False)                                            # the answer itself is wrong
    assert m.ask("how much money do i have left").get("answer") != BAL_A


def test_a_secret_wording_is_never_learned():
    m = _mind()
    m.teach(BAL, BAL_A)
    fake = "my password is Hunter2-FAKE-9c1d what is my balance"
    out = m.meaning_resolve(fake, {"verdict": "same", "row": _rid(m, BAL)})
    assert out["learned"] is False
    assert all("Hunter2" not in p for r in m.meaning.rows.values() for p in r["phrasings"])


def test_sessions_do_not_leak_through_meaning():
    m = _mind()
    m.session_open("alice")
    m.teach("what is my locker number", "Locker 41.")
    m.meaning_resolve("[s:alice] which locker is mine", {"verdict": "same",
                                                         "row": _rid(m, "[s:alice] what is my locker number")})
    assert m.ask("which locker is mine").get("answer") == "Locker 41."
    m.session_open("bob")
    assert m.ask("which locker is mine").get("answer") != "Locker 41."
    m.session_close()
    assert m.ask("which locker is mine").get("answer") != "Locker 41."


def test_the_meaning_index_selftest():
    from holographic.agents_and_reasoning.holographic_meaning import _selftest
    assert _selftest() == "ok"


def test_feedback_on_a_meaning_serve_corrects_the_link_not_the_answer():
    m = _calibrated(_mind())
    m.teach(BAL, BAL_A)
    m.meaning_resolve("how can i see my account balance", {"verdict": "same", "row": _rid(m, BAL)})
    q = "how can i see my account balance today"
    out = m.ask(q)
    assert out["via"] == "meaning"
    rep = m.answer_feedback(q, ok=False)
    assert rep["via"] == "meaning" and rep["marked"] == "bad"
    assert m.ask(q).get("via") != "meaning"                      # this wording is not served that row again
    assert m.ask(BAL)["answer"] == BAL_A                         # ...and the row's own answer is untouched


def test_the_answer_key_survives_save_and_reload(tmp_path):
    """Found by the CLM panel swarm (w4-measure): the answer key was not persisted, so after a reload two rows that
    serve the SAME answer became rivals and split the lead. Both rows must keep one key across a cold reload."""
    m = _mind()
    m.meaning.MERGE_SAME_ANSWER = False             # an OLDER partition's duplicate rows (before the E2.1 merge rule)
    m.teach("how do i check my account balance", BAL_A)
    m.meaning_resolve("show me what is in my account", {"verdict": "new", "answer": BAL_A})   # a duplicate row
    keys = {r["akey"] for r in m.meaning.rows.values()}
    assert len(m.meaning.rows) == 2 and len(keys) == 1
    root = str(tmp_path / "p")
    m.learning_rollover(root)
    m.learning_save(root)
    m2 = _mind()
    m2.learning_rollover(root)
    assert len({r.get("akey") for r in m2.meaning.rows.values()}) == 1


def test_direction_is_learned_from_verdicts_read_by_bind_and_persisted(tmp_path):
    """E3.1 wiring: a verdict's from_question teaches the direction reader which context words mark FROM and TO;
    bind() then reads the direction from the question instead of the method's first example's order, and a question
    that names the values but NO direction (taught as {"either": [...]}) is left open -- asked, never guessed.
    Measured on tests/data/exchange_direction.json through this door: tools/bench_meaning.py direction."""
    m = _calibrated(_mind())
    calls = []
    m.meaning_tool_register("fx", lambda **kw: calls.append(kw) or "rate")
    m.meaning_resolve("convert dollars to euros", {"verdict": "new", "answer": "...", "method": {
        "verb": "fx", "args": {"from": "USD", "to": "EUR"}, "from_question": {"from": "dollars", "to": "euros"},
        "live": True}})
    rid = next(r for r, row in m.meaning.rows.items() if row["kind"] == "method")
    same = [("how many euros can i get for my dollars", "USD", "EUR", "dollars", "euros"),
            ("change my euros into dollars", "EUR", "USD", "euros", "dollars"),
            ("how many yen do i get for euros", "EUR", "JPY", "euros", "yen"),
            ("convert yen to dollars please", "JPY", "USD", "yen", "dollars"),
            ("i want to swap my dollars for yen", "USD", "JPY", "dollars", "yen")]
    for q, f, t, fw, tw in same:
        m.meaning_resolve(q, {"verdict": "same", "row": rid, "args": {"from": f, "to": t},
                              "from_question": {"from": fw, "to": tw}})
    for q, a, b in (("what is the rate between dollars and yen", "dollars", "yen"),
                    ("exchange rate between euros and dollars", "euros", "dollars")):
        m.meaning_resolve(q, {"verdict": "same", "row": rid, "args": {"from": "USD", "to": "JPY"},
                              "from_question": {"either": [a, b]}})
    rd = m._direction_reader()
    assert rd.counts.get("from") == 6 and rd.counts.get("to") == 6 and rd.counts.get("either") == 4
    mi = m.meaning
    # "how many yen for my euros": the POSITIONAL rule (first currency named = from) says from=JPY -- wrong
    args, missing = mi.bind(rid, "how many yen can i get for my euros")
    assert (args["from"], args["to"]) == ("JPY", "EUR") and not missing           # the baseline's mistake
    args, missing = mi.bind(rid, "how many yen can i get for my euros", reader=m._meaning_reader())
    assert (args["from"], args["to"]) == ("EUR", "JPY") and not missing           # read from the context words
    args, missing = mi.bind(rid, "what is the rate between euros and yen", reader=m._meaning_reader())
    assert set(missing) == {"from", "to"}                                          # no direction: ask
    m._meaning_direction = False                                                   # the positional baseline switch
    assert m._meaning_reader() is None
    m._meaning_direction = True
    root = str(tmp_path / "p")
    m.learning_save(root)
    m2 = _mind()
    m2.learning_load(root)
    rd2 = m2._direction_reader()
    assert rd2.counts == rd.counts and rd2.digest() == rd.digest()
    args, _ = m2.meaning.bind(rid, "how many yen can i get for my euros", reader=m2._meaning_reader())
    assert (args["from"], args["to"]) == ("EUR", "JPY")

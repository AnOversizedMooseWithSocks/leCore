"""SelectionLedger (F3) at the seams: driven through the mind, composed with SignalProgram (the exact debt it
was built to cover), and its JSON persistence surviving a round trip with the tamper check live."""
import json

import numpy as np

import lecore
from holographic.agents_and_reasoning.holographic_honesty import bh_fdr


def test_the_faculty_hands_back_a_working_ledger_and_the_book_penalises_the_family_winner():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    led = mind.selection_ledger()
    led.record("winner", 0.0004, family="focused")
    for i in range(60):
        led.record("sweep_%d" % i, 0.5 + i * 0.005, family="sweep")
    assert led.correct(alpha=0.05, family="focused")["n_passed"] == 1
    whole = led.correct(alpha=0.05)
    assert [r for r in whole["rows"] if r["name"] == "winner"][0]["passed"] is False
    assert "everything RECORDED" in whole["scope"]


def test_the_ledger_closes_the_gap_signalprogram_declares():
    """The composition the backlog actually asked for: run TWO batteries on different days' data, keep only
    the second, and let the ledger carry both. A check that passes its own battery must be re-judged against
    the family of everything tried across both."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(0)
    led = mind.selection_ledger()

    # battery 1: 12 noise checks on noise -- refused, discarded... but RECORDED.
    states1 = rng.standard_normal((1200, 12))
    noise_t = rng.standard_normal(1200)
    prog1 = mind.signal_program(dim=256, seed=0)
    for j in range(12):
        prog1.add_check("day1_%d" % j, (lambda jj: (lambda s: s[:, jj]))(j))
    rep1 = prog1.screen(states1, noise_t)
    assert rep1["refused"]
    for row in rep1["checks"]:
        led.record(row["name"], row["p"], family="day1", effect=row["effect"])

    # battery 2: a moderate real effect that passes its battery.
    states2 = rng.standard_normal((1200, 12))
    weak = np.sign(states2[:, 0]) * np.abs(rng.standard_normal(1200)) * 0.35 + rng.standard_normal(1200)
    prog2 = mind.signal_program(dim=256, seed=0)
    for j in range(12):
        prog2.add_check("day2_%d" % j, (lambda jj: (lambda s: s[:, jj]))(j))
    rep2 = prog2.screen(states2, weak)
    for row in rep2["checks"]:
        led.record(row["name"], row["p"], family="day2", effect=row["effect"])

    # the report makes the two-level verdict explicit and the family sizes are stated out loud.
    rep = led.report(alpha=0.1)
    assert rep["total_recorded"] == 24
    assert rep["whole_book"]["family_size"] == 24
    day2 = [f for f in rep["families"] if f["family"] == "day2"][0]
    # whatever passed within day2 is a subset-or-equal of what survives the 24-look book.
    assert day2["passed_on_whole_book"] <= day2["passed_in_family"]


def test_persistence_round_trip_and_the_tamper_refusal_through_plain_json():
    """to_json output is plain JSON (an HTTP payload in waiting); a round trip preserves verdicts exactly, and
    a book with one row quietly removed refuses to load."""
    mind = lecore.UnifiedMind(dim=256, seed=0)
    led = mind.selection_ledger()
    for i in range(10):
        led.record("t%d" % i, (i + 1) / 20.0, family="f")
    s = led.to_json()
    from holographic.agents_and_reasoning.holographic_selectionledger import SelectionLedger
    back = SelectionLedger.from_json(s)
    assert back.correct(alpha=0.1)["rows"] == led.correct(alpha=0.1)["rows"]
    data = json.loads(s)
    data["entries"] = [e for e in data["entries"] if e["name"] != "t3"]
    try:
        SelectionLedger.from_json(json.dumps(data, sort_keys=True))
        raise AssertionError("a tampered book must not load")
    except ValueError:
        pass


def test_verdicts_delegate_to_bh_fdr_exactly():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    led = mind.selection_ledger()
    rng = np.random.default_rng(3)
    ps = [float(p) for p in rng.random(30)]
    for i, p in enumerate(ps):
        led.record("x%d" % i, p)
    rej, _ = bh_fdr(ps, alpha=0.1, dependent=True)
    assert [r["passed"] for r in led.correct(alpha=0.1)["rows"]] == [bool(x) for x in rej]


def test_withdrawal_keeps_the_cost_and_reruns_are_countable():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    led = mind.selection_ledger()
    led.record("a", 0.04, family="f")
    led.record("a", 0.02, family="f")             # the re-run
    led.record("b", 0.5, family="f")
    led.withdraw(2, reason="wrong dataset")
    c = led.correct(alpha=0.1, family="f")
    assert c["family_size"] == 3 and c["n_tested"] == 2
    rep = led.report(alpha=0.1)
    assert rep["families"][0]["reruns"] == 1

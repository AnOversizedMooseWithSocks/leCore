"""tests/test_swarm_learning_races.py -- the three learning-loop faults a 5-worker leStudio swarm found (the maple
field-journal run, 2026-09-27). Each test pins one fix:

  1. an escalation answered through meaning_resolve() (or taught word for word) leaves the escalation ledger --
     MEASURED before: 55 open questions after the run, 26 of them already serving at T0;
  2. a verdict is LABELLED against the candidates the model was shown, not a later re-ranking -- MEASURED before:
     three "new" verdicts that raced another worker's teach were stored as "the top row was wrong at g ~1.7",
     which held the calibrated gate under its bar (1 of 16 fresh rewordings served);
  3. a mind that loaded a dated generation saves back into it -- MEASURED before: a restarted service saved into
     the legacy state.lecore, which the loader never reads while a generation exists.

Every "model" here is a scripted verdict; no secrets are involved.
"""
import os

os.environ.setdefault("PYTHONHASHSEED", "0")

Q_BATCH = "how do i lay many strokes in one request"
A_BATCH = "POST /api/paint_batch with a list of stroke payloads, up to 512 per call."


def _mind():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    # a little unrelated memory so the question escalates with real candidates, as on a live partition
    m.teach("how do i add a new layer", "POST /api/layer with action add.")
    m.teach("how do i clear a layer", "POST /api/layer with action clear and the layer id.")
    return m


def _open(m):
    return [e["question"] for e in m.escalations()]


# ------------------------------------------------------------------------------------------------ 1. the ledger
def test_a_verdict_through_meaning_resolve_clears_the_escalation():
    m = _mind()
    assert m.serve(Q_BATCH)["served"] is False
    assert Q_BATCH in _open(m)
    out = m.meaning_resolve(Q_BATCH, {"verdict": "new", "answer": A_BATCH}, by="w1")
    assert out["cleared"] is True
    assert Q_BATCH not in _open(m)
    assert m.serve(Q_BATCH)["served"] is True                       # and it now serves from memory


def test_an_unclear_verdict_also_clears_it_but_a_rejected_reply_does_not():
    m = _mind()
    m.serve("ETH")
    m.serve("paint a tree")
    assert m.meaning_resolve("ETH", {"verdict": "unclear", "clarify": "the price, or an address?"})["cleared"]
    bad = m.meaning_resolve("paint a tree", {"verdict": "same", "row": "not-a-row"})
    assert bad.get("resolved") is False
    assert _open(m) == ["paint a tree"]                             # still open: nothing was answered


def test_teaching_the_exact_wording_clears_it_and_resolve_still_reports_cleared():
    m = _mind()
    m.serve(Q_BATCH)
    assert m.teach(Q_BATCH, A_BATCH).get("cleared") is True
    assert _open(m) == []
    m.serve("how do i undo a batch")
    assert m.resolve("how do i undo a batch", "ctrl+Z reverts the whole batch", by="w2")["cleared"] is True
    m.serve("how many strokes per batch")
    m.teach("how many strokes per call", "512")                    # a REWORDING does not clear the escalated one
    assert _open(m) == ["how many strokes per batch"]


# ------------------------------------------------------------------------------------------ 2. labels vs races
def test_a_racing_new_verdict_is_labelled_against_what_the_model_saw():
    m = _mind()
    # two workers ask the same question before either answers; both see the same candidates (no exact row)
    m.serve(Q_BATCH)
    m.serve(Q_BATCH)
    n0 = len(m.meaning.calib)
    m.meaning_resolve(Q_BATCH, {"verdict": "new", "answer": A_BATCH}, by="w1")
    # w2's verdict arrives after w1's row exists; re-ranked NOW the top row would be this very wording (score 1.0)
    assert m.meaning.answers(Q_BATCH)[0][1] >= 0.999
    # the packet w2 was shown is gone (popped by w1's resolve) -- the exact-wording rule keeps the label out
    m.meaning_resolve(Q_BATCH, {"verdict": "new", "answer": A_BATCH}, by="w2")
    labels = m.meaning.calib[n0:]
    assert len(labels) == 1                                         # w1's verdict only
    assert all(g < 1.5 for g, _ in labels)                          # never "top row wrong at g ~1.7"
    assert m.meaning.stats.get("race_unlabelled") == 1


def test_the_shown_candidates_are_used_when_memory_moved_since_the_escalation():
    m = _mind()
    packet = m.serve(Q_BATCH)["meaning"]
    shown_top = packet["candidates"][0]["score"]
    m.teach(Q_BATCH, A_BATCH)                                       # another worker teaches it meanwhile
    n0 = len(m.meaning.calib)
    m.meaning_resolve(Q_BATCH, {"verdict": "new", "answer": A_BATCH}, by="w3")
    (g, correct), = m.meaning.calib[n0:]
    assert correct is False or correct == 0
    assert g <= 2 * shown_top + 1e-9                                # the label's confidence is the SHOWN one


def test_the_shown_list_is_bounded():
    m = _mind()
    for k in range(4100):
        m.meaning_packet("question number %d about layers" % k)
    assert len(m._meaning_shown) == 4096


# -------------------------------------------------------------------------------------- 3. the save target
def test_a_mind_that_loaded_a_generation_saves_back_into_it(tmp_path):
    import lecore
    root = str(tmp_path / "part")
    a = _mind()
    a.learning_save(root)
    a.learning_rollover(root)
    gens = sorted(f for f in os.listdir(os.path.join(root, "learning")) if f.startswith("state-"))
    assert len(gens) == 1
    # a restarted service: a fresh mind loads the partition and saves WITHOUT a rollover
    b = lecore.UnifiedMind(dim=256, seed=0)
    b.learning_load(root)
    b.teach("what does a restart keep", "everything saved into the loaded generation")
    got = b.learning_save(root)
    assert os.path.basename(got["path"]) == gens[0]
    assert not os.path.exists(os.path.join(root, "learning", "state.lecore"))
    c = lecore.UnifiedMind(dim=256, seed=0)
    c.learning_load(root)
    assert c.ask("what does a restart keep")["answer"] == "everything saved into the loaded generation"


def test_a_generation_of_another_root_is_never_the_write_target(tmp_path):
    import lecore
    r1, r2 = str(tmp_path / "one"), str(tmp_path / "two")
    a = _mind()
    a.learning_save(r1)
    a.learning_rollover(r1)
    b = lecore.UnifiedMind(dim=256, seed=0)
    b.learning_load(r1)
    got = b.learning_save(r2)                                       # saving somewhere else
    assert os.path.dirname(os.path.abspath(got["path"])) == os.path.abspath(os.path.join(r2, "learning"))


def test_a_question_that_serves_later_leaves_the_ledger():
    m = _mind()
    q = "how do i check my account balance please"
    assert m.serve(q)["served"] is False
    assert q in _open(m)
    m.teach("how do i check my account balance", "Open the app and tap Accounts.")   # taught under ANOTHER wording
    got = m.serve(q)
    assert got["served"] is True                                    # near-exact: the cold gate serves it
    assert q not in _open(m)


def test_every_confirmed_wording_serves_even_on_a_row_with_many_wordings(tmp_path):
    """Row score = 0.2 x best phrasing + 0.8 x centroid, so an EXACT confirmed wording of a row with many varied
    wordings used to score ~0.7 and escalate again (15 of 48 swarm-linked wordings, after a restart). A confirmed
    wording is exact memory: it serves, before and after a reload -- and a vetoed one still does not."""
    import lecore
    m = _mind()
    m.teach(Q_BATCH, A_BATCH)
    rid = next(r for r, row in m.meaning.rows.items() if row["canonical"] == Q_BATCH)
    words = ["can i send a bunch of strokes in a single call", "whats the fastest way to paint lots of strokes",
             "is there a way to batch strokes", "how can i paint hundreds of strokes at once",
             "can it take lots of strokes in one go", "quickest way to send many brush strokes"]
    for w in words:
        assert m.meaning_resolve(w, {"verdict": "same", "row": rid}, by="w1").get("learned")
    root = str(tmp_path / "p")
    m.learning_save(root)
    fresh = lecore.UnifiedMind(dim=256, seed=0)
    fresh.learning_load(root)
    for mind in (m, fresh):
        for w in words:
            got = mind.serve(w)
            assert got["served"] is True and got["answer"] == A_BATCH, (w, got.get("via"))
    m._meaning_neg_add(words[0], rid)                               # a correction: this wording is NOT that row
    assert m.serve(words[0])["served"] is False

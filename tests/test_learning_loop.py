"""tests/test_learning_loop.py -- the learning loop ACROSS A RESTART, door by door (learning-loop audit, 2026-09-26).

The owner: "I want to be sure that our self improvement and self learning process is working as best it can." Each
door's own tests check one hop (learns live; round-trips through learning_save). tools/audit_learning_loop.py walks
the loop the way a deployment does -- decide, RESTART (learning_rollover in a fresh mind), report the outcome by id
AFTERWARDS, check the door learned, restart again, check it kept it, and check no (fake) secret reached the disk --
and found doors that learned live but trained NOTHING from an outcome reported after a restart, a guard that wrote
refused secrets to disk, and a correction that vetoed the wrong question. Each test below pins one fix; the numbers
before / after are docs/research/evidence/audit_learning_loop.json (runs.before / runs.after).

Owner's rules pinned here: never learn secrets; a correction must actually fix the answer; learned state must survive
a restart. Every secret is FAKE ("Hunter2-FAKE-9c1d"); all data is hand-written.
"""
import json
import os

import numpy as np
import pytest

os.environ.setdefault("PYTHONHASHSEED", "0")

SECRET = "Hunter2-FAKE-9c1d"


def _mind():
    import lecore
    return lecore.UnifiedMind(dim=256, seed=0)


def _restart(m, root, factory=_mind):
    """Save, then a COLD boot of a fresh mind on the partition (what the service / agent_boot do)."""
    m.learning_save(root)
    m2 = factory()
    m2.learning_rollover(root)
    return m2


def _boot(root, factory=_mind):
    m = factory()
    m.learning_rollover(root)
    return m


def _secret_on_disk(root, needle=SECRET):
    """Any container section under root whose meta holds the secret (raw or in a normalised question's form)."""
    import glob
    import re
    from holographic.io_and_interop.holographic_container import load_container
    forms = {needle.lower(), re.sub(r"[^a-z0-9]+", " ", needle.lower()).strip()}
    hits = []
    for f in sorted(glob.glob(os.path.join(root, "learning", "*.lecore"))):
        for sec in load_container(open(f, "rb").read())["sections"]:
            blob = json.dumps(sec["meta"], default=str).lower()
            if any(x in blob or x in re.sub(r"[^a-z0-9]+", " ", blob) for x in forms):
                hits.append(sec["kind"])
    return hits


# ============================================================ 1. outcome hooks that were live closures are now specs
def test_the_ledger_persists_a_keyed_closure_as_a_spec_and_never_runs_it_twice():
    from holographic.agents_and_reasoning.holographic_decisionrecord import _selftest_spec_keys
    assert _selftest_spec_keys()["ok"] is True


TIE_Q, TIE_USED = "denoise an image", "Denoise (domain)"        # a three-way lexical tie (tests/test_router_learning)


def test_a_route_outcome_reported_after_a_restart_teaches_the_router(tmp_path):
    """BEFORE: route_tiered's router hook was a live closure (hook_key 'router') -- a route outcome reported after a
    restart fed the route calibrator and the bridge, but the router's ProtoStore got 0 of 1 verdicts."""
    root = str(tmp_path / "p")
    m = _boot(root)
    m.router_mode(True)
    r = m.route_tiered(TIE_Q)
    assert m.decision_ledger().get(r["id"]).meta["hook"] == {"kind": "router"}     # the record names its door
    m2 = _restart(m, root)
    m2.router_mode(True)
    # the SAME decision made again in the restarted mind re-attaches the live closure to the reloaded record (same
    # id): the report must run the router ONCE, not once for the closure and once for the reloaded spec
    assert m2.route_tiered(TIE_Q)["id"] == r["id"]
    m2.decision_outcome(r["id"], TIE_USED)
    assert m2.router_report()["verdicts"] == 1
    assert m2.route_tiered(TIE_Q)["options"][0]["name"] == TIE_USED              # the next decision changed
    m3 = _restart(m2, root)
    assert m3.router_report()["verdicts"] == 1 and m3.router_mode() is True       # kept, and the switch with it


def _tool_mind():
    m = _mind()
    m.api_use = lambda service, endpoint, params=None, headers=None: {"ok": True, "data": {"endpoint": endpoint}}
    return m


TOOL_Q = "convert celsius fahrenheit units please"
TOOL_Q2 = "please convert these celsius temperatures to fahrenheit units"


def test_a_tool_correction_reported_after_a_restart_moves_the_tool_door(tmp_path):
    """BEFORE: serve()'s tool-door hook was a live closure (hook_key 'tooldoor'): 0 verdicts after a restart, and the
    next paraphrase was still picked by word overlap (f_to_c) instead of the corrected tool."""
    root = str(tmp_path / "p")
    m = _boot(root, _tool_mind)
    m.tool_reflex_teach("convert 77 fahrenheit into celsius units", "convertd", "f_to_c")
    m.tool_reflex_teach("convert 25 celsius into fahrenheit units", "convertd", "c_to_f")
    r = m.serve(TOOL_Q)
    assert r["tool"] == "convertd.f_to_c"                                         # word overlap's cold-start pick
    m2 = _restart(m, root, _tool_mind)
    m2.decision_outcome(r["id"], "convertd.c_to_f")                               # the judge: the OTHER tool was right
    assert m2._tool_audit()["verdicts"] == 1
    r2 = m2.serve(TOOL_Q2)
    assert r2["tool"] == "convertd.c_to_f" and r2["picked_by"] == "proto"


def test_a_compose_outcome_reported_after_a_restart_calibrates_the_compose_door(tmp_path):
    root = str(tmp_path / "p")
    m = _boot(root)
    v = m.call_encode("fx", {"from": "EUR", "to": "JPY"})["vector"]
    c = m.call_compose(v, {"fx": ["from", "to"]}, {"from": ["EUR", "JPY", "USD"], "to": ["EUR", "JPY", "USD"]})
    assert m.decision_ledger().get(c["id"]).meta["hook"]["kind"] == "calibrate"
    m2 = _restart(m, root)
    rep = m2.decision_outcome(c["id"], c["call"])
    assert rep["forwarded"]["door"] == "compose" and m2.door_calibration_report()["compose"]["labels"] == 1


def test_a_clm_outcome_reported_after_a_restart_calibrates_the_clm_door(tmp_path):
    import lecore
    from tools.audit_learning_loop import CLM_Q, _ClmStub
    stub = _ClmStub()
    try:
        def factory():
            mm = lecore.UnifiedMind(dim=64, seed=0, plugins=())
            mm._plugin_load("clm", config={"url": stub.url}, register_in_catalog=False)
            return mm
        root = str(tmp_path / "p")
        m = _boot(root, factory)
        r = m.clm_systemone("courier lost my parcel", CLM_Q)["topic"]
        m2 = _restart(m, root, factory)
        rep = m2.decision_outcome(r["id"], "shipping")
        assert rep["forwarded"]["door"] == "clm:topic" and rep["forwarded"]["n"] == 1
        # a yes/no-style strictness check of the spec: an outcome that names neither option confirms nothing
        r2 = m2.clm_systemone("charged twice on my card", CLM_Q)["topic"]
        assert m2.decision_outcome(r2["id"], "maybe")["forwarded"]["n"] == 2
        assert [y for _, y in m2.door_calibrator("clm:topic").pairs] == [True, False]
    finally:
        stub.close()


# ============================================================ 2. the rank door can re-encode a set after a restart
@pytest.mark.parametrize("cands,truth", [
    (["reset my password", "cancel my subscription", "track my parcel", "update billing address"],
     "cancel my subscription"),
    ([{"text": "refund", "id": "refund", "examples": ["money back please", "return my money"]},
      {"text": "deliver", "id": "deliver", "examples": ["where is my order"]}], "refund")])
def test_a_rank_outcome_reported_after_a_restart_moves_the_rank_door(tmp_path, cands, truth):
    """BEFORE: 'the truth's encoding is not in this process -- rank the set again': only the candidates' digests were
    persisted, so 0 of 2 outcomes learned after a restart. Now a plain string re-encodes from its label (the digest
    proves it) and a dict candidate from its registered (text, examples)."""
    root = str(tmp_path / "p")
    m = _boot(root)
    r = m.rank("where is my package right now" if isinstance(cands[0], str) else "i want my money returned", cands)
    m2 = _restart(m, root)
    rep = m2.decision_outcome(r["id"], truth)
    assert rep["forwarded"]["learned"] is True
    # the learned rows are exactly what the live process would have learned (same deterministic encoder)
    m_live = _mind()
    r_live = m_live.rank("where is my package right now" if isinstance(cands[0], str) else "i want my money returned",
                         cands)
    m_live.decision_outcome(r_live["id"], truth)
    assert m2.protostore("rank").digest() == m_live.protostore("rank").digest()


def test_a_rank_set_whose_texts_carry_a_secret_is_never_registered():
    m = _mind()
    r = m.rank("which one", [{"text": "rotate", "id": "a", "examples": ["my password is %s" % SECRET]},
                             {"text": "keep", "id": "b"}])
    rs = m._rank_state()
    assert m.decision_ledger().get(r["id"]).meta["set"] not in rs["sets"]


# ============================================================ 3. what a restart used to forget
def test_the_retile_keeps_the_reported_failures():
    """BEFORE: reflex_retile rebuilt the trace from the audit only -- failure-field norm 2.23 -> 0.00, so the outcome
    gate reopened on the look-alike traps it was measured to catch."""
    m = _mind()
    rng = np.random.default_rng(0)
    for _ in range(5):
        k = rng.standard_normal(2048)
        m.reflex_write(k, rng.standard_normal(2048))
        m.reflex_outcome(k, False)
    s0, f0 = m.experience.outcome_fields()
    m.reflex_retile(advisory_load=0.03)
    assert np.linalg.norm(f0) > 0
    assert all(np.allclose(t.outcome_fields()[1], f0) for t in m.experience.tiles)     # every new tile holds it
    probe = f0 / np.linalg.norm(f0)
    assert m.experience.read_gated(probe)["why"] == "outcome-memory"          # the gate still refuses the region


def test_open_escalations_survive_a_restart_and_a_secret_one_is_never_written(tmp_path):
    root = str(tmp_path / "p")
    m = _boot(root)
    m.serve("which colour is the owner's bicycle")
    m.serve("which colour is the owner's bicycle")
    m.serve("my password is %s what is it" % SECRET)
    assert len(m.escalations()) == 2
    m2 = _restart(m, root)
    assert [(e["question"], e["count"]) for e in m2.escalations()] == [("which colour is the owner's bicycle", 2)]
    assert not _secret_on_disk(root)
    assert m2.resolve("which colour is the owner's bicycle", "red")["cleared"] is True    # was False before
    m3 = _restart(m2, root)
    assert m3.escalations() == [] and m3.ask("which colour is the owner's bicycle")["answer"] == "red"


Q = {"cat": {"type": "choice", "options": ["billing", "shipping"],
             "examples": {"billing": ["card charged twice", "refund my invoice fee"],
                          "shipping": ["parcel lost in transit", "courier delivery late"]}}}


def test_verify_profile_and_drift_stream_survive_a_restart(tmp_path):
    """BEFORE: profile 0.503 and drift_z 0.152 before a restart, None and None after (the drift veto off until 8 new
    reports)."""
    root = str(tmp_path / "p")
    m = _boot(root)
    for s, t in [("my card was charged twice this month", "billing"), ("the courier never showed up", "shipping"),
                 ("refund the double charge on my invoice", "billing"), ("where is my parcel", "shipping"),
                 ("i was billed twice for one order", "billing"), ("the package arrived late", "shipping"),
                 ("charge on my card is wrong", "billing"), ("my order is lost", "shipping"),
                 ("double billed again", "billing")]:
        m.decision_outcome(m.systemone_decide(s, Q, scorer="nb", encoder="ngram")["cat"]["id"], t)
    v1 = m.verify_decision("my card was charged twice this month", "billing")["checks"]
    m2 = _restart(m, root)
    v2 = m2.verify_decision("my card was charged twice this month", "billing")["checks"]
    assert v1["profile"] is not None and v2["profile"] == pytest.approx(v1["profile"], abs=1e-9)
    assert v2["drift_z"] is not None
    assert m2._verify_n == m._verify_n


def test_a_section_this_build_does_not_know_survives_a_rollover(tmp_path):
    """The container promises that a reader keeps kinds it does not understand; learning_load dropped them, so a
    newer build's section was lost at the next save. One carrying a secret is still never re-written."""
    import glob
    from holographic.io_and_interop.holographic_container import load_container, save_container
    root = str(tmp_path / "p")
    m = _boot(root)
    m.teach("seed row", "a")
    m.learning_save(root)
    f = sorted(glob.glob(os.path.join(root, "learning", "*.lecore")))[-1]
    got = load_container(open(f, "rb").read())
    secs = got["sections"] + [
        {"kind": "lecore.future.section", "id": "v9", "meta": {"keep": "me"}, "arrays": {"x": np.arange(3.0)}},
        {"kind": "lecore.future.leaky", "id": "v1", "meta": {"note": "my password is %s" % SECRET}, "arrays": {}}]
    open(f, "wb").write(save_container(secs, meta=got["meta"]))
    m2 = _boot(root)
    kinds = [s["kind"] for f2 in glob.glob(os.path.join(root, "learning", "*.lecore"))
             for s in load_container(open(f2, "rb").read())["sections"]]
    assert "lecore.future.section" in kinds and "lecore.future.leaky" not in kinds
    sec = next(s for f2 in glob.glob(os.path.join(root, "learning", "*.lecore"))
               for s in load_container(open(f2, "rb").read())["sections"] if s["kind"] == "lecore.future.section")
    assert sec["meta"] == {"keep": "me"} and list(sec["arrays"]["x"]) == [0.0, 1.0, 2.0]
    assert m2.learning_save(root)["unknown_sections_kept"] == 1


def test_the_direction_reader_survives_without_meaning_rows_and_never_learns_a_secret(tmp_path):
    root = str(tmp_path / "p")
    m = _boot(root)
    m.direction_learn("convert dollars to pesos", {"from": "dollars", "to": "pesos"})
    m.direction_learn("how many yen for 20 euros", {"from": "euros", "to": "yen"})
    before = m.direction_read("change pesos into dollars", ["pesos", "dollars"])
    m2 = _restart(m, root)
    after = m2.direction_read("change pesos into dollars", ["pesos", "dollars"])
    assert after["via"] == before["via"] != "positional" and after["assignment"] == before["assignment"]
    refused = m2.direction_learn("my password is %s, convert dollars to pesos" % SECRET,
                                 {"from": "dollars", "to": "pesos"})
    assert refused["learned"] is False and not _secret_on_disk(root)


# ============================================================ 4. never learn (or write) a secret
def test_the_guard_never_writes_a_refused_secret_to_disk(tmp_path):
    """BEFORE: a pattern refusal taught the semantic guard the refused QUESTION verbatim, and learning_save wrote
    '... hunter2 fake 9c1d ...' into lecore.learning.guard. Now: learn_guard_example learns the REDACTED question,
    the save drops any example that carries a secret, and a partition that already holds one is cleaned on load."""
    from holographic.io_and_interop.holographic_container import load_container, save_container
    root = str(tmp_path / "p")
    m = _boot(root)
    assert m.teach("my password is %s please remember it" % SECRET, "ok noted")["taught"] is False
    assert m.teach("remember my api key sk-FAKE0123456789abcdefghijklmnop", "stored")["taught"] is False
    got = m.learn_guard_example("my password is %s, is that a credential" % SECRET, "credential")
    assert got["learned"] is True                                                  # learned REDACTED:
    assert "my password is redacted is that a credential" in m.semantic_guard.learned["credential"]
    # (the auto-learn inside teach()'s refusal still keeps the question verbatim IN MEMORY until the learnguard fix
    # proposed to its owner lands; what this pass guarantees is that it never reaches the disk -- below)
    m.learn_guard_example("what is the recovery code for my vault", "credential")    # an ordinary example
    m.learning_save(root)
    assert not _secret_on_disk(root) and not _secret_on_disk(root, "sk-FAKE0123456789abcdefghijklmnop")
    # a partition written BEFORE the fix: its guard section holds the normalised secret -- dropped on the way in
    import glob
    f = sorted(glob.glob(os.path.join(root, "learning", "*.lecore")))[-1]
    got = load_container(open(f, "rb").read())
    for s in got["sections"]:
        if s["kind"] == "lecore.learning.guard":
            s["meta"]["learned"]["credential"].append("my password is hunter2 fake 9c1d please remember it")
    open(f, "wb").write(save_container(got["sections"], meta=got["meta"]))
    m2 = _mind()
    rep = m2.learning_load(root, force=True)
    assert rep["guard_examples_dropped_sensitive"] >= 1
    assert not any("hunter2" in q for q in m2.semantic_guard.learned["credential"])
    assert "what is the recovery code for my vault" in m2.semantic_guard.learned["credential"]


# ============================================================ 5. a correction fixes ITS answer, and only its answer
def test_payload_keys_follow_a_tile_split_and_a_correction_vetoes_only_what_it_corrected(tmp_path):
    """BEFORE: after one live tile split 60 of 90 payload keys no longer located their own question, and three
    answer_feedback corrections vetoed two OTHER questions after a restart. Now the keys follow the layout and a
    correction marks the payload of the question it names (never a neighbour's fired atom)."""
    from tools.audit_learning_loop import _stale_payload_keys
    root = str(tmp_path / "p")
    m = _boot(root)
    qs = ["what is the storage code for warehouse %s shelf %s" % (w, s)
          for w in ("north", "south", "east", "west", "upper", "lower", "inner", "outer")
          for s in ("alpha", "beta", "gamma", "delta", "omega", "kappa", "sigma", "theta")]
    for i, q in enumerate(qs):
        m.teach(q, "bin %d" % i)
    assert m.experience.splits >= 1 and _stale_payload_keys(m) == 0
    for i in (0, 1, 2):
        assert m.answer_feedback(qs[i], ok=False)["marked"] == "bad"
    lost_live = [i for i, q in enumerate(qs) if m.ask(q).get("answer") != "bin %d" % i]
    m2 = _restart(m, root)
    lost = [i for i, q in enumerate(qs) if m2.ask(q).get("answer") != "bin %d" % i]
    assert lost_live == [0, 1, 2] and lost == [0, 1, 2]


def test_a_retile_remaps_the_payload_keys_too():
    m = _mind()
    qs = ["where is the spare key number %d kept" % i for i in range(20)]
    for i, q in enumerate(qs):
        m.teach(q, "drawer %d" % i)
    m.reflex_retile(advisory_load=0.01)                     # a new trace object, a new tiling
    from tools.audit_learning_loop import _stale_payload_keys
    m.ask(qs[0])                                            # any read brings the keys into step
    assert _stale_payload_keys(m) == 0
    assert all(m.ask(q + " please")["answer"] == "drawer %d" % i for i, q in enumerate(qs))


# ============================================================ 6. every door's store, and an honest save report
def test_a_superposed_doors_prototypes_and_relevances_survive_a_restart(tmp_path):
    """BEFORE: the decisions section carried only the 'rank' and 'verify:<tool>' stores, so any other door's
    ProtoStore (a superposed_door; a door merged in from the commons) and its channel relevances were lost."""
    root = str(tmp_path / "p")
    m = _boot(root)
    door = m.superposed_door("audit_door", channels=("ngram", "word"))
    door.add_option("billing", ["card charged twice", "refund my fee"])
    door.add_option("shipping", ["parcel lost in transit", "courier late"])
    for text, truth in [("double charge on my card", "billing"), ("my parcel never came", "shipping"),
                        ("refund the fee please", "billing")]:
        door.learn(text, truth)
    before = door.rank("refund my double charge")
    m2 = _restart(m, root)
    after = m2.superposed_door("audit_door", channels=("ngram", "word")).rank("refund my double charge")
    assert [l for l, _ in after] == [l for l, _ in before]
    assert [s for _, s in after] == pytest.approx([s for _, s in before], abs=1e-9)
    assert m2.superposed_relevance("audit_door").n == m.superposed_relevance("audit_door").n > 0


def test_learning_save_names_the_sections_whose_content_changed(tmp_path):
    """drift_vs_previous_save counts sections whose meta PREFIX changed (blind to arrays, so a correction learned in
    the trace barely moves it); sections_changed names the sections whose full content changed."""
    root = str(tmp_path / "p")
    m = _boot(root)
    m.teach("seed row", "a")
    m.learning_save(root)
    assert m.learning_save(root)["sections_changed"] == []
    a = m.systemone_decide("the invoice for my parcel shipment is wrong", Q, scorer="contrastive",
                           encoder="ngram")["cat"]
    m.learning_save(root)
    m.decision_outcome(a["id"], "shipping")
    rep = m.learning_save(root)
    assert {"lecore.learning.decisions", "lecore.learning.experience"} <= set(rep["sections_changed"])
    assert "lecore.learning.taught" not in rep["sections_changed"]


def test_the_live_decision_ledger_is_capped_and_an_evicted_id_fails_loudly():
    """The live ledger had no bound while lookups add a record each (learning-loop audit, 2026-09-27). It keeps the
    newest MAX_LIVE records; an evicted id raises KeyError like an id that fell off the saved cap -- never a silent
    wrong door -- and the newest records still report."""
    import pytest
    from holographic.agents_and_reasoning.holographic_decisionrecord import DecisionRecord, Ledger
    L = Ledger()
    assert L.max_live == Ledger.MAX_LIVE >= 4 * 4096
    L.max_live = 5
    ids = [L.add(DecisionRecord("state %d" % i, "q", ["a", "b"], "a", "typed")) for i in range(8)]
    assert len(L._rows) == len(L._order) == 5 and ids[0] not in L._rows
    with pytest.raises(KeyError):
        L.report(ids[0], "a")
    assert L.report(ids[-1], "a")["was_correct"] is True

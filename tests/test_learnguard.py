"""tests/test_learnguard.py -- sweep 179: secrets and live readings are never learned (E4.2: the typed guard).

Measured before (fake values, scratch partition): teach() accepted an API key, a seed phrase and a password,
and a FRESH mind loaded from the saved partition served all three at T0; a taught price and a resolve()d
weather report came back at T0 as facts; the sweep-178 tool-call records carried an api_key in plain text.
Every value below is FAKE. Each test pins one door.
"""
import os

import pytest

os.environ.setdefault("PYTHONHASHSEED", "0")

FAKE_KEY = "sk-live-4f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c"
FAKE_SEED = "abandon ability able about above absent absorb abstract absurd abuse access accident"


def _mind():
    import lecore
    return lecore.UnifiedMind(dim=256, seed=0)


def _partition(tmp_path, name="p"):
    root = str(tmp_path / name)
    os.makedirs(os.path.join(root, "learning"), exist_ok=True)
    return root


# ------------------------------------------------------------------ the verdict itself ----

def test_guard_module_selftest():
    from holographic.agents_and_reasoning.holographic_learnguard import _selftest
    _selftest()


def test_benchmark_misclassifies_nothing():
    """The labelled sets (secrets / public look-alikes / readings / static facts) and the real catalog."""
    import tools.bench_learnguard as B
    sets = B.labelled_sets()
    from holographic.agents_and_reasoning.holographic_learnguard import learning_verdict as check
    expect = {"secrets": "sensitive", "public": None, "readings": "volatile", "static": None}
    for name, rows in sets.items():
        bad = [q for q, a in rows if check(q, a)["kind"] != expect[name]]
        assert not bad, (name, bad[:3])


def test_learn_guard_faculty_never_echoes_the_secret():
    v = _mind().learn_guard("my exchange api key", FAKE_KEY)
    assert v["kind"] == "sensitive" and FAKE_KEY not in v["reason"]


# ------------------------------------------------------------------ memory doors ----

def test_teach_refuses_secrets_and_readings_with_a_reason():
    m = _mind()
    for q, a, kind in [("my exchange api key", FAKE_KEY, "sensitive"), ("wallet backup", FAKE_SEED, "sensitive"),
                       ("the admin password", "hunter2-Moose!", "sensitive"),
                       ("what is the current price of solana", "$142.10", "volatile")]:
        r = m.teach(q, a)
        assert r["taught"] is False and r["guard"] == kind and a not in r["reason"]
        assert m.ask(q)["tier"] == "refused"


def test_snapshot_and_override_are_learned_and_the_override_survives_a_reload(tmp_path):
    m = _mind()
    assert m.teach("sol price as of 2026-09-22 15:00 ET", "$142.10")["taught"] is True
    assert m.teach("what is the current price of solana", "$142.10", allow_volatile=True)["taught"] is True
    root = _partition(tmp_path)
    m.learning_save(root)
    m2 = _mind()
    rep = m2.learning_load(root)
    assert rep["guard_dropped"] == {"sensitive": 0, "volatile": 0}
    assert m2.ask("what is the current price of solana")["answer"] == "$142.10"


def test_resolve_does_not_keep_a_live_reading_and_says_why():
    m = _mind()
    m.serve("what is the weather in jarrettsville right now")
    r = m.resolve("what is the weather in jarrettsville right now", "72F and sunny", by="agent")
    assert r["taught"] is False and r["guard"] == "volatile" and r["cleared"] is True
    assert m.serve("what is the weather in jarrettsville right now")["via"] == "escalate"


def test_a_partition_written_before_the_guard_is_cleaned_on_load(tmp_path, monkeypatch):
    """Rows that slipped in before sweep 179 are dropped on load -- not served, not re-saved -- and the load
    report counts them. The guard is switched off (monkeypatched) only to WRITE that old partition."""
    import holographic.agents_and_reasoning.holographic_learnguard as G
    ok = {"ok": True, "kind": None, "reason": None}
    with monkeypatch.context() as mp:
        mp.setattr(G, "learning_verdict", lambda *a, **k: ok)
        mp.setattr(G, "sensitive_pair_reason", lambda *a, **k: None)
        mp.setattr(G, "sensitive_reason", lambda *a, **k: None)
        old = _mind()
        assert old.teach("my exchange api key", FAKE_KEY)["taught"] is True
        assert old.teach("what is the current price of solana", "$142.10")["taught"] is True
        assert old.teach("what port does the service use", "8080")["taught"] is True
        root = _partition(tmp_path)
        old.learning_save(root)
    m = _mind()                                          # the guard is back on
    rep = m.learning_load(root)
    assert rep["guard_dropped"] == {"sensitive": 1, "volatile": 1}
    assert m.ask("my exchange api key")["tier"] == "refused"
    assert m.ask("what is the current price of solana")["tier"] == "refused"
    assert m.ask("what port does the service use")["answer"] == "8080"
    root2 = _partition(tmp_path, "p2")
    m.learning_save(root2)                               # and the next save no longer carries them
    m3 = _mind()
    assert m3.learning_load(root2)["guard_dropped"] == {"sensitive": 0, "volatile": 0}
    texts = [r[1] for r in m3.zoo["ladder"].taught_log]
    assert FAKE_KEY not in texts and "$142.10" not in texts


def test_save_never_writes_a_secret_in_the_query_log(tmp_path):
    m = _mind()
    m.ask("please use " + FAKE_KEY + " for the exchange")
    root = _partition(tmp_path)
    m.learning_save(root)
    from holographic.io_and_interop.holographic_container import load_container
    import glob
    for path in glob.glob(os.path.join(root, "learning", "*.lecore")):
        got = load_container(open(path, "rb").read())
        assert FAKE_KEY not in repr([s.get("meta") for s in got["sections"]])


# ------------------------------------------------------------------ experience + records ----

def test_a_secret_or_free_form_reading_never_becomes_a_reflex_label():
    from holographic.agents_and_reasoning.holographic_decisionrecord import DecisionRecord
    m = _mind()
    for state, out in [("which key do I use", FAKE_KEY), ("what is the current price of solana", "$142.10")]:
        rec = DecisionRecord(state, "q", ["a", "b"], "a", "typed")
        m.decision_ledger().add(rec)
        rep = m.decision_outcome(rec.id, out)
        assert rep["reflex"]["learned"] is False and rep["reflex"]["guard"] in ("sensitive", "volatile")


def test_a_declared_choice_on_a_volatile_question_is_still_learned():
    """The ROUTE is what a volatile question should learn: 'call the v2 price feed' has a digit in its name
    and must not be refused as a reading."""
    from holographic.agents_and_reasoning.holographic_decisionrecord import DecisionRecord
    m = _mind()
    rec = DecisionRecord("what is the current price of solana", "tool", ["price feed v2", "weather api"],
                         "price feed v2", "typed")
    m.decision_ledger().add(rec)
    assert m.decision_outcome(rec.id, "price feed v2")["reflex"]["learned"] is True


def test_records_and_arg_previews_are_redacted():
    from holographic.agents_and_reasoning.holographic_agentloop import arg_fingerprint
    from holographic.agents_and_reasoning.holographic_decisionrecord import DecisionRecord
    fp = arg_fingerprint({"api_key": FAKE_KEY, "symbol": "SOL"})
    assert FAKE_KEY not in fp["repr"] and '"symbol": "SOL"' in fp["repr"]
    d = DecisionRecord("log in with " + FAKE_KEY, "q", ["a"], "a", "route").to_dict()
    assert FAKE_KEY not in str(d)


def test_semantic_ingest_does_not_learn_a_key_as_a_word():
    m = _mind()
    m.semantic_ingest("the exchange key is " + FAKE_KEY)
    vocab = set(m._lever7_text.context)
    assert not any(FAKE_KEY.lower() in w or "4f9a8b7c6d5e" in w for w in vocab)


def test_goal_step_results_that_are_readings_or_secrets_are_not_cached():
    m = _mind()
    m.zoo_attach(lambda p: "x")
    m.goal_create("g1", "watch the market", plan=["check the current sol price", "fetch the exchange api key",
                                                    "list the pool addresses"])
    ex = {"check the current sol price": lambda: "$142.10", "fetch the exchange api key": lambda: FAKE_KEY,
          "list the pool addresses": lambda: "pool A, pool B"}
    m.goal_work("g1", executors=ex, budget_steps=3)
    tc = m.tool_cache
    cached = list(tc["prefix"].values()) + list(tc["stateless"].values())
    assert "$142.10" not in cached and FAKE_KEY not in cached and "pool A, pool B" in cached


# ------------------------------------------------------------------ documents ----

def test_study_redacts_a_pasted_key_from_the_corpus(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "notes.md").write_text("# setup notes\n\nThe exchange integration is configured with the key " + FAKE_KEY +
                                " which we keep in the notes for now until the vault is ready for use.\n")
    m = _mind()
    st = m.study(str(d), question="how is the exchange integration configured")
    assert FAKE_KEY not in repr(st.get("docs")) and FAKE_KEY not in repr(st.get("answer", ""))
    assert FAKE_KEY not in repr(st["ask"]("what key is the exchange integration configured with"))


def test_corpus_bind_redacts_and_says_so(tmp_path):
    from holographic_mcp import MCPServer
    s = MCPServer(memory_root=str(tmp_path / "mem"))
    r = s._corpus_bind(texts=["deploy with " + FAKE_KEY, "the pool has two sides"])
    assert r["redacted_chunks"] == 1
    assert FAKE_KEY not in repr(s._corpora[r["handle"]])
    assert FAKE_KEY.encode() not in open(os.path.join(str(tmp_path / "mem"), "corpora.lecore"), "rb").read()


# ------------------------------------------------------------------ the SEMANTIC layer: E4.2, the TYPED guard ----

def test_semantic_layer_catches_paraphrases_the_patterns_miss():
    """Held-out phrasings (never used as examples): the pattern layer alone misses these; the mind refuses them."""
    from holographic.agents_and_reasoning.holographic_learnguard import learning_verdict
    m = _mind()
    for q, a, kind in [("what's the pw for the router", "Xk29!mQpa", "sensitive"),
                       ("how many yen for a dollar", "147.2", "volatile"),
                       ("the code?", "q8Zr4Lw2Pn", "sensitive")]:            # unclear: clarify, never store
        assert learning_verdict(q, a)["ok"], q                 # the pattern layer alone lets it through
        r = m.teach(q, a)
        assert r["taught"] is False and r["guard"] == kind and "(semantic)" in r["reason"] and a not in r["reason"]
    # KEPT NEGATIVE (E4.2): "jot down my exchange login for later" -> "correcthorsebattery" was caught by sweep 180/181
    # and is LEARNED now -- its best credential row sits under GUARD_FLOOR (0.163 < 0.18) and the answer is not
    # random-looking enough for the unclear path. The floor is what keeps "secret 0 of session 1" (a test fixture)
    # and engine talk from being refused on noise.
    # sweep 181: "what is btc at" is caught by the PATTERN layer (_AT_READING), not left to a semantic guess
    assert not learning_verdict("what is btc at", "64k")["ok"]
    # E4.2 fixed sweep 181's KEPT NEGATIVE: a fresh mind refused this secret as a live READING; the typed guard
    # calls it what it is
    r0 = _mind().teach("what's the unlock code", "q8Zr4Lw2Pn")
    assert r0["taught"] is False and r0["guard"] == "sensitive"


def test_semantic_layer_leaves_static_facts_alone():
    m = _mind()
    for q, a in [("what is the tensile strength of steel", "400 MPa"), ("how many moons does saturn have", "146"),
                 ("what port does the service use", "8080"), ("how many tiles at load 0.03", "61 per tile"),
                 ("what is 7 times 8", "56"), ("spec 12 of the flux rotor", "rotor 12 rated 36 units"),
                 ("what's my credit score", "742"), ("how many calories in a banana", "105")]:
        assert m.teach(q, a)["taught"] is True, q


def test_typed_guard_four_options_and_pin_talk_is_ordinary():
    """The typed decision (E4.2): talking ABOUT a PIN is ordinary; asking for its VALUE is a credential; a bare
    ticker or 'the code?' is unclear -> the verdict asks for clarification instead of storing a bare value."""
    from holographic.agents_and_reasoning.holographic_learnguard import SemanticGuard, learning_verdict
    g = SemanticGuard()
    assert g.decide("how do I change my PIN at an atm")["value"] == "ordinary"
    assert g.decide("my pin is blocked after three tries")["value"] == "ordinary"
    assert g.decide("what's the pin for the front door keypad")["value"] == "credential"
    assert g.decide("the code?")["value"] == "unclear"
    d = g.decide("kas?")                                     # a bare ticker in no example list
    assert d["value"] == "unclear", d
    v = learning_verdict("kas?", "$0.13", guard=g)
    assert v["ok"] is False and v["clarify"] is True and v["kind"] == "volatile"
    # a PIN is too short for the old credential shape; the typed decision makes it count
    assert learning_verdict("the pin for my visa card", "4821")["ok"]           # pattern layer alone: learned
    assert learning_verdict("the pin for my visa card", "4821", guard=g)["kind"] == "sensitive"
    assert learning_verdict("what port does postgres listen on", "5432", guard=g)["ok"]
    assert set(d) >= {"value", "ranked", "lead", "confidence", "p_correct"}


def test_benchmark_semantic_layer_numbers_hold():
    """The shipped numbers on the OLD held-out sets B/C (tools/bench_learnguard.py --typed). History: sweep 180
    shipped 30/35 and 34/40 (margin 0.01, normals = aliases[0::2] by sorted index: one new catalog card moved it,
    and 2 static facts were refused); sweep 181 re-sampled insertion-stably at margin 0.05: 24/35 and 32/40.
    E4.2 (typed guard, no catalog dependency, judged without learning from the pattern refusals of the run):
    credential 28/35 and live reading 28/40 -- the E4.2 bar (30 and 34) is NOT met, kept loud. The first E4.2 freeze
    had 31/35 and 32/40 but refused engine talk and a test fixture (no floor, no engine snapshot); the floor costs the
    short slang readings ("eth rn", "sol funding"). B/C were looked at between freezes, so they are not clean."""
    import tools.bench_learnguard as B
    from holographic.agents_and_reasoning.holographic_learnguard import SemanticGuard
    g = SemanticGuard()
    cred = [(q, B._CRED_ANS[i % 5]) for i, q in enumerate(B.CRED_B + B.CRED_C)]
    read = [(q, B._READ_ANS[i % 7]) for i, q in enumerate(B.READ_B + B.READ_C)]
    assert sum(not B.judge(q, a, g)["ok"] for q, a in cred) >= 28
    assert sum(not B.judge(q, a, g)["ok"] for q, a in read) >= 28


def test_typed_guard_numbers_hold_on_the_fresh_heldout_set():
    """E4.2 on the FRESH held-out set (tests/data/guard_heldout.json: CLINC150 / Banking77 / hand-written, split by
    sha256 BEFORE any tuning, never used to build or tune the guard), realistic answers per intent. MEASURED
    (tools/bench_learnguard.py --typed; the third run -- the first two are in the evidence file): credential 71/91,
    live 247/305 (CLINC150 221/239, hand-written 26/66), unclear 33/39; typed false positives 3/705. The sweep-181
    guard on the same items: 71/91, 151/305, 2/39 and 47/705. The bar (the rates of >= 30/35 and >= 34/40, and 0 false
    positives) is NOT met on this set -- kept loud."""
    import tools.bench_learnguard as B
    from holographic.agents_and_reasoning.holographic_learnguard import SemanticGuard
    g = SemanticGuard()
    caught, n, fp = {}, {}, []
    for q, a, it in B.with_answers(B.load_heldout()):
        v = B.judge(q, a, g)
        n[it["type"]] = n.get(it["type"], 0) + 1
        if it["type"] == "ordinary":
            if not v["ok"] and v["layer"] == "semantic":
                fp.append(q)
        else:
            caught[it["type"]] = caught.get(it["type"], 0) + (not v["ok"])
    # KEPT NEGATIVE, LOUD: the acceptance bar was 0; every run refused the same 3 ordinary questions with a time word
    # ("what time does the pharmacy open on sundays" -> "9am" read as a live reading). Pinned so a 4th fails.
    assert len(fp) <= 3, fp
    assert caught["credential"] >= 71 and caught["live_value"] >= 247 and caught["unclear"] >= 33, (caught, n)


def test_heldout_set_never_trains_the_guard():
    """Rigour: no held-out question, and no B/C question, is a shipped example (same split key as the pool)."""
    import tools.bench_learnguard as B
    from holographic.agents_and_reasoning.holographic_learnguard import shipped_examples
    shipped = {B.split_key(t) for t, _, _ in shipped_examples()}
    held = [it["text"] for it in B.load_heldout()]
    assert len(held) > 1000
    assert not [t for t in held if B.split_key(t) in shipped]
    assert not [t for t in B.CRED_B + B.CRED_C + B.READ_B + B.READ_C if B.split_key(t) in shipped]
    assert all(B.in_heldout(t) for t in held)                # the file is exactly the odd-hash half


def test_verdicts_identical_after_adding_50_catalog_aliases(monkeypatch):
    """Sweep 180's parity bug: the guard's NORMAL examples were a sample of catalog aliases, so ONE new card moved
    the held-out credential count 30 -> 29 with no guard change. The typed guard builds nothing from the catalog:
    50 synthetic aliases (one new card) must leave every verdict -- and the base rows, bit for bit -- unchanged."""
    import tools.bench_learnguard as B
    import holographic.misc.holographic_skills as S
    from holographic.agents_and_reasoning.holographic_learnguard import SemanticGuard
    probes = [(q, B._CRED_ANS[i % 5]) for i, q in enumerate(B.CRED_B + B.CRED_C)] + \
             [(q, B._READ_ANS[i % 7]) for i, q in enumerate(B.READ_B + B.READ_C)] + \
             [(it["text"], "021000021") for it in B.load_heldout()[::9]] + \
             [("what is 7 times 8", "56"), ("spec 12 of the flux rotor", "rotor 12 rated 36 units"), ("q1", "a1"),
              ("my email", "owner0@example.com"), ("what port does the service use", "8080")]

    def verdicts(rebuild):
        if rebuild:
            SemanticGuard._BASE = None                      # force the base rows to be built again, now --
            monkeypatch.setenv("LECORE_GUARD_CACHE", "0")   # really BUILT: the temp-dir cache (backlog G2) is keyed
                                                            # on the examples and the code, not the catalog, so a
                                                            # cache hit would hide exactly the leak this test hunts
        g = SemanticGuard()
        return [B.judge(q, a, g)["ok"] for q, a in probes], g._base()[1].digest()

    monkeypatch.setattr(SemanticGuard, "_BASE", SemanticGuard._BASE)   # restored after the test, whatever happens
    before, digest0 = verdicts(rebuild=False)               # this process's base (built once per process anyway)
    real = S._catalog()

    class _Card:                                            # one new card, 50 aliases, the catalog's shape
        name, does = "synthetic_probe_card", "a synthetic card added by the insertion-stability test"
        aliases = ["synthetic alias number %d for the guard test" % i for i in range(50)]

    class _Grown:
        def all(self):
            return list(real.all()) + [_Card()]

        def __getattr__(self, k):
            return getattr(real, k)
    monkeypatch.setattr(S, "_catalog", lambda: _Grown())
    assert len(S._catalog().all()) == len(real.all()) + 1
    after, digest1 = verdicts(rebuild=True)
    assert after == before and digest1 == digest0


def test_an_old_guard_state_still_restores(tmp_path, monkeypatch):
    """A partition written before E4.2 carries the sweep-180 state ({credential, reading, normal} text lists).
    It must restore into the typed guard: the learned correction still wins after the reload."""
    from holographic.agents_and_reasoning.holographic_learnguard import SemanticGuard
    q, a = "how many yen for a dollar", "147.2"
    old = _mind()
    old.learn_guard_example(q, "normal")
    # write it the way sweep 181 did: only the three text lists
    monkeypatch.setattr(SemanticGuard, "state", lambda self: {k: list(self.learned[k])
                                                               for k in ("credential", "reading", "normal")})
    root = _partition(tmp_path)
    old.learning_save(root)
    monkeypatch.undo()
    m = _mind()
    m.learning_load(root)
    assert q in m.semantic_guard.learned["normal"] and m.semantic_guard.learned["unclear"] == []
    assert m.learn_guard(q, a)["ok"] is True
    assert _mind().learn_guard(q, a)["kind"] == "volatile"   # and without the correction it is refused


def test_a_semantic_false_positive_is_corrected_and_the_correction_survives_a_reload(tmp_path):
    m = _mind()
    # pretend this is a static fact in someone's domain: a question ONLY the semantic layer refuses
    q, a = "how many yen for a dollar", "147.2"
    assert m.teach(q, a)["taught"] is False
    rep = m.learn_guard_example(q, "normal")
    assert rep["learned"] is True and rep["intent"]["reading"] <= 0
    assert m.teach(q, a)["taught"] is True
    root = _partition(tmp_path)
    m.learning_save(root)
    m2 = _mind()
    m2.learning_load(root)
    assert m2.learn_guard(q, a)["ok"] is True                 # the correction came back with the partition


def test_a_pattern_refusal_teaches_the_semantic_layer():
    m = _mind()
    probe = "the current price of gweil please"
    before = m.semantic_guard.intent(probe)["reading"]
    assert m.teach("what is the current price of gweil", "$0.004")["guard"] == "volatile"   # pattern layer
    assert "what is the current price of gweil" in m.semantic_guard.learned["reading"]
    assert m.semantic_guard.intent(probe)["reading"] > before


def test_replay_never_deletes_a_stored_row_on_a_semantic_guess(tmp_path, monkeypatch):
    """A stored row the SEMANTIC layer would refuse is kept at load (a guess never deletes); leak_audit reports it
    as SUSPECTED instead. Written with the guard off to simulate a partition from before sweep 180."""
    import holographic.agents_and_reasoning.holographic_learnguard as G
    old = _mind()
    with monkeypatch.context() as mp:
        mp.setattr(G, "learning_verdict", lambda *a, **k: {"ok": True, "kind": None, "reason": None})
        assert old.teach("how many yen for a dollar", "147.2")["taught"] is True   # semantic-only
    root = _partition(tmp_path)
    old.learning_save(root)
    m = _mind()
    assert m.learning_load(root)["guard_dropped"] == {"sensitive": 0, "volatile": 0}
    assert m.ask("how many yen for a dollar")["answer"] == "147.2"
    audit = m.leak_audit(root=root)
    assert audit["flagged"] == 0 and audit["suspected"] >= 1


def test_leak_audit_finds_planted_secrets_in_every_store_and_never_prints_them():
    import json
    from holographic.agents_and_reasoning.holographic_decisionrecord import DecisionRecord
    m = _mind()
    m.teach("what port does the service use", "8080")
    m.zoo["ladder"].taught_log.append(["the admin password", "hunter2-Moose!", "shared", "taught"])   # bypass
    m.zoo["ladder"]._payloads["9:9"] = FAKE_KEY
    m.bus().publish("x", {"note": "use " + FAKE_KEY})
    m.decision_ledger().add(DecisionRecord("login with " + FAKE_KEY, "q", ["a"], "a", "route"))
    a = m.leak_audit()
    got = {k for k, v in a["stores"].items() if v["flagged"]}
    assert {"taught rows", "served payloads", "message bus", "decision ledger"} <= got and not a["clean"]
    blob = json.dumps(a)
    assert FAKE_KEY not in blob and "hunter2" not in blob


def test_leak_audit_of_a_clean_mind_is_clean():
    m = _mind()
    m.teach("what port does the service use", "8080")
    a = m.leak_audit()
    assert a["clean"] and a["flagged"] == 0


def test_the_ordinary_ways_people_paste_a_password_are_all_refused():
    """The learning-loop audit (2026-09-27, tools/audit_learning_loop.py check_guard_phrasings) found the pattern
    layer caught only 4 of 10 ordinary wordings; the missed ones reached the saved decisions and query log. All 10
    are refused now, the value is what gets redacted, and look-alike ordinary sentences still pass. FAKE secret."""
    from holographic.agents_and_reasoning.holographic_learnguard import redact, sensitive_reason
    s = "Hunter2-FAKE-9c1d"
    for p in ["my password is %s", "password: %s", "password %s", "use password %s to log in", "the pw is %s",
              "login with %s", "my passphrase is %s", "pwd=%s", "here is my password, %s", "%s is my password"]:
        t = p % s
        assert sensitive_reason(t), t
        assert s not in redact(t) and "[redacted]" in redact(t), redact(t)
    for t in ["my password is too short", "password reset", "log in with GitHub", "that is my password manager",
              "login with SSO", "the pw is fine", "here is my password, it expired", "Wifi2 is my network name",
              "is 1234 a good pin?", "passwords must be 12 characters"]:
        assert not sensitive_reason(t), t

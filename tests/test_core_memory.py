"""tests/test_core_memory.py -- what a mind learns travels into the core memory; secrets travel only as ${ENV} names.

Owner direction (2026-09-27): the seed / core memory exists to cut model calls and pick tools better; a phrasing a
person used that memory did not know, a method, and how to use an API (keys as environment variables) must survive
into it. These pins cover the three halves: SECRETS -> PLACEHOLDERS at learn time (and migration of old state on
load), the EXTENDED DISTILLATION (tools/distill_release.py: wordings, methods, api specs, reflexes, the promotion
rule, the audited report and --check), and a FRESH mind using what was carried. Every secret here is FAKE; nothing
touches the network (an unset variable fails BEFORE any request, which is the contract under test).
"""
import json
import os
import sys

import pytest

os.environ.setdefault("PYTHONHASHSEED", "0")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE = "Hunter2-FAKE-9c1d-apikey"


def _mind():
    import lecore
    return lecore.UnifiedMind(dim=256, seed=0)


def _dr():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import distill_release
    return distill_release


def _rid(m, text):
    return next(r for r, row in m.meaning.rows.items() if row["canonical"] == text)


def _bytes_of(folder):
    """Every byte a bundle holds, with its compressed container read back as text too."""
    from holographic.io_and_interop.holographic_container import load_container
    raw = b""
    for base, _, files in os.walk(folder):
        for fn in files:
            with open(os.path.join(base, fn), "rb") as f:
                raw += f.read()
    with open(os.path.join(folder, "learning", "state.lecore"), "rb") as f:
        txt = json.dumps([s.get("meta") for s in load_container(f.read())["sections"]], default=str)
    return raw, txt


# ------------------------------------------------------------------------------------------ placeholders
def test_placeholder_names_are_deterministic_from_service_and_parameter():
    from holographic.io_and_interop.holographic_apilearn import placeholder_name
    assert placeholder_name("openweather", "appid") == "OPENWEATHER_APPID"
    assert placeholder_name("toyair", "X-Api-Key") == "TOYAIR_API_KEY"
    assert placeholder_name("github", "Authorization") == "GITHUB_TOKEN"
    assert placeholder_name("stripe", "stripe_secret") == "STRIPE_SECRET"          # no double prefix


def test_credentials_become_placeholders_and_ordinary_params_do_not():
    from holographic.io_and_interop.holographic_apilearn import (placeholderize, placeholderize_text,
                                                                  placeholderize_url)
    got = placeholderize({"symbol": "SOL", "api_key": FAKE, "primary_key": "id", "random_seed": "42",
                          "token": "sk-live-4f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c", "count": 5}, "px")
    assert got == {"symbol": "SOL", "api_key": "${PX_API_KEY}", "primary_key": "id", "random_seed": "42",
                   "token": "${PX_TOKEN}", "count": 5}
    assert placeholderize_url("https://api.x.io/v1?q=lisbon&appid=Hunter2FAKE9c1d77", "x") == \
        "https://api.x.io/v1?q=lisbon&appid=${X_APPID}"
    txt = placeholderize_text('curl -H "Authorization: Bearer ghp_abcdefghijklmnopqrstuvwxyz0123456789AB" '
                              'https://api.github.com/user')
    assert "${GITHUB_TOKEN}" in txt and "ghp_" not in txt
    # a seed phrase is NOT an API credential: it is left for the guard to REFUSE (such a row never ships)
    seed = "abandon ability able about above absent absorb abstract absurd abuse access accident"
    assert placeholderize_text(seed) == seed


def test_an_unset_variable_fails_loudly_naming_it():
    from holographic.io_and_interop.holographic_apilearn import MissingCredential, resolve_placeholders
    assert resolve_placeholders({"h": "Bearer ${GH_TOKEN}"}, env={"GH_TOKEN": FAKE}) == {"h": "Bearer " + FAKE}
    with pytest.raises(MissingCredential) as e:
        resolve_placeholders({"a": "${A_KEY}", "b": "${B_KEY}"}, env={})
    assert e.value.names == ["A_KEY", "B_KEY"] and "A_KEY" in str(e.value)


SPEC = {"info": {"title": "toyair"}, "servers": [{"url": "https://api.toyair.example"}],
        "components": {"securitySchemes": {"k": {"type": "apiKey", "in": "header", "name": "X-Api-Key"}}},
        "paths": {"/aqi/{zone}": {"get": {"operationId": "air_quality",
                                          "summary": "get the air quality index for a zone",
                                          "parameters": [{"name": "zone", "in": "path", "required": True}]}}}}


def test_api_learn_keeps_how_to_authenticate_and_never_calls_without_the_key():
    m = _mind()
    r = m.api_learn(SPEC)
    assert r["env"] == ["TOYAIR_API_KEY"]
    assert m.api_toolbox().services["toyair"]["auth"]["headers"] == {"X-Api-Key": "${TOYAIR_API_KEY}"}
    out = m.api_toolbox().call("toyair", "air_quality", params={"zone": 3}, env={})
    assert out["ok"] is False and out["missing_env"] == ["TOYAIR_API_KEY"] and "not called" in out["error"]
    # a raw key handed to api_learn's auth= becomes a placeholder before anything is stored
    r2 = m.api_learn(dict(SPEC, info={"title": "other"}, components={}), auth={"query": {"appid": FAKE}})
    assert r2["env"] == ["OTHER_APPID"]
    assert FAKE not in json.dumps([list(t) for t in m.zoo["ladder"].taught_log])


def test_tool_reflex_params_are_learned_as_placeholders():
    m = _mind()
    m.api_learn(SPEC)
    r = m.tool_reflex_teach("air quality index in zone 4", "toyair", "air_quality",
                            params={"appid": FAKE}, extract_numbers=["zone"])
    assert r["env"] == ["TOYAIR_APPID"]
    row = [t for t in m.zoo["ladder"].taught_log if str(t[0]).startswith("toolreflex: ")][-1]
    assert FAKE not in str(row[1]) and "${TOYAIR_APPID}" in str(row[1])
    # serve: the reflex matches, the variables are unset -> a loud escalation naming them, no request made
    os.environ.pop("TOYAIR_API_KEY", None)
    os.environ.pop("TOYAIR_APPID", None)
    out = m.serve("what is the air quality index in zone 12")
    assert out["served"] is False and "TOYAIR" in str(out.get("reason"))


def test_raw_keys_in_old_learned_state_are_migrated_on_load():
    """State written before the placeholder rule: a spec record, a tool reflex row and a method row, each holding a
    raw FAKE key, come back as placeholders -- and the raw text is gone from what the next save writes."""
    m = _mind()
    m.teach("warm up", "a plain answer so the ladder exists")
    lad = m.zoo["ladder"]
    lad.taught_log.append(["api spec record: legacy",
                           json.dumps({"base": "https://api.legacy.example/v1?apikey=Hunter2FAKE9c1d77",
                                       "endpoints": {}}), "shared", "taught"])
    lad.taught_log.append(["toolreflex: legacy lookup for thing", json.dumps(
        {"service": "legacy", "endpoint": "x", "params": {"api_key": FAKE}, "extract_numbers": []}),
        "shared", "toolreflex"])
    box = m.api_toolbox()
    box._rehydrate()
    assert box.migrated == 1 and "${LEGACY_APIKEY}" in box.services["legacy"]["base"]
    m._tool_reflexes = []
    m.serve("legacy lookup for thing please")                    # rebuilds the reflexes from the durable rows
    assert "Hunter2FAKE9c1d77" not in json.dumps([list(t) for t in lad.taught_log])
    assert FAKE not in json.dumps([list(t) for t in lad.taught_log])
    from holographic.agents_and_reasoning.holographic_meaning import MeaningIndex
    st = {"v": 1, "rows": [{"rid": "x", "kind": "method", "canonical": "quote of solana", "phrasings":
                            ["quote of solana"], "provenance": "taught",
                            "method": {"verb": "quote", "live": True, "args": {"api_key": FAKE, "symbol": {"slot":
                                                                                                         "symbol"}},
                                       "slots": {"symbol": {"type": "choice", "values": {"solana": "SOL"}}}}}]}
    mi = MeaningIndex.from_state(st)
    row = next(iter(mi.rows.values()))
    assert mi.migrated == 1 and row["method"]["args"]["api_key"] == "${QUOTE_API_KEY}"


def test_a_method_with_a_credential_constant_reads_the_environment_at_call_time(monkeypatch):
    m = _mind()
    seen = []
    m.meaning_tool_register("quote", lambda **kw: seen.append(kw) or "ok")
    m.meaning_resolve("what's the quote for solana", {"verdict": "new", "answer": "...", "method": {
        "verb": "quote", "args": {"symbol": "SOL", "api_key": FAKE}, "from_question": {"symbol": "solana"},
        "live": True}})
    row = next(r for r in m.meaning.rows.values() if r["kind"] == "method")
    assert row["method"]["args"]["api_key"] == "${QUOTE_API_KEY}"
    monkeypatch.delenv("QUOTE_API_KEY", raising=False)
    rid = next(k for k, r in m.meaning.rows.items() if r["kind"] == "method")
    out = m._meaning_run(rid, {"symbol": "SOL", "api_key": row["method"]["args"]["api_key"]})
    assert out["via"] == "method-failed" and out["missing_env"] == ["QUOTE_API_KEY"] and not seen
    monkeypatch.setenv("QUOTE_API_KEY", FAKE)
    out = m._meaning_run(rid, {"symbol": "SOL", "api_key": row["method"]["args"]["api_key"]})
    assert out["via"] == "method:quote" and seen == [{"symbol": "SOL", "api_key": FAKE}]


# ------------------------------------------------------------------------------------------ the distillation
ENGINE_Q = "how does the lecore reflex answer repeated questions"
ENGINE_A = "the reflex answers near-exact repeats at T0 and a lecore mind escalates what it cannot answer honestly"
MODEL_Q = "how does lecore keep an api key it learned"
MODEL_A = ("lecore keeps the name of an environment variable as a placeholder with its provenance and reads the key "
           "only at call time")


def _learner():
    """A mind that learned the way a served mind does: a taught engine row reworded by a person (a `same` verdict),
    a row the MODEL created (confirmed later, and one that never was), two methods (one confirmed by a second
    verdict, one not), a public API with a key, a private-host API, a tool reflex."""
    m = _mind()
    m.teach(ENGINE_Q, ENGINE_A)
    m.meaning_resolve("how does lecore handle a question i ask twice", {"verdict": "same", "row": _rid(m, ENGINE_Q)})
    m.meaning_resolve("[s:alice] lecore reflex and my repeated questions", {"verdict": "same",
                                                                            "row": _rid(m, ENGINE_Q)})
    m.meaning_resolve(MODEL_Q, {"verdict": "new", "answer": MODEL_A})
    m.meaning_resolve("where does lecore put api keys it has learned", {"verdict": "same", "row": _rid(m, MODEL_Q)})
    m.meaning_resolve("does lecore ever learn a reflex from the partition alone",
                      {"verdict": "new", "answer": "only the lecore reflex decides that, unconfirmed by anything"})
    m.meaning_resolve("what's the price of solana right now", {"verdict": "new", "answer": "$1", "method": {
        "verb": "price", "args": {"symbol": "SOL"}, "from_question": {"symbol": "solana"}, "live": True}})
    rid = next(k for k, r in m.meaning.rows.items() if r["kind"] == "method")
    m.meaning_resolve("how much is eth trading at", {"verdict": "same", "row": rid, "args": {"symbol": "ETH"},
                                                     "from_question": {"symbol": "eth"}})
    m.meaning_resolve("will it rain in lisbon", {"verdict": "new", "answer": "rain", "method": {
        "verb": "weather", "args": {"location": "Lisbon"}, "from_question": {"location": "lisbon"}, "live": True}})
    m.api_learn(SPEC)
    m.api_learn(dict(SPEC, info={"title": "homelab"}, servers=[{"url": "http://192.168.1.20:8080"}]))
    m.api_toolbox().services  # rehydrated
    m.tool_reflex_teach("what is the air quality index in zone 4", "toyair", "air_quality",
                        params={"appid": FAKE}, extract_numbers=["zone"])
    return m


def test_the_extended_distillation_carries_what_saves_model_calls_and_reports_every_exclusion(tmp_path):
    d = _dr()
    out = str(tmp_path / "bundle")
    rep = d.distill(None, out, src=_learner())
    arts = rep["artifacts"]
    # the person's rewording travels with its row; the session-salted one does not
    assert ["how does the lecore reflex answer repeated questions",
            "how does lecore handle a question i ask twice"] in arts["wordings"]["items"]
    assert arts["wordings"]["excluded"].get("session-specific") == 1
    # THE PROMOTION RULE: the model's row confirmed by a second verdict ships as 'confirmed'; the unconfirmed one not
    assert MODEL_Q in rep["questions"] and MODEL_Q in rep["promoted"]
    assert "does lecore ever learn a reflex from the partition alone" not in rep["questions"]
    assert rep["excluded"].get("provisional-provenance", 0) >= 1
    # methods: the confirmed one travels, the single-verdict one is excluded by name
    assert [x[2] for x in arts["methods"]["items"]] == ["price"]
    assert arts["methods"]["excluded"] == {"unconfirmed (one model verdict)": 1}
    # the public api travels with its auth as a placeholder; the private-host one stays home
    assert [a["service"] for a in arts["apis"]["items"]] == ["toyair"]
    assert arts["apis"]["excluded"] == {"private host (a user's own machine or network)": 1}
    assert arts["reflexes"]["carried"] == 1 and set(rep["env"]) == {"TOYAIR_API_KEY", "TOYAIR_APPID"}
    assert "reflex bridge experience" in rep["not_distilled"]
    raw, txt = _bytes_of(out)
    assert FAKE.encode() not in raw and FAKE not in txt, "a raw key reached the bundle"
    assert d.check(out)["ok"], d.check(out)
    # rows-only (the old rule) carries none of the above
    rep0 = d.distill(None, str(tmp_path / "rows"), src=_learner(), rows_only=True)
    assert "artifacts" not in rep0 and rep0["questions"] == rep["questions"]


def test_a_fresh_mind_booted_on_the_core_memory_uses_what_was_carried(tmp_path, monkeypatch):
    import lecore
    d = _dr()
    out = str(tmp_path / "bundle")
    d.distill(None, out, src=_learner())
    f = lecore.UnifiedMind()
    f.zoo_attach(lambda p: "")
    f.learning_load(out)
    got = f.ask("how does lecore handle a question i ask twice")          # the carried rewording: no model call
    assert got["tier"] == "T0" and got["answer"] == ENGINE_A
    m_out = f.ask("how much is eth trading at")                              # the carried method: a call plan
    assert m_out.get("call") == {"verb": "price", "args": {"symbol": "ETH"}}, m_out
    monkeypatch.delenv("TOYAIR_API_KEY", raising=False)
    s = f.serve("air quality index for zone 12 please")                     # the carried reflex + api spec
    assert s["served"] is False and "TOYAIR" in s["reason"], s              # env unset: loud, never unauthenticated


def test_check_fails_a_bundle_that_holds_a_raw_key(tmp_path):
    d = _dr()
    out = str(tmp_path / "bundle")
    d.distill(None, out, src=_learner())
    import lecore
    m = lecore.UnifiedMind()
    m.zoo_attach(lambda p: "")
    m.learning_load(out)
    # smuggle a raw-key spec record in past the distiller (as a hand-copied bundle would). `apikey=...` would be
    # dropped by learning_save itself (the guard never writes such a row); `appid=...` is a key the GUARD does not
    # recognise (no credential word, no strong shape) -- the placeholder rule does, and --check must say so
    m.zoo["ladder"].taught_log.append(["api spec record: smuggled", json.dumps(
        {"base": "https://api.smuggled.example?appid=q8Zr4Lw2Pn77", "endpoints": {}}), "shared", "taught"])
    m.learning_save(out, path=os.path.join(out, "learning", "state.lecore"))
    rep = d.check(out)
    assert not rep["ok"] and rep["failing"].get("api: a raw credential (not placeholders)") == 1, rep

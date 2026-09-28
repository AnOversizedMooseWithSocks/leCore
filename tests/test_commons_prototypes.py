"""tests/test_commons_prototypes.py -- the owner's answer to the CLM backlog's open question 3: "Should learned
prototypes travel through contribute / commons_pool? -> YES", with the learning guard still in front of it, plus the
three gaps the map found in contribute() (vetoed rows exported, provisional model provenance exported, a row the
guard refused in the bundle counted as kept).

Test data uses FAKE secrets only."""
import os

import numpy as np
import pytest

os.environ.setdefault("PYTHONHASHSEED", "0")


def _mind():
    import lecore
    return lecore.UnifiedMind(dim=256, seed=0)


def _kept_questions(dest):
    """The questions a bundle actually carries (a fresh mind loads it)."""
    m = _mind()
    m.learning_load(str(dest))
    return {str(r[0]) for r in getattr(m.zoo["ladder"], "taught_log", [])}


# ------------------------------------------------------------------ the three contribute() gaps
def test_contribute_never_exports_a_vetoed_row(tmp_path):
    m = _mind()
    m.teach("what colour is the sky on a clear day", "blue")
    m.teach("how many legs does a spider have", "six")
    m.answer_feedback("how many legs does a spider have", ok=False)       # this memory refused its own answer
    rep = m.contribute(str(tmp_path / "b"))
    assert any("vetoed" in why for q, why in rep["rejected"] if q.startswith("how many legs"))
    kept = _kept_questions(tmp_path / "b")
    assert "what colour is the sky on a clear day" in kept and "how many legs does a spider have" not in kept


def test_contribute_rejects_provisional_model_provenance(tmp_path):
    """'model:<by>' is what meaning_resolve writes for a model's reply; only 'model-cached' used to be refused."""
    m = _mind()
    m.teach("what colour is the sky on a clear day", "blue")
    m.teach("what is the tallest mountain on earth", "Everest")
    lg = m.zoo["ladder"].taught_log
    lg[-1] = [lg[-1][0], lg[-1][1], lg[-1][2], "model:some-llm"]          # as meaning_resolve stamps it
    rep = m.contribute(str(tmp_path / "b"))
    assert any("model:some-llm" in why for _, why in rep["rejected"])
    assert rep["kept"] == 1 and "what is the tallest mountain on earth" not in _kept_questions(tmp_path / "b")


def test_a_row_the_guard_refuses_in_the_bundle_is_counted_as_rejected_not_kept(tmp_path):
    """A row that reached the taught log before today's guard existed (an old partition) passes contribute's lexical
    screen, and the bundle's own teach() refuses it. It must be REPORTED as rejected, never counted as kept (it used
    to be: `kept` was the screened list, not what the bundle actually took)."""
    m = _mind()
    m.teach("what colour is the sky on a clear day", "blue")
    m.zoo["ladder"].taught_log.append(["what did alice write on the sticky note", "sk-Hunter2FAKE9c1dabcdefabcdefab",
                                       "shared", "taught"])       # an sk- key shape short enough to pass the lexical screen
    rep = m.contribute(str(tmp_path / "b"))
    assert rep["kept"] == 1, rep
    assert any("refused by the learning guard in the bundle" in why for _, why in rep["rejected"])
    assert _kept_questions(tmp_path / "b") == {"what colour is the sky on a clear day"}


# ------------------------------------------------------------------ prototypes travel, behind the guard
def _learned_mind():
    """A mind whose router and tool door have learned from judged verdicts."""
    m = _mind()
    r = m.route_tiered("denoise an image")
    m.decision_outcome(r["id"], "Denoise (domain)")                       # the router learns one route
    m.api_use = lambda service, endpoint, params=None, headers=None: {"ok": True, "data": {}}
    m.tool_reflex_teach("convert 77 fahrenheit into celsius units", "convertd", "f_to_c")
    m.tool_reflex_teach("convert 25 celsius into fahrenheit units", "convertd", "c_to_f")
    s = m.serve("convert celsius fahrenheit units please")
    m.decision_outcome(s["id"], "convertd.c_to_f")                        # the tool door learns one correction
    m.teach("what colour is the sky on a clear day", "blue")
    return m


def test_learned_prototypes_travel_through_contribute_and_commons_pool(tmp_path):
    a = _learned_mind()
    rep = a.contribute(str(tmp_path / "a"), author="alice")
    assert set(rep["stores"]) == {"route", "tool"}, rep
    assert rep["stores"]["route"]["verdicts"] == 1 and rep["stores"]["tool"]["verdicts"] >= 1
    pool = a.commons_pool([str(tmp_path / "a")], str(tmp_path / "commons"))
    assert set(pool["stores"]) == {"route", "tool"}
    # a mind that never learned either door draws from the commons: the prototypes arrive and USE them
    b = _mind()
    got = b.memory_import(str(tmp_path / "commons"))
    by_door = {s_["door"]: s_ for s_ in got["stores"]}
    assert by_door["route"]["added"] >= 1 and by_door["tool"]["added"] == 2
    assert "Denoise (domain)" in b.router_report()["top"][0]
    assert b.tool_predict(b.tool_key("please convert these celsius temperatures to fahrenheit units"), k=1)[0][0] == \
        "convertd.c_to_f"


def test_a_store_that_learned_a_refused_or_salted_question_never_travels(tmp_path):
    m = _learned_mind()
    for door, q in (("intent-a", "my password is Hunter2-FAKE-9c1d, what do I do"),
                    ("intent-b", "[s:alice] which plan did I pick")):
        st = m.protostore(door, dim=64)
        st.update(np.ones(64), "x")
        m.protostore_share(door, True)                                    # the owner opted the door in ...
        m._protostore_track(door, q)                                      # ... but it learned from this question
    st = m.protostore("intent-c", dim=64)
    st.update(np.ones(64), "y")                                           # never opted in
    assert m.protostore_share("intent-a")["eligible"] is False and m.protostore_share("intent-b")["eligible"] is False
    rep = m.contribute(str(tmp_path / "b"))
    assert set(rep["stores"]) == {"route", "tool"}
    why = {w["door"]: w["why"] for w in rep["stores_withheld"]}
    assert "refused by the learning guard" in why["intent-a"]
    assert "session-salted" in why["intent-b"]
    assert why["intent-c"] == "not marked shareable"


def test_the_router_and_tool_door_never_learn_a_secret_at_all():
    m = _mind()
    r = m.route_tiered("denoise an image with my api key sk-Hunter2FAKE9c1dabcdefabcdefabcdef")
    rep = m.decision_outcome(r["id"], "Denoise (domain)")
    fw = rep.get("forwarded")
    fws = fw if isinstance(fw, list) else [fw]
    assert any(isinstance(f, dict) and f.get("router") is False for f in fws)
    assert m.router_report()["rows"] == 0 and m.protostore_share("route")["guard_ok"] is True


def test_merge_rule_weights_by_verdicts_and_flags_a_conflict_without_overwriting():
    """_protostore_merge: a label only they have is added; a shared label is blended by verdict count; a shared label
    whose prototypes point apart is a CONFLICT -- flagged, the local row untouched."""
    from holographic.agents_and_reasoning.holographic_protostore import ProtoStore
    m = _mind()
    mine = m.protostore("demo", dim=8)
    e = np.eye(8)
    mine.update(e[0], "a")
    mine.update(e[1], "b")
    theirs = ProtoStore(8, name="demo")
    theirs.update(e[0] + 0.2 * e[2], "a")                                 # agrees with ours (cos ~0.98)
    theirs.update(e[0] + 0.2 * e[2], "a")
    theirs.update(e[5], "b")                                               # points elsewhere: a conflict
    theirs.update(e[6], "c")                                               # new to us
    b_before = mine.A[mine.index("b")].copy()
    rep = m._protostore_merge("demo", theirs, source="test")
    st = m.protostore("demo")
    assert rep["added"] == 1 and rep["merged"] == 1 and [c["label"] for c in rep["conflicts"]] == ["b"]
    assert np.allclose(st.A[st.index("b")], b_before)                     # never silently overwritten
    assert st.count[st.index("a")] == 1 + 2                               # verdict counts add
    pa = st.P[st.index("a")]
    assert pa[2] > 0 and pa[0] > pa[2]                                    # blended toward theirs, 2:3 by verdicts
    assert "c" in st and st.count[st.index("c")] == 1

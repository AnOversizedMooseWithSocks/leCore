"""Tests for holographic_skills (the agent-friendly discovery / suggest / route / autocomplete layer)."""
import holographic_skills as sk


def test_mind_methods_introspected():
    ms = sk.mind_methods()
    assert len(ms) > 100
    # a known method carries a real signature + summary
    assert "material_info" in ms and ms["material_info"]["signature"].startswith("(")


def test_complete_autocomplete():
    comp = sk.complete("learn_")
    assert comp and all(c["name"].startswith("learn_") for c in comp)
    assert all(c["signature"].startswith("(") for c in comp)


def test_skill_card_capability_and_method():
    cap = sk.skill_card("Index (search)")
    assert cap["kind"] == "capability" and cap["example"]
    m = sk.skill_card("material_info")
    assert m["kind"] == "method" and m["call"].startswith("mind.material_info(")
    assert sk.skill_card("no_such_skill_xyz") is None


def test_suggest_ranks_with_confidence():
    sug = sk.suggest("draw a picture")
    assert sug and "2d image" in sug[0]["name"].lower()
    assert 0.0 <= sug[0]["confidence"] <= 1.0 and sug[0]["call"]


def test_route_acts_when_confident():
    r = sk.route("start pause resume cancel a job")
    assert r["decision"] == "act" and "call" in r["skill"]


def test_route_chooses_when_ambiguous():
    r = sk.route("distributed coordinator farm")           # 3 distinct distributed homes -> ambiguous
    assert r["decision"] == "choose" and len(r["options"]) >= 2


def test_route_unknown_gives_hint():
    r = sk.route("qwzxvbn zzzqwx floobnarg")
    assert r["decision"] == "unknown" and "prompt" in r


def test_manifest_is_machine_readable():
    man = sk.manifest()
    assert man["counts"]["capabilities"] > 50 and man["counts"]["methods"] > 100
    assert all("call" in m for m in man["methods"][:20])


def test_module_duplicates_do_not_dilute_confidence():
    # a capability with a curated home + an auto-module twin should still route confidently
    r = sk.route("render a scene with global illumination")
    assert r["decision"] == "act"

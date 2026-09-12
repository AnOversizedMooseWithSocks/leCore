"""Tests for holographic_skills (the agent-friendly discovery / suggest / route / autocomplete layer)."""
import holographic.misc.holographic_skills as sk


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
    # The catalog now has a SECOND job-flavoured home ("Background cloud bake", which also advertises
    # pause/resume/cancel), so the bare phrase is genuinely ambiguous and the router honestly asks instead of
    # guessing. Use a query that names the capability distinctly, which is what "confident" means here.
    r = sk.route("checkpointable job lifecycle start pause resume cancel")
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
    """THE CONTRACT, tested directly (sweep 169): an auto-registered module entry is a pointer to a
    capability, not a competing skill, so it must not dilute the curated card's confidence -- and a
    runner-up from the SAME module is a facet, not a competitor.

    KEPT NEGATIVE: the previous version asserted that one English sentence ("render a scene with
    global illumination") routed `act`. That pinned a calibration artifact -- a 0.6 threshold against
    whatever the catalog happened to contain -- not this contract. It broke the moment an honest
    competitor card landed (sweep 154's bake-and-relight), and the first repair swapped the sentence
    for one that scored 0.611, which is chasing green. Testing the rule with a controlled catalog is
    independent of catalog density, which is the thing that actually moves."""
    from holographic.caching_and_storage.holographic_catalog import Catalog

    cat = Catalog()
    cat.register_capability("Widget frobnication (curated)", "frobnicate a widget with the frob kernel",
                            example="m.frobnicate(w)", aliases=("frobnicate a widget",), module="frob")
    cat.register_capability("holographic_frob", "auto module entry: frobnicate widget frob kernel",
                            example="from holographic_frob import frobnicate", module="frob")
    cat.register_capability("holographic_frobtwin", "another auto twin: frobnicate widget frob",
                            example="import holographic_frobtwin", module="frob")
    with_twins = cat.find_scored("frobnicate a widget", k=5)
    assert with_twins[0][0].name.startswith("Widget"), with_twins
    assert len(with_twins) >= 3, "the twins did not even match -- the test is not exercising the rule"

    # 1. _rank prefers curated homes: the twins vanish from the ranking when a curated card matches
    import holographic.misc.holographic_skills as sk
    real = sk._catalog
    sk._catalog = lambda: cat
    try:
        ranked = sk._rank("frobnicate a widget", k=5)
        assert [c.name for c, _ in ranked] == ["Widget frobnication (curated)"], ranked
        # 2. and a same-module runner-up never dilutes confidence: with or without the twins, equal
        alone = Catalog()
        alone.register_capability("Widget frobnication (curated)", "frobnicate a widget with the frob kernel",
                                  example="m.frobnicate(w)", aliases=("frobnicate a widget",), module="frob")
        conf_with = sk._confidence(with_twins)
        conf_alone = sk._confidence(alone.find_scored("frobnicate a widget", k=5))
        assert conf_with == conf_alone, (conf_with, conf_alone)
        assert sk.route("frobnicate a widget")["decision"] == "act"
        # 3. a DIFFERENT-module competitor of equal strength IS a real choice -- the decision says so
        cat.register_capability("Gadget frobnication (curated)", "frobnicate a widget the gadget way",
                                example="m.frobnicate_gadget(w)", aliases=("frobnicate a widget",), module="gadget")
        r = sk.route("frobnicate a widget")
        assert r["decision"] == "choose" and len(r["options"]) >= 2, r
    finally:
        sk._catalog = real

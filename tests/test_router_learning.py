"""tests/test_router_learning.py -- the CLM backlog's router items (F1): E3.4 (cache what is slow), E4.1 (the router
learns on the shared contrastive rule), E4.5 (serve asks the router).

Every pin is a contract, not a level: the measured levels live in tools/bench_router.py ->
docs/research/evidence/bench_router.json (they need the full 3,982-card catalog and minutes of CPU)."""
import hashlib
import json
import os

import pytest

os.environ.setdefault("PYTHONHASHSEED", "0")


# ============================================================================== E3.4 -- cache what is actually slow
def _route_tiered_pre_e34(cat, problem, k=5, z_answer=-0.1, z_refuse=-0.5, n_null=64, seed=0, clarify=False):
    """THE ORACLE: Catalog.route_tiered exactly as it was before E3.4 (a second find_scored pass, and families()
    recomputed card by card on every call), kept verbatim apart from the families line, which recomputes the map
    from Capability.family() directly so no memo can be involved. The shipped method must match it BIT FOR BIT."""
    import hashlib as _hl
    from holographic.agents_and_reasoning.holographic_decisionrecord import DecisionDict
    base = cat.route_or_abstain(problem, k=max(k, 8), n_null=n_null, z_min=-1e9, seed=seed)
    z = float(base["z"])
    ranked = cat.find_scored(problem, k=max(k, 8))
    fams = {}
    for c in cat.all():                                  # the old families() loop, verbatim
        f = c.family()
        fams[c.name] = (f, "module" if f else None)
    options = [{"name": c.name, "score": float(s), "family": fams.get(c.name, (None, None))[0],
                "method": c.method} for c, s in ranked[:k]]
    top = options[0]["score"] if options else 0.0
    ties = sum(1 for o in options if o["score"] == top) if options else 0
    top_fams = {o["family"] for o in options[:3] if o["family"]}
    if not options or z < z_refuse:
        tier, answer = "refuse", None
    elif z >= z_answer and ties == 1:
        tier, answer = "answer", options[0]["name"]
    elif clarify and len(top_fams) >= 2:
        tier, answer = "clarify", None
    else:
        tier, answer = "menu", None
    canon = "%s\x00%s\x00%s" % (problem.strip().lower(), tier, ",".join(o["name"] for o in options))
    rec = {"tier": tier, "answer": answer, "z": z, "score": float(top), "ties": ties,
           "families": sorted(top_fams), "options": options, "p": base.get("p"),
           "p_null": base.get("p"), "p_correct": None,
           "id": _hl.sha256(canon.encode("utf-8")).hexdigest()[:16],
           "question": ("which of these families did you mean: %s?" % ", ".join(sorted(top_fams)))
                       if tier == "clarify" else None,
           "reason": {"answer": "z=%.2f clears %.2f and the top score stands alone" % (z, z_answer),
                      "menu": "z=%.2f is above the refuse floor but not a clear answer; %d option(s), %d exact tie(s)" % (z, len(options), ties),
                      "clarify": "top candidates span %d families" % len(top_fams),
                      "refuse": "z=%.2f is in the gibberish band (< %.2f)" % (z, z_refuse)}[tier]}
    return DecisionDict(rec, _door="catalog")


def _canon(r):
    """A record as canonical JSON (dict(r) never touches the deprecated key's warning)."""
    return json.dumps(dict(r), sort_keys=True)


def test_route_tiered_is_bit_identical_to_the_pre_cache_implementation():
    """E3.4 acceptance, the correctness half: over a few hundred real queries (every 5th alias and card name of
    the 893-card default catalog, in catalog order, plus word salad and an empty query), the memoised families()
    and the reused ranking give EXACTLY the record the old code gave -- tier, options, z, p, id, reason. (The
    timing half, 12.9 -> 1.3 ms per route at 3,982 cards, is measured in tools/bench_router.py, not here: a timing
    assert in a unit test is a flake generator.)"""
    from holographic.caching_and_storage.holographic_catalog import default_catalog
    cat = default_catalog()
    qs = []
    for c in cat.all():
        qs.append(c.name)
        qs.extend(c.aliases)
    qs = qs[::5][:240] + ["purple monkey dishwasher", "asdf qwer zxcv", "", "the"]
    for q in qs:
        assert _canon(cat.route_tiered(q)) == _canon(_route_tiered_pre_e34(cat, q)), q
    for kw in ({"clarify": True}, {"k": 3}):                  # the other knobs, on a slice (the oracle is slow)
        for q in qs[::6]:
            assert _canon(cat.route_tiered(q, **kw)) == _canon(_route_tiered_pre_e34(cat, q, **kw)), (q, kw)


def test_families_memo_is_retired_by_every_content_change():
    """The memo is keyed on the catalog's content version: a registration after construction (what a plugin does),
    an unregistration, and an in-place edit of a card's does/module (what a test or a tool may do) each show up
    in the very next families() -- and in route_tiered's families annotation."""
    from holographic.caching_and_storage.holographic_catalog import Catalog
    cat = Catalog()
    cat.register_capability("mesh smoother", "smooth a mesh (holographic_meshsmooth)")
    cat.register_capability("plain thing", "does a plain thing")
    f1 = cat.families()
    assert f1["mesh smoother"][0] is not None and f1["plain thing"] == (None, None)
    assert cat.families() == f1 and cat._families_memo() is cat._families_memo()      # memo hit: same object
    # a card registered AFTER the first call (plugins do this)
    cat.register_capability("late card", "a late card (holographic_meshsmooth)")
    assert cat.families()["late card"][0] == f1["mesh smoother"][0]
    # an in-place edit of an input of family()
    cat.get("plain thing").does = "now it names holographic_meshsmooth"
    assert cat.families()["plain thing"][0] == f1["mesh smoother"][0]
    cat.get("plain thing").module = "no_such_module_stem"
    assert cat.families()["plain thing"] == (None, None)
    # unregister
    assert cat.unregister("late card") is True
    assert "late card" not in cat.families()
    # route_tiered's annotation reads the same memo
    r = cat.route_tiered("smooth a mesh")
    assert r["options"][0]["family"] == f1["mesh smoother"][0]
    # families() hands out a COPY: a caller editing it cannot poison the memo
    f = cat.families()
    f["mesh smoother"] = ("vandalised", "module")
    assert cat.families()["mesh smoother"][0] == f1["mesh smoother"][0]


# ============================================================================== E4.1 -- the router learns
def _mind():
    import lecore
    return lecore.UnifiedMind(dim=256, seed=0)


TIE_Q = "denoise an image"            # a three-way lexical TIE at 2.5 (measured): the tie-break is the whole decision
TIE_USED = "Denoise (domain)"


def test_a_fresh_mind_routes_exactly_as_the_lexical_catalog():
    """Router mode is OFF until the router is pretrained (router_learn) or switched on: a fresh mind's route_tiered
    is the catalog's lexical route, tier and options unchanged."""
    m = _mind()
    assert m.router_mode() is False
    r = m.route_tiered(TIE_Q)
    c = m._capability_catalog().route_tiered(TIE_Q)
    assert r["tier"] == c["tier"] and r["options"] == c["options"] and r["learned"] is None


def test_a_reported_route_outcome_teaches_the_router_and_breaks_the_tie_its_way():
    m = _mind()
    m.router_mode(True)                                  # an empty learned store: nothing labelled yet
    r = m.route_tiered(TIE_Q)
    names = [o["name"] for o in r["options"]]
    assert TIE_USED in names and r["learned"] is not None
    rep = m.decision_outcome(r["id"], TIE_USED)          # the caller used THIS card
    fw = rep.get("forwarded")
    assert any(isinstance(f, dict) and f.get("router") for f in (fw if isinstance(fw, list) else [fw]))
    info = m.router_report()
    assert info["verdicts"] == 1 and info["top"][0] == (TIE_USED, 1)
    r2 = m.route_tiered(TIE_Q)
    assert r2["options"][0]["name"] == TIE_USED and r2["options"][0]["verdicts"] == 1.0
    assert r2["options"][0]["lexical"] == 2.5            # the lexical score did not move; the learned terms did


def test_the_learned_router_survives_a_restart(tmp_path):
    m = _mind()
    m.router_mode(True)
    r = m.route_tiered(TIE_Q)
    m.decision_outcome(r["id"], TIE_USED)
    root = str(tmp_path / "p")
    os.makedirs(os.path.join(root, "learning"), exist_ok=True)
    m.learning_save(root)
    m2 = _mind()
    m2.learning_load(root)
    assert m2.router_mode() is True and m2.router_report()["top"][0] == (TIE_USED, 1)
    assert [o["name"] for o in m2.route_tiered(TIE_Q)["options"]] == [o["name"] for o in m.route_tiered(TIE_Q)["options"]]


def test_router_learn_pretrains_on_alias_pairs_with_leave_one_out_rivals():
    """router_learn takes (alias, card) pairs; each alias meets the rivals it would face if it were NOT in the index."""
    m = _mind()
    rep = m.router_learn(pairs=[("clean up a noisy photo", TIE_USED), ("remove grain from a picture", TIE_USED)])
    assert rep["pairs"] == 2 and rep["mode"] is True and rep["rows"] >= 2
    assert m.router_report()["top"][0] == (TIE_USED, 2)
    assert m.route_tiered(TIE_Q)["options"][0]["name"] == TIE_USED


# ============================================================================== E4.5 -- serve asks the router
def _alias_truth(m):
    truth = {}
    for c in m._capability_catalog().all():
        for a in (c.aliases or ()):
            truth.setdefault(a, set()).add(c.name)
    return truth


def test_serve_answers_a_capability_question_from_the_router():
    """The panel's G4: serve never consulted the router, so every alias probe escalated. A catalog alias (as the live
    service holds it: in the index) is now served as 'use capability X' with the route's id -- and it is RIGHT."""
    m = _mind()
    truth = _alias_truth(m)
    r = m.serve("decode a png")
    assert r["via"] == "route" and r["capability"] in truth["decode a png"], r
    assert m.decision_ledger().get(r["id"]).via == "route" and r["bar"]["on"].startswith("bench")
    assert "use capability" in r["answer"]


def test_serve_does_not_let_the_router_hijack_ordinary_questions():
    """Ordinary non-capability questions and a taught fact's rewording must never come back as 'use capability X'."""
    m = _mind()
    m.teach("who painted the mona lisa", "Leonardo da Vinci")
    for q in ("what is the capital of peru", "how many legs does a spider have", "who wrote hamlet",
              "remind me to call my mother tomorrow", "who painted the mona lisa?",
              "the mona lisa was painted by which artist"):
        r = m.serve(q)
        assert r["via"] != "route", (q, r)


def test_an_escalation_carries_the_route_menu():
    m = _mind()
    r = m.serve(TIE_Q)
    assert r["via"] == "escalate" and r["route"]["tier"] in ("menu", "refuse")
    assert TIE_USED in r["route"]["options"] and m.decision_ledger().get(r["route"]["id"]) is not None

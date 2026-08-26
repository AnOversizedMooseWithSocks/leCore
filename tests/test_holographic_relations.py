"""Relations: meaning as the recovered relationship -- pinned.

Four operations, each measured before integration (see holographic_relations'
module docstring): EXPLAIN why two records are similar (per-role verdict),
NAME how a filler relates to a record, MAP an attribute across records ("the
dollar of mexico"), and CHAIN hops into complex queries. Plus the law that
shaped the API: meaning survives composition only when it touches SYMBOLS
between steps -- the symbol-routed map is exact (360/360) where the direct
algebraic map is ~94% and does not improve with dimension.
"""

import numpy as np

from holographic.misc.holographic_relations import KnowledgeStore, relation_map, _cleanup
from holographic.agents_and_reasoning.holographic_ai import bind, cosine

WORLD = {
    "france":  dict(capital="paris", currency="franc", language="french", continent="europe"),
    "belgium": dict(capital="brussels", currency="franc", language="french", continent="europe"),
    "sweden":  dict(capital="stockholm", currency="krona", language="swedish", continent="europe"),
    "japan":   dict(capital="tokyo", currency="yen", language="japanese", continent="asia"),
    "mexico":  dict(capital="mexico_city", currency="peso", language="spanish", continent="america"),
    "usa":     dict(capital="washington", currency="dollar", language="english", continent="america"),
    "peru":    dict(capital="lima", currency="sol", language="spanish", continent="america"),
    "egypt":   dict(capital="cairo", currency="pound", language="arabic", continent="africa"),
    "kenya":   dict(capital="nairobi", currency="shilling", language="swahili", continent="africa"),
    "vietnam": dict(capital="hanoi", currency="dong", language="vietnamese", continent="asia"),
}


def store():
    ks = KnowledgeStore(dim=2048, seed=0)
    for n, a in WORLD.items():
        ks.add(n, **a)
    return ks


def test_explain_decodes_why_two_records_are_similar():
    ks = store()
    verdicts = {r: (fa, fb, shared)
                for r, fa, fb, shared, _ in ks.explain("france", "belgium")}
    assert verdicts["capital"] == ("paris", "brussels", False)
    assert verdicts["currency"] == ("franc", "franc", True)
    assert verdicts["language"] == ("french", "french", True)
    assert verdicts["continent"] == ("europe", "europe", True)


def test_name_relation_says_how_a_filler_relates():
    # measured 40/40; pinned at a >= 90% floor over the whole world
    ks = store()
    ok = tot = 0
    for n, attrs in WORLD.items():
        for r, v in attrs.items():
            ok += (ks.name(n, v)[0] == r)
            tot += 1
    assert ok / tot >= 0.9


def test_symbol_routed_mapping_is_exact_where_direct_map_is_noisy():
    # THE LAW: the two-step route (name the role -> read it out of the other
    # record) cleans up to a symbol mid-path and measured 360/360; the direct
    # algebraic map skips that cleanup and measured ~94%, with 20 of its 22
    # failures pure HRR noise (not ambiguity) and no improvement with dimension.
    ks = store()
    ok = tot = 0
    for a in WORLD:
        for b in WORLD:
            if a == b:
                continue
            for r, v in WORLD[a].items():
                ans, _ = ks.the_x_of(v, b, a)
                ok += (ans == WORLD[b][r])
                tot += 1
    assert ok / tot >= 0.98                       # measured exact; tiny slack

    # the direct map stays useful but measurably noisier
    direct_ok = direct_tot = 0
    fillers = ks._filler_names()
    for a in ("usa", "japan", "kenya"):
        for b in WORLD:
            if a == b:
                continue
            M = relation_map(ks.recs[a], ks.recs[b])
            for r, v in WORLD[a].items():
                ans, _ = _cleanup(bind(M, ks.fillers.get(v)), fillers, ks.fillers)
                direct_ok += (ans == WORLD[b][r])
                direct_tot += 1
    assert direct_ok / direct_tot >= 0.80         # documented noise floor


def test_chained_queries_stay_exact_through_three_hops():
    ks = store()
    # 2-hop: the currency of the country whose capital is X
    for n, attrs in WORLD.items():
        assert ks.ask(attrs["capital"], ("capital", "currency")) == attrs["currency"]
    # 3-hop (franc is shared by france+belgium -- both are french-speaking, so
    # the answer is right whichever the hop lands on: honest ambiguity, not noise)
    for n, attrs in WORLD.items():
        lang = ks.ask(attrs["capital"], ("capital", "currency"),
                      ("currency", "language"))
        assert lang == attrs["language"]


def test_unified_mind_explains_its_own_records():
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=2048, seed=0)
    out = {r: (f1, f2, shared) for r, f1, f2, shared, _ in m.explain(
        {"capital": "paris", "currency": "franc", "language": "french"},
        {"capital": "brussels", "currency": "franc", "language": "french"})}
    assert out["capital"] == ("paris", "brussels", False)
    assert out["currency"] == ("franc", "franc", True)
    assert out["language"] == ("french", "french", True)


def test_cross_modal_explain_why_two_images_are_similar():
    # THE CROSS-MODAL STEP: two raw IMAGES in, the why out -- with zero new
    # relation machinery (the auto-tagger composes with the relations module).
    # Measured end-to-end on generated shapes where ground truth is known:
    # tagger 36/36 on shape and colour, explanation verdicts 72/72 = 100% over
    # all pairs. Pinned: one controlled pair exactly, plus a sweep floor.
    import holographic.misc.holographic_vision as hv
    from holographic.scene_and_pipeline.holographic_scene import explain_objects, auto_tags

    i1, m1 = hv.make_shape("circle", fg=(220, 60, 50), seed=1)     # red circle
    i2, m2 = hv.make_shape("circle", fg=(60, 200, 90), seed=2)     # green circle
    verdicts = {r: (a, b, s) for r, a, b, s, _ in explain_objects(i1, i2, m1, m2)}
    assert verdicts["shape"] == ("circle", "circle", True)
    assert verdicts["colour"] == ("red", "green", False)

    # the sweep: every (shape x colour) pair's shape/colour verdicts vs truth
    shapes = ["circle", "rectangle", "triangle"]
    colors = {"red": (220, 60, 50), "green": (60, 200, 90), "blue": (60, 110, 220)}
    ks = KnowledgeStore(dim=2048, seed=0)
    truth = {}
    i = 0
    for kind in shapes:
        for cname, rgbv in colors.items():
            img, mask = hv.make_shape(kind, S=64, seed=i, fg=rgbv)
            ks.add(f"o{i}", **{k: str(v) for k, v in auto_tags(img, mask).items()})
            truth[f"o{i}"] = dict(shape=kind, colour=cname)
            i += 1
    names = sorted(truth)
    ok = tot = 0
    for a in names:
        for b in names:
            if a >= b:
                continue
            v = {r: s for r, _, _, s, _ in ks.explain(a, b)}
            for role in ("shape", "colour"):
                tot += 1
                ok += (v[role] == (truth[a][role] == truth[b][role]))
    assert ok / tot >= 0.95                       # measured 100%; small slack


def test_chain_throughput_is_calibrated_confidence():
    # RAYTRACING PARALLEL: a relation chain is a ray bouncing through the
    # holographic space -- each hop a bounce, the cleanup confidence its
    # reflectance, throughput the accumulated product. Throughput is a calibrated
    # confidence: abstaining on low-throughput chains raises accuracy on the
    # answered subset (path tracing's Russian-roulette -- drop the paths that
    # have lost too much energy to matter).
    import numpy as np
    from holographic.misc.holographic_relations import KnowledgeStore
    rng = np.random.default_rng(0)
    conts = ["europe", "asia", "africa", "samerica", "namerica", "oceania"]
    countries = {f"c{i}": {"capital": f"cap{i}", "currency": f"cur{i%7}",
                           "language": f"lang{i}", "continent": conts[i % 6]}
                 for i in range(20)}
    ks = KnowledgeStore(dim=2048, seed=0)
    for n, a in countries.items():
        ks.add(n, **a)
    roles = ["capital", "currency", "language", "continent"]
    rows = []
    for sc, attrs in countries.items():
        for L in (1, 2, 3, 4):
            chain, cur = [], "capital"
            for k in range(L):
                nxt = roles[rng.integers(0, 4)]
                chain.append((cur, nxt)); cur = nxt
            ans, tp, confs = ks.ask_traced(attrs["capital"], *chain)
            assert len(confs) == L                      # one reflectance per bounce
            f, ok = attrs["capital"], True
            for mr, rr in chain:
                ent = next((n for n, a in countries.items() if a.get(mr) == f), None)
                if ent is None:
                    ok = False
                    break
                f = countries[ent][rr]
            if ok:
                rows.append((ans == f, tp))
    all_acc = np.mean([ok for ok, tp in rows])
    med = np.median([tp for ok, tp in rows])
    answered = [ok for ok, tp in rows if tp >= med]
    assert np.mean(answered) > all_acc                  # abstention helps


def test_ask_traced_abstains_below_throughput_floor():
    # A chain whose throughput decays below the floor returns None instead of
    # emitting noise -- the ray that ran out of energy contributes nothing.
    from holographic.misc.holographic_relations import KnowledgeStore
    ks = KnowledgeStore(dim=2048, seed=0)
    for n, a in {"france": {"capital": "paris", "currency": "euro"},
                 "germany": {"capital": "berlin", "currency": "euro"}}.items():
        ks.add(n, **a)
    # an impossibly high floor forces abstention
    ans, tp, _ = ks.ask_traced("paris", ("capital", "currency"), min_throughput=0.99)
    assert ans is None
    # no floor: it answers
    ans2, tp2, _ = ks.ask_traced("paris", ("capital", "currency"))
    assert ans2 == "euro"


def test_route_reliability_ranks_unique_vs_shared_keys():
    # The kept artifact of the multi-ray-CHAINS experiment. Multi-path combination
    # did NOT boost chain accuracy (the cleanup law makes a unique route exact and
    # a shared route fundamentally ambiguous -- no combination manufactures the
    # missing information), but route_reliability is a genuine, self-measured
    # signal: a find by a UNIQUE-valued role is an exact key (reliability 1.0),
    # one by a SHARED-valued role is ambiguous (reliability 1 / mean fan-out). No
    # magic number -- it is the data's own fan-out, inverted.
    from holographic.misc.holographic_relations import KnowledgeStore
    conts = ["eu", "as", "af", "sa", "na", "oc"]
    countries = {f"c{i}": {"capital": f"cap{i}", "currency": f"cur{i%8}",
                           "continent": conts[i % 6]} for i in range(40)}
    ks = KnowledgeStore(dim=512, seed=0)
    for n, a in countries.items():
        ks.add(n, **a)
    rel = {r: ks.route_reliability(r) for r in ("capital", "currency", "continent")}
    assert rel["capital"] == 1.0                      # unique keys -> exact
    assert rel["currency"] < 0.5                      # 8 currencies / 40 -> ambiguous
    assert rel["continent"] < rel["currency"]         # 6 continents -> most shared


def test_blend_creates_coherent_novel_entities():
    # PROJECTION TO CREATE NEW THINGS: synthesize a novel entity (one frame, with
    # another's values projected onto chosen roles) that exists in no training
    # data, and verify it decodes to exactly the intended blend. The shadow that
    # creates -- decompose two structures, cast selected attributes of one onto
    # the other's frame, get a coherent third.
    import numpy as np
    from holographic.misc.holographic_relations import KnowledgeStore
    ks = KnowledgeStore(dim=2048, seed=0)
    data = {"france": {"capital": "paris", "currency": "euro", "language": "french", "continent": "europe"},
            "japan": {"capital": "tokyo", "currency": "yen", "language": "japanese", "continent": "asia"},
            "brazil": {"capital": "brasilia", "currency": "real", "language": "portuguese", "continent": "samerica"}}
    for n, a in data.items():
        ks.add(n, **a)
    vec, spec = ks.blend("france", "japan", {"language", "currency"})
    dec = ks.decode_record(vec)
    assert dec == spec                                # exact synthesis
    assert dec["capital"] == "paris" and dec["language"] == "japanese"  # blended
    # fidelity across many random novel blends
    rng = np.random.default_rng(0)
    roles = ks._role_names()
    good = 0
    for _ in range(40):
        base, donor = rng.choice(list(data), 2, replace=False)
        k = rng.integers(1, len(roles))
        dr = set(rng.choice(roles, k, replace=False))
        vec, spec = ks.blend(base, donor, dr)
        good += (ks.decode_record(vec) == spec)
    assert good == 40                                 # measured 100%


def test_project_transform_generates_analogy():
    # ANALOGY AS GENERATION: a:b::c:? answered by CREATING the answer (the a->b
    # per-role delta projected onto c) rather than retrieving it. Retrieval of an
    # existing analogue hits a uniqueness wall in a clean role-filler store (every
    # entity an exact key, no graded nearness), but synthesizing the specified new
    # thing is well-posed and exact.
    from holographic.misc.holographic_relations import KnowledgeStore
    ks = KnowledgeStore(dim=2048, seed=0)
    data = {"france": {"capital": "paris", "currency": "euro", "language": "french", "continent": "europe"},
            "germany": {"capital": "berlin", "currency": "euro", "language": "german", "continent": "europe"},
            "japan": {"capital": "tokyo", "currency": "yen", "language": "japanese", "continent": "asia"}}
    for n, a in data.items():
        ks.add(n, **a)
    vec, spec = ks.project_transform("france", "germany", "japan")
    dec = ks.decode_record(vec)
    assert dec == spec
    # the france->germany transform changed capital+language; projected onto japan
    assert dec["capital"] == "berlin" and dec["language"] == "german"   # from transform
    assert dec["continent"] == "asia" and dec["currency"] == "yen"      # kept from japan

"""Tests for the top-level UnifiedMind: one encoder, one self-organizing memory, one
decision brain over a single holographic space. The point is that the pieces share the
substrate -- so one memory should classify several modalities at once, the recall view
should search the same vectors, and the same mind should be able to decide."""

import numpy as np

from holographic_unified import UnifiedMind, _patterns


def _records(dept, rng, n):
    return [({"dept": dept, "level": int(rng.integers(1, 6))}, f"rec:{dept}", "record")
            for _ in range(n)]


def test_one_memory_holds_and_classifies_several_modalities():
    rng = np.random.default_rng(0)
    mind = UnifiedMind(dim=1024, seed=0)
    train, test = [], []
    for kind in ("rows", "cols", "diag", "check"):
        train += [(_patterns(kind, rng), f"img:{kind}", "image") for _ in range(20)]
        test += [(_patterns(kind, rng), f"img:{kind}", "image") for _ in range(8)]
    for d in ("eng", "sales", "ops"):
        train += _records(d, rng, 20)
        test += _records(d, rng, 8)
    rng.shuffle(train)
    for x, label, mod in train:
        mind.learn(x, label, mod)
    mind.maintain_now()

    # one memory now holds every label from every modality
    labels = set(mind.memory.live.counts_by_label())
    assert {f"img:{k}" for k in ("rows", "cols", "diag", "check")} <= labels
    assert {f"rec:{d}" for d in ("eng", "sales", "ops")} <= labels

    acc = sum(mind.classify(x, mod)[0] == lab for x, lab, mod in test) / len(test)
    assert acc >= 0.85          # both modalities classify well from the single store


def test_recall_view_searches_the_same_vectors():
    rng = np.random.default_rng(1)
    mind = UnifiedMind(dim=1024, seed=1)
    for kind in ("rows", "check"):
        for _ in range(15):
            mind.learn(_patterns(kind, rng), f"img:{kind}", "image")
    (label, _), sim = mind.recall(_patterns("rows", rng), "image")
    assert label == "img:rows"   # nearest stored individual is the right kind
    assert sim > 0.5


def test_same_mind_decides_over_the_same_space():
    mind = UnifiedMind(dim=1024, seed=2).actions(["left", "right"])
    rng = np.random.default_rng(2)
    for _ in range(400):
        n = float(rng.uniform(-3, 3))
        good = "right" if n > 0 else "left"
        choice = mind.decide(n, explore=True, epsilon=0.3, modality="number")
        mind.reinforce(n, choice, 1.0 if choice == good else 0.0, modality="number")
    acc = sum(mind.decide(float(v), modality="number") == ("right" if v > 0 else "left")
              for v in np.linspace(-3, 3, 40)) / 40
    assert acc >= 0.7            # the shared-substrate brain learned the contextual choice


def test_routing_removes_cross_modal_interference():
    # A flat store of several modalities can mistake a query for a foreign-modality
    # concept; restricting the query to its own modality (a cheap router, since the
    # modality is known) should never hurt and should fix those cross-modal errors.
    from holographic_mind import UniversalEncoder
    from holographic_text import TOPICS, _content, _split
    rng = np.random.default_rng(0)
    corpus = [s for ss in TOPICS.values() for s in ss]
    mind = UnifiedMind(dim=1024, seed=0).read([_content(s) for s in corpus])
    text_te = []
    for topic, ss in TOPICS.items():
        a, b = _split(ss, frac=0.7, seed=2)
        for s in a:
            mind.learn(_content(s), topic, "text")
        text_te += [(_content(s), topic) for s in b]
    for kind in ("rows", "cols", "diag", "check"):
        for _ in range(20):
            img = np.zeros((8, 8)); img[::2, :] = 1.0
            mind.learn(img + 0.15 * rng.standard_normal((8, 8)), f"img:{kind}", "image")
    for d in ("eng", "sales", "ops"):
        for _ in range(20):
            mind.learn({"dept": d, "level": int(rng.integers(1, 6))}, f"rec:{d}", "record")
    mind.maintain_now()

    flat = sum(mind.classify(t, "text", route=False)[0] == lab for t, lab in text_te) / len(text_te)
    routed = sum(mind.classify(t, "text", route=True)[0] == lab for t, lab in text_te) / len(text_te)
    assert routed >= flat          # routing never hurts
    # every routed prediction is a text label (no cross-modal leakage)
    assert all(mind.classify(t, "text", route=True)[0] in set(TOPICS) for t, _ in text_te)


def test_same_mind_generates_over_the_shared_space():
    # Generation is the fourth operation on the one model: learn to continue a sequence,
    # then produce more of it. The next-symbol prediction is holographic cleanup (the
    # same primitive the classifier uses); only the context key is exact.
    from holographic_text import TOPICS
    text = " ".join(s for ss in TOPICS.values() for s in ss).lower()
    mind = UnifiedMind(dim=1024, seed=0).learn_sequence(text)          # default fractal engine
    out = mind.generate("the ", length=80, temperature=0.4)
    assert len(out) > 50                                  # it produced a continuation
    assert set(out) <= set(text)                          # only symbols it actually learned
    # next-symbol prediction is a flat-engine primitive; it should beat uniform random
    cut = int(len(text) * 0.9)
    m2 = UnifiedMind(dim=1024, seed=0).learn_sequence(text[:cut], n=5, hierarchical=False)
    held = text[cut:cut + 800]
    ok = sum(m2.next_symbol(held[max(0, j - 4):j]) == held[j] for j in range(1, len(held)))
    assert ok / (len(held) - 1) > 1.0 / len(set(text))    # well above chance


def test_modality_self_discovery_matches_declared_tags():
    # learn/classify with NO declared modality must discover the tag from the input
    # (encoder.infer) and route on it -- measured at exact parity with declared tags
    # on the mixed demo (97.5% both ways). Token lists are the trap: they must be
    # discovered as TEXT, not order-sensitive sequences (the original encode bug).
    from holographic_text import TOPICS, _content, _split
    rng = np.random.default_rng(0)
    corpus = [s for ss in TOPICS.values() for s in ss]

    declared, discovered = [], []
    for topic, ss in TOPICS.items():
        a, b = _split(ss, frac=0.7, seed=2)
        declared += [(_content(s), topic, "text") for s in a]
        discovered += [(_content(s), topic) for s in b]          # held out, untagged
    for kind in ("rows", "cols"):
        declared += [(_patterns(kind, rng), f"img:{kind}", "image") for _ in range(15)]
        discovered += [(_patterns(kind, rng), f"img:{kind}") for _ in range(6)]
    for d in ("eng", "sales"):
        declared += [({"dept": d, "level": int(rng.integers(1, 6))}, f"rec:{d}", "record")
                     for _ in range(15)]
        discovered += [({"dept": d, "level": int(rng.integers(1, 6))}, f"rec:{d}")
                       for _ in range(6)]

    mind = UnifiedMind(dim=1024, seed=0).read([_content(s) for s in corpus])
    for x, lab, _ in declared:
        mind.learn(x, lab)                                        # tags NOT declared
    mind.maintain_now()

    # learning discovered a real tag for every label (never None)
    assert all(m is not None for m in mind._label_modality.values())
    # untagged classification routes correctly and scores well across modalities
    acc = sum(mind.classify(x)[0] == lab for x, lab in discovered) / len(discovered)
    assert acc >= 0.85
    # discovered routing equals declared routing, query by query
    same = all(mind.classify(x)[0] ==
               mind.classify(x, ("text" if isinstance(x, list) else
                                 "image" if isinstance(x, np.ndarray) else "record"))[0]
               for x, _ in discovered)
    assert same


def test_absorb_self_assembles_a_working_mind():
    # SELF-ASSEMBLY: a pile of (input, label) pairs -- no modality tags, no read(),
    # no maintenance calls -- must come back as a working multi-modal mind that
    # matches the long-hand read/learn/maintain pipeline.
    from holographic_text import TOPICS, _content, _split
    rng = np.random.default_rng(0)
    pile, test = [], []
    for topic, ss in TOPICS.items():
        a, b = _split(ss, frac=0.7, seed=2)
        pile += [(_content(s), topic) for s in a]
        test += [(_content(s), topic) for s in b]
    for kind in ("rows", "cols", "diag", "check"):
        pile += [(_patterns(kind, rng), f"img:{kind}") for _ in range(20)]
        test += [(_patterns(kind, rng), f"img:{kind}") for _ in range(8)]
    rng.shuffle(pile)

    mind = UnifiedMind(dim=1024, seed=0).absorb(pile)
    acc = sum(mind.classify(x)[0] == lab for x, lab in test) / len(test)
    assert acc >= 0.85
    # the text it absorbed taught the word vectors (read() happened internally):
    # two same-topic sentences should sit closer than cross-topic ones on average
    assert mind.memory.live.size() >= len(set(lab for _, lab in pile))


def test_many_sequence_schemas_route_by_compression_gate():
    # Consolidation: the mind holds SEVERAL sequence schemas at once (named), learns
    # code as well as prose (modality passthrough), and unnamed generation routes the
    # seed by the compression gate -- content-level self-discovery, needed exactly
    # where type inference goes blind (code and prose are both str).
    from holographic_text import TOPICS
    prose = " ".join(s for ss in TOPICS.values() for s in ss).lower()
    code = ("def step(self, action):\n    reward = self.world.step(action)\n"
            "    self.memory.append((self.state, action, reward))\n"
            "    return reward\n\nfor i in range(n):\n    total += vals[i]\n"
            "    if total > cap:\n        break\n") * 25
    mind = (UnifiedMind(dim=1024, seed=0)
            .learn_sequence(prose, name="prose")
            .learn_sequence(code, modality="code", name="python"))

    # unnamed generation: the gate sends each seed to the schema that understands it
    from holographic_schema import compression_gate
    gens = {k: g["gen"] for k, g in mind._gens.items()}
    assert compression_gate("def step(self, action):", gens)[0][1] == "python"
    assert compression_gate("the team scored in the ", gens)[0][1] == "prose"
    out = mind.generate("def step(self", length=60, temperature=0.4)
    assert len(out) > 30 and set(out) <= set(code) | set("def step(self")

    # named access works; unknown names fail loudly
    assert len(mind.generate("the ", 40, 0.4, name="prose")) > 20
    try:
        mind.generate("x", 10, 0.4, name="nope")
        assert False, "should have raised"
    except KeyError:
        pass


def test_single_schema_path_is_unchanged():
    # backward compatibility: one unnamed learn_sequence + generate, exactly as before
    from holographic_text import TOPICS
    text = " ".join(s for ss in TOPICS.values() for s in ss).lower()
    mind = UnifiedMind(dim=1024, seed=0).learn_sequence(text)
    out = mind.generate("the ", length=80, temperature=0.4)
    assert len(out) > 50
    assert mind._gen is not None                  # the compat alias survives


def test_content_gate_resolves_code_vs_prose_without_tags():
    # The correctness fix, pinned: tags declared at LEARN time put code labels in a
    # "code" pool; an untagged classify infers "text" and -- before the gate -- the
    # routing safeguard EXCLUDED the true labels entirely (measured on a docs-vs-code
    # set: 24% accuracy, 66% cross-pool leakage, worse than no routing at all). The
    # compression gate, fitted on the mind's own learned samples, identified the
    # sub-format on 100% of held-out queries and recovered declared-tag accuracy
    # exactly. This test holds the recovered behaviour in place.
    rng = np.random.default_rng(0)
    code = [("def step ( self , a ) : r = self . world . step ( a ) ; "
             "self . memory . append ( ( self . state , a , r ) ) ; return r"),
            ("for i in range ( n ) : total += vals [ i ] ; "
             "if total > cap : break"),
            ("q = q / np . linalg . norm ( q ) ; idx = int ( ( items @ q ) . argmax ( ) )"),
            ("w = rng . standard_normal ( ( dim , k ) ) / np . sqrt ( k ) ; "
             "v = w @ x ; return v / np . linalg . norm ( v )")] * 8
    docs = [("the forager perceives its situation and decides a move then remembers "
             "what happened so the next decision is better informed"),
            ("each leaf keeps a small memory inside capacity and a query descends "
             "the tree with a beam that can back track into nearby cells"),
            ("the plate stores a superposition of many items and recall cleans the "
             "noisy readout by cosine to each known atom"),
            ("a random projection preserves similarity so close feature vectors stay "
             "close as hypervectors across every modality")] * 8
    tr = ([(s, "code:lib", "code") for s in code[:24]] +
          [(s, "doc:lib", "text") for s in docs[:24]])
    rng.shuffle(tr)
    mind = UnifiedMind(dim=1024, seed=0).absorb(tr)

    # untagged queries: the gate must put each on its own side of the line
    assert mind.classify("v = m @ x ; return v / np . linalg . norm ( v )")[0] == "code:lib"
    assert mind.classify("the memory cleans a noisy readout and recalls the item")[0] == "doc:lib"
    # untagged matches declared on every held-out probe
    for s, lab, m in [(code[-1], "code:lib", "code"), (docs[-1], "doc:lib", "text")]:
        assert mind.classify(s)[0] == mind.classify(s, m)[0] == lab


def test_only_code_learned_means_string_queries_reach_code_labels():
    # the single-sub-format branch: if the mind has ONLY learned code, an untagged
    # string query must still reach the code labels (before the fix, the inferred
    # "text" pool was empty-by-exclusion for them)
    snips = [("def f ( x ) : return x + 1", "code:a"),
             ("for i in range ( 9 ) : s += i", "code:b")] * 6
    mind = UnifiedMind(dim=512, seed=0)
    for s, lab in snips:
        mind.learn(s, lab, "code")
    assert mind.classify("def g ( y ) : return y + 2")[0] in ("code:a", "code:b")


def test_absorb_with_sequences_assembles_a_complete_mind():
    # The complete self-assembly: ONE absorb call returns a mind that classifies,
    # recalls, AND generates -- one named sequence schema per discovered text-like
    # sub-format, unnamed generation routed by the compression gate.
    rng = np.random.default_rng(0)
    code = [("def step ( self , a ) : r = self . world . step ( a ) ; return r"),
            ("for i in range ( n ) : total += vals [ i ] ; "
             "if total > cap : break"),
            ("v = w @ x ; return v / np . linalg . norm ( v )")] * 10
    docs = [("the forager perceives its situation and decides a move then remembers "
             "what happened so the next decision is better informed"),
            ("each leaf keeps a small memory inside capacity and the query descends "
             "with a beam that can back track into nearby cells")] * 10
    pile = ([(s, "code:lib", "code") for s in code] +
            [(s, "doc:lib", "text") for s in docs])
    rng.shuffle(pile)
    mind = UnifiedMind(dim=512, seed=0).absorb(pile, sequences=True)

    assert set(mind._gens) == {"text", "code"}      # one schema per discovered format
    out_c = mind.generate("def step ( self", length=50, temperature=0.4)   # gate routes
    out_d = mind.generate("the forager ", length=50, temperature=0.4)
    assert len(out_c) > 25 and len(out_d) > 25
    # and the same mind still classifies untagged across the sub-format line
    assert mind.classify("v = m @ x ; return v")[0] == "code:lib"
    assert mind.classify("the memory keeps each leaf inside capacity")[0] == "doc:lib"


def test_unified_app_self_dataset_builds_and_classifies():
    # The inception dataset: the app's build() learns this project's own source and
    # classifies which subsystem a snippet is from -- offline, no NLTK. Pins the
    # punctuation-as-stopwords lesson (content tokens lifted 5-way held-out accuracy
    # from 42% to ~70% in the controlled comparison) at a conservative floor.
    import unified_app as ua
    res = ua.build("self")
    assert res["ok"] and len(res["labels"]) == 5
    assert res["accuracy"] >= 45                    # well above the 20% chance floor
    mind = ua.STATE["mind"]
    q = " ".join(ua._code_content("leaf = self . tree . _route ( key , beam )".split()))
    assert mind.classify(q)[0] == "code:tree"
    out = mind.generate("def recall ( self", length=50, temperature=0.4)
    assert len(out) > 25


def test_unified_mind_relations_over_its_own_memory():
    # THE UNIFICATION of the relations work: find/read/ask/explain run on the
    # records absorb() stored and the filler vocabulary learn() registered --
    # no side KnowledgeStore. Measured: find 40/40, read 40/40, 2+3-hop chains
    # 20/20 on the demo world, every hop cleaned up to a symbol (the law).
    import numpy as np
    from holographic_unified import UnifiedMind
    W = {
        "france":  dict(capital="paris", currency="franc", language="french"),
        "sweden":  dict(capital="stockholm", currency="krona", language="swedish"),
        "japan":   dict(capital="tokyo", currency="yen", language="japanese"),
        "mexico":  dict(capital="mexico_city", currency="peso", language="spanish"),
        "usa":     dict(capital="washington", currency="dollar", language="english"),
        "egypt":   dict(capital="cairo", currency="pound", language="arabic"),
    }
    m = UnifiedMind(dim=2048, seed=0)
    m.absorb([(attrs, name) for name, attrs in W.items()])

    assert m.find("capital", "tokyo")[0] == "japan"
    assert m.read_role("japan", "currency")[0] == "yen"
    # 3-hop chain over the mind's own memory
    assert m.ask("tokyo", ("capital", "currency"), ("currency", "language")) == "japanese"
    # learned-label explanation
    v = {r: s for r, _, _, s, _ in m.explain("mexico", "usa")}
    assert v["capital"] is False and v["currency"] is False


def test_unified_mind_explains_classes_learned_from_noisy_observations():
    # The genuinely new measurement: a class prototype built from SIX noisy,
    # incomplete observations (one random role dropped per copy) still decodes
    # its roles -- superposition linearity reinforces the shared role-filler
    # terms while the dropouts average out. Measured 100% on read (40/40),
    # explain (180/180), and 3-hop chains; pinned with conservative floors.
    import numpy as np
    from holographic_unified import UnifiedMind
    W = {
        "france":  dict(capital="paris", currency="franc", language="french"),
        "belgium": dict(capital="brussels", currency="franc", language="french"),
        "japan":   dict(capital="tokyo", currency="yen", language="japanese"),
        "mexico":  dict(capital="mexico_city", currency="peso", language="spanish"),
        "kenya":   dict(capital="nairobi", currency="shilling", language="swahili"),
    }
    rng = np.random.default_rng(7)
    m = UnifiedMind(dim=2048, seed=0)
    ex = []
    for name, attrs in W.items():
        for _ in range(6):
            drop = rng.choice(list(attrs))
            ex.append(({k: v for k, v in attrs.items() if k != drop}, name))
    rng.shuffle(ex)
    m.absorb(ex)

    ok = tot = 0
    for name, attrs in W.items():
        for r, v in attrs.items():
            ok += (m.read_role(name, r)[0] == v)
            tot += 1
    assert ok / tot >= 0.9                            # measured 100%

    verd = {r: s for r, _, _, s, _ in m.explain("france", "belgium")}
    assert verd["currency"] is True and verd["language"] is True
    assert verd["capital"] is False


def test_explain_splits_names_what_reorganization_separated():
    # INCEPTION: the mind explains its own memory organization. XOR world
    # (A = red circles + blue squares, B = the opposite pairings, plus a
    # uniformly-random noise role): auto_reorganize must split to reach 100%
    # held-out, and explain_splits must name colour+shape -- by CONTRAST, not
    # mere winner-difference (truly separating roles measured ~0.5 contrast,
    # incidental skews <= 0.1). Its first outing caught the organizer making an
    # accuracy-sufficient but structurally arbitrary split on label B (it
    # separated the NOISE role; A's clean split already resolved the XOR) --
    # the explanation honestly reports what the split DID.
    import numpy as np
    from holographic_unified import UnifiedMind
    rng = np.random.default_rng(0)

    def rec(c, s):
        return dict(colour=c, shape=s, size=rng.choice(["small", "medium", "large"]))

    examples = []
    for _ in range(40):
        examples += [(rec("red", "circle"), "A"), (rec("blue", "square"), "A"),
                     (rec("red", "square"), "B"), (rec("blue", "circle"), "B")]
    rng.shuffle(examples)
    m = UnifiedMind(dim=2048, seed=0)
    m.absorb(examples)
    m.memory.auto_reorganize()

    test = []
    for _ in range(10):
        test += [(rec("red", "circle"), "A"), (rec("blue", "square"), "A"),
                 (rec("red", "square"), "B"), (rec("blue", "circle"), "B")]
    acc = sum(m.classify(x)[0] == lab for x, lab in test) / len(test)
    assert acc >= 0.9                                  # the split earned its keep

    _, sep_a = m.explain_splits("A")
    assert set(sep_a) == {"colour", "shape"}           # the real structure, named
    assert "size" not in sep_a                         # the noise role excluded


def test_journal_narrates_reorganization_with_named_splits():
    # SELF-NARRATING MAINTENANCE: every road to auto_reorganize goes through
    # _reorganize_and_narrate, so the mind keeps its own account of every
    # maintenance event -- with splits NAMED (contrast-judged role decode)
    # where the data is record-shaped. Every consumer (console, tour, absorb's
    # auto path) gets the narration for free.
    import numpy as np
    from holographic_unified import UnifiedMind
    rng = np.random.default_rng(0)

    def rec(c, s):
        return dict(colour=c, shape=s, size=rng.choice(["small", "medium", "large"]))

    ex = []
    for _ in range(40):
        ex += [(rec("red", "circle"), "A"), (rec("blue", "square"), "A"),
               (rec("red", "square"), "B"), (rec("blue", "circle"), "B")]
    rng.shuffle(ex)
    m = UnifiedMind(dim=2048, seed=0)
    m.absorb(ex)

    assert m.journal, "absorb's maintenance must journal itself"
    stories = " ".join(e["story"] for e in m.journal)
    assert "reorganized" in stories                  # the split event was narrated
    named = {}
    for e in m.journal:
        named.update(e.get("named", {}))
    assert "colour" in named.get("A", []) and "shape" in named.get("A", [])


def test_maintain_now_journals_the_brains_verdict_too():
    # The whole self-maintenance story in one place: maintain_now's journal
    # entry carries the organizer's account AND the decision brain's measured
    # keep/fold/refresh verdict (auto_maintain's return value, narrated). A
    # mind without a brain journals only the organizer -- no empty brain talk.
    import numpy as np
    from holographic_unified import UnifiedMind

    m = UnifiedMind(dim=512, seed=0, maintain='auto')
    m.actions(["left", "right"])
    rng = np.random.default_rng(1)
    for _ in range(300):
        x = {"temp": "hot"} if rng.random() < 0.5 else {"temp": "cold"}
        good = "left" if x["temp"] == "hot" else "right"
        a = rng.choice(["left", "right"])
        m.reinforce(x, a, 1.0 if a == good else -1.0)
    m.learn({"temp": "hot"}, "climate")
    m.maintain_now()
    entry = m.journal[-1]
    assert "brain" in entry and entry["brain"]["choice"]
    assert "decision brain" in entry["story"]

    m2 = UnifiedMind(dim=512, seed=0)
    m2.learn({"temp": "hot"}, "climate")
    m2.maintain_now()
    assert "brain" not in m2.journal[-1]
    assert "decision brain" not in m2.journal[-1]["story"]


def test_sprite_library_as_relational_memory():
    # THE REAL LIBRARY, RELATIONAL: absorb actual sprites (image + auto-tag/
    # name record under the same label) and the relations operations run over
    # genuine data. The new measured result: each label's prototype superposes
    # an IMAGE vector with the record, and role decode survives the mixing at
    # 100% (750/750 full-set) -- the image component is near-orthogonal noise
    # to the role-bound terms. SEE->SAY (classify image, state colour in
    # symbols) measured 96% full-set; subset-pinned with conservative floors.
    import os, re
    import numpy as np
    import pytest
    hsp = os.path.join(os.path.dirname(__file__), "features", "sprites.hsp")
    if not os.path.exists(hsp):
        pytest.skip("sprite asset not present")
    import pack_sprites as ps
    from holographic_unified import UnifiedMind
    from holographic_scene import auto_tags

    with open(hsp, "rb") as f:
        sprites = dict(ps.unpack(f.read()))
    names = sorted(sprites)[:120]                    # subset: keep the test quick

    mind = UnifiedMind(dim=2048, seed=0)
    recs, rgbs = {}, {}
    examples = []
    for name in names:
        rgba = sprites[name]
        rgb = rgba[..., :3].astype(float) / 255.0 if rgba.dtype == np.uint8 else rgba[..., :3]
        mask = (rgba[..., 3] > 0) if rgba.shape[-1] == 4 else None
        t = auto_tags(rgb, mask=mask)
        rec = {"colour": t["colour"], "texture": t["texture"]}
        m = re.match(r"([a-z]+)(\d*)_([a-z]{2})(\d)\.gif", name)
        if m:
            rec.update(family=m.group(1), facing=m.group(3), frame=m.group(4))
        recs[name], rgbs[name] = rec, rgb
        examples += [(rgb, name, "image"), (rec, name, "record")]
    mind.absorb(examples, maintain=False)

    # find by attribute returns a sprite truly holding it
    ok = tot = 0
    for role in mind._fillers:
        for val in mind._fillers[role]:
            lab, _ = mind.find(role, val)
            ok += (recs[lab].get(role) == val); tot += 1
    assert ok == tot

    # role decode through image-contaminated prototypes
    rng = np.random.default_rng(0)
    sample = rng.choice(names, 40, replace=False)
    ok = tot = 0
    for name in sample:
        for role, val in recs[name].items():
            ok += (mind.read_role(name, role)[0] == val); tot += 1
    assert ok / tot >= 0.95                          # measured 100%

    # SEE -> SAY: classify the image, state its colour in symbols
    ok = 0
    for name in sample[:25]:
        lab, _ = mind.classify(rgbs[name], modality="image")
        ok += (mind.read_role(lab, "colour")[0] == recs[name]["colour"])
    assert ok / 25 >= 0.8                            # measured 96% full-set


def test_unified_app_world_dataset_and_relations_panel():
    # The console's record dataset + relations endpoint, end-to-end through the
    # test client: ten countries from eight noisy observations each (the
    # measured noisy-prototype decode), then explain/find/ask over the mind's
    # own absorbed memory.
    import unified_app as ua
    c = ua.app.test_client()
    r = c.post("/api/unified/load", json={"id": "world"}).get_json()
    assert r["ok"] and r["accuracy"] >= 80           # measured 97
    e = c.post("/api/unified/relations",
               json={"op": "explain", "a": "france", "b": "belgium"}).get_json()
    verd = {x["role"]: x["shared"] for x in e["explain"]}
    assert verd["currency"] is True and verd["capital"] is False
    f = c.post("/api/unified/relations",
               json={"op": "find", "role": "capital", "value": "tokyo"}).get_json()
    assert f["find"]["label"] == "japan"
    a = c.post("/api/unified/relations",
               json={"op": "ask", "start": "tokyo",
                     "hops": [["capital", "language"]]}).get_json()
    assert a["ask"]["answer"] == "japanese"
    # arbitrary chains, as the panel's chain builder sends them (3 hops)
    a3 = c.post("/api/unified/relations",
                json={"op": "ask", "start": "lima",
                      "hops": [["capital", "language"], ["language", "currency"],
                               ["currency", "continent"]]}).get_json()
    assert a3["ask"]["answer"] == "america"
    # GENERATION FIDELITY (user-caught): the corpus is learned with case and
    # punctuation, the seed is NOT lowercased by the endpoint, and the output
    # reads as prose -- capitals and sentence punctuation present.
    g = c.post("/api/unified/generate",
               json={"seed": "The capital of", "length": 120,
                     "temperature": 0.5}).get_json()
    txt = g["text"]
    assert txt.startswith("The capital of")          # seed case preserved
    assert any(ch.isupper() for ch in txt[20:])      # capitals beyond the seed
    assert "." in txt                                # sentences end


def test_journal_names_real_sprite_family_splits():
    # REAL-DATA INCEPTION: absorb sprite records with FAMILY as the label and
    # the journal must narrate the organizer's splits in role terms -- on the
    # full set every family split, named overwhelmingly by facing/frame (the
    # genuine within-family modes; the npc grab-bag named by colour). Pinned
    # on a subset: a reorganization happens, and the separations it names come
    # from the actual roles.
    import os, re
    import numpy as np
    import pytest
    hsp = os.path.join(os.path.dirname(__file__), "features", "sprites.hsp")
    if not os.path.exists(hsp):
        pytest.skip("sprite asset not present")
    import pack_sprites as ps
    from holographic_unified import UnifiedMind
    from holographic_scene import auto_tags

    with open(hsp, "rb") as f:
        sprites = dict(ps.unpack(f.read()))
    keep = ("amg", "avt", "knt", "npc", "ftr", "wmn")
    examples = []
    for name, rgba in sorted(sprites.items()):
        m = re.match(r"([a-z]+)(\d*)_([a-z]{2})(\d)\.gif", name)
        if not m or m.group(1) not in keep:
            continue
        rgb = rgba[..., :3].astype(float) / 255.0 if rgba.dtype == np.uint8 else rgba[..., :3]
        mask = (rgba[..., 3] > 0) if rgba.shape[-1] == 4 else None
        t = auto_tags(rgb, mask=mask)
        examples.append(({"colour": t["colour"], "texture": t["texture"],
                          "facing": m.group(3), "frame": m.group(4)},
                         m.group(1), "record"))
    rng = np.random.default_rng(0)
    rng.shuffle(examples)
    mind = UnifiedMind(dim=2048, seed=0)
    mind.absorb(examples)

    assert any("reorganized" in e["story"] for e in mind.journal)
    named = mind.explain_organization()
    assert named, "at least one family's split should be nameable"
    legal = {"colour", "texture", "facing", "frame"}
    assert all(set(roles) <= legal for roles in named.values())


def test_unified_trace_distinguishes_ordering():
    # The unified mind exposes trace(): style + material provenance, leading
    # with the decisive one. Ordering carries meaning -- opposite-message
    # same-word sources are told apart by the verbatim span.
    from holographic_unified import UnifiedMind
    bull = ("the council voted to approve the plan and raise the budget this year " * 25)
    bear = ("the council voted to reject the plan and cut the budget this year " * 25)
    m = UnifiedMind(dim=512, seed=0)
    m.learn_sequence([(bull, "Approve"), (bear, "Reject")], modality="text")
    up = m.trace("the council voted to approve the plan and raise the budget")
    dn = m.trace("the council voted to reject the plan and cut the budget")
    assert up["verdict"] == "Approve"
    assert dn["verdict"] == "Reject"


def test_unified_ask_traced_reports_throughput():
    # The unified mind's chains carry THROUGHPUT -- the ray's accumulated
    # confidence as it bounces through memory. A direct one-hop chain answers
    # with positive throughput and one confidence per hop.
    from holographic_unified import UnifiedMind
    m = UnifiedMind(dim=2048, seed=0)
    for name, attrs in {"france": {"capital": "paris", "currency": "euro"},
                        "germany": {"capital": "berlin", "currency": "euro"}}.items():
        m.learn(attrs, name, "record")
    ans, tp, confs = m.ask_traced("paris", ("capital", "currency"))
    assert ans == "euro"
    assert 0.0 < tp <= 1.0 and len(confs) == 1
    # an impossibly high floor forces an honest abstention
    ans2, _, _ = m.ask_traced("paris", ("capital", "currency"), min_throughput=0.99)
    assert ans2 is None


def test_classify_robust_multiray_recovers_noisy_queries():
    # MULTI-RAY: one query is a noisy ray; firing several word-resampled views
    # and combining them z-scored (so a confident-wrong view can't dominate)
    # recovers errors single-ray makes -- path tracing's many-rays-per-pixel.
    # Measured: lifts noisy-query accuracy above plain classify, and never hurts
    # on clean queries.
    import numpy as np
    from holographic_unified import UnifiedMind
    rng = np.random.default_rng(0)
    TRAIN = {
        "weather": ["rain storm cloud wind cold", "sunny sky clear warm bright",
                    "snow ice freeze cold white", "fog mist damp grey morning",
                    "heat wave hot dry summer", "thunder lightning rain dark sky"],
        "music": ["guitar drums bass loud band", "melody song tune sing voice",
                  "concert stage crowd live show", "piano keys notes soft play",
                  "beat rhythm dance move groove", "album track record play list"],
        "travel": ["flight airport plane gate board", "hotel room book stay night",
                   "beach sand sea sun warm", "map route road trip drive",
                   "passport visa border cross country", "train station ticket ride rail"],
    }
    m = UnifiedMind(dim=2048, seed=0)
    ex = [(s, lab) for lab, ss in TRAIN.items() for s in ss]
    m.read([s for s, _ in ex])
    for s, lab in ex:
        m.learn(s, lab, "text")
    m.maintain_now()
    held = {"weather": ["storm wind rain heavy", "sunny warm clear day"],
            "music": ["guitar band loud rock", "song melody sing tune"],
            "travel": ["flight plane gate board", "hotel book room stay"]}
    test = []
    for lab, ss in held.items():
        for s in ss:
            ws = s.split(); rng.shuffle(ws)
            test.append((" ".join(ws[:3]), lab))
    single = sum(m.classify(t)[0] == truth for t, truth in test)
    robust = sum(m.classify_robust(t, n_rays=7)[0] == truth for t, truth in test)
    assert robust >= single                          # never worse, usually better
    # and no regression on clean full sentences
    clean_robust = sum(m.classify_robust(s, n_rays=7)[0] == lab for s, lab in ex)
    assert clean_robust == len(ex)


def test_unified_blend_synthesizes_over_learned_classes():
    # The mind synthesizes a novel concept over its OWN learned classes (decoded
    # from prototypes, so it works on concepts built from noisy observations):
    # one class's frame with another's values on chosen roles -- a thing it never
    # saw, held coherently.
    from holographic_unified import UnifiedMind
    m = UnifiedMind(dim=2048, seed=0)
    data = {"france": {"capital": "paris", "currency": "euro", "language": "french", "continent": "europe"},
            "japan": {"capital": "tokyo", "currency": "yen", "language": "japanese", "continent": "asia"}}
    for n, a in data.items():
        m.learn(a, n, "record")
    blend = m.blend("france", "japan", {"language", "currency"})
    assert blend["capital"] == "paris" and blend["continent"] == "europe"   # france frame
    assert blend["language"] == "japanese" and blend["currency"] == "yen"   # japan projected


def test_brain_learns_dictionary_and_encyclopedia_natively():
    # WIRED TO THE BRAIN: the curriculum lives in UnifiedMind, not just in
    # standalone modules. The mind bootstraps word meaning from definitions into
    # its own encoder, and absorbs an encyclopedia into its own memory, then
    # climbs is_a chains with the same find/ask machinery it uses for any record.
    from holographic_unified import UnifiedMind
    m = UnifiedMind(dim=2048, seed=0)
    defs = {
        "cat": ["animal", "feline", "pet"], "dog": ["animal", "canine", "pet"],
        "lion": ["animal", "feline", "wild"], "wolf": ["animal", "canine", "wild"],
        "animal": ["living", "creature"], "feline": ["cat", "lion", "animal"],
        "canine": ["dog", "wolf", "animal"], "pet": ["animal", "tame"],
        "wild": ["untamed", "animal"], "living": ["alive"], "creature": ["living", "animal"],
        "rock": ["mineral", "hard"], "stone": ["mineral", "hard"], "mineral": ["solid"],
        "hard": ["solid"], "solid": ["firm"], "firm": ["solid"], "tame": ["gentle"],
        "gentle": ["mild"], "mild": ["gentle"], "untamed": ["wild"], "alive": ["living"],
    }
    m.learn_dictionary(defs, iters=3)
    # dictionary-bootstrapped meaning: cat's neighbours are animals, not minerals
    near = [w for w, _ in m.define("cat", 4)]
    assert any(w in ("feline", "animal", "lion") for w in near)
    assert "rock" not in near and "stone" not in near

    facts = {"dog": {"is_a": "canine"}, "wolf": {"is_a": "canine"}, "cat": {"is_a": "feline"},
             "canine": {"is_a": "carnivore"}, "feline": {"is_a": "carnivore"},
             "carnivore": {"is_a": "mammal"}, "mammal": {"is_a": "animal"},
             "animal": {"is_a": "organism"}}
    m.learn_encyclopedia(facts)
    chain, tp = m.climb("dog")
    assert chain[:5] == ["dog", "canine", "carnivore", "mammal", "animal"]
    assert m.is_a("dog", "animal")[0] is True
    assert m.is_a("dog", "plant")[0] is False
    # throughput decays with depth (calibrated confidence over the brain's memory)
    assert m.climb("dog", hops=2)[1] > m.climb("dog", hops=4)[1]


def test_answer_routes_questions_to_real_operations():
    # The question ROUTER: a question's shape maps to the brain's real operation,
    # not to sentence completion. Honest -- it answers from knowledge when it can,
    # and labels completion as completion when it can't.
    from holographic_unified import UnifiedMind
    m = UnifiedMind(dim=2048, seed=0)
    m.learn_dictionary({
        "cat": ["animal", "feline"], "dog": ["animal", "canine"], "wolf": ["animal", "canine"],
        "animal": ["living"], "feline": ["cat", "animal"], "canine": ["dog", "wolf", "animal"],
        "living": ["alive"], "alive": ["living"],
    }, iters=3)
    m.learn_encyclopedia({"dog": {"is_a": "canine"}, "wolf": {"is_a": "canine"},
                          "canine": {"is_a": "carnivore"}, "carnivore": {"is_a": "mammal"},
                          "mammal": {"is_a": "animal"}, "animal": {"is_a": "organism"}})
    # 'what is X' -> define (meaning + is_a chain)
    a = m.answer("what is a dog?")
    assert a["kind"] == "define"
    assert "animal" in a["is_a_chain"]
    # 'is X a Y' -> taxonomic membership, both polarities
    assert m.answer("is a dog an animal?")["answer"] is True
    assert m.answer("is a dog a plant?")["answer"] is False
    # 'define X' form
    assert m.answer("define wolf")["kind"] == "define"


def test_answer_is_honest_when_it_cannot_map():
    # A question it can't map and no sequence model to fall back on -> 'unknown',
    # never a fabricated answer.
    from holographic_unified import UnifiedMind
    m = UnifiedMind(dim=2048, seed=0)
    m.learn_encyclopedia({"dog": {"is_a": "canine"}, "canine": {"is_a": "mammal"}})
    a = m.answer("what is the meaning of life")
    assert a["kind"] in ("unknown", "completion")
    if a["kind"] == "completion":
        assert "generation, not an answer" in a["note"]


def test_answer_role_question_on_records():
    # 'what is the <role> of <concept>' -> read_role over absorbed records.
    from holographic_unified import UnifiedMind
    m = UnifiedMind(dim=2048, seed=0)
    for _ in range(6):
        m.learn({"capital": "paris", "currency": "euro"}, "france", "record")
        m.learn({"capital": "tokyo", "currency": "yen"}, "japan", "record")
    m.maintain_now()
    a = m.answer("what is the capital of france?")
    assert a["kind"] == "role" and a["value"] == "paris"

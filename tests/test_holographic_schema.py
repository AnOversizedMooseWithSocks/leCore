"""Modality-agnostic schema discovery: the SAME compress-by-merging mechanism tokenizes,
learns structure, lowers bits-per-atom, and generates from learned units -- for any data."""
import numpy as np
from holographic.simulation_and_physics.holographic_schema import to_symbols, from_symbols, Schema, SchemaModel, learn


def test_tokenizers_round_trip_any_modality():
    assert from_symbols(to_symbols("hello world", "text"), "text") == "hello world"
    assert from_symbols(to_symbols(b"\x01\x02\xff", "bytes"), "bytes") == b"\x01\x02\xff"
    nums = [0.1, 0.5, 0.9, 0.3]
    back = from_symbols(to_symbols(nums, "numbers", bins=16, num_range=(0, 1)),
                        "numbers", bins=16, num_range=(0, 1))
    assert all(abs(a - b) < 0.1 for a, b in zip(nums, back))


def test_schema_discovers_chunks_and_compresses_the_stream():
    syms = to_symbols("the cat sat on the mat " * 80, "text")
    sch = Schema(merges=60).learn(syms)
    encoded = sch.encode(syms)
    assert len(encoded) < len(syms) * 0.5            # chunking shrank the stream
    blob = " ".join(sch.emergent(syms, k=8))
    assert any(word in blob for word in ("the", "cat", "mat", "sat"))   # words emerged


def test_first_schema_level_lowers_bits_per_atom():
    syms = to_symbols("the cat sat on the mat " * 120, "text")
    cut = int(len(syms) * 0.8); tr, he = syms[:cut], syms[cut:]
    sch = Schema(merges=120).learn(tr)
    flat = SchemaModel(3).fit(sch.encode(tr, upto=0)).bits_per_atom(sch.encode(he, upto=0), len(he))
    chunked = SchemaModel(3).fit(sch.encode(tr)).bits_per_atom(sch.encode(he), len(he))
    assert chunked < flat                            # the discovered schema compresses better


def test_generation_emits_learned_units():
    sch, model = learn(to_symbols("the cat sat on the mat " * 100, "text"), merges=40, order=3)
    out = from_symbols(model.generate(40, seed=list("the "), rng=np.random.default_rng(0)), "text")
    assert any(w in out for w in ("the", "cat", "mat", "sat"))


def test_cross_level_backoff_beats_the_atom_level():
    from holographic.simulation_and_physics.holographic_schema import HierModel
    syms = to_symbols("the cat sat on the mat " * 150, "text")
    cut = int(len(syms) * 0.8); tr, he = syms[:cut], syms[cut:]
    sch = Schema(merges=200).learn(tr)
    atom_only = HierModel(sch, (0,)).fit(tr).bits_per_atom(he)
    hier = HierModel(sch, (0, 60, 150)).fit(tr).bits_per_atom(he)
    assert 0 < hier <= atom_only            # stacking levels helped on multi-scale data


def test_cross_level_backoff_runs_on_any_modality():
    from holographic.simulation_and_physics.holographic_schema import HierModel
    x = np.linspace(0, 20 * np.pi, 8000)
    syms = to_symbols(np.sin(x), "numbers")
    cut = int(len(syms) * 0.8); tr, he = syms[:cut], syms[cut:]
    sch = Schema(merges=120).learn(tr)
    b = HierModel(sch, (0, 60, 120)).fit(tr).bits_per_atom(he)
    assert 0 < b < 5                        # finite, sensible bits-per-atom for non-text data


def test_schema_router_routes_by_who_compresses_best():
    from holographic.simulation_and_physics.holographic_schema import SchemaRouter
    text = ("the quick brown fox jumps over the lazy dog near the river bank. " * 40).encode()
    nums = bytes(int(v) for v in (np.sin(np.linspace(0, 80 * np.pi, 6000)) * 40 + 120))
    r = SchemaRouter(modality="bytes", cuts=(0, 80, 200))
    r.learn("text", text[:2200]).learn("numeric", nums[:5000])
    assert r.route(text[2200:])[0] == "text"          # prose routes to the prose schema
    assert r.route(nums[5000:])[0] == "numeric"        # the signal routes to the signal schema


def test_compression_gate_primitive_ranks_by_bits_and_accepts_dict_or_pairs():
    from holographic.simulation_and_physics.holographic_schema import compression_gate, SchemaGenerator
    a = SchemaGenerator("text", cuts=(0, 40, 110)).fit("alpha alpha alpha beta gamma " * 40)
    b = SchemaGenerator("text", cuts=(0, 40, 110)).fit("one two three four five six " * 40)
    probe = "alpha alpha beta gamma alpha "
    ranked_dict = compression_gate(probe, {"A": a, "B": b})
    ranked_pairs = compression_gate(probe, [("A", a), ("B", b)])
    assert ranked_dict[0][1] == "A"                      # winner = the expert that understands it
    assert ranked_dict == ranked_pairs                   # same result from dict or pairs
    assert ranked_dict[0][0] <= ranked_dict[1][0]        # ranked ascending by bits


def test_hybrid_gate_reduces_to_compression_then_demotes_a_liar():
    from holographic.simulation_and_physics.holographic_schema import HybridGate, compression_gate
    base = "the cat sat on the mat and the dog ran across the green park today "
    g = HybridGate(modality="text", cuts=(0, 40, 110))
    g.learn("deep", base * 60).learn("shallow", base * 8)   # deep models the domain a touch better
    probe = base * 2
    pure = compression_gate(probe, {k: e["schema"] for k, e in g.experts.items()})[0][1]
    assert g.route(probe) == pure                # with no feedback: identical to compression gate
    liar = g.route(probe)
    for _ in range(12):                          # the best compressor turns out to lie
        g.observe(liar, correct=False)
    assert g.route(probe) != liar                # reward signal demotes it below the honest expert


def test_schema_discovers_code_structure_from_scratch():
    # The destination test: the SAME compress-by-merging primitive must teach itself
    # the structure of a NEW format with no labels. Code is the format, and the
    # corpus is -- recursively -- this very module's own source. Measured at full
    # scale (the whole library, ~500k chars): flat 2.98 -> fractal 2.28 bits/char
    # (23.7% fewer), with `def __init__`, `rng = np.random.default_rng(` and
    # indentation idioms among the emergent chunks. This is the fast version.
    import os
    from holographic.simulation_and_physics.holographic_schema import HierModel
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "holographic", "simulation_and_physics", "holographic_schema.py"),
               encoding="utf-8").read()
    atoms = to_symbols(src, "code")
    cut = int(len(atoms) * 0.9)
    tr, he = atoms[:cut], atoms[cut:]
    sch = Schema(merges=250).learn(tr)
    flat = HierModel(sch, (0,)).fit(tr, order=2).bits_per_atom(he)
    hier = HierModel(sch, (0, 80, 250)).fit(tr, order=2).bits_per_atom(he)
    assert hier < flat                       # discovered schema compresses code better
    blob = " ".join(sch.emergent(tr, k=24, min_atoms=4))
    # real Python syntax emerged from raw characters, with no labels
    assert any(idiom in blob for idiom in ("def ", "self.", "return", "    "))


def test_compression_gate_tells_code_from_prose_when_fairly_fed():
    # Structure discrimination by compression: a pure-code expert and a prose expert
    # each claim their own held-out format. The honest caveat (measured): feed the
    # code expert RAW source -- which is half English docstrings -- and it becomes a
    # better English model than a starved prose expert, so the gate mis-routes.
    # Representative corpora are part of the mechanism, not an optional nicety.
    from holographic.simulation_and_physics.holographic_schema import SchemaGenerator, compression_gate
    code = ("def step(self, action):\n    reward = self.world.step(action)\n"
            "    self.memory.append((self.state, action, reward))\n"
            "    return reward\n\nfor i in range(n):\n    total += vals[i]\n"
            "    if total > cap:\n        break\n") * 30
    prose = ("the river ran quietly past the old mill while the children walked "
             "along the bank and talked about the long summer that lay ahead of "
             "them, full of plans and small adventures. ") * 30
    code_gen = SchemaGenerator("code", cuts=(0, 60, 150)).fit(code[:int(len(code)*0.9)])
    prose_gen = SchemaGenerator("text", cuts=(0, 60, 150)).fit(prose[:int(len(prose)*0.9)])

    assert compression_gate(code[-400:], {"code": code_gen, "prose": prose_gen})[0][1] == "code"
    assert compression_gate(prose[-400:], {"code": code_gen, "prose": prose_gen})[0][1] == "prose"


def test_schema_discovers_image_structure_given_enough_data():
    # The third format: the SAME compress-by-merging primitive on IMAGES -- the
    # project's own 712-sprite set, each pixel an opaque colour-code atom in
    # raster order. The honest shape of this result: the schema is DATA-HUNGRY
    # here. At 60 training sprites the rare chunks starve and the fractal coder
    # LOSES to the flat pixel model (1.91 vs 1.96 bits/pixel, measured); at 150
    # sprites it wins across every split tried (e.g. 1.49 -> 1.30, and 23% fewer
    # bits at merges=400/order=3). Structure exists in the format, but feeding
    # the statistics is part of the mechanism -- the same lesson the code/prose
    # gate taught about representative corpora.
    import os
    import numpy as np
    from tools.pack_sprites import unpack
    from holographic.simulation_and_physics.holographic_schema import HierModel
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "features", "sprites.hsp")
    items = unpack(open(path, "rb").read())

    def atoms_of(idx):
        out = []
        for i in idx:
            a = items[i][1].astype(np.uint32)
            codes = (a[..., 0] << 24) | (a[..., 1] << 16) | (a[..., 2] << 8) | a[..., 3]
            out.extend(int(c) for c in codes.flatten())
        return out

    p = np.random.default_rng(0).permutation(len(items))
    tr, he = atoms_of(p[:150]), atoms_of(p[150:170])
    sch = Schema(merges=200).learn(tr)
    flat = HierModel(sch, (0,)).fit(tr, order=2).bits_per_atom(he)
    hier = HierModel(sch, (0, 80, 200)).fit(tr, order=2).bits_per_atom(he)
    assert hier < flat                  # fed enough sprites, the schema earns its keep


def test_generation_provenance_attributes_held_out_text():
    # PROVENANCE: fit on (text, source) documents and the model records which
    # document taught each context->token transition; attribute(text) ranks the
    # sources by the transitions a passage actually uses. The reliable question
    # is attributing GIVEN text -- measured 70% (4 books) / 92% (5 books) top-1
    # on clean held-out passages; near-uniform on freely-generated low-order
    # text, which honestly blends every source. Pinned with a synthetic corpus
    # of three sharply distinct "styles" so the test is fast and deterministic.
    from holographic.simulation_and_physics.holographic_schema import SchemaGenerator
    # three sources sharing an alphabet (realistic) but with distinct phrasing
    a = ("the cat sat softly on the warm windowsill watching birds " * 80)
    b = ("a dog ran quickly through the muddy field chasing rabbits " * 80)
    c = ("she wrote careful letters every evening beside the candle " * 80)
    g = SchemaGenerator(modality="text").fit([(a, "Cat"), (b, "Dog"), (c, "Letters")])
    # attribute spans long enough to contain the sources' characteristic chunks
    assert g.attribute("the cat sat softly on the warm windowsill watching")[0][0] == "Cat"
    assert g.attribute("a dog ran quickly through the muddy field chasing")[0][0] == "Dog"
    assert g.attribute("she wrote careful letters every evening beside the")[0][0] == "Letters"
    # a plain-string fit records no provenance -> attribute() refuses rather
    # than inventing sources
    plain = SchemaGenerator(modality="text").fit(a)
    assert plain.sources is None
    import pytest
    with pytest.raises(ValueError):
        plain.attribute("alpha bravo")


def test_provenance_localizes_spliced_segments():
    # The traced generator returns per-emission attribution, so a spliced
    # passage can be localized window-by-window. Measured 8/9 windows on real
    # books; here a clean splice of two sharply distinct styles.
    from holographic.simulation_and_physics.holographic_schema import SchemaGenerator
    x = ("the painter mixed crimson and azure on her favourite palette " * 60)
    y = ("the sailor counted barrels of salted fish below the deck " * 60)
    g = SchemaGenerator(modality="text").fit([(x, "Painter"), (y, "Sailor")])
    assert g.attribute("the painter mixed crimson and azure on her")[0][0] == "Painter"
    assert g.attribute("the sailor counted barrels of salted fish")[0][0] == "Sailor"


def test_coherent_attribution_weights_distinctive_evidence():
    # The user's principle: a passage usually comes from ONE source, so a
    # transition only one source taught (unique) should dominate a transition
    # all sources share. Coherent mode weights each transition's vote by its
    # specificity (1 / number of sources that taught it). Measured: lifts
    # confidence on ambiguous short passages and is a wash where evidence
    # already saturates -- and a sequential running-prior on top measured a
    # wash-to-negative (kept out; specificity already captures the insight).
    # Pinned at the mechanism level so it is deterministic.
    from holographic.simulation_and_physics.holographic_schema import SchemaGenerator
    shared = "the garden was calm and the morning air felt soft and cool "
    a = (shared + "she walked the gravel path admiring the roses there ") * 50
    b = (shared + "he read the morning paper beside the tall window there ") * 50
    c = (shared + "the chef seared a fillet of halibut for the late guests ") * 50
    g = SchemaGenerator(modality="text").fit([(a, "Roses"), (b, "Paper"), (c, "Chef")])
    # a passage of shared text plus ONE distinctive Chef phrase: coherent must
    # rank Chef first, and at least as confidently as independent voting
    probe = "the garden was calm and the chef seared a fillet of halibut"
    coh = g.attribute(probe, coherent=True)
    ind = g.attribute(probe, coherent=False)
    assert coh[0][0] == "Chef"
    chef_coh = dict(coh)["Chef"]
    chef_ind = dict(ind)["Chef"]
    assert chef_coh >= chef_ind                      # distinctive evidence not diluted


def test_alignment_distinguishes_opposite_message_same_words():
    # THE USER'S INSIGHT: meaning is in the ordering. Two sources sharing every
    # word in opposite arrangement (a bullish vs bearish thesis differing only
    # in 'up'/'down') fool the bag-of-transitions (it attributes the bearish
    # sentence to the bull source, the shared words swamping the one different
    # one) -- but SEQUENCE ALIGNMENT pins each to the right source, because the
    # longest contiguous run only appears verbatim in the matching one. This is
    # the nature-solves-it move: provenance by longest distinctive span, like
    # genome alignment, not by shared-token composition.
    from holographic.simulation_and_physics.holographic_schema import SchemaGenerator
    bull = ("market analysts say the price of bitcoin will go up by as much as "
            "thirty percent by the end of this quarter as demand keeps rising ") * 25
    bear = ("market analysts say the price of bitcoin will go down by as much as "
            "thirty percent by the end of this quarter as pressure keeps rising ") * 25
    g = SchemaGenerator(modality="text").fit([(bull, "Bull"), (bear, "Bear")])

    up = "market analysts say the price of bitcoin will go up by as much as thirty percent"
    dn = "market analysts say the price of bitcoin will go down by as much as thirty percent"
    # alignment / trace get BOTH right
    assert g.trace(up)["verdict"] == "Bull"
    assert g.trace(dn)["verdict"] == "Bear"
    assert g.trace(dn)["basis"] == "material"        # a verbatim span decided it
    # and the material ranking is correct even where the style bag is fooled
    assert g.align(dn)[0][0][0] == "Bear"


def test_trace_falls_back_to_style_without_a_distinctive_span():
    # When there is no long verbatim span (paraphrase / original-but-in-style),
    # trace() reports basis='style' and leads with the transition bag -- the two
    # methods answer different questions and trace picks the right one per text.
    from holographic.simulation_and_physics.holographic_schema import SchemaGenerator
    a = ("the meadow stretched green beneath a wide and gentle morning sky " * 40)
    b = ("machinery clanked in the foundry as iron poured white and molten " * 40)
    g = SchemaGenerator(modality="text").fit([(a, "Pastoral"), (b, "Industrial")])
    # a never-seen sentence in the pastoral REGISTER (shared short words only)
    t = g.trace("the field was calm under the soft sky")
    assert t["basis"] == "style"
    assert t["verdict"] in ("Pastoral", "Industrial")  # style call, span not decisive

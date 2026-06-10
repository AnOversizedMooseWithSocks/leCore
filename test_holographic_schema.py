"""Modality-agnostic schema discovery: the SAME compress-by-merging mechanism tokenizes,
learns structure, lowers bits-per-atom, and generates from learned units -- for any data."""
import numpy as np
from holographic_schema import to_symbols, from_symbols, Schema, SchemaModel, learn


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
    from holographic_schema import HierModel
    syms = to_symbols("the cat sat on the mat " * 150, "text")
    cut = int(len(syms) * 0.8); tr, he = syms[:cut], syms[cut:]
    sch = Schema(merges=200).learn(tr)
    atom_only = HierModel(sch, (0,)).fit(tr).bits_per_atom(he)
    hier = HierModel(sch, (0, 60, 150)).fit(tr).bits_per_atom(he)
    assert 0 < hier <= atom_only            # stacking levels helped on multi-scale data


def test_cross_level_backoff_runs_on_any_modality():
    from holographic_schema import HierModel
    x = np.linspace(0, 20 * np.pi, 8000)
    syms = to_symbols(np.sin(x), "numbers")
    cut = int(len(syms) * 0.8); tr, he = syms[:cut], syms[cut:]
    sch = Schema(merges=120).learn(tr)
    b = HierModel(sch, (0, 60, 120)).fit(tr).bits_per_atom(he)
    assert 0 < b < 5                        # finite, sensible bits-per-atom for non-text data


def test_schema_router_routes_by_who_compresses_best():
    from holographic_schema import SchemaRouter
    text = ("the quick brown fox jumps over the lazy dog near the river bank. " * 40).encode()
    nums = bytes(int(v) for v in (np.sin(np.linspace(0, 80 * np.pi, 6000)) * 40 + 120))
    r = SchemaRouter(modality="bytes", cuts=(0, 80, 200))
    r.learn("text", text[:2200]).learn("numeric", nums[:5000])
    assert r.route(text[2200:])[0] == "text"          # prose routes to the prose schema
    assert r.route(nums[5000:])[0] == "numeric"        # the signal routes to the signal schema


def test_compression_gate_primitive_ranks_by_bits_and_accepts_dict_or_pairs():
    from holographic_schema import compression_gate, SchemaGenerator
    a = SchemaGenerator("text", cuts=(0, 40, 110)).fit("alpha alpha alpha beta gamma " * 40)
    b = SchemaGenerator("text", cuts=(0, 40, 110)).fit("one two three four five six " * 40)
    probe = "alpha alpha beta gamma alpha "
    ranked_dict = compression_gate(probe, {"A": a, "B": b})
    ranked_pairs = compression_gate(probe, [("A", a), ("B", b)])
    assert ranked_dict[0][1] == "A"                      # winner = the expert that understands it
    assert ranked_dict == ranked_pairs                   # same result from dict or pairs
    assert ranked_dict[0][0] <= ranked_dict[1][0]        # ranked ascending by bits


def test_hybrid_gate_reduces_to_compression_then_demotes_a_liar():
    from holographic_schema import HybridGate, compression_gate
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

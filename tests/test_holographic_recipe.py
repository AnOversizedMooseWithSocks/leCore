"""Generative recipe-store: constructed structure serialises to its build-graph and replays bit-exact."""
import numpy as np
from holographic.misc.holographic_recipe import StructureRecipe


def _scene(seed=7):
    r = StructureRecipe(dim=512, seed=seed)
    ag, ac, ob = r.atom("agent", True), r.atom("action", True), r.atom("object", True)
    al, ru, ba = r.atom("alice"), r.atom("runs"), r.atom("ball")
    r.mark_output(r.bundle([r.bind(ag, al), r.bind(ac, ru), r.bind(ob, ba)]))
    return r


def test_replay_is_bit_exact_through_save_load():
    r = _scene()
    original = r.outputs()[0]
    r.save("/tmp/_t.rcp")
    rebuilt = StructureRecipe.load("/tmp/_t.rcp").outputs()[0]
    assert np.max(np.abs(original - rebuilt)) == 0.0     # no noise to store -> exact


def test_constructed_structure_compresses_losslessly():
    r = StructureRecipe(dim=512, seed=11)
    for i in range(500):
        r.mark_output(r.atom(f"tok_{i}"))
    assert r.compression_ratio() > 30                    # each atom: a small op, not a 512-float vector


def test_raw_data_gets_no_compression_kept_negative():
    rng = np.random.default_rng(0)
    r = StructureRecipe(dim=512, seed=3)
    for _ in range(100):
        r.mark_output(r.raw(rng.standard_normal(512)))
    assert 0.8 < r.compression_ratio() < 1.2             # non-constructed -> ~1x, no win and no harm


def test_deep_nesting_recovers_exactly():
    r = StructureRecipe(dim=1024, seed=5)
    L, R = r.atom("L", True), r.atom("R", True)
    def build(d):
        return r.atom("leaf") if d == 0 else r.bundle([r.bind(L, build(d-1)), r.bind(R, build(d-1))])
    r.mark_output(build(8))
    a = r.outputs()[0]
    b = StructureRecipe.from_dict(r.to_dict()).outputs()[0]
    assert np.max(np.abs(a - b)) == 0.0                  # exact at depth 8 (no capacity cliff)


def test_all_ops_survive_roundtrip():
    r = StructureRecipe(dim=256, seed=1)
    a, b = r.atom("a"), r.atom("b", True)
    r.mark_output(r.normalize(r.permute(r.bundle([r.bind(b, a), a]), 3)))
    before = r.outputs()[0]
    after = StructureRecipe.from_dict(r.to_dict()).outputs()[0]
    assert np.max(np.abs(before - after)) == 0.0


def test_atom_range_macro_reaches_large_ratio_bit_exact():
    from holographic.agents_and_reasoning.holographic_ai import derived_atom
    r = StructureRecipe(dim=512, seed=11)
    r.atom_range("tok_", 2000)                           # 2000 atoms as ONE op
    built = r.outputs()                                  # build once
    assert len(built) == 2000
    ref = [derived_atom(11, f"tok_{i}", 512) for i in range(2000)]
    assert max(np.max(np.abs(built[i] - ref[i])) for i in range(2000)) == 0.0   # bit-exact
    assert r.compression_ratio() > 10000                 # macro collapses N ops to 1 -> huge ratio


def test_repeat_template_matches_manual_build():
    from holographic.agents_and_reasoning.holographic_ai import derived_atom, bundle, permute
    seq = StructureRecipe(dim=256, seed=4)
    items = seq.repeat(15, [("atom", "item_{i}", False), ("permute", 0, "i")])
    seq.mark_output(seq.bundle(items))
    auto = seq.outputs()[0]
    manual = bundle([permute(derived_atom(4, f"item_{i}", 256), i) for i in range(15)])
    assert np.max(np.abs(auto - manual)) == 0.0


def test_macro_survives_save_load():
    r = StructureRecipe(dim=128, seed=2)
    r.atom_range("a_", 50)
    before = r.outputs()
    after = StructureRecipe.load_from_dict_roundtrip(r) if hasattr(StructureRecipe, "load_from_dict_roundtrip") \
        else StructureRecipe.from_dict(r.to_dict())
    after = after.outputs()
    assert len(after) == 50 and max(np.max(np.abs(before[i] - after[i])) for i in range(50)) == 0.0

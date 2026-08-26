"""Frozen core + persistence (G3): the kernel import surface is stable, and a trained
mind round-trips through save/load to an identical object. These are the tests that let
build-on-top code rely on the core. All hermetic -- no corpora needed.
"""
import os
import tempfile

import numpy as np
import pytest

import holographic.misc.holographic_core as core
from holographic.misc.holographic_core import save, load, to_state, from_state, STATE_VERSION
from holographic.agents_and_reasoning.holographic_ai import Vocabulary
from holographic.misc.holographic_creature import HolographicMind
from holographic.misc.holographic_tree import HoloForest


def _tmp(name):
    return os.path.join(tempfile.gettempdir(), name)


# ------------------------------ the kernel surface -------------------------

def test_core_reexports_the_kernel_with_stable_names():
    # Build-on-top code imports these from core; they must exist and be the same objects
    # as in holographic_ai (an extraction, not a fork).
    import holographic.agents_and_reasoning.holographic_ai as ai
    for name in ("random_vector", "unitary_vector", "bind", "unbind", "bundle",
                 "permute", "cosine", "slerp", "Vocabulary"):
        assert getattr(core, name) is getattr(ai, name)
    assert callable(core.cleanup)
    assert core.CORE_VERSION >= 1 and core.STATE_VERSION >= 1


def test_core_cleanup_matches_vocabulary_cleanup():
    v = Vocabulary(256, seed=0)
    for nm in ("a", "b", "c"):
        v.get(nm)
    noisy = v.get("b") + 0.2 * core.random_vector(256, np.random.default_rng(1))
    assert core.cleanup(noisy, v)[0] == v.cleanup(noisy)[0]


# ------------------------------ Vocabulary --------------------------------

def test_vocabulary_round_trips_exactly():
    v = Vocabulary(256, seed=1, unitary=True)
    for nm in ("cat", "dog", "fish", "bird", "tree"):
        v.get(nm)
    save(v, _tmp("voc_test.npz"), compress=False)      # bit-exact round-trip
    back = load(_tmp("voc_test.npz"))
    assert set(back.vectors) == set(v.vectors)
    assert back.unitary == v.unitary and back.dim == v.dim
    for nm in v.vectors:
        assert np.allclose(back.vectors[nm], v.vectors[nm])
    os.remove(_tmp("voc_test.npz"))


# ------------------------------ HolographicMind ---------------------------

def _trained_brain(consolidate=False):
    rng = np.random.default_rng(0)
    m = HolographicMind(dim=48, actions=["N", "S", "E", "W"], seed=0, capacity=8)
    for _ in range(500):
        m.remember([rng.standard_normal(48)], [int(rng.integers(4))], [rng.standard_normal()])
    if consolidate:
        m.consolidate(energy=0.95)
        for _ in range(100):
            m.remember([rng.standard_normal(48)], [int(rng.integers(4))], [rng.standard_normal()])
    return m


def test_trained_brain_round_trips_and_decides_identically():
    m = _trained_brain()
    save(m, _tmp("brain_test.npz"))
    back = load(_tmp("brain_test.npz"))
    rng = np.random.default_rng(99)
    for _ in range(50):
        p = rng.standard_normal(48)
        assert np.allclose([m.value(p, a)[0] for a in range(4)],
                           [back.value(p, a)[0] for a in range(4)])
    for a in range(4):
        assert len({len(back._unit[a]), len(back._cnt[a]),
                    len(back._ret[a]), len(back._sum[a])}) == 1   # banks lockstep
    os.remove(_tmp("brain_test.npz"))


def test_consolidated_brain_round_trips_with_its_basis():
    # The harder case: a brain that has projected into a low-rank basis must reload with
    # that basis intact and keep deciding identically.
    m = _trained_brain(consolidate=True)
    assert m._basis is not None
    back = from_state(to_state(m))
    assert back._basis is not None and back._basis.shape == m._basis.shape
    rng = np.random.default_rng(7)
    d = m._basis.shape[1]                                 # decide in the projected space
    for _ in range(40):
        p = rng.standard_normal(d)
        assert np.allclose([m.value(p, a)[0] for a in range(4)],
                           [back.value(p, a)[0] for a in range(4)])


# ------------------------------ HoloForest --------------------------------

def test_forest_round_trips_and_recalls_identically():
    rng = np.random.default_rng(3)
    items = rng.standard_normal((800, 64)); items /= np.linalg.norm(items, axis=1, keepdims=True)
    F = HoloForest(64, n_trees=4, leaf_size=32, seed=7).build(items)
    save(F, _tmp("forest_test.npz"))
    back = load(_tmp("forest_test.npz"))
    assert np.allclose(back.items, F.items)
    for _ in range(30):
        q = items[int(rng.integers(800))] + 0.1 * rng.standard_normal(64)
        q /= np.linalg.norm(q)
        assert F.recall(q) == back.recall(q)            # seed-derived trees rebuild identically
    os.remove(_tmp("forest_test.npz"))


# ------------------------------ SelfOrganizingMind ------------------------

def test_self_organizing_mind_round_trips_classifications():
    # The biggest stateful object (UnifiedMind's memory): its learned encoder + prototype
    # bank must reload deciding identically -- even on never-seen words, because the
    # vocab's rng state is persisted so post-reload mints match a never-saved run.
    from holographic.scene_and_pipeline.holographic_organizer import SelfOrganizingMind
    m = SelfOrganizingMind(dim=256, seed=0)
    for line in ("the cat sat on the mat", "money market rates rose today",
                 "the sun is bright in the sky"):
        m.encoder._text.learn(line.split())
    for x, lab in [("the cat sat", "animal"), ("money market", "finance"),
                   ("sun bright", "nature")] * 5:
        m.observe(x, lab)
    m.reorganize()
    save(m, _tmp("som_test.npz"), compress=False)      # bit-exact: classifications must match
    back = load(_tmp("som_test.npz"))
    assert back.live.size() == m.live.size()
    probes = ["the cat", "money rates", "sun today", "dog park", "stocks at noon",
              "entirely unseen tokens"]
    for q in probes:
        assert m.classify(q) == back.classify(q)        # identical, incl. unknown-word probes
    os.remove(_tmp("som_test.npz"))


def test_float32_compression_halves_the_file_and_keeps_behaviour():
    # Default save() stores float32 -> ~half the bytes of compress=False, and on realistic
    # probes the reloaded mind classifies the same (a decision only flips on an exact tie).
    from holographic.scene_and_pipeline.holographic_organizer import SelfOrganizingMind
    m = SelfOrganizingMind(dim=512, seed=0)
    for line in ("the cat sat on the mat", "money market rates rose today",
                 "stocks fell sharply at noon"):
        m.encoder._text.learn(line.split())
    for x, lab in [("the cat", "animal"), ("money market", "finance"),
                   ("stocks fell", "finance")] * 6:
        m.observe(x, lab)
    m.reorganize()
    save(m, _tmp("som_c.npz"), compress=True)
    save(m, _tmp("som_u.npz"), compress=False)
    c_size = os.path.getsize(_tmp("som_c.npz"))
    u_size = os.path.getsize(_tmp("som_u.npz"))
    assert c_size < 0.6 * u_size                        # roughly half the bytes
    back = load(_tmp("som_c.npz"))
    import numpy as _np
    rng = _np.random.default_rng(0)
    vocab = ["the", "cat", "money", "market", "stocks", "fell", "rose", "today"]
    agree = 0
    for _ in range(60):
        q = " ".join(rng.choice(vocab, size=3))
        agree += (m.classify(q)[0] == back.classify(q)[0])
    assert agree >= 58                                   # behaviour preserved on real probes
    os.remove(_tmp("som_c.npz")); os.remove(_tmp("som_u.npz"))


# ------------------------------ versioning --------------------------------

def test_load_rejects_a_version_mismatch_loudly():
    v = Vocabulary(64, seed=0); v.get("x")
    st = to_state(v)
    st["state_version"] = STATE_VERSION + 1
    with pytest.raises(ValueError):
        from_state(st)                                   # mismatched format -> loud failure


def test_load_rejects_an_unstamped_state():
    st = {"kind": "Vocabulary", "dim": 64, "names": [], "vectors": np.zeros((0, 64))}
    with pytest.raises(ValueError):
        from_state(st)                                   # no version stamp -> refuse


def test_to_state_rejects_an_object_without_persistence():
    with pytest.raises(TypeError):
        to_state(object())                               # no to_state() -> clear error


def test_int8_quantization_compresses_further_and_keeps_classifications():
    # The scalar-quantisation trick from vector databases, measured on this substrate:
    # int8 save is ~2-3x smaller than float32 (~5x+ vs float64) and -- because prototypes
    # are near-orthogonal at the working dimension -- the nearest-neighbour argmax is
    # unchanged, so classifications survive. Opt-in; float32 stays the default.
    import numpy as _np
    from holographic.scene_and_pipeline.holographic_organizer import SelfOrganizingMind, _multimodal_world
    enc, sample, K, _ = _multimodal_world(seed=0, modes=2)
    m = SelfOrganizingMind(dim=512, seed=0)
    rng = _np.random.default_rng(7)
    for _ in range(400):
        c = int(rng.integers(K)); m.observe(sample(c), c, "vector")
    m.reorganize()
    save(m, _tmp("q_f32.npz"), compress=True)
    save(m, _tmp("q_i8.npz"), quant="int8")
    f32 = os.path.getsize(_tmp("q_f32.npz"))
    i8 = os.path.getsize(_tmp("q_i8.npz"))
    assert i8 < 0.6 * f32                                  # meaningfully smaller than float32
    back = load(_tmp("q_i8.npz"))
    probes = [(sample(c), c) for c in (int(rng.integers(K)) for _ in range(120))]
    agree = sum(m.classify(x, "vector")[0] == back.classify(x, "vector")[0] for x, _ in probes)
    assert agree >= 118                                    # classifications preserved through int8
    os.remove(_tmp("q_f32.npz")); os.remove(_tmp("q_i8.npz"))


def test_auto_quant_is_dynamic_matches_float32_and_is_decision_safe():
    # Dynamic quantisation: each float array gets the coarsest DECISION-SAFE precision its
    # own structure supports -- a mix of int8 (where separation proves it lossless) and
    # float32 (tiny or marginal arrays). It compresses well below float32, classifications
    # match float32 exactly, and continued learning after reload is uncorrupted.
    import json
    import numpy as _np
    from holographic.scene_and_pipeline.holographic_organizer import SelfOrganizingMind, _multimodal_world
    enc, sample, K, _ = _multimodal_world(seed=0, modes=2)
    m = SelfOrganizingMind(dim=512, seed=0)
    rng = _np.random.default_rng(7)
    for _ in range(500):
        c = int(rng.integers(K)); m.observe(sample(c), c, "vector")
    m.reorganize()
    probes = [(sample(c), c) for c in (int(rng.integers(K)) for _ in range(200))]

    save(m, _tmp("auto.npz"), quant="auto")

    # the adaptive selector actually engages: more than one precision is used, and only
    # the decision-safe levels (never 1-bit binary, which is not generically safe)
    with _np.load(_tmp("auto.npz")) as z:
        qs = json.loads(bytes(z["__qspec__"]).decode("utf-8"))
    kinds = {v["k"] for v in qs.values()}
    assert len(kinds) >= 2                                  # genuinely dynamic, not one fixed level
    assert kinds <= {"int8", "f32"}                         # never auto-selects unsafe binary

    # compresses well under float32
    save(m, _tmp("f32.npz"), compress=True)
    assert os.path.getsize(_tmp("auto.npz")) < 0.6 * os.path.getsize(_tmp("f32.npz"))

    # classifications match float32 exactly
    back = load(_tmp("auto.npz"))
    agree = sum(m.classify(x, "vector")[0] == back.classify(x, "vector")[0] for x, _ in probes)
    assert agree == len(probes)

    # continued learning after reload is uncorrupted
    for _ in range(200):
        c = int(rng.integers(K)); back.observe(sample(c), c, "vector")
    acc = _np.mean([back.classify(x, "vector")[0] == c for x, c in probes])
    assert acc >= 0.95
    os.remove(_tmp("auto.npz")); os.remove(_tmp("f32.npz"))


def test_auto_quant_is_decision_safe_for_the_value_brain():
    # The cross-stack guarantee: auto must not change decisions on the creature VALUE brain,
    # whose value readback is far more sensitive than a classification argmax. (An earlier
    # binary level flipped 62/200 of its actions; the decision-safe int8/float32 ladder
    # flips at most tie-level.)
    import numpy as _np
    from holographic.misc.holographic_creature import HolographicMind
    b = HolographicMind(dim=256, actions=["N", "S", "E", "W"], seed=0, capacity=32)
    rng = _np.random.default_rng(1)
    probes = [rng.standard_normal(256) for _ in range(200)]
    for _ in range(1200):
        b.remember([rng.standard_normal(256)], [int(rng.integers(4))], [rng.standard_normal()])
    decide = lambda mind, p: int(_np.argmax([mind.value(p, a)[0] for a in range(4)]))
    orig = [decide(b, p) for p in probes]
    save(b, _tmp("brain_auto.npz"), quant="auto")
    back = load(_tmp("brain_auto.npz"))
    flips = sum(decide(back, p) != orig[i] for i, p in enumerate(probes))
    assert flips <= 2                                       # tie-level, not the binary-era 62
    os.remove(_tmp("brain_auto.npz"))


def test_auto_quant_size_floor_keeps_tiny_arrays_float():
    # The 'size' half of the rule: tiny arrays stay float (per-array overhead not worth it).
    import numpy as _np
    small = _np.random.default_rng(1).standard_normal((2, 16))     # 32 elements, below floor
    assert core._auto_quant_kind(small) == "f32"


def test_rd_quant_save_level_is_safe_and_preserves_decisions():
    # quant="rd" (rate-distortion code): on a normal object it falls back to int8 where there is no
    # low-rank structure, so it round-trips and preserves classifications -- the wiring is non-breaking.
    import numpy as _np
    from holographic.scene_and_pipeline.holographic_organizer import SelfOrganizingMind, _multimodal_world
    enc, sample, K, _ = _multimodal_world(seed=0, modes=2)
    m = SelfOrganizingMind(dim=512, seed=0)
    rng = _np.random.default_rng(7)
    for _ in range(400):
        c = int(rng.integers(K)); m.observe(sample(c), c, "vector")
    m.reorganize()
    save(m, _tmp("rd_test.npz"), quant="rd")
    back = load(_tmp("rd_test.npz"))
    probes = [(sample(c), c) for c in (int(rng.integers(K)) for _ in range(120))]
    agree = sum(m.classify(x, "vector")[0] == back.classify(x, "vector")[0] for x, _ in probes)
    assert agree >= 118
    os.remove(_tmp("rd_test.npz"))


def test_rd_quant_activates_and_shrinks_low_rank_state():
    # when a 2D float array IS low-rank, quant="rd" activates and the file is smaller than int8.
    import numpy as _np
    from holographic.misc.holographic_core import save as _save, load as _load
    from holographic.agents_and_reasoning.holographic_ai import random_vector, bundle, cosine
    # a Vocabulary-like object isn't low-rank, so build the low-rank matrix and round-trip via the codec
    from holographic.misc.holographic_ratedistortion import geometry_preserving_code, pack_code, unpack_code, reconstruct, bits_per_vector
    rng = _np.random.default_rng(0); D = 256; Kn = 16
    senses = [random_vector(D, rng) for _ in range(Kn)]
    X = _np.array([bundle([senses[j] for j in rng.choice(Kn, size=5, replace=False)]) for _ in range(400)])
    X /= _np.linalg.norm(X, axis=1, keepdims=True)
    code = geometry_preserving_code(X, target_cos=0.999)
    assert bits_per_vector(code) < 8 * D                       # rd activates (beats int8) on low-rank data
    back = reconstruct(unpack_code(pack_code(code)))           # full pack->unpack->reconstruct path
    cos = _np.mean([cosine(X[i], back[i]) for i in range(len(X))])
    assert cos >= 0.999

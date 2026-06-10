"""Tests for holographic_organizer: reorganizing a multi-modal store into
sub-prototypes, discovering the modes, an atomic non-destructive swap, and merge."""

import numpy as np

from holographic_organizer import (SelfOrganizingMind, SplitExpert, MergeExpert,
                                    SubPrototypeMemory, _multimodal_world)


def _run_stream(mind, sample, K, n, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        c = int(rng.integers(K))
        mind.observe(sample(c), c, "vector")


def test_reorganize_beats_naive_on_multimodal():
    enc, sample, K, _ = _multimodal_world(seed=0, modes=2)
    naive = SelfOrganizingMind(dim=512, seed=0)
    organizing = SelfOrganizingMind(dim=512, seed=0)
    _run_stream(naive, sample, K, 900, seed=1)
    _run_stream(organizing, sample, K, 900, seed=1)
    organizing.reorganize()

    rng = np.random.default_rng(99)
    test = [(sample(c := int(rng.integers(K))), c) for _ in range(400)]
    n_acc = np.mean([naive.classify(x, "vector")[0] == c for x, c in test])
    o_acc = np.mean([organizing.classify(x, "vector")[0] == c for x, c in test])
    assert n_acc < 0.7                       # a single prototype per label collapses
    assert o_acc >= 0.9                      # reorganized sub-prototypes recover it
    assert o_acc > n_acc + 0.25


def test_split_discovers_the_modes():
    _, sample, K, _ = _multimodal_world(seed=2, modes=2)
    mind = SelfOrganizingMind(dim=512, seed=2)
    _run_stream(mind, sample, K, 900, seed=3)
    rep = mind.reorganize()
    # it should have found about two modes per class -- without being told.
    for label in range(K):
        assert rep["per_label"][label] >= 2


def test_single_mode_stays_single():
    # When a class really is one cluster, the split expert must NOT over-split it.
    _, sample, K, _ = _multimodal_world(seed=4, modes=1)
    mind = SelfOrganizingMind(dim=512, seed=4)
    _run_stream(mind, sample, K, 600, seed=5)
    rep = mind.reorganize()
    for label in range(K):
        assert rep["per_label"][label] == 1


def test_swap_is_nondestructive_and_atomic():
    _, sample, K, _ = _multimodal_world(seed=6, modes=2)
    mind = SelfOrganizingMind(dim=512, seed=6)
    _run_stream(mind, sample, K, 600, seed=7)
    mind.reorganize()

    rng = np.random.default_rng(8)
    probe = [mind.encoder.encode(sample(int(rng.integers(K))), "vector") for _ in range(50)]
    before = [mind.live.classify(v)[0] for v in probe]
    shadow, _ = mind.build_shadow()          # build only -- must not touch live
    during = [mind.live.classify(v)[0] for v in probe]
    assert before == during                  # live unchanged while shadow is built
    mind.swap(shadow)                         # the only moment it can change


def test_merge_folds_same_label_duplicates_and_flags_collisions():
    rng = np.random.default_rng(0)
    u = rng.standard_normal(64); u /= np.linalg.norm(u)
    w = rng.standard_normal(64); w /= np.linalg.norm(w)
    near = u + 0.02 * rng.standard_normal(64); near /= np.linalg.norm(near)
    protos = [["A", u, u, 1], ["A", near, near, 1],   # same label, near-duplicate
              ["B", near, near, 1],                    # diff label, near-identical
              ["C", w, w, 1]]
    kept, collisions = MergeExpert(duplicate=0.9).prune(protos)
    labels = sorted(p[0] for p in kept)
    assert labels == ["A", "B", "C"]         # the two A's folded into one
    assert collisions >= 1                   # A/B near-identical was flagged


def test_self_trigger_fixes_cold_start_autonomously():
    from holographic_organizer import TriggerExpert
    _, sample, K, _ = _multimodal_world(seed=1, n_classes=2, modes=2)
    auto = SelfOrganizingMind(dim=512, seed=1, coherence=0.5)
    frozen = SelfOrganizingMind(dim=512, seed=1, coherence=0.5)
    trig = TriggerExpert(coherence_floor=0.6, novelty_rate=0.3, min_gap=200)
    rng = np.random.default_rng(2)
    fires = 0
    for i in range(1, 1001):
        c = int(rng.integers(2)); x = sample(c)
        auto.observe(x, c, "vector"); frozen.observe(x, c, "vector")
        if i % 50 == 0 and trig.assess(auto)["fire"]:
            auto.consider_reorganizing(trig); fires += 1

    r = np.random.default_rng(9)
    test = [(sample(c := int(r.integers(2))), c) for _ in range(400)]
    a_acc = np.mean([auto.classify(x, "vector")[0] == c for x, c in test])
    f_acc = np.mean([frozen.classify(x, "vector")[0] == c for x, c in test])
    assert fires >= 1                     # it decided to reorganize on its own
    assert f_acc < 0.7                    # leaving early data in place stays poor
    assert a_acc >= 0.9                   # self-triggered reorg re-places it well
    assert a_acc > f_acc + 0.25


def test_trigger_stays_quiet_when_already_coherent():
    # If a class really is one mode, the model is coherent and must NOT keep
    # reorganizing itself for no reason (no thrashing, no spurious splits).
    from holographic_organizer import TriggerExpert
    _, sample, K, _ = _multimodal_world(seed=3, n_classes=3, modes=1)
    mind = SelfOrganizingMind(dim=512, seed=3, coherence=0.5)
    trig = TriggerExpert(coherence_floor=0.6, novelty_rate=0.3, min_gap=200)
    rng = np.random.default_rng(4)
    fires = 0
    for i in range(1, 901):
        c = int(rng.integers(K)); mind.observe(sample(c), c, "vector")
        if i % 50 == 0 and trig.assess(mind)["fire"]:
            mind.consider_reorganizing(trig); fires += 1
    assert fires == 0                     # coherent already -> never fired
    assert all(v == 1 for v in mind.live.counts_by_label().values())


def test_autonomous_reorg_fixes_cold_start_without_thresholds():
    # auto_reorganize sets NO thresholds (no coherence floor, novelty rate, or gap).
    # It must still fix the cold-start blur and absorb a new class, by measurement.
    _, sample, K, rng = _multimodal_world(seed=1, n_classes=2, modes=2)
    _, sample3, _, _ = _multimodal_world(seed=7, n_classes=1, modes=2)
    auto = SelfOrganizingMind(dim=512, seed=1, coherence=0.5)
    never = SelfOrganizingMind(dim=512, seed=1, coherence=0.5)
    N = 1800
    rng2 = np.random.default_rng(2)
    for i in range(1, N + 1):
        if i > N // 2:
            c = int(rng2.integers(3)); x = sample3(0) if c == 2 else sample(c)
        else:
            c = int(rng2.integers(2)); x = sample(c)
        auto.observe(x, c, "vector"); never.observe(x, c, "vector")
        if i % 300 == 0:
            auto.auto_reorganize()

    r = np.random.default_rng(9)
    test = [((sample3(0), 2) if (c := int(r.integers(3))) == 2 else (sample(c), c)) for _ in range(400)]
    a_acc = np.mean([auto.classify(x, "vector")[0] == c for x, c in test])
    n_acc = np.mean([never.classify(x, "vector")[0] == c for x, c in test])
    assert n_acc < 0.7          # leaving the blur in place stays poor
    assert a_acc >= 0.9         # autonomous reorganization fixed it, no thresholds
    assert a_acc > n_acc + 0.25


def test_autonomous_reorg_does_not_oversplit_single_mode():
    # If a class really is one mode, every resolution ties on accuracy, so the
    # leanest (one prototype per label) must win -- no spurious splits.
    _, sample, K, _ = _multimodal_world(seed=3, n_classes=3, modes=1)
    mind = SelfOrganizingMind(dim=512, seed=3, coherence=0.5)
    rng = np.random.default_rng(4)
    for i in range(1, 1201):
        c = int(rng.integers(K)); mind.observe(sample(c), c, "vector")
        if i % 300 == 0:
            mind.auto_reorganize()
    r = np.random.default_rng(6)
    acc = np.mean([mind.classify(sample(c := int(r.integers(K))), "vector")[0] == c for _ in range(400)])
    assert acc >= 0.9
    assert mind.live.size() == K        # exactly one prototype per class, no over-split

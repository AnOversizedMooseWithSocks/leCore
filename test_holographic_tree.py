"""Tests for holographic_tree: deterministic recursive partition, capacity
restoration vs a flat memory, and the approximate-NN speed/recall tradeoff."""
import numpy as np
from holographic_ai import random_vector, HolographicMemory
from holographic_tree import HoloTree, HoloForest, capacity_curve, nn_benchmark, forest_benchmark


def _items(N, dim, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([random_vector(dim, rng) for _ in range(N)])


def test_tree_is_balanced_and_log_depth():
    tree = HoloTree(256, leaf_size=32, seed=0).build(_items(1000, 256))
    st = tree.stats()
    assert st["max_leaf"] <= 32 and st["leaves"] >= 1000 // 32
    assert st["depth"] <= 2 * np.log2(1000 / 32) + 3        # roughly log N / leaf


def test_build_is_deterministic():
    items = _items(500, 256)
    a = HoloTree(256, leaf_size=32, seed=7).build(items)
    b = HoloTree(256, leaf_size=32, seed=7).build(items)
    assert a.stats() == b.stats()
    q = items[3] + 0.4 * random_vector(256, np.random.default_rng(99))
    assert a.recall(q, beam=4) == b.recall(q, beam=4)       # same seed -> same answer


def test_exact_key_value_recall_perfect_in_tree():
    rng = np.random.default_rng(1); dim = 2048; N = 1024
    keys = np.stack([random_vector(dim, rng) for _ in range(N)])
    vals = np.stack([random_vector(dim, rng) for _ in range(N)])
    tree = HoloTree(dim, leaf_size=64, seed=0).build(keys, vals)
    ok = sum(int(tree.recall(keys[i]) == i) for i in range(N))
    assert ok / N >= 0.98                                   # leaves stay within capacity


def test_tree_beats_flat_when_dataset_is_big():
    # the headline: a single flat memory collapses past capacity; the tree holds
    rows = {r["N"]: r for r in capacity_curve([64, 1024], dim=2048, leaf_size=64, probes=120)}
    assert rows[64]["flat"] >= 0.95 and rows[64]["tree"] >= 0.95   # both fine when small
    assert rows[1024]["flat"] < 0.4                                # flat has collapsed
    assert rows[1024]["tree"] >= 0.95                              # tree still works


def test_nn_recall_improves_with_beam_and_is_cheaper():
    lo = nn_benchmark(N=1200, dim=512, leaf_size=64, beam=1, noise=0.5)
    hi = nn_benchmark(N=1200, dim=512, leaf_size=64, beam=16, noise=0.5)
    assert hi["tree_recall"] > lo["tree_recall"]            # more beam -> better recall
    assert hi["exact_recall"] >= 0.98                        # exact scan is the ceiling
    assert hi["tree_cmp"] < hi["exact_cmp"]                  # and the tree is cheaper


def test_flux_concentrates_like_veins():
    # after many varied queries, flux is uneven -- a few thick veins, many thin
    tree = HoloTree(256, leaf_size=32, seed=0).build(_items(800, 256))
    rng = np.random.default_rng(0)
    for _ in range(400):
        tree.recall(random_vector(256, rng), beam=3)
    flux = np.array(tree.flux())
    assert flux.sum() > 0 and flux.std() > 0                # not uniform


def test_forest_beats_single_tree_at_matched_cost():
    single = nn_benchmark(N=1500, dim=512, leaf_size=64, beam=4, noise=0.5)
    forest = forest_benchmark(N=1500, dim=512, leaf_size=64, n_trees=4, beam=4, noise=0.5)
    assert forest["forest_recall"] > single["tree_recall"]          # more trees, more recall
    assert forest["forest_recall"] >= 0.98                          # reaches ~exact
    assert forest["forest_cmp"] < forest["exact_cmp"]               # still cheaper than a scan


def test_forest_recall_is_correct_on_clean_keys():
    items = _items(600, 256)
    forest = HoloForest(256, n_trees=3, leaf_size=48, seed=0).build(items)
    ok = sum(int(forest.recall(items[i], beam=4) == i) for i in range(0, 600, 5))
    assert ok == len(range(0, 600, 5))                              # exact cues -> exact hits

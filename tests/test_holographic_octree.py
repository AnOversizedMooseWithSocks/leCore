"""Tests for the capacity-adaptive 3D holographic octree (TILE3D-1)."""

import numpy as np
from holographic.mesh_and_geometry.holographic_octree import HoloOctree, single_wave_recall


def _box():
    return (np.array([-1., -1, -1]), np.array([1., 1, 1]))


def _auc(stored, empty):
    return float(np.mean(np.asarray(stored)[:, None] > np.asarray(empty)[None, :]))


def test_octree_splits_when_over_capacity():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, (300, 3))
    tree = HoloOctree(_box(), capacity=32, dim=512).insert(pts)
    assert tree.children is not None                          # 300 > 32 -> it subdivided
    assert tree.n_vectors() > 1                               # more than one leaf wave
    # every leaf holds at most `capacity` points (the per-vector budget is respected, or it's at max depth)
    for lf in tree.leaves():
        assert len(lf.points) <= 32 or lf.depth >= tree.max_depth


def test_octree_is_a_bidirectional_index():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1, 1, (500, 3))
    tree = HoloOctree(_box(), capacity=40, dim=512).insert(pts)
    p = tree.all_points()[0]
    leaf = tree._leaf_for(p)                                  # forward: position -> leaf
    assert np.all(p >= leaf.lo - 1e-9) and np.all(p <= leaf.hi + 1e-9)   # the leaf's box contains it
    assert len(leaf.points) > 0                               # backward: the leaf holds its points
    assert tree.query(p) > tree.query(p + 0.6)               # semantic: stored recalls above shifted-empty


def test_octree_preserves_all_points():
    rng = np.random.default_rng(2)
    pts = rng.uniform(-1, 1, (250, 3))
    tree = HoloOctree(_box(), capacity=30, dim=512).insert(pts)
    assert len(tree.all_points()) == 250                     # tiling loses no points


def test_octree_beats_single_wave_at_scale():
    rng = np.random.default_rng(3)
    pts = rng.uniform(-1, 1, (800, 3))
    ps = pts[rng.choice(800, 40, replace=False)]; pe = rng.uniform(-1, 1, (40, 3))
    sw_s = single_wave_recall(pts, ps, dim=1024, bandwidth=8.0)
    sw_e = single_wave_recall(pts, pe, dim=1024, bandwidth=8.0)
    tree = HoloOctree(_box(), capacity=48, dim=1024, bandwidth=8.0).insert(pts)
    t_s = np.array([tree.query(p) for p in ps]); t_e = np.array([tree.query(p) for p in pe])
    # at N=800 the single wave is near chance; the octree still separates stored from empty
    assert _auc(t_s, t_e) > _auc(sw_s, sw_e)
    assert _auc(t_s, t_e) > 0.8


def test_capacity_controls_split_depth():
    rng = np.random.default_rng(4)
    pts = rng.uniform(-1, 1, (400, 3))
    coarse = HoloOctree(_box(), capacity=200, dim=256).insert(pts)   # high capacity -> few splits
    fine = HoloOctree(_box(), capacity=20, dim=256).insert(pts)      # low capacity -> many splits
    assert fine.n_vectors() > coarse.n_vectors()

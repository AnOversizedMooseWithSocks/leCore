"""Tests for holographic_merge.py -- reconciling forked worlds via pairwise opponent divergence."""
import numpy as np
from holographic_merge import merge_forks


def _base(dim=512, seed=0):
    b = np.random.default_rng(seed).standard_normal(dim); return b / np.linalg.norm(b)


def test_single_fork_slot_is_kept():
    b = _base()
    res = merge_forks([{"a": b}, {}])
    assert "a" in res["merged"] and not res["conflicts"]


def test_agreeing_forks_merge_conflict_free():
    b = _base(); rng = np.random.default_rng(1)
    v1 = b + 0.003 * rng.standard_normal(512)
    v2 = b + 0.003 * rng.standard_normal(512)
    res = merge_forks([{"pos": v1}, {"pos": v2}], policy="select")
    assert "pos" in res["merged"] and not res["conflicts"]
    mc = res["merged"]["pos"]
    assert float(np.dot(mc, b) / (np.linalg.norm(mc) * np.linalg.norm(b))) > 0.9


def test_disagreeing_forks_surface_as_conflict():
    rng = np.random.default_rng(2)
    w1, w2 = rng.standard_normal(512), rng.standard_normal(512)
    res = merge_forks([{"col": w1}, {"col": w2}], policy="select")
    assert not res["merged"] and len(res["conflicts"]) == 1 and res["conflicts"][0][0] == "col"


def test_left_right_callable_auto_policies():
    rng = np.random.default_rng(3)
    w1, w2 = rng.standard_normal(256), rng.standard_normal(256)
    assert np.allclose(merge_forks([{"c": w1}, {"c": w2}], policy="left")["merged"]["c"], w1)
    assert np.allclose(merge_forks([{"c": w1}, {"c": w2}], policy="right")["merged"]["c"], w2)
    cb = merge_forks([{"c": w1}, {"c": w2}], policy=lambda s, v: v[1])
    assert np.allclose(cb["merged"]["c"], w2)
    auto = merge_forks([{"c": w1}, {"c": w2}], policy="auto")
    assert not auto["merged"] and not auto["conflicts"]


def test_n_forks_reconciled_pairwise():
    b = _base(seed=5); rng = np.random.default_rng(6)
    v1 = b + 0.003 * rng.standard_normal(512)
    v2 = b + 0.003 * rng.standard_normal(512)
    v3 = b + 0.003 * rng.standard_normal(512)
    stray = rng.standard_normal(512)
    # all three agree -> merged
    assert "x" in merge_forks([{"x": v1}, {"x": v2}, {"x": v3}])["merged"]
    # two agree, one strays -> NOT all-agree -> conflict (pairwise, per leOS)
    res = merge_forks([{"x": v1}, {"x": v2}, {"x": stray}], policy="select")
    assert res["conflicts"] and res["conflicts"][0][0] == "x"


def test_through_mind():
    from holographic_unified import UnifiedMind
    m = UnifiedMind(dim=256, seed=0)
    b = _base(dim=256, seed=7); rng = np.random.default_rng(8)
    mine = {"scene": b + 0.003 * rng.standard_normal(256), "note": rng.standard_normal(256)}
    theirs = {"scene": b + 0.003 * rng.standard_normal(256)}
    res = m.merge_forks([mine, theirs], policy="select")
    assert "scene" in res["merged"] and "note" in res["merged"]     # scene agreed; note only in one fork

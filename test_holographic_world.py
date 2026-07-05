"""Tests for holographic_world.py -- the fork -> edit -> merge -> apply loop over vector slots."""
import numpy as np
from holographic_world import World, Fork, WorldSpace
from holographic_merge import merge_forks


def test_fork_is_copy_on_write():
    w = World("lab", {"ground": np.ones(8)})
    f = w.fork()
    f.set("sky", np.arange(8, dtype=float))
    assert "sky" not in w.slots                      # shared world untouched
    assert np.allclose(f.get("sky"), np.arange(8))   # fork sees its own edit
    assert np.allclose(f.get("ground"), np.ones(8))  # and the base


def test_forks_are_isolated_from_each_other():
    w = World("lab", {"g": np.ones(8)})
    a, b = w.fork(), w.fork()
    a.set("x", np.full(8, 2.0))
    b.set("y", np.full(8, 3.0))
    assert "y" not in a.slots() and "x" not in b.slots()   # neither sees the other's edits


def test_apply_writes_merged_delta_back():
    w = World("lab", {"g": np.ones(8)})
    changed = w.apply({"sky": np.full(8, 5.0), "sun": np.full(8, 6.0)})
    assert set(changed) == {"sky", "sun"}
    assert np.allclose(w.get("sky"), 5.0) and np.allclose(w.get("sun"), 6.0)


def test_full_loop_agree_merges_and_applies():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(256); base /= np.linalg.norm(base)
    ws = WorldSpace()
    ws.world("lab").set("ground", base)
    mine, theirs = ws.fork("lab"), ws.fork("lab")
    blue = rng.standard_normal(256)
    mine.set("sky", blue)
    theirs.set("sky", blue + 0.002 * rng.standard_normal(256))
    res = merge_forks([mine.delta, theirs.delta], policy="select")
    assert "sky" in res["merged"] and not res["conflicts"]
    ws.apply(res["merged"], "lab")
    assert "sky" in ws.world("lab").slots


def test_full_loop_conflict_surfaced_not_applied():
    rng = np.random.default_rng(1)
    ws = WorldSpace()
    a, b = ws.fork("lab"), ws.fork("lab")
    a.set("color", rng.standard_normal(64))
    b.set("color", rng.standard_normal(64))
    res = merge_forks([a.delta, b.delta], policy="select")
    assert not res["merged"] and res["conflicts"][0][0] == "color"
    ws.apply(res["merged"], "lab")
    assert "color" not in ws.world("lab").slots       # conflict was NOT silently written


def test_through_mind():
    from holographic_unified import UnifiedMind
    m = UnifiedMind(dim=128, seed=0)
    rng = np.random.default_rng(2)
    m.workspace.world("lab").set("g", rng.standard_normal(128))
    f = m.workspace.fork("lab")
    f.set("s", rng.standard_normal(128))
    m.apply(merge_forks([f.delta], policy="select")["merged"], world="lab")
    assert "s" in m.workspace.world("lab").slots

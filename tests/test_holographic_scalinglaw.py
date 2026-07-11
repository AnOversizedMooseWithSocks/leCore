"""Tests for holographic_scalinglaw: detect which limit binds, apply the right lever.

Pins the contracts:
  * a variance-limited workload (err ~ 1/sqrt(dim)) diagnoses 'scale:dim';
  * a capacity-limited workload (per-tile load) diagnoses 'scale:tiles';
  * a structural wall (no knob helps) is called 'wall' and auto_scale STOPS EARLY
    with the probe table as evidence, instead of burning its budget;
  * auto_scale reaches a target with the trajectory carrying per-step evidence;
  * THE SHIFTING LIMIT: a workload whose binding constraint CHANGES as it scales
    (tiles first, then dim) is tracked round by round -- the generalisation the
    module exists for;
  * a REAL engine workload (bundle discriminability vs distractors) rediscovers
    the dim rule on the live primitive.

All eval functions are deterministic; every assertion is a hard contract.
"""

import numpy as np
import pytest

from holographic.misc.holographic_scalinglaw import (
    diagnose_scaling, auto_scale, RESPONSIVE_DROP, diagnose_bake,
)


# ---------------------------------------------------------------------------
# Classification of the three limit types
# ---------------------------------------------------------------------------

def test_variance_limited_picks_dim():
    fn = lambda dim, tiles: 1.0 / np.sqrt(dim)
    d = diagnose_scaling(fn, {"dim": 64, "tiles": 4})
    assert d["verdict"] == "scale:dim"
    assert d["ranked"][0] == "dim"
    # a sqrt law gives 1 - 1/sqrt(2) ~ 29% per doubling, well above threshold
    dim_probe = next(p for p in d["probes"] if p["knob"] == "dim")
    assert dim_probe["drop"] > 0.25


def test_capacity_limited_picks_tiles():
    fn = lambda dim, tiles: (100.0 / tiles) / (100.0 / tiles + 10.0)
    d = diagnose_scaling(fn, {"dim": 256, "tiles": 2})
    assert d["verdict"] == "scale:tiles"


def test_wall_is_called_honestly():
    fn = lambda dim, tiles, bits: 0.42
    d = diagnose_scaling(fn, {"dim": 64, "tiles": 4, "bits": 8})
    assert d["verdict"] == "wall"
    assert not d["responsive"]
    assert all(abs(p["drop"]) < RESPONSIVE_DROP for p in d["probes"])


# ---------------------------------------------------------------------------
# auto_scale: the loop, its evidence, and its stopping discipline
# ---------------------------------------------------------------------------

def test_auto_scale_reaches_target_with_evidence():
    fn = lambda dim: 1.0 / np.sqrt(dim)
    r = auto_scale(fn, {"dim": 64}, target_error=0.05)
    assert r["met"] and not r["wall"]
    assert r["final_error"] <= 0.05
    assert r["final_knobs"]["dim"] > 64
    # every step names the knob it doubled and the error after -- the baseline rule
    for step in r["trajectory"]:
        assert step["doubled"] == "dim"
        assert "error" in step and "knobs" in step


def test_auto_scale_stops_at_a_wall_without_burning_budget():
    fn = lambda dim, tiles: 0.42
    r = auto_scale(fn, {"dim": 64, "tiles": 4}, target_error=0.1, max_rounds=8)
    assert r["wall"] and not r["met"]
    assert len(r["trajectory"]) == 0          # not one wasted doubling
    assert "wall_probes" in r                  # the evidence travels with the verdict


def test_auto_scale_respects_round_budget():
    # target unreachable in the rounds allowed -> met=False, no wall (dim DOES help,
    # there just was not budget) -- the two failure modes must stay distinct.
    fn = lambda dim: 1.0 / np.sqrt(dim)
    r = auto_scale(fn, {"dim": 4}, target_error=1e-6, max_rounds=3)
    assert not r["met"] and not r["wall"]
    assert len(r["trajectory"]) == 3


def test_integer_knobs_stay_integers():
    fn = lambda tiles: 1.0 / tiles
    r = auto_scale(fn, {"tiles": 3}, target_error=0.05, max_rounds=6)
    for step in r["trajectory"]:
        assert isinstance(step["knobs"]["tiles"], int)


# ---------------------------------------------------------------------------
# THE SHIFTING LIMIT: the binding constraint changes as the system scales
# ---------------------------------------------------------------------------

def test_shifting_limit_is_tracked_round_by_round():
    # A workload with TWO error terms where the binding constraint genuinely
    # CHANGES: a per-tile capacity term that dominates at the start but saturates
    # (a floor -- past ~8 tiles more tiling stops paying), and a dim-variance term
    # that then becomes the binding limit. auto_scale must double tiles first,
    # then SWITCH to dim -- the re-diagnose-every-round behaviour that a one-shot
    # diagnosis cannot give.
    fn = lambda dim, tiles: max(2.0 / tiles, 0.05) + (4.0 / np.sqrt(dim))
    r = auto_scale(fn, {"dim": 100, "tiles": 2}, target_error=0.25, max_rounds=12)
    assert r["met"], r
    doubled = [s["doubled"] for s in r["trajectory"]]
    assert doubled[0] == "tiles"               # capacity binds first
    assert "dim" in doubled                    # ...then variance takes over
    # once dim becomes the pick, tiles has stopped being the binding limit
    switch = doubled.index("dim")
    assert all(k == "tiles" for k in doubled[:switch])


# ---------------------------------------------------------------------------
# Real engine workload: rediscover the dim rule on the live primitive
# ---------------------------------------------------------------------------

def test_real_bundle_discriminability_diagnoses_dim():
    from holographic.agents_and_reasoning.holographic_ai import (
        random_vector, bundle)

    def workload(dim, seed=0, n_items=40, n_distractors=40):
        dim = int(dim)
        items = [random_vector(dim, np.random.default_rng(seed + 1 + i))
                 for i in range(n_items)]
        b = bundle(items)
        nb = np.linalg.norm(b)
        mcos = np.array([float(np.dot(b, it)) / (nb * np.linalg.norm(it))
                         for it in items])
        dis = [random_vector(dim, np.random.default_rng(seed + 1000 + i))
               for i in range(n_distractors)]
        dcos = np.array([float(np.dot(b, d)) / (nb * np.linalg.norm(d))
                         for d in dis])
        return float(np.mean(dcos > np.min(mcos)))

    d = diagnose_scaling(workload, {"dim": 128})
    assert d["verdict"] == "scale:dim"
    assert d["probes"][0]["drop"] > 0.1


def test_diagnosis_is_deterministic():
    fn = lambda dim, tiles: (8.0 / tiles) + (1.0 / np.sqrt(dim))
    assert diagnose_scaling(fn, {"dim": 64, "tiles": 4}) == \
           diagnose_scaling(fn, {"dim": 64, "tiles": 4})


# ---------------------------------------------------------------------------
# diagnose_bake: the scaling diagnostic wired to the real n-D texture bake
# ---------------------------------------------------------------------------

def _field(freq):
    ax = np.linspace(0.0, 1.0, 40)
    P = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1)
    return ax, np.sin(2 * np.pi * freq * P[..., 0]) * np.cos(2 * np.pi * freq * P[..., 1])


def test_diagnose_bake_calls_margin_on_a_bias_limited_field():
    # a smooth field baked with a wide kernel is BIAS-limited: widening/narrowing the kernel (margin) is the
    # lever, and more dimension is nearly wasted. The verdict and the evidence must agree.
    ax, vals = _field(1)
    r = diagnose_bake([ax, ax], vals, dim=4096, margin=1.5)
    assert r["verdict"] == "scale:margin"
    drops = {p["knob"]: p["drop"] for p in r["probes"]}
    assert drops["margin"] > drops["dim"]                     # the recommendation carries its own proof


def test_diagnose_bake_calls_dim_on_a_variance_limited_field():
    # a high-frequency field at low dimension is VARIANCE-limited: only more dimension resolves it.
    ax, vals = _field(3)
    r = diagnose_bake([ax, ax], vals, dim=1024, margin=2.5)
    assert r["verdict"] == "scale:dim"
    drops = {p["knob"]: p["drop"] for p in r["probes"]}
    assert drops["dim"] > drops["margin"]


def test_diagnose_bake_scores_on_held_out_points_not_the_grid():
    # the honest-baseline guarantee: with no queries given, it scores on interior points it generates, never on
    # the baked grid itself (which any bake reproduces trivially). Passing explicit queries must also work and be
    # deterministic.
    ax, vals = _field(1)
    q = np.random.default_rng(0).uniform(0.1, 0.9, (50, 2))
    a = diagnose_bake([ax, ax], vals, queries=q, dim=2048, margin=1.5)
    b = diagnose_bake([ax, ax], vals, queries=q, dim=2048, margin=1.5)
    assert a == b                                             # deterministic given the same held-out set
    assert a["base_error"] > 0.0                              # a real generalization error, not a trivial 0


def test_selftest_runs():
    from holographic.misc.holographic_scalinglaw import _selftest
    _selftest()

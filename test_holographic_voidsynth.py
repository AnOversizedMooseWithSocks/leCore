"""Tests for void-capability-gap program synthesis (SYNTH-1)."""

import numpy as np
from holographic_orchestrator import chain_signature
from holographic_voidsynth import synthesize_for_goal, blend_programs, verify_chain, fill_capability_gap


def _library(n=10, dim=256, seed=0):
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((n, dim))
    return L / np.linalg.norm(L, axis=1, keepdims=True)


def test_synthesizes_a_reachable_goal():
    L = _library()
    goal = chain_signature(L[[2, 5, 7]])                      # a real 3-tool chain signature
    res = synthesize_for_goal(L, goal, max_length=4, threshold=0.85)
    assert res["status"] == "synthesized" and res["coherence"] >= 0.85
    # the returned chain really does reach the goal (verification, not trust)
    assert verify_chain(L, res["chain"], goal) >= 0.85


def test_abstains_on_an_unreachable_goal():
    L = _library()
    rng = np.random.default_rng(99)
    junk = rng.standard_normal(256); junk /= np.linalg.norm(junk)   # independent of the library
    res = synthesize_for_goal(L, junk, max_length=4, threshold=0.85)
    assert res["status"] == "abstain"                        # declines rather than executing an incoherent program
    assert res["coherence"] < 0.85


def test_gate_separates_fillable_from_void_over_trials():
    syn, abst = 0, 0
    for t in range(8):
        L = _library(seed=t)
        rng = np.random.default_rng(1000 + t)
        g = chain_signature(L[rng.choice(10, 3, replace=False)])
        syn += synthesize_for_goal(L, g, threshold=0.85)["status"] == "synthesized"
        j = rng.standard_normal(256); j /= np.linalg.norm(j)
        abst += synthesize_for_goal(L, j, threshold=0.85)["status"] == "abstain"
    assert syn >= 7 and abst >= 7                             # the gate is reliable both ways


def test_blend_carries_both_goals():
    L = _library()
    gA = chain_signature(L[[1, 3]]); gB = chain_signature(L[[6, 8]])
    blend = blend_programs(gA, gB)
    cosA = float(blend @ gA) / (np.linalg.norm(blend) * np.linalg.norm(gA))
    cosB = float(blend @ gB) / (np.linalg.norm(blend) * np.linalg.norm(gB))
    assert cosA > 0.4 and cosB > 0.4                         # one blended program, both intents


def test_cross_domain_blend():
    gfx = _library(6, seed=1); aud = _library(6, seed=2)     # two separate "domains"
    g_gfx = chain_signature(gfx[[0, 2]]); g_aud = chain_signature(aud[[1, 5]])
    rg = synthesize_for_goal(gfx, g_gfx, threshold=0.85); ra = synthesize_for_goal(aud, g_aud, threshold=0.85)
    blend = blend_programs(chain_signature(gfx[rg["chain"]]), chain_signature(aud[ra["chain"]]))
    cg = float(blend @ g_gfx) / (np.linalg.norm(blend) * np.linalg.norm(g_gfx))
    ca = float(blend @ g_aud) / (np.linalg.norm(blend) * np.linalg.norm(g_aud))
    assert cg > 0.4 and ca > 0.4                             # one vector, coherent across both domains


def test_registry_hit_skips_synthesis():
    L = _library()
    g = chain_signature(L[[2, 5, 7]])
    res = fill_capability_gap(L, g, registry_hit=0.95, threshold=0.85)
    assert res["status"] == "registry"                       # no gap -> use the registered chain, don't synthesize

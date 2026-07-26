"""The reclock layer across faculties and through the mind: the transform, its built-in null (the seam with
holographic_honesty.pipeline_null), the duration channel, and the layer's headline kept negative -- two clock
mechanisms telling confident OPPOSITE stories about the same pure noise, both fake."""
import numpy as np

import lecore


def test_price_clock_mechanics_are_exact_and_causal():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    ramp = np.arange(101, dtype=float)
    ev = mind.reclock(ramp, step=5)
    assert ev["n_events"] == 20
    assert np.all(ev["duration"] == 5) and np.all(ev["rotation"] == 1)
    # causality by construction: each event lands at the FIRST index where its crossing is knowable.
    assert np.array_equal(ev["source_index"], np.arange(5, 101, 5))


def test_manufactured_persistence_is_caught_by_the_built_in_null():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    noise = np.random.default_rng(0).normal(size=4000)
    naive = mind.rotation_persistence(mind.reclock(noise, step=2.0))
    honest = mind.null_persistence(noise, step=2.0, n=80, seed=0)
    # a confident fake ~-25-point "reversion effect" on structureless input...
    assert abs(naive - 0.5) > 0.15
    # ...that the machinery's own null absorbs entirely.
    assert abs(honest["z"]) < 2.5 and not honest["collapsed"]
    assert abs(honest["null_mean"] - naive) < 0.05


def test_real_alternating_structure_still_separates_through_the_same_chain():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(1)
    legs = np.concatenate([np.full(50, s) for s in ([+0.5, -0.5] * 40)])
    trend = np.cumsum(legs + 0.3 * rng.normal(size=legs.size))
    honest = mind.null_persistence(trend, step=2.0, n=80, seed=0)
    assert abs(honest["z"]) > 3.0 and honest["collapsed"]


def test_duration_channel_and_its_resolution_gate():
    mind = lecore.UnifiedMind(dim=256, seed=0)
    # slow-up / fast-down sawtooth: rises take ~4x the samples falls do -- the asymmetry the channel exists for.
    saw = np.concatenate([np.concatenate([np.linspace(0, 8, 40, endpoint=False),
                                          np.linspace(8, 0, 10, endpoint=False)]) for _ in range(20)])
    ev = mind.reclock(saw, step=2.0)
    assert mind.duration_resolution_check(ev)["ok"]
    ds = mind.duration_stats(ev)
    assert ds["mean_up"] > 2.0 * ds["mean_down"] and ds["updown_z"] > 5.0
    # the -inf log-duration incident's signature: a step far below per-sample movement fails the gate loudly.
    jumpy = np.cumsum(np.random.default_rng(2).normal(0, 5.0, size=400))
    bad = mind.duration_resolution_check(mind.reclock(jumpy, step=0.5))
    assert not bad["ok"] and bad["skipped_gap"] > 0


def test_reclock_layer_is_deterministic_end_to_end():
    a, b = lecore.UnifiedMind(dim=256, seed=0), lecore.UnifiedMind(dim=256, seed=0)
    x = np.cumsum(np.random.default_rng(3).normal(size=600))
    ea, eb = a.reclock(x, step=1.5), b.reclock(x, step=1.5)
    assert all(np.array_equal(ea[k], eb[k]) for k in ("source_index", "duration", "rotation"))
    assert a.null_persistence(x, step=1.5, n=20, seed=1) == b.null_persistence(x, step=1.5, n=20, seed=1)

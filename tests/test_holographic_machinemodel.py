"""The leCore machine model: hardware units and memory tiers, named and MEASURED.

The contract this suite enforces:

  1. EVERY registered unit points at a symbol that actually resolves. A registry that names a function which is not
     there is a lie that looks like documentation, and it is the exact failure mode this whole module exists to fix.
  2. The cost arithmetic refuses to promise a win it cannot deliver: `break_even` returns `inf` when a unit's
     marginal cost is not below the baseline.
  3. KEPT NEGATIVE: the textbook latency ladder (registers < L1 < L2 < RAM) is FALSE here. A dense array index beats
     the fat-margin cache and the texture unit on a single scalar access, because none of them is a scalar unit.
  4. THE HEADLINE: `gather`'s marginal cost is CONSTANT in N -- that is what makes it a hardware unit rather than a
     helper function.
"""

import numpy as np
import pytest

from holographic.misc.holographic_machinemodel import (
    UNITS, break_even, machine_map, place, place_unit, resolve, spec_sheet, unit, units)


def test_selftest_runs():
    from holographic.misc import holographic_machinemodel as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# 1. the registry must not lie
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(UNITS))
def test_every_named_symbol_actually_resolves(name):
    obj = resolve(name)
    assert obj is not None


def test_every_unit_carries_its_cost_model_and_its_refusal():
    for name in units():
        rec = unit(name)
        for field in ("kind", "gpu_name", "module", "symbol", "setup", "marginal", "scaling",
                      "use_when", "do_not_use_when", "why"):
            assert rec[field], (name, field)
        assert rec["kind"] in ("compute", "memory")
        # a unit with no stated failure condition is a unit nobody has measured
        assert len(rec["do_not_use_when"]) > 20, name


def test_the_map_is_plain_data_and_covers_both_kinds():
    m = machine_map()
    assert len(m) == len(UNITS)
    assert len(machine_map("compute")) >= 8
    assert len(machine_map("memory")) >= 6
    assert len(machine_map("compute")) + len(machine_map("memory")) == len(m)
    import json
    json.dumps(m)                                     # must cross an HTTP boundary


def test_unknown_units_raise_rather_than_return_none():
    with pytest.raises(KeyError):
        unit("tensor_core_that_does_not_exist")


def test_the_gpu_vocabulary_is_present_because_that_is_the_point():
    names = {unit(n)["gpu_name"].lower() for n in units()}
    joined = " | ".join(names)
    for word in ("simd", "warp", "texture", "gather", "rt core", "rng", "scheduler", "occupancy"):
        assert word in joined, word


# ---------------------------------------------------------------------------------------------------------
# 2. the cost arithmetic
# ---------------------------------------------------------------------------------------------------------

def test_break_even_is_the_amortization_question_stated_once():
    # N * baseline > setup + N * marginal  =>  N > setup / (baseline - marginal)
    assert break_even(setup_ns=1000.0, marginal_ns=5.0, baseline_ns=10.0) == 200.0
    assert break_even(setup_ns=0.0, marginal_ns=1.0, baseline_ns=10.0) == 0.0


def test_break_even_refuses_to_promise_a_win_it_cannot_deliver():
    # a unit whose marginal cost is NOT below the baseline can never pay, at any N. `inf` is the answer.
    assert break_even(setup_ns=1.0, marginal_ns=10.0, baseline_ns=10.0) == float("inf")
    assert break_even(setup_ns=0.0, marginal_ns=20.0, baseline_ns=10.0) == float("inf")
    assert place(baseline_ns=1.0, n_calls=10 ** 9, setup_ns=0.0, marginal_ns=2.0)["use_unit"] is False


def test_place_answers_use_the_unit_or_not():
    hot = place(baseline_ns=10.0, n_calls=1000, setup_ns=1000.0, marginal_ns=5.0)
    assert hot["use_unit"] is True and hot["break_even_n"] == 200.0
    assert hot["speedup"] > 1.0

    cold = place(baseline_ns=10.0, n_calls=10, setup_ns=1000.0, marginal_ns=5.0)
    assert cold["use_unit"] is False and cold["speedup"] < 1.0
    assert cold["break_even_n"] == hot["break_even_n"]      # the break-even is a property of the unit, not of N


def test_place_at_exactly_the_break_even_does_not_claim_a_win():
    p = place(baseline_ns=10.0, n_calls=200, setup_ns=1000.0, marginal_ns=5.0)
    assert p["unit_total_ns"] == p["baseline_total_ns"]
    assert p["use_unit"] is False                            # a tie is not a win; the setup risk is unpaid


# ---------------------------------------------------------------------------------------------------------
# 3 + 4. the two structural findings, measured live
# ---------------------------------------------------------------------------------------------------------

def test_kept_negative_the_latency_ladder_is_not_monotone():
    # The textbook frame says registers < L1 < L2 < RAM. Measured here it inverts, because none of leCore's tiers
    # is a scalar unit -- they are BATCH units, and a bare array read has no Python call overhead at all.
    sheet = spec_sheet(quick=True)
    dense = sheet["baseline_dense_index"]["marginal_ns"]
    assert sheet["t1_margin_cache"]["marginal_ns"] > dense       # "L1" is slower than "RAM"
    assert sheet["texture_unit"]["marginal_ns"] > dense          # ... and the texture unit far more so
    assert dense < 2000.0                                        # a raw read really is fast


def test_the_gather_unit_has_constant_marginal_cost_in_n():
    # THE HEADLINE. One dot product, whatever N was -- which is what makes it a hardware unit and not a helper.
    from holographic.rendering.holographic_shader import bake_1d, gather, gather_rule
    import time

    xs = np.linspace(0, 1, 64)
    b = bake_1d(xs, np.sin(6 * xs), dim=1024)
    rng = np.random.default_rng(0)

    def marginal(n):
        rule = gather_rule(b, rng.uniform(0, 1, n), rng.normal(size=n))
        gather(b, rule)
        t0 = time.perf_counter()
        for _ in range(200):
            gather(b, rule)
        return (time.perf_counter() - t0) / 200

    small, large = marginal(8), marginal(256)
    assert large < 3.0 * small, (small, large)          # 32x the lookups must not cost 32x the time


def test_the_spec_sheet_measures_itself_rather_than_quoting_a_comment():
    sheet = spec_sheet(quick=True)
    assert "baseline_dense_index" in sheet
    for k, v in sheet.items():
        assert v["marginal_ns"] >= 0.0 and v["setup_ns"] >= 0.0 and v["note"], k
    # a bake costs far more than a fetch: that asymmetry IS the amortization question
    assert sheet["texture_unit"]["setup_ns"] > sheet["texture_unit"]["marginal_ns"]


def test_m2_every_unit_measures_itself():
    # THE BAR: place() must run on measured numbers, so every unit in UNITS has to appear in the spec sheet.
    sheet = spec_sheet(quick=True)
    assert not (set(UNITS) - set(sheet)), sorted(set(UNITS) - set(sheet))


def test_a_marginal_cost_of_zero_is_an_answer_not_a_hole():
    # A sleeping island costs nothing to skip; a colour wave is a list you already have. Those units pay their
    # whole price in setup, and the spec sheet must be allowed to say so.
    sheet = spec_sheet(quick=True)
    assert sheet["scheduler"]["marginal_ns"] == 0.0
    assert sheet["occupancy_gate"]["marginal_ns"] == 0.0
    assert sheet["scheduler"]["setup_ns"] > 0.0


def test_the_two_measurements_that_were_fictions_now_measure_real_work():
    # First draft timed `key in cc._store` behind a hasattr guard, and `Cold(payload).get()` on a still-WARM
    # object -- both no-ops. A spec sheet that benchmarks a no-op is worse than no spec sheet.
    sheet = spec_sheet(quick=True)
    assert sheet["t5_cold_store"]["marginal_ns"] > 0.0        # a real inflate
    assert sheet["t3_content_addressed"]["marginal_ns"] > 0.0  # a real LRU hit


def test_place_unit_runs_on_measured_numbers():
    sheet = spec_sheet(quick=True)
    base = sheet["baseline_dense_index"]["marginal_ns"]

    # against a raw array read almost nothing can pay, and the model must SAY so rather than flatter a unit
    assert place_unit("texture_unit", base, 10 ** 6, sheet=sheet)["break_even_n"] == float("inf")
    assert place_unit("t4_compressed_ram", base, 10 ** 6, sheet=sheet)["use_unit"] is False

    # against a genuinely expensive evaluator (50 us) the baked grid pays almost immediately
    hot = place_unit("t2_baked_grid", 50_000.0, 10 ** 6, sheet=sheet)
    assert hot["use_unit"] is True and hot["speedup"] > 5.0
    assert hot["unit"] == "t2_baked_grid" and hot["note"]

    with pytest.raises(KeyError):
        place_unit("no_such_unit", 100.0, 10, sheet=sheet)


def test_kept_negative_the_denominator_decides_the_verdict():
    # The program's oldest error, guarded. kernel_fusion replaces N PASSES. Priced against ONE pass it loses --
    # and that verdict is an artifact of the denominator, not a fact about the unit.
    sheet = spec_sheet(quick=True)
    one_pass = sheet["kernel_fusion"]["marginal_ns"]
    assert place_unit("kernel_fusion", one_pass, 10 ** 6, sheet=sheet)["break_even_n"] == float("inf")
    # priced against the 50 passes it actually replaces, it pays
    fifty = place_unit("kernel_fusion", 50 * one_pass, 10 ** 6, sheet=sheet)
    assert fifty["use_unit"] is True


# ---------------------------------------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------------------------------------

def test_fully_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    assert "gather_unit" in m.machine_units("compute")
    assert "t4_compressed_ram" in m.machine_units("memory")
    assert m.machine_unit("gather_unit")["scaling"] == "O(1) in N"
    assert len(m.machine_map()) == len(UNITS)

    p = m.machine_place(baseline_ns=10, n_calls=1000, setup_ns=1000, marginal_ns=5)
    assert p["use_unit"] is True and p["break_even_n"] == 200.0

    sheet = m.machine_spec_sheet(quick=True)
    assert sheet["baseline_dense_index"]["marginal_ns"] > 0
    assert not (set(UNITS) - set(sheet))
    pu = m.machine_place_unit("t2_baked_grid", 50_000.0, 10 ** 6, sheet=sheet)
    assert pu["use_unit"] is True and pu["unit"] == "t2_baked_grid"

    assert "machine model" in str(m.find_capability("what is the gpu equivalent here")[:3]).lower()


def test_cross_faculty_the_units_agree_with_the_gates_they_describe():
    # The machine model must not become a second, drifting opinion. `place()` is the same arithmetic as the gates
    # that already ship, so a unit's advice has to agree with its own module's gate.
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    # operator_power's stated rule is "k >= 20*d" -- which is exactly should_jump.
    assert "20*d" in m.machine_unit("operator_power")["use_when"].replace(" ", "")
    assert m.should_jump(24, 3840) is True and m.should_jump(24, 16) is False

    # t4_compressed_ram's stated refusal is worth_factoring -- and it really does refuse noise.
    assert "worth_factoring" in m.machine_unit("t4_compressed_ram")["do_not_use_when"]
    noisy = np.random.default_rng(0).normal(size=(64, 64))
    assert m.worth_factoring(noisy)["worth_factoring"] is False

    # occupancy_gate's stated refusal is the single-threshold flicker -- and SleepTracker enforces hysteresis.
    assert "flicker" in m.machine_unit("occupancy_gate")["do_not_use_when"]

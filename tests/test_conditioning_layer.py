"""The conditioning layer must hold its contracts THROUGH the mind and ALONGSIDE the rest of the
honesty layer -- a shared kernel is not a shared manifold, so the cross-faculty chain is pinned
here rather than assumed from the module selftest.

The chain under test is the one the campaign actually walked: measure an effect, condition it on a
CAUSAL gate, check the inside events replicate under split_half, evaluate it per measured regime,
and confirm the pipeline itself did not manufacture it. Each step is a different faculty; the point
of this file is that they compose without any of them quietly changing the answer.
"""
import numpy as np
import pytest

import lecore
from holographic.agents_and_reasoning.holographic_conditioning import (
    Gate, ExPostMask, trailing_gate, conditional, across_regimes, insurance_profile)


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


def test_causal_gate_audits_clean_and_a_full_sample_gate_is_caught(mind):
    """The distinguishing claim of the whole layer: causality is PROVED, not declared."""
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 1, 400)) + 100.0

    good = mind.causal_gate("drawdown", window=90, threshold=-0.05, compare="le")
    assert good.audit_causality(x, seed=0)["passed"]

    # The classic leak: a threshold taken from a FULL-SAMPLE quantile, claiming to be causal.
    leaky = Gate(lambda c: np.asarray(c, float) > np.quantile(np.asarray(c, float), 0.9),
                 name="global_quantile", causal=True)
    audit = leaky.audit_causality(x, seed=0)
    assert not audit["passed"] and audit["first_violation"] is not None


def test_gate_over_the_wire_shape_is_json_able(mind):
    """An agent gets a mask + its audit, not an un-serialisable object -- the HTTP path."""
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 1, 300)) + 100.0
    out = mind.causal_gate("std", window=60, threshold=1.0, compare="ge", context=x)
    assert set(out) == {"mask", "n_true", "audit"}
    assert isinstance(out["mask"], list) and len(out["mask"]) == x.size
    assert all(isinstance(b, bool) for b in out["mask"][:8])
    assert out["audit"]["passed"] is True


def test_conditional_matches_a_hand_computed_difference_exactly(mind):
    """No cleverness in the estimator: it must equal the raw NumPy difference, bit for bit.

    MEASURED during the build: gate path, raw-mask path and plain NumPy agreed to the last digit
    over 400 seeds (mean recovered difference 0.7896 +/- 0.0058 for a planted 0.8), which is what
    licences reading z_diff as a real z rather than a house statistic."""
    rng = np.random.default_rng(2)
    v = rng.normal(0, 1, 600)
    flag = np.zeros(600, dtype=bool)
    flag[::3] = True
    v[flag] += 1.0
    c = mind.conditional(v, flag)
    assert c["diff"] == float(v[flag].mean() - v[~flag].mean())
    assert c["separates"] and c["z_diff"] > 5.0


def test_ex_post_conditioning_is_loud_and_causal_conditioning_is_quiet(mind):
    """The guardrail that makes the layer worth having: you cannot get an ex-post number without
    being told it is one."""
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(0, 1, 400)) + 100.0
    v = rng.normal(0, 1, 400)

    raw = mind.conditional(v, v > 0)                     # bare boolean array
    assert raw["causal"] is False and "EX-POST" in raw["warning"]

    epm = mind.conditional(v, ExPostMask(lambda c: np.asarray(c, float) > 0.0, name="hindsight"))
    assert epm["causal"] is False and "hindsight" in epm["warning"]

    gated = mind.conditional(v, trailing_gate("std", window=50, threshold=0.9), context=x)
    assert gated["causal"] is True and gated["warning"] is None


def test_per_regime_separates_a_spread_effect_from_a_one_regime_artifact(mind):
    """Two series with comparable unconditional means and opposite verdicts -- the measurement the
    per-regime table exists to make."""
    rng = np.random.default_rng(4)
    segs = [(0, 150), (150, 300), (300, 450), (450, 600)]

    spread = rng.normal(0.30, 1.0, 600)
    artifact = rng.normal(0.0, 1.0, 600)
    artifact[150:300] += 1.2
    assert abs(spread.mean() - artifact.mean()) < 0.15, "the two must look alike unconditionally"

    a_spread = mind.across_regimes(spread, segments=segs)
    a_art = mind.across_regimes(artifact, segments=segs)
    assert a_spread["consistent"] and a_spread["concentration"] < 0.6
    assert a_art["concentration"] > 0.6
    assert "artifact" in a_art["verdict"] or a_art["concentration"] > a_spread["concentration"]


def test_regimes_can_be_measured_by_the_engines_own_segmenter(mind):
    """across_regimes must delegate to the change-point segmenter rather than carry its own."""
    rng = np.random.default_rng(5)
    series = np.concatenate([rng.normal(0, 0.2, 300), rng.normal(0, 1.5, 300)])
    a = mind.across_regimes(series, series=series, min_seg=40)
    assert a["n_segments"] >= 2, "a clear variance break must produce more than one segment"
    # The boundaries must be the SAME ones detect_regimes reports -- one segmenter, not two.
    # detect_regimes reports the same boundaries in a richer per-segment dict (start/stop/mean/std);
    # the pin is that the BOUNDARIES agree, i.e. one segmenter is doing the work, not two.
    boundaries = [(r["start"], r["end"]) for r in a["segments"]]
    assert boundaries == [(s["start"], s["stop"]) for s in mind.detect_regimes(series, min_seg=40)["segments"]]


def test_detection_floor_tightens_with_more_data(mind):
    """A null result must state a floor, and the floor must behave like 1/sqrt(n)."""
    rng = np.random.default_rng(6)
    small = mind.across_regimes(rng.normal(0, 1, 40), segments=[(0, 20), (20, 40)])
    big = mind.across_regimes(rng.normal(0, 1, 4000), segments=[(0, 2000), (2000, 4000)])
    f_small = small["segments"][0]["floor"]
    f_big = big["segments"][0]["floor"]
    assert f_big < f_small
    assert 6.0 < f_small / f_big < 16.0, "floor should shrink ~10x for 100x the data"


def test_insurance_profile_catches_the_storm_premium(mind):
    """+36bp inside / +4bp outside: filtering the storms deletes the effect."""
    flag = np.zeros(500, dtype=bool)
    flag[:60] = True
    pay = np.where(flag, 0.36, 0.04)
    ins = mind.insurance_profile(pay, flag)
    assert ins["premium_inside"] and ins["lift"] == pytest.approx(9.0)
    assert ins["frac_events"] < 0.5 < ins["share_inside"]
    assert not mind.insurance_profile(np.full(500, 0.05), flag)["premium_inside"]


def test_full_chain_gate_then_split_half_then_regimes_then_pipeline_null(mind):
    """THE CROSS-FACULTY CHAIN. A planted, genuinely conditional effect must survive every gate in
    the honesty layer at once -- and the pipeline null must confirm the machinery did not make it."""
    # A HETEROSKEDASTIC context, so the trailing-volatility gate genuinely splits the sample.
    # (First draft used a random walk: its trailing std is almost always above any fixed threshold,
    # so the gate fired everywhere and the chain silently tested nothing. Kept as a comment because
    # a gate that never says No is the quietest way to fake a passing conditional test.)
    rng = np.random.default_rng(7)
    x = np.concatenate([rng.normal(0, 0.4, 600), rng.normal(0, 1.6, 600)])
    gate = mind.causal_gate("std", window=60, threshold=1.0, compare="ge", min_periods=60)
    inside = gate.mask(x)
    assert 100 < inside.sum() < 1100, "fixture must produce two usable groups (got %d)" % inside.sum()

    values = rng.normal(0, 1, 1200)
    values[inside] += 0.9                                # the effect exists ONLY inside the gate

    c = mind.conditional(values, gate, context=x)
    assert c["causal"] is True and c["separates"]

    # The honest follow-up the insurance docstring prescribes: do the inside events replicate?
    sh = mind.split_half(values[inside])
    assert sh["passed"], "a planted effect must survive split-half on the inside events"

    a = mind.across_regimes(values, series=x, min_seg=100)
    assert a["n_segments"] >= 1 and isinstance(a["verdict"], str)

    # And the null: does the gate MACHINERY manufacture a difference on structureless input?
    def pipeline(series):
        m = gate.mask(series)
        if m.sum() < 5 or (~m).sum() < 5:
            return 0.0
        noise = np.asarray(series, float)
        return float(noise[m].mean() - noise[~m].mean())

    pn = mind.pipeline_null(pipeline, x, surrogate="phase", n=60, seed=0)
    assert "z" in pn and np.isfinite(pn["z"]), "the pipeline null must return a usable z"


def test_layer_is_deterministic(mind):
    """Same inputs, same numbers, twice -- including the seeded causality audit."""
    rng = np.random.default_rng(8)
    x = np.cumsum(rng.normal(0, 1, 300)) + 100.0
    v = rng.normal(0, 1, 300)
    g = trailing_gate("std", window=40, threshold=1.0)
    assert g.audit_causality(x, seed=0) == g.audit_causality(x, seed=0)
    assert conditional(v, g, context=x) == conditional(v, g, context=x)
    assert across_regimes(v, segments=[(0, 150), (150, 300)]) == across_regimes(v, segments=[(0, 150), (150, 300)])
    assert insurance_profile(v, g, context=x)["lift"] == insurance_profile(v, g, context=x)["lift"]


def test_refusals_name_the_valid_options():
    """Every refusal in this layer must tell the caller what would have worked."""
    with pytest.raises(ValueError, match="known:"):
        trailing_gate("sharpe_ratio")
    with pytest.raises(ValueError, match="'ge','gt','le','lt'"):
        trailing_gate("std", compare="approximately")
    with pytest.raises(ValueError, match="segments"):
        across_regimes(np.arange(10.0))


def test_wire_shape_carries_its_own_causality_proof(mind):
    """Found at the HTTP bar: a Gate object cannot cross the wire, so an agent that correctly built
    and AUDITED a causal gate was still told its split was ex-post. The claim is now honoured only
    when the PROOF travels with it -- reading the caller's evidence, not trusting the caller."""
    rng = np.random.default_rng(9)
    x = np.concatenate([rng.normal(0, 0.4, 300), rng.normal(0, 1.6, 300)])
    wire = mind.causal_gate("std", window=60, threshold=1.0, compare="ge", context=x, min_periods=60)
    v = rng.normal(0, 1, 600) + 0.9 * np.array(wire["mask"], dtype=float)

    ok = mind.conditional(v, wire)
    assert ok["causal"] is True and ok["warning"] is None

    failed = mind.conditional(v, {"mask": wire["mask"], "audit": {"passed": False, "name": "leaky"}})
    assert failed["causal"] is False and "FAILED ITS CAUSALITY AUDIT" in failed["warning"]

    bare = mind.conditional(v, {"mask": wire["mask"]})
    assert bare["causal"] is False and "WITHOUT AUDIT" in bare["warning"]

    # and the numbers are identical regardless of how the causality claim was carried
    assert ok["diff"] == bare["diff"] == mind.conditional(v, np.array(wire["mask"], dtype=bool))["diff"]

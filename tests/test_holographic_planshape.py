"""Tests for holographic_planshape: schema-guided typed plans/records + descend (the IFMATCH generalization)."""

import numpy as np

from holographic.mesh_and_geometry.holographic_planshape import ShapeVocab, PlanNode, plan_shape, encode_record, decode_record, decode_record_confident, encode_plan, decode_plan, descend, _selftest

ACTIONS = ["advance", "hold", "retreat", "scan", "sample", "abort", "reroute", "wait"]
SCOPES = ["global", "local", "step", "mission"]


def _vocab():
    return ShapeVocab(1024, seed=0)


def _example_plan():
    return PlanNode("advance", "mission", branches={
        "blocked": PlanNode("reroute", "local", branches={
            "lowfuel": PlanNode("hold", "step"),
            "anomaly": PlanNode("scan", "local")}),
        "contact": PlanNode("hold", "global", branches={
            "degraded": PlanNode("abort", "mission")})})


def _example_shape():
    return plan_shape(ACTIONS, SCOPES,
                      {"blocked": {"lowfuel": {}, "anomaly": {}}, "contact": {"degraded": {}}})


def test_module_selftest():
    _selftest()


# ---- the general flat-record path (the "bring your own shape" / science case) --------------------

def test_record_round_trips_against_its_schema():
    v = _vocab()
    rec = {"phase": "stationary", "regime": "high_vol", "call": "hold"}
    schema = {"phase": ["stationary", "drifting"], "regime": ["low_vol", "high_vol"], "call": ACTIONS}
    assert decode_record(encode_record(rec, v), schema, v) == rec


def test_record_confidence_is_the_measured_cosine():
    v = _vocab()
    rec = {"a": "advance", "b": "hold"}
    schema = {"a": ACTIONS, "b": ACTIONS}
    vals, conf = decode_record_confident(encode_record(rec, v), schema, v)
    assert vals == rec
    assert all(0.0 < c <= 1.0 for c in conf.values())     # an honest, measured per-field cosine


# ---- the contingency-plan round-trip (schema-guided decode) --------------------------------------

def test_plan_round_trips_exactly_through_its_shape():
    v = _vocab()
    plan, shape = _example_plan(), _example_shape()
    back = decode_plan(encode_plan(plan, v), shape, v)
    assert back == plan                                    # structure + every action/scope label exact
    assert back.branches["blocked"].branches["anomaly"].action == "scan"
    assert 0.0 < back.confidence <= 1.0                    # confidence = the measured decode cosine


def test_schema_guided_decode_holds_a_deep_tree():
    # A known shape turns decode into clean unbinds, so it stays exact where the blind resonator parse would
    # crater. A 4-level chain of single branches round-trips every label.
    v = _vocab()
    leaf = PlanNode("wait", "step")
    plan = PlanNode("advance", "mission", branches={"c1": PlanNode("scan", "local", branches={
        "c2": PlanNode("hold", "global", branches={"c3": PlanNode("reroute", "local", branches={
            "c4": leaf})})})})
    shape = plan_shape(ACTIONS, SCOPES, {"c1": {"c2": {"c3": {"c4": {}}}}})
    back = decode_plan(encode_plan(plan, v), shape, v)
    assert back == plan


# ---- descend: the IFMATCH generalization (match a situation to a branch, abstain if none apply) --

def test_descend_walks_to_the_matching_branch():
    v = _vocab()
    vec, shape = encode_plan(_example_plan(), v), _example_shape()
    assert descend(vec, "blocked", shape, v) == ["advance", "reroute"]
    assert descend(vec, "contact", shape, v) == ["advance", "hold"]


def test_descend_abstains_when_no_branch_applies():
    v = _vocab()
    vec, shape = encode_plan(_example_plan(), v), _example_shape()
    # 'clear' matches no branch at the root -> abstain, returning only the node's primary action
    assert descend(vec, "clear", shape, v) == ["advance"]


def test_descend_matches_a_state_vector_not_just_a_name():
    v = _vocab()
    vec, shape = encode_plan(_example_plan(), v), _example_shape()
    state = v.value("blocked")                              # a state vector near the 'blocked' condition
    assert descend(vec, state, shape, v) == ["advance", "reroute"]
    noise = np.random.default_rng(0).standard_normal(1024)
    noise = noise / np.linalg.norm(noise)                  # an unrelated state -> below the floor -> abstain
    assert descend(vec, noise, shape, v) == ["advance"]


# ---- determinism (Macklin): identical inputs -> bit-identical vector and identical walk -----------

def test_encode_and_descend_are_deterministic():
    v = _vocab()
    plan, shape = _example_plan(), _example_shape()
    assert np.array_equal(encode_plan(plan, v), encode_plan(plan, v))
    vec = encode_plan(plan, v)
    assert descend(vec, "blocked", shape, v) == descend(vec, "blocked", shape, v)
    # the measured noise floor is also deterministic from the seed
    assert v.noise_floor() == v.noise_floor()

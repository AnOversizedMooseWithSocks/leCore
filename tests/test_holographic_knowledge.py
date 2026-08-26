"""D3: the findings registry as a holographic knowledge structure (holographic_knowledge.py) -- structured
research claims recalled by similarity, with the log's own contradictions detected and classified flat vs
conditioned."""
import numpy as np

from holographic.agents_and_reasoning.holographic_knowledge import FindingRegistry, _selftest


def test_selftest_passes():
    _selftest()


def _planted():
    reg = FindingRegistry(dim=2048, seed=1)
    i0 = reg.add("efficiency_ratio", "momentum", +1, condition="horizon_10d")
    i1 = reg.add("efficiency_ratio", "momentum", -1, condition="intraday")
    reg.add("low_vol", "vol_expansion", +1)
    i4 = reg.add("bracket_order", "convexity", +1)
    i5 = reg.add("bracket_order", "convexity", -1)
    reg.add("momentum", "trend", +1)             # momentum as SUBJECT, not object
    return reg, i0, i1, i4, i5


def test_query_recalls_by_subject():
    reg, i0, i1, *_ = _planted()
    top = reg.query(subject="efficiency_ratio", k=2)
    assert {r["index"] for r in top} == {i0, i1}


def test_query_is_role_sensitive():
    # object=momentum recalls findings where momentum is the OBJECT, not where it is the subject
    reg, i0, i1, *_ = _planted()
    got = {r["index"] for r in reg.query(object="momentum", floor=0.4)}
    assert i0 in got and i1 in got
    assert 5 not in got                          # finding 5 has momentum in the SUBJECT slot


def test_conditioned_tension_is_distinguished_from_flat_contradiction():
    reg, i0, i1, i4, i5 = _planted()
    tens = {(t["a"], t["b"]): t["type"] for t in reg.tensions()}
    assert tens.get((i0, i1)) == "conditioned"   # different conditions -> reconcilable
    assert tens.get((i4, i5)) == "flat"          # same/no condition -> must resolve


def test_no_false_positive_tensions():
    reg, *_ = _planted()
    tens = reg.tensions()
    assert len(tens) == 2                         # only the two genuine tensions, nothing spurious


def test_same_direction_is_not_a_tension():
    # two findings making the same claim with the SAME polarity agree -- not a tension
    reg = FindingRegistry(dim=2048, seed=2)
    reg.add("x", "y", +1, condition="a")
    reg.add("x", "y", +1, condition="b")
    assert reg.tensions() == []


def test_polarity_must_be_signed():
    reg = FindingRegistry(dim=512, seed=0)
    try:
        reg.add("x", "y", 0)
        assert False, "polarity 0 should be rejected"
    except ValueError:
        pass


def test_save_load_round_trips_findings_and_tensions(tmp_path):
    # the registry is a research LOG: saving and loading must reproduce it exactly
    reg, i0, i1, i4, i5 = _planted()
    before = {(t["a"], t["b"]): t["type"] for t in reg.tensions()}
    path = tmp_path / "findings.json"
    reg.save(str(path))
    reg2 = FindingRegistry.load(str(path))
    assert [dict(f) for f in reg2.findings] == [dict(f) for f in reg.findings]
    assert {(t["a"], t["b"]): t["type"] for t in reg2.tensions()} == before
    # query reproduces too (the conditioned-tension pair is still recalled by subject)
    assert {r["index"] for r in reg2.query(subject="efficiency_ratio", k=2)} == {i0, i1}


def test_load_rebuilds_vectors_from_seed_not_from_file(tmp_path):
    # the file stores ONLY structured claims -- no vectors -- and the rebuilt vectors are bit-identical,
    # so the reloaded registry is not an approximation but the same object (the demoscene / determinism rule)
    reg, *_ = _planted()
    path = tmp_path / "findings.json"
    reg.save(str(path))
    import json
    state = json.loads(path.read_text())
    assert set(state) == {"dim", "seed", "findings"}            # no vectors persisted
    assert "vec" not in json.dumps(state) and "_vecs" not in json.dumps(state)
    reg2 = FindingRegistry.load(str(path))
    assert all(np.array_equal(a, b) for a, b in zip(reg._vecs, reg2._vecs))
    assert all(np.array_equal(a, b) for a, b in zip(reg._claims, reg2._claims))


def test_loaded_registry_keeps_growing(tmp_path):
    # a durable log: reload, add more findings, and a new tension is detected against the restored ones
    reg = FindingRegistry(dim=2048, seed=3)
    reg.add("compression", "recall_margin", +1, condition="few_factors")
    path = tmp_path / "log.json"
    reg.save(str(path))
    reg2 = FindingRegistry.load(str(path))
    reg2.add("compression", "recall_margin", -1, condition="many_factors")   # opposite, different condition
    tens = reg2.tensions()
    assert len(tens) == 1 and tens[0]["type"] == "conditioned"


"""Tests for holographic_regimegate -- the RegimeGate re-enable pattern."""
from holographic_regimegate import RegimeGate


def _gate():
    return RegimeGate("double_when_large", detect=lambda x: abs(x), threshold=10.0,
                      superior=lambda x: x * 2, fallback=lambda x: x)


def test_superior_only_in_regime():
    g = _gate()
    out, info = g.apply(50.0)
    assert out == 100.0 and info["used"] == "superior" and info["score"] == 50.0


def test_fallback_outside_regime():
    g = _gate()
    out, info = g.apply(3.0)
    assert out == 3.0 and info["used"] == "fallback"


def test_borderline_biases_to_fallback():
    g = _gate()
    assert g.decide(9.999) == "fallback"      # just under -> safe default
    assert g.decide(10.0) == "superior"       # at threshold -> superior (above=True)


def test_above_false_flips_comparison():
    g = RegimeGate("small", detect=lambda x: abs(x), threshold=5.0,
                   superior=lambda x: 0.0, fallback=lambda x: x, above=False)
    assert g.apply(2.0)[1]["used"] == "superior"     # small -> superior
    assert g.apply(9.0)[1]["used"] == "fallback"     # large -> fallback


def test_info_records_measurement():
    g = _gate()
    _, info = g.apply(50.0)
    assert set(info) == {"gate", "score", "threshold", "used"} and info["threshold"] == 10.0

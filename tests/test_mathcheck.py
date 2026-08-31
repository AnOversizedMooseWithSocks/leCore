"""Arithmetic gets CHECKED by computing it, not by trusting the model.

A language model emits the token most likely to follow "137 * 4 = ", which is not
multiplication. It is right often enough to be trusted and wrong often enough to
matter, and NOTHING IN THE OUTPUT DISTINGUISHES THE TWO -- a wrong sum is written
with exactly the confidence of a right one.
"""

import pytest

from holographic.agents_and_reasoning import holographic_mathcheck as mc


def test_it_refuses_to_execute_model_text():
    """**eval() on model output is arbitrary code execution.**

    This is the property that makes the checker safe to point at untrusted text,
    so it is asserted before anything about correctness."""
    for hostile in ("__import__('os').system('id')", "open('/etc/passwd').read()",
                    "[].__class__.__mro__", "x + 1", "print(1)"):
        with pytest.raises(mc.Unverifiable):
            mc.evaluate(hostile)


def test_it_refuses_a_denial_of_service_expression():
    """`2**999999999` is valid arithmetic that never returns."""
    with pytest.raises(mc.Unverifiable):
        mc.evaluate("2 ** 999999999")
    with pytest.raises(mc.Unverifiable):
        mc.evaluate("1 / 0")


def test_it_catches_a_wrong_product_and_passes_a_right_one():
    bad = mc.check("Each crate holds 137 items and we have 4, so 137 * 4 = 549.")
    assert not bad["ok"] and len(bad["wrong"]) == 1, bad
    assert bad["wrong"][0]["computed"] == 548

    good = mc.check("137 * 4 = 548, and 12 + 30 = 42.")
    assert good["ok"] and good["checked"] == 2, good


def test_unverifiable_is_never_reported_as_wrong():
    """**"I could not check this" and "this is false" are different results.**

    Collapsing them makes the checker untrustworthy in both directions: it cries
    wolf on what it cannot parse, and a reader learns to ignore it."""
    r = mc.check("2 ** 999999999 = 1")
    assert r["ok"], "an unverifiable claim was reported as an error"
    assert len(r["unverifiable"]) == 1 and r["checked"] == 0, r


def test_float_error_is_not_a_model_error():
    """0.1 + 0.2 == 0.30000000000000004 is IEEE 754, not bad arithmetic."""
    assert mc.check("0.1 + 0.2 = 0.30000000000000004")["ok"]
    assert mc.check("0.1 + 0.2 = 0.3")["ok"], "relative tolerance not applied"
    assert not mc.check("0.1 + 0.2 = 0.4")["ok"], "tolerance is far too loose"


def test_an_assignment_is_not_an_arithmetic_claim():
    """`x = 5` has no computation in it -- matching it would flag every code block."""
    assert mc.check("let x = 5 and y = 12")["checked"] == 0


def test_a_sentence_ending_period_does_not_hide_a_claim():
    """The first regex forbade any following '.', so "12 + 30 = 42." matched
    NOTHING -- silently skipping every claim that ends a sentence, which is most
    of them. A period is not a decimal point unless a digit follows it."""
    r = mc.check("and 12 + 30 = 42.")
    assert r["checked"] == 1, r
    assert mc.check("12 + 30 = 42.5")["wrong"], "a real decimal must still parse"


def test_the_faculties_are_wired():
    import lecore

    m = lecore.UnifiedMind(dim=64)
    assert m.do_math("137 * 4") == 548
    assert not m.check_math("137 * 4 = 549")["ok"]

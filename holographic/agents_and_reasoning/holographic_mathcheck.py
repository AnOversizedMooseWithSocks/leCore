"""Check arithmetic by DOING it, instead of trusting a model that wrote it.

A language model produces the token most likely to follow "137 * 4 = ", which is
not the same operation as multiplying. It is right often enough to be dangerous
and wrong often enough to matter, and NOTHING IN THE OUTPUT DISTINGUISHES THE TWO
CASES -- a wrong sum is written with exactly the confidence of a right one.

So: find the arithmetic in a piece of text, evaluate it here with Python's own
integer and float arithmetic, and report every claim that does not hold. This
does not make the model better at maths. It makes the model's maths CHECKABLE,
which is the part that was missing.

WHY ast AND NOT eval(). `eval` on model output is arbitrary code execution from
an untrusted source. This walks an ast and permits exactly the arithmetic node
types -- numbers, + - * / // % **, unary minus, parentheses. A name, a call, an
attribute or a subscript is REFUSED rather than evaluated, so `__import__("os")`
is a parse failure and not a shell.

EXPONENT CAP. `2**999999999` is a valid arithmetic expression that hangs the
process computing a number nobody wants. Bounded, and a refused check is
reported as unverifiable rather than silently skipped -- abstain, do not lie.

FLOAT COMPARISON IS RELATIVE. 0.1 + 0.2 == 0.30000000000000004, and calling that
a model error would be blaming it for IEEE 754. Integers compare exactly; floats
compare within a relative tolerance, and the tolerance is reported so nobody has
to guess what "close" meant.
"""

import ast
import math
import operator
import re

# Only these node types are evaluated. Everything else is a refusal.
_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

MAX_EXP = 64          # 2**64 is plenty for prose; beyond it, refuse
MAX_DIGITS = 40       # a literal longer than this is an identifier, not a number
REL_TOL = 1e-9


class Unverifiable(Exception):
    """The expression could not be evaluated safely -- NOT a wrong answer."""


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise Unverifiable("not a number: %r" % (node.value,))
        if isinstance(node.value, int) and len(str(abs(node.value))) > MAX_DIGITS:
            raise Unverifiable("literal too long")
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            # AN EXPONENT IS A DENIAL-OF-SERVICE. 2**999999999 is arithmetic
            # that never returns; refusing is the only safe answer.
            if not isinstance(right, int) or abs(right) > MAX_EXP:
                raise Unverifiable("exponent out of bounds: %r" % (right,))
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise Unverifiable("division by zero")
        return _BIN[type(node.op)](left, right)
    raise Unverifiable("disallowed expression node: %s" % type(node).__name__)


def evaluate(expr):
    """Evaluate ONE arithmetic expression safely. Raises Unverifiable, never eval()s.

    DELIBERATELY SEPARATE from holographic_soprunner.safe_verify (read in the
    dedup sweep): same guarded-AST costume, different contracts -- see the
    reasoning on the name_collisions budget line for 'evaluate'."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise Unverifiable("not parseable: %s" % e) from None
    return _eval(tree)


def _close(got, claimed):
    """Exact for integers, relative for floats -- see the module note on 0.1+0.2."""
    if isinstance(got, int) and isinstance(claimed, int):
        return got == claimed
    if got == claimed:
        return True
    return math.isclose(float(got), float(claimed), rel_tol=REL_TOL,
                        abs_tol=1e-12)


# `expr = number`, where expr contains at least one operator. The operator
# requirement is what stops this matching "x = 5" and every assignment in a code
# block: a claim needs a COMPUTATION to be worth checking.
_CLAIM = re.compile(
    r"(?<![\w.])"
    r"((?:\d[\d,]*(?:\.\d+)?)(?:\s*[-+*/%]+\s*(?:\d[\d,]*(?:\.\d+)?|\([^()]{0,80}\)))+)"
    r"\s*(?:=|==|is|equals|gives)\s*"
    # `\.\d` not `\.`: a SENTENCE-ENDING PERIOD is not a decimal point. The first
    # version forbade any following ".", so "12 + 30 = 42." matched nothing --
    # which silently skips every claim that ends a sentence, i.e. most of them.
    r"(-?\d[\d,]*(?:\.\d+)?)(?![\w]|\.\d)",
    re.IGNORECASE)


def find_claims(text):
    """Every `expression = result` arithmetic claim in `text`.

    Returns [(expression, claimed_value, span)]. Commas are stripped from
    numbers because prose writes 1,234 and Python reads that as a tuple."""
    out = []
    for mm in _CLAIM.finditer(text or ""):
        expr = mm.group(1).replace(",", "")
        claimed = mm.group(2).replace(",", "")
        out.append((expr.strip(), claimed.strip(), mm.span()))
    return out


def check(text, tolerance=REL_TOL):
    """Verify every arithmetic claim in `text` by COMPUTING it.

    Returns {ok, checked, wrong, unverifiable, claims}. `ok` is False only when
    something was actually computed and disagreed -- an unverifiable claim is
    reported separately and NEVER counted as an error, because "I could not check
    this" and "this is wrong" are different results and collapsing them makes the
    checker untrustworthy in both directions."""
    claims, wrong, unver = [], [], []
    for expr, claimed_s, span in find_claims(text):
        rec = {"expr": expr, "claimed": claimed_s, "span": span}
        try:
            got = evaluate(expr)
            claimed = evaluate(claimed_s)
        except Unverifiable as e:
            rec["why"] = str(e)
            unver.append(rec)
            claims.append(rec)
            continue
        rec["computed"] = got
        rec["agrees"] = _close(got, claimed)
        if not rec["agrees"]:
            wrong.append(rec)
        claims.append(rec)
    return {"ok": not wrong, "checked": len(claims) - len(unver),
            "wrong": wrong, "unverifiable": unver, "claims": claims}


def _selftest():
    # the contract, not "no exception"
    assert evaluate("137 * 4") == 548
    assert evaluate("(2 + 3) * 4") == 20
    assert evaluate("7 // 2") == 3 and evaluate("7 % 2") == 1

    for bad in ("__import__('os')", "open('/etc/passwd')", "x + 1",
                "2 ** 999999999", "1/0", "[].__class__"):
        try:
            evaluate(bad)
            raise AssertionError("evaluated something it must refuse: %r" % bad)
        except Unverifiable:
            pass

    good = check("The total is 137 * 4 = 548, and 12 + 30 = 42.")
    assert good["ok"] and good["checked"] == 2, good

    bad = check("So 137 * 4 = 549 which we carry forward.")
    assert not bad["ok"] and len(bad["wrong"]) == 1, bad
    assert bad["wrong"][0]["computed"] == 548, bad["wrong"][0]

    # IEEE 754 is not a model error
    assert check("0.1 + 0.2 = 0.30000000000000004")["ok"]

    # an assignment is not a claim -- no operator, nothing to verify
    assert check("let x = 5 and y = 12")["checked"] == 0

    # unverifiable is not wrong
    u = check("2 ** 999999999 = 1")
    assert u["ok"] and len(u["unverifiable"]) == 1, u
    print("holographic_mathcheck selftest OK")


if __name__ == "__main__":
    _selftest()

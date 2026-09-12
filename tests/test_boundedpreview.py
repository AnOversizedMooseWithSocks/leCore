"""Regression traps for bounded object previews (sweep 122 item A, NOOA "pass by reference").

The claim being defended has two halves and both are cheap to break:

  1. A large result reaches an agent as type + TRUE size + a head/tail sample instead of a million JSON
     numbers, and the omitted middle stays reachable through the ObjectRefs handle minted beside it.
  2. NOTHING CHANGES for anyone who does not ask for it. `_jsonable(o)` and `_jsonable(o, refs)` must
     produce the same bytes they produced before the budget parameter existed -- so this file pins a
     GOLDEN corpus captured from the pre-change implementation, not merely "the two new call shapes agree
     with each other", which would pass even if both had drifted together.
"""
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, ".")
from holographic.io_and_interop.holographic_boundedpreview import (  # noqa: E402
    bounded_preview, json_bytes, preview_text)
from holographic.io_and_interop.holographic_objectref import ObjectRefs  # noqa: E402
from holographic_service import _json_default, _jsonable  # noqa: E402


class Opaque:
    def __repr__(self):
        return "<Opaque>"


# GOLDEN: every string here was produced by `json.dumps(_jsonable(v))` BEFORE the budget parameter was
# added (git HEAD 1803a4d). This is the byte-identity contract for the default path -- the repo's hard
# additive rule -- and it is a golden precisely so it cannot be satisfied by two new paths agreeing.
GOLDEN = [
    ("none", None, "null"),
    ("bool", True, "true"),
    ("int", 7, "7"),
    ("float", 1.5, "1.5"),
    ("zero", 0.0, "0.0"),
    ("nan", float("nan"), "null"),
    ("inf", float("inf"), "null"),
    ("str", "abc", '"abc"'),
    ("bigstr", "x" * 50, '"%s"' % ("x" * 50)),
    ("bytes", b"\x00\x01\x02", '{"__bytes_b64__": "AAEC"}'),
    ("list", [1, 2.5, "x", True, None], '[1, 2.5, "x", true, null]'),
    ("list_nonfinite", [1.0, float("inf")], "[1.0, null]"),
    ("tuple", (1, 2, 3), "[1, 2, 3]"),
    ("dict", {"a": [1, 2], "b": {"c": 3}}, '{"a": [1, 2], "b": {"c": 3}}'),
    ("dict_nonstr_key", {1: "a", (2, 3): "b"}, '{"1": "a", "(2, 3)": "b"}'),
    ("np_scalar", np.float64(2.5), "2.5"),
    ("np_int", np.int64(9), "9.0"),
    ("arr1d", np.arange(5, dtype=float), "[0.0, 1.0, 2.0, 3.0, 4.0]"),
    ("arr2d", np.arange(6, dtype=float).reshape(2, 3), "[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]"),
    ("nested", {"z": np.zeros((2, 2)), "w": [np.arange(3)]},
     '{"z": [[0.0, 0.0], [0.0, 0.0]], "w": [[0, 1, 2]]}'),
    ("opaque", Opaque(), '{"type": "Opaque", "repr": "<Opaque>"}'),
    ("empty_list", [], "[]"),
    ("empty_dict", {}, "{}"),
    ("empty_arr", np.zeros(0), "[]"),
]


@pytest.mark.parametrize("name,value,expected", GOLDEN, ids=[g[0] for g in GOLDEN])
def test_default_jsonable_is_byte_identical_to_before(name, value, expected):
    """THE HARD CONSTRAINT. Adding an opt-in budget must not move a single byte of the default response;
    an existing client sees exactly what it saw before, plus nothing."""
    assert json.dumps(_jsonable(value)) == expected
    assert json.dumps(_jsonable(value, None)) == expected
    assert json.dumps(_jsonable(value, None, None)) == expected, "budget=None IS the default path"


def test_a_budget_that_fits_returns_the_value_whole_and_unchanged():
    """A budget is a BOUND, not a filter. Below it the caller gets the same bytes as before, because the
    preview envelope costs 200-1400 bytes and bounding a small value would make the response BIGGER --
    the kept negative that decided this seam's shape."""
    small = np.arange(20, dtype=float)
    assert json.dumps(_jsonable(small, None, 4096)) == json.dumps(_jsonable(small))


def test_a_result_over_budget_comes_back_bounded_with_a_live_handle():
    """The measured claim, end to end through the service's own coercion: 1e6 floats are 20,269,744 bytes
    whole and ~364 bounded, and the omitted middle is still reachable -- identity, not a copy."""
    a = np.random.default_rng(0).random(1000000)
    refs = ObjectRefs()
    whole = json.dumps(_jsonable(a), default=_json_default, allow_nan=False)
    small = json.dumps(_jsonable(a, refs, 4096), default=_json_default, allow_nan=False)
    assert len(whole) > 20000000, len(whole)
    assert len(small) < 512, small
    d = json.loads(small)
    assert d["size"] == 1000000 and d["shape"] == [1000000], d
    assert refs.get(d["ref"]) is a, "without the handle the preview is a lossy dead end"


def test_the_true_length_is_reported_never_the_truncated_one():
    """The failure that would make previews worse than useless: an agent reading an invented length sizes
    its next call to a number the tool made up."""
    assert bounded_preview(list(range(5000)))["length"] == 5000
    assert bounded_preview({"k%d" % i: i for i in range(700)})["length"] == 700
    assert bounded_preview("x" * 9000)["length"] == 9000
    assert bounded_preview(np.zeros((40, 5)))["shape"] == [40, 5]


def test_nested_containers_are_bounded_recursively():
    """The case that actually hurts. Bounding only the outer level of 1000 lists of 1000 still ships six
    full inner lists -- measured at >40 kB against ~1.4 kB for the recursive bound."""
    nl = [[float(i * j) for j in range(1000)] for i in range(1000)]
    p = bounded_preview(nl)
    inner = p["head"][0]
    assert isinstance(inner, dict) and inner["type"] == "list" and inner["length"] == 1000
    assert inner["truncated"] and inner["omitted"] == 994
    assert p["bytes_preview"] < 2048 < 6 * json_bytes(nl[1])["bytes"]
    assert p["truncated"] is True


def test_an_ndarray_keeps_its_shape_instead_of_being_flattened():
    """A (1000, 3) array of points flattened to 3000 numbers loses the one fact needed to write the next
    call: that the rows are triples."""
    b = np.random.default_rng(1).random((1000, 3))
    p = bounded_preview(b)
    assert p["shape"] == [1000, 3] and p["size"] == 3000
    assert len(p["head"]) == 3 and all(len(row) == 3 for row in p["head"])
    assert p["shown"] == 18 and p["omitted"] == 2982
    assert p["head"][0][0] == float(b[0, 0]), "the sample must be the REAL values"


def test_max_bytes_is_a_bound_and_an_unmeetable_one_is_declared():
    """It must not overrun silently, and it must not claim to have fitted."""
    nl = [[float(j) for j in range(500)] for _ in range(500)]
    tight = bounded_preview(nl, max_bytes=300)
    assert tight["bytes_preview"] <= 300 and tight["length"] == 500
    assert bounded_preview(nl, max_bytes=10).get("budget_exceeded") is True


def test_a_tail_appears_only_when_something_was_actually_cut():
    """A tail is EVIDENCE of truncation; emitting one for a whole value is the same lie in a different
    shape, and a caller reading `tail` as 'the end exists elsewhere' would be misled."""
    whole = bounded_preview([1, 2, 3])
    assert "tail" not in whole and whole["truncated"] is False
    cut = bounded_preview(list(range(50)))
    assert cut["tail"] == [47, 48, 49] and cut["truncated"] is True


def test_the_cost_instrument_agrees_with_the_service_and_flags_its_own_estimates():
    """A win measured against a flattering baseline is not a win. json_bytes must report what the service
    would REALLY have sent, and must say when it is modelling rather than measuring."""
    for v in ([1.0, 2.5, "x", None], {"a": [1, 2, 3]}, np.arange(20, dtype=float), "hello"):
        c = json_bytes(v)
        assert c["exact"] and c["bytes"] == len(json.dumps(_jsonable(v))), (v, c)
    big = np.arange(10000, dtype=float) / 3.0
    est = json_bytes(big)
    assert est["exact"] is False
    exact = len(json.dumps(_jsonable(big)))
    assert abs(est["bytes"] - exact) / float(exact) < 0.08, (est, exact)


def test_previews_are_deterministic_and_json_safe():
    """PYTHONHASHSEED=0 discipline plus the service's own allow_nan=False: a preview that reintroduced a
    bare NaN would hand the caller an answer no other language's parser accepts."""
    v = {"a": np.array([1.0, np.nan, np.inf]), "b": [float("nan")] * 200}
    p1, p2 = bounded_preview(v), bounded_preview(v)
    assert json.dumps(p1) == json.dumps(p2)
    json.dumps(p1, allow_nan=False)          # raises ValueError if a non-finite survived


def test_kept_negative_bounding_a_small_value_costs_more_than_sending_it_whole():
    """PINNED SO IT CANNOT BE OPTIMISED AWAY. The envelope is not free: below the measured crossover
    (~16 floats, ~10 dict keys, ~200 characters) the preview is LARGER than the value. This is the whole
    reason the /invoke seam bounds only what is already over budget."""
    tiny = np.random.default_rng(2).random(5)
    assert bounded_preview(tiny)["bytes_preview"] > json_bytes(tiny)["bytes"]
    assert bounded_preview({"a": 1})["bytes_preview"] > json_bytes({"a": 1})["bytes"]


def test_the_budget_seam_only_mints_a_handle_when_it_actually_bounds():
    """The churn this seam adds, pinned. Every bounded result takes a slot in a 512-object registry with
    oldest-first eviction, so a budget that fired on every call would evict the handles the Scene family
    lives behind. A result that FITS must mint NOTHING -- that is what keeps the common case free."""
    fits, over = np.arange(10, dtype=float), np.arange(10000, dtype=float)
    r = ObjectRefs()
    for _ in range(50):
        _jsonable(fits, r, 4096)
    assert r.stats()["minted"] == 0, "a value under budget must not touch the registry"
    for _ in range(50):
        _jsonable(over, r, 4096)
    assert r.stats()["minted"] == 50, "exactly one handle per bounded result, never one per ladder rung"


def test_the_prompt_line_carries_the_length_and_the_handle():
    """A dict is what a program consumes; a line is what a model reads. If the true size and the handle do
    not survive into the text, the property was bought and then thrown away at the last step."""
    refs = ObjectRefs()
    p = bounded_preview(np.zeros(4321), refs=refs)
    line = preview_text(p)
    assert "4321" in line and p["ref"] in line and "float64" in line


def test_the_mind_exposes_both_faculties_and_delegates():
    """A module reachable only by import is a gap: if find_capability cannot surface it and /invoke cannot
    call it, it does not exist."""
    import lecore
    m = lecore.UnifiedMind(dim=64, seed=0)
    p = m.bounded_preview(np.arange(100000, dtype=float), max_bytes=1024)
    assert p["size"] == 100000 and p["bytes_preview"] <= 1024
    assert m.value_cost([1, 2, 3]) == json_bytes([1, 2, 3])

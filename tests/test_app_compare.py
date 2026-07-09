"""The showcase app's compare panel: WHY two sprites are similar, on real
images, with the SEE->SAY cross-modal loop. The full library build is ~1 min
(measured, dim 2048), so the test shrinks SPRITES to two families -- same code
path, seconds instead."""
import numpy as np
import pytest


def test_compare_endpoint_explains_and_sees():
    import app as A
    if not A.SPRITES:
        pytest.skip("sprite asset not present")
    keep = {n: v for n, v in A.SPRITES.items()
            if n.startswith(("amg", "knt"))}
    saved = (A.SPRITES, dict(A._COMPARE))
    A.SPRITES = keep
    A._COMPARE.update({"mind": None, "recs": {}, "rgbs": {}})
    try:
        c = A.app.test_client()
        r = c.post("/api/compare",
                   data={"a": "amg1_lf1.gif", "b": "amg1_lf2.gif"}).get_json()
        verdict = {x["role"]: x["shared"] for x in r["rows"]}
        # adjacent walk frames: same character, same direction, different frame
        assert verdict["family"] is True and verdict["facing"] is True
        assert verdict["frame"] is False
        # SEE->SAY: shown the image with no name, the mind recognises and speaks
        for name in (r["a"], r["b"]):
            s = r["seesay"][name]
            assert s["colour"] == A._COMPARE["recs"][s["matched"]]["colour"]
        # random pair works and returns two distinct sprites
        r2 = c.post("/api/compare", data={}).get_json()
        assert r2["a"] != r2["b"] and len(r2["rows"]) >= 2
        # the names endpoint feeds the datalist
        n = c.get("/api/sprite_names").get_json()
        assert "amg1_lf1.gif" in n["names"]
    finally:
        A.SPRITES, comp = saved
        A._COMPARE.update(comp)


def test_plan_endpoint_discovers_proves_executes():
    # The app's plan panel exposes the full sequence pipeline end-to-end:
    # discover which classes are ordered, prove them, recover canonical order,
    # and execute with the honest contract (in-order fires, out-of-order blocks).
    import app as A
    c = A.app.test_client()
    j = c.post("/api/plan").get_json()
    # the three procedures are discovered ordered; the two bags are not
    seq = {n for n, info in j["classes"].items() if info["sequential"]}
    assert "make_tea" in seq and "send_email" in seq and "do_laundry" in seq
    assert "spices" not in seq and "tools" not in seq
    assert all(j["classes"][n]["executable"] for n in seq)
    # execution: in-order all fire, out-of-order has a block
    run = next(iter(j["execution"].values()))
    assert all(s["status"] == "fired" for s in run["in_order"])
    assert any(s["status"] == "blocked" for s in run["out_of_order"])

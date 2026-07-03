"""Modeling-app backlog C+D: the per-object modifier stack + dependency graph (O(change) re-eval) + introspection."""
import numpy as np
from holographic_modifier import ModifierStack, describe_object


def _counting_ops():
    calls = {"n": 0}
    def add_op(x, amount=0.0):
        calls["n"] += 1; return x + amount
    def mul_op(x, factor=1.0):
        calls["n"] += 1; return x * factor
    return calls, add_op, mul_op


def test_nondestructive_fold():
    _, add_op, mul_op = _counting_ops()
    st = ModifierStack(0.0)
    st.add("a", add_op, {"amount": 1.0}); st.add("b", mul_op, {"factor": 2.0})
    st.add("c", add_op, {"amount": 10.0}); st.add("d", mul_op, {"factor": 3.0})
    assert st.evaluate() == 36.0 and st.base == 0.0        # ((0+1)*2+10)*3, base untouched


def test_o_change_recomputes_only_downstream():
    calls, add_op, mul_op = _counting_ops()
    st = ModifierStack(0.0)
    st.add("a", add_op, {"amount": 1.0}); st.add("b", mul_op, {"factor": 2.0})
    h2 = st.add("c", add_op, {"amount": 10.0}); st.add("d", mul_op, {"factor": 3.0})
    st.evaluate(); calls["n"] = 0
    st.set_param(h2, amount=20.0)
    assert st.evaluate() == 66.0 and calls["n"] == 2       # only mods 2,3 re-ran


def test_change_at_base_reruns_all():
    calls, add_op, mul_op = _counting_ops()
    st = ModifierStack(0.0)
    h0 = st.add("a", add_op, {"amount": 1.0}); st.add("b", mul_op, {"factor": 2.0})
    st.add("c", add_op, {"amount": 10.0}); st.add("d", mul_op, {"factor": 3.0})
    st.evaluate(); calls["n"] = 0
    st.set_param(h0, amount=5.0)
    assert st.evaluate() == 60.0 and calls["n"] == 4


def test_mute_skips_without_removing():
    calls, add_op, mul_op = _counting_ops()
    st = ModifierStack(0.0)
    st.add("a", add_op, {"amount": 5.0}); h1 = st.add("b", mul_op, {"factor": 2.0})
    st.add("c", add_op, {"amount": 20.0}); st.add("d", mul_op, {"factor": 3.0})
    st.evaluate()
    st.set_muted(h1, True)
    assert st.evaluate() == 75.0 and h1 in st.handles()   # skipped but still present


def test_reorder_insert_remove_stable_handles():
    _, add_op, mul_op = _counting_ops()
    st = ModifierStack(0.0)
    ha = st.add("a", add_op, {"amount": 1.0}); hb = st.add("b", mul_op, {"factor": 2.0})
    st.move(hb, 0); assert st.names()[0] == "b" and {ha, hb} == set(st.handles())
    hx = st.insert(1, "x", add_op, {"amount": 9.0}); assert hx in st.handles()
    st.remove(hx); assert hx not in st.handles()


def test_describe_and_validate():
    _, add_op, _ = _counting_ops()
    st = ModifierStack(0.0)
    h = st.add("bevel", add_op, {"amount": 0.3},
               specs={"amount": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0}})
    sch = st.describe(h)
    assert sch[0]["name"] == "amount" and sch[0]["max"] == 1.0
    assert st.validate() == []
    st.set_param(h, amount=5.0)
    assert any("above max" in p for p in st.validate())


def test_describe_object():
    class Obj:
        name = "wheel"; material = "metal"; tags = {"kind": "round"}; params = {"radius": 0.5}
    sch = describe_object(Obj())
    names = [e["name"] for e in sch]
    assert "name" in names and "tag:kind" in names and "radius" in names


def test_deterministic():
    _, add_op, mul_op = _counting_ops()
    def build():
        s = ModifierStack(0.0); s.add("a", add_op, {"amount": 2.0}); s.add("b", mul_op, {"factor": 4.0})
        return s.evaluate()
    assert build() == build()

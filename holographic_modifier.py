"""holographic_modifier.py -- the per-object MODIFIER STACK + dependency graph (modeling-app backlog, items C + D).

A PROMOTION, not a fresh build. leCore already has the substrate:
  * holographic_recipe.StructureRecipe is an ordered, NON-DESTRUCTIVE op sequence with stable handles;
  * holographic_recipeops gives it validate / reorder / substitute / commute;
  * holographic_dirtyfield is the O(change) "recompute only what changed" idea.
Together those ARE a modifier stack -- for VSA structures. This module carries the same pattern to a modeling
app's per-object modifier stack over ANY payload (a mesh, a field, a vector), and adds the one thing a DEPENDENCY
GRAPH needs that a plain recipe doesn't:

  O(change) RE-EVALUATION. When you tweak a modifier's parameter, only the modifiers BELOW it re-run; the results
  ABOVE are reused from cache. "Each modifier depends on the one before, so recompute only downstream of a change"
  is exactly the dependency graph (Maya/Houdini's dep-graph, Blender's modifier stack) -- and it is
  holographic_dirtyfield's O(change) principle applied to a linear op chain.

NON-DESTRUCTIVE: the base is never mutated; the stack PRODUCES the result by folding the ops over it. Handles are
STABLE (a modifier keeps its handle across reorder / insert / remove), so a property panel or an animation can
target one specific modifier. `describe` (item D) enumerates a modifier's parameters as a schema -- the property
panel's introspection, which in VSA terms is just enumerate-the-roles of a record.

The op contract (kept simple and old-school): a modifier's op is `op(payload, **params) -> new_payload` and must
NOT mutate its input -- it returns a new payload, the way a non-destructive modifier should. Deterministic; NumPy
+ stdlib only.
"""


class Modifier:
    """One entry in the stack: a named operation with parameters, applied non-destructively to the previous
    result. `muted` skips it WITHOUT removing it (the eye-icon toggle). `specs` optionally declares each param's
    type/default/min/max for the property panel; without it the schema is inferred from the value."""

    def __init__(self, handle, name, op, params=None, specs=None, muted=False):
        self.handle = handle
        self.name = name
        self.op = op                                    # op(payload, **params) -> new payload (must not mutate)
        self.params = dict(params) if params else {}
        self.specs = dict(specs) if specs else {}       # optional: name -> {type, default, min, max}
        self.muted = bool(muted)


class ModifierStack:
    """A per-object modifier stack: a base payload + an ordered list of modifiers, evaluated non-destructively,
    re-evaluated O(change) (only downstream of a change), with stable handles and validation. This is the modifier
    stack AND the dependency graph -- the linear dependency (each modifier consumes the previous result) is the
    graph, and the dirty frontier is how the graph avoids recomputing what didn't change."""

    def __init__(self, base):
        self.base = base
        self._mods = []              # ordered list of Modifier
        self._cache = []             # cache[i] = the payload AFTER modifier i (parallel to _mods)
        self._dirty_from = 0         # the O(change) frontier: earliest index that must be recomputed (0 = all)
        self._n = 0

    # -- lookups ---------------------------------------------------------------------------------------------
    def _index(self, handle):
        for i, m in enumerate(self._mods):
            if m.handle == handle:
                return i
        raise KeyError("no such modifier: %r" % handle)

    def handles(self):
        return [m.handle for m in self._mods]

    def names(self):
        return [m.name for m in self._mods]

    def _touch(self, index):
        """Move the dirty frontier no later than `index` -- everything from here down must be recomputed, the rest
        is reused from cache. This is the whole trick behind O(change)."""
        self._dirty_from = min(self._dirty_from, max(0, index))

    # -- building / editing the stack (each records where recomputation must restart) ------------------------
    def add(self, name, op, params=None, specs=None, muted=False):
        """Append a modifier to the top of the stack; returns its stable handle."""
        h = "mod_%08d" % self._n
        self._n += 1
        self._mods.append(Modifier(h, name, op, params, specs, muted))
        self._cache.append(None)
        self._touch(len(self._mods) - 1)
        return h

    def insert(self, index, name, op, params=None, specs=None, muted=False):
        """Insert a modifier at `index` (everything from there down must recompute)."""
        h = "mod_%08d" % self._n
        self._n += 1
        index = max(0, min(index, len(self._mods)))
        self._mods.insert(index, Modifier(h, name, op, params, specs, muted))
        self._cache.insert(index, None)
        self._touch(index)
        return h

    def remove(self, handle):
        i = self._index(handle)
        del self._mods[i]
        del self._cache[i]
        self._touch(i)
        self._dirty_from = min(self._dirty_from, len(self._mods))   # keep the frontier in range

    def move(self, handle, to_index):
        """Reorder a modifier (a real modeling operation -- bevel-then-subdivide differs from the reverse). Stable
        handle; recompute from the earliest touched position down."""
        i = self._index(handle)
        m = self._mods.pop(i)
        self._cache.pop(i)
        to_index = max(0, min(to_index, len(self._mods)))
        self._mods.insert(to_index, m)
        self._cache.insert(to_index, None)
        self._touch(min(i, to_index))

    def set_muted(self, handle, muted):
        i = self._index(handle)
        self._mods[i].muted = bool(muted)
        self._touch(i)

    def set_param(self, handle, **params):
        """Change a modifier's parameters -- the common case a dependency graph optimises. Marks this modifier and
        everything below it dirty; the modifiers above are untouched."""
        i = self._index(handle)
        self._mods[i].params.update(params)
        self._touch(i)

    # -- evaluation (non-destructive, O(change)) -------------------------------------------------------------
    def evaluate(self):
        """Fold the stack over the base, non-destructively. Recomputes ONLY from the dirty frontier down; the
        cached results above are reused. Returns the final payload (or the base if the stack is empty)."""
        start = self._dirty_from
        payload = self.base if start == 0 else self._cache[start - 1]
        for i in range(start, len(self._mods)):
            m = self._mods[i]
            if not m.muted:
                payload = m.op(payload, **m.params)     # a fresh payload each step (the op must not mutate its input)
            self._cache[i] = payload                    # a muted modifier just passes the payload through
        self._dirty_from = len(self._mods)              # everything is now clean
        return self._cache[-1] if self._mods else self.base

    # -- validation ------------------------------------------------------------------------------------------
    def validate(self):
        """Well-formedness (like recipeops.validate): every op callable, and every declared param within its
        min/max. Returns a list of problem strings (empty == valid)."""
        problems = []
        for m in self._mods:
            if not callable(m.op):
                problems.append("%s (%s): op is not callable" % (m.handle, m.name))
            for pname, spec in m.specs.items():
                if pname in m.params:
                    v = m.params[pname]
                    if "min" in spec and v < spec["min"]:
                        problems.append("%s.%s = %r below min %r" % (m.name, pname, v, spec["min"]))
                    if "max" in spec and v > spec["max"]:
                        problems.append("%s.%s = %r above max %r" % (m.name, pname, v, spec["max"]))
        return problems

    # -- parameter introspection (item D) --------------------------------------------------------------------
    def describe(self, handle):
        """The property-panel schema for one modifier: each parameter's name, type, current value, and (if the
        modifier declared specs) default/min/max. In VSA terms, enumerate-the-roles of the modifier record."""
        m = self._mods[self._index(handle)]
        return _param_schema(m.params, m.specs)


def _param_schema(params, specs=None):
    """Turn a params dict (+ optional specs) into a list of {name, type, value, default, min, max} entries -- the
    generic 'list a record's roles' introspection behind a property panel."""
    specs = specs or {}
    schema = []
    for name, val in params.items():
        spec = specs.get(name, {})
        schema.append({
            "name": name,
            "type": spec.get("type", type(val).__name__),
            "value": val,
            "default": spec.get("default"),
            "min": spec.get("min"),
            "max": spec.get("max"),
        })
    return schema


def describe_object(obj):
    """Item D over a SceneObject: enumerate its editable roles (name, material, tags, params) as a schema for a
    property panel. `transform` and `geometry` are listed by type only (they aren't scalar params). Works on
    anything exposing name/material/tags/params."""
    schema = [
        {"name": "name", "type": "str", "value": getattr(obj, "name", None)},
        {"name": "material", "type": "material", "value": getattr(obj, "material", None)},
    ]
    for tag, val in getattr(obj, "tags", {}).items():
        schema.append({"name": "tag:" + tag, "type": type(val).__name__, "value": val})
    schema.extend(_param_schema(getattr(obj, "params", {})))
    return schema


def _selftest():
    """A stack folds non-destructively to the right result; changing a param recomputes ONLY downstream (O(change),
    the dependency-graph property); mute skips a modifier; reorder/insert/remove keep stable handles; describe
    lists the param schema; validate catches out-of-range; deterministic."""
    calls = {"n": 0}

    def add_op(x, amount=0.0):
        calls["n"] += 1
        return x + amount                                # returns a NEW value (non-destructive)

    def mul_op(x, factor=1.0):
        calls["n"] += 1
        return x * factor

    # a 4-modifier stack on base 0: +1, *2, +10, *3
    st = ModifierStack(base=0.0)
    h0 = st.add("offset", add_op, {"amount": 1.0}, specs={"amount": {"type": "float", "default": 0.0}})
    h1 = st.add("scale", mul_op, {"factor": 2.0})
    h2 = st.add("offset2", add_op, {"amount": 10.0})
    h3 = st.add("scale2", mul_op, {"factor": 3.0})

    # (1) correct result and NON-DESTRUCTIVE base: ((0+1)*2 + 10) * 3 = 36; base still 0
    assert st.evaluate() == 36.0
    assert st.base == 0.0
    assert calls["n"] == 4                               # all four ran the first time

    # (2) O(change): change the 3rd modifier's param -> only modifiers 2 and 3 re-run (not 0, 1)
    calls["n"] = 0
    st.set_param(h2, amount=20.0)                        # ((0+1)*2 + 20) * 3 = 66
    assert st.evaluate() == 66.0
    assert calls["n"] == 2, calls["n"]                   # ONLY the two downstream modifiers recomputed

    # (3) changing the TOP modifier re-runs everything below it
    calls["n"] = 0
    st.set_param(h0, amount=5.0)                         # ((0+5)*2 + 20) * 3 = 90
    assert st.evaluate() == 90.0
    assert calls["n"] == 4                               # all four (the change is at the base of the stack)

    # (4) MUTE skips a modifier without removing it, and only recomputes downstream
    calls["n"] = 0
    st.set_muted(h1, True)                               # skip the *2: (0+5 + 20) * 3 = 75
    assert st.evaluate() == 75.0
    assert calls["n"] == 2, calls["n"]                   # only mods 2,3 ran an op; the muted mod 1 makes no op call
    st.set_muted(h1, False)

    # (5) reorder keeps STABLE handles and changes the result (order matters)
    before = set(st.handles())
    st.move(h3, 0)                                       # move *3 to the bottom: (((0+5)*3)+... changes
    assert set(st.handles()) == before                  # same handles, new order
    assert st.names()[0] == "scale2"

    # (6) insert / remove
    hx = st.insert(1, "offset3", add_op, {"amount": 100.0})
    assert hx in st.handles()
    st.remove(hx)
    assert hx not in st.handles()

    # (7) describe (item D): the property-panel schema for a modifier
    st2 = ModifierStack(base=0.0)
    hh = st2.add("bevel", add_op, {"amount": 0.3}, specs={"amount": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0}})
    sch = st2.describe(hh)
    assert sch[0]["name"] == "amount" and sch[0]["type"] == "float" and sch[0]["max"] == 1.0

    # (8) validate catches an out-of-range param
    st2.set_param(hh, amount=5.0)                        # above max 1.0
    assert any("above max" in p for p in st2.validate())

    # (9) deterministic
    def build():
        s = ModifierStack(0.0)
        s.add("a", add_op, {"amount": 2.0}); s.add("b", mul_op, {"factor": 4.0})
        return s.evaluate()
    assert build() == build()

    print("holographic_modifier selftest OK: the stack folds non-destructively (base untouched, result 36); "
          "changing a mid-stack param recomputes ONLY the 2 modifiers below it (O(change) dependency graph), while "
          "changing the base modifier reruns all 4; mute skips without removing; reorder/insert/remove keep stable "
          "handles; describe lists the param schema; validate catches out-of-range")


if __name__ == "__main__":
    _selftest()

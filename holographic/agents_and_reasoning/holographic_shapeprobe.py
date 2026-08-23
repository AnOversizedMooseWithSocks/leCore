"""SHAPEPROBE -- what does this faculty ACTUALLY return, and how is it ACTUALLY called?

WHY THIS EXISTS, and it is the most expensive recurring failure on record. Six instrument errors in one
session, every one the same class -- a docstring's "Returns {a, b, c}" line names the return CONTENTS,
and it was read as the CALL SHAPE:

    route_or_abstain  -> hits is [(Capability, score)] TUPLES, not bare capabilities
    expand_query      -> `query` is the final TEXT, `expanded` is a BOOL
    fit_camera        -> a plain DICT of parameters, not a camera object with .ray_dirs()
    scene_light       -> PATH-TRACER area lights, which the rasteriser cannot consume
    material(P)       -> called with ONE argument; the docstring lists the 4-tuple it RETURNS

Each cost a crash or, worse, a silent wrong answer (a bool passed into a router; three sampled branches
handed one shared reply). The standing rule "probe the live object, never the docstring" was already
written down and was still violated six times, because probing meant hand-writing a throwaway script
each time. A rule that is expensive to follow is a rule that gets skipped. This makes it one call.

WHAT IT DOES NOT DO: guess, or execute anything. `signature_of` is pure introspection. `shape_of` runs
the callable ONLY on arguments you supply, so nothing fires without your say-so -- an auto-probe that
called faculties with invented arguments would be an arbitrary-execution tool wearing a helpful name.
"""
import inspect


def describe_shape(obj, depth=0, _max=3):
    """A compact, RECURSIVE description of a value's runtime shape -- the thing a docstring cannot promise.

    Containers report their element shape rather than just their type, because "list" is exactly the
    answer that was not useful: `hits` was a list, and the bug was what the list held. Dicts report their
    KEYS, since a returned dict is a record and its keys are its contract. numpy arrays report shape and
    dtype. Recursion stops at `_max` so a deep object graph cannot produce an unreadable wall."""
    if depth > _max:
        return "..."
    t = type(obj).__name__
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return t
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return "%s%s %s" % (t, tuple(obj.shape), obj.dtype)
    if isinstance(obj, dict):
        keys = list(obj)[:12]
        inner = ", ".join("%s=%s" % (k, describe_shape(obj[k], depth + 1, _max)) for k in keys)
        return "dict{%s%s}" % (inner, ", ..." if len(obj) > 12 else "")
    if isinstance(obj, tuple):
        # A TUPLE IS A RECORD, not a sequence: fixed arity, heterogeneous BY DESIGN, so every element is
        # described. Reporting "tuple[2 x mixed]" would hide the whole finding -- the bug was that `hits`
        # held (Capability, score) and the caller read the tuple as the capability.
        if not obj:
            return "tuple[empty]"
        inner = ", ".join(describe_shape(x, depth + 1, _max) for x in obj[:8])
        return "tuple(%s%s)" % (inner, ", ..." if len(obj) > 8 else "")
    if isinstance(obj, (list, set)):
        seq = list(obj)
        if not seq:
            return "%s[empty]" % t
        head = describe_shape(seq[0], depth + 1, _max)
        same = all(type(x) is type(seq[0]) for x in seq[:8])
        return "%s[%d x %s]" % (t, len(seq), head if same else "mixed")
    # A plain object: its PUBLIC ATTRIBUTES are what a consumer will reach for, and the light-family bug
    # was exactly "this object lacks .kind" -- so list them.
    attrs = [a for a in dir(obj) if not a.startswith("_")][:14]
    return "%s(%s)" % (t, ", ".join(attrs))


def signature_of(fn):
    """The call shape of `fn` -- parameter names, defaults, and arity. Pure introspection, nothing runs.

    Answers the half of the trap that `shape_of` cannot: `material(P)` took one argument while its
    docstring's tuple was read as four parameters. Arity is checkable without ever calling anything."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {"callable": callable(fn), "signature": None, "why": "no introspectable signature"}
    params = []
    for p in sig.parameters.values():
        params.append({"name": p.name, "kind": str(p.kind),
                       "default": None if p.default is inspect._empty else repr(p.default),
                       "required": p.default is inspect._empty
                       and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)})
    required = [p["name"] for p in params if p["required"]]
    return {"callable": True, "signature": str(sig), "params": params,
            "n_required": len(required), "required": required}


def shape_of(fn, *args, **kw):
    """Call `fn(*args, **kw)` ONCE and report the runtime shape of what comes back, beside its signature.

    The one call that answers "what will I actually get". Returns {signature, returns, error} -- and on a
    raised exception it reports the exception rather than propagating, because a probe that crashes tells
    you less than a probe that says WHICH call shape was wrong.

    NOTHING IS INVENTED: it runs on the arguments you pass and no others. This is deliberate -- a probe
    that auto-filled plausible arguments would be executing arbitrary faculties on guessed input, which
    is a different and much less welcome tool."""
    out = {"signature": signature_of(fn), "returns": None, "error": None}
    try:
        val = fn(*args, **kw)
    except Exception as e:
        out["error"] = "%s: %s" % (type(e).__name__, e)
        return out
    out["returns"] = describe_shape(val)
    return out


def _selftest():
    import numpy as np

    # 1. THE EXACT BUG THAT STARTED THIS: a list of (obj, score) tuples must NOT describe as merely "list".
    class Cap:
        def __init__(self, name):
            self.name = name
    hits = [(Cap("a"), 1.0), (Cap("b"), 0.5)]
    d = describe_shape(hits)
    assert d.startswith("list[2 x tuple("), d
    assert "Cap(" in d and "float" in d, \
        "a tuple is a RECORD -- each element must be named, or the (Capability, score) trap stays hidden"

    # 2. A returned dict is a RECORD -- its keys are its contract (the fit_camera case).
    cam = {"eye": np.zeros(3), "target": np.zeros(3), "up": (0, 1, 0), "fov_deg": 50.0, "aspect": 1.3}
    d2 = describe_shape(cam)
    assert d2.startswith("dict{") and "fov_deg=float" in d2 and "eye=ndarray(3,)" in d2, d2

    # 3. A plain object lists its PUBLIC ATTRIBUTES -- the light-family bug was "no attribute 'kind'".
    class RectLight:
        def __init__(self):
            self.position, self.width, self.height = (0, 0, 0), 1.0, 1.0
    d3 = describe_shape(RectLight())
    assert "RectLight(" in d3 and "kind" not in d3 and "width" in d3, d3

    # 4. ARITY, without calling anything -- the material(P) case.
    assert signature_of(lambda P: P)["n_required"] == 1
    assert signature_of(lambda P, N: P)["n_required"] == 2
    assert signature_of(lambda a, b=1, *c, **d: a)["required"] == ["a"]

    # 5. shape_of REPORTS a failure instead of raising -- a probe that crashes tells you less than one
    #    that names the wrong call shape. This is the whole point of using it to diagnose.
    bad = shape_of(lambda P, N: P, np.zeros((3, 3)))
    assert bad["error"] and "missing" in bad["error"], bad
    assert bad["signature"]["n_required"] == 2, "the signature must still be reported after a failed call"

    # 6. A BOOL is reported as a bool (the expand_query case: `expanded` was read as text).
    ok = shape_of(lambda q: {"query": "text", "expanded": True, "faithfulness": 1.0}, "q")
    assert "expanded=bool" in ok["returns"] and "query=str" in ok["returns"], ok

    # 7. Nothing runs without arguments you gave: signature_of never calls.
    fired = {"n": 0}
    def boom():
        fired["n"] += 1
        raise RuntimeError("should never run")
    signature_of(boom)
    assert fired["n"] == 0, "signature_of executed the callable"

    # 8. Empty containers are distinguishable from populated ones -- an empty `hits` on an abstain is a
    #    real state, not a missing one.
    assert describe_shape([]) == "list[empty]"
    print("holographic_shapeprobe selftest OK -- tuples-in-a-list named; dict keys are the contract; "
          "missing .kind visible; arity without calling; failures reported not raised; bool stays bool")


if __name__ == "__main__":
    _selftest()

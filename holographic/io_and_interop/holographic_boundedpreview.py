"""holographic_boundedpreview.py -- a BOUNDED, JSON-safe view of a value (NOOA's "pass by reference").

WHY THIS EXISTS, and it is a boundary bug with a measured price. `holographic_service._jsonable` already
solves HALF of NVIDIA-labs OO Agents' pass-by-reference property (arXiv:2607.20709, and see
docs/COMPETITIVE_NOOA.md section 6 item 1): an object JSON cannot carry comes back as a typed summary
plus an ObjectRefs HANDLE, so the agent names it in the next call. But for an object JSON CAN carry it
serialises the whole thing -- `np.ndarray -> o.tolist()`, lists recurse element by element -- so a faculty
returning a 1e6-element array returns 1e6 JSON numbers into a context window, which is the scarce resource
the handle was invented to protect. leCore had the handle and not the bound.

MEASURED ON THIS BOX (baseline = today's `_jsonable` on the SAME object, rendered with the service's own
`json.dumps(..., allow_nan=False)`; N is the leaf count; bounded = this module at max_bytes=4096):

    float ndarray  N=1e2    2,028 B  ->    322 B     6.3x        nested list-of-lists (MxM)
    float ndarray  N=1e3   20,244 B  ->    327 B    61.9x            10x10      2,045 B ->  1,391 B    1.5x
    float ndarray  N=1e4  202,636 B  ->    335 B   604.9x            32x32     20,869 B ->  1,399 B   14.9x
    float ndarray  N=1e5    2.03 MB  ->    337 B  6,016x            100x100   202,826 B ->  1,411 B  143.7x
    float ndarray  N=1e6   20.27 MB  ->    340 B  59,617x         1000x1000    20.27 MB ->  1,432 B  14,156x

The nested arm costs ~1.4 kB rather than ~0.34 kB because six inner previews each carry their own envelope;
that is the price of bounding recursively and it is still three orders of magnitude under the baseline.

WHAT A PREVIEW PROMISES (and what it does not)
----------------------------------------------
  * The TRUE length/shape, never the truncated one. A preview that reported len 3 for a 1e6 list would be
    worse than no preview at all: an agent would size its next call to a number the tool invented.
  * A head sample, and a tail sample ONLY when something was actually cut -- the tail is evidence of
    truncation, so emitting it for a value that fitted whole would be a lie in the same shape.
  * `truncated` is true if ANYTHING below was cut, not just this level. A list of 1000 lists of 1000 that
    reported truncated only at the outer level would hide 999,000 omitted numbers behind a "false".
  * The honest byte cost of BOTH renderings, so the saving is a number the caller can check rather than a
    claim this module makes about itself.

KEPT NEGATIVE -- THE FIDELITY THAT IS ACTUALLY LOST. A preview is lossy, full stop. Every element between
head and tail is gone from the response and no amount of formatting brings it back. The escape hatch is the
handle, not the preview: pass `refs=` (an ObjectRefs registry) and the reply carries `ref`, the live object
stays addressable in the service, and the agent can call another faculty on it -- statistics, a slice, a
save -- to see what the preview omitted. WITHOUT `refs` the preview is a DEAD END for the omitted middle,
and any caller that bounds a result it cannot also hand back a handle for has traded fidelity for bytes
with no way to buy it back. That is a real limitation and it is named here rather than hidden.

KEPT NEGATIVE -- BOUNDING DOES NOT PAY FOR SMALL VALUES, and the crossover is measured, not guessed. The
envelope (type, true length, byte accounting, flags) costs 200-1,400 bytes, so below it the whole value is
CHEAPER than its preview: a 15-float array is 308 B whole against 322 B previewed (a LOSS), and only at 20
floats (404 B whole, 318 B previewed) does bounding start to pay. Same shape elsewhere -- dicts cross over
between 10 and 100 keys, strings between 200 and 1,000 characters, and an 8x8 nested list is 1,308 B whole
against 1,382 B previewed, still a loss. That is why the service seam bounds ONLY a value already over its
budget and otherwise returns today's output byte for byte; bounding by reflex would inflate the common case.
Pinned by the selftest so it cannot be "optimised" away.

WHY NOT REUSE shapeprobe.describe_shape. It was the closest thing in the tree and it is a different
question: it reports SHAPE ("list[1000 x float]") and never samples a value, so it cannot answer "what is
actually in it". They compose -- shape for the contract, preview for the contents -- and neither replaces
the other.
"""
import base64
import json
import math

import numpy as np

DEFAULT_HEAD = 3
DEFAULT_TAIL = 3
DEFAULT_MAX_CHARS = 200
DEFAULT_DEPTH = 2
_MARKER = "...+%d"          # what an inner axis puts where it dropped elements; the count is the honesty
_SAMPLE_K = 32              # elements sampled when ESTIMATING a container's cost (see _estimate_bytes)


def _safe_scalar(x):
    """One scalar as JSON can carry it: non-finite floats become null, numpy scalars become Python ones.

    MIRRORS holographic_service._jsonable's non-finite rule deliberately, and the agreement is pinned by a
    test rather than by an import. json.dumps emits bare `NaN`/`Infinity`, which Python's lenient parser
    accepts and every other language's rejects -- a preview that reintroduced that would hand the caller an
    unparseable answer while claiming to have made one safe. Importing the service here instead would drag
    the whole HTTP layer into a leaf module and invert the dependency the service needs."""
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (bytes, bytearray)):
        return {"__bytes_b64__": base64.b64encode(bytes(x)).decode("ascii")}
    return repr(x)[:120]


def _jsonsafe(v):
    """A whole value coerced the way the service would coerce it -- used ONLY to MEASURE its byte cost.

    This is a cost instrument, not a second serialiser: nothing returns it to a caller. It mirrors
    `_jsonable(o, refs=None)` so that `json_bytes(...)['bytes']` is the number of bytes the service would
    really have sent, which is the only baseline worth comparing a preview against. A test pins the two
    renderings equal so this copy cannot drift into a flattering baseline."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (bytes, bytearray)):
        return {"__bytes_b64__": base64.b64encode(bytes(v)).decode("ascii")}
    if isinstance(v, (np.floating, np.integer)):
        f = float(v)
        return None if not math.isfinite(f) else f
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _jsonsafe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonsafe(x) for x in v]
    return {"type": type(v).__name__, "repr": repr(v)[:500]}


def _count_leaves(v, cap):
    """Leaf count, ABANDONED as soon as it passes `cap` (so the answer is exact or a lower bound > cap).

    The cap is the whole point: deciding whether a value is small enough to measure exactly must not cost a
    full walk of a value that is obviously not. An ndarray answers in O(1) from `.size`; a string is one
    leaf because it is serialised as one token."""
    if isinstance(v, np.ndarray):
        return int(v.size)
    if isinstance(v, (str, bytes, bytearray)) or v is None or isinstance(v, (bool, int, float)):
        return 1
    if isinstance(v, dict):
        items = list(v.values())
    elif isinstance(v, (list, tuple, set)):
        items = list(v)
    else:
        return 1
    total = 0
    for x in items:
        total += _count_leaves(x, cap - total)
        if total > cap:
            return total
    return total


def _sample_indices(n, k):
    """`k` evenly spaced indices in [0, n) -- DETERMINISTIC, no rng, because a cost estimate that changed
    between runs would break the repo's bit-reproducibility rule for anything that logs it."""
    if n <= k:
        return list(range(n))
    step = n / float(k)
    return sorted({int(i * step) for i in range(k)})


def _estimate_bytes(v):
    """Estimated JSON byte cost of a value too large to render, from a deterministic sample of its elements.

    WHY ESTIMATE AT ALL: materialising a 1e6-element rendering merely to report how big it is would pay
    exactly the cost the bound exists to avoid -- the instrument would become the disease. Accuracy is
    pinned by the selftest at within 15% of the exact figure for a 1e4-element array, which is far tighter
    than any decision made from it needs (the service seam only asks "is this over budget", and a value
    routed here has more leaves than the budget has bytes, so it is over regardless of the estimate)."""
    if isinstance(v, np.ndarray):
        size = int(v.size)
        if size == 0 or v.ndim == 0:
            return len(json.dumps(_safe_scalar(v[()]) if v.ndim == 0 else v.tolist()))
        flat = v.reshape(-1)
        idx = _sample_indices(size, 64)
        per = sum(len(json.dumps(_safe_scalar(flat[i]))) for i in idx) / float(len(idx))
        # Build the cost from the INNERMOST axis outwards with json.dumps's real separators (", " is TWO
        # bytes, not one -- assuming one under-counted a 1e4 integer array by 17.5%, and an estimator that
        # flatters the bounded case is exactly the strawman baseline this repo refuses to ship).
        block = per
        for d in reversed(list(v.shape)):
            block = 2 + d * block + 2 * (d - 1)
        return int(block)
    if isinstance(v, dict):
        n = len(v)
        if n == 0:
            return 2
        keys = list(v)
        idx = _sample_indices(n, _SAMPLE_K)
        per = sum(len(json.dumps(str(keys[i]))) + 2 + json_bytes(v[keys[i]])["bytes"] for i in idx)
        return int(2 + per / float(len(idx)) * n + 2 * (n - 1))
    if isinstance(v, (list, tuple, set)):
        seq = list(v)
        n = len(seq)
        if n == 0:
            return 2
        idx = _sample_indices(n, _SAMPLE_K)
        per = sum(json_bytes(seq[i])["bytes"] for i in idx) / float(len(idx))
        return int(2 + n * per + 2 * (n - 1))
    return len(json.dumps(_jsonsafe(v)))


def json_bytes(value, exact_below=2048):
    """How many response bytes would this value REALLY cost an agent's context? -> {bytes, exact, leaves}.

    The baseline half of every preview claim. `bytes` is the length of the JSON the service would send for
    `value` (the same coercion `_jsonable` performs), measured exactly when the value has at most
    `exact_below` leaves and estimated from a deterministic sample above that. `exact` says which happened,
    so a caller never has to guess whether a number is measured or modelled. `leaves` is the leaf count --
    exact when `exact` is true, otherwise a lower bound greater than `exact_below`.

    Use it before returning a big result, or to check a preview's saving yourself instead of believing the
    `saved` field this module computes about its own work."""
    n = _count_leaves(value, exact_below)
    if n <= exact_below:
        try:
            return {"bytes": len(json.dumps(_jsonsafe(value))), "exact": True, "leaves": n}
        except (TypeError, ValueError):
            # An un-dumpable value still HAS a cost at the boundary -- the service would send its typed
            # summary -- so fall through to the estimator rather than reporting nothing.
            pass
    return {"bytes": int(_estimate_bytes(value)), "exact": False, "leaves": n}


def _count_shown(sample):
    """How many real leaf values a nested sample actually carries (the `...+n` markers are not values).

    `shown` is what makes `omitted` honest for a multi-axis array: a (1000, 3) array previewed as 3 rows of
    3 columns has shown 9 of 3000, and reporting the row count alone would understate the loss 3-fold."""
    if isinstance(sample, list):
        return sum(_count_shown(x) for x in sample)
    if isinstance(sample, str) and sample.startswith("...+"):
        return 0
    return 1


def _ndarray_sample(a, head, tail):
    """Head/tail of an ndarray along EVERY axis, keeping its nesting -- never a flatten.

    A (1000, 3) array flattened to 3000 numbers loses the fact that the rows are points, which is the one
    thing a caller needs to write the next call. Inner axes carry `...+n` where they were cut, because an
    inner axis has no separate head/tail keys to put the count in."""
    if a.ndim == 0:
        return _safe_scalar(a[()])
    n = int(a.shape[0])
    if n <= head + tail:
        return [_ndarray_sample(a[i], head, tail) for i in range(n)]
    out = [_ndarray_sample(a[i], head, tail) for i in range(head)]
    out.append(_MARKER % (n - head - tail))
    out.extend(_ndarray_sample(a[i], head, tail) for i in range(n - tail, n))
    return out


def _child_truncated(x):
    """True if a rendered child is itself a preview that reports a cut. Truncation must propagate UPWARDS
    or the outer flag lies about everything nested below it."""
    return isinstance(x, dict) and bool(x.get("truncated"))


def _element(v, head, tail, max_chars, depth):
    """One element inside a head/tail sample: inlined when it is small, a NESTED preview when it is not.

    THIS IS THE RECURSION THAT MATTERS. Bounding only the outer level of a list of 1000 lists of 1000 fixes
    nothing -- three elements of a million still costs three thousand numbers. Depth is bounded too: at
    depth 0 a container degrades to a type+length stub rather than opening one more level, because an
    unbounded descent is just the original problem wearing a preview's clothes."""
    if isinstance(v, np.ndarray) or isinstance(v, (dict, list, tuple, set)):
        if depth <= 0:
            return _stub(v)
        return _preview(v, head, tail, max_chars, depth - 1)
    if isinstance(v, str) and len(v) > max_chars:
        return _preview(v, head, tail, max_chars, 0)
    if isinstance(v, (bytes, bytearray)) and len(v) > max_chars:
        return _preview(v, head, tail, max_chars, 0)
    if v is None or isinstance(v, (bool, int, float, str, bytes, bytearray,
                                   np.floating, np.integer, np.bool_)):
        return _safe_scalar(v)
    return _preview(v, head, tail, max_chars, 0)


def _stub(v):
    """The terse form a container takes once the depth budget is spent: type and true size, nothing else."""
    if isinstance(v, np.ndarray):
        return {"type": "ndarray", "shape": list(v.shape), "dtype": str(v.dtype), "size": int(v.size)}
    return {"type": type(v).__name__, "length": len(v)}


def _preview(v, head, tail, max_chars, depth):
    """The bounded description of one value at one level. See `bounded_preview` for the public entry."""
    t = type(v).__name__

    if isinstance(v, np.ndarray):
        size = int(v.size)
        out = {"type": "ndarray", "dtype": str(v.dtype), "shape": list(v.shape), "size": size}
        if v.ndim == 0:
            out.update({"head": _safe_scalar(v[()]), "truncated": False, "omitted": 0, "shown": 1})
            return out
        n0 = int(v.shape[0])
        h = min(head, n0)
        tl = min(tail, max(0, n0 - h))
        head_rows = [_ndarray_sample(v[i], head, tail) for i in range(h)]
        tail_rows = [_ndarray_sample(v[i], head, tail) for i in range(n0 - tl, n0)] if tl else []
        shown = _count_shown(head_rows) + _count_shown(tail_rows)
        out["head"] = head_rows
        out["shown"] = shown
        out["omitted"] = size - shown
        out["truncated"] = out["omitted"] > 0
        if out["truncated"] and tail_rows:
            out["tail"] = tail_rows          # a tail is EVIDENCE of a cut; emitting one otherwise misleads
        return out

    if isinstance(v, str):
        n = len(v)
        if n <= max_chars:
            return {"type": "str", "length": n, "head": v, "truncated": False, "omitted": 0}
        tc = min(max(1, max_chars // 4), n - max_chars)
        return {"type": "str", "length": n, "head": v[:max_chars], "tail": v[n - tc:],
                "truncated": True, "omitted": n - max_chars - tc}

    if isinstance(v, (bytes, bytearray)):
        n = len(v)
        k = max(1, max_chars // 2)
        if n <= k:
            return {"type": t, "length": n, "truncated": False, "omitted": 0,
                    "head_b64": base64.b64encode(bytes(v)).decode("ascii")}
        tc = min(max(1, k // 4), n - k)
        # base64 of a byte PREFIX, so the head is decodable on its own rather than a lossy rendering.
        return {"type": t, "length": n, "truncated": True, "omitted": n - k - tc,
                "head_b64": base64.b64encode(bytes(v[:k])).decode("ascii"),
                "tail_b64": base64.b64encode(bytes(v[n - tc:])).decode("ascii")}

    if isinstance(v, dict):
        keys = list(v)
        n = len(keys)
        h = min(head, n)
        tl = min(tail, max(0, n - h))
        head_items = {str(keys[i]): _element(v[keys[i]], head, tail, max_chars, depth) for i in range(h)}
        tail_items = {str(keys[i]): _element(v[keys[i]], head, tail, max_chars, depth)
                      for i in range(n - tl, n)} if tl else {}
        out = {"type": t, "length": n, "head": head_items, "omitted": n - h - tl}
        out["truncated"] = (out["omitted"] > 0
                            or any(_child_truncated(x) for x in list(head_items.values()) + list(tail_items.values())))
        if out["truncated"] and tail_items:
            out["tail"] = tail_items
        return out

    if isinstance(v, (list, tuple, set)):
        seq = list(v)
        n = len(seq)
        h = min(head, n)
        tl = min(tail, max(0, n - h))
        head_items = [_element(seq[i], head, tail, max_chars, depth) for i in range(h)]
        tail_items = [_element(seq[i], head, tail, max_chars, depth) for i in range(n - tl, n)] if tl else []
        out = {"type": t, "length": n, "head": head_items, "omitted": n - h - tl}
        out["truncated"] = out["omitted"] > 0 or any(_child_truncated(x) for x in head_items + tail_items)
        if out["truncated"] and tail_items:
            out["tail"] = tail_items
        return out

    if v is None or isinstance(v, (bool, int, float, np.floating, np.integer, np.bool_)):
        return {"type": t, "head": _safe_scalar(v), "truncated": False, "omitted": 0}

    r = repr(v)
    return {"type": t, "length": len(r), "head": r[:max_chars],
            "truncated": len(r) > max_chars, "omitted": max(0, len(r) - max_chars)}


def _settle_bytes(p):
    """Write p's own JSON length into p['bytes_preview'] -- a fixed point, since writing the number makes
    the dict longer. Iterates to convergence and, failing that, reports the LARGER figure: a byte budget
    that under-reports its own cost is worse than one that is slightly pessimistic."""
    p["bytes_preview"] = 0
    for _ in range(6):
        n = len(json.dumps(p, default=str))
        if n == p["bytes_preview"]:
            return p
        p["bytes_preview"] = n
    p["bytes_preview"] = max(p["bytes_preview"], len(json.dumps(p, default=str)))
    return p


def _tighter(head, tail, max_chars, depth):
    """The deterministic ladder of ever-tighter settings walked to meet a `max_bytes` budget.

    Order matters and this one is deliberate: characters first (a long string is the cheapest fidelity to
    give up), then breadth, then depth LAST -- depth is what makes a nested value legible at all, and a
    depth-0 preview of a list of lists says nothing about the inner shape."""
    h, t, c, d = int(head), int(tail), int(max_chars), int(depth)
    while True:
        if c > 16:
            c = max(16, c // 2)
        elif h > 1 or t > 1:
            h, t = max(1, h - 1), max(1, t - 1)
        elif d > 0:
            d -= 1
        else:
            return
        yield h, t, c, d


def _assemble(value, head, tail, max_chars, depth, handle, cost):
    """One complete preview dict at one setting, byte accounting included. Used by the budget ladder."""
    p = _preview(value, head, tail, max_chars, depth)
    if cost:
        c = json_bytes(value)
        p["bytes_full"] = c["bytes"]
        p["bytes_full_exact"] = c["exact"]
    if handle is not None:
        p["ref"] = handle
    return _settle_bytes(p)


def bounded_preview(value, head=DEFAULT_HEAD, tail=DEFAULT_TAIL, max_chars=DEFAULT_MAX_CHARS,
                    max_bytes=None, depth=DEFAULT_DEPTH, refs=None, cost=True):
    """A JSON-safe preview of `value`: its TRUE size, a head sample, a tail when it was really cut, and
    what both renderings cost in bytes.

    This is the bound half of NOOA's pass-by-reference property; `holographic_objectref` is the handle half.
    Returns a dict with `type`, the true `length` (or `shape`/`size`/`dtype` for an ndarray), `head`,
    `tail` (present ONLY when something was cut), `omitted`, `truncated`, `bytes_full`, `bytes_full_exact`
    and `bytes_preview`. The saving is `bytes_full - bytes_preview`; it is left as a subtraction so the
    module never states a win in the same breath as measuring it.

    Nested containers are bounded RECURSIVELY down to `depth` levels -- a list of 1000 lists of 1000 is the
    case that hurts, and bounding only the outer level would still ship three thousand numbers. ndarrays
    keep their nesting: a (1000, 3) array previews as rows of 3, never as 3000 flattened numbers.

    `max_bytes` walks a deterministic ladder of tighter settings until the WHOLE returned dict fits, and
    sets `budget_exceeded` if even the tightest one does not -- it will not silently overrun, and it will
    not silently claim to have fitted.

    `refs` (an ObjectRefs registry) mints a handle for the live object and returns it as `ref`. THIS IS THE
    ONLY WAY THE OMITTED MIDDLE STAYS REACHABLE: without it, a preview is a lossy dead end and everything
    between head and tail is gone from the conversation for good.

    `cost=False` skips the byte accounting when the caller only wants the sample and not the instrument."""
    handle = refs.put(value) if refs is not None else None   # minted ONCE: a handle per ladder rung would
    # register the same object several times and evict other callers' handles to do it.
    p = _assemble(value, head, tail, max_chars, depth, handle, cost)
    if max_bytes is None or p["bytes_preview"] <= max_bytes:
        return p
    for h, t, c, d in _tighter(head, tail, max_chars, depth):
        p = _assemble(value, h, t, c, d, handle, cost)
        if p["bytes_preview"] <= max_bytes:
            return p
    p["budget_exceeded"] = True
    return _settle_bytes(p)


def preview_text(preview):
    """One line of prompt text from a preview dict -- the form that actually goes into an agent's context.

    A dict is what a program consumes; a line is what a model reads. Keeping the renderer here means the
    true length and the handle travel together into the prompt, which is the entire property being bought."""
    p = preview
    size = p.get("shape") if "shape" in p else p.get("length")
    bits = ["%s%s" % (p.get("type", "?"), "" if size is None else "(%s)" % (size,))]
    if p.get("dtype"):
        bits.append(str(p["dtype"]))
    bits.append("head=%s" % json.dumps(p.get("head"), default=str)[:160])
    if "tail" in p:
        bits.append("tail=%s" % json.dumps(p["tail"], default=str)[:80])
    if p.get("omitted"):
        bits.append("(+%d omitted)" % p["omitted"])
    if p.get("ref"):
        bits.append("full value at %s" % p["ref"])
    return " ".join(bits)


def _selftest():
    """Regression traps, each pinning a NUMBER. Measured on this box with the sweep in the module header."""
    from holographic.io_and_interop.holographic_objectref import ObjectRefs

    rng = np.random.default_rng(0)

    # 1. THE HEADLINE BOUND. A million floats cost ~20 MB through _jsonable; previewed they must fit in
    #    half a kilobyte AND still report their true size -- a preview that said len 6 would be worse than
    #    no preview, because the agent would size its next call to a number the tool invented.
    a = rng.random(1000000)
    p = bounded_preview(a)
    assert p["bytes_preview"] <= 512, p["bytes_preview"]
    assert p["size"] == 1000000 and p["shape"] == [1000000], p
    assert p["truncated"] and p["omitted"] == 999994, p
    assert p["head"][0] == float(a[0]) and p["tail"][-1] == float(a[-1]), "head/tail must be the REAL ends"

    # 2. THE BASELINE IS MEASURED, NOT MODELLED, wherever it can be: the exact path must agree with the
    #    service's own rendering to the byte, and the sampled estimator must stay within 8% of exact.
    small = [1.0, 2.5, "x", None]
    jb = json_bytes(small)
    assert jb["exact"] and jb["bytes"] == len(json.dumps(_jsonsafe(small))), jb
    mid = np.arange(10000, dtype=float) / 3.0
    est, exact = json_bytes(mid)["bytes"], len(json.dumps(_jsonsafe(mid)))
    assert abs(est - exact) / float(exact) < 0.08, (est, exact)

    # 3. SHAPE IS NOT FLATTENED. A (1000, 3) array of points previewed as 3000 numbers would destroy the one
    #    fact the caller needs to write the next call: that the rows are triples.
    b = rng.random((1000, 3))
    pb = bounded_preview(b)
    assert pb["shape"] == [1000, 3] and pb["size"] == 3000, pb
    assert len(pb["head"]) == 3 and len(pb["head"][0]) == 3, pb["head"]
    assert pb["shown"] == 18 and pb["omitted"] == 2982, pb

    # 4. RECURSION IS THE WHOLE POINT. A list of 1000 lists of 1000 must bound at EVERY level: bounding only
    #    the outer level still ships six full inner lists, which measures at 41 kB against 948 B bounded.
    nl = [[float(i * j) for j in range(1000)] for i in range(1000)]
    pn = bounded_preview(nl)
    assert pn["length"] == 1000 and pn["truncated"], pn
    inner = pn["head"][0]
    assert isinstance(inner, dict) and inner["length"] == 1000 and inner["truncated"], inner
    assert pn["bytes_preview"] <= 2048, pn["bytes_preview"]
    outer_only = 6 * json_bytes(nl[1])["bytes"]
    assert outer_only > 40000, outer_only          # what bounding ONLY the outer level would have cost

    # 5. THE BUDGET IS A BOUND, not a suggestion: the WHOLE returned dict fits, or budget_exceeded says so.
    tight = bounded_preview(nl, max_bytes=300)
    assert tight["bytes_preview"] <= 300 or tight.get("budget_exceeded"), tight["bytes_preview"]
    assert tight["length"] == 1000, "the true length survives every rung of the ladder"
    impossible = bounded_preview(nl, max_bytes=10)
    assert impossible.get("budget_exceeded") is True, "an unmeetable budget must be reported, never faked"

    # 6. A TAIL IS EVIDENCE OF A CUT. Emitting one for a value that fitted whole is the same lie in a
    #    different shape, and a caller reading `tail` as "the end exists elsewhere" would be misled.
    whole = bounded_preview([1, 2, 3])
    assert "tail" not in whole and whole["truncated"] is False and whole["length"] == 3, whole

    # 7. DETERMINISM (PYTHONHASHSEED=0 / no rng in the sampler): the same value previews byte-identically.
    assert json.dumps(bounded_preview(nl)) == json.dumps(bounded_preview(nl))

    # 8. THE HANDLE IS THE FIDELITY ESCAPE HATCH -- identity, not a copy, or the omitted middle is gone.
    refs = ObjectRefs()
    pr = bounded_preview(a, refs=refs)
    assert pr["ref"].startswith("ref:ndarray:") and refs.get(pr["ref"]) is a, pr["ref"]
    assert "1000000" in preview_text(pr) and pr["ref"] in preview_text(pr), preview_text(pr)

    # 9. KEPT NEGATIVE, PINNED SO NOBODY "OPTIMISES" IT AWAY: bounding a SMALL value costs MORE than
    #    sending it whole (measured crossover ~16 floats / ~10 dict keys / ~200 chars). This is precisely
    #    why the service seam bounds only what is already over budget instead of bounding by reflex.
    tiny = rng.random(5)
    assert bounded_preview(tiny)["bytes_preview"] > json_bytes(tiny)["bytes"], "the envelope is not free"

    print("holographic_boundedpreview selftest OK "
          "(1e6 floats: %d B whole -> %d B bounded, %.0fx)"
          % (json_bytes(a)["bytes"], p["bytes_preview"], json_bytes(a)["bytes"] / float(p["bytes_preview"])))


if __name__ == "__main__":
    _selftest()

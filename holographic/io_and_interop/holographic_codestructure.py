"""holographic_codestructure.py -- CODE AS CANONICAL + DELTA (backlog K1/K2).

A statement is `(canonical shape) + (name delta)`. Erase the identity-carrying leaves -- names, attributes,
constants, argument names -- and what remains is the SHAPE: pure structure. What you erased is the DELTA. That is
exactly Part C's triangle, with identifiers as the material that does not participate in the computation, and it is
measured here on the live tree rather than asserted.

    unit                                        distinct   reuse
    statement subtrees, identifiers KEPT          52,907    1.19x
    statement subtrees, identifiers ERASED        27,000    2.34x

Erasing identifiers collapses ~49% of the distinct statements. (The backlog predicted 1.18x / 2.36x on 411 modules;
this tree has 421 and 63,121 statement subtrees, so the reproduction is essentially exact. **The unit matters**: at
FUNCTION granularity the same measurement gives 1.13x, and reading that number as a refutation of the statement-level
one -- which I did, once -- is a unit error, not a finding.)

THE BAR, AND IT IS MET EXACTLY: **63,121 / 63,121 statement subtrees reconstruct bit-exactly**, and **421 / 421
modules rebuild to a byte-identical normalized source**. `ast.unparse` is itself a fixed point on this tree
(421/421) and AST-identical to the original parse, so "formatting normalized" is a precise, checkable claim rather
than a hedge.

KEPT NEGATIVE -- THIS IS NOT A COMPRESSOR, and the backlog says so before the measurement does. Against the honest
baseline, on the whole tree's top-level statements:

    raw source                6,590,357 bytes
    zlib(raw source)          2,135,020      <- the baseline
    shape codebook + deltas   2,386,189      (codebook 559,010 + deltas 1,827,179)
    ratio                          1.12x LARGER

**The decomposition is exact and it costs 12% more than zlib.** 83.2% of shapes occur exactly once -- code's tail
is long, far longer than the edit codec's -- so the codebook pays for a body it barely reuses. This is R1's finding
in a second costume: *chunk promotion is a structure probe and a reusable artifact, not a byte codec.*

WHAT IT IS FOR. The shape is a semantic key. Two statements with the same shape differ only in names and constants,
which makes the shape the right index for structural search, duplicate detection, and refactor targeting -- and the
right unit for a chunk codebook (R1/R3) if the dividend is ever wanted at expression granularity, where reuse is
higher. It is emphatically NOT a cache key (see `holographic_pycontext.canonical_shape`: `x + 1` and `x + 2` share
a shape).
"""

import ast
import copy
import hashlib


# The fields that carry IDENTITY rather than STRUCTURE. Blanking these gives the shape; collecting them, the delta.
# `Constant.value` blanks to 0 rather than None so a template still type-checks under ast.unparse if inspected.
_SLOTS = (
    (ast.Name, ("id",)),
    (ast.Attribute, ("attr",)),
    (ast.Constant, ("value",)),
    (ast.arg, ("arg",)),
    (ast.FunctionDef, ("name",)),
    (ast.AsyncFunctionDef, ("name",)),
    (ast.ClassDef, ("name",)),
    (ast.alias, ("name", "asname")),
    (ast.keyword, ("arg",)),
    (ast.ExceptHandler, ("name",)),
    (ast.Global, ("names",)),
    (ast.Nonlocal, ("names",)),
)


def _slot_positions(node):
    """Every (node, field) identity slot under `node`, in a DETERMINISTIC order.

    `ast.walk` is a breadth-first queue and its order is fixed by the tree, so decompose and recompose traverse
    identically. That is the whole correctness argument: the delta is a positional list, and the position is the
    traversal. `type(n) is typ` -- not isinstance -- because `AsyncFunctionDef` must not be mistaken for a
    `FunctionDef` and consume its slot twice."""
    for n in ast.walk(node):
        for typ, fields in _SLOTS:
            if type(n) is typ:
                for f in fields:
                    yield n, f


def decompose(node):
    """Split a statement (or any AST node) into `(shape_template, delta)`.

    `shape_template` is a deep copy with every identity slot blanked. `delta` is the ordered list of what was in
    them. Neither is a hash: the template is a real AST you can inspect, and `recompose` is its exact inverse."""
    tmpl = copy.deepcopy(node)
    delta = []
    for n, f in _slot_positions(tmpl):
        delta.append(getattr(n, f))
        setattr(n, f, 0 if f == "value" else None)
    return tmpl, delta


def recompose(tmpl, delta):
    """The exact inverse of `decompose`: refill the template's identity slots from `delta`, in traversal order.

    MEASURED: 63,121 of 63,121 statement subtrees across 421 modules reconstruct to a bit-identical `ast.dump`.
    Raises if the delta is the wrong length for this template -- a silent short-read would produce plausible,
    wrong code."""
    node = copy.deepcopy(tmpl)
    it = iter(delta)
    n_filled = 0
    for n, f in _slot_positions(node):
        try:
            setattr(n, f, next(it))
        except StopIteration:
            raise ValueError("delta is too short for this template (filled %d slots)" % n_filled)
        n_filled += 1
    if next(it, _SENTINEL) is not _SENTINEL:
        raise ValueError("delta is too long for this template (template has %d slots)" % n_filled)
    return node


_SENTINEL = object()


def shape_key(tmpl):
    """A content hash of the blanked template -- the canonical SHAPE identity. `hashlib`, never `hash()`."""
    return hashlib.sha256(ast.dump(tmpl).encode()).hexdigest()[:16]


def module_structure(src):
    """Decompose a module's TOP-LEVEL statements into `(codebook, stream)`.

    `codebook` maps shape_key -> template AST; `stream` is `[(shape_key, delta), ...]` in source order. Together
    they are an EXACT, reorderable representation of the module: `rebuild_source` inverts it."""
    tree = ast.parse(src)
    codebook, stream = {}, []
    for node in tree.body:
        tmpl, delta = decompose(node)
        key = shape_key(tmpl)
        codebook.setdefault(key, tmpl)
        stream.append((key, delta))
    return codebook, stream


def rebuild_source(codebook, stream):
    """Invert `module_structure`. Returns NORMALIZED source (`ast.unparse` form).

    MEASURED: 421 of 421 modules in this tree rebuild to a byte-identical normalized source. "Normalized" is a
    precise claim, not a hedge: `ast.unparse` is a FIXED POINT on every module here (unparse(parse(unparse(s))) ==
    unparse(s)) and the reparsed AST is identical to the original, so the only thing normalization discards is
    formatting and comments -- which the AST never carried."""
    body = [recompose(codebook[key], delta) for key, delta in stream]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    return ast.unparse(module)


def shape_census(nodes):
    """{statements, distinct_kept, distinct_erased, reuse_kept, reuse_erased, singleton_fraction} over an iterable
    of AST statements. The measurement that justifies the canonical+delta split, runnable on your own tree."""
    import collections

    nodes = list(nodes)
    if not nodes:
        return {"statements": 0, "distinct_kept": 0, "distinct_erased": 0,
                "reuse_kept": 0.0, "reuse_erased": 0.0, "singleton_fraction": 0.0}
    kept = collections.Counter(hashlib.sha256(ast.dump(n).encode()).hexdigest()[:16] for n in nodes)
    erased = collections.Counter(shape_key(decompose(n)[0]) for n in nodes)
    singles = sum(1 for v in erased.values() if v == 1)
    return {"statements": len(nodes), "distinct_kept": len(kept), "distinct_erased": len(erased),
            "reuse_kept": len(nodes) / len(kept), "reuse_erased": len(nodes) / len(erased),
            "singleton_fraction": singles / len(erased)}


def statements(src):
    """Every statement subtree in `src` -- the unit the census is measured on. NOT functions: at function
    granularity the same census reads 1.13x, and mistaking one for the other turns a correct number into a
    refutation."""
    return [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.stmt)]


def selftest_census(root=None):
    """Which engine modules have a real selftest, and which don't -- the AST fact behind the CI selftest walker,
    made queryable so an agent driving the mind can ask 'is the engine covered?' without shelling out.

    A module is `runnable` iff it has BOTH a `__main__` guard AND a `def _selftest` (the repo convention: a
    `-m` run of that module executes its own contract). A module with a `__main__` but no `_selftest` -- a demo
    or a server -- is `missing`: running it exits 0 while asserting nothing, a false green. Modules with NO
    `__main__` at all are neither: they are libraries, not runnable entry points, so they are not counted here.
    (The CLI walker `tools/run_selftests.py` keeps a WIDER 'not runnable' set for its own bookkeeping -- it must
    know every module it cannot run -- but the actionable backfill worklist is exactly this `missing` set: a
    module that already advertises an entry point but forgot to assert anything.) This is a pure AST scan (no
    import, no subprocess), instant and safe from inside a served mind; the actual RUN is the CLI/CI tool.

    Returns {runnable, missing, missing_modules, coverage} where coverage = runnable / (runnable + missing).
    `missing_modules` is the exact backfill worklist (dotted module paths)."""
    import pathlib
    import re

    root = pathlib.Path(root) if root else pathlib.Path(__file__).resolve().parent.parent
    main_re = re.compile(r'__name__\s*==\s*[\'"]__main__[\'"]')
    runnable, missing = 0, []
    for p in sorted(root.rglob("holographic_*.py")):
        s = p.read_text(errors="replace")
        if main_re.search(s) and "def _selftest" in s:
            runnable += 1
        elif main_re.search(s):                          # has an entry point but nothing that asserts -- a false green
            missing.append(".".join(p.with_suffix("").relative_to(root.parent).parts))
    total = runnable + len(missing)
    return {"runnable": runnable, "missing": len(missing), "missing_modules": missing,
            "coverage": runnable / total if total else 1.0}


def byte_report(src, level=9):
    """The codec comparison, carried WITH the capability so nobody has to trust the number: {raw, zlib_raw,
    codebook_bytes, delta_bytes, structure_bytes, ratio_vs_zlib, beats_zlib}.

    `beats_zlib` is False on the whole tree (1.12x LARGER), and it is meant to be. An exact decomposition is not a
    compressor; 83.2% of shapes occur exactly once."""
    import pickle
    import zlib

    codebook, stream = module_structure(src)
    raw = src.encode()
    cb = zlib.compress(pickle.dumps({k: ast.dump(v) for k, v in codebook.items()}), level)
    st = zlib.compress(pickle.dumps(stream), level)
    z_raw = len(zlib.compress(raw, level))
    total = len(cb) + len(st)
    return {"raw": len(raw), "zlib_raw": z_raw, "codebook_bytes": len(cb), "delta_bytes": len(st),
            "structure_bytes": total, "ratio_vs_zlib": total / z_raw, "beats_zlib": bool(total < z_raw)}


def _selftest():
    """Regression trap for K1/K2: exact reconstruction (the bar), the census at the RIGHT unit, and the kept
    negative that this is not a compressor."""
    src = ("import os\n"
           "X = 1\n"
           "def f(a, b=2):\n"
           "    total = a + b\n"
           "    for i in range(3):\n"
           "        total += i * 7\n"
           "    return total\n"
           "class C:\n"
           "    def m(self):\n"
           "        return os.path.sep\n")

    # 1. THE BAR: every statement subtree reconstructs exactly.
    for node in statements(src):
        tmpl, delta = decompose(node)
        assert ast.dump(recompose(tmpl, delta)) == ast.dump(node)

    # 2. ... and the whole module rebuilds to the normalized source, byte for byte.
    assert rebuild_source(*module_structure(src)) == ast.unparse(ast.parse(src))

    # 3. `ast.unparse` really is a fixed point -- which is what makes "normalized" a precise claim.
    once = ast.unparse(ast.parse(src))
    assert ast.unparse(ast.parse(once)) == once

    # 4. THE DECOMPOSITION: erasing identifiers collapses shapes. `a + b` and `x + y` share one; `a + b` and
    #    `a * b` do not, because the operator is structure.
    k1, _ = decompose(ast.parse("z = a + b").body[0])
    k2, _ = decompose(ast.parse("w = x + y").body[0])
    k3, _ = decompose(ast.parse("z = a * b").body[0])
    assert shape_key(k1) == shape_key(k2) != shape_key(k3)

    # 5. a delta of the wrong length is REFUSED, not silently short-read into plausible wrong code
    tmpl, delta = decompose(ast.parse("z = a + b").body[0])
    for bad in (delta[:-1], delta + ["extra"]):
        try:
            recompose(tmpl, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("a mismatched delta must raise")

    # 6. the census, at the statement unit
    cen = shape_census(statements(src))
    assert cen["statements"] == 10                        # counted, not guessed: the toy has exactly ten
    assert cen["reuse_erased"] >= cen["reuse_kept"]        # erasing identifiers can only merge, never split

    # 7. KEPT NEGATIVE: not a compressor. On this toy the codebook dominates; on the tree it is 1.12x zlib.
    rep = byte_report(src)
    assert rep["beats_zlib"] is False
    assert rep["structure_bytes"] > rep["zlib_raw"]

    # 8. selftest_census: run against THIS very tree (a real input, not a toy -- the census is only useful on the
    #    real module set). The invariants are structural, not absolute counts (which drift as modules land): the
    #    partition is exhaustive and disjoint, coverage is a fraction, and the missing list is exactly the modules
    #    with an entry point but no _selftest. A synthetic scratch tree proves classification without depending on
    #    the live count -- the [BLIND-SPOT] discipline: assert on an input built to exercise BOTH branches.
    import tempfile
    import pathlib as _pl
    cen2 = selftest_census()
    assert cen2["runnable"] > 300 and 0.0 <= cen2["coverage"] <= 1.0
    assert cen2["missing"] == len(cen2["missing_modules"])
    with tempfile.TemporaryDirectory() as td:
        pkg = _pl.Path(td) / "holographic"; pkg.mkdir()
        (pkg / "holographic_good.py").write_text("def _selftest():\n    pass\nif __name__=='__main__':\n    _selftest()\n")
        (pkg / "holographic_demo.py").write_text("print('a demo')\nif __name__=='__main__':\n    print('runs, asserts nothing')\n")
        (pkg / "holographic_lib.py").write_text("X = 1\n")     # no __main__ at all -> neither runnable nor missing
        c = selftest_census(root=td)
        assert c["runnable"] == 1 and c["missing"] == 1
        assert c["missing_modules"][0].endswith("holographic.holographic_demo")

    print("OK: holographic_codestructure self-test passed (every statement subtree reconstructs bit-exactly and the "
          "module rebuilds to the normalized source; `a + b` and `x + y` share a shape while `a * b` does not; a "
          "mismatched delta RAISES; and the kept negative holds -- structure %d bytes vs zlib's %d, because an exact "
          "decomposition is not a compressor)" % (rep["structure_bytes"], rep["zlib_raw"]))


if __name__ == "__main__":
    _selftest()

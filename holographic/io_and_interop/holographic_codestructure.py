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


# ---------------------------------------------------------------------------
# POST-MERGE CENSUS (sweep 129). The rule NOTES states in capitals across sweeps 120, 121
# and 125 -- "after ANY merge, census DEFINITIONS, SIGNATURES and LINE COUNTS; and before
# restoring a shrunk file, check whether the content MOVED" -- with an instrument behind it.
#
# WHY HERE. `merge_trees` (p21) censuses two trees by sha256 and both-direction unique-LINE
# counts, at the FILE level, BEFORE a merge, as a decision sheet. It cannot see a definition
# that vanished inside a file it calls 'differ', and it never runs AFTER. This module already
# owns the AST census (shape_census, selftest_census) and imports nothing but stdlib, so the
# structural legs belong here and the mind verb belongs next to merge_trees. Partner, not
# sibling.
# ---------------------------------------------------------------------------

CENSUS_IGNORE = (".git", "__pycache__", ".pytest_cache", ".lecore_jobs", "node_modules")


def signature_of(node):
    """A def's full CALL SHAPE as one stable string: decorators, every parameter in order,
    which parameters carry a default, *args / keyword-only / **kw, and the arity.

    WHY PER-ARGUMENT DEFAULTS AND NOT A COUNT. Sweep 120's second casualty was a lost
    `record_every` passthrough, which a name-level census passed; the obvious fix is to
    count defaults, and that is still not enough. Keyword-only parameters may carry defaults
    in ANY order, so `def h(*, a=1, b)` and `def h(*, a, b=1)` have the same count and
    different meanings. Recording presence per argument costs nothing and closes that hole
    (pinned in the selftest).

    KEPT NEGATIVE: default VALUES are not recorded, only their presence. `def f(a=1)` ->
    `def f(a=2)` is a real behaviour change this census does not see; catching it needs a
    value-level diff, which is a different and much noisier instrument."""
    a = node.args
    parts = []
    for arg in list(a.posonlyargs) + list(a.args):
        parts.append(arg.arg)
    # positional defaults bind to the RIGHTMOST arguments, so mark them from the right
    npos = len(a.posonlyargs) + len(a.args)
    for i in range(len(a.defaults)):
        parts[npos - 1 - i] += "="
    if a.posonlyargs:
        parts.insert(len(a.posonlyargs), "/")
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    for arg, dflt in zip(a.kwonlyargs, a.kw_defaults):
        parts.append(arg.arg + ("=" if dflt is not None else ""))
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    decos = [_deco_name(d) for d in node.decorator_list]
    head = "".join("@%s " % d for d in decos)
    return "%sdef %s(%s)" % (head, node.name, ", ".join(parts))


def _deco_name(node):
    """A decorator's dotted name, as written. A call decorator keeps only its callee -- the
    ARGUMENTS of `@lru_cache(128)` are a tuning change, not a change to the call shape."""
    if isinstance(node, ast.Call):
        node = node.func
    bits = []
    while isinstance(node, ast.Attribute):
        bits.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        bits.append(node.id)
    return ".".join(reversed(bits)) or "<expr>"


def def_index(src):
    """Every name a module DEFINES, as {qualified_name: (kind, signature)}. Raises SyntaxError.

    Kinds: "def" (functions and methods, with the full signature), "class", "import", "assign".

    AN IMPORT ALIAS IS A DEFINITION OF THAT NAME, and this is the kept negative that shaped the
    whole function. Run on the real sweep-122 merge, a def-only census reported two LOST
    DEFINITIONS where upstream had promoted a triplicated `_f1` helper into
    `holographic_occlusion.recall_f1` and replaced each copy with `from ... import recall_f1 as
    _f1`. Nothing was removed; the census cried wolf twice. A census that cries wolf is a census
    the next session stops running, so `from X import y as name` binds `name` here.

    Module- and class-level ASSIGNMENTS are indexed too (kind "assign"), because a lost constant
    table is exactly the kind of thing a clean three-way merge eats and no name-of-a-function
    census would see it.

    SCOPE, deliberately: top level and class bodies only. A def nested inside a function is an
    implementation detail of its parent -- indexing it would report every refactor of a local
    helper as a lost definition, which is the cry-wolf failure in a second costume."""
    out = {}

    def walk(node, prefix=""):
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[prefix + child.name] = ("def", signature_of(child))
            elif isinstance(child, ast.ClassDef):
                bases = ", ".join(_deco_name(b) for b in child.bases)
                out[prefix + child.name] = ("class", "class %s(%s)" % (child.name, bases))
                walk(child, prefix + child.name + ".")
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for alias in child.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    if bound != "*":
                        out[prefix + bound] = ("import", "import " + bound)
            elif isinstance(child, ast.Assign):
                for tgt in child.targets:
                    if isinstance(tgt, ast.Name):
                        out[prefix + tgt.id] = ("assign", "assign " + tgt.id)
            elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                out[prefix + child.target.id] = ("assign", "assign " + child.target.id)

    walk(ast.parse(src))
    return out


def read_tree(root, ignore=CENSUS_IGNORE, max_bytes=2000000):
    """Every text file under a directory as {relative_path: source}. Binaries are SKIPPED, not
    guessed at: a file that does not decode as UTF-8 has no lines and no definitions to census.

    Deterministic: os.walk is sorted at both levels, so two runs enumerate identically."""
    import os

    out = {}
    root = str(root)
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d not in ignore)
        for f in sorted(fn):
            p = os.path.join(dp, f)
            try:
                if os.path.getsize(p) > max_bytes:
                    continue
                out[os.path.relpath(p, root)] = open(p, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                continue
    return out


def read_git_ref(repo, ref, ignore=CENSUS_IGNORE, max_bytes=2000000):
    """Every text file at a git ref as {relative_path: source}, WITHOUT checking anything out.

    `git archive` streams the whole tree in ONE subprocess and tarfile reads it from memory.
    The obvious alternative -- `git show ref:path` per file -- costs one process per file
    (1,788 of them on this repo), which is the difference between an audit you run after every
    merge and one you do not."""
    import io
    import subprocess
    import tarfile

    r = subprocess.run(["git", "archive", "--format=tar", str(ref)], cwd=str(repo),
                       capture_output=True)
    if r.returncode != 0:
        raise ValueError("git archive %r failed in %r: %s"
                         % (ref, str(repo), r.stderr.decode(errors="replace")[:200]))
    out = {}
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tf:
        for m in tf.getmembers():
            if not m.isfile() or m.size > max_bytes:
                continue
            if any(part in ignore for part in m.name.split("/")):
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            try:
                out[m.name] = f.read().decode("utf-8")
            except UnicodeDecodeError:
                continue
    return out


def _line_sets(texts):
    """{path: set of non-blank lines} -- the index the moved-content check joins against."""
    return {p: {ln for ln in t.splitlines() if ln.strip()} for p, t in texts.items()}


def merge_census(base, new, shrink_pct=10.0, base_is_ref=None, ignore=CENSUS_IGNORE,
                 move_gate=0.5, max_rows=200, max_bytes=2000000):
    """DID THE MERGE LOSE ANYTHING? Census two trees by DEFINITION, SIGNATURE and LINE COUNT,
    then ask of everything lost or shrunk whether the content MOVED.

    `base` is a directory, or a git ref resolved against the `new` working tree. Ambiguity is
    REFUSED, never guessed (the merge_trees house rule): a string that is BOTH an existing
    directory and a valid ref raises, and so does one that is neither -- pass `base_is_ref=`
    to decide. Returns a report with `verdict` in CLEAN / REVIEW / LOSSES FOUND.

    WHY ONE CALL AND NOT THREE, which is the real design question here. The three legs are not
    independent, and the history says so twice. Sweep 120: a name-level census passed a function
    that had quietly lost a `record_every` parameter, so the signature leg only earns its keep
    run TOGETHER with the definition leg over the same index. Sweep 125: a line-count leg found
    the catalog had shrunk by 970 lines and the reflex was to restore it -- the lines had MOVED
    into holographic_catalog_aliases.py, and restoring would have reverted somebody's refactor.
    A shrink report without the move join is not merely incomplete, it is ACTIVELY MISLEADING,
    and a caller who has to remember to run leg three after leg one is a caller who will not.
    So: one call, one index, one verdict -- and `def_index` / `signature_of` stay public for
    anyone who wants a leg on its own.

    The legs:
      * definitions -- present in base, absent in new. HARD ERROR unless the name is found
        elsewhere in the new tree, in which case it is reported as MOVED and does not count
        against the verdict.
      * signatures  -- same name, different call shape. REVIEW, never auto-judged: an additive
        parameter with a default and a deleted passthrough look identical to a counter.
      * line counts -- files that shrank by more than `shrink_pct`, each with the lines that
        went missing and the new-or-grown files that now contain them (`move_gate` is the
        fraction of missing lines a destination must hold to be called a move).
      * unparseable -- a file that no longer parses is its OWN bucket. The prototype let a
        SyntaxError in the new copy report every definition in that file as lost: one broken
        file, a hundred false hard errors. Loud and specific beats loud and wrong. Only a
        file that parsed in the BASE and does not now counts toward the verdict: this tree
        has one that fails in both, and counting it pinned the verdict at REVIEW forever.

    Definition legs cover .py only; the line-count leg covers every text file, because the two
    conflicts in this very merge were a .md and a .json.

    KEPT NEGATIVES, all four load-bearing:
      * It sees NAMES AND SHAPES, never semantics. A function that survives with its signature
        intact but whose body was gutted to `pass` passes every leg here. This is the same limit
        `result_usable` has in a different costume -- an ABSENT thing is detectable, a WRONG one
        needs a test suite, which is the instrument that runs after this one.
      * A file larger than `max_bytes` is invisible to every leg, silently. The bound exists so a
        census of a repo with a checked-in dataset does not read the dataset into memory twice.
      * The moved-content join is LINE-EXACT. A block that moved AND was reformatted on the way
        shows up as `unexplained`, which is a false alarm in the safe direction -- it asks for a
        human look rather than declaring a loss.
      * Default VALUES are not compared, only their presence (see signature_of)."""
    import os

    new_root = str(new)
    if base_is_ref is None:
        as_dir = os.path.isdir(str(base))
        as_ref = _is_git_ref(new_root, base)
        if as_dir and as_ref:
            raise ValueError(
                "ambiguous base %r: it is BOTH an existing directory and a valid git ref in %r "
                "-- pass base_is_ref=True or False" % (str(base), new_root))
        if not as_dir and not as_ref:
            raise ValueError(
                "base %r is neither an existing directory nor a git ref in %r"
                % (str(base), new_root))
        base_is_ref = as_ref
    if base_is_ref:
        b_texts = read_git_ref(new_root, base, ignore=ignore, max_bytes=max_bytes)
    else:
        b_texts = read_tree(base, ignore=ignore, max_bytes=max_bytes)
    n_texts = read_tree(new_root, ignore=ignore, max_bytes=max_bytes)

    def index(texts):
        idx, bad = {}, {}
        for p, t in sorted(texts.items()):
            if not p.endswith(".py"):
                continue
            try:
                idx[p] = def_index(t)
            except SyntaxError as e:
                bad[p] = str(e)[:80]
        return idx, bad

    b_idx, b_bad = index(b_texts)
    n_idx, n_bad = index(n_texts)

    # WHERE a name lives NOW: full qualified name and bare leaf, because a def that moves module
    # usually keeps its leaf and may gain or lose a class prefix on the way.
    where_qual, where_leaf = {}, {}
    for p, d in n_idx.items():
        for name in d:
            where_qual.setdefault(name, []).append(p)
            where_leaf.setdefault(name.rsplit(".", 1)[-1], []).append(p)

    def moved_to(name, exclude):
        hits = set(where_qual.get(name, ())) | set(where_leaf.get(name.rsplit(".", 1)[-1], ()))
        return sorted(hits - {exclude})

    files_deleted, lost, sig_changed = [], [], []
    for p in sorted(b_idx):
        if p not in n_idx:
            if p in n_texts:
                continue                      # still there, just unparseable -- reported below
            names = sorted(b_idx[p])
            files_deleted.append({"file": p, "defs": len(names),
                                  "moved": {n: moved_to(n, p) for n in names
                                            if moved_to(n, p)}})
            continue
        after = n_idx[p]
        for name in sorted(b_idx[p]):
            kind, sig = b_idx[p][name]
            if name not in after:
                dest = moved_to(name, p)
                lost.append({"file": p, "name": name, "kind": kind, "moved_to": dest})
            elif after[name][1] != sig:
                sig_changed.append({"file": p, "name": name, "kind": kind,
                                    "was": sig, "now": after[name][1]})

    shrunk = []
    grown_or_new = {p: t for p, t in n_texts.items()
                    if p not in b_texts or len(t.splitlines()) > len(b_texts[p].splitlines())}
    cand_lines = _line_sets(grown_or_new)
    for p in sorted(b_texts):
        if p not in n_texts:
            continue
        b_lines, n_lines = b_texts[p].splitlines(), n_texts[p].splitlines()
        if not b_lines or len(n_lines) >= len(b_lines) * (1.0 - shrink_pct / 100.0):
            continue
        n_set = set(n_lines)
        missing = sorted({ln for ln in b_lines if ln.strip() and ln not in n_set})
        found = []
        for q in sorted(cand_lines):
            if q == p:
                continue
            hit = sum(1 for ln in missing if ln in cand_lines[q])
            if hit:
                found.append({"file": q, "lines": hit,
                              "fraction": round(hit / max(len(missing), 1), 3)})
        found.sort(key=lambda r: (-r["lines"], r["file"]))
        best = found[0]["fraction"] if found else 0.0
        shrunk.append({"file": p, "base_lines": len(b_lines), "new_lines": len(n_lines),
                       "shrank_pct": round(100.0 * (1 - len(n_lines) / len(b_lines)), 1),
                       "missing_lines": len(missing), "moved_into": found[:3],
                       "verdict": "moved" if best >= move_gate else "unexplained"})

    lost_unexplained = [r for r in lost if not r["moved_to"]]
    deleted_unexplained = [r for r in files_deleted if len(r["moved"]) < r["defs"]]
    shrunk_unexplained = [r for r in shrunk if r["verdict"] == "unexplained"]
    # ONLY A *NEW* PARSE FAILURE IS A MERGE FINDING. Found by running this on the real merge:
    # tools/tour.py has an f-string this interpreter rejects and it fails in BOTH trees, so
    # counting it pinned the verdict at REVIEW forever -- a permanent yellow light is a light
    # nobody reads. Pre-existing breakage is the linter's business, not the census's.
    newly_bad = sorted(p for p in n_bad if p not in b_bad)
    hard = len(lost_unexplained) + len(deleted_unexplained)
    review = len(sig_changed) + len(shrunk_unexplained) + len(newly_bad)
    counts = {"base_files": len(b_texts), "new_files": len(n_texts),
              "base_py": len(b_idx), "new_py": len(n_idx),
              "files_deleted": len(files_deleted), "files_added": len(set(n_texts) - set(b_texts)),
              "lost": len(lost), "lost_moved": len(lost) - len(lost_unexplained),
              "lost_unexplained": len(lost_unexplained),
              "lost_by_kind": {k: sum(1 for r in lost_unexplained if r["kind"] == k)
                               for k in sorted({r["kind"] for r in lost_unexplained})},
              "signature_changed": len(sig_changed), "shrunk": len(shrunk),
              "shrunk_moved": len(shrunk) - len(shrunk_unexplained),
              "shrunk_unexplained": len(shrunk_unexplained),
              "unparseable_base": len(b_bad), "unparseable_new": len(n_bad),
              "unparseable_newly": len(newly_bad)}
    return {"base": str(base), "new": new_root, "base_is_ref": bool(base_is_ref),
            "counts": counts,
            "files_deleted": files_deleted[:max_rows],
            "lost": lost[:max_rows], "signature_changed": sig_changed[:max_rows],
            "shrunk": shrunk[:max_rows],
            "unparseable": {"base": sorted(b_bad)[:max_rows], "new": sorted(n_bad)[:max_rows],
                            "newly": newly_bad[:max_rows]},
            "truncated": any(len(x) > max_rows for x in (files_deleted, lost, sig_changed,
                                                         shrunk, b_bad, n_bad)),
            "verdict": "LOSSES FOUND" if hard else ("REVIEW" if review else "CLEAN"),
            "advice": ("HARD rows are definitions or files present in base and findable nowhere "
                       "in the new tree -- restore them. REVIEW rows are judgement calls: a "
                       "signature change may be an additive default, and a shrunk file whose "
                       "lines turn up in another file MOVED (sweep 125) -- check moved_into "
                       "before you restore anything, or you will revert somebody's refactor.")}


def _is_git_ref(repo, ref):
    """Is `ref` resolvable as a tree-ish in this repo? Used only to disambiguate, never to guess."""
    import subprocess

    try:
        r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", "%s^{tree}" % (ref,)],
                           cwd=str(repo), capture_output=True)
    except OSError:
        return False
    return r.returncode == 0


_FILLER = "\n".join("# filler line %03d, so mod_a's own line count barely moves and the" % i
                    for i in range(100))

# The eight injected faults, named once so the selftest, the tests and the report all count the
# same things. "leg" is which leg is SUPPOSED to catch it; "expect" is the exact row count.
CENSUS_FAULTS = (
    ("deleted_def",   "definitions", "a def removed outright"),
    ("dropped_param", "signatures",  "sweep 120's second casualty: a lost passthrough parameter"),
    ("moved_def",     "definitions", "a def that moved module -- lost HERE, present THERE"),
    ("moved_data",    "line_counts", "sweep 125's case: 90 data lines moved to a new file"),
    ("import_alias",  "none",        "def -> `from X import y as name`: NOTHING was removed"),
    ("deleted_file",  "definitions", "a whole file gone, its def findable nowhere"),
    ("kwonly_shuffle", "signatures", "a keyword-only default moved: same COUNT, different meaning"),
    ("new_syntax_error", "unparseable", "the new copy does not parse: its own bucket, not a storm"),
)


def _census_fixture(base, new):
    """Write two trees with KNOWN injected damage -- the honest way to measure a census.

    One fault per row of CENSUS_FAULTS, chosen so every leg has something only IT can catch and
    so the two historical false-positive shapes are both present: the import-alias promotion
    (which a def-only census reports as a loss) and the moved data block (which a line-count
    census tells you to restore). Shared by the module selftest and tests/test_merge_census.py so
    the numbers in both are the same numbers."""
    import os

    for d in (base, new):
        os.makedirs(d, exist_ok=True)

    def w(root, rel, text):
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        open(p, "w", encoding="utf-8").write(text)

    head = '"""module a"""\nimport os\n\nCONST = 1\n\n' + _FILLER + "\n\n"
    keep = ("def kept(a, b=1):\n    return a + b\n\n\n")
    w(base, "mod_a.py", head + keep +
      "def deleted_def(x):\n    return x\n\n\n"                       # fault: deleted_def
      "def sim(a, b, record_every=1):\n    return a\n\n\n"            # fault: dropped_param
      "def travels(q):\n    return q * 2\n\n\n"                       # fault: moved_def
      "def kwonly(*, a=1, b):\n    return a, b\n\n\n"                 # fault: kwonly_shuffle
      "def _f1(rec, true_set):\n    return 0.0\n")                    # fault: import_alias
    w(new, "mod_a.py", head + keep +
      "def sim(a, b):\n    return a\n\n\n"
      "def kwonly(*, a, b=1):\n    return a, b\n\n\n"
      "from pkg.occ import recall_f1 as _f1\n")
    w(base, "mod_b.py", "def already(x):\n    return x\n")
    w(new, "mod_b.py", "def already(x):\n    return x\n\n\ndef travels(q):\n    return q * 2\n")
    w(base, "gone.py", "def orphan():\n    return 1\n")               # fault: deleted_file
    rows = ["row %03d of the table that moves" % i for i in range(100)]
    w(base, "data.txt", "\n".join(rows) + "\n")                       # fault: moved_data
    w(new, "data.txt", "\n".join(rows[:10]) + "\n")
    w(new, "data_extra.txt", "\n".join(rows[10:]) + "\n")
    w(base, "broken.py", "def fine():\n    return 1\n")               # fault: new_syntax_error
    w(new, "broken.py", "def fine(:\n    return 1\n")
    return CENSUS_FAULTS


def _census_selftest():
    """Detection counts on the injected-damage fixture. Every assertion is a NUMBER, because the
    only question that matters about a census is how many faults it catches and how many it
    invents -- and the second number is the one that decides whether anyone runs it twice."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        b, n = td + "/base", td + "/new"
        _census_fixture(b, n)
        r = merge_census(b, n, base_is_ref=False)
        c = r["counts"]

        # -- LEG 1, definitions. deleted_def is unexplained; travels MOVED and must not count
        #    against the verdict; gone.py is a deleted FILE whose def is findable nowhere.
        assert c["lost"] == 2 and c["lost_unexplained"] == 1 and c["lost_moved"] == 1, c
        assert [x["name"] for x in r["lost"] if not x["moved_to"]] == ["deleted_def"]
        moved = [x for x in r["lost"] if x["moved_to"]]
        assert [x["name"] for x in moved] == ["travels"]
        assert moved[0]["moved_to"] == ["mod_b.py"], moved
        assert c["files_deleted"] == 1 and r["files_deleted"][0]["file"] == "gone.py"

        # -- THE KEPT NEGATIVE, and the reason this fixture exists at all. The sweep-123 `_f1`
        #    promotion removed NOTHING; a def-only census called it two lost definitions. Zero
        #    here, forever. A census that cries wolf is a census the next session stops running.
        assert "_f1" not in [x["name"] for x in r["lost"]], "an import alias was called a loss"

        # -- LEG 2, signatures. THREE, and each is invisible to a leg that is not this one:
        #    a dropped passthrough, a keyword-only default that moved (the COUNT is unchanged --
        #    a defaults-counting census misses this one), and def -> import alias.
        sig = {x["name"]: x for x in r["signature_changed"]}
        assert c["signature_changed"] == 3, r["signature_changed"]
        assert set(sig) == {"sim", "kwonly", "_f1"}, sorted(sig)
        assert sig["sim"]["was"] == "def sim(a, b, record_every=)"
        assert sig["sim"]["now"] == "def sim(a, b)"
        assert sig["kwonly"]["was"] == "def kwonly(*, a=, b)"
        assert sig["kwonly"]["now"] == "def kwonly(*, a, b=)", "kwonly default shuffle missed"
        assert sig["_f1"]["now"] == "import _f1"

        # -- LEG 3, line counts, WITH the join that makes it safe to act on. data.txt lost 90 of
        #    100 lines and every one of them is in data_extra.txt: MOVED, do not restore.
        assert c["shrunk"] == 1 and c["shrunk_moved"] == 1 and c["shrunk_unexplained"] == 0, c
        s = r["shrunk"][0]
        assert s["file"] == "data.txt" and s["base_lines"] == 100 and s["new_lines"] == 10
        assert s["verdict"] == "moved" and s["moved_into"][0]["file"] == "data_extra.txt"
        assert s["moved_into"][0]["fraction"] == 1.0, s

        # -- LEG 4, unparseable. ONE bucket entry, and -- the prototype's bug -- NOT one lost
        #    definition per def in the broken file.
        assert c["unparseable_newly"] == 1 and r["unparseable"]["newly"] == ["broken.py"]
        assert "fine" not in [x["name"] for x in r["lost"]], "a syntax error became a lost def"

        # -- NO FALSE POSITIVES anywhere else: the untouched names are silent.
        touched = {x["name"] for x in r["lost"]} | set(sig)
        assert touched & {"kept", "CONST", "os", "already"} == set(), touched

        assert r["verdict"] == "LOSSES FOUND", r["verdict"]

        # -- AMBIGUITY IS REFUSED, NOT GUESSED (the merge_trees house rule).
        for bad in ("definitely-not-a-ref-or-a-dir",):
            try:
                merge_census(bad, n)
                raise AssertionError("a base that is neither a dir nor a ref was accepted")
            except ValueError:
                pass

        # -- DETERMINISM: same trees, same report, byte for byte.
        import json as _json
        assert _json.dumps(merge_census(b, n, base_is_ref=False), sort_keys=True) == \
            _json.dumps(r, sort_keys=True)
    return {"faults": len(CENSUS_FAULTS), "lost_unexplained": 1, "lost_moved": 1,
            "signature_changed": 3, "shrunk_moved": 1, "unparseable_newly": 1,
            "false_positives": 0}


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

    # 9. THE POST-MERGE CENSUS, on a fixture with eight KNOWN injected faults. Detection is
    #    counted per leg and the false-positive count is asserted at 0 -- the import-alias
    #    promotion and the moved data block are both in the fixture precisely because each one
    #    is a shape a naive census gets WRONG, loudly, on real merges.
    cen3 = _census_selftest()
    assert cen3 == {"faults": 8, "lost_unexplained": 1, "lost_moved": 1, "signature_changed": 3,
                    "shrunk_moved": 1, "unparseable_newly": 1, "false_positives": 0}, cen3

    print("OK: holographic_codestructure self-test passed (every statement subtree reconstructs bit-exactly and the "
          "module rebuilds to the normalized source; `a + b` and `x + y` share a shape while `a * b` does not; a "
          "mismatched delta RAISES; and the kept negative holds -- structure %d bytes vs zlib's %d, because an exact "
          "decomposition is not a compressor)" % (rep["structure_bytes"], rep["zlib_raw"]))


if __name__ == "__main__":
    _selftest()

"""consolidation_scout.py -- a small, readable map of the codebase for the consolidation effort.

It answers two questions the backlog keeps asking, across EVERYTHING (engine modules, tests, tour, experiments):

  1. "Where is the same code written more than once?" -- group functions/methods by a NORMALISED shape of their
     body (the sequence of operations + the names of the functions they call, with literals and local variable
     names ignored). Functions that land in the same group are structural twins -- prime consolidation targets.

  2. "Who implements <concept>?" -- e.g. nearest-neighbour search. Find every function whose body CALLS the
     tell-tale operations of that concept (argsort/argmax over a distance/cosine, topk, etc.), so a home can route
     them instead of each carrying its own copy.

Pure stdlib (ast), deterministic, no third-party dependency -- the same constraints as the engine. Usage:
    python tools/consolidation_scout.py dupes [--min-ops 12] [--min-size 2]
    python tools/consolidation_scout.py concept nearest   # a built-in concept, or: concept --calls argsort,argmax,dot
"""
import ast
import os
import sys
import hashlib
from collections import defaultdict

# built-in concept fingerprints: a concept is present in a function if its body CALLS any of these names (by
# attribute or bare name). Deliberately broad -- the scout casts a wide net; a human reads the shortlist.
CONCEPTS = {
    "nearest": ["argsort", "argmax", "argpartition", "topk", "top_k", "nearest", "knn", "recall", "cosine",
                "cdist", "pairwise", "query", "kneighbors"],
    "cache": ["bake", "precompute", "cache", "memo", "lookup", "lru", "residency", "lut"],
    "field": ["sample_field", "trilinear", "advect", "grid", "voxel", "sdf", "reconstruct"],
    "blend": ["slerp", "lerp", "mix", "blend", "bundle", "superpose", "composite", "over_composite"],
    "transform": ["bind", "permute", "rotate", "warp", "backwardwarp", "transform_points", "apply_transform"],
    "denoise": ["denoise", "atrous", "bilateral", "svgf", "cleanup", "consolidat", "nlm", "wavelet"],
    "distribute": ["partition", "map_reduce", "reduce", "scatter", "monoid", "bucket", "lpt"],
}


def _iter_py(root):
    """Every .py file under root, skipping caches / vendored / build dirs."""
    skip = {"__pycache__", ".git", "build_pkg", "dist", "node_modules", "verify"}
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip and not d.endswith(".egg-info")]
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


def _norm_body(node):
    """A NORMALISED token list of a function body: node types + the names of called functions/attributes, with
    literals and local names dropped. Two functions with the same list do structurally the same work."""
    toks = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            fn = n.func
            if isinstance(fn, ast.Attribute):
                toks.append("call:" + fn.attr)                    # e.g. .argsort(...)  -> call:argsort
            elif isinstance(fn, ast.Name):
                toks.append("call:" + fn.id)                      # e.g. cosine(...)    -> call:cosine
            else:
                toks.append("call:?")
        elif isinstance(n, ast.BinOp):
            toks.append("op:" + type(n.op).__name__)
        elif isinstance(n, (ast.For, ast.While, ast.If, ast.comprehension, ast.Return, ast.Assign,
                            ast.AugAssign, ast.Subscript)):
            toks.append(type(n).__name__)
    return toks


def _funcs(path):
    """Yield (qualname, lineno, node) for every function/method defined in `path`."""
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError:
        return
    stack = [("", tree)]
    while stack:
        prefix, parent = stack.pop()
        for child in ast.iter_child_nodes(parent):
            if isinstance(child, ast.ClassDef):
                stack.append((prefix + child.name + ".", child))
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield (prefix + child.name, child.lineno, child)
                stack.append((prefix + child.name + ".", child))   # nested defs


def cmd_dupes(root, min_ops=12, min_size=2):
    """Group functions by normalised-body hash; report groups (structural twins) with >= min_size members whose
    bodies have >= min_ops operations (skip trivial one-liners). Ordered by group size, then op count."""
    groups = defaultdict(list)                                    # hash -> [(file, qual, lineno, nops)]
    for path in _iter_py(root):
        rel = os.path.relpath(path, root)
        for qual, lineno, node in _funcs(path):
            toks = _norm_body(node)
            if len(toks) < min_ops:
                continue
            h = hashlib.sha1(("|".join(toks)).encode()).hexdigest()[:12]
            groups[h].append((rel, qual, lineno, len(toks)))
    big = [(h, g) for h, g in groups.items() if len(g) >= min_size]
    big.sort(key=lambda hg: (-len(hg[1]), -hg[1][0][3]))
    print("STRUCTURAL TWINS: %d groups of >= %d functions (>= %d ops each)\n" % (len(big), min_size, min_ops))
    for h, g in big[:40]:
        files = sorted(set(x[0] for x in g))
        print("  [%s] x%d  (%d ops)  across %d file(s):" % (h, len(g), g[0][3], len(files)))
        for rel, qual, lineno, nops in sorted(g)[:8]:
            print("      %s:%d  %s" % (rel, lineno, qual))
        if len(g) > 8:
            print("      ... and %d more" % (len(g) - 8))
    return big


def cmd_concept(root, calls):
    """Find every function whose body calls any name in `calls`. Reports engine modules separately from tests/tour/
    experiments, since a module implementing the concept is a consolidation target while a test just uses it."""
    want = set(c.lower() for c in calls)
    hits = []
    for path in _iter_py(root):
        rel = os.path.relpath(path, root)
        for qual, lineno, node in _funcs(path):
            called = set()
            for n in ast.walk(node):
                if isinstance(n, ast.Call):
                    fn = n.func
                    nm = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else "")
                    if nm:
                        called.add(nm.lower())
            match = sorted(w for w in want if any(w in c for c in called))
            if match:
                hits.append((rel, lineno, qual, match))

    def bucket(rel):
        base = os.path.basename(rel)
        if base.startswith("test_"):
            return "tests"
        if base == "tour.py":
            return "tour"
        if base.startswith("holographic_") or base == "lecore.py":
            return "engine"
        return "experiments/other"
    by = defaultdict(list)
    for h in hits:
        by[bucket(h[0])].append(h)
    print("CONCEPT hits for calls ~ %s : %d functions\n" % (", ".join(sorted(want)), len(hits)))
    for b in ("engine", "tour", "experiments/other", "tests"):
        rows = by.get(b, [])
        if not rows:
            continue
        print("  --- %s (%d) ---" % (b, len(rows)))
        for rel, lineno, qual, match in sorted(rows)[:30]:
            print("      %s:%d  %s   [%s]" % (rel, lineno, qual, ",".join(match)))
        if len(rows) > 30:
            print("      ... and %d more" % (len(rows) - 30))
        print()
    return hits


def main(argv):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if not argv or argv[0] == "dupes":
        args = argv[1:]
        min_ops = int(args[args.index("--min-ops") + 1]) if "--min-ops" in args else 12
        min_size = int(args[args.index("--min-size") + 1]) if "--min-size" in args else 2
        cmd_dupes(root, min_ops=min_ops, min_size=min_size)
    elif argv[0] == "concept":
        rest = argv[1:]
        if "--calls" in rest:
            calls = rest[rest.index("--calls") + 1].split(",")
        elif rest and rest[0] in CONCEPTS:
            calls = CONCEPTS[rest[0]]
        else:
            print("concept needs a built-in name (%s) or --calls a,b,c" % ", ".join(CONCEPTS))
            return
        cmd_concept(root, calls)
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])

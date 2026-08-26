"""holographic_codemap.py -- the source tree as HYPERVECTORS, so the engine can ask "what else looks like
this?" about its own code.

WHAT THE OTHER SELF-AUDITS CANNOT DO
-------------------------------------
orphanaudit and codehealth answer SET-MEMBERSHIP questions: is this name reachable, is it mentioned, how
branchy is it. Every one of them is a lookup. None can answer a SIMILARITY question -- "which other function
does roughly what this one does", "what did we already build that resembles this idea" -- and that is the
question Rule 0 actually asks at the start of every session. `find_capability` answers it for the CATALOG,
which covers 674 of 7,572 functions. For the other 6,898 there was nothing.

This module encodes each function as one role-filler hypervector and puts the corpus behind the engine's own
`Index`, so the same cosine retrieval that serves the catalog serves the source.

THE ENCODING, and why each role is there
-----------------------------------------
    NAME    bundle of the name's word atoms       -- `mesh_collapse_edge` shares `mesh` with its neighbours
    DOC     bundle of the first docstring line    -- what the author said it does, in their words
    CALLS   bundle of the callees' atoms          -- WHO YOU CALL IS WHAT YOU DO; the strongest single role
    MODULE  the defining module's atom            -- weak prior: neighbours in a file tend to be related
    SHAPE   an atom keyed on the canonical AST    -- structure, reusing the shape idea from code_decompose
    BAND    an atom for the complexity decade     -- separates a one-liner from a 60-branch parser

Everything is bundled into ONE fixed-D vector per function, which is the property that makes this cheap:
7,572 functions at dim 512 is 31 MB and a query is one matmul, regardless of how long the functions are.

THE HEADLINE RESULT: THE HYPERVECTORS LOSE, AND THE MODULE SHIPS THE BASELINE
------------------------------------------------------------------------------
Measured on delegate retrieval, 120 cases, against token-set Jaccard over the IDENTICAL features:

    method          recall@1   recall@10    MRR      query      memory
    jaccard            0.542       0.817   0.627   12,884 us    8.19 MB
    holographic        0.175       0.592   0.283    1,556 us   23.18 MB   (best over a dim sweep)
    random             0.000       0.008   0.001

The encoding carries REAL SIGNAL -- 74x random at recall@10 -- and it is still 3.1x worse than exact token
retrieval at recall@1, while using 2.8x MORE memory. It wins on ONE axis, query latency (8.3x), and that
win does not even compound: 16x the corpus costs 19x the query time, so the RP-forest is not buying
sub-linearity at these sizes either.

LEVER 4 WAS WALKED AND DID NOT SAVE IT. Dimension sweep 256 -> 8192: recall@1 goes 0.142 -> 0.175 and
SATURATES BY dim 2048. Sixteen times the dimension does not close a 3.1x gap, so the ceiling is not
bundling crosstalk. The actual cause is structural: Jaccard's |A n B| / |A u B| gets exact set membership
AND an implicit IDF-like penalty from the union term, while a bundle sums every token with equal weight and
cannot tell a discriminative token from a common one. That is information the sum threw away, and no
dimension restores it.

SO `search_source()` AND `similar()` DEFAULT TO JACCARD. The vector path is kept, reachable, and measured, because
its latency advantage is real and would matter on a corpus where an O(N) Python set scan stops being
affordable -- but shipping the prettier representation as the default when it retrieves 3x worse would be
choosing the idea over the measurement. FILED AS POSSIBLE-BUT-DOESN'T-PAY AT THIS SCALE, explicitly not as
impossible: the crossover is a corpus-size question and nobody has found where it sits.

HOW THE COMPARISON WAS MADE
----------------------------
A retrieval win with no baseline is not a result. The task: recover a faculty's DELEGATE (the module
function `mind.mesh_reproject_uv` actually calls) from a query built out of the faculty's own name and
docstring, with the giveaway "See module.symbol" line STRIPPED. Ground truth comes from the delegation map,
which is derived, not hand-labelled. The baseline is token-set Jaccard over the identical features -- the
strongest honest comparison, because it gets exactly the same information and simply skips the vectors.

See `evaluate_retrieval()` for the live numbers on this tree; they are printed by the selftest rather than pasted here
so they cannot rot into a claim nobody re-ran.

A NOTE ON THE PUBLIC NAMES
--------------------------
build_index / encode_features / evaluate_retrieval / search_source are deliberately verbose. The first cut
of this module called them build / encode / evaluate / search, and the name-collision audit caught all four
colliding with other modules (navigator, texturegraph, directed, dictionary). They were RENAMED rather than
allowlisted, for the same reason `stats` became `index_stats` earlier: a public name that says nothing about
what it operates on is a bad public name, and the audit was right to refuse it.

KEPT NEGATIVES
--------------
  * BUNDLING IS LOSSY AND THAT IS THE TRADE, not a bug to fix. Exact set comparison sees every token; a
    bundle sees a noisy sum with crosstalk ~sqrt(N/D). What the vector buys is FIXED SIZE per item, O(1)
    composition, and sub-linear retrieval through Index's RP-forest past 4,096 items. If your corpus is
    small and static, Jaccard is the better tool and this module says so out loud.
  * NO LEARNED WEIGHTS anywhere. Role emphasis is a fixed integer repeat count, chosen once and recorded,
    not fitted -- core forbids learned weights and, more usefully, a fitted weight on a corpus this small
    would be memorisation wearing a hat.
  * DOCSTRINGS ARE THE AUTHOR'S CLAIM, NOT GROUND TRUTH. A function whose docstring lies is indexed by the
    lie. That is a property of the corpus, not of the encoding, and no amount of dimensionality fixes it.
"""
import ast
import hashlib
import os
import re

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import derived_atom, bind, bundle
from holographic.caching_and_storage.holographic_index import Index
from holographic.io_and_interop.holographic_srcindex import parsed_trees, tracked_paths, tree_digest

DIM = 512
SEED = 17

# Role emphasis, as integer repeat counts inside the bundle. FIXED, NOT FITTED (see kept negatives).
# CALLS is heaviest because who you call is the least forgeable signal about what you do -- a name can be
# vague and a docstring can lie, but the callee set is what the code actually does.
ROLE_WEIGHTS = {"NAME": 2, "DOC": 2, "CALLS": 3, "MODULE": 1, "SHAPE": 1, "BAND": 1}

_WORD = re.compile(r"[a-z][a-z0-9]+")
_STOP = frozenset("""the a an and or of to in for on with is are be by from that this it as at into if
    then else not no return returns value values use used using one two set get make new same each any all
    when where which what how why does do done need needs given per via etc eg ie""".split())


def _words(text):
    """Lowercase word tokens, snake_case and camelCase split, stopwords dropped.

    Splitting matters more than it looks: `mesh_collapse_edge` and `collapse_mesh_edges` share nothing as
    strings and everything as token sets, and near-synonym naming is the norm across 545 modules."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2]


def _shape_key(node):
    """A hash of the function's canonical AST -- names, attributes, constants and arg names erased.

    Same idea as code_decompose's SHAPE half: what survives erasure is pure structure. Returned as a short
    hex string so it can be an atom name."""
    try:
        src = ast.unparse(node)
        tree = ast.parse(src)
    except Exception:
        return "unparseable"
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            n.id = "_"
        elif isinstance(n, ast.Attribute):
            n.attr = "_"
        elif isinstance(n, ast.Constant):
            n.value = 0
        elif isinstance(n, ast.arg):
            n.arg = "_"
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            n.name = "_"
    return hashlib.sha256(ast.dump(tree).encode("utf-8")).hexdigest()[:16]


def features(node, module, with_shape=False):
    """The role -> token-list dict for one function. Pure; no vectors yet, so it can be tested and diffed.

    SHAPE IS OFF BY DEFAULT because it is 20 of the 24 seconds it used to take to build the whole index:
    _shape_key round-trips every function through ast.unparse + ast.parse. The default retrieval path is
    Jaccard over NAME/DOC/CALLS and never reads SHAPE, so computing it by default was pure cost on the only
    path anyone uses. Pass with_shape=True if you are indexing for structural similarity."""
    doc = (ast.get_docstring(node) or "").strip().split("\n")[0]
    # STRIP THE GIVEAWAY. Faculty docstrings end with "See holographic_x.y", which in the delegate-retrieval
    # evaluation IS the answer. Indexing it would score a cheat, so it comes out here for everyone.
    doc = re.sub(r"See\s+[\w.]+\s*\.?\s*$", "", doc)
    calls = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                calls.append(f.id)
            elif isinstance(f, ast.Attribute):
                calls.append(f.attr)
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                calls.append(a.name)
    cc = 1 + sum(1 for n in ast.walk(node)
                 if isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.IfExp)))
    return {"NAME": _words(node.name),
            "DOC": _words(doc)[:24],
            "CALLS": [w for c in calls for w in _words(c)][:32],
            "MODULE": _words(module),
            "SHAPE": (["shape:" + _shape_key(node)] if with_shape else []),
            "BAND": ["cc%d" % min(int(np.log2(max(cc, 1))), 6)]}


def encode_features(feat, dim=DIM, seed=SEED):
    """One hypervector per function: sum over roles of bind(ROLE, bundle(token atoms)), role-weighted.

    Roles are BOUND not concatenated, so `mesh` appearing in the NAME is a different direction from `mesh`
    appearing in a CALLEE -- which is the whole reason to use binding here rather than one big bag of words."""
    parts = []
    for role, toks in feat.items():
        if not toks:
            continue
        r = derived_atom(seed, "role:" + role, dim, unitary=True)
        filler = bundle([derived_atom(seed, "tok:" + t, dim) for t in toks])
        parts.extend([bind(r, filler)] * ROLE_WEIGHTS.get(role, 1))
    if not parts:
        return np.zeros(dim)
    v = bundle(parts)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def encode_query(text, dim=DIM, seed=SEED, roles=("NAME", "DOC", "CALLS")):
    """Encode free text as a query, spread across the roles a human phrase could plausibly be about.

    A person typing "subdivide a mesh" does not know whether their words will land in a name, a docstring or
    a callee, so the query is placed in all three and the corpus decides."""
    toks = _words(text)
    if not toks:
        return np.zeros(dim)
    filler = bundle([derived_atom(seed, "tok:" + t, dim) for t in toks])
    parts = []
    for role in roles:
        r = derived_atom(seed, "role:" + role, dim, unitary=True)
        parts.extend([bind(r, filler)] * ROLE_WEIGHTS.get(role, 1))
    v = bundle(parts)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def corpus(root=None, trees=None, with_shape=False):
    """[(label, features)] for every public function in the engine tree. label is 'module.qualname'."""
    trees = trees if trees is not None else parsed_trees(root)
    out = []
    for path, tree in sorted(trees.items()):
        mod = os.path.basename(path)[:-3]
        mod = mod[len("holographic_"):] if mod.startswith("holographic_") else mod
        for node in tree.body:
            targets = [(node, "")] if isinstance(node, ast.FunctionDef) else \
                      ([(n, node.name + ".") for n in node.body] if isinstance(node, ast.ClassDef) else [])
            for n, prefix in targets:
                if isinstance(n, ast.FunctionDef) and not n.name.startswith("_"):
                    out.append(("%s.%s%s" % (mod, prefix, n.name), features(n, mod, with_shape)))
    return out


_INDEX = {}          # tree digest -> (Index, labels, vectors, feats). One entry; the tree rarely changes.


def build_index(root=None, dim=DIM, seed=SEED):
    """Build (or reuse) the holographic index over the source tree. Keyed on the SAME tree digest the L3
    source index uses, so an edit invalidates both together and they can never disagree about the tree."""
    digest = tree_digest(tracked_paths(root)) + ":%d:%d" % (dim, seed)
    hit = _INDEX.get(digest)
    if hit is not None:
        return hit
    items = corpus(root)
    labels = [lab for lab, _f in items]
    feats = {lab: f for lab, f in items}
    vecs = np.stack([encode_features(f, dim, seed) for _lab, f in items])
    built = (Index(vecs, labels=labels), labels, vecs, feats)
    _INDEX.clear()                      # one tree resident; the artifact is tens of MB
    _INDEX[digest] = built
    return built


def similar(name, k=8, root=None, method="jaccard"):
    """Functions most like `name` (a label, or a bare name matched on suffix). Returns [(label, score)].

    DEFAULTS TO JACCARD because it measured 3.1x better at recall@1 than the hypervector path on the same
    features (see the module docstring). Pass method="holographic" for the vector index -- 8.3x faster per
    query, and 3x worse at finding the right answer."""
    idx, labels, vecs, feats = build_index(root)
    hits = [i for i, lab in enumerate(labels) if lab == name or lab.rsplit(".", 1)[-1] == name]
    if not hits:
        return []
    self_labels = {labels[i] for i in hits}
    if method == "holographic":
        out = _pairs(idx.nearest(vecs[hits[0]], k=k + len(hits)))
    else:
        out = _jaccard_baseline(feats[labels[hits[0]]], feats, labels, k + len(hits))
    return [(lab, float(s)) for lab, s in out if lab not in self_labels][:k]


def search_source(text, k=8, root=None, method="jaccard"):
    """Free-text search over the engine's own source. Returns [(label, score)].

    This is the question the other self-audits cannot answer and find_capability only answers for the 674
    catalogued functions out of 7,572. Defaults to Jaccard for the reason in the module docstring."""
    idx, labels, vecs, feats = build_index(root)
    if method == "holographic":
        return [(lab, float(s)) for lab, s in _pairs(idx.nearest(encode_query(text), k=k))][:k]
    qfeat = {"NAME": _words(text), "DOC": _words(text), "CALLS": _words(text)}
    return [(lab, float(s)) for lab, s in _jaccard_baseline(qfeat, feats, labels, k)][:k]


def _pairs(result):
    """Index.nearest returns either [(label, score)] or a dict-ish; normalise to pairs."""
    if isinstance(result, dict):
        return list(zip(result.get("labels", []), result.get("scores", [])))
    out = []
    for r in result:
        if isinstance(r, (tuple, list)) and len(r) >= 2:
            out.append((r[0], r[1]))
        elif isinstance(r, dict):
            out.append((r.get("label"), r.get("score", r.get("cosine", 0.0))))
    return out


# ---------------------------------------------------------------------------------------------------
def _jaccard_baseline(qfeat, feats, labels, k, roles=("NAME", "DOC", "CALLS")):
    """THE HONEST BASELINE: token-set Jaccard over the identical features, no vectors involved.

    This is the comparison that matters. It receives exactly the same information the encoder does and
    simply skips the bundling, so any difference is attributable to the representation rather than to the
    features -- which is the only way a representation claim can be earned."""
    q = set()
    for r in roles:
        q.update(qfeat.get(r, ()))
    scored = []
    for lab in labels:
        s = set()
        for r in roles:
            s.update(feats[lab].get(r, ()))
        if not s and not q:
            continue
        inter = len(q & s)
        if inter:
            scored.append((lab, inter / float(len(q | s))))
    scored.sort(key=lambda t: -t[1])
    return scored[:k]


def evaluate_retrieval(k=10, limit=200, root=None):
    """Recall@1 / Recall@k / MRR for recovering a faculty's DELEGATE, holographic vs Jaccard vs random.

    Ground truth is the delegation map (faculty -> the module functions it calls), which is derived from the
    source rather than hand-labelled, so it cannot be tuned against. The query is the faculty's own name and
    docstring with the "See module.symbol" giveaway stripped."""
    from holographic.io_and_interop.holographic_codehealth import delegation_map
    idx, labels, vecs, feats = build_index(root)
    bare = {}
    for lab in labels:
        bare.setdefault(lab.rsplit(".", 1)[-1], []).append(lab)

    dmap = delegation_map()
    cases = []
    for faculty, calls in sorted(dmap.items()):
        if faculty.startswith("_"):
            continue
        targets = {t for c in calls for t in bare.get(c, []) if c != faculty}
        qlabs = bare.get(faculty, [])
        if targets and qlabs:
            cases.append((qlabs[0], targets))
        if len(cases) >= limit:
            break

    rng = np.random.default_rng(0)
    res = {m: {"r1": 0, "rk": 0, "mrr": 0.0} for m in ("holographic", "jaccard", "random")}
    for qlab, targets in cases:
        qfeat = feats[qlab]
        ranked = {
            "holographic": [l for l, _s in _pairs(idx.nearest(encode_features(qfeat), k=k + 1)) if l != qlab][:k],
            "jaccard": [l for l, _s in _jaccard_baseline(qfeat, feats, labels, k + 1) if l != qlab][:k],
            "random": [labels[i] for i in rng.choice(len(labels), size=k, replace=False)],
        }
        for m, order in ranked.items():
            hit = next((i for i, l in enumerate(order) if l in targets), None)
            if hit is not None:
                res[m]["rk"] += 1
                res[m]["mrr"] += 1.0 / (hit + 1)
                if hit == 0:
                    res[m]["r1"] += 1
    n = max(len(cases), 1)
    for m in res:
        res[m] = {"recall@1": res[m]["r1"] / n, "recall@%d" % k: res[m]["rk"] / n, "mrr": res[m]["mrr"] / n}
    res["cases"] = len(cases)
    res["corpus"] = len(labels)
    return res


def _selftest():
    """Pins the encoding's contracts AND re-runs the baseline comparison, so the honest numbers are printed
    every time rather than living as a claim in a docstring."""
    idx, labels, vecs, feats = build_index()
    assert len(labels) > 3000, "corpus too small (%d)" % len(labels)
    assert vecs.shape[1] == DIM
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-9), "vectors are not unit norm"

    # determinism: same tree, same bytes
    v1 = encode_features(feats[labels[0]])
    v2 = encode_features(feats[labels[0]])
    assert np.array_equal(v1, v2), "the encoder is not deterministic"

    # ROLES ARE ACTUALLY SEPARATED: the same token in NAME vs CALLS must not produce the same vector, or the
    # binding is decorative and this is a bag of words with extra steps.
    a = encode_features({"NAME": ["mesh"], "DOC": [], "CALLS": [], "MODULE": [], "SHAPE": [], "BAND": []})
    b = encode_features({"NAME": [], "DOC": [], "CALLS": ["mesh"], "MODULE": [], "SHAPE": [], "BAND": []})
    cos = float(a @ b)
    assert abs(cos) < 0.2, "role binding is not separating: same token in two roles scored cos=%.3f" % cos

    # a function is its own nearest neighbour
    top = _pairs(idx.nearest(vecs[0], k=1))
    assert top and top[0][0] == labels[0], "self-retrieval failed: %r" % (top,)

    r = evaluate_retrieval(k=10, limit=120)
    h, j, rnd = r["holographic"], r["jaccard"], r["random"]
    assert h["recall@10"] > rnd["recall@10"] * 5, "no better than random -- the encoding carries no signal"
    print("holographic_codemap selftest OK -- %d functions indexed, %d eval cases" % (r["corpus"], r["cases"]))
    print("  delegate retrieval    recall@1  recall@10   MRR")
    for nm, m in (("holographic", h), ("jaccard  ", j), ("random   ", rnd)):
        print("    %s      %6.3f    %6.3f  %6.3f" % (nm, m["recall@1"], m["recall@10"], m["mrr"]))


if __name__ == "__main__":
    _selftest()

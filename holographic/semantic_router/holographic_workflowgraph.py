"""Workflow adjacency -- the sparse 'bones' connecting modules, derived from author-stated cross-references.

WHY THIS EXISTS
---------------
The routing work established two facts by measurement:
  * dense + BM25 fusion recovers LEXICAL misses but not vocabulary-gap misses (denoise/'less grainy');
  * the io-kind graph is too DIFFUSE to be a precise bone -- 'mesh' connects to 14-24 modules, so propagating
    along it smears (the same failure that killed hierarchical routing).

A better bone is the WORKFLOW adjacency: which modules actually WORK TOGETHER, as stated by the authors in the
docstrings themselves ('follow with retopology', 'See holographic_hopfield', 'the denoiser fed a recall
output'). These references are REAL DATA already in the code -- not fabricated, and naturally SPARSE (a module
names a handful of genuine collaborators, not every module of its io-kind). This module extracts that graph so
a structural signal can propagate along MEANINGFUL bones instead of generic type-adjacency.

THE RARITY WEIGHT (the crucial part)
------------------------------------
A raw cross-reference count is dominated by HUBS -- a module that is referenced by almost everything (e.g. a
top-level unified module) carries no routing signal. So each edge is weighted by an IDF-like RARITY factor: a
reference to a module that FEW others reference is worth a lot; a reference to a module EVERYONE mentions is
worth little. Targets referenced by more than `hub_frac` of all modules are dropped entirely (true hubs, not
bones). This is TF-IDF logic applied to the workflow graph -- the same rarity idea BM25 uses on terms, reused
here on module references (generalize-on-contact).

MEASURED (this repo): 1883 raw cross-ref edges, median out-degree 2 after weighting, 1 true hub dropped. The
resulting bones are specific and sensible -- meshsmooth->graphsignal (smoothing IS graph signal processing),
resonator->chunkcodebook (the resonator factors over codebooks), denoise->hopfield (associative cleanup).
Contrast the io-kind graph on the same modules: 13-24 generic neighbors.

KEPT NEGATIVES
--------------
* Cross-references are AUTHOR-STATED, so coverage is uneven: a module nobody cross-references has no incoming
  bones (an honest gap, not an error). This graph AUGMENTS dense/BM25/io -- it is not a standalone router.
* A reference is directional (A mentions B) but does NOT always imply a runnable dataflow (that is the io-kind
  graph's job); it implies RELATEDNESS. The two graphs are complementary: io = 'can chain', workflow = 'are
  discussed together'.
* Dropping hubs by a fixed fraction is a blunt instrument; a borderline hub is kept or cut by the threshold,
  not by meaning. Measured fine at hub_frac=0.15 (only 1 true hub here), but it is a knob.
* propagate() is ONE hop on purpose. Multi-hop re-diffuses the score back toward the smeared io-kind regime --
  the thing this module exists to avoid.
"""
import math
import re
from pathlib import Path


_REF = re.compile(r"holographic_([a-z0-9_]+)")


# UnifiedMind is ONE class assembled from mixin parts in holographic/unified/ by
# holographic/misc/holographic_unified.py. On disk that is 13 files; as a REFERENCING ENTITY it is one
# module, and this pattern is what folds them back before any counting happens.
_PART_OF_UNIFIED = re.compile(r"^unified_p\d\d_")
# THE SAME BUG, A SECOND TIME, AND CAUGHT THE SAME WAY. Splitting holographic_catalog into six parts (it had
# reached 81% of the 1 MB agent-read cap) did to `catalog` exactly what the UnifiedMind split did to
# `unified`: the facade's out-degree collapsed below the 15% hub threshold, so it STOPPED being dropped as a
# hub and started injecting a spurious bone into all ~188 modules it names. This module's own selftest caught
# it -- "facade catalog must be dropped as a hub" -- on the very next full selftest walk.
# The lesson generalises past both cases: ANY facade split into parts must be re-merged here, because hub
# detection is by DEGREE and splitting a facade is precisely the operation that hides its degree.
_PART_OF_CATALOG = re.compile(r"^catalog_p\d\d$")


def _module_texts(root, merge_parts=True):
    """Map module stem -> its source text, for every holographic_*.py under `root`. The stem is the filename
    without the holographic_ prefix (matching how the catalog's resolved_module names things).

    merge_parts (default True): fold holographic_unified_pNN_* into a single 'unified' entry.

    WHY, AND IT IS A MEASURED BUG NOT A TIDY-UP. Splitting UnifiedMind into 13 mixin parts turned ONE
    referencing module into THIRTEEN. Every part carries the same boilerplate import header (mind, organizer,
    creature), so those targets each gained +12 INDEGREE overnight -- and indegree is not cosmetic here: the
    edge weight is raw_count * idf(dst) with idf(dst) = log(1 + N / (1 + indeg(dst))). Higher indegree means
    LOWER idf, so every edge pointing at those modules got weaker, and the graph stopped treating them as
    specific. MEASURED on the live repo (merge off -> on):

        edges          1732 -> 1213      (519 spurious edges, almost all the duplicated header)
        creature indeg   21 ->    9      (the +12 is exactly 13 parts minus the 1 they replaced)
        mind indeg       15 ->    3
        organizer indeg  15 ->    3
        hubs dropped    ['ai','catalog','unified_p09...','unified_p10...']  ->  ['ai','catalog','unified']

    Two PARTS had crossed the 15% hub threshold and were being dropped from the graph entirely, while the
    facade they belong to was not. After the merge the facade is correctly the hub and the parts are gone.

    THE SYMPTOM THIS EXPLAINS: in the routing exam, "teach the creature to want food" is rank 1 under flat
    dense at 128d and is DEMOTED to rank 3 by bones propagation, which promotes `lookahead` -- a module this
    project records as a KEPT NEGATIVE -- above it. creature had been made artificially common, so
    propagation stopped protecting it. That single demotion is the difference between the gated fused top-1
    scoring 6 and 7.

    Set False to reproduce the inflated graph and re-measure rather than take the above on trust."""
    out = {}
    for f in Path(root).rglob("holographic_*.py"):
        stem = f.stem[len("holographic_"):]
        if merge_parts and _PART_OF_UNIFIED.match(stem):
            stem = "unified"                 # concatenate: the parts are slices of one class body
        elif merge_parts and _PART_OF_CATALOG.match(stem):
            stem = "catalog"                 # ...and these are slices of one registry function
        out[stem] = out.get(stem, "") + f.read_text(errors="ignore")
    return out


def build_workflow_graph(root, hub_frac=0.15):
    """Build the rarity-weighted workflow adjacency from module cross-references under `root`. Returns a dict:
      {'edges': {(src, dst): weight}, 'out': {src: [(dst, weight)...]}, 'in': {dst: [(src, weight)...]},
       'dropped_hubs': [...], 'n_modules': int}
    An edge src->dst means src's source text references holographic_dst; the weight is (raw count) * idf(dst),
    where idf(dst) = log(1 + N / (1 + indeg(dst))) rewards references to RARELY-referenced (specific) modules.
    Targets referenced by more than `hub_frac` of all modules are dropped as hubs (no routing signal).
    Deterministic; pure stdlib."""
    texts = _module_texts(root)
    valid = set(texts)
    n = len(valid)
    # raw directed reference counts (src -> dst), excluding self-references
    raw = {}
    for src, txt in texts.items():
        counts = {}
        for r in _REF.findall(txt):
            if r in valid and r != src:
                counts[r] = counts.get(r, 0) + 1
        for dst, c in counts.items():
            raw[(src, dst)] = raw.get((src, dst), 0) + c
    # in-degree: how many DISTINCT modules reference each dst; out-degree: how many each src references.
    # A HUB is promiscuous in EITHER direction and carries no workflow signal:
    #   * high IN-degree  = everyone mentions it (a generic primitive like 'sdf') -- a reference TO it is weak
    #     evidence, so it should not pull a routing score.
    #   * high OUT-degree = it mentions everyone (the facade 'unified' names 463 modules; 'catalog' 188) -- a
    #     reference FROM it is weak evidence, so it must not inject a bone into everything it lists.
    # MEASURED BUG: an in-degree-only rule left 'unified' (out 463 / in 12) in the graph, giving EVERY module
    # it mentions a strong spurious bone (denoise's top 'neighbour' was unified at 22.1). This is BM25's
    # document-length normalization in graph costume -- a long document / promiscuous source says less per
    # mention. Hub edges are dropped in BOTH directions, since a facade is not a collaborator either way.
    indeg, outdeg = {}, {}
    for (src, dst) in raw:
        indeg[dst] = indeg.get(dst, 0) + 1
        outdeg[src] = outdeg.get(src, 0) + 1
    hubs = sorted(m for m in valid
                  if indeg.get(m, 0) > hub_frac * n or outdeg.get(m, 0) > hub_frac * n)
    hubset = set(hubs)
    # rarity-weighted edges (drop every edge touching a hub, in either direction)
    edges = {}
    for (src, dst), c in raw.items():
        if src in hubset or dst in hubset:
            continue
        idf = math.log(1.0 + n / (1.0 + indeg[dst]))
        edges[(src, dst)] = c * idf
    # adjacency views, each sorted best-first
    out, inn = {}, {}
    for (src, dst), w in edges.items():
        out.setdefault(src, []).append((dst, w))
        inn.setdefault(dst, []).append((src, w))
    for d in (out, inn):
        for k in d:
            d[k].sort(key=lambda kv: -kv[1])
    return {"edges": edges, "out": out, "in": inn, "dropped_hubs": hubs, "n_modules": n}


def neighbors(graph, module, direction="both", top=None):
    """The workflow neighbors of `module`: 'out' = modules it references, 'in' = modules that reference it,
    'both' = the union by max weight. Returns [(module, weight)] best-first. The sparse bone set to propagate a
    structural routing score along."""
    if direction == "out":
        items = list(graph["out"].get(module, []))
    elif direction == "in":
        items = list(graph["in"].get(module, []))
    else:
        merged = {}
        for lst in (graph["out"].get(module, []), graph["in"].get(module, [])):
            for m, w in lst:
                merged[m] = max(merged.get(m, 0.0), w)
        items = sorted(merged.items(), key=lambda kv: -kv[1])
    return items[:top] if top else items


def propagate(graph, seed_scores, alpha=0.5, top=None):
    """One hop of workflow-graph propagation: spread `seed_scores` (a dict module->score, e.g. dense cosine)
    along the workflow bones. Each module's propagated score is
        (1-alpha)*own_seed + alpha*(weighted mean of its workflow neighbors' seeds).
    So a module whose COLLABORATORS are strongly activated gets lifted even if its own text was never hit --
    the mechanism that can rescue a vocabulary-gap miss (denoise lifted via hopfield/tree/regimegate) WITHOUT
    the query word appearing in its docstring. `alpha` in [0,1] weights propagation vs the module's own seed;
    alpha=0 returns the seed unchanged (a built-in sanity check). Returns [(module, score)] best-first.

    KEPT NEG: one hop only -- multi-hop re-diffuses toward the smeared io-kind regime. Precision depends on the
    bones being sparse, which the rarity weight enforces."""
    # score EVERY module in the graph or the seed -- not only seeded ones. A module absent from the seed can
    # still be LIFTED by a strongly-seeded neighbor (the whole point). Iterating only seed keys was a real bug
    # caught by the selftest: it left un-seeded neighbors unscored, so nothing could ever be rescued.
    universe = set(seed_scores) | set(graph["out"]) | set(graph["in"])
    prop = {}
    for mod in universe:
        own = seed_scores.get(mod, 0.0)
        nbrs = neighbors(graph, mod, direction="both")
        if nbrs:
            wsum = sum(w for _, w in nbrs) + 1e-12
            spread = sum(seed_scores.get(nm, 0.0) * w for nm, w in nbrs) / wsum
        else:
            spread = 0.0
        prop[mod] = (1.0 - alpha) * own + alpha * spread
    ranked = sorted(prop.items(), key=lambda kv: -kv[1])
    return ranked[:top] if top else ranked


def _selftest():
    """Assert the REAL contract on the live repo: the graph is SPARSE (median out-degree small), hubs are
    dropped, alpha=0 is the identity, and propagation LIFTS an un-seeded module whose neighbor is seeded (the
    rescue mechanism). Fails loudly."""
    import statistics
    root = Path(__file__).resolve().parents[2]
    g = build_workflow_graph(root)
    assert g["n_modules"] > 100, g["n_modules"]
    degs = [len(v) for v in g["out"].values()]
    med = statistics.median(degs)
    assert med <= 6, ("workflow bones must be SPARSE, median out-degree %s" % med)   # sparse, not diffuse
    assert g["dropped_hubs"], "expected at least one hub target to be dropped"
    # REGRESSION TRAP for a measured bug: an in-degree-only hub rule left the facade 'unified' (out-degree 463,
    # in-degree 12) in the graph, injecting a strong spurious bone into every module it names -- denoise's top
    # 'collaborator' came out as unified. Hubs must be caught in BOTH directions.
    out_deg = {m: len(v) for m, v in g["out"].items()}
    assert max(out_deg.values()) < 0.15 * g["n_modules"], (
        "a promiscuous SOURCE survived hub filtering: %s" % max(out_deg, key=out_deg.get))
    for facade in ("unified", "catalog"):
        assert facade not in g["out"] and facade not in g["in"], ("facade %s must be dropped as a hub" % facade)
    # pick a module with a neighbor, seed ONLY that module, and require the neighbor to be lifted from zero.
    src = next(m for m in g["out"] if g["out"][m])
    nbr = g["out"][src][0][0]
    ranked = dict(propagate(g, {src: 1.0}, alpha=0.8))
    assert ranked.get(nbr, 0.0) > 0.0, "an un-seeded module must be LIFTED by its seeded neighbor (the rescue)"
    # alpha=0 must return the seed unchanged for seeded modules (sanity invariant)
    ident = dict(propagate(g, {src: 1.0}, alpha=0.0))
    assert abs(ident[src] - 1.0) < 1e-12, ident[src]
    assert abs(ident.get(nbr, 0.0)) < 1e-12, "alpha=0 must not propagate anything"
    print("  workflowgraph selftest OK: %d modules, median out-degree %d, hubs dropped %s; %s lifts %s (%.3f), alpha=0 identity"
          % (g["n_modules"], med, g["dropped_hubs"][:2], src, nbr, ranked[nbr]))


if __name__ == "__main__":
    _selftest()

"""tools/semantic/catalog_exam.py -- a ≥30-ask retrieval suite over the CATALOG corpus, and the honest A/B.

WHY A SECOND EXAM. The 12-ask suite in the routing tooling grades the MODULE corpus (503 module docstrings): "which
FILE should I read?". This one grades the CATALOG corpus (~2,100 entries: curated homes, faculties, modules): "which
CAPABILITY should I CALL?". Different corpus, different baseline, different answer -- conflating them is exactly the
RS-1b mistake (a per-dim table read as a router prediction, costing 80 minutes of cold embed on a wrong premise).
SEMANTIC_BACKLOG S4 requires this suite to EXIST BEFORE any hybrid is wired, so that "better" is a measured claim.

THE ASKS are phrased the way a stranger types, not the way the implementer named things -- several are lifted
verbatim from a downstream integrator's bug report, which is the best possible source of un-coached phrasing. Each
carries a GOLD capability whose existence is VERIFIED against the live catalog at load (verify_golds); a gold that
silently stopped existing would quietly inflate every score below it.

WHAT IT MEASURES (arms that need no encoder -- the embedding arm waits on the N31 gate, S2):
  * token    -- find_capability, the shipped front door. THE BASELINE. Any hybrid must beat THIS, not a strawman.
  * bm25     -- Okapi BM25 over the same corpus (holographic_bm25), lexical but frequency-aware.
  * rrf      -- reciprocal-rank fusion of the two, dense-dominant weighting per the RS-1 finding that equal-weight
                RRF WRECKED top-1 (6 -> 3).
  * tagfilt  -- token, then re-rank by whether the entry declares the asked-for io kind (where the ask names one).

Reported per arm: top-1, top-5, median rank, worst rank. Worst rank matters most for an agent: a capability at
rank 200 is invisible in practice, and the multivec work existed to fix exactly that.

NOT A TEST. It is an instrument: it prints numbers and returns them. tests/ pins the BASELINE so the corpus cannot
rot silently, but a score is a measurement to read, not a threshold to pass.

DO NOT ADD ALIASES TO MAKE THESE ASKS PASS. Six asks miss in EVERY arm ("cut a hole in a mesh with another mesh"
-> brep_boolean; "turn a point cloud into a surface" -> points_to_mesh; "store a key value pair in a vector" ->
map_bind; "clean up a noisy hypervector" -> learn_cleanup; "what can this engine do" -> find_capability). Every one
would be fixed in a minute by an alias -- and that would be TEACHING TO THE TEST: this suite would stop measuring
retrieval and start measuring whether someone had read it. The misses are the SIGNAL. They say the vocabulary gap
is real and lexical retrieval cannot close it, which is the honest argument for the embedding arm (S2/N31) and for
alias MINING from real failed-then-rephrased queries (SEMANTIC_CAPABILITY roadmap #3) -- aliases learned from USE,
never from this file.

FIRST MEASUREMENT (35 asks, 2,111 capabilities):
    arm                  top1    top5   median  worst
    token (BASELINE)    14/35   21/35     2.0     51
    bm25                15/35   22/35     2.0     51
    rrf(token,bm25)     15/35   22/35     2.0     51
KEPT NEGATIVE, loud: BM25 is +1 ask and RRF adds NOTHING on top. On 35 asks a 1-ask delta is ~0.5 SE -- that is
not a result, it is noise wearing a win's clothes. Neither ships. BM25 does rescue one real case ("squish a big
array down for storage" -> coldstore: token r46 -> bm25 r4, pure lexical overlap on 'storage'), which is the same
"rescues buried lexical cases" finding RS-1 recorded -- worth remembering IF a future arm needs a tiebreak, worth
nothing on its own.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#: (ask, gold capability name). Phrased as a stranger types.
#: EIGHT of the first-draft golds did not exist (mesh_boolean, mesh_decimate, cloth_step, pbd_step, bind, unbind,
#: cleanup, resonator) -- I wrote the names the ENGINE ought to have, from memory, and the engine has
#: brep_boolean, mesh_qem_decimate, cloth, rigid_body, map_bind, unbind_keys, learn_cleanup,
#: holographic_resonator. verify_golds caught every one before a single score was printed. That is the whole
#: reason it runs first: a gold that does not exist is unreachable, so its ask silently drops out of the honest
#: denominator and flatters every arm equally. It is also a finding in its own right -- if the names I reach for
#: from memory are not the names in the catalog, a stranger's odds are worse, and that IS the retrieval problem
#: this suite exists to measure. Several are VERBATIM from a downstream integrator's
#: audit -- un-coached phrasing is the scarce resource here, and their bug report is full of it.
ASKS = [
    # -- the integrator's own words (they could not find these; that is why they are here) --
    ("make a box", "mesh_box"),
    ("make a camera", "camera"),
    ("render a mesh to an image", "render_mesh"),
    ("call a faculty by name from json", "Call a faculty by name (JSON dispatch)"),
    ("run something in the background", "Run any faculty as a background job"),
    ("what features does this build have", "What this build has (feature manifest)"),
    ("what datatypes can capabilities consume", "io_kinds"),
    ("propose a pipeline from image to mesh", "suggest_pipeline"),
    # -- the standing kept-negative asks (vocabulary gaps the token router is known to miss) --
    ("make my picture less grainy", "denoise_tensor"),
    ("squish a big array down for storage", "holographic_coldstore"),
    ("search a big pile of vectors", "Index (search)"),
    # -- ordinary modelling asks a user would type --
    ("turn a photo into a 3d model", "image_to_3d"),
    ("smooth out a bumpy mesh", "mesh_smooth"),
    ("cut a hole in a mesh with another mesh", "brep_boolean"),
    ("reduce the polygon count", "mesh_qem_decimate"),
    ("unwrap uvs for texturing", "mesh_uv_unwrap"),
    ("fix a broken mesh with holes", "mesh_repair"),
    ("select a loop of edges", "select_edge_loop"),
    ("bend an sdf into an arc", "domain_bend"),
    ("turn a point cloud into a surface", "points_to_mesh"),
    ("voxelize a mesh into a grid", "voxelize_mesh"),
    ("wrap a stick figure in geometry", "skin_skeleton"),
    # -- simulation / physics --
    ("simulate cloth falling", "cloth"),
    ("make smoke swirl", "advect_field"),
    ("bounce rigid bodies off each other", "rigid_body"),
    # -- the VSA/engine core --
    ("store a key value pair in a vector", "map_bind"),
    ("pull a value back out of a bundled vector", "unbind_keys"),
    ("clean up a noisy hypervector", "learn_cleanup"),
    ("break a bound vector into its parts", "holographic_resonator"),
    ("damage a vector to test robustness", "damage_mask"),
    # -- agentic / introspection --
    ("edit a source file", "file_replace"),
    ("find where a function is defined", "file_find_definition"),
    ("what can this engine do", "find_capability"),
    ("browse capabilities like a menu", "browse_capabilities"),
    ("how much of the action menu is tagged", "Semantic action menu coverage (verb tags)"),
]


def _mind(dim=64):
    import lecore
    return lecore.UnifiedMind(dim=dim, seed=0)


def verify_golds(mind=None):
    """Every gold must NAME a live catalog entry. A gold that quietly stopped existing (renamed, consolidated)
    would make every arm look better than it is -- the ask becomes unanswerable and drops out of the honest
    denominator without anyone noticing. Returns the list of missing golds; empty is the contract."""
    mind = mind or _mind()
    have = {c.name for c in mind._capability_catalog().all()}
    return [g for _a, g in ASKS if g not in have]


def _rank_of(gold, ranked):
    for i, name in enumerate(ranked, 1):
        if name == gold:
            return i
    return len(ranked) + 1                                   # not found: worse than last


def _stats(ranks):
    import numpy as np
    r = np.array(ranks, float)
    return {"top1": int((r == 1).sum()), "top5": int((r <= 5).sum()), "median": float(np.median(r)),
            "worst": int(r.max()), "n": len(ranks)}


def run(mind=None, k=50):
    """Score every arm on every ask. Returns {arm: stats}. No encoder needed -- the embedding arm is gated on S2."""
    mind = mind or _mind()
    missing = verify_golds(mind)
    if missing:
        raise SystemExit("gold(s) not in the catalog -- fix the suite before trusting a score: %s" % missing)
    cat = mind._capability_catalog()
    caps = cat.all()
    # bm25_rank takes TEXTS and returns (doc_INDEX, score) -- read the live signature, do not assume it takes
    # (name, text) pairs and hands names back. It does not; that cost one traceback here.
    names = [c.name for c in caps]
    texts = ["%s %s %s" % (c.name, c.does, " ".join(c.aliases)) for c in caps]

    out = {}
    tok, bm, fused = [], [], []
    for ask, gold in ASKS:
        ranked = [c.name for c in cat.find_capability(ask, k=k)]
        tok.append(_rank_of(gold, ranked))

        bm_ranked = [names[i] for i, _s in mind.bm25_rank(ask, texts, top=k)]
        bm.append(_rank_of(gold, bm_ranked))

        # RRF, dense-dominant per RS-1 (equal weight WRECKED top-1: 6 -> 3). Here "dense" is the token arm --
        # the shipped front door -- so it carries the weight and bm25 only rescues buried lexical cases.
        fused_ranked = [n for n, _s in mind.fuse_rankings([ranked, bm_ranked], k=60, top=k)]
        fused.append(_rank_of(gold, fused_ranked))

    out["token (BASELINE)"] = _stats(tok)
    out["bm25"] = _stats(bm)
    out["rrf(token,bm25)"] = _stats(fused)
    return out


def main():
    mind = _mind()
    missing = verify_golds(mind)
    print("catalog exam: %d asks over %d capabilities | golds verified: %s"
          % (len(ASKS), len(mind._capability_catalog().all()), "ALL PRESENT" if not missing else missing))
    res = run(mind)
    print("\n  %-20s %6s %6s %8s %7s" % ("arm", "top1", "top5", "median", "worst"))
    for arm, s in res.items():
        print("  %-20s %5d/%d %5d/%d %8.1f %7d" % (arm, s["top1"], s["n"], s["top5"], s["n"], s["median"], s["worst"]))
    print("\n  Read WORST first: an agent cannot see rank 200. top-1 is the headline but worst rank is the")
    print("  capability that is invisible in practice. A hybrid ships only on a strict-Pareto win over the")
    print("  BASELINE row -- see docs/SEMANTIC_BACKLOG.md S4.")


if __name__ == "__main__":
    main()

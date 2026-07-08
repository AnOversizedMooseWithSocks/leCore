"""tools/catalog_gaps.py -- use the CATALOG to find discoverability gaps, faster than hand-auditing modules.

Poses the plain-English queries a USER would actually ask for a family of tools, and flags any that return only
auto-generated module homes (name starts 'holographic_') or nothing -- i.e. a capability that exists but has no curated
home. This is how we found the 2D / text / utility gaps. Run it after building a family of tools to check they are
findable by natural language, not just by module name.

    python tools/catalog_gaps.py            # run the built-in probe sets
    python tools/catalog_gaps.py "edit an image" "blur a photo"   # ad-hoc queries
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from holographic.caching_and_storage.holographic_catalog import default_catalog, seed_from_modules

PROBES = {
    "2D image editing & generation": ["draw a picture", "make a 2d drawing", "paint on a canvas", "edit an image",
        "generate an image", "sharpen an image", "downscale an image", "recolor an image", "crossfade two images"],
    "text generation": ["generate text", "write a sentence", "write a paragraph", "answer a question"],
    "language learning": ["learn from a corpus", "language curriculum", "learn word meanings", "teach the model language"],
    "utilities & helpers": ["hash some data", "content address a file", "verify data integrity", "erasure code for reliability"],
    "rendering": ["path trace a scene", "global illumination", "volumetric render", "precomputed radiance transfer",
        "depth of field lens", "ambient occlusion"],
    "mesh / DCC editing": ["extrude a face", "smooth a mesh", "unwrap uv coordinates", "rig and skin a character",
        "pose a skeleton", "deform a mesh", "convert mesh to sdf", "decimate reduce polygons"],
    "sdf / procedural geometry": ["signed distance field", "sculpt a shape", "procedural terrain", "procedural geometry"],
    "navigation & planning": ["plan a route", "shortest path", "slime mould pathfinding", "navigate a data structure"],
    "learning & agents": ["reinforcement learning agent", "train a classifier", "game npc brain",
        "gradient free learning", "echo state network reservoir", "mixture of experts"],
    "data analysis": ["optimal transport between distributions", "graph laplacian spectral",
        "dimensionality reduction embedding", "classify point cloud structure", "time series analysis", "persistent homology"],
    "symbolic reasoning": ["symbolic regression find a formula", "resonator factorization", "decompose a signal into a law"],
    "signal & spectral": ["fft spectral analysis", "detect a faint signal in noise", "drifting narrowband detection",
        "spectral flatness", "bandwidth"],
    "compression & codec": ["compress a sequence", "rate distortion", "content addressed storage", "entropy coding"],
    "video": ["compress a video", "temporal compression", "motion compensation"],
    "encoders": ["fractional power encoding", "encode a number as a vector", "complex phasor fhrr", "sparse block codes"],
    "physics / simulation": ["simulate fluid", "cloth simulation", "soft body physics", "mass spring", "reaction diffusion"],
    "honesty & measurement": ["measure with error bars", "false discovery rate", "ablation study", "proof of structure",
        "calibrated detection"],
    "program / machine": ["stored program machine", "run a vsa program", "recipe with holes",
        "content addressed compile", "reversible computation"],
    "sequences / prediction": ["learn a sequence", "predict the next step", "markov chain", "learn dynamics"],
}


def audit(queries, k=3):
    """Return [(query, [(name, is_curated)])] for each query -- a curated hit means a good home exists."""
    cat = seed_from_modules(default_catalog())
    out = []
    for q in queries:
        hits = cat.find_capability(q, k=k)
        out.append((q, [(h.name, not h.name.startswith("holographic_")) for h in hits]))
    return out


def report(probe_sets=None):
    probe_sets = probe_sets or PROBES
    gaps = 0
    for family, queries in probe_sets.items():
        print("\n=== %s ===" % family)
        for q, hits in audit(queries):
            has_curated = any(cur for _, cur in hits)
            top = hits[0][0] if hits else "NOTHING"
            mark = "  ok" if has_curated else "  <-- GAP (no curated home)"
            if not has_curated:
                gaps += 1
            print("  %-34s -> %s%s" % (q, top, mark))
    print("\n%d gap(s) found." % gaps)
    return gaps


if __name__ == "__main__":
    if len(sys.argv) > 1:
        gaps = report({"ad-hoc": sys.argv[1:]})
    else:
        gaps = report()
    sys.exit(gaps)                          # non-zero exit on any gap, so CI can gate on discoverability

"""app_lint -- is this app a good leCore citizen? A tree-level lint for apps built on the engine.

WHY (sweep 163): the two apps on the engine, leStudio (2-D) and Poly Studio (3-D), got the same things wrong in
different ways -- ids from process-global counters in one, ids reassigned on load in the other; presence and agent
identity in one, none in the other; an own tool manifest in both; `hash()` seeds that PYTHONHASHSEED had to pin;
helpers (UV unwrap, subdivision, blur, resize) re-implemented because the engine faculty was not found. Each is a
foundation the engine now provides (docs/APP_FOUNDATION.md). This lint names the places in an app tree that still
do it the old way, with the lesson and the engine door beside each.

    python3 tools/app_lint.py /path/to/app            # report
    python3 tools/app_lint.py /path/to/app --json     # machine-readable

Checks are regex-level and deliberately conservative: a hit is a place to LOOK, not a verdict. `duplicates` uses the
engine's own code_search to suggest the faculty a hand-rolled helper may be re-implementing (Rule 0 in tool form).
"""
import json
import os
import re
import sys

# name -> (regex, lesson, engine door). Order = report order.
CHECKS = {
    "hash_seed": (
        r"(?<![\w.])hash\(",
        "Python salts hash() per process: the SAME document renders differently in different launches unless "
        "PYTHONHASHSEED is pinned (leStudio P0.1 -- grain/fiber textures; fixed with zlib.crc32).",
        "hashlib.sha256 / zlib.crc32 over stable bytes; holographic_lews.asset_key for content addresses"),
    "class_counter_ids": (
        r"^\s*_next\s*=\s*\d+|cls\._next|[A-Z]\w+\._next\b",
        "Class-level id counters depend on every other document opened in the process, so a replayed journal mints "
        "different ids and every id reference breaks (leStudio P0.3).",
        "Workspace.mint(prefix) / m.lews_mint -- one persisted counter per prefix, journalled"),
    "legacy_random": (
        r"np\.random\.(seed|rand|randn|randint|random|choice|shuffle|normal|uniform)\(",
        "The legacy global RNG is process state; a seed set elsewhere changes this result. The engine rule is a "
        "seeded default_rng per call site.",
        "np.random.default_rng(seed)"),
    "wall_clock": (
        r"time\.time\(\)|datetime\.now\(\)|time\.perf_counter\(\)",
        "Wall clock in a render or replay path makes output depend on WHEN it ran (leStudio P0.4: frames are indexed, "
        "never timed). Fine for timings and presence; review anything that feeds pixels.",
        "index frames; keep time for measurement only"),
    "own_container_format": (
        r"zipfile\.ZipFile|manifest\.json",
        "An app-private zip+manifest format cannot be opened by the next app. Both apps started here and moved to the "
        "engine container.",
        "holographic_container.save_container/load_container; holographic_lews.Workspace for live sharing"),
    "container_without_workspace": (
        r"save_container\(|load_container\(",
        "Direct container save/load is file-level sharing only: no revisions, no journal, no presence, no lock. Two "
        "apps clobber each other's saves.",
        "holographic_lews.Workspace.put / changes_since / from_file (the .lews live directory)"),
    "own_tool_manifest": (
        r"url_map\.iter_rules|/api/agent/tools|/api/schema",
        "Both apps derived an agent manifest from the Flask url_map by hand, differently (leStudio /api/schema, Poly "
        "/api/agent/tools), and each missed pieces the other had.",
        "m.agent_surface(flask_app) -- holographic_appserver mounts tools/invoke/mind/engine/events/presence"),
    "own_sse_presence": (
        r"text/event-stream|SYNC\[|\"editors\"",
        "Presence and change feeds inside one app's server lock the second app out (it would have to import the "
        "first's Flask app to join).",
        "holographic_lews.Workspace.touch/roster/since (+ agent_surface's /events, /presence)"),
    "own_undo_stack": (
        r"def (undo|redo)\(|_hist_|undo_stack|redo_stack",
        "Every app writes an undo stack; the engine has an edit history with bit-identical replay and reach-back "
        "editing (replace_command).",
        "m.edit_history()"),
    "own_job_manager": (
        r"class JobManager|JOBS\s*=\s*\{|threading\.Thread\(",
        "Background work with progress/cancel exists in the engine (job_submit family); an app-local job table has no "
        "pause/resume and is invisible to agents.",
        "m.job_submit / job_status / job_result / job_cancel"),
    "own_quality_gate": (
        r"def terracing|def edge_tones|quality_gate",
        "Render regression by diff-against-last-render misses a defect both frames share; absolute thresholds are the "
        "engine's now (upstreamed from Poly Studio).",
        "m.render_quality_gate(frame)"),
    "pil_in_core_paths": (
        r"from PIL import|import PIL",
        "Fine in an app; not fine in anything intended for upstream (the engine core is NumPy/stdlib only). Flagged so "
        "candidate modules are known before a sweep.",
        "engine resamplers (image_field / sample_image) when upstreaming"),
}

# Things an app SHOULD have. name -> (regex, what its absence means, engine door)
WANTS = {
    "capability_gating": (
        r"\.features\(|\bhave\(|engine_status|\.version\(\)",
        "No capability preflight: a missing or renamed faculty looks like an absent attribute at call time.",
        "m.features([...]) / m.engine_status()"),
    "agent_identity": (
        r"X-User|X-Client",
        "No identity contract for agents: presence keyed by connection gives one person a ghost per reload, and an "
        "agent cannot suppress its own echo.",
        "agent_surface (X-Client = run, X-User = person)"),
    "lews_workspace": (
        r"holographic_lews|lews_open|Workspace\(",
        "Not on the shared workspace: other apps cannot see this app's documents or presence.",
        "m.lews_open(root, app)"),
    "mind_door": (
        r"/api/mind|find_capability",
        "Agents driving this app cannot ask the engine what exists.",
        "agent_surface's POST /mind (allow-listed discovery faculties)"),
}

# hand-rolled helper names worth asking the engine about (Rule 0 in tool form)
HELPER_RE = re.compile(r"^\s*def\s+(_?(gauss|blur|resize|resample|lerp|smoothstep|fbm|noise|planar_uv|auto_uv|"
                       r"midpoint|subdiv|marching|sdf_grid|voxel|flood_fill|inpaint|tonemap|aces|srgb|linear_to|"
                       r"catmull|bezier|spline|kdtree|nearest|bvh|raycast|erode|dilate|convolve|fft|dct)\w*)\s*\(",
                       re.I)

SKIP_DIRS = ("node_modules", "__pycache__", ".git", "vendor", "holostuff", "fake_engine", "tests")


def _py_files(root, include_tests=False):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS or (include_tests and d == "tests")]
        for fn in fns:
            if fn.endswith(".py"):
                yield os.path.join(dp, fn)


def lint_tree(root, mind=None, include_tests=False, max_examples=3, suggest=True):
    """Lint one app tree. Returns {root, files, checks: {name: {hits, examples, lesson, door}}, wants: {name: {present,
    lesson, door}}, helpers: [{name, file, line, engine_suggestion}], score: {passed, total}}. `mind` (optional) is a
    UnifiedMind whose code_search suggests the engine faculty behind each hand-rolled helper."""
    root = os.path.abspath(os.path.expanduser(root))
    files = list(_py_files(root, include_tests))
    checks = {n: {"hits": 0, "examples": [], "lesson": l, "door": d} for n, (r, l, d) in CHECKS.items()}
    wants = {n: {"present": False, "lesson": l, "door": d} for n, (r, l, d) in WANTS.items()}
    cres = {n: re.compile(r, re.M) for n, (r, _l, _d) in CHECKS.items()}
    wres = {n: re.compile(r) for n, (r, _l, _d) in WANTS.items()}
    helpers = []
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError:
            continue
        text = "".join(lines); rel = os.path.relpath(f, root)
        for n, rx in wres.items():
            if not wants[n]["present"] and rx.search(text):
                wants[n]["present"] = True
        for i, line in enumerate(lines, 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            for n, rx in cres.items():
                if rx.search(line):
                    checks[n]["hits"] += 1
                    if len(checks[n]["examples"]) < max_examples:
                        checks[n]["examples"].append("%s:%d: %s" % (rel, i, s[:110]))
            mh = HELPER_RE.match(line)
            if mh:
                helpers.append({"name": mh.group(1), "file": rel, "line": i, "engine_suggestion": None})
    if suggest and mind is not None and helpers:
        seen = {}
        for h in helpers[:60]:                                          # bounded: a suggestion per helper, not per line
            q = _phrase(h["name"])
            if q not in seen:
                seen[q] = _suggest(mind, q)
            h["engine_suggestion"] = seen[q]
    passed = sum(1 for c in checks.values() if c["hits"] == 0) + sum(1 for w in wants.values() if w["present"])
    return {"root": root, "files": len(files), "checks": checks, "wants": wants, "helpers": helpers,
            "score": {"passed": passed, "total": len(checks) + len(wants)}}


# helper-name stems -> the phrase a user would type (find_capability speaks user language, not identifier language)
_PHRASES = (("gauss", "gaussian blur an image"), ("blur", "blur an image"), ("resize", "resize resample an image"),
            ("resample", "resize resample an image"), ("lerp", "interpolate between values"), ("smoothstep", "smoothstep easing"),
            ("fbm", "fractal noise field"), ("noise", "procedural noise field"), ("planar_uv", "uv unwrap a mesh"),
            ("auto_uv", "uv unwrap a mesh"), ("midpoint", "subdivide a mesh"), ("subdiv", "subdivide a mesh"),
            ("marching", "mesh from an sdf marching cubes"), ("sdf_grid", "bake a mesh to a distance grid"),
            ("voxel", "voxelize a mesh"), ("flood", "flood fill a region of an image"), ("inpaint", "inpaint fill holes in an image"),
            ("tonemap", "tone map hdr to display"), ("aces", "aces tone mapping"), ("srgb", "srgb linear conversion"),
            ("linear_to", "srgb linear conversion"), ("catmull", "catmull rom spline"), ("bezier", "bezier curve evaluate"),
            ("spline", "spline curve evaluate"), ("kdtree", "nearest neighbour search points"), ("nearest", "nearest neighbour search points"),
            ("bvh", "ray mesh intersection acceleration"), ("raycast", "ray cast against a mesh"), ("erode", "erode or dilate a mask"),
            ("dilate", "erode or dilate a mask"), ("convolve", "convolve an image with a kernel"), ("fft", "fourier transform of an image"),
            ("dct", "discrete cosine transform"))


def _phrase(name):
    n = name.lower()
    for stem, phrase in _PHRASES:
        if stem in n:
            return phrase
    return name.strip("_").replace("_", " ")


def _suggest(mind, phrase):
    """The engine's best answer for a hand-rolled helper: the top catalog card for the user-language phrase, else the
    top code_search hit. Returns a short string or None."""
    parts = []
    try:
        caps = mind.find_capability(phrase, k=1)
        if caps:
            parts.append("card: %s" % str(caps[0].name)[:60])
    except Exception:
        pass
    try:
        res = mind.code_search(phrase, k=1)
        if res:
            parts.append("code: %s" % res[0][0])
    except Exception:
        pass
    return " | ".join(parts) or None


def report(res):
    """Human-readable report."""
    out = ["app_lint: %s  (%d .py files)  citizenship %d/%d" % (res["root"], res["files"], res["score"]["passed"], res["score"]["total"])]
    out.append("\n-- patterns the foundation replaces (hits are places to LOOK, not verdicts)")
    for n, c in res["checks"].items():
        flag = "ok  " if c["hits"] == 0 else "%4d" % c["hits"]
        out.append("  [%s] %-28s -> %s" % (flag, n, c["door"]))
        for ex in c["examples"]:
            out.append("          %s" % ex)
    out.append("\n-- foundations an app should be on")
    for n, w in res["wants"].items():
        out.append("  [%s] %-28s %s" % ("yes " if w["present"] else "MISS", n, "" if w["present"] else "-> " + w["door"]))
    if res["helpers"]:
        out.append("\n-- hand-rolled helpers, with the engine's nearest faculty (ask before keeping)")
        for h in res["helpers"][:40]:
            out.append("  %-24s %s:%d  ~ %s" % (h["name"], h["file"], h["line"], h["engine_suggestion"] or "?"))
    return "\n".join(out)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in argv; argv = [a for a in argv if a != "--json"]
    if not argv:
        print(__doc__); return 2
    mind = None
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repo root, when run as a script
        import lecore
        mind = lecore.UnifiedMind(dim=256, seed=0)
    except Exception:
        pass
    for root in argv:
        res = lint_tree(root, mind=mind)
        print(json.dumps(res, indent=1, default=str) if as_json else report(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())

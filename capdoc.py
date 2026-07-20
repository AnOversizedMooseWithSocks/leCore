#!/usr/bin/env python3
"""
capdoc.py -- generate CAPABILITIES.md: a plain-language, grouped map of WHAT leCore can do and HOW to call it.

WHY THIS EXISTS (and how it differs from the other doc generators). docgen.py writes REFERENCE.md (every one of the
~280 modules) and apiquickref.py writes API_QUICKREF.md (one line per symbol for the app-building surface). Both answer
"what SYMBOLS exist". This answers a different, friendlier question: "I have a JOB to do -- which capability does it,
and what do I type?" It reads the engine's own capability CATALOG -- the curated 'homes' a person or an agent searches
with mind.find_capability / mind.suggest / mind.route -- and lays them out grouped by theme, each with the one call
that gets you started. It is the front-door menu for both using leCore standalone and building on top of it.

OLD-SCHOOL AND DEPENDENCY-FREE: standard library only. It imports holographic_catalog (a small, deterministic data
structure -- the curated homes; it does NOT import the ~280 engine modules), so it is fast and safe to run in CI.
Deliberately writes NO timestamp, so the output only changes when the CAPABILITIES change -- which keeps the CI
drift-check honest (a date would make it fail every day).

    Run it:   python capdoc.py
    Output:   CAPABILITIES.md
"""
import os

# ------------------------------------------------------------------------------------------------------------
# THEMES. Each curated capability home is filed under the FIRST theme whose keywords appear in the home's name or
# aliases. Edit this ordered table to re-group the menu; anything unmatched lands in "More capabilities" at the end.
# Kept as a plain, readable list on purpose -- no cleverness, just a menu order a newcomer would find sensible.
# ------------------------------------------------------------------------------------------------------------
THEMES = [
    ("Core algebra & datatypes",
     "the five primitives everything is built from -- bind, bundle, cleanup -- and the vector datatype itself",
     ("kernel verbs", "hypervector", "compute", "vsa-native", "transform", "warp", "encoder", "number to vector")),
    ("Discover & drive it (for agents)",
     "let the engine describe and route ITSELF -- suggest a capability for a task, autocomplete, skill cards",
     ("agent skills", "discover", "route")),
    ("Memory, search & recall",
     "store things and get them back by CONTENT, not by exact key",
     ("index", "search", "memory", "cache", "recall", "archive", "forest")),
    ("Geometry, modeling & rendering",
     "build shapes (mesh or SDF), texture and light them, and render to an image",
     ("geometry", "mesh", "sdf", "procedural", "render", "shading", "brdf", "lighting", "shadow", "material",
      "texture", "field", "2d image", "splat", "pipeline")),
    ("Scenes you can describe & adjust",
     "talk a 3-D scene into being, then adjust its named objects in words, and render or simulate it",
     ("scene from description", "scene", "semantic")),
    ("Simulation & physics",
     "step a solver forward -- fluids, smoke, cloth, soft bodies, collisions, reaction-diffusion",
     ("simulation", "physics", "chemistry")),
    ("Language, knowledge & text",
     "generate text, teach the engine language, and look words up in a real vendored dictionary",
     ("text generation", "language learning", "dictionary", "taxonomy")),
    ("Learning & agents",
     "gradient-free learners and agents -- an RL creature, a classifier, a reservoir, mixtures of experts",
     ("learning", "agent")),
    ("Data analysis & signals",
     "analyse data and signals -- transport, graphs, embeddings, topology, FFT, faint-signal detection",
     ("data analysis", "signal", "spectral", "symbolic", "scale", "distribute", "nystrom")),
    ("Compression, codecs & video",
     "shrink data losslessly or by rate-distortion, and handle temporal image sequences",
     ("compression", "codec", "video")),
    ("Honesty & measurement",
     "measure claims honestly -- error bars, ablations, calibrated detection, proof-of-structure",
     ("honesty", "measurement")),
    ("Navigation, planning & programs",
     "find paths, plan routes, and run stored vector programs on the VSA machine",
     ("navigation", "planning", "graph traversal", "program", "machine")),
    ("Run it as a service / distributed",
     "stand leCore up as an HTTP app, and scale work across a farm with jobs you can pause and resume",
     ("api service", "standalone", "distributed", "coordinator", "hardening", "farm", "job lifecycle", "job",
      "database", "query", "workspace", "concurrency", "command runner", "vsa programs")),
]


def _theme_for(cap):
    """Return the theme title a capability home belongs to, or None. Matches WHOLE words (so 'compute' doesn't grab
    'distribute compute'/'precompute') and looks at the home NAME first (curated and distinctive), then its aliases."""
    import re
    name = cap.name.lower()
    aliases = " ".join(cap.aliases).lower()

    def hit(text, keyword):
        return re.search(r"\b" + re.escape(keyword) + r"\b", text) is not None

    for where in (name, aliases):                               # name wins; aliases only break a tie the name didn't
        for title, _blurb, keywords in THEMES:
            if any(hit(where, k) for k in keywords):
                return title
    return None


def generate(root=None):
    """Write CAPABILITIES.md from the catalog's curated homes, grouped by theme. Returns the output path."""
    if root is None:
        root = os.path.dirname(os.path.abspath(__file__))
    from holographic.caching_and_storage.holographic_catalog import default_catalog
    cat = default_catalog()                                     # curated homes only (no module imports)
    homes = [c for c in cat.all() if not c.name.startswith("holographic_")]

    # bucket each home under its theme (stable: themes in table order, homes sorted by name within a theme)
    buckets = {title: [] for title, _b, _k in THEMES}
    leftovers = []
    for c in homes:
        t = _theme_for(c)
        (buckets[t] if t else leftovers).append(c)

    out = []
    out.append("# leCore Capabilities")
    out.append("")
    out.append("*A plain-language menu of what leCore can do and how to start -- auto-generated by `capdoc.py` from "
               "the engine's own capability catalog. For the full module reference see REFERENCE.md; for the "
               "app-building API surface see API_QUICKREF.md.*")
    out.append("")
    out.append("Every entry below is a **capability home**: a job the engine already solves, the one call that gets "
               "you started, and the words you can search it by. You don't have to read this list -- the engine can "
               "find the right home for you at runtime:")
    out.append("")
    out.append("```python")
    out.append("import lecore")
    out.append("mind = lecore.UnifiedMind()")
    out.append("mind.find_capability('search a big pile of vectors')   # -> the best-matching homes")
    out.append("mind.suggest('edit an image')                          # -> homes + a confidence + the call")
    out.append("mind.route('render a scene')                           # -> 'act' with the call, or 'choose' options")
    out.append("```")
    out.append("")
    out.append("Or over HTTP, once you run the service (see SERVICE.md): `GET /skills`, `POST /skills/suggest`, "
               "`POST /skills/route`.")
    out.append("")

    total = 0
    for title, blurb, _k in THEMES:
        group = sorted(buckets[title], key=lambda c: c.name)
        if not group:
            continue
        out.append("## %s" % title)
        out.append("")
        out.append("*%s.*" % blurb)
        out.append("")
        for c in group:
            total += 1
            out.append("### %s" % c.name)
            out.append(c.does + ".")
            out.append("")
            if c.example:
                out.append("```python")
                out.append(c.example)
                out.append("```")
            if c.aliases:
                out.append("*Find it by:* %s" % ", ".join(c.aliases[:8]))
            out.append("")

    if leftovers:
        out.append("## More capabilities")
        out.append("")
        for c in sorted(leftovers, key=lambda c: c.name):
            total += 1
            out.append("### %s" % c.name)
            out.append(c.does + ".")
            if c.example:
                out.append("")
                out.append("```python")
                out.append(c.example)
                out.append("```")
            out.append("")

    # a small footer so a reader knows the count and how to regenerate
    out.append("---")
    out.append("")
    out.append("*%d capability homes. Regenerate this file with `python capdoc.py` (it reads the live catalog, so it "
               "stays in step with the engine).*" % total)

    text = "\n".join(out).rstrip() + "\n"
    dest = os.path.join(root, "CAPABILITIES.md")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    return dest


# Schema version for the machine-readable artifact. BUMP THIS (and update any consumer) whenever the JSON
# shape changes in a backward-incompatible way -- consumers should refuse a major version they don't know.
CAPABILITIES_SCHEMA_VERSION = "1.0"


def generate_json(root=None):
    """Write capabilities.json -- the MACHINE-READABLE sibling of CAPABILITIES.md, built from the same catalog.

    An app that wants to ingest 'what can leCore do' should read THIS, not parse the markdown: it is a stable,
    versioned contract (schema_version + a flat list of capability records), needs no engine import to consume,
    and never drifts from the prose because both come from one catalog in one run. Deterministic and timestamp-
    free (like CAPABILITIES.md) so the CI drift-gate only trips on a real change."""
    import json
    if root is None:
        root = os.path.dirname(os.path.abspath(__file__))
    from holographic.caching_and_storage.holographic_catalog import default_catalog
    cat = default_catalog()
    homes = [c for c in cat.all() if not c.name.startswith("holographic_")]

    records = []
    for c in sorted(homes, key=lambda c: c.name):
        records.append({
            "name": c.name,
            "does": c.does,
            "example": c.example or "",
            "aliases": list(c.aliases),
            "native": bool(c.native),
            "semantic": getattr(c, "semantic", None) or None,  # the File->Export->PNG verb path (None if untagged)
            "method": getattr(c, "method", None) or None,      # C7: verified mind.<method>() name; None = import-only
            "consumes": list(getattr(c, "consumes", ()) or ()),  # S3 io-shape: datatype(s) consumed (empty if untagged)
            "produces": list(getattr(c, "produces", ()) or ()),  # S3 io-shape: datatype(s) produced
            "theme": _theme_for(c) or "More capabilities",
        })

    payload = {
        "schema_version": CAPABILITIES_SCHEMA_VERSION,
        # SCOPE, stated in the artifact itself (a downstream integrator was burned by a generated doc that read a
        # poorer catalog than the engine -- pipelines.json, now fixed). THIS file's curated-only scope is
        # DELIBERATE: default_catalog() imports no engine modules, so capdoc stays stdlib-runnable. The FULL live
        # catalog (~2,100 entries: every auto-registered faculty, verified `method` callability, io edges) is a
        # RUNTIME object -- serve it via mind.find_capability / mind.pipeline_map / GET /tools, do not look for
        # it here. A client that treats this file as the whole engine will under-count by ~5x and miss every
        # faculty-only capability.
        "scope": "curated capability homes only -- the full live catalog is served at runtime by "
                 "mind.find_capability / mind.pipeline_map / GET /tools",
        "count": len(records),
        "capabilities": records,
    }
    # sort_keys so the bytes are deterministic run-to-run (drift-gate friendly); ensure_ascii=False keeps
    # any non-ASCII in descriptions readable rather than escaped.
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    dest = os.path.join(root, "capabilities.json")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    return dest


if __name__ == "__main__":
    p = generate()
    print("wrote", p)
    j = generate_json()
    print("wrote", j)

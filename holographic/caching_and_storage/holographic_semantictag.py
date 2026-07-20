"""holographic_semantictag.py -- infer a capability's SEMANTIC TAXONOMY tag from its name and one-line docstring.

WHY THIS EXISTS. `browse_semantic()` renders the File -> Export -> PNG action menu, and capuri is explicit that
"untagged capabilities are omitted". The tag was optional and hand-written, so coverage was 108 of 2,095 (5.2%):
every one of the 1,203 auto-registered mind METHODS carried no tag, which meant the engine's actual verb surface --
the things a user DOES -- was 95% invisible in its own action tree. Hand-tagging 1,200 methods is not a fix: it is
1,200 chances to rot, and the next faculty lands untagged anyway. The tag has to be DERIVED at registration, the
same way `_METHOD_ALIASES` fixed dark method names (D1).

WHAT IT IS. A deterministic stem -> "root/sub" table, matched against the method name (leading token first, then any
token) and, failing that, the first verb of the docstring. Roots are the TEN in docs/SEMANTIC_TAXONOMY.md and are
never invented here -- that doc says a new root is a design conversation, not a registration. Sub-branches are only
those the doc already grounds.

IT ABSTAINS. `infer_semantic` returns None when no stem matches confidently. That is deliberate and matches the
house rule the semantic router already follows: an honest None beats a confident wrong route. A wrong tag is worse
than a missing one -- it files a capability under a verb a user will never look for, and unlike a missing tag it
looks done. Coverage is therefore a MEASURED outcome, not a target to hit by loosening the table.

NO LEARNED WEIGHTS, no model, no network: a frozen table and string matching, so the same name always yields the
same tag (the engine's determinism rule applies to its own metadata too).
"""

#: Ordered (stems, tag) rules. ORDER IS LOAD-BEARING: earlier rules win, so specific stems ("subdivide") must sit
#: above generic ones ("modify"). Every tag's root is one of the ten in SEMANTIC_TAXONOMY.md, and every sub-branch
#: is one that doc already lists -- this table may not mint taxonomy, only apply it.
_RULES = (
    # --- io/ : crossing the disk/format boundary. FIRST because "export_splats" is io, not render. ---
    (("file",), "io/file"),
    (("export", "save", "write", "dump"), "io/export"),
    (("import", "load", "open", "read"), "io/import"),
    (("workspace", "checkpoint", "snapshot", "session"), "io/workspace"),
    (("obj", "gltf", "glb", "ply", "stl"), "io/import"),

    # --- animate/ : driving values over time (before transform: "pose" is animation, not a gizmo) ---
    (("keyframe", "timeline", "easing", "ease"), "animate/keyframe"),
    (("pose",), "animate/pose"),
    (("skin", "bone", "rig", "weightpaint"), "animate/skin"),
    (("animate", "animation"), "animate/keyframe"),

    # --- simulate/ : evolving a physical field ---
    (("fluid", "smoke", "advect", "vorticity"), "simulate/fluid"),
    (("pbd", "xpbd"), "simulate/pbd"),
    (("cloth",), "simulate/cloth"),
    (("particle", "swarm", "boid", "flock"), "simulate/particles"),
    (("simulate", "step", "evolve", "integrate", "tick", "run", "solve"), "simulate/step"),

    # --- render/ : producing an image ---
    (("raymarch", "spheretrace", "sphere_trace", "march", "ray_sdf"), "render/raymarch"),
    (("raster", "rasterize", "scanline"), "render/raster"),
    (("splat",), "render/splat"),
    (("shade", "shader", "shading", "brdf", "lighting"), "render/shade"),
    (("frame", "serve", "viewport"), "render/frame"),
    (("render", "draw", "paint", "ascii"), "render/raster"),

    # --- convert/ : changing representation ---
    (("voxelize", "voxel"), "convert/voxelize"),
    (("isosurface", "marching", "contour", "dual_contour"), "convert/isosurface"),
    (("uv", "unwrap", "lscm", "atlas"), "convert/uv"),
    (("glsl", "wgsl", "codegen", "compile", "transpile", "emit"), "convert/emit"),
    (("convert", "encode", "decode", "serialize", "quantize", "pack", "unpack", "bake"), "convert/emit"),

    # --- measure/ : reading a quantity off the data ---
    (("distance", "dist"), "measure/distance"),
    (("bounds", "bbox", "aabb", "extent"), "measure/bounds"),
    (("area", "volume", "perimeter", "mass"), "measure/area"),
    (("curvature",), "measure/curvature"),
    (("geodesic",), "measure/geodesic"),
    (("measure", "count", "estimate", "metric", "score", "benchmark", "profile"), "measure/distance"),

    # --- modify/ : changing topology or shape ---
    (("extrude",), "modify/extrude"),
    (("bevel", "fillet", "chamfer"), "modify/bevel"),
    (("inset",), "modify/inset"),
    (("boolean", "csg", "union", "subtract", "intersect_mesh"), "modify/boolean"),
    (("subdivide", "subdiv", "tessellate", "tessellation", "loop_cut"), "modify/subdivide"),
    (("weld", "stitch", "merge", "decimate", "simplify", "remesh"), "modify/weld"),
    (("sculpt", "brush"), "modify/sculpt"),
    (("deform", "bend", "twist", "smooth", "relax", "denoise", "sharpen", "filter",
      "blur", "damage", "perturb", "noise_up", "displace", "blend"), "modify/deform"),

    # --- transform/ : moving without changing topology ---
    (("gizmo",), "transform/gizmo"),
    (("snap",), "transform/snap"),
    (("pivot",), "transform/pivot"),
    (("translate", "move", "offset"), "transform/translate"),
    (("rotate", "rotation", "orient", "quaternion", "quat"), "transform/rotate"),
    (("scale", "resize"), "transform/scale"),
    (("transform", "warp", "reproject"), "transform/translate"),

    # --- select/ : choosing a subset ---
    (("loop", "ring", "boundary"), "select/loop"),
    (("symmetry", "mirror"), "select/symmetry"),
    (("proportional", "falloff", "soft_select"), "select/soft"),
    (("region", "frustum", "lasso"), "select/region"),
    (("select", "pick", "choose"), "select/element"),

    # --- analyze/ : factoring, recalling, reasoning over structure ---
    (("resonator", "factor", "unbind", "decompose"), "analyze/factor"),
    (("recall", "remember", "archive", "retrieve", "memory", "learn", "absorb", "teach"), "analyze/recall"),
    (("spectral", "fourier", "fft", "eigen", "wavelet", "laplacian"), "analyze/spectral"),
    (("capability", "capabilities", "browse", "resolve", "route", "pipeline", "skill",
      "suggest", "catalog"), "analyze/pipeline"),
    (("analyze", "describe", "explain", "audit", "report", "inspect", "diff", "compare",
      "why", "classify", "detect", "recognize", "predict", "verify", "validate",
      "check", "test", "calibrate", "honesty"), "analyze/pipeline"),

    # --- create/ : bringing new data into being. LAST among specifics: "make_*" is the weakest signal. ---
    (("sphere", "box", "cylinder", "torus", "plane", "capsule", "cone", "primitive"), "create/primitive"),
    (("fractal", "procedural", "curve", "spline", "terrain"), "create/procedural"),
    (("scene", "compose"), "create/scene"),
    (("generate", "synth", "spawn", "mint", "make", "create", "new", "build", "init"), "create/procedural"),

    # --- search: after analyze/ so "find_capability" routes to the pipeline branch, not a generic lookup ---
    (("find", "search", "query", "lookup", "match", "nearest", "knn"), "select/element"),
)

#: Docstring verbs are a WEAKER signal than the name -- a doc's first word is often "Return"/"Compute" boilerplate.
#: Only these unambiguous openers are trusted, and only when the name said nothing.
_DOC_VERBS = (
    ("render", "render/raster"), ("draw", "render/raster"), ("simulate", "simulate/step"),
    ("export", "io/export"), ("import", "io/import"), ("load", "io/import"), ("save", "io/export"),
    ("convert", "convert/emit"), ("measure", "measure/distance"), ("select", "select/element"),
    ("rotate", "transform/rotate"), ("translate", "transform/translate"), ("scale", "transform/scale"),
    ("subdivide", "modify/subdivide"), ("smooth", "modify/deform"), ("deform", "modify/deform"),
)

#: The ten roots. Kept in sync with docs/SEMANTIC_TAXONOMY.md and the catalog's S4.1 drift gate BY TEST, not by
#: hope -- tests/test_semantictag.py asserts this set equals the gate's.
ROOTS = frozenset({"create", "select", "transform", "modify", "measure", "convert",
                   "render", "simulate", "animate", "analyze", "io"})


def _tokens(name):
    """Split a capability/method name into lowercase word tokens: 'render_scene_2d' -> ['render','scene','2d']."""
    out, cur = [], ""
    for ch in str(name).lower():
        if ch.isalnum():
            cur += ch
        else:
            if cur:
                out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


def infer_semantic(name, doc=""):
    """Infer a `semantic=` taxonomy tag ('root/sub') for a capability, or None if no rule matches confidently.

    Deterministic and model-free: matches a frozen stem table against the name's LEADING token first (the verb, in
    this codebase's naming convention), then any token, then a small set of unambiguous docstring openers. Returns
    None rather than guessing -- an honest abstention keeps a capability out of the wrong menu branch, which is
    worse than being absent from every branch because it looks correct.

        infer_semantic("render_scene")        -> 'render/raster'
        infer_semantic("subdivide_mesh")      -> 'modify/subdivide'
        infer_semantic("damage_mask")         -> 'modify/deform'
        infer_semantic("frobnicate")          -> None
    """
    toks = _tokens(name)
    if not toks:
        return None
    # ONE pass, in RULE ORDER, over every token. Order is the whole disambiguation strategy: specific stems sit
    # above generic ones, so "measure_area" reaches measure/area before the catch-all measure/ rule. An earlier
    # draft checked the LEADING token first as "the verb" -- it fought the ordering and mis-filed measure_area as
    # measure/distance. Kept as a negative: the name's word ORDER is a weaker signal than the table's precedence.
    tokset = set(toks)
    for stems, tag in _RULES:
        if tokset.intersection(stems):
            return tag
    # 3. the docstring's first word, but only if it is an unambiguous verb.
    first = _tokens(doc)[:1]
    if first:
        for verb, tag in _DOC_VERBS:
            if first[0] == verb:
                return tag
    return None


def coverage(catalog=None):
    """Report tag coverage over a catalog: {'total', 'tagged', 'untagged', 'pct'} -- the number this module exists
    to move, so it is measurable from the engine itself rather than from a one-off script."""
    if catalog is None:
        from holographic.caching_and_storage.holographic_catalog import default_catalog
        catalog = default_catalog()
    caps = catalog.all()
    tagged = sum(1 for c in caps if getattr(c, "semantic", None))
    return {"total": len(caps), "tagged": tagged, "untagged": len(caps) - tagged,
            "pct": round(100.0 * tagged / max(len(caps), 1), 1)}


def _selftest():
    # Every tag this table can emit must sit under a documented root -- the table may APPLY taxonomy, never mint it.
    for stems, tag in _RULES:
        assert tag.split("/")[0] in ROOTS, ("rule emits an unknown root", tag)
        assert stems, ("empty stem list", tag)
    for _v, tag in _DOC_VERBS:
        assert tag.split("/")[0] in ROOTS, ("doc-verb rule emits an unknown root", tag)

    # DETERMINISM and the naming convention: the leading token decides.
    assert infer_semantic("render_scene") == "render/raster"
    assert infer_semantic("render_scene") == infer_semantic("render_scene"), "must be a pure function"
    assert infer_semantic("subdivide_mesh") == "modify/subdivide"
    assert infer_semantic("export_splats") == "io/export", "export_* is io, not render -- rule ORDER is load-bearing"
    assert infer_semantic("skin_mesh") == "animate/skin"
    assert infer_semantic("measure_area") == "measure/area", "specific stem must beat the generic measure/ rule"
    assert infer_semantic("holographic_raypick") is None, "a module name is not an action -- must abstain"

    # ABSTENTION is the contract, not a shortfall: a name with no verb must yield None.
    assert infer_semantic("frobnicate") is None
    assert infer_semantic("") is None
    assert infer_semantic("xyzzy_plugh") is None

    # KEPT NEGATIVE (loud): the docstring is a WEAK signal and is consulted ONLY when the name says nothing.
    # 'Return the bounding box' must NOT become measure/ off the word 'return'; and a name that already matched
    # must never be overridden by its prose.
    assert infer_semantic("frobnicate", "Return the widget") is None, "boilerplate doc verbs must not tag"
    assert infer_semantic("export_splats", "Render the splats") == "io/export", "name must beat doc"
    assert infer_semantic("frobnicate", "Render the thing") == "render/raster", "doc used ONLY as a fallback"

    print("holographic_semantictag selftest OK (every rule under a documented root; rule ORDER disambiguates -- "
          "measure_area beats the generic measure/ rule and export_* stays io; abstains on unknown verbs and on "
          "module names; doc is a fallback that never overrides a name)")


if __name__ == "__main__":
    _selftest()

"""holographic_catalog.py -- the capability CATALOG (consolidation backlog C1): "search before you build".

WHY THIS EXISTS
---------------
The engine has grown to ~340 modules, and the recurring cost is DUPLICATION: the same job (search a pile of
vectors, bake a slow factor and look it up, represent a field) is often already solved somewhere, but a new
session -- human or AI -- can't easily find it and builds a fourth copy. This catalog is the index of what exists.
Describe a problem in plain English and it points you at the home that already does it.

Each entry says, in plain English, what it DOES, gives a copy-paste EXAMPLE, and flags whether it is NATIVE
(stays in the batched / fusable vector domain) or hops to Python. `find_capability` runs a small, READABLE
token-overlap match over the entries -- no training, fully deterministic -- so "search a big pile of vectors"
finds the search index without anyone having to know its module name.

This PROMOTES two shipped things into one home: the auto-listing of the mind's public faculties
(holographic_query.capability_registry, which builds a SQL-queryable table) and query-by-description. The catalog
is the richer home (does/example/native + find); `seed_from_mind` reuses the faculty walk so every faculty is
findable, and `to_rows` can still hand the entries to the SQL table path. NumPy-free, stdlib only.
"""
import re

# small stop-word list so a problem sentence matches on its CONTENT words, not "how do I ..." scaffolding
_STOP = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "with", "how", "do", "i", "my", "is", "are",
    "it", "that", "this", "at", "by", "as", "be", "can", "want", "need", "get", "from", "into", "over", "some",
    "you", "me", "we", "our", "your", "using", "use", "make", "build", "create", "given", "when", "where", "what",
}


def _tokens(text):
    """Lower-cased content words of `text` (drop stop-words and 1-char tokens). Readable and deterministic."""
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in _STOP and len(w) > 1]


class Capability:
    """One catalog entry: a named home, what it DOES (plain English), a copy-paste EXAMPLE, whether it is NATIVE
    (True = batched / fusable / stays in the vector domain; False = crosses to Python), and search `aliases`
    (extra words a problem might use for it -- e.g. 'knn', 'lookup' for the search index)."""

    def __init__(self, name, does, example="", native=True, aliases=()):
        self.name = str(name)
        self.does = str(does)
        self.example = str(example)
        self.native = bool(native)
        self.aliases = tuple(aliases)

    def __repr__(self):
        return "Capability(%r, %s)" % (self.name, "native" if self.native else "python")


class Catalog:
    """The registry + the search. `register_capability` adds/updates an entry; `find_capability` ranks entries
    against a problem description by how many content words they share (favouring entries that cover the query)."""

    def __init__(self):
        self._by_name = {}                                           # name -> Capability (insertion order kept)

    def register_capability(self, name, does, example="", native=True, aliases=()):
        """Add (or replace) a capability. Returns the entry. Additive -- registering the same name again updates it."""
        cap = Capability(name, does, example, native, aliases)
        self._by_name[name] = cap
        return cap

    def get(self, name):
        return self._by_name.get(name)

    def all(self):
        return list(self._by_name.values())

    def __len__(self):
        return len(self._by_name)

    def find_capability(self, problem, k=3):
        """Return up to `k` capabilities whose description best matches `problem`, best first. The score is the
        number of shared content words, normalised by the query length so a short query isn't swamped -- plus a
        small bonus when a query word appears in the entry's NAME (a strong signal). Deterministic ties break by
        name so the result is stable run-to-run (the engine's determinism rule)."""
        q = set(_tokens(problem))
        if not q:
            return []
        scored = []
        for cap in self._by_name.values():
            name_words = set(_tokens(cap.name))
            hay = name_words | set(_tokens(cap.does)) | set(w.lower() for w in cap.aliases)
            overlap = len(q & hay)
            if overlap == 0:
                continue
            score = overlap + 0.5 * len(q & name_words)              # a name-word hit counts extra
            scored.append((score, cap.name, cap))
        scored.sort(key=lambda s: (-s[0], s[1]))                     # best score first, then name (stable)
        return [cap for _, _, cap in scored[:k]]

    def find_scored(self, problem, k=3):
        """Like find_capability, but returns [(capability, score)] best-first -- so an agentic layer can turn the raw
        match scores into a CONFIDENCE (how dominant the top hit is). Same scoring, same deterministic tie-break."""
        q = set(_tokens(problem))
        if not q:
            return []
        scored = []
        for cap in self._by_name.values():
            name_words = set(_tokens(cap.name))
            hay = name_words | set(_tokens(cap.does)) | set(w.lower() for w in cap.aliases)
            overlap = len(q & hay)
            if overlap == 0:
                continue
            score = overlap + 0.5 * len(q & name_words)
            scored.append((score, cap.name, cap))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [(cap, float(sc)) for sc, _, cap in scored[:k]]

    def to_rows(self):
        """Export entries as plain dict rows (name/does/native) -- e.g. to hand to the SQL capability table."""
        return [{"name": c.name, "does": c.does, "native": c.native} for c in self._by_name.values()]


def seed_from_mind(catalog, mind):
    """Reuse the faculty walk (as holographic_query.capability_registry does) to auto-register every public method
    of `mind` as a catalog entry, using its one-line docstring as `does`. Curated entries (registered explicitly)
    win, because they carry better `does`/`example`/`native` -- we don't overwrite them here."""
    import inspect
    for name in dir(mind):
        if name.startswith("_") or name in catalog._by_name:
            continue
        attr = getattr(type(mind), name, None)
        if not callable(attr):
            continue
        doc = (inspect.getdoc(attr) or "").strip().split("\n")[0][:160]
        catalog.register_capability(name, doc or name, example="mind.%s(...)" % name, native=True)
    return catalog


def default_catalog():
    """A catalog seeded with the CONSOLIDATION HOMES and the key shipped modules the audits named -- the search
    indices, the caches / bakes, and the field types -- so they are findable TODAY. As each consolidation home
    (Index, Cache, Field, ...) lands, register it here with native=True."""
    c = Catalog()

    # --- search / recall: the INDICES (audit named ~7) ---
    c.register_capability(
        "Index (search)", "nearest-neighbour / recall over a pile of vectors with ONE interface (Index.nearest(q,k)): "
        "exact cosine scan for small sets, sub-linear RP-forest for large, plus a calibrated abstain",
        example="from holographic_index import Index; Index(vectors, labels=names).nearest(query, k=5)",
        native=True, aliases=("knn", "nearest", "lookup", "recall", "retrieve", "similarity", "search", "index"))
    c.register_capability("holographic_spatial.knn", "EUCLIDEAN k-nearest over a POINT cloud (a spatial grid) -- a "
                          "different metric than the cosine Index; use for geometry, not vectors",
                          example="SpatialGrid(points).knn(query, k)", native=True, aliases=("spatial", "euclidean", "points", "knn"))
    c.register_capability("holographic_rayindex", "which pixels/objects a RAY touches (ray<->object index) -- not a "
                          "nearest(query,k); a distinct spatial ray structure", example="build_ray_index(ctx, camera, w, h)",
                          native=True, aliases=("ray", "pixels", "reshade", "spatial"))
    c.register_capability("holographic_tree.HoloForest", "sub-linear approximate nearest-neighbour search over many "
                          "vectors (random-projection forest) with cross-tree agreement", example="HoloForest(V).recall(q,k)",
                          native=True, aliases=("forest", "ann", "knn"))
    c.register_capability("holographic_pivot", "recursive pivot-tree index for nearest-neighbour search",
                          example="from holographic_pivot import ...", native=True, aliases=("pivot", "index"))
    c.register_capability("holographic_rayindex", "spatial index built for RAY queries (ray <-> object)",
                          example="from holographic_rayindex import ...", native=True, aliases=("ray", "spatial", "bvh"))
    c.register_capability("holographic_archive", "content-addressable image memory (WHT plates), damage-tolerant",
                          example="from holographic_archive import ...", native=True, aliases=("image", "store", "recall"))

    # --- caching / baking: the CACHES (audit named ~9) = bake_and_query ---
    c.register_capability(
        "Cache (bake-and-query)", "bake a slow evaluator over what VARIES (position/view/time/constant) then look it "
        "up cheaply -- one shared grid-sample core over the scattered bakes (matbake, sdfbake, viewlut, anim)",
        example="from holographic_cachehome import Cache; Cache.bake(fn, vary='position', lo=lo, hi=hi, res=24)",
        native=True, aliases=("bake", "precompute", "lookup", "cache", "memoise", "irradiance", "lut", "grid"))
    c.register_capability("holographic_domecache", "cached DOME / sky-ambient light: bake PRT at coarse anchors, "
                          "smooth interpolate, recompute edges (three-tier)", example="render_scene_document(..., dome_cache=True)",
                          native=True, aliases=("dome", "ambient", "ao", "sky"))
    c.register_capability("holographic_lightcache", "cached SOFT AREA lights + one-bounce INDIRECT / global "
                          "illumination, baked noise-free at anchors (the shared cached_screen_shade engine)",
                          example="render_scene_document(..., soft_light_cache=True, indirect_cache=True)",
                          native=True, aliases=("gi", "indirect", "bounce", "area", "penumbra", "shadow", "speckle"))
    c.register_capability("holographic_modulate", "modulate/demodulate primitive (= bind/unbind): split radiance into "
                          "albedo x irradiance to denoise or upscale the smooth part cleanly",
                          example="from holographic_modulate import demodulate, remodulate", native=True,
                          aliases=("albedo", "irradiance", "denoise", "upscale", "demodulate"))
    c.register_capability("holographic_matbake", "bake POSITION-dependent material channels to a grid, trilinear "
                          "lookup", example="from holographic_matbake import ...", native=True, aliases=("material", "bake"))
    c.register_capability("holographic_prt", "precomputed radiance transfer: bake light transport, relight by a dot "
                          "product", example="from holographic_prt import precompute_transfer, shade_prt", native=True,
                          aliases=("relight", "sh", "transfer", "light"))

    # --- 2D image editing & generation, text generation, language learning, utilities (curated families) ---
    c.register_capability("2D image editing & generation", "the engine's 2D IMAGE toolkit: edit (recolor_image / "
                          "colour transfer, sharpen_loop, svgf_denoise, downscale), generate & blend (blend_images "
                          "crossfade/morph, pattern_field procedural noise/fbm/checker/stripes, svg_canvas vector "
                          "drawing), store & compare (image_archive damage-tolerant recall, compare_images / "
                          "image_distance perceptual similarity). Raster and vector, all on the VSA substrate",
                          example="mind.recolor_image(img, ref); mind.blend_images(a, b); mind.pattern_field('fbm'); mind.svg_canvas()",
                          native=True, aliases=("2d", "image", "edit an image", "generate an image", "draw", "draw a picture",
                                                "make a drawing", "paint", "paint on a canvas", "canvas", "sharpen", "blur",
                                                "downscale", "resize", "recolor", "colour transfer", "color transfer",
                                                "crossfade", "morph", "sprite", "vector graphics", "svg", "procedural texture",
                                                "picture", "photo", "raster", "pixels"))
    c.register_capability("Text generation", "GENERATE text on the VSA substrate: generate(seed, length, temperature) "
                          "and generate_structured (n-gram / beam), respond(query) for a query-conditioned reply, and "
                          "answer(question) / answer_text for factual answers. The engine's write-a-sentence faculties",
                          example="mind.generate('once upon a', length=120); mind.respond('describe a sunset'); mind.answer('what is gravity')",
                          native=True, aliases=("generate text", "write", "write a sentence", "write a paragraph",
                                                "text generation", "compose text", "respond", "reply", "answer a question",
                                                "language model", "ngram", "sentence", "paragraph", "prose"))
    c.register_capability("Language learning", "TEACH the mind language natively: read (read a corpus), "
                          "learn_dictionary / learn_vocabulary (word meaning from definitions -- including the vendored "
                          "dictionary), learn_encyclopedia (relational facts + is_a taxonomy), and learn_sequence "
                          "(order/grammar). The language CURRICULUM -- definitions, then facts, then reading",
                          example="mind.read(corpus); mind.learn_vocabulary(words); mind.learn_encyclopedia(facts)",
                          native=True, aliases=("learn from a corpus", "train on text", "teach the model", "teach language",
                                                "language curriculum", "learn word meanings", "learn a language",
                                                "read a corpus", "curriculum", "learn vocabulary", "learn facts"))
    c.register_capability("Utilities & helpers", "the engine's cross-cutting UTILITY tools: content addressing & "
                          "hashing (uri), tamper-evident verification (verify), erasure/rateless coding for reliability "
                          "(fountain), chunked delta chains with integrity proofs (deltachain), versioned rollback "
                          "history (history), lossless compression (compress/codec), and the determinism contract "
                          "(determinism). The plumbing every faculty leans on",
                          example="from holographic_uri import address_from_content, make_key; from holographic_verify import CompositionTree",
                          native=True, aliases=("utility", "helper", "tool", "hash", "checksum", "content address",
                                                "content id", "verify integrity", "tamper", "erasure code", "reliability",
                                                "delta chain", "version history", "rollback", "compress", "determinism",
                                                "plumbing", "reliability code"))
    # --- describe a scene in words, build it, adjust named objects, render or simulate ---
    c.register_capability("Scene from description (semantic)", "DESCRIBE a 3-D scene in plain words and the engine "
                          "builds it, then you ADJUST it by talking to named objects: mind.build_scene('a big red metal "
                          "sphere and a small blue glass box on a sunny day') returns a live SemanticScene; then "
                          "scene.adjust('make the sphere bigger'), scene.adjust('change the box to metal'), "
                          "scene.set('the red sphere', material='glass'), scene.render(), scene.simulate(). When a command is unclear "
                          "it SUGGESTS rather than fails -- scene.interpret(cmd) previews what it understood + 'did you mean?' hints, scene.options() lists what you can say, scene.feedback holds the last report. Or wrap an "
                          "existing object list with mind.semantic_scene(objects). Controlled vocabulary, deterministic",
                          example="scene = mind.build_scene('a red metal sphere and a blue glass box'); scene.adjust('make the box bigger'); scene.render()",
                          native=True, aliases=("scene", "describe a scene", "build a scene", "make a scene", "create a scene",
                                                "scene from text", "3d scene", "adjust the scene", "semantic scene",
                                                "named objects", "make the sphere bigger", "change the material", "render a scene",
                                                "text to 3d", "text to scene", "scene editor", "reference objects by name"))
    # --- agent-friendly discovery: describe / suggest / route / autocomplete over the whole engine ---
    c.register_capability("Agent skills (discover & route)", "the AGENT-FRIENDLY layer: mind.skills() lists every "
                          "capability + method with how to CALL it (skill descriptions, real signatures); "
                          "mind.suggest(task) ranks capabilities for a plain-English task WITH a confidence + the call; "
                          "mind.route(task) is a decision node ('act' with the call when confident, else 'choose' the "
                          "options); mind.complete_method(prefix) autocompletes method names; mind.describe_skill(name) is a "
                          "skill card. Also over HTTP: GET /skills, POST /skills/suggest|route|complete|card",
                          example="mind.route('render a scene'); mind.suggest('edit an image'); mind.complete_method('learn_')",
                          native=True, aliases=("agent", "agentic", "skills", "skill description", "autocomplete",
                                                "suggest", "decision tree", "route", "what can you do", "how do i",
                                                "which tool", "find a tool", "capabilities", "manifest", "discover", "help"))
    # --- domain families surfaced by the catalog-gap sweep (tools existed, homes did not) ---
    c.register_capability("Rendering (path trace)", "render a scene to an image: path_trace (Monte-Carlo global "
                          "illumination), a camera controller, indirect-light gather + irradiance cache "
                          "(globalillum), precomputed radiance transfer (prt), volumetric integration, and lens/DOF + "
                          "post-FX. The analysis-by-synthesis render path", example="mind.path_trace(scene); mind.camera(); from holographic_raymarch import sphere_trace",
                          native=True, aliases=("render a scene", "path trace", "ray tracing", "global illumination",
                                                "camera", "depth of field", "lens", "volumetric render", "radiance transfer",
                                                "prt", "ambient occlusion", "post processing", "gbuffer", "raytrace", "render"))
    c.register_capability("Mesh editing (DCC)", "modeling/DCC edits on a Mesh: extrude/inset faces (meshpoly), "
                          "subdivide + smooth (meshsubdiv, Catmull-Clark), deform/warp (deform), rig-skin-pose a "
                          "skeleton (blendpose), UV unwrap (chart), decimate/QEM, booleans, and mesh<->SDF. "
                          "Blender-parity polygon editing", example="mind.deform(mesh, ...); mind.mesh_to_sdf(mesh); from holographic_meshverbs import extrude_face",
                          native=True, aliases=("edit a mesh", "extrude", "bevel", "inset", "subdivide", "smooth a mesh",
                                                "decimate", "reduce polygons", "uv unwrap", "unwrap uv", "rig", "skin",
                                                "pose a skeleton", "skeleton", "deform", "boolean", "remesh", "dcc", "modeling"))
    c.register_capability("SDF & procedural geometry", "implicit + procedural geometry: signed distance fields (sdf), "
                          "sphere-trace raymarching with ambient occlusion (raymarch), sculpting, procedural terrain "
                          "(procgen), spatial tiling + octree, and voxelization. Native-first shape building",
                          example="from holographic_raymarch import sphere_trace; mind.terrain(...); from holographic_sdf import ...",
                          native=True, aliases=("sdf", "signed distance field", "raymarch", "sphere trace", "sculpt",
                                                "procedural terrain", "procedural geometry", "voxelize", "voxel", "octree",
                                                "tile in space", "implicit surface", "marching"))
    c.register_capability("Navigation & planning", "find a way through a space or structure: A*/shortest-path route "
                          "planning (plan), slime-mould flow networks (flow), tree/graph navigation (navigator), and "
                          "maze solving. Pathfinding on the VSA substrate", example="from holographic_plan import ...; mind.solve_maze(world); from holographic_flow import ...",
                          native=True, aliases=("navigation", "plan a route", "pathfinding", "shortest path", "maze",
                                                "slime mould", "flow network", "route", "navigate", "wayfinding", "traverse"))
    c.register_capability("Learning & agents", "gradient-free learning on the substrate: an RL agent with a value head "
                          "+ drives (agent), a holographic classifier, an echo-state reservoir (reservoir), "
                          "mixture-of-experts (moe), KAN, forward-forward, recurrent/predictive nets, and dreaming. NPC "
                          "brains and on-line learners with NO autodiff", example="mind.agent(...); mind.classify(x); mind.reservoir(...)",
                          native=True, aliases=("reinforcement learning", "rl agent", "train a classifier", "classify",
                                                "policy", "npc brain", "game ai", "reservoir", "echo state", "mixture of experts",
                                                "moe", "kan", "forward forward", "gradient free", "learn a policy", "predictor", "agent"))
    c.register_capability("Data analysis", "analyse data with VSA-native methods: optimal transport / Wasserstein "
                          "(transport), graph Laplacian + spectral filtering (graphsignal), Nystrom embedding / "
                          "dimensionality reduction, persistent-homology topology, kernel density estimate, "
                          "point-cloud structure (cosmic), and time-series / market analysis", example="from holographic_transport import wasserstein; from holographic_graphsignal import laplacian_filter",
                          native=True, aliases=("data analysis", "cluster", "optimal transport", "wasserstein", "graph laplacian",
                                                "spectral", "dimensionality reduction", "embedding", "topology", "persistent homology",
                                                "kernel density", "point cloud", "time series", "statistics", "analytics"))
    c.register_capability("Symbolic reasoning", "recover structure symbolically: symbolic regression to find a formula "
                          "(symbolic), resonator networks that FACTOR a bound vector into its parts (sbc/resonator), "
                          "is_a taxonomy climbing, and relational reasoning over records. Turning data and vectors back "
                          "into laws", example="from holographic_symbolic import ...; mind.climb('dog'); from holographic_sbc import ...",
                          native=True, aliases=("symbolic regression", "find a formula", "factor a vector", "resonator",
                                                "factorization", "decompose a signal", "reason", "reasoning", "climb hierarchy",
                                                "relational", "law from data"))
    c.register_capability("Signal & spectral", "1-D signal processing: FFT / spectral analysis (spectral), "
                          "faint-signal detection in noise with a calibrated false-discovery rate (signal_structure), "
                          "drifting-narrowband / de-Doppler search (dedoppler), spectral flatness, and bandwidth. The "
                          "radio-SETI-style detection stack", example="from holographic_spectral import ...; from holographic_dedoppler import ...",
                          native=True, aliases=("signal processing", "fft", "spectral", "spectrum", "detect a signal",
                                                "faint signal", "narrowband", "doppler", "dedoppler", "drift", "flatness",
                                                "bandwidth", "frequency", "audio"))
    c.register_capability("Compression & codec", "shrink data losslessly or by rate-distortion: a sequence/entropy "
                          "codec (codec), general compression (compress), rate-distortion quantization "
                          "(ratedistortion), and content-addressed storage (storage). How the engine fits vectors into "
                          "bytes", example="from holographic_codec import ...; from holographic_ratedistortion import ...",
                          native=True, aliases=("compress", "compression", "codec", "entropy coding", "rate distortion",
                                                "quantize", "content addressed storage", "encode data", "shrink data", "deduplicate"))
    c.register_capability("Video (temporal)", "temporal image sequences: video compression with keyframe/delta coding "
                          "(video), temporal compression, motion/phase morph between frames (phasemorph), and frame "
                          "interpolation. Moving pictures on the substrate", example="from holographic_video import ...; mind.blend_images(a, b)",
                          native=True, aliases=("video", "compress a video", "temporal compression", "frames", "motion",
                                                "interpolate frames", "keyframe", "sequence of images", "movie"))
    c.register_capability("Honesty & measurement", "measure claims honestly: error bars + significance (measure), "
                          "ablation studies (ablate), proof-of-structure against a null (structure), calibrated "
                          "detection with false-discovery control, benchmark + variance harness, and stress tests. The "
                          "engine's own truth-in-advertising tools", example="from holographic_measure import ...; from holographic_ablate import ...",
                          native=True, aliases=("measure", "error bars", "significance", "ablation", "false discovery rate",
                                                "calibrated", "benchmark", "variance", "stress test", "proof of structure",
                                                "honesty", "null model", "confidence interval"))
    c.register_capability("Program & machine (VM)", "the VSA computer: a stored-program holographic machine "
                          "(machine/HoloMachine) that runs vector programs, recipes with holes / hygienic templates "
                          "(template), a content-addressed compile cache (compile), tool-orchestration planning "
                          "(orchestrator/voidsynth), and reversible computation. Programs as data", example="from holographic_machine import HoloMachine; from holographic_template import RecipeTemplate",
                          native=True, aliases=("virtual machine", "stored program", "run a program", "vm", "recipe",
                                                "template", "recipe with holes", "compile", "content addressed compile",
                                                "orchestrate", "plan tools", "reversible computation", "bytecode"))
    # --- vendored knowledge: a real dictionary + taxonomy for contextual awareness ---
    c.register_capability("Dictionary + taxonomy (vendored)", "a comprehensive vendored English DICTIONARY (~144k "
                          "words: definition, part of speech, synonyms, example) AND an is_a TAXONOMY (encyclopedia "
                          "side: 'a dog is a kind of domestic animal...'), giving the engine real world-knowledge for "
                          "contextual awareness beyond its internal machinery. Stdlib-only lazy load (gzip+json); the "
                          "mind can also LEARN meaning from it. Princeton WordNet, free with attribution",
                          example="mind.lookup('gravity'); mind.word_taxonomy('dog'); mind.learn_vocabulary(my_words)",
                          native=True, aliases=("dictionary", "define", "definition", "word meaning", "synonyms",
                                                "encyclopedia", "taxonomy", "is a", "wordnet", "vocabulary",
                                                "contextual awareness", "knowledge", "lexicon", "what does word mean"))
    # --- material LIBRARIES: render appearance + physical properties, and the bridge between them ---
    c.register_capability("Material library (render + physical)", "the engine's material LIBRARIES, discoverable in "
                          "one place: ~141 RENDER presets (metals/gems/woods/stones/liquids/biomes -- PBR appearance) "
                          "and ~120 PHYSICAL materials in 12 categories (metals/liquids/gases/polymers/ceramics/glass/"
                          "minerals/stone/wood/biological/building/semiconductors) with density, refractive index, "
                          "viscosity, Young's modulus, sound speed, specific heat, thermal conductivity/expansion, "
                          "melting/boiling point, phase -- validated, unit-documented, for solvers/scientists. "
                          "material_info(name) gives BOTH how "
                          "a material looks AND how it behaves; find_materials()/materials() search + list across both. "
                          "Users can add their own to either library",
                          example="mind.material_info('gold'); mind.find_materials('clear liquid'); mind.materials()",
                          native=True, aliases=("material library", "materials", "physical material", "material properties",
                                                "density", "refractive index", "render material", "pbr preset", "gold",
                                                "copper", "diamond", "material data", "material list", "scientist material"))
    # --- material + shading (consolidation R3) ---
    c.register_capability("Material (channels)", "the material as a record of named channels (albedo/metallic/"
                          "roughness/normal/...) you sample per point; its position-dependent channels BAKE via the "
                          "Cache home and shade via the Shading home", example="from holographic_material import Material",
                          native=True, aliases=("material", "channels", "albedo", "roughness", "metallic", "shader"))
    c.register_capability("Shading (BRDF)", "the shade model: cook_torrance (full specular+diffuse per light), "
                          "lambert (diffuse term), sample_brdf (importance-sampled bounce) -- call these instead of "
                          "re-deriving Fresnel/GGX/diffuse", example="from holographic_brdf import cook_torrance, lambert",
                          native=True, aliases=("shade", "brdf", "cook_torrance", "lambert", "fresnel", "ggx", "specular", "diffuse"))
    c.register_capability("Standalone API service", "run the engine as a standalone DATABASE server on any OS and "
                          "talk to it over HTTP/JSON: full SQL (CREATE/INSERT/SELECT/UPDATE/DELETE/JOIN/DROP), a "
                          "GraphQL front door for nested documents, disk PERSISTENCE (data survives a restart), "
                          "capability discovery, and an optional bearer-token gate. Stdlib-only (numpy aside); a "
                          "drop-in DB replacement for other apps. Launched by serve.sh (Linux/macOS) / serve.bat (Windows)",
                          example="./serve.sh --persist mydb.json   # then: curl -X POST .../sql -d '{\"sql\":\"SELECT ...\"}'",
                          native=True, aliases=("api", "server", "service", "standalone", "http", "rest", "daemon",
                                                "database", "sql", "graphql", "persistence", "drop-in database",
                                                "run as server", "endpoint", "launch", "serve"))
    c.register_capability("Job lifecycle control", "start / pause / resume / cancel long-running work (renders, "
                          "simulations, dataset processing) as CHECKPOINTABLE monoid jobs: completed buckets fold into "
                          "partials, so a job pauses at a bucket boundary, saves to disk, survives an app restart, and "
                          "resumes only the remaining buckets. Works across any coordinator backend (local pool / farm)",
                          example="from holographic_jobs import JobManager; m.create(id, buckets, worker); m.start(id, background=True); m.pause(id); m.resume(id)",
                          native=True, aliases=("job", "start", "pause", "resume", "cancel", "checkpoint", "render job",
                                                "long running", "background task", "resumable", "progress", "lifecycle"))
    c.register_capability("Distributed hardening (R5)", "fault tolerance + verification for untrusted farm nodes: "
                          "retry-with-backoff (a reissue reassigns a dead node\'s work), redundant computation + "
                          "majority VOTING (accept only what independent nodes agree on -- a node can\'t force a "
                          "result), canary buckets (known answers reject an untrusted node), and speculative straggler "
                          "backups. The BOINC/SETI@home discipline, mandatory before public contributors",
                          example="from holographic_hardening import HardenedCoordinator; HardenedCoordinator(farm, redundancy=3).run(buckets, worker, cache, reduce, canaries=[...])",
                          native=True, aliases=("voting", "redundant compute", "retry", "fault tolerance", "canary",
                                                "untrusted node", "quorum", "straggler", "backup execution", "verify result"))
    c.register_capability("Network render farm", "run the coordinator\'s monoid workers on OTHER machines: a worker "
                          "daemon per node (stdlib http/json), the read-only cache shipped ONCE by content hash and "
                          "reused, buckets dispatched concurrently and reduced -- the same Coordinator.run as the local "
                          "pool. Buckets are data, workers are registered code; a node runs only its registered workers",
                          example="from holographic_farm import WorkerDaemon, NetworkFarm; Coordinator(NetworkFarm([addr])).run(buckets, 'worker_name', cache, reduce)",
                          native=True, aliases=("render farm", "distributed", "network", "seti", "worker daemon",
                                                "remote", "cluster", "node", "another machine", "farm"))
    c.register_capability("Command runner (external tools)", "run any registered ALLOWLISTED program/script as a "
                          "task (subprocess, no shell, time-boxed) and wire it as an orchestrator Tool the Planner "
                          "can chain, with a CircuitBreaker on a flaky one -- the door to external tools and services. "
                          "SECURITY: allowlist only, never a command from untrusted input, values fill placeholders",
                          example="from holographic_command import CommandRunner, command_as_tool; r.register('ffmpeg', [...]); r.run('ffmpeg', args)",
                          native=True, aliases=("run command", "external tool", "subprocess", "shell", "run program",
                                                "ffmpeg", "job runner", "allowlist", "command backend"))
    c.register_capability("Distributed coordinator", "run monoid work (partition -> worker -> shared read-only cache "
                          "-> reduce) on a pluggable BACKEND: an in-process default or a persistent local process pool "
                          "(ProcessPoolExecutor + shared_memory, cache shipped ONCE, workers in separate interpreters). "
                          "Sits behind distribute; includes a margin-gated canonical tie-break so distributed results "
                          "agree on knife-edge decisions",
                          example="from holographic_coordinator import Coordinator, LocalPool; Coordinator(LocalPool(4)).run(buckets, worker, cache, reduce)",
                          native=True, aliases=("coordinator", "distribute compute", "process pool", "parallel",
                                                "render farm", "offload", "shared memory", "backend", "tie-break",
                                                "local pool", "worker pool", "monoid reduce"))
    c.register_capability("Graph traversal (exact)", "reachability over a table\'s edges -- neighbors, descendants, "
                          "reachable, shortest path -- what recursive SQL CTEs make painful. Uses an EXACT adjacency "
                          "index by design: the holographic graph store\'s recall collapses at scale, so traversal is "
                          "a plain deterministic graph (tombstone-aware, directed or undirected)",
                          example="from holographic_querygraph import EdgeGraph; EdgeGraph(t,'src','dst').path(a,b)",
                          native=True, aliases=("graph", "reachable", "descendants", "shortest path", "traversal",
                                                "adjacency", "recursive cte", "edges", "network"))
    c.register_capability("Single-writer concurrency", "B8 concurrency: one writer at a time (serialised by an "
                          "exclusive lock; a second writer waits or fails fast) plus lock-free reader SNAPSHOTS (a "
                          "consistent point-in-time view immune to later writes). MVCC deferred, stated honestly",
                          example="from holographic_querylock import SingleWriterLock; with lock.write(): ...",
                          native=True, aliases=("lock", "single writer", "concurrency", "snapshot read", "writer lock",
                                                "isolation", "consistent read"))
    c.register_capability("Workspace folders", "a shallow grouping tree over a database\'s tables (database > folder "
                          "> table): each table has one HOME folder (ownership -> lifecycle/tier) plus any number of "
                          "ASSOCIATION links (grouping, no deletion on unlink). Scoped search runs over just a "
                          "subtree. Folders reference existing tables, they do not copy them",
                          example="from holographic_queryfolder import FolderTree; ft.set_home('user.sales','reports'); ft.tables_in('reports')",
                          native=True, aliases=("folder", "group tables", "namespace tree", "organize tables",
                                                "home folder", "association folder", "scoped search", "drill down"))
    c.register_capability("VSA programs as DB objects", "installable, runnable 'stored procedures' that are "
                          "hypervectors the machine executes (LOAD/BIND/APPLY/HALT -- not arbitrary code): install, "
                          "list a queryable catalog, find a program BY MEANING (fuzzy over its doc), EXPLAIN (dry "
                          "run), and EXECUTE over query rows sandboxed to whitelisted handlers + step-bounded, result "
                          "carrying a calibrated confidence. Safer than a SQL stored procedure",
                          example="from holographic_queryprog import ProgramCatalog; cat.install(...); cat.find('cluster a series')",
                          native=True, aliases=("stored procedure", "install program", "execute program", "udf",
                                                "pg_proc", "find program", "run program", "vsa program", "program catalog"))
    c.register_capability("Query time-travel & audit", "git-for-data on a query table: SELECT as-of a past version "
                          "(time travel), blame a row across versions, diff two versions (added/removed/changed with "
                          "field detail), revert, branch/compare/discard, and prove/locate-tampering (Merkle root + "
                          "O(log n) which-row-changed). Wires the shipped versioning faculties into the query layer",
                          example="from holographic_querytime import TableHistory, select_as_of, diff_versions, prove",
                          native=True, aliases=("time travel", "as of", "temporal", "blame", "diff versions", "revert",
                                                "branch", "git for data", "tamper", "audit", "version history", "undo"))
    c.register_capability("Workspaces (durable DB + transient sessions)", "WS3-WS6: run one persistent user database "
                          "alongside many TRANSIENT per-session workspaces (loose scratch tables + the 3D/sim/render "
                          "context) that stay isolated -- clearing or resetting one never touches the persistent DB or "
                          "a sibling. Make / switch / clear / reset-keeping-data, export/import a workspace, and combine "
                          "two with an EXPLICIT collision policy (a merge is a decision, not a guess)",
                          example="from holographic_workspace import WorkspaceManager; m=WorkspaceManager(); m.new_workspace('sessionA'); m.switch_workspace('sessionA')",
                          native=True, aliases=("workspace", "session", "scratch tables", "transient tables", "isolate "
                                                "session", "reset keep data", "export workspace", "combine workspaces",
                                                "per-session", "sandbox tables"))
    c.register_capability("Durability & crash recovery", "B7: make the query store survive a crash. Take a durable "
                          "SNAPSHOT of the persistent tiers (replay-based, so it rebuilds byte-identically), keep a "
                          "write-ahead JOURNAL of inserts/updates/deletes since the snapshot, and RECOVER to the last "
                          "consistent point by loading the snapshot and replaying the journal. The snapshot+WAL "
                          "discipline, on top of the plain save/load the service already exposes",
                          example="from holographic_query_durable import save_snapshot, Journal, recover; recover(snap_path, journal_path)",
                          native=True, aliases=("durability", "crash recovery", "journal", "write ahead log", "wal",
                                                "snapshot recover", "point in time recovery", "replay journal", "recover"))
    c.register_capability("Splat aniso-refine (re-enable)", "full-3DGS anisotropic refinement composed coarse-first: "
                          "fit cheap isotropic splats, then gradient-refine the RESIDUAL (what iso missed -- sharp / "
                          "oriented features) with anisotropic Gaussians. Strictly >= the isotropic baseline (no harm "
                          "mode); big win on sharp edges. Opt-in (no reliable cheap detector for WHEN it pays)",
                          example="from holographic_splat import fit_coarse_first; fit_coarse_first(target, K_iso, K_aniso)",
                          native=True, aliases=("splat refine", "anisotropic splat", "3dgs", "gaussian splat", "coarse "
                                                "first splat", "aniso fit", "residual refine", "gradient refine"))
    c.register_capability("Nystrom kernel (re-enable)", "apply a kernel-weighted field in O(N*m) instead of exact "
                          "O(N^2), gated by a low-rank probe: if a cheap held-out probe says the kernel is low-rank "
                          "(smooth) use Nystrom (measured 6-14x faster, near-exact), else fall back to exact. The "
                          "exact fallback is always correct, so the gate can't be wrong",
                          example="from holographic_nystrom import apply_kernel_gated; apply_kernel_gated(points, sources, weights, sigma)",
                          native=True, aliases=("nystrom", "landmark", "low rank", "kernel", "rbf field", "large field",
                                                "spectral embedding", "o(n^2)", "smooth field"))
    c.register_capability("Coarse-first refine (re-enable)", "run the cheap method everywhere, measure a per-cell "
                          "residual/uncertainty, and escalate to the expensive method ONLY where it's high -- the "
                          "shared detector for the Group-B re-enables (adaptive AA, Nystrom, splat refine, volint). "
                          "concentration() is the honest breakeven check (low => uniform is just as good)",
                          example="from holographic_coarsefirst import refine_where_uncertain, concentration",
                          native=True, aliases=("coarse first", "coarse-to-fine", "adaptive refine", "refine where "
                                                "uncertain", "residual", "escalate", "adaptive sampling", "uncertainty"))
    c.register_capability("Multi-scatter BRDF (re-enable)", "energy-conserving GGX for rough metals: the Kulla-Conty "
                          "multi-scatter term adds back the energy single-scatter GGX loses (white-furnace ~0.4 -> "
                          "~1.0 at high roughness), GATED by roughness so smooth surfaces skip it (the term overshoots "
                          "at low roughness). Detector is the exact material roughness",
                          example="from holographic_brdf import brdf_gated, cook_torrance_ms; brdf_gated(N,V,L,color,metallic,roughness)",
                          native=True, aliases=("multi-scatter", "multiscatter", "kulla-conty", "energy conservation",
                                                "brdf", "ggx", "rough metal", "white furnace", "roughness"))
    c.register_capability("Adaptive record (load-gated)", "a role->filler memory that picks its representation by "
                          "LOAD and FIDELITY need -- cheap real-HRR at low load, FHRR phasors past the capacity knee, "
                          "or tensor-product binding for EXACT recall (perfect to M~dim, at dim*dim storage). Uniform "
                          "add/recall; deciders are exact integers/flags, no harm mode on recall",
                          example="from holographic_loadmemory import AdaptiveRoleFillerMemory; m=AdaptiveRoleFillerMemory(dim, pairs, exact=True)",
                          native=True, aliases=("adaptive record", "role filler memory", "fhrr", "phasor", "tensor",
                                                "exact recall", "load", "capacity", "high load recall", "bind pairs"))
    c.register_capability("Regime gate (re-enable)", "run a superior-but-niche method ONLY in its regime, behind a "
                          "cheap conservative detector, with a safe fallback everywhere else -- the pattern for "
                          "re-enabling a shelved 'kept negative' now that adaptive dispatch can spot its regime "
                          "(e.g. closed-form iterate for linear/bind operators)",
                          example="from holographic_regimegate import RegimeGate; RegimeGate(name, detect, threshold, superior, fallback)",
                          native=True, aliases=("regime gate", "re-enable", "adaptive dispatch", "gate", "detector",
                                                "niche method", "fallback", "closed form iterate"))
    c.register_capability("Hypervector (datatype)", "the first-class hypervector: a raw vector + its dim / encoder / "
                          "tag, with the five verbs (bind/unbind/bundle/cleanup/permute) as methods. Encoders are the "
                          "constructors; the raw array stays one attribute away (.array / np.asarray(hv))",
                          example="from holographic_hypervector import Hypervector; Hypervector.encode(encoder, value).bind(other)",
                          native=True, aliases=("hypervector", "datatype", "vector", "vsa", "hdvector", "symbol",
                                                "bind", "bundle", "permute", "cleanup", "encode"))
    c.register_capability("Sampling", "Monte-Carlo sampling: low-discrepancy / blue-noise patterns, cosine-hemisphere "
                          "directions, MIS weighting, firefly-clamped accumulation -- one home over the shipped samplers",
                          example="from holographic_samplinghome import Sampling; Sampling.cosine_hemisphere(N, n, seed)",
                          native=True, aliases=("sample", "sampling", "blue_noise", "poisson", "quasi", "halton",
                                                "hemisphere", "mis", "jitter", "firefly", "accumulate"))

    # --- fields (audit named ~8) ---
    c.register_capability(
        "Field", "sample a scalar/vector field at points with ONE interface (field.sample(points)); the backend is "
        "chosen by cost: callable/oracle, dense grid, narrow-band sparse (spectral/FPE/region/dirty are backends too)",
        example="from holographic_fieldhome import Field; Field.grid(arr, lo, hi).sample(pts)", native=True,
        aliases=("field", "grid", "volume", "density", "sdf", "sample", "voxel"))
    c.register_capability("holographic_sparsefield", "narrow-band sparse field -- cost scales with surface area, "
                          "not volume", example="from holographic_sparsefield import ...", native=True,
                          aliases=("narrow", "band", "sparse", "field"))
    c.register_capability("holographic_fpefield", "fractional-power-encoded N-D field (surface as one hypervector)",
                          example="from holographic_fpefield import ...", native=True, aliases=("fpe", "field", "continuous"))

    # --- scale / compute / the kernel verbs ---
    c.register_capability("holographic_distribute", "scale out a commutative-monoid computation: partition into "
                          "buckets, run independently, reduce (sum/min/max/bundle)", example="from holographic_distribute import partition, reduce_sum, reduce_min, reduce_bundle",
                          native=True, aliases=("scale", "parallel", "partition", "mapreduce", "distribute", "raid"))
    c.register_capability("holographic_fuse", "fuse a bind chain into ~2 FFTs with no Python between ops (stay "
                          "VSA-native)", example="from holographic_fuse import fuse", native=True,
                          aliases=("fuse", "native", "fft", "chain", "compute"))
    c.register_capability("kernel verbs", "the five primitives: bind (attach/transform), unbind (query), bundle "
                          "(superpose/blend), permute (order), cleanup (recognise/denoise)",
                          example="from holographic_ai import bind, bundle; from holographic_ai import Vocabulary  # Vocabulary(...).cleanup(x)", native=True,
                          aliases=("bind", "unbind", "bundle", "cleanup", "permute", "superpose", "blend"))

    # --- the catalog itself ---
    c.register_capability("holographic_catalog", "THIS catalog: search the engine's own capabilities before building "
                          "a duplicate (register_capability / find_capability)", example="find_capability('search vectors')",
                          native=False, aliases=("catalog", "capability", "registry", "find", "discover", "duplicate"))

    # --- the pipeline (consolidation R1): the one entry point that composes a render/sim run ---
    c.register_capability(
        "Pipeline (render/sim)", "compose a render or sim run as ordered stages that declare what they need/produce; "
        "dispatch among render strategies (pathtrace/raymarch/prt/radiance) and catch a missing input before running",
        example="from holographic_pipeline import build_pipeline, PipelineConfig, RenderSpec", native=False,
        aliases=("pipeline", "stage", "compose", "run", "render", "strategy", "dispatch", "route"))

    # --- top-level DOMAIN pipelines: one findable pointer per subsystem, so no whole domain is buried ---
    c.register_capability("Lighting (domain)", "one home for lighting: the light types (point/directional/spot/area/"
                          "dome/IES) and the shade INTEGRAL in each mode -- direct NEE, PRT relight, environment SH; "
                          "render methods call it", example="from holographic_lightinghome import Lighting, RectLight",
                          native=True, aliases=("lighting", "light", "lamp", "shadow", "dome", "area", "ies", "spot",
                                                "nee", "direct", "prt", "irradiance"))
    c.register_capability("Shadow / visibility (domain)", "test whether light or the environment reaches a point: "
                          "SDF soft shadow (Quilez penumbra), ambient occlusion, hard shadow-ray (NEE), and PRT baked "
                          "visibility -- one home of strategies render paths call",
                          example="from holographic_shadowhome import Shadow; Shadow.soft(sdf, P, Ldir)",
                          native=True, aliases=("shadow", "visibility", "occlusion", "ambient occlusion", "penumbra",
                                                "shadow ray", "soft shadow", "unoccluded"))
    c.register_capability("Geometry (domain)", "build and edit shapes three ways: explicit MESH (half-edge + verbs), "
                          "implicit SDF (CSG + raymarch), and SPLATS (Gaussian clouds) -- convertible via meshbridge",
                          example="from holographic_mesh import Mesh; from holographic_sdf import box, sphere",
                          native=True, aliases=("geometry", "mesh", "sdf", "splat", "shape", "model", "csg", "subdivide"))
    c.register_capability("Texture (domain)", "procedural + example-based surface detail as FIELDS you plug into a "
                          "Material channel: fbm noise, Voronoi/cellular cracks, divergence-free curl, patch synthesis; "
                          "plus the weathering set (burn/oxidation/inclusions)",
                          example="from holographic_texturehome import Texture; Param(field=Texture.voronoi(kind='edge'))",
                          native=True, aliases=("texture", "noise", "fbm", "voronoi", "curl", "procedural", "weathering",
                                                "pattern", "detail", "cellular"))
    c.register_capability("Simulation (domain)", "a shared STEP LOOP over any solver (fluids/smoke, fire/combustion, "
                          "softbody/cloth, hair, MPM, collision, reaction-diffusion) -- each keeps its own math; the "
                          "scaffold gives them one step(dt) and exposes their field for the Pipeline to render",
                          example="from holographic_simulationhome import Simulation; Simulation.for_fluid(fluid).run(10)",
                          native=True, aliases=("simulation", "solver", "fluid", "smoke", "fire", "cloth", "softbody",
                                                "step", "advance", "sim loop", "mpm", "reaction diffusion",
                                                "particle system", "particles", "emitter", "mass spring", "spring",
                                                "rigid body", "collision"))
    c.register_capability("Encoders (number to vector)", "turn raw values into hypervectors: scalar & fractional-power "
                          "encoding (encoders/fpe -- nearby numbers map to nearby vectors), N-D coordinate fields "
                          "(fpefield), complex-phasor FHRR (fhrr), sparse block codes (sbc), geometric-algebra Clifford "
                          "(clifford), and exact integer arithmetic over phasors (rns). How data ENTERS the substrate",
                          example="from holographic_encoders import ScalarEncoder; from holographic_fpe import ...",
                          native=True, aliases=("encode", "encoder", "number to vector", "scalar encoding",
                                                "fractional power encoding", "fpe", "encode coordinates", "phasor", "fhrr",
                                                "sparse block codes", "sbc", "clifford", "geometric algebra",
                                                "exact integer arithmetic", "rns", "embed a value"))
    c.register_capability("Physics & chemistry (domain)", "physical/chemical PROPERTIES and their evolution: the matter "
                          "model (Mixture/matter_step: smoke->oil separation), diffusion, equilibrium propagation, "
                          "thin-film iridescence, oxidation/weathering", example="from holographic_mixture import Mixture, matter_step",
                          native=True, aliases=("physics", "chemistry", "matter", "mixture", "diffusion", "material properties",
                                                "iridescence", "oxidation", "phase"))
    c.register_capability("Adaptive rendering", "the render call that picks its own methods/quality: the converging "
                          "sampler that stops per-pixel when the confidence interval is tight, and the render-method "
                          "auto-picker", example="from holographic_gbuffer import render_auto, converge_samples",
                          native=True, aliases=("adaptive", "auto", "quality", "converge", "raytracing mode", "render mode"))
    c.register_capability("Denoise (domain)", "clean a render or signal with one home: image SVGF (variance-guided "
                          "a-trous) or demodulated (divide albedo out), sharpen, and the signal manifold denoisers "
                          "(adaptive/manifold/codebook/trajectory)",
                          example="from holographic_denoisehome import Denoise; Denoise.image(img, N, A, D, method='svgf')",
                          native=True, aliases=("denoise", "svgf", "clean", "smooth", "nlm", "demodulate", "sharpen",
                                                "noise reduction", "restore"))
    c.register_capability("Compute (VSA-native)", "stay in the vector/frequency domain with no Python hops: FUSE a "
                          "bind/bundle/permute chain into ~2 FFTs (measure the FFT drop), the fuse-runs SCHEDULER, "
                          "width, and running logic as a VSA PROGRAM. Rule: push decisions/cleanups to the boundaries",
                          example="from holographic_computehome import Compute; Compute.fuse_record(keys, values)",
                          native=True, aliases=("compute", "fuse", "fused", "schedule", "execute", "program", "machine",
                                                "fft", "chain", "collapse", "vsa native"))
    c.register_capability("Memory (cache hierarchy)", "keep the hot working set where the CPU can reach it fast: FFT "
                          "spectrum residency (skip recomputing a reused transform), batched contiguous bind (one FFT "
                          "for a whole record), tiling to fit a cache level, and the opt-in GPU / numba backends",
                          example="from holographic_memoryhome import Memory; Memory.bind_cached(a, b, cache)",
                          native=True, aliases=("memory", "cache", "residency", "resident", "spectrum cache", "batch",
                                                "bind_batch", "backend", "gpu", "jit", "working set", "hot"))
    c.register_capability("Transform (warp)", "move / rotate / warp across representations: VSA bind (rigid) + "
                          "permute (order), 4x4 matrices (translate/scale/rotate/compose/decompose/look_at + "
                          "quaternions), clifford rotors, anisotropic steering -- one facade",
                          example="from holographic_transformhome import Transform; Transform.translation(t)",
                          native=True, aliases=("transform", "warp", "rotate", "translate", "scale", "rigid", "affine",
                                                "matrix", "quaternion", "rotor", "bind", "permute", "gizmo"))
    c.register_capability("Blend (combine)", "combine things into one: bundle (superposition, weighted = soft "
                          "mixture), lerp / slerp interpolation, Frechet mean on the sphere, front-to-back alpha "
                          "composite, and dict/scene merge with a conflict policy",
                          example="from holographic_blendhome import Blend; Blend.bundle(vectors, weights)",
                          native=True, aliases=("blend", "combine", "merge", "interpolate", "lerp", "slerp", "mix",
                                                "composite", "superpose", "average", "crossfade", "morph"))
    c.register_capability("Scale (distribute)", "make something bigger than one box / one pass can hold: partition a "
                          "job, run the pieces independently, reassemble with a commutative monoid -- map_reduce, "
                          "load-balanced partition, image tiles / volume bricks; strategies tiling/octree/multires/"
                          "superposed/sparsefield", example="from holographic_scalehome import Scale; Scale.map_reduce(buckets, worker, reduce='sum')",
                          native=True, aliases=("scale", "distribute", "partition", "map reduce", "tile", "brick",
                                                "parallel", "shard", "chunk", "monoid", "scale out"))
    c.register_capability("Query / database (domain)", "treat VSA stores as a database: SQL over tables, similarity/"
                          "time-travel/diff, durable + concurrent + graph + history query layers",
                          example="from holographic_query import run_sql, UserTable", native=False,
                          aliases=("query", "sql", "database", "table", "history", "diff", "time travel"))
    return c


# module-name keyword -> the domain words to tag it with, so a problem description can reach a whole family. Broad on
# purpose (the module's own docstring is the real match; these just add domain synonyms).
_FAMILY_KEYWORDS = [
    (("light", "lamp", "dome", "ies", "spot", "spharm", "prt", "radiance", "illum", "occlusion", "shadow"),
     ("lighting", "light")),
    (("mat", "brdf", "shade", "thinfilm", "iridesc", "clearcoat"), ("material", "shading")),
    (("tex", "noise", "weather", "burn", "oxid", "cellular", "inclusion", "curl", "tiling", "displace"),
     ("texture", "procedural")),
    (("field", "ndfield", "voxel", "regionfield", "dirtyfield", "sparsefield", "spectralfield", "fpefield"),
     ("field",)),
    (("mesh", "splat", "gltf", "svg", "subdiv", "sculpt", "poly", "geodes", "uv", "sdf", "octree", "terrain",
      "meshverbs", "deform", "skin", "blendpose", "curvature"), ("geometry", "mesh")),
    (("cache", "bake", "lut", "compile", "residency", "anim", "multires", "mipmap"), ("cache", "bake")),
    (("fluid", "smoke", "fire", "combust", "softbody", "cosserat", "groom", "mpm", "collide", "automaton",
      "cloth", "eulerops", "fields"), ("simulation", "physics")),
    (("matter", "mixture", "diffus", "equilibrium", "chem", "emitter", "scatter", "phasemorph", "physics"),
     ("physics", "chemistry", "matter")),
    (("tree", "pivot", "rayindex", "archive", "forest", "organizer", "navigator", "spatial", "pack", "vault"),
     ("search", "index", "recall")),
    (("sampl", "discrepancy", "mis", "accumulate", "traverse", "lowdiscrepancy"), ("sampling",)),
    (("denoise", "svgf", "sharpen", "downscale", "diffuse", "modulate", "diffusion"), ("denoise",)),
    (("render", "path", "ray", "raster", "postfx", "lens", "volint", "camera", "gbuffer", "brdf", "globalillum",
      "pathtrace", "radiance"), ("render", "rendering")),
    (("query", "sql", "table", "knowledge", "relation", "encyclopedia", "workspace"), ("query", "database")),
    (("distribute", "nystrom", "spectral", "graph", "transport", "topology", "cosmic", "market", "kde", "chart"),
     ("scale", "analysis")),
    (("text", "lexicon", "encyclopedia", "intent", "answer", "respond", "deliberate", "grammar", "segment",
      "meaning", "lang", "vision", "reasoning"), ("language", "text")),
    (("creature", "agent", "value", "drive", "classifier", "kan", "moe", "forward", "recurrent", "reservoir",
      "predictive", "dream", "partition", "orchestrator", "voidsynth"), ("learning", "agent")),
    (("honesty", "measure", "ablate", "structure", "protocol", "flatness", "benchmark", "stress"),
     ("honesty", "measurement")),
]


def _family_of(name):
    """The domain words a module belongs to (from its name), for catalog search. Empty -> ('engine',)."""
    n = name.replace("holographic_", "")
    fams = set()
    for keys, fam in _FAMILY_KEYWORDS:
        if any(k in n for k in keys):
            fams.update(fam)
    return tuple(sorted(fams)) or ("engine",)


def seed_from_modules(catalog, module_dir=None):
    """Register EVERY engine module as a findable capability, so nothing built stays buried. AST-reads each
    holographic_*.py module's DOCSTRING without importing it (the docgen discipline -- safe, side-effect-free) and
    adds it with its one-line summary as `does`, tagged with its domain family. Curated homes and mind faculties
    already registered are NOT overwritten (they carry better descriptions). This is what makes the catalog COMPLETE:
    a plain-English problem can surface any module, home, or faculty in the engine."""
    import ast
    import os
    import glob
    if module_dir is None:
        module_dir = os.path.dirname(os.path.abspath(__file__))
    for path in sorted(glob.glob(os.path.join(module_dir, "holographic_*.py"))):
        name = os.path.basename(path)[:-3]
        if name in catalog._by_name:
            continue
        try:
            tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
        except (SyntaxError, OSError):
            continue
        doc = ast.get_docstring(tree) or ""
        summary = doc.strip().replace("\n", " ").split("  ")[0][:200] if doc else name
        catalog.register_capability(name, summary or name, example="import %s" % name, native=True,
                                    aliases=_family_of(name))
    return catalog


def _selftest():
    c = default_catalog()
    # the headline: describe a problem, get the right home
    hits = c.find_capability("search a big pile of vectors")
    assert hits and "Index" in hits[0].name, [h.name for h in hits]
    assert any("Cache" in h.name or "bake" in h.does for h in c.find_capability("precompute a slow thing and look it up"))
    assert any("Field" in h.name for h in c.find_capability("represent a density volume over space"))
    assert any("light" in (h.name + h.does).lower() for h in c.find_capability("my placed light has speckle noise"))
    # register + find a fresh capability
    c.register_capability("MyThing", "does a special new job with widgets", aliases=("widget",))
    assert c.find_capability("widget job")[0].name == "MyThing"
    # native flag is carried
    assert c.get("holographic_catalog").native is False and c.get("Index (search)").native is True
    print("OK: holographic_catalog self-test passed (%d capabilities; 'search a big pile of vectors' -> %s)"
          % (len(c), c.find_capability("search a big pile of vectors")[0].name))


if __name__ == "__main__":
    _selftest()

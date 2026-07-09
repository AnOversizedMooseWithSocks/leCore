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
        example="from holographic.caching_and_storage.holographic_index import Index; Index(vectors, labels=names).nearest(query, k=5)",
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
                          example="from holographic.misc.holographic_pivot import ...", native=True, aliases=("pivot", "index"))
    c.register_capability("holographic_rayindex", "spatial index built for RAY queries (ray <-> object)",
                          example="from holographic.rendering.holographic_rayindex import ...", native=True, aliases=("ray", "spatial", "bvh"))
    c.register_capability("holographic_archive", "content-addressable image memory (WHT plates), damage-tolerant",
                          example="from holographic.misc.holographic_archive import ...", native=True, aliases=("image", "store", "recall"))

    # --- caching / baking: the CACHES (audit named ~9) = bake_and_query ---
    c.register_capability(
        "Cache (bake-and-query)", "bake a slow evaluator over what VARIES (position/view/time/constant) then look it "
        "up cheaply -- one shared grid-sample core over the scattered bakes (matbake, sdfbake, viewlut, anim)",
        example="from holographic.caching_and_storage.holographic_cachehome import Cache; Cache.bake(fn, vary='position', lo=lo, hi=hi, res=24)",
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
                          example="from holographic.misc.holographic_modulate import demodulate, remodulate", native=True,
                          aliases=("albedo", "irradiance", "denoise", "upscale", "demodulate"))
    c.register_capability("holographic_matbake", "bake POSITION-dependent material channels to a grid, trilinear "
                          "lookup", example="from holographic.materials_and_texture.holographic_matbake import ...", native=True, aliases=("material", "bake"))
    c.register_capability("holographic_prt", "precomputed radiance transfer: bake light transport, relight by a dot "
                          "product", example="from holographic.misc.holographic_prt import precompute_transfer, shade_prt", native=True,
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
                          example="from holographic.io_and_interop.holographic_uri import address_from_content, make_key; from holographic.misc.holographic_verify import CompositionTree",
                          native=True, aliases=("utility", "helper", "tool", "hash", "checksum", "content address",
                                                "content id", "verify integrity", "tamper", "erasure code", "reliability",
                                                "delta chain", "version history", "rollback", "compress", "determinism",
                                                "plumbing", "reliability code"))
    # --- describe a scene in words, build it, adjust named objects, render or simulate ---
    c.register_capability("Scene from description (semantic)", "DESCRIBE a 3-D scene in plain words and the engine "
                          "builds it, then you ADJUST it by talking to named objects: mind.build_scene('a big red metal "
                          "sphere and a small blue glass box on a sunny day') returns a live SemanticScene; then "
                          "scene.adjust('make the sphere bigger'), scene.adjust('change the box to metal'), "
                          "scene.set('the red sphere', material='glass'), scene.render(), scene.simulate(). NAME objects "
                          "to reference them easily -- scene.name('the red sphere', 'hero') or scene.adjust('call the box "
                          "crate'), then scene.adjust('make hero glass'); scene.rename('hero','champion'). PAINT a "
                          "procedural TEXTURE by talking to it -- scene.adjust('give hero a rusty texture'), scene.paint("
                          "'crate', 'marbled') (rusty/marbled/mossy/cloudy/lava/striped/noisy) -- and scene.render() "
                          "paints it on. Set the MOOD with a time-of-day/lighting word in the description -- 'a white "
                          "sphere at sunset', '...at noon', '...on an overcast day', 'a dramatic ...' -- which sets the "
                          "sun direction, colour and ambient (noon/morning/afternoon/sunset/sunrise/golden/dusk/overcast/"
                          "night/moonlit/studio/dramatic). Attach an EXTERNAL image file as a texture -- scene.attach_texture_file('the "
                          "sphere', 'project/textures/wave.png') -- and the scene tracks it in an AssetLibrary: if the "
                          "files move, scene.set_asset_roots([...]) + scene.resolve_assets() (or scene.relink(one, new)) "
                          "re-find them and render() reloads them, falling back to the object's colour if one is missing. "
                          "When a command is unclear it SUGGESTS rather than fails -- scene.interpret(cmd) "
                          "previews what it understood + 'did you mean?' hints, scene.options() lists what you can say, "
                          "scene.feedback holds the last report. Or wrap an existing object list with "
                          "mind.semantic_scene(objects). Controlled vocabulary, deterministic",
                          example="scene = mind.build_scene('a red metal sphere and a blue box'); scene.name('the sphere','hero'); scene.adjust('give hero a rusty texture'); scene.render()",
                          native=True, aliases=("scene", "describe a scene", "build a scene", "make a scene", "create a scene",
                                                "scene from text", "3d scene", "adjust the scene", "semantic scene",
                                                "named objects", "make the sphere bigger", "change the material", "render a scene",
                                                "text to 3d", "text to scene", "scene editor", "reference objects by name",
                                                "name an object", "rename object", "give it a texture", "rusty texture", "paint the scene",
                                                "cylinder", "cone", "torus", "donut", "pyramid", "tube", "pillar", "ring shape",
                                                "teal", "navy", "silver", "brown", "lavender", "crimson", "colours", "shapes",
                                                "at sunset", "at noon", "golden hour", "time of day", "lighting", "overcast",
                                                "dramatic lighting", "moody lighting", "studio lighting", "night scene", "sunrise"))
    c.register_capability("Instancing (shared definition + type-safe binding)", "place ONE shared definition many "
                          "times so editing it once updates every copy (edit-once): mind.shared_definition('chair', "
                          "mesh, 'metal') then scene.place(defn, transform) in mind.instanced_scene(); repaint the "
                          "definition and all instances change. The material<->geometry binding is TYPE-CHECKED at "
                          "compose time -- a surface material only binds to a mesh, a volumetric one (fog/smoke/fire) "
                          "only to a volume -- so a bad binding is refused, not rendered wrong. flatten_surface() "
                          "materialises the surface instances into one mesh. CMP4",
                          example="chair = mind.shared_definition('chair', box_mesh, 'metal'); s = mind.instanced_scene(); s.place(chair); chair.set_material('glass')",
                          native=True, aliases=("instance", "instancing", "shared definition", "edit once", "duplicate",
                                                "reuse geometry", "material binding", "surface volume", "place copies",
                                                "instanced scene", "clone", "prototype"))
    c.register_capability("Messaging across machines (distributed bus)", "the same publish/subscribe/send bus, spread "
                          "across nodes: mind.distributed_bus(peers, token, node_id) publishes locally AND fans out to "
                          "peer nodes (each running holographic_distbus.serve_bus), so agents on different machines "
                          "share topics -- a swarm coordinates across the farm the way it does in one process. Received "
                          "messages deliver local-only (no loops), dedup by a global id, and a dead peer never blocks "
                          "the publisher. Bound a mailbox (open_mailbox(maxlen=)) for backpressure at high fan-out.",
                          example="bus = mind.distributed_bus(['hostB:9100'], token, node_id='A'); from holographic.scene_and_pipeline.holographic_distbus import serve_bus  # serve_bus(bus, port=9100, token) in a thread",
                          native=True, aliases=("distributed bus", "messaging across machines", "cross-node messaging",
                                                "swarm messaging", "pub sub across nodes", "fan out", "gossip",
                                                "backpressure", "bounded mailbox", "flow control", "topic across nodes"))
    c.register_capability("Distributed compute across machines (farm)", "run the same partition-and-reduce work across "
                          "a FARM of machines. Each node runs holographic.scene_and_pipeline.holographic_coordinator.serve_worker(workers={name: fn}); "
                          "mind.farm(['host1:9000','host2:9000'], token).run(buckets, worker_name, cache, reduce) "
                          "round-robins the buckets across nodes and reassembles by the monoid reducer -- the same call "
                          "as the local pool, just cross-machine. SAFE by design: workers run BY NAME (a node only runs "
                          "workers it registered), so no code crosses the wire, only data. stdlib sockets/JSON.",
                          example="from holographic.scene_and_pipeline.holographic_coordinator import serve_worker; serve_worker(port=9000, workers={'sum': fn})  # then: mind.farm(['host:9000'], token).run(buckets, 'sum', None, reduce_sum)",
                          native=True, aliases=("farm", "distributed compute", "cluster", "network farm", "worker node",
                                                "serve_worker", "render farm", "compute across machines", "scale out",
                                                "map reduce", "parallel across nodes", "grid"))
    c.register_capability("Who's online (presence registry)", "mind.registry tracks live actors: announce(principal) is "
                          "a heartbeat, registry.list(kind=, workspace=) discovers who's here, is_online() checks one, "
                          "and an actor that stops heart-beating for `ttl` seconds drops out on its own. Rides the "
                          "mind's bus so presence is visible across nodes -- how a swarm or farm finds its peers.",
                          example="mind.registry.announce(agent); online = mind.registry.list(kind='agent'); mind.registry.is_online(agent)",
                          native=True, aliases=("registry", "presence", "who is online", "heartbeat", "discover peers",
                                                "list agents", "who's connected", "liveness", "roster", "online users",
                                                "node discovery"))
    c.register_capability("Invite guests and share selectively (access control)", "control who reads what. mind.invite("
                          "kind, grants) mints a token admitting a guest with specific initial read grants; mind.admit("
                          "code, id) redeems it into a scoped Principal that reads ONLY what it was granted (default: "
                          "nothing but its own namespace) and writes only its own. mind.grant / mind.revoke share and "
                          "un-share namespaces later; holographic_access.require_readable is the read chokepoint (the "
                          "symmetric twin of the DB's write-only-your-own rule).",
                          example="code = mind.invite(kind='user', grants={'read':['lab/scene']}); g = mind.admit(code, 'visitor'); mind.grant(g, read='lab/notes')",
                          native=True, aliases=("access control", "invite", "grant", "revoke", "permissions", "share",
                                                "who can read", "admit a guest", "invite token", "selective sharing",
                                                "read grant", "guest access", "authorize", "private namespace"))
    c.register_capability("Fork and apply a shared world (workspace)", "mind.workspace.fork(name) hands out a "
                          "copy-on-write editing view of a named world (a set of vector SLOTS): reads fall through to "
                          "the shared base, writes accumulate in the fork's private .delta, so your edits don't touch "
                          "the shared world (or another fork) until you reconcile. Feed the deltas to mind.merge_forks, "
                          "then mind.apply(merged, world=name) writes the agreed edits back. Closes the "
                          "fork -> edit -> merge -> apply loop; a world is a seed + deltas, so only the sparse changes "
                          "travel.",
                          example="f = mind.workspace.fork('lab'); f.set('sky', v); mind.apply(mind.merge_forks([f.delta, other])['merged'], world='lab')",
                          native=True, aliases=("workspace", "fork a world", "apply changes", "copy on write", "world",
                                                "shared world", "checkout", "branch a world", "edit in isolation",
                                                "commit changes", "seed and deltas", "single-player fork"))
    c.register_capability("Merge forked worlds (fork/merge)", "mind.merge_forks(forks, policy, tol) reconciles several "
                          "forked copies of a world, each a {slot: vector} delta. Slots the forks AGREE on merge "
                          "conflict-free into the consensus (pairwise opponent divergence below tol, matching leOS's "
                          "pairwise convention); slots they DISAGREE on are handled by policy: 'select' surfaces the "
                          "conflict for a human, 'auto' keeps only agreements, 'left'/'right'/callable resolve it. "
                          "Because a world is a seed + deltas, forking to single-player and merging back is cheap. "
                          "Returns {merged, conflicts}.",
                          example="res = mind.merge_forks([mine, theirs], policy='select'); apply(res['merged']); resolve(res['conflicts'])",
                          native=True, aliases=("merge", "merge forks", "fork and merge", "reconcile", "combine worlds",
                                                "resolve conflicts", "multiplayer merge", "branch and merge", "diff merge",
                                                "three-way merge", "collaborative edit", "sync changes"))
    c.register_capability("Scoped identity for any actor (Principal)", "mind.principal(id, workspace, kind) gives an "
                          "agent, user, service, or peer leCore instance ONE scoped identity where isolation is the "
                          "default: a private database namespace (it writes only there), a directed inbox topic (it "
                          "reads only its own messages, sender-stamped), a provenance role that tags everything it "
                          "contributes (holographic_provenance.source_role / from_external), and an optional private "
                          "learning overlay. Signals and state can't cross between principals -- so multiplayer "
                          "workspaces, agent swarms, and guest peer nodes are the same isolation solved once.",
                          example="alice = mind.principal('alice', workspace='lab', kind='user'); alice.send(mind.bus(), to='bob', payload={...}); alice.poll(mind.bus())",
                          native=True, aliases=("principal", "identity", "scoped identity", "per-agent state",
                                                "per-user namespace", "multiplayer", "multi-user", "swarm", "agent isolation",
                                                "inbox", "directed message", "provenance", "source role", "who sent this",
                                                "guest", "peer node", "federation", "workspace member"))
    c.register_capability("Serve leCore as a tool (/tools + /invoke)", "run the HTTP service (holographic_service.serve) "
                          "and any harness, LLM, or another leCore drives this node over two endpoints: GET /tools "
                          "returns the manifest of every public faculty (name, description, params); POST /invoke with "
                          "{name, args} runs one faculty and returns its result as JSON. Token-gated; private methods "
                          "are refused. This is leCore AS a tool provider -- the same shape every node speaks.",
                          example="from holographic_service import serve; serve(host='127.0.0.1', port=8080, token='secret')  # GET /tools ; POST /invoke {name,args}",
                          native=True, aliases=("serve as a tool", "tool server", "/tools", "/invoke", "expose faculties",
                                                "http api", "call leCore remotely", "function calling", "tool manifest",
                                                "let an agent use leCore", "let an llm call leCore"))
    c.register_capability("Use external tools (remote nodes / LLMs / commands)", "leCore CALLS tools in the same shape it "
                          "serves them. holographic.io_and_interop.holographic_toolclient.remote_tools(base_url, token) fetches another node's "
                          "/tools and yields each as a callable RemoteTool (its run(args) POSTs to that node's /invoke). "
                          "mind.attach_llm(callable) wires an LLM (any text->text, no SDK). mind.orchestrator.register / "
                          "register_command / register_remote add remote tools, shell programs (allowlisted), and whole "
                          "remote nodes so a planner can chain local faculties, remote tools, LLMs, and commands "
                          "uniformly.",
                          example="for t in remote_tools('http://host:8080', token='x'): mind.orchestrator.register(t)  # + mind.attach_llm(llm); mind.orchestrator.register_command('ffmpeg', ['ffmpeg','-i','{}'])",
                          native=True, aliases=("call a tool", "remote tools", "use an llm", "attach llm", "orchestrator",
                                                "register a tool", "run a command", "shell command tool", "call another node",
                                                "chain tools", "planner", "toolclient", "peer node", "federation"))
    c.register_capability("Agreement across estimates (opponent)", "given TWO estimates of the SAME thing (two models, "
                          "two solvers, two forked worlds, two farm nodes), mind.opponent_channels(a, b) decomposes "
                          "their disagreement (opponent-processing, ported from leOS) into: agreement (what both see), "
                          "a_exclusive / b_exclusive (what only each sees), magnitude_dispute, PURPLE (a_exclusive + "
                          "b_exclusive -- the emergent signal in NEITHER alone), and divergence_score (the angular "
                          "disagreement). Act on the agreement when divergence is small; surface the conflict when "
                          "it's large. classify() names the disagreement type; blend() mixes them by the channels.",
                          example="ch = mind.opponent_channels(est_a, est_b); if ch['divergence_score'] < 0.2: use ch['agreement']  # else look at ch['purple']",
                          native=True, aliases=("opponent", "agreement", "disagreement", "purple channel", "consensus",
                                                "vote", "voting", "ensemble", "combine estimates", "reconcile",
                                                "who agrees", "divergence", "abstain when uncertain", "cross-check",
                                                "opponent channels", "emergent signal", "leos opponent"))
    c.register_capability("Refine loop (produce / critique / adjust)", "mind.refine(produce, critique, adjust, accept, "
                          "budget) makes a result, has a CRITIC score it (a metric, opponent agreement, a model, or a "
                          "human), adjusts, and retries until it's good enough or the budget runs out -- the pipeline "
                          "middle that sits leCore between a big compute and a checker. Returns {result, score, "
                          "accepted, tries}. The callable-critic sibling of project_onto_constraints.",
                          example="log = mind.refine(produce=lambda: gen(), critique=score, adjust=lambda r,s: tweak(r,s), accept=0.9)",
                          native=True, aliases=("refine", "iterate", "produce critique adjust", "retry until good",
                                                "optimization loop", "analysis by synthesis", "draft and revise",
                                                "improve until accepted", "critic loop", "feedback loop"))
    c.register_capability("Import artist file formats (OBJ/glTF/textures/volume)", "import the files artists hand you: "
                          "mind.load_obj('model.obj') reads Wavefront geometry + its .mtl (UVs, normals, per-face "
                          "material, map_* textures); mind.load_glb('model.glb') reads glTF/GLB geometry AND its full PBR "
                          "channels (base colour / metallic-roughness / normal / occlusion / emissive) with embedded "
                          "textures and per-vertex UVs/normals, AND for rigged models its ANIMATIONS (keyframed node "
                          "transforms -- clip.sample(t), rotations slerped) and SKINS (joints + inverse-bind + weights); "
                          "mind.load_texture_set(folder) turns a folder of Adobe Substance 3D Painter export maps "
                          "(basecolor/roughness/metallic/normal/height/ao/emissive, matched by name) into one "
                          "PBRMaterial; mind.load_volume('grid.npy') wraps a 3-D density grid as a field for "
                          "render_volume. mind.import_asset(path) dispatches by extension. Once a rigged glTF is loaded, "
                          "mind.deform_mesh(loaded, clip, t) actually MOVES it -- linear-blend skinning by the animated "
                          "skeleton plus morph-target blending, returning the deformed mesh at time t. Stdlib+NumPy; PIL "
                          "lazy for textures. HONEST: proprietary .sbsar/.spp and sparse OpenVDB .vdb need their vendor tools -- "
                          "import the exported open forms.",
                          example="lm = mind.load_obj('chair.obj'); glb = mind.load_glb('robot.glb'); mat = mind.load_texture_set('exports/brick'); vol, b = mind.load_volume('smoke.npy')",
                          native=True, aliases=("import", "load obj", "load gltf", "load glb", "mtl", "wavefront",
                                                "substance painter", "adobe painter", "texture set", "pbr material import",
                                                "load model", "import mesh", "volumetric", "load volume", "vdb", "voxel",
                                                "density grid", "import material", "3d file", "asset import",
                                                "animation", "skin", "rigged", "keyframe", "skeleton", "uv", "channels",
                                                "deform", "skinning", "linear blend skinning", "morph", "blend shape",
                                                "pose a rig", "animate a model"))
    c.register_capability("Cold storage (compress inactive data)", "shrink INACTIVE data to save memory and disk, and "
                          "inflate it back on demand: store = mind.cold_store(keep_warm=8) keeps only the K most-recently-"
                          "used values live and compresses the rest, warming any of them transparently on get(); "
                          "mind.cool(big_table) wraps ONE value so c.cool() frees its RAM and c.get() brings it back "
                          "bit-identical. Works on tables, whole databases, big arrays, any picklable structure; "
                          "codec='lzma' packs smaller, spill_dir=... writes cold blobs to disk. Honest: high-entropy VSA "
                          "vectors barely compress (the win there is freeing the live object / spilling to disk); "
                          "redundant/text/structured data compresses a lot. The query Database can auto-cool its own "
                          "idle tables: db.enable_cold_storage(keep_warm=K) then db.cool_idle() compresses tables you "
                          "haven't queried lately and a query warms them back -- and a DB shipped to a distributed "
                          "worker arrives warm + cooling-off, so a shared read-only cache is never mutated.",
                          example="store = mind.cold_store(keep_warm=4); store.put('t1', big_table); store.get('t1')  # transparently warmed",
                          native=True, aliases=("cold storage", "compress inactive", "evict", "spill to disk", "cool",
                                                "warm", "fold up", "shrink memory", "free ram", "compress table",
                                                "compress database", "lazy inflate", "lru cache eviction", "page out",
                                                "auto cool tables", "idle table compression"))
    c.register_capability("File map ingest (folder / zip -> queryable)", "point at a FOLDER, a .zip, or a file and "
                          "digest it into a queryable FILE MAP: fm = mind.ingest_files('project/') (or 'bundle.zip'). "
                          "Query it by NAME/glob (fm.find('*.png')), KIND (fm.by_kind('model'): image/text/model/data/"
                          "code/archive), METADATA (larger_than/newer_than/by_ext), text CONTENT (fm.search_text('shader "
                          "normal') -- an inverted index over the text files), and MEANING (fm.build_meaning_index() then "
                          "fm.find_by_meaning('lighting')). fm.tree() is the folder hierarchy. Every file is also tracked "
                          "for RELOCATION/CHANGE (fm.missing()/changed()/relink(one,new)/resolve_assets(roots)), so a "
                          "moved/edited tree self-heals. Stdlib only; text indexing is size-capped.",
                          example="fm = mind.ingest_files('my_project.zip'); fm.find('*.obj'); fm.search_text('normal map'); fm.tree()",
                          native=True, aliases=("ingest", "ingest files", "index a folder", "digest a folder", "read a zip",
                                                "scan folder", "file map", "make files queryable", "search my files",
                                                "index files", "folder to database", "query a directory", "catalog files",
                                                "import a folder", "unzip and index"))
    c.register_capability("Asset relocation / relink (external files)", "track the EXTERNAL files a scene depends on "
                          "(textures, models, ...) and repair their paths when they move -- the '3-D missing textures' "
                          "problem. lib = mind.asset_library(); lib.add(path); then when a folder moves, lib.relink("
                          "one_asset, its_new_path) re-finds every OTHER moved file automatically (it works out the "
                          "moved parent and rewrites the rest, then structurally SEARCHES for anything reorganised). "
                          "lib.changed() spots files edited on disk (size/mtime or content hash); lib.search_under("
                          "folder) finds missing files under a folder; lib.resolve(asset, roots=) locates a file by "
                          "CONTENT HASH across machines (the distributed fallback). Saves/loads a JSON manifest.",
                          example="lib = mind.asset_library(); lib.add('project/textures/water/wave.png'); lib.relink(lib.assets[0], 'newroot/project/textures/water/wave.png')",
                          native=True, aliases=("asset", "assets", "relink", "relocate", "missing textures", "broken path",
                                                "fix paths", "external files", "find moved files", "asset paths",
                                                "texture path", "reconnect assets", "repath", "file moved", "asset manifest"))
    c.register_capability("Message bus + agent (LLM) bridge", "connect a person AND an agent to the running tool at "
                          "once, and let the app PUSH to the agent instead of the agent polling: mind.bus() is a "
                          "message bus (publish/subscribe by topic, mailboxes to pull an inbox, history); "
                          "mind.run_task('render', fn, background=True) runs a job and publishes 'render.done' with a "
                          "small summary when it finishes; mind.agent_bridge(llm=my_fn).notify_on('render.done', 'does "
                          "it look right?') calls YOUR llm (any text->reply callable -- no LLM library is imported, so "
                          "it's fully optional) and posts the reply on the bus. Over HTTP a remote agent uses "
                          "/bus/publish + /bus/poll. The LLM is optional; leCore runs with no agent attached.",
                          example="bridge = mind.agent_bridge(llm=my_llm); bridge.notify_on('render.done', 'does it look right?'); mind.run_task('render', lambda: scene.render(), background=True)",
                          native=True, aliases=("message bus", "event bus", "pubsub", "publish subscribe", "agent bridge",
                                                "llm bridge", "notify the agent", "push notification", "on render done",
                                                "connect an agent", "send message to agent", "mailbox", "inbox",
                                                "trigger the llm", "watch for events", "task done event"))
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
                          "post-FX. The analysis-by-synthesis render path", example="mind.path_trace(scene); mind.camera(); from holographic.rendering.holographic_raymarch import sphere_trace",
                          native=True, aliases=("render a scene", "path trace", "ray tracing", "global illumination",
                                                "camera", "depth of field", "lens", "volumetric render", "radiance transfer",
                                                "prt", "ambient occlusion", "post processing", "gbuffer", "raytrace", "render"))
    c.register_capability("Mesh editing (DCC)", "modeling/DCC edits on a Mesh: extrude/inset faces (meshpoly), "
                          "subdivide + smooth (meshsubdiv, Catmull-Clark), deform/warp (deform), rig-skin-pose a "
                          "skeleton (blendpose), UV unwrap (chart), decimate/QEM, booleans, and mesh<->SDF. "
                          "Blender-parity polygon editing", example="mind.deform(mesh, ...); mind.mesh_to_sdf(mesh); from holographic.mesh_and_geometry.holographic_meshverbs import extrude_face",
                          native=True, aliases=("edit a mesh", "extrude", "bevel", "inset", "subdivide", "smooth a mesh",
                                                "decimate", "reduce polygons", "uv unwrap", "unwrap uv", "rig", "skin",
                                                "pose a skeleton", "skeleton", "deform", "boolean", "remesh", "dcc", "modeling"))
    c.register_capability("SDF & procedural geometry", "implicit + procedural geometry: signed distance fields (sdf), "
                          "sphere-trace raymarching with ambient occlusion (raymarch), sculpting, procedural terrain "
                          "(procgen), spatial tiling + octree, and voxelization. Native-first shape building",
                          example="from holographic.rendering.holographic_raymarch import sphere_trace; mind.terrain(...); from holographic.mesh_and_geometry.holographic_sdf import ...",
                          native=True, aliases=("sdf", "signed distance field", "raymarch", "sphere trace", "sculpt",
                                                "procedural terrain", "procedural geometry", "voxelize", "voxel", "octree",
                                                "tile in space", "implicit surface", "marching"))
    c.register_capability("Navigation & planning", "find a way through a space or structure: A*/shortest-path route "
                          "planning (plan), slime-mould flow networks (flow), tree/graph navigation (navigator), and "
                          "maze solving. Pathfinding on the VSA substrate", example="from holographic.scene_and_pipeline.holographic_plan import ...; mind.solve_maze(world); from holographic.misc.holographic_flow import ...",
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
                          "point-cloud structure (cosmic), and time-series / market analysis", example="from holographic.misc.holographic_transport import wasserstein; from holographic.misc.holographic_graphsignal import laplacian_filter",
                          native=True, aliases=("data analysis", "cluster", "optimal transport", "wasserstein", "graph laplacian",
                                                "spectral", "dimensionality reduction", "embedding", "topology", "persistent homology",
                                                "kernel density", "point cloud", "time series", "statistics", "analytics"))
    c.register_capability("Symbolic reasoning", "recover structure symbolically: symbolic regression to find a formula "
                          "(symbolic), resonator networks that FACTOR a bound vector into its parts (sbc/resonator), "
                          "is_a taxonomy climbing, and relational reasoning over records. Turning data and vectors back "
                          "into laws", example="from holographic.agents_and_reasoning.holographic_symbolic import ...; mind.climb('dog'); from holographic.misc.holographic_sbc import ...",
                          native=True, aliases=("symbolic regression", "find a formula", "factor a vector", "resonator",
                                                "factorization", "decompose a signal", "reason", "reasoning", "climb hierarchy",
                                                "relational", "law from data"))
    c.register_capability("Signal & spectral", "1-D signal processing: FFT / spectral analysis (spectral), "
                          "faint-signal detection in noise with a calibrated false-discovery rate (signal_structure), "
                          "drifting-narrowband / de-Doppler search (dedoppler), spectral flatness, and bandwidth. The "
                          "radio-SETI-style detection stack", example="from holographic.sampling_and_signal.holographic_spectral import ...; from holographic.sampling_and_signal.holographic_dedoppler import ...",
                          native=True, aliases=("signal processing", "fft", "spectral", "spectrum", "detect a signal",
                                                "faint signal", "narrowband", "doppler", "dedoppler", "drift", "flatness",
                                                "bandwidth", "frequency", "audio"))
    c.register_capability("Compression & codec", "shrink data losslessly or by rate-distortion: a sequence/entropy "
                          "codec (codec), general compression (compress), rate-distortion quantization "
                          "(ratedistortion), and content-addressed storage (storage). How the engine fits vectors into "
                          "bytes", example="from holographic.misc.holographic_codec import ...; from holographic.misc.holographic_ratedistortion import ...",
                          native=True, aliases=("compress", "compression", "codec", "entropy coding", "rate distortion",
                                                "quantize", "content addressed storage", "encode data", "shrink data", "deduplicate"))
    c.register_capability("Video (temporal)", "temporal image sequences: video compression with keyframe/delta coding "
                          "(video), temporal compression, motion/phase morph between frames (phasemorph), and frame "
                          "interpolation. Moving pictures on the substrate", example="from holographic.io_and_interop.holographic_video import ...; mind.blend_images(a, b)",
                          native=True, aliases=("video", "compress a video", "temporal compression", "frames", "motion",
                                                "interpolate frames", "keyframe", "sequence of images", "movie"))
    c.register_capability("Honesty & measurement", "measure claims honestly: error bars + significance (measure), "
                          "ablation studies (ablate), proof-of-structure against a null (structure), calibrated "
                          "detection with false-discovery control, benchmark + variance harness, and stress tests. The "
                          "engine's own truth-in-advertising tools", example="from holographic.misc.holographic_measure import ...; from holographic.misc.holographic_ablate import ...",
                          native=True, aliases=("measure", "error bars", "significance", "ablation", "false discovery rate",
                                                "calibrated", "benchmark", "variance", "stress test", "proof of structure",
                                                "honesty", "null model", "confidence interval"))
    c.register_capability("Program & machine (VM)", "the VSA computer: a stored-program holographic machine "
                          "(machine/HoloMachine) that runs vector programs, recipes with holes / hygienic templates "
                          "(template), a content-addressed compile cache (compile), tool-orchestration planning "
                          "(orchestrator/voidsynth), and reversible computation. Programs as data", example="from holographic.agents_and_reasoning.holographic_machine import HoloMachine; from holographic.simulation_and_physics.holographic_template import RecipeTemplate",
                          native=True, aliases=("virtual machine", "stored program", "run a program", "vm", "recipe",
                                                "template", "recipe with holes", "compile", "content addressed compile",
                                                "orchestrate", "plan tools", "reversible computation", "bytecode"))
    # --- vendored knowledge: a real dictionary + taxonomy for contextual awareness ---
    c.register_capability("Dictionary + taxonomy (vendored)", "a comprehensive vendored English DICTIONARY (~144k "
                          "words: definition, part of speech, synonyms, example) AND an is_a TAXONOMY (encyclopedia "
                          "side: 'a dog is a kind of domestic animal...'), giving the engine real world-knowledge for "
                          "contextual awareness beyond its internal machinery. OPT-IN + lazy: it never loads from "
                          "importing leCore or building a mind -- only the first language call decompresses it (lzma, "
                          "~3.3 MB on disk) into a plain dict in RAM (~22 MB), after which lookups are instant. Control "
                          "it explicitly with holographic.misc.holographic_dictionary.is_loaded()/preload()/unload()/stats(). Stdlib-only "
                          "(lzma+json); the mind can also LEARN meaning from it. Princeton WordNet, free with attribution",
                          example="mind.lookup('gravity'); mind.word_taxonomy('dog'); import holographic.misc.holographic_dictionary as hd; hd.stats()",
                          native=True, aliases=("dictionary", "define", "definition", "word meaning", "synonyms",
                                                "encyclopedia", "taxonomy", "is a", "wordnet", "vocabulary",
                                                "contextual awareness", "knowledge", "lexicon", "what does word mean",
                                                "preload dictionary", "unload dictionary", "optional language"))
    c.register_capability("Semantic word index (find words by meaning)", "the fuzzy REVERSE of a dictionary: describe "
                          "an idea and get the words whose definitions mean it. mind.build_semantic_index(words=...) "
                          "places words in a meaning space by RANDOM INDEXING over their glosses, then idx.find('un"
                          "expected good luck') -> 'serendipity' and idx.similar('puppy') -> 'dog','kitten'. OPT-IN and "
                          "separate: nothing loads or builds until you call it. Approximate by design (this is where "
                          "leCore's geometry-preserving/lossy side belongs) -- reliable for the top hit, noisy in the "
                          "tail, and word-sense sensitive.",
                          example="idx = mind.build_semantic_index(words=my_vocab); idx.find('a young dog'); idx.similar('ocean')",
                          native=True, aliases=("semantic index", "find words by meaning", "reverse dictionary",
                                                "words like", "similar words", "meaning search", "word similarity",
                                                "describe a word", "what's the word for", "concept to word", "synonym search"))
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
                          "Cache home and shade via the Shading home", example="from holographic.materials_and_texture.holographic_material import Material",
                          native=True, aliases=("material", "channels", "albedo", "roughness", "metallic", "shader"))
    c.register_capability("Multi-material (mask-blended)", "combine N materials by per-point MASKS -- generalises the "
                          "2-way Material.blend to a weighted mix where each material's weight is a mask (a texture "
                          "graph, a field, or a constant) that varies over the surface: paint rust into metal, moss "
                          "onto stone, a decal onto a surface. 'blend' = soft weighted sum (weights normalised so "
                          "brightness stays put); 'select' = hard pick the dominant material (a material-ID / splat "
                          "map). CMP3",
                          example="mind.multi_material([metal, rust], [1.0, mind.texture_leaf('fbm', n_dims=2)]).sample('albedo', [0.3, 0.7])",
                          native=True, aliases=("multi-material", "multimaterial", "blend materials", "material mask",
                                                "material map", "splat map", "material id", "paint materials", "mix materials",
                                                "layer materials by mask"))
    c.register_capability("Layered material (order schema)", "an ORDERED stack of material layers -- base -> diffuse "
                          "-> specular/reflection -> coat/clearcoat -- where the order is a SCHEMA checked at compose "
                          "time, so you can't put a reflection under a diffuse (an out-of-order stack is refused up "
                          "front). Each layer composites OVER the one below by a coverage alpha (a number, field, or "
                          "texture graph). Honest: fixes the stacking, not the energy-conserving radiometry of a true "
                          "layered BRDF. CMP2",
                          example="mind.layered_material([mind.material_layer('base', paint), mind.material_layer('clearcoat', gloss, alpha=0.3)]).sample('albedo', [0.3, 0.7])",
                          native=True, aliases=("layered material", "material layers", "clearcoat", "coat", "layer stack",
                                                "material stack", "over compositing", "base diffuse specular coat",
                                                "stacked material", "material order"))
    c.register_capability("Shading (BRDF)", "the shade model: cook_torrance (full specular+diffuse per light), "
                          "lambert (diffuse term), sample_brdf (importance-sampled bounce) -- call these instead of "
                          "re-deriving Fresnel/GGX/diffuse", example="from holographic.rendering.holographic_brdf import cook_torrance, lambert",
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
                          example="from holographic.scene_and_pipeline.holographic_jobs import JobManager; m.create(id, buckets, worker); m.start(id, background=True); m.pause(id); m.resume(id)",
                          native=True, aliases=("job", "start", "pause", "resume", "cancel", "checkpoint", "render job",
                                                "long running", "background task", "resumable", "progress", "lifecycle"))
    c.register_capability("Code / file editing (agentic)", "read, view (line-numbered), write, exact-string replace, "
                          "replace-lines, insert/delete lines, grep, find-definition, list, tree, archive, move, and "
                          "UNDO -- structured source-file editing for an agent working the codebase, scoped to a project "
                          "ROOT so a path can never escape it. Atomic writes; replace requires a unique match; every "
                          "mutation is reversible with file_undo; replace_across renames a string across many files "
                          "(with a dry-run preview); python_check (syntax) and import_check (real import in a subprocess) "
                          "catch a broken edit immediately. Exposed as mind.file_* methods, so callable over the HTTP "
                          "tool protocol (GET /tools, POST /invoke) like any faculty",
                          example="mind.set_file_root('.'); mind.file_find_definition('make_cloud'); mind.file_replace('a.py', 'old()', 'new()'); mind.file_import_check('a.py'); mind.file_undo()",
                          native=True, aliases=("edit file", "edit code", "modify file", "modify code", "write file",
                                                "read file", "replace in file", "patch", "insert lines", "delete file",
                                                "archive file", "move file", "rename file", "grep", "search code",
                                                "list files", "create file", "file editing", "source editing",
                                                "undo edit", "undo my last edit", "find definition", "jump to definition",
                                                "rename symbol", "rename everywhere", "replace across files", "directory tree",
                                                "check imports", "did my edit break", "view file", "see the file"))
    c.register_capability("Background cloud bake (resumable)", "run the slow fBm noise bake behind a cloud render as a "
                          "monitorable background JOB you can pause/resume/cancel (even across a process restart), then "
                          "feed the baked grid straight into a render without re-baking. The agent-friendly way to "
                          "handle a render that takes minutes: kick it off, poll progress, do other work",
                          example="jid = mind.bake_cloud_job(radius=1.0, seed=0, background=True); mind.job_status(jid); mind.job_pause(jid); mind.job_resume(jid); grid = mind.job_result(jid)",
                          native=True, aliases=("bake cloud", "background render", "resumable render", "monitor render",
                                                "pause render", "long render", "render job", "noise bake"))
    c.register_capability("Compare rendered images (files)", "perceptual similarity in [0,1] between two images given "
                          "as FILE PATHS (e.g. two rendered PNGs) -- SSIM + colour + edge, shift/lighting-tolerant, the "
                          "on-disk companion to compare_images. The call an agent makes to check 'did my render change "
                          "or match the target?' when the images are files",
                          example="mind.compare_image_files('render_a.png', 'render_b.png')  # -> {similarity, distance, ...}",
                          native=True, aliases=("compare images", "image diff", "render diff", "compare renders",
                                                "image comparison", "did the render change", "image similarity"))
    c.register_capability("Distributed hardening (R5)", "fault tolerance + verification for untrusted farm nodes: "
                          "retry-with-backoff (a reissue reassigns a dead node\'s work), redundant computation + "
                          "majority VOTING (accept only what independent nodes agree on -- a node can\'t force a "
                          "result), canary buckets (known answers reject an untrusted node), and speculative straggler "
                          "backups. The BOINC/SETI@home discipline, mandatory before public contributors",
                          example="from holographic.misc.holographic_hardening import HardenedCoordinator; HardenedCoordinator(farm, redundancy=3).run(buckets, worker, cache, reduce, canaries=[...])",
                          native=True, aliases=("voting", "redundant compute", "retry", "fault tolerance", "canary",
                                                "untrusted node", "quorum", "straggler", "backup execution", "verify result"))
    c.register_capability("Network render farm", "run the coordinator\'s monoid workers on OTHER machines: a worker "
                          "daemon per node (stdlib http/json), the read-only cache shipped ONCE by content hash and "
                          "reused, buckets dispatched concurrently and reduced -- the same Coordinator.run as the local "
                          "pool. Buckets are data, workers are registered code; a node runs only its registered workers",
                          example="from holographic.misc.holographic_farm import WorkerDaemon, NetworkFarm; Coordinator(NetworkFarm([addr])).run(buckets, 'worker_name', cache, reduce)",
                          native=True, aliases=("render farm", "distributed", "network", "seti", "worker daemon",
                                                "remote", "cluster", "node", "another machine", "farm"))
    c.register_capability("Command runner (external tools)", "run any registered ALLOWLISTED program/script as a "
                          "task (subprocess, no shell, time-boxed) and wire it as an orchestrator Tool the Planner "
                          "can chain, with a CircuitBreaker on a flaky one -- the door to external tools and services. "
                          "SECURITY: allowlist only, never a command from untrusted input, values fill placeholders",
                          example="from holographic.scene_and_pipeline.holographic_command import CommandRunner, command_as_tool; r.register('ffmpeg', [...]); r.run('ffmpeg', args)",
                          native=True, aliases=("run command", "external tool", "subprocess", "shell", "run program",
                                                "ffmpeg", "job runner", "allowlist", "command backend"))
    c.register_capability("False-discovery gate over an ablation table", "one claim gets an honest CI from measure(); "
                          "a TABLE of ablations is a scan, and scanning enough subsystems means one clears its bar by "
                          "luck. measure.fdr_gate(rows, alpha) applies Benjamini-Yekutieli across the whole family "
                          "(paired permutation p-values, dependent=True) and reports how many survive.",
                          example="aug, n_load_bearing, n_survive = measure.fdr_gate(rows, alpha=0.1)",
                          native=True, aliases=("fdr", "false discovery", "ablation table", "multiple testing",
                                                "look elsewhere", "benjamini", "is this component load-bearing"))
    c.register_capability("Database layers: durability, locking, history, graph", "opt-in layers on the query Database: "
                          "db.snapshot(path)/Database.restore(path) and db.journal(path) for crash-safe durability; "
                          "db.writer_lock() and db.snapshot_reader(table) for one-writer/many-reader concurrency; "
                          "db.versioned(table) for committed history and time travel; db.adjacency(edges, src, dst) "
                          "for graph traversal. A plain Database pays nothing for them.",
                          example="db.snapshot('/tmp/s.json'); db2 = Database.restore('/tmp/s.json'); vt = db.versioned('shop.items')",
                          native=True, aliases=("durable database", "snapshot", "journal", "wal", "crash safe",
                                                "single writer lock", "concurrency", "time travel", "versioned table",
                                                "graph traversal", "adjacency", "database layers"))
    c.register_capability("Compose new scenes from tags (forward generation)", "the resonator run FORWARD: bind an "
                          "object's (colour, shape, texture) tags into a composite vector and superpose objects into "
                          "a scene -- composing what was never stored, rather than morphing what was. "
                          "mind.novel_object_specs() enumerates the whole generation space.",
                          example="specs = mind.novel_object_specs(); scene = mind.compose_from_tags(specs[:3])",
                          native=True, aliases=("compose a scene", "forward generation", "generate new objects",
                                                "novel combinations", "compose from tags", "procedural scene"))
    c.register_capability("Regime-shift detector (fast/slow layers)", "borrowed from ocean physics: a FAST component "
                          "tracks the present while a SLOW one holds the persistent state; when their divergence stays "
                          "high the system commits to a new LAYER. mind.regime_detector().observe(x) -> (divergence, "
                          "layer, started_new_layer). Tells a genuine regime CHANGE from a wobble.",
                          example="d = mind.regime_detector(); div, layer, new = d.observe(x)",
                          # NB: no bare "diffusion" alias -- it collides with the reaction-diffusion automaton home
                          # and displaced it for the probe "reaction diffusion cellular automaton". One token, two
                          # unrelated meanings; the specific phrase keeps this findable without stealing that query.
                          native=True, aliases=("regime shift", "change point", "drift detection", "has the data changed",
                                                "double-diffusive", "layer detection", "concept drift",
                                                "has the regime changed"))
    c.register_capability("Grid-free PDE solve on an SDF (Walk on Stars)", "solve Laplace or Poisson inside an SDF "
                          "domain with NO MESH, no grid and no global linear system: mind.solve_laplace(sdf, points, "
                          "boundary_value) walks from each point to the boundary and averages what it finds there. "
                          "Pointwise (evaluate only where you care), progressive (error falls as 1/sqrt(walks)), and "
                          "farm-parallel with NO seed coordination -- every random number is a pure function of "
                          "position (hash_unit). Pass dirichlet_sdf to make the rest of the boundary zero-flux "
                          "(Neumann/insulating) -- that is Walk on STARS, which vanilla Walk on Spheres cannot do.",
                          example="u = mind.solve_laplace(sdf.eval, pts, boundary_value, walks=1024, dim=3)  # + dirichlet_sdf= for insulating walls",
                          native=True, aliases=("solve laplace", "poisson equation", "pde without a mesh", "walk on spheres",
                                                "walk on stars", "grid free solver", "harmonic function", "steady state heat",
                                                "boundary value problem", "mesh free", "monte carlo pde", "diffusion curves"))
    c.register_capability("Stateless coordinate-keyed randomness (hash_unit)", "np.random carries STATE, so the n-th "
                          "draw depends on every draw before it -- fatal for farm work, where bucket order then "
                          "changes the numbers. hash_unit(x, y, walk, step, seed) makes the value a pure FUNCTION of "
                          "where and which: same inputs, same value, on any node, in any order, with no seed "
                          "coordination at all. Pure integer arithmetic, independent of PYTHONHASHSEED. "
                          "hash_direction() gives a uniform direction on the sphere or circle.",
                          example="from holographic.misc.holographic_determinism import hash_unit, hash_direction; u = hash_unit(x, y, bounce, seed)",
                          native=True, aliases=("stateless random", "hash noise", "coordinate keyed", "no seed coordination",
                                                "reproducible random", "farm parallel sampling", "hash_unit",
                                                "deterministic sampling", "per pixel random"))
    c.register_capability("Exact periodic PDE solve (spectral Laplace)", "on a PERIODIC grid the Laplacian is a "
                          "circular convolution, so it is DIAGONAL in the Fourier basis and the solve is closed "
                          "form. mind.solve_poisson_periodic(f) inverts laplacian(u)=f in one FFT; "
                          "mind.diffuse_periodic(T, alpha, t) evolves the heat equation to ANY time t in one "
                          "evaluation (each mode decays by exp(-alpha k^2 t)) -- no time step, no stability limit, "
                          "no substepping. Measured exact to 6.7e-16 where 1000 iterative steps sit at 1.5e-4. "
                          "Periodic only: the Neumann/edge-replicated Laplacian is NOT circular.",
                          example="u = mind.solve_poisson_periodic(f, dx=1/64); T = mind.diffuse_periodic(T0, alpha=0.01, t=1e6, dx=1/64)",
                          native=True, aliases=("spectral laplace", "poisson fft", "closed form heat", "exact diffusion",
                                                "periodic pde", "fourier solve", "no time step", "steady state exact",
                                                "diagonalise the laplacian"))
    c.register_capability("Multi-way tensor compression (Tucker / TT)", "compress data with structure along SEVERAL "
                          "axes -- a field over (x,y,t), a frame stack, a BRDF table, a volume -- by factoring every "
                          "mode at once. mind.compress_tensor(X, method='tucker') uses HOSVD with a RANK GATE that "
                          "picks ranks from the singular spectrum; method='tt' uses a Tensor Train whose storage is "
                          "linear in the number of modes. Measured on a real diffusing field: 57x at rel-err 7.5e-3, "
                          "against 5.9x for a per-slice SVD (which sees structure within a frame but none across "
                          "frames). On data with NO low-rank structure the gate returns full rank -- store it raw. "
                          "Never CP: for 3+ modes a best rank-R CP approximation may not even exist.",
                          example="code = mind.compress_tensor(field, energy=0.999); X = mind.decompress_tensor(code)",
                          native=True, aliases=("tensor compression", "tucker", "hosvd", "tensor train", "low rank tensor",
                                                "compress a volume", "compress a frame stack", "multiway svd",
                                                "rank gate", "should i compress this"))
    c.register_capability("Denoise multi-way data (low-rank tensor prior)", "clean a noisy field over several axes "
                          "-- (x,y,t), a frame stack, a volume -- by projecting onto the low-rank manifold the noise "
                          "level implies. mind.denoise_tensor(X) estimates sigma itself and keeps only singular "
                          "values a noise matrix could not produce. Measured: 31.5 dB -> 48.6 dB on a real diffusing "
                          "field, where a per-slice SVD denoiser reaches 39.5 (it is blind to correlation ACROSS "
                          "slices). KEPT NEGATIVE: a low-rank prior is a claim about the signal -- on a FULL-RANK "
                          "signal it destroys the data (43 dB -> 17 dB). Check the rank gate first.",
                          example="clean, ranks, sigma = mind.denoise_tensor(noisy_field)",
                          native=True, aliases=("denoise a volume", "denoise a field", "low rank denoise",
                                                "tensor denoising", "clean a frame stack", "remove noise from a field",
                                                "multiway denoise"))
    c.register_capability("Store a multi-way array (tensor-train file)", "holographic_tucker.save_tensor(X, path) "
                          "writes a volume / frame stack / BRDF table as a Tensor-Train code, and load_tensor reads "
                          "it back. Measured on a real (24,32,32) field: 4,433 bytes at rel-err 3.9e-5, against "
                          "int8's 24,576 bytes at 9.5e-3 -- 5.6x smaller AND 244x more accurate. The bar is INT8 (1 "
                          "byte/element), not float64: on data with no cross-mode structure the TT code is bigger, "
                          "and the file falls back to storing the array RAW and exact. core.save(quant='rd'/'auto') "
                          "carries the same decision for 3+ mode state arrays.",
                          example="from holographic.caching_and_storage.holographic_tucker import save_tensor, load_tensor; save_tensor(volume, 'v.tt'); X = load_tensor('v.tt')",
                          native=True, aliases=("save a volume", "store a frame stack", "tensor train file",
                                                "compress and save a field", "tt file", "multiway storage"))
    c.register_capability("Will compression pay? (area law vs volume law)", "mind.tensor_structure(X) answers, before "
                          "you pay to find out, whether a tensor factorisation can help. It compares the rank kept at "
                          "every cut (how many numbers must cross that boundary) to the most it could possibly be. "
                          "Ranks far below the bound = an AREA LAW: the cost of a cut is set by its boundary, not the "
                          "volume it encloses, and a tensor train is cheap. Ranks that saturate = a VOLUME LAW: every "
                          "degree of freedom is independent, store it raw. Measured: a diffusing field scores 0.21 "
                          "(TT: 4,394 B vs int8 24,576); white noise scores 1.00 (TT: 104,782 B).",
                          example="v = mind.tensor_structure(field); v['verdict']  # 'area-law' or 'volume-law'",
                          native=True, aliases=("will compression help", "area law", "volume law", "schmidt rank",
                                                "bond rank", "is this compressible", "should i compress this",
                                                "entanglement entropy", "structure diagnostic"))
    c.register_capability("N filter passes in one evaluation (shader algebra)", "a circular convolution is diagonal "
                          "in the Fourier basis, so applying it N times is just the transfer raised to the N-th "
                          "power. mind.filter_passes(field, kernel, N) costs the same whether N is 1 or 1,000,000 "
                          "(measured 1,824x faster at N=4096, exact to 2.3e-14). Two things a GPU cannot do: N may "
                          "be FRACTIONAL (half a blur pass; two halves compose to one), and N may be INFINITE -- "
                          "mind.filter_limit returns the steady state as an idempotent projection, where a literal "
                          "loop can need 200,000 passes.",
                          example="soft = mind.filter_passes(img, blur, 64); half = mind.filter_passes(img, blur, 0.5); steady = mind.filter_limit(img, blur)",
                          native=True, aliases=("many blur passes", "iterated filter", "blur n times", "fractional blur",
                                                "half a pass", "steady state filter", "filter to convergence",
                                                "shader algebra", "operator power"))
    c.register_capability("Bake a function into one vector (texture unit)", "mind.bake_field(xs, ys) stores a sampled "
                          "function as a SINGLE hypervector; mind.fetch_field(bake, x) reads it back at ANY x with one "
                          "dot product -- interpolation is built into the algebra, no grid, no lookup table. THE "
                          "ALGEBRA HAS A NYQUIST: the phasor bandwidth sets the finest detail the code can hold, and "
                          "below the signal's maximum angular frequency the bake does not blur, it returns a "
                          "confident WRONG answer and raises nothing. So the bandwidth is chosen from the data "
                          "(measured: RMS error under 0.06 at every frequency tried; half that bandwidth gives "
                          "0.09-0.30). Supplying too small a bandwidth warns.",
                          example="b = mind.bake_field(xs, ys); y = mind.fetch_field(b, 0.37)   # any x, one dot product",
                          native=True, aliases=("bake a function", "texture unit", "store a curve", "lookup table",
                                                "interpolate anywhere", "bandwidth", "nyquist", "sample a field",
                                                "function encoding"))
    c.register_capability("Detrend before you bake (non-periodic functions)", "the bandwidth probe is an FFT, and an "
                          "FFT treats its samples as PERIODIC. Any function whose endpoints disagree carries an "
                          "implicit jump at the wrap, and a jump has an unbounded spectrum -- so a STRAIGHT LINE "
                          "probes at 607.9 where sqrt probes at 789.7 and a real 2-cycle sine probes at 12.5, and "
                          "the bake spends its capacity on frequencies that do not exist. "
                          "mind.bake_field(xs, ys, detrend=True) subtracts the endpoint line, bakes the residual, "
                          "and restores the line analytically at fetch time. Measured absolute relative error, mean "
                          "+- sd over 12 seeds, plain vs detrended: sqrt 0.111 +- 0.038 -> 0.009 +- 0.005 (12.6x), "
                          "cube root 0.140 -> 0.017 (8.3x), f(x)=x 0.133 -> exactly 0.000. The plain bake is also "
                          "UNSTABLE (1/(x+0.05) scores 1.83 +- 4.25) because an inflated bandwidth collapses the "
                          "kernel toward a delta. It costs nothing when the endpoints already agree. RETIRED "
                          "NEGATIVE: 'near-singular functions need domain warping' -- wrong cause (the wrap, not "
                          "the singularity) and the weaker fix (warping buys 1.9x where detrending buys 8-16x).",
                          example="b = mind.bake_field(xs, ys, detrend=True); y = mind.fetch_field(b, 0.37, normalize=True)",
                          native=True, aliases=("detrend", "bake a lookup table", "bake sqrt", "non-periodic bake",
                                                "endpoint jump", "spectral leakage", "lut", "near singular function"))
    c.register_capability("Bake an N-D function into one vector (n-D texture unit)", "mind.bake_field_nd(grids, "
                          "values) stores a gridded function of several variables as a SINGLE hypervector, read back "
                          "at any point with mind.fetch_field_nd. The per-axis bandwidths are probed FROM THE DATA, "
                          "because the underlying n-D encoder's default of 3.0 measures at 1.0019 scale-free RMS on "
                          "a 2-D sine -- literally no information, silently. Probed, the same bake lands at 0.101. "
                          "There is NO capacity budget on the number of bundled points (a bundled function is only "
                          "ever summed, never unbound): at a fixed bandwidth the error is flat as the grid goes 400 "
                          "-> 6400 points (0.098 -> 0.118). BANDWIDTH IS A BIAS-VARIANCE DIAL AND dim IS THE "
                          "VARIANCE BUDGET, and the causal variable is B = margin * w_max, not margin: on a 1-cycle "
                          "sine margin 1.5 (B=9.4) is bias-limited and 16x the dimension buys nothing (0.1179 at "
                          "D=4096 vs 0.1191 at D=65536), while at B=18.8 the same signal is variance-limited and D "
                          "pays (0.122 -> 0.043). THE DIAGNOSTIC COSTS ONE EXTRA BAKE: double dim -- if the error "
                          "drops keep spending dimension, if it does not move raise the margin. KEPT NEGATIVE: at "
                          "the default margin this is a SHAPE estimator, amplitude gain 0.66; raise margin and dim "
                          "together or calibrate the gain.",
                          example="b = mind.bake_field_nd([xs, ys], V); v = mind.fetch_field_nd(b, [0.3, 0.7])",
                          native=True, aliases=("bake a 2d function", "n-d texture unit", "bake a volume",
                                                "multivariate lookup table", "encode a 2d point", "bake a grid",
                                                "n dimensional function encoding", "bake a field over a grid"))
    c.register_capability("Subdivision limit surface (closed form)", "mind.mesh_limit_surface(mesh) returns where "
                          "infinite Loop subdivision would put every vertex, plus the EXACT limit normal there -- in "
                          "O(V), performing no subdivision at all. The ring-to-ring block of the local Loop operator "
                          "is exactly a CIRCULANT, i.e. a bind operator, so iterate.transfer diagonalises it for "
                          "free: mode 0 (eigenvalue 5/8 at every valence) gives the limit position, modes +-1 span "
                          "the tangent plane so the normal is exact rather than area-weighted (0.0000 degrees "
                          "against a 6x-subdivided icosphere), and Warren's beta is read off the spectrum instead "
                          "of hard-coded. Deep subdivision converges to it: 6.0e-4 -> 3.7e-5 -> 2.3e-6 at k=4/6/8. "
                          "HONEST SCOPE: this is the k -> infinity case; a FINITE number of levels on an irregular "
                          "mesh still needs the full Stam evaluation, so use mind.mesh_subdivide(mesh, k) there.",
                          example="positions, normals = mind.mesh_limit_surface(mesh)",
                          native=True, aliases=("limit surface", "loop limit", "subdivision limit",
                                                "exact limit normal", "infinite subdivision", "smooth normals",
                                                "push vertices to the limit", "stam evaluation"))
    c.register_capability("Frequency-lifted (Gabor) splats", "mind.splat_field(img, k, basis='gabor') gives each "
                          "splat a FREQUENCY, ORIENTATION and PHASE -- a Gabor atom, seven numbers instead of four. "
                          "A Gabor atom is a BANDPASS primitive, so it buys you exactly the band it is tuned to. "
                          "Measured at equal PARAMETER budget against a jointly-refit Gaussian fit: +7.0 dB on a "
                          "narrowband oriented grating, +0.2 dB on a sharp broadband edge, +0.1 dB on noise-like "
                          "texture -- and it costs 89x the fitting time (a 196-atom dictionary per placement against "
                          "4). The extra dimensions are a levy paid up front, so the win grows with budget (+0.6 dB "
                          "at 224 numbers, +7.5 dB at 1,344). KEPT NEGATIVE, against the prediction that motivated "
                          "it: this does NOT dissolve the splatsharpen negative, which was recorded on a sharp edge "
                          "-- an edge is not a band, it is every band at once. And the Gaussian basis it was "
                          "supposed to beat was never saturated: that flat-in-K curve was greedy matching pursuit's "
                          "overlap double-counting, which splat_refit already fixed (12.9 -> 20.9 dB across K). "
                          "Use mind.spectral_detail to check whether a fit STORED the sharpness, since PSNR will "
                          "not tell you.",
                          example="atoms, img = mind.splat_field(grating, k=64, basis='gabor'); hf = mind.spectral_detail(img)",
                          native=True, aliases=("gabor splat", "gabor atom", "frequency lifted splat",
                                                "oriented splat", "fit a grating", "fit a texture with splats",
                                                "bandpass primitive", "recover high frequency detail",
                                                "does my fit store the sharpness"))
    c.register_capability("Blend M shader variants into one transfer", "an LOD stack, a multi-scale filter, an "
                          "MIS-weighted combination, a parameter sweep you intend to average -- any FIXED linear "
                          "combination of compiled pipelines is itself linear and shift-invariant, so the transfers "
                          "just add. mind.shader_combine(pipes, weights) returns one Pipeline; the cost does not "
                          "depend on M (measured exact to 2.2e-16, and 4.3x / 9.3x / 30.0x faster at M = 4 / 16 / "
                          "64 than staging the variants and blending their images). KEPT NEGATIVE: superposing the "
                          "variants under distinct keys so you can unbind any one back out does NOT work -- "
                          "unbinding recovers a variant at 1/sqrt(M), real variants are correlated copies of one "
                          "field so cleanup cannot resolve them, and the bank still pays M inverse transforms, so "
                          "it measured slower than the direct path. Superposition buys width only when items are "
                          "near-orthogonal AND a cleanup follows the readout.",
                          example="from holographic.rendering.holographic_shader import Pipeline, gauss_kernel\n"
                                  "pipes = [Pipeline(img.shape).blur(gauss_kernel(len(img), s)) for s in (2, 6, 14)]\n"
                                  "out = mind.shader_combine(pipes, [0.5, 0.3, 0.2]).apply(img)",
                          native=True, aliases=("blend filters", "combine shader variants", "lod stack",
                                                "multi-scale filter", "parameter sweep", "average many blurs",
                                                "variant bank", "mip chain"))
    c.register_capability("Gather N lookups in one dot product (superposed gather)", "a quadrature rule, a filter "
                          "stencil or a set of light samples -- sum_j w_j f(u_j) -- compiles into ONE query vector "
                          "Q = sum_j w_j Z(u_j) before the field is ever touched. mind.gather_field(bake, Q) is then "
                          "a single dot product no matter how many taps the rule has, and it is EXACT against "
                          "running the lookups separately (measured 7e-15), because a dot product is linear. There "
                          "is NO sqrt(N/D) crosstalk wall: a gather never unbinds, so more taps make it MORE "
                          "accurate as the bake's per-point errors average down (0.053 -> 0.008 RMS, N=2 -> 512). "
                          "mind.translate_rule slides the whole rule to any offset for one bind, at a cost "
                          "independent of N. Measured 190x amortised over 200 fields with a 64-tap rule. Over an "
                          "HTTP /invoke boundary the bake and the rule are live objects that do not survive JSON, "
                          "so mind.gather_samples(xs, ys, points, weights) is the stateless one-shot twin: plain "
                          "numbers in, a plain number out, no reuse win.",
                          example="b = mind.bake_field(xs, ys); Q = mind.gather_rule(b, us, ws); v = mind.gather_field(b, Q)",
                          native=True, aliases=("gather", "quadrature rule", "filter stencil", "many lookups at once",
                                                "weighted sum of samples", "compile a stencil", "slide a stencil",
                                                "superposed gather", "interpolate from many points"))
    c.register_capability("Compile a filter graph to one pass (shader pipeline)", "chain blurs, translations, gains "
                          "and unsharp blends -- every stage is linear and shift-invariant, so the WHOLE GRAPH "
                          "collapses into ONE transfer function before any data is touched. "
                          "mind.shader_pipeline(shape).blur(k, 8).translate(3).unsharp(kw, 0.6).apply(img) costs one "
                          "FFT, one multiply, one inverse FFT no matter how many stages it has. Measured exact to "
                          "6.7e-16 against running the stages, and 6.0x faster per application. Fractional passes "
                          "and sub-sample (fractional) translations are exact -- neither has a GPU analogue.",
                          example="out = mind.shader_pipeline(img.shape).blur(k, 8).translate(3).unsharp(kw, 0.6).apply(img)",
                          native=True, aliases=("filter graph", "compose filters", "shader pipeline", "multi pass",
                                                "fuse passes", "post process chain", "unsharp", "sub-pixel shift"))
    # --- ENGINE CONTRACTS (D-3). These are not user-facing features; they are the rules a CONTRIBUTOR must cite
    # instead of hand-rolling. They were unfindable, which is exactly why four modules hand-rolled the tie-break.
    c.register_capability("Deterministic tie-break (argmax_tiebreak)", "the engine's ARGMAX CONTRACT: the index of the "
                          "maximum with ties resolved to the LOWEST index (ISA-1). The argmax IS the observable "
                          "decision (which atom is recalled), and scores are not bit-stable across backends, orders "
                          "and bucket counts -- a 1e-17 delta flips the winner. Cite this rule; never call np.argmax "
                          "directly in a decision path. Adoption is enforced by tests/test_unifier_adoption.py.",
                          example="from holographic.misc.holographic_determinism import argmax_tiebreak; idx = argmax_tiebreak(codebook @ query)",
                          native=True, aliases=("break ties deterministically", "argmax", "argmax tiebreak", "tie break",
                                                "which atom wins", "deterministic decision", "lowest index wins",
                                                "bit-exact decision", "ISA-1", "cleanup decision rule"))
    c.register_capability("Closed-form operator iteration (iterate)", "a bind operator is DIAGONAL in the Fourier "
                          "basis, so iterating it k times is one closed-form evaluation (raise the transfer to the "
                          "k-th power) and the k->infinity limit is a mask -- no loop. Measured 41x (k=64) to 1059x "
                          "(k=4096); k=1,000,000 costs the same as k=1, fractional k is well defined, and a divergent "
                          "operator RAISES instead of silently overflowing to nan. Use it instead of "
                          "`for _ in range(k): x = step(x)`.",
                          example="from holographic.misc.holographic_iterate import step_k, limit; x_k = step_k(x, U, k); x_inf = limit(x, U)",
                          native=True, aliases=("iterate a linear operator many steps", "k steps at once", "operator power",
                                                "steady state", "fixed point", "rollout k steps", "closed form iteration",
                                                "repeat a filter n times", "propagator jump", "diffusion steady state"))
    c.register_capability("Exact order-independent sum (reduce_sum_exact / rns)", "float addition is not associative, "
                          "so a distributed SUM depends on bucket order and count (measured spread 4.6e-5). Reducing "
                          "through exact integer residue arithmetic makes the result BIT-IDENTICAL across orders and "
                          "bucket counts by construction -- the 'bit-exact distributed sum is impossible' caveat is "
                          "retired. Use it wherever a reduction must be reproducible across a farm.",
                          example="from holographic.scene_and_pipeline.holographic_distribute import reduce_sum_exact; total = reduce_sum_exact(parts, bits=40)",
                          native=True, aliases=("exact distributed sum", "bit exact sum", "order independent reduction",
                                                "reproducible sum", "float associativity", "rns", "exact integer sum",
                                                "farm reduce"))
    c.register_capability("Manifold-correct normal quantization (octnormal)", "quantize a unit normal on its own "
                          "manifold (octahedral mapping) instead of packing three floats and re-normalizing, which "
                          "distorts the sphere. The canonical home for compressing normals in meshes, g-buffers, "
                          "splats and curvature.",
                          example="from holographic.mesh_and_geometry.holographic_octnormal import oct_quantize, oct_dequantize; codes = oct_quantize(normals, bits=8)",
                          native=True, aliases=("quantize a unit normal", "compress normals", "normal packing",
                                                "octahedral normal", "unit vector quantization", "gbuffer normals"))
    c.register_capability("Distributed coordinator", "run monoid work (partition -> worker -> shared read-only cache "
                          "-> reduce) on a pluggable BACKEND: an in-process default or a persistent local process pool "
                          "(ProcessPoolExecutor + shared_memory, cache shipped ONCE, workers in separate interpreters). "
                          "Sits behind distribute; includes a margin-gated canonical tie-break so distributed results "
                          "agree on knife-edge decisions",
                          example="from holographic.scene_and_pipeline.holographic_coordinator import Coordinator, LocalPool; Coordinator(LocalPool(4)).run(buckets, worker, cache, reduce)",
                          native=True, aliases=("coordinator", "distribute compute", "process pool", "parallel",
                                                "render farm", "offload", "shared memory", "backend", "tie-break",
                                                "local pool", "worker pool", "monoid reduce"))
    c.register_capability("Graph traversal (exact)", "reachability over a table\'s edges -- neighbors, descendants, "
                          "reachable, shortest path -- what recursive SQL CTEs make painful. Uses an EXACT adjacency "
                          "index by design: the holographic graph store\'s recall collapses at scale, so traversal is "
                          "a plain deterministic graph (tombstone-aware, directed or undirected)",
                          example="from holographic.agents_and_reasoning.holographic_querygraph import EdgeGraph; EdgeGraph(t,'src','dst').path(a,b)",
                          native=True, aliases=("graph", "reachable", "descendants", "shortest path", "traversal",
                                                "adjacency", "recursive cte", "edges", "network"))
    c.register_capability("Single-writer concurrency", "B8 concurrency: one writer at a time (serialised by an "
                          "exclusive lock; a second writer waits or fails fast) plus lock-free reader SNAPSHOTS (a "
                          "consistent point-in-time view immune to later writes). MVCC deferred, stated honestly",
                          example="from holographic.agents_and_reasoning.holographic_querylock import SingleWriterLock; with lock.write(): ...",
                          native=True, aliases=("lock", "single writer", "concurrency", "snapshot read", "writer lock",
                                                "isolation", "consistent read"))
    c.register_capability("Workspace folders", "a shallow grouping tree over a database\'s tables (database > folder "
                          "> table): each table has one HOME folder (ownership -> lifecycle/tier) plus any number of "
                          "ASSOCIATION links (grouping, no deletion on unlink). Scoped search runs over just a "
                          "subtree. Folders reference existing tables, they do not copy them",
                          example="from holographic.agents_and_reasoning.holographic_queryfolder import FolderTree; ft.set_home('user.sales','reports'); ft.tables_in('reports')",
                          native=True, aliases=("folder", "group tables", "namespace tree", "organize tables",
                                                "home folder", "association folder", "scoped search", "drill down"))
    c.register_capability("VSA programs as DB objects", "installable, runnable 'stored procedures' that are "
                          "hypervectors the machine executes (LOAD/BIND/APPLY/HALT -- not arbitrary code): install, "
                          "list a queryable catalog, find a program BY MEANING (fuzzy over its doc), EXPLAIN (dry "
                          "run), and EXECUTE over query rows sandboxed to whitelisted handlers + step-bounded, result "
                          "carrying a calibrated confidence. Safer than a SQL stored procedure",
                          example="from holographic.agents_and_reasoning.holographic_queryprog import ProgramCatalog; cat.install(...); cat.find('cluster a series')",
                          native=True, aliases=("stored procedure", "install program", "execute program", "udf",
                                                "pg_proc", "find program", "run program", "vsa program", "program catalog"))
    c.register_capability("Query time-travel & audit", "git-for-data on a query table: SELECT as-of a past version "
                          "(time travel), blame a row across versions, diff two versions (added/removed/changed with "
                          "field detail), revert, branch/compare/discard, and prove/locate-tampering (Merkle root + "
                          "O(log n) which-row-changed). Wires the shipped versioning faculties into the query layer",
                          example="from holographic.agents_and_reasoning.holographic_querytime import TableHistory, select_as_of, diff_versions, prove",
                          native=True, aliases=("time travel", "as of", "temporal", "blame", "diff versions", "revert",
                                                "branch", "git for data", "tamper", "audit", "version history", "undo"))
    c.register_capability("Workspaces (durable DB + transient sessions)", "WS3-WS6: run one persistent user database "
                          "alongside many TRANSIENT per-session workspaces (loose scratch tables + the 3D/sim/render "
                          "context) that stay isolated -- clearing or resetting one never touches the persistent DB or "
                          "a sibling. Make / switch / clear / reset-keeping-data, export/import a workspace, and combine "
                          "two with an EXPLICIT collision policy (a merge is a decision, not a guess)",
                          example="from holographic.scene_and_pipeline.holographic_workspace import WorkspaceManager; m=WorkspaceManager(); m.new_workspace('sessionA'); m.switch_workspace('sessionA')",
                          native=True, aliases=("workspace", "session", "scratch tables", "transient tables", "isolate "
                                                "session", "reset keep data", "export workspace", "combine workspaces",
                                                "per-session", "sandbox tables"))
    c.register_capability("Durability & crash recovery", "B7: make the query store survive a crash. Take a durable "
                          "SNAPSHOT of the persistent tiers (replay-based, so it rebuilds byte-identically), keep a "
                          "write-ahead JOURNAL of inserts/updates/deletes since the snapshot, and RECOVER to the last "
                          "consistent point by loading the snapshot and replaying the journal. The snapshot+WAL "
                          "discipline, on top of the plain save/load the service already exposes",
                          example="from holographic.agents_and_reasoning.holographic_query_durable import save_snapshot, Journal, recover; recover(snap_path, journal_path)",
                          native=True, aliases=("durability", "crash recovery", "journal", "write ahead log", "wal",
                                                "snapshot recover", "point in time recovery", "replay journal", "recover"))
    c.register_capability("Splat aniso-refine (re-enable)", "full-3DGS anisotropic refinement composed coarse-first: "
                          "fit cheap isotropic splats, then gradient-refine the RESIDUAL (what iso missed -- sharp / "
                          "oriented features) with anisotropic Gaussians. Strictly >= the isotropic baseline (no harm "
                          "mode); big win on sharp edges. Opt-in (no reliable cheap detector for WHEN it pays)",
                          example="from holographic.rendering.holographic_splat import fit_coarse_first; fit_coarse_first(target, K_iso, K_aniso)",
                          native=True, aliases=("splat refine", "anisotropic splat", "3dgs", "gaussian splat", "coarse "
                                                "first splat", "aniso fit", "residual refine", "gradient refine"))
    c.register_capability("Nystrom kernel (re-enable)", "apply a kernel-weighted field in O(N*m) instead of exact "
                          "O(N^2), gated by a low-rank probe: if a cheap held-out probe says the kernel is low-rank "
                          "(smooth) use Nystrom (measured 6-14x faster, near-exact), else fall back to exact. The "
                          "exact fallback is always correct, so the gate can't be wrong",
                          example="from holographic.sampling_and_signal.holographic_nystrom import apply_kernel_gated; apply_kernel_gated(points, sources, weights, sigma)",
                          native=True, aliases=("nystrom", "landmark", "low rank", "kernel", "rbf field", "large field",
                                                "spectral embedding", "o(n^2)", "smooth field"))
    c.register_capability("Lossless set-packing for image families", "single-file codecs compress every image on "
                          "its own, so a SET that shares structure (a logo suite, sprite variants, UI frames, "
                          "scanned pages) pays for the shared part in every file. mind.pack_images(images) stores "
                          "ONE reference plus per-image deltas, zlib-coded; mind.unpack_images(blob) returns them "
                          "byte for byte (the residual is mod 256, so the round trip is bit-exact). Measured on a "
                          "6-logo suite: 1,744 B against 3,553 B of per-file PNG and 3,162 B of gzip-the-whole-set. "
                          "KEPT NEGATIVE, loud: it LOSES by 16x on content that is already compressible on its own "
                          "(smooth gradients, photographs) -- 32,274 B against 1,987 B. It is content-dependent, so "
                          "mind.pack_benchmark(images) prints the table. Run it; do not guess.",
                          example="blob = mind.pack_images(logos); back = mind.unpack_images(blob)   # bit-exact",
                          native=True, aliases=("pack images", "compress a set of images", "delta compression",
                                                "sprite sheet compression", "store the diff not the frame",
                                                "image family", "lossless set packer"))
    c.register_capability("Learned navigator (adaptive search budget)", "the creature, repurposed to search the "
                          "data tree. mind.train_navigator(items) trains an agent that reads a region, senses how "
                          "confident the answer looks, and decides arrive-or-keep-moving; mind.navigator_find(cue) "
                          "searches, fronted by a ReflexCache that recognises FAMILIAR queries instantly -- it gets "
                          "faster at whatever you ask for most. WHY: a fixed beam spends the same effort on every "
                          "query, so it must be wide enough for the hard minority and overpays on the easy majority. "
                          "MEASURED against the tree's own fixed-beam curve (the strongest baseline, not a "
                          "strawman): the navigator reaches 98.0% recall at 173 comparisons; the cheapest fixed beam "
                          "matching that recall is beam 12 at 450 (2.6x more), and at the navigator's own budget the "
                          "best fixed beam reaches only 81.6%. mind.navigator_benchmark() reproduces both readings.",
                          example="mind.train_navigator(items, queries=1500)\n"
                                  "hit = mind.navigator_find(cue)          # {'index':..., 'comparisons':...}\n"
                                  "mind.navigator_benchmark()              # recall + the fixed-beam baseline",
                          native=True, aliases=("navigator", "adaptive search", "learned search", "search a tree",
                                                "nearest neighbour search", "beam search", "spend less effort on "
                                                "easy queries", "reflex cache", "find an item by cue"))
    c.register_capability("Encyclopedia (relational knowledge)", "the third rung of the dictionary -> grammar -> "
                          "encyclopedia curriculum: a dictionary tells you what a word MEANS, an encyclopedia places "
                          "it in a web of relations. mind.encyclopedia_add(concept, is_a=, has=) teaches one concept "
                          "(key them by a sense id like 'dog.n.01' so senses do not collapse); encyclopedia_is_a is "
                          "one hop with a cleanup confidence; encyclopedia_climb walks the is_a chain as a relation "
                          "ray whose throughput DECAYS with depth on purpose (a longer deduction is less certain) "
                          "and ABSTAINS rather than emit noise; encyclopedia_is_a_transitive answers taxonomic "
                          "membership; encyclopedia_siblings and encyclopedia_relatedness give relatedness from "
                          "STRUCTURE, not word overlap -- 'dog' and 'wolf' share no letters. Relatedness is "
                          "1/(1+depth_a+depth_b) to the nearest common ancestor: identical 1.000, parent 0.500, "
                          "siblings 0.333, cousins 0.200, unrelated 0.000. The state lives on the mind, so a "
                          "long-lived service accumulates knowledge across /invoke calls.",
                          example="mind.encyclopedia_add('dog.n.01', is_a='canine.n.01', has=['tail'])\n"
                                  "mind.encyclopedia_relatedness('dog.n.01', 'wolf.n.01')   # 0.333, siblings",
                          native=True, aliases=("encyclopedia", "taxonomy", "ontology", "is a hierarchy",
                                                "how are two concepts related", "relatedness between concepts",
                                                "teach the mind a fact", "what is a dog related to",
                                                "parent concept", "concept siblings", "walk up the taxonomy",
                                                "structured knowledge about a topic", "relational knowledge"))
    c.register_capability("Run an allowlisted external command", "mind.run_command(name, args) runs an external "
                          "program that an OPERATOR put on the allowlist (ffmpeg, a solver, a shell script, an API "
                          "client), returning {stdout, stderr, returncode, ok}. It joins the same VSA fabric as an "
                          "internal faculty -- mind.command_tool wraps one as an orchestrator Tool the Planner can "
                          "select and chain, with the CircuitBreaker tripping on a flaky one. SECURITY: the "
                          "allowlist is the boundary and it is set IN PROCESS (registration is private, so it is not "
                          "reachable over /invoke -- measured: an agent could register `sh` before that was fixed). "
                          "run_command can only run a name already on the list; values fill {placeholders} one token "
                          "in one token out with NO shell, so an injection attempt in a value is a literal value.",
                          example="info = mind.run_command('probe', {'path': 'clip.mp4'})  # 'probe' registered in process",
                          native=True, aliases=("run a command", "external program", "shell out", "run ffmpeg",
                                                "call an external tool", "run a script", "job runner",
                                                "wrap a program as a tool"))
    c.register_capability("Coarse-first refine (re-enable)", "run the cheap method everywhere, measure a per-cell "
                          "residual/uncertainty, and escalate to the expensive method ONLY where it's high. "
                          "mind.refine_where_uncertain(coarse, uncertainty, refine_fn, frac=0.25). Measured on "
                          "adaptive anti-aliasing of a hard edge: 6.2x fewer samples than supersampling everywhere, "
                          "for a 21% RMSE cost -- and the same budget spent at RANDOM cells is 3x worse, so it is "
                          "the SIGNAL that pays, not the budget. TWO NECESSARY CONDITIONS: (1) the uncertainty must "
                          "be CONCENTRATED -- mind.uncertainty_concentration is the free gate, and near 0 rules "
                          "coarse-first out entirely; (2) the expensive method must be priced PER CELL, because a "
                          "greedy placement method (matching pursuit) is already adaptive and a mask tells it "
                          "nothing -- measured 21.0 dB with and without, at 0.9x the speed. THE TRAP: a GREEDY "
                          "coarse pass destroys the concentration its own refinement needs (0.416 for a uniform "
                          "base, 0.106 for a greedy one). Coarse-first wants a cheap, uniform, dumb base pass. "
                          "THE LAW BOTH CONDITIONS COLLAPSE INTO: coarse-first buys adaptivity for a method that has "
                          "NONE. RETIRED CLIENTS, each already adaptive: splat (greedy placement), volint (a closed "
                          "form -- no cells to escalate), and volume_render (empty_skip + early_term ARE coarse-"
                          "first, buying 15.2x where a residual mask buys 1.0x).",
                          example="u = mind.gradient_uncertainty(coarse)\n"
                                  "if mind.uncertainty_concentration(u) > 0.3:\n"
                                  "    fine, mask, n = mind.refine_where_uncertain(coarse, u, expensive_fn, frac=0.25)",
                          native=True, aliases=("coarse first", "coarse-to-fine", "adaptive refine",
                                                "refine where uncertain", "escalate", "adaptive sampling",
                                                "uncertainty mask", "spend compute where it matters",
                                                "is adaptive refinement worth it", "adaptive antialiasing"))
    c.register_capability("Multi-scatter BRDF (re-enable)", "energy-conserving GGX for rough metals: the Kulla-Conty "
                          "multi-scatter term adds back the energy single-scatter GGX loses (white-furnace ~0.4 -> "
                          "~1.0 at high roughness), GATED by roughness so smooth surfaces skip it (the term overshoots "
                          "at low roughness). Detector is the exact material roughness",
                          example="from holographic.rendering.holographic_brdf import brdf_gated, cook_torrance_ms; brdf_gated(N,V,L,color,metallic,roughness)",
                          native=True, aliases=("multi-scatter", "multiscatter", "kulla-conty", "energy conservation",
                                                "brdf", "ggx", "rough metal", "white furnace", "roughness"))
    c.register_capability("Adaptive record (load-gated)", "a role->filler memory that picks its representation by "
                          "LOAD and FIDELITY need -- cheap real-HRR at low load, FHRR phasors past the capacity knee, "
                          "or tensor-product binding for EXACT recall (perfect to M~dim, at dim*dim storage). Uniform "
                          "add/recall; deciders are exact integers/flags, no harm mode on recall",
                          example="from holographic.simulation_and_physics.holographic_loadmemory import AdaptiveRoleFillerMemory; m=AdaptiveRoleFillerMemory(dim, pairs, exact=True)",
                          native=True, aliases=("adaptive record", "role filler memory", "fhrr", "phasor", "tensor",
                                                "exact recall", "load", "capacity", "high load recall", "bind pairs"))
    c.register_capability("Regime gate (re-enable)", "run a superior-but-niche method ONLY in its regime, behind a "
                          "cheap conservative detector, with a safe fallback everywhere else -- the pattern for "
                          "re-enabling a shelved 'kept negative' now that adaptive dispatch can spot its regime "
                          "(e.g. closed-form iterate for linear/bind operators)",
                          example="from holographic.misc.holographic_regimegate import RegimeGate; RegimeGate(name, detect, threshold, superior, fallback)",
                          native=True, aliases=("regime gate", "re-enable", "adaptive dispatch", "gate", "detector",
                                                "niche method", "fallback", "closed form iterate"))
    c.register_capability("Hypervector (datatype)", "the first-class hypervector: a raw vector + its dim / encoder / "
                          "tag, with the five verbs (bind/unbind/bundle/cleanup/permute) as methods. Encoders are the "
                          "constructors; the raw array stays one attribute away (.array / np.asarray(hv))",
                          example="from holographic.sampling_and_signal.holographic_hypervector import Hypervector; Hypervector.encode(encoder, value).bind(other)",
                          native=True, aliases=("hypervector", "datatype", "vector", "vsa", "hdvector", "symbol",
                                                "bind", "bundle", "permute", "cleanup", "encode"))
    c.register_capability("Sampling", "Monte-Carlo sampling: low-discrepancy / blue-noise patterns, cosine-hemisphere "
                          "directions, MIS weighting, firefly-clamped accumulation -- one home over the shipped samplers",
                          example="from holographic.sampling_and_signal.holographic_samplinghome import Sampling; Sampling.cosine_hemisphere(N, n, seed)",
                          native=True, aliases=("sample", "sampling", "blue_noise", "poisson", "quasi", "halton",
                                                "hemisphere", "mis", "jitter", "firefly", "accumulate"))

    # --- fields (audit named ~8) ---
    c.register_capability(
        "Field", "sample a scalar/vector field at points with ONE interface (field.sample(points)); the backend is "
        "chosen by cost: callable/oracle, dense grid, narrow-band sparse (spectral/FPE/region/dirty are backends too)",
        example="from holographic.misc.holographic_fieldhome import Field; Field.grid(arr, lo, hi).sample(pts)", native=True,
        aliases=("field", "grid", "volume", "density", "sdf", "sample", "voxel"))
    c.register_capability("holographic_sparsefield", "narrow-band sparse field -- cost scales with surface area, "
                          "not volume", example="from holographic.misc.holographic_sparsefield import ...", native=True,
                          aliases=("narrow", "band", "sparse", "field"))
    c.register_capability("holographic_fpefield", "fractional-power-encoded N-D field (surface as one hypervector)",
                          example="from holographic.sampling_and_signal.holographic_fpefield import ...", native=True, aliases=("fpe", "field", "continuous"))

    # --- scale / compute / the kernel verbs ---
    c.register_capability("holographic_distribute", "scale out a commutative-monoid computation: partition into "
                          "buckets, run independently, reduce (sum/min/max/bundle)", example="from holographic.scene_and_pipeline.holographic_distribute import partition, reduce_sum, reduce_min, reduce_bundle",
                          native=True, aliases=("scale", "parallel", "partition", "mapreduce", "distribute", "raid"))
    c.register_capability("holographic_fuse", "fuse a bind chain into ~2 FFTs with no Python between ops (stay "
                          "VSA-native)", example="from holographic.misc.holographic_fuse import fuse", native=True,
                          aliases=("fuse", "native", "fft", "chain", "compute"))
    c.register_capability("kernel verbs", "the five primitives: bind (attach/transform), unbind (query), bundle "
                          "(superpose/blend), permute (order), cleanup (recognise/denoise)",
                          example="from holographic.agents_and_reasoning.holographic_ai import bind, bundle; from holographic.agents_and_reasoning.holographic_ai import Vocabulary  # Vocabulary(...).cleanup(x)", native=True,
                          aliases=("bind", "unbind", "bundle", "cleanup", "permute", "superpose", "blend"))

    # --- the catalog itself ---
    c.register_capability("holographic_catalog", "THIS catalog: search the engine's own capabilities before building "
                          "a duplicate (register_capability / find_capability)", example="find_capability('search vectors')",
                          native=False, aliases=("catalog", "capability", "registry", "find", "discover", "duplicate"))

    # --- the pipeline (consolidation R1): the one entry point that composes a render/sim run ---
    c.register_capability(
        "Pipeline (render/sim)", "compose a render or sim run as ordered stages that declare what they need/produce; "
        "dispatch among render strategies (pathtrace/raymarch/prt/radiance) and catch a missing input before running",
        example="from holographic.scene_and_pipeline.holographic_pipeline import build_pipeline, PipelineConfig, RenderSpec", native=False,
        aliases=("pipeline", "stage", "compose", "run", "render", "strategy", "dispatch", "route"))

    # --- top-level DOMAIN pipelines: one findable pointer per subsystem, so no whole domain is buried ---
    c.register_capability("Lighting (domain)", "one home for lighting: the light types (point/directional/spot/area/"
                          "dome/IES) and the shade INTEGRAL in each mode -- direct NEE, PRT relight, environment SH; "
                          "render methods call it", example="from holographic.rendering.holographic_lightinghome import Lighting, RectLight",
                          native=True, aliases=("lighting", "light", "lamp", "shadow", "dome", "area", "ies", "spot",
                                                "nee", "direct", "prt", "irradiance"))
    c.register_capability("Shadow / visibility (domain)", "test whether light or the environment reaches a point: "
                          "SDF soft shadow (Quilez penumbra), ambient occlusion, hard shadow-ray (NEE), and PRT baked "
                          "visibility -- one home of strategies render paths call",
                          example="from holographic.rendering.holographic_shadowhome import Shadow; Shadow.soft(sdf, P, Ldir)",
                          native=True, aliases=("shadow", "visibility", "occlusion", "ambient occlusion", "penumbra",
                                                "shadow ray", "soft shadow", "unoccluded"))
    c.register_capability("Geometry (domain)", "build and edit shapes three ways: explicit MESH (half-edge + verbs), "
                          "implicit SDF (CSG + raymarch), and SPLATS (Gaussian clouds) -- convertible via meshbridge",
                          example="from holographic.mesh_and_geometry.holographic_mesh import Mesh; from holographic.mesh_and_geometry.holographic_sdf import box, sphere",
                          native=True, aliases=("geometry", "mesh", "sdf", "splat", "shape", "model", "csg", "subdivide"))
    c.register_capability("Texture (domain)", "procedural + example-based surface detail as FIELDS you plug into a "
                          "Material channel: fbm noise, Voronoi/cellular cracks, divergence-free curl, patch synthesis; "
                          "plus the weathering set (burn/oxidation/inclusions)",
                          example="from holographic.materials_and_texture.holographic_texturehome import Texture; Param(field=Texture.voronoi(kind='edge'))",
                          native=True, aliases=("texture", "noise", "fbm", "voronoi", "curl", "procedural", "weathering",
                                                "pattern", "detail", "cellular"))
    c.register_capability("Texture graph (composable maps)", "build a texture as a TREE of maps: an op "
                          "(mix/multiply/over/scale/remap/...) over TYPED inputs -- map | color | field | number -- each of "
                          "which may be another map, so graphs nest to any depth. Sampling walks the tree; the input types "
                          "are checked at COMPOSE time so a bad graph (a colour used as a weight, a missing input) is refused "
                          "up front, not rendered wrong. Encode a graph to a hypervector to cache/search it. CMP1",
                          example="mind.texture_op('mix', a=mind.texture_leaf(value=[1,0,0]), b=mind.texture_leaf(value=[0,0,1]), t=mind.texture_leaf('fbm', n_dims=2)); mind.sample_texture(g, [0.3,0.7])",
                          native=True, aliases=("texture graph", "map graph", "shader graph", "compose texture",
                                                "layered texture", "node graph", "blend maps", "mix textures", "procedural graph"))
    c.register_capability("Simulation (domain)", "a shared STEP LOOP over any solver (fluids/smoke, fire/combustion, "
                          "softbody/cloth, hair, MPM, collision, reaction-diffusion) -- each keeps its own math; the "
                          "scaffold gives them one step(dt) and exposes their field for the Pipeline to render. "
                          "mind.simulation(solver, step_fn, field_fn) wraps ANY solver in process; "
                          "mind.run_simulation(kind, steps) is the stateless twin for /invoke -- build a known "
                          "solver ('fluid' or 'automaton'), run it, and return its field grid as plain JSON (the "
                          "live wrapper holds a solver+adapter that does not survive serialization).",
                          example="grid = mind.run_simulation('fluid', 30)   # step a fresh fluid and return its density",
                          native=True, aliases=("simulation", "solver", "fluid", "smoke", "fire", "cloth", "softbody",
                                                "step", "advance", "sim loop", "mpm", "reaction diffusion",
                                                "particle system", "particles", "emitter", "mass spring", "spring",
                                                "rigid body", "collision"))
    c.register_capability("Encoders (number to vector)", "turn raw values into hypervectors: scalar & fractional-power "
                          "encoding (encoders/fpe -- nearby numbers map to nearby vectors), N-D coordinate fields "
                          "(fpefield), complex-phasor FHRR (fhrr), sparse block codes (sbc), geometric-algebra Clifford "
                          "(clifford), and exact integer arithmetic over phasors (rns). How data ENTERS the substrate",
                          example="from holographic.io_and_interop.holographic_encoders import ScalarEncoder; from holographic.sampling_and_signal.holographic_fpe import ...",
                          native=True, aliases=("encode", "encoder", "number to vector", "scalar encoding",
                                                "fractional power encoding", "fpe", "encode coordinates", "phasor", "fhrr",
                                                "sparse block codes", "sbc", "clifford", "geometric algebra",
                                                "exact integer arithmetic", "rns", "embed a value"))
    c.register_capability("Physics & chemistry (domain)", "physical/chemical PROPERTIES and their evolution: the matter "
                          "model (Mixture/matter_step: smoke->oil separation), diffusion, equilibrium propagation, "
                          "thin-film iridescence, oxidation/weathering", example="from holographic.misc.holographic_mixture import Mixture, matter_step",
                          native=True, aliases=("physics", "chemistry", "matter", "mixture", "diffusion", "material properties",
                                                "iridescence", "oxidation", "phase"))
    c.register_capability("Adaptive rendering", "the render call that picks its own methods/quality: the converging "
                          "sampler that stops per-pixel when the confidence interval is tight, and the render-method "
                          "auto-picker", example="from holographic.rendering.holographic_gbuffer import render_auto, converge_samples",
                          native=True, aliases=("adaptive", "auto", "quality", "converge", "raytracing mode", "render mode"))
    c.register_capability("Render graph (bake vs live)", "the PIPELINE composing the texture/material/scene graphs: "
                          "mind.render_graph() registers texture graphs (static or dynamic) + a CMP4 instanced scene, "
                          "then plan() shows what it will do and WHY and prepare() runs it. The adaptive decision it "
                          "adds is BAKE a static texture graph to a grid (O(1) bilinear lookup, mind.bake_texture) vs "
                          "SAMPLE it live -- baking amortises a deep graph over many hits, live avoids re-baking a "
                          "changing map every frame. Trade: memory + interpolation error. CMP5",
                          example="rg = mind.render_graph(); rg.add_texture('rust', graph, static=True).set_scene(scene); rg.plan(); prep = rg.prepare()",
                          native=True, aliases=("render graph", "bake texture", "bake vs live", "prepare scene",
                                                "resolve textures", "orchestrate render", "material lod", "precompute texture",
                                                "static texture", "render pipeline graphs"))
    c.register_capability("Preview (swatch & material ball)", "SEE what you composed: mind.preview_texture(graph) "
                          "renders a CMP1 texture graph as a flat RGB swatch, and mind.preview_material(material) "
                          "renders a material on the classic MATERIAL BALL sphere (Cook-Torrance shaded, using the "
                          "material's roughness/metallic channels) -- works on a plain Material or a CMP2/CMP3 "
                          "layered/multi material. Returns a float image in [0,1] to save/view. The missing step "
                          "between composing a texture/material and looking at it.",
                          example="img = mind.preview_texture(graph); ball = mind.preview_material(layered_material)",
                          native=True, aliases=("preview", "swatch", "material ball", "material preview", "texture preview",
                                                "see the texture", "render swatch", "thumbnail", "material sphere",
                                                "visualize texture", "visualize material", "look at the material"))
    c.register_capability("Textured object render (paint composed maps)", "paint a COMPOSED texture or material "
                          "(CMP1 graph / CMP2-3 material) onto an object and render it: "
                          "mind.render_textured(scene, {object_name: texture_graph}) marches the scene, UV-wraps each "
                          "texture onto its object (spherical map on a sphere, planar on a box), and shades with the "
                          "real Cook-Torrance BRDF + a light + a hard shadow. This is the composability stack driving "
                          "a full 3-D render, not just a swatch. Honest: textbook UV (seams), single hard light.",
                          example="tex = mind.texture_op('mix', a=mind.texture_leaf(value='orange'), b=mind.texture_leaf(value='purple'), t=mind.texture_leaf('fbm', n_dims=2)); mind.render_textured(scene, {scene.names()[0]: tex})",
                          native=True, aliases=("textured render", "paint texture on object", "wrap texture", "uv render",
                                                "texture the sphere", "composed texture render", "map onto object"))
    c.register_capability("Denoise (domain)", "clean a render or signal with one home: image SVGF (variance-guided "
                          "a-trous) or demodulated (divide albedo out), sharpen, and the signal manifold denoisers "
                          "(adaptive/manifold/codebook/trajectory)",
                          example="from holographic.rendering.holographic_denoisehome import Denoise; Denoise.image(img, N, A, D, method='svgf')",
                          native=True, aliases=("denoise", "svgf", "clean", "smooth", "nlm", "demodulate", "sharpen",
                                                "noise reduction", "restore"))
    c.register_capability("Compute (VSA-native)", "stay in the vector/frequency domain with no Python hops: FUSE a "
                          "bind/bundle/permute chain into ~2 FFTs (measure the FFT drop), the fuse-runs SCHEDULER, "
                          "width, and running logic as a VSA PROGRAM. Rule: push decisions/cleanups to the boundaries",
                          example="from holographic.misc.holographic_computehome import Compute; Compute.fuse_record(keys, values)",
                          native=True, aliases=("compute", "fuse", "fused", "schedule", "execute", "program", "machine",
                                                "fft", "chain", "collapse", "vsa native"))
    c.register_capability("Memory (cache hierarchy)", "keep the hot working set where the CPU can reach it fast: FFT "
                          "spectrum residency (skip recomputing a reused transform), batched contiguous bind (one FFT "
                          "for a whole record), tiling to fit a cache level, and the opt-in GPU / numba backends",
                          example="from holographic.simulation_and_physics.holographic_memoryhome import Memory; Memory.bind_cached(a, b, cache)",
                          native=True, aliases=("memory", "cache", "residency", "resident", "spectrum cache", "batch",
                                                "bind_batch", "backend", "gpu", "jit", "working set", "hot"))
    c.register_capability("Transform (warp)", "move / rotate / warp across representations: VSA bind (rigid) + "
                          "permute (order), 4x4 matrices (translate/scale/rotate/compose/decompose/look_at + "
                          "quaternions), clifford rotors, anisotropic steering -- one facade",
                          example="from holographic.misc.holographic_transformhome import Transform; Transform.translation(t)",
                          native=True, aliases=("transform", "warp", "rotate", "translate", "scale", "rigid", "affine",
                                                "matrix", "quaternion", "rotor", "bind", "permute", "gizmo"))
    c.register_capability("Blend (combine)", "combine things into one: bundle (superposition, weighted = soft "
                          "mixture), lerp / slerp interpolation, Frechet mean on the sphere, front-to-back alpha "
                          "composite, and dict/scene merge with a conflict policy",
                          example="from holographic.misc.holographic_blendhome import Blend; Blend.bundle(vectors, weights)",
                          native=True, aliases=("blend", "combine", "merge", "interpolate", "lerp", "slerp", "mix",
                                                "composite", "superpose", "average", "crossfade", "morph"))
    c.register_capability("Scale (distribute)", "make something bigger than one box / one pass can hold: partition a "
                          "job, run the pieces independently, reassemble with a commutative monoid -- map_reduce, "
                          "load-balanced partition, image tiles / volume bricks; strategies tiling/octree/multires/"
                          "superposed/sparsefield", example="from holographic.misc.holographic_scalehome import Scale; Scale.map_reduce(buckets, worker, reduce='sum')",
                          native=True, aliases=("scale", "distribute", "partition", "map reduce", "tile", "brick",
                                                "parallel", "shard", "chunk", "monoid", "scale out"))
    c.register_capability("Query / database (domain)", "treat VSA stores as a database: SQL over tables, similarity/"
                          "time-travel/diff, durable + concurrent + graph + history query layers",
                          example="from holographic.agents_and_reasoning.holographic_query import run_sql, UserTable", native=False,
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
    a plain-English problem can surface any module, home, or faculty in the engine.

    The engine modules live under the `holographic` package (holographic/<family>/holographic_*.py). We walk that
    whole tree -- not just one directory -- and register each module under its DOTTED import path (e.g.
    `holographic.rendering.holographic_render`), so the `import ...` example the catalog serves actually works."""
    import ast
    import os
    import glob
    if module_dir is None:
        # this file lives in holographic/caching_and_storage/, so the package root is two levels up
        here = os.path.dirname(os.path.abspath(__file__))
        pkg_root = os.path.dirname(here)                       # .../holographic
        repo_root = os.path.dirname(pkg_root)                  # repo root (parent of holographic/)
    else:
        # backward compatible: if a caller passes an explicit dir, treat it as the package root's parent
        pkg_root = module_dir
        repo_root = os.path.dirname(os.path.abspath(module_dir))
    for path in sorted(glob.glob(os.path.join(pkg_root, "**", "holographic_*.py"), recursive=True)):
        base = os.path.basename(path)[:-3]
        if base.startswith("test_"):
            continue
        if base in catalog._by_name:
            continue
        # dotted module path relative to the repo root: holographic/rendering/holographic_render.py
        #   -> holographic.rendering.holographic_render
        rel = os.path.relpath(path, repo_root)
        dotted = rel[:-3].replace(os.sep, ".")
        try:
            tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
        except (SyntaxError, OSError):
            continue
        doc = ast.get_docstring(tree) or ""
        summary = doc.strip().replace("\n", " ").split("  ")[0][:200] if doc else base
        catalog.register_capability(base, summary or base, example="import %s" % dotted, native=True,
                                    aliases=_family_of(base))
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

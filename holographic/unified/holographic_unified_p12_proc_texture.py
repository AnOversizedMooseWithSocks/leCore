"""Part 12 of UnifiedMind's faculty surface -- 107 methods, proc_texture .. recall_procedure.

NOT A STANDALONE MODULE. This is one slice of the single `UnifiedMind` class, which grew to 17.4k lines
in one file and went past the 1 MB cap an agent can read in a single pass -- so the engine could no
longer read its own central nervous system. The class is assembled from these parts by
holographic/misc/holographic_unified.py, which is still the only import path anyone uses.

Every method here is a real attribute of UnifiedMind at runtime (mixin, not delegation), so `mind.x()`,
`dir(mind)`, the doc generators and the service's tool introspection all behave exactly as before. The
bodies were moved by line range, not regenerated, so they are byte-identical to the originals.

KEPT NEGATIVE, so nobody "tidies" it: these part classes are NOT a public API and must never be
imported or subclassed directly. They carry no `__init__` and assume the state UnifiedMind.__init__
builds; instantiated alone they would fail on the first attribute access. The leading underscore on
the class name says so, and the reachability audit reads them as referenced-by-unified, not as
standalone capabilities.
"""
import numpy as np

from holographic.agents_and_reasoning.holographic_mind import UniversalEncoder, _Index
from holographic.scene_and_pipeline.holographic_organizer import SelfOrganizingMind
from holographic.misc.holographic_creature import HolographicMind
from holographic.unified import check_part


class _UnifiedPart12:

    def proc_texture(self, name, **params):
        """The standard 3D-app texture MENU as a FIELD: proc_texture('voronoi', kind='f2f1', scale=4) -> a
        callable f(P (M,3)) -> values. Menu: noise, fbm, white, voronoi (f1/f2/f2f1/cell/smooth), musgrave
        (ridged/hybrid/fbm), wave (bands/rings + distortion), marble, wood, brick, magic, checker, stripes,
        gradient, dots. One field serves 2D (texture_image), 3D volumes (texture_volume -- cloud densities),
        or any points (a mesh's surface, a Material channel). Deterministic in seed. See holographic_proctex."""
        from holographic.materials_and_texture.holographic_proctex import proc_texture
        return proc_texture(name, **params)

    def texture_image(self, name, size=256, region=((0.0, 0.0), (1.0, 1.0)), z=0.0, **params):
        """Rasterise a standard procedural texture to a 2D (size,size) image in [0,1] -- 2D texturing IS the
        3D field sampled on a plane (change z to slide through the solid texture). texture_image('marble',
        512) is the app's marble button. See holographic_proctex / proc_texture for the menu."""
        from holographic.materials_and_texture.holographic_proctex import proc_texture_image
        return proc_texture_image(name, size=size, region=region, z=z, **params)

    def texture_volume(self, name, res=48, bounds=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), **params):
        """Sample a standard procedural texture on a (res,res,res) 3D grid in [0,1] -- cloud/smoke densities
        (feed a volume render), 3D displacement, solid materials. Same menu/field as texture_image, sampled
        in the volume. See holographic_proctex."""
        from holographic.materials_and_texture.holographic_proctex import proc_texture_volume
        return proc_texture_volume(name, res=res, bounds=bounds, **params)

    def sample_image(self, image, uv, mode="bilinear", wrap="clamp"):
        """READ a texture as NUMBERS: sample a raster (H,W) or (H,W,C) at uv (M,2) in [0,1]^2 -- bilinear
        (GPU-default smooth) or nearest (exact texel reads); clamp or repeat wrapping; half-texel-centre
        convention matching GPU samplers. The texture-as-numerical-input primitive: drive any parameter from
        a painted map. Closing contract: sampling values_to_texture(v) at texel centres returns v exactly.
        See holographic_proctex.sample_image."""
        from holographic.materials_and_texture.holographic_proctex import sample_image
        return sample_image(image, uv, mode=mode, wrap=wrap)

    def image_field(self, image, scale=1.0, wrap="repeat", mode="bilinear"):
        """WRAP a raster image as a FIELD f(P (M,3)) (x/y*scale = uv, z ignored) so a PAINTED map plugs in
        anywhere the engine takes a field -- a Material channel, cloud_scene(texture=...) density via a
        custom field, a displacement source. The raster sibling of proc_texture's analytic menu.
        See holographic_proctex.image_field."""
        from holographic.materials_and_texture.holographic_proctex import image_field
        return image_field(image, scale=scale, wrap=wrap, mode=mode)

    def ramp(self, positions, values, interp="linear"):
        """A STOP RAMP (the ColorRamp node): map scalars in [0,1] through sorted stops -- values scalar or
        RGB; interp 'linear' / 'constant' (hard bands) / 'smooth' (eased); ends clamp; a stop's own position
        returns its exact value in every mode. Returns a callable t -> values; bake it with ramp_texture.
        See holographic_proctex.ramp."""
        from holographic.materials_and_texture.holographic_proctex import ramp
        return ramp(positions, values, interp=interp)

    def ramp_texture(self, positions, values, size=256, interp="linear"):
        """ASSIGN a ramp's numbers to a TEXTURE: bake the stop ramp at texel centres to a (size,) or
        (size,C) strip -- the 1-D gradient texture every 3D app ships; sample_image reads back exactly what
        the ramp says there. See holographic_proctex.ramp_texture / values_to_texture for arbitrary arrays."""
        from holographic.materials_and_texture.holographic_proctex import ramp_texture
        return ramp_texture(positions, values, size=size, interp=interp)

    def values_to_texture(self, values, normalize=False):
        """ASSIGN arbitrary NUMBERS to a texture: (N,) becomes a one-row strip, (H,W)/(H,W,C) pass through;
        normalize=True affinely maps to [0,1] for display; the default keeps values untouched because the
        round trip is the point -- sample_image(values_to_texture(v), texel_centres) == v exactly.
        See holographic_proctex.values_to_texture."""
        from holographic.materials_and_texture.holographic_proctex import values_to_texture
        return values_to_texture(values, normalize=normalize)

    def mask_refraction(self, image, mask, strength=12.0, ior=1.33, profile="lens", edge_width=None,
                        chromatic=0.0, ripple=None, seed=0):
        """Refract an image through a 2D SHAPE: the mask is read as a LENS -- the jump-flood distance
        transform gives distance-to-edge, a meniscus profile turns it into a height bump, and small-angle
        Snell displaces each pixel by -(ior-1)*strength*grad(height): distortion is automatically STRONGEST
        NEAR THE MASK EDGE and zero on the interior plateau and outside (a water droplet / glass blob over
        the image). chromatic>0 adds dispersion fringes; ripple=(amp_px, scale) adds fbm water shimmer.
        Screen-space single-interface approximation (no TIR/caustics -- true refraction is path_trace's
        dielectric); the fast 2D water/glass compositing effect. See holographic_proctex.mask_refraction."""
        from holographic.materials_and_texture.holographic_proctex import mask_refraction
        import numpy as _np
        return mask_refraction(_np.asarray(image, float), _np.asarray(mask), strength=strength, ior=ior,
                               profile=profile, edge_width=edge_width, chromatic=chromatic,
                               ripple=ripple, seed=seed)

    def make_water(self, res=128, extent=40.0, t=0.0, seed=0, preset="ocean", shaded=False, **overrides):
        """ONE CALL -> a WATER surface (Gerstner/trochoidal ocean; Fournier & Reeves 1986, Tessendorf 2001).
        Returns {height, positions, normals, bank, ...} on a res x res grid over `extent` metres; `shaded=True`
        adds a sun-shaded preview image. Presets: 'ocean' (default), 'calm', 'storm'; any gerstner_waves keyword
        overrides (wind_heading, n_waves, choppiness, ...). Animate by calling again with a different `t` and the
        SAME seed -- deterministic bank, coherent frames, dispersion kills visible looping. Height feeds
        spectral_ocean to EVOLVE under real dispersion; positions feed the height-field mesh builders. Exact
        analytic normals (no finite differences). See holographic_ocean."""
        from holographic.simulation_and_physics.holographic_ocean import make_water as _mw
        return _mw(res=res, extent=extent, t=t, seed=seed, preset=preset, shaded=shaded, **overrides)

    def render_textured(self, scene, textures, width=256, height=192, **kw):
        """Render a SemanticScene with COMPOSED textures/materials painted onto its objects -- the composability stack
        driving a FULL 3-D render, not just a swatch. `textures` maps an object NAME (from scene.names()) to a CMP1
        texture graph or a Material; the texture is UV-wrapped onto that object's surface (spherical map on a sphere,
        planar on a box) and shaded with the real Cook-Torrance BRDF + a light + a hard shadow. Objects with no entry
        keep their scene colour. Returns an (H,W,3) image in [0,1]. Honest limits: textbook UV mapping (seams/pole
        pinch), single hard light (no GI). See holographic_texturerender."""
        from holographic.rendering.holographic_texturerender import render_textured
        return render_textured(scene, textures, width=width, height=height, **kw)

    # ---- message bus + optional agent (LLM) bridge: person and agent both connected -----------------------
    def bus(self):
        """The mind's MessageBus -- one shared bus for this mind that the app, the person, and an agent all talk
        through (publish/subscribe/mailboxes/history). Created on first use. See holographic_bus."""
        if getattr(self, "_bus", None) is None:
            from holographic.misc.holographic_bus import MessageBus
            self._bus = MessageBus()
        return self._bus

    def agent_bridge(self, llm=None, name="agent"):
        """Connect an OPTIONAL agent to this mind's bus. `llm` is any callable text->reply (your wrapper around any
        model -- nothing here depends on an LLM library, so it's fully optional and leCore runs with no agent). Use
        bridge.notify_on('render.done', 'does it look right?') so the app REACHES the agent when a job finishes, and
        bridge.ask('...') to ask it directly. See holographic_agent_bridge."""
        from holographic.agents_and_reasoning.holographic_agent_bridge import AgentBridge
        return AgentBridge(self.bus(), llm=llm, name=name)

    def run_task(self, name, fn, *args, background=False, summarize=None, **kwargs):
        """Run `fn(...)` as a named task on this mind's bus: publishes '<name>.start' then '<name>.done' (with a small
        summary an agent can read) or '<name>.error'. background=True runs it off-thread and returns immediately -- the
        bus's '<name>.done' is how everyone (including a watching agent) learns it finished, so nobody has to poll. See
        holographic_agent_bridge.run_task."""
        from holographic.agents_and_reasoning.holographic_agent_bridge import run_task
        return run_task(self.bus(), name, fn, *args, background=background, summarize=summarize, **kwargs)

    # ---- OPTIONAL semantic index over the dictionary: find words by meaning -------------------------------
    def build_semantic_index(self, words=None, dim=256, seed=0, include_synonyms=True, max_words=None):
        """Build an OPT-IN semantic index over the dictionary so you can find words by MEANING (idx.find('unexpected
        good luck') -> 'serendipity'). Nothing loads until you call this -- a user who just wants the library pays
        nothing for it. `words` scopes the vocabulary (default all ~144k, ~150 MB at dim=256 -- pass a list or
        max_words to keep it small). Approximate (random indexing over glosses) -- great for the top hit, noisy in the
        tail. See holographic_word_index."""
        from holographic.caching_and_storage.holographic_word_index import build_semantic_index
        return build_semantic_index(words=words, dim=dim, seed=seed,
                                    include_synonyms=include_synonyms, max_words=max_words)

    # ---- external asset paths: track files, repair them when they move --------------------------------------
    def asset_library(self):
        """A fresh AssetLibrary for tracking the EXTERNAL files a scene depends on (textures, models, ...) and repairing
        their paths when they move: add() them, ask which are missing(), then relink(one, new_path) to re-find the rest
        by the moved-parent + structural search, changed() to spot on-disk edits, and resolve(..., roots=) to find a
        file by content hash across machines. See holographic_assets."""
        from holographic.misc.holographic_assets import AssetLibrary
        return AssetLibrary()

    def ingest_files(self, source, extract_to=None, with_hash=True, index_text=True, max_text_bytes=1_000_000):
        """Digest a FOLDER, a .zip, or a single file into a queryable FILE MAP: mind.ingest_files('project/') or
        mind.ingest_files('bundle.zip'). The returned FileMap lets you query by name (find('*.png')), kind
        (by_kind('model')), metadata, text content (search_text('shader normal')), and -- after build_meaning_index() --
        meaning (find_by_meaning('lighting')); inspect its tree(); and repair paths when files move via its built-in
        AssetLibrary (missing/changed/relink/resolve_assets). See holographic_filemap."""
        from holographic.simulation_and_physics.holographic_filemap import ingest
        return ingest(source, extract_to=extract_to, with_hash=with_hash, index_text=index_text,
                      max_text_bytes=max_text_bytes)

    # ---- cold storage: compress INACTIVE structures, inflate on demand -------------------------------------
    def cold_store(self, keep_warm=8, codec="zlib", spill_dir=None):
        """A keyed store that bounds memory: keeps at most `keep_warm` values live and compresses the rest, warming any
        of them transparently on get(). Park inactive tables/arrays/databases here. codec='lzma' packs smaller (slower);
        codec='fast' is the numeric-array fast path (byte-plane shuffle + zlib-1: measured 0.72 ratio vs zlib's 0.95
        AND ~2x faster both directions on a structured float64 field; non-arrays fall back to the pickle path);
        spill_dir writes cold blobs to disk to free RAM entirely. See holographic_coldstore."""
        from holographic.caching_and_storage.holographic_coldstore import ColdStore
        return ColdStore(keep_warm=keep_warm, codec=codec, spill_dir=spill_dir)

    def cool(self, value, codec="zlib", spill_dir=None):
        """Wrap ONE value so it can be folded up (compressed) when idle and inflated on demand: c = mind.cool(big_table);
        c.cool() frees its RAM, c.get() brings it back bit-identical. codec='fast' for numeric ndarrays
        (smaller AND faster than zlib -- the measured shuffle path). See holographic_coldstore.Cold."""
        from holographic.caching_and_storage.holographic_coldstore import Cold
        return Cold(value, codec=codec, spill_dir=spill_dir)

    # ---- import the file formats artists hand you: OBJ/MTL, glTF/GLB, texture sets, volumetric grids -------
    def import_asset(self, path):
        """Import an artist file by extension: .obj (+its .mtl) or .glb/.gltf -> a LoadedMesh (geometry + PBR
        materials + textures); a volumetric .npy/.raw -> (GridField, bounds) for render_volume. For a Substance 3D
        Painter export, point load_texture_set at the folder. See holographic_assetimport."""
        from holographic.io_and_interop.holographic_assetimport import import_asset
        return import_asset(path)

    def load_obj(self, path):
        """Load a Wavefront .obj and its .mtl into a LoadedMesh (positions, UVs, normals, per-face material, and the
        materials with their map_* textures)."""
        from holographic.io_and_interop.holographic_assetimport import load_obj
        return load_obj(path)

    def load_glb(self, path):
        """Load a .glb/.gltf into a LoadedMesh with its PBR materials (base colour / metallic-roughness / normal /
        emissive) and embedded textures."""
        from holographic.io_and_interop.holographic_assetimport import load_glb
        return load_glb(path)

    def split_by_material(self, loaded_mesh):
        """Split a LoadedMesh (from load_glb / import_asset) into one submesh PER MATERIAL -- returns an ordered
        {material_name: LoadedMesh}, each reindexed to its own compact vertex set with UVs/normals subset to match.

        WHY: a .glb import merges the whole scene into one mesh, so sampling a multi-material scan with a single
        texture paints most faces with the WRONG image. Splitting first lets each material render/LOD with its own
        texture. Delegates to LoadedMesh.split_by_material (see holographic_assetimport)."""
        return loaded_mesh.split_by_material()

    def load_texture_set(self, folder, name=None):
        """Build one PBRMaterial from a folder of maps exported by Adobe Substance 3D Painter (or any tool):
        basecolor/roughness/metallic/normal/height/ao/emissive matched by file name. Reads the exported maps (the
        .spp/.sbsar project files are proprietary)."""
        from holographic.io_and_interop.holographic_assetimport import load_texture_set
        return load_texture_set(folder, name=name)

    def load_volume(self, path, dims=None, dtype="float32", bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))):
        """Load a 3-D density grid (.npy, or raw floats + dims) into (GridField, bounds) you can hand to
        render_volume. OpenVDB .vdb is proprietary/sparse -- export a dense grid instead."""
        from holographic.io_and_interop.holographic_assetimport import load_volume
        return load_volume(path, dims=dims, dtype=dtype, bounds=bounds)

    def deform_mesh(self, loaded, clip=None, t=0.0):
        """DEFORM an imported rig at time t: morph-blend the base shape (if it has blend shapes) then apply
        linear-blend skinning (if it has a skeleton), returning a deformed Mesh with the vertices moved. Pass one of
        loaded.animations as `clip` to pose it; clip=None gives the rest pose. This is what makes a loaded glTF
        actually move. See holographic_skindeform."""
        from holographic.mesh_and_geometry.holographic_skindeform import deform
        return deform(loaded, clip=clip, t=t)

    # ---- composability: agreement/disagreement across estimates, and the refine loop -----------------------
    def opponent_channels(self, vec_a, vec_b, interrupt_threshold=0.35):
        """Decompose the disagreement between TWO estimates of the same thing (the opponent-processing decomposition,
        ported from leOS) into {agreement, a_exclusive, b_exclusive, magnitude_dispute, purple (=a_exclusive+
        b_exclusive, the emergent signal in NEITHER alone), divergence_score, cosine_similarity, interrupt}. Act on
        the agreement when divergence is small; surface the conflict when it's large. See holographic_opponent."""
        from holographic.rendering.holographic_opponent import opponent_channels
        return opponent_channels(vec_a, vec_b, interrupt_threshold=interrupt_threshold)

    def refine(self, produce, critique, adjust, accept=0.9, budget=8):
        """Produce a result, have a CRITIC (any callable -- a metric, opponent agreement, a model, a human) score it,
        adjust, and retry until accepted or the budget runs out. Returns {result, score, accepted, tries}. The
        pipeline middle: sit leCore between a big compute and a checker. See holographic_refine."""
        from holographic.misc.holographic_refine import refine
        return refine(produce, critique, adjust, accept=accept, budget=budget)

    def attach_llm(self, llm, name="agent"):
        """Attach an LLM -- ANY callable text->text (leCore imports no model SDK). Returns an AgentBridge wired to the
        mind's bus, so the LLM can receive bus events and publish replies, and is now usable as a tool or a refine
        critic. Pass None for a bus-only bridge. See holographic_agent_bridge."""
        from holographic.agents_and_reasoning.holographic_agent_bridge import AgentBridge
        self._llm = llm
        self._agent_bridge = AgentBridge(self.bus(), llm=llm, name=name)
        return self._agent_bridge

    @property
    def orchestrator(self):
        """A tool Orchestrator for this mind: register(local faculty / remote tool / callable), register_command(shell
        program, allowlisted), register_remote(base_url) to pull another node's /tools, then plan over them all in one
        shape. Built lazily. See holographic_orchestrator.Orchestrator."""
        if getattr(self, "_orchestrator", None) is None:
            from holographic.scene_and_pipeline.holographic_orchestrator import Orchestrator
            self._orchestrator = Orchestrator(dim=self.dim, seed=self.seed)
        return self._orchestrator

    @property
    def db(self):
        """A shared query Database that principals carve private namespaces out of (their isolated stores). Built
        lazily; distinct from the mind's own query path. See holographic_query.Database."""
        if getattr(self, "_principal_db", None) is None:
            from holographic.agents_and_reasoning.holographic_query import Database
            self._principal_db = Database()
        return self._principal_db

    @property
    def base(self):
        """A partition SharedMind over this mind -- the frozen shared base that principals branch private, copy-on-write
        learning overlays from (so a population of actors learns without disturbing each other). See
        holographic_partition.SharedMind."""
        if getattr(self, "_principal_base", None) is None:
            from holographic.misc.holographic_partition import SharedMind
            self._principal_base = SharedMind(self)
        return self._principal_base

    def principal(self, actor_id, workspace="default", kind="agent", overlay=False):
        """A scoped Principal identity for an actor (agent/user/service/peer), wired to this mind's db, bus, and (if
        overlay=True) a private partition overlay. Isolation is by construction: it writes only its own namespace,
        reads only its own inbox, and tags its contributions with its own provenance role. See holographic_principal."""
        from holographic.misc.holographic_principal import Principal
        base = self.base if overlay else None
        p = Principal(base, self.db, actor_id, workspace=workspace, kind=kind, dim=self.dim, seed=self.seed)
        return p.connect(self.bus())

    def merge_forks(self, forks, policy="select", tol=0.2):
        """Reconcile several forked worlds, each a {slot: vector} delta layer. Slots the forks AGREE on merge
        conflict-free (pairwise opponent divergence < tol); slots they DISAGREE on go to the policy ('select' surfaces
        them, 'auto' keeps only agreements, 'left'/'right'/callable resolve). Returns {merged, conflicts}. See
        holographic_merge."""
        from holographic.misc.holographic_merge import merge_forks
        return merge_forks(forks, policy=policy, tol=tol)

    @property
    def workspace(self):
        """A WorldSpace of named worlds (each a set of vector SLOTS) for the fork -> edit -> merge -> apply loop:
        mind.workspace.fork(name) hands out a copy-on-write editing view whose .delta feeds mind.merge_forks. Built
        lazily. (Distinct from the DB workspace tiers on the query layer.) See holographic_world.WorldSpace."""
        if getattr(self, "_worldspace", None) is None:
            from holographic.misc.holographic_world import WorldSpace
            self._worldspace = WorldSpace()
        return self._worldspace

    def apply(self, delta, world="default"):
        """Write a merged delta ({slot: vector}, e.g. from mind.merge_forks(...)["merged"]) back into a shared world.
        Returns the slots changed. Completes the fork/merge/apply loop. See holographic_world."""
        return self.workspace.apply(delta, name=world)

    @property
    def registry(self):
        """A presence Registry -- who is online. Actors announce (heartbeat); registry.list(kind=)/is_online discover
        them. Rides the mind's bus so presence is visible across nodes. Built lazily. See holographic_registry."""
        if getattr(self, "_registry", None) is None:
            from holographic.scene_and_pipeline.holographic_registry import Registry
            self._registry = Registry(bus=self.bus())
        return self._registry

    def invite(self, kind="user", grants=None, code=None):
        """Create an invite token admitting a guest as `kind` with initial READ grants (e.g. {'read': ['lab/scene']}).
        Hand the returned invite's .code to the guest; redeem it with mind.admit(code_or_invite, actor_id). Default is
        nothing shared -- a guest sees only what the invite (and later grants) allow. See holographic_access."""
        from holographic.misc.holographic_access import invite as _invite
        inv = _invite(kind=kind, grants=grants, code=code)
        self._invites = getattr(self, "_invites", {})
        self._invites[inv.code] = inv
        return inv

    def admit(self, invite_or_code, actor_id, workspace="default"):
        """Redeem an invite (an Invite or its code) to admit a guest: a scoped Principal of the invite's kind with
        EXACTLY the invite's read grants (and nothing more), connected to the bus and announced to the registry.
        Writes stay own-namespace-only. See holographic_principal / holographic_access."""
        from holographic.misc.holographic_access import apply_invite, AccessError
        invites = getattr(self, "_invites", {})
        inv = invite_or_code if hasattr(invite_or_code, "code") else invites.get(invite_or_code)
        if inv is None:
            raise AccessError("unknown invite code: %r" % invite_or_code)
        if not inv.is_valid():
            raise AccessError("invite already used: %r" % inv.code)
        p = self.principal(actor_id, workspace=workspace, kind=inv.kind)
        apply_invite(p, inv)
        self.registry.announce(p)
        return p

    def create_invite_link(self, workspace="default", base_url="http://127.0.0.1:5050/",
                           grants=None, kind="user"):
        """The INVITE BUTTON, one call: mint an invite and return a ready-to-share LINK (plus the raw code) that a
        friend pastes/opens to join this session. Wraps mind.invite -- so the guest can be admitted later with
        mind.join_from_link(link, actor_id). `grants` are the initial READ grants (default: read the workspace's
        scene, i.e. {'read': [workspace+'/scene']}); pass your own to share more or less. Returns
        {code, link, workspace, kind, grants}: `link` is base_url with ?join=<code> for the button to copy, `code`
        is the bare token for a 'type the code' box.

        WHY a wrapper: invite/admit are the low-level access primitives (kind, grants, namespaces); a front-end
        invite button should not have to assemble a URL or know the grant vocabulary. This packages the common case
        so the button is one call, while power users still have invite/admit/grant directly. Delegates -- it does
        not reimplement the access logic. See holographic_access."""
        from urllib.parse import urlencode
        if grants is None:
            grants = {"read": [workspace + "/scene"]}    # the sensible default: a guest can see the shared scene
        inv = self.invite(kind=kind, grants=grants)
        sep = "&" if ("?" in base_url) else "?"
        link = base_url + sep + urlencode({"join": inv.code})
        return {"code": inv.code, "link": link, "workspace": workspace, "kind": kind, "grants": grants}

    def join_from_link(self, link_or_code, actor_id, workspace="default"):
        """The JOIN BUTTON, one call: admit a guest from EITHER a full invite link (…?join=<code>) OR a bare code.
        Extracts the code if given a URL, then redeems it via mind.admit -- returning the scoped guest Principal
        (read-only to exactly what the invite granted; writes stay in the guest's own namespace). Raises AccessError
        on an unknown/used code, same as admit. This is the front-end counterpart to create_invite_link so the join
        box accepts whatever the user pastes. Delegates to admit. See holographic_principal / holographic_access."""
        from urllib.parse import urlparse, parse_qs
        code = link_or_code
        if isinstance(link_or_code, str) and "join=" in link_or_code:
            q = parse_qs(urlparse(link_or_code).query)
            if q.get("join"):
                code = q["join"][0]                      # pull the code out of a pasted link
        return self.admit(code, actor_id, workspace=workspace)

    def grant(self, principal, read=None):
        """Share selectively: grant a principal READ access to a namespace (or list of them). See holographic_access."""
        from holographic.misc.holographic_access import grant as _grant
        return _grant(principal, read=read)

    def revoke(self, principal, read=None):
        """Stop sharing: revoke a principal's READ access to a namespace (or list). See holographic_access."""
        from holographic.misc.holographic_access import revoke as _revoke
        return _revoke(principal, read=read)

    # ---- P10: the two "dark" modules, wired. Both were real capability with tests and ZERO engine references --
    # discoverable by nobody, called by nothing. Neither is superseded, so both get a door rather than an archive.
    def compose_from_tags(self, tag_list, dim=1024, seed=0):
        """FORWARD compositional generation: bind each object's (colour, shape, texture) tags into a composite
        vector and superpose them into a scene -- the inverse of the resonator's factor(). This is the step up
        from morphing what is stored to COMPOSING what was never stored. `tag_list` is a list of tag-dicts (one
        per object); pass a single dict for one object. Returns the scene vector. See holographic_compose."""
        from holographic.misc.holographic_compose import compose_object, compose_scene
        import holographic.scene_and_pipeline.holographic_scene as hs
        coder = hs.SceneCoder(dim=dim, seed=seed)
        if isinstance(tag_list, dict):
            return compose_object(coder, tag_list)
        return compose_scene(coder, tag_list)

    def novel_object_specs(self):
        """Every (colour, shape, texture) combination the scene coder can compose -- including the ones never
        stored. The generation space, enumerated. See holographic_compose.all_object_specs."""
        from holographic.misc.holographic_compose import all_object_specs
        return all_object_specs()

    def regime_detector(self, fast=0.5, slow=0.02, threshold=0.3, trigger=1.2):
        """A DOUBLE-DIFFUSIVE regime/layer detector, borrowed from ocean physics: a FAST component tracks the
        present while a SLOW one holds the persistent state, and when their divergence stays high the system
        commits to a new LAYER. `detector.observe(x)` -> (divergence, layer_index, started_new_layer).

        Use it to notice that a stream has genuinely CHANGED REGIME rather than merely wobbled -- the same
        question the coherence-gated reorganizer asks of its store. See holographic_diffusion.DoubleDiffusion."""
        from holographic.misc.holographic_diffusion import DoubleDiffusion
        return DoubleDiffusion(fast=fast, slow=slow, threshold=threshold, trigger=trigger)

    def denoise_tensor(self, X, sigma=None):
        """Tier 5 -- denoise MULTI-WAY data by projecting onto the low-rank Tucker manifold implied by the noise
        level. The Milanfar reframe applied to tensors: a denoiser IS a map of the manifold clean signals live on,
        and for multi-way data that manifold is 'low rank along EVERY axis at once'. Ranks are chosen by the noise
        floor (a singular value above sigma*(sqrt(m)+sqrt(n)) cannot be noise), and sigma is estimated from the data
        if not given. Returns (clean, ranks, sigma).

        MEASURED on a real diffusing field: 31.5 dB noisy -> 48.6 dB, where a per-slice SVD denoiser (blind to the
        correlation ACROSS slices) reaches 39.5 -- about +7 dB, at every noise level tried.

        KEPT NEGATIVE: a low-rank prior is a CLAIM about the signal. On a FULL-RANK signal it destroys the data
        (43.1 dB -> 17.1 dB). Call mind.compress_tensor / rank_gate first: (near-)full ranks mean this is the wrong
        map. See holographic_tucker.tucker_denoise."""
        from holographic.caching_and_storage.holographic_tucker import tucker_denoise
        return tucker_denoise(X, sigma=sigma)

    def shader_pipeline(self, shape):
        """H1 -- a filter GRAPH compiled to ONE transfer before any data is touched. Every stage (blur, translate,
        gain, unsharp blend) is linear and shift-invariant, hence a multiplication in Fourier, so the whole graph
        collapses into a single operator:

            out = mind.shader_pipeline(img.shape).blur(k, 8).translate(3).unsharp(k_wide, 0.6).apply(img)

        A GPU runs three passes over the image; this runs one multiply. Measured on a 3-stage graph: exact to
        6.7e-16 against the staged computation, 34 us per application against 205 us (6.0x) -- and the compiled cost
        does not depend on the number of stages at all. `translate` accepts fractional (sub-sample) shifts, and
        `blur` a fractional number of passes.

        .apply() takes a 2-D field OR an RGB image (H,W,3): a channel axis is BATCHED (transfer applied to all
        channels in one call, exactly equal to a per-channel loop), so RGB callers stop looping (~1.4x on CPU at
        1080p from the shared FFT plan). The pipeline is pure FFT algebra, so it inherits use_gpu FOR FREE: with the
        GPU backend on, apply runs on the device (throughput path, matches numpy to a tolerance -- not the bit-exact
        CPU guarantee). GPU off (default) is byte-identical. See holographic_shader.Pipeline."""
        from holographic.rendering.holographic_shader import Pipeline
        return Pipeline(shape)

    def shader_combine(self, pipelines, weights=None):
        """H7 -- blend M compiled shader variants into ONE transfer, exactly. An LOD stack, a multi-scale filter, an
        MIS-weighted combination, a parameter sweep you intend to average: any FIXED linear combination of pipelines
        is itself linear and shift-invariant, so the transfers just add. Returns a Pipeline you can keep chaining.

        Measured against staging the variants and blending their images: identical to 2.2e-16, and the cost does not
        depend on M -- 4.3x at M=4, 9.3x at M=16, 30.0x at M=64.

        KEPT NEGATIVE, so nobody rebuilds it: the OTHER half of H7 -- superposing M variants under distinct keys so
        you can unbind any one back out -- does not work. Unbinding recovers a variant at 1/sqrt(M) (measured .712 /
        .353 / .177 at M = 2 / 8 / 32, matching 1/sqrt(M) to three digits), not at the 1 - sqrt(M/D) the plan
        assumed; real variants are filtered copies of one field and so are strongly correlated (mean |cos| 0.487 at
        M=2), which defeats the cleanup that normally resets crosstalk; and the bank still pays M inverse transforms,
        so it measured SLOWER (0.83-0.87x) than just applying the variants. Superposition buys width only when the
        items are near-orthogonal and a cleanup follows the readout. See holographic_shader.combine."""
        from holographic.rendering.holographic_shader import combine
        return combine(pipelines, weights)

    def bake_field(self, xs, ys, dim=4096, seed=0, margin=1.5, bandwidth=None, detrend=False):
        """H3 -- bake a sampled function into ONE hypervector ("the texture unit"): F = sum f(x_i) Z(x_i). Fetch any
        point afterwards with mind.fetch_field(bake, x) -- a single dot product, at any x, sampled or not.

        THE ALGEBRA HAS A NYQUIST. The phasor bandwidth B decides the finest detail the code can hold, and below the
        signal's maximum angular frequency the bake does not blur -- it returns a confident, smooth-looking, WRONG
        answer, and raises nothing. So B is chosen FROM THE DATA (margin * omega_max, via the bandwidth probe).
        Measured: with B set this way a fetch lands within 0.06 RMS at every frequency tried, where B = 0.5*omega_max
        gives 0.09-0.30. Supplying your own B below omega_max warns. See holographic_shader.

        H4 -- `detrend=True` subtracts the line joining the endpoints, bakes the RESIDUAL, and restores the line
        analytically. Turn it on for anything that is not already periodic. The probe is an FFT, so it treats the
        samples as wrapping: a function whose endpoints disagree carries an implicit JUMP, and a jump has an
        unbounded spectrum. A STRAIGHT LINE therefore probes at 607.95 where sqrt probes at 789.70 and an actual
        2-cycle sine probes at 12.53. It was never the singularity; it was the wrap.

        Measured absolute relative error, mean +- sd over 12 encoder seeds (D=4096, 400 samples), plain vs
        detrended: sqrt 0.1105 +- 0.0379 -> 0.0087 +- 0.0051 (12.6x); cube root 0.1404 -> 0.0170 (8.3x); f(x)=x
        0.1330 -> exactly 0.0000. It costs nothing when the endpoints already agree (a periodic sine is unchanged).
        The plain bake is not merely worse, it is UNSTABLE -- an inflated bandwidth collapses the kernel toward a
        delta, so the fetch returns whichever sample lands nearest, and 1/(x+0.05) scores 1.83 +- 4.25. A detrended
        bake must be read with normalize=True.
        """
        from holographic.rendering.holographic_shader import bake_1d
        return bake_1d(xs, ys, dim=dim, seed=seed, margin=margin, bandwidth=bandwidth, detrend=detrend)

    def fetch_field(self, bake, x, normalize=False):
        """Query a baked field at any point: one dot product. See holographic_shader.fetch.

        `normalize=True` divides by the bake's density field, returning the kernel AVERAGE rather than the kernel
        SUM. Use it whenever the samples were not uniformly spaced -- the raw fetch's gain is the local sample
        density, and reading it as f(x) is off by a factor nobody wrote down (measured: raw fetch on 3:1 clumped
        samples lands at 1.229 RMS, worse than predicting the mean; normalized, 0.283, with nothing fitted)."""
        from holographic.rendering.holographic_shader import fetch
        return fetch(bake, x, normalize=normalize)

    def gather_rule(self, bake, points, weights=None):
        """H2 -- compile N weighted lookups into ONE query vector: Q = sum_j w_j Z(u_j). A quadrature rule, a filter
        stencil, a set of light samples: whatever the rule, it becomes a single hypervector, and applying it to a
        baked field is then one dot product forever (mind.gather_field). See holographic_shader.gather_rule.

        Compiling costs the same N encodings the naive path pays, so this pays on REUSE: measured on a 64-tap rule
        against 200 baked fields, naive 2,806 ms vs 0.48 ms of dot products -- 190x amortised including the compile."""
        from holographic.rendering.holographic_shader import gather_rule
        return gather_rule(bake, points, weights)

    def gather_field(self, bake, rule, normalize=False):
        """H2 -- apply a compiled rule to a baked field: ONE dot product, whatever N was.

        EXACT, not approximate: <F, sum_j w_j Z(u_j)> is sum_j w_j <F, Z(u_j)> by linearity (measured to 7e-15). And
        there is NO sqrt(N/D) crosstalk wall here -- a gather never unbinds, so more taps make it MORE accurate, the
        bake's independent per-point errors averaging each other down (measured 0.053 -> 0.008 RMS from N=2 to 512).
        The crosstalk law governs cleanup-gated recall (holographic_superposed), not superposition you only ever sum.
        See holographic_shader.gather."""
        from holographic.rendering.holographic_shader import gather
        return gather(bake, rule, normalize=normalize)

    def bake_field_nd(self, grids, values, dim=8192, seed=0, margin=1.5):
        """H5 -- the texture unit in N dimensions: a gridded function baked into ONE hypervector, with the per-axis
        bandwidths probed FROM THE DATA. Read it back at any point with mind.fetch_field_nd(bake, point).

        The underlying n-D encoder defaults to bandwidth 3.0 on every axis, which measures at ~1.00 scale-free RMS
        on a 2-D sine -- literally no information, and it raises nothing. Probed from the data, it lands near 0.12.

        There is NO capacity budget on how many points you bundle (a bundled function is only ever summed, never
        unbound): at a fixed bandwidth the error is flat as the grid goes 400 -> 6,400 points (0.098 -> 0.118).

        BANDWIDTH IS A BIAS-VARIANCE DIAL AND `dim` IS THE VARIANCE BUDGET. The causal variable is the bandwidth
        B = margin * w_max, not `margin`, so the same margin is a different kernel on different data. On a 1-cycle
        sine, margin 1.5 (B = 9.4) is BIAS-limited: sixteen times the dimension buys NOTHING (scale-free RMS 0.1179
        at D=4,096 against 0.1191 at D=65,536). At B = 18.8 the same signal is VARIANCE-limited and D pays (0.122 ->
        0.043), as does a 2-cycle sine at that same B. THE DIAGNOSTIC COSTS ONE EXTRA BAKE: double `dim`. If the
        error drops you are variance-limited, so keep spending dimension; if it does not move you are bias-limited,
        so raise the margin instead. You cannot buy your way out of a bad bandwidth with dimension.

        KEPT NEGATIVE: at the default margin this is a SHAPE estimator, not a calibrated one (amplitude gain 0.66).
        Read shape, not amplitude, unless you raised margin and dim together. See holographic_shader.bake_nd."""
        from holographic.rendering.holographic_shader import bake_nd
        return bake_nd(grids, values, dim=dim, seed=seed, margin=margin)

    def fetch_field_nd(self, bake, point, normalize=True):
        """H5 -- query an n-D baked field at any point: one dot product (two, normalized).
        See holographic_shader.fetch_nd. Read shape, not amplitude, unless you raised the margin."""
        from holographic.rendering.holographic_shader import fetch_nd
        return fetch_nd(bake, point, normalize=normalize)

    def gather_samples(self, xs, ys, points, weights=None, dim=4096, seed=0, margin=1.5, bandwidth=None,
                       normalize=True):
        """H2 -- bake, compile a rule and gather in ONE call, from plain numbers to a plain number.

        mind.bake_field / mind.gather_rule hand back live objects (an encoder, a hypervector), which serialise to
        dead dictionaries across an HTTP /invoke boundary. This is the stateless twin: same math, no handles,
        callable with nothing but JSON. It re-bakes every call, so it buys none of the reuse win -- use the
        bake+rule pair in-process when you have more than one query. See holographic_shader.gather_samples."""
        from holographic.rendering.holographic_shader import gather_samples
        return gather_samples(xs, ys, points, weights=weights, dim=dim, seed=seed, margin=margin,
                              bandwidth=bandwidth, normalize=normalize)

    def translate_rule(self, bake, rule, dx):
        """H2 -- slide an entire compiled gather rule by `dx`, for ONE bind, at a cost independent of N. The encoder
        is a fractional power encoding, so binding Z(dx) translates every tap of the superposition at once (measured:
        cosine 1.0000000000 against re-encoding all N taps). A GPU re-fetches N taps per offset.
        See holographic_shader.translate_rule."""
        from holographic.rendering.holographic_shader import translate_rule
        return translate_rule(bake, rule, dx)

    def filter_passes(self, field, kernel, n_passes):
        """H6 -- apply N passes of a circular filter in ONE evaluation. A GPU runs the kernel N times; a bind is
        diagonal in Fourier, so N passes is just the transfer raised to the N-th power. Measured 1,824x at N=4096,
        exact to 2.3e-14, and N=1,000,000 costs the same as N=1.

        Two things a GPU structurally cannot do: N may be FRACTIONAL (half a blur pass is well defined, and two
        halves compose to one), and N may be INFINITE -- see filter_limit. See holographic_shader."""
        from holographic.rendering.holographic_shader import filter_k
        return filter_k(field, kernel, n_passes)

    def filter_limit(self, field, kernel, tol=1e-6):
        """H6 -- the N -> infinity steady state of a circular filter, in closed form: an idempotent PROJECTION onto
        the modes it does not decay. Reached in one O(D) evaluation; a literal loop can need hundreds of thousands
        of passes (measured: a 3-tap blur's slowest mode decays as 0.999849^N). Raises if the filter amplifies."""
        from holographic.rendering.holographic_shader import filter_limit
        return filter_limit(field, kernel, tol=tol)

    def tensor_structure(self, X, tol=1e-6):
        """Will a tensor factorisation pay for this array -- BEFORE you pay to find out?

        Compares the rank kept at every cut (the Schmidt rank: how many numbers must cross that boundary) to the
        most it could possibly be. Ranks far below the bound mean the cost of a cut is set by its BOUNDARY, not by
        the volume it encloses -- an AREA LAW -- and a tensor train is cheap. Ranks that saturate the bound mean
        every degree of freedom is independent -- a VOLUME LAW -- and nothing will compress it.

        Measured: a diffusing field scores saturation 0.21 (area-law) and its TT code is 4,394 B against int8's
        24,576; white noise scores 1.00 (volume-law) and its TT code is 104,782 B. The verdict predicts the byte
        outcome. Returns {ranks, bound, saturation, verdict}. See holographic_tucker.structure_verdict."""
        from holographic.caching_and_storage.holographic_tucker import structure_verdict
        return structure_verdict(X, tol=tol)

    def compress_tensor(self, X, energy=0.99, method="tucker", tol=1e-4):
        """A7/M1 -- compress MULTI-WAY data (a field over x,y,t; a frame stack; a BRDF table; a volume) by factoring
        structure out of EVERY axis at once, not one flattened SVD and not per-slice.

          method="tucker" -- HOSVD: a small dense core plus one factor per mode. Ranks chosen by the RANK GATE from
                             `energy`. Measured on a real diffusing field: 57x compression at rel-err 7.5e-3, where
                             per-slice SVD manages 5.9x -- a 9.7x gain, because per-slice sees structure WITHIN a
                             frame but none ACROSS frames.
          method="tt"     -- Tensor Train: storage LINEAR in the number of modes (Tucker's core is exponential in
                             it), error tracks `tol`.

        NEVER CP: for 3+ modes a best rank-R CP approximation may not exist at all (the set isn't closed), so this
        offers only the two SVD-based decompositions, which always exist. The gate returns FULL rank on data with no
        low-rank structure -- the honest 'store it raw' answer. Use `mind.decompress_tensor(code)` to rebuild.
        See holographic_tucker."""
        from holographic.caching_and_storage.holographic_tucker import tucker_compress, tt_compress
        if method == "tucker":
            return tucker_compress(X, energy=energy)
        if method == "tt":
            return tt_compress(X, tol=tol)
        raise ValueError("method must be 'tucker' or 'tt' (never CP -- it may have no best approximation)")

    def decompress_tensor(self, code):
        """Rebuild a tensor from mind.compress_tensor()."""
        from holographic.caching_and_storage.holographic_tucker import tucker_reconstruct, tt_reconstruct
        return tt_reconstruct(code) if "cores" in code else tucker_reconstruct(code)

    def solve_poisson_periodic(self, f, dx=1.0):
        """L5/M2 -- solve laplacian(u) = f on a PERIODIC grid in CLOSED FORM (one FFT; u_hat = f_hat / -|k|^2).
        Exact to machine precision on band-limited data (measured 6.7e-16), where an iterative stepper is still at
        1e-4 after 1000 steps. Only valid because the periodic Laplacian is a circular convolution -- see
        holographic_laplacian.is_circular."""
        from holographic.simulation_and_physics.holographic_laplacian import solve_poisson_spectral
        return solve_poisson_spectral(f, dx=dx)

    def diffuse_periodic(self, temp, alpha, t, dx=1.0):
        """L5/M2 -- evolve dT/dt = alpha*laplacian(T) on a PERIODIC grid to time `t` EXACTLY, in one evaluation:
        every Fourier mode just decays by exp(-alpha |k|^2 t). No time step, no stability limit, no substeps, and
        t = 1e-6 costs the same as t = 1e6. See holographic_laplacian.diffuse_spectral."""
        from holographic.simulation_and_physics.holographic_laplacian import diffuse_spectral
        return diffuse_spectral(temp, alpha, t, dx=dx)

    def solve_laplace(self, sdf_eval, points, boundary_value, walks=256, eps=1e-3, seed=0,
                      source=None, dirichlet_sdf=None, dim=3, max_steps=64):
        """A2 -- solve Laplace/Poisson on an SDF domain with NO MESH and no global linear system: Walk on Spheres
        (and Walk on *Stars* when `dirichlet_sdf` marks the absorbing part, making the rest zero-flux/Neumann).
        The only geometry query is distance-to-boundary, which is what an SDF returns -- so this is the solver
        leCore was already built for.

        Pointwise (evaluate only where you care), progressive (error ~ 1/sqrt(walks)), and farm-parallel with NO
        seed coordination: every random number is a pure function of position and walk index (determinism.hash_unit),
        so any node computes any walk in any order and gets the same answer. See holographic_wost."""
        from holographic.simulation_and_physics.holographic_wost import solve_laplace
        return solve_laplace(sdf_eval, points, boundary_value, walks=walks, max_steps=max_steps, eps=eps,
                             seed=seed, source=source, dirichlet_sdf=dirichlet_sdf, dim=dim)

    def farm(self, nodes, token=None, timeout=60.0, redundancy=1, attempts=1):
        """A distributed-compute Coordinator over a NetworkFarm of remote worker nodes ('host:port' each running
        serve_worker with the same worker names). farm.run(buckets, worker_name, cache, reduce) partitions the work
        across the nodes and reassembles by the monoid reducer -- same call as the local pool, just cross-machine.

        P9 -- HARDENING (the public-farm guardrail, opt-in): `attempts>1` retries a bucket that fails transiently;
        `redundancy>1` runs each bucket on several nodes and accepts the result only on AGREEMENT, which is the
        detector for an untrusted node returning a plausible-but-wrong answer. Both delegate to
        `hardening.HardenedCoordinator` via `coordinator.hardened()`; the defaults (1, 1) are the plain
        Coordinator, so a trusted pool pays nothing. See holographic_coordinator / holographic_hardening."""
        from holographic.scene_and_pipeline.holographic_coordinator import Coordinator, NetworkFarm, hardened
        backend = NetworkFarm(nodes, token=token, timeout=timeout)
        if redundancy > 1 or attempts > 1:
            return hardened(backend, redundancy=redundancy, attempts=attempts)
        return Coordinator(backend)

    def distributed_bus(self, peers=None, token=None, node_id="node"):
        """A DistributedBus: the same publish/subscribe/send bus, but publishes also fan out to peer nodes ('host:port'
        each running holographic_distbus.serve_bus), so agents on different machines share topics. Local delivery is
        the unchanged deterministic MessageBus. See holographic_distbus.DistributedBus."""
        from holographic.scene_and_pipeline.holographic_distbus import DistributedBus
        return DistributedBus(peers=peers, token=token, node_id=node_id)

    def encode_scene(self, objects):
        """Encode parsed objects into ONE composable scene hypervector: superpose bind(OBJ_i, record_i). Returns
        (scene_vector, [record_vectors], [role_atoms]). Query it back by slot with query_scene_slot -- the scene is
        content-addressable, every attribute recoverable through the superposition by cleanup."""
        from holographic.simulation_and_physics.holographic_semantic import encode_scene
        return encode_scene(objects, self)

    def query_scene_slot(self, scene_vector, roles, slot):
        """Read object `slot` back OUT of the bundled scene hypervector (unbind its role, then decode each attribute
        against its codebook). The bidirectional semantic read."""
        from holographic.simulation_and_physics.holographic_semantic import query_scene
        return query_scene(scene_vector, roles, self, slot)

    def render_scene_description(self, text, camera, width=256, height=256, post=None, quality="fast", spp=24,
                                 adaptive_spp=0, bake=None, relax=1.0):
        """The full text -> 3-D pipeline in one call: parse the description and render. quality='fast' uses the
        single-pass adaptive-AA renderer (seconds; inter-object shadows, see-through glass, volumetric fog/smoke/
        fire). quality='hyperreal' routes through the Monte-Carlo PATH TRACER with real Cook-Torrance/GGX materials
        (true global illumination, colour bleeding, emissive objects that light the scene, REFRACTIVE glass) --
        offline, spp-controlled. `adaptive_spp`>0 enables variance-driven adaptive sampling (extra samples only on
        noisy pixels). `post` is an optional holographic_postfx.PostChain. `bake`=grid-resolution precomputes the SDF to
        a grid so the shader samples it O(1) (fast on complex/animated scenes; see bake_sdf); `relax`>1 turns on opt-in
        over-relaxed marching (faster on grazing scenes, a small quality trade). The fast path always uses the free,
        bit-exact active-only marcher."""
        from holographic.simulation_and_physics.holographic_semantic import parse_description, render_scene, render_scene_pbr
        scene = parse_description(text)
        env = scene["environment"]
        sun = env.get("sun") or "bright"; sky = env.get("sky") or "clear"
        if quality == "hyperreal":
            return render_scene_pbr(scene["objects"], camera, width=width, height=height, spp=spp,
                                    post=post, sun=sun, sky=sky, adaptive_spp=adaptive_spp)
        return render_scene(scene["objects"], camera, width=width, height=height, post=post,
                            sun=sun, sky=sky, bake=bake, relax=relax)

    def scene_control_spec(self, command):
        """Turn a control phrase ('control the ball size and how metallic it is') into UI control descriptors
        (sliders / selects) a front-end can render directly -- the engine emits the spec, the browser draws the
        widgets. See holographic_semantic.control_spec."""
        from holographic.simulation_and_physics.holographic_semantic import control_spec
        return control_spec(command)

    def post_process(self, image, chain=None, depth=None):
        """Apply a post-processing PROGRAM (a holographic_postfx.PostChain -- an ordered, named chain of effects) to a
        rasterized (H,W,3) frame. `chain` defaults to postfx.default_chain(). The convolution family (bloom, glare,
        DOF, blur) rides the engine's FFT-convolution primitive (the same operator as bind, one dimension up); the
        tone/colour curves are plain NumPy in the same pipeline. Pass `depth` (the renderer's depth buffer) for DOF.
        See holographic_postfx."""
        from holographic.rendering.holographic_postfx import default_chain
        if chain is None:
            chain = default_chain()
        return chain.apply(image, depth=depth)

    def postfx_chain(self, *steps):
        """Build a post-processing PostChain program from (name, params) steps, e.g.
        postfx_chain(("exposure", {"ev": 0.3}), ("aces", {}), ("vignette", {"strength": 0.4})). With no steps,
        returns the default preset. See holographic_postfx.PostChain / default_chain / cinematic_chain."""
        from holographic.rendering.holographic_postfx import PostChain, default_chain
        if not steps:
            return default_chain()
        return PostChain(list(steps))

    def postfx_to_glsl(self, chain, name="postfx", skip_unsupported=False):
        """Compile a postfx PostChain (or a [(name, params), ...] step list) to a complete Shadertoy-style FRAGMENT
        SHADER, so a host runs the whole colour pipeline on the viewer's GPU at display rate -- live video grading in
        the browser, zero per-frame server cost. Emits the POINTWISE stages (exposure/reinhard/aces/gamma/
        color_grade/vignette) EXACTLY (matches PostChain.apply to float precision); a neighbour/blur/depth stage
        (bloom/glare/dof/...) raises unless skip_unsupported=True (multi-pass/depth is not single-pass fragment-
        emittable). Pairs with the SDF emitter (to_shadertoy) so 'leCore generates your GPU code' is an
        architecture. See holographic_postfx.chain_to_glsl."""
        from holographic.rendering.holographic_postfx import PostChain, chain_to_glsl
        steps = chain.steps if isinstance(chain, PostChain) else list(chain)
        return chain_to_glsl(steps, name=name, skip_unsupported=skip_unsupported)

    def bloom_passes(self, threshold=0.8, sigma=4.0, intensity=0.6):
        """Emit BLOOM as an ordered multi-pass GPU DAG (bright-pass -> separable blur H,V -> composite) with wiring --
        the honest multi-pass form of the effect that a single fragment pass (postfx_to_glsl) has to refuse. Returns
        {passes, targets}; the host allocates two ping-pong targets and runs the passes in order. See
        holographic_postfx.bloom_glsl_passes."""
        from holographic.rendering.holographic_postfx import bloom_glsl_passes
        return bloom_glsl_passes(threshold=threshold, sigma=sigma, intensity=intensity)

    def compose_shader(self, functions, entry=None, header=""):
        """Compose emitted GLSL function pieces (from pattern_to_glsl / cosine_palette_to_glsl / postfx_to_glsl /
        sdf map emitters) into ONE shader source, deterministically -- with a duplicate-function-name check that
        RAISES rather than silently shadow one (rename via the emitter's fn_name=). The first-class way to build a
        composed look (pattern -> palette -> grade). See holographic_emit.assemble_glsl."""
        from holographic.io_and_interop.holographic_emit import assemble_glsl
        return assemble_glsl(functions, entry=entry, header=header)

    def wrap_webgl2(self, shadertoy_src, uniforms=("sampler2D iChannel0", "vec3 iResolution"), entry="mainImage"):
        """Wrap a Shadertoy-style GLSL source (defining void <entry>(out vec4, in vec2)) into a COMPLETE WebGL2
        (GLSL ES 3.00) fragment shader -- #version + precision preamble, declared uniforms, out vec4, and the main()
        bridge. The one true wrapper, so callers stop hand-rolling a preamble that drifts. See
        holographic_emit.webgl2_wrap."""
        from holographic.io_and_interop.holographic_emit import webgl2_wrap
        return webgl2_wrap(shadertoy_src, uniforms=uniforms, entry=entry)

    def signed_distance_field_3d(self, inside_mask, h=1.0):
        """Occupancy VOLUME -> signed distance field via 3-D fast-sweeping eikonal (Numba ~230x on 96^3, bit-exact
        vs pure; pure-Python fallback). The 3-D twin of signed_distance_field for mesh import / sculpt volumes. See
        holographic_jit.signed_distance_3d."""
        from holographic.misc.holographic_jit import signed_distance_3d
        return signed_distance_3d(inside_mask, h=h)

    def distance_transform(self, seed_mask, h=1.0):
        """Distance from every cell to the nearest True seed cell, via fast sweeping (same optional-Numba path).
        See holographic_jit.distance_transform."""
        from holographic.misc.holographic_jit import distance_transform
        return distance_transform(seed_mask, h=h)

    def field_deflect(self, query, attractors, masses=None, sigma=0.5, strength=0.1):
        """Slide a query toward a local mass concentration in a field of attractors -- a soft, continuous cousin of
        cleanup (drift toward the weighted local centre of mass rather than hard-snapping to one atom). Extracted
        from leOS's gravitational lens. Returns (lensed_vector, deflection_radians, force_magnitude). See
        holographic_lens.deflect."""
        from holographic.rendering.holographic_lens import deflect
        return deflect(query, attractors, masses=masses, sigma=sigma, strength=strength)

    def detect_caustic(self, query, attractors, masses=None, sigma=0.5):
        """Routing-ambiguity (caustic) score at a query: high when the two strongest attractors pull in opposite
        directions with similar strength -- a fold/decision-boundary where a tiny move flips the winner.
        Complementary to RecallNull ('is this a match?') -- this asks 'is this AMBIGUOUS between matches?'. Returns
        (caustic_score in [0,1], n_significant_attractors). See holographic_lens.detect_caustic."""
        from holographic.rendering.holographic_lens import detect_caustic
        return detect_caustic(query, attractors, masses=masses, sigma=sigma)

    def navigate_field(self, query, attractors, masses=None, sigma=0.5, strength=0.6):
        """Climb the attractor field from a query toward an attractor (iterated, decaying-step deflection),
        reporting the strongest caustic met en route. A heuristic drift that APPROACHES an attractor, not an exact
        nearest-cluster solver. See holographic_lens.navigate."""
        from holographic.rendering.holographic_lens import navigate
        return navigate(query, attractors, masses=masses, sigma=sigma, strength=strength)

    def local_structure(self, point, cloud, k=12):
        """Classify a point by the shape of its local neighbourhood (the 'cosmic web' method extracted from leOS):
        VOID / FILAMENT (1-D thread) / WALL (2-D sheet) / NODE (dense cluster), from a local PCA of its k nearest
        neighbours, plus a continuous intrinsic_dim (participation ratio of the eigenvalues). Tells you what kind
        of structure a point lives in -- useful before denoising (project a filament point along its one
        direction), sampling (avoid voids), or summarising a cloud's geometry. holostuff already had GLOBAL
        dimension estimates (box-counting, spectral); this is the PER-POINT local type. KEPT NEGATIVE:
        high-dimensional noise inflates the apparent dimension. See holographic_cosmic.local_structure."""
        from holographic.misc.holographic_cosmic import local_structure
        return local_structure(point, cloud, k=k)

    def classify_cloud(self, cloud, k=12):
        """Classify every point of a cloud into void/filament/wall/node and return (labels, info, summary) -- the
        summary giving the fraction of each structure type and the mean intrinsic dimension, a compact geometric
        fingerprint of the cloud. See holographic_cosmic.classify_cloud."""
        from holographic.misc.holographic_cosmic import classify_cloud
        return classify_cloud(cloud, k=k)

    def frechet_mean(self, vectors, weights=None, max_iters=12):
        """The Frechet (Karcher) mean of unit vectors -- the geometrically-correct average on the sphere, the
        point minimizing the sum of squared geodesic distances (extracted from leOS's spherical geometry). This is
        the right operation for a class PROTOTYPE, a cluster centre, or a consolidation anchor -- distinct from
        `bundle`, which is a SUPERPOSITION (stays similar to every part) for binding records. Provably lower
        geodesic variance than a re-normalized Euclidean mean. HONEST: for well-separated/tight clusters its
        downstream edge over Euclidean-normalize is marginal; the geometry pays when distributions are genuinely
        spread or skewed. See holographic_sphere.frechet_mean."""
        from holographic.mesh_and_geometry.holographic_sphere import frechet_mean
        return frechet_mean(vectors, weights=weights, max_iters=max_iters)

    def parallel_transport(self, v, p, q):
        """Transport a tangent vector `v` (a 'displacement', a move from one state to another) from the tangent
        plane at `p` to the tangent plane at `q`, along the geodesic -- preserving its length and surface
        relationship (extracted from leOS). This is how a displacement measured at one point is correctly reused
        at another, which is what lets displacements be composed/compared across distant regions of the space.
        See holographic_sphere.parallel_transport."""
        from holographic.mesh_and_geometry.holographic_sphere import parallel_transport
        return parallel_transport(v, p, q)

    def use_gpu(self, enable=True):
        """User setting: turn the optional CuPy GPU backend on/off for the heavy array-parallel kernels (fluid
        solver, and any kernel that allocates via the backend). Returns whether the GPU is now ACTIVE (requested
        AND a CUDA device is present). HONEST: GPU is for throughput, not for the deterministic/tie-sensitive
        paths -- it matches NumPy only to a tolerance and can vary run-to-run, so the bit-exact guarantees are a
        CPU property. Falls back to NumPy silently when no GPU is available."""
        from holographic.misc.holographic_backend import enable_gpu
        # THE RESOURCE POLICY IS A VETO, and it is checked BEFORE touching the backend rather than after:
        # gpu='off' must mean the device is never initialised, not merely that we stop using it. An operator
        # who forbids the GPU on a machine that HAS one gets False, which is the observable that makes the
        # setting testable without owning a device.
        if enable and not self._resource_policy().gpu_allowed():
            return False
        return enable_gpu(enable)

    def gpu_crossover(self, kind="cleanup", dims=(512, 1024), counts=(256, 1024, 4096),
                      batches=(1, 8, 64, 256), repeats=5, seed=0, text=False):
        """MEASURE WHERE A DEVICE STARTS WINNING (holographic_gpubench, M1) -> {adapter, trustworthy, rows,
        crossover, note}, or the readable table with text=True.
        THE ONE NUMBER THAT BLOCKS THE COMPUTE BACKLOG. Everything else is wired and waiting on it:
        cleanup_batch(backend='wgsl') and wgsl_bind_batch exist, should_offload gates them, place_work
        composes the decision and resource_policy caps it -- but should_offload's thresholds are ARITHMETIC
        FROM PCIe BANDWIDTH, not measurements, and are marked provisional everywhere they surface. Feed
        `crossover` into MIN_BYTES_PROVISIONAL and the intensity at that point into
        MIN_INTENSITY_PROVISIONAL.
        HANDLES THE TIMING TRAP: GPU calls are ASYNCHRONOUS, so timing a dispatch without forcing completion
        measures KERNEL LAUNCH rather than execution and yields numbers that look spectacular and are wrong.
        Every device timing here READS ITS RESULT BACK, which forces completion and measures the round trip
        a real caller actually pays, transfer included -- which is the number should_offload needs.
        REFUSES TO FLATTER A SOFTWARE ADAPTER: on llvmpipe or WARP (adapter_type='CPU') the report sets
        trustworthy=False and leads with a MEANINGLESS banner, because a timing there is NumPy against a CPU
        driver emulating a GPU. `crossover: never` is a RESULT and should be published as one, not hidden.
        kind='cleanup' (codebook similarity + argmax) or 'bind' (batched circular convolution)."""
        from holographic.io_and_interop.holographic_gpubench import crossover, crossover_report
        result = crossover(kind=kind, dims=dims, counts=counts, batches=batches, repeats=repeats, seed=seed)
        return crossover_report(result) if text else result

    def gpu_report(self):
        """WHAT GPU COMPUTE IS REACHABLE, PER PATH, AND WHY NOT WHEN IT IS NOT (holographic_gpureport).
        use_gpu(True) returns a bare bool that conflates FOUR different states -- no CuPy installed, CuPy
        present but no device, a device present but the resource policy forbids it, and actually enabled --
        and three of those four are fixable by the user while one is not. This distinguishes them.
        Covers BOTH paths, because a report that only knew about CuPy would tell an Apple or AMD user they
        have no GPU, which is false: CuPy is NVIDIA-only and transparent; WGSL is vendor-neutral
        (Vulkan/Metal/DX12/WebGPU) and explicit. Also lists which modules actually route through the CuPy
        backend, DISCOVERED from the live tree rather than typed -- a hand-maintained second copy of a list
        is always the stale one. Never raises: the common case is a machine with no GPU."""
        from holographic.io_and_interop.holographic_gpureport import gpu_report
        return gpu_report(policy=self._resource_policy())

    def should_offload(self, n_bytes, flops_per_byte, round_trips=1):
        """WOULD MOVING THIS JOB TO THE GPU PAY (holographic_gpureport) -> (verdict, why). The GPU mirror of
        should_pool, same shape and same discipline. Refuses on four independent grounds: no device or the
        policy forbids one; too little DATA (the round trip dominates); too little WORK PER BYTE (an
        elementwise pass is transfer-bound by construction -- it reads and writes everything and computes
        almost nothing); or REPEATED ROUND TRIPS, where the answer is not 'offload' but 'fuse first', and
        shader_pipeline collapses N linear stages into one before any transfer.
        THE THRESHOLDS ARE PROVISIONAL AND THE VERDICT SAYS SO -- no host<->device crossover has ever been
        measured in this project, so they are arithmetic from PCIe bandwidth, not results. Replace them the
        first time this runs on a real device."""
        from holographic.io_and_interop.holographic_gpureport import should_offload
        return should_offload(n_bytes, flops_per_byte, round_trips=round_trips,
                              policy=self._resource_policy())

    def wgsl_device(self):
        """WHAT DEVICE WOULD RUN AN EMITTED WGSL KERNEL (holographic_wgpurun, WGPU-1) --
        {available, device, backend, type} or {available: False, why}. This is the VENDOR-NEUTRAL path:
        WGSL runs on Vulkan, Metal, DX12 and WebGPU, where use_gpu()'s CuPy backend is CUDA/NVIDIA ONLY.
        Reports software adapters (llvmpipe, WARP) as available too, deliberately -- that is what makes
        correctness CI-testable on a runner with no GPU."""
        from holographic.io_and_interop.holographic_wgpurun import device_info
        return device_info()

    def run_wgsl_kernel(self, fn, data, extra_args=(), workgroup=64):
        """RUN AN ANNOTATED PYTHON KERNEL ON ANY GPU via its own WGSL projection (holographic_wgpurun).
        emit_kernel already turned `fn` into WGSL; this wraps it in a @compute entry point with storage
        bindings and a bounds guard, dispatches it, and returns float32.
        SCOPE: elementwise maps over a 1-D array, f32 only (WGSL has no f64). A bounded `for range(N)` is
        fine; a CROSS-INVOCATION REDUCTION -- what bundle and cleanup need -- is not solved here.
        RAISES rather than falling back when wgpu is absent: a caller who explicitly asked for the device
        path deserves to know they did not get it. (use_gpu falls back silently, which is right for a
        transparent accelerator and wrong for an explicit request.)
        Use verify_wgsl_kernel to check the projection against the Python original on your own data --
        exactness holds for single-expression kernels and NOT for accumulating ones."""
        from holographic.io_and_interop.holographic_wgpurun import run_kernel
        from holographic.io_and_interop.holographic_emit import emit
        return run_kernel(emit(fn, "wgsl"), fn.__name__, data, extra_args=extra_args, workgroup=workgroup)

    def wgsl_reduce(self, op, data, workgroup=64):
        """REDUCE A 1-D ARRAY ON ANY GPU: 'sum' | 'max' | 'min' (holographic_wgpurun, W1).
        The primitive that unlocks the VSA half of the kernels -- run_wgsl_kernel does elementwise maps,
        which serve the rendering path and NONE of bundle / cleanup / resonator / amp / htcodebook, every one
        of which is a cross-invocation reduction.
        TWO-STAGE BY DESIGN: each workgroup reduces its slice in shared memory and writes one partial; the
        host finishes the much shorter partial array. A single-pass whole-array reduction would need a
        grid-wide barrier (WGSL has none) or atomics, which are float-nondeterministic -- the wrong side of
        this engine's determinism rule. Two stages keep the device work order-defined within a workgroup.
        NOT bit-exact with NumPy in general: the device sums in tree order and f32 throughout. Measure."""
        from holographic.io_and_interop.holographic_wgpurun import reduce_kernel
        return reduce_kernel(op, data, workgroup=workgroup)

    def wgsl_matvec(self, matrix, vector, workgroup=64):
        """MATRIX @ VECTOR ON ANY GPU (holographic_wgpurun, W2) -- ONE WORKGROUP PER ROW, so the rows never
        communicate and no cross-workgroup reduction is needed. Each lane walks its row with a stride, so D
        need not be a multiple of the workgroup size.
        THIS IS THE KERNEL THAT MATTERS FOR VSA: measured on CPU, the codebook similarity is 98-100% of a
        cleanup's cost at any real codebook size, while the argmax is single-digit microseconds -- so the
        argmax was the wrong thing to offload and this is the right one."""
        from holographic.io_and_interop.holographic_wgpurun import matvec_kernel
        return matvec_kernel(matrix, vector, workgroup=workgroup)

    def wgsl_bind_batch(self, a_stack, b_stack, workgroup=64):
        """BIND over stacks of (K, D) on any GPU (holographic_wgpurun, W3) -- the vendor-neutral batched
        circular convolution. Matches the shipped bind_batch within f32 tolerance.
        WHY DIRECT CONVOLUTION AND NOT AN FFT: bind IS a plain circular convolution (verified to 7e-15), so
        it can be done as rfft->multiply->irfft in O(D log D) or DIRECTLY in O(D^2). Direct is ~100x more
        arithmetic at D=1024 and is the right trade here -- it reuses the SAME workgroup-reduction shape as
        wgsl_matvec and wgsl_matmul (proven, tail-safe, no bit-reversal or twiddle tables), and ARITHMETIC IS
        WHAT A GPU HAS. That turned an L item into an M one.
        THE TRADE IS NOT MEASURED: whether 100x more arithmetic is recovered by parallelism depends entirely
        on the device, and this box has only a CPU adapter. Correctness is established; the crossover is not.
        If a real device shows direct convolution losing to the CPU FFT, the Stockham route is the fallback
        and its constraints are recorded (one dispatch per row, D <= 4096 fits shared memory).
        BATCHED ON PURPOSE: a SINGLE bind costs ~0.03 ms on CPU at D=1024, below any plausible dispatch
        floor -- the batch is the only shape that can pay."""
        from holographic.io_and_interop.holographic_wgpurun import bind_batch_kernel
        return bind_batch_kernel(a_stack, b_stack, workgroup=workgroup)

    def wgsl_matmul(self, matrix, queries, workgroup=64):
        """`queries @ matrix.T` on any GPU -- the BATCHED form, and the one that pays (holographic_wgpurun).
        ONE WORKGROUP PER OUTPUT ELEMENT (row, query) on a 2-D grid, a direct extension of wgsl_matvec's
        one-per-row, with still no cross-workgroup communication.
        WHY BATCHED: measured on CPU at M=1024 D=512, ONE query costs 0.095 ms -- below any plausible
        dispatch floor, so it can never pay -- while 256 queries cost 2.98 ms, comfortably above it. Building
        the single-query form first was the same mistake this path made with argmax and with a single bind:
        THE NATURAL UNIT OF WORK IS USUALLY SMALLER THAN THE DISPATCH FLOOR."""
        from holographic.io_and_interop.holographic_wgpurun import matmul_kernel
        return matmul_kernel(matrix, queries, workgroup=workgroup)

    def wgsl_cleanup_batch(self, codebook, queries, workgroup=64):
        """Cleanup a STACK of queries on any GPU -> (indices, scores) (holographic_wgpurun). The shape a
        device can actually win on; the single-query wgsl_cleanup is kept for the K=1 case and should stay on
        CPU in practice. Indices resolve host-side by lowest index, so BATCHING DOES NOT CHANGE THE CANONICAL
        TIE RULE."""
        from holographic.io_and_interop.holographic_wgpurun import cleanup_batch_kernel
        return cleanup_batch_kernel(codebook, queries, workgroup=workgroup)

    def wgsl_cleanup(self, codebook, query, workgroup=64):
        """A FULL VSA CLEANUP ON ANY GPU -> (index, score) (holographic_wgpurun, W2). Similarity and argmax
        FUSED in one dispatch: splitting them would pay the submission cost twice and ship the M-length
        intermediate back across the bus, the same 'fuse before you dispatch' rule shader_pipeline and
        should_offload's round_trips gate both encode.
        The INDEX resolves host-side by lowest index (the canonical tie rule), because an argmax is a
        DECISION and existing decisions may never flip.
        MEASURED RESIDUAL RISK: the matvec is not bit-exact, so a similarity gap at or below ~1e-7 can flip
        the decision (3/150 at 1e-7; 0/150 at 1e-6 and above). That is FOUR ORDERS below the smallest
        sensible tie margin, so pair this with tied_candidates and every flippable case arrives as a
        DECLARED AMBIGUITY rather than a silent difference."""
        from holographic.io_and_interop.holographic_wgpurun import cleanup_kernel
        return cleanup_kernel(codebook, query, workgroup=workgroup)

    def wgsl_argmax(self, data, workgroup=64):
        """DEVICE ARGMAX over a 1-D array -> (index, value) (holographic_wgpurun, W1).
        ARGMAX IS A DECISION, NOT A VALUE, so this is deliberately split: the VALUE reduction runs on the
        device, and the INDEX is resolved on the host by taking the FIRST index attaining the max. Ties
        therefore break by LOWEST INDEX -- the same canonical rule determinism.argmax_tiebreak uses -- rather
        than by whichever workgroup finished first, which is exactly the property that must not vary when
        existing decisions may never flip.
        MEASURED on adversarial exact ties (random data almost never ties, so a random-data test would prove
        nothing): 200/200 agreement with CPU argmax, and 40/40 on real VSA cleanup queries.
        A fully device-side (value, index) reduction is NOT built: it would make tie resolution depend on
        reduction order. Revisit only if the host-side finish is measured to dominate -- with a tie test."""
        from holographic.io_and_interop.holographic_wgpurun import argmax_kernel
        return argmax_kernel(data, workgroup=workgroup)

    def sdf_trace_shader(self, node, width, height, steps=96, eps=1e-3):
        """The WGSL for a per-pixel sphere trace of an SDF: the tree's own emitted map() plus an elementwise
        entry point run_wgsl_kernel can dispatch. Returned as TEXT, so it is inspectable and testable with no
        device present. `node` is a live SDF or its DSL text. See holographic_wgpurun.sdf_trace_shader."""
        from holographic.io_and_interop.holographic_wgpurun import sdf_trace_shader
        return sdf_trace_shader(node, width, height, steps=steps, eps=eps)

    def sdf_depth_device(self, node, width, height, eye=(0.0, 0.0, 3.0), fov=1.0, near=0.01, far=50.0,
                         steps=96, eps=1e-3, workgroup=64):
        """SPHERE-TRACE AN SDF ON ANY GPU -> (H,W) float32 depth, -1 where the ray missed. The bridge two
        parallel merges left open: sdf_dialect emitted WGSL that nothing dispatched, while wgpurun could
        dispatch WGSL that nothing emitted. Sphere tracing is elementwise over PIXELS, so this reuses
        run_wgsl_kernel's 1-D binding layout unchanged rather than adding a second dispatch path.
        RAISES without an adapter rather than falling back -- an explicit device request that silently ran on
        the CPU makes its own timing meaningless. Pair with sdf_depth_cpu (same rays) and sdf_depth_agrees.
        See holographic_wgpurun.sdf_depth_device."""
        from holographic.io_and_interop.holographic_wgpurun import sdf_depth_device
        return sdf_depth_device(node, width, height, eye=eye, fov=fov, near=near, far=far, steps=steps,
                                eps=eps, workgroup=workgroup)

    def sdf_trace_placement(self, width, height, steps=96):
        """WHERE SHOULD THIS SPHERE TRACE RUN -> place_work's verdict computed from the trace's OWN numbers,
        so a caller never hand-derives n_bytes/flops_per_byte (the two everyone gets wrong: bytes MOVED not
        touched, flops per BYTE not per pixel). The seam the post-merge sweep found missing -- the render arc
        never consulted the placement layer, so the one path that pays for a device could not ask.
        MEASURED: the trace presents 144 flops/byte at ANY resolution (both terms scale with pixel count)
        against a 4.0 bar, while an elementwise postfx pass presents 0.8 and is correctly refused.
        A 'cpu' verdict is a RESULT, not a failure. See holographic_wgpurun.sdf_trace_placement."""
        from holographic.io_and_interop.holographic_wgpurun import sdf_trace_placement
        return sdf_trace_placement(width, height, steps=steps, mind=self)

    def sdf_trace_workload(self, width, height, steps=96):
        """The (n_bytes, flops_per_byte) a sphere trace of this size actually presents -- the arithmetic
        behind sdf_trace_placement, exposed so the numbers can be inspected rather than trusted.
        See holographic_wgpurun.sdf_trace_workload."""
        from holographic.io_and_interop.holographic_wgpurun import sdf_trace_workload
        return sdf_trace_workload(width, height, steps=steps)

    def sdf_depth_cpu(self, node, width, height, eye=(0.0, 0.0, 3.0), fov=1.0, near=0.01, far=50.0,
                      steps=96, eps=1e-3):
        """The NumPy reference for sdf_depth_device -- same rays, same bounded march, same miss sentinel.
        The baseline that makes the device number checkable. See holographic_wgpurun.sdf_depth_cpu."""
        from holographic.io_and_interop.holographic_wgpurun import sdf_depth_cpu
        return sdf_depth_cpu(node, width, height, eye=eye, fov=fov, near=near, far=far, steps=steps, eps=eps)

    def sdf_depth_agrees(self, node, width=32, height=24, tol=2e-2, **kw):
        """Differentially test the device sphere trace against the NumPy one -> {max_abs, miss_mismatch,
        agrees, n}. Both sides trace the SAME emitted tree, so they are CHECKED rather than trusted; a
        MISS/HIT disagreement is counted apart from rounding because it is a decision.
        See holographic_wgpurun.sdf_depth_agrees."""
        from holographic.io_and_interop.holographic_wgpurun import sdf_depth_agrees
        return sdf_depth_agrees(node, width=width, height=height, tol=tol, **kw)

    def verify_wgsl_kernel(self, fn, data, extra_args=(), workgroup=64):
        """DIFFERENTIALLY TEST a kernel: run `fn` in Python AND as its own WGSL projection, report
        {max_abs, max_rel, exact, n} (holographic_wgpurun).
        THIS IS THE POINT OF THE PROJECTION DESIGN, made executable -- the shader is generated from the same
        function that runs on CPU, so the two can be CHECKED rather than trusted. A CuPy kernel cannot be
        tested this way: there is no shared source, only two implementations supposed to agree.
        MEASURED: x*g+0.5 is bit-exact; a 4-term accumulating loop deviates 5.7e-06 relative, because Python
        accumulates in float64 and casts while WGSL accumulates in f32 throughout."""
        from holographic.io_and_interop.holographic_wgpurun import verify_against_numpy
        return verify_against_numpy(fn, data, extra_args=extra_args, workgroup=workgroup)

    def import_footprint(self, entry, root="."):
        """What does `entry` ACTUALLY need at import time? Returns {required, naive, ratio, required_modules,
        required_external, optional_external, ...} -- the REQUIRED closure (imports that really run on import)
        against what a naive follow-every-import tracer reports. `required_external` is the pip-dependency answer
        a bundler/embedder needs. Complements accelerator_report (what's INSTALLED here) and tools/audit_imports
        (does an import RESOLVE). See holographic_deptrace.footprint_report."""
        from holographic.io_and_interop.holographic_deptrace import footprint_report
        return footprint_report(entry, root=root)

    def trace_imports(self, entry, root=".", follow=("hard",)):
        """The detailed import closure of `entry`, classifying every edge by WHERE it sits: hard (module top
        level -- runs on import, fatal if missing), guarded (inside try -- optional accelerator), deferred
        (inside a function -- does not run on import at all). `follow` picks which edge kinds the walk crosses.
        See holographic_deptrace.trace."""
        from holographic.io_and_interop.holographic_deptrace import trace
        return trace(entry, root=root, follow=follow)

    def accelerator_report(self):
        """Every optional dependency in one report: installed?, version, what it UNLOCKS (with the measured
        numbers -- 'faster' without a number is advertising), and the exact pip command. NumPy is the only
        required row; numba, ziglang (native 2-5x batch kernels, bit-identical in safe mode), cupy (GPU),
        sympy, pillow, flask are all opt-in. See holographic_backend.accelerator_report."""
        from holographic.misc.holographic_backend import accelerator_report
        return accelerator_report()

    def backend_status(self):
        """A human-readable line describing the current compute backend (GPU enabled/available/unavailable)."""
        from holographic.misc.holographic_backend import device_report
        return device_report()

    def run_procedure_batch(self, name_or_program, init_accs, max_steps=512):
        """Run a straight-line procedure over a BATCH of accumulators (N, D) in ONE interpret pass -- the
        data-parallel form of run_procedure (the architecture sweep's vectorisation of a Python-loop hot spot).
        Decodes each instruction once and applies it to all N rows at once; ~10x faster than looping
        run_procedure per item at N=2000, matching it to machine epsilon. SCOPE: value+register programs only
        (LOAD/BIND/BUNDLE/PERMUTE/STORE/RECALL/HALT); control/host ops (IFMATCH/CALL/APPLY/...) raise -- loop
        run_procedure for those. Returns the (N, D) accumulator batch. See HoloMachine.run_batch."""
        M = self._machine()
        if isinstance(name_or_program, str):
            if name_or_program not in M.functions:
                raise KeyError(f"no procedure named {name_or_program!r} -- learn_procedure it first")
            pv = M.functions[name_or_program]
        else:
            pv = M.assemble(name_or_program)
        return M.run_batch(pv, init_accs, max_steps=max_steps)

    def run_procedure(self, name_or_program, init_acc=None, max_steps=512,
                      stop=None, max_loop=64, converge_tol=0.999, branch_tol=0.5):
        """Execute a procedure and return (accumulator, trace). `name_or_program` is the name of a
        stored procedure or a fresh list of (opcode, operand) instructions (which may CALL stored
        procedures). `init_acc` seeds the accumulator -- pass a vector from the mind's own space to
        transform it, which is what makes a procedure an operation on the mind's data. Control-flow
        knobs: `stop` is a predicate acc->bool that lets an ITERATE loop exit when the desired OUTPUT is
        reached; `max_loop` caps loop iterations; `converge_tol` is the fixed-point tolerance an ITERATE
        converges at; `branch_tol` is the IFMATCH match threshold."""
        M = self._machine()
        if isinstance(name_or_program, str):
            if name_or_program not in M.functions:
                raise KeyError(f"no procedure named {name_or_program!r} -- learn_procedure it first")
            pv = M.functions[name_or_program]
        else:
            pv = M.assemble(name_or_program)
        return M.run(pv, init_acc=init_acc, max_steps=max_steps, handlers=self._procedure_handlers(),
                     stop=stop, max_loop=max_loop, converge_tol=converge_tol, branch_tol=branch_tol)

    def _procedure_handlers(self):
        """Handlers for APPLY <faculty>: one unary acc->acc map per faculty name, each delegating to a
        real mind faculty. 'cleanup' relaxes the accumulator toward the nearest known VALUE atom (the
        dense associative cleanup -- recovers a noisy accumulator); 'denoise' is the general manifold
        denoiser. KEPT NEGATIVE: denoise helps only when the accumulator carries low-rank/self-similar
        structure; on bare random value atoms there is no manifold, so 'cleanup' is the operative one
        there. Extend this dict (and DEFAULT_FACULTIES) to give procedures more unary faculties."""
        import numpy as _np
        M = self._machine()
        codebook = _np.stack([M.data_atoms[d] for d in M.data_names])

        def _cleanup(acc):
            from holographic.agents_and_reasoning.holographic_hopfield import dense_cleanup
            return dense_cleanup(acc, codebook, beta=25.0, steps=3)

        def _denoise(acc):
            return self.denoise(acc, method="auto")

        handlers = {"cleanup": _cleanup, "denoise": _denoise}
        W = getattr(self, "_matmul_W", None)
        if W is not None:                               # APPLY matmul := ACC <- W @ ACC (exact RNS matmul)
            def _matmul(acc):
                return self.exact_matmul(W, acc)
            handlers["matmul"] = _matmul
        inv = getattr(self, "_invprob", None)
        if inv is not None:                             # INV: restore a degraded measurement AS A PROGRAM --
            def _datafit(acc):                          # ITERATE [APPLY datafit; APPLY denoise] is the PnP/RED loop
                return acc - inv["mu"] * inv["adjoint"](inv["forward"](acc) - inv["y"])
            handlers["datafit"] = _datafit
            handlers["denoise"] = inv["prior"]          # the inverse problem's own manifold prior (overrides generic)
        gen = getattr(self, "_generator", None)
        if gen is not None:                             # GEN: generative diffusion AS A PROGRAM: ITERATE [APPLY diffuse]
            def _diffuse(acc):                          # one self-scheduled denoise-from-noise step: anneal beta up,
                import numpy as _np                     # cool injected noise down; ITERATE stops when the cooled,
                from holographic.agents_and_reasoning.holographic_hopfield import dense_cleanup   # sharpened cleanup reaches a fixed point on the manifold
                s = gen; frac = s["t"] / max(1, s["steps"] - 1)
                beta = s["beta0"] + (s["beta1"] - s["beta0"]) * frac
                noise = s["noise0"] * max(0.0, 1.0 - frac)
                z = dense_cleanup(acc, s["codebook"], beta=beta, steps=1, readout=s["readout"])
                if noise > 0:
                    z = z + noise * s["rng"].standard_normal(z.size) / _np.sqrt(z.size)
                z = z / (_np.linalg.norm(z) + 1e-12); s["t"] += 1
                return z
            handlers["diffuse"] = _diffuse
        pipe = getattr(self, "_pipe", None)
        if pipe is not None:                            # PIPE-1: the data-analysis pipeline's faculties.
            handlers.update(self._pipeline_handlers(pipe))   # (its 'denoise' overrides the generic one,
        user = getattr(self, "_apply_handlers", None)   # user-registered faculties (octree/nystrom/agent/...)
        if user:                                        # a registered name overrides a built-in of the same name
            handlers.update(user)
        return handlers                                 # because a raw signal needs a signal-shaped prior.)

    def register_apply_handler(self, name, fn):
        """Make any unary acc->acc faculty callable from a HoloMachine program as `APPLY <name>` -- INCLUDING
        stateful spatial ops (an octree query, a Nystrom approximation) and agent behaviours, since `fn` is a
        closure that may capture a built octree, a fitted embedding, an Agent, a DriveSystem, etc. The name is
        registered as a faculty atom on the VM so APPLY's operand cleans to it, and the handler is merged into
        the live handler set. This is the general extension point the APPLY docstring invites: the engine's
        faculties (and your own) become PROGRAMMABLE STEPS, so a synthesised or hand-written VSA program can
        denoise, recall, query space, approximate, or act, all inline -- the bridge from 'the agent drives a
        program' to 'a program drives the engine'. Backward-compatible: built-ins are untouched; a registered
        name overrides a built-in of the same name. `fn` must be callable acc->acc."""
        if not callable(fn):
            raise TypeError("an APPLY handler must be callable as acc -> acc")
        if getattr(self, "_apply_handlers", None) is None:
            self._apply_handlers = {}
        self._apply_handlers[name] = fn
        M = self._machine()
        if name not in M.faculty_names:                 # register the faculty atom so APPLY <name> cleans
            M.faculty_names.append(name)
            M.fac_atoms[name] = M._atom(f"fac:{name}")
        return self

    def set_matmul(self, W):
        """Configure the matrix used by APPLY matmul: that opcode then does ACC := W @ ACC, carried by the
        EXACT RNS matmul (no crosstalk; floats are fixed-point quantized). Pass a dim x dim matrix so the
        accumulator keeps its shape and an ITERATE can loop it -- which makes `ITERATE [APPLY matmul]` a
        recurrent linear map iterated to a fixed point (the literal input->process->feed-back pattern with
        real linear algebra). Set to None to disable (APPLY matmul becomes a safe no-op again)."""
        import numpy as _np
        self._matmul_W = None if W is None else _np.asarray(W, dtype=float)
        return self

    def set_inverse_problem(self, y, forward, adjoint, prior, mu=0.8):
        """Configure the inverse problem that APPLY datafit / APPLY denoise solve as a PROGRAM. After this,
        ITERATE [APPLY datafit; APPLY denoise] runs the Plug-and-Play/RED restoration loop: datafit pulls ACC
        toward the measurement (ACC <- ACC - mu*adjoint(forward(ACC) - y)) and denoise applies `prior`, an
        acc->acc manifold map. `forward`/`adjoint` are the operator A and its transpose A^T. Pass y=None to
        disable (datafit becomes a no-op and denoise reverts to the generic faculty). Usually set via
        restore_procedure(); exposed for hand-built restoration programs."""
        M = self._machine()
        if "datafit" not in M.fac_atoms:                # register the faculty atom so a program can name it
            M.fac_atoms["datafit"] = M._atom("fac:datafit"); M.faculty_names.append("datafit")
        self._invprob = None if y is None else {"y": np.asarray(y, float), "forward": forward,
                                                 "adjoint": adjoint, "prior": prior, "mu": float(mu)}
        return self

    def set_generator(self, codebook, steps=12, beta0=4.0, beta1=40.0, noise0=0.6, seed=0, readout="softmax"):
        """Configure the generative diffusion that APPLY diffuse runs as a PROGRAM. After this,
        ITERATE [APPLY diffuse] denoises from pure noise onto the `codebook` manifold -- the B10 sampler as a
        procedure. The diffuse step anneals beta up and injected noise down over `steps`; ITERATE halts when the
        cooled, sharpened cleanup reaches a fixed point. A composed/continuous codebook gives novel-but-valid
        samples; a BARE codebook degenerates to a stored atom (the kept B10 negative). Deterministic in `seed`.
        Usually set via generate_procedure(); exposed for hand-built generative programs."""
        import numpy as _np
        M = self._machine()
        if "diffuse" not in M.fac_atoms:
            M.fac_atoms["diffuse"] = M._atom("fac:diffuse"); M.faculty_names.append("diffuse")
        V = _np.atleast_2d(_np.asarray(codebook, float))
        V = V / (_np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)   # unit rows, as hopfield.generate uses
        self._generator = {"codebook": V, "steps": int(steps), "beta0": float(beta0), "beta1": float(beta1),
                           "noise0": float(noise0), "readout": readout, "rng": _np.random.default_rng(seed), "t": 0}
        return self

    def generate_procedure(self, codebook, steps=12, beta0=4.0, beta1=40.0, noise0=0.6, seed=0,
                           readout="softmax", max_loop=None):
        """Generate a sample by running the B10 diffusion AS A VSA PROGRAM -- ITERATE [APPLY diffuse] from a noise
        seed -- so the generative PROCESS is a stored, composable procedure (savable via to_recipe), not Python
        control flow. Process, not object. Returns (sample_vector, trace). Matches hopfield.generate's quality
        (it lands on the manifold), up to ITERATE's convergence-stop replacing the fixed step count. Kept
        negatives: B10's (a bare codebook converges to a stored atom; feed a composed manifold for novelty) and
        the procedure tax (a noisy unbind-and-clean per instruction read -- slower than the direct loop, which is
        the price of being data, not a faster path)."""
        import numpy as _np
        self.set_generator(codebook, steps=steps, beta0=beta0, beta1=beta1, noise0=noise0, seed=seed, readout=readout)
        M = self._machine()
        if "_diffuse_step" not in M.functions:
            self.learn_procedure("_diffuse_step", [("APPLY", "diffuse"), ("HALT", "a")])
        z0 = self._generator["rng"].standard_normal(self.dim)
        z0 = z0 / (_np.linalg.norm(z0) + 1e-12)
        return self.run_procedure([("ITERATE", "_diffuse_step"), ("HALT", "a")], init_acc=z0,
                                  converge_tol=0.9999, max_loop=max_loop or max(40, steps * 3))

    def restore_procedure(self, y, forward, adjoint, samples, mu=0.8, rank=24, steps=60):
        """Restore a degraded measurement y = forward(clean) + noise by running Plug-and-Play/RED AS A VSA PROGRAM
        -- ITERATE [APPLY datafit; APPLY denoise] -- so the restoration LOOP is a stored, composable procedure
        rather than Python control flow. The manifold prior is fit from `samples` (clean rows). Returns
        (restored_vector, trace). Reaches the same error-to-truth as denoise(method='pnp'); ITERATE halts at the
        fixed point, typically in far fewer iterations than a fixed step budget. Kept negative: the procedure tax
        (noisy instruction reads) -- this is the being-data form of restoration, not a faster path."""
        import numpy as _np
        from holographic.rendering.holographic_denoise import fit_manifold_full, adaptive_manifold_denoise
        S = _np.atleast_2d(_np.asarray(samples, float))
        basis, _, mean = fit_manifold_full(S, rank=min(int(rank), S.shape[1]))
        prior = lambda v: adaptive_manifold_denoise(v, basis, mean, sigma=None)
        self.set_inverse_problem(y, forward, adjoint, prior, mu=mu)
        M = self._machine()
        if "_pnp_step" not in M.functions:
            self.learn_procedure("_pnp_step", [("APPLY", "datafit"), ("APPLY", "denoise"), ("HALT", "a")])
        return self.run_procedure([("ITERATE", "_pnp_step"), ("HALT", "a")],
                                  init_acc=_np.asarray(adjoint(y), float), converge_tol=0.99999, max_loop=steps)

    # ---- PIPE-1: an automatic data-analysis pipeline expressed as a VSA PROGRAM ----------------------
    # The pipeline is NOT Python control flow -- it is a HoloMachine program (APPLY faculties, an ITERATE
    # loop, an IFMATCH branch, a CALL sub-routine) that orchestrates real faculties. The accumulator carries
    # the signal through the denoise loop, then decompose hands back a structured/noise FLAG atom the IFMATCH
    # branches on. Findings are recorded in self._pipe['report']; each APPLY step delegates downward.

    def _denoise_signal(self, sig, window=None):
        """Denoise a lone 1-D signal against its own low-rank trajectory (SSA/Cadzow). Now a thin delegate to
        the promoted denoise(method='trajectory') -- the primitive lives in the denoise faculty (a general
        prior-free 1-D denoiser beside nlm), so the pipeline and any other caller share one implementation
        rather than this owning a private copy."""
        return self.denoise(np.asarray(sig, float).ravel(), method="trajectory")

    def _pipeline_handlers(self, pipe):
        """The APPLY faculties the pipeline program calls. Each is a unary acc->acc map that reads the
        working signal and records findings in pipe['report'], delegating to a real faculty:
          analyze   -> detect_topology + basic stats
          denoise   -> _denoise_signal (self-similar trajectory denoise; overrides the generic denoiser)
          decompose -> decompose_signal (the generative law) + residual; returns a 'structured'/'noise' flag
          train     -> a learned KAN readout over the signal (fit_function)
          validate  -> held-out generalization: refit on 80%, predict the last 20%
          save      -> persist the structured form, the COMPACT generative law (the law IS the small file)."""
        import numpy as _np
        M = self._machine()
        rep = pipe["report"]

        def _analyze(acc):
            sig = _np.asarray(acc, float).ravel()
            pipe["signal"] = sig
            try:
                from holographic.mesh_and_geometry.holographic_manifold import detect_topology
                topo, _ = detect_topology(_np.arange(sig.size, dtype=float), sig)
            except Exception:
                topo = "unknown"
            rep.update({"n": int(sig.size), "mean": round(float(sig.mean()), 4),
                        "std": round(float(sig.std()), 4), "topology": topo})
            return acc

        def _denoise(acc):                              # overrides the generic denoiser: signals need a prior
            den = self._denoise_signal(_np.asarray(acc, float).ravel())
            pipe["signal"] = den
            return den

        def _decompose(acc):
            sig = _np.asarray(acc, float).ravel()
            try:
                f, info = self.decompose_signal(sig)
                recon = _np.asarray(f.generate(_np.arange(sig.size, dtype=float))).ravel()
                explained = max(0.0, 1.0 - float(_np.var(sig - recon[:sig.size])) / (float(_np.var(sig)) + 1e-12))
            except Exception as e:                      # decompose failed -> treat as no structure found
                rep["decompose_error"] = str(e)
                return M.data_atoms["noise"]
            pipe["structure"] = f
            pipe["residual"] = sig - recon[:sig.size]
            rep.update({"topology": info.get("topology"), "n_terms": info.get("n_terms"),
                        "explained_var": round(explained, 3),
                        "compression_ratio": round(float(info.get("compression_ratio") or 0.0), 1)})
            # hand the program a FLAG atom so its IFMATCH can branch on whether a law was actually found
            return M.data_atoms["structured"] if explained >= 0.5 else M.data_atoms["noise"]

        def _train(acc):                                # a learned (nonparametric) readout over the signal
            sig = pipe.get("signal")
            pipe["model"] = pipe.get("structure")
            try:
                self.fit_function(_np.arange(sig.size, dtype=float).reshape(-1, 1), sig)
                rep["trained"] = True
            except Exception:
                rep["trained"] = pipe["model"] is not None
            return acc

        def _validate(acc):                             # held-out: refit on 80%, predict the unseen last 20%
            sig = pipe.get("signal")
            n = len(sig); cut = max(8, int(0.8 * n))
            try:
                f, _ = self.decompose_signal(sig[:cut])
                pred = _np.asarray(f.generate(_np.arange(n, dtype=float))).ravel()[cut:]
                rms = float(_np.sqrt(_np.mean((sig[cut:] - pred) ** 2)))
                rep["heldout_rms"] = round(rms, 4)
                rep["heldout_rel"] = round(rms / (float(_np.std(sig)) + 1e-12), 3)
            except Exception as e:
                rep["validate_error"] = str(e)
            return acc

        def _save(acc):                                 # store the structured form: the compact generative law(s)
            laws = pipe.get("level_laws") or ([pipe["structure"]] if pipe.get("structure") is not None else [])
            has_law = rep.get("explained_var", 0.0) >= 0.5 and len(laws) > 0   # only if real structure found
            if has_law:
                import os, tempfile
                total = 0
                for i, f in enumerate(laws):
                    if hasattr(f, "save"):
                        p = os.path.join(tempfile.gettempdir(), f"holostuff_pipe_law_{i}.json")
                        try:
                            f.save(p); total += os.path.getsize(p)
                        except Exception:
                            pass
                rep["law_bytes"] = total
                rep["saved_as"] = "law_ladder" if len(laws) > 1 else "generative_law"
            else:                                       # no compressible structure -> honest: keep raw samples
                rep["saved_as"] = "raw_only"
            return acc

        def _peel_step(acc):                            # one LEVEL of the recursive peel: decompose the residual
            r = _np.asarray(acc, float).ravel()
            if pipe.get("peel_input_var") is None:      # remember the energy we started peeling from
                pipe["peel_input_var"] = float(_np.var(r)) + 1e-12
            if float(_np.std(r)) < 0.01 * (pipe["peel_input_var"] ** 0.5):    # residual already negligible
                return r                                # (unchanged -> ITERATE converges) -- nothing left to peel
            try:
                f, info = self.decompose_signal(r)
                recon = _np.asarray(f.generate(_np.arange(r.size, dtype=float))).ravel()[:r.size]
                ev = max(0.0, 1.0 - float(_np.var(r - recon)) / (float(_np.var(r)) + 1e-12))
            except Exception:
                return r                                # can't decompose -> residual unchanged -> ITERATE stops
            # Stop on the engine's OWN MDL verdict: decompose returns n_terms==0 when no real structure remains
            # (it already gates out noise-fitting). A level may explain only a MODEST fraction yet be real --
            # a line trend under a comparable sine -- so the right test is "did the MDL gate admit a term?",
            # not "did this level explain most of the residual?".
            if (info.get("n_terms") or 0) < 1:
                return r                                # MDL found nothing -> residual is noise -> stop peeling
            pipe["levels"].append({"level": len(pipe["levels"]) + 1, "topology": info.get("topology"),
                                   "n_terms": info.get("n_terms"), "explained_of_residual": round(ev, 3)})
            pipe["level_laws"].append(f)
            return r - recon                            # hand the next level what THIS one could not explain

        def _assess(acc):                               # after peeling: finalize the report, flag the branch
            r = _np.asarray(acc, float).ravel()
            base = pipe.get("peel_input_var") or (float(_np.var(r)) + 1e-12)
            cum = max(0.0, 1.0 - float(_np.var(r)) / base)
            lv = pipe["levels"]
            rep.update({"n_levels": len(lv), "levels": lv, "cumulative_explained": round(cum, 3),
                        "final_residual_energy": round(float(_np.std(r)), 4),
                        "explained_var": round(cum, 3),         # same field the branch/save read in single mode
                        "n_terms": sum(int(d["n_terms"] or 0) for d in lv)})
            return M.data_atoms["structured"] if len(lv) > 0 else M.data_atoms["noise"]

        return {"analyze": _analyze, "denoise": _denoise, "decompose": _decompose,
                "train": _train, "validate": _validate, "save": _save,
                "peel": _peel_step, "assess": _assess}

    def analyze_dataset(self, data):
        """Set up the automatic-analysis pipeline over a 1-D signal: register the analyze/decompose/train/
        validate/save APPLY faculties and the 'structured'/'noise' flag atoms the program's IFMATCH branches
        on, and clear the findings report. The pipeline faculty/flag atoms are added to the machine ON DEMAND
        so the default VM stays lean. Then run the pipeline PROGRAM with run_analysis_pipeline (or assemble
        your own). Returns self."""
        import numpy as _np
        M = self._machine()
        for flag in ("structured", "noise"):            # IFMATCH branch flags (data codebook)
            if flag not in M.data_atoms:
                M.data_atoms[flag] = M._atom(f"dat:{flag}"); M.data_names.append(flag)
        for fac in ("analyze", "decompose", "train", "validate", "save", "peel", "assess"):  # APPLY faculties
            if fac not in M.fac_atoms:
                M.fac_atoms[fac] = M._atom(f"fac:{fac}"); M.faculty_names.append(fac)
        self._pipe = {"report": {}, "signal": _np.asarray(data, dtype=float),
                      "structure": None, "residual": None, "model": None,
                      "levels": [], "level_laws": [], "peel_input_var": None}
        return self

    def pipeline_report(self):
        """The findings recorded by the last analysis pipeline run (topology, explained variance, n_terms,
        held-out generalization, saved size)."""
        return dict(getattr(self, "_pipe", {}).get("report", {}))

    def run_analysis_pipeline(self, data, program=None, max_loop=12, recursive=False):
        """Run the standard automatic-analysis pipeline -- a VSA PROGRAM -- over a 1-D signal and return the
        findings report. The PROGRAM (not Python) drives the logic:

            APPLY  analyze            profile the data (stats + topology)
            ITERATE _denoise_step     denoise until the signal SETTLES   (the loop)
            APPLY  decompose          find the generative law; residual; flag structured/noise
            IFMATCH structured        only if a law was actually found...
              CALL _train_validate    ...train a readout and check held-out generalization
            APPLY  save               store the structured form: the compact generative law
            HALT

        With recursive=True the single `APPLY decompose` becomes `ITERATE _peel_step` -- decompose the
        DOMINANT law, then peel its residual and decompose THAT, layer by layer, until the residual has no
        structure left (the loop converges when decompose's own MDL gate admits no term -- n_terms==0 -- or
        the residual is already negligible). An `APPLY assess` then flags the branch and records the level
        ladder. This is the "access every level" mode: a line trend + a periodic part are caught on SEPARATE
        levels that one decompose cannot fit together (the trend explains only a modest fraction, so a level
        is kept by the MDL verdict, not by how much it explains).

        Every APPLY step delegates to a real faculty (see _pipeline_handlers). On a structured signal the
        program finds the law(s) and runs train+validate; on pure noise decompose reports no structure, the
        IFMATCH SKIPS train+validate, and the report says so honestly -- both branches from one program.
        Pass `program` to override the default."""
        import numpy as _np
        self.analyze_dataset(data)
        self.learn_procedure("_denoise_step", [("APPLY", "denoise"), ("HALT", "a")])
        self.learn_procedure("_train_validate", [("APPLY", "train"), ("APPLY", "validate"), ("HALT", "a")])
        self.learn_procedure("_peel_step", [("APPLY", "peel"), ("HALT", "a")])
        if program is None:
            if recursive:
                program = [("APPLY", "analyze"),
                           ("ITERATE", "_denoise_step"),
                           ("ITERATE", "_peel_step"),       # peel structure layer by layer (every level)
                           ("APPLY", "assess"),             # finalize the ladder, flag the branch
                           ("IFMATCH", "structured"),
                           ("CALL", "_train_validate"),
                           ("APPLY", "save"),
                           ("HALT", "a")]
            else:
                program = [("APPLY", "analyze"),
                           ("ITERATE", "_denoise_step"),
                           ("APPLY", "decompose"),
                           ("IFMATCH", "structured"),
                           ("CALL", "_train_validate"),
                           ("APPLY", "save"),
                           ("HALT", "a")]
        _, trace = self.run_procedure(program, init_acc=_np.asarray(data, dtype=float), max_loop=max_loop)
        rep = self.pipeline_report()
        rep["_ops"] = [t[0] for t in trace]             # which opcodes actually executed (shows the branch)
        return rep

    def index_procedures(self, names=None, probe_seed="canonical_probe"):
        """Build a behavioural FINGERPRINT index: run each procedure ONCE on a canonical probe and
        cache its output. Recall can then identify a bind-transform from a single example with ZERO
        further program runs (recover the key, match the implied fingerprint). One-time O(library) cost,
        amortised across all later recalls. Call again after adding procedures to refresh the index."""
        from holographic.agents_and_reasoning.holographic_ai import derived_atom
        M = self._machine()
        names = list(names) if names is not None else list(M.functions)
        self._proc_probe = derived_atom(self.seed, probe_seed, self.dim, unitary=True)
        handlers = self._procedure_handlers()
        self._proc_fp = {n: M.run(M.functions[n], init_acc=self._proc_probe, handlers=handlers)[0]
                         for n in names}
        # Cache a unit-normalized fingerprint MATRIX so recall is ONE matrix-vector product (cosine of the
        # implied kernel against every fingerprint at once) instead of a Python loop -- measured 6-26x faster
        # than the loop and 3-7x faster than a HoloForest index, which only beats a linear scan past ~4000
        # procedures (a regime the library vector cannot even hold). Rows are unit norm so mat @ qhat == cosine.
        names_l = list(self._proc_fp)
        mat = np.stack([self._proc_fp[n] for n in names_l]) if names_l else np.zeros((0, self.dim))
        self._proc_fp_names = names_l
        self._proc_fp_mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
        return self

    def recall_procedure(self, input_vec, output_vec, names=None, method="auto", fp_floor=0.5):
        """Goal-addressable recall over the procedure library: given ONE (input -> output) example,
        return (name, score) of the stored procedure whose behaviour best reproduces it. Returns
        (None, score) if the library is empty.

        method='behavioral' runs each candidate on the input and matches its output -- general (works
        for ANY procedure), at O(library size) executions. method='fingerprint' uses the precomputed
        index (see index_procedures): it recovers the transform's kernel from the example and matches
        the IMPLIED fingerprint, with ZERO program runs -- exact for the LINEAR (convolution) class,
        which is bind AND permute and their compositions (permutation is convolution by a shifted
        delta, so it commutes with binding the same way), ~30x cheaper. It scores near zero for
        genuinely NONLINEAR procedures (e.g. ones with an APPLY cleanup/denoise step). method='auto'
        (default) gets the best of both: try the fingerprint shortcut first IF an index exists, trust
        it only when its match clears fp_floor (so it fires only for the linear transforms it actually
        covers -- nonlinear guesses score near zero), and otherwise fall back to the behavioural scan.
        With no index, 'auto' is exactly the behavioural scan, so this stays backward-compatible.

        The fingerprint scan is VECTORIZED: one matrix-vector product of the implied kernel against the
        cached unit-normalized fingerprint matrix gives every cosine at once (measured 6-26x faster than the
        per-candidate Python loop). A HoloForest index was measured and REJECTED: it is 3-7x SLOWER than this
        vectorized scan for any realistic library and only beats a linear scan past ~4000 procedures -- a
        regime the single library vector cannot hold anyway, so the sub-linear index is premature here. The
        right fix for an O(N) scan that was slow was to vectorize it, not to index it."""
        from holographic.agents_and_reasoning.holographic_ai import cosine as _cos, bind as _bind, unbind as _unbind
        M = self._machine()
        # -- fingerprint fast-path (zero program runs; gated by confidence so it only fires when right)
        if method in ("auto", "fingerprint") and getattr(self, "_proc_fp", None):
            implied = _bind(self._proc_probe, _unbind(output_vec, input_vec))   # assumes a bind transform
            mat = getattr(self, "_proc_fp_mat", None)
            if names is None and mat is not None and len(self._proc_fp_names):
                qhat = implied / max(float(np.linalg.norm(implied)), 1e-12)
                scores = mat @ qhat                          # cosine vs every fingerprint in ONE matvec
                bi = int(scores.argmax())
                best, bs = self._proc_fp_names[bi], float(scores[bi])
            else:                                            # named subset (or no matrix): scan the dict
                cands = list(names) if names is not None else list(self._proc_fp)
                best, bs = None, -9.0
                for n in cands:
                    if n in self._proc_fp:
                        s = float(_cos(implied, self._proc_fp[n]))
                        if s > bs:
                            bs, best = s, n
            if method == "fingerprint" or bs >= fp_floor:
                return best, bs
        # -- behavioural scan (general fallback)
        cands = list(names) if names is not None else list(M.functions)
        handlers = self._procedure_handlers()
        best, best_s = None, -9.0
        for n in cands:
            out, _ = M.run(M.functions[n], init_acc=input_vec, handlers=handlers)
            s = float(_cos(out, output_vec))
            if s > best_s:
                best_s, best = s, n
        return best, best_s


def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p12_proc_texture", "_UnifiedPart12")
    print("holographic_unified_p12_proc_texture selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()

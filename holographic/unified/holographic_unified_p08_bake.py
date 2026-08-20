"""Part 08 of UnifiedMind's faculty surface -- 134 methods, bake .. sparse_reconstruct.

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


class _UnifiedPart08:

    def bake(self, evaluator, vary="position", **kw):
        """CONSOLIDATION CACHE (H2) -- bake a slow `evaluator` over the thing that VARIES, then look it up cheaply.
        vary='position' (kw: lo, hi, res) returns a BakedGrid with .sample(points); 'constant' computes once; 'view'
        and 'time' delegate to the BRDF LUT / frame bakes. See holographic_cachehome.Cache."""
        from holographic.caching_and_storage.holographic_cachehome import Cache
        return Cache.bake(evaluator, vary=vary, **kw)

    def build_index(self, vectors, labels=None, method="auto", seed=0):
        """CONSOLIDATION INDEX (H1) -- build a nearest-neighbour index over `vectors` with one interface: an exact
        cosine scan for small sets, the sub-linear RP-forest for large ones (chosen by `method='auto'`). The result
        has `.nearest(query, k, abstain=alpha)` -> [(label_or_index, score), ...], with an optional calibrated
        abstain. See holographic_index.Index."""
        from holographic.caching_and_storage.holographic_index import Index
        return Index(vectors, labels=labels, method=method, seed=seed)

    def find_capability(self, problem, k=3, accepts=None, produces=None):
        """CONSOLIDATION CATALOG (C1) -- 'search before you build'. Describe a problem in plain English and get the
        engine homes that already solve it, best first, so you don't build a duplicate. The catalog is seeded with
        the consolidation homes (Index/Cache/Field/...) AND this mind's own public faculties (their docstrings), so
        both curated homes and live methods are findable. Returns a list of holographic_catalog.Capability (each has
        .name, .does, .example, .native, .semantic, .consumes, .produces).

        S3 io-shape filter: `accepts='mesh'` keeps only capabilities that consume a mesh ('what can I run on this
        mesh?'); `produces='mesh'` keeps only those that yield one. Untagged capabilities are unspecified and always
        shown. See holographic_catalog."""
        return self._capability_catalog().find_capability(problem, k=k, accepts=accepts, produces=produces)

    def find_capability_uris(self, problem, k=3):
        """Like find_capability, but each result is annotated with its disambiguating capability URI(s) so a caller
        NEVER gets a bare ambiguous name (holographic_catalog + holographic_capuri). Returns [{name, does, example,
        uris}] where `uris` is the full path(s) the name resolves to -- one for a unique name, several for a
        colliding one (e.g. 'rotation' -> both meshskin and scenegraph). The collision fix at the discovery layer:
        the agent sees the path to supply, not just the name. See holographic_catalog.find_capability_uris."""
        return self._capability_catalog().find_capability_uris(problem, k=k)

    def suggest_pipeline(self, start_kind, goal_kind, max_len=4, require_step=False):
        """Propose a PIPELINE from `start_kind` to `goal_kind` (io kinds, see holographic_iokinds) by chaining
        capabilities whose `produces` feeds the next's `consumes` (holographic_catalog). Returns the shortest chain
        as a list of {name, consumes, produces} steps, or None if no route within `max_len`. The render-graph idea
        over the whole catalog: instead of one capability, the engine proposes a ROUTE -- e.g. 'transform'->'selection'
        chains transform_selection then mesh_selection. Answers 'how do I get from what I have to what I want?'.

        `require_step=True` on a same-kind query ('mesh'->'mesh') demands a real TRANSFORMING edge (mesh_smooth,
        mesh_subdivide, ...) instead of the empty 'already there' pipeline -- the 'what can I DO to a mesh?' answer.
        See holographic_catalog.suggest_pipeline."""
        return self._capability_catalog().suggest_pipeline(start_kind, goal_kind, max_len=max_len,
                                                           require_step=require_step)

    def io_kinds(self):
        """The closed vocabulary of io DATATYPE kinds a capability can consume/produce (holographic_iokinds) --
        mesh, points, sdf, sdf_scene, field, image, hypervector, transform, selection, scalar, curve, skeleton.
        These are the kinds the find_capability accepts=/produces= filter and suggest_pipeline route over. Coarse on
        purpose: the fine distinctions live in each capability's docstring.

        THE CONTRACT (a client hardcoded this list as a dropdown fallback and asked what it can rely on):
        * STABLE -- mesh, points, sdf, sdf_scene, field, image, hypervector, transform, selection, scalar. These
          carry live pipeline edges and will not be renamed or removed; build UI against them.
        * PROVISIONAL -- curve, skeleton, timeseries, spectrum. Real and routable, but thinly populated: curve and
          skeleton currently have NO tagged producer (you import a skeleton, you draw a curve -- the engine
          consumes them rather than making them), so they show up as `source_only` in pipeline_map().gaps. That is
          an honest gap, not an oversight, and it is the shape most likely to change as tagging fills in.
        The vocabulary only ever GROWS (a new kind is additive); a rename would break every tag at once, which is
        why the closed list is validated at registration. Read it from here rather than hardcoding -- that is what
        this faculty is for. See holographic_iokinds.IO_KINDS and mind.pipeline_map()["gaps"]."""
        from holographic.caching_and_storage.holographic_iokinds import IO_KINDS
        return list(IO_KINDS)

    def route_structured(self, request, module_texts):
        """Route a REQUEST to modules by holographic role-STRUCTURE, not a bag-of-words mean: parse request
        and each module into a {action, object, quality} record, bind+bundle via encode_record, match the
        bound records. Separates the case a flat mean buries (denoise vs fsr on 'less grainy'). module_texts
        is {name: docstring}; returns [(name, score)] high-to-low; [] if the request does not parse (caller
        falls back to flat). See holographic_holoroute.route_structured."""
        import holographic.semantic_router.holographic_holoroute as HR
        return HR.route_structured(self, request, module_texts)

    def extract_roles(self, text):
        """Parse a request or docstring into a holographic role record {action, object, quality} over the
        controlled vocabulary (object fillers = io-kinds). Missing roles omitted, never fabricated; {} means
        unparsed. The structured half of route_structured. See holographic_holoroute.extract_roles."""
        import holographic.semantic_router.holographic_holoroute as HR
        return HR.extract_roles(text)

    def match_record(self, query_fields, candidates, top=None):
        """DOMAIN-GENERAL structured matching: rank `candidates` ({name: {role: filler}}) by how well their
        role-filler RECORD matches `query_fields`, via bound-record similarity (bind+bundle+cosine). The
        general form of route_structured -- the SAME primitive classifies a physics regime, a market event, an
        astronomy source, or a mesh repair, because it only needs items sharing a schema of codebook fillers.
        Returns [(name, score)] high-to-low (top-k if given); [] if the query record is empty (abstains, never
        guesses). See holographic_relations.match_record."""
        from holographic.misc.holographic_relations import match_record
        return match_record(self.encode_record, query_fields, candidates, top=top)

    def match_prototype(self, query, prototypes, encode=None):
        """UNSTRUCTURED classification (the twin of match_record): when an item has NO role schema -- it is a
        bag/blend, not a record -- match it to the nearest class PROTOTYPE by cosine. The general form of the
        VSA intent router: classify a question, a gesture, a regime, a writing style by the blend of its
        features. `prototypes` is {class: unit vector} (see build_prototypes); `encode` maps the query (default
        self.perceive text). Returns ranked [(class, score)]; composes with decide_or_abstain. See
        holographic_relations.match_prototype."""
        from holographic.misc.holographic_relations import match_prototype
        enc = encode or (lambda x: self.perceive(x, "text"))
        return match_prototype(enc, query, prototypes)

    def build_prototypes(self, classes, encode=None):
        """Build class prototypes from EXAMPLES: {class: [example, ...]} -> {class: unit mean vector} (the mean
        bundle of each class's example encodings). The example-driven setup for match_prototype -- supply
        instances, not a schema. `encode` defaults to self.perceive text. See holographic_relations.build_prototypes."""
        from holographic.misc.holographic_relations import build_prototypes
        enc = encode or (lambda x: self.perceive(x, "text"))
        return build_prototypes(enc, classes)

    def decide_or_abstain(self, ranked, margin=0.1, min_score=None):
        """The shared DECISION step: given ranked [(name, score)] from match_record / match_prototype / any
        scorer, return (winner, score, confident) where confident requires the top-1 to beat top-2 by >= margin
        (and clear min_score). One honest abstention rule the classify callers share instead of each inventing
        its own. Cheap gap gate -- for calibrated significance use a shuffle null. See
        holographic_relations.decide_or_abstain."""
        from holographic.misc.holographic_relations import decide_or_abstain
        return decide_or_abstain(ranked, margin=margin, min_score=min_score)

    def pipeline_map(self):
        """The WHOLE workflow graph as data: every typed edge (consume_kind -> produce_kind -> capability)
        derived from the live catalog's consumes/produces tags, plus per-kind producers/consumers, coverage,
        and a gap report (dead-end / source-only / untouched kinds). Where suggest_pipeline answers ONE route
        ('mesh -> image?'), this returns the entire map an agent can plan over without re-deriving it. See
        pipelinemap.generate (which also writes docs/PIPELINE_MAP.md + pipelines.json). Returns the dict."""
        # IN THE PACKAGE, NOT AT THE ROOT. pipelinemap.py lived beside
        # setup.py and was never in the wheel, so this call worked from a repo
        # checkout (cwd shadowing) and raised ModuleNotFoundError for anyone
        # who pip-installed -- across SEVEN releases, 0.2.3 to 0.2.14, because
        # every release check ran from the repo root.
        # Moving it into holographic/ kills the whole TOP-LEVEL-MODULE bug
        # class rather than adding one more manifest entry to forget.
        from holographic.caching_and_storage import (
            holographic_pipelinemap as pipelinemap)
        cat = self._capability_catalog()
        edges = pipelinemap._edges(cat)
        produce, consume = pipelinemap._adjacency(edges)
        from holographic.caching_and_storage.holographic_iokinds import IO_KINDS
        dead_end, source_only, untouched = pipelinemap._orphans(edges, IO_KINDS)
        total = len(cat._by_name)
        tagged = sum(1 for c in cat._by_name.values() if c.consumes and c.produces)
        return {"coverage": {"tagged": tagged, "total": total,
                             "percent": (100 * tagged // total) if total else 0},
                # C7: every edge carries the CALLABLE name as data. `capability` is often prose ("Mesh repair
                # (weld + split non-manifold + fill + compact)"), so a client had to regex `m.foo(` out of the
                # example to actually invoke an edge -- fragile enough to need its own EXCLUDED.md. `method` is
                # verified callable against this mind (seed_from_mind nulls the liars), and None means honestly
                # import-only rather than a name that would fail at call time.
                "edges": [{"consumes": ci, "produces": po, "capability": n,
                           "method": getattr(cat._by_name.get(n), "method", None)} for ci, po, n in edges],
                "produced_by": produce, "consumed_by": consume,
                "gaps": {"dead_end": dead_end, "source_only": source_only, "untouched": untouched}}

    def _embedding_router(self):
        """Locate + load the shipped routing index (lecore_data/routing/index_<dim>d.npz), preferring 128d --
        the MEASURED champion dim (dense 6/12 top-1; with gamma=0.5 bone fusion 7/12, median 1, zero per-ask
        regressions) -- falling back to 64d (measured WEAKER: dense 2/12, and bones do NOT help there; the
        lexical channel does). Cached; returns None honestly when no artifact ships (route_semantic then
        returns None and the caller falls back to find_capability). WHY THIS EXISTS AS A GUARDED HELPER: the
        original helper was LOST in a branch reconciliation and route_semantic raised AttributeError on every
        call -- found by dogfooding, not by the audits (they check wiring, not execution). The regression trap
        is the integration test asserting route_semantic never raises."""
        r = getattr(self, "_embedding_router_cache", None)
        if r is not None:
            return r if r is not False else None
        import os
        try:
            from holographic.semantic_router.holographic_router import EmbeddingRouter
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            for dim in (128, 64):
                path = os.path.join(root, "lecore_data", "routing", "index_%dd.npz" % dim)
                if os.path.isfile(path):
                    self._embedding_router_cache = EmbeddingRouter(path)
                    return self._embedding_router_cache
        except Exception:
            pass                                              # a broken artifact must not take routing down
        self._embedding_router_cache = False                  # tried, absent -> do not re-scan every call
        return None

    def _query_embedder(self):
        """Locate + load the offline QUERY embedder (N31: free text -> nomic-64 vector with NO model, via the
        distilled token-vector artifact from distill_map.py --export, expected under lecore_data/routing/).
        SAME guarded pattern as _embedding_router, for the same measured reason: this helper was ALSO lost in
        the branch reconciliation -- route_semantic raised AttributeError the moment a routing index was
        present and a free-text query reached the embed step. Found only by running the FULL production loop
        against a real artifact (the earlier no-artifact test returned None before this line -- a coverage
        hole, now closed in tests). Cached; honest None when no artifact ships (route_semantic then returns
        None and the caller falls back to find_capability)."""
        qe = getattr(self, "_query_embedder_cache", None)
        if qe is not None:
            return qe if qe is not False else None
        import os
        try:
            from holographic.semantic_router.holographic_queryembed import QueryEmbedder
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            # Canonical name FIRST. This list and export_query_embed's recommended path are pinned to agree
            # by tests/test_queryembed_artifact.py -- they had already diverged three ways (the loader wanted
            # queryembed_64d/query_map_64d, the exporter said query_embed.npz), so a fitted artifact would
            # have landed and silently never loaded: route_semantic would keep returning None with the cure
            # sitting in the right directory under the wrong name.
            for name in ("query_embed_128d.npz", "query_embed.npz", "queryembed_64d.npz", "query_map_64d.npz"):
                path = os.path.join(root, "lecore_data", "routing", name)
                if os.path.isfile(path):
                    self._query_embedder_cache = QueryEmbedder(path)
                    return self._query_embedder_cache
        except Exception:
            pass                                              # a broken artifact must not take routing down
        self._query_embedder_cache = False
        return None

    def set_embedder(self, fn, verify=True, min_rate=0.30, sample=12, k=5, seed=0):
        """BRING YOUR OWN QUERY EMBEDDER (holographic_embedseam, SEAM-1) -- install ANY callable text->vector
        so route_semantic can reach the dense index from free text. Same contract as attach_llm: leCore
        imports no model SDK; you bring the model. Pass fn=None to remove it.

        WHY IT IS NEEDED, AND WHY THE OBVIOUS ALTERNATIVE IS NOT COMING: the shipped artifact is the
        DOCUMENT side only (509 modules x 128d), so free text routes to an honest None. The natural fix --
        distil the encoder into a token table plus one ridge matrix W (tools/semantic/distill_map.py) and
        ship that as the query side -- WAS TRIED, MEASURED ON THE REAL CORPUS, AND REFUSED. At the shipped
        128d:
            [floor] SIF token-pool, NO learned map    top-1 3/12   median 13
            [ours]  SIF @ W, the distilled map        top-1 1/12   median 19
        The learned map is WORSE THAN NOT HAVING ONE: ridge explained R^2 +0.06 of held-out variance and
        cost 2 top-1 and 6 median rank to apply. distill_map's export gate (EXPORT_BAR_TOP1=4,
        EXPORT_BAR_MEDIAN=8) refuses it, and tests/test_queryembed_artifact.py pins BOTH halves -- the gate
        and the absence of any committed artifact. So this seam is not a stopgap while the distilled path
        matures; IT IS THE ROUTE. Do not re-propose shipping a distilled query artifact without new numbers
        that beat the floor.

        VERIFICATION IS ON BY DEFAULT, and this is the point of the seam rather than a bare setter. The index
        lives in ONE embedding space (nomic, ABTT-corrected). A cosine between a query embedded by a
        DIFFERENT model and a nomic document is not a weak signal, it is a MEANINGLESS one -- and it still
        returns five ranked names with confident-looking scores. Dimension is checkable and is checked;
        SPACE IS NOT, so verify=True runs a round-trip probe: sample modules that are in the index, embed
        each one's OWN docstring summary, and require it to self-recall at top-k. Chance is 5/509 = 0.0098,
        so the default 0.30 bar is ~30x chance -- loose on purpose, because the job is separating "right
        space" from "unrelated space", not grading embedding quality.

        Raises ValueError with the measured rate and the misses when the probe fails. Pass verify=False to
        install unchecked (documented, not recommended). Returns the probe dict on success, None when
        clearing. NOTE the probe is necessary, not sufficient: it cannot tell a good in-space embedder from
        a mediocre one."""
        if fn is None:
            self._user_embedder = None
            return None
        if not callable(fn):
            raise TypeError("set_embedder needs a callable text->vector, got %r" % type(fn))
        report = None
        if verify:
            r = self._embedding_router()
            if r is None:
                raise ValueError("no routing index is present, so an embedder cannot be verified against it; "
                                 "pass verify=False to install it unchecked")
            from holographic.semantic_router.holographic_embedseam import probe_embedder_space
            import os
            root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__import__("holographic").__file__))))
            report = probe_embedder_space(fn, r, root, sample=sample, k=k, seed=seed)
            if report.get("ok") is False or report["rate"] < min_rate:
                raise ValueError(
                    "embedder failed the space-agreement probe: %s -- required rate >= %.2f. This almost "
                    "always means the model is not in the shipped index's space (nomic/128d), in which case "
                    "its cosines against the index are meaningless. Misses: %s"
                    % (report["reason"], min_rate, report["misses"][:5]))
            report["ok"] = True
        self._user_embedder = fn
        return report

    def route_semantic(self, problem, k=5, query_vec=None, gamma=1.0):
        """N28 -- route a request to the right MODULE by cosine in nomic's embedding space, not token
        overlap. Uses the shipped q8 index (lecore_data/routing/index_128d.npz preferred, 64d fallback)
        with the ABTT correction baked in. THE DEFAULT gamma IS THE MEASURED CHAMPION PER CORPUS EPOCH,
        not a setting: 0.5 was crowned at 537 corpus entries (7/12 top-1, median 1 on the 12-ask suite);
        at 715 entries CI's full sweep showed gamma=1.0 Pareto-dominating 0.5 at the ship dim (top-1 6
        vs 5, median 2 vs 2.5, worst 80 vs 90, top-5 equal) and at 768d, so the default moved 0.5 -> 1.0
        with the exam's SHIPPED_GAMMA and the CI bars in lockstep (tools/semantic/knowledge_index.py has
        the full record). Keyword-overlap baseline for scale: 1/12 top-1. Pass gamma=0.0 for plain
        cosine. Boneless index -> gamma degrades gracefully to plain dense, so old artifacts stay safe.

        It needs a query VECTOR. Supply one via `query_vec` (a 64d nomic vector your app produced), OR
        rely on the build-time cache for a known phrase. With NEITHER -- a brand-new free-text query and
        no model present -- this returns None and the caller should use find_capability (token) instead.
        It NEVER fabricates an embedding: an honest None beats a confident wrong route.

        Returns [(module_name, cosine)] best-first, or None if the query could not be embedded. Default
        off in spirit: find_capability is unchanged; this is an additive second path.
        See holographic_router.EmbeddingRouter."""
        r = self._embedding_router()
        if r is None:
            return None
        if query_vec is not None:
            return r.route(query_vec, k=k, gamma=gamma)       # gamma>0 = measured dense+bones fusion (7/12)
        # USER-SUPPLIED EMBEDDER (set_embedder) comes BEFORE the build-time cache, and the order is a
        # correctness call, not a preference: the cache holds vectors in the SHIPPED index's space, while a
        # user embedder is in whatever space its model uses. Mixing the two inside one ranking would compare
        # cosines from two different geometries -- so once a caller installs an embedder, every query goes
        # through it and the ranking stays internally consistent. An explicit query_vec above still wins,
        # because that is the caller being explicit.
        fn = getattr(self, "_user_embedder", None)
        if fn is not None:
            try:
                v = fn(problem)
            except Exception:
                v = None                                        # a broken embedder is a miss, never a raise
            if v is not None:
                return r.route(v, k=k, gamma=gamma)
        cache = getattr(self, "_query_cache", None)
        if cache is not None:
            hit = r.route_cached(problem, cache, k=k)
            if hit is not None:
                return hit
        qe = self._query_embedder()                             # N31: offline text -> nomic-64, no model
        if qe is not None:
            v = qe.embed(problem)
            if v is not None:
                return r.route(v, k=k)
        return None                                             # no vector, no map, no model -> honest miss

    def bm25_rank(self, query, docs, k1=1.5, b=0.75, top=None, expand=False):
        """LEXICAL ranking: rank `docs` (list of text strings) by Okapi BM25 against `query` -- exact term
        matching with tf-saturation (k1) and length normalization (b), pure NumPy/stdlib, no model. The
        complement to route_semantic's dense cosine: BM25 catches asks whose query WORDS appear in the target
        text but whose meaning-geometry buries them. expand=True additionally matches DERIVATIONAL siblings at
        half weight (a query saying 'emissive' reaches a doc saying 'emission' -- same root, different suffix);
        default off, exact matches always dominate. Returns [(doc_index, score)] best-first. See
        holographic_bm25.BM25."""
        from holographic.semantic_router.holographic_bm25 import BM25
        return BM25(list(docs), k1=k1, b=b).rank(query, top=top, expand=expand)

    def perfect_recall_index(self, filter_bits=2048, k=4, tile=512, tile_bits=1 << 16):
        """EXACT-containment index with GUARANTEED perfect recall at any corpus size (time is the only
        scaling cost): per-doc sparse binary superposition filters (a Bloom filter is the simplest
        computing-in-superposition, Kleyko et al. 2020/2022 -- structurally ZERO false negatives) under
        OR-baked tile probes (irradiance-map cull; probes have their OWN resolution, a measured
        saturation negative), with every candidate exactly verified against sha256 term-hash sets (the
        depth test -- ZERO false positives). idx.add({'token': [...], 'trigram': [...]}) per doc;
        idx.query(terms, channel) returns EXACTLY the ground-truth AND-containment set, multi-channel,
        no BM25 anywhere. KEPT NEG: containment is not relevance ranking; a ubiquitous term degenerates
        toward the O(N) verify scan -- 'no limit but time', literally. See
        holographic_perfectrecall.PerfectRecallIndex."""
        from holographic.caching_and_storage.holographic_perfectrecall import PerfectRecallIndex
        return PerfectRecallIndex(filter_bits=filter_bits, k=k, tile=tile, tile_bits=tile_bits)

    def guard_candidates(self, ranked, query_terms, index, budget=500, channel="token"):
        """RECALL GUARD: union a ranked candidate list with exact-containment TIERS from a
        perfect_recall_index (all-m-terms first, then m-1, ...) until `budget` fills -- returns
        (candidates, certificate) where the certificate states the coordination level down to which
        completeness is a GUARANTEE ('every doc sharing >= c query terms is present'), verified
        exhaustively in the module selftest. The ranked head is preserved untouched (the beauty pass;
        the guard is the coverage pass under it). MEASURED on real NFCorpus vs BM25 top-200: reachable
        relevant misses 1165 -> 732 at budget 1000; Recall@1000 0.278 -> 0.311. KEPT NEG: guards only
        lexically-reachable docs (zero-shared-term relevance needs the semantics arm); low tiers on
        common terms may exceed budget -- the certificate then admits less rather than lying; ranking
        quality inside the list is unchanged by design. See holographic_recallguard.guard_candidates."""
        from holographic.semantic_router.holographic_recallguard import guard_candidates as _g
        return _g(ranked, query_terms, index, budget=budget, channel=channel)

    def retrieval_dispatch(self, query, docs, dense_scores=None, k=5, tau=0.25, shortlist=32):
        """ADAPTIVE retrieval cascade -- the render-strategy dispatcher pointed at search: exact-phrase
        short-circuit (perfect recall, cost ~0) -> dense arm gated on the top-1/top-2 margin (stop when the
        answer is PROVEN, like adaptive path tracing) -> BM25 as a LAST-PASS denoise fit over the dense
        shortlist ONLY (O(shortlist), never the corpus) fused dense-dominant by RRF -> honest abstain when
        both arms are flat. This replaces the run-both-arms-always hybrid: BM25 runs only when the dense
        margin is narrow, and then only over the ambiguous window. Pass dense_scores from route_semantic /
        find_capability; None uses a token-overlap fallback. tau=1.0 forces refine always (the old behavior);
        tau=0.0 never refines. Returns {'ranked','stage','margin','shortlist_size'}. KEPT NEG: refine cannot
        rescue gold OUTSIDE the shortlist (widen shortlist, not the lexical weight), and a confidently-wrong
        dense top-1 passes the gate un-refined (lower tau where wrongness is costly). See
        holographic_retrievaldispatch.dispatch_retrieval."""
        from holographic.semantic_router.holographic_retrievaldispatch import dispatch_retrieval
        return dispatch_retrieval(query, list(docs), dense_scores=dense_scores, k=k, tau=tau,
                                  shortlist=shortlist)

    def fuse_rankings(self, ranked_lists, k=60, top=None, weights=None):
        """Reciprocal Rank Fusion (Cormack 2009): fuse several ranked id-lists into one by summing w/(k+rank).
        Uses only RANKS, so it needs no score calibration -- the right way to combine dense cosine (in [-1,1])
        with BM25 (unbounded), whose raw scores are not comparable. An item ranked well by MORE retrievers
        rises. `weights` (per-list multipliers, default equal) is the dense-dominance knob: SR-BETA measured
        that fusing a strong dense list with a weak BM25 one wants DENSE-DOMINANT weights ~(1.0, 0.3) -- that
        keeps every dense HIT (a spurious BM25 top never overtakes a dense-#1 gold at beta<=1) while rescuing
        a gold that dense ranked LOW but still returned; a gold ABSENT from the dense top-k needs beta>1 (the
        hard-conflict regime) and is better fixed by widening the retriever's k than by fusion. Returns fused
        [(item_id, score)] best-first. See holographic_bm25.reciprocal_rank_fusion."""
        from holographic.semantic_router.holographic_bm25 import reciprocal_rank_fusion
        return reciprocal_rank_fusion(list(ranked_lists), k=k, top=top, weights=weights)

    def workflow_graph(self, root=None, hub_frac=0.15):
        """The WORKFLOW adjacency: the sparse 'bones' of which modules actually work together, derived from the
        cross-references the authors already wrote in the docstrings (module A naming holographic_B). Edges are
        RARITY-weighted (a reference to a module few others mention is worth more) and true hubs are dropped,
        so the bones stay SPECIFIC -- measured median out-degree 2, versus the io-kind graph's 13-24 generic
        neighbors on the same modules. Cached per root. Use with workflow_neighbors / workflow_propagate. See
        holographic_workflowgraph.build_workflow_graph."""
        from holographic.semantic_router.holographic_workflowgraph import build_workflow_graph
        import os
        if root is None:
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cache = getattr(self, "_workflow_graph_cache", None)
        if cache is None:
            cache = self._workflow_graph_cache = {}
        key = (str(root), float(hub_frac))
        if key not in cache:
            cache[key] = build_workflow_graph(root, hub_frac=hub_frac)
        return cache[key]

    def workflow_neighbors(self, module, direction="both", top=None, root=None):
        """Which modules WORK WITH `module`, by author-stated cross-reference, best-first as [(module, weight)].
        'out' = modules it references, 'in' = modules referencing it, 'both' = the union. Sparse and specific
        (meshsmooth -> graphsignal; resonator -> chunkcodebook), unlike generic same-io-kind adjacency. See
        holographic_workflowgraph.neighbors."""
        from holographic.semantic_router.holographic_workflowgraph import neighbors
        return neighbors(self.workflow_graph(root=root), module, direction=direction, top=top)

    def workflow_propagate(self, seed_scores, alpha=0.5, top=None, root=None, graph=None):
        """Spread per-module scores ONE hop along the workflow bones: a module whose COLLABORATORS are strongly
        scored gets lifted even if its own text was never matched. The structural complement to dense/BM25 --
        it can surface a module the query has no words in common with. alpha weights propagation vs the seed;
        alpha=0 returns the seed unchanged. Returns [(module, score)] best-first. KEPT NEG: one hop only --
        multi-hop re-diffuses toward the smeared io-kind regime.

        `graph` (GS-C sweep): pass ANY directed weighted graph to spread scores over something other than the
        module bones (scene-selection growth, encyclopedia priming, any node set with weighted neighbours). Shape
        is workflowgraph's own: {"out": {node: [(nbr, w), ...]}, "in": {node: [(nbr, w), ...]}} -- lists of
        (neighbour, weight) pairs, not dicts. The kernel was already general; only this faculty was pinned to the
        module graph. Default None = the workflow graph, so every existing caller is unchanged. KEPT NEG (regime,
        do not "unify"): field inpainting and mesh diffusion_transfer are the same spreading idea under DIFFERENT
        discretizations (grid / mesh Laplacian); they are not clients of this pair-list-graph kernel. See
        holographic_workflowgraph.propagate."""
        from holographic.semantic_router.holographic_workflowgraph import propagate
        g = self.workflow_graph(root=root) if graph is None else graph
        return propagate(g, dict(seed_scores), alpha=alpha, top=top)

    def find_scored(self, problem, k=3):
        """Like find_capability, but returns [(capability, score)] -- so a caller can tell a HIT from a FALLBACK.

        This matters more than it looks. `find_capability` always returns its best three, even when the best is
        nothing: a query about a faculty the engine does not have still comes back with three confident-looking
        names. Twice in this program an audit concluded "absent" from a fallback and "present" from a coincidence,
        and both times the score would have said so immediately. A dominant top score means the engine really does
        solve this; a flat low-scoring list means it does not, whatever the names look like.

        Use `capability_confidence` for the one-number version. See holographic_catalog.Catalog.find_scored."""
        return self._capability_catalog().find_scored(problem, k=k)

    def capability_confidence(self, problem):
        """{top, score, margin, confident} -- how dominant the best capability match is. `margin` is the top score
        minus the runner-up's; `confident` is True when the top scores at all AND leads by at least 1.0.

        The honest answer to "does this engine already do X?" -- and the antidote to reading a fallback as a hit.
        Pair it with a symbol grep (tools/backlog_probe.py does both) before concluding anything is missing."""
        # DELEGATE the confident/margin judgment to the shared decide_or_abstain (min_score just above 0 so a
        # zero-scoring top never counts; margin=1.0 preserves the original 'leads by >= 1.0' rule).
        from holographic.misc.holographic_relations import decide_or_abstain
        scored = self._capability_catalog().find_scored(problem, k=2)
        if not scored:
            return {"top": None, "score": 0.0, "margin": 0.0, "confident": False}
        ranked = [(c.name, s) for c, s in scored]
        s0 = ranked[0][1]
        s1 = ranked[1][1] if len(ranked) > 1 else 0.0
        _, _, confident = decide_or_abstain(ranked, margin=1.0, min_score=1e-12)
        return {"top": ranked[0][0], "score": float(s0), "margin": float(s0 - s1),
                "confident": bool(confident)}

    def suggest(self, task, k=5):
        """AGENT-FRIENDLY autocomplete: turn a plain-English task into the best capabilities to use, each with a
        CONFIDENCE (0..1) and the concrete call to make. Like find_capability, but scored + call-ready so an agent (or
        a person) can decide what to invoke. See holographic_skills.suggest."""
        import holographic.misc.holographic_skills as _sk
        return _sk.suggest(task, k=k)

    def route(self, task):
        """AGENT-FRIENDLY decision node: when one skill clearly wins, returns {'decision':'act', 'skill':..., 'call'}
        so the agent just does it; when it's ambiguous, returns {'decision':'choose', 'options':[...]} so it asks
        instead of guessing. 'Act when confident, ask when not', score-based. See holographic_skills.route."""
        import holographic.misc.holographic_skills as _sk
        return _sk.route(task)

    def describe_skill(self, name):
        """A machine-readable SKILL CARD for a capability or a UnifiedMind method by name: what it does + how to CALL
        it (real signature for methods). The 'skill description' an agent reads before invoking. See holographic_skills."""
        import holographic.misc.holographic_skills as _sk
        return _sk.skill_card(name)

    def complete_method(self, prefix, k=15):
        """Method-name AUTOCOMPLETE: this mind's methods starting with `prefix`, each with its signature -- what an
        agent (or an IDE) offers while constructing a `mind.<prefix...` call. See holographic_skills.complete."""
        import holographic.misc.holographic_skills as _sk
        return _sk.complete(prefix, k=k)

    def skills(self, include_methods=True):
        """The full machine-readable SKILL MANIFEST: every curated capability home plus every public method (with its
        signature). What an agent loads ONCE to know the whole surface it can drive. See holographic_skills.manifest."""
        import holographic.misc.holographic_skills as _sk
        return _sk.manifest(include_methods=include_methods)

    def _capability_catalog(self):
        """Lazily build (once) and cache the capability catalog: the curated homes + this mind's faculties + EVERY
        engine module (by docstring), so a problem description can surface anything built -- nothing stays buried."""
        cat = getattr(self, "_catalog_cache", None)
        if cat is None:
            from holographic.caching_and_storage.holographic_catalog import default_catalog, seed_from_mind, seed_from_modules
            cat = seed_from_modules(seed_from_mind(default_catalog(), self))
            self._catalog_cache = cat
        return cat

    def composite_layers(self, layers, meta=None, background=None):
        """Composite a layer stack into one image -- the SHARED blend kernel (L-1).
        Compositing is the one operation every image-consuming app must perform IDENTICALLY, and it existed only
        inside leStudio: ten modes in that app's own __init__.py, nothing in the engine. Any second app reading a
        shared workspace had to re-implement all ten plus the alpha-over loop, and TWO COPIES OF THE SAME MATHS
        DRIFT -- the same document then renders differently in the modeller than in the painter, which is the exact
        failure the shared container format was built to prevent.
        `normal` stays in agreement by luck; the nine others are where copies diverge, because each is a one-line
        formula four people will round, clamp and order differently. softlight here is the W3C form, not the cheap
        approximation -- they differ visibly in the dark end, and A SHARED KERNEL THAT IS ALMOST THE SAME IS WORSE
        THAN NONE, because the discrepancy is unattributable.
        Reads the layer records exactly as a container section carries them: visible / opacity / blend / order /
        mask. Two contracts pinned by selftest because both are commonly got wrong: OPACITY FADES RATHER THAN
        DARKENS (scaling colour sends a half-opacity white layer over white to grey), and a multiply layer over
        transparency is ITSELF rather than black (which is what compositing before blending gets wrong).
        See holographic_composite.BLEND_MODES."""
        from holographic.materials_and_texture.holographic_composite import (
            composite_layers)
        return composite_layers(layers, meta=meta, background=background)

    def live_session(self, name="session", ttl=30.0):
        """A revision counter, presence and a change feed, owned by NEITHER app (L-2).
        leStudio's multiplayer is app-local -- a rev bumped by a Flask after_request hook, an SSE feed, presence
        defined as an open stream -- all inside one app's web server. So file-level sharing works today (both apps
        read and write one container) while LIVE CO-EDITING ACROSS TWO APPS CANNOT EXIST, because the second app
        would have to import the first one's Flask app to join.
        RULED OUT FIRST, as the backlog asks: WorkspaceManager is not this. It checkpoints a live DB's scratch
        tables by replay; coordinating several concurrent editors is a different problem, and the container
        module's own docstring already says so.
        TRANSPORT-AGNOSTIC IS THE POINT AND THE HARD PART TO HOLD: no socket, no SSE, no thread, no Flask. The
        moment this imports a web framework it becomes leStudio's implementation with a different filename and the
        second app is locked out again. Each app drives it with the transport it already has.
        bump(src) -> a monotonic revision; since(rev, exclude=src) -> the feed from a client's last-seen point,
        minus its own echo; participants() -> presence as a HEARTBEAT WITH A TIMEOUT rather than an open stream,
        so a polling client counts and a wedged process with an open socket does not.
        See holographic_livesession.LiveSession."""
        from holographic.io_and_interop.holographic_livesession import (
            LiveSession)
        return LiveSession(name=name, ttl=ttl)

    def container_kinds(self, kind=None, describe="", container=None):
        """Which section kinds this build understands (L-3), and the canonical image kind (L-4).
        save_container accepted ANY kind string and nothing could tell you which ones a build knew, so an app
        hardcoded the kinds it recognised and could not tell a user WHY a section was inert -- it carried them
        silently. Now a UI can say "this file also contains 3 polystudio.object sections (not editable here)".
        AND L-4: `lecore.image` is the canonical texture section -- RGBA float 0..1 straight alpha plus
        colour_space and dpi. Without one name, each app PAIR needs its own adapter: N^2 adapters for N apps,
        which is why leStudio writes lestudio.document and a modeller writes polystudio.texture and neither can
        read the other's picture. One documented kind makes it N.
        See holographic_container.register_kind / known_kinds / image_section."""
        from holographic.io_and_interop import holographic_container as C
        if container is not None:
            return C.describe_sections(container)
        if kind is not None:
            return C.register_kind(kind, describe)
        return C.known_kinds()

    def register_capability(self, name, does, example="", native=True, aliases=(),
                            consumes=(), produces=(), method=None, semantic=None,
                            polymorphic=False, module=None):
        """Register a capability in the catalog so future `find_capability` calls surface it (backlog C1: as each
        consolidation home lands, register it here). Additive; returns the entry.

        REGISTRATION IS THE TAGGING API. The catalog's register_capability has
        taken `consumes`/`produces`/`method` all along; THIS WRAPPER DROPPED
        THEM, so anything registered through the mind arrived untagged and
        invisible to suggest_pipeline. That is most of why io-kind coverage sat
        at 110 of 3,239 -- not because tagging was hard, but because the door
        everyone registers through had no keyhole for it.
        `consumes`/`produces` are io kinds (see io_kinds()); `method` is the
        UnifiedMind attribute a planner should call, which is what makes a
        proposed step executable rather than prose."""
        return self._capability_catalog().register_capability(
            name, does, example=example, native=native, aliases=aliases,
            consumes=consumes, produces=produces, method=method,
            semantic=semantic, polymorphic=polymorphic, module=module)

    def scene_to_render(self, scene, default_material="matte_gray"):
        """Flatten a Scene document to (sdf, material_fn) for the path tracer, without rendering -- the bridge
        render_scene_document uses. Useful when you want the scene's SDF/material to hand to a custom render
        (dispersion, caustics, a preview session). See holographic_scene_render.scene_to_render."""
        from holographic.rendering.holographic_scene_render import scene_to_render
        return scene_to_render(scene, default_material=default_material)

    def render_demodulated_upscale(self, sdf, camera, low_wh, high_wh, material_fn, sky=None, quality="medium",
                                   max_bounce=3, seed=0, lights=None):
        """M5 -- render a HIGH-resolution frame at LOW-resolution lighting cost. Render the expensive lighting at
        `low_wh`=(w,h), read the cheap high-res G-buffer at `high_wh`, and combine by demodulation: upscale the
        smooth irradiance, multiply the crisp high-res albedo back (holographic_modulate). ~2.6x faster than a full
        high-res render and cleaner than a plain upscale ON TEXTURED surfaces (the detail lives in the albedo).
        Kept negative: neutral on uniform-albedo (no texture to restore); diffuse only."""
        from holographic.misc.holographic_modulate import render_demodulated_upscale
        return render_demodulated_upscale(sdf, camera, low_wh, high_wh, material_fn, sky=sky, quality=quality,
                                          max_bounce=max_bounce, seed=seed, lights=lights)

    def fluid_solver(self, shape, **kwargs):
        """A grid-based STABLE-FLUIDS solver (Stam 1999) for smoke, buoyant plumes, and combustion/FIRE -- the
        method professional smoke engines (Houdini, Bifrost Aero) are built on. Incompressibility is enforced by
        an FFT pressure projection (a Helmholtz-Hodge decomposition = the engine's periodic circular-convolution
        algebra), and advection is unconditionally-stable semi-Lagrangian. Returns a holographic_fluid.StableFluid
        carrying velocity + smoke density + temperature + fuel; call .step() to advance, .add_source() to emit.
        MEASURED: divergence-free to machine precision; 128^2 ~10ms/step, 64^3 ~0.5s/step (offline NumPy BRAIN,
        NOT GPU-realtime -- the method matches the pros, the throughput does not); vorticity confinement keeps
        ~88x more swirl. KEPT NEGATIVE: semi-Lagrangian advection is dissipative (~20% smoke mass lost over 60
        steps to interpolation; a MacCormack/BFECC or FLIP scheme conserves better); boundaries are periodic."""
        from holographic.simulation_and_physics.holographic_fluid import StableFluid
        return StableFluid(shape, **kwargs)

    def delta_chain(self, base, tol=0.0, codebook=None):
        """A chunked DELTA CHAIN (DELTA-1): store a SEQUENCE of (N, D) chunks as a base + per-chunk deltas, each
        delta taken against the BASE or the PRIOR chunk -- whichever is smaller -- so memory is O(actual change).
        A SHA-256 hash chain + Merkle root make integrity PROVABLE: append() folds each chunk into the chain,
        get(i) reconstructs AND verifies (a corrupted delta / wrong base / broken propagation raises
        IntegrityError), root() is the one 'fractal' proof of the whole sequence. Bit-exact, deterministic,
        vectorized (no per-element Python on the data). With `codebook` set, changed rows that EXACTLY equal an
        atom store an index, not a float row -- lossless, and its size win scales with D and sequence length
        (measured 86x on the delta portion at D=256). HONEST: exact integrity is hashlib, not a (lossy) VSA
        bundle -- the case where VSA-native is NOT beneficial. See holographic_deltachain.DeltaChain."""
        from holographic.agents_and_reasoning.holographic_deltachain import DeltaChain
        return DeltaChain(base, tol=tol, codebook=codebook)

    def execution_replay(self, program, chunk=14, init_acc=None):
        """Run a long program CHUNKED and record a verifiable, O(change) REPLAY LOG (DELTA-1 x WIRE-2): each
        chunk's full machine state (acc + registers + stack, as rows) is appended to a DeltaChain, so the
        execution can be audited, RESUMED from any seam, and integrity-checked. This is where the run_chunked
        state-threading and the delta chain meet: consecutive seam states share most rows (registers that didn't
        change), so the log is O(actual change), and the hash chain proves it wasn't tampered with. Returns
        (acc, trace, replay) -- replay a DeltaChain whose chunk i is the state after seam i+1 (its base is the
        first seam's state), or None if the program produced no seams."""
        M = self._machine()
        acc, trace, states = M.run_chunked(program, chunk=chunk, init_acc=init_acc,
                                           handlers=self._procedure_handlers(), record=True)
        if not states:
            return acc, trace, None
        from holographic.agents_and_reasoning.holographic_deltachain import DeltaChain
        replay = DeltaChain(states[0])
        for s in states[1:]:
            replay.append(s)
        return acc, trace, replay

    def nystrom_field(self, points, sources, weights, sigma, m=None):
        """Approximate a kernel-weighted field f(p) = sum_j weights[j]*K(p, sources[j]) (Gaussian RBF) via m
        LANDMARK sources -- O((Np+Ns)*m) instead of the exact O(Np*Ns), for LARGE memory or physics sims
        (sources=particles+charges, or stored items+payload; points=where you sample). MEASURED ~13x at N=2000
        on a smooth field, the win growing with N. KEPT NEGATIVE: exact only for a LOW-RANK (smooth) field; a
        high-frequency field is full-rank and the landmark approximation degrades (corr ~0.2). See
        holographic_nystrom.nystrom_kernel_apply."""
        from holographic.sampling_and_signal.holographic_nystrom import nystrom_kernel_apply
        return nystrom_kernel_apply(points, sources, weights, sigma, m=m)

    def consolidate_subspace(self, memories, k=8, landmarks=None):
        """The consolidated low-rank SUBSPACE of stored memories (top-k principal directions) + mean. With
        `landmarks`=m, approximate it from m farthest-point memories instead of all N (the Nystrom sketch for a
        LARGE store). Returns (basis, mean). See holographic_dream.dream_subspace."""
        from holographic.agents_and_reasoning.holographic_dream import dream_subspace
        return dream_subspace(memories, k=k, landmarks=landmarks)

    def dream(self, basis, mean, n=8, seed=0, noise=1.0, codebook=None):
        """DREAM = generative replay over the consolidated subspace: draw noise, project onto the subspace (the
        manifold denoiser run from noise), optionally clean -> samples ON the manifold (valid) yet NOVEL (not a
        stored item). Over the consolidated (composed) subspace this yields novel COMPOSITIONS, the interesting
        regime B10 flagged. Returns an (n, D) array. See holographic_dream.dream."""
        from holographic.agents_and_reasoning.holographic_dream import dream
        return dream(basis, mean, n=n, seed=seed, noise=noise, codebook=codebook)

    def holographic_value_head(self, n_actions, dim=None, routed=False, n_buckets=64):
        """The creature's value/policy AS a pure-VSA program. Returns a HolographicValueHead (or, with
        routed=True, a RoutedValueHead whose routing fabric pushes the capacity cliff back ~n_buckets-fold):
        a drop-in for the creature's value backend (same value(state, action)->(value, support) and
        absorb(state, action, ret) API), but the whole per-action policy is bundles -- Q_a (return-weighted
        state superposition) and N_a (the normaliser) -- so value(s,a) = <s,Q_a>/<s,N_a> reproduces the
        brain's Nadaraya-Watson average while learning is one bundling step and the policy is a fixed-size,
        savable, COMPOSABLE hypervector ({Q, N}, or policy_atom() folded into two bindable vectors) instead of
        a growing (vector, scalar) table. KEPT NEGATIVE: a single bundle pair has finite capacity (matches the
        tabular brain at low load, degrades past ~dim distinct situations); routing trades n_buckets-fold
        memory to push that cliff back."""
        d = self.dim if dim is None else dim
        if routed:
            from holographic.agents_and_reasoning.holographic_valuehead import RoutedValueHead
            return RoutedValueHead(d, n_actions, n_buckets=n_buckets)
        from holographic.agents_and_reasoning.holographic_valuehead import HolographicValueHead
        return HolographicValueHead(d, n_actions)

    def fast_creature_encoder(self, dim=None, seed=1):
        """Compiled, fully in-VSA perception for the creature loop: a FastCreatureEncoder whose per-step
        role/filler binds (FFT convolutions) are precomputed once into a codebook, so perceiving recurring
        senses is a gather+sum -- no per-step FFT. Bit-identical to the plain CreatureEncoder, ~8x faster at
        steady state. With the holographic value head (decide=dot, learn=bundle) the whole perceive->decide->
        learn loop becomes array ops, keeping the creature inside the holographic space (no Python<->VSA
        round-trip per step). `perception_codebook()` exposes the compiled (features, dim) matrix."""
        from holographic.misc.holographic_creature import FastCreatureEncoder
        return FastCreatureEncoder(self.dim if dim is None else dim, seed=seed)

    def soft_body(self, positions, inv_mass=None, velocities=None):
        """A PBD/XPBD softbody: particles + distance constraints, time-stepped under gravity/forces. Inverse
        mass 0 pins a particle. The constraint sweep is the same iterate-a-projection engine as the resonator/
        denoiser/IK; XPBD adds time-step-independent stiffness. See holographic_softbody.SoftBody."""
        from holographic.simulation_and_physics.holographic_softbody import SoftBody
        return SoftBody(positions, inv_mass=inv_mass, velocities=velocities)

    def mesh_to_softbody(self, mesh, compliance=0.0, pin=None):
        """Turn ANY mesh into a simulatable SoftBody (vertices -> particles, edges -> distance constraints) so a
        PROJECTED surface mesh can be driven by the whole physics layer: gravity, fluid drag (drag_force_3d as
        external_force), self-collision (add_self_collision), the constraint solver. The bridge that lets the
        mesh pipeline take advantage of the physics -- sculpt -> surface_mesh -> mesh_to_softbody -> simulate.
        A surface mesh is a shell (behaves like cloth); `pin` anchors vertices. See SoftBody.from_mesh."""
        from holographic.simulation_and_physics.holographic_softbody import SoftBody
        return SoftBody.from_mesh(mesh, compliance=compliance, pin=pin)

    def dynamics_to_mesh(self, source, bounds=None, radius=0.5, level=0.5, resolution=48,
                         axes=None, face_type="triangle"):
        """Export ANY dynamics state as a watertight mesh -- the unified surfacing entry point. `source` is one of:
          * a SoftBody or RigidBody built via from_mesh -> its current (deformed/moved) positions + faces
            (source.to_mesh()); the soft/rigid case.
          * a point cloud (N,3 array) -- particles or a LIQUID/SPH front -> surfaced by wrapping the points in a
            metaball field (sum of Gaussians, `radius`) and marching its `level` isosurface over `bounds`
            (= surface_mesh_stable); the particle/liquid case.
          * a (density_grid, axes) pair OR a grid with `axes=` given -- a SMOKE/volume density field -> its
            `level` isosurface marched directly. The smoke case.
        `face_type` ('triangle'|'quad'|'ngon') applies the meshpoly merge to the result. Returns a Mesh (or, for
        a point cloud / field, the surface_mesh_stable dict via the field path). This keeps everything on the
        engine's own field<->mesh bridge -- particles are a metaball field, smoke is a field, a soft body is the
        mesh it was built from, all surfaced by the same marching tetrahedra."""
        from holographic.simulation_and_physics.holographic_softbody import SoftBody, RigidBody
        from holographic.mesh_and_geometry.holographic_meshbridge import metaball_field, marching_tetrahedra_vec
        import numpy as _np
        if isinstance(source, (SoftBody, RigidBody)):          # soft / rigid: re-export the carried faces
            mesh = source.to_mesh()
            return self.mesh_face_type(mesh, face_type) if face_type != "triangle" else mesh
        if isinstance(source, tuple) and len(source) == 2:     # (density_grid, axes): smoke/volume isosurface
            values, ax = source
            mesh = marching_tetrahedra_vec(_np.asarray(values, float), ax, level=level)
            return self.mesh_face_type(mesh, face_type) if face_type != "triangle" else mesh
        arr = _np.asarray(source, float)
        if axes is not None:                                   # a bare density grid + axes: smoke
            mesh = marching_tetrahedra_vec(arr, axes, level=level)
            return self.mesh_face_type(mesh, face_type) if face_type != "triangle" else mesh
        if arr.ndim == 2 and arr.shape[1] in (2, 3):           # a point cloud: particles / liquid -> metaball surface
            if bounds is None:
                lo = arr.min(0) - 3 * radius; hi = arr.max(0) + 3 * radius
                bounds = (lo, hi)
            field = metaball_field(arr, radius=radius)
            return self.surface_mesh_stable(field, bounds, resolution=resolution, level=level, face_type=face_type)
        raise ValueError("source must be a SoftBody/RigidBody, a point cloud (N,3), or (density_grid, axes)")

    def cloth(self, rows, cols, spacing=1.0, compliance=0.0):
        """A rectangular cloth softbody (structural + shear distance constraints, top row pinned)."""
        from holographic.simulation_and_physics.holographic_softbody import SoftBody
        return SoftBody.cloth(rows, cols, spacing=spacing, compliance=compliance)

    def cloth3d(self, rows, cols, spacing=1.0, compliance=0.0, bending=None):
        """A 3-D cloth that drapes under gravity; pass `bending` (a compliance) to add bend springs that
        resist folding so the sheet stays flatter. See holographic_softbody.SoftBody.cloth3d."""
        from holographic.simulation_and_physics.holographic_softbody import SoftBody
        return SoftBody.cloth3d(rows, cols, spacing=spacing, compliance=compliance, bending=bending)

    def soft_box(self, nx, ny, nz, spacing=1.0, compliance=0.0, volume_compliance=0.0):
        """A soft 3-D solid (a lattice of tetrahedra with volume constraints) that resists being squashed and
        springs back. See holographic_softbody.SoftBody.soft_box."""
        from holographic.simulation_and_physics.holographic_softbody import SoftBody
        return SoftBody.soft_box(nx, ny, nz, spacing=spacing, compliance=compliance,
                                 volume_compliance=volume_compliance)

    def rope(self, n, spacing=1.0, compliance=0.0, start=(0.0, 0.0)):
        """A hanging rope softbody of n particles (particle 0 pinned)."""
        from holographic.simulation_and_physics.holographic_softbody import SoftBody
        return SoftBody.rope(n, spacing=spacing, compliance=compliance, start=start)

    def rigid_body(self, positions, inv_mass=None, velocities=None):
        """A hardbody via shape matching (polar decomposition): falls and rotates under forces but never
        deforms. See holographic_softbody.RigidBody."""
        from holographic.simulation_and_physics.holographic_softbody import RigidBody
        return RigidBody(positions, inv_mass=inv_mass, velocities=velocities)

    def diffuse_field(self, field, amount):
        """Diffuse a scalar/component grid field by the heat kernel via the FFT -- a Gaussian bind on the
        torus (the engine's own operator). Conserves mass. See holographic_fields."""
        from holographic.misc.holographic_fields import diffuse
        return diffuse(field, amount)

    def make_incompressible(self, vx, vy):
        """Pressure-project a velocity field to divergence-free (the fluid solver's Helmholtz step, an FFT
        solve). Returns (vx, vy) with divergence ~0. See holographic_fields.project_divergence_free."""
        from holographic.misc.holographic_fields import project_divergence_free
        return project_divergence_free(vx, vy)

    def field_divergence(self, vx, vy):
        """The divergence field of a velocity grid (compression/expansion), computed spectrally."""
        from holographic.misc.holographic_fields import divergence
        return divergence(vx, vy)

    def advect_field(self, field, vx, vy, dt=0.1):
        """Transport a scalar field (density/temperature) along a velocity field (semi-Lagrangian)."""
        from holographic.misc.holographic_fields import advect
        return advect(field, vx, vy, dt)

    def fluid_step(self, vx, vy, density, dt=0.1, viscosity=0.0, fx=None, fy=None, source=None, solid=None,
                   boundary="wrap"):
        """One Stable-Fluids step on the torus (add force -> diffuse -> project -> advect), built on the FFT.
        Returns (vx, vy, density). Pass `solid` (a 0/1 mask) for an obstacle the flow goes around. See
        holographic_fields.fluid_step."""
        from holographic.misc.holographic_fields import fluid_step
        return fluid_step(vx, vy, density, dt=dt, viscosity=viscosity, fx=fx, fy=fy, source=source, solid=solid, boundary=boundary)

    def smoke_step(self, vx, vy, density, temperature, dt=0.1, viscosity=0.0, ambient=0.0,
                   buoyancy=1.0, gravity=0.0, confinement=0.0, dens_source=None, temp_source=None, solid=None):
        """One smoke step: temperature drives velocity by buoyancy (hot rises), vorticity confinement keeps it
        curly, density + temperature advect with the flow. Pass `solid` for an obstacle smoke flows around.
        Returns (vx, vy, density, temperature). See holographic_fields.smoke_step."""
        from holographic.misc.holographic_fields import smoke_step
        return smoke_step(vx, vy, density, temperature, dt=dt, viscosity=viscosity, ambient=ambient,
                          buoyancy=buoyancy, gravity=gravity, confinement=confinement,
                          dens_source=dens_source, temp_source=temp_source, solid=solid)

    def fluid_step_3d(self, vx, vy, vz, density, dt=0.1, viscosity=0.0, fx=None, fy=None, fz=None, source=None,
                      solid=None):
        """One 3-D Stable-Fluids step on a 3-D periodic grid (the same FFT solver, generalised via the n-D real
        FFT). Pass `solid` (a 3-D mask, e.g. from sphere_mask) for an obstacle the flow goes around. Returns
        (vx, vy, vz, density). See holographic_fields.fluid_step_3d."""
        from holographic.misc.holographic_fields import fluid_step_3d
        return fluid_step_3d(vx, vy, vz, density, dt=dt, viscosity=viscosity, fx=fx, fy=fy, fz=fz,
                             source=source, solid=solid)

    def smoke_step_3d(self, vx, vy, vz, density, temperature, dt=0.1, viscosity=0.0, ambient=0.0,
                      buoyancy=1.0, gravity=0.0, confinement=0.0, dens_source=None, temp_source=None,
                      solid=None):
        """One 3-D smoke step: buoyancy lifts hot fluid along +y, 3-D vorticity confinement keeps it curly,
        density + temperature advect. Pass `solid` for a 3-D obstacle smoke rises and curls around. Returns
        (vx, vy, vz, density, temperature). See holographic_fields.smoke_step_3d."""
        from holographic.misc.holographic_fields import smoke_step_3d
        return smoke_step_3d(vx, vy, vz, density, temperature, dt=dt, viscosity=viscosity, ambient=ambient,
                             buoyancy=buoyancy, gravity=gravity, confinement=confinement,
                             dens_source=dens_source, temp_source=temp_source, solid=solid)

    def make_incompressible_3d(self, vx, vy, vz):
        """Pressure-project a 3-D velocity field to divergence-free (the 3-D FFT Helmholtz solve). Returns
        (vx, vy, vz). See holographic_fields.project_divergence_free_3d."""
        from holographic.misc.holographic_fields import project_divergence_free_3d
        return project_divergence_free_3d(vx, vy, vz)

    def field_divergence_3d(self, vx, vy, vz):
        """The 3-D divergence field of a velocity grid, computed spectrally."""
        from holographic.misc.holographic_fields import divergence_3d
        return divergence_3d(vx, vy, vz)

    def spectral_field(self, shape, beta=2.0, seed=0):
        """Synthesise a SEAMLESS FRACTAL volume (2-D or 3-D) in the Fourier domain -- a 1/f^beta procedural
        field that tiles with no seam (periodic by construction) and is reproducible from just (shape, beta,
        seed). The demoscene 'rich volume from a tiny seed' move on the engine's FFT; composes with tile_field
        / the fluid solver / the archive. See holographic_fields.spectral_field."""
        from holographic.misc.holographic_fields import spectral_field
        return spectral_field(shape, beta=beta, seed=seed)

    def seam_continuity(self, field, axis=0):
        """How seamlessly a field tiles along `axis` (~1.0 = seamless, >>1 = visible seam). See
        holographic_fields.seam_continuity."""
        from holographic.misc.holographic_fields import seam_continuity
        return seam_continuity(field, axis=axis)

    def disc_mask(self, shape, center, radius):
        """A circular solid-obstacle mask (1 inside the disc) for fluid_step/smoke_step `solid`. See
        holographic_fields.disc_mask."""
        from holographic.misc.holographic_fields import disc_mask
        return disc_mask(shape, center, radius)

    def enforce_solid(self, vx, vy, solid_mask, solid_vx=0.0, solid_vy=0.0, iters=2):
        """Force the flow to respect a solid obstacle (velocity -> solid velocity inside the mask, then
        re-project so the flow diverts around). See holographic_fields.enforce_solid."""
        from holographic.misc.holographic_fields import enforce_solid
        return enforce_solid(vx, vy, solid_mask, solid_vx=solid_vx, solid_vy=solid_vy, iters=iters)

    def sphere_mask(self, shape, center, radius):
        """A spherical solid-obstacle mask (1 inside the ball) for fluid_step_3d/smoke_step_3d `solid` -- the
        3-D lift of disc_mask. See holographic_fields.sphere_mask."""
        from holographic.misc.holographic_fields import sphere_mask
        return sphere_mask(shape, center, radius)

    def enforce_solid_3d(self, vx, vy, vz, solid_mask, solid_vx=0.0, solid_vy=0.0, solid_vz=0.0, iters=2):
        """The 3-D immersed boundary: force flow to the solid's velocity inside the mask, then re-project so it
        diverts around the ball. See holographic_fields.enforce_solid_3d."""
        from holographic.misc.holographic_fields import enforce_solid_3d
        return enforce_solid_3d(vx, vy, vz, solid_mask, solid_vx=solid_vx, solid_vy=solid_vy,
                                solid_vz=solid_vz, iters=iters)

    def spatial_hash_pairs(self, positions, radius):
        """Find all index pairs within `radius` via a uniform-grid cull -- O(N) expected vs O(N^2), the
        'cull, don't batch' primitive (any dimension). Reusable for collision, particle interaction, patch
        matching. See holographic_fields.spatial_hash_pairs."""
        from holographic.misc.holographic_fields import spatial_hash_pairs
        return spatial_hash_pairs(positions, radius)

    def game_shard(self, dt=1.0/60.0, seed=0, gravity=(0.0, 0.0, 0.0), region=None,
                   restitution=0.2):
        """An authoritative, deterministic fixed-tick GAME WORLD shard: player commands in, ticks,
        AOI snapshots, deltas, region handoff for massive-world sharding across the farm. Use when
        building a game/multiplayer simulation server on the engine. Delegates entirely to
        holographic_gameshard.GameShard (default-off: nothing runs until you step it)."""
        from holographic.simulation_and_physics.holographic_gameshard import GameShard
        return GameShard(dt=dt, seed=seed, gravity=gravity, region=region,
                         restitution=restitution)

    def run_game_shard(self, commands, ticks, dt=1.0/60.0, seed=0, gravity=(0.0, 0.0, 0.0),
                       region=None, restitution=0.2, state=None, aoi=None):
        """JSON-in/JSON-out one-shot game-world run: feed commands (+ optional saved state), get the
        final state blob, per-tick lockstep digests, departures, and an optional AOI snapshot. The
        /invoke-callable face of game_shard(). See holographic_gameshard.run_shard."""
        from holographic.simulation_and_physics.holographic_gameshard import run_shard
        return run_shard(commands, ticks, dt=dt, seed=seed, gravity=gravity, region=region,
                         restitution=restitution, state=state, aoi=aoi)

    def game_world(self, cell=64.0, dt=1.0/60.0, seed=0, gravity=(0.0, 0.0, 0.0), restitution=0.2):
        """A MASSIVE sharded game world: a lazy grid of game shards with deterministic cross-shard
        entity migration, seam-free cross-shard AOI snapshots, a world-level lockstep digest, and the
        collect_only/receive() bus-transport seam for spreading shards across the farm. See
        holographic_gameshard.ShardWorld."""
        from holographic.simulation_and_physics.holographic_gameshard import ShardWorld
        return ShardWorld(cell=cell, dt=dt, seed=seed, gravity=gravity, restitution=restitution)

    def run_game_world(self, commands, ticks, cell=64.0, dt=1.0/60.0, seed=0,
                       gravity=(0.0, 0.0, 0.0), restitution=0.2, aoi=None):
        """JSON one-shot over the sharded world: spawn/drive entities, get per-tick world digests,
        every cross-shard migration, shard count, and an optional cross-shard AOI snapshot. The
        /invoke-callable face of game_world(). See holographic_gameshard.run_world."""
        from holographic.simulation_and_physics.holographic_gameshard import run_world
        return run_world(commands, ticks, cell=cell, dt=dt, seed=seed, gravity=gravity,
                         restitution=restitution, aoi=aoi)

    def game_bus_host(self, bus, world, own_keys, world_id="w0"):
        """Host one node's slice of a sharded game world on the EXISTING message/distributed bus:
        owns a set of cell keys, publishes handoffs on per-cell topics, receives arrivals via
        subscribe -- the interaction layer's handshake with the data layer (bus/coordinator/presence),
        duplicating none of it. Rounds are barriered: publish in round R, join at R+1. See
        holographic_gameshard.BusShardHost."""
        from holographic.simulation_and_physics.holographic_gameshard import BusShardHost
        return BusShardHost(bus, world, own_keys, world_id=world_id)

    def buoyancy_force(self, temperature, density=None, alpha=0.0, beta=1.0, ambient=0.0):
        """Boussinesq buoyancy force from a temperature (and optional density) field -- hot rises, heavy sinks.
        Returns (fx, fy). See holographic_fields.buoyancy_force."""
        from holographic.misc.holographic_fields import buoyancy_force
        return buoyancy_force(temperature, density=density, alpha=alpha, beta=beta, ambient=ambient)

    def vorticity_confinement(self, vx, vy, epsilon=0.5):
        """Vorticity confinement force (Fedkiw 2001) -- restores the small vortices semi-Lagrangian advection
        damps, keeping smoke curly. Returns (fx, fy). See holographic_fields.vorticity_confinement."""
        from holographic.misc.holographic_fields import vorticity_confinement
        return vorticity_confinement(vx, vy, epsilon=epsilon)

    def particle_system(self, positions, velocities=None):
        """A ParticleSystem on the grid: particles feel forces (gravity, attractors, any (N,2) force a VSA
        program supplies) and can ride a solved velocity field. See holographic_fields.ParticleSystem."""
        from holographic.misc.holographic_fields import ParticleSystem
        return ParticleSystem(positions, velocities)

    def attractor_force(self, positions, center, strength=1.0, softening=1.0):
        """A force pulling particles toward a point (negative strength repels). See holographic_fields."""
        from holographic.misc.holographic_fields import attractor_force
        return attractor_force(positions, center, strength=strength, softening=softening)

    def pairwise_repulsion(self, positions, radius, strength=1.0):
        """Short-range particle-particle repulsion (the n-body short-range force), CULLED by spatial_hash_pairs
        so it is O(N + pairs) not O(N^2). Returns an (N, D) force array to pass to particle_system.step(force=)
        -- granular piles, collision avoidance, flocking separation. See holographic_fields.pairwise_repulsion."""
        from holographic.misc.holographic_fields import pairwise_repulsion
        return pairwise_repulsion(positions, radius, strength=strength)

    def blue_noise_sample(self, radius, bounds, k=30, seed=0):
        """Poisson-disk (blue-noise) point sampling by Bridson dart-throwing: a maximal point set with every
        pair >= `radius` apart and the blue-noise spectrum (suppressed low frequencies, a ring at the spacing).
        The exclusion principle done right -- nearly matches adaptive matching-pursuit placement and beats random
        by ~3 dB on a fixed-budget splat fit, and is the gold standard for stippling / particle init / Monte
        Carlo. `bounds`=(min,max), any dimension. See holographic_sampling.poisson_disk_sample."""
        from holographic.sampling_and_signal.holographic_samplinghome import Sampling                 # via the Sampling home  consolidation R4
        return Sampling.poisson_disk(radius, bounds, k=k, seed=seed)

    def sample_field(self, field, positions):
        """Read a grid field at continuous particle positions (bilinear, periodic) -- how particles feel a
        VSA-encoded or solved field. See holographic_fields.sample_field."""
        from holographic.misc.holographic_fields import sample_field
        return sample_field(field, positions)

    def plan_stage_execution(self, stages, frames):
        """Decide bake-vs-compute PER pipeline stage (PW3), extending adaptive.plan_render's break-even to the
        stage level: static stages bake once reused across enough frames, dynamic (and unannotated) stages compute.
        Returns a per-stage plan with reasons. See holographic_stageplan."""
        from holographic.scene_and_pipeline.holographic_stageplan import plan_stages
        return plan_stages(stages, frames)

    def plan_pipeline_bakes(self, cfg, frames, registry=None, options=None, cache=None):
        """Compile a pipeline (PW2) then decide which of its stages to BAKE vs COMPUTE over `frames` frames (PW3) --
        the decision layer that tells PW1's bake_pipeline which stages are worth baking. Returns (pipeline, plan).
        See holographic_stageplan / holographic_pipecompile."""
        from holographic.scene_and_pipeline.holographic_stageplan import plan_stages
        pipe = self.compile_pipeline(cfg, registry=registry, options=options, cache=cache)
        return pipe, plan_stages(pipe.stages, frames)

    def diffuse_readout(self, field, amount, k):
        """Read out the matter model's LINEAR diffusion sub-step at any time k in ONE evaluation (PW4): diagonalise
        the diffusion bind in the Fourier basis and raise its transfer to the k-th power, instead of marching k
        steps (k may be fractional). Matches k marched diffusions exactly. See holographic_simreadout. Nonlinear
        advection/buoyancy/tension still march."""
        from holographic.misc.holographic_simreadout import diffuse_at
        return diffuse_at(field, amount, k)

    def diffuse_steady_state(self, field):
        """The closed-form limit of unbounded diffusion (PW4): the flat mean field -- every non-DC mode decays, the
        mean is preserved -- without marching to it. See holographic_simreadout."""
        from holographic.misc.holographic_simreadout import diffuse_limit
        return diffuse_limit(field)

    def compile_pipeline(self, cfg, registry=None, options=None, cache=None):
        """Compile a render/sim pipeline's plan ONCE per config (PW2): select+auto-include+toposort the stages,
        keyed by the config's content, so repeated frames reuse the ordered Pipeline with no re-planning. Reuses the
        content-addressed compile cache. See holographic_pipecompile."""
        from holographic.scene_and_pipeline.holographic_pipecompile import compiled_pipeline
        return compiled_pipeline(cfg, registry=registry, options=options, cache=cache)

    def run_pipeline(self, cfg, scene=None, seed=0, prev_frame=None, renderer=None, registry=None, options=None, cache=None):
        """Run a pipeline for `cfg`, compiling its plan on the first frame and reusing it thereafter (PW1/PW2) -- the
        everyday frame-loop entry point. Returns the final FrameState. See holographic_pipecompile."""
        from holographic.scene_and_pipeline.holographic_pipecompile import run_compiled
        return run_compiled(cfg, scene=scene, seed=seed, prev_frame=prev_frame, renderer=renderer,
                            registry=registry, options=options, cache=cache)

    def bake_view_lut(self, metallic=1.0, base_color=(1.0, 1.0, 1.0), res_view=16, res_rough=16, samples=8192, seed=0):
        """Pre-integrate the view-DEPENDENT specular (MC3): bake directional_albedo over a (view_cos, roughness)
        grid ONCE, so per-pixel specular reflectance is a bilinear LUT lookup instead of a hemisphere integral --
        the 'add a dimension' move that turns the last view-dependent axis into a query. Returns a ViewLUT with
        .sample(view_cos, roughness). See holographic_viewlut."""
        from holographic.rendering.holographic_viewlut import bake_view_lut
        return bake_view_lut(metallic=metallic, base_color=base_color, res_view=res_view,
                             res_rough=res_rough, samples=samples, seed=seed)

    def bake_material(self, material, lo, hi, res=24):
        """Bake a material's VIEW-INDEPENDENT channels into field lookups (MC2): a procedural texture becomes a
        trilinear grid sample (O(1) per hit, no field re-evaluation), constants stay folded (MC1). `lo,hi` are the
        object bounds to bake over. Returns a shade(points)->channels kernel of folds + lookups. See
        holographic_matbake. The remaining view-DEPENDENT channels (specular) are MC3's LUT."""
        from holographic.materials_and_texture.holographic_matbake import bake_material
        return bake_material(material, lo, hi, res=res)

    def compile_material(self, material, cache=None):
        """Compile a material's socket graph into ONE cached shade(points)->channels kernel (MC1): built once, keyed
        by the material's content spec, reused for every hit/instance/frame; constant channels are folded so only the
        procedural sockets re-resolve per hit. Reuses the content-addressed compile cache. See holographic_matcompile."""
        from holographic.materials_and_texture.holographic_matcompile import compiled_shader
        return compiled_shader(material, cache=cache)

    def scatter_surface(self, instance, sdf, bounds, count, scale=1.0, density=None, cell_size=0.25, seed=0):
        """Scatter instances (grass, rocks, barnacles...) onto ANY surface -- a scatter LAYER that emits geometry
        instead of colour, weighted by an optional density map, reusing emit_from_surface. Returns the placements
        (points, normals, per-placement bound vectors) + the bundled, region-queryable layer vector. See
        holographic_scatterlayer."""
        from holographic.rendering.holographic_scatterlayer import ScatterLayer
        layer = ScatterLayer(instance, count, scale=scale, density=density, cell_size=cell_size, seed=seed)
        return layer.apply(sdf, bounds)

    def scale_node(self, scene, lod_px=8.0):
        """A cosmic-scale summariser over a scene hierarchy: a parent carries the monoid-accumulated value of its
        children (mass SUMs, appearance BUNDLES) via the wired distribute_compute; zoom out reads the summary, zoom
        in descends. Same accumulation at every scale (atom -> ... -> galaxy). See holographic_scalenode."""
        from holographic.misc.holographic_scalenode import ScaleNode
        return ScaleNode(scene, mind=self, lod_px=lod_px)

    def make_mixture(self, shape, solvent_density=1.0, buoyancy=1.0, tension=0.0):
        """Create a Mixture -- the multi-channel matter model (smoke/dye/milk/oil-water are this with different
        dials). Add components with .add(name, field, density, diffusivity), then advance with matter_step. See
        holographic_mixture."""
        from holographic.misc.holographic_mixture import Mixture
        return Mixture(shape, solvent_density=solvent_density, buoyancy=buoyancy, tension=tension)

    def matter_step(self, mix, vx, vy, dt=0.1, drift_strength=0.0):
        """Advance a Mixture one step on ONE shared incompressible flow: blend density -> buoyancy -> a single
        fluid_step -> per-channel advect + diffuse (+ optional double-well tension and drift) -> renormalise. This
        delegates to the wired advect/diffuse/buoyancy_force/fluid_step; it is NOT a second solver. Returns the
        updated (vx, vy); the mixture is mutated in place. See holographic_mixture."""
        from holographic.misc.holographic_mixture import matter_step
        return matter_step(mix, vx, vy, dt=dt, drift_strength=drift_strength)

    def smoke_preset(self, name="rising", nx=48, ny=48, steps=40, dt=0.1, seed=0):
        """Run one of the six named SMOKE PRESETS on the wired FFT smoke solver and return its fields
        (density/temperature/vx/vy). The presets (rising, wispy, billow, heavy, still_room, stratified) are just
        dial settings -- smoke is the 1-channel, tension-0 corner of the matter model -- so this delegates to
        smoke_step, it does not add a solver. Use smoke_preset_names() for the list. See holographic_smokepresets."""
        from holographic.misc.holographic_smokepresets import simulate
        return simulate(name, nx=nx, ny=ny, steps=steps, dt=dt, seed=seed)

    def smoke_preset_names(self):
        """The available smoke preset names."""
        from holographic.misc.holographic_smokepresets import preset_names
        return preset_names()

    def scatter_to_field(self, shape, positions, values):
        """The adjoint of sample_field: imprint per-particle values onto a grid (bilinear, periodic) -- e.g. a
        moving body depositing momentum into the fluid velocity grid (cloth->fluid coupling)."""
        from holographic.misc.holographic_fields import scatter_to_field
        return scatter_to_field(shape, positions, values)

    def drag_force(self, positions, velocities, vx, vy, k=1.0):
        """Drag force on particles from a fluid, F = k*(v_fluid - v_particle) (fluid->cloth coupling). See
        holographic_fields.drag_force."""
        from holographic.misc.holographic_fields import drag_force
        return drag_force(positions, velocities, vx, vy, k=k)

    def sample_field_3d(self, field, positions):
        """Read a 3-D grid field at continuous positions (N,3), trilinear+periodic -- how a softbody/particle
        feels a 3-D solved or VSA-encoded field. See holographic_fields.sample_field_3d."""
        from holographic.misc.holographic_fields import sample_field_3d
        return sample_field_3d(field, positions)

    def scatter_to_field_3d(self, shape, positions, values):
        """The exact adjoint of sample_field_3d: imprint per-node values onto a 3-D grid (trilinear, periodic)
        -- the body->fluid half of 3-D two-way coupling. See holographic_fields.scatter_to_field_3d."""
        from holographic.misc.holographic_fields import scatter_to_field_3d
        return scatter_to_field_3d(shape, positions, values)

    def drag_force_3d(self, positions, velocities, vx, vy, vz, k=1.0):
        """Drag on nodes from a 3-D fluid: k*(v_fluid - v_node), sampled trilinearly -- so a softbody couples to
        fluid_step_3d exactly as it does to the 2-D solver (pass as external_force). See
        holographic_fields.drag_force_3d."""
        from holographic.misc.holographic_fields import drag_force_3d
        return drag_force_3d(positions, velocities, vx, vy, vz, k=k)

    def encode_pairs(self, keys, values):
        """Encode parallel arrays of keys and values -- bundle of bind(key_i, value_i) -- in ONE batched FFT
        (bundle_bind), the vectorised form of the role/filler encode loop VSA programs run constantly. Keeps
        the operation inside the holographic space (one array op) instead of a Python loop of per-pair binds.
        (Distinct from encode_record(fields): that takes a {field: value} record and pairs with decode_record;
        this takes two parallel arrays. Renamed from a former encode_record overload that shadowed the record
        encoder -- one name, one faculty.)"""
        from holographic.agents_and_reasoning.holographic_ai import bundle_bind
        return bundle_bind(keys, values)

    def unbind_keys(self, trace, keys):
        """Unbind one trace against many keys in ONE batched FFT (unbind_all) -- the vectorised form of the
        per-key unbind loop decoders/resonators run. Returns (k, dim): row i is the estimate freed by key i."""
        from holographic.agents_and_reasoning.holographic_ai import unbind_all
        return unbind_all(trace, keys)

    def nearest_in(self, query, matrix):
        """Nearest row of a codebook matrix to `query` -- argmax(matrix @ query) (nearest), the reusable
        matmul form of a per-row cosine loop. Exact. Returns (index, score)."""
        from holographic.agents_and_reasoning.holographic_ai import nearest
        return nearest(query, matrix)

    def tensor_bind(self, keys, values, rank=None):
        """A TENSOR-PRODUCT (outer-product) binding memory -- the uncompressed cousin of HRR's circular
        convolution -- optionally truncated to a rank-r 2-site TENSOR TRAIN (MPS). Stoudenmire's seat: HRR's
        bind is a compressed projection of the tensor product a (X) b, and a tensor-network representation
        interpolates between the two. Returns a holographic_tensor.TensorBindMemory with `.recall(key)` and
        `.n_numbers` (its honest storage).

        What the capacity comparison shows (see the integration tests): at a fixed LOAD the tensor-product bind
        recalls far more accurately than HRR -- crosstalk is suppressed by the key inner products (~1/sqrt(D))
        -- because it spends D x the storage; and with ORTHOGONAL keys it recalls EXACTLY up to M = D, where
        circular convolution cannot. An MPS truncation LOSSLESSLY compresses a low-rank binding matrix (the
        tensor-network win). KEPT NEGATIVE: on the capacity-per-STORED-NUMBER frontier the two are equal --
        HRR's compression gives up nothing there -- and a generic (full-rank) binding cannot be MPS-compressed
        without losing recall, so this is a different point on the storage/fidelity tradeoff, not a free
        improvement over the engine's bind."""
        from holographic.sampling_and_signal.holographic_tensor import TensorBindMemory
        return TensorBindMemory(np.asarray(keys, float), np.asarray(values, float), rank=rank)

    def clifford(self):
        """Cl(3,0) geometric algebra as a PARALLEL binding mode (holographic_clifford) -- the geometric-product
        cousin of tensor_bind, for GEOMETRIC structure. Its seat: rotors compose 3D rotations EXACTLY (the
        product of two rotors IS the rotor of the composed rotation, measured error ~1e-15) and
        NON-COMMUTATIVELY (rotation order is preserved), which the engine's COMMUTATIVE convolution bind cannot
        do -- a commutative binding gives one answer for both orders and so carries the whole order-gap as
        error. Returns a CliffordAlgebra with product / rotor / rotate / compose / reverse. KEPT NEGATIVE: 2^d
        dimension growth (Cl(3,0)=8, Cl(10,0)=1024), and it binds VERSORS (rotors), not arbitrary atoms -- a
        parallel tool for the rotation-shaped corner, not a general HRR bind replacement."""
        if getattr(self, "_clifford", None) is None:
            from holographic.mesh_and_geometry.holographic_clifford import CliffordAlgebra
            self._clifford = CliffordAlgebra()
        return self._clifford

    def splat_aniso(self, field, k=12, steps=200, denoise=False, early_stop=False, stats=None):
        """Represent an n-D field (a 2-D image or a 3-D volume) as a superposition of ANISOTROPIC Gaussian
        splats -- the real 3D-Gaussian-Splatting primitive (oriented, elliptical/ellipsoidal Gaussians with a
        full covariance), fit by gradient descent on the reconstruction MSE (holographic_splat.aniso_fit:
        analytical NumPy gradients + a tiny built-in Adam, no autodiff framework). The anisotropic, n-D
        extension of `splat_field`'s isotropic matching pursuit: one aligned splat replaces many circular ones
        wherever structure is oriented or elongated, in 2-D OR 3-D. Returns (splats, rendered) where each splat
        is (center, amplitude, L) with L the inverse-covariance Cholesky factor; denoise=True returns just the
        rendered field (a few smooth Gaussians cannot hold high-frequency noise).

        ADAPTIVE STOP (C3, opt-in early_stop=True; pass stats={} to read stats['steps']): stop the fixed Adam
        schedule once the reconstruction MSE has converged -- measured ~40% fewer steps at a few-percent MSE
        cost. KEPT CAVEAT: this is a SPEED/QUALITY knob, not free -- a continuous fit has only a soft plateau,
        not the resonator's exact certificate, so stopping always costs a little MSE; off by default
        (bit-identical to the fixed-step fit). A min_steps floor inside aniso_fit guards Adam's ~30-step warm-up
        (a naive relative test would mistake the warm-up for convergence and stop with a terrible fit).

        KEPT NEGATIVE / SCOPE: the loss is non-convex, so this finds a LOCAL optimum -- more splats do not help
        monotonically (a clean K=4 fit can beat a messier K=8 one) and the result depends on the isotropic warm
        start; and this is the from-scratch core of 3DGS only -- no tile rasteriser, no spherical-harmonic
        view-dependent colour, no GPU speed."""
        from holographic.rendering.holographic_splat import aniso_fit
        splats, rendered = aniso_fit(np.asarray(field, float), k, steps=steps,
                                     early_stop=early_stop, stats=stats)
        return rendered if denoise else (splats, rendered)

    def splat_densify(self, field, k=12, stage_steps=(50, 80, 210), denoise=False, stats=None):
        """Fit an n-D field with K ANISOTROPIC Gaussian splats COARSE-TO-FINE (C1) -- 3D-Gaussian-Splatting
        densification, from scratch (holographic_splat.densify_fit). Rather than placing all K splats at once and
        running one joint gradient fit (`splat_aniso`), grow the set in stages: place a fraction on the current
        residual (coarse scales first), jointly optimise, then place more where the re-optimised reconstruction
        still errs, and optimise again. `stage_steps` is the Adam steps per stage (the last should be long enough
        to fully converge the whole set). Returns (splats, rendered); denoise=True returns just the rendered
        field; pass stats={} to read stats['stages'].

        WHY USE THIS over splat_aniso (measured): the staged placement is a far better WARM START for the final
        joint fit, landing in a better basin of the non-convex loss. On a multi-scale target (a broad blob + small
        sharp details) it reaches MSE the one-shot CANNOT reach at any step count (~1e-6 vs ~1e-3, where the
        one-shot then DIVERGES past ~300 steps) -- directly addressing splat_aniso's local-optimum kept negative
        (its result 'depends on the isotropic warm start'; a staged warm start is a much better one). The trade is
        more total compute (several optimisation rounds); the win is on MULTI-SCALE content -- on a single-scale
        field the one-shot is already near-optimal."""
        from holographic.rendering.holographic_splat import densify_fit
        splats, rendered = densify_fit(np.asarray(field, float), k, stage_steps=stage_steps, stats=stats)
        return rendered if denoise else (splats, rendered)

    def image_archive(self, shape, capacity, keep=None, dim=32768, thumb=12):
        """The DCT/Walsh-Hadamard plate archive (holographic_archive.HolographicArchive) as a mind faculty --
        the exact, CROSS-MODAL counterpart to the lossy `splat_archive`. Images are superposed into orthogonal
        key plates and any one reconstructs EXACTLY when undamaged (a single adjoint per channel) or gracefully
        under erasure (joint masked recovery -- ~0.002 error even at 40% plate loss). Cross-modal both ways:
        `.add(image, tags=[...], nums={...})` attaches a descriptive address, `.recall_by_tags(words=[...])`
        returns the best-matching image FROM THE DESCRIPTION ALONE (Ozcan's describe-then-retrieve, soft-AND
        over the query tags), and `.tags_of(i, candidates)` runs the reverse -- the description the archive
        would give a stored image. `keep` is the DCT coefficients kept per image (defaults to all shape[0]^2,
        bit-exact; fewer trades exactness for compression). Requires capacity*keep <= dim. Seeded by this mind."""
        from holographic.misc.holographic_archive import HolographicArchive
        if keep is None:
            keep = shape[0] * shape[0]                  # all coefficients -> exact recall by default
        return HolographicArchive(shape, capacity, keep=keep, dim=dim, seed=self.seed, thumb=thumb)

    def federated_archive(self, shape, capacity, K=4, keep=2000, dim=32768, thumb=12):
        """A FEDERATED image archive (holographic_archive.FederatedArchive) -- the storage array's federation
        applied to the content archive, and the archive twin of `storage_array`. One archive's per-vector budget
        is conserved, so to hold more images you spread them across K aligned HolographicArchive shards with a
        directory (image i -> shard i mod K): total capacity is K x per-shard, recovery quality holds at a fixed
        per-image budget (more images = more shards, not one archive blurring), and it extends one shard at a
        time. Returns the coordinator: `.add(image, tags=, nums=)` stores and routes (returning a global index),
        `.recover(i, mask=)` recovers image i from its shard. Same per-shard DCT / Walsh-Hadamard plate algebra
        the single archive uses -- only a routing layer added, never a new image codec. Seeded by this mind."""
        from holographic.misc.holographic_archive import FederatedArchive
        return FederatedArchive(shape, capacity, K=K, keep=keep, dim=dim, seed=self.seed, thumb=thumb)

    def splat_archive(self, shape, keep=40):
        """Open a SPLAT-BUNDLE image archive (holographic_splat_archive) -- a gallery stored as Gaussian-
        splat codes BESIDE the WHT-plate archive (a splat scene is a bundle). add(image) fits K splats per
        channel; recover(i, k) renders them (a k-prefix is a progressively-refined preview, since matching
        pursuit stores them in importance order); recall(query) finds an image by content; region(i, box)
        is an EXACT 'what is here' query. Returns a fresh SplatArchive for `shape`.

        KEPT NEGATIVE: this is LOSSY -- the WHT-plate archive is EXACT undamaged and, on DCT-friendly
        images, beats it on quality at a matched byte budget; the splat archive's win is the ADDED
        region-query + progressive-refinement (and a compact code), not quality parity. It sits beside the
        plates, not in place of them."""
        from holographic.rendering.holographic_splat_archive import SplatArchive
        return SplatArchive(shape, keep=keep)

    def splat_scene(self, field, grid=16, tile=8, levels=5, k=30, dim=4096, seed=0):
        """Build a CONTENT-ADDRESSABLE splat scene from a field -- a coarse occupancy map stored as ONE
        hypervector per tile that you can query by region ('what is at this cell?'). Fits k Gaussian splats,
        then encodes them with TILED bundling so region recall stays accurate at FINE resolution. The plain
        single-bundle scene's region readback is decode-via-cleanup (unbind a cell role, clean up to a level)
        and so caps as the grid gets finer -- measured ~98% at grid 16 down to ~75% at grid 32 at dim 4096 --
        while routing each cell to a tile bundle of at most tile*tile bindings holds recall ~100% at any total
        resolution, the same chunk-to-beat-the-cap trade the route/sequence faculties make (one vector per
        tile). The EXACT complement is splat_archive's region (explicit per-splat); this is the compact,
        content-addressable, coarse-but-robust one. Returns the scene; read a cell back with splat_region."""
        from holographic.rendering.holographic_splat import splat_fit, splat_bundle_tiled
        field = np.asarray(field, float)
        splats = splat_fit(field, k)
        return splat_bundle_tiled(splats, field.shape, dim=dim, grid=grid, levels=levels, tile=tile, seed=seed)

    def splat_region(self, scene, cell):
        """Read a region's occupancy in [0, 1] back from a splat scene built by splat_scene -- the
        content-addressable 'what is at grid cell (gy, gx)?' query, routed to the cell's tile bundle."""
        from holographic.rendering.holographic_splat import recall_region_tiled
        return recall_region_tiled(scene, tuple(cell))

    def splat_prune(self, splats, target, keep):
        """PRUNE a splat set to its `keep` highest-contribution splats (largest |amplitude|, since each splat's
        reconstruction energy is amp^2 for the engine's unit-norm gaussians) and refit the survivors (holographic_
        splatprune). Contribution-ranked prune + refit degrades gracefully and beats naive pruning by a wide margin
        (~20 dB at half the splats on a smooth field). Returns the pruned, refitted splat list. The splat twin of
        mesh decimation."""
        from holographic.rendering.holographic_splatprune import splat_prune
        return splat_prune(splats, target, keep)

    def splat_merge(self, splats, target, radius):
        """MERGE splats closer than `radius` into one (amplitude-weighted centre and scale, summed amplitude) and
        refit (holographic_splatprune) -- reduces the count with bounded quality loss. Returns the merged list. Kept
        negative: merge is lossy by construction (one Gaussian cannot equal two); the radius trades count for
        quality."""
        from holographic.rendering.holographic_splatprune import splat_merge
        return splat_merge(splats, target, radius)

    def splat_lod_chain(self, splats, target, keeps=(40, 20, 10, 5)):
        """Build a splat LEVEL-OF-DETAIL chain (holographic_splatprune): prune to each count in `keeps`, measuring
        reconstruction PSNR at each. Returns a fine->coarse list of (splats, count, psnr); the first is the refitted
        full set. The splat-domain twin of mesh_lod_chain -- the engine's error-budget resolution selection, with the
        budget in PSNR. Pair with splat_select_lod."""
        from holographic.rendering.holographic_splatprune import splat_lod_chain
        return splat_lod_chain(splats, target, keeps=keeps)

    def splat_select_lod(self, chain, min_psnr):
        """Choose a splat LOD by QUALITY budget (holographic_splatprune): the index of the FEWEST-splat level whose
        PSNR still meets `min_psnr` -- the cheapest splat set that looks right -- falling back to the finest level if
        none clears it. The PSNR-budget analog of mesh_select_lod's pixel budget."""
        from holographic.rendering.holographic_splatprune import select_splat_lod
        return select_splat_lod(chain, min_psnr)

    def splat_clone_split(self, splats, target, n_densify=None, scale_thresh=None):
        """SCALE-AWARE CLONE-VS-SPLIT DENSIFICATION (holographic_splatdensify) -- the 3DGS densification DISTINCTION
        that the engine's existing splat_densify (`densify_fit`, staged residual placement) was missing: it adds
        capacity where error is high but is SCALE-BLIND. This refines an EXISTING splat set by asking, per high-error
        splat, whether the region needs COVERING or RESOLVING. Rank the splats by the residual error in their
        footprint and, for the highest-error ones, CLONE if narrow (sigma < scale_thresh -- add a same-scale copy to
        COVER an under-served region) else SPLIT (replace a wide splat with two narrower ones to RESOLVE fine
        structure it was smearing). `n_densify` caps how many splats to densify (default all); `scale_thresh` defaults
        to the set's median sigma. The inverse of splat_prune/merge: those COARSEN, this REFINES -- and it sharpens
        WHERE new capacity goes (measured: scale-aware beats always-clone and always-split on a mixed-error target at
        a fixed budget, and the WRONG move can be worse than nothing -- splitting a small splat loses coverage). Kept
        negative: complements splat_densify's from-scratch placement (the new part is the cover-vs-resolve decision on
        an existing set); isotropic splats."""
        from holographic.rendering.holographic_splatdensify import clone_split_densify
        return clone_split_densify(splats, target, n_densify=n_densify, scale_thresh=scale_thresh)

    def splat_relocate(self, splats, target, dead_frac=0.05):
        """MCMC BIRTH-DEATH RELOCATION (holographic_relocate) -- the successor to evict-rarest. The engine's bounded
        memory DROPS the rarest when a store is full (the creature's `memory_cap` path); 3DGS-as-MCMC instead
        RELOCATES a dead atom to an under-represented region, CONSERVING the budget. This moves the DEAD splats
        (|amplitude| below `dead_frac` of the largest) each to the current residual peak -- the most under-represented
        region -- subtracting after each so successive relocations find distinct peaks, and keeping the splat COUNT
        fixed (a birth-death move, not a drop). The discrete kin of the B10 generative-denoising sampler. Measured:
        relocating to residual peaks beats DROPPING (~4x lower MSE -- eviction shrinks and wastes the budget) and
        beats RANDOM relocation (the principled target matters), at a conserved count. Kept negative: the drop was
        already in the box (the creature's eviction is unchanged); this conserves capacity instead -- isotropic
        splats, and a no-op when nothing is dead."""
        from holographic.misc.holographic_relocate import birth_death_relocate
        return birth_death_relocate(splats, target, dead_frac=dead_frac)

    def render_scene(self, tag_list, S=96, seed=0):
        """Render composed attribute tags to an actual RGB image via the scene renderer."""
        from holographic.scene_and_pipeline.holographic_scene import make_scene
        return make_scene([(t["shape"], t["colour"]) for t in tag_list], S=S, seed=seed)

    def camera(self, eye=(0.0, 0.0, 3.0), target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0),
               fov_deg=50.0, aspect=1.0, near=0.05, far=100.0):
        """A pinhole Camera (eye looks at target, vertical fov) -- view+projection matrices and per-pixel rays.
        The viewpoint for rasterised and volumetric renders. See holographic_render.Camera."""
        from holographic.rendering.holographic_render import Camera
        return Camera(eye=eye, target=target, up=up, fov_deg=fov_deg, aspect=aspect, near=near, far=far)

    def light(self, kind="directional", direction=(-0.4, -0.8, -0.5), position=(2.0, 3.0, 2.0),
              color=(1.0, 1.0, 1.0), intensity=1.0):
        """A Light: 'directional' (sun), 'point', or 'ambient' (fill). See holographic_render.Light."""
        from holographic.rendering.holographic_render import Light
        return Light(kind=kind, direction=direction, position=position, color=color, intensity=intensity)

    def render_mesh(self, mesh, camera, width=512, height=512, lights=None, base_color=(0.8, 0.8, 0.8),
                    background=(0.05, 0.06, 0.08), ambient=0.15, vectorized=True, texture=None, uvs=None,
                    smooth=False, dtype=None, two_sided=False, vertex_colors=None,
                   pbr=None):
        """Rasterise a mesh to an (H,W,3) RGB image with a z-buffer and Lambert shading (frustum + back-face
        culled). `base_color` may be a PBRMaterial's base_color. vectorized=True (default) uses the batched
        fragment-scatter path (the per-triangle Python loop ported to one array op -- ~8-15x faster, image
        identical); vectorized=False is the readable reference loop. CPU renderer -- the authoring brain's
        offline / preview frame; the GPU stays the muscle for a heavy interactive viewport. Pass `texture` (H,W,3)
        + per-vertex `uvs` (or mesh.uvs) to render a TEXTURED mesh -- bilinear per-fragment sampling; what
        showing a UV-transferred or baked texture needs. See holographic_render.

        JSON-DRIVABLE (C2): `mesh` may be a Mesh or {'vertices','faces'}; `camera` may be a Camera, a
        CameraController, or {'eye','target',...} -- so a node pack or POST /invoke can call this with plain JSON
        and no class imports. A downstream audit found this path unreachable from ANY JSON client, including our
        own service. Real objects pass through by IDENTITY, so existing callers are bit-identical."""
        from holographic.io_and_interop.holographic_coerce import as_camera, as_mesh
        from holographic.rendering.holographic_render import rasterize_mesh
        mesh, camera = as_mesh(mesh), as_camera(camera)      # the ONLY permissive edge; the renderer stays strict
        if hasattr(base_color, "base_color"):
            base_color = base_color.base_color
        img = rasterize_mesh(mesh, camera, width=width, height=height, lights=lights,
                             base_color=base_color, background=background, ambient=ambient,
                             vectorized=vectorized, texture=texture, uvs=uvs, smooth=smooth,
                             two_sided=two_sided, vertex_colors=vertex_colors, pbr=pbr)
        # C16: dtype= (default None = float64, byte-identical to before). The rasteriser works in float64 and a
        # downstream client cast every frame to float32 itself; doing it here saves the extra full-image copy on a
        # big render. Cast at the EXIT only -- casting earlier would change the shading maths, and this method must
        # not move a single pixel it used to produce.
        return img if dtype is None else img.astype(dtype, copy=False)

    def ray_path_index(self, objects, camera, width=256, height=256, sun="bright", sky="clear"):
        """Build a BIDIRECTIONAL ray<->object index for a scene: which objects each camera ray TOUCHED along its path
        (primary hit + objects seen THROUGH glass). `index.pixels_touching(ids)` then returns the exact pixels to
        re-shade when those objects change -- INCLUDING indirect pixels (an object seen through glass) that a
        primary-id-only incremental renderer misses. Pair with `delta_reshade_scene` for a bounded, bit-exact update.
        The trace already knows where every ray went; this keeps it instead of re-gathering it every frame.
        See holographic_rayindex."""
        from holographic.simulation_and_physics.holographic_semantic import _scene_setup
        from holographic.rendering.holographic_rayindex import build_ray_index
        ctx = _scene_setup(objects, True, sky, sun, (0.75, 0.9, 0.85))
        if ctx is None:
            return None
        return build_ray_index(ctx, camera, width, height)

    def delta_reshade_scene(self, edited_objects, index, changed_ids, base_frame, camera, sun="bright", sky="clear"):
        """Apply a material/colour/light EDIT as a bounded delta: re-shade ONLY the pixels whose ray touched a changed
        object (from `ray_path_index`), composite into `base_frame`. Deterministic -> the updated pixels are bit-exact
        vs a full re-render, at a fraction of the work, and the through-glass pixels update correctly. Geometry must be
        unchanged (a colour/material/light edit); a MOVE needs the index rebuilt for the affected region first."""
        from holographic.simulation_and_physics.holographic_semantic import _scene_setup
        from holographic.rendering.holographic_rayindex import delta_reshade
        ctx_new = _scene_setup(edited_objects, True, sky, sun, (0.75, 0.9, 0.85))
        return delta_reshade(ctx_new, index, changed_ids, base_frame, camera)

    def brick_ray_index(self, objects, camera, width=256, height=256, grid=10, samples=12, sun="bright", sky="clear"):
        """Build a REGION-keyed ray index: which spatial bricks each ray traversed (eye->hit). Where ray_path_index
        keys edits by object, this keys them by REGION, so a MOVE (geometry change) gets the same bounded delta -- the
        affected pixels are the rays that passed through the bricks the object vacated or now occupies. Pair with
        delta_reshade_move. See holographic_rayindex.BrickRayIndex."""
        from holographic.simulation_and_physics.holographic_semantic import _scene_setup
        from holographic.rendering.holographic_rayindex import build_brick_index
        ctx = _scene_setup(objects, True, sky, sun, (0.75, 0.9, 0.85))
        if ctx is None:
            return None
        self._last_brick_ctx = ctx
        return build_brick_index(ctx, camera, width, height, grid=grid, samples=samples)

    def delta_reshade_move(self, obj_id, delta, brick_index, base_frame, camera):
        """Apply a MOVE of object `obj_id` by `delta` as a bounded, bit-exact delta: re-shade only the rays that
        traversed the bricks the object vacated (old position) or now occupies (new position), from `brick_ray_index`
        (which cached the scene ctx). Returns (updated_frame, mask, ctx_new). See holographic_rayindex."""
        from holographic.rendering.holographic_rayindex import delta_reshade_move as _drm
        ctx = getattr(self, "_last_brick_ctx", None)
        if ctx is None:
            raise ValueError("call brick_ray_index(...) first to build the index and cache the scene")
        return _drm(ctx, obj_id, delta, brick_index, base_frame, camera)

    def incremental_renderer(self, camera, width=256, height=256, sun="bright", sky="clear", ground=True, ss=1):
        """A render SESSION: render the first frame, then re-render the SAME scene for FREE (cached), apply colour/
        material/light edits and geometry moves as bounded bit-exact DELTAS, and stream only the changed pixels. This
        is the path for repeated rendering / live editing / pixel streaming -- calling render_scene every frame re-does
        the whole trace even when nothing changed; the session pays only for what changed. Default ss=1 (delta-exact,
        for streaming); use a one-shot render_scene(ss=2) for a final still. See holographic_rayindex.IncrementalRenderer.

        Usage:
            r = mind.incremental_renderer(camera, 256, 256)
            frame, mask = r.render(objects)        # first frame (full); mask = whole frame
            frame, mask = r.render(objects)        # SAME scene -> free, mask empty
            frame, mask = r.edit(0, 'color', 'gold')   # delta; mask = changed pixels
            frame, mask = r.move(1, (0.3, 0, 0))       # delta; mask = changed pixels
            ys, xs, rgb = r.stream_delta(mask)     # the wire payload: only changed pixels
        """
        from holographic.rendering.holographic_rayindex import IncrementalRenderer
        return IncrementalRenderer(camera, width, height, sun=sun, sky=sky, ground=ground, ss=ss)

    def region_field(self, regions):
        """Compose a LABELLED REGION FIELD: a set of boundaries (SDFs), each tagging the points inside it with how to
        REGARD them -- a material, or a behaviour (cloth / fire / smoke / fluid), or a biome -- resolved by priority.
        One `classify(points)` call then drives material lookup, behaviour (which SIM) lookup, and precise culling
        (points outside every region are known-empty, skipped with no marching). `slice(origin, u, v)` cuts the volume
        open to reveal the material LAYERS. This is the composable substrate for treating anything as mesh / particle /
        smoke / fluid / light over one field. `regions` is a list of holographic_regionfield.Region. See
        holographic_regionfield.

        Example:
            from holographic.misc.holographic_regionfield import Region
            from holographic.simulation_and_physics.holographic_semantic import _SphereSDF
            rf = mind.region_field([
                Region(_SphereSDF((0,0,0), 1.0), 'crust',  priority=1, material=(0.4,0.3,0.2)),
                Region(_SphereSDF((0,0,0), 0.4), 'core',   priority=2, material=(1.0,0.85,0.3)),
            ])
            img, labels = rf.slice((0,0,0), (1,0,0), (0,1,0))   # cut it open -> see the layers
            keep = rf.cull(points)                              # precise culling for free
        """
        from holographic.misc.holographic_regionfield import RegionField
        return RegionField(list(regions))

    def reflect_transform(self, O, D, P_hit, N, bounce=None):
        """A secondary (bounce) ray as a TRANSFORM of its parent: origin -> hit point, direction -> reflected about the
        normal, bounce counter -> +1. N bounces are N applications of this one transform. See holographic_raycoherence."""
        from holographic.rendering.holographic_raycoherence import reflect_transform as _rt
        return _rt(O, D, P_hit, N, bounce=bounce)

    def coherent_reflection(self, ctx, P, N, D, ids, mirror, width, height, stride=4, var_tol=0.03):
        """Reconstruct the reflection over reflective pixels from a SPARSE trace + gated bilinear interpolation of the
        perpendicular neighbours, with an exact-trace fallback on reflection edges -- because neighbouring reflection
        rays off a smooth surface are coherent, this traces far fewer rays than per-pixel for a close result. Returns
        (reflected (H*W,3), n_traced, n_mirror). Honest limit: sharp reflected-CONTENT edges blur (the coherence is in
        the reflector geometry, not the reflected image). See holographic_raycoherence."""
        from holographic.rendering.holographic_raycoherence import coherent_reflection as _cr
        return _cr(ctx, P, N, D, ids, mirror, width, height, stride=stride, var_tol=var_tol)

    def ray_pencil(self, O, D, C, R, eps=0.03):
        """Emit a ray's PERPENDICULAR FRAME (centre + 4 marginal rays offset +-eps) and transport it through a
        reflection off a sphere: the reflected pencil converges (concave far wall -> caustic) or diverges (convex cap).
        Returns (P (5,3), D2 (5,3), hit). The pencil's cross-section IS the Gaussian of secondary rays. See
        holographic_raydiff."""
        from holographic.rendering.holographic_raydiff import transport_pencil
        return transport_pencil(O, D, C, R, eps)

    def caustic_focus(self, O, D, C, R, eps=0.03, s_max=6.0):
        """Where a reflected pencil is tightest -- the focus / caustic point -- and the pencil radius there. A 5-ray
        frame predicts the focus a dense bundle would show. Returns (s_focus, radius). See holographic_raydiff."""
        from holographic.rendering.holographic_raydiff import transport_pencil, find_focus
        P, D2, hit = transport_pencil(O, D, C, R, eps)
        return find_focus(P, D2, s_max=s_max)

    def lobe_sigma(self, O, D, C, R, s, eps=0.03, roughness=0.0, light_half_angle=0.0):
        """The Gaussian lobe half-width of the whole secondary bundle at distance s: geometric pencil spread combined
        with surface roughness (micro-imperfections -> glossy) and a soft light's angular size (penumbra). One number
        standing in for the entire bundle of secondary rays. See holographic_raydiff."""
        from holographic.rendering.holographic_raydiff import transport_pencil, lobe_sigma as _ls
        P, D2, hit = transport_pencil(O, D, C, R, eps)
        return _ls(P, D2, s, roughness=roughness, light_half_angle=light_half_angle)

    def dispersion_spread(self, D, N, iors):
        """The chromatic angular fan from refracting one ray at several wavelength IORs (eta = n_in/n_out per colour) --
        the same pencil split by wavelength, which IS dispersion. See holographic_raydiff."""
        from holographic.rendering.holographic_raydiff import dispersion_spread as _ds
        return _ds(D, N, iors)

    def grid_graph(self, shape, blocked=None):
        """Adjacency dict {cell: [neighbours]} for an N-D grid (a 2D/3D/.../ND maze is the same object). Feed to any
        graph solver. See holographic_ndfield."""
        from holographic.misc.holographic_ndfield import grid_graph
        return grid_graph(shape, blocked)

    def solve_grid_maze(self, shape, blocked, start, goal, steps=200, mu=1.5, dt=0.2):
        """Solve an N-D grid maze with the Tero slime-mould flow solver -- the SAME solver the 2D maze used, unchanged,
        because it operates on the graph not the coordinates. 3D (or 7D) is trivial. See holographic_ndfield."""
        from holographic.misc.holographic_ndfield import solve_grid_maze
        return solve_grid_maze(shape, blocked, start, goal, steps=steps, mu=mu, dt=dt)

    def sparse_reconstruct(self, oracle, lo, hi, n_seed=96, n_refine=96, bandwidth=None, seed=0):
        """The reusable sparse-probe-interpolate-refine pattern: reconstruct a known deterministic field (oracle) over
        an N-D box from a sparse ADAPTIVE sample (Nadaraya-Watson kernel + refine where uncertain). The pattern under
        coherent reflection, ray differentials, and the radiance field, named once. Returns (points, values,
        reconstruct_fn). See holographic_ndfield."""
        from holographic.misc.holographic_ndfield import sparse_reconstruct
        return sparse_reconstruct(oracle, lo, hi, n_seed=n_seed, n_refine=n_refine, bandwidth=bandwidth, seed=seed)


    def scene_light(self, kind="sun", target=None, width=1.0, height=1.0, up=(0.0, 1.0, 0.0), **kw):
        """Build a PATH-TRACER light by name -- the one door to all ten types, for render_scene_document.

        kind is a word an agent would actually type: 'sun'/'directional', 'point'/'lamp', 'spot',
        'rect'/'area'/'softbox'/'panel', 'disk', 'sphere', 'dome'/'environment'/'sky'/'hdri'/'ibl',
        'ambient'/'fill', 'mesh'/'emissive', 'ies'. An unknown kind raises with the full list rather than a
        bare KeyError, because guessing is what a caller does when it has never seen the API.

        `target` is the part that saves real work: give it a point to light and the panel/disk/spot/sun is
        ORIENTED for you. Without it, aiming a softbox means hand-building an orthonormal basis out of
        u_vec/v_vec half-edges -- measured as the place 3-D authoring stalls. Everything else is passed
        straight to the class (position, color, intensity, radius, inner_deg/outer_deg, ground_color, ...).

        'dome' is the one to reach for first: an environment/IBL light is shadowed, so soft contact shadows
        (ambient occlusion) fall out for free, and it is most of what makes a render read as a photograph.
        Pair a dome with a DARK sky -- a bright sky AND a dome counts the environment twice for diffuse.

        WHY THIS FACULTY EXISTS: nine of these ten classes shipped reachable by nothing. See
        holographic_lights.make_light."""
        import holographic.rendering.holographic_lights as _lg
        return _lg.make_light(kind=kind, target=target, width=width, height=height, up=up, **kw)

    def aim_light_basis(self, position, target, width=1.0, height=1.0, up=(0.0, 1.0, 0.0)):
        """Half-edge vectors (u_vec, v_vec) for a rectangular panel at `position` FACING `target`.

        The low-level half of scene_light('softbox', target=...), exposed on its own for callers building a
        RectLight directly or reusing the basis for something else (a gobo frame, a camera-facing card).
        Ordering is chosen so the emitting face points at the target: reversed, you get a valid light that
        renders the scene black. See holographic_lights.aim_basis."""
        import holographic.rendering.holographic_lights as _lg
        return _lg.aim_basis(position, target, width=width, height=height, up=up)


def _selftest():
    """Delegates to holographic.unified.check_part -- one home for the shared contract."""
    n = check_part("holographic.unified.holographic_unified_p08_bake", "_UnifiedPart08")
    print("holographic_unified_p08_bake selftest OK -- %d members reached UnifiedMind, none shadowed" % n)


if __name__ == "__main__":
    _selftest()
